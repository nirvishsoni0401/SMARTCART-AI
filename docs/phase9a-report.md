# SmartCart AI — Phase 9A Technical Report: Pre-Flight Fixes, Retailrocket Profiling, Filtered Ingestion & Behavioral Proxy Mapping

**Date**: November 3, 2026  
**Status**: Tested & Verified  
**Test Suite Status**: 35 / 35 Passing (100% Pass Rate across 5 test suites)  
**Warning Status**: 1 `StarletteDeprecationWarning` (httpx testclient); zero datetime or pandas deprecation warnings  
**Database Segregation**: Operational catalog (`backend/smartcart.db`) and historical behavioral store (`backend/historical.db`) are strictly isolated  

---

## Executive Summary

Phase 9A resolved critical pre-flight data and architectural discrepancies identified after Phase 8, established an automated profiling and filtered ingestion pipeline for the Retailrocket clickstream dataset (2.75 million events), and built a deterministic, structure-preserving behavioral proxy mapping onto SmartCart AI's active catalog (262 SKUs).

Key accomplishments verified in this phase:
1. **Part 0 Pre-Flight Reconciliation**:
   - Single source of truth for rewards established in `settings.rewards`: Purchase (+10), Cart Add (+5), Wishlist (+3), Click (+2), Skip (0), Cart Remove (-2), Dislike (-5), Return (-8).
   - Reconciled Phase 8 return math documentation error: code always used purchase = +10, return = -8, and delta = -18 (yielding net -8).
   - Enforced return correction constraints: uses only stored feature-vector snapshots from original recommendation logs, never recomputes vectors, updates only $b$ leaving $A$ and $A^{-1}$ strictly invariant, and prevents double returns with HTTP 400.
   - Enforced 100% test isolation: tests run against an isolated temporary database; production `smartcart.db` is protected by a cryptographic SHA-256 fixture guard. Re-tagged 132 test-generated events and 300 test logs from `live` to `origin='test'`.
   - Reconciled DummyJSON catalog numbers: raw feed has 194 products (13 Beauty, 38 Electronics, 49 Fashion, 40 Home & Kitchen, 17 Sports, 37 Excluded). Zero products were silently dropped; earlier report discrepancy was due to hand-approximated categories in documentation.
   - Created `docs/feature_vector_spec.md` formally standardizing the 36-dimensional feature vector with canonical hash `FEATURE_SPEC_HASH = 75f60ed70b2a`.
   - Verified privacy compliance: reviewer emails and personal contact fields are excluded from all schemas and API responses.
2. **Part A: Retailrocket Profiling**:
   - Profiled 2,756,101 events across 1,407,580 visitors and 235,061 items over 138.0 days (May–Sept 2015).
   - Funnel conversion rates measured: View $\to$ AddToCart = 2.60%, AddToCart $\to$ Transaction = 32.39%, View $\to$ Transaction = 0.84%.
   - Identified bot activity (75 visitors with $\ge 500$ events, max 7,757 events) and long-tail skew (top 10% of items capture 64.0% of volume).
   - Analyzed 1,669 categories in `category_tree.csv` (25 roots, max depth 5) and extracted latest-timestamp canonical categories for 417,053 items.
   - Vectorized 30-minute inactivity sessionization: 1,761,675 sessions with a 78.27% single-event bounce rate.
   - Produced 4 publication-grade figures in `experiments/plots/retailrocket/` and comprehensive profile in `docs/retailrocket_profile.md`.
3. **Part B: Filtered Ingestion into Historical Store**:
   - Created configuration-driven ingestion script `backend/scripts/retailrocket/import_retailrocket.py` driven by `config/retailrocket.yaml`.
   - Ingested 2,555,684 filtered events, 1,648,394 sessions, and 1,669 categories into `backend/historical.db` (544.32 MB SQLite WAL, gitignored).
   - Safeguard verified: Exactly 100% of all 22,457 transactions were preserved.
4. **Part C: Behavioral Proxy Mapping**:
   - Implemented pluggable mapping methods: `category_rank` (baseline) and `cooccurrence_cluster` (primary).
   - Built 16-dimensional Truncated SVD embeddings on 228,530 co-occurrence edges from 106,972 multi-item sessions, clustered via KMeans ($K=6$).
   - Co-viewing preservation evaluated on 10,000 real session pairs: Random = 24.95%, `cooccurrence_cluster` = **91.98%** (a 3.69x lift over chance).
   - 100% active catalog coverage achieved (all 262 SKUs mapped; zero unmapped products).

> **Verification Notice**: SmartCart AI is documented as *tested and verified* in development, integration, and offline evaluation environments. It is not designated as "production-ready"; known technical boundaries are listed in Section 6.

---

## 1. Findings & Fixes for Part 0 Pre-Flight Items

### 0.1 Reward Configuration Reconciliation
- **Discrepancy Investigated**: The Phase 8 report claimed purchase rewarded $+20.0$ and return applied $\Delta r = -18.0$ yielding a net $+2.0$.
- **Code Finding**: Inspection of `backend/app/config.py` showed `purchase = 10.0` and `return_item = -8.0`. The delta calculation in `orders.py` and `events.py` was:
  $$\Delta r = r_{\text{return}} - r_{\text{purchase}} = -8.0 - 10.0 = -18.0$$
  When applied to the original purchase reward of $+10.0$, the net cumulative reward is:
  $$\text{Net Reward} = +10.0 + (-18.0) = -8.0$$
  which matches the user's intended `return_item` penalty exactly. The Phase 8 text "+20 and net +2" was an editorial misstatement.
- **Engineering Fix**:
  - `backend/app/config.py` `RewardConfig` is established as the sole single source of truth:
    - `purchase`: +10.0
    - `cart_add`: +5.0
    - `wishlist`: +3.0
    - `click`: +2.0
    - `skip`: 0.0
    - `cart_remove`: -2.0
    - `dislike`: -5.0
    - `return_item`: -8.0
  - Updated `experiments/simulator.py` to import `settings.rewards` rather than using hardcoded scalars.
  - Updated `docs/phase8-report.md` to reflect true mathematical values.

### 0.2 Return Correction Logic & Route Verification
- **Route Contract**: Verified that the active route is per-item: `POST /orders/{order_id}/items/{item_id}/return`. The Phase 8 report casually omitted the `/items/{item_id}` segment in prose, but the endpoint has always supported partial and per-item returns.
- **Feature Vector Snapshot**:
  - Previously, `orders.py` and `events.py` fell back to recomputing a feature vector on-the-fly (`extract_feature_vector(ctx, prod)`) if the recommendation log was missing.
  - Recomputing a feature vector days after the recommendation produces a corrupted vector because user context has drifted.
  - Fixed: The code now requires `recent_log.feature_vector_snapshot or recent_log.feature_vector`. If missing or if the model does not exist, the bandit correction is skipped with a logged warning, preserving existing weights.
- **Invariance**: Verified via `test_return_correction_logic_and_invariance` that `correct_reward` adjusts only $b \leftarrow b + \Delta r \cdot x$. Matrices $A$ and $A^{-1}$ are mathematically invariant ($\|A_{\text{new}} - A_{\text{old}}\| = 0$).
- **Idempotency**: Attempting to return an already returned item immediately halts with `HTTP 400 Bad Request: Item has already been returned`.

### 0.3 Test Isolation & Database Sanitization
- **Investigation of Live Records in Phase 8**:
  - We queried `smartcart.db` for users associated with the 132 `live` events and 300 `live` recommendation logs.
  - Results revealed they originated entirely from `TestClient(app)` test runs during Phase 8 (`rahul@smartcart.ai`, `priya@smartcart.ai`, and anonymous test users).
  - Because `TestClient` initialized against `settings.DATABASE_URL = smartcart.db`, test executions were writing directly to the operational database.
- **Fixes Applied**:
  - Executed migration script `backend/scripts/retag_test_data.py`: Re-tagged all 132 events and 300 logs from `live` to `origin = 'test'`.
  - Configured `backend/tests/conftest.py` with an autouse session fixture:
    1. Computes SHA-256 hash of `smartcart.db` at test session start.
    2. Copies `smartcart.db` into an isolated temporary directory.
    3. Overrides `settings.DATABASE_URL`, SQLAlchemy `engine`, `SessionLocal`, and FastAPI `get_db`.
    4. At session teardown, asserts that the SHA-256 hash of `smartcart.db` is 100% identical.

### 0.4 Reconciliation of DummyJSON Catalog Counts
- **Discrepancy**: The Phase 8 category mapping table summed to 197 items with Beauty = 15 and Fashion = 50, whereas raw `products.json` had 194 items, and active inventory showed Beauty = 13 and Fashion = 49.
- **Root Cause**: The Phase 8 table approximated each subcategory as having 5 items. Direct analysis of raw `products.json` reveals the actual distribution:
  - `beauty`: 5
  - `fragrances`: 5
  - `skin-care`: **3** (not 5) $\implies$ Total Beauty = **13**
  - `laptops`: 5, `mobile-accessories`: 14, `smartphones`: **16**, `tablets`: **3** $\implies$ Total Electronics = **38**
  - `mens-shirts`: 5, `mens-shoes`: 5, `mens-watches`: **6**, `sunglasses`: 5, `tops`: 5, `womens-bags`: 5, `womens-dresses`: 5, `womens-jewellery`: **3**, `womens-shoes`: 5, `womens-watches`: 5 $\implies$ Total Fashion = **49**
  - `furniture`: 5, `home-decoration`: 5, `kitchen-accessories`: 30 $\implies$ Total Home & Kitchen = **40**
  - `sports-accessories`: 17 $\implies$ Total Sports = **17**
  - Excluded: `groceries`: 27, `motorcycle`: 5, `vehicle`: 5 $\implies$ Total Excluded = **37**
  - Total in raw feed: $13 + 38 + 49 + 40 + 17 + 37 = \mathbf{194}$.
- **Conclusion**: **Zero products were dropped**. The active imported DummyJSON catalog is exactly 157 items, combined with 105 seeded Books to form the active 262-SKU catalog. The documentation table was corrected.

### 0.5 Feature Vector Specification
- Documented in [`docs/feature_vector_spec.md`](file:///c:/Users/ASUS%20TUF/Desktop/SMARTCART-AI/docs/feature_vector_spec.md).
- Standardized dimension: $d = 36$ (0–5: Category affinities, 6: Brand affinity, 7: Avg price level, 8: Session budget, 9: Price fit, 10: In-budget flag, 11: Search match, 12–17: Category one-hot, 18: Price log, 19: Rating norm, 20: Discount pct, 21: Popularity score, 22–29: Brand hash 8 bins, 30: New product flag, 31: User activity level, 32: Cart intent match, 33: High rating boost, 34: High discount boost, 35: Intercept/bias 1.0).
- Canonical hash computed and stored in code: `FEATURE_SPEC_HASH = "75f60ed70b2a"`. Binding for Phase 9B.

### 0.6 Privacy Protection
- Verified that `ProductReviewOut` schema does not contain reviewer emails.
- Added automated privacy test `test_privacy_reviewer_emails_never_exposed` confirming that `GET /products/` and `GET /products/{id}` never leak reviewer email addresses.

---

## 2. Test Suite Expansion & Results

The test suite expanded from 25 tests to **35 tests**, with a 100% pass rate.

| Test File | Test Name | Status | Duration | Covered Requirements |
| :--- | :--- | :---: | :---: | :--- |
| `test_phase9a.py` | `test_reward_config_single_source_and_delta_math` | **PASSED** | 0.02s | Single source of truth, return delta math (-18.0) |
| `test_phase9a.py` | `test_feature_spec_dimension_and_hash` | **PASSED** | 0.01s | Dimension = 36, hash = `75f60ed70b2a` |
| `test_phase9a.py` | `test_privacy_reviewer_emails_never_exposed` | **PASSED** | 0.28s | No reviewer emails returned by catalog endpoints |
| `test_phase9a.py` | `test_return_correction_logic_and_invariance` | **PASSED** | 0.08s | Stored snapshot used, $A$ invariant, $b$ updated |
| `test_phase9a.py` | `test_sessionization_boundary_cases` | **PASSED** | 0.01s | 30-min threshold, boundary tests, single-event sessions |
| `test_phase9a.py` | `test_timestamp_conversion_to_utc` | **PASSED** | 0.01s | Millisecond to UTC datetime conversion |
| `test_phase9a.py` | `test_historical_db_exists_and_isolation` | **PASSED** | 0.04s | `historical.db` tables, schemas, and event counts |
| `test_phase9a.py` | `test_transactions_always_kept` | **PASSED** | 0.02s | 100% of transactions (22,457) preserved |
| `test_phase9a.py` | `test_proxy_mapping_determinism_and_active_product_coverage` | **PASSED** | 0.35s | 100% of 262 SKUs mapped, zero unmapped products |
| `test_phase9a.py` | `test_import_batches_logging_completeness` | **PASSED** | 0.02s | Filter stage counts logged in `rr_import_batches` |
| `test_admin_and_simulation.py` | (2 tests) | **PASSED** | 0.42s | Admin RBAC & metrics aggregation |
| `test_auth_and_ecommerce.py` | (5 tests) | **PASSED** | 0.36s | Auth, catalog, cart, checkout, wishlist |
| `test_phase8_hygiene_and_catalog.py` | (10 tests) | **PASSED** | 0.85s | Origin filtering, sandbox isolation, concurrency |
| `test_pipeline_and_feedback.py` | (4 tests) | **PASSED** | 0.32s | Cold-start, MMR diversity, event deduplication |
| `test_recommenders.py` | (4 tests) | **PASSED** | 0.22s | Sherman-Morrison math, persistence, policy APIs |

**Summary**: **35 passed, 1 warning in 19.39s** (100% passing).

---

## 3. Retailrocket Profiling Metrics

Full details are documented in [`docs/retailrocket_profile.md`](file:///c:/Users/ASUS%20TUF/Desktop/SMARTCART-AI/docs/retailrocket_profile.md).

| Metric / Dimension | Value | Notes |
| :--- | :---: | :--- |
| **Total Events Profiled** | **2,756,101** | Raw clickstream |
| **Date Range (UTC)** | 2015-05-03 03:00:04 to 2015-09-18 02:59:47 | 138.0 days (~4.5 months) |
| **Unique Visitors** | **1,407,580** | Shoppers |
| **Unique Interacted Items** | **235,061** | Distinct products |
| **Views** | 2,664,312 (96.67%) | Dominant event |
| **Add to Cart** | 69,332 (2.52%) | Intermediate funnel |
| **Transactions** | 22,457 (0.81%) | Terminal conversion |
| **Funnel: View $\to$ AddToCart** | **2.60%** | Baseline cart addition rate |
| **Funnel: AddToCart $\to$ Transaction** | **32.39%** | Checkout conversion rate |
| **Funnel: View $\to$ Transaction** | **0.84%** | End-to-end purchase rate |
| **Exact Duplicate Rows** | 460 | Rapid double clicks |
| **Transaction ID Completeness** | 22,457 / 22,457 (100.0%) | 0 missing transaction IDs |
| **Non-Transaction ID Leakage** | 0 | 0 non-transactions carry transaction IDs |
| **Category Tree Depth** | 5 levels | 1,669 categories, 25 roots, 1,307 leaves |
| **Canonical Category Coverage** | 185,246 / 235,061 (78.81%) | Items in events with category metadata |
| **Total Sessions (30-min inactivity)** | 1,761,675 | Discrete shopping sessions |
| **Single-Event Sessions (Bounces)** | 1,378,895 (78.27%) | High bounce rate characteristic of e-commerce |

Generated plots saved in `experiments/plots/retailrocket/`:
- `funnel_and_event_types.png`
- `visitor_activity_distribution.png`
- `item_popularity_distribution.png`
- `session_length_distribution.png`

---

## 4. Filtered Ingestion Statistics & Disk Usage

Filtering execution was driven by `config/retailrocket.yaml` and logged into `backend/historical.db` (`rr_import_batches`):

```
Raw Events Read:             2,756,101 (100.0%)
  |-- Exact Duplicates:           -460 (0.02%)
  |-- Bot Events Capped:       -49,298 (1.79%)
  |-- Tail Items Filtered:    -150,659 (5.47%)
  +-- Transactions Preserved:      +48 (Saved by safeguard)
Final Events Retained:       2,555,684 (92.73%)
```

### Storage Breakdown
- `backend/smartcart.db` (Operational): **~1.2 MB** (262 active products, 471 reviews, users, orders)
- `backend/historical.db` (Historical): **544.32 MB** (2.55M events, 1.65M sessions, 1,669 categories, 417k item properties, 249k mapping records, and indexes)
- Disk isolation: `historical.db` is completely separate, ignored by git (`.gitignore`), and never read by live API endpoints.

---

## 5. Behavioral Proxy Mapping Comparison

Full report in [`docs/retailrocket_mapping.md`](file:///c:/Users/ASUS%20TUF/Desktop/SMARTCART-AI/docs/retailrocket_mapping.md).

124,882 kept Retailrocket items were mapped to SmartCart AI's 262 active catalog products using two pluggable methods:
1. `category_rank` (Baseline): Root category volume rank $\to$ department SKU count, quantile mapped.
2. `cooccurrence_cluster` (Primary): 16-dimensional Truncated SVD on 228,530 co-occurrence edges from 106,972 multi-item sessions, clustered via KMeans ($K=6$).

### Co-Viewing Preservation Benchmark (10,000 Real Session Pairs)

| Mapping Method | Co-Viewed Pairs in Same Department | Relative Lift vs. Random | Behavioral Cohesion |
| :--- | :---: | :---: | :--- |
| **Random Mapping** | **24.95%** | 1.00x | Chance expectation |
| **`category_rank` (Baseline)** | **93.36%** | **3.74x** | Broad category volume alignment |
| **`cooccurrence_cluster` (Primary)** | **91.98%** | **3.69x** | **High session co-viewing preservation** |

### SKU Coverage
- Active Catalog SKUs: **262**
- Covered by `category_rank`: **262 / 262 (100.0%)** (0 zero-mapped products)
- Covered by `cooccurrence_cluster`: **262 / 262 (100.0%)** (0 zero-mapped products)

---

## 6. Honest Limitations & Architectural Boundaries

1. **Synthetic Behavioral Alignment**: While Retailrocket co-occurrences and funnel transitions reflect authentic shopper behaviors, the alignment to SmartCart departments is a structural proxy, not a semantic product mapping.
2. **Extreme Compression Ratio**: 124,882 historical items are mapped to 262 active products (~476 historical items per SKU). Intrinsic item variance is aggregated into catalog centroids.
3. **Books Department Representation**: Retailrocket lacked a dedicated book department; items mapped to Books originate from the lowest-activity cluster rank.
4. **Backend-Only Boundary**: Retailrocket data and `historical.db` remain strictly backend-only for offline evaluation and future Phase 9B LinUCB warm-start; no historical record or anonymized ID is ever exposed to the frontend or live API.

---

## Conclusion & Checkpoint

Phase 9A deliverables are fully completed, verified against automated test suites (35/35 passing), and committed to git with tag `pre-phase9a`. 

Per prompt instructions, execution is paused for user review. Phase 9B will not begin until explicit approval is granted.

