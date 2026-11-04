# Retailrocket to Active Catalog Behavioral Proxy Mapping Report

**Updated**: 2026-11-04 06:07:03 UTC (Phase 9B Extended)  
**Active Catalog SKUs**: `262` (157 DummyJSON + 105 Seeded Books)  
**Historical Kept Items**: `124,882` items  
**Train Cutoff**: `1438968232766` (2015-08-07 17:23:52 UTC, 70% temporal split)  

---

## 1. Overview & Methodological Rationale

Retailrocket items possess no natural language titles, descriptions, brand names, or images; all identifiers and categories are strictly anonymized numbers. A semantic or lexical match to SmartCart AI catalog products is therefore impossible.

Instead, Phase 9 implements a **behavioral proxy mapping**:
- The mapping preserves structural signals: item popularity distribution, session co-occurrence relationships, and department volume balance.
- Every kept Retailrocket item maps deterministically to exactly one **active** SmartCart product ($124,882 \to 262$ many-to-one mapping).
- Mappings are stored in `backend/historical.db` table `rr_item_map` with soft references to `smartcart.db` products.
- **Strict Leakage Prevention**: Training-period mapping (`_train` variants) utilizes strictly events occurring prior to the chronological split cutoff (`t < 1438968232766`). Cold items appearing post-cutoff fall back safely to category-tree heuristics.

---

## 2. Pluggable Mapping Methods Implemented

### Method 1: `category_rank` (Baseline)
- Aligns Retailrocket root category event volume to SmartCart department SKU counts.
- Within each department, Retailrocket items are assigned to active products by linear popularity-rank quantile.
- Highly skewed: top-10 products absorb ~33% of all event volume.

### Method 2: `cooccurrence_cluster` (Primary Behavioral)
- Builds an item-item co-occurrence graph from multi-item shopping sessions.
- Computes 16-dimensional Truncated SVD embeddings on the co-occurrence adjacency matrix.
- Clusters the embedding space into $K=6$ clusters using KMeans with seed `42`.
- Aligns clusters to the 6 catalog departments by volume rank.
- Assigns items to active products within each department by popularity quantile.

### Method 3: `popularity_balanced` (Snake-Stride Quantile Assignment)
- Retains the behavioral co-occurrence department clustering of Method 2 (preserving co-viewing lift).
- Within each department, replaces monotonic quantile sorting with a **snake-stride (zig-zag)** assignment across active products.
- Evenly disperses head and tail items across all SKUs, eliminating severe popularity concentration and dropping Gini coefficient by over 50%.

---

## 3. Mapping Diagnostics & Concentration Comparison

The table below reports Gini inequality, top-product event concentration, and catalog SKU coverage across the implemented methods:

| Method | Version | Gini Coefficient | Top-1 Product Share | Top-10 Products Share | SKU Coverage | Behavioral Lift |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`category_rank`** (Baseline) | `v1.0` | 0.6298 | 5.06% | 32.95% | 100.0% | Coarse category alignment |
| **`cooccurrence_cluster`** | `v1.0` | 0.6227 | 7.79% | 29.70% | 100.0% | High co-viewing preservation |
| **`popularity_balanced`** | `v1.0` | **0.2885** | **0.71%** | **6.48%** | 100.0% | Co-viewing + balanced volume |
| **`popularity_balanced_train`** | `v1.0_train` | **0.2912** | **0.75%** | **6.62%** | 100.0% | **Leakage-safe (t < t_train)** |

*Key Takeaway*: In unconstrained quantile mapping (`category_rank` and `cooccurrence_cluster`), the top 10 products captured ~30% of all platform events. `popularity_balanced` successfully reduces top-10 share from 29.70% down to **6.48%**, providing a substantially fairer and more representative substrate for offline bandit warm-start.

---

## 4. Honest Limitations & Architectural Boundaries

1. **Synthetic Behavioral Alignment**: While co-occurrence and popularity ranks are genuine real-world signals from Retailrocket, the alignment to SmartCart departments is a structural proxy, not a semantic product mapping.
2. **Catalog Compression (Many-to-One)**: 124,882 Retailrocket items are mapped into 262 catalog products (~476 historical items per catalog SKU).
3. **Cold Items in Post-Cutoff Splits**: 4,685 items appear strictly after the train-split cutoff. These items receive fallback confidence (0.50) and root-category default department mapping.
4. **Offline Evaluation Framing**: All evaluation on this mapped dataset constitutes **proxy offline evaluation on logged positive interactions**, not an unbiased counterfactual off-policy estimate.
