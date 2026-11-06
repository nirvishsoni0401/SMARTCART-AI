# Phase 10D Consolidation Report: Bug Fixes, History Parity, and Final Evaluation Freeze

> [!IMPORTANT]
> **STATUS**: Consolidated, verified, and **PERMANENTLY FROZEN**. All numbers in this report are auto-generated from `experiments/results/offline_master_results.json` by `scripts/generate_offline_report.py`. Hand-typed numbers are strictly prohibited.

## Executive Summary

Phase 10D successfully consolidates all offline evaluation artifacts across SmartCart AI:
1. **Department Heuristic Bug Repaired**: Identified and reproduced the 90.25% (full catalog) and 93.54% (pure discovery) figures from Phase 10C. Proved the root causes: future target department leakage (`score_for_target_dept(t_cat)`), zero-score rank-1 inflation on cold events, and unhandled score ties. Implemented ONE shared deterministic tie-aware ranking function (`compute_rank`), breaking ties by popularity prior, then product index.
2. **History Parity Established**: Evaluated RecentlyViewed, Department Heuristic, and LinUCB under identical history definitions across both **Lifetime History** (persistent cookies/accounts) and **30-Minute Session History**.
3. **Hybrid Slate Evaluation**: Evaluated $k=3$ and $k=5$ hybrid slates with visitor-clustered bootstrap CIs. Proven that hybrid slates statistically significantly outperform pure LinUCB without using unjustified 'optimal' claims.
4. **Single Master Results File**: All tables, metrics, and markdown documents across the repository are derived from a single JSON source of truth.
5. **Real Freeze**: The offline evaluation is permanently frozen. Further evaluation proceeds in the live interactive simulator.

---

## Part A. Department Heuristic Bug Reproduction & Tie-Aware Ranking Fix

### A1. Bug Reproduction
- **Reproduced Buggy Macro HR@10 (Full Catalog)**: 90.25%
- **Reproduced Buggy Micro HR@10 (Full Catalog)**: 74.25%
- **Reproduced Buggy Macro HR@10 (50 Candidates)**: 96.00%
- **Reproduced Buggy Micro HR@10 (50 Candidates)**: 89.49%

### A2. Root Causes Identified
1. **Target Category Leakage**: The legacy function `score_for_target_dept(t_cat)` received the future target's category `catalog[t_pid]['category']`, boosting the exact department of the upcoming item.
2. **Zero-Score Rank-1 Inflation**: For events where history was empty (226,410 cold session starts), the legacy code returned `np.zeros(262)`. Because rank was computed as `np.count_nonzero(scores > target_score) + 1`, `0 > 0` evaluated to 0, artificially assigning rank 1 (100% HR@10) to all 226,410 cold interactions.
3. **Tie-Breaking Failure**: `np.count_nonzero(scores > target_score) + 1` assumes the target strictly beats every item with which it is tied.

### A3. The Repair: Shared Deterministic Tie-Aware Ranking
All policies now use `scripts.retailrocket.shared_ranking.compute_rank`: ties are broken strictly by **popularity prior** (descending), then **product index** (ascending). Unit tests in `backend/tests/test_phase10d.py` confirm that a constant-score ranker achieves theoretical parity: $10/262 = 3.82\%$ on full catalog and $10/50 = 20.0\%$ on 50 candidates.

---

## Part B. History Parity Evaluation

### Table B1: Lifetime History (Primary Headline — Full Catalog)
*Evaluated across 335,612 events (194,155 visitors). Ground-truth repeat items defined by lifetime engagement history.*

#### Overall Test Events
| Policy | Macro HR@10 | 95% Macro CI | Micro HR@10 | 95% Micro CI |
| :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | 5.76% | [5.66%, 5.85%] | 5.92% | [5.80%, 6.04%] |
| **Department Heuristic** | 7.46% | [7.35%, 7.56%] | 10.12% | [9.93%, 10.31%] |
| **RecentlyViewed** | 13.10% | [12.99%, 13.24%] | 24.97% | [24.69%, 25.27%] |
| **Cold LinUCB** | 6.73% | [6.63%, 6.83%] | 8.39% | [8.23%, 8.55%] |
| **Warm LinUCB** | 11.20% | [11.08%, 11.32%] | 18.33% | [18.08%, 18.57%] |
| **Warm LinTS (v=0.25)** | 11.07% | [10.95%, 11.19%] | 18.21% | [17.98%, 18.44%] |
| **Hybrid Slate (k=3 RV + 7 LinUCB)** | 13.16% | [13.04%, 13.29%] | 24.41% | [24.16%, 24.68%] |
| **Hybrid Slate (k=5 RV + 5 LinUCB)** | 13.01% | [12.89%, 13.13%] | 24.67% | [24.41%, 24.95%] |

#### Novel Target Events
| Policy | Macro HR@10 | 95% Macro CI | Micro HR@10 | 95% Micro CI |
| :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | 5.76% | [5.66%, 5.85%] | 5.86% | [5.77%, 5.95%] |
| **Department Heuristic** | 6.42% | [6.32%, 6.52%] | 7.57% | [7.47%, 7.68%] |
| **RecentlyViewed** | 5.66% | [5.57%, 5.76%] | 5.42% | [5.33%, 5.51%] |
| **Cold LinUCB** | 5.60% | [5.51%, 5.70%] | 5.72% | [5.63%, 5.80%] |
| **Warm LinUCB** | 5.82% | [5.72%, 5.92%] | 6.13% | [6.04%, 6.22%] |
| **Warm LinTS (v=0.25)** | 5.75% | [5.65%, 5.84%] | 6.05% | [5.95%, 6.14%] |
| **Hybrid Slate (k=3 RV + 7 LinUCB)** | 5.83% | [5.74%, 5.93%] | 5.94% | [5.85%, 6.03%] |
| **Hybrid Slate (k=5 RV + 5 LinUCB)** | 5.58% | [5.49%, 5.68%] | 5.59% | [5.50%, 5.68%] |

#### Repeat Target Events
| Policy | Macro HR@10 | 95% Macro CI | Micro HR@10 | 95% Micro CI |
| :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | 5.99% | [5.75%, 6.25%] | 6.16% | [5.82%, 6.51%] |
| **Department Heuristic** | 20.28% | [19.83%, 20.68%] | 19.50% | [18.86%, 20.16%] |
| **RecentlyViewed** | 99.81% | [99.78%, 99.84%] | 96.84% | [95.49%, 97.86%] |
| **Cold LinUCB** | 19.71% | [19.27%, 20.13%] | 18.23% | [17.61%, 18.84%] |
| **Warm LinUCB** | 71.36% | [70.87%, 71.81%] | 63.18% | [61.77%, 64.42%] |
| **Warm LinTS (v=0.25)** | 70.76% | [70.27%, 71.23%] | 62.92% | [61.54%, 64.11%] |
| **Hybrid Slate (k=3 RV + 7 LinUCB)** | 98.28% | [98.17%, 98.39%] | 92.30% | [90.94%, 93.40%] |
| **Hybrid Slate (k=5 RV + 5 LinUCB)** | 99.31% | [99.25%, 99.38%] | 94.80% | [93.44%, 95.90%] |

#### Paired Differences (Lifetime History — Overall Events)
| Comparison | Macro Diff | 95% Diff CI | $p$-value | Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Hybrid (k=3) vs RecentlyViewed** | +0.06% [-0.05%, 0.17%] | [-0.05%, | 0.1305 | Non-significant |
| **Hybrid (k=3) vs Warm LinUCB** | +1.97% [1.88%, 2.05%] | [1.88%, | < 0.0005 | Statistically Significant |
| **Hybrid (k=5) vs RecentlyViewed** | -0.10% [-0.18%, -0.01%] | [-0.18%, | 0.9845 | Non-significant |
| **Hybrid (k=5) vs Warm LinUCB** | +1.81% [1.71%, 1.92%] | [1.71%, | < 0.0005 | Statistically Significant |
| **Department Heuristic vs Popularity** | +1.70% [1.66%, 1.74%] | [1.66%, | < 0.0005 | Statistically Significant |
| **Warm LinUCB vs Popularity** | +5.44% [5.29%, 5.59%] | [5.29%, | < 0.0005 | Statistically Significant |

---

### Table B2: 30-Minute Session History Parity (Full Catalog)
*Evaluated under strict 30-minute session resets (reflecting unauthenticated guest browsing).*

#### Overall Test Events (Session History)
| Policy | Macro HR@10 | 95% Macro CI | Micro HR@10 | 95% Micro CI |
| :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | 5.76% | [5.66%, 5.85%] | 5.92% | [5.80%, 6.04%] |
| **Department Heuristic** | 6.99% | [6.89%, 7.09%] | 9.12% | [8.94%, 9.30%] |
| **RecentlyViewed** | 10.90% | [10.79%, 11.02%] | 19.05% | [18.83%, 19.30%] |
| **Cold LinUCB** | 6.40% | [6.30%, 6.49%] | 7.61% | [7.48%, 7.75%] |
| **Warm LinUCB** | 9.57% | [9.46%, 9.68%] | 14.68% | [14.50%, 14.87%] |
| **Warm LinTS (v=0.25)** | 9.45% | [9.33%, 9.55%] | 14.53% | [14.36%, 14.71%] |
| **Hybrid Slate (k=3 RV + 7 LinUCB)** | 10.98% | [10.86%, 11.09%] | 18.92% | [18.71%, 19.15%] |
| **Hybrid Slate (k=5 RV + 5 LinUCB)** | 10.80% | [10.68%, 10.91%] | 18.95% | [18.73%, 19.18%] |

#### Repeat Target Events (Session History)
| Policy | Macro HR@10 | 95% Macro CI | Micro HR@10 | 95% Micro CI |
| :--- | :--- | :--- | :--- | :--- |
| **Popularity Baseline** | 5.99% | [5.75%, 6.25%] | 6.16% | [5.82%, 6.51%] |
| **Department Heuristic** | 16.10% | [15.71%, 16.48%] | 15.81% | [15.24%, 16.44%] |
| **RecentlyViewed** | 73.23% | [72.78%, 73.67%] | 68.71% | [67.80%, 69.55%] |
| **Cold LinUCB** | 15.92% | [15.54%, 16.31%] | 14.62% | [14.14%, 15.10%] |
| **Warm LinUCB** | 52.99% | [52.48%, 53.50%] | 46.22% | [45.28%, 47.07%] |
| **Warm LinTS (v=0.25)** | 52.52% | [52.00%, 53.03%] | 45.94% | [45.03%, 46.76%] |
| **Hybrid Slate (k=3 RV + 7 LinUCB)** | 72.41% | [71.95%, 72.85%] | 66.53% | [65.62%, 67.38%] |
| **Hybrid Slate (k=5 RV + 5 LinUCB)** | 73.05% | [72.59%, 73.50%] | 67.91% | [66.98%, 68.74%] |

---

## Part C. Master Dataset Counts & Discrepancies Reconciliation

| Subset / Slice | Events ($n$) | Visitors ($n_v$) | Description |
| :--- | :--- | :--- | :--- |
| **Total Stream** | 335,612 | 194,155 | All held-out test events |
| **Novel Targets** | 263,849 | 194,155 | Unseen by shopper in test stream |
| **Repeat Targets** | 71,763 | 31,609 | Previously seen by shopper in test stream |
| **Cold Context** | 226,410 | 194,155 | Session-starting events (history = 0) |
| **First Exposure** | 287,176 | 194,155 | First exposure per session-product |

---

## Part D. Mapping Comparison Invalidation

> [!WARNING]
> The Phase 10B/10C mapping comparison evaluated a model warm-started on popularity_balanced_train against test events mapped under category_rank_train and cooccurrence_cluster_train. This cross-mapping evaluation creates an invalid representation domain shift where learned theta weights do not correspond to item identities. Furthermore, category_rank and cooccurrence_cluster suffer from extreme traffic concentration (Gini 0.65, 33.8% top-10 volume share) that artificially distorts popularity baselines. Per Phase 10D protocol, this cross-mapping evaluation is labeled mathematically invalid and formally removed from all documents.

---

## Part E. Gamma Sweep & Prequential Replay

### E1. Validation Gamma Sweep (Single n=20,000, Fixed Seed)

| Discount Factor ($\gamma$) | Validation HR@10 | Notes |
| :--- | :--- | :--- |
| **1.0000** | 16.23% | Best (Un-discounted) |
| **0.9999** | 15.79% | Monotonic degradation |
| **0.9995** | 13.91% | Monotonic degradation |
| **0.9990** | 11.06% | Monotonic degradation |
| **0.9950** | 6.77% | Monotonic degradation |
| **0.9900** | 5.92% | Monotonic degradation |

### E2. Prequential Replay over 20,000 Steps
*Rolling window $w=1000$. First step where Cold rolling HR enters within 1 SE of Warm rolling HR: **Step 8349**.*

| Step | Cold Rolling HR@10 | Warm Rolling HR@10 | Cold Cumulative HR@10 | Warm Cumulative HR@10 | Novel-Only Cold Cum | Novel-Only Warm Cum |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2,500** | 7.50% | 16.60% | 4.04% | 14.64% | 1.68% | 6.43% |
| **5,000** | 11.90% | 15.70% | 7.80% | 14.94% | 2.17% | 6.34% |
| **7,500** | 14.10% | 16.30% | 9.69% | 15.56% | 2.49% | 6.45% |
| **10,000** | 14.10% | 16.00% | 10.72% | 15.64% | 2.66% | 6.44% |
| **15,000** | 14.30% | 16.50% | 12.23% | 16.07% | 3.27% | 6.38% |
| **20,000** | 17.70% | 17.80% | 13.06% | 16.15% | 3.59% | 5.96% |

---

## Part F & G. Scientific Verification & Real Freeze

- All unit tests pass (`backend/tests/test_phase10d.py`).
- The offline Retailrocket evaluation is **PERMANENTLY FROZEN**.
- No further evaluation phases or offline benchmarks will be conducted.
