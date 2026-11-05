# SmartCart AI — Phase 10C Master Document Reconciliation & Evaluation Re-Freeze

**Execution Date**: 2026-11-05 UTC  
**Evaluation Scope**: Offline Proxy Evaluation on Retailrocket Logged Interactions  
**Target Environment**: FastAPI Backend + SQLite (WAL mode, `smartcart.db`, untouched)  
**Catalog**: 262 Active SKUs (157 DummyJSON + 105 Seeded Books)  
**Historical DB**: `backend/historical.db` (2,555,684 filtered interactions, 124,882 items)  
**Feature Vector**: $d = 36$, `FEATURE_SPEC_HASH = "75f60ed70b2a"`  
**Frozen Hyperparameters**: $N_{\text{cap}} = 10,000, m = 3, \lambda = 1.0, \alpha = 0.5$  
**Evaluation Status**: **PERMANENTLY FROZEN**

> [!IMPORTANT]
> **Strict Evaluation Framing & Protocol Compliance**:
> All results in this report are strictly a **proxy offline evaluation on logged positive-only interactions**.
> 1. The system is **NOT** described as "production-ready".
> 2. The term **"causal"** is **NEVER** used.
> 3. No live database state has been altered, and warm-start weights have **NOT** been applied to live models in `smartcart.db`.

---

## Part A. Print the Truth (Master Discrepancy Reconciliation)

During Phase 10B, discrepancies emerged between the chat summary and the preliminary drafted reports (`docs/offline_evaluation_final.md` and `docs/phase10b-report.md`). The table below presents the master ground truth from `experiments/results/phase10b_master_results.json`, juxtaposes the values, and documents the root cause of every discrepancy.

| Quantity / Question | Chat Summary Reported | Document Draft Reported | Master JSON Ground Truth (`phase10b_master_results.json`) | Exact Cause of Discrepancy |
| :--- | :---: | :---: | :---: | :--- |
| **1) Novel-Target Warm LinUCB Macro HR@10** | 5.58% [5.48%, 5.67%] | 5.77% [5.64%, 5.90%] | **5.58% [5.48%, 5.67%]** (Micro: 5.88%, $n=263,849$, $n_v=194,155$) | **Stale Draft Text**: The doc author carried over preliminary 10A draft figures before the full test run completed. |
| **1b) Novel Paired Diff: Warm LinUCB vs Popularity** | $-0.18\%$ ($p=0.9990$) | $+0.01\%$ ($p=0.3840$) | **$-0.18\%$ [$-0.30\%$, $-0.06\%$]** ($p=0.9990$, negligible size $<0.5\%$) | **Stale Draft Text**: Master test stream shows Warm LinUCB achieves $5.58\%$ vs Popularity $5.76\%$. |
| **2) Repeat-Target Warm LinUCB Macro / Micro** | 52.78% / 46.04% | 23.41% / 46.25% | **Macro: 52.78% [52.26%, 53.29%]**, **Micro: 46.04%** ($n=71,763$, $n_v=31,609$) | **Macro Denominator Definition**: Doc draft divided total hits by all 194,155 visitors rather than the 31,609 visitors who had repeat events. |
| **2b) Repeat-Target RecentlyViewed Macro / Micro** | 73.22% / 68.57% | 99.71% / 99.78% | **Macro: 73.22% [72.77%, 73.66%]**, **Micro: 68.57%** ($n=71,763$, $n_v=31,609$) | **Session Reset vs Lifetime History**: The 99.7% draft evaluated lifetime history without session reset. Under 30-min sessionization, 26.4% of repeats start with empty session history. |
| **2c) Repeat-Target Cold LinUCB Macro** | 5.85% | 9.08% (10A: 15.92%) | **Macro: 5.85% [5.60%, 6.10%]**, **Micro: 5.98%** ($n=71,763$, $n_v=31,609$) | **Macro Denominator & Aliasing Fix**: Evaluated across visitors experiencing repeats under verified sessionization. |
| **3) Pure Discovery (Candidate Exclusion v2)** | 9.46% vs 5.77% | 5.58% vs 5.57% | **Overall: 9.46% vs 5.77%**; **Novel: 5.63% vs 5.77%** (Table D2: 5.56% vs 5.77%) | **Target Subset Confusion**: Chat cited Overall events under Exclusion v2 ($n=335,612$), while doc cited Novel-only events ($n=263,849$). |
| **4) Feature 6 Refit on Repeat Targets** | 52.78% $\rightarrow$ 5.74% | 23.41% $\rightarrow$ 5.81% | **52.78% $\rightarrow$ 5.74%** (89.1% collapse; Novel: 5.58% $\rightarrow$ 5.63%) | **Macro Denominator Alignment**: Correcting the full model repeat macro from 23.41% to 52.78% aligns the refit collapse cleanly. |
| **5) Prequential Cold vs Warm Learning** | Cumulative: 12.96% vs 14.73% | Cumulative: 2.84% vs 5.78% | **Rolling: 17.70% vs 17.40%**; **Cumulative: 12.96% vs 14.73%** (Novel: 3.51% vs 5.88%) | **Subset & Estimand Mislabeling**: Doc cited novel-only cumulative from early checkpoint. Rolling curves catch up near step 10k, but Warm keeps cumulative lead. |

---

## Part B. Precise Definitions & Metric Labeling Standard

To prevent ambiguity, every metric in SmartCart AI must be explicitly labeled by:
1. **Setting**: `Full Catalog (262 SKUs)` or `50 Candidates Pool` (target + 49 random items, seed 42).
2. **Target Subset**: `Novel` (first interaction with product), `Repeat` (previously engaged product), or `Overall` (all events).
3. **Exclusion Mode**: `Unconstrained` (all candidates), `Exclusion v1` (in-cart/purchased masked), or `Exclusion v2` (all previously engaged products masked — pure discovery).
4. **Estimand**: `Macro HR@10` (visitor-level) or `Micro HR@10` (event-level).
5. **Example Mode**: `Per-Event` (raw interaction stream, $N=335,612$) or `First-Exposure` (first event per session-product, $N=287,176$).
6. **Model Variant**: Canonical policy label.

### Mathematical Definitions:

- **Event-Level (Micro) Hit Rate @ 10**:
  $$\text{Micro HR@10} = \frac{1}{|E|} \sum_{e \in E} \mathbb{I}(\text{rank}(t_e) \le 10)$$
  where $E$ is the set of test events in the specified target subset. Each interaction contributes equally.

- **Visitor-Level (Macro) Hit Rate @ 10**:
  $$\text{Macro HR@10} = \frac{1}{|V_E|} \sum_{v \in V_E} \left( \frac{1}{|E_v|} \sum_{e \in E_v} \mathbb{I}(\text{rank}(t_e) \le 10) \right)$$
  where $V_E$ is the set of visitors who have **at least one interaction in subset $E$** ($|E_v| \ge 1$). Visitors who have **zero examples in subset $E$ are strictly excluded from the denominator $|V_E|$** (they do not receive a score of zero; they do not enter the average). For the overall set, $|V_E| = 194,155$; for the repeat subset, $|V_E| = 31,609$.

---

## Part C. Baseline Leak Check & History Size Analysis

### C1. Timing Invariant Verification
A dedicated unit test (`backend/tests/test_phase10c.py::test_recently_viewed_history_strictly_prior_to_event`) verifies that:
1. The history passed to `RecentlyViewedBaseline.score(history)` contains strictly prior interactions ($	au < t$).
2. The current event target product ($t$) is **never present in history** during evaluation.
3. If current interaction $	au \le t$ is artificially appended to history prior to ranking, the repeat hit rate becomes an artificial $100.0\%$.

### C2. Explanation of the 99.71% vs 73.22% Figure
- **The 99.71% Draft Result**: Produced when `RecentlyViewed` was evaluated with **lifetime unreset history** (ignoring session boundaries). Because the repeat target subset is defined as items previously interacted with, and **89.89% of visitors interact with $\le 10$ distinct products in their lifetime**, ranking past history in reverse chronological order mechanically guarantees $\approx 97\%-99.7\%$ hit rate.
- **The 73.22% Master Result**: Under verified sessionization (30-minute inactivity timeout), `curr_hist` is reset to `[]` whenever a new session starts. Therefore, cross-session repeat events (where the user returns days later to buy an item viewed in an earlier session) start with empty session history.

### C3. Repeat Examples Breakdown by History-Size Bucket ($n = 71,763$ Repeat Events)

| Session History Size Bucket | Number of Repeat Events | Percent of Repeat Set | RecentlyViewed Micro HR@10 | Explanation |
| :--- | :---: | :---: | :---: | :--- |
| **Bucket 0 (Cross-Session Repeat)** | 18,980 | 26.4% | **6.10%** | Session history starts empty after 30-min gap; RV falls back to static popularity prior. |
| **Bucket 1** | 20,754 | 28.9% | **92.14%** | History has 1 product; placed in slot 1. High hit rate. |
| **Bucket 2 - 5** | 21,651 | 30.2% | **93.18%** | History has 2-5 products; target is almost always in top 5. |
| **Bucket 6 - 10** | 5,423 | 7.6% | **91.04%** | History has 6-10 products; target ranked within top 10. |
| **Bucket > 10** | 4,955 | 6.9% | **76.95%** | History has >10 items; LRU ordering occasionally drops older targets. |
| **Overall Repeats** | **71,763** | **100.0%** | **68.57%** (Macro: **73.22%**) | Weighted mixture of 26.4% empty-session repeats and 73.6% active-session repeats. |

*Takeaway*: The repeat subset suffers from an extreme **selection effect**: by definition, the target is known to be in the user's past. Ranking past items first is near-trivial for active sessions.

---

## Part D. The Allowed New Tables: Overall Events & Pure Discovery

Evaluated across the entire test stream ($N=335,612$, $194,155$ visitors) with 2,000-resample visitor-clustered bootstrap confidence intervals (`experiments/results/phase10c_part_d_results.json`).

### Table D1: Overall Test Events (Unconstrained, All $335,612$ Events)

| Setting | Policy | Macro HR@10 | 95% Clustered CI | Micro HR@10 | 95% Clustered CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full Catalog (262 SKUs)** | **Department Heuristic** | **90.25%** | $[90.16\%, 90.34\%]$ | **74.25%** | $[73.80\%, 74.67\%]$ |
| Full Catalog (262 SKUs) | **RecentlyViewed** | 10.90% | $[10.79\%, 11.01\%]$ | 19.02% | $[18.81\%, 19.26\%]$ |
| Full Catalog (262 SKUs) | **Hybrid Slate (k=3 RV + 7 LinUCB)** | 10.75% | $[10.64\%, 10.85\%]$ | 18.55% | $[18.35\%, 18.76\%]$ |
| Full Catalog (262 SKUs) | **Hybrid Slate (k=5 RV + 5 LinUCB)** | 10.64% | $[10.53\%, 10.75\%]$ | 18.78% | $[18.57\%, 19.00\%]$ |
| Full Catalog (262 SKUs) | **Warm LinTS ($v=0.25$)** | 9.46% | $[9.36\%, 9.57\%]$ | 14.51% | $[14.34\%, 14.69\%]$ |
| Full Catalog (262 SKUs) | **Warm LinUCB ($\alpha=0.5$)** | 9.29% | $[9.18\%, 9.40\%]$ | 14.44% | $[14.26\%, 14.62\%]$ |
| Full Catalog (262 SKUs) | **Popularity Baseline** | 5.76% | $[5.66\%, 5.85\%]$ | 5.92% | $[5.81\%, 6.04\%]$ |
| **50 Candidates Pool** | **Department Heuristic** | **96.00%** | $[95.94\%, 96.06\%]$ | **89.49%** | $[89.27\%, 89.71\%]$ |
| 50 Candidates Pool | **RecentlyViewed** | 33.77% | $[33.59\%, 33.97\%]$ | 40.38% | $[40.14\%, 40.63\%]$ |
| 50 Candidates Pool | **Hybrid Slate (k=5 RV + 5 LinUCB)** | 33.61% | $[33.42\%, 33.80\%]$ | 40.66% | $[40.41\%, 40.93\%]$ |
| 50 Candidates Pool | **Hybrid Slate (k=3 RV + 7 LinUCB)** | 33.56% | $[33.37\%, 33.76\%]$ | 40.61% | $[40.35\%, 40.87\%]$ |
| 50 Candidates Pool | **Warm LinTS ($v=0.25$)** | 33.01% | $[32.81\%, 33.18\%]$ | 39.67% | $[39.42\%, 39.92\%]$ |
| 50 Candidates Pool | **Warm LinUCB ($\alpha=0.5$)** | 32.83% | $[32.64\%, 33.01\%]$ | 39.53% | $[39.28\%, 39.78\%]$ |
| 50 Candidates Pool | **Popularity Baseline** | 30.04% | $[29.85\%, 30.23\%]$ | 30.75% | $[30.50\%, 31.00\%]$ |

#### Paired Differences for Hybrid Slates (Table D1):
- **Full Catalog**:
  - Hybrid ($k=3$) vs Warm LinUCB: **$+1.46\%$** [$+1.38\%, +1.54\%$], $p < 0.0005$ (Statistically significant).
  - Hybrid ($k=3$) vs RecentlyViewed: **$-0.15\%$** [$-0.26\%, -0.04\%$], $p = 0.9930$ (Negligible).
  - Hybrid ($k=5$) vs Warm LinUCB: **$+1.35\%$** [$+1.25\%, +1.46\%$], $p < 0.0005$ (Statistically significant).
  - Hybrid ($k=5$) vs RecentlyViewed: **$-0.26\%$** [$-0.35\%, -0.17\%$], $p = 1.0000$ (Negligible).
- **50 Candidates**:
  - Hybrid ($k=3$) vs Warm LinUCB: **$+0.73\%$** [$+0.61\%, +0.86\%$], $p < 0.0005$ (Significant).
  - Hybrid ($k=5$) vs Warm LinUCB: **$+0.78\%$** [$+0.63\%, +0.94\%$], $p < 0.0005$ (Significant).

---

### Table D2: Pure Discovery Task (Novel Targets, Exclusion v2 All Engaged Masked, $n=263,849$)

| Setting | Policy | Macro HR@10 | 95% Clustered CI | Micro HR@10 | 95% Clustered CI | Paired Macro Diff vs Pop | p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Catalog** | **Department Heuristic** | **93.54%** | $[93.46\%, 93.62\%]$ | **83.07%** | $[82.81\%, 83.36\%]$ | $+87.77\%$ | $< 0.0005$ |
| Full Catalog | **Popularity Baseline** | 5.77% | $[5.67\%, 5.87\%]$ | 5.93% | $[5.84\%, 6.03\%]$ | Reference | — |
| Full Catalog | **Warm LinTS ($v=0.25$)** | 5.77% | $[5.68\%, 5.87\%]$ | 6.23% | $[6.13\%, 6.32\%]$ | **$+0.00\%$** | $0.4940$ |
| Full Catalog | **Warm LinUCB ($\alpha=0.5$)** | 5.56% | $[5.47\%, 5.66\%]$ | 6.09% | $[6.00\%, 6.19\%]$ | **$-0.21\%$** | $0.9985$ |
| **50 Candidates** | **Department Heuristic** | **97.32%** | $[97.28\%, 97.37\%]$ | **93.15%** | $[93.02\%, 93.28\%]$ | $+67.26\%$ | $< 0.0005$ |
| 50 Candidates | **Popularity Baseline** | 30.06% | $[29.85\%, 30.26\%]$ | 30.84% | $[30.64\%, 31.05\%]$ | Reference | — |
| 50 Candidates | **Warm LinTS ($v=0.25$)** | 29.50% | $[29.30\%, 29.69\%]$ | 31.01% | $[30.82\%, 31.20\%]$ | $-0.56\%$ | $1.0000$ |
| 50 Candidates | **Warm LinUCB ($\alpha=0.5$)** | 29.31% | $[29.10\%, 29.50\%]$ | 30.78% | $[30.59\%, 30.97\%]$ | $-0.76\%$ | $1.0000$ |

> [!IMPORTANT]
> **Definitive Answer to the Discovery Question**:
> Under strict candidate exclusion (masking all previously engaged products), **Warm LinUCB achieves 5.56% Macro HR@10 vs Popularity Baseline 5.77%** (difference $-0.21\%$, $p = 0.9985$, negligible). Warm LinTS achieves **5.77% vs 5.77%** (identically zero difference, $p=0.4940$). **Contextual bandits provide no discovery advantage over static popularity on unviewed products.**

---

## Part E. Prequential Online Replay Details

- **Window Size**: Rolling window size $w = 1,000$ chronological interactions.
- **Target Subset**: Rolling curves evaluate ALL incoming test events chronologically. Cumulative curves track both ALL events and NOVEL-ONLY events.

| Chronological Step | Cold LinUCB Rolling ($w=1000$) | Warm LinUCB Rolling ($w=1000$) | Cold LinUCB Cumulative HR@10 | Warm LinUCB Cumulative HR@10 | Novel-Only Warm Cumulative HR@10 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.00% | 10.00% | 0.00% | 10.00% | 8.51% |
| **500** | 1.20% | 13.60% | 1.20% | 13.60% | 6.18% |
| **2,000** | 3.70% | 13.80% | 2.75% | 13.70% | 5.79% |
| **5,000** | 12.40% | 14.00% | 8.02% | 13.96% | 5.87% |
| **7,200 (Catch-Up)** | **14.60%** | **14.60%** | 9.42% | 14.12% | 5.82% |
| **10,000** | 14.30% | 13.50% | 11.00% | 14.15% | 5.79% |
| **15,000** | 14.20% | 15.90% | 12.29% | 15.07% | 5.85% |
| **20,000** | 17.70% | 17.40% | **12.96%** | **14.73%** | **5.88%** |

*Accurate Prequential Statement*:
In **rolling terms** ($w=1,000$), Cold LinUCB catches up with Warm LinUCB at **Step 7,200** ($14.60\%$ vs $14.60\%$) as it learns visitor preferences and repeat engagements online. However, in **cumulative terms**, Warm LinUCB retains its initial exploration lead across the entire 20,000 steps ($14.73\%$ vs $12.96\%$). Cold LinUCB cumulative HR@10 never surpasses Warm LinUCB within 20,000 steps; Claim C7 remains formally withdrawn.

---

## Part F. Loose Ends Verification

1. **Validation Gamma Sweep (`phase10b_master_results.json["gamma_sweep_validation"]`)**:
   Warm-started on validation split ($n=20,000$). Verified monotonic degradation with discounting:
   $\gamma=1.0000 \rightarrow 5.20\%, \gamma=0.9995 \rightarrow 4.61\%, \gamma=0.9990 \rightarrow 4.14\%, \gamma=0.9950 \rightarrow 4.23\%, \gamma=0.9900 \rightarrow 4.04\%$.
2. **First-Exposure Evaluation (`experiments/results/retailrocket_phase10b_first_exposure.csv`)**:
   Full test split ($N=287,176$ first-exposure events): Warm LinUCB Overall Macro HR@10 = **7.96%**, Micro HR@10 = **11.23%**.
3. **Cold-Context Slices (`experiments/results/retailrocket_phase10b_slices.csv`)**:
   $n = 226,410$ cold events ($67.5\%$ of test stream): Warm LinUCB Macro HR@10 = **5.67%**, Popularity = **5.76%**.
4. **Mapping Comparison (`experiments/results/retailrocket_phase10b_mapping_comparison.csv`)**:
   - `popularity_balanced_train`: Ceiling = 7.64%, Pop Baseline = 5.94%, Warm LinUCB = 5.88%
   - `cooccurrence_cluster_train`: Ceiling = 25.10%, Pop Baseline = 24.94%, Warm LinUCB = 7.32%
   - `category_rank_train`: Ceiling = 27.59%, Pop Baseline = 14.78%, Warm LinUCB = 4.24%
5. **Feature Spec Hash & Table**:
   Canonical feature spec hash `75f60ed70b2a` ($d=36$) verified by automated test.
6. **9C-to-10A Headline Reconciliation (`experiments/results/retailrocket_phase10b_reconciliation.csv`)**:
   Warm Macro $10.79\% \rightarrow 9.37\%$ driven by 30-min session timeout reclassifying 32,255 events to cold context. Splits never shifted.
7. **Spearman Shrinkage Result (Partial Support)**:
   Across all 262 SKUs, feature norm correlates with popularity prior ($r = 0.7126, p = 6.59 \times 10^{-42}$). However, within the top-20 popular products alone, rank correlation is only $r = 0.2692$ (Cold vs Pop) and $r = 0.4902$ (Warm vs Pop). Feature norms provide broad popularity-tier separation, rather than exact micro-ranking.

---

## Part G & H. Enforcement & Permanent Freeze

The offline Retailrocket evaluation is **PERMANENTLY FROZEN**. All documents (`README.md`, `docs/offline_evaluation_final.md`, `docs/phase10b-report.md`) have been audited and updated to match `phase10b_master_results.json` and `phase10c_part_d_results.json`.
