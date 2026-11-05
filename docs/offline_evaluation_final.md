# SmartCart AI — Final Offline Evaluation Report (Retailrocket Historical Dataset)

**Document Version**: `v2.0 (Phase 10C Reconciled Freeze)`  
**Execution Timestamp**: 2026-11-05 UTC  
**Target Environment**: FastAPI Backend + SQLite (WAL mode, `smartcart.db`, live database untouched)  
**Historical Database**: `historical.db` (2,555,684 filtered interactions, 124,882 items)  
**Catalog Representation**: 262 Active SKUs (157 DummyJSON + 105 Seeded Books)  
**Feature Vector**: $d = 36$, `FEATURE_SPEC_HASH = "75f60ed70b2a"`  
**Frozen Hyperparameters**: $N_{\text{cap}} = 10,000, m = 3, \lambda = 1.0, \alpha = 0.5$  
**Status**: **PERMANENTLY FROZEN**

> [!IMPORTANT]
> **Strict Evaluation Protocol Compliance**:
> All results in this report represent a **proxy offline evaluation on logged positive-only interactions**.
> 1. The system is **NOT** described as "production-ready".
> 2. The term **"causal"** is **NEVER** used. Observational logged clicks with many-to-one proxy mappings cannot establish causal treatment effects or counterfactual guarantees.
> 3. No live database state in `smartcart.db` has been modified, and warm-start weights have **NOT** been applied to live models.

---

## 1. Evaluation Methodology & Metric Definitions

Using chronological time splits:
- **Training Period** (May 3 to Aug 7, 2015, $73.6\%$ of events): Used to extract sufficient statistics ($A_{\text{hist}}, b_{\text{hist}}$) under pseudo-negative sampling ($m=3$).
- **Validation Period** (Aug 7 to Aug 28, 2015, $13.3\%$ of events): Used to tune exploration hyperparameters and verify discounting degradation.
- **Test Period** (Aug 28 to Sep 18, 2015, $335,612$ interactions from $194,155$ unique visitors): Held out for final evaluation.

### Metric Definitions:
- **Event-Level (Micro) Hit Rate @ 10**:
  $$\text{Micro HR@10} = \frac{1}{|E|} \sum_{e \in E} \mathbb{I}(\text{rank}(t_e) \le 10)$$
- **Visitor-Level (Macro) Hit Rate @ 10**:
  $$\text{Macro HR@10} = \frac{1}{|V_E|} \sum_{v \in V_E} \left( \frac{1}{|E_v|} \sum_{e \in E_v} \mathbb{I}(\text{rank}(t_e) \le 10) \right)$$
  where $V_E$ includes strictly visitors with $\ge 1$ interaction in subset $E$ ($|V_E| = 194,155$ overall; $|V_E| = 31,609$ for repeats).
- **Confidence Intervals**: Computed using 2,000-resample visitor-clustered bootstrap.

---

## 2. Supported Claims with Reconciled Evidence

| # | Supported Claim | Empirical Evidence (`phase10b_master_results.json`) | Statistical Significance |
| :--- | :--- | :--- | :---: |
| **S1** | **Bandit Matches Popularity on Novel Targets** | Warm LinUCB achieves **5.58% [5.48%, 5.67%]** Macro HR@10 vs Popularity **5.76% [5.66%, 5.85%]** on novel targets ($78.6\%$ of events). | Diff $-0.18\%$, $p = 0.9990$ (Negligible $<0.5\%$) |
| **S2** | **Department Heuristic Outperforms Bandit for Discovery** | Department Heuristic achieves **93.50% [93.42%, 93.58%]** Macro HR@10 on novel targets by concentrating on target departments. | $p < 0.0005$ (Statistically significant) |
| **S3** | **RecentlyViewed Dominates Repeats** | Simple `RecentlyViewed` heuristic achieves **73.22% [72.77%, 73.66%]** Macro HR@10 on repeat targets, beating Warm LinUCB (**52.78%**). | Simple LRU cache beats bandit by **+20.44%** |
| **S4** | **Candidate Exclusion v2 Levels Discovery Playing Field** | Masking all previously engaged products reduces Warm LinUCB Novel Macro HR@10 to **5.63%**, matching Popularity (**5.77%**). | Diff $-0.14\%$, $p = 0.9985$ |
| **S5** | **Feature Norm Correlates with Popularity (Partial Support)** | Cold LinUCB feature norm score ($\alpha \|\phi\|_2$) correlates with popularity prior ($r = 0.7126, p = 6.59 \times 10^{-42}$). Within top-20, $r=0.2692$. | Broad catalog tiering supported |
| **S6** | **Brand Affinity Drives Repeat Advantage** | Refitting without Feature 6 (`affinity_brand`) collapses repeat Macro HR from **52.78% to 5.74%**, while novel HR is untouched (**5.58% $\rightarrow$ 5.63%**). | Exact leave-group-out refit verification |
| **S7** | **Zero Discounting ($\gamma=1.0$) Outperforms Forgetting** | Stationary LinUCB ($\gamma=1.0$) achieves $5.20\%$ HR on validation replay, outperforming discounted models ($\gamma \le 0.9995 \rightarrow 4.04\% - 4.61\%$). | Monotonic degradation confirmed |
| **S8** | **Hybrid Slate Boosts LinUCB via Recency** | Hybrid slate (top 3 from RV + 7 from LinUCB) achieves **10.75%** Overall Macro HR@10, beating pure Warm LinUCB ($9.29\%$). | Paired Diff $+1.46\%$, $p < 0.0005$ |

---

## 3. Withdrawn or Revised Claims

| # | Previous Claim | Revised Status in Phase 10C | Reason for Revision / Withdrawal |
| :--- | :--- | :--- | :--- |
| **W1** | *"Warm LinUCB provides a 10.79% macro hit rate, doubling popularity (5.76%)."* | **REVISED**: Advantage is confined to repeats ($21.4\%$ of events). On novel discovery targets, Warm LinUCB equals Popularity (**5.58% vs 5.76%**). | Headline metrics were inflated by repeat visits; disaggregation reveals zero discovery advantage over popularity. |
| **W2** | *"Cold LinUCB crosses Warm LinUCB at ~1,000 test steps."* (Claim C7) | **FORMALLY WITHDRAWN**. Cold LinUCB cumulative HR@10 never surpasses Warm LinUCB within 20,000 steps (**12.96% vs 14.73%**). | Phase 10A script had inverted candidate scoring artifact. In rolling terms, Cold matches Warm near step 7,200, but Warm retains cumulative lead. |
| **W3** | *"Ablating brand drops performance to 1.31%, showing brand is critical."* | **REVISED**: Zeroing features at inference caused dot-product mismatch. Valid leave-group-out refits drop repeat HR from **52.78% to 5.74%**, while novel HR is unaffected (**5.58% $\rightarrow$ 5.63%**). | Ridge refit rebalances remaining 35 weights without falling below the random floor. |
| **W4** | *"RecentlyViewed achieves 99.71% on repeat targets."* | **REVISED**: 99.71% occurred under unreset lifetime history. Under verified 30-min sessionization, RecentlyViewed achieves **73.22% Macro / 68.57% Micro** because 26.4% of repeats occur across sessions where history starts empty. | Sessionization boundary enforcement correctly captures cross-session cold starts. |

---

## 4. Reconciled Master Evaluation Tables

### 4.1 Primary Headline: Novel Target Interactions ($n = 263,849$, $78.62\%$ of Test Stream)
*Setting: Full Catalog (262 SKUs) | Mode: Per-Event | Target: Novel | Exclusion: Unconstrained*

```
Policy                               Macro HR@10 (95% Clustered CI)    Micro HR@10 (95% Clustered CI)
-----------------------------------------------------------------------------------------------------
Department Heuristic                 93.50% [93.42%, 93.58%]           82.79% [82.52%, 83.08%]
DepartmentRecency                     6.33% [ 6.23%,  6.43%]            7.40% [ 7.26%,  7.54%]
SameBrandPopularity                   5.86% [ 5.76%,  5.96%]            6.14% [ 6.01%,  6.26%]
Popularity Baseline                   5.76% [ 5.66%,  5.85%]            5.86% [ 5.74%,  5.98%]
RecentlyViewed                        5.69% [ 5.59%,  5.78%]            5.55% [ 5.43%,  5.66%]
Warm LinUCB (Optimal UCB)             5.58% [ 5.48%,  5.67%]            5.88% [ 5.76%,  6.00%]
Cold LinUCB (alpha=0.5)               5.57% [ 5.48%,  5.67%]            5.65% [ 5.53%,  5.77%]
Warm LinTS (v=0.25)                   5.39% [ 5.30%,  5.48%]            5.72% [ 5.60%,  5.84%]
Aggregated Warm LinUCB                3.11% [ 3.04%,  3.18%]            3.74% [ 3.63%,  3.84%]
```

### 4.2 Secondary Breakdown: Repeat Target Interactions ($n = 71,763$, $21.38\%$ of Test Stream)
*Setting: Full Catalog (262 SKUs) | Mode: Per-Event | Target: Repeat | Exclusion: Unconstrained*

```
Policy                               Macro HR@10 (95% Clustered CI)    Micro HR@10
-----------------------------------------------------------------------------------
RecentlyViewed (LRU Cache)           73.22% [72.77%, 73.66%]           68.57%
Warm LinUCB (Optimal UCB)            52.78% [52.26%, 53.29%]           46.04%
Warm LinTS (v=0.25)                  52.45% [51.94%, 52.95%]           45.76%
SameBrandPopularity                  50.33% [49.80%, 50.86%]           44.49%
Department Heuristic                 42.93% [42.40%, 43.45%]           42.85%
DepartmentRecency                    16.79% [16.40%, 17.20%]           16.47%
Popularity Baseline                   5.99% [ 5.72%,  6.25%]            6.16%
Cold LinUCB (alpha=0.5)               5.85% [ 5.60%,  6.10%]            5.98%
```

### 4.3 Overall Test Events with Hybrid Slates ($n = 335,612$, $194,155$ Visitors)
*Setting: Full Catalog (262 SKUs) | Mode: Per-Event | Target: Overall | Exclusion: Unconstrained*

```
Policy                               Macro HR@10 (95% Clustered CI)    Micro HR@10 (95% Clustered CI)
-----------------------------------------------------------------------------------------------------
Department Heuristic                 90.25% [90.16%, 90.34%]           74.25% [73.80%, 74.67%]
RecentlyViewed                       10.90% [10.79%, 11.01%]           19.02% [18.81%, 19.26%]
Hybrid Slate (k=3 RV + 7 LinUCB)     10.75% [10.64%, 10.85%]           18.55% [18.35%, 18.76%]
Hybrid Slate (k=5 RV + 5 LinUCB)     10.64% [10.53%, 10.75%]           18.78% [18.57%, 19.00%]
Warm LinTS (v=0.25)                   9.46% [ 9.36%,  9.57%]           14.51% [14.34%, 14.69%]
Warm LinUCB (Optimal UCB)             9.29% [ 9.18%,  9.40%]           14.44% [14.26%, 14.62%]
Popularity Baseline                   5.76% [ 5.66%,  5.85%]            5.92% [ 5.81%,  6.04%]
```

---

## 5. Honest Limitations

1. **Proxy Mapping & Many-to-One Aliasing**:
   Retailrocket product IDs are anonymized item identifiers mapped via category and price proxies to the 262-SKU catalog. Results demonstrate transferability of contextual behavior (brand affinity, price sensitivity, category preference), rather than item-level identity.
2. **Positive-Only Logged Observational Logs**:
   The dataset logs exclusively positive interactions (clicks, adds-to-cart, transactions). Offline evaluations measure *whether the model ranks the user's eventual chosen item in the top 10*, not whether showing recommended items would have induced higher click-through rates.
3. **Absence of Counterfactual / Causal Validity**:
   No offline evaluation using static logs can establish causal treatment effects or guarantee performance in a live environment. The results are strictly descriptive of historical replay under proxy assumptions.
4. **Cold Context Dominance**:
   In $67.46\%$ of test events, the visitor has zero prior history in that session. Under cold contexts, contextual models lack customer context and degenerate into static popularity or norm-based rankings.
5. **Repeat Bias Inflation**:
   In unconstrained evaluations, aggregate metrics are dominated by the $21.4\%$ of repeat events where customer intent is already established. A trivial recency cache outperforms complex contextual models on this slice.

---

## 6. Offline Evaluation Freeze

With the delivery of this reconciled report and all associated test suites (`backend/tests/test_phase10c.py`), the offline Retailrocket evaluation for SmartCart AI is **PERMANENTLY FROZEN**. No further offline evaluation phases or metric revisions will be conducted.
