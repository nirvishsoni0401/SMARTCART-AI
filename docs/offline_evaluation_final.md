# SmartCart AI — Final Offline Evaluation Report (Retailrocket Proxy)

**Document Status**: **PERMANENTLY FROZEN**  
**Version**: `v1.0 (Final Summary)`  
**Date**: November 2026  
**Evaluation Target**: Contextual Multi-Armed Bandit Recommender System (LinUCB / LinTS)  
**Historical Log Dataset**: Retailrocket E-Commerce Interaction Stream ($2,555,684$ filtered events)  
**Catalog**: 262 Active SKUs (157 DummyJSON Products + 105 Curated Books)  
**Feature Space**: 36-Dimensional Joint Context-Item Representation (`FEATURE_SPEC_HASH = "75f60ed70b2a"`)  
**Frozen Hyperparameters**: $N_{\text{cap}} = 10,000, m = 3, \lambda = 1.0, \alpha = 0.5$

---

## 1. Methodology in Plain Language

SmartCart AI incorporates a Contextual Multi-Armed Bandit recommendation engine designed to dynamically rank products for e-commerce customers based on customer interaction context (e.g., categories viewed, brand preferences, price sensitivity, session budget).

To evaluate whether offline pre-training (warm-start) benefits the live system, we utilized a historical e-commerce clickstream from Retailrocket ($2.55$ million events spanning May to September 2015). Because Retailrocket item IDs are anonymized numbers, we established an offline proxy mapping that connects Retailrocket items to SmartCart's active 262-product catalog using category structure, co-occurrence clustering, and popularity rank preservation.

Using chronological time splits:
- **Training Period** (first 73.6% of timeline, May 3 to Aug 7, 2015): Used to extract sufficient statistics ($A_{\text{hist}}, b_{\text{hist}}$) under positive interaction rewards and pseudo-negative sampling ($m=3$).
- **Validation Period** (middle 13.3%, Aug 7 to Aug 28, 2015): Used to tune exploration hyperparameters ($\alpha=0.5, \lambda=1.0, N_{\text{cap}}=10,000$) and evaluate discounting factors ($\gamma=1.0$).
- **Test Period** (final 13.1%, Aug 28 to Sep 18, 2015, $335,612$ interactions from $194,155$ unique visitors): Held out for final evaluation.

At each test interaction:
1. The visitor's history prior to that exact event is assembled into a 36-dimensional user context vector.
2. The recommender policy scores all candidate products in the catalog.
3. We record whether the user's actual target product appeared in the **Top-10 recommendations (Hit Rate @ 10)**.
4. Confidence intervals ($95\%$) are computed using **paired visitor-clustered bootstrap resampling (2,000 resamples)** to account for visitor-level clustering.

---

## 2. Supported Claims with Empirical Evidence

| # | Supported Claim | Empirical Metric / Evidence | Statistical Significance |
| :--- | :--- | :--- | :---: |
| **S1** | **Bandit Matches Popularity on Novel Targets** | Warm LinUCB achieves **5.77% [5.64%, 5.90%]** Macro HR@10 vs Popularity **5.76% [5.63%, 5.89%]** on novel targets ($78.6\%$ of events). | Diff $+0.01\%$, $p = 0.3840$ (Negligible) |
| **S2** | **Department Heuristic Outperforms Bandit for Discovery** | Department Heuristic achieves **6.83% [6.68%, 6.98%]** Macro HR@10 on novel targets, beating Warm LinUCB by **+1.06%**. | $p < 0.0005$ (Statistically significant) |
| **S3** | **RecentlyViewed Dominates Repeats** | Simple `RecentlyViewed` heuristic achieves **99.71%** Macro HR@10 on repeat targets, compared to Warm LinUCB's **23.41%**. | Trivial recency cache dominates bandit by $+76.3\%$ |
| **S4** | **Candidate Exclusion v2 Levels Discovery Playing Field** | Excluding all previously viewed items reduces Warm LinUCB Macro HR@10 to **5.58%**, matching Popularity (**5.57%**). | Diff $+0.01\%$, $p = 0.4120$ |
| **S5** | **Feature Norm Correlates with Popularity** | Cold LinUCB exploration bonus ($\alpha \|\phi\|_2$) strongly correlates with catalog popularity prior: Spearman $r = 0.7126$. | $p = 6.59 \times 10^{-42}$ |
| **S6** | **Brand Affinity Drives Repeat Advantage** | Refitting without Feature 6 (`affinity_brand`) collapses repeat Macro HR from $23.41\%$ to $5.81\%$, but leaves novel HR unaffected ($5.77\% \rightarrow 5.76\%$). | Exact leave-group-out refit verification |
| **S7** | **Zero Discounting ($\gamma=1.0$) Outperforms Forgetting** | Stationary LinUCB ($\gamma=1.0$) achieves $5.20\%$ HR on validation replay, outperforming discounted models ($\gamma \le 0.9995$). | Verified on 20,000-event validation stream |

---

## 3. Withdrawn or Revised Claims Since Phase 9B

| # | Previous Claim (Phase 9B/9C/10A) | Revised Status in Phase 10B | Reason for Revision / Withdrawal |
| :--- | :--- | :--- | :--- |
| **W1** | *"Warm LinUCB provides a 10.79% macro hit rate, doubling popularity (5.76%)."* | **REVISED**: Advantage is confined to repeats ($21.4\%$ of events). On novel discovery targets, Warm LinUCB equals Popularity ($5.77\%$ vs $5.76\%$). | Headline metrics were inflated by repeat visits; disaggregation reveals zero discovery advantage over popularity. |
| **W2** | *"Cold LinUCB crosses Warm LinUCB at ~1,000 test steps."* (Claim C7) | **FORMALLY WITHDRAWN**. Cold LinUCB never crosses Warm LinUCB within 20,000 steps ($2.84\%$ vs $5.78\%$). | Phase 10A prequential script contained an inverted ranking calculation artifact. Repaired replay shows no crossover. |
| **W3** | *"Ablating brand drops performance to 1.31%, showing brand is critical."* | **REVISED**: Zeroing features at inference caused a severe train/test mismatch. With valid leave-group-out refits, brand ablation drops repeat HR from $23.4\%$ to $5.8\%$, while novel HR is unaffected ($5.77\%$). | True Ridge refit rebalances remaining 35 weights without falling below the random floor. |
| **W4** | *"Session reset caused split boundary shift from 1440756710277 to 1440788640321."* | **CLARIFIED**: Typographical typo in Phase 10A text. Exact split boundaries in code and SQLite database remained identical ($1440756710277$). | Database and code inspection confirmed zero shift in event boundaries. |

---

## 4. Novel-First Headline Results (Full Catalog 262 SKUs)

### Primary Headline: Novel Target Interactions ($n = 263,849$, $78.62\%$ of Test Stream)
*Evaluates the engine's capability to discover and recommend products the customer has NOT previously viewed.*

```
Policy                               Macro HR@10 (95% Clustered CI)    Micro HR@10 (95% Clustered CI)
-----------------------------------------------------------------------------------------------------
Department Heuristic                 6.83% [6.68%, 6.98%]              7.12% [6.95%, 7.29%]
DepartmentRecency                    6.83% [6.68%, 6.98%]              7.12% [6.95%, 7.29%]
SameBrandPopularity                  5.78% [5.65%, 5.91%]              5.95% [5.80%, 6.10%]
Warm LinUCB (Optimal UCB)            5.77% [5.64%, 5.90%]              5.98% [5.82%, 6.13%]
Aggregated Warm LinUCB               5.76% [5.63%, 5.89%]              5.96% [5.81%, 6.11%]
RecentlyViewed                       5.76% [5.63%, 5.89%]              5.92% [5.77%, 6.07%]
Popularity Baseline                  5.76% [5.63%, 5.89%]              5.92% [5.77%, 6.07%]
Warm LinTS (v=0.25)                  5.71% [5.58%, 5.84%]              5.91% [5.76%, 6.06%]
Cold LinUCB (alpha=0.5)              5.67% [5.54%, 5.80%]              5.85% [5.70%, 6.00%]
```

### Secondary Breakdown: Repeat Target Interactions ($n = 71,763$, $21.38\%$ of Test Stream)
*Evaluates re-engagement with previously viewed products.*

```
Policy                               Macro HR@10                       Micro HR@10
-----------------------------------------------------------------------------------
RecentlyViewed Baseline              99.71%                            99.78%
SameBrandPopularity                  67.05%                            71.20%
Warm LinUCB (Optimal UCB)            23.41%                            46.25%
DepartmentRecency                    16.18%                            22.40%
Cold LinUCB (alpha=0.5)               9.08%                            14.07%
Popularity Baseline                   5.76%                             5.92%
```

---

## 5. Honest Limitations

1. **Proxy Mapping & Many-to-One Aliasing**:
   The Retailrocket dataset contains 124,882 distinct items, whereas SmartCart's catalog contains 262 SKUs. Mapping thousands of distinct external items to single catalog products introduces aliasing: multiple unrelated consumer intents converge on the same product ID.
2. **Positive-Only Logged Observational Logs**:
   The dataset logs exclusively positive interactions (clicks, adds-to-cart, transactions). Unchosen candidates were never shown to the user in the logged session. Offline evaluations measure *whether the model ranks the user's eventual chosen item in the top 10*, not whether showing recommended items would have induced higher click-through rates.
3. **Absence of Counterfactual / Causal Validity**:
   No offline evaluation using static logs can establish causal treatment effects or guarantee performance in a live environment. The results are strictly descriptive of historical replay under proxy assumptions.
4. **Cold Context Dominance**:
   In $67.46\%$ of test events, the visitor has zero prior history in that session. Under cold contexts, contextual models lack customer context and degenerate into static popularity or norm-based rankings.
5. **Repeat Bias Inflation**:
   In unconstrained evaluations, aggregate metrics are dominated by the $21.4\%$ of repeat events where customer intent is already established. A trivial recency cache outperforms complex contextual models on this slice.

---

## 6. Offline Evaluation Freeze

With the delivery of this final report, all code, test suites (`backend/tests/test_phase10b.py`), master results files (`experiments/results/phase10b_master_results.json`), and generated figures, the offline Retailrocket evaluation for SmartCart AI is **PERMANENTLY FROZEN**.

No further offline evaluation phases, parameter sweeps, or metric revisions will be undertaken. Future work on recommendations should focus on live online A/B testing with randomized traffic.

