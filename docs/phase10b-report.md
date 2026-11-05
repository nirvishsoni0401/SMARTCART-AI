# SmartCart AI — Phase 10B Master Offline Evaluation & Freeze Report

**Document Version**: `v1.0 (Final Freeze)`  
**Execution Timestamp**: 2026-11-05 UTC  
**Evaluation Scope**: Offline Proxy Evaluation on Retailrocket Logged Interactions  
**Target Environment**: FastAPI Backend + SQLite (WAL mode, `smartcart.db`)  
**Catalog Representation**: 262 Active SKUs (157 DummyJSON + 105 Seeded Books)  
**Historical Database**: `historical.db` (2,555,684 filtered interactions, 124,882 items)  
**Feature Dimension**: $d = 36$, `FEATURE_SPEC_HASH = "75f60ed70b2a"`  
**Frozen Hyperparameters**: $N_{\text{cap}} = 10,000, m = 3, \lambda = 1.0, \alpha = 0.5$  
**Status**: **PERMANENTLY FROZEN** — Final offline evaluation deliverable.

> [!IMPORTANT]
> **Evaluation Protocol & Terminology Compliance**:
> All results in this report represent a **proxy offline evaluation on logged positive-only interactions**. In accordance with strict evaluation protocols:
> 1. The system is **NOT** described as "production-ready"; it is documented as thoroughly tested and empirically verified under historical logged replay.
> 2. The term **"causal"** is **NEVER** used; logged positive-only observational data with many-to-one product proxy aliasing cannot establish causal treatment effects or counterfactual guarantees.
> 3. No product features have been added, no live database state in `smartcart.db` has been modified, and warm-start weights have **NOT** been applied to live models.

---

## 1. Executive Summary & Headline Findings

Phase 10B represents the final audit, repair, and permanent freeze of the SmartCart AI offline evaluation pipeline. Prior phases (Phase 9B, 9C, and 10A) uncovered discrepancies between reported metrics, anomalous prequential trajectories, invalid zeroed-feature ablations, and undocumented headline shifts. Phase 10B resolves every discrepancy, replaces flawed evaluation paths with mathematically sound formulations, and presents a **novel-first headline** disaggregated by discovery versus repeat engagement.

### Key Conclusions:
1. **Headline Finding on Novel Targets (Discovery)**:
   On novel product interactions (where the user has never previously engaged with the target product in the session or history, representing **78.62%** of all test events):
   - **Warm LinUCB achieves Macro HR@10 of 5.77% [95% CI: 5.64%, 5.90%]**, compared to the static **Popularity Baseline of 5.76% [95% CI: 5.63%, 5.89%]**.
   - The paired difference is $+0.01\%$ [$95\%$ CI: $-0.04\%, +0.06\%$, $p = 0.38$], which is **statistically non-significant and negligible in size ($< 0.5\%$)**.
   - The contextual bandit provides **no discovery advantage** over a static popularity baseline when recommending unviewed products from cold contexts.
2. **Where the Warm-Start Advantage Originates (Repeat Bias)**:
   Warm LinUCB's apparent headline advantage in overall unconstrained metrics ($9.54\%$ Macro HR@10 vs Popularity $5.76\%$) concentrates exclusively on **repeat target events (21.38% of events)**, where Warm LinUCB achieves **23.41%** Macro HR@10.
   - However, a trivial **`RecentlyViewed` heuristic** (which ranks the visitor's previously viewed items in reverse chronological order and fills remainder with popularity) achieves **99.71% Macro HR@10** on repeats.
   - The contextual bandit is vastly inferior to a simple recency cache on repeats ($23.41\%$ vs $99.71\%$). The bandit may not be claimed to provide a practical benefit for repeats when a simple recency list trivially dominates it.
3. **Pure Discovery Task (Candidate Exclusion v2)**:
   Under **Candidate Exclusion v2** (where ALL previously engaged products are strictly excluded from recommendation candidates):
   - All repeat advantages collapse. Warm LinUCB achieves **5.58% Macro HR@10**, identical to Popularity (**5.57% Macro HR@10**).
4. **Repaired Prequential Replay**:
   The anomalous Phase 10A rolling hit rates ($66\%$ to $92\%$) were traced to an evaluation artifact where scores were computed against inverted candidate subsets or unreset history accumulators. Clean chronological replay over 20,000 steps demonstrates realistic trajectories:
   - Warm LinUCB maintains a steady **5.8% - 7.0%** rolling HR@10.
   - Cold LinUCB starts at **0.0%** and gradually learns online, reaching **2.8%** at step 20,000.
   - **Cold LinUCB never crosses Warm LinUCB within 20,000 steps**; Phase 10A Claim C7 is formally withdrawn.
5. **Valid Leave-Group-Out Ridge Refit Ablations**:
   Ablations zeroing features at inference time produced catastrophic degradation (brand ablation dropping to $1.31\%$, below the $3.8\%$ random floor) due to train/test dot-product mismatch. Redoing ablations as **leave-group-out Ridge refits** from sufficient statistics demonstrates that dropping brand affinity drops repeat HR from $23.41\%$ to $5.81\%$, while novel HR is completely unaffected ($5.77\% \rightarrow 5.76\%$).

---

## 2. Part A: Reconciliation of Changed Numbers (Phase 9C vs 10A vs 10B)

In Phase 10A, the reported headline metrics shifted without documented explanation: Warm LinUCB Macro HR@10 dropped from $10.79\%$ to $9.54\%$, and Micro HR@10 dropped from $17.97\%$ to $14.60\%$, while Popularity remained exactly $5.76\%$.

### A1. Git Diff & Root Cause Analysis
A full git diff between `phase9c-complete` (`evaluate_offline_proxy.py`) and `phase10a-complete` (`evaluate_phase10a.py`) identified the single mechanism driving the change:

```python
# Phase 9C (evaluate_offline_proxy.py line 412):
# History was ONLY cleared when visitor_id changed:
if v_id != curr_v:
    curr_v = v_id
    curr_hist = []

# Phase 10A (evaluate_phase10a.py line 507):
# 30-minute session inactivity timeout added:
if v_id != curr_v or (ts - curr_sess_ts > 1800000):
    curr_v = v_id
    curr_hist = []
```

Because Retailrocket visitors frequently return over multiple days, Phase 9C accumulated interaction histories across entire visitor lifetimes. In Phase 10A, sessions inactive for $>30$ minutes reset `curr_hist = []`. This reclassified **32,255 events** from warm context (`len(curr_hist) >= 1`) to cold context (`len(curr_hist) == 0`).
- Because Warm LinUCB achieves **5.67%** HR on cold contexts versus **38.61%** on light contexts and **29.38%** on heavy contexts, expanding the cold slice directly reduced the aggregate macro and micro averages.
- **Popularity Baseline was mathematically invariant ($5.76\%$ Macro / $5.92\%$ Micro)** across all phases because static catalog popularity does not read visitor history.

### A2. Before / After Reconciliation Table

| Metric / Setting | Phase 9C Reported | Phase 10A Reported | Phase 10B Verified | Exact Cause of Change |
| :--- | :---: | :---: | :---: | :--- |
| **Warm LinUCB Macro HR@10 (Full)** | 10.79% | 9.54% | **9.54%** | 30-min session timeout shifted 32,255 events to cold context ($5.67\%$ HR). |
| **Warm LinUCB Micro HR@10 (Full)** | 17.97% | 14.60% | **14.60%** | Session reset truncated multi-day visitor history. |
| **Cold LinUCB Macro HR@10 (Full)** | 6.73% | 6.40% | **6.40%** | Expanded cold slice re-weighted static norm ranking. |
| **Cold LinUCB Micro HR@10 (Full)** | 8.39% | 7.61% | **7.61%** | Shift in event-level weighting under session resets. |
| **Popularity Baseline Macro HR@10** | 5.76% | 5.76% | **5.76%** | **Invariant**: Popularity ignores visitor session/history state. |
| **Popularity Baseline Micro HR@10** | 5.92% | 5.92% | **5.92%** | **Invariant**: Identical static ranking. |
| **Warm LinUCB Macro HR@10 (50 Cands)** | 22.84% | 20.45% | **20.45%** | Session reset reduced repeat candidate identification. |
| **Cold Context Slice Count ($n$)** | 194,155 | 226,410 | **226,410** | 194,155 unique visitors vs 226,410 unique 30-min sessions. |
| **Light Context Slice Count ($n$)** | 97,672 | 85,970 | **85,970** | Truncation of multi-day histories. |
| **Heavy Context Slice Count ($n$)** | 43,785 | 23,232 | **23,232** | High-depth histories partitioned by session reset. |

### A3. Exact Split Boundaries Verification
In the Phase 10A report narrative, a typographical error listed $T_{\text{val\_end}} = 1440788640321$. We verified via database queries and code diffs that split boundaries in `historical.db` and evaluation scripts have **never shifted**:
- **Training Period**: $ts < 1438968232766$ (2015-08-07 17:23:52.766 UTC) $\rightarrow 1,880,634$ events ($73.59\%$).
- **Validation Period**: $1438968232766 \le ts < 1440756710277$ (2015-08-28 10:11:50.277 UTC) $\rightarrow 339,438$ events ($13.28\%$).
- **Test Period**: $ts \ge 1440756710277$ (2015-08-28 10:11:50.277 UTC to 2015-09-18 03:00:00 UTC) $\rightarrow 335,612$ events ($13.13\%$).

---

## 3. Part B: Example Construction (Per-Event vs First-Exposure)

The Phase 9B specification called for training and evaluating on **first-exposure** examples (one example per visitor-session-product with highest-stage reward and context strictly before the first event on that product). However, previous phases evaluated per raw event.

In Phase 10B, we built:
1. `data/historical/sufficient_stats_m3_aggregated.npz`: Trained from $1,576,835$ unique `(visitor, session, mapped_product)` positive examples with highest-stage reward ($10.0$ transaction $> 5.0$ addtocart $> 2.0$ view) and $m=3$ pseudo-negatives.
2. Dual evaluation modes: `per_event` ($n = 335,612$) and `first_exposure` ($n = 287,176$).

### Example Construction Comparison Table (Frozen Config: $N_{\text{cap}}=10000, m=3, \lambda=1.0, \alpha=0.5$)

| Evaluation Mode | Test Examples ($n$) | Warm Model Training | Novel Macro HR@10 | Repeat Macro HR@10 | Overall Macro HR@10 | Overall Micro HR@10 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Per-Event** | 335,612 | Per-Event (`stats_m3`) | **5.77%** | **23.41%** | **9.54%** | **14.60%** |
| **Per-Event** | 335,612 | Aggregated (`stats_m3_agg`) | **5.76%** | **23.18%** | **9.49%** | **14.48%** |
| **First-Exposure** | 287,176 | Per-Event (`stats_m3`) | **5.77%** | **21.84%** | **8.05%** | **11.23%** |
| **First-Exposure** | 287,176 | Aggregated (`stats_m3_agg`) | **5.76%** | **21.61%** | **8.01%** | **11.12%** |

*Finding*: Training on aggregated first-exposure examples yields virtually indistinguishable performance from per-event training (Macro HR diff $< 0.05\%$). In first-exposure evaluation, overall Macro HR drops from $9.54\%$ to $8.05\%$ because repeated views of the same product within a session are collapsed into a single evaluation point.

---

## 4. Part C: Novel-First Headline & Proper Baselines

### C1. Primary Headline Table: NOVEL Targets (Full Catalog 262 SKUs)
*Target events where the visitor has never interacted with the target product in history ($n = 263,849$, $78.62\%$ of all test events).*

| Policy | Macro HR@10 (%) | Macro 95% Clustered CI | Micro HR@10 (%) | Micro 95% Clustered CI |
| :--- | :---: | :---: | :---: | :---: |
| **Warm LinUCB (Optimal UCB)** | **5.77%** | [5.64%, 5.90%] | **5.98%** | [5.82%, 6.13%] |
| **Aggregated Warm LinUCB** | **5.76%** | [5.63%, 5.89%] | **5.96%** | [5.81%, 6.11%] |
| **Warm LinTS ($v=0.25$)** | **5.71%** | [5.58%, 5.84%] | **5.91%** | [5.76%, 6.06%] |
| **Popularity Baseline** | **5.76%** | [5.63%, 5.89%] | **5.92%** | [5.77%, 6.07%] |
| **Department Heuristic** | **6.83%** | [6.68%, 6.98%] | **7.12%** | [6.95%, 7.29%] |
| **DepartmentRecency** | **6.83%** | [6.68%, 6.98%] | **7.12%** | [6.95%, 7.29%] |
| **RecentlyViewed** | **5.76%** | [5.63%, 5.89%] | **5.92%** | [5.77%, 6.07%] |
| **SameBrandPopularity** | **5.78%** | [5.65%, 5.91%] | **5.95%** | [5.80%, 6.10%] |
| **Cold LinUCB ($\alpha=0.5$)** | **5.67%** | [5.54%, 5.80%] | **5.85%** | [5.70%, 6.00%] |

### C2. Paired Hypothesis Tests on Novel Targets (2,000 Resamples)

| Comparison (Policy A vs Policy B) | Setting | Macro Difference (%) | Macro 95% CI | Two-Sided $p$-Value | Empirical Outcome |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Warm LinUCB vs Popularity** | Full Catalog | **+0.01%** | [-0.04%, +0.06%] | $p = 0.3840$ | **Not Significant / Negligible** |
| **Warm LinUCB vs Dept Heuristic** | Full Catalog | **-1.06%** | [-1.15%, -0.97%] | $p < 0.0005$ | **Dept Heuristic Wins** |
| **Warm LinUCB vs Cold LinUCB** | Full Catalog | **+0.10%** | [+0.05%, +0.15%] | $p < 0.0005$ | Stat Sig, but Negligible ($< 0.5\%$) |
| **Warm LinTS vs Popularity** | Full Catalog | **-0.05%** | [-0.10%, +0.00%] | $p = 0.0480$ | **Not Significant** |
| **Warm LinUCB vs RecentlyViewed** | Full Catalog | **+0.01%** | [-0.04%, +0.06%] | $p = 0.3840$ | **Tied** (RV falls back to Pop) |
| **Warm LinUCB vs Popularity** | 50 Candidates | **+0.03%** | [-0.10%, +0.16%] | $p = 0.3210$ | **Not Significant / Negligible** |

### C3. Repeat Targets & Proper Recency Baselines ($n = 71,763$, $21.38\%$ of events)

| Policy | Macro HR@10 (%) | Micro HR@10 (%) | Evaluation Takeaway |
| :--- | :---: | :---: | :--- |
| **RecentlyViewed Baseline** | **99.71%** | **99.78%** | **Dominant ceiling on repeat interactions.** |
| **SameBrandPopularity** | **67.05%** | **71.20%** | Strong brand recency heuristic. |
| **Warm LinUCB (Optimal UCB)** | **23.41%** | **46.25%** | **Fails to beat RecentlyViewed by 76.3%**. |
| **DepartmentRecency** | **16.18%** | **22.40%** | Department-level recency boost. |
| **Cold LinUCB ($\alpha=0.5$)** | **9.08%** | **14.07%** | Modest feature norm matching. |
| **Popularity Baseline** | **5.76%** | **5.92%** | Static baseline (unaware of visitor history). |

> [!NOTE]
> **Definitive Finding on Repeat Targets**:
> The contextual bandit may **ONLY** be claimed to assist on repeat interactions if it outperforms a simple `RecentlyViewed` heuristic. Because `RecentlyViewed` achieves **99.71%**, Warm LinUCB ($23.41\%$) is heavily suboptimal. The bandit's higher score on repeats is not an intelligent synthesis—it is merely an imperfect linear shadow of the user's recent clicks that a 5-line LRU cache captures perfectly.

### C4. Candidate Exclusion v2 (Pure Discovery Evaluation)
*Masks ALL products previously engaged by the visitor from the recommendation candidate pool.*

| Policy | Unconstrained Macro HR@10 | Exclusion v1 (In-Cart/Purchased) | Exclusion v2 (All Engaged / Discovery) | Discovery Shift |
| :--- | :---: | :---: | :---: | :--- |
| **Warm LinUCB** | 9.54% | 9.51% | **5.58%** | $-3.96\%$ (repeat advantage vanishes) |
| **Popularity Baseline** | 5.76% | 5.74% | **5.57%** | $-0.19\%$ (slight candidate set reduction) |
| **Department Heuristic** | 7.92% | 7.88% | **6.52%** | Maintained department-level preference |
| **Cold LinUCB** | 6.40% | 6.38% | **5.49%** | $-0.91\%$ |

*Conclusion*: When evaluating genuine discovery (Exclusion v2), Warm LinUCB and Popularity are identical ($5.58\%$ vs $5.57\%$).

---

## 5. Part D: Fixed Prequential Online Replay

### D1. Bug Diagnosis of Phase 10A Anomalous Numbers
Phase 10A reported rolling HR@10 of $66\%$ to $92\%$, which contradicted static offline evaluation ($5\%$ to $15\%$). Audit revealed that the Phase 10A script was never executed to completion during the evaluation run; instead, pre-generated placeholder CSVs with inverted ranking counts (`sc < sc[target]` instead of `sc > sc[target]`) had been exported. Because target scores for cold models were small, an inverted rank check placed targets in the top 10 over $80\%$ of the time.

We added unit test `test_prequential_toy_stream` in `backend/tests/test_phase10b.py` to mathematically verify strictly descending rank maintenance before re-running the full chronological stream.

### D2. Repaired Prequential Replay Trajectories (20,000 Chronological Steps)
*Window Size = 1,000 steps, Checkpoints at steps 100 to 20,000.*

| Test Step | Rolling Cold HR@10 | Rolling Warm HR@10 | Rolling LinTS HR@10 | Cumul. Cold HR@10 | Cumul. Warm HR@10 | Novel-Only Warm Cumul. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.00% | 7.00% | 6.00% | 0.00% | 7.00% | 5.88% |
| **250** | 0.80% | 6.80% | 5.60% | 0.80% | 6.80% | 5.74% |
| **500** | 1.00% | 6.80% | 5.80% | 1.00% | 6.80% | 5.71% |
| **1,000** | 1.70% | 7.00% | 6.10% | 1.70% | 7.00% | 5.79% |
| **2,000** | 2.30% | 4.80% | 4.60% | 2.00% | 5.90% | 5.68% |
| **5,000** | 2.80% | 5.90% | 5.70% | 2.44% | 5.94% | 5.72% |
| **10,000** | 2.90% | 5.80% | 5.60% | 2.68% | 5.86% | 5.75% |
| **20,000** | 3.10% | 5.70% | 5.50% | 2.84% | 5.78% | 5.76% |

### D3. Crossover Analysis & Claim C7 Formal Withdrawal
- **Cold LinUCB does NOT cross Warm LinUCB within 20,000 steps.**
- Cold LinUCB begins at $0.0\%$ and rises steadily to $2.84\%$ cumulative HR@10, but the learning rate is bounded by single-event updates. Warm LinUCB maintains a steady $5.78\%$ to $7.00\%$ trajectory.
- **Formal Status**: **Claim C7 from Phase 10A ("Cold crossover at ~1,000 steps") is formally WITHDRAWN as an artifact of the inverted ranking bug.**

---

## 6. Part E: Valid Leave-Group-Out Ridge Refit Ablations

In Phase 10A, ablations zeroed features at inference time on models trained with them. For example, zeroing brand affinity caused the score to drop to $1.31\%$ (below the $3.8\%$ random chance floor) because large positive training weights ($\theta_6 = 1.28$) were subtracted from the dot product without model rebalancing.

In Phase 10B, all ablations were re-implemented as **leave-group-out Ridge refits** directly from the saved training sufficient statistics ($A_{\text{hist}}, b_{\text{hist}}$):
$$\hat{\theta}_{\text{sub}} = \left(\lambda I + s A_{\text{hist}}[I_{\text{keep}}, I_{\text{keep}}]\right)^{-1} \left(s b_{\text{hist}}[I_{\text{keep}}]\right)$$
Evaluated across the **FULL 335,612 test interactions** with 2,000-resample visitor-clustered CIs.

### Leave-Group-Out Refit Ablation Table (Full Test Split)

| Ablation Configuration | Dimension ($d$) | $\|\hat{\theta}\|$ | Overall Macro HR@10 | Overall Micro HR@10 | Novel Macro HR@10 | Repeat Macro HR@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Features (Full Model)** | 36 | 1.404 | **9.54%** | **14.60%** | **5.77%** | **23.41%** |
| **Ablate: Category Affinities (0-5)** | 30 | 1.382 | **9.51%** | **14.52%** | **5.76%** | **23.29%** |
| **Ablate: Brand Affinity (6)** | 35 | 0.394 | **5.79%** | **6.01%** | **5.76%** | **5.81%** |
| **Ablate: Price & Budget Group (7-10)** | 32 | 1.401 | **9.53%** | **14.58%** | **5.77%** | **23.38%** |
| **Ablate: Category One-Hot (12-17)** | 30 | 1.397 | **9.52%** | **14.55%** | **5.76%** | **23.35%** |
| **Ablate: Price, Rating, Discount (18-20)** | 33 | 1.402 | **9.54%** | **14.59%** | **5.77%** | **23.40%** |
| **Ablate: Product Popularity (21)** | 35 | 1.398 | **9.48%** | **14.49%** | **5.72%** | **23.28%** |
| **Ablate: Brand Hash Buckets (22-29)** | 28 | 1.400 | **9.53%** | **14.58%** | **5.77%** | **23.39%** |
| **Ablate: Session & Intent (11, 31, 32)** | 33 | 1.403 | **9.54%** | **14.60%** | **5.77%** | **23.41%** |
| **Ablate: High Boosts & New Flag (30, 33, 34)** | 33 | 1.404 | **9.54%** | **14.60%** | **5.77%** | **23.41%** |
| **Intercept Only (Baseline)** | 1 | 0.536 | **5.67%** | **5.85%** | **5.67%** | **5.67%** |

### Feature 6 (`affinity_brand`) Analysis
- **Empirical Measurement**:
  - Non-zero on Novel targets: **5.72%** of events (mean value: $0.0465$).
  - Non-zero on Repeat targets: **100.00%** of events (mean value: $0.9177$).
- **Explanation of Weight $\theta_6 = 1.28$**:
  Feature 6 is active on every repeat interaction. The Ridge regression assigns it a dominant positive weight ($\theta_6 = 1.28$). When an item matches a brand previously clicked by the user, its score is boosted by $1.28 \times 0.92 \approx +1.18$.
  Refitting the model without Feature 6 collapses repeat Macro HR from $23.41\%$ to $5.81\%$, but leaves novel target Macro HR completely unaffected ($5.77\% \rightarrow 5.76\%$). This definitively confirms that the warm-start model's learned weights are almost entirely dedicated to repeat brand matching.

---

## 7. Part F: Documentation Consistency & Empirical Hypotheses

### F1. Canonical Feature Table
The feature table in the Phase 10A report listed outdated feature descriptions (e.g. index 11 as "cart depth", 31 as "session velocity"). We regenerated the canonical specification directly from `backend/app/services/feature_engineering.py`:
- `FEATURE_SPEC_HASH = "75f60ed70b2a"` (verified).
- Dimension $d = 36$.
- Index 7: User average historical price level (log-scaled).
- Index 8: Session budget cap (log-scaled).
- Index 9: Price fit score ($1.0 - |p_{\text{log}} - b_{\text{log}}| / 2$).
- Index 10: In-budget binary indicator ($p \le \text{budget}$).
- Index 11: Category search/filter match.
- Index 31: User activity level ($\log(1 + N_{\text{interactions}}) / \log(50)$).
- Index 32: Cart intent match (fraction of active cart in same category).

### F2. Empirical Test of the Cold-Start Shrinkage Hypothesis
In frozen "Cold LinUCB", $\theta = \mathbf{0}$ and $A = I$, so the score is purely the exploration bonus:
$$\text{score}(p) = \alpha \sqrt{\phi(u, p)^T I \phi(u, p)} = \alpha \|\phi(u, p)\|_2$$
We empirically tested whether $\alpha \|\phi\|_2$ acts as an implicit popularity prior:
- **Spearman Rank Correlation** between product feature norm $\|\phi\|_2$ and catalog popularity prior:
  $$r = 0.7126 \quad (p = 6.59 \times 10^{-42})$$
- **Conclusion**: The hypothesis is **CONFIRMED**. Products with higher catalog popularity possess higher ratings, larger discounts, and non-zero feature activations that systematically increase $\|\phi\|_2$. Frozen "Cold LinUCB" is not an online learner; it is an implicit, static feature-norm proxy for item popularity.

### F3. Train-Only Mapping Comparison vs Test Popularity Ceiling

| Mapping Method | Mapping Version | Test Ceiling HR@10 | Popularity Baseline HR@10 | Warm LinUCB HR@10 |
| :--- | :---: | :---: | :---: | :---: |
| **Popularity Balanced** | `v1.0_train` | **22.58%** | **5.76%** | **9.54%** |
| **Co-occurrence Cluster** | `v1.0_train` | **24.12%** | **5.12%** | **8.84%** |
| **Category Rank** | `v1.0_train` | **21.90%** | **4.98%** | **8.42%** |

All three train-only mapping methods perform within expected theoretical bounds well below the test popularity ceiling.

---

## 8. Part G: Limitations & Final Freeze Declaration

### Formal Limitations
1. **Proxy Mapping Aliasing**: 124,882 Retailrocket item IDs are mapped onto 262 active SmartCart catalog SKUs. High-volume items alias multiple distinct behaviors into single products.
2. **Positive-Only Logged Observational Feedback**: Retailrocket logs only views, adds-to-cart, and transactions. Unchosen candidates were never shown to users in the logged history; logged bandit evaluation cannot verify whether recommended items would have generated clicks if actually displayed.
3. **Repeat Bias Dominance**: $21.38\%$ of test events are repeat engagements. The bandit's aggregate metrics are inflated by repeat interactions that a simple recency cache handles significantly better.
4. **Cold Start Ubiquity**: $67.46\%$ of test interactions ($n = 226,410$) take place in cold contexts where no visitor history exists, collapsing contextual models to static popularity/norm rankings.
5. **Absence of Counterfactual / Causal Validity**: No offline evaluation on logged positive interactions can provide causal guarantees. Real-world validation requires live online A/B testing with randomized traffic.

### Final Freeze Declaration
With the delivery of this report, the single master results file (`experiments/results/phase10b_master_results.json`), associated publication plots, and passing test suite (`backend/tests/test_phase10b.py`), the offline Retailrocket evaluation for SmartCart AI is **PERMANENTLY FROZEN**. No further offline evaluation phases or model retuning will be performed.

