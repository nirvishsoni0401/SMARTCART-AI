# SmartCart AI — Phase 12A-2 Scientific Benchmark Report: Hardened Parity, Determinism & Two-Stage Hybrids

> [!IMPORTANT]
> **Epistemic Scope & Synthetic Assumptions**:
> All results describe recommender model behaviour **strictly under the calibrated position-biased simulator's stated synthetic assumptions** (`experiments/sim_v2/`).
> Evaluated over **20,000 rounds** across **30 evaluation seeds** (paired Common Random Numbers).
> They do **not** establish causal customer impact in the wild and must **never** be cited as evidence of production customer behaviour without an online A/B test.
> **Superseded Status**: All Phase 11 and Phase 12A numbers are permanently superseded due to mixed-vintage caching and non-deterministic tie-breaking. Phase 12A-2 is the sole authoritative ground truth.

---

## 1. Executive Summary & Headline Benchmark
- **Evaluation Scenario**: **Abrupt Preference Drift** at midpoint ($t=10,000$), **Medium Difficulty** (Gaussian logit noise $\\sigma = 0.40$), Horizon $T = 20,000$ rounds, 30 independent evaluation seeds.
- **Tuned Hyper-parameters**: $\\alpha^* = 0.5$, $\\gamma^* = 0.9995$ (tuned on separate tuning seeds 101-105).
- **A2 Identity Resolved**: Discounted LinUCB with $\\gamma = 1.0$ matches Cold LinUCB identically to $10^{-16}$ floating precision bit-for-bit across all 30 evaluation seeds (2,942.0 vs 2,942.0). Root causes (process-salted `hash()` on brand and unstable sorting ties) have been permanently resolved.
- **Two-Stage Hybrids Outperform the Department Heuristic**:
  - `Dept-Restricted LinUCB (Cold)` achieves **1,429.6** cumulative regret, outperforming the Department Heuristic (**2,225.6**) by **796.1 points** (Holm-corrected $p < 0.001$).
  - `Dept-Restricted LinUCB (Discounted)` achieves **1,438.3**, achieving lower regret than the heuristic by **787.3 points**.
  - `Residual LinUCB` achieves **1,602.1**, achieving lower regret than the heuristic by **623.6 points**.
- **Trajectory Oracle Expected Reward Reconciliation**:
  For every policy $\\pi$, step regret is $R^*(u_t, t) - R(u_t, a_t)$. Summing over all $T$ rounds yields the exact mathematical identity:
  $$\\text{Cumulative Reward}(\\pi) + \\text{Cumulative Regret}(\\pi) \\equiv R^*_{\\text{trajectory}}(\\pi) = \\sum_{t=1}^T R^*(u_t, t)$$
  The trajectory oracle varies from **7,588.9** (Random minimum) to **8,132.7** (Oracle) because the repeat bonus accumulates along active served paths.
- **Parity Test Scope**: The parity test covers policy scoring logic with identical inputs, not the live feature extraction pipeline.

### Headline Results (All 26 Evaluated Policies):
| Policy | Cumulative Regret (Mean [95% CI]) | Traj. Oracle Reward | Regret Share (%) | Slate CTR (%) | Cart Rate (%) | Conv. Rate (%) | Neg. Feed. (%) | Early Reg. (0-1k) | Post-Drift Reg. | Cumulative Reward |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle** | 0.0 [0.0, 0.0] | 8,132.7 | 0.0% | 17.39% | 2.85% | 33.73% | 3.98% | 0.0 | 0.0 | 8,132.7 |
| **Discounted LinUCB (gamma=0.99)** | 3,881.0 [3,854.0, 3,910.2] | 7,809.2 | 49.7% | 8.95% | 2.49% | 32.89% | 3.94% | 219.4 | 349.2 | 3,928.2 |
| **Discounted LinUCB (gamma=0.995)** | 3,360.8 [3,325.1, 3,397.8] | 7,857.9 | 42.8% | 10.16% | 2.67% | 32.10% | 3.87% | 204.6 | 326.2 | 4,497.0 |
| **Discounted LinUCB (gamma=0.998)** | 2,784.8 [2,736.8, 2,829.2] | 7,894.1 | 35.3% | 11.31% | 2.73% | 32.53% | 3.75% | 186.5 | 291.5 | 5,109.3 |
| **Discounted LinUCB (gamma=0.999)** | 2,586.1 [2,521.9, 2,651.4] | 7,902.1 | 32.7% | 11.77% | 2.76% | 32.42% | 3.89% | 189.0 | 295.4 | 5,316.0 |
| **Discounted LinUCB (gamma=0.9995)** | 2,534.6 [2,451.1, 2,617.8] | 7,905.4 | 32.1% | 11.91% | 2.63% | 32.08% | 3.91% | 175.5 | 312.8 | 5,370.7 |
| **Discounted LinUCB (gamma=0.9999)** | 2,853.4 [2,766.2, 2,939.3] | 7,871.7 | 36.3% | 11.17% | 2.69% | 30.72% | 3.83% | 186.8 | 326.3 | 5,018.4 |
| **Discounted LinUCB (gamma=1.0)** | 2,942.0 [2,858.1, 3,031.7] | 7,855.4 | 37.5% | 10.93% | 2.66% | 34.42% | 3.86% | 195.2 | 326.5 | 4,913.4 |
| **Cold LinUCB** | 2,942.0 [2,858.1, 3,031.7] | 7,855.4 | 37.5% | 10.93% | 2.66% | 34.42% | 3.86% | 195.2 | 326.5 | 4,913.4 |
| **Warm LinUCB** | 3,381.5 [3,339.7, 3,425.6] | 7,875.7 | 42.9% | 10.08% | 2.57% | 34.50% | 3.77% | 156.2 | 310.9 | 4,494.1 |
| **Cold LinTS** | 2,893.9 [2,764.8, 3,109.3] | 7,870.7 | 36.8% | 11.06% | 2.61% | 33.54% | 3.80% | 227.4 | 327.8 | 4,976.8 |
| **Warm LinTS** | 3,339.5 [3,311.3, 3,368.0] | 7,888.1 | 42.3% | 10.21% | 2.70% | 33.57% | 3.75% | 147.7 | 328.2 | 4,548.6 |
| **Recently Viewed + Popularity** | 4,925.1 [4,907.7, 4,947.0] | 7,596.6 | 64.8% | 6.21% | 2.33% | 37.34% | 3.81% | 316.0 | 356.9 | 2,671.5 |
| **Department Heuristic** | 2,225.6 [2,213.5, 2,237.7] | 7,831.5 | 28.4% | 12.55% | 2.78% | 33.44% | 3.78% | 137.7 | 183.7 | 5,605.9 |
| **Department Heuristic (With Diversity)** | 2,489.6 [2,475.9, 2,504.1] | 7,799.8 | 31.9% | 12.03% | 2.71% | 33.64% | 3.77% | 158.9 | 191.5 | 5,310.2 |
| **Dept-Prior LinUCB (w=0.5)** | 2,837.7 [2,747.0, 2,925.3] | 7,885.8 | 36.0% | 11.24% | 2.77% | 32.48% | 3.78% | 164.5 | 326.3 | 5,048.1 |
| **Slate Mixing (k=3 Dept + LinUCB)** | 2,683.5 [2,644.8, 2,721.7] | 7,860.9 | 34.1% | 11.63% | 2.53% | 34.69% | 3.88% | 165.9 | 239.4 | 5,177.4 |
| **Slate Mixing (k=5 Dept + LinUCB)** | 2,631.9 [2,610.2, 2,653.7] | 7,834.1 | 33.6% | 11.76% | 2.67% | 34.21% | 3.84% | 167.4 | 214.0 | 5,202.2 |
| **Residual LinUCB** | 1,602.1 [1,518.0, 1,685.3] | 7,935.6 | 20.2% | 13.97% | 2.64% | 32.02% | 3.84% | 137.5 | 168.8 | 6,333.5 |
| **Dept-Restricted LinUCB (Cold)** | 1,429.6 [1,371.8, 1,498.0] | 7,970.9 | 17.9% | 14.33% | 2.82% | 30.81% | 3.88% | 103.5 | 165.5 | 6,541.3 |
| **Dept-Restricted LinUCB (Discounted)** | 1,438.3 [1,398.9, 1,481.7] | 7,975.8 | 18.0% | 14.32% | 2.77% | 32.33% | 3.96% | 101.0 | 164.1 | 6,537.5 |
| **Dept-Restricted LinUCB + Expl (Cold, m=2)** | 1,771.6 [1,709.8, 1,840.0] | 7,957.9 | 22.3% | 13.71% | 2.81% | 32.50% | 3.84% | 120.6 | 170.5 | 6,186.4 |
| **Dept-Restricted LinUCB + Expl (Discounted, m=2)** | 1,877.8 [1,852.7, 1,906.4] | 7,966.3 | 23.6% | 13.50% | 2.79% | 34.31% | 3.82% | 121.8 | 173.0 | 6,088.4 |
| **Epsilon-Greedy** | 2,963.6 [2,905.0, 3,029.0] | 7,867.1 | 37.7% | 10.89% | 2.64% | 31.38% | 3.93% | 198.0 | 331.7 | 4,903.5 |
| **Popularity** | 4,989.5 [4,971.7, 5,012.4] | 7,598.1 | 65.7% | 6.07% | 2.44% | 35.66% | 3.93% | 318.7 | 360.7 | 2,608.7 |
| **Random** | 5,527.8 [5,508.0, 5,548.2] | 7,588.9 | 72.8% | 4.97% | 2.40% | 30.69% | 3.83% | 350.6 | 398.3 | 2,061.1 |


---

## 2. Dedicated Discounting Test Suite (Part A1 & A2)
Paired Holm-corrected tests comparing Discounted LinUCB for the pre-specified tuned $\\gamma^* = 0.9995$ against $\\gamma = 1.0$ of the same class and against Cold LinUCB:
| Comparison | Scenario | Horizon Window | Mean Difference Delta | 95% Bootstrap CI | Cohen's d | Raw p-value | Holm Adj. p-value | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Discounted (gamma*=0.9995) vs Discounted (gamma=1.0) | no_drift | Full Horizon [0, 20000) | -131.9 | [-269.8, 2.3] | -0.333 | 0.0549 | 0.8238 | Interval spans zero / No significant difference |
| Discounted (gamma*=0.9995) vs Cold LinUCB | no_drift | Full Horizon [0, 20000) | -131.9 | [-269.8, 2.3] | -0.333 | 0.0549 | 0.8238 | Interval spans zero / No significant difference |
| Discounted (gamma=1.0) vs Cold LinUCB | no_drift | Full Horizon [0, 20000) | 0.0 | [0.0, 0.0] | 0.000 | 1.0000 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=0.999) vs Cold LinUCB | no_drift | Full Horizon [0, 20000) | -12.2 | [-192.4, 130.1] | -0.026 | 0.5978 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma*=0.9995) vs Discounted (gamma=1.0) | abrupt_drift | Full Horizon [0, 20000) | -407.4 | [-502.1, -311.1] | -1.491 | 1.3039e-07 | 2.4773e-06 | Discounted LinUCB (gamma=0.9995) beats Discounted LinUCB (gamma=1.0) (statistically significant) |
| Discounted (gamma*=0.9995) vs Discounted (gamma=1.0) | abrupt_drift | Post-Drift 1,000 Rounds [10000, 11000) | -3.1 | [-6.9, 0.7] | -0.284 | 0.1142 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma*=0.9995) vs Discounted (gamma=1.0) | abrupt_drift | Post-Drift Second Half [10000, 20000) | -306.1 | [-354.5, -256.0] | -2.098 | 1.8626e-08 | 4.4703e-07 | Discounted LinUCB (gamma=0.9995) beats Discounted LinUCB (gamma=1.0) (statistically significant) |
| Discounted (gamma*=0.9995) vs Cold LinUCB | abrupt_drift | Full Horizon [0, 20000) | -407.4 | [-502.1, -311.1] | -1.491 | 1.3039e-07 | 2.4773e-06 | Discounted LinUCB (gamma=0.9995) beats Cold LinUCB (statistically significant) |
| Discounted (gamma*=0.9995) vs Cold LinUCB | abrupt_drift | Post-Drift 1,000 Rounds [10000, 11000) | -3.1 | [-6.9, 0.7] | -0.284 | 0.1142 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma*=0.9995) vs Cold LinUCB | abrupt_drift | Post-Drift Second Half [10000, 20000) | -306.1 | [-354.5, -256.0] | -2.098 | 1.8626e-08 | 4.4703e-07 | Discounted LinUCB (gamma=0.9995) beats Cold LinUCB (statistically significant) |
| Discounted (gamma=1.0) vs Cold LinUCB | abrupt_drift | Full Horizon [0, 20000) | 0.0 | [0.0, 0.0] | 0.000 | 1.0000 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=1.0) vs Cold LinUCB | abrupt_drift | Post-Drift 1,000 Rounds [10000, 11000) | 0.0 | [0.0, 0.0] | 0.000 | 1.0000 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=1.0) vs Cold LinUCB | abrupt_drift | Post-Drift Second Half [10000, 20000) | 0.0 | [0.0, 0.0] | 0.000 | 1.0000 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=0.999) vs Cold LinUCB | abrupt_drift | Full Horizon [0, 20000) | -355.9 | [-464.0, -259.3] | -1.236 | 2.6077e-08 | 5.7369e-07 | Discounted LinUCB (gamma=0.999) beats Cold LinUCB (statistically significant) |
| Discounted (gamma=0.999) vs Cold LinUCB | abrupt_drift | Post-Drift 1,000 Rounds [10000, 11000) | -8.9 | [-13.9, -4.5] | -0.654 | 0.0017 | 0.0275 | Discounted LinUCB (gamma=0.999) beats Cold LinUCB (statistically significant) |
| Discounted (gamma=0.999) vs Cold LinUCB | abrupt_drift | Post-Drift Second Half [10000, 20000) | -318.1 | [-359.6, -275.7] | -2.479 | 1.8626e-09 | 5.2154e-08 | Discounted LinUCB (gamma=0.999) beats Cold LinUCB (statistically significant) |
| Discounted (gamma*=0.9995) vs Discounted (gamma=1.0) | gradual_drift | Full Horizon [0, 20000) | -333.8 | [-407.8, -261.4] | -1.601 | 4.6566e-08 | 9.7789e-07 | Discounted LinUCB (gamma=0.9995) beats Discounted LinUCB (gamma=1.0) (statistically significant) |
| Discounted (gamma*=0.9995) vs Discounted (gamma=1.0) | gradual_drift | Post-Drift 1,000 Rounds [10000, 11000) | -0.5 | [-2.8, 2.0] | -0.077 | 0.7766 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma*=0.9995) vs Discounted (gamma=1.0) | gradual_drift | Post-Drift Second Half [10000, 20000) | -250.9 | [-285.5, -215.5] | -2.495 | 5.5879e-09 | 1.5087e-07 | Discounted LinUCB (gamma=0.9995) beats Discounted LinUCB (gamma=1.0) (statistically significant) |
| Discounted (gamma*=0.9995) vs Cold LinUCB | gradual_drift | Full Horizon [0, 20000) | -333.8 | [-407.8, -261.4] | -1.601 | 4.6566e-08 | 9.7789e-07 | Discounted LinUCB (gamma=0.9995) beats Cold LinUCB (statistically significant) |
| Discounted (gamma*=0.9995) vs Cold LinUCB | gradual_drift | Post-Drift 1,000 Rounds [10000, 11000) | -0.5 | [-2.8, 2.0] | -0.077 | 0.7766 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma*=0.9995) vs Cold LinUCB | gradual_drift | Post-Drift Second Half [10000, 20000) | -250.9 | [-285.5, -215.5] | -2.495 | 5.5879e-09 | 1.5087e-07 | Discounted LinUCB (gamma=0.9995) beats Cold LinUCB (statistically significant) |
| Discounted (gamma=1.0) vs Cold LinUCB | gradual_drift | Full Horizon [0, 20000) | 0.0 | [0.0, 0.0] | 0.000 | 1.0000 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=1.0) vs Cold LinUCB | gradual_drift | Post-Drift 1,000 Rounds [10000, 11000) | 0.0 | [0.0, 0.0] | 0.000 | 1.0000 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=1.0) vs Cold LinUCB | gradual_drift | Post-Drift Second Half [10000, 20000) | 0.0 | [0.0, 0.0] | 0.000 | 1.0000 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=0.999) vs Cold LinUCB | gradual_drift | Full Horizon [0, 20000) | -231.9 | [-330.6, -136.8] | -0.867 | 6.2866e-05 | 0.0011 | Discounted LinUCB (gamma=0.999) beats Cold LinUCB (statistically significant) |
| Discounted (gamma=0.999) vs Cold LinUCB | gradual_drift | Post-Drift 1,000 Rounds [10000, 11000) | 1.4 | [-2.3, 5.2] | 0.133 | 0.2894 | 1.0000 | Interval spans zero / No significant difference |
| Discounted (gamma=0.999) vs Cold LinUCB | gradual_drift | Post-Drift Second Half [10000, 20000) | -212.5 | [-260.2, -166.8] | -1.624 | 9.3132e-09 | 2.3283e-07 | Discounted LinUCB (gamma=0.999) beats Cold LinUCB (statistically significant) |


- **A2 Bit-for-Bit Identity**: Across all 30 evaluation seeds, Discounted LinUCB ($\\gamma = 1.0$) and Cold LinUCB produce identically matching slates and cumulative regret ($2,942.0$ vs $2,942.0$, mean delta $= 0.0000000000000000$).
- **No-Drift Constraint**: Per specification, post-drift windows are omitted for `no_drift` as no preference drift occurs.

---

## 3. 7-Value Gamma Grid Evaluation Results (Part C1 / A3)
Candidate gammas evaluated on separate tuning seeds (101-105) for parameter selection and on 30 evaluation seeds (1-30) for unbiased evaluation:
| Candidate Gamma | Tuning Regret (Seeds 101-105) | Abrupt Drift Regret (30 Seeds) | Selected Status |
| :--- | :--- | :--- | :--- |
| **gamma=0.99** | 3,918.5 | 3,881.0 | Grid Candidate |
| **gamma=0.995** | 3,420.8 | 3,360.8 | Grid Candidate |
| **gamma=0.998** | 2,906.0 | 2,784.8 | Grid Candidate |
| **gamma=0.999** | 2,618.6 | 2,586.1 | Grid Candidate |
| **gamma=0.9995** | 2,570.9 | 2,534.6 | **gamma* (Tuned Best)** |
| **gamma=0.9999** | 2,896.2 | 2,853.4 | Grid Candidate |
| **gamma=1.0** | 2,723.9 | 2,942.0 | Grid Candidate |


- **Selection Curve**: On tuning seeds 101-105, $\\gamma = 0.9995$ achieves lowest regret (2,570.9) across the 7-value candidate grid, yielding the lowest tuning regret under abrupt drift.

---

## 4. Two-Stage Hybrids & Candidate Restriction (Part B)
Evaluating candidate restriction (Stage 1 heuristic department selection + Stage 2 LinUCB ranking) and residual learning:
| Policy | Cumulative Regret (Mean [95% CI]) | Cumulative Reward | Slate CTR (%) | Cart Rate (%) | Conv. Rate (%) | Neg. Feed. (%) | Early Reg. (0-1k) | Post-Drift Reg. | Paired vs Dept Heuristic (Delta [95% CI], p_holm) | Paired vs Cold LinUCB (Delta [95% CI], p_holm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Department Heuristic** | 2,225.6 [2,213.5, 2,237.7] | 5,605.9 | 12.55% | 2.78% | 33.44% | 3.78% | 137.7 | 183.7 | Baseline (0.0) | -716.3 [-806.9, -630.9], < 0.001 |
| **Cold LinUCB** | 2,942.0 [2,858.1, 3,031.7] | 4,913.4 | 10.93% | 2.66% | 34.42% | 3.86% | 195.2 | 326.5 | +716.3 [+630.9, +806.9], < 0.001 | Baseline (0.0) |
| **Discounted LinUCB (gamma=0.9995)** | 2,534.6 [2,451.1, 2,617.8] | 5,370.7 | 11.91% | 2.63% | 32.08% | 3.91% | 175.5 | 312.8 | — | -407.4 [-502.1, -311.1], < 0.001 |
| **Dept-Restricted LinUCB (Cold)** | 1,429.6 [1,371.8, 1,498.0] | 6,541.3 | 14.33% | 2.82% | 30.81% | 3.88% | 103.5 | 165.5 | -796.1 [-856.5, -729.9], < 0.001 | -1,512.4 [-1,613.5, -1,412.0], < 0.001 |
| **Dept-Restricted LinUCB (Discounted)** | 1,438.3 [1,398.9, 1,481.7] | 6,537.5 | 14.32% | 2.77% | 32.33% | 3.96% | 101.0 | 164.1 | -787.3 [-826.1, -747.7], < 0.001 | -1,503.7 [-1,596.5, -1,413.9], < 0.001 |
| **Dept-Restricted LinUCB + Expl (Cold, m=2)** | 1,771.6 [1,709.8, 1,840.0] | 6,186.4 | 13.71% | 2.81% | 32.50% | 3.84% | 120.6 | 170.5 | -454.1 [-514.6, -388.8], < 0.001 | -1,170.4 [-1,281.5, -1,063.7], < 0.001 |
| **Dept-Restricted LinUCB + Expl (Discounted, m=2)** | 1,877.8 [1,852.7, 1,906.4] | 6,088.4 | 13.50% | 2.79% | 34.31% | 3.82% | 121.8 | 173.0 | -347.8 [-379.0, -314.5], < 0.001 | -1,064.2 [-1,169.4, -963.6], < 0.001 |
| **Residual LinUCB** | 1,602.1 [1,518.0, 1,685.3] | 6,333.5 | 13.97% | 2.64% | 32.02% | 3.84% | 137.5 | 168.8 | -623.6 [-706.3, -540.1], < 0.001 | -1,339.9 [-1,448.0, -1,237.3], < 0.001 |
| **Dept-Prior LinUCB (w=0.5)** | 2,837.7 [2,747.0, 2,925.3] | 5,048.1 | 11.24% | 2.77% | 32.48% | 3.78% | 164.5 | 326.3 | +612.1 [+521.4, +699.0], < 0.001 | -104.3 [-222.0, +2.4], p=0.2806 |
| **Slate Mixing (k=3 Dept + LinUCB)** | 2,683.5 [2,644.8, 2,721.7] | 5,177.4 | 11.63% | 2.53% | 34.69% | 3.88% | 165.9 | 239.4 | +457.9 [+418.7, +496.6], < 0.001 | -258.5 [-353.6, -167.9], < 0.001 |
| **Slate Mixing (k=5 Dept + LinUCB)** | 2,631.9 [2,610.2, 2,653.7] | 5,202.2 | 11.76% | 2.67% | 34.21% | 3.84% | 167.4 | 214.0 | +406.3 [+385.1, +427.9], < 0.001 | -310.1 [-405.7, -217.6], < 0.001 |


### Scientific Findings on Hybrids:
1. **Candidate Restriction Beats Both Baselines**: Restricting Cold LinUCB to rank products within the user's active target department (`Dept-Restricted LinUCB (Cold)`) eliminates catalog-wide exploration penalties while retaining the capacity to rank items by rating, price, and repeat bonus. This drops regret from $2,942.0$ (Cold LinUCB) and $2,225.6$ (Department Heuristic) down to **1,429.6** (a **796.1 point improvement** over the heuristic, $p < 0.001$).
2. **Residual LinUCB Rescaled**: The heuristic offset ($h_{\text{offset}} = 0.35$) was derived from first principles to match the simulator's latent expected utility scale ($\mathbb{E}[V] \approx 0.1..0.5$) rather than fit on evaluation seeds. Checked across tuning seeds 101-105 only, this offset enables Residual LinUCB to achieve **1,602.1**, resolving the Phase 12A implementation defect. Because the offset is informed by simulator latent utility scales, Residual LinUCB is labelled environment-informed and excluded from the core supported claims table.
3. **Exploration Quota Trade-Off**: Allocating 2 exploratory slots across the catalog (`m_explore=2`) yields regret **1,771.6**, which outperforms the Department Heuristic by 454.1 points while providing active cross-department discovery.

---

## 5. Confounded Scenarios Separation (Part C2 / D)
Separating preference drift ($t=10,000$) from the introduction of 12 cold-start items (2 per category with highest IDs, $t=10,000$):
| Policy | Drift Only Regret | New Items Only Regret | Both (Primary) Regret | Traj. Oracle Reward | Regret Share (%) | New Item Exposure (%) | New Item Reward | New Item Regret |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle** | 0.0 | 0.0 | 0.0 | 8,132.7 | 0.0% | 3.995% | 285.2 | 0.0 |
| **Discounted LinUCB (gamma=0.99)** | 3,822.3 | 3,344.1 | 3,881.0 | 7,809.2 | 49.7% | 4.410% | 120.6 | 225.6 |
| **Discounted LinUCB (gamma=0.995)** | 3,360.9 | 2,920.6 | 3,360.8 | 7,857.9 | 42.8% | 4.862% | 163.3 | 195.5 |
| **Discounted LinUCB (gamma=0.998)** | 2,856.2 | 2,438.1 | 2,784.8 | 7,894.1 | 35.3% | 5.538% | 230.1 | 162.4 |
| **Discounted LinUCB (gamma=0.999)** | 2,623.8 | 2,269.5 | 2,586.1 | 7,902.1 | 32.7% | 6.274% | 279.5 | 143.5 |
| **Discounted LinUCB (gamma=0.9995)** | 2,563.8 | 2,145.2 | 2,534.6 | 7,905.4 | 32.1% | 6.883% | 285.8 | 159.7 |
| **Discounted LinUCB (gamma=0.9999)** | 2,928.4 | 2,440.4 | 2,853.4 | 7,871.7 | 36.3% | 7.069% | 222.3 | 196.1 |
| **Discounted LinUCB (gamma=1.0)** | 2,898.1 | 2,446.1 | 2,942.0 | 7,855.4 | 37.5% | 7.676% | 215.7 | 201.5 |
| **Cold LinUCB** | 2,898.1 | 2,446.1 | 2,942.0 | 7,855.4 | 37.5% | 7.676% | 215.7 | 201.5 |
| **Warm LinUCB** | 3,424.4 | 2,929.2 | 3,381.5 | 7,875.7 | 42.9% | 6.568% | 205.7 | 212.5 |
| **Cold LinTS** | 2,874.1 | 2,411.0 | 2,893.9 | 7,870.7 | 36.8% | 7.020% | 199.9 | 200.6 |
| **Warm LinTS** | 3,390.0 | 2,830.1 | 3,339.5 | 7,888.1 | 42.3% | 4.426% | 141.2 | 224.4 |
| **Recently Viewed + Popularity** | 4,944.0 | 4,137.6 | 4,925.1 | 7,596.6 | 64.8% | 0.000% | 0.0 | 280.2 |
| **Department Heuristic** | 2,251.7 | 1,934.4 | 2,225.6 | 7,831.5 | 28.4% | 8.643% | 354.5 | 64.9 |
| **Department Heuristic (With Diversity)** | 2,479.7 | 2,150.0 | 2,489.6 | 7,799.8 | 31.9% | 6.675% | 283.0 | 63.5 |
| **Dept-Prior LinUCB (w=0.5)** | 2,748.3 | 2,324.0 | 2,837.7 | 7,885.8 | 36.0% | 6.985% | 192.3 | 209.4 |
| **Slate Mixing (k=3 Dept + LinUCB)** | 2,673.2 | 2,214.7 | 2,683.5 | 7,860.9 | 34.1% | 7.054% | 235.8 | 148.7 |
| **Slate Mixing (k=5 Dept + LinUCB)** | 2,590.2 | 2,234.9 | 2,631.9 | 7,834.1 | 33.6% | 9.006% | 358.0 | 45.3 |
| **Residual LinUCB** | 1,689.3 | 1,568.2 | 1,602.1 | 7,935.6 | 20.2% | 6.957% | 310.4 | 120.4 |
| **Dept-Restricted LinUCB (Cold)** | 1,396.2 | 1,323.7 | 1,429.6 | 7,970.9 | 17.9% | 7.547% | 365.9 | 91.2 |
| **Dept-Restricted LinUCB (Discounted)** | 1,353.3 | 1,306.5 | 1,438.3 | 7,975.8 | 18.0% | 7.697% | 373.6 | 96.9 |
| **Dept-Restricted LinUCB + Expl (Cold, m=2)** | 1,757.7 | 1,493.9 | 1,771.6 | 7,957.9 | 22.3% | 7.820% | 325.2 | 112.4 |
| **Dept-Restricted LinUCB + Expl (Discounted, m=2)** | 1,851.2 | 1,677.5 | 1,877.8 | 7,966.3 | 23.6% | 6.062% | 283.6 | 110.8 |
| **Epsilon-Greedy** | 2,892.8 | 2,485.9 | 2,963.6 | 7,867.1 | 37.7% | 5.866% | 167.4 | 218.0 |
| **Popularity** | 5,007.4 | 4,193.2 | 4,989.5 | 7,598.1 | 65.7% | 0.000% | 0.0 | 280.2 |
| **Random** | 5,545.9 | 4,678.3 | 5,527.8 | 7,588.9 | 72.8% | 4.584% | 91.2 | 242.1 |


- **Standardized Exposure Definition**: New-item exposure is consistently measured over the second half ($t \\in [10000, 20000)$, 100,000 slots). Oracle new-item exposure is **3.995%** and negative feedback is **3.984%**, reported to 3 decimal places.

---

## 6. Misspecification Robustness Suite & Rewritten Verdict (Part D)
Evaluating policy stability when the true environment violates LinUCB's 36-dimensional linear assumption:
| Policy | Base Regret | Hidden Style Regret | Nonlinear Regret | High Funnel Regret | Base Reg. Share (%) | Hidden Reg. Share (%) | Nonlin. Reg. Share (%) | Funnel Reg. Share (%) | Base Traj. Oracle | Hidden Traj. Oracle | Nonlin. Traj. Oracle | Funnel Traj. Oracle |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% | 0.0% | 0.0% | 8,132.7 | 16,494.1 | 28,557.3 | 11,033.2 |
| **Discounted LinUCB (gamma=0.99)** | 3,881.0 | 10,337.8 | 6,913.0 | 5,271.0 | 49.7% | 67.3% | 25.0% | 49.8% | 7,809.2 | 15,352.7 | 27,609.8 | 10,591.0 |
| **Discounted LinUCB (gamma=0.995)** | 3,360.8 | 9,716.1 | 5,941.6 | 4,601.0 | 42.8% | 63.1% | 21.4% | 43.1% | 7,857.9 | 15,408.9 | 27,749.7 | 10,672.7 |
| **Discounted LinUCB (gamma=0.998)** | 2,784.8 | 9,039.2 | 6,012.7 | 3,854.7 | 35.3% | 58.5% | 21.8% | 36.0% | 7,894.1 | 15,447.0 | 27,606.7 | 10,704.0 |
| **Discounted LinUCB (gamma=0.999)** | 2,586.1 | 8,794.0 | 6,072.4 | 3,494.2 | 32.7% | 56.9% | 22.2% | 32.6% | 7,902.1 | 15,460.4 | 27,521.2 | 10,731.5 |
| **Discounted LinUCB (gamma=0.9995)** | 2,534.6 | 8,800.6 | 7,195.6 | 3,612.6 | 32.1% | 57.0% | 26.6% | 33.8% | 7,905.4 | 15,438.3 | 27,311.5 | 10,700.9 |
| **Discounted LinUCB (gamma=0.9999)** | 2,853.4 | 9,215.8 | 8,207.3 | 3,931.9 | 36.3% | 59.8% | 30.5% | 36.8% | 7,871.7 | 15,422.0 | 27,130.1 | 10,683.9 |
| **Discounted LinUCB (gamma=1.0)** | 2,942.0 | 9,296.1 | 8,253.3 | 4,047.5 | 37.5% | 60.3% | 30.6% | 38.0% | 7,855.4 | 15,416.4 | 27,189.7 | 10,669.4 |
| **Cold LinUCB** | 2,942.0 | 9,296.1 | 8,253.3 | 4,047.5 | 37.5% | 60.3% | 30.6% | 38.0% | 7,855.4 | 15,416.4 | 27,189.7 | 10,669.4 |
| **Warm LinUCB** | 3,381.5 | 9,665.7 | 7,612.3 | 4,324.5 | 42.9% | 62.6% | 27.9% | 40.4% | 7,875.7 | 15,440.3 | 27,256.6 | 10,706.0 |
| **Cold LinTS** | 2,893.9 | 9,231.2 | 7,890.2 | 4,055.0 | 36.8% | 59.8% | 29.3% | 38.1% | 7,870.7 | 15,428.9 | 27,274.5 | 10,672.4 |
| **Warm LinTS** | 3,339.5 | 9,682.5 | 8,355.0 | 4,362.8 | 42.3% | 62.6% | 30.9% | 40.7% | 7,888.1 | 15,458.7 | 27,088.4 | 10,720.5 |
| **Recently Viewed + Popularity** | 4,925.1 | 11,676.7 | 21,817.4 | 6,789.2 | 64.8% | 77.6% | 88.1% | 65.9% | 7,596.6 | 15,047.1 | 24,761.7 | 10,298.8 |
| **Department Heuristic** | 2,225.6 | 8,487.0 | 17,970.0 | 3,114.9 | 28.4% | 55.2% | 71.8% | 29.3% | 7,831.5 | 15,368.2 | 25,010.5 | 10,613.1 |
| **Department Heuristic (With Diversity)** | 2,489.6 | 8,806.3 | 18,371.4 | 3,473.1 | 31.9% | 57.4% | 73.6% | 32.9% | 7,799.8 | 15,332.4 | 24,977.4 | 10,569.7 |
| **Dept-Prior LinUCB (w=0.5)** | 2,837.7 | 9,217.9 | 10,275.4 | 4,003.6 | 36.0% | 59.7% | 39.0% | 37.5% | 7,885.8 | 15,433.3 | 26,649.5 | 10,682.4 |
| **Slate Mixing (k=3 Dept + LinUCB)** | 2,683.5 | 9,096.7 | 14,788.1 | 3,823.6 | 34.1% | 59.1% | 56.5% | 35.9% | 7,860.9 | 15,382.2 | 26,160.2 | 10,651.2 |
| **Slate Mixing (k=5 Dept + LinUCB)** | 2,631.9 | 8,980.5 | 15,001.7 | 3,710.5 | 33.6% | 58.5% | 57.8% | 34.9% | 7,834.1 | 15,361.5 | 25,979.0 | 10,620.1 |
| **Residual LinUCB** | 1,602.1 | 7,678.3 | 11,193.2 | 2,333.0 | 20.2% | 49.5% | 42.6% | 21.8% | 7,935.6 | 15,506.0 | 26,380.7 | 10,740.2 |
| **Dept-Restricted LinUCB (Cold)** | 1,429.6 | 7,472.6 | 13,364.2 | 2,013.5 | 17.9% | 48.0% | 51.6% | 18.6% | 7,970.9 | 15,566.1 | 25,898.1 | 10,805.2 |
| **Dept-Restricted LinUCB (Discounted)** | 1,438.3 | 7,497.5 | 13,253.7 | 2,037.7 | 18.0% | 48.2% | 51.1% | 18.8% | 7,975.8 | 15,562.4 | 25,934.3 | 10,815.4 |
| **Dept-Restricted LinUCB + Expl (Cold, m=2)** | 1,771.6 | 7,799.9 | 12,380.9 | 2,457.8 | 22.3% | 50.2% | 47.5% | 22.8% | 7,957.9 | 15,545.0 | 26,108.1 | 10,795.2 |
| **Dept-Restricted LinUCB + Expl (Discounted, m=2)** | 1,877.8 | 8,006.2 | 12,087.9 | 2,629.7 | 23.6% | 51.5% | 46.2% | 24.4% | 7,966.3 | 15,554.3 | 26,190.4 | 10,793.8 |
| **Epsilon-Greedy** | 2,963.6 | 9,331.5 | 10,038.8 | 4,141.2 | 37.7% | 60.6% | 37.7% | 38.9% | 7,867.1 | 15,404.0 | 26,870.4 | 10,663.2 |
| **Popularity** | 4,989.5 | 11,799.2 | 22,004.7 | 6,877.6 | 65.7% | 78.4% | 88.9% | 66.8% | 7,598.1 | 15,048.8 | 24,762.8 | 10,301.7 |
| **Random** | 5,527.8 | 12,500.4 | 21,947.1 | 7,620.2 | 72.8% | 83.1% | 88.1% | 74.1% | 7,588.9 | 15,044.3 | 24,903.7 | 10,290.2 |


### Analysis of Heuristic Performance Under Model Misspecification:
> [!NOTE]
> **Heuristic Sensitivity to Nonlinear Customer Choice**:
> While the Department Heuristic performs well in the base environment where category affinity dominates, its regret share rises to **71.8%** (17,970.0 cumulative regret) under nonlinear feature interactions (price $\\times$ category offsets and rating $\\times$ discount synergies).
> In contrast, adaptive bandits adjust to interaction effects, achieving **8,253.3** (Cold LinUCB, 30.6% regret share), **7,195.6** (Discounted LinUCB, 26.6%), and **13,364.2** (Dept-Restricted LinUCB, 51.6%).
> Under nonlinear customer choice, adaptive learning policies accumulate lower regret than the static heuristic by **9,716.7 points**.

---

## 7. Warm-Start Decision Sweep for Live Demo (Part E)
Sweep of historical observation weight $N_{\\text{cap}} \\in \\{0, 500, 2000, 10000\\}$:
| Policy | N_cap | Full Cumulative Regret | Early Regret (Rounds 0 - 1,000) | Post-Drift Regret |
| :--- | :--- | :--- | :--- | :--- |
| Warm LinUCB | Cold Start (N_cap=0) | 2,942.0 [2,858.1, 3,031.7] | 195.2 | 326.5 |
| Warm LinUCB | N_cap=500 | 2,677.3 [2,643.4, 2,711.3] | 170.3 | 317.8 |
| Warm LinUCB | N_cap=2,000 | 2,813.4 [2,782.5, 2,842.2] | 144.0 | 314.3 |
| Warm LinUCB | N_cap=10,000 | 3,381.5 [3,339.7, 3,425.6] | 156.2 | 310.9 |
| Warm LinTS | Cold Start (N_cap=0) | 2,893.9 [2,764.8, 3,109.3] | 227.4 | 327.8 |
| Warm LinTS | N_cap=500 | 2,765.6 [2,730.1, 2,804.2] | 184.8 | 321.9 |
| Warm LinTS | N_cap=2,000 | 2,913.5 [2,883.4, 2,940.6] | 145.7 | 317.3 |
| Warm LinTS | N_cap=10,000 | 3,339.5 [3,311.3, 3,368.0] | 147.7 | 328.2 |


- **Zero-Cap Parity**: $N_{\\text{cap}} = 0$ reproduces Cold LinUCB (2,942.0) and Cold LinTS (2,893.9) identically.

---

## 8. Quarterly Learning Dynamics & Slopes (Part A5)
Per-quarter regret trajectories and ratio to Random baseline:
| Policy | Q1 Regret | Q2 Regret | Q3 Regret | Q4 Regret | Q1/Rand_Q1 | Q2/Rand_Q2 | Q3/Rand_Q3 | Q4/Rand_Q4 | Slope Q2 (/1k) | Slope Q4 (/1k) | Spike Ratio (Q3/Q2) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle** | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 | 0.000 | 0.000 | 0.000 | +0.00 | +0.00 | 0.00x |
| **Discounted LinUCB (gamma=0.99)** | 1,071.8 | 1,079.5 | 864.9 | 864.7 | 0.609 | 0.610 | 0.866 | 0.865 | -0.54 | +0.57 | 0.80x |
| **Discounted LinUCB (gamma=0.995)** | 903.8 | 863.9 | 800.8 | 792.3 | 0.514 | 0.488 | 0.802 | 0.793 | +0.05 | +0.28 | 0.93x |
| **Discounted LinUCB (gamma=0.998)** | 748.8 | 657.1 | 704.9 | 674.0 | 0.426 | 0.371 | 0.706 | 0.674 | +0.23 | -1.29 | 1.08x |
| **Discounted LinUCB (gamma=0.999)** | 709.8 | 585.2 | 675.6 | 615.6 | 0.403 | 0.331 | 0.676 | 0.616 | -0.96 | +0.42 | 1.16x |
| **Discounted LinUCB (gamma=0.9995)** | 667.1 | 564.3 | 711.4 | 591.7 | 0.379 | 0.319 | 0.712 | 0.592 | -1.10 | -0.53 | 1.28x |
| **Discounted LinUCB (gamma=0.9999)** | 711.4 | 581.8 | 806.9 | 753.3 | 0.404 | 0.329 | 0.808 | 0.754 | -2.49 | -1.75 | 1.42x |
| **Discounted LinUCB (gamma=1.0)** | 736.4 | 596.4 | 814.0 | 795.2 | 0.418 | 0.337 | 0.815 | 0.796 | -2.10 | -0.17 | 1.41x |
| **Cold LinUCB** | 736.4 | 596.4 | 814.0 | 795.2 | 0.418 | 0.337 | 0.815 | 0.796 | -2.10 | -0.17 | 1.41x |
| **Warm LinUCB** | 796.2 | 1,017.1 | 797.2 | 771.0 | 0.452 | 0.575 | 0.798 | 0.771 | -20.07 | -1.36 | 0.79x |
| **Cold LinTS** | 745.3 | 538.1 | 816.4 | 794.2 | 0.424 | 0.304 | 0.817 | 0.795 | -1.96 | -0.83 | 1.60x |
| **Warm LinTS** | 771.1 | 955.0 | 820.9 | 792.5 | 0.438 | 0.540 | 0.822 | 0.793 | -6.44 | -1.32 | 0.86x |
| **Recently Viewed + Popularity** | 1,568.8 | 1,565.1 | 896.0 | 895.2 | 0.892 | 0.884 | 0.897 | 0.896 | -0.16 | +0.31 | 0.57x |
| **Department Heuristic** | 695.7 | 702.2 | 439.9 | 387.8 | 0.395 | 0.397 | 0.440 | 0.388 | +0.26 | -1.20 | 0.63x |
| **Department Heuristic (With Diversity)** | 800.5 | 807.4 | 462.7 | 419.0 | 0.455 | 0.456 | 0.463 | 0.419 | +0.25 | -0.99 | 0.57x |
| **Dept-Prior LinUCB (w=0.5)** | 671.4 | 577.9 | 807.6 | 780.8 | 0.382 | 0.327 | 0.808 | 0.781 | -1.21 | -0.48 | 1.45x |
| **Slate Mixing (k=3 Dept + LinUCB)** | 768.0 | 754.1 | 590.9 | 570.5 | 0.436 | 0.426 | 0.592 | 0.571 | +0.60 | -0.38 | 0.79x |
| **Slate Mixing (k=5 Dept + LinUCB)** | 800.3 | 798.4 | 529.1 | 504.1 | 0.455 | 0.451 | 0.530 | 0.504 | +0.43 | -0.32 | 0.66x |
| **Residual LinUCB** | 478.2 | 353.5 | 408.1 | 362.3 | 0.272 | 0.200 | 0.409 | 0.363 | -1.29 | -1.21 | 1.19x |
| **Dept-Restricted LinUCB (Cold)** | 386.2 | 307.8 | 393.6 | 341.9 | 0.219 | 0.174 | 0.394 | 0.342 | -0.77 | -1.15 | 1.32x |
| **Dept-Restricted LinUCB (Discounted)** | 385.5 | 305.2 | 393.8 | 353.8 | 0.219 | 0.172 | 0.394 | 0.354 | -0.87 | -0.49 | 1.31x |
| **Dept-Restricted LinUCB + Expl (Cold, m=2)** | 517.2 | 476.2 | 407.8 | 370.3 | 0.294 | 0.269 | 0.408 | 0.371 | -0.63 | -0.28 | 0.87x |
| **Dept-Restricted LinUCB + Expl (Discounted, m=2)** | 515.2 | 468.2 | 445.1 | 449.4 | 0.293 | 0.265 | 0.446 | 0.450 | +0.32 | -0.55 | 0.96x |
| **Epsilon-Greedy** | 731.1 | 586.9 | 829.4 | 816.2 | 0.415 | 0.332 | 0.830 | 0.817 | -1.27 | +0.14 | 1.43x |
| **Popularity** | 1,589.3 | 1,588.2 | 906.0 | 906.0 | 0.903 | 0.897 | 0.907 | 0.907 | -0.18 | +0.32 | 0.57x |
| **Random** | 1,759.8 | 1,769.8 | 998.9 | 999.4 | 1.000 | 1.000 | 1.000 | 1.000 | -0.07 | +0.49 | 0.56x |


- **Difficulty Shift Across Quarters**: Random baseline regret drops from 1,759.8 in $Q_1$ to 998.9 in $Q_3$, demonstrating that post-drift customer preferences alter baseline environment difficulty. Measuring regret as a ratio to Random normalizes for this shift.

---

## 9. Before / After Comparison Table (Part A1)
Reconciling all numbers that changed across code versions and establishing the authoritative ground truth:
| Policy | Phase 12A Baseline | Phase 12A-2 Authoritative | Status & Authoritative Ground Truth |
| :--- | :--- | :--- | :--- |
| **Department Heuristic** | 2,225.6 | **2,225.6** | Authoritative Ground Truth (Phase 12A-2) |
| **Cold LinUCB** | 2,943.0 | **2,942.0** | Authoritative Ground Truth (Phase 12A-2) |
| **Discounted LinUCB (gamma=0.9995)** | 2,609.0 | **2,534.6** | Authoritative Ground Truth (Phase 12A-2) |
| **Discounted LinUCB (gamma=1.0)** | 2,865.9 | **2,942.0** | Authoritative Ground Truth (Phase 12A-2) |
| **Warm LinUCB** | 3,390.8 | **3,381.5** | Authoritative Ground Truth (Phase 12A-2) |
| **Cold LinTS** | 2,829.4 | **2,893.9** | Authoritative Ground Truth (Phase 12A-2) |
| **Warm LinTS** | 3,424.3 | **3,339.5** | Authoritative Ground Truth (Phase 12A-2) |
| **Slate Mixing (k=5 Dept + LinUCB)** | 2,585.8 | **2,631.9** | Authoritative Ground Truth (Phase 12A-2) |
| **Residual LinUCB** | 3,536.2 | **1,602.1** | Authoritative Ground Truth (Phase 12A-2) |
| **Oracle** | 0.0 | **0.0** | Authoritative Ground Truth (Phase 12A-2) |
| **Random** | 5,527.8 | **5,527.8** | Authoritative Ground Truth (Phase 12A-2) |


All Phase 12A-2 numbers are computed under deterministic cache validation (`config_hash`) with verified cross-process RNG parity.
