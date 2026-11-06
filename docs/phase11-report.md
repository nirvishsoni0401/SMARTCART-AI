# SmartCart AI — Phase 11 Calibrated Simulator v2 Benchmark Report

> [!IMPORTANT]
> **Verified Evaluation Status & Scope Notice**:
> This report evaluates recommendation policies within a calibrated, position-biased synthetic environment
> (`experiments/sim_v2/`). All offline Retailrocket evaluation metrics remain permanently frozen (Phases 9-10).
> Results describe model behaviour **strictly under the simulator's stated assumptions**.
> They do **not** constitute proof of causal production readiness.

## 1. Executive Summary & Headline Results
- **Simulation Horizon**: 20,000 rounds per seed.
- **Evaluation Replications**: 30 independent evaluation seeds with Common Random Numbers (paired design).
- **Catalog & Feature Dimension**: 262 active SKUs across 6 departments, $d=36$ feature vector.
- **Tuned Parameters**: $\alpha^* = 0.25$ (tuned on 5 independent seeds), $\gamma^* = 0.9999$.

### Table 1: Primary Benchmark Headline Comparison (30 Seeds, Abrupt Drift, Medium Difficulty)
| Policy | Cumulative Regret (Mean [95% CI]) | Cumulative Reward | Slate CTR (%) | Cart Rate (%) | Conv Rate (%) | Diversity (Cats) | Coverage (%) | Step Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle** | 159.5 [145.4, 170.1] | 10,628.7 | 20.67% | 2.90% | 33.37% | 1.38 | 95.3% | 0.256 |
| **Discounted LinUCB** | 3,219.7 [2,873.8, 3,796.7] | 6,765.2 | 13.66% | 2.76% | 32.45% | 2.62 | 86.4% | 0.447 |
| **Cold LinUCB** | 3,807.7 [3,161.7, 4,949.3] | 5,938.3 | 12.18% | 2.57% | 33.57% | 2.55 | 79.8% | 0.536 |
| **Cold LinTS** | 3,841.8 [3,250.6, 6,817.5] | 6,030.7 | 12.32% | 2.60% | 33.42% | 2.42 | 96.4% | 2.341 |
| **Warm LinTS** | 4,456.3 [3,846.4, 5,257.2] | 5,395.4 | 11.09% | 2.73% | 33.69% | 2.32 | 90.6% | 2.217 |
| **Warm LinUCB** | 4,342.1 [3,716.2, 5,446.5] | 5,431.0 | 11.24% | 2.57% | 33.96% | 2.44 | 83.5% | 0.550 |
| **Recently Viewed + Popularity** | 4,451.2 [4,357.3, 4,541.0] | 5,471.0 | 11.43% | 2.68% | 32.84% | 1.58 | 99.0% | 0.180 |
| **Epsilon-Greedy** | 3,842.6 [3,461.3, 4,508.7] | 5,849.4 | 11.98% | 2.59% | 32.22% | 2.46 | 100.0% | 0.339 |
| **Department Heuristic** | 3,888.2 [3,795.7, 4,030.5] | 5,855.4 | 12.14% | 2.69% | 32.81% | 1.00 | 24.8% | 0.268 |
| **Popularity** | 6,555.3 [6,396.6, 6,708.5] | 2,982.0 | 6.44% | 2.36% | 32.51% | 1.00 | 3.8% | 0.176 |
| **Random** | 7,203.6 [7,036.9, 7,362.4] | 2,302.2 | 5.14% | 2.29% | 35.67% | 4.35 | 100.0% | 0.075 |

---

## 2. Paired Statistical Hypothesis Testing
All statistical tests use paired differences on identical customer streams with Holm-Bonferroni correction ($m=11$ hypotheses).

### Table 2: Paired Regret Differences vs Cold LinUCB and vs Popularity Baseline
| Policy | Diff vs Cold LinUCB (Delta) | 95% Bootstrap CI | Cohen's d | Wilcoxon p-value | Holm Sig? | Diff vs Popularity | 95% Bootstrap CI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Oracle** | -3,648.2 | [-3,838.7, -3,470.3] | -6.90 | 1.86e-09 | Yes | -6,395.8 | [-6,428.9, -6,363.4] |
| **Discounted LinUCB** | -587.9 | [-769.0, -401.3] | -1.10 | 2.05e-07 | Yes | -3,335.6 | [-3,432.4, -3,210.9] |
| **Cold LinUCB** | +0.0 | [+0.0, +0.0] | +0.00 | 1.00e+00 | No | -2,747.7 | [-2,935.2, -2,549.8] |
| **Cold LinTS** | +34.1 | [-264.9, +388.8] | +0.04 | 7.61e-01 | No | -2,713.6 | [-2,987.2, -2,344.8] |
| **Warm LinTS** | +648.7 | [+422.1, +872.1] | +0.99 | 4.41e-05 | Yes | -2,099.0 | [-2,236.8, -1,961.2] |
| **Warm LinUCB** | +534.5 | [+315.7, +768.3] | +0.83 | 1.53e-04 | Yes | -2,213.2 | [-2,357.8, -2,055.4] |
| **Recently Viewed + Popularity** | +643.5 | [+455.3, +823.5] | +1.22 | 1.99e-06 | Yes | -2,104.2 | [-2,126.7, -2,080.0] |
| **Epsilon-Greedy** | +34.9 | [-149.2, +230.8] | +0.06 | 3.82e-01 | No | -2,712.7 | [-2,823.8, -2,596.5] |
| **Department Heuristic** | +80.6 | [-106.9, +257.7] | +0.16 | 1.71e-01 | No | -2,667.1 | [-2,697.0, -2,639.4] |
| **Popularity** | +2,747.7 | [+2,549.8, +2,935.2] | +4.93 | 1.86e-09 | Yes | +0.0 | [+0.0, +0.0] |
| **Random** | +3,396.0 | [+3,198.0, +3,584.9] | +6.10 | 1.86e-09 | Yes | +648.3 | [+631.7, +665.2] |

---

## 3. Warm-Start vs Cold-Start Empirical Analysis
- **Misspecified Prior Penalty**: Warm LinUCB incurs +534.5 higher cumulative regret than Cold LinUCB.
  Because the Retailrocket sufficient statistics reflect historical catalog co-occurrences that disagree with the active catalog structure, the warm-start acts as a misspecified prior that requires thousands of rounds to unlearn.
- **Cold-Start Early Regret (Rounds 1 - 1,000)**: Cold LinUCB incurs 245.7 regret while Warm LinUCB incurs 209.7.
- **New Inventory Discovery**: When 12 cold-start products enter at round 10,000, Cold LinUCB allocates 6.87% of slate impressions to them via upper confidence bounds, compared to 0.00% under Popularity.

---

## 4. Drift Adaptation & Non-Stationary Dynamics
- **Post-Drift Regret (Rounds 10,001 - 12,000)**:
  - **Discounted LinUCB (gamma=0.9999)**: 316.2
  - **Standard Cold LinUCB**: 346.1
  - **Popularity**: 398.8
  Discounted LinUCB adapts significantly faster after abrupt drift by decaying prior sample weights.

---

## 5. Production Pipeline Ablation (Candidate Gen + Bandit + Diversity)
### Table 3: Full Catalog Bandit-Only vs Multi-Stage Production Pipeline
| Policy | Mode | Cumulative Regret | Cumulative Reward | Slate CTR (%) | Diversity (Cats) | Catalog Coverage (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cold LinUCB** | Bandit-Only | 3,807.7 | 5,938.3 | 12.18% | 2.55 | 79.8% |
| **Cold LinUCB** | Pipeline (50+Div) | 3,228.1 | 6,275.2 | 13.02% | 2.76 | 33.6% |
| **Discounted LinUCB** | Bandit-Only | 3,219.7 | 6,765.2 | 13.66% | 2.62 | 86.4% |
| **Discounted LinUCB** | Pipeline (50+Div) | 3,078.2 | 6,847.8 | 13.93% | 2.75 | 34.4% |
| **Recently Viewed + Popularity** | Bandit-Only | 4,451.2 | 5,471.0 | 11.43% | 1.58 | 99.0% |
| **Recently Viewed + Popularity** | Pipeline (50+Div) | 4,367.2 | 5,435.7 | 11.42% | 2.08 | 34.4% |
| **Popularity** | Bandit-Only | 6,555.3 | 2,982.0 | 6.44% | 1.00 | 3.8% |
| **Popularity** | Pipeline (50+Div) | 6,200.7 | 3,243.2 | 6.93% | 2.00 | 5.7% |
| **Random** | Bandit-Only | 7,203.6 | 2,302.2 | 5.14% | 4.35 | 100.0% |
| **Random** | Pipeline (50+Div) | 6,021.7 | 3,401.8 | 7.40% | 3.18 | 34.4% |

---

## 6. Hyper-Parameter Sensitivity (Tuning Sweep)
### Table 4: Exploration Parameter Sweep (Cold LinUCB, 10,000 Rounds)
| alpha | Cumulative Regret |
| :--- | :--- |
| 0.1 | 1,475.5 |
| 0.25 | 1,410.9 |
| 0.5 | 1,509.7 |
| 1.0 | 1,472.4 |

### Table 5: Discount Factor Sweep (Discounted LinUCB, 10,000 Rounds)
| gamma | Cumulative Regret |
| :--- | :--- |
| 0.999 | 1,959.1 |
| 0.9995 | 1,757.5 |
| 0.9999 | 1,317.8 |
| 1.0 | 1,509.0 |

---

## 7. Where Contextual Bandits Do NOT Beat Simple Baselines
> [!WARNING]
> **Transparent Failure Modes**:
> 1. **Short Sessions / Cold Users**: On the first 1-3 sessions of a visitor, `Recently Viewed + Popularity` achieves lower instantaneous regret than Cold LinUCB because LinUCB's exploration bonus over-explores unfamiliar departments.
> 2. **Uncalibrated Historical Warm-Start**: Injecting offline warm-start sufficient statistics from Retailrocket degraded online cumulative regret significantly (+534.5 regret vs cold LinUCB), proving that out-of-domain historical logs can hurt online bandit convergence.
> 3. **Latency**: Cold LinUCB average step latency (0.536 ms) is ~10x higher than Popularity (0.176 ms), though well within sub-50ms web SLAs.

---

## 8. Artifacts & Generated Assets
- Master Results JSON: `experiments/results/sim_v2_master_results.json`
- Assumption Card: `docs/sim_assumptions.md`
- Publication Figures: `experiments/plots/sim_v2/` (6 figures at 300 DPI)