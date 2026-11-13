# SmartCart AI — Authoritative Final Claims & Scientific Findings (Phase 12B Delivery)

> [!IMPORTANT]
> **Epistemic Scope & Synthetic Assumptions**:
> All empirical results in this document describe recommender policy behaviour **strictly under the calibrated position-biased simulator's stated synthetic assumptions** (`experiments/sim_v2/`).
> Evaluated over **20,000 rounds** across **30 evaluation seeds** (paired Common Random Numbers).
> They do **not** represent production readiness, do **not** establish causal customer impact in the wild,
> and must **never** be cited as evidence of production customer behaviour without an online A/B test.
> For the complete specification of synthetic user personas, examination bias, and utility equations, see [`docs/sim_assumptions.md`](file:///c:/Users/ASUS%20TUF/Desktop/SMARTCART-AI/docs/sim_assumptions.md).

---

## 1. Supported Scientific Claims (With Paired Statistical Evidence)

### Claim 1: Two-Stage Hybrid (Dept-Restricted LinUCB) Outperforms the Department Heuristic
- **Statement**: Restricting candidate generation to the user's active target department while applying LinUCB ranking across department SKUs achieves significantly lower cumulative regret than the static Department Heuristic under well-specified user choices.
- **Evidence**:
  - `Dept-Restricted LinUCB (Cold)` achieves **1,429.6** cumulative regret [1,371.8, 1,498.0], outperforming the Department Heuristic (**2,225.6**) by **-796.1 points** (95% CI [-856.5, -729.9]).
  - Paired Holm-adjusted $p < 0.001$, Cohen's $d = 4.354$.

### Claim 2: Two-Stage Hybrid Outperforms Standard Catalog-Wide Cold LinUCB
- **Statement**: In multi-department retail catalogs with strong category clustering, restricting LinUCB to rank products within the target department eliminates catalog-wide exploration penalties while retaining personalized ranking by price, rating, and recency.
- **Evidence**:
  - Drops regret from **2,942.0** (Cold LinUCB) to **1,429.6** (Dept-Restricted LinUCB), an improvement of **-1,512.4 points** (95% CI [-1,613.5, -1,412.0]).
  - Paired Holm-adjusted $p < 0.001$, Cohen's $d = 4.989$.

### Claim 3: Discounted LinUCB (gamma*=0.9995) Adapts to Abrupt Preference Drift
- **Statement**: Exponential discounting of historical interactions significantly accelerates adaptation after preference shift ($t=10,000$), outperforming standard LinUCB.
- **Evidence**:
  - `Discounted LinUCB (gamma*=0.9995)` achieves **2,534.6** vs **2,942.0** for Cold LinUCB, a reduction of **407.4 points** (95% CI [311.1, 502.1]).
  - Paired Holm-adjusted $p < 0.001$, Cohen's $d = -1.491$.

### Claim 4: Bit-for-Bit Determinism & Identity Verification
- **Statement**: When $\gamma = 1.0$, Discounted LinUCB reduces mathematically and operationally to Cold LinUCB.
- **Evidence**:
  - Cumulative regret across all 30 evaluation seeds is identically 2,942.0 vs 2,942.0, with mean difference exactly $0.0000000000000000$ floating precision.
  - Salted hash dependencies have been audited and replaced with deterministic SHA-256 (`compute_brand_bucket`), ensuring verified cross-process RNG invariance across different `PYTHONHASHSEED` configurations.

### Claim 5: Mathematical Trajectory Oracle Identity
- **Statement**: For any recommendation policy $\pi$, step regret equals the difference between the optimal slate reward along $\pi$'s active trace and the served slate reward.
- **Evidence**:
  $$\text{Cumulative Reward}(\pi) + \text{Cumulative Regret}(\pi) \equiv R^*_{\text{trajectory}}(\pi) = \sum_{t=1}^T R^*(u_t, t)$$
  Holds identically across all 26 evaluated policies. Trajectory oracle expected rewards range from 7,588.9 (Random) to 8,132.7 (Oracle).

### Claim 6: Moderate Warm-Start Benefit vs Negative Transfer Threshold
- **Statement**: Initializing bandit sufficient statistics with modest historical interaction weight ($N_{\text{cap}} = 500$) accelerates early-horizon learning, but excessive weight ($N_{\text{cap}} = 10,000$) causes negative transfer under preference drift.
- **Evidence**:
  - Warm LinUCB cumulative regret (regenerated post brand-hash fix): $N_{\text{cap}}=0 \rightarrow 2,942.0$, $N_{\text{cap}}=500 \rightarrow 2,677.3$, $N_{\text{cap}}=2,000 \rightarrow 2,813.4$, $N_{\text{cap}}=10,000 \rightarrow 3,381.5$.
  - $N_{\text{cap}}=500$ achieves lowest full-horizon regret (2,677.3), while $N_{\text{cap}}=10,000$ degrades performance by 439.5 points compared to cold start.

---

## 2. Policy Robustness Across Synthetic Environments (Regret Share %)

The table below presents the **Regret Share (%)** (lower is better, where Random = 100.0% and Oracle = 0.0%) across all four evaluated environment variants (30 evaluation seeds, $T=20,000$ rounds):

| Policy | Well-Specified Base | Hidden Style Factor | Nonlinear Interactions | High Funnel Conversion | Average Share | Worst-Case Share |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Dept-Restricted LinUCB (Cold, m=0)** | 17.9% | 48.0% | 51.6% | 18.6% | 34.1% | 51.6% |
| **Dept-Restricted LinUCB (Cold, m=2)** | 22.3% | 50.2% | 47.5% | 22.8% | 35.7% | 50.2% |
| **Dept-Restricted LinUCB (Disc, m=0)** | 18.0% | 48.2% | 51.1% | 18.8% | 34.0% | 51.1% |
| **Dept-Restricted LinUCB (Disc, m=2)** | 23.6% | 51.5% | 46.2% | 24.4% | 36.4% | 51.5% |
| **Residual LinUCB** | 20.2% | 49.5% | 42.6% | 21.8% | 33.5% | 49.5% |
| **Cold LinUCB** | 37.5% | 60.3% | 30.6% | 38.0% | 41.6% | 60.3% |
| **Discounted LinUCB (gamma*=0.9995)** | 32.1% | 57.0% | 26.6% | 33.8% | 37.4% | 57.0% |
| **Department Heuristic** | 28.4% | 55.2% | 71.8% | 29.3% | 46.2% | 71.8% |
| **Warm LinUCB (N_cap=10,000)** | 42.9% | 62.6% | 27.9% | 40.4% | 43.5% | 62.6% |
| **Recently Viewed + Popularity** | 64.8% | 77.6% | 88.1% | 65.9% | 74.1% | 88.1% |
| **Popularity Baseline** | 65.7% | 78.4% | 88.9% | 66.8% | 74.9% | 88.9% |
| **Random Baseline** | 72.8% | 83.1% | 88.1% | 74.1% | 79.5% | 88.1% |

---

## 3. Withdrawn, Excluded & Unsupported Claims (Honest Epistemic Boundaries)

### Excluded from Supported Claims: Residual LinUCB is Environment-Informed
- **Classification**: **ENVIRONMENT-INFORMED (EXCLUDED FROM CLAIMS)**.
- **Explanation**:
  - Residual LinUCB relies on an empirical heuristic offset $h_{\text{offset}} = 0.35$ derived from the simulator's latent utility scale ($\mathbb{E}[V] \in [0.1, 0.5]$), checked on tuning seeds 101–105 only.
  - Because this hyperparameter is informed by environment structure that would be unobserved in a real production system without simulator knowledge, it is classified as environment-informed and excluded from the core supported claims table.

### Parity Test Scope
- **Statement**: The parity test covers policy scoring logic with identical inputs, not the live feature extraction pipeline.

### Withdrawn Claim 1: "A Single Policy Dominates Across All Retail Environments"
- **Status**: **WITHDRAWN / UNSUPPORTED**.
- **Explanation**:
  - In the well-specified base environment where category affinity dominates, `Dept-Restricted LinUCB` achieves lowest regret (17.9% regret share for m=0, 22.3% for m=2).
  - However, under **Nonlinear Feature Interactions** (price $\times$ category sensitivity offsets and rating $\times$ discount synergies), the Department Heuristic regret share reaches **71.8%**.
  - Under nonlinear choices, adaptive learning bandits like `Discounted LinUCB` (**26.6%**) and `Cold LinUCB` (**30.6%**) outperform the heuristic by more than 8,000 regret points.
  - Because no single policy strictly dominates all environments, SmartCart AI exposes policy switching directly to administrators in the live Admin Dashboard.
  - Because no single policy strictly dominates all environments, SmartCart AI exposes policy switching directly to administrators in the live Admin Dashboard.

### Withdrawn Claim 2: "The Offline Retailrocket Evaluation Proves Online Customer Lift"
- **Status**: **WITHDRAWN / PROXY ONLY**.
- **Explanation**:
  - The Retailrocket offline evaluation (`experiments/results/phase10b_master_results.json`) was declared **permanently frozen in Phase 10D**.
  - It is strictly a **proxy offline evaluation on logged positive-only interactions** (views, cart adds, transactions), subject to unobserved impression exposure bias.
  - Offline prequential replay metrics (HR@10, MRR) cannot establish causal customer purchase lift without an online randomized controlled trial (A/B test).

### Withdrawn Claim 3: "Retailrocket Warm-Start Snapshots Are Ready for Live Production"
- **Status**: **WITHDRAWN / DISABLED IN LIVE MODELS**.
- **Explanation**:
  - Retailrocket warm-start snapshots remain **strictly DISABLED** (`warm_start: OFF`) in live models in `smartcart.db`.
  - While $N_{\text{cap}}=500$ proved beneficial in synthetic simulation, mapping between Retailrocket historical item IDs and DummyJSON/catalog SKUs is an imperfect heuristic proxy. Live models initialize from cold-start prior ($A = \lambda I, b = 0$).

---

## 4. Hash and Ordering Audit Erratum

### Root Cause Analysis:
Prior to Phase 12B, feature indices 22–29 (`brand_bucket_0`..`brand_bucket_7`) in `backend/app/services/feature_engineering.py` computed brand bucket assignments via Python's builtin `abs(hash(p_brand)) % 8`. Because Python initializes a random process salt (`PYTHONHASHSEED`) on startup, brand bucket features varied across different Python process runs.

### Remediation:
1. Replaced salted `hash()` with a deterministic SHA-256 hash bucket (`compute_brand_bucket`) across live, offline, and simulator code paths.
2. Enforced strict deterministic tie-breaking in recommendation ranking (`key=lambda x: (-x.score, x.candidate.product_id)`).
3. Verified bit-for-bit invariance across independent subprocesses with differing `PYTHONHASHSEED` configurations.

### Impact on Frozen Offline Study:
- Feature ablation refits in `experiments/results/phase10b_master_results.json` (`ablations_refit`) confirm that completely ablating the 8 brand hash buckets (Features 22–29) changed overall macro HR@10 by only -0.37% (from 9.37% [9.26, 9.48] to 9.00% [8.89, 9.11]) and repeat macro HR@10 by only -0.14% (52.78% to 52.64%).
- In contrast, ablating Feature 6 (`user_brand_affinity`) collapsed repeat HR@10 from 52.78% to 5.74%.
- **Conclusion**: The process-salted brand hash buckets had a negligible influence ($\le 0.37\%$) on the frozen offline evaluation results. The offline study remains accepted under its stated proxy caveats without requiring a rerun.
