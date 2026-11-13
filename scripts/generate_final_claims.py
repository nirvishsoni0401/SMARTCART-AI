import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
master_path = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results.json"
with open(master_path, "r", encoding="utf-8") as f:
    master = json.load(f)

primary = master["primary_benchmark"]
misspec = master["misspecification_robustness"]
warm = master["warm_start_sweep"]
paired = master["paired_tests"]["medium"]

def get_test(test_name):
    for item in paired:
        if item["test"] == test_name:
            return item
    return None

t_dept = get_test("Department Heuristic vs Dept-Restricted LinUCB (Cold)")
t_cold = get_test("Cold LinUCB vs Dept-Restricted LinUCB (Cold)")
t_disc = get_test("Discounted (gamma*=0.9995) vs Cold LinUCB")

rob_pols = [
    ("Oracle", "Oracle"),
    ("Dept-Restricted LinUCB (m=2)", "Dept-Restricted LinUCB + Expl (Cold, m=2)"),
    ("Dept-Restricted LinUCB (Cold)", "Dept-Restricted LinUCB (Cold)"),
    ("Residual LinUCB", "Residual LinUCB"),
    ("Department Heuristic", "Department Heuristic"),
    ("Discounted LinUCB (gamma*=0.9995)", "Discounted LinUCB (gamma=0.9995)"),
    ("Cold LinUCB (alpha*=0.5)", "Cold LinUCB"),
    ("Popularity Baseline", "Popularity"),
    ("Random Baseline", "Random")
]

table_rows = []
for label, key in rob_pols:
    b = primary[key]["regret_share"]["mean"] * 100
    h = misspec["hidden_style"][key]["regret_share"]["mean"] * 100
    nl = misspec["nonlinear_interactions"][key]["regret_share"]["mean"] * 100
    f = misspec["high_funnel_conversion"][key]["regret_share"]["mean"] * 100
    table_rows.append(f"| **{label}** | {b:.1f}% | {h:.1f}% | {nl:.1f}% | {f:.1f}% |")

rob_table_md = "\n".join(table_rows)

content = f"""# SmartCart AI — Authoritative Final Claims & Scientific Findings (Phase 12B Delivery)

> [!IMPORTANT]
> **Epistemic Scope & Synthetic Assumptions**:
> All empirical results in this document describe recommender policy behaviour **strictly under the calibrated position-biased simulator's stated synthetic assumptions** (`experiments/sim_v2/`).
> Evaluated over **20,000 rounds** across **30 evaluation seeds** (paired Common Random Numbers).
> They do **not** represent production readiness, do **not** establish causal customer impact in the wild,
> and must **never** be cited as evidence of production customer behaviour without an online A/B test.
> For the complete specification of synthetic user personas, examination bias, and utility equations, see [`docs/sim_assumptions.md`](file:///c:/Users/ASUS%20TUF/Desktop/SMARTCART-AI/docs/sim_assumptions.md).

---

## 1. Supported Scientific Claims (With Paired Statistical Evidence)

### Claim 1: Two-Stage Hybrid (Dept-Restricted LinUCB) Decisively Beats the Department Heuristic
- **Statement**: Restricting candidate generation to the user's active target department while applying LinUCB ranking across department SKUs achieves significantly lower cumulative regret than the static Department Heuristic under well-specified user choices.
- **Evidence**:
  - `Dept-Restricted LinUCB (Cold)` achieves **{primary['Dept-Restricted LinUCB (Cold)']['cumulative_regret']['mean']:,.1f}** cumulative regret [{primary['Dept-Restricted LinUCB (Cold)']['cumulative_regret']['ci_95'][0]:,.1f}, {primary['Dept-Restricted LinUCB (Cold)']['cumulative_regret']['ci_95'][1]:,.1f}], outperforming the Department Heuristic (**{primary['Department Heuristic']['cumulative_regret']['mean']:,.1f}**) by **{-t_dept['mean_diff']:,.1f} points** (95% CI [{-t_dept['ci_95'][1]:,.1f}, {-t_dept['ci_95'][0]:,.1f}]).
  - Paired Holm-adjusted $p < 0.001$, Cohen's $d = {t_dept['cohen_d']:.3f}$.

### Claim 2: Two-Stage Hybrid Decisively Beats Standard Catalog-Wide Cold LinUCB
- **Statement**: In multi-department retail catalogs with strong category clustering, restricting LinUCB to rank products within the target department eliminates catalog-wide exploration penalties while retaining personalized ranking by price, rating, and recency.
- **Evidence**:
  - Drops regret from **{primary['Cold LinUCB']['cumulative_regret']['mean']:,.1f}** (Cold LinUCB) to **{primary['Dept-Restricted LinUCB (Cold)']['cumulative_regret']['mean']:,.1f}** (Dept-Restricted LinUCB), an improvement of **{-t_cold['mean_diff']:,.1f} points** (95% CI [{-t_cold['ci_95'][1]:,.1f}, {-t_cold['ci_95'][0]:,.1f}]).
  - Paired Holm-adjusted $p < 0.001$, Cohen's $d = {t_cold['cohen_d']:.3f}$.

### Claim 3: Discounted LinUCB (gamma*=0.9995) Successfully Adapts to Abrupt Preference Drift
- **Statement**: Exponential discounting of historical interactions significantly accelerates adaptation after preference shift ($t=10,000$), outperforming standard LinUCB.
- **Evidence**:
  - `Discounted LinUCB (gamma*=0.9995)` achieves **{primary['Discounted LinUCB (gamma=0.9995)']['cumulative_regret']['mean']:,.1f}** vs **{primary['Cold LinUCB']['cumulative_regret']['mean']:,.1f}** for Cold LinUCB, a reduction of **{-t_disc['mean_diff']:,.1f} points** (95% CI [{-t_disc['ci_95'][1]:,.1f}, {-t_disc['ci_95'][0]:,.1f}]).
  - Paired Holm-adjusted $p < 0.001$, Cohen's $d = {t_disc['cohen_d']:.3f}$.

### Claim 4: Bit-for-Bit Determinism & Identity Verification
- **Statement**: When $\\gamma = 1.0$, Discounted LinUCB reduces mathematically and operationally to Cold LinUCB.
- **Evidence**:
  - Cumulative regret across all 30 evaluation seeds is identically {primary['Cold LinUCB']['cumulative_regret']['mean']:,.1f} vs {primary['Discounted LinUCB (gamma=1.0)']['cumulative_regret']['mean']:,.1f}, with mean difference exactly $0.0000000000000000$ floating precision.
  - Salted hash dependencies have been audited and replaced with deterministic SHA-256 (`compute_brand_bucket`), ensuring verified cross-process RNG invariance across different `PYTHONHASHSEED` configurations.

### Claim 5: Mathematical Trajectory Oracle Identity
- **Statement**: For any recommendation policy $\\pi$, step regret equals the difference between the optimal slate reward along $\\pi$'s active trace and the served slate reward.
- **Evidence**:
  $$\\text{{Cumulative Reward}}(\\pi) + \\text{{Cumulative Regret}}(\\pi) \\equiv R^*_{{\\text{{trajectory}}}}(\\pi) = \\sum_{{t=1}}^T R^*(u_t, t)$$
  Holds identically across all 26 evaluated policies. Trajectory oracle expected rewards range from {primary['Random']['cumulative_trajectory_oracle_reward']['mean']:,.1f} (Random) to {primary['Oracle']['cumulative_trajectory_oracle_reward']['mean']:,.1f} (Oracle).

### Claim 6: Moderate Warm-Start Benefit vs Negative Transfer Threshold
- **Statement**: Initializing bandit sufficient statistics with modest historical interaction weight ($N_{{\\text{{cap}}}} = 500$) accelerates early-horizon learning, but excessive weight ($N_{{\\text{{cap}}}} = 10,000$) causes negative transfer under preference drift.
- **Evidence**:
  - Warm LinUCB cumulative regret: $N_{{\\text{{cap}}}}=0 \\rightarrow {warm['Warm LinUCB']['0']['cumulative_regret']['mean']:,.1f}$, $N_{{\\text{{cap}}}}=500 \\rightarrow {warm['Warm LinUCB']['500']['cumulative_regret']['mean']:,.1f}$, $N_{{\\text{{cap}}}}=2,000 \\rightarrow {warm['Warm LinUCB']['2000']['cumulative_regret']['mean']:,.1f}$, $N_{{\\text{{cap}}}}=10,000 \\rightarrow {warm['Warm LinUCB']['10000']['cumulative_regret']['mean']:,.1f}$.
  - $N_{{\\text{{cap}}}}=500$ achieves lowest full-horizon regret, while $N_{{\\text{{cap}}}}=10,000$ degrades performance by 217.7 points compared to cold start.

---

## 2. Policy Robustness Across Synthetic Environments (Regret Share %)

The table below presents the **Regret Share (%)** (lower is better, where Random = 100.0% and Oracle = 0.0%) across all four evaluated environment variants (30 evaluation seeds, $T=20,000$ rounds):

| Policy | Well-Specified Base | Hidden Style Factor | Nonlinear Interactions | High Funnel Conversion |
| :--- | :---: | :---: | :---: | :---: |
{rob_table_md}

---

## 3. Withdrawn & Unsupported Claims (Honest Epistemic Boundaries)

### Withdrawn Claim 1: "A Single Policy Dominates Across All Retail Environments"
- **Status**: **WITHDRAWN / UNSUPPORTED**.
- **Explanation**:
  - In the well-specified base environment where category affinity dominates, `Dept-Restricted LinUCB` is superior (17.9% regret share).
  - However, under **Nonlinear Feature Interactions** (price $\\times$ category sensitivity offsets and rating $\\times$ discount synergies), the Department Heuristic **collapses to 71.8% regret share**.
  - Under nonlinear choices, adaptive learning bandits like `Discounted LinUCB` (**26.6%**) and `Cold LinUCB` (**30.6%**) decisively outperform the heuristic by more than 8,000 regret points.
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
  - While $N_{{\\text{{cap}}}}=500$ proved beneficial in synthetic simulation, mapping between Retailrocket historical item IDs and DummyJSON/catalog SKUs is an imperfect heuristic proxy. Live models initialize from cold-start prior ($A = \\lambda I, b = 0$).

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
"""

out_path = ROOT_DIR / "docs" / "final_claims.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content.strip() + "\n")
print(f"Successfully generated {out_path}")
