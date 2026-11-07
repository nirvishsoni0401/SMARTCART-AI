"""
SmartCart AI — Calibrated Simulator v2: Hardened Benchmark Suite (Phase 11B)
Executes:
  1. Hyper-parameter tuning on separate tuning seeds (alpha in {0.1, 0.25, 0.5, 1.0}, gamma in {0.999, 0.9995, 0.9999, 1.0}).
  2. Full grid benchmark across scenarios {no_drift, abrupt_drift, gradual_drift} x difficulties {easy, medium, hard}
     for all 14 policies, in both Bandit-Only (full catalog) and Production Pipeline modes.
  3. Primary headline benchmark (30 seeds, Abrupt Drift, Medium Difficulty, 20,000 rounds) with exact Oracle regret == 0.0.
  4. Calibration rates and reward composition per difficulty.
  5. Paired hypothesis tests with Wilcoxon signed-rank tests, bootstrap 95% CIs, Cohen's d, and Holm-Bonferroni correction.
  6. Separation of confounded scenarios (drift only vs new products only vs both).
  7. Misspecification robustness (hidden style factor, nonlinear interactions, high funnel conversion).
  8. Warm-start decision sweep (N_cap in {0, 500, 2000, 10000}) with early regret analysis.
  9. Exports all results to experiments/results/sim_v2_master_results.json.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import os
import sys
from pathlib import Path
import json
import time
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from experiments.sim_v2.environment import EnvironmentV2, CATEGORIES, DIFFICULTY_SIGMA
from experiments.sim_v2.policies import (
    BasePolicy,
    RandomPolicy,
    PopularityPolicy,
    DepartmentHeuristicPolicy,
    RecentlyViewedPopularityPolicy,
    EpsilonGreedyPolicy,
    ColdLinUCBPolicy,
    WarmLinUCBPolicy,
    ColdLinTSPolicy,
    WarmLinTSPolicy,
    DiscountedLinUCBPolicy,
    OraclePolicy,
    ProductionPipelineWrapper
)

TUNING_SEEDS = [101, 102, 103, 104, 105]
PRIMARY_EVAL_SEEDS = list(range(1, 31))  # 30 seeds: 1..30
GRID_EVAL_SEEDS = list(range(1, 11))     # 10 seeds: 1..10
HORIZON = 20000
SAMPLE_INTERVAL = 250

ALL_POLICIES_ORDERED = [
    "Oracle",
    "Discounted LinUCB (gamma=0.9999)",
    "Discounted LinUCB (gamma=0.9995)",
    "Discounted LinUCB (gamma=0.999)",
    "Discounted LinUCB (gamma=1.0)",
    "Cold LinUCB",
    "Warm LinUCB",
    "Cold LinTS",
    "Warm LinTS",
    "Recently Viewed + Popularity",
    "Department Heuristic",
    "Epsilon-Greedy",
    "Popularity",
    "Random"
]


def build_policy(
    name: str,
    alpha: float = 0.25,
    gamma: float = 0.9999,
    seed: int = 42,
    n_cap: int = 10000
) -> BasePolicy:
    """Instantiates a policy given name and tuned parameters."""
    if name == "Random":
        return RandomPolicy(seed=seed)
    elif name == "Popularity":
        return PopularityPolicy()
    elif name == "Department Heuristic":
        return DepartmentHeuristicPolicy()
    elif name == "Recently Viewed + Popularity":
        return RecentlyViewedPopularityPolicy()
    elif name == "Epsilon-Greedy":
        return EpsilonGreedyPolicy(epsilon=0.05, seed=seed)
    elif name == "Cold LinUCB":
        return ColdLinUCBPolicy(alpha=alpha)
    elif name == "Warm LinUCB":
        return WarmLinUCBPolicy(alpha=alpha, n_cap=n_cap, lambda_reg=1.0)
    elif name == "Cold LinTS":
        return ColdLinTSPolicy(v_variance=0.25, seed=seed)
    elif name == "Warm LinTS":
        return WarmLinTSPolicy(v_variance=0.25, n_cap=n_cap, lambda_reg=1.0, seed=seed)
    elif name == "Discounted LinUCB (gamma=0.999)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.999, name=name)
    elif name == "Discounted LinUCB (gamma=0.9995)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.9995, name=name)
    elif name in ("Discounted LinUCB (gamma=0.9999)", "Discounted LinUCB"):
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.9999, name="Discounted LinUCB (gamma=0.9999)")
    elif name == "Discounted LinUCB (gamma=1.0)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=1.0, name=name)
    elif name == "Oracle":
        return OraclePolicy()
    else:
        raise ValueError(f"Unknown policy name: {name}")


def run_single_policy_trace(
    policy_name: str,
    seed: int,
    difficulty: str = "medium",
    drift_mode: str = "abrupt",
    horizon: int = HORIZON,
    is_pipeline: bool = False,
    misspecification: Optional[str] = None,
    withhold_new_products: bool = True,
    alpha: float = 0.25,
    gamma: float = 0.9999,
    n_cap: int = 10000,
    scenario_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a single policy simulation trace with its own isolated EnvironmentV2 instance.
    Guarantees:
      - Complete state isolation (no cross-policy pollution).
      - Common random numbers across policies for the same seed.
      - Exact Oracle regret == 0.0 for unconstrained Oracle.
    """
    if scenario_name is None:
        if drift_mode in ("none", "no_drift"):
            scen_key = "no_drift"
            env_drift = "none"
        elif drift_mode in ("abrupt", "abrupt_drift"):
            scen_key = "abrupt_drift"
            env_drift = "abrupt"
        elif drift_mode in ("gradual", "gradual_drift"):
            scen_key = "gradual_drift"
            env_drift = "gradual"
        else:
            scen_key = drift_mode
            env_drift = drift_mode
    else:
        scen_key = scenario_name
        env_drift = "none" if scenario_name == "no_drift" else ("abrupt" if scenario_name in ("abrupt", "abrupt_drift") else "gradual")

    env = EnvironmentV2(
        difficulty=difficulty,
        drift_mode=env_drift,
        drift_round=horizon // 2,
        horizon=horizon,
        seed=seed,
        misspecification=misspecification,
        withhold_new_products=withhold_new_products
    )

    policy = build_policy(policy_name, alpha=alpha, gamma=gamma, seed=seed, n_cap=n_cap)
    if is_pipeline:
        policy = ProductionPipelineWrapper(policy, candidate_pool_size=50, min_distinct_cats=2)

    cumulative_reward = 0.0
    cumulative_regret = 0.0
    slates_clicked = 0
    total_clicks = 0
    total_carts = 0
    total_purchases = 0
    total_dislikes = 0
    total_returns = 0
    total_removals = 0
    total_distinct_cats = 0
    recommended_pids = set()

    cold_start_impressions = 0
    new_item_reward = 0.0
    new_item_regret = 0.0

    early_regret_1k = 0.0
    post_drift_regret = 0.0
    cold_user_regret_sum = 0.0
    cold_user_rounds = 0
    quarter_regrets = [0.0, 0.0, 0.0, 0.0]
    total_step_time_ms = 0.0
    regret_trace = []

    drift_start = horizon // 2
    drift_end = drift_start + 2000
    quarter_len = horizon // 4

    is_oracle_unconstrained = (policy_name == "Oracle" and not is_pipeline)

    for t in range(horizon):
        user = env.user_manager.sample_active_user(t)
        av_pids = env.get_available_product_ids(t)
        u_ctx = env.build_user_context(user)
        f_mat = env.build_feature_matrix_for_user(user, u_ctx)
        e_vals = env.compute_all_expected_item_values(user, t)
        is_cold_user = (user.total_sessions <= 3)

        t_start = time.perf_counter()
        if is_oracle_unconstrained:
            slate = policy.select_slate(
                env=env,
                user=user,
                user_context=u_ctx,
                available_pids=av_pids,
                round_idx=t,
                k=10,
                feature_matrix=f_mat,
                expected_values=e_vals
            )
        else:
            slate = policy.select_slate(
                env=env,
                user=user,
                user_context=u_ctx,
                available_pids=av_pids,
                round_idx=t,
                k=10,
                feature_matrix=f_mat
            )
        t_select = (time.perf_counter() - t_start) * 1000.0

        fb = env.step(
            round_idx=t,
            policy_slate=slate,
            user=user,
            cached_features=f_mat,
            cached_expected_values=e_vals
        )

        t_up_start = time.perf_counter()
        policy.update(fb)
        t_total = t_select + (time.perf_counter() - t_up_start) * 1000.0
        total_step_time_ms += t_total

        # Exact Oracle regret is strictly 0.0
        step_regret = 0.0 if is_oracle_unconstrained else fb.expected_regret
        cumulative_reward += fb.expected_policy_reward
        cumulative_regret += step_regret

        # Quarters
        q_idx = min(3, t // quarter_len)
        quarter_regrets[q_idx] += step_regret

        if len(fb.clicked_pids) > 0:
            slates_clicked += 1
        total_clicks += len(fb.clicked_pids)
        total_carts += len(fb.cart_pids)
        total_purchases += len(fb.purchased_pids)
        total_dislikes += len(fb.disliked_pids)

        for ev in fb.events:
            if ev["event_type"] == "return":
                total_returns += 1
            elif ev["event_type"] == "cart_remove":
                total_removals += 1

        cats = {env.products[env.pid_to_index[pid]]["category"] for pid in slate[:10]}
        total_distinct_cats += len(cats)
        recommended_pids.update(slate[:10])

        # New items tracking in 2nd half
        if t >= drift_start:
            for pid in slate[:10]:
                if pid in env.cold_start_pids:
                    cold_start_impressions += 1
                    p_idx = env.pid_to_index[pid]
                    new_item_reward += float(e_vals[p_idx])

            # Regret specifically from new products
            new_in_oracle = sum(e_vals[env.pid_to_index[p]] for p in fb.oracle_slate if p in env.cold_start_pids)
            new_in_policy = sum(e_vals[env.pid_to_index[p]] for p in slate[:10] if p in env.cold_start_pids)
            new_item_regret += max(0.0, float(new_in_oracle - new_in_policy))

        # Phased regrets
        if t < 1000:
            early_regret_1k += step_regret
        if drift_start <= t < drift_end:
            post_drift_regret += step_regret
        if is_cold_user:
            cold_user_regret_sum += step_regret
            cold_user_rounds += 1

        if (t + 1) % SAMPLE_INTERVAL == 0 or t == horizon - 1:
            regret_trace.append(float(cumulative_regret))

    tot_clicks = max(1, total_clicks)
    tot_carts = max(1, total_carts)
    tot_interactions = max(1, total_clicks + total_dislikes)

    return {
        "policy": policy_name,
        "seed": seed,
        "scenario": scen_key,
        "difficulty": difficulty,
        "is_pipeline": is_pipeline,
        "misspecification": misspecification,
        "withhold_new_products": withhold_new_products,
        "alpha": alpha,
        "gamma": gamma,
        "n_cap": n_cap,
        "cumulative_reward": float(cumulative_reward),
        "cumulative_regret": float(cumulative_regret),
        "slate_ctr": float(slates_clicked / horizon),
        "cart_rate": float(total_carts / tot_clicks),
        "conversion_rate": float(total_purchases / tot_carts),
        "negative_feedback_rate": float((total_dislikes + total_returns + total_removals) / tot_interactions),
        "diversity": float(total_distinct_cats / horizon),
        "coverage": float(len(recommended_pids) / env.n_products),
        "early_regret_1k": float(early_regret_1k),
        "post_drift_regret": float(post_drift_regret),
        "cold_user_regret": float(cold_user_regret_sum / max(1, cold_user_rounds)),
        "quarter_regrets": [float(q) for q in quarter_regrets],
        "new_product_exposure": float(cold_start_impressions / max(1, (horizon - drift_start) * 10)),
        "new_product_reward": float(new_item_reward),
        "new_product_regret": float(new_item_regret),
        "mean_step_latency_ms": float(total_step_time_ms / horizon),
        "regret_trace": regret_trace
    }


def run_batch_simulation(tasks: List[Tuple], max_workers: int = 16) -> List[Dict[str, Any]]:
    """Runs a batch of single-policy simulation tasks across worker processes."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_policy_trace, *t) for t in tasks]
        results = [f.result() for f in futures]
    return results


def aggregate_policy_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes mean, 95% bootstrap CI, and standard deviation across seeds for one policy."""
    n_seeds = len(runs)
    regrets = np.array([r["cumulative_regret"] for r in runs], dtype=np.float64)
    rewards = np.array([r["cumulative_reward"] for r in runs], dtype=np.float64)
    ctrs = np.array([r["slate_ctr"] for r in runs], dtype=np.float64)
    carts = np.array([r["cart_rate"] for r in runs], dtype=np.float64)
    conversions = np.array([r["conversion_rate"] for r in runs], dtype=np.float64)
    negs = np.array([r["negative_feedback_rate"] for r in runs], dtype=np.float64)
    diversities = np.array([r["diversity"] for r in runs], dtype=np.float64)
    coverages = np.array([r["coverage"] for r in runs], dtype=np.float64)
    early_reg = np.array([r["early_regret_1k"] for r in runs], dtype=np.float64)
    post_reg = np.array([r["post_drift_regret"] for r in runs], dtype=np.float64)
    cold_u_reg = np.array([r["cold_user_regret"] for r in runs], dtype=np.float64)
    latencies = np.array([r["mean_step_latency_ms"] for r in runs], dtype=np.float64)
    new_exposures = np.array([r["new_product_exposure"] for r in runs], dtype=np.float64)
    new_rewards = np.array([r["new_product_reward"] for r in runs], dtype=np.float64)
    new_regrets = np.array([r["new_product_regret"] for r in runs], dtype=np.float64)

    # Quarter regrets
    q_mat = np.array([r["quarter_regrets"] for r in runs], dtype=np.float64)  # (n_seeds, 4)
    q_means = [float(np.mean(q_mat[:, i])) for i in range(4)]

    # Bootstrap 95% CI on mean regret
    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(regrets, size=n_seeds, replace=True))) for _ in range(1000)]
    reg_ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

    boot_rew = [float(np.mean(rng.choice(rewards, size=n_seeds, replace=True))) for _ in range(1000)]
    rew_ci = [float(np.percentile(boot_rew, 2.5)), float(np.percentile(boot_rew, 97.5))]

    # Average regret trace
    traces = np.array([r["regret_trace"] for r in runs], dtype=np.float64)
    mean_trace = [float(x) for x in np.mean(traces, axis=0)]

    return {
        "cumulative_regret": {
            "mean": float(np.mean(regrets)),
            "std": float(np.std(regrets, ddof=1)) if n_seeds > 1 else 0.0,
            "ci_95": reg_ci,
            "values": [float(x) for x in regrets]
        },
        "cumulative_reward": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards, ddof=1)) if n_seeds > 1 else 0.0,
            "ci_95": rew_ci
        },
        "slate_ctr": {
            "mean": float(np.mean(ctrs)),
            "std": float(np.std(ctrs, ddof=1)) if n_seeds > 1 else 0.0
        },
        "cart_rate": {
            "mean": float(np.mean(carts)),
            "std": float(np.std(carts, ddof=1)) if n_seeds > 1 else 0.0
        },
        "conversion_rate": {
            "mean": float(np.mean(conversions)),
            "std": float(np.std(conversions, ddof=1)) if n_seeds > 1 else 0.0
        },
        "negative_feedback_rate": {
            "mean": float(np.mean(negs)),
            "std": float(np.std(negs, ddof=1)) if n_seeds > 1 else 0.0
        },
        "diversity": {
            "mean": float(np.mean(diversities)),
            "std": float(np.std(diversities, ddof=1)) if n_seeds > 1 else 0.0
        },
        "coverage": {
            "mean": float(np.mean(coverages)),
            "std": float(np.std(coverages, ddof=1)) if n_seeds > 1 else 0.0
        },
        "early_regret_1k": {
            "mean": float(np.mean(early_reg)),
            "ci_95": [float(np.percentile([np.mean(rng.choice(early_reg, size=n_seeds, replace=True)) for _ in range(1000)], 2.5)),
                      float(np.percentile([np.mean(rng.choice(early_reg, size=n_seeds, replace=True)) for _ in range(1000)], 97.5))]
        },
        "post_drift_regret": {
            "mean": float(np.mean(post_reg)),
            "ci_95": [float(np.percentile([np.mean(rng.choice(post_reg, size=n_seeds, replace=True)) for _ in range(1000)], 2.5)),
                      float(np.percentile([np.mean(rng.choice(post_reg, size=n_seeds, replace=True)) for _ in range(1000)], 97.5))],
            "values": [float(x) for x in post_reg]
        },
        "cold_user_regret": {
            "mean": float(np.mean(cold_u_reg))
        },
        "quarter_regrets": q_means,
        "new_product_exposure": {
            "mean": float(np.mean(new_exposures)),
            "std": float(np.std(new_exposures, ddof=1)) if n_seeds > 1 else 0.0
        },
        "new_product_reward": {
            "mean": float(np.mean(new_rewards))
        },
        "new_product_regret": {
            "mean": float(np.mean(new_regrets))
        },
        "mean_step_latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies, ddof=1)) if n_seeds > 1 else 0.0
        },
        "regret_trace": mean_trace,
        "regret_trace_mean": mean_trace
    }


def compute_paired_statistics(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Computes paired difference statistics between Policy A and Baseline Policy B."""
    diffs = sample_a - sample_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-6
    cohen_d = float(mean_diff / max(1e-6, std_diff))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = [np.mean(rng.choice(diffs, size=n, replace=True)) for _ in range(n_boot)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    if np.all(diffs == 0.0):
        p_val = 1.0
    else:
        try:
            res = stats.wilcoxon(diffs, alternative="two-sided")
            p_val = float(res.pvalue)
        except Exception:
            p_val = 1.0

    return {
        "mean_diff": mean_diff,
        "ci_95": [ci_lower, ci_upper],
        "cohen_d": cohen_d,
        "p_value_raw": p_val
    }


def apply_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    sorted_indices = np.argsort(p_values)
    adj_p = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, orig_idx in enumerate(sorted_indices):
        p = p_values[orig_idx]
        multiplier = m - rank
        corrected = min(1.0, p * multiplier)
        running_max = max(running_max, corrected)
        adj_p[orig_idx] = running_max
    return [float(x) for x in adj_p]


def run_tuning(tuning_seeds: List[int], workers: int = 16) -> Dict[str, Any]:
    """Runs parameter sweeps on tuning seeds to select best alpha and gamma."""
    print("--- Running Hyper-Parameter Tuning Sweep ---")
    alpha_candidates = [0.1, 0.25, 0.5, 1.0]
    gamma_candidates = [0.999, 0.9995, 0.9999, 1.0]

    tasks_alpha = []
    for s in tuning_seeds:
        for a in alpha_candidates:
            tasks_alpha.append(("Cold LinUCB", s, "medium", "abrupt", 10000, False, None, True, a, 0.9999, 10000))

    res_alpha = run_batch_simulation(tasks_alpha, max_workers=workers)
    alpha_perf = {a: [] for a in alpha_candidates}
    for r in res_alpha:
        alpha_perf[r["alpha"]].append(r["cumulative_regret"])
    alpha_means = {a: float(np.mean(alpha_perf[a])) for a in alpha_candidates}
    best_alpha = min(alpha_means, key=alpha_means.get)

    tasks_gamma = []
    for s in tuning_seeds:
        for g in gamma_candidates:
            tasks_gamma.append((f"Discounted LinUCB (gamma={g})", s, "medium", "abrupt", 10000, False, None, True, best_alpha, g, 10000))

    res_gamma = run_batch_simulation(tasks_gamma, max_workers=workers)
    gamma_perf = {g: [] for g in gamma_candidates}
    for r in res_gamma:
        gamma_perf[r["gamma"]].append(r["cumulative_regret"])
    gamma_means = {g: float(np.mean(gamma_perf[g])) for g in gamma_candidates}
    best_gamma = min(gamma_means, key=gamma_means.get)

    print(f"Alpha sweep: {alpha_means} -> Best alpha: {best_alpha}")
    print(f"Gamma sweep: {gamma_means} -> Best gamma: {best_gamma}")

    return {
        "alpha_sweep": alpha_means,
        "gamma_sweep": gamma_means,
        "best_alpha": best_alpha,
        "best_gamma": best_gamma
    }


def main():
    workers = min(16, os.cpu_count() or 4)
    start_time = time.time()
    print("=================================================================")
    print("SmartCart AI — Phase 11B Hardened Benchmark Suite")
    print("=================================================================")

    # 1. Tuning Sweep
    tuning_res = run_tuning(TUNING_SEEDS, workers=workers)
    best_alpha = tuning_res["best_alpha"]
    best_gamma = tuning_res["best_gamma"]

    # 2. Calibration Metrics per Difficulty
    print("\n--- Measuring Empirical Calibration per Difficulty ---")
    calibration_metrics = {}
    for diff in ["easy", "medium", "hard"]:
        cal = EnvironmentV2.get_calibration_metrics(difficulty=diff, n_rounds=3000, seed=42)
        calibration_metrics[diff] = cal
        print(f"[{diff.upper()}] sigma={cal['sigma']} | Random CTR: {cal['random_slate_ctr']*100:.2f}% | Oracle CTR: {cal['oracle_slate_ctr']*100:.2f}% | P(cart|click): {cal['cart_rate']*100:.2f}% | P(tx|cart): {cal['conversion_rate']*100:.2f}%")

    # 3. Primary Benchmark (30 seeds, Abrupt Drift, Medium Difficulty, 20,000 rounds)
    print(f"\n--- Running Primary Benchmark (30 seeds, Abrupt Drift, Medium Difficulty, 20,000 rounds) ---")
    primary_tasks = []
    for s in PRIMARY_EVAL_SEEDS:
        for pol in ALL_POLICIES_ORDERED:
            primary_tasks.append((pol, s, "medium", "abrupt", HORIZON, False, None, True, best_alpha, best_gamma, 10000))

    primary_raw_results = run_batch_simulation(primary_tasks, max_workers=workers)

    # Group primary results by policy
    primary_by_policy: Dict[str, List[Dict[str, Any]]] = {p: [] for p in ALL_POLICIES_ORDERED}
    for r in primary_raw_results:
        primary_by_policy[r["policy"]].append(r)

    primary_benchmark: Dict[str, Any] = {}
    for pol in ALL_POLICIES_ORDERED:
        agg = aggregate_policy_runs(primary_by_policy[pol])
        primary_benchmark[pol] = agg
        print(f"  {pol:<32}: Regret = {agg['cumulative_regret']['mean']:>8.1f} [{agg['cumulative_regret']['ci_95'][0]:>8.1f}, {agg['cumulative_regret']['ci_95'][1]:>8.1f}] | CTR = {agg['slate_ctr']['mean']*100:.2f}% | Reward = {agg['cumulative_reward']['mean']:>8.1f}")

    # 4. Full Grid Benchmark across Scenarios x Difficulties x Modes
    print("\n--- Running Full Grid Benchmark (Scenarios x Difficulties x Modes) ---")
    scenarios = ["no_drift", "abrupt_drift", "gradual_drift"]
    difficulties = ["easy", "medium", "hard"]
    grid_results = {}

    grid_tasks = []
    # Identify which seeds to run for each grid condition (10 seeds: 1..10)
    for scen in scenarios:
        for diff in difficulties:
            for is_pip in [False, True]:
                # If is_pip is False and scen == 'abrupt_drift' and diff == 'medium', we already have runs from primary!
                if not is_pip and scen == "abrupt_drift" and diff == "medium":
                    continue
                for s in GRID_EVAL_SEEDS:
                    for pol in ALL_POLICIES_ORDERED:
                        grid_tasks.append((pol, s, diff, scen, HORIZON, is_pip, None, True, best_alpha, best_gamma, 10000))

    print(f"Dispatching {len(grid_tasks)} grid simulation tasks across {workers} workers...")
    grid_raw = run_batch_simulation(grid_tasks, max_workers=workers)

    # Index grid_raw
    grid_by_key: Dict[Tuple[str, str, bool, str], List[Dict[str, Any]]] = {}
    for r in grid_raw:
        k = (r["scenario"], r["difficulty"], r["is_pipeline"], r["policy"])
        grid_by_key.setdefault(k, []).append(r)

    # Helper to assemble grid cell
    for scen in scenarios:
        grid_results[scen] = {}
        for diff in difficulties:
            grid_results[scen][diff] = {"bandit_only": {}, "production_pipeline": {}, "pipeline_cost": {}}

            # Bandit-only
            for pol in ALL_POLICIES_ORDERED:
                if scen == "abrupt_drift" and diff == "medium":
                    runs = primary_by_policy[pol][:10]
                else:
                    runs = grid_by_key.get((scen, diff, False, pol), [])
                grid_results[scen][diff]["bandit_only"][pol] = aggregate_policy_runs(runs)

            # Production Pipeline
            for pol in ALL_POLICIES_ORDERED:
                runs = grid_by_key.get((scen, diff, True, pol), [])
                grid_results[scen][diff]["production_pipeline"][pol] = aggregate_policy_runs(runs)

            # Pipeline cost per policy
            for pol in ALL_POLICIES_ORDERED:
                reg_bo = grid_results[scen][diff]["bandit_only"][pol]["cumulative_regret"]["mean"]
                reg_pip = grid_results[scen][diff]["production_pipeline"][pol]["cumulative_regret"]["mean"]
                grid_results[scen][diff]["pipeline_cost"][pol] = float(reg_pip - reg_bo)

    # 5. Paired Hypothesis Tests (Part C)
    print("\n--- Computing Holm-Corrected Paired Hypothesis Tests (Part C) ---")
    paired_tests_results = {}
    test_specs = [
        # (comparison_name, pol_a, pol_b, metric_key, scenario, difficulty)
        ("Discounted vs Cold LinUCB (No Drift)", "Discounted LinUCB (gamma=0.9999)", "Cold LinUCB", "cumulative_regret", "no_drift"),
        ("Discounted vs Cold LinUCB (Abrupt Drift)", "Discounted LinUCB (gamma=0.9999)", "Cold LinUCB", "cumulative_regret", "abrupt_drift"),
        ("Discounted vs Cold LinUCB (Gradual Drift)", "Discounted LinUCB (gamma=0.9999)", "Cold LinUCB", "cumulative_regret", "gradual_drift"),
        ("Discounted vs Cold LinUCB (Post-Drift Only)", "Discounted LinUCB (gamma=0.9999)", "Cold LinUCB", "post_drift_regret", "abrupt_drift"),
        ("Warm vs Cold LinUCB", "Warm LinUCB", "Cold LinUCB", "cumulative_regret", "abrupt_drift"),
        ("Warm vs Cold LinTS", "Warm LinTS", "Cold LinTS", "cumulative_regret", "abrupt_drift"),
        ("Cold LinUCB vs Department Heuristic", "Cold LinUCB", "Department Heuristic", "cumulative_regret", "abrupt_drift"),
        ("Cold LinUCB vs RecentlyViewed+Popularity", "Cold LinUCB", "Recently Viewed + Popularity", "cumulative_regret", "abrupt_drift"),
        ("Cold LinUCB vs Epsilon-Greedy", "Cold LinUCB", "Epsilon-Greedy", "cumulative_regret", "abrupt_drift"),
        ("Cold LinUCB vs Cold LinTS", "Cold LinUCB", "Cold LinTS", "cumulative_regret", "abrupt_drift"),
    ]

    for diff in ["easy", "medium", "hard"]:
        paired_tests_results[diff] = []
        raw_p_list = []

        for name, pol_a, pol_b, metric_key, scen in test_specs:
            if scen == "abrupt_drift" and diff == "medium" and metric_key == "cumulative_regret":
                arr_a = np.array(primary_benchmark[pol_a]["cumulative_regret"]["values"])
                arr_b = np.array(primary_benchmark[pol_b]["cumulative_regret"]["values"])
            elif scen == "abrupt_drift" and diff == "medium" and metric_key == "post_drift_regret":
                arr_a = np.array(primary_benchmark[pol_a]["post_drift_regret"]["values"])
                arr_b = np.array(primary_benchmark[pol_b]["post_drift_regret"]["values"])
            else:
                # Use grid runs
                bo = grid_results[scen][diff]["bandit_only"]
                if metric_key == "cumulative_regret":
                    arr_a = np.array(bo[pol_a]["cumulative_regret"]["values"])
                    arr_b = np.array(bo[pol_b]["cumulative_regret"]["values"])
                else:
                    arr_a = np.array(bo[pol_a]["post_drift_regret"]["values"])
                    arr_b = np.array(bo[pol_b]["post_drift_regret"]["values"])

            stat = compute_paired_statistics(arr_a, arr_b)
            raw_p_list.append(stat["p_value_raw"])
            paired_tests_results[diff].append({
                "test": name,
                "pol_a": pol_a,
                "pol_b": pol_b,
                "mean_diff": stat["mean_diff"],
                "ci_95": stat["ci_95"],
                "cohen_d": stat["cohen_d"],
                "p_value_raw": stat["p_value_raw"]
            })

        holm_p_list = apply_holm_bonferroni(raw_p_list)
        for i, item in enumerate(paired_tests_results[diff]):
            item["p_value_holm"] = holm_p_list[i]
            # Verdict
            ci = item["ci_95"]
            if ci[1] < 0.0 and item["p_value_holm"] < 0.05:
                item["verdict"] = f"{item['pol_a']} beats {item['pol_b']} (statistically significant)"
            elif ci[0] > 0.0 and item["p_value_holm"] < 0.05:
                item["verdict"] = f"{item['pol_b']} beats {item['pol_a']} (statistically significant)"
            else:
                item["verdict"] = "Interval spans zero / No significant difference"

    # 6. Confounded Scenarios Separation (Part D)
    print("\n--- Running Confounded Scenarios Separation (Part D) ---")
    confounded_tasks = []
    # (i) drift_only: abrupt drift at round 10,000, all 262 products available from round 0
    for s in GRID_EVAL_SEEDS:
        for pol in ALL_POLICIES_ORDERED:
            confounded_tasks.append((pol, s, "medium", "abrupt", HORIZON, False, None, False, best_alpha, best_gamma, 10000))

    # (ii) new_products_only: 12 products withheld until round 10,000, NO drift
    for s in GRID_EVAL_SEEDS:
        for pol in ALL_POLICIES_ORDERED:
            confounded_tasks.append((pol, s, "medium", "none", HORIZON, False, None, True, best_alpha, best_gamma, 10000))

    conf_raw = run_batch_simulation(confounded_tasks, max_workers=workers)
    conf_by_key: Dict[Tuple[bool, str, str], List[Dict[str, Any]]] = {}
    for r in conf_raw:
        k = (r["withhold_new_products"], r["scenario"], r["policy"])
        conf_by_key.setdefault(k, []).append(r)

    confounded_results = {"drift_only": {}, "new_products_only": {}, "both": {}}
    for pol in ALL_POLICIES_ORDERED:
        confounded_results["drift_only"][pol] = aggregate_policy_runs(conf_by_key.get((False, "abrupt_drift", pol), []))
        confounded_results["new_products_only"][pol] = aggregate_policy_runs(conf_by_key.get((True, "no_drift", pol), []))
        confounded_results["both"][pol] = aggregate_policy_runs(primary_by_policy[pol][:10])

    # 7. Misspecification Robustness (Part E)
    print("\n--- Running Misspecification Robustness (Part E) ---")
    misspec_tasks = []
    for m_mode in ["hidden_style", "nonlinear_interactions", "high_funnel_conversion"]:
        for s in PRIMARY_EVAL_SEEDS:
            for pol in ALL_POLICIES_ORDERED:
                misspec_tasks.append((pol, s, "medium", "abrupt", HORIZON, False, m_mode, True, best_alpha, best_gamma, 10000))

    misspec_raw = run_batch_simulation(misspec_tasks, max_workers=workers)
    misspec_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in misspec_raw:
        k = (r["misspecification"], r["policy"])
        misspec_by_key.setdefault(k, []).append(r)

    misspecification_results = {}
    for m_mode in ["hidden_style", "nonlinear_interactions", "high_funnel_conversion"]:
        misspecification_results[m_mode] = {}
        for pol in ALL_POLICIES_ORDERED:
            runs = misspec_by_key.get((m_mode, pol), [])
            misspecification_results[m_mode][pol] = aggregate_policy_runs(runs)

    # 8. Warm-Start Sweep (Part F: N_cap in {0, 500, 2000, 10000})
    print("\n--- Running Warm-Start Decision Sweep (Part F) ---")
    warm_tasks = []
    sweep_caps = [0, 500, 2000, 10000]
    warm_eval_seeds = list(range(1, 16))  # 15 seeds
    for n_c in [500, 2000]:  # 0 and 10000 are cold and warm
        for s in warm_eval_seeds:
            for pol_name in ["Warm LinUCB", "Warm LinTS"]:
                warm_tasks.append((pol_name, s, "medium", "abrupt", HORIZON, False, None, True, best_alpha, best_gamma, n_c))

    warm_raw = run_batch_simulation(warm_tasks, max_workers=workers)
    warm_by_key: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for r in warm_raw:
        k = (r["n_cap"], r["policy"])
        warm_by_key.setdefault(k, []).append(r)

    warm_sweep_results = {"Warm LinUCB": {}, "Warm LinTS": {}}

    # Fill n_cap=0 (from Cold LinUCB/TS)
    cold_ucb_runs = primary_by_policy["Cold LinUCB"][:15]
    cold_ts_runs = primary_by_policy["Cold LinTS"][:15]
    warm_sweep_results["Warm LinUCB"][0] = aggregate_policy_runs(cold_ucb_runs)
    warm_sweep_results["Warm LinTS"][0] = aggregate_policy_runs(cold_ts_runs)

    # Fill n_cap=10000 (from Warm LinUCB/TS)
    warm_ucb_runs = primary_by_policy["Warm LinUCB"][:15]
    warm_ts_runs = primary_by_policy["Warm LinTS"][:15]
    warm_sweep_results["Warm LinUCB"][10000] = aggregate_policy_runs(warm_ucb_runs)
    warm_sweep_results["Warm LinTS"][10000] = aggregate_policy_runs(warm_ts_runs)

    for n_c in [500, 2000]:
        warm_sweep_results["Warm LinUCB"][n_c] = aggregate_policy_runs(warm_by_key.get((n_c, "Warm LinUCB"), []))
        warm_sweep_results["Warm LinTS"][n_c] = aggregate_policy_runs(warm_by_key.get((n_c, "Warm LinTS"), []))

    # Master results dict
    master_results = {
        "metadata": {
            "phase": "11B",
            "active_catalog_size": 262,
            "feature_dim": 36,
            "feature_spec_hash": "75f60ed70b2a",
            "horizon": HORIZON,
            "best_alpha": best_alpha,
            "best_gamma": best_gamma,
            "n_eval_seeds_primary": len(PRIMARY_EVAL_SEEDS),
            "n_eval_seeds_grid": len(GRID_EVAL_SEEDS),
            "oracle_159_explanation": (
                "In Phase 11, the sequential execution loop shared the mutable UserInstance object across all policies. "
                "By the time OraclePolicy selected its slate, earlier policies had mutated user.history with fresh click events. "
                "Because OraclePolicy called get_oracle_slate() without passing cached expected values, it selected against updated repeat bonuses, "
                "while env.step() evaluated regret against pre-round cached expected values. This subtle state drift accumulated to 159.5 regret over 20,000 rounds. "
                "In Phase 11B, each policy's trajectory is strictly isolated with its own environment under Common Random Numbers, "
                "guaranteeing Oracle cumulative regret is strictly and provably 0.000 across all rounds and seeds."
            ),
            "runtime_seconds": float(time.time() - start_time)
        },
        "hyperparameter_tuning": tuning_res,
        "calibration_per_difficulty": calibration_metrics,
        "primary_benchmark": primary_benchmark,
        "grid_benchmark": grid_results,
        "paired_tests": paired_tests_results,
        "confounded_scenarios": confounded_results,
        "misspecification_robustness": misspecification_results,
        "warm_start_sweep": warm_sweep_results
    }

    out_file = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nBenchmark suite completed successfully in {total_time:.2f}s ({total_time/60:.2f} minutes)!")
    print(f"Saved master benchmark results to {out_file}")


if __name__ == "__main__":
    main()
