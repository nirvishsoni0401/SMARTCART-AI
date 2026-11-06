"""
SmartCart AI — Calibrated Simulator v2: Main Benchmark Runner
Executes:
  1. Hyper-parameter tuning on separate tuning seeds (alpha in {0.1, 0.25, 0.5, 1.0}, gamma in {0.999, 0.9995, 0.9999, 1.0}).
  2. Main paired benchmark across 30 evaluation seeds (20,000 rounds each) on the 11 policies.
  3. Robustness scenarios: No Drift, Gradual Drift, Easy, Hard.
  4. Production Pipeline mode: Candidate Generation (50) + Bandit Ranking + Diversity Re-ranking.
  5. Paired statistics: bootstrap 95% CIs, Wilcoxon signed-rank tests, Cohen's d, Holm correction.
  6. Exports results to experiments/results/sim_v2_master_results.json.
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

from experiments.sim_v2.environment import EnvironmentV2, CATEGORIES
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
EVAL_SEEDS = list(range(1, 31))  # 30 seeds: 1..30
HORIZON = 20000
SAMPLE_INTERVAL = 250


def build_policy(name: str, alpha: float = 0.5, gamma: float = 0.9995, seed: int = 42) -> BasePolicy:
    """Instantiates a policy given name and tuned hyper-parameters."""
    if name == "Random":
        return RandomPolicy(seed=seed)
    elif name == "Popularity":
        return PopularityPolicy()
    elif name == "Department Heuristic":
        return DepartmentHeuristicPolicy()
    elif name == "Recently Viewed + Popularity":
        return RecentlyViewedPopularityPolicy()
    elif name == "Epsilon-Greedy":
        return EpsilonGreedyPolicy(epsilon=0.10, seed=seed)
    elif name == "Cold LinUCB":
        return ColdLinUCBPolicy(alpha=alpha)
    elif name == "Warm LinUCB":
        return WarmLinUCBPolicy(alpha=alpha, n_cap=10000, lambda_reg=1.0)
    elif name == "Cold LinTS":
        return ColdLinTSPolicy(v_variance=0.25, seed=seed)
    elif name == "Warm LinTS":
        return WarmLinTSPolicy(v_variance=0.25, n_cap=10000, lambda_reg=1.0, seed=seed)
    elif name == "Discounted LinUCB":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=gamma)
    elif name == "Oracle":
        return OraclePolicy()
    else:
        raise ValueError(f"Unknown policy name: {name}")


def run_single_seed_paired(
    seed: int,
    policy_names: List[str],
    drift_mode: str = "abrupt",
    difficulty: str = "medium",
    is_pipeline: bool = False,
    alpha: float = 0.5,
    gamma: float = 0.9995,
    horizon: int = HORIZON
) -> Dict[str, Any]:
    """
    Executes a single simulation run across all policies using Common Random Numbers (paired design).
    """
    env = EnvironmentV2(
        difficulty=difficulty,
        drift_mode=drift_mode,
        drift_round=horizon // 2,
        horizon=horizon,
        seed=seed
    )

    # Initialize policy instances
    policies: Dict[str, BasePolicy] = {}
    for name in policy_names:
        pol = build_policy(name, alpha=alpha, gamma=gamma, seed=seed)
        if is_pipeline:
            pol = ProductionPipelineWrapper(pol, candidate_pool_size=50, min_distinct_cats=2)
        policies[name] = pol

    # Tracking metrics per policy
    metrics = {
        name: {
            "cumulative_reward": 0.0,
            "cumulative_regret": 0.0,
            "slates_clicked": 0,
            "total_clicks": 0,
            "total_carts": 0,
            "total_purchases": 0,
            "total_dislikes": 0,
            "total_returns": 0,
            "total_removals": 0,
            "total_distinct_cats": 0,
            "recommended_pids": set(),
            "cold_start_impressions": 0,
            "early_regret_1k": 0.0,
            "post_drift_regret": 0.0,
            "cold_user_regret_sum": 0.0,
            "cold_user_rounds": 0,
            "total_step_time_ms": 0.0,
            "regret_trace": []
        }
        for name in policy_names
    }

    drift_start = horizon // 2
    drift_end = drift_start + 2000

    # Simulation loop
    for t in range(horizon):
        user = env.user_manager.sample_active_user(t)
        av_pids = env.get_available_product_ids(t)
        u_ctx = env.build_user_context(user)
        f_mat = env.build_feature_matrix_for_user(user, u_ctx)
        e_vals = env.compute_all_expected_item_values(user, t)
        is_cold_user = (user.total_sessions <= 3)

        for name, policy in policies.items():
            t_start = time.perf_counter()
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

            t_update_start = time.perf_counter()
            policy.update(fb)
            t_total = t_select + (time.perf_counter() - t_update_start) * 1000.0

            # Record stats
            m = metrics[name]
            m["cumulative_reward"] += fb.realized_reward
            m["cumulative_regret"] += fb.expected_regret
            m["total_step_time_ms"] += t_total

            if len(fb.clicked_pids) > 0:
                m["slates_clicked"] += 1
            m["total_clicks"] += len(fb.clicked_pids)
            m["total_carts"] += len(fb.cart_pids)
            m["total_purchases"] += len(fb.purchased_pids)
            m["total_dislikes"] += len(fb.disliked_pids)

            for ev in fb.events:
                if ev["event_type"] == "return":
                    m["total_returns"] += 1
                elif ev["event_type"] == "cart_remove":
                    m["total_removals"] += 1

            # Diversity and coverage
            cats = {env.products[env.pid_to_index[pid]]["category"] for pid in slate}
            m["total_distinct_cats"] += len(cats)
            m["recommended_pids"].update(slate)

            # Cold-start products exposure in second half
            if t >= drift_start:
                m["cold_start_impressions"] += sum(1 for pid in slate if pid in env.cold_start_pids)

            # Phased regrets
            if t < 1000:
                m["early_regret_1k"] += fb.expected_regret
            if drift_start <= t < drift_end:
                m["post_drift_regret"] += fb.expected_regret
            if is_cold_user:
                m["cold_user_regret_sum"] += fb.expected_regret
                m["cold_user_rounds"] += 1

            # Trace sampling
            if (t + 1) % SAMPLE_INTERVAL == 0 or t == horizon - 1:
                m["regret_trace"].append(float(m["cumulative_regret"]))

    # Final summary per policy for this seed
    results = {}
    for name, m in metrics.items():
        total_clicks = max(1, m["total_clicks"])
        total_carts = max(1, m["total_carts"])
        total_interactions = max(1, total_clicks + m["total_dislikes"])

        results[name] = {
            "cumulative_reward": float(m["cumulative_reward"]),
            "cumulative_regret": float(m["cumulative_regret"]),
            "slate_ctr": float(m["slates_clicked"] / horizon),
            "cart_rate": float(m["total_carts"] / total_clicks),
            "conversion_rate": float(m["total_purchases"] / total_carts),
            "negative_feedback_rate": float((m["total_dislikes"] + m["total_returns"] + m["total_removals"]) / total_interactions),
            "diversity": float(m["total_distinct_cats"] / horizon),
            "coverage": float(len(m["recommended_pids"]) / env.n_products),
            "early_regret_1k": float(m["early_regret_1k"]),
            "post_drift_regret": float(m["post_drift_regret"]),
            "cold_user_regret": float(m["cold_user_regret_sum"] / max(1, m["cold_user_rounds"])),
            "new_product_exposure": float(m["cold_start_impressions"] / max(1, (horizon - drift_start) * 10)),
            "mean_step_latency_ms": float(m["total_step_time_ms"] / horizon),
            "regret_trace": m["regret_trace"]
        }

    return {"seed": seed, "policies": results}


def compute_paired_statistics(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired difference statistics between Policy A and Baseline Policy B:
      - Mean diff: E[A - B]
      - 95% bootstrap CI on the mean difference
      - Wilcoxon signed-rank test p-value (two-sided)
      - Cohen's d effect size
    """
    diffs = sample_a - sample_b
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 1e-6
    cohen_d = float(mean_diff / max(1e-6, std_diff))

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(diffs)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(np.mean(diffs[idx]))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Wilcoxon signed-rank test
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
        "p_value": p_val
    }


def run_tuning(tuning_seeds: List[int], workers: int = 10) -> Dict[str, Any]:
    """Sweeps alpha and gamma on tuning seeds in parallel to select optimal parameters."""
    print("--- Running Hyper-Parameter Tuning Sweep ---")
    alpha_candidates = [0.1, 0.25, 0.5, 1.0]
    gamma_candidates = [0.999, 0.9995, 0.9999, 1.0]

    tasks = []
    for seed in tuning_seeds:
        for a in alpha_candidates:
            tasks.append((seed, ["Cold LinUCB"], "abrupt", "medium", False, a, 0.9995, 10000))
        for g in gamma_candidates:
            tasks.append((seed, ["Discounted LinUCB"], "abrupt", "medium", False, 0.5, g, 10000))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_seed_paired, *t) for t in tasks]
        results_list = [f.result() for f in futures]

    alpha_results = {a: [] for a in alpha_candidates}
    gamma_results = {g: [] for g in gamma_candidates}

    for task_tuple, res in zip(tasks, results_list):
        pol_name = task_tuple[1][0]
        if pol_name == "Cold LinUCB":
            a = task_tuple[5]
            alpha_results[a].append(res["policies"]["Cold LinUCB"]["cumulative_regret"])
        else:
            g = task_tuple[6]
            gamma_results[g].append(res["policies"]["Discounted LinUCB"]["cumulative_regret"])

    alpha_means = {a: float(np.mean(alpha_results[a])) for a in alpha_candidates}
    gamma_means = {g: float(np.mean(gamma_results[g])) for g in gamma_candidates}

    best_alpha = min(alpha_means.items(), key=lambda x: x[1])[0]
    best_gamma = min(gamma_means.items(), key=lambda x: x[1])[0]

    print(f"Alpha sweep: {alpha_means} -> Best alpha: {best_alpha}")
    print(f"Gamma sweep: {gamma_means} -> Best gamma: {best_gamma}")

    return {
        "best_alpha": best_alpha,
        "best_gamma": best_gamma,
        "alpha_sweep": alpha_means,
        "gamma_sweep": gamma_means
    }


def run_benchmark_suite():
    """Main benchmark orchestration routine."""
    print("================================================================")
    print("SmartCart AI — Phase 11 Calibrated Simulator v2 Benchmark Suite")
    print("================================================================")

    t_start = time.time()

    workers = min(10, os.cpu_count() or 4)

    # Step 1: Tuning on separate seeds
    tuning_data = run_tuning(TUNING_SEEDS, workers=workers)
    best_alpha = tuning_data["best_alpha"]
    best_gamma = tuning_data["best_gamma"]

    all_policies = [
        "Random",
        "Popularity",
        "Department Heuristic",
        "Recently Viewed + Popularity",
        "Epsilon-Greedy",
        "Cold LinUCB",
        "Warm LinUCB",
        "Cold LinTS",
        "Warm LinTS",
        "Discounted LinUCB",
        "Oracle"
    ]

    # Step 2: Primary Benchmark: Abrupt Drift, Medium difficulty, 30 Evaluation Seeds
    print(f"\n--- Running Primary Benchmark (30 seeds, Abrupt Drift, Medium Difficulty, {HORIZON} rounds) ---")
    workers = min(10, os.cpu_count() or 4)

    tasks_primary = [
        (s, all_policies, "abrupt", "medium", False, best_alpha, best_gamma, HORIZON)
        for s in EVAL_SEEDS
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_seed_paired, *t) for t in tasks_primary]
        primary_runs = [f.result() for f in futures]

    # Aggregate primary benchmark across seeds
    primary_aggregated = {}
    for name in all_policies:
        runs_for_policy = [r["policies"][name] for r in primary_runs]
        agg_m = {}
        for metric_key in runs_for_policy[0].keys():
            if metric_key == "regret_trace":
                traces = np.array([p["regret_trace"] for p in runs_for_policy])
                agg_m["regret_trace_mean"] = [float(x) for x in np.mean(traces, axis=0)]
                continue
            vals = np.array([p[metric_key] for p in runs_for_policy])
            mean_val = float(np.mean(vals))
            ci_low = float(np.percentile(vals, 2.5))
            ci_high = float(np.percentile(vals, 97.5))
            agg_m[metric_key] = {
                "mean": mean_val,
                "ci_95": [ci_low, ci_high],
                "std": float(np.std(vals, ddof=1))
            }
        primary_aggregated[name] = agg_m

    # Paired statistical comparisons vs Cold LinUCB and vs Popularity
    print("\n--- Computing Paired Differences and Hypothesis Tests ---")
    paired_vs_cold_linucb = {}
    paired_vs_popularity = {}

    cold_linucb_regrets = np.array([r["policies"]["Cold LinUCB"]["cumulative_regret"] for r in primary_runs])
    pop_regrets = np.array([r["policies"]["Popularity"]["cumulative_regret"] for r in primary_runs])

    p_values_for_holm = []
    hypothesis_keys = []

    for name in all_policies:
        pol_regrets = np.array([r["policies"][name]["cumulative_regret"] for r in primary_runs])

        # vs Cold LinUCB
        stat_cold = compute_paired_statistics(pol_regrets, cold_linucb_regrets)
        paired_vs_cold_linucb[name] = stat_cold
        p_values_for_holm.append(stat_cold["p_value"])
        hypothesis_keys.append(f"{name}_vs_ColdLinUCB")

        # vs Popularity
        stat_pop = compute_paired_statistics(pol_regrets, pop_regrets)
        paired_vs_popularity[name] = stat_pop

    # Holm-Bonferroni correction
    m_tests = len(p_values_for_holm)
    sorted_pairs = sorted(zip(p_values_for_holm, hypothesis_keys))
    corrected_sig = {}
    for rank, (pval, key) in enumerate(sorted_pairs):
        alpha_crit = 0.05 / (m_tests - rank)
        is_sig = bool(pval < alpha_crit)
        corrected_sig[key] = {
            "p_value": pval,
            "alpha_crit": alpha_crit,
            "significant_holm": is_sig
        }
        # Annotate
        policy_name = key.replace("_vs_ColdLinUCB", "")
        paired_vs_cold_linucb[policy_name]["significant_holm"] = is_sig

    # Step 3: Production Pipeline Mode (10 seeds)
    print("\n--- Running Production Pipeline Mode (10 seeds, Abrupt Drift) ---")
    pipe_seeds = EVAL_SEEDS[:10]
    tasks_pipeline = [
        (s, all_policies, "abrupt", "medium", True, best_alpha, best_gamma, HORIZON)
        for s in pipe_seeds
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures_pipe = [executor.submit(run_single_seed_paired, *t) for t in tasks_pipeline]
        pipe_runs = [f.result() for f in futures_pipe]

    pipeline_aggregated = {}
    for name in all_policies:
        runs_for_policy = [r["policies"][name] for r in pipe_runs]
        pipeline_aggregated[name] = {
            "cumulative_regret": float(np.mean([p["cumulative_regret"] for p in runs_for_policy])),
            "cumulative_reward": float(np.mean([p["cumulative_reward"] for p in runs_for_policy])),
            "slate_ctr": float(np.mean([p["slate_ctr"] for p in runs_for_policy])),
            "diversity": float(np.mean([p["diversity"] for p in runs_for_policy])),
            "coverage": float(np.mean([p["coverage"] for p in runs_for_policy])),
        }

    # Step 4: Robustness Scenarios: No Drift & Gradual Drift (10 seeds each)
    print("\n--- Running Robustness Scenarios (No Drift, Gradual Drift, Easy, Hard) ---")
    scenarios_to_run = [
        ("no_drift", "none", "medium"),
        ("gradual_drift", "gradual", "medium"),
        ("difficulty_easy", "abrupt", "easy"),
        ("difficulty_hard", "abrupt", "hard"),
    ]

    robustness_results = {}
    core_subset = ["Random", "Popularity", "Cold LinUCB", "Warm LinUCB", "Discounted LinUCB", "Oracle"]

    for sc_name, d_mode, diff in scenarios_to_run:
        tasks_sc = [
            (s, core_subset, d_mode, diff, False, best_alpha, best_gamma, HORIZON)
            for s in EVAL_SEEDS[:10]
        ]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures_sc = [executor.submit(run_single_seed_paired, *t) for t in tasks_sc]
            sc_runs = [f.result() for f in futures_sc]

        sc_agg = {}
        for name in core_subset:
            runs_p = [r["policies"][name] for r in sc_runs]
            sc_agg[name] = {
                "cumulative_regret": float(np.mean([p["cumulative_regret"] for p in runs_p])),
                "cumulative_reward": float(np.mean([p["cumulative_reward"] for p in runs_p])),
                "slate_ctr": float(np.mean([p["slate_ctr"] for p in runs_p])),
                "post_drift_regret": float(np.mean([p["post_drift_regret"] for p in runs_p])),
            }
        robustness_results[sc_name] = sc_agg

    total_time = time.time() - t_start
    print(f"\nBenchmark suite completed successfully in {total_time:.2f}s ({total_time / 60:.2f} minutes)!")

    # Master results dictionary
    master_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": "pre-phase11",
            "horizon": HORIZON,
            "n_evaluation_seeds": len(EVAL_SEEDS),
            "tuning_seeds": TUNING_SEEDS,
            "catalog_size": 262,
            "cold_start_items": 12,
            "best_alpha": best_alpha,
            "best_gamma": best_gamma,
            "tuning_sweep": tuning_data,
            "total_benchmark_runtime_seconds": round(total_time, 2)
        },
        "primary_benchmark": primary_aggregated,
        "paired_statistics": {
            "vs_cold_linucb": paired_vs_cold_linucb,
            "vs_popularity": paired_vs_popularity
        },
        "production_pipeline": pipeline_aggregated,
        "robustness_scenarios": robustness_results
    }

    # Save to JSON
    out_dir = ROOT_DIR / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sim_v2_master_results.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    print(f"Saved master benchmark results to {out_path}")
    return master_results


if __name__ == "__main__":
    run_benchmark_suite()
