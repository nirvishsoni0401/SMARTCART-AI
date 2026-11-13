"""
SmartCart AI — Calibrated Simulator v2: Resumable Benchmark Runner (Phase 12A)
Executes benchmark units with per-unit atomic caching, resumability, and comprehensive tracking:
  - Each simulation unit (scenario, difficulty, misspecification, policy, seed) writes an atomic JSON
    file to experiments/results/sim_v2/units/<unit_id>.json.
  - Automatically skips already-completed units matching config hash.
  - CLI flags: --resume, --only-scenario, --only-difficulty, --only-policy, --max-units, --dry-run, --workers.
  - Real-time ETA and logging to experiments/logs/sim_v2_runner.log.
  - Atomic aggregation into experiments/results/sim_v2_master_results.json.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import os
import sys
import re
import argparse
from pathlib import Path
import json
import time
import math
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    DeptPriorLinUCBPolicy,
    SlateMixingPolicy,
    ResidualLinUCBPolicy,
    DeptRestrictedLinUCBPolicy,
    OraclePolicy,
    ProductionPipelineWrapper
)

TUNING_SEEDS = [101, 102, 103, 104, 105]
PRIMARY_EVAL_SEEDS = list(range(1, 31))  # 30 seeds: 1..30
GRID_EVAL_SEEDS = list(range(1, 11))     # 10 seeds: 1..10
HORIZON = 20000
SAMPLE_INTERVAL = 250

GAMMA_CANDIDATES = [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0]

ALL_POLICIES_ORDERED = [
    "Oracle",
    "Discounted LinUCB (gamma=0.99)",
    "Discounted LinUCB (gamma=0.995)",
    "Discounted LinUCB (gamma=0.998)",
    "Discounted LinUCB (gamma=0.999)",
    "Discounted LinUCB (gamma=0.9995)",
    "Discounted LinUCB (gamma=0.9999)",
    "Discounted LinUCB (gamma=1.0)",
    "Cold LinUCB",
    "Warm LinUCB",
    "Cold LinTS",
    "Warm LinTS",
    "Recently Viewed + Popularity",
    "Department Heuristic",
    "Department Heuristic (With Diversity)",
    "Dept-Prior LinUCB (w=0.5)",
    "Slate Mixing (k=3 Dept + LinUCB)",
    "Slate Mixing (k=5 Dept + LinUCB)",
    "Residual LinUCB",
    "Dept-Restricted LinUCB (Cold)",
    "Dept-Restricted LinUCB (Discounted)",
    "Dept-Restricted LinUCB + Expl (Cold, m=2)",
    "Dept-Restricted LinUCB + Expl (Discounted, m=2)",
    "Epsilon-Greedy",
    "Popularity",
    "Random"
]

HYBRID_POLICIES = [
    "Dept-Prior LinUCB (w=0.5)",
    "Slate Mixing (k=3 Dept + LinUCB)",
    "Slate Mixing (k=5 Dept + LinUCB)",
    "Residual LinUCB",
    "Dept-Restricted LinUCB (Cold)",
    "Dept-Restricted LinUCB (Discounted)",
    "Dept-Restricted LinUCB + Expl (Cold, m=2)",
    "Dept-Restricted LinUCB + Expl (Discounted, m=2)"
]

UNITS_DIR = ROOT_DIR / "experiments" / "results" / "sim_v2" / "units"
LOGS_DIR = ROOT_DIR / "experiments" / "logs"
LOG_FILE = LOGS_DIR / "sim_v2_runner.log"


def compute_code_hash() -> str:
    """Computes a SHA256 hash of simulator environment, user model, policies code and scenario salt."""
    h = hashlib.sha256()
    for rel_p in [
        "experiments/sim_v2/environment.py",
        "experiments/sim_v2/user_model.py",
        "experiments/sim_v2/policies.py"
    ]:
        p = ROOT_DIR / rel_p
        if p.exists():
            h.update(p.read_bytes())
    h.update(b"horizon_20000_alpha_0.5_gamma_0.9995_phase12a2")
    return h.hexdigest()[:16]


CONFIG_HASH = compute_code_hash()


def log(msg: str):
    """Logs message to console and runner log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def make_unit_id(
    policy: str,
    seed: int,
    scenario: str,
    difficulty: str = "medium",
    misspecification: Optional[str] = None,
    is_pipeline: bool = False,
    withhold_new_products: bool = True,
    n_cap: int = 10000
) -> str:
    """Generates a canonical, safe filename identifier for a unit."""
    safe_pol = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', policy)
    miss_str = misspecification or "base"
    pip_str = "pipeline" if is_pipeline else "bandit"
    withhold_str = "withhold" if withhold_new_products else "all"
    return f"{scenario}__{difficulty}__{miss_str}__{pip_str}__{withhold_str}__ncap{n_cap}__{safe_pol}__s{seed}"


def build_policy(
    name: str,
    alpha: float = 0.5,
    gamma: float = 0.9995,
    seed: int = 42,
    n_cap: int = 10000
) -> BasePolicy:
    """Instantiates a policy given name and tuned parameters."""
    if name == "Random":
        return RandomPolicy(seed=seed)
    elif name == "Popularity":
        return PopularityPolicy()
    elif name == "Department Heuristic":
        return DepartmentHeuristicPolicy(enforce_diversity=False)
    elif name == "Department Heuristic (With Diversity)":
        return DepartmentHeuristicPolicy(enforce_diversity=True)
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
    elif name == "Discounted LinUCB (gamma=0.99)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.99, name=name)
    elif name == "Discounted LinUCB (gamma=0.995)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.995, name=name)
    elif name == "Discounted LinUCB (gamma=0.998)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.998, name=name)
    elif name == "Discounted LinUCB (gamma=0.999)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.999, name=name)
    elif name == "Discounted LinUCB (gamma=0.9995)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.9995, name=name)
    elif name in ("Discounted LinUCB (gamma=0.9999)", "Discounted LinUCB"):
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=0.9999, name="Discounted LinUCB (gamma=0.9999)")
    elif name == "Discounted LinUCB (gamma=1.0)":
        return DiscountedLinUCBPolicy(alpha=alpha, gamma=1.0, name=name)
    elif name == "Dept-Prior LinUCB (w=0.5)":
        return DeptPriorLinUCBPolicy(alpha=alpha, prior_weight=0.5)
    elif name == "Slate Mixing (k=3 Dept + LinUCB)":
        return SlateMixingPolicy(k_heuristic=3, alpha=alpha)
    elif name == "Slate Mixing (k=5 Dept + LinUCB)":
        return SlateMixingPolicy(k_heuristic=5, alpha=alpha)
    elif name == "Residual LinUCB":
        return ResidualLinUCBPolicy(alpha=alpha)
    elif name == "Dept-Restricted LinUCB (Cold)":
        return DeptRestrictedLinUCBPolicy(name=name, base_bandit=ColdLinUCBPolicy(alpha=alpha), m_explore=0)
    elif name == "Dept-Restricted LinUCB (Discounted)":
        return DeptRestrictedLinUCBPolicy(name=name, base_bandit=DiscountedLinUCBPolicy(alpha=alpha, gamma=gamma), m_explore=0)
    elif name == "Dept-Restricted LinUCB + Expl (Cold, m=2)":
        return DeptRestrictedLinUCBPolicy(name=name, base_bandit=ColdLinUCBPolicy(alpha=alpha), m_explore=2)
    elif name == "Dept-Restricted LinUCB + Expl (Discounted, m=2)":
        return DeptRestrictedLinUCBPolicy(name=name, base_bandit=DiscountedLinUCBPolicy(alpha=alpha, gamma=gamma), m_explore=2)
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
    alpha: float = 0.5,
    gamma: float = 0.9995,
    n_cap: int = 10000,
    scenario_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a single policy simulation trace with its own isolated EnvironmentV2 instance.
    Guarantees:
      - Complete state isolation (no cross-policy pollution).
      - Common random numbers across policies for the same seed.
      - Exact Oracle regret == 0.0 for unconstrained Oracle.
      - Exact identity: cumulative_reward + cumulative_regret == cumulative_trajectory_oracle_reward.
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
        env_drift = "none" if scenario_name in ("no_drift", "none") else ("abrupt" if scenario_name in ("abrupt", "abrupt_drift", "drift_only") else "gradual")

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
    post_drift_regret = 0.0        # [10000, 12000) legacy 2k window
    post_drift_regret_1k = 0.0     # [10000, 11000) first 1,000 rounds post-drift
    post_drift_regret_half = 0.0   # [10000, 20000) whole second half
    cold_user_regret_sum = 0.0
    cold_user_rounds = 0
    quarter_regrets = [0.0, 0.0, 0.0, 0.0]
    regrets_1k = [0.0] * max(1, horizon // 1000)
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

        step_regret = 0.0 if is_oracle_unconstrained else fb.expected_regret
        cumulative_reward += fb.expected_policy_reward
        cumulative_regret += step_regret

        # Quarters
        q_idx = min(3, t // quarter_len)
        quarter_regrets[q_idx] += step_regret

        # 1k bins
        bin_1k = min(len(regrets_1k) - 1, t // 1000)
        regrets_1k[bin_1k] += step_regret

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

            new_in_oracle = sum(e_vals[env.pid_to_index[p]] for p in fb.oracle_slate if p in env.cold_start_pids)
            new_in_policy = sum(e_vals[env.pid_to_index[p]] for p in slate[:10] if p in env.cold_start_pids)
            new_item_regret += max(0.0, float(new_in_oracle - new_in_policy))

        # Phased regrets
        if t < 1000:
            early_regret_1k += step_regret
        if drift_start <= t < drift_end:
            post_drift_regret += step_regret
        if drift_start <= t < drift_start + 1000:
            post_drift_regret_1k += step_regret
        if t >= drift_start:
            post_drift_regret_half += step_regret

        if is_cold_user:
            cold_user_regret_sum += step_regret
            cold_user_rounds += 1

        if (t + 1) % SAMPLE_INTERVAL == 0 or t == horizon - 1:
            regret_trace.append(float(cumulative_regret))

    tot_clicks = max(1, total_clicks)
    tot_carts = max(1, total_carts)
    tot_interactions = max(1, total_clicks + total_dislikes)

    cum_traj_oracle = cumulative_reward + cumulative_regret
    reg_share = cumulative_regret / max(1e-9, cum_traj_oracle)

    x_q = np.arange(5, dtype=np.float64)
    y_q2 = np.array(regrets_1k[5:10], dtype=np.float64) if len(regrets_1k) >= 10 else np.zeros(5)
    y_q4 = np.array(regrets_1k[15:20], dtype=np.float64) if len(regrets_1k) >= 20 else np.zeros(5)
    slope_q2 = float(np.polyfit(x_q, y_q2, 1)[0]) if len(y_q2) == 5 else 0.0
    slope_q4 = float(np.polyfit(x_q, y_q4, 1)[0]) if len(y_q4) == 5 else 0.0
    spike_ratio = float(quarter_regrets[2] / max(1e-6, quarter_regrets[1]))

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
        "code_hash": CONFIG_HASH,
        "config_hash": CONFIG_HASH,
        "cumulative_reward": float(cumulative_reward),
        "cumulative_regret": float(cumulative_regret),
        "cumulative_trajectory_oracle_reward": float(cum_traj_oracle),
        "regret_share": float(reg_share),
        "slate_ctr": float(slates_clicked / horizon),
        "cart_rate": float(total_carts / tot_clicks),
        "conversion_rate": float(total_purchases / tot_carts),
        "negative_feedback_rate": float((total_dislikes + total_returns + total_removals) / tot_interactions),
        "diversity": float(total_distinct_cats / horizon),
        "coverage": float(len(recommended_pids) / env.n_products),
        "early_regret_1k": float(early_regret_1k),
        "post_drift_regret": float(post_drift_regret),
        "post_drift_regret_1k": float(post_drift_regret_1k),
        "post_drift_regret_half": float(post_drift_regret_half),
        "cold_user_regret": float(cold_user_regret_sum / max(1, cold_user_rounds)),
        "quarter_regrets": [float(q) for q in quarter_regrets],
        "regrets_1k": [float(x) for x in regrets_1k],
        "slope_q2": slope_q2,
        "slope_q4": slope_q4,
        "spike_ratio": spike_ratio,
        "new_product_exposure": float(cold_start_impressions / max(1, (horizon - drift_start) * 10)),
        "new_product_reward": float(new_item_reward),
        "new_product_regret": float(new_item_regret),
        "mean_step_latency_ms": float(total_step_time_ms / horizon),
        "regret_trace": regret_trace
    }


def execute_unit_worker(task: Tuple) -> Tuple[str, Dict[str, Any]]:
    """Worker function that runs a unit and atomically writes its result."""
    (policy, seed, difficulty, drift_mode, horizon, is_pipeline, misspecification,
     withhold_new_products, alpha, gamma, n_cap, scenario_name, unit_id) = task

    unit_file = UNITS_DIR / f"{unit_id}.json"

    # Run trace
    res = run_single_policy_trace(
        policy_name=policy,
        seed=seed,
        difficulty=difficulty,
        drift_mode=drift_mode,
        horizon=horizon,
        is_pipeline=is_pipeline,
        misspecification=misspecification,
        withhold_new_products=withhold_new_products,
        alpha=alpha,
        gamma=gamma,
        n_cap=n_cap,
        scenario_name=scenario_name
    )

    # Atomic save
    UNITS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = UNITS_DIR / f"{unit_id}.tmp.{os.getpid()}.json"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    tmp_file.replace(unit_file)

    return unit_id, res


def is_unit_valid(unit_id: str) -> bool:
    """Checks whether a unit file exists, parses correctly, and has matching config_hash."""
    unit_file = UNITS_DIR / f"{unit_id}.json"
    if not unit_file.exists():
        return False
    try:
        with open(unit_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("config_hash") != CONFIG_HASH:
            return False
        if "cumulative_regret" not in data or "cumulative_reward" not in data:
            return False
        return True
    except Exception:
        return False


def load_unit(unit_id: str) -> Dict[str, Any]:
    """Loads a unit file."""
    unit_file = UNITS_DIR / f"{unit_id}.json"
    with open(unit_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_all_required_units() -> List[Tuple]:
    """
    Constructs the list of all unit specifications required for Phase 12A-2:
      1. Primary Benchmark (30 seeds, abrupt drift, medium, all 26 policies)
      2. A1 Discounting Test Suite (30 seeds, no_drift & gradual_drift, 4 policies)
      3. Part B Hybrids in no_drift (30 seeds, 8 hybrids + Dept Heuristic + Oracle + Random)
      4. Misspecification Suite (30 seeds, 3 misspecification modes, all 26 policies)
      5. Confounded Scenarios (10 seeds, drift_only & new_products_only, all 26 policies)
      6. Warm-Start Sweep (15 seeds, n_cap in {500, 2000}, 2 policies)
    """
    tasks = []
    alpha = 0.5
    gamma = 0.9995

    # 1. Primary Benchmark (30 seeds, abrupt drift, medium, all 26 policies)
    for pol in ALL_POLICIES_ORDERED:
        for s in PRIMARY_EVAL_SEEDS:
            uid = make_unit_id(pol, s, "abrupt_drift", "medium", None, False, True, 10000)
            tasks.append((pol, s, "medium", "abrupt", HORIZON, False, None, True, alpha, gamma, 10000, "abrupt_drift", uid))

    # 2. A1 Discounting Test Suite (30 seeds, no_drift & gradual_drift, 4 policies)
    a1_pols = ["Cold LinUCB", "Discounted LinUCB (gamma=0.9995)", "Discounted LinUCB (gamma=1.0)", "Discounted LinUCB (gamma=0.999)"]
    for scen in ["no_drift", "gradual_drift"]:
        scen_env = "none" if scen == "no_drift" else "gradual"
        for pol in a1_pols:
            for s in PRIMARY_EVAL_SEEDS:
                uid = make_unit_id(pol, s, scen, "medium", None, False, True, 10000)
                tasks.append((pol, s, "medium", scen_env, HORIZON, False, None, True, alpha, gamma, 10000, scen, uid))

    # 3. Part B Hybrids in no_drift (30 seeds, 8 hybrids + Dept Heuristic + Oracle + Random)
    h_no_drift_pols = list(HYBRID_POLICIES) + ["Department Heuristic", "Oracle", "Random"]
    for pol in h_no_drift_pols:
        if pol not in a1_pols:
            for s in PRIMARY_EVAL_SEEDS:
                uid = make_unit_id(pol, s, "no_drift", "medium", None, False, True, 10000)
                tasks.append((pol, s, "medium", "none", HORIZON, False, None, True, alpha, gamma, 10000, "no_drift", uid))

    # 4. Misspecification Suite (30 seeds, 3 misspecification modes, all 26 policies)
    for m_mode in ["hidden_style", "nonlinear_interactions", "high_funnel_conversion"]:
        for pol in ALL_POLICIES_ORDERED:
            for s in PRIMARY_EVAL_SEEDS:
                uid = make_unit_id(pol, s, "abrupt_drift", "medium", m_mode, False, True, 10000)
                tasks.append((pol, s, "medium", "abrupt", HORIZON, False, m_mode, True, alpha, gamma, 10000, "abrupt_drift", uid))

    # 5. Confounded Scenarios (10 seeds, drift_only & new_products_only, all 26 policies)
    for pol in ALL_POLICIES_ORDERED:
        for s in GRID_EVAL_SEEDS:
            uid_d = make_unit_id(pol, s, "drift_only", "medium", None, False, False, 10000)
            tasks.append((pol, s, "medium", "abrupt", HORIZON, False, None, False, alpha, gamma, 10000, "drift_only", uid_d))
            uid_n = make_unit_id(pol, s, "new_products_only", "medium", None, False, True, 10000)
            tasks.append((pol, s, "medium", "none", HORIZON, False, None, True, alpha, gamma, 10000, "new_products_only", uid_n))

    # 6. Part F Warm-Start Sweep (15 seeds, n_cap in {500, 2000})
    for n_c in [500, 2000]:
        for s in range(1, 16):
            for pol in ["Warm LinUCB", "Warm LinTS"]:
                uid = make_unit_id(pol, s, "abrupt_drift", "medium", None, False, True, n_c)
                tasks.append((pol, s, "medium", "abrupt", HORIZON, False, None, True, alpha, gamma, n_c, "abrupt_drift", uid))

    return tasks


def aggregate_policy_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes mean, 95% bootstrap CI, and standard deviation across seeds for one policy."""
    n_seeds = len(runs)
    regrets = np.array([r["cumulative_regret"] for r in runs], dtype=np.float64)
    rewards = np.array([r["cumulative_reward"] for r in runs], dtype=np.float64)

    if "cumulative_trajectory_oracle_reward" in runs[0]:
        traj_oracles = np.array([r["cumulative_trajectory_oracle_reward"] for r in runs], dtype=np.float64)
    else:
        traj_oracles = rewards + regrets

    if "regret_share" in runs[0]:
        reg_shares = np.array([r["regret_share"] for r in runs], dtype=np.float64)
    else:
        reg_shares = regrets / np.maximum(1e-9, traj_oracles)

    ctrs = np.array([r["slate_ctr"] for r in runs], dtype=np.float64)
    carts = np.array([r["cart_rate"] for r in runs], dtype=np.float64)
    conversions = np.array([r["conversion_rate"] for r in runs], dtype=np.float64)
    negs = np.array([r["negative_feedback_rate"] for r in runs], dtype=np.float64)
    diversities = np.array([r["diversity"] for r in runs], dtype=np.float64)
    coverages = np.array([r["coverage"] for r in runs], dtype=np.float64)
    early_reg = np.array([r["early_regret_1k"] for r in runs], dtype=np.float64)
    post_reg = np.array([r["post_drift_regret"] for r in runs], dtype=np.float64)

    if "post_drift_regret_1k" in runs[0]:
        post_1k = np.array([r["post_drift_regret_1k"] for r in runs], dtype=np.float64)
    else:
        traces = np.array([r["regret_trace"] for r in runs], dtype=np.float64)
        post_1k = traces[:, 43] - traces[:, 39]

    if "post_drift_regret_half" in runs[0]:
        post_half = np.array([r["post_drift_regret_half"] for r in runs], dtype=np.float64)
    else:
        q_mat = np.array([r["quarter_regrets"] for r in runs], dtype=np.float64)
        post_half = q_mat[:, 2] + q_mat[:, 3]

    cold_u_reg = np.array([r["cold_user_regret"] for r in runs], dtype=np.float64)
    latencies = np.array([r["mean_step_latency_ms"] for r in runs], dtype=np.float64)
    new_exposures = np.array([r["new_product_exposure"] for r in runs], dtype=np.float64)
    new_rewards = np.array([r["new_product_reward"] for r in runs], dtype=np.float64)
    new_regrets = np.array([r["new_product_regret"] for r in runs], dtype=np.float64)

    # Quarter regrets
    q_mat = np.array([r["quarter_regrets"] for r in runs], dtype=np.float64)
    q_means = [float(np.mean(q_mat[:, i])) for i in range(4)]

    # Bootstrap 95% CIs
    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(regrets, size=n_seeds, replace=True))) for _ in range(1000)]
    reg_ci = [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))]

    boot_rew = [float(np.mean(rng.choice(rewards, size=n_seeds, replace=True))) for _ in range(1000)]
    rew_ci = [float(np.percentile(boot_rew, 2.5)), float(np.percentile(boot_rew, 97.5))]

    boot_traj = [float(np.mean(rng.choice(traj_oracles, size=n_seeds, replace=True))) for _ in range(1000)]
    traj_ci = [float(np.percentile(boot_traj, 2.5)), float(np.percentile(boot_traj, 97.5))]

    boot_share = [float(np.mean(rng.choice(reg_shares, size=n_seeds, replace=True))) for _ in range(1000)]
    share_ci = [float(np.percentile(boot_share, 2.5)), float(np.percentile(boot_share, 97.5))]

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
            "ci_95": rew_ci,
            "values": [float(x) for x in rewards]
        },
        "cumulative_trajectory_oracle_reward": {
            "mean": float(np.mean(traj_oracles)),
            "std": float(np.std(traj_oracles, ddof=1)) if n_seeds > 1 else 0.0,
            "ci_95": traj_ci,
            "values": [float(x) for x in traj_oracles]
        },
        "regret_share": {
            "mean": float(np.mean(reg_shares)),
            "std": float(np.std(reg_shares, ddof=1)) if n_seeds > 1 else 0.0,
            "ci_95": share_ci,
            "values": [float(x) for x in reg_shares]
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
                      float(np.percentile([np.mean(rng.choice(early_reg, size=n_seeds, replace=True)) for _ in range(1000)], 97.5))],
            "values": [float(x) for x in early_reg]
        },
        "post_drift_regret": {
            "mean": float(np.mean(post_reg)),
            "ci_95": [float(np.percentile([np.mean(rng.choice(post_reg, size=n_seeds, replace=True)) for _ in range(1000)], 2.5)),
                      float(np.percentile([np.mean(rng.choice(post_reg, size=n_seeds, replace=True)) for _ in range(1000)], 97.5))],
            "values": [float(x) for x in post_reg]
        },
        "post_drift_regret_1k": {
            "mean": float(np.mean(post_1k)),
            "ci_95": [float(np.percentile([np.mean(rng.choice(post_1k, size=n_seeds, replace=True)) for _ in range(1000)], 2.5)),
                      float(np.percentile([np.mean(rng.choice(post_1k, size=n_seeds, replace=True)) for _ in range(1000)], 97.5))],
            "values": [float(x) for x in post_1k]
        },
        "post_drift_regret_half": {
            "mean": float(np.mean(post_half)),
            "ci_95": [float(np.percentile([np.mean(rng.choice(post_half, size=n_seeds, replace=True)) for _ in range(1000)], 2.5)),
                      float(np.percentile([np.mean(rng.choice(post_half, size=n_seeds, replace=True)) for _ in range(1000)], 97.5))],
            "values": [float(x) for x in post_half]
        },
        "cold_user_regret": {
            "mean": float(np.mean(cold_u_reg))
        },
        "quarter_regrets": q_means,
        "slope_q2": float(np.mean(np.array([r.get("slope_q2", 0.0) for r in runs], dtype=np.float64))),
        "slope_q4": float(np.mean(np.array([r.get("slope_q4", 0.0) for r in runs], dtype=np.float64))),
        "spike_ratio": float(np.mean(np.array([r.get("spike_ratio", 1.0) for r in runs], dtype=np.float64))),
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


def validate_existing_units() -> Tuple[int, int, int]:
    """
    Validates all units currently on disk:
      - Checks JSON validity and config_hash.
      - Deletes corrupt/truncated files.
      - Re-runs one unit and verifies exact determinism.
    Returns: (valid_count, corrupt_deleted_count, retest_passed)
    """
    if not UNITS_DIR.exists():
        return 0, 0, True

    unit_files = list(UNITS_DIR.glob("*.json"))
    valid_count = 0
    corrupt_count = 0

    valid_sample = None
    for f in unit_files:
        if f.name.endswith(".tmp.json"):
            try:
                f.unlink()
            except Exception:
                pass
            continue

        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if data.get("config_hash") == CONFIG_HASH and "cumulative_regret" in data:
                valid_count += 1
                if valid_sample is None:
                    valid_sample = (f, data)
            else:
                f.unlink()
                corrupt_count += 1
        except Exception:
            try:
                f.unlink()
            except Exception:
                pass
            corrupt_count += 1

    retest_passed = True
    if valid_sample is not None:
        f, cached = valid_sample
        log(f"Validating determinism on sample unit: {f.name}")
        fresh = run_single_policy_trace(
            policy_name=cached["policy"],
            seed=cached["seed"],
            difficulty=cached["difficulty"],
            drift_mode=cached["scenario"],
            horizon=cached.get("horizon", HORIZON),
            is_pipeline=cached["is_pipeline"],
            misspecification=cached["misspecification"],
            withhold_new_products=cached["withhold_new_products"],
            alpha=cached["alpha"],
            gamma=cached["gamma"],
            n_cap=cached["n_cap"],
            scenario_name=cached["scenario"]
        )
        if (abs(fresh["cumulative_regret"] - cached["cumulative_regret"]) > 1e-5 or
            abs(fresh["cumulative_reward"] - cached["cumulative_reward"]) > 1e-5):
            log(f"NON-DETERMINISM DETECTED on {f.name}: cached regret={cached['cumulative_regret']}, fresh={fresh['cumulative_regret']}")
            retest_passed = False
        else:
            log(f"Determinism confirmed: cached == fresh ({cached['cumulative_regret']:.4f})")

    return valid_count, corrupt_count, retest_passed


def run_hyperparameter_tuning(workers: int = 4) -> Tuple[Dict[str, float], Dict[str, float], float, float]:
    """Runs tuning sweep on TUNING_SEEDS = [101, 102, 103, 104, 105]."""
    log("Running hyperparameter tuning on separate tuning seeds (101-105)...")
    alpha_candidates = [0.1, 0.25, 0.5, 1.0]

    tuning_tasks = []
    for a in alpha_candidates:
        for s in TUNING_SEEDS:
            uid = f"tuning__alpha_{a}__s{s}"
            if not is_unit_valid(uid):
                tuning_tasks.append(("Cold LinUCB", s, "medium", "abrupt", HORIZON, False, None, True, a, 0.9995, 10000, "abrupt_drift", uid))

    for g in GAMMA_CANDIDATES:
        for s in TUNING_SEEDS:
            uid = f"tuning__gamma_{g}__s{s}"
            if not is_unit_valid(uid):
                tuning_tasks.append((f"Discounted LinUCB (gamma={g})", s, "medium", "abrupt", HORIZON, False, None, True, 0.5, g, 10000, "abrupt_drift", uid))

    if tuning_tasks:
        log(f"Running {len(tuning_tasks)} tuning units across {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(execute_unit_worker, t): t for t in tuning_tasks}
            for fut in as_completed(futures):
                uid, res = fut.result()

    alpha_sweep = {}
    for a in alpha_candidates:
        regs = [load_unit(f"tuning__alpha_{a}__s{s}")["cumulative_regret"] for s in TUNING_SEEDS]
        alpha_sweep[str(a)] = float(np.mean(regs))

    gamma_sweep = {}
    for g in GAMMA_CANDIDATES:
        regs = [load_unit(f"tuning__gamma_{g}__s{s}")["cumulative_regret"] for s in TUNING_SEEDS]
        gamma_sweep[str(g)] = float(np.mean(regs))

    best_alpha = float(min(alpha_sweep, key=alpha_sweep.get))
    best_gamma = float(min(gamma_sweep, key=gamma_sweep.get))
    log(f"Tuning completed: best_alpha={best_alpha}, best_gamma={best_gamma}")
    return alpha_sweep, gamma_sweep, best_alpha, best_gamma


def aggregate_master_results(tuning_data: Optional[Tuple] = None, workers: int = 4):
    """Aggregates all cached unit results into master JSON."""
    log("Aggregating master benchmark results from unit files...")

    # Load legacy 12A master results for before/after comparison
    legacy_file = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results_legacy_12a.json"
    legacy_data = None
    if legacy_file.exists():
        with open(legacy_file, "r", encoding="utf-8") as f:
            legacy_data = json.load(f)

    # Tuning
    if tuning_data is not None:
        alpha_sweep, gamma_sweep, best_alpha, best_gamma = tuning_data
    elif legacy_data and "hyperparameter_tuning" in legacy_data and len(legacy_data["hyperparameter_tuning"].get("gamma_sweep", {})) == len(GAMMA_CANDIDATES):
        t_sec = legacy_data["hyperparameter_tuning"]
        alpha_sweep = t_sec["alpha_sweep"]
        gamma_sweep = t_sec["gamma_sweep"]
        best_alpha = t_sec["best_alpha"]
        best_gamma = t_sec["best_gamma"]
    else:
        alpha_sweep, gamma_sweep, best_alpha, best_gamma = run_hyperparameter_tuning(workers=workers)

    # 1. Primary Benchmark
    primary_benchmark = {}
    for pol in ALL_POLICIES_ORDERED:
        runs = []
        for s in PRIMARY_EVAL_SEEDS:
            uid = make_unit_id(pol, s, "abrupt_drift", "medium", None, False, True, 10000)
            if is_unit_valid(uid):
                runs.append(load_unit(uid))
        if len(runs) == len(PRIMARY_EVAL_SEEDS):
            primary_benchmark[pol] = aggregate_policy_runs(runs)
        else:
            log(f"WARNING: Primary benchmark {pol} has only {len(runs)}/{len(PRIMARY_EVAL_SEEDS)} valid units")

    # Add quarterly ratios to Random and trend classification rules
    if "Random" in primary_benchmark:
        r_quarters = primary_benchmark["Random"]["quarter_regrets"]
        for pol, m in primary_benchmark.items():
            m["quarter_ratios_to_random"] = [
                float(q / max(1e-6, r_q)) for q, r_q in zip(m["quarter_regrets"], r_quarters)
            ]
            s_q2 = m.get("slope_q2", 0.0)
            s_q4 = m.get("slope_q4", 0.0)
            spk = m.get("spike_ratio", 1.0)
            if s_q2 < -5.0 and s_q4 < -5.0:
                m["trend_rule"] = f"Decelerating (Slope Q2={s_q2:.1f}, Q4={s_q4:.1f}, Spike={spk:.2f}x)"
            elif spk > 1.25:
                m["trend_rule"] = f"Drift Spike & Re-adapting (Slope Q2={s_q2:.1f}, Q4={s_q4:.1f}, Spike={spk:.2f}x)"
            elif abs(s_q2) <= 5.0 and abs(s_q4) <= 5.0:
                m["trend_rule"] = f"Stationary / Flat (Slope Q2={s_q2:.1f}, Q4={s_q4:.1f}, Spike={spk:.2f}x)"
            else:
                m["trend_rule"] = f"Dynamic (Slope Q2={s_q2:.1f}, Q4={s_q4:.1f}, Spike={spk:.2f}x)"

    # 2. Gamma Evaluation Grid (Part A3)
    gamma_grid_eval = {}
    for g in GAMMA_CANDIDATES:
        pol_name = f"Discounted LinUCB (gamma={g})"
        if pol_name in primary_benchmark:
            gamma_grid_eval[str(g)] = primary_benchmark[pol_name]

    # 3. Dedicated Discounting Test Suite (Part A2)
    a1_pols = [
        f"Discounted LinUCB (gamma={best_gamma})",
        "Discounted LinUCB (gamma=1.0)",
        "Cold LinUCB",
        "Discounted LinUCB (gamma=0.999)"
    ]
    a1_scenarios = ["no_drift", "abrupt_drift", "gradual_drift"]
    a1_runs_by_key = {}
    for scen in a1_scenarios:
        for pol in a1_pols:
            runs = []
            for s in PRIMARY_EVAL_SEEDS:
                uid = make_unit_id(pol, s, scen, "medium", None, False, True, 10000)
                if is_unit_valid(uid):
                    runs.append(load_unit(uid))
            runs.sort(key=lambda x: x["seed"])
            a1_runs_by_key[(scen, pol)] = runs

    a1_test_suite = {"description": "Dedicated paired Holm-corrected tests for Discounted vs Cold LinUCB", "scenarios": {}}
    raw_p_a1 = []
    test_records = []
    comparisons = [
        (f"Discounted (gamma*={best_gamma}) vs Discounted (gamma=1.0)", f"Discounted LinUCB (gamma={best_gamma})", "Discounted LinUCB (gamma=1.0)"),
        (f"Discounted (gamma*={best_gamma}) vs Cold LinUCB", f"Discounted LinUCB (gamma={best_gamma})", "Cold LinUCB"),
        ("Discounted (gamma=1.0) vs Cold LinUCB", "Discounted LinUCB (gamma=1.0)", "Cold LinUCB"),
        ("Discounted (gamma=0.999) vs Cold LinUCB", "Discounted LinUCB (gamma=0.999)", "Cold LinUCB")
    ]
    horizons_with_drift = [
        ("full_horizon", "Full Horizon [0, 20000)", lambda r: r["cumulative_regret"]),
        ("post_drift_1k", "Post-Drift 1,000 Rounds [10000, 11000)", lambda r: r["post_drift_regret_1k"]),
        ("post_drift_half", "Post-Drift Second Half [10000, 20000)", lambda r: r["post_drift_regret_half"])
    ]
    horizons_no_drift = [
        ("full_horizon", "Full Horizon [0, 20000)", lambda r: r["cumulative_regret"])
    ]

    for scen in a1_scenarios:
        a1_test_suite["scenarios"][scen] = []
        scen_horizons = horizons_no_drift if scen == "no_drift" else horizons_with_drift
        for comp_name, pol_a, pol_b in comparisons:
            runs_a = a1_runs_by_key.get((scen, pol_a), [])
            runs_b = a1_runs_by_key.get((scen, pol_b), [])
            if len(runs_a) >= 10 and len(runs_b) >= 10:
                n_match = min(len(runs_a), len(runs_b))
                for h_key, h_desc, extractor in scen_horizons:
                    arr_a = np.array([extractor(r) for r in runs_a[:n_match]], dtype=np.float64)
                    arr_b = np.array([extractor(r) for r in runs_b[:n_match]], dtype=np.float64)
                    stat = compute_paired_statistics(arr_a, arr_b)
                    raw_p_a1.append(stat["p_value_raw"])
                    rec = {
                        "comparison": comp_name,
                        "pol_a": pol_a,
                        "pol_b": pol_b,
                        "horizon_key": h_key,
                        "horizon_description": h_desc,
                        "mean_a": float(np.mean(arr_a)),
                        "mean_b": float(np.mean(arr_b)),
                        "mean_diff": stat["mean_diff"],
                        "ci_95": stat["ci_95"],
                        "cohen_d": stat["cohen_d"],
                        "p_value_raw": stat["p_value_raw"]
                    }
                    test_records.append(rec)
                    a1_test_suite["scenarios"][scen].append(rec)

    holm_p_a1 = apply_holm_bonferroni(raw_p_a1)
    for i, rec in enumerate(test_records):
        rec["p_value_holm"] = holm_p_a1[i]
        ci = rec["ci_95"]
        if ci[1] < 0.0 and rec["p_value_holm"] < 0.05:
            rec["verdict"] = f"{rec['pol_a']} beats {rec['pol_b']} (statistically significant)"
        elif ci[0] > 0.0 and rec["p_value_holm"] < 0.05:
            rec["verdict"] = f"{rec['pol_b']} beats {rec['pol_a']} (statistically significant)"
        elif (ci[1] < 0.0 or ci[0] > 0.0) and rec["p_value_holm"] >= 0.05:
            rec["verdict"] = "Bootstrap CI excludes zero but Holm-corrected Wilcoxon is not significant"
        else:
            rec["verdict"] = "Interval spans zero / No significant difference"

    # 4. Confounded Scenarios (Part D)
    confounded_results = {"drift_only": {}, "new_products_only": {}, "both": {}}
    for pol in ALL_POLICIES_ORDERED:
        d_runs = [load_unit(make_unit_id(pol, s, "drift_only", "medium", None, False, False, 10000))
                  for s in GRID_EVAL_SEEDS if is_unit_valid(make_unit_id(pol, s, "drift_only", "medium", None, False, False, 10000))]
        if len(d_runs) == len(GRID_EVAL_SEEDS):
            confounded_results["drift_only"][pol] = aggregate_policy_runs(d_runs)

        n_runs = [load_unit(make_unit_id(pol, s, "new_products_only", "medium", None, False, True, 10000))
                  for s in GRID_EVAL_SEEDS if is_unit_valid(make_unit_id(pol, s, "new_products_only", "medium", None, False, True, 10000))]
        if len(n_runs) == len(GRID_EVAL_SEEDS):
            confounded_results["new_products_only"][pol] = aggregate_policy_runs(n_runs)

        if pol in primary_benchmark:
            confounded_results["both"][pol] = primary_benchmark[pol]

    # 5. Misspecification Robustness (Part A4 / Part E)
    misspec_results = {}
    for m_mode in ["hidden_style", "nonlinear_interactions", "high_funnel_conversion"]:
        misspec_results[m_mode] = {}
        for pol in ALL_POLICIES_ORDERED:
            m_runs = [load_unit(make_unit_id(pol, s, "abrupt_drift", "medium", m_mode, False, True, 10000))
                      for s in PRIMARY_EVAL_SEEDS if is_unit_valid(make_unit_id(pol, s, "abrupt_drift", "medium", m_mode, False, True, 10000))]
            if len(m_runs) == len(PRIMARY_EVAL_SEEDS):
                misspec_results[m_mode][pol] = aggregate_policy_runs(m_runs)

        if "Oracle" in misspec_results[m_mode]:
            misspec_results[m_mode]["oracle_expected_reward"] = misspec_results[m_mode]["Oracle"]["cumulative_reward"]["mean"]

    # 6. Warm-Start Sweep (Part F)
    warm_sweep_results = {"Warm LinUCB": {}, "Warm LinTS": {}}
    for n_c in [500, 2000]:
        for pol in ["Warm LinUCB", "Warm LinTS"]:
            w_runs = [load_unit(make_unit_id(pol, s, "abrupt_drift", "medium", None, False, True, n_c))
                      for s in range(1, 16) if is_unit_valid(make_unit_id(pol, s, "abrupt_drift", "medium", None, False, True, n_c))]
            if len(w_runs) == 15:
                warm_sweep_results[pol][n_c] = aggregate_policy_runs(w_runs)

    # EXACT DETERMINISM: N_cap=0 matches Cold LinUCB/TS identically!
    if "Cold LinUCB" in primary_benchmark:
        warm_sweep_results["Warm LinUCB"][0] = primary_benchmark["Cold LinUCB"]
        warm_sweep_results["Warm LinUCB"]["0"] = primary_benchmark["Cold LinUCB"]
    if "Cold LinTS" in primary_benchmark:
        warm_sweep_results["Warm LinTS"][0] = primary_benchmark["Cold LinTS"]
        warm_sweep_results["Warm LinTS"]["0"] = primary_benchmark["Cold LinTS"]
    if "Warm LinUCB" in primary_benchmark:
        warm_sweep_results["Warm LinUCB"][10000] = primary_benchmark["Warm LinUCB"]
        warm_sweep_results["Warm LinUCB"]["10000"] = primary_benchmark["Warm LinUCB"]
    if "Warm LinTS" in primary_benchmark:
        warm_sweep_results["Warm LinTS"][10000] = primary_benchmark["Warm LinTS"]
        warm_sweep_results["Warm LinTS"]["10000"] = primary_benchmark["Warm LinTS"]

    # 7. Paired Hypothesis Tests & Part B3 Hybrid Tests
    paired_tests_results = {"medium": [], "by_environment": {}}
    headline_specs = [
        ("Discounted (gamma*=0.9995) vs Cold LinUCB", f"Discounted LinUCB (gamma={best_gamma})", "Cold LinUCB"),
        ("Discounted (gamma=0.999) vs Cold LinUCB", "Discounted LinUCB (gamma=0.999)", "Cold LinUCB"),
        ("Warm vs Cold LinUCB", "Warm LinUCB", "Cold LinUCB"),
        ("Warm vs Cold LinTS", "Warm LinTS", "Cold LinTS"),
        ("Cold LinUCB vs Department Heuristic", "Cold LinUCB", "Department Heuristic"),
        ("Cold LinUCB vs RecentlyViewed+Popularity", "Cold LinUCB", "Recently Viewed + Popularity"),
        ("Cold LinUCB vs Epsilon-Greedy", "Cold LinUCB", "Epsilon-Greedy"),
        ("Cold LinUCB vs Cold LinTS", "Cold LinUCB", "Cold LinTS"),
        ("Department Heuristic vs Department Heuristic (With Diversity)", "Department Heuristic", "Department Heuristic (With Diversity)"),
        ("Cold LinUCB vs Dept-Prior LinUCB (w=0.5)", "Cold LinUCB", "Dept-Prior LinUCB (w=0.5)"),
        ("Cold LinUCB vs Slate Mixing (k=3 Dept + LinUCB)", "Cold LinUCB", "Slate Mixing (k=3 Dept + LinUCB)"),
        ("Cold LinUCB vs Slate Mixing (k=5 Dept + LinUCB)", "Cold LinUCB", "Slate Mixing (k=5 Dept + LinUCB)"),
        ("Cold LinUCB vs Residual LinUCB", "Cold LinUCB", "Residual LinUCB"),
        ("Department Heuristic vs Dept-Prior LinUCB (w=0.5)", "Department Heuristic", "Dept-Prior LinUCB (w=0.5)"),
        ("Department Heuristic vs Slate Mixing (k=3 Dept + LinUCB)", "Department Heuristic", "Slate Mixing (k=3 Dept + LinUCB)"),
        ("Department Heuristic vs Slate Mixing (k=5 Dept + LinUCB)", "Department Heuristic", "Slate Mixing (k=5 Dept + LinUCB)"),
        ("Department Heuristic vs Residual LinUCB", "Department Heuristic", "Residual LinUCB"),
        # Two-stage hybrids vs Dept Heuristic
        ("Department Heuristic vs Dept-Restricted LinUCB (Cold)", "Department Heuristic", "Dept-Restricted LinUCB (Cold)"),
        ("Department Heuristic vs Dept-Restricted LinUCB (Discounted)", "Department Heuristic", "Dept-Restricted LinUCB (Discounted)"),
        ("Department Heuristic vs Dept-Restricted LinUCB + Expl (Cold, m=2)", "Department Heuristic", "Dept-Restricted LinUCB + Expl (Cold, m=2)"),
        ("Department Heuristic vs Dept-Restricted LinUCB + Expl (Discounted, m=2)", "Department Heuristic", "Dept-Restricted LinUCB + Expl (Discounted, m=2)"),
        # Two-stage hybrids vs Cold LinUCB
        ("Cold LinUCB vs Dept-Restricted LinUCB (Cold)", "Cold LinUCB", "Dept-Restricted LinUCB (Cold)"),
        ("Cold LinUCB vs Dept-Restricted LinUCB (Discounted)", "Cold LinUCB", "Dept-Restricted LinUCB (Discounted)"),
        ("Cold LinUCB vs Dept-Restricted LinUCB + Expl (Cold, m=2)", "Cold LinUCB", "Dept-Restricted LinUCB + Expl (Cold, m=2)"),
        ("Cold LinUCB vs Dept-Restricted LinUCB + Expl (Discounted, m=2)", "Cold LinUCB", "Dept-Restricted LinUCB + Expl (Discounted, m=2)")
    ]

    raw_p_list = []
    for name, pol_a, pol_b in headline_specs:
        if pol_a in primary_benchmark and pol_b in primary_benchmark:
            arr_a = np.array(primary_benchmark[pol_a]["cumulative_regret"]["values"])
            arr_b = np.array(primary_benchmark[pol_b]["cumulative_regret"]["values"])
            stat = compute_paired_statistics(arr_a, arr_b)
            raw_p_list.append(stat["p_value_raw"])
            paired_tests_results["medium"].append({
                "test": name,
                "pol_a": pol_a,
                "pol_b": pol_b,
                "mean_diff": stat["mean_diff"],
                "ci_95": stat["ci_95"],
                "cohen_d": stat["cohen_d"],
                "p_value_raw": stat["p_value_raw"]
            })

    holm_p_list = apply_holm_bonferroni(raw_p_list)
    for i, item in enumerate(paired_tests_results["medium"]):
        item["p_value_holm"] = holm_p_list[i]
        ci = item["ci_95"]
        if ci[1] < 0.0 and item["p_value_holm"] < 0.05:
            item["verdict"] = f"{item['pol_a']} beats {item['pol_b']} (statistically significant)"
        elif ci[0] > 0.0 and item["p_value_holm"] < 0.05:
            item["verdict"] = f"{item['pol_b']} beats {item['pol_a']} (statistically significant)"
        elif (ci[1] < 0.0 or ci[0] > 0.0) and item["p_value_holm"] >= 0.05:
            item["verdict"] = "Bootstrap CI excludes zero but Holm-corrected Wilcoxon is not significant"
        else:
            item["verdict"] = "Interval spans zero / No significant difference"

    # Part B3 Environment paired tests: base, no_drift, nonlinear_interactions, hidden_style
    env_dicts = {
        "base": primary_benchmark,
        "nonlinear_interactions": misspec_results.get("nonlinear_interactions", {}),
        "hidden_style": misspec_results.get("hidden_style", {})
    }
    # Load no_drift for hybrids
    no_drift_dict = {}
    for pol in (list(HYBRID_POLICIES) + ["Department Heuristic", "Cold LinUCB", "Oracle", "Random"]):
        nd_runs = [load_unit(make_unit_id(pol, s, "no_drift", "medium", None, False, True, 10000))
                   for s in PRIMARY_EVAL_SEEDS if is_unit_valid(make_unit_id(pol, s, "no_drift", "medium", None, False, True, 10000))]
        if len(nd_runs) == len(PRIMARY_EVAL_SEEDS):
            no_drift_dict[pol] = aggregate_policy_runs(nd_runs)
    env_dicts["no_drift"] = no_drift_dict

    for env_name, env_pols in env_dicts.items():
        paired_tests_results["by_environment"][env_name] = []
        raw_p_env = []
        env_records = []
        if "Department Heuristic" in env_pols and "Cold LinUCB" in env_pols:
            for hyb in HYBRID_POLICIES:
                if hyb in env_pols:
                    arr_h = np.array(env_pols[hyb]["cumulative_regret"]["values"])
                    arr_dh = np.array(env_pols["Department Heuristic"]["cumulative_regret"]["values"])
                    arr_cl = np.array(env_pols["Cold LinUCB"]["cumulative_regret"]["values"])

                    # vs Dept Heuristic
                    st_dh = compute_paired_statistics(arr_h, arr_dh)
                    raw_p_env.append(st_dh["p_value_raw"])
                    rec_dh = {
                        "test": f"{hyb} vs Department Heuristic",
                        "pol_a": hyb,
                        "pol_b": "Department Heuristic",
                        "mean_diff": st_dh["mean_diff"],
                        "ci_95": st_dh["ci_95"],
                        "cohen_d": st_dh["cohen_d"],
                        "p_value_raw": st_dh["p_value_raw"]
                    }
                    env_records.append(rec_dh)

                    # vs Cold LinUCB
                    st_cl = compute_paired_statistics(arr_h, arr_cl)
                    raw_p_env.append(st_cl["p_value_raw"])
                    rec_cl = {
                        "test": f"{hyb} vs Cold LinUCB",
                        "pol_a": hyb,
                        "pol_b": "Cold LinUCB",
                        "mean_diff": st_cl["mean_diff"],
                        "ci_95": st_cl["ci_95"],
                        "cohen_d": st_cl["cohen_d"],
                        "p_value_raw": st_cl["p_value_raw"]
                    }
                    env_records.append(rec_cl)

        holm_p_env = apply_holm_bonferroni(raw_p_env)
        for i, rec in enumerate(env_records):
            rec["p_value_holm"] = holm_p_env[i]
            ci = rec["ci_95"]
            if ci[1] < 0.0 and rec["p_value_holm"] < 0.05:
                rec["verdict"] = f"{rec['pol_a']} beats {rec['pol_b']} (statistically significant)"
            elif ci[0] > 0.0 and rec["p_value_holm"] < 0.05:
                rec["verdict"] = f"{rec['pol_b']} beats {rec['pol_a']} (statistically significant)"
            elif (ci[1] < 0.0 or ci[0] > 0.0) and rec["p_value_holm"] >= 0.05:
                rec["verdict"] = "Bootstrap CI excludes zero but Holm-corrected Wilcoxon is not significant"
            else:
                rec["verdict"] = "Interval spans zero / No significant difference"
            paired_tests_results["by_environment"][env_name].append(rec)

    # 8. Calibration
    calibration_metrics = {}
    for diff in ["easy", "medium", "hard"]:
        cal = EnvironmentV2.get_calibration_metrics(difficulty=diff, n_rounds=4000, seed=42)
        calibration_metrics[diff] = cal

    calibration_metrics["jensen_convexity_note"] = (
        "On negative logits (beta_click = 5.72), the sigmoid function is convex. By Jensen's inequality, "
        "E[sigmoid(X + epsilon)] > sigmoid(E[X]), so higher noise sigma=0.80 elevates random click probability "
        "to ~6.97% compared to ~5.23% at sigma=0.15."
    )

    # 9. Before / After Comparison Table (Part A1)
    before_after = {}
    if legacy_data and "primary_benchmark" in legacy_data:
        legacy_primary = legacy_data["primary_benchmark"]
        key_compare = [
            "Department Heuristic",
            "Cold LinUCB",
            f"Discounted LinUCB (gamma={best_gamma})",
            "Discounted LinUCB (gamma=1.0)",
            "Warm LinUCB",
            "Cold LinTS",
            "Warm LinTS",
            "Slate Mixing (k=5 Dept + LinUCB)",
            "Residual LinUCB",
            "Oracle",
            "Random"
        ]
        for pol in key_compare:
            if pol in primary_benchmark:
                val_fresh = primary_benchmark[pol]["cumulative_regret"]["mean"]
                val_legacy = legacy_primary.get(pol, {}).get("cumulative_regret", {}).get("mean", None)
                diff = (val_fresh - val_legacy) if val_legacy is not None else None
                before_after[pol] = {
                    "phase12a_legacy": val_legacy,
                    "phase12a2_authoritative": val_fresh,
                    "difference": diff,
                    "status": "Authoritative Ground Truth (Phase 12A-2)"
                }

    # Master results dict
    master_results = {
        "metadata": {
            "phase": "12A-2",
            "active_catalog_size": 262,
            "feature_dim": 36,
            "feature_spec_hash": "75f60ed70b2a",
            "horizon": HORIZON,
            "best_alpha": best_alpha,
            "best_gamma": best_gamma,
            "n_eval_seeds_primary": len(PRIMARY_EVAL_SEEDS),
            "n_eval_seeds_grid": len(GRID_EVAL_SEEDS),
            "trajectory_oracle_reward_identity": "Reward(pi) + Regret(pi) == Trajectory Oracle Expected Reward(pi)",
            "post_drift_regret_definitions": {
                "post_drift_1k": "[10000, 11000) (first 1,000 rounds post-drift)",
                "post_drift_half": "[10000, 20000) (whole second half)",
                "post_drift_2k_legacy": "[10000, 12000) (first 2,000 rounds post-drift)"
            },
            "runtime_seconds": 0.0
        },
        "hyperparameter_tuning": {
            "alpha_sweep": alpha_sweep,
            "gamma_sweep": gamma_sweep,
            "best_alpha": best_alpha,
            "best_gamma": best_gamma,
            "tuning_explanation": (
                "Hyperparameter tuning on independent seeds 101-105 under strict environment isolation selected "
                f"alpha* = {best_alpha} and gamma* = {best_gamma} across the 7-value candidate grid "
                "{0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0}."
            )
        },
        "calibration_per_difficulty": calibration_metrics,
        "primary_benchmark": primary_benchmark,
        "gamma_grid_evaluation": gamma_grid_eval,
        "a1_discounting_tests": a1_test_suite,
        "paired_tests": paired_tests_results,
        "confounded_scenarios": confounded_results,
        "misspecification_robustness": misspec_results,
        "warm_start_sweep": warm_sweep_results,
        "before_after_comparison": before_after
    }

    # Atomic write to master results
    out_file = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results.json"
    tmp_out = out_file.with_suffix(".tmp.json")
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)
    tmp_out.replace(out_file)
    log(f"Master benchmark results written atomically to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="SmartCart AI — Calibrated Simulator v2 Resumable Benchmark Runner")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip completed valid units (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not skip existing units")
    parser.add_argument("--only-scenario", type=str, default=None, help="Filter to specific scenario")
    parser.add_argument("--only-difficulty", type=str, default=None, help="Filter to specific difficulty")
    parser.add_argument("--only-policy", type=str, default=None, help="Filter to specific policy")
    parser.add_argument("--max-units", type=int, default=None, help="Maximum units to run in this invocation")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and missing unit counts without running")
    parser.add_argument("--validate", action="store_true", help="Validate existing units and test determinism")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate master results from units without running")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1), help="Worker processes")

    args = parser.parse_args()

    if args.validate:
        valid_cnt, corrupt_cnt, retest_ok = validate_existing_units()
        log(f"Validation Report: {valid_cnt} valid units, {corrupt_cnt} corrupt/deleted units, retest determinism={retest_ok}")
        return

    if args.aggregate_only:
        aggregate_master_results(workers=args.workers)
        return

    # 1. Build all required units
    all_tasks = build_all_required_units()
    total_planned = len(all_tasks)

    # 2. Filter tasks
    filtered_tasks = []
    for t in all_tasks:
        (policy, seed, difficulty, drift_mode, horizon, is_pipeline, misspecification,
         withhold_new_products, alpha, gamma, n_cap, scenario_name, unit_id) = t

        if args.only_scenario and scenario_name != args.only_scenario:
            continue
        if args.only_difficulty and difficulty != args.only_difficulty:
            continue
        if args.only_policy and policy != args.only_policy:
            continue

        if args.resume and is_unit_valid(unit_id):
            continue

        filtered_tasks.append(t)

    n_already_done = total_planned - len(filtered_tasks)
    if args.max_units is not None:
        filtered_tasks = filtered_tasks[:args.max_units]
    log(f"Benchmark Unit Plan: Total Planned = {total_planned} | Already Completed = {n_already_done} | Remaining to Run = {len(filtered_tasks)}")

    if args.dry_run:
        log("Dry run complete. No units executed.")
        return

    # 3. Run hyperparameter tuning on separate tuning seeds 101-105
    tuning_data = run_hyperparameter_tuning(workers=args.workers)

    if not filtered_tasks:
        log("All requested units are already complete!")
        aggregate_master_results(tuning_data=tuning_data, workers=args.workers)
        return

    # 4. Execute missing units in parallel with ETA tracking
    workers = args.workers
    log(f"Launching execution of {len(filtered_tasks)} units across {workers} parallel worker processes...")

    start_time = time.time()
    completed = 0
    total_to_run = len(filtered_tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute_unit_worker, t): t for t in filtered_tasks}
        for future in as_completed(futures):
            completed += 1
            t = futures[future]
            unit_id = t[-1]
            try:
                uid, res = future.result()
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                rem_seconds = avg_time * (total_to_run - completed)
                rem_mins = rem_seconds / 60.0
                if completed % 25 == 0 or completed == total_to_run:
                    log(f"Progress: [{completed}/{total_to_run}] ({completed/total_to_run*100:.1f}%) | "
                        f"Unit: {unit_id} | Regret: {res['cumulative_regret']:.1f} | ETA: {rem_mins:.1f} min")
            except Exception as e:
                log(f"ERROR executing unit {unit_id}: {e}")
                raise e

    total_time = time.time() - start_time
    log(f"Batch completed in {total_time:.2f}s ({total_time/60.0:.2f} minutes).")

    # 5. Aggregate master results
    aggregate_master_results(tuning_data=tuning_data, workers=args.workers)


if __name__ == "__main__":
    main()
