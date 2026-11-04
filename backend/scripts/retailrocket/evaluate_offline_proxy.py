"""
SmartCart AI — Phase 9C Hardened Proxy Offline Evaluation Suite
Implements:
  - Dual estimands: Event-level (micro) and Visitor-level (macro) side by side
  - 2,000-resample visitor-clustered bootstrap 95% CIs and p-values (< 1/B)
  - Full test split evaluation (335,612 interactions) with compact rank persistence
  - Paired 50-candidate comparisons (same candidate pool per example across all policies)
  - Three-mapping comparison with empirical Test Popularity Ceiling
  - Baselines: Department Heuristic, Hybrid (Dept x Pop), Popularity, Random, Controls
  - Feature importance top 10 & systematic feature group ablations
  - Prequential test replay over 20,000 chronological steps with tabular checkpointing
  - Slices with sample sizes n, CIs, and low-n flags (< 100)
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
import sqlite3
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    CATEGORIES,
    CATEGORY_INDEX,
    UserContext,
    build_context_from_history,
    extract_feature_vector
)
from app.recommender.linucb import LinUCBRecommender
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.linucb_thompson import LinThompsonSamplingRecommender
from app.recommender.epsilon_greedy import EpsilonGreedyRecommender

HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
DATA_DIR = Path("data/historical")
RESULTS_DIR = Path("experiments/results")
PLOTS_DIR = Path("experiments/plots/retailrocket")

T_TRAIN_END = 1438968232766
T_VAL_END = 1440756710277
T_TEST_END = 1442545187788

REWARD_MAP = {
    "view": 2.0,
    "click": 2.0,
    "addtocart": 5.0,
    "cart_add": 5.0,
    "transaction": 10.0,
    "purchase": 10.0
}


def evaluate_ranking(ranked_items, target_pid, k_vals=(1, 3, 5, 10)):
    """Evaluate 1-based rank, MRR, HR@K, NDCG@K for a single ranking."""
    rank = None
    for i, item in enumerate(ranked_items):
        pid = item[0] if isinstance(item, (tuple, list)) else item
        if pid == target_pid:
            rank = i + 1
            break

    res = {
        "rank": rank,
        "mrr": 1.0 / rank if rank is not None else 0.0,
    }
    for k in k_vals:
        res[f"hr@{k}"] = 1.0 if (rank is not None and rank <= k) else 0.0
        res[f"ndcg@{k}"] = (1.0 / math.log2(rank + 1)) if (rank is not None and rank <= k) else 0.0

    return res



def load_catalog(smartcart_db_path: Path):
    """Load the 262 active catalog products and precompute static representations."""
    conn = sqlite3.connect(smartcart_db_path)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, name, category, brand, price, rating, discount_pct, popularity_score, popularity_prior
        FROM products
        WHERE is_active = 1
        ORDER BY id;
    """).fetchall()
    conn.close()

    catalog = {}
    dept_pids = defaultdict(list)
    pid_to_idx = {}
    idx_to_pid = []

    for idx, r in enumerate(rows):
        pid, name, cat, brand, price, rating, discount, pop, pop_prior = r
        prod = {
            "id": pid,
            "name": name,
            "category": cat,
            "brand": brand or "",
            "price": float(price),
            "rating": float(rating),
            "discount_pct": float(discount),
            "popularity_score": float(pop),
            "popularity_prior": float(pop_prior or 0.0)
        }
        catalog[pid] = prod
        dept_pids[cat].append(pid)
        pid_to_idx[pid] = idx
        idx_to_pid.append(pid)

    return catalog, dept_pids, pid_to_idx, idx_to_pid


def load_mapping(historical_db_path: Path, method: str, version: str = "v1.0_train"):
    conn = sqlite3.connect(historical_db_path)
    c = conn.cursor()
    rows = c.execute("""
        SELECT item_id, product_id
        FROM rr_item_map
        WHERE mapping_method = ? AND mapping_version = ?;
    """, (method, version)).fetchall()
    conn.close()
    return dict(rows)


def build_candidate_matrix(catalog: dict, idx_to_pid: list, context: UserContext):
    """Vectorized construction of the 262 x 36 feature matrix for a given user context."""
    X = np.zeros((len(idx_to_pid), FEATURE_DIMENSION), dtype=np.float64)
    for i, pid in enumerate(idx_to_pid):
        X[i] = extract_feature_vector(context, catalog[pid])
    return X


def calculate_metrics_from_ranks(ranks: np.ndarray, k_vals=(5, 10, 20)):
    """Computes HR@K, NDCG@K, MRR from an array of 1-based true item ranks."""
    out = {}
    for k in k_vals:
        out[f"hr@{k}"] = (ranks <= k).astype(np.float64)
        # NDCG@K: 1 / log2(rank + 1) if rank <= k else 0.0
        dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
        out[f"ndcg@{k}"] = dcg
    out["mrr"] = np.where(ranks <= 100, 1.0 / ranks, 0.0)
    return out


def bootstrap_ci_dual(event_values: np.ndarray, visitor_ids: np.ndarray, n_boot: int = 2000, rng=None):
    """
    Computes both Event-Level (Micro) and Visitor-Level (Macro) point estimates
    and 95% visitor-clustered bootstrap confidence intervals.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Group event indices by visitor
    v_dict = defaultdict(list)
    for idx, v in enumerate(visitor_ids):
        v_dict[v].append(idx)

    unique_visitors = list(v_dict.keys())
    n_v = len(unique_visitors)

    # Visitor-level means
    v_means = np.array([np.mean(event_values[v_dict[v]]) for v in unique_visitors], dtype=np.float64)
    macro_point = float(np.mean(v_means))
    micro_point = float(np.mean(event_values))

    # Pre-index visitor arrays for fast bootstrap sampling
    v_event_counts = np.array([len(v_dict[v]) for v in unique_visitors], dtype=np.int32)
    v_event_sums = np.array([np.sum(event_values[v_dict[v]]) for v in unique_visitors], dtype=np.float64)

    boot_micro = np.empty(n_boot, dtype=np.float64)
    boot_macro = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sample_idx = rng.choice(n_v, size=n_v, replace=True)
        boot_macro[b] = np.mean(v_means[sample_idx])
        sum_events = np.sum(v_event_sums[sample_idx])
        cnt_events = np.sum(v_event_counts[sample_idx])
        boot_micro[b] = sum_events / max(1, cnt_events)

    macro_ci = (float(np.percentile(boot_macro, 2.5)), float(np.percentile(boot_macro, 97.5)))
    micro_ci = (float(np.percentile(boot_micro, 2.5)), float(np.percentile(boot_micro, 97.5)))

    return {
        "macro_mean": macro_point,
        "macro_ci": macro_ci,
        "micro_mean": micro_point,
        "micro_ci": micro_ci,
        "boot_macro": boot_macro,
        "boot_micro": boot_micro
    }


def paired_bootstrap_test(boot_a: np.ndarray, boot_b: np.ndarray):
    """Computes paired difference and p-value from bootstrap distributions."""
    diff = boot_a - boot_b
    diff_mean = float(np.mean(diff))
    diff_ci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
    p_val = float(np.mean(diff <= 0.0))
    p_val_str = f"< {1.0 / len(diff):.4f}" if p_val == 0.0 else f"{p_val:.4f}"
    return diff_mean, diff_ci, p_val_str


def run_hardened_evaluation():
    print("=" * 80, flush=True)
    print("SMARTCART AI: PHASE 9C HARDENED OFFLINE EVALUATION & VERIFICATION", flush=True)
    print("=" * 80, flush=True)

    start_time = time.time()
    rng = np.random.default_rng(42)
    random.seed(42)
    np.random.seed(42)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load active catalog
    catalog, dept_pids, pid_to_idx, idx_to_pid = load_catalog(SMARTCART_DB)
    n_catalog = len(catalog)
    all_pids = list(idx_to_pid)
    print(f"Catalog: Loaded {n_catalog} active SKUs across {len(dept_pids)} departments.", flush=True)

    # Popularity prior array aligned to catalog indices
    pop_priors = np.array([catalog[p]["popularity_prior"] for p in idx_to_pid], dtype=np.float64)
    # Popularity sampling distribution
    p_weights = np.maximum(1.0, pop_priors) ** 0.75
    p_sampling = p_weights / np.sum(p_weights)

    # 2. Load primary training mapping
    primary_map = load_mapping(HISTORICAL_DB, "popularity_balanced_train", "v1.0_train")
    print(f"Primary Mapping: Loaded {len(primary_map):,} items for 'popularity_balanced_train'.", flush=True)

    # 3. Load warm-start models
    stats_m3 = np.load(DATA_DIR / "sufficient_stats_m3.npz", allow_pickle=True)
    assert str(stats_m3["feature_spec_hash"]) == FEATURE_SPEC_HASH

    # Frozen configuration
    FROZEN_NCAP = 10000
    FROZEN_M = 3
    FROZEN_LAMBDA = 1.0
    FROZEN_ALPHA = 0.5

    warm_linucb = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=FROZEN_ALPHA, lambda_reg=FROZEN_LAMBDA)
    warm_linucb.warm_start_from_stats(stats_m3, n_cap=FROZEN_NCAP, lambda_reg=FROZEN_LAMBDA)

    cold_linucb = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=FROZEN_ALPHA, lambda_reg=FROZEN_LAMBDA)

    warm_lints = LinThompsonSamplingRecommender(feature_dim=FEATURE_DIMENSION, v_variance=0.25, lambda_reg=FROZEN_LAMBDA)
    warm_lints.warm_start_from_stats(stats_m3, n_cap=FROZEN_NCAP, lambda_reg=FROZEN_LAMBDA)

    warm_discounted = DiscountedLinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=FROZEN_ALPHA, lambda_reg=FROZEN_LAMBDA, gamma=0.95)
    warm_discounted.warm_start_from_stats(stats_m3, n_cap=FROZEN_NCAP, lambda_reg=FROZEN_LAMBDA)

    warm_eps = EpsilonGreedyRecommender(feature_dim=FEATURE_DIMENSION, epsilon=0.0, lambda_reg=FROZEN_LAMBDA)
    warm_eps.warm_start_from_stats(stats_m3, n_cap=FROZEN_NCAP, lambda_reg=FROZEN_LAMBDA)

    # Shuffled reward control
    shuf_stats = dict(stats_m3)
    shuf_stats["b_hist"] = rng.permutation(shuf_stats["b_hist"])
    ctrl_shuf_reward = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=FROZEN_ALPHA, lambda_reg=FROZEN_LAMBDA)
    ctrl_shuf_reward.warm_start_from_stats(shuf_stats, n_cap=FROZEN_NCAP, lambda_reg=FROZEN_LAMBDA)

    theta_warm = warm_linucb.get_theta()
    theta_cold = cold_linucb.get_theta()
    theta_shuf = ctrl_shuf_reward.get_theta()
    A_inv_warm = warm_linucb.A_inv
    A_inv_cold = cold_linucb.A_inv

    print("Models initialized. Theta norms: Warm=" + f"{np.linalg.norm(theta_warm):.4f}, Cold={np.linalg.norm(theta_cold):.4f}")

    # -------------------------------------------------------------------------
    # STAGE 1: VALIDATION HYPERPARAMETER SENSITIVITY SWEEP (50,000 EXAMPLES)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 1: HARDENED VALIDATION SWEEP (50,000 EXAMPLES WITH CIs)", flush=True)
    print("=" * 60, flush=True)

    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_hist = conn_hist.cursor()

    # Query 50,000 validation events ordered by visitor and ts
    val_rows = c_hist.execute("""
        SELECT visitor_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts >= ? AND ts < ?
        ORDER BY visitor_id, ts
        LIMIT 65000;
    """, (T_TRAIN_END, T_VAL_END)).fetchall()

    val_examples = []
    val_vis_hist = defaultdict(list)
    for v_id, i_id, ev_type, ts in val_rows:
        pid = primary_map.get(i_id)
        if pid not in catalog:
            continue
        hist = list(val_vis_hist[v_id])
        val_examples.append((v_id, pid, ev_type, ts, hist))
        val_vis_hist[v_id].append((catalog[pid], ev_type, ts, REWARD_MAP.get(ev_type, 2.0)))
        if len(val_examples) >= 50000:
            break

    print(f"Validation Sample: {len(val_examples):,} events across {len(set(e[0] for e in val_examples)):,} visitors.", flush=True)

    val_sweep_records = []
    n_cap_grid = [0, 500, 2000, 10000, 50000, None]
    m_grid = [0, 1, 3, 5]
    lambda_grid = [0.1, 1.0, 10.0]
    alpha_grid = [0.0, 0.2, 0.5, 1.0]

    val_vids = np.array([e[0] for e in val_examples])

    # Sweep N_cap
    print("  Evaluating N_cap sweep...", flush=True)
    for n_cap in n_cap_grid:
        m_temp = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0)
        m_temp.warm_start_from_stats(stats_m3, n_cap=n_cap, lambda_reg=1.0)
        th = m_temp.get_theta()
        a_inv = m_temp.A_inv

        ranks_val = np.empty(len(val_examples), dtype=np.int32)
        for idx, (v_id, t_pid, ev_type, ts, hist) in enumerate(val_examples):
            ctx = build_context_from_history(hist, reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))
            X_u = build_candidate_matrix(catalog, idx_to_pid, ctx)
            expected = X_u @ th
            var = np.maximum(0.0, np.sum((X_u @ a_inv) * X_u, axis=1))
            scores = expected + 0.5 * np.sqrt(var)
            target_idx = pid_to_idx[t_pid]
            ranks_val[idx] = np.count_nonzero(scores > scores[target_idx]) + 1

        res = bootstrap_ci_dual((ranks_val <= 10).astype(float), val_vids, n_boot=1000, rng=rng)
        label_ncap = "all" if n_cap is None else str(n_cap)
        val_sweep_records.append({
            "parameter": "N_cap",
            "setting": label_ncap,
            "macro_hr@10": res["macro_mean"],
            "macro_ci_low": res["macro_ci"][0],
            "macro_ci_high": res["macro_ci"][1],
            "micro_hr@10": res["micro_mean"],
            "micro_ci_low": res["micro_ci"][0],
            "micro_ci_high": res["micro_ci"][1]
        })
        print(f"    N_cap={label_ncap:5s}: Macro HR@10={res['macro_mean']*100:.2f}% [{res['macro_ci'][0]*100:.2f}%, {res['macro_ci'][1]*100:.2f}%], Micro={res['micro_mean']*100:.2f}%")

    # Sweep m
    print("  Evaluating pseudo-negatives m sweep...", flush=True)
    for m in m_grid:
        st_m = np.load(DATA_DIR / f"sufficient_stats_m{m}.npz", allow_pickle=True)
        m_temp = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0)
        m_temp.warm_start_from_stats(st_m, n_cap=10000, lambda_reg=1.0)
        th = m_temp.get_theta()
        a_inv = m_temp.A_inv

        ranks_val = np.empty(len(val_examples), dtype=np.int32)
        for idx, (v_id, t_pid, ev_type, ts, hist) in enumerate(val_examples):
            ctx = build_context_from_history(hist, reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))
            X_u = build_candidate_matrix(catalog, idx_to_pid, ctx)
            expected = X_u @ th
            var = np.maximum(0.0, np.sum((X_u @ a_inv) * X_u, axis=1))
            scores = expected + 0.5 * np.sqrt(var)
            target_idx = pid_to_idx[t_pid]
            ranks_val[idx] = np.count_nonzero(scores > scores[target_idx]) + 1

        res = bootstrap_ci_dual((ranks_val <= 10).astype(float), val_vids, n_boot=1000, rng=rng)
        val_sweep_records.append({
            "parameter": "pseudo_negatives_m",
            "setting": str(m),
            "macro_hr@10": res["macro_mean"],
            "macro_ci_low": res["macro_ci"][0],
            "macro_ci_high": res["macro_ci"][1],
            "micro_hr@10": res["micro_mean"],
            "micro_ci_low": res["micro_ci"][0],
            "micro_ci_high": res["micro_ci"][1]
        })
        print(f"    m={m}: Macro HR@10={res['macro_mean']*100:.2f}% [{res['macro_ci'][0]*100:.2f}%, {res['macro_ci'][1]*100:.2f}%]")

    # Sweep lambda
    print("  Evaluating lambda sweep...", flush=True)
    for lam in lambda_grid:
        m_temp = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=lam)
        m_temp.warm_start_from_stats(stats_m3, n_cap=10000, lambda_reg=lam)
        th = m_temp.get_theta()
        a_inv = m_temp.A_inv

        ranks_val = np.empty(len(val_examples), dtype=np.int32)
        for idx, (v_id, t_pid, ev_type, ts, hist) in enumerate(val_examples):
            ctx = build_context_from_history(hist, reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))
            X_u = build_candidate_matrix(catalog, idx_to_pid, ctx)
            expected = X_u @ th
            var = np.maximum(0.0, np.sum((X_u @ a_inv) * X_u, axis=1))
            scores = expected + 0.5 * np.sqrt(var)
            target_idx = pid_to_idx[t_pid]
            ranks_val[idx] = np.count_nonzero(scores > scores[target_idx]) + 1

        res = bootstrap_ci_dual((ranks_val <= 10).astype(float), val_vids, n_boot=1000, rng=rng)
        val_sweep_records.append({
            "parameter": "lambda",
            "setting": str(lam),
            "macro_hr@10": res["macro_mean"],
            "macro_ci_low": res["macro_ci"][0],
            "macro_ci_high": res["macro_ci"][1],
            "micro_hr@10": res["micro_mean"],
            "micro_ci_low": res["micro_ci"][0],
            "micro_ci_high": res["micro_ci"][1]
        })
        print(f"    lambda={lam}: Macro HR@10={res['macro_mean']*100:.2f}% [{res['macro_ci'][0]*100:.2f}%, {res['macro_ci'][1]*100:.2f}%]")

    # Sweep alpha
    print("  Evaluating alpha sweep...", flush=True)
    for a in alpha_grid:
        ranks_val = np.empty(len(val_examples), dtype=np.int32)
        for idx, (v_id, t_pid, ev_type, ts, hist) in enumerate(val_examples):
            ctx = build_context_from_history(hist, reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))
            X_u = build_candidate_matrix(catalog, idx_to_pid, ctx)
            expected = X_u @ theta_warm
            if a > 0:
                var = np.maximum(0.0, np.sum((X_u @ A_inv_warm) * X_u, axis=1))
                scores = expected + a * np.sqrt(var)
            else:
                scores = expected
            target_idx = pid_to_idx[t_pid]
            ranks_val[idx] = np.count_nonzero(scores > scores[target_idx]) + 1

        res = bootstrap_ci_dual((ranks_val <= 10).astype(float), val_vids, n_boot=1000, rng=rng)
        val_sweep_records.append({
            "parameter": "alpha",
            "setting": str(a),
            "macro_hr@10": res["macro_mean"],
            "macro_ci_low": res["macro_ci"][0],
            "macro_ci_high": res["macro_ci"][1],
            "micro_hr@10": res["micro_mean"],
            "micro_ci_low": res["micro_ci"][0],
            "micro_ci_high": res["micro_ci"][1]
        })
        print(f"    alpha={a}: Macro HR@10={res['macro_mean']*100:.2f}% [{res['macro_ci'][0]*100:.2f}%, {res['macro_ci'][1]*100:.2f}%]")

    df_val_sweep = pd.DataFrame(val_sweep_records)
    df_val_sweep.to_csv(RESULTS_DIR / "retailrocket_offline_eval_validation_9c.csv", index=False)
    print("  Validation sweep saved to experiments/results/retailrocket_offline_eval_validation_9c.csv", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 2: STREAMING FULL TEST SPLIT EVALUATION (335,612 INTERACTIONS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 2: STREAMING FULL TEST SPLIT EVALUATION (ALL TEST EXAMPLES)", flush=True)
    print("=" * 60, flush=True)

    # Stream all test events ordered by visitor and ts
    test_cursor = c_hist.execute("""
        SELECT visitor_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts >= ?
        ORDER BY visitor_id, ts;
    """, (T_VAL_END,))

    # We evaluate 11 policies on full catalog and 50 candidates:
    # Policies:
    #  0: Cold LinUCB
    #  1: Warm LinUCB (UCB alpha=0.5)
    #  2: Warm LinUCB (Greedy alpha=0.0)
    #  3: Warm Discounted LinUCB (Identical to 1 in frozen setting)
    #  4: Warm LinTS (v=0.25)
    #  5: Warm Epsilon-Greedy (Greedy eps=0.0)
    #  6: Department Heuristic
    #  7: Hybrid (Dept x Pop)
    #  8: Popularity Baseline
    #  9: Random Baseline
    # 10: Control: Shuffled Rewards
    # 11: Control: Shuffled Context

    policy_names = [
        "Cold LinUCB (alpha=0.5)",
        "Warm LinUCB (Optimal UCB)",
        "Warm LinUCB (Greedy alpha=0)",
        "Warm Discounted LinUCB",
        "Warm LinTS (v=0.25)",
        "Warm Epsilon-Greedy (eps=0)",
        "Department Heuristic",
        "Hybrid (Dept x Pop)",
        "Popularity Baseline",
        "Random Baseline",
        "Control: Shuffled Rewards",
        "Control: Shuffled Context"
    ]
    n_policies = len(policy_names)

    # Load pre-shuffled contexts queue for Control: Shuffled Context
    # We sample a pool of random past visitor histories to assign as shuffled context
    shuf_context_pool = []

    print("  Streaming test interactions and vectorizing candidate evaluations...", flush=True)
    
    test_vids = []
    test_pids = []
    test_types = []
    test_ctx_levels = []
    test_c32_vals = []
    test_ranks_full = []
    test_ranks_50 = []
    test_cand50_true_positions = []

    curr_v = None
    curr_hist = []

    t_start_stream = time.time()
    n_processed = 0

    for v_id, i_id, ev_type, ts in test_cursor:
        n_processed += 1
        if v_id != curr_v:
            curr_v = v_id
            curr_hist = []

        t_pid = primary_map.get(i_id)
        if t_pid not in catalog:
            continue

        ctx_level = "cold" if len(curr_hist) == 0 else ("light" if len(curr_hist) < 5 else "heavy")
        ctx = build_context_from_history(curr_hist, reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))

        # True target product index
        t_idx = pid_to_idx[t_pid]

        # Add to shuffled context pool (keep size bounded)
        if len(shuf_context_pool) < 2000 and len(curr_hist) > 0:
            shuf_context_pool.append(list(curr_hist))

        # 1. Feature Matrix for candidate set X_u (262 x 36)
        X_u = build_candidate_matrix(catalog, idx_to_pid, ctx)
        c32_val = float(X_u[t_idx, 32])

        # 2. Score vectors for all policies
        # LinUCB expected rewards and exploration bonuses
        exp_warm = X_u @ theta_warm
        exp_cold = X_u @ theta_cold
        exp_shuf = X_u @ theta_shuf

        var_warm = np.maximum(0.0, np.sum((X_u @ A_inv_warm) * X_u, axis=1))
        bonus_warm = 0.5 * np.sqrt(var_warm)

        var_cold = np.maximum(0.0, np.sum((X_u @ A_inv_cold) * X_u, axis=1))
        bonus_cold = 0.5 * np.sqrt(var_cold)

        # Department engagement vector
        dept_counts = Counter()
        for p_item, e_type, _, _ in curr_hist:
            dept_counts[p_item["category"]] += 1
        total_dept_ev = sum(dept_counts.values())

        dept_scores = np.zeros(n_catalog, dtype=np.float64)
        for i_c, pid_c in enumerate(idx_to_pid):
            cat_c = catalog[pid_c]["category"]
            d_share = (dept_counts[cat_c] / total_dept_ev) if total_dept_ev > 0 else 0.0
            dept_scores[i_c] = d_share

        score_matrix = np.empty((n_catalog, n_policies), dtype=np.float64)
        score_matrix[:, 0] = exp_cold + bonus_cold                    # Cold LinUCB
        score_matrix[:, 1] = exp_warm + bonus_warm                    # Warm LinUCB (UCB)
        score_matrix[:, 2] = exp_warm                                 # Warm LinUCB (Greedy)
        score_matrix[:, 3] = exp_warm + bonus_warm                    # Warm Discounted (Frozen)
        score_matrix[:, 4] = exp_warm + rng.normal(0.0, 0.25, size=n_catalog) * np.sqrt(var_warm)  # LinTS
        score_matrix[:, 5] = exp_warm                                 # Warm Epsilon-Greedy (Greedy)
        score_matrix[:, 6] = dept_scores * 100.0 + pop_priors         # Department Heuristic
        score_matrix[:, 7] = (1.0 + dept_scores) * (1.0 + pop_priors) # Hybrid
        score_matrix[:, 8] = pop_priors                               # Popularity Baseline
        score_matrix[:, 9] = rng.uniform(0.0, 1.0, size=n_catalog)    # Random Baseline
        score_matrix[:, 10] = exp_shuf + 0.5 * np.sqrt(var_warm)      # Shuffled Reward

        # Shuffled context control
        if shuf_context_pool:
            rand_hist = shuf_context_pool[rng.integers(len(shuf_context_pool))]
            rand_ctx = build_context_from_history(rand_hist, reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))
            X_shuf = build_candidate_matrix(catalog, idx_to_pid, rand_ctx)
            score_matrix[:, 11] = X_shuf @ theta_warm
        else:
            score_matrix[:, 11] = pop_priors

        # Compute full-catalog true item rank (1-based)
        target_scores = score_matrix[t_idx, :]
        ranks_full = np.count_nonzero(score_matrix > target_scores, axis=0) + 1

        # 3. Paired 50-Candidate Pool Construction
        # Same pool for all policies!
        # Guarantee min 2 per department + random popularity sampling
        cand50_set = {t_idx}
        for d, pids in dept_pids.items():
            d_indices = [pid_to_idx[p] for p in pids if pid_to_idx[p] not in cand50_set]
            if d_indices:
                k = min(len(d_indices), 2)
                chosen = rng.choice(d_indices, size=k, replace=False)
                cand50_set.update(chosen)

        needed = 50 - len(cand50_set)
        if needed > 0:
            remaining = [i for i in range(n_catalog) if i not in cand50_set]
            probs_rem = p_sampling[remaining] / np.sum(p_sampling[remaining])
            chosen_rem = rng.choice(remaining, size=needed, replace=False, p=probs_rem)
            cand50_set.update(chosen_rem)

        cand50_arr = list(cand50_set)
        rng.shuffle(cand50_arr)
        true_cand50_pos = cand50_arr.index(t_idx)

        # 50-candidate score matrix
        sub_scores = score_matrix[cand50_arr, :]
        sub_target_scores = score_matrix[t_idx, :]
        ranks_50 = np.count_nonzero(sub_scores > sub_target_scores, axis=0) + 1

        # Record event outcomes
        test_vids.append(v_id)
        test_pids.append(t_pid)
        test_types.append(ev_type)
        test_ctx_levels.append(ctx_level)
        test_c32_vals.append(c32_val)
        test_ranks_full.append(ranks_full.astype(np.int16))
        test_ranks_50.append(ranks_50.astype(np.int16))
        test_cand50_true_positions.append(true_cand50_pos)

        # Append interaction to visitor history for subsequent events
        curr_hist.append((catalog[t_pid], ev_type, ts, REWARD_MAP.get(ev_type, 2.0)))
        if len(curr_hist) > 20:
            curr_hist.pop(0)

        if n_processed % 100000 == 0:
            elapsed = time.time() - t_start_stream
            print(f"    Streamed {n_processed:,} events ({len(test_ranks_full):,} valid evaluations) in {elapsed:.1f}s", flush=True)

    conn_hist.close()

    test_vids = np.array(test_vids)
    test_pids = np.array(test_pids)
    test_types = np.array(test_types)
    test_ctx_levels = np.array(test_ctx_levels)
    test_c32_vals = np.array(test_c32_vals)
    test_ranks_full = np.array(test_ranks_full, dtype=np.int16)
    test_ranks_50 = np.array(test_ranks_50, dtype=np.int16)
    test_cand50_pos = np.array(test_cand50_true_positions, dtype=np.int8)

    n_test_total = len(test_vids)
    print(f"\nCompleted Full Test Split Stream: {n_test_total:,} evaluated interactions across {len(set(test_vids)):,} visitors.", flush=True)

    # Persist compact per-example integer ranks to disk
    ranks_path = DATA_DIR / "test_ranks_full.npz"
    np.savez_compressed(
        ranks_path,
        visitor_ids=test_vids,
        target_pids=test_pids,
        event_types=test_types,
        context_levels=test_ctx_levels,
        feature32_vals=test_c32_vals,
        ranks_full=test_ranks_full,
        ranks_50=test_ranks_50,
        cand50_positions=test_cand50_pos,
        policy_names=np.array(policy_names)
    )
    print(f"Persisted compact rank matrices to {ranks_path} ({ranks_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 3: DUAL ESTIMANDS (MICRO & MACRO) WITH 2,000 VISITOR BOOTSTRAP CIs
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 3: DUAL ESTIMANDS & 2,000 RESAMPLE VISITOR BOOTSTRAP", flush=True)
    print("=" * 60, flush=True)

    summary_records = []
    boot_distributions = {}

    for mode_name, ranks_arr in [("full_catalog", test_ranks_full), ("50_candidates", test_ranks_50)]:
        print(f"\n--- Candidate Mode: {mode_name} ---", flush=True)
        for p_idx, p_name in enumerate(policy_names):
            p_ranks = ranks_arr[:, p_idx]
            hit10 = (p_ranks <= 10).astype(np.float64)
            ndcg10 = np.where(p_ranks <= 10, 1.0 / np.log2(p_ranks + 1.0), 0.0)
            mrr = np.where(p_ranks <= 100, 1.0 / p_ranks, 0.0)

            res_hr10 = bootstrap_ci_dual(hit10, test_vids, n_boot=2000, rng=rng)
            res_ndcg10 = bootstrap_ci_dual(ndcg10, test_vids, n_boot=2000, rng=rng)
            res_mrr = bootstrap_ci_dual(mrr, test_vids, n_boot=2000, rng=rng)

            boot_distributions[(mode_name, p_name)] = res_hr10

            rec = {
                "candidate_mode": mode_name,
                "policy": p_name,
                "macro_hr@10": res_hr10["macro_mean"],
                "macro_hr@10_ci_low": res_hr10["macro_ci"][0],
                "macro_hr@10_ci_high": res_hr10["macro_ci"][1],
                "micro_hr@10": res_hr10["micro_mean"],
                "micro_hr@10_ci_low": res_hr10["micro_ci"][0],
                "micro_hr@10_ci_high": res_hr10["micro_ci"][1],
                "macro_ndcg@10": res_ndcg10["macro_mean"],
                "macro_ndcg@10_ci_low": res_ndcg10["macro_ci"][0],
                "macro_ndcg@10_ci_high": res_ndcg10["macro_ci"][1],
                "micro_ndcg@10": res_ndcg10["micro_mean"],
                "micro_ndcg@10_ci_low": res_ndcg10["micro_ci"][0],
                "micro_ndcg@10_ci_high": res_ndcg10["micro_ci"][1],
                "macro_mrr": res_mrr["macro_mean"],
                "micro_mrr": res_mrr["micro_mean"],
            }
            summary_records.append(rec)
            print(f"  {p_name:28s}: Macro HR@10 = {rec['macro_hr@10']*100:.2f}% [{rec['macro_hr@10_ci_low']*100:.2f}%, {rec['macro_hr@10_ci_high']*100:.2f}%] | Micro = {rec['micro_hr@10']*100:.2f}% [{rec['micro_hr@10_ci_low']*100:.2f}%, {rec['micro_hr@10_ci_high']*100:.2f}%]")

    df_test_summary = pd.DataFrame(summary_records)
    df_test_summary.to_csv(RESULTS_DIR / "retailrocket_offline_eval_test_9c.csv", index=False)
    print("\nSaved dual estimands table to experiments/results/retailrocket_offline_eval_test_9c.csv", flush=True)

    # Paired hypothesis tests
    print("\n--- Paired Bootstrap Differences vs Baselines (Visitor-Level Macro) ---", flush=True)
    boot_warm = boot_distributions[("full_catalog", "Warm LinUCB (Optimal UCB)")]["boot_macro"]
    boot_cold = boot_distributions[("full_catalog", "Cold LinUCB (alpha=0.5)")]["boot_macro"]
    boot_pop = boot_distributions[("full_catalog", "Popularity Baseline")]["boot_macro"]
    boot_dept = boot_distributions[("full_catalog", "Department Heuristic")]["boot_macro"]
    boot_lints = boot_distributions[("full_catalog", "Warm LinTS (v=0.25)")]["boot_macro"]

    diff_wc, ci_wc, p_wc = paired_bootstrap_test(boot_warm, boot_cold)
    diff_wp, ci_wp, p_wp = paired_bootstrap_test(boot_warm, boot_pop)
    diff_wd, ci_wd, p_wd = paired_bootstrap_test(boot_warm, boot_dept)
    diff_tsp, ci_tsp, p_tsp = paired_bootstrap_test(boot_lints, boot_pop)
    diff_tsd, ci_tsd, p_tsd = paired_bootstrap_test(boot_lints, boot_dept)

    print(f"  Warm LinUCB vs Cold LinUCB: Lift = {diff_wc*100:+.2f}% (95% CI: [{ci_wc[0]*100:+.2f}%, {ci_wc[1]*100:+.2f}%], p {p_wc})")
    print(f"  Warm LinUCB vs Popularity:   Lift = {diff_wp*100:+.2f}% (95% CI: [{ci_wp[0]*100:+.2f}%, {ci_wp[1]*100:+.2f}%], p {p_wp})")
    print(f"  Warm LinUCB vs Dept Heurist: Lift = {diff_wd*100:+.2f}% (95% CI: [{ci_wd[0]*100:+.2f}%, {ci_wd[1]*100:+.2f}%], p {p_wd})")
    print(f"  Warm LinTS  vs Popularity:   Lift = {diff_tsp*100:+.2f}% (95% CI: [{ci_tsp[0]*100:+.2f}%, {ci_tsp[1]*100:+.2f}%], p {p_tsp})")
    print(f"  Warm LinTS  vs Dept Heurist: Lift = {diff_tsd*100:+.2f}% (95% CI: [{ci_tsd[0]*100:+.2f}%, {ci_tsd[1]*100:+.2f}%], p {p_tsd})")

    # -------------------------------------------------------------------------
    # STAGE 4: SLICED EVALUATIONS WITH SAMPLE SIZES N, CIs & LOW-N FLAGS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 4: SLICED EVALUATIONS WITH SAMPLE SIZES & CIs", flush=True)
    print("=" * 60, flush=True)

    slice_records = []
    warm_ranks_50 = test_ranks_50[:, 1]  # Warm LinUCB
    cold_ranks_50 = test_ranks_50[:, 0]  # Cold LinUCB

    # Slice by Target Interaction Type
    print("\n--- Slices by Target Event Type ---", flush=True)
    for etype in ["transaction", "addtocart", "view"]:
        mask = (test_types == etype)
        n_slice = int(np.count_nonzero(mask))
        low_n_flag = "LOW_N (<100)" if n_slice < 100 else "VALID"
        vids_slice = test_vids[mask]

        res_warm = bootstrap_ci_dual((warm_ranks_50[mask] <= 10).astype(float), vids_slice, n_boot=1000, rng=rng)
        res_cold = bootstrap_ci_dual((cold_ranks_50[mask] <= 10).astype(float), vids_slice, n_boot=1000, rng=rng)

        slice_records.append({
            "slice_category": "event_type",
            "slice_name": etype,
            "sample_size_n": n_slice,
            "status": low_n_flag,
            "warm_macro_hr@10": res_warm["macro_mean"],
            "warm_macro_ci_low": res_warm["macro_ci"][0],
            "warm_macro_ci_high": res_warm["macro_ci"][1],
            "warm_micro_hr@10": res_warm["micro_mean"],
            "cold_macro_hr@10": res_cold["macro_mean"],
            "cold_micro_hr@10": res_cold["micro_mean"]
        })
        print(f"  [{etype:12s}] n={n_slice:,} ({low_n_flag}): Warm Macro={res_warm['macro_mean']*100:.2f}% [{res_warm['macro_ci'][0]*100:.2f}%, {res_warm['macro_ci'][1]*100:.2f}%], Micro={res_warm['micro_mean']*100:.2f}%")

    # Slice by User Context Level
    print("\n--- Slices by Context Level ---", flush=True)
    for clevel in ["cold", "light", "heavy"]:
        mask = (test_ctx_levels == clevel)
        n_slice = int(np.count_nonzero(mask))
        low_n_flag = "LOW_N (<100)" if n_slice < 100 else "VALID"
        vids_slice = test_vids[mask]

        res_warm = bootstrap_ci_dual((warm_ranks_50[mask] <= 10).astype(float), vids_slice, n_boot=1000, rng=rng)
        res_cold = bootstrap_ci_dual((cold_ranks_50[mask] <= 10).astype(float), vids_slice, n_boot=1000, rng=rng)

        slice_records.append({
            "slice_category": "context_level",
            "slice_name": clevel,
            "sample_size_n": n_slice,
            "status": low_n_flag,
            "warm_macro_hr@10": res_warm["macro_mean"],
            "warm_macro_ci_low": res_warm["macro_ci"][0],
            "warm_macro_ci_high": res_warm["macro_ci"][1],
            "warm_micro_hr@10": res_warm["micro_mean"],
            "cold_macro_hr@10": res_cold["macro_mean"],
            "cold_micro_hr@10": res_cold["micro_mean"]
        })
        print(f"  [{clevel:12s}] n={n_slice:,} ({low_n_flag}): Warm Macro={res_warm['macro_mean']*100:.2f}% [{res_warm['macro_ci'][0]*100:.2f}%, {res_warm['macro_ci'][1]*100:.2f}%], Micro={res_warm['micro_mean']*100:.2f}%")

    df_slices = pd.DataFrame(slice_records)
    df_slices.to_csv(RESULTS_DIR / "retailrocket_offline_eval_slices_9c.csv", index=False)

    # 50-Candidate generator neutrality check
    mean_cand_pos = float(np.mean(test_cand50_pos))
    rand_hr10_50 = float(np.mean(test_ranks_50[:, 9] <= 10))
    print(f"\nCandidate Generator Sanity Check:")
    print(f"  True item mean position in 50-candidate pool: {mean_cand_pos:.2f} (Expected uniform = 24.50)")
    print(f"  Random Baseline HR@10 on 50 candidates:       {rand_hr10_50*100:.2f}% (Expected chance = 20.00%)")

    # -------------------------------------------------------------------------
    # STAGE 5: THREE TRAIN-ONLY MAPPINGS & POPULARITY CEILING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 5: MAPPING FAIRNESS & EMPIRICAL POPULARITY CEILING", flush=True)
    print("=" * 60, flush=True)

    mappings_to_test = [
        "popularity_balanced_train",
        "cooccurrence_cluster_train",
        "category_rank_train"
    ]
    mapping_results = []

    # Take a sample of 25,000 test events to evaluate across mappings
    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_hist = conn_hist.cursor()
    map_test_events = c_hist.execute("""
        SELECT visitor_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts >= ?
        ORDER BY visitor_id, ts
        LIMIT 35000;
    """, (T_VAL_END,)).fetchall()
    conn_hist.close()

    for m_name in mappings_to_test:
        cur_map = load_mapping(HISTORICAL_DB, m_name, "v1.0_train")
        
        # 1. Compute empirical test item frequencies to identify Test Popularity Ceiling
        mapped_test_pids = [cur_map[e[1]] for e in map_test_events if e[1] in cur_map and cur_map[e[1]] in catalog]
        test_pop_counts = Counter(mapped_test_pids)
        oracle_top10_pids = set([p for p, _ in test_pop_counts.most_common(10)])
        total_eval_map = len(mapped_test_pids)
        ceiling_hits = sum(1 for p in mapped_test_pids if p in oracle_top10_pids)
        popularity_ceiling_hr10 = (ceiling_hits / total_eval_map) if total_eval_map else 0.0

        # Compute train popularity baseline scores under this mapping
        # Count training period frequencies
        conn_hist = sqlite3.connect(HISTORICAL_DB)
        c_h = conn_hist.cursor()
        train_counts = Counter()
        for iid, cnt in c_h.execute("SELECT item_id, COUNT(*) FROM rr_events WHERE ts < ? GROUP BY item_id;", (T_TRAIN_END,)).fetchall():
            p = cur_map.get(iid)
            if p in catalog:
                train_counts[p] += cnt
        conn_hist.close()

        m_pop_priors = np.array([train_counts.get(p, 0) for p in idx_to_pid], dtype=float)
        m_top10_pop = set([idx_to_pid[i] for i in np.argsort(m_pop_priors)[::-1][:10]])
        pop_hits = sum(1 for p in mapped_test_pids if p in m_top10_pop)
        pop_hr10 = (pop_hits / total_eval_map) if total_eval_map else 0.0

        # Evaluate Warm LinUCB and Cold LinUCB
        ranks_w = []
        ranks_c = []
        v_hist_m = defaultdict(list)
        for v_id, i_id, ev_type, ts in map_test_events:
            pid = cur_map.get(i_id)
            if pid not in catalog:
                continue
            h = list(v_hist_m[v_id])
            ctx = build_context_from_history(h, reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))
            X = build_candidate_matrix(catalog, idx_to_pid, ctx)
            t_i = pid_to_idx[pid]

            # Warm
            sc_w = X @ theta_warm + 0.5 * np.sqrt(np.maximum(0.0, np.sum((X @ A_inv_warm) * X, axis=1)))
            ranks_w.append(np.count_nonzero(sc_w > sc_w[t_i]) + 1)

            # Cold
            sc_c = X @ theta_cold + 0.5 * np.sqrt(np.maximum(0.0, np.sum((X @ A_inv_cold) * X, axis=1)))
            ranks_c.append(np.count_nonzero(sc_c > sc_c[t_i]) + 1)

            v_hist_m[v_id].append((catalog[pid], ev_type, ts, REWARD_MAP.get(ev_type, 2.0)))
            if len(ranks_w) >= 25000:
                break

        hr_warm = float(np.mean(np.array(ranks_w) <= 10))
        hr_cold = float(np.mean(np.array(ranks_c) <= 10))

        mapping_results.append({
            "mapping_method": m_name,
            "popularity_ceiling_hr@10": popularity_ceiling_hr10,
            "popularity_baseline_hr@10": pop_hr10,
            "cold_linucb_hr@10": hr_cold,
            "warm_linucb_hr@10": hr_warm,
            "warm_lift_over_cold": hr_warm - hr_cold,
            "warm_lift_over_popularity": hr_warm - pop_hr10
        })
        print(f"  [{m_name:28s}] Popularity Ceiling = {popularity_ceiling_hr10*100:.2f}%, Pop Baseline = {pop_hr10*100:.2f}%, Cold LinUCB = {hr_cold*100:.2f}%, Warm LinUCB = {hr_warm*100:.2f}%")

    df_mappings = pd.DataFrame(mapping_results)
    df_mappings.to_csv(RESULTS_DIR / "retailrocket_mapping_comparison_9c.csv", index=False)

    # -------------------------------------------------------------------------
    # STAGE 6: FEATURE IMPORTANCE & SYSTEMATIC ABLATIONS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 6: FEATURE IMPORTANCE & GROUP ABLATIONS", flush=True)
    print("=" * 60, flush=True)

    # Feature names
    from app.services.feature_engineering import CANONICAL_FEATURE_SPEC
    feature_names = [
        "affinity_electronics (0)", "affinity_fashion (1)", "affinity_home_kitchen (2)",
        "affinity_beauty (3)", "affinity_sports (4)", "affinity_books (5)",
        "affinity_brand (6)", "user_avg_price_level (7)", "session_budget_cap (8)",
        "price_fit_score (9)", "in_budget_flag (10)", "category_search_match (11)",
        "cat_onehot_electronics (12)", "cat_onehot_fashion (13)", "cat_onehot_home_kitchen (14)",
        "cat_onehot_beauty (15)", "cat_onehot_sports (16)", "cat_onehot_books (17)",
        "product_price_norm (18)", "product_rating_norm (19)", "product_discount_pct (20)",
        "product_popularity (21)", "brand_hash_bin_0 (22)", "brand_hash_bin_1 (23)",
        "brand_hash_bin_2 (24)", "brand_hash_bin_3 (25)", "brand_hash_bin_4 (26)",
        "brand_hash_bin_5 (27)", "brand_hash_bin_6 (28)", "brand_hash_bin_7 (29)",
        "new_product_flag (30)", "user_activity_level (31)", "cart_intent_match (32)",
        "high_rating_boost (33)", "high_discount_boost (34)", "intercept_bias (35)"
    ]

    abs_th = np.abs(theta_warm)
    top_indices = np.argsort(abs_th)[::-1][:10]
    print("  Top 10 Features by |theta_i| in Warm LinUCB:")
    for rank_f, f_i in enumerate(top_indices, 1):
        print(f"    {rank_f:2d}. {feature_names[f_i]:32s}: theta = {theta_warm[f_i]:+7.4f} (|theta| = {abs_th[f_i]:.4f})")

    # Feature group ablations on a 25,000 test slice
    ablation_groups = {
        "Full Warm LinUCB": [],
        "Ablate: Popularity": [21],
        "Ablate: Category Affinity": [0, 1, 2, 3, 4, 5],
        "Ablate: Brand Affinity & Hash": [6, 22, 23, 24, 25, 26, 27, 28, 29],
        "Ablate: Price Fit & Budget": [7, 8, 9, 10, 18],
        "Ablate: Session & Cart (feat 32)": [11, 31, 32],
        "Intercept Only": [i for i in range(35)]
    }

    ablation_records = []
    # Sample 25,000 test events for ablation
    sub_eval = test_ranks_full[:25000]
    sub_vids = test_vids[:25000]

    for abl_name, zero_dims in ablation_groups.items():
        th_abl = np.copy(theta_warm)
        th_abl[zero_dims] = 0.0

        # Evaluate ranks
        ranks_abl = np.empty(len(sub_eval), dtype=np.int32)
        # Fast evaluation using sub_scores
        for i_ev in range(len(sub_eval)):
            # We reconstruct scores with masked theta
            ctx_i = build_context_from_history(val_vis_hist[sub_vids[i_ev]][:5], reference_time=datetime.now(timezone.utc))
            X_abl = build_candidate_matrix(catalog, idx_to_pid, ctx_i)
            X_abl[:, zero_dims] = 0.0
            scores = X_abl @ th_abl
            t_i = pid_to_idx[test_pids[i_ev]]
            ranks_abl[i_ev] = np.count_nonzero(scores > scores[t_i]) + 1

        res = bootstrap_ci_dual((ranks_abl <= 10).astype(float), sub_vids, n_boot=1000, rng=rng)
        ablation_records.append({
            "ablation_configuration": abl_name,
            "macro_hr@10": res["macro_mean"],
            "macro_ci_low": res["macro_ci"][0],
            "macro_ci_high": res["macro_ci"][1],
            "micro_hr@10": res["micro_mean"]
        })
        print(f"    {abl_name:32s}: Macro HR@10 = {res['macro_mean']*100:.2f}% [{res['macro_ci'][0]*100:.2f}%, {res['macro_ci'][1]*100:.2f}%]")

    # C1 specific: Test transaction and addtocart HR with feature 32 zeroed vs normal
    trans_mask = (test_types[:25000] == "transaction") | (test_types[:25000] == "addtocart")
    n_high_intent = int(np.count_nonzero(trans_mask))
    if n_high_intent > 0:
        th_no32 = np.copy(theta_warm)
        th_no32[32] = 0.0
        ranks_normal = sub_eval[trans_mask, 1]  # Warm LinUCB
        # Compute without feature 32
        ranks_no32 = np.empty(n_high_intent, dtype=np.int32)
        sub_indices = np.where(trans_mask)[0]
        for idx_pos, i_ev in enumerate(sub_indices):
            ctx_i = build_context_from_history(val_vis_hist[sub_vids[i_ev]][:5], reference_time=datetime.now(timezone.utc))
            X_i = build_candidate_matrix(catalog, idx_to_pid, ctx_i)
            X_i[:, 32] = 0.0
            scores = X_i @ th_no32
            t_i = pid_to_idx[test_pids[i_ev]]
            ranks_no32[idx_pos] = np.count_nonzero(scores > scores[t_i]) + 1

        hr_normal = float(np.mean(ranks_normal <= 10))
        hr_no32 = float(np.mean(ranks_no32 <= 10))
        print(f"\n  Feature 32 (Cart-Intent Match) Impact on High-Intent Events (n={n_high_intent}):")
        print(f"    Normal Feature 32: HR@10 = {hr_normal*100:.2f}%")
        print(f"    Zeroed Feature 32: HR@10 = {hr_no32*100:.2f}% (Delta = {(hr_normal - hr_no32)*100:+.2f}%)")

    df_abl = pd.DataFrame(ablation_records)
    df_abl.to_csv(RESULTS_DIR / "retailrocket_feature_ablation_9c.csv", index=False)

    # -------------------------------------------------------------------------
    # STAGE 7: PREQUENTIAL TEST REPLAY (20,000 CHRONOLOGICAL STEPS)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 7: PREQUENTIAL ONLINE TEST REPLAY (20,000 STEPS TABULAR)", flush=True)
    print("=" * 60, flush=True)

    rep_cold = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0)
    rep_warm = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0)
    rep_warm.warm_start_from_stats(stats_m3, n_cap=10000, lambda_reg=1.0)
    rep_lints = LinThompsonSamplingRecommender(feature_dim=FEATURE_DIMENSION, v_variance=0.25, lambda_reg=1.0)
    rep_lints.warm_start_from_stats(stats_m3, n_cap=10000, lambda_reg=1.0)
    rep_disc = DiscountedLinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0, gamma=0.95)
    rep_disc.warm_start_from_stats(stats_m3, n_cap=10000, lambda_reg=1.0)

    preq_checkpoints = [100, 250, 500, 1000, 2000, 5000, 10000, 20000]
    preq_records = []

    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_h = conn_hist.cursor()
    preq_events = c_h.execute("""
        SELECT visitor_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts >= ?
        ORDER BY ts
        LIMIT 25000;
    """, (T_VAL_END,)).fetchall()
    conn_hist.close()

    cold_hits, warm_hits, lints_hits, disc_hits = [], [], [], []
    v_hist_rep = defaultdict(list)

    cold_match_step = None
    initial_warm_hr = None

    for step_i, (v_id, i_id, ev_type, ts) in enumerate(preq_events):
        pid = primary_map.get(i_id)
        if pid not in catalog:
            continue

        ctx = build_context_from_history(v_hist_rep[v_id], reference_time=datetime.fromtimestamp(ts/1000.0, tz=timezone.utc))
        target_prod = catalog[pid]
        target_phi = extract_feature_vector(ctx, target_prod)
        rew = REWARD_MAP.get(ev_type, 2.0)

        X = build_candidate_matrix(catalog, idx_to_pid, ctx)
        t_i = pid_to_idx[pid]

        # 1. Cold LinUCB
        sc_c = X @ rep_cold.get_theta() + 0.5 * np.sqrt(np.maximum(0.0, np.sum((X @ rep_cold.A_inv) * X, axis=1)))
        rk_c = np.count_nonzero(sc_c > sc_c[t_i]) + 1
        h_c = 1.0 if rk_c <= 10 else 0.0
        rep_cold.update(target_phi, rew)
        cold_hits.append(h_c)

        # 2. Warm LinUCB
        sc_w = X @ rep_warm.get_theta() + 0.5 * np.sqrt(np.maximum(0.0, np.sum((X @ rep_warm.A_inv) * X, axis=1)))
        rk_w = np.count_nonzero(sc_w > sc_w[t_i]) + 1
        h_w = 1.0 if rk_w <= 10 else 0.0
        rep_warm.update(target_phi, rew)
        warm_hits.append(h_w)

        # 3. Warm LinTS
        sc_ts = X @ rep_lints.get_theta() + rng.normal(0.0, 0.25, size=n_catalog) * np.sqrt(np.maximum(0.0, np.sum((X @ rep_lints.A_inv) * X, axis=1)))
        rk_ts = np.count_nonzero(sc_ts > sc_ts[t_i]) + 1
        h_ts = 1.0 if rk_ts <= 10 else 0.0
        rep_lints.update(target_phi, rew)
        lints_hits.append(h_ts)

        # 4. Warm Discounted
        sc_d = X @ rep_disc.get_theta() + 0.5 * np.sqrt(np.maximum(0.0, np.sum((X @ rep_disc.A_inv) * X, axis=1)))
        rk_d = np.count_nonzero(sc_d > sc_d[t_i]) + 1
        h_d = 1.0 if rk_d <= 10 else 0.0
        rep_disc.update(target_phi, rew)
        disc_hits.append(h_d)

        v_hist_rep[v_id].append((target_prod, ev_type, ts, rew))

        current_n = len(cold_hits)
        if initial_warm_hr is None and current_n == 250:
            initial_warm_hr = np.mean(warm_hits[:250])

        if cold_match_step is None and initial_warm_hr is not None and current_n > 500:
            # Check rolling window of 250
            recent_cold = np.mean(cold_hits[-250:])
            if recent_cold >= initial_warm_hr:
                cold_match_step = current_n

        if current_n in preq_checkpoints:
            win = min(current_n, 500)
            rec_step = {
                "step": current_n,
                "cold_linucb_hr@10": float(np.mean(cold_hits[-win:])),
                "warm_linucb_hr@10": float(np.mean(warm_hits[-win:])),
                "warm_lints_hr@10": float(np.mean(lints_hits[-win:])),
                "warm_discounted_hr@10": float(np.mean(disc_hits[-win:]))
            }
            preq_records.append(rec_step)
            print(f"    Step {current_n:5d}: Cold LinUCB = {rec_step['cold_linucb_hr@10']*100:.2f}%, Warm LinUCB = {rec_step['warm_linucb_hr@10']*100:.2f}%, LinTS = {rec_step['warm_lints_hr@10']*100:.2f}%, Discounted = {rec_step['warm_discounted_hr@10']*100:.2f}%")

        if current_n >= 20000:
            break

    df_preq = pd.DataFrame(preq_records)
    df_preq.to_csv(RESULTS_DIR / "retailrocket_prequential_steps_9c.csv", index=False)
    print(f"\n  Prequential analysis: Initial Warm LinUCB HR@10 = {initial_warm_hr*100:.2f}%.")
    if cold_match_step:
        print(f"  Events needed for Cold LinUCB to match Warm LinUCB initial performance: {cold_match_step:,} events.")
    else:
        print("  Cold LinUCB did not catch up to Warm LinUCB initial performance within 20,000 steps.")

    # -------------------------------------------------------------------------
    # STAGE 8: HARDENED PUBLICATION PLOTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STAGE 8: GENERATING VISUALIZATIONS", flush=True)
    print("=" * 60, flush=True)

    # Plot 1: Dual Estimands Comparison Bar Chart
    plt.figure(figsize=(10, 5), dpi=150)
    df_sub = df_test_summary[df_test_summary["candidate_mode"] == "full_catalog"]
    x = np.arange(len(df_sub))
    w = 0.35
    plt.bar(x - w/2, df_sub["macro_hr@10"]*100, width=w, label="Visitor-Level (Macro)", color="#1f77b4")
    plt.bar(x + w/2, df_sub["micro_hr@10"]*100, width=w, label="Event-Level (Micro)", color="#aec7e8")
    plt.xticks(x, [p.split("(")[0].strip() for p in df_sub["policy"]], rotation=25, ha="right")
    plt.ylabel("HitRate@10 (%)")
    plt.title("Dual Estimands Comparison: Visitor-Level (Macro) vs Event-Level (Micro) on Full Catalog")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    p1 = PLOTS_DIR / "bootstrap_ci_dual_9c.png"
    plt.savefig(p1, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p1}", flush=True)

    # Plot 2: Prequential Trajectory
    plt.figure(figsize=(9, 5), dpi=150)
    w_size = 500
    r_cold = pd.Series(cold_hits).rolling(w_size).mean()
    r_warm = pd.Series(warm_hits).rolling(w_size).mean()
    r_ts = pd.Series(lints_hits).rolling(w_size).mean()
    r_disc = pd.Series(disc_hits).rolling(w_size).mean()
    plt.plot(r_warm * 100, label="Warm LinUCB (N_cap=10k)", color="#1f77b4", linewidth=2)
    plt.plot(r_cold * 100, label="Cold LinUCB", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.plot(r_ts * 100, label="Warm LinTS", color="#2ca02c", linestyle=":", linewidth=2)
    plt.plot(r_disc * 100, label="Warm Discounted LinUCB", color="#9467bd", linestyle="-.", linewidth=1.5)
    plt.title("Prequential Test Stream Replay: Rolling HitRate@10 (window=500)")
    plt.xlabel("Chronological Event Step")
    plt.ylabel("Rolling HitRate@10 (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p2 = PLOTS_DIR / "prequential_hitrate_9c.png"
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p2}", flush=True)

    # Plot 3: Mapping Ceiling Comparison
    plt.figure(figsize=(8, 4.5), dpi=150)
    x_m = np.arange(len(df_mappings))
    plt.bar(x_m - 0.2, df_mappings["popularity_ceiling_hr@10"]*100, width=0.4, label="Test Popularity Ceiling (Oracle)", color="#d62728", alpha=0.8)
    plt.bar(x_m + 0.2, df_mappings["warm_linucb_hr@10"]*100, width=0.4, label="Warm LinUCB", color="#1f77b4", alpha=0.9)
    plt.xticks(x_m, [m.replace("_train", "") for m in df_mappings["mapping_method"]], rotation=15)
    plt.ylabel("HitRate@10 (%)")
    plt.title("Mapping Comparison: Popularity Ceiling vs Warm LinUCB Performance")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    p3 = PLOTS_DIR / "mapping_comparison_ceiling_9c.png"
    plt.savefig(p3, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p3}", flush=True)

    # Plot 4: Feature Importance
    plt.figure(figsize=(8, 5), dpi=150)
    top_names = [feature_names[i].split("(")[0].strip() for i in top_indices]
    top_vals = [theta_warm[i] for i in top_indices]
    y_pos = np.arange(len(top_names))
    plt.barh(y_pos, top_vals, color=["#1f77b4" if v >= 0 else "#d62728" for v in top_vals])
    plt.yticks(y_pos, top_names)
    plt.xlabel("Learned Parameter Weight (theta_i)")
    plt.title("Top 10 Feature Weights in Warm LinUCB (N_cap=10k, m=3)")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.3)
    p4 = PLOTS_DIR / "feature_importance_9c.png"
    plt.savefig(p4, bbox_inches="tight")
    plt.close()
    print(f"  Saved {p4}", flush=True)

    print(f"\n[DONE] Hardened Phase 9C Evaluation Suite completed in {time.time() - start_time:.2f}s!", flush=True)


if __name__ == "__main__":
    run_hardened_evaluation()
