"""
SmartCart AI — Phase 12C Part B: Direct Erratum for the Frozen Offline Evaluation
Recomputes frozen headline metrics on the full test split (lifetime history, full catalog:
overall, novel, repeat, macro and micro with 2,000-resample visitor-clustered 95% CIs)
for:
  1. Popularity Baseline
  2. Department Heuristic
  3. RecentlyViewed
  4. Cold LinUCB
  5. Warm LinUCB

Compares against frozen Phase 10D results and reports before/after with exact deltas.
"""

import sys
import os
import json
import sqlite3
import time
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    CATEGORIES,
    MAX_PRICE_SCALE,
    UserContext,
    build_context_from_history,
    extract_feature_vector
)
from app.recommender.linucb import LinUCBRecommender
from scripts.retailrocket.shared_ranking import compute_rank
from scripts.retailrocket.offline_baselines import (
    PopularityBaseline, DepartmentHeuristicBaseline, RecentlyViewedBaseline
)

HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
DATA_DIR = ROOT_DIR / "data" / "historical"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

T_VAL_END = 1440756710277
FROZEN_NCAP = 10000
FROZEN_M = 3
FROZEN_LAMBDA = 1.0
FROZEN_ALPHA = 0.5

REWARD_MAP = {
    "view": 2.0,
    "addtocart": 5.0,
    "transaction": 10.0,
    "click": 2.0,
    "cart_add": 5.0,
    "purchase": 10.0
}


def load_catalog(db_path: Path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, name, category, brand, price, rating, discount_pct, popularity_score, popularity_prior
        FROM products WHERE is_active=1 ORDER BY id;
    """).fetchall()
    conn.close()

    catalog = {}
    dept_pids = defaultdict(list)
    pid_to_idx = {}
    idx_to_pid = []

    for idx, r in enumerate(rows):
        pid, name, cat, brand, price, rating, discount, pop, pop_prior = r
        catalog[pid] = {
            "id": pid, "name": name, "category": cat, "brand": brand or "",
            "price": float(price), "rating": float(rating), "discount_pct": float(discount),
            "popularity_score": float(pop), "popularity_prior": float(pop_prior if pop_prior is not None else 0.0)
        }
        dept_pids[cat].append(pid)
        pid_to_idx[pid] = idx
        idx_to_pid.append(pid)

    return catalog, dept_pids, pid_to_idx, idx_to_pid


def load_mapping(db_path: Path, method: str, version: str) -> Dict[int, int]:
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    rows = c.execute("SELECT item_id, product_id FROM rr_item_map WHERE mapping_method=? AND mapping_version=?;",
                     (method, version)).fetchall()
    conn.close()
    return dict(rows)


def build_candidate_matrix(catalog: dict, idx_to_pid: list, context: UserContext) -> np.ndarray:
    X = np.zeros((len(idx_to_pid), FEATURE_DIMENSION), dtype=np.float64)
    for i, pid in enumerate(idx_to_pid):
        X[i] = extract_feature_vector(context, catalog[pid])
    return X


def build_candidate_matrix_fast(X_static: np.ndarray, catalog: dict, idx_to_pid: list, catalog_prices: np.ndarray, context: UserContext) -> np.ndarray:
    if context.is_cold_start:
        return X_static

    X = np.copy(X_static)
    for i, cat in enumerate(CATEGORIES):
        X[:, i] = context.category_affinities.get(cat, 0.0)

    if context.brand_affinities:
        for i, pid in enumerate(idx_to_pid):
            b = catalog[pid]["brand"]
            if b in context.brand_affinities:
                X[i, 6] = context.brand_affinities[b]

    ref_price = context.avg_price if context.avg_price is not None else 1800.0
    X[:, 7] = math.log(1.0 + ref_price) / MAX_PRICE_SCALE

    budget = context.session_budget if context.session_budget is not None else ref_price
    X[:, 8] = math.log(1.0 + budget) / MAX_PRICE_SCALE

    target_log = math.log(1.0 + budget)
    price_logs = X_static[:, 18] * MAX_PRICE_SCALE
    price_diff = np.abs(price_logs - target_log)
    X[:, 9] = np.maximum(0.0, 1.0 - (price_diff / 2.0))

    X[:, 10] = (catalog_prices <= budget).astype(np.float64)
    X[:, 31] = min(1.0, math.log(1.0 + context.total_interactions) / math.log(50.0))

    total_cart = sum(context.cart_categories.values())
    if total_cart > 0:
        for i, pid in enumerate(idx_to_pid):
            cat = catalog[pid]["category"]
            X[i, 32] = context.cart_categories.get(cat, 0.0) / total_cart
    else:
        X[:, 32] = 0.0

    return X


def compute_visitor_bootstrap(
    hits_matrix: np.ndarray,
    v_codes: np.ndarray,
    v_counts: np.ndarray,
    n_v: int,
    n_boot: int = 2000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pols = hits_matrix.shape[1]
    rng = np.random.default_rng(seed)

    V_means = np.zeros((n_pols, n_v), dtype=np.float64)
    V_sums = np.zeros((n_pols, n_v), dtype=np.float64)

    for p in range(n_pols):
        s_p = np.bincount(v_codes, weights=hits_matrix[:, p], minlength=n_v)
        V_sums[p] = s_p
        V_means[p] = s_p / v_counts

    macro_points = np.mean(V_means, axis=1)
    tot_events = len(v_codes)
    micro_points = np.sum(V_sums, axis=1) / tot_events

    batch_size = 100
    n_batches = n_boot // batch_size
    boot_macro_list = []
    boot_micro_list = []

    for _ in range(n_batches):
        boot_idx = rng.choice(n_v, size=(batch_size, n_v), replace=True)
        b_macro = np.zeros((n_pols, batch_size), dtype=np.float64)
        b_micro = np.zeros((n_pols, batch_size), dtype=np.float64)
        sampled_counts = v_counts[boot_idx]
        sum_counts = np.sum(sampled_counts, axis=1)

        for p in range(n_pols):
            sampled_means = V_means[p, boot_idx]
            b_macro[p] = np.mean(sampled_means, axis=1)
            sampled_sums = V_sums[p, boot_idx]
            b_micro[p] = np.sum(sampled_sums, axis=1) / sum_counts

        boot_macro_list.append(b_macro)
        boot_micro_list.append(b_micro)

    boot_macro = np.concatenate(boot_macro_list, axis=1)
    boot_micro = np.concatenate(boot_micro_list, axis=1)

    macro_ci_low = np.percentile(boot_macro, 2.5, axis=1)
    macro_ci_high = np.percentile(boot_macro, 97.5, axis=1)
    micro_ci_low = np.percentile(boot_micro, 2.5, axis=1)
    micro_ci_high = np.percentile(boot_micro, 97.5, axis=1)

    return macro_points, macro_ci_low, macro_ci_high, micro_points, micro_ci_low, micro_ci_high


def run_part_b_erratum():
    print("=" * 80)
    print("      SMARTCART AI — PHASE 12C PART B: DIRECT OFFLINE ERRATUM EVALUATION")
    print("=" * 80)
    t_start = time.time()

    catalog, dept_pids, pid_to_idx, idx_to_pid = load_catalog(SMARTCART_DB)
    n_catalog = len(catalog)
    catalog_prices = np.array([catalog[p]["price"] for p in idx_to_pid], dtype=np.float64)
    popularity_priors = np.array([catalog[p]["popularity_prior"] for p in idx_to_pid], dtype=np.float64)

    primary_map = load_mapping(HISTORICAL_DB, "popularity_balanced_train", "v1.0_train")

    stats_m3 = np.load(DATA_DIR / "sufficient_stats_m3.npz", allow_pickle=True)
    warm_linucb = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=FROZEN_ALPHA, lambda_reg=FROZEN_LAMBDA)
    warm_linucb.warm_start_from_stats(stats_m3, n_cap=FROZEN_NCAP, lambda_reg=FROZEN_LAMBDA)
    theta_warm = warm_linucb.get_theta()
    A_inv_warm = warm_linucb.A_inv

    cold_ctx_default = UserContext()
    X_cold = build_candidate_matrix(catalog, idx_to_pid, cold_ctx_default)
    exp_warm_cold = X_cold @ theta_warm
    var_warm_cold = np.maximum(0.0, np.sum((X_cold @ A_inv_warm) * X_cold, axis=1))
    bonus_warm_cold = FROZEN_ALPHA * np.sqrt(var_warm_cold)

    theta_cold = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    A_inv_cold = np.eye(FEATURE_DIMENSION, dtype=np.float64) / FROZEN_LAMBDA
    exp_cold_static = X_cold @ theta_cold
    var_cold_static = np.maximum(0.0, np.sum((X_cold @ A_inv_cold) * X_cold, axis=1))
    bonus_cold_static = FROZEN_ALPHA * np.sqrt(var_cold_static)

    pop_baseline = PopularityBaseline(catalog, idx_to_pid)
    dept_baseline = DepartmentHeuristicBaseline(catalog, idx_to_pid, dept_pids)
    rv_baseline = RecentlyViewedBaseline(catalog, idx_to_pid)

    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_hist = conn_hist.cursor()
    test_stream = c_hist.execute("""
        SELECT visitor_id, session_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts >= ?
        ORDER BY visitor_id, ts ASC;
    """, (T_VAL_END,)).fetchall()
    conn_hist.close()

    n_events = len(test_stream)
    print(f"Total test interactions loaded: {n_events:,}")

    policies = [
        "Popularity Baseline",
        "Department Heuristic",
        "RecentlyViewed",
        "Cold LinUCB",
        "Warm LinUCB"
    ]
    n_pols = len(policies)

    hits_life_full = np.zeros((n_events, n_pols), dtype=np.float32)
    is_novel_mask = np.zeros(n_events, dtype=bool)
    test_vids = []
    visitor_engaged_pids = defaultdict(set)
    v_lifetime_hist = defaultdict(list)

    t_eval_start = time.time()

    for ev_idx, (v_id, s_id, i_id, ev_type, ts) in enumerate(test_stream):
        t_pid = primary_map.get(i_id)
        t_idx = pid_to_idx[t_pid]
        test_vids.append(v_id)

        # Lifetime novel vs repeat
        if t_pid not in visitor_engaged_pids[v_id]:
            is_novel_mask[ev_idx] = True
        visitor_engaged_pids[v_id].add(t_pid)

        life_hist = v_lifetime_hist[v_id]
        ref_dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)

        # Baselines
        sc_pop = pop_baseline.priors
        sc_dept_life = dept_baseline.score(life_hist)
        sc_rv_life = rv_baseline.score(life_hist)

        # LinUCB
        if len(life_hist) == 0:
            exp_w_life = exp_warm_cold
            bonus_w_life = bonus_warm_cold
            exp_c_life = exp_cold_static
            bonus_c_life = bonus_cold_static
        else:
            ctx_life = build_context_from_history(life_hist, reference_time=ref_dt)
            X_life = build_candidate_matrix_fast(X_cold, catalog, idx_to_pid, catalog_prices, ctx_life)
            exp_w_life = X_life @ theta_warm
            var_w_life = np.maximum(0.0, np.sum((X_life @ A_inv_warm) * X_life, axis=1))
            bonus_w_life = FROZEN_ALPHA * np.sqrt(var_w_life)
            exp_c_life = X_life @ theta_cold
            var_c_life = np.maximum(0.0, np.sum((X_life @ A_inv_cold) * X_life, axis=1))
            bonus_c_life = FROZEN_ALPHA * np.sqrt(var_c_life)

        sc_linucb_life = exp_w_life + bonus_w_life
        sc_cold_life = exp_c_life + bonus_c_life

        # Ranks
        hits_life_full[ev_idx, 0] = 1.0 if compute_rank(sc_pop, t_idx, popularity_priors) <= 10 else 0.0
        hits_life_full[ev_idx, 1] = 1.0 if compute_rank(sc_dept_life, t_idx, popularity_priors) <= 10 else 0.0
        hits_life_full[ev_idx, 2] = 1.0 if compute_rank(sc_rv_life, t_idx, popularity_priors) <= 10 else 0.0
        hits_life_full[ev_idx, 3] = 1.0 if compute_rank(sc_cold_life, t_idx, popularity_priors) <= 10 else 0.0
        hits_life_full[ev_idx, 4] = 1.0 if compute_rank(sc_linucb_life, t_idx, popularity_priors) <= 10 else 0.0

        item_tuple = (catalog[t_pid], ev_type, ts, REWARD_MAP.get(ev_type, 2.0))
        life_hist.append(item_tuple)
        if len(life_hist) > 20:
            life_hist.pop(0)

        if (ev_idx + 1) % 100000 == 0:
            print(f"  Processed {ev_idx+1:,} / {n_events:,} events in {time.time()-t_eval_start:.1f}s...", flush=True)

    print(f"Streaming evaluation finished in {time.time()-t_eval_start:.1f}s.")

    # Bootstrapping
    test_vids_arr = np.array(test_vids)
    unique_visitors, v_codes = np.unique(test_vids_arr, return_inverse=True)
    n_unique_v = len(unique_visitors)
    v_counts = np.bincount(v_codes, minlength=n_unique_v)

    slices = [
        ("overall", np.ones(n_events, dtype=bool)),
        ("novel", is_novel_mask),
        ("repeat", ~is_novel_mask)
    ]

    regenerated_results = {}

    for slice_name, mask in slices:
        sub_hits = hits_life_full[mask]
        sub_vcodes = v_codes[mask]
        _, sub_v_renumbered = np.unique(sub_vcodes, return_inverse=True)
        sub_nv = len(np.unique(sub_vcodes))
        sub_vcounts = np.bincount(sub_v_renumbered, minlength=sub_nv)

        macro_m, macro_l, macro_h, micro_m, micro_l, micro_h = compute_visitor_bootstrap(
            sub_hits, sub_v_renumbered, sub_vcounts, sub_nv, n_boot=2000, seed=42
        )

        regenerated_results[slice_name] = {}
        for p_idx, pol in enumerate(policies):
            regenerated_results[slice_name][pol] = {
                "macro_hr@10": {
                    "mean": float(macro_m[p_idx]),
                    "ci_low": float(macro_l[p_idx]),
                    "ci_high": float(macro_h[p_idx])
                },
                "micro_hr@10": {
                    "mean": float(micro_m[p_idx]),
                    "ci_low": float(micro_l[p_idx]),
                    "ci_high": float(micro_h[p_idx])
                }
            }

    # Load frozen Phase 10D results
    with open(RESULTS_DIR / "offline_master_results.json", "r", encoding="utf-8") as f:
        frozen_master = json.load(f)

    frozen_evals = frozen_master["evaluations"]["lifetime_history"]

    print("\n" + "=" * 105)
    print("=== PART B ERRATUM TABLE: FROZEN VALUE vs REGENERATED VALUE (FULL CATALOG, LIFETIME) ===")
    print("=" * 105)

    max_delta_pp = 0.0
    erratum_data = {}

    for slice_name in ["overall", "novel", "repeat"]:
        print(f"\n--- Slice: {slice_name.upper()} ---")
        print(f"{'Policy':<22} | {'Metric':<8} | {'Frozen Phase 10D':<25} | {'Regenerated (Post-Fix)':<25} | {'Delta pp':<10}")
        print("-" * 100)
        erratum_data[slice_name] = {}
        
        # Build policy map from frozen list
        frozen_slice_list = frozen_evals[slice_name]["full_catalog"]
        frozen_slice_map = {item["policy"]: item for item in frozen_slice_list}

        for pol in policies:
            f_pol = frozen_slice_map.get(pol, {})
            r_pol = regenerated_results[slice_name].get(pol, {})

            f_mac_m = f_pol.get("macro_hr@10", 0) * 100
            f_mac_ci_arr = f_pol.get("macro_ci", [0, 0])
            r_macro = r_pol.get("macro_hr@10", {})
            r_mac_m = r_macro.get("mean", 0) * 100
            r_mac_ci_arr = [r_macro.get("ci_low", 0), r_macro.get("ci_high", 0)]
            mac_delta = r_mac_m - f_mac_m
            f_mac_ci = f"[{f_mac_ci_arr[0]*100:.2f}%, {f_mac_ci_arr[1]*100:.2f}%]"
            r_mac_ci = f"[{r_mac_ci_arr[0]*100:.2f}%, {r_mac_ci_arr[1]*100:.2f}%]"
            print(f"{pol:<22} | Macro    | {f_mac_m:5.2f}% {f_mac_ci:<18} | {r_mac_m:5.2f}% {r_mac_ci:<18} | {mac_delta:+6.3f} pp")

            f_mic_m = f_pol.get("micro_hr@10", 0) * 100
            f_mic_ci_arr = f_pol.get("micro_ci", [0, 0])
            r_micro = r_pol.get("micro_hr@10", {})
            r_mic_m = r_micro.get("mean", 0) * 100
            r_mic_ci_arr = [r_micro.get("ci_low", 0), r_micro.get("ci_high", 0)]
            mic_delta = r_mic_m - f_mic_m
            f_mic_ci = f"[{f_mic_ci_arr[0]*100:.2f}%, {f_mic_ci_arr[1]*100:.2f}%]"
            r_mic_ci = f"[{r_mic_ci_arr[0]*100:.2f}%, {r_mic_ci_arr[1]*100:.2f}%]"
            print(f"{pol:<22} | Micro    | {f_mic_m:5.2f}% {f_mic_ci:<18} | {r_mic_m:5.2f}% {r_mic_ci:<18} | {mic_delta:+6.3f} pp")

            max_delta_pp = max(max_delta_pp, abs(mac_delta), abs(mic_delta))
            erratum_data[slice_name][pol] = {
                "macro_frozen": {"mean": f_pol.get("macro_hr@10", 0), "ci": f_mac_ci_arr},
                "macro_regenerated": r_macro,
                "macro_delta_pp": mac_delta,
                "micro_frozen": {"mean": f_pol.get("micro_hr@10", 0), "ci": f_mic_ci_arr},
                "micro_regenerated": r_micro,
                "micro_delta_pp": mic_delta
            }

    print("\n" + "=" * 105)
    print(f"Maximum discrepancy across all cells: {max_delta_pp:.4f} percentage points.")
    if max_delta_pp > 0.3:
        print("ALERT: Maximum discrepancy EXCEEDS 0.3 percentage points! Docs must be updated.")
    else:
        print("CONFIRMED: Discrepancy <= 0.3 percentage points across all cells. Offline evaluation claims remain intact.")
    print("=" * 105)

    # Save erratum output JSON
    output_path = RESULTS_DIR / "offline_erratum_phase12c.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "max_delta_pp": max_delta_pp,
            "exceeds_threshold_0_3pp": bool(max_delta_pp > 0.3),
            "results": erratum_data
        }, f, indent=2)
    print(f"Erratum saved to {output_path}")

if __name__ == "__main__":
    run_part_b_erratum()
