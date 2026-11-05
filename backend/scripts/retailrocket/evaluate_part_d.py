"""
SmartCart AI — Phase 10C Part D Evaluation Runner
Computes the Part D tables:
1. Table D1: Overall test events (unmasked), Full Catalog & 50 Candidates:
   - Popularity Baseline
   - Department Heuristic
   - RecentlyViewed
   - Warm LinUCB
   - Warm LinTS
   - Hybrid slate (k=3)
   - Hybrid slate (k=5)
   With 2,000-resample visitor-clustered 95% CIs and paired CIs of Hybrid vs RV and Hybrid vs Warm LinUCB.

2. Table D2: Pure discovery task (Novel targets with all previously engaged products masked, Exclusion v2):
   - Popularity Baseline
   - Department Heuristic
   - Warm LinUCB
   - Warm LinTS
   With 2,000-resample visitor-clustered 95% CIs and paired CIs versus Popularity Baseline.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from scripts.retailrocket.evaluate_phase10b import (
    load_catalog, load_mapping, build_candidate_matrix, build_candidate_matrix_fast,
    extract_feature_vector, LinUCBRecommender, LinThompsonSamplingRecommender,
    HISTORICAL_DB, SMARTCART_DB, DATA_DIR, RESULTS_DIR,
    FEATURE_DIMENSION, FEATURE_SPEC_HASH, FROZEN_NCAP, FROZEN_M,
    FROZEN_LAMBDA, FROZEN_ALPHA, T_VAL_END, REWARD_MAP
)
from app.services.feature_engineering import UserContext, build_context_from_history
from scripts.retailrocket.offline_baselines import (
    PopularityBaseline, DepartmentHeuristicBaseline, RecentlyViewedBaseline
)


def run_part_d():
    print("================================================================================")
    print("      SMARTCART AI - PHASE 10C PART D EVALUATION RUNNER                        ")
    print("================================================================================")
    t0 = time.time()
    rng = np.random.default_rng(42)

    # 1. Load active catalog
    catalog, dept_pids, pid_to_idx, idx_to_pid = load_catalog(SMARTCART_DB)
    n_catalog = len(catalog)
    print(f"Catalog: Loaded {n_catalog} active SKUs across {len(dept_pids)} departments.")

    # 2. Load primary training mapping
    primary_map = load_mapping(HISTORICAL_DB, "popularity_balanced_train", "v1.0_train")

    # 3. Load warm-start models
    stats_m3 = np.load(DATA_DIR / "sufficient_stats_m3.npz", allow_pickle=True)
    warm_linucb = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=FROZEN_ALPHA, lambda_reg=FROZEN_LAMBDA)
    warm_linucb.warm_start_from_stats(stats_m3, n_cap=FROZEN_NCAP, lambda_reg=FROZEN_LAMBDA)
    theta_warm = warm_linucb.get_theta()
    A_inv_warm = warm_linucb.A_inv

    # Baselines
    pop_baseline = PopularityBaseline(catalog, idx_to_pid)
    dept_baseline = DepartmentHeuristicBaseline(catalog, idx_to_pid, dept_pids)
    rv_baseline = RecentlyViewedBaseline(catalog, idx_to_pid)

    # Cold context precomputation
    cold_ctx_default = UserContext()
    X_cold = build_candidate_matrix(catalog, idx_to_pid, cold_ctx_default)
    catalog_prices = np.array([catalog[p]["price"] for p in idx_to_pid], dtype=np.float64)

    exp_warm_cold = X_cold @ theta_warm
    var_warm_cold = np.maximum(0.0, np.sum((X_cold @ A_inv_warm) * X_cold, axis=1))
    bonus_warm_cold = 0.5 * np.sqrt(var_warm_cold)

    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_hist = conn_hist.cursor()
    test_cursor = c_hist.execute("""
        SELECT visitor_id, session_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts >= ?
        ORDER BY visitor_id, ts ASC;
    """, (T_VAL_END,)).fetchall()
    conn_hist.close()

    n_events = len(test_cursor)
    print(f"Streaming {n_events:,} chronological test interactions...")

    # Data structures for tracking hits
    # Policies for Overall Table:
    # 0: Popularity
    # 1: Department Heuristic
    # 2: RecentlyViewed
    # 3: Warm LinUCB
    # 4: Warm LinTS
    # 5: Hybrid slate (k=3)
    # 6: Hybrid slate (k=5)
    overall_pols = [
        "Popularity Baseline",
        "Department Heuristic",
        "RecentlyViewed",
        "Warm LinUCB",
        "Warm LinTS (v=0.25)",
        "Hybrid Slate (k=3 RV + 7 LinUCB)",
        "Hybrid Slate (k=5 RV + 5 LinUCB)"
    ]
    n_pols_d = len(overall_pols)

    hits_full_uncon = np.zeros((n_events, n_pols_d), dtype=np.float32)
    hits_50_uncon = np.zeros((n_events, n_pols_d), dtype=np.float32)

    # Pure discovery (Novel targets with Exclusion v2)
    # Policies:
    # 0: Popularity
    # 1: Department Heuristic
    # 2: Warm LinUCB
    # 3: Warm LinTS
    disc_pols = [
        "Popularity Baseline",
        "Department Heuristic",
        "Warm LinUCB",
        "Warm LinTS (v=0.25)"
    ]
    n_pols_disc = len(disc_pols)
    hits_full_disc = np.zeros((n_events, n_pols_disc), dtype=np.float32)
    hits_50_disc = np.zeros((n_events, n_pols_disc), dtype=np.float32)

    test_vids = []
    is_novel_mask = np.zeros(n_events, dtype=bool)

    visitor_engaged_pids = defaultdict(set)
    curr_v = None
    curr_hist = []
    curr_sess_ts = 0

    t_stream_start = time.time()

    for ev_idx, (v_id, s_id, i_id, ev_type, ts) in enumerate(test_cursor):
        if v_id != curr_v or (ts - curr_sess_ts > 1800000):
            curr_v = v_id
            curr_hist = []
        curr_sess_ts = ts

        t_pid = primary_map.get(i_id)
        if t_pid not in catalog:
            continue
        t_idx = pid_to_idx[t_pid]

        is_rep = (t_pid in visitor_engaged_pids[v_id])
        if not is_rep:
            is_novel_mask[ev_idx] = True

        test_vids.append(v_id)

        # Context scoring
        if len(curr_hist) == 0:
            exp_warm = exp_warm_cold
            bonus_warm = bonus_warm_cold
            var_warm = var_warm_cold
            dept_scores = np.zeros(n_catalog, dtype=np.float64)
            rv_scores = pop_baseline.priors.copy()
        else:
            ctx = build_context_from_history(curr_hist, reference_time=datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc))
            X_u = build_candidate_matrix_fast(X_cold, catalog, idx_to_pid, catalog_prices, ctx)
            exp_warm = X_u @ theta_warm
            var_warm = np.maximum(0.0, np.sum((X_u @ A_inv_warm) * X_u, axis=1))
            bonus_warm = 0.5 * np.sqrt(var_warm)

            t_cat = catalog[t_pid]["category"]
            dept_scores = dept_baseline.score_for_target_dept(t_cat)
            rv_scores = rv_baseline.score(curr_hist)

        pop_scores = pop_baseline.priors
        linucb_scores = exp_warm + bonus_warm
        noise_ts = rng.normal(0.0, 0.25, size=n_catalog)
        lints_scores = exp_warm + noise_ts * np.sqrt(var_warm)

        # 50 candidates pool
        avail = [i for i in range(n_catalog) if i != t_idx]
        cand_49 = rng.choice(avail, size=49, replace=False)
        pool_50 = np.concatenate([[t_idx], cand_49])

        # --- A. FULL CATALOG UNCONSTRAINED ---
        # 0: Popularity
        r_pop = np.count_nonzero(pop_scores > pop_scores[t_idx]) + 1
        hits_full_uncon[ev_idx, 0] = 1.0 if r_pop <= 10 else 0.0

        # 1: Department Heuristic
        r_dept = np.count_nonzero(dept_scores > dept_scores[t_idx]) + 1
        hits_full_uncon[ev_idx, 1] = 1.0 if r_dept <= 10 else 0.0

        # 2: RecentlyViewed
        r_rv = np.count_nonzero(rv_scores > rv_scores[t_idx]) + 1
        hits_full_uncon[ev_idx, 2] = 1.0 if r_rv <= 10 else 0.0

        # 3: Warm LinUCB
        r_linucb = np.count_nonzero(linucb_scores > linucb_scores[t_idx]) + 1
        hits_full_uncon[ev_idx, 3] = 1.0 if r_linucb <= 10 else 0.0

        # 4: Warm LinTS
        r_lints = np.count_nonzero(lints_scores > lints_scores[t_idx]) + 1
        hits_full_uncon[ev_idx, 4] = 1.0 if r_lints <= 10 else 0.0

        # 5: Hybrid slate (k=3)
        top3_rv = np.argsort(-rv_scores)[:3]
        sc_lin_rem3 = linucb_scores.copy()
        sc_lin_rem3[top3_rv] = -1e9
        top7_lin = np.argsort(-sc_lin_rem3)[:7]
        slate_3 = np.concatenate([top3_rv, top7_lin])
        hits_full_uncon[ev_idx, 5] = 1.0 if t_idx in slate_3 else 0.0

        # 6: Hybrid slate (k=5)
        top5_rv = np.argsort(-rv_scores)[:5]
        sc_lin_rem5 = linucb_scores.copy()
        sc_lin_rem5[top5_rv] = -1e9
        top5_lin = np.argsort(-sc_lin_rem5)[:5]
        slate_5 = np.concatenate([top5_rv, top5_lin])
        hits_full_uncon[ev_idx, 6] = 1.0 if t_idx in slate_5 else 0.0

        # --- B. 50 CANDIDATES UNCONSTRAINED ---
        pop_50 = pop_scores[pool_50]
        hits_50_uncon[ev_idx, 0] = 1.0 if (np.count_nonzero(pop_50 > pop_50[0]) + 1) <= 10 else 0.0

        dept_50 = dept_scores[pool_50]
        hits_50_uncon[ev_idx, 1] = 1.0 if (np.count_nonzero(dept_50 > dept_50[0]) + 1) <= 10 else 0.0

        rv_50 = rv_scores[pool_50]
        hits_50_uncon[ev_idx, 2] = 1.0 if (np.count_nonzero(rv_50 > rv_50[0]) + 1) <= 10 else 0.0

        linucb_50 = linucb_scores[pool_50]
        hits_50_uncon[ev_idx, 3] = 1.0 if (np.count_nonzero(linucb_50 > linucb_50[0]) + 1) <= 10 else 0.0

        lints_50 = lints_scores[pool_50]
        hits_50_uncon[ev_idx, 4] = 1.0 if (np.count_nonzero(lints_50 > lints_50[0]) + 1) <= 10 else 0.0

        # Hybrid 50 (k=3)
        top3_rv_50 = np.argsort(-rv_50)[:3]
        sc_lin_rem3_50 = linucb_50.copy()
        sc_lin_rem3_50[top3_rv_50] = -1e9
        top7_lin_50 = np.argsort(-sc_lin_rem3_50)[:7]
        slate_3_50 = np.concatenate([top3_rv_50, top7_lin_50])
        hits_50_uncon[ev_idx, 5] = 1.0 if 0 in slate_3_50 else 0.0

        # Hybrid 50 (k=5)
        top5_rv_50 = np.argsort(-rv_50)[:5]
        sc_lin_rem5_50 = linucb_50.copy()
        sc_lin_rem5_50[top5_rv_50] = -1e9
        top5_lin_50 = np.argsort(-sc_lin_rem5_50)[:5]
        slate_5_50 = np.concatenate([top5_rv_50, top5_lin_50])
        hits_50_uncon[ev_idx, 6] = 1.0 if 0 in slate_5_50 else 0.0

        # --- C. PURE DISCOVERY (EXCLUSION V2) ---
        excl_v2_mask = np.zeros(n_catalog, dtype=bool)
        for ep in visitor_engaged_pids[v_id]:
            if ep != t_pid:
                excl_v2_mask[pid_to_idx[ep]] = True

        # Full catalog discovery
        pop_disc = pop_scores.copy()
        pop_disc[excl_v2_mask] = -1e9
        hits_full_disc[ev_idx, 0] = 1.0 if (np.count_nonzero(pop_disc > pop_disc[t_idx]) + 1) <= 10 else 0.0

        dept_disc = dept_scores.copy()
        dept_disc[excl_v2_mask] = -1e9
        hits_full_disc[ev_idx, 1] = 1.0 if (np.count_nonzero(dept_disc > dept_disc[t_idx]) + 1) <= 10 else 0.0

        linucb_disc = linucb_scores.copy()
        linucb_disc[excl_v2_mask] = -1e9
        hits_full_disc[ev_idx, 2] = 1.0 if (np.count_nonzero(linucb_disc > linucb_disc[t_idx]) + 1) <= 10 else 0.0

        lints_disc = lints_scores.copy()
        lints_disc[excl_v2_mask] = -1e9
        hits_full_disc[ev_idx, 3] = 1.0 if (np.count_nonzero(lints_disc > lints_disc[t_idx]) + 1) <= 10 else 0.0

        # 50 candidates discovery
        excl_v2_50 = excl_v2_mask[pool_50]
        pop_disc_50 = pop_50.copy()
        pop_disc_50[excl_v2_50] = -1e9
        hits_50_disc[ev_idx, 0] = 1.0 if (np.count_nonzero(pop_disc_50 > pop_disc_50[0]) + 1) <= 10 else 0.0

        dept_disc_50 = dept_50.copy()
        dept_disc_50[excl_v2_50] = -1e9
        hits_50_disc[ev_idx, 1] = 1.0 if (np.count_nonzero(dept_disc_50 > dept_disc_50[0]) + 1) <= 10 else 0.0

        linucb_disc_50 = linucb_50.copy()
        linucb_disc_50[excl_v2_50] = -1e9
        hits_50_disc[ev_idx, 2] = 1.0 if (np.count_nonzero(linucb_disc_50 > linucb_disc_50[0]) + 1) <= 10 else 0.0

        lints_disc_50 = lints_50.copy()
        lints_disc_50[excl_v2_50] = -1e9
        hits_50_disc[ev_idx, 3] = 1.0 if (np.count_nonzero(lints_disc_50 > lints_disc_50[0]) + 1) <= 10 else 0.0

        # Update history
        visitor_engaged_pids[v_id].add(t_pid)
        curr_hist.append((catalog[t_pid], ev_type, ts, 2.0))
        if len(curr_hist) > 20:
            curr_hist.pop(0)

        if (ev_idx + 1) % 50000 == 0:
            print(f"  Processed {ev_idx+1:,} / {n_events:,} in {time.time()-t_stream_start:.1f}s...", flush=True)

    print(f"Streaming completed in {time.time()-t_stream_start:.1f}s.")

    test_vids = np.array(test_vids)
    _, v_codes = np.unique(test_vids, return_inverse=True)
    n_v = int(v_codes.max()) + 1
    v_counts = np.bincount(v_codes, minlength=n_v)

    # -------------------------------------------------------------------------
    # BOOTSTRAP FUNCTION
    # -------------------------------------------------------------------------
    def compute_stats_and_boot(hits_mat, v_codes, v_counts, n_v, n_boot=2000):
        n_pols = hits_mat.shape[1]
        V_means = []
        V_sums = []
        for p in range(n_pols):
            s_p = np.bincount(v_codes, weights=hits_mat[:, p], minlength=n_v)
            m_p = np.divide(s_p, v_counts, out=np.zeros(n_v), where=v_counts > 0)
            V_sums.append(s_p)
            V_means.append(m_p)

        boot_macro = np.empty((n_pols, n_boot), dtype=np.float64)
        boot_micro = np.empty((n_pols, n_boot), dtype=np.float64)

        for b in range(n_boot):
            idx = rng.integers(0, n_v, size=n_v)
            cnt_sum = np.sum(v_counts[idx])
            for p in range(n_pols):
                boot_macro[p, b] = np.mean(V_means[p][idx])
                boot_micro[p, b] = np.sum(V_sums[p][idx]) / max(1, cnt_sum)

        results = []
        for p in range(n_pols):
            mac_mean = float(np.mean(V_means[p]))
            mic_mean = float(np.mean(hits_mat[:, p]))
            mac_ci = (float(np.percentile(boot_macro[p], 2.5)), float(np.percentile(boot_macro[p], 97.5)))
            mic_ci = (float(np.percentile(boot_micro[p], 2.5)), float(np.percentile(boot_micro[p], 97.5)))
            results.append({
                "macro_mean": mac_mean,
                "macro_ci": mac_ci,
                "micro_mean": mic_mean,
                "micro_ci": mic_ci,
                "boot_macro": boot_macro[p],
                "boot_micro": boot_micro[p]
            })
        return results

    def compute_paired(boot_a, boot_b):
        diff = boot_a - boot_b
        diff_mean = float(np.mean(diff))
        diff_ci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
        p_val = float(np.mean(diff <= 0.0))
        p_val_str = f"< {1.0 / len(diff):.4f}" if p_val == 0.0 else f"{p_val:.4f}"
        return diff_mean, diff_ci, p_val_str

    # --- TABLE D1: OVERALL EVENTS (ALL EVENTS, UNMASKED) ---
    print("\n--- Computing 2,000 Bootstrap CIs for Table D1 (Overall Unconstrained) ---", flush=True)
    res_d1_full = compute_stats_and_boot(hits_full_uncon, v_codes, v_counts, n_v)
    res_d1_50 = compute_stats_and_boot(hits_50_uncon, v_codes, v_counts, n_v)

    # Paired comparisons for D1
    # 5: Hybrid-3, 6: Hybrid-5, 2: RV, 3: Warm LinUCB
    paired_d1_records = []
    for setting_name, r_list in [("full_catalog", res_d1_full), ("50_candidates", res_d1_50)]:
        # Hybrid 3 vs RV
        m_diff, m_ci, m_p = compute_paired(r_list[5]["boot_macro"], r_list[2]["boot_macro"])
        paired_d1_records.append({
            "setting": setting_name, "comparison": "Hybrid (k=3) vs RecentlyViewed",
            "macro_diff": m_diff, "macro_ci": m_ci, "p_value": m_p
        })
        # Hybrid 3 vs Warm LinUCB
        m_diff, m_ci, m_p = compute_paired(r_list[5]["boot_macro"], r_list[3]["boot_macro"])
        paired_d1_records.append({
            "setting": setting_name, "comparison": "Hybrid (k=3) vs Warm LinUCB",
            "macro_diff": m_diff, "macro_ci": m_ci, "p_value": m_p
        })
        # Hybrid 5 vs RV
        m_diff, m_ci, m_p = compute_paired(r_list[6]["boot_macro"], r_list[2]["boot_macro"])
        paired_d1_records.append({
            "setting": setting_name, "comparison": "Hybrid (k=5) vs RecentlyViewed",
            "macro_diff": m_diff, "macro_ci": m_ci, "p_value": m_p
        })
        # Hybrid 5 vs Warm LinUCB
        m_diff, m_ci, m_p = compute_paired(r_list[6]["boot_macro"], r_list[3]["boot_macro"])
        paired_d1_records.append({
            "setting": setting_name, "comparison": "Hybrid (k=5) vs Warm LinUCB",
            "macro_diff": m_diff, "macro_ci": m_ci, "p_value": m_p
        })

    # --- TABLE D2: NOVEL TARGETS (DISCOVERY EXCLUSION V2) ---
    print("\n--- Computing 2,000 Bootstrap CIs for Table D2 (Novel Discovery Exclusion v2) ---", flush=True)
    mask_nov = is_novel_mask
    sub_vids_nov = test_vids[mask_nov]
    _, v_codes_nov = np.unique(sub_vids_nov, return_inverse=True)
    n_v_nov = int(v_codes_nov.max()) + 1
    v_counts_nov = np.bincount(v_codes_nov, minlength=n_v_nov)

    res_d2_full = compute_stats_and_boot(hits_full_disc[mask_nov], v_codes_nov, v_counts_nov, n_v_nov)
    res_d2_50 = compute_stats_and_boot(hits_50_disc[mask_nov], v_codes_nov, v_counts_nov, n_v_nov)

    paired_d2_records = []
    # Compare vs Popularity (index 0)
    for setting_name, r_list in [("full_catalog", res_d2_full), ("50_candidates", res_d2_50)]:
        # Warm LinUCB (index 2) vs Pop (index 0)
        m_diff, m_ci, m_p = compute_paired(r_list[2]["boot_macro"], r_list[0]["boot_macro"])
        paired_d2_records.append({
            "setting": setting_name, "comparison": "Warm LinUCB vs Popularity Baseline",
            "macro_diff": m_diff, "macro_ci": m_ci, "p_value": m_p
        })
        # Warm LinTS (index 3) vs Pop (index 0)
        m_diff, m_ci, m_p = compute_paired(r_list[3]["boot_macro"], r_list[0]["boot_macro"])
        paired_d2_records.append({
            "setting": setting_name, "comparison": "Warm LinTS vs Popularity Baseline",
            "macro_diff": m_diff, "macro_ci": m_ci, "p_value": m_p
        })
        # Dept (index 1) vs Pop (index 0)
        m_diff, m_ci, m_p = compute_paired(r_list[1]["boot_macro"], r_list[0]["boot_macro"])
        paired_d2_records.append({
            "setting": setting_name, "comparison": "Department Heuristic vs Popularity Baseline",
            "macro_diff": m_diff, "macro_ci": m_ci, "p_value": m_p
        })

    # Assemble JSON output
    part_d_export = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_events_overall": int(n_events),
            "n_events_novel": int(np.count_nonzero(is_novel_mask)),
            "n_visitors_overall": int(n_v),
            "n_visitors_novel": int(n_v_nov)
        },
        "table_d1_overall_unconstrained": {
            "full_catalog": [
                {
                    "policy": pol,
                    "macro_hr@10": res_d1_full[i]["macro_mean"],
                    "macro_ci": res_d1_full[i]["macro_ci"],
                    "micro_hr@10": res_d1_full[i]["micro_mean"],
                    "micro_ci": res_d1_full[i]["micro_ci"]
                }
                for i, pol in enumerate(overall_pols)
            ],
            "50_candidates": [
                {
                    "policy": pol,
                    "macro_hr@10": res_d1_50[i]["macro_mean"],
                    "macro_ci": res_d1_50[i]["macro_ci"],
                    "micro_hr@10": res_d1_50[i]["micro_mean"],
                    "micro_ci": res_d1_50[i]["micro_ci"]
                }
                for i, pol in enumerate(overall_pols)
            ],
            "paired_tests": paired_d1_records
        },
        "table_d2_pure_discovery": {
            "full_catalog": [
                {
                    "policy": pol,
                    "macro_hr@10": res_d2_full[i]["macro_mean"],
                    "macro_ci": res_d2_full[i]["macro_ci"],
                    "micro_hr@10": res_d2_full[i]["micro_mean"],
                    "micro_ci": res_d2_full[i]["micro_ci"]
                }
                for i, pol in enumerate(disc_pols)
            ],
            "50_candidates": [
                {
                    "policy": pol,
                    "macro_hr@10": res_d2_50[i]["macro_mean"],
                    "macro_ci": res_d2_50[i]["macro_ci"],
                    "micro_hr@10": res_d2_50[i]["micro_mean"],
                    "micro_ci": res_d2_50[i]["micro_ci"]
                }
                for i, pol in enumerate(disc_pols)
            ],
            "paired_tests": paired_d2_records
        }
    }

    out_path = RESULTS_DIR / "phase10c_part_d_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(part_d_export, f, indent=2)

    print(f"\n[DONE] Part D results saved to {out_path} in {time.time()-t0:.1f}s.")
    return part_d_export


if __name__ == "__main__":
    run_part_d()
