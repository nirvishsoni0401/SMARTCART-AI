"""
SmartCart AI — Build Aggregated Training Sufficient Statistics
Creates sufficient statistics (A_hist, b_hist, N_hist) using the aggregated-example spec:
One training example per (visitor_id, session_id, mapped_product_id) with:
  1. Context constructed strictly before the FIRST event on that product in that session.
  2. Highest-stage reward achieved on that product in that session:
     transaction (10.0) > addtocart (5.0) > view (2.0).
  3. Frozen m=3 pseudo-negatives with reward 0.
Saves to data/historical/sufficient_stats_m3_aggregated.npz.
"""

import sys
import os
import time
import sqlite3
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

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

HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
OUTPUT_DIR = Path("data/historical")
TRAIN_CUTOFF_MS = 1438968232766

REWARD_HIERARCHY = {
    "transaction": 10.0,
    "addtocart": 5.0,
    "view": 2.0
}


def load_catalog(smartcart_db_path: Path):
    conn = sqlite3.connect(smartcart_db_path)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, name, category, brand, price, rating, discount_pct, popularity_score, popularity_prior
        FROM products WHERE is_active = 1 ORDER BY id;
    """).fetchall()
    conn.close()

    catalog = {}
    for r in rows:
        pid, name, cat, brand, price, rating, discount, pop, pop_prior = r
        catalog[pid] = {
            "id": pid, "name": name, "category": cat, "brand": brand or "",
            "price": float(price), "rating": float(rating), "discount_pct": float(discount),
            "popularity_score": float(pop), "popularity_prior": float(pop_prior or 0.0)
        }
    return catalog


def load_mapping(historical_db_path: Path, method: str = "popularity_balanced_train", version: str = "v1.0_train"):
    conn = sqlite3.connect(historical_db_path)
    c = conn.cursor()
    rows = c.execute("""
        SELECT item_id, product_id
        FROM rr_item_map
        WHERE mapping_method = ? AND mapping_version = ?;
    """, (method, version)).fetchall()
    conn.close()
    return dict(rows)


def build_aggregated_statistics():
    start_time = time.time()
    print("=== BUILDING AGGREGATED SUFFICIENT STATISTICS (m=3) ===")
    catalog = load_catalog(SMARTCART_DB)
    item_map = load_mapping(HISTORICAL_DB)
    catalog_pids = sorted(list(catalog.keys()))
    n_catalog = len(catalog_pids)

    # Precompute empty context vectors
    empty_ctx = build_context_from_history([])
    cold_vecs = {pid: extract_feature_vector(empty_ctx, catalog[pid]) for pid in catalog_pids}

    # Popularity prior for negative sampling
    pop_priors = np.array([catalog[pid]["popularity_prior"] for pid in catalog_pids], dtype=np.float64)
    if pop_priors.sum() > 0:
        sampling_probs = pop_priors / pop_priors.sum()
    else:
        sampling_probs = np.ones(n_catalog) / n_catalog

    conn = sqlite3.connect(HISTORICAL_DB)
    c = conn.cursor()
    # Stream train events chronologically
    cursor = c.execute("""
        SELECT visitor_id, session_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts < ?
        ORDER BY visitor_id, ts ASC;
    """, (TRAIN_CUTOFF_MS,))

    # We process per visitor
    # In each visitor stream:
    # Track visitor history (last 20 events)
    # Group events by session_id
    A_mat = np.zeros((FEATURE_DIMENSION, FEATURE_DIMENSION), dtype=np.float64)
    b_vec = np.zeros(FEATURE_DIMENSION, dtype=np.float64)

    buf_x_pos = []
    buf_r_pos = []
    buf_x_neg = []

    batch_size = 50000
    n_positives = 0

    rng = np.random.default_rng(42)
    pool_size = 100000
    pool_indices = rng.choice(n_catalog, size=pool_size, p=sampling_probs)
    pool_ptr = 0

    def flush():
        nonlocal A_mat, b_vec, buf_x_pos, buf_r_pos, buf_x_neg
        if not buf_x_pos:
            return
        X_p = np.array(buf_x_pos, dtype=np.float64)
        R_p = np.array(buf_r_pos, dtype=np.float64)
        X_n = np.array(buf_x_neg, dtype=np.float64)

        A_mat += X_p.T @ X_p + X_n.T @ X_n
        b_vec += X_p.T @ R_p

        buf_x_pos.clear()
        buf_r_pos.clear()
        buf_x_neg.clear()

    curr_visitor_id = None
    visitor_events = []

    def process_visitor(v_id, events):
        nonlocal pool_ptr, pool_indices, n_positives
        if not events:
            return

        # Sort strictly by ts
        events.sort(key=lambda x: x[3])  # (session_id, pid, event_type, ts)

        # To implement: one example per (session_id, pid) with context strictly before FIRST event on pid in that session
        # First pass: find the highest reward for each (session_id, pid)
        max_rewards = defaultdict(float)
        first_event_idx = {}
        for idx, (sess_id, pid, ev_type, ts) in enumerate(events):
            r = REWARD_HIERARCHY.get(ev_type, 2.0)
            key = (sess_id, pid)
            if r > max_rewards[key]:
                max_rewards[key] = r
            if key not in first_event_idx:
                first_event_idx[key] = idx

        # Second pass: simulate visitor history chronologically
        v_hist = []
        for idx, (sess_id, pid, ev_type, ts) in enumerate(events):
            prod = catalog[pid]
            r = REWARD_HIERARCHY.get(ev_type, 2.0)
            key = (sess_id, pid)

            # If this is the FIRST event on this product in this session, construct example!
            if first_event_idx[key] == idx:
                # Build context strictly from v_hist prior to this event
                if not v_hist:
                    x_pos = cold_vecs[pid]
                    ctx = None
                else:
                    ctx = build_context_from_history(
                        history=v_hist,
                        reference_time=datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
                    )
                    x_pos = extract_feature_vector(ctx, prod)

                # Highest-stage reward for this (session, pid)
                highest_r = max_rewards[key]

                # Sample m=3 pseudo-negatives
                if pool_ptr + 3 >= pool_size:
                    pool_indices = rng.choice(n_catalog, size=pool_size, p=sampling_probs)
                    pool_ptr = 0
                neg_pids = [catalog_pids[pool_indices[pool_ptr + k]] for k in range(3)]
                pool_ptr += 3

                if not v_hist:
                    x_neg = [cold_vecs[np_id] for np_id in neg_pids]
                else:
                    x_neg = [extract_feature_vector(ctx, catalog[np_id]) for np_id in neg_pids]

                buf_x_pos.append(x_pos)
                buf_r_pos.append(highest_r)
                buf_x_neg.extend(x_neg)
                n_positives += 1

                if len(buf_x_pos) >= batch_size:
                    flush()

            # Append to visitor history (sliding window 20)
            v_hist.append((prod, ev_type, ts, r))
            if len(v_hist) > 20:
                v_hist.pop(0)

    total_rows = 0
    for v_id, s_id, item_id, ev_type, ts in cursor:
        total_rows += 1
        pid = item_map.get(item_id)
        if pid not in catalog:
            continue

        if v_id != curr_visitor_id:
            if curr_visitor_id is not None:
                process_visitor(curr_visitor_id, visitor_events)
            curr_visitor_id = v_id
            visitor_events = []

        visitor_events.append((s_id, pid, ev_type, ts))

        if total_rows % 300000 == 0:
            print(f"  Streamed {total_rows:,} raw events | Formed {n_positives:,} aggregated examples ({time.time()-start_time:.1f}s)...", flush=True)

    if curr_visitor_id is not None:
        process_visitor(curr_visitor_id, visitor_events)

    flush()
    conn.close()

    total_examples = n_positives * 4  # 1 positive + 3 negatives
    out_file = OUTPUT_DIR / "sufficient_stats_m3_aggregated.npz"
    np.savez_compressed(
        out_file,
        A_hist=A_mat,
        b_hist=b_vec,
        n_hist=total_examples,
        n_positives=n_positives,
        n_negatives=n_positives * 3,
        pseudo_negatives_m=3,
        feature_spec_hash=FEATURE_SPEC_HASH,
        feature_dim=FEATURE_DIMENSION,
        train_cutoff_ts=TRAIN_CUTOFF_MS,
        mapping_method="popularity_balanced_train",
        mapping_version="v1.0_train",
        example_construction="first_exposure_aggregated"
    )

    elapsed = time.time() - start_time
    print(f"\n[DONE] Built aggregated sufficient stats: {n_positives:,} positives, {total_examples:,} total in {elapsed:.1f}s!")
    print(f"  Saved to: {out_file}")
    return out_file


if __name__ == "__main__":
    build_aggregated_statistics()

