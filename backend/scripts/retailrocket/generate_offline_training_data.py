"""
SmartCart AI — Retailrocket Offline Training Data & Sufficient Statistics Generator
Extracts streaming interactions from rr_events strictly within the training period (t < 1438968232766).
Uses shared feature engineering (build_context_from_history and extract_feature_vector).
Computes sufficient statistics (A_hist, b_hist, N_hist) for pseudo-negative counts m in {0, 1, 3, 5}.
Saves compact .npz files in data/historical/.
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
import sqlite3
import numpy as np

# Ensure backend directory is in sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    UserContext,
    build_context_from_history,
    extract_feature_vector,
    CATEGORIES
)
from app.config import settings

HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
OUTPUT_DIR = Path("data/historical")
TRAIN_CUTOFF_MS = 1438968232766

REWARD_MAP = {
    "view": 2.0,
    "click": 2.0,
    "addtocart": 5.0,
    "cart_add": 5.0,
    "transaction": 10.0,
    "purchase": 10.0
}


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()[:10]
    except Exception:
        return "pre-phase9b"


def load_catalog_and_priors(smartcart_db_path: Path):
    """Load the 262 active catalog products."""
    conn = sqlite3.connect(smartcart_db_path)
    c = conn.cursor()
    rows = c.execute("""
        SELECT id, name, category, brand, price, rating, discount_pct, popularity_score
        FROM products
        WHERE is_active = 1
        ORDER BY id;
    """).fetchall()
    conn.close()

    prods = {}
    for r in rows:
        pid, name, cat, brand, price, rating, discount, pop = r
        prods[pid] = {
            "id": pid,
            "name": name,
            "category": cat,
            "brand": brand or "",
            "price": float(price),
            "rating": float(rating),
            "discount_pct": float(discount),
            "popularity_score": float(pop)
        }
    return prods


def precompute_cold_vectors(prods: dict) -> dict:
    """Precompute the 36-dim feature vectors for all catalog products under cold user context."""
    cold_ctx = UserContext()
    cold_vecs = {}
    for pid, p in prods.items():
        cold_vecs[pid] = extract_feature_vector(cold_ctx, p)
    return cold_vecs


def run_generate_training_data(
    historical_db_path: Path = HISTORICAL_DB,
    smartcart_db_path: Path = SMARTCART_DB,
    mapping_method: str = "popularity_balanced_train",
    mapping_version: str = "v1.0_train",
    train_cutoff: int = TRAIN_CUTOFF_MS,
    random_seed: int = 42,
    batch_size: int = 10000,
    output_prefix: str = ""
):
    print("=" * 70, flush=True)
    print("SMARTCART AI: GENERATING OFFLINE TRAINING DATA & SUFFICIENT STATISTICS", flush=True)
    print(f"Mapping Method: {mapping_method} ({mapping_version}) | Cutoff: {train_cutoff}", flush=True)
    print(f"Feature Spec Hash: {FEATURE_SPEC_HASH} (dim={FEATURE_DIMENSION})", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    np.random.seed(random_seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load active catalog
    print("\n[1/4] Loading active SmartCart catalog...", flush=True)
    catalog = load_catalog_and_priors(smartcart_db_path)
    catalog_pids = sorted(catalog.keys())
    print(f"  Loaded {len(catalog)} active products", flush=True)

    # 2. Precompute cold feature vectors
    print("[2/4] Precomputing cold feature vectors...", flush=True)
    cold_vecs = precompute_cold_vectors(catalog)

    # 3. Load item-to-product mapping
    print("\n[3/4] Loading item mapping from historical.db...", flush=True)
    conn_hist = sqlite3.connect(historical_db_path)
    c_hist = conn_hist.cursor()

    item_map = dict(c_hist.execute("""
        SELECT item_id, product_id
        FROM rr_item_map
        WHERE mapping_method = ? AND mapping_version = ?;
    """, (mapping_method, mapping_version)).fetchall())
    print(f"  Loaded {len(item_map):,} mapped items", flush=True)

    # 4. Compute catalog product popularity prior from training period for negative sampling
    print("  Calculating product sampling distribution for pseudo-negatives...", flush=True)
    prod_counts = Counter()
    ev_counts_raw = c_hist.execute("""
        SELECT item_id, COUNT(*)
        FROM rr_events
        WHERE ts < ?
        GROUP BY item_id;
    """, (train_cutoff,)).fetchall()

    for iid, cnt in ev_counts_raw:
        pid = item_map.get(iid)
        if pid in catalog:
            prod_counts[pid] += cnt

    # Unigram negative sampling distribution p_i \propto (popularity)^0.75
    weights = np.array([max(1.0, float(prod_counts.get(pid, 1))) ** 0.75 for pid in catalog_pids], dtype=np.float64)
    sampling_probs = weights / np.sum(weights)

    # Pre-sample pseudo-negatives in large blocks for high performance
    print("\n[4/4] Streaming training events and accumulating sufficient statistics...", flush=True)
    total_train_events = c_hist.execute("SELECT COUNT(*) FROM rr_events WHERE ts < ?;", (train_cutoff,)).fetchone()[0]
    print(f"  Total events to process: {total_train_events:,}", flush=True)

    # Sufficient statistics accumulators
    d = FEATURE_DIMENSION
    A_m0 = np.zeros((d, d), dtype=np.float64)
    A_m1 = np.zeros((d, d), dtype=np.float64)
    A_m3 = np.zeros((d, d), dtype=np.float64)
    A_m5 = np.zeros((d, d), dtype=np.float64)
    b_hist = np.zeros(d, dtype=np.float64)

    n_positives = 0
    n_negatives = 0

    # Stream query ordered by visitor_id and timestamp
    query = """
        SELECT visitor_id, item_id, event_type, ts
        FROM rr_events
        WHERE ts < ?
        ORDER BY visitor_id, ts;
    """
    cursor = conn_hist.cursor()
    cursor.execute(query, (train_cutoff,))

    # Batch buffers for BLAS operations
    buf_x_pos = []
    buf_r_pos = []
    buf_x_neg1 = []
    buf_x_neg2 = []
    buf_x_neg3 = []
    buf_x_neg4 = []
    buf_x_neg5 = []

    curr_visitor_id = None
    visitor_history = []

    def flush_batch():
        nonlocal A_m0, A_m1, A_m3, A_m5, b_hist
        if not buf_x_pos:
            return
        X_pos = np.array(buf_x_pos, dtype=np.float64)
        r_pos = np.array(buf_r_pos, dtype=np.float64)
        X_n1 = np.array(buf_x_neg1, dtype=np.float64)
        X_n2 = np.array(buf_x_neg2, dtype=np.float64)
        X_n3 = np.array(buf_x_neg3, dtype=np.float64)
        X_n4 = np.array(buf_x_neg4, dtype=np.float64)
        X_n5 = np.array(buf_x_neg5, dtype=np.float64)

        # Outer product accumulation via GEMM
        A_pos = X_pos.T @ X_pos
        b_hist += X_pos.T @ r_pos

        A_m0 += A_pos
        A_m1 += A_pos + (X_n1.T @ X_n1)
        A_m3 += A_pos + (X_n1.T @ X_n1) + (X_n2.T @ X_n2) + (X_n3.T @ X_n3)
        A_m5 += A_pos + (X_n1.T @ X_n1) + (X_n2.T @ X_n2) + (X_n3.T @ X_n3) + (X_n4.T @ X_n4) + (X_n5.T @ X_n5)

        buf_x_pos.clear()
        buf_r_pos.clear()
        buf_x_neg1.clear()
        buf_x_neg2.clear()
        buf_x_neg3.clear()
        buf_x_neg4.clear()
        buf_x_neg5.clear()

    # Pre-generate negative samples pool
    pool_size = 500000
    pool_indices = np.random.choice(len(catalog_pids), size=pool_size, p=sampling_probs)
    pool_ptr = 0

    row_count = 0
    t_last_log = time.time()

    for visitor_id, item_id, event_type, ts in cursor:
        row_count += 1
        if visitor_id != curr_visitor_id:
            curr_visitor_id = visitor_id
            visitor_history = []

        pid = item_map.get(item_id)
        if pid not in catalog:
            continue

        prod = catalog[pid]
        reward = REWARD_MAP.get(event_type, 2.0)

        # Build context from strictly prior events
        is_cold = (len(visitor_history) == 0)
        if is_cold:
            x_pos = cold_vecs[pid]
            ctx = None
        else:
            ctx = build_context_from_history(
                history=visitor_history,
                reference_time=datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
            )
            x_pos = extract_feature_vector(ctx, prod)

        # Append current event to visitor history for future interactions
        visitor_history.append((prod, event_type, ts, reward))
        if len(visitor_history) > 20:
            visitor_history.pop(0)

        # Sample 5 pseudo-negatives
        if pool_ptr + 5 >= pool_size:
            pool_indices = np.random.choice(len(catalog_pids), size=pool_size, p=sampling_probs)
            pool_ptr = 0
        neg_pids = [catalog_pids[pool_indices[pool_ptr + i]] for i in range(5)]
        pool_ptr += 5

        if is_cold:
            x_n1 = cold_vecs[neg_pids[0]]
            x_n2 = cold_vecs[neg_pids[1]]
            x_n3 = cold_vecs[neg_pids[2]]
            x_n4 = cold_vecs[neg_pids[3]]
            x_n5 = cold_vecs[neg_pids[4]]
        else:
            x_n1 = extract_feature_vector(ctx, catalog[neg_pids[0]])
            x_n2 = extract_feature_vector(ctx, catalog[neg_pids[1]])
            x_n3 = extract_feature_vector(ctx, catalog[neg_pids[2]])
            x_n4 = extract_feature_vector(ctx, catalog[neg_pids[3]])
            x_n5 = extract_feature_vector(ctx, catalog[neg_pids[4]])

        buf_x_pos.append(x_pos)
        buf_r_pos.append(reward)
        buf_x_neg1.append(x_n1)
        buf_x_neg2.append(x_n2)
        buf_x_neg3.append(x_n3)
        buf_x_neg4.append(x_n4)
        buf_x_neg5.append(x_n5)

        n_positives += 1
        n_negatives += 5

        if len(buf_x_pos) >= batch_size:
            flush_batch()

        if row_count % 300000 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {row_count:,} / {total_train_events:,} events ({row_count / total_train_events * 100:.1f}%) in {elapsed:.1f}s", flush=True)

    flush_batch()
    conn_hist.close()

    commit_hash = get_git_commit()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Save sufficient statistics for m in {0, 1, 3, 5}
    m_configs = [
        (0, A_m0, n_positives),
        (1, A_m1, n_positives * 2),
        (3, A_m3, n_positives * 4),
        (5, A_m5, n_positives * 6),
    ]

    prefix = f"{output_prefix}_" if output_prefix else ""

    for m, A_mat, total_n in m_configs:
        fname = OUTPUT_DIR / f"{prefix}sufficient_stats_m{m}.npz"
        np.savez_compressed(
            fname,
            A_hist=A_mat,
            b_hist=b_hist,
            n_hist=total_n,
            n_positives=n_positives,
            n_negatives=total_n - n_positives,
            pseudo_negatives_m=m,
            feature_spec_hash=FEATURE_SPEC_HASH,
            feature_dim=FEATURE_DIMENSION,
            train_cutoff_ts=train_cutoff,
            mapping_method=mapping_method,
            mapping_version=mapping_version,
            git_commit=commit_hash,
            created_at=now_iso
        )
        print(f"  Saved {fname} (N={total_n:,}, m={m})", flush=True)

    print(f"\n[DONE] Sufficient statistics generation completed in {time.time() - start_time:.2f}s!", flush=True)
    return {
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "m_files": [str(OUTPUT_DIR / f"{prefix}sufficient_stats_m{m}.npz") for m in [0, 1, 3, 5]]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sufficient statistics from Retailrocket training period")
    parser.add_argument("--mapping-method", default="popularity_balanced_train")
    parser.add_argument("--mapping-version", default="v1.0_train")
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()

    run_generate_training_data(
        mapping_method=args.mapping_method,
        mapping_version=args.mapping_version,
        output_prefix=args.output_prefix
    )

