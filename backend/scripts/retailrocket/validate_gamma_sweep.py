import sys
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import time

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    UserContext,
    build_context_from_history,
    extract_feature_vector
)
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.linucb import LinUCBRecommender

HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
DATA_DIR = Path("data/historical")
T_TRAIN_END = 1438968232766
T_VAL_END = 1440756710277

REWARD_MAP = {
    "view": 2.0,
    "addtocart": 5.0,
    "transaction": 10.0
}

def main():
    print("=== SWEEPING GAMMA FOR DISCOUNTED LINUCB ON VALIDATION DATA ===")
    
    # Load catalog
    conn_sc = sqlite3.connect(SMARTCART_DB)
    c_sc = conn_sc.cursor()
    rows = c_sc.execute("SELECT id, name, category, brand, price, rating, discount_pct, popularity_score, popularity_prior FROM products WHERE is_active=1 ORDER BY id;").fetchall()
    conn_sc.close()
    
    catalog = {}
    idx_to_pid = []
    pid_to_idx = {}
    for idx, r in enumerate(rows):
        pid, name, cat, brand, price, rating, discount, pop, pop_prior = r
        catalog[pid] = {
            "id": pid, "name": name, "category": cat, "brand": brand or "",
            "price": float(price), "rating": float(rating), "discount_pct": float(discount),
            "popularity_score": float(pop), "popularity_prior": float(pop_prior or 0.0)
        }
        idx_to_pid.append(pid)
        pid_to_idx[pid] = idx

    n_catalog = len(catalog)

    # Load mapping
    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_hist = conn_hist.cursor()
    mapping = dict(c_hist.execute("""
        SELECT item_id, product_id FROM rr_item_map 
        WHERE mapping_method='popularity_balanced_train' AND mapping_version='v1.0_train'
    """).fetchall())

    # Load 20,000 chronological validation events
    val_events = c_hist.execute("""
        SELECT visitor_id, item_id, event_type, ts 
        FROM rr_events 
        WHERE ts >= ? AND ts < ?
        ORDER BY ts
        LIMIT 25000;
    """, (T_TRAIN_END, T_VAL_END)).fetchall()
    conn_hist.close()

    valid_val = [(v, mapping[i], ev, ts) for v, i, ev, ts in val_events if mapping.get(i) in catalog][:20000]
    print(f"Loaded {len(valid_val):,} chronological validation events.")

    stats_m3 = np.load(DATA_DIR / "sufficient_stats_m3.npz", allow_pickle=True)

    gamma_candidates = [0.99, 0.995, 0.999, 0.9995, 1.0]
    results = {}

    for gamma in gamma_candidates:
        print(f"\nEvaluating gamma = {gamma}...", flush=True)
        if gamma == 1.0:
            model = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0)
        else:
            model = DiscountedLinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0, gamma=gamma)
        model.warm_start_from_stats(stats_m3, n_cap=10000, lambda_reg=1.0)

        hits = []
        visitor_histories = defaultdict(list)
        t_start = time.time()

        for step, (v_id, t_pid, ev_type, ts) in enumerate(valid_val):
            hist = visitor_histories[v_id]
            ctx = build_context_from_history(hist, reference_time=None)
            
            # Predict
            X_u = np.zeros((n_catalog, FEATURE_DIMENSION), dtype=np.float64)
            for i, p in enumerate(idx_to_pid):
                X_u[i] = extract_feature_vector(ctx, catalog[p])
            
            scores = X_u @ model.theta + model.alpha * np.sqrt(np.maximum(0.0, np.sum((X_u @ model.A_inv) * X_u, axis=1)))
            t_idx = pid_to_idx[t_pid]
            rank = np.count_nonzero(scores > scores[t_idx]) + 1
            hit = 1 if rank <= 10 else 0
            hits.append(hit)

            # Update online
            x_target = X_u[t_idx]
            reward = REWARD_MAP.get(ev_type, 2.0)
            model.update(x_target, reward)

            # Update history
            hist.append((catalog[t_pid], ev_type, ts, reward))
            if len(hist) > 20:
                hist.pop(0)

        overall_hr10 = float(np.mean(hits))
        last5k_hr10 = float(np.mean(hits[-5000:]))
        results[gamma] = {
            "overall_hr10": overall_hr10,
            "last5k_hr10": last5k_hr10,
            "elapsed": time.time() - t_start
        }
        print(f"  Gamma={gamma:6.4f}: Overall HR@10 = {overall_hr10*100:.2f}%, Last 5k HR@10 = {last5k_hr10*100:.2f}% (took {time.time()-t_start:.1f}s)")

    print("\n=== SUMMARY OF GAMMA SWEEP ===")
    for g, res in results.items():
        print(f"Gamma={g:6.4f}: Overall HR@10 = {res['overall_hr10']*100:.2f}%, Final 5k HR@10 = {res['last5k_hr10']*100:.2f}%")

if __name__ == "__main__":
    main()
