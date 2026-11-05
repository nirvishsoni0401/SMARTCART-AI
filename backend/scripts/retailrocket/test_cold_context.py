import sys
import sqlite3
import numpy as np
from pathlib import Path
from collections import defaultdict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    UserContext,
    extract_feature_vector
)
from app.recommender.linucb import LinUCBRecommender

HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
DATA_DIR = Path("data/historical")
T_VAL_END = 1440756710277

def test_cold_context_behavior():
    print("=== INVESTIGATING COLD CONTEXT CONTRADICTION ===")
    
    # 1. Load catalog
    conn_sc = sqlite3.connect(SMARTCART_DB)
    c_sc = conn_sc.cursor()
    rows = c_sc.execute("SELECT id, name, category, brand, price, rating, discount_pct, popularity_score, popularity_prior FROM products WHERE is_active=1 ORDER BY id;").fetchall()
    conn_sc.close()
    
    catalog = {}
    dept_pids = defaultdict(list)
    idx_to_pid = []
    pid_to_idx = {}
    for idx, r in enumerate(rows):
        pid, name, cat, brand, price, rating, discount, pop, pop_prior = r
        prod = {
            "id": pid, "name": name, "category": cat, "brand": brand or "",
            "price": float(price), "rating": float(rating), "discount_pct": float(discount),
            "popularity_score": float(pop), "popularity_prior": float(pop_prior or 0.0)
        }
        catalog[pid] = prod
        dept_pids[cat].append(pid)
        idx_to_pid.append(pid)
        pid_to_idx[pid] = idx

    pop_priors = np.array([catalog[p]["popularity_prior"] for p in idx_to_pid], dtype=np.float64)
    p_weights = np.maximum(1.0, pop_priors) ** 0.75
    p_sampling = p_weights / np.sum(p_weights)

    # 2. Warm LinUCB
    stats_m3 = np.load(DATA_DIR / "sufficient_stats_m3.npz", allow_pickle=True)
    warm_linucb = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0)
    warm_linucb.warm_start_from_stats(stats_m3, n_cap=10000, lambda_reg=1.0)
    th_warm = warm_linucb.get_theta()
    A_inv_warm = warm_linucb.A_inv

    # 3. Cold LinUCB
    cold_linucb = LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=0.5, lambda_reg=1.0)
    th_cold = cold_linucb.get_theta()  # zeros
    A_inv_cold = cold_linucb.A_inv     # identity I

    # Cold context feature matrix
    cold_ctx = UserContext()
    X_cold = np.zeros((len(idx_to_pid), FEATURE_DIMENSION), dtype=np.float64)
    for i, pid in enumerate(idx_to_pid):
        X_cold[i] = extract_feature_vector(cold_ctx, catalog[pid])

    # Examine score vectors
    # Cold LinUCB scores: 0.0 + 0.5 * sqrt(diag(X_cold @ I @ X_cold^T)) = 0.5 * ||X_cold[i]||_2
    var_cold = np.sum(X_cold * X_cold, axis=1)
    scores_cold = 0.5 * np.sqrt(var_cold)

    # Warm LinUCB scores: X_cold @ th_warm + 0.5 * sqrt(diag(X_cold @ A_inv_warm @ X_cold^T))
    exp_warm = X_cold @ th_warm
    var_warm = np.sum((X_cold @ A_inv_warm) * X_cold, axis=1)
    scores_warm = exp_warm + 0.5 * np.sqrt(np.maximum(0.0, var_warm))

    print(f"Scores on cold context across all 262 SKUs:")
    print(f"  Cold LinUCB scores (norm of cold feature vector): min={scores_cold.min():.4f}, max={scores_cold.max():.4f}, mean={scores_cold.mean():.4f}")
    print(f"  Warm LinUCB scores: min={scores_warm.min():.4f}, max={scores_warm.max():.4f}, mean={scores_warm.mean():.4f}")

    # Check which items get highest score under Cold LinUCB
    top_cold_indices = np.argsort(scores_cold)[::-1][:10]
    print("\nTop 10 SKUs under Cold LinUCB (pure feature norm ||x||):")
    for i in top_cold_indices:
        p = catalog[idx_to_pid[i]]
        print(f"  pid={p['id']:3d} ({p['name'][:30]:30s}) cat={p['category']:15s} score={scores_cold[i]:.4f} pop={p['popularity_prior']:.1f}")

    top_warm_indices = np.argsort(scores_warm)[::-1][:10]
    print("\nTop 10 SKUs under Warm LinUCB on Cold Context:")
    for i in top_warm_indices:
        p = catalog[idx_to_pid[i]]
        print(f"  pid={p['id']:3d} ({p['name'][:30]:30s}) cat={p['category']:15s} score={scores_warm[i]:.4f} pop={p['popularity_prior']:.1f}")

if __name__ == "__main__":
    test_cold_context_behavior()

