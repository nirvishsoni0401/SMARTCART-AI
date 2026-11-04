"""
SmartCart AI — Retailrocket LinUCB Warm-Start Model Trainer
Loads precomputed sufficient statistics (A_hist, b_hist, N_hist) from data/historical/.
Initializes LinUCB, Discounted LinUCB, LinTS, and Epsilon-Greedy policies with:
  A_init = lambda * I + s * A_hist
  b_init = s * b_hist
where s = min(1.0, N_cap / N_hist).

Validates SPD via Cholesky decomposition, computes condition number, and saves
versioned model snapshots into smartcart.db table model_states under warm_start:v1:<algo>.
CRITICAL: Never touches or overwrites live operational model states.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_engineering import FEATURE_DIMENSION, FEATURE_SPEC_HASH
from app.recommender.linucb import LinUCBRecommender
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.linucb_thompson import LinThompsonSamplingRecommender
from app.recommender.epsilon_greedy import EpsilonGreedyRecommender

SMARTCART_DB = BACKEND_DIR / "smartcart.db"
HISTORICAL_DATA_DIR = Path("data/historical")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()[:10]
    except Exception:
        return "pre-phase9b"


def run_warm_start(
    stats_path: Path = None,
    m: int = 3,
    n_cap: int = 10000,
    lambda_reg: float = 1.0,
    alpha: float = 0.5,
    gamma: float = 0.95,
    v_variance: float = 0.25,
    epsilon: float = 0.1,
    smartcart_db_path: Path = SMARTCART_DB,
    version_tag: str = "v1"
):
    print("=" * 70, flush=True)
    print("SMARTCART AI: BANDIT POLICY WARM-START INITIALIZATION", flush=True)
    print("=" * 70, flush=True)

    if stats_path is None:
        stats_path = HISTORICAL_DATA_DIR / f"sufficient_stats_m{m}.npz"

    if not stats_path.exists():
        raise FileNotFoundError(f"Sufficient statistics file not found: {stats_path}")

    print(f"Loading sufficient statistics from {stats_path}...", flush=True)
    stats_data = np.load(stats_path, allow_pickle=True)

    # 1. Feature spec hash validation
    file_spec_hash = str(stats_data.get("feature_spec_hash", ""))
    if file_spec_hash != FEATURE_SPEC_HASH:
        raise ValueError(
            f"Feature spec hash mismatch! File has '{file_spec_hash}', expected '{FEATURE_SPEC_HASH}'."
        )
    print(f"  Feature Spec Hash Verified: {FEATURE_SPEC_HASH} (dim={FEATURE_DIMENSION})", flush=True)

    n_hist = int(stats_data["n_hist"])
    print(f"  Total Historical Examples (N_hist): {n_hist:,} (m={m} pseudo-negatives)", flush=True)

    # 2. Build model instances
    models = {
        "linucb": LinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=alpha, lambda_reg=lambda_reg),
        "discounted_linucb": DiscountedLinUCBRecommender(feature_dim=FEATURE_DIMENSION, alpha=alpha, lambda_reg=lambda_reg, gamma=gamma),
        "linucb_thompson": LinThompsonSamplingRecommender(feature_dim=FEATURE_DIMENSION, v_variance=v_variance, lambda_reg=lambda_reg),
        "epsilon_greedy": EpsilonGreedyRecommender(feature_dim=FEATURE_DIMENSION, epsilon=epsilon, lambda_reg=lambda_reg)
    }

    warm_start_results = {}
    commit_hash = get_git_commit()
    now_iso = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(smartcart_db_path)
    c = conn.cursor()

    print(f"\nApplying warm-start initialization (N_cap={n_cap if n_cap is not None else 'None (all)'}, lambda={lambda_reg})...", flush=True)

    for algo_name, model in models.items():
        # Execute warm-start
        diag = model.warm_start_from_stats(stats_data, n_cap=n_cap, lambda_reg=lambda_reg)
        scale_s = diag["scale_s"]
        cond = diag["condition_number"]

        # Validate theta norm
        theta = model.get_theta()
        theta_norm = float(np.linalg.norm(theta))

        print(f"  [{algo_name}]")
        print(f"    Scaling s:          {scale_s:.6f}")
        print(f"    Condition Number:   {cond:.2f}")
        print(f"    Updates Recorded:   {model.n_updates:,}")
        print(f"    ||theta||_2:        {theta_norm:.4f}")

        # Construct versioned state payload
        state_dict = model.get_state()
        snapshot_metadata = {
            "algorithm": algo_name,
            "warm_start": True,
            "version": version_tag,
            "feature_spec_hash": FEATURE_SPEC_HASH,
            "feature_dim": FEATURE_DIMENSION,
            "n_hist": n_hist,
            "n_cap": n_cap,
            "scale_s": scale_s,
            "lambda_reg": lambda_reg,
            "alpha": alpha,
            "pseudo_negatives_m": m,
            "condition_number": cond,
            "theta_norm": theta_norm,
            "stats_file": str(stats_path),
            "git_commit": commit_hash,
            "created_at": now_iso
        }
        
        full_payload = {
            "metadata": snapshot_metadata,
            "model_state": state_dict
        }

        # Crucial safeguard: key name prefixed with warm_start:
        snapshot_key = f"warm_start:{version_tag}:{algo_name}"
        state_blob_str = json.dumps(full_payload)

        # Store into model_states table
        c.execute("""
            INSERT OR REPLACE INTO model_states (algorithm, state_blob, n_updates, updated_at)
            VALUES (?, ?, ?, ?);
        """, (snapshot_key, state_blob_str, model.n_updates, now_iso))

        warm_start_results[algo_name] = {
            "snapshot_key": snapshot_key,
            "condition_number": cond,
            "scale_s": scale_s,
            "n_updates": model.n_updates,
            "theta_norm": theta_norm
        }

    conn.commit()
    conn.close()

    print(f"\n[DONE] Successfully created and saved {len(models)} warm-start snapshots to {smartcart_db_path}!", flush=True)
    return warm_start_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save LinUCB warm-start model snapshots")
    parser.add_argument("--m", type=int, default=3, help="Pseudo-negatives count (0, 1, 3, 5)")
    parser.add_argument("--n-cap", type=int, default=10000, help="Observation cap N_cap (or 0 for cold start)")
    parser.add_argument("--lambda-reg", type=float, default=1.0, help="Ridge regularization parameter")
    parser.add_argument("--alpha", type=float, default=0.5, help="LinUCB exploration parameter")
    parser.add_argument("--version", default="v1", help="Version tag (e.g. v1)")
    args = parser.parse_args()

    run_warm_start(
        m=args.m,
        n_cap=args.n_cap,
        lambda_reg=args.lambda_reg,
        alpha=args.alpha,
        version_tag=args.version
    )

