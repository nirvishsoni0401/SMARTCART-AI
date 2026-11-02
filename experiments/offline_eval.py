import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

root_path = Path(__file__).resolve().parent.parent
backend_path = root_path / "backend"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(backend_path))

from app.recommender.linucb import LinUCBRecommender
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.popularity import PopularityRecommender
from app.recommender.random_policy import RandomRecommender
from app.recommender.base import CandidateItem

DATA_DIR = Path(__file__).resolve().parent / "data"

def load_retailrocket_or_fallback(num_samples: int = 2000, seed: int = 42):
    """
    Attempts to load the RetailRocket e-commerce dataset from experiments/data/events.csv.
    If not found, generates a realistic logged bandit replay dataset with known propensities.
    """
    events_file = DATA_DIR / "events.csv"
    if events_file.exists():
        print(f"[OfflineEval] Found RetailRocket dataset at {events_file}!")
        df = pd.read_csv(events_file)
        # Parse timestamp, visitorid, event, itemid
        return process_retailrocket_dataframe(df, num_samples)

    print("[OfflineEval] Notice: Real RetailRocket dataset not found at experiments/data/events.csv.")
    print("             To use real data: download from https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset")
    print("             and place events.csv into experiments/data/events.csv.")
    print("[OfflineEval] Generating synthetic logged replay dataset with logging policy propensities...")

    np.random.seed(seed)
    d = 36
    num_arms = 20
    propensity = 1.0 / num_arms  # uniform random logging policy

    logged_events = []
    # Ground truth latent linear parameter for generating true responses
    true_theta = np.random.randn(d)
    true_theta /= np.linalg.norm(true_theta)

    for i in range(num_samples):
        # Generate context and arm features
        arm_features = {arm_id: np.random.randn(d) for arm_id in range(num_arms)}
        for arm_id in arm_features:
            arm_features[arm_id] /= np.linalg.norm(arm_features[arm_id])

        # Logging policy chooses uniformly at random
        logged_arm = np.random.randint(0, num_arms)
        phi_logged = arm_features[logged_arm]

        # Ground truth probabilistic response
        latent_val = float(np.dot(true_theta, phi_logged))
        prob_positive = 1.0 / (1.0 + np.exp(-3.0 * latent_val))
        
        if np.random.rand() < prob_positive:
            reward = 10.0 if np.random.rand() < 0.3 else 2.0
        else:
            reward = 0.0

        logged_events.append({
            "round": i,
            "arm_features": arm_features,
            "logged_arm": logged_arm,
            "reward": reward,
            "propensity": propensity
        })

    return logged_events

def run_replay_evaluation():
    print("=" * 70)
    print("      OFFLINE POLICY EVALUATION — REPLAY METHOD & IPS           ")
    print("               (Li et al., WSDM 2010 Formulation)                ")
    print("=" * 70)

    logged_events = load_retailrocket_or_fallback(num_samples=3000, seed=42)
    d = 36

    policies = {
        "LinUCB (alpha=0.5)": LinUCBRecommender(feature_dim=d, alpha=0.5),
        "DiscountedLinUCB": DiscountedLinUCBRecommender(feature_dim=d, alpha=0.5, gamma=0.98),
        "Popularity Baseline": PopularityRecommender(feature_dim=d),
        "Random Policy": RandomRecommender(feature_dim=d)
    }

    results = {}

    for name, policy in policies.items():
        matched_events = 0
        total_ips_reward = 0.0
        total_unweighted_reward = 0.0

        for ev in logged_events:
            arm_features = ev["arm_features"]
            # Candidate items for target policy
            candidates = [
                CandidateItem(arm_id, None, phi)
                for arm_id, phi in arm_features.items()
            ]

            # Policy chooses best arm
            ranked = policy.select(candidates, top_k=1)
            chosen_arm = ranked[0].candidate.product_id

            # Replay match condition: chosen arm == logged arm
            if chosen_arm == ev["logged_arm"]:
                matched_events += 1
                r = ev["reward"]
                p0 = ev["propensity"]
                
                # IPS Weight: w = 1 / p_0(a|x)
                ips_reward = r / p0
                total_ips_reward += ips_reward
                total_unweighted_reward += r

                # Update target policy with matched event
                policy.update(arm_features[chosen_arm], r)

        acceptance_rate = (matched_events / len(logged_events)) * 100.0
        mean_unweighted_reward = total_unweighted_reward / max(1, matched_events)
        mean_ips_reward = total_ips_reward / len(logged_events)

        results[name] = {
            "Matched Events": matched_events,
            "Acceptance Rate (%)": round(acceptance_rate, 2),
            "Cumulative Replay Reward": round(total_unweighted_reward, 2),
            "Mean Reward per Match": round(mean_unweighted_reward, 3),
            "Mean IPS Reward": round(mean_ips_reward, 3)
        }

    res_df = pd.DataFrame(results).T
    print("\n" + "=" * 80)
    print(res_df.to_string())
    print("=" * 80)

    out_csv = Path(__file__).resolve().parent / "results" / "offline_eval_results.csv"
    res_df.to_csv(out_csv)
    print(f"\n[Artifact] Offline evaluation summary written to {out_csv}")

if __name__ == "__main__":
    run_replay_evaluation()
