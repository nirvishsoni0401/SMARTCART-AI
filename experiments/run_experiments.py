import os
import sys
import time
import math
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add backend and project root directory to sys.path
root_path = Path(__file__).resolve().parent.parent
backend_path = root_path / "backend"
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(backend_path))

from app.database import SessionLocal, init_db
from app.models import Product
from app.recommender.linucb import LinUCBRecommender
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.linucb_thompson import LinThompsonSamplingRecommender
from app.recommender.epsilon_greedy import EpsilonGreedyRecommender
from app.recommender.popularity import PopularityRecommender
from app.recommender.random_policy import RandomRecommender
from app.recommender.base import CandidateItem
from app.services.feature_engineering import UserContext, extract_feature_vector
from app.services.diversity import apply_diversity_reranking
from experiments.simulator import ShoppingSimulator, CustomerPersona
from experiments.metrics import (
    compute_ndcg, compute_precision_at_k, compute_intra_list_diversity,
    compute_catalogue_coverage, compute_novelty
)

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def instantiate_policies(feature_dim: int = 36, alpha: float = 0.5, gamma: float = 0.95):
    return {
        "LinUCB": LinUCBRecommender(feature_dim=feature_dim, alpha=alpha),
        "DiscountedLinUCB": DiscountedLinUCBRecommender(feature_dim=feature_dim, alpha=alpha, gamma=gamma),
        "LinTS": LinThompsonSamplingRecommender(feature_dim=feature_dim, v_variance=0.25),
        "EpsilonGreedy": EpsilonGreedyRecommender(feature_dim=feature_dim, epsilon=0.1),
        "Popularity": PopularityRecommender(feature_dim=feature_dim),
        "Random": RandomRecommender(feature_dim=feature_dim),
    }

def load_catalog_products() -> List[Any]:
    """Loads active products from database with a fixed-seed synthetic fallback."""
    init_db()
    db = SessionLocal()
    try:
        products = db.query(Product).filter(Product.is_active == True).all()
        if not products:
            products = db.query(Product).all()
    finally:
        db.close()

    if not products:
        print("[CatalogLoader] Warning: No products found in DB. Generating fixed-seed synthetic fallback (120 items).")
        import random
        from types import SimpleNamespace
        rng = random.Random(42)
        cats = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Books"]
        products = []
        for i in range(1, 121):
            cat = rng.choice(cats)
            p = SimpleNamespace(
                id=i,
                name=f"Synthetic {cat} SKU #{i}",
                category=cat,
                brand=f"Brand_{rng.randint(1, 8)}",
                price=round(rng.uniform(350.0, 18000.0), 2),
                mrp=round(rng.uniform(500.0, 22000.0), 2),
                discount_pct=round(rng.uniform(0.0, 45.0), 1),
                rating=round(rng.uniform(3.6, 4.9), 1),
                rating_count=rng.randint(15, 600),
                popularity_score=round(rng.uniform(25.0, 95.0), 1),
                stock=50,
                attributes={"is_synthetic": True},
                source="synthetic",
                is_active=True
            )
            products.append(p)
    return products

def run_experiment_suite(num_rounds: int = 1500, num_seeds: int = 5, drift_at_round: int = 750):
    print("=" * 70)
    print("         SMARTCART AI — EXPERIMENT & EVALUATION SUITE            ")
    print(f" Rounds per run: {num_rounds} | Random Seeds: {num_seeds} | Drift round: {drift_at_round}")
    print("=" * 70)

    # 1. Load active products from SQLite DB with synthetic fallback
    products = load_catalog_products()

    if not products:
        raise ValueError("No products found in DB! Please run backend/app/seed.py first.")

    total_catalog_size = len(products)
    print(f"[Setup] Loaded {total_catalog_size} products from SQLite catalog.")

    # Popularity distribution for novelty computation
    pop_scores = np.array([p.popularity_score for p in products], dtype=float)
    pop_probs = {p.id: float(p.popularity_score / np.sum(pop_scores)) for p in products}

    algo_names = ["LinUCB", "DiscountedLinUCB", "LinTS", "EpsilonGreedy", "Popularity", "Random"]

    # Storage for multi-seed curves
    all_cumulative_rewards = {name: np.zeros((num_seeds, num_rounds)) for name in algo_names}
    all_cumulative_regret = {name: np.zeros((num_seeds, num_rounds)) for name in algo_names}
    all_metrics = {name: [] for name in algo_names}

    for seed_idx, seed in enumerate(range(42, 42 + num_seeds)):
        print(f"\n--- Running Seed {seed_idx + 1}/{num_seeds} (Seed: {seed}) ---")
        simulator = ShoppingSimulator(catalog_products=products, seed=seed)
        policies = instantiate_policies(feature_dim=36, alpha=0.5, gamma=0.95)

        # Trackers per seed
        cum_reward = {name: 0.0 for name in algo_names}
        cum_regret = {name: 0.0 for name in algo_names}
        clicks = {name: 0 for name in algo_names}
        purchases = {name: 0 for name in algo_names}
        impressions = {name: 0 for name in algo_names}
        dislikes = {name: 0 for name in algo_names}
        returns = {name: 0 for name in algo_names}
        recommended_ids = {name: set() for name in algo_names}
        ndcg_list = {name: [] for name in algo_names}
        precision_list = {name: [] for name in algo_names}
        diversity_list = {name: [] for name in algo_names}
        latencies = {name: [] for name in algo_names}

        for t in range(num_rounds):
            # Mid-run preference drift check
            if t == drift_at_round:
                simulator.trigger_preference_drift()

            persona, session_ctx = simulator.sample_customer_session()

            # Sample 40 candidates for this round
            cand_products = simulator.rng.sample(products, min(40, total_catalog_size))
            oracle_reward, _ = simulator.compute_oracle_best_reward(persona, cand_products)

            # UserContext
            ctx = UserContext(
                session_id=session_ctx["session_id"],
                session_category=session_ctx["session_category"],
                session_budget=session_ctx["session_budget"]
            )

            # Build CandidateItem list with feature vectors
            candidate_items = [
                CandidateItem(p.id, p, extract_feature_vector(ctx, p))
                for p in cand_products
            ]

            # Evaluate each algorithm on the same user session
            for name in algo_names:
                policy = policies[name]
                t0 = time.perf_counter()
                ranked = policy.select(candidate_items, top_k=10)
                diverse = apply_diversity_reranking(ranked, top_k=6)
                t1 = time.perf_counter()
                latencies[name].append((t1 - t0) * 1000.0)

                # Simulate user feedback on the top-1 recommended item
                top_item = diverse[0] if diverse else None
                if top_item:
                    prod = top_item.candidate.product_data
                    ev_type, r, ground_utility = simulator.simulate_user_response(persona, prod)
                    
                    # Update policy
                    policy.update(top_item.candidate.feature_vector, r)

                    # Update trackers
                    cum_reward[name] += r
                    regret_step = max(0.0, oracle_reward - r)
                    cum_regret[name] += regret_step

                    impressions[name] += 1
                    if ev_type == "click":
                        clicks[name] += 1
                    elif ev_type == "cart_add":
                        clicks[name] += 1
                    elif ev_type == "purchase":
                        clicks[name] += 1
                        purchases[name] += 1
                    elif ev_type == "dislike":
                        dislikes[name] += 1
                    elif ev_type == "return":
                        returns[name] += 1

                    for it in diverse:
                        recommended_ids[name].add(it.candidate.product_id)

                    # Relevance for ranking metrics
                    relevances = [max(0.0, simulator.simulate_user_response(persona, it.candidate.product_data)[1]) for it in diverse]
                    ndcg_list[name].append(compute_ndcg(relevances, k=6))
                    precision_list[name].append(compute_precision_at_k(relevances, k=6))
                    diversity_list[name].append(compute_intra_list_diversity([it.candidate.product_data for it in diverse], k=6))

                all_cumulative_rewards[name][seed_idx, t] = cum_reward[name]
                all_cumulative_regret[name][seed_idx, t] = cum_regret[name]

        # Record metrics for this seed
        for name in algo_names:
            impr = max(1, impressions[name])
            c = clicks[name]
            p = purchases[name]
            total_events = impr + c + purchases[name] + dislikes[name] + returns[name]

            all_metrics[name].append({
                "seed": seed,
                "cumulative_reward": cum_reward[name],
                "average_reward": cum_reward[name] / float(num_rounds),
                "cumulative_regret": cum_regret[name],
                "ctr": (c / impr) * 100.0,
                "conversion_rate": (p / max(1, c)) * 100.0,
                "negative_feedback_rate": ((dislikes[name] + returns[name]) / max(1, total_events)) * 100.0,
                "ndcg_at_k": float(np.mean(ndcg_list[name])),
                "precision_at_k": float(np.mean(precision_list[name])),
                "diversity": float(np.mean(diversity_list[name])),
                "coverage": (len(recommended_ids[name]) / float(total_catalog_size)) * 100.0,
                "avg_latency_ms": float(np.mean(latencies[name]))
            })

    # 3. Aggregate & Report Mean +/- 95% CI
    summary_rows = []
    print("\n" + "=" * 90)
    print("               FINAL BENCHMARK RESULTS (MEAN ± 95% CONFIDENCE INTERVALS)")
    print("=" * 90)
    print(f"{'Algorithm':<18} | {'Cum Reward':<16} | {'Cum Regret':<16} | {'CTR (%)':<10} | {'Conv (%)':<10} | {'Cov (%)':<10} | {'Latency':<9}")
    print("-" * 90)

    for name in algo_names:
        df_seed = pd.DataFrame(all_metrics[name])
        mean_r = df_seed["cumulative_reward"].mean()
        ci_r = 1.96 * (df_seed["cumulative_reward"].std() / math.sqrt(num_seeds)) if num_seeds > 1 else 0.0

        mean_reg = df_seed["cumulative_regret"].mean()
        ci_reg = 1.96 * (df_seed["cumulative_regret"].std() / math.sqrt(num_seeds)) if num_seeds > 1 else 0.0

        mean_ctr = df_seed["ctr"].mean()
        mean_conv = df_seed["conversion_rate"].mean()
        mean_cov = df_seed["coverage"].mean()
        mean_lat = df_seed["avg_latency_ms"].mean()
        mean_div = df_seed["diversity"].mean()
        mean_ndcg = df_seed["ndcg_at_k"].mean()

        print(f"{name:<18} | {mean_r:7.1f} ± {ci_r:5.1f} | {mean_reg:7.1f} ± {ci_reg:5.1f} | {mean_ctr:6.2f}%   | {mean_conv:6.2f}%   | {mean_cov:6.1f}%   | {mean_lat:5.2f} ms")

        summary_rows.append({
            "Algorithm": name,
            "Cumulative_Reward_Mean": round(mean_r, 2),
            "Cumulative_Reward_CI": round(ci_r, 2),
            "Cumulative_Regret_Mean": round(mean_reg, 2),
            "Cumulative_Regret_CI": round(ci_reg, 2),
            "CTR_Percent": round(mean_ctr, 2),
            "Conversion_Rate_Percent": round(mean_conv, 2),
            "NDCG_at_K": round(mean_ndcg, 4),
            "Diversity": round(mean_div, 4),
            "Catalogue_Coverage_Percent": round(mean_cov, 2),
            "Avg_Latency_ms": round(mean_lat, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = RESULTS_DIR / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n[Artifact] Saved benchmark summary to {summary_csv_path}")

    # 4. Generate Publication-Quality Plots
    generate_publication_plots(all_cumulative_rewards, all_cumulative_regret, summary_df, num_rounds, drift_at_round)

    # 5. Run Reward-Sensitivity & Cold-Start Analysis
    run_sensitivity_and_cold_start(products, num_rounds=500)

def generate_publication_plots(rewards_dict, regret_dict, summary_df, num_rounds, drift_at_round):
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    colors = {
        "LinUCB": "#4f46e5",
        "DiscountedLinUCB": "#06b6d4",
        "LinTS": "#10b981",
        "EpsilonGreedy": "#f59e0b",
        "Popularity": "#8b5cf6",
        "Random": "#94a3b8"
    }

    rounds = np.arange(1, num_rounds + 1)

    # --- Plot 1: Learning Curves (Cumulative Reward) ---
    plt.figure(figsize=(10, 6))
    for name, matrix in rewards_dict.items():
        mean_curve = np.mean(matrix, axis=0)
        std_err = np.std(matrix, axis=0) / math.sqrt(matrix.shape[0])
        plt.plot(rounds, mean_curve, label=name, color=colors.get(name, "#000"), linewidth=2.2)
        plt.fill_between(rounds, mean_curve - 1.96 * std_err, mean_curve + 1.96 * std_err, alpha=0.15, color=colors.get(name, "#000"))

    plt.axvline(x=drift_at_round, color='#ef4444', linestyle='--', linewidth=1.5, label='Preference Drift Event')
    plt.title("Cumulative Reward Learning Curves (10,000+ Interactions • 95% CI)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Interaction Rounds (t)", fontsize=11)
    plt.ylabel("Cumulative Reward", fontsize=11)
    plt.legend(frameon=True, loc='upper left', fontsize=10)
    plt.tight_layout()
    p1 = PLOTS_DIR / "learning_curves_cumulative_reward.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {p1}")

    # --- Plot 2: Cumulative Regret Curves ---
    plt.figure(figsize=(10, 6))
    for name, matrix in regret_dict.items():
        mean_curve = np.mean(matrix, axis=0)
        plt.plot(rounds, mean_curve, label=name, color=colors.get(name, "#000"), linewidth=2.0)

    plt.axvline(x=drift_at_round, color='#ef4444', linestyle='--', linewidth=1.5, label='Preference Drift Event')
    plt.title("Cumulative Regret Comparison (Oracle Policy Ground Truth)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Interaction Rounds (t)", fontsize=11)
    plt.ylabel("Cumulative Regret (Lower is Better)", fontsize=11)
    plt.legend(frameon=True, loc='upper left', fontsize=10)
    plt.tight_layout()
    p2 = PLOTS_DIR / "cumulative_regret.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {p2}")

    # --- Plot 3: Preference Drift Detail (Post-drift adaptation slope) ---
    plt.figure(figsize=(9, 5))
    post_rounds = rounds[drift_at_round:]
    for name in ["DiscountedLinUCB", "LinUCB", "Popularity"]:
        matrix = rewards_dict[name]
        mean_curve = np.mean(matrix, axis=0)[drift_at_round:]
        plt.plot(post_rounds, mean_curve - mean_curve[0], label=f"{name} Post-Drift Recovery", color=colors[name], linewidth=2.5)

    plt.title("Post-Drift Adaptation: Discounted LinUCB vs Standard LinUCB", fontsize=13, fontweight='bold')
    plt.xlabel("Rounds Post Preference Shift", fontsize=11)
    plt.ylabel("Reward Accumulated After Shift", fontsize=11)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p3 = PLOTS_DIR / "preference_drift_adaptation.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {p3}")

    # --- Plot 4: CTR & Conversion Rate Bar Chart ---
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.35
    plt.bar(x - width/2, summary_df["CTR_Percent"], width, label='Click-Through Rate (CTR %)', color='#4f46e5', alpha=0.9)
    plt.bar(x + width/2, summary_df["Conversion_Rate_Percent"], width, label='Conversion Rate (%)', color='#10b981', alpha=0.9)
    plt.xticks(x, summary_df["Algorithm"], rotation=15, fontsize=10, fontweight='semibold')
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.title("CTR & Conversion Rate Benchmark Across Algorithms", fontsize=13, fontweight='bold')
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p4 = PLOTS_DIR / "ctr_and_conversion_rates.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {p4}")

    # --- Plot 5: Diversity & Catalogue Coverage ---
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary_df))
    plt.bar(x - width/2, summary_df["Diversity"] * 100.0, width, label='Intra-List Diversity (%)', color='#f59e0b', alpha=0.9)
    plt.bar(x + width/2, summary_df["Catalogue_Coverage_Percent"], width, label='Catalogue Coverage (%)', color='#8b5cf6', alpha=0.9)
    plt.xticks(x, summary_df["Algorithm"], rotation=15, fontsize=10, fontweight='semibold')
    plt.ylabel("Score / Percentage (%)", fontsize=11)
    plt.title("Intra-List Diversity & Catalogue Coverage (Exploration Breadth)", fontsize=13, fontweight='bold')
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p5 = PLOTS_DIR / "intra_list_diversity_and_coverage.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {p5}")

def run_sensitivity_and_cold_start(products, num_rounds: int = 500):
    print("\n[Analysis] Running Alpha Exploration Sensitivity & Cold-Start Analysis...")
    alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
    alpha_rewards = {}

    simulator = ShoppingSimulator(catalog_products=products, seed=99)
    for a in alphas:
        model = LinUCBRecommender(feature_dim=36, alpha=a)
        rewards = []
        cum = 0.0
        for _ in range(num_rounds):
            persona, s_ctx = simulator.sample_customer_session()
            cands = [CandidateItem(p.id, p, extract_feature_vector(UserContext(session_id=s_ctx["session_id"]), p)) for p in simulator.rng.sample(products, 30)]
            ranked = model.select(cands, top_k=5)
            if ranked:
                _, r, _ = simulator.simulate_user_response(persona, ranked[0].candidate.product_data)
                model.update(ranked[0].candidate.feature_vector, r)
                cum += r
            rewards.append(cum)
        alpha_rewards[a] = rewards

    plt.figure(figsize=(9, 5))
    for a, rew in alpha_rewards.items():
        plt.plot(rew, label=f"alpha = {a}", linewidth=2.0)
    plt.title("LinUCB Hyperparameter Sensitivity: Exploration Parameter alpha", fontsize=13, fontweight='bold')
    plt.xlabel("Rounds", fontsize=11)
    plt.ylabel("Cumulative Reward", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    p_alpha = PLOTS_DIR / "alpha_sensitivity.png"
    plt.savefig(p_alpha, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {p_alpha}")

    # Cold Start Trajectory (First 150 interactions)
    plt.figure(figsize=(9, 5))
    plt.plot(np.array(alpha_rewards[0.5][:150]), label="LinUCB (alpha=0.5)", color="#4f46e5", linewidth=2.5)
    plt.title("Cold-Start Trajectory (First 150 User Interactions)", fontsize=13, fontweight='bold')
    plt.xlabel("Initial Rounds (Cold Start)", fontsize=11)
    plt.ylabel("Cumulative Reward", fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    p_cold = PLOTS_DIR / "cold_start_trajectory.png"
    plt.savefig(p_cold, dpi=300)
    plt.close()
    print(f"[Plot] Saved: {p_cold}")

if __name__ == "__main__":
    run_experiment_suite(num_rounds=1500, num_seeds=5, drift_at_round=750)
