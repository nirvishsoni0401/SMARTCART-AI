import random
import math
import numpy as np
from typing import Dict, Any, List
from app.database import products_collection, experiments_collection
from app.recommendation.feature_engineering import extract_features, FEATURE_DIMENSION
from app.recommendation.linucb import LinUCBAgent


class SimulationEnvironment:
    """
    Offline research simulation framework to compare:
    1. Popularity-Based
    2. Epsilon-Greedy
    3. Contextual LinUCB
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.products = list(products_collection.find({"is_active": True}, {"_id": 0}))
        # Cohorts of synthetic customer archetypes
        self.cohorts = [
            {"name": "Tech Enthusiast", "preferred_category": "smartphones", "price_sensitivity": 0.2, "budget": 600.0},
            {"name": "Beauty & Wellness Lover", "preferred_category": "beauty", "price_sensitivity": 0.4, "budget": 35.0},
            {"name": "Home Decorator", "preferred_category": "furniture", "price_sensitivity": 0.5, "budget": 180.0},
            {"name": "Bargain Hunter", "preferred_category": "groceries", "price_sensitivity": 0.9, "budget": 15.0},
            {"name": "Fashion Trendsetter", "preferred_category": "mens-watches", "price_sensitivity": 0.3, "budget": 250.0}
        ]

    def _simulate_user_feedback(self, product: Dict[str, Any], cohort: Dict[str, Any]) -> Tuple_Outcome:
        cat = product.get("category", "").lower()
        price = float(product.get("price", 50.0))
        rating = float(product.get("rating", 3.0))

        # Base affinity
        cat_match = 1.0 if cat == cohort["preferred_category"] else 0.15
        price_diff = abs(price - cohort["budget"]) / max(1.0, cohort["budget"])
        price_penalty = cohort["price_sensitivity"] * min(2.0, price_diff)
        rating_bonus = (rating / 5.0) * 0.4

        utility = (cat_match * 0.7) + rating_bonus - (price_penalty * 0.4)
        utility = max(-1.0, min(1.0, utility + random.gauss(0, 0.15)))

        # Determine outcome based on utility
        if utility > 0.65:
            return "purchase", 10.0
        elif utility > 0.45:
            return "add_to_cart", 5.0
        elif utility > 0.2:
            return "click", 2.0
        elif utility < -0.3:
            return "dislike", -5.0
        else:
            return "impression", 0.0

    def run_experiment(self, num_steps: int = 150, k: int = 4) -> Dict[str, Any]:
        """
        Run side-by-side simulation of the three bandit strategies.
        """
        if not self.products:
            return {"error": "No products loaded in database for simulation"}

        # 1. State for Popularity
        pop_counts = {p["external_id"]: int(p.get("stock", 10)) for p in self.products}
        sorted_by_pop = sorted(self.products, key=lambda x: pop_counts.get(x["external_id"], 0), reverse=True)

        # 2. State for Epsilon-Greedy
        eps_values = {p["external_id"]: 0.0 for p in self.products}
        eps_pulls = {p["external_id"]: 0 for p in self.products}
        epsilon = 0.15

        # 3. State for LinUCB
        linucb = LinUCBAgent(d=FEATURE_DIMENSION, alpha=0.6)

        results = {
            "popularity": {"rewards": [], "clicks": 0, "purchases": 0, "dislikes": 0, "categories": []},
            "epsilon_greedy": {"rewards": [], "clicks": 0, "purchases": 0, "dislikes": 0, "categories": []},
            "linucb": {"rewards": [], "clicks": 0, "purchases": 0, "dislikes": 0, "categories": []}
        }

        # Step-by-step simulation loop
        for step in range(num_steps):
            cohort = random.choice(self.cohorts)
            sim_context = {
                "dominant_category": cohort["preferred_category"],
                "preferred_categories": [cohort["preferred_category"]],
                "average_viewed_price": cohort["budget"],
                "category_affinities": {cohort["preferred_category"]: 5},
                "total_interactions": step
            }

            candidate_sample = random.sample(self.products, min(24, len(self.products)))

            # --- A. Popularity Baseline ---
            pop_recs = sorted(candidate_sample, key=lambda p: pop_counts.get(p["external_id"], 0), reverse=True)[:k]
            for p in pop_recs:
                outcome, reward = self._simulate_user_feedback(p, cohort)
                results["popularity"]["rewards"].append(reward)
                if outcome == "click": results["popularity"]["clicks"] += 1
                elif outcome == "purchase": results["popularity"]["purchases"] += 1
                elif outcome == "dislike": results["popularity"]["dislikes"] += 1
                results["popularity"]["categories"].append(p.get("category"))

            # --- B. Epsilon-Greedy ---
            if random.random() < epsilon:
                eps_recs = random.sample(candidate_sample, k)
            else:
                eps_recs = sorted(candidate_sample, key=lambda p: eps_values.get(p["external_id"], 0), reverse=True)[:k]
            for p in eps_recs:
                outcome, reward = self._simulate_user_feedback(p, cohort)
                results["epsilon_greedy"]["rewards"].append(reward)
                if outcome == "click": results["epsilon_greedy"]["clicks"] += 1
                elif outcome == "purchase": results["epsilon_greedy"]["purchases"] += 1
                elif outcome == "dislike": results["epsilon_greedy"]["dislikes"] += 1
                results["epsilon_greedy"]["categories"].append(p.get("category"))
                
                # Update arm value estimate
                pid = p["external_id"]
                eps_pulls[pid] += 1
                eps_values[pid] += (reward - eps_values[pid]) / eps_pulls[pid]

            # --- C. Contextual LinUCB ---
            scored_candidates = []
            for p in candidate_sample:
                arm_id = f"cat:{p.get('category', '').lower()}"
                x = extract_features(p, sim_context)
                ucb_score, _, _ = linucb.score_arm(arm_id, x)
                scored_candidates.append((p, arm_id, x, ucb_score))

            scored_candidates.sort(key=lambda t: t[3], reverse=True)
            linucb_recs = scored_candidates[:k]

            for p, arm_id, x, _ in linucb_recs:
                outcome, reward = self._simulate_user_feedback(p, cohort)
                results["linucb"]["rewards"].append(reward)
                if outcome == "click": results["linucb"]["clicks"] += 1
                elif outcome == "purchase": results["linucb"]["purchases"] += 1
                elif outcome == "dislike": results["linucb"]["dislikes"] += 1
                results["linucb"]["categories"].append(p.get("category"))
                
                # LinUCB online update
                linucb.update_arm(arm_id, x, reward)

        # Compute summary metrics for each algorithm
        summary = {}
        for algo in ["popularity", "epsilon_greedy", "linucb"]:
            r_list = results[algo]["rewards"]
            total_impressions = len(r_list)
            cum_reward = float(np.sum(r_list))
            avg_reward = float(np.mean(r_list)) if total_impressions > 0 else 0.0
            ctr = float((results[algo]["clicks"] + results[algo]["purchases"]) / max(1, total_impressions))
            conversion = float(results[algo]["purchases"] / max(1, total_impressions))
            neg_rate = float(results[algo]["dislikes"] / max(1, total_impressions))
            unique_cats = len(set(results[algo]["categories"]))

            # Cumulative reward curve for charting
            cum_curve = np.cumsum(r_list).tolist()
            # Downsample curve to ~20 points for fast chart rendering
            step_size = max(1, len(cum_curve) // 20)
            sampled_curve = [round(cum_curve[i], 1) for i in range(0, len(cum_curve), step_size)]

            summary[algo] = {
                "algorithm": algo.replace("_", " ").title(),
                "total_impressions": total_impressions,
                "cumulative_reward": round(cum_reward, 1),
                "average_reward": round(avg_reward, 3),
                "click_through_rate": round(ctr, 3),
                "conversion_rate": round(conversion, 3),
                "negative_feedback_rate": round(neg_rate, 3),
                "category_diversity": unique_cats,
                "cumulative_reward_curve": sampled_curve
            }

        return {
            "simulation_steps": num_steps,
            "arms_tested": len(self.products),
            "results": summary
        }


# Type helper
Tuple_Outcome = tuple[str, float]

