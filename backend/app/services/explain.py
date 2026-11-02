from typing import Optional
import numpy as np
from app.recommender.base import ScoredCandidate
from app.services.feature_engineering import UserContext, CATEGORIES

def generate_explanation(
    scored_item: ScoredCandidate,
    context: UserContext,
    theta: Optional[np.ndarray] = None,
    cold_start: bool = False
) -> str:
    """
    Generate transparent, evidence-based explanations for recommendations
    by analyzing per-feature contributions (theta_i * phi_i) and exploration flags.
    """
    cand = scored_item.candidate
    prod = cand.product_data
    phi = cand.feature_vector

    # 1. Cold start scenario
    if cold_start or context.total_interactions <= 1:
        if getattr(prod, "rating", 0) >= 4.5:
            return f"Popular with shoppers • {prod.rating}★ verified satisfaction"
        return "Popular with new shoppers to help personalize your experience"

    # 2. Exploration bandit recommendation
    if scored_item.was_exploration or (scored_item.ucb_bonus > 0 and scored_item.ucb_bonus > abs(scored_item.expected_reward) * 0.4):
        return "Suggested to help our AI learn and tune your taste"

    # 3. Budget constraint match
    if context.session_budget and prod.price <= context.session_budget:
        # Check price fit feature (index 9) and in-budget (index 10)
        if phi[9] > 0.7 or phi[10] > 0.5:
            return f"Matches your target budget (under ₹{int(context.session_budget):,})"

    # 4. Compute per-feature contribution if theta available
    if theta is not None and len(theta) == len(phi):
        contributions = theta * phi

        # Category affinity contributions (indices 0..5)
        cat_contributions = [(CATEGORIES[i], contributions[i]) for i in range(6)]
        cat_contributions.sort(key=lambda x: x[1], reverse=True)
        top_cat, top_cat_val = cat_contributions[0]

        if top_cat_val > 0.2 and context.category_affinities.get(top_cat, 0.0) > 0.3:
            return f"Recommended because you frequently browse {top_cat}"

        # Brand affinity contribution (index 6)
        if contributions[6] > 0.2 and context.brand_affinities.get(prod.brand, 0.0) > 0.3:
            return f"From your preferred brand {prod.brand}"

        # Cart intent match (index 32)
        if contributions[32] > 0.15:
            return f"Complements items currently in your cart"

        # High discount (index 34)
        if contributions[34] > 0.15 and prod.discount_pct >= 25:
            return f"Blockbuster deal: {int(prod.discount_pct)}% off today"

        # High rating (index 33)
        if contributions[33] > 0.15 and prod.rating >= 4.5:
            return f"Top rated choice ({prod.rating}★ rating)"

    # Fallbacks based on direct affinity or popularity
    cat_aff = context.category_affinities.get(prod.category, 0.0)
    if cat_aff > 0.4:
        return f"Based on your frequent interest in {prod.category}"

    if getattr(prod, "rating", 0) >= 4.5:
        return f"Consistently rated {prod.rating}★ by verified buyers"

    return "Curated selection matching your shopping profile"

