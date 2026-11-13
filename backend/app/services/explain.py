from typing import Optional
import numpy as np
from app.recommender.base import ScoredCandidate
from app.services.feature_engineering import UserContext, CATEGORIES

def generate_explanation(
    scored_item: ScoredCandidate,
    context: UserContext,
    theta: Optional[np.ndarray] = None,
    cold_start: bool = False,
    algorithm: Optional[str] = None
) -> str:
    """
    Generate transparent, evidence-based explanations for recommendations
    by analyzing per-feature contributions (theta_i * phi_i) and exploration flags.
    For department-restricted policies, formats as 'Because you browse {department} • {feature_reason}'
    and labels exploration slots explicitly.
    """
    cand = scored_item.candidate
    prod = cand.product_data
    phi = cand.feature_vector
    is_dept_restricted = bool(algorithm and "dept_restricted" in algorithm.lower())

    # 1. Exploration bandit recommendation
    if scored_item.was_exploration or (scored_item.ucb_bonus > 0 and scored_item.ucb_bonus > abs(scored_item.expected_reward) * 0.4):
        return "Exploration Pick: Suggested to discover new favorites outside your current department"

    # 2. Cold start scenario
    if cold_start or context.total_interactions <= 1:
        if getattr(prod, "rating", 0) >= 4.5:
            return f"Popular with shoppers • {prod.rating}★ verified satisfaction"
        return "Popular with new shoppers to help personalize your experience"

    # 3. Determine top feature reason
    feature_reason = None
    if theta is not None and len(theta) == len(phi):
        contributions = theta * phi

        # Brand affinity contribution (index 6)
        p_brand = getattr(prod, "brand", "") if not isinstance(prod, dict) else prod.get("brand", "")
        if contributions[6] > 0.2 and context.brand_affinities.get(p_brand, 0.0) > 0.3:
            feature_reason = f"From your preferred brand {p_brand}"
        # Cart intent match (index 32)
        elif contributions[32] > 0.15:
            feature_reason = "Complements items currently in your cart"
        # High discount (index 34)
        elif contributions[34] > 0.15 and getattr(prod, "discount_pct", 0) >= 25:
            feature_reason = f"Blockbuster deal: {int(prod.discount_pct)}% off today"
        # High rating (index 33)
        elif contributions[33] > 0.15 and getattr(prod, "rating", 0) >= 4.5:
            feature_reason = f"Top rated choice ({prod.rating}★ rating)"
        # Budget constraint match
        elif context.session_budget and getattr(prod, "price", 0) <= context.session_budget and (phi[9] > 0.7 or phi[10] > 0.5):
            feature_reason = f"Matches your target budget (under ₹{int(context.session_budget):,})"

    if not feature_reason:
        if getattr(prod, "rating", 0) >= 4.5:
            feature_reason = f"Top rated choice ({prod.rating}★ rating)"
        elif getattr(prod, "discount_pct", 0) >= 20:
            feature_reason = f"Great discount ({int(prod.discount_pct)}% off today)"
        else:
            feature_reason = f"Verified satisfaction ({getattr(prod, 'rating', 4.0)}★)"

    # Department-restricted formatting requirement (C2)
    if is_dept_restricted:
        p_cat = getattr(prod, "category", "") if not isinstance(prod, dict) else prod.get("category", "")
        return f"Because you browse {p_cat} • {feature_reason}"

    # Standard feature contributions for general policies
    if theta is not None and len(theta) == len(phi):
        contributions = theta * phi
        cat_contributions = [(CATEGORIES[i], contributions[i]) for i in range(6)]
        cat_contributions.sort(key=lambda x: x[1], reverse=True)
        top_cat, top_cat_val = cat_contributions[0]
        if top_cat_val > 0.2 and context.category_affinities.get(top_cat, 0.0) > 0.3:
            return f"Recommended because you frequently browse {top_cat}"

    return f"Curated selection matching your shopping profile • {feature_reason}"


