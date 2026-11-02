from typing import List, Dict
import numpy as np
from app.recommender.base import ScoredCandidate

def apply_diversity_reranking(
    scored_candidates: List[ScoredCandidate],
    top_k: int = 10,
    max_per_category: int = 3,
    mmr_lambda: float = 0.65
) -> List[ScoredCandidate]:
    """
    Stage 3 Diversity Re-ranking:
    1. Enforces hard constraint: at most max_per_category (default 3) items per category in top-10.
    2. Applies MMR-style penalty against near-duplicates (same category and brand).
    """
    if len(scored_candidates) <= 1:
        return scored_candidates

    selected: List[ScoredCandidate] = []
    category_counts: Dict[str, int] = {}
    remaining = list(scored_candidates)

    while remaining and len(selected) < top_k:
        best_idx = -1
        best_mmr_score = -float("inf")

        for idx, item in enumerate(remaining):
            cat = getattr(item.candidate.product_data, "category", "")
            brand = getattr(item.candidate.product_data, "brand", "")

            # Hard category cap check
            if category_counts.get(cat, 0) >= max_per_category:
                continue

            # Compute maximum similarity to already selected items
            max_sim = 0.0
            for sel in selected:
                sel_cat = getattr(sel.candidate.product_data, "category", "")
                sel_brand = getattr(sel.candidate.product_data, "brand", "")

                sim = 0.0
                if sel_cat == cat:
                    sim += 0.5
                if sel_brand == brand:
                    sim += 0.5
                if sim > max_sim:
                    max_sim = sim

            # MMR combined objective: lambda * score - (1 - lambda) * max_sim
            normalized_score = item.score
            mmr_val = mmr_lambda * normalized_score - (1.0 - mmr_lambda) * (max_sim * 5.0)

            if mmr_val > best_mmr_score:
                best_mmr_score = mmr_val
                best_idx = idx

        if best_idx == -1:
            # If all remaining violate category constraints, relax cap to fill top_k
            best_idx = 0

        chosen = remaining.pop(best_idx)
        chosen_cat = getattr(chosen.candidate.product_data, "category", "")
        category_counts[chosen_cat] = category_counts.get(chosen_cat, 0) + 1
        selected.append(chosen)

    return selected

