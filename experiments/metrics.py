import math
from typing import List, Dict, Set, Any
import numpy as np

def compute_dcg(relevance_scores: List[float], k: int = 10) -> float:
    """Discounted Cumulative Gain at rank K."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 2.0)
    return dcg

def compute_ndcg(relevance_scores: List[float], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank K."""
    actual_dcg = compute_dcg(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = compute_dcg(ideal_scores, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return actual_dcg / ideal_dcg

def compute_precision_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """Fraction of top-K recommended items that achieved positive relevance (e.g. click/cart/purchase)."""
    if not relevance_scores or k <= 0:
        return 0.0
    top_k = relevance_scores[:k]
    relevant_count = sum(1 for r in top_k if r > 0.0)
    return relevant_count / float(k)

def compute_intra_list_diversity(items: List[Any], k: int = 10) -> float:
    """
    Intra-List Diversity (ILD) at rank K.
    Measures the average pairwise category and brand distance between recommended items.
    """
    sub_items = items[:k]
    n = len(sub_items)
    if n <= 1:
        return 1.0

    total_dist = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            c1 = getattr(sub_items[i], "category", "")
            c2 = getattr(sub_items[j], "category", "")
            b1 = getattr(sub_items[i], "brand", "")
            b2 = getattr(sub_items[j], "brand", "")

            dist = 0.0
            if c1 != c2:
                dist += 0.65
            if b1 != b2:
                dist += 0.35
            total_dist += dist
            pairs += 1

    return total_dist / float(pairs) if pairs > 0 else 0.0

def compute_novelty(recommended_ids: List[int], item_popularity_probs: Dict[int, float], k: int = 10) -> float:
    """
    Novelty at rank K: average negative log2 probability of items.
    Novelty = (1 / K) * sum(-log2(P(item)))
    """
    sub_ids = recommended_ids[:k]
    if not sub_ids:
        return 0.0

    novelty_sum = 0.0
    for pid in sub_ids:
        prob = item_popularity_probs.get(pid, 0.001)
        prob = max(1e-5, min(1.0, prob))
        novelty_sum += -math.log2(prob)
    return novelty_sum / float(len(sub_ids))

def compute_catalogue_coverage(recommended_unique_ids: Set[int], total_catalogue_size: int = 630) -> float:
    """Fraction of total catalogue items exposed to at least one user."""
    if total_catalogue_size <= 0:
        return 0.0
    return len(recommended_unique_ids) / float(total_catalogue_size)

