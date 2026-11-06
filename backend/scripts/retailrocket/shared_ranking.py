"""
SmartCart AI — Shared Deterministic Tie-Aware Ranking Module
Provides a single, shared, deterministic tie-aware ranking function for all policies.
Order of precedence:
  1. Policy Score (descending)
  2. Popularity Prior (descending)
  3. Product Index / ID (ascending)

This ensures a strict total order: no ties remain unresolved, eliminating artificial
rank-1 inflation bugs.
"""

from typing import Optional, Sequence, Tuple
import numpy as np


def compute_rank(
    scores: np.ndarray,
    target_idx: int,
    pops: np.ndarray,
    item_ids: Optional[np.ndarray] = None,
    atol: float = 1e-12
) -> int:
    """
    Computes the 1-based rank of target_idx under the strict lexicographic ordering:
      (-score, -pop, item_id)
    An item j is strictly ranked ahead of target t if:
      - score[j] > score[t] + atol, OR
      - |score[j] - score[t]| <= atol AND pop[j] > pop[t] + atol, OR
      - |score[j] - score[t]| <= atol AND |pop[j] - pop[t]| <= atol AND item_id[j] < item_id[t]
    
    Returns:
      1-based integer rank in [1, len(scores)].
    """
    n = len(scores)
    if item_ids is None:
        item_ids = np.arange(n, dtype=np.int64)
    
    s_t = scores[target_idx]
    p_t = pops[target_idx]
    id_t = item_ids[target_idx]

    score_diff = scores - s_t
    pop_diff = pops - p_t

    strictly_better = (
        (score_diff > atol) |
        ((np.abs(score_diff) <= atol) & (pop_diff > atol)) |
        ((np.abs(score_diff) <= atol) & (np.abs(pop_diff) <= atol) & (item_ids < id_t))
    )
    return int(np.count_nonzero(strictly_better) + 1)


def get_ranked_indices(
    scores: np.ndarray,
    pops: np.ndarray,
    item_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Returns candidate indices sorted from best (rank 1) to worst under the exact
    same strict lexicographic order: (-score, -pop, item_id).
    Uses np.lexsort.
    """
    n = len(scores)
    if item_ids is None:
        item_ids = np.arange(n, dtype=np.int64)
    
    # lexsort sorts by keys in reverse order: tertiary first, primary last
    return np.lexsort((item_ids, -pops, -scores))


def compute_hybrid_slate(
    rv_scores: np.ndarray,
    linucb_scores: np.ndarray,
    pops: np.ndarray,
    k_recency: int = 3,
    slate_size: int = 10,
    item_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Builds a hybrid slate taking top k_recency items from RecentlyViewed
    and filling the remaining (slate_size - k_recency) items from LinUCB without duplicates,
    using deterministic tie-breaking for both rankings.
    
    Returns:
      Array of candidate indices of length slate_size.
    """
    order_rv = get_ranked_indices(rv_scores, pops, item_ids)
    top_k_rv = order_rv[:k_recency]
    
    order_lin = get_ranked_indices(linucb_scores, pops, item_ids)
    
    # Fill remaining from order_lin without duplicates
    slate = list(top_k_rv)
    seen_set = set(top_k_rv)
    for idx in order_lin:
        if idx not in seen_set:
            slate.append(idx)
            seen_set.add(idx)
            if len(slate) == slate_size:
                break
    return np.array(slate, dtype=np.int64)

