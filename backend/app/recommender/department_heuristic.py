from typing import List, Dict, Any, Optional
import numpy as np
from app.recommender.base import BaseRecommender, CandidateItem, ScoredCandidate

class DepartmentHeuristicRecommender(BaseRecommender):
    """
    Department Heuristic baseline policy with deterministic tie-breaking.
    Ranks items based on the visitor's most recently engaged or highest affinity department (+100 boost),
    falling back to catalog popularity score. Ties are broken deterministically on product_id.
    """

    def __init__(self, feature_dim: int = 36, name: str = "department_heuristic"):
        super().__init__(name=name, feature_dim=feature_dim)
        self.total_reward = 0.0

    def select(
        self,
        candidates: List[CandidateItem],
        top_k: int = 10,
        context: Optional[Any] = None
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        # Determine target department from context
        target_cat = None
        if context is not None:
            sess_cat = getattr(context, "session_category", None)
            if sess_cat and str(sess_cat).lower() != "all":
                target_cat = str(sess_cat)
            else:
                affs = getattr(context, "category_affinities", {})
                if affs and any(v > 0 for v in affs.values()):
                    target_cat = max(affs.items(), key=lambda x: x[1])[0]

        scored: List[ScoredCandidate] = []
        for cand in candidates:
            p = cand.product_data
            p_cat = getattr(p, "category", "") if not isinstance(p, dict) else p.get("category", "")
            pop = float(getattr(p, "popularity_score", 50.0) if not isinstance(p, dict) else p.get("popularity_score", 50.0) or 50.0)

            match_boost = 100.0 if (target_cat and p_cat == target_cat) else 0.0
            score = match_boost + pop

            scored.append(
                ScoredCandidate(
                    candidate=cand,
                    score=score,
                    ucb_bonus=0.0,
                    expected_reward=score,
                    was_exploration=False
                )
            )

        # Deterministic sort: higher score first, lower product_id breaks ties
        scored.sort(key=lambda x: (-x.score, x.candidate.product_id))
        return scored[:top_k]

    def update(self, feature_vector: np.ndarray, reward: float) -> None:
        self.total_reward += float(reward)
        self.n_updates += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "n_updates": self.n_updates,
            "total_reward": self.total_reward
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.n_updates = int(state.get("n_updates", 0))
        self.total_reward = float(state.get("total_reward", 0.0))
