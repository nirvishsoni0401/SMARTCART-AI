from typing import List, Dict, Any, Optional
import random
import numpy as np
from app.recommender.base import BaseRecommender, CandidateItem, ScoredCandidate

class RandomRecommender(BaseRecommender):
    """
    Uniform random baseline policy.
    Ranks items randomly to establish minimum performance floor.
    """

    def __init__(self, feature_dim: int = 36):
        super().__init__(name="random", feature_dim=feature_dim)
        self.total_reward = 0.0

    def select(
        self,
        candidates: List[CandidateItem],
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        scored: List[ScoredCandidate] = []
        for cand in candidates:
            score = random.uniform(0.0, 1.0)
            scored.append(
                ScoredCandidate(
                    candidate=cand,
                    score=score,
                    ucb_bonus=score,
                    expected_reward=0.5,
                    was_exploration=True
                )
            )

        # Sort descending by score with deterministic product_id tie-breaking
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

