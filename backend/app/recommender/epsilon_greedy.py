from typing import List, Dict, Any, Optional
import random
import numpy as np
from app.recommender.base import BaseRecommender, CandidateItem, ScoredCandidate

class EpsilonGreedyRecommender(BaseRecommender):
    """
    Epsilon-Greedy linear regression policy.
    With probability (1 - epsilon), it exploits greedy predictions theta^T phi.
    With probability epsilon, it explores random candidates.
    """

    def __init__(self, feature_dim: int = 36, epsilon: float = 0.1, lambda_reg: float = 1.0):
        super().__init__(name="epsilon_greedy", feature_dim=feature_dim)
        self.epsilon = float(epsilon)
        self.lambda_reg = float(lambda_reg)

        self.A = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)
        self.b = np.zeros(self.feature_dim, dtype=np.float64)

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def get_theta(self) -> np.ndarray:
        return self.theta

    def select(
        self,
        candidates: List[CandidateItem],
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        theta = self.theta
        explore_round = random.random() < self.epsilon

        scored: List[ScoredCandidate] = []
        for cand in candidates:
            phi = cand.feature_vector
            expected_reward = float(np.dot(theta, phi))
            
            if explore_round:
                # Random score for exploration
                exploration_noise = random.uniform(0.0, 5.0)
                score = expected_reward + exploration_noise
                ucb_bonus = exploration_noise
                was_exploration = True
            else:
                score = expected_reward
                ucb_bonus = 0.0
                was_exploration = False

            scored.append(
                ScoredCandidate(
                    candidate=cand,
                    score=score,
                    ucb_bonus=ucb_bonus,
                    expected_reward=expected_reward,
                    was_exploration=was_exploration
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def update(self, feature_vector: np.ndarray, reward: float) -> None:
        phi = np.asarray(feature_vector, dtype=np.float64)
        reward = float(reward)

        self.A += np.outer(phi, phi)
        w = self.A_inv @ phi
        denom = 1.0 + float(np.dot(phi, w))
        if abs(denom) > 1e-12:
            self.A_inv -= np.outer(w, w) / denom

        self.b += reward * phi
        self.n_updates += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "epsilon": self.epsilon,
            "lambda_reg": self.lambda_reg,
            "n_updates": self.n_updates,
            "A": self.A.tolist(),
            "A_inv": self.A_inv.tolist(),
            "b": self.b.tolist()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.feature_dim = state.get("feature_dim", self.feature_dim)
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.lambda_reg = float(state.get("lambda_reg", self.lambda_reg))
        self.n_updates = int(state.get("n_updates", 0))
        self.A = np.array(state["A"], dtype=np.float64)
        self.A_inv = np.array(state["A_inv"], dtype=np.float64)
        self.b = np.array(state["b"], dtype=np.float64)

