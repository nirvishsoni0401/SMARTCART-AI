from typing import List, Dict, Any, Optional
import numpy as np
from app.recommender.base import BaseRecommender, CandidateItem, ScoredCandidate

class LinUCBRecommender(BaseRecommender):
    """
    Contextual Bandit recommendation policy using LinUCB with Sherman-Morrison
    rank-1 updates for O(d^2) matrix inverse maintenance.

    Parameters:
      feature_dim: dimension d of feature vector phi (default 36)
      alpha: exploration weight parameter (default 0.5)
      lambda_reg: ridge regularization weight for identity prior (default 1.0)
    """

    def __init__(self, feature_dim: int = 36, alpha: float = 0.5, lambda_reg: float = 1.0):
        super().__init__(name="linucb", feature_dim=feature_dim)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)

        # Initialize A = lambda * I, b = zeros(d)
        self.A = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
        # Maintain A_inv directly using Sherman-Morrison
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
        scored: List[ScoredCandidate] = []

        for cand in candidates:
            phi = cand.feature_vector
            expected_reward = float(np.dot(theta, phi))
            
            # Confidence bound: alpha * sqrt(phi^T A_inv phi)
            variance = float(phi @ self.A_inv @ phi)
            variance = max(0.0, variance)  # Prevent negative underflow
            ucb_bonus = self.alpha * np.sqrt(variance)
            total_score = expected_reward + ucb_bonus

            # An item is flagged as exploration if exploration bonus is significant (> 15% of total score or cold user)
            is_exploring = (ucb_bonus > 0.3) or (total_score > 0 and (ucb_bonus / max(1e-5, abs(total_score))) > 0.25)

            scored.append(
                ScoredCandidate(
                    candidate=cand,
                    score=total_score,
                    ucb_bonus=ucb_bonus,
                    expected_reward=expected_reward,
                    was_exploration=is_exploring
                )
            )

        # Sort descending by total UCB score
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def update(self, feature_vector: np.ndarray, reward: float) -> None:
        """
        Incremental Sherman-Morrison rank-1 update:
        A_new = A + phi phi^T
        A_inv_new = A_inv - (A_inv phi phi^T A_inv) / (1 + phi^T A_inv phi)
        b_new = b + r * phi
        """
        phi = np.asarray(feature_vector, dtype=np.float64)
        reward = float(reward)

        # 1. Update A
        self.A += np.outer(phi, phi)

        # 2. Sherman-Morrison update for A_inv
        w = self.A_inv @ phi
        denominator = 1.0 + float(np.dot(phi, w))
        if abs(denominator) > 1e-12:
            self.A_inv -= np.outer(w, w) / denominator

        # 3. Update b
        self.b += reward * phi
        self.n_updates += 1

        # Periodic re-synchronization every 1000 updates to remove numerical drift
        if self.n_updates % 1000 == 0:
            try:
                self.A_inv = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                pass

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "n_updates": self.n_updates,
            "A": self.A.tolist(),
            "A_inv": self.A_inv.tolist(),
            "b": self.b.tolist()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.feature_dim = state.get("feature_dim", self.feature_dim)
        self.alpha = float(state.get("alpha", self.alpha))
        self.lambda_reg = float(state.get("lambda_reg", self.lambda_reg))
        self.n_updates = int(state.get("n_updates", 0))
        self.A = np.array(state["A"], dtype=np.float64)
        self.A_inv = np.array(state["A_inv"], dtype=np.float64)
        self.b = np.array(state["b"], dtype=np.float64)

