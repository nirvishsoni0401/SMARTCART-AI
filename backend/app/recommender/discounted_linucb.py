from typing import List, Dict, Any, Optional
import numpy as np
from app.recommender.base import BaseRecommender, CandidateItem, ScoredCandidate

class DiscountedLinUCBRecommender(BaseRecommender):
    """
    Discounted LinUCB for non-stationary environments and preference drift.
    Applies an exponential forgetting factor gamma in (0, 1] to prior observations:
      A_t = gamma * A_{t-1} + (1 - gamma) * lambda * I + phi * phi^T
      b_t = gamma * b_{t-1} + r * phi
    """

    def __init__(
        self,
        feature_dim: int = 36,
        alpha: float = 0.5,
        lambda_reg: float = 1.0,
        gamma: float = 0.95
    ):
        super().__init__(name="discounted_linucb", feature_dim=feature_dim)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        self.gamma = float(gamma)

        self.A = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
        self.b = np.zeros(self.feature_dim, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)

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
            variance = float(phi @ self.A_inv @ phi)
            variance = max(0.0, variance)
            ucb_bonus = self.alpha * np.sqrt(variance)
            total_score = expected_reward + ucb_bonus

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

        # Sort descending by total UCB score with deterministic product_id tie-breaking
        scored.sort(key=lambda x: (-x.score, x.candidate.product_id))
        return scored[:top_k]

    def update(self, feature_vector: np.ndarray, reward: float) -> None:
        phi = np.asarray(feature_vector, dtype=np.float64)
        reward = float(reward)

        # Apply exponential decay to prior history
        self.A = self.gamma * self.A + (1.0 - self.gamma) * self.lambda_reg * np.eye(self.feature_dim) + np.outer(phi, phi)
        self.b = self.gamma * self.b + reward * phi
        self.n_updates += 1

        # Re-compute inverse (with regularized fallback)
        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.pinv(self.A)

    def correct_reward(self, feature_vector: np.ndarray, delta_reward: float) -> None:
        """
        Revision for an already-counted sample.
        Updates ONLY the reward vector b (b += delta_reward * phi).
        A and A_inv are unchanged.
        """
        phi = np.asarray(feature_vector, dtype=np.float64)
        self.b += float(delta_reward) * phi
        self.n_updates += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "alpha": self.alpha,
            "lambda_reg": self.lambda_reg,
            "gamma": self.gamma,
            "n_updates": self.n_updates,
            "A": self.A.tolist(),
            "A_inv": self.A_inv.tolist(),
            "b": self.b.tolist()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.feature_dim = state.get("feature_dim", self.feature_dim)
        self.alpha = float(state.get("alpha", self.alpha))
        self.lambda_reg = float(state.get("lambda_reg", self.lambda_reg))
        self.gamma = float(state.get("gamma", self.gamma))
        self.n_updates = int(state.get("n_updates", 0))
        self.A = np.array(state["A"], dtype=np.float64)
        self.A_inv = np.array(state["A_inv"], dtype=np.float64)
        self.b = np.array(state["b"], dtype=np.float64)

