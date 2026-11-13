from typing import List, Dict, Any, Optional
import numpy as np
from app.recommender.base import BaseRecommender, CandidateItem, ScoredCandidate

class LinThompsonSamplingRecommender(BaseRecommender):
    """
    Linear Thompson Sampling (LinTS) recommendation policy.
    Samples posterior weight vector:
      theta_tilde ~ N(A_inv @ b, v^2 * A_inv)
    """

    def __init__(self, feature_dim: int = 36, v_variance: float = 0.25, lambda_reg: float = 1.0):
        super().__init__(name="linucb_thompson", feature_dim=feature_dim)
        self.v_variance = float(v_variance)
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

        mean_theta = self.theta
        # Covariance matrix for sampling
        cov = (self.v_variance ** 2) * self.A_inv
        
        # Symmetrize and ensure positive semi-definite
        cov = (cov + cov.T) / 2.0
        cov += 1e-8 * np.eye(self.feature_dim)

        try:
            sampled_theta = np.random.multivariate_normal(mean_theta, cov)
        except Exception:
            sampled_theta = mean_theta

        scored: List[ScoredCandidate] = []
        for cand in candidates:
            phi = cand.feature_vector
            expected_reward = float(np.dot(mean_theta, phi))
            sampled_reward = float(np.dot(sampled_theta, phi))
            ts_bonus = sampled_reward - expected_reward

            is_exploring = abs(ts_bonus) > 0.2

            scored.append(
                ScoredCandidate(
                    candidate=cand,
                    score=sampled_reward,
                    ucb_bonus=abs(ts_bonus),
                    expected_reward=expected_reward,
                    was_exploration=is_exploring
                )
            )

        # Sort descending by sampled score with deterministic product_id tie-breaking
        scored.sort(key=lambda x: (-x.score, x.candidate.product_id))
        return scored[:top_k]

    def update(self, feature_vector: np.ndarray, reward: float) -> None:
        phi = np.asarray(feature_vector, dtype=np.float64)
        reward = float(reward)

        self.A += np.outer(phi, phi)

        # Sherman-Morrison update for A_inv
        w = self.A_inv @ phi
        denom = 1.0 + float(np.dot(phi, w))
        if abs(denom) > 1e-12:
            self.A_inv -= np.outer(w, w) / denom

        self.b += reward * phi
        self.n_updates += 1

    def correct_reward(self, feature_vector: np.ndarray, delta_reward: float) -> None:
        phi = np.asarray(feature_vector, dtype=np.float64)
        self.b += float(delta_reward) * phi
        self.n_updates += 1

    def get_theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "v_variance": self.v_variance,
            "lambda_reg": self.lambda_reg,
            "n_updates": self.n_updates,
            "A": self.A.tolist(),
            "A_inv": self.A_inv.tolist(),
            "b": self.b.tolist()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.feature_dim = state.get("feature_dim", self.feature_dim)
        self.v_variance = float(state.get("v_variance", self.v_variance))
        self.lambda_reg = float(state.get("lambda_reg", self.lambda_reg))
        self.n_updates = int(state.get("n_updates", 0))
        self.A = np.array(state["A"], dtype=np.float64)
        self.A_inv = np.array(state["A_inv"], dtype=np.float64)
        self.b = np.array(state["b"], dtype=np.float64)

