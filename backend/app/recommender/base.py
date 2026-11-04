from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np

class CandidateItem:
    """Wrapper for a candidate product with its computed feature vector phi."""
    def __init__(self, product_id: int, product_data: Any, feature_vector: np.ndarray):
        self.product_id = product_id
        self.product_data = product_data
        self.feature_vector = feature_vector

class ScoredCandidate:
    """Result of policy scoring."""
    def __init__(
        self,
        candidate: CandidateItem,
        score: float,
        ucb_bonus: float = 0.0,
        expected_reward: float = 0.0,
        was_exploration: bool = False
    ):
        self.candidate = candidate
        self.score = float(score)
        self.ucb_bonus = float(ucb_bonus)
        self.expected_reward = float(expected_reward)
        self.was_exploration = bool(was_exploration)

class BaseRecommender(ABC):
    """
    Abstract Base Class for Multi-Armed Bandit recommendation policies.
    Every policy supports:
      - select(candidates, top_k) -> List[ScoredCandidate]
      - update(feature_vector, reward)
      - get_state() -> dict
      - load_state(dict)
    """

    def __init__(self, name: str, feature_dim: int = 36):
        self.name = name
        self.feature_dim = feature_dim
        self.n_updates = 0

    @abstractmethod
    def select(
        self,
        candidates: List[CandidateItem],
        top_k: int = 10,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ScoredCandidate]:
        """Rank and return top_k candidates with predicted scores and exploration flags."""
        pass

    @abstractmethod
    def update(self, feature_vector: np.ndarray, reward: float) -> None:
        """Online parameter update based on observed reward signal."""
        pass

    def correct_reward(self, feature_vector: np.ndarray, delta_reward: float) -> None:
        """
        Revision/correction for an already-observed sample (e.g. delayed return compensation).
        Must update ONLY the reward vector b (b += delta_reward * x) and recompute theta,
        leaving covariance matrix A and its inverse A_inv strictly unchanged.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Serialize internal policy state to JSON-serializable dictionary."""
        pass

    @abstractmethod
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore internal policy state from dictionary."""
        pass

    def get_theta(self) -> np.ndarray:
        """Return the current mean parameter vector theta (default zeros if non-linear)."""
        return np.zeros(self.feature_dim, dtype=float)

    def warm_start_from_stats(
        self,
        stats: Any,
        n_cap: Optional[int] = None,
        lambda_reg: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Base warm-start implementation for linear contextual bandit policies.
        Scales sufficient statistics A_hist and b_hist by s = min(1.0, n_cap / n_hist).
        A_init = lambda * I + s * A_hist
        b_init = s * b_hist
        """
        A_hist = np.asarray(stats["A_hist"], dtype=np.float64)
        b_hist = np.asarray(stats["b_hist"], dtype=np.float64)
        n_hist = int(stats.get("n_hist", stats.get("n_examples", 1)))

        if A_hist.shape != (self.feature_dim, self.feature_dim):
            raise ValueError(f"A_hist shape {A_hist.shape} does not match feature_dim {self.feature_dim}")

        if lambda_reg is not None and hasattr(self, "lambda_reg"):
            self.lambda_reg = float(lambda_reg)
        lam = getattr(self, "lambda_reg", 1.0)

        if n_cap is not None:
            if n_cap <= 0:
                s = 0.0
            else:
                s = min(1.0, float(n_cap) / float(max(1, n_hist)))
        else:
            s = 1.0

        A_init = lam * np.eye(self.feature_dim, dtype=np.float64) + s * A_hist
        b_init = s * b_hist

        # Enforce symmetry
        A_init = (A_init + A_init.T) / 2.0

        # Validate SPD via Cholesky
        try:
            _ = np.linalg.cholesky(A_init)
        except np.linalg.LinAlgError:
            A_init += 1e-6 * np.eye(self.feature_dim)
            _ = np.linalg.cholesky(A_init)

        cond = float(np.linalg.cond(A_init))
        if hasattr(self, "A"):
            self.A = A_init
        if hasattr(self, "A_inv"):
            self.A_inv = np.linalg.inv(A_init)
        if hasattr(self, "b"):
            self.b = b_init
        self.n_updates = int(s * n_hist)

        return {
            "scale_s": float(s),
            "n_cap": n_cap,
            "n_hist": n_hist,
            "lambda_reg": lam,
            "condition_number": cond,
            "n_updates": self.n_updates
        }


