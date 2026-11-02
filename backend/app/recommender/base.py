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

