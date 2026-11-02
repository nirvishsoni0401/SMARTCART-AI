import numpy as np
from typing import Dict, Any, Tuple, Optional
from app.recommendation.feature_engineering import FEATURE_DIMENSION


class LinUCBAgent:
    """
    Contextual Multi-Armed Bandit using the LinUCB algorithm.
    Maintains ridge regression matrices A_a and vectors b_a for each arm.
    """

    def __init__(self, d: int = FEATURE_DIMENSION, alpha: float = 0.8, l2_reg: float = 1.0):
        self.d = d
        self.alpha = alpha
        self.l2_reg = l2_reg
        # Arm matrices: arm_id -> { "A": np.ndarray (d x d), "b": np.ndarray (d,) }
        self.arms: Dict[str, Dict[str, np.ndarray]] = {}
        # Global fallback arm
        self._init_arm("__global__")

    def _init_arm(self, arm_id: str):
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                "A": np.identity(self.d, dtype=np.float64) * self.l2_reg,
                "b": np.zeros(self.d, dtype=np.float64),
                "count": 0
            }

    def get_arm_params(self, arm_id: str) -> Dict[str, np.ndarray]:
        if arm_id not in self.arms:
            self._init_arm(arm_id)
        return self.arms[arm_id]

    def score_arm(self, arm_id: str, x: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute LinUCB score:
        theta_hat = A^-1 * b
        expected_reward = theta_hat^T * x
        exploration_bonus = alpha * sqrt(x^T * A^-1 * x)
        ucb_score = expected_reward + exploration_bonus
        Returns (ucb_score, expected_reward, exploration_bonus)
        """
        arm = self.get_arm_params(arm_id)
        A = arm["A"]
        b = arm["b"]

        try:
            A_inv = np.linalg.pinv(A)
            theta_hat = A_inv.dot(b)
            expected_reward = float(np.dot(theta_hat, x))
            var = float(np.dot(np.dot(x.T, A_inv), x))
            exploration_bonus = float(self.alpha * np.sqrt(max(0.0, var)))
            ucb_score = expected_reward + exploration_bonus
            return ucb_score, expected_reward, exploration_bonus
        except Exception:
            return 0.0, 0.0, 0.0

    def update_arm(self, arm_id: str, x: np.ndarray, reward: float):
        """
        Ridge regression update:
        A_a <- A_a + x * x^T
        b_a <- b_a + r * x
        """
        self._init_arm(arm_id)
        arm = self.arms[arm_id]
        x_col = x.reshape(-1, 1)
        arm["A"] += np.dot(x_col, x_col.T)
        arm["b"] += reward * x
        arm["count"] = arm.get("count", 0) + 1

        # Also update global model with a damped weight for global generalization
        global_arm = self.arms["__global__"]
        global_arm["A"] += 0.25 * np.dot(x_col, x_col.T)
        global_arm["b"] += 0.25 * reward * x
        global_arm["count"] = global_arm.get("count", 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model state to JSON-serializable dict"""
        serializable_arms = {}
        for arm_id, arm in self.arms.items():
            serializable_arms[arm_id] = {
                "A": arm["A"].tolist(),
                "b": arm["b"].tolist(),
                "count": int(arm.get("count", 0))
            }
        return {
            "d": self.d,
            "alpha": self.alpha,
            "l2_reg": self.l2_reg,
            "arms": serializable_arms
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinUCBAgent":
        agent = cls(d=data.get("d", FEATURE_DIMENSION), alpha=data.get("alpha", 0.8), l2_reg=data.get("l2_reg", 1.0))
        for arm_id, arm_data in data.get("arms", {}).items():
            agent.arms[arm_id] = {
                "A": np.array(arm_data["A"], dtype=np.float64),
                "b": np.array(arm_data["b"], dtype=np.float64),
                "count": int(arm_data.get("count", 0))
            }
        return agent

