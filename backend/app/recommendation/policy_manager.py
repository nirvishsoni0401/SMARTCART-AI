import threading
import datetime
from typing import Dict, Any, Optional
import numpy as np
from app.recommendation.linucb import LinUCBAgent
from app.recommendation.model_storage import load_bandit_model, save_bandit_model, ACTIVE_MODEL_VERSION
from app.recommendation.reward_engine import RewardEngine


class PolicyManager:
    _instance: Optional["PolicyManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PolicyManager, cls).__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self):
        self.lock = threading.Lock()
        self.agent: LinUCBAgent = load_bandit_model(ACTIVE_MODEL_VERSION)
        self.reward_engine = RewardEngine()
        self.total_updates = 0
        self.cumulative_reward = 0.0
        self.reward_history = []
        self.last_saved_time = datetime.datetime.now()

    def get_agent(self) -> LinUCBAgent:
        with self.lock:
            return self.agent

    def update_policy(self, arm_id: str, feature_vector: np.ndarray, reward: float, event_type: str = "custom") -> Dict[str, Any]:
        """
        Thread-safe atomic update to LinUCB parameters.
        """
        with self.lock:
            self.agent.update_arm(arm_id, feature_vector, reward)
            self.total_updates += 1
            self.cumulative_reward += reward
            
            entry = {
                "step": self.total_updates,
                "arm_id": arm_id,
                "reward": reward,
                "event_type": event_type,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            self.reward_history.append(entry)
            if len(self.reward_history) > 500:
                self.reward_history.pop(0)

            # Periodic persistence every 5 updates or 60 seconds
            now = datetime.datetime.now()
            if self.total_updates % 5 == 0 or (now - self.last_saved_time).total_seconds() > 60:
                try:
                    save_bandit_model(
                        self.agent,
                        version=ACTIVE_MODEL_VERSION,
                        metadata={
                            "total_updates": self.total_updates,
                            "cumulative_reward": self.cumulative_reward,
                            "avg_reward": (self.cumulative_reward / self.total_updates) if self.total_updates > 0 else 0.0
                        }
                    )
                    self.last_saved_time = now
                except Exception as e:
                    print(f"Warning saving bandit model: {e}")

            return {
                "step": self.total_updates,
                "total_updates": self.total_updates,
                "cumulative_reward": round(self.cumulative_reward, 2),
                "avg_reward": round(self.cumulative_reward / max(1, self.total_updates), 3)
            }

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            avg_reward = (self.cumulative_reward / self.total_updates) if self.total_updates > 0 else 0.0
            return {
                "algorithm": "Contextual LinUCB",
                "version": ACTIVE_MODEL_VERSION,
                "alpha": self.agent.alpha,
                "dimension": self.agent.d,
                "total_updates": self.total_updates,
                "cumulative_reward": round(self.cumulative_reward, 2),
                "average_reward": round(avg_reward, 3),
                "arms_count": len(self.agent.arms),
                "recent_history": self.reward_history[-20:]
            }


policy_manager = PolicyManager()

