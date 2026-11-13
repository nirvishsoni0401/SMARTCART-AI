"""
SmartCart AI — Calibrated Simulator v2: Recommendation Policies
Defines the 11 benchmark policies and the Production Pipeline wrapper:
  1. RandomPolicy
  2. PopularityPolicy
  3. DepartmentHeuristicPolicy (shared deterministic tie-breaking)
  4. RecentlyViewedPopularityPolicy
  5. EpsilonGreedyPolicy
  6. ColdLinUCBPolicy
  7. WarmLinUCBPolicy (initialized with Retailrocket sufficient_stats_m3.npz)
  8. ColdLinTSPolicy (Linear Thompson Sampling)
  9. WarmLinTSPolicy
  10. DiscountedLinUCBPolicy
  11. OraclePolicy

Execution Modes:
  (a) Bandit-Only (full catalog)
  (b) ProductionPipelineWrapper (Candidate Generation -> Policy Scoring -> Diversity Re-ranking)
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import math
from pathlib import Path
import numpy as np

from experiments.sim_v2.environment import EnvironmentV2, SlateFeedback, POSITION_BIAS
from experiments.sim_v2.user_model import UserInstance
from app.services.feature_engineering import UserContext, CATEGORIES


class BasePolicy:
    """Base class for all benchmark policies."""

    def __init__(self, name: str):
        self.name = name

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        raise NotImplementedError

    def update(self, feedback: SlateFeedback) -> None:
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        pass


class RandomPolicy(BasePolicy):
    """Uniformly random selection of k items from available products without replacement."""

    def __init__(self, seed: int = 42):
        super().__init__("Random")
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        k_actual = min(k, len(available_pids))
        chosen = self.rng.choice(available_pids, size=k_actual, replace=False)
        return list(chosen)


class PopularityPolicy(BasePolicy):
    """Ranks items by popularity score descending, breaking ties by product ID ascending."""

    def __init__(self):
        super().__init__("Popularity")

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        # Sort by (-popularity_score, id)
        sorted_pids = sorted(
            available_pids,
            key=lambda pid: (-env.products[env.pid_to_index[pid]]["popularity_score"], pid)
        )
        return sorted_pids[:k]


class DepartmentHeuristicPolicy(BasePolicy):
    """
    Ranks items based on the visitor's most recently viewed department (tau < t),
    falling back to user's highest affinity department, with deterministic tie-breaking.
    If enforce_diversity is True, enforces that at least 2 distinct categories are represented in the top-k slate.
    """

    def __init__(self, enforce_diversity: bool = False, name: Optional[str] = None):
        policy_name = name or ("Department Heuristic (With Diversity)" if enforce_diversity else "Department Heuristic")
        super().__init__(policy_name)
        self.enforce_diversity = enforce_diversity

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        # Determine target department from recent history if available
        target_cat = None
        if user.history:
            last_pid = user.history[-1][0]
            if last_pid in env.pid_to_index:
                target_cat = env.products[env.pid_to_index[last_pid]]["category"]

        # If no history, pick highest affinity category from user context
        if not target_cat:
            target_cat = max(user_context.category_affinities.items(), key=lambda x: x[1])[0]

        # Score items: +100 boost for matching department, plus popularity score
        scored_items = []
        for pid in available_pids:
            p = env.products[env.pid_to_index[pid]]
            match_boost = 100.0 if p["category"] == target_cat else 0.0
            score = match_boost + p["popularity_score"]
            scored_items.append((score, p["popularity_score"], pid))

        # Sort descending by (score, pop, -pid)
        scored_items.sort(key=lambda x: (-x[0], -x[1], x[2]))
        slate = [x[2] for x in scored_items[:k]]

        if self.enforce_diversity and len(slate) == k:
            cats = {env.products[env.pid_to_index[pid]]["category"] for pid in slate}
            if len(cats) < 2:
                for _, _, pid in scored_items[k:]:
                    if env.products[env.pid_to_index[pid]]["category"] != target_cat:
                        slate[-1] = pid
                        break
        return slate


class RecentlyViewedPopularityPolicy(BasePolicy):
    """
    Puts visitor's recently viewed products first (up to k_recent=3),
    filling the remaining slots with Popularity.
    """

    def __init__(self, k_recent: int = 3):
        super().__init__("Recently Viewed + Popularity")
        self.k_recent = k_recent

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        avail_set = set(available_pids)
        # Unique recently viewed in reverse chronological order
        recent_pids = []
        seen = set()
        for item in reversed(user.history):
            pid = item[0]
            if pid in avail_set and pid not in seen:
                recent_pids.append(pid)
                seen.add(pid)
                if len(recent_pids) >= self.k_recent:
                    break

        slate = list(recent_pids)

        # Fill remaining slots with popularity
        if len(slate) < k:
            sorted_pop = sorted(
                available_pids,
                key=lambda pid: (-env.products[env.pid_to_index[pid]]["popularity_score"], pid)
            )
            for pid in sorted_pop:
                if pid not in seen:
                    slate.append(pid)
                    seen.add(pid)
                    if len(slate) == k:
                        break

        return slate[:k]


class EpsilonGreedyPolicy(BasePolicy):
    """Linear epsilon-greedy bandit with Sherman-Morrison rank-1 updates."""

    def __init__(self, epsilon: float = 0.10, lambda_reg: float = 1.0, feature_dim: int = 36, seed: int = 42):
        super().__init__("Epsilon-Greedy")
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.A_inv = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)
        self.b = np.zeros(self.feature_dim, dtype=np.float64)
        self.n_updates = 0

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        if self.rng.random() < self.epsilon:
            # Explore: uniform random
            return list(self.rng.choice(available_pids, size=min(k, len(available_pids)), replace=False))

        # Exploit: score with theta
        theta = self.theta
        avail_indices = np.array([env.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        if feature_matrix is None:
            feature_matrix = env.build_feature_matrix_for_user(user, user_context)

        avail_features = feature_matrix[avail_indices]
        scores = avail_features @ theta

        top_k_idx = np.argsort(-scores)[:k]
        return [available_pids[i] for i in top_k_idx]

    def update(self, feedback: SlateFeedback) -> None:
        # Update using displayed slate items and realized rewards
        reward_map = {ev["pid"]: ev["reward"] for ev in feedback.events}
        for pos, pid in enumerate(feedback.slate_pids):
            phi = feedback.feature_matrix[pos]
            r = reward_map.get(pid, 0.0)

            # Sherman-Morrison rank-1 update
            w = self.A_inv @ phi
            denom = 1.0 + float(np.dot(phi, w))
            if abs(denom) > 1e-12:
                self.A_inv -= np.outer(w, w) / denom
            self.b += r * phi
            self.n_updates += 1


class ColdLinUCBPolicy(BasePolicy):
    """Cold LinUCB with Sherman-Morrison rank-1 updates."""

    def __init__(self, alpha: float = 0.5, lambda_reg: float = 1.0, feature_dim: int = 36):
        super().__init__("Cold LinUCB")
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        self.feature_dim = feature_dim
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self.A = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)
        self.b = np.zeros(self.feature_dim, dtype=np.float64)
        self.n_updates = 0

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        theta = self.theta
        avail_indices = np.array([env.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        if feature_matrix is None:
            feature_matrix = env.build_feature_matrix_for_user(user, user_context)

        X = feature_matrix[avail_indices]  # (M, 36)
        expected_rewards = X @ theta

        # Confidence bounds: alpha * sqrt(diag(X A_inv X^T))
        # Vectorized variance: sum( (X @ A_inv) * X, axis=1 )
        variances = np.sum((X @ self.A_inv) * X, axis=1)
        variances = np.maximum(0.0, variances)
        ucb_bonuses = self.alpha * np.sqrt(variances)

        ucb_scores = expected_rewards + ucb_bonuses
        top_k_idx = np.argsort(-ucb_scores, kind="stable")[:k]
        return [available_pids[i] for i in top_k_idx]

    def update(self, feedback: SlateFeedback) -> None:
        reward_map = {ev["pid"]: ev["reward"] for ev in feedback.events}
        Phi = feedback.feature_matrix[:len(feedback.slate_pids)]
        r_vec = np.array([reward_map.get(pid, 0.0) for pid in feedback.slate_pids], dtype=np.float64)

        self.A += Phi.T @ Phi
        self.b += Phi.T @ r_vec
        self.n_updates += len(feedback.slate_pids)

        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.pinv(self.A)


class WarmLinUCBPolicy(BasePolicy):
    """
    Warm LinUCB initialized with historical Retailrocket sufficient statistics
    (representing an empirical misspecified prior).
    """

    def __init__(
        self,
        alpha: float = 0.5,
        feature_dim: int = 36,
        stats_path: Optional[str] = "data/historical/sufficient_stats_m3.npz",
        n_cap: int = 10000,
        lambda_reg: float = 1.0
    ):
        super().__init__("Warm LinUCB")
        self.alpha = float(alpha)
        self.feature_dim = feature_dim
        self.stats_path = stats_path
        self.n_cap = n_cap
        self.lambda_reg = lambda_reg
        self._load_stats()
        self.reset()

    def _load_stats(self):
        if self.stats_path and Path(self.stats_path).exists():
            data = np.load(self.stats_path)
            A_hist = np.array(data["A_hist"], dtype=np.float64)
            b_hist = np.array(data["b_hist"], dtype=np.float64)
            n_hist = int(data.get("n_hist", 1))
            s = min(1.0, float(self.n_cap) / max(1.0, float(n_hist)))
            self.A_init = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64) + s * A_hist
            self.A_init = (self.A_init + self.A_init.T) / 2.0
            self.b_init = s * b_hist
            self.A_inv_init = np.linalg.inv(self.A_init)
        else:
            self.A_init = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
            self.b_init = np.zeros(self.feature_dim, dtype=np.float64)
            self.A_inv_init = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> None:
        self.A = self.A_init.copy()
        self.b = self.b_init.copy()
        self.A_inv = self.A_inv_init.copy()
        self.n_updates = 0

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        theta = self.theta
        avail_indices = np.array([env.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        if feature_matrix is None:
            feature_matrix = env.build_feature_matrix_for_user(user, user_context)

        X = feature_matrix[avail_indices]
        expected_rewards = X @ theta
        variances = np.sum((X @ self.A_inv) * X, axis=1)
        variances = np.maximum(0.0, variances)
        ucb_bonuses = self.alpha * np.sqrt(variances)

        ucb_scores = expected_rewards + ucb_bonuses
        top_k_idx = np.argsort(-ucb_scores, kind="stable")[:k]
        return [available_pids[i] for i in top_k_idx]

    def update(self, feedback: SlateFeedback) -> None:
        reward_map = {ev["pid"]: ev["reward"] for ev in feedback.events}
        Phi = feedback.feature_matrix[:len(feedback.slate_pids)]
        r_vec = np.array([reward_map.get(pid, 0.0) for pid in feedback.slate_pids], dtype=np.float64)

        self.A += Phi.T @ Phi
        self.b += Phi.T @ r_vec
        self.n_updates += len(feedback.slate_pids)

        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.pinv(self.A)


class ColdLinTSPolicy(BasePolicy):
    """Cold Linear Thompson Sampling policy."""

    def __init__(self, v_variance: float = 0.25, lambda_reg: float = 1.0, feature_dim: int = 36, seed: int = 42):
        super().__init__("Cold LinTS")
        self.v_variance = float(v_variance)
        self.lambda_reg = float(lambda_reg)
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.A = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)
        self.b = np.zeros(self.feature_dim, dtype=np.float64)
        self.n_updates = 0

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def sample_theta(self) -> np.ndarray:
        mean_theta = self.theta
        cov = (self.v_variance ** 2) * self.A_inv
        cov = (cov + cov.T) / 2.0 + 1e-8 * np.eye(self.feature_dim)
        try:
            return self.rng.multivariate_normal(mean_theta, cov)
        except Exception:
            return mean_theta

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        theta_sample = self.sample_theta()
        avail_indices = np.array([env.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        if feature_matrix is None:
            feature_matrix = env.build_feature_matrix_for_user(user, user_context)

        X = feature_matrix[avail_indices]
        sampled_scores = X @ theta_sample
        top_k_idx = np.argsort(-sampled_scores)[:k]
        return [available_pids[i] for i in top_k_idx]

    def update(self, feedback: SlateFeedback) -> None:
        reward_map = {ev["pid"]: ev["reward"] for ev in feedback.events}
        for pos, pid in enumerate(feedback.slate_pids):
            phi = feedback.feature_matrix[pos]
            r = reward_map.get(pid, 0.0)

            self.A += np.outer(phi, phi)
            w = self.A_inv @ phi
            denom = 1.0 + float(np.dot(phi, w))
            if abs(denom) > 1e-12:
                self.A_inv -= np.outer(w, w) / denom
            self.b += r * phi
            self.n_updates += 1

            if self.n_updates % 1000 == 0:
                try:
                    self.A_inv = np.linalg.inv(self.A)
                except np.linalg.LinAlgError:
                    pass


class WarmLinTSPolicy(BasePolicy):
    """Warm Linear Thompson Sampling initialized with Retailrocket prior."""

    def __init__(
        self,
        v_variance: float = 0.25,
        feature_dim: int = 36,
        stats_path: Optional[str] = "data/historical/sufficient_stats_m3.npz",
        n_cap: int = 10000,
        lambda_reg: float = 1.0,
        seed: int = 42
    ):
        super().__init__("Warm LinTS")
        self.v_variance = float(v_variance)
        self.feature_dim = feature_dim
        self.stats_path = stats_path
        self.n_cap = n_cap
        self.lambda_reg = lambda_reg
        self.rng = np.random.default_rng(seed)
        self._load_stats()
        self.reset()

    def _load_stats(self):
        if self.stats_path and Path(self.stats_path).exists():
            data = np.load(self.stats_path)
            A_hist = np.array(data["A_hist"], dtype=np.float64)
            b_hist = np.array(data["b_hist"], dtype=np.float64)
            n_hist = int(data.get("n_hist", 1))
            s = min(1.0, float(self.n_cap) / max(1.0, float(n_hist)))
            self.A_init = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64) + s * A_hist
            self.A_init = (self.A_init + self.A_init.T) / 2.0
            self.b_init = s * b_hist
            self.A_inv_init = np.linalg.inv(self.A_init)
        else:
            self.A_init = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
            self.b_init = np.zeros(self.feature_dim, dtype=np.float64)
            self.A_inv_init = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.A = self.A_init.copy()
        self.b = self.b_init.copy()
        self.A_inv = self.A_inv_init.copy()
        self.n_updates = 0

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def sample_theta(self) -> np.ndarray:
        mean_theta = self.theta
        cov = (self.v_variance ** 2) * self.A_inv
        cov = (cov + cov.T) / 2.0 + 1e-8 * np.eye(self.feature_dim)
        try:
            return self.rng.multivariate_normal(mean_theta, cov)
        except Exception:
            return mean_theta

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        theta_sample = self.sample_theta()
        avail_indices = np.array([env.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        if feature_matrix is None:
            feature_matrix = env.build_feature_matrix_for_user(user, user_context)

        X = feature_matrix[avail_indices]
        sampled_scores = X @ theta_sample
        top_k_idx = np.argsort(-sampled_scores)[:k]
        return [available_pids[i] for i in top_k_idx]

    def update(self, feedback: SlateFeedback) -> None:
        reward_map = {ev["pid"]: ev["reward"] for ev in feedback.events}
        for pos, pid in enumerate(feedback.slate_pids):
            phi = feedback.feature_matrix[pos]
            r = reward_map.get(pid, 0.0)

            self.A += np.outer(phi, phi)
            w = self.A_inv @ phi
            denom = 1.0 + float(np.dot(phi, w))
            if abs(denom) > 1e-12:
                self.A_inv -= np.outer(w, w) / denom
            self.b += r * phi
            self.n_updates += 1

            if self.n_updates % 1000 == 0:
                try:
                    self.A_inv = np.linalg.inv(self.A)
                except np.linalg.LinAlgError:
                    pass


class DiscountedLinUCBPolicy(BasePolicy):
    """
    Discounted LinUCB with exponential decay gamma in (0, 1] for adapting to drift.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 0.9995, lambda_reg: float = 1.0, feature_dim: int = 36, name: Optional[str] = None):
        policy_name = name or f"Discounted LinUCB (gamma={gamma})"
        super().__init__(policy_name)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.lambda_reg = float(lambda_reg)
        self.feature_dim = feature_dim
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self.A = self.lambda_reg * np.eye(self.feature_dim, dtype=np.float64)
        self.A_inv = (1.0 / self.lambda_reg) * np.eye(self.feature_dim, dtype=np.float64)
        self.b = np.zeros(self.feature_dim, dtype=np.float64)
        self.n_updates = 0

    @property
    def theta(self) -> np.ndarray:
        return self.A_inv @ self.b

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        theta = self.theta
        avail_indices = np.array([env.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        if feature_matrix is None:
            feature_matrix = env.build_feature_matrix_for_user(user, user_context)

        X = feature_matrix[avail_indices]
        expected_rewards = X @ theta
        variances = np.sum((X @ self.A_inv) * X, axis=1)
        variances = np.maximum(0.0, variances)
        ucb_bonuses = self.alpha * np.sqrt(variances)

        ucb_scores = expected_rewards + ucb_bonuses
        top_k_idx = np.argsort(-ucb_scores, kind="stable")[:k]
        return [available_pids[i] for i in top_k_idx]

    def update(self, feedback: SlateFeedback) -> None:
        reward_map = {ev["pid"]: ev["reward"] for ev in feedback.events}
        Phi = feedback.feature_matrix[:len(feedback.slate_pids)]
        r_vec = np.array([reward_map.get(pid, 0.0) for pid in feedback.slate_pids], dtype=np.float64)

        if self.gamma >= 1.0:
            self.A += Phi.T @ Phi
            self.b += Phi.T @ r_vec
        else:
            self.A = self.gamma * self.A + (1.0 - self.gamma) * self.lambda_reg * np.eye(self.feature_dim) + (Phi.T @ Phi)
            self.b = self.gamma * self.b + (Phi.T @ r_vec)
        self.n_updates += len(feedback.slate_pids)

        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.pinv(self.A)


class OraclePolicy(BasePolicy):
    """
    Ground-truth Oracle policy: selects top-k items maximizing exact expected slate reward.
    Expected regret is strictly 0.0 at all rounds.
    """

    def __init__(self):
        super().__init__("Oracle")

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None,
        expected_values: Optional[np.ndarray] = None
    ) -> List[int]:
        oracle_slate, _ = env.get_oracle_slate(user, available_pids, round_idx, k=k, expected_values=expected_values)
        return oracle_slate


class ProductionPipelineWrapper(BasePolicy):
    """
    Simulates production multi-stage recommendation pipeline:
      Stage 1: Candidate Generation (retrieves top 50 candidates: 25 popularity + 25 category match)
      Stage 2: Policy Ranking (applies base policy to rank the 50 candidates)
      Stage 3: Diversity Re-ranking (enforces >= 2 distinct departments in top-10 slate)
    """

    def __init__(self, base_policy: BasePolicy, candidate_pool_size: int = 50, min_distinct_cats: int = 2):
        super().__init__(f"{base_policy.name} (Production Pipeline)")
        self.base_policy = base_policy
        self.candidate_pool_size = candidate_pool_size
        self.min_distinct_cats = min_distinct_cats

    def reset(self, seed: Optional[int] = None) -> None:
        self.base_policy.reset(seed)

    def _generate_candidates(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int]
    ) -> List[int]:
        is_all = (len(available_pids) == env.n_products)
        sorted_pop = env.sorted_pids_all if is_all else env.sorted_pids_initial
        pids_by_cat = env.pids_by_cat_all if is_all else env.pids_by_cat_initial

        pop_candidates = sorted_pop[:25]
        candidate_set = set(pop_candidates)
        candidates = list(pop_candidates)

        # 2. Top 25 by User Affinity
        aff_cats = sorted(user_context.category_affinities.items(), key=lambda x: -x[1])
        for cat, _ in aff_cats:
            for pid in pids_by_cat.get(cat, []):
                if pid not in candidate_set:
                    candidates.append(pid)
                    candidate_set.add(pid)
                    if len(candidates) >= self.candidate_pool_size:
                        break
            if len(candidates) >= self.candidate_pool_size:
                break

        # Fill remainder from popularity if needed
        if len(candidates) < self.candidate_pool_size:
            for pid in sorted_pop:
                if pid not in candidate_set:
                    candidates.append(pid)
                    candidate_set.add(pid)
                    if len(candidates) >= self.candidate_pool_size:
                        break

        return candidates[:self.candidate_pool_size]

    def _apply_diversity_reranking(
        self,
        env: EnvironmentV2,
        ranked_pids: List[int],
        candidate_pool: List[int],
        k: int = 10
    ) -> List[int]:
        slate = list(ranked_pids[:k])
        slate_cats = {env.products[env.pid_to_index[pid]]["category"] for pid in slate}

        if len(slate_cats) >= self.min_distinct_cats:
            return slate

        # Slate is homogeneous: find best candidate outside current category to swap into 10th slot
        existing_cat = next(iter(slate_cats))
        swap_candidate = None
        for pid in candidate_pool:
            if pid not in slate:
                p_cat = env.products[env.pid_to_index[pid]]["category"]
                if p_cat != existing_cat:
                    swap_candidate = pid
                    break

        if swap_candidate is not None:
            slate[-1] = swap_candidate
        else:
            # Fallback across all catalog categories
            for cat, pids in env.pids_by_cat_all.items():
                if cat != existing_cat and pids:
                    for p in pids:
                        if p not in slate:
                            slate[-1] = p
                            return slate

        return slate

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        # Stage 1: Candidate Generation
        candidates = self._generate_candidates(env, user, user_context, available_pids)

        # Stage 2: Policy Ranking
        ranked_slate = self.base_policy.select_slate(
            env=env,
            user=user,
            user_context=user_context,
            available_pids=candidates,
            round_idx=round_idx,
            k=len(candidates),
            feature_matrix=feature_matrix
        )

        # Stage 3: Diversity Re-ranking
        final_slate = self._apply_diversity_reranking(env, ranked_slate, candidates, k=k)
        return final_slate

    def update(self, feedback: SlateFeedback) -> None:
        self.base_policy.update(feedback)


class DeptPriorLinUCBPolicy(ColdLinUCBPolicy):
    """
    Department-Prior LinUCB:
    Initializes A and b with an inductive prior on category and brand affinity features,
    with prior strength N_prior = 100.0 tuned on tuning seeds.
    """

    def __init__(self, alpha: float = 0.5, lambda_reg: float = 1.0, prior_weight: float = 0.5, prior_strength: float = 100.0, feature_dim: int = 36):
        self.prior_weight = float(prior_weight)
        self.prior_strength = float(prior_strength)
        super().__init__(alpha=alpha, lambda_reg=lambda_reg, feature_dim=feature_dim)
        self.name = f"Dept-Prior LinUCB (w={prior_weight})"
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        super().reset(seed)
        if hasattr(self, "prior_weight") and hasattr(self, "prior_strength"):
            for c in range(6):
                self.A[c, c] += self.prior_strength
                self.b[c] += self.prior_strength * self.prior_weight
            self.A[6, 6] += self.prior_strength
            self.b[6] += self.prior_strength * (self.prior_weight * 0.5)
            try:
                self.A_inv = np.linalg.inv(self.A)
            except np.linalg.LinAlgError:
                self.A_inv = np.linalg.pinv(self.A)


class SlateMixingPolicy(BasePolicy):
    """
    Slate Mixing Hybrid:
    Presents top k slots from Department Heuristic, and fills remaining 10 - k slots from LinUCB,
    filtering out duplicate products.
    """

    def __init__(self, k_heuristic: int = 3, alpha: float = 0.5, base_bandit: Optional[BasePolicy] = None):
        super().__init__(f"Slate Mixing (k={k_heuristic} Dept + LinUCB)")
        self.k_heuristic = k_heuristic
        self.heuristic = DepartmentHeuristicPolicy()
        self.bandit = base_bandit if base_bandit is not None else ColdLinUCBPolicy(alpha=alpha)

    def reset(self, seed: Optional[int] = None) -> None:
        self.heuristic.reset(seed)
        self.bandit.reset(seed)

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        h_slate = self.heuristic.select_slate(
            env=env,
            user=user,
            user_context=user_context,
            available_pids=available_pids,
            round_idx=round_idx,
            k=self.k_heuristic,
            feature_matrix=feature_matrix
        )
        seen = set(h_slate)
        rem_pids = [p for p in available_pids if p not in seen]
        b_slate = self.bandit.select_slate(
            env=env,
            user=user,
            user_context=user_context,
            available_pids=rem_pids,
            round_idx=round_idx,
            k=k - len(h_slate),
            feature_matrix=feature_matrix
        )
        return h_slate + b_slate

    def update(self, feedback: SlateFeedback) -> None:
        self.heuristic.update(feedback)
        self.bandit.update(feedback)


class ResidualLinUCBPolicy(ColdLinUCBPolicy):
    """
    Residual LinUCB Hybrid:
    Scores items with a base Department Heuristic offset, while LinUCB learns residual
    personalization and exploration adjustments.
    Updates on the residual reward: r_residual = r - h_offset.
    """

    def __init__(self, alpha: float = 0.5, lambda_reg: float = 1.0, scale: float = 0.05, feature_dim: int = 36):
        super().__init__(alpha=alpha, lambda_reg=lambda_reg, feature_dim=feature_dim)
        self.name = "Residual LinUCB"
        self.scale = scale
        self.last_offsets: Dict[int, float] = {}

    def _get_heuristic_offsets(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int]
    ) -> np.ndarray:
        target_cat = None
        if user.history:
            last_pid = user.history[-1][0]
            if last_pid in env.pid_to_index:
                target_cat = env.products[env.pid_to_index[last_pid]]["category"]
        if not target_cat:
            target_cat = max(user_context.category_affinities.items(), key=lambda x: x[1])[0]

        offsets = np.zeros(len(available_pids), dtype=np.float64)
        for i, pid in enumerate(available_pids):
            p = env.products[env.pid_to_index[pid]]
            match_boost = 0.35 if p["category"] == target_cat else 0.0
            offsets[i] = match_boost + (p["popularity_score"] / 100.0) * 0.05
        return offsets

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        h_offsets = self._get_heuristic_offsets(env, user, user_context, available_pids)
        theta = self.theta
        avail_indices = np.array([env.pid_to_index[pid] for pid in available_pids], dtype=np.int32)
        if feature_matrix is None:
            feature_matrix = env.build_feature_matrix_for_user(user, user_context)

        X = feature_matrix[avail_indices]
        mean_rewards = X @ theta
        variances = np.sum((X @ self.A_inv) * X, axis=1)
        ucb_scores = mean_rewards + self.alpha * np.sqrt(np.maximum(1e-8, variances))
        combined_scores = ucb_scores + h_offsets

        top_k_idx = np.argsort(-combined_scores, kind="stable")[:k]
        chosen_pids = [available_pids[i] for i in top_k_idx]
        self.last_offsets = {available_pids[i]: float(h_offsets[i]) for i in top_k_idx}
        return chosen_pids

    def update(self, feedback: SlateFeedback) -> None:
        reward_map = {ev["pid"]: ev["reward"] for ev in feedback.events}
        Phi = feedback.feature_matrix[:len(feedback.slate_pids)]
        r_res = np.array([reward_map.get(pid, 0.0) - self.last_offsets.get(pid, 0.0) for pid in feedback.slate_pids], dtype=np.float64)

        self.A += Phi.T @ Phi
        self.b += Phi.T @ r_res
        self.n_updates += len(feedback.slate_pids)

        try:
            self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.pinv(self.A)


class DeptRestrictedLinUCBPolicy(BasePolicy):
    """
    Two-Stage Hybrid (Department-Restricted LinUCB):
      Stage 1 (Candidate Generation):
        Identifies the user's target department c* using the same deterministic rule
        as DepartmentHeuristicPolicy (most recently engaged category if available,
        otherwise highest affinity category). Retrieves candidates from c*.
      Stage 2 (Bandit Ranking):
        Ranks department candidates using a contextual bandit (Cold LinUCB with alpha*
        or Discounted LinUCB with gamma*).
      Exploration Quota:
        If m_explore > 0, reserves m_explore slots in the slate for cross-department
        exploration (highest bandit score items outside c*), while 10 - m_explore slots
        come strictly from c*.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        base_bandit: Optional[BasePolicy] = None,
        m_explore: int = 0,
        alpha: float = 0.5,
        gamma: float = 0.9995
    ):
        if base_bandit is not None:
            self.bandit = base_bandit
        else:
            self.bandit = ColdLinUCBPolicy(alpha=alpha)
        self.m_explore = m_explore

        if name:
            policy_name = name
        elif m_explore > 0:
            b_type = "Discounted" if isinstance(self.bandit, DiscountedLinUCBPolicy) else "Cold"
            policy_name = f"Dept-Restricted LinUCB + Expl ({b_type}, m={m_explore})"
        else:
            b_type = "Discounted" if isinstance(self.bandit, DiscountedLinUCBPolicy) else "Cold"
            policy_name = f"Dept-Restricted LinUCB ({b_type})"

        super().__init__(policy_name)

    def reset(self, seed: Optional[int] = None) -> None:
        self.bandit.reset(seed)

    def select_slate(
        self,
        env: EnvironmentV2,
        user: UserInstance,
        user_context: UserContext,
        available_pids: List[int],
        round_idx: int,
        k: int = 10,
        feature_matrix: Optional[np.ndarray] = None
    ) -> List[int]:
        # Stage 1: Department selection (exact Department Heuristic rule)
        target_cat = None
        if user.history:
            last_pid = user.history[-1][0]
            if last_pid in env.pid_to_index:
                target_cat = env.products[env.pid_to_index[last_pid]]["category"]
        if not target_cat:
            target_cat = max(user_context.category_affinities.items(), key=lambda x: x[1])[0]

        dept_pids = [pid for pid in available_pids if env.products[env.pid_to_index[pid]]["category"] == target_cat]
        other_pids = [pid for pid in available_pids if env.products[env.pid_to_index[pid]]["category"] != target_cat]

        m = min(self.m_explore, k)
        k_dept = k - m

        # Stage 2: Bandit ranking
        slate_dept = []
        if dept_pids:
            slate_dept = self.bandit.select_slate(
                env=env,
                user=user,
                user_context=user_context,
                available_pids=dept_pids,
                round_idx=round_idx,
                k=k_dept,
                feature_matrix=feature_matrix
            )

        slate_expl = []
        needed_expl = k - len(slate_dept)
        if needed_expl > 0 and other_pids:
            slate_expl = self.bandit.select_slate(
                env=env,
                user=user,
                user_context=user_context,
                available_pids=other_pids,
                round_idx=round_idx,
                k=needed_expl,
                feature_matrix=feature_matrix
            )

        return slate_dept + slate_expl

    def update(self, feedback: SlateFeedback) -> None:
        self.bandit.update(feedback)

