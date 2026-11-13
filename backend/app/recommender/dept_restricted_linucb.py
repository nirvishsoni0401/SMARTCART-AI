from typing import List, Dict, Any, Optional
import numpy as np
from app.recommender.base import BaseRecommender, CandidateItem, ScoredCandidate
from app.recommender.linucb import LinUCBRecommender
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.department_heuristic import DepartmentHeuristicRecommender

class DeptRestrictedLinUCBRecommender(BaseRecommender):
    """
    Two-Stage Hybrid: Department-Restricted LinUCB.
      Stage 1 (Candidate Generation):
        Identifies target department c* from user history / affinity.
        Restricts primary candidates to c*.
      Stage 2 (Bandit Ranking):
        Ranks department candidates using LinUCB (or Discounted LinUCB).
      Exploration Quota:
        If m_explore > 0, reserves m_explore slots in the slate for cross-department
        exploration (highest bandit score items outside c*), while top_k - m_explore slots
        come strictly from c*.
      Cold-Start Fallback:
        Brand-new users with no history fall back to Department Heuristic, then Popularity.
    """

    def __init__(
        self,
        feature_dim: int = 36,
        alpha: float = 0.5,
        lambda_reg: float = 1.0,
        m_explore: int = 2,
        discounted: bool = False,
        gamma: float = 0.9995,
        name: Optional[str] = None
    ):
        if name is None:
            if m_explore > 0:
                name = f"dept_restricted_linucb_m{m_explore}" if m_explore != 2 else "dept_restricted_linucb"
            else:
                name = "dept_restricted_linucb_m0"

        super().__init__(name=name, feature_dim=feature_dim)
        self.m_explore = m_explore
        self.discounted = discounted

        if discounted:
            self.bandit = DiscountedLinUCBRecommender(
                feature_dim=feature_dim,
                alpha=alpha,
                lambda_reg=lambda_reg,
                gamma=gamma
            )
        else:
            self.bandit = LinUCBRecommender(
                feature_dim=feature_dim,
                alpha=alpha,
                lambda_reg=lambda_reg
            )

        self.heuristic = DepartmentHeuristicRecommender(feature_dim=feature_dim)

    def select(
        self,
        candidates: List[CandidateItem],
        top_k: int = 10,
        context: Optional[Any] = None
    ) -> List[ScoredCandidate]:
        if not candidates:
            return []

        # Check for cold start or absent history: fall back to Department Heuristic / Popularity
        is_cold = False
        if context is None:
            is_cold = True
        elif getattr(context, "is_cold_start", False) or getattr(context, "total_interactions", 0) == 0:
            affs = getattr(context, "category_affinities", {})
            if not affs or not any(v > 0 for v in affs.values()):
                is_cold = True

        if is_cold:
            return self.heuristic.select(candidates, top_k=top_k, context=context)

        # Stage 1: Determine target department from context
        target_cat = None
        sess_cat = getattr(context, "session_category", None)
        if sess_cat and str(sess_cat).lower() != "all":
            target_cat = str(sess_cat)
        else:
            affs = getattr(context, "category_affinities", {})
            if affs and any(v > 0 for v in affs.values()):
                target_cat = max(affs.items(), key=lambda x: x[1])[0]

        if not target_cat:
            return self.heuristic.select(candidates, top_k=top_k, context=context)

        # Partition candidates into target department and others
        dept_cands: List[CandidateItem] = []
        other_cands: List[CandidateItem] = []

        for cand in candidates:
            p = cand.product_data
            p_cat = getattr(p, "category", "") if not isinstance(p, dict) else p.get("category", "")
            if p_cat == target_cat:
                dept_cands.append(cand)
            else:
                other_cands.append(cand)

        m = min(self.m_explore, top_k)
        k_dept = top_k - m

        # Stage 2: Bandit ranking
        slate_dept: List[ScoredCandidate] = []
        if dept_cands:
            slate_dept = self.bandit.select(dept_cands, top_k=k_dept)

        slate_expl: List[ScoredCandidate] = []
        needed_expl = top_k - len(slate_dept)
        if needed_expl > 0 and other_cands:
            raw_expl = self.bandit.select(other_cands, top_k=needed_expl)
            for sc in raw_expl:
                sc.was_exploration = True
                slate_expl.append(sc)

        combined = slate_dept + slate_expl

        # If still short of top_k (e.g. small candidate set), fill from remaining
        if len(combined) < top_k:
            chosen_ids = {s.candidate.product_id for s in combined}
            remaining = [c for c in candidates if c.product_id not in chosen_ids]
            if remaining:
                extra = self.bandit.select(remaining, top_k=top_k - len(combined))
                for sc in extra:
                    sc.was_exploration = True
                    combined.append(sc)

        return combined[:top_k]

    def update(self, feature_vector: np.ndarray, reward: float) -> None:
        self.bandit.update(feature_vector, reward)
        self.n_updates = self.bandit.n_updates

    def correct_reward(self, feature_vector: np.ndarray, delta_reward: float) -> None:
        self.bandit.correct_reward(feature_vector, delta_reward)

    def get_theta(self) -> np.ndarray:
        return self.bandit.get_theta()

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_dim": self.feature_dim,
            "m_explore": self.m_explore,
            "discounted": self.discounted,
            "n_updates": self.n_updates,
            "bandit": self.bandit.get_state()
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.m_explore = int(state.get("m_explore", self.m_explore))
        if "bandit" in state:
            self.bandit.load_state(state["bandit"])
            self.n_updates = self.bandit.n_updates
        if "n_updates" in state:
            self.n_updates = int(state["n_updates"])
            self.bandit.n_updates = self.n_updates
