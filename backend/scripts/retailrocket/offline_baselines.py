"""
SmartCart AI — Phase 10B Offline Baselines
Defines baseline ranking policies for offline evaluation:
  1. PopularityRecommender: Static ranking by catalog popularity prior / score.
  2. DepartmentHeuristicRecommender: Filter to target's department, rank by popularity.
  3. RecentlyViewedRecommender: Rank visitor's previously engaged products by recency and frequency,
     filling remainder by popularity prior.
  4. DepartmentRecencyRecommender: Rank products from the visitor's most recently viewed department
     by popularity, filling remainder by popularity.
  5. SameBrandPopularityRecommender: Rank products from the visitor's most recently engaged brand
     by popularity, filling remainder by popularity.
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np


class PopularityBaseline:
    def __init__(self, catalog: dict, idx_to_pid: list):
        self.catalog = catalog
        self.idx_to_pid = idx_to_pid
        self.n_catalog = len(idx_to_pid)
        self.pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}
        self.priors = np.array([catalog[p].get("popularity_prior", 0.0) for p in idx_to_pid], dtype=np.float64)

    def score(self, context=None, history=None) -> np.ndarray:
        return self.priors.copy()


class DepartmentHeuristicBaseline:
    def __init__(self, catalog: dict, idx_to_pid: list, dept_pids: dict):
        self.catalog = catalog
        self.idx_to_pid = idx_to_pid
        self.dept_pids = dept_pids
        self.pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}
        self.priors = np.array([catalog[p].get("popularity_prior", 0.0) for p in idx_to_pid], dtype=np.float64)

    def score(self, history: Optional[List[Any]] = None) -> np.ndarray:
        """
        True history-dependent department heuristic:
        Boosts products matching the category of the most recently viewed item in history (strictly tau < t).
        If history is empty, falls back to catalog popularity priors.
        """
        scores = self.priors.copy()
        if not history:
            return scores
        recent_item = history[-1]
        prod = recent_item[0] if isinstance(recent_item, (tuple, list)) else getattr(recent_item, "product", recent_item)
        cat = prod["category"] if isinstance(prod, dict) else getattr(prod, "category", "")
        if cat and cat in self.dept_pids:
            for p in self.dept_pids[cat]:
                scores[self.pid_to_idx[p]] += 100.0
        return scores

    def score_for_target_dept(self, target_dept: str) -> np.ndarray:
        """DEPRECATED: Leaks future target category. Preserved strictly for reproduction of Phase 10C bug."""
        scores = self.priors.copy()
        if target_dept in self.dept_pids:
            for p in self.dept_pids[target_dept]:
                scores[self.pid_to_idx[p]] += 100.0
        return scores


class RecentlyViewedBaseline:
    def __init__(self, catalog: dict, idx_to_pid: list):
        self.catalog = catalog
        self.idx_to_pid = idx_to_pid
        self.pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}
        self.priors = np.array([catalog[p].get("popularity_prior", 0.0) for p in idx_to_pid], dtype=np.float64)

    def score(self, history: List[Any]) -> np.ndarray:
        scores = self.priors.copy()
        if not history:
            return scores

        # History is ordered chronologically [(prod, ev_type, ts, reward), ...]
        seen = {}
        for rank_pos, item in enumerate(reversed(history)):
            prod = item[0] if isinstance(item, (tuple, list)) else getattr(item, "product", item)
            pid = prod["id"] if isinstance(prod, dict) else getattr(prod, "id", None)
            if pid is not None and pid in self.pid_to_idx:
                if pid not in seen:
                    # Give high boost based on recency order + count
                    seen[pid] = len(history) - rank_pos

        for pid, boost in seen.items():
            scores[self.pid_to_idx[pid]] += 10000.0 + boost * 1000.0
        return scores


class DepartmentRecencyBaseline:
    def __init__(self, catalog: dict, idx_to_pid: list, dept_pids: dict):
        self.catalog = catalog
        self.idx_to_pid = idx_to_pid
        self.dept_pids = dept_pids
        self.pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}
        self.priors = np.array([catalog[p].get("popularity_prior", 0.0) for p in idx_to_pid], dtype=np.float64)

    def score(self, history: List[Any]) -> np.ndarray:
        scores = self.priors.copy()
        if not history:
            return scores

        # Find most recent product's department
        recent_item = history[-1]
        prod = recent_item[0] if isinstance(recent_item, (tuple, list)) else getattr(recent_item, "product", recent_item)
        cat = prod["category"] if isinstance(prod, dict) else getattr(prod, "category", "")

        if cat and cat in self.dept_pids:
            for p in self.dept_pids[cat]:
                scores[self.pid_to_idx[p]] += 100.0
        return scores


class SameBrandPopularityBaseline:
    def __init__(self, catalog: dict, idx_to_pid: list):
        self.catalog = catalog
        self.idx_to_pid = idx_to_pid
        self.pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}
        self.priors = np.array([catalog[p].get("popularity_prior", 0.0) for p in idx_to_pid], dtype=np.float64)

    def score(self, history: List[Any]) -> np.ndarray:
        scores = self.priors.copy()
        if not history:
            return scores

        # Find most recent product with non-empty brand
        recent_brand = ""
        for item in reversed(history):
            prod = item[0] if isinstance(item, (tuple, list)) else getattr(item, "product", item)
            brand = prod["brand"] if isinstance(prod, dict) else getattr(prod, "brand", "")
            if brand:
                recent_brand = brand
                break

        if recent_brand:
            for p in self.idx_to_pid:
                b = self.catalog[p]["brand"] if isinstance(self.catalog[p], dict) else getattr(self.catalog[p], "brand", "")
                if b == recent_brand:
                    scores[self.pid_to_idx[p]] += 100.0
        return scores

