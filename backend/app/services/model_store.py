import json
import threading
from typing import Dict, Optional, List
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.config import settings
from app.database import SessionLocal
from app.models import ModelState
from app.recommender.base import BaseRecommender
from app.recommender.linucb import LinUCBRecommender
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.linucb_thompson import LinThompsonSamplingRecommender
from app.recommender.epsilon_greedy import EpsilonGreedyRecommender
from app.recommender.popularity import PopularityRecommender
from app.recommender.random_policy import RandomRecommender
from app.recommender.department_heuristic import DepartmentHeuristicRecommender
from app.recommender.dept_restricted_linucb import DeptRestrictedLinUCBRecommender


def create_fresh_models(feature_dim: int) -> Dict[str, BaseRecommender]:
    return {
        "linucb": LinUCBRecommender(feature_dim=feature_dim, alpha=settings.LINUCB_ALPHA, lambda_reg=settings.LINUCB_LAMBDA),
        "discounted_linucb": DiscountedLinUCBRecommender(feature_dim=feature_dim, alpha=settings.LINUCB_ALPHA, gamma=settings.DISCOUNT_GAMMA),
        "linucb_thompson": LinThompsonSamplingRecommender(feature_dim=feature_dim, v_variance=settings.TS_VARIANCE),
        "epsilon_greedy": EpsilonGreedyRecommender(feature_dim=feature_dim, epsilon=settings.EPSILON),
        "popularity": PopularityRecommender(feature_dim=feature_dim),
        "random": RandomRecommender(feature_dim=feature_dim),
        "department_heuristic": DepartmentHeuristicRecommender(feature_dim=feature_dim),
        "dept_restricted_linucb": DeptRestrictedLinUCBRecommender(feature_dim=feature_dim, alpha=settings.LINUCB_ALPHA, lambda_reg=settings.LINUCB_LAMBDA, m_explore=2, name="dept_restricted_linucb"),
        "dept_restricted_linucb_m0": DeptRestrictedLinUCBRecommender(feature_dim=feature_dim, alpha=settings.LINUCB_ALPHA, lambda_reg=settings.LINUCB_LAMBDA, m_explore=0, name="dept_restricted_linucb_m0"),
        "dept_restricted_linucb_m2": DeptRestrictedLinUCBRecommender(feature_dim=feature_dim, alpha=settings.LINUCB_ALPHA, lambda_reg=settings.LINUCB_LAMBDA, m_explore=2, name="dept_restricted_linucb_m2"),
    }


class ModelStore:
    """
    Central registry and persistent storage for all recommendation bandit policies.
    Thread-safe concurrency protection via RLock.
    Supports isolated sandbox models (sim:*), snapshot archiving, and rollback.
    """

    def __init__(self, feature_dim: int = 36):
        self.feature_dim = feature_dim
        self._lock = threading.RLock()
        self.models: Dict[str, BaseRecommender] = create_fresh_models(feature_dim)
        self.sandbox_models: Dict[str, BaseRecommender] = create_fresh_models(feature_dim)
        self.load_all_from_db()

    def get_model(self, algorithm_name: Optional[str] = None, sandbox: bool = False) -> BaseRecommender:
        """Fetch model by name. If sandbox=True, accesses isolated sandbox model."""
        with self._lock:
            name = (algorithm_name or settings.GLOBAL_ALGORITHM).lower()
            if name.startswith("sim:"):
                sandbox = True
                name = name[4:]

            registry = self.sandbox_models if sandbox else self.models
            if name not in registry:
                name = "linucb"
            return registry[name]

    def save_model_to_db(self, algorithm_name: str, sandbox: bool = False, db: Optional[Session] = None) -> None:
        """Persist algorithm state JSON blob to SQLite model_states table."""
        with self._lock:
            name = algorithm_name.lower()
            if name.startswith("sim:"):
                sandbox = True
                base_name = name[4:]
            else:
                base_name = name

            registry = self.sandbox_models if sandbox else self.models
            model = registry.get(base_name)
            if not model:
                return

            db_key = f"sim:{base_name}" if sandbox else base_name

            owns_session = False
            if db is None:
                db = SessionLocal()
                owns_session = True

            try:
                state_data = model.get_state()
                state_json = json.dumps(state_data)

                state_row = db.query(ModelState).filter(ModelState.algorithm == db_key).first()
                now_utc = datetime.now(timezone.utc)
                if not state_row:
                    state_row = ModelState(
                        algorithm=db_key,
                        state_blob=state_json,
                        n_updates=model.n_updates,
                        updated_at=now_utc
                    )
                    db.add(state_row)
                else:
                    state_row.state_blob = state_json
                    state_row.n_updates = model.n_updates
                    state_row.updated_at = now_utc

                db.commit()
            finally:
                if owns_session:
                    db.close()

    def load_all_from_db(self) -> None:
        """Load persisted model weights from SQLite on startup."""
        with self._lock:
            db = SessionLocal()
            try:
                states = db.query(ModelState).all()
                for row in states:
                    key = row.algorithm
                    if key.startswith("sim:"):
                        base = key[4:]
                        if base in self.sandbox_models:
                            try:
                                state_dict = json.loads(row.state_blob)
                                self.sandbox_models[base].load_state(state_dict)
                            except Exception as e:
                                print(f"[ModelStore] Warning: failed to load sandbox state for {key}: {e}")
                    elif key in self.models:
                        try:
                            state_dict = json.loads(row.state_blob)
                            self.models[key].load_state(state_dict)
                        except Exception as e:
                            print(f"[ModelStore] Warning: failed to load state for {key}: {e}")
            finally:
                db.close()

    def reset_all(self, db: Optional[Session] = None, sandbox_only: bool = False) -> None:
        """Reset bandit models to uninitialized prior states (preserves snapshots)."""
        with self._lock:
            base_names = [
                "linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy", "popularity", "random",
                "department_heuristic", "dept_restricted_linucb", "dept_restricted_linucb_m0", "dept_restricted_linucb_m2"
            ]
            if sandbox_only:
                self.sandbox_models = create_fresh_models(self.feature_dim)
            else:
                self.models = create_fresh_models(self.feature_dim)
                self.sandbox_models = create_fresh_models(self.feature_dim)

            owns = False
            if db is None:
                db = SessionLocal()
                owns = True
            try:
                if sandbox_only:
                    db.query(ModelState).filter(ModelState.algorithm.like("sim:%")).delete(synchronize_session=False)
                else:
                    # Delete live models and sandbox models, but preserve snapshot:* rows
                    db.query(ModelState).filter(
                        (ModelState.algorithm.in_(base_names)) |
                        (ModelState.algorithm.like("sim:%"))
                    ).delete(synchronize_session=False)
                db.commit()
            finally:
                if owns:
                    db.close()

    def archive_snapshot(self, snapshot_name: str, db: Optional[Session] = None) -> int:
        """Archive current live model states as a versioned snapshot."""
        with self._lock:
            owns = False
            if db is None:
                db = SessionLocal()
                owns = True
            count = 0
            try:
                now_utc = datetime.now(timezone.utc)
                for name, model in self.models.items():
                    snap_key = f"snapshot:{snapshot_name}:{name}"
                    state_json = json.dumps(model.get_state())
                    row = db.query(ModelState).filter(ModelState.algorithm == snap_key).first()
                    if not row:
                        row = ModelState(
                            algorithm=snap_key,
                            state_blob=state_json,
                            n_updates=model.n_updates,
                            updated_at=now_utc
                        )
                        db.add(row)
                    else:
                        row.state_blob = state_json
                        row.n_updates = model.n_updates
                        row.updated_at = now_utc
                    count += 1
                db.commit()
            finally:
                if owns:
                    db.close()
            return count

    def rollback_snapshot(self, snapshot_name: str, db: Optional[Session] = None) -> bool:
        """Restore model weights from a versioned snapshot."""
        with self._lock:
            owns = False
            if db is None:
                db = SessionLocal()
                owns = True
            try:
                clean_name = snapshot_name
                if clean_name.startswith("snapshot:"):
                    clean_name = clean_name[len("snapshot:"):]
                
                prefix = f"snapshot:{clean_name}:"
                rows = db.query(ModelState).filter(ModelState.algorithm.like(f"{prefix}%")).all()
                if not rows:
                    # Try single model exact match or partial match
                    single = db.query(ModelState).filter(
                        (ModelState.algorithm == snapshot_name) |
                        (ModelState.algorithm == f"snapshot:{clean_name}")
                    ).first()
                    if single:
                        rows = [single]
                        prefix = single.algorithm.rsplit(":", 1)[0] + ":"
                if not rows:
                    return False
                for r in rows:
                    algo = r.algorithm[len(prefix):]
                    if algo in self.models:
                        state_dict = json.loads(r.state_blob)
                        self.models[algo].load_state(state_dict)
                        # Also save to live key
                        self.save_model_to_db(algo, sandbox=False, db=db)
                return True
            finally:
                if owns:
                    db.close()



# Global singleton
model_store = ModelStore()
