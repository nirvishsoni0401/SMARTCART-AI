import json
from typing import Dict, Optional
from datetime import datetime
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

class ModelStore:
    """Central registry and persistent storage for all recommendation bandit policies."""

    def __init__(self, feature_dim: int = 36):
        self.feature_dim = feature_dim
        self.models: Dict[str, BaseRecommender] = {
            "linucb": LinUCBRecommender(feature_dim=feature_dim, alpha=settings.LINUCB_ALPHA, lambda_reg=settings.LINUCB_LAMBDA),
            "discounted_linucb": DiscountedLinUCBRecommender(feature_dim=feature_dim, alpha=settings.LINUCB_ALPHA, gamma=settings.DISCOUNT_GAMMA),
            "linucb_thompson": LinThompsonSamplingRecommender(feature_dim=feature_dim, v_variance=settings.TS_VARIANCE),
            "epsilon_greedy": EpsilonGreedyRecommender(feature_dim=feature_dim, epsilon=settings.EPSILON),
            "popularity": PopularityRecommender(feature_dim=feature_dim),
            "random": RandomRecommender(feature_dim=feature_dim)
        }
        self.load_all_from_db()

    def get_model(self, algorithm_name: Optional[str] = None) -> BaseRecommender:
        """Fetch model by name or fall back to global configured algorithm."""
        name = (algorithm_name or settings.GLOBAL_ALGORITHM).lower()
        if name not in self.models:
            name = "linucb"
        return self.models[name]

    def save_model_to_db(self, algorithm_name: str, db: Optional[Session] = None) -> None:
        """Persist algorithm state JSON blob to SQLite model_states table."""
        model = self.models.get(algorithm_name)
        if not model:
            return

        owns_session = False
        if db is None:
            db = SessionLocal()
            owns_session = True

        try:
            state_data = model.get_state()
            state_json = json.dumps(state_data)

            state_row = db.query(ModelState).filter(ModelState.algorithm == algorithm_name).first()
            if not state_row:
                state_row = ModelState(
                    algorithm=algorithm_name,
                    state_blob=state_json,
                    n_updates=model.n_updates,
                    updated_at=datetime.utcnow()
                )
                db.add(state_row)
            else:
                state_row.state_blob = state_json
                state_row.n_updates = model.n_updates
                state_row.updated_at = datetime.utcnow()

            db.commit()
        finally:
            if owns_session:
                db.close()

    def load_all_from_db(self) -> None:
        """Load persisted model weights from SQLite on startup."""
        db = SessionLocal()
        try:
            states = db.query(ModelState).all()
            for row in states:
                if row.algorithm in self.models:
                    try:
                        state_dict = json.loads(row.state_blob)
                        self.models[row.algorithm].load_state(state_dict)
                    except Exception as e:
                        print(f"[ModelStore] Warning: failed to load state for {row.algorithm}: {e}")
        finally:
            db.close()

    def reset_all(self, db: Optional[Session] = None) -> None:
        """Reset all bandit models to uninitialized prior states."""
        self.models = {
            "linucb": LinUCBRecommender(feature_dim=self.feature_dim, alpha=settings.LINUCB_ALPHA, lambda_reg=settings.LINUCB_LAMBDA),
            "discounted_linucb": DiscountedLinUCBRecommender(feature_dim=self.feature_dim, alpha=settings.LINUCB_ALPHA, gamma=settings.DISCOUNT_GAMMA),
            "linucb_thompson": LinThompsonSamplingRecommender(feature_dim=self.feature_dim, v_variance=settings.TS_VARIANCE),
            "epsilon_greedy": EpsilonGreedyRecommender(feature_dim=self.feature_dim, epsilon=settings.EPSILON),
            "popularity": PopularityRecommender(feature_dim=self.feature_dim),
            "random": RandomRecommender(feature_dim=self.feature_dim)
        }
        owns = False
        if db is None:
            db = SessionLocal()
            owns = True
        try:
            db.query(ModelState).delete()
            db.commit()
        finally:
            if owns:
                db.close()

# Global singleton
model_store = ModelStore()

