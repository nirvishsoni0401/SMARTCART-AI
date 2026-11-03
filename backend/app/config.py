import os
from pathlib import Path
from pydantic import BaseModel, Field

# Locate smartcart.db absolutely inside the backend directory
BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BACKEND_DIR / "smartcart.db"
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"

class RewardConfig(BaseModel):
    purchase: float = 10.0
    cart_add: float = 5.0
    wishlist: float = 3.0
    click: float = 2.0
    skip: float = 0.0
    cart_remove: float = -2.0
    dislike: float = -5.0
    return_item: float = -8.0

class Settings(BaseModel):
    PROJECT_NAME: str = "SmartCart AI"
    DATABASE_URL: str = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
    JWT_SECRET: str = os.getenv("JWT_SECRET", "smartcart_ai_jwt_super_secret_key_2026")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # LinUCB default parameters
    LINUCB_ALPHA: float = 0.5
    LINUCB_LAMBDA: float = 1.0
    DISCOUNT_GAMMA: float = 0.95
    EPSILON: float = 0.1
    TS_VARIANCE: float = 0.25
    
    # Recommendation pipeline defaults
    CANDIDATE_POOL_SIZE: int = 50
    DEFAULT_RECOMMENDATION_COUNT: int = 10
    MAX_PER_CATEGORY_TOP10: int = 3
    MIN_CANDIDATES_PER_CATEGORY: int = 2
    IMPRESSION_TIMEOUT_MINUTES: int = 5
    USD_TO_INR_RATE: float = float(os.getenv("USD_TO_INR_RATE", "85.0"))
    
    # Global algorithm override ("linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy", "popularity", "random", or "ab_test")
    GLOBAL_ALGORITHM: str = "linucb"
    
    # Feature specification metadata (binding for bandit models)
    FEATURE_DIMENSION: int = 36
    FEATURE_SPEC_HASH: str = "75f60ed70b2a"
    
    # Reward mapping
    rewards: RewardConfig = RewardConfig()

settings = Settings()
