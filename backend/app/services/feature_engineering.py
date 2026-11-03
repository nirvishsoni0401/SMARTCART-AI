"""
SmartCart AI — Feature Engineering Service
Constructs the joint contextual bandit feature vector phi(user_context, product) in R^36.

Vector Dimension Layout:
---------------------------------------------------------------------------------------------------
Index Range | Feature Description                       | Bounds / Type
---------------------------------------------------------------------------------------------------
0 - 5       | User category affinity (6 categories)     | [0, 1] continuous, time-decayed
6           | User preferred-brand affinity             | [0, 1] continuous
7           | User average historical price level       | [0, 1] log-normalized
8           | Session budget cap                        | [0, 1] log-normalized
9           | Price fit score                           | [0, 1] continuous similarity to budget
10          | In-budget indicator                       | {0, 1} binary
11          | Category search/filter match              | {0, 1} binary
12 - 17     | Product category one-hot (6 categories)   | {0, 1}^6 binary one-hot
18          | Product price                             | [0, 1] log-scaled
19          | Product rating                            | [0, 1] normalized (rating - 1) / 4
20          | Product discount percentage               | [0, 1] discount_pct / 100
21          | Product popularity score                  | [0, 1] popularity_score / 100
22 - 29     | Brand hash bucket (8 bins)                | {0, 1}^8 binary one-hot
30          | New product flag                          | {0, 1} binary
31          | User activity level                       | [0, 1] log of interaction count
32          | Cart intent match flag                    | [0, 1] fraction of cart in same cat
33          | High-rating boost flag (>= 4.5)           | {0, 1} binary
34          | High-discount boost flag (>= 30%)         | {0, 1} binary
35          | Intercept / bias term                     | Constant 1.0
---------------------------------------------------------------------------------------------------
Total dimensions: 36
"""

import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
from sqlalchemy.orm import Session
from app.models import Product, InteractionEvent, CartItem, OrderItem

FEATURE_DIMENSION: int = 36

CANONICAL_FEATURE_SPEC: str = (
    "dim=36:"
    "0-5=user_category_affinity(6):"
    "6=user_brand_affinity:"
    "7=user_avg_price_log:"
    "8=session_budget_log:"
    "9=price_fit_score:"
    "10=in_budget_flag:"
    "11=category_search_match:"
    "12-17=product_category_onehot(6):"
    "18=product_price_log:"
    "19=product_rating_norm:"
    "20=product_discount_pct:"
    "21=product_popularity_score:"
    "22-29=brand_hash_bucket(8):"
    "30=new_product_flag:"
    "31=user_activity_level:"
    "32=cart_intent_match:"
    "33=high_rating_flag:"
    "34=high_discount_flag:"
    "35=intercept_bias"
)

FEATURE_SPEC_HASH: str = hashlib.sha256(CANONICAL_FEATURE_SPEC.encode("utf-8")).hexdigest()[:12]

CATEGORIES = [
    "Electronics",
    "Fashion",
    "Home & Kitchen",
    "Beauty",
    "Sports",
    "Books"
]
CATEGORY_INDEX = {cat: i for i, cat in enumerate(CATEGORIES)}
MAX_PRICE_SCALE = math.log(1.0 + 200000.0)

class UserContext:
    def __init__(
        self,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        session_category: Optional[str] = None,
        session_query: Optional[str] = None,
        session_budget: Optional[float] = None,
        category_affinities: Optional[Dict[str, float]] = None,
        brand_affinities: Optional[Dict[str, float]] = None,
        avg_price: Optional[float] = None,
        total_interactions: int = 0,
        cart_categories: Optional[Dict[str, int]] = None
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.session_category = session_category
        self.session_query = session_query
        self.session_budget = session_budget
        self.category_affinities = category_affinities or {cat: 0.0 for cat in CATEGORIES}
        self.brand_affinities = brand_affinities or {}
        self.avg_price = avg_price
        self.total_interactions = total_interactions
        self.cart_categories = cart_categories or {}
        self.is_cold_start = (total_interactions == 0)

def build_user_context(
    db: Session,
    user_id: Optional[int] = None,
    session_id: Optional[str] = None,
    session_category: Optional[str] = None,
    session_query: Optional[str] = None,
    session_budget: Optional[float] = None
) -> UserContext:
    """Extract and aggregate user interaction history into a UserContext."""
    category_affinities = {cat: 0.0 for cat in CATEGORIES}
    brand_affinities: Dict[str, float] = {}
    purchased_prices: List[float] = []
    total_interactions = 0
    cart_categories: Dict[str, int] = {}

    # Query events for user or session
    events_query = db.query(InteractionEvent)
    if user_id:
        events_query = events_query.filter(InteractionEvent.user_id == user_id)
    elif session_id:
        events_query = events_query.filter(InteractionEvent.session_id == session_id)
    else:
        events_query = None

    if events_query:
        events = events_query.order_by(InteractionEvent.created_at.desc()).limit(100).all()
        now = datetime.now(timezone.utc)

        for ev in events:
            total_interactions += 1
            # Exponential decay: half-life ~ 3 days (decay_rate ~ 0.23 per day)
            age_days = (now - ev.created_at.replace(tzinfo=timezone.utc)).total_seconds() / 86400.0
            decay = math.exp(-0.23 * max(0.0, age_days))
            weight = ev.reward * decay

            if ev.product:
                cat = ev.product.category
                if cat in category_affinities:
                    category_affinities[cat] += weight

                brand = ev.product.brand
                if brand:
                    brand_affinities[brand] = brand_affinities.get(brand, 0.0) + weight

                if ev.event_type in ("purchase", "cart_add"):
                    purchased_prices.append(ev.product.price)

    # Normalize category affinities into [0, 1]
    max_cat = max(category_affinities.values()) if category_affinities else 0.0
    if max_cat > 0:
        for cat in category_affinities:
            category_affinities[cat] = max(0.0, min(1.0, category_affinities[cat] / max_cat))

    # Normalize brand affinities
    max_brand = max(brand_affinities.values()) if brand_affinities else 0.0
    if max_brand > 0:
        for b in brand_affinities:
            brand_affinities[b] = max(0.0, min(1.0, brand_affinities[b] / max_brand))

    # Cart categories
    cart_items = []
    if user_id:
        cart_items = db.query(CartItem).filter(CartItem.user_id == user_id).all()
    elif session_id:
        cart_items = db.query(CartItem).filter(CartItem.session_id == session_id).all()

    for item in cart_items:
        if item.product:
            c = item.product.category
            cart_categories[c] = cart_categories.get(c, 0) + item.quantity

    avg_price = float(np.mean(purchased_prices)) if purchased_prices else None

    return UserContext(
        user_id=user_id,
        session_id=session_id,
        session_category=session_category,
        session_query=session_query,
        session_budget=session_budget,
        category_affinities=category_affinities,
        brand_affinities=brand_affinities,
        avg_price=avg_price,
        total_interactions=total_interactions,
        cart_categories=cart_categories
    )

def extract_feature_vector(context: UserContext, product: Product) -> np.ndarray:
    """Generate the 36-dimensional joint feature vector phi(u, p)."""
    phi = np.zeros(36, dtype=np.float64)

    # 0 - 5: User Category Affinity
    for i, cat in enumerate(CATEGORIES):
        phi[i] = context.category_affinities.get(cat, 0.0)

    # 6: Brand Affinity
    phi[6] = context.brand_affinities.get(product.brand, 0.0)

    # 7: User Average Historical Price Level
    ref_price = context.avg_price if context.avg_price is not None else 1800.0
    phi[7] = math.log(1.0 + ref_price) / MAX_PRICE_SCALE

    # 8: Session Budget Cap
    budget = context.session_budget if context.session_budget is not None else ref_price
    phi[8] = math.log(1.0 + budget) / MAX_PRICE_SCALE

    # 9: Price Fit Score
    p_log = math.log(1.0 + product.price)
    target_log = math.log(1.0 + budget)
    price_diff = abs(p_log - target_log)
    phi[9] = max(0.0, 1.0 - (price_diff / 2.0))

    # 10: In-Budget Flag
    phi[10] = 1.0 if product.price <= budget else 0.0

    # 11: Category Search Match
    if context.session_category and context.session_category.lower() == product.category.lower():
        phi[11] = 1.0
    elif context.session_query and product.category.lower() in context.session_query.lower():
        phi[11] = 1.0
    else:
        phi[11] = 0.0

    # 12 - 17: Product Category One-Hot
    cat_idx = CATEGORY_INDEX.get(product.category)
    if cat_idx is not None:
        phi[12 + cat_idx] = 1.0

    # 18: Product Price (log-normalized)
    phi[18] = math.log(1.0 + product.price) / MAX_PRICE_SCALE

    # 19: Product Rating
    phi[19] = max(0.0, min(1.0, (product.rating - 1.0) / 4.0))

    # 20: Product Discount Percentage
    phi[20] = max(0.0, min(1.0, product.discount_pct / 100.0))

    # 21: Product Popularity Score
    phi[21] = max(0.0, min(1.0, product.popularity_score / 100.0))

    # 22 - 29: Brand Hash Buckets (8 bins)
    bucket = abs(hash(product.brand or "")) % 8
    phi[22 + bucket] = 1.0

    # 30: New Product Flag (simulate items created in catalog)
    phi[30] = 1.0 if getattr(product, "id", 0) > 550 else 0.0

    # 31: User Activity Level
    phi[31] = min(1.0, math.log(1.0 + context.total_interactions) / math.log(50.0))

    # 32: Cart Intent Match
    total_cart = sum(context.cart_categories.values())
    if total_cart > 0:
        cat_cart_count = context.cart_categories.get(product.category, 0)
        phi[32] = cat_cart_count / total_cart
    else:
        phi[32] = 0.0

    # 33: High-Rating Flag (>= 4.5)
    phi[33] = 1.0 if product.rating >= 4.5 else 0.0

    # 34: High-Discount Flag (>= 30%)
    phi[34] = 1.0 if product.discount_pct >= 30.0 else 0.0

    # 35: Intercept / Bias term
    phi[35] = 1.0

    return phi

