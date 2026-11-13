from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from app.models import Product, Order, OrderItem
from app.config import settings
from app.recommender.base import CandidateItem
from app.services.feature_engineering import UserContext, extract_feature_vector, CATEGORIES

def generate_candidates(
    db: Session,
    context: UserContext,
    candidate_pool_size: int = 50
) -> List[CandidateItem]:
    """
    Stage 1 Candidate Generation:
    1. Excludes out-of-stock items (stock > 0)
    2. Enforces active products only (is_active == True)
    3. Excludes recently purchased items by this user
    4. Respects category or query filters if specified in context
    5. Samples across categories using configurable MIN_CANDIDATES_PER_CATEGORY
    6. Returns candidate items packaged with computed 36-dim feature vectors phi.
    """
    # 1. Collect recently purchased product IDs to avoid duplicate recommendations
    excluded_ids = set()
    if context.user_id:
        recent_orders = (
            db.query(Order)
            .filter(Order.user_id == context.user_id)
            .order_by(Order.created_at.desc())
            .limit(3)
            .all()
        )
        for ord_obj in recent_orders:
            for item in ord_obj.items:
                excluded_ids.add(item.product_id)

    # 2. Build base candidate query: active inventory only
    query = db.query(Product).filter(
        Product.is_active == True,
        Product.stock > 0
    )
    if excluded_ids:
        query = query.filter(~Product.id.in_(list(excluded_ids)))

    # Apply category filter if active
    if context.session_category and context.session_category.lower() != "all":
        query = query.filter(Product.category.ilike(context.session_category))

    # Apply search query filter if active
    if context.session_query:
        clean_q = f"%{context.session_query.strip()}%"
        query = query.filter(
            or_(
                Product.name.ilike(clean_q),
                Product.brand.ilike(clean_q),
                Product.category.ilike(clean_q)
            )
        )

    # Apply session budget cap if specified
    if context.session_budget and context.session_budget > 0:
        query = query.filter(Product.price <= context.session_budget * 1.5)

    candidates: List[Product] = []

    # If no specific category is filtered, sample round-robin across all 6 categories
    # with scaled per-category minimum so the bandit can explore without rigid forcing
    if not context.session_category or context.session_category.lower() == "all":
        min_per_cat = getattr(settings, "MIN_CANDIDATES_PER_CATEGORY", 2)
        per_cat = max(min_per_cat, candidate_pool_size // len(CATEGORIES) + 1)
        for cat in CATEGORIES:
            cat_products = (
                query.filter(Product.category == cat)
                .order_by(desc(Product.popularity_score), desc(Product.rating), Product.id.asc())
                .limit(per_cat)
                .all()
            )
            candidates.extend(cat_products)

        # Shuffle slightly based on interaction count for reproducibility
        import random
        random.seed(42 + context.total_interactions)
        random.shuffle(candidates)
        candidates = candidates[:candidate_pool_size]
    else:
        candidates = (
            query.order_by(desc(Product.popularity_score), desc(Product.rating), Product.id.asc())
            .limit(candidate_pool_size)
            .all()
        )

    # Package as CandidateItems with feature vectors
    candidate_items: List[CandidateItem] = []
    for prod in candidates:
        phi = extract_feature_vector(context, prod)
        candidate_items.append(CandidateItem(product_id=prod.id, product_data=prod, feature_vector=phi))

    return candidate_items
