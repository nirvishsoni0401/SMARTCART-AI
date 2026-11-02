import math
import numpy as np
from typing import Dict, Any

FEATURE_DIMENSION = 8


def extract_features(product: Dict[str, Any], context: Dict[str, Any]) -> np.ndarray:
    r"""
    Construct a dense normalized feature vector x \in R^d representing the interaction
    between customer context and product attributes.
    Dimension d = 8:
    1. Bias / constant term (1.0)
    2. Normalized product price (log scaled in INR, 0 to 1)
    3. Normalized product rating (0 to 1)
    4. Stock availability factor (0 to 1)
    5. Category affinity match (0 to 1)
    6. Price proximity to customer budget (0 to 1)
    7. Dominant category match indicator (0 or 1)
    8. User engagement level indicator (log scaled)
    """
    # 1. Bias
    f_bias = 1.0

    # 2. Normalized price (INR prices scale from ₹100 up to ₹250,000)
    price = float(product.get("discounted_price") or product.get("price", 499.0))
    # log scale: log1p(price) / log1p(250000)
    f_price = min(1.0, math.log1p(price) / math.log1p(250000.0))

    # 3. Rating
    rating = float(product.get("rating", 3.5))
    f_rating = min(1.0, max(0.0, rating / 5.0))

    # 4. Stock factor
    stock = int(product.get("stock_quantity") or product.get("stock", 10))
    f_stock = min(1.0, stock / 100.0)

    # 5. Category affinity match
    product_cat = product.get("category", "").lower()
    affinities = context.get("category_affinities", {})
    raw_affinity = affinities.get(product_cat, 0)
    max_affinity = max(affinities.values()) if affinities and max(affinities.values()) > 0 else 1
    f_cat_affinity = min(1.0, raw_affinity / max_affinity)

    # 6. Price proximity to customer's average viewed price (INR)
    avg_price = context.get("average_viewed_price", 1500.0)
    diff = abs(price - avg_price)
    f_price_proximity = max(0.0, 1.0 - (diff / max(avg_price, price, 1.0)))

    # 7. Dominant category match
    dom_cat = context.get("dominant_category", "")
    if dom_cat:
        dom_cat = dom_cat.lower()
    f_dom_match = 1.0 if dom_cat and (product_cat == dom_cat or dom_cat in product_cat) else 0.0

    # 8. User engagement level
    total_interactions = context.get("total_interactions", 0)
    f_engagement = min(1.0, math.log1p(total_interactions) / math.log1p(50.0))

    feature_vec = np.array([
        f_bias,
        f_price,
        f_rating,
        f_stock,
        f_cat_affinity,
        f_price_proximity,
        f_dom_match,
        f_engagement
    ], dtype=np.float64)

    # Normalize vector to unit norm
    norm = np.linalg.norm(feature_vec)
    if norm > 0:
        return feature_vec / norm
    return feature_vec
