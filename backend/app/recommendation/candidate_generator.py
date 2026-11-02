from typing import List, Dict, Any
from app.database import products_collection


def generate_candidates(
    context: Dict[str, Any],
    limit: int = 50,
    exclude_ids: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve candidate products for recommendation ranking across 10,000+ catalog.
    Balances exploitation (customer preferred categories) with exploration (diverse categories).
    """
    exclude_ids = exclude_ids or []
    base_query: Dict[str, Any] = {
        "is_active": True,
        "$or": [{"stock_quantity": {"$gt": 0}}, {"stock": {"$gt": 0}}]
    }
    if exclude_ids:
        base_query["$and"] = [
            {"product_id": {"$nin": exclude_ids}},
            {"external_id": {"$nin": exclude_ids}}
        ]

    candidates = []
    seen_ids = set(exclude_ids)

    # 1. Exploitation: Products from preferred or dominant categories
    dominant_cat = context.get("dominant_category")
    preferred_cats = context.get("preferred_categories", [])
    target_cats = list(set([dominant_cat] + preferred_cats))
    target_cats = [c for c in target_cats if c]

    if target_cats:
        cat_regexes = [{"category": {"$regex": f"^{c}$", "$options": "i"}} for c in target_cats]
        cat_query = {
            "is_active": True,
            "$or": cat_regexes
        }
        if exclude_ids:
            cat_query["product_id"] = {"$nin": list(seen_ids)}

        cat_products = list(
            products_collection.find(cat_query)
            .sort([("rating", -1), ("popularity_score", -1)])
            .limit(int(limit * 0.6))
        )
        for p in cat_products:
            p_id = p.get("product_id") or p.get("external_id")
            if p_id not in seen_ids:
                candidates.append(p)
                seen_ids.add(p_id)

    # 2. Exploration: High-rated & trending products across all other categories
    remaining_needed = limit - len(candidates)
    if remaining_needed > 0:
        explore_query: Dict[str, Any] = {"is_active": True}
        if seen_ids:
            explore_query["product_id"] = {"$nin": list(seen_ids)}
        
        explore_products = list(
            products_collection.find(explore_query)
            .sort([("popularity_score", -1), ("rating", -1)])
            .limit(remaining_needed)
        )
        for p in explore_products:
            p_id = p.get("product_id") or p.get("external_id")
            if p_id not in seen_ids:
                candidates.append(p)
                seen_ids.add(p_id)

    # Clean MongoDB internal _id before returning
    for c in candidates:
        c.pop("_id", None)
        # Ensure currency and images
        c["currency"] = "INR"
        if not c.get("thumbnail") and c.get("images"):
            c["thumbnail"] = c["images"][0]

    return candidates
