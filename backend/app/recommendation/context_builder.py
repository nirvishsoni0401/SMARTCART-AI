from typing import Optional, Dict, Any, List
from app.database import events_collection, users_collection, orders_collection

# Standard category taxonomy in SmartCart AI (10 Major Categories + Subcategories)
KNOWN_CATEGORIES = [
    "electronics",
    "fashion",
    "home & kitchen",
    "home-and-kitchen",
    "beauty & personal care",
    "beauty-and-personal-care",
    "sports & fitness",
    "sports-and-fitness",
    "books",
    "automotive",
    "toys & baby products",
    "toys-and-baby",
    "grocery & food",
    "grocery-and-food",
    "office & stationery",
    "office-and-stationery"
]


def build_customer_context(user_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Build real-time customer context from past events, orders, and user profile.
    Returns a dictionary of contextual features in INR currency.
    """
    category_counts: Dict[str, int] = {cat: 0 for cat in KNOWN_CATEGORIES}
    total_events = 0
    viewed_prices: List[float] = []
    
    # Query criteria
    query = {}
    if user_id:
        query["user_id"] = user_id
    elif session_id:
        query["session_id"] = session_id

    # 1. Fetch recent events (last 50 interactions)
    if query:
        recent_events = list(events_collection.find(query).sort("created_at", -1).limit(50))
        for ev in recent_events:
            total_events += 1
            cat = ev.get("category", "").lower()
            if cat:
                if cat not in category_counts:
                    category_counts[cat] = 0
                # Weight by event type
                ev_type = ev.get("event_type", "")
                weight = 1
                if ev_type in ("add_to_cart", "explicit_like"):
                    weight = 3
                elif ev_type == "purchase":
                    weight = 5
                elif ev_type == "explicit_dislike":
                    weight = -2
                category_counts[cat] = max(0, category_counts[cat] + weight)

            price = ev.get("metadata", {}).get("price")
            if price is not None:
                try:
                    viewed_prices.append(float(price))
                except (ValueError, TypeError):
                    pass

    # 2. Check user profile for preferred categories
    top_preferred = []
    if user_id:
        user = users_collection.find_one({"id": user_id})
        if user and user.get("preferred_categories"):
            for pref in user["preferred_categories"]:
                clean_pref = pref.lower().strip()
                top_preferred.append(clean_pref)
                if clean_pref not in category_counts:
                    category_counts[clean_pref] = 0
                category_counts[clean_pref] += 4

    # 3. Calculate dominant category and average budget
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    dominant_category = sorted_cats[0][0] if sorted_cats and sorted_cats[0][1] > 0 else None
    
    avg_price = sum(viewed_prices) / len(viewed_prices) if viewed_prices else 1850.0
    # Normalize price affinity (economy < 1000 INR, mid 1000 - 5000 INR, premium > 5000 INR)
    budget_level = "economy" if avg_price < 1000 else ("mid" if avg_price <= 5000 else "premium")

    return {
        "user_id": user_id,
        "session_id": session_id,
        "total_interactions": total_events,
        "category_affinities": category_counts,
        "dominant_category": dominant_category,
        "preferred_categories": top_preferred,
        "average_viewed_price": round(avg_price, 2),
        "budget_level": budget_level
    }
