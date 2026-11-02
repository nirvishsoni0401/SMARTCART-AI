import uuid
import datetime
from typing import List, Dict, Any, Optional
from app.database import recommendations_collection, products_collection, events_collection
from app.recommendation.context_builder import build_customer_context
from app.recommendation.candidate_generator import generate_candidates
from app.recommendation.feature_engineering import extract_features
from app.recommendation.policy_manager import policy_manager


def generate_recommendation_explanation(product: Dict[str, Any], context: Dict[str, Any], exp_score: float) -> str:
    category = product.get("category", "")
    dominant_cat = context.get("dominant_category")
    avg_price = context.get("average_viewed_price", 1500.0)
    prod_price = float(product.get("discounted_price") or product.get("price", 0))
    rating = float(product.get("rating", 0))

    if dominant_cat and category.lower() == dominant_cat.lower():
        return f"Because you frequently browse {category}"
    
    if abs(prod_price - avg_price) / max(1.0, avg_price) < 0.35:
        return f"Matches your preferred ~₹{int(avg_price):,} budget"

    if rating >= 4.5:
        return f"Top-rated ({rating}★) customer favorite"

    if exp_score > 0.4:
        return f"High adaptive match for your current session"

    return f"Trending product in {category}"


def get_recommendations_for_user(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 8,
    exclude_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Produce personalized recommendations using Contextual LinUCB across 10,000+ catalog.
    """
    # 1. Build real-time customer context
    context = build_customer_context(user_id=user_id, session_id=session_id)

    # 2. Candidate generation
    candidates = generate_candidates(context=context, limit=limit * 4, exclude_ids=exclude_ids or [])
    if not candidates:
        return {"recommendations": [], "recommendation_id": None, "context": context}

    # 3. Score candidates with LinUCB
    agent = policy_manager.get_agent()
    scored_items = []

    for prod in candidates:
        arm_id = f"cat:{prod.get('category', 'general').lower()}"
        x = extract_features(product=prod, context=context)
        ucb_score, expected_reward, bonus = agent.score_arm(arm_id, x)

        explanation = generate_recommendation_explanation(prod, context, expected_reward)

        scored_items.append({
            "product": prod,
            "arm_id": arm_id,
            "ucb_score": round(ucb_score, 4),
            "expected_reward": round(expected_reward, 4),
            "exploration_bonus": round(bonus, 4),
            "explanation": explanation,
            "feature_vector": x.tolist()
        })

    # Sort descending by UCB score
    scored_items.sort(key=lambda item: item["ucb_score"], reverse=True)
    top_items = scored_items[:limit]

    recommendation_id = f"rec_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Structure response items with full backward and forward compatibility
    formatted_recommendations = []
    for item in top_items:
        p = item["product"]
        pid = p.get("product_id") or p.get("external_id")
        title = p.get("title") or p.get("name", "Product")
        img = p.get("thumbnail") or p.get("image_url") or (p.get("images")[0] if p.get("images") else "")
        disc_price = p.get("discounted_price") or p.get("price", 0)

        formatted_recommendations.append({
            "product_id": pid,
            "external_id": pid,
            "title": title,
            "name": title,
            "category": p.get("category"),
            "subcategory": p.get("subcategory"),
            "brand": p.get("brand"),
            "price": p.get("price"),
            "discounted_price": disc_price,
            "discount_percentage": p.get("discount_percentage", 0),
            "currency": "INR",
            "rating": p.get("rating"),
            "review_count": p.get("review_count", 0),
            "thumbnail": img,
            "image_url": img,
            "images": p.get("images", [img]),
            "stock_quantity": p.get("stock_quantity", p.get("stock", 0)),
            "stock": p.get("stock_quantity", p.get("stock", 0)),
            "availability": p.get("availability", "In Stock"),
            "explanation": item["explanation"],
            "ucb_score": item["ucb_score"],
            "expected_reward": item["expected_reward"],
            "exploration_bonus": item["exploration_bonus"],
            "arm_id": item["arm_id"]
        })

    # Record recommendation log in DB for attribution
    try:
        recommendations_collection.insert_one({
            "recommendation_id": recommendation_id,
            "user_id": user_id,
            "session_id": session_id,
            "items": [
                {
                    "product_id": it["product"].get("product_id") or it["product"].get("external_id"),
                    "arm_id": it["arm_id"],
                    "ucb_score": it["ucb_score"],
                    "feature_vector": it["feature_vector"]
                }
                for it in top_items
            ],
            "created_at": now_iso
        })
    except Exception as e:
        print(f"Warning storing recommendation log: {e}")

    return {
        "recommendation_id": recommendation_id,
        "algorithm": "LinUCB",
        "context_summary": {
            "dominant_category": context.get("dominant_category"),
            "budget_level": context.get("budget_level"),
            "interactions": context.get("total_interactions")
        },
        "recommendations": formatted_recommendations
    }


def process_interaction_event(event_doc: Dict[str, Any]):
    """
    Map customer interaction to LinUCB policy reward and trigger atomic update.
    """
    event_type = event_doc.get("event_type")
    product_id = event_doc.get("product_id")
    rec_id = event_doc.get("recommendation_id")

    reward_engine = policy_manager.reward_engine
    raw_reward = reward_engine.get_reward_value(event_type)

    if raw_reward == 0.0 and event_type != "product_impression":
        return

    # Find product and construct feature vector
    prod = products_collection.find_one({"$or": [{"product_id": product_id}, {"external_id": product_id}]})
    if not prod:
        return

    category = prod.get("category", "general").lower()
    arm_id = f"cat:{category}"

    # Reconstruct context
    user_id = event_doc.get("user_id")
    session_id = event_doc.get("session_id")
    context = build_customer_context(user_id=user_id, session_id=session_id)
    feature_vector = extract_features(product=prod, context=context)

    # Compute deduplicated attributed reward
    reward_to_apply = raw_reward
    if rec_id:
        prior_events = list(events_collection.find({
            "recommendation_id": rec_id,
            "product_id": product_id
        }))
        prior_accum = sum(reward_engine.get_reward_value(e.get("event_type")) for e in prior_events if e.get("event_type") != event_type)
        prior_stage = prior_events[-1].get("event_type") if prior_events else None
        reward_to_apply = reward_engine.calculate_attributed_reward(event_type, prior_accum, prior_stage)

    if reward_to_apply != 0.0:
        policy_manager.update_policy(
            arm_id=arm_id,
            feature_vector=feature_vector,
            reward=reward_to_apply,
            event_type=event_type
        )
