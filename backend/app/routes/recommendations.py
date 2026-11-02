from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException, Depends, Request
from pydantic import BaseModel
from app.services.auth_service import get_current_user_optional
from app.recommendation.recommendation_service import get_recommendations_for_user
from app.recommendation.policy_manager import policy_manager
from app.database import recommendations_collection, products_collection
from app.recommendation.feature_engineering import extract_features
from app.recommendation.context_builder import build_customer_context

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])


class FeedbackRequest(BaseModel):
    recommendation_id: Optional[str] = None
    product_id: int
    action: str  # "like", "dislike", "click", "not_interested", "cart", "purchase"


@router.get("/")
def get_recommendations(
    request: Request,
    limit: int = Query(8, ge=1, le=24),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    user_id = current_user.get("id") if current_user else None
    session_id = request.headers.get("x-session-id") or request.cookies.get("smartcart_session_id") or "guest_session"

    result = get_recommendations_for_user(
        user_id=user_id,
        session_id=session_id,
        limit=limit
    )

    return result


@router.post("/feedback")
def submit_recommendation_feedback(
    payload: FeedbackRequest,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    action_map = {
        "like": ("explicit_like", 4.0),
        "dislike": ("explicit_dislike", -5.0),
        "click": ("product_click", 2.0),
        "cart": ("add_to_cart", 5.0),
        "purchase": ("purchase", 10.0),
        "not_interested": ("explicit_dislike", -4.0)
    }

    if payload.action not in action_map:
        raise HTTPException(status_code=400, detail=f"Invalid action '{payload.action}'. Valid: {list(action_map.keys())}")

    event_type, reward = action_map[payload.action]
    prod = products_collection.find_one({"external_id": payload.product_id})
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found")

    user_id = current_user.get("id") if current_user else None
    session_id = request.headers.get("x-session-id") or "guest_session"

    context = build_customer_context(user_id=user_id, session_id=session_id)
    feature_vector = extract_features(prod, context)
    arm_id = f"cat:{prod.get('category', 'general').lower()}"

    update_result = policy_manager.update_policy(
        arm_id=arm_id,
        feature_vector=feature_vector,
        reward=reward,
        event_type=event_type
    )

    return {
        "status": "success",
        "action": payload.action,
        "reward_applied": reward,
        "policy_update": update_result
    }


@router.get("/history")
def get_recommendation_history(
    request: Request,
    limit: int = Query(20, ge=1, le=50),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    query = {}
    if current_user and current_user.get("id"):
        query["user_id"] = current_user["id"]
    else:
        session_id = request.headers.get("x-session-id")
        if session_id:
            query["session_id"] = session_id

    records = list(recommendations_collection.find(query, {"_id": 0}).sort("created_at", -1).limit(limit))
    return {
        "history": records,
        "total": len(records)
    }


@router.get("/status")
def get_recommendation_status():
    return policy_manager.get_stats()

