import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field
from app.database import events_collection, products_collection
from app.services.auth_service import get_current_user_optional

router = APIRouter(prefix="/api/events", tags=["Events"])

VALID_EVENT_TYPES = {
    "product_impression",
    "product_click",
    "product_view",
    "add_to_cart",
    "remove_from_cart",
    "add_to_wishlist",
    "purchase",
    "product_skip",
    "explicit_dislike",
    "explicit_like",
    "product_return",
    "search"
}


class TrackEventRequest(BaseModel):
    event_type: str
    product_id: Optional[int] = None
    session_id: Optional[str] = None
    recommendation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchEventsRequest(BaseModel):
    events: List[TrackEventRequest]


@router.post("/", status_code=status.HTTP_201_CREATED)
def record_event(
    req: TrackEventRequest,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    if req.event_type not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type '{req.event_type}'. Valid: {list(VALID_EVENT_TYPES)}"
        )

    user_id = current_user.get("id") if current_user else None
    session_id = req.session_id or request.headers.get("x-session-id") or request.cookies.get("smartcart_session_id") or "anonymous_session"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    product_category = None
    if req.product_id:
        prod = products_collection.find_one({"external_id": req.product_id}, {"category": 1})
        if prod:
            product_category = prod.get("category")

    event_doc = {
        "event_type": req.event_type,
        "product_id": req.product_id,
        "category": product_category,
        "user_id": user_id,
        "session_id": session_id,
        "recommendation_id": req.recommendation_id,
        "metadata": req.metadata or {},
        "created_at": now_iso
    }

    events_collection.insert_one(event_doc)

    # If it is a rewardable feedback event for the LinUCB bandit, trigger bandit feedback handling
    # (clicks, likes, dislikes, add_to_cart, purchase, etc.)
    try:
        from app.recommendation.recommendation_service import process_interaction_event
        process_interaction_event(event_doc)
    except Exception as e:
        # Don't let recommendation feedback processing break event logging
        print(f"Bandit event processing note: {e}")

    return {
        "status": "success",
        "event_type": req.event_type,
        "timestamp": now_iso
    }


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def record_events_batch(
    req: BatchEventsRequest,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    user_id = current_user.get("id") if current_user else None
    session_id = request.headers.get("x-session-id") or "anonymous_session"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    docs = []
    for ev in req.events:
        if ev.event_type in VALID_EVENT_TYPES:
            docs.append({
                "event_type": ev.event_type,
                "product_id": ev.product_id,
                "user_id": user_id,
                "session_id": ev.session_id or session_id,
                "recommendation_id": ev.recommendation_id,
                "metadata": ev.metadata or {},
                "created_at": now_iso
            })

    if docs:
        events_collection.insert_many(docs)

    return {"status": "success", "count": len(docs)}

