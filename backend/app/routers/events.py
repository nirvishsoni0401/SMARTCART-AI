from datetime import datetime, timedelta, timezone
from typing import Optional, List
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import InteractionEvent, RecommendationLog, Product, User
from app.schemas import EventCreate, EventOut, ImpressionBatchCreate
from app.auth import get_current_user_optional
from app.config import settings
from app.services.reward import get_reward_for_event
from app.services.feature_engineering import build_user_context, extract_feature_vector
from app.services.model_store import model_store

router = APIRouter(prefix="/events", tags=["Events"])

def resolve_stale_impressions(db: Session) -> None:
    """
    Lazy resolution: find recommendation logs older than IMPRESSION_TIMEOUT_MINUTES
    that were never clicked or interacted with, and resolve them as 'skip'.
    """
    timeout_threshold = datetime.now(timezone.utc) - timedelta(minutes=settings.IMPRESSION_TIMEOUT_MINUTES)
    stale_logs = (
        db.query(RecommendationLog)
        .filter(
            RecommendationLog.resolved == False,
            RecommendationLog.created_at <= timeout_threshold
        )
        .limit(50)
        .all()
    )

    if not stale_logs:
        return

    for log in stale_logs:
        log.resolved = True
        skip_reward = settings.rewards.skip
        if skip_reward != 0.0 and log.feature_vector:
            model = model_store.get_model(log.algorithm)
            phi = np.array(log.feature_vector, dtype=np.float64)
            model.update(phi, skip_reward)
            model_store.save_model_to_db(log.algorithm, db=db)

    db.commit()

@router.post("", response_model=EventOut, status_code=status.HTTP_201_CREATED)
def record_event(
    event_in: EventCreate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    # 1. Lazily resolve stale unclicked impressions
    resolve_stale_impressions(db)

    # 2. Check user consent
    should_track = current_user.consent_tracking if current_user else True
    if not should_track:
        raise HTTPException(
            status_code=status.HTTP_200_OK,
            detail="Tracking consent is disabled; event discarded."
        )

    # 3. Deduplication check: do not update model twice for same (log_id, event_type)
    if event_in.recommendation_log_id:
        existing_event = db.query(InteractionEvent).filter(
            InteractionEvent.recommendation_log_id == event_in.recommendation_log_id,
            InteractionEvent.event_type == event_in.event_type
        ).first()
        if existing_event:
            return EventOut.model_validate(existing_event)

    product = db.query(Product).filter(Product.id == event_in.product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    reward = get_reward_for_event(event_in.event_type)
    algo_name = current_user.assigned_algorithm if current_user else settings.GLOBAL_ALGORITHM
    feature_vector = None

    # Retrieve feature vector from recommendation log if available
    rec_log = None
    if event_in.recommendation_log_id:
        rec_log = db.query(RecommendationLog).filter(RecommendationLog.id == event_in.recommendation_log_id).first()
        if rec_log:
            algo_name = rec_log.algorithm
            feature_vector = np.array(rec_log.feature_vector, dtype=np.float64)
            rec_log.resolved = True

    # Fallback to computing feature vector on-the-fly
    if feature_vector is None:
        ctx = build_user_context(
            db=db,
            user_id=current_user.id if current_user else None,
            session_id=event_in.session_id
        )
        feature_vector = extract_feature_vector(ctx, product)

    # Online model update
    model = model_store.get_model(algo_name)
    model.update(feature_vector, reward)
    model_store.save_model_to_db(algo_name, db=db)

    # Persist interaction event
    event = InteractionEvent(
        user_id=current_user.id if current_user else None,
        session_id=event_in.session_id,
        product_id=event_in.product_id,
        event_type=event_in.event_type,
        reward=reward,
        algorithm=algo_name,
        recommendation_log_id=event_in.recommendation_log_id
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    return EventOut.model_validate(event)

@router.post("/impressions", status_code=status.HTTP_201_CREATED)
def record_impressions_batch(
    batch: ImpressionBatchCreate,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    should_track = current_user.consent_tracking if current_user else True
    if not should_track:
        return {"status": "ignored", "reason": "tracking_consent_disabled"}

    algo_name = current_user.assigned_algorithm if current_user else settings.GLOBAL_ALGORITHM

    for i, prod_id in enumerate(batch.product_ids):
        log_id = batch.recommendation_log_ids[i] if batch.recommendation_log_ids and i < len(batch.recommendation_log_ids) else None
        event = InteractionEvent(
            user_id=current_user.id if current_user else None,
            session_id=batch.session_id,
            product_id=prod_id,
            event_type="impression",
            reward=0.0,
            algorithm=algo_name,
            recommendation_log_id=log_id
        )
        db.add(event)

    db.commit()
    return {"status": "recorded", "count": len(batch.product_ids)}

