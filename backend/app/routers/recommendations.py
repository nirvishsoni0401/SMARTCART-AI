import time
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Product, RecommendationLog, User
from app.schemas import RecommendationResponse, RecommendationItemOut, ProductOut
from app.auth import get_current_user_optional
from app.config import settings
from app.services.model_store import model_store
from app.services.candidate_generation import generate_candidates
from app.services.feature_engineering import build_user_context
from app.services.diversity import apply_diversity_reranking
from app.services.explain import generate_explanation
from app.services.latency_tracker import latency_tracker

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("", response_model=RecommendationResponse)
def get_recommendations(
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    session_id: Optional[str] = None,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    start_time = time.perf_counter()
    recommendation_id = str(uuid.uuid4())

    # Build context for user or session
    context = build_user_context(
        db=db,
        user_id=current_user.id if current_user else None,
        session_id=session_id,
        session_category=category,
        session_budget=max_price
    )

    is_cold_start = context.is_cold_start
    assigned_algo = settings.GLOBAL_ALGORITHM
    if current_user and current_user.assigned_algorithm:
        assigned_algo = current_user.assigned_algorithm

    # If user declined consent, strictly serve popularity baseline without tracking
    if current_user and not current_user.consent_tracking:
        assigned_algo = "popularity"

    model = model_store.get_model(assigned_algo)

    # 1. Candidate Generation (Stage 1: filter active + sample candidates)
    candidates = generate_candidates(
        db=db,
        context=context,
        candidate_pool_size=settings.CANDIDATE_POOL_SIZE
    )

    # 2. Policy Ranking (Stage 2: score candidates using active bandit)
    scoring_start = time.perf_counter()
    scored = model.select(candidates, top_k=min(len(candidates), limit * 3), context=None)

    # 3. Diversity Re-ranking (Stage 3: max per category + MMR near-duplicate penalty)
    diverse_scored = apply_diversity_reranking(
        scored_candidates=scored,
        top_k=limit,
        max_per_category=settings.MAX_PER_CATEGORY_TOP10
    )
    scoring_latency_ms = (time.perf_counter() - scoring_start) * 1000.0
    latency_tracker.record_scoring(scoring_latency_ms)

    # 4. Generate Explanations and persist RecommendationLogs
    theta = model.get_theta()
    items_out: List[RecommendationItemOut] = []
    ranked_list_ids = [it.candidate.product_id for it in diverse_scored]

    for rank, item in enumerate(diverse_scored, start=1):
        explanation = generate_explanation(
            scored_item=item,
            context=context,
            theta=theta,
            cold_start=is_cold_start
        )

        log_id = None
        # Only log individual recommendations if tracking consent is allowed
        should_log = current_user.consent_tracking if current_user else True
        if should_log:
            fv_list = item.candidate.feature_vector.tolist()
            rec_log = RecommendationLog(
                recommendation_id=recommendation_id,
                user_id=current_user.id if current_user else None,
                session_id=session_id,
                product_id=item.candidate.product_id,
                algorithm=model.name,
                model_version="v2.0",
                feature_vector=fv_list,
                feature_vector_snapshot=fv_list,
                score=float(item.score),
                ucb_bonus=float(item.ucb_bonus),
                was_exploration=bool(item.was_exploration),
                rank=rank,
                ranked_list=ranked_list_ids,
                origin="live",
                resolved=False
            )
            db.add(rec_log)
            db.flush()
            log_id = rec_log.id

        items_out.append(
            RecommendationItemOut(
                product=ProductOut.model_validate(item.candidate.product_data),
                score=round(float(item.score), 4),
                ucb_bonus=round(float(item.ucb_bonus), 4),
                was_exploration=item.was_exploration,
                explanation=explanation,
                algorithm=model.name,
                log_id=log_id
            )
        )

    if should_log:
        db.commit()

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return RecommendationResponse(
        recommendations=items_out,
        algorithm=model.name,
        cold_start=is_cold_start,
        latency_ms=round(latency_ms, 2),
        session_id=session_id
    )
