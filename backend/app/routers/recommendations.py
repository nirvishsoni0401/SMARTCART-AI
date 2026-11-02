import time
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User, Product, RecommendationLog
from app.schemas import RecommendationResponse, RecommendationItemOut, ProductOut
from app.auth import get_current_user_optional
from app.config import settings
from app.services.feature_engineering import build_user_context
from app.services.candidate_generation import generate_candidates
from app.services.diversity import apply_diversity_reranking
from app.services.explain import generate_explanation
from app.services.model_store import model_store

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

@router.get("", response_model=RecommendationResponse)
def get_recommendations(
    category: Optional[str] = Query(None, description="Category filter context"),
    query: Optional[str] = Query(None, description="Search keyword context"),
    max_price: Optional[float] = Query(None, description="Session budget cap in INR"),
    session_id: Optional[str] = Query(None, description="Guest session identifier"),
    limit: int = Query(10, ge=1, le=30),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    start_time = time.perf_counter()

    # Determine algorithm and consent
    assigned_algo = settings.GLOBAL_ALGORITHM
    if assigned_algo == "ab_test" and current_user:
        assigned_algo = current_user.assigned_algorithm

    # If tracking consent is declined, strictly fall back to popularity
    if current_user and not current_user.consent_tracking:
        assigned_algo = "popularity"

    # Build user context
    context = build_user_context(
        db=db,
        user_id=current_user.id if current_user else None,
        session_id=session_id,
        session_category=category,
        session_query=query,
        session_budget=max_price
    )

    is_cold_start = context.total_interactions <= 1
    model = model_store.get_model(assigned_algo)

    # 1. Candidate Generation (Stage 1: filter + sample 50 candidates)
    candidates = generate_candidates(
        db=db,
        context=context,
        candidate_pool_size=settings.CANDIDATE_POOL_SIZE
    )

    # 2. Policy Ranking (Stage 2: score candidates using active bandit)
    # Score 2x the requested limit to give diversity re-ranker enough items
    scored = model.select(candidates, top_k=min(len(candidates), limit * 3), context=None)

    # 3. Diversity Re-ranking (Stage 3: max 3 per category + MMR near-duplicate penalty)
    diverse_scored = apply_diversity_reranking(
        scored_candidates=scored,
        top_k=limit,
        max_per_category=settings.MAX_PER_CATEGORY_TOP10
    )

    # 4. Generate Explanations and persist RecommendationLogs
    theta = model.get_theta()
    items_out: List[RecommendationItemOut] = []

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
            rec_log = RecommendationLog(
                user_id=current_user.id if current_user else None,
                session_id=session_id,
                product_id=item.candidate.product_id,
                algorithm=model.name,
                feature_vector=item.candidate.feature_vector.tolist(),
                score=float(item.score),
                ucb_bonus=float(item.ucb_bonus),
                was_exploration=bool(item.was_exploration),
                rank=rank,
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

