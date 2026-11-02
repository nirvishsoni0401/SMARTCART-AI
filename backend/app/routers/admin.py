from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, desc
from app.database import get_db
from app.models import (
    User, Product, InteractionEvent, RecommendationLog, ModelState
)
from app.schemas import (
    AlgorithmUpdateRequest, RewardConfigUpdate, SimulateRequest
)
from app.auth import get_current_admin, get_current_user_optional
from app.config import settings
from app.services.model_store import model_store

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])

@router.get("/metrics")
def get_metrics(
    window: Optional[str] = Query("all", description="all, 1h, 24h, 7d"),
    db: Session = Depends(get_db)
):
    """
    Computes comparative multi-armed bandit metrics across all algorithms:
    Cumulative reward, average reward, CTR, conversion rate, negative feedback rate,
    recommendation latency, and catalogue coverage.
    """
    query = db.query(InteractionEvent)
    log_query = db.query(RecommendationLog)

    if window == "1h":
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        query = query.filter(InteractionEvent.created_at >= cutoff)
        log_query = log_query.filter(RecommendationLog.created_at >= cutoff)
    elif window == "24h":
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        query = query.filter(InteractionEvent.created_at >= cutoff)
        log_query = log_query.filter(RecommendationLog.created_at >= cutoff)
    elif window == "7d":
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        query = query.filter(InteractionEvent.created_at >= cutoff)
        log_query = log_query.filter(RecommendationLog.created_at >= cutoff)

    events = query.all()
    logs = log_query.all()
    total_products = db.query(Product).count() or 630

    # Aggregate by algorithm
    algo_names = ["linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy", "popularity", "random"]
    algo_stats = {
        name: {
            "algorithm": name,
            "cumulative_reward": 0.0,
            "events_count": 0,
            "impressions": 0,
            "clicks": 0,
            "cart_adds": 0,
            "purchases": 0,
            "dislikes": 0,
            "returns": 0,
            "skips": 0,
            "recommended_product_ids": set(),
            "exploration_count": 0,
            "total_logs": 0
        }
        for name in algo_names
    }

    for ev in events:
        algo = ev.algorithm.lower()
        if algo not in algo_stats:
            continue
        stats = algo_stats[algo]
        stats["cumulative_reward"] += ev.reward
        stats["events_count"] += 1

        t = ev.event_type
        if t == "impression":
            stats["impressions"] += 1
        elif t == "click":
            stats["clicks"] += 1
        elif t == "cart_add":
            stats["cart_adds"] += 1
        elif t == "purchase":
            stats["purchases"] += 1
        elif t == "dislike":
            stats["dislikes"] += 1
        elif t == "return":
            stats["returns"] += 1
        elif t == "skip":
            stats["skips"] += 1

    for lg in logs:
        algo = lg.algorithm.lower()
        if algo in algo_stats:
            stats = algo_stats[algo]
            stats["total_logs"] += 1
            stats["recommended_product_ids"].add(lg.product_id)
            if lg.was_exploration:
                stats["exploration_count"] += 1

    results = []
    for name in algo_names:
        st = algo_stats[name]
        impr = max(1, st["impressions"])
        clicks = st["clicks"]
        purchases = st["purchases"]
        neg_events = st["dislikes"] + st["returns"]
        total_ev = max(1, st["events_count"])

        ctr = clicks / impr
        conv_rate = purchases / max(1, clicks)
        neg_rate = neg_events / total_ev
        avg_reward = st["cumulative_reward"] / total_ev if st["events_count"] > 0 else 0.0
        coverage = len(st["recommended_product_ids"]) / total_products

        results.append({
            "algorithm": name,
            "cumulative_reward": round(st["cumulative_reward"], 2),
            "average_reward": round(avg_reward, 3),
            "events_count": st["events_count"],
            "impressions": st["impressions"],
            "clicks": st["clicks"],
            "purchases": st["purchases"],
            "cart_adds": st["cart_adds"],
            "dislikes": st["dislikes"],
            "returns": st["returns"],
            "ctr": round(ctr * 100.0, 2),
            "conversion_rate": round(conv_rate * 100.0, 2),
            "negative_feedback_rate": round(neg_rate * 100.0, 2),
            "catalogue_coverage": round(coverage * 100.0, 2),
            "unique_items_recommended": len(st["recommended_product_ids"]),
            "exploration_ratio": round((st["exploration_count"] / max(1, st["total_logs"])) * 100.0, 1)
        })

    return {
        "window": window,
        "total_catalog_products": total_products,
        "global_algorithm": settings.GLOBAL_ALGORITHM,
        "algorithms": results
    }

@router.get("/algorithms")
def get_algorithms():
    return {
        "active_algorithm": settings.GLOBAL_ALGORITHM,
        "available_algorithms": [
            {"id": "linucb", "name": "LinUCB (Sherman-Morrison)", "description": "Ridge regression with upper confidence bound exploration"},
            {"id": "discounted_linucb", "name": "Discounted LinUCB", "description": "Exponential discount factor gamma for non-stationary taste drift"},
            {"id": "linucb_thompson", "name": "Linear Thompson Sampling (LinTS)", "description": "Bayesian posterior parameter sampling"},
            {"id": "epsilon_greedy", "name": "Epsilon-Greedy", "description": "Greedy regression with epsilon random exploration"},
            {"id": "popularity", "name": "Popularity Baseline", "description": "Non-personalized popularity ranker"},
            {"id": "random", "name": "Random Policy", "description": "Uniform random selection baseline"},
            {"id": "ab_test", "name": "Round-Robin A/B Testing", "description": "Users assigned to different policies dynamically"}
        ]
    }

@router.put("/algorithms")
def set_active_algorithm(
    body: AlgorithmUpdateRequest,
    current_admin: User = Depends(get_current_admin)
):
    valid_algos = ["linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy", "popularity", "random", "ab_test"]
    if body.algorithm.lower() not in valid_algos:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Algorithm must be one of {valid_algos}")

    settings.GLOBAL_ALGORITHM = body.algorithm.lower()
    return {"status": "updated", "active_algorithm": settings.GLOBAL_ALGORITHM}

@router.get("/reward-config")
def get_reward_config():
    return settings.rewards.model_dump()

@router.put("/reward-config")
def update_reward_config(
    body: RewardConfigUpdate,
    current_admin: User = Depends(get_current_admin)
):
    if body.purchase is not None:
        settings.rewards.purchase = body.purchase
    if body.cart_add is not None:
        settings.rewards.cart_add = body.cart_add
    if body.wishlist is not None:
        settings.rewards.wishlist = body.wishlist
    if body.click is not None:
        settings.rewards.click = body.click
    if body.skip is not None:
        settings.rewards.skip = body.skip
    if body.cart_remove is not None:
        settings.rewards.cart_remove = body.cart_remove
    if body.dislike is not None:
        settings.rewards.dislike = body.dislike
    if body.return_item is not None:
        settings.rewards.return_item = body.return_item

    return {"status": "updated", "rewards": settings.rewards.model_dump()}

@router.post("/reset-models")
def reset_bandit_models(
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    model_store.reset_all(db=db)
    return {"status": "success", "message": "All bandit models have been reset to prior state."}

@router.post("/simulate")
def run_simulation_traffic(
    sim_req: SimulateRequest,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Run rounds of simulated shopping traffic across all algorithms
    to populate real-time metrics and charts in the admin dashboard.
    """
    import random
    from app.services.candidate_generation import generate_candidates
    from app.services.feature_engineering import build_user_context
    from app.services.diversity import apply_diversity_reranking
    from app.services.reward import get_reward_for_event

    target_algos = sim_req.algorithms or ["linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy", "popularity", "random"]
    num_rounds = sim_req.num_rounds
    products = db.query(Product).all()
    if not products:
        raise HTTPException(status_code=400, detail="No products in database")

    total_events_generated = 0

    for rnd in range(num_rounds):
        # Pick random algorithm
        algo_name = random.choice(target_algos)
        model = model_store.get_model(algo_name)

        # Create simulated session context
        dummy_session = f"sim_sess_{random.randint(100, 999)}"
        sim_budget = random.choice([1500.0, 3500.0, 10000.0, 25000.0, None])
        sim_cat = random.choice(["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Books", None])

        ctx = build_user_context(
            db=db,
            session_id=dummy_session,
            session_category=sim_cat,
            session_budget=sim_budget
        )

        candidates = generate_candidates(db, ctx, candidate_pool_size=30)
        scored = model.select(candidates, top_k=10)
        diverse = apply_diversity_reranking(scored, top_k=6)

        for rank, item in enumerate(diverse, start=1):
            prod = item.candidate.product_data
            # Log recommendation
            rec_log = RecommendationLog(
                session_id=dummy_session,
                product_id=prod.id,
                algorithm=algo_name,
                feature_vector=item.candidate.feature_vector.tolist(),
                score=float(item.score),
                ucb_bonus=float(item.ucb_bonus),
                was_exploration=bool(item.was_exploration),
                rank=rank,
                resolved=True
            )
            db.add(rec_log)
            db.flush()

            # Record impression
            db.add(InteractionEvent(
                session_id=dummy_session,
                product_id=prod.id,
                event_type="impression",
                reward=0.0,
                algorithm=algo_name,
                recommendation_log_id=rec_log.id
            ))
            total_events_generated += 1

            # Probabilistic user response model based on quality + match
            match = 0.5
            if sim_cat and prod.category == sim_cat:
                match += 0.35
            if sim_budget and prod.price <= sim_budget:
                match += 0.15
            match += (prod.rating - 3.5) * 0.1

            p_click = min(0.85, max(0.05, match * 0.45))
            if random.random() < p_click:
                # User clicked
                db.add(InteractionEvent(
                    session_id=dummy_session,
                    product_id=prod.id,
                    event_type="click",
                    reward=settings.rewards.click,
                    algorithm=algo_name,
                    recommendation_log_id=rec_log.id
                ))
                model.update(item.candidate.feature_vector, settings.rewards.click)
                total_events_generated += 1

                # Cart add?
                if random.random() < 0.35:
                    db.add(InteractionEvent(
                        session_id=dummy_session,
                        product_id=prod.id,
                        event_type="cart_add",
                        reward=settings.rewards.cart_add,
                        algorithm=algo_name,
                        recommendation_log_id=rec_log.id
                    ))
                    model.update(item.candidate.feature_vector, settings.rewards.cart_add)
                    total_events_generated += 1

                    # Purchase?
                    if random.random() < 0.5:
                        db.add(InteractionEvent(
                            session_id=dummy_session,
                            product_id=prod.id,
                            event_type="purchase",
                            reward=settings.rewards.purchase,
                            algorithm=algo_name,
                            recommendation_log_id=rec_log.id
                        ))
                        model.update(item.candidate.feature_vector, settings.rewards.purchase)
                        total_events_generated += 1
            else:
                # Small chance of dislike
                if random.random() < 0.05:
                    db.add(InteractionEvent(
                        session_id=dummy_session,
                        product_id=prod.id,
                        event_type="dislike",
                        reward=settings.rewards.dislike,
                        algorithm=algo_name,
                        recommendation_log_id=rec_log.id
                    ))
                    model.update(item.candidate.feature_vector, settings.rewards.dislike)
                    total_events_generated += 1

        # Save model after round
        model_store.save_model_to_db(algo_name, db=db)

    db.commit()
    return {
        "status": "simulation_complete",
        "rounds_executed": num_rounds,
        "total_events_generated": total_events_generated
    }

@router.get("/logs")
def get_recommendation_logs(
    limit: int = Query(25, ge=1, le=100),
    algorithm: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(RecommendationLog)
    if algorithm:
        query = query.filter(RecommendationLog.algorithm == algorithm)
    logs = query.order_by(RecommendationLog.created_at.desc()).limit(limit).all()

    return [
        {
            "id": lg.id,
            "product_id": lg.product_id,
            "product_name": lg.product.name if lg.product else "Product",
            "category": lg.product.category if lg.product else "N/A",
            "price": lg.product.price if lg.product else 0.0,
            "algorithm": lg.algorithm,
            "score": round(lg.score, 4),
            "ucb_bonus": round(lg.ucb_bonus, 4),
            "was_exploration": lg.was_exploration,
            "rank": lg.rank,
            "resolved": lg.resolved,
            "created_at": lg.created_at.isoformat()
        }
        for lg in logs
    ]

