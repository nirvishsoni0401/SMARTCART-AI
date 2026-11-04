import time
import uuid
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct, desc
from app.database import get_db
from app.models import (
    User, Product, InteractionEvent, RecommendationLog, ModelState, ModelHistory
)
from app.schemas import (
    AlgorithmUpdateRequest, RewardConfigUpdate, SimulateRequest, WarmStartApplyRequest
)
from app.auth import get_current_admin, get_current_user_optional
from app.config import settings
from app.services.model_store import model_store
from app.services.latency_tracker import latency_tracker

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])

@router.get("/metrics")
def get_metrics(
    window: Optional[str] = Query("all", description="all, 1h, 24h, 7d"),
    origin: Optional[str] = Query("live", description="live, simulated, legacy, all"),
    db: Session = Depends(get_db)
):
    """
    Computes comparative multi-armed bandit metrics across all algorithms:
    Filtered by origin (live, simulated, legacy, all) and time window.
    Includes p50/p95/p99 latency percentiles for both e2e requests and model scoring.
    """
    query = db.query(InteractionEvent)
    log_query = db.query(RecommendationLog)

    # Origin filter
    if origin and origin != "all":
        query = query.filter(InteractionEvent.origin == origin)
        log_query = log_query.filter(RecommendationLog.origin == origin)

    # Time window filter
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
    active_products = db.query(Product).filter(Product.is_active == True).count()
    total_users = db.query(User).count()

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
        coverage = len(st["recommended_product_ids"]) / max(1, active_products)

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
            "exploration_pct": round((st["exploration_count"] / max(1, st["total_logs"])) * 100.0, 2),
            "total_recommendations_served": st["total_logs"]
        })

    # Overall system stats
    total_all_events = len(events)
    overall_reward = sum(ev.reward for ev in events)
    overall_avg_reward = overall_reward / max(1, total_all_events) if total_all_events > 0 else 0.0
    overall_clicks = sum(st["clicks"] for st in algo_stats.values())
    overall_impr = max(1, sum(st["impressions"] for st in algo_stats.values()))
    overall_ctr = (overall_clicks / overall_impr) * 100.0

    lat_percentiles = latency_tracker.get_percentiles()

    return {
        "active_algorithm": settings.GLOBAL_ALGORITHM,
        "total_users": total_users,
        "total_catalog_products": total_products,
        "active_catalog_products": active_products,
        "origin_filter": origin,
        "algorithms": results,
        "metrics": results,
        "latency_percentiles": lat_percentiles,
        "overall": {
            "total_events": total_all_events,
            "overall_avg_reward": round(overall_avg_reward, 3),
            "overall_ctr": round(overall_ctr, 2),
            "e2e_p50_ms": lat_percentiles["end_to_end"]["p50"],
            "e2e_p95_ms": lat_percentiles["end_to_end"]["p95"],
            "e2e_p99_ms": lat_percentiles["end_to_end"]["p99"],
            "model_scoring_p50_ms": lat_percentiles["model_scoring"]["p50"],
            "model_scoring_p95_ms": lat_percentiles["model_scoring"]["p95"],
            "model_scoring_p99_ms": lat_percentiles["model_scoring"]["p99"]
        }
    }

@router.get("/algorithms")
def get_algorithms(current_admin: User = Depends(get_current_admin)):
    return {
        "available_algorithms": [
            {"id": "linucb", "name": "LinUCB (Contextual Bandit)", "description": "Balances reward exploitation with upper confidence bounds via Sherman-Morrison rank-1 updates"},
            {"id": "discounted_linucb", "name": "Discounted LinUCB", "description": "Exponential forgetting factor (gamma=0.95) for rapid adaptation to customer taste shifts"},
            {"id": "linucb_thompson", "name": "Linear Thompson Sampling", "description": "Bayesian posterior sampling of parameters from covariance matrix"},
            {"id": "epsilon_greedy", "name": "Epsilon-Greedy Bandit", "description": "Exploits predicted reward with 10% uniform random exploration"},
            {"id": "popularity", "name": "Popularity Prior Baseline", "description": "Non-personalized heuristic baseline ranking items by rating and sales volume"},
            {"id": "random", "name": "Uniform Random Policy", "description": "Unbiased exploration baseline sampling uniformly across active products"}
        ],
        "active_algorithm": settings.GLOBAL_ALGORITHM
    }

@router.put("/algorithms")
@router.post("/algorithms")
def set_algorithm(
    body: AlgorithmUpdateRequest,
    current_admin: User = Depends(get_current_admin)
):
    valid = ["linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy", "popularity", "random"]
    if body.algorithm.lower() not in valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid algorithm. Must be one of: {', '.join(valid)}"
        )
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
def reset_all_models(
    sandbox_only: bool = Query(False, description="Reset only sandbox models"),
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    model_store.reset_all(db=db, sandbox_only=sandbox_only)
    target = "Sandbox" if sandbox_only else "All"
    return {"status": "success", "message": f"{target} bandit models have been reset to prior state."}

@router.post("/models/snapshot")
def create_model_snapshot(
    name: str = Query("pre_phase8", description="Snapshot label name"),
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    count = model_store.archive_snapshot(name, db=db)
    return {"status": "success", "snapshot_name": name, "models_archived": count}

@router.post("/models/rollback")
def rollback_model_snapshot(
    name: str = Query("pre_phase8", description="Snapshot label name to restore"),
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    success = model_store.rollback_snapshot(name, db=db)
    if not success:
        raise HTTPException(status_code=404, detail=f"Snapshot '{name}' not found")
    return {"status": "success", "message": f"Successfully restored models from snapshot '{name}'"}

@router.post("/simulate")
def run_simulation_traffic(
    sim_req: SimulateRequest,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Run rounds of simulated shopping traffic.
    Unless train_live=True is passed, writes ONLY origin='simulated' data
    and updates ONLY separate sandbox model states (sim:*), leaving live models untouched.
    """
    import random
    from app.services.candidate_generation import generate_candidates
    from app.services.feature_engineering import build_user_context
    from app.services.diversity import apply_diversity_reranking
    from app.services.reward import get_reward_for_event

    train_live = sim_req.train_live
    sandbox = not train_live
    data_origin = "live" if train_live else "simulated"

    target_algos = sim_req.algorithms or ["linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy", "popularity", "random"]
    num_rounds = sim_req.num_rounds
    products = db.query(Product).filter(Product.is_active == True).all()
    if not products:
        # Fallback to any product if active not yet populated
        products = db.query(Product).all()
    if not products:
        raise HTTPException(status_code=400, detail="No products in database")

    total_events_generated = 0

    for rnd in range(num_rounds):
        algo_name = random.choice(target_algos)
        # Fetch appropriate model: sandbox (sim:*) vs live
        model = model_store.get_model(algo_name, sandbox=sandbox)

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

        ranked_list_ids = [it.candidate.product_id for it in diverse]
        rec_id = str(uuid.uuid4())

        for rank, item in enumerate(diverse, start=1):
            prod = item.candidate.product_data
            fv_list = item.candidate.feature_vector.tolist()

            # Log recommendation with proper origin
            rec_log = RecommendationLog(
                recommendation_id=rec_id,
                session_id=dummy_session,
                product_id=prod.id,
                algorithm=algo_name,
                model_version="v2.0-sim" if sandbox else "v2.0",
                feature_vector=fv_list,
                feature_vector_snapshot=fv_list,
                score=float(item.score),
                ucb_bonus=float(item.ucb_bonus),
                was_exploration=bool(item.was_exploration),
                rank=rank,
                ranked_list=ranked_list_ids,
                origin=data_origin,
                resolved=True
            )
            db.add(rec_log)
            db.flush()

            # Record impression with proper origin
            db.add(InteractionEvent(
                session_id=dummy_session,
                product_id=prod.id,
                event_type="impression",
                reward=0.0,
                algorithm=algo_name,
                origin=data_origin,
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
                    origin=data_origin,
                    recommendation_log_id=rec_log.id
                ))
                model.update(item.candidate.feature_vector, settings.rewards.click)
                total_events_generated += 1

                # Dislike (thumbs-down) feedback (5% of clicks)
                if random.random() < 0.05:
                    db.add(InteractionEvent(
                        session_id=dummy_session,
                        product_id=prod.id,
                        event_type="dislike",
                        reward=settings.rewards.dislike,
                        algorithm=algo_name,
                        origin=data_origin,
                        recommendation_log_id=rec_log.id
                    ))
                    model.update(item.candidate.feature_vector, settings.rewards.dislike)
                    total_events_generated += 1

                # Add to cart (35% of clicks)
                elif random.random() < 0.35:
                    db.add(InteractionEvent(
                        session_id=dummy_session,
                        product_id=prod.id,
                        event_type="cart_add",
                        reward=settings.rewards.cart_add,
                        algorithm=algo_name,
                        origin=data_origin,
                        recommendation_log_id=rec_log.id
                    ))
                    model.update(item.candidate.feature_vector, settings.rewards.cart_add)
                    total_events_generated += 1

                    # Purchase checkout (40% of cart adds)
                    if random.random() < 0.40:
                        db.add(InteractionEvent(
                            session_id=dummy_session,
                            product_id=prod.id,
                            event_type="purchase",
                            reward=settings.rewards.purchase,
                            algorithm=algo_name,
                            origin=data_origin,
                            recommendation_log_id=rec_log.id
                        ))
                        model.update(item.candidate.feature_vector, settings.rewards.purchase)
                        total_events_generated += 1

                        # Delayed return (10% of purchases)
                        if random.random() < 0.10:
                            db.add(InteractionEvent(
                                session_id=dummy_session,
                                product_id=prod.id,
                                event_type="return",
                                reward=settings.rewards.return_item,
                                algorithm=algo_name,
                                origin=data_origin,
                                recommendation_log_id=rec_log.id
                            ))
                            delta = settings.rewards.return_item - settings.rewards.purchase
                            model.correct_reward(item.candidate.feature_vector, delta)
                            total_events_generated += 1

        # Persist model updates to sandbox vs live
        model_store.save_model_to_db(algo_name, sandbox=sandbox, db=db)

    db.commit()

    return {
        "status": "success",
        "rounds_executed": num_rounds,
        "train_live": train_live,
        "origin": data_origin,
        "sandbox_mode": sandbox,
        "events_generated": total_events_generated,
        "total_events_generated": total_events_generated
    }

@router.get("/logs")
def get_recommendation_logs(
    limit: int = Query(50, ge=1, le=200),
    algorithm: Optional[str] = None,
    origin: Optional[str] = Query("live", description="live, simulated, legacy, all"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    query = db.query(RecommendationLog)
    if origin and origin != "all":
        query = query.filter(RecommendationLog.origin == origin)
    if algorithm and algorithm.lower() != "all":
        query = query.filter(RecommendationLog.algorithm == algorithm.lower())

    logs = query.order_by(desc(RecommendationLog.created_at)).limit(limit).all()

    return [
        {
            "id": log.id,
            "recommendation_id": log.recommendation_id,
            "user_id": log.user_id,
            "session_id": log.session_id,
            "product_id": log.product_id,
            "product_name": log.product.name if log.product else f"Product #{log.product_id}",
            "category": log.product.category if log.product else "N/A",
            "algorithm": log.algorithm,
            "model_version": log.model_version,
            "origin": log.origin,
            "score": round(log.score, 4),
            "ucb_bonus": round(log.ucb_bonus, 4),
            "was_exploration": log.was_exploration,
            "rank": log.rank,
            "created_at": log.created_at.isoformat()
        }
        for log in logs
    ]


@router.get("/models/warm-start/snapshots")
def get_warm_start_snapshots(
    db: Session = Depends(get_db)
):
    """
    Lists available precomputed warm-start snapshots saved in model_states
    (keys prefixed with 'warm_start:'), their metadata, updates, and condition numbers.
    """
    rows = db.query(ModelState).filter(ModelState.algorithm.like("warm_start:%")).all()
    snapshots = []
    for r in rows:
        meta = {}
        try:
            parsed = json.loads(r.state_blob)
            meta = parsed.get("metadata", {})
        except Exception:
            pass
        snapshots.append({
            "snapshot_key": r.algorithm,
            "algorithm": meta.get("algorithm", r.algorithm.split(":")[-1]),
            "n_updates": r.n_updates,
            "condition_number": meta.get("condition_number"),
            "scale_s": meta.get("scale_s"),
            "n_hist": meta.get("n_hist"),
            "n_cap": meta.get("n_cap"),
            "lambda_reg": meta.get("lambda_reg"),
            "alpha": meta.get("alpha"),
            "pseudo_negatives_m": meta.get("pseudo_negatives_m"),
            "feature_spec_hash": meta.get("feature_spec_hash"),
            "created_at": meta.get("created_at") or (r.updated_at.isoformat() if r.updated_at else None),
            "updated_at": r.updated_at.isoformat() if r.updated_at else None
        })
    return snapshots


@router.post("/models/warm-start/apply")
def apply_warm_start(
    req: WarmStartApplyRequest,
    current_admin: User = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """
    Copies a specified warm-start snapshot into the active operational slot
    for that algorithm. Requires admin authentication.
    Creates an audit log record in model_history with origin='warm_start'.
    """
    snapshot = db.query(ModelState).filter(ModelState.algorithm == req.snapshot_key).first()
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Warm-start snapshot '{req.snapshot_key}' not found"
        )

    try:
        payload = json.loads(snapshot.state_blob)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deserialize snapshot state: {str(e)}"
        )

    metadata = payload.get("metadata", {})
    model_state = payload.get("model_state", payload)
    target_algo = req.target_algorithm or metadata.get("algorithm") or req.snapshot_key.split(":")[-1]

    valid_algos = ["linucb", "discounted_linucb", "linucb_thompson", "epsilon_greedy"]
    if target_algo not in valid_algos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Target algorithm '{target_algo}' is invalid. Must be one of {valid_algos}"
        )

    # Load state into active in-memory model
    model = model_store.get_model(target_algo)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model store has no registered policy '{target_algo}'"
        )
    model.load_state(model_state)

    # Persist into active model slot in database
    active_row = db.query(ModelState).filter(ModelState.algorithm == target_algo).first()
    now = datetime.now(timezone.utc)
    if not active_row:
        active_row = ModelState(
            algorithm=target_algo,
            state_blob=json.dumps(model_state),
            n_updates=model.n_updates,
            updated_at=now
        )
        db.add(active_row)
    else:
        active_row.state_blob = json.dumps(model_state)
        active_row.n_updates = model.n_updates
        active_row.updated_at = now

    # Write audit log to model_history
    audit_entry = ModelHistory(
        algorithm=target_algo,
        action="apply_warm_start",
        snapshot_key=req.snapshot_key,
        origin="warm_start",
        metadata_json=metadata,
        applied_by=current_admin.email,
        created_at=now
    )
    db.add(audit_entry)
    db.commit()

    return {
        "status": "success",
        "message": f"Successfully applied warm-start snapshot '{req.snapshot_key}' to active '{target_algo}'",
        "target_algorithm": target_algo,
        "snapshot_key": req.snapshot_key,
        "n_updates": model.n_updates,
        "applied_by": current_admin.email,
        "applied_at": now.isoformat()
    }

