from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from app.database import (
    users_collection,
    products_collection,
    orders_collection,
    events_collection,
    recommendations_collection
)
from app.recommendation.policy_manager import policy_manager
from app.recommendation.simulation import SimulationEnvironment

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


class SimulationRequest(BaseModel):
    steps: int = 150
    seed: int = 42


@router.get("/overview")
def get_analytics_overview():
    total_customers = users_collection.count_documents({})
    total_products = products_collection.count_documents({"is_active": True})
    total_orders = orders_collection.count_documents({})
    total_events = events_collection.count_documents({})
    total_recs = recommendations_collection.count_documents({})

    clicks = events_collection.count_documents({"event_type": "product_click"})
    dislikes = events_collection.count_documents({"event_type": "explicit_dislike"})
    purchases = orders_collection.count_documents({})

    bandit_stats = policy_manager.get_stats()

    ctr = round((clicks / max(1, total_events)) * 100, 2)
    conversion_rate = round((purchases / max(1, total_events)) * 100, 2)
    negative_rate = round((dislikes / max(1, total_events)) * 100, 2)

    return {
        "total_customers": total_customers,
        "total_products": total_products,
        "total_orders": total_orders,
        "total_interactions": total_events,
        "total_recommendations_served": total_recs,
        "click_through_rate": ctr,
        "conversion_rate": conversion_rate,
        "negative_feedback_rate": negative_rate,
        "bandit": bandit_stats
    }


@router.get("/rewards")
def get_rewards_analytics():
    bandit_stats = policy_manager.get_stats()
    
    # Aggregate recent events by day or type
    pipeline = [
        {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    event_distribution = list(events_collection.aggregate(pipeline))

    return {
        "cumulative_reward": bandit_stats["cumulative_reward"],
        "average_reward": bandit_stats["average_reward"],
        "total_policy_updates": bandit_stats["total_updates"],
        "reward_events": bandit_stats["recent_history"],
        "event_distribution": event_distribution
    }


@router.get("/engagement")
def get_engagement_analytics():
    # Category popularity from events
    cat_pipeline = [
        {"$match": {"category": {"$ne": None}}},
        {"$group": {"_id": "$category", "interactions": {"$sum": 1}}},
        {"$sort": {"interactions": -1}},
        {"$limit": 10}
    ]
    popular_categories = list(events_collection.aggregate(cat_pipeline))

    # Event types breakdown
    type_pipeline = [
        {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    event_breakdown = list(events_collection.aggregate(type_pipeline))

    return {
        "popular_categories": popular_categories,
        "event_types": event_breakdown
    }


@router.get("/comparison")
def get_algorithm_comparison():
    sim = SimulationEnvironment(seed=42)
    benchmark = sim.run_experiment(num_steps=120, k=4)
    return benchmark


@router.post("/simulate")
def run_custom_simulation(req: SimulationRequest):
    steps = min(500, max(20, req.steps))
    sim = SimulationEnvironment(seed=req.seed)
    return sim.run_experiment(num_steps=steps, k=4)

