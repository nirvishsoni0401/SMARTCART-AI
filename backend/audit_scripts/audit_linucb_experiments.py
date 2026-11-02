import sys
import uuid
import numpy as np
from pathlib import Path
from collections import Counter

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.database import (
    users_collection,
    events_collection,
    recommendations_collection,
    products_collection,
    bandit_models_collection,
)
from app.recommendation.context_builder import build_customer_context
from app.recommendation.candidate_generator import generate_candidates
from app.recommendation.feature_engineering import extract_features
from app.recommendation.policy_manager import policy_manager, ACTIVE_MODEL_VERSION
from app.recommendation.model_storage import load_bandit_model, save_bandit_model
from app.recommendation.recommendation_service import get_recommendations_for_user, process_interaction_event

def run_linucb_experiments():
    print("===================================================================")
    print("           3. LINUCB RECOMMENDATION ENGINE AUDIT & EXPERIMENTS     ")
    print("===================================================================")

    # -------------------------------------------------------------
    # EXPERIMENT A: Cold-Start User
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENT A: Cold-Start User (No prior history) <<<")
    cold_user_id = f"cold_user_{uuid.uuid4().hex[:8]}"
    cold_session_id = f"sess_{uuid.uuid4().hex[:8]}"
    
    # Context generation check
    ctx_cold = build_customer_context(user_id=cold_user_id, session_id=cold_session_id)
    print(f"  Context generated:")
    print(f"    - total_interactions: {ctx_cold['total_interactions']}")
    print(f"    - dominant_category:  {ctx_cold['dominant_category']}")
    print(f"    - budget_level:       {ctx_cold['budget_level']}")
    print(f"    - avg_viewed_price:   Rs. {ctx_cold['average_viewed_price']}")
    
    # Recommendation generation
    rec_cold = get_recommendations_for_user(user_id=cold_user_id, session_id=cold_session_id, limit=8)
    items_cold = rec_cold["recommendations"]
    cats_cold = [it["category"] for it in items_cold]
    ucb_scores_cold = [it["ucb_score"] for it in items_cold]
    bonuses_cold = [it["exploration_bonus"] for it in items_cold]
    exp_rewards_cold = [it["expected_reward"] for it in items_cold]
    
    print(f"  Cold-start recommendations (8 items):")
    print(f"    - Categories represented: {set(cats_cold)} ({len(set(cats_cold))} distinct departments)")
    print(f"    - Mean exploration bonus: {np.mean(bonuses_cold):.4f}")
    print(f"    - Mean expected reward:   {np.mean(exp_rewards_cold):.4f}")
    print(f"    - Mean UCB score:         {np.mean(ucb_scores_cold):.4f}")
    exp_a_verified = ctx_cold["total_interactions"] == 0 and len(items_cold) == 8 and np.mean(bonuses_cold) > 0

    # -------------------------------------------------------------
    # EXPERIMENT B: Repeated Positive Interactions with Electronics
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENT B: Positive Interactions Adaptation (Electronics) <<<")
    user_b_id = f"user_elec_{uuid.uuid4().hex[:8]}"
    session_b_id = f"sess_elec_{uuid.uuid4().hex[:8]}"
    
    # Baseline recommendations before interactions
    rec_b_before = get_recommendations_for_user(user_id=user_b_id, session_id=session_b_id, limit=8)["recommendations"]
    elec_before_cnt = sum(1 for it in rec_b_before if it["category"].lower() == "electronics")
    print(f"  Before interaction:")
    print(f"    - Electronics in top 8: {elec_before_cnt} / 8 ({elec_before_cnt/8*100:.1f}%)")

    # Simulate 5 positive interactions on Electronics
    elec_prods = list(products_collection.find({"category": "Electronics", "is_active": True}).limit(5))
    for p in elec_prods:
        pid = p["product_id"]
        # Track click
        ev_click = {
            "user_id": user_b_id,
            "session_id": session_b_id,
            "event_type": "product_click",
            "product_id": pid,
            "category": "Electronics",
            "metadata": {"price": p["discounted_price"]}
        }
        events_collection.insert_one(ev_click)
        process_interaction_event(ev_click)

    # Track 1 add-to-cart on an electronics product
    ev_cart = {
        "user_id": user_b_id,
        "session_id": session_b_id,
        "event_type": "add_to_cart",
        "product_id": elec_prods[0]["product_id"],
        "category": "Electronics",
        "metadata": {"price": elec_prods[0]["discounted_price"]}
    }
    events_collection.insert_one(ev_cart)
    process_interaction_event(ev_cart)

    # Context after interaction
    ctx_b_after = build_customer_context(user_id=user_b_id, session_id=session_b_id)
    rec_b_after = get_recommendations_for_user(user_id=user_b_id, session_id=session_b_id, limit=8)["recommendations"]
    elec_after_cnt = sum(1 for it in rec_b_after if it["category"].lower() == "electronics")

    print(f"  After 6 positive Electronics interactions (5 clicks + 1 cart):")
    print(f"    - Updated dominant category: {ctx_b_after['dominant_category']}")
    print(f"    - Electronics category affinity: {ctx_b_after['category_affinities'].get('electronics')}")
    print(f"    - Electronics in top 8: {elec_after_cnt} / 8 ({elec_after_cnt/8*100:.1f}%)")
    print(f"    - Sample explanation: '{rec_b_after[0]['explanation']}'")
    exp_b_verified = ctx_b_after["dominant_category"] == "electronics" and elec_after_cnt > elec_before_cnt

    # -------------------------------------------------------------
    # EXPERIMENT C: Negative Feedback Exposure Reduction
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENT C: Negative Feedback Adaptation <<<")
    user_c_id = f"user_neg_{uuid.uuid4().hex[:8]}"
    session_c_id = f"sess_neg_{uuid.uuid4().hex[:8]}"
    
    # Establish interest in Books first
    book_prods = list(products_collection.find({"category": "Books", "is_active": True}).limit(5))
    for p in book_prods:
        ev = {
            "user_id": user_c_id,
            "session_id": session_c_id,
            "event_type": "product_click",
            "product_id": p["product_id"],
            "category": "Books",
            "metadata": {"price": p["discounted_price"]}
        }
        events_collection.insert_one(ev)
        process_interaction_event(ev)
    
    ctx_c_before = build_customer_context(user_id=user_c_id, session_id=session_c_id)
    rec_c_before = get_recommendations_for_user(user_id=user_c_id, session_id=session_c_id, limit=8)["recommendations"]
    books_before_cnt = sum(1 for it in rec_c_before if it["category"].lower() == "books")
    print(f"  Before negative feedback (user interested in Books):")
    print(f"    - Dominant category: {ctx_c_before['dominant_category']}")
    print(f"    - Books affinity: {ctx_c_before['category_affinities'].get('books')}")
    print(f"    - Books in top 8: {books_before_cnt} / 8 ({books_before_cnt/8*100:.1f}%)")

    # Now apply explicit dislike feedback on Books
    for p in book_prods[:3]:
        ev_dislike = {
            "user_id": user_c_id,
            "session_id": session_c_id,
            "event_type": "explicit_dislike",
            "product_id": p["product_id"],
            "category": "Books",
            "metadata": {"price": p["discounted_price"]}
        }
        events_collection.insert_one(ev_dislike)
        process_interaction_event(ev_dislike)

    ctx_c_after = build_customer_context(user_id=user_c_id, session_id=session_c_id)
    rec_c_after = get_recommendations_for_user(user_id=user_c_id, session_id=session_c_id, limit=8)["recommendations"]
    books_after_cnt = sum(1 for it in rec_c_after if it["category"].lower() == "books")

    print(f"  After 3 explicit dislikes on Books:")
    print(f"    - Updated Books affinity: {ctx_c_after['category_affinities'].get('books')}")
    print(f"    - Books in top 8: {books_after_cnt} / 8 ({books_after_cnt/8*100:.1f}%)")
    exp_c_verified = books_after_cnt < books_before_cnt or ctx_c_after['category_affinities'].get('books', 0) < ctx_c_before['category_affinities'].get('books', 0)

    # -------------------------------------------------------------
    # EXPERIMENT D: Model Persistence & Restoration Across Restarts
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENT D: Persistence and Restoration Check <<<")
    agent = policy_manager.get_agent()
    # Force save current agent state
    save_bandit_model(agent, version=ACTIVE_MODEL_VERSION, metadata={"test": "persistence_audit"})
    
    # Verify in MongoDB
    saved_doc = bandit_models_collection.find_one({"version": ACTIVE_MODEL_VERSION})
    doc_exists = saved_doc is not None
    arms_in_db = len(saved_doc.get("arms", {})) if doc_exists else 0
    
    # Reload fresh instance from DB
    reloaded_agent = load_bandit_model(ACTIVE_MODEL_VERSION)
    
    # Check arm parameter equality
    arms_match = True
    for arm_id, arm_data in agent.arms.items():
        if arm_id not in reloaded_agent.arms:
            arms_match = False
            break
        a_orig = arm_data["A"]
        a_reload = reloaded_agent.arms[arm_id]["A"]
        b_orig = arm_data["b"]
        b_reload = reloaded_agent.arms[arm_id]["b"]
        if not np.allclose(a_orig, a_reload) or not np.allclose(b_orig, b_reload):
            arms_match = False
            break

    print(f"  Bandit model document in MongoDB Atlas/Local: {'FOUND' if doc_exists else 'NOT FOUND'}")
    print(f"  Persisted arms count: {arms_in_db}")
    print(f"  Parameter matrix & vector exact match: {'VERIFIED (np.allclose = True)' if arms_match else 'FAILED'}")
    exp_d_verified = doc_exists and arms_match

    # -------------------------------------------------------------
    # EXPERIMENT E: LinUCB vs Non-Personalized Popularity Baseline
    # -------------------------------------------------------------
    print("\n>>> EXPERIMENT E: LinUCB vs Non-Personalized Popularity Baseline <<<")
    # Simulate 50 interaction steps with 5 distinct customer personas (Electronics lover, Fashion buyer, Book reader, Home improver, Budget shopper)
    np.random.seed(42)
    personas = [
        {"name": "Tech Enthusiast", "preferred": "Electronics", "budget": 45000},
        {"name": "Fashionista", "preferred": "Fashion", "budget": 4000},
        {"name": "Home Decorator", "preferred": "Home & Kitchen", "budget": 12000},
        {"name": "Avid Reader", "preferred": "Books", "budget": 1200},
        {"name": "Fitness Coach", "preferred": "Sports & Fitness", "budget": 9000},
    ]

    # Run simulation comparing LinUCB vs Popularity
    from app.recommendation.simulation import SimulationEnvironment
    sim_env = SimulationEnvironment(seed=42)
    sim_results = sim_env.run_experiment(num_steps=50)
    
    linucb_cum_reward = sim_results["results"]["linucb"]["cumulative_reward"]
    pop_cum_reward = sim_results["results"]["popularity"]["cumulative_reward"]
    eps_cum_reward = sim_results["results"]["epsilon_greedy"]["cumulative_reward"]

    linucb_avg_reward = sim_results["results"]["linucb"]["average_reward"]
    pop_avg_reward = sim_results["results"]["popularity"]["average_reward"]
    eps_avg_reward = sim_results["results"]["epsilon_greedy"]["average_reward"]

    print(f"  Evaluation Period: 50 Simulated Interaction Steps")
    print(f"  Baseline (Popularity):   Cumulative Reward = {pop_cum_reward:.2f} | Avg/Step = {pop_avg_reward:.3f}")
    print(f"  Epsilon-Greedy:          Cumulative Reward = {eps_cum_reward:.2f} | Avg/Step = {eps_avg_reward:.3f}")
    print(f"  Adaptive Context LinUCB: Cumulative Reward = {linucb_cum_reward:.2f} | Avg/Step = {linucb_avg_reward:.3f}")
    
    lift_over_pop = ((linucb_cum_reward - pop_cum_reward) / max(1.0, pop_cum_reward)) * 100
    print(f"  LinUCB Cumulative Reward Improvement over Popularity: +{lift_over_pop:.1f}%")
    exp_e_verified = linucb_cum_reward >= pop_cum_reward

    print("\n-------------------------------------------------------------")
    print("EXPERIMENTS SUMMARY:")
    print(f"  Experiment A (Cold-Start):                 {'VERIFIED' if exp_a_verified else 'FAILED'}")
    print(f"  Experiment B (Positive Adaptation):        {'VERIFIED' if exp_b_verified else 'FAILED'}")
    print(f"  Experiment C (Negative Feedback Penalty):  {'VERIFIED' if exp_c_verified else 'FAILED'}")
    print(f"  Experiment D (Persistence & Restoration):  {'VERIFIED' if exp_d_verified else 'FAILED'}")
    print(f"  Experiment E (LinUCB vs Popularity Lift):  {'VERIFIED' if exp_e_verified else 'FAILED'}")
    print("-------------------------------------------------------------")

if __name__ == "__main__":
    run_linucb_experiments()
