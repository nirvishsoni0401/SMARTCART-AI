import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.model_store import model_store

client = TestClient(app)

def test_recommendations_cold_start():
    # Cold start query with no user authentication
    res = client.get("/recommendations?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert len(data["recommendations"]) == 10
    assert data["cold_start"] is True
    assert data["latency_ms"] < 250.0  # target under 100-250 ms in test environment
    assert all("explanation" in rec for rec in data["recommendations"])
    assert all(rec["explanation"] for rec in data["recommendations"])

def test_diversity_reranking_constraint():
    # Ensure maximum 3 items per category in top-10
    res = client.get("/recommendations?limit=10")
    assert res.status_code == 200
    recs = res.json()["recommendations"]
    
    cat_counts = {}
    for r in recs:
        c = r["product"]["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    for cat, count in cat_counts.items():
        assert count <= 3, f"Category '{cat}' appeared {count} times (max allowed 3)"

def test_event_feedback_and_deduplication():
    # 1. Get recommendation to obtain a valid recommendation_log_id
    rec_res = client.get("/recommendations?limit=5")
    assert rec_res.status_code == 200
    first_item = rec_res.json()["recommendations"][0]
    prod_id = first_item["product"]["id"]
    log_id = first_item["log_id"]

    model = model_store.get_model("linucb")
    initial_updates = model.n_updates

    # 2. Record click event (+2 reward)
    ev_res = client.post("/events", json={
        "product_id": prod_id,
        "event_type": "click",
        "recommendation_log_id": log_id
    })
    assert ev_res.status_code == 201
    assert model.n_updates == initial_updates + 1

    # 3. Duplicate event with same (log_id, event_type) must be deduplicated
    ev_dup = client.post("/events", json={
        "product_id": prod_id,
        "event_type": "click",
        "recommendation_log_id": log_id
    })
    assert ev_dup.status_code == 201
    assert model.n_updates == initial_updates + 1, "Duplicate event must not double-update the model"

    # 4. Record dislike event (-5 reward) on a different product
    dislike_res = client.post("/events", json={
        "product_id": 2,
        "event_type": "dislike"
    })
    assert dislike_res.status_code == 201
    assert model.n_updates == initial_updates + 2

def test_consent_declined_fallback():
    # 1. Register a user with consent_tracking = False
    import time
    email = f"noconsent_{int(time.time())}@example.com"
    reg = client.post("/auth/register", json={
        "name": "Private User",
        "email": email,
        "password": "password123",
        "consent_tracking": False
    })
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Request recommendations
    res = client.get("/recommendations?limit=5", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["algorithm"] == "popularity", "Declined consent must strictly fall back to popularity"
    assert all(r["log_id"] is None for r in data["recommendations"]), "No recommendation logs should be saved when consent is declined"

