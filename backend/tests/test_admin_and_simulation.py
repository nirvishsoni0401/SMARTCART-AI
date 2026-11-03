import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_admin_route_protection():
    # Anonymous request to update algorithm must be rejected 401
    res = client.put("/admin/algorithms", json={"algorithm": "linucb"})
    assert res.status_code == 401

    # Regular user request to update algorithm must be forbidden 403
    user_login = client.post("/auth/login", json={"email": "rahul@smartcart.ai", "password": "user123"})
    user_token = user_login.json()["access_token"]
    res_user = client.put("/admin/algorithms", json={"algorithm": "linucb"}, headers={"Authorization": f"Bearer {user_token}"})
    assert res_user.status_code == 403

def test_admin_metrics_and_simulation():
    # Login as admin
    admin_login = client.post("/auth/login", json={"email": "admin@smartcart.ai", "password": "admin123"})
    assert admin_login.status_code == 200
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Run simulated traffic (5 rounds)
    sim_res = client.post("/admin/simulate", json={"num_rounds": 5}, headers=admin_headers)
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    assert sim_data["rounds_executed"] == 5
    assert sim_data["total_events_generated"] > 0

    # 2. Get metrics
    met_res = client.get("/admin/metrics?window=all")
    assert met_res.status_code == 200
    met_data = met_res.json()
    assert "algorithms" in met_data
    assert len(met_data["algorithms"]) == 6

    # 3. Get and update reward config
    cfg_res = client.get("/admin/reward-config")
    assert cfg_res.status_code == 200
    assert cfg_res.json()["purchase"] == 10.0

    put_cfg = client.put("/admin/reward-config", json={"purchase": 12.0}, headers=admin_headers)
    assert put_cfg.status_code == 200
    assert put_cfg.json()["rewards"]["purchase"] == 12.0

    # Revert purchase reward to 10.0
    client.put("/admin/reward-config", json={"purchase": 10.0}, headers=admin_headers)

    # 4. Get recommendation logs (check origin=all to include simulated / legacy logs)
    logs_res = client.get("/admin/logs?limit=10&origin=all")
    assert logs_res.status_code == 200
    assert len(logs_res.json()) > 0

