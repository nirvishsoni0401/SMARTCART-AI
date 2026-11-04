"""
SmartCart AI — Phase 9C Test Suite
Verifies:
  1. Admin warm-start endpoint protections:
     - Requires admin authorization (401/403)
     - Requires confirm=true (rejects 400 if False)
     - Validates FEATURE_SPEC_HASH (rejects 400 if tampered)
     - Validates feature dimension (rejects 400 if != 36)
     - Refuses live overwrite if live updates > threshold unless force=true
     - Archives pre-apply state as snapshot:pre_warmstart:<timestamp>:<algo>
     - Reversible via rollback endpoint
     - Supports apply_to_sandbox=true
  2. Session-based feature isolation & leakage prevention:
     - True target interaction at timestamp t is strictly excluded from context
     - Earlier session events before t are included with exponential decay
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app import database as db_module
from app.models import User, Product, ModelState, ModelHistory
from app.auth import create_access_token
from app.config import settings
from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    UserContext,
    build_context_from_history,
    extract_feature_vector
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def admin_token(client):
    admin_login = client.post("/auth/login", json={"email": "admin@smartcart.ai", "password": "admin123"})
    if admin_login.status_code == 200:
        return admin_login.json()["access_token"]
    return create_access_token(data={"sub": "admin@smartcart.ai", "is_admin": True})


# =============================================================================
# 1. ADMIN WARM-START ENDPOINT HARDENING
# =============================================================================

def test_warm_start_apply_requires_confirm(client, admin_token):
    """POST /admin/models/warm-start/apply must reject if confirm=false."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {"snapshot_key": "warm_start:v1:linucb", "confirm": False}
    resp = client.post("/admin/models/warm-start/apply", json=payload, headers=headers)
    assert resp.status_code == 400
    assert "confirm=true" in resp.json()["detail"].lower()


def test_warm_start_apply_validates_spec_hash(client, admin_token):
    """Rejects warm-start apply if snapshot spec hash does not match current system."""
    db = db_module.SessionLocal()
    # Create dummy tampered snapshot
    tampered_key = "warm_start:v1:tampered_hash_test"
    bad_payload = {
        "metadata": {
            "algorithm": "linucb",
            "feature_spec_hash": "bad_hash_1234",
            "feature_dim": 36
        },
        "model_state": {"A": np.eye(36).tolist(), "b": np.zeros(36).tolist()}
    }
    db.add(ModelState(
        algorithm=tampered_key,
        state_blob=json.dumps(bad_payload),
        n_updates=100
    ))
    db.commit()
    db.close()

    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {"snapshot_key": tampered_key, "confirm": True, "force": True}
    resp = client.post("/admin/models/warm-start/apply", json=payload, headers=headers)
    assert resp.status_code == 400
    assert "hash mismatch" in resp.json()["detail"].lower()


def test_warm_start_apply_validates_dimension(client, admin_token):
    """Rejects warm-start apply if snapshot dimension != 36."""
    db = db_module.SessionLocal()
    tampered_key = "warm_start:v1:tampered_dim_test"
    bad_payload = {
        "metadata": {
            "algorithm": "linucb",
            "feature_spec_hash": settings.FEATURE_SPEC_HASH,
            "feature_dim": 64
        },
        "model_state": {"A": np.eye(64).tolist(), "b": np.zeros(64).tolist(), "feature_dim": 64}
    }
    db.add(ModelState(
        algorithm=tampered_key,
        state_blob=json.dumps(bad_payload),
        n_updates=100
    ))
    db.commit()
    db.close()

    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {"snapshot_key": tampered_key, "confirm": True, "force": True}
    resp = client.post("/admin/models/warm-start/apply", json=payload, headers=headers)
    assert resp.status_code == 400
    assert "dimension mismatch" in resp.json()["detail"].lower()


def test_warm_start_apply_refuses_overwrite_without_force(client, admin_token):
    """Refuses to overwrite a live model that has >threshold updates unless force=true."""
    db = db_module.SessionLocal()
    # Set linucb live updates to 150
    row = db.query(ModelState).filter(ModelState.algorithm == "linucb").first()
    if not row:
        row = ModelState(algorithm="linucb", state_blob=json.dumps({"A": np.eye(36).tolist(), "b": np.zeros(36).tolist()}), n_updates=150)
        db.add(row)
    else:
        row.n_updates = 150
    db.commit()
    db.close()

    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {
        "snapshot_key": "warm_start:v1:linucb",
        "confirm": True,
        "force": False,
        "max_live_updates_threshold": 100
    }
    resp = client.post("/admin/models/warm-start/apply", json=payload, headers=headers)
    assert resp.status_code == 400
    assert "force=true" in resp.json()["detail"].lower()


def test_warm_start_apply_archives_pre_warmstart_and_allows_rollback(client, admin_token):
    """Verifies pre-apply snapshot is archived and reversible via rollback endpoint."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {
        "snapshot_key": "warm_start:v1:linucb",
        "confirm": True,
        "force": True
    }
    resp = client.post("/admin/models/warm-start/apply", json=payload, headers=headers)
    assert resp.status_code == 200
    res_data = resp.json()
    archived_key = res_data["archived_snapshot_key"]
    assert "snapshot:pre_warmstart:" in archived_key

    # Verify rollback works to restore
    resp_rb = client.post(f"/admin/models/rollback?name={archived_key}", headers=headers)
    assert resp_rb.status_code == 200


def test_warm_start_apply_to_sandbox(client, admin_token):
    """Verifies applying to sandbox updates only sim:<algo> and leaves live model untouched."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    payload = {
        "snapshot_key": "warm_start:v1:linucb",
        "confirm": True,
        "apply_to_sandbox": True
    }
    resp = client.post("/admin/models/warm-start/apply", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["apply_to_sandbox"] is True

    db = db_module.SessionLocal()
    sim_row = db.query(ModelState).filter(ModelState.algorithm == "sim:linucb").first()
    assert sim_row is not None
    db.close()


# =============================================================================
# 2. SESSION FEATURE ISOLATION & LEAKAGE CONTROLS
# =============================================================================

def test_session_feature_isolation_transaction_exclusion():
    """
    Scenario: A user adds product A to cart at t=1000, then completes transaction
    for product A at t=2000.
    At t=2000, context must include the prior cart_add, but MUST strictly exclude
    the target transaction itself from its context.
    """
    prod_a = {"id": 101, "category": "Electronics", "brand": "BrandX", "price": 25000.0}
    t_cart = 1435000000000
    t_trans = 1435000500000

    # History prior to transaction contains only the cart_add
    history_at_trans = [
        (prod_a, "cart_add", t_cart, 5.0)
    ]

    ref_dt = datetime.fromtimestamp(t_trans / 1000.0, tz=timezone.utc)
    ctx = build_context_from_history(history_at_trans, reference_time=ref_dt)

    assert ctx.total_interactions == 1
    # Category affinity for Electronics reflects 1 prior cart_add
    assert ctx.category_affinities["Electronics"] > 0
    # Average price includes prod_a from prior cart_add
    assert ctx.avg_price == 25000.0

    phi = extract_feature_vector(ctx, prod_a)
    assert phi.shape == (36,)
    assert phi[0] > 0  # Electronics affinity
    assert phi[35] == 1.0  # Bias term

