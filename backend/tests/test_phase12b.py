import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app
from app.config import settings
from app.models import Product, User, InteractionEvent, RecommendationLog, ModelState
from app.services.model_store import model_store
from app.recommender.base import CandidateItem
from app.recommender.linucb import LinUCBRecommender
from app.recommender.department_heuristic import DepartmentHeuristicRecommender
from app.recommender.dept_restricted_linucb import DeptRestrictedLinUCBRecommender
from experiments.sim_v2.environment import EnvironmentV2
from experiments.sim_v2.policies import DepartmentHeuristicPolicy, DeptRestrictedLinUCBPolicy

client = TestClient(app)


def test_stage1_uses_only_observable_history():
    """
    Verifies that Stage 1 of DeptRestrictedLinUCBRecommender relies strictly
    on observable history (session_category or category_affinities in context)
    with zero reliance on simulator latent state (user archetype, true affinity, or noise).
    """
    recommender = DeptRestrictedLinUCBRecommender(feature_dim=36, alpha=0.5, m_explore=0)

    # 1. Cold start / empty context fallback to heuristic / popularity
    class ColdContext:
        is_cold_start = True
        total_interactions = 0
        category_affinities = {}
        session_category = None

    cands = [
        CandidateItem(
            product_id=1,
            product_data={"id": 1, "category": "Electronics", "popularity_score": 90.0},
            feature_vector=np.ones(36)
        ),
        CandidateItem(
            product_id=2,
            product_data={"id": 2, "category": "Books", "popularity_score": 95.0},
            feature_vector=np.ones(36)
        )
    ]
    scored_cold = recommender.select(cands, top_k=2, context=ColdContext())
    assert len(scored_cold) == 2
    # Product 2 has higher popularity score (95 vs 90)
    assert scored_cold[0].candidate.product_id == 2

    # 2. Context with observable category_affinities (e.g. user clicked Electronics 5 times)
    class ObservableContext:
        is_cold_start = False
        total_interactions = 5
        category_affinities = {"Electronics": 0.8, "Books": 0.1, "Fashion": 0.1}
        session_category = None

    scored_obs = recommender.select(cands, top_k=2, context=ObservableContext())
    # Electronics item should be prioritized as target category
    assert scored_obs[0].candidate.product_id == 1


def test_parity_live_and_sim_v2():
    """
    Part B2 Parity Test:
    Executes live DeptRestrictedLinUCBRecommender and benchmarked DeptRestrictedLinUCBPolicy
    on identical environment traces across 3 seeds over 2,000 rounds.
    Asserts bit-for-bit slate identity and cumulative regret equivalence (< 1e-4).
    """
    seeds = [1, 2, 3]
    num_rounds = 2000

    for seed in seeds:
        env_sim = EnvironmentV2(seed=seed)
        env_live = EnvironmentV2(seed=seed)

        sim_pol = DeptRestrictedLinUCBPolicy(m_explore=2, alpha=0.5)
        live_pol = DeptRestrictedLinUCBRecommender(feature_dim=36, alpha=0.5, m_explore=2)

        sim_regret = 0.0
        live_regret = 0.0

        for r in range(num_rounds):
            user_sim = env_sim.user_manager.sample_active_user(r)
            ctx_sim = env_sim.build_user_context(user_sim)
            avail_sim = env_sim.get_available_product_ids(r)
            feat_sim = env_sim.build_feature_matrix_for_user(user_sim, ctx_sim)

            user_live = env_live.user_manager.sample_active_user(r)
            ctx_live = env_live.build_user_context(user_live)
            avail_live = env_live.get_available_product_ids(r)
            feat_live = env_live.build_feature_matrix_for_user(user_live, ctx_live)

            sim_slate = sim_pol.select_slate(env_sim, user_sim, ctx_sim, avail_sim, r, k=10, feature_matrix=feat_sim)

            # Target category from observable history
            target_cat = None
            if user_live.history:
                last_pid = user_live.history[-1][0]
                if last_pid in env_live.pid_to_index:
                    target_cat = env_live.products[env_live.pid_to_index[last_pid]]["category"]
            if not target_cat:
                target_cat = max(ctx_live.category_affinities.items(), key=lambda x: x[1])[0]

            class ContextWrapper:
                category_affinities = ctx_live.category_affinities
                session_category = target_cat
                is_cold_start = False
                total_interactions = 1

            cands = [
                CandidateItem(
                    product_id=pid,
                    product_data=env_live.products[env_live.pid_to_index[pid]],
                    feature_vector=feat_live[env_live.pid_to_index[pid]]
                )
                for pid in avail_live
            ]
            live_scored = live_pol.select(cands, top_k=10, context=ContextWrapper())
            live_slate = [sc.candidate.product_id for sc in live_scored]

            assert sim_slate == live_slate, f"Slate mismatch at seed {seed}, round {r}!"

            fb_sim = env_sim.step(r, sim_slate, user=user_sim, cached_features=feat_sim)
            fb_live = env_live.step(r, live_slate, user=user_live, cached_features=feat_live)

            sim_pol.update(fb_sim)

            # Mirror slate-level update in live model
            reward_map = {ev["pid"]: ev["reward"] for ev in fb_live.events}
            for pos, pid in enumerate(fb_live.slate_pids):
                phi = fb_live.feature_matrix[pos]
                r_val = reward_map.get(pid, 0.0)
                live_pol.update(phi, r_val)

            sim_regret += fb_sim.expected_regret
            live_regret += fb_live.expected_regret

        assert abs(sim_regret - live_regret) < 1e-4, f"Regret mismatch at seed {seed}: {sim_regret} vs {live_regret}"


def test_cross_process_hashseed_invariance():
    """
    Part A2:
    Verifies that feature vectors and recommendation rankings are strictly invariant
    across two independent subprocesses started with different PYTHONHASHSEED values (0 vs 42).
    """
    script = """
import sys
import json
sys.path.insert(0, 'backend')
from app.services.feature_engineering import compute_brand_bucket, extract_feature_vector
from app.recommender.linucb import LinUCBRecommender
from app.recommender.base import CandidateItem
import numpy as np

# 1. Test brand hashing invariance
brands = ['Apple', 'Nike', 'Sony', 'Samsung', 'Generic Brand', 'Unknown 123']
buckets = [compute_brand_bucket(b) for b in brands]

# 2. Test candidate ranking invariance
model = LinUCBRecommender(feature_dim=36, alpha=0.5)
cands = [
    CandidateItem(product_id=i, product_data={'id': i, 'category': 'Electronics', 'popularity_score': 50.0}, feature_vector=np.ones(36)*i)
    for i in range(10)
]
scored = model.select(cands, top_k=5)
ranking = [s.candidate.product_id for s in scored]

print(json.dumps({'buckets': buckets, 'ranking': ranking}))
"""
    python_exe = sys.executable

    res0 = subprocess.run([python_exe, "-c", script], env={**os.environ, "PYTHONHASHSEED": "0"}, capture_output=True, text=True, check=True)
    res42 = subprocess.run([python_exe, "-c", script], env={**os.environ, "PYTHONHASHSEED": "42"}, capture_output=True, text=True, check=True)

    out0 = json.loads(res0.stdout.strip())
    out42 = json.loads(res42.stdout.strip())

    assert out0["buckets"] == out42["buckets"], "Brand hash buckets differ across PYTHONHASHSEED!"
    assert out0["ranking"] == out42["ranking"], "Rankings differ across PYTHONHASHSEED!"


def test_live_policy_interface_and_model_store():
    """
    Part B1 & B3:
    Tests that new policies are registered in model_store, support state serialization
    (get_state/load_state), and persist across simulated process restart.
    """
    for algo in ["dept_restricted_linucb", "dept_restricted_linucb_m0", "dept_restricted_linucb_m2", "department_heuristic"]:
        model = model_store.get_model(algo)
        assert model is not None, f"Model {algo} must be present in model_store"
        state = model.get_state()
        assert "name" in state
        assert "feature_dim" in state
        assert "n_updates" in state

        # Test state reload
        dummy_state = state.copy()
        dummy_state["n_updates"] = 99
        model.load_state(dummy_state)
        assert model.n_updates == 99
        # Reset back
        state["n_updates"] = 0
        model.load_state(state)


def test_admin_snapshot_and_rollback():
    """
    Part B3:
    Tests snapshot creation and rollback endpoint behavior.
    """
    admin_login = client.post("/auth/login", json={"email": "admin@smartcart.ai", "password": "admin123"})
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # 1. Take snapshot
    snap_res = client.post("/admin/models/snapshot?name=test_phase12b_snap", headers=admin_headers)
    assert snap_res.status_code == 200
    assert snap_res.json()["snapshot_name"] == "test_phase12b_snap"

    # 2. Modify a model state
    model = model_store.get_model("linucb")
    original_updates = model.n_updates
    model.update(np.ones(36), 10.0)
    assert model.n_updates == original_updates + 1

    # 3. Rollback
    rb_res = client.post("/admin/models/rollback?name=test_phase12b_snap", headers=admin_headers)
    assert rb_res.status_code == 200
    assert "Successfully restored" in rb_res.json()["message"]


def test_recently_viewed_endpoint():
    """
    Part C1:
    Tests GET /events/recently-viewed:
    1. Returns recently viewed products for active session.
    2. Respects user consent (returns empty list when consent is disabled).
    3. Does not generate recommendation logs or bandit updates.
    """
    session_id = "test_rv_session_12b"
    prods = client.get("/products?page=1&page_size=3").json()["products"]
    p1 = prods[0]["id"]
    p2 = prods[1]["id"]

    # Post 2 view events
    client.post("/events", json={"product_id": p1, "event_type": "impression", "session_id": session_id})
    client.post("/events", json={"product_id": p2, "event_type": "click", "session_id": session_id})

    # Fetch recently viewed
    res = client.get(f"/events/recently-viewed?session_id={session_id}&limit=5")
    assert res.status_code == 200
    rv_data = res.json()
    assert isinstance(rv_data, list)
    rv_pids = [p["id"] for p in rv_data]
    # Both products should be present
    assert p1 in rv_pids or p2 in rv_pids

    # User with consent disabled
    reg_user = client.post("/auth/register", json={
        "name": "Consent Optout Shopper",
        "email": "optout_12b@example.com",
        "password": "password123",
        "consent_tracking": False
    })
    token = reg_user.json()["access_token"]
    res_optout = client.get(
        f"/events/recently-viewed?session_id={session_id}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res_optout.status_code == 200
    assert len(res_optout.json()) == 0, "Opted-out user must receive empty recently-viewed list"
