import json
import concurrent.futures
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models import Product, Order, OrderItem, CartItem, WishlistItem, InteractionEvent, RecommendationLog, ModelState
from app.services.model_store import model_store
from app.recommender.linucb import LinUCBRecommender
from app.config import settings

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dummyjson_category_map_exhaustiveness():
    """Verify that every single category in DummyJSON is explicitly mapped or excluded with a reason."""
    cats_file = PROJECT_ROOT / "data" / "raw" / "dummyjson" / "categories.json"
    map_file = PROJECT_ROOT / "data" / "mappings" / "dummyjson_category_map.json"

    assert cats_file.exists(), "categories.json must exist"
    assert map_file.exists(), "dummyjson_category_map.json must exist"

    with open(cats_file, "r", encoding="utf-8") as f:
        categories = json.load(f)
    with open(map_file, "r", encoding="utf-8") as f:
        category_map = json.load(f)

    valid_departments = {"Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Books", "excluded"}

    for cat_item in categories:
        slug = cat_item["slug"]
        assert slug in category_map, f"Category slug '{slug}' is missing from dummyjson_category_map.json"
        entry = category_map[slug]
        assert "department" in entry
        assert entry["department"] in valid_departments, f"Invalid department for {slug}: {entry['department']}"
        assert "reason" in entry and len(entry["reason"].strip()) > 5, f"Missing rationale reason for category {slug}"


def test_import_idempotency_and_counts():
    """Running import_dummyjson_catalog twice must be idempotent and produce identical counts."""
    from scripts.import_dummyjson import import_dummyjson_catalog

    # Run import second time
    res = import_dummyjson_catalog(dry_run=False)
    assert res["imported_active"] == 157
    assert res["reviews_count"] == 0, "Second import must not duplicate existing product reviews"

    db = SessionLocal()
    try:
        active_count = db.query(Product).filter(Product.is_active == True).count()
        assert active_count == 262, f"Active catalog count must be 262 (157 DummyJSON + 105 Books), got {active_count}"
        inactive_count = db.query(Product).filter(Product.is_active == False).count()
        assert inactive_count == 525, f"Deactivated non-book seed items must be 525, got {inactive_count}"
    finally:
        db.close()


def test_price_conversion_and_mrp():
    """Verify USD to INR currency conversion and MRP derivation from discount percentage."""
    db = SessionLocal()
    try:
        sample = db.query(Product).filter(Product.source == "dummyjson").first()
        assert sample is not None
        assert sample.raw_price_usd is not None and sample.raw_price_usd > 0
        expected_inr = round(sample.raw_price_usd * settings.USD_TO_INR_RATE, 2)
        assert abs(sample.price - expected_inr) < 0.05, f"Price {sample.price} does not match converted {expected_inr}"
        assert sample.mrp >= sample.price, "MRP must be greater than or equal to selling price"
        if sample.discount_pct > 0:
            calc_mrp = round(sample.price / (1.0 - sample.discount_pct / 100.0), 2)
            assert abs(sample.mrp - calc_mrp) < 1.0
    finally:
        db.close()


def test_foreign_keys_intact_after_import():
    """Verify that orders, order_items, cart_items, and events have intact product references."""
    db = SessionLocal()
    try:
        for ord_item in db.query(OrderItem).all():
            assert ord_item.product is not None, f"OrderItem #{ord_item.id} has broken product FK"
        for cart_item in db.query(CartItem).all():
            assert cart_item.product is not None, f"CartItem #{cart_item.id} has broken product FK"
        for w_item in db.query(WishlistItem).all():
            assert w_item.product is not None, f"WishlistItem #{w_item.id} has broken product FK"
    finally:
        db.close()


def test_inactive_products_never_returned():
    """Verify that catalog search, product detail, and recommendations never serve inactive products."""
    # 1. Product catalog list
    res = client.get("/products?page=1&page_size=50")
    assert res.status_code == 200
    for p in res.json()["products"]:
        assert p["is_active"] is True

    # 2. Inactive product detail returns 404
    db = SessionLocal()
    try:
        inactive_prod = db.query(Product).filter(Product.is_active == False).first()
        if inactive_prod:
            det_res = client.get(f"/products/{inactive_prod.id}")
            assert det_res.status_code == 404
    finally:
        db.close()

    # 3. Recommendations never recommend inactive products
    rec_res = client.get("/recommendations?limit=15")
    assert rec_res.status_code == 200
    for rec in rec_res.json()["recommendations"]:
        assert rec["product"]["is_active"] is True


def test_api_response_shapes_unchanged():
    """Snapshot schema verification for core customer and admin endpoints."""
    # /products
    p_res = client.get("/products?page=1&page_size=5")
    assert p_res.status_code == 200
    p_data = p_res.json()
    for key in ["total", "page", "page_size", "total_pages", "products"]:
        assert key in p_data

    # /recommendations
    r_res = client.get("/recommendations?limit=5")
    assert r_res.status_code == 200
    r_data = r_res.json()
    for key in ["recommendations", "algorithm", "cold_start", "latency_ms"]:
        assert key in r_data
    if r_data["recommendations"]:
        item = r_data["recommendations"][0]
        for key in ["product", "score", "ucb_bonus", "was_exploration", "explanation", "algorithm"]:
            assert key in item

    # /admin/metrics
    m_res = client.get("/admin/metrics")
    assert m_res.status_code == 200
    m_data = m_res.json()
    for key in ["active_algorithm", "total_users", "total_catalog_products", "active_catalog_products", "algorithms", "latency_percentiles", "overall"]:
        assert key in m_data


def test_origin_filtering():
    """Verify origin-based filtering on admin metrics."""
    # Fetch metrics for live, simulated, legacy, all
    m_live = client.get("/admin/metrics?origin=live").json()
    m_sim = client.get("/admin/metrics?origin=simulated").json()
    m_legacy = client.get("/admin/metrics?origin=legacy").json()
    m_test = client.get("/admin/metrics?origin=test").json()
    m_all = client.get("/admin/metrics?origin=all").json()

    assert m_live["origin_filter"] == "live"
    assert m_sim["origin_filter"] == "simulated"
    assert m_legacy["origin_filter"] == "legacy"
    assert m_test["origin_filter"] == "test"
    assert m_all["origin_filter"] == "all"

    # All events count must equal sum of disjoint origin partitions
    ev_all = m_all["overall"]["total_events"]
    ev_sum = (
        m_live["overall"]["total_events"]
        + m_sim["overall"]["total_events"]
        + m_legacy["overall"]["total_events"]
        + m_test["overall"]["total_events"]
    )
    assert ev_all == ev_sum


def test_sandbox_isolation():
    """Verify simulated traffic writes ONLY origin=simulated data and does not touch live model states."""
    admin_login = client.post("/auth/login", json={"email": "admin@smartcart.ai", "password": "admin123"})
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    live_linucb = model_store.get_model("linucb", sandbox=False)
    live_updates_before = live_linucb.n_updates
    live_theta_before = live_linucb.get_theta().copy()

    # Run simulation with train_live=False (default)
    sim_res = client.post("/admin/simulate", json={"num_rounds": 5, "train_live": False}, headers=admin_headers)
    assert sim_res.status_code == 200
    assert sim_res.json()["sandbox_mode"] is True

    # Live model state must be completely untouched
    live_updates_after = live_linucb.n_updates
    live_theta_after = live_linucb.get_theta()
    assert live_updates_before == live_updates_after, "Live model updates must not change during sandbox simulation"
    assert np.array_equal(live_theta_before, live_theta_after), "Live model theta must not change during sandbox simulation"

    # Verify sandbox model was persisted with sim: prefix
    db = SessionLocal()
    try:
        sim_states = db.query(ModelState).filter(ModelState.algorithm.like("sim:%")).all()
        assert len(sim_states) > 0, "Sandbox model states must exist with 'sim:' prefix"
    finally:
        db.close()


def test_return_correction_math_and_no_double_return():
    """
    Mathematical verification of delayed return correction:
    1. A and A_inv must remain strictly unchanged.
    2. theta must update according to delta * A_inv * phi.
    3. The same return cannot be applied twice on an order item.
    """
    model = LinUCBRecommender(feature_dim=36, alpha=0.5)
    phi = np.ones(36, dtype=np.float64) / 6.0

    # 1. First initial purchase observation
    model.update(phi, reward=10.0)
    A_before = model.A.copy()
    A_inv_before = model.A_inv.copy()
    theta_before = model.get_theta().copy()

    # 2. Delayed return compensation: delta = -8.0 - 10.0 = -18.0
    delta_reward = -18.0
    model.correct_reward(phi, delta_reward)

    # A and A_inv must be bitwise identical
    assert np.array_equal(model.A, A_before), "Covariance matrix A must not change on return correction"
    assert np.array_equal(model.A_inv, A_inv_before), "Inverse covariance A_inv must not change on return correction"

    # Theta must change
    theta_after = model.get_theta()
    assert not np.array_equal(theta_before, theta_after), "Parameter theta must update after return correction"
    expected_theta_after = theta_before + delta_reward * (A_inv_before @ phi)
    assert np.allclose(theta_after, expected_theta_after, atol=1e-10)

    # 3. Test API order item double-return rejection
    user_login = client.post("/auth/login", json={"email": "rahul@smartcart.ai", "password": "user123"})
    token = user_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Add active item to cart & checkout
    active_prods = client.get("/products?page=1&page_size=1").json()["products"]
    p_id = active_prods[0]["id"]
    client.post("/cart", json={"product_id": p_id, "quantity": 1}, headers=headers)
    order_res = client.post("/orders", json={
        "shipping_address": {"name": "Rahul", "street": "Road", "city": "BLR", "state": "KA", "pin": "560001"}
    }, headers=headers)
    order_data = order_res.json()
    order_id = order_data["id"]
    item_id = order_data["items"][0]["id"]

    # First return -> 200 OK
    ret1 = client.post(f"/orders/{order_id}/items/{item_id}/return", json={"reason": "Fit issue"}, headers=headers)
    assert ret1.status_code == 200

    # Second return on same item -> 400 Bad Request
    ret2 = client.post(f"/orders/{order_id}/items/{item_id}/return", json={"reason": "Fit issue"}, headers=headers)
    assert ret2.status_code == 400
    assert "already been returned" in ret2.json()["detail"].lower()


def test_concurrent_event_posts():
    """Verify thread-safety and concurrency protection on model update and persist path."""
    active_prods = client.get("/products?page=1&page_size=1").json()["products"]
    p_id = active_prods[0]["id"]

    model = model_store.get_model("linucb")
    initial_updates = model.n_updates

    def send_event(idx):
        return client.post("/events", json={
            "product_id": p_id,
            "event_type": "click",
            "session_id": f"concurrent_sess_{idx}"
        })

    num_threads = 15
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_event, i) for i in range(num_threads)]
        results = [f.result() for f in futures]

    for r in results:
        assert r.status_code == 201

    assert model.n_updates == initial_updates + num_threads
    assert not np.isnan(model.A).any()
    assert not np.isnan(model.A_inv).any()

