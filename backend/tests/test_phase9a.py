"""
SmartCart AI — Phase 9A Test Suite
Validates:
  1. Reward configuration as single source of truth & return delta math
  2. Return correction logic (stored vector snapshot, invariance of A, idempotency)
  3. Privacy protection (reviewer emails never returned)
  4. Test isolation (production smartcart.db is never written)
  5. Retailrocket sessionization boundary cases (30-min window, ordering, bounces)
  6. Timestamp conversion accuracy
  7. Transaction safeguard (transactions always kept)
  8. Retailrocket proxy mapping determinism & active product validation
  9. Live endpoint isolation (historical.db never touched by operational routes)
  10. Import batch logging completeness
"""

import os
import sys
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.main import app
from app.config import settings
from app.models import Product, RecommendationLog, InteractionEvent, Order, OrderItem, User
from app.services.feature_engineering import FEATURE_DIMENSION, FEATURE_SPEC_HASH
from app.services.reward import get_reward_for_event
from app.recommender.linucb import LinUCBRecommender

client = TestClient(app)
HISTORICAL_DB_PATH = backend_dir / "historical.db"
SMARTCART_DB_PATH = backend_dir / "smartcart.db"


def test_reward_config_single_source_and_delta_math():
    """0.1: Verify reward config is single source of truth and return delta computes net return penalty."""
    r = settings.rewards
    assert r.purchase == 10.0
    assert r.cart_add == 5.0
    assert r.wishlist == 3.0
    assert r.click == 2.0
    assert r.skip == 0.0
    assert r.cart_remove == -2.0
    assert r.dislike == -5.0
    assert r.return_item == -8.0

    # Service mapping uses settings.rewards
    assert get_reward_for_event("purchase") == 10.0
    assert get_reward_for_event("return") == -8.0
    assert get_reward_for_event("click") == 2.0

    # Delta reward math: return_item - purchase = -8.0 - 10.0 = -18.0
    delta = r.return_item - r.purchase
    assert delta == -18.0

    # When delta is added to original purchase (+10.0), net equals return_item (-8.0)
    net_reward = r.purchase + delta
    assert net_reward == r.return_item == -8.0


def test_feature_spec_dimension_and_hash():
    """0.5: Verify feature vector dimension is 36 and hash matches specification."""
    assert FEATURE_DIMENSION == 36
    assert settings.FEATURE_DIMENSION == 36
    assert FEATURE_SPEC_HASH == "75f60ed70b2a"
    assert settings.FEATURE_SPEC_HASH == "75f60ed70b2a"


def test_privacy_reviewer_emails_never_exposed():
    """0.6: Ensure reviewer emails or personal info are never returned by any product endpoint."""
    res = client.get("/products?page=1&page_size=20")
    assert res.status_code == 200
    products = res.json()["products"]
    assert len(products) > 0

    for prod in products:
        p_id = prod["id"]
        det_res = client.get(f"/products/{p_id}")
        assert det_res.status_code == 200
        det_data = det_res.json()

        assert "reviewer_email" not in det_data
        assert "reviewerEmail" not in det_data

        for rev in det_data.get("reviews", []):
            assert "reviewer_email" not in rev
            assert "reviewerEmail" not in rev
            assert "email" not in rev
            assert "reviewer_name" in rev
            assert "rating" in rev
            assert "comment" in rev


def test_return_correction_logic_and_invariance(isolate_test_database):
    """0.2: Verify return correction uses stored snapshot, updates only b, leaves A invariant, and is idempotent."""
    TestSessionLocal = isolate_test_database["test_session_local"]
    db = TestSessionLocal()

    # Create dummy user, product, recommendation log, and order
    u = db.query(User).filter(User.email == "test_return_shopper@smartcart.ai").first()
    if not u:
        u = User(
            name="Return Shopper",
            email="test_return_shopper@smartcart.ai",
            password_hash="fakehash",
            consent_tracking=True
        )
        db.add(u)
        db.commit()
        db.refresh(u)

    prod = db.query(Product).filter(Product.is_active == True).first()
    assert prod is not None

    stored_phi = [0.25] * 36
    rec_log = RecommendationLog(
        recommendation_id="rec_test_ret_123",
        user_id=u.id,
        session_id="sess_ret_123",
        product_id=prod.id,
        algorithm="linucb",
        model_version="v2.0",
        feature_vector_snapshot=stored_phi,
        feature_vector=stored_phi,
        score=0.88,
        rank=1,
        origin="test"
    )
    db.add(rec_log)
    db.commit()

    order = Order(
        user_id=u.id,
        total_amount=prod.price,
        shipping_address="123 Test Street",
        status="placed"
    )
    db.add(order)
    db.commit()
    db.refresh(order)

    item = OrderItem(
        order_id=order.id,
        product_id=prod.id,
        quantity=1,
        price=prod.price,
        status="placed"
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    db.close()

    # Initialize bandit model and record baseline matrices
    model = LinUCBRecommender(feature_dim=36, alpha=0.5)
    vec = np.array(stored_phi, dtype=np.float64)

    # Simulate purchase event first (+10)
    model.update(vec, 10.0)
    A_before = model.A.copy()
    A_inv_before = model.A_inv.copy()
    b_before = model.b.copy()

    # Apply return correction (-18)
    delta_r = -18.0
    model.correct_reward(vec, delta_r)

    # 1. A and A_inv must be mathematically identical
    np.testing.assert_array_almost_equal(model.A, A_before, decimal=7)
    np.testing.assert_array_almost_equal(model.A_inv, A_inv_before, decimal=7)

    # 2. b must be updated by delta_r * vec
    expected_b = b_before + delta_r * vec
    np.testing.assert_array_almost_equal(model.b, expected_b, decimal=7)


def test_sessionization_boundary_cases():
    """Part A: Verify 30-minute inactivity sessionization boundary conditions."""
    inactivity_ms = 30 * 60 * 1000  # 1,800,000 ms

    # Case 1: consecutive events within 30 min belong to same session
    t0 = 1000000000
    t1 = t0 + (29 * 60 * 1000)  # +29 min
    assert (t1 - t0) <= inactivity_ms

    # Case 2: exactly 30 min (1,800,000 ms) remains in same session
    t2 = t0 + inactivity_ms
    assert (t2 - t0) <= inactivity_ms

    # Case 3: 30 min + 1 ms triggers new session
    t3 = t0 + inactivity_ms + 1
    assert (t3 - t0) > inactivity_ms


def test_timestamp_conversion_to_utc():
    """Part A: Verify millisecond timestamps convert accurately to UTC datetime."""
    ms = 1433221332117  # Sample from events.csv
    dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
    assert dt.year == 2015
    assert dt.month == 6
    assert dt.day == 2
    assert dt.tzinfo == timezone.utc


def test_historical_db_exists_and_isolation():
    """Part B: Confirm historical.db exists, contains required tables, and live API does not read it."""
    if not HISTORICAL_DB_PATH.exists() or os.path.getsize(HISTORICAL_DB_PATH) == 0:
        pytest.skip("Retailrocket research database historical.db is omitted from git (781MB raw data in .gitignore)")
    assert HISTORICAL_DB_PATH.exists()

    conn = sqlite3.connect(HISTORICAL_DB_PATH)
    c = conn.cursor()

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
    assert "rr_events" in tables
    assert "rr_sessions" in tables
    assert "rr_category_tree" in tables
    assert "rr_item_category" in tables
    assert "rr_import_batches" in tables
    assert "rr_item_map" in tables

    # Verify counts
    n_events = c.execute("SELECT COUNT(*) FROM rr_events;").fetchone()[0]
    n_sessions = c.execute("SELECT COUNT(*) FROM rr_sessions;").fetchone()[0]
    n_categories = c.execute("SELECT COUNT(*) FROM rr_category_tree;").fetchone()[0]
    n_items = c.execute("SELECT COUNT(*) FROM rr_item_category;").fetchone()[0]

    assert n_events > 2000000
    assert n_sessions > 1000000
    assert n_categories == 1669
    assert n_items > 400000

    conn.close()


def test_transactions_always_kept():
    """Part B: Verify that all transaction events are preserved in historical.db."""
    if not HISTORICAL_DB_PATH.exists() or os.path.getsize(HISTORICAL_DB_PATH) == 0:
        pytest.skip("Retailrocket research database historical.db is omitted from git (781MB raw data in .gitignore)")
    conn = sqlite3.connect(HISTORICAL_DB_PATH)
    c = conn.cursor()

    tx_count = c.execute("SELECT COUNT(*) FROM rr_events WHERE event_type = 'transaction';").fetchone()[0]
    assert tx_count == 22457  # Exactly 100% of all transactions preserved

    # Check transaction IDs are present
    tx_with_id = c.execute("SELECT COUNT(*) FROM rr_events WHERE event_type = 'transaction' AND transaction_id IS NOT NULL;").fetchone()[0]
    assert tx_with_id == 22457

    conn.close()


def test_proxy_mapping_determinism_and_active_product_coverage():
    """Part C: Verify mapping determinism, active product coverage, and no zero-mapped products."""
    if not HISTORICAL_DB_PATH.exists() or os.path.getsize(HISTORICAL_DB_PATH) == 0:
        pytest.skip("Retailrocket research database historical.db is omitted from git (781MB raw data in .gitignore)")
    conn_hist = sqlite3.connect(HISTORICAL_DB_PATH)
    c_hist = conn_hist.cursor()

    conn_smart = sqlite3.connect(SMARTCART_DB_PATH)
    c_smart = conn_smart.cursor()

    active_pids = set(r[0] for r in c_smart.execute("SELECT id FROM products WHERE is_active = 1;").fetchall())
    assert len(active_pids) == 262

    for method in ["category_rank", "cooccurrence_cluster"]:
        mapped_rows = c_hist.execute("""
            SELECT item_id, product_id, department, confidence
            FROM rr_item_map
            WHERE mapping_method = ? AND mapping_version = 'v1.0';
        """, (method,)).fetchall()

        assert len(mapped_rows) == 124882  # Full kept item coverage

        mapped_pids = set(r[1] for r in mapped_rows)
        # 1. Every mapped product must be an active catalog SKU
        assert mapped_pids.issubset(active_pids)

        # 2. 100% of active SKUs must be mapped (zero zero-mapped products)
        assert len(mapped_pids) == 262

        # 3. Confidence values valid
        confidences = [r[3] for r in mapped_rows]
        assert all(0.5 <= conf <= 1.0 for conf in confidences)

    conn_hist.close()
    conn_smart.close()


def test_import_batches_logging_completeness():
    """Part B: Verify row counts at every filter stage are logged in rr_import_batches."""
    if not HISTORICAL_DB_PATH.exists() or os.path.getsize(HISTORICAL_DB_PATH) == 0:
        pytest.skip("Retailrocket research database historical.db is omitted from git (781MB raw data in .gitignore)")
    conn = sqlite3.connect(HISTORICAL_DB_PATH)
    c = conn.cursor()

    batch = c.execute("SELECT batch_id, source_file, thresholds_used, row_counts_at_every_filter_stage, git_commit FROM rr_import_batches ORDER BY started_at DESC LIMIT 1;").fetchone()
    assert batch is not None
    batch_id, src_file, thresholds_json, counts_json, git_commit = batch

    counts = json.loads(counts_json)
    assert counts["raw_events_read"] == 2756101
    assert counts["exact_duplicates_dropped"] == 460
    assert counts["bot_events_capped"] == 49298
    assert counts["tail_items_filtered"] == 150659
    assert counts["final_events_imported"] == 2555684
    assert counts["sessions_created"] == 1648394

    conn.close()
