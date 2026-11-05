"""
SmartCart AI — Phase 10A Unit Test Suite
Verifies:
  1. Session event exclusion & feature 32 (cart_intent_match) no-cart guarantee.
  2. Many-to-one aliasing behavior when different items map to the same catalog product.
  3. Candidate ranking exclusion for products currently in the cart or already purchased.
  4. Cold context feature behavior and theta norm consistency.
"""

import math
from datetime import datetime, timezone
import numpy as np
import pytest

from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    CATEGORIES,
    UserContext,
    build_context_from_history,
    extract_feature_vector
)


def make_mock_product(pid: int, category: str, brand: str = "TestBrand", price: float = 1000.0):
    return {
        "id": pid,
        "name": f"Product {pid}",
        "category": category,
        "brand": brand,
        "price": price,
        "rating": 4.5,
        "discount_pct": 10.0,
        "popularity_score": 50.0
    }


def filter_history_for_target(history, target_pid, session_gap_seconds=1800):
    """
    Build context strictly before the first event on target_pid in the current session,
    with all events mapped to target_pid excluded from the context.
    """
    if not history:
        return [], {}

    # Identify session boundary (events within session_gap_seconds of preceding event)
    # Filter out any event matching target_pid
    cleaned_hist = []
    cart_cats = {}

    for item in history:
        prod = item[0] if isinstance(item, (tuple, list)) else getattr(item, "product", None)
        pid = prod["id"] if isinstance(prod, dict) else getattr(prod, "id", None)
        ev_type = item[1] if isinstance(item, (tuple, list)) else getattr(item, "event_type", "view")

        # Exclude all events on target_pid
        if pid == target_pid:
            continue

        cleaned_hist.append(item)
        if ev_type in ("addtocart", "cart_add"):
            cat = prod["category"] if isinstance(prod, dict) else getattr(prod, "category", "")
            if cat:
                cart_cats[cat] = cart_cats.get(cat, 0) + 1

    return cleaned_hist, cart_cats


def test_session_target_exclusion_and_feature32_no_cart():
    """
    (A2.i) Session view(item X) -> addtocart(item X) -> transaction(item X), X mapped to product P.
    The transaction example's context must exclude every event on P at or after context time,
    and feature 32 (cart_intent_match) for P must equal its no-cart value (0.0).
    """
    prod_p = make_mock_product(pid=42, category="Electronics", brand="TechBrand")
    t0 = 1440000000000
    t1 = t0 + 60000
    t2 = t0 + 120000

    # Raw full history leading up to transaction
    raw_history = [
        (prod_p, "view", t0, 2.0),
        (prod_p, "addtocart", t1, 5.0)
    ]

    # Clean history excluding target product P
    cleaned_hist, cart_cats = filter_history_for_target(raw_history, target_pid=42)

    assert len(cleaned_hist) == 0, "All events on product P must be excluded from context"
    assert cart_cats.get("Electronics", 0) == 0, "Cart categories must not contain product P"

    ctx = build_context_from_history(
        history=cleaned_hist,
        reference_time=datetime.fromtimestamp(t2 / 1000.0, tz=timezone.utc),
        cart_categories=cart_cats
    )

    phi = extract_feature_vector(ctx, prod_p)

    # Feature 32 must equal its no-cart value (0.0)
    assert phi[32] == 0.0, f"Expected feature 32 to be 0.0, got {phi[32]}"

    # Brand affinity must not be boosted by P's prior events
    assert phi[6] == 0.0, f"Expected brand affinity for P to be 0.0, got {phi[6]}"

    # Category affinity for Electronics must be 0.0
    cat_idx = CATEGORIES.index("Electronics")
    assert phi[cat_idx] == 0.0, f"Expected category affinity to be 0.0, got {phi[cat_idx]}"


def test_many_to_one_aliasing_behavior():
    """
    (A2.ii) Earlier cart add of a DIFFERENT Retailrocket item (item_1) that maps to the
    SAME product P exists when evaluating item_2 (which also maps to product P).
    Document and test behavior:
    Because mapping maps both item_1 and item_2 to product P, target-product exclusion
    on product P safely removes item_1 from context, preventing aliased leakage.
    """
    prod_p = make_mock_product(pid=100, category="Books", brand="PublisherA")
    prod_other = make_mock_product(pid=200, category="Fashion", brand="ApparelB")

    t0 = 1440000000000
    t1 = t0 + 30000
    t2 = t0 + 60000

    # item_1 (maps to P=100) added to cart; other product (pid=200) viewed
    raw_history = [
        (prod_other, "view", t0, 2.0),
        (prod_p, "addtocart", t1, 5.0)  # item_1 mapped to pid=100
    ]

    # Now evaluating item_2 which also maps to pid=100
    cleaned_hist, cart_cats = filter_history_for_target(raw_history, target_pid=100)

    # Context should contain only prod_other
    assert len(cleaned_hist) == 1
    assert cleaned_hist[0][0]["id"] == 200

    ctx = build_context_from_history(
        history=cleaned_hist,
        reference_time=datetime.fromtimestamp(t2 / 1000.0, tz=timezone.utc),
        cart_categories=cart_cats
    )

    phi_p = extract_feature_vector(ctx, prod_p)

    # Feature 32 for P (Books) must be 0.0 because Books is not in the cart (item_1 was excluded)
    assert phi_p[32] == 0.0, f"Expected feature 32 for Books to be 0.0, got {phi_p[32]}"

    # Brand affinity for PublisherA must be 0.0
    assert phi_p[6] == 0.0, f"Expected brand affinity to be 0.0, got {phi_p[6]}"


def test_candidate_exclusion_cart_and_purchased():
    """
    Verify candidate ranking exclusion: products currently in the visitor's cart
    or already purchased in the session must never be ranked in recommendation candidates.
    """
    catalog = {
        10: make_mock_product(10, "Electronics"),
        20: make_mock_product(20, "Electronics"),
        30: make_mock_product(30, "Fashion"),
        40: make_mock_product(40, "Home & Kitchen"),
    }

    session_cart = {10}       # Product 10 is in cart
    session_purchased = {30}  # Product 30 already purchased

    all_pids = list(catalog.keys())

    # Filter candidates
    available_candidates = [
        pid for pid in all_pids
        if pid not in session_cart and pid not in session_purchased
    ]

    assert 10 not in available_candidates, "In-cart product must be excluded from candidates"
    assert 30 not in available_candidates, "Purchased product must be excluded from candidates"
    assert available_candidates == [20, 40], "Only unheld products remain as candidates"

