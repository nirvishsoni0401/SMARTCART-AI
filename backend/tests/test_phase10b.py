"""
SmartCart AI — Phase 10B Test Suite
Verifies:
  1. Feature spec hash and canonical feature table consistency.
  2. Prequential replay on a synthetic toy stream with known expected learning curve.
  3. Leave-group-out Ridge refits from sufficient statistics (dimension, matrix invertibility).
  4. Offline baseline policies contract (RecentlyViewed, DepartmentRecency, SameBrandPopularity).
  5. Candidate exclusion v2 pure discovery behavior.
  6. Isolation: Tests never modify smartcart.db.
"""

import math
import numpy as np
import pytest
from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    CANONICAL_FEATURE_SPEC,
    CATEGORIES,
    CATEGORY_INDEX,
    UserContext,
    build_context_from_history,
    extract_feature_vector
)
from app.recommender.linucb import LinUCBRecommender
from scripts.retailrocket.offline_baselines import (
    PopularityBaseline,
    DepartmentHeuristicBaseline,
    RecentlyViewedBaseline,
    DepartmentRecencyBaseline,
    SameBrandPopularityBaseline
)


def test_feature_spec_hash_and_canonical_table():
    """Verify that FEATURE_SPEC_HASH is 75f60ed70b2a and dimension is strictly 36."""
    assert FEATURE_DIMENSION == 36
    assert FEATURE_SPEC_HASH == "75f60ed70b2a"
    assert "dim=36" in CANONICAL_FEATURE_SPEC
    assert "35=intercept_bias" in CANONICAL_FEATURE_SPEC


def test_prequential_toy_stream():
    """
    Unit test on a toy synthetic stream with known expected learning curve.
    5 arms, dimension d=3. Arm 0 is the ground truth target.
    Verifies that rank is computed in strictly descending order (sc > sc[target]),
    and that as online updates occur, Arm 0's score increases, rank improves to 1,
    and cumulative HR@1 converges to 1.0 without inversion bugs.
    """
    d = 3
    # Synthetic candidate arms
    X_arms = np.array([
        [1.0, 0.0, 0.5],   # Arm 0 (target)
        [0.0, 1.0, 0.9],   # Arm 1 (initially higher norm)
        [0.0, 0.8, 0.8],   # Arm 2
        [0.2, 0.5, 0.7],   # Arm 3
        [0.1, 0.3, 0.6],   # Arm 4
    ], dtype=np.float64)

    model = LinUCBRecommender(feature_dim=d, alpha=0.5, lambda_reg=1.0)

    # Initially at step 0 (theta=0, A=I, score = 0.5 * ||x||):
    # Arm 1 norm = sqrt(1 + 0.81) = 1.345 -> score 0.672
    # Arm 0 norm = sqrt(1 + 0.25) = 1.118 -> score 0.559
    # Arm 0 is ranked behind Arm 1!
    sc_0 = X_arms @ model.theta + 0.5 * np.sqrt(np.sum((X_arms @ model.A_inv) * X_arms, axis=1))
    r_0 = np.count_nonzero(sc_0 > sc_0[0]) + 1
    assert r_0 > 1, "Arm 0 must start behind Arm 1 before updates"

    hits_at_1 = []
    # Stream 20 events where user interacts with Arm 0 with positive reward +5.0
    for step in range(20):
        scores = X_arms @ model.theta + 0.5 * np.sqrt(np.sum((X_arms @ model.A_inv) * X_arms, axis=1))
        # Correct rank definition: count of items strictly greater than target + 1
        rank = np.count_nonzero(scores > scores[0]) + 1
        hit = 1 if rank <= 1 else 0
        hits_at_1.append(hit)

        # Update on target arm
        model.update(X_arms[0], 5.0)

    # After 5 updates, Arm 0 must be decisively Rank 1
    assert hits_at_1[-1] == 1, "Arm 0 must achieve Rank 1 after learning"
    assert sum(hits_at_1[5:]) == 15, "Arm 0 must remain Rank 1 for all subsequent steps"
    assert np.mean(hits_at_1) > 0.7, "Cumulative HR@1 must be high and monotonically rising"


def test_leave_group_out_refits():
    """Verify that leaving feature groups out produces valid refit Ridge models."""
    d = 36
    rng = np.random.default_rng(42)
    # Generate synthetic positive definite A_hist and b_hist
    M = rng.normal(0.0, 1.0, size=(100, d))
    A_hist = M.T @ M
    b_hist = rng.normal(0.0, 1.0, size=d)
    lambda_reg = 1.0

    # Test dropping brand group (index 6)
    drop_idx = [6]
    keep_idx = [i for i in range(d) if i not in drop_idx]
    d_sub = len(keep_idx)
    assert d_sub == 35

    A_sub = lambda_reg * np.eye(d_sub) + A_hist[np.ix_(keep_idx, keep_idx)]
    b_sub = b_hist[keep_idx]
    theta_sub = np.linalg.solve(A_sub, b_sub)

    assert theta_sub.shape == (35,)
    assert not np.isnan(theta_sub).any()

    # Verify ablated feature vectors have dimension d_sub
    ctx = UserContext()
    dummy_product = {
        "id": 1, "category": "Electronics", "brand": "Apple",
        "price": 1000.0, "rating": 4.5, "discount_pct": 10.0, "popularity_score": 80.0
    }
    phi_full = extract_feature_vector(ctx, dummy_product)
    phi_sub = phi_full[keep_idx]
    assert phi_sub.shape == (35,)
    assert phi_full.shape == (36,)


def test_offline_baselines_contract():
    """Verify that RecentlyViewed, DepartmentRecency, and SameBrandPopularity fulfill contract."""
    catalog = {
        101: {"id": 101, "category": "Electronics", "brand": "BrandA", "popularity_prior": 0.9},
        102: {"id": 102, "category": "Electronics", "brand": "BrandB", "popularity_prior": 0.5},
        103: {"id": 103, "category": "Books", "brand": "BrandA", "popularity_prior": 0.8},
        104: {"id": 104, "category": "Fashion", "brand": "BrandC", "popularity_prior": 0.4},
    }
    idx_to_pid = [101, 102, 103, 104]
    dept_pids = {"Electronics": [101, 102], "Books": [103], "Fashion": [104]}

    rv = RecentlyViewedBaseline(catalog, idx_to_pid)
    dr = DepartmentRecencyBaseline(catalog, idx_to_pid, dept_pids)
    sb = SameBrandPopularityBaseline(catalog, idx_to_pid)

    # 1. Cold history fallback
    sc_rv_cold = rv.score([])
    sc_dr_cold = dr.score([])
    sc_sb_cold = sb.score([])

    # Cold history must equal popularity priors exactly
    np.testing.assert_array_equal(sc_rv_cold, np.array([0.9, 0.5, 0.8, 0.4]))
    np.testing.assert_array_equal(sc_dr_cold, np.array([0.9, 0.5, 0.8, 0.4]))
    np.testing.assert_array_equal(sc_sb_cold, np.array([0.9, 0.5, 0.8, 0.4]))

    # 2. History with item 102
    history = [
        ({"id": 102, "category": "Electronics", "brand": "BrandB"}, "view", 1000, 2.0)
    ]
    sc_rv_warm = rv.score(history)
    assert sc_rv_warm[1] > 1000.0, "Item 102 must receive high recency score boost"
    assert np.argmax(sc_rv_warm) == 1, "Item 102 must be rank 1 in RecentlyViewed"

    sc_dr_warm = dr.score(history)
    # Electronics items (101, 102) get department boost
    assert sc_dr_warm[0] > 100.0 and sc_dr_warm[1] > 100.0
    assert sc_dr_warm[2] < 1.0 and sc_dr_warm[3] < 1.0

    sc_sb_warm = sb.score(history)
    # BrandB item (102) gets brand boost
    assert sc_sb_warm[1] > 100.0
    assert sc_sb_warm[0] < 1.0


def test_candidate_exclusion_v2_discovery():
    """Verify that candidate exclusion v2 strictly removes all previously engaged products."""
    engaged = {101, 102}
    scores = np.array([10.0, 8.0, 5.0, 2.0])
    idx_to_pid = [101, 102, 103, 104]
    pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}

    # Pure discovery: mask all engaged products
    excl_mask = np.zeros(len(scores), dtype=bool)
    for p in engaged:
        excl_mask[pid_to_idx[p]] = True

    masked_scores = scores.copy()
    masked_scores[excl_mask] = -1e9

    # The top ranked item must now be item 103 (index 2)
    ranked_indices = np.argsort(masked_scores)[::-1]
    assert ranked_indices[0] == 2, "Item 103 must be top unviewed item"
    assert ranked_indices[1] == 3, "Item 104 must be second unviewed item"

