import pytest
import numpy as np
from app.recommender.base import CandidateItem
from app.recommender.linucb import LinUCBRecommender
from app.recommender.discounted_linucb import DiscountedLinUCBRecommender
from app.recommender.linucb_thompson import LinThompsonSamplingRecommender
from app.recommender.epsilon_greedy import EpsilonGreedyRecommender
from app.recommender.popularity import PopularityRecommender
from app.recommender.random_policy import RandomRecommender

def test_sherman_morrison_rank1_math():
    """
    Verify that the incremental Sherman-Morrison update in LinUCB yields
    an A_inv matrix identical to np.linalg.inv(A) within tight tolerance (< 1e-6).
    """
    d = 36
    model = LinUCBRecommender(feature_dim=d, alpha=0.5, lambda_reg=1.0)

    np.random.seed(42)
    # Perform 50 sequential updates
    for _ in range(50):
        phi = np.random.randn(d)
        phi /= np.linalg.norm(phi)  # normalize
        reward = np.random.choice([10.0, 5.0, 2.0, -2.0, -5.0, -8.0])
        model.update(phi, reward)

    # Compare Sherman-Morrison A_inv with exact matrix inverse
    exact_inv = np.linalg.inv(model.A)
    max_diff = np.max(np.abs(model.A_inv - exact_inv))
    assert max_diff < 1e-6, f"Sherman-Morrison drift exceeded tolerance: {max_diff}"

def test_linucb_reward_adaptation():
    """
    Verify that positive rewards on a category feature increase its score,
    and negative rewards decrease its score.
    """
    d = 36
    model = LinUCBRecommender(feature_dim=d, alpha=0.1, lambda_reg=1.0)

    # Feature vector 1 (e.g. Electronics)
    phi_electronics = np.zeros(d)
    phi_electronics[0] = 1.0  # user category affinity
    phi_electronics[12] = 1.0  # category one-hot
    phi_electronics[-1] = 1.0  # bias

    # Feature vector 2 (e.g. Books)
    phi_books = np.zeros(d)
    phi_books[5] = 1.0
    phi_books[17] = 1.0
    phi_books[-1] = 1.0

    cand1 = CandidateItem(1, None, phi_electronics)
    cand2 = CandidateItem(2, None, phi_books)

    # Initial equal prior predictions
    initial_scores = {s.candidate.product_id: s.score for s in model.select([cand1, cand2], top_k=2)}
    assert abs(initial_scores[1] - initial_scores[2]) < 1e-4

    # Apply 5 positive rewards to Electronics (+10.0 purchase)
    for _ in range(5):
        model.update(phi_electronics, 10.0)

    # Apply negative reward to Books (-5.0 dislike)
    model.update(phi_books, -5.0)

    updated_scores = {s.candidate.product_id: s.score for s in model.select([cand1, cand2], top_k=2)}
    assert updated_scores[1] > updated_scores[2], "Electronics should score significantly higher than Books after positive rewards"
    assert updated_scores[1] > initial_scores[1], "Positive reward should increase score"

def test_model_persistence():
    """Verify that get_state and load_state restore identical parameters and predictions."""
    d = 36
    m1 = LinUCBRecommender(feature_dim=d, alpha=0.8, lambda_reg=1.5)

    np.random.seed(123)
    for _ in range(10):
        phi = np.random.randn(d)
        m1.update(phi, 5.0)

    state = m1.get_state()

    m2 = LinUCBRecommender(feature_dim=d)
    m2.load_state(state)

    assert np.allclose(m1.A, m2.A)
    assert np.allclose(m1.A_inv, m2.A_inv)
    assert np.allclose(m1.b, m2.b)
    assert np.allclose(m1.theta, m2.theta)

def test_all_policies_interface():
    """Ensure all 6 policies satisfy BaseRecommender interface and execute select/update."""
    d = 36
    policies = [
        LinUCBRecommender(feature_dim=d),
        DiscountedLinUCBRecommender(feature_dim=d, gamma=0.9),
        LinThompsonSamplingRecommender(feature_dim=d),
        EpsilonGreedyRecommender(feature_dim=d, epsilon=0.2),
        PopularityRecommender(feature_dim=d),
        RandomRecommender(feature_dim=d)
    ]

    class MockProduct:
        popularity_score = 80.0
        rating = 4.5

    candidates = [
        CandidateItem(i, MockProduct(), np.random.randn(d))
        for i in range(1, 11)
    ]

    for p in policies:
        # Select
        results = p.select(candidates, top_k=5)
        assert len(results) == 5
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "was_exploration") for r in results)

        # Update
        p.update(candidates[0].feature_vector, 10.0)
        assert p.n_updates == 1

        # State serialization
        state = p.get_state()
        assert "n_updates" in state
        p.load_state(state)

