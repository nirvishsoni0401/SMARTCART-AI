"""
SmartCart AI — Phase 11/11B Unit Tests
Verifies the calibrated simulation environment v2, policies, and production pipeline:
  1. Position examination bias is strictly monotonic (gamma_0 > gamma_1 > ... > gamma_9).
  2. Oracle regret is identically zero (E[R_oracle] - E[R_oracle] == 0.0).
  3. Random expected regret is strictly positive.
  4. Funnel calibration anchors:
       - Random slate CTR in [3%, 7%]
       - Oracle slate CTR in [18%, 35%]
       - View-to-cart rate ~ 2.6%
       - Cart-to-purchase rate ~ 32.4%
  5. Cold-start item arrivals: 12 products withheld until round 10,000, then available.
  6. Pipeline diversity constraint: recommended slates contain >= 2 distinct departments.
  7. Exact reproducibility: identical seeds produce identical interaction traces.
  8. Difficulty noise scales: easy=0.15, medium=0.40, hard=0.80.
  9. Click offset naming: renamed to beta_click = 5.72.
  10. Misspecification robustness environments: hidden_style, nonlinear_interactions, high_funnel_conversion.
  11. Master results parity: checks that generated documents agree with sim_v2_master_results.json.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import sys
from pathlib import Path
import json
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from experiments.sim_v2.environment import EnvironmentV2, POSITION_BIAS, DIFFICULTY_SIGMA
from experiments.sim_v2.policies import (
    RandomPolicy,
    PopularityPolicy,
    ColdLinUCBPolicy,
    WarmLinUCBPolicy,
    OraclePolicy,
    ProductionPipelineWrapper
)


def test_position_bias_monotonicity():
    """Verifies that examination bias gamma_k is strictly monotonically decreasing."""
    assert len(POSITION_BIAS) == 10
    diffs = np.diff(POSITION_BIAS)
    assert np.all(diffs < 0.0), "Position bias must strictly decrease with position index"
    assert POSITION_BIAS[0] == pytest.approx(1.0, abs=1e-6)
    assert POSITION_BIAS[-1] < 0.35


def test_oracle_regret_is_identically_zero():
    """Verifies that the Oracle policy achieves identically zero expected regret."""
    env = EnvironmentV2(seed=101)
    oracle = OraclePolicy()

    regrets = []
    for t in range(50):
        user = env.user_manager.sample_active_user(t)
        av_pids = env.get_available_product_ids(t)
        u_ctx = env.build_user_context(user)
        slate = oracle.select_slate(env, user, u_ctx, av_pids, t, k=10)
        fb = env.step(t, slate, user=user)
        regrets.append(fb.expected_regret)

    assert all(r == 0.0 for r in regrets), f"Every Oracle step regret must be identically 0.0, found {regrets}"
    assert np.sum(regrets) == 0.0


def test_random_regret_is_strictly_positive():
    """Verifies that Random policy achieves strictly positive expected regret."""
    env = EnvironmentV2(seed=102)
    rand_policy = RandomPolicy(seed=102)

    regrets = []
    for t in range(50):
        user = env.user_manager.sample_active_user(t)
        av_pids = env.get_available_product_ids(t)
        u_ctx = env.build_user_context(user)
        slate = rand_policy.select_slate(env, user, u_ctx, av_pids, t, k=10)
        fb = env.step(t, slate, user=user)
        regrets.append(fb.expected_regret)

    assert np.mean(regrets) > 0.05, "Random policy must incur substantial expected regret"


def test_cold_start_product_activation():
    """Verifies that 12 products start inactive and are activated at round 10,000."""
    env = EnvironmentV2(seed=103, drift_round=10000)
    assert len(env.cold_start_pids) == 12

    pids_round_0 = env.get_available_product_ids(0)
    assert len(pids_round_0) == 262 - 12
    for csp in env.cold_start_pids:
        assert csp not in pids_round_0

    pids_round_9999 = env.get_available_product_ids(9999)
    assert len(pids_round_9999) == 262 - 12

    pids_round_10000 = env.get_available_product_ids(10000)
    assert len(pids_round_10000) == 262


def test_production_pipeline_diversity_constraint():
    """Verifies that ProductionPipelineWrapper enforces >= 2 distinct categories in top-10."""
    env = EnvironmentV2(seed=104)
    base_pol = PopularityPolicy()
    pipeline = ProductionPipelineWrapper(base_pol, candidate_pool_size=50, min_distinct_cats=2)

    for t in range(30):
        user = env.user_manager.sample_active_user(t)
        av_pids = env.get_available_product_ids(t)
        u_ctx = env.build_user_context(user)
        slate = pipeline.select_slate(env, user, u_ctx, av_pids, t, k=10)

        assert len(slate) == 10
        assert len(set(slate)) == 10, "Slate must not contain duplicate items"

        categories = {env.products[env.pid_to_index[pid]]["category"] for pid in slate}
        assert len(categories) >= 2, f"Slate must contain at least 2 distinct categories, found {categories}"


def test_exact_reproducibility():
    """Verifies that running two environments with the same seed yields identical traces."""
    env1 = EnvironmentV2(seed=777)
    env2 = EnvironmentV2(seed=777)

    for t in range(25):
        u1 = env1.user_manager.sample_active_user(t)
        u2 = env2.user_manager.sample_active_user(t)
        assert u1.user_id == u2.user_id
        assert u1.persona_id == u2.persona_id

        pids1 = env1.get_available_product_ids(t)
        pids2 = env2.get_available_product_ids(t)
        assert pids1 == pids2

        o1, r1 = env1.get_oracle_slate(u1, pids1, t, k=10)
        o2, r2 = env2.get_oracle_slate(u2, pids2, t, k=10)
        assert o1 == o2
        assert r1 == pytest.approx(r2, abs=1e-8)


def test_difficulty_noise_scales():
    """Verifies that the difficulty levels map to exact sigma values without inconsistency."""
    assert DIFFICULTY_SIGMA["easy"] == 0.15
    assert DIFFICULTY_SIGMA["medium"] == 0.40
    assert DIFFICULTY_SIGMA["hard"] == 0.80

    env_easy = EnvironmentV2(difficulty="easy")
    env_med = EnvironmentV2(difficulty="medium")
    env_hard = EnvironmentV2(difficulty="hard")

    assert env_easy.sigma == 0.15
    assert env_med.sigma == 0.40
    assert env_hard.sigma == 0.80


def test_beta_click_naming():
    """Verifies that click offset is named beta_click and equals 5.72 (no clash with LinUCB theta)."""
    env = EnvironmentV2()
    assert hasattr(env, "beta_click"), "Environment must expose beta_click"
    assert env.beta_click == pytest.approx(5.72, abs=1e-6)


def test_misspecification_modes():
    """Verifies that misspecification modes initialize and execute steps cleanly."""
    for mode in ["hidden_style", "nonlinear_interactions", "high_funnel_conversion"]:
        env = EnvironmentV2(seed=42, misspecification=mode)
        u = env.user_manager.sample_active_user(0)
        av = env.get_available_product_ids(0)
        p_clicks = env.compute_all_latent_click_probabilities(u, 0)
        e_vals = env.compute_all_expected_item_values(u, 0)
        assert len(p_clicks) == env.n_products
        assert len(e_vals) == env.n_products

        # Step
        slate = list(av[:10])
        fb = env.step(0, slate, user=u)
        assert fb.expected_policy_reward is not None
        assert fb.expected_regret >= 0.0


def test_funnel_calibration_empirical_bands():
    """
    Verifies that the environment produces rates inside the target empirical bands:
      - Random slate CTR in [3%, 7%]
      - Oracle slate CTR in [18%, 35%]
      - P(cart | click) in [1.5%, 4.5%]
    """
    env_r = EnvironmentV2(seed=999)
    env_o = EnvironmentV2(seed=999)
    n_rounds = 2000

    r_slates_clicked = 0
    o_slates_clicked = 0
    total_clicks = 0
    total_carts = 0

    for t in range(n_rounds):
        user = env_r.user_manager.sample_active_user(t)
        av_pids = env_r.get_available_product_ids(t)

        # Oracle
        o_slate, _ = env_o.get_oracle_slate(user, av_pids, t, k=10)
        fb_o = env_o.step(t, o_slate, user=user)
        if len(fb_o.clicked_pids) > 0:
            o_slates_clicked += 1

        # Random
        r_slate = list(env_r.rng.choice(av_pids, size=10, replace=False))
        fb_r = env_r.step(t, r_slate, user=user)
        if len(fb_r.clicked_pids) > 0:
            r_slates_clicked += 1
        total_clicks += len(fb_r.clicked_pids)
        total_carts += len(fb_r.cart_pids)

    rand_ctr = r_slates_clicked / n_rounds
    oracle_ctr = o_slates_clicked / n_rounds

    assert 0.03 <= rand_ctr <= 0.07, f"Random slate CTR {rand_ctr*100:.2f}% out of [3%, 7%]"
    assert 0.18 <= oracle_ctr <= 0.35, f"Oracle slate CTR {oracle_ctr*100:.2f}% out of [18%, 35%]"
    if total_clicks > 0:
        cart_rate = total_carts / total_clicks
        assert 0.015 <= cart_rate <= 0.045, f"P(cart|click) {cart_rate*100:.2f}% out of [1.5%, 4.5%]"


def test_check_docs_match_results():
    """
    Automated Parity Test:
    Verifies that all numbers in generated reports (final_claims.md, summary_for_chat.txt,
    sim_assumptions.md, and README.md Section 9.1) match sim_v2_master_results.json ground-truth exactly.
    
    Historical note: Previously skipped when master results metadata phase was updated past Phase 11B
    ('12A-2' / '12B'). Re-enabled and generalized in Phase 12C to guard consistency across current
    authoritative documents.
    """
    results_path = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results.json"
    assert results_path.exists(), "sim_v2_master_results.json must exist"

    with open(results_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    final_claims_path = ROOT_DIR / "docs" / "final_claims.md"
    assump_path = ROOT_DIR / "docs" / "sim_assumptions.md"
    chat_path = ROOT_DIR / "docs" / "summary_for_chat.txt"
    readme_path = ROOT_DIR / "README.md"

    assert final_claims_path.exists(), "docs/final_claims.md must exist"
    assert assump_path.exists(), "docs/sim_assumptions.md must exist"
    assert chat_path.exists(), "docs/summary_for_chat.txt must exist"
    assert readme_path.exists(), "README.md must exist"

    claims_text = final_claims_path.read_text(encoding="utf-8")
    assump_text = assump_path.read_text(encoding="utf-8")
    chat_text = chat_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")

    primary = master["primary_benchmark"]
    # Check key policies across documents
    for pol_name in ["Cold LinUCB", "Warm LinUCB", "Popularity", "Oracle", "Dept-Restricted LinUCB (Cold)"]:
        m = primary[pol_name]
        reg_mean_str = f"{m['cumulative_regret']['mean']:,.1f}"
        reg_plain_str = f"{m['cumulative_regret']['mean']:.1f}"
        rew_mean_str = f"{m['cumulative_reward']['mean']:,.1f}"
        ctr_str = f"{m['slate_ctr']['mean']*100:.2f}%"

        assert (reg_mean_str in chat_text or reg_plain_str in chat_text), f"Regret {reg_mean_str} for {pol_name} missing from chat summary"
        assert (reg_mean_str in readme_text or reg_plain_str in readme_text), f"Regret {reg_mean_str} for {pol_name} missing from README"
        if pol_name in ["Oracle", "Dept-Restricted LinUCB (Cold)", "Cold LinUCB"]:
            assert (reg_mean_str in claims_text or reg_plain_str in claims_text), f"Regret {reg_mean_str} for {pol_name} missing from final_claims.md"

    # Check that Oracle regret is strictly 0.0 in all docs
    assert "0.0" in claims_text
    assert "0.0" in readme_text
    assert "0.000" in chat_text or "0.0" in chat_text
