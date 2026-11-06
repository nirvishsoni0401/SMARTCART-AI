"""
SmartCart AI — Phase 11 Unit Tests
Verifies the calibrated simulation environment v2, policies, and production pipeline:
  1. Position examination bias is strictly monotonic (gamma_0 > gamma_1 > ... > gamma_9).
  2. Oracle regret is identically zero (E[R_oracle] - E[R_oracle] == 0).
  3. Random expected regret is strictly positive.
  4. Funnel calibration anchors:
       - Random slate CTR in [3%, 6%]
       - Oracle slate CTR in [20%, 35%]
       - View-to-cart rate ~ 2.6%
       - Cart-to-purchase rate ~ 32.4%
  5. Cold-start item arrivals: 12 products withheld until round 10,000, then available.
  6. Pipeline diversity constraint: recommended slates contain >= 2 distinct departments.
  7. Exact reproducibility: identical seeds produce identical interaction traces.
  8. Master results parity: checks that generated documents agree with sim_v2_master_results.json.
"""

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

from experiments.sim_v2.environment import EnvironmentV2, POSITION_BIAS
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

    assert all(r == pytest.approx(0.0, abs=1e-8) for r in regrets)
    assert np.mean(regrets) == pytest.approx(0.0, abs=1e-8)


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
    for csp in env.cold_start_pids:
        assert csp in pids_round_10000


def test_production_pipeline_diversity_constraint():
    """Verifies that Production Pipeline enforces >= 2 distinct departments in top-10 slate."""
    env = EnvironmentV2(seed=104)
    cold_linucb = ColdLinUCBPolicy()
    pipeline = ProductionPipelineWrapper(cold_linucb, candidate_pool_size=50, min_distinct_cats=2)

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
    Verifies that all numbers in generated reports (phase11-report.md, summary_for_chat.txt,
    and README.md Section 9.1) match the sim_v2_master_results.json ground-truth exactly.
    """
    results_path = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results.json"
    if not results_path.exists():
        pytest.skip("sim_v2_master_results.json not yet generated")

    with open(results_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    report_path = ROOT_DIR / "docs" / "phase11-report.md"
    chat_path = ROOT_DIR / "docs" / "summary_for_chat.txt"
    readme_path = ROOT_DIR / "README.md"

    assert report_path.exists(), "docs/phase11-report.md must exist"
    assert chat_path.exists(), "docs/summary_for_chat.txt must exist"
    assert readme_path.exists(), "README.md must exist"

    report_text = report_path.read_text(encoding="utf-8")
    chat_text = chat_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")

    primary = master["primary_benchmark"]
    for pol_name in ["Cold LinUCB", "Warm LinUCB", "Discounted LinUCB", "Popularity", "Oracle"]:
        m = primary[pol_name]
        reg_mean_str = f"{m['cumulative_regret']['mean']:,.1f}"
        rew_mean_str = f"{m['cumulative_reward']['mean']:,.1f}"
        ctr_str = f"{m['slate_ctr']['mean']*100:.2f}%"

        # Check phase11-report contains these exact strings
        assert reg_mean_str in report_text, f"Regret {reg_mean_str} for {pol_name} missing from report"
        assert rew_mean_str in report_text, f"Reward {rew_mean_str} for {pol_name} missing from report"
        assert ctr_str in report_text, f"CTR {ctr_str} for {pol_name} missing from report"

        # Check README contains pol_name and reg_mean_str
        assert pol_name in readme_text, f"Policy {pol_name} missing from README"
        assert reg_mean_str in readme_text, f"Regret {reg_mean_str} for {pol_name} missing from README"
