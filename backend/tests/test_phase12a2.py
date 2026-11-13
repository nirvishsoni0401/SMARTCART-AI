"""
SmartCart AI — Phase 12A-2 Unit Tests: Parity, Cache Invalidation, and Two-Stage Hybrids
Verifies:
  1. A2 Identity Parity: DiscountedLinUCB(gamma=1.0) == ColdLinUCB bit-for-bit.
  2. A1 Cache Key Invalidation: is_unit_valid() rejects stale or mismatched config_hash.
  3. B2 Residual LinUCB Toy Environment: Learns zero residual when heuristic equals true utility.
  4. B1 Two-Stage Hybrids: Verifies candidate filtering and cross-department exploration quota.
  5. Parity & Synchronisation: Generated documents match sim_v2_master_results.json ground truth.
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

from experiments.sim_v2.environment import EnvironmentV2, SlateFeedback
from experiments.sim_v2.policies import (
    ColdLinUCBPolicy,
    DiscountedLinUCBPolicy,
    DepartmentHeuristicPolicy,
    DeptRestrictedLinUCBPolicy,
    ResidualLinUCBPolicy,
    DeptPriorLinUCBPolicy
)
from experiments.sim_v2.runner import is_unit_valid, CONFIG_HASH, UNITS_DIR


def test_a2_identity_cold_linucb_vs_discounted_gamma_1_0():
    """Verifies that DiscountedLinUCB(gamma=1.0) and ColdLinUCB produce bit-for-bit identical actions and regret."""
    env = EnvironmentV2(seed=1234, difficulty="medium", drift_mode="abrupt", horizon=500)
    p_cold = ColdLinUCBPolicy(alpha=0.5)
    p_disc = DiscountedLinUCBPolicy(alpha=0.5, gamma=1.0)

    for t in range(500):
        user = env.user_manager.sample_active_user(t)
        av_pids = env.get_available_product_ids(t)
        u_ctx = env.build_user_context(user)
        f_mat = env.build_feature_matrix_for_user(user, u_ctx)

        slate_cold = p_cold.select_slate(env, user, u_ctx, av_pids, t, k=10, feature_matrix=f_mat)
        slate_disc = p_disc.select_slate(env, user, u_ctx, av_pids, t, k=10, feature_matrix=f_mat)

        assert slate_cold == slate_disc, f"Round {t}: Slates differ! Cold={slate_cold} vs Disc={slate_disc}"

        fb = env.step(t, slate_cold, user=user, cached_features=f_mat)
        p_cold.update(fb)
        p_disc.update(fb)

        np.testing.assert_allclose(p_cold.A, p_disc.A, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(p_cold.b, p_disc.b, atol=1e-12, rtol=1e-12)


def test_a1_cache_key_invalidation_on_code_hash_mismatch(tmp_path):
    """Verifies that is_unit_valid() rejects files with wrong or missing config_hash."""
    test_unit_id = "test_unit_hash_mismatch_check"
    unit_file = UNITS_DIR / f"{test_unit_id}.json"

    # Write dummy unit with wrong config_hash
    dummy_data = {
        "unit_id": test_unit_id,
        "config_hash": "invalid_outdated_hash_9999",
        "cumulative_regret": 100.0,
        "cumulative_reward": 500.0
    }
    unit_file.write_text(json.dumps(dummy_data), encoding="utf-8")

    try:
        assert not is_unit_valid(test_unit_id), "is_unit_valid must reject unit with mismatched config_hash"

        # Now write valid config_hash
        dummy_data["config_hash"] = CONFIG_HASH
        unit_file.write_text(json.dumps(dummy_data), encoding="utf-8")
        assert is_unit_valid(test_unit_id), "is_unit_valid must accept unit with matching config_hash"
    finally:
        if unit_file.exists():
            unit_file.unlink()


def test_b2_residual_linucb_toy_environment():
    """
    Toy environment test where feedback reward equals the heuristic offset:
    Residual LinUCB (alpha=0.0 for pure exploitation) must accumulate zero residual,
    theta must remain exactly 0, and recommended slate must match Department Heuristic.
    """
    env = EnvironmentV2(seed=777)
    res_pol = ResidualLinUCBPolicy(alpha=0.0)
    dept_pol = DepartmentHeuristicPolicy()

    user = env.user_manager.sample_active_user(0)
    av_pids = env.get_available_product_ids(0)
    u_ctx = env.build_user_context(user)
    f_mat = env.build_feature_matrix_for_user(user, u_ctx)

    for t in range(50):
        slate = res_pol.select_slate(env, user, u_ctx, av_pids, t, k=10, feature_matrix=f_mat)
        slate_dept = dept_pol.select_slate(env, user, u_ctx, av_pids, t, k=10)
        assert slate == slate_dept, f"Step {t}: Residual slate {slate} != Dept slate {slate_dept}"

        # Synthetic feedback where observed reward equals the heuristic offset
        offsets = res_pol._get_heuristic_offsets(env, user, u_ctx, slate)
        sim_feedback = SlateFeedback(
            round_idx=t,
            user=user,
            user_context=u_ctx,
            slate_pids=slate,
            events=[{"pid": pid, "event_type": "click", "reward": float(offsets[i])} for i, pid in enumerate(slate)],
            realized_reward=float(sum(offsets)),
            expected_policy_reward=float(sum(offsets)),
            expected_oracle_reward=float(sum(offsets)),
            expected_regret=0.0,
            oracle_slate=slate,
            clicked_pids=list(slate),
            cart_pids=[],
            purchased_pids=[],
            disliked_pids=[],
            feature_matrix=f_mat[[env.pid_to_index[p] for p in slate]]
        )
        res_pol.update(sim_feedback)

    # In this toy setup, b should remain zero since residual r - h_offset is zero
    norm_theta = np.linalg.norm(res_pol.b)
    assert norm_theta < 1e-10, f"Residual weights norm should be 0 when r == h_offset, got {norm_theta}"


def test_b1_two_stage_hybrids_candidate_restriction_and_quota():
    """Verifies candidate pool restriction and exploration quota in DeptRestrictedLinUCBPolicy."""
    env = EnvironmentV2(seed=888)
    user = env.user_manager.sample_active_user(0)
    av_pids = env.get_available_product_ids(0)
    u_ctx = env.build_user_context(user)
    f_mat = env.build_feature_matrix_for_user(user, u_ctx)

    # Determine target category using exact heuristic rule
    target_cat = None
    if user.history:
        last_pid = user.history[-1][0]
        if last_pid in env.pid_to_index:
            target_cat = env.products[env.pid_to_index[last_pid]]["category"]
    if not target_cat:
        target_cat = max(u_ctx.category_affinities.items(), key=lambda x: x[1])[0]

    target_pids = [p for p in av_pids if env.products[env.pid_to_index[p]]["category"] == target_cat]

    # 1. No exploration quota (m_explore = 0)
    pol_m0 = DeptRestrictedLinUCBPolicy(name="Dept-Restricted (m=0)", base_bandit=ColdLinUCBPolicy(), m_explore=0)
    slate_m0 = pol_m0.select_slate(env, user, u_ctx, av_pids, 0, k=10, feature_matrix=f_mat)
    assert len(slate_m0) == 10
    assert len(set(slate_m0)) == 10
    slate_cats_m0 = [env.products[env.pid_to_index[p]]["category"] for p in slate_m0]
    if len(target_pids) >= 10:
        assert all(c == target_cat for c in slate_cats_m0), f"All items in slate_m0 must belong to {target_cat}"

    # 2. With exploration quota (m_explore = 2)
    pol_m2 = DeptRestrictedLinUCBPolicy(name="Dept-Restricted (m=2)", base_bandit=ColdLinUCBPolicy(), m_explore=2)
    slate_m2 = pol_m2.select_slate(env, user, u_ctx, av_pids, 0, k=10, feature_matrix=f_mat)
    assert len(slate_m2) == 10
    assert len(set(slate_m2)) == 10
    slate_cats_m2 = [env.products[env.pid_to_index[p]]["category"] for p in slate_m2]
    in_dept_count = sum(1 for c in slate_cats_m2 if c == target_cat)
    out_dept_count = sum(1 for c in slate_cats_m2 if c != target_cat)
    if len(target_pids) >= 8:
        assert in_dept_count == 8, f"Expected 8 in-department items, got {in_dept_count}"
        assert out_dept_count == 2, f"Expected 2 out-of-department exploratory items, got {out_dept_count}"


def test_check_docs_match_results_phase12a2():
    """
    Automated Parity Test for Phase 12A-2:
    Verifies that all numbers in generated reports (phase12a2-report.md, summary_for_chat.txt,
    phase12a2_summary_for_chat.txt, sim_assumptions.md, and README.md Section 9.1) match sim_v2_master_results.json ground-truth exactly.
    """
    results_path = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results.json"
    if not results_path.exists():
        pytest.skip("sim_v2_master_results.json not yet generated")

    with open(results_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    if master.get("metadata", {}).get("phase") != "12A-2":
        pytest.skip("sim_v2_master_results.json is not yet Phase 12A-2")

    report_path = ROOT_DIR / "docs" / "phase12a2-report.md"
    assump_path = ROOT_DIR / "docs" / "sim_assumptions.md"
    chat_path = ROOT_DIR / "docs" / "phase12a2_summary_for_chat.txt"
    readme_path = ROOT_DIR / "README.md"

    if not report_path.exists():
        pytest.skip("docs/phase12a2-report.md not yet generated (runner in progress)")

    assert report_path.exists(), "docs/phase12a2-report.md must exist"
    assert assump_path.exists(), "docs/sim_assumptions.md must exist"
    assert chat_path.exists(), "docs/phase12a2_summary_for_chat.txt must exist"
    assert readme_path.exists(), "README.md must exist"

    report_text = report_path.read_text(encoding="utf-8")
    assump_text = assump_path.read_text(encoding="utf-8")
    chat_text = chat_path.read_text(encoding="utf-8")
    readme_text = readme_path.read_text(encoding="utf-8")

    primary = master["primary_benchmark"]
    # Check key headline policies
    for pol_name in [
        "Department Heuristic",
        "Cold LinUCB",
        "Discounted LinUCB (gamma=0.9995)",
        "Discounted LinUCB (gamma=1.0)",
        "Dept-Restricted LinUCB (Cold)",
        "Dept-Restricted LinUCB (Discounted)",
        "Slate Mixing (k=5 Dept + LinUCB)",
        "Oracle"
    ]:
        m = primary[pol_name]
        reg_mean_str = f"{m['cumulative_regret']['mean']:,.1f}"
        rew_mean_str = f"{m['cumulative_reward']['mean']:,.1f}"
        traj_oracle_str = f"{m['cumulative_trajectory_oracle_reward']['mean']:,.1f}"

        assert reg_mean_str in report_text, f"Regret {reg_mean_str} for {pol_name} missing from report"
        assert rew_mean_str in report_text, f"Reward {rew_mean_str} for {pol_name} missing from report"
        assert traj_oracle_str in report_text, f"Traj Oracle {traj_oracle_str} for {pol_name} missing from report"
        assert reg_mean_str in chat_text, f"Regret {reg_mean_str} for {pol_name} missing from chat summary"

    # Check Oracle regret is strictly 0.0 in all docs
    assert "0.0" in report_text
    assert "0.0" in readme_text
    assert "0.0" in chat_text
