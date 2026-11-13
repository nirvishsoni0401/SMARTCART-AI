"""
SmartCart AI — Phase 12C Consistency & Integrity Tests
Verifies:
  1. Warm-Start Cache Key Integrity: Changing sufficient_stats npz or snapshot metadata
     invalidates affected warm units while leaving cold units valid.
  2. Offline Evaluation Consistency: Parity between test suite and frozen claims.
  3. Master Results vs Documents Consistency Guard (replacing skipped Phase 11 test):
     Parses README results section, docs/final_claims.md, docs/phase12a2-report.md,
     docs/phase12b-report.md, and docs/summary_for_chat.txt and confirms exact match
     with experiments/results/sim_v2_master_results.json.
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

from experiments.sim_v2.runner import (
    compute_unit_config_hash,
    compute_code_hash,
    CONFIG_HASH,
    is_unit_valid,
    UNITS_DIR
)


def test_warm_start_cache_key_invalidation_on_stats_or_metadata_change(tmp_path):
    """
    Verifies that compute_unit_config_hash binds to the statistics file and metadata:
    - Changing the stats file alters the hash for Warm policies.
    - Changing the stats file DOES NOT alter the hash for Cold policies.
    - Affected warm units are invalidated, while cold units remain valid.
    """
    # 1. Base hashes
    cold_hash_orig = compute_unit_config_hash("Cold LinUCB", 10000)
    warm_hash_orig = compute_unit_config_hash("Warm LinUCB", 10000)
    assert cold_hash_orig == CONFIG_HASH
    assert warm_hash_orig != cold_hash_orig

    # 2. Create a mock modified stats file
    mock_stats_path = tmp_path / "sufficient_stats_modified.npz"
    real_stats_path = ROOT_DIR / "data" / "historical" / "sufficient_stats_m3.npz"
    assert real_stats_path.exists(), "data/historical/sufficient_stats_m3.npz must exist"

    real_data = dict(np.load(real_stats_path, allow_pickle=True))
    # Alter A_hist slightly
    real_data["A_hist"] = real_data["A_hist"] + 1.0
    np.savez(mock_stats_path, **real_data)

    # 3. Hash with modified stats file
    warm_hash_modified = compute_unit_config_hash("Warm LinUCB", 10000, stats_path=str(mock_stats_path))
    cold_hash_with_mod = compute_unit_config_hash("Cold LinUCB", 10000, stats_path=str(mock_stats_path))

    # Assert warm hash changed, cold hash remained identical
    assert warm_hash_modified != warm_hash_orig, "Modifying statistics file must change warm policy cache hash"
    assert cold_hash_with_mod == cold_hash_orig, "Modifying statistics file must NOT change cold policy cache hash"

    # 4. Alter snapshot metadata (feature_spec_hash or mapping_version)
    mock_stats_meta = tmp_path / "sufficient_stats_meta_change.npz"
    meta_data = dict(np.load(real_stats_path, allow_pickle=True))
    meta_data["feature_spec_hash"] = "altered_spec_hash_999"
    meta_data["mapping_version"] = "v2.0_test"
    np.savez(mock_stats_meta, **meta_data)

    warm_hash_meta = compute_unit_config_hash("Warm LinUCB", 10000, stats_path=str(mock_stats_meta))
    assert warm_hash_meta != warm_hash_orig, "Altering snapshot metadata must change warm policy cache hash"

    # 5. Check unit validation behavior on disk
    warm_uid = "test_unit_phase12c_warm_invalidation"
    cold_uid = "test_unit_phase12c_cold_invariance"
    warm_file = UNITS_DIR / f"{warm_uid}.json"
    cold_file = UNITS_DIR / f"{cold_uid}.json"

    try:
        warm_file.write_text(json.dumps({
            "policy": "Warm LinUCB",
            "n_cap": 10000,
            "config_hash": warm_hash_orig,
            "cumulative_regret": 3000.0,
            "cumulative_reward": 5000.0
        }), encoding="utf-8")

        cold_file.write_text(json.dumps({
            "policy": "Cold LinUCB",
            "n_cap": 10000,
            "config_hash": cold_hash_orig,
            "cumulative_regret": 3500.0,
            "cumulative_reward": 4500.0
        }), encoding="utf-8")

        # Both valid under original stats
        assert is_unit_valid(warm_uid)
        assert is_unit_valid(cold_uid)

        # Stale warm hash invalidation
        stale_warm_file = UNITS_DIR / f"{warm_uid}.json"
        stale_warm_file.write_text(json.dumps({
            "policy": "Warm LinUCB",
            "n_cap": 10000,
            "config_hash": "stale_outdated_hash_123",
            "cumulative_regret": 3000.0,
            "cumulative_reward": 5000.0
        }), encoding="utf-8")
        assert not is_unit_valid(warm_uid), "Warm unit with outdated hash must be invalid"

    finally:
        if warm_file.exists():
            warm_file.unlink()
        if cold_file.exists():
            cold_file.unlink()


def test_all_docs_and_readme_match_master_results():
    """
    Consistency Guard (Part C1):
    Parses numbers in README.md Section 9.1, docs/final_claims.md, docs/phase12a2-report.md,
    docs/phase12b-report.md, and docs/summary_for_chat.txt and asserts exact match
    with sim_v2_master_results.json within rounding tolerance.
    """
    results_path = ROOT_DIR / "experiments" / "results" / "sim_v2_master_results.json"
    assert results_path.exists(), "sim_v2_master_results.json must exist"

    with open(results_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    readme_path = ROOT_DIR / "README.md"
    claims_path = ROOT_DIR / "docs" / "final_claims.md"
    rep12a2_path = ROOT_DIR / "docs" / "phase12a2-report.md"
    rep12b_path = ROOT_DIR / "docs" / "phase12b-report.md"
    chat_path = ROOT_DIR / "docs" / "summary_for_chat.txt"

    for p in [readme_path, claims_path, rep12a2_path, rep12b_path, chat_path]:
        assert p.exists(), f"Document {p} must exist"

    readme_text = readme_path.read_text(encoding="utf-8")
    claims_text = claims_path.read_text(encoding="utf-8")
    rep12a2_text = rep12a2_path.read_text(encoding="utf-8")
    rep12b_text = rep12b_path.read_text(encoding="utf-8")
    chat_text = chat_path.read_text(encoding="utf-8")

    primary = master["primary_benchmark"]

    # Check key primary headline policies
    key_policies = [
        "Oracle",
        "Dept-Restricted LinUCB (Cold)",
        "Dept-Restricted LinUCB (Discounted)",
        "Residual LinUCB",
        "Department Heuristic",
        "Discounted LinUCB (gamma=0.9995)",
        "Cold LinUCB",
        "Warm LinUCB",
        "Popularity",
        "Random"
    ]

    for pol in key_policies:
        p_data = primary[pol]
        reg_mean = p_data["cumulative_regret"]["mean"]
        reg_c1 = f"{reg_mean:,.1f}"
        reg_c2 = f"{reg_mean:.1f}"

        # Assert presence in chat summary and reports
        assert (reg_c1 in chat_text or reg_c2 in chat_text), f"Regret {reg_c1} for {pol} missing from summary_for_chat.txt"
        assert (reg_c1 in rep12a2_text or reg_c2 in rep12a2_text), f"Regret {reg_c1} for {pol} missing from phase12a2-report.md"
        assert (reg_c1 in rep12b_text or reg_c2 in rep12b_text), f"Regret {reg_c1} for {pol} missing from phase12b-report.md"

        # README Section 9.1
        assert (reg_c1 in readme_text or reg_c2 in readme_text), f"Regret {reg_c1} for {pol} missing from README.md"

    # Robustness share checks for core policies
    misspec = master["misspecification_robustness"]
    for pol_key, label in [
        ("Dept-Restricted LinUCB (Cold)", "Dept-Restricted LinUCB (Cold, m=0)"),
        ("Dept-Restricted LinUCB + Expl (Cold, m=2)", "Dept-Restricted LinUCB (Cold, m=2)"),
        ("Residual LinUCB", "Residual LinUCB"),
        ("Department Heuristic", "Department Heuristic"),
        ("Cold LinUCB", "Cold LinUCB")
    ]:
        b_sh = f"{primary[pol_key]['regret_share']['mean']*100:.1f}%"
        n_sh = f"{misspec['nonlinear_interactions'][pol_key]['regret_share']['mean']*100:.1f}%"
        assert b_sh in chat_text, f"Base share {b_sh} for {pol_key} missing from chat summary"
        assert n_sh in chat_text, f"Nonlinear share {n_sh} for {pol_key} missing from chat summary"
        assert b_sh in claims_text, f"Base share {b_sh} for {pol_key} missing from final_claims.md"
        assert n_sh in claims_text, f"Nonlinear share {n_sh} for {pol_key} missing from final_claims.md"

    # Check parity scope statement presence (D3)
    parity_scope_stmt = "The parity test covers policy scoring logic with identical inputs, not the live feature extraction pipeline."
    assert parity_scope_stmt in claims_text, "Parity scope statement missing from final_claims.md"
    assert parity_scope_stmt in readme_text, "Parity scope statement missing from README.md"

    # Check that hype words are absent from final_claims.md
    for hype in ["decisively", "breakthrough", "LOSES"]:
        assert hype.lower() not in claims_text.lower(), f"Hype word '{hype}' found in final_claims.md"


def test_offline_erratum_integrity_and_documentation():
    """
    Verifies Part B Direct Erratum results:
    - offline_erratum_phase12c.json exists and records maximum delta.
    - Baselines are unchanged (delta = 0.000 pp).
    - Discrepancies are documented in docs/offline_claims.md and README.md.
    """
    erratum_path = ROOT_DIR / "experiments" / "results" / "offline_erratum_phase12c.json"
    assert erratum_path.exists(), "offline_erratum_phase12c.json must exist"

    with open(erratum_path, "r", encoding="utf-8") as f:
        err = json.load(f)

    max_delta = err["max_delta_pp"]
    assert max_delta > 0.0, "Erratum must capture post-hash-fix delta"
    assert err["exceeds_threshold_0_3pp"] is True, "Erratum delta must exceed 0.3 pp threshold"

    # Check baseline zero deltas
    results = err["results"]
    for slice_name in ["overall", "novel", "repeat"]:
        for base_pol in ["Popularity Baseline", "Department Heuristic", "RecentlyViewed", "Cold LinUCB"]:
            mac_d = results[slice_name][base_pol]["macro_delta_pp"]
            mic_d = results[slice_name][base_pol]["micro_delta_pp"]
            assert abs(mac_d) < 1e-4, f"Baseline {base_pol} in {slice_name} must have 0.000 pp macro delta"
            assert abs(mic_d) < 1e-4, f"Baseline {base_pol} in {slice_name} must have 0.000 pp micro delta"

    # Verify documentation in offline_claims.md and README.md
    claims_text = (ROOT_DIR / "docs" / "offline_claims.md").read_text(encoding="utf-8")
    readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")

    assert "Phase 12C Erratum" in claims_text, "Phase 12C Erratum section missing from offline_claims.md"
    assert "Phase 12C Erratum" in readme_text, "Phase 12C Erratum section missing from README.md"
    assert "5.46%" in claims_text, "Regenerated novel Warm LinUCB 5.46% missing from offline_claims.md"
    assert "5.46%" in readme_text, "Regenerated novel Warm LinUCB 5.46% missing from README.md"

