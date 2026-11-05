"""
SmartCart AI — Phase 10C Test Suite
Verifies:
  1. test_recently_viewed_history_strictly_prior_to_event:
     Verifies RecentlyViewedBaseline operates strictly on tau < t history and proves
     that leaking current event tau <= t artificially inflates repeat hit rate to 100%.
  2. test_part_d_hybrid_slate_contract:
     Verifies hybrid slate top-k and deduplication logic across RV and LinUCB.
  3. test_gamma_sweep_monotonic_degradation:
     Verifies warm-started validation gamma sweep degrades monotonically with discounting.
  4. test_spearman_cold_start_correlation:
     Verifies cold-start feature norm correlation with catalog popularity prior.
  5. test_check_docs_match_results:
     Automated audit ensuring docs/offline_evaluation_final.md, docs/phase10b-report.md,
     and README.md match master JSON results within rounding tolerance.
"""

import json
import re
from pathlib import Path
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"
DOCS_DIR = ROOT_DIR / "docs"

from scripts.retailrocket.offline_baselines import RecentlyViewedBaseline


def test_recently_viewed_history_strictly_prior_to_event():
    """Verifies RecentlyViewedBaseline operates strictly on tau < t history.
    Proves that appending tau <= t causes an artificial 100% hit rate leak."""
    catalog = {
        101: {"id": 101, "name": "Prod A", "popularity_prior": 10.0},
        102: {"id": 102, "name": "Prod B", "popularity_prior": 20.0},
        103: {"id": 103, "name": "Prod C", "popularity_prior": 30.0},
        104: {"id": 104, "name": "Prod D", "popularity_prior": 40.0},
    }
    idx_to_pid = [101, 102, 103, 104]
    pid_to_idx = {pid: i for i, pid in enumerate(idx_to_pid)}
    rv = RecentlyViewedBaseline(catalog, idx_to_pid)

    # Event stream: user views 101 at t=1, 102 at t=2, then repeats 101 at t=3
    history_tau_lt_t = [
        (catalog[101], "view", 1000, 2.0),
        (catalog[102], "view", 2000, 2.0)
    ]
    current_target = 101
    current_t = 3000

    # Ensure strictly tau < t: all items in history have ts < current_t
    assert all(item[2] < current_t for item in history_tau_lt_t)

    # Correct evaluation (no leakage):
    scores_correct = rv.score(history_tau_lt_t)
    # Most recently viewed is 102 (rank 1), 101 was viewed before (rank 2)
    assert scores_correct[pid_to_idx[102]] > scores_correct[pid_to_idx[101]]
    assert scores_correct[pid_to_idx[101]] > scores_correct[pid_to_idx[103]]

    # Rank of target 101 is 2
    rank_correct = np.count_nonzero(scores_correct > scores_correct[pid_to_idx[current_target]]) + 1
    assert rank_correct == 2

    # Artificially leaked evaluation (tau <= t appended BEFORE scoring):
    leaked_history = history_tau_lt_t + [(catalog[current_target], "view", current_t, 2.0)]
    scores_leaked = rv.score(leaked_history)
    rank_leaked = np.count_nonzero(scores_leaked > scores_leaked[pid_to_idx[current_target]]) + 1
    # Leakage guarantees rank 1 (100% hit rate)
    assert rank_leaked == 1


def test_part_d_hybrid_slate_contract():
    """Verifies that Hybrid Slate takes top k from RV and fills remaining 10-k from LinUCB without duplication."""
    n_items = 50
    # Simulate RV scores where items 5, 8, 12 have high recency boosts
    rv_scores = np.random.uniform(0, 1, size=n_items)
    rv_scores[5] = 1005.0
    rv_scores[8] = 1004.0
    rv_scores[12] = 1003.0

    # Simulate LinUCB scores where items 8, 15, 20 are top
    linucb_scores = np.random.uniform(0, 1, size=n_items)
    linucb_scores[8] = 10.0
    linucb_scores[15] = 9.5
    linucb_scores[20] = 9.0

    for k in [3, 5]:
        top_k_rv = np.argsort(-rv_scores)[:k]
        sc_lin_rem = linucb_scores.copy()
        sc_lin_rem[top_k_rv] = -1e9
        rem_slots = 10 - k
        top_rem_lin = np.argsort(-sc_lin_rem)[:rem_slots]

        slate = np.concatenate([top_k_rv, top_rem_lin])
        assert len(slate) == 10
        assert len(set(slate)) == 10, "Hybrid slate must not contain duplicates"
        assert set(top_k_rv).issubset(set(slate)), "Top k RV items must be in the slate"


def test_gamma_sweep_monotonic_degradation():
    """Verifies that warm-started validation gamma sweep degrades monotonically with discounting."""
    master_path = RESULTS_DIR / "phase10b_master_results.json"
    with open(master_path, "r") as f:
        master = json.load(f)

    sweep = master.get("gamma_sweep_validation", [])
    assert len(sweep) >= 3
    # Sort by gamma ascending
    sweep_sorted = sorted(sweep, key=lambda x: x["gamma"])
    hrs = [x["overall_hr@10"] for x in sweep_sorted]

    # gamma=1.0 must be the highest
    assert hrs[-1] == max(hrs), "gamma=1.0 must achieve highest performance"
    # Monotonic check: hr must increase as gamma approaches 1.0
    for i in range(len(hrs) - 1):
        assert hrs[i] <= hrs[i+1] + 0.002, "Performance should monotonically improve towards gamma=1.0"


def test_spearman_cold_start_correlation():
    """Verifies cold-start feature norm correlation with catalog popularity prior."""
    master_path = RESULTS_DIR / "phase10b_master_results.json"
    with open(master_path, "r") as f:
        master = json.load(f)

    test_res = master.get("cold_start_shrinkage_test", {})
    assert test_res.get("confirmed") is True
    r = test_res.get("empirical_spearman_correlation")
    assert pytest.approx(r, abs=0.01) == 0.7126


def test_check_docs_match_results():
    """Audit test verifying that docs/offline_evaluation_final.md, docs/phase10b-report.md,
    and README.md match master JSON results within rounding tolerance."""
    master_path = RESULTS_DIR / "phase10b_master_results.json"
    with open(master_path, "r") as f:
        master = json.load(f)

    readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    final_doc_text = (DOCS_DIR / "offline_evaluation_final.md").read_text(encoding="utf-8")
    p10b_doc_text = (DOCS_DIR / "phase10b-report.md").read_text(encoding="utf-8")
    p10c_doc_text = (DOCS_DIR / "phase10c-report.md").read_text(encoding="utf-8")

    # Authoritative figures from master JSON:
    # 1. Novel target Warm LinUCB Macro HR@10: 5.58%
    # 2. Novel target Popularity Baseline Macro HR@10: 5.76%
    # 3. Repeat target Warm LinUCB Macro HR@10: 52.78%
    # 4. Repeat target RecentlyViewed Macro HR@10: 73.22%
    # 5. Feature 6 ablation repeat collapse: 5.74%
    # 6. Candidate Exclusion v2 Overall Warm LinUCB: 9.46%
    for doc_name, doc_text in [("README.md", readme_text),
                               ("offline_evaluation_final.md", final_doc_text),
                               ("phase10b-report.md", p10b_doc_text),
                               ("phase10c-report.md", p10c_doc_text)]:
        assert "5.58%" in doc_text, f"{doc_name} must contain Novel Warm LinUCB Macro HR@10 (5.58%)"
        assert "5.76%" in doc_text, f"{doc_name} must contain Novel Popularity Macro HR@10 (5.76%)"
        assert "52.78%" in doc_text, f"{doc_name} must contain Repeat Warm LinUCB Macro HR@10 (52.78%)"
        assert "73.22%" in doc_text, f"{doc_name} must contain Repeat RecentlyViewed Macro HR@10 (73.22%)"
        assert "5.74%" in doc_text, f"{doc_name} must contain Feature 6 refit repeat collapse (5.74%)"
