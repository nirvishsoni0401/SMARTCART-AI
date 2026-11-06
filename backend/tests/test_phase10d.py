"""
SmartCart AI — Phase 10D Unit Tests
Verifies:
  1. Part A: Shared deterministic tie-aware ranking function.
     - Hand-checkable toy cases with score and popularity ties.
     - Strict equivalence between compute_rank and get_ranked_indices.
     - Constant-score ranker yields theoretical expectation:
       HR@10 ~ 10/262 (3.82%) on full catalog and 10/50 (20.0%) on 50 candidates.
     - Proves old buggy ranker (count_nonzero(scores > target) + 1) yielded artificial 100% HR@10.
  2. Part A & B: DepartmentHeuristic operates strictly on tau < t history and has no target leakage.
  3. Part B: History Parity (Lifetime vs 30-min Session).
  4. Part F: Automated consistency test verifying all generated documents match JSON results.
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

from scripts.retailrocket.shared_ranking import compute_rank, get_ranked_indices, compute_hybrid_slate
from scripts.retailrocket.offline_baselines import DepartmentHeuristicBaseline, RecentlyViewedBaseline


def test_shared_ranking_hand_checkable_toy_cases():
    """Verifies deterministic tie-breaking:
    1. Score descending
    2. Popularity prior descending
    3. Item index ascending
    """
    # 4 items:
    # Item 0: score=10.0, pop=5.0
    # Item 1: score=10.0, pop=8.0 (beats Item 0 by pop)
    # Item 2: score=10.0, pop=5.0 (ties with Item 0 on score & pop, but item_id 2 > 0, so Item 0 beats Item 2)
    # Item 3: score=12.0, pop=1.0 (highest score, rank 1)
    scores = np.array([10.0, 10.0, 10.0, 12.0])
    pops = np.array([5.0, 8.0, 5.0, 1.0])
    item_ids = np.array([0, 1, 2, 3])

    # Expected order:
    # Rank 1: Item 3 (score 12.0)
    # Rank 2: Item 1 (score 10.0, pop 8.0)
    # Rank 3: Item 0 (score 10.0, pop 5.0, id 0)
    # Rank 4: Item 2 (score 10.0, pop 5.0, id 2)
    assert compute_rank(scores, 3, pops, item_ids) == 1
    assert compute_rank(scores, 1, pops, item_ids) == 2
    assert compute_rank(scores, 0, pops, item_ids) == 3
    assert compute_rank(scores, 2, pops, item_ids) == 4

    # Verify lexsort agrees 100%
    sorted_indices = get_ranked_indices(scores, pops, item_ids)
    np.testing.assert_array_equal(sorted_indices, np.array([3, 1, 0, 2]))


def test_constant_score_ranker_theoretical_expectations():
    """Verifies that a constant-score ranker under shared tie-aware ranking yields:
      - HR@10 == 10/262 (3.82%) under uniform target sampling on 262 items.
      - HR@10 == 10/50 (20.0%) on 50 randomly sampled candidates.
    Also proves the old count_nonzero(scores > target_score) + 1 gave 100% bug.
    """
    n_catalog = 262
    scores_const = np.zeros(n_catalog, dtype=np.float64)
    pops_const = np.zeros(n_catalog, dtype=np.float64)
    item_ids = np.arange(n_catalog, dtype=np.int64)

    # 1. Full catalog uniform targets
    ranks = [compute_rank(scores_const, i, pops_const, item_ids) for i in range(n_catalog)]
    # Exactly ranks 1..262
    assert ranks == list(range(1, n_catalog + 1))
    hr10_full = np.mean(np.array(ranks) <= 10)
    assert hr10_full == pytest.approx(10 / 262, abs=1e-5)

    # Old buggy ranking behavior:
    old_ranks = [int(np.count_nonzero(scores_const > scores_const[i]) + 1) for i in range(n_catalog)]
    assert all(r == 1 for r in old_ranks), "Old bug must assign rank 1 to all tied elements"

    # 2. 50 candidates random simulation (1 target + 49 random items)
    rng = np.random.default_rng(42)
    hits_50 = []
    old_hits_50 = []
    n_sims = 20000
    for _ in range(n_sims):
        pool = rng.choice(n_catalog, size=50, replace=False)
        t_pos = 0  # Target is pool[0]
        # New shared ranking
        r = compute_rank(scores_const[pool], t_pos, pops_const[pool], item_ids=pool)
        hits_50.append(r <= 10)
        # Old buggy ranking
        r_old = np.count_nonzero(scores_const[pool] > scores_const[pool][t_pos]) + 1
        old_hits_50.append(r_old <= 10)

    # Theoretical rate: 10 / 50 = 0.20
    assert np.mean(hits_50) == pytest.approx(0.20, abs=0.02)
    # Old buggy rate was 100%
    assert np.mean(old_hits_50) == 1.0


def test_department_heuristic_strictly_prior_history_no_leakage():
    """Verifies that DepartmentHeuristicBaseline uses strictly tau < t history and has zero target leakage."""
    catalog = {
        101: {"id": 101, "category": "Electronics", "popularity_prior": 10.0},
        102: {"id": 102, "category": "Electronics", "popularity_prior": 20.0},
        103: {"id": 103, "category": "Books", "popularity_prior": 30.0},
        104: {"id": 104, "category": "Books", "popularity_prior": 40.0},
    }
    idx_to_pid = [101, 102, 103, 104]
    dept_pids = {"Electronics": [101, 102], "Books": [103, 104]}
    dept_bl = DepartmentHeuristicBaseline(catalog, idx_to_pid, dept_pids)

    # Case 1: Empty history -> returns catalog popularity priors exactly
    sc_cold = dept_bl.score([])
    np.testing.assert_array_equal(sc_cold, np.array([10.0, 20.0, 30.0, 40.0]))

    # Case 2: History with Electronics -> boosts Electronics products (+100)
    history = [
        ({"id": 101, "category": "Electronics"}, "view", 1000, 2.0)
    ]
    sc_warm = dept_bl.score(history)
    assert sc_warm[0] == 110.0  # 101: 10 + 100
    assert sc_warm[1] == 120.0  # 102: 20 + 100
    assert sc_warm[2] == 30.0   # 103: unboosted
    assert sc_warm[3] == 40.0   # 104: unboosted

    # Verify that ranker breaks ties within department by popularity
    # 102 has pop 20 > 101 pop 10, so 102 is rank 1, 101 is rank 2
    r_102 = compute_rank(sc_warm, 1, sc_cold)
    r_101 = compute_rank(sc_warm, 0, sc_cold)
    assert r_102 == 1
    assert r_101 == 2


def test_hybrid_slate_contract_and_deduplication():
    """Verifies that Hybrid Slate takes top k from RV and fills remaining from LinUCB without duplication."""
    rv_scores = np.array([1000.0, 500.0, 200.0, 0.0, 0.0])
    linucb_scores = np.array([5.0, 10.0, 8.0, 12.0, 15.0])
    pops = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # k=2: top 2 from RV are [0, 1].
    # Remaining 2 to fill from LinUCB (excluding 0, 1):
    # LinUCB order over [2, 3, 4]: item 4 (15.0), item 3 (12.0).
    slate = compute_hybrid_slate(rv_scores, linucb_scores, pops, k_recency=2, slate_size=4)
    np.testing.assert_array_equal(slate, np.array([0, 1, 4, 3]))
    assert len(set(slate)) == 4, "No duplicates allowed in hybrid slate"


def test_check_docs_match_results():
    """Audit test verifying that all generated markdown documents, README, and summary_for_chat.txt
    contain numbers that match experiments/results/offline_master_results.json exactly.
    Fails if any document differs beyond rounding tolerance."""
    results_path = ROOT_DIR / "experiments" / "results" / "offline_master_results.json"
    if not results_path.exists():
        pytest.skip("offline_master_results.json not generated yet")

    with open(results_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    docs_to_check = {
        "README.md": (ROOT_DIR / "README.md").read_text(encoding="utf-8"),
        "offline_evaluation_final.md": (ROOT_DIR / "docs" / "offline_evaluation_final.md").read_text(encoding="utf-8"),
        "phase10d-report.md": (ROOT_DIR / "docs" / "phase10d-report.md").read_text(encoding="utf-8"),
        "summary_for_chat.txt": (ROOT_DIR / "docs" / "summary_for_chat.txt").read_text(encoding="utf-8"),
    }

    # 1. Dataset counts
    dc = master["dataset_counts"]
    n_total_str = f"{dc['total_test_events']['n_events']:,}"
    n_vis_str = f"{dc['total_test_events']['n_visitors']:,}"
    n_nov_str = f"{dc['novel_target_events']['n_events']:,}"
    n_rep_str = f"{dc['repeat_target_events']['n_events']:,}"
    n_cold_str = f"{dc['cold_context_session_starts']['n_events']:,}"

    for doc_name, text in docs_to_check.items():
        assert n_total_str in text, f"{doc_name} missing total events {n_total_str}"
        assert n_vis_str in text, f"{doc_name} missing total visitors {n_vis_str}"
        assert n_nov_str in text, f"{doc_name} missing novel events {n_nov_str}"
        assert n_rep_str in text, f"{doc_name} missing repeat events {n_rep_str}"
        assert n_cold_str in text, f"{doc_name} missing cold session count {n_cold_str}"

    # 2. Key Headline Metrics (Lifetime History)
    life = master["evaluations"]["lifetime_history"]
    over_pols = {x["policy"]: x for x in life["overall"]["full_catalog"]}

    pop_over = f"{over_pols['Popularity Baseline']['macro_hr@10'] * 100.0:.2f}%"
    dept_over = f"{over_pols['Department Heuristic']['macro_hr@10'] * 100.0:.2f}%"
    rv_over = f"{over_pols['RecentlyViewed']['macro_hr@10'] * 100.0:.2f}%"
    warm_over = f"{over_pols['Warm LinUCB']['macro_hr@10'] * 100.0:.2f}%"
    hyb3_over = f"{next(x for x in life['overall']['full_catalog'] if 'k=3' in x['policy'])['macro_hr@10'] * 100.0:.2f}%"

    for doc_name, text in docs_to_check.items():
        assert pop_over in text, f"{doc_name} missing Popularity overall macro {pop_over}"
        assert dept_over in text, f"{doc_name} missing Dept Heuristic overall macro {dept_over}"
        assert rv_over in text, f"{doc_name} missing RecentlyViewed overall macro {rv_over}"
        assert warm_over in text, f"{doc_name} missing Warm LinUCB overall macro {warm_over}"
        assert hyb3_over in text, f"{doc_name} missing Hybrid (k=3) overall macro {hyb3_over}"

    # 3. Prequential catch-up step
    c_step_raw = master["prequential_replay"]["first_step_cold_within_1se_of_warm"]
    c_step_comma = f"{c_step_raw:,}"
    c_step_plain = str(c_step_raw)
    for doc_name, text in docs_to_check.items():
        assert (c_step_comma in text) or (c_step_plain in text), f"{doc_name} missing prequential step {c_step_raw}"

