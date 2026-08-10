import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "review_paper" / "experiment.py"
SPEC = importlib.util.spec_from_file_location("realm_experiment", MODULE_PATH)
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


def test_per_query_aggregate_and_joint_diverge():
    retrieved = [{1}, {3, 4}]
    golds = [(1, 2), (3, 4)]

    recall, joint = experiment._per_query(retrieved, golds)

    np.testing.assert_allclose(recall, [0.5, 1.0])
    np.testing.assert_allclose(joint, [0.0, 1.0])


def test_rrf_merge_is_deterministic_and_budgeted():
    first = np.array([[1, 2, 3], [4, 5, 6]])
    second = np.array([[3, 2, 7], [6, 8, 4]])

    merged = experiment._rrf_merge(first, second, k=2)

    assert merged == [{2, 3}, {4, 6}]
    assert all(len(row) == 2 for row in merged)


def test_weakest_link_rank_is_one_based_and_capped():
    rankings = np.array([[8, 2, 5], [7, 6, 4]])
    golds = [(8, 5), (7, 99)]

    ranks = experiment._weakest_link_ranks(rankings, golds)

    np.testing.assert_array_equal(ranks, [3, 4])


def test_mcnemar_counts_directional_changes():
    single = np.array([1, 0, 1, 0, 0])
    second = np.array([1, 1, 0, 1, 0])

    result = experiment._mcnemar_exact(single, second)

    assert result["single_only"] == 1
    assert result["prf_rrf_only"] == 2
    assert 0 <= result["p_value_two_sided"] <= 1


def test_mcnemar_large_balanced_case_is_stable():
    single = np.r_[np.ones(6000), np.zeros(6000)]
    second = np.r_[np.zeros(6000), np.ones(6000)]

    result = experiment._mcnemar_exact(single, second)

    assert result["single_only"] == 6000
    assert result["prf_rrf_only"] == 6000
    assert result["p_value_two_sided"] == 1.0
