import torch

from src.experiments.backends import commit_persistent_storage
from src.experiments.l1_unified import (
    DatasetAgnosticRouter,
    _budget_robust_key,
    _balanced_epoch_batches,
    _unified_fusion_specs,
)


def test_persistent_storage_commit_is_a_noop_off_modal():
    assert commit_persistent_storage() is False


def test_unified_router_keeps_dense_skip_head_and_normalized_outputs():
    torch.manual_seed(3)
    model = DatasetAgnosticRouter(
        dimension=4,
        relational_heads=2,
        hidden=8,
    )
    query = torch.randn(3, 4)
    dense_seed = torch.randn(3, 4)

    positions, weights = model(query, dense_seed)

    assert positions.shape == (3, 3, 4)
    assert weights.shape == (3, 3)
    assert torch.allclose(
        positions[:, 0],
        torch.nn.functional.normalize(query, dim=-1),
    )
    assert torch.allclose(
        torch.linalg.vector_norm(positions, dim=-1),
        torch.ones(3, 3),
        atol=1e-6,
    )
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)


def test_balanced_batches_are_equal_per_dataset_and_reproducible():
    eligible = {
        "large": list(range(11)),
        "small": list(range(3)),
    }

    first = _balanced_epoch_batches(
        eligible,
        batch_size=4,
        seed=42,
        epoch=2,
    )
    second = _balanced_epoch_batches(
        eligible,
        batch_size=4,
        seed=42,
        epoch=2,
    )

    assert first == second
    assert sum(dataset == "large" for dataset, _ in first) == 3
    assert sum(dataset == "small" for dataset, _ in first) == 3
    assert all(len(batch) == 4 for _, batch in first)


def test_unified_fusion_policy_uses_signals_not_dataset_names():
    specs = _unified_fusion_specs(("shared_partition_p10q10",))

    assert any(spec["label"] == "dense" for spec in specs)
    assert any(spec["label"] == "shared_router" for spec in specs)
    assert any(
        set(spec["weights"])
        == {"dense", "shared_router", "shared_partition_p10q10"}
        for spec in specs
    )
    assert all("dataset" not in spec for spec in specs)


def test_model_selection_rewards_coverage_across_all_candidate_budgets():
    robust = {
        "fullcov": {"20": 60.0, "50": 60.0, "100": 60.0},
        "recall": {"20": 70.0, "50": 70.0, "100": 70.0},
        "weakest_positive_rank": 40.0,
    }
    top100_only = {
        "fullcov": {"20": 20.0, "50": 40.0, "100": 70.0},
        "recall": {"20": 40.0, "50": 60.0, "100": 80.0},
        "weakest_positive_rank": 35.0,
    }

    assert _budget_robust_key(robust) > _budget_robust_key(top100_only)
