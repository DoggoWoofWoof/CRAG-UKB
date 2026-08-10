from types import SimpleNamespace
import json

import torch

from src.experiments.l1_optimize import (
    _coverage_objective,
    _split_manifest,
    _write_candidates,
)
from src.experiments.overlap_retrain import _splits
from src.experiments.sync import local_files, parse_option


def _question(node_id, split, neighbor, content="question"):
    return SimpleNamespace(
        node_id=node_id,
        content=content,
        metadata={"type": "question", "split": split},
        neighbors=[neighbor],
    )


def test_splits_preserve_official_boundaries():
    engine = SimpleNamespace(
        all_nodes=[
            _question("q_train", "train", "d1"),
            _question("q_dev", "dev", "d1"),
            _question("q_test", "test", "d1"),
        ]
    )
    splits = _splits(engine, {"d1": {7}})

    assert [row[0].node_id for row in splits["train"]] == ["q_train"]
    assert [row[0].node_id for row in splits["val"]] == ["q_dev"]
    assert [row[0].node_id for row in splits["test"]] == ["q_test"]


def test_coverage_objective_rewards_the_weakest_positive():
    positive_mask = torch.tensor([[True, True, False, False, False]])
    weak_logits = torch.tensor([[3.0, -2.0, 2.0, 1.0, 0.0]])
    covered_logits = torch.tensor([[3.0, 3.0, 2.0, 1.0, 0.0]])

    weak_kl, weak_coverage = _coverage_objective(
        weak_logits, positive_mask, target_topk=2
    )
    covered_kl, covered_coverage = _coverage_objective(
        covered_logits, positive_mask, target_topk=2
    )

    assert covered_kl < weak_kl
    assert covered_coverage < weak_coverage


def test_split_manifest_tracks_semantic_inputs():
    original = {
        "train": [(_question("q1", "train", "d1", "first text"), {1}, ["d1"])]
    }
    changed_text = {
        "train": [(_question("q1", "train", "d1", "changed text"), {1}, ["d1"])]
    }
    changed_gold = {
        "train": [(_question("q1", "train", "d1", "first text"), {2}, ["d2"])]
    }

    original_hash = _split_manifest(original)["train"]["sha256"]
    assert _split_manifest(changed_text)["train"]["sha256"] != original_hash
    assert _split_manifest(changed_gold)["train"]["sha256"] != original_hash


def test_sync_enumerates_new_fingerprint_files(tmp_path):
    cache = tmp_path / "cache" / "L1" / "fingerprint"
    cache.mkdir(parents=True)
    query_cache = cache / "queries_train.npz"
    query_cache.write_bytes(b"cache")

    assert local_files(str(tmp_path / "cache" / "L1")) == [
        query_cache.as_posix()
    ]
    assert parse_option(["--run-id", "confirm"], "--run-id") == "confirm"


def test_candidate_export_contains_stable_ids(tmp_path):
    output = tmp_path / "candidates.jsonl"
    _write_candidates(
        output,
        query_ids=["q1"],
        order=[[2, 0, 1]],
        gold=[[1]],
        document_ids=["d0", "d1", "d2"],
        seed=42,
        method="test",
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["candidate_doc_indices"] == [2, 0, 1]
    assert row["candidate_doc_ids"] == ["d2", "d0", "d1"]
    assert row["gold_doc_ids"] == ["d1"]
    assert not output.with_suffix(".jsonl.tmp").exists()
