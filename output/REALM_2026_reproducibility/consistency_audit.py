"""Reproduce the stratified repeat-coding consistency audit.

The primary snapshot is immutable pre-reconciliation data. The second-pass file
contains source-grounded recodes. Reconciled labels live in coding_sheet.csv.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEED = 20260803
DIMS = [
    "S1_answer",
    "S2_retrieval_recall",
    "A1_evidence_complete",
    "A2_stop_decision",
    "A3_recovery",
    "A4_trajectory",
    "A5_tool_error",
    "C1_calls",
    "C2_cost",
    "C3_latency",
]
ALLOCATION = {"iterative": 2, "agentic": 1, "graph": 3, "dense": 1}


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row.get("paper", "").strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kappa(left: list[int], right: list[int]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("kappa requires two non-empty label vectors of equal length")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    left_pos = sum(left) / n
    right_pos = sum(right) / n
    expected = left_pos * right_pos + (1 - left_pos) * (1 - right_pos)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def _expected_sample(corpus: list[dict[str, str]]) -> set[str]:
    by_group: dict[str, list[str]] = {}
    for row in corpus:
        by_group.setdefault(row["group"], []).append(row["paper"])
    rng = random.Random(SEED)
    selected: set[str] = set()
    for group, count in ALLOCATION.items():
        selected.update(rng.sample(sorted(by_group[group]), count))
    return selected


def audit() -> dict[str, object]:
    corpus_path = HERE / "coding_sheet.csv"
    primary_path = HERE / "consistency_primary_snapshot.csv"
    second_path = HERE / "consistency_second_pass.csv"
    corpus = _load(corpus_path)
    primary = {row["paper"]: row for row in _load(primary_path)}
    second = {row["paper"]: row for row in _load(second_path)}
    reconciled = {row["paper"]: row for row in corpus}

    expected = _expected_sample(corpus)
    if set(primary) != expected or set(second) != expected:
        raise ValueError(
            f"consistency sample mismatch: expected={sorted(expected)}, "
            f"primary={sorted(primary)}, second={sorted(second)}"
        )

    left: list[int] = []
    right: list[int] = []
    disagreements: list[dict[str, object]] = []
    per_dimension: dict[str, dict[str, object]] = {}
    for dim in DIMS:
        dim_left: list[int] = []
        dim_right: list[int] = []
        for paper in sorted(expected):
            first = int(primary[paper][dim])
            repeat = int(second[paper][dim])
            final = int(reconciled[paper][dim])
            if first not in (0, 1) or repeat not in (0, 1) or final not in (0, 1):
                raise ValueError(f"non-binary code for {paper}/{dim}")
            if final != repeat:
                raise ValueError(
                    f"reconciled code does not match source-grounded repeat code: {paper}/{dim}"
                )
            left.append(first)
            right.append(repeat)
            dim_left.append(first)
            dim_right.append(repeat)
            if first != repeat:
                disagreements.append(
                    {
                        "paper": paper,
                        "dimension": dim,
                        "primary": first,
                        "second_pass": repeat,
                        "reconciled": final,
                        "evidence_notes": second[paper]["evidence_notes"],
                    }
                )
        per_dimension[dim] = {
            "raw_agreement": sum(a == b for a, b in zip(dim_left, dim_right)) / len(dim_left),
            "kappa": _kappa(dim_left, dim_right),
            "primary_positive": sum(dim_left),
            "second_pass_positive": sum(dim_right),
        }

    confusion = {
        "both_positive": sum(a == 1 and b == 1 for a, b in zip(left, right)),
        "primary_only": sum(a == 1 and b == 0 for a, b in zip(left, right)),
        "second_only": sum(a == 0 and b == 1 for a, b in zip(left, right)),
        "both_negative": sum(a == 0 and b == 0 for a, b in zip(left, right)),
    }
    return {
        "sample_seed": SEED,
        "allocation": ALLOCATION,
        "sample_papers": sorted(expected),
        "papers": len(expected),
        "dimensions": len(DIMS),
        "cells": len(left),
        "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(left),
        "pooled_cohen_kappa": _kappa(left, right),
        "disagreement_count": len(disagreements),
        "confusion": confusion,
        "disagreements": disagreements,
        "per_dimension": per_dimension,
        "input_sha256": {
            "coding_sheet": _sha256(corpus_path),
            "primary_snapshot": _sha256(primary_path),
            "second_pass": _sha256(second_path),
        },
    }


def main() -> None:
    result = audit()
    output = HERE / "consistency_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
