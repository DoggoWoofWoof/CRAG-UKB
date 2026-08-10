"""Validate the published REALM experiment artifacts from per-query records."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_estimate(actual: np.ndarray, reported: dict, label: str) -> None:
    expected = round(100 * float(actual.mean()), 2)
    if expected != reported["estimate"]:
        raise AssertionError(f"{label}: expected {expected}, found {reported['estimate']}")
    low, high = reported["ci95"]
    if not low <= reported["estimate"] <= high:
        raise AssertionError(f"{label}: estimate is outside CI {reported['ci95']}")


def validate(results_path: Path, per_query_path: Path) -> dict:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows_by_dataset: dict[str, list[dict]] = defaultdict(list)
    with per_query_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_dataset[row["dataset"]].append(row)

    if set(rows_by_dataset) != set(results["datasets"]):
        raise AssertionError("dataset names differ between aggregate and per-query artifacts")

    validated: dict[str, dict] = {}
    for dataset, reported in results["datasets"].items():
        rows = rows_by_dataset[dataset]
        if len(rows) != reported["queries"]:
            raise AssertionError(f"{dataset}: query count mismatch")
        support_counts = Counter(str(row["support_count"]) for row in rows)
        if dict(sorted(support_counts.items())) != reported["support_count_distribution"]:
            raise AssertionError(f"{dataset}: support-count distribution mismatch")

        for k in (10, 20):
            key = f"k{k}"
            single_recall = np.asarray([row[key]["single_recall"] for row in rows])
            single_joint = np.asarray([row[key]["single_joint"] for row in rows])
            prf_recall = np.asarray([row[key]["prf_rrf_recall"] for row in rows])
            prf_joint = np.asarray([row[key]["prf_rrf_joint"] for row in rows])
            values = {
                "single.aggregate_recall": single_recall,
                "single.joint_recall": single_joint,
                "single.aggregate_minus_joint": single_recall - single_joint,
                "prf_rrf.aggregate_recall": prf_recall,
                "prf_rrf.joint_recall": prf_joint,
                "prf_rrf.aggregate_minus_joint": prf_recall - prf_joint,
                "paired_delta.aggregate_recall": prf_recall - single_recall,
                "paired_delta.joint_recall": prf_joint - single_joint,
            }
            for label, actual in values.items():
                section, metric = label.split(".")
                _assert_estimate(actual, reported[key][section][metric], f"{dataset}.{key}.{label}")

            expected_single_only = int(np.sum((single_joint == 1) & (prf_joint == 0)))
            expected_prf_only = int(np.sum((single_joint == 0) & (prf_joint == 1)))
            mcnemar = reported[key]["joint_mcnemar"]
            if (mcnemar["single_only"], mcnemar["prf_rrf_only"]) != (
                expected_single_only,
                expected_prf_only,
            ):
                raise AssertionError(f"{dataset}.{key}: McNemar cells mismatch")

        weakest = np.asarray([row["weakest_link_rank"] for row in rows])
        weakest_report = reported["weakest_link"]
        weakest_checks = {
            "median_one_based": float(np.median(weakest)),
            "p75_one_based": float(np.percentile(weakest, 75)),
            "within_top20_pct": round(100 * float(np.mean(weakest <= 20)), 2),
            "censored_pct": round(
                100 * float(np.mean(weakest > weakest_report["rank_depth"])), 2
            ),
        }
        for metric, expected in weakest_checks.items():
            if expected != weakest_report[metric]:
                raise AssertionError(f"{dataset}.weakest_link.{metric}: expected {expected}")
        validated[dataset] = {"queries": len(rows), "status": "validated"}

    manifest = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_sha256": _sha256(results_path),
        "per_query_sha256": _sha256(per_query_path),
        "total_queries": sum(len(rows) for rows in rows_by_dataset.values()),
        "datasets": validated,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "review_metric_v2" / "experiment_results.json",
    )
    parser.add_argument(
        "--per-query",
        type=Path,
        default=ROOT / "results" / "review_metric_v2" / "experiment_per_query.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.results.parent / "validation_manifest.json"
    manifest = validate(args.results, args.per_query)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
