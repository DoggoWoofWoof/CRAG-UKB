"""Backfill answer metadata without rebuilding document indexes or graphs."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests


log = logging.getLogger("pipeline.backfill_answers")
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _download_squad(path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        log.info("Downloading SQuAD v2 train split once to %s", path)
        with requests.get(SQUAD_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _answers(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    output = []
    for value in values or []:
        text = value.get("text") if isinstance(value, dict) else value
        if text:
            output.append(str(text))
    return list(dict.fromkeys(output))


def backfill_hotpot(
    master_path: Path = Path("data/processed/master_nodes_hotpotqa_clean.json"),
    raw_path: Path = Path("data/raw/hotpotqa_dev.jsonl"),
) -> Dict[str, int]:
    nodes = json.loads(master_path.read_text(encoding="utf-8"))
    by_id = {}
    by_question = {}
    for item in _read_jsonl(raw_path):
        answers = _answers(item.get("golden_answers") or item.get("answer"))
        record = {"answers": answers, "answer": answers[0] if answers else ""}
        by_id[f"hotpot_q_{item.get('id', item.get('_id', ''))}"] = record
        by_question[str(item.get("question", ""))] = record

    updated = 0
    missing = 0
    for node in nodes:
        metadata = node.get("metadata", {})
        if metadata.get("type") != "question":
            continue
        record = by_id.get(node["node_id"]) or by_question.get(node.get("content", ""))
        if record is None:
            missing += 1
            continue
        metadata.update(record)
        updated += 1
    _atomic_json(master_path, nodes)
    return {"updated": updated, "missing": missing}


def _squad_question_records(raw: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    index = 0
    for article in raw["data"]:
        for paragraph in article["paragraphs"]:
            for question in paragraph.get("qas", []):
                answers = _answers(question.get("answers"))
                yield {
                    "node_id": f"squad_clean_q_{index}",
                    "question": str(question["question"]),
                    "answers": answers,
                    "answer": answers[0] if answers else "",
                    "is_impossible": bool(question.get("is_impossible", False)),
                }
                index += 1


def backfill_squad(
    master_path: Path = Path("data/processed/master_nodes_squad_clean.json"),
    raw_path: Path = Path("data/raw/squad_v2.json"),
) -> Dict[str, int]:
    _download_squad(raw_path)
    nodes = json.loads(master_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    records = {record["node_id"]: record for record in _squad_question_records(raw)}
    updated = 0
    missing = 0
    for node in nodes:
        metadata = node.get("metadata", {})
        if metadata.get("type") != "question":
            continue
        record = records.get(node["node_id"])
        if record is None or record["question"] != node.get("content"):
            missing += 1
            continue
        metadata.update(
            {
                "answers": record["answers"],
                "answer": record["answer"],
                "is_impossible": record["is_impossible"],
            }
        )
        updated += 1
    _atomic_json(master_path, nodes)
    return {"updated": updated, "missing": missing}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Backfill clean benchmark answer metadata")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("hotpotqa_clean", "squad_clean"),
        default=("hotpotqa_clean", "squad_clean"),
    )
    args = parser.parse_args(argv)
    results = {}
    if "hotpotqa_clean" in args.datasets:
        results["hotpotqa_clean"] = backfill_hotpot()
    if "squad_clean" in args.datasets:
        results["squad_clean"] = backfill_squad()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

