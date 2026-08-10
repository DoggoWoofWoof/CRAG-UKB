"""Canonical data contract and metrics for end-to-end SOTA RAG evaluation.

The contract separates a retriever/indexer from the common reader and evaluator.
External systems may keep their native outputs, but paper comparisons consume a
normalized JSONL file with document IDs, contexts, predictions, latency, and
token usage. Dataset bundles are immutable and content-addressed so expensive
graph ingestion can be reused safely.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.evaluation.benchmark_generation import _answers_from_metadata
from src.experiments.overlap_retrain import _hard_membership, _splits


CONTRACT_VERSION = 1
DEFAULT_KS = (2, 5, 10, 20, 50, 100)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: Any, *, indent: Optional[int] = 2) -> None:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=indent is not None,
        indent=indent,
        default=str,
    )
    _atomic_text(path, text + ("\n" if indent is not None else ""))


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_stable_json(row) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _title(node: Any) -> str:
    title = node.metadata.get("title") or node.metadata.get("name")
    return str(title).strip() if title else str(node.node_id)


def _canonical_documents(engine: Any) -> List[Dict[str, Any]]:
    documents = []
    for index, node in enumerate(sorted(engine.nodes, key=lambda item: item.node_id)):
        documents.append(
            {
                "id": str(node.node_id),
                "index": index,
                "title": _title(node),
                "text": str(node.content),
                "metadata": {
                    key: value
                    for key, value in node.metadata.items()
                    if key not in {"answer", "answers", "synthetic_neighbors"}
                },
            }
        )
    return documents


def _canonical_edges(
    engine: Any,
    *,
    include_synthetic_edges: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    document_ids = {str(node.node_id) for node in engine.nodes}
    seen = set()
    edges: List[Dict[str, Any]] = []
    degree = Counter()
    type_counts = Counter()

    def add(source: str, target: str, relation: str, provenance: str) -> None:
        key = (source, relation, target)
        if source == target or key in seen:
            return
        if source not in document_ids or target not in document_ids:
            return
        seen.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "provenance": provenance,
            }
        )
        degree[source] += 1
        degree[target] += 1
        type_counts[relation] += 1

    for node in engine.nodes:
        source = str(node.node_id)
        for target in node.neighbors:
            add(source, str(target), "original_link", "dataset")
        if include_synthetic_edges:
            for target in node.metadata.get("synthetic_neighbors", []):
                add(source, str(target), "semantic_knn", "indexer")

    edges.sort(key=lambda row: (row["source"], row["relation"], row["target"]))
    node_count = len(document_ids)
    isolated = node_count - len(degree)
    graph_stats = {
        "node_count": node_count,
        "directed_edge_count": len(edges),
        "edge_type_counts": dict(sorted(type_counts.items())),
        "isolated_node_count": isolated,
        "isolated_node_fraction": round(isolated / node_count, 8) if node_count else 0.0,
        "mean_incident_degree": round(sum(degree.values()) / node_count, 6)
        if node_count
        else 0.0,
        "include_synthetic_edges": include_synthetic_edges,
    }
    return edges, graph_stats


def _canonical_queries(
    engine: Any,
    document_by_id: Mapping[str, Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    splits = _splits(engine, _hard_membership(engine))
    output: Dict[str, List[Dict[str, Any]]] = {}
    for split, examples in splits.items():
        rows = []
        for question_node, gold_partitions, gold_documents in examples:
            supporting_ids = [
                str(document_id)
                for document_id in gold_documents
                if str(document_id) in document_by_id
            ]
            rows.append(
                {
                    "id": str(question_node.node_id),
                    "question": str(question_node.content),
                    "answers": _answers_from_metadata(question_node.metadata),
                    "supporting_document_ids": supporting_ids,
                    "supporting_document_titles": [
                        str(document_by_id[document_id]["title"])
                        for document_id in supporting_ids
                    ],
                    "gold_partitions": [int(value) for value in gold_partitions],
                    "split": split,
                }
            )
        rows.sort(key=lambda row: row["id"])
        output[split] = rows
    return output


def _bundle_fingerprint(
    dataset: str,
    documents: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    queries: Mapping[str, Sequence[Mapping[str, Any]]],
    include_synthetic_edges: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"contract={CONTRACT_VERSION}\n".encode())
    digest.update(f"dataset={dataset}\n".encode())
    digest.update(f"synthetic={int(include_synthetic_edges)}\n".encode())
    for document in documents:
        digest.update(_stable_json(document).encode("utf-8"))
        digest.update(b"\n")
    for edge in edges:
        digest.update(_stable_json(edge).encode("utf-8"))
        digest.update(b"\n")
    for split in sorted(queries):
        for query in queries[split]:
            digest.update(_stable_json(query).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def _write_canonical(
    bundle: Path,
    documents: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    queries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, str]:
    canonical = bundle / "canonical"
    document_path = canonical / "documents.jsonl"
    edge_path = canonical / "edges.jsonl"
    _atomic_jsonl(document_path, documents)
    _atomic_jsonl(edge_path, edges)
    artifacts = {
        document_path.relative_to(bundle).as_posix(): sha256_file(document_path),
        edge_path.relative_to(bundle).as_posix(): sha256_file(edge_path),
    }
    for split, rows in queries.items():
        path = canonical / "queries" / f"{split}.jsonl"
        _atomic_jsonl(path, rows)
        artifacts[path.relative_to(bundle).as_posix()] = sha256_file(path)
    return artifacts


def _write_hipporag_adapter(
    bundle: Path,
    alias: str,
    documents: Sequence[Mapping[str, Any]],
    queries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, str]:
    root = bundle / "adapters" / "hipporag" / "reproduce" / "dataset"
    corpus_path = root / f"{alias}_corpus.json"
    corpus = [
        {"title": doc["id"], "text": doc["text"], "idx": int(doc["index"])}
        for doc in documents
    ]
    _atomic_json(corpus_path, corpus, indent=None)
    artifacts = {corpus_path.relative_to(bundle).as_posix(): sha256_file(corpus_path)}
    document_by_id = {str(doc["id"]): doc for doc in documents}
    for split, rows in queries.items():
        samples = []
        for row in rows:
            paragraphs = []
            for document_id in row["supporting_document_ids"]:
                document = document_by_id[document_id]
                paragraphs.append(
                    {
                        "title": document_id,
                        "text": document["text"],
                        "is_supporting": True,
                        "idx": int(document["index"]),
                    }
                )
            samples.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answers"],
                    "answer_aliases": row["answers"][1:],
                    "answerable": bool(row["answers"]),
                    "paragraphs": paragraphs,
                    "crag_split": split,
                }
            )
        name = alias if split == "test" else f"{alias}_{split}"
        path = root / f"{name}.json"
        _atomic_json(path, samples, indent=None)
        artifacts[path.relative_to(bundle).as_posix()] = sha256_file(path)
    return artifacts


def _write_gfmrag_adapter(
    bundle: Path,
    alias: str,
    documents: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    queries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, str]:
    dataset_root = bundle / "adapters" / "gfmrag" / "data" / alias
    raw = dataset_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    documents_path = raw / "documents.json"
    _atomic_json(
        documents_path,
        {str(doc["id"]): f"{doc['title']}\n{doc['text']}" for doc in documents},
        indent=None,
    )
    artifacts = {
        documents_path.relative_to(bundle).as_posix(): sha256_file(documents_path)
    }
    for split, rows in queries.items():
        payload = [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answers"][0] if row["answers"] else "",
                "answer_aliases": row["answers"][1:],
                "supporting_documents": row["supporting_document_ids"],
                "crag_split": split,
            }
            for row in rows
        ]
        path = raw / f"{split}.json"
        _atomic_json(path, payload, indent=None)
        artifacts[path.relative_to(bundle).as_posix()] = sha256_file(path)

    # This is a structural control for Path B in GFM-RAG. The primary matched
    # run still uses raw documents so the author's NER/OpenIE graph is built.
    stage1 = dataset_root / "provided_crag_graph" / "processed" / "stage1"
    stage1.mkdir(parents=True, exist_ok=True)
    nodes_path = stage1 / "nodes.csv"
    relations_path = stage1 / "relations.csv"
    edges_path = stage1 / "edges.csv"
    with nodes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "type", "attributes"])
        for document in documents:
            writer.writerow(
                [
                    document["id"],
                    "document",
                    _stable_json(
                        {
                            "title": document["title"],
                            "text": document["text"],
                            "crag_node_id": document["id"],
                        }
                    ),
                ]
            )
    relation_names = sorted({str(edge["relation"]) for edge in edges})
    with relations_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "attributes"])
        for relation in relation_names:
            writer.writerow([relation, _stable_json({"source": "crag_bundle"})])
    with edges_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "relation", "target", "attributes"])
        for edge in edges:
            writer.writerow(
                [
                    edge["source"],
                    edge["relation"],
                    edge["target"],
                    _stable_json({"provenance": edge["provenance"]}),
                ]
            )
    for path in (nodes_path, relations_path, edges_path):
        artifacts[path.relative_to(bundle).as_posix()] = sha256_file(path)
    return artifacts


def _write_text_adapter(
    bundle: Path,
    documents: Sequence[Mapping[str, Any]],
    queries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, str]:
    root = bundle / "adapters" / "text_files"
    root.mkdir(parents=True, exist_ok=True)
    adapter_path = root / "adapter.json"
    _atomic_json(
        adapter_path,
        {
            "source": "../../canonical/documents.jsonl",
            "format": "crag_sota_documents_jsonl",
            "materialization": "{index:08d}.txt with '<title>\\n<text>'",
            "document_count": len(documents),
        },
    )
    artifacts = {
        adapter_path.relative_to(bundle).as_posix(): sha256_file(adapter_path),
    }
    for split, rows in queries.items():
        path = root / f"queries_{split}.jsonl"
        _atomic_jsonl(path, rows)
        artifacts[path.relative_to(bundle).as_posix()] = sha256_file(path)
    return artifacts


def export_dataset_bundle(
    dataset: str,
    output_root: Path,
    *,
    include_synthetic_edges: bool = False,
) -> Dict[str, Any]:
    """Export one immutable full-corpus bundle and all supported adapters."""
    from src.core.engine import CoreEngine

    engine = CoreEngine(source=dataset)
    documents = _canonical_documents(engine)
    document_by_id = {str(document["id"]): document for document in documents}
    edges, graph_stats = _canonical_edges(
        engine,
        include_synthetic_edges=include_synthetic_edges,
    )
    queries = _canonical_queries(engine, document_by_id)
    fingerprint = _bundle_fingerprint(
        dataset,
        documents,
        edges,
        queries,
        include_synthetic_edges,
    )
    bundle = output_root / "bundles" / dataset / fingerprint[:16]
    manifest_path = bundle / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            return manifest

    alias = {
        "2wiki_clean": "2wikimultihopqa_crag",
        "musique_clean": "musique_crag",
        "hotpotqa_clean": "hotpotqa_crag",
        "squad_clean": "squad_crag",
        "metaqa": "metaqa_crag",
    }.get(dataset, f"{dataset}_crag")
    artifacts = {}
    artifacts.update(_write_canonical(bundle, documents, edges, queries))
    artifacts.update(_write_hipporag_adapter(bundle, alias, documents, queries))
    artifacts.update(_write_gfmrag_adapter(bundle, alias, documents, edges, queries))
    artifacts.update(_write_text_adapter(bundle, documents, queries))

    split_stats = {}
    for split, rows in queries.items():
        split_stats[split] = {
            "count": len(rows),
            "with_answers": sum(bool(row["answers"]) for row in rows),
            "mean_support_documents": round(
                statistics.fmean(len(row["supporting_document_ids"]) for row in rows),
                6,
            )
            if rows
            else 0.0,
            "sha256": _sha256_bytes(
                b"\n".join(_stable_json(row).encode("utf-8") for row in rows)
            ),
        }
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "dataset": dataset,
        "adapter_alias": alias,
        "fingerprint": fingerprint,
        "bundle_dir": bundle.as_posix(),
        "document_count": len(documents),
        "document_text_bytes": sum(len(doc["text"].encode("utf-8")) for doc in documents),
        "graph": graph_stats,
        "splits": split_stats,
        "artifacts": dict(sorted(artifacts.items())),
        "bundle_size_bytes": directory_size(bundle),
        "labels_excluded_from_index": True,
        "questions_excluded_from_index": True,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        output_root / "bundles" / dataset / "latest.json",
        {
            "dataset": dataset,
            "fingerprint": fingerprint,
            "bundle_dir": bundle.as_posix(),
            "manifest": manifest_path.as_posix(),
        },
    )
    return manifest


def latest_bundle(output_root: Path, dataset: str) -> Path:
    pointer = output_root / "bundles" / dataset / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError(
            f"No exported SOTA bundle for {dataset}; run the export command first."
        )
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return Path(payload["bundle_dir"])


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_exact_match(prediction: str, answers: Sequence[str]) -> float:
    prediction = normalize_answer(prediction)
    return float(bool(prediction) and any(prediction == normalize_answer(a) for a in answers))


def answer_token_f1(prediction: str, answers: Sequence[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens or not answers:
        return 0.0
    prediction_counts = Counter(prediction_tokens)
    best = 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue
        common = prediction_counts & Counter(answer_tokens)
        overlap = sum(common.values())
        if not overlap:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def _context_text(row: Mapping[str, Any]) -> str:
    contexts = row.get("contexts") or []
    pieces = []
    for context in contexts:
        if isinstance(context, str):
            pieces.append(context)
        elif isinstance(context, Mapping):
            pieces.append(str(context.get("text") or context.get("content") or ""))
    return "\n".join(pieces)


def validate_run_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_predictions: bool = False,
) -> None:
    seen = set()
    for index, row in enumerate(rows):
        for field in ("id", "question", "supporting_document_ids", "retrieved_document_ids"):
            if field not in row:
                raise ValueError(f"Row {index} is missing required field {field!r}.")
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"Duplicate query id {row_id!r}.")
        seen.add(row_id)
        if require_predictions and "prediction" not in row:
            raise ValueError(f"Row {index} has no prediction.")
        retrieved = [str(value) for value in row["retrieved_document_ids"]]
        if len(retrieved) != len(set(retrieved)):
            raise ValueError(f"Row {index} contains duplicate retrieved document IDs.")


def _retrieval_metrics(
    retrieved: Sequence[str],
    gold: Sequence[str],
    ks: Sequence[int],
) -> Dict[str, Any]:
    gold_set = set(map(str, gold))
    ranked = list(map(str, retrieved))
    output = {
        "recall": {},
        "full_coverage": {},
        "hit_rate": {},
        "evidence_precision": {},
        "evidence_f1": {},
    }
    for k in ks:
        top = ranked[:k]
        overlap = len(gold_set & set(top))
        recall = overlap / len(gold_set) if gold_set else 0.0
        precision = overlap / len(top) if top else 0.0
        output["recall"][str(k)] = recall
        output["full_coverage"][str(k)] = float(bool(gold_set) and overlap == len(gold_set))
        output["hit_rate"][str(k)] = float(overlap > 0)
        output["evidence_precision"][str(k)] = precision
        output["evidence_f1"][str(k)] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    ranks = {document_id: index + 1 for index, document_id in enumerate(ranked)}
    first = min((ranks[item] for item in gold_set if item in ranks), default=0)
    output["mrr"] = 1.0 / first if first else 0.0
    output["weakest_positive_rank"] = max(
        (ranks.get(item, len(ranked) + 1) for item in gold_set),
        default=0,
    )
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, document_id in enumerate(ranked[:10])
        if document_id in gold_set
    )
    idcg = sum(1.0 / math.log2(index + 2) for index in range(min(len(gold_set), 10)))
    output["ndcg@10"] = dcg / idcg if idcg else 0.0
    return output


def _bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> Optional[Dict[str, float]]:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1 or samples <= 0:
        mean = float(array.mean())
        return {"mean": mean, "low": mean, "high": mean}
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = array[rng.integers(0, len(array), len(array))].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "low": float(low),
        "high": float(high),
    }


def _latency_summary(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 3),
        "p50": round(float(np.quantile(array, 0.50)), 3),
        "p95": round(float(np.quantile(array, 0.95)), 3),
        "p99": round(float(np.quantile(array, 0.99)), 3),
    }


def evaluate_end_to_end(
    rows: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
    pricing: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate retrieval, evidence, answer quality, grounding, and efficiency."""
    validate_run_rows(rows, require_predictions=False)
    per_query: List[Dict[str, Any]] = []
    for row in rows:
        retrieved = [str(value) for value in row.get("retrieved_document_ids", [])]
        gold = [str(value) for value in row.get("supporting_document_ids", [])]
        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        prediction = str(row.get("prediction") or "")
        retrieval = _retrieval_metrics(retrieved, gold, ks)
        answer_em = answer_exact_match(prediction, answers) if answers else None
        answer_f1 = answer_token_f1(prediction, answers) if answers else None
        support_f1 = retrieval["evidence_f1"].get(str(max(ks)), 0.0)
        context = normalize_answer(_context_text(row))
        normalized_prediction = normalize_answer(prediction)
        answer_in_context = (
            float(any(normalize_answer(answer) in context for answer in answers if answer))
            if answers
            else None
        )
        prediction_in_context = (
            float(bool(normalized_prediction) and normalized_prediction in context)
            if prediction
            else None
        )
        latency = row.get("latency_ms") or {}
        usage = row.get("usage") or {}
        query_metrics = {
            "id": str(row["id"]),
            "retrieval": retrieval,
            "answer_em": answer_em,
            "answer_f1": answer_f1,
            "joint_em": answer_em * retrieval["full_coverage"][str(max(ks))]
            if answer_em is not None
            else None,
            "joint_f1": answer_f1 * support_f1 if answer_f1 is not None else None,
            "answer_in_context": answer_in_context,
            "prediction_in_context": prediction_in_context,
            "faithfulness": row.get("faithfulness"),
            "answer_relevance": row.get("answer_relevance"),
            "latency_ms": {
                "retrieval": latency.get("retrieval"),
                "generation": latency.get("generation"),
                "total": latency.get("total"),
            },
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        }
        per_query.append(query_metrics)

    summary: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "n_queries": len(per_query),
        "retrieval": {},
        "answers": {},
        "grounding": {},
        "efficiency": {},
        "confidence_intervals_95": {},
    }
    for metric in (
        "recall",
        "full_coverage",
        "hit_rate",
        "evidence_precision",
        "evidence_f1",
    ):
        summary["retrieval"][metric] = {
            str(k): round(
                100.0
                * statistics.fmean(
                    query["retrieval"][metric][str(k)] for query in per_query
                ),
                2,
            )
            if per_query
            else None
            for k in ks
        }
    for metric in ("mrr", "ndcg@10"):
        summary["retrieval"][metric] = (
            round(100.0 * statistics.fmean(q["retrieval"][metric] for q in per_query), 2)
            if per_query
            else None
        )
    summary["retrieval"]["weakest_positive_rank"] = (
        round(
            statistics.fmean(
                q["retrieval"]["weakest_positive_rank"] for q in per_query
            ),
            2,
        )
        if per_query
        else None
    )

    for metric in ("answer_em", "answer_f1", "joint_em", "joint_f1"):
        values = [q[metric] for q in per_query if q[metric] is not None]
        summary["answers"][metric] = (
            round(100.0 * statistics.fmean(values), 2) if values else None
        )
    for metric in (
        "answer_in_context",
        "prediction_in_context",
        "faithfulness",
        "answer_relevance",
    ):
        values = [q[metric] for q in per_query if q[metric] is not None]
        summary["grounding"][metric] = (
            round(100.0 * statistics.fmean(map(float, values)), 2) if values else None
        )

    for phase in ("retrieval", "generation", "total"):
        values = [
            float(q["latency_ms"][phase])
            for q in per_query
            if q["latency_ms"][phase] is not None
        ]
        summary["efficiency"][f"{phase}_latency_ms"] = _latency_summary(values)
        summary["efficiency"][f"{phase}_throughput_qps"] = (
            round(1000.0 / statistics.fmean(values), 4)
            if values and statistics.fmean(values) > 0
            else None
        )
    for token_type in ("prompt_tokens", "completion_tokens"):
        values = [
            int(q["usage"][token_type])
            for q in per_query
            if q["usage"][token_type] is not None
        ]
        summary["efficiency"][token_type] = {
            "total": int(sum(values)),
            "mean": round(statistics.fmean(values), 2),
        } if values else None
    pricing = pricing or {}
    prompt_total = (summary["efficiency"].get("prompt_tokens") or {}).get("total", 0)
    completion_total = (
        summary["efficiency"].get("completion_tokens") or {}
    ).get("total", 0)
    summary["efficiency"]["generation_cost_usd"] = round(
        prompt_total * float(pricing.get("prompt", 0.0)) / 1_000_000
        + completion_total * float(pricing.get("completion", 0.0)) / 1_000_000,
        6,
    )

    ci_metrics = {}
    for k in ks:
        for metric in (
            "recall",
            "full_coverage",
            "hit_rate",
            "evidence_precision",
            "evidence_f1",
        ):
            ci_metrics[f"{metric}@{k}"] = (
                [q["retrieval"][metric][str(k)] for q in per_query],
                100.0,
            )
    ci_metrics.update(
        {
            "mrr": ([q["retrieval"]["mrr"] for q in per_query], 100.0),
            "ndcg@10": ([q["retrieval"]["ndcg@10"] for q in per_query], 100.0),
            "weakest_positive_rank": (
                [q["retrieval"]["weakest_positive_rank"] for q in per_query],
                1.0,
            ),
            "answer_em": (
                [q["answer_em"] for q in per_query if q["answer_em"] is not None],
                100.0,
            ),
            "answer_f1": (
                [q["answer_f1"] for q in per_query if q["answer_f1"] is not None],
                100.0,
            ),
            "joint_em": (
                [q["joint_em"] for q in per_query if q["joint_em"] is not None],
                100.0,
            ),
            "joint_f1": (
                [q["joint_f1"] for q in per_query if q["joint_f1"] is not None],
                100.0,
            ),
        }
    )
    for index, (name, (values, scale)) in enumerate(ci_metrics.items()):
        ci = _bootstrap_ci(
            values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        summary["confidence_intervals_95"][name] = (
            {key: round(scale * value, 2) for key, value in ci.items()}
            if ci
            else None
        )
    return summary, per_query


def _paired_sign_flip_pvalue(
    differences: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> float:
    """Two-sided paired randomization test with deterministic Monte Carlo draws."""
    array = np.asarray(differences, dtype=np.float64)
    if not len(array) or np.allclose(array, 0.0):
        return 1.0
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    draws = max(1, samples)
    for _ in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(array))
        if abs(float((array * signs).mean())) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1.0) / (draws + 1.0)


def _mcnemar_exact_pvalue(
    baseline: Sequence[float],
    treatment: Sequence[float],
) -> Optional[float]:
    baseline_only = sum(
        int(float(base) == 1.0 and float(treat) == 0.0)
        for base, treat in zip(baseline, treatment)
    )
    treatment_only = sum(
        int(float(base) == 0.0 and float(treat) == 1.0)
        for base, treat in zip(baseline, treatment)
    )
    discordant = baseline_only + treatment_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(baseline_only, treatment_only) + 1)
    ) / (2.0**discordant)
    return min(1.0, 2.0 * tail)


def compare_end_to_end(
    baseline_rows: Sequence[Mapping[str, Any]],
    treatment_rows: Sequence[Mapping[str, Any]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute paired deltas and uncertainty on identical test questions."""
    validate_run_rows(baseline_rows)
    validate_run_rows(treatment_rows)
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    treatment_by_id = {str(row["id"]): row for row in treatment_rows}
    if set(baseline_by_id) != set(treatment_by_id):
        missing_baseline = sorted(set(treatment_by_id) - set(baseline_by_id))
        missing_treatment = sorted(set(baseline_by_id) - set(treatment_by_id))
        raise ValueError(
            "Paired comparison requires identical query IDs; "
            f"missing from baseline={len(missing_baseline)}, "
            f"missing from treatment={len(missing_treatment)}."
        )

    ordered_ids = [str(row["id"]) for row in baseline_rows]
    for query_id in ordered_ids:
        baseline = baseline_by_id[query_id]
        treatment = treatment_by_id[query_id]
        for field in ("question", "supporting_document_ids", "answers"):
            if baseline.get(field) != treatment.get(field):
                raise ValueError(
                    f"Paired query {query_id!r} differs in frozen field {field!r}."
                )

    _, baseline_metrics = evaluate_end_to_end(
        [baseline_by_id[query_id] for query_id in ordered_ids],
        ks=ks,
        bootstrap_samples=0,
    )
    _, treatment_metrics = evaluate_end_to_end(
        [treatment_by_id[query_id] for query_id in ordered_ids],
        ks=ks,
        bootstrap_samples=0,
    )

    metric_specs: Dict[str, Tuple[List[Optional[float]], List[Optional[float]], float, str, bool]] = {}
    for k in ks:
        for metric in ("recall", "full_coverage", "evidence_f1"):
            metric_specs[f"{metric}@{k}"] = (
                [row["retrieval"][metric][str(k)] for row in baseline_metrics],
                [row["retrieval"][metric][str(k)] for row in treatment_metrics],
                100.0,
                "higher",
                metric == "full_coverage",
            )
    for metric, direction in (
        ("mrr", "higher"),
        ("ndcg@10", "higher"),
        ("weakest_positive_rank", "lower"),
    ):
        metric_specs[metric] = (
            [row["retrieval"][metric] for row in baseline_metrics],
            [row["retrieval"][metric] for row in treatment_metrics],
            1.0 if metric == "weakest_positive_rank" else 100.0,
            direction,
            False,
        )
    for metric in ("answer_em", "answer_f1", "joint_em", "joint_f1"):
        metric_specs[metric] = (
            [row[metric] for row in baseline_metrics],
            [row[metric] for row in treatment_metrics],
            100.0,
            "higher",
            metric in {"answer_em", "joint_em"},
        )
    for phase in ("retrieval", "generation", "total"):
        metric_specs[f"{phase}_latency_ms"] = (
            [row["latency_ms"][phase] for row in baseline_metrics],
            [row["latency_ms"][phase] for row in treatment_metrics],
            1.0,
            "lower",
            False,
        )

    summary: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "n_paired_queries": len(ordered_ids),
        "delta_definition": "treatment_minus_baseline",
        "metrics": {},
    }
    per_query = [{"id": query_id, "deltas": {}} for query_id in ordered_ids]
    for metric_index, (
        name,
        (baseline_values, treatment_values, scale, direction, binary),
    ) in enumerate(metric_specs.items()):
        paired = [
            (index, float(base), float(treat))
            for index, (base, treat) in enumerate(
                zip(baseline_values, treatment_values)
            )
            if base is not None and treat is not None
        ]
        if not paired:
            summary["metrics"][name] = None
            continue
        differences = [treat - base for _, base, treat in paired]
        ci = _bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed + metric_index,
        )
        wins = sum(
            (difference > 0.0 if direction == "higher" else difference < 0.0)
            for difference in differences
        )
        losses = sum(
            (difference < 0.0 if direction == "higher" else difference > 0.0)
            for difference in differences
        )
        ties = len(differences) - wins - losses
        baseline_present = [base for _, base, _ in paired]
        treatment_present = [treat for _, _, treat in paired]
        pvalue = (
            _mcnemar_exact_pvalue(baseline_present, treatment_present)
            if binary
            else _paired_sign_flip_pvalue(
                differences,
                samples=bootstrap_samples,
                seed=bootstrap_seed + 10_000 + metric_index,
            )
        )
        summary["metrics"][name] = {
            "direction": direction,
            "n": len(differences),
            "baseline_mean": round(
                scale * statistics.fmean(baseline_present),
                4,
            ),
            "treatment_mean": round(
                scale * statistics.fmean(treatment_present),
                4,
            ),
            "delta": {
                key: round(scale * value, 4) for key, value in (ci or {}).items()
            },
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "p_value_two_sided": round(float(pvalue), 6),
            "test": "mcnemar_exact" if binary else "paired_sign_flip",
        }
        for index, base, treat in paired:
            per_query[index]["deltas"][name] = scale * (treat - base)
    return summary, per_query
