"""
Matched-corpus SOTA retrieval baselines for the Level 1 paper protocol.

Every method sees the same clean UKB, query split, test subset, and top-100
budget as ``l1_optimize``. Static baselines are evaluated directly; the only
tuned baseline, weighted dense+BM25 RRF, selects its lexical weight on
validation and touches test once. Heavy method rankings are cached so cloud
runs can resume without repeating model inference.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path

import faiss
import numpy as np

from src.core.encoders import DenseEncoder
from src.core.engine import CoreEngine
from src.core.splade_scorer import SPLADE_MODEL, SpladeScorer
from src.experiments.l1_optimize import (
    ENCODER_NAME,
    _atomic_json_dump,
    _cap,
    _document_manifest,
    _prepare_cached,
    _query_cache_valid,
    _split_manifest,
)
from src.experiments.overlap_retrain import (
    _hard_membership,
    _reconstruct,
    _splits,
)
from src.experiments.stats import paired

log = logging.getLogger("experiments.sota_baselines")

KS = (2, 5, 10, 20, 50, 100)
MAX_K = max(KS)
CORE_METHODS = ("dense", "bm25", "hybrid_rrf", "hybrid_tuned")
OPTIONAL_METHODS = ("hybrid_ce", "splade", "colbert")
ALL_METHODS = CORE_METHODS + OPTIONAL_METHODS
RRF_WEIGHTS = (0.25, 0.5, 1.0, 2.0, 4.0)
RRF_K = 60
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLBERT_MODEL = "colbert-ir/colbertv2.0"


class _InvertedBM25:
    """Exact rank_bm25 scoring over postings needed by this run's queries."""

    def __init__(self, bm25, query_texts):
        vocabulary = {
            token
            for text in query_texts
            for token in text.lower().split()
            if token in bm25.idf
        }
        postings = {term: [[], []] for term in vocabulary}
        for document_index, frequencies in enumerate(bm25.doc_freqs):
            for term in frequencies.keys() & vocabulary:
                postings[term][0].append(document_index)
                postings[term][1].append(frequencies[term])

        document_lengths = np.asarray(bm25.doc_len, dtype=np.float64)
        self.corpus_size = int(bm25.corpus_size)
        self.postings = {}
        for term, (indices, frequencies) in postings.items():
            indices = np.asarray(indices, dtype=np.int64)
            frequencies = np.asarray(frequencies, dtype=np.float64)
            denominator = frequencies + bm25.k1 * (
                1.0 - bm25.b
                + bm25.b * document_lengths[indices] / bm25.avgdl
            )
            contributions = (
                bm25.idf[term]
                * frequencies
                * (bm25.k1 + 1.0)
                / denominator
            )
            self.postings[term] = (indices, contributions)

    def rank(self, query, k=MAX_K):
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for token in query.lower().split():
            posting = self.postings.get(token)
            if posting is not None:
                scores[posting[0]] += posting[1]
        nonzero = np.flatnonzero(scores)
        ranked = nonzero[
            np.lexsort((nonzero, -scores[nonzero]))
        ].tolist()
        if len(ranked) < k:
            selected = set(ranked)
            ranked.extend(
                document_index
                for document_index in range(self.corpus_size)
                if document_index not in selected
            )
        return ranked[:k]


def _rrf(orders, weights, k=MAX_K):
    if not orders:
        return []
    n_queries = len(orders[0])
    output = []
    for query_index in range(n_queries):
        scores = {}
        best_rank = {}
        for method_order, weight in zip(orders, weights):
            for rank, doc_index in enumerate(method_order[query_index]):
                doc_index = int(doc_index)
                if doc_index < 0:
                    continue
                scores[doc_index] = (
                    scores.get(doc_index, 0.0) + weight / (RRF_K + rank + 1)
                )
                best_rank[doc_index] = min(best_rank.get(doc_index, rank), rank)
        ranked = sorted(
            scores,
            key=lambda doc_index: (
                -scores[doc_index],
                best_rank[doc_index],
                doc_index,
            ),
        )
        output.append(ranked[:k])
    return output


def _query_metrics(order, gold):
    positives = set(map(int, gold))
    ranks = {int(doc): rank + 1 for rank, doc in enumerate(order)}
    row = {"recall": {}, "fullcov": {}, "hit_rate": {}}
    for k in KS:
        retrieved = set(map(int, order[:k]))
        overlap = len(positives & retrieved)
        row["recall"][str(k)] = overlap / len(positives)
        row["fullcov"][str(k)] = int(overlap == len(positives))
        row["hit_rate"][str(k)] = int(overlap > 0)
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_index in enumerate(order[:10])
        if int(doc_index) in positives
    )
    idcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(min(len(positives), 10))
    )
    row["ndcg@10"] = dcg / idcg if idcg else 0.0
    first = min((ranks[doc] for doc in positives if doc in ranks), default=MAX_K + 1)
    row["mrr"] = 1.0 / first if first <= MAX_K else 0.0
    positive_ranks = [ranks.get(doc, MAX_K + 1) for doc in positives]
    row["weakest_positive_rank"] = max(positive_ranks)
    return row


def _evaluate(order, gold):
    rows = [
        _query_metrics(ranked, positives)
        for ranked, positives in zip(order, gold)
        if positives
    ]
    summary = {}
    for group in ("recall", "fullcov", "hit_rate"):
        summary[group] = {
            str(k): round(
                float(np.mean([row[group][str(k)] for row in rows])) * 100,
                2,
            )
            for k in KS
        }
    summary["ndcg@10"] = round(
        float(np.mean([row["ndcg@10"] for row in rows])) * 100, 2
    )
    summary["mrr"] = round(
        float(np.mean([row["mrr"] for row in rows])) * 100, 2
    )
    summary["weakest_positive_rank"] = round(
        float(np.mean([row["weakest_positive_rank"] for row in rows])), 2
    )
    summary["n_queries"] = len(rows)
    return summary, rows


def _selection_key(metrics):
    return (
        metrics["fullcov"]["100"],
        metrics["recall"]["100"],
        metrics["fullcov"]["50"],
        metrics["ndcg@10"],
        -metrics["weakest_positive_rank"],
    )


def _atomic_numpy_save(path, matrix):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, np.asarray(matrix, dtype=np.int32))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _load_order(path, n_queries):
    if not path.exists():
        return None
    try:
        matrix = np.load(path)
    except (OSError, ValueError, EOFError):
        log.warning("Ignoring invalid SOTA ranking cache: %s", path)
        return None
    if matrix.shape != (n_queries, MAX_K):
        return None
    return matrix.astype(np.int64).tolist()


def _as_matrix(order):
    matrix = np.full((len(order), MAX_K), -1, dtype=np.int32)
    for row_index, values in enumerate(order):
        values = list(values)[:MAX_K]
        matrix[row_index, : len(values)] = values
    return matrix


def _cached_method(cache_path, n_queries, builder):
    cached = _load_order(cache_path, n_queries)
    if cached is not None:
        log.info("Reusing SOTA ranking cache: %s", cache_path)
        return cached
    order = builder()
    _atomic_numpy_save(cache_path, _as_matrix(order))
    return order


def _bm25_order_cached(cache_path, scorer, rows, k):
    cached = _load_order(cache_path, len(rows))
    if cached is not None:
        log.info("Reusing UKB BM25 cache: %s", cache_path)
        return cached
    order = [scorer.rank(node.content, k=k) for node, _, _ in rows]
    _atomic_numpy_save(cache_path, _as_matrix(order))
    return order


def _latency_stats(values):
    if not values:
        return None
    milliseconds = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "mean": round(float(milliseconds.mean()), 3),
        "p50": round(float(np.percentile(milliseconds, 50)), 3),
        "p95": round(float(np.percentile(milliseconds, 95)), 3),
        "n_queries": len(values),
        "query_encoding_included": False,
    }


def _benchmark_core_latency(
    bm25_scorer,
    index,
    query_vectors,
    rows,
    lexical_weight,
):
    timings = {method: [] for method in CORE_METHODS}
    for query_index, (node, _, _) in enumerate(rows[: min(200, len(rows))]):
        start = time.perf_counter()
        _, dense = index.search(query_vectors[query_index : query_index + 1], MAX_K)
        timings["dense"].append(time.perf_counter() - start)

        start = time.perf_counter()
        lexical = bm25_scorer.rank(node.content, k=MAX_K)
        timings["bm25"].append(time.perf_counter() - start)

        start = time.perf_counter()
        _rrf([[dense[0].tolist()], [lexical]], [1.0, 1.0])
        timings["hybrid_rrf"].append(
            timings["dense"][-1] + timings["bm25"][-1]
            + time.perf_counter() - start
        )

        start = time.perf_counter()
        _rrf([[dense[0].tolist()], [lexical]], [1.0, lexical_weight])
        timings["hybrid_tuned"].append(
            timings["dense"][-1] + timings["bm25"][-1]
            + time.perf_counter() - start
        )
    return {method: _latency_stats(values) for method, values in timings.items()}


def _cross_encoder_order(rows, base_order, documents, device):
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CE_MODEL, device=str(device) if device else None)
    output = []
    for query_index, ((query, _, _), candidates) in enumerate(
        zip(rows, base_order)
    ):
        candidates = [int(doc) for doc in candidates if int(doc) >= 0]
        scores = model.predict(
            [(query.content, documents[doc].content) for doc in candidates],
            batch_size=64,
            show_progress_bar=False,
        )
        output.append([candidates[i] for i in np.argsort(-scores)])
        if (query_index + 1) % 100 == 0:
            log.info("Cross-encoder reranked %d/%d queries", query_index + 1, len(rows))
    return output


def _splade_order(dataset, rows, id_to_idx, device):
    scorer = SpladeScorer(dataset, device=device)
    if not scorer.available():
        return None
    output = []
    for query_index, (query, _, _) in enumerate(rows):
        document_ids = scorer.top_doc_ids(query.content, k=MAX_K)
        output.append(
            [id_to_idx[doc_id] for doc_id in document_ids if doc_id in id_to_idx]
        )
        if (query_index + 1) % 100 == 0:
            log.info("SPLADE ranked %d/%d queries", query_index + 1, len(rows))
    return output


def _colbert_order(engine, rows):
    if engine.colbert is None:
        return None
    exact = {}
    prefix = {}
    for document_index, node in enumerate(engine.nodes):
        exact.setdefault(node.content, document_index)
        prefix.setdefault(node.content[:100], document_index)
    output = []
    for query_index, (query, _, _) in enumerate(rows):
        results = engine.colbert.search(query.content, k=MAX_K)
        ranked = []
        for result in results:
            content = result.get("content", "")
            document_index = exact.get(content, prefix.get(content[:100]))
            if document_index is not None and document_index not in ranked:
                ranked.append(document_index)
        output.append(ranked)
        if (query_index + 1) % 100 == 0:
            log.info("ColBERT ranked %d/%d queries", query_index + 1, len(rows))
    return output


def _write_candidates(path, query_ids, order, document_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
            for query_id, ranked in zip(query_ids, order):
                row = {
                    "query_id": query_id,
                    "candidate_doc_ids": [
                        document_ids[int(doc)]
                        for doc in ranked[:MAX_K]
                        if int(doc) >= 0
                    ],
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_per_query(path, rows, gold, document_ids, method_rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for query_index, ((query, _, _), positives) in enumerate(zip(rows, gold)):
                record = {
                    "query_id": query.node_id,
                    "gold_doc_ids": [document_ids[int(doc)] for doc in positives],
                    "methods": {
                        method: metrics[query_index]
                        for method, metrics in method_rows.items()
                    },
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run(
    dataset,
    run_id="sota_v1",
    limit=15000,
    methods=CORE_METHODS,
    device=None,
):
    methods = tuple(dict.fromkeys(methods))
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"Unknown SOTA baseline methods: {sorted(unknown)}")
    engine = CoreEngine(source=dataset)
    documents = _reconstruct(engine.node_index).astype("float32")
    faiss.normalize_L2(documents)
    index = faiss.IndexFlatIP(documents.shape[1])
    index.add(documents)

    splits = _splits(engine, _hard_membership(engine))
    split_seed = {"train": 101, "val": 202, "test": 303}
    splits = {
        name: _cap(rows, limit, split_seed[name])
        for name, rows in splits.items()
    }
    if not all(splits.values()):
        raise RuntimeError(f"{dataset}: train/val/test split is incomplete")

    split_manifest = _split_manifest(splits)
    fingerprint_payload = {
        "version": 2,
        "dataset": dataset,
        "encoder": ENCODER_NAME,
        "document_manifest": _document_manifest(engine, documents),
        "splits": split_manifest,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    l1_cache = (
        Path("data") / "ukb_storage" / dataset / "cache" / "L1" / fingerprint
    )
    query_paths = {
        name: l1_cache / f"queries_{name}.npz" for name in splits
    }
    encoder = None
    if not all(
        _query_cache_valid(
            query_paths[name],
            splits[name],
            dense_seeds=10,
            dimension=documents.shape[1],
        )
        for name in splits
    ):
        encoder = DenseEncoder(ENCODER_NAME)
    prepared = {
        name: _prepare_cached(
            query_paths[name],
            splits[name],
            encoder,
            index,
            engine.node_id_to_idx,
            dense_seeds=10,
        )
        for name in splits
    }

    sota_cache = (
        Path("data") / "ukb_storage" / dataset / "cache" / "SOTA" / fingerprint
    )
    result_dir = (
        Path("data")
        / "ukb_storage"
        / dataset
        / "results"
        / "baselines"
        / run_id
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    run_signature = hashlib.sha256(
        json.dumps(
            {
                "version": 1,
                "fingerprint": fingerprint,
                "methods": methods,
                "rrf_weights": RRF_WEIGHTS,
                "rrf_k": RRF_K,
                "ce_model": CE_MODEL,
                "splade_model": SPLADE_MODEL,
                "colbert_model": COLBERT_MODEL,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]

    dense_orders = {
        split: index.search(prepared[split][0], MAX_K)[1].tolist()
        for split in ("val", "test")
    }
    bm25_scorer = _InvertedBM25(
        engine.bm25,
        [
            node.content
            for split in ("val", "test")
            for node, _, _ in splits[split]
        ],
    )
    bm25_orders = {
        split: _bm25_order_cached(
            l1_cache / f"bm25_{split}_{MAX_K}.npy",
            bm25_scorer,
            splits[split],
            MAX_K,
        )
        for split in ("val", "test")
    }

    validation_hybrids = {}
    validation_orders = {}
    for lexical_weight in RRF_WEIGHTS:
        label = f"dense1_bm25{lexical_weight:g}"
        order = _rrf(
            [dense_orders["val"], bm25_orders["val"]],
            [1.0, lexical_weight],
        )
        validation_orders[label] = order
        validation_hybrids[label] = _evaluate(
            order, prepared["val"][2]
        )[0]
    selected_hybrid = max(
        validation_hybrids,
        key=lambda label: _selection_key(validation_hybrids[label]),
    )
    lexical_weight = float(selected_hybrid.rsplit("bm25", 1)[1])

    orders = {}
    statuses = {}
    if "dense" in methods:
        orders["dense"] = dense_orders["test"]
        statuses["dense"] = {"status": "completed"}
    if "bm25" in methods:
        orders["bm25"] = bm25_orders["test"]
        statuses["bm25"] = {"status": "completed"}
    if "hybrid_rrf" in methods:
        orders["hybrid_rrf"] = _rrf(
            [dense_orders["test"], bm25_orders["test"]], [1.0, 1.0]
        )
        statuses["hybrid_rrf"] = {
            "status": "completed",
            "weights": {"dense": 1.0, "bm25": 1.0},
        }
    tuned_test = _rrf(
        [dense_orders["test"], bm25_orders["test"]],
        [1.0, lexical_weight],
    )
    if "hybrid_tuned" in methods:
        orders["hybrid_tuned"] = tuned_test
        statuses["hybrid_tuned"] = {
            "status": "completed",
            "weights": {"dense": 1.0, "bm25": lexical_weight},
            "selected_on": "val",
        }

    if "hybrid_ce" in methods:
        cache_path = sota_cache / f"hybrid_ce_{run_signature}.npy"
        orders["hybrid_ce"] = _cached_method(
            cache_path,
            len(splits["test"]),
            lambda: _cross_encoder_order(
                splits["test"], tuned_test, engine.nodes, device
            ),
        )
        statuses["hybrid_ce"] = {
            "status": "completed",
            "base_pool": "hybrid_tuned@100",
            "model": CE_MODEL,
        }

    if "splade" in methods:
        scorer = SpladeScorer(dataset, device=device)
        if not scorer.available():
            statuses["splade"] = {
                "status": "skipped",
                "reason": (
                    f"missing data/ukb_storage/{dataset}/splade_doc_embs.pkl"
                ),
                "model": SPLADE_MODEL,
            }
        else:
            cache_path = sota_cache / f"splade_{run_signature}.npy"
            orders["splade"] = _cached_method(
                cache_path,
                len(splits["test"]),
                lambda: _splade_order(
                    dataset,
                    splits["test"],
                    engine.node_id_to_idx,
                    device,
                ),
            )
            statuses["splade"] = {
                "status": "completed",
                "model": SPLADE_MODEL,
            }

    if "colbert" in methods:
        if engine.colbert is None:
            statuses["colbert"] = {
                "status": "skipped",
                "reason": "no real ColBERT document index; lexical fallback disabled",
                "model": COLBERT_MODEL,
            }
        else:
            cache_path = sota_cache / f"colbert_{run_signature}.npy"
            orders["colbert"] = _cached_method(
                cache_path,
                len(splits["test"]),
                lambda: _colbert_order(engine, splits["test"]),
            )
            statuses["colbert"] = {
                "status": "completed",
                "model": COLBERT_MODEL,
            }

    summaries = {}
    method_rows = {}
    for method, order in orders.items():
        summaries[method], method_rows[method] = _evaluate(
            order, prepared["test"][2]
        )

    baseline_method = "dense"
    if baseline_method not in method_rows:
        dense_summary, dense_rows = _evaluate(
            dense_orders["test"], prepared["test"][2]
        )
        summaries["_dense_reference"] = dense_summary
    else:
        dense_rows = method_rows[baseline_method]
    significance = {}
    for method, rows in method_rows.items():
        if method == baseline_method:
            continue
        significance[method] = {
            f"fullcov@{k}": paired(
                [row["fullcov"][str(k)] for row in rows],
                [row["fullcov"][str(k)] for row in dense_rows],
            )
            for k in (20, 100)
        }

    latency = _benchmark_core_latency(
        bm25_scorer,
        index,
        prepared["test"][0],
        splits["test"],
        lexical_weight,
    )
    document_ids = [node.node_id for node in engine.nodes]
    query_ids = prepared["test"][3]
    for method, order in orders.items():
        _write_candidates(
            result_dir / f"candidates_{method}.jsonl.gz",
            query_ids,
            order,
            document_ids,
        )
    _write_per_query(
        result_dir / "per_query.jsonl",
        splits["test"],
        prepared["test"][2],
        document_ids,
        method_rows,
    )

    output = {
        "dataset": dataset,
        "run_id": run_id,
        "run_signature": run_signature,
        "protocol": {
            "candidate_budget": MAX_K,
            "budgets": list(KS),
            "selection_split": "val",
            "test_used_after_selection": True,
            "limit_per_split": limit,
            "split_manifest": split_manifest,
            "cache_fingerprint": fingerprint,
            "cache_manifest": fingerprint_payload,
            "query_encoding_latency_excluded": True,
        },
        "models": {
            "dense": ENCODER_NAME,
            "bm25": "rank_bm25.BM25Okapi",
            "cross_encoder": CE_MODEL,
            "splade": SPLADE_MODEL,
            "colbert": COLBERT_MODEL,
        },
        "validation_hybrid_sweep": validation_hybrids,
        "selected_hybrid": {
            "label": selected_hybrid,
            "weights": {"dense": 1.0, "bm25": lexical_weight},
        },
        "method_status": statuses,
        "test": summaries,
        "significance_vs_dense": significance,
        "retrieval_latency_ms": latency,
        "result_dir": str(result_dir),
    }
    _atomic_json_dump(output, result_dir / "summary.json")
    log.info(
        "[%s] SOTA baselines: %s",
        dataset,
        " | ".join(
            f"{method} FCov@100={metrics['fullcov']['100']:.2f}"
            for method, metrics in summaries.items()
        ),
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Matched-corpus SOTA retrieval baselines."
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--run-id", default="sota_v1")
    parser.add_argument("--limit", type=int, default=15000)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=ALL_METHODS,
        default=list(CORE_METHODS),
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    for dataset in args.datasets:
        run(
            dataset,
            run_id=args.run_id,
            limit=args.limit,
            methods=args.methods,
            device=args.device,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
