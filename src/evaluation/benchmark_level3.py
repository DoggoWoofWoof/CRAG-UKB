"""
Level 3 — Traversal and Context Curation Benchmark
==================================================

Evaluates the contribution of C-RAG's graph traversal layer after Level 1
partition selection and Level 2 document reranking.

Default variants:
    level2_only                 Top Level 2 documents, no graph expansion
    traverse_no_synthetic       Priority traversal over original graph edges only
    traverse_with_synthetic     Same traversal, but allows KNN synthetic graph edges

This benchmark is retrieval/context focused. Generation EM/F1 is handled by
benchmark_generation.py after contexts are exported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.alignment.mlp_encoder import TextPartitionMLP
from src.core.encoders import DenseEncoder
from src.core.engine import CoreEngine
from src.strategies.crag import CRAG
from src.evaluation.benchmark_reranking import (
    BEST_HNM_CHECKPOINTS,
    K_VALUES,
    _get_split_queries,
)


SUPPORTED_SEED_RERANKERS = {"no_rerank", "bm25", "faiss_dense", "splade", "external"}
AUTO_METRIC = "mrr"


def _load_best_mlp(dataset: str, checkpoint_path: Optional[str], device: torch.device):
    path = checkpoint_path or BEST_HNM_CHECKPOINTS.get(dataset)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"No MLP checkpoint found for dataset '{dataset}': {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    first_weight = state_dict["net.0.weight"]
    second_weight = state_dict["net.3.weight"]

    model = TextPartitionMLP(
        input_dim=int(checkpoint.get("input_dim", first_weight.shape[1])),
        hidden_dim=int(checkpoint.get("hidden_dim", first_weight.shape[0])),
        output_dim=int(checkpoint.get("output_dim", second_weight.shape[0])),
        dropout=float(checkpoint.get("dropout", 0.4)),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, path


def _rerank_faiss_dense(engine: CoreEngine, query_vector: np.ndarray, pool: Sequence[Any], top_k: int):
    indices = []
    valid_nodes = []
    for node in pool:
        idx = engine.node_id_to_idx.get(node.node_id)
        if idx is not None:
            indices.append(int(idx))
            valid_nodes.append(node)

    if not indices:
        return []

    node_vectors = np.stack([engine.node_index.reconstruct(idx) for idx in indices])
    qv = query_vector.flatten()
    qv_norm = np.linalg.norm(qv) + 1e-8
    norms = np.linalg.norm(node_vectors, axis=1) + 1e-8
    scores = np.dot(node_vectors, qv) / (norms * qv_norm)
    top_indices = np.argsort(-scores)[: min(top_k, len(scores))]
    return [(valid_nodes[idx], float(scores[idx])) for idx in top_indices]


def _rerank_bm25(engine: CoreEngine, query: str, pool: Sequence[Any], top_k: int):
    indices = []
    valid_nodes = []
    for node in pool:
        idx = engine.node_id_to_idx.get(node.node_id)
        if idx is not None:
            indices.append(int(idx))
            valid_nodes.append(node)
    if not indices:
        return []

    tokenized_query = query.lower().split()
    if hasattr(engine.bm25, "get_batch_scores"):
        scores = np.array(engine.bm25.get_batch_scores(tokenized_query, indices))
    else:
        all_scores = engine.bm25.get_scores(tokenized_query)
        scores = np.array([all_scores[idx] for idx in indices])
    top_indices = np.argsort(-scores)[: min(top_k, len(scores))]
    return [(valid_nodes[idx], float(scores[idx])) for idx in top_indices]


def _prepare_splade_resources(engine: CoreEngine, dataset: str, device: torch.device):
    cache_path = os.path.join("data", "ukb_storage", dataset, "splade_doc_embs.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"SPLADE seed reranking requested, but cache is missing: {cache_path}"
        )
    if not hasattr(engine, "_level3_splade_matrix"):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        engine._level3_splade_matrix = data["matrix"]
        engine._level3_splade_id_to_idx = data["id_to_idx"]

    if not hasattr(engine, "_level3_splade_tokenizer"):
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model_name = "naver/splade-cocondenser-ensembledistil"
        engine._level3_splade_tokenizer = AutoTokenizer.from_pretrained(model_name)
        engine._level3_splade_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        engine._level3_splade_model.eval()


def _rerank_splade(
    engine: CoreEngine,
    dataset: str,
    query: str,
    pool: Sequence[Any],
    top_k: int,
    device: torch.device,
):
    import scipy.sparse

    _prepare_splade_resources(engine, dataset, device)
    tokenizer = engine._level3_splade_tokenizer
    model = engine._level3_splade_model
    matrix = engine._level3_splade_matrix
    id_to_idx = engine._level3_splade_id_to_idx

    inputs = tokenizer(
        [query], return_tensors="pt", padding=True, truncation=True, max_length=64
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        relu_log = torch.log(1 + torch.relu(logits))
        mask = inputs.attention_mask.unsqueeze(-1)
        query_sparse = torch.max(relu_log * mask, dim=1).values.cpu().numpy()[0]

    pool_indices = []
    valid_nodes = []
    for node in pool:
        idx = id_to_idx.get(node.node_id)
        if idx is not None:
            pool_indices.append(idx)
            valid_nodes.append(node)

    if not valid_nodes:
        return []

    pool_submatrix = matrix[pool_indices]
    query_csr = scipy.sparse.csr_matrix(query_sparse)
    scores = pool_submatrix.dot(query_csr.T).toarray().flatten()
    top_indices = np.argsort(-scores)[: min(top_k, len(scores))]
    return [(valid_nodes[idx], float(scores[idx])) for idx in top_indices]


def _load_candidate_jsonl(path: Optional[str], engine: CoreEngine) -> Dict[str, List[Tuple[Any, float]]]:
    if not path:
        return {}

    node_by_id = {node.node_id: node for node in engine.nodes}
    candidates_by_question: Dict[str, List[Tuple[Any, float]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row.get("id") or row.get("question_id")
            candidate_rows = row.get("candidates") or row.get("retrieved") or []
            parsed = []
            for rank, item in enumerate(candidate_rows):
                if isinstance(item, str):
                    node_id, score = item, float(len(candidate_rows) - rank)
                else:
                    node_id = item.get("node_id") or item.get("id")
                    score = item.get("score", float(len(candidate_rows) - rank))
                node = node_by_id.get(node_id)
                if node is not None:
                    parsed.append((node, float(score)))
            if question_id and parsed:
                candidates_by_question[question_id] = parsed
    return candidates_by_question


def _choose_seed_reranker(
    dataset: str,
    split: str,
    requested: str,
    candidate_jsonl: Optional[str],
) -> str:
    if candidate_jsonl:
        return "external"
    if requested != "auto":
        return requested

    summary_path = os.path.join("results", "level_2", f"{dataset}_level_2_reranking.json")
    if not os.path.exists(summary_path):
        return "faiss_dense"

    with open(summary_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    best_method = "faiss_dense"
    best_score = float("-inf")
    for method, split_results in results.items():
        if method not in SUPPORTED_SEED_RERANKERS:
            continue
        metrics = split_results.get(split) if isinstance(split_results, dict) else None
        if not isinstance(metrics, dict):
            continue
        score = metrics.get(AUTO_METRIC, metrics.get("recall@10", 0.0))
        if score is not None and float(score) > best_score:
            best_method = method
            best_score = float(score)

    if best_method == "splade":
        cache_path = os.path.join("data", "ukb_storage", dataset, "splade_doc_embs.pkl")
        if not os.path.exists(cache_path):
            return "faiss_dense"
    return best_method


def _rerank_seed_candidates(
    engine: CoreEngine,
    dataset: str,
    method: str,
    query: str,
    query_vector: np.ndarray,
    pool: Sequence[Any],
    top_k: int,
    device: torch.device,
):
    if method == "faiss_dense":
        return _rerank_faiss_dense(engine, query_vector, pool, top_k)
    if method == "bm25":
        return _rerank_bm25(engine, query, pool, top_k)
    if method == "splade":
        return _rerank_splade(engine, dataset, query, pool, top_k, device)
    if method == "no_rerank":
        return [(node, float(top_k - rank)) for rank, node in enumerate(pool[:top_k])]
    raise ValueError(f"Unsupported Level 3 seed reranker: {method}")


def _level2_candidates(
    engine: CoreEngine,
    dataset: str,
    encoder: DenseEncoder,
    mlp_model,
    question_node,
    top_k_partitions: int,
    top_k_entry: int,
    seed_reranker: str,
    device: torch.device,
    external_candidates: Optional[List[Tuple[Any, float]]] = None,
):
    import faiss

    query_vector = encoder.encode([question_node.content]).astype("float32")
    faiss.normalize_L2(query_vector)

    with torch.no_grad():
        projected = mlp_model(
            torch.tensor(query_vector, dtype=torch.float32).to(device)
        ).cpu().numpy()

    partition_results = engine.search_centroids(projected, k=top_k_partitions)
    partition_ids = [partition_id for partition_id, _ in partition_results]

    pool = []
    for partition_id in partition_ids:
        pool.extend(engine.get_partition_nodes(partition_id))

    candidate_limit = top_k_entry * top_k_partitions
    if external_candidates:
        candidates = external_candidates[:candidate_limit]
    else:
        candidates = _rerank_seed_candidates(
            engine=engine,
            dataset=dataset,
            method=seed_reranker,
            query=question_node.content,
            query_vector=query_vector,
            pool=pool,
            top_k=candidate_limit,
            device=device,
        )
    return query_vector, partition_ids, pool, candidates


def _retrieval_metrics(
    retrieved_ids: Sequence[str],
    gt_doc_ids: Sequence[str],
    engine: CoreEngine,
    selected_trace: Optional[Sequence[Dict[str, Any]]] = None,
    initial_partitions: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    gt_set = set(gt_doc_ids)
    retrieved_set = set(retrieved_ids)
    num_gt = len(gt_set)

    for k in K_VALUES:
        top_k = list(retrieved_ids[:k])
        top_k_set = set(top_k)
        hits = len(gt_set & top_k_set)

        metrics[f"recall@{k}"] = hits / num_gt if num_gt else 0.0
        metrics[f"precision@{k}"] = hits / k
        metrics[f"full_coverage@{k}"] = float(bool(gt_set and gt_set.issubset(top_k_set)))

        dcg = 0.0
        for idx, node_id in enumerate(top_k):
            if node_id in gt_set:
                dcg += 1.0 / np.log2(idx + 2)
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(num_gt, k)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    metrics["mrr"] = 0.0
    for idx, node_id in enumerate(retrieved_ids):
        if node_id in gt_set:
            metrics["mrr"] = 1.0 / (idx + 1)
            break

    total_hits = len(gt_set & retrieved_set)
    metrics["context_recall"] = total_hits / num_gt if num_gt else 0.0
    metrics["context_precision"] = total_hits / len(retrieved_ids) if retrieved_ids else 0.0
    metrics["oracle_hit"] = float(total_hits > 0)
    metrics["context_nodes"] = float(len(retrieved_ids))
    metrics["gt_doc_count"] = float(num_gt)
    metrics["connected_full_coverage"] = float(
        _connected_full_coverage(engine, retrieved_ids, gt_doc_ids)
    )
    retrieved_partitions = {
        int(engine.partition_map[node_id])
        for node_id in retrieved_ids
        if node_id in engine.partition_map
    }
    initial_partition_set = set(int(pid) for pid in (initial_partitions or []))
    metrics["context_partition_count"] = float(len(retrieved_partitions))
    metrics["new_partition_count"] = float(len(retrieved_partitions - initial_partition_set))

    depth_by_node = {}
    for row in selected_trace or []:
        node_id = row.get("node_id")
        if node_id is not None:
            depth_by_node[node_id] = row.get("depth", 0)
    hit_depths = [depth_by_node[node_id] for node_id in gt_set if node_id in depth_by_node]
    metrics["min_hit_depth"] = float(min(hit_depths)) if hit_depths else -1.0
    metrics["max_hit_depth"] = float(max(hit_depths)) if hit_depths else -1.0
    return metrics


def _connected_full_coverage(
    engine: CoreEngine, retrieved_ids: Sequence[str], gt_doc_ids: Sequence[str]
) -> bool:
    gt_set = set(gt_doc_ids)
    retrieved_set = set(retrieved_ids)
    if not gt_set or not gt_set.issubset(retrieved_set):
        return False
    if len(gt_set) == 1:
        return True

    adjacency = {node_id: set() for node_id in retrieved_set}
    for node_id in retrieved_set:
        for neighbor in engine.get_neighbors(node_id):
            if neighbor.node_id in retrieved_set:
                adjacency[node_id].add(neighbor.node_id)
                adjacency[neighbor.node_id].add(node_id)

    start = next(iter(gt_set))
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor_id in adjacency.get(current, set()):
            if neighbor_id not in seen:
                seen.add(neighbor_id)
                stack.append(neighbor_id)
    return gt_set.issubset(seen)


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_variant: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        variant = row["variant"]
        for key, value in row.items():
            if key in {"variant", "question_id"}:
                continue
            if isinstance(value, (int, float)):
                by_variant[variant][key].append(float(value))

    percent_prefixes = (
        "recall@",
        "precision@",
        "full_coverage@",
        "ndcg@",
    )
    percent_keys = {
        "mrr",
        "context_recall",
        "context_precision",
        "oracle_hit",
        "connected_full_coverage",
    }

    summary: Dict[str, Dict[str, float]] = {}
    for variant, metrics in by_variant.items():
        summary[variant] = {}
        for key, values in metrics.items():
            if not values:
                continue
            mean_value = float(np.mean(values))
            if key in percent_keys or key.startswith(percent_prefixes):
                mean_value *= 100.0
            summary[variant][key] = round(mean_value, 2)
    return summary


def _save_summary_csv(summary: Dict[str, Dict[str, float]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant"]
    for metrics in summary.values():
        for key in metrics:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for variant, metrics in summary.items():
            writer.writerow({"variant": variant, **metrics})


def run_level3_benchmark(
    dataset: str = "squad",
    split: str = "test",
    limit: Optional[int] = None,
    seed_reranker: str = "auto",
    candidate_jsonl: Optional[str] = None,
    top_k_partitions: int = 20,
    top_k_entry: int = 10,
    max_traverse_steps: int = 20,
    score_threshold: float = 0.3,
    expand_threshold: Optional[float] = None,
    max_context_nodes: int = 10,
    beam_width: int = 50,
    expand_top_neighbors: int = 8,
    min_context_nodes: int = 3,
    max_dynamic_partitions: int = 3,
    partition_admission_threshold: float = 0.35,
    partition_fetch_k: int = 5,
    l2_score_weight: float = 0.25,
    partition_prior_weight: float = 0.15,
    path_coherence_weight: float = 0.10,
    redundancy_penalty_weight: float = 0.10,
    depth_penalty_weight: float = 0.03,
    partition_balance_weight: float = 0.04,
    checkpoint_path: Optional[str] = None,
    output_dir: str = "results/level_3",
    save_examples: bool = False,
) -> Dict[str, Any]:
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp, resolved_checkpoint = _load_best_mlp(dataset, checkpoint_path, device)

    splits = _get_split_queries(engine, dataset)
    queries = splits.get(split, [])
    if limit is not None:
        queries = queries[:limit]
    if not queries:
        raise RuntimeError(f"No {split} queries found for dataset '{dataset}'.")

    resolved_seed_reranker = _choose_seed_reranker(
        dataset=dataset,
        split=split,
        requested=seed_reranker,
        candidate_jsonl=candidate_jsonl,
    )
    external_candidate_lookup = _load_candidate_jsonl(candidate_jsonl, engine)

    variants = {
        "level2_only": None,
        "node_traverse_no_synthetic": {
            "exclude_synthetic": True,
            "dynamic_partition_expansion": False,
        },
        "dynamic_partition_no_synthetic": {
            "exclude_synthetic": True,
            "dynamic_partition_expansion": True,
        },
        "dynamic_partition_with_synthetic": {
            "exclude_synthetic": False,
            "dynamic_partition_expansion": True,
        },
    }

    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []

    for question_node, gt_partitions, gt_doc_ids in tqdm(
        queries, desc=f"Level 3 ({dataset}/{split})"
    ):
        t0 = time.time()
        query_vector, partition_ids, pool, candidates = _level2_candidates(
            engine=engine,
            dataset=dataset,
            encoder=encoder,
            mlp_model=mlp,
            question_node=question_node,
            top_k_partitions=top_k_partitions,
            top_k_entry=top_k_entry,
            seed_reranker=resolved_seed_reranker,
            device=device,
            external_candidates=external_candidate_lookup.get(question_node.node_id),
        )
        level12_latency_ms = (time.time() - t0) * 1000.0

        for variant, variant_cfg in variants.items():
            v0 = time.time()
            traversal_stats: Dict[str, Any] = {}

            if variant == "level2_only":
                nodes = [node for node, _ in candidates[:max_context_nodes]]
            else:
                retriever = CRAG(
                    engine=engine,
                    llm=None,
                    encoder=encoder,
                    mode="mlp",
                    reranker="faiss",
                    top_k_partitions=top_k_partitions,
                    top_k_entry=top_k_entry,
                    max_traverse_steps=max_traverse_steps,
                    score_threshold=score_threshold,
                    expand_threshold=expand_threshold,
                    max_context_nodes=max_context_nodes,
                    beam_width=beam_width,
                    expand_top_neighbors=expand_top_neighbors,
                    min_context_nodes=min_context_nodes,
                    dynamic_partition_expansion=bool(
                        variant_cfg["dynamic_partition_expansion"]
                    ),
                    max_dynamic_partitions=max_dynamic_partitions,
                    partition_admission_threshold=partition_admission_threshold,
                    partition_fetch_k=partition_fetch_k,
                    l2_score_weight=l2_score_weight,
                    partition_prior_weight=partition_prior_weight,
                    path_coherence_weight=path_coherence_weight,
                    redundancy_penalty_weight=redundancy_penalty_weight,
                    depth_penalty_weight=depth_penalty_weight,
                    partition_balance_weight=partition_balance_weight,
                    exclude_synthetic_edges=bool(variant_cfg["exclude_synthetic"]),
                    mlp_encoder=mlp,
                )
                nodes = retriever.traverse_candidates(
                    question_node.content,
                    query_vector,
                    candidates,
                    selected_partitions=partition_ids,
                )
                traversal_stats = retriever._last_traversal_trace

            variant_latency_ms = (time.time() - v0) * 1000.0
            retrieved_ids = [node.node_id for node in nodes]
            metrics = _retrieval_metrics(
                retrieved_ids=retrieved_ids,
                gt_doc_ids=gt_doc_ids,
                engine=engine,
                selected_trace=traversal_stats.get("selected", []),
                initial_partitions=partition_ids,
            )
            rows.append({
                "question_id": question_node.node_id,
                "variant": variant,
                **metrics,
                "level12_latency_ms": level12_latency_ms,
                "variant_latency_ms": variant_latency_ms,
                "total_latency_ms": level12_latency_ms + variant_latency_ms,
                "pool_size": float(len(pool)),
                "candidate_count": float(len(candidates)),
                "selected_count": float(len(retrieved_ids)),
                "visited_count": float(traversal_stats.get("visited_count", 0)),
                "expanded_count": float(traversal_stats.get("expanded_count", 0)),
                "pruned_count": float(traversal_stats.get("pruned_count", 0)),
                "synthetic_edges_skipped": float(
                    traversal_stats.get("synthetic_edges_skipped", 0)
                ),
                "fallback_nodes_added": float(traversal_stats.get("fallback_nodes_added", 0)),
                "cross_partition_edges_seen": float(
                    traversal_stats.get("cross_partition_edges_seen", 0)
                ),
                "dynamic_partitions_admitted": float(
                    traversal_stats.get("dynamic_partitions_admitted", 0)
                ),
                "dynamic_partitions_rejected": float(
                    traversal_stats.get("dynamic_partitions_rejected", 0)
                ),
                "partition_fetch_nodes_queued": float(
                    traversal_stats.get("partition_fetch_nodes_queued", 0)
                ),
            })

            if save_examples:
                examples.append({
                    "question_id": question_node.node_id,
                    "variant": variant,
                    "question": question_node.content,
                    "gt_doc_ids": gt_doc_ids,
                    "gt_partitions": gt_partitions,
                    "retrieved_doc_ids": retrieved_ids,
                    "selected_trace": traversal_stats.get("selected", []),
                    "partition_ids": partition_ids,
                    "admitted_partitions": traversal_stats.get("admitted_partitions", []),
                    "dynamic_partitions": traversal_stats.get("dynamic_partitions", []),
                })

    summary = _summarize(rows)
    config = {
        "dataset": dataset,
        "split": split,
        "limit": limit,
        "seed_reranker": resolved_seed_reranker,
        "requested_seed_reranker": seed_reranker,
        "candidate_jsonl": candidate_jsonl,
        "top_k_partitions": top_k_partitions,
        "top_k_entry": top_k_entry,
        "max_traverse_steps": max_traverse_steps,
        "score_threshold": score_threshold,
        "expand_threshold": expand_threshold if expand_threshold is not None else score_threshold,
        "max_context_nodes": max_context_nodes,
        "beam_width": beam_width,
        "expand_top_neighbors": expand_top_neighbors,
        "min_context_nodes": min_context_nodes,
        "max_dynamic_partitions": max_dynamic_partitions,
        "partition_admission_threshold": partition_admission_threshold,
        "partition_fetch_k": partition_fetch_k,
        "l2_score_weight": l2_score_weight,
        "partition_prior_weight": partition_prior_weight,
        "path_coherence_weight": path_coherence_weight,
        "redundancy_penalty_weight": redundancy_penalty_weight,
        "depth_penalty_weight": depth_penalty_weight,
        "partition_balance_weight": partition_balance_weight,
        "checkpoint": resolved_checkpoint,
        "num_queries": len(queries),
    }
    result = {"config": config, "summary": summary}
    if save_examples:
        result["examples"] = examples

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    json_path = os.path.join(output_dir, f"{dataset}_level_3_traversal.json")
    csv_path = os.path.join(output_dir, f"{dataset}_level_3_traversal.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    _save_summary_csv(summary, csv_path)

    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Level 3 traversal benchmark")
    parser.add_argument("--dataset", default="squad", choices=sorted(BEST_HNM_CHECKPOINTS))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seed-reranker",
        default="auto",
        choices=("auto", "no_rerank", "bm25", "faiss_dense", "splade"),
        help="Level 2 reranker used to seed Level 3. 'auto' chooses the best saved Level 2 summary.",
    )
    parser.add_argument(
        "--candidate-jsonl",
        default=None,
        help="Optional per-query candidate JSONL with id/question_id and candidates[{node_id, score}].",
    )
    parser.add_argument("--top-k-partitions", type=int, default=20)
    parser.add_argument("--top-k-entry", type=int, default=10)
    parser.add_argument("--max-traverse-steps", type=int, default=20)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--expand-threshold", type=float, default=None)
    parser.add_argument("--max-context-nodes", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=50)
    parser.add_argument("--expand-top-neighbors", type=int, default=8)
    parser.add_argument("--min-context-nodes", type=int, default=3)
    parser.add_argument("--max-dynamic-partitions", type=int, default=3)
    parser.add_argument("--partition-admission-threshold", type=float, default=0.35)
    parser.add_argument("--partition-fetch-k", type=int, default=5)
    parser.add_argument("--l2-score-weight", type=float, default=0.25)
    parser.add_argument("--partition-prior-weight", type=float, default=0.15)
    parser.add_argument("--path-coherence-weight", type=float, default=0.10)
    parser.add_argument("--redundancy-penalty-weight", type=float, default=0.10)
    parser.add_argument("--depth-penalty-weight", type=float, default=0.03)
    parser.add_argument("--partition-balance-weight", type=float, default=0.04)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="results/level_3")
    parser.add_argument("--save-examples", action="store_true")
    args = parser.parse_args(argv)

    result = run_level3_benchmark(
        dataset=args.dataset,
        split=args.split,
        limit=args.limit,
        seed_reranker=args.seed_reranker,
        candidate_jsonl=args.candidate_jsonl,
        top_k_partitions=args.top_k_partitions,
        top_k_entry=args.top_k_entry,
        max_traverse_steps=args.max_traverse_steps,
        score_threshold=args.score_threshold,
        expand_threshold=args.expand_threshold,
        max_context_nodes=args.max_context_nodes,
        beam_width=args.beam_width,
        expand_top_neighbors=args.expand_top_neighbors,
        min_context_nodes=args.min_context_nodes,
        max_dynamic_partitions=args.max_dynamic_partitions,
        partition_admission_threshold=args.partition_admission_threshold,
        partition_fetch_k=args.partition_fetch_k,
        l2_score_weight=args.l2_score_weight,
        partition_prior_weight=args.partition_prior_weight,
        path_coherence_weight=args.path_coherence_weight,
        redundancy_penalty_weight=args.redundancy_penalty_weight,
        depth_penalty_weight=args.depth_penalty_weight,
        partition_balance_weight=args.partition_balance_weight,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        save_examples=args.save_examples,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
