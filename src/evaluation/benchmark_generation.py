"""
Paper 2 generation benchmark utilities.

This module bridges the Level 1/2 retrieval pipeline to answer-generation
evaluation. It can export JSONL records containing questions, retrieved
contexts, gold document IDs, optional answer strings, and an answer prompt.
Predictions from any generator can then be scored with exact-match and token F1.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
BEST_HNM_CHECKPOINTS = {
    "squad": "checkpoints/squad/hnm_ablation/alignment_mlp_kl_div_tau_0.1_hnm_18.pth",
    "metaqa": "checkpoints/metaqa/hnm_ablation/alignment_mlp_kl_div_tau_0.01_hnm_0.pth",
    "musique": "checkpoints/musique/hnm_ablation/alignment_mlp_kl_div_tau_0.05_hnm_33.pth",
    "2wiki": "checkpoints/2wiki/hnm_ablation/alignment_mlp_kl_div_tau_0.07_hnm_149.pth",
}


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, answers: Sequence[str]) -> float:
    if not answers:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    return float(any(normalized_prediction == normalize_answer(answer) for answer in answers))


def token_f1_score(prediction: str, answers: Sequence[str]) -> float:
    if not answers:
        return 0.0

    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0

    best_f1 = 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue

        common = set(prediction_tokens) & set(answer_tokens)
        num_same = sum(min(prediction_tokens.count(tok), answer_tokens.count(tok)) for tok in common)
        if num_same == 0:
            continue

        precision = num_same / len(prediction_tokens)
        recall = num_same / len(answer_tokens)
        best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))

    return best_f1


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_predictions(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score JSONL rows containing prediction plus answers/gold_answers."""
    exact_matches = []
    f1_scores = []
    oracle_hits = []
    oracle_full = []
    unscored = 0

    for row in records:
        prediction = row.get("prediction")
        answers = row.get("answers") or row.get("gold_answers") or []
        if isinstance(answers, str):
            answers = [answers]

        if answers and prediction:
            exact_matches.append(exact_match_score(prediction, answers))
            f1_scores.append(token_f1_score(prediction, answers))
        else:
            unscored += 1

        if "oracle_context_hit" in row:
            oracle_hits.append(float(bool(row["oracle_context_hit"])))
        if "oracle_full_coverage" in row:
            oracle_full.append(float(bool(row["oracle_full_coverage"])))

    scored = len(exact_matches)
    return {
        "scored_examples": scored,
        "unscored_examples": unscored,
        "exact_match": round(100.0 * sum(exact_matches) / scored, 2) if scored else None,
        "token_f1": round(100.0 * sum(f1_scores) / scored, 2) if scored else None,
        "oracle_context_hit": (
            round(100.0 * sum(oracle_hits) / len(oracle_hits), 2) if oracle_hits else None
        ),
        "oracle_full_coverage": (
            round(100.0 * sum(oracle_full) / len(oracle_full), 2) if oracle_full else None
        ),
    }


def build_generation_prompt(question: str, contexts: Sequence[Dict[str, str]]) -> str:
    context_text = "\n\n".join(
        f"[{idx + 1}] {ctx.get('title', '')}\n{ctx.get('content', '')}".strip()
        for idx, ctx in enumerate(contexts)
    )
    return (
        "Answer the question using only the context below. "
        "If the answer is not supported, say that it is not available.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    )


def _answers_from_metadata(metadata: Dict[str, Any]) -> List[str]:
    answer_keys = ("answers", "answer", "answer_text", "gold_answer", "gold_answers")
    answers: List[str] = []
    for key in answer_keys:
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            answers.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    answers.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("answer")
                    if text:
                        answers.append(str(text))
        elif isinstance(value, dict):
            text = value.get("text") or value.get("answer")
            if text:
                answers.append(str(text))
    return sorted(set(answer for answer in answers if answer))


def _get_split_queries(engine: Any) -> Dict[str, List[Tuple[Any, List[int], List[str]]]]:
    all_pairs = []
    for node in engine.all_nodes:
        if node.metadata.get("type") != "question":
            continue

        gt_doc_ids = []
        gt_partitions = []
        for neighbor_id in node.neighbors:
            partition_id = engine.partition_map.get(neighbor_id)
            if partition_id is not None:
                gt_doc_ids.append(neighbor_id)
                gt_partitions.append(int(partition_id))

        if gt_partitions:
            all_pairs.append((node.node_id, node, sorted(set(gt_partitions)), gt_doc_ids))

    all_pairs.sort(key=lambda item: item[0])
    random.Random(SPLIT_SEED).shuffle(all_pairs)

    train_end = int(len(all_pairs) * TRAIN_RATIO)
    val_end = train_end + int(len(all_pairs) * VAL_RATIO)

    def to_queries(pairs: Sequence[Tuple[str, Any, List[int], List[str]]]):
        return [(node, partitions, doc_ids) for _, node, partitions, doc_ids in pairs]

    return {
        "train": to_queries(all_pairs[:train_end]),
        "val": to_queries(all_pairs[train_end:val_end]),
        "test": to_queries(all_pairs[val_end:]),
    }


def _load_best_mlp(dataset: str, checkpoint_path: Optional[str], device: Any) -> Any:
    import torch

    from src.alignment.mlp_encoder import TextPartitionMLP

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
    return model


def _rerank_faiss_dense(engine: Any, query_vector: Any, pool: Sequence[Any], top_k: int):
    import numpy as np

    indices = []
    valid_nodes = []
    for node in pool:
        node_idx = engine.node_id_to_idx.get(node.node_id)
        if node_idx is not None:
            indices.append(int(node_idx))
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


def build_context_dataset(
    dataset: str,
    output_path: str,
    split: str = "test",
    limit: Optional[int] = None,
    top_k_partitions: int = 20,
    top_k_docs: int = 10,
    reranker: str = "faiss_dense",
    checkpoint_path: Optional[str] = None,
    max_context_chars_per_doc: int = 1800,
    use_level3: bool = False,
    score_threshold: float = 0.3,
    expand_threshold: Optional[float] = None,
    max_traverse_steps: int = 20,
    beam_width: int = 50,
    expand_top_neighbors: int = 8,
    min_context_nodes: int = 3,
    exclude_synthetic_edges: bool = True,
    dynamic_partition_expansion: bool = True,
    max_dynamic_partitions: int = 3,
    partition_admission_threshold: float = 0.35,
    partition_fetch_k: int = 5,
) -> List[Dict[str, Any]]:
    """Export retrieval contexts for downstream answer-generation runs."""
    import faiss
    import numpy as np
    import torch
    from tqdm import tqdm

    from src.core.encoders import DenseEncoder
    from src.core.engine import CoreEngine
    from src.evaluation.benchmark_level3 import _choose_seed_reranker, _rerank_seed_candidates
    from src.strategies.crag import CRAG

    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = _load_best_mlp(dataset, checkpoint_path, device)
    resolved_reranker = _choose_seed_reranker(
        dataset=dataset, split=split, requested=reranker, candidate_jsonl=None
    )

    splits = _get_split_queries(engine)
    queries = splits.get(split, [])
    if limit is not None:
        queries = queries[:limit]

    records: List[Dict[str, Any]] = []
    for question_node, gt_partitions, gt_doc_ids in tqdm(queries, desc="Building contexts"):
        retrieval_started = time.perf_counter()
        query_vector = encoder.encode([question_node.content]).astype("float32")
        faiss.normalize_L2(query_vector)

        with torch.no_grad():
            projected = mlp(torch.tensor(query_vector, dtype=torch.float32).to(device)).cpu().numpy()

        partition_results = engine.search_centroids(projected, k=top_k_partitions)
        retrieved_partitions = [partition_id for partition_id, _ in partition_results]

        pool = []
        for partition_id in retrieved_partitions:
            pool.extend(engine.get_partition_nodes(partition_id))

        scored_nodes = _rerank_seed_candidates(
            engine=engine,
            dataset=dataset,
            method=resolved_reranker,
            query=question_node.content,
            query_vector=query_vector,
            pool=pool,
            top_k=top_k_docs * top_k_partitions,
            device=device,
        )
        scored_nodes = scored_nodes[:top_k_docs]

        traversal_trace: Dict[str, Any] = {}
        if use_level3:
            retriever = CRAG(
                engine=engine,
                llm=None,
                encoder=encoder,
                mode="mlp",
                reranker="faiss",
                top_k_partitions=top_k_partitions,
                top_k_entry=top_k_docs,
                max_traverse_steps=max_traverse_steps,
                score_threshold=score_threshold,
                expand_threshold=expand_threshold,
                max_context_nodes=top_k_docs,
                beam_width=beam_width,
                expand_top_neighbors=expand_top_neighbors,
                min_context_nodes=min_context_nodes,
                dynamic_partition_expansion=dynamic_partition_expansion,
                max_dynamic_partitions=max_dynamic_partitions,
                partition_admission_threshold=partition_admission_threshold,
                partition_fetch_k=partition_fetch_k,
                exclude_synthetic_edges=exclude_synthetic_edges,
                mlp_encoder=mlp,
            )
            selected_nodes = retriever.traverse_candidates(
                question_node.content,
                query_vector,
                scored_nodes,
                selected_partitions=retrieved_partitions,
            )
            traversal_trace = retriever._last_traversal_trace
            score_lookup = {
                item["node_id"]: item["score"]
                for item in traversal_trace.get("selected", [])
            }
            context_items = [
                (node, score_lookup.get(node.node_id, 0.0)) for node in selected_nodes
            ]
        else:
            context_items = scored_nodes

        contexts = []
        for rank, (node, score) in enumerate(context_items, start=1):
            contexts.append({
                "rank": rank,
                "node_id": node.node_id,
                "title": node.metadata.get("title", ""),
                "content": node.content[:max_context_chars_per_doc],
                "score": score,
            })

        retrieved_doc_ids = [ctx["node_id"] for ctx in contexts]
        gt_set = set(gt_doc_ids)
        retrieved_set = set(retrieved_doc_ids)

        record = {
            "id": question_node.node_id,
            "dataset": dataset,
            "split": split,
            "question": question_node.content,
            "answers": _answers_from_metadata(question_node.metadata),
            "gt_doc_ids": gt_doc_ids,
            "supporting_document_ids": gt_doc_ids,
            "gt_partitions": gt_partitions,
            "retrieved_partitions": retrieved_partitions,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_document_ids": retrieved_doc_ids,
            "contexts": contexts,
            "oracle_context_hit": bool(gt_set & retrieved_set),
            "oracle_full_coverage": bool(gt_set and gt_set.issubset(retrieved_set)),
            "pool_contains_hit": bool(gt_set & {node.node_id for node in pool}),
            "prompt": build_generation_prompt(question_node.content, contexts),
            "latency_ms": {
                "retrieval": round(
                    (time.perf_counter() - retrieval_started) * 1000.0,
                    3,
                )
            },
            "metadata": {
                "selector": "mlp_best_hnm",
                "reranker": resolved_reranker,
                "requested_reranker": reranker,
                "context_builder": "level3_traversal" if use_level3 else "level2_only",
                "top_k_partitions": top_k_partitions,
                "top_k_docs": top_k_docs,
                "score_threshold": score_threshold,
                "expand_threshold": expand_threshold if expand_threshold is not None else score_threshold,
                "exclude_synthetic_edges": exclude_synthetic_edges,
                "dynamic_partition_expansion": dynamic_partition_expansion,
                "max_dynamic_partitions": max_dynamic_partitions,
                "partition_admission_threshold": partition_admission_threshold,
                "partition_fetch_k": partition_fetch_k,
                "traversal": traversal_trace,
            },
        }
        records.append(record)

    write_jsonl(output_path, records)
    return records


def _build_context_command(args: argparse.Namespace) -> None:
    records = build_context_dataset(
        dataset=args.dataset,
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        top_k_partitions=args.top_k_partitions,
        top_k_docs=args.top_k_docs,
        reranker=args.reranker,
        checkpoint_path=args.checkpoint,
        max_context_chars_per_doc=args.max_context_chars_per_doc,
        use_level3=args.use_level3,
        score_threshold=args.score_threshold,
        expand_threshold=args.expand_threshold,
        max_traverse_steps=args.max_traverse_steps,
        beam_width=args.beam_width,
        expand_top_neighbors=args.expand_top_neighbors,
        min_context_nodes=args.min_context_nodes,
        exclude_synthetic_edges=not args.include_synthetic_edges,
        dynamic_partition_expansion=not args.disable_dynamic_partitions,
        max_dynamic_partitions=args.max_dynamic_partitions,
        partition_admission_threshold=args.partition_admission_threshold,
        partition_fetch_k=args.partition_fetch_k,
    )
    summary = evaluate_predictions(records)
    print(json.dumps({"output": args.output, "records": len(records), **summary}, indent=2))


def _score_command(args: argparse.Namespace) -> None:
    records = load_jsonl(args.predictions)
    summary = evaluate_predictions(records)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Paper 2 generation benchmark utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-context", help="Export retrieval contexts")
    build_parser.add_argument("--dataset", required=True, choices=sorted(BEST_HNM_CHECKPOINTS))
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    build_parser.add_argument("--limit", type=int, default=None)
    build_parser.add_argument("--top-k-partitions", type=int, default=20)
    build_parser.add_argument("--top-k-docs", type=int, default=10)
    build_parser.add_argument(
        "--reranker",
        default="auto",
        choices=("auto", "faiss_dense", "bm25", "splade", "no_rerank"),
    )
    build_parser.add_argument("--checkpoint", default=None)
    build_parser.add_argument("--max-context-chars-per-doc", type=int, default=1800)
    build_parser.add_argument("--use-level3", action="store_true")
    build_parser.add_argument("--score-threshold", type=float, default=0.3)
    build_parser.add_argument("--expand-threshold", type=float, default=None)
    build_parser.add_argument("--max-traverse-steps", type=int, default=20)
    build_parser.add_argument("--beam-width", type=int, default=50)
    build_parser.add_argument("--expand-top-neighbors", type=int, default=8)
    build_parser.add_argument("--min-context-nodes", type=int, default=3)
    build_parser.add_argument("--include-synthetic-edges", action="store_true")
    build_parser.add_argument("--disable-dynamic-partitions", action="store_true")
    build_parser.add_argument("--max-dynamic-partitions", type=int, default=3)
    build_parser.add_argument("--partition-admission-threshold", type=float, default=0.35)
    build_parser.add_argument("--partition-fetch-k", type=int, default=5)
    build_parser.set_defaults(func=_build_context_command)

    score_parser = subparsers.add_parser("score", help="Score generated predictions")
    score_parser.add_argument("--predictions", required=True)
    score_parser.add_argument("--output", default=None)
    score_parser.set_defaults(func=_score_command)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
