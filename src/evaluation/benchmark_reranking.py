"""
Level 2 — Intra-Partition Reranking Benchmark
===============================================
Evaluates how well different rerankers (Cross-Encoder vs Dense FAISS)
can locate the exact ground-truth document chunks *within* the Top-K
partitions selected by the Phase 1 MLP.

This is the definitive empirical proof that Level 2 adds value on top
of Level 1's partition-level retrieval.
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP

# ═══════════════════════════════════════════════════════════════════
# Deterministic Split (identical to benchmark_partition_selection.py)
# ═══════════════════════════════════════════════════════════════════

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


def _get_split_queries(engine: CoreEngine, dataset: str):
    """Replicate the exact same split logic used in Phase 1."""
    all_pairs = []
    for node in engine.all_nodes:
        if node.metadata.get("type") == "question":
            gt_doc_ids = []
            gt_pids = []
            for neighbor_id in node.neighbors:
                pid = engine.partition_map.get(neighbor_id)
                if pid is not None:
                    gt_pids.append(int(pid))
                    gt_doc_ids.append(neighbor_id)
            if gt_pids:
                all_pairs.append((node.node_id, node, list(set(gt_pids)), gt_doc_ids))

    if not all_pairs:
        return {"train": [], "val": [], "test": []}

    all_pairs.sort(key=lambda p: p[0])
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(all_pairs)

    n = len(all_pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    def to_queries(pairs):
        return [(node, pids, doc_ids) for _, node, pids, doc_ids in pairs]

    return {
        "train": to_queries(all_pairs[:train_end]),
        "val": to_queries(all_pairs[train_end:val_end]),
        "test": to_queries(all_pairs[val_end:]),
    }


# ═══════════════════════════════════════════════════════════════════
# Chunk-Level Metrics
# ═══════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5, 10]
BEST_HNM_CHECKPOINTS = {
    "squad": "checkpoints/squad/hnm_ablation/alignment_mlp_kl_div_tau_0.1_hnm_18.pth",
    "metaqa": "checkpoints/metaqa/hnm_ablation/alignment_mlp_kl_div_tau_0.01_hnm_0.pth",
    "musique": "checkpoints/musique/hnm_ablation/alignment_mlp_kl_div_tau_0.05_hnm_33.pth",
    "2wiki": "checkpoints/2wiki/hnm_ablation/alignment_mlp_kl_div_tau_0.07_hnm_149.pth",
}


def compute_chunk_metrics(retrieved_ids: List[str], gt_doc_ids: List[str]) -> Dict:
    """Compute chunk-level retrieval metrics."""
    metrics = {}
    gt_set = set(gt_doc_ids)
    num_gt = len(gt_set)

    for k in K_VALUES:
        top_k = retrieved_ids[:k]
        top_k_set = set(top_k)
        hits = len(gt_set & top_k_set)

        metrics[f"recall@{k}"] = hits / num_gt if num_gt > 0 else 0.0
        metrics[f"precision@{k}"] = hits / k
        p = hits / k
        r = hits / num_gt if num_gt > 0 else 0.0
        metrics[f"f1@{k}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        dcg = 0.0
        for i, nid in enumerate(top_k):
            if nid in gt_set:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_gt, k)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    # MRR
    metrics["mrr"] = 0.0
    for i, nid in enumerate(retrieved_ids):
        if nid in gt_set:
            metrics["mrr"] = 1.0 / (i + 1)
            break

    return metrics


# ═══════════════════════════════════════════════════════════════════
# Reranking Benchmark
# ═══════════════════════════════════════════════════════════════════

def benchmark_reranker(
    engine: CoreEngine,
    encoder: DenseEncoder,
    mlp_model,
    queries: List[Tuple],
    reranker_method: str,
    top_k_partitions: int = 20,
    top_k_rerank: int = 10,
    device: torch.device = None,
) -> Dict:
    """
    For each query:
    1. Run Level 1 MLP to get the Top-K partitions
    2. Pool all document nodes from those partitions
    3. Rerank the pool using the specified method
    4. Measure chunk-level recall against ground-truth doc IDs
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics = defaultdict(list)
    latencies = []
    partition_sizes = []

    for q_node, gt_pids, gt_doc_ids in tqdm(
        queries, desc=f"L2 Benchmark ({reranker_method})", leave=False
    ):
        t0 = time.time()

        # ── Level 1: MLP Partition Selection ──────────────────────
        import faiss as faiss_lib
        query_vector = encoder.encode([q_node.content]).astype("float32")
        faiss_lib.normalize_L2(query_vector)

        with torch.no_grad():
            qv = torch.tensor(query_vector, dtype=torch.float32).to(device)
            projected = mlp_model(qv).cpu().numpy()

        results = engine.search_centroids(projected, k=top_k_partitions)
        partition_ids = [pid for pid, _ in results]

        # ── Level 2: Pool nodes from partitions ───────────────────
        pool = []
        for pid in partition_ids:
            pool.extend(engine.get_partition_nodes(pid))
        partition_sizes.append(len(pool))

        if not pool:
            latencies.append(time.time() - t0)
            for k in K_VALUES:
                all_metrics[f"recall@{k}"].append(0.0)
                all_metrics[f"precision@{k}"].append(0.0)
                all_metrics[f"f1@{k}"].append(0.0)
                all_metrics[f"ndcg@{k}"].append(0.0)
            all_metrics["mrr"].append(0.0)
            continue

        # ── Level 2: Rerank ───────────────────────────────────────
        if reranker_method == "cross_encoder":
            scored = engine.rerank_cross_encoder(
                q_node.content, pool, top_k=top_k_rerank
            )
            retrieved_ids = [n.node_id for n, _ in scored]

        elif reranker_method == "faiss_dense":
            # Cosine similarity rerank using FAISS vectors
            indices = []
            valid_nodes = []
            for node in pool:
                idx = engine.node_id_to_idx.get(node.node_id)
                if idx is not None:
                    indices.append(int(idx))
                    valid_nodes.append(node)

            if indices:
                node_vecs = np.stack(
                    [engine.node_index.reconstruct(i) for i in indices]
                )
                qv_flat = query_vector.flatten()
                qv_norm = np.linalg.norm(qv_flat) + 1e-8
                norms = np.linalg.norm(node_vecs, axis=1) + 1e-8
                scores = np.dot(node_vecs, qv_flat) / (norms * qv_norm)

                top_count = min(top_k_rerank, len(scores))
                top_idx = np.argsort(-scores)[:top_count]
                retrieved_ids = [valid_nodes[i].node_id for i in top_idx]
            else:
                retrieved_ids = []

        elif reranker_method == "no_rerank":
            # Baseline: just take the first N nodes from the pool (random order)
            retrieved_ids = [n.node_id for n in pool[:top_k_rerank]]

        else:
            raise ValueError(f"Unknown reranker: {reranker_method}")

        latency = time.time() - t0
        latencies.append(latency)

        metrics = compute_chunk_metrics(retrieved_ids, gt_doc_ids)
        for key, val in metrics.items():
            all_metrics[key].append(val)

    # Aggregate
    summary = {}
    for key, vals in all_metrics.items():
        summary[key] = round(float(np.mean(vals)) * 100, 2)

    summary["avg_latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
    summary["p50_latency_ms"] = round(float(np.percentile(latencies, 50)) * 1000, 2)
    summary["p95_latency_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 2)
    summary["avg_pool_size"] = round(float(np.mean(partition_sizes)), 1)
    summary["total_queries"] = len(queries)
    summary["method"] = reranker_method
    return summary


# ═══════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════

def run_level2_benchmark(
    dataset: str = "squad",
    checkpoints_dir: str = "checkpoints",
    top_k_partitions: int = 20,
    top_k_rerank: int = 10,
):
    log.info(f"═══ Level 2 Reranking Benchmark: {dataset.upper()} ═══")

    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the same HNM-optimized checkpoint family used by Paper 2.
    ckpt_path = BEST_HNM_CHECKPOINTS.get(dataset)
    if ckpt_path and checkpoints_dir != "checkpoints":
        ckpt_path = os.path.join(
            checkpoints_dir, dataset, "hnm_ablation", os.path.basename(ckpt_path)
        )
    if not ckpt_path:
        ckpt_path = os.path.join(checkpoints_dir, dataset, "alignment_mlp.pth")
    if not os.path.exists(ckpt_path):
        log.error(f"MLP checkpoint not found: {ckpt_path}")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    input_dim = state_dict["net.0.weight"].shape[1]
    hidden_dim = ckpt.get("hidden_dim", 256)
    output_dim = state_dict["net.3.weight"].shape[0] if "net.3.weight" in state_dict else input_dim

    mlp = TextPartitionMLP(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    ).to(device)
    mlp.load_state_dict(state_dict)
    mlp.eval()
    log.info(f"Loaded MLP: {input_dim}→{hidden_dim}→{output_dim}")

    splits = _get_split_queries(engine, dataset)
    test_queries = splits.get("test", [])
    if not test_queries:
        log.error("No test queries found!")
        return {}
    log.info(f"Test queries: {len(test_queries)}")

    methods = ["no_rerank", "faiss_dense", "cross_encoder"]
    all_results = {}

    for method in methods:
        log.info(f"Running: {method}")
        res = benchmark_reranker(
            engine, encoder, mlp, test_queries,
            reranker_method=method,
            top_k_partitions=top_k_partitions,
            top_k_rerank=top_k_rerank,
            device=device,
        )
        all_results[method] = res
        log.info(
            f"  R@1={res['recall@1']:.1f}%  R@5={res['recall@5']:.1f}%  "
            f"R@10={res['recall@10']:.1f}%  MRR={res['mrr']:.1f}%  "
            f"Lat={res['avg_latency_ms']:.1f}ms  Pool={res['avg_pool_size']:.0f}"
        )

    # ── Print Summary Table ───────────────────────────────────────
    W = 100
    print(f"\n{'═' * W}")
    print(f"  LEVEL 2 RERANKING BENCHMARK — {dataset.upper()}")
    print(f"  Top-{top_k_partitions} partitions → Rerank to Top-{top_k_rerank} chunks")
    print(f"{'═' * W}")
    print(
        f"  {'Method':<17} │ {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} │ "
        f"{'MRR':>6} {'NDCG@5':>7} │ {'Lat':>7} {'Pool':>5}"
    )
    print(f"  {'─' * (W - 2)}")

    for method, m in all_results.items():
        print(
            f"  {method:<17} │ "
            f"{m['recall@1']:>5.1f}% {m['recall@3']:>5.1f}% {m['recall@5']:>5.1f}% {m['recall@10']:>5.1f}% │ "
            f"{m['mrr']:>5.1f}% {m['ndcg@5']:>6.1f}% │ "
            f"{m['avg_latency_ms']:>6.1f}ms {m['avg_pool_size']:>5.0f}"
        )
    print(f"{'═' * W}")

    # ── Export JSON ───────────────────────────────────────────────
    out_dir = os.path.join("results", "level_2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset}_level_2_reranking.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Level 2 Reranking Benchmark")
    parser.add_argument("--dataset", default="squad", help="Dataset to benchmark")
    parser.add_argument("--top-k-partitions", type=int, default=20)
    parser.add_argument("--top-k-rerank", type=int, default=10)
    args = parser.parse_args()

    run_level2_benchmark(
        dataset=args.dataset,
        top_k_partitions=args.top_k_partitions,
        top_k_rerank=args.top_k_rerank,
    )
