import os
import sys
import csv
import time
import random
import logging
import numpy as np
import torch
from torch_geometric.data import Batch
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.gnn_encoders import GINAlignmentEncoder, GCNAlignmentEncoder, SAGEAlignmentEncoder


# ═══════════════════════════════════════════════════════════════════
# Deterministic Split (must match train_alignment.py exactly)
# ═══════════════════════════════════════════════════════════════════

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


def _get_split_queries(engine: CoreEngine, dataset: str) -> Dict[str, List[Tuple[object, List[int]]]]:
    """
    Collect all (query_node, gt_pids) pairs for a dataset, split
    deterministically into train / val / test (70/20/10).
    Uses the exact same logic as train_alignment.get_split_pairs().
    """
    all_pairs = []
    for node in engine.all_nodes:
        if node.metadata.get("type") == "question":
            gt_pids = []
            for neighbor_id in node.neighbors:
                pid = engine.partition_map.get(neighbor_id)
                if pid is not None:
                    gt_pids.append(int(pid))
            if gt_pids:
                all_pairs.append((node.node_id, node, list(set(gt_pids))))

    if not all_pairs:
        log.warning(f"No {dataset} questions found for benchmarking.")
        return {"train": [], "val": [], "test": []}

    all_pairs.sort(key=lambda p: p[0])  # sort by node_id
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(all_pairs)

    n = len(all_pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    def to_queries(pairs):
        return [(n, pids) for _, n, pids in pairs]

    splits = {
        "train": to_queries(all_pairs[:train_end]),
        "val": to_queries(all_pairs[train_end:val_end]),
        "test": to_queries(all_pairs[val_end:]),
    }
    log.info(
        f"Benchmark split ({dataset}): "
        f"{len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test"
    )
    return splits


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

K_VALUES = [1, 3, 5, 10, 20]


def compute_multi_gt_metrics(retrieved_pids: List[int], gt_pids: List[int]) -> Dict:
    """Compute comprehensive retrieval metrics for multiple ground-truth partitions."""
    metrics = {}
    gt_set = set(gt_pids)
    num_gt = len(gt_set)

    for k in K_VALUES:
        top_k = retrieved_pids[:k]
        top_k_set = set(top_k)
        hits = len(gt_set & top_k_set)

        metrics[f"recall@{k}"] = 1.0 if hits > 0 else 0.0
        metrics[f"gt_recall@{k}"] = hits / num_gt
        metrics[f"precision@{k}"] = hits / k

        p = hits / k
        r = hits / num_gt
        metrics[f"f1@{k}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        dcg = 0.0
        for i, pid in enumerate(top_k):
            if pid in gt_set:
                dcg += 1.0 / np.log2(i + 2)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_gt, k)))
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

    metrics["mrr"] = 0.0
    for i, pid in enumerate(retrieved_pids):
        if pid in gt_set:
            metrics["mrr"] = 1.0 / (i + 1)
            break

    metrics["first_hit_pos"] = 0
    for i, pid in enumerate(retrieved_pids):
        if pid in gt_set:
            metrics["first_hit_pos"] = i + 1
            break

    metrics["full_coverage@20"] = 1.0 if gt_set.issubset(set(retrieved_pids[:20])) else 0.0
    metrics["num_gt"] = float(num_gt)

    return metrics


def benchmark(
    engine: CoreEngine,
    encoder: DenseEncoder,
    method: str,
    queries: List[Tuple[object, List[int]]],
    k: int = 20,
    model=None,
    partition_embs=None,
    precomputed_embs: np.ndarray = None,
    topo_partition_prototypes: np.ndarray = None,
) -> Dict:
    """Run a single method on all queries and return aggregated metrics."""
    all_metrics: Dict[str, List[float]] = defaultdict(list)
    latencies: List[float] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, (q_node, gt_pids) in tqdm(
        enumerate(queries),
        desc=f"Benchmarking ({method})",
        total=len(queries),
        leave=False
    ):
        t0 = time.time()

        if precomputed_embs is not None:
            query_vector = precomputed_embs[i:i + 1].astype("float32")
        else:
            query_vector = (
                encoder.encode([q_node.content]).astype("float32")
                if method != "colbert_centroid"
                else None
            )

        if query_vector is not None:
            import faiss
            faiss.normalize_L2(query_vector)

        if method == "faiss_centroid":
            results = engine.search_centroids(query_vector, k=k)
            retrieved = [pid for pid, _ in results]

        elif method.startswith("faiss_vote_"):
            vote_k = int(method.split("_")[-1])

            t_search_start = time.time()
            dense_nodes = engine.search_dense(query_vector, k=vote_k)
            t_search = time.time() - t_search_start

            t_tally_start = time.time()
            vote_counts = {}
            for node in dense_nodes:
                pid = engine.partition_map.get(node.node_id)
                if pid is not None:
                    vote_counts[int(pid)] = vote_counts.get(int(pid), 0) + 1

            sorted_pids = sorted(vote_counts.keys(), key=lambda p: vote_counts[p], reverse=True)
            retrieved = sorted_pids[:k]
            t_tally = time.time() - t_tally_start

            if i % 500 == 0:
                log.info(
                    f"      [VOTE DEBUG] search: {t_search*1000:.2f}ms | "
                    f"tally: {t_tally*1000:.2f}ms | nodes: {len(dense_nodes)}"
                )

        elif method == "colbert_centroid":
            results = engine.search_colbert_centroid(q_node.content, k=k)
            retrieved = [pid for pid, _ in results]

        elif method == "mlp" and model is not None:
            with torch.no_grad():
                qv = torch.tensor(query_vector, dtype=torch.float32).to(device)
                projected = model(qv).cpu().numpy()
            results = engine.search_centroids(projected, k=k)
            retrieved = [pid for pid, _ in results]

        elif method == "mlp_topo" and model is not None:
            with torch.no_grad():
                qv = torch.tensor(query_vector, dtype=torch.float32).to(device)
                projected = model(qv).cpu().numpy()

            if topo_partition_prototypes is None:
                raise ValueError(
                    "mlp_topo benchmark requires topology-aware partition prototypes from checkpoint."
                )

            proj = projected[0]
            proj_norm = np.linalg.norm(proj)
            if proj_norm > 0:
                proj = proj / proj_norm

            sims = topo_partition_prototypes @ proj
            retrieved = np.argsort(-sims)[:k].tolist()

        elif method in ["gin", "gcn", "sage"] and model is not None:
            with torch.no_grad():
                qv = torch.tensor(query_vector, dtype=torch.float32).to(device)
                q_proj = model.project_text(qv, device).cpu().numpy()

            if partition_embs is not None:
                dist = np.sum((partition_embs - q_proj) ** 2, axis=1)
                retrieved = np.argsort(dist)[:k].tolist()
            else:
                results = engine.search_centroids(q_proj, k=k)
                retrieved = [pid for pid, _ in results]

        else:
            results = engine.search_centroids(query_vector, k=k)
            retrieved = [pid for pid, _ in results]

        latency = time.time() - t0
        latencies.append(latency)

        metrics = compute_multi_gt_metrics(retrieved, gt_pids)
        for key, val in metrics.items():
            all_metrics[key].append(val)

    summary = {}
    for key, vals in all_metrics.items():
        if key == "num_gt":
            summary["avg_gt_partitions"] = round(float(np.mean(vals)), 2)
            summary["min_gt_partitions"] = int(np.min(vals))
            summary["max_gt_partitions"] = int(np.max(vals))
            summary["median_gt_partitions"] = round(float(np.median(vals)), 1)
            summary["std_gt_partitions"] = round(float(np.std(vals)), 2)
        elif key == "first_hit_pos":
            summary["avg_first_hit_pos"] = round(float(np.mean(vals)), 2)
            summary["median_first_hit_pos"] = round(float(np.median(vals)), 1)
        else:
            summary[key] = round(float(np.mean(vals)) * 100, 2)

    summary["avg_latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
    summary["p50_latency_ms"] = round(float(np.percentile(latencies, 50)) * 1000, 2)
    summary["p95_latency_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 2)
    summary["p99_latency_ms"] = round(float(np.percentile(latencies, 99)) * 1000, 2)
    summary["total_queries"] = len(queries)
    summary["method"] = method
    return summary


# ═══════════════════════════════════════════════════════════════════
# CSV Export
# ═══════════════════════════════════════════════════════════════════

CSV_COLUMNS = [
    "dataset", "method", "split", "total_queries",
    "recall@1", "recall@3", "recall@5", "recall@10", "recall@20",
    "gt_recall@1", "gt_recall@3", "gt_recall@5", "gt_recall@10", "gt_recall@20",
    "precision@1", "precision@3", "precision@5", "precision@10", "precision@20",
    "f1@1", "f1@3", "f1@5", "f1@10", "f1@20",
    "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@20",
    "mrr", "full_coverage@20",
    "avg_gt_partitions", "min_gt_partitions", "max_gt_partitions",
    "median_gt_partitions", "std_gt_partitions",
    "avg_first_hit_pos", "median_first_hit_pos",
    "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
]


def _export_csv(dataset: str, all_results: Dict[str, Dict[str, Dict]]):
    """Export all results to a CSV file with one row per method×split."""
    csv_path = os.path.join("results", "level_1", f"{dataset}_level_1_benchmark_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for method, splits in all_results.items():
            for split_name, metrics in splits.items():
                row = {"dataset": dataset, "method": method, "split": split_name}
                row.update(metrics)
                writer.writerow(row)

    log.info(f"CSV results exported to {csv_path}")
    return csv_path


# ═══════════════════════════════════════════════════════════════════
# Main Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

def run_benchmark(dataset: str = "2wiki", checkpoints_dir: str = "checkpoints"):
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()

    split_queries = _get_split_queries(engine, dataset=dataset)
    if not any(split_queries.values()):
        return {}

    methods = [
        "faiss_centroid",
        "colbert_centroid",
        "faiss_vote_50",
        "faiss_vote_100",
        "faiss_vote_200",
        "mlp",
        "mlp_topo",
        "gin",
        "gcn",
        "sage",
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_dir = os.path.join("data/ukb_storage", dataset)
    graph_path = os.path.join(src_dir, "graph.pt")
    full_graph = torch.load(graph_path, weights_only=False) if os.path.exists(graph_path) else None
    node_features = None
    if full_graph is not None:
        import faiss
        node_index_path = os.path.join(src_dir, "nodes.index")
        if os.path.exists(node_index_path):
            index = faiss.read_index(node_index_path)
            log.warning(
                f"Reconstructing {index.ntotal} vectors from FAISS to build full node features. "
                f"This is memory intensive!"
            )
            features_np = index.reconstruct_n(0, index.ntotal)
            node_features = torch.tensor(features_np, dtype=torch.float32)
        else:
            node_features = torch.tensor(
                encoder.encode([n.content for n in engine.nodes]),
                dtype=torch.float32
            )

    precomputed_split_embs = {}
    for split_name in ["train", "val", "test"]:
        if split_name in split_queries and split_queries[split_name]:
            texts = []
            for q_node, _ in split_queries[split_name]:
                if not hasattr(q_node, "content"):
                    raise TypeError(
                        f"[{split_name}] Expected query node with '.content', got {type(q_node)}"
                    )
                if not isinstance(q_node.content, str):
                    raise TypeError(
                        f"[{split_name}] Expected q_node.content to be str, got {type(q_node.content)} "
                        f"for node_id={getattr(q_node, 'node_id', 'unknown')}"
                    )
                texts.append(q_node.content)

            log.info(f"Batch encoding {len(texts)} '{split_name}' queries (runs exactly once)...")
            precomputed_split_embs[split_name] = encoder.encode(texts)

    shared_part_batch = None
    all_results: Dict[str, Dict[str, Dict]] = {}

    for method in methods:
        if method == "colbert_centroid" and getattr(engine, "colbert_centroid", None) is None:
            log.warning("Skipping colbert_centroid because ColBERT centroid search is unavailable.")
            continue

        log.info(f"Benchmarking: {method}")
        model = None
        partition_embs = None
        topo_partition_prototypes = None

        ckpt_path = os.path.join(checkpoints_dir, dataset, f"alignment_{method}.pth")
        if (
            method not in ["faiss_centroid", "colbert_centroid"]
            and not method.startswith("faiss_vote_")
            and os.path.exists(ckpt_path)
        ):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

            if "output_dim" in ckpt:
                embed_dim = ckpt["output_dim"]
            elif "input_dim" in ckpt:
                embed_dim = ckpt["input_dim"]
            else:
                log.warning(
                    f"Checkpoint for '{method}' has no hidden dim keys! "
                    f"Falling back to 384, this might cause shape mismatches."
                )
                embed_dim = 384

            hidden_dim = ckpt.get("hidden_dim", 256)

            if method in ["mlp", "mlp_topo"]:
                state_dict = ckpt["model_state_dict"]
                actual_in_channels = state_dict["net.0.weight"].shape[1]
                actual_out_channels = (
                    state_dict["net.3.weight"].shape[0]
                    if "net.3.weight" in state_dict else embed_dim
                )
                model = TextPartitionMLP(
                    input_dim=actual_in_channels,
                    hidden_dim=hidden_dim,
                    output_dim=actual_out_channels
                ).to(device)

                if method == "mlp_topo":
                    topo_partition_prototypes = ckpt.get("topo_partition_prototypes", None)
                    if topo_partition_prototypes is None:
                        raise ValueError(
                            f"Checkpoint for '{method}' does not contain 'topo_partition_prototypes'. "
                            "Retrain mlp_topo with the updated training script."
                        )
                    topo_partition_prototypes = np.asarray(topo_partition_prototypes, dtype=np.float32)

            elif method in ["gin", "gcn", "sage"]:
                actual_in_channels = ckpt["model_state_dict"]["input_proj.weight"].shape[1]

                if method == "gin":
                    model = GINAlignmentEncoder(
                        in_channels=actual_in_channels,
                        hidden_channels=hidden_dim,
                        out_channels=embed_dim
                    ).to(device)
                elif method == "gcn":
                    model = GCNAlignmentEncoder(
                        in_channels=actual_in_channels,
                        hidden_channels=hidden_dim,
                        out_channels=embed_dim
                    ).to(device)
                elif method == "sage":
                    model = SAGEAlignmentEncoder(
                        in_channels=actual_in_channels,
                        hidden_channels=hidden_dim,
                        out_channels=embed_dim
                    ).to(device)

            if model:
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                if method in ["gin", "gcn", "sage"] and full_graph is not None and node_features is not None:
                    if shared_part_batch is None:
                        from src.alignment.train_alignment import get_partition_subgraphs, augment_node_features

                        num_partitions = max(engine.partition_map.values()) + 1
                        node_id_to_idx = {n.node_id: i for i, n in enumerate(engine.nodes)}
                        partition_ids = [0] * len(engine.nodes)
                        for node_id, pid in engine.partition_map.items():
                            if node_id in node_id_to_idx:
                                partition_ids[node_id_to_idx[node_id]] = int(pid)

                        augmented_features = augment_node_features(
                            node_features,
                            graph_path,
                            partition_ids,
                            num_partitions,
                            len(engine.nodes)
                        )

                        subs = get_partition_subgraphs(
                            full_graph.edge_index,
                            augmented_features,
                            partition_ids,
                            num_partitions
                        )
                        shared_part_batch = Batch.from_data_list(subs).to(device)

                    with torch.no_grad():
                        partition_embs = model(
                            shared_part_batch.x,
                            shared_part_batch.edge_index,
                            shared_part_batch.batch
                        ).cpu().numpy()

        method_results = {}
        for split_name in ["train", "val", "test"]:
            queries = split_queries[split_name]
            if not queries:
                continue

            embs = precomputed_split_embs.get(split_name, None)
            res = benchmark(
                engine,
                encoder,
                method,
                queries,
                model=model,
                partition_embs=partition_embs,
                precomputed_embs=embs,
                topo_partition_prototypes=topo_partition_prototypes,
            )
            method_results[split_name] = res

            log.info(
                f"  [{split_name}] R@1={res.get('recall@1', 0):.1f}% "
                f"GTR@20={res.get('gt_recall@20', 0):.1f}% "
                f"FCov@20={res.get('full_coverage@20', 0):.1f}% "
                f"MRR={res.get('mrr', 0):.1f}%"
            )

            if dataset == "metaqa":
                hops = defaultdict(list)
                hop_embs = defaultdict(list) if embs is not None else None
                for i, (q_node, gt) in enumerate(queries):
                    hop = q_node.metadata.get("hop", "unknown")
                    hops[hop].append((q_node, gt))
                    if embs is not None:
                        hop_embs[hop].append(embs[i])

                for hop in sorted(hops.keys()):
                    h_queries = hops[hop]
                    h_embs = np.array(hop_embs[hop]) if hop_embs else None
                    h_res = benchmark(
                        engine,
                        encoder,
                        method,
                        h_queries,
                        model=model,
                        partition_embs=partition_embs,
                        precomputed_embs=h_embs,
                        topo_partition_prototypes=topo_partition_prototypes,
                    )
                    method_results[f"{split_name}_hop{hop}"] = h_res
                    log.info(
                        f"    ↳ Hop {hop}: R@1={h_res.get('recall@1', 0):.1f}% "
                        f"GTR@20={h_res.get('gt_recall@20', 0):.1f}% "
                        f"FCov@20={h_res.get('full_coverage@20', 0):.1f}%"
                    )

        all_results[method] = method_results

    csv_path = _export_csv(dataset, all_results)

    _print_dataset_summary(engine, dataset, split_queries)
    _print_recall_table(all_results)
    _print_detailed_metrics(all_results)
    _print_per_method_summary(all_results)
    _print_overall_summary(all_results)

    print(f"\n  📄 Full CSV: {csv_path}\n")
    return all_results


# ═══════════════════════════════════════════════════════════════════
# Dataset Summary
# ═══════════════════════════════════════════════════════════════════

def _print_dataset_summary(engine, dataset, split_queries):
    """Print a comprehensive dataset summary."""
    num_doc = len(engine.nodes)
    num_all = len(engine.all_nodes)
    num_q = num_all - num_doc
    num_parts = max(engine.partition_map.values()) + 1 if engine.partition_map else 0
    graph_edges = engine.graph.edge_index.shape[1] // 2 if hasattr(engine.graph, "edge_index") else 0

    all_gt = []
    for split in ["train", "val", "test"]:
        for _, pids in split_queries.get(split, []):
            all_gt.append(len(pids))
    gt = np.array(all_gt) if all_gt else np.array([0])

    psizes = defaultdict(int)
    for pid in engine.partition_map.values():
        psizes[int(pid)] += 1
    sz = np.array(list(psizes.values())) if psizes else np.array([0])

    W = 90
    print(f"\n{'═' * W}")
    print(f"  DATASET SUMMARY: {dataset.upper()}")
    print(f"{'═' * W}")
    print(f"  Doc Nodes:       {num_doc:>8,}     │  Graph Edges:     {graph_edges:>8,}  (undirected)")
    print(f"  Question Nodes:  {num_q:>8,}     │  Partitions:      {num_parts:>8}")
    print(f"  Total Nodes:     {num_all:>8,}     │")
    if num_parts > 0 and num_parts < 20:
        print(f"  {'─' * (W-2)}")
        print(f"  ⚠️ MAX RETRIEVAL CAP: Mathematically locked to K={num_parts} (Index Size Constraint)")
    print(f"  {'─' * (W-2)}")
    print(
        f"  Partition Size:  avg={sz.mean():.1f}  min={sz.min()}  max={sz.max()}  "
        f"median={np.median(sz):.0f}  std={sz.std():.1f}"
    )
    print(f"  {'─' * (W-2)}")
    n_train = len(split_queries.get("train", []))
    n_val = len(split_queries.get("val", []))
    n_test = len(split_queries.get("test", []))
    print(f"  Queries:         train={n_train:,}  val={n_val:,}  test={n_test:,}  total={n_train+n_val+n_test:,}")
    print(f"  {'─' * (W-2)}")
    print(
        f"  GT Parts/Query:  avg={gt.mean():.2f}  min={gt.min()}  max={gt.max()}  "
        f"median={np.median(gt):.0f}  std={gt.std():.2f}"
    )

    gt_dist = Counter(all_gt)
    parts = []
    for c in sorted(gt_dist.keys()):
        pct = gt_dist[c] / len(all_gt) * 100
        parts.append(f"{c}-GT:{gt_dist[c]}({pct:.1f}%)")
    print(f"  GT Distribution: {' '.join(parts)}")
    print(f"{'═' * W}")


# ═══════════════════════════════════════════════════════════════════
# Recall Table (compact, all splits)
# ═══════════════════════════════════════════════════════════════════

def _print_recall_table(all_results):
    """Print the main recall / GT recall / MRR table."""
    splits = ["train", "val", "test"]
    W = 120
    print(f"\n{'═' * W}")
    print(f"  PARTITION SELECTION RECALL")
    print(f"{'═' * W}")
    print(
        f"  {'Method':<17} {'Split':<6} │ {'R@1':>5} {'R@3':>5} {'R@5':>5} {'R@10':>5} {'R@20':>5} │ "
        f"{'GTR@5':>5} {'GTR@10':>6} {'GTR@20':>6} │ {'MRR':>5} {'FCov':>5} │ {'Lat':>6} {'#Q':>5}"
    )
    print(f"  {'─' * (W-2)}")

    for method in all_results:
        mr = all_results[method]
        for split in splits:
            m = mr.get(split, {})
            if not m:
                continue
            print(
                f"  {method:<17} {split:<6} │ "
                f"{m.get('recall@1',0):>4.1f}% {m.get('recall@3',0):>4.1f}% {m.get('recall@5',0):>4.1f}% "
                f"{m.get('recall@10',0):>4.1f}% {m.get('recall@20',0):>4.1f}% │ "
                f"{m.get('gt_recall@5',0):>4.1f}% {m.get('gt_recall@10',0):>5.1f}% {m.get('gt_recall@20',0):>5.1f}% │ "
                f"{m.get('mrr',0):>4.1f}% {m.get('full_coverage@20',0):>4.1f}% │ "
                f"{m.get('avg_latency_ms',0):>5.1f}ms {m.get('total_queries',0):>5}"
            )
        print(f"  {'─' * (W-2)}")
    print(f"{'═' * W}")


# ═══════════════════════════════════════════════════════════════════
# Detailed Metrics (Precision, F1, NDCG — test only)
# ═══════════════════════════════════════════════════════════════════

def _print_detailed_metrics(all_results):
    """Print precision, F1, NDCG, and GT stats for test split."""
    W = 115
    print(f"\n{'═' * W}")
    print(f"  DETAILED METRICS (Test Split)")
    print(f"{'═' * W}")
    print(
        f"  {'Method':<17} │ {'P@1':>5} {'P@3':>5} {'P@5':>5} {'P@10':>5} {'P@20':>5} │ "
        f"{'F1@5':>5} {'F1@10':>5} {'F1@20':>5} │ {'NDCG@5':>6} {'NDCG@10':>7} {'NDCG@20':>7}"
    )
    print(f"  {'─' * (W-2)}")

    for method in all_results:
        m = all_results[method].get("test", {})
        if not m:
            continue
        print(
            f"  {method:<17} │ "
            f"{m.get('precision@1',0):>4.1f}% {m.get('precision@3',0):>4.1f}% {m.get('precision@5',0):>4.1f}% "
            f"{m.get('precision@10',0):>4.1f}% {m.get('precision@20',0):>4.1f}% │ "
            f"{m.get('f1@5',0):>4.1f}% {m.get('f1@10',0):>4.1f}% {m.get('f1@20',0):>4.1f}% │ "
            f"{m.get('ndcg@5',0):>5.1f}% {m.get('ndcg@10',0):>6.1f}% {m.get('ndcg@20',0):>6.1f}%"
        )

    print(f"{'═' * W}")


# ═══════════════════════════════════════════════════════════════════
# Per-Method Summary (test split, detailed breakdown)
# ═══════════════════════════════════════════════════════════════════

def _print_per_method_summary(all_results):
    """Print per-method breakdown similar to the user's evaluation summary format."""
    W = 70
    print(f"\n{'═' * W}")
    print(f"  PER-METHOD BREAKDOWN (Test Split)")
    print(f"{'═' * W}")

    for method in all_results:
        m = all_results[method].get("test", {})
        if not m:
            continue

        print(f"\n--- {method.upper()} ---")
        print(f"  Total queries:       {m.get('total_queries', 0)}")
        print(f"  Recall@1:            {m.get('recall@1', 0):.1f}%")
        print(f"  Recall@5:            {m.get('recall@5', 0):.1f}%")
        print(f"  Recall@10:           {m.get('recall@10', 0):.1f}%")
        print(f"  Recall@20:           {m.get('recall@20', 0):.1f}%")
        print(f"  GT Recall@5:         {m.get('gt_recall@5', 0):.1f}%")
        print(f"  GT Recall@10:        {m.get('gt_recall@10', 0):.1f}%")
        print(f"  GT Recall@20:        {m.get('gt_recall@20', 0):.1f}%")
        print(f"  Full Coverage@20:    {m.get('full_coverage@20', 0):.1f}%")
        print(f"  MRR:                 {m.get('mrr', 0):.1f}%")
        print(f"  NDCG@5:              {m.get('ndcg@5', 0):.1f}%")
        print(f"  NDCG@20:             {m.get('ndcg@20', 0):.1f}%")
        print(f"  Avg GT Partitions:   {m.get('avg_gt_partitions', 0):.2f}")
        print(f"  Min/Max GT Parts:    {m.get('min_gt_partitions', 0)} / {m.get('max_gt_partitions', 0)}")
        print(f"  --- Hit Position ---")
        print(f"    Avg First Hit:     {m.get('avg_first_hit_pos', 0):.2f}")
        print(f"    Median First Hit:  {m.get('median_first_hit_pos', 0):.1f}")
        print(f"  --- Latency ---")
        print(f"    Avg Latency:       {m.get('avg_latency_ms', 0):.2f}ms")
        print(f"    P50 Latency:       {m.get('p50_latency_ms', 0):.2f}ms")
        print(f"    P95 Latency:       {m.get('p95_latency_ms', 0):.2f}ms")
        print(f"    P99 Latency:       {m.get('p99_latency_ms', 0):.2f}ms")

    print(f"\n{'═' * W}")


# ═══════════════════════════════════════════════════════════════════
# Overall Summary (cross-method comparison, ranking)
# ═══════════════════════════════════════════════════════════════════

def _print_overall_summary(all_results):
    """Print cross-method comparison with ranking and best/worst analysis."""
    test_results = {m: r["test"] for m, r in all_results.items() if "test" in r}
    if not test_results:
        log.warning("No test results to summarize.")
        return

    key_metrics = [
        "recall@1", "recall@5", "recall@10", "recall@20",
        "gt_recall@5", "gt_recall@10", "gt_recall@20",
        "precision@1", "precision@5",
        "ndcg@5", "ndcg@10", "ndcg@20",
        "mrr", "full_coverage@20",
    ]

    W = 95
    print(f"\n{'═' * W}")
    print(f"  OVERALL SUMMARY (Test Split)")
    print(f"{'═' * W}")

    print(f"\n  {'Metric':<17} │ {'Best Method':<17} {'Score':>8} │ {'Worst Method':<17} {'Score':>8} │ {'Δ':>6}")
    print(f"  {'─' * (W-2)}")

    for metric in key_metrics:
        scores = {m: r.get(metric, 0) for m, r in test_results.items()}
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        delta = scores[best] - scores[worst]
        print(
            f"  {metric:<17} │ {best:<17} {scores[best]:>7.1f}% │ "
            f"{worst:<17} {scores[worst]:>7.1f}% │ {delta:>5.1f}%"
        )

    print(f"\n  {'─' * (W-2)}")
    print(f"  AGGREGATE RANKING (avg rank across {len(key_metrics)} metrics, lower = better):\n")

    method_ranks: Dict[str, List[int]] = defaultdict(list)
    for metric in key_metrics:
        scores = sorted(test_results.items(), key=lambda x: x[1].get(metric, 0), reverse=True)
        for rank, (method, _) in enumerate(scores, 1):
            method_ranks[method].append(rank)

    ranked = sorted(method_ranks.items(), key=lambda x: np.mean(x[1]))

    for i, (method, ranks) in enumerate(ranked, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        avg_r = np.mean(ranks)
        wins = sum(
            1 for metric in key_metrics
            if max(test_results, key=lambda m, metric=metric: test_results[m].get(metric, 0)) == method
        )
        print(
            f"  {medal} #{i} {method:<17} avg_rank={avg_r:.1f}  "
            f"(best={min(ranks)}, worst={max(ranks)})  "
            f"metric_wins={wins}/{len(key_metrics)}"
        )

    print(f"\n  {'─' * (W-2)}")
    print(f"  LATENCY COMPARISON:")
    print(f"  * Note: ColBERT intrinsically includes query text-encoding time.")
    print(f"    Other models evaluate via internally pre-encoded vectors.\n")
    for method in test_results:
        t = test_results[method]
        avg = t.get("avg_latency_ms", 0)
        p95 = t.get("p95_latency_ms", 0)
        bar = "█" * min(int(avg / 2), 40)
        print(f"  {method:<17} avg={avg:>7.1f}ms  p95={p95:>7.1f}ms  {bar}")

    print(f"\n{'═' * W}")


# ═══════════════════════════════════════════════════════════════════

def run_partition_selection_benchmark(dataset: str = "2wiki"):
    return run_benchmark(dataset=dataset)


if __name__ == "__main__":
    run_benchmark()