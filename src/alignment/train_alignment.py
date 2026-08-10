import os
import sys
import json
import random
import logging
from collections import defaultdict
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:  # torch_geometric is only needed for the GNN alignment path; keep it optional so
    from torch_geometric.data import Data, Batch          # non-GNN importers (e.g. overlap_retrain
    from torch_geometric.utils import subgraph            # -> train_mlp -> here) don't require it
except ModuleNotFoundError:                               # (fresh cloud envs without torch-geometric)
    Data = Batch = subgraph = None
from tqdm import tqdm
from typing import Dict, List, Tuple
import faiss

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pipeline.standardizer import load_nodes
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.gnn_encoders import GINAlignmentEncoder, GCNAlignmentEncoder, SAGEAlignmentEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Shared Feature Augmentation
# ═══════════════════════════════════════════════════════════════════

def augment_node_features(node_features, graph_path, partition_ids, num_partitions, num_doc_nodes):
    """
    Augment raw FAISS node features with topological signals:
      - is_node flag (1-d)
      - normalized degree (1-d)
      - clustering coefficient (1-d)
      - partition one-hot (num_partitions-d)

    Returns augmented features tensor. Falls back to raw features on error.
    Used by both training and evaluation.
    """
    try:
        import networkx as nx
        from torch_geometric.utils import to_networkx
        import torch.nn.functional as Func

        full_graph = torch.load(graph_path, weights_only=False)
        G_nx = to_networkx(full_graph, to_undirected=True)

        # 1. is_node flag
        is_node_tensor = torch.ones((num_doc_nodes, 1), dtype=torch.float32)

        # 2. Normalized degree
        degrees = dict(G_nx.degree())
        max_deg = max(degrees.values()) if degrees else 1
        deg_tensor = torch.tensor(
            [degrees.get(i, 0) / max_deg for i in range(num_doc_nodes)],
            dtype=torch.float32
        ).unsqueeze(1)

        # 3. Clustering coefficient
        clustering = nx.clustering(G_nx)
        clust_tensor = torch.tensor(
            [clustering.get(i, 0.0) for i in range(num_doc_nodes)],
            dtype=torch.float32
        ).unsqueeze(1)

        # 4. Partition one-hot
        part_tensor = torch.tensor(partition_ids, dtype=torch.long)
        part_onehot = Func.one_hot(part_tensor, num_classes=num_partitions).float()

        augmented = torch.cat(
            [node_features, is_node_tensor, deg_tensor, clust_tensor, part_onehot],
            dim=1
        )
        log.info(f"Topological features injected. Shape upgraded: {node_features.shape} → {augmented.shape}")
        return augmented
    except Exception as e:
        log.warning(f"Error computing topological attributes: {e}. Using raw features.")
        return node_features


# ═══════════════════════════════════════════════════════════════════
# Deterministic Split Helper
# ═══════════════════════════════════════════════════════════════════

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


def get_split_pairs(nodes, partition_map, dataset_name: str) -> Dict[str, List[Tuple]]:
    """
    Collect all (node, gt_pids) pairs for a dataset, then split them
    deterministically into train / val / test (70/20/10).
    Returns dict with keys "train", "val", "test", each mapping to a list of (node, gt_pids).
    """
    all_pairs = []
    for node in nodes:
        if node.metadata.get("source") == dataset_name and node.metadata.get("type") == "question":
            gt_pids = []
            for neighbor_id in node.neighbors:
                pid = partition_map.get(neighbor_id)
                if pid is not None:
                    gt_pids.append(int(pid))
            if gt_pids:
                all_pairs.append((node, list(set(gt_pids))))

    if not all_pairs:
        log.warning(f"No {dataset_name} questions with ground-truth partitions found!")
        return {"train": [], "val": [], "test": []}

    all_pairs.sort(key=lambda p: p[0].node_id)
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(all_pairs)

    n = len(all_pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:],
    }
    log.info(
        f"Split {dataset_name}: {len(splits['train'])} train / "
        f"{len(splits['val'])} val / {len(splits['test'])} test"
    )
    return splits


# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════

class AlignmentDataset(Dataset):
    """
    Inputs: dense embeddings of question nodes.
    Targets: list of all ground-truth partition IDs.
    """
    def __init__(self, pairs: List[Tuple], encoder, split_name: str = "train"):
        self.samples: List[Dict] = []

        if not pairs:
            return

        log.info(f"Pre-encoding {len(pairs)} {split_name} queries...")
        texts = [n.content for n, _ in pairs]
        embeddings = encoder.encode(texts)
        faiss.normalize_L2(embeddings)

        for i, (_, pids) in enumerate(pairs):
            self.samples.append({
                "query_emb": embeddings[i],
                "all_gt_pids": pids
            })

        log.info(f"Dataset ({split_name}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return torch.tensor(sample["query_emb"], dtype=torch.float32), sample["all_gt_pids"]


def collate_fn(batch):
    query_embs, all_gt_pids = zip(*batch)
    return torch.stack(query_embs), list(all_gt_pids)


# ═══════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════

def multilabel_infonce_loss(zq, coarse_part_embs, query_coarse_ids, temperature=0.1):
    """
    Multi-label (soft-OR) InfoNCE contrastive loss.

    NOTE: this is NOT a coverage loss. Its numerator is logsumexp over the
    positive set, so it is satisfied by the single STRONGEST positive and does
    not enforce that ALL required partitions rank highly. For a true FullCov /
    worst-positive objective see src/alignment/coverage_losses.py
    (partition_coverage_loss) and the coverage_* losses in train_mlp.py.
    (Formerly named `partition_coverage_loss`, which was misleading.)

    zq: (B, D)
    coarse_part_embs: (num_coarse, D)
    query_coarse_ids: list of lists of coarse partition ids
    """
    B, _ = zq.shape
    num_coarse = coarse_part_embs.shape[0]

    zq = F.normalize(zq, p=2, dim=-1)
    coarse_part_embs = F.normalize(coarse_part_embs, p=2, dim=-1)

    logits = torch.matmul(zq, coarse_part_embs.T) / temperature

    pos_mask = torch.zeros(B, num_coarse, dtype=torch.bool, device=zq.device)
    for i, part_ids in enumerate(query_coarse_ids):
        for pid in part_ids:
            if 0 <= pid < num_coarse:
                pos_mask[i, pid] = True

    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=zq.device, requires_grad=True)

    logits = logits[valid]
    pos_mask = pos_mask[valid]

    log_sum_all = torch.logsumexp(logits, dim=1)
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    log_sum_pos = torch.logsumexp(pos_logits, dim=1)

    loss = (log_sum_all - log_sum_pos).mean()
    return loss


# ═══════════════════════════════════════════════════════════════════
# Training Helpers
# ═══════════════════════════════════════════════════════════════════

def get_partition_subgraphs(edge_index, node_features, partition_map_list, num_partitions):
    """Pre-extract all partition subgraphs as PyG Data objects."""
    log.info(f"Extracting {num_partitions} partition subgraphs...")
    partition_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for node_idx, pid in enumerate(partition_map_list):
        partition_to_nodes[pid].append(node_idx)

    subgraphs_data = []
    for pid in range(num_partitions):
        nodes_in_part = torch.tensor(partition_to_nodes[pid], dtype=torch.long)
        if len(nodes_in_part) == 0:
            sub_data = Data(
                x=torch.zeros((1, node_features.size(1))),
                edge_index=torch.zeros((2, 0), dtype=torch.long)
            )
        else:
            sub_ei, _ = subgraph(nodes_in_part, edge_index, relabel_nodes=True)
            sub_x = node_features[nodes_in_part]
            sub_data = Data(x=sub_x, edge_index=sub_ei)
        subgraphs_data.append(sub_data)

    return subgraphs_data


def _is_mlp_family(model_type: str) -> bool:
    return model_type in ["mlp", "mlp_topo"]


def _uses_topology(model_type: str) -> bool:
    return model_type in ["mlp_topo", "gin", "gcn", "sage"]


def _is_gnn(model_type: str) -> bool:
    return model_type in ["gin", "gcn", "sage"]


def _compute_eval_loss(model, model_type, dataloader, all_partition_embs, device):
    """Compute average loss on a dataloader without gradients."""
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for query_embs, query_gt_pids in dataloader:
            query_embs = query_embs.to(device)
            if _is_mlp_family(model_type):
                q_proj = model(query_embs)
            else:
                q_proj = model.project_text(query_embs, device)
            loss = multilabel_infonce_loss(q_proj, all_partition_embs, query_gt_pids)
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


# ═══════════════════════════════════════════════════════════════════
# Main Training Function
# ═══════════════════════════════════════════════════════════════════

GLOBAL_CACHE: Dict = {}


def train(
    model_type="mlp",
    dataset_name="2wiki",
    epochs=100,
    batch_size=64,
    lr=1e-4,
    nodes_path="data/processed/master_nodes.json",
    partition_map_path=None,
    graph_path=None,
    output_path=None,
):
    global GLOBAL_CACHE

    if output_path is None:
        output_path = f"checkpoints/alignment_{model_type}_{dataset_name}.pth"

    src_dir = f"data/ukb_storage/{dataset_name}"
    if partition_map_path is None:
        partition_map_path = os.path.join(src_dir, "partition_map.json")
    if graph_path is None:
        graph_path = os.path.join(src_dir, "graph.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training {model_type} for {dataset_name} on {device}...")

    # ── Shared per-dataset base cache ──
    base_cache_key = f"base_data_{dataset_name}"
    if base_cache_key not in GLOBAL_CACHE:
        all_nodes = load_nodes(nodes_path)
        source_nodes = [n for n in all_nodes if n.metadata.get("source") == dataset_name]
        log.info(f"Filtered to {len(source_nodes)} nodes for source '{dataset_name}' (from {len(all_nodes)} total)")

        with open(partition_map_path, "r") as f:
            partition_map = json.load(f)

        encoder = DenseEncoder()
        node_index_path = os.path.join(src_dir, "nodes.index")

        if os.path.exists(node_index_path):
            log.info("Loading node features directly from FAISS index to save time...")
            index = faiss.read_index(node_index_path)
            features_np = index.reconstruct_n(0, index.ntotal)
            raw_node_features = torch.tensor(features_np, dtype=torch.float32)

            # FAISS only contains document/entity nodes (not questions)
            doc_nodes = [n for n in source_nodes if n.metadata.get("type") != "question"]
            log.info(f"Aligned doc-only nodes ({len(doc_nodes)}) to FAISS index ({index.ntotal} vectors)")
        else:
            log.info("Encoding all nodes from scratch...")
            doc_nodes = [n for n in source_nodes if n.metadata.get("type") != "question"]
            node_contents = [n.content for n in doc_nodes]
            raw_node_features = torch.tensor(encoder.encode(node_contents), dtype=torch.float32)

        embed_dim = raw_node_features.size(1)

        doc_node_id_to_idx = {n.node_id: i for i, n in enumerate(doc_nodes)}
        num_doc_nodes = len(doc_nodes)
        partition_ids = [0] * num_doc_nodes
        for node_id, pid in partition_map.items():
            if node_id in doc_node_id_to_idx:
                partition_ids[doc_node_id_to_idx[node_id]] = int(pid)

        num_partitions = max(partition_ids) + 1 if partition_ids else 0

        GLOBAL_CACHE[base_cache_key] = {
            "source_nodes": source_nodes,
            "partition_map": partition_map,
            "encoder": encoder,
            "raw_node_features": raw_node_features,
            "embed_dim": embed_dim,
            "partition_ids": partition_ids,
            "num_partitions": num_partitions,
            "num_doc_nodes": num_doc_nodes,
        }
    else:
        log.info(f"Loaded base dataset cache for {dataset_name}.")

    base_data = GLOBAL_CACHE[base_cache_key]
    source_nodes = base_data["source_nodes"]
    partition_map = base_data["partition_map"]
    encoder = base_data["encoder"]
    raw_node_features = base_data["raw_node_features"]
    embed_dim = base_data["embed_dim"]
    partition_ids = base_data["partition_ids"]
    num_partitions = base_data["num_partitions"]
    num_doc_nodes = base_data["num_doc_nodes"]

    # ── Model-specific feature cache ──
    feature_variant = "topo" if _uses_topology(model_type) else "raw"
    feature_cache_key = f"node_features_{dataset_name}_{feature_variant}"

    if feature_cache_key not in GLOBAL_CACHE:
        if _uses_topology(model_type):
            node_features = augment_node_features(
                raw_node_features,
                graph_path,
                partition_ids,
                num_partitions,
                num_doc_nodes,
            )
        else:
            node_features = raw_node_features
            log.info(f"Using raw node features for {model_type}. Shape: {node_features.shape}")

        GLOBAL_CACHE[feature_cache_key] = node_features
    else:
        node_features = GLOBAL_CACHE[feature_cache_key]
        log.info(f"Loaded {feature_variant} node features for {dataset_name} from cache.")

    # ── Prepare model ──
    hidden_dim = 256
    feat_dim = node_features.size(1)

    if model_type == "mlp":
        # raw 384-d query embeddings -> 384-d centroid space
        model = TextPartitionMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim
        ).to(device)
    elif model_type == "mlp_topo":
        # still consumes raw query embeddings; topology only affects partition-side representation
        model = TextPartitionMLP(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim
        ).to(device)
    elif model_type == "gin":
        model = GINAlignmentEncoder(
            in_channels=feat_dim,
            hidden_channels=hidden_dim,
            out_channels=embed_dim
        ).to(device)
    elif model_type == "gcn":
        model = GCNAlignmentEncoder(
            in_channels=feat_dim,
            hidden_channels=hidden_dim,
            out_channels=embed_dim
        ).to(device)
    elif model_type == "sage":
        model = SAGEAlignmentEncoder(
            in_channels=feat_dim,
            hidden_channels=hidden_dim,
            out_channels=embed_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ── Cache: partition subgraphs (GNN only, topo features only) ──
    partition_subgraphs = None
    if _is_gnn(model_type):
        sg_cache_key = f"partition_subgraphs_{dataset_name}_{feature_variant}"
        if sg_cache_key not in GLOBAL_CACHE:
            full_graph = torch.load(graph_path, weights_only=False)
            partition_subgraphs = get_partition_subgraphs(
                full_graph.edge_index,
                node_features,
                partition_ids,
                num_partitions
            )
            GLOBAL_CACHE[sg_cache_key] = partition_subgraphs
        else:
            partition_subgraphs = GLOBAL_CACHE[sg_cache_key]
            log.info("Loaded partition subgraphs from cache.")

    # ── Cache: split pairs + datasets (shared per dataset) ──
    split_cache_key = f"splits_{dataset_name}"
    if split_cache_key not in GLOBAL_CACHE:
        split_pairs = get_split_pairs(source_nodes, partition_map, dataset_name)
        train_ds = AlignmentDataset(split_pairs["train"], encoder, split_name=f"{dataset_name}/train")
        val_ds = AlignmentDataset(split_pairs["val"], encoder, split_name=f"{dataset_name}/val")
        test_ds = AlignmentDataset(split_pairs["test"], encoder, split_name=f"{dataset_name}/test")
        GLOBAL_CACHE[split_cache_key] = {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "split_pairs": split_pairs,
        }
    else:
        cached = GLOBAL_CACHE[split_cache_key]
        train_ds = cached["train"]
        val_ds = cached["val"]
        test_ds = cached["test"]
        log.info(f"Loaded {dataset_name} split datasets from cache.")

    if len(train_ds) == 0:
        log.warning(f"Empty training set for {dataset_name}, skipping.")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
        if len(val_ds) > 0 else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
        if len(test_ds) > 0 else None
    )

    # ── Cache: partition representations ──
    topo_partition_prototypes_to_save = None

    if _is_mlp_family(model_type):
        centroid_variant = "topo" if model_type == "mlp_topo" else "raw"
        cent_cache_key = f"centroids_gpu_{dataset_name}_{centroid_variant}"

        if cent_cache_key not in GLOBAL_CACHE:
            if model_type == "mlp":
                log.info(f"Loading pre-computed centroids for {dataset_name} alignment...")
                centroid_index_path = os.path.join(src_dir, "centroids.index")
                cent_idx = faiss.read_index(centroid_index_path)
                num_partitions_idx = cent_idx.ntotal
                if num_partitions_idx != num_partitions:
                    log.warning(
                        f"[Mismatch] Partition count inconsistent for dataset '{dataset_name}': "
                        f"centroids.index has {num_partitions_idx}, while partition_map.json lists {num_partitions}. "
                        "Using centroid index count to avoid crash."
                    )
                    num_partitions = num_partitions_idx

                centroids_np = np.zeros((num_partitions, embed_dim), dtype=np.float32)
                for i in range(num_partitions):
                    centroids_np[i] = cent_idx.reconstruct(i)

                centroids_gpu = torch.tensor(centroids_np, dtype=torch.float32).to(device)
                GLOBAL_CACHE[cent_cache_key] = centroids_gpu

            else:
                # mlp_topo: build topology-aware partition prototypes directly in embed_dim space
                # by starting from raw embedding centroid and injecting compact topo summaries.
                log.info(f"Building topology-aware partition prototypes for {dataset_name}...")

                # Recover compact topo stats from augmented features:
                # [raw_embed | is_node | norm_degree | clustering | partition_onehot]
                raw_dim = embed_dim
                is_node_idx = raw_dim
                degree_idx = raw_dim + 1
                clustering_idx = raw_dim + 2

                partition_buckets: Dict[int, List[torch.Tensor]] = defaultdict(list)
                degree_buckets: Dict[int, List[float]] = defaultdict(list)
                clustering_buckets: Dict[int, List[float]] = defaultdict(list)

                for node_idx, pid in enumerate(partition_ids):
                    pid = int(pid)
                    partition_buckets[pid].append(raw_node_features[node_idx])
                    degree_buckets[pid].append(float(node_features[node_idx, degree_idx].item()))
                    clustering_buckets[pid].append(float(node_features[node_idx, clustering_idx].item()))

                topo_centroids = np.zeros((num_partitions, embed_dim), dtype=np.float32)
                for pid in range(num_partitions):
                    if partition_buckets[pid]:
                        raw_stack = torch.stack(partition_buckets[pid], dim=0)
                        raw_centroid = raw_stack.mean(dim=0).cpu().numpy()

                        mean_deg = float(np.mean(degree_buckets[pid])) if degree_buckets[pid] else 0.0
                        mean_clust = float(np.mean(clustering_buckets[pid])) if clustering_buckets[pid] else 0.0

                        # Inject topology into the dense space in a simple, deterministic way.
                        # This preserves embed_dim and avoids any random, untrained projector.
                        topo_vec = raw_centroid.copy()
                        topo_vec = topo_vec * (1.0 + 0.10 * mean_deg + 0.10 * mean_clust)

                        # Small directional bias using first two coordinates for stability.
                        if embed_dim >= 2:
                            topo_vec[0] += mean_deg
                            topo_vec[1] += mean_clust

                        topo_centroids[pid] = topo_vec
                    else:
                        topo_centroids[pid] = np.zeros((embed_dim,), dtype=np.float32)

                # Normalize once so training and inference use the same geometry
                norms = np.linalg.norm(topo_centroids, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                topo_centroids = topo_centroids / norms

                topo_partition_prototypes_to_save = topo_centroids.copy()
                centroids_gpu = torch.tensor(topo_centroids, dtype=torch.float32).to(device)
                GLOBAL_CACHE[cent_cache_key] = centroids_gpu
        else:
            centroids_gpu = GLOBAL_CACHE[cent_cache_key].to(device)
            if model_type == "mlp_topo":
                topo_partition_prototypes_to_save = centroids_gpu.detach().cpu().numpy()
    else:
        centroids_gpu = None

    # ── Training loop with early stopping ──
    EARLY_STOP_PATIENCE = 20
    log.info(f"Using LR={lr:.1e} for {model_type} (scheduler will adapt)")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=7
    )

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    pbar = tqdm(range(epochs), desc=f"Training {model_type.upper()} ({dataset_name})")
    for epoch in pbar:
        model.train()
        total_loss = 0.0

        # Compute partition embeddings once per epoch
        if _is_gnn(model_type):
            model.eval()
            with torch.no_grad():
                all_part_batch = Batch.from_data_list(partition_subgraphs).to(device)
                all_partition_embs = model(
                    all_part_batch.x,
                    all_part_batch.edge_index,
                    all_part_batch.batch
                )
            model.train()
        else:
            all_partition_embs = centroids_gpu

        all_partition_embs = F.normalize(all_partition_embs, dim=-1)

        # Train
        for query_embs, query_gt_pids in train_loader:
            query_embs = query_embs.to(device)

            if _is_mlp_family(model_type):
                q_proj = model(query_embs)
            else:
                q_proj = model.project_text(query_embs, device)

            q_proj = F.normalize(q_proj, dim=-1)
            loss = multilabel_infonce_loss(q_proj, all_partition_embs, query_gt_pids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        postfix = {"TrL": f"{avg_train_loss:.4f}"}
        val_loss = avg_train_loss
        if val_loader:
            val_loss = _compute_eval_loss(model, model_type, val_loader, all_partition_embs, device)
            postfix["VaL"] = f"{val_loss:.4f}"
        if test_loader:
            test_loss = _compute_eval_loss(model, model_type, test_loader, all_partition_embs, device)
            postfix["TeL"] = f"{test_loss:.4f}"

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            postfix["★"] = "best"
        else:
            epochs_without_improvement += 1

        pbar.set_postfix(postfix)

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            log.info(
                f"Early stopping at epoch {epoch + 1} "
                f"(no val improvement for {EARLY_STOP_PATIENCE} epochs). "
                f"Best VaL: {best_val_loss:.4f}"
            )
            break

    # ── Save best checkpoint ──
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_state = best_state_dict if best_state_dict is not None else model.state_dict()

    actual_input_dim = embed_dim if _is_mlp_family(model_type) else feat_dim
    ckpt = {
        "model_state_dict": save_state,
        "model_type": model_type,
        "input_dim": actual_input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": embed_dim,
        "best_val_loss": best_val_loss,
        "feature_variant": feature_variant,
    }

    if model_type == "mlp_topo" and topo_partition_prototypes_to_save is not None:
        ckpt["topo_partition_prototypes"] = topo_partition_prototypes_to_save

    torch.save(ckpt, output_path)
    log.info(f"Saved best model (VaL={best_val_loss:.4f}) to {output_path}")

    # Move cached GPU tensors to CPU to save VRAM
    for key in [
        f"centroids_gpu_{dataset_name}_raw",
        f"centroids_gpu_{dataset_name}_topo",
    ]:
        if key in GLOBAL_CACHE and isinstance(GLOBAL_CACHE[key], torch.Tensor):
            GLOBAL_CACHE[key] = GLOBAL_CACHE[key].cpu()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        default="mlp",
        choices=["mlp", "mlp_topo", "gin", "gcn", "sage"]
    )
    parser.add_argument("--dataset", type=str, default="2wiki")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["squad", "musique", "2wiki", "metaqa"]:
            train(model_type=args.model_type, dataset_name=ds, epochs=args.epochs)
    else:
        train(model_type=args.model_type, dataset_name=args.dataset, epochs=args.epochs)