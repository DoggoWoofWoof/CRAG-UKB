"""
ColBERT Partition Router — ported from v4-query-graph-selection branch.
Source: src/crag/routing/colbert.py

Implements MaxSim late-interaction for partition-level routing (the Teleport phase).
Each partition is represented by a token matrix (K-means cluster centers of its node embeddings).
Queries are encoded and scored against each partition via MaxSim.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from .structural import StructuralAligner

logger = logging.getLogger(__name__)


class ColBERTPartitionRouter:
    """
    Production Partition-Level ColBERT Router.
    Uses MaxSim late interaction for precise partition selection.

    Teleport Phase:
        1. Build partition token matrices offline (K-means over node embeddings)
        2. At query time: encode query → compute MaxSim against all partitions
        3. Return top-K partition IDs with their scores

    Used exclusively by:
        - v4_graph_selection/pipeline.py (Teleport step)
    """

    def __init__(self, partition_matrix_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.partition_embs = None      # [NumPartitions, MaxTokens, Hidden]
        self.partition_masks = None     # [NumPartitions, MaxTokens] bool
        self.partition_metadata: List[Dict] = []
        self.partition_centroids = None # [NumPartitions, Hidden]
        self.encoder = None
        self.structural_aligner = StructuralAligner()

        if partition_matrix_path:
            self.load(partition_matrix_path)

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("intfloat/e5-base-v2", device=self.device)
        return self.encoder

    def build_partition_matrices(self, graph_engine, num_tokens_per_partition: int = 32):
        """
        Build partition token matrices from graph data.
        Uses K-means clustering on partition node embeddings to select representative tokens.
        """
        if not hasattr(graph_engine.data, "part_id") or graph_engine.data.part_id is None:
            logger.error("Graph has no partitions. Run partitioner.py first.")
            return

        if graph_engine.data.x is None:
            logger.error("Graph has no embeddings. Run embedder.py first.")
            return

        part_ids = graph_engine.data.part_id
        embeddings = graph_engine.data.x
        num_partitions = int(part_ids.max().item()) + 1
        hidden_dim = embeddings.size(1)

        logger.info(f"Building ColBERT matrices for {num_partitions} partitions...")

        self.partition_embs = torch.zeros(num_partitions, num_tokens_per_partition, hidden_dim)
        self.partition_masks = torch.zeros(num_partitions, num_tokens_per_partition, dtype=torch.bool)
        self.partition_centroids = torch.zeros(num_partitions, hidden_dim)
        self.partition_metadata = [{} for _ in range(num_partitions)]

        for pid in range(num_partitions):
            mask = (part_ids == pid)
            partition_nodes = mask.nonzero(as_tuple=True)[0]
            partition_embs = embeddings[partition_nodes]
            num_nodes = len(partition_nodes)

            if num_nodes == 0:
                continue

            if num_nodes <= num_tokens_per_partition:
                self.partition_embs[pid, :num_nodes] = partition_embs
                self.partition_masks[pid, :num_nodes] = True
            else:
                try:
                    from sklearn.cluster import KMeans
                    km = KMeans(n_clusters=num_tokens_per_partition, random_state=42, n_init=10)
                    km.fit(partition_embs.cpu().numpy())
                    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
                    self.partition_embs[pid] = centers
                    self.partition_masks[pid] = True
                except Exception as e:
                    logger.warning(f"KMeans failed for partition {pid}: {e}. Using first N nodes.")
                    self.partition_embs[pid] = partition_embs[:num_tokens_per_partition]
                    self.partition_masks[pid] = True

            self.partition_centroids[pid] = partition_embs.mean(dim=0)

    def route(self, query: str, top_k: int = 3) -> Tuple[List[int], List[float]]:
        """
        Route a query to the top-K most relevant partitions via MaxSim.

        Returns:
            partition_ids: List[int] — top-K partition indices
            scores: List[float] — corresponding MaxSim scores
        """
        if self.partition_embs is None:
            logger.warning("Partition matrices not built. Returning empty routing.")
            return [], []

        encoder = self._get_encoder()
        query_emb = torch.tensor(
            encoder.encode([query], normalize_embeddings=True),
            dtype=torch.float32,
            device=self.device
        )  # [1, Hidden]

        # Centroid similarity (fast coarse pass)
        centroids = self.partition_centroids.to(self.device)
        centroid_scores = F.cosine_similarity(query_emb, centroids, dim=-1)

        # Top-K by centroid for MaxSim refinement (avoid full scan)
        candidates = min(top_k * 5, len(centroid_scores))
        candidate_ids = centroid_scores.topk(candidates).indices.tolist()

        # MaxSim late interaction over candidate partitions
        maxsim_scores = {}
        for pid in candidate_ids:
            embs = self.partition_embs[pid].to(self.device)  # [MaxTokens, Hidden]
            valid = self.partition_masks[pid].to(self.device)
            if not valid.any():
                continue
            sim = (query_emb @ embs[valid].T).squeeze(0)     # [ValidTokens]
            maxsim_scores[pid] = sim.max().item()

        sorted_partitions = sorted(maxsim_scores, key=maxsim_scores.get, reverse=True)[:top_k]
        scores = [maxsim_scores[p] for p in sorted_partitions]

        return sorted_partitions, scores

    def save(self, path: str):
        torch.save({
            "partition_embs": self.partition_embs,
            "partition_masks": self.partition_masks,
            "partition_centroids": self.partition_centroids,
            "partition_metadata": self.partition_metadata,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.partition_embs = ckpt["partition_embs"]
        self.partition_masks = ckpt["partition_masks"]
        self.partition_centroids = ckpt["partition_centroids"]
        self.partition_metadata = ckpt["partition_metadata"]
        logger.info(f"Loaded ColBERT partition matrices from {path}")
