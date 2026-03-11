"""Structural Aligner — ported from v4-query-graph-selection branch."""

import torch
import logging
from typing import Dict, Any, List
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class StructuralAligner:
    """
    Evaluates how well a Query Graph fits into a Graph Partition.
    Computes structural fingerprints (node type histograms, edge relation sets)
    and returns a [0, 1] alignment score.

    Used by ColBERTPartitionRouter to blend structural and semantic scores.
    """

    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings

    def compute_fingerprint(self, data: Data) -> Dict[str, Any]:
        fp = {"node_types": set(), "edge_types": set(), "num_nodes": data.num_nodes}
        if hasattr(data, "node_types"):
            fp["node_types"].update(data.node_types)
        if hasattr(data, "edge_types"):
            fp["edge_types"].update(data.edge_types)
        return fp

    def score(self, query_graph: Data, partition_fp: Dict[str, Any]) -> float:
        if query_graph is None or partition_fp is None:
            return 0.5
        if not hasattr(query_graph, "num_nodes") or query_graph.num_nodes == 0:
            return 0.5
        try:
            q_fp = self.compute_fingerprint(query_graph)
            if not q_fp["node_types"] and not q_fp["edge_types"]:
                return 0.5
            score, n = 0.0, 0
            if q_fp["node_types"]:
                p_types = partition_fp.get("node_types", set())
                if p_types:
                    score += len(q_fp["node_types"] & p_types) / len(q_fp["node_types"])
                    n += 1
            if q_fp["edge_types"]:
                p_rels = partition_fp.get("edge_types", set())
                if p_rels:
                    score += len(q_fp["edge_types"] & p_rels) / len(q_fp["edge_types"])
                    n += 1
            return score / n if n > 0 else 0.5
        except Exception as e:
            logger.error(f"StructuralAligner.score error: {e}")
            return 0.5

    def batch_score(self, query_graph: Data, partition_fps: List[Dict[str, Any]]) -> List[float]:
        if not partition_fps:
            return []
        try:
            return [self.score(query_graph, fp) for fp in partition_fps]
        except Exception as e:
            logger.error(f"StructuralAligner.batch_score error: {e}")
            return [0.5] * len(partition_fps)
