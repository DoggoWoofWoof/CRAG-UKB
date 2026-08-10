import os
import pickle
import faiss
import numpy as np
import torch
import json
import logging
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Set
from src.pipeline.standardizer import StandardNode, load_nodes

log = logging.getLogger(__name__)


class CoreEngine:
    """Engine for all UKB access — vector, graph, centroid, ColBERT.

    Each instance is scoped to a single dataset source.  The caller
    passes ``source`` (e.g. "squad", "musique", "2wiki") and the engine
    loads indices from ``storage_dir/{source}/``.
    """

    def __init__(self, storage_dir: str = "data/ukb_storage",
                 master_nodes_path: str = "data/processed/master_nodes.json",
                 source: str = "", index_subdir: str = ""):
        self.source = source
        self.storage_dir = storage_dir
        # index_subdir: when set (e.g. "gte_qwen"), the graph / partition_map / centroids are read
        # from src_dir/index_subdir — i.e. the kNN graph + METIS partitions were built in THAT
        # encoder's space, not MiniLM's. Retrieval embeddings already come from the subdir, so this
        # makes every operation (kNN, partitioning, centroids, retrieval) use one base encoder.
        self.index_subdir = index_subdir
        # Per-source master file override (non-destructive): if the caller left the
        # default master path but a data/processed/master_nodes_{source}.json exists
        # (e.g. the label-free "{ds}_clean" rebuild), prefer it.
        if source and master_nodes_path == "data/processed/master_nodes.json":
            per_source = os.path.join("data", "processed", f"master_nodes_{source}.json")
            if os.path.exists(per_source):
                master_nodes_path = per_source
                log.info(f"Using per-source master file: {per_source}")
        self.master_nodes_path = master_nodes_path

        # Resolve the actual index directory
        if source:
            src_dir = os.path.join(storage_dir, source)
        else:
            src_dir = storage_dir
        self._src_dir = src_dir
        # Resolve where the per-encoder graph/partition/centroids live (falls back to root if absent).
        idx_dir = os.path.join(src_dir, index_subdir) if index_subdir else src_dir
        if index_subdir and not os.path.exists(os.path.join(idx_dir, "partition_map.json")):
            log.warning(f"index_subdir '{index_subdir}' has no partition_map.json; "
                        f"falling back to root substrate {src_dir}")
            idx_dir = src_dir
        self._idx_dir = idx_dir

        # Load nodes and filter to source
        all_nodes = load_nodes(master_nodes_path)
        if source:
            source_nodes = [n for n in all_nodes if n.metadata.get("source") == source]
        else:
            source_nodes = all_nodes

        # all_nodes: everything including questions (cached for training/eval queries)
        self.all_nodes = source_nodes
        # nodes: doc-only — aligned with FAISS node_index, graph.pt, bm25.pkl, partition_map
        self.nodes = [n for n in source_nodes if n.metadata.get("type") != "question"]
        self.node_id_to_idx = {n.node_id: i for i, n in enumerate(self.nodes)}

        # ── Core Indices ────────────────────────────────────────────────
        log.info(f"Loading UKB Indices for source='{source or 'ALL'}' from {src_dir}...")
        self.node_index = faiss.read_index(os.path.join(src_dir, "nodes.index"))

        with open(os.path.join(src_dir, "bm25.pkl"), "rb") as f:
            self.bm25 = pickle.load(f)

        self.graph = torch.load(os.path.join(idx_dir, "graph.pt"), weights_only=False)
        self._attach_synthetic_neighbor_metadata()

        # ── Partition Map ───────────────────────────────────────────────
        self.partition_map: Dict[str, int] = self._load_json(
            os.path.join(idx_dir, "partition_map.json"))
        # Build reverse map: partition_id → list of node indices
        self._partition_to_nodes: Dict[int, List[int]] = {}
        for nid, pid in self.partition_map.items():
            self._partition_to_nodes.setdefault(int(pid), []).append(
                self.node_id_to_idx.get(nid, -1))
        # Remove any invalid entries
        for pid in self._partition_to_nodes:
            self._partition_to_nodes[pid] = [
                i for i in self._partition_to_nodes[pid] if i >= 0]

        # ── Centroid Index ──────────────────────────────────────────────
        centroid_path = os.path.join(idx_dir, "centroids.index")
        centroid_pids_path = os.path.join(idx_dir, "centroid_pids.json")
        if os.path.exists(centroid_path) and os.path.exists(centroid_pids_path):
            self.centroid_index = faiss.read_index(centroid_path)
            with open(centroid_pids_path, 'r') as f:
                self.centroid_pids = json.load(f)  # ordered list of partition ids
            log.info(f"Loaded centroid index with {len(self.centroid_pids)} partitions.")
        else:
            self.centroid_index = None
            self.centroid_pids = []
            log.warning("Centroid index not found — search_centroids() will be unavailable.")

        # ── ColBERT ─────────────────────────────────────────────────────
        colbert_dir = os.path.join(src_dir, "colbert_ukb")
        colbert_cent_dir = os.path.join(src_dir, "colbert_centroids")
        if os.path.exists(colbert_dir) and os.listdir(colbert_dir):
            try:
                from ragatouille import RAGPretrainedModel
                self.colbert = RAGPretrainedModel.from_index(colbert_dir)
                log.info("Loaded ColBERT index.")
            except Exception as e:
                self.colbert = None
                log.warning(f"ColBERT index exists but failed to load: {e}")
        else:
            self.colbert = None
            log.info("ColBERT index not found — search_colbert() will fall back.")

        if os.path.exists(colbert_cent_dir) and os.listdir(colbert_cent_dir):
            try:
                from ragatouille import RAGPretrainedModel
                self.colbert_centroid = RAGPretrainedModel.from_index(colbert_cent_dir)
                log.info("Loaded ColBERT Centroid index.")
            except Exception as e:
                self.colbert_centroid = None
                log.warning(f"ColBERT Centroid index exists but failed to load: {e}")
        else:
            self.colbert_centroid = None
            log.info("ColBERT Centroid index not found.")

        log.info(f"CoreEngine ready: source='{source or 'ALL'}', "
                 f"{len(self.nodes)} doc nodes, {len(self.all_nodes)} total (incl. questions).")

    # ═══════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════════

    def _load_json(self, path: str) -> Dict:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _attach_synthetic_neighbor_metadata(self) -> None:
        """Recover synthetic KNN edges from graph.pt for Level-3 pruning.

        The current master_nodes.json stores original dataset graph edges in
        node.neighbors, while graph.pt also contains index-time KNN bridges.
        CRAG's Level 3 needs to tell those apart so ablations can exclude
        synthetic edges without rebuilding the whole UKB.
        """
        edge_index = getattr(self.graph, "edge_index", None)
        if edge_index is None:
            return

        try:
            src_indices = edge_index[0].tolist()
            dst_indices = edge_index[1].tolist()
        except Exception as exc:
            log.warning(f"Could not inspect graph edge_index for synthetic edges: {exc}")
            return

        doc_node_ids = set(self.node_id_to_idx)
        original_neighbors: List[Set[str]] = [
            set(node.neighbors) & doc_node_ids for node in self.nodes
        ]
        for src_idx, node in enumerate(self.nodes):
            for neighbor_id in node.neighbors:
                dst_idx = self.node_id_to_idx.get(neighbor_id)
                if dst_idx is not None:
                    original_neighbors[dst_idx].add(node.node_id)
        synthetic_neighbors: Dict[int, Set[str]] = {}

        for src_idx, dst_idx in zip(src_indices, dst_indices):
            if not (0 <= src_idx < len(self.nodes) and 0 <= dst_idx < len(self.nodes)):
                continue

            dst_node_id = self.nodes[dst_idx].node_id
            if dst_node_id not in original_neighbors[src_idx]:
                synthetic_neighbors.setdefault(src_idx, set()).add(dst_node_id)

        synthetic_edge_count = 0
        for src_idx, neighbor_ids in synthetic_neighbors.items():
            if not neighbor_ids:
                continue
            node = self.nodes[src_idx]
            existing = set(node.metadata.get("synthetic_neighbors", []))
            existing.update(neighbor_ids)
            node.metadata["synthetic_neighbors"] = sorted(existing)
            synthetic_edge_count += len(neighbor_ids)

        if synthetic_edge_count:
            log.info(
                f"Recovered {synthetic_edge_count} synthetic graph edges for Level-3 pruning."
            )

    # ═══════════════════════════════════════════════════════════════════
    # Core Search Methods
    # ═══════════════════════════════════════════════════════════════════

    def search_dense(self, query_vector: np.ndarray, k: int = 10) -> List[StandardNode]:
        """FAISS dense (semantic) search over all nodes."""
        _, indices = self.node_index.search(query_vector.astype('float32'), k)
        return [self.nodes[idx] for idx in indices[0] if idx < len(self.nodes)]

    def search_lexical(self, query: str, k: int = 10) -> List[StandardNode]:
        """BM25 lexical search over all nodes."""
        # Lowercase to match the corpus vocabulary (built with content.lower().split()
        # in indexers). Without this, capitalized query tokens miss every idf key and
        # contribute 0 to BM25 — silently zeroing proper-noun signal.
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:k]
        return [self.nodes[i] for i in top_n]

    def get_neighbors(self, node_id: str) -> List[StandardNode]:
        """Graph neighbor lookup via PyG edge_index."""
        if node_id in self.node_id_to_idx:
            idx = self.node_id_to_idx[node_id]
            edges = self.graph.edge_index
            neighbors_idx = edges[1][edges[0] == idx].tolist()
            return [self.nodes[i] for i in neighbors_idx if i < len(self.nodes)]
        return []

    # ═══════════════════════════════════════════════════════════════════
    # Partition-Level Search (Strategy 3 — Level 1)
    # ═══════════════════════════════════════════════════════════════════

    def search_centroids(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if self.centroid_index is None:
            log.warning("Centroid index not loaded. Returning empty.")
            return []

        if isinstance(query_vector, str):
            raise TypeError("search_centroids expected an embedding vector, got raw string.")

        qv = np.asarray(query_vector, dtype=np.float32)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)

        distances, indices = self.centroid_index.search(qv, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.centroid_pids):
                pid = int(self.centroid_pids[idx])
                results.append((pid, float(distances[0][i])))
        return results

    def get_partition_nodes(self, partition_id: int) -> List[StandardNode]:
        """Return all nodes belonging to a given partition."""
        node_indices = self._partition_to_nodes.get(partition_id, [])
        return [self.nodes[i] for i in node_indices if i < len(self.nodes)]

    # ═══════════════════════════════════════════════════════════════════
    # ColBERT Late-Interaction Search
    # ═══════════════════════════════════════════════════════════════════

    def search_colbert(self, query: str, k: int = 5) -> List[StandardNode]:
        """
        ColBERT search via Ragatouille. Falls back to lexical if unavailable.
        Returns top-K nodes ranked by late-interaction score.
        """
        if self.colbert is not None:
            try:
                results = self.colbert.search(query, k=k)
                matched_nodes = []
                for r in results:
                    content = r.get("content", "")
                    for node in self.nodes:
                        if node.content.startswith(content[:100]):
                            matched_nodes.append(node)
                            break
                return matched_nodes[:k]
            except Exception as e:
                log.warning(f"ColBERT search failed: {e}. Falling back to lexical.")
        return self.search_lexical(query, k=k)

    def search_colbert_centroid(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        ColBERT search over partition centroids. Returns top-K (partition_id, score).

        This method is ColBERT-only and does not fall back to FAISS.
        """
        if getattr(self, "colbert_centroid", None) is None:
            raise RuntimeError(
                "colbert_centroid is not initialized. "
                "ColBERT centroid search cannot run for this engine."
            )

        max_k = len(self.centroid_pids) if hasattr(self, "centroid_pids") else 0
        if max_k == 0 and hasattr(self, "partition_map") and self.partition_map:
            max_k = max(self.partition_map.values()) + 1

        safe_k = min(max_k, max(k, 1)) if max_k > 0 else max(k, 1)

        try:
            results = self.colbert_centroid.search(query, k=safe_k)
        except Exception as e:
            raise RuntimeError(f"ColBERT centroid search failed: {e}") from e

        matched_pids = []
        for r in results:
            pid_str = r.get("document_id", "")
            if pid_str.startswith("centroid_"):
                matched_pids.append((int(pid_str.split("_")[1]), float(r.get("score", 0.0))))

        return matched_pids[:k]

    # ═══════════════════════════════════════════════════════════════════
    # Level 2 — Cross-Encoder Reranking
    # ═══════════════════════════════════════════════════════════════════

    _cross_encoder = None  # Class-level lazy singleton

    @classmethod
    def _get_cross_encoder(cls):
        """Lazy-load the Cross-Encoder model on first use."""
        if cls._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder as CE
                cls._cross_encoder = CE(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    max_length=512,
                )
                log.info("Cross-Encoder loaded: cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                log.error(f"Failed to load Cross-Encoder: {e}")
                raise
        return cls._cross_encoder

    def rerank_cross_encoder(
        self, query: str, nodes: List[StandardNode], top_k: int = 10
    ) -> List[Tuple[StandardNode, float]]:
        """
        Score each (query, node.content) pair with a Cross-Encoder and
        return the top-K nodes sorted by descending relevance score.
        """
        if not nodes:
            return []

        ce = self._get_cross_encoder()
        pairs = [(query, n.content) for n in nodes]
        scores = ce.predict(pairs, show_progress_bar=False)

        scored = list(zip(nodes, scores.tolist()))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ═══════════════════════════════════════════════════════════════════
    # Bidirectional Vector ↔ Graph Conversion
    # ═══════════════════════════════════════════════════════════════════

    def vector_to_graph_search(self, query_vector: np.ndarray, k_entry: int = 3,
                                max_hops: int = 2) -> List[StandardNode]:
        """
        Vector-to-Graph: dense search → entry points → BFS expansion.
        """
        entry_nodes = self.search_dense(query_vector, k=k_entry)
        if not entry_nodes:
            return []

        visited_ids = set(n.node_id for n in entry_nodes)
        result_nodes = list(entry_nodes)
        queue = deque([(n.node_id, 0) for n in entry_nodes])

        while queue:
            current_id, current_hop = queue.popleft()
            if current_hop >= max_hops:
                continue
            neighbors = self.get_neighbors(current_id)
            for nbr in neighbors:
                if nbr.node_id not in visited_ids:
                    visited_ids.add(nbr.node_id)
                    result_nodes.append(nbr)
                    queue.append((nbr.node_id, current_hop + 1))

        return result_nodes

    def graph_to_vector_embedding(self, node_ids: List[str]) -> Optional[np.ndarray]:
        """
        Graph-to-Vector: collect node embeddings → mean pool → single vector.
        """
        valid_indices = [self.node_id_to_idx[nid]
                         for nid in node_ids if nid in self.node_id_to_idx]
        if not valid_indices:
            return None

        vectors = []
        for idx in valid_indices:
            try:
                vec = self.node_index.reconstruct(int(idx))
                vectors.append(vec)
            except Exception:
                continue

        if not vectors:
            return None

        stacked = np.stack(vectors)
        centroid = np.mean(stacked, axis=0)
        return centroid
