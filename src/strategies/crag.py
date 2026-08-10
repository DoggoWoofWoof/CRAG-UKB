"""
CRAG (Cognitive Retrieval-Augmented Generation) — Strategy 3
=============================================================
Three-level partitioning architecture:
    Level 1: Partition Selection (FAISS centroid / ColBERT centroid / MLP)
    Level 2: Partition Entry — intra-partition re-ranking (FAISS / ColBERT)
    Level 3: deterministic Think-Act-Observe traversal (score-based expand/prune)

The same class produces 6 benchmark combinations (3 selectors × 2 rerankers).
After benchmarking Levels 1&2, the best combo is selected and the full CRAG system
(with Level 3 deterministic traversal) is benchmarked against VectorRAG and GraphRAG.
"""

import logging
import heapq
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Set

from .base import BaseRetriever, RetrievalResult

log = logging.getLogger(__name__)


class CRAG(BaseRetriever):
    """
    CRAG partition-based retrieval with deterministic traversal.
    
    Modes (Level 1 — partition selection):
        "faiss_centroid"   → BERT query embed → FAISS L2 on centroids.index
        "colbert_centroid" → ColBERT search on centroid text index
        "mlp"              → BERT embed → MLP project → FAISS search

    Rerankers (Level 2 — intra-partition entry):
        "cross_encoder" → HuggingFace Cross-Encoder pair scoring (RECOMMENDED)
        "faiss"         → Batched dense cosine similarity within partitions
        "colbert"       → [DEPRECATED] Global ColBERT search + partition filter

    Level 3 — deterministic traversal is always active after Level 2.
    """

    def __init__(self, engine, llm, encoder,
                 mode: str = "faiss_centroid",
                 reranker: str = "faiss",
                 top_k_partitions: int = 3,
                 top_k_entry: int = 10,
                 max_traverse_steps: int = 20,
                 score_threshold: float = 0.3,
                 expand_threshold: Optional[float] = None,
                 max_context_nodes: int = 10,
                 beam_width: int = 50,
                 expand_top_neighbors: int = 8,
                 min_context_nodes: int = 3,
                 dynamic_partition_expansion: bool = True,
                 max_dynamic_partitions: int = 3,
                 partition_admission_threshold: float = 0.35,
                 partition_fetch_k: int = 5,
                 l2_score_weight: float = 0.25,
                 partition_prior_weight: float = 0.15,
                 path_coherence_weight: float = 0.10,
                 redundancy_penalty_weight: float = 0.10,
                 depth_penalty_weight: float = 0.03,
                 partition_balance_weight: float = 0.04,
                 exclude_synthetic_edges: bool = True,
                 mlp_encoder=None):
        """
        Args:
            engine: CoreEngine instance
            llm: LLMManager instance
            encoder: DenseEncoder for query embedding
            mode: Partition selection method
            reranker: Intra-partition ranking method
            top_k_partitions: Number of partitions to select
            top_k_entry: Number of initial candidate nodes from partition
            max_traverse_steps: Max traversal steps
            score_threshold: Minimum relevance score to keep a node
            expand_threshold: Minimum score required before expanding neighbors
            max_context_nodes: Maximum nodes retained for generation context
            beam_width: Maximum pending frontier nodes kept after each expansion
            expand_top_neighbors: Highest-scoring neighbors to enqueue per expanded node
            min_context_nodes: Minimum context fallback from Level 2 seeds
            dynamic_partition_expansion: Whether cross-partition graph edges can admit new partitions
            max_dynamic_partitions: Max partitions admitted beyond Level 1's initial set
            partition_admission_threshold: Minimum admission score for a new partition
            partition_fetch_k: Nodes queued from each newly admitted partition
            *_weight: Path-aware scoring weights used during traversal/context composition
            mlp_encoder: Optional trained TextPartitionMLP for mode="mlp"
        """
        self.engine = engine
        self.llm = llm
        self.encoder = encoder
        self.mode = mode
        self.reranker = reranker
        self.top_k_partitions = top_k_partitions
        self.top_k_entry = top_k_entry
        self.max_traverse_steps = max_traverse_steps
        self.score_threshold = score_threshold
        self.expand_threshold = expand_threshold if expand_threshold is not None else score_threshold
        self.max_context_nodes = max_context_nodes
        self.beam_width = beam_width
        self.expand_top_neighbors = expand_top_neighbors
        self.min_context_nodes = min_context_nodes
        self.dynamic_partition_expansion = dynamic_partition_expansion
        self.max_dynamic_partitions = max_dynamic_partitions
        self.partition_admission_threshold = partition_admission_threshold
        self.partition_fetch_k = partition_fetch_k
        self.l2_score_weight = l2_score_weight
        self.partition_prior_weight = partition_prior_weight
        self.path_coherence_weight = path_coherence_weight
        self.redundancy_penalty_weight = redundancy_penalty_weight
        self.depth_penalty_weight = depth_penalty_weight
        self.partition_balance_weight = partition_balance_weight
        self.exclude_synthetic_edges = exclude_synthetic_edges
        self.mlp_encoder = mlp_encoder
        self._last_traversal_trace: Dict[str, Any] = {}
        self._node_vec_cache: Dict[str, np.ndarray] = {}
        self._partition_score_cache: Dict[int, float] = {}
        self._centroid_pid_to_pos: Optional[Dict[int, int]] = None

    def retrieve(self, query: str) -> RetrievalResult:
        """Full 3-level retrieval pipeline."""
        import time
        t0 = time.time()

        query_vector = self.encoder.encode([query])

        # ── Level 1: Select Partitions ──────────────────────────────
        partition_ids = self._select_partitions(query, query_vector)
        if not partition_ids:
            return RetrievalResult(
                query=query,
                nodes=[], answer="No relevant partitions found.",
                latency=time.time() - t0,
                metadata={"mode": self.mode, "reranker": self.reranker})

        # ── Level 2: Enter Partitions — get initial candidates ──────
        candidates = self._enter_partitions(query, query_vector, partition_ids)

        # ── Level 3: Deterministic Traversal — Think-Act-Observe ───
        curated_nodes = self._think_act_observe(
            query, query_vector, candidates, selected_partitions=partition_ids
        )

        # ── Generate Answer ─────────────────────────────────────────
        context = self._format_context(curated_nodes)
        prompt = (f"Context:\n{context}\n\nQuestion: {query}\n"
                  f"Answer based only on the context above:")
        answer = self.llm.generate(prompt)

        latency = time.time() - t0
        return RetrievalResult(
            query=query,
            nodes=curated_nodes,
            answer=answer,
            latency=latency,
            metadata={
                "mode": self.mode,
                "reranker": self.reranker,
                "partitions_selected": partition_ids,
                "candidates_after_entry": len(candidates),
                "nodes_after_traverse": len(curated_nodes),
                "traversal": self._last_traversal_trace,
                "latency_s": round(latency, 4),
            }
        )

    # ═══════════════════════════════════════════════════════════════
    # Level 1 — Partition Selection
    # ═══════════════════════════════════════════════════════════════

    def _select_partitions(self, query: str, query_vector: np.ndarray) -> List[int]:
        """Select top-K partitions based on the configured mode."""

        if self.mode == "faiss_centroid":
            # Direct FAISS search on partition centroids
            results = self.engine.search_centroids(query_vector, k=self.top_k_partitions)
            return [pid for pid, dist in results]

        elif self.mode == "colbert_centroid":
            # ColBERT search — match query text against centroid documents
            # Fall back to FAISS if ColBERT unavailable
            if self.engine.colbert is not None:
                try:
                    colbert_results = self.engine.colbert.search(query, k=self.top_k_partitions * 5)
                    # Map ColBERT results back to partition IDs
                    seen_pids = set()
                    partition_ids = []
                    for r in colbert_results:
                        content = r.get("content", "")
                        # Find which partition this node belongs to
                        for node in self.engine.nodes:
                            if node.content.startswith(content[:100]):
                                pid = self.engine.partition_map.get(node.node_id)
                                if pid is not None and pid not in seen_pids:
                                    seen_pids.add(pid)
                                    partition_ids.append(int(pid))
                                    if len(partition_ids) >= self.top_k_partitions:
                                        break
                        if len(partition_ids) >= self.top_k_partitions:
                            break
                    return partition_ids
                except Exception as e:
                    log.warning(f"ColBERT partition selection failed: {e}")
            # Fallback
            results = self.engine.search_centroids(query_vector, k=self.top_k_partitions)
            return [pid for pid, dist in results]

        elif self.mode == "mlp":
            # MLP-projected query → FAISS centroid search
            if self.mlp_encoder is not None:
                import torch
                with torch.no_grad():
                    qv_tensor = torch.tensor(query_vector, dtype=torch.float32)
                    projected = self.mlp_encoder(qv_tensor).numpy()
                results = self.engine.search_centroids(projected, k=self.top_k_partitions)
                return [pid for pid, dist in results]
            else:
                log.warning("MLP encoder not loaded, falling back to FAISS centroid.")
                results = self.engine.search_centroids(query_vector, k=self.top_k_partitions)
                return [pid for pid, dist in results]

        else:
            raise ValueError(f"Unknown partition selection mode: {self.mode}")

    # ═══════════════════════════════════════════════════════════════
    # Level 2 — Partition Entry (Intra-partition re-ranking)
    # ═══════════════════════════════════════════════════════════════

    def _enter_partitions(self, query: str, query_vector: np.ndarray,
                          partition_ids: List[int]):
        """Get initial candidate nodes from selected partitions.
        
        Reranker modes:
            "faiss"          — Batched dense cosine similarity (optimized)
            "cross_encoder"  — HuggingFace Cross-Encoder pair scoring (recommended)
            "colbert"        — [DEPRECATED] Global ColBERT search + partition filter
        """
        # Step 1: Pool all document nodes from the selected partitions
        pool = []
        for pid in partition_ids:
            partition_nodes = self.engine.get_partition_nodes(pid)
            pool.extend(partition_nodes)

        if not pool:
            return []

        if self.reranker == "cross_encoder":
            # ── Cross-Encoder: score (query, node.content) pairs ────────
            scored = self.engine.rerank_cross_encoder(
                query, pool, top_k=self.top_k_entry * self.top_k_partitions
            )
            return scored

        elif self.reranker == "faiss":
            # ── Optimized FAISS: batched reconstruction + vectorized dot ─
            indices = []
            valid_nodes = []
            for node in pool:
                idx = self.engine.node_id_to_idx.get(node.node_id)
                if idx is not None:
                    indices.append(int(idx))
                    valid_nodes.append(node)

            if not indices:
                return []

            # Batch reconstruct all vectors at once
            node_vecs = np.stack(
                [self.engine.node_index.reconstruct(i) for i in indices]
            )
            qv = query_vector.flatten()
            qv_norm = np.linalg.norm(qv) + 1e-8
            norms = np.linalg.norm(node_vecs, axis=1) + 1e-8
            scores = np.dot(node_vecs, qv) / (norms * qv_norm)

            # Sort descending, take top-K
            top_count = min(
                self.top_k_entry * self.top_k_partitions, len(scores)
            )
            top_idx = np.argsort(-scores)[:top_count]
            return [(valid_nodes[i], float(scores[i])) for i in top_idx]

        elif self.reranker == "colbert":
            # ── [DEPRECATED] Global ColBERT + partition intersection ─────
            log.warning(
                "ColBERT reranker is DEPRECATED for Level 2. "
                "Use reranker='cross_encoder' for proper partition-scoped scoring."
            )
            all_candidates = []
            colbert_results = self.engine.search_colbert(
                query, k=self.top_k_entry * 2
            )
            pool_nids = {n.node_id for n in pool}
            for node in colbert_results:
                if node.node_id in pool_nids:
                    all_candidates.append((node, 1.0))
            return all_candidates[: self.top_k_entry * self.top_k_partitions]

        else:
            raise ValueError(f"Unknown reranker: {self.reranker}")

    # ═══════════════════════════════════════════════════════════════
    # Level 3 — Deterministic Think-Act-Observe Traversal
    # ═══════════════════════════════════════════════════════════════

    def traverse_candidates(self, query: str, query_vector: np.ndarray,
                            seed_candidates,
                            selected_partitions: Optional[List[int]] = None) -> list:
        """Run Level 3 traversal over pre-ranked Level 2 candidates."""
        return self._think_act_observe(
            query, query_vector, seed_candidates, selected_partitions=selected_partitions
        )

    def _think_act_observe(self, query: str, query_vector: np.ndarray,
                           seed_candidates,
                           selected_partitions: Optional[List[int]] = None) -> list:
        """
        Deterministic Think-Act-Observe traversal:
        
        THINK:   Score each node's relevance to query
        ACT:     If score ≥ threshold → SELECT node, expand neighbors
                 If score < threshold → PRUNE node, skip branch
        OBSERVE: Track frontier, pruning, synthetic-edge skips, and context size
        
        This is a priority/beam graph search. It is deliberately deterministic
        so Level 3 can be benchmarked and ablated without hidden LLM decisions.
        """
        selected_records = []
        selected_ids = set()
        visited = set()
        seed_records = []
        frontier = []
        push_count = 0
        initial_partitions: Set[int] = set(int(pid) for pid in (selected_partitions or []))
        admitted_partitions: Set[int] = set(initial_partitions)
        dynamic_partitions: Set[int] = set()
        partition_selection_counts: Dict[int, int] = {}
        normalized_seeds = self._normalize_seed_candidates(seed_candidates)

        for rank, (node, raw_l2_score, l2_score) in enumerate(normalized_seeds):
            if node is None:
                continue

            partition_id = self._partition_id(node)
            if partition_id is not None:
                initial_partitions.add(partition_id)
                admitted_partitions.add(partition_id)

            score_parts = self._score_candidate(
                query_vector=query_vector,
                node=node,
                depth=0,
                source="seed",
                l2_score=l2_score,
                parent_node=None,
                selected_records=[],
                partition_selection_counts=partition_selection_counts,
            )
            seed_records.append((node, float(score_parts["combined_score"]), l2_score))
            heapq.heappush(
                frontier,
                (
                    -float(score_parts["combined_score"]),
                    push_count,
                    node,
                    0,
                    "seed",
                    None,
                    l2_score,
                    raw_l2_score,
                    rank,
                ),
            )
            push_count += 1

        stats = {
            "seed_candidates": len(seed_records),
            "selected_count": 0,
            "visited_count": 0,
            "expanded_count": 0,
            "pruned_count": 0,
            "queued_neighbors": 0,
            "synthetic_edges_skipped": 0,
            "fallback_nodes_added": 0,
            "cross_partition_edges_seen": 0,
            "dynamic_partitions_admitted": 0,
            "dynamic_partitions_rejected": 0,
            "partition_fetch_nodes_queued": 0,
            "max_frontier_size": len(frontier),
            "max_depth": 0,
            "steps": 0,
            "score_threshold": self.score_threshold,
            "expand_threshold": self.expand_threshold,
            "partition_admission_threshold": self.partition_admission_threshold,
            "max_context_nodes": self.max_context_nodes,
            "beam_width": self.beam_width,
            "expand_top_neighbors": self.expand_top_neighbors,
            "dynamic_partition_expansion": self.dynamic_partition_expansion,
            "max_dynamic_partitions": self.max_dynamic_partitions,
            "partition_fetch_k": self.partition_fetch_k,
            "exclude_synthetic_edges": self.exclude_synthetic_edges,
            "initial_partitions": sorted(initial_partitions),
            "admitted_partitions": [],
            "selected": [],
        }

        while frontier and stats["steps"] < self.max_traverse_steps:
            _, _, node, depth, source, parent_node, l2_score, raw_l2_score, seed_rank = heapq.heappop(frontier)
            if node.node_id in visited:
                continue
            visited.add(node.node_id)
            stats["visited_count"] += 1
            stats["steps"] += 1
            stats["max_depth"] = max(stats["max_depth"], depth)

            partition_id = self._partition_id(node)
            parent_partition = self._partition_id(parent_node) if parent_node is not None else None
            if (
                partition_id is not None
                and parent_partition is not None
                and partition_id != parent_partition
            ):
                stats["cross_partition_edges_seen"] += 1

            if partition_id is not None and partition_id not in admitted_partitions:
                if self._should_admit_partition(
                    query_vector=query_vector,
                    node=node,
                    partition_id=partition_id,
                    parent_node=parent_node,
                    depth=depth,
                    dynamic_partitions=dynamic_partitions,
                ):
                    admitted_partitions.add(partition_id)
                    dynamic_partitions.add(partition_id)
                    stats["dynamic_partitions_admitted"] += 1
                    push_count, queued_count = self._queue_partition_slice(
                        query_vector=query_vector,
                        partition_id=partition_id,
                        frontier=frontier,
                        push_count=push_count,
                        parent_node=node,
                        depth=depth,
                        visited=visited,
                        selected_ids=selected_ids,
                    )
                    stats["partition_fetch_nodes_queued"] += queued_count
                else:
                    stats["dynamic_partitions_rejected"] += 1
                    continue

            score_parts = self._score_candidate(
                query_vector=query_vector,
                node=node,
                depth=depth,
                source=source,
                l2_score=l2_score,
                parent_node=parent_node,
                selected_records=selected_records,
                partition_selection_counts=partition_selection_counts,
            )
            score = float(score_parts["combined_score"])

            # ── ACT: Select or Prune ────────────────────────────────
            if score >= self.score_threshold:
                if node.node_id not in selected_ids:
                    selected_ids.add(node.node_id)
                    if partition_id is not None:
                        partition_selection_counts[partition_id] = (
                            partition_selection_counts.get(partition_id, 0) + 1
                        )
                    selected_records.append({
                        "node": node,
                        **score_parts,
                        "depth": depth,
                        "source": source,
                        "partition_id": partition_id,
                        "parent_id": parent_node.node_id if parent_node is not None else None,
                        "raw_l2_score": raw_l2_score,
                        "seed_rank": seed_rank,
                    })

                if score >= self.expand_threshold:
                    # Expand neighbors — explore graph structure
                    neighbors = self.engine.get_neighbors(node.node_id)
                    synthetic_set = set(node.metadata.get("synthetic_neighbors", []))
                    scored_neighbors = []

                    for nbr in neighbors:
                        # Prevent graph corruption by adhering strictly to topological boundaries
                        if self.exclude_synthetic_edges and nbr.node_id in synthetic_set:
                            stats["synthetic_edges_skipped"] += 1
                            continue
                        
                        if nbr.node_id not in visited and nbr.node_id not in selected_ids:
                            nbr_score = self._score_candidate(
                                query_vector=query_vector,
                                node=nbr,
                                depth=depth + 1,
                                source=node.node_id,
                                l2_score=0.0,
                                parent_node=node,
                                selected_records=selected_records,
                                partition_selection_counts=partition_selection_counts,
                            )["combined_score"]
                            scored_neighbors.append((nbr_score, nbr))

                    scored_neighbors.sort(key=lambda item: item[0], reverse=True)
                    for nbr_score, nbr in scored_neighbors[: self.expand_top_neighbors]:
                        # 9-tuple must match the heappop schema at the top of the
                        # loop: (-score, push_count, node, depth, source,
                        # parent_node, l2_score, raw_l2_score, seed_rank).
                        # Expanded neighbors carry the parent's id as source, the
                        # parent node object, no L2 seed score, and no seed rank.
                        heapq.heappush(
                            frontier,
                            (
                                -float(nbr_score),
                                push_count,
                                nbr,
                                depth + 1,
                                node.node_id,
                                node,
                                0.0,
                                None,
                                None,
                            ),
                        )
                        push_count += 1
                        stats["queued_neighbors"] += 1

                    stats["expanded_count"] += 1
            else:
                stats["pruned_count"] += 1

            if len(frontier) > self.beam_width:
                frontier = heapq.nsmallest(self.beam_width, frontier)
                heapq.heapify(frontier)

            stats["max_frontier_size"] = max(stats["max_frontier_size"], len(frontier))

        if len(selected_records) < self.min_context_nodes:
            for node, score, l2_score in sorted(seed_records, key=lambda item: item[1], reverse=True):
                if len(selected_records) >= min(self.min_context_nodes, self.max_context_nodes):
                    break
                if node.node_id in selected_ids:
                    continue
                selected_ids.add(node.node_id)
                partition_id = self._partition_id(node)
                selected_records.append({
                    "node": node,
                    "combined_score": float(score),
                    "node_score": self._score_node(query_vector, node),
                    "l2_score": l2_score,
                    "partition_score": self._partition_score(query_vector, partition_id),
                    "path_coherence": 0.0,
                    "redundancy_penalty": 0.0,
                    "depth_penalty": 0.0,
                    "partition_balance_penalty": 0.0,
                    "depth": 0,
                    "source": "fallback_seed",
                    "partition_id": partition_id,
                    "parent_id": None,
                    "raw_l2_score": None,
                    "seed_rank": None,
                })
                stats["fallback_nodes_added"] += 1

        context_records = self._compose_context_records(query_vector, selected_records)
        selected = [record["node"] for record in context_records]
        stats["selected_count"] = len(selected)
        stats["admitted_partitions"] = sorted(admitted_partitions)
        stats["dynamic_partitions"] = sorted(dynamic_partitions)
        stats["selected"] = [
            {
                "node_id": record["node"].node_id,
                "score": round(float(record["combined_score"]), 4),
                "node_score": round(float(record.get("node_score", 0.0)), 4),
                "l2_score": round(float(record.get("l2_score", 0.0)), 4),
                "partition_score": round(float(record.get("partition_score", 0.0)), 4),
                "path_coherence": round(float(record.get("path_coherence", 0.0)), 4),
                "depth": int(record["depth"]),
                "source": record["source"],
                "partition_id": record.get("partition_id"),
                "parent_id": record.get("parent_id"),
            }
            for record in context_records[: min(len(context_records), 20)]
        ]
        self._last_traversal_trace = stats

        log.debug(f"Level 3 traversal: {len(selected)} nodes selected "
                  f"from {len(visited)} visited in {stats['steps']} steps")
        return selected

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    def _score_node(self, query_vector: np.ndarray, node) -> float:
        """Cosine similarity between query vector and node's FAISS embedding."""
        try:
            node_vec = self._node_vector(node)
            if node_vec is None:
                return 0.0
            qv = query_vector.flatten()
            score = float(np.dot(qv, node_vec) /
                          (np.linalg.norm(qv) * np.linalg.norm(node_vec) + 1e-8))
            return score
        except Exception:
            return 0.0

    def _node_vector(self, node) -> Optional[np.ndarray]:
        if node is None:
            return None
        cached = self._node_vec_cache.get(node.node_id)
        if cached is not None:
            return cached
        idx = self.engine.node_id_to_idx.get(node.node_id)
        if idx is None:
            return None
        try:
            vec = self.engine.node_index.reconstruct(int(idx))
            self._node_vec_cache[node.node_id] = vec
            return vec
        except Exception:
            return None

    def _node_similarity(self, left_node, right_node) -> float:
        left = self._node_vector(left_node)
        right = self._node_vector(right_node)
        if left is None or right is None:
            return 0.0
        return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-8))

    def _partition_id(self, node) -> Optional[int]:
        if node is None:
            return None
        partition_id = self.engine.partition_map.get(node.node_id)
        return int(partition_id) if partition_id is not None else None

    def _partition_score(self, query_vector: np.ndarray, partition_id: Optional[int]) -> float:
        if partition_id is None or getattr(self.engine, "centroid_index", None) is None:
            return 0.0
        if partition_id in self._partition_score_cache:
            return self._partition_score_cache[partition_id]
        try:
            if self._centroid_pid_to_pos is None:
                self._centroid_pid_to_pos = {
                    int(pid): idx for idx, pid in enumerate(self.engine.centroid_pids)
                }
            pos = self._centroid_pid_to_pos.get(int(partition_id))
            if pos is None:
                return 0.0
            centroid = self.engine.centroid_index.reconstruct(int(pos))
            qv = query_vector.flatten()
            score = float(np.dot(qv, centroid) / (np.linalg.norm(qv) * np.linalg.norm(centroid) + 1e-8))
            self._partition_score_cache[partition_id] = score
            return score
        except Exception:
            return 0.0

    def _normalize_seed_candidates(self, seed_candidates) -> List[Tuple[Any, Optional[float], float]]:
        parsed = []
        for candidate in seed_candidates:
            node, score = self._unpack_candidate(candidate)
            if node is None:
                continue
            parsed.append((node, score))

        numeric_scores = [
            float(score) for _, score in parsed
            if isinstance(score, (int, float, np.floating)) and np.isfinite(float(score))
        ]
        if not parsed:
            return []
        if not numeric_scores:
            denom = max(1, len(parsed) - 1)
            return [
                (node, score, 1.0 - (rank / denom))
                for rank, (node, score) in enumerate(parsed)
            ]

        min_score = min(numeric_scores)
        max_score = max(numeric_scores)
        denom = max_score - min_score
        normalized = []
        rank_denom = max(1, len(parsed) - 1)
        for rank, (node, score) in enumerate(parsed):
            if isinstance(score, (int, float, np.floating)) and np.isfinite(float(score)):
                norm_score = (float(score) - min_score) / denom if denom > 1e-8 else 1.0
            else:
                norm_score = 1.0 - (rank / rank_denom)
            normalized.append((node, score, float(norm_score)))
        return normalized

    def _score_candidate(
        self,
        query_vector: np.ndarray,
        node,
        depth: int,
        source: str,
        l2_score: float,
        parent_node,
        selected_records: List[Dict[str, Any]],
        partition_selection_counts: Dict[int, int],
    ) -> Dict[str, float]:
        partition_id = self._partition_id(node)
        node_score = self._score_node(query_vector, node)
        partition_score = self._partition_score(query_vector, partition_id)
        path_coherence = self._node_similarity(parent_node, node) if parent_node is not None else 0.0
        redundancy_penalty = self._max_selected_similarity(node, selected_records)
        depth_penalty = float(depth)
        partition_balance_penalty = float(partition_selection_counts.get(partition_id, 0))

        combined = (
            node_score
            + self.l2_score_weight * float(l2_score or 0.0)
            + self.partition_prior_weight * partition_score
            + self.path_coherence_weight * path_coherence
            - self.redundancy_penalty_weight * redundancy_penalty
            - self.depth_penalty_weight * depth_penalty
            - self.partition_balance_weight * partition_balance_penalty
        )
        return {
            "combined_score": float(combined),
            "node_score": float(node_score),
            "l2_score": float(l2_score or 0.0),
            "partition_score": float(partition_score),
            "path_coherence": float(path_coherence),
            "redundancy_penalty": float(redundancy_penalty),
            "depth_penalty": float(depth_penalty),
            "partition_balance_penalty": float(partition_balance_penalty),
        }

    def _max_selected_similarity(self, node, selected_records: List[Dict[str, Any]]) -> float:
        if not selected_records:
            return 0.0
        return max(
            self._node_similarity(node, record["node"])
            for record in selected_records[: self.max_context_nodes]
        )

    def _should_admit_partition(
        self,
        query_vector: np.ndarray,
        node,
        partition_id: int,
        parent_node,
        depth: int,
        dynamic_partitions: Set[int],
    ) -> bool:
        if not self.dynamic_partition_expansion:
            return False
        if len(dynamic_partitions) >= self.max_dynamic_partitions:
            return False
        node_score = self._score_node(query_vector, node)
        partition_score = self._partition_score(query_vector, partition_id)
        path_score = self._node_similarity(parent_node, node) if parent_node is not None else 0.0
        admission_score = (
            0.50 * node_score
            + 0.35 * partition_score
            + 0.15 * path_score
            - self.depth_penalty_weight * float(depth)
        )
        return admission_score >= self.partition_admission_threshold

    def _queue_partition_slice(
        self,
        query_vector: np.ndarray,
        partition_id: int,
        frontier: List[Tuple],
        push_count: int,
        parent_node,
        depth: int,
        visited: Set[str],
        selected_ids: Set[str],
    ) -> Tuple[int, int]:
        scored_nodes = []
        for node in self.engine.get_partition_nodes(partition_id):
            if node.node_id in visited or node.node_id in selected_ids:
                continue
            score = self._score_candidate(
                query_vector=query_vector,
                node=node,
                depth=depth + 1,
                source=f"partition_{partition_id}",
                l2_score=0.0,
                parent_node=parent_node,
                selected_records=[],
                partition_selection_counts={},
            )["combined_score"]
            scored_nodes.append((score, node))
        scored_nodes.sort(key=lambda item: item[0], reverse=True)
        queued_count = 0
        for score, node in scored_nodes[: self.partition_fetch_k]:
            heapq.heappush(
                frontier,
                (-float(score), push_count, node, depth + 1, f"partition_{partition_id}", parent_node, 0.0, None, None),
            )
            push_count += 1
            queued_count += 1
        return push_count, queued_count

    def _compose_context_records(
        self, query_vector: np.ndarray, selected_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        remaining = sorted(
            selected_records,
            key=lambda item: item.get("combined_score", 0.0),
            reverse=True,
        )
        chosen = []
        chosen_ids = set()
        while remaining and len(chosen) < self.max_context_nodes:
            best_idx = 0
            best_score = float("-inf")
            for idx, record in enumerate(remaining):
                if record["node"].node_id in chosen_ids:
                    continue
                redundancy = max(
                    [self._node_similarity(record["node"], chosen_record["node"]) for chosen_record in chosen]
                    or [0.0]
                )
                partition_reuse = sum(
                    1 for chosen_record in chosen
                    if chosen_record.get("partition_id") == record.get("partition_id")
                )
                composed_score = (
                    record.get("combined_score", 0.0)
                    - self.redundancy_penalty_weight * redundancy
                    - self.partition_balance_weight * partition_reuse
                )
                if composed_score > best_score:
                    best_score = composed_score
                    best_idx = idx

            record = remaining.pop(best_idx)
            chosen_ids.add(record["node"].node_id)
            chosen.append(record)
        return chosen

    def _unpack_candidate(self, candidate):
        """Accept either `(node, score)` tuples or raw node objects."""
        if isinstance(candidate, tuple) and candidate:
            node = candidate[0]
            score = candidate[1] if len(candidate) > 1 else None
            return node, score
        return candidate, None

    def _format_context(self, nodes: list) -> str:
        """Format selected nodes into a context string for the LLM."""
        if not nodes:
            return "No relevant context found."
        parts = []
        for i, node in enumerate(nodes):
            title = node.metadata.get("title", "")
            header = f"[{i+1}] {title}" if title else f"[{i+1}]"
            parts.append(f"{header}\n{node.content}")
        context = "\n\n".join(parts)

        # Truncate to token budget if LLM has truncation
        if hasattr(self.llm, 'truncate_context'):
            context = self.llm.truncate_context(context)
        return context
