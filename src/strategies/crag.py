"""
CRAG (Cognitive Retrieval-Augmented Generation) — Strategy 3
=============================================================
Three-level partitioning architecture:
    Level 1: Partition Selection (FAISS centroid / ColBERT centroid / MLP)
    Level 2: Partition Entry — intra-partition re-ranking (FAISS / ColBERT)
    Level 3: Agentic Think-Act-Observe traversal (score-based expand/prune)

The same class produces 6 benchmark combinations (3 selectors × 2 rerankers).
After benchmarking Levels 1&2, the best combo is selected and the full CRAG system
(with Level 3 agentic traversal) is benchmarked against VectorRAG and GraphRAG.
"""

import logging
import numpy as np
from collections import deque
from typing import List, Optional, Tuple, Dict, Any

from .base import BaseRetriever, RetrievalResult

log = logging.getLogger(__name__)


class CRAG(BaseRetriever):
    """
    CRAG partition-based retrieval with agentic traversal.
    
    Modes (Level 1 — partition selection):
        "faiss_centroid"   → BERT query embed → FAISS L2 on centroids.index
        "colbert_centroid" → ColBERT search on centroid text index
        "mlp"              → BERT embed → MLP project → FAISS search

    Rerankers (Level 2 — intra-partition entry):
        "cross_encoder" → HuggingFace Cross-Encoder pair scoring (RECOMMENDED)
        "faiss"         → Batched dense cosine similarity within partitions
        "colbert"       → [DEPRECATED] Global ColBERT search + partition filter

    Level 3 — Agentic traversal is always active after Level 2.
    """

    def __init__(self, engine, llm, encoder,
                 mode: str = "faiss_centroid",
                 reranker: str = "faiss",
                 top_k_partitions: int = 3,
                 top_k_entry: int = 10,
                 max_traverse_steps: int = 20,
                 score_threshold: float = 0.3,
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
            max_traverse_steps: Max agentic traversal steps
            score_threshold: Minimum relevance score to keep a node
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
        self.exclude_synthetic_edges = exclude_synthetic_edges
        self.mlp_encoder = mlp_encoder

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

        # ── Level 3: Agentic Traversal — Think-Act-Observe ──────────
        curated_nodes = self._think_act_observe(query, query_vector, candidates)

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
    # Level 3 — Agentic Think-Act-Observe Traversal
    # ═══════════════════════════════════════════════════════════════

    def _think_act_observe(self, query: str, query_vector: np.ndarray,
                           seed_candidates) -> list:
        """
        ReAct-style agentic loop:
        
        THINK:   Score each node's relevance to query
        ACT:     If score ≥ threshold → SELECT node, expand neighbors
                 If score < threshold → PRUNE node, skip branch
        OBSERVE: Track score trajectory (increasing = keep, decreasing = backtrack)
        
        Combines vector similarity (THINK) with graph structure (ACT/expand).
        This is where CRAG fuses vector and graph RAG approaches.
        """
        selected = []
        visited = set()
        
        # Initialize frontier from Level 2 candidates
        frontier = deque()
        for node, score in seed_candidates:
            frontier.append((node, score))

        prev_score = 0.0
        expanding = True  # Whether we're in an expanding phase
        consecutive_drops = 0

        for step in range(self.max_traverse_steps):
            if not frontier:
                break

            node, entry_score = frontier.popleft()
            if node.node_id in visited:
                continue
            visited.add(node.node_id)

            # ── THINK: Score this node ──────────────────────────────
            score = self._score_node(query_vector, node)

            # ── OBSERVE: Check trajectory ───────────────────────────
            improving = score >= prev_score * 0.9  # 10% tolerance
            if score < prev_score:
                consecutive_drops += 1
            else:
                consecutive_drops = 0

            # If 3+ consecutive drops, stop expanding this branch
            if consecutive_drops >= 3:
                expanding = False

            # ── ACT: Select or Prune ────────────────────────────────
            if score >= self.score_threshold:
                selected.append(node)
                prev_score = score

                if expanding and improving:
                    # Expand neighbors — explore graph structure
                    neighbors = self.engine.get_neighbors(node.node_id)
                    synthetic_list = node.metadata.get("synthetic_neighbors", [])

                    for nbr in neighbors:
                        # Prevent graph corruption by adhering strictly to topological boundaries
                        if self.exclude_synthetic_edges and nbr.node_id in synthetic_list:
                            continue
                        
                        if nbr.node_id not in visited:
                            # Pre-score neighbor for frontier priority
                            nbr_score = self._score_node(query_vector, nbr)
                            frontier.append((nbr, nbr_score))
            else:
                # PRUNE: Don't expand this branch
                pass

        log.debug(f"Agentic traversal: {len(selected)} nodes selected "
                  f"from {len(visited)} visited in {step + 1} steps")
        return selected

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    def _score_node(self, query_vector: np.ndarray, node) -> float:
        """Cosine similarity between query vector and node's FAISS embedding."""
        idx = self.engine.node_id_to_idx.get(node.node_id)
        if idx is None:
            return 0.0
        try:
            node_vec = self.engine.node_index.reconstruct(int(idx))
            qv = query_vector.flatten()
            score = float(np.dot(qv, node_vec) /
                          (np.linalg.norm(qv) * np.linalg.norm(node_vec) + 1e-8))
            return score
        except Exception:
            return 0.0

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
