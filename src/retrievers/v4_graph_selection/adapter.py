"""
V4GraphSelectionAdapter — BaseRetriever-compatible wrapper for the v4 pipeline.

This adapter bridges the v4-query-graph-selection architecture into the
PipelineFactory interface so it can be benchmarked identically to the other
four retrieval strategies.

The v4 pipeline internally uses:
  1. ColBERTPartitionRouter (Teleport: partition selection via MaxSim)
  2. GraphEngine.extract_subgraph (Stitch: subgraph assembly + BFS fallback)
  3. NeuroHybridRetrievalModule (Traverse: AdaptiveGating + NeuralSubgraphMatcher)
  4. HybridReranker (final cross-encoder reranking)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class V4RetrievalResult:
    """Result from the v4 agentic pipeline."""
    answer: str
    retrieved_nodes: List[Dict[str, Any]]
    reasoning_path: List[str]          # Ordered node IDs along the traversal
    partitions_selected: List[int]     # Partition IDs chosen by ColBERT router
    colbert_scores: List[float]        # MaxSim scores per partition
    latency_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class V4GraphSelectionAdapter:
    """
    Wraps the v4 NeuroHybridRetrievalModule as a BaseRetriever-compatible component.

    Independently testable:
        from src.retrievers.v4_graph_selection import build_v4_pipeline
        pipeline = build_v4_pipeline(config)
        result = pipeline.retrieve("your query here")

    PipelineFactory-compatible:
        Added as "v4_graph_selection" strategy key in factory.py:
            "v4_graph_selection": V4GraphSelectionAdapter(shared, config)
    """

    strategy_name = "v4_graph_selection"

    def __init__(self, shared_components, config: dict):
        """
        Args:
            shared_components: SharedComponents injected by PipelineFactory
                               (.graph_engine, .vector_store, .llm)
            config: Full unified.yaml config dict
        """
        self.shared = shared_components
        self.config = config
        self._pipeline = None   # Lazy-loaded on first retrieve() call
        self._llm = shared_components.llm

    def _build_internal_pipeline(self):
        """
        Lazy-builds the v4 NeuroHybridRetrievalModule using shared components.
        Called once on first retrieve() invocation to avoid startup cost.
        """
        from .pipeline import build_v4_pipeline_internal
        self._pipeline = build_v4_pipeline_internal(
            graph_engine=self.shared.graph_engine,
            vector_store=self.shared.vector_store,
            config=self.config,
        )
        logger.info("[V4GraphSelectionAdapter] Internal pipeline initialized.")

    def retrieve(self, query: str) -> V4RetrievalResult:
        """Main entry point. Called by PipelineFactory benchmark runner."""
        if self._pipeline is None:
            self._build_internal_pipeline()

        start = time.time()
        try:
            raw = self._pipeline.retrieve(query, top_k=self.config.get("retrieval", {}).get("colbert_final_k", 5))
        except Exception as e:
            logger.error(f"[V4] Retrieval failed for query '{query[:60]}': {e}")
            return V4RetrievalResult(
                answer=f"Error: {e}",
                retrieved_nodes=[],
                reasoning_path=[],
                partitions_selected=[],
                colbert_scores=[],
                latency_seconds=time.time() - start,
            )

        # Assemble answer via shared LLM (token-budget enforced)
        from src.common.utils import truncate_to_token_budget, serialize_node_to_passage
        nodes = truncate_to_token_budget(raw.nodes, budget=self.config.get("llm", {}).get("token_budget", 3000))
        context = "\n".join(serialize_node_to_passage(n) for n in nodes)
        prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer concisely:"
        answer = self._llm.generate(prompt)

        return V4RetrievalResult(
            answer=answer,
            retrieved_nodes=[n if isinstance(n, dict) else vars(n) for n in nodes],
            reasoning_path=getattr(raw, "reasoning_path", []),
            partitions_selected=getattr(raw, "partitions_selected", []),
            colbert_scores=getattr(raw, "colbert_scores", []),
            latency_seconds=time.time() - start,
            metadata={"strategy": self.strategy_name},
        )


def build_v4_pipeline(config: dict) -> V4GraphSelectionAdapter:
    """
    Standalone entry point for testing the v4 pipeline WITHOUT PipelineFactory.

    Example:
        from src.retrievers.v4_graph_selection import build_v4_pipeline
        import yaml

        config = yaml.safe_load(open("configs/unified.yaml"))
        pipeline = build_v4_pipeline(config)
        result = pipeline.retrieve("Who built GPT-4?")
        print(result.answer)
        print("Partitions selected:", result.partitions_selected)
        print("Reasoning path:", result.reasoning_path)
    """
    from src.common.graph_engine import GraphEngine
    from src.common.vector_store import UnifiedVectorStore
    from src.common.llm_client import build_llm_client

    class _StandaloneShared:
        def __init__(self, cfg):
            self.graph_engine = GraphEngine(cfg["kg_store_path"])
            self.vector_store = UnifiedVectorStore(cfg["kg_store_path"])
            self.llm = build_llm_client(cfg["llm"])

    shared = _StandaloneShared(config)
    return V4GraphSelectionAdapter(shared, config)
