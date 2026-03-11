"""
v4 internal pipeline builder.
Constructs the NeuroHybridRetrievalModule from SharedComponents.
This file is the single place where v4-specific component instantiation lives.
"""

from __future__ import annotations

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_v4_pipeline_internal(graph_engine, vector_store, config: dict):
    """
    Assembles the v4 NeuroHybridRetrievalModule from pre-loaded SharedComponents.
    
    Args:
        graph_engine: GraphEngine singleton from SharedComponents
        vector_store: UnifiedVectorStore singleton from SharedComponents  
        config: Full unified.yaml config dict

    Returns:
        NeuroHybridRetrievalModule — the complete v4 agentic pipeline
    """
    from .routing.colbert import ColBERTPartitionRouter
    from .model.gnn import NeuralSubgraphMatcher
    from .model.query_graph import QueryGraphGenerator
    from .model.cross_encoder import HybridReranker
    from .neural_hybrid import NeuroHybridRetrievalModule
    from src.common.llm_client import build_llm_client

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_colbert = config.get("colbert", {})
    cfg_gnn = config.get("gnn", {})
    cfg_llm = config.get("llm", {})

    # 1. ColBERT Partition Router (Teleport phase)
    colbert_router = ColBERTPartitionRouter(device=cfg_colbert.get("device", device))
    if hasattr(graph_engine.data, "part_id") and graph_engine.data.part_id is not None:
        colbert_router.build_partition_matrices(
            graph_engine,
            num_tokens_per_partition=config.get("partitioning", {}).get("tokens_per_partition", 32)
        )
    else:
        logger.warning("[V4] Graph has no partition IDs. ColBERT router will use centroid-only mode.")

    # 2. Neural Subgraph Matcher (GNN for Traverse scoring)
    neural_matcher = NeuralSubgraphMatcher(
        in_channels=cfg_gnn.get("in_channels", 768),
        hidden_channels=cfg_gnn.get("hidden_channels", 256),
        out_channels=cfg_gnn.get("out_channels", 256),
    )
    checkpoint_path = cfg_gnn.get("checkpoint_path")
    if checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            neural_matcher.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"[V4] Loaded GNN checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"[V4] Could not load GNN checkpoint: {e}. Using random weights.")

    # 3. Query Graph Generator (LLM-based query decomposition)
    llm_client = build_llm_client(cfg_llm)
    query_gen = QueryGraphGenerator(llm_client)

    # 4. Hybrid Reranker (ColBERT cross-encoder for final re-ranking)
    reranker = HybridReranker(device=device)

    # 5. Assemble NeuroHybridRetrievalModule
    pipeline = NeuroHybridRetrievalModule(
        vector_store=vector_store,
        graph_engine=graph_engine,
        query_gen=query_gen,
        neural_matcher=neural_matcher,
        colbert_router=colbert_router,
        reranker=reranker,
        use_adaptive_gating=config.get("retrieval", {}).get("use_adaptive_gating", True),
        device=device,
    )

    return pipeline
