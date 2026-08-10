"""
Query Graph + GNN — Strategy 4: Experimental Architecture Skeleton
===================================================================
Converts a text query into a small query graph (entity extraction → nodes,
co-occurrence → edges), then uses a GIN encoder to produce graph-level
embeddings for matching against partition embeddings.

# EXPERIMENTAL — architecture skeleton only, no trained GNN weights.
# This inherits directly from CRAG to ensure the exact same Level 2 (Entry)
# and Level 3 deterministic traversal loops are used, swapping ONLY the
# Level 1 partition selection mechanism for fair interoperability benchmarking.
"""

import logging
import numpy as np
from typing import List, Optional

from .crag import CRAG

log = logging.getLogger(__name__)


class QueryGraphGNN(CRAG):
    """
    EXPERIMENTAL: Text → Entity Graph → GIN Encode → Partition Match → CRAG Traverse
    
    Architecture (Level 1 hook):
    1. Text Query → spaCy/regex entity extraction → query graph
    2. Query graph → GIN encoder → graph-level embedding
    3. Embedding → FAISS search against partition centroids
    
    Levels 2 & 3:
    Inherits identical CRAG FAISS/ColBERT entry and deterministic traversal execution.
    """

    def __init__(self, engine, llm, encoder, 
                 gin_checkpoint_path: str = None,
                 reranker: str = "faiss",
                 top_k_partitions: int = 3,
                 top_k_entry: int = 10,
                 max_traverse_steps: int = 20,
                 score_threshold: float = 0.3,
                 expand_threshold: float = None,
                 max_context_nodes: int = 10,
                 beam_width: int = 50,
                 expand_top_neighbors: int = 8,
                 min_context_nodes: int = 3,
                 **traversal_kwargs):
        
        # Initialize the underlying CRAG system, marking mode as 'gnn'
        super().__init__(engine, llm, encoder, 
                         mode="gnn", 
                         reranker=reranker,
                         top_k_partitions=top_k_partitions,
                         top_k_entry=top_k_entry,
                         max_traverse_steps=max_traverse_steps,
                         score_threshold=score_threshold,
                         expand_threshold=expand_threshold,
                         max_context_nodes=max_context_nodes,
                         beam_width=beam_width,
                         expand_top_neighbors=expand_top_neighbors,
                         min_context_nodes=min_context_nodes,
                         **traversal_kwargs)

        self.gin_encoder = None
        self.gin_checkpoint = gin_checkpoint_path

        if gin_checkpoint_path:
            self._load_gin_encoder(gin_checkpoint_path)

    def _load_gin_encoder(self, path: str):
        """Load trained GIN encoder from checkpoint."""
        try:
            import torch
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            log.info(f"GIN checkpoint loaded from {path}")
            log.warning("GIN encoder instantiation not yet implemented — using fallback.")
        except Exception as e:
            log.warning(f"Failed to load GIN encoder: {e}")

    def _select_partitions(self, query: str, query_vector: np.ndarray) -> List[int]:
        """
        OVERRIDE CRAG Level 1 Selector:
        Use GNN graph-to-graph matching instead of dense text-to-cluster matching.
        """
        # Step 1: Build query graph from text
        query_graph = self._build_query_graph(query)

        if query_graph is not None and self.gin_encoder is not None:
            # Step 2: GIN encode → graph embedding
            graph_embedding = self._gin_encode(query_graph)
            
            # Step 3: FAISS search against partition embeddings
            return self._match_partitions(graph_embedding)
        else:
            # Fallback: use identical FAISS centroid logic as CRAG
            log.info("GIN encoder not available, using FAISS centroid fallback for Level 1.")
            results = self.engine.search_centroids(query_vector, k=self.top_k_partitions)
            return [pid for pid, dist in results]

    def _build_query_graph(self, query: str):
        """
        Extract entities from query text and build a small query graph.
        
        Returns a PyG Data object or None if entity extraction fails.
        """
        try:
            import spacy
            import torch
            from torch_geometric.data import Data

            nlp = spacy.load("en_core_web_sm")
            doc = nlp(query)

            entities = []
            for ent in doc.ents:
                entities.append(ent.text)
            for chunk in doc.noun_chunks:
                if chunk.text not in entities:
                    entities.append(chunk.text)

            if len(entities) < 2:
                return None

            # Create node features using DenseEncoder
            node_features = self.encoder.encode(entities)
            x = torch.tensor(node_features, dtype=torch.float32)

            # Create fully-connected edges (small graph)
            n = len(entities)
            edges = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edges.append([i, j])
            edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty(2, 0, dtype=torch.long)

            return Data(x=x, edge_index=edge_index, num_nodes=n)

        except Exception as e:
            log.debug(f"Query graph construction failed: {e}")
            return None

    def _gin_encode(self, query_graph):
        """Run GIN encoder forward pass on query graph."""
        import torch
        batch = torch.zeros(query_graph.num_nodes, dtype=torch.long)
        with torch.no_grad():
            graph_emb, _ = self.gin_encoder(query_graph.x, query_graph.edge_index, batch)
        return graph_emb.numpy()

    def _match_partitions(self, graph_embedding: np.ndarray) -> List[int]:
        """FAISS search against partition centroids using graph embedding."""
        results = self.engine.search_centroids(graph_embedding, k=self.top_k_partitions)
        return [pid for pid, dist in results]
