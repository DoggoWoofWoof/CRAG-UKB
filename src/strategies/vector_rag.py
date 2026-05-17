import numpy as np
import time
from typing import List, Dict

from .base import BaseRetriever, RetrievalResult
from src.core.engine import CoreEngine
from src.core.llm_manager import LLMManager
from src.core.encoders import DenseEncoder

class VectorRAG(BaseRetriever):
    def __init__(self, engine: CoreEngine, llm: LLMManager, encoder: DenseEncoder):
        super().__init__(engine)
        self.llm = llm
        self.encoder = encoder

    def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        start_time = time.time()
        
        # 1. Fetch Dense Results
        query_vector = self.encoder.encode([query])
        dense_results = self.engine.search_dense(query_vector, k=k*2)
        
        # 2. Fetch Lexical Results
        lexical_results = self.engine.search_lexical(query, k=k*2)
        
        # 3. Reciprocal Rank Fusion (RRF)
        combined_nodes = self._rrf(dense_results, lexical_results, k=k)
        
        # 4. Generate Answer
        context = self._format_context(combined_nodes)
        truncated_context = self.llm.truncate_context(context)
        
        prompt = f"Context:\n{truncated_context}\n\nQuestion: {query}\nAnswer:"
        answer = self.llm.generate(prompt)
        
        latency = time.time() - start_time
        return RetrievalResult(
            query=query,
            nodes=combined_nodes,
            answer=answer,
            latency=latency,
            metadata={"strategy": "VectorRAG (Hybrid)"}
        )

    def _rrf(self, dense: List, lexical: List, k: int, c: int = 60) -> List:
        scores = {}
        for rank, node in enumerate(dense):
            scores[node.node_id] = scores.get(node.node_id, 0) + 1.0 / (c + rank)
        for rank, node in enumerate(lexical):
            scores[node.node_id] = scores.get(node.node_id, 0) + 1.0 / (c + rank)
            
        # Map node_id back to node object
        node_map = {node.node_id: node for node in dense + lexical}
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]
        
        return [node_map[nid] for nid in sorted_ids]
