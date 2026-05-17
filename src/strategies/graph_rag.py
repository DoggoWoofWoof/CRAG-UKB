import time
from typing import List, Dict, Set

from .base import BaseRetriever, RetrievalResult
from src.core.engine import CoreEngine
from src.core.llm_manager import LLMManager
from src.core.encoders import DenseEncoder

class GraphRAG(BaseRetriever):
    def __init__(self, engine: CoreEngine, llm: LLMManager, encoder: DenseEncoder):
        super().__init__(engine)
        self.llm = llm
        self.encoder = encoder

    def retrieve(self, query: str, hops: int = 2) -> RetrievalResult:
        start_time = time.time()
        
        # 1. Find Entry Point (Seed Node)
        query_vector = self.encoder.encode([query])
        entry_nodes = self.engine.search_dense(query_vector, k=1)
        
        if not entry_nodes:
            return RetrievalResult(
                query=query,
                nodes=[],
                answer="No entry node found.",
                latency=0.0,
                metadata={"strategy": "GraphRAG"}
            )
            
        seed_node = entry_nodes[0]
        
        # 2. Static Multi-Hop BFS Traversal
        traversed_nodes = self._bfs(seed_node.node_id, hops=hops)
        
        # 3. Generate Answer
        context = self._format_context(traversed_nodes)
        truncated_context = self.llm.truncate_context(context)
        
        prompt = f"Context:\n{truncated_context}\n\nQuestion: {query}\nAnswer:"
        answer = self.llm.generate(prompt)
        
        latency = time.time() - start_time
        return RetrievalResult(
            query=query,
            nodes=traversed_nodes,
            answer=answer,
            latency=latency,
            metadata={"strategy": "GraphRAG (Static BFS)", "seed_node": seed_node.node_id}
        )

    def _bfs(self, start_node_id: str, hops: int) -> List:
        visited = {start_node_id}
        queue = [(start_node_id, 0)]
        results = []
        
        # Map node_id to node object (we might need a smarter way if engine nodes are large)
        node_map = {n.node_id: n for n in self.engine.nodes}
        
        if start_node_id in node_map:
            results.append(node_map[start_node_id])
        
        idx = 0
        while idx < len(queue):
            curr_id, curr_hop = queue[idx]
            idx += 1
            
            if curr_hop < hops:
                neighbors = self.engine.get_neighbors(curr_id)
                for neighbor in neighbors:
                    if neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        queue.append((neighbor.node_id, curr_hop + 1))
                        results.append(neighbor)
                        
        return results
        
# Note: For large graphs, node_map inside BFS should be replaced by engine lookups.
