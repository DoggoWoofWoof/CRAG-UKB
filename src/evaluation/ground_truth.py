import random
import json
import torch
import yaml
from typing import List, Dict
from tqdm import tqdm

from src.pipeline.standardizer import load_nodes
from src.core.llm_manager import MockLLMManager

class GroundTruthGenerator:
    def __init__(self, master_nodes_path: str, graph_path: str, config_path: str = "configs/config.yaml"):
        self.nodes = load_nodes(master_nodes_path)
        self.node_id_to_idx = {n.node_id: i for i, n in enumerate(self.nodes)}
        self.graph = torch.load(graph_path, weights_only=False) # PyG Data object
        
        # Load LLM from config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        llm_model = config.get("retrieval", {}).get("models", {}).get("generator", "gpt-3.5-turbo")
        self.llm = MockLLMManager(model_name=llm_model)
        
    def generate_random_walk_benchmarks(self, n_queries: int = 50, hops: int = 3) -> List[Dict]:
        benchmarks = []
        edge_index = self.graph.edge_index
        
        # Build adjacency mapping for fast traversal
        adj = {}
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if u not in adj: adj[u] = []
            adj[u].append(v)
            
        print(f"Generating {n_queries} synthetic queries via LLM...")
        
        for _ in tqdm(range(n_queries)):
            # 1. Select a random start node
            curr_idx = random.choice(list(self.node_id_to_idx.values()))
            walk_idxs = [curr_idx]
            
            # 2. Perform Random Walk
            for _ in range(hops):
                neighbors = adj.get(curr_idx, [])
                if not neighbors: break
                curr_idx = random.choice(neighbors)
                if curr_idx not in walk_idxs: # avoid immediate loops
                    walk_idxs.append(curr_idx)
            
            # 3. Create Ground Truth Entry
            truth_nodes = [self.nodes[idx].node_id for idx in walk_idxs]
            truth_texts = [self.nodes[idx].content for idx in walk_idxs]
            
            # 4. LLM Question Synthesis
            # We prompt the LLM to write a question that requires the full hop logic to answer.
            context_string = "\n\n".join([f"Hop {i}: {text}" for i, text in enumerate(truth_texts)])
            prompt = f"Given the following sequential information steps, write a single complex question that requires connecting the first step to the final step to answer. Do not include the answer.\n\nContext:\n{context_string}"
            
            # Call the LLM
            query = self.llm.generate(prompt)
            
            benchmarks.append({
                "query": query,
                "truth_nodes": truth_nodes,
                "truth_texts": truth_texts
            })
            
        return benchmarks

if __name__ == "__main__":
    generator = GroundTruthGenerator("data/processed/master_nodes.json", "data/ukb_storage/graph.pt")
    benchmarks = generator.generate_random_walk_benchmarks()
    
    output_path = "data/processed/synthetic_benchmark.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmarks, f, indent=2)
    print(f"Benchmark saved to {output_path}")
