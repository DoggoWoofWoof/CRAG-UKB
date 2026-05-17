import os
import json
import networkx as nx
import numpy as np
from typing import List, Dict
import torch

try:
    import metis
except ImportError:
    metis = None

from src.pipeline.standardizer import StandardNode, load_nodes

class GraphPartitioner:
    def __init__(self, nodes: List[StandardNode]):
        self.nodes = nodes
        self.node_id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
        
    def build_networkx_graph(self) -> nx.Graph:
        G = nx.Graph()
        for i, node in enumerate(self.nodes):
            G.add_node(i)
            for neighbor in node.neighbors:
                if neighbor in self.node_id_to_idx:
                    G.add_edge(i, self.node_id_to_idx[neighbor])
        return G

    def partition(self, n_partitions: int = None) -> List[int]:
        G = self.build_networkx_graph()
        if n_partitions is None:
            # Target ~200 nodes per partition
            n_partitions = max(1, len(self.nodes) // 200)
            
        print(f"Partitioning graph into {n_partitions} communities...")
        
        if metis:
            try:
                # metis.partition returns (cutting_edge_count, partition_list)
                _, parts = metis.partition(G, n_partitions)
                return parts
            except Exception as e:
                print(f"METIS partitioning failed: {e}. Falling back to greedy.")
        
        # Fallback to greedy partitioning or Louvain if metis fails
        return self._greedy_partition(G, n_partitions)

    def _greedy_partition(self, G: nx.Graph, n_partitions: int) -> List[int]:
        # Simple greedy Partitioning using connected components / node degree
        # For a truly robust fallback, community detection could be used.
        # But here we just return a simple range-based if graph is sparse or other.
        # This is a place-holder for a robust fallback.
        print("Using naive fallback partitioning.")
        parts = [0] * len(self.nodes)
        partition_size = len(self.nodes) // n_partitions
        for i in range(len(self.nodes)):
            parts[i] = min(i // partition_size, n_partitions - 1)
        return parts

def run_partitioning(nodes_path: str, output_path: str):
    nodes = load_nodes(nodes_path)
    partitioner = GraphPartitioner(nodes)
    parts = partitioner.partition()
    
    # Save partition map
    partition_map = {nodes[i].node_id: int(parts[i]) for i in range(len(nodes))}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(partition_map, f, indent=2)
    print(f"Partition map saved to {output_path}")

if __name__ == "__main__":
    node_file = "data/processed/master_nodes.json"
    part_file = "data/ukb_storage/partition_map.json"
    if os.path.exists(node_file):
        run_partitioning(node_file, part_file)
    else:
        print(f"Nodes file {node_file} not found.")
