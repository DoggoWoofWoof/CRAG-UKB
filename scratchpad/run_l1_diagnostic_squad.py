"""L1 diagnostic for squad_clean: partition recall + candidate recall + docs exposed, and direct global vs partition at matched budget"""
import json, os, collections, numpy as np, faiss
from src.core.engine import CoreEngine
from src.core.splade_scorer import SpladeScorer

src='squad_clean'
eng = CoreEngine(src, index_subdir='')  # loads partition_map, centroids, graph, nodes.index
# Load queries and gold
import pathlib
# Use benchmark_partition_selection logic simplified: load master_nodes for squad_clean
from src.pipeline.standardizer import load_nodes
nodes = load_nodes(f'data/processed/master_nodes_{src}.json')
# Filter questions
q_nodes = [n for n in nodes if n.metadata.get('type')=='question']
print(f'{src}: total nodes {len(nodes)}, questions {len(q_nodes)}')
# Build question -> gold partitions
pm = json.load(open(f'data/ukb_storage/{src}/partition_map.json'))
# Map node_id -> idx for gold lookup
# Use question.neighbors as gold doc ids
# For each question, collect gold partitions
gold_parts = []
for q in q_nodes:
    parts=set()
    for nid in q.neighbors:
        if nid in pm:
            parts.add(pm[nid])
    gold_parts.append(parts)

# Partition routing via dense vote
# Load query vectors for squad_clean? Use encoder directly for demo on subset
from src.core.encoders import DenseEncoder
enc = DenseEncoder()
# Sample 100 questions for quick diagnostic
import random
random.seed(42)
sample_idx = random.sample(range(len(q_nodes)), min(200, len(q_nodes)))
sample_q = [q_nodes[i] for i in sample_idx]
sample_gold = [gold_parts[i] for i in sample_idx]
print(f'sample {len(sample_q)} questions, avg gold parts {np.mean([len(g) for g in sample_gold]):.2f}')

# Dense vote
vote_k=100
import time
# Build FAISS node index via engine
# Use engine.search_dense for each query
def partition_recall_at_k(k):
    hits=0
    for q, gold in zip(sample_q, sample_gold):
        if not gold: continue
        qvec = enc.encode([q.content])
        faiss.normalize_L2(qvec)
        # engine search
        # Use engine's method: search_centroids for partition ranking vs search_dense for node vote
        # For historical count vote: search_dense k=vote_k
        # Simulate: get top vote_k nodes via faiss
        import faiss
        # Use nodes.index directly
        D,I = eng.index.search(qvec, vote_k)
        # Map indices to node_ids
        # Need idx->node_id mapping
        pass
