"""
Step 3: Multi-Modal Indexing — Unified Knowledge Base Factory
=============================================================
Builds ALL 6 views of the Unified Knowledge Base from master_nodes.json:

    1. FAISS Node Index        (Dense / Semantic)
    2. BM25 Inverted Index     (Lexical / Keyword)
    3. PyG Graph               (Structural / Topological)
    4. Partition Map           (Community detection via PyMETIS, chunking fallback)
    5. FAISS Centroid Index    (Hierarchical / Partition-level Dense)
    6. ColBERT Index           (Late-Interaction via Ragatouille)

Also provides HierarchicalIndexer for building centroid and ColBERT indices
from pre-partitioned nodes.

Run:
    python -m src.core.indexers
"""

import os
import sys
import json
import pickle
import logging

import faiss
import numpy as np
import torch
import networkx as nx
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from collections import defaultdict
from tqdm import tqdm

# ── project imports ─────────────────────────────────────────────────────────
from src.pipeline.standardizer import StandardNode, load_nodes
from src.core.encoders import DenseEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save(path: str, obj):
    """Pickle-save helper."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _json_save(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


# ═══════════════════════════════════════════════════════════════════════════
# 1. FAISS Node Index
# ═══════════════════════════════════════════════════════════════════════════

def build_faiss_node_index(nodes: List[StandardNode], encoder: DenseEncoder, out_dir: str):
    # Filter out questions to prevent Data Leakage!
    nodes = [n for n in nodes if n.metadata.get("type") != "question"]
    log.info(f"[1/6] Building FAISS Node Index  ({len(nodes)} document/entity nodes)…")

    texts = [n.content for n in nodes]
    embeddings = encoder.encode(texts)             # (N, D) float32
    # Ensure normalization for IndexFlatIP (Cosine Similarity)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    path = os.path.join(out_dir, "nodes.index")
    faiss.write_index(index, path)
    log.info(f"    ✔  Saved  {path}  ({index.ntotal} vectors, dim={dim})")
    return embeddings                              # reuse downstream


# ═══════════════════════════════════════════════════════════════════════════
# 2. BM25 Inverted Index
# ═══════════════════════════════════════════════════════════════════════════

def build_bm25_index(nodes: List[StandardNode], out_dir: str):
    nodes = [n for n in nodes if n.metadata.get("type") != "question"]
    log.info(f"[2/6] Building BM25 Index ({len(nodes)} nodes)…")

    tokenized_corpus = [n.content.lower().split() for n in nodes]
    bm25 = BM25Okapi(tokenized_corpus)

    path = os.path.join(out_dir, "bm25.pkl")
    _save(path, bm25)
    log.info(f"    ✔  Saved  {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PyG Graph  (and NetworkX copy for partitioning)
# ═══════════════════════════════════════════════════════════════════════════

def build_pyg_graph(nodes: List[StandardNode], embeddings: np.ndarray, out_dir: str) -> nx.Graph:
    nodes = [n for n in nodes if n.metadata.get("type") != "question"]
    log.info(f"[3/6] Building PyG Graph ({len(nodes)} nodes) with Dense KNN Fallback...")

    from torch_geometric.data import Data

    node_id_to_idx: Dict[str, int] = {n.node_id: i for i, n in enumerate(nodes)}

    # --- UNIVERSAL SEMANTIC KNN BRIDGING ---
    # Weave all nodes into a global semantic web to ensure no fragmented islands.
    from tqdm import tqdm
    log.info(f"    Weaving universal semantic edges (KNN, k=3) for all {len(nodes)} nodes...")
    dim = embeddings.shape[1]
    
    # Normalize for Cosine Similarity (Inner Product)
    vectors = embeddings.astype("float32")
    faiss.normalize_L2(vectors)
    
    # Search for Top-4 (to get self + 3 neighbors)
    # Using IndexFlatIP for exact search since we want quality semantic links
    knn_index = faiss.IndexFlatIP(dim)
    knn_index.add(vectors)
    
    distances, indices = knn_index.search(vectors, 4)
    
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(len(nodes)))

    synthetic_count = 0
    for i, node in enumerate(nodes):
        # indices[i][0] is usually self (dist=1.0)
        # We take the next 3
        for local_idx in indices[i][1:4]:
            if local_idx != -1:
                j = int(local_idx)
                if i != j:
                    nbr_id = nodes[j].node_id
                    # Add to StandardNode neighbors (for Graph)
                    if nbr_id not in node.neighbors:
                        node.neighbors.append(nbr_id)
                        # Mark as synthetic for agent-level pruning
                        if "synthetic_neighbors" not in node.metadata:
                            node.metadata["synthetic_neighbors"] = []
                        node.metadata["synthetic_neighbors"].append(nbr_id)
                    
                    # Add to NetworkX for METIS
                    if not G_nx.has_edge(i, j):
                        G_nx.add_edge(i, j)
                        synthetic_count = synthetic_count + 1

    log.info(f"    ✔  Added {synthetic_count} universal semantic edges.")

    edge_src, edge_dst = [], []
    for i, node in enumerate(nodes):
        for nbr_id in node.neighbors:
            j = node_id_to_idx.get(nbr_id)
            if j is not None and i != j:
                edge_src.append(i); edge_dst.append(j)
                edge_src.append(j); edge_dst.append(i)   # undirected mirror
                G_nx.add_edge(i, j)

    # --- KNN SEMANTIC BRIDGING FOR ISOLATED NODES ---
    isolates = list(nx.isolates(G_nx))
    if isolates:
        from tqdm import tqdm
        log.info(f"    Found {len(isolates)} isolated nodes. Weaving dense semantic edges (KNN)...")
        dim = embeddings.shape[1]
        
        # Optimize with IVFFlat for ~20x speedup over brute-force L2
        nlist = int(np.sqrt(len(embeddings)))  # standard FAISS clustering rule
        vectors = embeddings.astype("float32")
        faiss.normalize_L2(vectors)
        
        log.info(f"    Training FAISS IVFFlat quantizer with {nlist} clusters...")
        quantizer = faiss.IndexFlatIP(dim)
        knn_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        knn_index.train(vectors)
        knn_index.add(vectors)
        # Dynamic nprobe: search 10% of clusters for stable recall
        knn_index.nprobe = max(1, nlist // 10)
        
        knn_added = 0
        batch_size = 20000
        
        # Batch execute to avoid thread locking and provide visual progress
        for b_start in tqdm(range(0, len(isolates), batch_size), desc="Weaving Isolated Clusters"):
            b_end = min(b_start + batch_size, len(isolates))
            batch_isolates = isolates[b_start:b_end]
            
            iso_embeddings = embeddings[batch_isolates].astype("float32")
            faiss.normalize_L2(iso_embeddings)
            distances, indices = knn_index.search(iso_embeddings, 4) # Top 4 to guarantee 3 non-self 
            
            for idx_in_batch, original_i in enumerate(batch_isolates):
                node_internal = nodes[original_i]
                for j in indices[idx_in_batch]:
                    if j != original_i and j != -1: # FAISS returns -1 if missing
                        # Add structural edge
                        if not G_nx.has_edge(original_i, int(j)):
                            edge_src.append(original_i); edge_dst.append(int(j))
                            edge_src.append(int(j)); edge_dst.append(original_i)
                            G_nx.add_edge(original_i, int(j))
                            
                            # Update synthetic metadata ONLY (avoid base neighbors mutation)
                            if "synthetic_neighbors" not in node_internal.metadata:
                                node_internal.metadata["synthetic_neighbors"] = []
                            target_nid = nodes[j].node_id
                            if target_nid not in node_internal.metadata["synthetic_neighbors"]:
                                node_internal.metadata["synthetic_neighbors"].append(target_nid)
                            
                            if "synthetic_neighbors" not in nodes[j].metadata:
                                nodes[j].metadata["synthetic_neighbors"] = []
                            source_nid = node_internal.node_id
                            if source_nid not in nodes[j].metadata["synthetic_neighbors"]:
                                nodes[j].metadata["synthetic_neighbors"].append(source_nid)
                                
                            knn_added += 1
                            
        log.info(f"    ✔ Added {knn_added} synthetic edges. Graph isolated clusters bridged.")

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    data = Data(edge_index=edge_index, num_nodes=len(nodes))

    path = os.path.join(out_dir, "graph.pt")
    torch.save(data, path)
    log.info(f"    ✔  Saved  {path}  ({len(nodes)} nodes, {len(edge_src)//2} edges)")
    return G_nx


# ═══════════════════════════════════════════════════════════════════════════
# 4. Partition Map  (greedy modularity, target ~200 nodes/partition)
# ═══════════════════════════════════════════════════════════════════════════

def build_partition_map(nodes: List[StandardNode], G_nx: nx.Graph, out_dir: str) -> List[int]:
    nodes = [n for n in nodes if n.metadata.get("type") != "question"]
    log.info(f"[4/6] Building Partition Map ({len(nodes)} nodes)…")

    n_nodes = len(nodes)
    parts: List[int] = [0] * n_nodes

    if G_nx.number_of_edges() > 0:
        try:
            import pymetis
            log.info("    Using PyMETIS for graph partitioning (Target: ~1000 nodes/partition)…")
            n_parts = max(1, n_nodes // 1000)
            
            # Convert NetworkX to adjacency list for PyMETIS
            adjacency_list = [list(G_nx.neighbors(i)) for i in range(n_nodes)]
            
            n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency_list)
            for i, part_id in enumerate(membership):
                parts[i] = part_id
                
            log.info(f"    PyMETIS created {n_parts} partitions with {n_cuts} edge cuts.")
        except Exception as e:
            log.warning(f"    PyMETIS partitioning failed ({e}); using naive chunking.")
            target_size = 1000
            for i in range(n_nodes):
                parts[i] = i // target_size
    else:
        log.warning("    Graph has no edges — using naive partition (every 1000 nodes).")
        target_size = 1000
        for i in range(n_nodes):
            parts[i] = i // target_size

    # Build {node_id -> partition_id}
    part_map: Dict[str, int] = {nodes[i].node_id: parts[i] for i in range(n_nodes)}
    path = os.path.join(out_dir, "partition_map.json")
    _json_save(path, part_map)

    n_parts = int(max(parts)) + 1 if parts else 0
    counts = [parts.count(p) for p in range(n_parts)]
    if counts:
        import numpy as np
        log.info(f"    ✔ Partition Stats (Docs): Min={min(counts)}, Max={max(counts)}, Median={np.median(counts):.1f}")
    
    log.info(f"    ✔  Saved  {path}  ({n_parts} partitions)")
    return parts


# ═══════════════════════════════════════════════════════════════════════════
# 5. FAISS Centroid Index  (hierarchical, partition-level Dense)
# ═══════════════════════════════════════════════════════════════════════════

def build_faiss_centroid_index(
    nodes: List[StandardNode],
    parts: List[int],
    embeddings: np.ndarray,
    out_dir: str,
):
    nodes = [n for n in nodes if n.metadata.get("type") != "question"]
    log.info(f"[5/6] Building FAISS Centroid Index…")

    # Aggregate embeddings and structural combinations per partition
    partition_data: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    for i, pid in enumerate(parts):
        # Degree weighting (+1 smoothing to ensure isolated nodes contribute slightly)
        degree_weight = float(len(nodes[i].neighbors)) + 1.0
        if pid not in partition_data:
            partition_data[pid] = []
        partition_data[pid].append((embeddings[i], degree_weight))

    pids_sorted = sorted(partition_data.keys())
    
    # Calculate topology-weighted mean for each partition
    centroids_list = []
    for pid in pids_sorted:
        vecs = [item[0] for item in partition_data[pid]]
        weights = [item[1] for item in partition_data[pid]]
        centroids_list.append(np.average(vecs, axis=0, weights=weights))
        
    centroids = np.stack(centroids_list).astype("float32")
    faiss.normalize_L2(centroids)

    dim = centroids.shape[1]
    centroid_index = faiss.IndexFlatIP(dim)
    centroid_index.add(centroids)

    idx_path = os.path.join(out_dir, "centroids.index")
    faiss.write_index(centroid_index, idx_path)

    pid_path = os.path.join(out_dir, "centroid_pids.json")
    _json_save(pid_path, pids_sorted)

    log.info(f"    ✔  Saved  {idx_path}  ({len(pids_sorted)} centroids, dim={dim})")


# ═══════════════════════════════════════════════════════════════════════════
# 6. ColBERT Late-Interaction Index  (Ragatouille)
# ═══════════════════════════════════════════════════════════════════════════

def build_colbert_index(nodes: List[StandardNode], out_dir: str):
    nodes = [n for n in nodes if n.metadata.get("type") != "question"]
    log.info(f"[6/6] Building ColBERT Index via Ragatouille ({len(nodes)} nodes)…")

    try:
        from ragatouille import RAGPretrainedModel
    except Exception as e:
        log.warning(f"    ragatouille import failed — skipping ColBERT index. Error: {e}")
        import traceback
        log.warning(traceback.format_exc())
        return

    colbert_dir = os.path.join(out_dir, "colbert_ukb")
    os.makedirs(colbert_dir, exist_ok=True)

    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    # Use NodeID as document id so we can map back later
    doc_ids   = [n.node_id  for n in nodes]
    doc_texts = [n.content[:512] for n in nodes]   # max 512 chars per doc

    RAG.index(
        collection=doc_texts,
        document_ids=doc_ids,
        index_name="colbert_ukb",
        max_document_length=256,
        split_documents=True,
        use_faiss=True,
        overwrite_index=True,
    )

    # Also build a ColBERT Centroid index:
    # encode one representative text per partition, then index those
    log.info("    Building ColBERT Centroid partition view…")
    partition_map_path = os.path.join(out_dir, "partition_map.json")
    with open(partition_map_path) as f:
        part_map = json.load(f)

    # Option 1: Concatenate content of Top-3 Hubs (highest degree) per partition
    log.info("    Aggregating Top-3 Hubs per partition for ColBERT Centroids…")
    partition_to_nodes: Dict[int, List[StandardNode]] = defaultdict(list)
    for node in nodes:
        pid = part_map.get(node.node_id, 0)
        partition_to_nodes[pid].append(node)

    pid_to_repr: Dict[int, str] = {}
    for pid, part_nodes in partition_to_nodes.items():
        # Sort by degree (number of neighbors) descending
        sorted_nodes = sorted(part_nodes, key=lambda n: len(n.neighbors), reverse=True)
        # Take up to 3 nodes and concatenate content
        hubs = sorted_nodes[:3]
        combined_text = " ".join([n.content[:256] for n in hubs])
        pid_to_repr[pid] = combined_text[:512]

    cent_pids   = sorted(pid_to_repr.keys())
    cent_texts  = [pid_to_repr[p] for p in cent_pids]
    cent_ids    = [f"centroid_{p}" for p in cent_pids]
    import shutil
    
    colbert_default_dir = os.path.abspath(".ragatouille/colbert/indexes/colbert_ukb")
    if os.path.exists(colbert_default_dir):
        if os.path.exists(colbert_dir):
            shutil.rmtree(colbert_dir)
        shutil.move(colbert_default_dir, colbert_dir)

    RAG_cent = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RAG_cent.index(
        collection=cent_texts,
        document_ids=cent_ids,
        index_name="colbert_centroids",
        max_document_length=256,
        split_documents=False,
        use_faiss=True,
        overwrite_index=True,
    )

    colbert_cent_dir = os.path.join(out_dir, "colbert_centroids")
    colbert_cent_default = os.path.abspath(".ragatouille/colbert/indexes/colbert_centroids")
    if os.path.exists(colbert_cent_default):
        if os.path.exists(colbert_cent_dir):
            shutil.rmtree(colbert_cent_dir)
        shutil.move(colbert_cent_default, colbert_cent_dir)

    log.info(f"    ✔  ColBERT Node index + Centroid partition index complete.")

# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def build_all(
    master_nodes_path: str = "data/processed/master_nodes.json",
    out_dir: str           = "data/ukb_storage",
    skip_colbert: bool     = False,
    target_datasets: List[str] = None,
    force_rebuild: bool    = False,
):
    """Build separate index suites per dataset source.

    Each source (squad, musique, 2wiki, etc) gets its own subdirectory under
    ``out_dir`` containing nodes.index, bm25.pkl, graph.pt,
    partition_map.json, centroids.index, and colbert_ukb/.
    """
    os.makedirs(out_dir, exist_ok=True)

    log.info("═" * 60)
    log.info("  C-RAG  ·  Step 3: Per-Dataset UKB Indexing")
    log.info("═" * 60)

    # Load all nodes
    log.info(f"Loading nodes from  {master_nodes_path} …")
    all_nodes = load_nodes(master_nodes_path)
    log.info(f"  {len(all_nodes)} StandardNodes loaded.")

    # Group by source
    from collections import defaultdict
    source_groups: Dict[str, List[StandardNode]] = defaultdict(list)
    for n in all_nodes:
        src = n.metadata.get("source", "unknown")
        source_groups[src].append(n)

    sources = sorted(source_groups.keys())
    if target_datasets:
        sources = [s for s in sources if s in target_datasets]
        
    log.info(f"  Targeting {len(sources)} sources: {sources}")

    # Shared encoder
    encoder = DenseEncoder()

    for src_name in sources:
        nodes = source_groups[src_name]
        src_dir = os.path.join(out_dir, src_name)
        os.makedirs(src_dir, exist_ok=True)
        
        log.info("─" * 60)
        log.info(f"  Building indices for source: {src_name} ({len(nodes)} nodes)")
        log.info("─" * 60)

        embeddings = build_faiss_node_index(nodes, encoder, src_dir)
        build_bm25_index(nodes, src_dir)
        G_nx       = build_pyg_graph(nodes, embeddings, src_dir)
        parts      = build_partition_map(nodes, G_nx, src_dir)
        build_faiss_centroid_index(nodes, parts, embeddings, src_dir)

        if not skip_colbert:
            build_colbert_index(nodes, src_dir)
        else:
            log.info("[6/6] Skipping ColBERT index (skip_colbert=True).")

        log.info(f"  ✔  Source '{src_name}' complete → {os.path.abspath(src_dir)}")

    log.info("═" * 60)
    log.info("  ✔  All requested per-dataset UKB views processed!")
    log.info(f"  Output root: {os.path.abspath(out_dir)}")
    log.info("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# HierarchicalIndexer — Centroid + ColBERT from pre-partitioned nodes
# (merged from src/core/hierarchical_indexer.py)
# ═══════════════════════════════════════════════════════════════════════════

# HierarchicalIndexer was removed as it contained legacy unweighted L2 logic.


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the Unified Knowledge Base indices.")
    parser.add_argument("--nodes",         default="data/processed/master_nodes.json")
    parser.add_argument("--out",           default="data/ukb_storage")
    parser.add_argument("--skip-colbert",  action="store_true",
                        help="Skip the ColBERT index (faster, but no late-interaction view).")
    args = parser.parse_args()

    build_all(
        master_nodes_path=args.nodes,
        out_dir=args.out,
        skip_colbert=args.skip_colbert,
    )
