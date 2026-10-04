"""Partition-quality only A/B/C using exact historical benchmark_partition_selection metrics K=[1,3,5,10,20,50,100,200] on full validation population"""
import os, json, random
import numpy as np
from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.evaluation.benchmark_partition_selection import _get_split_queries, compute_multi_gt_metrics, K_VALUES, COVERAGE_K_VALUES

# Use full K set for partition quality: 1,3,5,10,20,50,100,200
KS = [1,3,5,10,20,50,100,200]

def run_variant(dataset, variant):
    # Load engine for existing A, or build for B/C via ablation helper
    # For A, use existing partition_map/centroids/graph
    # For B/C, build via ablation's build_graph_and_partitions but reuse same vote logic via benchmark's faiss_vote_100
    # To keep exact historical vote logic, we will call benchmark's faiss_vote_100 path directly via engine.search_dense
    # So we need an engine with the variant's partition_map
    if variant == "A":
        engine = CoreEngine(source=dataset)
    else:
        # Build B/C via ablation helper to get new partition_map in scratchpad
        from scratchpad.ablation.l1_routing_graph_ablation import build_graph_and_partitions, load_embeddings_faiss, load_master
        nodes, doc_nodes, q_nodes, doc_id_to_idx = load_master(dataset)
        embeddings, _ = load_embeddings_faiss(dataset)
        out_dir = f"scratchpad/ablation/{dataset}/variant_{variant}"
        # Build if not exists
        if not os.path.exists(os.path.join(out_dir, "partition_map.json")):
            print(f"Building {variant} for {dataset}")
            G_nx, parts, n_cuts, pmap, centroids, pids, *_ = build_graph_and_partitions(dataset, variant, embeddings, doc_nodes, doc_id_to_idx, target=100, out_dir=out_dir)
        else:
            import json as j
            pmap = j.load(open(os.path.join(out_dir, "partition_map.json")))
        # Create a mock engine with variant partition_map but same nodes/index
        # We can load original engine and override partition_map
        engine = CoreEngine(source=dataset)
        engine.partition_map = {k: int(v) for k,v in pmap.items()}
        # Keep original graph and centroids for voting? For vote, only partition_map matters, not centroids
    encoder = DenseEncoder()
    splits = _get_split_queries(engine, dataset=dataset)
    # Use validation split (official dev or held-out val) for graph selection, not test
    # _get_split_queries returns train/val/test 70/20/10 deterministic
    # Use val for validation
    val_queries = splits["val"]
    if not val_queries:
        print(f"No val queries for {dataset}, using train held-out")
        val_queries = splits["train"][:2000]
    print(f"{dataset} {variant}: val {len(val_queries)} queries, n_parts {len(set(engine.partition_map.values()))}")
    # Precompute query vectors
    q_nodes = [q for q,_ in val_queries]
    q_texts = [q.content for q in q_nodes]
    qvecs = encoder.encode(q_texts).astype('float32')
    import faiss
    faiss.normalize_L2(qvecs)
    # For each query, run faiss_vote_100 logic exactly as benchmark:245
    n_parts = len(set(int(p) for p in engine.partition_map.values()))
    k_full = max(max(KS), n_parts, max(COVERAGE_K_VALUES))
    results = {f"recall@{k}": [] for k in KS}
    results.update({f"full_coverage@{k}": [] for k in KS})
    # Also need gt_recall etc but primary is recall and full_coverage
    for i, (q_node, gt_pids) in enumerate(val_queries):
        qvec = qvecs[i:i+1]
        dense_nodes = engine.search_dense(qvec, k=100)
        vote_counts = {}
        for node in dense_nodes:
            pid = engine.partition_map.get(node.node_id)
            if pid is not None:
                vote_counts[int(pid)] = vote_counts.get(int(pid), 0) + 1
        sorted_pids = sorted(vote_counts.keys(), key=lambda p: vote_counts[p], reverse=True)
        retrieved = sorted_pids[:k_full]
        # Pad with remaining pids if needed for full ranking? Benchmark pads with remaining sorted by id after vote, but for recall@K up to 200 and n_parts up to 658, the remaining not voted are considered miss (sentinel n_parts+1). For faiss_vote, retrieved is only voted partitions, not full n_parts ranking. But benchmark's k_full ensures retrieved is full vote ranking (only voted partitions, then remaining not ranked are considered miss via sentinel). compute_multi_gt_metrics handles miss_sentinel = n_parts+1 when n_parts provided.
        metrics = compute_multi_gt_metrics(retrieved, gt_pids, num_partitions=n_parts)
        for k in KS:
            results[f"recall@{k}"].append(metrics[f"recall@{k}"])
            results[f"full_coverage@{k}"].append(metrics[f"full_coverage@{k}"])
    # Aggregate
    agg = {k: round(float(np.mean(results[f"recall@{k}"]))*100,2) for k in KS}
    agg_cov = {k: round(float(np.mean(results[f"full_coverage@{k}"]))*100,2) for k in KS}
    print(f"{dataset} {variant} recall:", agg)
    print(f"{dataset} {variant} full_cov:", agg_cov)
    return agg, agg_cov

for ds in ["squad_clean","2wiki_clean","musique_clean"]:
    for var in ["A","B","C"]:
        # Skip B/C for KB? For these text datasets, all variants available
        # Check NER availability for B/C
        if var in ["B","C"]:
            import os as _os
            if not _os.path.exists(f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl"):
                print(f"{ds} {var} N/A no NER")
                continue
        run_variant(ds, var)

# Also for metaqa/webqsp where B/C N/A, just A
for ds in ["metaqa","webqsp"]:
    run_variant(ds, "A")
