"""Canonical Qwen topology build: C = A_QWEN ∪ NER, B = STRUCT∪NER (missing only). Zero Qwen kNN recompute."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json, pickle, hashlib, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import networkx as nx
from src.pipeline.standardizer import load_nodes

log = logging.getLogger("build_canonical")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TARGET = 100

def load_doc_nodes(dataset):
    mpath = f"data/processed/master_nodes_{dataset}.json"
    if not os.path.exists(mpath):
        mpath = "data/processed/master_nodes.json"
    all_nodes = load_nodes(mpath)
    # filter to source if aggregated file
    srcs = set(n.metadata.get("source","") for n in all_nodes)
    if len(srcs)>1:
        all_nodes = [n for n in all_nodes if n.metadata.get("source")==dataset]
    doc_nodes = [n for n in all_nodes if n.metadata.get("type")!="question"]
    doc_id_to_idx = {n.node_id:i for i,n in enumerate(doc_nodes)}
    return doc_nodes, doc_id_to_idx

def structural_adj(doc_nodes, doc_id_to_idx):
    N=len(doc_nodes)
    adj=[set() for _ in range(N)]
    for i, nd in enumerate(doc_nodes):
        for nb in nd.neighbors:
            j=doc_id_to_idx.get(nb)
            if j is not None and j!=i:
                adj[i].add(j)
                adj[j].add(i)
    return adj

def ner_adj(dataset, doc_id_to_idx):
    pkl = f"data/ukb_storage/{dataset}/ner_edges_w_df25.pkl"
    N=len(doc_id_to_idx)
    adj=[set() for _ in range(N)]
    if not os.path.exists(pkl):
        log.warning(f"NER missing {dataset} {pkl}")
        return adj, 0
    A = pickle.load(open(pkl,"rb"))
    if A.shape[0]!=N:
        log.warning(f"NER shape mismatch {dataset} {A.shape} vs N={N}")
        return adj,0
    coo=A.tocoo()
    cnt=0
    for r,c in zip(coo.row, coo.col):
        if r==c: continue
        if r<c:
            cnt+=1
        adj[r].add(int(c))
    # ensure symmetric (already)
    return adj, cnt

def load_A_adj(dataset):
    gpath=f"data/ukb_storage/{dataset}/gte_qwen/graph.pt"
    A=torch.load(gpath, map_location="cpu", weights_only=False)
    ei=A.edge_index
    N=int(A.num_nodes)
    adj=[set() for _ in range(N)]
    src=ei[0].tolist(); dst=ei[1].tolist()
    for u,v in zip(src,dst):
        if u!=v:
            adj[u].add(int(v))
    # symmetrize already
    # ensure undirected dedup counts
    return adj, N, A

def union_adj(*adjs):
    N=len(adjs[0])
    out=[set(s) for s in adjs[0]]
    for adj in adjs[1:]:
        for i in range(N):
            out[i].update(adj[i])
    return out

def metis_partition(adj):
    import pymetis
    N=len(adj)
    n_parts=max(1, N//TARGET)
    adjacency_list=[list(nei) for nei in adj]
    n_cuts, membership = pymetis.part_graph(n_parts, adjacency=adjacency_list)
    return n_parts, n_cuts, membership

def save_topology(dataset, variant, adj, membership, n_parts, n_cuts):
    out_dir=f"scratchpad/ablation_qwen/{dataset}/variant_{variant}"
    os.makedirs(out_dir, exist_ok=True)
    # graph.pt
    N=len(adj)
    edge_src=[]; edge_dst=[]
    for i in range(N):
        for j in adj[i]:
            edge_src.append(i); edge_dst.append(j)
    # dedup will be stored directed (both ways already)
    from torch_geometric.data import Data
    edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long)
    data=Data(edge_index=edge_index, num_nodes=N)
    torch.save(data, os.path.join(out_dir,"graph.pt"))
    # partition_map
    doc_nodes,_ = load_doc_nodes(dataset)  # need node_ids order
    # doc_nodes order must match adj indexing: we used doc_nodes order earlier, so reuse
    # But to get node_ids, reload doc_nodes
    # Ensure membership length matches N
    part_map={doc_nodes[i].node_id: int(membership[i]) for i in range(N)}
    json.dump(part_map, open(os.path.join(out_dir,"partition_map.json"),"w"), indent=2)
    # stats
    counts=[0]*n_parts
    for p in membership: counts[p]+=1
    stats={
        "dataset":dataset, "variant":variant, "n_nodes":N, "n_parts":n_parts, "n_cuts":int(n_cuts),
        "edge_cut_ratio": float(n_cuts/(len(edge_src))) if edge_src else 0,
        "partition_counts":counts, "min_size":min(counts) if counts else 0, "max_size":max(counts) if counts else 0,
        "mean_size":float(np.mean(counts)) if counts else 0, "median_size":float(np.median(counts)) if counts else 0,
        "n_edges_undirected": len(edge_src)//2, "synthetic_qwen_edges": "reused from A" if variant=="C" else "none",
    }
    json.dump(stats, open(os.path.join(out_dir,"stats.json"),"w"), indent=2)
    # hash for provenance
    h=hashlib.sha256("".join(sorted(part_map.keys())).encode()).hexdigest()[:12]
    log.info(f"[{dataset}/{variant}] N={N} n_parts={n_parts} cuts={n_cuts} mean={stats['mean_size']:.1f} edges_und={stats['n_edges_undirected']} -> {out_dir}")
    return stats

def verify_A(dataset):
    gpath=f"data/ukb_storage/{dataset}/gte_qwen/graph.pt"
    npy=f"data/ukb_storage/{dataset}/gte_qwen/nodes.npy"
    pmap=f"data/ukb_storage/{dataset}/gte_qwen/partition_map.json"
    g=torch.load(gpath, map_location="cpu", weights_only=False)
    N=g.num_nodes
    E2=g.edge_index.shape[1]//2
    rows=np.load(npy, mmap_mode='r').shape[0]
    pm=len(json.load(open(pmap)))
    # synthetic count via engine? approximate via degree vs structural
    # Use load_doc_nodes to get structural count
    doc_nodes,_=load_doc_nodes(dataset)
    syn_est = E2 - sum(len(s) for s in structural_adj(doc_nodes, {n.node_id:i for i,n in enumerate(doc_nodes)}))//2
    log.info(f"[VERIFY A] {dataset:16s} N={N} rows={rows} pm={pm} Eund={E2} syn_est≈{syn_est} ok={N==rows==pm}")
    return {"dataset":dataset,"N":N,"rows":rows,"pm":pm,"Eund":E2,"syn_est":syn_est}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["metaqa"])
    p.add_argument("--verify_only", action="store_true")
    args=p.parse_args()
    if args.verify_only:
        for ds in args.datasets:
            verify_A(ds)
    else:
        # verify first
        for ds in args.datasets:
            verify_A(ds)
        # build logic per ds/variant as per plan
        for ds in args.datasets:
            # Load once per ds structural and NER
            doc_nodes, doc_id_to_idx = load_doc_nodes(ds)
            N=len(doc_nodes)
            struct = structural_adj(doc_nodes, doc_id_to_idx)
            ner, ner_cnt = ner_adj(ds, doc_id_to_idx)
            log.info(f"[{ds}] structural edges {sum(len(s) for s in struct)//2} ner edges {ner_cnt}")
            # For C: need A adj
            # Check if A exists
            A_adj, N2, _ = load_A_adj(ds)
            assert N==N2, f"N mismatch {N} vs {N2}"
            # Build B for missing only? But we build C for all, B for hotpot/metaqa/webqsp
            # Decide per dataset
            need_B = ds in ["hotpotqa_clean","metaqa","webqsp"]
            need_C = True
            if need_C:
                C_adj = union_adj(A_adj, ner)
                log.info(f"[{ds}/C] union A ({sum(len(s) for s in A_adj)//2}) + NER ({ner_cnt}) -> {sum(len(s) for s in C_adj)//2} undirected")
                n_parts, n_cuts, memb = metis_partition(C_adj)
                save_topology(ds, "C", C_adj, memb, n_parts, n_cuts)
            if need_B:
                B_adj = union_adj(struct, ner)
                log.info(f"[{ds}/B] union STRUCT ({sum(len(s) for s in struct)//2}) + NER -> {sum(len(s) for s in B_adj)//2}")
                n_parts, n_cuts, memb = metis_partition(B_adj)
                save_topology(ds, "B", B_adj, memb, n_parts, n_cuts)
            # For 2wiki etc, we could copy existing B to ablation_qwen for uniformity
            if ds in ["2wiki_clean","musique_clean","squad_clean"]:
                # copy B
                import shutil, pathlib
                src=f"scratchpad/ablation/{ds}/variant_B"
                dst=f"scratchpad/ablation_qwen/{ds}/variant_B"
                if os.path.exists(src) and not os.path.exists(dst):
                    log.info(f"Copy B {src} -> {dst}")
                    shutil.copytree(src, dst)
                # also copy A? Actually A is in gte_qwen, but for uniformity we can symlink or copy partition_map
                # Create A dir as reference to gte_qwen?
                dstA=f"scratchpad/ablation_qwen/{ds}/variant_A"
                if not os.path.exists(dstA):
                    os.makedirs(dstA, exist_ok=True)
                    # copy partition_map and graph for provenance
                    for fname in ["partition_map.json","graph.pt","centroids.index","centroid_pids.json"]:
                        srcp=f"data/ukb_storage/{ds}/gte_qwen/{fname}"
                        dstp=os.path.join(dstA, fname)
                        if os.path.exists(srcp) and not os.path.exists(dstp):
                            if fname.endswith(".json"):
                                import shutil as sh
                                sh.copy(srcp, dstp)
                            else:
                                # for binary, copy
                                import shutil as sh
                                sh.copy(srcp, dstp)
                    log.info(f"Created A ref {dstA}")
