#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G2 Track-A L1 geometry harness (MetaQA first). Parameter-free. VAL/dev only; TEST untouched.
This module: (1) loads frozen artifacts in the nodes.npy row space, (2) builds PROVENANCE-PURE
adjacency (structural from official kb.txt in name-space; kNN recomputed exact k=3), (3) reproduces
the frozen P50 scope + per-hop ANY/ALL coverage (harness-validation gate), (4) provides A1 residual /
A2 displacement / A5 graph-expansion primitives. NO gold, NO question-node neighbors, NO hop feature,
NO dataset id, NO test split used anywhere in inference."""
import json, os, re, time
import numpy as np

DS = "metaqa"
BASE = f"data/ukb_storage/{DS}/gte_qwen/"
K0 = 60; K_LOCK = 100; P_MAIN = 50   # frozen L1_LOCKED constants

# ---------------------------------------------------------------- loaders
def load_artifacts():
    j = json.load(open(BASE + "query_ids_all.json"))
    from src.pipeline.standardizer import load_nodes
    all_nodes = load_nodes(f"data/processed/master_nodes_{DS}.json")
    srcs = set(n.metadata.get("source", "") for n in all_nodes)
    if len(srcs) > 1:
        all_nodes = [n for n in all_nodes if n.metadata.get("source") == DS]
    doc_nodes = [n for n in all_nodes if n.metadata.get("type") != "question"]
    doc_id_to_idx = {n.node_id: i for i, n in enumerate(doc_nodes)}
    # frozen topology C
    pm = json.load(open(f"scratchpad/ablation_qwen/{DS}/variant_C/partition_map.json"))
    N = len(doc_nodes)
    hard = np.array([int(pm.get(doc_nodes[i].node_id, -1)) for i in range(N)], dtype=np.int64)
    npart = int(hard.max()) + 1
    # mem_idx = own + 1-hop neighbor partitions (FROZEN definition uses master_nodes.neighbors == mixed graph)
    mem_idx = []
    for nd in doc_nodes:
        s = {int(hard[doc_id_to_idx[nd.node_id]])}
        for nb in nd.neighbors:
            k = doc_id_to_idx.get(nb)
            if k is not None:
                s.add(int(hard[k]))
        mem_idx.append(sorted(s))
    return dict(j=j, doc_nodes=doc_nodes, doc_id_to_idx=doc_id_to_idx,
                hard=hard, npart=npart, mem_idx=mem_idx, N=N)

def load_embs():
    return np.load(BASE + "nodes.npy", mmap_mode="r"), np.load(BASE + "queries_all.npy", mmap_mode="r")

# ---------------------------------------------------------------- provenance-pure adjacency
def build_structural_adj(doc_id_to_idx):
    """Typed structural adjacency directly in the nodes.npy row space, from official kb.txt (name space).
    Returns adj[row] = list of (nbr_row, relation, direction) ; direction in {+1 forward, -1 reverse}."""
    adj = {}
    n_edges = 0; n_unmapped = 0
    def norm(name): return "metaqa_ent_" + name.strip().lower()
    for line in open("data/original/metaqa/kb.txt", encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) != 3:
            continue
        h, rel, t = parts
        hi = doc_id_to_idx.get(norm(h)); ti = doc_id_to_idx.get(norm(t))
        if hi is None or ti is None:
            n_unmapped += 1; continue
        adj.setdefault(hi, []).append((ti, rel, +1))
        adj.setdefault(ti, []).append((hi, rel, -1))   # reverse for traversal
        n_edges += 1
    return adj, dict(n_edges=n_edges, n_unmapped=n_unmapped, n_nodes_with_edges=len(adj))

def build_knn_adj(doc_id_to_idx, tsv="data/canonical/metaqa/graph_knn.tsv"):
    """kNN adjacency in name-space. graph_knn.tsv uses integer-suffixed ids (metaqa_ent_<int>) which are a
    DIFFERENT id space than the name ids; we can only use it if those ids resolve. If not, caller should
    recompute exact k=3 from nodes.npy (recompute_knn_adj)."""
    adj = {}; miss = 0; hit = 0
    for line in open(tsv, encoding="utf-8"):
        a, b, w = line.rstrip("\n").split("\t")
        ai = doc_id_to_idx.get(a); bi = doc_id_to_idx.get(b)
        if ai is None or bi is None:
            miss += 1; continue
        adj.setdefault(ai, []).append((bi, "knn", 0)); hit += 1
    return adj, dict(hit=hit, miss=miss)

def recompute_knn_adj(nodes, k=3, bs=2048):
    """Exact global k-NN (cosine, IndexFlatIP-faithful) in nodes.npy row space. Returns adj[row]=list of (nbr_row,'knn',0)."""
    import torch
    X = torch.tensor(np.ascontiguousarray(nodes), dtype=torch.float32)
    X = torch.nn.functional.normalize(X, dim=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(dev)
    N = X.shape[0]; adj = {}
    for s in range(0, N, bs):
        q = X[s:s+bs]
        sim = q @ X.T                       # (bs, N)
        top = torch.topk(sim, k + 1, dim=1).indices.cpu().numpy()  # +1 to drop self
        for i, row in enumerate(top):
            r = s + i
            nbrs = [int(v) for v in row if int(v) != r][:k]
            adj[r] = [(v, "knn", 0) for v in nbrs]
    return adj

# ---------------------------------------------------------------- seeds (inference-safe)
_BRK = re.compile(r"\[(.+?)\]")
def resolve_seeds(qtext, doc_id_to_idx):
    """MetaQA seed = bracketed entity in the question text (inference-safe). Returns list of rows."""
    rows = []
    for m in _BRK.findall(qtext):
        r = doc_id_to_idx.get("metaqa_ent_" + m.strip().lower())
        if r is not None:
            rows.append(r)
    return rows

# ---------------------------------------------------------------- frozen P50 scope
def partition_ranking(order_K, mem_idx, npart, K=K_LOCK):
    """Replicates l1_eval_phase1.partition_ranking for ONE router (order_K = (nq,K) doc rows)."""
    nq = len(order_K)
    S = np.zeros((nq, npart), np.float32); M = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        for r, nd in enumerate(order_K[qi][:K]):
            if nd < 0 or nd >= len(mem_idx):
                continue
            w = 1.0 / (K0 + r)
            for p in mem_idx[int(nd)]:
                S[qi, p] += w
                if w > M[qi, p]:
                    M[qi, p] = w
    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), np.int32); rows = np.arange(nq)[:, None]
        rank[rows, order] = np.arange(npart)[None, :]
        return 1.0 / (K0 + rank)
    votes = rr(S) + rr(M)
    return np.argsort(-votes, axis=1).astype(np.int32)

def fused_ranking(dense_K, splade_K, mem_idx, npart, K=K_LOCK):
    dr = partition_ranking(dense_K, mem_idx, npart, K)
    sr = partition_ranking(splade_K, mem_idx, npart, K)
    nq = dr.shape[0]; rows = np.arange(nq)[:, None]; cols = np.arange(npart)[None, :]
    pos_d = np.empty((nq, npart), np.int32); pos_d[rows, dr] = cols
    pos_s = np.empty((nq, npart), np.int32); pos_s[rows, sr] = cols
    fv = 1.0/(K0 + pos_d) + 1.0/(K0 + pos_s)
    fv_in = np.take_along_axis(fv, dr, axis=1)
    idx2 = np.argsort(-fv_in, axis=1, kind="stable")
    return np.take_along_axis(dr, idx2, axis=1).astype(np.int32)

def p50_scope_partitions(sample_rows, A, K=K_LOCK, P=P_MAIN):
    """Return, per sampled query, the set of selected top-P partitions (frozen dense+splade router)."""
    dense_all = np.load(BASE + "dense_top200_all.npy", mmap_mode="r")
    splade_all = np.load(BASE + "splade_top200_all.npy", mmap_mode="r")
    dK = np.stack([np.asarray(dense_all[i][:K]) for i in sample_rows]).astype(np.int64)
    sK = np.stack([np.asarray(splade_all[i][:K]) for i in sample_rows]).astype(np.int64)
    rank = fused_ranking(dK, sK, A["mem_idx"], A["npart"], K)
    return [set(int(x) for x in rank[qi][:P]) for qi in range(len(sample_rows))]
