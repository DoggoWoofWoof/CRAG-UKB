"""
L3 latency vs recall: is the graph search in the SOTA-RAG latency class?
========================================================================
Not sub-ms -- the target is "comparable to SOTA RAG", i.e. the retrieval/traversal
step should sit in the single-digit-to-low-hundreds-of-ms range that dense
retrieval and graph RAGs (HippoRAG query-time PPR) occupy (end-to-end is dominated
by the LLM reader anyway). This times the core per-query cost of each L3 candidate
and reports it next to gold recall, so "comparable to SOTA" is a measured claim.

Methods (seeds = dense top-N; graph = title UNION kNN, the fixed substrate):
  dense    : single ANN (q @ X), top-k                      (SOTA single-hop ref)
  ppr      : query-time personalized PageRank, 20 iters     (= HippoRAG class)
  beam     : bounded best-first from seeds, hard node cap    (your traversal, bounded)
Reports median / p95 per-query ms (single-threaded, warm) + gt_recall@100.

MUST run on a quiet machine -- timings are meaningless under CPU contention.
Writes results/research/l3_latency_{dataset}.json.
"""
import os
import json
import time
import logging
import argparse

import numpy as np
import scipy.sparse as sp
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership

log = logging.getLogger("experiments.l3_latency")


def _P(engine, n, id2idx):
    rows, cols = [], []
    docset = set(id2idx)
    for i, node in enumerate(engine.nodes):
        for nb in list(node.neighbors) + list(node.metadata.get("synthetic_neighbors", ())):
            if nb in docset and nb != node.node_id:
                rows.append(i); cols.append(id2idx[nb])
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).maximum(
        sp.csr_matrix((np.ones(len(rows), np.float32), (cols, rows)), shape=(n, n)))
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return (sp.diags(1.0 / d) @ A).tocsr(), A.tocsr()


def _recall_at(order_row, gold, k):
    if not gold:
        return None
    return len(set(gold) & set(order_row[:k].tolist())) / len(gold)


def run(dataset, N_seed=20, alpha=0.5, iters=20, beam_cap=2000, k=100, limit=500, reps=3):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx; n = len(id2idx)
    P, A = _P(engine, n, id2idx)
    indptr, indices = A.indptr, A.indices
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    nq = len(test)

    def dense_one(qi):
        s = X @ q[qi]
        return np.argpartition(-s, k)[:k][np.argsort(-s[np.argpartition(-s, k)[:k]])], s

    def ppr_one(qi, s):
        seeds = np.argpartition(-s, N_seed)[:N_seed]
        v = np.zeros(n, np.float32); v[seeds] = 1.0 / N_seed
        p = v.copy()
        for _ in range(iters):
            p = (1 - alpha) * v + alpha * (P.T @ p)      # P row-normalized; P.T@p = mass flow
        return np.argpartition(-p, k)[:k][np.argsort(-p[np.argpartition(-p, k)[:k]])]

    def beam_one(qi, s):
        seeds = np.argpartition(-s, N_seed)[:N_seed]
        visited = set(int(x) for x in seeds)
        frontier = list(visited)
        # 2 hops, keep the beam_cap best-by-query-cosine reached nodes
        for _ in range(2):
            nxt = []
            for d0 in frontier:
                nxt.extend(int(x) for x in indices[indptr[d0]:indptr[d0 + 1]])
            fresh = [x for x in set(nxt) if x not in visited]
            visited.update(fresh)
            frontier = fresh
            if len(visited) >= beam_cap:
                break
        cand = np.array(sorted(visited, key=lambda d: -s[d])[:max(k, beam_cap)], dtype=np.int64)
        return cand[:k]

    # recall: build each method's top-k order once
    methods = {"dense": [], "ppr": [], "beam": []}
    for qi in range(nq):
        o_d, s = dense_one(qi); methods["dense"].append(o_d)
        methods["ppr"].append(ppr_one(qi, s))
        methods["beam"].append(beam_one(qi, s))
    # timing (each method re-computes the shared dense score, since all need it in practice)
    def t_dense():
        t0 = time.perf_counter()
        for qi in range(nq): dense_one(qi)
        return (time.perf_counter() - t0) / nq * 1000.0
    def t_ppr():
        t0 = time.perf_counter()
        for qi in range(nq): ppr_one(qi, X @ q[qi])
        return (time.perf_counter() - t0) / nq * 1000.0
    def t_beam():
        t0 = time.perf_counter()
        for qi in range(nq): beam_one(qi, X @ q[qi])
        return (time.perf_counter() - t0) / nq * 1000.0
    timings = {"dense": [t_dense() for _ in range(reps)],
               "ppr": [t_ppr() for _ in range(reps)],
               "beam": [t_beam() for _ in range(reps)]}

    out = {"dataset": dataset, "n_docs": n, "n_test": nq, "k": k, "N_seed": N_seed,
           "alpha": alpha, "iters": iters, "beam_cap": beam_cap, "reps": reps, "methods": {}}
    for m, orders in methods.items():
        rec = [ _recall_at(orders[qi], gold[qi], k) for qi in range(nq) ]
        rec = [r for r in rec if r is not None]
        out["methods"][m] = {
            "median_ms_per_query": round(float(np.median(timings[m])), 3),
            "p95_ms_per_query": round(float(np.percentile(timings[m], 95)), 3),
            f"gt_recall@{k}": round(float(np.mean(rec)) * 100, 2) if rec else None,
        }
        log.info(f"  [{dataset} {m:6}] {out['methods'][m]}")
    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/l3_latency_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/research/l3_latency_{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 latency vs recall (SOTA-RAG latency class check).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa"])
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--beam_cap", type=int, default=2000)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== L3 LATENCY vs RECALL: {ds.upper()} =====")
        run(ds, beam_cap=a.beam_cap, k=a.k, limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
