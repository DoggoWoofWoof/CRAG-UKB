"""
PPR (HippoRAG-style) L3 traversal vs bounded 1-hop (research improvement).
==========================================================================
Literature: HippoRAG (1/2) recovers multi-hop evidence by Personalized PageRank
diffusion from dense-retrieved seed docs over the doc graph, rather than a fixed
k-hop expansion. Our L3 is bounded 1-hop from dense anchors. This compares three
recovery strategies on the clean substrate, at matched doc budgets:
  - dense-only : top-B docs by query-doc cosine (no graph)
  - 1hop       : dense top-N seeds + their 1-hop neighbours, then fill by cosine
  - ppr        : PPR from dense-top-N seeds over the (structural) doc graph
Metric: gold-doc recall and all-golds FullCov at doc budget B. PPR is expected to
help most where the graph is genuinely relational (metaqa KB); on title-sparse
text graphs it should be closer to dense/1hop. Writes results/research/ppr_{ds}.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import scipy.sparse as sp
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership

log = logging.getLogger("experiments.research_ppr")


def _adj(engine, n, id2idx):
    rows, cols = [], []
    for i, node in enumerate(engine.nodes):
        for nb in node.neighbors:                     # structural doc-doc edges
            j = id2idx.get(nb)
            if j is not None and j != i:
                rows.append(i); cols.append(j)
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)                                # undirected
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    D_inv = sp.diags(1.0 / deg)
    return (D_inv @ A).tocsr()                        # row-normalized transition


def _ppr(seeds_mat, P, alpha=0.5, iters=20):
    """Batched PPR: seeds_mat (nq, n) restart distribution; returns (nq, n) scores."""
    p = seeds_mat.copy()
    for _ in range(iters):
        p = (1 - alpha) * seeds_mat + alpha * (p @ P)
    return p


def _rrf(orders, maxb, k=60):
    """Reciprocal-rank fusion of several per-query doc orderings -> (nq, maxb)."""
    nq = orders[0].shape[0]
    fused = []
    for qi in range(nq):
        score = {}
        for od in orders:
            for rank, d in enumerate(od[qi].tolist()):
                score[d] = score.get(d, 0.0) + 1.0 / (k + rank)
        fused.append(sorted(score, key=lambda d: -score[d])[:maxb])
    return np.array(fused)


def _fullcov(order_docs, gold, budgets):
    fc = {b: [] for b in budgets}; gr = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        for b in budgets:
            top = set(order_docs[qi][:b])
            cov = len(gs & top)
            fc[b].append(1.0 if cov == len(gs) else 0.0)
            gr[b].append(cov / len(gs))
    return ({f"fullcov@{b}": round(float(np.mean(fc[b])) * 100, 2) for b in budgets},
            {f"gt_recall@{b}": round(float(np.mean(gr[b])) * 100, 2) for b in budgets})


def run(dataset, N_seed=20, budgets=(50, 100, 200, 500), alphas=(0.1, 0.3, 0.5, 0.7, 0.9), limit=3000):
    engine = CoreEngine(source=dataset)
    nv = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(nv)
    n = len(engine.nodes); id2idx = engine.node_id_to_idx
    P = _adj(engine, n, id2idx)
    nnz_per = P.nnz / n
    sp_test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(sp_test) > limit:                # cap for the dense (nq x n) PPR matrix memory
        sp_test = sp_test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in sp_test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in sp_test]
    nq = len(sp_test)

    qsim = q @ nv.T                                   # (nq, n) dense scores
    dense_order = np.argsort(-qsim, axis=1)[:, :max(budgets)]

    # seed matrix: dense top-N, uniform restart
    seeds = sp.lil_matrix((nq, n), dtype=np.float32)
    for qi in range(nq):
        for d in dense_order[qi, :N_seed]:
            seeds[qi, d] = 1.0 / N_seed
    seeds = seeds.tocsr()
    seeds_dense = seeds.toarray().astype(np.float32)
    ppr_orders = {a: np.argsort(-_ppr(seeds_dense, P, alpha=a), axis=1)[:, :max(budgets)] for a in alphas}

    # 1-hop: seeds + their neighbours, ordered by dense score within that set, then dense fill
    onehop_order = []
    Pcoo = P.tocsr()
    for qi in range(nq):
        seedset = list(dense_order[qi, :N_seed])
        nbrs = set()
        for s in seedset:
            nbrs.update(Pcoo.indices[Pcoo.indptr[s]:Pcoo.indptr[s + 1]].tolist())
        cand = list(dict.fromkeys(seedset + sorted(nbrs, key=lambda d: -qsim[qi, d])))
        fill = [d for d in dense_order[qi] if d not in set(cand)]
        onehop_order.append((cand + fill)[:max(budgets)])
    onehop_order = np.array(onehop_order)

    out = {"dataset": dataset, "n_docs": n, "avg_doc_degree": round(nnz_per, 2),
           "N_seed": N_seed, "alphas": list(alphas), "budgets": list(budgets), "results": {}}
    methods = [("dense", dense_order), ("1hop", onehop_order)]
    methods += [(f"ppr_a{a:g}", ppr_orders[a]) for a in alphas]
    for name, order in methods:
        fc, gr = _fullcov(order, gold, budgets)
        out["results"][name] = {"fullcov": fc, "gt_recall": gr}
        log.info(f"  [{dataset} {name:9}] FullCov {fc}")
    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/ppr_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/research/ppr_{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="PPR (HippoRAG-style) L3 vs bounded 1-hop.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "metaqa"])
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--limit", type=int, default=3000)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== PPR alpha-sweep (1-hop = low-alpha limit) L3: {ds.upper()} =====")
        run(ds, N_seed=a.N_seed, alphas=tuple(a.alphas), limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
