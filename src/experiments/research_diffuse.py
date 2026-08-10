"""
Representational combine: graph-diffused embeddings (APPNP-style).
==================================================================
Rank/output fusion (RRF, alpha-PPR) averages and can't beat best-of-both because
the winning signal FLIPS by graph type. The smart combine is at the
REPRESENTATION: personalized-propagate the dense doc embeddings over the graph,
  H = (1-a) X + a * (A_norm @ H)   (power-iterated; APPNP, ICLR'19)
then retrieve with a single dense search q @ H.T. Structure is folded INTO the
vectors, so:
  - sparse/text graph -> little propagation -> H ~= X (recovers dense strength)
  - relational/KB graph -> H absorbs neighbour structure (a KB movie's vector
    picks up its director/actor entities) -> a query about the neighbour now
    matches the doc = multi-hop via representation.
Self-adaptive by construction (graph density modulates diffusion), so ONE alpha
may work on both. Compares FullCov of q@H.T vs pure dense (a=0) across alpha, on
2wiki_clean (text) + metaqa (KB). Writes results/research/diffuse_{ds}.json.
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

log = logging.getLogger("experiments.research_diffuse")


def _adj_norm(engine, n, id2idx):
    rows, cols = [], []
    for i, node in enumerate(engine.nodes):
        for nb in node.neighbors:
            j = id2idx.get(nb)
            if j is not None and j != i:
                rows.append(i); cols.append(j)
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    return (sp.diags(1.0 / deg) @ A).tocsr()


def _diffuse(X, A, alpha, iters=15):
    H = X.copy()
    for _ in range(iters):
        H = (1 - alpha) * X + alpha * (A @ H)         # APPNP personalized propagation
    return H


def _fullcov(order, gold, budgets):
    fc = {b: [] for b in budgets}; gr = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        for b in budgets:
            hit = len(gs & set(order[qi][:b].tolist()))
            fc[b].append(1.0 if hit == len(gs) else 0.0)
            gr[b].append(hit / len(gs))
    return ({f"fullcov@{b}": round(float(np.mean(fc[b])) * 100, 2) for b in budgets},
            {f"gt_recall@{b}": round(float(np.mean(gr[b])) * 100, 2) for b in budgets})


def run(dataset, alphas=(0.0, 0.3, 0.5, 0.7, 0.85), budgets=(50, 100, 200, 500), limit=3000):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n = len(engine.nodes); id2idx = engine.node_id_to_idx
    A = _adj_norm(engine, n, id2idx)
    sp_test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(sp_test) > limit:
        sp_test = sp_test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in sp_test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in sp_test]

    out = {"dataset": dataset, "n_docs": n, "avg_doc_degree": round(A.nnz / n, 2),
           "budgets": list(budgets), "alphas": list(alphas), "results": {}}
    for a in alphas:
        H = X if a == 0.0 else _diffuse(X, A, a)
        Hn = H.copy(); faiss.normalize_L2(Hn)
        order = np.argsort(-(q @ Hn.T), axis=1)[:, :max(budgets)]
        fc, gr = _fullcov(order, gold, budgets)
        out["results"][f"diffuse_a{a:g}"] = {"fullcov": fc, "gt_recall": gr}
        tag = " (=pure dense)" if a == 0.0 else ""
        log.info(f"  [{dataset} a={a:g}{tag}] FullCov {fc}")

    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/diffuse_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/research/diffuse_{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Graph-diffused embeddings (APPNP) representational combine.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "metaqa"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 0.85])
    p.add_argument("--limit", type=int, default=3000)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== DIFFUSED-EMBEDDING combine: {ds.upper()} =====")
        run(ds, alphas=tuple(a.alphas), limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
