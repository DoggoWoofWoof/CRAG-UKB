"""
Score-level (representational) fusion of dense + query-personalized PPR.
========================================================================
Rank fusion (RRF) and static diffusion (APPNP) both failed: RRF averages ranks
(weak signal pollutes), APPNP is query-independent (can't do KB's query-conditional
hop). The right combine fuses the two SIGNALS per-doc, per-query:
  dense(q,d)  -- semantic similarity (strong on text, ~flat on KB)
  ppr(d|q)    -- personalized-PageRank reachability from dense-top-N seeds
                 (strong on KB via multi-hop, thin on sparse text)
z-normalize each per query, then combine. Test SUM (dense_z+ppr_z), MAX
(union: a doc strong in EITHER signal wins -> weak signal can't drown the strong,
unlike RRF), and CONFIDENCE-weighted (weight each signal by its per-query score
std = discriminativeness). If a single fusion matches best-of-both (2wiki ~1hop,
metaqa ~ppr), the representational combine works with no per-corpus gate.
Writes results/research/fuse_{dataset}.json.
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

log = logging.getLogger("experiments.research_fuse")


def _adj(engine, n, id2idx):
    r, c = [], []
    for i, node in enumerate(engine.nodes):
        for nb in node.neighbors:
            j = id2idx.get(nb)
            if j is not None and j != i:
                r.append(i); c.append(j)
    A = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n)); A = A.maximum(A.T)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return (sp.diags(1.0 / d) @ A).tocsr()


def _zn(M):
    mu = M.mean(1, keepdims=True); sd = M.std(1, keepdims=True); sd[sd == 0] = 1
    return (M - mu) / sd


def _fullcov(order, gold, budgets):
    fc = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        for b in budgets:
            fc[b].append(1.0 if len(gs & set(order[qi][:b].tolist())) == len(gs) else 0.0)
    return {f"fullcov@{b}": round(float(np.mean(fc[b])) * 100, 2) for b in budgets}


def run(dataset, N_seed=20, alpha=0.5, iters=20, budgets=(50, 100, 200, 500), limit=3000):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n = len(engine.nodes); id2idx = engine.node_id_to_idx
    A = _adj(engine, n, id2idx)
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    nq = len(test)

    dense = q @ X.T                                                     # (nq, n)
    seeds = np.zeros((nq, n), np.float32)
    topN = np.argsort(-dense, axis=1)[:, :N_seed]
    for qi in range(nq):
        seeds[qi, topN[qi]] = 1.0 / N_seed
    ppr = seeds.copy()
    for _ in range(iters):
        ppr = (1 - alpha) * seeds + alpha * (ppr @ A)

    dz, pz = _zn(dense), _zn(ppr)
    sd_d = dense.std(1, keepdims=True); sd_p = ppr.std(1, keepdims=True)
    wd = sd_d / (sd_d + sd_p + 1e-9); wp = 1 - wd                       # confidence weights (per query)
    variants = {
        "dense": dense, "ppr": ppr,
        "fuse_sum": dz + pz,
        "fuse_max": np.maximum(dz, pz),
        "fuse_conf": wd * dz + wp * pz,
    }
    out = {"dataset": dataset, "n_docs": n, "avg_doc_degree": round(A.nnz / n, 2),
           "alpha": alpha, "N_seed": N_seed, "budgets": list(budgets), "results": {}}
    for name, S in variants.items():
        order = np.argsort(-S, axis=1)[:, :max(budgets)]
        fc = _fullcov(order, gold, budgets)
        out["results"][name] = fc
        log.info(f"  [{dataset} {name:9}] {fc}")
    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/fuse_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/research/fuse_{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Score-level fusion of dense + personalized-PPR.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "metaqa"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--limit", type=int, default=3000)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== SCORE-FUSION dense+PPR: {ds.upper()} =====")
        run(ds, N_seed=a.N_seed, alpha=a.alpha, limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
