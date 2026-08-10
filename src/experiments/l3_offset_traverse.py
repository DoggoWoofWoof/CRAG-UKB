"""
Relation-offset reachability at L3 (phase 21, redirected + made cheap).
=======================================================================
Offsets failed at L1 routing; the probe validated them for DOC->DOC. This tests the
cheap, decisive L3 question: from the query's dense seeds, do relation-offset hops
(seed + r_k -> nearest docs) reach GOLDS that graph-1hop and dense miss? One
batched faiss search per query (N_seed x K), no per-node cost. Compares gold
reachability of:
  seed   : the N_seed dense seeds
  graph  : seeds ∪ 1-hop graph neighbours (title/kNN edges)
  offset : seeds ∪ relation-offset neighbours (K learned directions)
  both   : seeds ∪ graph ∪ offset
If offset > graph (esp on metaqa), relation-offset hops add reach the stored graph
can't — the payoff of the king->queen finding, at the level it belongs (L3).
Writes data/ukb_storage/{ds}/results/L3/offset_reach.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l3_reachability import _adj
from src.experiments.relation_route import _learn_offsets
from src.pipeline.ukb_results import rpath


def run(dataset, K=16, N_seed=10, n_off=3, limit=800):
    log = logging.getLogger("experiments.l3_offset_traverse")
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d); index.add(X)
    id2idx = engine.node_id_to_idx
    adj, _, _ = _adj(engine, id2idx)
    R = _learn_offsets(X, engine, id2idx, K)
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    _, seed_I = index.search(q, N_seed)

    reach = {k: [] for k in ("seed", "graph", "offset", "both")}
    ball = {k: [] for k in ("graph", "offset")}
    for qi in range(len(test)):
        gs = set(gold[qi])
        if not gs:
            continue
        seeds = [int(s) for s in seed_I[qi]]
        g = set(seeds)
        for s in seeds:
            g.update(int(x) for x in adj[s])
        V = X[seeds]                                              # (N_seed, d)
        Q = (V[:, None, :] + R[None, :, :]).reshape(-1, d)       # (N_seed*K, d)
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        _, I = index.search(Q.astype("float32"), n_off + 1)
        o = set(seeds) | {int(j) for j in I.ravel()}
        both = g | o
        reach["seed"].append(len(gs & set(seeds)) / len(gs))
        reach["graph"].append(len(gs & g) / len(gs))
        reach["offset"].append(len(gs & o) / len(gs))
        reach["both"].append(len(gs & both) / len(gs))
        ball["graph"].append(len(g)); ball["offset"].append(len(o))

    out = {"dataset": dataset, "K": K, "N_seed": N_seed, "n_off": n_off, "n_test": len(reach["seed"]),
           "reach_pct": {k: round(np.mean(v) * 100, 2) for k, v in reach.items()},
           "avg_frontier": {k: round(np.mean(v), 1) for k, v in ball.items()},
           "offset_over_graph": round((np.mean(reach["offset"]) - np.mean(reach["graph"])) * 100, 2)}
    with open(rpath(dataset, "L3", "offset_reach"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] reach {out['reach_pct']} | offset-over-graph {out['offset_over_graph']} | "
             f"frontier {out['avg_frontier']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Relation-offset reachability at L3 (cheap).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "2wiki_clean", "musique_clean"])
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--n_off", type=int, default=3)
    p.add_argument("--limit", type=int, default=800)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        run(ds, K=a.K, n_off=a.n_off, limit=a.limit)


if __name__ == "__main__":
    main()
