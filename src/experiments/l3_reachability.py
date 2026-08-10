"""
L3 feasibility: reachability ceiling vs frontier explosion (seed-and-traverse).
================================================================================
Under the seed-and-traverse design (L1+L2 pick seed nodes; L3 = best-first search
across the WHOLE graph with pruning + an info-sufficiency stop; miss => best-effort),
the two numbers that decide feasibility are:
  (a) REACHABILITY CEILING: from the seeds, what fraction of gold docs are graph-
      reachable within h=1,2,3 hops. This is the "if it's in the answer it'll be
      connected" assumption made quantitative -- the best-effort recall an
      exhaustive traversal could ever reach.
  (b) FRONTIER GROWTH: how many unique docs the h-hop ball contains (cumulative).
      If 2 hops already engulfs most of the corpus, "traverse until found" is
      hopeless without aggressive pruning -- so this is the pruning/stop burden.
Also reports the min-hop-to-each-gold distribution (how deep a perfect-stop search
must go) and, for reached golds, how many docs are expanded before the LAST gold
is hit in a best-first (by query-cosine) expansion order (the realistic cost).

Seeds = dense top-N over the full corpus (a generous L1+L2 proxy). Graph = the
FIXED substrate: node.neighbors (title/structural) UNION synthetic kNN edges.
Writes results/research/reach_{dataset}.json.
"""
import os
import json
import logging
import argparse
from collections import deque

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership

log = logging.getLogger("experiments.l3_reachability")

HOPS = 3


def _adj(engine, id2idx):
    """doc-doc adjacency (title/struct UNION synthetic kNN) as arrays for fast BFS."""
    docset = set(id2idx)
    adj = [[] for _ in range(len(id2idx))]
    n_struct = n_syn = 0
    for node in engine.nodes:
        i = id2idx[node.node_id]
        seen = set()
        for nb in node.neighbors:
            if nb in docset and nb != node.node_id:
                j = id2idx[nb]
                if j not in seen:
                    adj[i].append(j); seen.add(j); n_struct += 1
        for nb in node.metadata.get("synthetic_neighbors", ()):
            if nb in docset and nb != node.node_id:
                j = id2idx[nb]
                if j not in seen:
                    adj[i].append(j); seen.add(j); n_syn += 1
    adj = [np.array(a, dtype=np.int64) for a in adj]
    return adj, round(n_struct / len(id2idx), 2), round(n_syn / len(id2idx), 2)


def run(dataset, N_seed=20, limit=800, device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx
    n = len(id2idx)
    adj, deg_struct, deg_syn = _adj(engine, id2idx)
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    dense = q @ X.T
    seeds_all = np.argsort(-dense, axis=1)[:, :N_seed]

    reach = {h: [] for h in range(HOPS + 1)}      # frac golds reached within h hops
    ball = {h: [] for h in range(HOPS + 1)}       # cumulative unique docs in h-hop ball
    minhop = []                                    # min hop to reach a gold (reached only)
    cost_last_gold = []                            # #docs expanded (best-first) before last gold
    avg_golds = []
    for qi, (_, _, golds) in enumerate(test):
        gset = set(id2idx[g] for g in golds if g in id2idx)
        if not gset:
            continue
        avg_golds.append(len(gset))
        # BFS layers from the seed set
        frontier = set(int(s) for s in seeds_all[qi])
        visited = set(frontier)
        hop_of = {d: 0 for d in frontier}
        for h in range(HOPS + 1):
            if h > 0:
                nxt = set()
                for d in frontier:
                    for j in adj[d]:
                        jj = int(j)
                        if jj not in visited:
                            visited.add(jj); nxt.add(jj); hop_of[jj] = h
                frontier = nxt
            reach[h].append(len(gset & visited) / len(gset))
            ball[h].append(len(visited))
        for g in gset:
            if g in hop_of:
                minhop.append(hop_of[g])
        # realistic best-first cost: expand the h<=HOPS ball in descending query-cosine,
        # count how many expansions until the last REACHED gold is seen
        reached_g = gset & visited
        if reached_g:
            order = sorted(visited, key=lambda d: -dense[qi, d])
            pos = {d: r for r, d in enumerate(order)}
            cost_last_gold.append(max(pos[g] for g in reached_g) + 1)

    def pct(x): return round(float(np.mean(x)) * 100, 1) if x else None
    def avg(x): return round(float(np.mean(x)), 1) if x else None
    out = {
        "dataset": dataset, "n_docs": n, "n_test": len(avg_golds),
        "avg_golds": round(float(np.mean(avg_golds)), 2), "N_seed": N_seed,
        "deg_struct": deg_struct, "deg_syn": deg_syn,
        "reachability_pct": {f"h{h}": pct(reach[h]) for h in range(HOPS + 1)},
        "frontier_docs": {f"h{h}": avg(ball[h]) for h in range(HOPS + 1)},
        "frontier_pct_of_corpus": {f"h{h}": round(avg(ball[h]) / n * 100, 1) if ball[h] else None
                                   for h in range(HOPS + 1)},
        "min_hop_to_gold_mean": avg(minhop),
        "bestfirst_cost_to_last_gold_mean": avg(cost_last_gold),
    }
    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/reach_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] {json.dumps(out)}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 reachability ceiling vs frontier growth.")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "metaqa", "squad_clean"])
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--limit", type=int, default=800)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== L3 REACHABILITY: {ds.upper()} =====")
        run(ds, N_seed=a.N_seed, limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
