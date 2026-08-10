"""
Probe: can pure semantic-space traversal replace the stored relational graph? (#18)
====================================================================================
Decisive test of the "graph == semantics" premise. From identical dense seeds,
compare gold reachability under two traversals over h=1,2,3 hops:
  SEM   : query-time kNN in the embedding space (implicit graph = each node's
          k nearest vectors, computed on the fly via the faiss doc index). NO
          stored edges. This is "traverse the semantic space directly".
  GRAPH : the stored UKB graph (node.neighbors = title/KB relational edges UNION
          synthetic kNN) — what L3 uses today.
If SEM ~= GRAPH, the graph can go fully implicit (drop stored edges). If GRAPH >>
SEM — expected on KB (metaqa: golds at median dense rank 8671, a semantic
discontinuity a vector hop can't cross) — the RELATIONAL layer adds reach that
vector-kNN cannot, so we keep a thin relational graph and let semantics do the rest.
gap(GRAPH - SEM) per dataset = the value of the relational edges, quantified.
Writes results/research/sem_probe_{dataset}.json.
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

log = logging.getLogger("experiments.l3_semantic_probe")
HOPS = 3


def _sem_reach(index, X, seeds, gold, k, hops):
    """gold reachability via query-time kNN traversal (implicit semantic graph)."""
    visited = set(int(s) for s in seeds)
    frontier = list(visited)
    reach, ball = [], []
    gs = set(gold)
    for h in range(hops):
        if frontier:
            _, I = index.search(X[frontier], k + 1)                     # +1 to drop self
            nxt = set(int(j) for row in I for j in row) - visited
            visited |= nxt
            frontier = list(nxt)
        reach.append(len(gs & visited) / len(gs)); ball.append(len(visited))
    return reach, ball


def _graph_reach(adj, seeds, gold, hops):
    visited = set(int(s) for s in seeds); frontier = set(visited); gs = set(gold)
    reach, ball = [], []
    for h in range(hops):
        nxt = set()
        for d in frontier:
            nxt.update(int(x) for x in adj[d])
        frontier = nxt - visited; visited |= frontier
        reach.append(len(gs & visited) / len(gs)); ball.append(len(visited))
    return reach, ball


def run(dataset, N_seed=20, k=8, limit=400, device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)                 # query-time semantic kNN
    id2idx = engine.node_id_to_idx
    adj, deg_struct, deg_syn = _adj(engine, id2idx)
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]

    _, seed_I = index.search(q, N_seed)
    sem = {h: [] for h in range(HOPS)}; gph = {h: [] for h in range(HOPS)}
    sem_ball = {h: [] for h in range(HOPS)}; gph_ball = {h: [] for h in range(HOPS)}
    for qi in range(len(test)):
        if not gold[qi]:
            continue
        sr, sb = _sem_reach(index, X, seed_I[qi], gold[qi], k, HOPS)
        gr, gb = _graph_reach(adj, seed_I[qi], gold[qi], HOPS)
        for h in range(HOPS):
            sem[h].append(sr[h]); gph[h].append(gr[h]); sem_ball[h].append(sb[h]); gph_ball[h].append(gb[h])

    def pct(x): return round(float(np.mean(x)) * 100, 1) if x else None
    out = {"dataset": dataset, "n_docs": X.shape[0], "N_seed": N_seed, "sem_k": k,
           "deg_struct": deg_struct, "deg_syn": deg_syn,
           "semantic_reach_pct": {f"h{h+1}": pct(sem[h]) for h in range(HOPS)},
           "graph_reach_pct": {f"h{h+1}": pct(gph[h]) for h in range(HOPS)},
           "gap_graph_minus_sem": {f"h{h+1}": round((pct(gph[h]) or 0) - (pct(sem[h]) or 0), 1) for h in range(HOPS)},
           "semantic_ball_docs": {f"h{h+1}": round(float(np.mean(sem_ball[h])), 0) for h in range(HOPS)},
           "graph_ball_docs": {f"h{h+1}": round(float(np.mean(gph_ball[h])), 0) for h in range(HOPS)}}
    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/sem_probe_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] SEM {out['semantic_reach_pct']}  GRAPH {out['graph_reach_pct']}  "
             f"gap {out['gap_graph_minus_sem']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Pure-semantic vs stored-graph traversal reachability probe.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "metaqa", "musique_clean"])
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--k", type=int, default=8, help="query-time kNN degree (match stored avg degree)")
    p.add_argument("--limit", type=int, default=400)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== SEMANTIC-TRAVERSAL PROBE: {ds.upper()} =====")
        run(ds, N_seed=a.N_seed, k=a.k, limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
