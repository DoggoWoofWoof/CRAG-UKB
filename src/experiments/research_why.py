"""
WHY does PPR win on metaqa but lose on 2wiki? Decompose where the golds live.
=============================================================================
Hypothesis: it's whether answer docs are DENSE-reachable or only GRAPH-reachable.
  - 2wiki: the query names the answer entities (film titles) -> golds are dense-
    close -> dense/1hop find them directly -> PPR's graph re-ranking dilutes them.
  - metaqa: the query names a DIFFERENT entity (the actor); answer movies are
    semantically unrelated -> dense misses them -> only graph hops from the
    query-entity seed reach them -> PPR essential.
Measures, per dataset (no training):
  (1) gold dense rank: % of gold docs in dense top-{20,100,500}  [dense-reachable?]
  (2) seed goldness: % of dense-top-20 seeds that ARE gold docs  [seed = answer or a bridge entity?]
  (3) graph reachability of golds from dense-top-20 seeds: % golds within 1/2 hops
  (4) the crux: for golds NOT in dense-top-100, what % are graph-reachable
      (<=2 hops) from the seeds  [the docs only the graph can recover]
Writes results/research/why_{dataset}.json.
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

log = logging.getLogger("experiments.research_why")


def _neighbors(engine):
    docset = set(engine.node_id_to_idx)
    return {node.node_id: [x for x in node.neighbors if x in docset] for node in engine.nodes}


def run(dataset, N_seed=20, limit=3000):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
    nb = _neighbors(engine)
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    dense = q @ X.T
    order = np.argsort(-dense, axis=1)                       # full dense ranking

    gold_in = {20: [], 100: [], 500: []}
    med_rank = []
    seed_goldness = []
    reach1 = []; reach2 = []
    denseonly_miss_graphrecover = []                          # golds not in dense-top100 but graph-reachable
    avg_golds = []
    for qi, (_, _, golds) in enumerate(test):
        gset = set(id2idx[g] for g in golds if g in id2idx)
        if not gset:
            continue
        avg_golds.append(len(gset))
        rank_of = {int(d): r for r, d in enumerate(order[qi])}
        ranks = [rank_of[g] for g in gset]
        med_rank.append(np.median(ranks))
        for k in gold_in:
            gold_in[k].append(np.mean([1.0 if r < k else 0.0 for r in ranks]))
        seeds = order[qi, :N_seed].tolist()
        seed_goldness.append(np.mean([1.0 if s in gset else 0.0 for s in seeds]))
        # graph reachability from seeds
        seed_ids = [idx2id[s] for s in seeds]
        h1 = set()
        for s in seed_ids:
            h1.update(nb.get(s, ()))
        h2 = set(h1)
        for d in list(h1):
            h2.update(nb.get(d, ()))
        h1i = set(id2idx[d] for d in h1 if d in id2idx)
        h2i = set(id2idx[d] for d in h2 if d in id2idx)
        reach1.append(np.mean([1.0 if (g in h1i or g in set(seeds)) else 0.0 for g in gset]))
        reach2.append(np.mean([1.0 if (g in h2i or g in set(seeds)) else 0.0 for g in gset]))
        # crux: golds dense misses (rank>=100) — are they graph-reachable?
        missed = [g for g in gset if rank_of[g] >= 100]
        if missed:
            denseonly_miss_graphrecover.append(np.mean([1.0 if (g in h2i or g in set(seeds)) else 0.0 for g in missed]))

    def pct(x): return round(float(np.mean(x)) * 100, 1) if x else None
    out = {
        "dataset": dataset, "n_test": len(med_rank), "avg_golds_per_q": round(float(np.mean(avg_golds)), 2),
        "gold_dense_reachable": {"in_top20": pct(gold_in[20]), "in_top100": pct(gold_in[100]),
                                 "in_top500": pct(gold_in[500]), "median_gold_dense_rank": round(float(np.median(med_rank)), 1)},
        "seed_goldness_top20": pct(seed_goldness),
        "gold_graph_reachable_from_seeds": {"within_1hop": pct(reach1), "within_2hop": pct(reach2)},
        "dense_missed_golds_graph_recoverable": pct(denseonly_miss_graphrecover),
    }
    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/why_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] {json.dumps(out, indent=2)}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Decompose why PPR wins/loses by graph type.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "metaqa"])
    p.add_argument("--limit", type=int, default=3000)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== WHY (gold reachability decomposition): {ds.upper()} =====")
        run(ds, limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
