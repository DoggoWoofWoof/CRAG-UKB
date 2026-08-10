"""
L2 cross-encoder rerank -> minimal seed selection (phase 6).
============================================================
L2 turns L1's high-recall pool into a few high-precision, well-connected SEEDS for
L3. It is NOT scored on full recall (L3 traverses to the rest) but on: how few
seeds are needed so L3's graph reach covers X% of golds. A cross-encoder
(query+doc joint scoring) should beat the bi-encoder (dense) at picking launch
points; SPLADE can be a cheap pre-rank before it (not implemented here — dense pool
is the first stage).

Protocol per query: L1 pool = dense top-`pool` docs (front-end proxy; or the
champion router if champion_model.pt exists) -> rerank the pool by {dense, cross-
encoder} -> for seed budgets N, measure gold reachability within h hops over the
UKB graph (structural title UNION synthetic kNN). Report reach@N curves + the min
N to hit `target`, cross-encoder vs dense. Writes L2/seed_selection.json.

Cross-encoder on CPU is pairwise-slow: use --limit + modest --pool.
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
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.l2_rerank")
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _reach(seed_idxs, adj, gold, hops=2):
    """fraction of gold idxs reachable within `hops` of the seed set."""
    if not gold:
        return None
    visited = set(seed_idxs)
    frontier = set(seed_idxs)
    for _ in range(hops):
        nxt = set()
        for d in frontier:
            nxt.update(int(x) for x in adj[d])
        frontier = nxt - visited
        visited |= frontier
    return len(set(gold) & visited) / len(gold)


def run(dataset, pool=200, seed_budgets=(5, 10, 20, 50), hops=2, target=0.9,
        limit=400, ce_model=CE_MODEL, device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx
    adj, deg_struct, deg_syn = _adj(engine, id2idx)
    docs_content = [n.content for n in engine.nodes]
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(ce_model, device=str(device) if device else None)

    dense = q @ X.T
    pool_idx = np.argsort(-dense, axis=1)[:, :pool]                       # L1 pool (dense proxy)
    maxN = max(seed_budgets)
    reach = {m: {N: [] for N in seed_budgets} for m in ("dense", "cross")}
    for qi, (qn, _, _) in enumerate(test):
        cand = pool_idx[qi].tolist()
        dense_order = cand[:maxN]                                         # dense already sorted
        pairs = [(qn.content, docs_content[c]) for c in cand]
        scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
        cross_order = [cand[i] for i in np.argsort(-scores)][:maxN]
        for N in seed_budgets:
            reach["dense"][N].append(_reach(dense_order[:N], adj, gold[qi], hops))
            reach["cross"][N].append(_reach(cross_order[:N], adj, gold[qi], hops))
        if qi % 50 == 0:
            log.info(f"  [{dataset}] reranked {qi+1}/{len(test)}")

    def summ(m):
        out = {}
        for N in seed_budgets:
            vals = [r for r in reach[m][N] if r is not None]
            out[f"reach@{N}seeds"] = round(float(np.mean(vals)) * 100, 2) if vals else None
        # min seeds to hit target (interp over budgets)
        xs = list(seed_budgets); ys = [out[f"reach@{N}seeds"] / 100 for N in seed_budgets]
        minN = next((N for N in seed_budgets if out[f"reach@{N}seeds"] and out[f"reach@{N}seeds"] >= target * 100), None)
        out["min_seeds_for_target"] = minN
        return out

    result = {"dataset": dataset, "pool": pool, "hops": hops, "target": target,
              "n_test": len(test), "deg_struct": deg_struct, "deg_syn": deg_syn,
              "ce_model": ce_model, "seed_budgets": list(seed_budgets),
              "dense": summ("dense"), "cross_encoder": summ("cross")}
    with open(rpath(dataset, "L2", "seed_selection"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(f"[{dataset}] dense {result['dense']}  |  cross {result['cross_encoder']}")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="L2 cross-encoder rerank -> minimal seeds.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--pool", type=int, default=200)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--target", type=float, default=0.9)
    p.add_argument("--limit", type=int, default=400)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        run(ds, pool=a.pool, hops=a.hops, target=a.target, limit=a.limit)


if __name__ == "__main__":
    main()
