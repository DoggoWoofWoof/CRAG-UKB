"""
L3 solver: bounded PPR-guided best-first traversal + composite stop + traceability.
====================================================================================
The settled L3 (see l3-design-findings): from L2 seeds, expand the graph best-first
where a node's priority PROPAGATES from its parent (PPR-like graph mass) blended
with query cosine — so on KB (cosine flat) the graph structure steers, on text the
query prior steers. Bounded frontier (hard cap) + a COMPOSITE stop (halt on the
first signal that fires):
  - entity_coverage : every named entity in the question is grounded in a result
  - anchor_meet     : frontiers from two different seeds converge (bridge/answer)
  - marginal_saturation : the last W pops are all below a query-relevance floor
                          (diminishing returns -> best-effort return)
  - reader_sufficiency : optional callback (a reader confirms the answer) — off by default
  - budget          : hard node cap (fallback)
Every result carries its PATH back to a seed (traceability — the connecting
evidence vanilla top-k RAG can't give). `traverse()` is the reusable core;
run() benchmarks recall / nodes-expanded / stop-reason over the test split and
saves a few example traces. Writes L3/traverse.json.
"""
import os
import json
import time
import heapq
import logging
import argparse

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l3_reachability import _adj
from src.pipeline.ner_edges import _entities_regex
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.l3_traverse")


def traverse(qvec, qents, seeds, adj, nv, content_lower, budget=200, alpha=0.5,
             marg_win=15, marg_floor=0.15, min_explore=25, reader=None,
             offsets=None, index=None, n_off=2):
    """Bounded PPR-guided best-first from `seeds`. Expands to `budget` for the recall
    measurement; the composite stop conditions are recorded as ANNOTATIONS (where a
    real system would halt for efficiency) rather than cutting the expansion short —
    with a min_explore floor so they can't fire on the seed region itself. A stop
    also requires evidence beyond the seeds (entity covered by a NON-seed node;
    anchor-meet at a non-seed node). Returns order, parent map, per-condition
    stop-node, and the composite stop (earliest fire)."""
    seedset = set(int(s) for s in seeds)
    heap = []
    parent, seed_of = {}, {}
    for si, s in enumerate(seeds):
        pri = max(float(nv[s] @ qvec), 1e-3)
        heapq.heappush(heap, (-pri, int(s)))
        parent[int(s)] = None; seed_of[int(s)] = si
    visited, order = set(), []
    covered, recent, meet = set(), [], None
    stop_at = {}                                             # condition -> node count at first fire
    while heap and len(order) < budget:
        neg, d = heapq.heappop(heap)
        if d in visited:
            continue
        visited.add(d); order.append(d); i = len(order)
        cl = content_lower[d]
        if d not in seedset:                                # only NON-seed nodes ground entities
            for e in qents:
                if e not in covered and e in cl:
                    covered.add(e)
        recent.append(float(nv[d] @ qvec))
        if len(recent) > marg_win:
            recent.pop(0)
        if i >= min_explore:                                # can't stop inside the seed region
            if "entity_coverage" not in stop_at and qents and qents <= covered:
                stop_at["entity_coverage"] = i
            if "anchor_meet" not in stop_at and meet is not None:
                stop_at["anchor_meet"] = i
            if ("marginal_saturation" not in stop_at and len(recent) == marg_win
                    and max(recent) < marg_floor):
                stop_at["marginal_saturation"] = i
            if reader is not None and "reader_sufficiency" not in stop_at and reader(order):
                stop_at["reader_sufficiency"] = i
        for nb in adj[d]:
            nb = int(nb)
            if nb in visited:
                continue
            prop = alpha * (-neg) + (1 - alpha) * max(float(nv[nb] @ qvec), 0.0)
            heapq.heappush(heap, (-prop, nb))
            if nb not in parent:
                parent[nb] = d; seed_of[nb] = seed_of[d]
            elif nb not in seedset and seed_of.get(nb) is not None and seed_of[nb] != seed_of[d] and meet is None:
                meet = nb                                   # two seed-lineages converge at a non-seed node
        if offsets is not None and index is not None:       # relation-offset hops (doc->doc, king->queen)
            for r in offsets:
                v = nv[d] + r; v = v / (np.linalg.norm(v) + 1e-9)
                _, I = index.search(v[None].astype("float32"), n_off + 1)
                for nb in I[0]:
                    nb = int(nb)
                    if nb in visited or nb == d:
                        continue
                    prop = alpha * (-neg) + (1 - alpha) * max(float(nv[nb] @ qvec), 0.0)
                    heapq.heappush(heap, (-prop, nb))
                    parent.setdefault(nb, d); seed_of.setdefault(nb, seed_of[d])
    stop_reason = min(stop_at, key=stop_at.get) if stop_at else "budget"
    stop_node = stop_at[stop_reason] if stop_at else len(order)
    return {"order": order, "parent": parent, "stop": stop_reason, "stop_node": stop_node, "meet": meet}


def _path(parent, d):
    p = [d]
    while parent.get(p[-1]) is not None:
        p.append(parent[p[-1]])
    return list(reversed(p))


def run(dataset, N_seed=10, budget=200, alpha=0.5, limit=500, device=None, n_traces=3):
    engine = CoreEngine(source=dataset)
    nv = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(nv)
    id2idx = engine.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
    adj, deg_struct, deg_syn = _adj(engine, id2idx)
    content_lower = [n.content.lower() for n in engine.nodes]
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]

    dense_top = np.argsort(-(q @ nv.T), axis=1)[:, :N_seed]              # L2-seed proxy (dense top-N)
    budgets = [20, 50, 100, budget]
    rec = {b: [] for b in budgets}; nodes_used = []; stop_nodes = []; stops = {}
    lat = []; traces = []
    for qi, (qn, _, _) in enumerate(test):
        qents = _entities_regex(qn.content)
        qents = {e for e in qents if len(e) >= 3}
        t0 = time.perf_counter()
        res = traverse(q[qi], qents, dense_top[qi], adj, nv, content_lower,
                       budget=budget, alpha=alpha)
        lat.append((time.perf_counter() - t0) * 1000)
        order = res["order"]; nodes_used.append(len(order)); stop_nodes.append(res["stop_node"])
        stops[res["stop"]] = stops.get(res["stop"], 0) + 1
        if gold[qi]:
            gs = set(gold[qi])
            for b in budgets:
                rec[b].append(len(gs & set(order[:b])) / len(gs))
        if len(traces) < n_traces and gold[qi]:
            hit = [g for g in gold[qi] if g in set(order)]
            if hit:
                traces.append({"question": qn.content[:120],
                               "stop": res["stop"], "nodes": len(order),
                               "example_path_to_gold": [idx2id[x] for x in _path(res["parent"], hit[0])]})

    out = {"dataset": dataset, "N_seed": N_seed, "budget": budget, "alpha": alpha,
           "n_test": len(test), "deg_struct": deg_struct, "deg_syn": deg_syn,
           "gt_recall": {f"@{b}": round(float(np.mean(rec[b])) * 100, 2) for b in budgets if rec[b]},
           "avg_nodes_expanded": round(float(np.mean(nodes_used)), 1),
           "avg_nodes_to_stop": round(float(np.mean(stop_nodes)), 1),
           "median_latency_ms": round(float(np.median(lat)), 2),
           "stop_reason_counts": stops, "example_traces": traces}
    with open(rpath(dataset, "L3", "traverse"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] recall={out['gt_recall']} nodes~{out['avg_nodes_expanded']} "
             f"stops={stops} -> {rpath(dataset,'L3','traverse')}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 bounded PPR-guided best-first traversal + traceability.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--N_seed", type=int, default=10)
    p.add_argument("--budget", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=500)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        run(ds, N_seed=a.N_seed, budget=a.budget, alpha=a.alpha, limit=a.limit)


if __name__ == "__main__":
    main()
