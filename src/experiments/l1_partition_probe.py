"""
Partition-reachability probe — does OVERLAP make multi-hop golds routable in L1?
================================================================================
Core design point: overlapping partitions exist so a cross-boundary multi-hop gold is
STILL a member of the seed's (or a routed) partition, so routing retrieves it — no
traversal. "gold 2 graph-hops away" != "unroutable". This probe tests that directly by
comparing HARD membership (one partition/node) vs OVERLAP membership (own + 1-hop-neighbour
partitions, the champion's overlap1):

  spanned_partitions      : # distinct HARD partitions a query's golds span
  pct_multihop_golds      : golds 2+ GRAPH-hops from the dense seed
  oracle_fullcov          : route to partitions that cover the golds' membership; pull top
                            quota/partition by query-cos -> FullCov (ceiling), HARD vs OVERLAP
  pred_partition_fullcov  : route to top-P partitions by query.centroid; pull quota ->
                            FullCov end-to-end, HARD vs OVERLAP  (+ MULTIHOP-gold-only)
If OVERLAP >> HARD (esp. on multi-hop golds), the overlap mechanism recovers multi-hop
within L1 and the fix is a learned MULTI-LABEL partition router over the overlap membership.
No training. Writes L1/partition_probe.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, _onehop_membership
from src.experiments.l3_reachability import _adj

log = logging.getLogger("experiments.l1_partition_probe")
BUDGETS = [50, 100, 200]
TOPP = [10, 20]


def _hopset(seed, targets, adj, max_hop=2):
    want = set(int(t) for t in targets); got = {}
    seen = {int(seed)}; frontier = {int(seed)}
    for h in range(1, max_hop + 1):
        nxt = set()
        for dd in frontier:
            for j in adj[dd]:
                jj = int(j)
                if jj not in seen:
                    seen.add(jj); nxt.add(jj)
                    if jj in want:
                        got[jj] = h
        frontier = nxt
        if not frontier:
            break
    return got


def run(dataset, limit=800, device=None):
    engine = CoreEngine(source=dataset); enc = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n = X.shape[0]; id2idx = engine.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
    adj, _, _ = _adj(engine, id2idx)
    hard = np.array([int(engine.partition_map.get(nd.node_id, -1)) for nd in engine.nodes])
    npart = int(hard.max()) + 1
    ov = _onehop_membership(engine)                          # node_id -> {own, 1-hop-neighbour partitions}
    mem = [sorted(ov.get(idx2id[i], {int(hard[i])})) for i in range(n)]  # idx -> membership partition list
    docs_hard = [np.where(hard == p)[0] for p in range(npart)]
    inv = [[] for _ in range(npart)]                         # overlap: p -> nodes whose membership includes p
    for i in range(n):
        for p in mem[i]:
            inv[p].append(i)
    docs_ov = [np.array(v, dtype=np.int64) for v in inv]
    cents = _reconstruct(engine.centroid_index).astype("float32"); faiss.normalize_L2(cents)
    cpids = [int(p) for p in engine.centroid_pids]
    test = _splits(engine, _hard_membership(engine))["test"][:limit]
    q = enc.encode([nd.content for nd, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    qsim = q @ X.T; seed = np.argmax(qsim, axis=1); qcent = q @ cents.T

    def pool_from(parts, qi, budget, docs_by_part):
        per = max(1, budget // max(len(parts), 1)); pool = set()
        for p in parts:
            idxs = docs_by_part[p]
            if len(idxs):
                pool.update(int(x) for x in idxs[np.argsort(-qsim[qi, idxs])[:per]])
        return pool

    spanned = []; mh_frac = []
    orc = {("hard", b): [] for b in BUDGETS}; orc.update({("ov", b): [] for b in BUDGETS})
    pred = {(kind, P, b): [] for kind in ("hard", "ov") for P in TOPP for b in BUDGETS}
    pred_mh = {(P, b): [] for P in TOPP for b in BUDGETS}
    for qi in range(len(test)):
        g = gold[qi]
        if not g:
            continue
        spanned.append(len(set(int(hard[x]) for x in g)))
        hops = _hopset(seed[qi], g, adj, 2)
        mh = [x for x in g if hops.get(x, 9) >= 2]
        mh_frac.append(len(mh) / len(g))
        gp_hard = sorted(set(int(hard[x]) for x in g))
        gp_ov = sorted(set(p for x in g for p in mem[x]))
        for b in BUDGETS:
            orc[("hard", b)].append(1.0 if set(g) <= pool_from(gp_hard, qi, b, docs_hard) else 0.0)
            orc[("ov", b)].append(1.0 if set(g) <= pool_from(gp_ov, qi, b, docs_ov) else 0.0)
        pranked = [cpids[j] for j in np.argsort(-qcent[qi])]
        for P in TOPP:
            pp = pranked[:P]
            for b in BUDGETS:
                ph = pool_from(pp, qi, b, docs_hard); po = pool_from(pp, qi, b, docs_ov)
                pred[("hard", P, b)].append(1.0 if set(g) <= ph else 0.0)
                pred[("ov", P, b)].append(1.0 if set(g) <= po else 0.0)
                if mh:
                    pred_mh[(P, b)].append(1.0 if set(mh) <= po else 0.0)

    def m(x):
        return round(float(np.mean(x)) * 100, 2) if x else None
    out = {"dataset": dataset, "n_test": len(spanned), "n_partitions": npart,
           "avg_hard_partitions_spanned": round(float(np.mean(spanned)), 2),
           "pct_multihop_golds": round(float(np.mean(mh_frac)) * 100, 1),
           "oracle_fullcov_hard": {f"@{b}": m(orc[("hard", b)]) for b in BUDGETS},
           "oracle_fullcov_overlap": {f"@{b}": m(orc[("ov", b)]) for b in BUDGETS},
           "pred_fullcov_hard": {f"top{P}@{b}": m(pred[("hard", P, b)]) for P in TOPP for b in BUDGETS},
           "pred_fullcov_overlap": {f"top{P}@{b}": m(pred[("ov", P, b)]) for P in TOPP for b in BUDGETS},
           "pred_fullcov_overlap_MULTIHOP": {f"top{P}@{b}": m(pred_mh[(P, b)]) for P in TOPP for b in BUDGETS}}
    os.makedirs(os.path.join("data", "ukb_storage", dataset, "results", "L1"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "partition_probe.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] golds span {out['avg_hard_partitions_spanned']} hard-parts, {out['pct_multihop_golds']}% golds 2+hop | "
             f"ORACLE hard {out['oracle_fullcov_hard']} vs overlap {out['oracle_fullcov_overlap']} | "
             f"PRED hard {out['pred_fullcov_hard']} vs overlap {out['pred_fullcov_overlap']} | "
             f"PRED-overlap MULTIHOP {out['pred_fullcov_overlap_MULTIHOP']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Partition-reachability probe: does OVERLAP make multi-hop golds routable? (no training)")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean"])
    p.add_argument("--limit", type=int, default=800)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 PARTITION PROBE (overlap): {ds.upper()} =====")
        try:
            run(ds, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
