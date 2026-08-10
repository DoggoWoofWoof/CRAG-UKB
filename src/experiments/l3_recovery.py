"""
L3 graph-traversal recovery of L1 FullCov failures.
====================================================
Claim to test: the gold docs an overlapped router MISSES at L1 are the
reasoning-chain partners of the golds it FOUND — i.e. one graph hop away from a
retrieved doc — so L3 traversal recovers them. This quantifies exactly that.

Pipeline: train the overlap1+knn1 router (KL+HNM), route test queries to top-K
partitions -> candidate pool. For each query split golds into covered (partition
in top-K) vs missed. Then expand by graph traversal (original node.neighbors)
and measure how many missed golds become reachable within 1 / 2 hops, from two
anchor sets:
  - covered-golds  (recovery CEILING: assumes L2 surfaces the covered golds)
  - top-N pool docs by dense score (REALISTIC: L3 traverses from retrieved docs)
Reports L1 vs post-L3 gt_recall and FullCov. Writes results/l3_recovery/{ds}.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _centroids, _splits, _train, membership_ref, TAU, HNK
from src.experiments.adaptive_k import _build

log = logging.getLogger("experiments.l3_recovery")


def _neighbors(engine):
    docset = set(engine.node_id_to_idx)
    nb = {}
    for node in engine.nodes:
        nb[node.node_id] = [x for x in node.neighbors if x in docset]
    return nb


def _reach(anchors, nb, hops):
    """Set of doc ids reachable within `hops` graph hops from anchors (excluding anchors' own set growth per hop)."""
    frontier = set(anchors)
    reached = set()
    for _ in range(hops):
        nxt = set()
        for d in frontier:
            nxt.update(nb.get(d, ()))
        reached |= nxt
        frontier = nxt
    return reached


def run_dataset(dataset, config="overlap1+knn1", K=20, topN=20, epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    log.info(f"===== L3-RECOVERY: {dataset.upper()} config={config} K={K} =====")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    id2idx = engine.node_id_to_idx

    membership = _build(engine, node_vecs, config)
    membership_ref[config] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    model, best_state, final_state, Cg = _train(
        engine, C, splits, split_embs, device, tau, hn_k, epochs,
        os.path.join("logs", dataset, f"l3rec_{config.replace('+','_')}"), config, "kl")
    model.load_state_dict(final_state); model.eval()

    # route test
    test = splits["test"]; test_e = split_embs["test"]
    ranked = []
    with torch.no_grad():
        for i in range(0, len(test_e), 256):
            proj = F.normalize(model(torch.tensor(test_e[i:i+256], dtype=torch.float32, device=device)), dim=-1)
            ranked.extend(torch.argsort(-(proj @ Cg.T), dim=1)[:, :K].cpu().tolist())

    # reverse membership: partition -> set(doc ids)
    rev = {p: set() for p in range(npart)}
    for nid, pids in membership.items():
        for p in pids:
            rev[p].add(nid)
    nb = _neighbors(engine)

    # accumulators
    L1_frac, cov_ceil1, cov_ceil2, cov_real1, cov_real2 = [], [], [], [], []
    fc_L1 = fc_ceil1 = fc_ceil2 = fc_real1 = fc_real2 = 0
    n = 0
    pool_sizes, front1_sizes, front2_sizes = [], [], []   # narrowing diagnostics
    for qi, (_, _, golds) in enumerate(test):
        golds = [g for g in golds if g in membership]
        if not golds:
            continue
        n += 1
        topk = set(ranked[qi])
        covered = [g for g in golds if membership[g] & topk]
        missed = [g for g in golds if g not in set(covered)]
        L1_frac.append(len(covered) / len(golds))
        if not missed:
            for lst in (cov_ceil1, cov_ceil2, cov_real1, cov_real2): lst.append(1.0)
            fc_L1 += 1; fc_ceil1 += 1; fc_ceil2 += 1; fc_real1 += 1; fc_real2 += 1
            continue
        # realistic anchors: top-N pool docs by dense score to the query
        pool = set()
        for p in topk:
            pool |= rev[p]
        pool = list(pool)
        if pool:
            pidx = [id2idx[d] for d in pool if d in id2idx]
            q = test_e[qi]
            sims = node_vecs[pidx] @ q
            order = np.argsort(-sims)[:topN]
            top_docs = [pool[j] for j in order.tolist()]
        else:
            top_docs = []
        # recovery sets
        reach_c1 = _reach(covered, nb, 1); reach_c2 = reach_c1 | _reach(covered, nb, 2)
        reach_r1 = _reach(top_docs, nb, 1); reach_r2 = reach_r1 | _reach(top_docs, nb, 2)
        pool_sizes.append(len(pool)); front1_sizes.append(len(reach_r1)); front2_sizes.append(len(reach_r2))
        def frac(reach):
            rec = [m for m in missed if m in reach]
            return (len(covered) + len(rec)) / len(golds)
        f_c1, f_c2 = frac(reach_c1), frac(reach_c2)
        f_r1, f_r2 = frac(reach_r1), frac(reach_r2)
        cov_ceil1.append(f_c1); cov_ceil2.append(f_c2); cov_real1.append(f_r1); cov_real2.append(f_r2)
        fc_L1 += 0
        fc_ceil1 += 1 if f_c1 >= 0.999 else 0
        fc_ceil2 += 1 if f_c2 >= 0.999 else 0
        fc_real1 += 1 if f_r1 >= 0.999 else 0
        fc_real2 += 1 if f_r2 >= 0.999 else 0

    def pct(x): return round(float(np.mean(x)) * 100, 2)
    out = {
        "dataset": dataset, "config": config, "K": K, "topN_anchor": topN, "n_test": n,
        "gt_recall": {
            "L1": pct(L1_frac),
            "L3_ceiling_1hop": pct(cov_ceil1), "L3_ceiling_2hop": pct(cov_ceil2),
            "L3_realistic_1hop": pct(cov_real1), "L3_realistic_2hop": pct(cov_real2),
        },
        "full_coverage": {
            "L1": round(fc_L1 / n * 100, 2),
            "L3_ceiling_1hop": round(fc_ceil1 / n * 100, 2), "L3_ceiling_2hop": round(fc_ceil2 / n * 100, 2),
            "L3_realistic_1hop": round(fc_real1 / n * 100, 2), "L3_realistic_2hop": round(fc_real2 / n * 100, 2),
        },
        "narrowing": {   # is L3 a precise add or a wide net?
            "avg_L1_pool_docs": round(float(np.mean(pool_sizes)), 1) if pool_sizes else 0,
            "avg_L3_frontier_1hop_docs": round(float(np.mean(front1_sizes)), 1) if front1_sizes else 0,
            "avg_L3_frontier_2hop_docs": round(float(np.mean(front2_sizes)), 1) if front2_sizes else 0,
            "note": "frontier = unique docs reached by traversal from the 20 anchor docs (added beyond routing)",
        },
    }
    out_dir = os.path.join("results", "l3_recovery")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_{config.replace('+','_')}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"  gt_recall  L1={out['gt_recall']['L1']} -> ceil1={out['gt_recall']['L3_ceiling_1hop']} "
             f"real1={out['gt_recall']['L3_realistic_1hop']} real2={out['gt_recall']['L3_realistic_2hop']}")
    log.info(f"  FullCov    L1={out['full_coverage']['L1']} -> ceil1={out['full_coverage']['L3_ceiling_1hop']} "
             f"real1={out['full_coverage']['L3_realistic_1hop']} real2={out['full_coverage']['L3_realistic_2hop']}")
    log.info(f"Saved results/l3_recovery/{dataset}_{config.replace('+','_')}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 graph-traversal recovery of L1 FullCov failures.")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--config", default="overlap1+knn1")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--topN", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        run_dataset(ds, config=a.config, K=a.K, topN=a.topN, epochs=a.epochs, limit=a.limit, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
