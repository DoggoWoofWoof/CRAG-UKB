"""
Adaptive-K routing probe (Level-1 efficiency lever).
=====================================================
Fixed-K routing spends the SAME retrieval budget on every query, but queries
differ in how many partitions they actually need (2wiki: avg 1.76 gold
partitions, up to 4). Single-gold queries are covered at tiny K; the pool is
wasted on them. Multi-gold queries — the ones that FAIL FullCov@20 — are
starved. Adaptive-K reallocates: K_q grows with the query's gold-count.

This measures the ORACLE ceiling (upper bound on any gold-COUNT predictor): set
K_q = clip(ceil(s * g_q)) using the true number of gold partitions g_q, sweep
the scale s, and trace the (avg candidate-pool, FullCov) frontier. Compared head
to head against the fixed-K frontier on the SAME partitions and router. If the
adaptive frontier dominates (more coverage at equal pool), a predictor is worth
building; if it doesn't, adaptive-K is foreclosed. We evaluate on `hard`
(isolates the lever) and overlap1 (does it help on top of overlap), and drop the
overlap1 fixed-K=20 point in as the "blanket overlap" reference.

Reuses overlap_retrain's membership builders + centroid rebuild + router train
(no fixed-K coupling); training-free routing is not used — we rank with the
trained MLP so the ranking matches the real system. Writes results/adaptive_k/.
"""
import os
import json
import math
import logging
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import (
    _reconstruct, _centroids, _splits, _train, membership_ref,
    _hard_membership, _onehop_membership, _knn_membership, _twohop_membership,
    _synhop_membership, TAU, HNK,
)

log = logging.getLogger("experiments.adaptive_k")


def _atom(engine, node_vecs, name):
    if name == "hard":
        return _hard_membership(engine)
    if name == "overlap1":
        return _onehop_membership(engine)
    if name == "overlap2":
        return _twohop_membership(engine)
    if name == "syn1":
        return _synhop_membership(engine)
    if name.startswith("knn"):
        return _knn_membership(engine, node_vecs, int(name[3:]))
    raise ValueError(f"unknown membership atom {name!r}")


def _build(engine, node_vecs, cfg):
    atoms = [_atom(engine, node_vecs, a) for a in cfg.split("+")]
    if len(atoms) == 1:
        return atoms[0]
    keys = set().union(*[set(a) for a in atoms])
    return {k: set().union(*[a.get(k, set()) for a in atoms]) for k in keys}


def _rank_test(model, Cg, test_embs, device):
    """Full partition ranking per test query from the trained router."""
    ranked = []
    with torch.no_grad():
        for i in range(0, len(test_embs), 256):
            embs = torch.tensor(test_embs[i:i + 256], dtype=torch.float32, device=device)
            proj = F.normalize(model(embs), dim=-1)
            sims = proj @ Cg.T
            ranked.extend(torch.argsort(-sims, dim=1).cpu().tolist())
    return ranked


def _reverse(membership, npart):
    rev = [set() for _ in range(npart)]
    for nid, pids in membership.items():
        for p in pids:
            if 0 <= p < npart:
                rev[p].add(nid)
    return rev


def _point(ranked, golds_list, gmembership_list, rev, k_per_query):
    """(avg candidate-pool size in docs, FullCov %) for a per-query K policy."""
    pools, covs = [], []
    for qi in range(len(ranked)):
        k = max(1, int(k_per_query[qi]))
        topk = set(ranked[qi][:k])
        pool = set()
        for p in topk:
            pool |= rev[p]
        pools.append(len(pool))
        gm = gmembership_list[qi]
        covs.append(1.0 if gm and all(ms & topk for ms in gm) else 0.0)
    return round(float(np.mean(pools)), 1), round(float(np.mean(covs)) * 100, 2)


def run_dataset(dataset, configs=("hard", "overlap1"), epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    log.info(f"===== ADAPTIVE-K: {dataset.upper()} (tau={tau:g}, hn_k={hn_k}) configs={list(configs)} =====")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1

    fixed_ks = [1, 3, 5, 10, 15, 20, 30, 40, 50]
    scales = [1, 2, 3, 5, 8, 12, 20, 30]        # K_q = ceil(s * g_q)
    out = {"dataset": dataset, "npart": npart, "fixed_ks": fixed_ks, "scales": scales, "configs": {}}

    for cfg in configs:
        membership = _build(engine, node_vecs, cfg)
        membership_ref[cfg] = membership
        C, _ = _centroids(engine, node_vecs, membership, npart)
        splits = _splits(engine, membership)
        if limit:
            splits = {s: q[:limit] for s, q in splits.items()}
        if not splits["train"]:
            continue
        split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                      for s in splits if splits[s]}
        model, best_state, final_state, Cg = _train(
            engine, C, splits, split_embs, device, tau, hn_k, epochs,
            os.path.join("logs", dataset, f"adaptivek_{cfg}"), cfg, "kl")
        model.load_state_dict(final_state); model.eval()

        test = splits["test"]
        ranked = _rank_test(model, Cg, split_embs["test"], device)
        rev = _reverse(membership, npart)
        golds_list = [golds for _, _, golds in test]
        gmembership_list = [[membership[g] for g in golds if g in membership] for _, _, golds in test]
        gcount_list = [len(pids) for _, pids, _ in test]        # oracle gold-partition count g_q

        fixed = [{"K": k, **dict(zip(("avg_pool", "full_coverage"),
                  _point(ranked, golds_list, gmembership_list, rev, [k] * len(ranked))))}
                 for k in fixed_ks]
        adaptive = []
        for s in scales:
            kpq = [min(npart, math.ceil(s * max(1, g))) for g in gcount_list]
            ap, fc = _point(ranked, golds_list, gmembership_list, rev, kpq)
            adaptive.append({"scale": s, "avg_K": round(float(np.mean(kpq)), 2), "avg_pool": ap, "full_coverage": fc})

        out["configs"][cfg] = {
            "avg_gold_partitions": round(float(np.mean(gcount_list)), 3),
            "fixed_k": fixed, "adaptive_k": adaptive,
        }
        f20 = next(x["full_coverage"] for x in fixed if x["K"] == 20)
        p20 = next(x["avg_pool"] for x in fixed if x["K"] == 20)
        log.info(f"  [{cfg}] fixed K=20: FCov={f20}% pool={p20} | adaptive best: "
                 f"{max(adaptive, key=lambda a: a['full_coverage'])}")

    out_dir = os.path.join("results", "adaptive_k")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_adaptive_k.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/adaptive_k/{dataset}_adaptive_k.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Adaptive-K oracle-ceiling probe.")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--configs", nargs="+", default=["hard", "overlap1"],
                   help="Membership configs to evaluate (hard | overlap1 | knn<m> | a+b unions).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        run_dataset(ds, configs=tuple(a.configs), epochs=a.epochs, limit=a.limit, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
