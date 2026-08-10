"""
Pool narrowing: from the huge overlap pool to a small candidate set for L2/L3.
==============================================================================
overlap1+knn1 routes to ~33k docs (top-20 overlapped partitions) — great coverage
but far too many to rerank. This narrows that pool with a cheap dense score and
measures how FEW candidate nodes still hold the golds, and how much a 1-hop
graph-traversal augmentation recovers on top.

Per query: L1 route -> top-K partitions -> pool. Dense-score the pool vs the
query, keep top-N (the narrowed candidate set). Optionally augment with the 1-hop
graph neighbours of the candidates (L3). Report, per N: avg candidate-set size
(with/without traversal), gt_recall and FullCov. The sweep answers "how many
candidate nodes do we need" and "does traversal let us use a smaller N".
Router seed pinned for reproducibility. Writes results/pool_narrow/{ds}.json.
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
from src.experiments.l3_recovery import _neighbors, _reach

log = logging.getLogger("experiments.pool_narrow")


def run_dataset(dataset, config="overlap1+knn1", K=20, Ns=(50, 100, 200, 500, 1000),
                epochs=100, limit=0, seed=42, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    log.info(f"===== POOL-NARROW: {dataset.upper()} config={config} K={K} =====")
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
    model, _, final_state, Cg = _train(
        engine, C, splits, split_embs, device, tau, hn_k, epochs,
        os.path.join("logs", dataset, f"poolnarrow_{config.replace('+','_')}"), config, "kl")
    model.load_state_dict(final_state); model.eval()

    test, test_e = splits["test"], split_embs["test"]
    ranked = []
    with torch.no_grad():
        for i in range(0, len(test_e), 256):
            proj = F.normalize(model(torch.tensor(test_e[i:i+256], dtype=torch.float32, device=device)), dim=-1)
            ranked.extend(torch.argsort(-(proj @ Cg.T), dim=1)[:, :K].cpu().tolist())
    rev = {p: set() for p in range(npart)}
    for nid, pids in membership.items():
        for p in pids:
            rev[p].add(nid)
    nb = _neighbors(engine)

    full_pool_sizes = []
    acc = {N: {"cand": [], "cand_trav": [], "gtr": [], "gtr_trav": [], "fc": [], "fc_trav": []} for N in Ns}
    for qi, (_, _, golds) in enumerate(test):
        golds = [g for g in golds if g in membership]
        if not golds:
            continue
        pool = set()
        for p in ranked[qi]:
            pool |= rev[p]
        full_pool_sizes.append(len(pool))
        pidx = [id2idx[d] for d in pool if d in id2idx]
        pool_docs = [d for d in pool if d in id2idx]
        sims = node_vecs[pidx] @ test_e[qi]
        order = np.argsort(-sims)
        ranked_pool = [pool_docs[j] for j in order.tolist()]
        for N in Ns:
            cand = ranked_pool[:N]
            frontier = _reach(cand, nb, 1)
            final = set(cand) | frontier
            def cov(s):
                c = [g for g in golds if g in s]
                return len(c) / len(golds), 1.0 if len(c) == len(golds) else 0.0
            g0, f0 = cov(set(cand)); g1, f1 = cov(final)
            acc[N]["cand"].append(len(cand)); acc[N]["cand_trav"].append(len(final))
            acc[N]["gtr"].append(g0); acc[N]["gtr_trav"].append(g1)
            acc[N]["fc"].append(f0); acc[N]["fc_trav"].append(f1)

    def m(x): return round(float(np.mean(x)) * 100, 2)
    def a(x): return round(float(np.mean(x)), 1)
    out = {"dataset": dataset, "config": config, "K": K, "seed": seed,
           "avg_full_pool_docs": a(full_pool_sizes), "n_test": len(full_pool_sizes), "sweep": []}
    for N in Ns:
        d = acc[N]
        out["sweep"].append({
            "N": N, "avg_candidates": a(d["cand"]), "avg_candidates_with_traversal": a(d["cand_trav"]),
            "gt_recall": m(d["gtr"]), "gt_recall_with_traversal": m(d["gtr_trav"]),
            "full_coverage": m(d["fc"]), "full_coverage_with_traversal": m(d["fc_trav"]),
        })
        s = out["sweep"][-1]
        log.info(f"  N={N:>4}: cand={s['avg_candidates']}(+trav {s['avg_candidates_with_traversal']}) "
                 f"gtR={s['gt_recall']}->{s['gt_recall_with_traversal']} "
                 f"FCov={s['full_coverage']}->{s['full_coverage_with_traversal']}")
    log.info(f"  (full L1 pool was {out['avg_full_pool_docs']} docs)")
    out_dir = os.path.join("results", "pool_narrow")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_{config.replace('+','_')}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/pool_narrow/{dataset}_{config.replace('+','_')}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Pool narrowing: overlap pool -> small candidate set (+traversal).")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--config", default="overlap1+knn1")
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--Ns", nargs="+", type=int, default=[50, 100, 200, 500, 1000])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        run_dataset(ds, config=a.config, K=a.K, Ns=tuple(a.Ns), epochs=a.epochs, limit=a.limit, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
