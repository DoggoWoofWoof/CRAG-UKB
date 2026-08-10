"""
Partition-design ablation: nodes/partition + METIS edge set, across datasets.
=============================================================================
The partition granularity and the graph METIS cuts on are the *substrate* every
other Level-1 result sits on, so they must be chosen by measurement, not guessed.
This sweeps, per dataset, training-free (raw dense query -> partition centroid,
which isolates partition QUALITY from router quality):
  - edge set: structural-only (node.neighbors: title/KB/adjacency) vs
              structural+kNN (adds the semantic synthetic edges)
  - granularity: target docs/partition in {100,250,500,1000,2000}
Metric is the FullCov-vs-candidate-pool frontier (fair across granularities:
more partitions => smaller pool/partition but need more in top-K). We interpolate
FullCov at fixed pool budgets (2%/5%/10% of corpus) so configs are comparable at
equal retrieval cost. Writes results/partition_ablation/{dataset}.json.
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

log = logging.getLogger("experiments.partition_ablation")
KSWEEP = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 130, 200, 320]


def _partition(adj, n, target):
    nparts = max(2, n // target)
    if sum(len(a) for a in adj) == 0:                     # no edges -> contiguous chunks
        return [i // max(1, n // nparts) for i in range(n)]
    import pymetis
    _, mem = pymetis.part_graph(nparts, adjacency=[sorted(a) for a in adj])
    return mem


def _frontier(q, nv, mem, golds_idx):
    npart = int(max(mem)) + 1
    part_of = np.asarray(mem)
    sizes = np.bincount(part_of, minlength=npart).astype(np.float64)
    C = np.zeros((npart, nv.shape[1]), dtype=np.float32)
    for i, p in enumerate(part_of):
        C[p] += nv[i]
    faiss.normalize_L2(C)
    ranked = np.argsort(-(q @ C.T), axis=1)
    ks = [k for k in KSWEEP if k <= npart]
    covs = {k: [] for k in ks}; pools = {k: [] for k in ks}
    for qi in range(q.shape[0]):
        gp = set(int(part_of[g]) for g in golds_idx[qi])
        if not gp:
            continue
        r = ranked[qi]; cum = np.cumsum(sizes[r])
        for k in ks:
            covs[k].append(1.0 if gp <= set(r[:k].tolist()) else 0.0)
            pools[k].append(float(cum[k - 1]))
    fr = [(k, round(float(np.mean(pools[k])), 0), round(float(np.mean(covs[k])) * 100, 2)) for k in ks]
    return npart, fr


def _fcov_at_pool(frontier, pool_budget):
    xs = [p for _, p, _ in frontier]; ys = [c for _, _, c in frontier]
    if not xs:
        return None
    return round(float(np.interp(pool_budget, xs, ys)), 2)


def run(dataset, targets=(100, 250, 500, 1000, 2000)):
    engine = CoreEngine(source=dataset)
    nv = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(nv)
    id2idx = engine.node_id_to_idx
    n = len(engine.nodes)
    struct = [set() for _ in range(n)]; knn = [set() for _ in range(n)]
    for i, node in enumerate(engine.nodes):
        for nb in node.neighbors:
            j = id2idx.get(nb)
            if j is not None and j != i:
                struct[i].add(j); struct[j].add(i)
        for nb in node.metadata.get("synthetic_neighbors", []):
            j = id2idx.get(nb)
            if j is not None and j != i:
                knn[i].add(j); knn[j].add(i)
    sp = _splits(engine, _hard_membership(engine)); test = sp["test"]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    golds_idx = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]

    budgets = [round(n * f) for f in (0.02, 0.05, 0.10)]
    out = {"dataset": dataset, "n_docs": n, "n_test": len(test), "pool_budgets": budgets, "configs": []}
    for eset, adj in [("structural", struct),
                      ("structural+knn", [struct[i] | knn[i] for i in range(n)])]:
        struct_edges = sum(len(a) for a in adj) // 2
        for tgt in targets:
            npart, fr = _frontier(q, nv, _partition(adj, n, tgt), golds_idx)
            at = {f"fcov@{int(b)}docs({p}%)": _fcov_at_pool(fr, b)
                  for b, p in zip(budgets, (2, 5, 10))}
            out["configs"].append({"edge_set": eset, "target_per_partition": tgt, "npart": npart,
                                    "edges": struct_edges, "fcov_at_pool": at, "frontier": fr})
            log.info(f"  [{dataset} {eset:14s} tgt={tgt:>4} npart={npart:>4}] {at}")

    os.makedirs("results/partition_ablation", exist_ok=True)
    with open(f"results/partition_ablation/{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/partition_ablation/{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Partition-design ablation (granularity + edge set).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa", "squad"])
    p.add_argument("--targets", nargs="+", type=int, default=[100, 250, 500, 1000, 2000])
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== PARTITION ABLATION: {ds.upper()} =====")
        run(ds, targets=tuple(a.targets))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
