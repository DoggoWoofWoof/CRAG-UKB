"""
Overlap metrics x partition size, per dataset (training-free).
==============================================================
Does the overlap lever (graph / centroid-kNN membership) help more or less at
different partition granularities? For each dataset x partition-size x overlap
config, builds the partition in-memory (METIS on structural+kNN), applies the
overlap membership, and measures the FullCov-vs-candidate-pool frontier
(pool-matched -> comparable across sizes/configs). Training-free (raw dense ->
centroid) to isolate structure from router. Writes
results/overlap_partsize/{dataset}.json.
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

log = logging.getLogger("experiments.overlap_by_partsize")
KSWEEP = [1, 2, 3, 5, 8, 13, 20, 30, 50, 80, 130, 200, 320, 500]
CONFIGS = ["hard", "overlap1", "knn1", "overlap1+knn1"]


def _metis(adj, n, target):
    nparts = max(2, n // target)
    if sum(len(a) for a in adj) == 0:
        return np.array([i // max(1, n // nparts) for i in range(n)])
    import pymetis
    _, mem = pymetis.part_graph(nparts, adjacency=[sorted(a) for a in adj])
    return np.asarray(mem)


def _membership(cfg, n, part_of, npart, struct, C_norm, nv_norm):
    """doc_idx -> set(partition ids), for the given overlap config on this partition."""
    mem = [{int(part_of[i])} for i in range(n)]
    parts = cfg.split("+")
    if "overlap1" in parts:
        for i in range(n):
            for j in struct[i]:
                mem[i].add(int(part_of[j]))
    for p in parts:
        if p.startswith("knn"):
            m = int(p[3:])
            topm = np.argsort(-(nv_norm @ C_norm.T), axis=1)[:, :m]
            for i in range(n):
                mem[i].update(int(x) for x in topm[i])
    return mem


def _eval(cfg, q, nv, part_of, npart, mem, golds_idx, budgets):
    sizes = np.bincount(part_of, minlength=npart).astype(np.float64)
    C = np.zeros((npart, nv.shape[1]), dtype=np.float32)
    for i, p in enumerate(part_of):
        C[p] += nv[i]
    faiss.normalize_L2(C)
    rev = [[] for _ in range(npart)]
    for i in range(len(mem)):
        for p in mem[i]:
            rev[p].append(i)
    rev_sizes = np.array([len(r) for r in rev], dtype=np.float64)   # docs per partition WITH overlap
    ranked = np.argsort(-(q @ C.T), axis=1)
    ks = [k for k in KSWEEP if k <= npart]
    covs = {k: [] for k in ks}; pools = {k: [] for k in ks}
    for qi in range(q.shape[0]):
        golds = golds_idx[qi]
        if not golds:
            continue
        r = ranked[qi]
        cum = np.cumsum(rev_sizes[r])                 # overlap pool grows with duplicated docs
        topset = set()
        prev = 0
        # per K coverage: gold covered if any of its membership parts in top-K
        gm = [mem[g] for g in golds]
        for k in ks:
            tk = set(r[:k].tolist())
            covs[k].append(1.0 if all(ms & tk for ms in gm) else 0.0)
            pools[k].append(float(cum[k - 1]))
    fr = [(k, round(float(np.mean(pools[k])), 0), round(float(np.mean(covs[k])) * 100, 2)) for k in ks]
    at = {}
    xs = [p for _, p, _ in fr]; ys = [c for _, _, c in fr]
    for b, lab in zip(budgets, ("2%", "5%", "10%")):
        at[f"fcov@{lab}"] = round(float(np.interp(b, xs, ys)), 2) if xs else None
    return at, fr


def run(dataset, targets=(100, 250, 500, 1000)):
    engine = CoreEngine(source=dataset)
    nv = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(nv)
    id2idx = engine.node_id_to_idx; n = len(engine.nodes)
    struct = [set() for _ in range(n)]; adj = [set() for _ in range(n)]
    for i, node in enumerate(engine.nodes):
        for nb in node.neighbors:
            j = id2idx.get(nb)
            if j is not None and j != i:
                struct[i].add(j); struct[j].add(i); adj[i].add(j); adj[j].add(i)
        for nb in node.metadata.get("synthetic_neighbors", []):
            j = id2idx.get(nb)
            if j is not None and j != i:
                adj[i].add(j); adj[j].add(i)                     # METIS uses structural+kNN
    sp = _splits(engine, _hard_membership(engine)); test = sp["test"]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    golds_idx = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    budgets = [round(n * f) for f in (0.02, 0.05, 0.10)]

    out = {"dataset": dataset, "n_docs": n, "pool_budgets": budgets, "grid": []}
    for tgt in targets:
        part_of = _metis(adj, n, tgt); npart = int(part_of.max()) + 1
        C0 = np.zeros((npart, nv.shape[1]), dtype=np.float32)
        for i, p in enumerate(part_of):
            C0[p] += nv[i]
        faiss.normalize_L2(C0)
        for cfg in CONFIGS:
            mem = _membership(cfg, n, part_of, npart, struct, C0, nv)
            at, _ = _eval(cfg, q, nv, part_of, npart, mem, golds_idx, budgets)
            mpd = round(float(np.mean([len(m) for m in mem])), 2)
            out["grid"].append({"target_per_partition": tgt, "npart": npart, "config": cfg,
                                 "mem_per_doc": mpd, "fcov_at_pool": at})
            log.info(f"  [{dataset} tgt={tgt:>4} {cfg:14s} mem/doc={mpd:>4}] {at}")

    os.makedirs("results/overlap_partsize", exist_ok=True)
    with open(f"results/overlap_partsize/{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/overlap_partsize/{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Overlap metrics x partition size, per dataset.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa", "squad"])
    p.add_argument("--targets", nargs="+", type=int, default=[100, 250, 500, 1000])
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== OVERLAP x PARTSIZE: {ds.upper()} =====")
        run(ds, targets=tuple(a.targets))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
