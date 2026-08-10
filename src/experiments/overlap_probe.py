"""
Overlapped-partition probe (training-free structural pre-ablation).
===================================================================
Question: is Level-1 coverage bottlenecked by the HARD (single-membership)
partition assignment? i.e. a gold doc lives in exactly one partition, so if
routing misses that partition the doc is unreachable. Overlapping membership
lets a doc belong to several partitions, giving more chances to be covered — at
the cost of a bigger candidate pool ("explosion").

This is deliberately minimal + local + training-free: routing still uses the
EXISTING frozen centroids (raw dense query -> top-K centroids); only MEMBERSHIP
is overlapped. For overlap level m, each doc belongs to its METIS partition PLUS
its m nearest centroids. m=0 reproduces the current hard-partition baseline.

Reports, per overlap level and per K: FullCov@K, gt_recall@K, and the explosion
metrics (mean memberships/doc, mean candidate-pool size). Writes
results/overlap_probe/{dataset}_overlap.json. No checkpoints, no training.
"""
import os
import json
import logging
import argparse
from collections import defaultdict

import numpy as np

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.evaluation.benchmark_partition_selection import _get_split_queries, COVERAGE_K_VALUES

log = logging.getLogger("experiments.overlap_probe")


def _reconstruct(index):
    return np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype=np.float32)


def _build_membership(engine, node_vecs, centroids, centroid_pids, overlap):
    """node_id -> set(pid): METIS partition + `overlap` nearest centroids."""
    import faiss
    nv = node_vecs.copy(); faiss.normalize_L2(nv)
    cv = centroids.copy(); faiss.normalize_L2(cv)
    membership = {}
    metis = {nid: int(p) for nid, p in engine.partition_map.items()}
    if overlap > 0:
        sims = nv @ cv.T                                   # (Ndoc, Ncent)
        topm = np.argsort(-sims, axis=1)[:, :overlap]
    for i, node in enumerate(engine.nodes):
        s = set()
        if node.node_id in metis:
            s.add(metis[node.node_id])
        if overlap > 0:
            s.update(int(centroid_pids[j]) for j in topm[i])
        if s:
            membership[node.node_id] = s
    return membership


def _reverse(membership):
    rev = defaultdict(set)
    for nid, pids in membership.items():
        for p in pids:
            rev[p].add(nid)
    return rev


def run_dataset(dataset, overlaps=(0, 1, 2), limit=0):
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    if engine.centroid_index is None:
        log.warning(f"No centroid index for {dataset}; skipping.")
        return None

    node_vecs = _reconstruct(engine.node_index)
    centroids = _reconstruct(engine.centroid_index)
    centroid_pids = [int(p) for p in engine.centroid_pids]

    splits = _get_split_queries(engine, dataset=dataset)
    test = splits.get("test", [])
    if limit:
        test = test[:limit]
    if not test:
        log.warning(f"No test queries for {dataset}; skipping.")
        return None
    q_embs = encoder.encode([q.content for q, _ in test]).astype("float32")
    import faiss
    faiss.normalize_L2(q_embs)
    maxk = max(COVERAGE_K_VALUES)
    # Precompute each query's ranked partition list once (routing is overlap-independent).
    ranked = []
    for i in range(len(test)):
        res = engine.search_centroids(q_embs[i:i + 1], k=maxk)
        ranked.append([pid for pid, _ in res])

    n_docs = len(engine.nodes)
    results = {}
    for m in overlaps:
        membership = _build_membership(engine, node_vecs, centroids, centroid_pids, m)
        rev = _reverse(membership)
        mem_per_doc = np.mean([len(membership.get(nd.node_id, set())) for nd in engine.nodes]) if n_docs else 0.0

        fc = {k: [] for k in COVERAGE_K_VALUES}
        gtr = {k: [] for k in COVERAGE_K_VALUES}
        pool20 = []
        for qi, (q_node, _) in enumerate(test):
            golds = [nid for nid in q_node.neighbors if nid in membership]
            if not golds:
                continue
            top = ranked[qi]
            for k in COVERAGE_K_VALUES:
                topk = set(top[:k])
                covered = [g for g in golds if membership[g] & topk]
                fc[k].append(1.0 if len(covered) == len(golds) else 0.0)
                gtr[k].append(len(covered) / len(golds))
            # explosion: candidate pool = union of docs whose membership hits top-20
            top20 = top[:20]
            pool = set()
            for p in top20:
                pool |= rev.get(p, set())
            pool20.append(len(pool))

        label = "hard(m=0)" if m == 0 else f"overlap(m={m})"
        results[label] = {
            "overlap": m,
            "mean_memberships_per_doc": round(float(mem_per_doc), 3),
            "mean_pool@20": round(float(np.mean(pool20)), 1) if pool20 else 0.0,
            "explosion_x_vs_hard": None,   # filled below
            **{f"full_coverage@{k}": round(float(np.mean(fc[k])) * 100, 2) for k in COVERAGE_K_VALUES},
            **{f"gt_recall@{k}": round(float(np.mean(gtr[k])) * 100, 2) for k in COVERAGE_K_VALUES},
            "n_test": len(pool20),
        }
        log.info(f"  [{label}] mem/doc={results[label]['mean_memberships_per_doc']} "
                 f"pool@20={results[label]['mean_pool@20']} "
                 f"FCov@20={results[label]['full_coverage@20']}% "
                 f"FCov@50={results[label]['full_coverage@50']}% "
                 f"gtR@20={results[label]['gt_recall@20']}%")

    base_pool = results.get("hard(m=0)", {}).get("mean_pool@20") or 1.0
    for lbl, r in results.items():
        r["explosion_x_vs_hard"] = round(r["mean_pool@20"] / base_pool, 2) if base_pool else None

    out_dir = os.path.join("results", "overlap_probe")
    os.makedirs(out_dir, exist_ok=True)
    payload = {"dataset": dataset, "routing": "raw_dense->centroids (training-free)", "results": results}
    with open(os.path.join(out_dir, f"{dataset}_overlap.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Saved results/overlap_probe/{dataset}_overlap.json")
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description="Overlapped-partition coverage probe (training-free).")
    p.add_argument("--datasets", nargs="+", default=["2wiki", "musique", "squad", "metaqa"])
    p.add_argument("--overlaps", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== OVERLAP PROBE: {ds.upper()} =====")
        run_dataset(ds, overlaps=tuple(a.overlaps), limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
