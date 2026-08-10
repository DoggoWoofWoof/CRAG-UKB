"""
Multi-prototype partition routing (improve coverage WITHOUT increasing overlap).
=================================================================================
At fixed membership the candidate pool is fixed, so the only lever left is ranking
the required partitions higher. We've saturated the query side (MLP, HNM, encoder
fine-tune). The untouched side is the routing TARGET: each partition is a single
frozen degree-weighted mean centroid — a lossy summary of a semantically diverse
(especially overlapped) partition. A required partition can rank low because its
MEAN is far from the query even though the gold doc inside it is close.

Fix, at zero extra pool: represent each partition by c sub-centroids (k-means over
its member docs) and score the partition by the MAX similarity over its
sub-centroids. Same docs, same pool — but a partition surfaces if ANY sub-region
matches. This is a training-free probe (raw dense query -> max-sim sub-centroid);
if c>1 lifts FullCov at identical membership, the routing target is a real
bottleneck and a trained multi-prototype router is worth building. Sweeps c and
reports FullCov@K per c (pool is constant across c by construction).
Writes results/multiproto/{dataset}_{config}.json.
"""
import os
import json
import logging
import argparse

import numpy as np

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits
from src.experiments.adaptive_k import _build
from src.evaluation.benchmark_partition_selection import COVERAGE_K_VALUES

log = logging.getLogger("experiments.multiproto")


def _subcentroids(vecs, c):
    """c sub-centroids for one partition's member doc vectors (k-means; mean if c==1)."""
    import faiss
    n = vecs.shape[0]
    if n == 0:
        return np.zeros((0, vecs.shape[1] if vecs.ndim == 2 else 1), dtype=np.float32)
    if c <= 1 or n <= c:
        # c==1 -> plain mean; n<=c -> each doc is its own prototype
        return vecs.copy() if (n <= c and c > 1) else vecs.mean(axis=0, keepdims=True)
    km = faiss.Kmeans(vecs.shape[1], c, niter=20, seed=42, verbose=False)
    km.train(vecs.astype(np.float32))
    return km.centroids.reshape(c, vecs.shape[1]).astype(np.float32)


def run_dataset(dataset, config="overlap1", protos=(1, 2, 4, 8), limit=0):
    import faiss
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1

    membership = _build(engine, node_vecs, config)
    id2idx = {n.node_id: i for i, n in enumerate(engine.nodes)}
    part_doc_idx = [[] for _ in range(npart)]
    for nid, pids in membership.items():
        if nid in id2idx:
            for p in pids:
                if 0 <= p < npart:
                    part_doc_idx[p].append(id2idx[nid])

    splits = _splits(engine, membership)
    test = splits["test"][:limit] if limit else splits["test"]
    q = encoder.encode([qn.content for qn, _, _ in test]).astype("float32")
    faiss.normalize_L2(q)
    gold_docs_list = [golds for _, _, golds in test]
    maxk = max(COVERAGE_K_VALUES)

    results = {}
    for c in protos:
        # build sub-centroids + column->partition map
        blocks, col2part = [], []
        for p in range(npart):
            vecs = node_vecs[part_doc_idx[p]] if part_doc_idx[p] else np.zeros((0, node_vecs.shape[1]), dtype=np.float32)
            sc = _subcentroids(vecs, c)
            if sc.shape[0]:
                blocks.append(sc); col2part.extend([p] * sc.shape[0])
        SC = np.concatenate(blocks, axis=0).astype("float32")
        faiss.normalize_L2(SC)
        col2part = np.array(col2part)

        # per-query partition score = max sim over that partition's sub-centroids
        sims = q @ SC.T                                       # (n, total_sub)
        part_score = np.full((len(test), npart), -1e9, dtype=np.float32)
        for p in range(npart):
            cols = np.where(col2part == p)[0]
            if cols.size:
                part_score[:, p] = sims[:, cols].max(axis=1)
        ranked = np.argsort(-part_score, axis=1)[:, :max(maxk, npart)]

        fc = {k: [] for k in COVERAGE_K_VALUES}
        gtr = {k: [] for k in COVERAGE_K_VALUES}
        for qi in range(len(test)):
            top = ranked[qi]
            gms = [membership[g] for g in gold_docs_list[qi] if g in membership]
            for k in COVERAGE_K_VALUES:
                topk = set(top[:k].tolist())
                covered = [ms for ms in gms if ms & topk]
                fc[k].append(1.0 if gms and len(covered) == len(gms) else 0.0)
                gtr[k].append(len(covered) / len(gms) if gms else 0.0)
        avg_sub = round(SC.shape[0] / npart, 2)
        results[f"c{c}"] = {
            "protos": c, "avg_subcentroids_per_partition": avg_sub,
            **{f"full_coverage@{k}": round(float(np.mean(fc[k])) * 100, 2) for k in COVERAGE_K_VALUES},
            **{f"gt_recall@{k}": round(float(np.mean(gtr[k])) * 100, 2) for k in COVERAGE_K_VALUES},
            "n_test": len(test),
        }
        log.info(f"  [{config} c={c}] sub/part={avg_sub} FCov@10={results[f'c{c}']['full_coverage@10']}% "
                 f"FCov@20={results[f'c{c}']['full_coverage@20']}% gtR@20={results[f'c{c}']['gt_recall@20']}%")

    out_dir = os.path.join("results", "multiproto")
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{dataset}_{config.replace('+', '_')}.json"
    with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "config": config, "routing": "raw_dense->max-sim subcentroid (training-free)",
                   "results": results}, f, indent=2)
    log.info(f"Saved results/multiproto/{fn}")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Multi-prototype partition routing probe (no extra overlap).")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--configs", nargs="+", default=["overlap1", "overlap1+knn1"])
    p.add_argument("--protos", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    for ds in a.datasets:
        for cfg in a.configs:
            log.info(f"===== MULTIPROTO: {ds.upper()} config={cfg} protos={a.protos} =====")
            run_dataset(ds, config=cfg, protos=tuple(a.protos), limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
