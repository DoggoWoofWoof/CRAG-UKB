"""
Test partition-specificity (IDF) vote re-weighting.
===================================================
Blocker finding: ~2/3 of musique/2wiki misses are OUTRANKED — the gold partition IS voted but ranked
~31, out-voted by PROMISCUOUS "hub" partitions that (via overlap membership) collect votes from nearly
every retrieved node. Fix: weight each partition's vote by its SPECIFICITY — down-weight partitions that
appear in many nodes' membership. freq[p] = #nodes with p in their overlap membership; idf[p] = a
decreasing function of freq[p]. Apply to the sum/max vote matrices before RRF-ranking.

Compares dense-voting FullCov@20 across idf variants on all 5 datasets (per-dataset, memory-safe).
Default encoder gte_qwen (cached queries). Local, read-only.
"""
import os
import argparse
import logging

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.experiments.encoder_swap import load_docs_and_encoder
from src.experiments.overlap_retrain import _splits, _hard_membership, _onehop_membership
from src.experiments.l1_rerank100 import _feats, _rr, _fullcov, MAXK

log = logging.getLogger("experiments.l1_idf_test")

DATASETS = ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]


def _idf_variants(freq, N):
    inv = N / np.maximum(freq, 1.0)
    return {
        "none":    np.ones_like(freq),
        "log":     np.log1p(inv),
        "sqrt":    np.sqrt(inv),
        "linear":  inv,
    }


def run(datasets=None, subdir="gte_qwen", limit=8000, topn=200):
    datasets = datasets or DATASETS
    rows = {}
    for d in datasets:
        eng = CoreEngine(source=d, index_subdir=subdir)
        X, eq, tag = load_docs_and_encoder(eng, d, subdir); X = X.astype("float32")
        n = X.shape[0]; npart = max(int(p) for p in eng.partition_map.values()) + 1
        id2idx = eng.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
        hard = np.array([int(eng.partition_map.get(idx2id[i], -1)) for i in range(n)])
        mem = _onehop_membership(eng)
        mem_idx = [sorted(mem.get(idx2id[i], {int(hard[i])})) for i in range(n)]
        # partition promiscuity: how many nodes carry p in their overlap membership
        freq = np.zeros(npart, dtype=np.float64)
        for m in mem_idx:
            for p in m:
                freq[p] += 1.0
        index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
        sp = _splits(eng, _hard_membership(eng)); test = sp["test"][:limit]
        cache = os.path.join("data", "ukb_storage", d, subdir, "queries_test.npy") if subdir else None
        Q = (np.load(cache)[:len(test)].astype("float32") if cache and os.path.exists(cache)
             else eq([nd.content for nd, _, _ in test]).astype("float32"))
        _, I = index.search(Q, MAXK)
        S, M = _feats(I, mem_idx, npart, topn=topn)
        gpl = [[mem_idx[id2idx[g]] for g in golds if g in id2idx] for _, _, golds in test]

        variants = _idf_variants(freq, n)
        rows[d] = {}
        for name, idf in variants.items():
            score = _rr(S * idf[None, :]) + _rr(M * idf[None, :])
            rows[d][name] = _fullcov(score, gpl, npart)[20]
        log.info(f"[{d}] " + "  ".join(f"{k}={v:.2f}" for k, v in rows[d].items()))
        del X, index, S, M, Q, I; import gc; gc.collect()

    names = ["none", "log", "sqrt", "linear"]
    print(f"\n=== IDF vote re-weighting (dense-voting FullCov@20, {subdir}) ===")
    print(f"{'dataset':16s} " + " ".join(f"{k:>7s}" for k in names))
    for d in datasets:
        print(f"{d:16s} " + " ".join(f"{rows[d][k]:7.2f}" for k in names))
    print(f"{'MEAN':16s} " + " ".join(f"{np.mean([rows[d][k] for d in datasets]):7.2f}" for k in names))
    print(f"{'#>=95':16s} " + " ".join(f"{sum(1 for d in datasets if rows[d][k]>=95):7d}" for k in names))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir)


if __name__ == "__main__":
    main()
