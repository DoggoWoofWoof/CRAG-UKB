"""
Paging tradeoff: if L2/L3 get ONLY L1's top-K partitions (not the full graph), what completeness do we
get vs how much of the graph do we page in?
  FullCov@K   : fraction of queries whose EVERY gold partition is in the top-K  (=answerable fully in the scope)
  docfrac@K   : avg fraction of the corpus that lives in the top-K partitions   (=working-set size for L3)
Also oracle FullCov@K (min-cover ceiling). Encoder gte_qwen, dense+overlap-voting. Local, read-only.
"""
import os
import argparse
import logging
from collections import Counter

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.experiments.encoder_swap import load_docs_and_encoder
from src.experiments.overlap_retrain import _splits, _hard_membership, _onehop_membership
from src.experiments.l1_rerank100 import _feats, _rr, MAXK

log = logging.getLogger("experiments.l1_paging")
DATASETS = ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]
KS = [20, 50, 100, 200]


def run(datasets=None, subdir="gte_qwen", limit=8000, topn=200):
    datasets = datasets or DATASETS
    out = {}
    for d in datasets:
        eng = CoreEngine(source=d, index_subdir=subdir)
        X, eq, tag = load_docs_and_encoder(eng, d, subdir); X = X.astype("float32")
        n = X.shape[0]; npart = max(int(p) for p in eng.partition_map.values()) + 1
        id2idx = eng.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
        hard = np.array([int(eng.partition_map.get(idx2id[i], -1)) for i in range(n)])
        mem = _onehop_membership(eng)
        mem_idx = [sorted(mem.get(idx2id[i], {int(hard[i])})) for i in range(n)]
        # hard-partition sizes (docs per partition) for the working-set fraction
        psize = np.zeros(npart)
        for i in range(n):
            psize[int(hard[i])] += 1
        index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
        sp = _splits(eng, _hard_membership(eng)); test = sp["test"][:limit]
        cache = os.path.join("data", "ukb_storage", d, subdir, "queries_test.npy") if subdir else None
        Q = (np.load(cache)[:len(test)].astype("float32") if cache and os.path.exists(cache)
             else eq([nd.content for nd, _, _ in test]).astype("float32"))
        _, I = index.search(Q, MAXK)
        S, M = _feats(I, mem_idx, npart, topn=topn)
        votes = _rr(S) + _rr(M)

        full = {k: 0.0 for k in KS}; docfrac = {k: 0.0 for k in KS}; nq = 0
        for qi, (_, _, golds) in enumerate(test):
            gidx = [id2idx[g] for g in golds if g in id2idx]
            if not gidx:
                continue
            nq += 1
            order = np.argsort(-votes[qi])
            for k in KS:
                topk = set(int(p) for p in order[:k])
                covered = all(any(p in topk for p in mem_idx[g]) for g in gidx)
                full[k] += 1.0 if covered else 0.0
                docfrac[k] += psize[list(topk)].sum() / n
        out[d] = {"npart": npart, "n_test": nq,
                  "FullCov": {k: round(100 * full[k] / nq, 2) for k in KS},
                  "docfrac": {k: round(100 * docfrac[k] / nq, 1) for k in KS}}
        log.info(f"[{d}] npart={npart} FullCov={out[d]['FullCov']} docfrac%={out[d]['docfrac']}")
        del X, index, S, M, Q, I; import gc; gc.collect()

    print(f"\n=== L1 paging tradeoff ({subdir}) — FullCov@K (completeness) / docfrac@K (%% of graph paged in) ===")
    print(f"{'dataset':16s} {'npart':>6s}  " + "  ".join(f"K={k:<3d}" for k in KS))
    for d in datasets:
        cov = "  ".join(f"{out[d]['FullCov'][k]:5.1f}" for k in KS)
        print(f"{d:16s} {out[d]['npart']:6d}  {cov}   (FullCov@K)")
        frac = "  ".join(f"{out[d]['docfrac'][k]:5.1f}" for k in KS)
        print(f"{'':16s} {'':6s}  {frac}   (% of corpus paged in)")
    print(f"\n{'MEAN FullCov':16s} {'':6s}  " + "  ".join(f"{np.mean([out[d]['FullCov'][k] for d in datasets]):5.1f}" for k in KS))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir)


if __name__ == "__main__":
    main()
