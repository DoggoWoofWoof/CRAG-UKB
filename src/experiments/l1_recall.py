"""
Recall report (complements the strict FullCov@20).
===================================================
We've been reporting FullCov@20 = fraction of queries whose EVERY gold's partition is in the top-20
(all-or-nothing). This adds the more standard RECALL views:
  partition gold-recall@20 : mean fraction of a query's golds whose partition IS in the top-20 (partial credit)
  doc recall@{20,100,200}  : mean fraction of a query's gold DOCS in the top-k retrieved nodes (node level)
Dense + overlap-voting, encoder gte_qwen (cached queries). Local, read-only.
"""
import os
import argparse
import logging

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.experiments.encoder_swap import load_docs_and_encoder
from src.experiments.overlap_retrain import _splits, _hard_membership, _onehop_membership
from src.experiments.l1_rerank100 import _feats, _rr, MAXK

log = logging.getLogger("experiments.l1_recall")
DATASETS = ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]


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
        index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
        sp = _splits(eng, _hard_membership(eng)); test = sp["test"][:limit]
        cache = os.path.join("data", "ukb_storage", d, subdir, "queries_test.npy") if subdir else None
        Q = (np.load(cache)[:len(test)].astype("float32") if cache and os.path.exists(cache)
             else eq([nd.content for nd, _, _ in test]).astype("float32"))
        _, I = index.search(Q, MAXK)
        S, M = _feats(I, mem_idx, npart, topn=topn)
        votes = _rr(S) + _rr(M)

        full = part_rec = 0.0; dr = {20: 0.0, 100: 0.0, 200: 0.0}; nq = 0
        for qi, (_, _, golds) in enumerate(test):
            gidx = [id2idx[g] for g in golds if g in id2idx]
            if not gidx:
                continue
            nq += 1
            top20 = set(int(p) for p in np.argsort(-votes[qi])[:20])
            covered = [g for g in gidx if any(p in top20 for p in mem_idx[g])]
            part_rec += len(covered) / len(gidx)
            full += 1.0 if len(covered) == len(gidx) else 0.0
            gset = set(gidx)
            for k in dr:
                hits = sum(1 for nd in I[qi][:k] if int(nd) in gset)
                dr[k] += hits / len(gidx)
        out[d] = {"FullCov@20": round(100 * full / nq, 2), "partRecall@20": round(100 * part_rec / nq, 2),
                  "docRecall@20": round(100 * dr[20] / nq, 2), "docRecall@100": round(100 * dr[100] / nq, 2),
                  "docRecall@200": round(100 * dr[200] / nq, 2)}
        log.info(f"[{d}] {out[d]}")
        del X, index, S, M, Q, I; import gc; gc.collect()

    cols = ["FullCov@20", "partRecall@20", "docRecall@20", "docRecall@100", "docRecall@200"]
    print(f"\n=== L1 recall views ({subdir}, dense+overlap-voting) ===")
    print(f"{'dataset':16s} " + " ".join(f"{c:>14s}" for c in cols))
    for d in datasets:
        print(f"{d:16s} " + " ".join(f"{out[d][c]:14.2f}" for c in cols))
    print(f"{'MEAN':16s} " + " ".join(f"{np.mean([out[d][c] for d in datasets]):14.2f}" for c in cols))
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
