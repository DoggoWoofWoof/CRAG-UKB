"""
L1 blocker analysis — WHY don't musique/2wiki gold partitions reach top-20?
===========================================================================
For each failing test query (a gold whose partition is NOT in the top-20 voted partitions),
classify the miss per uncovered gold:
  NOT_RETRIEVED : the gold node itself isn't in the top-MAXK dense retrieval  -> encoder can't reach it
  NOT_VOTED     : gold retrieved, but none of its overlap-partitions get any vote -> membership gap
  OUTRANKED     : gold's partition IS voted, just ranked >20 -> dilution (noise partitions out-vote it)
Also: golds-per-query and, for multi-partition golds (2wiki comparisons), how many gold partitions are
missed (one entity vs both). Uses dense + overlap-voting (the structural signal; the offset head barely
moves L1). Default encoder = gte_qwen (cached queries). Local, read-only.
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

log = logging.getLogger("experiments.l1_blocker")


def analyze(dataset, subdir="gte_qwen", limit=8000, topn=200):
    eng = CoreEngine(source=dataset, index_subdir=subdir)
    X, eq, tag = load_docs_and_encoder(eng, dataset, subdir); X = X.astype("float32")
    n = X.shape[0]; npart = max(int(p) for p in eng.partition_map.values()) + 1
    id2idx = eng.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
    hard = np.array([int(eng.partition_map.get(idx2id[i], -1)) for i in range(n)])
    mem = _onehop_membership(eng)
    mem_idx = [sorted(mem.get(idx2id[i], {int(hard[i])})) for i in range(n)]
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
    sp = _splits(eng, _hard_membership(eng)); test = sp["test"][:limit]

    # cached gte/bge queries if present
    cache = os.path.join("data", "ukb_storage", dataset, subdir, "queries_test.npy") if subdir else None
    if cache and os.path.exists(cache):
        Q = np.load(cache)[:len(test)].astype("float32")
    else:
        Q = eq([nd.content for nd, _, _ in test]).astype("float32")

    D, I = index.search(Q, MAXK)                             # dense order per query
    rank_of = [{int(nd): r for r, nd in enumerate(row)} for row in I]   # node -> dense rank

    buckets = Counter(); n_fail = 0; n_q = 0
    gpart_counts = Counter(); missed_frac = []
    outranked_ranks = []
    for qi, (nd, _, golds) in enumerate(test):
        gidx = [id2idx[g] for g in golds if g in id2idx]
        if not gidx:
            continue
        n_q += 1
        # partition votes (overlap membership) from this query's dense order
        S, M = _feats(I[qi:qi + 1], mem_idx, npart, topn=topn)
        votes = (_rr(S) + _rr(M))[0]
        part_order = np.argsort(-votes)
        prank = {int(p): r for r, p in enumerate(part_order)}
        top20 = set(int(p) for p in part_order[:20])

        gparts_all = set()
        uncovered = []
        for g in gidx:
            gp = mem_idx[g]; gparts_all |= set(gp)
            if not any(p in top20 for p in gp):
                uncovered.append(g)
        gpart_counts[len(gparts_all)] += 1
        if not uncovered:
            continue
        n_fail += 1
        # fraction of this query's gold PARTITIONS that are missed (one entity vs both)
        missed_parts = set(p for g in uncovered for p in mem_idx[g])
        missed_frac.append(round(len(missed_parts & set().union(*[set(mem_idx[g]) for g in gidx])) /
                                 max(len(gparts_all), 1), 2))
        for g in uncovered:
            dr = rank_of[qi].get(g, MAXK)
            gp = mem_idx[g]
            voted = [p for p in gp if votes[p] > 0]
            if dr >= MAXK:
                buckets["NOT_RETRIEVED"] += 1
            elif not voted:
                buckets["NOT_VOTED"] += 1
            else:
                buckets["OUTRANKED"] += 1
                outranked_ranks.append(min(prank[p] for p in voted))

    print(f"\n===== {dataset} ({tag}) =====")
    print(f"  n_test={n_q}  failed_queries={n_fail} ({100*n_fail/max(n_q,1):.1f}%)")
    print(f"  golds-per-query partition spread: {dict(sorted(gpart_counts.items()))}")
    tot = sum(buckets.values())
    print(f"  uncovered-gold miss types (of {tot}):")
    for b in ("NOT_RETRIEVED", "NOT_VOTED", "OUTRANKED"):
        c = buckets[b]
        print(f"    {b:14s} {c:5d} ({100*c/max(tot,1):.1f}%)")
    if outranked_ranks:
        arr = np.array(outranked_ranks)
        print(f"  OUTRANKED gold-partition rank: median={int(np.median(arr))} "
              f"p25={int(np.percentile(arr,25))} p75={int(np.percentile(arr,75))} "
              f"(<=30: {100*np.mean(arr<30):.0f}%, <=50: {100*np.mean(arr<50):.0f}%)")
    if missed_frac:
        mf = np.array(missed_frac)
        print(f"  of failing queries: fraction of gold-partitions missed: "
              f"mean={mf.mean():.2f}  (==1.0 all missed: {100*np.mean(mf>=0.999):.0f}%, partial: {100*np.mean(mf<0.999):.0f}%)")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["musique_clean", "2wiki_clean"])
    p.add_argument("--subdir", default="gte_qwen")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    for d in a.datasets:
        analyze(d, subdir=a.subdir)


if __name__ == "__main__":
    main()
