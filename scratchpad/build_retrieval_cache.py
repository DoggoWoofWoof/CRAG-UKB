"""
RETRIEVAL CACHES
================
Exact top-K for every canonical query against the FULL canonical corpus, cached so downstream
work does not re-score 11.4M documents every run.

WHY ONE DEEP K RATHER THAN A SET OF K
    K = 50, 100, 250, 500, 1000 are all prefixes of K = 1000. Materialising the deep one and
    truncating is not a shortcut, it is the only way to avoid choosing K at all: a cache built
    at a single depth cannot have had its depth tuned on an outcome. Nothing here reads a
    label, a split, or an accuracy number -- the cache is a function of the corpus, the
    queries and the frozen encoders, and of nothing else.

WHAT IS SCORED
    The full corpus, always. Query subsetting is allowed; corpus subsetting by query is
    forbidden, so every query is scored against every canonical node -- no pre-filter, no
    candidate pool, no partition.

TIES ARE REAL HERE, SO THEY ARE ORDERED RATHER THAN LEFT TO THE SORT
    The pointer index is legitimately many-to-one: canonical nodes whose encoder input is
    token-identical share one vector, so equal scores are common rather than a measure-zero
    curiosity. Left to argpartition, which of two tied documents lands in the top-K would
    depend on partition order and could change with a NumPy version.

    So score and position are packed into ONE unsigned 64-bit key:

        high 32 bits   the float32 score in its order-preserving unsigned form
        low  32 bits   0xFFFFFFFF - position, so that among equal scores the SMALLER
                       canonical position sorts higher

    Positions are unique, so the key is unique: there are no ties left to break, and
    selecting the largest K keys is exactly "highest score, ascending position on ties".

    WHAT THAT DOES AND DOES NOT BUY, MEASURED
        It makes the cache reproducible for a GIVEN score matrix. It does not make the stored
        ORDER implementation-independent, and an earlier version of this docstring claimed
        "the cache is byte-reproducible" without qualification, which is false.

        The tie-break only fires when two scores come out BIT-equal. This builder scores a
        (queries x dim) @ (dim x block) matrix-matrix product; scoring the same query as a
        matrix-vector product sums in a different order and lands one ulp away, which turns a
        tie into an ordering -- a different one. Sparse splade dots have a fixed accumulation
        order and do not show this at all.

        Independently recomputed (DENSE_CACHE_ORDER_DIAGNOSIS.json, 24 queries across the
        three dense caches): the top-K SET is identical in every case, score mass lost against
        a float64 arbiter is exactly 0.0, and the earliest ordering disagreement is at rank 130
        with a score delta of 9.5e-08. So the CONTENT is exact and implementation-independent;
        the ORDER among equally-scoring documents is reproducible by this builder but is not a
        property a consumer should rely on across implementations.

MEMORY
    The naive form of this -- concatenate the running best with a whole document block and
    sort the result -- wants about 4 GB per query block on a 117k-document corpus, which is
    how it was first written and why it is not written that way now. Each document block is
    reduced to its own top-K first, so the merge only ever touches 2K columns.

    That fixed the per-block cost but left a second, larger one: the per-block results were
    accumulated in a list and concatenated at the end, so peak memory scaled with the number
    of QUERIES. On metaqa (407,513 queries, K=1000) the list, its concatenation, the second
    concatenation against the running best, and argsort's int64 index array come to roughly
    18 GB against 15.7 GB of RAM.

    It did finish that way -- metaqa dense completed in 2109 s -- so the honest statement is
    that the old shape DEPENDED ON PAGING, not that it was impossible. It is still the wrong
    shape: it borrows swap on a 98%-full volume, and the same arithmetic on hotpotqa or 2wiki
    scales past anything the pagefile could cover. The running best is now PREALLOCATED once
    and updated in place, so nothing is accumulated and nothing is concatenated at full width.
    Peak is bounded by the best matrix plus one query block, independent of query count and of
    how the corpus is blocked.

    Queries are held in float16 and cast per block rather than kept as a float32 matrix, which
    is another 1.25 GB on metaqa for no loss: the scores are accumulated in float32 either way.

WHAT THIS DOES NOT DO
    It does not rank, fuse, rerank, or evaluate. It is substrate: the raw exact top-K, from
    which any router or reranker can be built later without re-scoring.
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "data/final_canonical")
from pointer_resolver import CanonicalEmbeddings  # noqa: E402

ROOT = "data/final_canonical"
DOC_BLOCK = 250_000
KEY_BUDGET = 500_000_000          # bytes allowed for the uint64 key matrix


def order_key(scores, positions):
    """
    (score, position) -> one uint64 whose numeric order IS the ranking order.

    float32 does not sort as its raw bits when negative, so the standard IEEE transform is
    applied: flip the sign bit for non-negative values, flip every bit for negative ones.
    After that, unsigned comparison of the 32-bit pattern equals float comparison.
    """
    u = np.ascontiguousarray(scores, dtype=np.float32).view(np.uint32)
    mono = np.where((u >> 31) == 0, u | np.uint32(0x80000000), ~u)
    return (mono.astype(np.uint64) << np.uint64(32)) | \
           (np.uint64(0xFFFFFFFF) - positions.astype(np.uint64))


def unpack(keys):
    mono = (keys >> np.uint64(32)).astype(np.uint32)
    u = np.where((mono >> 31) == 1, mono & np.uint32(0x7FFFFFFF), ~mono)
    s = u.view(np.float32)
    pos = (np.uint64(0xFFFFFFFF) - (keys & np.uint64(0xFFFFFFFF))).astype(np.int32)
    return s, pos


def take_top(keys, k):
    """Largest k keys per row, in descending order. Keys are unique, so this is exact."""
    if keys.shape[1] <= k:
        o = np.argsort(-keys, axis=1, kind="stable")
        return np.take_along_axis(keys, o, 1)
    p = np.argpartition(keys, keys.shape[1] - k, axis=1)[:, keys.shape[1] - k:]
    c = np.take_along_axis(keys, p, 1)
    o = np.argsort(-c, axis=1, kind="stable")
    return np.take_along_axis(c, o, 1)


def run(ds, model, K, out):
    E = CanonicalEmbeddings(ds, model, "docs", root=ROOT)
    Q = CanonicalEmbeddings(ds, model, "queries", root=ROOT)
    nd, nq = len(E), len(Q)
    k = min(K, nd)
    t0 = time.time()

    if model == "dense":
        qm = np.ascontiguousarray(Q.gather(np.arange(nq)), dtype=np.float16)
    else:
        qm = Q.gather(np.arange(nq)).tocsr().astype(np.float32)

    # Preallocated once. Filled on the first document block, merged in place afterwards.
    best = np.empty((nq, k), dtype=np.uint64)
    first = True
    for b0 in range(0, nd, DOC_BLOCK):
        b1 = min(b0 + DOC_BLOCK, nd)
        B = b1 - b0
        pos = np.arange(b0, b1, dtype=np.int64)
        if model == "dense":
            dm = np.ascontiguousarray(E.gather(np.arange(b0, b1)), dtype=np.float32).T
        else:
            dm = E.gather(np.arange(b0, b1)).tocsr().astype(np.float32).T.tocsc()
        qb = max(64, min(nq, int(KEY_BUDGET // (max(B, 1) * 8))))
        for q0 in range(0, nq, qb):
            q1 = min(q0 + qb, nq)
            if model == "dense":
                s = np.asarray(qm[q0:q1], dtype=np.float32) @ dm
            else:
                s = np.asarray((qm[q0:q1] @ dm).todense(), dtype=np.float32)
            kk = order_key(s, np.broadcast_to(pos, (q1 - q0, B)))
            top = take_top(kk, min(k, B))
            del s, kk
            if first:
                # k = min(K, nd) and the first block is the widest, so a short first block
                # would mean nd < k. Padding it would write a key that unpacks to id -1 and a
                # NaN score, so this refuses instead of inventing rows.
                if top.shape[1] != k:
                    raise SystemExit("first document block yielded %d < K=%d columns"
                                     % (top.shape[1], k))
                best[q0:q1] = top
            else:
                best[q0:q1] = take_top(np.concatenate([best[q0:q1], top], axis=1), k)
            del top
        first = False
        del dm
        print("    docs %s/%s qb=%d  %.0fs"
              % (format(b1, ","), format(nd, ","), qb, time.time() - t0), flush=True)

    del qm
    ids = np.empty((nq, k), dtype=np.int32)
    scf = np.empty((nq, k), dtype=np.float16)
    for a in range(0, nq, 20000):
        s_, i_ = unpack(best[a:a + 20000])
        ids[a:a + 20000] = i_.astype(np.int32)
        scf[a:a + 20000] = s_.astype(np.float16)
    del best
    sc = scf
    np.savez(out, ids=ids, scores=sc)
    return {"n_queries": nq, "n_docs": nd, "K": k, "seconds": round(time.time() - t0, 1),
            "score_min": float(sc.min()), "score_max": float(sc.max()),
            "top1_score_median": float(np.median(sc[:, 0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--models", default="dense,splade")
    a = ap.parse_args()
    p = ROOT + "/RETRIEVAL_CACHE.json"
    rec = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}
    rec["RECORD"] = "CANONICAL_V1_RETRIEVAL_CACHE"
    rec["METHOD"] = {
        "dense": "exact inner product on L2-normalised float16 vectors, float32 accumulation",
        "splade": "exact sparse dot product over the 30,522-dim vocabulary",
        "corpus": "the FULL canonical node set for every query -- no pre-filter, no pool",
        "ties": ("score and canonical position are packed into one unique uint64 sort key, "
                 "so equal scores resolve to the smaller position deterministically rather "
                 "than by partition order. Ties are common because the pointer index is "
                 "many-to-one."),
        "K_policy": ("one deep K, materialised once; 50/100/250/500 are prefixes of it. K was "
                     "NOT selected from any accuracy number -- nothing in this builder reads "
                     "a label or a split"),
        "ids_are": ("canonical positions (nodes.jsonl line numbers), the same space as the "
                    "pointer index and the graphs"),
        "query_order": "the order of queries/pointer_index/query_ids.json",
        "vectors_read_via": ("the pointer index, never the Phase-C shards directly -- those "
                             "still hold the 142,633 corrupted dense rows")}
    for ds in a.datasets:
        d = "%s/%s/retrieval_cache" % (ROOT, ds)
        os.makedirs(d, exist_ok=True)
        for model in a.models.split(","):
            o = "%s/%s_top%d.npz" % (d, model, a.k)
            print("== %s / %s -> %s" % (ds, model, o), flush=True)
            r = run(ds, model, a.k, o)
            r["file"] = o
            r["bytes"] = os.path.getsize(o)
            rec.setdefault("caches", {}).setdefault(ds, {})[model] = r
            print("   %s q=%s docs=%s K=%d  %.0fs  %.2f GB  top1_median=%.4f"
                  % (model, format(r["n_queries"], ","), format(r["n_docs"], ","), r["K"],
                     r["seconds"], r["bytes"] / 1e9, r["top1_score_median"]), flush=True)
            json.dump(rec, io.open(p, "w", encoding="utf-8"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
