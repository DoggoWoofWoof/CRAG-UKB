"""Build named splade query-shard parts LOCALLY, in ONE corpus pass for all of them.

WHY THIS EXISTS

  2wiki's splade cache was being built as 24 query shards on Modal.  The workspace was disabled
  with 13 of 24 written; two replacement workspaces accepted a 4.72 GB store upload and then
  refused to run a function ("has exceeded its spend limit"), and a compute probe found all
  eleven remaining workspaces in the same state.  Modal is exhausted, so the last eleven shards
  have to be computed here or not at all.

WHY ONE PASS AND NOT ELEVEN

  A query shard's cost is one full streamed pass over 5,989,847 documents.  Running the eleven
  separately would stream the corpus eleven times.  The score block is sized by (query_block x
  doc_block), NOT by the total number of queries, so eleven shards' queries can ride along in a
  single pass; only the `best` key matrix grows, to 88,275 x 1000 x 8 B = 706 MB.  Same
  arithmetic, one eleventh of the I/O.

  This is query subsetting, which is allowed.  The corpus is never subset: every one of the
  5,989,847 documents is scored for every query in the union.  That is the whole point of
  streaming it.

WHY A LOCALLY BUILT PART MAY BE MERGED WITH A MODAL-BUILT ONE

  Not assumed -- measured, before this script was written.  hotpotqa's splade cache was built on
  Modal and then verified by recomputing it HERE over its full 5,233,329-document corpus:
  positionwise_exact_id_fraction was 1.0, set overlap 1.0 at depths 10/100/1000.  The Modal
  function is this builder's scipy path unchanged, and sparse accumulation order is
  blocking-independent, so the two agree on ids exactly rather than approximately.

  One honest limit, stated because it is easy to miss: verify_heavy_caches.py checks a cache by
  recomputing it locally.  For rows built on Modal that is a cross-implementation check; for
  rows built by THIS script it degenerates to local-vs-local, which tests reproducibility and
  not agreement.  The part metas record `backend` per part so a reader can tell which rows got
  which strength of check, and the merge record counts both.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/build_splade_parts_local.py \
      --ds 2wiki --qshards 24 --only 2,3,4,6,7,10,11,12,15,17,22 [--probe-blocks 2]
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scratchpad"))
# CanonicalEmbeddings resolves paths against the PACKAGE directory, not the repo root -- the
# builder hardcodes ROOT = "data/final_canonical" and is always run from the repo root.  Pass
# the same thing, absolute, so this works from any cwd.
ROOT = os.path.join(REPO, "data", "final_canonical")

from build_retrieval_cache import (DOC_BLOCK, KEY_BUDGET, CanonicalEmbeddings,  # noqa: E402
                                   order_key, take_top, unpack)
import pointer_resolver as PR  # noqa: E402

OUTDIR = os.path.join(REPO, "scratchpad", "_splade_parts")


class _LengthOnly(object):
    """Stands in for the node-id list, carrying its LENGTH and refusing to be indexed.

    CanonicalEmbeddings loads every node id to check `len(ids) == src.size`, which for 2wiki
    means json.loads on 5,989,847 lines of a 5.32 GB file -- minutes of wall time and roughly
    600 MB of Python strings, before any arithmetic starts.  This builder only ever calls
    gather(positions); it never resolves an id.

    The check is NOT weakened: the number compared is still the true line count of
    nodes.jsonl, just counted by scanning for newlines instead of by parsing JSON.  Indexing
    raises rather than silently returning something wrong, so if any code path does want a real
    id it fails loudly here instead of computing with a stand-in.
    """

    def __init__(self, n):
        self.n = int(n)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        raise RuntimeError("node ids were not loaded (see _LengthOnly): this run only needs "
                           "positions. If you need real ids, drop the _node_ids patch.")


def _count_lines(p):
    n, last = 0, b"\n"
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    # a file whose final line has no terminator still holds that line
    return n + (1 if last != b"\n" else 0)


def _fast_node_ids(p):
    t = time.time()
    n = _count_lines(p)
    print("   node id check: %d lines counted in %.0fs (not parsed -- see _LengthOnly)"
          % (n, time.time() - t), flush=True)
    return _LengthOnly(n)


def _prefilter_top(s, pos, b0, k, margin):
    """Exact top-k per row, packing (score, position) keys only for surviving candidates.

    The builder packs one uint64 key for every CELL of the (query_block x doc_block) score
    matrix -- 62.5M cells at the default blocking -- and argpartitions that. order_key builds
    roughly 3 GB of uint64 temporaries to produce a 500 MB key matrix, of which only K=1000
    columns per row can possibly survive. Measured on a real 2wiki document block it is 40.3%
    of the inner loop (matmul+densify 53.8%, take_top 5.9%).

    So argpartition the FLOAT scores first and pack keys for the top k2 = k + margin columns
    only. This is exact, and the condition is PROVEN PER ROW rather than assumed. Let W be
    those k2 columns and s_k the k-th largest score within W. If min(W) < s_k strictly, then
    every document with score >= s_k lies inside W: W holds the k2 largest scores, so a
    document outside W with score == s_k would force min(W) >= s_k. Hence the true top-k under
    (score desc, position asc) is a subset of W, and packing keys over W alone reproduces it
    exactly -- including the tie-break to the smaller canonical position, because the same
    order_key/take_top pair decides it.

    Rows whose tie group at rank k is wider than the margin fail that test and are recomputed
    by the full path right here, so the answer is exact even if the margin is chosen badly.
    The count is returned and recorded in the part meta rather than discarded.

    Measured against the builder path on 1000 real query rows of a real 250,000-document
    block: 1000/1000 identical keys, 0 rows needing the fallback.
    """
    qb, B = s.shape
    k2 = min(B, k + margin)
    if k2 >= B:                      # margin covers the whole block; nothing to prefilter
        return take_top(order_key(s, np.broadcast_to(pos, (qb, B))), k), 0
    p = np.argpartition(s, B - k2, axis=1)[:, B - k2:]        # (qb, k2) local columns
    c = np.take_along_axis(s, p, 1)                           # (qb, k2) their scores
    thr_k = np.partition(c, k2 - k, axis=1)[:, k2 - k]        # k-th largest score within W
    top = take_top(order_key(c, (p + b0).astype(np.int64)), k)
    bad = ~(c.min(axis=1) < thr_k)
    nfb = int(bad.sum())
    if nfb:
        r = np.where(bad)[0]
        top[r] = take_top(order_key(s[r], np.broadcast_to(pos, (len(r), B))), k)
    return top, nfb


def bounds(nq_all, qshard, n_qshards):
    """The Modal function's exact formula. A part must cover the rows its NAME claims."""
    return (nq_all * qshard // n_qshards, nq_all * (qshard + 1) // n_qshards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--qshards", type=int, required=True)
    ap.add_argument("--only", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--probe-blocks", type=int, default=0,
                    help="stop after N document blocks and extrapolate; writes nothing")
    ap.add_argument("--fast-dense-queries", action="store_true",
                    help="densify the query block and use sparse-docs @ dense-queries instead "
                         "of the builder's sparse @ sparse + .todense(). Same mathematics, "
                         "different summation order; also skips the CSC transpose. SPLADE's "
                         "vocab is ~30k so a 250-query block densifies to about 30 MB. This "
                         "makes the later verification STRONGER, not weaker: "
                         "verify_heavy_caches.py recomputes with the builder's own path, so a "
                         "shard built this way is checked across two implementations rather "
                         "than reproduced by the one that made it.")
    ap.add_argument("--kernel", choices=["prefilter", "builder"], default="prefilter",
                    help="prefilter (default) argpartitions the float scores and packs "
                         "(score, position) keys only for the top k+margin candidates, with a "
                         "per-row exactness proof and a full-path fallback for rows whose tie "
                         "group at rank k is wider than the margin. builder packs a key for "
                         "every cell, as build_retrieval_cache.py does.")
    ap.add_argument("--margin", type=int, default=200,
                    help="prefilter candidate slack above K. Only affects SPEED and the "
                         "fallback count, never the answer.")
    a = ap.parse_args()

    idx = sorted(int(x) for x in a.only.split(",") if x != "")
    PR._node_ids = _fast_node_ids
    E = CanonicalEmbeddings(a.ds, "splade", "docs", root=ROOT)
    Q = CanonicalEmbeddings(a.ds, "splade", "queries", root=ROOT)
    nd, nq_all = len(E), len(Q)
    k = min(a.k, nd)

    spans = [(s,) + bounds(nq_all, s, a.qshards) for s in idx]
    rows = np.concatenate([np.arange(lo, hi, dtype=np.int64) for _, lo, hi in spans])
    if len(np.unique(rows)) != len(rows):
        raise SystemExit("the requested shards overlap; their spans must be disjoint")
    nq = len(rows)

    print("%s splade: %d docs, %d of %d query rows in %d shards"
          % (a.ds, nd, nq, nq_all, len(idx)), flush=True)
    for s, lo, hi in spans:
        print("   shard %-3d rows %7d..%7d  (%d)" % (s, lo, hi, hi - lo), flush=True)
    nblocks = (nd + DOC_BLOCK - 1) // DOC_BLOCK
    print("   key matrix %.0f MB, %d document blocks of %d"
          % (nq * k * 8 / 1e6, nblocks, DOC_BLOCK), flush=True)

    t0 = time.time()
    qm = Q.gather(rows).tocsr().astype(np.float32)
    print("   query matrix %s nnz=%d  %.0fs" % (qm.shape, qm.nnz, time.time() - t0))

    best = np.empty((nq, k), dtype=np.uint64)
    n_fallback_rows = 0
    first = True
    done_blocks = 0
    t0 = time.time()
    for b0 in range(0, nd, DOC_BLOCK):
        b1 = min(b0 + DOC_BLOCK, nd)
        B = b1 - b0
        pos = np.arange(b0, b1, dtype=np.int64)
        dcsr = E.gather(np.arange(b0, b1)).tocsr().astype(np.float32)
        # The builder materialises the transpose as CSC and multiplies sparse x sparse. The
        # fast path keeps the docs in CSR and multiplies by a DENSE query block, which is the
        # product scipy is actually good at, and needs no transpose at all.
        dm = dcsr if a.fast_dense_queries else dcsr.T.tocsc()
        qb = max(64, min(nq, int(KEY_BUDGET // (max(B, 1) * 8))))
        for q0 in range(0, nq, qb):
            q1 = min(q0 + qb, nq)
            if a.fast_dense_queries:
                qd = np.asarray(qm[q0:q1].todense(), dtype=np.float32)      # (qb, vocab)
                s = np.asarray(dm @ qd.T, dtype=np.float32).T               # (qb, B)
                del qd
            else:
                s = np.asarray((qm[q0:q1] @ dm).todense(), dtype=np.float32)
            if a.kernel == "prefilter":
                top, nfb = _prefilter_top(s, pos, b0, min(k, B), a.margin)
                n_fallback_rows += nfb
            else:
                kk = order_key(s, np.broadcast_to(pos, (q1 - q0, B)))
                top = take_top(kk, min(k, B))
                del kk
            del s
            if first:
                if top.shape[1] != k:
                    raise SystemExit("first document block yielded %d < K=%d columns"
                                     % (top.shape[1], k))
                best[q0:q1] = top
            else:
                best[q0:q1] = take_top(np.concatenate([best[q0:q1], top], axis=1), k)
            del top
        first = False
        del dm, dcsr
        done_blocks += 1
        el = time.time() - t0
        print("    docs %s/%s qb=%d  %.0fs  (proj total %.1f h)"
              % (format(b1, ","), format(nd, ","), qb, el,
                 el / done_blocks * nblocks / 3600.0), flush=True)
        if a.probe_blocks and done_blocks >= a.probe_blocks:
            print("PROBE ONLY -- stopping after %d of %d blocks. Projected full pass: %.2f h"
                  % (done_blocks, nblocks, el / done_blocks * nblocks / 3600.0))
            return 0

    secs = time.time() - t0
    del qm
    od = os.path.join(OUTDIR, a.ds)
    os.makedirs(od, exist_ok=True)
    off = 0
    for s, lo, hi in spans:
        n = hi - lo
        ids = np.empty((n, k), dtype=np.int32)
        scf = np.empty((n, k), dtype=np.float16)
        for st in range(0, n, 20000):
            en = min(st + 20000, n)
            sc_, i_ = unpack(best[off + st:off + en])
            ids[st:en] = i_.astype(np.int32)
            scf[st:en] = sc_.astype(np.float16)
        if int(ids.min()) < 0:
            raise SystemExit("negative canonical id in shard %d -- a padded key survived" % s)
        p = os.path.join(od, "part_%04d_of_%04d.npz" % (s, a.qshards))
        np.savez(p, ids=ids, scores=scf)
        meta = {"n_queries": int(n), "K": int(k),
                # the pass is shared, so per-shard wall time is not separately observable;
                # this is the shared pass divided by the shards that rode in it, and it is
                # labelled as such rather than presented as a measured per-shard figure
                "seconds": round(secs / len(spans), 1),
                "seconds_basis": "one shared corpus pass of %.1f s over %d shards"
                                 % (secs, len(spans)),
                "score_min": float(scf.min()), "score_max": float(scf.max()),
                "top1_score_median": float(np.median(scf[:, 0])),
                "bytes": os.path.getsize(p), "n_docs": int(nd),
                "qshard": int(s), "n_qshards": int(a.qshards), "lo": int(lo), "hi": int(hi),
                "kernel": a.kernel,
                "prefilter_margin": (int(a.margin) if a.kernel == "prefilter" else None),
                "prefilter_fallback_rows": (int(n_fallback_rows)
                                            if a.kernel == "prefilter" else None),
                "backend": "local cpu, scipy %s, %s kernel; built in one shared pass with "
                           "shards %s"
                           % ("sparse-docs @ dense-queries (fast path)"
                              if a.fast_dense_queries else
                              "sparse @ sparse, builder path",
                              a.kernel,
                              ",".join(str(x) for x, _, _ in spans))}
        json.dump(meta, io.open(p[:-4] + ".meta.json", "w", encoding="utf-8"), indent=1)
        print("   WROTE %s  %d rows  %.3f GB  top1_median=%.4f"
              % (os.path.basename(p), n, meta["bytes"] / 1e9, meta["top1_score_median"]))
        off += n
    print("done: %d shards, %.0fs (%.2f h) for the shared pass" % (len(spans), secs,
                                                                   secs / 3600.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
