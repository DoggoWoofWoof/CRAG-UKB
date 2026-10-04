"""Streamed stage 2 of the frozen composite (FREEZE_QMAX_BALANCED_H2_PATCH1.json, stage "Cmax"):
the block max of the plain dense similarity q . e_v over ALL nodes, read from the fp16 node shards in
position order (ShardedRows.blocks) instead of the fp32 whole-corpus matrix the pinned
_l1g_candidate.dense_blockmax_rank(D, cq=200) reads through D.E.

The pinned function is NOT changed and NOT copied: this module computes the same mathematical quantity
by another memory path.  It is the only way to run stage 2 on corpora whose fp32 matrix does not fit
(hotpotqa 5.23 M x 1536 x 4 = 32 GB, 2wiki 37 GB).

What "same" means here, measured before anything was pre-registered (metaqa, 200 rows, this machine's
scipy-openblas 0.3.30): a chunked sgemm call Q @ E[a:b].T does NOT reproduce the whole-matrix call
Q @ E.T bit-for-bit -- OpenBLAS's column tiling at the end of a call and its thread partition of the
column range change the fp32 summation order for a small fraction of entries (|delta| <= 2.4e-7 at
|q . e| ~ 1, i.e. a few fp32 ulps of a 1536-term dot product; ~1 % of entries multithreaded, ~0.1 %
single-threaded; the metaqa dry run of the composite runner saw 11,993 of 863,136 block maxima differ,
max 9 ulp, with 0 of 1,998 rows changing their first 200 fused blocks).  So bitwise identity of the
stage-2 ranking is NOT attainable by any streamed implementation on this BLAS (and the pinned output is
itself a function of the thread count and of N).  The identity test (_l1c_transfer_composite.py
--reproduce) therefore records, per DEV_A cache: (i) whether the rankings are bitwise identical,
(ii) the max absolute score difference (gate: <= 1e-5 -- a logic bug such as a wrong block assignment
gives >= 1e-3) and its size in fp32 ulps (descriptive), (iii) how many cache rows differ in the stage-2
order / the first 200 fused blocks / the fused P50 set, and (iv) the coverage-level flips of the streamed
composite chain against the pinned one on split A (gate: <= 0.5 % of rows) -- a reproduction check
with a pre-registered acceptance gate, not a selection.

    blockmax_scores_streamed(D)      -> (nq, npart) float32 block maxima
    dense_blockmax_rank_streamed(D)  -> (nq, npart) int64 ranking (stable descending), the pinned function's contract
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

CQ = 200            # the pinned query batch (dense_blockmax_rank cq=200)
BLOCK = 40000       # node rows per streamed chunk (= the canonical dense shard size; a chunk never crosses a shard)


def blockmax_scores_streamed(D, cq=CQ, block=BLOCK, log=None):
    """(nq, npart) float32: max_{v in block} q . e_v from D.Q (fp32 unit queries), D.hard (node -> block) and
    D._E16 (fp16 node rows, memory-mapped shards walked in position order).  Never holds the fp32 corpus."""
    Q = np.ascontiguousarray(D.Q, dtype=np.float32)
    hard = np.asarray(D.hard, np.int64)
    npart, nq, N = int(D.npart), int(D.nq), int(hard.shape[0])
    qb = np.full((nq, npart), -np.inf, np.float32)
    t0 = time.time()
    for a, b, blk in D._E16.blocks(block):
        Ec = np.ascontiguousarray(np.asarray(blk, np.float32))        # fp16 -> fp32 exact; C-contiguous (c, dim)
        hc = hard[a:b]
        lperm = np.argsort(hc, kind="stable")
        hs = hc[lperm]
        ub, starts = np.unique(hs, return_index=True)                 # blocks present in this chunk; segment starts in lperm order
        for qa in range(0, nq, cq):
            qz = min(nq, qa + cq)
            QE = Q[qa:qz] @ Ec.T                                      # (m, c)
            loc = np.maximum.reduceat(QE[:, lperm], starts, axis=1)   # (m, |ub|) block max inside the chunk
            qb[qa:qz][:, ub] = np.maximum(qb[qa:qz][:, ub], loc)
        if log is not None and (b == N or (a // block) % 25 == 0):
            log("  streamed blockmax rows %d/%d (%.0fs)" % (b, N, time.time() - t0))
    assert np.isfinite(qb).all(), "a block received no node (empty block)"
    return qb


def dense_blockmax_rank_streamed(D, cq=CQ, block=BLOCK, log=None):
    """Same contract as _l1g_candidate.dense_blockmax_rank(D, cq): (nq, npart) int64 block ranking per cache
    row, descending block max of q . e_v, ties by block id (stable argsort)."""
    qb = blockmax_scores_streamed(D, cq=cq, block=block, log=log)
    return np.argsort(-qb, axis=1, kind="stable").astype(np.int64)


def blockmax_scores_pinned_path(D, cq=CQ):
    """The pinned arithmetic of _l1g_candidate.dense_blockmax_rank, returning the block maxima it sorts
    (the pinned function returns only the ranking).  Needs D.E (the fp32 whole-corpus matrix) -- used by the
    identity test on the DEV_A caches only; the caller asserts argsort(-qb) == the pinned function's output."""
    E, Q = D.E, D.Q
    hard = np.asarray(D.hard, np.int64)
    npart, nq = int(D.npart), int(D.nq)
    perm = np.argsort(hard, kind="stable")
    starts = np.searchsorted(hard[perm], np.arange(npart))
    qb = np.empty((nq, npart), np.float32)
    for a in range(0, nq, cq):
        b = min(nq, a + cq)
        QE = Q[a:b] @ E.T
        qb[a:b] = np.maximum.reduceat(QE[:, perm], starts, axis=1)
    return qb


def ulp_distance_max(x, y):
    """max over entries of |x - y| in units of the fp32 spacing at max(|x|, |y|) (0 = bitwise equal)."""
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    sp = np.spacing(np.maximum(np.abs(x), np.abs(y)).astype(np.float32))
    return float((np.abs(x.astype(np.float64) - y.astype(np.float64)) / sp).max())


if __name__ == "__main__":
    print(__doc__)
