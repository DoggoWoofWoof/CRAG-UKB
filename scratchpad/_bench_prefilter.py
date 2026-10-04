"""Profile the builder's inner loop by STAGE, and test an exact float-prefilter kernel.

The builder ranks by packing (score, position) into one uint64 per CELL of a
(query_block x doc_block) matrix -- 250 x 250,000 = 62.5M cells -- and argpartitions that.
But only K=1000 of the 250,000 columns can survive.  This times the stages to see where the
5.14 s actually goes, then tests a kernel that argpartitions the FLOAT scores first and packs
keys only for the surviving candidates.

The prefilter is exact, not approximate, and the condition is checked per row rather than
assumed.  Let W be the top-k2 columns by score (k2 = K + margin) and s_k the K-th largest
score within W.  If min(W) < s_k strictly, then every document with score >= s_k is inside W,
because W holds the k2 largest scores; therefore the true top-K under (score desc, position
asc) is a subset of W and packing keys over W alone reproduces it exactly.  Rows failing that
test have a tie group at rank K wider than the margin and fall back to the full path.  This
run reports how many rows fail.

Nothing is written.
"""
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scratchpad"))
sys.path.insert(0, os.path.join(REPO, "src", "dataset_canonical"))

from build_retrieval_cache import DOC_BLOCK, CanonicalEmbeddings, order_key, take_top  # noqa
import pointer_resolver as PR  # noqa: E402
from build_splade_parts_local import _fast_node_ids  # noqa: E402

K = 1000
NQ_BLOCKS = 4
QB = 250
MARGIN = 200

PR._node_ids = _fast_node_ids
ROOT = os.path.join(REPO, "data", "final_canonical")
E = CanonicalEmbeddings("2wiki", "splade", "docs", root=ROOT)
Q = CanonicalEmbeddings("2wiki", "splade", "queries", root=ROOT)
nd = len(E)
qm = Q.gather(np.arange(0, NQ_BLOCKS * QB)).tocsr().astype(np.float32)
B = DOC_BLOCK
t = time.time()
dcsr = E.gather(np.arange(0, B)).tocsr().astype(np.float32)
t_docload = time.time() - t
print("docs %d  doc block %s nnz=%d  loaded %.1fs" % (nd, dcsr.shape, dcsr.nnz, t_docload),
      flush=True)
dm = dcsr.T.tocsc()
pos = np.arange(0, B, dtype=np.int64)
posb = np.broadcast_to(pos, (QB, B))


def prefilter_top(s, b0, k, margin):
    """Exact top-k via float prefilter. Returns (keys (qb,k), n_rows_needing_fallback)."""
    qb, Bl = s.shape
    k2 = min(Bl, k + margin)
    p = np.argpartition(s, Bl - k2, axis=1)[:, Bl - k2:]          # (qb, k2) cols
    c = np.take_along_axis(s, p, 1)                                # (qb, k2) scores
    thr_k = np.partition(c, k2 - k, axis=1)[:, k2 - k]             # k-th largest in W
    bad = int((~(c.min(axis=1) < thr_k)).sum())
    kk = order_key(c, (p + b0).astype(np.int64))
    return take_top(kk, k), bad


# ---- stage profile of the builder path ----------------------------------------------
acc = {"matmul+dense": 0.0, "order_key": 0.0, "take_top": 0.0}
nnz_prod = 0
tops_a = []
for i in range(NQ_BLOCKS):
    t = time.time()
    sp = qm[i * QB:(i + 1) * QB] @ dm
    nnz_prod += sp.nnz
    s = np.asarray(sp.todense(), dtype=np.float32)
    del sp
    acc["matmul+dense"] += time.time() - t
    t = time.time()
    kk = order_key(s, posb)
    acc["order_key"] += time.time() - t
    t = time.time()
    tops_a.append(take_top(kk, K))
    acc["take_top"] += time.time() - t
    del s, kk
tot_a = sum(acc.values())
print("")
print("BUILDER path, %d query blocks of %d, per block of %d:" % (NQ_BLOCKS, QB, QB))
for kx in ("matmul+dense", "order_key", "take_top"):
    print("   %-14s %6.2fs  (%4.1f%%)" % (kx, acc[kx] / NQ_BLOCKS,
                                          100.0 * acc[kx] / tot_a))
print("   %-14s %6.2fs" % ("TOTAL", tot_a / NQ_BLOCKS))
print("   score matrix density: %.4f  (nnz %s of %s cells)"
      % (nnz_prod / float(NQ_BLOCKS * QB * B), format(nnz_prod, ","),
         format(NQ_BLOCKS * QB * B, ",")))

# ---- prefilter path -----------------------------------------------------------------
t = time.time()
tops_c = []
bad_tot = 0
for i in range(NQ_BLOCKS):
    sp = qm[i * QB:(i + 1) * QB] @ dm
    s = np.asarray(sp.todense(), dtype=np.float32)
    del sp
    top, bad = prefilter_top(s, 0, K, MARGIN)
    tops_c.append(top)
    bad_tot += bad
    del s
t_c = time.time() - t
print("")
print("PREFILTER path : %.2fs per block of %d   (margin %d)" % (t_c / NQ_BLOCKS, QB, MARGIN))
print("   rows needing full-path fallback: %d of %d" % (bad_tot, NQ_BLOCKS * QB))

same_keys = tot = 0
for A, C in zip(tops_a, tops_c):
    same_keys += int((A == C).all(axis=1).sum())
    tot += A.shape[0]
print("   rows with identical KEYS vs builder: %d of %d  (%.4f)"
      % (same_keys, tot, same_keys / tot))

nblocks = (nd + DOC_BLOCK - 1) // DOC_BLOCK
print("")
print("PROJECTIONS (%d doc blocks, doc load %.1fs/block):" % (nblocks, t_docload))
for name, per_qb in (("builder  ", tot_a / NQ_BLOCKS), ("prefilter", t_c / NQ_BLOCKS)):
    for label, nq in (("shard 22 only (8,025 q)", 8025), ("all 11 shards (88,275 q)", 88275)):
        nqb = (nq + QB - 1) // QB
        secs = (per_qb * nqb + t_docload) * nblocks
        print("   %s  %-26s %6.2f h" % (name, label, secs / 3600.0))
