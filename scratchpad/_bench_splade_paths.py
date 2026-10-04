"""Time the builder's sparse x sparse path against sparse-docs @ dense-queries, and check
whether they pick the SAME top-K ids.

Two questions, one run, on one real document block of the real 2wiki corpus:

  1. How much faster is the dense-query path?  This decides whether the remaining eleven
     query shards are worth building with it.
  2. Do the two paths agree on the top-K ids exactly?  Same mathematics in a different
     summation order is only safe if it does not move a boundary.  Measured here on real
     data before committing hours, not asserted from the float format.

Nothing is written.  Uses a small number of query blocks so it finishes in minutes.
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
NQ_BLOCKS = 4          # query blocks of 250 to time
QB = 250

PR._node_ids = _fast_node_ids
E = CanonicalEmbeddings("2wiki", "splade", "docs", root=os.path.join(REPO, "data",
                                                                    "final_canonical"))
Q = CanonicalEmbeddings("2wiki", "splade", "queries", root=os.path.join(REPO, "data",
                                                                       "final_canonical"))
nd = len(E)
print("docs %d  queries %d" % (nd, len(Q)), flush=True)

t = time.time()
qm = Q.gather(np.arange(0, NQ_BLOCKS * QB)).tocsr().astype(np.float32)
print("query block %s nnz=%d vocab=%d  %.1fs" % (qm.shape, qm.nnz, qm.shape[1],
                                                 time.time() - t), flush=True)

B = DOC_BLOCK
t = time.time()
dcsr = E.gather(np.arange(0, B)).tocsr().astype(np.float32)
print("doc block %s nnz=%d  loaded %.1fs" % (dcsr.shape, dcsr.nnz, time.time() - t),
      flush=True)
pos = np.arange(0, B, dtype=np.int64)

# ---- builder path -------------------------------------------------------------------
t = time.time()
dm = dcsr.T.tocsc()
t_transpose = time.time() - t
print("builder transpose to CSC: %.1fs" % t_transpose, flush=True)

t = time.time()
tops_a = []
for i in range(NQ_BLOCKS):
    s = np.asarray((qm[i * QB:(i + 1) * QB] @ dm).todense(), dtype=np.float32)
    kk = order_key(s, np.broadcast_to(pos, (s.shape[0], B)))
    tops_a.append(take_top(kk, K))
    del s, kk
t_a = time.time() - t
print("BUILDER  sparse@sparse : %.1fs for %d query blocks (%.2fs per block of %d)"
      % (t_a, NQ_BLOCKS, t_a / NQ_BLOCKS, QB), flush=True)
del dm

# ---- fast path ----------------------------------------------------------------------
t = time.time()
tops_b = []
for i in range(NQ_BLOCKS):
    qd = np.asarray(qm[i * QB:(i + 1) * QB].todense(), dtype=np.float32)
    s = np.asarray(dcsr @ qd.T, dtype=np.float32).T
    kk = order_key(s, np.broadcast_to(pos, (s.shape[0], B)))
    tops_b.append(take_top(kk, K))
    del s, kk, qd
t_b = time.time() - t
print("FAST     sparse@dense  : %.1fs for %d query blocks (%.2fs per block of %d)"
      % (t_b, NQ_BLOCKS, t_b / NQ_BLOCKS, QB), flush=True)

# ---- agreement ----------------------------------------------------------------------
same_keys = same_ids = 0
tot = 0
for A, Bk in zip(tops_a, tops_b):
    same_keys += int((A == Bk).all(axis=1).sum())
    ia = (0xFFFFFFFF - (A & np.uint64(0xFFFFFFFF))).astype(np.int64)
    ib = (0xFFFFFFFF - (Bk & np.uint64(0xFFFFFFFF))).astype(np.int64)
    same_ids += int((ia == ib).all(axis=1).sum())
    tot += A.shape[0]
print("")
print("rows compared               : %d" % tot)
print("rows with identical KEYS    : %d  (%.4f)" % (same_keys, same_keys / tot))
print("rows with identical ID order: %d  (%.4f)" % (same_ids, same_ids / tot))

nblocks = (nd + DOC_BLOCK - 1) // DOC_BLOCK
nqb_full = (88275 + QB - 1) // QB
print("")
print("EXTRAPOLATION to the real job (88,275 queries = %d query blocks, %d doc blocks):"
      % (nqb_full, nblocks))
for name, per_qb, extra in (("builder", t_a / NQ_BLOCKS, t_transpose),
                            ("fast   ", t_b / NQ_BLOCKS, 0.0)):
    per_block = per_qb * nqb_full + extra
    print("  %s  %.0fs per doc block  ->  %.1f h total"
          % (name, per_block, per_block * nblocks / 3600.0))
