"""The frozen static candidate of the L1_STATIC lane: multi-radius static node -> blocks tables.

Tables (all query independent, computed once from the partition and the structural family only):
  R0  own block
  RL  own block + blocks of the DIRECTED structural out-neighbours       (= the frozen BASE table)
  R1  own block + blocks of the UNDIRECTED structural neighbours
  R2  R1 plus, for every undirected neighbour v whose own R1 span is <= P_MAIN (=50) blocks, the R1 table of v
      (budget rule: an intermediate whose neighbourhood already spans more than the P50 budget cannot be
      followed, so it is not passed through; no threshold was fitted -- P_MAIN is the frozen budget)
Channels: the served dense top-K_LOCK (=100) hits and the served SPLADE top-100 hits, each voted through each
table with the frozen partition_ranking numerics (rank votes 1/(K0+r), rr(sum)+rr(max)).
Fusion: the frozen rrf_partitions over the channel rankings (RL-dense first: the frozen tie-break channel).
  PRIMARY    RL+R0+R1+R2   (8 channels)
  SECONDARY  RL+R0+R1      (6 channels; fallback if the primary fails the confirmation rule)
No graph is walked at query time; no parameter is introduced; the block budget stays P50."""
import numpy as np
import scipy.sparse as sp

import _l1s_core as S

ARMS = {"PRIMARY": ("RL", "R0", "R1", "R2"), "SECONDARY": ("RL", "R0", "R1")}


def _table(N, npart, r, p):
    keys = np.unique(r * np.int64(npart) + p)
    rr = keys // npart
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(mlen)
    return (ptr, (keys % npart).astype(np.int64))


def tables(D):
    """D: _l1s_core.Data -> {"R0","RL","R1","R2"}: (ptr, flat) node -> blocks tables."""
    N, npart = D.N, D.npart
    hard = D.hard.astype(np.int64)
    xadj, adj = D.cd.struct_csr(directed=False)
    adj = np.asarray(adj, np.int64)
    D.cd._csr.clear()
    own = np.arange(N, dtype=np.int64)
    src1 = np.repeat(own, np.diff(xadj))
    R1 = _table(N, npart, np.concatenate([own, src1]), np.concatenate([hard, hard[adj]]))
    B = D.C.blockmat().astype(np.float32)
    A1 = sp.csr_matrix((np.ones(len(src1), np.float32), (src1, adj)), shape=(N, N))
    NB1 = ((A1 + sp.eye(N, format="csr", dtype=np.float32)) @ B).tocsr()
    NB1.data[:] = 1.0
    span = np.diff(NB1.indptr)
    keep = (span <= S.P_MAIN).astype(np.float32)
    NB2 = ((A1 @ sp.diags(keep)) @ NB1 + NB1).tocsr()
    NB2.sort_indices()
    R2 = (NB2.indptr.astype(np.int64), NB2.indices.astype(np.int64))
    return {"R0": D.mem_hard, "RL": D.mem, "R1": R1, "R2": R2}


def channels(D, T):
    nq, npart = D.nq, D.npart
    CH = {}
    for k, mem in T.items():
        CH[k] = (S.canon_channel_rank([D.d_ids[i, :S.K_LOCK] for i in range(nq)], mem, npart),
                 S.canon_channel_rank([D.s_ids[i, :S.K_LOCK] for i in range(nq)], mem, npart))
    return CH


def arm_rank(CH, sub):
    chans = []
    for k in sub:
        chans += list(CH[k])
    return S.rrf_ranks(chans)


def all_arm_ranks(D):
    T = tables(D)
    CH = channels(D, T)
    return {name: arm_rank(CH, sub) for name, sub in ARMS.items()}, T
