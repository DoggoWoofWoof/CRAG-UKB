"""Multi-radius static evidence (DEV_A, any cache): every hit votes through static node -> blocks tables of
radius 0 (own block), the legacy directed-out table (BASE), radius 1 (undirected) and radius 2 (undirected,
budget rule: no intermediate whose neighbours span > P50 blocks).  Dense and SPLADE each; frozen rank votes,
frozen per-channel rank transform, frozen RRF over all channels.  Subsets of the tables are the arms."""
import itertools
import os
import sys

import numpy as np
import scipy.sparse as sp

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name, dense_fp32=False)
C = D.C
PTAG = "frozen"
if len(sys.argv) > 2:                       # optional scratch partition: results/L1_STATIC/parts/<ds>__<TAG>.npy
    PTAG = sys.argv[2]
    D.override_partition(np.load(os.path.join(S.OUT, "parts", "%s__%s.npy" % (name, PTAG))), PTAG)
npart, nq, N = D.npart, D.nq, D.N
res = {"cache": name, "partition": PTAG, "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}}
S.log("%s [%s] DEV_A n=%d BASE %.4f" % (name, PTAG, int(D.A.sum()), res["BASE_A"]["ALL"]))
hard = D.hard.astype(np.int64)
xadj, adj = D.cd.struct_csr(directed=False)
adj = np.asarray(adj, np.int64)
D.cd._csr.clear()
own = np.arange(N, dtype=np.int64)
src1 = np.repeat(own, np.diff(xadj))


def table(r, p):
    keys = np.unique(r * np.int64(npart) + p)
    rr = keys // npart
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(mlen)
    return (ptr, (keys % npart).astype(np.int64))


R1 = table(np.concatenate([own, src1]), np.concatenate([hard, hard[adj]]))
B = C.blockmat().astype(np.float32)
A1 = sp.csr_matrix((np.ones(len(src1), np.float32), (src1, adj)), shape=(N, N))
NB1 = ((A1 + sp.eye(N, format="csr", dtype=np.float32)) @ B).tocsr()
NB1.data[:] = 1.0
span = np.diff(NB1.indptr)
keep = (span <= S.P_MAIN).astype(np.float32)
NB2 = ((A1 @ sp.diags(keep)) @ NB1 + NB1).tocsr()
NB2.sort_indices()
R2 = (NB2.indptr.astype(np.int64), NB2.indices.astype(np.int64))
TABLES = {"R0": D.mem_hard, "RL": D.mem, "R1": R1, "R2": R2}
S.log("table spans: " + ", ".join("%s %.1f" % (k, np.diff(v[0]).mean()) for k, v in TABLES.items()))
CH = {}
for k, mem in TABLES.items():
    CH[k] = (S.canon_channel_rank([D.d_ids[i, :S.K_LOCK] for i in range(nq)], mem, npart),
             S.canon_channel_rank([D.s_ids[i, :S.K_LOCK] for i in range(nq)], mem, npart))


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]


order = ["RL", "R0", "R1", "R2"]
for n in range(1, 5):
    for sub in itertools.combinations(order, n):
        chans = []
        for k in sub:
            chans += list(CH[k])
        # RL first when present (frozen dense tie-break semantics), else the first table's dense channel
        run("+".join(sub), S.rrf_ranks(chans))
S.wj(os.path.join(S.OUT, "radius_A_%s%s.json" % (name, "" if PTAG == "frozen" else "__" + PTAG)), res)
