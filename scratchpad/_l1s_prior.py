"""Diagnostic: is there a QUERY-INDEPENDENT static block prior hiding in the direct SPLADE signature?
(a) block degree-mass prior (sum of structural degrees of the members) -> P50 -> ALL on DEV_A
(b) query-invariance of the direct SPLADE mean-pool ranking (top-50 overlap across queries)
(c) the prior fused with BASE / with the 2-hop table channel."""
import os
import sys

import numpy as np
import scipy.sparse as sp

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name, dense_fp32=False)
C = D.C
npart, nq, N = D.npart, D.nq, D.N
res = {"cache": name, "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}}
S.log("%s DEV_A n=%d BASE %.4f" % (name, int(D.A.sum()), res["BASE_A"]["ALL"]))
xadj, adj = D.cd.struct_csr(directed=False)
deg = np.diff(xadj).astype(np.float64)
mass = np.bincount(D.hard, weights=deg, minlength=npart)
prior = np.argsort(-mass)
prior_rank = np.tile(prior, (nq, 1))


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]
    return allv


run("PRIOR block degree mass (query independent)", prior_rank)
run("PRIOR block size (query independent)", np.tile(np.argsort(-D.sizes.astype(np.float64)), (nq, 1)))
Pm = D.splade_pool("mean")
Ss = np.asarray((D.Qs @ Pm.T).todense(), np.float64)
r_sp = np.argsort(-Ss, axis=1, kind="stable")
top = [set(r_sp[i, :50].tolist()) for i in range(nq)]
ov = np.mean([len(top[i] & top[j]) / 50.0 for i in range(0, nq, 7) for j in range(i + 1, nq, 13)])
S.log("direct SPLADE mean-pool: mean top-50 overlap between different queries %.3f (1 = query-independent)" % ov)
mass_sp = np.asarray(Pm.sum(axis=1)).ravel()
S.log("  Spearman(block SPLADE term mass, degree mass) = %.3f" % np.corrcoef(np.argsort(np.argsort(mass_sp)), np.argsort(np.argsort(mass)))[0, 1])
res["splade_pool_top50_overlap_between_queries"] = float(ov)
rd0 = S.canon_channel_rank([D.d_ids[i, :S.K_LOCK] for i in range(nq)], D.mem, npart)
rs0 = S.canon_channel_rank([D.s_ids[i, :S.K_LOCK] for i in range(nq)], D.mem, npart)
run("PRIOR + BASE (RRF 3)", S.rrf_ranks([rd0, rs0, prior_rank]))
# 2-hop budget table
hard = D.hard.astype(np.int64)
own = np.arange(N)
src1 = np.repeat(own, np.diff(xadj))
B = C.blockmat().astype(np.float32)
A1 = sp.csr_matrix((np.ones(len(src1), np.float32), (src1, np.asarray(adj, np.int64))), shape=(N, N))
NB1 = ((A1 + sp.eye(N, format="csr", dtype=np.float32)) @ B).tocsr()
NB1.data[:] = 1.0
span = np.diff(NB1.indptr)
keep = (span <= S.P_MAIN).astype(np.float32)
NB2 = ((A1 @ sp.diags(keep)) @ NB1 + NB1).tocsr()
NB2.sort_indices()
mem2 = (NB2.indptr.astype(np.int64), NB2.indices.astype(np.int64))
rd2 = S.canon_channel_rank([D.d_ids[i, :S.K_LOCK] for i in range(nq)], mem2, npart)
rs2 = S.canon_channel_rank([D.s_ids[i, :S.K_LOCK] for i in range(nq)], mem2, npart)
run("mem2 span<=P50 + BASE (RRF 4)", S.rrf_ranks([rd0, rs0, rd2, rs2]))
run("mem2 span<=P50 + BASE + PRIOR (RRF 5)", S.rrf_ranks([rd0, rs0, rd2, rs2, prior_rank]))
# how concentrated are gold blocks?  fraction of all DEV_A gold-block incidences falling in the top-50 prior blocks
A = np.nonzero(D.A)[0]
inc = np.zeros(npart)
for i in A:
    for p in C.gb[i]:
        inc[p] += 1
S.log("gold-block incidences (DEV_A): %.3f fall in the 50 highest-degree-mass blocks (11.6%% of blocks); Gini-like top-50 share of sizes %.3f" % (
    inc[prior[:50]].sum() / inc.sum(), D.sizes[prior[:50]].sum() / D.sizes.sum()))
res["gold_incidence_in_top50_prior_blocks"] = float(inc[prior[:50]].sum() / inc.sum())
S.wj(os.path.join(S.OUT, "prior_A_%s.json" % name), res)
