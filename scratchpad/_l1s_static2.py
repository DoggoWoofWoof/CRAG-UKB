"""Static node -> blocks tables and block-graph propagation (no query-time traversal): DEV_A ladder.

BASE's membership is a static table mem1_out[node] = {own block} u {blocks of the DIRECTED out-neighbours}.
Arms below only change that precomputed table or add a precomputed block-block matrix:
  M-undirected   mem1[node] = own u blocks of ALL structural neighbours (in + out)
  M-2hop         mem2[node] = mem1 u blocks of the 2-hop neighbourhood (undirected), rank votes
  D-blockgraph   S1 = S0 + S0 @ W~  with W~ the row-normalised STRUCT (or SK) cut-weight matrix (D: static SAFE ordering)
Diagnostics first: reach of the gold blocks under each table from the top-K hits (a block with zero evidence
cannot be ranked by any aggregation)."""
import os
import sys

import numpy as np
import scipy.sparse as sp

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name, dense_fp32=False)
C = D.C
npart, nq, N = D.npart, D.nq, D.N
res = {"cache": name, "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}, "diag": {}}
S.log("%s DEV_A n=%d BASE ALL %.4f" % (name, int(D.A.sum()), res["BASE_A"]["ALL"]))


# ------------------------------------------------------------------ static tables
def table_from_pairs(r, p):
    keys = np.unique(r.astype(np.int64) * np.int64(npart) + p.astype(np.int64))
    rr = keys // npart
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(mlen)
    return (ptr, (keys % npart).astype(np.int64))


hard = D.hard.astype(np.int64)
xadj, adj = D.cd.struct_csr(directed=False)          # undirected, deduped, loop-free (the frozen expansion graph)
adj = np.asarray(adj, np.int64)
deg = np.diff(xadj).astype(np.int64)
own = np.arange(N, dtype=np.int64)
src1 = np.repeat(own, deg)
mem1 = table_from_pairs(np.concatenate([own, src1]), np.concatenate([hard, hard[adj]]))
S.log("mem1 (undirected 1-hop): mean blocks/node %.2f (legacy out-only %.2f)" % (np.diff(mem1[0]).mean(), np.diff(D.mem[0]).mean()))
# 2-hop: blocks of neighbours-of-neighbours (undirected).  Built per node in chunks with the block-membership matrix:
B = C.blockmat().astype(np.float32)                    # N x npart
A1 = sp.csr_matrix((np.ones(len(src1), np.float32), (src1, adj)), shape=(N, N))
NB1 = ((A1 + sp.eye(N, format="csr", dtype=np.float32)) @ B).tocsr()   # node -> block counts at radius <= 1
NB1.data[:] = 1.0
NB2 = (A1 @ NB1).tocsr()                               # node -> block counts at radius <= 2 (multiplicity = #paths)
NB2 = (NB2 + NB1).tocsr()
NB2.sort_indices()
mem2 = (NB2.indptr.astype(np.int64), NB2.indices.astype(np.int64))
S.log("mem2 (undirected <=2-hop): mean blocks/node %.2f, max %d" % (np.diff(mem2[0]).mean(), np.diff(mem2[0]).max()))

# ------------------------------------------------------------------ reach diagnostics (DEV_A)
A = np.nonzero(D.A)[0]
hops = C.hops


def reach(mem, k):
    ptr, flat = mem
    ok_all = np.zeros(len(A), bool)
    for j, i in enumerate(A):
        blocks = set()
        for ids in (D.d_ids[i, :k], D.s_ids[i, :k]):
            for nd in ids:
                blocks.update(flat[ptr[nd]:ptr[nd + 1]].tolist())
        ok_all[j] = all(p in blocks for p in C.gb[i])
    return ok_all


def by_hop(v, idx=A):
    return {("hop%d" % h): round(float(v[hops[idx] == h].mean()), 4) for h in sorted(set(int(x) for x in hops[idx] if x >= 0))}


for tag, mem in (("legacy_out", D.mem), ("hard", D.mem_hard), ("mem1_undirected", mem1), ("mem2_undirected", mem2)):
    rk = reach(mem, S.K_LOCK)
    res["diag"]["reach_all_gold_blocks_%s_K100" % tag] = {"ALL": round(float(rk.mean()), 4), "per_hop": by_hop(rk)}
    S.log("  reach (all gold blocks have >0 evidence from top-100 dense+SPLADE hits) %-16s %.4f %s" % (tag, rk.mean(), by_hop(rk)))

# ------------------------------------------------------------------ arms
def channel(ids, k, mem, how="canon"):
    if how == "canon":
        return S.canon_channel_rank([ids[i, :k] for i in range(nq)], mem, npart)
    w = np.tile(S.rankvec(k), (nq, 1))
    Sb = S.hits_to_blocks(ids[:, :k], w, mem, npart, how, sizes=D.sizes)
    return np.argsort(-Sb, axis=1, kind="stable").astype(np.int64), Sb


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]
    return allv


rd0 = channel(D.d_ids, S.K_LOCK, D.mem)
rs0 = channel(D.s_ids, S.K_LOCK, D.mem)
r0 = S.rrf_ranks([rd0, rs0])
base_allv = run("A0 BASE", r0)
for tag, mem in (("mem1_undirected", mem1), ("mem2_undirected", mem2)):
    rd = channel(D.d_ids, S.K_LOCK, mem)
    rs = channel(D.s_ids, S.K_LOCK, mem)
    run("M %s (replaces legacy table)" % tag, S.rrf_ranks([rd, rs]))
    run("M %s + BASE (RRF 4)" % tag, S.rrf_ranks([rd0, rs0, rd, rs]))
# hub-damped 2-hop table: divide a node's 2-hop votes by its 2-hop table size (mass conservation per hit)
ptr2, flat2 = mem2
def channel_conserved(ids, k, mem):
    ptr, flat = mem
    w = np.tile(S.rankvec(k), (nq, 1))
    cnt = (ptr[ids[:, :k] + 1] - ptr[ids[:, :k]]).astype(np.float64)
    Sb = S.hits_to_blocks(ids[:, :k], w / np.maximum(cnt, 1), mem, npart, "sum")
    return np.argsort(-Sb, axis=1, kind="stable").astype(np.int64)
rd2c = channel_conserved(D.d_ids, S.K_LOCK, mem2)
rs2c = channel_conserved(D.s_ids, S.K_LOCK, mem2)
run("M mem2 mass-conserved (vote / table size)", S.rrf_ranks([rd2c, rs2c]))
run("M mem2 mass-conserved + BASE (RRF 4)", S.rrf_ranks([rd0, rs0, rd2c, rs2c]))

# ------------------------------------------------------------------ budget-consistent 2-hop table: do not pass through an
# intermediate whose neighbours span more than P_MAIN blocks (such a node's 2-hop contribution can never fit the budget)
span = np.diff(mem1[0])                                                  # blocks spanned by own + neighbours
for tau_tag, tau in (("span<=P50", S.P_MAIN), ("span<=25", 25), ("span<=100", 100)):
    keep = (span <= tau).astype(np.float32)
    A1k = A1 @ sp.diags(keep)                                            # only edges INTO non-hub intermediates
    NB2k = (A1k @ NB1 + NB1).tocsr()
    NB2k.sort_indices()
    mem2k = (NB2k.indptr.astype(np.int64), NB2k.indices.astype(np.int64))
    rk = reach(mem2k, S.K_LOCK)
    S.log("  mem2 %s: mean blocks/node %.1f | reach %.4f %s" % (tau_tag, np.diff(mem2k[0]).mean(), rk.mean(), by_hop(rk)))
    rd = channel(D.d_ids, S.K_LOCK, mem2k)
    rs = channel(D.s_ids, S.K_LOCK, mem2k)
    run("M mem2 %s (replaces legacy table)" % tau_tag, S.rrf_ranks([rd, rs]))
    run("M mem2 %s + BASE (RRF 4)" % tau_tag, S.rrf_ranks([rd0, rs0, rd, rs]))
    rd1 = channel(D.d_ids, S.K_LOCK, mem1)
    rs1 = channel(D.s_ids, S.K_LOCK, mem1)
    run("M mem1 + mem2 %s + BASE (RRF 6)" % tau_tag, S.rrf_ranks([rd0, rs0, rd1, rs1, rd, rs]))

# ------------------------------------------------------------------ D: block-graph propagation with precomputed cut weights
for fam in ("struct", "sk"):
    W = D.cut_weights(fam).astype(np.float64)
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    Wn = sp.diags(1.0 / np.maximum(rowsum, 1)) @ W                     # row-normalised
    nnz_frac = W.nnz / float(npart * npart)
    S.log("  block graph %s: nnz fraction %.3f, mean cut weight %.1f" % (fam, nnz_frac, W.data.mean()))
    for ch_tag, ids in (("dense", D.d_ids), ("splade", D.s_ids)):
        pass
    # block scores of the BASE channels (rank votes, legacy table), one propagation step
    w = np.tile(S.rankvec(S.K_LOCK), (nq, 1))
    Sd = S.hits_to_blocks(D.d_ids[:, :S.K_LOCK], w, D.mem, npart, "sum")
    Ss = S.hits_to_blocks(D.s_ids[:, :S.K_LOCK], w, D.mem, npart, "sum")
    for tag, Sd1, Ss1 in (("S0 + S0 W~", Sd + Sd @ Wn.T.toarray(), Ss + Ss @ Wn.T.toarray()),
                          ("S0 W~ only", Sd @ Wn.T.toarray(), Ss @ Wn.T.toarray())):
        rd = np.argsort(-Sd1, axis=1, kind="stable")
        rs = np.argsort(-Ss1, axis=1, kind="stable")
        run("D blockgraph %s %s" % (fam, tag), S.rrf_ranks([rd, rs]))
        run("D blockgraph %s %s + BASE (RRF 4)" % (fam, tag), S.rrf_ranks([rd0, rs0, rd, rs]))
    # static SAFE ordering: keep BASE's top blocks, re-rank the tail by cut weight to the head
    head = 25
    S0f = np.zeros((nq, npart))
    for i in range(nq):
        S0f[i, r0[i, :head]] = 1.0 / (S.K0 + np.arange(head))
    tail = S0f @ Wn.T.toarray()
    rank_safe = np.empty((nq, npart), np.int64)
    for i in range(nq):
        h = r0[i, :head]
        rest = np.setdiff1d(np.arange(npart), h)
        o = rest[np.argsort(-tail[i, rest], kind="stable")]
        rank_safe[i] = np.concatenate([h, o])
    run("D static SAFE: BASE head 25 + cut-weight tail (%s)" % fam, rank_safe)

S.wj(os.path.join(S.OUT, "static2_A_%s.json" % name), res)
