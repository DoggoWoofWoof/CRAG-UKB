"""Refinements of the one universal static candidate (undirected 1-hop table fused with BASE), DEV_A, any cache.
Arms: hub rule (a hit whose table spans > P50 blocks votes for its own block only -- budget-derived, no tuning),
mass conservation (vote / table size), own-block (hard) channels added for precision, and combinations."""
import os
import sys

import numpy as np

import _l1s_core as S

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = S.Data(name, dense_fp32=False)
C = D.C
npart, nq, N = D.npart, D.nq, D.N
res = {"cache": name, "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}}
S.log("%s DEV_A n=%d BASE %.4f" % (name, int(D.A.sum()), res["BASE_A"]["ALL"]))
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


mem1 = table(np.concatenate([own, src1]), np.concatenate([hard, hard[adj]]))
span1 = np.diff(mem1[0])
hub = span1 > S.P_MAIN
S.log("mem1: mean span %.2f; hubs (span > P50) %d nodes (%.3f%%)" % (span1.mean(), hub.sum(), 100.0 * hub.mean()))
# hub rule: hubs keep only their own block
keep = ~hub[src1]
mem1h = table(np.concatenate([own, src1[keep]]), np.concatenate([hard, hard[adj[keep]]]))
# same rule applied to the legacy (directed out) table
xo, ao = D.cd.struct_csr(directed=True)
ao = np.asarray(ao, np.int64)
D.cd._csr.clear()
srco = np.repeat(own, np.diff(xo))
spano = np.diff(D.mem[0])
hubo = spano > S.P_MAIN
keepo = ~hubo[srco]
memLh = table(np.concatenate([own, srco[keepo]]), np.concatenate([hard, hard[ao[keepo]]]))


def ch(ids, mem, conserve=False):
    if not conserve:
        return S.canon_channel_rank([ids[i, :S.K_LOCK] for i in range(nq)], mem, npart)
    ptr, flat = mem
    k = S.K_LOCK
    w = np.tile(S.rankvec(k), (nq, 1))
    cnt = (ptr[ids[:, :k] + 1] - ptr[ids[:, :k]]).astype(np.float64)
    Sb = S.hits_to_blocks(ids[:, :k], w / np.maximum(cnt, 1), mem, npart, "sum")
    Sm = S.hits_to_blocks(ids[:, :k], w / np.maximum(cnt, 1), mem, npart, "max")
    return np.argsort(-(S.rank_votes(Sb) + S.rank_votes(Sm)), axis=1).astype(np.int64)


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]


rd0, rs0 = ch(D.d_ids, D.mem), ch(D.s_ids, D.mem)
rdh, rsh = ch(D.d_ids, D.mem_hard), ch(D.s_ids, D.mem_hard)
rd1, rs1 = ch(D.d_ids, mem1), ch(D.s_ids, mem1)
rd1h, rs1h = ch(D.d_ids, mem1h), ch(D.s_ids, mem1h)
rd1c, rs1c = ch(D.d_ids, mem1, True), ch(D.s_ids, mem1, True)
rdLh, rsLh = ch(D.d_ids, memLh), ch(D.s_ids, memLh)
run("BASE", S.rrf_ranks([rd0, rs0]))
run("BASE hub rule on the legacy table", S.rrf_ranks([rdLh, rsLh]))
run("mem1 + BASE (RRF 4)", S.rrf_ranks([rd0, rs0, rd1, rs1]))
run("mem1 hub rule (replaces legacy)", S.rrf_ranks([rd1h, rs1h]))
run("mem1 hub rule + BASE (RRF 4)", S.rrf_ranks([rd0, rs0, rd1h, rs1h]))
run("mem1 mass-conserved + BASE (RRF 4)", S.rrf_ranks([rd0, rs0, rd1c, rs1c]))
run("mem1 mass-conserved (replaces legacy)", S.rrf_ranks([rd1c, rs1c]))
run("own-block channels + BASE (RRF 4)", S.rrf_ranks([rd0, rs0, rdh, rsh]))
run("own-block + mem1 + BASE (RRF 6)", S.rrf_ranks([rd0, rs0, rdh, rsh, rd1, rs1]))
run("own-block + mem1 hub rule + BASE (RRF 6)", S.rrf_ranks([rd0, rs0, rdh, rsh, rd1h, rs1h]))
run("own-block + mem1 (RRF 4, no legacy)", S.rrf_ranks([rdh, rsh, rd1, rs1]))
# fused at the block-vote level instead of RRF: sum of the legacy and undirected block votes per channel
w = np.tile(S.rankvec(S.K_LOCK), (nq, 1))
def votes(ids, mem):
    Sb = S.hits_to_blocks(ids[:, :S.K_LOCK], w, mem, npart, "sum")
    Sm = S.hits_to_blocks(ids[:, :S.K_LOCK], w, mem, npart, "max")
    return Sb, Sm
Sd0, Md0 = votes(D.d_ids, D.mem); Ss0, Ms0 = votes(D.s_ids, D.mem)
Sd1, Md1 = votes(D.d_ids, mem1); Ss1, Ms1 = votes(D.s_ids, mem1)
rdv = np.argsort(-(S.rank_votes(Sd0 + Sd1) + S.rank_votes(np.maximum(Md0, Md1))), axis=1)
rsv = np.argsort(-(S.rank_votes(Ss0 + Ss1) + S.rank_votes(np.maximum(Ms0, Ms1))), axis=1)
run("legacy + undirected votes summed per channel (RRF 2)", S.rrf_ranks([rdv, rsv]))
S.wj(os.path.join(S.OUT, "mem1_A_%s.json" % name), res)
