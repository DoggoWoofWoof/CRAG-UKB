"""Controls for the directional block signal D1d / D2d (DEV_A rows only, other rows carry BASE):
    true      per-seed block max_{v in B} cos(q - e_s, e_v - e_s) -> frozen RRF over the 5 seeds        (= D1d of _l1g_dir.py)
    reversed  the same with the direction e_s - q                     (wrong orientation, same geometry)
    random    the same with a random unit direction per (query, seed) (no orientation, same geometry; rng seed 0)
    qmax      block max_{v in B} (q . e_v) -- exhaustive dense block max without any seed / direction (aggregation control)
    pooled    'true' restricted to the served candidate pool dense top-1000 u SPLADE top-1000 (scalability variant)
    size      blocks by size, descending (size-bias control)
Each control alone and fused with BASE through the frozen RRF (3 channels), P50, DEV_A ALL vs BASE.  Nothing fitted."""
import os
import sys
import time

import numpy as np

import _l1g_core as G
import _ta_prepartition as TA

name = sys.argv[1] if len(sys.argv) > 1 else "musique"
t0 = time.time()
D = G.Data(name, dense_fp32=True)
C = D.C
E, Q = D.E, D.Q
N, dim = E.shape
npart, nq = D.npart, D.nq
m = D.A
rows = np.flatnonzero(m)
K0, KL = G.K0, G.K_LOCK
seeds = C.seeds.astype(np.int64)
ns = seeds.shape[1]
res = {"cache": name, "partition": G.PARTITION_OF[name], "n_DEV_A": int(m.sum()), "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}}
G.log("%s [%s] DEV_A n=%d N=%d blocks %d BASE %.4f" % (name, G.PARTITION_OF[name], int(m.sum()), N, npart, res["BASE_A"]["ALL"]))
hard = D.hard.astype(np.int64)
perm = np.argsort(hard, kind="stable")
starts = np.searchsorted(hard[perm], np.arange(npart))
sizes = np.bincount(hard, minlength=npart)
Es = E[seeds]
R = Q[:, None, :] - Es
Rn = R / (np.linalg.norm(R, axis=2, keepdims=True) + 1e-9)
rng = np.random.default_rng(0)
Rr = rng.standard_normal(Rn.shape).astype(np.float32)
Rr /= (np.linalg.norm(Rr, axis=2, keepdims=True) + 1e-9)
rs_true = np.einsum("qsd,qsd->qs", Rn, Es)
rs_rand = np.einsum("qsd,qsd->qs", Rr, Es)
pool = [set(D.d_ids[i].tolist()) | set(D.s_ids[i].tolist()) for i in range(nq)]


def blockmax_rank(cos_qsN):
    """(n, ns, N) per-seed node scores -> per-seed block max over actual nodes -> frozen RRF over seeds -> (n, npart)."""
    n = cos_qsN.shape[0]
    bms = np.maximum.reduceat(cos_qsN.reshape(n * ns, N)[:, perm], starts, axis=1).reshape(n, ns, npart)
    rks = [np.argsort(-bms[:, j], axis=1, kind="stable").astype(np.int64) for j in range(ns)]
    return G.S.rrf_ranks(rks)


base_full = G.F0([G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)], npart)     # (nq, npart) frozen RRF order
blk = {k: base_full.copy() for k in ("true", "reversed", "random", "qmax", "pooled")}
cq = max(8, min(200, int(2e7 // (ns * N))))
for a in range(0, len(rows), cq):
    idx = rows[a:a + cq]
    n = len(idx)
    V = np.concatenate([Rn[idx].reshape(n * ns, dim), Es[idx].reshape(n * ns, dim), Rr[idx].reshape(n * ns, dim), Q[idx]], axis=0)
    P = V @ E.T
    A = P[:n * ns].reshape(n, ns, N)
    B = P[n * ns:2 * n * ns].reshape(n, ns, N)
    Ar = P[2 * n * ns:3 * n * ns].reshape(n, ns, N)
    QE = P[3 * n * ns:]                                                     # (n, N) plain dense similarity
    den = np.sqrt(np.maximum(2.0 - 2.0 * B, 1e-12))
    cos = (A - rs_true[idx][:, :, None]) / den
    cos_r = (Ar - rs_rand[idx][:, :, None]) / den
    for j in range(ns):
        cos[np.arange(n), j, seeds[idx, j]] = -2.0
        cos_r[np.arange(n), j, seeds[idx, j]] = -2.0
    blk["true"][idx] = blockmax_rank(cos)
    blk["reversed"][idx] = blockmax_rank(np.where(cos > -2.0, -cos, -2.0))
    blk["random"][idx] = blockmax_rank(cos_r)
    cp = np.full_like(cos, -2.0)
    for i in range(n):
        pl = np.fromiter(pool[idx[i]], np.int64)
        cp[i][:, pl] = cos[i][:, pl]
    blk["pooled"][idx] = blockmax_rank(cp)
    qb = np.maximum.reduceat(QE[:, perm], starts, axis=1)
    blk["qmax"][idx] = np.argsort(-qb, axis=1, kind="stable").astype(np.int64)
    if a == 0:
        G.log("  chunk %d queries in %.0fs (chunks of %d)" % (n, time.time() - t0, cq))
G.log("  scored controls for %d DEV_A queries in %.0fs" % (len(rows), time.time() - t0))
del E, D.E
size_rank = np.tile(np.argsort(-sizes, kind="stable").astype(np.int64)[None, :], (nq, 1))
blk["size"] = size_rank
rd, rs_ = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)


def run(tag, rank):
    out, _ = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]


run("D0 BASE (frozen RRF)", G.F0([rd, rs_], npart))
for k in ("true", "reversed", "random", "qmax", "pooled", "size"):
    run("C1 %-8s | alone" % k, blk[k])
    run("C2 %-8s | + BASE (frozen RRF, 3 ch)" % k, G.F0([rd, rs_, blk[k]], npart))
res["block_sizes"] = {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())}
G.S.wj(os.path.join(G.OUT, "dircontrol_A_%s.json" % name), res)
G.log("done %.0fs" % (time.time() - t0))
