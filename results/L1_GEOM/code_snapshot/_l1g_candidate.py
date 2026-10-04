"""L1_GEOM candidate arms, fixed after the DEV_A factorial (results/L1_GEOM/ladder_A_*.json, dir_A_*.json) and BEFORE any
DEV_B look.

Channels (block rankings; the first three go through the frozen legacy node -> blocks table and the frozen
partition_ranking with K_LOCK = 100 hits):
    Cd    Dense(q)   served top-100 (replay cache)
    Cs    SPLADE(q)  served top-100 (replay cache)
    Cg    Dense(2q - mu)  exact top-100 over the canonical dense index; mu = mean of the frozen SEED_K = 5 seed embeddings
                          (unit-normalised); seeds = the cache's frozen RRF top-5 nodes.  One extra query point, no traversal.
    Cdir  directional block signal over ACTUAL nodes: for each seed s_i, D_i(v) = cos(q - e_{s_i}, e_v - e_{s_i}) for every
          node v != s_i (exact, all nodes); block score_i(B) = max_{v in B} D_i(v); the 5 per-seed block rankings are fused
          by the frozen RRF (K0 = 60) -> one block ranking.  No centroid, no table, no relation labels, no traversal.
    Cmax  exhaustive dense block max: score(B) = max_{v in B} q . e_v over all actual nodes (no seeds, no direction) -- the
          aggregation control of the directional signal (results/L1_GEOM/dircontrol_A_*.json), carried as a decomposition arm.
Fusions (parameter-free; P50 = the first 50 blocks of the fused ranking):
    F0   the frozen RRF (K0 = 60)                              -- the reference selector's fusion
    IL   round-robin interleave: block key = min_c (pos_c * C + c), ascending (C = number of channels)
Arms:
    PRIMARY    D2d    = F0(Cd, Cs, Cdir)   the directional cell D2 exactly as specified (per-seed block max -> RRF over seeds
                                           -> the same RRF with the Dense and SPLADE block ranks -> P50)
    SECONDARY  G2_IL  = IL(Cd, Cs, Cg)     the point-geometry cell G2 with the best parameter-free fusion of the DEV_A ladder
    FUSION     IL     = IL(Cd, Cs)         fusion lever alone (decomposition, reported, never decided on)
    GEOMETRY   G2_F0  = F0(Cd, Cs, Cg)     point-geometry lever alone through the frozen RRF (decomposition, reported)
    AGGREGATION QMAX_F0 = F0(Cd, Cs, Cmax) exhaustive dense block max as the third channel (decomposition: how much of the
                                           PRIMARY's gain is the orientation, how much the exhaustive block-max aggregation)
Reference: BASE = F0(Cd, Cs) = the frozen selector.  Zero fitted parameters anywhere; nothing is walked; no dataset rule."""
import numpy as np

import _l1g_core as G

ARMS = {"PRIMARY": "D2d", "SECONDARY": "G2_IL", "FUSION": "IL", "GEOMETRY": "G2_F0", "AGGREGATION": "QMAX_F0"}
POINT = "G2 extrapolated 2q - mu"


def directional_block_rank(D):
    """Cdir for every cache row (needs D.E, the fp32 node matrix).  Same arithmetic and chunking as _l1g_dir.py (D1d)."""
    E, Q = D.E, D.Q
    N, dim = E.shape
    npart, nq = D.npart, D.nq
    seeds = D.C.seeds.astype(np.int64)
    ns = seeds.shape[1]
    hard = D.hard.astype(np.int64)
    perm = np.argsort(hard, kind="stable")
    starts = np.searchsorted(hard[perm], np.arange(npart))
    assert (np.bincount(hard, minlength=npart) > 0).all()
    Es = E[seeds]
    R = Q[:, None, :] - Es
    Rn = R / (np.linalg.norm(R, axis=2, keepdims=True) + 1e-9)
    rs_dot = np.einsum("qsd,qsd->qs", Rn, Es)
    blk = np.zeros((nq, npart), np.int64)
    cq = max(8, min(200, int(2e7 // (ns * N))))
    for a in range(0, nq, cq):
        b = min(nq, a + cq)
        n = b - a
        V = np.concatenate([Rn[a:b].reshape(n * ns, dim), Es[a:b].reshape(n * ns, dim)], axis=0)
        P = V @ E.T
        A = P[:n * ns].reshape(n, ns, N)
        B = P[n * ns:].reshape(n, ns, N)
        num = A - rs_dot[a:b][:, :, None]
        den = np.sqrt(np.maximum(2.0 - 2.0 * B, 1e-12))
        cos = num / den
        for j in range(ns):
            cos[np.arange(n), j, seeds[a:b, j]] = -2.0
        bms = np.maximum.reduceat(cos.reshape(n * ns, N)[:, perm], starts, axis=1).reshape(n, ns, npart)
        rks = [np.argsort(-bms[:, j], axis=1, kind="stable").astype(np.int64) for j in range(ns)]
        blk[a:b] = G.S.rrf_ranks(rks)
    return blk


def dense_blockmax_rank(D, cq=200):
    """Cmax for every cache row: block max over actual nodes of the plain dense similarity q . e_v (needs D.E)."""
    E, Q = D.E, D.Q
    npart, nq = D.npart, D.nq
    hard = D.hard.astype(np.int64)
    perm = np.argsort(hard, kind="stable")
    starts = np.searchsorted(hard[perm], np.arange(npart))
    blk = np.zeros((nq, npart), np.int64)
    for a in range(0, nq, cq):
        b = min(nq, a + cq)
        QE = Q[a:b] @ E.T
        qb = np.maximum.reduceat(QE[:, perm], starts, axis=1)
        blk[a:b] = np.argsort(-qb, axis=1, kind="stable").astype(np.int64)
    return blk


def channels(D):
    rd, rs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
    pts, gdiag = G.geometry(D)
    ids, _ = G.retrieve(D, pts[POINT])
    rg = G.block_channel(D, ids)
    rdir = directional_block_rank(D)
    rmax = dense_blockmax_rank(D)
    return {"Cd": rd, "Cs": rs, "Cg": rg, "Cdir": rdir, "Cmax": rmax}, gdiag


def candidate_ranks(D):
    """-> ({arm key: (nq, npart) block ranking}, channels, geometry diagnostics)."""
    ch, gdiag = channels(D)
    npart = D.npart
    ranks = {
        "D2d": G.F0([ch["Cd"], ch["Cs"], ch["Cdir"]], npart),
        "G2_IL": G.F_interleave([ch["Cd"], ch["Cs"], ch["Cg"]], npart),
        "IL": G.F_interleave([ch["Cd"], ch["Cs"]], npart),
        "G2_F0": G.F0([ch["Cd"], ch["Cs"], ch["Cg"]], npart),
        "QMAX_F0": G.F0([ch["Cd"], ch["Cs"], ch["Cmax"]], npart),
    }
    return ranks, ch, gdiag
