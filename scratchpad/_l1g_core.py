"""L1_GEOM lane -- maximise P(all gold evidence inside L1's 50 blocks) with three levers only:
   (1) partition quality (frozen Mt-KaHyPar H4_SK vs the validated Zoltan-PHG partitions of the same corpora),
   (2) parameter-free query-time fusion of the Dense / SPLADE block channels (the frozen RRF is the reference),
   (3) a universal, parameter-free query transform derived from q and its frozen SEED_K=5 retrieved entities
       (centroid mu, extrapolated 2q - mu, residual q - mu, seed-subspace projection q_hat, shifted seeds),
       each used as ONE more retrieval point through the SAME dense index -> one more block channel.
No graph traversal, no relation labels, no learned weights, no fitted constants (K0 / K_LOCK / P50 / SEED_K frozen).
Read-only over the replay caches, the canonical embeddings and the structural family (legacy node -> blocks table).
Populations: DEV_A exploration, DEV_B confirmation (sha1(query_id) parity), exactly as _l1x90_core."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1x90_core as X  # noqa: E402
import _l1s_core as S  # noqa: E402  (Data loader + frozen numerics wrappers; unchanged)
import _ta_prepartition as TA  # noqa: E402

# the validated PHG replay caches of the same corpora (same rows, same served hits, other partition)
X.CACHES.setdefault("squad_phg", os.path.join(X.REPO, "data", "l1_lowmem", "squad", "replay_cache__LOWMEM__PHG_con.npz"))
X.DS_OF.setdefault("squad_phg", "squad")
PARTITION_OF = {"metaqa": "MTKAHYPAR", "squad": "MTKAHYPAR", "musique": "PHG", "metaqa_phg": "PHG", "squad_phg": "PHG"}

OUT = os.path.join(X.REPO, "results", "L1_GEOM")
os.makedirs(OUT, exist_ok=True)
K0, K_LOCK, P_MAIN = S.K0, S.K_LOCK, S.P_MAIN
log = S.log
Data = S.Data


# ------------------------------------------------------------------------------------ block channels
def block_channel(D, ids):
    """served / computed node hits (nq, >=K_LOCK) -> frozen block ranking through the legacy table."""
    return S.canon_channel_rank([ids[i, :K_LOCK] for i in range(D.nq)], D.mem, D.npart)


def evidence(D, ids):
    """(nq, npart) bool: block received at least one vote from the top-K_LOCK hits (through the legacy table)."""
    w = np.ones((D.nq, K_LOCK), np.float64)
    return S.hits_to_blocks(ids[:, :K_LOCK], w, D.mem, D.npart, "sum") > 0


def positions(rank):
    nq, npart = rank.shape
    pos = np.empty((nq, npart), np.int64)
    pos[np.arange(nq)[:, None], rank] = np.arange(npart)[None, :]
    return pos


def order_by(score, base):
    """descending score, ties (and equal scores) broken by the base ranking order -- the frozen RRF's convention."""
    fv_in = np.take_along_axis(score, base, axis=1)
    idx2 = np.argsort(-fv_in, axis=1, kind="stable")
    return np.take_along_axis(base, idx2, axis=1).astype(np.int64)


# ------------------------------------------------------------------------------------ parameter-free fusions
def F0(ranks, npart, **kw):
    """the frozen RRF (reference)."""
    return S.rrf_ranks(ranks)


def F_masked(ranks, npart, ev=None, **kw):
    """RRF where a channel votes for a block only if it has evidence for it (no phantom mass for unseen blocks)."""
    fv = np.zeros(ranks[0].shape, np.float64)
    for rk, e in zip(ranks, ev):
        fv += (1.0 / (K0 + positions(rk))) * e
    return order_by(fv, ranks[0])


def F_mnz(ranks, npart, ev=None, **kw):
    """CombMNZ on reciprocal ranks: RRF sum x number of channels with evidence for the block."""
    fv = np.zeros(ranks[0].shape, np.float64)
    n = np.zeros(ranks[0].shape, np.float64)
    for rk, e in zip(ranks, ev):
        fv += 1.0 / (K0 + positions(rk))
        n += e
    return order_by(fv * n, ranks[0])


def F_interleave(ranks, npart, **kw):
    """round-robin merge of the channel rankings (each channel's top-25 is guaranteed inside P50 for 2 channels)."""
    C = len(ranks)
    key = np.full(ranks[0].shape, np.iinfo(np.int64).max, np.int64)
    for c, rk in enumerate(ranks):
        key = np.minimum(key, positions(rk) * C + c)
    return np.argsort(key, axis=1, kind="stable").astype(np.int64)


def agreement_set(ranks, k=P_MAIN):
    """A_q = |top-k(dense) & top-k(splade)| / k over the first two channels."""
    a = ranks[0][:, :k]
    b = ranks[1][:, :k]
    nq = a.shape[0]
    out = np.zeros(nq)
    for i in range(nq):
        out[i] = len(set(a[i].tolist()) & set(b[i].tolist())) / float(k)
    return out


def agreement_rr(ranks, k=P_MAIN):
    """A_q = sum_{b in top-k(d) & top-k(s)} 1/(r_d(b)+r_s(b)) normalised by its maximum (identical rankings): sum_r 1/(2r)."""
    pd, ps = positions(ranks[0]) + 1, positions(ranks[1]) + 1
    inter = (pd <= k) & (ps <= k)
    val = (inter / (pd + ps)).sum(axis=1)
    return val / (0.5 * np.sum(1.0 / np.arange(1, k + 1)))


def F_blend(ranks, npart, A=None, **kw):
    """agreement-blended position: A_q * pos_RRF + (1 - A_q) * pos_interleave (channels agree -> consensus order;
    channels disagree -> union order); A_q measured from the query's own rankings, nothing fitted."""
    p0 = positions(F0(ranks, npart))
    p1 = positions(F_interleave(ranks, npart))
    key = A[:, None] * p0 + (1.0 - A[:, None]) * p1
    return order_by(-key, ranks[0])


FUSIONS = {"F0 frozen RRF": F0, "F1a evidence-masked RRF": F_masked, "F1b RRF x MNZ": F_mnz,
           "F1c interleave (round robin)": F_interleave, "F1d agreement blend (set overlap)": F_blend,
           "F1e agreement blend (reciprocal-rank agreement)": F_blend}


def fuse(name, ranks, npart, ev=None):
    f = FUSIONS[name]
    kw = {"ev": ev}
    if name.startswith("F1d"):
        kw["A"] = agreement_set(ranks)
    elif name.startswith("F1e"):
        kw["A"] = agreement_rr(ranks)
    return f(ranks, npart, **kw)


# ------------------------------------------------------------------------------------ dense retrieval of arbitrary points
def retrieve(D, points, k=K_LOCK, chunk=20000):
    """points (nq, m, dim) -> ids (nq, k) of the nearest nodes by max_i cosine, exact, over the fp16 mmap shards."""
    nq, m, dim = points.shape
    P = points.reshape(nq * m, dim).astype(np.float32)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    best_s = np.full((nq, k), -np.inf, np.float32)
    best_i = np.full((nq, k), -1, np.int64)
    N = D.N
    chunk = max(1000, chunk // m)                                          # bounded (nq*m, chunk) similarity block
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        Ec = np.asarray(D._E16[a:b], np.float32)
        sims = (P @ Ec.T).reshape(nq, m, b - a).max(axis=1)              # (nq, b-a)
        cand_s = np.concatenate([best_s, sims], axis=1)
        cand_i = np.concatenate([best_i, np.arange(a, b)[None, :].repeat(nq, 0)], axis=1)
        top = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(cand_s, top, axis=1)
        best_i = np.take_along_axis(cand_i, top, axis=1)
    srt = np.argsort(-best_s, axis=1, kind="stable")
    return np.take_along_axis(best_i, srt, axis=1), np.take_along_axis(best_s, srt, axis=1)


# ------------------------------------------------------------------------------------ query geometry (frozen seeds)
def geometry(D):
    """points from q and its frozen SEED_K=5 RRF seeds; every entry is (nq, m, dim)."""
    Q = D.Q.astype(np.float32)
    seeds = D.C.seeds.astype(np.int64)
    Es = np.stack([np.asarray(D._E16[seeds[:, j]], np.float32) for j in range(seeds.shape[1])], axis=1)   # (nq, 5, dim)
    Es /= (np.linalg.norm(Es, axis=2, keepdims=True) + 1e-9)
    mu = Es.mean(axis=1)                                                  # (nq, dim)
    r = Q - mu
    Xc = Es - mu[:, None, :]                                              # (nq, 5, dim) centred seed configuration
    G = np.einsum("qid,qjd->qij", Xc, Xc)                                 # (nq, 5, 5) Gram
    Gp = np.linalg.pinv(G, rcond=1e-5)                                    # Moore-Penrose; sum_i (e_i - mu) = 0 -> rank <= 4, the
                                                                          # null direction is numerical noise and must not be inverted
    coef = np.einsum("qij,qj->qi", Gp, np.einsum("qid,qd->qi", Xc, r))
    delta = np.einsum("qi,qid->qd", coef, Xc)                             # projection of r onto span(e_i - mu)
    pts = {
        "G0 seeds (Delta = 0)": Es,
        "G1 centroid mu": mu[:, None, :],
        "G2 extrapolated 2q - mu": (2.0 * Q - mu)[:, None, :],
        "G3 residual direction q - mu": r[:, None, :],
        "G4 projected q_hat = q + P(q - mu)": (Q + delta)[:, None, :],
        "G5 shifted seeds e_i + P(q - mu)": Es + delta[:, None, :],
    }
    diag = {"norm_r_mean": float(np.linalg.norm(r, axis=1).mean()), "norm_delta_mean": float(np.linalg.norm(delta, axis=1).mean()),
            "cos_q_mu_mean": float((Q * mu).sum(1).mean() / (np.linalg.norm(mu, axis=1).mean() + 1e-9)),
            "delta_over_r_mean": float((np.linalg.norm(delta, axis=1) / (np.linalg.norm(r, axis=1) + 1e-9)).mean())}
    return pts, diag


# ------------------------------------------------------------------------------------ ceilings / diagnostics
def gold_mask(D):
    """(nq, npart) bool gold blocks."""
    M = np.zeros((D.nq, D.npart), bool)
    for i, g in enumerate(D.C.gb):
        for p in g:
            M[i, p] = True
    return M


def reach_all(D, ev_any, gm):
    """fraction of DEV_A queries whose gold blocks all have evidence."""
    ok = (gm & ~ev_any).sum(axis=1) == 0
    return ok


def ceilings(D, ranks, evs, gm, m):
    """rank-free / fusion ceilings on DEV_A: reach (any evidence), oracle-channel top-50, union of channel top-50s."""
    ev_any = np.zeros_like(gm)
    for e in evs:
        ev_any |= e
    reach = reach_all(D, ev_any, gm)
    in50 = np.zeros_like(gm)
    for rk in ranks:
        in50 |= positions(rk) < P_MAIN
    union50 = ((gm & ~in50).sum(axis=1) == 0) & (gm.sum(axis=1) <= P_MAIN)
    return {"reach_any_evidence": round(float(reach[m].mean()), 4), "union_of_channel_top50": round(float(union50[m].mean()), 4),
            "feasible_P50": round(float((gm.sum(axis=1) <= P_MAIN)[m].mean()), 4)}


def new_evidence(D, ids_new, ids_ref, gm, ev_ref, m):
    """what a transformed point adds: fraction of its top-100 hits not in dense(q)'s top-100, and whether those new hits
    reach gold blocks that had no evidence before (per DEV_A query)."""
    nq = D.nq
    ref = [set(ids_ref[i, :K_LOCK].tolist()) for i in range(nq)]
    frac_new = np.array([len(set(ids_new[i].tolist()) - ref[i]) / float(K_LOCK) for i in range(nq)])
    ev_new = evidence(D, ids_new)
    gained = ((gm & ev_new & ~ev_ref).sum(axis=1) > 0)
    reach_after = reach_all(D, ev_ref | ev_new, gm)
    gold_nodes = [set(g.tolist()) for g in D.C.gold_nodes]
    hit_gold_new = np.array([len((set(ids_new[i].tolist()) - ref[i]) & gold_nodes[i]) > 0 for i in range(nq)])
    return {"frac_new_hits": round(float(frac_new[m].mean()), 3), "q_with_new_gold_node_hit": round(float(hit_gold_new[m].mean()), 4),
            "q_gaining_unreached_gold_block": round(float(gained[m].mean()), 4), "reach_after": round(float(reach_after[m].mean()), 4)}
