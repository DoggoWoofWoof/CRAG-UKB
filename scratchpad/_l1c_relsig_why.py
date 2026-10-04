"""Attribution for R1 (relsig) on DEV_A: for every gold block of every hop-2/hop-3 query, the best (lowest) hit rank that voted for it
under R1, split by vote type (base = served own/out membership; compat = relation-compatible typed membership), and where the block
ended up in the fused order.  Answers: when R1 still misses a reached gold block, did a TOP hit expose it (the discriminator saw it but
could not rank it) or only a deep hit (nothing to discriminate -- a second hop is missing)?  Rebuilds the R1 tables exactly as
_l1c_relsig does (same admissibility, same R(q)); no new arm, no ranking change.

    python -u _l1c_relsig_why.py <cache>     -> results/L1_COVPART/relsig_why_A_<cache>.json
"""
import json
import os
import sys

import numpy as np

import _l1g_core as G

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
P90 = os.path.join(G.X.REPO, "results", "L1_P90_EXPLOIT")
name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
D = G.Data(name, dense_fp32=False)
C = D.C
nq, N, npart = D.nq, D.N, D.npart
m = D.A
hops = np.asarray(C.hops)
hard = D.hard.astype(np.int64)
cap = int(round(N / npart))
K0, K_LOCK = G.S.K0, G.K_LOCK
xu, au = D.cd.struct_csr(directed=False)
deg_u = np.diff(np.asarray(xu, np.int64))
D.cd._csr.clear()
hub = (deg_u + 1) > cap
s_e, d_e, r_e, _ = D.struct_edges()
vocab = D.relation_vocab()
typed_graph = (r_e is not None) and (len(vocab) >= 2)
served = D.legacy_mem()
D.cd._csr.clear()
sptr, sflat = np.asarray(served[0], np.int64), np.asarray(served[1], np.int64)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_l1c_relsig.py"), encoding="utf-8").read()
exec(src[src.index("def csr_pairs"):src.index("G.log(")])          # csr_pairs, admissible edges, typed table, RQ (identical construction)
dl = [D.d_ids[i, :K_LOCK] for i in range(nq)]
sl = [D.s_ids[i, :K_LOCK] for i in range(nq)]
BIG = 10 ** 6


def votes(node_lists):
    """per query: S (float32 as in R1), best hit rank per block with any vote, best hit rank per block with a compat vote."""
    S = np.zeros((nq, npart), np.float32)
    M = np.zeros((nq, npart), np.float32)
    best_any = np.full((nq, npart), BIG, np.int64)
    best_comp = np.full((nq, npart), BIG, np.int64)
    for qi in range(nq):
        rq = RQ[qi]
        for r, nd in enumerate(node_lists[qi]):
            nd = int(nd)
            if nd < 0 or nd >= N:
                continue
            w = 1.0 / (K0 + r)
            bb = sflat[sptr[nd]:sptr[nd + 1]]
            eb = np.zeros(0, np.int64)
            if len(rq):
                a, b = T_ptr[nd], T_ptr[nd + 1]
                sel = np.isin(T_rel[a:b], rq)
                if sel.any():
                    eb = np.unique(T_blk[a:b][sel])
            if len(eb) == 0:
                ps, v = bb, np.full(len(bb), w, np.float32)
            else:
                ps, cnt = np.unique(np.concatenate([bb, eb]), return_counts=True)
                v = (w * cnt).astype(np.float32)
            S[qi, ps] += v
            np.maximum.at(M[qi], ps, v)
            best_any[qi, ps] = np.minimum(best_any[qi, ps], r)
            if len(eb):
                best_comp[qi, eb] = np.minimum(best_comp[qi, eb], r)

    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), np.int32)
        rank[np.arange(nq)[:, None], order] = np.arange(npart)[None, :]
        return 1.0 / (K0 + rank)
    return np.argsort(-(rr(S) + rr(M)), axis=1).astype(np.int64), S, best_any, best_comp


Cd, Sd, ba_d, bc_d = votes(dl)
Cs, Ss, ba_s, bc_s = votes(sl)
base = G.F0([Cd, Cs], npart)
pos = G.positions(base)
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
feas = gs <= G.P_MAIN
ok = (gm & (pos < G.P_MAIN)).sum(axis=1) == gs
ev = (Sd > 0) | (Ss > 0)
reach = (gm & ~ev).sum(axis=1) == 0
best_any = np.minimum(ba_d, ba_s)
best_comp = np.minimum(bc_d, bc_s)
res = {"cache": name, "n_DEV_A": int(m.sum()), "R1_BASE_ALL": round(float(ok[m].mean()), 4), "bins": "best hit rank (0-based) that voted for the worst gold block"}


def bucket(x):
    return "rank0" if x == 0 else ("rank1-4" if x < 5 else ("rank5-19" if x < 20 else ("rank20-99" if x < BIG else "none")))


for h in (1, 2, 3):
    s = m & (hops == h)
    rw = np.nonzero(s & feas & reach & ~ok)[0]              # reached-weak / fusion-fixable failures
    hit = np.nonzero(s & ok)[0]
    out = {"n": int(s.sum()), "R1_ok": int(len(hit)), "reached_failed": int(len(rw)),
           "R_q_empty_among_reached_failed": int(sum(len(RQ[i]) == 0 for i in rw)),
           "worst_gold_block__best_ANY_vote_rank": {}, "worst_gold_block__best_COMPAT_vote_rank": {},
           "worst_gold_block_position_median": None,
           "correct_queries__worst_gold_block__best_COMPAT_vote_rank": {}}
    wp = []
    for i in rw:
        gb = np.nonzero(gm[i])[0]
        wb = gb[np.argmax(pos[i, gb])]                       # the worst-ranked gold block
        wp.append(int(pos[i, wb]))
        for key, arr in (("worst_gold_block__best_ANY_vote_rank", best_any), ("worst_gold_block__best_COMPAT_vote_rank", best_comp)):
            k = bucket(int(arr[i, wb]))
            out[key][k] = out[key].get(k, 0) + 1
    for i in hit:
        gb = np.nonzero(gm[i])[0]
        wb = gb[np.argmax(pos[i, gb])]
        k = bucket(int(best_comp[i, wb]))
        out["correct_queries__worst_gold_block__best_COMPAT_vote_rank"][k] = out["correct_queries__worst_gold_block__best_COMPAT_vote_rank"].get(k, 0) + 1
    out["worst_gold_block_position_median"] = int(np.median(wp)) if wp else None
    # how many blocks received a compat vote from a rank-0 hit (the anchor's compatible exposure) and where the gold blocks sit among them
    n_anchor_comp = [int((best_comp[i] == 0).sum()) for i in np.nonzero(s)[0]]
    out["blocks_with_a_rank0_compat_vote_per_query_mean"] = round(float(np.mean(n_anchor_comp)), 1) if n_anchor_comp else None
    out["queries_where_every_gold_block_has_a_rank0_compat_vote"] = int(sum(bool(((best_comp[i] == 0) | ~gm[i]).all()) for i in np.nonzero(s)[0]))
    out["queries_where_every_gold_block_has_a_top5_compat_vote"] = int(sum(bool(((best_comp[i] < 5) | ~gm[i]).all()) for i in np.nonzero(s)[0]))
    res["hop%d" % h] = out
G.log(json.dumps(res))
G.S.wj(os.path.join(OUT, "relsig_why_A_%s.json" % name), res)
