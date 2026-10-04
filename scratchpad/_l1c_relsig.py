"""L1_COVPART route 1 (2026-09-14, fourth ruling of the day): a query-conditioned, parameter-free discriminator over blocks that the
served votes and the static 1-hop memberships already reach -- NO walk, NO expansion, NO learned weights, NO new membership sets.

    S_q(b) = S_base(b) + compat(q, static relation signature of the hit -> b membership)

Static (precomputed once from the edge IDs of the structural family, like the served membership table):
    TM(v) = {(P(u), rho) : (v, u, rho) admissible typed STRUCT edge}
            admissible = every directed out-edge v -> u (all v)  +  every in-edge u -> v for NON-HUB v
            (the frozen A1/UND_NONHUB admissibility; hub := deg_undirected + 1 > cap, the H4/PATH2 cap rule)
Query side (existing lexical representation only): R(q) = SET of relation labels of the frozen typed-path rule's schedule
    (results/L1_P90_EXPLOIT/_uni_<cache>.npz 'orders' -- lexical Porter/irregular-lemma match of relation labels in the question with the
    entity-mention tokens masked; order ignored here; empty for 7 % of metaqa dev, for every query of an untyped graph).
Runtime (direct block scoring, per channel, per hit v at rank r, w_r = 1/(K0 + r)):
    base   : own block + directed out-neighbour blocks (the served table, D.legacy_mem)      -> w_r each   (= S_base, untouched)
    compat : {b : (b, rho) in TM(v), rho in R(q)}                                            -> w_r each   (+ compat)
    a block in both gets 2 w_r; S += , M = max; then the frozen rr(S) + rr(M), the frozen RRF over dense + SPLADE, P50.
    R(q) empty -> compat empty -> exactly the served ranking.  Untyped graph (< 2 relation labels) -> compat never built -> served.
Arms: A0 served (assert == TA.partition_ranking), A1 UND_NONHUB full (assert == massvote A1), R1 as above.  DEV_A only.

    python -u _l1c_relsig.py <cache>        -> results/L1_COVPART/relsig_A_<cache>.json
"""
import json
import os
import sys
import time

import numpy as np

import _l1g_candidate as CAND
import _l1g_core as G
import _ta_prepartition as TA

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
P90 = os.path.join(G.X.REPO, "results", "L1_P90_EXPLOIT")
name = sys.argv[1]
arms = (sys.argv[2] if len(sys.argv) > 2 else "A0,A1,R1").split(",")
t0 = time.time()
D = G.Data(name, dense_fp32=True)
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


def csr_pairs(v, key2, width):
    """rows v with secondary keys key2 (< width) -> (ptr, key2 sorted unique per row)."""
    keys = np.unique(v * np.int64(width) + key2)
    rows = keys // width
    cnt = np.bincount(rows, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    return ptr, (keys % width).astype(np.int64)


# admissible edge list (v receives the membership of u): out-edges for all v, in-edges for non-hub v
v_out, u_out = s_e, d_e
keep_in = ~hub[d_e]
v_in, u_in = d_e[keep_in], s_e[keep_in]
v_adm = np.concatenate([v_out, v_in])
u_adm = np.concatenate([u_out, u_in])
IN_ptr, IN_flat = csr_pairs(v_in, hard[u_in], npart)                                     # untyped in-neighbour blocks (A1's extra)
if typed_graph:
    r_adm = np.concatenate([r_e, r_e[keep_in]]).astype(np.int64)
    nrel = len(vocab)
    T_ptr, T_key = csr_pairs(v_adm, hard[u_adm] * np.int64(nrel) + r_adm, npart * nrel)  # typed memberships (block, rel) per node
    T_blk, T_rel = T_key // nrel, T_key % nrel
    z = np.load(os.path.join(P90, "_uni_%s.npz" % name), allow_pickle=True)
    orders = z["orders"]
    assert len(orders) == nq, (len(orders), nq)
    rid = {lbl: i for i, lbl in enumerate(vocab)}
    RQ = [np.array(sorted({rid[x] for x in o.split(",") if x}), np.int64) if o else np.zeros(0, np.int64) for o in orders]
else:
    nrel = 0
    RQ = [np.zeros(0, np.int64) for _ in range(nq)]
    T_ptr = T_blk = T_rel = None
G.log("%s: N %d blocks %d cap %d hubs %d; typed_graph %s (%d labels); admissible typed edges %d; |R(q)|>=1 on DEV_A %.3f (mean |R(q)| %.2f)" % (
    name, N, npart, cap, int(hub.sum()), typed_graph, nrel, len(v_adm), float(np.mean([len(RQ[i]) >= 1 for i in np.nonzero(m)[0]])),
    float(np.mean([len(RQ[i]) for i in np.nonzero(m)[0]]))))
dl = [D.d_ids[i, :K_LOCK] for i in range(nq)]
sl = [D.s_ids[i, :K_LOCK] for i in range(nq)]
ref = TA.partition_ranking(dl, served, npart).astype(np.int64)
D.cd._csr.clear()
rdir = CAND.directional_block_rank(D)
EMPTY = np.zeros(0, np.int64)


def extra_none(qi, nd):
    return EMPTY


def extra_in(qi, nd):
    return IN_flat[IN_ptr[nd]:IN_ptr[nd + 1]]


def extra_compat(qi, nd):
    rq = RQ[qi]
    if len(rq) == 0:
        return EMPTY
    a, b = T_ptr[nd], T_ptr[nd + 1]
    sel = np.isin(T_rel[a:b], rq)
    if not sel.any():
        return EMPTY
    return np.unique(T_blk[a:b][sel])


def ranking(node_lists, extra, mode):
    """_ta_prepartition.partition_ranking with per-hit extra blocks: mode 'union' -> w_r once per block (A1);
    mode 'add' -> w_r per reason, a block in base and extra gets 2 w_r (R1).  Returns (rank, S, n_extra_blocks)."""
    S = np.zeros((nq, npart), np.float32)
    M = np.zeros((nq, npart), np.float32)
    n_extra = np.zeros(nq, np.int64)
    for qi in range(nq):
        for r, nd in enumerate(node_lists[qi]):
            nd = int(nd)
            if nd < 0 or nd >= N:
                continue
            w = 1.0 / (K0 + r)
            bb = sflat[sptr[nd]:sptr[nd + 1]]
            eb = extra(qi, nd)
            n_extra[qi] += len(eb)
            if len(eb) == 0:
                ps, v = bb, np.full(len(bb), w, np.float32)
            elif mode == "union":
                ps = np.union1d(bb, eb)
                v = np.full(len(ps), w, np.float32)
            else:
                ps, cnt = np.unique(np.concatenate([bb, eb]), return_counts=True)
                v = (w * cnt).astype(np.float32)
            S[qi, ps] += v
            np.maximum.at(M[qi], ps, v)

    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), np.int32)
        rank[np.arange(nq)[:, None], order] = np.arange(npart)[None, :]
        return 1.0 / (K0 + rank)
    votes = rr(S) + rr(M)
    return np.argsort(-votes, axis=1).astype(np.int64), S, n_extra


ARMS = {"A0": (extra_none, "add"), "A1": (extra_in, "union"), "R1": (extra_compat, "add")}
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
feas = gs <= G.P_MAIN


def evaluate(tag):
    extra, mode = ARMS[tag]
    Cd, Sd, nxd = ranking(dl, extra, mode)
    Cs, Ss, nxs = ranking(sl, extra, mode)
    if tag == "A0":
        assert np.array_equal(Cd, ref), "A0 ranking != frozen partition_ranking"
        D.mem = served
        ev_ref = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
        assert np.array_equal(ev_ref, (Sd > 0) | (Ss > 0)), "A0 evidence != served evidence"
    ev = (Sd > 0) | (Ss > 0)
    base = G.F0([Cd, Cs], npart)
    pos = G.positions(base)
    in50 = pos < G.P_MAIN
    base_all = (gm & in50).sum(axis=1) == gs
    reach = (gm & ~ev).sum(axis=1) == 0
    pd_, ps_ = G.positions(Cd) < G.P_MAIN, G.positions(Cs) < G.P_MAIN
    union50 = ((gm & ~(pd_ | ps_)).sum(axis=1) == 0) & feas
    fail = ~base_all
    worst = np.where(gm, np.where(ev, pos, npart), -1).max(axis=1)
    out = {"arm": tag,
           "BASE_ALL": round(float(base_all[m].mean()), 4), "reach_all": round(float(reach[m].mean()), 4), "feasible": round(float(feas[m].mean()), 4),
           "fail_pts": round(float(fail[m].mean()) * 100, 1),
           "fail_infeasible_pts": round(float((fail & ~feas)[m].mean()) * 100, 1),
           "fail_UNREACHED_pts": round(float((fail & feas & ~reach)[m].mean()) * 100, 1),
           "fail_reached_weak_pts": round(float((fail & feas & reach & ~union50)[m].mean()) * 100, 1),
           "fail_fusion_fixable_pts": round(float((fail & feas & reach & union50)[m].mean()) * 100, 1),
           "worst_gold_block_rank_median_failed": int(np.median(worst[m & fail])) if (m & fail).any() else None,
           "worst_gold_block_rank_median_reached_failed": int(np.median(worst[m & fail & reach])) if (m & fail & reach).any() else None,
           "worst_gold_block_rank_median_all_reached": int(np.median(worst[m & reach])) if (m & reach).any() else None,
           "blocks_voted_per_query_mean": round(float(ev[m].sum(axis=1).mean()), 1),
           "extra_blocks_per_query_mean_dense_splade": [round(float(nxd[m].mean()), 1), round(float(nxs[m].mean()), 1)],
           "scope_nodes_P50": round(float(np.mean([D.sizes[base[i, :G.P_MAIN]].sum() for i in np.nonzero(m)[0]])), 1),
           "per_hop": {}}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        out["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "BASE_ALL": round(float(base_all[s].mean()), 4), "reach_all": round(float(reach[s].mean()), 4),
                                       "UNREACHED_pts": round(float((fail & feas & ~reach)[s].mean()) * 100, 1),
                                       "reached_weak_pts": round(float((fail & feas & reach & ~union50)[s].mean()) * 100, 1),
                                       "worst_gold_block_rank_median_reached_failed": int(np.median(worst[s & fail & reach])) if (s & fail & reach).any() else None}
    d2d = G.F0([Cd, Cs, rdir], npart)
    o, v = D.eval_rank(d2d, "D2d[%s]" % tag, quiet=True)
    out["D2d_ALL"] = o["A"]["ALL"]
    return out, base_all, reach, v.astype(bool)


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": int(m.sum()), "cap": cap, "hubs": int(hub.sum()), "typed_graph": bool(typed_graph),
       "relation_vocabulary": vocab, "query_relation_source": "results/L1_P90_EXPLOIT/_uni_%s.npz orders (frozen typed-path rule; set, order ignored)" % name,
       "definitions": {"A0": "served: own + directed out-neighbour blocks, w_r each",
                       "A1": "UND_NONHUB: served + in-neighbour blocks of non-hub hits, w_r each (union)",
                       "R1": "served + relation-compatible typed 1-hop memberships {b : (b, rho) in TM(v), rho in R(q)}, w_r each (additive: base+compat = 2 w_r)"},
       "frozen": "rr(S)+rr(M) per channel, RRF K0=%d over dense+splade, P50, exposure = 50 core blocks" % K0, "arms": {}}
kept = {}
for tag in arms:
    o, b, r, v = evaluate(tag)
    kept[tag] = (b, r, v)
    if tag == "A0":
        assert abs(o["BASE_ALL"] - round(float(D.base_all[m].mean()), 4)) < 1e-9
    for refn in [x for x in ("A0", "A1") if x in kept and x != tag]:
        for key, cur, base_ in (("BASE_ALL", b, kept[refn][0]), ("reach_all", r, kept[refn][1]), ("D2d_ALL", v, kept[refn][2])):
            g_, l_, p_ = G.X.mcnemar(base_[m], cur[m])
            o[key + "_vs_" + refn] = {"gained": g_, "lost": l_, "p": p_}
        for h in o["per_hop"]:
            s = m & (hops == int(h[3:]))
            g_, l_, p_ = G.X.mcnemar(kept[refn][0][s], b[s])
            o["per_hop"][h]["BASE_vs_" + refn] = {"gained": g_, "lost": l_, "p": p_}
        if o["per_hop"]:
            s = m & (hops >= 2)
            g_, l_, p_ = G.X.mcnemar(kept[refn][0][s], b[s])
            o["hop2_hop3_pooled_BASE_vs_" + refn] = {"n": int(s.sum()), "gained": g_, "lost": l_, "p": p_}
    res["arms"][tag] = o
    G.log("%-3s %s" % (tag, json.dumps({k: o[k] for k in ("BASE_ALL", "reach_all", "fail_UNREACHED_pts", "fail_reached_weak_pts", "fail_fusion_fixable_pts",
                                                       "worst_gold_block_rank_median_reached_failed", "worst_gold_block_rank_median_all_reached", "blocks_voted_per_query_mean",
                                                       "extra_blocks_per_query_mean_dense_splade", "scope_nodes_P50", "D2d_ALL")})))
    G.log("    per hop %s" % json.dumps({h: (x["BASE_ALL"], x["reach_all"], x["UNREACHED_pts"], x["reached_weak_pts"], x["worst_gold_block_rank_median_reached_failed"]) for h, x in o["per_hop"].items()}))
    for refn in ("A0", "A1"):
        if "BASE_ALL_vs_" + refn in o:
            G.log("    vs %s: BASE %s reach %s D2d %s | %s | hop2+3 pooled %s" % (refn, o["BASE_ALL_vs_" + refn], o["reach_all_vs_" + refn], o["D2d_ALL_vs_" + refn],
                                                                                {h: (x["BASE_ALL"], x["BASE_vs_" + refn]) for h, x in o["per_hop"].items()},
                                                                                o.get("hop2_hop3_pooled_BASE_vs_" + refn)))
D.mem = served
G.S.wj(os.path.join(OUT, "relsig_A_%s.json" % name), res)
G.log("done %.0fs" % (time.time() - t0))
