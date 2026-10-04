"""L1_COVPART step 5: when a retrieved node exposes several blocks through its static 1-hop STRUCT memberships, how should its
FIXED vote budget be distributed among them?  (ruling of 2026-09-14: exactly two diagnostics on top of the memdir arms)

    A0  OUT          served: every membership block receives the full rank weight w_r = 1/(K0 + r)          (memdir OUT)
    A1  UND_NONHUB   undirected memberships (frozen cap hub rule), every membership receives the full w_r   (memdir UND_NONHUB)
    A2  UND_NONHUB   uniform mass-conserving:      vote(v, b) = w_r / |M(v)|
    A3  UND_NONHUB   support-normalised:           vote(v, b) = w_r * a(v, b) / sum_c a(v, c)
                     a(v, b) = 1[P(v) = b] + |{u in N_adm(v) : P(u) = b}|,  N_adm(v) = out-neighbours u (in-neighbours if v is non-hub)
                     (= row v of T = D^-1 (I + A_adm) H, the static node -> block affinity matrix; query time s_q = x_q^T T)

Everything else is the frozen contract: per channel S = sum of votes, M = max of votes, block rank = argsort(rr(S) + rr(M))
(_ta_prepartition.partition_ranking, replicated here with per-membership weights and asserted bit-identical on A0), RRF K0 over
Dense + SPLADE, P50, exposure = the 50 core blocks.  A2/A3 sum to w_r per hit exactly like A0 sums to w_r per membership.
DEV_A only.   python -u _l1c_massvote.py <cache> [A0,A1,A2,A3]  ->  results/L1_COVPART/massvote_A_<cache>.json
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1g_candidate as CAND
import _l1g_core as G
import _ta_prepartition as TA

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
name = sys.argv[1]
arms = (sys.argv[2] if len(sys.argv) > 2 else "A0,A1,A2,A3").split(",")
t0 = time.time()
D = G.Data(name, dense_fp32=True)
C = D.C
nq, N, npart = D.nq, D.N, D.npart
m = D.A
hops = np.asarray(C.hops)
hard = D.hard.astype(np.int64)
cap = int(round(N / npart))
K0, K_LOCK = G.S.K0, G.K_LOCK
xo, ao = D.cd.struct_csr(directed=True)
xo, ao = np.asarray(xo, np.int64), np.asarray(ao, np.int64)
xu, au = D.cd.struct_csr(directed=False)
deg_u = np.diff(np.asarray(xu, np.int64))
D.cd._csr.clear()
hub = (deg_u + 1) > cap
A_in = sp.csr_matrix((np.ones(len(ao), np.int8), ao, xo), shape=(N, N)).T.tocsr()
xi, ai = np.asarray(A_in.indptr, np.int64), np.asarray(A_in.indices, np.int64)


def weighted_table(neigh, rule):
    """neigh: list of (ptr, idx, node_mask) admissible neighbour CSRs.  Returns (ptr, blocks, probs) with probs per membership:
    rule 'full' -> 1 for every membership; 'uniform' -> 1/|M(v)|; 'support' -> a(v,b)/sum_c a(v,c)."""
    rs, ps = [np.arange(N, dtype=np.int64)], [hard]
    for ptr, idx, mask in neigh:
        cnt = np.diff(ptr)
        if mask is not None:
            nb = idx[np.repeat(mask, cnt)]
            cnt = np.where(mask, cnt, 0)
        else:
            nb = idx
        rs.append(np.repeat(np.arange(N, dtype=np.int64), cnt))
        ps.append(hard[nb])
    r, p = np.concatenate(rs), np.concatenate(ps)
    keys, a = np.unique(r * np.int64(npart) + p, return_counts=True)          # a(v,b): own block counts 1 + neighbours in b
    rr = keys // npart
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(mlen)
    blocks = (keys % npart).astype(np.int64)
    if rule == "full":
        probs = np.ones(len(keys), np.float64)
    elif rule == "uniform":
        probs = 1.0 / np.repeat(mlen, mlen).astype(np.float64)
    elif rule == "support":
        tot = np.bincount(rr, weights=a, minlength=N)                           # sum_c a(v,c) = 1 + |N_adm(v)|
        probs = a.astype(np.float64) / np.repeat(tot, mlen)
    else:
        raise ValueError(rule)
    return (ptr, blocks, probs), a


def weighted_partition_ranking(node_lists, tab):
    """_ta_prepartition.partition_ranking with a per-membership weight: vote = w_r * prob.  Same float32 accumulators, same
    promote-add-cast path (fancy-index in-place add), same rr(S) + rr(M) rank rule, same K0."""
    ptr, blocks, probs = tab
    S = np.zeros((nq, npart), np.float32)
    M = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        for r, nd in enumerate(node_lists[qi]):
            nd = int(nd)
            if nd < 0 or nd >= N:
                continue
            w = 1.0 / (K0 + r)
            a, b = ptr[nd], ptr[nd + 1]
            ps = blocks[a:b]
            v = (w * probs[a:b]).astype(np.float32)        # the reference adds a weak python float to float32 rows: float32 arithmetic
            S[qi, ps] += v
            np.maximum.at(M[qi], ps, v)

    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), np.int32)
        rank[np.arange(nq)[:, None], order] = np.arange(npart)[None, :]
        return 1.0 / (K0 + rank)
    votes = rr(S) + rr(M)
    return np.argsort(-votes, axis=1).astype(np.int64), S


ADM = [(xo, ao, None), (xi, ai, ~hub)]
TABLES = {"A0": weighted_table([(xo, ao, None)], "full"), "A1": weighted_table(ADM, "full"), "A2": weighted_table(ADM, "uniform"), "A3": weighted_table(ADM, "support")}
# contract checks: A0 == served legacy table, and the weighted ranking with unit probs == the frozen partition_ranking
served = D.legacy_mem()
assert np.array_equal(TABLES["A0"][0][0], served[0]) and np.array_equal(TABLES["A0"][0][1], np.asarray(served[1], np.int64))
D.cd._csr.clear()
dl = [D.d_ids[i, :K_LOCK] for i in range(nq)]
sl = [D.s_ids[i, :K_LOCK] for i in range(nq)]
ref = TA.partition_ranking(dl, served, npart).astype(np.int64)
chk, _ = weighted_partition_ranking(dl, TABLES["A0"][0])
assert np.array_equal(chk, ref), "weighted ranking with unit weights != frozen partition_ranking"
G.log("%s: N %d blocks %d cap %d hubs %d; weighted ranking reproduces the frozen partition_ranking on A0 (dense channel)" % (name, N, npart, cap, int(hub.sum())))
rdir = CAND.directional_block_rank(D)
hit_nodes = np.zeros(N, bool)
for i in np.nonzero(m)[0]:
    hit_nodes[D.d_ids[i, :K_LOCK]] = True
    hit_nodes[D.s_ids[i, :K_LOCK]] = True


def table_stats(tab, a):
    ptr, blocks, probs = tab
    ml = np.diff(ptr)
    own = np.zeros(N, np.float64)                     # p(P(v) | v): the share of a node's budget that stays in its own block
    rows = np.repeat(np.arange(N), ml)
    is_own = blocks == hard[rows]
    own[rows[is_own]] = probs[is_own]
    # concentration: max_b p(b|v)
    pmax = np.zeros(N, np.float64)
    np.maximum.at(pmax, rows, probs)
    q = lambda v, f=3: {"mean": round(float(v.mean()), f), "p50": round(float(np.median(v)), f), "p95": round(float(np.percentile(v, 95)), f), "max": round(float(v.max()), f)}
    return {"memberships": {"all": q(ml.astype(np.float64), 2), "served_hits": q(ml[hit_nodes].astype(np.float64), 2)},
            "own_block_share": {"all": q(own), "served_hits": q(own[hit_nodes]), "nodes_with_gt1_memberships": q(own[ml > 1])},
            "max_block_share": {"all": q(pmax), "nodes_with_gt1_memberships": q(pmax[ml > 1])},
            "total_budget_per_hit_mean": round(float(np.bincount(rows, weights=probs, minlength=N)[hit_nodes].mean()), 4)}


def evaluate(tag):
    tab, a = TABLES[tag]
    D.mem = (tab[0], tab[1])                                              # evidence (reach) only needs the membership sets
    ev = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
    Cd, Sd = weighted_partition_ranking(dl, tab)
    Cs, Ss = weighted_partition_ranking(sl, tab)
    base = G.F0([Cd, Cs], npart)
    pos = G.positions(base)
    gm = G.gold_mask(D)
    gs = gm.sum(axis=1)
    feas = gs <= G.P_MAIN
    in50 = pos < G.P_MAIN
    base_all = (gm & in50).sum(axis=1) == gs
    reach = (gm & ~ev).sum(axis=1) == 0
    pd_, ps_ = G.positions(Cd) < G.P_MAIN, G.positions(Cs) < G.P_MAIN
    union50 = ((gm & ~(pd_ | ps_)).sum(axis=1) == 0) & feas
    fail = ~base_all
    worst = np.where(gm, np.where(ev, pos, npart), -1).max(axis=1)
    out = {"arm": tag, "table": table_stats(tab, a),
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


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": int(m.sum()), "cap": cap, "hubs": int(hub.sum()),
       "definitions": {"A0": "OUT served, full w_r per membership", "A1": "UND_NONHUB, full w_r per membership",
                       "A2": "UND_NONHUB, w_r / |M(v)|", "A3": "UND_NONHUB, w_r * a(v,b) / sum_c a(v,c); a = 1[P(v)=b] + #admissible neighbours in b"},
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
    res["arms"][tag] = o
    G.log("%-3s %s" % (tag, json.dumps({k: o[k] for k in ("BASE_ALL", "reach_all", "fail_UNREACHED_pts", "fail_reached_weak_pts", "fail_fusion_fixable_pts",
                                                       "worst_gold_block_rank_median_reached_failed", "worst_gold_block_rank_median_all_reached", "blocks_voted_per_query_mean", "scope_nodes_P50", "D2d_ALL")})))
    G.log("    per hop %s" % json.dumps({h: (x["BASE_ALL"], x["reach_all"], x["UNREACHED_pts"], x["reached_weak_pts"], x["worst_gold_block_rank_median_reached_failed"]) for h, x in o["per_hop"].items()}))
    G.log("    table: own-block share (nodes with >1 memberships) %s | max share %s" % (o["table"]["own_block_share"]["nodes_with_gt1_memberships"], o["table"]["max_block_share"]["nodes_with_gt1_memberships"]))
    for refn in ("A0", "A1"):
        if "BASE_ALL_vs_" + refn in o:
            G.log("    vs %s: BASE %s reach %s D2d %s | %s" % (refn, o["BASE_ALL_vs_" + refn], o["reach_all_vs_" + refn], o["D2d_ALL_vs_" + refn],
                                                            {h: (x["BASE_ALL"], x["BASE_vs_" + refn]) for h, x in o["per_hop"].items()}))
D.mem = served
G.S.wj(os.path.join(OUT, "massvote_A_%s.json" % name), res)
G.log("done %.0fs" % (time.time() - t0))
