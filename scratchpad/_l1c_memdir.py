"""L1_COVPART step 4: membership DIRECTION -- the one thing in "owner + static 1-hop STRUCT memberships that vote before P50"
which the served chain does not already do.

Served contract (src/l1_canonical/replay_cache.py:154-165, scratchpad/_ta_prepartition.partition_ranking, replayed here by
_l1s_core.Data.legacy_mem):  mem(v) = {P(v)} u {P(u) : v -> u is a DIRECTED STRUCT out-edge}; every served hit at rank r votes
1/(K0 + r) for all of mem(v) per channel BEFORE the block RRF and the P50 cut.  Exposure is the 50 core blocks (no node halo;
SAFE only swaps <= 6 whole blocks).  So "memberships vote pre-P50" is the served behaviour -- in the OUT direction.  The reach
diagnostic (reach_diag_A_metaqa.json) says 99.8 % of the unreached gold nodes are unreachable by out-edges and that every
hop-1 unreached gold is an IN-neighbour of a hit at distance 1: this module tests the direction of the SAME table.

Tables (same partition, same graph, same 1-hop boundary, same P50 budget, exposure unchanged):
    OUT         served table
    UND         {P(v)} u {P(u) : u ~ v} (both directions), no hub rule
    UND_NONHUB  OUT u {P(u) : u -> v} for non-hub v only (frozen cap rule deg_undirected(v) + 1 <= cap, the H4/PATH2 hub
                definition); a hub keeps the served table, so a hit on a hub cannot vote for hundreds of blocks
    IN          {P(v)} u {P(u) : u -> v} (attribution only)

    python -u _l1c_memdir.py <cache> [OUT,UND,UND_NONHUB,IN]      -> results/L1_COVPART/memdir_A_<cache>.json   (DEV_A only)
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1g_candidate as CAND
import _l1g_core as G

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
name = sys.argv[1]
tables = (sys.argv[2] if len(sys.argv) > 2 else "OUT,UND,UND_NONHUB,IN").split(",")
t0 = time.time()
D = G.Data(name, dense_fp32=True)
C = D.C
nq, N, npart = D.nq, D.N, D.npart
m = D.A
hops = np.asarray(C.hops)
hard = D.hard.astype(np.int64)
cap = int(round(N / npart))
xo, ao = D.cd.struct_csr(directed=True)
xo, ao = np.asarray(xo, np.int64), np.asarray(ao, np.int64)
xu, au = D.cd.struct_csr(directed=False)
xu, au = np.asarray(xu, np.int64), np.asarray(au, np.int64)
D.cd._csr.clear()
deg_out = np.diff(xo)
deg_u = np.diff(xu)
hub = (deg_u + 1) > cap
A_out = sp.csr_matrix((np.ones(len(ao), np.int8), ao, xo), shape=(N, N))
A_in = A_out.T.tocsr()
xi, ai = np.asarray(A_in.indptr, np.int64), np.asarray(A_in.indices, np.int64)
deg_in = np.diff(xi)
G.log("%s: N %d blocks %d cap %d hubs %d (deg_u+1 > cap); out-edges %d in-edges %d undirected %d" % (name, N, npart, cap, int(hub.sum()), len(ao), len(ai), len(au) // 2))


def table(pairs_ptr_idx_list):
    """(ptr, idx) CSR lists of neighbours whose OWNER blocks become memberships, plus the own block -> (mem_ptr, mem_flat)."""
    rs, ps = [np.arange(N, dtype=np.int64)], [hard]
    for ptr, idx, mask in pairs_ptr_idx_list:
        cnt = np.diff(ptr)
        if mask is not None:
            cnt = np.where(mask, cnt, 0)
            keep = np.repeat(mask, np.diff(ptr))
            nb = idx[keep]
        else:
            nb = idx
        rs.append(np.repeat(np.arange(N, dtype=np.int64), cnt))
        ps.append(hard[nb])
    r, p = np.concatenate(rs), np.concatenate(ps)
    keys = np.unique(r * np.int64(npart) + p)
    rr = keys // npart
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(mlen)
    return (ptr, (keys % npart).astype(np.int64))


MEM = {
    "OUT": table([(xo, ao, None)]),
    "UND": table([(xu, au, None)]),
    "UND_NONHUB": table([(xo, ao, None), (xi, ai, ~hub)]),
    "IN": table([(xi, ai, None)]),
}
served = D.legacy_mem()
assert np.array_equal(MEM["OUT"][0], served[0]) and np.array_equal(MEM["OUT"][1], np.asarray(served[1], np.int64)), "OUT table != served legacy_mem"
D.cd._csr.clear()
hit_nodes = np.zeros(N, bool)
for i in np.nonzero(m)[0]:
    hit_nodes[D.d_ids[i, :G.K_LOCK]] = True
    hit_nodes[D.s_ids[i, :G.K_LOCK]] = True
rdir = CAND.directional_block_rank(D)                       # the confirmed D2d channel: block max over nodes, independent of the table


def memstats(mem):
    ml = np.diff(mem[0])
    q = lambda v: {"mean": round(float(v.mean()), 3), "p50": int(np.median(v)), "p95": int(np.percentile(v, 95)), "p99": int(np.percentile(v, 99)), "max": int(v.max()),
                   "frac_gt1": round(float((v > 1).mean()), 4)}
    return {"all_nodes": q(ml), "served_hit_nodes_DEV_A": q(ml[hit_nodes]), "hub_nodes": q(ml[hub]) if hub.any() else None,
            "replication_factor_sum_M_over_N": round(float(ml.sum()) / N, 3)}


def evaluate(tag, mem):
    D.mem = mem
    ev = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
    Cd, Cs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
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
    out = {"table": tag, "memberships": memstats(mem),
           "BASE_ALL": round(float(base_all[m].mean()), 4), "reach_all": round(float(reach[m].mean()), 4), "feasible": round(float(feas[m].mean()), 4),
           "fail_pts": round(float(fail[m].mean()) * 100, 1),
           "fail_infeasible_pts": round(float((fail & ~feas)[m].mean()) * 100, 1),
           "fail_UNREACHED_pts": round(float((fail & feas & ~reach)[m].mean()) * 100, 1),
           "fail_reached_weak_pts": round(float((fail & feas & reach & ~union50)[m].mean()) * 100, 1),
           "fail_fusion_fixable_pts": round(float((fail & feas & reach & union50)[m].mean()) * 100, 1),
           "worst_gold_block_rank_median_failed": int(np.median(worst[m & fail])) if (m & fail).any() else None,
           "worst_gold_block_rank_median_reached_failed": int(np.median(worst[m & fail & reach])) if (m & fail & reach).any() else None,
           "blocks_voted_per_query_mean": round(float(ev[m].sum(axis=1).mean()), 1),
           "scope_nodes_P50": round(float(np.mean([D.sizes[base[i, :G.P_MAIN]].sum() for i in np.nonzero(m)[0]])), 1),
           "per_hop": {}}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        out["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "BASE_ALL": round(float(base_all[s].mean()), 4), "reach_all": round(float(reach[s].mean()), 4),
                                       "UNREACHED_pts": round(float((fail & feas & ~reach)[s].mean()) * 100, 1)}
    d2d = G.F0([Cd, Cs, rdir], npart)
    o, v = D.eval_rank(d2d, "D2d[%s]" % tag, quiet=True)
    out["D2d_ALL"] = o["A"]["ALL"]
    return out, base_all, reach, v.astype(bool)


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": int(m.sum()), "cap": cap, "hubs": int(hub.sum()),
       "served_definition": "mem(v) = {P(v)} u {P(u): v->u out-edge}; hits vote 1/(K0+r) for all memberships per channel before RRF/P50 (replay_cache.py:154-165, _ta_prepartition.partition_ranking)",
       "exposure": "50 core blocks in every arm (tables change votes only)", "arms": {}}
ref = None
for tag in tables:
    o, b, r, v = evaluate(tag, MEM[tag])
    if tag == "OUT":
        assert abs(o["BASE_ALL"] - round(float(D.base_all[m].mean()), 4)) < 1e-9, (o["BASE_ALL"], float(D.base_all[m].mean()))
        ref = (b, r, v, o)
    else:
        for key, cur, base_ in (("BASE_ALL", b, ref[0]), ("reach_all", r, ref[1]), ("D2d_ALL", v, ref[2])):
            g_, l_, p_ = G.X.mcnemar(base_[m], cur[m])
            o[key + "_vs_OUT"] = {"gained": g_, "lost": l_, "p": p_}
        for h in o["per_hop"]:
            hh = int(h[3:])
            s = m & (hops == hh)
            g_, l_, p_ = G.X.mcnemar(ref[0][s], b[s])
            o["per_hop"][h]["BASE_vs_OUT"] = {"gained": g_, "lost": l_, "p": p_}
    res["arms"][tag] = o
    G.log("%-10s %s" % (tag, json.dumps({k: o[k] for k in ("BASE_ALL", "reach_all", "fail_UNREACHED_pts", "fail_reached_weak_pts", "fail_fusion_fixable_pts",
                                                          "worst_gold_block_rank_median_failed", "blocks_voted_per_query_mean", "scope_nodes_P50", "D2d_ALL")})))
    G.log("           memberships %s" % json.dumps(o["memberships"]["all_nodes"]))
    if tag != "OUT":
        G.log("           vs OUT: BASE %s reach %s D2d %s | %s" % (o["BASE_ALL_vs_OUT"], o["reach_all_vs_OUT"], o["D2d_ALL_vs_OUT"],
                                                                {h: (o["per_hop"][h]["BASE_ALL"], o["per_hop"][h]["BASE_vs_OUT"]) for h in o["per_hop"]}))
D.mem = served
G.S.wj(os.path.join(OUT, "memdir_A_%s.json" % name), res)
G.log("done %.0fs" % (time.time() - t0))
