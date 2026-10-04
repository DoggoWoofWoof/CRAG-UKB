"""L1_COVPART lane, step 0 (DEV_A, diagnostic only): WHERE are the UNREACHED metaqa gold nodes relative to the served hits?

BASE reach mechanics (frozen): a served hit h (dense top-100 or SPLADE top-100) votes for block(h) and for block(v) of every
DIRECTED structural out-neighbour v of h (the legacy table).  A gold node g is REACHED iff some hit votes for block(g).
So a partition can convert an unreached gold node only by co-locating it with a hit or with a hit's out-neighbour, and
co-location is only meaningful along structure that does not pass through hubs (nodes whose closed star exceeds the block
capacity cap = 100 = the H4 cap; a hub cannot be kept with its neighbourhood by any balanced partition).

Per DEV_A query and gold node: reached?, undirected STRUCT distance from the hit set, the same with hubs removed as
intermediates ("non-hub distance"), directed-out distance, hub flag, BASE rank of its block.  Aggregated by hop and by
distance -> the radius at which a co-location (partition) fix is even possible, and the mass no partition can touch.
Records: results/L1_COVPART/reach_diag_A_<cache>.json.  Nothing under data/ is touched; no query-dependent algorithm is
proposed here -- this is the failure diagnosis that decides the ONE construction the lane tests.
"""
import json
import os
import sys
import time

import numpy as np

import _l1g_core as G

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
os.makedirs(OUT, exist_ok=True)
CAP = 100                      # H4 cap = target block size: a node with closed star > CAP is a hub
MAXD = 4
name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
t0 = time.time()
D = G.Data(name, dense_fp32=False)
C = D.C
nq, N, npart = D.nq, D.N, D.npart
m = D.A
hops = np.asarray(C.hops)
G.log("%s [%s] N=%d blocks=%d DEV_A n=%d BASE %.4f" % (name, G.PARTITION_OF[name], N, npart, int(m.sum()), D.r_base["ALL_split"]["A"]["ALL"]))

# --- structure
xu, au = D.cd.struct_csr(directed=False)
au = np.asarray(au, np.int64); xu = np.asarray(xu, np.int64)
xo, ao = D.cd.struct_csr(directed=True)
ao = np.asarray(ao, np.int64); xo = np.asarray(xo, np.int64)
D.cd._csr.clear()
deg_u = np.diff(xu)
hub = (deg_u + 1) > CAP
G.log("undirected STRUCT edges (pins) %d, directed out-edges %d, hubs (closed star > %d) %d, max degree %d" % (len(au), len(ao), CAP, int(hub.sum()), int(deg_u.max())))


def bfs(sources, xadj, adj, expand_mask=None, maxd=MAXD):
    """multi-source BFS distances (−1 = beyond maxd); nodes with expand_mask False receive a distance but are not expanded."""
    dist = np.full(N, -1, np.int64)
    dist[sources] = 0
    front = np.unique(sources)
    for d in range(1, maxd + 1):
        if expand_mask is not None:
            front = front[expand_mask[front]]
        if len(front) == 0:
            break
        cnt = xadj[front + 1] - xadj[front]
        idx = np.repeat(xadj[front], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        nb = adj[idx]
        nb = nb[dist[nb] < 0]
        if len(nb) == 0:
            break
        nb = np.unique(nb)
        dist[nb] = d
        front = nb
    return dist


# --- evidence / BASE ranking on this partition
ev = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)                       # (nq, npart) block voted by any served hit
Cd, Cs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
base = G.F0([Cd, Cs], npart)
pos = G.positions(base)
gm = G.gold_mask(D)
base_all = ((gm & (pos < G.P_MAIN)).sum(axis=1) == gm.sum(axis=1))
assert (base_all == D.base_all).all(), "BASE not reproduced"
gsizes = gm.sum(axis=1)
feasible = gsizes <= G.P_MAIN
reached_all = ((gm & ~ev).sum(axis=1) == 0)

# --- non-hub 2-hop ball size of every node (what a partition would have to keep in one block)
deg_out = np.diff(xo)
nonhub = ~hub


def ball2_size(u):
    n1 = au[xu[u]:xu[u + 1]]
    n1e = n1[nonhub[n1]]
    if len(n1e):
        cnt = xu[n1e + 1] - xu[n1e]
        idx = np.repeat(xu[n1e], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        n2 = au[idx]
        return int(len(np.unique(np.concatenate([[u], n1, n2]))))
    return int(len(n1)) + 1


B2 = np.array([ball2_size(u) if nonhub[u] else -1 for u in range(N)], np.int64)
res_ball = {"nonhub_nodes": int(nonhub.sum()), "ball2_quantiles_nonhub": {q: int(np.percentile(B2[nonhub], q)) for q in (10, 25, 50, 75, 90, 95, 99)},
            "ball2_le_cap_frac": round(float((B2[nonhub] <= CAP).mean()), 4)}
G.log("non-hub 2-hop ball sizes: %s" % res_ball)

# --- per gold node
rows = []          # (q, hop, g, reached, hub, d_undir, d_nonhub, d_out, block_pos, block_voted)
near = []          # per unreached gold: (hop, d_nonhub, min ball2 of the nearest hits, any nearest hit with out-degree > 0)
per_q = []
for i in np.nonzero(m)[0]:
    hits = np.unique(np.concatenate([D.d_ids[i, :G.K_LOCK], D.s_ids[i, :G.K_LOCK]]))
    gold = np.asarray(C.gold_nodes[i], np.int64)
    du = bfs(hits, xu, au)
    dn = bfs(hits, xu, au, expand_mask=~hub)
    do = bfs(hits, xo, ao)
    hset = set(hits.tolist())
    inhit = np.zeros(N, bool); inhit[hits] = True
    qnear = []
    for g in gold:
        b = int(D.hard[g])
        rows.append((int(i), int(hops[i]), int(g), bool(ev[i, b]), bool(hub[g]), int(du[g]), int(dn[g]), int(do[g]), int(pos[i, b]), bool(ev[i, b])))
        if not ev[i, b] and dn[g] >= 1:
            dg = bfs(np.array([g]), xu, au, expand_mask=nonhub, maxd=int(dn[g]))
            nh = np.nonzero(inhit & (dg == dn[g]))[0]
            if len(nh):
                bb = B2[nh]; bb = bb[bb >= 0]
                mb = int(bb.min()) if len(bb) else -1
                near.append((int(hops[i]), int(dn[g]), mb, bool((deg_out[nh] > 0).any())))
                qnear.append(mb)
    unre = [r for r in rows[-len(gold):] if not r[3]]
    per_q.append({"q": int(i), "hop": int(hops[i]), "ngold": int(len(gold)), "ngold_blocks": int(gsizes[i]), "feasible": bool(feasible[i]),
                  "BASE_all": bool(base_all[i]), "reached_all": bool(reached_all[i]), "n_unreached_nodes": len(unre),
                  "max_nonhub_dist_unreached": (max(r[6] if r[6] >= 0 else 99 for r in unre) if unre else -1),
                  "max_undir_dist_unreached": (max(r[5] if r[5] >= 0 else 99 for r in unre) if unre else -1),
                  "unreached_hub_only": bool(unre) and all(r[4] for r in unre),
                  "max_ball2_nearest_hit_unreached": (max(qnear) if qnear else -1),
                  "worst_gold_block_pos": int(max(pos[i, b] if ev[i, b] else 10 ** 6 for b in np.nonzero(gm[i])[0]))})
G.log("per-node rows %d (%.0fs)" % (len(rows), time.time() - t0))
R = np.array([(r[1], int(r[3]), int(r[4]), r[5], r[6], r[7], r[8]) for r in rows], np.int64)   # hop, reached, hub, du, dn, do, pos


def hist(v, cats):
    v = np.where(v < 0, 99, v)
    tot = max(len(v), 1)
    out = {}
    for lab, lo, hi in cats:
        out[lab] = round(float(((v >= lo) & (v <= hi)).sum()) / tot, 4)
    return out


CATS = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), (">4", 5, 99)]
res = {"cache": name, "partition": G.PARTITION_OF[name], "N": N, "blocks": npart, "cap_hub": CAP, "hubs": int(hub.sum()), "n_DEV_A": int(m.sum()),
       "BASE_ALL_A": round(float(base_all[m].mean()), 4), "reach_all_A": round(float(reached_all[m].mean()), 4), "feasible_A": round(float(feasible[m].mean()), 4),
       "by_hop": {}, "unreached_gold_nodes": {}, "queries_with_unreached": {}}
un = R[R[:, 1] == 0]
res["unreached_gold_nodes"]["n"] = int(len(un))
res["unreached_gold_nodes"]["frac_hub"] = round(float(un[:, 2].mean()), 4) if len(un) else None
res["unreached_gold_nodes"]["dist_undirected"] = hist(un[:, 3], CATS)
res["unreached_gold_nodes"]["dist_nonhub"] = hist(un[:, 4], CATS)
res["unreached_gold_nodes"]["dist_directed_out"] = hist(un[:, 5], CATS)
res["unreached_gold_nodes"]["nonhub_dist_by_hub_flag"] = {"hub": hist(un[un[:, 2] == 1][:, 4], CATS), "nonhub": hist(un[un[:, 2] == 0][:, 4], CATS)}
re_ = R[(R[:, 1] == 1)]
res["reached_gold_nodes"] = {"n": int(len(re_)), "dist_nonhub": hist(re_[:, 4], CATS), "frac_hub": round(float(re_[:, 2].mean()), 4),
                             "block_pos_median": int(np.median(re_[:, 6])), "block_pos_outside_P50": round(float((re_[:, 6] >= G.P_MAIN).mean()), 4)}
# queries
Q = per_q
qa = np.array([q["reached_all"] for q in Q]); qb = np.array([q["BASE_all"] for q in Q]); qf = np.array([q["feasible"] for q in Q])
mx = np.array([q["max_nonhub_dist_unreached"] for q in Q]); hubonly = np.array([q["unreached_hub_only"] for q in Q]); qh = np.array([q["hop"] for q in Q])
fail_unre = ~qa & qf
res["queries_with_unreached"] = {
    "n_fail": int((~qb).sum()), "n_unreached_feasible": int(fail_unre.sum()),
    "unreached_mass_pts": round(float(fail_unre.mean()) * 100, 1),
    "by_max_nonhub_distance_of_unreached_golds": {lab: round(float((fail_unre & (np.where(mx < 0, 99, mx) >= lo) & (np.where(mx < 0, 99, mx) <= hi)).mean()) * 100, 1) for lab, lo, hi in CATS},
    "unreached_hub_only_pts": round(float((fail_unre & hubonly).mean()) * 100, 1),
    "cumulative_convertible_pts_if_colocation_perfect_within_radius": {str(r): round(float((fail_unre & (mx >= 0) & (mx <= r)).mean()) * 100, 1) for r in (1, 2, 3, 4)},
    "worst_gold_block_pos_median_reached_but_failed": int(np.median([q["worst_gold_block_pos"] for q in Q if q["reached_all"] and not q["BASE_all"]])) if any(q["reached_all"] and not q["BASE_all"] for q in Q) else None}
for h in sorted(set(qh.tolist())):
    s = qh == h
    unh = un[un[:, 0] == h]
    res["by_hop"]["hop%d" % h] = {"n": int(s.sum()), "BASE_ALL": round(float(qb[s].mean()), 4), "reach_all": round(float(qa[s].mean()), 4),
                                  "unreached_mass_pts": round(float((fail_unre & s).mean() / max(s.mean(), 1e-9)) * 100, 1),
                                  "unreached_nodes": int(len(unh)), "unreached_nodes_frac_hub": round(float(unh[:, 2].mean()), 4) if len(unh) else None,
                                  "unreached_nodes_dist_nonhub": hist(unh[:, 4], CATS), "unreached_nodes_dist_out": hist(unh[:, 5], CATS),
                                  "gold_nodes_per_q": round(float(np.mean([q["ngold"] for q in Q if q["hop"] == h])), 2),
                                  "gold_blocks_per_q": round(float(np.mean([q["ngold_blocks"] for q in Q if q["hop"] == h])), 2)}
NB = np.array(near, np.int64) if near else np.zeros((0, 4), np.int64)
BC = [("<=50", 0, 50), ("51-100", 51, 100), ("101-200", 101, 200), ("201-500", 201, 500), (">500", 501, 10 ** 9)]
res["ball2"] = res_ball
res["unreached_gold_nodes"]["nearest_hit_ball2"] = {lab: round(float(((NB[:, 2] >= lo) & (NB[:, 2] <= hi)).mean()), 4) for lab, lo, hi in BC} if len(NB) else None
res["unreached_gold_nodes"]["nearest_hit_ball2_by_dist"] = {str(d): {lab: round(float(((NB[NB[:, 1] == d][:, 2] >= lo) & (NB[NB[:, 1] == d][:, 2] <= hi)).mean()), 4) for lab, lo, hi in BC}
                                                            for d in (1, 2, 3) if (NB[:, 1] == d).any()} if len(NB) else None
res["unreached_gold_nodes"]["nearest_hit_has_out_edges_frac"] = round(float(NB[:, 3].mean()), 4) if len(NB) else None
mb = np.array([q["max_ball2_nearest_hit_unreached"] for q in Q])
res["queries_with_unreached"]["by_max_ball2_of_nearest_hits"] = {lab: round(float((fail_unre & (mb >= lo) & (mb <= hi)).mean()) * 100, 1) for lab, lo, hi in BC}
res["queries_with_unreached"]["capacity_convertible_pts_radius2_ball_le_cap"] = round(float((fail_unre & (mx >= 0) & (mx <= 2) & (mb >= 0) & (mb <= CAP)).mean()) * 100, 1)
res["hub_nodes"] = {"n": int(hub.sum()), "degree_min": int(deg_u[hub].min()) if hub.any() else None, "degree_max": int(deg_u.max()),
                    "pins_in_hubs_frac": round(float(deg_u[hub].sum()) / float(deg_u.sum()), 4)}
G.S.wj(os.path.join(OUT, "reach_diag_A_%s.json" % name), res)
print(json.dumps({k: v for k, v in res.items() if k not in ("by_hop",)}, indent=1))
for h, v in res["by_hop"].items():
    print(h, json.dumps(v))
G.log("done %.0fs" % (time.time() - t0))
