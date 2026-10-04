"""Attribution for the edge-geometry channel (T3_C1000 vs T0) on DEV_A -- rebuilds the channels exactly as _l1c_edgegeo does
(same source slice executed), adds no arm and changes no ranking.  Questions answered:
  1. which queries T3 gains / loses per hop, and whether the T3 gains are the SAME queries the closed A1 (UND_NONHUB full-vote
     membership table) gained -- i.e. is the edge channel a scored version of "vote for the anchor's in-neighbour blocks"?
  2. for every T3-gained query: is a newly exposed gold block the endpoint block of a top-100 edge INCIDENT TO THE RANK-0 node hit
     (dense or SPLADE top-1)?  at what edge rank did the block first receive an edge vote?
  3. for reached-failed queries under T3 (hop 2/3): the best edge rank (mid or dir channel) that voted for the worst gold block --
     rank0 / 1-4 / 5-19 / 20-99 / none (does the discriminator see the block and fail to rank it, or never see it?)
  4. composition of the top-100 edge lists: share incident to the rank-0 hit, share incident to any node top-10 hit, share with
     both endpoints outside the node top-100.
    python -u _l1c_edgegeo_why.py <cache>     -> results/L1_COVPART/edgegeo_why_A_<cache>.json
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

import _l1g_core as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
src = open(os.path.join(HERE, "_l1c_edgegeo.py"), encoding="utf-8").read()
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- node channels")]
head = head.replace("name = sys.argv[1]", "name = %r" % name)
exec(head)                                                   # D, edges, scores, tops, CH_E ... identical construction
Cd, Cs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
ev_node = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
feas = gs <= P_MAIN
(Cm, Sm), (Cr, Sr) = CH_E["C1000"]["mid"], CH_E["C1000"]["dir"]
pos0 = G.positions(G.F0([Cd, Cs], npart))
pos3 = G.positions(G.F0([Cd, Cs, Cm, Cr], npart))
ok0 = (gm & (pos0 < P_MAIN)).sum(axis=1) == gs
ok3 = (gm & (pos3 < P_MAIN)).sum(axis=1) == gs
assert abs(float(ok0[m].mean()) - float(D.base_all[m].mean())) < 1e-9
ev3 = ev_node | (Sm > 0) | (Sr > 0)
reach3 = (gm & ~ev3).sum(axis=1) == 0

# ---------------------------------------------------------------- A1 (UND_NONHUB full vote) per-query correctness, as _l1c_massvote builds it
cap = int(round(N / npart))
xo, ao = D.cd.struct_csr(directed=True)
xo, ao = np.asarray(xo, np.int64), np.asarray(ao, np.int64)
xu, au = D.cd.struct_csr(directed=False)
deg_u = np.diff(np.asarray(xu, np.int64))
D.cd._csr.clear()
hub = (deg_u + 1) > cap
A_in = sp.csr_matrix((np.ones(len(ao), np.int8), ao, xo), shape=(N, N)).T.tocsr()
xi, ai = np.asarray(A_in.indptr, np.int64), np.asarray(A_in.indices, np.int64)
msrc = open(os.path.join(HERE, "_l1c_massvote.py"), encoding="utf-8").read()
exec(msrc.split("def weighted_table")[1].split("ADM = ")[0].replace("(neigh, rule):", "def weighted_table(neigh, rule):", 1))
T1tab, _ = weighted_table([(xo, ao, None), (xi, ai, ~hub)], "full")
dl = [D.d_ids[i, :K_LOCK] for i in range(nq)]
sl = [D.s_ids[i, :K_LOCK] for i in range(nq)]
C1d, _ = weighted_partition_ranking(dl, T1tab)
C1s, _ = weighted_partition_ranking(sl, T1tab)
pos1 = G.positions(G.F0([C1d, C1s], npart))
ok1 = (gm & (pos1 < P_MAIN)).sum(axis=1) == gs

# ---------------------------------------------------------------- per (query, block): best edge rank with a vote (mid / dir), and via an anchor-incident edge
BIG = 10 ** 6
best_edge = np.full((nq, npart), BIG, np.int64)
best_anchor_edge = np.full((nq, npart), BIG, np.int64)
comp = {"incident_to_rank0_hit": {"mid": 0.0, "dir": 0.0}, "incident_to_a_top10_hit": {"mid": 0.0, "dir": 0.0}, "both_endpoints_outside_node_top100": {"mid": 0.0, "dir": 0.0}}
for j, qi in enumerate(rowsA):
    anchors = {int(D.d_ids[qi, 0]), int(D.s_ids[qi, 0])}
    t10 = set(int(x) for x in np.concatenate([D.d_ids[qi, :10], D.s_ids[qi, :10]]))
    h100 = top100[j]
    for k, lst in (("mid", tops["C1000"]["mid"][j]), ("dir", tops["C1000"]["dir"][j])):
        n_anc = n_t10 = n_out = 0
        for r, e in enumerate(lst):
            s_, o_ = int(s_e[e]), int(o_e[e])
            ps = (int(bs_e[e]), int(bo_e[e]))
            best_edge[qi, list(ps)] = np.minimum(best_edge[qi, list(ps)], r)
            inc = (s_ in anchors) or (o_ in anchors)
            if inc:
                best_anchor_edge[qi, list(ps)] = np.minimum(best_anchor_edge[qi, list(ps)], r)
            n_anc += inc
            n_t10 += (s_ in t10) or (o_ in t10)
            n_out += (s_ not in h100) and (o_ not in h100)
        if len(lst):
            comp["incident_to_rank0_hit"][k] += n_anc / len(lst)
            comp["incident_to_a_top10_hit"][k] += n_t10 / len(lst)
            comp["both_endpoints_outside_node_top100"][k] += n_out / len(lst)
for key in comp:
    for k in comp[key]:
        comp[key][k] = round(comp[key][k] / nA, 4)


def bucket(x):
    return "rank0" if x == 0 else ("rank1-4" if x < 5 else ("rank5-19" if x < 20 else ("rank20-99" if x < BIG else "none")))


res = {"cache": name, "n_DEV_A": nA, "T0_BASE_ALL": round(float(ok0[m].mean()), 4), "A1_BASE_ALL": round(float(ok1[m].mean()), 4),
       "T3_C1000_BASE_ALL": round(float(ok3[m].mean()), 4), "top100_edge_composition_C1000": comp, "per_hop": {}}
for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
    s = m & (hops == h)
    gained = np.nonzero(s & ~ok0 & ok3)[0]
    lost = np.nonzero(s & ok0 & ~ok3)[0]
    a1g = s & ~ok0 & ok1
    out = {"n": int(s.sum()), "T0_ok": int((s & ok0).sum()), "A1_ok": int((s & ok1).sum()), "T3_ok": int((s & ok3).sum()),
           "T3_gained": int(len(gained)), "T3_lost": int(len(lost)),
           "A1_gained": int(a1g.sum()), "T3_gained_also_A1_gained": int((a1g & ~ok0 & ok3 & s).sum()),
           "A1_gained_not_T3_gained": int((a1g & ~ok3).sum()), "T3_gained_not_A1_gained": int(sum(not a1g[i] for i in gained)),
           "gained__new_gold_block_first_edge_vote_rank": {}, "gained__new_gold_block_via_edge_incident_to_rank0_hit": 0,
           "gained__new_gold_block_was_UNREACHED_under_T0": 0,
           "lost__pushed_out_gold_block_position_under_T3": [],
           "T3_reached_failed": 0, "T3_reached_failed__worst_gold_block_best_edge_vote_rank": {}, "T3_UNREACHED": 0}
    for i in gained:
        newb = np.nonzero(gm[i] & (pos0[i] >= P_MAIN) & (pos3[i] < P_MAIN))[0]        # gold blocks that entered P50
        r_best = int(best_edge[i, newb].min()) if len(newb) else BIG
        k = bucket(r_best)
        out["gained__new_gold_block_first_edge_vote_rank"][k] = out["gained__new_gold_block_first_edge_vote_rank"].get(k, 0) + 1
        out["gained__new_gold_block_via_edge_incident_to_rank0_hit"] += int(len(newb) > 0 and best_anchor_edge[i, newb].min() < BIG)
        out["gained__new_gold_block_was_UNREACHED_under_T0"] += int(len(newb) > 0 and (~ev_node[i, newb]).any())
    for i in lost:
        outb = np.nonzero(gm[i] & (pos0[i] < P_MAIN) & (pos3[i] >= P_MAIN))[0]
        out["lost__pushed_out_gold_block_position_under_T3"].append(int(pos3[i, outb].max()) if len(outb) else None)
    rf = np.nonzero(s & feas & reach3 & ~ok3)[0]
    out["T3_reached_failed"] = int(len(rf))
    out["T3_UNREACHED"] = int((s & feas & ~reach3).sum())
    for i in rf:
        gb = np.nonzero(gm[i])[0]
        wb = gb[np.argmax(pos3[i, gb])]
        k = bucket(int(best_edge[i, wb]))
        out["T3_reached_failed__worst_gold_block_best_edge_vote_rank"][k] = out["T3_reached_failed__worst_gold_block_best_edge_vote_rank"].get(k, 0) + 1
    res["per_hop"]["hop%d" % h] = out
G.log(json.dumps(res))
G.S.wj(os.path.join(OUT, "edgegeo_why_A_%s.json" % name), res)
