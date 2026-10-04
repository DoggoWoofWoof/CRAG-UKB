"""Attribution for MICRO_L3_H2 on DEV_A -- re-runs the beam exactly as _l1c_microl3 does (same source slice executed), adds no arm
and changes no ranking.  Questions:
  1. node level: does the beam visit a gold NODE (the answer entity itself)?  at which visited rank (seed / hop-1 beam / hop-2 beam)?
  2. block level, per hop and per outcome (ALL@P50 / reached-failed / UNREACHED under H2): position of the worst gold block in the
     repair order (rep2) and in the fused H2 order; the visited-node rank that first voted for it; the repair vote count of the worst
     gold block vs the top repair block (how many visited nodes voted for each).
  3. hop-1 -> hop-2 transitions: queries correct after hop 1 but not after hop 2 (lost by adding the hop-2 votes): where the gold
     block moved; how many blocks the repair channel votes after hop 1 vs hop 2 (saturation); mean membership size of beam nodes.
    python -u _l1c_microl3_why.py <cache>     -> results/L1_COVPART/microl3_why_A_<cache>.json
"""
import json
import os
import sys

import numpy as np

import _l1g_core as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
src = open(os.path.join(HERE, "_l1c_microl3.py"), encoding="utf-8").read()
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
exec(head)                                                   # D, beams1, beams2, repair_channel ... identical construction
rep1, S1 = repair_channel(beams1)
rep2, S2 = repair_channel([beams1[qi] + beams2[qi] for qi in range(nq)])
base_rank = np.asarray(C.base_rank, np.int64)
sel0 = X.fuse(C, base_rank, base_rank, "T")
sel1 = X.fuse(C, base_rank, rep1, "RRF")
sel2 = X.fuse(C, base_rank, rep2, "RRF")
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
feas = gs <= P_MAIN
ev_node = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
BIG = 10 ** 6


def full_rrf_positions(rep):
    """position of every block in the FULL symmetric-RRF order of (base_rank, rep) -- the order whose top-50 is the fused P50."""
    pt = X.rank_pos(base_rank, npart)
    ps = X.rank_pos(rep, npart)
    pos = np.empty((nq, npart), np.int64)
    for i in range(nq):
        wt = 1.0 / (K0 + pt[i].astype(np.float64))
        ws = 1.0 / (K0 + ps[i].astype(np.float64))
        wt[pt[i] >= BIG] = 0.0
        ws[ps[i] >= BIG] = 0.0
        o = np.argsort(-(wt + ws), kind="stable")
        pos[i, o] = np.arange(npart)
    return pos, ps


def ok_of(sel):
    allv, _ = C.cover(sel)
    return allv.astype(bool)


ok0, ok1, ok2 = ok_of(sel0), ok_of(sel1), ok_of(sel2)
assert (ok0[m] == np.asarray(D.base_all, bool)[m]).all()
reach0 = (gm & ~ev_node).sum(axis=1) == 0
reach2 = (gm & ~(ev_node | (S2 > 0))).sum(axis=1) == 0
pos1, rp1 = full_rrf_positions(rep1)
pos2, rp2 = full_rrf_positions(rep2)
pos0 = X.rank_pos(base_rank, npart)

# first visited-node rank (0.. hop-1 beam, 100.. hop-2 beam) that votes for each block, and the vote COUNT per block
first_rank = np.full((nq, npart), BIG, np.int64)
n_votes = np.zeros((nq, npart), np.int64)
mem_size_b1, mem_size_b2, mem_size_hits = [], [], []
for qi in rowsA:
    lst = beams1[qi] + beams2[qi]
    for r, nd in enumerate(lst):
        ps = mem_flat[mem_ptr[nd]:mem_ptr[nd + 1]]
        first_rank[qi, ps] = np.minimum(first_rank[qi, ps], r)
        n_votes[qi, ps] += 1
    mem_size_b1.append(np.mean([mem_ptr[nd + 1] - mem_ptr[nd] for nd in beams1[qi]]) if beams1[qi] else np.nan)
    mem_size_b2.append(np.mean([mem_ptr[nd + 1] - mem_ptr[nd] for nd in beams2[qi]]) if beams2[qi] else np.nan)
    mem_size_hits.append(np.mean([mem_ptr[nd + 1] - mem_ptr[nd] for nd in D.d_ids[qi, :K_LOCK]]))


def bucket_rank(r):
    if r >= BIG:
        return "none"
    if r < 10:
        return "hop1_rank0-9"
    if r < 50:
        return "hop1_rank10-49"
    if r < 100:
        return "hop1_rank50-99"
    if r < 150:
        return "hop2_rank0-49"
    return "hop2_rank50-99"


def bucket_pos(p):
    if p >= BIG:
        return "unranked"
    return "0-9" if p < 10 else ("10-49" if p < 50 else ("50-59" if p < 60 else ("60-99" if p < 100 else "100+")))


def inc(d, k):
    d[k] = d.get(k, 0) + 1


res = {"cache": name, "n_DEV_A": nA, "L1_BASE_ALL": round(float(ok0[m].mean()), 4), "H1_BASE_ALL": round(float(ok1[m].mean()), 4),
       "H2_BASE_ALL": round(float(ok2[m].mean()), 4),
       "repair_channel_saturation": {"blocks": npart, "blocks_voted_after_hop1_mean": round(float((S1[m] > 0).sum(1).mean()), 1),
                                     "blocks_voted_after_hop2_mean": round(float((S2[m] > 0).sum(1).mean()), 1),
                                     "membership_size_mean__L1_top100_hits": round(float(np.nanmean(mem_size_hits)), 2),
                                     "membership_size_mean__hop1_beam_nodes": round(float(np.nanmean(mem_size_b1)), 2),
                                     "membership_size_mean__hop2_beam_nodes": round(float(np.nanmean(mem_size_b2)), 2)},
       "per_hop": {}}
groups = [("hop%d" % h, m & (hops == h)) for h in sorted(set(int(x) for x in hops[m] if x >= 0))] or [("all", m)]
for hname, smask in groups:
    s = np.nonzero(smask)[0]
    out = {"n": int(len(s)), "L1_ok": int(ok0[s].sum()), "H1_ok": int(ok1[s].sum()), "H2_ok": int(ok2[s].sum()),
           "gold_node_visited": {"any_gold_node__seed": 0, "any_gold_node__hop1_beam": 0, "any_gold_node__hop2_beam": 0, "any_gold_node__not_visited": 0,
                                 "all_gold_nodes_visited": 0, "n_gold_nodes_mean": 0.0},
           "H2_outcome": {"ALL_gold_P50": 0, "reached_failed": 0, "UNREACHED": 0, "infeasible": 0},
           "reached_failed__worst_gold_block": {"position_in_repair_order": {}, "position_in_fused_H2_order": {}, "first_voting_visited_rank": {},
                                                "vote_count_gold_mean": 0.0, "vote_count_top_repair_block_mean": 0.0, "position_in_served_L1_order": {}},
           "ALL_gold_P50__worst_gold_block": {"first_voting_visited_rank": {}, "position_in_served_L1_order": {}},
           "lost_from_H1_to_H2": {"n": 0, "worst_gold_block_position_H1_fused": [], "worst_gold_block_position_H2_fused": [],
                                  "worst_gold_block_position_in_repair_order_H1": [], "worst_gold_block_position_in_repair_order_H2": []},
           "gained_from_H1_to_H2": int((~ok1[s] & ok2[s]).sum()),
           "lost_from_L1_to_H2": {"n": 0, "worst_gold_block_position_in_served_L1_order": {}, "worst_gold_block_position_in_repair_order": {},
                                  "worst_gold_block_position_in_fused_H2_order": {}, "served_P50_blocks_displaced_mean": 0.0},
           "served_P50_blocks_displaced_by_H2_mean": round(float(np.mean([len(set(int(x) for x in sel0[qi]) - set(int(x) for x in sel2[qi])) for qi in s])), 2)}
    ngn, vc_g, vc_t, n_rf, disp = [], [], [], 0, []
    for qi in s:
        if ok0[qi] and not ok2[qi]:
            L = out["lost_from_L1_to_H2"]
            L["n"] += 1
            gb_ = np.nonzero(gm[qi])[0]
            wb_ = gb_[np.argmax(pos2[qi, gb_])]
            inc(L["worst_gold_block_position_in_served_L1_order"], bucket_pos(int(pos0[qi, wb_])))
            inc(L["worst_gold_block_position_in_repair_order"], bucket_pos(int(rp2[qi, wb_])))
            inc(L["worst_gold_block_position_in_fused_H2_order"], bucket_pos(int(pos2[qi, wb_])))
            disp.append(len(set(int(x) for x in sel0[qi]) - set(int(x) for x in sel2[qi])))
        gn = C.gold_nodes[qi]
        ngn.append(len(gn))
        sd = set(int(x) for x in seeds[qi] if x >= 0)
        b1, b2 = set(beams1[qi]), set(beams2[qi])
        vis = sd | b1 | b2
        g = set(int(x) for x in gn)
        if g & sd:
            inc(out["gold_node_visited"], "any_gold_node__seed")
        elif g & b1:
            inc(out["gold_node_visited"], "any_gold_node__hop1_beam")
        elif g & b2:
            inc(out["gold_node_visited"], "any_gold_node__hop2_beam")
        else:
            inc(out["gold_node_visited"], "any_gold_node__not_visited")
        out["gold_node_visited"]["all_gold_nodes_visited"] += int(len(g) > 0 and g <= vis)
        gb = np.nonzero(gm[qi])[0]
        if not feas[qi]:
            inc(out["H2_outcome"], "infeasible")
            continue
        wb = gb[np.argmax(pos2[qi, gb])]                       # worst gold block under the H2 fused order
        if ok2[qi]:
            inc(out["H2_outcome"], "ALL_gold_P50")
            inc(out["ALL_gold_P50__worst_gold_block"]["first_voting_visited_rank"], bucket_rank(int(first_rank[qi, wb])))
            inc(out["ALL_gold_P50__worst_gold_block"]["position_in_served_L1_order"], bucket_pos(int(pos0[qi, wb])))
        elif reach2[qi]:
            inc(out["H2_outcome"], "reached_failed")
            n_rf += 1
            d = out["reached_failed__worst_gold_block"]
            inc(d["position_in_repair_order"], bucket_pos(int(rp2[qi, wb])))
            inc(d["position_in_fused_H2_order"], bucket_pos(int(pos2[qi, wb])))
            inc(d["first_voting_visited_rank"], bucket_rank(int(first_rank[qi, wb])))
            inc(d["position_in_served_L1_order"], bucket_pos(int(pos0[qi, wb])))
            vc_g.append(int(n_votes[qi, wb]))
            vc_t.append(int(n_votes[qi].max()))
        else:
            inc(out["H2_outcome"], "UNREACHED")
        if ok1[qi] and not ok2[qi]:
            L = out["lost_from_H1_to_H2"]
            L["n"] += 1
            wb1 = gb[np.argmax(pos1[qi, gb])]
            L["worst_gold_block_position_H1_fused"].append(int(pos1[qi, wb1]))
            L["worst_gold_block_position_H2_fused"].append(int(pos2[qi, wb]))
            L["worst_gold_block_position_in_repair_order_H1"].append(int(rp1[qi, wb1]) if rp1[qi, wb1] < BIG else None)
            L["worst_gold_block_position_in_repair_order_H2"].append(int(rp2[qi, wb]) if rp2[qi, wb] < BIG else None)
    out["gold_node_visited"]["n_gold_nodes_mean"] = round(float(np.mean(ngn)), 2)
    if disp:
        out["lost_from_L1_to_H2"]["served_P50_blocks_displaced_mean"] = round(float(np.mean(disp)), 2)
    if n_rf:
        out["reached_failed__worst_gold_block"]["vote_count_gold_mean"] = round(float(np.mean(vc_g)), 2)
        out["reached_failed__worst_gold_block"]["vote_count_top_repair_block_mean"] = round(float(np.mean(vc_t)), 2)
    res["per_hop"][hname] = out
G.log(json.dumps(res))
G.S.wj(os.path.join(OUT, "microl3_why_A_%s.json" % name), res)
