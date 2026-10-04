"""H2_PATCH1 (eighth ruling of 2026-09-15; pre-registered in PREREGISTRATION_H2_PATCH1_DEV_A.json, STATUS POSTHOC_MECHANISM_TEST):
canonical SAFE with its weakest admitted block (the sixth of the frozen F6 competition) replaced by a query-local virtual block of
exactly the same size, filled with the pinned uL3 beam's visited nodes that the 49 kept blocks do not expose (frozen first-visit
order), topped up from the removed block's own nodes when there are fewer novel nodes than the block's size.  Unique-node exposure
is therefore matched exactly with canonical SAFE.  Metric: ALL-gold under matched exposure (every gold NODE served); for L1 and
SAFE this is asserted equal to the frozen block-level ALL-gold@P50.
    python -u _l1c_h2patch.py <cache>      -> results/L1_COVPART/h2patch_A_<cache>.json
    python -u _l1c_h2patch.py --summary    -> results/L1_COVPART/h2patch_A_SUMMARY.json  (applies the pre-registered rule)
"""
import hashlib
import json
import os
import sys

import numpy as np

import _l1g_core as G
import _l1kb_core as KB
import _l1ps_router as RT

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PINNED_SHA = "19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5"
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
SAFE_RECORD = {"metaqa": "results/L1_CANONICAL/L1_REPLAY_metaqa.json",
               "squad": "results/L1_CANONICAL/L1_REPLAY_squad.json",
               "metaqa_phg": "results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
               "squad_phg": "results/L1_LOWMEM/L1_REPLAY_squad__LOWMEM__PHG_con.json",
               "musique": "results/L1_LOWMEM/L1_REPLAY_musique__LOWMEM__PHG_C1_con.json"}
CFG = dict(KB.BASE_CFG)                                      # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
B = CFG["B"]
ARM = "H2_PATCH1"


def summary():
    recs = {c: json.load(open(os.path.join(OUT, "h2patch_A_%s.json" % c), encoding="utf-8")) for c in CACHES}
    text_ok = {c: not recs[c]["arms"][ARM]["significant_loss_vs_SAFE"] for c in ("squad", "squad_phg", "musique")}
    mq = {}
    for c in ("metaqa", "metaqa_phg"):
        h = recs[c]["arms"][ARM]["per_hop"]["hop2"]["vs_SAFE"]
        mq[c] = {"gained": h["gained"], "lost": h["lost"], "p": h["p"], "pass": bool(h["gained"] > h["lost"] and h["p"] < 0.01)}
    verdict = "PASS" if all(text_ok.values()) and all(v["pass"] for v in mq.values()) else "FAIL"
    S = {"RECORD": "H2_PATCH1_DEV_A_SUMMARY", "STATUS": "POSTHOC_MECHANISM_TEST",
         "rule": "text safety on squad/squad_phg/musique (no significant loss vs SAFE) AND hop-2 gain vs SAFE with p < 0.01 on both metaqa caches",
         "text_safety": text_ok, "metaqa_hop2_vs_SAFE": mq, "verdict": verdict,
         "per_cache": {c: {"L1": recs[c]["arms"]["L1"]["ALL"], "SAFE": recs[c]["arms"]["SAFE"]["ALL"], ARM: recs[c]["arms"][ARM]["ALL"],
                           ARM + "_vs_SAFE": recs[c]["arms"][ARM]["ALL_vs_SAFE"], ARM + "_vs_L1": recs[c]["arms"][ARM]["ALL_vs_L1"],
                           "recovered_fraction_of_H2_RRF_hop2_gain": recs[c]["arms"][ARM].get("recovery_of_section14_gain", {}).get("hop2", {}).get("fraction"),
                           "diagnostic_98": recs[c].get("diagnostic_block_level_pool_repairs", {}).get("hop2")}
                       for c in CACHES}}
    G.S.wj(os.path.join(OUT, "h2patch_A_SUMMARY.json"), S)
    G.log("SUMMARY verdict %s text_ok %s metaqa %s" % (verdict, text_ok, mq))


if "--summary" in sys.argv:
    summary()
    sys.exit(0)

name = sys.argv[1]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
assert hashlib.sha256(raw).hexdigest() == PINNED_SHA, "pinned uL3 module changed"
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
head = head.replace("name = sys.argv[1]", "name = %r" % name).replace('LATENT = "--latent" in sys.argv', "LATENT = False")
exec(head)                                                   # D, C, beams1, beams2, repair_channel, hard, hops, m, rowsA ... identical beam

# ---------------------------------------------------------------- identical beam: rebuild section 14.1's MICRO_L3_H2 and match its record
rep2, S2 = repair_channel([beams1[qi] + beams2[qi] for qi in range(nq)])
base_rank = np.asarray(C.base_rank, np.int64)
allv_rrf, _ = C.cover(X.fuse(C, base_rank, rep2, "RRF"))
allv_rrf = allv_rrf.astype(bool)
rec14 = json.load(open(os.path.join(OUT, "microl3_A_%s.json" % name), encoding="utf-8"))
a0 = np.asarray(D.base_all, bool)
g14, l14, _ = X.mcnemar(a0[m], allv_rrf[m])
assert round(float(allv_rrf[m].mean()), 4) == rec14["arms"]["MICRO_L3_H2"]["BASE_ALL"], "beam differs from section 14.1"
assert (g14, l14) == tuple(rec14["arms"]["MICRO_L3_H2"]["BASE_ALL_vs_L1"][k] for k in ("gained", "lost")), "beam differs from section 14.1"
G.log("identical beam: MICRO_L3_H2 rebuilt = %.4f (+%d/-%d) == section 14.1 record" % (allv_rrf[m].mean(), g14, l14))
del rep2, S2

# ---------------------------------------------------------------- canonical SAFE machinery (frozen code path), on the raw cache arrays
z0 = np.load(C.path, allow_pickle=True)
zz = {k: z0[k] for k in z0.files if k != "meta_json"}
meta = C.meta
assert int(meta["n_dev_queries"]) == nq
Cc = RT.build_cache(name, zz, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
ctxs = KB.contexts(zz, meta, Cc, B, CFG)
goldp = Cc["goldp"]
ind_base = np.asarray(Cc["ind_base"], bool)
assert (ind_base[m] == a0[m]).all(), "SAFE machinery BASE differs from the lane's served BASE"
sizes = np.asarray(C.part_sizes, np.int64)
assert int(sizes.sum()) == N and (np.bincount(hard, minlength=npart) == sizes).all()
order_nodes = np.argsort(hard, kind="stable")
ptr = np.zeros(npart + 1, np.int64)
ptr[1:] = np.cumsum(sizes)


def nodes_of(b):
    return order_nodes[ptr[b]:ptr[b + 1]]


gold_nodes = [np.asarray(C.gold_nodes[qi], np.int64) for qi in range(nq)]
for qi in rowsA:                                             # gold blocks are exactly the blocks of the gold nodes
    assert set(int(hard[g]) for g in gold_nodes[qi]) == goldp[qi]

# ---------------------------------------------------------------- the arm
finals = {"L1": [set() for _ in range(nq)], "SAFE": [set() for _ in range(nq)]}
patch = [None] * nq                                          # per DEV_A query: dict(kept, weakest, nodes, ...)
for qi in rowsA:
    c = ctxs[qi]
    finals["L1"][qi] = set(c["base50"])
    Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
    safe50 = c["prot_set"] | set(Xs)
    assert len(safe50) == P_MAIN
    finals["SAFE"][qi] = safe50
    weakest = int(Xs[-1])                                    # the sixth block of the F6 competition = SAFE's weakest admission
    kept = safe50 - {weakest}
    nb = int(sizes[weakest])
    b1, b2 = beams1[qi], beams2[qi]
    nov1 = [int(v) for v in b1 if int(hard[v]) not in kept]
    nov2 = [int(v) for v in b2 if int(hard[v]) not in kept]
    novel = nov1 + nov2                                      # frozen first-visit order (the pre-registered order)
    assert len(set(novel)) == len(novel)
    nodes = novel[:nb]
    n_novel_used = len(nodes)
    n_fill = 0
    if len(nodes) < nb:
        rpos_node = {int(v): r for r, v in enumerate(C.ret_rrf[qi]) if v >= 0}
        used = set(nodes)
        fill = sorted((int(v) for v in nodes_of(weakest) if int(v) not in used), key=lambda v: (rpos_node.get(v, 10 ** 9), v))
        n_fill = nb - len(nodes)
        nodes = nodes + fill[:n_fill]
    assert len(nodes) == nb and len(set(nodes)) == nb
    pset = set(nodes)
    s1 = set(nov1)
    patch[qi] = {"kept": kept, "weakest": weakest, "nb": nb, "nodes": pset, "k": len(novel), "nov1": len(nov1), "nov2": len(nov2),
                 "truncated": int(len(novel) > nb), "n_novel_used": n_novel_used, "n_fill": n_fill,
                 "used_hop1": sum(1 for v in nodes[:n_novel_used] if v in s1), "used_hop2": sum(1 for v in nodes[:n_novel_used] if v not in s1),
                 "novel_set": set(novel), "novel_pos": {v: r for r, v in enumerate(novel)}}
    # matched exposure
    assert int(sizes[list(kept)].sum()) + nb == int(sizes[list(safe50)].sum())


def served_by(arm, qi, g):
    """is gold node g served by the arm?"""
    b = int(hard[g])
    if arm == "L1":
        return b in finals["L1"][qi]
    if arm == "SAFE":
        return b in finals["SAFE"][qi]
    p = patch[qi]
    return b in p["kept"] or g in p["nodes"]


ind = {arm: np.zeros(nq, bool) for arm in ("L1", "SAFE", ARM)}
ind_any = {arm: np.zeros(nq, bool) for arm in ("L1", "SAFE", ARM)}
for arm in ind:
    for qi in rowsA:
        flags = [served_by(arm, qi, int(g)) for g in gold_nodes[qi]]
        ind[arm][qi] = bool(flags) and all(flags)
        ind_any[arm][qi] = any(flags)
# node-level == block-level for the block arms; canonical SAFE reproduced against the frozen replay record
assert (ind["L1"][m] == ind_base[m]).all()
ind_safe_block = np.array([int(goldp[qi] <= finals["SAFE"][qi]) for qi in range(nq)], bool)
assert (ind["SAFE"][m] == ind_safe_block[m]).all()
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert int(recS["population"]["nq"]) == nq and os.path.basename(recS["cache"]["file"]) == os.path.basename(C.path), (recS["cache"]["file"], C.path)
assert (ind["L1"][m] == np.asarray(recS["_ind_BASE"], bool)[m]).all(), "BASE differs from the frozen replay record"
assert (ind["SAFE"][m] == np.asarray(recS["_ind_SAFE"], bool)[m]).all(), "canonical SAFE not reproduced on DEV_A rows"
G.log("canonical SAFE reproduced on DEV_A rows against %s (SAFE %.4f, BASE %.4f); node-level == block-level asserted" % (os.path.basename(SAFE_RECORD[name]), ind["SAFE"][m].mean(), ind["L1"][m].mean()))

# ---------------------------------------------------------------- per-query gold accounting of the patch (evaluation only)
gain_nodes = np.zeros(nq, np.int64)       # gold nodes served by the patch arm and not by SAFE
gain_hop1 = np.zeros(nq, np.int64)
loss_nodes = np.zeros(nq, np.int64)       # gold nodes served by SAFE (in the removed block) and dropped by the patch arm
for qi in rowsA:
    p = patch[qi]
    for g in gold_nodes[qi]:
        g = int(g)
        sS, sP = served_by("SAFE", qi, g), served_by(ARM, qi, g)
        if sP and not sS:
            gain_nodes[qi] += 1
            gain_hop1[qi] += int(p["novel_pos"].get(g, 10 ** 9) < p["nov1"])
        elif sS and not sP:
            loss_nodes[qi] += 1
            assert int(hard[g]) == p["weakest"]


def mc(a, b, s):
    g_, l_, p_ = X.mcnemar(a[s], b[s])
    return {"gained": g_, "lost": l_, "p": p_}


def evaluate(arm):
    iv, ia = ind[arm], ind_any[arm]
    o = {"arm": arm, "ALL": round(float(iv[m].mean()), 4), "ANY": round(float(ia[m].mean()), 4), "per_hop": {}}
    if arm == ARM:
        o["exposure"] = "matched: |served nodes| equal to canonical SAFE per query (asserted)"
        o["novel_nodes_per_query_mean"] = round(float(np.mean([patch[qi]["k"] for qi in rowsA])), 1)
        o["novel_hop1_per_query_mean"] = round(float(np.mean([patch[qi]["nov1"] for qi in rowsA])), 1)
        o["novel_hop2_per_query_mean"] = round(float(np.mean([patch[qi]["nov2"] for qi in rowsA])), 1)
        o["patch_size_mean"] = round(float(np.mean([patch[qi]["nb"] for qi in rowsA])), 1)
        o["patch_composition_mean (hop-1 novel / hop-2 novel / fill from removed block)"] = [round(float(np.mean([patch[qi][k] for qi in rowsA])), 1) for k in ("used_hop1", "used_hop2", "n_fill")]
        o["queries_truncated"] = int(sum(patch[qi]["truncated"] for qi in rowsA))
        o["queries_with_no_novel_node (patch == removed block)"] = int(sum(patch[qi]["k"] == 0 for qi in rowsA))
        o["gold_nodes_gained_vs_SAFE"] = int(gain_nodes[rowsA].sum())
        o["gold_nodes_gained_vs_SAFE_from_hop1_novel"] = int(gain_hop1[rowsA].sum())
        o["queries_with_gold_node_gained_vs_SAFE"] = int((gain_nodes[rowsA] > 0).sum())
        o["gold_nodes_lost_with_removed_block"] = int(loss_nodes[rowsA].sum())
        o["queries_with_gold_node_lost_with_removed_block"] = int((loss_nodes[rowsA] > 0).sum())
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        o["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "ALL": round(float(iv[s].mean()), 4), "ANY": round(float(ia[s].mean()), 4)}
        if arm != "L1":
            o["per_hop"]["hop%d" % h]["vs_L1"] = mc(ind["L1"], iv, s)
        if arm == ARM:
            o["per_hop"]["hop%d" % h]["vs_SAFE"] = mc(ind["SAFE"], iv, s)
            o["per_hop"]["hop%d" % h]["gold_nodes_gained"] = int(gain_nodes[s].sum())
            o["per_hop"]["hop%d" % h]["gold_nodes_lost_with_removed_block"] = int(loss_nodes[s].sum())
    if arm != "L1":
        o["ALL_vs_L1"] = mc(ind["L1"], iv, m)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_L1"] = {"n": int(s.sum()), **mc(ind["L1"], iv, s), "pooled_L1": round(float(ind["L1"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    if arm == ARM:
        o["ALL_vs_SAFE"] = mc(ind["SAFE"], iv, m)
        o["ANY_vs_SAFE"] = mc(ind_any["SAFE"], ia, m)
        o["significant_loss_vs_SAFE"] = bool(o["ALL_vs_SAFE"]["lost"] > o["ALL_vs_SAFE"]["gained"] and o["ALL_vs_SAFE"]["p"] < 0.05)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_SAFE"] = {"n": int(s.sum()), **mc(ind["SAFE"], iv, s), "pooled_SAFE": round(float(ind["SAFE"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
            recov = {}
            for key, s in [("overall", m)] + [(h, m & (hops == int(h[3:]))) for h in o["per_hop"]] + [("hop2_hop3_pooled", m & (hops >= 2))]:
                d14 = float(allv_rrf[s].mean() - ind["L1"][s].mean())
                d_ = float(iv[s].mean() - ind["SAFE"][s].mean())
                recov[key] = {"H2_RRF_minus_L1_pts": round(100 * d14, 1), "PATCH_minus_SAFE_pts": round(100 * d_, 1), "fraction": (round(d_ / d14, 3) if abs(d14) > 1e-12 else None)}
            o["recovery_of_section14_gain"] = recov
    return o


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "POSTHOC_MECHANISM_TEST", "arm": ARM,
       "beam_module_sha256": PINNED_SHA, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_record": SAFE_RECORD[name], "identical_beam_check": {"MICRO_L3_H2_rebuilt": round(float(allv_rrf[m].mean()), 4), "gained": g14, "lost": l14},
       "definition": {"weakest": "sixth block of the frozen F6 competition order (SAFE's weakest admission)",
                      "novel": "pinned-beam visited nodes (hop-1 then hop-2 beam, frozen fused order, seeds excluded) whose block is not among the 49 kept blocks",
                      "patch": "first min(k, n_b) novel nodes; remaining positions = removed block's own nodes in frozen fused node order (ret_rrf position, then node id)",
                      "metric": "ALL-gold under matched unique-node exposure (every gold node served); node-level == block-level for L1/SAFE (asserted)"},
       "arms": {}}
for arm in ("L1", "SAFE", ARM):
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-9s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "recovery_of_section14_gain", "arm")})))
    if o["per_hop"]:
        G.log("          per hop %s" % json.dumps(o["per_hop"]))
    if arm == ARM and o.get("recovery_of_section14_gain"):
        G.log("          recovery %s" % json.dumps(o["recovery_of_section14_gain"]))

# ---------------------------------------------------------------- flips vs SAFE with their cause
flips = {"newly_covered_vs_SAFE": {"n": 0, "gold_nodes_entering_via_patch": 0, "of_which_hop1_novel": 0, "queries_needing_more_than_one_patch_node": 0},
         "newly_lost_vs_SAFE": {"n": 0, "gold_nodes_dropped_with_removed_block": 0, "queries_also_gaining_a_gold_node": 0}}
for qi in rowsA:
    if ind[ARM][qi] and not ind["SAFE"][qi]:
        f = flips["newly_covered_vs_SAFE"]
        f["n"] += 1
        f["gold_nodes_entering_via_patch"] += int(gain_nodes[qi])
        f["of_which_hop1_novel"] += int(gain_hop1[qi])
        f["queries_needing_more_than_one_patch_node"] += int(gain_nodes[qi] > 1)
    elif ind["SAFE"][qi] and not ind[ARM][qi]:
        f = flips["newly_lost_vs_SAFE"]
        f["n"] += 1
        f["gold_nodes_dropped_with_removed_block"] += int(loss_nodes[qi])
        f["queries_also_gaining_a_gold_node"] += int(gain_nodes[qi] > 0)
res["attribution_flips"] = flips
G.log("flips %s" % json.dumps(flips))

# ---------------------------------------------------------------- THE DIAGNOSTIC: the block-level complete repairs of section 15 (98 / 86) and the node-level availability
groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else [("all", list(rowsA))]
diag_block, diag_node = {}, {}
for gname, rows_h in groups:
    ob = {"L1_failures (gold block outside served P50)": 0, "repairable_by_any_six_slot_rule (missing blocks <= 6)": 0,
          "all_missing_blocks_in_SAFE_H2_pool (section 15)": 0, "of_which_covered_by_SAFE": 0, "of_which_covered_by_H2_PATCH1": 0,
          "of_which_all_missing_gold_NODES_visited_by_beam": 0, "of_which_all_missing_gold_NODES_visited_and_covered_by_H2_PATCH1": 0}
    on = {"SAFE_failures (a gold node not served by SAFE)": 0, "all_missing_gold_nodes_visited_by_beam (node-level repair available)": 0,
          "of_which_covered_by_H2_PATCH1": 0, "not_covered__truncated (a missing gold node beyond the patch capacity)": 0,
          "not_covered__gold_node_dropped_with_removed_block": 0, "not_covered__other": 0,
          "missing_gold_nodes_per_SAFE_failure_mean": [], "missing_gold_nodes_visited_fraction_per_SAFE_failure_mean": [],
          "SAFE_failures_with_no_missing_gold_node_visited": 0}
    for qi in rows_h:
        c = ctxs[qi]
        p = patch[qi]
        h2b = set(int(hard[v]) for v in beams1[qi] + beams2[qi])
        vis = p["novel_set"] | set(int(v) for v in beams1[qi] + beams2[qi])
        miss_b = [b for b in goldp[qi] if b not in c["base50"]]
        if miss_b:
            ob["L1_failures (gold block outside served P50)"] += 1
            if len(miss_b) <= B:
                ob["repairable_by_any_six_slot_rule (missing blocks <= 6)"] += 1
                pool = set(c["bnd"]) | set(c["chal"]) | h2b
                if all(b in pool for b in miss_b):
                    ob["all_missing_blocks_in_SAFE_H2_pool (section 15)"] += 1
                    ob["of_which_covered_by_SAFE"] += int(ind["SAFE"][qi])
                    ob["of_which_covered_by_H2_PATCH1"] += int(ind[ARM][qi])
                    miss_n = [int(g) for g in gold_nodes[qi] if not served_by("SAFE", qi, int(g))]
                    if all(g in vis for g in miss_n):
                        ob["of_which_all_missing_gold_NODES_visited_by_beam"] += 1
                        ob["of_which_all_missing_gold_NODES_visited_and_covered_by_H2_PATCH1"] += int(ind[ARM][qi])
        if not ind["SAFE"][qi] and len(gold_nodes[qi]):
            on["SAFE_failures (a gold node not served by SAFE)"] += 1
            miss_n = [int(g) for g in gold_nodes[qi] if not served_by("SAFE", qi, int(g))]
            nv = sum(1 for g in miss_n if g in vis)
            on["missing_gold_nodes_per_SAFE_failure_mean"].append(len(miss_n))
            on["missing_gold_nodes_visited_fraction_per_SAFE_failure_mean"].append(nv / max(1, len(miss_n)))
            on["SAFE_failures_with_no_missing_gold_node_visited"] += int(nv == 0)
            if nv == len(miss_n):
                on["all_missing_gold_nodes_visited_by_beam (node-level repair available)"] += 1
                if ind[ARM][qi]:
                    on["of_which_covered_by_H2_PATCH1"] += 1
                elif any(p["novel_pos"].get(g, -1) >= p["nb"] for g in miss_n):
                    on["not_covered__truncated (a missing gold node beyond the patch capacity)"] += 1
                elif loss_nodes[qi] > 0:
                    on["not_covered__gold_node_dropped_with_removed_block"] += 1
                else:
                    on["not_covered__other"] += 1
    on["missing_gold_nodes_per_SAFE_failure_mean"] = round(float(np.mean(on["missing_gold_nodes_per_SAFE_failure_mean"])), 2) if on["missing_gold_nodes_per_SAFE_failure_mean"] else None
    on["missing_gold_nodes_visited_fraction_per_SAFE_failure_mean"] = round(float(np.mean(on["missing_gold_nodes_visited_fraction_per_SAFE_failure_mean"])), 3) if on["missing_gold_nodes_visited_fraction_per_SAFE_failure_mean"] else None
    diag_block[gname], diag_node[gname] = ob, on
    G.log("diag %s block-level %s" % (gname, json.dumps(ob)))
    G.log("diag %s node-level  %s" % (gname, json.dumps(on)))
res["diagnostic_block_level_pool_repairs"] = diag_block
res["diagnostic_node_level_availability"] = diag_node
# cross-check the section 15 ceiling record where it exists
cp = os.path.join(OUT, "h2safe_ceiling_A_%s.json" % name)
if os.path.exists(cp):
    ce = json.load(open(cp, encoding="utf-8"))["groups"]
    for gname in diag_block:
        assert diag_block[gname]["all_missing_blocks_in_SAFE_H2_pool (section 15)"] == ce[gname]["repairable_and_all_missing_in_SAFE_H2_candidate_pool"], gname
    res["ceiling_record_crosscheck"] = "section 15 pool counts reproduced"
    G.log("section 15 pool counts reproduced against %s" % os.path.basename(cp))

G.S.wj(os.path.join(OUT, "h2patch_A_%s.json" % name), res)
G.log("done")
