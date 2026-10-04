"""H3_PATCH1 (ninth ruling of 2026-09-15; pre-registered in PREREGISTRATION_H3_PATCH1_DEV_A.json, STATUS POSTHOC_MECHANISM_TEST):
the section-16 rule (canonical SAFE minus its weakest admitted block + one query-local block of the same size holding the pinned
beam's visited nodes that the 49 kept blocks do not expose, frozen first-visit order, fill from the removed block) with the pinned
uL3 beam executed at DEPTH = 3 instead of 2 -- the only changed factor.  H2_PATCH1 is recomputed here from beams 1+2 and asserted
equal to the section-16 record; hops 1-2 of the depth-3 beam are the identical code (asserted against the section-14.1 record).
    python -u _l1c_h3patch.py <cache>      -> results/L1_COVPART/h3patch_A_<cache>.json
    python -u _l1c_h3patch.py --summary    -> results/L1_COVPART/h3patch_A_SUMMARY.json  (applies the pre-registered rule)
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
H2, H3 = "H2_PATCH1", "H3_PATCH1"
ARMS = ("L1", "SAFE", H2, H3)


def summary():
    recs = {c: json.load(open(os.path.join(OUT, "h3patch_A_%s.json" % c), encoding="utf-8")) for c in CACHES}

    def bad(d):                                              # significant loss
        return bool(d["lost"] > d["gained"] and d["p"] < 0.05)

    text = {c: {"vs_H2_PATCH1": recs[c]["arms"][H3]["ALL_vs_H2_PATCH1"], "vs_SAFE": recs[c]["arms"][H3]["ALL_vs_SAFE"]} for c in ("squad", "squad_phg", "musique")}
    text_ok = {c: not bad(text[c]["vs_H2_PATCH1"]) and not bad(text[c]["vs_SAFE"]) for c in text}
    hop3, hop2 = {}, {}
    for c in ("metaqa", "metaqa_phg"):
        h = recs[c]["arms"][H3]["per_hop"]["hop3"]["vs_H2_PATCH1"]
        hop3[c] = {**h, "pass": bool(h["gained"] > h["lost"] and h["p"] < 0.01)}
        h = recs[c]["arms"][H3]["per_hop"]["hop2"]["vs_H2_PATCH1"]
        hop2[c] = {**h, "no_regression": not bad(h)}
    verdict = "PASS" if all(text_ok.values()) and all(v["pass"] for v in hop3.values()) and all(v["no_regression"] for v in hop2.values()) else "FAIL"
    S = {"RECORD": "H3_PATCH1_DEV_A_SUMMARY", "STATUS": "POSTHOC_MECHANISM_TEST",
         "rule": "text safety on squad/squad_phg/musique (no significant loss vs H2_PATCH1 and vs SAFE) AND hop-3 gain vs H2_PATCH1 with p < 0.01 on both "
                 "metaqa caches AND no significant hop-2 loss vs H2_PATCH1 on both metaqa caches",
         "text_safety": text_ok, "text_contrasts": text, "metaqa_hop3_vs_H2_PATCH1": hop3, "metaqa_hop2_vs_H2_PATCH1": hop2, "verdict": verdict,
         "per_cache": {c: {a: recs[c]["arms"][a]["ALL"] for a in ARMS} | {H3 + "_vs_H2_PATCH1": recs[c]["arms"][H3]["ALL_vs_H2_PATCH1"],
                                                                          H3 + "_vs_SAFE": recs[c]["arms"][H3]["ALL_vs_SAFE"],
                                                                          "H2_PATCH1_reproduced": recs[c]["H2_PATCH1_reproduction_check"]} for c in CACHES}}
    G.S.wj(os.path.join(OUT, "h3patch_A_SUMMARY.json"), S)
    G.log("SUMMARY verdict %s text_ok %s hop3 %s hop2 %s" % (verdict, text_ok, hop3, hop2))


if "--summary" in sys.argv:
    summary()
    sys.exit(0)

name = sys.argv[1]
raw = open(os.path.join(HERE, "_l1c_microl3.py"), "rb").read()
assert hashlib.sha256(raw).hexdigest() == PINNED_SHA, "pinned uL3 module changed"
src = raw.decode("utf-8")
head = src[src.index("import json"):src.index("# ---------------------------------------------------------------- repair + evaluation")]
PRE = json.load(open(os.path.join(OUT, "PREREGISTRATION_H3_PATCH1_DEV_A.json"), encoding="utf-8"))
SUBS = [("name = sys.argv[1]", "name = %r" % name),
        ('LATENT = "--latent" in sys.argv', "LATENT = False"),
        ("DEPTH = 2", "DEPTH = 3"),
        ("beams1, beams2 = [[] for _ in range(nq)], [[] for _ in range(nq)]",
         "beams1, beams2, beams3 = [[] for _ in range(nq)], [[] for _ in range(nq)], [[] for _ in range(nq)]"),
        ("beams1[qi], beams2[qi] = beams[0], beams[1]", "beams1[qi], beams2[qi], beams3[qi] = beams[0], beams[1], beams[2]")]
assert [list(x) for x in SUBS[1:]] == [list(x) for x in PRE["frozen_and_reused_verbatim"]["head_substitutions"][1:]], "substitutions differ from the pre-registration"
for a, b in SUBS:
    assert head.count(a) == 1, a
    head = head.replace(a, b)
exec(head)                                                   # D, C, beams1..3, edges_h (nA, 3), secs, repair_channel, hard, hops, m, rowsA ...
assert DEPTH == 3 and edges_h.shape[1] == 3

# ---------------------------------------------------------------- identical hops 1-2: rebuild section 14.1's MICRO_L3_H2 and match its record
rep2, S2 = repair_channel([beams1[qi] + beams2[qi] for qi in range(nq)])
base_rank = np.asarray(C.base_rank, np.int64)
allv_rrf, _ = C.cover(X.fuse(C, base_rank, rep2, "RRF"))
allv_rrf = allv_rrf.astype(bool)
rec14 = json.load(open(os.path.join(OUT, "microl3_A_%s.json" % name), encoding="utf-8"))
a0 = np.asarray(D.base_all, bool)
g14, l14, _ = X.mcnemar(a0[m], allv_rrf[m])
assert round(float(allv_rrf[m].mean()), 4) == rec14["arms"]["MICRO_L3_H2"]["BASE_ALL"], "hops 1-2 differ from section 14.1"
assert (g14, l14) == tuple(rec14["arms"]["MICRO_L3_H2"]["BASE_ALL_vs_L1"][k] for k in ("gained", "lost")), "hops 1-2 differ from section 14.1"
G.log("identical hops 1-2: MICRO_L3_H2 rebuilt = %.4f (+%d/-%d) == section 14.1 record" % (allv_rrf[m].mean(), g14, l14))
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
for qi in rowsA:
    assert set(int(hard[g]) for g in gold_nodes[qi]) == goldp[qi]

# ---------------------------------------------------------------- the arms (H2_PATCH1 from beams 1+2, H3_PATCH1 from beams 1+2+3; same rule)
finals = {"L1": [set() for _ in range(nq)], "SAFE": [set() for _ in range(nq)]}
patch = {H2: [None] * nq, H3: [None] * nq}
for qi in rowsA:
    c = ctxs[qi]
    finals["L1"][qi] = set(c["base50"])
    Xs, sc = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], B)
    safe50 = c["prot_set"] | set(Xs)
    assert len(safe50) == P_MAIN
    finals["SAFE"][qi] = safe50
    weakest = int(Xs[-1])
    kept = safe50 - {weakest}
    nb = int(sizes[weakest])
    nov = [[int(v) for v in bl if int(hard[v]) not in kept] for bl in (beams1[qi], beams2[qi], beams3[qi])]
    rpos_node = None
    for arm, depth in ((H2, 2), (H3, 3)):
        novel = sum(nov[:depth], [])                         # frozen first-visit order: hop-1, hop-2 (, hop-3)
        assert len(set(novel)) == len(novel)
        nodes = novel[:nb]
        n_novel_used = len(nodes)
        n_fill = 0
        if len(nodes) < nb:
            if rpos_node is None:
                rpos_node = {int(v): r for r, v in enumerate(C.ret_rrf[qi]) if v >= 0}
            used = set(nodes)
            fill = sorted((int(v) for v in nodes_of(weakest) if int(v) not in used), key=lambda v: (rpos_node.get(v, 10 ** 9), v))
            n_fill = nb - len(nodes)
            nodes = nodes + fill[:n_fill]
        assert len(nodes) == nb and len(set(nodes)) == nb
        hop_of = {}
        for h, lst in enumerate(nov[:depth]):
            for v in lst:
                hop_of[v] = h + 1
        comp = [sum(1 for v in nodes[:n_novel_used] if hop_of[v] == h) for h in (1, 2, 3)]
        patch[arm][qi] = {"kept": kept, "weakest": weakest, "nb": nb, "nodes": set(nodes), "k": len(novel), "nov": [len(x) for x in nov[:depth]],
                          "truncated": int(len(novel) > nb), "n_novel_used": n_novel_used, "n_fill": n_fill, "comp": comp,
                          "novel_pos": {v: r for r, v in enumerate(novel)}, "hop_of": hop_of, "fill_set": set(nodes[n_novel_used:])}
        assert int(sizes[list(kept)].sum()) + nb == int(sizes[list(safe50)].sum())      # matched exposure
    assert patch[H2][qi]["kept"] == patch[H3][qi]["kept"]


def served_by(arm, qi, g):
    b = int(hard[g])
    if arm == "L1":
        return b in finals["L1"][qi]
    if arm == "SAFE":
        return b in finals["SAFE"][qi]
    p = patch[arm][qi]
    return b in p["kept"] or g in p["nodes"]


ind = {arm: np.zeros(nq, bool) for arm in ARMS}
ind_any = {arm: np.zeros(nq, bool) for arm in ARMS}
for arm in ARMS:
    for qi in rowsA:
        flags = [served_by(arm, qi, int(g)) for g in gold_nodes[qi]]
        ind[arm][qi] = bool(flags) and all(flags)
        ind_any[arm][qi] = any(flags)
assert (ind["L1"][m] == ind_base[m]).all()
ind_safe_block = np.array([int(goldp[qi] <= finals["SAFE"][qi]) for qi in range(nq)], bool)
assert (ind["SAFE"][m] == ind_safe_block[m]).all()
recS = json.load(open(os.path.join(G.X.REPO, SAFE_RECORD[name]), encoding="utf-8"))
assert int(recS["population"]["nq"]) == nq and os.path.basename(recS["cache"]["file"]) == os.path.basename(C.path)
assert (ind["L1"][m] == np.asarray(recS["_ind_BASE"], bool)[m]).all(), "BASE differs from the frozen replay record"
assert (ind["SAFE"][m] == np.asarray(recS["_ind_SAFE"], bool)[m]).all(), "canonical SAFE not reproduced on DEV_A rows"
G.log("canonical SAFE reproduced on DEV_A rows against %s (SAFE %.4f, BASE %.4f)" % (os.path.basename(SAFE_RECORD[name]), ind["SAFE"][m].mean(), ind["L1"][m].mean()))


def mc(a, b, s):
    g_, l_, p_ = X.mcnemar(a[s], b[s])
    return {"gained": g_, "lost": l_, "p": p_}


# ---------------------------------------------------------------- H2_PATCH1 reproduction against the section-16 record
rec16 = json.load(open(os.path.join(OUT, "h2patch_A_%s.json" % name), encoding="utf-8"))["arms"][H2]
h2_all, h2_vs_safe = round(float(ind[H2][m].mean()), 4), mc(ind["SAFE"], ind[H2], m)
assert h2_all == rec16["ALL"] and (h2_vs_safe["gained"], h2_vs_safe["lost"]) == (rec16["ALL_vs_SAFE"]["gained"], rec16["ALL_vs_SAFE"]["lost"]), (h2_all, h2_vs_safe, rec16["ALL"], rec16["ALL_vs_SAFE"])
for h in rec16["per_hop"]:
    s = m & (hops == int(h[3:]))
    assert round(float(ind[H2][s].mean()), 4) == rec16["per_hop"][h]["ALL"], h
G.log("H2_PATCH1 reproduced: ALL %.4f (+%d/-%d vs SAFE) == section 16 record" % (h2_all, h2_vs_safe["gained"], h2_vs_safe["lost"]))

# ---------------------------------------------------------------- per-query gold accounting (evaluation only)
acc = {}
for arm in (H2, H3):
    gain = np.zeros(nq, np.int64)
    gain_h = np.zeros((nq, 3), np.int64)
    loss = np.zeros(nq, np.int64)
    for qi in rowsA:
        p = patch[arm][qi]
        for g in gold_nodes[qi]:
            g = int(g)
            sS, sP = served_by("SAFE", qi, g), served_by(arm, qi, g)
            if sP and not sS:
                gain[qi] += 1
                gain_h[qi, p["hop_of"][g] - 1] += 1
            elif sS and not sP:
                loss[qi] += 1
                assert int(hard[g]) == p["weakest"]
    acc[arm] = (gain, gain_h, loss)


def evaluate(arm):
    iv, ia = ind[arm], ind_any[arm]
    o = {"arm": arm, "ALL": round(float(iv[m].mean()), 4), "ANY": round(float(ia[m].mean()), 4), "per_hop": {}}
    if arm in (H2, H3):
        gain, gain_h, loss = acc[arm]
        P_ = [patch[arm][qi] for qi in rowsA]
        d = 2 if arm == H2 else 3
        o["exposure"] = "matched: |served nodes| equal to canonical SAFE per query (asserted)"
        o["depth"] = d
        o["novel_nodes_per_query_mean"] = round(float(np.mean([p["k"] for p in P_])), 1)
        o["novel_per_hop_mean"] = [round(float(np.mean([p["nov"][h] for p in P_])), 1) for h in range(d)]
        o["patch_size_mean"] = round(float(np.mean([p["nb"] for p in P_])), 1)
        o["patch_composition_mean (hop-1 / hop-2 / hop-3 novel / fill from removed block)"] = [round(float(np.mean([p["comp"][h] for p in P_])), 1) for h in range(3)] + [round(float(np.mean([p["n_fill"] for p in P_])), 1)]
        o["queries_truncated"] = int(sum(p["truncated"] for p in P_))
        o["queries_with_no_novel_node (patch == removed block)"] = int(sum(p["k"] == 0 for p in P_))
        o["gold_nodes_gained_vs_SAFE"] = int(gain[rowsA].sum())
        o["gold_nodes_gained_vs_SAFE_by_hop (1 / 2 / 3)"] = [int(gain_h[rowsA, h].sum()) for h in range(3)]
        o["queries_with_gold_node_gained_vs_SAFE"] = int((gain[rowsA] > 0).sum())
        o["gold_nodes_lost_with_removed_block"] = int(loss[rowsA].sum())
        o["queries_with_gold_node_lost_with_removed_block"] = int((loss[rowsA] > 0).sum())
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        o["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "ALL": round(float(iv[s].mean()), 4), "ANY": round(float(ia[s].mean()), 4)}
        if arm != "L1":
            o["per_hop"]["hop%d" % h]["vs_L1"] = mc(ind["L1"], iv, s)
        if arm in (H2, H3):
            o["per_hop"]["hop%d" % h]["vs_SAFE"] = mc(ind["SAFE"], iv, s)
            o["per_hop"]["hop%d" % h]["gold_nodes_gained"] = int(acc[arm][0][s].sum())
            o["per_hop"]["hop%d" % h]["gold_nodes_gained_by_hop (1 / 2 / 3)"] = [int(acc[arm][1][s, hh].sum()) for hh in range(3)]
            o["per_hop"]["hop%d" % h]["gold_nodes_lost_with_removed_block"] = int(acc[arm][2][s].sum())
        if arm == H3:
            o["per_hop"]["hop%d" % h]["vs_H2_PATCH1"] = mc(ind[H2], iv, s)
            o["per_hop"]["hop%d" % h]["ANY_vs_H2_PATCH1"] = mc(ind_any[H2], ia, s)
    if arm != "L1":
        o["ALL_vs_L1"] = mc(ind["L1"], iv, m)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_L1"] = {"n": int(s.sum()), **mc(ind["L1"], iv, s), "pooled_L1": round(float(ind["L1"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    if arm in (H2, H3):
        o["ALL_vs_SAFE"] = mc(ind["SAFE"], iv, m)
        o["ANY_vs_SAFE"] = mc(ind_any["SAFE"], ia, m)
        o["significant_loss_vs_SAFE"] = bool(o["ALL_vs_SAFE"]["lost"] > o["ALL_vs_SAFE"]["gained"] and o["ALL_vs_SAFE"]["p"] < 0.05)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_SAFE"] = {"n": int(s.sum()), **mc(ind["SAFE"], iv, s), "pooled_SAFE": round(float(ind["SAFE"][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    if arm == H3:
        o["ALL_vs_H2_PATCH1"] = mc(ind[H2], iv, m)
        o["ANY_vs_H2_PATCH1"] = mc(ind_any[H2], ia, m)
        o["significant_loss_vs_H2_PATCH1"] = bool(o["ALL_vs_H2_PATCH1"]["lost"] > o["ALL_vs_H2_PATCH1"]["gained"] and o["ALL_vs_H2_PATCH1"]["p"] < 0.05)
        if o["per_hop"]:
            s = m & (hops >= 2)
            o["hop2_hop3_pooled_vs_H2_PATCH1"] = {"n": int(s.sum()), **mc(ind[H2], iv, s), "pooled_H2_PATCH1": round(float(ind[H2][s].mean()), 4), "pooled": round(float(iv[s].mean()), 4)}
    return o


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "status": "POSTHOC_MECHANISM_TEST", "arm": H3, "prereg": "PREREGISTRATION_H3_PATCH1_DEV_A.json",
       "beam_module_sha256": PINNED_SHA, "head_substitutions": SUBS, "selector": {"cfg": CFG, "code": "_l1ps_router.build_cache + _l1kb_core.contexts / f6_select (unchanged)"},
       "SAFE_record": SAFE_RECORD[name], "identical_hops12_check": {"MICRO_L3_H2_rebuilt": round(float(allv_rrf[m].mean()), 4), "gained": g14, "lost": l14},
       "H2_PATCH1_reproduction_check": {"ALL": h2_all, "vs_SAFE": h2_vs_safe, "equals_section16_record": True},
       "cost_depth3": {"transitions_scored_per_query_mean (hop1 / hop2 / hop3)": [round(float(edges_h[:, i].mean()), 1) for i in range(3)],
                       "beam_wall_clock_ms_mean": round(1000 * float(secs.mean()), 1), "beam_wall_clock_ms_p95": round(1000 * float(np.percentile(secs, 95)), 1)},
       "definition": {"weakest": "sixth block of the frozen F6 competition order (SAFE's weakest admission)",
                      "novel": "pinned-beam visited nodes (hop-1, hop-2, hop-3 beams in frozen fused order, seeds excluded) whose block is not among the 49 kept blocks",
                      "patch": "first min(k, n_b) novel nodes; remaining positions = removed block's own nodes in frozen fused node order (ret_rrf position, then node id)",
                      "metric": "ALL-gold under matched unique-node exposure (every gold node served); node-level == block-level for L1/SAFE (asserted)"},
       "arms": {}}
for arm in ARMS:
    o = evaluate(arm)
    res["arms"][arm] = o
    G.log("%-9s %s" % (arm, json.dumps({k: o[k] for k in o if k not in ("per_hop", "arm")})))
    if o["per_hop"]:
        G.log("          per hop %s" % json.dumps(o["per_hop"]))

# ---------------------------------------------------------------- flips H3 vs H2 with their cause
flips = {"newly_covered_vs_H2_PATCH1": {"n": 0, "gold_nodes_entering_via_hop3_positions": 0, "gold_nodes_entering_via_hop1_or_hop2_positions": 0, "queries_needing_more_than_one_new_node": 0},
         "newly_lost_vs_H2_PATCH1": {"n": 0, "gold_nodes_that_were_fill_under_H2_PATCH1": 0, "gold_nodes_that_were_cut_hop2_positions": 0, "gold_nodes_other": 0}}
for qi in rowsA:
    p2, p3 = patch[H2][qi], patch[H3][qi]
    if ind[H3][qi] and not ind[H2][qi]:
        f = flips["newly_covered_vs_H2_PATCH1"]
        f["n"] += 1
        new = [int(g) for g in gold_nodes[qi] if served_by(H3, qi, int(g)) and not served_by(H2, qi, int(g))]
        f["gold_nodes_entering_via_hop3_positions"] += sum(1 for g in new if p3["hop_of"].get(g) == 3)
        f["gold_nodes_entering_via_hop1_or_hop2_positions"] += sum(1 for g in new if p3["hop_of"].get(g) in (1, 2))
        f["queries_needing_more_than_one_new_node"] += int(len(new) > 1)
    elif ind[H2][qi] and not ind[H3][qi]:
        f = flips["newly_lost_vs_H2_PATCH1"]
        f["n"] += 1
        lost = [int(g) for g in gold_nodes[qi] if served_by(H2, qi, int(g)) and not served_by(H3, qi, int(g))]
        f["gold_nodes_that_were_fill_under_H2_PATCH1"] += sum(1 for g in lost if g in p2["fill_set"])
        f["gold_nodes_that_were_cut_hop2_positions"] += sum(1 for g in lost if g in p2["nodes"] and g not in p2["fill_set"])
        f["gold_nodes_other"] += sum(1 for g in lost if g not in p2["nodes"])
res["attribution_flips"] = flips
G.log("flips %s" % json.dumps(flips))

# ---------------------------------------------------------------- node-level availability at depth 2 and 3 (SAFE failures)
groups = [("hop%d" % h, [qi for qi in rowsA if hops[qi] == h]) for h in (1, 2, 3)] if name.startswith("metaqa") else [("all", list(rowsA))]
diag = {}
for gname, rows_h in groups:
    dd = {}
    for arm, depth in ((H2, 2), (H3, 3)):
        on = {"SAFE_failures (a gold node not served by SAFE)": 0, "all_missing_gold_nodes_visited_by_beam (node-level repair available)": 0,
              "of_which_covered": 0, "not_covered__truncated (a missing gold node beyond the patch capacity)": 0,
              "not_covered__gold_node_dropped_with_removed_block": 0, "not_covered__other": 0,
              "missing_gold_nodes_visited_fraction_mean": [], "SAFE_failures_with_no_missing_gold_node_visited": 0,
              "missing_gold_nodes_visited_at_hop (1 / 2 / 3)": [0, 0, 0], "missing_gold_nodes_total": 0}
        loss = acc[arm][2]
        for qi in rows_h:
            if ind["SAFE"][qi] or not len(gold_nodes[qi]):
                continue
            p = patch[arm][qi]
            bl = (beams1[qi], beams2[qi], beams3[qi])[:depth]
            vis_hop = {}
            for h, lst in enumerate(bl):
                for v in lst:
                    vis_hop.setdefault(int(v), h + 1)
            on["SAFE_failures (a gold node not served by SAFE)"] += 1
            miss_n = [int(g) for g in gold_nodes[qi] if not served_by("SAFE", qi, int(g))]
            on["missing_gold_nodes_total"] += len(miss_n)
            nv = 0
            for g in miss_n:
                if g in vis_hop:
                    nv += 1
                    on["missing_gold_nodes_visited_at_hop (1 / 2 / 3)"][vis_hop[g] - 1] += 1
            on["missing_gold_nodes_visited_fraction_mean"].append(nv / max(1, len(miss_n)))
            on["SAFE_failures_with_no_missing_gold_node_visited"] += int(nv == 0)
            if nv == len(miss_n):
                on["all_missing_gold_nodes_visited_by_beam (node-level repair available)"] += 1
                if ind[arm][qi]:
                    on["of_which_covered"] += 1
                elif any(p["novel_pos"].get(g, -1) >= p["nb"] for g in miss_n):
                    on["not_covered__truncated (a missing gold node beyond the patch capacity)"] += 1
                elif loss[qi] > 0:
                    on["not_covered__gold_node_dropped_with_removed_block"] += 1
                else:
                    on["not_covered__other"] += 1
        on["missing_gold_nodes_visited_fraction_mean"] = round(float(np.mean(on["missing_gold_nodes_visited_fraction_mean"])), 3) if on["missing_gold_nodes_visited_fraction_mean"] else None
        dd["depth%d (%s)" % (depth, arm)] = on
        G.log("diag %s depth %d %s" % (gname, depth, json.dumps(on)))
    diag[gname] = dd
res["diagnostic_node_level_availability"] = diag
G.S.wj(os.path.join(OUT, "h3patch_A_%s.json" % name), res)
G.log("done")
