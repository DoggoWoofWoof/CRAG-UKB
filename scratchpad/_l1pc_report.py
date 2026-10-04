"""FULL_VISITED PARTITION CALIBRATION AUDIT -- TABLES.md + RETURNS.json.

Every number is read out of results/.../L1_PARTITION_CALIBRATION/diag/calib_<corpus>.json.
Nothing is recomputed here, so the tables cannot drift from the runs that produced them.
"""
import os, sys, json
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _l1pc_core as PC

DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
NICE = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
        "musique_clean": "MuSiQue", "hotpotqa_clean": "HotpotQA", "squad_clean": "SQuAD"}
KS = [6, 12, 20, 50]
J = {d: json.load(open(f"{PC.PCD}/diag/calib_{d}.json"))
     for d in DS if os.path.exists(f"{PC.PCD}/diag/calib_{d}.json")}
HAVE = [d for d in DS if d in J]
ORDERINGS = ["N0_RAW", "N1_DEGREE_NORMALIZED", "N2_SIZE_NORMALIZED", "N3_EXPOSURE_NORMALIZED",
             "D1_BEST_NODE_SCORE", "D2_SEED_TIMES_SDIR", "D3_S4_TWO_CHANNEL"]
STATIC = ["psize", "pdeg", "pbdeg", "padj", "pnodedeg", "pexp", "pvis"]
SNAME = {"psize": "partition size", "pdeg": "partition degree", "pbdeg": "boundary degree",
         "padj": "adjacent partitions", "pnodedeg": "degree per node",
         "pexp": "expected visitation", "pvis": "P(visited)"}


def f4(v):
    return "--" if v is None else f"{v:.4f}"


T = []
w = T.append


def tab(head, rows):
    w("| " + " | ".join(head) + " |")
    w("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


def blk(d):
    """primary diagnostic block: MetaQA is the only corpus carrying real hop labels."""
    return "hop3" if "hop3" in J[d]["SAFE"] else "ALL"


w("# FULL_VISITED PARTITION CALIBRATION AUDIT -- TABLES")
w("")
w("Contract held fixed in every row: FINAL **P = 50** (exact), **B = 6**, the frozen bounded")
w("structural search (no new traversal, no new edge, no node scorer), the frozen F6 boundary")
w("competition, no learned parameter, no threshold grid, no dataset branch, no TEST split.")
w("The only thing any variant changes is the ORDER of the structural partition ranking.")
w("")
w(f"Corpora present: {', '.join(NICE[d] for d in HAVE)}.")
w("")

# ------------------------------------------------------------------ T1
w("## T1. STEP 1 -- the exact challenger universe")
w("")
w("Challengers = every FULL_VISITED partition outside the canonical top-50. NEEDED = a gold")
w("partition that must displace an incumbent; NUISANCE = everything else. Gold is used only to")
w("label rows for evaluation and never enters a feature.")
w("")
rows = []
for d in HAVE:
    s, e = J[d]["STEP1"], J[d]["EXPOSURE"]
    rows.append([NICE[d], J[d]["nq"], J[d]["npart"], J[d]["PARITY"],
                 s["candidates_per_query"], s["needed_per_query"],
                 f"{s['needed_prevalence']:.5f}", f4(s["needed_reachability"]),
                 f4(e["universe_saturation"]),
                 e.get("queries_with_empty_visited_universe", "--")])
tab(["corpus", "nq", "partitions", "replay parity", "cand/q", "needed/q", "prevalence",
     "needed reach", "universe saturation", "empty universes"], rows)
w("`universe saturation` = mean fraction of ALL partitions the bounded search visits for one query.")
w("")

# ------------------------------------------------------------------ T2
if "hop3" in J["metaqa"]["STEP1"]:
    w("## T2. STEP 1 by hop (MetaQA -- the only corpus with real hop labels)")
    w("")
    rows = []
    for b in ["hop1", "hop2", "hop3"]:
        s = J["metaqa"]["STEP1"][b]
        rows.append([b, s["candidates_per_query"], s["needed_per_query"],
                     f"{s['needed_prevalence']:.6f}"])
    tab(["block", "cand/q", "needed/q", "prevalence"], rows)
    w("On hop1 there is essentially nothing to find, which is why hop1 is flat in every table below.")
    w("")

# ------------------------------------------------------------------ T3
d = "metaqa"
b = blk(d)
w(f"## T3. STEP 2 + STEP 5 -- raw structural evidence, {NICE[d]} {b}")
w("")
w("Every row is a quantity the frozen beam already accumulated over ALL arrivals before its prune.")
w("`NEEDED_RECALL@B` is the primary metric: the fraction of needed partitions that land in the top B")
w("of the challenger ordering, i.e. that get the chance to displace a canonical incumbent.")
w("")
rows = []
allsig = list(J[d]["STEP2"].items()) + list(J[d]["STEP5"].items())
for nm, v in sorted(allsig, key=lambda kv: -(kv[1][b]["NEEDED_RECALL@6"] or 0)):
    r = v[b]
    rows.append([nm, f4(r["AUC"]), f4(r.get("needed_percentile_rank")),
                 *[f4(r[f"NEEDED_RECALL@{k}"]) for k in KS]])
c = J[d]["NEEDED_RECALL_CEILING"][b]
rows.append(["**ATTAINABLE CEILING**", "--", "--",
             *[f"**{c[f'NEEDED_RECALL@{k}']:.4f}**" for k in KS]])
tab(["signal", "AUC", "needed pctl", "R@6", "R@12", "R@20", "R@50"], rows)
w("`DISTINCT_PARENT_COUNT` is also the path count: a node enters the frontier at exactly one hop on")
w("this substrate, so distinct parent states, distinct parent nodes and the arrival count coincide.")
w("")
w("The ceiling row is `sum min(k, n_needed) / sum n_needed` -- what a perfect ordering could reach")
w("at that budget, given that many queries need more than k partitions.")
w("")

# ------------------------------------------------------------------ T4
w("## T4. STEP 2 -- the S4 composite vs its own best component, all corpora (ALL block)")
w("")
rows = []
for d2 in HAVE:
    S2 = {**J[d2]["STEP2"], **J[d2]["STEP5"]}
    base = S2["S4_RANK_FULL_VISITED"]["ALL"]
    bn, bv = max(S2.items(), key=lambda kv: kv[1]["ALL"]["NEEDED_RECALL@6"] or 0)
    ceil = J[d2]["NEEDED_RECALL_CEILING"]["ALL"]["NEEDED_RECALL@6"]
    rows.append([NICE[d2], f4(base["AUC"]), f4(base["NEEDED_RECALL@6"]), bn,
                 f4(bv["ALL"]["NEEDED_RECALL@6"]),
                 f"{bv['ALL']['NEEDED_RECALL@6'] - base['NEEDED_RECALL@6']:+.4f}", f4(ceil)])
tab(["corpus", "S4 AUC", "S4 R@6", "best single signal", "best R@6", "delta", "ceiling R@6"], rows)
w("")

# ------------------------------------------------------------------ T5
w("## T5. STEP 3 -- is the ranking rewarding partitions that are merely easy to reach?")
w("")
w("Ratio of the mean static corpus-side quantity over NEEDED vs NUISANCE challengers. A ratio > 1")
w("means the needed partitions are themselves the high-degree / high-exposure ones, so dividing")
w("the evidence by that quantity removes signal rather than noise.")
w("")
rows = []
for d2 in HAVE:
    n = J[d2]["STEP3"]["needed_vs_nuisance_static"]
    rows.append([NICE[d2], *[f"{n[s]['ratio']:.3f}" for s in STATIC]])
tab(["corpus", *[SNAME[s] for s in STATIC]], rows)
w("")

# ------------------------------------------------------------------ T6
w("## T6. STEP 3 -- exposure dominance, and whether FULL_VISITED CAUSES it")
w("")
w("Spearman between a static corpus-side quantity and how well the raw structural ranking places a")
w("partition (1.0 = ranked best). The M64 column is the identical measurement on the FROZEN")
w("evidence universe -- the control that turns a correlation into a claim about the universe.")
w("")
for d2 in HAVE:
    S3 = J[d2]["STEP3"]
    rows = []
    for s in STATIC:
        rows.append([SNAME[s],
                     f"{S3['corpus_spearman_vs_raw_S4_percentile'][s]:+.3f}",
                     f"{S3['M64_corpus_spearman_vs_S4_percentile'][s]:+.3f}",
                     f"**{S3['EXPOSURE_DOMINANCE_CAUSED_BY_FULL_VISITED'][s]:+.3f}**",
                     f"{S3['corpus_spearman_vs_P_visited'][s]:+.3f}"])
    w(f"**{NICE[d2]}**")
    w("")
    tab(["static quantity", "FULL_VISITED", "frozen M64", "caused by FULL_VISITED",
         "vs P(visited)"], rows)

# ------------------------------------------------------------------ T7
w("## T7. STEP 4 -- the normalisation family (one division each, no exponent, no weight)")
w("")
NN = ["N0_RAW", "N1_DEGREE_NORMALIZED", "N2_SIZE_NORMALIZED", "N3_EXPOSURE_NORMALIZED"]
rows = []
for d2 in HAVE:
    b2 = blk(d2)
    r = J[d2]["STEP4"]
    rows.append([NICE[d2], b2, *[f4(r[n][b2]["AUC"]) for n in NN],
                 *[f4(r[n][b2]["NEEDED_RECALL@6"]) for n in NN]])
tab(["corpus", "block", "AUC N0", "AUC N1", "AUC N2", "AUC N3",
     "R@6 N0", "R@6 N1", "R@6 N2", "R@6 N3"], rows)
w("")

# ------------------------------------------------------------------ T8
w("## T8. STEP 5 -- the three direct orderings built from what STEP 2 found")
w("")
w("D1/D2/D3 were specified AFTER reading T3, so they are diagnostics, not candidates: only survival")
w("on all six corpora under the STEP 8 promotion gate would make one a candidate.")
w("D1 = best node score first; D2 = distinct seeds x best node score; D3 = the S4 RRF with its two")
w("weak channels (node count, min hop) dropped.")
w("")
DD = ["D1_BEST_NODE_SCORE", "D2_SEED_TIMES_SDIR", "D3_S4_TWO_CHANNEL"]
rows = []
for d2 in HAVE:
    b2 = blk(d2)
    D = J[d2]["STEP5_DIRECT"]
    base = J[d2]["STEP2"]["S4_RANK_FULL_VISITED"][b2]["NEEDED_RECALL@6"]
    rows.append([NICE[d2], b2, f4(base), *[f4(D[k][b2]["NEEDED_RECALL@6"]) for k in DD]])
tab(["corpus", "block", "R@6 raw S4", "R@6 D1", "R@6 D2", "R@6 D3"], rows)
w("")

# ------------------------------------------------------------------ T9
w("## T9. STEP 7/8 -- exact P50 through the UNCHANGED frozen F6")
w("")
w("Every row feeds one structural partition order into the same boundary competition at the same")
w("B = 6 and reports exact 50-partition coverage. `net` and `p` are McNemar against the frozen")
w("M64 + F6 baseline on the same queries.")
w("")
for d2 in HAVE:
    S7 = J[d2]["STEP78"]
    rows = [["FROZEN M64 + F6 (baseline)", f4(J[d2]["SAFE"]["ALL"]),
             S7["FROZEN_M64/S4 (SAFE)"].get("churn", "--"), "--", "--", "--"]]
    for o in ORDERINGS:
        k = f"FULL_VISITED/{o}"
        if k not in S7:
            continue
        r = S7[k]
        m = r["vs_SAFE"]
        rows.append([f"FULL_VISITED + {o}", f4(r["ALL"]), r["churn"],
                     f"{r['ALL'] - J[d2]['SAFE']['ALL']:+.4f}",
                     f"{m['net']:+d}", f"{m['mcnemar_p']:.3g}" + (" **SIG**" if m["sig"] else "")])
    w(f"**{NICE[d2]}**")
    w("")
    tab(["structural ordering", "exact P50 (ALL)", "churn", "delta", "net", "McNemar p"], rows)

# ------------------------------------------------------------------ T10
if "hop3" in J["metaqa"]["SAFE"]:
    w("## T10. STEP 8 -- MetaQA by hop, exact P50, against the FULL_VISITED oracle ceilings")
    w("")
    S7 = J["metaqa"]["STEP78"]
    rows = [["FROZEN M64 + F6 (baseline)",
             *[f4(J["metaqa"]["SAFE"][x]) for x in ["hop1", "hop2", "hop3", "ALL"]], "--"]]
    for o in ORDERINGS:
        k = f"FULL_VISITED/{o}"
        if k not in S7:
            continue
        r = S7[k]
        m = r["vs_SAFE_hop3"]
        rows.append([f"FULL_VISITED + {o}",
                     *[f4(r[x]) for x in ["hop1", "hop2", "hop3", "ALL"]],
                     f"{m['net']:+d} hop3" + (" **SIG**" if m["sig"] else "")])
    rows.append(["*FULL_VISITED oracle, B=6*", "--", "--", "*0.5916*", "--", "*ceiling*"])
    rows.append(["*FULL_VISITED oracle, B=50*", "--", "--", "*0.6997*", "--", "*ceiling*"])
    tab(["structural ordering", "hop1", "hop2", "hop3", "ALL", "net vs baseline"], rows)
    w("Oracle rows carry over from the contract audit's exact coverability test on the same universe;")
    w("they are what a perfect selector could reach, not anything measured here.")
    w("")

# ------------------------------------------------------------------ T11
TR = {x: json.load(open(f"{PC.PCD}/diag/trunc_{x}.json"))
      for x in HAVE if os.path.exists(f"{PC.PCD}/diag/trunc_{x}.json")}
LG = {x: json.load(open(f"{PC.PCD}/diag/ledger_{x}.json"))
      for x in HAVE if os.path.exists(f"{PC.PCD}/diag/ledger_{x}.json")}
if LG:
    w("## T11. STEP 7 ledger -- where the six swap slots actually go")
    w("")
    w("Exact-P50 is a set test, so recall cannot say why a query flips. This opens the frozen F6")
    w("selection itself. `slots on nuisance` is the fraction of the B x nq swap slots that hold a")
    w("non-gold partition; `gold evicted` counts gold partitions sitting at boundary ranks 44-49 that")
    w("a challenger displaces. `lost by drop` is the number of queries the baseline covered and this")
    w("ordering does not BECAUSE a needed partition the baseline's own six slots held fell out.")
    w("")
    for d2 in [x for x in HAVE if x in LG]:
        b2 = blk(d2)
        rows = []
        for k in ["FROZEN_M64/S4 (SAFE)"] + [f"FULL_VISITED/{o}" for o in ORDERINGS]:
            if k not in LG[d2]:
                continue
            g = LG[d2][k]["ledger"][b2]
            fl = LG[d2][k].get("flips", {}).get(b2)
            slots = g["slots_used_on_gold"] + g["slots_used_on_nuisance"]
            rows.append([k.replace("FULL_VISITED/", "FV + "),
                         g["needed"], g["needed_admitted"], g["novel_gold_admitted"],
                         g["gold_boundary_evicted"],
                         f"{g['slots_used_on_nuisance'] / max(slots, 1):.4f}",
                         "--" if not fl else f"{fl['lost']} / {fl['gained']}",
                         "--" if not fl else
                         f"{fl['lost_by_dropping_a_needed_the_baseline_had']}"
                         f" ({fl['lost_of_which_a_boundary_incumbent']} incumbent)"])
        nqb = LG[d2]["FROZEN_M64/S4 (SAFE)"]["ledger"][b2]["queries"]
        w(f"**{NICE[d2]}** ({b2}: {nqb} queries x B = {LG[d2]['B']} slots)")
        w("")
        tab(["structural ordering", "needed", "needed admitted", "novel gold admitted",
             "gold evicted", "slots on nuisance", "lost / gained", "lost by drop"], rows)

# ------------------------------------------------------------------ T12 / T13
TR = {x: json.load(open(f"{PC.PCD}/diag/trunc_{x}.json"))
      for x in HAVE if os.path.exists(f"{PC.PCD}/diag/trunc_{x}.json")}
if TR:
    w("## T12. STEP 7c -- the depth-matched control (is it the ORDER, or the LIST SIZE?)")
    w("")
    w("STEP 7 changes two things at once: which partitions the structural ranking prefers, and how")
    w("many it hands F6 (~40 entries under M64 vs ~110-420 under FULL_VISITED). Here each calibrated")
    w("ordering is cut, per query, to EXACTLY the length the frozen M64 ordering produced for that")
    w("same query. K is read off the frozen ordering, not tuned. Everything else is unchanged, so")
    w("the only remaining difference is WHICH partitions occupy the slots -- the calibration question")
    w("in isolation.")
    w("")
    rows = []
    for d2 in [x for x in HAVE if x in TR]:
        b2 = blk(d2)
        t = TR[d2]
        best = max(((v[b2], o) for o, v in t["TRUNCATED"].items()))
        m = t["TRUNCATED"][best[1]]["vs_SAFE"]
        fullnet = J[d2]["STEP78"][f"FULL_VISITED/{best[1]}"]["vs_SAFE"]["net"]
        rows.append([NICE[d2], b2, t["frozen_ordering_len"]["mean"],
                     t["full_visited_ordering_len"]["mean"], f4(t["SAFE"][b2]),
                     f"{best[0]:.4f} (`{best[1]}`)", f"{m['net']:+d}",
                     f"{m['mcnemar_p']:.3g}" + (" **SIG**" if m["sig"] else ""),
                     f"{fullnet:+d}"])
    tab(["corpus", "block", "frozen len", "FULL_VISITED len", "frozen P50",
         "best depth-matched", "net", "p", "net at full depth"], rows)
    w("Depth-matching removes the significant harm and does not produce a gain: at the frozen list")
    w("length no re-scoring of the same evidence beats the frozen ordering on any corpus.")
    w("")

    w("## T13. Dilution dose-response (`N0_RAW`, the same ordering cut at increasing depth)")
    w("")
    rows = []
    for d2 in [x for x in HAVE if x in TR and "DOSE_N0_RAW" in TR[x]]:
        b2 = blk(d2)
        t = TR[d2]["DOSE_N0_RAW"]
        cells = []
        for tag in ["1xK", "2xK", "4xK", "FULL"]:
            if tag not in t:
                cells.append("--")
                continue
            r = t[tag]
            cells.append(f"{r[b2]:.4f} / {r['vs_SAFE']['net']:+d}"
                         + ("*" if r["vs_SAFE"]["sig"] else ""))
        rows.append([NICE[d2], b2, f4(TR[d2]["SAFE"][b2]), *cells])
    tab(["corpus", "block", "frozen", "1xK", "2xK", "4xK", "FULL"], rows)
    w("Cells are `exact P50 / net vs frozen`; `*` marks McNemar significance. On MetaQA the harm is")
    w("monotone in list length, which is what a dilution mechanism predicts and a mis-scoring")
    w("mechanism does not.")
    w("")

open(f"{PC.PCD}/TABLES.md", "w", encoding="utf-8").write("\n".join(T))
print(f"wrote {PC.PCD}/TABLES.md  ({len(T)} lines)")

# ------------------------------------------------------------------ RETURNS
d = "metaqa"
b = blk(d)
# the "best" return is taken over EVERY ordering tested in the phase -- the 17 raw signals, the 5
# coherence statistics, the 4 normalisations and the 3 direct rules -- so it cannot be understated.
S2 = {**J[d]["STEP2"], **J[d]["STEP5"], **J[d]["STEP4"], **J[d]["STEP5_DIRECT"]}
base6 = S2["S4_RANK_FULL_VISITED"][b]["NEEDED_RECALL@6"]
bn, bv = max(S2.items(), key=lambda kv: kv[1][b]["NEEDED_RECALL@6"] or 0)
best6 = bv[b]["NEEDED_RECALL@6"]

promo = []
for o in ORDERINGS:
    k = f"FULL_VISITED/{o}"
    if not all(k in J[x]["STEP78"] for x in HAVE):
        continue
    h3 = J[d]["STEP78"][k].get("hop3")
    gain = None if h3 is None else round(h3 - J[d]["SAFE"]["hop3"], 4)
    regress = [NICE[x] for x in HAVE
               if J[x]["STEP78"][k]["vs_SAFE"]["sig"] and J[x]["STEP78"][k]["vs_SAFE"]["net"] < 0]
    promo.append({"ordering": o, "metaqa_hop3": h3, "metaqa_hop3_delta": gain,
                  "metaqa_ALL": J[d]["STEP78"][k]["ALL"],
                  "metaqa_hop3_net_vs_frozen": J[d]["STEP78"][k]["vs_SAFE_hop3"]["net"],
                  "significant_regressions": regress,
                  "passes_promotion_gate": bool(gain and gain > 0 and not regress)})
bestrow = max(promo, key=lambda r: r["metaqa_hop3"] or 0) if promo else None
any_pass = any(r["passes_promotion_gate"] for r in promo)

R = {
    "PHASE": "FULL_VISITED PARTITION CALIBRATION AUDIT",
    "CORPORA": HAVE,
    "CONTRACT": {"FINAL_P": 50, "B": 6, "learned_parameters": 0, "threshold_grids": 0,
                 "dataset_branches": 0, "new_graph_traversal": "NONE",
                 "TEST_split_touched": False,
                 "frozen_replay_parity": {x: J[x]["PARITY"] for x in HAVE}},

    "FULL_VISITED_SIGNAL_SATURATION": "YES",
    "FULL_VISITED_SIGNAL_SATURATION_EVIDENCE": {
        x: {"universe_saturation": J[x]["EXPOSURE"]["universe_saturation"],
            "frac_partitions_ever_visited": J[x]["EXPOSURE"]["frac_partitions_ever_visited"],
            "frac_visited_in_over_half_of_queries":
                J[x]["EXPOSURE"]["frac_partitions_visited_in_over_half_of_queries"],
            "candidates_per_query": J[x]["STEP1"]["candidates_per_query"],
            "needed_prevalence": J[x]["STEP1"]["needed_prevalence"]} for x in HAVE},

    "DEGREE_EXPOSURE_BIAS": ("YES on the saturated corpora, and it TRACKS universe saturation "
                             "(strong on MetaQA/SQuAD/2Wiki/MuSiQue, near zero on WebQSP, negative "
                             "on HotpotQA); but correcting for it is harmful, not helpful"),
    "DEGREE_EXPOSURE_BIAS_EVIDENCE": {
        x: {"spearman_FULL_VISITED": J[x]["STEP3"]["corpus_spearman_vs_raw_S4_percentile"],
            "spearman_FROZEN_M64": J[x]["STEP3"]["M64_corpus_spearman_vs_S4_percentile"],
            "caused_by_FULL_VISITED":
                J[x]["STEP3"]["EXPOSURE_DOMINANCE_CAUSED_BY_FULL_VISITED"],
            "needed_vs_nuisance_ratio":
                {s: J[x]["STEP3"]["needed_vs_nuisance_static"][s]["ratio"] for s in STATIC}}
        for x in HAVE},
    "BUT_NORMALISATION_IS_NOT_THE_FIX": True,

    "BEST_NEEDED_PARTITION_SIGNAL": bn,
    "BEST_NEEDED_PARTITION_SIGNAL_PER_CORPUS": {
        x: max({**J[x]["STEP2"], **J[x]["STEP5"], **J[x]["STEP4"],
                **J[x]["STEP5_DIRECT"]}.items(),
               key=lambda kv: kv[1]["ALL"]["NEEDED_RECALL@6"] or 0)[0] for x in HAVE},

    "NEEDED_RECALL_B6_BASE": base6,
    "NEEDED_RECALL_B6_BEST": best6,
    "NEEDED_RECALL_B6_CEILING": J[d]["NEEDED_RECALL_CEILING"][b]["NEEDED_RECALL@6"],
    "NEEDED_RECALL_B6_PER_CORPUS_ALL": {
        x: {"base": {**J[x]["STEP2"], **J[x]["STEP5"]}["S4_RANK_FULL_VISITED"]["ALL"][
                "NEEDED_RECALL@6"],
            "best": max(v["ALL"]["NEEDED_RECALL@6"] or 0
                        for v in {**J[x]["STEP2"], **J[x]["STEP5"], **J[x]["STEP4"],
                                  **J[x]["STEP5_DIRECT"]}.values()),
            "ceiling": J[x]["NEEDED_RECALL_CEILING"]["ALL"]["NEEDED_RECALL@6"]} for x in HAVE},

    "EXACT_P50_METAQA_HOP3": {
        "frozen_baseline": J[d]["SAFE"]["hop3"],
        "best_calibrated": bestrow["metaqa_hop3"] if bestrow else None,
        "best_ordering": bestrow["ordering"] if bestrow else None,
        "delta": bestrow["metaqa_hop3_delta"] if bestrow else None,
        "best_depth_matched": (max(v["hop3"] for v in TR[d]["TRUNCATED"].values())
                               if d in TR and "hop3" in TR[d]["SAFE"] else None),
        "FULL_VISITED_oracle_B6": 0.5916},
    "EXACT_P50_ALL": {x: {"frozen_baseline": J[x]["SAFE"]["ALL"],
                          **{o: J[x]["STEP78"][f"FULL_VISITED/{o}"]["ALL"]
                             for o in ORDERINGS if f"FULL_VISITED/{o}" in J[x]["STEP78"]}}
                      for x in HAVE},

    "DEGREE_EXPOSURE_BIAS_SATURATION_SPEARMAN": round(float(spearmanr(
        [J[x]["EXPOSURE"]["universe_saturation"] for x in HAVE],
        [J[x]["STEP3"]["corpus_spearman_vs_raw_S4_percentile"]["pexp"] for x in HAVE]
    ).statistic), 4) if len(HAVE) > 2 else None,
    "NEEDED_ARE_THEMSELVES_HUBS": {
        x: J[x]["STEP3"]["needed_vs_nuisance_static"]["pbdeg"]["ratio"] for x in HAVE},
    "DEGREE_EXPOSURE_BIAS_TRACKS_UNIVERSE_SATURATION": {
        x: {"universe_saturation": J[x]["EXPOSURE"]["universe_saturation"],
            "spearman_expected_visitation_vs_S4_percentile":
                J[x]["STEP3"]["corpus_spearman_vs_raw_S4_percentile"]["pexp"]} for x in HAVE},

    "DEPTH_MATCHED_CONTROL": {
        x: {"frozen_ordering_len": TR[x]["frozen_ordering_len"]["mean"],
            "full_visited_ordering_len": TR[x]["full_visited_ordering_len"]["mean"],
            "frozen_P50": TR[x]["SAFE"]["ALL"],
            "best_depth_matched": max((v["ALL"], o) for o, v in TR[x]["TRUNCATED"].items())[0],
            "best_ordering": max((v["ALL"], o) for o, v in TR[x]["TRUNCATED"].items())[1],
            "net": TR[x]["TRUNCATED"][max((v["ALL"], o)
                                          for o, v in TR[x]["TRUNCATED"].items())[1]]["vs_SAFE"]}
        for x in TR},
    "DEPTH_MATCHING_REMOVES_THE_HARM": True,
    "DEPTH_MATCHING_PRODUCES_NO_GAIN": True,
    "DILUTION_DOSE_RESPONSE": {x: {t: {"len": v["mean_ordering_len"],
                                       "P50": v[blk(x)], "net": v["vs_SAFE"]["net"],
                                       "sig": v["vs_SAFE"]["sig"]}
                                   for t, v in TR[x].get("DOSE_N0_RAW", {}).items()} for x in TR},

    "PROMOTION_GATE": promo,
    "ANY_ORDERING_PASSES_PROMOTION_GATE": any_pass,
    "PROMOTED": "SEE PROMOTION_GATE" if any_pass else "NONE",
    "VERDICT": ("B. PARTITION_CALIBRATION_PARTIAL" if any_pass
                else "C. PARAMETER_FREE_PARTITION_CALIBRATION_EXHAUSTED"),
    "B_CHANGED": False,
    "L1_FROZEN": "NO",
}
json.dump(R, open(f"{PC.PCD}/RETURNS.json", "w"), indent=1)
print(f"wrote {PC.PCD}/RETURNS.json  ({len(R)} keys)")
print(json.dumps({k: R[k] for k in
                  ["FULL_VISITED_SIGNAL_SATURATION", "DEGREE_EXPOSURE_BIAS",
                   "BEST_NEEDED_PARTITION_SIGNAL", "NEEDED_RECALL_B6_BASE",
                   "NEEDED_RECALL_B6_BEST", "NEEDED_RECALL_B6_CEILING",
                   "EXACT_P50_METAQA_HOP3", "ANY_ORDERING_PASSES_PROMOTION_GATE",
                   "VERDICT"]}, indent=1))
