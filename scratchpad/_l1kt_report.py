"""Assemble TABLES.md and the numeric half of RETURNS.json for the L1 CAPACITY + TRANSFORMATION
phase from the diag JSONs.  Verdict strings are written by hand into FINAL_REPORT.md; everything
numeric here is derived, never retyped.

  python scratchpad/_l1kt_report.py
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
D = f"{KTD}/diag"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
FAMS = ["F0_TRANSLATION", "F1_HOUSEHOLDER", "F2_MIN_ROTATION", "F3_RANK1_TRANSPORT"]
SH = {"F0_TRANSLATION": "F0_TRANS", "F1_HOUSEHOLDER": "F1_HOUSE",
      "F2_MIN_ROTATION": "F2_ROT", "F3_RANK1_TRANSPORT": "F3_RANK1"}
ZS = ["q_raw", "r_q"]
VARS = ["V1_SINGLE", "V2_CHAIN", "V3_MAXVIEW", "V4_SRC_EXIT"]
DEL = ["K0+0", "K0+4", "K0+8", "K0+16", "K0+32"]
L = []
w = L.append


def jf(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def tbl(head, rows, widths=None):
    widths = widths or [max(len(str(head[i])), *(len(str(r[i])) for r in rows)) + 2
                        for i in range(len(head))] if rows else [len(h) + 2 for h in head]
    fmt = lambda r: "".join(str(x).rjust(widths[i]) for i, x in enumerate(r))
    w("```")
    w(fmt(head))
    w("-" * sum(widths))
    for r in rows:
        w(fmt(r))
    w("```")


PA = {d: jf(f"{D}/parta_{d}.json") for d in DS}
CL = jf(f"{D}/closure.json")
GT = jf(f"{D}/gates.json")
TF = {d: jf(f"{D}/tf_{d}.json") for d in DS}
PB = {d: jf(f"{D}/pb_{d}.json") for d in DS}
SW = {d: jf(f"{D}/swap_{d}.json") for d in DS}
R = {}

w("# L1 CAPACITY + TRANSFORMATION EVICTION -- TABLES")
w("")
w("Every number below is read straight out of `diag/`; nothing is retyped by hand.")
w("")

# ------------------------------------------------------------------ PART A
w("## PART A1 -- candidate oracle vs candidate depth (append-only, nothing evicted)")
w("")
rows, cur = [], {}
for d in DS:
    if not PA[d]:
        continue
    for nm, o in PA[d]["oracle"].items():
        if nm != "ALL" and d != "metaqa":
            continue
        rows.append([f"{d}/{nm}", f"{PA[d]['pool_size']['K0+0']:.1f}"]
                    + [f"{o[x]:.4f}" for x in DEL] + [f"{o['K0+32'] - o['K0+0']:+.4f}"])
        cur[f"{d}/{nm}"] = {x: o[x] for x in DEL}
tbl(["corpus/slice", "K(q)"] + DEL + ["delta"], rows)
R["K_CAPACITY_CURVE"] = cur
w("")

w("## PART A2 -- do the ACTUAL selectors convert it?  exact P50, net vs frozen (K0+0 / F6)")
w("")
rows = []
for d in DS:
    if not PA[d]:
        continue
    for nm, p in PA[d]["p50"].items():
        if nm != "ALL" and d != "metaqa":
            continue
        for s in ("F6", "G4"):
            r0 = p[f"K0+0/{s}"]
            cells = []
            for x in DEL:
                c = p[f"{x}/{s}"]
                cells.append(f"{c['net']:+d}{'*' if c['sig'] else ''}")
            rows.append([f"{d}/{nm}", s] + cells
                        + [f"{p['K0+32/' + s]['net'] - r0['net']:+d}"])
tbl(["corpus/slice", "sel"] + DEL + ["capacity-only"], rows)
w("")
w("`capacity-only` holds the SELECTOR fixed and moves only the depth, so it isolates what the")
w("extra candidates are worth.  `*` = significant (McNemar, vs the frozen system).")
w("")

if CL:
    w("## PART A -- why F6 converts exactly zero (algebraic closure, all six corpora)")
    w("")
    rows = [[d, CL[d]["appended_candidates"], CL[d]["appended_with_struct_or_ret_evidence"],
             CL[d]["appended_that_could_enter_top6"], f"{CL[d]['max_appended_F6_score']:.6f}",
             f"{CL[d]['min_6th_best_pool_score']:.6f}"] for d in DS if CL.get(d)]
    tbl(["corpus", "appended", "with spos|rpos", "could enter top6",
         "max appended score", "min 6th pool score"], rows)
    w("")
    w("The SAFE pool is `bnd ++ chal` and `chal` is DEFINED as everything carrying struct or ret")
    w("evidence outside base50, so an appendable partition has neither and scores at most")
    w("`1/(K0+cpos) <= 1/110 = 0.009091`, while all six `bnd` members score at least")
    w("`1/(60+49) = 0.009174`.  The bound is tight and holds exactly on every corpus.")
    w("")

# ------------------------------------------------------------------ operators
if GT:
    w("## STEP 1 / 2 / 6 -- operator gates on real replayed edges")
    w("")
    rows = [[f, f"{GT['G1_max_abs_T_u_minus_v'][f]:.2e}",
             f"{GT['G2_norm_ratio_mean_std'][f][0]:.6f} +- {GT['G2_norm_ratio_mean_std'][f][1]:.6f}",
             f"{GT['G5_chain_equals_manual_max_err'][f]:.2e}",
             f"{max(GT['G6_scalar_vs_vector_max_err'][f].values()):.2e}"] for f in FAMS]
    tbl(["family", "max|T(u)-v|", "||Tz||/||z|| on random unit z",
         "chain==manual", "scalar==vector"], rows)
    w("")
    w(f"F2 identity on span(u,v)-perp {GT['G3_fixes_complement']['F2_MIN_ROTATION']:.2e}; "
      f"F1 identity on (u-v)-perp {GT['G3_fixes_complement']['F1_HOUSEHOLDER']:.2e}; "
      f"F1 involution {GT['G4_F1_involution_max_err']:.2e}; "
      f"{GT['n_edges']} real edges.")
    w("")

# ------------------------------------------------------------------ STEP 3 / 5
w("## STEP 3 + STEP 5 -- edge-level AUC (missing-required target vs non-required target)")
w("")
rows = []
for d in DS:
    t = TF.get(d)
    if not t:
        continue
    for z in ZS:
        b = t["STEP3_single_hop"][f"{z}/NO_TRANSFORM"]["AUC"]
        s1 = [t["STEP3_single_hop"][f"{z}/{f}"]["AUC"] for f in FAMS]
        s2 = [t["STEP5_composition_by_hop"][f"{z}/{f}"]["AUC"] for f in FAMS]
        rows.append([d, z, f"{b:.4f}"] + [f"{x:.4f}" for x in s1] + [f"{x:.4f}" for x in s2])
tbl(["corpus", "state", "NO_TF"] + [SH[f] + " 1hop" for f in FAMS]
    + [SH[f] + " chain" for f in FAMS], rows)
w("")
w("`NO_TF` is the identity control -- `cos(z0, x_v)` on the SAME edge population.  Any cell that")
w("does not beat it is a transformation that destroys information rather than adding it.")
w("")

w("## STEP 3 -- marginal target-partition rank of missing-required (percentile, lower better)")
w("")
rows = []
for d in DS:
    t, p = TF.get(d), PB.get(d)
    if not t:
        continue
    c = p["STEP3_partition_rank_NO_TRANSFORM"] if p else None
    for z in ZS:
        g = lambda f, m: t["STEP3_partition_rank"].get(f"{z}/{f}/{m}")
        rows.append([d, z, f"{c:.4f}" if (c is not None and z == "q_raw") else ""]
                    + [f"{g(f, 'single'):.4f}" if g(f, "single") is not None else "n/a" for f in FAMS]
                    + [f"{g(f, 'chain'):.4f}" if g(f, "chain") is not None else "n/a" for f in FAMS])
tbl(["corpus", "state", "NO_TF"] + [SH[f] + " 1hop" for f in FAMS]
    + [SH[f] + " chain" for f in FAMS], rows)
w("")

w("## STEP 6 -- composition stability (chain, q_raw)")
w("")
rows = []
for d in DS:
    t = TF.get(d)
    if not t:
        continue
    for f in FAMS:
        v = t["STEP6_stability"][f"q_raw/{f}"]
        g = lambda h, k: v[f"hop{h}"][k]
        fm = lambda h, k: (f"{g(h, k):.3f}" if g(h, k) is not None else "n/a")
        rows.append([d, f] + [fm(h, "norm_ratio") for h in (1, 2, 3)]
                    + [fm(h, "cos_drift_vs_z0") for h in (1, 2, 3)])
tbl(["corpus", "family", "|z1|/|z0|", "|z2|/|z0|", "|z3|/|z0|",
     "cos(z1,z0)", "cos(z2,z0)", "cos(z3,z0)"], rows)
w("")

# ------------------------------------------------------------------ STEP 7
w("## STEP 7 -- MISSING_REQUIRED_ADMISSION_RECALL inside FULL_VISITED")
w("")
KEYS = ["FROZEN_SAFE", "SRC_ASSIGN", "OFFSET_RAW", "V0_NOTRANSFORM"]
best = {}
for d in DS:
    p = PB.get(d)
    if not p:
        continue
    rec = p["STEP7_missing_required_admission_recall"]
    cand = [k for k in rec if "/" in k]
    bb = max(cand, key=lambda k: (rec[k]["@6"] or 0)) if cand else None
    best[d] = bb
    rows = [[k] + [f"{rec[k][f'@{n}']:.4f}" for n in (6, 12, 20, 50)] for k in KEYS if k in rec]
    if bb:
        rows.append([f"BEST TF: {bb}"] + [f"{rec[bb][f'@{n}']:.4f}" for n in (6, 12, 20, 50)])
        f0 = f"q_raw/F0_TRANSLATION/V2_CHAIN"
        if f0 in rec:
            rows.append([f"TRANSLATION: {f0}"] + [f"{rec[f0][f'@{n}']:.4f}" for n in (6, 12, 20, 50)])
    w(f"**{d}** ({p['STEP7_n_queries_with_missing_required']} queries with a missing-required "
      f"partition)")
    w("")
    tbl(["ranker over FULL_VISITED", "@6", "@12", "@20", "@50"], rows)
    w("")
R["STEP7_BEST_PER_CORPUS"] = best

# ------------------------------------------------------------------ STEP 8
w("## STEP 8 -- multi-state coverage: the four state views, not collapsed to one scalar")
w("")
w("`T_SINGLE` = V1_SINGLE, one transformation applied to q0 with no composition.  `T_ASSIGN` =")
w("V2_CHAIN, every partition taking its OWN best transported state.  `T_MAX_VIEW` = V3_MAXVIEW, the")
w("union view that also contains the untransformed q0.  `V4_SRC_EXIT` restricts the view to edges")
w("leaving the protected core.  Cells are STEP-7 recall@6 / @50 at each corpus's own best (state,")
w("family), so this is the most favourable reading available to the transformation side.")
w("")
rows = []
for d in DS:
    p = PB.get(d)
    if not p:
        continue
    rec = p["STEP7_missing_required_admission_recall"]
    cand = [k for k in rec if "/" in k]
    if not cand:
        continue
    bb = max(cand, key=lambda k: (rec[k]["@6"] or 0))
    z, f, _ = bb.split("/")
    cells = []
    for v in VARS:
        k = f"{z}/{f}/{v}"
        cells.append(f"{rec[k]['@6']:.4f} / {rec[k]['@50']:.4f}" if k in rec else "n/a")
    rows.append([d, f"{z}/{f}"] + cells
                + [f"{rec['SRC_ASSIGN']['@6']:.4f} / {rec['SRC_ASSIGN']['@50']:.4f}"])
tbl(["corpus", "best state/family", "T_SINGLE (V1)", "T_ASSIGN (V2)", "T_MAX_VIEW (V3)",
     "V4_SRC_EXIT", "SRC_ASSIGN (no transform)"], rows)
w("")

# ------------------------------------------------------------------ B4
w("## PART B4 -- forced-swap margin  score(admitted SRC candidate) - score(evicted SAFE candidate)")
w("")
rows = []
for d in DS:
    p = PB.get(d)
    if not p:
        continue
    pr = p["pairs"]["A1_SRC"]
    rows.append([d, pr["GOOD"], pr["BAD"], pr["TIE"], pr["total"]])
tbl(["corpus", "GOOD_SWAP", "BAD_SWAP", "unlabelled", "all forced pairs"], rows)
w("")
w("A corpus with an empty class has nothing for any score to separate.")
w("")
# the universality test: the SAME score key on every corpus, sign checked
have = [d for d in DS if PB.get(d)]
keys = [k for k in (PB[have[0]]["B4_swap_margin"] if have else {}) if k.startswith("A1_SRC/")]
auc = {k: {d: PB[d]["B4_swap_margin"][k]["AUC"] for d in have} for k in keys}
def consistent(k):
    v = [x for x in auc[k].values() if x is not None]
    return len(v) > 1 and (all(x > 0.5 for x in v) or all(x < 0.5 for x in v))
rows = []
order = sorted(keys, key=lambda k: -np.mean([abs((auc[k][d] or 0.5) - 0.5) for d in have]))
for k in order:
    rows.append([k.split("/", 1)[1]]
                + [f"{auc[k][d]:.4f}" if auc[k][d] is not None else "n/a" for d in have]
                + ["YES" if consistent(k) else "no"])
tbl(["score"] + have + ["sign consistent"], rows)
R["GOOD_BAD_SWAP_AUC"] = {k: auc[k] for k in order[:3]}
R["GOOD_BAD_SWAP_SIGN_CONSISTENT"] = {k: consistent(k) for k in order}
w("")
w("`sign consistent` = the score points the SAME way (all AUC above 0.5, or all below) on every")
w("corpus where both classes exist.  A score that inverts is not a usable eviction signal.")
w("")

# the same test on the honest subset: pairs where BOTH candidates actually have evidence.
# outside that subset the margin is decided by the FLOOR, i.e. by presence, not by geometry.
auce = {k: {d: PB[d]["B4_swap_margin"][k].get("AUC_both_evid") for d in have} for k in keys}
def consistent_e(k):
    v = [x for x in auce[k].values() if x is not None]
    return len(v) > 1 and (all(x > 0.5 for x in v) or all(x < 0.5 for x in v))
rows = []
order_e = sorted(keys, key=lambda k: -min([(auce[k][d] if auce[k][d] is not None else 0.5)
                                           for d in have]))
for k in order_e:
    rows.append([k.split("/", 1)[1]]
                + [f"{auce[k][d]:.4f}" if auce[k][d] is not None else "n/a" for d in have]
                + ["YES" if consistent_e(k) else "no"])
tbl(["score (BOTH-evidence pairs only)"] + have + ["sign consistent"], rows)
R["GOOD_BAD_SWAP_AUC_BOTH_EVIDENCE"] = {k: auce[k] for k in order_e[:3]}
R["GOOD_BAD_SWAP_SIGN_CONSISTENT_BOTH_EVIDENCE"] = {k: consistent_e(k) for k in order_e}
rows = [[d] + [f"{PB[d]['B4_swap_margin'][order_e[0]]['n_good_ev']}/"
               f"{PB[d]['B4_swap_margin'][order_e[0]]['n_bad_ev']}"] for d in have]
tbl(["corpus", "GOOD/BAD pairs with evidence on both sides"], rows)
w("")

# ------------------------------------------------------------------ STEP 9/10 = B5/B6
if any(SW.values()):
    w("## STEPS 9-10 / B5-B6 -- fixed-K transformation-driven REPLACEMENT at exact P50")
    w("")
    rows = []
    for d in DS:
        s = SW.get(d)
        if not s:
            continue
        for nm, p in s["p50"].items():
            for k, v in p.items():
                if k == "FROZEN":
                    continue
                rows.append([f"{d}/{nm}", k, f"{p['FROZEN']:.4f}", f"{v['acc']:.4f}",
                             f"{v['net']:+d}", f"+{v['gained']}/-{v['lost']}",
                             f"{v['p']:.4f}" + ("*" if v["sig"] else "")])
    tbl(["corpus/slice", "score / selector", "frozen", "acc", "net", "gain/loss", "p"], rows)
    w("")
    rows = [[d, SW[d]["queries_with_a_changed_pool"], f"{SW[d]['transform_overhead_ms_per_q']:.2f}"]
            for d in DS if SW.get(d)]
    tbl(["corpus", "queries whose pool actually changed", "transform ms/q"], rows)
    w("")

# ------------------------------------------------------------------ cost
w("## STEP 12 / A3 -- cost")
w("")
rows = []
for d in DS:
    if PA[d]:
        a = PA[d]["latency_ms"]["admission"]
        t = TF.get(d)
        rows.append([d] + [f"{a[x]:.4f}" for x in DEL]
                    + [f"{t['latency_ms_per_q']['per_family_per_state']:.3f}" if t else "n/a",
                       f"{t['edges_per_query']:.0f}" if t else "n/a"])
tbl(["corpus"] + [f"admit {x}" for x in DEL] + ["1 family x 1 state ms/q", "edges/q"], rows)
w("")

os.makedirs(KTD, exist_ok=True)
open(f"{KTD}/TABLES.md", "w").write("\n".join(L) + "\n")
json.dump(R, open(f"{D}/_report_derived.json", "w"), indent=1)
print("\n".join(L[:0]) or f"wrote {KTD}/TABLES.md  ({len(L)} lines)")
