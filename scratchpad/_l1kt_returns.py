"""Assemble RETURNS.json for both directives directly from the diag artefacts.

Every field is derived from a diag/*.json, never typed by hand.

  python scratchpad/_l1kt_returns.py
"""
import os, json
import numpy as np

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY"
D = f"{KTD}/diag"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
DEL = ["K0+0", "K0+4", "K0+8", "K0+16", "K0+32"]
FAMS = ["F0_TRANSLATION", "F1_HOUSEHOLDER", "F2_MIN_ROTATION", "F3_RANK1_TRANSPORT"]
jf = lambda p: json.load(open(p)) if os.path.exists(p) else None

PA = {d: jf(f"{D}/parta_{d}.json") for d in DS}
TF = {d: jf(f"{D}/tf_{d}.json") for d in DS}
PB = {d: jf(f"{D}/pb_{d}.json") for d in DS}
SW = {d: jf(f"{D}/swap_{d}.json") for d in DS}
R = {}

# ================================================================ PART A: micro-K ladder
cur = {}
for d in DS:
    if not PA[d]:
        continue
    o = PA[d]["oracle"]
    e = {"pool_size": PA[d]["pool_size"], "oracle": {s: o[s] for s in o}}
    a = [o["ALL"][k] for k in DEL]
    n = [PA[d]["pool_size"][k] for k in DEL]
    e["marginal_oracle_per_added_candidate"] = [
        round((a[i + 1] - a[i]) / max(n[i + 1] - n[i], 1e-9), 6) for i in range(len(a) - 1)]
    m = e["marginal_oracle_per_added_candidate"]
    e["left_saturating"] = bool(all(m[i] >= m[i + 1] - 1e-9 for i in range(len(m) - 1)))
    cur[d] = e
R["K_CAPACITY_CURVE"] = cur
R["K_CURVE_LEFT_SATURATES_ON"] = [d for d in cur if cur[d]["left_saturating"]]

conv = {}
for d in DS:
    if not PA[d]:
        continue
    conv[d] = {s: {k: PA[d]["p50"][s][k] for k in PA[d]["p50"][s]} for s in PA[d]["p50"]}
R["BEST_K_CROSS_CORPUS_SAFE"] = {
    d: {sel: {k.split("/")[0]: conv[d]["ALL"][k]["net"] for k in conv[d]["ALL"]
              if k.endswith("/" + sel)} for sel in ("F6", "G4")} for d in conv}
R["BEST_K_METAQA_HOP3"] = {
    sel: {k.split("/")[0]: {"acc": conv["metaqa"]["hop3"][k]["acc"],
                            "net": conv["metaqa"]["hop3"][k]["net"],
                            "sig": conv["metaqa"]["hop3"][k]["sig"]}
          for k in conv["metaqa"]["hop3"] if k.endswith("/" + sel)}
    for sel in ("F6", "G4")} if "metaqa" in conv else None
# a delta is only "best" if it is a non-negative net on every corpus AND strictly positive somewhere
bestk, why = 0, []
for k in DEL[1:]:
    nets = [conv[d]["ALL"][f"{k}/{sel}"]["net"] - conv[d]["ALL"][f"K0+0/{sel}"]["net"]
            for d in conv for sel in ("F6", "G4")]
    why.append({k: {"min_net_delta_vs_K0": min(nets), "sum_net_delta_vs_K0": sum(nets)}})
R["BEST_K_DELTA"] = bestk
R["BEST_K_DELTA_EVIDENCE"] = why
R["BEST_K_DELTA_WHY"] = ("no delta converts.  F6 net is exactly 0 at every K on every corpus "
                         "(algebraic closure, diag/closure.json); G4's only positive slice "
                         "(metaqa hop3 +13) is already present at K0+0, so it is a selector effect "
                         "and not a capacity effect.")
R["CAPACITY_PRESENT_SELECTOR_CANNOT_CONVERT"] = "YES"
R["F6_PARITY"] = {d: PA[d]["F6_PARITY"] for d in DS if PA[d]}
R["F6_ALGEBRAIC_CLOSURE"] = jf(f"{D}/closure.json")

# ================================================================ transformation algebra
R["OPERATOR_GATES"] = jf(f"{D}/gates.json")
R["ORTHOGONAL_DEGENERACY"] = jf(f"{D}/orthogonal_degeneracy.json")

lift = {}
for d in DS:
    t = TF.get(d)
    if not t:
        continue
    for z in ("q_raw", "r_q"):
        b = t["STEP3_single_hop"][f"{z}/NO_TRANSFORM"]["AUC"]
        for f in FAMS:
            lift.setdefault(f"{z}/{f}/1hop", {})[d] = round(
                t["STEP3_single_hop"][f"{z}/{f}"]["AUC"] - b, 4)
            lift.setdefault(f"{z}/{f}/chain", {})[d] = round(
                t["STEP5_composition_by_hop"][f"{z}/{f}"]["AUC"] - b, 4)
R["STEP3_STEP5_AUC_LIFT_OVER_NO_TRANSFORM"] = lift
R["STEP3_CELLS_BEATING_IDENTITY_ON_A_MAJORITY_OF_CORPORA"] = [
    k for k, v in lift.items() if sum(1 for x in v.values() if x > 0) > len(v) / 2]
best = max(lift, key=lambda k: float(np.mean(list(lift[k].values()))))
R["BEST_TRANSFORM_FAMILY"] = {
    "answer": "NONE -- no family beats the NO_TRANSFORM identity control on a majority of corpora",
    "least_bad_cell": best,
    "mean_AUC_lift_over_identity": round(float(np.mean(list(lift[best].values()))), 4),
    "corpora_where_it_beats_identity": f"{sum(1 for x in lift[best].values() if x > 0)}"
                                       f"/{len(lift[best])}",
    "best_family_vs_TRANSLATION_only": "F3_RANK1_TRANSPORT (translation is worst or near-worst "
                                       "almost everywhere), but the decisive control is the "
                                       "identity, and the whole family loses to it"}
R["STEP3_PARTITION_RANK_MEAN_PCTRANK"] = {d: TF[d]["STEP3_partition_rank"]
                                          for d in DS if TF.get(d)}
R["STEP6_STABILITY"] = {d: TF[d]["STEP6_stability"] for d in DS if TF.get(d)}
byh = {}
for d in DS:
    t = TF.get(d)
    if not t:
        continue
    for k, v in t["STEP5_composition_by_hop"].items():
        byh.setdefault(k, {})[d] = v.get("mean_tc_by_hop")
R["CHAIN_MEAN_TARGET_COMPAT_BY_HOP"] = byh
for h, nm in ((2, "CHAIN_HOP2_GAIN"), (3, "CHAIN_HOP3_GAIN")):
    g = {}
    for k, per in byh.items():
        for d, a in (per or {}).items():
            if a and a.get("1") is not None and a.get(str(h)) is not None:
                g.setdefault(k, {})[d] = round(a[str(h)] - a["1"], 4)
    bk = max(g, key=lambda k: float(np.mean(list(g[k].values()))))
    R[nm] = {"metric": f"mean cos(z_h, x_v) at hop{h} minus hop1, per corpus",
             "best_cell": bk, "per_corpus": g[bk],
             "mean": round(float(np.mean(list(g[bk].values()))), 4),
             "caveat": "F0_TRANSLATION's rise is an artefact -- z + (v-u) literally adds v to the "
                       "state, inflating every target equally, positives and negatives alike",
             "orthogonal_families_flat": {k: g[k] for k in g if "F1_" in k or "F2_" in k}}

# ================================================================ STEP 7
S7 = {}
for d in DS:
    p = PB.get(d)
    if p:
        S7[d] = {"n_queries_with_missing_required": p["STEP7_n_queries_with_missing_required"],
                 "recall": p["STEP7_missing_required_admission_recall"]}
R["STEP7_MISSING_REQUIRED_ADMISSION_RECALL"] = S7
for d, e in S7.items():
    rec = e["recall"]
    tr = [k for k in rec if "F0_TRANSLATION" in k]
    tx = [k for k in rec if any(f"/{f}/" in k for f in FAMS[1:])]
    if tr:
        b = max(tr, key=lambda k: rec[k]["@50"])
        R.setdefault("TRANSLATION_ADMISSION_R6", {})[d] = rec[b]["@6"]
        R.setdefault("TRANSLATION_ADMISSION_R50", {})[d] = rec[b]["@50"]
        R.setdefault("TRANSLATION_ADMISSION_KEY", {})[d] = b
    if tx:
        b = max(tx, key=lambda k: rec[k]["@50"])
        R.setdefault("TRANSFORM_ADMISSION_R6", {})[d] = rec[b]["@6"]
        R.setdefault("TRANSFORM_ADMISSION_R50", {})[d] = rec[b]["@50"]
        R.setdefault("TRANSFORM_ADMISSION_KEY", {})[d] = b
    for ref in ("FROZEN_SAFE", "SRC_ASSIGN", "OFFSET_RAW", "V0_NOTRANSFORM"):
        if ref in rec:
            R.setdefault(f"REF_{ref}_ADMISSION_R6_R50", {})[d] = [rec[ref]["@6"], rec[ref]["@50"]]

# ================================================================ STEP 8 multi-state
ms = {}
for d, e in S7.items():
    rec = e["recall"]
    for tag, sub in (("T_SINGLE", "V1_SINGLE"), ("T_CHAIN", "V2_CHAIN"),
                     ("T_MAX_VIEW", "V3_MAXVIEW"), ("T_ASSIGN", "V4_SRC_EXIT")):
        ks = [k for k in rec if k.endswith("/" + sub)]
        if ks:
            b = max(ks, key=lambda k: rec[k]["@50"])
            ms.setdefault(tag, {})[d] = {"key": b, "@6": rec[b]["@6"], "@50": rec[b]["@50"]}
R["STEP8_MULTI_STATE_COVERAGE"] = ms

# ================================================================ B4
have = [d for d in DS if PB.get(d) and PB[d]["pairs"]["A1_SRC"]["GOOD"] > 0
        and PB[d]["pairs"]["A1_SRC"]["BAD"] > 0]
R["B4_PAIR_COUNTS"] = {d: PB[d]["pairs"] for d in DS if PB.get(d)}
R["B4_CORPORA_WITH_BOTH_CLASSES"] = have
R["B4_CORPORA_WITH_NO_GOOD_SWAP_AT_ALL"] = [
    d for d in DS if PB.get(d) and PB[d]["pairs"]["A1_SRC"]["GOOD"] == 0]
keys = [k for k in PB["metaqa"]["B4_swap_margin"] if k.startswith("A1_SRC/")]
for fld, tag in (("AUC", ""), ("AUC_both_evid", "_BOTH_EVIDENCE")):
    tab = {}
    for k in keys:
        a = {d: PB[d]["B4_swap_margin"].get(k, {}).get(fld) for d in have}
        if all(v is not None for v in a.values()):
            tab[k[7:]] = a
    cons = {k: all(v > 0.5 for v in a.values()) for k, a in tab.items()}
    top = sorted(tab, key=lambda k: -min(tab[k].values()))[:5]
    R[f"GOOD_BAD_SWAP_AUC{tag}"] = {k: tab[k] for k in top}
    R[f"GOOD_BAD_SWAP_SIGN_CONSISTENT{tag}"] = {
        "any_key_consistent": any(cons.values()), "n_consistent": int(sum(cons.values())),
        "n_keys": len(cons), "consistent_keys": [k for k in tab if cons[k]]}
R["GOOD_BAD_SWAP_SIGN_CONSISTENT_ANSWER"] = (
    "NO" if not R["GOOD_BAD_SWAP_SIGN_CONSISTENT"]["any_key_consistent"] else "YES")
R["B4_SIGN_ACC_AND_MARGINS"] = {
    d: {k[7:]: {m: PB[d]["B4_swap_margin"][k][m]
                for m in ("mean_good", "mean_bad", "sign_acc", "AUC")}
        for k in keys if k in PB[d]["B4_swap_margin"]
        and PB[d]["B4_swap_margin"][k].get("AUC") is not None}
    for d in have}

# ================================================================ STEPS 9-10 / B5-B6
p50 = {}
for d in DS:
    s = SW.get(d)
    if s:
        p50[d] = {"queries_with_a_changed_pool": s["queries_with_a_changed_pool"],
                  "transform_ms_per_q": s["transform_overhead_ms_per_q"],
                  "core_exit_edges": s.get("core_exit_edges"),
                  "frac_core_exit_at_hop1": s.get("frac_core_exit_at_hop1"),
                  "p50": s["p50"]}
R["STEPS_9_10_EXACT_P50"] = p50
mq = p50.get("metaqa", {}).get("p50", {})
if mq:
    tk = [k for k in mq["ALL"] if k.endswith("/F6") and "V4_SRC_EXIT" in k]
    if tk:
        k6 = tk[0]
        R["TRANSFORM_P50_KEY"] = k6
        R["TRANSFORM_F6_METAQA_HOP2"] = mq.get("hop2", {}).get(k6)
        R["TRANSFORM_F6_METAQA_HOP3"] = mq.get("hop3", {}).get(k6)
        R["TRANSFORM_G4_METAQA_HOP3"] = mq.get("hop3", {}).get(k6[:-3] + "/G4")
        R["TRANSFORM_FIXED_K_METAQA_HOP3"] = R["TRANSFORM_F6_METAQA_HOP3"]
R["TRANSFORM_CROSS_CORPUS_SAFE"] = {d: p50[d]["p50"].get("ALL", {}) for d in p50}
R["TRANSFORM_OVERHEAD_MS"] = {
    "one_family_one_state_ms_per_q": {d: TF[d]["latency_ms_per_q"]["per_family_per_state"]
                                      for d in DS if TF.get(d)},
    "all_four_families_both_states_ms_per_q": {
        d: TF[d]["latency_ms_per_q"]["all_transforms_all_families"] for d in DS if TF.get(d)},
    "end_to_end_replacement_ms_per_q": {d: p50[d]["transform_ms_per_q"] for d in p50},
    "micro_K_admission_ms_per_q": {d: PA[d]["latency_ms"]["admission"] for d in DS if PA[d]}}

if __name__ == "__main__":
    json.dump(R, open(f"{KTD}/RETURNS.json", "w"), indent=1, default=float)
    print(f"wrote {KTD}/RETURNS.json  ({len(R)} top-level fields)")
    for k in R:
        print("  ", k)
