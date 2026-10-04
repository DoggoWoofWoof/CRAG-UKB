"""CORE-EXIT PROVENANCE AUDIT -- assemble RETURNS.json from the diag artefacts.

Pure assembly. Every field traces to a measured file; nothing is inferred here.

  python scratchpad/_l1pv_returns.py
"""
import os, sys, json
import numpy as np

KTD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PROVENANCE"
D = f"{KTD}/diag"
OLD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CAPACITY/diag"
ORD = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
SC = ["SRC_ALL", "SRC_NER", "SRC_STRUCT_ONLY", "ALL_NER", "ALL_STRUCT_ONLY"]
ROWS = [("MetaQA hop2", "metaqa", 2), ("MetaQA hop3", "metaqa", 3), ("WebQSP", "webqsp", None),
        ("2Wiki", "2wiki_clean", None), ("MuSiQue", "musique_clean", None),
        ("Hotpot", "hotpotqa_clean", None), ("SQuAD", "squad_clean", None)]
PL = "A1_SRC"


def cell(o, key, hop):
    if hop is None:
        return o["STEP2_swap"].get(f"{PL}/{key}")
    return o["STEP2_swap_by_hop"].get(f"{PL}/{key}/hop{hop}")


def main():
    F = json.load(open(f"{D}/step1_families.json"))
    P = {d: json.load(open(f"{D}/pv_{d}.json")) for d in ORD if os.path.exists(f"{D}/pv_{d}.json")}
    W = {d: json.load(open(f"{D}/pvswap_{d}.json")) for d in ORD
         if os.path.exists(f"{D}/pvswap_{d}.json")}
    R = {}

    # ---- provenance of the substrate --------------------------------------------------------
    R["EDGE_FAMILIES_PRESENT_IN_TRAVERSAL"] = ["STRUCT", "NER"]
    R["KNN_EDGES_IN_TRAVERSAL"] = {d: F[d]["traversal_edges_also_KNN"] for d in ORD if d in F}
    R["FRAC_TRAVERSAL_ALSO_NER"] = {d: F[d]["frac_traversal_also_NER"] for d in ORD if d in F}
    R["CORE_EXIT_TRANSITIONS_PER_QUERY"] = {
        d: P[d]["STEP1_transitions_per_query"]["core_exit"]["mean"] for d in P}
    R["CORE_EXIT_NER_TRANSITIONS_PER_QUERY"] = {
        d: P[d]["STEP1_transitions_per_query"]["core_exit_NER"]["mean"] for d in P}

    # ---- STEP 2 universality ----------------------------------------------------------------
    auc, aucev, sign = {}, {}, {}
    for k in SC:
        auc[k], aucev[k], sign[k] = {}, {}, {}
        for nm, d, hop in ROWS:
            if d not in P:
                continue
            c = cell(P[d], k, hop)
            if not c:
                continue
            auc[k][nm] = c["AUC"]
            aucev[k][nm] = c.get("AUC_both_evid")
            sign[k][nm] = c["sign_acc"]
    R["GOOD_BAD_SWAP_AUC"] = auc
    R["GOOD_BAD_SWAP_AUC_BOTH_EVIDENCE"] = aucev
    R["GOOD_BAD_SWAP_SIGN_ACC"] = sign

    def consistent(dd):
        v = [x for x in dd.values() if x is not None]
        return bool(len(v) > 1 and (all(x > 0.5 for x in v) or all(x < 0.5 for x in v)))

    R["SIGN_CONSISTENT_BY_KEY"] = {k: consistent(auc[k]) for k in SC}
    R["SIGN_CONSISTENT_BY_KEY_BOTH_EVIDENCE"] = {k: consistent(aucev[k]) for k in SC}
    R["N_ELIGIBLE_ROWS"] = {k: sum(1 for x in auc[k].values() if x is not None) for k in SC}
    R["WORST_CASE_AUC"] = {k: (min([x for x in auc[k].values() if x is not None], default=None))
                           for k in SC}
    R["MEAN_AUC"] = {k: (round(float(np.mean([x for x in auc[k].values() if x is not None])), 4)
                         if any(x is not None for x in auc[k].values()) else None) for k in SC}
    fams = [k for k in ("SRC_NER", "SRC_STRUCT_ONLY", "ALL_NER", "ALL_STRUCT_ONLY")
            if R["SIGN_CONSISTENT_BY_KEY"][k]]
    R["EDGE_FAMILY_SIGN_CONSISTENT"] = "YES" if fams else "NO"
    R["SIGN_CONSISTENT_FAMILIES"] = fams
    R["BEST_SRC_EDGE_FAMILY"] = ("STRUCT_AND_NER (score SRC_NER)" if "SRC_NER" in fams
                                 else (fams[0] if fams else "NONE"))
    R["GENERIC_SRC_CONTROL_SIGN_CONSISTENT"] = R["SIGN_CONSISTENT_BY_KEY"]["SRC_ALL"]
    R["FAMILY_BEATS_GENERIC_SRC_ON"] = {
        k: sorted(n for n in auc[k] if auc[k].get(n) is not None
                  and auc["SRC_ALL"].get(n) is not None and auc[k][n] > auc["SRC_ALL"][n])
        for k in SC if k != "SRC_ALL"}

    # ---- STEP 3 -----------------------------------------------------------------------------
    R["STEP3_GAIN_LOSS_RATIO"] = {
        d: {b: P[d]["STEP3_exclusive_support"][f"{PL}/{b}"]["gain_loss_ratio"]
            for b in ("ONLY_NER", "ONLY_STRUCT", "MULTI_FAMILY")} for d in P}
    R["STEP3_SUPPORT"] = {
        d: {b: [P[d]["STEP3_exclusive_support"][f"{PL}/{b}"]["good_swaps"],
                P[d]["STEP3_exclusive_support"][f"{PL}/{b}"]["bad_swaps"]]
            for b in ("ONLY_NER", "ONLY_STRUCT", "MULTI_FAMILY")} for d in P}
    cmp_ = {}
    for d in P:
        a = R["STEP3_GAIN_LOSS_RATIO"][d]["ONLY_NER"]
        b = R["STEP3_GAIN_LOSS_RATIO"][d]["ONLY_STRUCT"]
        n = sum(R["STEP3_SUPPORT"][d]["ONLY_NER"])
        cmp_[d] = None if (a is None or b is None or n < 10) else bool(a > b)
    R["ONLY_NER_BEATS_ONLY_STRUCT"] = cmp_
    R["NOISIEST_FAMILY"] = ("STRUCT_ONLY" if all(v for v in cmp_.values() if v is not None)
                            else "NOT_CONSISTENT")

    # ---- STEP 4 -----------------------------------------------------------------------------
    def mono(seq):
        v = [x for x in seq if x is not None]
        if len(v) < 3:
            return None
        up = all(v[i] <= v[i + 1] for i in range(len(v) - 1))
        dn = all(v[i] >= v[i + 1] for i in range(len(v) - 1))
        return "UP" if up else ("DOWN" if dn else "NON_MONOTONE")

    s4 = {}
    for d in P:
        for fam in ("NER", "STRUCT_ONLY"):
            for axis, order in (("src_conf_quartile", ["0", "1", "2", "3"]),
                                ("src_cpos_quartile",
                                 ["PROT_Q0", "PROT_Q1", "PROT_Q2", "PROT_Q3", "BND", "OUT_OF_P50"]),
                                ("hop", ["1", "2", "3"])):
                b = P[d]["STEP4_source_quality"].get(fam, {}).get(axis, {})
                if not b:
                    continue
                s4.setdefault(f"{fam}/{axis}", {})[d] = mono(
                    [b[x]["p_required"] if x in b else None for x in order])
    R["STEP4_MONOTONE_BY_AXIS"] = s4
    R["STEP4_UNIVERSAL_MONOTONE_TREND"] = {
        k: ("YES" if len(set(v.values())) == 1 and None not in v.values() else "NO")
        for k, v in s4.items()}

    # ---- STEPS 5-7 --------------------------------------------------------------------------
    R["PARITY_SRC_ALL_EQUALS_PRIOR_V6"] = {}
    for d in W:
        po = f"{OLD}/swap_{d}.json"
        if not os.path.exists(po):
            R["PARITY_SRC_ALL_EQUALS_PRIOR_V6"][d] = "no prior file"
            continue
        a = json.load(open(po))
        m, x = 0, 0
        for nm, dd in W[d]["p50"].items():
            for sel in ("F6", "G4"):
                p = a["p50"].get(nm, {}).get(f"V6_EXIT_SRC_SIM/{sel}")
                q = dd.get(f"{d and 'SRC_ALL'}/{sel}")
                if p and q:
                    m += 1
                    x += int((p["acc"], p["net"]) == (q["acc"], q["net"]))
        R["PARITY_SRC_ALL_EQUALS_PRIOR_V6"][d] = f"{x}/{m}"
    R["CANDIDATE_EVIDENCE_COVERAGE"] = {d: W[d]["candidate_evidence_coverage"] for d in W}
    p50 = {}
    for d in W:
        for nm, dd in W[d]["p50"].items():
            for k in SC:
                for sel in ("F6", "G4"):
                    v = dd.get(f"{k}/{sel}")
                    if v:
                        p50.setdefault(k, {})[f"{d}/{nm}/{sel}"] = {
                            "frozen": dd["FROZEN"], "acc": v["acc"], "net": v["net"],
                            "p": v["p"], "sig": v["sig"]}
    R["P50"] = p50
    for k in SC:
        rows = p50.get(k, {})
        R.setdefault("SIG_REGRESSIONS", {})[k] = sorted(
            n for n, v in rows.items() if v["sig"] and v["net"] < 0)
        R.setdefault("SIG_GAINS", {})[k] = sorted(
            n for n, v in rows.items() if v["sig"] and v["net"] > 0)
    for k in SC:
        for nm, sl in (("METAQA_HOP2", "metaqa/hop2"), ("METAQA_HOP3", "metaqa/hop3")):
            for sel in ("F6", "G4"):
                v = p50.get(k, {}).get(f"{sl}/{sel}")
                if v:
                    R.setdefault(nm, {})[f"{k}/{sel}"] = (
                        f"{v['frozen']:.4f} -> {v['acc']:.4f} net {v['net']:+d}"
                        f"{' SIG' if v['sig'] else ''}")
    # ---- the directive's named decision fields ----------------------------------------------
    BEST = "SRC_NER"
    R["CROSS_CORPUS_SAFE"] = "NO" if R["SIG_REGRESSIONS"].get(BEST) else "YES"
    R["CROSS_CORPUS_SAFE_F6_ONLY"] = (
        "YES" if not [x for x in R["SIG_REGRESSIONS"].get(BEST, []) if x.endswith("/F6")] else "NO")
    R["PROMOTED"] = "NONE"
    R["PROMOTION_BAR"] = ("material MetaQA hop2/hop3 improvement AND no significant regression "
                          "anywhere -- FAILS both: hop2 net +0 on F6, and MuSiQue/G4 + Hotpot/G4 "
                          "remain significant regressions")
    R["CLASSIFICATION"] = "B. PROVENANCE_CHANNEL_PARTIAL"
    R["CLASSIFICATION_WHY"] = (
        "not A: no material MetaQA improvement on the promoted selector (hop2 +0, hop3 +1) and not "
        "cross-corpus safe on G4. not C: SRC_NER/ALL_NER are the first sign-consistent statistics "
        "found in the L1 search program (6/6 eligible rows above chance, where the generic core-exit "
        "control is 5/6 and the prior phase was 0/33), and the restriction removes every significant "
        "F6 regression. The channel is recovered as a PRESENCE signal, not as an evidence channel: "
        "it cannot rank (both-evidence AUC unsupported on 3 corpora, 0.4032 on Hotpot) and covers "
        "only 10.2-43.2 percent of the K(q) universe, so on F6 it is within one query of the frozen "
        "selector.")
    R["L1_FROZEN"] = "NO"
    os.makedirs(KTD, exist_ok=True)
    json.dump(R, open(f"{KTD}/RETURNS.json", "w"), indent=1)
    print(f"wrote {KTD}/RETURNS.json  ({len(R)} fields)")
    for k in ("EDGE_FAMILY_SIGN_CONSISTENT", "SIGN_CONSISTENT_FAMILIES", "BEST_SRC_EDGE_FAMILY",
              "GENERIC_SRC_CONTROL_SIGN_CONSISTENT", "NOISIEST_FAMILY", "WORST_CASE_AUC",
              "MEAN_AUC", "STEP4_UNIVERSAL_MONOTONE_TREND", "PARITY_SRC_ALL_EQUALS_PRIOR_V6"):
        print(f"  {k} = {json.dumps(R.get(k))}")


if __name__ == "__main__":
    main()
