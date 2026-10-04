"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- RETURNS.json.

Assembles the three required return blocks (EDGE SUBSTRATE / PARTITION UTILITY / SYSTEM) and the
three verdicts from the measured artifacts.  Every verdict is produced by an explicit rule stated
next to it, so the label is reproducible from the numbers rather than asserted.

  python scratchpad/_l1ep_returns.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ep_sub as EP

OUT = EP.OUT
DS = ["metaqa", "2wiki_clean", "musique_clean", "squad_clean", "hotpotqa_clean", "webqsp"]
FROZEN = {"MASTER_TOPOLOGY": "C", "P_MAIN": 50, "B": 6, "K0": 60, "M_struct": 64, "M_ret": 32,
          "MAX_HOPS": 3, "BEAM": 64, "DEG_CAP": 300, "MAX_EDGES_SCORED": 400000, "SEED_K": 5,
          "M_MAX": 256, "K_LOCK": 100, "agg": "S4", "selector": "F6",
          "traversed_family": "STRUCT ONLY (master_nodes.neighbors)"}
SIG = 0.05


def _load():
    D = json.load(open(f"{OUT}/diagnostics/_derived.json"))
    af = f"{OUT}/diagnostics/_analysis.json"
    A = json.load(open(af)) if os.path.exists(af) else {}
    return D, A


# ------------------------------------------------------------------ EDGE SUBSTRATE
def edge_block(D, A):
    A0 = D["A0"]
    C = D["CORPORA"]
    algebra = {}
    for ds, v in A0.items():
        e = v.get("edge_algebra", v)
        algebra[ds] = {k: e.get(k) for k in
                       ["n_nodes", "S", "N", "K", "NER", "A_file", "C_file", "S_and_N", "S_and_K",
                        "N_and_K", "S_u_N_u_K", "S_u_K_equals_A", "SuNuK_equals_C_file",
                        "SuNuK_minus_C_file", "C_file_minus_SuNuK", "CSR_S_equals_frozen",
                        "frac_C_never_traversed"] if k in e}
    p50, sigcells, ncells = {}, 0, 0
    for ds in DS:
        t = C.get(ds, {}).get("A11_EXACT_P50", {}).get("ALL")
        if not t:
            continue
        p50[ds] = {k: {"ALL_P50": v["ALL_P50"], "delta": v["delta_vs_frozen"], "net": v["net"],
                       "p": v["p"], "sig": v["sig"]} for k, v in t.items()}
        for k, v in t.items():
            if k == "E0_STRUCT":
                continue
            ncells += 1
            sigcells += int(bool(v["sig"]))
    best_gain = max((v["delta"] for d in p50.values() for k, v in d.items()
                     if k != "E0_STRUCT"), default=0.0)
    worst = min((v["delta"] for d in p50.values() for k, v in d.items()
                 if k != "E0_STRUCT"), default=0.0)
    sig_pos = [(ds, k) for ds, d in p50.items() for k, v in d.items()
               if k != "E0_STRUCT" and v["sig"] and v["delta"] > 0]
    sig_neg = [(ds, k) for ds, d in p50.items() for k, v in d.items()
               if k != "E0_STRUCT" and v["sig"] and v["delta"] < 0]
    # universally safe = some non-frozen substrate that never significantly regresses AND
    # significantly gains somewhere
    universal = None
    subs = set(k for d in p50.values() for k in d if k != "E0_STRUCT")
    for k in sorted(subs):
        present = [ds for ds in p50 if k in p50[ds]]
        if len(present) < len(p50):
            continue
        if any(p50[ds][k]["sig"] and p50[ds][k]["delta"] < 0 for ds in present):
            continue
        if any(p50[ds][k]["sig"] and p50[ds][k]["delta"] > 0 for ds in present):
            universal = k
            break
    dist = {ds: C[ds].get("STEP4_KNN_REDUNDANCY", {}).get("novel_needed")
            for ds in DS if C.get(ds, {}).get("STEP4_KNN_REDUNDANCY")}
    cost = {ds: C[ds]["A12_COST"] for ds in DS if C.get(ds, {}).get("A12_COST")}
    sat = {ds: C[ds]["A7_SATURATION"] for ds in DS if C.get(ds, {}).get("A7_SATURATION")}
    costume = {ds: v["NERX"].get("DENSE_WEARING_GRAPH_COSTUME")
               for ds, v in A.get("A9_DISTINCT_EVIDENCE", {}).items()
               if v["NERX"].get("DENSE_WEARING_GRAPH_COSTUME")}
    return {
        "A0_EDGE_ALGEBRA": algebra,
        "A0_STRUCT_AND_NERX_DISJOINT_ALL_CORPORA": all(
            v.get("S_and_N") == 0 for v in algebra.values() if "S_and_N" in v),
        "A0_KNN_AND_NERX_OVERLAP_NONZERO_ALL_CORPORA": all(
            v.get("N_and_K", 0) > 0 for v in algebra.values() if "N_and_K" in v),
        "A0_BITMASK_PROVENANCE_REQUIRED": True,
        "A3_PARITY": {ds: C[ds]["PARITY_pass1"].get("T0_PARITY") for ds in DS
                      if C.get(ds, {}).get("PARITY_pass1")},
        "A4_UNIQUE_NEEDED_REACH_BY_BITMASK": {ds: C[ds]["A4_BITMASK"] for ds in DS
                                              if C.get(ds, {}).get("A4_BITMASK")},
        "A5_ATTRIBUTION_AND_INTERACTION": {ds: C[ds]["A5_ATTRIBUTION"] for ds in DS
                                           if C.get(ds, {}).get("A5_ATTRIBUTION")},
        "A6_DENSE_WEARING_GRAPH_COSTUME": costume,
        "A7_SATURATION_EXPOSURE": sat,
        "A8_PATH_FAMILIES": A.get("A8_PATH_FAMILIES", {}),
        "A9_DISTINCT_EVIDENCE": A.get("A9_DISTINCT_EVIDENCE", {}),
        "A9_KNN_NOVEL_AND_NEEDED_COUNT": dist,
        "A11_EXACT_P50": p50,
        "A11_CELLS": ncells, "A11_SIGNIFICANT_CELLS": sigcells,
        "A11_BEST_GAIN": round(best_gain, 4), "A11_WORST_LOSS": round(worst, 4),
        "A11_SIGNIFICANT_GAINS": sig_pos, "A11_SIGNIFICANT_REGRESSIONS": sig_neg,
        "A12_COST": cost,
        "UNIVERSALLY_SAFE_EDGE_SUBSTRATE": universal or "NONE",
    }


# ------------------------------------------------------------------ PARTITION UTILITY
def part_block(D, A):
    C = D["CORPORA"]
    util, lead = {}, {}
    for ds in DS:
        b = C.get(ds, {}).get("B")
        if not b or "P1_METIS_CURRENT" not in b:
            continue
        v = b["P1_METIS_CURRENT"]
        degenerate = (v.get("B7_MIN_PARTITIONS_REQUIRED", {}).get("max") == 1
                      and v.get("B3_ALL_GOLD_CONTAINED") == 1.0)
        util[ds] = {
            "B1_GOLD_COMPRESSION": v.get("B1_GOLD_COMPRESSION"),
            "B1_PARTITION_FETCH_SAVING": v.get("B1_PARTITION_FETCH_SAVING"),
            "B2_PAIR_COLOCATION": "N/A_SINGLE_GOLD_PER_QUERY" if degenerate
            else v.get("B2_PAIR_COLOCATION"),
            "B2_PAIR_COLOCATION_ADJUSTED": "N/A_SINGLE_GOLD_PER_QUERY" if degenerate
            else v.get("B2_PAIR_COLOCATION_ADJUSTED"),
            "B3_ALL_GOLD_CONTAINED": v.get("B3_ALL_GOLD_CONTAINED"),
            "B3_GOLD_BLOCK_DENSITY": v.get("B3_GOLD_BLOCK_DENSITY"),
            "B4_SEED_REQUIRED_COLOCATION": v.get("B4_SEED_REQUIRED_COLOCATION"),
            "B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL":
                v.get("B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL"),
            "B5_GOLD_EDGE_CONTAINMENT": v.get("B5_GOLD_EDGE_CONTAINMENT"),
            "B7_MIN_PARTITIONS_REQUIRED": v.get("B7_MIN_PARTITIONS_REQUIRED"),
            "B8_ORACLE_ALL_AT_50": v.get("B8_ORACLE_ALL_AT_50"),
            "B12_QUALITY": v.get("B12_QUALITY"),
            "by_hop": v.get("by_hop")}
        lead[ds] = {t: {k: b[t].get(k) for k in
                        ["B2_PAIR_COLOCATION_ADJUSTED", "B3_ALL_GOLD_CONTAINED",
                         "B5_GOLD_EDGE_CONTAINMENT", "B8_ORACLE_ALL_AT_50"]}
                    for t in b if t != "P1_METIS_CURRENT"}
    ceiling = {ds: v["B8_ORACLE_ALL_AT_50"] for ds, v in util.items()}
    return {"B0_PRODUCTION_PARTITIONER": json.load(
        open(f"{OUT}/PARTITION_UTILITY/B0_provenance.json"))
        if os.path.exists(f"{OUT}/PARTITION_UTILITY/B0_provenance.json") else "see FINAL_REPORT",
        "B1_B8_SHIPPED_ASSIGNMENT": util,
        "B8_ORACLE_CEILING_AT_P50": ceiling,
        "B9_B14_LEADERBOARD_OFFLINE": lead,
        "B15_STABILITY": A.get("B15_STABILITY", {})}


# ------------------------------------------------------------------ SYSTEM
def sys_block(D, A):
    C = D["CORPORA"]
    ph = {}
    for ds in DS:
        c = C.get(ds, {}).get("C")
        if not c or "PM_CURRENT_EXACT" not in c:
            continue
        base = c["PM_CURRENT_EXACT"]
        ph[ds] = {"C1_PARITY": base.get("C1_PARITY"),
                  "production_F6_ALL_P50": base["F6_ALL_P50"],
                  "production_BASE_ALL_P50": base["BASE_ALL_P50"],
                  "by_hop": base.get("by_hop"),
                  "candidates": {t: {"F6_ALL_P50": v["F6_ALL_P50"],
                                     "delta": round(v["F6_ALL_P50"] - base["F6_ALL_P50"], 4),
                                     "exposure_nodes": v["BASE_SCOPE_NODES"],
                                     "by_hop": v.get("by_hop")}
                                 for t, v in c.items() if t != "PM_CURRENT_EXACT"}}
    return {"C1_C2_PHASE_C": ph, "C4_DECOMPOSITION": A.get("C4_DECOMPOSITION", {})}


def main():
    D, A = _load()
    R = {"PROGRAM": "L1_EDGE_AND_PARTITION_FINAL",
         "FROZEN_CONTRACT": FROZEN,
         "CORPORA_WITH_PHASE_A": [d for d in DS if D["CORPORA"].get(d, {}).get("A11_EXACT_P50")],
         "CORPORA_WITH_PHASE_B": [d for d in DS if D["CORPORA"].get(d, {}).get("B")],
         "CORPORA_WITH_PHASE_C": [d for d in DS if D["CORPORA"].get(d, {}).get("C")],
         "NO_TEST_SPLIT_TOUCHED": True, "NO_LEARNED_COMPONENT": True,
         "NO_GOLD_OR_QUERY_IN_PARTITION_BUILD": True,
         "EDGE_SUBSTRATE": edge_block(D, A),
         "PARTITION_UTILITY": part_block(D, A),
         "SYSTEM": sys_block(D, A)}
    import _l1ep_verdict as VD
    V = VD.verdicts(R, A)
    R["VERDICTS"] = {
        "EDGE_SUBSTRATE_VERDICT": V["EDGE_SUBSTRATE_VERDICT"],
        "PARTITIONING_VERDICT": V["PARTITIONING_VERDICT"],
        "COMBINED_L1_VERDICT": V["COMBINED_L1_VERDICT"],
        "PROMOTED": V["PROMOTED"], "L1_FROZEN": V["L1_FROZEN"],
        "PROMOTION_GATES": V["PROMOTION_GATES"],
        "SPECIAL_LABELS": list(V["SPECIAL_LABELS"]),
        "CLASS_SPLIT_P4_CE_LOCAL_ONLY": V["CLASS_SPLIT_P4_CE_LOCAL_ONLY"],
        "METAQA_BY_HOP_P4_CE_LOCAL_ONLY": V["METAQA_BY_HOP_P4_CE_LOCAL_ONLY"]}
    fp = f"{OUT}/RETURNS.json"
    json.dump(R, open(fp, "w"), indent=1)
    print("wrote", fp)
    e = R["EDGE_SUBSTRATE"]
    print(f"  A11 cells {e['A11_CELLS']}  significant {e['A11_SIGNIFICANT_CELLS']}  "
          f"best {e['A11_BEST_GAIN']:+.4f}  worst {e['A11_WORST_LOSS']:+.4f}  "
          f"universal safe substrate: {e['UNIVERSALLY_SAFE_EDGE_SUBSTRATE']}")
    cs = R["VERDICTS"]["CLASS_SPLIT_P4_CE_LOCAL_ONLY"]
    print(f"  verdicts E={R['VERDICTS']['EDGE_SUBSTRATE_VERDICT'][:1]} "
          f"P={R['VERDICTS']['PARTITIONING_VERDICT'][:1]} "
          f"C={R['VERDICTS']['COMBINED_L1_VERDICT'][:1]}  "
          f"promoted={R['VERDICTS']['PROMOTED']}  "
          f"clean KB/text split={cs['CLEAN_KB_VS_TEXT_SPLIT']} "
          f"(KB {cs['kb_sig_positive']}/{cs['kb_n']} sig+, text {cs['text_sig_negative']}/{cs['text_n']} sig-)")
    return R


if __name__ == "__main__":
    main()
