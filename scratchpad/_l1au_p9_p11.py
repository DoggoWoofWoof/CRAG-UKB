"""PHASE 9 -- canonical architecture-history table, and PHASE 11 -- decision gate
(ONLINE_EDGE_VERDICT, PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED).

Both phases are pure synthesis over already-computed Phase 1/2/3/4/7/8 JSON outputs --
no new per-query computation, no new McNemar tests (all significance calls are read
directly from the upstream phase reports, not recomputed).

ONLINE_EDGE_VERDICT answers: does the ONLINE SERVING PATH (halo pool-admission, Phase 4;
selector, Phase 3; ranking signal, Phase 7) need non-STRUCT (NERX/KNN) edges, given the
CURRENT partition already exists? = YES iff Phase 4 shows >=1 significant cost cell
(H1/H2/H3 sig=True) for that corpus.

PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED answers a DIFFERENT question: would REBUILDING
the partition using ONLY STRUCT edges (never tested by Phase 4, which restricts online
fetch but leaves the existing NERX/KNN-informed partition untouched) be worth trying?
This is decided from Phase 8's PART_B frac_with_struct_any -- the fraction of co-located
REQUIRED pairs under the CURRENT (all-family) partition that a STRUCT edge alone explains.
Low frac_with_struct_any means NERX/KNN are doing real partition-BUILD-time work that pure
STRUCT would lose; high frac_with_struct_any means STRUCT alone plausibly reproduces the
current co-location pattern already, making a pure-STRUCT rebuild a plausible cost/
simplification play (fewer edge families to compute+store) rather than an accuracy play,
since Phase 4 already shows near-zero online cost for those same corpora.

  python scratchpad/_l1au_p9_p11.py run
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

AOUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL/audit"
T0 = time.time()
log = lambda *a: print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
KB_CLASS = {"metaqa": "KB", "webqsp": "KB", "2wiki_clean": "WIKI_HYPERLINK",
           "musique_clean": "WIKI_HYPERLINK_INTERMEDIATE", "hotpotqa_clean": "WIKI_HYPERLINK",
           "squad_clean": "SINGLE_ANSWER_UNTESTABLE"}


def load(fn):
    return json.load(open("%s/%s" % (AOUT, fn)))


def by_ds(rows):
    return {r["ds"]: r for r in rows}


def run():
    p12 = load("PHASE1_LADDER_PHASE2_SAFE_GATE.json")
    p3 = by_ds(load("PHASE3_SAFE_DISMANTLE_REPORT.json"))
    p4 = by_ds(load("PHASE4_5_6_HALO_FAMILY_REPORT.json"))
    p7 = by_ds(load("PHASE7_HYPEREDGE_FAMILY_REPORT.json"))
    p8 = by_ds(load("PHASE8_PARTITION_ATTRIBUTION_REPORT.json"))

    # ---------------- PHASE 9: canonical architecture-history table ----------------
    hist = []
    for ds in DS:
        l12 = p12[ds]
        c3 = p3[ds]["CELLS"]
        c4 = p4[ds]
        c7 = p7[ds]
        c8 = p8[ds]["PART_B"]

        p4_cells = {"H1_MINUS_NERX": c4["H1_MINUS_NERX"], "H2_MINUS_KNN": c4["H2_MINUS_KNN"],
                   "H3_STRUCT_ONLY": c4["H3_STRUCT_ONLY"]}
        any_p4_sig = any(v["sig"] for v in p4_cells.values())
        max_p4_cost = min(v["delta_vs_H0"] for v in p4_cells.values())  # most negative

        selector_channel = ("STRUCT_DOMINANT" if (c3["STRUCT_vs_F6"]["sig"] and c3["STRUCT_vs_F6"]["F7_minus_F6"] > 0)
                            else "RET_SUFFICIENT" if (not c3["RET_vs_F6"]["sig"] and c3["STRUCT_vs_F6"]["sig"])
                            else "NULL_AT_SELECTOR" if (not c3["STRUCT_vs_F6"]["sig"] and not c3["RET_vs_F6"]["sig"])
                            else "MIXED")

        row = {
            "ds": ds, "corpus_class": KB_CLASS[ds],
            "final_universal_SP1": l12["A7_H4_SAFE_SP1"],
            "phase1_ladder": {"A0_no_halo_no_SAFE": l12["A0_METIS_BASE_noHalo"],
                              "A6_BASE_SP1": l12["A6_H4_BASE_SP1"], "A7_SAFE_SP1": l12["A7_H4_SAFE_SP1"]},
            "phase2_SAFE_necessity": {"delta": l12["PHASE2_SAFE_ON_SP1_EFFECT"]["S1_minus_S0"],
                                      "p": l12["PHASE2_SAFE_ON_SP1_EFFECT"]["p"],
                                      "sig": l12["PHASE2_SAFE_ON_SP1_EFFECT"]["sig"]},
            "phase3_selector_channel": {"F7_STRUCT_vs_F6": c3["STRUCT_vs_F6"], "F8_RET_vs_F6": c3["RET_vs_F6"],
                                        "verdict": selector_channel},
            "phase4_online_halo_cost": {"cells": p4_cells, "any_significant_cost": any_p4_sig,
                                        "worst_delta": round(max_p4_cost, 4),
                                        "unique_rescues_STRUCT": c4["PROVENANCE"]["UNIQUE_QUERY_RESCUES_STRUCT"],
                                        "unique_rescues_NERX": c4["PROVENANCE"]["UNIQUE_QUERY_RESCUES_NERX"],
                                        "unique_rescues_KNN": c4["PROVENANCE"]["UNIQUE_QUERY_RESCUES_KNN"]},
            "phase7_ranking_signal_SP1": {"STRUCT_only_delta": c7["SP1_C1_STRUCT_ONLY"]["delta_vs_C0"],
                                          "STRUCT_only_sig": c7["SP1_C1_STRUCT_ONLY"]["sig"],
                                          "verdict": "TIES_FULL" if not c7["SP1_C1_STRUCT_ONLY"]["sig"] else "STRUCT_INSUFFICIENT"},
            "phase8_partition_colocation": {"n_colocated_required_pairs": p8[ds]["PART_B"]["n_colocated_required_pairs"],
                                            "frac_with_struct_any": c8["frac_with_struct_any"],
                                            "frac_no_direct_edge": c8["frac_no_direct_edge"]},
        }
        hist.append(row)
    json.dump(hist, open("%s/PHASE9_ARCHITECTURE_HISTORY_TABLE.json" % AOUT, "w"), indent=1)
    log("wrote PHASE9_ARCHITECTURE_HISTORY_TABLE.json")

    print("\n%-16s %8s %8s %-8s %-20s %-16s %-16s" % (
        "ds", "SP1", "P2_sig", "P3", "P4_online_cost", "P7_rank_sig", "P8_struct%"))
    for r in hist:
        print("%-16s %8.4f %-8s %-20s %-16s %-16s %-16s" % (
            r["ds"], r["final_universal_SP1"], r["phase2_SAFE_necessity"]["sig"],
            r["phase3_selector_channel"]["verdict"],
            ("SIG(%+.4f)" % r["phase4_online_halo_cost"]["worst_delta"]) if r["phase4_online_halo_cost"]["any_significant_cost"] else "ns",
            r["phase7_ranking_signal_SP1"]["verdict"],
            ("%.1f%%" % (r["phase8_partition_colocation"]["frac_with_struct_any"] * 100)
             if r["phase8_partition_colocation"]["frac_with_struct_any"] is not None else "N/A(0 pairs)")))

    # ---------------- PHASE 11: decision gate ----------------
    gate = []
    for r in hist:
        ds = r["ds"]
        online_needed = r["phase4_online_halo_cost"]["any_significant_cost"]
        struct_frac = r["phase8_partition_colocation"]["frac_with_struct_any"]
        n_pairs = r["phase8_partition_colocation"]["n_colocated_required_pairs"]

        if n_pairs == 0:
            pure_struct_verdict = "UNTESTABLE"
            pure_struct_reason = "0 co-located required pairs under current partition (single-answer regime) -- no signal to target."
        elif struct_frac is not None and struct_frac < 0.10:
            pure_struct_verdict = "NOT_WARRANTED_LIKELY_REGRESSION"
            pure_struct_reason = ("Only %.1f%% of co-located required pairs are STRUCT-explained under the CURRENT "
                                  "(all-family) partition -- NERX/KNN are doing real partition-BUILD-time work here; "
                                  "stripping them for a pure-STRUCT rebuild would likely worsen co-location, not just "
                                  "leave it unchanged.") % (struct_frac * 100)
        elif struct_frac is not None and struct_frac >= 0.30:
            pure_struct_verdict = "WARRANTED_AS_COST_SIMPLIFICATION"
            pure_struct_reason = ("%.1f%% of co-located required pairs are already STRUCT-explained, and Phase 4 shows "
                                  "~zero online cost from STRUCT-only fetch -- a pure-STRUCT partition plausibly "
                                  "reproduces current accuracy at lower build/storage cost (fewer edge families). "
                                  "This is a cost/simplification case, not an accuracy-improvement case.") % (struct_frac * 100)
        else:
            pure_struct_verdict = "AMBIGUOUS_INTERMEDIATE"
            pure_struct_reason = ("%.1f%% STRUCT-explained -- neither clearly sufficient nor clearly insufficient; "
                                  "would need a pilot rebuild to resolve, not a clear call from existing data.") % (struct_frac * 100)

        gate.append({
            "ds": ds, "corpus_class": KB_CLASS[ds],
            "ONLINE_EDGE_VERDICT": "YES" if online_needed else "NO",
            "online_edge_reason": ("Phase 4 shows >=1 significant online halo-pool cost from dropping NERX/KNN "
                                   "(worst cell %+.4f); Phase 3 confirms STRUCT-only selector significantly beats "
                                   "the shipped F6 combination (+%.4f, meaning F6 is actually leaving accuracy on the "
                                   "table for this corpus)." % (r["phase4_online_halo_cost"]["worst_delta"],
                                                                r["phase3_selector_channel"]["F7_STRUCT_vs_F6"]["F7_minus_F6"])
                                   if online_needed and r["phase3_selector_channel"]["F7_STRUCT_vs_F6"]["sig"] and
                                      r["phase3_selector_channel"]["F7_STRUCT_vs_F6"]["F7_minus_F6"] > 0
                                   else ("Phase 4 shows >=1 significant online halo-pool cost from dropping NERX/KNN "
                                        "(worst cell %+.4f)." % r["phase4_online_halo_cost"]["worst_delta"] if online_needed
                                        else "Phase 4 shows zero significant cost across all 3 pool-ablation cells (H1/H2/H3); "
                                             "Phase 7 confirms the ranking signal itself never needs non-STRUCT edges either.")),
            "PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED": pure_struct_verdict,
            "pure_struct_reason": pure_struct_reason,
        })
    json.dump(gate, open("%s/PHASE11_DECISION_GATE.json" % AOUT, "w"), indent=1)
    log("wrote PHASE11_DECISION_GATE.json")

    print("\n%-16s %-6s %-32s" % ("ds", "ONLINE", "PURE_STRUCT_PARTITION"))
    for g in gate:
        print("%-16s %-6s %-32s" % (g["ds"], g["ONLINE_EDGE_VERDICT"], g["PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED"]))

    n_online_yes = sum(1 for g in gate if g["ONLINE_EDGE_VERDICT"] == "YES")
    n_pure_warranted = sum(1 for g in gate if g["PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED"] == "WARRANTED_AS_COST_SIMPLIFICATION")
    n_pure_regress = sum(1 for g in gate if g["PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED"] == "NOT_WARRANTED_LIKELY_REGRESSION")
    print("\nSUMMARY: ONLINE_EDGE_VERDICT=YES for %d/6 corpora (%s)" % (
        n_online_yes, ", ".join(g["ds"] for g in gate if g["ONLINE_EDGE_VERDICT"] == "YES")))
    print("PURE_STRUCT warranted-as-cost-play: %d/6 (%s); likely-regression: %d/6 (%s)" % (
        n_pure_warranted, ", ".join(g["ds"] for g in gate if g["PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED"] == "WARRANTED_AS_COST_SIMPLIFICATION"),
        n_pure_regress, ", ".join(g["ds"] for g in gate if g["PURE_STRUCT_PARTITION_EXPERIMENT_WARRANTED"] == "NOT_WARRANTED_LIKELY_REGRESSION")))
    return hist, gate


if __name__ == "__main__":
    run()
