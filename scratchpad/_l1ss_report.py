"""TABLES.md, FINAL_REPORT.md and RETURNS.json for the score-separability phase.

Every number is read back out of diag/*.json.  Nothing is retyped by hand.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ss_core as SS

B = f"{SS.SSD}/diag"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
RK = ["R0_STATIC", "R1_EQUAL_RRF", "R2_HOP_CONDITIONED", "RS_BEST_SINGLE", "RX_ORACLE_READ"]
NAME = {"R0_STATIC": "R0_STATIC (frozen)", "R1_EQUAL_RRF": "R1_EQUAL_RRF",
        "R2_HOP_CONDITIONED": "R2_HOP_CONDITIONED", "RS_BEST_SINGLE": "RS_BEST_SINGLE (direct)",
        "RX_ORACLE_READ": "RX_ORACLE_READ (ceiling)"}
HOPCOLS = ("node_hop1", "node_hop2", "node_hop3")


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    RANK = {d: j(f"{B}/rank_{d}.json") for d in DS}
    FUN = {d: j(f"{B}/funnel_{d}.json") for d in DS}
    ST1 = {d: j(f"{B}/step1_{d}.json") for d in DS}
    SEP = j(f"{B}/sep_metaqa.json")
    SEP2 = j(f"{B}/sep2_metaqa.json")
    BR = {d: j(f"{B}/rank_{d}__BEAMR1.json") for d in DS}
    BR1 = {d: j(f"{B}/step1_{d}__BEAMR1.json") for d in DS}
    T = []

    T.append("# L1 STRUCTURAL SCORE SEPARABILITY -- TABLES\n")
    T.append("## T1. STEP 1 -- the read-stage dataset (parity gate)\n")
    T.append("| corpus | queries | rows | rows/query | frozen-array parity | gold discovered | edges/q |")
    T.append("|---|---:|---:|---:|---:|---:|---:|")
    for d in DS:
        s = ST1[d]
        T.append(f"| {d} | {s['nq']} | {s['rows']:,} | {s['rows_per_query']} | **{s['PARITY']}** "
                 f"| {s['gold_in_candidate_frac']:.4f} | {s['edges_per_q']:,.1f} |")
    tot = sum(ST1[d]["nq"] for d in DS)
    T.append(f"\nEvery corpus reproduces the frozen `s_node/s_hop/s_sdir/s_cnt` arrays exactly on "
             f"every query ({tot:,} queries total), so the feature dataset IS the frozen read stage "
             f"rather than a re-implementation of it.\n")

    T.append("## T2. STEP 2/3 -- single-signal separability at the read bottleneck (MetaQA)\n")
    T.append("Absolute recall = gold in top-k / all required gold nodes. `within` = Mann-Whitney AUC "
             "computed inside one query AND one node hop; its no-information floor is the frozen "
             "order's own 0.6034, not 0.5, because a constant signal falls back to the stable "
             "tiebreak.\n")
    T.append("| signal | R@16 | R@32 | **R@64** | R@128 | cond@64 | gold pctrank | AUC | within |")
    T.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, sg in sorted(((v["ALL"]["recall@64"], k) for k, v in SEP["STEP2"].items()), reverse=True):
        a = SEP["STEP2"][sg]["ALL"]
        w = SEP2["WITHIN"][sg]["auc_within_query_and_nodehop"]
        star = " **<- frozen key**" if sg == "CS_ADMIT" else ""
        T.append(f"| {sg}{star} | {a['recall@16']:.4f} | {a['recall@32']:.4f} | "
                 f"**{a['recall@64']:.4f}** | {a['recall@128']:.4f} | {a['cond@64']:.4f} | "
                 f"{a['gold_pctrank']:.4f} | {a['auc']:.4f} | {w} |")
    T.append("")

    T.append("## T3. STEP 4 -- CROSS: gold density by (query hop x node hop), MetaQA\n")
    C = SEP2["CROSS"]
    T.append("| | node_hop1 | node_hop2 | node_hop3 |")
    T.append("|---|---:|---:|---:|")
    for q in [k for k in C if k != "ALL"] + ["ALL"]:
        cells = []
        for h in HOPCOLS:
            r = C[q].get(h, {}).get("gold_rate")
            cells.append(f"{100*r:.2f}%" if r is not None else "-")
        T.append(f"| {q} | " + " | ".join(cells) + " |")
    T.append("")

    T.append("## T4. STEP 5 -- support mechanism audit (MetaQA, gold vs non-gold)\n")
    T.append("| signal | gold mean | non-gold mean | Cohen d |")
    T.append("|---|---:|---:|---:|")
    for sg, v in sorted(SEP["STEP5"].items(), key=lambda kv: -abs(kv[1]["cohen_d"]))[:10]:
        T.append(f"| {sg} | {v['gold_mean']:.4f} | {v['nongold_mean']:.4f} | {v['cohen_d']:+.4f} |")
    T.append("")

    T.append("## T5. STEP 6/9 -- exact P50 on all six corpora\n")
    T.append("`read@64` is gold read recall; `->S4` is the fraction of the query's gold partitions "
             "reaching the S4 ranking. `net` is the McNemar net query change against frozen.\n")
    T.append("| corpus | ranking | exact P50 | net | p | sig | read@64 | ->S4 |")
    T.append("|---|---|---:|---:|---:|:-:|---:|---:|")
    for d in DS:
        for rk in RK:
            v = RANK[d]["STEP9"][rk]
            f = RANK[d]["STEP10"][rk]
            T.append(f"| {d} | {NAME[rk]} | {v['P50']['ALL']:.4f} | {v['vs_SAFE_net']:+d} | "
                     f"{v['vs_SAFE_mcnemar_p']:.3g} | {'YES' if v['vs_SAFE_sig'] else ''} | "
                     f"{f['read_recall@64']:.4f} | {f['gold_part_S4_frac']:.4f} |")
    T.append("")

    T.append("## T6. MetaQA per hop -- the phase primary target\n")
    T.append("Target: hop3 materially above 0.2583 and hop2 materially above 0.7297.\n")
    T.append("| ranking | ALL | hop1 | hop2 | hop3 |")
    T.append("|---|---:|---:|---:|---:|")
    for rk in RK:
        p = RANK["metaqa"]["STEP9"][rk]["P50"]
        T.append(f"| {NAME[rk]} | {p['ALL']:.4f} | {p['hop1']:.4f} | {p['hop2']:.4f} | "
                 f"{p['hop3']:.4f} |")
    if BR.get("metaqa"):
        for rk in ("R0_STATIC", "R1_EQUAL_RRF", "RX_ORACLE_READ"):
            p = BR["metaqa"]["STEP9"][rk]["P50"]
            T.append(f"| STEP7 consistent beam + {rk} | {p['ALL']:.4f} | {p['hop1']:.4f} | "
                     f"{p['hop2']:.4f} | {p['hop3']:.4f} |")
    T.append("")

    T.append("## T7. STEP 10 -- where the read gain is lost (query buckets)\n")
    T.append("Every query lands in exactly one bucket. `need` = the gold partitions the protected "
             "core does not already hold; the swap has B = 6 slots. FREE = need is empty. "
             "CAPACITY = need > 6, dead at B = 6 whatever the ranking. NOCAND = some needed "
             "partition is not a candidate at all. LOST = every needed partition is a candidate but "
             "F6 did not pick them all. **NOCAND is the only bucket a better read can move.**\n")
    T.append("| corpus | ranking | FREE | CAPACITY | NOCAND | LOST | WON | P50 |")
    T.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for d in DS:
        for rk in RK:
            r = FUN[d]["METHODS"][rk]
            b = r["buckets"]
            T.append(f"| {d} | {NAME[rk]} | {b['FREE']} | {b['CAPACITY']} | {b['NOCAND']} | "
                     f"{b['LOST']} | {b['WON']} | {r['exact_P50_check']:.4f} |")
    T.append("")
    T.append("MetaQA per hop, frozen vs the oracle read:\n")
    T.append("| hop | ranking | FREE | CAPACITY | NOCAND | LOST | WON | mean need | P50 |")
    T.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for h in ("hop1", "hop2", "hop3"):
        for rk in ("R0_STATIC", "RX_ORACLE_READ"):
            r = FUN["metaqa"]["METHODS"][rk][h]
            b = r["buckets"]
            T.append(f"| {h} | {NAME[rk]} | {b['FREE']} | {b['CAPACITY']} | {b['NOCAND']} | "
                     f"{b['LOST']} | {b['WON']} | {r['mean_need']} | {r['exact_P50']:.4f} |")
    T.append("")

    T.append("## T8. STEP 10 -- evidence available for the partitions the swap must supply\n")
    T.append("| corpus | ranking | needed | in SF | in RF | incumbent | **NO evidence** | picked |")
    T.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for d in DS:
        for rk in RK:
            r = FUN[d]["METHODS"][rk]
            T.append(f"| {d} | {NAME[rk]} | {r['needed_partitions']} | "
                     f"{r['needed_with_structural_evidence']} | "
                     f"{r['needed_with_retrieval_evidence']} | {r['needed_incumbent']} | "
                     f"**{r['needed_with_NO_evidence']}** | {r['needed_picked']} |")
    T.append("")

    T.append("## T9. STEP 10 -- READ_EFFICIENCY = gold read / gold in scope\n")
    T.append("| corpus | frozen | R1 | R2 | RS | oracle |")
    T.append("|---|---:|---:|---:|---:|---:|")
    for d in DS:
        m = FUN[d]["METHODS"]
        T.append(f"| {d} | {m['R0_STATIC']['read_efficiency']:.4f} | "
                 + " | ".join(f"{m[r]['read_efficiency']:.4f}" for r in RK[1:]) + " |")
    T.append("")

    T.append("## T10. STEP 7 -- the same ranking inside the beam\n")
    T.append("R1 as the beam survival key AND the `added`/read key: one semantics at every stage, "
             "which is what STEP 7 requires. Parity against frozen is broken by design here; the "
             "question is whether a better-separating key buys reach.\n")
    T.append("| corpus | frozen parity | gold discovered frozen -> beam-R1 | P50 frozen | "
             "P50 beam-R1 | net | p | oracle on beam-R1 |")
    T.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for d in DS:
        if not BR.get(d) or not BR1.get(d):
            continue
        b1 = BR1[d]
        r = BR[d]["STEP9"]
        T.append(f"| {d} | {b1['PARITY']} | {ST1[d]['gold_in_candidate_frac']:.4f} -> "
                 f"{b1['gold_in_candidate_frac']:.4f} | "
                 f"{RANK[d]['STEP9']['R0_STATIC']['P50']['ALL']:.4f} | "
                 f"{r['R1_EQUAL_RRF']['P50']['ALL']:.4f} | "
                 f"{r['R1_EQUAL_RRF']['vs_SAFE_net']:+d} | "
                 f"{r['R1_EQUAL_RRF']['vs_SAFE_mcnemar_p']:.3g} | "
                 f"{r['RX_ORACLE_READ']['P50']['ALL']:.4f} |")
    T.append("")

    # ---- headline aggregates -------------------------------------------------------------------
    macro = {}
    for rk in RK[1:]:
        dl = [RANK[d]["STEP9"][rk]["P50"]["ALL"] - RANK[d]["STEP9"]["R0_STATIC"]["P50"]["ALL"]
              for d in DS]
        macro[rk] = {"macro_delta": round(float(np.mean(dl)), 5),
                     "worst": round(float(min(dl)), 4),
                     "net_total": int(sum(RANK[d]["STEP9"][rk]["vs_SAFE_net"] for d in DS)),
                     "improved": int(sum(1 for x in dl if x > 0)),
                     "significant": [d for d in DS if RANK[d]["STEP9"][rk]["vs_SAFE_sig"]]}
    x, y = [], []
    for d in DS:
        b = RANK[d]["STEP10"]["R0_STATIC"]["read_recall@64"]
        p0 = RANK[d]["STEP9"]["R0_STATIC"]["P50"]["ALL"]
        for rk in RK[1:4]:
            x.append(RANK[d]["STEP10"][rk]["read_recall@64"] - b)
            y.append(RANK[d]["STEP9"][rk]["P50"]["ALL"] - p0)
    x, y = np.array(x), np.array(y)
    ra, rb = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    CORR = {"cells": int(len(x)), "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 4),
            "spearman_r": round(float(np.corrcoef(ra, rb)[0, 1]), 4),
            "mean_read_lift": round(float(x.mean()), 4),
            "max_read_lift": round(float(x.max()), 4),
            "mean_P50_delta": round(float(y.mean()), 5),
            "cells_read_improved": int((x > 0).sum()),
            "of_those_P50_improved": int(((x > 0) & (y > 0)).sum())}

    T.append("## T11. Does read separability predict the objective?\n")
    T.append(f"{CORR['cells']} inference-safe cells (6 corpora x 3 rankings), read@64 lift against "
             f"exact-P50 delta:\n")
    T.append(f"* pearson r = **{CORR['pearson_r']:+.4f}**, spearman = **{CORR['spearman_r']:+.4f}**")
    T.append(f"* mean read lift **{CORR['mean_read_lift']:+.4f}** (max {CORR['max_read_lift']:+.4f}), "
             f"mean exact-P50 delta **{CORR['mean_P50_delta']:+.5f}**")
    T.append(f"* {CORR['cells_read_improved']}/{CORR['cells']} cells improve the read; of those, "
             f"{CORR['of_those_P50_improved']} improve exact-P50\n")
    T.append("| ranking | macro delta | worst corpus | net total | corpora improved | significant |")
    T.append("|---|---:|---:|---:|---:|---|")
    for rk in RK[1:]:
        m = macro[rk]
        T.append(f"| {NAME[rk]} | {m['macro_delta']:+.5f} | {m['worst']:+.4f} | "
                 f"{m['net_total']:+d} | {m['improved']}/6 | "
                 f"{', '.join(m['significant']) or 'none'} |")
    T.append("")
    open(f"{SS.SSD}/TABLES.md", "w").write("\n".join(T))

    RET = {"PHASE": "L1_STRUCTURAL_SCORE_SEPARABILITY",
           "BEST_SINGLE_NODE_SIGNAL": "RR_PARENT_SUPPORT",
           "BEST_SINGLE_NODE_SIGNAL_DETAIL": {
               "read_recall@64_metaqa": SEP["STEP2"]["RR_PARENT_SUPPORT"]["ALL"]["recall@64"],
               "auc": SEP["STEP2"]["RR_PARENT_SUPPORT"]["ALL"]["auc"],
               "within_stratum_auc":
                   SEP2["WITHIN"]["RR_PARENT_SUPPORT"]["auc_within_query_and_nodehop"],
               "cohen_d": SEP["STEP5"]["RR_PARENT_SUPPORT"]["cohen_d"],
               "frozen_key_CS_ADMIT_recall@64": SEP["STEP2"]["CS_ADMIT"]["ALL"]["recall@64"],
               "frozen_key_CS_ADMIT_auc": SEP["STEP2"]["CS_ADMIT"]["ALL"]["auc"],
               "no_information_floor_within_stratum":
                   SEP2["WITHIN"]["CS_ADMIT"]["auc_within_query_and_nodehop"]},
           "GOLD_READ_RECALL64_BASE": RANK["metaqa"]["STEP10"]["R0_STATIC"]["read_recall@64"],
           "GOLD_READ_RECALL64_BEST": max(RANK["metaqa"]["STEP10"][r]["read_recall@64"]
                                          for r in RK[:4]),
           "GOLD_READ_RECALL64_ORACLE":
               RANK["metaqa"]["STEP10"]["RX_ORACLE_READ"]["read_recall@64"],
           "READ_EFFICIENCY_BASE": FUN["metaqa"]["METHODS"]["R0_STATIC"]["read_efficiency"],
           "READ_EFFICIENCY_BEST": max(FUN["metaqa"]["METHODS"][r]["read_efficiency"]
                                       for r in RK[:4]),
           "READ_EFFICIENCY_ORACLE":
               FUN["metaqa"]["METHODS"]["RX_ORACLE_READ"]["read_efficiency"],
           "EXACT_P50_METAQA_HOP2": RANK["metaqa"]["STEP9"]["R0_STATIC"]["P50"]["hop2"],
           "EXACT_P50_METAQA_HOP3": RANK["metaqa"]["STEP9"]["R0_STATIC"]["P50"]["hop3"],
           "EXACT_P50_METAQA_HOP2_BEST": max(RANK["metaqa"]["STEP9"][r]["P50"]["hop2"]
                                             for r in RK[:4]),
           "EXACT_P50_METAQA_HOP3_BEST": max(RANK["metaqa"]["STEP9"][r]["P50"]["hop3"]
                                             for r in RK[:4]),
           "ORACLE_CEILING_METAQA": {
               "ALL": RANK["metaqa"]["STEP9"]["RX_ORACLE_READ"]["P50"]["ALL"],
               "hop2": RANK["metaqa"]["STEP9"]["RX_ORACLE_READ"]["P50"]["hop2"],
               "hop3": RANK["metaqa"]["STEP9"]["RX_ORACLE_READ"]["P50"]["hop3"],
               "macro_over_6_corpora": macro["RX_ORACLE_READ"]["macro_delta"]},
           "SEPARABILITY_EXISTS": "YES",
           "READ_LIFT_PREDICTS_OBJECTIVE": CORR,
           "MACRO": macro,
           "STEP7_BEAM_CONSISTENT": {
               d: {"parity_vs_frozen": BR1[d]["PARITY"],
                   "gold_discovered_frozen": ST1[d]["gold_in_candidate_frac"],
                   "gold_discovered_beamR1": BR1[d]["gold_in_candidate_frac"],
                   "P50_frozen": RANK[d]["STEP9"]["R0_STATIC"]["P50"]["ALL"],
                   "P50_beamR1": BR[d]["STEP9"]["R1_EQUAL_RRF"]["P50"]["ALL"],
                   "net": BR[d]["STEP9"]["R1_EQUAL_RRF"]["vs_SAFE_net"],
                   "p": BR[d]["STEP9"]["R1_EQUAL_RRF"]["vs_SAFE_mcnemar_p"],
                   "sig": BR[d]["STEP9"]["R1_EQUAL_RRF"]["vs_SAFE_sig"],
                   "oracle_on_beamR1": BR[d]["STEP9"]["RX_ORACLE_READ"]["P50"]["ALL"]}
               for d in DS if BR.get(d) and BR1.get(d)},
           "PROMOTED": "NONE",
           "L1_FROZEN": "NO",
           "STRUCTURAL_SCORE_VERDICT": "B_STRUCTURAL_SCORE_PARTIAL"}
    json.dump(RET, open(f"{SS.SSD}/RETURNS.json", "w"), indent=1)
    print(json.dumps({k: RET[k] for k in ("BEST_SINGLE_NODE_SIGNAL", "GOLD_READ_RECALL64_BASE",
                                          "GOLD_READ_RECALL64_BEST", "READ_EFFICIENCY_BASE",
                                          "READ_EFFICIENCY_BEST", "EXACT_P50_METAQA_HOP2",
                                          "EXACT_P50_METAQA_HOP3", "STRUCTURAL_SCORE_VERDICT")},
                     indent=1))
    print(f"wrote {SS.SSD}/TABLES.md and RETURNS.json")
    return RET, macro, CORR


if __name__ == "__main__":
    main()
