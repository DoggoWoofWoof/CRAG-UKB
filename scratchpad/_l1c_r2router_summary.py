"""R2_LEGACY_ROUTER summary (thirteenth ruling): apply the pre-registered decision rule to the three replay records.

    python -u scratchpad/_l1c_r2router_summary.py   -> results/L1_COVPART/r2router_SUMMARY.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1c_br2_lib as L  # noqa: E402

X = L.X
OUT = L.OUT
DS = ("metaqa", "squad", "musique")
PRE = os.path.join(OUT, "PREREGISTRATION_R2_LEGACY_ROUTER.json")
PRIMARY = "R2ROUTER_P50 (primary)"
SECONDARY = "R2ROUTER_MATCHED_LE (secondary)"


def main():
    pre = json.load(open(PRE, encoding="utf-8"))
    fp = os.path.join(OUT, "r2router_SUMMARY.json")
    assert not os.path.exists(fp), "write-once"
    recs = {ds: json.load(open(os.path.join(OUT, "r2router_A_%s.json" % ds), encoding="utf-8")) for ds in DS}
    for ds, r in recs.items():
        assert r["preregistration"]["sha256"] == L.sha_file(PRE) and r["DEV"] == "A" and r["RECORD"] == "R2_LEGACY_ROUTER_REPLAY"
    v = {ds: r["verdict"] for ds, r in recs.items()}
    safe = {ds: bool(v[ds]["SAFE"]) for ds in ("squad", "musique")}
    gain = bool(v["metaqa"]["GAIN"])
    efficient = bool(v["metaqa"]["EFFICIENT"])
    matched_gain = bool(v["metaqa"]["MATCHED_GAIN"])
    text_safe = all(safe.values())
    if text_safe and gain:
        outcome = "PASS_GAIN"
    elif text_safe and efficient:
        outcome = "PASS_EFFICIENCY"
    else:
        outcome = "FAIL"
    why = []
    m = v["metaqa"]
    why.append("metaqa R2ROUTER_P50 vs HARD_MTK_BASE %s (+%d/-%d p=%.3g; %.4f vs %.4f) at %.0f%% of the hard unique exposure -> GAIN %s, EFFICIENT %s; matched %s (+%d/-%d p=%.3g; %.4f)" % (
        m["primary_label_R2ROUTER_P50"], m["R2ROUTER_P50_vs_gate"]["gained"], m["R2ROUTER_P50_vs_gate"]["lost"], m["R2ROUTER_P50_vs_gate"]["p"], m["R2ROUTER_P50_vs_gate"]["cell_ALL"], m["R2ROUTER_P50_vs_gate"]["base_ALL"],
        100.0 * m["unique_over_hard_ratio"], gain, efficient, m["matched_label_R2ROUTER_MATCHED_LE"], m["R2ROUTER_MATCHED_LE_vs_gate"]["gained"], m["R2ROUTER_MATCHED_LE_vs_gate"]["lost"], m["R2ROUTER_MATCHED_LE_vs_gate"]["p"], m["R2ROUTER_MATCHED_LE_vs_gate"]["cell_ALL"]))
    for ds in ("squad", "musique"):
        t = v[ds]
        why.append("%s %s: P50 %s (+%d/-%d p=%.3g; %.4f vs %.4f) at %.0f%% of the hard unique exposure; matched %s (+%d/-%d p=%.3g; %.4f)" % (
            ds, "SAFE" if safe[ds] else "NOT safe", t["primary_label_R2ROUTER_P50"], t["R2ROUTER_P50_vs_gate"]["gained"], t["R2ROUTER_P50_vs_gate"]["lost"], t["R2ROUTER_P50_vs_gate"]["p"], t["R2ROUTER_P50_vs_gate"]["cell_ALL"], t["R2ROUTER_P50_vs_gate"]["base_ALL"],
            100.0 * t["unique_over_hard_ratio"], t["matched_label_R2ROUTER_MATCHED_LE"], t["R2ROUTER_MATCHED_LE_vs_gate"]["gained"], t["R2ROUTER_MATCHED_LE_vs_gate"]["lost"], t["R2ROUTER_MATCHED_LE_vs_gate"]["p"], t["R2ROUTER_MATCHED_LE_vs_gate"]["cell_ALL"]))
    table = {}
    for ds, r in recs.items():
        c = r["cells"]
        g = r["verdict"]["gate"]
        rt = r["reach_tables"]
        table[ds] = {"gate": g, "gate_ALL": c[g]["ALL"]["all"], "gate_exposure": c[g]["exposure_A"]["mean"],
                     "R2ROUTER_P50_ALL": c[PRIMARY]["ALL"]["all"], "R2ROUTER_P50_unique": c[PRIMARY]["exposure_unique_A"]["mean"], "unique_over_hard_ratio": r["verdict"]["unique_over_hard_ratio"],
                     "R2ROUTER_P50_vs_gate": c[PRIMARY]["vs_" + g]["all"],
                     "R2ROUTER_MATCHED_LE_ALL": c[SECONDARY]["ALL"]["all"], "R2ROUTER_MATCHED_LE_blocks": c[SECONDARY]["blocks_used_A"]["mean"], "R2ROUTER_MATCHED_LE_vs_gate": c[SECONDARY]["vs_" + g]["all"],
                     "section22_R2_P50_ALL": c["R2_MEMBERSHIP_VOTE_P50 (section 22 reproduced)"]["ALL"]["all"],
                     "router_value_vs_section22": c[PRIMARY]["vs_R2_MEMBERSHIP_VOTE_P50 (the router's value on the same substrate; informational)"]["all"],
                     "OWNERS_LEGACY_P50_ALL": c["OWNERS_LEGACY_P50 (R = 1 control)"]["ALL"]["all"], "OWNERS_LEGACY_MATCHED_LE_ALL": c["OWNERS_LEGACY_MATCHED_LE (R = 1 control)"]["ALL"]["all"],
                     "alternates_value_under_router_P50": r["verdict"]["alternates_value_under_router_P50 (informational)"], "alternates_value_under_router_matched": r["verdict"]["alternates_value_under_router_matched (informational)"],
                     "reach": {nm: {"reached_all": rt[nm]["reached_all_gold_rate"]["all"], "hub_gold": rt[nm]["gold_node_reached_rate_A_by_hubness"]["hub"], "non_hub_gold": rt[nm]["gold_node_reached_rate_A_by_hubness"]["non_hub"],
                                    "votes_per_hit": rt[nm]["votes_per_hit_mean (distinct blocks)"], "evidence_blocks": rt[nm]["blocks_with_evidence_per_query_mean"]} for nm in rt},
                     "reach_decomposition_primary": {kk: r["reach_decomposition_R2ROUTER_P50"][kk] for kk in ("failures_A", "UNREACHED", "REACHED_WEAK", "reached_rate_A")},
                     "failure_overlap_primary": r["failure_overlap_with_gate_A"]["R2ROUTER_P50"],
                     "RF": r["representation"]["RF"], "VERDICT": r["verdict"]["VERDICT"]}
        if ds == "metaqa":
            table[ds]["ladder_ALL_DEV_A"] = r["ladder_ALL_DEV_A"]
            table[ds]["per_hop_P50_vs_gate"] = {kk: vv for kk, vv in c[PRIMARY]["vs_" + g].items() if kk != "all"}
            table[ds]["per_hop_MATCHED_LE_vs_gate"] = {kk: vv for kk, vv in c[SECONDARY]["vs_" + g].items() if kk != "all"}
    follows = pre["decision_rule_pre_registered"]["what_follows"]
    out = {"RECORD": "R2_LEGACY_ROUTER_SUMMARY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "DEV": "A",
           "preregistration": L.pin(PRE), "records": {ds: L.pin(os.path.join(OUT, "r2router_A_%s.json" % ds)) for ds in DS},
           "question": pre["question"], "decision_rule": pre["decision_rule_pre_registered"], "per_dataset": table,
           "text_SAFE": safe, "metaqa_GAIN": gain, "metaqa_EFFICIENT": efficient, "metaqa_MATCHED_GAIN (informational)": matched_gain,
           "OUTCOME": outcome, "why": why, "what_follows": follows.get(outcome, follows.get("FAIL")),
           "status": "post-hoc DEV_A mechanism result; nothing promoted; no composition; served L1 unchanged; WebQSP primary transfer remains frozen H2_PATCH1 (section 16)"}
    L.S.wj(fp, out)
    L.log("R2_LEGACY_ROUTER: %s\n  %s\n  -> %s" % (outcome, "\n  ".join(why), os.path.relpath(fp, X.REPO)))
    return out


if __name__ == "__main__":
    main()
