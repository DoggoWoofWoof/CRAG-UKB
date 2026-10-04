"""BOUNDED_VERTEXCUT_R2 summary (twelfth ruling): apply the pre-registered universal decision rule to the three replay records.

    python -u scratchpad/_l1c_br2_summary.py   -> results/L1_COVPART/br2_SUMMARY.json
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


def main():
    pre = json.load(open(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json"), encoding="utf-8"))
    recs = {ds: json.load(open(os.path.join(OUT, "br2_replay_A_%s.json" % ds), encoding="utf-8")) for ds in DS}
    for ds, r in recs.items():
        assert r["preregistration"]["sha256"] == L.sha_file(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json")) and r["DEV"] == "A"
    v = {ds: r["verdict"] for ds, r in recs.items()}
    retained = bool(v["metaqa"]["RETAINED"])
    safe = {ds: bool(v[ds]["SAFE"]) for ds in ("squad", "musique")}
    eff = {ds: bool(v[ds]["EFFICIENT"]) for ds in DS}
    if retained and all(safe.values()) and eff["squad"] and eff["musique"]:
        outcome = "SUPPORTED"
    elif retained and all(safe.values()):
        outcome = "PARTIALLY_SUPPORTED"
    else:
        outcome = "NOT_SUPPORTED"
    why = []
    why.append("metaqa locality gain %s at the matched budget (R2_MATCHED_LE vs HARD_MTK_BASE %s: +%d/-%d p=%.3g)" % (
        "RETAINED" if retained else "NOT retained", v["metaqa"]["matched_label_R2_MATCHED_LE"], v["metaqa"]["R2_MATCHED_LE_vs_gate"]["gained"], v["metaqa"]["R2_MATCHED_LE_vs_gate"]["lost"], v["metaqa"]["R2_MATCHED_LE_vs_gate"]["p"]))
    for ds in ("squad", "musique"):
        why.append("%s %s at the matched budget (%s: +%d/-%d p=%.3g), P50 %s at %.0f%% of the hard unique exposure -> %s" % (
            ds, "SAFE" if safe[ds] else "NOT safe", v[ds]["matched_label_R2_MATCHED_LE"], v[ds]["R2_MATCHED_LE_vs_gate"]["gained"], v[ds]["R2_MATCHED_LE_vs_gate"]["lost"], v[ds]["R2_MATCHED_LE_vs_gate"]["p"],
            v[ds]["primary_label_R2_P50"], 100.0 * v[ds]["R2_mean_unique_exposure_P50"] / v[ds]["hard_mean_exposure"], "EFFICIENT" if eff[ds] else "not efficient"))
    table = {}
    for ds, r in recs.items():
        c = r["cells"]
        g = r["verdict"]["gate"]
        table[ds] = {"gate": g, "gate_ALL": c[g]["ALL"]["all"], "gate_exposure": c[g]["exposure_A"]["mean"],
                     "R2_P50_ALL": c["R2_P50"]["ALL"]["all"], "R2_P50_unique": c["R2_P50"]["exposure_unique_A"]["mean"], "R2_P50_vs_gate": c["R2_P50"]["vs_" + g]["all"],
                     "R2_MATCHED_LE_ALL": c["R2_MATCHED_LE (signal)"]["ALL"]["all"], "R2_MATCHED_LE_blocks": c["R2_MATCHED_LE (signal)"]["blocks_used_A"]["mean"], "R2_MATCHED_LE_vs_gate": c["R2_MATCHED_LE (signal)"]["vs_" + g]["all"],
                     "OWNERS_ONLY_MATCHED_LE_ALL": c["OWNERS_ONLY_MATCHED_LE (R = 1 control)"]["ALL"]["all"], "alternates_value_at_matched": r["verdict"]["alternates_value_at_matched_exposure (informational)"],
                     "RF": r["representation"]["RF"], "share_lambda_2": r["representation"]["share_lambda_2"], "block_max": r["representation"]["block_size"]["max"],
                     "containment_1hop": r["representation"]["containment_1hop"], "containment_1hop_owners_only": r["representation"]["containment_1hop_owners_only"],
                     "VERDICT": r["verdict"]["VERDICT"]}
        if ds == "metaqa" and r.get("v1_metaqa"):
            table[ds]["R2_vs_V1_P50"] = r["v1_metaqa"]["R2_P50_vs_V1_P50 (informational)"]["all"]
            table[ds]["R2_vs_V1_MATCHED_LE"] = r["v1_metaqa"]["R2_MATCHED_LE_vs_V1_MATCHED_LE (informational)"]["all"]
    out = {"RECORD": "BOUNDED_VERTEXCUT_R2_SUMMARY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "DEV": "A",
           "preregistration": L.pin(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json")), "records": {ds: L.pin(os.path.join(OUT, "br2_replay_A_%s.json" % ds)) for ds in DS},
           "question": pre["question"], "decision_rule": pre["decision_rule_pre_registered"], "per_dataset": table, "metaqa_RETAINED": retained, "text_SAFE": safe, "EFFICIENT": eff,
           "OUTCOME": outcome, "why": why, "status": "post-hoc DEV_A structural-mechanism result; nothing promoted; no composition; served L1 unchanged"}
    fp = os.path.join(OUT, "br2_SUMMARY.json")
    L.S.wj(fp, out)
    L.log("BOUNDED_VERTEXCUT_R2: %s\n  %s\n  -> %s" % (outcome, "\n  ".join(why), os.path.relpath(fp, X.REPO)))
    return out


if __name__ == "__main__":
    main()
