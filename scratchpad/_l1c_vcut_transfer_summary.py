"""STRUCT_VERTEXCUT_V1_TRANSFER summary: the pre-registered universal-statement label from the section-19 metaqa result and the two
text transfer records (PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json, decision_rule_pre_registered.universal_statement).

    python -u scratchpad/_l1c_vcut_transfer_summary.py   -> results/L1_COVPART/vcut_transfer_SUMMARY.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1c_vcut_replay as R  # noqa: E402

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")


def ld(fn):
    return json.load(open(os.path.join(OUT, fn), encoding="utf-8"))


pre = ld("PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json")
m = ld("vcut_replay_A_metaqa.json")
recs = {ds: ld("vcut_transfer_A_%s.json" % ds) for ds in ("squad", "musique")}
metaqa_ok = m["verdict"]["VERDICT"] == "MECHANISM_POSITIVE_AT_LOWER_EXPOSURE"
per = {}
for ds, r in recs.items():
    v = r["verdict"]
    per[ds] = {"VERDICT": v["VERDICT"], "primary_label": v["primary_label"], "exposure_flag": v["exposure_flag"], "matched_label": v["matched_label"],
               "gate": v["gate"], "P50_ALL": r["cells"]["VCUT_P50"]["ALL"]["all"], "gate_ALL": r["cells"][v["gate"]]["ALL"]["all"],
               "primary_vs_gate": v["primary_VCUT_P50_vs_gate"], "unique_exposure": v["primary_mean_unique_exposure"], "hard_exposure": v["hard_mean_exposure"],
               "MATCHED_LE_ALL": r["cells"]["VCUT_MATCHED_LE"]["ALL"]["all"], "matched_vs_gate": v["secondary_VCUT_MATCHED_LE_vs_gate"],
               "matched_blocks_mean": r["cells"]["VCUT_MATCHED_LE"]["blocks_used_A"]["mean"], "matched_exposure_mean": r["cells"]["VCUT_MATCHED_LE"]["exposure_unique_A"]["mean"],
               "degree_0_placement": r["representation"]["degree_0_placement"], "block_size_after_placement": r["representation"]["block_size_after_placement"],
               "RF_after_placement": r["representation"]["RF_after_placement"], "k": r["representation"]["k"]}
no_loss = all(p["primary_label"] != "LOSS" for p in per.values())
all_le = all(p["exposure_flag"] == "EXPOSURE_LE_HARD" for p in per.values())
if not no_loss:
    stmt = "NOT_SUPPORTED"
elif metaqa_ok and all_le:
    stmt = "SUPPORTED"
else:
    stmt = "PARTIALLY_SUPPORTED"
res = {"RECORD": "STRUCT_VERTEXCUT_V1_TRANSFER_SUMMARY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "preregistration": R.pin(os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json")),
       "inputs": {"metaqa_section_19": R.pin(os.path.join(OUT, "vcut_replay_A_metaqa.json")), **{ds: R.pin(os.path.join(OUT, "vcut_transfer_A_%s.json" % ds)) for ds in recs}},
       "metaqa_section_19_verdict": m["verdict"]["VERDICT"], "per_dataset": per,
       "universal_statement": {"statement": pre["decision_rule_pre_registered"]["universal_statement"]["statement"], "label": stmt,
                               "rule": pre["decision_rule_pre_registered"]["universal_statement"], "metaqa_positive_at_lower_exposure": metaqa_ok,
                               "no_LOSS_on_text": no_loss, "both_text_exposure_le_hard": all_le},
       "status": "post-hoc DEV_A evidence (all three datasets); nothing promoted; no composition run; the served L1 unchanged",
       "code": R.pin(os.path.abspath(__file__))}
fp = os.path.join(OUT, "vcut_transfer_SUMMARY.json")
S.wj(fp, res)
print(json.dumps({"universal_statement": stmt, "per_dataset": {ds: p["VERDICT"] + " / matched " + p["matched_label"] for ds, p in per.items()}}, indent=1))
