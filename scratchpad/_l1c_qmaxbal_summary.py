"""Pre-registered summary of QMAX_BALANCED_H2_PATCH1 (fourteenth ruling, item 1) over the five DEV_A caches: applies the decision rule of
PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json to the five qmaxbal_A_<cache>.json records and writes results/L1_COVPART/qmaxbal_SUMMARY.json.
    python -u scratchpad/_l1c_qmaxbal_summary.py
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1x90_core as X  # noqa: E402

OUT = os.path.join(X.REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json")
FP = os.path.join(OUT, "qmaxbal_SUMMARY.json")
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
PRIMARY, BAL, Q50, QSAFE, H2 = "QMAX_BALANCED_H2_PATCH1", "BALANCED_H2_PATCH1", "QMAX_F0", "QMAX_SAFE", "H2_PATCH1"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


assert not os.path.exists(FP), "write-once"
pre = json.load(open(PRE, encoding="utf-8"))
for mod in ("_l1c_qmaxbal.py", "_l1c_qmaxbal_summary.py"):
    assert sha_file(os.path.join(HERE, mod)) == pre["code"]["new_modules"][mod]["sha256"], "%s changed since the pre-registration" % mod
pre_sha = sha_file(PRE)
recs, shas = {}, {}
for c in CACHES:
    p = os.path.join(OUT, "qmaxbal_A_%s.json" % c)
    recs[c] = json.load(open(p, encoding="utf-8"))
    shas[c] = sha_file(p)
    assert recs[c]["preregistration"]["sha256"] == pre_sha and recs[c]["arm"] == PRIMARY and recs[c]["cache"] == c


def fmt(o):
    return "%s (+%d/-%d p=%.1e)" % (o["label"], o["gained"], o["lost"], o["p"])


per = {}
for c in CACHES:
    a = recs[c]["arms"]
    P = a[PRIMARY]
    row = {"ALL": {k: a[k]["ALL"] for k in a}, "exposure_nodes_mean": {k: a[k].get("exposure_nodes_mean") for k in a},
           "PRIMARY_vs_BALANCED_H2_PATCH1": P["ALL_vs_BALANCED_H2_PATCH1"], "PRIMARY_vs_QMAX_F0": P["ALL_vs_QMAX_F0"],
           "PRIMARY_vs_H2_PATCH1": P["ALL_vs_H2_PATCH1"], "PRIMARY_vs_SAFE": P["ALL_vs_SAFE"], "PRIMARY_vs_L1": P["ALL_vs_L1"], "PRIMARY_vs_QMAX_SAFE": P["ALL_vs_QMAX_SAFE"],
           "QMAX_SAFE_vs_SAFE": a[QSAFE]["ALL_vs_SAFE"], "QMAX_SAFE_vs_QMAX_F0": a[QSAFE]["ALL_vs_QMAX_F0"],
           "stage_increments": recs[c]["stage_increments_DEV_A"], "cell": recs[c]["per_cache_outcome"]["cell"],
           "flips_PRIMARY_vs_BALANCED": recs[c]["attribution_flips"]["PRIMARY_vs_BALANCED_H2_PATCH1"],
           "composed_chain_identities": recs[c]["composed_chain_identities"], "seconds": recs[c]["seconds"]}
    if P["per_hop"]:
        row["per_hop"] = {h: {"n": e["n"], "PRIMARY": e["ALL"], BAL: a[BAL]["per_hop"][h]["ALL"], Q50: a[Q50]["per_hop"][h]["ALL"], H2: a[H2]["per_hop"][h]["ALL"],
                              "SAFE": a["SAFE"]["per_hop"][h]["ALL"], "L1": a["L1"]["per_hop"][h]["ALL"],
                              "PRIMARY_vs_BALANCED": e["vs_BALANCED_H2_PATCH1"], "PRIMARY_vs_QMAX_F0": e["vs_QMAX_F0"]} for h, e in P["per_hop"].items()}
    per[c] = row
    X.log("%-11s L1 %.4f SAFE %.4f H2 %.4f BAL %.4f | QMAX %.4f QSAFE %.4f PRIMARY %.4f | vs BAL %s | vs QMAX %s | vs H2 %s | cell %s" % (
        c, a["L1"]["ALL"], a["SAFE"]["ALL"], a[H2]["ALL"], a[BAL]["ALL"], a[Q50]["ALL"], a[QSAFE]["ALL"], P["ALL"],
        fmt(P["ALL_vs_BALANCED_H2_PATCH1"]), fmt(P["ALL_vs_QMAX_F0"]), fmt(P["ALL_vs_H2_PATCH1"]), row["cell"]))
    if "per_hop" in row:
        X.log("            per hop %s" % json.dumps({h: {"PRIMARY": v["PRIMARY"], BAL: v[BAL], Q50: v[Q50], "vsBAL": [v["PRIMARY_vs_BALANCED"]["gained"], v["PRIMARY_vs_BALANCED"]["lost"]],
                                                          "vsQMAX": [v["PRIMARY_vs_QMAX_F0"]["gained"], v["PRIMARY_vs_QMAX_F0"]["lost"]]} for h, v in row["per_hop"].items()}))

# ---- the pre-registered rule
lab_B = {c: per[c]["PRIMARY_vs_BALANCED_H2_PATCH1"]["label"] for c in CACHES}
lab_Q = {c: per[c]["PRIMARY_vs_QMAX_F0"]["label"] for c in CACHES}
conflict = [c for c in CACHES if "LOSS" in (lab_B[c], lab_Q[c])]
no_conflict = not conflict
musique_gain_over_balanced = lab_B["musique"] == "GAIN"
metaqa_gain_over_qmax = lab_Q["metaqa"] == "GAIN" and lab_Q["metaqa_phg"] == "GAIN"
passed = no_conflict and musique_gain_over_balanced and metaqa_gain_over_qmax
if passed:
    outcome = "PASS_STACKS"
elif not no_conflict:
    outcome = "FAIL_CONFLICT"
else:
    outcome = "FAIL_NOT_STACKED"
flags = {"ADDITIVE_ON_MUSIQUE (PRIMARY vs QMAX_F0 = GAIN on musique: SAFE + patch add on top of the aggregation)": lab_Q["musique"] == "GAIN",
         "ADDITIVE_ON_METAQA (PRIMARY vs BALANCED = GAIN on both metaqa caches: the aggregation adds on top of the structural chain)": lab_B["metaqa"] == "GAIN" and lab_B["metaqa_phg"] == "GAIN",
         "SQUAD_NON_REGRESSION (no LOSS vs either parent on squad / squad_phg)": all(c not in conflict for c in ("squad", "squad_phg"))}
S = {"RECORD": "QMAX_BALANCED_H2_PATCH1_DEV_A_SUMMARY", "STATUS": "PREREGISTERED_COMPOSITION_TEST_DEV_A", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "preregistration": {"path": os.path.relpath(PRE, X.REPO).replace("\\", "/"), "sha256": pre_sha},
     "records": {c: {"path": "results/L1_COVPART/qmaxbal_A_%s.json" % c, "sha256": shas[c]} for c in CACHES},
     "rule": pre["decision_rule_pre_registered"],
     "components": {"no_CONFLICT_on_any_cache (a)": no_conflict, "conflict_caches": conflict, "musique_PRIMARY_vs_BALANCED_GAIN (b)": musique_gain_over_balanced,
                    "metaqa_and_metaqa_phg_PRIMARY_vs_QMAX_F0_GAIN (c)": metaqa_gain_over_qmax},
     "labels": {"PRIMARY_vs_BALANCED_H2_PATCH1": lab_B, "PRIMARY_vs_QMAX_F0": lab_Q}, "cells": {c: per[c]["cell"] for c in CACHES},
     "OUTCOME": outcome, "secondary_flags": flags, "per_dataset": per,
     "what_follows": pre["decision_rule_pre_registered"]["what_follows"][outcome]}
X.wj(FP, S)
X.log("OUTCOME %s | (a) no conflict %s %s | (b) musique vs BALANCED %s | (c) metaqa vs QMAX_F0 %s / %s | flags %s -> %s sha256 %s" % (
    outcome, no_conflict, conflict, lab_B["musique"], lab_Q["metaqa"], lab_Q["metaqa_phg"], {k.split(" ")[0]: v for k, v in flags.items()}, os.path.relpath(FP, X.REPO), sha_file(FP)[:12]))
