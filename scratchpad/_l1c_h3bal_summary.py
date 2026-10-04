"""Pre-registered summary of BALANCED_H3_PATCH1 (fourteenth ruling, item 3) over the five DEV_A caches: applies the decision rule of
PREREGISTRATION_BALANCED_H3_PATCH1.json to the five h3bal_A_<cache>.json records and writes results/L1_COVPART/h3bal_SUMMARY.json.
    python -u scratchpad/_l1c_h3bal_summary.py
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
PRE = os.path.join(OUT, "PREREGISTRATION_BALANCED_H3_PATCH1.json")
FP = os.path.join(OUT, "h3bal_SUMMARY.json")
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
ARM, BAL, H3, H2 = "BALANCED_H3_PATCH1", "BALANCED_H2_PATCH1", "H3_PATCH1", "H2_PATCH1"
ALPHA = 0.01


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt(o):
    return "%s (+%d/-%d p=%.1e)" % (o.get("label", "-"), o["gained"], o["lost"], o["p"])


assert not os.path.exists(FP), "write-once"
pre = json.load(open(PRE, encoding="utf-8"))
for mod in ("_l1c_h3bal.py", "_l1c_h3bal_summary.py"):
    assert sha_file(os.path.join(HERE, mod)) == pre["code"]["new_modules"][mod]["sha256"], "%s changed since the pre-registration" % mod
pre_sha = sha_file(PRE)
recs, shas = {}, {}
for c in CACHES:
    p = os.path.join(OUT, "h3bal_A_%s.json" % c)
    recs[c] = json.load(open(p, encoding="utf-8"))
    shas[c] = sha_file(p)
    assert recs[c]["preregistration"]["sha256"] == pre_sha and recs[c]["arm"] == ARM and recs[c]["cache"] == c

per = {}
for c in CACHES:
    a = recs[c]["arms"]
    P = a[ARM]
    row = {"ALL": {k: a[k]["ALL"] for k in a}, "BALANCED_H3_vs_BALANCED_H2_PATCH1": P["ALL_vs_BALANCED_H2_PATCH1"], "BALANCED_H3_vs_H3_PATCH1": P["ALL_vs_H3_PATCH1"],
           "BALANCED_H3_vs_H2_PATCH1": P["ALL_vs_H2_PATCH1"], "BALANCED_H3_vs_SAFE": P["ALL_vs_SAFE"], "BALANCED_H3_vs_L1": P["ALL_vs_L1"],
           "H3_PATCH1_vs_H2_PATCH1 (section 17, reproduced)": a[H3]["ALL_vs_H2_PATCH1"], "BALANCED_H2_vs_H2_PATCH1 (section 21, reproduced)": a[BAL]["ALL_vs_H2_PATCH1"],
           "patch": {k: P[k] for k in ("depth", "novel_nodes_per_query_mean", "novel_per_hop_mean", "patch_size_mean",
                                       "patch_composition_mean (hop-1 / hop-2 / hop-3 novel / fill from removed block)", "queries_truncated",
                                       "gold_nodes_gained_vs_SAFE", "gold_nodes_gained_vs_SAFE_by_hop (1 / 2 / 3)", "gold_nodes_lost_with_removed_block")},
           "H3_PATCH1_patch": {k: a[H3][k] for k in ("novel_per_hop_mean", "patch_composition_mean (hop-1 / hop-2 / hop-3 novel / fill from removed block)",
                                                     "gold_nodes_gained_vs_SAFE_by_hop (1 / 2 / 3)")},
           "labels": recs[c]["per_cache_labels"], "flips": recs[c]["attribution_flips"], "depth3_beams": recs[c]["depth3_beams_DEV_A (gold-free)"],
           "capacity": recs[c]["balanced_depth3_patch_capacity (gold-free)"], "diagnostic": recs[c]["diagnostic_depth3_reach_vs_gold"], "seconds": recs[c]["seconds"]}
    if P["per_hop"]:
        row["per_hop"] = {h: {"n": e["n"], ARM: e["ALL"], BAL: a[BAL]["per_hop"][h]["ALL"], H3: a[H3]["per_hop"][h]["ALL"], H2: a[H2]["per_hop"][h]["ALL"],
                              "SAFE": a["SAFE"]["per_hop"][h]["ALL"], "L1": a["L1"]["per_hop"][h]["ALL"],
                              "BALANCED_H3_vs_BALANCED_H2": e["vs_BALANCED_H2_PATCH1"], "BALANCED_H3_vs_H3_PATCH1": e["vs_H3_PATCH1"], "BALANCED_H3_vs_H2_PATCH1": e["vs_H2_PATCH1"],
                              "gold_nodes_gained_by_hop (1 / 2 / 3)": e["gold_nodes_gained_by_hop (1 / 2 / 3)"],
                              "H3_PATCH1_gold_nodes_gained_by_hop": a[H3]["per_hop"][h]["gold_nodes_gained_by_hop (1 / 2 / 3)"]} for h, e in P["per_hop"].items()}
    per[c] = row
    X.log("%-11s L1 %.4f SAFE %.4f H2 %.4f H3 %.4f BAL %.4f BAL_H3 %.4f | vs BAL %s | vs H3 %s | vs H2 %s | vs SAFE %s" % (
        c, a["L1"]["ALL"], a["SAFE"]["ALL"], a[H2]["ALL"], a[H3]["ALL"], a[BAL]["ALL"], P["ALL"], fmt(P["ALL_vs_BALANCED_H2_PATCH1"]), fmt(P["ALL_vs_H3_PATCH1"]),
        fmt(P["ALL_vs_H2_PATCH1"]), fmt(P["ALL_vs_SAFE"])))
    if "per_hop" in row:
        X.log("            per hop %s" % json.dumps({h: {ARM: v[ARM], BAL: v[BAL], H3: v[H3], "vsBAL": [v["BALANCED_H3_vs_BALANCED_H2"]["gained"], v["BALANCED_H3_vs_BALANCED_H2"]["lost"],
                                                                                                     round(v["BALANCED_H3_vs_BALANCED_H2"]["p"], 4)],
                                                         "vsH3": [v["BALANCED_H3_vs_H3_PATCH1"]["gained"], v["BALANCED_H3_vs_H3_PATCH1"]["lost"], round(v["BALANCED_H3_vs_H3_PATCH1"]["p"], 4)]}
                                                     for h, v in row["per_hop"].items()}))

# ---- the pre-registered rule
mq = {}
for c in ("metaqa", "metaqa_phg"):
    P = recs[c]["arms"][ARM]
    h3 = P["per_hop"]["hop3"]["vs_BALANCED_H2_PATCH1"]
    reg = {}
    for h in ("hop1", "hop2"):
        x = P["per_hop"][h]["vs_BALANCED_H2_PATCH1"]
        reg[h] = {"gained": x["gained"], "lost": x["lost"], "p": x["p"], "regression": bool(x["lost"] > x["gained"] and x["p"] < 0.05)}
    mq[c] = {"hop3_vs_BALANCED_H2": h3, "hop3_gain": bool(h3["gained"] > h3["lost"] and h3["p"] < ALPHA), "hop3_loss": bool(h3["lost"] > h3["gained"] and h3["p"] < ALPHA),
             "hop1_hop2": reg, "no_regression": not any(v["regression"] for v in reg.values()),
             "ALL_vs_BALANCED_H2": P["ALL_vs_BALANCED_H2_PATCH1"], "ALL_loss_vs_BALANCED_H2": P["ALL_vs_BALANCED_H2_PATCH1"]["label"] == "LOSS",
             "hop3_vs_H3_PATCH1 (descriptive: the frontier effect at depth 3)": P["per_hop"]["hop3"]["vs_H3_PATCH1"]}
text = {}
for c in ("squad", "squad_phg", "musique"):
    P = recs[c]["arms"][ARM]
    text[c] = {"loss_vs_BALANCED_H2": P["significant_loss_vs_BALANCED_H2_PATCH1"], "loss_vs_H2_PATCH1": P["significant_loss_vs_H2_PATCH1"], "loss_vs_SAFE": P["significant_loss_vs_SAFE"],
               "vs_BALANCED_H2": P["ALL_vs_BALANCED_H2_PATCH1"], "vs_H3_PATCH1": P["ALL_vs_H3_PATCH1"], "vs_H2_PATCH1": P["ALL_vs_H2_PATCH1"], "vs_SAFE": P["ALL_vs_SAFE"],
               "ok": not (P["significant_loss_vs_BALANCED_H2_PATCH1"] or P["significant_loss_vs_H2_PATCH1"] or P["significant_loss_vs_SAFE"])}
gain_ok = all(v["hop3_gain"] for v in mq.values())
reg_ok = all(v["no_regression"] for v in mq.values())
text_ok = all(v["ok"] for v in text.values())
metaqa_loss = any(v["hop3_loss"] or v["ALL_loss_vs_BALANCED_H2"] for v in mq.values())
if gain_ok and reg_ok and text_ok:
    outcome = "PASS_H3"
elif metaqa_loss or not reg_ok or not text_ok:
    outcome = "FAIL_WORSE"
else:
    outcome = "FAIL_NOT_BETTER"
S = {"RECORD": "BALANCED_H3_PATCH1_DEV_A_SUMMARY", "STATUS": "PREREGISTERED_SUCCESSOR_TEST_DEV_A", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "preregistration": {"path": os.path.relpath(PRE, X.REPO).replace("\\", "/"), "sha256": pre_sha},
     "records": {c: {"path": "results/L1_COVPART/h3bal_A_%s.json" % c, "sha256": shas[c]} for c in CACHES},
     "rule": pre["decision_rule_pre_registered"],
     "components": {"hop3_gain_vs_BALANCED_H2_both_metaqa (alpha 0.01)": gain_ok, "no_hop1_hop2_regression_vs_BALANCED_H2 (0.05)": reg_ok,
                    "text_safe_vs_BALANCED_H2_H2_PATCH1_SAFE (0.05)": text_ok, "metaqa_loss_vs_BALANCED_H2 (hop3 or ALL at 0.01)": metaqa_loss},
     "metaqa": mq, "text_safety": text, "OUTCOME": outcome, "per_dataset": per,
     "what_follows": pre["decision_rule_pre_registered"]["what_follows"][outcome]}
X.wj(FP, S)
X.log("OUTCOME %s | hop3 gain both metaqa %s | no hop1/hop2 regression %s | text safe %s | metaqa loss %s -> %s sha256 %s" % (
    outcome, gain_ok, reg_ok, text_ok, metaqa_loss, os.path.relpath(FP, X.REPO), sha_file(FP)[:12]))
