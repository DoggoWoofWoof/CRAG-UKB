"""Pre-registered summary of COVSCORE_H2_PATCH1 (fourteenth ruling, item 2) over the five DEV_A caches: applies the decision rule of
PREREGISTRATION_COVSCORE_H2_PATCH1.json to the five covscore_A_<cache>.json records and writes results/L1_COVPART/covscore_SUMMARY.json.
    python -u scratchpad/_l1c_covscore_summary.py
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
PRE = os.path.join(OUT, "PREREGISTRATION_COVSCORE_H2_PATCH1.json")
FP = os.path.join(OUT, "covscore_SUMMARY.json")
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
ARM, BAL, H2 = "COVSCORE_H2_PATCH1", "BALANCED_H2_PATCH1", "H2_PATCH1"
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
for mod in ("_l1c_covscore.py", "_l1c_covscore_summary.py"):
    assert sha_file(os.path.join(HERE, mod)) == pre["code"]["new_modules"][mod]["sha256"], "%s changed since the pre-registration" % mod
pre_sha = sha_file(PRE)
recs, shas = {}, {}
for c in CACHES:
    p = os.path.join(OUT, "covscore_A_%s.json" % c)
    recs[c] = json.load(open(p, encoding="utf-8"))
    shas[c] = sha_file(p)
    assert recs[c]["preregistration"]["sha256"] == pre_sha and recs[c]["arm"] == ARM and recs[c]["cache"] == c

per = {}
for c in CACHES:
    a = recs[c]["arms"]
    P = a[ARM]
    row = {"ALL": {k: a[k]["ALL"] for k in a}, "COVSCORE_vs_BALANCED_H2_PATCH1": P["ALL_vs_BALANCED_H2_PATCH1"], "COVSCORE_vs_H2_PATCH1": P["ALL_vs_H2_PATCH1"],
           "COVSCORE_vs_SAFE": P["ALL_vs_SAFE"], "COVSCORE_vs_L1": P["ALL_vs_L1"], "BALANCED_vs_H2_PATCH1": a[BAL]["ALL_vs_H2_PATCH1"],
           "labels": recs[c]["per_cache_labels"], "flips_vs_BALANCED": recs[c]["attribution_flips"]["COVSCORE_vs_BALANCED_H2_PATCH1"],
           "covscore_beam": recs[c]["covscore_beam_DEV_A"], "balanced_beam_sizes": recs[c]["beam_diversity_DEV_A"]["beam_size_mean (hop1 / hop2)"],
           "diagnostic": recs[c]["diagnostic_frontier_compression_vs_gold"], "seconds": recs[c]["seconds"]}
    if P["per_hop"]:
        row["per_hop"] = {h: {"n": e["n"], ARM: e["ALL"], BAL: a[BAL]["per_hop"][h]["ALL"], H2: a[H2]["per_hop"][h]["ALL"], "SAFE": a["SAFE"]["per_hop"][h]["ALL"], "L1": a["L1"]["per_hop"][h]["ALL"],
                              "COVSCORE_vs_BALANCED": e["vs_BALANCED_H2_PATCH1"], "COVSCORE_vs_H2_PATCH1": e["vs_H2_PATCH1"]} for h, e in P["per_hop"].items()}
    per[c] = row
    X.log("%-11s L1 %.4f SAFE %.4f H2 %.4f BAL %.4f COVSCORE %.4f | vs BAL %s | vs H2 %s | vs SAFE %s" % (
        c, a["L1"]["ALL"], a["SAFE"]["ALL"], a[H2]["ALL"], a[BAL]["ALL"], P["ALL"], fmt(P["ALL_vs_BALANCED_H2_PATCH1"]), fmt(P["ALL_vs_H2_PATCH1"]), fmt(P["ALL_vs_SAFE"])))
    if "per_hop" in row:
        X.log("            per hop %s" % json.dumps({h: {ARM: v[ARM], BAL: v[BAL], H2: v[H2], "vsBAL": [v["COVSCORE_vs_BALANCED"]["gained"], v["COVSCORE_vs_BALANCED"]["lost"], round(v["COVSCORE_vs_BALANCED"]["p"], 4)]}
                                                     for h, v in row["per_hop"].items()}))

# ---- the pre-registered rule
mq = {}
for c in ("metaqa", "metaqa_phg"):
    P = recs[c]["arms"][ARM]
    h2 = P["per_hop"]["hop2"]["vs_BALANCED_H2_PATCH1"]
    reg = {}
    for h in ("hop1", "hop3"):
        x = P["per_hop"][h]["vs_BALANCED_H2_PATCH1"]
        reg[h] = {"gained": x["gained"], "lost": x["lost"], "p": x["p"], "regression": bool(x["lost"] > x["gained"] and x["p"] < 0.05)}
    mq[c] = {"hop2_vs_BALANCED": h2, "hop2_gain": bool(h2["gained"] > h2["lost"] and h2["p"] < ALPHA), "hop2_loss": bool(h2["lost"] > h2["gained"] and h2["p"] < ALPHA),
             "hop1_hop3": reg, "no_regression": not any(v["regression"] for v in reg.values()),
             "ALL_vs_BALANCED": P["ALL_vs_BALANCED_H2_PATCH1"], "ALL_loss_vs_BALANCED": P["ALL_vs_BALANCED_H2_PATCH1"]["label"] == "LOSS"}
text = {}
for c in ("squad", "squad_phg", "musique"):
    P = recs[c]["arms"][ARM]
    text[c] = {"loss_vs_BALANCED": P["significant_loss_vs_BALANCED_H2_PATCH1"], "loss_vs_H2_PATCH1": P["significant_loss_vs_H2_PATCH1"], "loss_vs_SAFE": P["significant_loss_vs_SAFE"],
               "vs_BALANCED": P["ALL_vs_BALANCED_H2_PATCH1"], "vs_H2_PATCH1": P["ALL_vs_H2_PATCH1"], "vs_SAFE": P["ALL_vs_SAFE"],
               "ok": not (P["significant_loss_vs_BALANCED_H2_PATCH1"] or P["significant_loss_vs_H2_PATCH1"] or P["significant_loss_vs_SAFE"])}
gain_ok = all(v["hop2_gain"] for v in mq.values())
reg_ok = all(v["no_regression"] for v in mq.values())
text_ok = all(v["ok"] for v in text.values())
metaqa_loss = any(v["hop2_loss"] or v["ALL_loss_vs_BALANCED"] for v in mq.values())
if gain_ok and reg_ok and text_ok:
    outcome = "PASS_SUCCESSOR"
elif metaqa_loss or not reg_ok or not text_ok:
    outcome = "FAIL_WORSE"
else:
    outcome = "FAIL_NOT_BETTER"
# the frontier for item 3 (pre-registered): the arm with more coverage on the metaqa caches; BALANCED keeps the role unless COVSCORE passes
item3_frontier = ARM if outcome == "PASS_SUCCESSOR" else BAL
S = {"RECORD": "COVSCORE_H2_PATCH1_DEV_A_SUMMARY", "STATUS": "PREREGISTERED_SUCCESSOR_TEST_DEV_A", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "preregistration": {"path": os.path.relpath(PRE, X.REPO).replace("\\", "/"), "sha256": pre_sha},
     "records": {c: {"path": "results/L1_COVPART/covscore_A_%s.json" % c, "sha256": shas[c]} for c in CACHES},
     "rule": pre["decision_rule_pre_registered"],
     "components": {"hop2_gain_vs_BALANCED_both_metaqa (alpha 0.01)": gain_ok, "no_hop1_hop3_regression_vs_BALANCED (0.05)": reg_ok, "text_safe_vs_BALANCED_H2_PATCH1_SAFE (0.05)": text_ok,
                    "metaqa_loss_vs_BALANCED (hop2 or ALL at 0.01)": metaqa_loss},
     "metaqa": mq, "text_safety": text, "OUTCOME": outcome, "frontier_for_item_3": item3_frontier, "per_dataset": per,
     "what_follows": pre["decision_rule_pre_registered"]["what_follows"][outcome]}
X.wj(FP, S)
X.log("OUTCOME %s | hop2 gain both metaqa %s | no hop1/hop3 regression %s | text safe %s | metaqa loss %s | frontier for item 3 = %s -> %s sha256 %s" % (
    outcome, gain_ok, reg_ok, text_ok, metaqa_loss, item3_frontier, os.path.relpath(FP, X.REPO), sha_file(FP)[:12]))
