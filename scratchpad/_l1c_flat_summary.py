"""Summary of the pre-registered flat-vs-partition measurement (results/L1_COVPART/PREREGISTRATION_FLAT_VS_PARTITION.json).

The user's question (2026-09-25, verbatim): "when we come back to our question do the partitions even help than purely having
dense and splade rankings? have we measured that with our hypergraph approach?"
Headline per cache (pre-registered): the exact McNemar label of L1@50 vs FLAT_RRF and of PRIMARY vs FLAT_RRF at the route's own
per-query exposure -- PARTITIONS_HELP (the route covers significantly more queries' gold sets), FLAT_BEATS_PARTITIONS, or
NO_SIGNIFICANT_DIFFERENCE (alpha = 0.01).  Reads only the five write-once flat_A_<cache>.json records; computes nothing from data.
    python -u scratchpad/_l1c_flat_summary.py       -> results/L1_COVPART/flat_SUMMARY.json (write-once)
"""
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_FLAT_VS_PARTITION.json")
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
KBS = ["metaqa", "metaqa_phg"]
TEXT = ["squad", "squad_phg", "musique"]
P_CURVE = (1, 2, 5, 10, 20, 50)
KEY_PAIRS = "pairs (pre-registered; route = reference, flat = arm at the route's per-query budget)"
HEAD_L1, HEAD_PRIM = "L1@50 vs FLAT_RRF", "PRIMARY vs FLAT_RRF"
CHANNELS = ("DENSE_VOTE vs FLAT_DENSE", "SPLADE_VOTE vs FLAT_SPLADE")
GMAX_KEYS = ("_gmax_FLAT_RRF (0-based deepest gold position per split-A row)", "_gmax_FLAT_DENSE", "_gmax_FLAT_SPLADE")
SAME_QUERIES = (("metaqa", "metaqa_phg"), ("squad", "squad_phg"))
PAIR_WORDS = {"GAIN": "FLAT_BEATS_PARTITIONS", "LOSS": "PARTITIONS_HELP", "NEUTRAL": "NO_SIGNIFICANT_DIFFERENCE"}
K_G, K_L = "gained (flat covers, route does not)", "lost (route covers, flat does not)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


pre = json.load(open(PRE, encoding="utf-8"))
pre_sha = sha_file(PRE)
for mod, pn in pre["code"]["new_modules"].items():
    assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
fp_out = os.path.join(OUT, "flat_SUMMARY.json")
assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
flat_sha = pre["code"]["new_modules"]["_l1c_flat.py"]["sha256"]
recs = {}
for c in CACHES:
    p = os.path.join(OUT, "flat_A_%s.json" % c)
    assert os.path.exists(p), "missing record %s" % p
    r = json.load(open(p, encoding="utf-8"))
    assert r["preregistration"]["sha256"] == pre_sha and r["code"]["sha256"] == flat_sha, c
    recs[c] = r


def brief(e):
    o = e["paired"]
    return {"ALL_route": e["ALL_route"], "ALL_flat": e["ALL_flat"], "points_flat_minus_route": e["points_flat_minus_route"],
            K_G: o[K_G], K_L: o[K_L], "p": o["p"], "label": e["label"], "verdict": e["verdict"],
            "budget_nodes_mean": e["budget_nodes"]["mean"]}


per_cache = {}
for c in CACHES:
    r = recs[c]
    PR = r[KEY_PAIRS]
    curve = {str(P): brief(PR["L1@%d vs FLAT_RRF" % P]) for P in P_CURVE}
    per_cache[c] = {
        "n_split_A": r["n_split_A"], "N": r["N"], "npart": r["npart"], "partition": r["partition"],
        "HEADLINE_L1@50": brief(PR[HEAD_L1]), "HEADLINE_PRIMARY": brief(PR[HEAD_PRIM]),
        "P_curve (L1@P vs FLAT_RRF at L1@P's per-query budget)": curve,
        "P_curve_verdicts": [[P, curve[str(P)]["verdict"], curve[str(P)]["points_flat_minus_route"]] for P in P_CURVE],
        "channels (each vote channel vs its own flat list)": {k: brief(PR[k]) for k in CHANNELS},
        "descriptive_pairs": {k: brief(v) for k, v in r["descriptive_pairs"].items()},
        "per_hop (L1@50 vs FLAT_RRF)": PR[HEAD_L1]["per_hop"], "per_hop (PRIMARY vs FLAT_RRF)": PR[HEAD_PRIM]["per_hop"],
        "per_n_gold (L1@50 vs FLAT_RRF)": PR[HEAD_L1]["per_n_gold"],
        "where_they_differ (L1@50 vs FLAT_RRF)": PR[HEAD_L1]["where_they_differ"],
        "where_they_differ (PRIMARY vs FLAT_RRF)": PR[HEAD_PRIM]["where_they_differ"],
        "flat_fixed_budgets": r["flat_fixed_budgets"],
        "exposure_to_cover_ALL (budget-free)": r["exposure_to_cover_ALL (budget-free, descriptive)"],
        "flat_validation": r["flat_validation"], "reference_reproduced": r["reference_reproduced"],
        "seconds": r["seconds"], "process_rss_mb_at_end": r["process_rss_mb_at_end"]}

# the flat side is partition-free: on the same queries it must give the same gold positions under either partition
identity = {}
for a, b in SAME_QUERIES:
    ra, rb = recs[a], recs[b]
    same = (ra["cache_file"]["row_query_ids_sha256"] is not None
            and ra["cache_file"]["row_query_ids_sha256"] == rb["cache_file"]["row_query_ids_sha256"] and ra["n_split_A"] == rb["n_split_A"])
    if same:
        for k in GMAX_KEYS:
            assert ra[k] == rb[k], "flat gold positions differ between %s and %s on the same queries (%s)" % (a, b, k)
        identity["%s == %s" % (a, b)] = "IDENTICAL flat gold positions on the same split-A queries (FLAT_RRF, FLAT_DENSE, FLAT_SPLADE)"
    else:
        identity["%s vs %s" % (a, b)] = "NOT COMPARED (different cache rows)"


def classify(key):
    lab = {c: per_cache[c][key]["label"] for c in CACHES}
    helps = [c for c in CACHES if lab[c] == "LOSS"]
    flat = [c for c in CACHES if lab[c] == "GAIN"]
    tie = [c for c in CACHES if lab[c] == "NEUTRAL"]
    if len(helps) == len(CACHES):
        s = "PARTITIONS_HELP_ON_ALL_FIVE"
    elif len(flat) == len(CACHES):
        s = "FLAT_BEATS_PARTITIONS_ON_ALL_FIVE"
    elif len(tie) == len(CACHES):
        s = "NO_SIGNIFICANT_DIFFERENCE_ON_ALL_FIVE"
    elif not helps:
        s = "PARTITIONS_NEVER_HELP (flat beats them or ties on every cache)"
    elif not flat:
        s = "PARTITIONS_HELP_OR_TIE (flat never beats them)"
    else:
        s = "REGIME_DEPENDENT (partitions help on some caches, flat beats them on others)"
    return {"label": s, "PARTITIONS_HELP": helps, "FLAT_BEATS_PARTITIONS": flat, "NO_SIGNIFICANT_DIFFERENCE": tie,
            "by_cache": {c: PAIR_WORDS[lab[c]] for c in CACHES},
            "by_regime": {"KB (metaqa, metaqa_phg)": sorted(set(PAIR_WORDS[lab[c]] for c in KBS)),
                          "text (squad, squad_phg, musique)": sorted(set(PAIR_WORDS[lab[c]] for c in TEXT))}}


verdicts = {"L1@50 vs FLAT_RRF": classify("HEADLINE_L1@50"), "PRIMARY vs FLAT_RRF": classify("HEADLINE_PRIMARY")}


def cl(c, P):
    return per_cache[c]["P_curve (L1@P vs FLAT_RRF at L1@P's per-query budget)"][str(P)]["label"]


chk = {"F1": all(per_cache[c]["HEADLINE_L1@50"]["label"] == "LOSS" and per_cache[c]["HEADLINE_PRIMARY"]["label"] == "LOSS" for c in KBS),
       "F2": all(per_cache[c]["HEADLINE_L1@50"]["label"] != "LOSS" and per_cache[c]["HEADLINE_PRIMARY"]["label"] != "LOSS" for c in ("squad", "squad_phg")),
       "F4": all(cl(c, P) == "GAIN" for c in TEXT for P in P_CURVE if P <= 5),
       "F6": all(cl(c, P) == "LOSS" for c in KBS for P in P_CURVE if P >= 10)}
stated = pre["predictions_stated_now"]
pred = {k: {"stated": stated[k], "held": bool(v)} for k, v in chk.items()}
pred["F3 (not predicted)"] = {"stated": stated["F3"], "observed": per_cache["musique"]["HEADLINE_L1@50"]["verdict"]}
pred["F5 (not predicted)"] = {"stated": stated["F5"], "observed": per_cache["musique"]["HEADLINE_PRIMARY"]["verdict"]}
pred["F4 detail (text, P <= 5)"] = {c: {str(P): PAIR_WORDS[cl(c, P)] for P in P_CURVE if P <= 5} for c in TEXT}
pred["F6 detail (KB, P >= 10)"] = {c: {str(P): PAIR_WORDS[cl(c, P)] for P in P_CURVE if P >= 10} for c in KBS}

headline = "; ".join("%s: %s" % (k, ", ".join("%s %s" % (c, v["by_cache"][c]) for c in CACHES)) for k, v in verdicts.items())
res = {"RECORD": "FLAT_VS_PARTITION_SUMMARY", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "question": pre["question"], "preregistration": {"path": "results/L1_COVPART/PREREGISTRATION_FLAT_VS_PARTITION.json", "sha256": pre_sha},
       "records": {c: {"path": "results/L1_COVPART/flat_A_%s.json" % c, "sha256": sha_file(os.path.join(OUT, "flat_A_%s.json" % c))} for c in CACHES},
       "code": {"path": "scratchpad/_l1c_flat_summary.py", "sha256": sha_file(os.path.abspath(__file__))},
       "HEADLINE (pre-registered)": headline, "verdicts": verdicts,
       "flat_side_partition_independence": identity,
       "per_cache": per_cache, "predictions": pred,
       "what_this_does_not_decide": pre["what_is_not_decided"]}
with open(fp_out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=True)
print("HEADLINE %s" % headline)
print("verdicts %s" % json.dumps({k: v["label"] for k, v in verdicts.items()}))
print("predictions %s" % json.dumps({k: v.get("held", v.get("observed")) for k, v in pred.items() if "detail" not in k}))
print("identity %s" % json.dumps(identity))
print("-> %s sha256 %s" % (os.path.relpath(fp_out, REPO), sha_file(fp_out)[:12]))
