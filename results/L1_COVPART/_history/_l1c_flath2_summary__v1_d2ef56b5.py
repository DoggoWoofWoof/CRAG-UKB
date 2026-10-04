"""Summary of the pre-registered FLAT_BALANCED_H2 ablation (results/L1_COVPART/PREREGISTRATION_FLAT_BALANCED_H2.json).

The question (the user's ruling of 2026-09-25, verbatim): "Are partitions actually necessary, or is the KB gain just graph
expansion/co-location?"
Reads only the eight write-once records -- flath2_E_<dataset>.json (webqsp, hotpotqa, 2wiki: the one-shot fresh transfers, D1 =
FLAT@5000 vs FLAT_BALANCED_H2@5000) and flath2_A_<cache>.json (the five DEV_A caches, non-selecting: D2 = PRIMARY vs
FLAT_BALANCED_H2 and D3 = FLAT vs FLAT_BALANCED_H2 at PRIMARY's per-query exposure) -- and applies the pre-registered verdict
rules.  Computes nothing from data.
    python -u scratchpad/_l1c_flath2_summary.py                       -> results/L1_COVPART/flath2_SUMMARY.json (write-once)
    python -u scratchpad/_l1c_flath2_summary.py --dry --dry-in=<dir>  (dry records outside the repository; nothing written)
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results", "L1_COVPART")
PRE = os.path.join(OUT, "PREREGISTRATION_FLAT_BALANCED_H2.json")
DEV_A = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
KBS_A = ["metaqa", "metaqa_phg"]
FRESH = ["webqsp", "hotpotqa", "2wiki"]
M_MAIN = 5000
KEY_D1 = "D1 (pre-registered: FLAT@%d vs FLAT_BALANCED_H2@%d; gained = the patch helps)" % (M_MAIN, M_MAIN)
KEY_D2 = "D2 (PRIMARY vs FLAT_BALANCED_H2 at B_q; gained = FLAT_H2 covers, PRIMARY does not)"
KEY_D3 = "D3 (FLAT vs FLAT_BALANCED_H2 at B_q; gained = the patch helps)"
KEY_P = "paired (gained = arm covers ALL gold, reference does not)"
SCENARIO_1 = ("the user's first scenario (metaqa flat=0.39, flat+H2~0.78, partition+H2~0.80): most of what we attributed to "
              "partitions was semantic seeds + real graph traversal -> seriously consider deleting partitioning from the universal "
              "retrieval architecture")
SCENARIO_2 = ("the user's second scenario (flat+H2=0.60 while partition+H2=0.80): partitions do something genuinely important -- "
              "co-locating related answer sets so one retrieval unit exposes many graph-connected answers efficiently")
DRY = "--dry" in sys.argv
DRY_IN = next((a_.split("=", 1)[1] for a_ in sys.argv if a_.startswith("--dry-in=")), None)
assert not DRY or (DRY_IN and not os.path.abspath(DRY_IN).lower().startswith(REPO.lower())), "--dry needs --dry-in=<dir outside the repository>"
assert DRY or DRY_IN is None


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def brief(e):
    o = e[KEY_P]
    return {"reference": e["reference"], "arm": e["arm"], "ALL_reference": e["ALL_reference"], "ALL_arm": e["ALL_arm"],
            "points_arm_minus_reference": e["points_arm_minus_reference"], "gained": o["gained"], "lost": o["lost"], "p": o["p"],
            "label": e["label"], "verdict": e["verdict"]}


CODE_SHA = sha_file(os.path.abspath(__file__))
pre, pre_sha, main_sha = None, None, None
if not DRY:
    pre = json.load(open(PRE, encoding="utf-8"))
    pre_sha = sha_file(PRE)
    for mod, pn in pre["code"]["new_modules"].items():
        assert sha_file(os.path.join(REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    main_sha = pre["code"]["new_modules"]["_l1c_flath2.py"]["sha256"]
fp_out = os.path.join(OUT, "flath2_SUMMARY.json")
assert DRY or not os.path.exists(fp_out), "write-once: %s exists" % fp_out
src = DRY_IN if DRY else OUT
recA, recE, shas = {}, {}, {}
for c in DEV_A:
    p = os.path.join(src, "flath2_A_%s.json" % c)
    assert os.path.exists(p), "missing record %s" % p
    r = json.load(open(p, encoding="utf-8"))
    if not DRY:
        assert r["preregistration"]["sha256"] == pre_sha and r["code"]["sha256"] == main_sha, c
    recA[c], shas["flath2_A_%s.json" % c] = r, sha_file(p)
for ds in FRESH:
    p = os.path.join(src, "flath2_E_%s.json" % ds)
    assert os.path.exists(p), "missing record %s" % p
    r = json.load(open(p, encoding="utf-8"))
    assert r["dry"] == DRY
    if not DRY:
        assert r["preregistration"]["sha256"] == pre_sha and r["code"]["sha256"] == main_sha, ds
        for k in ("json", "npz"):
            assert sha_file(os.path.join(REPO, r["stage_G"][k]["path"])) == r["stage_G"][k]["sha256"], (ds, k)
    recE[ds], shas["flath2_E_%s.json" % ds] = r, sha_file(p)

# ---------------------------------------------------------------- fresh one-shot transfers (the pre-registered headline)
fresh = {}
for ds in FRESH:
    r = recE[ds]
    fresh[ds] = {"eval_split": r["eval_split"], "n_split_A": r["n_split_A"], "N": r["N"], "D1": brief(r[KEY_D1]),
                 "curve (FLAT_ALL / FLAT_H2_ALL by M)": {m: [v["FLAT_ALL"], v["FLAT_H2_ALL"]] for m, v in r["curve (fixed M; M_MAIN is D1, the others descriptive)"].items()},
                 "patch_nodes_mean_at_M_MAIN": r["patch_at_M_MAIN"]["patch_nodes (novel beam nodes served)"]["mean"],
                 "gold_accounting_at_M_MAIN": r["curve (fixed M; M_MAIN is D1, the others descriptive)"][str(M_MAIN)]["gold_accounting"],
                 "strata_D1": r[KEY_D1]["strata"]}
lab = {ds: fresh[ds]["D1"]["label"] for ds in FRESH}
if any(v == "LOSS" for v in lab.values()):
    cross = "NOT_UNIVERSAL"
    cross_txt = "the graph patch significantly HURTS at a fixed node budget on %s" % ", ".join(ds for ds in FRESH if lab[ds] == "LOSS")
elif lab["webqsp"] == "GAIN":
    cross = "FLAT_BALANCED_H2_UNIVERSAL"
    cross_txt = ("no fresh corpus loses and the NAME_ONLY KB (webqsp) gains significantly: flat semantic retrieval + one block-equivalent "
                 "of bounded real-edge expansion keeps the text advantage and rescues the KB without partitions")
else:
    cross = "NO_KB_GAIN"
    cross_txt = "no fresh corpus loses, but webqsp does not gain significantly: the patch alone does not rescue the fresh KB"

# ---------------------------------------------------------------- DEV_A diagnostics (non-selecting)
devA = {}
for c in DEV_A:
    r = recA[c]
    devA[c] = {"partition": r["partition"], "n_split_A": r["n_split_A"], "budget_B_q_mean": r["budget_B_q (PRIMARY per-query exposure)"]["mean"],
               "D2 (PRIMARY vs FLAT_H2 at B_q)": brief(r[KEY_D2]), "D3 (FLAT vs FLAT_H2 at B_q)": brief(r[KEY_D3]),
               "fixed_M (FLAT_ALL / FLAT_H2_ALL)": {m: [v["FLAT_ALL"], v["FLAT_H2_ALL"]] for m, v in r["fixed_M (descriptive)"].items()},
               "patch_nodes_mean_at_B_q": r["patch_at_B_q"]["patch_nodes (novel beam nodes served)"]["mean"],
               "patch_nodes_from_hop1_mean_at_B_q": r["patch_at_B_q"]["patch_nodes_from_hop1"]["mean"],
               "gold_accounting_at_B_q": r["gold_accounting_at_B_q"], "strata_D2": r[KEY_D2]["strata"], "strata_D3": r[KEY_D3]["strata"]}
kb = [devA[c]["D2 (PRIMARY vs FLAT_H2 at B_q)"]["label"] for c in KBS_A]
if all(x == "LOSS" for x in kb):
    kb_read, scen = "PARTITIONS_BEAT_FLAT_H2_ON_THE_DEV_A_KB", SCENARIO_2
elif all(x in ("NEUTRAL", "GAIN") for x in kb):
    kb_read, scen = "PARTITIONS_NOT_SHOWN_NECESSARY_ON_THE_DEV_A_KB", SCENARIO_1
else:
    kb_read, scen = "MIXED_BETWEEN_PARTITIONINGS", None
res = {"record": "FLAT_BALANCED_H2_SUMMARY", "dry": DRY, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "question (the user's ruling, verbatim)": "Are partitions actually necessary, or is the KB gain just graph expansion/co-location?",
       "preregistration": {"path": "results/L1_COVPART/PREREGISTRATION_FLAT_BALANCED_H2.json", "sha256": pre_sha},
       "code": {"path": "scratchpad/_l1c_flath2_summary.py", "sha256": CODE_SHA}, "records_read (sha256)": shas,
       "FRESH_VERDICT (pre-registered: D1 labels on the three untouched populations)": {"verdict": cross, "reading": cross_txt, "D1_labels": lab},
       "fresh (split A; split B sealed)": fresh,
       "DEV_A_KB_READING (non-selecting diagnostic: D2 on metaqa under both partitionings)": {"reading": kb_read, "D2_labels": dict(zip(KBS_A, kb)),
                                                                                             "matches": scen},
       "DEV_A (non-selecting; no selection, no adoption)": devA,
       "not_done": ["no partition arm on webqsp / hotpotqa / 2wiki (no canonical partition exists; not required by the ruling)",
                    "split B of every fresh population sealed; DEV_B not read", "no adoption: any change to the frozen L1 is the user's ruling"]}
if DRY:
    print(json.dumps({"FRESH_VERDICT": cross, "DEV_A_KB_READING": kb_read, "bytes": len(json.dumps(res))}), flush=True)
    print("DRY RUN: summary assembled from dry records (STAND-IN gold) and NOT written.", flush=True)
    sys.exit(0)
assert sha_file(os.path.abspath(__file__)) == CODE_SHA
with open(fp_out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=True)
print("FRESH_VERDICT %s -- %s" % (cross, cross_txt), flush=True)
for ds in FRESH:
    d = fresh[ds]["D1"]
    print("  D1 %-8s FLAT@%d %.4f  FLAT_H2@%d %.4f  %+6.2f pts  +%d/-%d p=%.3g  %s" % (ds, M_MAIN, d["ALL_reference"], M_MAIN, d["ALL_arm"],
                                                                                 d["points_arm_minus_reference"], d["gained"], d["lost"], d["p"], d["verdict"]), flush=True)
print("DEV_A_KB_READING %s" % kb_read, flush=True)
for c in DEV_A:
    for k in ("D2 (PRIMARY vs FLAT_H2 at B_q)", "D3 (FLAT vs FLAT_H2 at B_q)"):
        d = devA[c][k]
        print("  %s %-10s ref %.4f arm %.4f %+6.2f pts +%d/-%d p=%.3g %s" % (k[:2], c, d["ALL_reference"], d["ALL_arm"], d["points_arm_minus_reference"],
                                                                         d["gained"], d["lost"], d["p"], d["verdict"]), flush=True)
print("-> %s sha256 %s" % (os.path.relpath(fp_out, REPO).replace("\\", "/"), sha_file(fp_out)), flush=True)
