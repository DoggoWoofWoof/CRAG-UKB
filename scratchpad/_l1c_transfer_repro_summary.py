"""Summary of the streamed-stage-2 identity test (PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json, acceptance gate g1-g3) over the
five DEV_A caches -> results/L1_COVPART/transfer_repro_SUMMARY.json (write-once).  Reads the five transfer_repro_<cache>.json records only."""
import hashlib
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(REPO, "results", "L1_COVPART")
CACHES = ["metaqa", "metaqa_phg", "squad", "squad_phg", "musique"]
PRE = os.path.join(OUT, "PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json")
fp_out = os.path.join(OUT, "transfer_repro_SUMMARY.json")
assert not os.path.exists(fp_out), "write-once"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


pre = json.load(open(PRE, encoding="utf-8"))
recs, per = {}, {}
for c in CACHES:
    p = os.path.join(OUT, "transfer_repro_%s.json" % c)
    r = json.load(open(p, encoding="utf-8"))
    assert r["mode"] == "REPRODUCE_DEV_A" and r["code"]["sha256"] == pre["code"]["new_modules"]["_l1c_transfer_composite.py"]["sha256"]
    assert r["code"]["stage2_module"]["sha256"] == pre["code"]["new_modules"]["_l1c_transfer_blockmax.py"]["sha256"]
    recs[c] = {"path": os.path.relpath(p, REPO).replace("\\", "/"), "sha256": sha_file(p)}
    d, g, s = r["stage2"]["divergence_streamed_vs_pinned"], r["streamed_stage2_gate"], r["streamed_vs_pinned_stage2"]
    per[c] = {"n_split_A": r["n_split_A"], "npart": len(r["arms"]) and None,
              "stage2_ranking_bitwise_identical": d["stage2_ranking_bitwise_identical"],
              "score_entries_differing / total": [d["score_entries_differing"], d["score_entries_total"]],
              "max_abs_score_difference": d["max_abs_score_difference"], "max_ulp": d["max_score_difference_in_fp32_ulps"],
              "rows_with_a_different_stage2_order": d["rows_with_a_different_stage2_order"],
              "rows_first_200_fused_differ": d["rows_whose_first_200_fused_QMAX_F0_blocks_differ"],
              "rows_P50_set_differs (all / A)": d["rows_whose_fused_P50_SET_differs (all rows / split A)"],
              "split_A_rows_flipped QMAX_F0 / QMAX_SAFE / PRIMARY": [sum(s["arms"][a]["rows_flipped (streamed covers, pinned not / pinned covers, streamed not)"]) for a in ("QMAX_F0", "QMAX_SAFE", "QMAX_BALANCED_H2_PATCH1")],
              "rows_with_identical_PRIMARY_served_set": s["rows_with_identical_PRIMARY_served_set"],
              "ALL PRIMARY streamed / pinned": [s["arms"]["QMAX_BALANCED_H2_PATCH1"]["ALL_streamed_stage2"], s["arms"]["QMAX_BALANCED_H2_PATCH1"]["ALL_pinned_stage2"]],
              "section24_cell streamed / record": [s["section24_cell_streamed"], s["section24_cell_pinned_record"]],
              "gate": {k: g[k] for k in g if k.startswith("g") or k.startswith("STREAMED")},
              "seconds": r["seconds"], "streamed_stage2_seconds": r["stage2"]["streamed"]["seconds"], "pinned_stage2_seconds": r["stage2"]["pinned"]["seconds"],
              "reference_reproduced": r["reference_reproduced"]}
    del per[c]["npart"]
accepted = all(per[c]["gate"]["STREAMED_STAGE2_ACCEPTED_ON_THIS_CACHE"] for c in CACHES)
res = {"RECORD": "TRANSFER_REPRO_SUMMARY", "STATUS": "STREAMED_STAGE2_ACCEPTED" if accepted else "STREAMED_STAGE2_NOT_ACCEPTED",
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "rule": pre["streamed_stage2_identity_test (runs BEFORE any transfer number; results/L1_COVPART/transfer_repro_<cache>.json, write-once)"]["acceptance_gate (fixed now; per cache)"],
       "prediction_stated_in_the_preregistration": pre["streamed_stage2_identity_test (runs BEFORE any transfer number; results/L1_COVPART/transfer_repro_<cache>.json, write-once)"]["prediction"],
       "per_cache": per, "records": recs, "preregistration": {"path": "results/L1_COVPART/PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json", "sha256": sha_file(PRE)},
       "what_follows": ("the streamed stage 2 is the transfer's stage 2 (accepted on all five DEV_A caches by the pre-registered gate); the transfer itself waits for a resource decision "
                        "(TRANSFER_RESOURCE_SURVEY_2WIKI_HOTPOTQA.json: PHG NP = 4 infeasible on this host for both corpora)" if accepted else
                        "the streamed stage 2 is NOT accepted; the transfer is not run under this pre-registration; nothing is tuned; a new ruling is needed")}
with open(fp_out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1)
print(json.dumps({c: {"accepted": per[c]["gate"]["STREAMED_STAGE2_ACCEPTED_ON_THIS_CACHE"], "max_abs": per[c]["max_abs_score_difference"], "ulp": per[c]["max_ulp"],
                      "rows_stage2_order": per[c]["rows_with_a_different_stage2_order"], "rows_first200": per[c]["rows_first_200_fused_differ"],
                      "flips": per[c]["split_A_rows_flipped QMAX_F0 / QMAX_SAFE / PRIMARY"], "ALL": per[c]["ALL PRIMARY streamed / pinned"], "s": per[c]["seconds"]} for c in CACHES}, indent=1))
print(res["STATUS"], "->", fp_out, sha_file(fp_out)[:12])
