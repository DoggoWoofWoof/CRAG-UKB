"""FBX_SCALE declaration addendum 16 (write-once): Track B, the INTERIOR of the V-cycle frontier on WebQSP, declared BEFORE any of its tops is partitioned, sub-partitioned or scored.

Addendum 14 (REPORT §58) found, for the V-cycle top with FM refinement at every level of the M2_W512 ladder (S = 25, K in {100, 250, 500}, 786 rows, addendum 1's gate): 3.3x -> 4 rows RECALL_PRESERVING,
6.3x -> 7 RECALL_PRESERVING (on the line), 14.2x -> 14 BORDERLINE (label-propagation control 15).  Freebase needs the coarsest surrogate to hold <= ~0.39e9 pins: 10.9x for the STRUCT-only family
(4.25e9 pins), 15.4x for SK (6e9 pins).  The interior between 6.3x and 14.2x was not scored, and so far the thinning was a single axis (the net cap at the coarsest ladder level 6).  Two axes decide the
surrogate (scratchpad/_h2lt.py spec grammar Q<W>L<l>N<c>): the net cap c and the ladder level l (= the number of clusters of the quotient: 131,547 / 37,560 / 11,639 / 5,913 at l = 3 / 4 / 5 / 6).
A LABEL-FREE survey of the pins of every (c, l) (results/FREEBASE_SCALE/vc/PIN_SURVEY__v1.json, no Zoltan, no gold label) gave the reduction of each combination; this addendum declares three arms from it.
It changes no gate and no earlier record.   Usage: python scratchpad/_vc_addendum16.py   (refuses to overwrite)
"""
import glob
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_16.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % i for i in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10", "11", "12", "13", "14", "15")]
SURVEY = FB + "/vc/PIN_SURVEY__v1.json"
EVAL14 = FB + "/CALIB_EVAL_WEBQSP__vc_v1.json"
CODE = ["scratchpad/_vc.py", "scratchpad/_vc_ref.c", "scratchpad/_vc_ref2.c", "scratchpad/_vc_sub.py", "scratchpad/_vc_subk.py", "scratchpad/_vc16_eval.py", "scratchpad/_vc_pinsurvey.py", "scratchpad/_vc16_run.py",
        "scratchpad/_h2lt.py", "scratchpad/_h2lt_eval.py", "scratchpad/_h2l.py", "scratchpad/_ml_run.py", "scratchpad/_ml_coarsen.py", "scratchpad/_ml2_eval.py", "scratchpad/_fbx_calib.py",
        "scratchpad/_fbx_part.py", "scratchpad/_l1h_host.py", "scratchpad/_host_yield.py", "src/l1_canonical/hypergraph.py", "src/l1_lowmem/phg.py", "src/l1_lowmem/phg_repair.py",
        "src/l1_lowmem/phg_driver/phg_driver.c", "src/l1_lowmem/phg_driver/phg_driver_vw.c"]
# (arm, spec, cap, level, role)
ARMS = [("VCQ512L5N4FM2", "Q512L5N4", 4, 5, "the same cap as the borderline 14.2x arm on a quotient with 2.0x more clusters (11,639 vs 5,913): does the cluster count carry the 14 rows"),
        ("VCQ512L4N4FM2", "Q512L4N4", 4, 4, "cap 4 on a quotient with 6.4x more clusters (37,560): the finer end of the cluster-count axis, still above the STRUCT-only requirement"),
        ("VCQ512L6N5FM2", "Q512L6N5", 5, 6, "the cap axis at the coarsest quotient: the interior between the preserving 6.3x (cap 8) and the borderline 14.2x (cap 4); below the STRUCT-only requirement, it locates the frontier")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    sv = json.load(io.open(SURVEY, encoding="utf-8"))
    assert sv["label_free"] and sv["gold_labels_read"] == 0
    red = {}
    for r in sv["survey"]:
        for lv, v in r["levels"].items():
            red[(r["cap"], int(lv))] = {"clusters": v["clusters"], "pins": v["pins"], "pin_reduction_vs_top": v["pin_reduction_vs_top"], "freebase_pins": v["freebase_pins_at_this_reduction"], "fits_0.39e9": v["fits_0.39e9"]}
    arms = []
    for a, s, c, lv, role in ARMS:
        x = red[(c, lv)]
        arms.append({"arm": a, "spec": s, "refiner": "fm2", "net_cap": c, "ladder_level": lv, "clusters": x["clusters"], "surrogate_pins": x["pins"], "surrogate_pin_reduction": x["pin_reduction_vs_top"],
                     "freebase_coarsest_pins": x["freebase_pins"], "fits_0.39e9": x["fits_0.39e9"], "role": role})
    rec = {
        "stage": "FBX_SCALE / addendum 16: the interior of the V-cycle frontier for the nested partition (WebQSP, S = 25), declared before any of its tops is partitioned, sub-partitioned or scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "basis": {"addendum-14 evaluation (REPORT §58)": {EVAL14: sha(EVAL14)}, "label-free pin survey (no Zoltan, no gold label)": {SURVEY: sha(SURVEY)},
                  "addendum-14 results": {"VCQ512L6FM2 3.29x": "max loss 4 RECALL_PRESERVING", "VCQ512L6N8FM2 6.33x": "max loss 7 RECALL_PRESERVING (on the line)", "VCQ512L6N4FM2 14.15x": "max loss 14 BORDERLINE",
                                          "VCQ512L6N4LP2 14.15x": "max loss 15 BORDERLINE (control)", "floor": "F = 8 (addendum 8)"},
                  "observation_that_motivates_the_arms": "the nested maps of addendum 14 have a LOWER weighted KM1 on the full K-way hypergraph than the flat PHG_SK at every K (0.93 - 0.99 x for the FM2 arms), "
                                                         "yet lose rows at 14.2x: KM1 is not what the loss tracks there; the loss follows what the surrogate thins (REPORT §55.3 (i), §58.3 (iv)), hence the two thinning axes below"},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism and nothing here changes a gate",
        "question": "where between 6.3x (preserving) and 14.2x (borderline) does the V-cycle top stop preserving, and does a finer quotient (more clusters) with the same net cap 4 keep the gate at a reduction that still fits the "
                    "STRUCT-only Freebase top (>= 10.9x)?",
        "not_claimed": ["no Freebase partition and no claim that a WebQSP reduction transfers", "no streaming implementation (the in-memory refiner of addendum 14 is used unchanged)",
                        "no change to addendum 1's thresholds and no re-scoring of any earlier arm", "no S other than 25, no K above 500, no spec beyond the three arms below, no LP arm"],
        "construction": {
            "unchanged_from_addendum_14": ["the V-cycle (scratchpad/_vc.py: project the surrogate labels from the quotient level down to level 0, refine at every level on the exact quotient of the FULL top hypergraph with FM2)",
                                           "the refiner scratchpad/_vc_ref2.c mode 1 (6 passes, stop after 500 non-improving moves, per-block cap int(1.01 N / 25) = 104,752)", "the nested K-way stage (scratchpad/_vc_sub.py -> _h2l.cmd_sub)",
                                           "the frozen M2_W512 ladder, the validity rule (largest block <= ceil(1.03 N / 25), no empty block), the evaluator arithmetic and the gate"],
            "new_in_this_addendum": "only the surrogate: the stored addendum-9 tops cover (c, l) in {(8,6), (4,6), (none,6)}; this addendum partitions THREE NEW surrogate tops with the frozen vertex-weighted driver "
                                    "(scratchpad/_h2lt.py TOP <spec> 25: NP 4, IMBALANCE_TOL 1.01, net cap first then the quotient), then runs the V-cycle on each (scratchpad/_vc.py D1 / TOP), exactly as addendum 14",
            "arms": arms, "arm_names": "VC<spec><REFINER upper-case>, scored as H2LT_<arm>_25",
            "sequence": "scratchpad/_vc16_run.py TOPS (per spec, resumable by existing records: _h2lt TOP, _vc D1, _vc TOP) -> scratchpad/_vc_subk.py <K> 25 <arms> for K in 100, 250, 500 (one job per K) -> "
                        "scratchpad/_vc16_eval.py EVAL vc16_v1 v1 vc_v1 H2LT_<arm>_25,... (the only step that reads gold; control = the addendum-14 EVAL, whose PHG cells it must reproduce bit for bit)",
            "controls": ["each TOP step asserts the stored surrogate top is constant on the quotient clusters and that the V-cycle KM1 equals its D1 record", "the PHG_SK cells of the evaluation reproduce CALIB_EVAL_WEBQSP__vc_v1 bit for bit"],
            "validity": "a surrogate top or V-cycle top that violates the frozen rule is PARTITION_INVALID: the arm has no map at any K, is counted as not scored, never relaxed, no repair beyond the frozen empty-block repair of the K-way stage"},
        "gate": "addendum 1's verdict rule, unchanged: rows below PHG_SK's ALL-gold count at matched B_P, every B_N in {100, 250, 500, 1000, 2000, 5000}, K in {100, 250, 500}; at most 7 of 786 RECALL_PRESERVING, "
                "at least 24 FAILED, else BORDERLINE; also FAILED if B_P = K on more than half the rows",
        "freebase_requirement": {"coarsest_surrogate_pin_budget": 0.39e9, "STRUCT_only_pins": 4.25e9, "SK_pins": 6.0e9, "needed_reduction_STRUCT_only": round(4.25e9 / 0.39e9, 2), "needed_reduction_SK": round(6.0e9 / 0.39e9, 2)},
        "reported_for_every_arm": ["surrogate pins / clusters / reduction (basis survey) and the Zoltan top's KM1 on the full top hypergraph; the V-cycle's per-level KM1 trajectory; KM1 of the nested K-way map relative to the flat PHG_SK",
                                  "the gate table (rows below PHG_SK per K and B_N), the verdict, the two-sided churn at the worst cell, F = 8 beside the maximum loss (descriptive only; relaxes nothing)",
                                  "the frontier table of ALL FM2 arms of addenda 14 and 16 sorted by pin reduction, with the STRUCT-only and SK requirements marked"],
        "decision_table": {
            "an arm with surrogate reduction >= 10.9x is RECALL_PRESERVING": "it is the reported Track B top for a Freebase STRUCT-only systems measurement (the largest such arm if several); the next addendum builds the streaming implementation, "
                                                                               "regresses it bit for bit against these records and runs that measurement; the SK requirement (15.4x) stays open and is stated",
            "no arm >= 10.9x is RECALL_PRESERVING and the best of them is BORDERLINE": "reported with its table beside F; the decision is the user's",
            "every arm >= 10.9x is FAILED": "the frontier is the largest RECALL_PRESERVING arm of addenda 14 and 16 and the gap to 10.9x / 15.4x is stated; the user rules on the next lever",
            "PARTITION_INVALID": "counted as not scored (as addendum 1); never relaxed"},
        "not_done_here": ["any Freebase partition", "a streaming implementation", "a net cap of 3 (366,406 vertices without a net) and any surrogate beyond the three arms (a later addendum if these preserve)", "K above 500", "an LP control"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))
    for a in arms:
        print(a["arm"], a["clusters"], a["surrogate_pins"], a["surrogate_pin_reduction"], a["freebase_coarsest_pins"])


if __name__ == "__main__":
    main()
