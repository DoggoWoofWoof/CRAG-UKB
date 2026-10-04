"""FBX_SCALE declaration addendum 14 (write-once): Track B, a V-CYCLE top level for the nested partition on WebQSP, declared BEFORE any V-cycle top is written as a record, sub-partitioned or scored.

Addendum 9 measured how far the top hypergraph can be shrunk before the nested partition's ALL-gold recall leaves addendum 1's gate: 1.93x at most (Q512L2); every surrogate of >= 2.6x FAILED (28-43 rows).  Its
diagnosis (REPORT §55): the loss tracks the surrogate partition's KM1 on the FULL top hypergraph (about 1.03 x the flat top is preserving, 1.14 x and above fails), i.e. the coarse partition is a poor solution of the
fine problem.  The user's ruling of 2026-10-01 (1A) is to build the out-of-core PHG engine: a semi-external multilevel scheme -- coarsest-level partition on a surrogate that fits memory, then REFINEMENT AT EVERY LEVEL
on the exact quotient of the FULL hypergraph (pins streamed, O(V) state in RAM).  Development step D1 (scratchpad/_vc.py D1; KM1 only, NO gold label read) measured that refinement on WebQSP: starting from the stored
addendum-9 surrogate tops, project up the frozen M2_W512 ladder (levels 6..1) and refine at each level with a k-way KM1 refiner (label propagation, or Fiduccia-Mattheyses with a gain cache).  This addendum asks the
gated question: does a V-cycle top keep the nested partition's recall inside addendum 1's gate at the pin reduction Freebase needs (about 15x)?
It changes no gate and no earlier record.   Usage: python scratchpad/_vc_addendum14.py   (refuses to overwrite)
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
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_14.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % i for i in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10", "11", "12", "13")]
D1 = sorted(glob.glob(FB + "/vc/webqsp__VC_*_top_k25.D1.json"))
EVAL9 = FB + "/CALIB_EVAL_WEBQSP__top_v1.json"
CODE = ["scratchpad/_vc.py", "scratchpad/_vc_ref.c", "scratchpad/_vc_ref2.c", "scratchpad/_vc_ref_test.py", "scratchpad/_vc_sub.py", "scratchpad/_vc_subk.py", "scratchpad/_vc_eval.py",
        "scratchpad/_h2lt.py", "scratchpad/_h2lt_eval.py", "scratchpad/_h2l.py", "scratchpad/_ml_run.py", "scratchpad/_ml_coarsen.py", "scratchpad/_ml2_eval.py", "scratchpad/_fbx_calib.py",
        "scratchpad/_fbx_part.py", "scratchpad/_l1h_host.py", "scratchpad/_host_yield.py", "src/l1_canonical/hypergraph.py", "src/l1_lowmem/phg.py", "src/l1_lowmem/phg_repair.py",
        "src/l1_lowmem/phg_driver/phg_driver.c", "src/l1_lowmem/phg_driver/phg_driver_vw.c"]
# (arm, spec, refiner, pin reduction of the surrogate top, role)
ARMS = [("VCQ512L6N4FM2", "Q512L6N4", "fm2", "14.2x", "the arm that matters: the reduction Freebase needs, FM refinement at every level"),
        ("VCQ512L6N8FM2", "Q512L6N8", "fm2", "6.3x", "frontier point between the preserving 1.9x of addendum 9 and 14.2x"),
        ("VCQ512L6FM2", "Q512L6", "fm2", "3.3x", "a top that addendum 9 FAILED unrefined (3.3x): does refinement alone repair it"),
        ("VCQ512L6N4LP2", "Q512L6N4", "lp2", "14.2x", "CONTROL for the refiner: the same top and level, label propagation only (cheaper, weaker); separates what FM adds from what the V-cycle adds")]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def d1_table():
    t = []
    for p in D1:
        d = json.load(io.open(p, encoding="utf-8"))
        t.append({"spec": d["spec"], "refiner": d["refiner"], "km1_on_full_top_hypergraph": d["final"]["km1_on_full_top_hypergraph"], "km1_ratio_vs_flat_top": d["final"]["km1_ratio_vs_flat_top"],
                  "km1_ratio_unrefined_stored_top": d["stored_top"]["km1_ratio_unrefined"], "max_block": d["final"]["blocks"]["max"], "validity": d["final"]["validity"]["gate"], "record": p, "record_sha256": sha(p)})
    return t


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    assert len(D1) == 7, D1
    rec = {
        "stage": "FBX_SCALE / addendum 14: V-cycle top level for the nested partition (WebQSP, S = 25), declared before any V-cycle top is written as a record, sub-partitioned or scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "basis": {"addendum-9 evaluation (the unrefined surrogate tops this addendum starts from)": {EVAL9: sha(EVAL9)}, "D1 development records (KM1 only; no gold label was read by any of them)": d1_table()},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism and nothing here changes a gate",
        "question": "does a partition built by the semi-external multilevel scheme (coarsest-level surrogate partition + refinement at every ladder level on the exact quotient of the full top hypergraph) keep the nested "
                    "partition's ALL-gold recall (K in {100, 250, 500}) inside addendum 1's gate at 14.2x fewer pins at the coarsest level -- the reduction addendum 9 showed unrefined tops cannot survive (every top of "
                    ">= 2.6x FAILED)?  Freebase needs about 15x (4.25e9 STRUCT pins -> 0.28e9; 6e9 SK pins -> 0.40e9, against the ~0.39e9 pins that fit the 73.4 GB budget at the addendum-9 figure of ~190 B/pin).",
        "not_claimed": ["no Freebase partition and no claim that a WebQSP reduction transfers: pins-per-net and hub structure differ on Freebase; the WebQSP measurement is the calibration, the Freebase run is a later addendum",
                        "no out-of-core implementation here: the refiner of this addendum (scratchpad/_vc_ref*.c) holds the level in RAM (WebQSP is 29.6M pins); the streaming implementation and its bit-for-bit regression "
                        "against these records is the next addendum (user ruling 1A)",
                        "no change to addendum 1's thresholds and no re-scoring of any earlier arm; the addendum-9 evaluation and the flat-top arm H2L25 (BORDERLINE, max loss 8) are the references, not re-run",
                        "no S other than 25, no K above 500, no spec beyond the four arms below"],
        "construction": {
            "starting_point": "the stored addendum-9 surrogate top of the spec (results/FREEBASE_SCALE/h2l/parts/webqsp__H4_H2LT_<spec>_top_k25__PHG_con, repair None, constant on the level-6 clusters -- asserted); "
                              "no new Zoltan run for the top level",
            "ladder": "the frozen M2_W512 cumulative cluster maps, levels 1..6 (results/FREEBASE_SCALE/ml2/clusters__M2_W512_L<l>.npy, hashes checked against COARSEN__M2_W512.json); level 0 = the top hypergraph itself "
                      "(N = 2,592,894 vertices, 5,181,600 nets, 29,574,350 pins, S = 25, cap = int(1.01 N / 25) = 104,752; the frozen validity bound ceil(1.03 N / 25) = 106,828)",
            "levels": "the exact quotient of the FULL top hypergraph (not of the net-capped surrogate) at every level: vertex weight = cluster size, identical nets merged with summed weight; level 6 has 5,913 vertices and "
                      "3.294x fewer pins than level 0",
            "v_cycle": "project the surrogate labels from level 6 down to level 0; at each level run the refiner on that level's quotient with the frozen per-block cap, then project to the next finer level",
            "refiners": {"fm2": "scratchpad/_vc_ref2.c mode 1: Fiduccia-Mattheyses on the k-way KM1 (connectivity-1) objective with an incremental gain cache, a max-heap with stale-key re-push, moves of negative gain allowed, "
                                "vertex locking per pass, rollback to the best prefix, 6 passes, stop after 500 consecutive non-improving moves; the end-of-run cache is checked against a rebuild",
                         "lp2": "scratchpad/_vc_ref2.c mode 0: sequential label-propagation sweeps, strict positive KM1 gain only, up to 10 passes; identical to scratchpad/_vc_ref.c mode 0 (v1) -- the D1 regression reproduced "
                                "the v1 KM1 of Q512L6N4 bit for bit (839,272,952)"},
            "invariants_checked_by_the_binaries": "the KM1 recomputed from the labels equals the reported value before and after (exit 4 on drift), the gain cache equals a rebuild (exit 5), no block that was within the cap "
                                                  "is pushed over it; unit tests scratchpad/_vc_ref_test.py (12 random weighted hypergraphs + a planted-partition recovery) PASS for both sources; scratchpad/_vc.py TEST PASS on the host "
                                                  "(projection chain equals the direct cumulative map; every level's quotient KM1 equals the fine KM1)",
            "validity": "the frozen rule on the FULL top hypergraph (largest block <= ceil(1.03 N / 25), no empty block); a violating V-cycle top is PARTITION_INVALID (the arm is invalid at every K, recorded, never relaxed); "
                        "no repair step is applied",
            "sub_stage": "unchanged from addendum 7 (scratchpad/_h2l.py cmd_sub, called by scratchpad/_vc_sub.py): net-split K-way sub-problems from the frozen K-way hypergraph, tolerance floor_4dp(1.03 (N/K)(K/25)/n_b), "
                         "frozen validity and repair; one job per K runs the arms in sequence (the sub-stage run directory does not carry the arm name)",
            "arms": [{"arm": a, "spec": s, "refiner": r, "surrogate_pin_reduction": red, "role": role} for a, s, r, red, role in ARMS],
            "arm_names": "the arm name is VC<spec><REFINER upper-case> (no underscore: the evaluator splits on underscores); scored as H2LT_<arm>_25",
            "controls": ["the K-way hypergraph read by every SUB has the frozen H4_SK content digest (asserted by _h2l.cmd_sub, as in addendum 7)",
                         "the PHG_SK cells of the evaluation reproduce CALIB_EVAL_WEBQSP__top_v1 bit for bit (control tag top_v1, bundle v1)",
                         "the D1 KM1 of each arm (basis table) is recomputed by the TOP step and must agree exactly: the TOP record stores it and the step refuses to write a map whose KM1 differs from the D1 record"]},
        "d1_label_free_expectation": {"statement": "written before any gold label is read, NOT a criterion: the label-free KM1 ratio is a proxy that earlier arms ordered correctly (<= 1.035 preserving: Q512L2 1.033 -> 4 rows, N16 0.992 -> 4; "
                                                   "N8 1.007 -> 10 BORDERLINE; >= 1.14 -> 28-43 rows FAILED).  By the proxy the three FM2 arms (0.981 / 1.005 / 0.979 x the flat-top KM1 for Q512L6N4 / Q512L6N8 / Q512L6) are expected RECALL_PRESERVING "
                                                   "and the LP2 control (1.060) between BORDERLINE and FAILED.  The gate decides; a mismatch is reported as a result about the proxy",
                                     "d1_table": "see basis"},
        "gate": "addendum 1's verdict rule, unchanged, via scratchpad/_vc_eval.py (a wrapper of the addendum-9 evaluator that changes only the declaration it cites): rows below PHG_SK's ALL-gold count at matched B_P, "
                "every B_N in {100, 250, 500, 1000, 2000, 5000}, at K in {100, 250, 500}; at most 7 of 786 RECALL_PRESERVING, at least 24 FAILED, else BORDERLINE; also FAILED if B_P = K on more than half the rows",
        "execution_order": ["python scratchpad/_vc.py TOP <spec> <refiner>  for the four arms (reads no gold label)", "python scratchpad/_vc_subk.py <K> 25 <arm> ...  for K in 100, 250, 500 (reads no gold label)",
                            "python scratchpad/_vc_eval.py EVAL vc_v1 v1 top_v1 H2LT_<arm>_25,...   (the only step that reads gold)"],
        "reported_for_every_arm": ["top pins and the pin reduction at the surrogate level; the V-cycle's per-level KM1 trajectory and seconds; peak resident memory of the refiner as a share of the level's pins",
                                  "KM1 of the V-cycle top on the full top hypergraph relative to the flat top", "the gate table (rows below PHG_SK per K and B_N), the verdict, the descriptive two-sided churn",
                                  "the same-algorithm replicate floor F = 8 of addendum 8 beside each arm's maximum loss (descriptive only; relaxes nothing)",
                                  "Freebase arithmetic: pins of the coarsest surrogate at 4.25e9 and 6e9 pins / reduction against the ~0.39e9 that fit"],
        "decision_table": {
            "the 14.2x FM2 arm is RECALL_PRESERVING": "a V-cycle top at the reduction Freebase needs keeps the recall: the semi-external multilevel scheme is adopted as the Track B partition architecture on WebQSP evidence; "
                                                       "the next addendum builds the streaming implementation, regresses it bit for bit against these records and runs a Freebase STRUCT-only systems measurement",
            "the 14.2x FM2 arm is BORDERLINE": "reported with its table beside F; the decision is the user's",
            "the 14.2x FM2 arm is FAILED (>= 24)": "refused at 14.2x; the largest preserving declared arm (6.3x or 3.3x) is the reported frontier and the gap to ~15x is stated; the user rules on the next lever",
            "the LP2 control": "descriptive: whether FM is needed or label propagation suffices decides which refiner the streaming implementation must build; it is never used to select an arm after the fact",
            "PARTITION_INVALID": "counted as not scored (as addendum 1); never relaxed"},
        "not_done_here": ["any Freebase partition", "a streaming implementation", "a surrogate more reduced than Q512L6N4 (that needs a new Zoltan top and a new declaration)", "K above 500", "any arm beyond the four declared"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
