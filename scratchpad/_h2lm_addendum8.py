"""FBX_SCALE declaration addendum 8 (write-once): the run-to-run NOISE FLOOR of the addendum-1 gate on WebQSP, declared BEFORE any replicate is partitioned or scored.

Addendum 7 left both nested (two-level) arms one row over the pass line (8 rows below PHG_SK against a line of 7) and noted that no run-to-run noise floor of the flat PHG was measured, so the
one-row miss cannot be judged against one.  This addendum measures it: the frozen flat PHG recipe on the frozen H4_SK_K hypergraph, re-run with ONE declared change that perturbs the
partition without changing the algorithm, scored against the frozen PHG_SK maps with addendum 1's rule, unchanged.  It changes no gate, no earlier verdict and no earlier record.
Usage: python scratchpad/_h2lm_addendum8.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_8.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % i for i in ("1", "2", "3", "4", "5", "6", "6A", "7")]
CODE = ["scratchpad/_h2lm_noise.py", "scratchpad/_h2lm_eval.py", "scratchpad/_h2l.py", "scratchpad/_ml2_eval.py", "scratchpad/_fbx_calib.py", "scratchpad/_l1h_host.py", "src/l1_canonical/hypergraph.py",
        "src/l1_lowmem/phg.py", "src/l1_lowmem/phg_repair.py", "src/l1_lowmem/freight.py", "src/l1_lowmem/phg_driver/phg_driver.c"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    rec = {
        "stage": "FBX_SCALE / addendum 8: run-to-run noise floor of the addendum-1 gate (flat PHG recipe, one declared change, WebQSP), declared before any replicate is partitioned or scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism and nothing here changes a gate",
        "question": "how many of the 786 rows does a SAME-ALGORITHM replicate of the frozen flat PHG lose to the frozen PHG_SK under addendum 1's rule (rows below PHG_SK's ALL-gold count at matched B_P, every B_N, "
                    "K in {100, 250, 500})?  That count is the run-to-run floor of the gate's statistic; addendum 7's nested arms scored 8 against a pass line of 7.",
        "not_claimed": ["no change to addendum 1's thresholds (at most 7 = RECALL_PRESERVING, at least 24 = FAILED, else BORDERLINE) and no re-scoring or re-labelling of any earlier arm: the addendum-7 verdicts stand as recorded",
                        "the replicates are not candidate partitioners and nothing is adopted from them",
                        "two replicates estimate a floor, not a distribution: the result is 'a same-algorithm replicate scored X', never a confidence interval"],
        "arms": {
            "reference": "PHG_SK = the frozen flat WebQSP maps on H4_SK at K in {100, 250, 500} (Zoltan PHG, NP 4); their cells must reproduce the earlier EVAL (CALIB_EVAL_WEBQSP__ml2_v1) bit for bit (control)",
            "SKR1": "the frozen driver, parameters, hypergraph, validity rule and empty-block repair, NP 4, with PHG_RANDOMIZE_INPUT = 1 (the frozen value is 0) and nothing else changed; DETERMINISTIC stays 1",
            "SKN3": "the same with NP = 3 (rank count; the frozen contract notes that partitions are not assumed identical across rank counts) and every parameter at its frozen value",
            "hypergraph_control": "the K-way hypergraph read by each replicate has the frozen H4_SK content digest at each K (scratchpad/_h2l.py CONTROL, EQUAL at K 100, 250, 500 in addendum 7)"},
        "gate": "addendum 1's verdict rule, unchanged, via scratchpad/_h2lm_eval.py (scratchpad/_ml2_eval.py's arithmetic, the 786-row population, matched-B_P comparison): per replicate and K, rows below PHG_SK at each "
                "B_N in {100, 250, 500, 1000, 2000, 5000}; verdict = max over all cells: at most 7 RECALL_PRESERVING, at least 24 FAILED, else BORDERLINE; also FAILED if B_P = K on more than half the rows",
        "reading_rule": {
            "statistic": "F = the larger of the two replicates' maximum cell losses (rows below PHG_SK) over all K and B_N; also the per-cell table, the rows gained (above PHG_SK) and the descriptive two-sided churn, as in addendum 7",
            "F >= 8": "the pass line of 7 lies inside the flat recipe's own replicate-to-replicate variation on this population: the gate as written cannot certify a same-algorithm replicate of the reference, and the "
                      "8-row results of the nested arms are not distinguishable from re-running PHG_SK; reported to the user with the tables.  The gate is NOT relaxed here: any change of threshold is the user's ruling",
            "F <= 7": "both replicates are RECALL_PRESERVING: the gate is attainable by the flat recipe, and the nested arms' 8 is a measured miss of the gate as written; the BORDERLINE verdicts stand and the options of "
                      "addendum 7 (A/B/C) remain the user's",
            "any replicate FAILED (>= 24)": "reported; it would show the floor is far above the pass line and the gate's statistic is dominated by partition-to-partition variation",
            "PARTITION_INVALID cell": "reported as not scored (as addendum 1); never relaxed"},
        "measured_constants_carried_in": {"flat_job": "WebQSP H4_SK, 29.57M pins, NP 4: ~129 B/pin RSS, ~187-345 s partition + 50-95 s load per K", "addendum_7_gate_result": "H2L5 and H2L25 BORDERLINE, max cell loss 8"},
        "not_done_here": ["the top-level surrogate for the nested partition (addendum 9, declared after the user's ruling on addendum 7 and after the net-size survey)", "any Freebase partition",
                          "K above 500", "a third replicate or any other perturbation after seeing the result"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
