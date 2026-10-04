"""FBX_SCALE declaration addendum 9 (write-once): Track B, a SURROGATE for the TOP level of the nested partition on WebQSP, declared BEFORE any surrogate top is partitioned or scored.

Addendum 7: the nested (two-level) partition's K-way stage costs about 1/S of the flat memory, but its TOP stage (a flat S-way PHG) still reads every pin of H4_SK.  The user's ruling wanted the
expensive global step on something that fits; M2 coarsening alone cannot give that (its pin reduction saturates near 3.9x on WebQSP: REPORT §49/§50), and the net-size survey
(results/FREEBASE_SCALE/h2l/H2LM_NETSIZE_SURVEY__webqsp__v1.json) shows where the pins are: nets of more than 8 pins hold 54 % (STRUCT) / 29 % (KNN) of the pins and 8 % of the weighted signal.  This
addendum asks how far the top hypergraph can be shrunk by those two levers, singly and together, before the nested partition's ALL-gold recall leaves addendum 1's gate.
It changes no gate and no earlier record.   Usage: python scratchpad/_h2lt_addendum9.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_9.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % i for i in ("1", "2", "3", "4", "5", "6", "6A", "7", "8")]
SURVEY = FB + "/h2l/H2LM_NETSIZE_SURVEY__webqsp__v1.json"
CODE = ["scratchpad/_h2lt.py", "scratchpad/_h2lt_eval.py", "scratchpad/_h2l.py", "scratchpad/_ml_run.py", "scratchpad/_ml_coarsen.py", "scratchpad/_ml2_eval.py", "scratchpad/_fbx_calib.py",
        "scratchpad/_l1h_host.py", "src/l1_canonical/hypergraph.py", "src/l1_lowmem/phg.py", "src/l1_lowmem/phg_repair.py", "src/l1_lowmem/phg_driver/phg_driver.c", "src/l1_lowmem/phg_driver/phg_driver_vw.c"]
ARMS = ["N16", "N8", "N4", "Q512L2", "Q512L6", "Q512L6N8", "Q512L6N4"]


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
        "stage": "FBX_SCALE / addendum 9: surrogate top level for the nested partition (WebQSP, S = 25), declared before any surrogate top is partitioned or scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "basis": {SURVEY: sha(SURVEY)},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism and nothing here changes a gate",
        "question": "how much can the pins of the TOP hypergraph be reduced -- by a net-size cap, by the frozen M2 quotient, or both -- before the nested partition's ALL-gold recall (K in {100, 250, 500}) leaves "
                    "addendum 1's gate?  The output is a frontier (pin reduction vs verdict); Freebase needs a top of about 20x fewer pins than H4_SK (6e9 pins against ~0.3e9 that fit the 81.6 GB host at ~190 B/pin), "
                    "which is reported against every arm.",
        "not_claimed": ["no Freebase partition and no claim that any arm fits Freebase: 20x is the requirement, the WebQSP reduction is the measurement",
                        "no change to addendum 1's thresholds and no re-scoring of any earlier arm; the addendum-7 flat-top arm H2L25 (BORDERLINE, max loss 8) is the in-memory reference, not re-run",
                        "no refinement and no other S: the regime that fits the host is S >= 25 (addendum 7 measured memory), so S = 25 only"],
        "construction": {
            "top_hypergraph": "the frozen H4_SK rule at k = S = 25 (cap = round(N / 25); exactly the addendum-7 top hypergraph, content digest checked by file hash against its record)",
            "surrogates (spec names of scratchpad/_h2lt.py)": {
                "N<c>": "drop every net of more than c pins; weights of the remaining nets unchanged",
                "Q<W>L<l>": "the exact quotient under the frozen M2 clustering results/FREEBASE_SCALE/ml2/clusters__M2_W<W>_L<l>.npy (cluster map hash checked against its COARSEN record); vertex weight = cluster size; identical nets merged with summed weight",
                "Q<W>L<l>N<c>": "first the net cap, then the quotient"},
            "arms": ARMS,
            "arm_rationale": "caps 16 / 8 / 4 from the survey (pins kept x1.44 / x1.69 / x2.58, signal kept 0.973 / 0.920 / 0.723; 0 / 2 / 10,487 vertices left without a net); Q512L2 and Q512L6 are the M2 levels with 1.93x and 3.30x "
                             "pin reduction whose cluster size (<= 512) is below Zoltan's 1 % tolerance at S = 25 (1,037 vertices); the two combinations take the cap at 8 and 4 on the most coarse W512 level.  Fixed before any arm is partitioned; "
                             "no spec is added, removed or altered after seeing a result",
            "top_partition": "the frozen vertex-weighted driver (scratchpad _ml_run vw_build; NP 4; unit weights for N<c> arms) at k = 25 on the surrogate hypergraph, IMBALANCE_TOL 1.01 (as addendum 7's flat top); projected to the vertices; "
                             "the frozen validity rule on the FULL top hypergraph (largest block <= ceil(1.03 N / 25)), frozen empty-block repair; a violating top is PARTITION_INVALID (arm invalid at every K, recorded, never relaxed); no refinement",
            "sub_stage": "unchanged from addendum 7 (scratchpad/_h2l.py cmd_sub): net-split K-way sub-problems from the frozen K-way hypergraph, tolerance floor_4dp(1.03 (N/K)(K/25)/n_b), frozen validity and repair",
            "controls": ["scratchpad/_h2lt.py TEST passes before launch (net cap keeps exactly the nets of <= c pins; the capped hypergraph quotients; spec names parse)",
                         "the K-way hypergraph read by every SUB has the frozen H4_SK content digest (asserted by _h2l.cmd_sub, as in addendum 7)",
                         "the PHG_SK cells of the reference reproduce CALIB_EVAL_WEBQSP__ml2_v1 bit for bit"]},
        "gate": "addendum 1's verdict rule, unchanged, via scratchpad/_h2lt_eval.py: rows below PHG_SK's ALL-gold count at matched B_P, every B_N in {100, 250, 500, 1000, 2000, 5000}, at K in {100, 250, 500}; "
                "at most 7 of 786 RECALL_PRESERVING, at least 24 FAILED, else BORDERLINE; also FAILED if B_P = K on more than half the rows",
        "reported_for_every_arm": ["top pins and pin reduction against the top hypergraph; top Zoltan seconds and peak RSS per rank", "KM1 of the projected top on the full top hypergraph relative to the flat top of addendum 7",
                                  "the gate table (rows below PHG_SK per K and B_N), the verdict, the descriptive two-sided churn", "the same-algorithm replicate floor F of addendum 8 beside each arm's maximum loss (descriptive only)",
                                  "Freebase arithmetic: top pins at 6e9 / reduction, and the reduction still missing to reach 20x"],
        "decision_table": {
            "an arm is RECALL_PRESERVING": "that pin reduction is recall-preserving for the top; the LARGEST such reduction is the frontier; if it is >= 20x the top surrogate is adopted for the nested design and the next addendum scales it; "
                                          "otherwise the gap to 20x is reported and the top needs a further lever (the user's ruling: candidates are a streaming top with a 1.01 balance bound, net sampling, or a sparser quotient)",
            "an arm is BORDERLINE": "reported with its table beside F; the decision is the user's",
            "an arm is FAILED (>= 24)": "that reduction is refused for the top (no map adopted); arms with larger reductions built from the same lever are expected to fail and are still scored as declared",
            "PARTITION_INVALID": "counted as not scored (as addendum 1); never relaxed"},
        "not_done_here": ["any Freebase partition", "a streaming top, net sampling, or any refinement", "K above 500", "any spec or S beyond the declared seven arms at S = 25"],
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
