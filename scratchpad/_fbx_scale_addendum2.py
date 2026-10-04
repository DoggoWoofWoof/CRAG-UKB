"""FBX_SCALE declaration addendum 2 (write-once): rung 3 of the partitioner ladder (balanced label propagation), declared and built BEFORE any LP recall is scored.

Addendum 1 fixed the ladder (hash -> streaming LDG/FENNEL -> balanced LP -> two-level PHG), the gate cells and the pass / fail rule.  Nothing in it changes.
This addendum (a) discloses what had been read before it was written, (b) defines R3 exactly, and (c) states that the ONE full-population EVAL scores hash, LDG and
LP together under the addendum-1 gate.

Usage: python scratchpad/_fbx_scale_addendum2.py            (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_2.json"
A1 = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_1.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    smoke = "results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__smoke200.json"
    lpm = "results/FREEBASE_SCALE/CALIB_MAPS_WEBQSP__lp_v1.json"
    rec = {
        "stage": "FBX_SCALE / addendum 2: rung 3 (balanced label propagation) declared and built before it is scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime()),
        "extends": {A1: sha(A1)},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism",
        "disclosure_of_what_was_read_first": {
            "what": "a 200-row SMOKE of the EVAL (the first 200 of the 786 split-A rows; record %s, sha256 %s) was run to test the pipeline and was read: on those 200 rows the PHG "
                    "regression prefix check passed exactly (B_P / contacted mass / LO equal on 200 of 200 rows, all 1,714 served and unrouted gold positions equal), and at MATCHED "
                    "B_P the LDG on graph S was 33 to 48 rows under PHG at K 100 / 250 / 500 (of 200) and balanced hash 110 under. The smoke is NOT a verdict (the gate is on 786 rows)." % (smoke, sha(smoke)),
            "consequence": "rung 3 is declared after that observation and before any LP recall has been computed on any population. The full-population EVAL scores R1, R2 and R3 in one run under the UNCHANGED addendum-1 gate; "
                           "the R3 definition below was fixed before its maps were scored and is not tuned afterwards (a failed R3 goes to rung 4 or to the user, it is not re-parameterised)."},
        "R3_definition": {
            "name": "balanced label propagation, a refinement of the R2 LDG block table (same graph, same adjacency streams, same memory class: 4 N bytes of labels + O(K) tables, CSRs memory-mapped)",
            "init": "the R2 LDG map on the same graph (LP_S from LDG_S on graph S; LP_SK from LDG_SK on graph SK); LDG's map hashes are checked against CALIB_MAPS_WEBQSP__v1.json before use",
            "balance": "cap = ceil(1.03 N / K) (the LDG cap) and floor lo = floor(0.97 N / K): a vertex may only enter a block with load < cap and only leave one with load > lo",
            "sweep": "vertices in position order 0 .. N-1, asynchronous (a move is visible to later vertices of the same sweep), single thread, deterministic",
            "move": "counts of neighbour labels over the adjacency entries (multi-edges keep their multiplicity, self loops are not counted); candidates = blocks other than the current one with a "
                    "positive count and load < cap; target = argmax count, ties -> smaller load, then smaller id; move iff target count > the count of the current block (a strict gain)",
            "stop": "at most 10 sweeps, or a sweep that moves fewer than N / 1000 vertices",
            "verification": "scratchpad/_fbx_lp.py lp_reference is the executable statement of the rule; TESTLP compares the C program with it bit for bit on five random graphs with one and with two "
                            "adjacency streams (PASS on the host, C source sha12 e1ab7a66fb5c)",
            "cost_class": "same as LDG: one sequential pass over both CSR streams per sweep, random reads of the 4 N byte label table; no graph is materialised beyond the served CSRs"},
        "maps_built_before_scoring": {lpm: sha(lpm), "note": "K grid 100 250 500 1000 2000 5000 25928, both graphs; structure and cost only (cut, closed-neighbourhood blocks, sizes, seconds, RSS); no recall"},
        "gate (unchanged from addendum 1)": "gate cells K in {100, 250, 500}, graph S only (LP_S / LDG_S / hash worst of 3 seeds), every B_N; PASS = matched-B_P ALL count at most 7 rows (of 786) under PHG at every gate cell; "
                                            "FAIL = 24 or more rows under at any gate cell or B_P = K on more than half the rows; else BORDERLINE (reported to the user). Graph SK is diagnostic only.",
        "scoring_run": "ONE EVAL over the 786 rows, methods PHG, HASH_s0..2, LDG_S, LDG_SK, LP_S, LP_SK, on the full bundle (FLAT identity vs flath2_G_webqsp counted; the EVAL asserts the PHG_k25928 regression "
                       "against the l3w_webqsp__v1 record on all 786 rows before it writes anything). The bundle's recorded scratchpad/_fbx_calib.py sha is the pre-LP file; the only later change to "
                       "that file is METHODS / load_maps / the two extra verdict lines (the BUNDLE code is unchanged).",
        "if_R3_passes": "R3 is the chosen method (first passing rung), stage 3b runs it on Freebase at K = 1000, 2000, 5000, 10000, 20000 (and K = 100 as the calibration anchor) plus hash as the structural control, "
                        "under its own dated addendum",
        "if_R3_fails_or_is_borderline": "borderline -> the user; fail -> rung 4 (two-level structural partitioning) under a further addendum, or a user ruling",
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT)[:16])


if __name__ == "__main__":
    main()
