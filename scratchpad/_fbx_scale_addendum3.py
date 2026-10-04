"""FBX_SCALE declaration addendum 3 (write-once): rung 3b of the partitioner ladder (closed-neighbourhood-connectivity label propagation), declared and built BEFORE any R3b recall is scored.

Addendum 1 fixed the ladder, the gate cells and the pass / fail rule; addendum 2 declared rung 3 (balanced LP).  The full-population EVAL (CALIB_EVAL_WEBQSP__v1) scored rungs 1-3 under the
addendum-1 gate: all three FAILED.  Nothing about the gate changes here.  This addendum (a) discloses what had been read before it was written, (b) defines R3b exactly, (c) states the
restricted scoring run and its control check, (d) records the cost class and what is decided if R3b passes, borders or fails.

Usage: python scratchpad/_fbx_scale_addendum3.py            (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_3.json"
A1 = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_1.json"
A2 = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_2.json"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    v1 = "results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__v1.json"
    cnm = "results/FREEBASE_SCALE/CALIB_MAPS_WEBQSP__cn_v1.json"
    rec = {
        "stage": "FBX_SCALE / addendum 3: rung 3b (closed-neighbourhood-connectivity label propagation) declared and built before it is scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime()),
        "extends": {A1: sha(A1), A2: sha(A2)},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism",
        "disclosure_of_what_was_read_first": {
            "the_full_verdict": "the full-population EVAL %s (sha256 %s) was READ before this addendum was written: R1 hash, R2 LDG on S and R3 LP on S all FAIL the addendum-1 gate (max loss vs PHG at matched B_P: 411, 181 and 156 "
                                "rows of 786); diagnostics LDG_SK 155 and LP_SK 138." % (v1, sha(v1)),
            "the_finding_that_motivates_R3b": "in that record, recall at matched fan-out orders with the mean number of distinct blocks in an anchor's closed neighbourhood e(u) = {u} U N(u) (the "
                                              "objective the frozen H4 hypergraph partitioner minimises), not with the edge cut: at K >= 500 the R3 map has the LOWER cut than PHG and still the worse "
                                              "recall (K 1000: ALL@5000 598 vs 672), while closed-neighbourhood blocks order PHG 2.42 < LP 2.71 < LDG 2.92 < hash 4.77.",
            "the_R3b_map_structure": "the R3b maps were built (record %s, sha256 %s) BEFORE this addendum and BEFORE any recall was computed for them; their STRUCTURE (KM1, cut, closed-neighbourhood blocks, "
                                     "sizes, seconds, RSS) was read.  Recall was not." % (cnm, sha(cnm)),
            "variant_count": "R3b is a variant of rung 3 designed AFTER the v1 verdict from the finding above, so it is the fourth partitioner scored on the same 786 split-A rows (hash, LDG, LP, R3b); "
                             "it is counted, not hidden.  The rows are a development population, no held-out row is read (split B stays sealed, TEST unread).",
            "consequence": "R3b's parameters (below) were fixed before any R3b recall existed and are not tuned afterwards; a failed R3b goes to the user (rung 4 or accept-with-disclosure), it is not re-parameterised."},
        "R3b_definition": {
            "name": "balanced label propagation whose move gain is the change of the closed-neighbourhood connectivity KM1 = sum_u (lambda(e(u)) - 1), a REFINEMENT of the R3 LP map "
                    "(CN_S from LP_S on graph S; CN_SK from LP_SK on graph SK); same graph, same adjacency streams; cost class stated below",
            "balance": "cap = ceil(1.03 N / K) and floor lo = floor(0.97 N / K), as R3",
            "sweep": "vertices in position order 0 .. N-1, asynchronous, single thread, deterministic; at most 10 sweeps, or stop after a sweep that moves fewer than N / 1000 vertices",
            "move": "for vertex v (skipped if it has no neighbour, or more than DMAX = 5,000 adjacency entries) in block c: candidates = the first T = 4 blocks p != c ordered by (neighbour count "
                    "descending, id ascending) with a positive count and load < cap; v may leave only if load_c > lo; delta(b) = sum over the anchors u in {v} U adj(v) of ([hist_u[b] == 0] - "
                    "[hist_u[c] == 1]), hist_u = exact per-anchor block histogram (pins with multiplicity); target = the candidate with the smallest delta, ties -> larger neighbour count, smaller "
                    "load, smaller id; move iff delta < 0 (a strict KM1 gain)",
            "fixed_parameters": {"T": 4, "DMAX": 5000, "sweeps_max": 10},
            "verification": "scratchpad/_fbx_cn.py cn_reference is the executable statement of the rule; TESTCN compares the C program with it bit for bit on six random graphs, with one and with two "
                            "adjacency streams, with hub skipping and with multi-edges (PASS on the host, C source sha12 4c8aaa6dd4be), and checks on multi-edge-free graphs that the reported KM1 equals "
                            "the set-based definition and that the summed chosen deltas explain its change",
            "cost_class": "exact per-anchor histograms: a pool of 2 int32 per pin (WebQSP 0.14 GB; Freebase estimated 4.4 G pins -> about 35 GB) plus 12 N bytes of offsets/counts and the 4 N byte label table; "
                          "the CSRs stay memory-mapped; one extra pass over both CSR streams per sweep with random reads of the histograms.  Measured, not assumed, at stage 3b."},
        "maps_built_before_scoring": {cnm: sha(cnm), "note": "K grid 100 250 500 1000, both graphs; structure and cost only; no recall"},
        "gate (unchanged from addendum 1)": "gate cells K in {100, 250, 500}, graph S only (CN_S; the deployable arm), every B_N in {100, 250, 500, 1000, 2000, 5000}; PASS = matched-B_P ALL count at most "
                                            "7 rows (of 786) under PHG at every gate cell; FAIL = 24 or more rows under at any gate cell or B_P = K on more than half the rows; else BORDERLINE (reported to the "
                                            "user).  Graph SK (CN_SK) is diagnostic only.  K = 1000 is scored as a diagnostic and does not enter the verdict.",
        "scoring_run": "ONE restricted EVAL (scratchpad/_fbx_calib_cn.py) over the 786 rows on the v1 bundle, methods PHG, LP_S (control), CN_S, CN_SK at K in {100, 250, 500, 1000}.  The full-population "
                       "regression of _fbx_calib.py (PHG_k25928 vs the l3w record) is not re-run (K 25,928 is outside this grid); in its place EVERY PHG and LP_S cell (SV and BP arrays, own and matched, "
                       "all rows, all four K) must equal the v1 EVAL npz bit for bit, or the run aborts before it writes anything.",
        "if_R3b_passes": "R3b is the first passing rung: the choice by (peak RAM, wall time) is among passers; stage 3b runs it on Freebase at K = 1000, 2000, 5000, 10000, 20000 (K = 100 as the calibration "
                         "anchor) plus hash as the structural control, under its own dated addendum; its RAM (about 35 GB pool + 5 GB) is measured there",
        "if_R3b_is_borderline_or_fails": "borderline -> the user; fail -> the user chooses between rung 4 (a two-level structural / hierarchical PHG; an estimated days of host compute per K) and "
                                         "accepting the best cheap rung as the systems-scale partitioner with its recall loss stated; no further variant is tried without a ruling",
        "code_pinned": {n: sha("scratchpad/" + n) for n in ("_fbx_cn.c", "_fbx_cn.py", "_fbx_cn_maps.py", "_fbx_calib_cn.py")},
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT)[:16])


if __name__ == "__main__":
    main()
