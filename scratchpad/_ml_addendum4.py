"""FBX_SCALE declaration addendum 4 (write-once): the user's rung-4 ruling (stage 4A), declared BEFORE any ML_S recall is scored.

Addendum 1 fixed the ladder, the gate cells and the pass / fail rule; addenda 2 and 3 declared rungs 3 and 3b (all four cheap rungs FAILED).  On 2026-09-30 the user ruled: rung 4, as the FINAL
recall-preservation attempt, with a tight scope (a hypergraph-aware multilevel prototype on WebQSP only, compared against canonical PHG under the existing gate; pass -> a progressive Freebase build,
fail -> the lane is closed as a measured negative; "No rung 5. No more heuristic search.").  This addendum (a) discloses what was read before it was written, (b) defines the prototype exactly, (c) states
the scoring run with its two references and its regression checks, (d) pre-declares the decision table, including the case in which the two references disagree, and the Freebase feasibility rule.

Usage: python scratchpad/_ml_addendum4.py            (refuses to overwrite; reads the fetched records)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
OUT = "results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_4.json"
A = ["results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%d.json" % i for i in (1, 2, 3)]
V1 = "results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__v1.json"
CN = "results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__cn_v1.json"
CTL = "results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__phgs_v1.json"
CLU = "results/FREEBASE_SCALE/ml/webqsp__ML_S__clusters.RUN.json"
REGR = "results/FREEBASE_SCALE/ml/ML_DRIVER_REGRESSION.json"
BUILD = "results/FREEBASE_SCALE/ml/ML_DRIVER_BUILD.json"
KS = (100, 250, 500, 1000)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    return json.load(io.open(p, encoding="utf-8"))


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    ctl, clu, reg = jl(CTL), jl(CLU), jl(REGR)
    assert reg["STATUS"] == "PASS"
    cells = {}
    for K in KS:
        if K == 1000:
            fj = "results/FREEBASE_SCALE/ml/webqsp__ML_S_k1000.FAILED.json"
            r = jl(fj)
            assert r["STATUS"] == "PARTITION_INVALID"
        else:
            r = jl("results/FREEBASE_SCALE/ml/webqsp__ML_S_k%d.RUN.json" % K)
            assert r["STATUS"] == "OK"
        cells[K] = r
    ph = {K: jl("results/FREEBASE_SCALE/parts_S/webqsp__H4_S_k%d__PHG_con.RUN.json" % K) for K in KS}
    pv = ctl["verdicts"]["PHG_S control vs PHG (S-arm ceiling)"]
    struct = {}
    for K in KS:
        c = cells[K]
        mem = c["phg_coarse"]["memory"]["sum_of_rank_peaks_kb"] * 1024.0
        struct[str(K)] = {"coarse_V": c["coarse_hypergraph"]["V"], "coarse_M": c["coarse_hypergraph"]["M"], "coarse_P": c["coarse_hypergraph"]["P"],
                          "pin_reduction": c["coarse_hypergraph"]["pin_reduction"], "phg_coarse_sum_of_rank_peaks_MB": round(mem / 1e6, 1),
                          "phg_coarse_bytes_per_coarse_pin": round(mem / c["coarse_hypergraph"]["P"], 1),
                          "phg_S_fine_sum_of_rank_peaks_MB": round(ph[K]["run"]["memory"]["sum_of_rank_peaks_kb"] / 1024.0, 1),
                          "phg_S_fine_bytes_per_pin": round(ph[K]["run"]["memory"]["sum_of_rank_peaks_kb"] * 1024.0 / ph[K]["hypergraph"]["P"], 1),
                          "km1_weighted": {"PHG_S": ph[K]["raw"]["km1"], "ML_projection": c["projection"]["km1_weighted"], "ML_refined": c["refinement"]["km1_weighted"] if "refinement" in c else None},
                          "blocks_per_net_unweighted": {"ML_projection": c["projection"]["blocks_per_net_unweighted"],
                                                        "ML_refined": c["refinement"]["blocks_per_net_unweighted"] if "refinement" in c else None},
                          "status": c["STATUS"]}
        if c["STATUS"] != "OK":
            pj = c["projection"]
            struct[str(K)]["invalid_reason"] = {"max_block_after_projection": pj["validity"]["max_block"], "contract_bound_ceil_1.03_N_over_K": pj["validity"]["contract_bound_ceil_1.03_N_over_k"],
                                                 "zoltan_imbalance_on_cluster_weights": c["phg_coarse"]["zoltan_eval"]["imbalance"],
                                                 "empty_blocks_repaired_before_projection_check": c["repair"]["empty_blocks"], "record": "results/FREEBASE_SCALE/ml/webqsp__ML_S_k1000.FAILED.json",
                                                 "record_sha256": sha("results/FREEBASE_SCALE/ml/webqsp__ML_S_k1000.FAILED.json"),
                                                 "consequence": "no ML_S map exists at K = 1000: PARTITION_INVALID under the same validity contract every method meets (Zoltan reported imbalance 1.036 on the cluster weights, i.e. above "
                                                                "its 1.03 tolerance, and 6 blocks came out empty).  Cluster granularity (up to 32 vertices per cluster against ~2,593 per block) is the "
                                                                "likely cause and is NOT tested.  K = 1000 is a diagnostic cell outside the gate and enters no verdict; no parameter is changed in response"}
    rec = {
        "stage": "FBX_SCALE / addendum 4: rung 4 (hypergraph-aware multilevel partitioner ML_S), stage 4A on WebQSP, declared and built before it is scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime()),
        "extends": {a: sha(a) for a in A},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism",
        "user_ruling_2026-09-30": {
            "decision": "\"Proceed to rung 4 as the final recall-preservation attempt.\" (option B), with a tight scope: NOT \"days of PHG at every Freebase K\"",
            "sequence": ["4A: a hypergraph-aware multilevel prototype on WebQSP only; compare it against canonical PHG under the exact existing gate; do not touch Freebase",
                         "4A passes -> 4B: build Freebase K = 1000 first (RAM, build time, disk, balance, cut and neighbourhood spread), and only if feasible continue progressively to 2k / 5k / 10k / 20k",
                         "4A fails -> \"Freebase recall-preserving low-memory partitioning = unresolved\": the lane is closed as a measured negative.  \"No rung 5. No more heuristic search.\""],
            "input_graph": "the canonical structural input exactly as the lane defines it; the two relations carrying 68 % of the edges are NOT removed or down-weighted: \"same graph semantics, different scalable "
                           "partitioning algorithm\"",
            "method_requirement": "NOT a cheap-LP coarse layer followed by a per-part PHG: the coarse level itself must be hypergraph-aware (hypergraph-aware coarsening -> a much smaller hypergraph -> PHG / balanced "
                                  "partition -> uncoarsen + refinement)",
            "threshold_wording": "the ruling states the target as loss <= 24 / 786 rows at every gate cell; addendum 1's rule (PASS <= 7, BORDERLINE 8-23 reported to the user, FAIL >= 24) is kept unchanged: the "
                                 "FAIL boundary coincides, and 8-23 is reported to the user with its numbers instead of being counted as a pass"},
        "disclosure_of_what_was_read_first": {
            "addendum_3_verdict": "the four cheap rungs FAILED the gate (hash 411, LDG_S 181, LP_S 156, CN_S 128 rows; records %s, %s)" % (V1, CN),
            "coarsening_probe": "before this addendum, a STRUCTURE-ONLY probe of the coarsener (laptop, no gold row, no recall) was run with the parameters below: open-twin contraction 2,592,894 -> 1,136,091 vertices "
                                "(pins 14,357,896 -> 9,347,505, 1.54x); three agglomeration passes -> 115,105 vertices, 5,438,199 unsplit pins (2.64x of the fine pins); the pass stack stopped because a pass "
                                "removed 0.9 % (< 5 %) of the vertices, not because the pin target was reached.  The parameters were fixed before this probe and were not changed after it.",
            "control_PHG_S_full_EVAL": "the frozen PHG on the STRUCT-only hypergraph (H4_S, NP 4, four K) was scored under the gate BEFORE this addendum: record %s (sha256 %s).  PHG_S FAILS the addendum-1 gate "
                                       "against PHG on H4_SK: max loss %d rows (rows below PHG at matched B_P, by B_N 100/250/500/1000/2000/5000: K100 %s, K250 %s, K500 %s).  The gate compares graph-S maps with maps "
                                       "built on S u KNN; the frozen partitioner on S alone does not pass it, so a graph-S method can pass it only by beating PHG on S."
                                       % (CTL, sha(CTL), pv["max_loss_rows_vs_PHG_matched_over_gate_cells"], pv["loss_rows_by_cell_and_B_N"]["PHG_S K100"], pv["loss_rows_by_cell_and_B_N"]["PHG_S K250"],
                                          pv["loss_rows_by_cell_and_B_N"]["PHG_S K500"]),
            "cluster_map": "the K-independent cluster map was built before scoring (record %s, sha256 %s): structure only" % (CLU, sha(CLU)),
            "cells_structure_read": "the ML_S cells (K 100 250 500 1000) were built before this addendum; their STRUCTURE was read (coarse sizes, KM1, blocks per net, sizes, seconds, RSS; below), recall was not.  "
                                    "K = 1000 came out PARTITION_INVALID (see cost_read); the gate cells K 100 / 250 / 500 are valid.",
            "driver_regression": "phg_driver_vw on a unit-weight manifest reproduces the frozen driver's partition exactly (%s, %s)" % (REGR, reg["STATUS"]),
            "variant_count": "ML_S is the fifth partitioner scored on the same 786 split-A rows (hash, LDG, LP, CN, ML_S) and is counted, not hidden.  No held-out row is read (split B sealed, TEST unread).  "
                             "Its parameters were fixed before it was built and are not tuned afterwards; per the ruling there is no re-parameterisation and no rung 5."},
        "ML_S_definition": {
            "graph": "S = the STRUCT family of the canonical webqsp substrate (unchanged; all relations kept; no down-weighting).  Freebase would use the same (it has no KNN family)",
            "fine_hypergraph": "H4_S = the frozen builder (src/l1_canonical/hypergraph.py, H4_SPLIT_PRESERVE, cap = round(N / K), weight max(1, rint(1000 / (|e| - 1)))) on STRUCT only: the SAME hypergraph semantics as the "
                               "frozen PHG_SK, without the KNN nets",
            "coarsening": {"description": "a vertex clustering; the coarse hypergraph is the exact quotient of the fine one (pins mapped to clusters, duplicate pins removed, single-pin nets dropped, identical nets merged "
                                          "with their weights summed, vertex weight = cluster size).  For any partition of the clusters the weighted KM1 of the quotient equals the weighted KM1 of the fine hypergraph "
                                          "under the projected partition (TESTML checks it; every cell asserts it against the Python recomputation)",
                           "operators": ["T: open-twin contraction (vertices with identical open neighbourhoods; verified by comparing the neighbour lists, not only hashes), chunks of at most Wmax",
                                         "A: agglomerative heavy-connectivity clustering (_ml_agg.c): a free vertex, visited in ascending (weight, id), joins the cluster with the largest sum over shared nets "
                                         "(2 <= |e| <= Lmax) of w_e / (|e| - 1) among clusters with weight + its weight <= Wmax; repeated on the quotient"],
                           "parameters": clu["params"], "stop": "pins <= fine pins / R, or a pass removes < MIN_GAIN of the vertices, or MAXLEVELS passes",
                           "verification": "scratchpad/_ml_coarsen.py TESTML: the C pass equals the plain-Python reference bit for bit on four random hypergraphs with uneven vertex weights; the quotient identity holds; "
                                           "open-twin classes are exact",
                           "result": {"clusters": clu["clusters"], "cluster_size": clu["cluster_size"], "pin_reduction_unsplit": clu["final_unsplit_hypergraph"]["pin_reduction"], "seconds": clu["seconds"],
                                     "peak_rss_mb": clu["peak_rss_mb"]}},
            "coarse_partition": "the frozen Zoltan PHG driver with vertex weights (src/l1_lowmem/phg_driver/phg_driver_vw.c: the same Zoltan calls, the same 25 parameters, NP = 4; the only change is the vertex-weight "
                                "input under manifest header 'N M 3'); the balance constraint is on cluster weights (= fine vertex counts)",
            "uncoarsening": "projection to the fine vertices (an empty block would be repaired by the frozen repair rule, as PHG_S), then the R3b closed-neighbourhood KM1 label propagation on the fine graph, its "
                            "parameters unchanged from addendum 3 (T = 4, DMAX = 5000, <= 10 sweeps, balance 1.03 / 0.97)",
            "cost_read": struct},
        "gate (unchanged from addendum 1)": "gate cells K in {100, 250, 500}, graph S, every B_N in {100, 250, 500, 1000, 2000, 5000}; rows below the reference's ALL count at matched B_P: <= 7 PASS, >= 24 FAIL "
                                            "(or B_P = K on more than half the rows), else BORDERLINE.  K = 1000 is a diagnostic and does not enter a verdict (ML_S has no valid K = 1000 map; PHG and PHG_S are scored there).",
        "scoring_run": "ONE EVAL (scratchpad/_ml_eval.py EVAL) on the v1 bundle: methods PHG (the seven H4_SK maps; the gate's reference), PHG_S (control), ML_S (the candidate), ML_S_proj (the projection before the LP; "
                       "diagnostic); modes own / matched (B_P = PHG's own B_P: THE GATE) / matchedS (B_P = PHG_S's own B_P: the same-graph reference).  Before it writes anything, every PHG and PHG_S cell must equal the "
                       "control EVAL npz bit for bit, or the run aborts.",
        "two_verdicts": {"THE_GATE": "ML_S vs PHG (H4_SK), matched B_P: exactly addendum 1",
                         "SAME_GRAPH": "ML_S vs PHG_S (the frozen partitioner on the same graph and hypergraph semantics), matched to PHG_S's own B_P, the same cells and thresholds; the ruling's phrase 'same graph "
                                       "semantics, different scalable partitioning algorithm'.  Diagnostic role of the control: it is what the S arm can reach when the partitioner is the frozen one."},
        "decision_table": {
            "THE_GATE = RECALL_PRESERVING": "4A PASSES; the SAME_GRAPH verdict is reported beside it",
            "THE_GATE not PASS and SAME_GRAPH = RECALL_PRESERVING": "the two readings DISAGREE (the gate cannot be passed on S even by the frozen partitioner; recorded above before scoring): reported to the user as "
                                                                    "BORDERLINE with both tables; NO lane closure and NO Freebase build until the user rules",
            "SAME_GRAPH = BORDERLINE (either THE_GATE state)": "reported to the user; no closure, no Freebase build",
            "SAME_GRAPH = FAILED (>= 24 rows below PHG_S at some gate cell)": "4A FAILS under both readings: 'Freebase recall-preserving low-memory partitioning = unresolved', the lane is closed as a measured negative, "
                                                                            "no rung 5, no re-parameterisation; the numbers are the systems conclusion (cheap O(N) graph partitioners preserve cut, not neighbourhood locality; "
                                                                            "the hypergraph-aware multilevel construction ... as measured)"},
        "freebase_feasibility_rule_if_4A_passes": "4B is attempted only if the projected peak memory fits: (Freebase STRUCT pins 4,253,534,391 / the pin reduction the coarsener reaches on Freebase) x (bytes per coarse "
                                                  "pin measured on WebQSP, table above) must be <= 0.9 x the 81.6 GB available to the crag lane.  The Freebase coarsening itself (structure only, bounded memory) is the first 4B step and "
                                                  "MEASURES the pin reduction there; if the projection exceeds the budget the verdict is FBX_ML_PHG_PREDICTED_INFEASIBLE_ON_HOST and it is not attempted (an out-of-memory would endanger the "
                                                  "shared host; precedent: section 46).  Rungs 1-3b stay available only as the systems-measurement fallback the user described (recall stated as degraded).",
        "code_pinned": {n: sha(n) for n in ("scratchpad/_ml_agg.c", "scratchpad/_ml_coarsen.py", "scratchpad/_ml_run.py", "scratchpad/_ml_eval.py", "src/l1_lowmem/phg_driver/phg_driver_vw.c",
                                            "scratchpad/_fbx_cn.c", "scratchpad/_fbx_cn.py")},
        "records_pinned": {CTL: sha(CTL), CLU: sha(CLU), REGR: sha(REGR), BUILD: sha(BUILD)},
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT)[:16])


if __name__ == "__main__":
    main()
