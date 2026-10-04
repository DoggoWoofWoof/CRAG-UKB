"""FBX_SCALE declaration addendum 5 (write-once): the user's two-track ruling of 2026-09-30, Track B (the WebQSP partitioner laboratory, SK to SK), declared BEFORE any SK-arm candidate is scored.

Nothing recall-related has been read for any candidate.  What has been read (structure only) is listed under disclosure_of_what_was_read_first; the candidate grid, the attempt rule, the shared
refinement, the three gates and the decision table are fixed here.  Usage: python scratchpad/_ml2_addendum5.py    (refuses to overwrite; reads the fetched records)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_5.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%d.json" % i for i in (1, 2, 3, 4)]
ML2 = FB + "/ml2"
METHODS = ("M1", "M2", "M3", "M4")
WS = (32, 128, 512, 2048)
KS = (100, 250, 500)
N = 2592894
BLOCK_RULE = 0.10
LADDER_W, LADDER_MIN = 512, 1.9
FBX_STRUCT_PINS = 4253534391
BUDGET_GB = 0.9 * 81.6


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
    coarsen = {}
    for m in METHODS:
        for w in WS:
            p = "%s/COARSEN__%s_W%d.json" % (ML2, m, w)
            assert os.path.exists(p), "missing %s (all 16 COARSEN records must exist before the grid is declared)" % p
            coarsen[(m, w)] = jl(p)
    oracle_p = ML2 + "/ORACLE_PIN_REDUCTION__v1.json"
    oracle = jl(oracle_p)
    grid, table = [], []
    for (m, w), r in sorted(coarsen.items()):
        last = max(x["level"] for x in r["maps"])
        for x in r["maps"]:
            tag = "%s_W%d_L%d" % (m, w, x["level"])
            row = {"candidate": tag, "method": m, "Wmax": w, "level": x["level"], "final_level": x["level"] == last, "clusters": x["clusters"], "cluster_size": x["cluster_size"],
                   "unsplit_pin_reduction": x["pin_reduction"], "map_sha256": x["sha256"]}
            table.append(row)
            if x["level"] == last or (w == LADDER_W and x["level"] >= 2 and x["pin_reduction"] >= LADDER_MIN):
                att = [K for K in KS if w <= BLOCK_RULE * N / float(K)]
                grid.append(dict(row, attempted_at_K=att, skipped_by_rule_at_K=[K for K in KS if K not in att]))
    ml_s = {K: jl(FB + "/ml/webqsp__ML_S_k%d.RUN.json" % K) for K in KS}
    bpp = {K: round(ml_s[K]["phg_coarse"]["memory"]["sum_of_rank_peaks_kb"] * 1024.0 / ml_s[K]["coarse_hypergraph"]["P"], 1) for K in KS}
    b_max = max(bpp.values())
    r_star = FBX_STRUCT_PINS * b_max / (BUDGET_GB * 1e9)
    m0r = {}
    for K in KS:
        p = "%s/webqsp__M0R_k%d.RUN.json" % (ML2, K)
        if os.path.exists(p):
            r = jl(p)
            m0r[str(K)] = {"km1_weighted_before": r["km1_weighted_before"], "km1_weighted_after": r["km1_weighted_after"], "blocks_per_net_before": r["blocks_per_net_unweighted_before"],
                           "blocks_per_net_after": r["blocks_per_net_unweighted_after"], "record_sha256": sha(p)}
    rec = {
        "stage": "FBX_SCALE / addendum 5: SK-arm partitioner laboratory on WebQSP (Track B), declared before any SK-arm candidate is scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S+05:30", time.localtime()),
        "extends": {a: sha(a) for a in A},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism",
        "user_ruling_2026-09-30": {
            "tracks": "A = finish the complete Freebase substrate (query-independent: STRUCT, KNN, NER as separate families, WebQSP/RoG bridge, per-family statistics); B = use WebQSP as a partitioner laboratory "
                      "and decide which SCALABLE algorithm preserves PHG recall best.  Both run in parallel; neither waits for the other; no candidate runs on full Freebase before WebQSP shows a compression "
                      "level that preserves PHG recall",
            "B_comparison": "SK to SK only: the same nodes, the same STRUCT and KNN families, balance 1.03, the same K, the same frozen routing and evaluation.  M0 = the frozen PHG_SK (the recall target); "
                            "no S-only candidate and no PHG_S in any verdict of this addendum",
            "B_candidates": {"M0": "frozen PHG_SK (Zoltan PHG, NP 4) -- the reference",
                             "M1": "heavy-connectivity multilevel: exact twins + heavy-connectivity agglomeration + weighted coarse vertices + vertex-weighted PHG + projection",
                             "M2": "closed-neighbourhood-aware coarsening (the SPAN rating: the weighted net-span reduction of merging a vertex into a cluster)",
                             "M3": "repeated modest contraction (caps 2, 4, 8, ..., Wmax; one modest pass per level) with the heavy-connectivity rating",
                             "M4": "hybrid: exact twins, then closed-neighbourhood (SPAN) merges on the same modest schedule, projection through the shared refinement; deterministic, non-learning"},
            "B_candidate_implementation_notes": "M2 and M4 use the SPAN rating as their closed-neighbourhood-aware rule; M4 has no SEPARATE 'heavy-connectivity fill' stage (its SPAN merges continue to the "
                                               "cap, the geometric schedule playing the role of the fill); the 'optional boundary refinement' is applied to EVERY candidate (and to M0 as M0R), so it never distinguishes "
                                               "one candidate from another.  These are the definitions scored; they are not adjusted after scoring",
            "B_gates": ["(1) recall: delta ALL-gold recall vs PHG_SK at matched (B_P, B_N) cells -- the primary gate",
                        "(2) compression: pin reduction r_pins = fine pins / coarse pins (Freebase needs about %.1fx; stage 4A reached about 2.5x)" % r_star,
                        "(3) systems cost among the recall-equivalent methods: peak RAM, disk, runtime, temporary storage, number of passes -- the cheapest recall-preserving method is chosen"],
            "B_compression_sweep": "several compression levels per algorithm; ALL-gold recall against pin compression with the predicted Freebase RAM",
            "B_finish_line": "a WebQSP point with r_pins >= the Freebase-required compression, |delta ALL| within the PHG envelope, balance <= 1.03; then Freebase is an engineering job",
            "OOC_engine": "in parallel and separable from the scoring rule: stream pins -> merge proposals -> external sort -> resolve clusters -> rewrite pins -> deduplicate -> checkpoint"},
        "disclosure_of_what_was_read_first": {
            "stage_4A_gate_result": "ML_S (H4_S, the STRUCT-only hypergraph) FAILED the gate against PHG on H4_SK (max loss 48 rows), was BORDERLINE against PHG_S (12), pins reduced only %.2f-%.2fx; PHG_S itself "
                                    "FAILED against PHG_SK (59).  That comparison mixed graph and algorithm, which is why every candidate here is SK to SK (record %s)" % (
                                        min(ml_s[K]["coarse_hypergraph"]["pin_reduction"] for K in KS), max(ml_s[K]["coarse_hypergraph"]["pin_reduction"] for K in KS), FB + "/CALIB_EVAL_WEBQSP__ml_v1.json"),
            "coarsening_structure": "all 16 clustering runs (method x Wmax in %s) were built and their STRUCTURE read (cluster counts, cluster weights, pins after every level) before this addendum; no gold "
                                    "label and no recall was read.  Per-level results are in the COARSEN records pinned below" % (list(WS),),
            "structural_ceiling_finding": "the pin reduction of an exact quotient is governed by the NUMBER of clusters V, not by how the clustering was found: the frozen PHG_SK maps read as clusterings "
                                          "(the best locality-aware clusterings the lane owns) give %s (record %s).  The balance constraint (1.03) bounds a cluster's weight to a few percent of a block, i.e. "
                                          "V >= tens of thousands at K 100-500, so the compression the cluster-and-quotient family can reach here is about 2-3.5x; the finish-line compression is NOT expected to be "
                                          "reachable by this family, and the sweep below is run to MEASURE the recall-vs-compression curve, not to hit the finish line" % (
                                              "; ".join("%s clusters -> %.3fx" % (r["clusters"], r["pin_reduction_unsplit"]) for r in oracle["rows"] if r["clusters"] >= 500), oracle_p),
            "M0R_structure": "the shared refinement applied to the frozen PHG_SK maps (KM1 only, no recall): %s" % (json.dumps(m0r) if m0r else "not yet built when this addendum was written"),
            "variant_count": "the SK-arm scoring adds the candidates of the grid below to the partitioners already scored on the same 786 split-A rows (hash, LDG_S, LP_S, CN_S, ML_S; PHG_S as a control): every "
                             "one is counted and reported, none hidden.  No held-out row is read (split B sealed, TEST unread)"},
        "candidates": {
            "clustering": {"builder": "scratchpad/_ml2_run.py COARSEN (C pass scratchpad/_ml2_agg.c, coarsener scratchpad/_ml2_coarsen.py)", "params": coarsen[("M1", 32)]["params"],
                           "input": "the frozen H4_SK nets: one net per anchor per family (anchor + its neighbours), weight max(1, rint(1000 / (|e| - 1))); the clustering is K-independent and built on the unsplit nets, "
                                    "the quotient is applied to each K's own frozen fine hypergraph (rebuilt and digest-checked against its frozen record), so weighted KM1 of the coarse partition equals "
                                    "weighted KM1 of the projected fine partition (asserted in every cell)",
                           "rating_and_schedule": {"M1": "heavy / flat", "M2": "span / flat", "M3": "heavy / geometric", "M4": "span / geometric"},
                           "twins": "exact open-twin contraction in BOTH families first, in chunks of at most Wmax"},
            "cell": "clusters -> exact quotient -> frozen Zoltan PHG driver with vertex weights (phg_driver_vw, NP 4, the 25 frozen parameters) -> projection -> empty-block repair of the frozen contract "
                    "-> the SHARED refinement -> validity gate (balance 1.03, every block used)",
            "shared_refinement": {"code": "scratchpad/_ml2_cn.c / _ml2_cn.py (tested: TEST PASS)", "layers": "STRUCT and KNN as two layers", "objective": "the frozen H4 objective: anchor weight max(1, rint(1000 / deg_family(u)))",
                                  "balance_repair": "cheapest-first out of every block above ceil(1.03 N / K) before the sweeps", "sweeps": 10, "T": 4, "dmax": 5000,
                                  "same_for_every_candidate": True, "control": "M0R = the frozen PHG_SK map put through the same refinement (no clustering, no coarse PHG): separates the effect of the refinement "
                                                                              "from the effect of the coarsening"},
            "grid_rule": "(a) the FINAL map of every (method, Wmax) with method in %s and Wmax in %s (%d candidates); (b) for Wmax = %d, every non-final level L >= 2 with unsplit pin reduction >= %.1f "
                         "(the compression ladder); (c) M0R.  Complete: a further variant needs its own addendum" % (list(METHODS), list(WS), len(METHODS) * len(WS), LADDER_W, LADDER_MIN),
            "attempt_rule": "a cluster candidate is built at K only if its declared Wmax <= %.2f N / K (K 100: <= %d, K 250: <= %d, K 500: <= %d); otherwise SKIPPED_BY_RULE and it is not scored at K.  "
                            "A cell whose validity gate fails is PARTITION_INVALID and is reported as such (not repaired, not retried)" % (BLOCK_RULE, BLOCK_RULE * N / 100, BLOCK_RULE * N / 250, BLOCK_RULE * N / 500),
            "grid": grid, "all_levels_read": table},
        "gate_recall (addendum 1, unchanged)": "gate cells K in {100, 250, 500}, every B_N in {100, 250, 500, 1000, 2000, 5000}; rows below PHG_SK's ALL count at matched B_P: <= 7 of 786 RECALL_PRESERVING, "
                                               ">= 24 FAILED (or B_P = K on more than half the rows), else BORDERLINE (reported to the user).  A candidate's verdict is the worst over the gate cells it was built at; "
                                               "a candidate built at K 100 only (Wmax 2048) is labelled PARTIAL",
        "compression_and_cost_readout": {
            "r_pins": "per cell: fine H4_SK pins / coarse pins (RUN.json coarse_hypergraph.pin_reduction)",
            "predicted_freebase_ram": "Freebase STRUCT pins %d / r_pins x bytes per coarse pin measured in the cell (sum of the four ranks' peak RSS / coarse pins; stage 4A measured %s B per pin at K 100/250/500), "
                                      "compared with 0.9 x 81.6 GB = %.1f GB.  The Freebase pin count is the STRUCT one (KNN does not exist for Freebase, see Track A); with KNN the requirement would be higher" % (
                                          FBX_STRUCT_PINS, "/".join(str(bpp[K]) for K in KS), BUDGET_GB),
            "required_compression_r_star": round(r_star, 2),
            "systems_cost": "coarsener peak RSS and seconds (COARSEN record), levels (passes), coarse-PHG memory and wall seconds, refinement seconds, disk of the maps"},
        "decision_table": {
            "FINISH_LINE (a candidate is RECALL_PRESERVING at all its gate cells AND r_pins >= r_star AND balance <= 1.03)": "reported; Freebase becomes an engineering job: the OOC engine is built for that "
                                                                                                                              "candidate's scoring rule, validated against the WebQSP in-memory result, then "
                                                                                                                              "progressively K = 1k, 2k, 5k, 10k, 20k plus the hash control",
            "no candidate reaches r_star (recall-preserving or not)": "the curve is reported as the result: 'Freebase recall-preserving PHG through cluster-and-quotient = infeasible at the compression this "
                                                                     "family can reach', with the RAM prediction per point.  Options for the user (NOT run here): a hierarchical two-level PHG (super-blocks, "
                                                                     "then per-block PHG with net splitting), or the systems-only route with a degraded partitioner (recall stated as degraded)",
            "a candidate reaches r_star but is BORDERLINE or FAILED": "reported to the user with both tables; no Freebase run",
            "M0R": "if the refinement alone changes ALL recall by more than the gate's 7 rows it is reported as a confound and the candidates are read against M0R as well as against M0"},
        "not_done": ["no candidate runs on the full Freebase graph", "no re-parameterisation after scoring", "no S-only candidate or PHG_S verdict", "no held-out row is read (split B sealed, TEST unread)"],
        "scoring_run": "ONE EVAL (scratchpad/_ml2_eval.py EVAL) on the v1 bundle: PHG (the reference), M0R and every cell built by the grid; modes own / matched (B_P = PHG's own B_P: THE GATE); the PHG cells must "
                       "reproduce the earlier EVAL record bit for bit or the run aborts",
        "code_pinned": {n: sha(n) for n in ("scratchpad/_ml2_agg.c", "scratchpad/_ml2_coarsen.py", "scratchpad/_ml2_run.py", "scratchpad/_ml2_cn.c", "scratchpad/_ml2_cn.py", "scratchpad/_ml2_cell.py",
                                            "scratchpad/_ml2_eval.py", "scratchpad/_ml2_oracle.py", "scratchpad/_ml_run.py", "scratchpad/_ml_coarsen.py", "src/l1_lowmem/phg_driver/phg_driver_vw.c")},
        "records_pinned": dict({oracle_p: sha(oracle_p)}, **{"%s/COARSEN__%s_W%d.json" % (ML2, m, w): sha("%s/COARSEN__%s_W%d.json" % (ML2, m, w)) for (m, w) in sorted(coarsen)}),
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote", OUT, sha(OUT)[:16], "grid candidates", len(grid), "r_star %.2f" % r_star)


if __name__ == "__main__":
    main()
