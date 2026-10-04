"""Write addendum 10 (write-once) of the FBX_SCALE host-stage declaration: IVF approximate search over the PQ64 codes (Track C step 2), declared before any IVF result exists."""
import hashlib
import io
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_10.json")


def sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


R = "results/FREEBASE_SCALE/"
ext = {R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n: sha(R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n) for n in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9")}
basis = {R + "pq/WEBQSP_PQ64_KNN__v1.json": sha(R + "pq/WEBQSP_PQ64_KNN__v1.json"), R + "CALIB_EVAL_WEBQSP__pq_v1.json": sha(R + "CALIB_EVAL_WEBQSP__pq_v1.json"),
         R + "CALIB_EVAL_WEBQSP__noise_v1.json": sha(R + "CALIB_EVAL_WEBQSP__noise_v1.json"), R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json": sha(R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json")}
code = {c: sha(c) for c in ("scratchpad/_ivf_knn.py", "scratchpad/_ivf_phg.py", "scratchpad/_ivf_eval.py", "scratchpad/_pq_knn.py", "scratchpad/_pq_phg.py", "scratchpad/_pq_eval.py", "scratchpad/_ml2_eval.py",
                                "scratchpad/_fbx_calib.py", "scratchpad/_l1h_host.py", "scratchpad/_l1d_lib.py", "src/l1_canonical/hypergraph.py", "src/l1_lowmem/phg.py", "src/l1_lowmem/phg_repair.py",
                                "src/l1_lowmem/phg_driver/phg_driver.c", "src/l1_lowmem/phg_driver/phg_driver_vw.c")}

rec = {
    "stage": "FBX_SCALE / addendum 10: IVF approximate search over the PQ64 codes (Track C step 2), WebQSP calibration, declared before any IVF recall or any IVF-keyed partition exists",
    "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "extends": ext,
    "basis": basis,
    "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism and nothing here changes a gate",
    "question": ("Track C accepted product quantisation at 64 B/row (OPQ + PQ 64 x 8 bit) on the strength of EXHAUSTIVE decoded-vs-decoded search.  The Freebase run cannot search exhaustively (245,350,737 names x 245,350,737 names): "
                 "the deployed search is an inverted file over the stored codes.  Which (number of lists, lists probed) keeps WebQSP's KNN family end-to-end recall-preserving under addendum 1's gate, and what does that "
                 "operating point cost?  Phase A measures neighbour recall against the exhaustive PQ64 truth on a query sample; phase B runs every row as a query at the arms phase A selects by the rule below, "
                 "pushes the resulting KNN family through the frozen partitioner (H4_{STRUCT u KNN_IVF}, K in {100, 250, 500}) and scores it under the gate."),
    "not_claimed": ["no Freebase index, no Freebase KNN and no claim that a WebQSP scan fraction or probe count transfers to 245 M names: the transfer is measured later, on growing prefixes of the real Freebase codes against exhaustive GPU truth, in its own addendum",
                    "no change to the PQ contract (m = 64, the accepted codebook, the encoder) and no re-scoring of the PQ64 exhaustive arm",
                    "no residual coding, no 4-bit fast-scan, no graph index, no GPU search: if the IVF arms do not clear the gate those are new levers, each its own addendum",
                    "no change to addendum 1's thresholds; the same-algorithm replicate floor F = 8 of addendum 8 is reported beside each arm's maximum loss and relaxes nothing"],
    "construction": {
        "database": "the 1,791,533 distinct rows of WebQSP as their PQ64 codes (data/freebase_scale/pq/pq64/codes.npy) with the accepted codebook; the centroid table is rounded to fp16 first, so a decoded vector is bitwise the fp16 storage vector the exhaustive search of addendum 6 used (asserted against faiss's own decode on 2,000 real rows)",
        "query": "the decoded vector of the query row: decoded-vs-decoded, the deployed regime",
        "coarse_quantiser": "faiss spherical k-means, niter 20, nredo 1, seed 2026, trained on 262,144 decoded unit-norm rows (RandomState(0)); nlist in {1024, 4096}; a row is listed under its largest inner product centroid",
        "scoring": "faiss IndexIVFPQ, by_residual = False, inner product with the accepted PQ centroids: ADC = <decoded query, decoded row> exactly; the best R = 512 by ADC are re-ranked by cosine (ADC / (||query|| ||row||), stored decoded norms); the top 4 with self included (as the exhaustive search) are returned",
        "rerank_depth_R": ("R = 512, fixed BEFORE any grid cell was measured by a full-probe (nprobe = nlist = 1024) wiring check on 300 queries (RandomState(99)): ADC ranks by <q, x> = ||x|| cos and the decoded norms spread 0.689-1.018 (p05 0.836, median 0.912, p95 0.961), so a shallow re-rank loses true neighbours even when every list is probed: "
                           "R = 32 -> tie-tolerant recall@3 0.9789, R = 128 -> 0.9978, R = 512 -> 1.0000 (the smallest of {32, 128, 512} with ceiling >= 0.999 is 512); this checks the re-rank, it says nothing about the probing grid"),
        "phase_A_SAMPLE": {"queries": "100,000 distinct rows, RandomState(1)", "grid": {"1024": [8, 16, 32, 64, 128, 256], "4096": [32, 64, 128, 256, 512, 1024]},
                           "metrics": ["rowwise recall@3 vs the exhaustive PQ64 top-4 (ids; addendum 6's definition)", "nearest-neighbour recall", "tie-tolerant recall@3 (a returned neighbour counts if its cosine >= the exhaustive third-best non-self cosine - 1e-5; rows with identical codes tie)",
                                       "self in the returned top 4", "scan fraction = mean rows in the probed lists / N, and nprobe / nlist", "queries per second and threads"]},
        "phase_B_arm_rule": ("targets T in {0.90, 0.95, 0.98} on the TIE-TOLERANT recall@3.  nlist* = the nlist whose smallest grid nprobe reaching 0.95 has the smaller measured scan fraction (if neither reaches 0.95: the nlist with the higher "
                             "recall at its largest nprobe).  At nlist*, arm IVF_T<100T> = the smallest grid nprobe with tie-tolerant recall@3 >= T; a target the largest grid nprobe does not reach is not run; arms with the same nprobe run once, named by the "
                             "lowest target.  Arm tags are IVF_T90 / IVF_T95 / IVF_T98.  Each arm: every row as a query (all 1,791,533), the canonical merge, the keys file (STRUCT unchanged, KNN = pairs minus STRUCT)."),
        "phase_B_partition": "scratchpad/_ivf_phg.py: the frozen H4_SK rule (H4_SPLIT_PRESERVE, cap = round(N / K)), the frozen driver (NP 4, parameters, repair, validity gate) at K in {100, 250, 500}; only the KNN key set differs",
        "phase_B_scoring": "scratchpad/_ivf_eval.py: addendum 6's evaluation (its own control: the PHG_SK cells reproduced bit for bit)"},
    "gate": ("addendum 1's verdict rule, unchanged: rows below PHG_SK's ALL-gold count at matched B_P, every B_N in {100, 250, 500, 1000, 2000, 5000}, at K in {100, 250, 500}; at most 7 of 786 RECALL_PRESERVING, at least 24 FAILED, "
             "else BORDERLINE; also FAILED if B_P = K on more than half the rows"),
    "reported_for_every_arm": ["phase A: the full grid table (recall variants, scan fraction, queries per second per thread)", "phase B: recall vs the exhaustive PQ64 truth over all rows; KNN pair recall/precision against the frozen exact KNN and against the PQ64 exhaustive family",
                               "the gate table (rows below PHG_SK per K and B_N), the verdict, the descriptive churn, the replicate floor F beside the maximum loss, the PQ64 exhaustive arm's verdict beside it",
                               "first-order cost at Freebase: rows scanned = N_fb x scan_fraction x N_fb and the same at constant probe count with nlist scaled to hold the list length; thread-hours from the measured WebQSP rate (a plan input, not a claim)"],
    "decision_table": {"an IVF arm is RECALL_PRESERVING": "that scan fraction / probe count keeps the end-to-end gate on WebQSP; the cheapest such arm is the candidate operating point; the Freebase transfer is then measured on prefixes of the real codes (next addendum), not assumed",
                       "no arm RECALL_PRESERVING (all BORDERLINE/FAILED)": "report the frontier and the nearest arm's maximum loss against F = 8; the next lever is chosen by the user (candidates: more lists with a hierarchical coarse quantiser, residual/fast-scan scoring, a larger R, GPU exact scoring of a candidate set)",
                       "phase A never reaches a target": "that arm is not run; the largest reached recall is reported and no gate is claimed for it"},
    "not_done_here": ["any Freebase search or index", "any change to the accepted PQ codebook or the encoding contract", "any nlist other than {1024, 4096} or probe counts off the declared grid", "R other than 512"],
    "code_pinned": code,
}
assert not os.path.exists(OUT), "write-once"
with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rec, indent=1, ensure_ascii=False))
os.replace(OUT + ".tmp", OUT)
print("written", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
