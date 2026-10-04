"""Write addendum 12 (write-once) of the FBX_SCALE host-stage declaration: re-rank depth diagnostic on the T1 indexes (Track C step 3b), declared before any recall at R > 512 exists on Freebase codes."""
import hashlib
import io
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_12.json")


def sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


R = "results/FREEBASE_SCALE/"
ext = {R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n: sha(R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n) for n in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10", "11")}
basis = {R + "ivf/FBX_IVF_TRANSFER_T1__v1.json": sha(R + "ivf/FBX_IVF_TRANSFER_T1__v1.json"), R + "ivf/WEBQSP_IVF_SAMPLE__v1.json": sha(R + "ivf/WEBQSP_IVF_SAMPLE__v1.json"),
         R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json": sha(R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json")}
code = {c: sha(c) for c in ("scratchpad/_ivf_rdiag.py", "scratchpad/_ivf_transfer.py", "scratchpad/_ivf_knn.py", "scratchpad/_fbx_encode.py")}

rec = {
    "stage": "FBX_SCALE / addendum 12: re-rank depth diagnostic on the T1 indexes (Track C step 3b), declared before any recall at R > 512 exists on Freebase codes",
    "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "extends": ext,
    "basis": basis,
    "status": "DEVELOPMENT (a systems measurement); a diagnostic that isolates a cause; it selects no lever, builds no KNN family and changes no gate or contract",
    "observed_before_declaring (T1, section 54, read)": {
        "R1": "L437 at 32 chunks (4,194,304 rows, 9,589 lists): tie-tolerant recall@3 0.9337 / 0.9493 / 0.9582 at nprobe 32 / 64 / 128 vs the WebQSP baseline 0.9456 / 0.9672 / 0.9838: D = -0.012 / -0.018 / -0.026 (rule: holds if D >= -0.010) => DEGRADES",
        "R2": "L437 slope per doubling of n over c in {4, 8, 16, 32}: -0.0125 / -0.0110 / -0.0095 / -0.0087 / -0.0078 at nprobe 32 / 64 / 128 / 256 / 512 (rule: falling if < -0.005) => FALLING",
        "other": "at high probe counts the recall saturates: c = 32 L437 nprobe 256 -> 512 gains 0.0036 for twice the scan, reaching 0.9663 (WebQSP 0.9979); F4096 c = 32 nprobe 512 0.9680; self_in_top4 (the query row inside its own returned top 4) falls 0.99925 / 0.9975 / 0.99475 / 0.98825 / 0.9825 / 0.98125 over c = 1 ... 32"},
    "hypothesis_declared": ("ADC ranks the candidates of the probed lists by <q, x_hat> = ||x_hat|| cos(q, x_hat), and the cosine is only applied to the best R = 512 of them.  The decoded norms spread 0.689-1.018 (addendum 10), so a true neighbour with a small decoded norm can be pushed out of the best 512 by "
                            "rows of larger norm that lie farther in angle; the larger the database, the more such rows are scanned.  If this is the cause, recall at the same probe counts rises with R, the saturation at high nprobe lifts, and a full probe (every list) with a large R recovers the truth.  "
                            "If it is not, recall does not move with R and the loss is in the probing (cluster structure of the real codes, k-means quality at this nlist).  The hypothesis is NOT established by T1; this addendum measures it."),
    "question": "At c = 32 (and 8) chunks of the real Freebase PQ64 codes, how much of the recall gap to the WebQSP baseline is the re-rank depth R and how much is the number of lists probed?",
    "not_claimed": ["no end-to-end claim and no KNN family; no change to R = 512 as addendum 10's frozen WebQSP operating point; R > 512 here is a diagnostic cell, not a candidate operating point",
                    "no claim about 245 M rows; the cost of a larger R at that scale is not extrapolated from these cells",
                    "no claim that the prefix is representative (addendum 11, section on the node-order prefix); T2 (strided chunks) remains pending"],
    "design": {
        "indexes": "the T1 indexes, re-built from the T1 caches (the coarse quantiser and the row assignment files under data/freebase_scale/ivf_transfer/t1/c<NN>/) of addendum 11; a missing cache aborts (a re-trained quantiser is a different index); the same 4,000 queries (RandomState(1)) and the same exact truth file",
        "cells": {"c = 32, L437 (nlist 9,589)": "nprobe {32, 128, 512, 9,589 = EVERY list} x R {512, 2048, 8192}", "c = 32, F4096 (nlist 4,096)": "nprobe {32, 512} x R {512, 2048, 8192}", "c = 8, L437 (nlist 2,397)": "nprobe {32, 128} x R {512, 2048, 8192}"},
        "scoring": "addendum 10's: faiss IndexIVFPQ, by_residual = False, inner product with the accepted centroids; the best R by ADC re-ranked by cosine from the stored decoded norms; ONLY R (and the probe count of the full-probe wiring cell) differ from T1",
        "metrics": ["tie-tolerant recall@3 (PRIMARY), ids recall@3, nearest recall, self_in_top4, four-fold SE", "scan fraction, seconds and queries per second per cell (the cost of R)",
                    "norm_diag (descriptive): at c = 32 L437 nprobe 128 R 512, the median decoded norm of the true non-self top-3 neighbours that the search missed (by id) vs those it returned, and of all rows"],
        "execution": "one host job, 6 threads, resumable per cell; record results/FREEBASE_SCALE/ivf/FBX_IVF_TRANSFER_T1R__v1.json written once at the end"},
    "reading_rules_declared_before_any_recall_at_R_gt_512_is_read": {
        "Q1_R_limited": ("G = recall(R = 8192) - recall(R = 512), tie-tolerant, same cell.  At c = 32 L437 and F4096 nprobe 512: G >= 0.010 => 'R-limited at high probe'; G < 0.003 => 'not R-limited'; between => 'partly'.  "
                         "Likewise at nprobe 32 and 128 (reported, same labels).  The intermediate R = 2048 gives the shape (reported, not used for a label)"),
        "Q2_wiring": "full probe (nprobe = nlist 9,589) at c = 32 L437: recall at R = 8192 >= 0.995 => the ADC + cosine re-rank path recovers the truth when R is large and every list is probed ('R explains the full-probe ceiling'); < 0.995 => something other than R or probing limits the ceiling (reported with the R curve)",
        "Q3_transfer_with_larger_R": "D8192 = recall(c = 32, L437, R = 8192) - WebQSP baseline (R = 512) at nprobe 32 and 128 (the declared cells): 'holds' if D8192 >= -0.010 (addendum 11's R1 threshold), else 'degrades'.  The WebQSP baseline is the frozen R = 512 operating point, not a re-measurement at R = 8192 on WebQSP",
        "Q4_R_needed_vs_n": "at nprobe 128 L437: the smallest R in {512, 2048, 8192} whose recall is within 0.003 of the R = 8192 recall, at c = 8 and c = 32 ('not reached' if none); a larger R at c = 32 than at c = 8 is reported as R growing with n (two points: no functional form is fitted)",
        "Q5_cost": "queries per second per R at each cell (the cost of the larger R at fixed nprobe)"},
    "decision_table": {
        "Q1 R-limited AND Q3 holds": "the loss is the re-rank depth; the candidate lever for the Freebase index is 'R grows with n' (at the 245 M scale the cost of R is measured next, not extrapolated); the decision to adopt a larger R or an alternative (normalised-code scoring, GPU exact re-score of a candidate set) is the user's, because the frozen WebQSP operating point has R = 512",
        "Q1 R-limited AND Q3 degrades": "R explains part of the gap, probing the rest; report both parts; the next lever (more lists probed, a hierarchical coarse quantiser, GPU exact scoring) is the user's call",
        "Q1 not R-limited": "the loss is in the probing / the cluster structure of the real codes (T2 strided sample would show whether it is node-order specific); R is ruled out as the lever and the user chooses the next",
        "Q2 fails": "the full-probe ceiling is not explained by R: report the R curve; investigate ties / duplicate codes in the exact truth before any further lever"},
    "next_declared": ["T2: strided chunk sample (after the encode), same sizes", "T3: the full 245.35 M database (GPU coarse quantiser), after the user chooses the lever"],
    "code_pinned": code,
}
assert not os.path.exists(OUT), "write-once"
with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rec, indent=1, ensure_ascii=False))
os.replace(OUT + ".tmp", OUT)
print("written", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
