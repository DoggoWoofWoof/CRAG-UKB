"""Write addendum 13 (write-once) of the FBX_SCALE host-stage declaration: the encode ORDER change (32-chunk stratified sample first) and T2, the IVF transfer measurement on that sample with the re-rank depth as a design axis.
Declared before any chunk of the sample beyond the node-order prefix is read for recall and before any T2 recall exists."""
import hashlib
import io
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = "results/FREEBASE_SCALE/"
OUT = os.path.join(REPO, R, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_13.json")


def sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


ext = {R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n: sha(R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n) for n in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10", "11", "12")}
basis = {R + "ivf/FBX_IVF_TRANSFER_T1__v1.json": sha(R + "ivf/FBX_IVF_TRANSFER_T1__v1.json"), R + "ivf/FBX_IVF_TRANSFER_T1R__v1.json": sha(R + "ivf/FBX_IVF_TRANSFER_T1R__v1.json"),
         R + "ivf/WEBQSP_IVF_SAMPLE__v1.json": sha(R + "ivf/WEBQSP_IVF_SAMPLE__v1.json"), R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json": sha(R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json")}
code = {c: sha(c) for c in ("scratchpad/_ivf_t2.py", "scratchpad/_fbx_enc_order.py", "scratchpad/_ivf_transfer.py", "scratchpad/_ivf_knn.py", "scratchpad/_fbx_encode.py")}

import sys  # noqa: E402
sys.path.insert(0, os.path.join(REPO, "scratchpad"))
import _fbx_enc_order as O  # noqa: E402

rec = {
    "stage": "FBX_SCALE / addendum 13: the encode ORDER (a 32-chunk stratified sample first) and T2 -- the IVF transfer measurement on that sample with the re-rank depth R as a design axis (Track C step 3c)",
    "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "extends": ext,
    "basis": basis,
    "status": "DEVELOPMENT (a systems measurement); selects no lever, builds no KNN family, changes no gate or contract",
    "user_ruling": ("2026-10-01 (reply to the open decisions): item 3 'yes: do the 32-chunk strided sample now.  Getting T2 in ~2.4 h is worth changing encode order only; keep the final encoded artifact identical.'  Item 4 'defer: use T2/full-scale evidence before choosing R.  "
                    "Test 512 / 2048 / 8192 rather than guessing now.'  Item 2 'all 245M names'."),
    "encode_order_change": {
        "what": "the sequential encode job 261001-123023-fbx-enc-chunks-2348 (CHUNKS 0 1872, ascending) was cancelled by rx right after it wrote chunk 133 (log line 19:15:39 host clock); the job 261001-191519-fbx-enc-order-1f8c runs scratchpad/_fbx_enc_order.py RETRY 20: "
                "phase 1 encodes the sample chunks that are still missing, phase 2 every remaining chunk ascending.  Chunks 0 .. 133 existed before the swap (node order).",
        "sample": {"strata": 32, "chunks_total": 1872, "rule": "the chunk at the midpoint of each of 32 equal strata: int((k + 0.5) * 1872 / 32), k = 0 .. 31", "by_stratum": O.SAMPLE_BY_STRATUM,
                   "nested_order": O.SAMPLE_ORDER, "nested_order_rule": "bit-reversed stratum index, so the first 2^j chunks are an even spread over the name space (j = 0 .. 5)"},
        "artifact_invariance": ("every chunk is written by _fbx_encode.cmd_chunks itself (same model, revision, max_seq_length, batch composition within a chunk, OPQ+PQ quantiser, record fields and checksum); the wrapper only changes which chunks cmd_chunks sees as missing "
                                "(chunk_ok is answered True for non-sample chunks in phase 1).  A chunk is a function of its own rows alone, so the final 1,872 chunk files are the ones a sequential run produces; VERIFY passes on the same criterion.  "
                                "Not claimed: bitwise equality of GPU float arithmetic across two separate runs (the quantised codes are the artifact; the TEST in _fbx_encode.py checks a re-encode against the stored codes bit for bit)."),
        "cost": "one chunk in flight (<= 4.5 min) was lost to the cancel; the sample's 30 missing chunks take about 2.1 h at 4.2 min each; the total encode time is unchanged (the same chunks, a different order)"},
    "question": ("Do the T1 (prefix, R = 512) and T1R (R in {512, 2048, 8192} at a few prefix cells) findings hold on a stratified sample of the name space, and how does the re-rank depth R needed to reach the WebQSP-level neighbour recall "
                 "grow with the database size from 131 k to 4.2 M rows?"),
    "observed_before_declaring": {"T1": "L437 prefix recall falls about 1 pt per doubling of n at R = 512 (tie-tolerant 0.9337 / 0.9493 / 0.9582 at nprobe 32 / 64 / 128, c = 32)",
                                  "T1R": "the loss is the re-rank depth: R = 8192 at c = 32 gives 0.9623 / 0.9867 / 0.9952 at nprobe 32 / 128 / 512 and 0.9972 at every list; the small-norm-crowding mechanism was not supported (missed neighbours have higher norm)",
                                  "limit": "a prefix of the chunks is the first names in node first-occurrence order and is not a random sample; the sizes were never compared with a spread sample"},
    "design": {
        "script": "scratchpad/_ivf_t2.py T2 (6 threads, 12 GB, resumable per stage and cell; one write-once record results/FREEBASE_SCALE/ivf/FBX_IVF_TRANSFER_T2__v1.json at the end); runs only after all 32 sample chunks verify",
        "database": "the PQ64 codes of SAMPLE_ORDER[:c], c in {1, 2, 4, 8, 16, 32}, concatenated in that order (n = c * 131,072); every chunk record verifies against the frozen contract and codebook",
        "search": "addendum 10's scoring unchanged (faiss IndexIVFPQ, by_residual False, inner product, fp16-rounded centroid table; ADC by ||x|| cos; the best R re-ranked by cosine from the stored decoded norms; top 4 with self), the T1 code paths",
        "queries_and_truth": "4,000 rows of the database (RandomState(1)), EXACT decoded-vs-decoded top 4 including self (blocked fp32 inner product over every row)",
        "coarse": "spherical k-means niter 20, seed 2026, on min(n, 64 * nlist) decoded unit rows (RandomState(0)), as T1",
        "cells": {"L437 (nlist = round(n / 437.386))": "every c; nprobe {32, 64, 128, 256, 512} (never above nlist / 2) x R {512, 2048, 8192}",
                  "F4096 (nlist 4096)": "c = 32 only; nprobe {32, 128, 512} x R {512, 2048, 8192}"},
        "adc_rank (descriptive)": ("at c in {8, 32}, L437, nprobe in {32, 128}: for the true non-self top-3 neighbours of every query, where each sits in the ADC order of the probed lists' best 8,192: rank < 512 | 512 .. 2047 | 2048 .. 8191 | in a probed list but rank >= 8192 | "
                                   "in no probed list"),
        "duplicate_code_fraction (descriptive)": "per c, the fraction of rows whose 64-byte code equals another row's",
        "metrics": "tie-tolerant recall@3 (PRIMARY), ids recall@3, nearest recall, self_in_top4, four-fold SE, scan fraction, seconds and queries per second per cell"},
    "reading_rules_declared_before_any_T2_recall_exists": {
        "P1_representativeness": ("D = recall(T2 strided, L437, R = 512) - recall(T1 prefix) at the same c and nprobe.  Over c in {8, 16, 32} and nprobe in {32, 128, 512}: all |D| <= 0.010 => AGREE (the node-order prefix of T1 / T1R was representative at these sizes); "
                                  "some D < -0.010 and none > +0.010 => STRIDED_LOWER (the prefix was optimistic); some D > +0.010 and none < -0.010 => STRIDED_HIGHER; otherwise MIXED.  The fold SE of a cell (0.001 - 0.004) is reported beside D"),
        "P2_slope": "least-squares slope of recall per doubling of n over c in {4, 8, 16, 32}, L437, at R = 512 and at R = 8192, for each nprobe; R = 512 at nprobe 128 below -0.005 => FALLING (T1's rule)",
        "P3_R_limited": "G = recall(R = 8192) - recall(R = 512) at c in {8, 32}, nprobe in {32, 128, 512}: G >= 0.010 R_LIMITED, G < 0.003 NOT_R_LIMITED, else PARTLY (T1R's rule); and the smallest R in {512, 2048, 8192} within 0.003 of the R = 8192 recall at nprobe 128 for c = 8 and c = 32 ('not reached' if none)",
        "P4_transfer_with_larger_R": "D8192 = recall(c = 32, L437, R = 8192) - the frozen WebQSP R = 512 baseline at nprobe 32 and 128: HOLDS if both >= -0.010 (T1R's rule), else DEGRADES",
        "P5_adc_rank": "descriptive: the share of the true neighbours missed at R = 512 that lie in ADC rank 512 .. 8191 (a depth problem), in a probed list beyond 8192, or in no probed list (a probing problem)"},
    "decision_table": {
        "P1 AGREE and P3 R_LIMITED and P4 HOLDS": "T1 / T1R stand on a spread sample; the lever is the re-rank depth R (grown with n); the value for 245 M is measured at T3 on the full encode, not extrapolated; the choice of R stays the user's",
        "P1 STRIDED_LOWER or STRIDED_HIGHER": "the node-order prefix was biased: T2 supersedes T1 / T1R numbers for the transfer question (T1 / T1R stay as records); the R decision waits for T3 either way",
        "P3 NOT_R_LIMITED or PARTLY on the sample": "R is not the whole cause on a spread sample: report the R curve and the P5 decomposition, and the next lever (more lists probed, a hierarchical coarse quantiser, GPU exact re-score of a candidate set) is the user's call",
        "P4 DEGRADES at R = 8192": "even the large R does not restore the WebQSP level at 4 M rows: the user chooses between a larger R, a different scoring, or accepting the loss",
        "all cases": "no end-to-end claim and no KNN family is produced; the end-to-end gate stays unrun at scale; T3 (the full 245 M, GPU coarse quantiser) follows the full encode"},
    "not_claimed": ["no claim about 245 M rows from 4.2 M (T3 measures it)", "no change to addendum 10's R = 512 operating point; R > 512 cells are measurements, not a chosen operating point",
                    "no claim that the sample is a random sample of names: it is a systematic stratified sample of chunks (each chunk is a contiguous block of node first-occurrence order); within a chunk the order is the node order",
                    "no claim about the end-to-end effect of the recall differences (the gate's noise floor, F = 8 rows, cannot separate them)"],
    "next_declared": ["T3: the full 245.35 M database (GPU coarse quantiser), after the full encode and VERIFY; the R choice is made on T2 / T3 evidence (user ruling 2026-10-01)"],
    "code_pinned": code,
}
assert not os.path.exists(OUT), "write-once"
with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rec, indent=1, ensure_ascii=False))
os.replace(OUT + ".tmp", OUT)
print("written", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
