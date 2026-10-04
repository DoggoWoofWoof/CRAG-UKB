"""Write addendum 11 (write-once) of the FBX_SCALE host-stage declaration: IVF transfer pilot T1 on prefixes of the real Freebase PQ64 chunk codes (Track C step 3), declared before any Freebase IVF recall exists."""
import hashlib
import io
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_11.json")


def sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


R = "results/FREEBASE_SCALE/"
ext = {R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n: sha(R + "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%s.json" % n) for n in ("1", "2", "3", "4", "5", "6", "6A", "7", "8", "9", "10")}
basis = {R + "ivf/WEBQSP_IVF_SAMPLE__v1.json": sha(R + "ivf/WEBQSP_IVF_SAMPLE__v1.json"), R + "CALIB_EVAL_WEBQSP__ivf_v1.json": sha(R + "CALIB_EVAL_WEBQSP__ivf_v1.json"),
         R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json": sha(R + "enc/FBX_QWEN_ENCODING_CONTRACT__v1.json"), R + "enc/FBX_QWEN_CODEBOOK__v1.json": sha(R + "enc/FBX_QWEN_CODEBOOK__v1.json")}
code = {c: sha(c) for c in ("scratchpad/_ivf_transfer.py", "scratchpad/_ivf_knn.py", "scratchpad/_pq_knn.py", "scratchpad/_fbx_encode.py")}

rec = {
    "stage": "FBX_SCALE / addendum 11: IVF transfer pilot T1 on prefixes of the real Freebase PQ64 codes (Track C step 3), declared before any Freebase IVF recall exists",
    "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "extends": ext,
    "basis": basis,
    "status": "DEVELOPMENT (a systems measurement); nothing here selects a retrieval mechanism, builds a KNN family, partitions anything or changes a gate",
    "question": ("Addendum 10 fixed, on WebQSP (1,791,533 distinct names), an inverted file over the PQ64 codes (4,096 lists of about 437 rows; 32 / 64 / 128 lists probed; re-rank R = 512) that keeps the end-to-end gate of addendum 1.  "
                 "The Freebase run will search 245,350,737 names.  Does that search quality TRANSFER to real Freebase codes as the database grows: how does the tie-tolerant recall@3 against the EXACT decoded-vs-decoded truth move with the database size n "
                 "(a) when the list length is held at WebQSP's value (nlist grows with n) and (b) when nlist is held at 4,096 (the list length grows with n), for the same probe counts, and what scan fraction / rows per query does each cell cost?"),
    "not_claimed": ["no end-to-end claim: neighbour recall is measured, the partition gate is not re-run; a KNN family, an ANN index file and a canonical H4_SK are not built here",
                    "no claim about 245 M rows: the largest database here is 32 chunks = 4,194,304 rows (1.7 % of the names); any statement about Freebase scale is an extrapolation from a trend over c in {4, 8, 16, 32} and is labelled so",
                    "no claim that a prefix is a sample: chunk rows are the DISTINCT NAMES in first-occurrence order of the node shards, so the first c chunks are the first c x 131,072 names of that order and may be homogeneous in type / language / source; this is a LOWER-BOUND-on-heterogeneity pilot, and a strided (cross-chunk) sample is the pre-declared T2 after the encode finishes",
                    "recall is measured WITHIN the prefix database (exact truth over the same codes); neighbours that exist only in the remaining 98 % of the names are not seen",
                    "no change to the PQ contract, the codebook, the encoder, R = 512, the arm definitions or addendum 1's thresholds; the replicate floor F = 8 of addendum 8 is not used here"],
    "design": {
        "database": "the PQ64 codes of the first c chunks, c in {1, 2, 4, 8, 16, 32}: n = c x 131,072 rows (131,072 ... 4,194,304); every chunk must verify (codes sha256, contract sha256 = aef5fb58..., codebook sha256) at load; a missing or failing chunk aborts with no partial record",
        "decode": "the accepted codebook's PQ centroid table rounded to fp16, then fp32 (addendum 10): the decoded vector is the fp16 storage vector of the exhaustive search",
        "query": "4,000 rows of the prefix, RandomState(1) without replacement, decoded-vs-decoded (the deployed regime); the four interleaved quarters of the sample give a fold spread (reported as SE = SD / 2)",
        "truth": "EXACT top 4 including self by cosine of the decoded vectors: every database row decoded and renormalised in fp32, inner product in fp32, row blocks of 32,768 (the regime of addendum 6; blocked top-4 merge asserted against brute force on a synthetic set); ties by block order are absorbed by the tie-tolerant metric",
        "layouts": {"L437": "nlist = round(n / 437.386) (437.386 = 1,791,533 / 4,096, WebQSP's mean list length): " + " / ".join("{:,}".format(int(round(c * 131072 / (1791533 / 4096.0)))) for c in (1, 2, 4, 8, 16, 32)) + " lists for c = 1, 2, 4, 8, 16, 32",
                    "F4096": "nlist = 4,096 for c >= 4 (c = 1 has fewer rows than the 64 x 4,096 training rows; c = 2 would train on every row at a list length of 64: both outside the question)"},
        "nprobe": "{32, 64, 128, 256, 512} per cell; a probe count above nlist / 2 is not run (a probe of more than half the lists is nearly an exhaustive scan and says nothing about transfer)",
        "coarse_quantiser": "faiss spherical k-means, niter 20, nredo 1, seed 2026, max_points_per_centroid 1024, trained on min(n, 64 x nlist) decoded unit-norm prefix rows (RandomState(0)) (WebQSP: 262,144 = 64 x 4,096), a row listed under its largest inner product centroid",
        "scoring": "addendum 10's, unchanged: faiss IndexIVFPQ, by_residual = False, inner product with the accepted PQ centroids, ADC = <decoded query, decoded row> exactly; best R = 512 by ADC re-ranked by cosine from the stored decoded norms; top 4 including self",
        "metrics": ["tie-tolerant recall@3 (PRIMARY; a returned neighbour counts if its cosine >= the exact third-best non-self cosine - 1e-5)", "ids recall@3, nearest-neighbour recall, self in the returned top 4",
                    "scan fraction = mean rows in the probed lists / n, and rows scanned per query", "queries per second and threads; exact-truth, coarse k-means, assignment and index-build seconds per cell"],
        "baseline": "WebQSP at nlist 4,096 (addendum 10 phase A, 100,000 queries): tie-tolerant recall@3 0.9456 / 0.9672 / 0.9838 / 0.9949 / 0.9979 at nprobe 32 / 64 / 128 / 256 / 512 (scan 0.89 / 1.75 / 3.46 / 6.84 / 13.52 %)",
        "execution": "one host job, 6 threads, resumable per stage (truth, coarse, assignment) and per cell; the final record results/FREEBASE_SCALE/ivf/FBX_IVF_TRANSFER_T1__v1.json is written once at the end"},
    "reading_rules_declared_before_any_recall_is_read": {
        "R1_fixed_probe_transfer (layout L437)": ("for nprobe in {32, 64, 128}: D(nprobe) = tie-tolerant recall at c = 32 minus the WebQSP baseline at that nprobe.  'Holds' if D >= -0.010 (about five fold standard errors at 4,000 queries); 'degrades' otherwise, with D reported.  "
                                                  "c = 32 (4.19 M rows, 9,589 lists) is the cell closest to WebQSP's size at the same list length; c = 8 (1.05 M rows, 2,400 lists) is reported beside it as the other side of WebQSP's n"),
        "R2_trend_in_n": ("for each layout and nprobe the tie-tolerant recall against log2 n over c in {4, 8, 16, 32}: the least-squares slope per doubling of n.  L437: 'flat' if |slope| <= 0.005 per doubling, 'falling' if slope < -0.005, 'rising' if > +0.005.  "
                          "The extrapolation to 245.35 M rows (5.87 doublings beyond c = 32) is computed and labelled an extrapolation; it is never used as a result"),
        "R3_probe_count_needed": "per layout and c: the smallest declared nprobe whose tie-tolerant recall >= 0.90 / 0.95 / 0.98, 'not reached' if none; for L437 the ratio of that count at c = 32 to c = 8 is reported (a ratio near 1 = a constant probe count suffices as n grows)",
        "R4_cost": "scan fraction, rows per query and measured queries per second per cell; for L437 the rows per query as a function of n beside the same for F4096 (which grows linearly with n at a constant probe count)"},
    "decision_table": {
        "R1 holds at nprobe 32 AND R2 flat or rising for L437 at nprobe 32": "the working hypothesis of the Freebase index is 'constant list length, constant probe count' (about 561,000 lists, 32 probed): the next addenda are T2 (the strided cross-chunk sample once the encode finishes) and T3 (the coarse quantiser at full size on the GPU); nothing is fixed for Freebase by this pilot",
        "R1 degrades or R2 falling for L437": "report D, the slope and the probe count R3 needs at each c; the next lever (more probes, a hierarchical coarse quantiser, a larger R, GPU exact re-scoring of a candidate set) is the user's call",
        "F4096": "a comparator only: its rows per query grow linearly with n, so a constant-nlist index does not scale to 245 M names (addendum 10 section 53.6); nothing is decided from it",
        "a cell with too few training rows / a missing chunk": "not run / abort, no partial record; no cell is substituted"},
    "next_declared": ["T2: a strided sample of chunks (every k-th chunk across the whole encode) at the same sizes, once the encode has finished, to test whether the prefix result depends on node order", "T3: the full 245.35 M database: GPU coarse quantiser training and assignment, ANN index file, node -> K nearest edge list (the user's sequence)"],
    "code_pinned": code,
}
assert not os.path.exists(OUT), "write-once"
with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rec, indent=1, ensure_ascii=False))
os.replace(OUT + ".tmp", OUT)
print("written", OUT, hashlib.sha256(open(OUT, "rb").read()).hexdigest())
