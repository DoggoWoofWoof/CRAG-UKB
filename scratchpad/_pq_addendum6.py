"""FBX_SCALE declaration addendum 6 (write-once): Track C, the compressed-KNN calibration on WebQSP, declared BEFORE any PQ-KNN arm is searched, partitioned or scored.

The user's ruling of 2026-09-30 (late): the Freebase KNN family is built from the SAME canonical Qwen encoder with compressed ANN storage, batch-encode -> quantise immediately -> discard the float
batch; it is validated FIRST on WebQSP (canonical KNN versus PQ-KNN from the same embeddings) and accepted only if the END-TO-END recall survives (H4_{S+KNN_PQ} -> PHG -> ALL-gold vs the frozen
PHG_SK), before any Freebase vector is encoded.  Nothing PQ-related has been run when this file is written.  Usage: python scratchpad/_pq_addendum6.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_6.json"
A = [FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_%d.json" % i for i in (1, 2, 3, 4, 5)]
CODE = ["scratchpad/_pq_knn.py", "scratchpad/_pq_phg.py", "scratchpad/_pq_eval.py", "scratchpad/_ml2_eval.py", "scratchpad/_fbx_calib.py", "scratchpad/_l1h_host.py", "src/l1_canonical/hypergraph.py",
        "scratchpad/modal_canonical_knn.py", "scratchpad/_fbx_name_census.py", "scratchpad/_fbx_ner_build.py"]


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
        "stage": "FBX_SCALE / addendum 6: compressed-KNN (PQ) calibration on WebQSP (Track C), declared before any PQ arm is searched or scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {p: sha(p) for p in A},
        "status": "DEVELOPMENT (calibration of a systems substrate); nothing here selects a retrieval mechanism",
        "user_ruling_2026-09-30_late": {
            "partitioner": "stop the M1-M4 search (M2 is the best coarsener; do not chase 7.7x contraction); architecture = modest M2 coarsening (about 2x) -> disk-backed / out-of-core PHG -> projection + refinement; verify on WebQSP that the out-of-core path reproduces the in-memory result (Track B, separate addendum)",
            "NER": "approved: the pinned legacy recipe in a project-owned env (crag-ner); run once, hash, freeze E_NER^FB; NER stays OUT of the partition hypergraph (H4_SK = STRUCT + KNN) unless explicitly ablated",
            "KNN": "same canonical Qwen encoder, compressed ANN storage (IVF-PQ / OPQ-PQ), batch-encode -> quantise immediately -> discard the float batch; validate FIRST on WebQSP: canonical KNN vs PQ-KNN from the same embeddings, neighbour recall@k AND end-to-end H4_{S+KNN_PQ} -> PHG -> ALL-gold recall vs PHG(S+KNN_canonical); accept only if end-to-end recall is preserved",
            "sequence": "freeze the Qwen encoding contract -> validate compression on WebQSP -> encode 302M in deterministic resumable chunks converted immediately to the validated representation -> ANN index -> freeze the node -> K-nearest edge list (+ similarity) -> canonical H4_SK -> partition",
            "tracks": "A substrate (STRUCT done, NER running, names dedup), B scalable partitioner, C compressed-KNN calibration: in parallel; all compute on the host (user, 2026-09-30: 'run on host, stop overkilling this laptop')"},
        "disclosure_of_what_was_read_first": {
            "frozen_knn_definition": "data/final_canonical/webqsp/graph/GRAPH_MANIFEST.json and scratchpad/modal_canonical_knn.py: exact, fp16 storage, fp32 renormalise + accumulate, TF32 off, search_k 4 incl. self, drop self, k = 3, over DISTINCT source rows, every position inherits its row's neighbours mapped to the lowest position of each neighbour, undirected, weight = cosine; l1_canonical keys: KNN = pairs \\ STRUCT",
            "freebase_name_census": "during the name-dedup census the per-bucket group counts were seen (about 3.83M distinct names per 4.7M names in a hash bucket, i.e. roughly 80 % distinct): it informs the encoding workload, not the calibration",
            "pq_results": "NONE: no PQ code has been trained, no decoded vector searched, no partition built or scored when this file is written (a synthetic-data unit test and a faiss crash diagnosis (OPQ needs at least as many training rows as dimensions) are the only runs)"},
        "arms": {
            "reference": "PHG_SK = the seven frozen WebQSP PHG maps on H4_SK (STRUCT u canonical KNN, NP 4); K in {100, 250, 500} are scored",
            "PQ64, PQ128, PQ192": "OPQ rotation (faiss OPQMatrix(1536, m), 20 training iterations: the faiss default of 50 was too slow on the host for a development calibration; fixed here, not tuned) then ProductQuantizer(1536, m, 8 bit): m bytes per distinct row (64 -> 48x, 128 -> 24x, 192 -> 16x smaller than fp16 1536-d); the training sample is 131,072 distinct rows drawn with RandomState(0); every other row is unseen by training; ALL distinct rows are encoded",
            "search": "the deployed regime: floats are discarded at encode time, so query and database are both DECODED vectors; the search is EXHAUSTIVE decoded-vs-decoded (the IVF/nprobe approximation is a separate measurement against this truth, not mixed into it); similarity = cosine of the fp32-renormalised decoded vectors; SEARCH_K 4 incl. self, then the canonical merge (inheritance to duplicate positions, lowest position, undirected, highest weight wins a duplicated pair); KNN key set = pairs \\ STRUCT (the l1_canonical rule); STRUCT unchanged",
            "partition": "the frozen H4_SK hypergraph rule and the frozen Zoltan PHG driver (NP 4, repair, validity gate), unchanged, on H4_{STRUCT u KNN_PQ<m>}; the keys -> hypergraph path is first checked to reproduce the frozen H4_SK_k100 content digest on the frozen keys (CONTROL)",
            "distinct_rows": "rows = classes of bitwise-identical fp16 vectors of the canonical dense docs, numbered by ascending first position; their count must equal the frozen KNN's n_distinct_source_rows (1,791,533)",
            "exact_recompute": "the canonical exact search is recomputed on the host (scratchpad/_pq_knn.py EXACT) and its pair set compared with graph/knn.npz; the FROZEN family remains the reference of every gate; the recompute serves the neighbour-recall diagnostic only"},
        "gates": {
            "end_to_end (the only accepting gate)": "addendum 1's verdict rule, unchanged, via scratchpad/_pq_eval.py (scratchpad/_ml2_eval.py's arithmetic): rows below PHG_SK's ALL-gold count at matched B_P, over every B_N in {100, 250, 500, 1000, 2000, 5000}, at each of K in {100, 250, 500}: at most 7 of 786 = RECALL_PRESERVING, at least 24 = FAILED, else BORDERLINE; also FAILED if B_P = K on more than half the rows; the PHG cells of the reference must reproduce bit for bit the earlier EVAL (control)",
            "diagnostics (not gating)": "row-wise neighbour recall@3 and nearest-neighbour recall against the exact search; recall of the exact KNN pair set and precision of the PQ one; reconstruction cosine (mean, 5th percentile) on 100,000 rows; bytes per row"},
        "decision_table": {
            "an arm is RECALL_PRESERVING at K 100, 250 and 500": "ACCEPTED; the smallest accepted m (bytes per row) is the representation frozen into the Freebase encoding contract; the Freebase encode then starts",
            "no arm is RECALL_PRESERVING": "reported with the three tables; NO Freebase vector is encoded; the next step is the user's call (larger m, a refine stage, another quantiser)",
            "the smallest passing arm is BORDERLINE": "reported to the user; no Freebase encode until ruled",
            "neighbour recall alone": "never accepts or rejects an arm"},
        "not_done": {
            "ivf_nprobe": "the IVF-PQ approximation at Freebase scale (nprobe against exhaustive-PQ truth on a sample) is a separate, later measurement",
            "freebase_encoding": "no Freebase vector is encoded before an arm is accepted; the Qwen encoding contract record is written before the first chunk"},
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
