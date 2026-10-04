"""FBX_SCALE declaration addendum 6A (write-once): a correction to addendum 6's `distinct_rows` clause, declared before any PQ arm is searched, partitioned or scored.

Addendum 6 said the calibration's "rows" are the classes of bitwise-identical fp16 vectors and that their count must equal the frozen KNN's 1,791,533.  The first PREP run (host, before any
PQ or exact search) found 1,791,525 bitwise classes: the frozen family keeps 8 more rows apart than bitwise equality does.  The frozen family's rows are the classes of the materialisation
pointer index (position -> source row) named by the dense docs manifest.  The calibration therefore uses THAT definition, so that the exact recompute is the frozen search on the frozen rows.
Usage: python scratchpad/_pq_addendum6a.py   (refuses to overwrite)
"""
import hashlib
import io
import json
import os
import time

ROOT = "C:/Users/Swastik/Desktop/CRAG"
os.chdir(ROOT)
FB = "results/FREEBASE_SCALE"
OUT = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_6A.json"
A6 = FB + "/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_6.json"
FIRST = FB + "/pq/WEBQSP_DISTINCT_ROWS__v1.json"
POINTER = "data/final_canonical/_history/materialize/webqsp__pointer_index__dense.npz"
CODE = ["scratchpad/_pq_knn.py", "scratchpad/_pq_phg.py", "scratchpad/_pq_eval.py", "scratchpad/_pq_addendum6.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit("refusing: %s exists (write-once)" % OUT)
    first = json.load(io.open(FIRST, encoding="utf-8"))
    rec = {
        "stage": "FBX_SCALE / addendum 6A: correction of addendum 6's distinct-row definition (Track C), before any PQ arm is searched or scored",
        "declared": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extends": {A6: sha(A6)},
        "status": "DEVELOPMENT (calibration of a systems substrate); a definition fix, no result exists",
        "what_happened": {
            "first_prep": "WEBQSP_DISTINCT_ROWS__v1.json (kept, not edited): rows = classes of bitwise-identical fp16 vectors -> %d rows; the frozen KNN manifest says %d; the count assertion failed and PREP stopped; no search, PQ training or scoring had run" % (first["n_distinct_rows"], first["frozen_knn_manifest_n_distinct_source_rows"]),
            "reading": "the frozen family's source rows are the pointer-index classes (data/final_canonical/webqsp/embeddings/dense/docs/manifest.json 'materialised': pointer_index, distinct_source_rows 1,791,533); 8 of them carry vectors that are bitwise equal to another row's and were nevertheless searched as separate rows",
            "first_attempt_arrays": "data/freebase_scale/pq/pos2u.npy and first_pos.npy of the first attempt were moved (host) to data/freebase_scale/pq/_attempt1_bitwise/, not used"},
        "supersedes_in_addendum_6": {
            "distinct_rows": "rows = the classes of the frozen materialisation pointer index %s (sha256 %s; src all 0), numbered by ascending first position; their count must equal the frozen KNN's n_distinct_source_rows (1,791,533); every position's vector must be bitwise equal to its row representative's (rechecked); the number of bitwise classes of the same vectors (1,791,525) is recorded alongside" % (POINTER, sha(POINTER)),
            "consequence": "the exact recompute (EXACT) is the frozen search on the frozen rows; the PQ arms search the same rows; everything else in addendum 6 (arms, search regime, partition path, gates, decision table, not_done) stands unchanged",
            "new_record": "results/FREEBASE_SCALE/pq/WEBQSP_DISTINCT_ROWS__v2.json"},
        "code_pinned": {c: sha(c) for c in CODE if os.path.exists(c)},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
