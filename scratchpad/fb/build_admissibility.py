# -*- coding: utf-8 -*-
"""Build CRAG_FREEBASE_INFERENCE_ADMISSIBILITY_V1: a row-for-row sidecar over the inference
overlay marking which generated names are admissible as NAMES.

THE FINDING THIS ACTS ON
    INFERENCE_RELATION_HEAD_CENSUS_V2 measured all 64,038,024 overlay rows exactly. The defect is
    confined to one rule. RELATION_SIGNATURE_NAME produces 2,210,625 rows and 2,015,215 of them
    (91.160%) are headed by a bookkeeping relation. Every other source is effectively clean:
    MEDIATOR_ROLE_NAME 671 of 51,938,570, MERGE_SUCCESSOR_NAME 21 of 1,513,307,
    TYPE_AND_NEIGHBOR_NAME 0, SCHEMA_PATH_NAME 0.

    Exactly four heads account for all 2,015,215 rows, with remainder zero:
        Last referenced by  1,298,787      Id           4,748
        Freeq                 708,801      Permission   2,879
    No borderline term in the vocabulary (Creator, Timestamp, Namespace, ...) fires at all, so the
    rule is not resting on a judgement call about an ambiguous relation.

WHY THESE ARE REJECTED
    The governing criterion for this layer is that inference may only upgrade a weak structural
    name into a more specific SEMANTIC name: "Film performance" -> "Tom Hanks performance in
    Forrest Gump" is an upgrade; "Film performance" -> "Film performance record" is not.
    "Last referenced by of X" names a provenance relationship and "Freeq of job_<uuid>_var_<hash>"
    is a pseudo-key, not a name -- and this project already refuses the 58,144 freeq pseudo-titles
    on authority grounds, so admitting freeq-headed GENERATED names would contradict a decision
    already taken.

WHY REJECTING IS SAFE
    The standing rule is that the existing baseline is a floor the new pass may not degrade.
    Rejecting a row does not delete anything: the node falls back to its frozen display name
    ("Unnamed Freebase entity" for the 3,173,841 Group E rows), which IS the floor. So rejection
    returns a node to the baseline and can never push it below.

WHAT THIS COSTS, STATED PLAINLY
    Group E previously read as 1,298,608 upgrades over 3,173,841 rows (40.916%), all of them from
    this one rule and all reading "Last referenced by of X". Under this rule that coverage is
    withdrawn, because it was never real coverage. Recording the withdrawal is the point: the
    alternative was to freeze 1.3M vacuous names and call them semantics.

This does NOT modify the overlay. It is a sidecar with the same row order, the same pattern the
resolution overlay already uses. Per the standing contract rule this is a NEW record with its own
hash, never an edit of a frozen one.
"""
import glob, io, json, os, re, sys, time
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from collections import Counter

INF = "data/final_canonical/freebase_v3/inference_overlay_v1"
OUT = "data/final_canonical/freebase_v3/inference_admissibility_v1"
FILES = sorted(glob.glob(INF + "/*.parquet"))

BOOK_HEAD = {"last referenced by", "freeq", "permission", "permissions", "creator", "created by",
             "timestamp", "id", "guid", "key", "namespace", "attribution", "provenance",
             "last referenced", "referenced by", "write permission", "read permission"}
SPLIT = re.compile(r" of | — | in | on | for | as | about | at ")
ADMISSIBLE, REJECT_BOOKKEEPING = 0, 1

t0 = time.time()
os.makedirs(OUT, exist_ok=True)
tot = rej = 0
by_src = Counter()
by_head = Counter()
rows_by_src = Counter()

for i, f in enumerate(FILES):
    t = pq.read_table(f, columns=["inferred_name", "inference_source"])
    nm = t["inferred_name"].combine_chunks()
    src = pc.cast(t["inference_source"], "string").combine_chunks().to_numpy(zero_copy_only=False)
    dd = nm.dictionary_encode()
    dic = dd.dictionary.to_pylist()
    idx = dd.indices.to_numpy(zero_copy_only=False)
    # decide once per DISTINCT name, then fan out by index
    bad = np.zeros(len(dic), bool)
    head = np.empty(len(dic), object)
    for j, v in enumerate(dic):
        v = v or ""
        m = SPLIT.search(v)
        h = (v[:m.start()] if m else v).strip().lower()
        head[j] = h
        bad[j] = h in BOOK_HEAD
    r = np.where(bad[idx], REJECT_BOOKKEEPING, ADMISSIBLE).astype(np.int8)
    pq.write_table(pa.table({"reject_reason": pa.array(r, type=pa.int8())}),
                   os.path.join(OUT, os.path.basename(f)), compression="zstd")
    m = r != ADMISSIBLE
    tot += r.size
    rej += int(m.sum())
    for s in np.unique(src):
        sm = src == s
        rows_by_src[s] += int(sm.sum())
        by_src[s] += int((m & sm).sum())
    u, c = np.unique(idx[m], return_counts=True)
    for k, n in zip(u.tolist(), c.tolist()):
        by_head[head[k]] += n
    print("  [%d/%d] rows=%s rejected=%s  %.0fs"
          % (i + 1, len(FILES), format(tot, ","), format(rej, ","), time.time() - t0), flush=True)

rec = {"RECORD": "CRAG_FREEBASE_INFERENCE_ADMISSIBILITY_V1",
       "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "status": "MEASURED",
       "WHAT_THIS_IS": ("A row-for-row sidecar over inference_overlay_v1 marking which generated "
                        "names are admissible as NAMES. It does not modify the overlay and it "
                        "deletes nothing: a rejected node falls back to its frozen display name, "
                        "which is the baseline floor."),
       "evidence": "INFERENCE_RELATION_HEAD_CENSUS_V2 (scratchpad/fb/head_census.json)",
       "RULE": ("REJECT iff the relation head of the generated name -- the text before the first "
                "template boundary, so never the neighbour title -- is in BOOKKEEPING_HEAD_"
                "VOCABULARY. Such a name names the RECORD (provenance, access control, pseudo-key) "
                "rather than the node."),
       "reject_reason_codes": {"0": "ADMISSIBLE", "1": "REJECT_BOOKKEEPING_RELATION_NAME"},
       "TOTAL_ROWS": tot,
       "ADMISSIBLE_N": tot - rej,
       "REJECT_BOOKKEEPING_RELATION_NAME_N": rej,
       "REJECT_PCT_OF_LAYER": round(100.0 * rej / tot, 4),
       "REJECTED_BY_SOURCE": {k: int(v) for k, v in by_src.items()},
       "ROWS_BY_SOURCE": {k: int(v) for k, v in rows_by_src.items()},
       "REJECTED_BY_HEAD": {k: int(v) for k, v in by_head.most_common()},
       "BOOKKEEPING_HEAD_VOCABULARY": sorted(BOOK_HEAD),
       "WHY_SAFE": ("the frozen floor already makes every node readable, so rejection returns a "
                    "node to the baseline and cannot push it below it"),
       "COST_STATED": ("Group E read as 1,298,608 upgrades over 3,173,841 rows (40.916%), all from "
                       "RELATION_SIGNATURE_NAME/TOP_RELATION_PLUS_NEIGHBOUR and all of the form "
                       "'Last referenced by of X'. That coverage is withdrawn because it was never "
                       "real coverage."),
       "sidecar_dir": OUT, "elapsed_s": round(time.time() - t0, 1)}
json.dump(rec, io.open("scratchpad/fb/admissibility.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("\nTOTAL %s  REJECTED %s (%.4f%%)" % (format(tot, ","), format(rej, ","), 100.0 * rej / tot))
for k, v in by_head.most_common(10):
    print("   %12s  %s" % (format(v, ","), k))
print("\nby source:")
for k, v in sorted(by_src.items(), key=lambda kv: -kv[1]):
    print("   %-26s %12s of %s" % (k, format(v, ","), format(rows_by_src[k], ",")))
print("\nwrote", OUT)
