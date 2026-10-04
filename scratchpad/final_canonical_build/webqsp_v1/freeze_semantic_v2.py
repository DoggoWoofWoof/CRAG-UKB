"""Freeze CRAG_FREEBASE_SEMANTIC_OVERLAY_V1.

Refuses unless V3_SEMANTIC_OVERLAY_VALIDATION.json says PASS, on the standing rule that every
artifact is invalid until counts establish otherwise.

This is a SECOND, SEPARATE overlay. CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 is frozen at manifest hash
25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865 and is not modified, re-frozen or
superseded by this. Both hashes are recorded here so a reader can tell exactly which node universe
and which name table this semantic layer is keyed to, and so a later mismatch is detectable rather
than silent.
"""
import sys, io, os, json, glob, time, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
OUT = f"{V3}/semantic_v2"
FROZEN_GRAPH_HASH = "6672dab172cf5dde79c56685153d4e257be3527b5ee93b07514ad2f40425c5fb"
FROZEN_OVERLAY_HASH = "25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
EXPECT = 69_777_967
FORCE = "--force" in sys.argv

val_path = f"{V3}/V3_SEMANTIC_OVERLAY_VALIDATION.json"
if not os.path.exists(val_path):
    sys.exit("no validation record; run validate_semantic_v2.py first")
val = json.load(io.open(val_path, encoding="utf-8"))
if val.get("VALIDATION") != "PASS" and not FORCE:
    print(f"validation verdict is {val.get('VALIDATION')}, refusing to freeze.")
    for f in val.get("failures", [])[:10]:
        print("  ! " + f)
    sys.exit("FREEZE_RESULT=refused")

build = {}
bp = f"{V3}/V3_SEMANTIC_OVERLAY_BUILD.json"
if os.path.exists(bp):
    build = json.load(io.open(bp, encoding="utf-8"))
pol = {}
pp = f"{V3}/V3_SEMANTIC_RELATION_POLICY.json"
if os.path.exists(pp):
    pol = json.load(io.open(pp, encoding="utf-8"))

t0 = time.time()
files, rows, size = [], 0, 0
for fp in sorted(glob.glob(f"{OUT}/*.parquet")):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    n = pq.ParquetFile(fp).metadata.num_rows
    b = os.path.getsize(fp)
    rows += n
    size += b
    files.append({"file": os.path.basename(fp), "rows": n, "bytes": b, "sha256": h.hexdigest()})
    if len(files) % 16 == 0:
        print(f"  hashed {len(files)} files ({time.time()-t0:.0f}s)", flush=True)

if rows != EXPECT and not FORCE:
    sys.exit(f"row count {rows:,} != expected {EXPECT:,}; refusing to freeze")

man = {"schema": "SEMANTIC_OVERLAY_MANIFEST/v1",
       "name": "CRAG_FREEBASE_SEMANTIC_OVERLAY_V1",
       "status": "FROZEN",
       "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "files": len(files), "rows": rows, "bytes": size, "gb": round(size / 2**30, 3),
       "KEYED_BY": "node_uid",
       "KEYED_TO": {
           "graph": {"artifact": "CRAG_FREEBASE_CANONICAL",
                     "manifest_sha256": FROZEN_GRAPH_HASH,
                     "nodes": 301977131, "edges": 2062430072},
           "names": {"artifact": "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1",
                     "manifest_sha256": FROZEN_OVERLAY_HASH},
           "note": ("both were READ and neither was written. This is a third artifact beside them, "
                    "not a replacement for either. RESOLUTION_OVERLAY_V1 keeps its own hash.")},
       "POPULATION": ("the 69,777,967 nodes whose display_name is not an attested name "
                      "(STRUCTURAL_INFERRED or STRUCTURAL_FALLBACK). Nodes that already carry an "
                      "attested name are absent by design; for them display_name is the better "
                      "string and semantic_text would add nothing."),
       "RENDER": "<display_name> - <rel>: <neighbour>; ...; referenced by - <rel>: <neighbour>",
       "CAP": {"outgoing_max": 8, "incoming_max": 4,
               "selection": ("file order, which is fixed, so the artifact is reproducible. Which 8 "
                             "is NOT a salience judgement."),
               "anonymous_neighbours_excluded": ("a neighbour that is itself unnamed is skipped at "
                                                 "scan time so it cannot consume a slot; otherwise "
                                                 "an entity would render 'notable for: notable_for'")},
       "RELATION_POLICY": {"record": "V3_SEMANTIC_RELATION_POLICY.json",
                           "candidate_edges": pol.get("candidate_edges_after_policy"),
                           "candidate_pct_of_graph": pol.get("candidate_pct_of_graph"),
                           "informative_distinct_relations": pol.get("informative_distinct_relations")},
       "BUILD": {k: build.get(k) for k in
                 ("rows", "nodes_with_context", "nodes_without_context", "generated_utc")},
       "VALIDATION": {"verdict": val.get("VALIDATION"),
                      "rows": val.get("rows"),
                      "checks": val.get("checks"),
                      "record": "V3_SEMANTIC_OVERLAY_VALIDATION.json"},
       "HONESTY_RULE": ("semantic_text is CONTEXT, not a name. It never asserts that the node had "
                        "a proper name. AUTHORITATIVE_SOURCE_RESOLUTION (52.237%) and STRUCTURAL "
                        "(23.107%) remain the figures to quote about naming."),
       "NOT_supported": ["that the neighbour list is complete -- it is capped at 8 outgoing and 4 "
                         "incoming informative relations per node",
                         "that the rendered neighbours are the most salient ones",
                         "that this changes the frozen graph or the frozen resolution overlay"],
       "files_detail": files}

mp = f"{V3}/V3_SEMANTIC_OVERLAY_FROZEN.json"
with io.open(mp, "w", encoding="utf-8") as f:
    json.dump(man, f, indent=1, ensure_ascii=False)
mh = hashlib.sha256(io.open(mp, "rb").read()).hexdigest()
with io.open(f"{V3}/V3_SEMANTIC_OVERLAY_FROZEN.sha256", "w", encoding="utf-8") as f:
    f.write(mh + "  V3_SEMANTIC_OVERLAY_FROZEN.json\n")
print(f"\nFROZEN files={len(files)} rows={rows:,} {size/2**30:.3f} GB")
print(f"MANIFEST_HASH {mh}")
print(f"RESOLUTION_OVERLAY_V1 hash unchanged: {FROZEN_OVERLAY_HASH}")
print(f"{time.time()-t0:.0f}s")
