"""Freeze CRAG_FREEBASE_RESOLUTION_OVERLAY_V1.

Refuses unless V3_RESOLUTION_OVERLAY_VALIDATION.json says PASS, on the standing rule that every
artifact is invalid until counts establish otherwise. The frozen graph is not touched: its
manifest hash is copied in as a cross-reference so a reader can tell which node universe this
overlay is keyed to, and a mismatch later is detectable rather than silent.
"""
import sys, io, os, json, glob, time, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
OUT = f"{V3}/overlay_v1"
FROZEN_GRAPH_HASH = "6672dab172cf5dde79c56685153d4e257be3527b5ee93b07514ad2f40425c5fb"
FORCE = "--force" in sys.argv

val_path = f"{V3}/V3_RESOLUTION_OVERLAY_VALIDATION.json"
if not os.path.exists(val_path):
    sys.exit("no validation record; run validate_display_names.py first")
val = json.load(io.open(val_path, encoding="utf-8"))
if val.get("VERDICT") != "PASS" and not FORCE:
    print(f"validation verdict is {val.get('VERDICT')}, refusing to freeze.")
    for f in val.get("failures", [])[:10]:
        print("  ! " + f)
    sys.exit("FREEZE_RESULT=refused")

t0 = time.time()
files, rows, size = [], 0, 0
for fp in sorted(glob.glob(f"{OUT}/*.parquet")):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    n = pq.ParquetFile(fp).metadata.num_rows
    b = os.path.getsize(fp)
    rows += n; size += b
    files.append({"file": os.path.basename(fp), "rows": n, "bytes": b, "sha256": h.hexdigest()})
    if len(files) % 60 == 0:
        print(f"  hashed {len(files)} files ({time.time()-t0:.0f}s)", flush=True)

man = {"schema": "RESOLUTION_OVERLAY_MANIFEST/v1",
       "name": "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1",
       "status": "FROZEN",
       "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "files": len(files), "rows": rows, "bytes": size,
       "gb": round(size / 2**30, 3),
       "KEYED_TO": {"artifact": "CRAG_FREEBASE_CANONICAL",
                    "manifest_sha256": FROZEN_GRAPH_HASH,
                    "nodes": 301977131, "edges": 2062430072,
                    "note": "this overlay is a SIDECAR. The frozen graph was read, never written; "
                            "its manifest hash above is unchanged by this build."},
       "CONTRACT": ["V3_RESOLUTION_OVERLAY_CONTRACT.json (v1)",
                    "V3_RESOLUTION_OVERLAY_CONTRACT_V2.json (amendment: column names, tier "
                    "precedence, corrected key-tier sizing)",
                    "V3_RESOLUTION_OVERLAY_CONTRACT_V3.json (amendment: blank-is-not-a-name, "
                    "blank literals, degenerate URIs, final blank guarantee)"],
       "VALIDATION": {"verdict": val.get("VERDICT"),
                      "nodes": val.get("nodes"),
                      "DISPLAY_NAME_EMPTY": val.get("DISPLAY_NAME_EMPTY"),
                      "NAKED_MID_DISPLAY": val.get("NAKED_MID_DISPLAY"),
                      "NAKED_MID_SELF": val.get("NAKED_MID_SELF"),
                      "NAKED_MID_DERIVED": val.get("NAKED_MID_DERIVED"),
                      "MID_SHAPED_TEXT_ON_MID_KIND": val.get("NAKED_MID_SHAPED_TEXT_ON_MID_KIND"),
                      "MID_SHAPED_TEXT_IS_NOT_OPACITY": val.get("MID_SHAPED_TEXT_IS_NOT_OPACITY"),
                      "OPERATIONALLY_UNRESOLVED": val.get("OPERATIONALLY_UNRESOLVED")},
       "RESOLUTION_MIX": {"by_source": val.get("by_source"),
                          "AUTHORITATIVE_SOURCE_RESOLUTION_PCT": val.get("AUTHORITATIVE_SOURCE_RESOLUTION_PCT"),
                          "STRUCTURAL_PCT": val.get("STRUCTURAL_PCT")},
       "HOW_TO_DESCRIBE_THIS_OVERLAY": {
         "supported": ("Every node in CRAG_FREEBASE_CANONICAL has a deterministic human-readable "
                       "representation. Original or historically recovered Freebase names are used "
                       "wherever they exist; inherently anonymous or unrecoverable objects are "
                       "represented from their source-derived type and graph semantics."),
         "NOT_supported": [
           "that every node's original Freebase name was recovered -- most unnamed nodes never had one",
           "that a STRUCTURAL_INFERRED or STRUCTURAL_FALLBACK display_name is what the node was called",
           "that zero opaque nodes means zero information loss"],
         "always_report_together": ["AUTHORITATIVE_SOURCE_RESOLUTION_PCT", "STRUCTURAL_PCT"]},
       "files_detail": files,
       "elapsed_s": round(time.time() - t0, 1)}
mh = hashlib.sha256(json.dumps(man["files_detail"], sort_keys=True,
                               separators=(",", ":")).encode()).hexdigest()
man["MANIFEST_HASH"] = mh
with io.open(f"{OUT}/OVERLAY_MANIFEST.json", "w", encoding="utf-8") as f:
    json.dump(man, f, indent=1, ensure_ascii=False)
brief = {k: v for k, v in man.items() if k != "files_detail"}
with io.open(f"{V3}/V3_RESOLUTION_OVERLAY_FROZEN.json", "w", encoding="utf-8") as f:
    json.dump(brief, f, indent=1, ensure_ascii=False)
print(f"\nFROZEN  files={len(files)}  rows={rows:,}  {man['gb']} GB")
print(f"MANIFEST_HASH {mh}")
print("FREEZE_RESULT=frozen")
