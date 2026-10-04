import io, os

P = "scratchpad/final_canonical_build/mirror_audit.py"
src = io.open(P, encoding="utf-8").read()

ANCHOR = '            "reuse_map/index.json")'
NEW = '''            # --- V1 build + NSM acquisition branch, 2026-09-06 ---
            # The Route-A acquisition record (pre-registration written before any payload byte, the
            # bytes/sha256 provenance, the MID-preservation test that converted [I]->[V]) and the V1
            # section 1/2 outputs.  All are written deliverables; without them here they live only
            # under the gitignored data/ tree and the branch would be unreproducible from the repo.
            "NSM_ACQUISITION_PREREGISTRATION.json", "NSM_ACQUISITION_PROVENANCE.json",
            "NSM_MID_PRESERVATION.json", "NSM_UNION_RESULT.json", "NSM_NAMING_RESOURCE_AUDIT.json",
            "NSM_PROJECTION_DIAGNOSTIC.json", "NAME_MAPPING_COVERAGE.json",
            "ROG_ATTRIBUTABLE_COLLAPSE.json", "COLLISION_DEGREE_TEST.json",
            "ROG_UNION_REBUILD.json", "V1_TABLES_REPORT.json", "V1_RELATION_LABEL_ANALYSIS.json",
            # v1/*.parquet are NOT listed: 78 MB of build output, not a report.  V1_TABLES_REPORT
            # carries their counts, byte sizes and NODE_ORDER/CONTENT hashes, which is what the
            # tracked mirror needs to detect drift.
            "reuse_map/index.json")'''

assert src.count(ANCHOR) == 1, f"anchor found {src.count(ANCHOR)}x"
src = src.replace(ANCHOR, NEW)
io.open(P + ".tmp", "w", encoding="utf-8", newline="\n").write(src)
os.replace(P + ".tmp", P)
print("mirror_audit.py extended")
