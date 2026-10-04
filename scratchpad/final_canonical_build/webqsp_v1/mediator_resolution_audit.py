"""MEDIATOR_RESOLUTION_AUDIT -- append-only. Does NOT touch the frozen graph.

PASS A left 562 of 2,211 declared mediator type MIDs unresolved to a type path, and recorded that
as a bounded under-count of CVT_MEDIATOR. Its own reading was that this is "what a declared type
with no instances and no properties looks like" -- but that was a reading, not a measurement.

Before reaching for an external source (the deleted-triples dump), ask what OUR OWN frozen
metadata says. PASS A's join required a key naming a type THAT OCCURS IN THE DATA. A MID whose key
names a type with no instances would have failed that join while still carrying a perfectly good
human-readable path in the key table. Those are recoverable at zero cost and from the source of
record, which is strictly better provenance than any external dump.
"""
import json, time, sys
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
MD = f"{V3}/canonical/metadata"
frz = json.load(open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
UNRES = frz["DECLARATIONS"]["MEDIATOR_MIDS_UNRESOLVED"]
RESOLVED_PATHS = set(frz["DECLARATIONS"]["MEDIATOR_TYPE_PATHS"])
MID2PATH = frz["DECLARATIONS"]["MID_TO_TYPE_PATH"]
assert len(UNRES) == 562, len(UNRES)
targets = pa.array(sorted(UNRES))
tset = set(UNRES)
print(f"{len(UNRES)} unresolved mediator MIDs", flush=True)

def probe(table, cols, key="subject"):
    """stream a metadata table, keep only rows whose subject is one of the 562."""
    t0 = time.time()
    pf = pq.ParquetFile(f"{MD}/{table}.parquet")
    out = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=cols)
        m = pc.is_in(t.column(key), value_set=targets)
        if pc.sum(m).as_py():
            out.extend(t.filter(m).to_pylist())
    print(f"  {table:16s} {len(out):>6,} rows  ({time.time()-t0:.0f}s)", flush=True)
    return out

hits = {}
for tbl, cols in (("key", ["subject", "key"]),
                  ("name", ["subject", "lexical", "lang"]),
                  ("type", ["subject", "type"]),
                  ("type_hints", ["subject", "field", "value"]),
                  ("property_schema", ["subject", "field", "value"]),
                  ("alias", None),
                  ("description", None)):
    if cols is None:
        pf = pq.ParquetFile(f"{MD}/{tbl}.parquet")
        cols = [f.name for f in pf.schema_arrow]
    hits[tbl] = probe(tbl, cols)

# --- what did the key table give us that PASS A's join discarded? -------------------------------
# a freebase key like /american_football/football_game_score is the same path in slash form; the
# type path is the dotted form. Convert and check whether it is a NEW path or one we already hold.
def key_to_path(k):
    return k.strip("/").replace("/", ".")

by_mid = {}
for r in hits["key"]:
    by_mid.setdefault(r["subject"], []).append(r["key"])

recovered, already_known, non_type_keys = {}, {}, {}
for mid, keys in by_mid.items():
    cand = [key_to_path(k) for k in keys]
    new = [c for c in cand if c.count(".") == 1]   # domain.type shape
    if not new:
        non_type_keys[mid] = keys
    elif any(c in RESOLVED_PATHS for c in new):
        already_known[mid] = new
    else:
        recovered[mid] = new

print()
print(f"MIDs with at least one key row      : {len(by_mid):>4} / 562")
print(f"  -> key yields a domain.type path  : {len(recovered) + len(already_known):>4}")
print(f"       of which path already held   : {len(already_known):>4}")
print(f"       of which NEWLY recovered     : {len(recovered):>4}")
print(f"  -> key is not type-shaped         : {len(non_type_keys):>4}")
print(f"MIDs with a name row                : {len({r['subject'] for r in hits['name']}):>4} / 562")
print(f"MIDs appearing as a type SUBJECT    : {len({r['subject'] for r in hits['type']}):>4} / 562")
print(f"MIDs with type_hints rows           : {len({r['subject'] for r in hits['type_hints']}):>4} / 562")

json.dump({
    "schema": "MEDIATOR_RESOLUTION_AUDIT/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "APPEND_ONLY": "this audit does not modify canonical/. The frozen node and edge universe stays "
                   "at 301,977,131 nodes and 2,062,430,072 edges regardless of what it finds.",
    "unresolved_input": len(UNRES),
    "STAGE_1_OUR_OWN_METADATA": {
        "mids_with_key_rows": len(by_mid),
        "key_yields_type_path": len(recovered) + len(already_known),
        "path_already_held": len(already_known),
        "NEWLY_RECOVERED": len(recovered),
        "key_not_type_shaped": len(non_type_keys),
        "mids_with_name": len({r["subject"] for r in hits["name"]}),
        "mids_as_type_subject": len({r["subject"] for r in hits["type"]}),
        "mids_with_type_hints": len({r["subject"] for r in hits["type_hints"]}),
    },
    "RECOVERED": recovered,
    "ALREADY_KNOWN": already_known,
    "NON_TYPE_KEYS": non_type_keys,
    "NAMES": {r["subject"]: r["lexical"] for r in hits["name"]},
    "TYPE_ROWS": hits["type"][:400],
    "TYPE_HINT_ROWS": hits["type_hints"][:400],
}, open(f"{V3}/V3_MEDIATOR_RESOLUTION_AUDIT.json", "w", encoding="utf-8"), indent=1)
print("\nwrote V3_MEDIATOR_RESOLUTION_AUDIT.json")
