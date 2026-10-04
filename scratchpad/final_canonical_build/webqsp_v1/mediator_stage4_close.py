"""Stage 4: do the newly recovered mediator paths have instances? Closes the 562. Append-only.

Stage 1B gave paths for 239 of the 562 from keys in the frozen metadata (zero instances).
Stage 3 gave paths for a further 164 from the deleted-triples dump. This tests all of them
against type.object.type. Any instance found is a node that should have been CVT_MEDIATOR and was
not, which would make CVT_MEDIATOR = 52,272,634 an under-count.

Changes nothing: 301,977,131 nodes / 2,062,430,072 edges either way.
"""
import sys, io, json, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"

def canon(p):
    """/base/popstra/lawsuit -> base.popstra.lawsuit"""
    return ".".join(x for x in p.split("/") if x)

b1b = json.load(io.open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE1B.json", encoding="utf-8"))
s3 = json.load(io.open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE3.json", encoding="utf-8"))
frz = json.load(io.open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
known = frz["DECLARATIONS"]["MEDIATOR_TYPE_PATHS"]
known = set(known.keys() if isinstance(known, dict) else known)

path_of = collections.defaultdict(set)
for mid, v in (b1b.get("MID_TO_PATH_FROM_KEY") or {}).items():
    for p in (v if isinstance(v, list) else [v]):
        path_of[mid].add(canon(p))
n_from_frozen = len(path_of)
for mid, v in (s3.get("NEWLY_RESOLVED") or {}).items():
    for p in (v if isinstance(v, list) else [v]):
        path_of[mid].add(canon(p))

cands = sorted({p for v in path_of.values() for p in v})
new_paths = sorted(set(cands) - known)
print(f"MIDs with a path: {len(path_of)} ({n_from_frozen} from frozen keys, "
      f"{len(path_of)-n_from_frozen} added by the deleted dump)")
print(f"candidate paths: {len(cands):,}   of which NOT already a known mediator path: {len(new_paths):,}")

vs = pa.array(new_paths, pa.string())
t0 = time.time()
pf = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
found = collections.Counter()
rows = 0
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["subject", "type"])
    rows += t.num_rows
    m = pc.is_in(t["type"], value_set=vs)
    if pc.sum(pc.cast(m, "int64")).as_py():
        for v in pc.value_counts(t.filter(m)["type"]).to_pylist():
            found[v["values"]] += v["counts"]
print(f"type rows scanned: {rows:,}  ({time.time()-t0:.0f}s)")

inst = sum(found.values())
mids_left = [m for m in (frz["DECLARATIONS"]["MEDIATOR_MIDS_UNRESOLVED"]) if m not in path_of] \
    if isinstance(frz["DECLARATIONS"]["MEDIATOR_MIDS_UNRESOLVED"], list) else \
    [m for m in frz["DECLARATIONS"]["MEDIATOR_MIDS_UNRESOLVED"] if m not in path_of]

print(f"\ninstances under a newly recovered mediator path: {inst:,}")
for p, c in found.most_common(15):
    print(f"   {c:>8,}  {p}")
print(f"MIDs of the 562 still with NO path at all: {len(mids_left)}")

rec = {"schema": "MEDIATOR_RESOLUTION_STAGE4/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("does not modify canonical/. 301,977,131 nodes / 2,062,430,072 edges "
                       "regardless of the result."),
       "QUESTION": "do the recovered mediator type paths have instances we failed to class as CVT?",
       "mids_of_562_with_a_path": len(path_of),
       "paths_from_frozen_keys": n_from_frozen,
       "paths_added_by_deleted_dump": len(path_of) - n_from_frozen,
       "mids_still_with_no_path": len(mids_left),
       "candidate_paths": len(cands),
       "candidate_paths_not_already_known_mediators": len(new_paths),
       "type_rows_scanned": rows,
       "instances_found": inst,
       "instances_by_path": dict(found.most_common(50)),
       "VERDICT": (
         f"CLOSED. {len(path_of)} of the 562 unresolved mediator MIDs now have a type path "
         f"({n_from_frozen} from keys in the frozen metadata, {len(path_of)-n_from_frozen} from the "
         f"deleted-triples dump), and NONE of those paths has a single instance among "
         f"{rows:,} type rows. Combined with stage 2 (no instantiation by raw MID), the 562 "
         f"account for zero missing CVT nodes. CVT_MEDIATOR = 52,272,634 stands."
         if inst == 0 else
         f"UNDER-COUNT FOUND. {inst:,} nodes instantiate a mediator path recovered for the 562 and "
         f"were not classified CVT_MEDIATOR. This is a known limit of the frozen artifact and must "
         f"travel with it; the frozen graph is NOT edited to hide it."),
       "RESIDUAL": (f"{len(mids_left)} of the 562 still have no recoverable path from any source, "
                    f"so they cannot be tested by path. Stage 2 showed none of the 562 is used as "
                    f"a raw MID in type.object.type, which is the only other way they could be "
                    f"instantiated. The residual risk is therefore bounded by these "
                    f"{len(mids_left)} MIDs having an unknown path that coincides with one of the "
                    f"6,698 unmapped instantiated paths from stage 1C."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE4.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print("\n" + rec["VERDICT"])
print("RESIDUAL: " + rec["RESIDUAL"])
