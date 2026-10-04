"""Do the 562 unresolved mediator type MIDs have any instances? Append-only; changes nothing.

This is the question the whole 562 investigation exists to answer. PASS A found 2,211 mediator
type MIDs and could map 1,649 to type paths; the other 562 stayed unresolved, so if any node
declares its type AS ONE OF THOSE MIDS, that node is a mediator we did not classify and
CVT_MEDIATOR = 52,272,634 is an under-count.

Stage 1B showed the 239 MIDs whose paths were recoverable have zero instances, but that left the
6,698 instantiated paths that could not be mapped back to a MID as the place a missed mediator
could still hide. This tests the direct form of the question instead of the contrapositive: scan
the type column itself for the raw MIDs.

The frozen artifact is not modified. 301,977,131 nodes / 2,062,430,072 edges either way.
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
frz = json.load(io.open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
dec = frz["DECLARATIONS"]
unres = dec["MEDIATOR_MIDS_UNRESOLVED"]
unres = sorted(unres.keys() if isinstance(unres, dict) else unres)
known = dec["MEDIATOR_TYPE_PATHS"]
known = set(known.keys() if isinstance(known, dict) else known)
print(f"unresolved mediator MIDs: {len(unres):,}   known mediator paths: {len(known):,}")
vs = pa.array(unres, pa.string())

t0 = time.time()
pf = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
hits, rows = {}, 0
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["subject", "type"])
    rows += t.num_rows
    m = pc.is_in(t["type"], value_set=vs)          # 562-value set: small, safe to use is_in
    if pc.sum(pc.cast(m, "int64")).as_py():
        f = t.filter(m)
        for s, ty in zip(f["subject"].to_pylist(), f["type"].to_pylist()):
            hits.setdefault(ty, []).append(s)
    if rg % 100 == 0:
        print(f"  rg {rg}/{pf.num_row_groups} rows={rows:,} hits={sum(len(v) for v in hits.values()):,}"
              f" ({time.time()-t0:.0f}s)", flush=True)

n_inst = sum(len(v) for v in hits.values())
print(f"\ntype rows scanned            : {rows:,}")
print(f"rows typed by an unresolved MID: {n_inst:,}")
print(f"distinct unresolved MIDs used  : {len(hits):,} / {len(unres):,}")
for ty, subs in list(hits.items())[:10]:
    print(f"   {ty}  x{len(subs):,}  e.g. {subs[:3]}")

rec = {"schema": "MEDIATOR_RESOLUTION_STAGE2/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("this audit does not modify canonical/. The frozen node and edge universe "
                       "stays at 301,977,131 nodes and 2,062,430,072 edges regardless of the "
                       "result."),
       "QUESTION": ("do any of the 562 unresolved mediator type MIDs appear as a VALUE in "
                    "type.object.type? If so, those subjects are mediators we did not classify "
                    "and CVT_MEDIATOR = 52,272,634 is an under-count."),
       "METHOD": ("direct scan of the type column for the raw MIDs, over all "
                  f"{rows:,} type rows. Stage 1B tested the contrapositive (paths -> instances) "
                  "and left 6,698 unmapped instantiated paths as an open hiding place; this tests "
                  "the question directly and does not depend on the MID->path map that was the "
                  "source of the ambiguity."),
       "type_rows_scanned": rows,
       "unresolved_mediator_mids": len(unres),
       "rows_typed_by_an_unresolved_mid": n_inst,
       "distinct_unresolved_mids_instantiated": len(hits),
       "VERDICT": ("NO_INSTANCES: the 562 unresolved mediator MIDs type nothing in the graph, so "
                   "they cannot account for any missing CVT. The CVT population is complete with "
                   "respect to this question."
                   if n_inst == 0 else
                   f"INSTANCES_FOUND: {n_inst:,} subjects are typed by an unresolved mediator MID. "
                   "CVT_MEDIATOR is an under-count by at most that many nodes and the finding must "
                   "be carried as a known limit of the frozen artifact."),
       "instances": {k: v[:50] for k, v in list(hits.items())[:50]},
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE2.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print("\n" + rec["VERDICT"])
