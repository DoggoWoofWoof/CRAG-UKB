"""TIER 1 against the REAL target: the 19,387,189 ENTITY_MIDs with no display text.

The 562 mediators taught the lesson -- PASS A's naming rule used type.object.name only, and a node
with no name can still carry a perfectly human-readable /wikipedia/en/... key, an alias, or a
description. Those are already in our frozen metadata, cost nothing to acquire, and carry the
strongest provenance available (CURRENT_FREEBASE_EXACT). Measure that reach BEFORE leaning on any
historical dump.
"""
import glob, json, time
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"

t0 = time.time(); chunks = []
for sh in sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet")):
    pf = pq.ParquetFile(sh)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["node_id", "kind", "display_text"])
        m = pc.and_(pc.equal(t.column("kind"), "ENTITY_MID"), pc.is_null(t.column("display_text")))
        if pc.sum(m).as_py():
            chunks.append(t.filter(m).column("node_id").combine_chunks())
unnamed = pa.concat_arrays([c.cast(pa.string()) for c in chunks]); del chunks
print(f"unnamed ENTITY_MIDs: {len(unnamed):,}  ({time.time()-t0:.0f}s)", flush=True)

res = {}
for tbl, col in (("key", "key"), ("alias", None), ("description", None)):
    pf = pq.ParquetFile(f"{V3}/canonical/metadata/{tbl}.parquet")
    cols = [f.name for f in pf.schema_arrow]
    val = col or cols[1]
    t0 = time.time(); hits = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["subject", val])
        m = pc.is_in(t.column("subject"), value_set=unnamed)
        if pc.sum(m).as_py():
            hits.append(t.filter(m).column("subject").combine_chunks().cast(pa.string()))
    if hits:
        allh = pa.concat_arrays(hits)
        distinct = len(pc.unique(allh))
    else:
        distinct = 0
    res[tbl] = {"rows": int(sum(len(h) for h in hits)), "distinct_unnamed_subjects": distinct}
    print(f"  {tbl:12s} rows={res[tbl]['rows']:>12,}  distinct unnamed subjects={distinct:>12,}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

tot = len(unnamed)
print(f"\nunnamed ENTITY_MID total      : {tot:,}")
for k, v in res.items():
    print(f"  reachable via {k:12s}: {v['distinct_unnamed_subjects']:>12,}"
          f"  ({v['distinct_unnamed_subjects']/tot*100:.3f}%)")

json.dump({
    "schema": "TIER1_UNNAMED_ENTITY_REACH/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "unnamed_entity_mids": tot,
    "REACH": res,
    "note": "distinct_unnamed_subjects counts unnamed ENTITY_MIDs that have at least one row in "
            "that metadata table. The three tables overlap; this is not a partition and the "
            "figures must not be summed.",
}, open(f"{V3}/V3_TIER1_UNNAMED_ENTITY_REACH.json", "w", encoding="utf-8"), indent=1)
print("\nwrote V3_TIER1_UNNAMED_ENTITY_REACH.json")
