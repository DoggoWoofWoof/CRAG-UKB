"""How far do the authoritative tiers actually get on the 19,387,189 unnamed ENTITY_MIDs?

Chunked hash joins, not per-row-group is_in: the previous attempt rebuilt a 19.4M-element hash
table 200 times over and never finished. Arrow's join builds it once per chunk instead.

Reports, in priority order:
  FREEBASE_KEY_EXACT   a /type/object/key in the frozen metadata     (current, strongest)
  HISTORICAL_NAME      a /type/object/name in the deleted dump       (real name, later deleted)
  remainder            needs structural description
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
            chunks.append(t.filter(m).column("node_id").combine_chunks().cast(pa.string()))
unnamed = pa.table({"node_id": pa.chunked_array(chunks)}).combine_chunks()
N = unnamed.num_rows
del chunks
print(f"unnamed ENTITY_MIDs: {N:,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- tier 2: historical names from the deleted dump ---------------------------------------------
t0 = time.time()
dn = pq.read_table(f"{V3}/_acquisition/deleted_names.parquet", columns=["subject", "object", "lang"])
# '/m/0abc' -> 'm.0abc' so it joins to node_id
sub = pc.replace_substring(pc.replace_substring(dn.column("subject"), "/m/", "m."), "/g/", "g.")
dn = dn.set_column(0, "node_id", sub)
print(f"deleted name rows: {dn.num_rows:,} loaded ({time.time()-t0:.0f}s)", flush=True)
t0 = time.time()
hist = unnamed.join(dn, keys="node_id", join_type="inner")
hist_ids = pc.unique(hist.column("node_id").combine_chunks())
print(f"HISTORICAL_NAME  : {len(hist_ids):>12,} unnamed MIDs recovered "
      f"({len(hist_ids)/N*100:.3f}%)  [{hist.num_rows:,} candidate rows]  ({time.time()-t0:.0f}s)",
      flush=True)
del dn, hist

# ---- tier 1b: current keys ----------------------------------------------------------------------
pf = pq.ParquetFile(f"{V3}/canonical/metadata/key.parquet")
RG = pf.num_row_groups
t0 = time.time(); keyhits = []
STEP = 25
for i in range(0, RG, STEP):
    kt = pf.read_row_groups(list(range(i, min(i + STEP, RG))), columns=["subject", "key"])
    kt = kt.rename_columns(["node_id", "key"])
    j = unnamed.join(kt, keys="node_id", join_type="inner")
    if j.num_rows:
        keyhits.append(j.combine_chunks())
    print(f"   key chunk {i//STEP+1}/{(RG+STEP-1)//STEP}  hits so far="
          f"{sum(h.num_rows for h in keyhits):,}  ({time.time()-t0:.0f}s)", flush=True)
keytab = pa.concat_tables(keyhits) if keyhits else None
key_ids = pc.unique(keytab.column("node_id").combine_chunks()) if keytab is not None else pa.array([])
print(f"FREEBASE_KEY_EXACT: {len(key_ids):>12,} unnamed MIDs "
      f"({len(key_ids)/N*100:.3f}%)  [{keytab.num_rows if keytab is not None else 0:,} key rows]")

# ---- union / remainder ---------------------------------------------------------------------------
both = pc.unique(pa.concat_arrays([hist_ids.cast(pa.string()), key_ids.cast(pa.string())]))
print(f"\nunion of authoritative tiers : {len(both):,}  ({len(both)/N*100:.3f}%)")
print(f"remainder -> STRUCTURAL      : {N-len(both):,}  ({(N-len(both))/N*100:.3f}%)")

json.dump({
    "schema": "RESOLUTION_TIER_MEASURE/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "unnamed_entity_mids": N,
    "HISTORICAL_NAME_recovered": len(hist_ids),
    "FREEBASE_KEY_EXACT_recovered": int(len(key_ids)),
    "union_authoritative": len(both),
    "remainder_needing_structural": N - len(both),
    "pct_authoritative": round(len(both) / N * 100, 4),
}, open(f"{V3}/V3_RESOLUTION_TIER_MEASURE.json", "w", encoding="utf-8"), indent=1)
if keytab is not None:
    pq.write_table(keytab, f"{V3}/_acquisition/unnamed_keys.parquet", compression="zstd")
pq.write_table(pa.table({"node_id": hist_ids}), f"{V3}/_acquisition/unnamed_hist_ids.parquet",
               compression="zstd")
print("\nwrote V3_RESOLUTION_TIER_MEASURE.json")
