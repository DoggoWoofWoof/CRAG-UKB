"""TIER 3 SIZING -- does the Freebase/Wikidata mapping actually reach the nodes that need names?

The intuition to distrust: fb2w is 2.1M links, and an entity notable enough to have a Wikidata item
is usually an entity that already carries a type.object.name in Freebase. If that holds, fb2w
overlaps almost entirely with the ALREADY-named population and buys us close to nothing on the
19.4M unnamed ENTITY_MIDs. Measure it before building anything on top of it.
"""
import gzip, glob, json, time
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
t0 = time.time()
mids, qids = [], 0
with gzip.open(f"{V3}/_acquisition/raw/fb2w.nt.gz", "rt", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        s = line.split("\t", 1)[0]
        i = s.find("/ns/")
        if i < 0:
            continue
        mids.append(s[i+4:].rstrip(">"))
        qids += 1
print(f"fb2w links parsed: {qids:,} ({time.time()-t0:.0f}s)", flush=True)
mset = pa.array(sorted(set(mids)))
print(f"distinct freebase MIDs in fb2w: {len(mset):,}", flush=True)
del mids

named_hit = unnamed_hit = absent = 0
unnamed_total = named_total = 0
t0 = time.time()
for sh in sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet")):
    pf = pq.ParquetFile(sh)
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["node_id", "kind", "display_text"])
        m = pc.equal(t.column("kind"), "ENTITY_MID")
        if not pc.sum(m).as_py():
            continue
        t = t.filter(m)
        isnull = pc.is_null(t.column("display_text"))
        unnamed_total += pc.sum(isnull).as_py()
        named_total += len(t) - pc.sum(isnull).as_py()
        inmap = pc.is_in(t.column("node_id"), value_set=mset)
        unnamed_hit += pc.sum(pc.and_(inmap, isnull)).as_py()
        named_hit += pc.sum(pc.and_(inmap, pc.invert(isnull))).as_py()
print(f"scanned node table in {time.time()-t0:.0f}s\n", flush=True)

tot_hit = named_hit + unnamed_hit
print(f"ENTITY_MID with a name        : {named_total:>12,}")
print(f"ENTITY_MID without a name     : {unnamed_total:>12,}")
print(f"fb2w MIDs found in our graph  : {tot_hit:>12,}  of {len(mset):,}")
print(f"  -> land on ALREADY-named    : {named_hit:>12,}")
print(f"  -> land on UNNAMED          : {unnamed_hit:>12,}   <-- tier 3's real reach")
if unnamed_total:
    print(f"     that is {unnamed_hit/unnamed_total*100:.4f}% of the unnamed population")

json.dump({
    "schema": "TIER3_FB2W_COVERAGE/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "fb2w_links": qids,
    "fb2w_distinct_mids": len(mset),
    "entity_mid_named": named_total,
    "entity_mid_unnamed": unnamed_total,
    "fb2w_mids_present_in_graph": tot_hit,
    "fb2w_lands_on_already_named": named_hit,
    "FB2W_LANDS_ON_UNNAMED": unnamed_hit,
    "pct_of_unnamed_reached": round(unnamed_hit / unnamed_total * 100, 6) if unnamed_total else None,
    "CAVEAT": "a Wikidata Q-id is an identifier, not a label. Even where fb2w reaches an unnamed "
              "node it yields a Q-number; turning that into human-readable text needs a second "
              "source (a Wikidata label dump or API), which is a separate acquisition.",
}, open(f"{V3}/V3_TIER3_FB2W_COVERAGE.json", "w", encoding="utf-8"), indent=1)
print("\nwrote V3_TIER3_FB2W_COVERAGE.json")
