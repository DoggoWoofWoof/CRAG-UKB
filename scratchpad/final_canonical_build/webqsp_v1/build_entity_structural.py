"""For every ENTITY_MID with no Freebase name, derive a type-based structural description.

Freebase never named these -- that is a property of the source, not a gap in the build -- so the
honest resolution is to say what the node IS, from its declared types, rather than to keep hunting
for a name that was never assigned. Output feeds the STRUCTURAL_INFERRED tier of the overlay.

Two joins over big tables, both done with the chunked pa.Table.join pattern rather than pc.is_in,
because is_in rebuilds its hash table once per row group and never finishes on a value set this
size.
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
STEP = 25                       # row groups joined at a time
t0 = time.time()

# ---- 1. the unnamed ENTITY_MID population -------------------------------------------------
ids = []
for fp in sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet")):
    t = pq.read_table(fp, columns=["node_id", "kind", "display_text"])
    # fill_null before comparing: pc.or_ is not Kleene, so or_(true, null) is null and a filter
    # built that way silently drops every unnamed row -- which is exactly what it did.
    m = pc.and_(pc.equal(t["kind"], "ENTITY_MID"),
                pc.equal(pc.fill_null(t["display_text"], ""), ""))
    sel = t.filter(m)
    if sel.num_rows:
        ids.append(sel["node_id"])
unnamed = pa.chunked_array(ids).combine_chunks()
del ids
print(f"unnamed ENTITY_MID: {len(unnamed):,}  ({time.time()-t0:.0f}s)", flush=True)
probe = pa.table({"subject": unnamed})

# ---- 2. their declared types ---------------------------------------------------------------
pf = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
hits = []
for lo in range(0, pf.num_row_groups, STEP):
    rgs = list(range(lo, min(lo + STEP, pf.num_row_groups)))
    t = pf.read_row_groups(rgs, columns=["subject", "type"])
    j = t.join(probe, keys="subject", join_type="inner")
    if j.num_rows:
        hits.append(j.combine_chunks())
    print(f"  type rg {rgs[0]}-{rgs[-1]}  matched {sum(h.num_rows for h in hits):,} "
          f"({time.time()-t0:.0f}s)", flush=True)
typed = pa.concat_tables(hits).combine_chunks() if hits else pa.table({"subject": [], "type": []})
del hits
print(f"type rows for unnamed entities: {typed.num_rows:,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- 3. one deterministic type per entity ---------------------------------------------------
# common.topic is carried by nearly everything and says nothing, so it is only used when it is all
# there is. Among the rest the lexicographically smallest wins, matching build_cvt_names.
import numpy as np
GENERIC = pa.array(["common.topic", "base.type_ontology.topic"])
rank = pa.table({"subject": typed["subject"], "type": typed["type"],
                 "gen": pc.cast(pc.is_in(typed["type"], value_set=GENERIC), pa.int8())})
rank = rank.sort_by([("subject", "ascending"), ("gen", "ascending"), ("type", "ascending")])
sa = rank["subject"].combine_chunks().to_numpy(zero_copy_only=False)
keep = np.empty(len(sa), dtype=bool)
keep[0] = True
keep[1:] = sa[1:] != sa[:-1]
best = rank.filter(pa.array(keep))
print(f"entities with a type: {best.num_rows:,}  ({time.time()-t0:.0f}s)", flush=True)

# ---- 4. type path -> Freebase name ----------------------------------------------------------
paths = pc.unique(best["type"]).to_pylist()
print(f"distinct types used: {len(paths):,}")
want = pa.table({"subject": pa.array(sorted(paths), pa.string())})
pfn = pq.ParquetFile(f"{V3}/canonical/metadata/name.parquet")
nm = {}
for lo in range(0, pfn.num_row_groups, STEP):
    rgs = list(range(lo, min(lo + STEP, pfn.num_row_groups)))
    t = pfn.read_row_groups(rgs, columns=["subject", "lexical", "lang"])
    t = t.filter(pc.equal(t["lang"], "en"))
    j = t.join(want, keys="subject", join_type="inner")
    for s, l in zip(j["subject"].to_pylist(), j["lexical"].to_pylist()):
        if l and (s not in nm or l < nm[s]):
            nm[s] = l
print(f"types with an en name: {len(nm):,}/{len(paths):,}  ({time.time()-t0:.0f}s)", flush=True)

def pretty(p):
    return p.rsplit(".", 1)[-1].replace("_", " ").strip() or p

label = {p: (nm.get(p) or pretty(p)) for p in paths}
disp = pa.array([label[t] for t in best["type"].to_pylist()], pa.string())
out = pa.table({"node_id": best["subject"], "type_path": best["type"], "type_label": disp})
os.makedirs(f"{V3}/_acquisition", exist_ok=True)
pq.write_table(out, f"{V3}/_acquisition/entity_type_names.parquet", compression="zstd")

top = collections.Counter(out["type_label"].to_pylist()).most_common(20)
rec = {"schema": "ENTITY_STRUCTURAL_TYPES/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "unnamed_entity_mids": len(unnamed),
       "with_at_least_one_type": best.num_rows,
       "without_any_type": len(unnamed) - best.num_rows,
       "distinct_types_used": len(paths),
       "types_with_freebase_en_name": len(nm),
       "type_selection": ("common.topic and base.type_ontology.topic are used only when nothing "
                          "else is declared; among the remainder the lexicographically smallest "
                          "type wins, so the choice is deterministic and reproducible."),
       "top_type_labels": [{"label": l, "entities": c} for l, c in top],
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_ENTITY_STRUCTURAL_TYPES.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nunnamed={len(unnamed):,}  typed={best.num_rows:,}  untyped={len(unnamed)-best.num_rows:,}")
for l, c in top[:15]:
    print(f"  {c:>10,}  {l}")
print(f"{time.time()-t0:.0f}s")
