"""STAGE 1b -- correcting my own probe, and settling whether 562 is an under-count at all.

Stage 1 rejected all 239 key-bearing MIDs as "not type-shaped" because it demanded a two-segment
domain.type path. That was wrong: /user/<who>/default_domain/<x> and /base/<project>/<x> are
perfectly valid Freebase type paths, they are just deeper. So the keys DO give human-readable
paths, from our own source of record.

That makes the real question answerable exactly rather than by reading: if we now know the paths,
we can count how many nodes in the graph are instances of them. That converts "bounded under-count
of unknown size" into a number.
"""
import json, time
from collections import Counter
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa

V3 = "data/final_canonical/freebase_v3"
a = json.load(open(f"{V3}/V3_MEDIATOR_RESOLUTION_AUDIT.json", encoding="utf-8"))
frz = json.load(open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
HELD = set(frz["DECLARATIONS"]["MEDIATOR_TYPE_PATHS"])

def key_to_path(k):
    return k.strip("/").replace("/", ".")

mid2paths, allpaths = {}, set()
for mid, keys in a["NON_TYPE_KEYS"].items():
    ps = [key_to_path(k) for k in keys]
    mid2paths[mid] = ps
    allpaths.update(ps)
print(f"MIDs with a recoverable path from key : {len(mid2paths)}")
print(f"distinct candidate type paths         : {len(allpaths)}")
print(f"of those already in MEDIATOR_TYPE_PATHS: {len(allpaths & HELD)}")

ns = Counter(p.split(".")[0] for p in allpaths)
print("namespace of recovered paths          :", dict(ns.most_common(8)))

# --- do any of these paths actually have instances? ---------------------------------------------
tgt = pa.array(sorted(allpaths))
pf = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
hits = Counter(); t0 = time.time()
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["type"])
    m = pc.is_in(t.column("type"), value_set=tgt)
    if pc.sum(m).as_py():
        for v in t.filter(m).column("type").to_pylist():
            hits[v] += 1
print(f"\nscanned 254,946,431 type rows in {time.time()-t0:.0f}s")
print(f"candidate paths WITH >=1 instance     : {len(hits)}")
print(f"total instance rows they account for  : {sum(hits.values()):,}")
if hits:
    for p, n in hits.most_common(10):
        print(f"    {n:>8,}  {p}")

out = {
    "schema": "MEDIATOR_RESOLUTION_STAGE1B/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "CORRECTION": "stage 1 reported 0 paths recovered from keys. That was a bug in this audit, not "
                  "a property of the data: it required a two-segment domain.type shape and these "
                  "are deeper user/base namespace paths. The keys were always there.",
    "mids_with_path_from_key": len(mid2paths),
    "distinct_candidate_paths": len(allpaths),
    "paths_with_instances": len(hits),
    "instance_rows": int(sum(hits.values())),
    "namespaces": dict(ns),
    "MID_TO_PATH_FROM_KEY": mid2paths,
    "PATHS_WITH_INSTANCES": dict(hits),
}
json.dump(out, open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE1B.json", "w", encoding="utf-8"), indent=1)
print("\nwrote V3_MEDIATOR_RESOLUTION_STAGE1B.json")
