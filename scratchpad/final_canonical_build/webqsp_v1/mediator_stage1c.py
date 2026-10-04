"""STAGE 1c -- the complement test, which settles the remaining 323 keyless MIDs without needing
their paths at all.

The worry behind "bounded under-count" is: some unresolved mediator MID X has type path P, P has
instances in the graph, and because we never learned X<->P we failed to mark those instances as
CVT_MEDIATOR. Rather than chase P for each X, invert it: enumerate every type path that ACTUALLY
HAS INSTANCES and ask whether each one is a path we can already account for. If every instantiated
path is accounted for, then no unresolved MID can be hiding instances behind it -- for all 562,
keyless ones included.
"""
import json, time
from collections import Counter
import pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
frz = json.load(open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
D = frz["DECLARATIONS"]
KNOWN_TYPE_PATHS = set(D["MID_TO_TYPE_PATH"].values())
MEDIATOR_PATHS   = set(D["MEDIATOR_TYPE_PATHS"])
print(f"type paths we can name a MID for : {len(KNOWN_TYPE_PATHS):,}")
print(f"of which declared mediator       : {len(MEDIATOR_PATHS):,}")

pf = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
# value_counts stays in C++: 254.9M rows never become python strings, only the ~16k distinct
# paths do. The naive to_pylist() loop here was minutes of pure allocator churn.
inst = Counter(); t0 = time.time()
for rg in range(pf.num_row_groups):
    vc = pc.value_counts(pf.read_row_group(rg, columns=["type"]).column("type").combine_chunks())
    for k, c in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
        inst[k] += c
print(f"\nscanned {sum(inst.values()):,} type rows in {time.time()-t0:.0f}s")
print(f"distinct type paths WITH instances: {len(inst):,}")

unaccounted = {p: n for p, n in inst.items() if p not in KNOWN_TYPE_PATHS}
acc_rows = sum(n for p, n in inst.items() if p in KNOWN_TYPE_PATHS)
print(f"\npaths we can name a MID for       : {len(inst) - len(unaccounted):,}  ({acc_rows:,} rows)")
print(f"paths we CANNOT name a MID for    : {len(unaccounted):,}  ({sum(unaccounted.values()):,} rows)")
if unaccounted:
    print("\n  largest unaccounted paths:")
    for p, n in Counter(unaccounted).most_common(15):
        print(f"    {n:>12,}  {p}")
    ns = Counter(p.split(".")[0] for p in unaccounted)
    print("\n  their namespaces:", dict(ns.most_common(10)))

json.dump({
    "schema": "MEDIATOR_RESOLUTION_STAGE1C/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "QUESTION": "can any of the 562 unresolved mediator MIDs be hiding instantiated nodes that we "
                "failed to classify as CVT_MEDIATOR?",
    "METHOD": "enumerate every type path that has at least one instance and check whether it maps "
              "back to a known type MID. Unaccounted instantiated paths are the only place a "
              "missed mediator could hide.",
    "distinct_instantiated_paths": len(inst),
    "paths_mapped_to_a_known_mid": len(inst) - len(unaccounted),
    "paths_not_mapped": len(unaccounted),
    "rows_under_unmapped_paths": int(sum(unaccounted.values())),
    "UNACCOUNTED": dict(Counter(unaccounted).most_common(500)),
}, open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE1C.json", "w", encoding="utf-8"), indent=1)
print("\nwrote V3_MEDIATOR_RESOLUTION_STAGE1C.json")
