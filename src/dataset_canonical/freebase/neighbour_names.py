# -*- coding: utf-8 -*-
"""Names of every non-floor node adjacent to a floor node (the evidence the floor rules render).

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/neighbour_names.py

Reads scratchpad/fb4/evidence/floor_edges.parquet (every edge incident to a floor node) and the
composed names_v4 shards; writes scratchpad/fb4/neighbour_names.parquet
    node_uid int64 (sorted), name, name_kind int8, kind int8
for every neighbour that is not itself a floor node.  Every neighbour must be found (the edge
endpoints are all in the universe) -- asserted.
"""
import glob
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
OUT = "data/final_canonical/freebase"
SCR = "scratchpad/fb4"
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]
KCODE = {k: i for i, k in enumerate(KINDS)}
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


def member(sorted_arr, q):
    p = np.searchsorted(sorted_arr, q)
    p2 = np.minimum(p, len(sorted_arr) - 1)
    return sorted_arr[p2] == q


floor = np.load(SCR + "/floor_uids.npy")
pf = pq.ParquetFile(SCR + "/evidence/floor_edges.parquet")
parts = []
for rg in range(pf.num_row_groups):
    o = pf.read_row_group(rg, columns=["other_uid"])["other_uid"].to_numpy()
    parts.append(np.unique(o))
nb = np.unique(np.concatenate(parts))
nb = nb[~member(floor, nb)]
log("distinct non-floor neighbours", len(nb))

shards = sorted(os.path.basename(f)[:-8] for f in glob.glob(SCR + "/names_v4/*.parquet"))
assert len(shards) == 262, len(shards)
found = np.zeros(len(nb), dtype=bool)
outs = []
for si, s in enumerate(shards):
    t = pq.read_table(SCR + "/names_v4/%s.parquet" % s, columns=["node_uid", "name", "name_kind"])
    u = t["node_uid"].to_numpy()
    m = member(nb, u)
    if m.any():
        idx = np.flatnonzero(m)
        sub = t.take(pa.array(idx))
        k = pq.read_table(FB + "/canonical/nodes/%s.parquet" % s, columns=["kind"]).take(pa.array(idx))["kind"]
        kc = pa.array(np.fromiter((KCODE[x] for x in k.to_pylist()), dtype=np.int8, count=len(idx)))
        sub = sub.append_column("kind", kc)
        outs.append(sub)
        found[np.searchsorted(nb, u[m])] = True
    if si % 40 == 0:
        log("shard", si, s, "found", int(found.sum()))
assert found.all(), "neighbours not found: %d" % int((~found).sum())
t = pa.concat_tables(outs).sort_by("node_uid")
assert t.num_rows == len(nb) and np.array_equal(t["node_uid"].to_numpy(), nb)
pq.write_table(t, SCR + "/neighbour_names.parquet", compression="zstd")
nk = t["name_kind"].to_numpy()
rec = {"neighbours": len(nb), "by_name_kind": {int(k): int(c) for k, c in zip(*np.unique(nk, return_counts=True))},
       "by_kind": {KINDS[int(k)]: int(c) for k, c in zip(*np.unique(t["kind"].to_numpy(), return_counts=True))},
       "seconds": round(time.time() - t0, 1)}
json.dump(rec, open(SCR + "/NEIGHBOUR_NAMES.json", "w"), indent=1)
log("done", json.dumps(rec))
