# -*- coding: utf-8 -*-
"""Step 5: ID bridge from the served webqsp dataset into Freebase positions.

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/build_bridge.py

webqsp/nodes.jsonl line i (position i) carries `source_mid` (a Freebase MID, or null for nodes
that were identified by text).  bridge/webqsp_positions.npy int32[n_webqsp] holds the Freebase
position of that MID, -1 where the node has no MID or the MID is not in the canonical universe.
Also writes bridge/schema_index.parquet: node_id -> position for every SCHEMA_* node (60K), so a
consumer can resolve schema ids without reproducing CPython's hash.
"""
import io
import json
import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = "data/final_canonical/freebase"
SCR = "scratchpad/fb4"
N = 301977131
t0 = time.time()
os.makedirs(OUT + "/bridge", exist_ok=True)
uids = np.load(OUT + "/nodes/node_uid.npy", mmap_mode="r")
kind = np.load(OUT + "/nodes/kind.npy")


def positions_of(strings):
    h = np.fromiter((hash(s.encode("utf-8")) for s in strings), dtype=np.int64, count=len(strings))
    p = np.searchsorted(uids, h)
    p[p >= N] = 0
    ok = uids[p] == h
    return np.where(ok, p, -1).astype(np.int32), ok


mids = []
n = 0
with open("data/final_canonical/webqsp/nodes.jsonl", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        r = json.loads(line)
        assert r["position"] == i
        mids.append(r.get("source_mid"))
        n += 1
has = np.array([m is not None for m in mids])
pos = np.full(n, -1, dtype=np.int32)
if has.any():
    p, ok = positions_of([m for m in mids if m is not None])
    pos[np.flatnonzero(has)] = p
    kinds_hit = {int(k): int(c) for k, c in zip(*np.unique(kind[p[ok]], return_counts=True))}
else:
    ok = np.zeros(0, dtype=bool); kinds_hit = {}
np.save(OUT + "/bridge/webqsp_positions.npy", pos)
sample_mids = [m for m in mids if m is not None][:3]

sn = pq.read_table(SCR + "/schema_nodes.parquet", columns=["node_uid", "node_id", "kind"])
sp = np.searchsorted(uids, sn["node_uid"].to_numpy())
assert np.array_equal(uids[sp], sn["node_uid"].to_numpy())
pq.write_table(pa.table({"node_id": sn["node_id"], "kind": sn["kind"], "position": pa.array(sp.astype(np.int32))}).sort_by("node_id"),
               OUT + "/bridge/schema_index.parquet", compression="zstd")
rec = {"RECORD": "FREEBASE_BRIDGE", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "webqsp_nodes": n, "webqsp_nodes_with_source_mid": int(has.sum()), "resolved_to_freebase_position": int(ok.sum()),
       "unresolved_mids": int((~ok).sum()), "resolved_by_kind_code": kinds_hit, "sample_mids": sample_mids,
       "schema_index_rows": sn.num_rows, "seconds": round(time.time() - t0, 1)}
json.dump(rec, open(SCR + "/BRIDGE.json", "w"), indent=1)
print(json.dumps(rec))
