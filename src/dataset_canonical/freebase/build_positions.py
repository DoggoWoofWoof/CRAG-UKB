# -*- coding: utf-8 -*-
"""Step 1 of the normalised Freebase tree: the position space.

position i  ==  rank of the node's 64-bit uid in ascending order  (canonical/_node_uids_sorted.npy)

So `nodes/node_uid.npy[i]` is strictly increasing and uid -> position is one binary search; no
hash map, no join.  This script writes the position-indexed kind array and, for every canonical
node shard, the int32 position of each of its rows (the alignment every later step uses).

Outputs
    data/final_canonical/freebase/nodes/node_uid.npy      int64[N]  strictly increasing
    data/final_canonical/freebase/nodes/kind.npy          int8[N]   KIND code (see KINDS)
    scratchpad/fb4/pos/<canonical shard>.npy              int32[rows of that shard] -> position
    scratchpad/fb4/POSITIONS.json                         counts + checks
Reads only the frozen canonical layer.
"""
import glob
import io
import json
import os
import shutil
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
OUT = "data/final_canonical/freebase"
SCR = "scratchpad/fb4"
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]
KCODE = {k: i for i, k in enumerate(KINDS)}
N_EXPECTED = 301977131
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


os.makedirs(OUT + "/nodes", exist_ok=True)
os.makedirs(SCR + "/pos", exist_ok=True)

src = FB + "/canonical/_node_uids_sorted.npy"
uids = np.load(src, mmap_mode="r")
assert uids.dtype == np.int64 and uids.ndim == 1
N = len(uids)
assert N == N_EXPECTED, N
# strictly increasing, checked in chunks so the 2.4 GB array is never duplicated
prev = None
for s in range(0, N, 20_000_000):
    c = np.asarray(uids[s:s + 20_000_000])
    assert np.all(np.diff(c) > 0), "not strictly increasing inside chunk at %d" % s
    if prev is not None:
        assert c[0] > prev
    prev = int(c[-1])
log("sorted uid array verified strictly increasing, N =", N)

dst = OUT + "/nodes/node_uid.npy"
if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
    shutil.copyfile(src, dst)
    log("copied node_uid.npy")
chk = np.load(dst, mmap_mode="r")
assert len(chk) == N and chk[0] == uids[0] and chk[-1] == uids[-1]

kind = np.full(N, -1, dtype=np.int8)
seen = np.zeros(N, dtype=bool)
by_kind = {k: 0 for k in KINDS}
shards = sorted(glob.glob(FB + "/canonical/nodes/*.parquet"))
assert len(shards) == 262, len(shards)
rows_total = 0
for i, f in enumerate(shards):
    name = os.path.basename(f)[:-8]
    pf = pq.ParquetFile(f)
    pos_parts = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["node_uid", "kind"])
        u = t["node_uid"].to_numpy()
        p = np.searchsorted(uids, u)
        assert p.max() < N
        assert np.array_equal(np.asarray(uids[p]), u), "uid missing from sorted array in " + name
        k = t["kind"].to_pylist() if t["kind"].type == "string" else t["kind"].to_pylist()
        kc = np.fromiter((KCODE[x] for x in k), dtype=np.int8, count=len(k))
        assert not seen[p].any(), "position assigned twice in " + name
        seen[p] = True
        kind[p] = kc
        pos_parts.append(p.astype(np.int32))
        for kk, cnt in zip(*np.unique(kc, return_counts=True)):
            by_kind[KINDS[kk]] += int(cnt)
        rows_total += len(u)
    np.save(SCR + "/pos/%s.npy" % name, np.concatenate(pos_parts))
    if i % 25 == 0:
        log("shard", i, name, "rows so far", rows_total)

assert rows_total == N, (rows_total, N)
assert seen.all(), "some position never assigned"
assert (kind >= 0).all()
np.save(OUT + "/nodes/kind.npy", kind)
rec = {
    "RECORD": "FREEBASE_POSITIONS",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "position_rule": "position i == rank of node_uid in ascending order; nodes/node_uid.npy is strictly increasing",
    "nodes": N,
    "KINDS": KINDS,
    "by_kind": by_kind,
    "CHECKS": {"every_canonical_row_found": True, "every_position_assigned_once": True,
               "rows_total_equals_N": rows_total == N,
               "by_kind_matches_manifest": by_kind == {"LITERAL": 105351616, "ENTITY_MID": 69839243, "EXTERNAL_URI": 74454117,
                                                       "CVT_MEDIATOR": 52272634, "SCHEMA_TYPE": 2239, "SCHEMA_PROPERTY": 20063,
                                                       "SCHEMA_OTHER": 37219}},
    "outputs": {"node_uid": OUT + "/nodes/node_uid.npy", "kind": OUT + "/nodes/kind.npy", "shard_positions": SCR + "/pos/<shard>.npy"},
    "seconds": round(time.time() - t0, 1),
}
json.dump(rec, open(SCR + "/POSITIONS.json", "w"), indent=1)
log("done", json.dumps(rec["CHECKS"]))
