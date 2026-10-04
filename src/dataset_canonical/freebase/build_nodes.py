# -*- coding: utf-8 -*-
"""Step 3 of the normalised Freebase tree: the node table in POSITION order, names included.

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/build_nodes.py

For every canonical node shard (row-aligned with overlay_v1 / semantic_v1 / scratchpad names_v4 /
scratchpad pos) the rows are scattered by position into 76 output shards of 4,000,000 positions,
then each output shard is sorted and checked: row r of nodes/shard_XXXXX.parquet is position
4,000,000*XXXXX + r, its node_uid equals nodes/node_uid.npy[position] and its kind equals
nodes/kind.npy[position].

Columns: position int32, node_uid int64, node_id string, kind int8, name string, name_kind int8,
name_source, name_rule, is_original_name bool, nameless_grade, recovery_class, floor_reason int8,
lexical_form, datatype, language (literal identity, null otherwise), semantic_text (K=6 role text
from semantic_v1, null where that layer holds none).
Also writes nodes/name_kind.npy int8[N] and scratchpad/fb4/NODES_BUILD.json.
"""
import collections
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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
OUT = "data/final_canonical/freebase"
SCR = "scratchpad/fb4"
PIECES = SCR + "/node_pieces"
N = 301977131
SHARD = 4_000_000
NSHARD = -(-N // SHARD)  # 76
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]
KCODE = {k: i for i, k in enumerate(KINDS)}
NAME_KINDS = ["ORIGINAL", "RECOVERED_ORIGINAL", "RECOVERED_EXTERNAL", "URI_SELF", "KEY_SEMANTIC", "INFERRED", "GENERATED_FLOOR"]
SCHEMA = pa.schema([
    ("position", pa.int32()), ("node_uid", pa.int64()), ("node_id", pa.string()), ("kind", pa.int8()),
    ("name", pa.string()), ("name_kind", pa.int8()), ("name_source", pa.string()), ("name_rule", pa.string()),
    ("is_original_name", pa.bool_()), ("nameless_grade", pa.string()), ("recovery_class", pa.string()), ("floor_reason", pa.int8()),
    ("lexical_form", pa.string()), ("datatype", pa.string()), ("language", pa.string()), ("semantic_text", pa.string()),
])
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


def undict(col):
    return col.cast(col.type.value_type) if pa.types.is_dictionary(col.type) else col


if os.path.exists(PIECES):
    shutil.rmtree(PIECES)
os.makedirs(PIECES)
os.makedirs(OUT + "/nodes", exist_ok=True)
node_uid = np.load(OUT + "/nodes/node_uid.npy", mmap_mode="r")
kind_arr = np.load(OUT + "/nodes/kind.npy")
shards = sorted(os.path.basename(f)[:-8] for f in glob.glob(FB + "/canonical/nodes/*.parquet"))
assert len(shards) == 262

# ---- pass 1: scatter -------------------------------------------------------------------------------
writers = {}
n_scattered = 0
SL = 1_000_000
for si, s in enumerate(shards):
    cn = pq.read_table(FB + "/canonical/nodes/%s.parquet" % s, columns=["node_uid", "node_id", "kind", "lexical_form", "datatype", "language"])
    se = pq.read_table(FB + "/semantic_v1/%s.parquet" % s)
    nv = pq.read_table(SCR + "/names_v4/%s.parquet" % s)
    pos = np.load(SCR + "/pos/%s.npy" % s)
    n_all = cn.num_rows
    assert se.num_rows == nv.num_rows == len(pos) == n_all, s
    u = cn["node_uid"].to_numpy()
    assert np.array_equal(u, se["node_uid"].to_numpy()) and np.array_equal(u, nv["node_uid"].to_numpy()), "misaligned " + s
    assert np.array_equal(np.asarray(node_uid[pos]), u), "positions do not resolve to these uids " + s
    kcode = np.fromiter((KCODE[x] for x in cn["kind"].to_pylist()), dtype=np.int8, count=n_all)
    assert np.array_equal(kind_arr[pos], kcode), "kind mismatch " + s
    for lo in range(0, n_all, SL):
        hi = min(lo + SL, n_all)
        n = hi - lo
        p = pos[lo:hi]
        tbl = pa.table({
            "position": pa.array(p), "node_uid": pa.array(u[lo:hi]), "node_id": cn["node_id"].slice(lo, n).combine_chunks(),
            "kind": pa.array(kcode[lo:hi]),
            "name": nv["name"].slice(lo, n).combine_chunks(), "name_kind": nv["name_kind"].slice(lo, n).combine_chunks(),
            "name_source": undict(nv["name_source"].slice(lo, n).combine_chunks()), "name_rule": undict(nv["name_rule"].slice(lo, n).combine_chunks()),
            "is_original_name": nv["is_original_name"].slice(lo, n).combine_chunks(),
            "nameless_grade": undict(nv["nameless_grade"].slice(lo, n).combine_chunks()), "recovery_class": undict(nv["recovery_class"].slice(lo, n).combine_chunks()),
            "floor_reason": nv["floor_reason"].slice(lo, n).combine_chunks(),
            "lexical_form": cn["lexical_form"].slice(lo, n).combine_chunks(), "datatype": cn["datatype"].slice(lo, n).combine_chunks(),
            "language": cn["language"].slice(lo, n).combine_chunks(), "semantic_text": se["semantic_text"].slice(lo, n).combine_chunks(),
        }, schema=SCHEMA)
        b = p // SHARD
        order = np.argsort(b, kind="stable")
        bs = b[order]
        cuts = np.flatnonzero(np.diff(bs)) + 1
        starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(bs)]])
        for a, e in zip(starts, ends):
            k = int(bs[a])
            if k not in writers:
                writers[k] = pq.ParquetWriter(PIECES + "/piece_%05d.parquet" % k, SCHEMA, compression="zstd", use_dictionary=False)
            writers[k].write_table(tbl.take(pa.array(order[a:e])))
        n_scattered += n
    del cn, se, nv
    if si % 20 == 0:
        log("scatter", si, s, "rows", n_scattered)
for w in writers.values():
    w.close()
assert n_scattered == N, n_scattered
log("scatter done", n_scattered)

# ---- pass 2: sort each output shard, verify, write ---------------------------------------------------
name_kind_all = np.full(N, -1, dtype=np.int8)
stats = collections.Counter()
shard_rows = []
for k in range(NSHARD):
    lo, hi = k * SHARD, min((k + 1) * SHARD, N)
    t = pq.read_table(PIECES + "/piece_%05d.parquet" % k)
    assert t.num_rows == hi - lo, (k, t.num_rows, hi - lo)
    t = t.sort_by("position")
    p = t["position"].to_numpy()
    assert np.array_equal(p, np.arange(lo, hi, dtype=np.int32)), "positions not contiguous in shard %d" % k
    assert np.array_equal(t["node_uid"].to_numpy(), np.asarray(node_uid[lo:hi]))
    assert np.array_equal(t["kind"].to_numpy(), kind_arr[lo:hi])
    nk = t["name_kind"].to_numpy()
    name_kind_all[lo:hi] = nk
    for c, n in zip(*np.unique(nk, return_counts=True)):
        stats[NAME_KINDS[c]] += int(n)
    empty = pc.sum(pc.or_kleene(pc.is_null(t["name"]), pc.equal(t["name"], ""))).as_py() or 0
    assert empty == 0, ("empty names in shard", k, empty)
    for col in ("name_source", "name_rule", "nameless_grade", "recovery_class"):
        t = t.set_column(t.schema.get_field_index(col), col, pc.dictionary_encode(t[col]))
    pq.write_table(t, OUT + "/nodes/shard_%05d.parquet" % k, compression="zstd", row_group_size=500_000)
    shard_rows.append(t.num_rows)
    os.remove(PIECES + "/piece_%05d.parquet" % k)
    if k % 10 == 0:
        log("shard", k, "rows", t.num_rows)
shutil.rmtree(PIECES)
assert (name_kind_all >= 0).all()
np.save(OUT + "/nodes/name_kind.npy", name_kind_all)
rec = {"RECORD": "FREEBASE_NODES_BUILD", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "nodes": N, "shards": NSHARD, "rows_per_shard": SHARD, "shard_rows": shard_rows,
       "by_name_kind": dict(stats), "columns": [f.name for f in SCHEMA],
       "CHECKS": {"positions_contiguous": True, "node_uid_matches_index": True, "kind_matches_index": True, "empty_names": 0},
       "seconds": round(time.time() - t0, 1)}
json.dump(rec, open(SCR + "/NODES_BUILD.json", "w"), indent=1)
log("done", json.dumps(stats))
