# -*- coding: utf-8 -*-
"""Gather every piece of in-graph evidence the canonical universe holds about the FLOOR nodes
(scratchpad/fb4/floor_uids.npy from floor_census.py): their node_id and kind, their declared
types, descriptions, aliases and keys (metadata tables), and every canonical edge they touch in
either direction, with the relation name.  Reads only.

Outputs (parquet, under <out>/evidence/):
    floor_ids.parquet        node_uid, node_id, kind
    floor_types.parquet      node_uid, type
    floor_desc.parquet       node_uid, lexical, lang
    floor_alias.parquet      node_uid, lexical, lang
    floor_keys.parquet       node_uid, key
    floor_edges.parquet      node_uid, rel, relation, other_uid, direction ('out' = node is src)
"""
import glob
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
OUT = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/fb4"
EV = OUT + "/evidence"
os.makedirs(EV, exist_ok=True)
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


def member(sorted_arr, q):
    p = np.searchsorted(sorted_arr, q)
    p2 = np.minimum(p, len(sorted_arr) - 1)
    return sorted_arr[p2] == q


floor = np.load(OUT + "/floor_uids.npy")
assert np.all(np.diff(floor) > 0)
log("floor uids", len(floor))

# ---- A: ids and kinds -------------------------------------------------------------------------
if not os.path.exists(EV + "/floor_ids.parquet"):
    parts = []
    for f in sorted(glob.glob(FB + "/canonical/nodes/*.parquet")):
        if os.path.basename(f).startswith("lit_"):
            continue  # census: no floor node is a literal
        t = pq.read_table(f, columns=["node_uid", "node_id", "kind"])
        m = member(floor, t["node_uid"].to_numpy())
        if m.any():
            parts.append(t.filter(pa.array(m)))
    ids = pa.concat_tables(parts)
    assert ids.num_rows == len(floor), (ids.num_rows, len(floor))
    pq.write_table(ids, EV + "/floor_ids.parquet", compression="zstd")
    log("floor_ids", ids.num_rows)
ids = pq.read_table(EV + "/floor_ids.parquet")
id_set = pa.array(ids["node_id"].to_pylist())  # value set for is_in
uid_of_id = dict(zip(ids["node_id"].to_pylist(), ids["node_uid"].to_numpy().tolist()))
log("id set", len(uid_of_id))


def filter_meta(table_name, cols, out_name):
    if os.path.exists(EV + "/" + out_name):
        log("skip", out_name)
        return
    pf = pq.ParquetFile(FB + "/canonical/metadata/%s.parquet" % table_name)
    parts = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["subject"] + cols)
        m = pc.is_in(t["subject"], value_set=id_set)
        if pc.any(m).as_py():
            parts.append(t.filter(m))
    if parts:
        t = pa.concat_tables(parts)
        u = pa.array([uid_of_id[s] for s in t["subject"].to_pylist()], pa.int64())
        t = t.drop(["subject"]).add_column(0, "node_uid", u)
    else:
        t = pa.table({"node_uid": pa.array([], pa.int64()), **{c: pa.array([], pa.string()) for c in cols}})
    pq.write_table(t, EV + "/" + out_name, compression="zstd")
    log(out_name, t.num_rows, "rows on", len(set(t["node_uid"].to_pylist())), "floor nodes")


filter_meta("type", ["type"], "floor_types.parquet")
filter_meta("description", ["lexical", "lang"], "floor_desc.parquet")
filter_meta("alias", ["lexical", "lang"], "floor_alias.parquet")
filter_meta("key", ["key"], "floor_keys.parquet")

# ---- C: incident edges ------------------------------------------------------------------------
if not os.path.exists(EV + "/floor_edges.parquet"):
    rels = pq.read_table(FB + "/canonical/relations.parquet", columns=["rel_uid", "relation"])
    rel_name = dict(zip(rels["rel_uid"].to_numpy().tolist(), rels["relation"].to_pylist()))
    outs = []
    n_seen = 0
    for i, f in enumerate(sorted(glob.glob(FB + "/canonical/edges/*.parquet"))):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=["src", "rel", "dst"])
            s, r, d = t["src"].to_numpy(), t["rel"].to_numpy(), t["dst"].to_numpy()
            n_seen += len(s)
            ms, md = member(floor, s), member(floor, d)
            if ms.any():
                outs.append(pa.table({"node_uid": s[ms], "rel": r[ms], "other_uid": d[ms],
                                      "direction": pa.array(["out"] * int(ms.sum()), pa.string())}))
            if md.any():
                outs.append(pa.table({"node_uid": d[md], "rel": r[md], "other_uid": s[md],
                                      "direction": pa.array(["in"] * int(md.sum()), pa.string())}))
        if i % 20 == 0:
            log("edge part", i, os.path.basename(f), "scanned", n_seen, "hits", sum(x.num_rows for x in outs))
    e = pa.concat_tables(outs)
    e = e.add_column(2, "relation", pa.array([rel_name[x] for x in e["rel"].to_numpy().tolist()], pa.string()))
    e = e.sort_by([("node_uid", "ascending"), ("direction", "ascending"), ("rel", "ascending")])
    pq.write_table(e, EV + "/floor_edges.parquet", compression="zstd")
    log("floor_edges", e.num_rows, "rows; edges scanned", n_seen, "CHECK", n_seen == 2062430072)
log("done")
