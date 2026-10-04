# -*- coding: utf-8 -*-
"""Step 4 of the normalised Freebase tree: the 12 metadata tables keyed by POSITION.

    PYTHONHASHSEED=0 python src/dataset_canonical/freebase/build_metadata.py

Every frozen metadata table (canonical/metadata/<t>.parquet) carries `subject` as the
namespace-stripped Freebase id.  The node uid of that id is CPython hash() of its UTF-8 bytes
under PYTHONHASHSEED=0 (the minting rule of PASS B), so subject -> uid -> position is a hash plus
one binary search in nodes/node_uid.npy.  Each table is rewritten as
    metadata/<t>.parquet      position int32 first column, rows sorted by position, other columns verbatim
                              (master_property / reverse_property also get object_position)
and the type table additionally as a CSR over type ids:
    metadata/type_indptr.npy  int64[N+1]      metadata/type_val.npy int32[rows]   (type_id, ascending within a node)
    metadata/type_dict.parquet type_id, type, type_position (position of the SCHEMA_TYPE node, -1 if none)
Row counts in == rows out for every table; unresolved subjects are counted and must be zero.
Large tables are bucketed by position range on disk (76 buckets of 4,000,000 positions) so no table
is ever held in RAM at once.
"""
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

FB = "data/final_canonical/freebase_v3/canonical/metadata"
OUT = "data/final_canonical/freebase"
SCR = "scratchpad/fb4"
TMP = SCR + "/meta_tmp"
N = 301977131
SHARD = 4_000_000
NB = -(-N // SHARD)
TABLES = ["alias", "description", "key", "label_residue", "master_property", "name", "property_schema",
          "rdf_type", "rdf_type_residue", "reverse_property", "type", "type_hints"]
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


os.makedirs(OUT + "/metadata", exist_ok=True)
uids = np.load(OUT + "/nodes/node_uid.npy", mmap_mode="r")


def positions_of(strings):
    h = np.fromiter((hash(s.encode("utf-8")) for s in strings), dtype=np.int64, count=len(strings))
    p = np.searchsorted(uids, h)
    p[p >= N] = 0
    ok = uids[p] == h
    return p.astype(np.int32), ok


class NpyAppender:
    def __init__(self, path, dtype, n):
        self.fh = open(path, "wb")
        np.lib.format.write_array_header_1_0(self.fh, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False, "shape": (n,)})
        self.dtype, self.n, self.written = np.dtype(dtype), n, 0

    def append(self, arr):
        arr = np.ascontiguousarray(arr, dtype=self.dtype); arr.tofile(self.fh); self.written += len(arr)

    def close(self):
        assert self.written == self.n, (self.written, self.n); self.fh.close()


rec = {"RECORD": "FREEBASE_METADATA_BUILD", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tables": {}}
for tname in TABLES:
    if os.path.exists(TMP):
        shutil.rmtree(TMP)
    os.makedirs(TMP)
    pf = pq.ParquetFile(FB + "/%s.parquet" % tname)
    n_in = pf.metadata.num_rows
    cols = [c for c in pf.schema_arrow.names if c != "subject"]
    miss = 0
    obj_unres = 0
    type_strings = set()
    writers = {}
    schema = None
    # ---- pass 1: resolve + scatter by position bucket ------------------------------------------
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg)
        if t.num_rows == 0:
            continue
        p, ok = positions_of(t["subject"].to_pylist())
        miss += int((~ok).sum())
        p = np.where(ok, p, -1).astype(np.int32)
        piece = t.drop(["subject"]).add_column(0, "position", pa.array(p))
        if tname in ("master_property", "reverse_property"):
            op, ook = positions_of(t["object"].to_pylist())
            piece = piece.append_column("object_position", pa.array(np.where(ook, op, -1).astype(np.int32)))
            obj_unres += int((~ook).sum())
        if tname == "type":
            type_strings.update(pc.unique(t["type"]).to_pylist())
        schema = piece.schema
        b = np.maximum(p, 0) // SHARD
        order = np.argsort(b, kind="stable")
        bs = b[order]
        cuts = np.flatnonzero(np.diff(bs)) + 1
        starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(bs)]])
        for a, e in zip(starts, ends):
            k = int(bs[a])
            if k not in writers:
                writers[k] = pq.ParquetWriter(TMP + "/b_%05d.parquet" % k, schema, compression="zstd")
            writers[k].write_table(piece.take(pa.array(order[a:e])))
        if rg % 100 == 0 and rg:
            log(tname, "rg", rg, "/", pf.num_row_groups)
    for w in writers.values():
        w.close()
    # ---- pass 2: per bucket sort, stream out ----------------------------------------------------
    out_path = OUT + "/metadata/%s.parquet" % tname
    n_out = 0
    if schema is None:  # empty table (label_residue, rdf_type_residue): keep an empty, correctly-typed file
        schema = pa.schema([("position", pa.int32())] + [(c, pf.schema_arrow.field(c).type) for c in cols])
        pq.write_table(schema.empty_table(), out_path, compression="zstd")
    else:
        ow = pq.ParquetWriter(out_path, schema, compression="zstd")
        type_ids = None
        if tname == "type":
            tnames = sorted(type_strings)
            type_ids = {s: i for i, s in enumerate(tnames)}
            indptr = np.zeros(N + 1, dtype=np.int64)
            tv = NpyAppender(OUT + "/metadata/type_val.npy", np.int32, n_in)
        for k in range(NB):
            f = TMP + "/b_%05d.parquet" % k
            if not os.path.exists(f):
                continue
            t = pq.read_table(f)
            pos = t["position"].to_numpy()
            if tname == "type":
                tid = np.fromiter((type_ids[s] for s in t["type"].to_pylist()), dtype=np.int32, count=t.num_rows)
                o = np.lexsort((tid, pos))
                t = t.take(pa.array(o))
                pos, tid = pos[o], tid[o]
                assert (pos >= 0).all()
                lo, hi = k * SHARD, min((k + 1) * SHARD, N)
                indptr[lo + 1:hi + 1] = n_out + np.cumsum(np.bincount(pos - lo, minlength=hi - lo))
                tv.append(tid)
            else:
                t = t.sort_by("position")
            for s in range(0, t.num_rows, 1_000_000):
                ow.write_table(t.slice(s, min(1_000_000, t.num_rows - s)))
            n_out += t.num_rows
            os.remove(f)
        ow.close()
        if tname == "type":
            tv.close()
    assert n_out == n_in, (tname, n_out, n_in)
    info = {"rows": n_in, "unresolved_subjects": miss, "columns": ["position"] + cols + (["object_position"] if tname in ("master_property", "reverse_property") else [])}
    if tname in ("master_property", "reverse_property"):
        info["object_unresolved"] = obj_unres
    if tname == "type":
        # indptr: buckets without rows left zeros; forward-fill so indptr is non-decreasing
        np.maximum.accumulate(indptr, out=indptr)
        assert indptr[-1] == n_in
        np.save(OUT + "/metadata/type_indptr.npy", indptr)
        tp, tok = positions_of(tnames)
        pq.write_table(pa.table({"type_id": pa.array(np.arange(len(tnames), dtype=np.int32)), "type": pa.array(tnames, pa.string()),
                                 "type_position": pa.array(np.where(tok, tp, -1).astype(np.int32))}),
                       OUT + "/metadata/type_dict.parquet", compression="zstd")
        info["distinct_types"] = len(tnames)
        info["types_without_a_schema_node"] = int((~tok).sum())
        info["csr"] = {"type_indptr": "int64[N+1]", "type_val": "int32[rows] type_id, ascending within a node",
                       "nodes_with_types": int((np.diff(indptr) > 0).sum())}
    rec["tables"][tname] = info
    log(tname, json.dumps(info))
shutil.rmtree(TMP, ignore_errors=True)
rec["CHECKS"] = {"all_rows_kept": True, "unresolved_subjects_total": sum(v["unresolved_subjects"] for v in rec["tables"].values())}
rec["seconds"] = round(time.time() - t0, 1)
json.dump(rec, open(SCR + "/METADATA_BUILD.json", "w"), indent=1)
log("done", json.dumps(rec["CHECKS"]))
