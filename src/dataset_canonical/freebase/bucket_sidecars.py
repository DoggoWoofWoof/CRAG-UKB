# -*- coding: utf-8 -*-
"""Re-partition the uid-keyed name sidecars by CANONICAL NODE SHARD so the name composition can
run as one aligned pass over canonical/nodes + overlay_v1 + semantic_v1.

Sources (all frozen, read only):
    inference_overlay_v1/part_*.parquet + inference_admissibility_v1 (row-aligned) + v2 amendment
        -> effective reject_reason per inferred row (v2 amendment overrides v1 by node_uid; code 2)
    _acquisition/cascade_names.parquet            534,532 recovered ACTUAL-layer names
    _acquisition/semantic_kind_v2_1.parquet       nameless_grade / recovery_class for the 69.8M residue

Output: scratchpad/fb4/sidecar/<canonical shard>.parquet, one row per canonical row that has any
sidecar fact, sorted by row:  row int32, node_uid int64, inf_name, inf_source, inf_rule,
inf_reject int8, cascade_name, cascade_source, cascade_original bool, nameless_grade, recovery_class.
Alignment: position = searchsorted(nodes/node_uid.npy, uid); (shard, row) = inverse of pos/<shard>.npy.
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
import pyarrow as pa
import pyarrow.parquet as pq

FB = "data/final_canonical/freebase_v3"
OUT = "data/final_canonical/freebase"
SCR = "scratchpad/fb4"
SIDE = SCR + "/sidecar"
PIECES = SCR + "/sidecar_pieces"
N = 301977131
t0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - t0), *a, flush=True)


for d in (SIDE, PIECES):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
uids = np.load(OUT + "/nodes/node_uid.npy", mmap_mode="r")
shards = sorted(os.path.basename(f)[:-4] for f in glob.glob(SCR + "/pos/*.npy"))
assert len(shards) == 262, len(shards)
shard_of_pos = np.full(N, -1, dtype=np.int16)
row_of_pos = np.full(N, -1, dtype=np.int32)
for si, s in enumerate(shards):
    p = np.load(SCR + "/pos/%s.npy" % s)
    shard_of_pos[p] = si
    row_of_pos[p] = np.arange(len(p), dtype=np.int32)
assert (shard_of_pos >= 0).all()
log("inverse position map built")


def locate(u):
    p = np.searchsorted(uids, u)
    p[p >= N] = 0
    ok = uids[p] == u
    return p, ok


class ShardWriters:
    """One ParquetWriter per (family, shard), rows appended as they arrive."""

    def __init__(self, family):
        self.family = family
        self.w = {}

    def write(self, si, table):
        if si not in self.w:
            self.w[si] = pq.ParquetWriter(PIECES + "/%s_%s.parquet" % (self.family, shards[si]), table.schema, compression="zstd")
        self.w[si].write_table(table)

    def close(self):
        for w in self.w.values():
            w.close()


def undict(t):
    cols = []
    for c in t.columns:
        cols.append(c.cast(c.type.value_type) if pa.types.is_dictionary(c.type) else c)
    return pa.table(cols, names=t.column_names)


def scatter(writers, t, si, row):
    order = np.argsort(si, kind="stable")
    si_s = si[order]
    cuts = np.flatnonzero(np.diff(si_s)) + 1
    starts = np.concatenate([[0], cuts]); ends = np.concatenate([cuts, [len(si_s)]])
    for s, e in zip(starts, ends):
        sel = order[s:e]
        sub = t.take(pa.array(sel)).add_column(0, "row", pa.array(row[sel].astype(np.int32)))
        writers.write(int(si_s[s]), sub)


# ---- inference (with effective admissibility) ----------------------------------------------------
amend = json.load(open(FB + "/inference_admissibility_v2_amendment.json", encoding="utf-8"))
assert amend["NEW_REJECT_REASON"]["code"] == 2
amend_uids = np.array(sorted(int(r["node_uid"]) for r in amend["REJECTED_ROWS"]), dtype=np.int64)
assert len(amend_uids) == 36
inf_parts = sorted(glob.glob(FB + "/inference_overlay_v1/part_*.parquet"))
adm_parts = sorted(glob.glob(FB + "/inference_admissibility_v1/part_*.parquet"))
assert len(inf_parts) == len(adm_parts) == 32
wi = ShardWriters("inf")
n_inf = 0
reject_hist = np.zeros(4, dtype=np.int64)
amend_hits = 0
for f, g in zip(inf_parts, adm_parts):
    t = undict(pq.read_table(f, columns=["node_uid", "inferred_name", "inference_source", "inference_rule"]))
    a = pq.read_table(g)
    assert t.num_rows == a.num_rows, (f, t.num_rows, a.num_rows)
    u = t["node_uid"].to_numpy()
    rr = a["reject_reason"].to_numpy().astype(np.int8).copy()
    hit = np.isin(u, amend_uids)
    rr[hit] = 2
    amend_hits += int(hit.sum())
    reject_hist += np.bincount(rr, minlength=4)[:4]
    p, ok = locate(u)
    assert ok.all(), "inference uid not in node universe: " + f
    t = t.append_column("inf_reject", pa.array(rr)).rename_columns(["node_uid", "inf_name", "inf_source", "inf_rule", "inf_reject"])
    scatter(wi, t, shard_of_pos[p], row_of_pos[p])
    n_inf += t.num_rows
    log("inference", os.path.basename(f), n_inf)
wi.close()
assert amend_hits == 36, amend_hits
log("inference rows", n_inf, "effective reject hist", reject_hist.tolist())

# ---- cascade ---------------------------------------------------------------------------------------
c = undict(pq.read_table(FB + "/_acquisition/cascade_names.parquet", columns=["node_uid", "display_name", "source", "is_original_name"]))
assert c.num_rows == 534532
u = c["node_uid"].to_numpy()
assert len(np.unique(u)) == len(u)
p, ok = locate(u)
assert ok.all()
c = c.rename_columns(["node_uid", "cascade_name", "cascade_source", "cascade_original"])
wc = ShardWriters("cas")
scatter(wc, c, shard_of_pos[p], row_of_pos[p])
wc.close()
log("cascade bucketed")

# ---- semantic kind (nameless grade) --------------------------------------------------------------
pf = pq.ParquetFile(FB + "/_acquisition/semantic_kind_v2_1.parquet")
ws = ShardWriters("sk")
n_sk = 0
for rg in range(pf.num_row_groups):
    t = undict(pf.read_row_group(rg, columns=["node_uid", "nameless_grade", "recovery_class"]))
    u = t["node_uid"].to_numpy()
    p, ok = locate(u)
    assert ok.all(), "semantic_kind uid not in universe"
    scatter(ws, t, shard_of_pos[p], row_of_pos[p])
    n_sk += t.num_rows
    if rg % 10 == 0:
        log("semantic_kind rg", rg, n_sk)
ws.close()
assert n_sk == 69777967, n_sk
del shard_of_pos, row_of_pos

# ---- per shard: full outer join of the three families on row ---------------------------------------
written = 0
fam_rows = {"inf": 0, "cas": 0, "sk": 0}
for si, s in enumerate(shards):
    tabs = {}
    for fam in ("inf", "cas", "sk"):
        f = PIECES + "/%s_%s.parquet" % (fam, s)
        if os.path.exists(f):
            t = pq.read_table(f)
            r = t["row"].to_numpy()
            assert len(np.unique(r)) == len(r), ("duplicate rows", fam, s)
            tabs[fam] = t
            fam_rows[fam] += t.num_rows
    if not tabs:
        continue
    joined = None
    for fam, t in tabs.items():
        if joined is None:
            joined = t
        else:
            joined = joined.join(t, keys=["row", "node_uid"], join_type="full outer")
    joined = joined.sort_by("row")
    r = joined["row"].to_numpy()
    assert len(np.unique(r)) == len(r)
    for col, typ in [("inf_name", pa.string()), ("inf_source", pa.string()), ("inf_rule", pa.string()), ("inf_reject", pa.int8()),
                     ("cascade_name", pa.string()), ("cascade_source", pa.string()), ("cascade_original", pa.bool_()),
                     ("nameless_grade", pa.string()), ("recovery_class", pa.string())]:
        if col not in joined.column_names:
            joined = joined.append_column(col, pa.nulls(joined.num_rows, typ))
    joined = joined.select(["row", "node_uid", "inf_name", "inf_source", "inf_rule", "inf_reject", "cascade_name", "cascade_source",
                            "cascade_original", "nameless_grade", "recovery_class"])
    pq.write_table(joined, SIDE + "/%s.parquet" % s, compression="zstd")
    written += joined.num_rows
    if si % 25 == 0:
        log("sidecar", s, joined.num_rows)
shutil.rmtree(PIECES)
assert fam_rows == {"inf": n_inf, "cas": 534532, "sk": n_sk}, fam_rows
rec = {"RECORD": "SIDECAR_BUCKETS", "inference_rows": n_inf, "inference_reject_hist_effective": reject_hist.tolist(),
       "amendment_rows_applied": amend_hits, "cascade_rows": 534532, "semantic_kind_rows": n_sk,
       "sidecar_rows_written": written, "seconds": round(time.time() - t0, 1)}
json.dump(rec, open(SCR + "/SIDECAR_BUCKETS.json", "w"), indent=1)
log("done", json.dumps(rec))
