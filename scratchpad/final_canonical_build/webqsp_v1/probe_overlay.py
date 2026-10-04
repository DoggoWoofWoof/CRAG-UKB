"""Classify a list of MIDs in the frozen overlay + report their in-graph name/key/alias status.

    PYTHONHASHSEED=0 python .../probe_overlay.py m.0jtlqs m.0k7nk5 ...

Reads semantic_kind_v2_1.parquet by row group so the 659 MB table is never fully materialised.
"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()
mids = sys.argv[1:]
want = np.array([hash(m) for m in mids], np.int64)
name_of = dict(zip(want.tolist(), mids))

COLS = ["node_uid", "node_kind_frozen", "semantic_kind", "evidence", "nameless_grade",
        "recovery_class", "set_named_rate_band", "max_type_named_rate"]
pf = pq.ParquetFile(f"{ACQ}/semantic_kind_v2_1.parquet")
have = {c for c in pf.schema_arrow.names}
cols = [c for c in COLS if c in have]
print(f"columns present: {cols}\nmissing: {[c for c in COLS if c not in have]}", flush=True)
found = {}
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=cols)
    u = t["node_uid"].to_numpy()
    m = np.isin(u, want)
    if not m.any():
        continue
    sub = t.filter(pa.array(m))
    for i in range(sub.num_rows):
        d = {c: sub[c][i].as_py() for c in cols}
        found[d["node_uid"]] = d
    if len(found) == len(want):
        break
print(f"overlay rows found: {len(found)}/{len(want)}  ({time.time()-t0:.0f}s)\n", flush=True)
for uid in want.tolist():
    d = found.get(uid)
    print(f"{name_of[uid]:<14} " + ("<<< NOT IN OVERLAY >>>" if d is None else
          "  ".join(f"{c}={d[c]!r}" for c in cols if c != "node_uid")))
print(f"\n{time.time()-t0:.0f}s")
