"""W0 Task 7/SHARDING — build the canonical dense retrieval manifest for a world corpus.

Per the locked methodology (EXHAUSTIVE_SHARDED_FP32_COMPUTE), persist for every shard:
  shard_id, world_node_id range (first,last), embedding row range, dtype, shape, sha256.
Global results must map back to canonical world_node_ids (never shard-local rows).

Writes: data/canonical/<ds>/encodings/dense/docs/retrieval_manifest.json
Usage:  python scratchpad/_w0_retrieval_manifest.py <ds> [dense|splade]
"""
import os, sys, json, hashlib
import numpy as np

ds = sys.argv[1]
enc = sys.argv[2] if len(sys.argv) > 2 else "dense"
BASE = f"data/canonical/{ds}/encodings/{enc}/docs"
mf = json.load(open(f"{BASE}/manifest.json", encoding="utf-8"))
n_items = mf["n_items"]; shard_size = mf["shard_size"]; n_shards = mf["n_shards"]

def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()

def npy_header(path):
    """Read shape/dtype from .npy header WITHOUT loading the array into RAM."""
    a = np.load(path, mmap_mode="r")
    return a.shape, str(a.dtype)

shards = []
row_cursor = 0
total_rows = 0
for s in range(n_shards):
    npy = f"{BASE}/shard_{s:05d}.npy"
    idp = f"{BASE}/ids_{s:05d}.json"
    ids = json.load(open(idp, encoding="utf-8"))
    shape, dtype = npy_header(npy)
    assert shape[0] == len(ids), f"shard {s}: emb rows {shape[0]} != ids {len(ids)}"
    rec = dict(
        shard_id=s,
        row_start=row_cursor, row_end=row_cursor + shape[0],  # [start,end)
        n_rows=shape[0], dim=shape[1], dtype=dtype,
        world_node_id_first=ids[0], world_node_id_last=ids[-1],
        emb_sha256=sha256(npy), ids_sha256=sha256(idp),
    )
    shards.append(rec)
    row_cursor += shape[0]
    total_rows += shape[0]
    print(f"  shard {s:05d}: rows[{rec['row_start']}:{rec['row_end']}) dim={rec['dim']} {dtype} "
          f"ids[{rec['world_node_id_first']}..{rec['world_node_id_last']}]", flush=True)

assert total_rows == n_items, f"total rows {total_rows} != n_items {n_items}"
out = dict(
    dataset=ds, encoder_kind=enc, encoder=mf.get("encoder"),
    n_items=n_items, total_rows=total_rows, dim=shards[0]["dim"], dtype=shards[0]["dtype"],
    shard_size=shard_size, n_shards=n_shards,
    id_space="canonical world_node_id (row order is NOT the id system)",
    source_sha256=mf.get("source_sha256"),
    retrieval_method="EXHAUSTIVE_SHARDED_FP32_COMPUTE",
    shards=shards,
)
outp = f"{BASE}/retrieval_manifest.json"
json.dump(out, open(outp, "w", encoding="utf-8"), indent=2)
print(f"WROTE {outp}  n_shards={n_shards} total_rows={total_rows}==n_items:{total_rows==n_items}")
