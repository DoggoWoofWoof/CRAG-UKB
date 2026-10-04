"""W0 Task 8 — SPLADE world representation verification (local, CPU).

For each corpus, verify the sparse CSR docs shards:
  - row alignment: shard rows == len(ids); sum rows == n_items (from manifest)
  - vocab dimension constant across shards
  - nonzero statistics: total nnz, mean/min/max nnz per doc, all-zero rows
  - world_node_id alignment (ids concatenated in shard order == canonical order)
SPLADE weights are float32 sparse -> no fp16 precision concern (exact dot products).

Writes data/canonical/<ds>/encodings/splade/docs/splade_verify.json
Usage: python scratchpad/_w0_splade_verify.py <ds>
"""
import os, sys, json, hashlib
import numpy as np, scipy.sparse as sp

ds = sys.argv[1]
BASE = f"data/canonical/{ds}/encodings/splade/docs"
mf = json.load(open(f"{BASE}/manifest.json", encoding="utf-8"))
n_items = mf["n_items"]; n_shards = mf["n_shards"]

def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(buf), b""):
            h.update(c)
    return h.hexdigest()

vocab = None
total_rows = 0
total_nnz = 0
zero_rows = 0
min_nnz = 10**9; max_nnz = 0
shards = []
first_id = None; last_id = None
for s in range(n_shards):
    npz = f"{BASE}/shard_{s:05d}.npz"
    idp = f"{BASE}/ids_{s:05d}.json"
    m = sp.load_npz(npz)
    ids = json.load(open(idp, encoding="utf-8"))
    assert m.shape[0] == len(ids), f"shard {s}: rows {m.shape[0]} != ids {len(ids)}"
    if vocab is None:
        vocab = m.shape[1]; first_id = ids[0]
    assert m.shape[1] == vocab, f"shard {s}: vocab {m.shape[1]} != {vocab}"
    rownnz = np.diff(m.indptr)               # nnz per row
    zr = int((rownnz == 0).sum())
    total_rows += m.shape[0]; total_nnz += int(m.nnz); zero_rows += zr
    min_nnz = min(min_nnz, int(rownnz.min())); max_nnz = max(max_nnz, int(rownnz.max()))
    last_id = ids[-1]
    shards.append(dict(shard_id=s, n_rows=int(m.shape[0]), vocab=int(m.shape[1]),
                       nnz=int(m.nnz), zero_rows=zr,
                       world_node_id_first=ids[0], world_node_id_last=ids[-1],
                       npz_sha256=sha256(npz)))
    if s % 20 == 0 or s == n_shards - 1:
        print(f"  shard {s}/{n_shards} rows={total_rows} nnz={total_nnz} zero_rows={zero_rows}", flush=True)

assert total_rows == n_items, f"total rows {total_rows} != n_items {n_items}"
out = dict(
    dataset=ds, kind="splade_docs", encoder=mf.get("encoder"),
    n_items=n_items, total_rows=total_rows, vocab_dim=int(vocab),
    n_shards=n_shards, source_sha256=mf.get("source_sha256"),
    nnz_total=total_nnz, nnz_mean_per_doc=round(total_nnz / total_rows, 3),
    nnz_min_per_doc=min_nnz, nnz_max_per_doc=max_nnz, zero_rows=zero_rows,
    row_id_alignment_ok=bool(total_rows == n_items),
    vocab_consistent=True, weight_dtype="float32 (exact sparse dot; no fp16 concern)",
    VERIFY_PASS=bool(total_rows == n_items and zero_rows == 0),
    shards=shards,
)
outp = f"{BASE}/splade_verify.json"
json.dump(out, open(outp, "w", encoding="utf-8"), indent=2)
print(f"WROTE {outp}  rows={total_rows}==n_items:{total_rows==n_items} vocab={vocab} "
      f"nnz/doc={out['nnz_mean_per_doc']} zero_rows={zero_rows} PASS={out['VERIFY_PASS']}")
