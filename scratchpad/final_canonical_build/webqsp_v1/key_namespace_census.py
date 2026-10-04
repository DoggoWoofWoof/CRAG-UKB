"""Which key namespaces does the UNNAMED residue actually carry?  A census of key.parquet.

    PYTHONHASHSEED=0 python .../key_namespace_census.py

WHY
  key.parquet holds 143,981,520 rows of /type/object/key, and it is the graph's OWN data: no
  network, no reconciliation, no third party.  One namespace in it has already paid for itself
  sixteen times over -- the WordNet sense keys decoded 206,977 lemmas, more than five external
  sources combined.  The cascade still shows WIKIPEDIA_TITLE with no source at all, yet a key of the
  form /wikipedia/en/Some_Title IS a Wikipedia title, held by Freebase, reachable by exact
  identifier with no name matching of any kind.  Before writing a decoder for any namespace, this
  measures which namespaces the residue actually has and how many nodes each would reach.

WHAT IS COUNTED
  For every key row whose subject is an unnamed residue node, the namespace is the key path minus
  its last segment (/wikipedia/en/Foo -> /wikipedia/en).  Reported per namespace:
      rows            how many key rows
      nodes           how many DISTINCT residue nodes (this is the reach, rows are not reach)
      hunt_nodes      how many of those are in the 14,956,306 hunt residue
      sample keys     so the decodability of the namespace can be judged, not assumed
  Nodes are counted distinctly per namespace with a bitmap, because a node can hold many keys in one
  namespace and rows would badly overstate reach.

Read-only.  OUTPUT: _acquisition/_key_namespace_census.parquet and V3_KEY_NAMESPACE_CENSUS.json
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()
TOPN = 120

U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
HUNTC = pa.array(["LIKELY_REAL_ENTITY", "UNTYPED_CANDIDATE", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY",
                  "LIKELY_CREATIVE_WORK", "LIKELY_PERSON"])
H = np.sort(pc.filter(sk["node_uid"], pc.is_in(rc, value_set=HUNTC)).to_numpy())
del sk, rc
hunt_mask = np.zeros(len(U), bool)
p = np.clip(np.searchsorted(U, H), 0, len(U) - 1)
hunt_mask[p[U[p] == H]] = True
print(f"residue {len(U):,}  hunt {len(H):,} ({time.time()-t0:.0f}s)", flush=True)

KEY = f"{V3}/canonical/metadata/key.parquet"
pf = pq.ParquetFile(KEY)
print(f"key.parquet {pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups\n"
      f"schema {pf.schema_arrow.names}", flush=True)

rows = collections.Counter()
samples = collections.defaultdict(list)
# per-namespace distinct-node bitmaps, added lazily so memory stays bounded by the namespaces seen
bitmaps = {}
n_hit = 0
# key.parquet is keyed on the node_id STRING, not on node_uid, so each row group's subjects are
# hashed here. node_uid = hash(node_id) under PYTHONHASHSEED=0, which is why this file refuses to
# run without it: a different seed would silently match nothing.
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["subject", "key"])
    sub = t["subject"]
    sub = pc.cast(sub, pa.string()) if pa.types.is_dictionary(sub.type) else sub
    subj = np.fromiter((hash(x) for x in sub.to_pylist()), np.int64, t.num_rows)
    pos = np.clip(np.searchsorted(U, subj), 0, len(U) - 1)
    hit = U[pos] == subj
    if not hit.any():
        continue
    n_hit += int(hit.sum())
    keys = pc.filter(t["key"], pa.array(hit))
    keys = pc.cast(keys, pa.string()) if pa.types.is_dictionary(keys.type) else keys
    hp = pos[hit]
    for k, i in zip(keys.to_pylist(), hp.tolist()):
        ns = k.rsplit("/", 1)[0] if "/" in k else k
        rows[ns] += 1
        b = bitmaps.get(ns)
        if b is None:
            b = bitmaps[ns] = np.zeros(len(U), bool)
        b[i] = True
        s = samples[ns]
        if len(s) < 4:
            s.append(k)
    if g % 10 == 0:
        print(f"  rg {g}/{pf.metadata.num_row_groups}  residue key rows {n_hit:,}  "
              f"namespaces {len(bitmaps):,}  ({time.time()-t0:.0f}s)", flush=True)

out = []
for ns, b in bitmaps.items():
    out.append((ns, rows[ns], int(b.sum()), int((b & hunt_mask).sum()), samples[ns]))
out.sort(key=lambda r: -r[2])
pq.write_table(pa.table({"namespace": pa.array([r[0] for r in out]),
                         "key_rows": pa.array([r[1] for r in out], pa.int64()),
                         "distinct_residue_nodes": pa.array([r[2] for r in out], pa.int64()),
                         "distinct_hunt_nodes": pa.array([r[3] for r in out], pa.int64()),
                         "sample_keys": pa.array([" | ".join(r[4]) for r in out])}),
                f"{ACQ}/_key_namespace_census.parquet", compression="zstd")

any_key = np.zeros(len(U), bool)
for b in bitmaps.values():
    any_key |= b
rec = {"schema": "KEY_NAMESPACE_CENSUS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "read-only over the frozen graph; new record only.",
       "key_rows_total": int(pf.metadata.num_rows),
       "key_rows_on_residue_nodes": n_hit,
       "residue_nodes_holding_at_least_one_key": int(any_key.sum()),
       "hunt_nodes_holding_at_least_one_key": int((any_key & hunt_mask).sum()),
       "distinct_namespaces": len(out),
       "REACH_IS_NODES_NOT_ROWS": "distinct_residue_nodes is the reach. key_rows overstates it "
                                  "wherever a node holds several keys in one namespace.",
       "top_namespaces": [{"namespace": r[0], "key_rows": r[1], "distinct_residue_nodes": r[2],
                           "distinct_hunt_nodes": r[3], "sample_keys": r[4]} for r in out[:TOPN]],
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_KEY_NAMESPACE_CENSUS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items() if k != "top_namespaces"}, indent=1))
print("\ntop namespaces on unnamed residue nodes (reach = distinct nodes):")
for r in out[:45]:
    print(f"  {r[0][:52]:<52} nodes {r[2]:>10,}  hunt {r[3]:>9,}  rows {r[1]:>11,}")
    print(f"      {r[4][:2]}")
