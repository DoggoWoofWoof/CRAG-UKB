"""Type-SET named-rate census over the whole frozen graph.

    PYTHONHASHSEED=0 python .../typeset_rate.py

WHAT IT MEASURES
  The per-type census (_type_named_rate.json) says how often nodes carrying type T have a name.
  That is weaker than it looks: 1.4% of common.document nodes are named, but nearly all of those
  are documents that are ALSO common.topic. The question the recovery campaign needs answered is
  about the node's exact situation -- "among every node in Freebase whose type signature is
  exactly {common.document}, how many have a name?" -- because that is the population an
  unnamed {common.document} node actually belongs to. So this pass computes, for every node in
  canonical/metadata/type.parquet, the exact set of types it carries, and counts per set:
      total     nodes in the graph with exactly this type set
      unnamed   of those, nodes in the 69.78M unresolved residue
      named     total - unnamed
  A set with named == 0 has zero named instances anywhere in the snapshot: the empirical
  evidence grade the user specified, measured at the strictest granularity we have.

HOW
  type.parquet is sorted only within two-row-group blocks (one block per source member), so a
  node's type rows recur across blocks and a streaming groupby is unsound. Two passes instead:
    pass 1  stream the 400 row groups, turn each row into (node_uid, type_id) with
            node_uid = hash(node_id) under PYTHONHASHSEED=0, and append it to one of 16 bucket
            files chosen by the low bits of node_uid -- every row of a node lands in one bucket.
    pass 2  per bucket (~16M rows): sort by (node_uid, type_id), drop duplicate pairs, cut runs,
            and take each run's signature as the uint64 wrap-around sum of a fixed random 64-bit
            value per type (order-independent; collision odds ~1e-6 over a few million sets).
  Residue membership is a searchsorted lookup in the sorted residue array.

OUTPUT (all under _acquisition/, append-only)
  _typeset_named_rate.parquet   sig, types ("|"-joined dotted paths), n_types, total, unnamed, named
  _residue_typeset_sig.parquet  node_uid, sig            for every residue node that has a type
  _typeset_rate_summary.json    counts and spot checks
"""
import sys, io, os, json, time, shutil
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
TMP = f"{ACQ}/_typeset_buckets"
NB = 16
t0 = time.time()

U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
assert np.all(np.diff(U) > 0)
print(f"residue: {len(U):,} node_uids", flush=True)

# ------------------------------------------------------------------ pass 1: bucket the pairs
f = pq.ParquetFile(f"{V3}/canonical/metadata/type.parquet")
NRG = f.metadata.num_row_groups
type_id = {}
shutil.rmtree(TMP, ignore_errors=True)
os.makedirs(TMP)
fh = [open(f"{TMP}/b{b:02d}.bin", "wb") for b in range(NB)]
n_rows = 0
for g in range(NRG):
    rg = f.read_row_group(g)
    d = pc.dictionary_encode(rg.column("subject").combine_chunks())
    suid = np.fromiter(map(hash, d.dictionary.to_pylist()), np.int64, len(d.dictionary))
    uid = suid[d.indices.to_numpy()]
    ty = pc.dictionary_encode(rg.column("type").combine_chunks())
    tmap = np.fromiter((type_id.setdefault(p, len(type_id)) for p in ty.dictionary.to_pylist()),
                       np.int64, len(ty.dictionary))
    tid = tmap[ty.indices.to_numpy()]
    b = (uid & (NB - 1)).astype(np.int64)
    o = np.argsort(b, kind="stable")
    cuts = np.searchsorted(b[o], np.arange(NB + 1))
    rec = np.empty(len(uid), dtype=[("u", np.int64), ("t", np.int32)])
    rec["u"], rec["t"] = uid[o], tid[o]
    for k in range(NB):
        if cuts[k + 1] > cuts[k]:
            fh[k].write(rec[cuts[k]:cuts[k + 1]].tobytes())
    n_rows += len(uid)
    if g % 50 == 0:
        print(f"  pass1 rg {g}/{NRG} rows={n_rows:,} types={len(type_id):,} ({time.time()-t0:.0f}s)",
              flush=True)
for h in fh:
    h.close()
rng = np.random.default_rng(20260907)
type_rnd = rng.integers(1, 2**63, size=len(type_id), dtype=np.uint64)
print(f"pass1 done: {n_rows:,} rows, {len(type_id):,} types ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ pass 2: group per bucket
tot, unn, sig_types = {}, {}, {}
res_uid, res_sig = [], []
n_nodes = 0
for k in range(NB):
    rec = np.fromfile(f"{TMP}/b{k:02d}.bin", dtype=[("u", np.int64), ("t", np.int32)])
    o = np.lexsort((rec["t"], rec["u"]))
    su, st = rec["u"][o], rec["t"][o].astype(np.int64)
    del rec, o
    keep = np.r_[True, (su[1:] != su[:-1]) | (st[1:] != st[:-1])]
    su, st = su[keep], st[keep]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    ends = np.r_[starts[1:], len(su)]
    sigs = np.add.reduceat(type_rnd[st], starts).astype(np.uint64)
    uids = su[starts]
    u, first, inv = np.unique(sigs, return_index=True, return_inverse=True)
    for s, i in zip(u.tolist(), first.tolist()):
        if s not in sig_types:
            sig_types[s] = tuple(int(x) for x in st[starts[i]:ends[i]])
    pos = np.searchsorted(U, uids)
    pos[pos >= len(U)] = 0
    inres = U[pos] == uids
    cnt = np.bincount(inv, minlength=len(u))
    cun = np.bincount(inv, weights=inres, minlength=len(u)).astype(np.int64)
    for s, a, b in zip(u.tolist(), cnt.tolist(), cun.tolist()):
        tot[s] = tot.get(s, 0) + a
        if b:
            unn[s] = unn.get(s, 0) + b
    res_uid.append(uids[inres]); res_sig.append(sigs[inres])
    n_nodes += len(uids)
    print(f"  pass2 bucket {k}/{NB} nodes={n_nodes:,} sets={len(tot):,} ({time.time()-t0:.0f}s)",
          flush=True)
shutil.rmtree(TMP, ignore_errors=True)

inv = {v: k for k, v in type_id.items()}
sigs = sorted(tot, key=lambda s: -tot[s])
pq.write_table(pa.table({
    "sig": pa.array(sigs, pa.uint64()),
    "types": pa.array(["|".join(inv[t] for t in sig_types[s]) for s in sigs]),
    "n_types": pa.array([len(sig_types[s]) for s in sigs], pa.int32()),
    "total": pa.array([tot[s] for s in sigs], pa.int64()),
    "unnamed": pa.array([unn.get(s, 0) for s in sigs], pa.int64()),
    "named": pa.array([tot[s] - unn.get(s, 0) for s in sigs], pa.int64())}),
    f"{ACQ}/_typeset_named_rate.parquet", compression="zstd")
ru, rs = np.concatenate(res_uid), np.concatenate(res_sig)
pq.write_table(pa.table({"node_uid": pa.array(ru, pa.int64()), "sig": pa.array(rs, pa.uint64())}),
               f"{ACQ}/_residue_typeset_sig.parquet", compression="zstd")

zero = [s for s in sigs if tot[s] == unn.get(s, 0)]
spot = {}
for want in ("common.document", "type.content_import", "type.content", "type.permission",
             "common.image", "common.topic|common.document", "film.performance",
             "base.schemastaging.nutrition_information", "pipeline.vote"):
    ids = tuple(sorted(type_id[p] for p in want.split("|") if p in type_id))
    s = next((s for s in sigs if sig_types[s] == ids), None)
    if s is not None:
        spot[want] = {"total": tot[s], "unnamed": unn.get(s, 0), "named": tot[s] - unn.get(s, 0)}
rec = {"schema": "TYPESET_NAMED_RATE/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "source": "canonical/metadata/type.parquet (frozen); residue = _unresolved_population.npz",
       "type_rows": n_rows, "typed_nodes_in_graph": n_nodes,
       "distinct_type_sets": len(tot), "distinct_types": len(type_id),
       "type_sets_with_zero_named_instances": len(zero),
       "graph_nodes_in_zero_named_sets": sum(tot[s] for s in zero),
       "residue_nodes_in_zero_named_sets": sum(unn.get(s, 0) for s in zero),
       "residue_typed_nodes": int(len(ru)),
       "spot_checks_exact_set": spot,
       "signature": "uint64 wraparound sum of a fixed random 64-bit value per type (seed 20260907); (node,type) pairs deduplicated",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{ACQ}/_typeset_rate_summary.json", "w", encoding="utf-8") as fo:
    json.dump(rec, fo, indent=1)
print(json.dumps(rec, indent=1))
