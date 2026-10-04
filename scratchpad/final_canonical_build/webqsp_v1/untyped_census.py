"""What ARE the 5,832,113 UNTYPED_CANDIDATE nodes?  A predicate census over every edge.

    PYTHONHASHSEED=0 python .../untyped_census.py

WHY
  UNTYPED_CANDIDATE is the second-largest class in the 14,956,306 hunt residue and the only one with
  no type evidence at all: no type row, so the type-set named-rate machinery says nothing about it.
  The ONLY thing the frozen graph states about these nodes is which edges touch them.  This pass
  answers, without guessing:
    * how many are pure sinks (only ever a dst), pure sources, or both
    * which relations point AT them, and which point away
    * the degree distribution, so "referenced exactly once by one property" can be separated from
      "a real hub that merely lost its type"
  A node that is only ever the object of /common/topic/image is an image blob; a node that is the
  subject of dozens of properties is a real entity that lost its type row.  Those two need different
  terminal buckets, and this census is what tells them apart.

APPROACH
  One streaming pass over canonical/edges/*.parquet reading only (src, rel, dst).  Membership by
  np.searchsorted against the sorted untyped uid array.  Per-relation in/out counts accumulated in
  int64 arrays indexed by rel id; per-node in/out degree accumulated with np.bincount over the
  membership positions.  Nothing is materialised per edge.

OUTPUT (append-only)
  _acquisition/_untyped_predicate_census.parquet   rel_id, relation, n_in, n_out, n_distinct_in...
  V3_UNTYPED_CENSUS.json                           the summary and the top predicates
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()

# ---------------------------------------------------------------- the cohort
sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
m = pc.equal(rc, "UNTYPED_CANDIDATE")
U = np.sort(pc.filter(sk["node_uid"], m).to_numpy())
del sk, rc, m
print(f"UNTYPED_CANDIDATE nodes: {len(U):,} ({time.time()-t0:.0f}s)", flush=True)

# relation vocabulary: `rel` in the edges is the int64 rel_uid hash, NOT a dense index,
# so relations are mapped to dense ids by searchsorted over the sorted rel_uid array.
rel_t = pq.read_table(f"{V3}/canonical/relations.parquet", columns=["rel_uid", "relation"])
RU = rel_t["rel_uid"].to_numpy()
RN = rel_t["relation"].to_pylist()
o = np.argsort(RU)
RU = RU[o]
RN = [RN[i] for i in o.tolist()]
NR = len(RU)
print(f"relations: {NR:,}", flush=True)

n_in = np.zeros(NR + 2, np.int64)     # edges whose dst is untyped
n_out = np.zeros(NR + 2, np.int64)    # edges whose src is untyped
deg_in = np.zeros(len(U), np.int64)
deg_out = np.zeros(len(U), np.int64)

files = sorted(glob.glob(f"{V3}/canonical/edges/*.parquet"))
print(f"edge files: {len(files)}", flush=True)
tot = 0
for fi, fp in enumerate(files):
    pf = pq.ParquetFile(fp)
    for g in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(g, columns=["src", "rel", "dst"])
        src = rg["src"].to_numpy(); rel = rg["rel"].to_numpy(); dst = rg["dst"].to_numpy()
        tot += len(src)
        for side, arr, ncnt, dcnt in (("in", dst, n_in, deg_in), ("out", src, n_out, deg_out)):
            p = np.searchsorted(U, arr); np.clip(p, 0, len(U) - 1, out=p)
            hit = U[p] == arr
            if not hit.any():
                continue
            q = np.searchsorted(RU, rel[hit]); np.clip(q, 0, NR - 1, out=q)
            ok = RU[q] == rel[hit]
            ncnt += np.bincount(np.where(ok, q, NR), minlength=NR + 2)
            dcnt += np.bincount(p[hit], minlength=len(U))
    if fi % 20 == 0:
        print(f"  file {fi}/{len(files)} edges {tot:,} in={n_in.sum():,} out={n_out.sum():,} "
              f"({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- shape of the class
both = int(((deg_in > 0) & (deg_out > 0)).sum())
sink = int(((deg_in > 0) & (deg_out == 0)).sum())
source = int(((deg_in == 0) & (deg_out > 0)).sum())
iso = int(((deg_in == 0) & (deg_out == 0)).sum())


def hist(d):
    b = np.bincount(np.clip(d, 0, 11))
    h = {str(i): int(b[i]) for i in range(min(11, len(b)))}
    if len(b) > 11:
        h["11+"] = int(b[11])
    return h


rows = []
for r in range(NR + 2):
    if n_in[r] or n_out[r]:
        rows.append((r, RN[r] if r < NR else "<unknown rel_uid>", int(n_in[r]), int(n_out[r])))
rows.sort(key=lambda x: -(x[2] + x[3]))
pq.write_table(pa.table({"rel_id": pa.array([r[0] for r in rows], pa.int32()),
                         "relation": pa.array([r[1] for r in rows]),
                         "n_edges_into_untyped": pa.array([r[2] for r in rows], pa.int64()),
                         "n_edges_out_of_untyped": pa.array([r[3] for r in rows], pa.int64())}),
                f"{ACQ}/_untyped_predicate_census.parquet", compression="zstd")

rec = {"schema": "UNTYPED_PREDICATE_CENSUS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "new record; frozen graph read-only.",
       "population": int(len(U)),
       "population_source": "semantic_kind_v2_1.parquet recovery_class == UNTYPED_CANDIDATE",
       "edges_scanned": int(tot),
       "edges_touching_class": {"into": int(n_in.sum()), "out_of": int(n_out.sum())},
       "node_shape": {"referenced_only (pure sink)": sink, "refers_only (pure source)": source,
                      "both": both, "isolated (no edge at all)": iso},
       "in_degree_hist": hist(deg_in), "out_degree_hist": hist(deg_out),
       "distinct_relations_touching_class": len(rows),
       "top_40_by_total": [{"relation": r[1], "into": r[2], "out_of": r[3]} for r in rows[:40]],
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_UNTYPED_CENSUS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items() if k != "top_40_by_total"}, indent=1))
print("\ntop 30 predicates touching the untyped class:")
for r in rows[:30]:
    print(f"  {r[1][:70]:<70} in {r[2]:>12,}  out {r[3]:>12,}")
print(f"{time.time()-t0:.0f}s")
