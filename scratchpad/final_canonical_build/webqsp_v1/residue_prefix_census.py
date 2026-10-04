"""What do the unresolved node_ids actually LOOK like?

The overlay carries node_uid but not node_id, so the residue's identifier shapes have never been
counted. This decides whether P2671 (Google Knowledge Graph IDs, mostly /g/) can matter at all: if
the residue holds no g. nodes, the /g/ half of step 4 is answerable without fetching 8.18M
statements. It also emits the residue's node_ids, which every later cascade step needs.

69.8M node_ids do not fit in a Python list on this box, so matched rows are spilled per shard and
the prefix histogram is built with Arrow kernels rather than a Python loop.

Reads canonical/nodes and overlay_v1. Writes only under _acquisition/.
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"
OUT = f"{V3}/_acquisition/residue_ids"
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
os.makedirs(OUT, exist_ok=True)
z = np.load(CACHE)
U, KC = z["U"], z["KC"]
NU = len(U)
print(f"unresolved population: {NU:,}", flush=True)

pref = collections.Counter()
pref_ent = collections.Counter()
seen = 0
files = sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet"))
for k, fp in enumerate(files):
    t = pq.read_table(fp, columns=["node_uid", "node_id"])
    u = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    j = np.searchsorted(U, u)
    np.clip(j, 0, NU - 1, out=j)
    m = U[j] == u
    if m.any():
        idx = np.flatnonzero(m)
        ia = pa.array(idx)
        ids = t["node_id"].take(ia).combine_chunks()
        kc = KC[j[idx]]
        seen += len(idx)
        # Arrow slice + value_counts: a Python loop over 69.8M strings is the slow half here
        p2 = pc.utf8_slice_codeunits(ids, 0, 2)
        vc = pc.value_counts(p2)
        for st in vc:
            pref[st["values"].as_py()] += st["counts"].as_py()
        em = np.flatnonzero(kc == 0)
        if em.size:
            vc = pc.value_counts(pc.utf8_slice_codeunits(ids.take(pa.array(em)), 0, 2))
            for st in vc:
                pref_ent[st["values"].as_py()] += st["counts"].as_py()
        pq.write_table(pa.table({"node_uid": pa.array(u[idx], pa.int64()),
                                 "node_id": ids,
                                 "kind_code": pa.array(kc, pa.int8())}),
                       f"{OUT}/{os.path.basename(fp)}", compression="zstd")
        del ids, ia, idx, kc
    del t, u, j, m
    if (k + 1) % 40 == 0:
        print(f"  [{k+1:3d}/{len(files)}] matched={seen:,} ({time.time()-t0:.0f}s)", flush=True)

rec = {"schema": "RESIDUE_PREFIX_CENSUS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "reads canonical/ and overlay_v1/, writes neither.",
       "population": int(NU), "matched_in_node_table": seen,
       "COMPLETE": seen == NU,
       "WHY": ("decides whether P2671 (/g/ Google Knowledge Graph IDs) can reach this residue at "
               "all, and materialises residue node_ids for every later cascade step."),
       "prefix_all": dict(pref.most_common(20)),
       "prefix_entity_mid": dict(pref_ent.most_common(20)),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_RESIDUE_PREFIX_CENSUS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nmatched {seen:,} / {NU:,}  COMPLETE={seen==NU}")
print("all residue:       ", dict(pref.most_common(8)))
print("ENTITY_MID residue:", dict(pref_ent.most_common(8)))
print(f"{time.time()-t0:.0f}s")
