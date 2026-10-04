"""semantic_text for every node whose display_name is not a real Freebase name.

ONE pass over all 2,062,430,072 edges. The frozen graph is read and never written.

WHAT THIS RENDERS
  A CVT is anonymous by design -- Freebase never named mediators -- so the honest description is
  what it IS and what roles it carries, not a name that never existed:
      "Marriage - roles: spouse, spouse, from, to; referenced as: spouse_s"
  An unnamed entity gets the same treatment from its own incident relations.

WHY NO SCRATCH DISK
  The obvious implementation spills a fact table and sorts it, which costs ~20 GB. It is not
  needed: with a per-node cap K the whole result fits a fixed (NEED x K) int32 array written by
  index, so there is no spill, no sort and no shuffle. Cost is RAM = NEED * K * 4 bytes.

  Storing the NEIGHBOUR of each role instead of just the relation would need another
  NEED x K int64 plus a 302M-row name join -- two shuffles and ~20 GB of scratch. That is the
  --neighbours design; it is deliberately NOT the default, because "type + roles" is what a
  mediator actually has to say.

DIRECTION IS NOT FAKED
  Edges are stored once in canonical direction, so a node's roles sit on both sides. An outgoing
  relation names the node's OWN role; an incoming one names the other end's role. They are kept in
  separate lists ("roles" vs "referenced as") rather than blended, so the text never implies the
  node holds a role it does not.
"""
import sys, io, os, json, glob, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

V3 = "data/final_canonical/freebase_v3"
OVL = f"{V3}/overlay_v1"
OUT = f"{V3}/semantic_v1"
EDGES = sorted(glob.glob(f"{V3}/canonical/edges/*.parquet"))
K = int(os.environ.get("SEM_K", "6"))          # roles kept per node
BATCH = 4_000_000
STRUCT = {"STRUCTURAL_INFERRED", "STRUCTURAL_FALLBACK"}
t0 = time.time()

# ------------------------------------------------------------------ 1. the NEED population
# exactly the nodes whose display_name is not a real name: the ones a role description helps.
uids = []
for fp in sorted(glob.glob(f"{OVL}/*.parquet")):
    t = pq.read_table(fp, columns=["node_uid", "display_name_source"])
    s = t["display_name_source"].combine_chunks()
    d = s if isinstance(s.type, pa.DictionaryType) else s.dictionary_encode()
    dv = d.dictionary.to_pylist()
    want = np.array([v in STRUCT for v in dv], dtype=bool)
    m = want[d.indices.to_numpy(zero_copy_only=False)]
    if m.any():
        uids.append(t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)[m])
    del t, s, d
NEED = np.sort(np.concatenate(uids))
del uids
N = len(NEED)
print(f"nodes needing semantic_text: {N:,}  ({time.time()-t0:.0f}s)", flush=True)

roles = np.full((N, K), np.iinfo(np.int32).min, dtype=np.int32)   # sentinel = empty slot
cnt = np.zeros(N, dtype=np.uint8)
print(f"roles array: {roles.nbytes/2**30:.2f} GB  K={K}", flush=True)

# ------------------------------------------------------------------ 2. relation lookup
rel_t = pq.read_table(f"{V3}/canonical/relations.parquet", columns=["rel_uid", "relation"])
r_uid = rel_t["rel_uid"].combine_chunks().to_numpy(zero_copy_only=False)
r_nm = np.array(rel_t["relation"].to_pylist(), dtype=object)
o = np.argsort(r_uid, kind="stable")
r_uid, r_nm = r_uid[o], r_nm[o]
del rel_t, o
# a role is the LAST segment of the predicate: people.marriage.spouse -> spouse
r_leaf = np.array([str(x).rsplit(".", 1)[-1].replace("_", " ") if x else "related"
                   for x in r_nm] + ["related"], dtype=object)
MISS = len(r_uid)      # sentinel row in r_leaf for a rel_uid absent from the relation table
print(f"relations: {len(r_uid):,}  ({time.time()-t0:.0f}s)", flush=True)

def take(side, rel_arr, incoming):
    """Record up to K roles for the need-nodes on this side of the batch.

    Edges are partitioned by src, so ALL of a mediator's outgoing edges arrive in the same batch.
    Keeping one row per node per batch would therefore give every CVT exactly one role, which is
    the whole point of the pass. Rows are ranked within their node group instead, so a node can
    fill several slots from a single batch.
    """
    j = np.searchsorted(NEED, side)
    np.clip(j, 0, N - 1, out=j)
    hit = NEED[j] == side
    idx = np.flatnonzero(hit)
    if not idx.size:
        return 0
    jj = j[idx]
    free = cnt[jj] < K
    idx, jj = idx[free], jj[free]
    if not idx.size:
        return 0
    order = np.argsort(jj, kind="stable")          # stable: file order decides which roles win
    idx, jj = idx[order], jj[order]
    newg = np.empty(len(jj), dtype=bool)
    newg[0] = True
    newg[1:] = jj[1:] != jj[:-1]
    starts = np.flatnonzero(newg)
    rank = np.arange(len(jj), dtype=np.int64) - np.repeat(starts, np.diff(np.append(starts, len(jj))))
    slot = cnt[jj].astype(np.int64) + rank
    keep = slot < K
    idx, jj, slot = idx[keep], jj[keep], slot[keep]
    if not idx.size:
        return 0
    ri = np.searchsorted(r_uid, rel_arr[idx])
    np.clip(ri, 0, len(r_uid) - 1, out=ri)
    ri = np.where(r_uid[ri] == rel_arr[idx], ri, MISS)     # never silently relabel as relation 0
    val = (-(ri.astype(np.int64) + 1) if incoming else ri.astype(np.int64)).astype(np.int32)
    roles[jj, slot] = val
    # slot rises within each group, so the group's last row carries its max; those indices are
    # unique, making this a plain assignment rather than a scattered maximum.
    last = np.empty(len(jj), dtype=bool)
    last[-1] = True
    last[:-1] = jj[:-1] != jj[1:]
    cnt[jj[last]] = (slot[last] + 1).astype(np.uint8)
    return int(idx.size)

# ------------------------------------------------------------------ 3. the 2.06B-edge pass
seen = kept = 0
for fi, fp in enumerate(EDGES):
    for b in pq.ParquetFile(fp).iter_batches(batch_size=BATCH, columns=["src", "rel", "dst"]):
        s = b.column("src").to_numpy(zero_copy_only=False)
        r = b.column("rel").to_numpy(zero_copy_only=False)
        d = b.column("dst").to_numpy(zero_copy_only=False)
        seen += len(s)
        kept += take(s, r, False)
        kept += take(d, r, True)
        del s, r, d, b
    if fi % 20 == 0 or fi == len(EDGES) - 1:
        print(f"  [{fi+1:3d}/{len(EDGES)}] edges={seen:,} roles={kept:,} "
              f"filled={int((cnt>0).sum()):,} ({time.time()-t0:.0f}s)", flush=True)
print(f"edges scanned: {seen:,}  (expected 2,062,430,072)", flush=True)

# ------------------------------------------------------------------ 4. render + write
os.makedirs(OUT, exist_ok=True)
covered = int((cnt > 0).sum())
written = 0
for fp in sorted(glob.glob(f"{OVL}/*.parquet")):
    base = os.path.basename(fp)
    t = pq.read_table(fp, columns=["node_uid", "display_name"])
    u = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    nm = t["display_name"].to_pylist()
    j = np.searchsorted(NEED, u)
    np.clip(j, 0, N - 1, out=j)
    hit = NEED[j] == u
    # null where display_name already says everything. Writing display_name again for the other
    # 232M nodes would duplicate ~8 GB of strings for no information; consumers fall back to
    # display_name when semantic_text is null. Row alignment with the overlay is still exact.
    txt = [None] * len(u)
    for i in np.flatnonzero(hit & (cnt[j] > 0)).tolist():
        k = j[i]
        out_r, in_r = [], []
        for v in roles[k, :cnt[k]].tolist():
            if v >= 0:
                out_r.append(r_leaf[v])
            else:
                in_r.append(r_leaf[-v - 1])
        parts = []
        if out_r:
            parts.append("roles: " + ", ".join(dict.fromkeys(out_r)))
        if in_r:
            parts.append("referenced as: " + ", ".join(dict.fromkeys(in_r)))
        if parts:
            txt[i] = f"{nm[i]} - " + "; ".join(parts)
    pq.write_table(pa.table({"node_uid": pa.array(u, pa.int64()),
                             "semantic_text": pa.array(txt, pa.string())}),
                   f"{OUT}/{base}", compression="zstd")
    written += len(u)
print(f"\nwrote {written:,} rows to {OUT}")

rec = {"schema": "SEMANTIC_TEXT_BUILD/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("sidecar keyed by node_uid. canonical/ is read, never written: "
                       "301,977,131 nodes / 2,062,430,072 edges unchanged."),
       "edges_scanned": seen, "edges_expected": 2062430072,
       "EDGES_COMPLETE": seen == 2062430072,
       "nodes_needing_semantic_text": N,
       "nodes_given_at_least_one_role": covered,
       "nodes_with_no_incident_edge": N - covered,
       "roles_cap_K": K,
       "rows_written": written,
       "SEMANTIC_TEXT_IS_NULL_WHERE": ("display_name is already a real name. Null means 'nothing to "
                                       "add', not 'missing'; consumers fall back to display_name. "
                                       "Rows stay aligned with the overlay shard for shard."),
       "DIRECTION_RULE": ("outgoing relations are the node's own roles; incoming relations name "
                          "the other end's role and are listed separately as 'referenced as'. "
                          "They are never blended, so the text cannot imply a role the node does "
                          "not hold."),
       "NOT_supported": ["that semantic_text is a name",
                         "that the role list is complete -- it is capped at K per node"],
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_SEMANTIC_TEXT_BUILD.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"needing={N:,}  with roles={covered:,}  without={N-covered:,}  {time.time()-t0:.0f}s")
