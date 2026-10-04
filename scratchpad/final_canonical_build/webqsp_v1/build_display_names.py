"""CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 -- a display_name for every one of the 301,977,131 nodes.

A SIDECAR keyed by node_uid. The frozen graph is not read for anything but its node table and is
never rewritten: 301,977,131 nodes / 2,062,430,072 edges and manifest hash
6672dab172cf5dde79c56685153d4e257be3527b5ee93b07514ad2f40425c5fb stay exactly as they are.

Deliberately does NOT depend on the 2.06B-edge semantic pass. Every tier here is a lookup or a
pure function of the identifier, so the zero-opaque invariant is reached first and the edge pass
only enriches semantic_text afterwards. If the long pass is interrupted, the graph is still clean.

Tier precedence for a node with no name in the frozen metadata:
    FREEBASE_DELETED_EXACT  a real type.object.name, recovered from the deleted-triples dump
    FREEBASE_KEY_EXACT      a readable /type/object/key
    STRUCTURAL_INFERRED     rendered from the node's declared type (CVTs and unnamed entities)
    STRUCTURAL_FALLBACK     nothing survives; the text carries no identity claim at all
Deleted names are preferred over keys even though the contract scores them 0.9 against 1.0: the
confidence scale measures TEMPORAL certainty, and this column is a display name, so an actual
historical name beats a current non-name identifier. Both confidences are reported unchanged.
"""
import sys, io, os, json, glob, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from uri_render import render

V3 = "data/final_canonical/freebase_v3"
OUT = f"{V3}/overlay_v1"
t0 = time.time()

SRC = ["LITERAL_SELF", "FREEBASE_CURRENT_EXACT", "FREEBASE_DELETED_EXACT", "FREEBASE_KEY_EXACT",
       "URI_DERIVED", "STRUCTURAL_INFERRED", "STRUCTURAL_FALLBACK"]
S = {n: i for i, n in enumerate(SRC)}
CONF = np.array([1.0, 1.0, 0.9, 1.0, 1.0, 0.5, 0.0], dtype="float32")
ORIG = np.array([True, True, True, False, False, False, False])
REF = ["the node is a literal; display_name is its lexical form",
       "canonical/metadata/name.parquet type.object.name",
       "_acquisition/deleted_names.parquet from deleted_freebase.tar.gz",
       "_acquisition/key_names.parquet /type/object/key, readability-filtered",
       "rule:uri_render.render over node_id",
       "rule:type_label from canonical/metadata/type.parquet",
       "rule:kind only; no type or name survives for this node"]

# ---------------------------------------------------------------- sidecar
names, name_ix = [], {}
def code(s):
    i = name_ix.get(s)
    if i is None:
        i = name_ix[s] = len(names); names.append(s)
    return i

parts = []   # (uid int64 array, code int32 array, src int8)
def add(node_ids, labels, src):
    u = np.fromiter((hash(x) for x in node_ids), dtype=np.int64, count=len(node_ids))
    c = np.fromiter((code(l) for l in labels), dtype=np.int32, count=len(labels))
    parts.append((u, c, np.full(len(u), S[src], dtype=np.int8)))
    print(f"  sidecar += {len(u):>12,}  {src}  ({time.time()-t0:.0f}s)", flush=True)

def add_parquet(path, id_col, label_col, src, prefix="", batch=2_000_000):
    """Same, streamed. to_pylist() over a whole 52M-row column materialises 52M Python strings at
    once, which is the allocator anti-pattern this project has already been bitten by twice."""
    pf = pq.ParquetFile(path)
    us, cs, n = [], [], 0
    for b in pf.iter_batches(batch_size=batch, columns=[id_col, label_col]):
        ids = b.column(id_col).to_pylist()
        labs = b.column(label_col).to_pylist()
        us.append(np.fromiter((hash(x) for x in ids), np.int64, len(ids)))
        cs.append(np.fromiter((code(prefix + l) for l in labs), np.int32, len(labs)))
        n += len(ids)
        del ids, labs, b
    u = np.concatenate(us); c = np.concatenate(cs)
    del us, cs
    parts.append((u, c, np.full(n, S[src], dtype=np.int8)))
    print(f"  sidecar += {n:>12,}  {src}  <- {os.path.basename(path)} ({time.time()-t0:.0f}s)",
          flush=True)

# deleted names first (highest precedence among the recovered tiers)
d = pq.read_table(f"{V3}/_acquisition/deleted_names.parquet")
d = d.filter(pc.equal(d["lang"], "en"))
subj = pc.replace_substring(d["subject"], "/m/", "m.")
subj = pc.replace_substring(subj, "/g/", "g.")
dt = pa.table({"node_id": subj, "nm": d["object"]}).sort_by(
    [("node_id", "ascending"), ("nm", "ascending")])
ids = dt["node_id"].combine_chunks().to_numpy(zero_copy_only=False)
keep = np.empty(len(ids), bool); keep[0] = True; keep[1:] = ids[1:] != ids[:-1]
dt = dt.filter(pa.array(keep))
add(dt["node_id"].to_pylist(), dt["nm"].to_pylist(), "FREEBASE_DELETED_EXACT")
del d, dt, ids, keep

k = pq.read_table(f"{V3}/_acquisition/key_names.parquet")
add(k["node_id"].to_pylist(), k["display_name"].to_pylist(), "FREEBASE_KEY_EXACT")
del k

add_parquet(f"{V3}/_acquisition/cvt_names.parquet", "node_id", "display_name",
            "STRUCTURAL_INFERRED")
add_parquet(f"{V3}/_acquisition/entity_type_names.parquet", "node_id", "type_label",
            "STRUCTURAL_INFERRED", prefix="Unnamed ")

sc_uid = np.concatenate([p[0] for p in parts])
sc_code = np.concatenate([p[1] for p in parts])
sc_src = np.concatenate([p[2] for p in parts])
del parts
prec = np.array([9, 9, 0, 1, 9, 2, 9], dtype=np.int8)[sc_src]     # lower wins
o = np.lexsort((prec, sc_uid))
sc_uid, sc_code, sc_src = sc_uid[o], sc_code[o], sc_src[o]
first = np.empty(len(sc_uid), bool); first[0] = True
first[1:] = sc_uid[1:] != sc_uid[:-1]
sc_uid, sc_code, sc_src = sc_uid[first], sc_code[first], sc_src[first]
del o, first, prec
NAMES = np.array(names, dtype=object)
print(f"sidecar: {len(sc_uid):,} uids, {len(names):,} distinct labels ({time.time()-t0:.0f}s)",
      flush=True)

# ---------------------------------------------------------------- per shard
os.makedirs(OUT, exist_ok=True)
tot = np.zeros(len(SRC), dtype=np.int64)
empty = naked = caught = 0
for n, fp in enumerate(sorted(glob.glob(f"{V3}/canonical/nodes/*.parquet"))):
    base = os.path.basename(fp)
    t = pq.read_table(fp, columns=["node_uid", "node_id", "kind", "display_text"])
    uid = t["node_uid"].combine_chunks().to_numpy(zero_copy_only=False)
    kindA = t["kind"].combine_chunks()
    dtxA = pc.fill_null(t["display_text"].combine_chunks(), "")
    m = len(uid)
    # vectorised for the 3 bulk cases; Python only touches the rows that actually need it
    # BLANK, not empty: Freebase contains a handful of type.object.name values that are pure
    # whitespace (m.011q6t0v is a single space). A name made of spaces is not a name, so those
    # rows must fall through to the structural tiers rather than be published as a blank.
    has_dt = pc.not_equal(pc.utf8_trim_whitespace(dtxA), "").to_numpy(zero_copy_only=False)
    is_lit = pc.equal(kindA, "LITERAL").to_numpy(zero_copy_only=False)
    is_uri = pc.equal(kindA, "EXTERNAL_URI").to_numpy(zero_copy_only=False)
    out = dtxA.to_pylist()
    src = np.full(m, S["STRUCTURAL_FALLBACK"], dtype=np.int8)
    src[has_dt] = S["FREEBASE_CURRENT_EXACT"]
    src[is_lit] = S["LITERAL_SELF"]            # a literal's display_text IS its lexical form
    nid = None
    ui = np.flatnonzero(is_uri & ~has_dt)
    if ui.size:
        nid = t["node_id"].to_pylist()
        for i in ui.tolist():
            # a few URIs are degenerate (http://www. has an empty host label) and render to
            # nothing. The raw URI is itself self-describing, so it is the honest fallback.
            out[i] = render(nid[i])[0].strip() or nid[i]
        src[ui] = S["URI_DERIVED"]
    # a literal's display IS its lexical form, but 17 literals have a lexical form that is pure
    # whitespace. They are real, distinct nodes under (lexical_form, datatype, language), so they
    # are rendered as an exact escaped description of the value rather than published blank. The
    # tier stays LITERAL_SELF: nothing is inferred, the value is known exactly and shown losslessly.
    lb = np.flatnonzero(is_lit & ~has_dt)
    for i in lb.tolist():
        v = out[i]
        out[i] = "(empty string)" if v == "" else "(whitespace: %s)" % ascii(v)[1:-1]
    todo = np.flatnonzero(~has_dt & ~is_lit & ~is_uri)
    if todo.size:
        if nid is None:
            nid = t["node_id"].to_pylist()
        q = uid[todo]
        j = np.searchsorted(sc_uid, q)
        np.clip(j, 0, len(sc_uid) - 1, out=j)
        hit = sc_uid[j] == q
        kinds = kindA.to_pylist()
        for pos, i in enumerate(todo.tolist()):
            if hit[pos]:
                out[i] = NAMES[sc_code[j[pos]]]
                src[i] = sc_src[j[pos]]
            elif kinds[i].startswith("SCHEMA"):
                out[i] = nid[i].rsplit(".", 1)[-1].replace("_", " ") or nid[i]
                src[i] = S["STRUCTURAL_INFERRED"]
            else:                       # nothing at all survives for this node
                out[i] = "Unnamed Freebase entity"
                src[i] = S["STRUCTURAL_FALLBACK"]
    arr = pa.array(out, pa.string())
    # The zero-opaque invariant is enforced HERE, by construction, instead of relying on every
    # tier above happening to emit text. Anything still blank is relabelled and demoted, so a
    # future tier that returns "" can never silently reintroduce a blank display_name.
    blank = pc.equal(pc.utf8_trim_whitespace(pc.fill_null(arr, "")), "")
    nb = int(pc.sum(pc.cast(blank, "int64")).as_py() or 0)
    if nb:
        for i in np.flatnonzero(blank.to_numpy(zero_copy_only=False)).tolist():
            out[i] = "Unnamed Freebase node"
            src[i] = S["STRUCTURAL_FALLBACK"]
        arr = pa.array(out, pa.string())
        caught += nb
    kind = kindA
    empty += int(pc.sum(pc.cast(pc.equal(pc.utf8_trim_whitespace(arr), ""), "int64")).as_py() or 0)
    # a display_name equal to the node's own MID would be an opaque node wearing a name column
    mid_kind = pc.is_in(kind, value_set=pa.array(["ENTITY_MID", "CVT_MEDIATOR"]))
    same = pc.equal(arr, t["node_id"].combine_chunks())
    naked += int(pc.sum(pc.cast(pc.and_(mid_kind, same), "int64")).as_py() or 0)
    tot += np.bincount(src, minlength=len(SRC)).astype(np.int64)
    pq.write_table(pa.table({
        "node_uid": pa.array(uid, pa.int64()),
        "display_name": arr,
        "display_name_source": pa.array([SRC[s] for s in src]).dictionary_encode(),
        "is_original_name": pa.array(ORIG[src], pa.bool_()),
        "resolution_confidence": pa.array(CONF[src], pa.float32()),
        "source_reference": pa.array([REF[s] for s in src]).dictionary_encode(),
    }), f"{OUT}/{base}", compression="zstd")
    if n % 20 == 0 or n == 261:
        print(f"  [{n+1:3d}/262] {base} {m:>9,}  total={tot.sum():,} ({time.time()-t0:.0f}s)",
              flush=True)

N = int(tot.sum())
auth = int(tot[S["LITERAL_SELF"]] + tot[S["FREEBASE_CURRENT_EXACT"]] +
           tot[S["FREEBASE_DELETED_EXACT"]] + tot[S["FREEBASE_KEY_EXACT"]])
rec = {"schema": "RESOLUTION_OVERLAY_BUILD/v1",
       "name": "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "nodes_covered": N, "nodes_expected": 301977131,
       "COVERS_EVERY_NODE": N == 301977131,
       "blank_caught_by_final_guarantee": caught,
       "DISPLAY_NAME_EMPTY": int(empty), "NAKED_MID_DISPLAY": int(naked),
       "by_source": {SRC[i]: int(tot[i]) for i in range(len(SRC))},
       "AUTHORITATIVE_SOURCE_RESOLUTION": auth,
       "AUTHORITATIVE_SOURCE_RESOLUTION_PCT": round(100 * auth / N, 4) if N else 0,
       "URI_DERIVED_PCT": round(100 * int(tot[S["URI_DERIVED"]]) / N, 4) if N else 0,
       "STRUCTURAL_PCT": round(100 * int(tot[S["STRUCTURAL_INFERRED"]] +
                                         tot[S["STRUCTURAL_FALLBACK"]]) / N, 4) if N else 0,
       "honesty_rule": ("zero opaque nodes is an OPERATIONAL claim that CRAG never shows a bare "
                        "MID. It is NOT a claim that every original name was recovered. The "
                        "authoritative and structural figures above must always be reported "
                        "together."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_RESOLUTION_OVERLAY_BUILD.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nnodes={N:,} (expected 301,977,131)  empty={empty:,}  naked_mid={naked:,}")
for i, s in enumerate(SRC):
    print(f"  {tot[i]:>12,}  {100*tot[i]/N:6.3f}%  {s}")
print(f"authoritative={auth:,} ({100*auth/N:.3f}%)   {time.time()-t0:.0f}s")
