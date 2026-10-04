"""Materialise what the key-namespace census found, in ONE more pass over key.parquet.

    PYTHONHASHSEED=0 python .../key_evidence.py

WHY A SECOND PASS AT ALL
  The census counted namespaces with bitmaps and threw the node lists away, so it can say "92,350
  hunt nodes carry a FreeQ key" but cannot say WHICH.  Every decoder built from here would otherwise
  rescan 143,981,520 rows for its own few thousand nodes.  This pass writes the node lists once.

WHAT THE CENSUS ACTUALLY DECIDED (counts, not impressions)
  WIKIPEDIA_TITLE IS DEAD AS A TIER.  /wikipedia/en reaches exactly 1 residue node ('Sky_99.5') and
  /wikipedia/en_title exactly 1; every Wikipedia TITLE namespace together reaches about 15.  The
  cascade shows WIKIPEDIA_TITLE with no source not because a decoder was missing but because a node
  with a Wikipedia article kept its name.  No decoder is worth writing.  This file records that.

  KEY.PARQUET IS EXHAUSTED AS A NAME SOURCE.  Its three largest residue namespaces are all opaque
  machine keys: /dataworld/freeq (5,445,830 nodes, values shaped job_<uuid>_var_dummy_cvt_...),
  /source/corpwatch (208,314, values rel_<a>_<b>, i.e. relationship mediators, 160 in hunt) and
  /base/medicaldrugs/fda_load (190,343, values articlekey_<hash>, 0 in hunt).  Outside FreeQ the
  whole residue key space offers roughly 12,000 hunt nodes of exact-identifier material, and the
  largest decodable one -- MusicBrainz, about 3,380 hunt nodes -- is already running.

  So the one LARGE thing this census found is a SUBTRACTION, not an addition.

THE SUBTRACTION, AND THE CHECK AGAINST IT
  A key of the form /dataworld/freeq/job_<uuid>_var_dummy_cvt_/m/0zbdscn/m/0zb3yvm is a FreeQ job
  variable: a row a loading job created to hold a compound value.  92,350 such nodes are in the name
  hunt.  Proposing them for a terminal bucket is worth as much as finding names, because the stated
  end condition is that the remainder consists only of objects that demonstrably never had names.

  But EMPIRICALLY_NAMELESS has already been falsified once in this campaign -- the 2010 topic dump
  named four nodes graded nameless -- so this file refuses to make the proposal without first testing
  it the same way: every FreeQ hunt node is intersected against every name recovered so far.  If the
  2010 dumps name a node whose key says 'dummy_cvt', the proposal is wrong and the count says so.
  NOTHING IS APPLIED HERE.  Reclassification is a new overlay record with its own hash, never an edit.

OUTPUT (append-only, all under _acquisition/)
  _key_evidence/part_*.parquet     node_uid, key, namespace, in_hunt  for the small decodable spaces
  _freeq_hunt.parquet              node_uid, freeq_key, var_shape     for the terminal proposal
  V3_KEY_EVIDENCE.json
"""
import sys, io, os, re, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
EVI = f"{ACQ}/_key_evidence"
os.makedirs(EVI, exist_ok=True)
t0 = time.time()

FREEQ = "/dataworld/freeq/"
# every namespace the census showed to be BOTH small and reachable by exact identifier, plus the
# handful of Wikipedia spaces so the negative result is recorded with its node ids rather than
# asserted. Cheap: all of these together are under 400k rows.
CAPTURE = ("/lang/", "/wikipedia/", "/authority/", "/tag/ukguardian/", "/user/jamie/gx/id/",
           "/base/wsjtopics/", "/freebase/relevance/", "/base/fbontology/metaschema/",
           "/user/ovguide/", "/base/dspl/", "/user/mysqlguru/", "/source/corpwatch/")

ESC = re.compile(r"\$([0-9A-Fa-f]{4})")


def unesc(s):
    """Freebase key escaping: $002F is '/', $002E is '.', $0410 is a Cyrillic A."""
    return ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


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

# every name recovered so far, so the terminal proposal is tested and not merely asserted
namedirs = ["fb2010_names", "fb2010_quad_names", "wex_names", "fb2010_tsv_names"]
npaths = []
for d in namedirs:
    npaths += sorted(glob.glob(f"{ACQ}/{d}/*.parquet"))
NAMED = (np.sort(pa.concat_tables([pq.read_table(x, columns=["node_uid"]) for x in npaths])
                 ["node_uid"].to_numpy()) if npaths else np.zeros(0, np.int64))
print(f"names recovered so far, for the falsification check: {len(NAMED):,} rows "
      f"from {len(npaths)} parts", flush=True)

pf = pq.ParquetFile(f"{V3}/canonical/metadata/key.parquet")
NRG = pf.metadata.num_row_groups
ev = {"uid": [], "key": [], "ns": [], "hunt": []}
part = 0
fq_uid, fq_key, fq_shape = [], [], []
shape_all = collections.Counter()
shape_hunt = collections.Counter()
n_res = n_freeq = n_cap = 0


def flush(force=False):
    global part, ev
    if not ev["uid"] or (not force and len(ev["uid"]) < 400000):
        return
    pq.write_table(pa.table({"node_uid": pa.array(ev["uid"], pa.int64()),
                             "key": pa.array(ev["key"]),
                             "namespace": pa.array(ev["ns"]),
                             "in_hunt": pa.array(ev["hunt"], pa.bool_())}),
                   f"{EVI}/part_{part:05d}.parquet", compression="zstd")
    part += 1
    ev = {"uid": [], "key": [], "ns": [], "hunt": []}


def shape_of(k):
    """job_<uuid>_var_<name> -> the leading alphabetic token of <name>; that token is the FreeQ
    variable's role, which is what says whether the row is a dummy CVT or a real payload."""
    i = k.find("_var_")
    if i < 0:
        return "NO_VAR_SEGMENT"
    v = k[i + 5:]
    m = re.match(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)*", v)
    return m.group(0)[:48] if m else "NON_ALPHA_VAR"


for g in range(NRG):
    t = pf.read_row_group(g, columns=["subject", "key"])
    sub = t["subject"]
    sub = pc.cast(sub, pa.string()) if pa.types.is_dictionary(sub.type) else sub
    sub = sub.to_pylist()
    subj = np.fromiter((hash(x) for x in sub), np.int64, t.num_rows)
    pos = np.clip(np.searchsorted(U, subj), 0, len(U) - 1)
    hit = U[pos] == subj
    if not hit.any():
        continue
    keys = t["key"]
    keys = pc.cast(keys, pa.string()) if pa.types.is_dictionary(keys.type) else keys
    keys = keys.to_pylist()
    for i in np.flatnonzero(hit).tolist():
        k = keys[i]
        n_res += 1
        u = int(subj[i])
        inh = bool(hunt_mask[pos[i]])
        if k.startswith(FREEQ):
            n_freeq += 1
            sh = shape_of(k[len(FREEQ):])
            shape_all[sh] += 1
            if inh:
                shape_hunt[sh] += 1
                fq_uid.append(u); fq_key.append(k[:300]); fq_shape.append(sh)
            continue
        if k.startswith(CAPTURE):
            n_cap += 1
            ev["uid"].append(u); ev["key"].append(k[:400])
            ev["ns"].append(k.rsplit("/", 1)[0]); ev["hunt"].append(inh)
    flush()
    if g % 25 == 0:
        print(f"  rg {g}/{NRG}  residue rows {n_res:,}  freeq {n_freeq:,}  captured {n_cap:,}  "
              f"({time.time()-t0:.0f}s)", flush=True)
flush(True)

fq = np.array(fq_uid, np.int64) if fq_uid else np.zeros(0, np.int64)
fq_nodes = np.unique(fq)
if len(fq_nodes):
    pq.write_table(pa.table({"node_uid": pa.array(fq, pa.int64()),
                             "freeq_key": pa.array(fq_key),
                             "var_shape": pa.array(fq_shape)}),
                   f"{ACQ}/_freeq_hunt.parquet", compression="zstd")

# THE CHECK: does anything already recovered name a node whose key says it is a job variable?
if len(NAMED) and len(fq_nodes):
    q = np.clip(np.searchsorted(NAMED, fq_nodes), 0, len(NAMED) - 1)
    contra = fq_nodes[NAMED[q] == fq_nodes]
else:
    contra = np.zeros(0, np.int64)

dummy = sum(v for k, v in shape_hunt.items() if k.startswith("dummy"))
rec = {"schema": "KEY_EVIDENCE/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/; frozen graph and frozen overlay untouched.",
       "WIKIPEDIA_TITLE_TIER": {
           "VERDICT": "REFUTED AS A TIER -- no decoder should be written",
           "evidence": "/wikipedia/en reaches 1 residue node, /wikipedia/en_title 1, all Wikipedia "
                       "TITLE namespaces together about 15 (V3_KEY_NAMESPACE_CENSUS).",
           "reading": "a node holding a Wikipedia title key kept its /type/object/name, so the "
                      "unnamed residue does not hold these keys. The empty tier was never a "
                      "missing decoder.",
           "still_live_and_unrelated": "the 56,285 wikipedia_en_page_id values captured from "
                                       "ARCHIVED PAGES are a different source and are unaffected."},
       "KEY_PARQUET_AS_A_NAME_SOURCE": {
           "VERDICT": "EXHAUSTED",
           "largest_three_residue_namespaces_are_opaque": {
               "/dataworld/freeq": "5,445,830 nodes, values job_<uuid>_var_<name>",
               "/source/corpwatch": "208,314 nodes, values rel_<a>_<b> (relationship mediators), "
                                    "160 in hunt",
               "/base/medicaldrugs/fda_load": "190,343 nodes, values articlekey_<hash>, 0 in hunt"},
           "decodable_and_in_hunt_total": "roughly 12,000 nodes, largest of which (MusicBrainz, "
                                          "about 3,380) is already running"},
       "key_rows_on_residue": n_res,
       "freeq_rows_on_residue": n_freeq,
       "captured_rows": n_cap,
       "capture_prefixes": list(CAPTURE),
       "evidence_parts": part,
       "FREEQ_TERMINAL_PROPOSAL": {
           "hunt_nodes_with_a_freeq_job_key": int(len(fq_nodes)),
           "of_those_var_shape_starts_with_dummy": int(dummy),
           "STATUS": "PROPOSED, NOT APPLIED. Grade assignment is not made here: a FreeQ job key is "
                     "machine-origin evidence, which is not the same act as Freebase declaring the "
                     "object nameless, and the three grades stay separate.",
           "FALSIFICATION_CHECK": {
               "tested_against": namedirs,
               "names_available_for_the_test": int(len(NAMED)),
               "freeq_hunt_nodes_that_ARE_named_by_a_recovered_source": int(len(contra)),
               "MEANING": "any non-zero count falsifies the proposal for those nodes exactly as the "
                          "2010 dump falsified four EMPIRICALLY_NAMELESS grades."}},
       "top_freeq_var_shapes_on_hunt_nodes": [{"var_shape": k, "hunt_rows": v,
                                               "all_residue_rows": shape_all[k]}
                                              for k, v in shape_hunt.most_common(30)],
       "top_freeq_var_shapes_all_residue": [{"var_shape": k, "rows": v}
                                            for k, v in shape_all.most_common(20)],
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_KEY_EVIDENCE.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items()
                  if k not in ("top_freeq_var_shapes_on_hunt_nodes",
                               "top_freeq_var_shapes_all_residue", "capture_prefixes")},
                 indent=1, ensure_ascii=False))
print("\nfreeq var shapes on HUNT nodes:")
for d in rec["top_freeq_var_shapes_on_hunt_nodes"][:18]:
    print(f"  {d['var_shape'][:44]:<44} hunt {d['hunt_rows']:>8,}  all {d['all_residue_rows']:>9,}")
if len(contra):
    print(f"\nFALSIFIED for {len(contra):,} nodes -- sample uids: {contra[:8].tolist()}")
