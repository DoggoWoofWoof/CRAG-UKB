"""semantic_kind: what each residual node IS, separately from whether we have found its name.

    PYTHONHASHSEED=0 python .../semantic_kind.py

WHY A NEW FIELD AND NOT A CORRECTION OF node_kind
  node_kind is frozen inside CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 (manifest hash
  25b734fe...b27d865) and is not touched here. It is also a structural inference: a node was called
  CVT_MEDIATOR because of the shape of the relations around it. That inference put 17,809,849 nodes
  in ENTITY_MID, and the source itself disagrees about millions of them -- the MusicBrainz NGS
  loader wrote "..._cvt_id_<uuid>" into their keys, and Freebase's own schema marks 2,211 types
  mediator=true. semantic_kind records what the SOURCE says, alongside the frozen field, so both
  remain readable and neither overwrites the other.

WHY THIS IS NOT "GIVING UP ON A NAME"
  A node whose key is "mb_ngs:artist_contribution_cvt_id_<uuid>::<uuid>" is a reified relationship.
  There is no proper noun to recover because there never was one. Separating those from genuine
  entity-like nodes is the difference between an honest residue and an unbounded search. The two
  populations the recensus needs are TRUE_NAMED_ENTITY_RESIDUE and
  SOURCE_ATTESTED_NAMELESS_INTERNAL_OBJECTS, and this file is what makes them computable.

EVIDENCE IS GRADED, AND THE GRADE IS STORED
  SOURCE_KEY       the node's own /type/object/key says it     (loader-written, per node)
  SOURCE_DECLARED  Freebase's schema declares its type a mediator, or declares it not one
  INFERRED         our structural node_kind, or a name we recovered -- weakest, and marked as such
  A caller that wants only the defensible core can filter to the first two grades; the counts are
  reported both ways.

WHAT IS DELIBERATELY LEFT UNDECIDED
  ENTITY_CANDIDATE is not a claim that a node is a named entity. It means no source statement was
  found either way. That bucket is the input to the remaining historical recovery, and its type
  census is printed so the large undeclared types can be ruled on explicitly rather than guessed at.

Reads canonical/ and _acquisition/. Writes only under _acquisition/.
"""
import sys, io, os, re, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
ESC = re.compile(r"\$([0-9A-Fa-f]{4})")
VAR = re.compile(r"_var_(.*)$")
HEX = re.compile(r"^[0-9a-f]{6,}\|?_?$")
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

# ------------------------------------------------------------------ the vocabulary
KIND = ["UNSET", "WORD_SENSE", "CVT_MEDIATOR", "LOAD_ARTIFACT", "INTERNAL_RECORD",
        "EXTERNAL_RESOURCE", "NAMED_ENTITY", "ENTITY_CANDIDATE"]
K = {n: i for i, n in enumerate(KIND)}

# (evidence, grade, kind) in precedence order. The node's own key outranks the schema, because the
# key was written for THAT node while the schema hint is a property of a whole type; both outrank
# anything we inferred.
EVID = [
    ("UNSET",                      "",                "UNSET"),
    ("KEY_WORDNET_SENSEKEY",       "SOURCE_KEY",      "WORD_SENSE"),
    ("KEY_DECLARED_CVT",           "SOURCE_KEY",      "CVT_MEDIATOR"),
    ("KEY_DECLARED_LOAD",          "SOURCE_KEY",      "LOAD_ARTIFACT"),
    ("KEY_DECLARED_SCHEMA",        "SOURCE_KEY",      "INTERNAL_RECORD"),
    ("KEY_AUTHORITY_ID",           "SOURCE_KEY",      "NAMED_ENTITY"),
    ("KEY_URI_REFERENCE",          "SOURCE_KEY",      "EXTERNAL_RESOURCE"),
    ("KEY_TRUNCATED_TITLE_REFUSED", "SOURCE_KEY",     "ENTITY_CANDIDATE"),
    ("TYPE_DECLARED_MEDIATOR",     "SOURCE_DECLARED", "CVT_MEDIATOR"),
    ("TYPE_DECLARED_NOT_MEDIATOR", "SOURCE_DECLARED", "NAMED_ENTITY"),
    ("NAME_RECOVERED",             "INFERRED",        "NAMED_ENTITY"),
    ("STRUCTURAL_CVT_INFERENCE",   "INFERRED",        "CVT_MEDIATOR"),
    ("NO_DECLARATION",             "INFERRED",        "ENTITY_CANDIDATE"),
]
E = {e[0]: i for i, e in enumerate(EVID)}

# key bucket -> evidence code. Buckets are the ones residue_key_census.py established.
BUCKET_EVID = {
    "wordnet_sensekey":            "KEY_WORDNET_SENSEKEY",
    "freeq_musicbrainz_cvt":       "KEY_DECLARED_CVT",
    "freeq_declared_dummy_cvt":    "KEY_DECLARED_CVT",
    "freeq_mid_pair":              "KEY_DECLARED_CVT",
    "freeq_hex_blob":              "KEY_DECLARED_LOAD",
    "freeq_literal_value":         "KEY_DECLARED_LOAD",
    "base_medicaldrugs_load":      "KEY_DECLARED_LOAD",
    "base_wsjtopics_articleid":    "KEY_DECLARED_LOAD",
    "base_fbontology_metaschema":  "KEY_DECLARED_SCHEMA",
    "uri_reference":               "KEY_URI_REFERENCE",
    "freeq_other_text":            "KEY_TRUNCATED_TITLE_REFUSED",
}
# WHY freeq_mid_pair COUNTS AS A DECLARED CVT: the loader variable is two MIDs and a date, e.g.
# /m/02h40lc/m/0dw4b/1998. That is the definition of a reified binary relation, written by the
# loader itself. It is not a truncated name.
# WHY freeq_other_text IS ENTITY_CANDIDATE AND NOT A NAME: "_var_pringfield (medley): ... / Gracie
# Fi" is missing the leading S and is cut at a key-length limit; "_var_kit #1" is "Skit #1". The
# node looks like a real track, so it stays a candidate, but the string is mutilated at both ends
# and is refused as a name -- exactly the standing ruling on the 58,144.


def dec(s):
    return ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def classify(k, d):
    """Identical to residue_key_census.classify. Duplicated deliberately: that script is a census
    and this one is an artifact builder, and a shared import would let a later edit to one silently
    change the other."""
    if k.startswith("/user/jamie/wordnet/sensekey"):
        return "wordnet_sensekey" if re.match(r"^/user/jamie/wordnet/sensekey/'(.+?)%", d) \
            else "wordnet_unparsed"
    if k.startswith("/authority/"):
        return "authority:" + k.split("/")[2]
    if k.startswith("/wikipedia/"):
        return "wikipedia:" + k.split("/")[2]
    if k.startswith("/uri/"):
        return "uri_reference"
    if k.startswith("/dataworld/freeq"):
        g = VAR.search(d)
        if not g:
            return "freeq_other"
        v = g.group(1)
        if v.startswith("dummy_cvt"):
            return "freeq_declared_dummy_cvt"
        if "_cvt_id_" in v or v.startswith("mb_ngs:"):
            return "freeq_musicbrainz_cvt"
        if v.startswith("/m/") or v.startswith("/g/"):
            return "freeq_mid_pair"
        if HEX.match(v):
            return "freeq_hex_blob"
        if re.fullmatch(r"[\d.]+", v):
            return "freeq_literal_value"
        return "freeq_other_text"
    if k.startswith("/base/medicaldrugs"):
        return "base_medicaldrugs_load"
    if k.startswith("/base/fbontology"):
        return "base_fbontology_metaschema"
    if k.startswith("/base/wsjtopics"):
        return "base_wsjtopics_articleid"
    if k.startswith("/user/"):
        return "other_user_namespace"
    return "other:" + "/".join(k.split("/")[:3])


# ------------------------------------------------------------------ population
z = np.load(f"{ACQ}/_unresolved_population.npz")
U, KC = z["U"], z["KC"]                     # sorted node_uid, frozen node_kind code
N = len(U)
print(f"residue nodes: {N:,}   frozen kinds: {dict(collections.Counter(KC.tolist()))}", flush=True)

ev = np.zeros(N, np.uint8)                  # index into EVID; 0 = nothing seen yet


def claim(idx, code):
    """First (strongest) evidence wins; EVID is in precedence order, so a lower index outranks."""
    c = E[code]
    cur = ev[idx]
    take = (cur == 0) | (c < cur)
    ev[idx] = np.where(take, c, cur)


# ------------------------------------------------------------------ 1. the nodes' own keys
kb = collections.Counter()
for fp in sorted(glob.glob(f"{ACQ}/residue_keys/*.parquet")):
    t = pq.read_table(fp)
    ids, ks = t["node_id"].to_pylist(), t["key"].to_pylist()
    us = t["node_uid"].to_pylist() if "node_uid" in t.schema.names else [hash(x) for x in ids]
    pos, codes = [], []
    for nid, k, u in zip(ids, ks, us):
        j = int(np.searchsorted(U, u))
        if j >= N or U[j] != u:
            continue
        b = classify(k, dec(k))
        kb[b] += 1
        c = BUCKET_EVID.get(b)
        if c is None and (b.startswith("authority:") or b.startswith("wikipedia:")):
            c = "KEY_AUTHORITY_ID"
        if c:
            pos.append(j)
            codes.append(E[c])
    if pos:
        p, c = np.array(pos), np.array(codes, np.uint8)
        cur = ev[p]
        ev[p] = np.where((cur == 0) | (c < cur), c, cur)
print(f"key evidence applied; {int((ev > 0).sum()):,} nodes claimed  ({time.time()-t0:.0f}s)",
      flush=True)

# ------------------------------------------------------------------ 2. Freebase's own type hints
mj = json.load(io.open(f"{ACQ}/_mediator_type_paths.json", encoding="utf-8"))
MED, NOTMED = set(mj["mediator_true_paths"]), set(mj["mediator_false_paths"])
print(f"schema declares {len(MED):,} mediator types and {len(NOTMED):,} non-mediator types",
      flush=True)

has_type = np.zeros(N, bool)
med_hit = np.zeros(N, bool)
notmed_hit = np.zeros(N, bool)
undeclared_types = collections.Counter()
for fp in sorted(glob.glob(f"{ACQ}/residue_types/*.parquet")):
    t = pq.read_table(fp, columns=["node_uid", "type"])
    uu = np.asarray(t["node_uid"], dtype=np.int64)
    ty = t["type"].combine_chunks().cast(pa.string()).to_pylist()
    j = np.searchsorted(U, uu)
    ok = (j < N) & (U[np.minimum(j, N - 1)] == uu)
    j = j[ok]
    ty = [ty[i] for i in np.flatnonzero(ok)]
    has_type[j] = True
    m = np.array([x in MED for x in ty])
    nm = np.array([x in NOTMED for x in ty])
    if m.any():
        med_hit[j[m]] = True
    if nm.any():
        notmed_hit[j[nm]] = True
    for i, x in enumerate(ty):
        if not m[i] and not nm[i]:
            undeclared_types[x] += 1
print(f"typed residue nodes: {int(has_type.sum()):,}   declared-mediator: {int(med_hit.sum()):,}   "
      f"declared-not-mediator: {int(notmed_hit.sum()):,}  ({time.time()-t0:.0f}s)", flush=True)

claim(np.flatnonzero(med_hit), "TYPE_DECLARED_MEDIATOR")
claim(np.flatnonzero(notmed_hit & ~med_hit), "TYPE_DECLARED_NOT_MEDIATOR")

# ------------------------------------------------------------------ 3. names we have recovered
NAME_SRC = ["cascade_names.parquet", "external_authority_names.parquet",
            "external_authority_mediawiki.parquet", "external_authority_musicbrainz.parquet",
            "external_authority_small.parquet", "uri_authority_names.parquet"] + \
    [os.path.basename(p) for p in sorted(glob.glob(f"{ACQ}/uri_page_*.parquet"))]
named = np.zeros(N, bool)
for fn in NAME_SRC:
    fp = f"{ACQ}/{fn}"
    if not os.path.exists(fp):
        continue
    uu = np.asarray(pq.read_table(fp, columns=["node_uid"])["node_uid"], dtype=np.int64)
    j = np.searchsorted(U, uu)
    ok = (j < N) & (U[np.minimum(j, N - 1)] == uu)
    named[j[ok]] = True
    print(f"  names from {fn}: {len(uu):,} rows", flush=True)
print(f"nodes with a recovered name: {int(named.sum()):,}", flush=True)
claim(np.flatnonzero(named), "NAME_RECOVERED")

# ------------------------------------------------------------------ 4. our own structural label
claim(np.flatnonzero((ev == 0) & (KC == 1)), "STRUCTURAL_CVT_INFERENCE")
claim(np.flatnonzero(ev == 0), "NO_DECLARATION")

# ------------------------------------------------------------------ write
kind = np.array([K[EVID[i][2]] for i in range(len(EVID))], np.uint8)[ev]
grade = np.array([EVID[i][1] for i in range(len(EVID))], object)[ev]
pq.write_table(pa.table({
    "node_uid": pa.array(U, pa.int64()),
    "node_kind_frozen": pa.array(np.where(KC == 0, "ENTITY_MID",
                                          np.where(KC == 1, "CVT_MEDIATOR", "OTHER")).tolist()
                                 ).dictionary_encode(),
    "semantic_kind": pa.array([KIND[x] for x in kind]).dictionary_encode(),
    "evidence": pa.array([EVID[i][0] for i in ev]).dictionary_encode(),
    "evidence_grade": pa.array(grade.tolist()).dictionary_encode(),
    "has_recovered_name": pa.array(named, pa.bool_())}),
    f"{ACQ}/semantic_kind.parquet", compression="zstd")

# ------------------------------------------------------------------ report
kc = collections.Counter(KIND[x] for x in kind)
ec = collections.Counter(EVID[i][0] for i in ev)
gc = collections.Counter(grade.tolist())
ent = KC == 0
kc_ent = collections.Counter(KIND[x] for x in kind[ent])

INTERNAL_KINDS = {"CVT_MEDIATOR", "LOAD_ARTIFACT", "INTERNAL_RECORD"}
src_att = int(sum(1 for i, x in zip(ev, kind)
                  if KIND[x] in INTERNAL_KINDS and EVID[i][1] != "INFERRED"))
true_res = int(sum(1 for x in kind if KIND[x] in ("NAMED_ENTITY", "ENTITY_CANDIDATE")))

# what is left that nothing declares -- printed so the big types can be ruled on, not guessed
is_cand = kind == K["ENTITY_CANDIDATE"]
cand_types = collections.Counter()
for fp in sorted(glob.glob(f"{ACQ}/residue_types/*.parquet")):
    t = pq.read_table(fp, columns=["node_uid", "type"])
    uu = np.asarray(t["node_uid"], dtype=np.int64)
    ty = t["type"].combine_chunks().cast(pa.string()).to_pylist()
    j = np.searchsorted(U, uu)
    ok = (j < N) & (U[np.minimum(j, N - 1)] == uu)
    j = j[ok]
    ty = [ty[i] for i in np.flatnonzero(ok)]
    keep = is_cand[j]
    for i in np.flatnonzero(keep):
        cand_types[ty[i]] += 1
cand_untyped = int((is_cand & ~has_type).sum())

rec = {"schema": "SEMANTIC_KIND_OVERLAY/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("adds a field, mutates none. node_kind and "
                       "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 (manifest hash 25b734fe9acf2ca7481"
                       "4cf9f3444757636f19100305daa28b3ec19a2fb27d865) are untouched."),
       "population": "the 69,777,967 nodes with no name in the frozen overlay",
       "OUT_OF_SCOPE": ("nodes outside the residue already carry a name in the frozen graph; this "
                        "artifact answers a question about what is LEFT, so it does not restate "
                        "them."),
       "vocabulary": KIND[1:],
       "precedence": [{"evidence": e, "grade": g, "semantic_kind": k} for e, g, k in EVID[1:]],
       "by_semantic_kind": dict(kc.most_common()),
       "by_semantic_kind_entity_mid_only": dict(kc_ent.most_common()),
       "by_evidence": dict(ec.most_common()),
       "by_grade": dict(gc.most_common()),
       "SOURCE_ATTESTED_NAMELESS_INTERNAL_OBJECTS": src_att,
       "TRUE_NAMED_ENTITY_RESIDUE": true_res,
       "schema_mediator_types": len(MED),
       "schema_non_mediator_types": len(NOTMED),
       "typed_residue_nodes": int(has_type.sum()),
       "undeclared_types_top": dict(undeclared_types.most_common(40)),
       "entity_candidate_types_top": dict(cand_types.most_common(40)),
       "entity_candidate_total": int(is_cand.sum()),
       "entity_candidate_untyped": cand_untyped,
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_SEMANTIC_KIND.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nsemantic_kind over {N:,} residual nodes")
for k, v in kc.most_common():
    print(f"  {v:>12,}  {k}")
print("\nENTITY_MID residue only (17,809,849):")
for k, v in kc_ent.most_common():
    print(f"  {v:>12,}  {k}")
print("\nby evidence:")
for k, v in ec.most_common():
    print(f"  {v:>12,}  {k:<30} {EVID[E[k]][1]}")
print(f"\nSOURCE_ATTESTED_NAMELESS_INTERNAL_OBJECTS: {src_att:,}")
print(f"TRUE_NAMED_ENTITY_RESIDUE:                 {true_res:,}")
print("\ntop types among ENTITY_CANDIDATE (nothing declares these -- rule on them explicitly):")
for k, v in cand_types.most_common(25):
    print(f"  {v:>12,}  {k}")
print(f"\n{time.time()-t0:.0f}s")
