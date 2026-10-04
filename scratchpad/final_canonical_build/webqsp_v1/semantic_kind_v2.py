"""semantic_kind v2: the three nameless grades, the inferred-CVT split, and the hunt partition.

    PYTHONHASHSEED=0 python .../semantic_kind_v2.py

WHY A V2 RATHER THAN AN EDIT
  V1 (semantic_kind.parquet, V3_SEMANTIC_KIND.json) is the record of what was known when the
  50,363,784 / 16,943,048 split was first computed. It is left as written. Everything here is a new
  artifact with its own record, so the two can be compared and the earlier one cited as it was.

THREE THINGS CHANGE, EACH ON ITS OWN AXIS
  1. NAMELESS EVIDENCE IS GRADED, AND ONLY ONE GRADE COUNTS AS HARD.
       SOURCE_DECLARED_NAMELESS  Freebase's schema calls the type a mediator, or the node's own key
                                 says it is a load record. This is the only grade in the hard count.
       EMPIRICALLY_NAMELESS      every type the node carries has ZERO named instances across all
                                 302M nodes of the snapshot. Measured, not declared: nothing in the
                                 source says a common.document cannot have a name, we have only
                                 observed that none does. Reported separately from the hard count.
       INFERRED_NAMELESS         our own structural CVT inference. Weakest; held apart.
     The type-level measurement is named_rate from _type_named_rate.json, computed by scanning
     type.parquet once against the residue: total instances, unnamed instances, and their ratio,
     for every type in the graph.

  2. THE 2,179,025 STRUCTURALLY INFERRED MEDIATORS GET THEIR OWN VALUE.
     semantic_kind = CVT_MEDIATOR_INFERRED, evidence_grade = INFERRED. They are not upgraded into
     CVT_MEDIATOR until a key or a schema hint says so. Two P646 matches on nodes this inference
     had called CVTs turned out to be people; that is why the separation is kept.

  3. THE RECOVERY POPULATION IS PARTITIONED SO EXPENSIVE WORK GOES WHERE NAMES EXIST.
     recovery_class is a PRIORITISATION label, derived from which Freebase domains the node's
     types belong to. It is not a claim about the node. LIKELY_PERSON is "carries people.person";
     it does not assert the node is a person. Its job is ordering: a class whose types exhibit
     proper names elsewhere in Freebase is worked before millions of type.content_import records.

WHAT THIS RECOMPUTES
  TRUE_NAME_RECOVERY_RESIDUE_V2: the V1 figure minus every node since named by MusicBrainz, Discogs,
  the archive tiers, or any other source that has landed. The source rows read are stamped into the
  record, so a later re-run after more sources land is distinguishable from this one.

Reads _acquisition/ and canonical/metadata/. Writes semantic_kind_v2.parquet and a new record only.
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

# ---------------------------------------------------------------- v1, the starting point
v1 = pq.read_table(f"{ACQ}/semantic_kind.parquet")
U = np.asarray(v1["node_uid"], dtype=np.int64)
N = len(U)
sk1 = v1["semantic_kind"].combine_chunks()
ev1 = v1["evidence"].combine_chunks()
kf1 = v1["node_kind_frozen"].combine_chunks()
# dictionary arrays -> code arrays + vocab, so 69.78M rows never become Python objects
def codes(arr):
    d = arr.dictionary.to_pylist()
    return np.asarray(arr.indices, dtype=np.int32), d
sk_c, SK = codes(sk1)
ev_c, EV = codes(ev1)
kf_c, KF = codes(kf1)
print(f"v1 loaded: {N:,} rows  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- the type-level measurement
rate = json.load(io.open(f"{ACQ}/_type_named_rate.json", encoding="utf-8"))
print(f"type named-rate table: {len(rate):,} types", flush=True)
zero_named = {k for k, v in rate.items() if v["named"] == 0}
near = {k for k, v in rate.items() if 0 < v["named_rate"] < 1e-4}
print(f"  types with zero named instances graph-wide: {len(zero_named):,} "
      f"(covering {sum(rate[k]['total'] for k in zero_named):,} type rows)", flush=True)
print(f"  types with 0 < named_rate < 1e-4: {len(near):,}", flush=True)

# domain -> prioritisation class. Freebase's own domain membership, not our reading of type names.
PERSON_TYPES = {"people.person", "people.deceased_person"}
PLACE_DOMAINS = {"location", "geography"}
MEDIA_DOMAINS = {"film", "tv", "music", "broadcast", "media_common"}
WORK_DOMAINS = {"book", "visual_art", "theater", "comic_books", "opera", "cvg", "computer",
                "fictional_universe"}
CLASS = ["LIKELY_REAL_ENTITY", "LIKELY_CREATIVE_WORK", "LIKELY_MEDIA_ENTITY", "LIKELY_PLACE",
         "LIKELY_PERSON"]                       # index = priority; higher wins on a multi-typed node


def type_class(tp):
    if tp in PERSON_TYPES:
        return 4
    dom = tp.split(".", 1)[0]
    if dom in PLACE_DOMAINS:
        return 3
    if dom in MEDIA_DOMAINS:
        return 2
    if dom in WORK_DOMAINS:
        return 1
    return 0


tcls = {k: type_class(k) for k in rate}
trate = {k: v["named_rate"] for k, v in rate.items()}

has_type = np.zeros(N, bool)
all_zero = np.ones(N, bool)                     # stays True only if every type seen is zero-named
max_rate = np.zeros(N, np.float32)
best_cls = np.zeros(N, np.int8)
for i, fp in enumerate(sorted(glob.glob(f"{ACQ}/residue_types/*.parquet"))):
    t = pq.read_table(fp, columns=["node_uid", "type"])
    uu = np.asarray(t["node_uid"], dtype=np.int64)
    ty = t["type"].combine_chunks().cast(pa.string()).to_pylist()
    j = np.searchsorted(U, uu)
    ok = (j < N) & (U[np.minimum(j, N - 1)] == uu)
    j = j[ok]
    ty = [ty[x] for x in np.flatnonzero(ok)]
    r = np.fromiter((trate.get(x, 0.0) for x in ty), np.float32, len(ty))
    c = np.fromiter((tcls.get(x, 0) for x in ty), np.int8, len(ty))
    z = np.fromiter((x in zero_named for x in ty), bool, len(ty))
    has_type[j] = True
    np.maximum.at(max_rate, j, r)
    np.maximum.at(best_cls, j, c)
    all_zero[j[~z]] = False
    if i % 100 == 0:
        print(f"  types {i}/400 ({time.time()-t0:.0f}s)", flush=True)
all_zero &= has_type
print(f"typed: {int(has_type.sum()):,}  all-types-zero-named: {int(all_zero.sum()):,}  "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- names that have landed
NAME_SRC = ["cascade_names.parquet", "external_authority_musicbrainz.parquet",
            "external_authority_small.parquet"] + \
    [os.path.relpath(p, ACQ) for p in sorted(glob.glob(f"{ACQ}/uri_*_hits/*.parquet"))] + \
    [os.path.relpath(p, ACQ) for p in sorted(glob.glob(f"{ACQ}/mb_hits/*.parquet"))]
named = np.zeros(N, bool)
src_rows = {}
for fn in NAME_SRC:
    fp = f"{ACQ}/{fn}"
    if not os.path.exists(fp):
        continue
    uu = np.asarray(pq.read_table(fp, columns=["node_uid"])["node_uid"], dtype=np.int64)
    j = np.searchsorted(U, uu)
    ok = (j < N) & (U[np.minimum(j, N - 1)] == uu)
    named[j[ok]] = True
    src_rows[fn] = len(uu)
print(f"nodes with a recovered name: {int(named.sum()):,} from {len(src_rows)} sources",
      flush=True)

# ---------------------------------------------------------------- the three new axes
E = {e: i for i, e in enumerate(EV)}
S = {s: i for i, s in enumerate(SK)}
is_struct = ev_c == E.get("STRUCTURAL_CVT_INFERENCE", -1)
is_cand = sk_c == S.get("ENTITY_CANDIDATE", -1)
is_trunc = ev_c == E.get("KEY_TRUNCATED_TITLE_REFUSED", -1)
hard = np.isin(ev_c, [E[e] for e in ("TYPE_DECLARED_MEDIATOR", "KEY_DECLARED_CVT",
                                     "KEY_DECLARED_LOAD", "KEY_DECLARED_SCHEMA") if e in E])

# semantic_kind v2: identical to v1 except the inferred mediators get their own value
SK2 = list(SK) + ["CVT_MEDIATOR_INFERRED"]
sk2 = sk_c.copy()
sk2[is_struct] = len(SK)

# nameless_grade
NG = ["NOT_NAMELESS", "SOURCE_DECLARED_NAMELESS", "EMPIRICALLY_NAMELESS", "INFERRED_NAMELESS"]
ng = np.zeros(N, np.int8)
ng[hard] = 1
ng[is_struct] = 3
ng[is_cand & all_zero & ~named] = 2

# recovery_class
RC = ["NAME_RECOVERED", "SOURCE_DECLARED_NAMELESS_INTERNAL", "INFERRED_CVT_HELD_APART",
      "WORD_SENSE", "EXTERNAL_RESOURCE", "EMPIRICALLY_NAMELESS_INTERNAL", "UNTYPED_CANDIDATE",
      "LIKELY_REAL_ENTITY", "LIKELY_CREATIVE_WORK", "LIKELY_MEDIA_ENTITY", "LIKELY_PLACE",
      "LIKELY_PERSON"]
R = {r: i for i, r in enumerate(RC)}
rc = np.full(N, R["LIKELY_REAL_ENTITY"], np.int8)
rc[is_cand & ~has_type] = R["UNTYPED_CANDIDATE"]
rc[is_cand & has_type] = (R["LIKELY_REAL_ENTITY"] + best_cls[is_cand & has_type]).astype(np.int8)
rc[ng == 2] = R["EMPIRICALLY_NAMELESS_INTERNAL"]
rc[sk2 == S.get("EXTERNAL_RESOURCE", -1)] = R["EXTERNAL_RESOURCE"]
rc[sk2 == S.get("WORD_SENSE", -1)] = R["WORD_SENSE"]
rc[is_struct] = R["INFERRED_CVT_HELD_APART"]
rc[hard] = R["SOURCE_DECLARED_NAMELESS_INTERNAL"]
rc[named] = R["NAME_RECOVERED"]
# a NAMED_ENTITY by authority-id key that is not yet named keeps its LIKELY_* class from its types;
# if untyped it is still a real-entity lead, because an authority issued it an id
ne = sk2 == S.get("NAMED_ENTITY", -1)
rc[ne & ~named & ~has_type] = R["LIKELY_REAL_ENTITY"]
rc[ne & ~named & has_type] = (R["LIKELY_REAL_ENTITY"] + best_cls[ne & ~named & has_type]
                              ).astype(np.int8)

# ---------------------------------------------------------------- write
pq.write_table(pa.table({
    "node_uid": pa.array(U, pa.int64()),
    "node_kind_frozen": pa.DictionaryArray.from_arrays(pa.array(kf_c, pa.int32()), pa.array(KF)),
    "semantic_kind": pa.DictionaryArray.from_arrays(pa.array(sk2, pa.int32()), pa.array(SK2)),
    "evidence": pa.DictionaryArray.from_arrays(pa.array(ev_c, pa.int32()), pa.array(EV)),
    "nameless_grade": pa.DictionaryArray.from_arrays(pa.array(ng, pa.int8()), pa.array(NG)),
    "recovery_class": pa.DictionaryArray.from_arrays(pa.array(rc, pa.int8()), pa.array(RC)),
    "has_recovered_name": pa.array(named, pa.bool_()),
    "max_type_named_rate": pa.array(max_rate, pa.float32()),
    "truncated_title_key": pa.array(is_trunc, pa.bool_())}),
    f"{ACQ}/semantic_kind_v2.parquet", compression="zstd")


def census(arr, vocab, mask=None):
    m = arr if mask is None else arr[mask]
    c = np.bincount(m.astype(np.int64), minlength=len(vocab))
    return {vocab[i]: int(c[i]) for i in np.argsort(-c) if c[i]}


ent = kf_c == KF.index("ENTITY_MID")
hunt = np.isin(rc, [R[x] for x in RC if x.startswith("LIKELY_") or x == "UNTYPED_CANDIDATE"])
v2_residue = int(hunt.sum())
by_rc = census(rc, RC)
by_rc_ent = census(rc, RC, ent)
by_ng = census(ng, NG)
by_sk = census(sk2, SK2)

# which types dominate each hunt class, and their graph-wide named rate -- the ordering evidence
cls_types = {c: collections.Counter() for c in RC if c.startswith("LIKELY_") or
             c in ("UNTYPED_CANDIDATE", "EMPIRICALLY_NAMELESS_INTERNAL")}
for fp in sorted(glob.glob(f"{ACQ}/residue_types/*.parquet")):
    t = pq.read_table(fp, columns=["node_uid", "type"])
    uu = np.asarray(t["node_uid"], dtype=np.int64)
    ty = t["type"].combine_chunks().cast(pa.string()).to_pylist()
    j = np.searchsorted(U, uu)
    ok = (j < N) & (U[np.minimum(j, N - 1)] == uu)
    for x in np.flatnonzero(ok):
        c = RC[rc[j[x]]]
        if c in cls_types:
            cls_types[c][ty[x]] += 1
top_types = {c: [{"type": k, "n": v, "graph_named_rate": rate.get(k, {}).get("named_rate")}
                 for k, v in cnt.most_common(12)] for c, cnt in cls_types.items()}

now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rec = {"schema": "SEMANTIC_KIND_OVERLAY/v2",
       "generated_utc": now,
       "SUPERSEDES_NOTHING": ("V3_SEMANTIC_KIND.json (v1) is left as written; this is a new record "
                              "with the three added axes, not an edit of it."),
       "APPEND_ONLY": ("node_kind and CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 (manifest hash "
                       "25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865) untouched."),
       "population": int(N),
       "NAME_SOURCES_READ": src_rows,
       "nodes_with_recovered_name": int(named.sum()),
       "NAMELESS_GRADES": {
           "SOURCE_DECLARED_NAMELESS": ("schema mediator=true on a type the node carries, or the "
                                        "node's own key is a load record. THE ONLY HARD GRADE."),
           "EMPIRICALLY_NAMELESS": ("every type the node carries has zero named instances across "
                                    "all 302M nodes. Measured; the source declares nothing."),
           "INFERRED_NAMELESS": "our structural CVT inference; weakest, held apart"},
       "by_nameless_grade": by_ng,
       "HARD_COUNT_SOURCE_DECLARED_NAMELESS": by_ng.get("SOURCE_DECLARED_NAMELESS", 0),
       "EMPIRICALLY_NAMELESS_NOT_IN_HARD_COUNT": by_ng.get("EMPIRICALLY_NAMELESS", 0),
       "INFERRED_CVT_HELD_APART": by_ng.get("INFERRED_NAMELESS", 0),
       "TYPE_LEVEL": {
           "types_in_graph": len(rate),
           "EMPIRICALLY_NAMELESS_TYPE_count": len(zero_named),
           "EMPIRICALLY_NAMELESS_TYPE_rule": "named instances == 0 across the whole snapshot",
           "near_nameless_types_lt_1e-4": len(near),
           "near_nameless_NOT_treated_as_nameless": ("a type with a handful of named instances is "
                                                     "reported here and its nodes stay in the hunt "
                                                     "at lowest priority; only exact zero earns "
                                                     "the empirical grade")},
       "by_semantic_kind": by_sk,
       "by_recovery_class": by_rc,
       "by_recovery_class_entity_mid_only": by_rc_ent,
       "RECOVERY_CLASS_IS_PRIORITISATION": ("LIKELY_* is derived from Freebase domain membership "
                                            "of the node's types (people.person; location/geography;"
                                            " film/tv/music/broadcast/media_common; book/visual_art/"
                                            "theater/comic_books/opera/cvg/computer/fictional_"
                                            "universe). It orders work; it asserts nothing."),
       "TRUE_NAME_RECOVERY_RESIDUE_V2": v2_residue,
       "TRUE_NAME_RECOVERY_RESIDUE_V2_definition": ("LIKELY_* plus UNTYPED_CANDIDATE: nodes with no "
                                                    "source declaration, no empirical-nameless "
                                                    "typing, and no recovered name yet"),
       "V1_TRUE_NAMED_ENTITY_RESIDUE_for_comparison": 16943048,
       "truncated_title_keys_in_residue": int((is_trunc & hunt).sum()),
       "truncated_title_NOTE": ("bucket D (INFORMATION_DESTROYED) is not assigned yet: a node whose "
                                "key is a mutilated title may still resolve through MusicBrainz or "
                                "an archive, so it stays in the hunt until those are exhausted"),
       "hunt_class_top_types": top_types,
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_SEMANTIC_KIND_V2.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nnameless grade:")
for k, v in by_ng.items():
    print(f"  {v:>12,}  {k}")
print(f"\nrecovery class (all {N:,}):")
for k, v in by_rc.items():
    print(f"  {v:>12,}  {k}")
print(f"\nrecovery class, ENTITY_MID residue only:")
for k, v in by_rc_ent.items():
    print(f"  {v:>12,}  {k}")
print(f"\nTRUE_NAME_RECOVERY_RESIDUE_V2: {v2_residue:,}   (v1: 16,943,048)")
for c in ["LIKELY_PERSON", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY", "LIKELY_CREATIVE_WORK",
          "LIKELY_REAL_ENTITY", "EMPIRICALLY_NAMELESS_INTERNAL"]:
    print(f"\n  {c}:")
    for x in top_types.get(c, [])[:8]:
        print(f"     {x['n']:>10,}  {x['type']:<50} named_rate={x['graph_named_rate']}")
print(f"\n{time.time()-t0:.0f}s")
