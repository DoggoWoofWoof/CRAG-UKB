"""FREEBASE_DELETED_NAME -- tier 2 of the locked order, from Freebase's own deleted-triples dump.

    PYTHONHASHSEED=0 python .../deleted_name_join.py

WHY THIS EXISTED ON DISK AND WAS NEVER USED
  V3_TIER2_DELETED_EXTRACT.json already parsed deleted_freebase.tar.gz (md5 3a2b9038..., 20 members,
  63,036,271 triples, deletions through March 2013, line count matching the published figure) and
  wrote 5,863,639 /type/object/name rows to _acquisition/deleted_names.parquet.  It then measured
  those rows against UNRESOLVED MEDIATORS only -- 3,532 rows -- and stopped.  The full 69,777,967
  node residue was never joined, and cascade_union.py carried FREEBASE_DELETED_NAME with an empty
  file list, printing "no source yet" while the source sat on disk.

THE SLASH/DOT TRAP, WHICH COST A FALSE ZERO ONCE ALREADY
  Subjects in the deleted dump are SLASH form, /m/040_1l9.  Canonical node_id is DOT form,
  m.040_1l9, and node_uid = hash(node_id).  Joining the raw column gives exactly 0 hits out of
  5,863,639 -- a clean, plausible, entirely wrong answer.  Normalised (strip the leading slash, then
  "/" -> "."), the same data hits 44,404 rows on 43,033 distinct residue nodes.  A hard zero from a
  large source is treated here as a suspected bug, never as a finding; this campaign has already
  been burned once by two IDIR tables that intersected to zero for a comparable reason.

WHAT THESE NAMES ARE
  Freebase itself asserted /type/object/name on the object, and later deleted the triple.  So:
      is_original_name        = True   -- Freebase asserted it, this is not a reconciliation
      is_current_snapshot_name = False -- it is not in the 2015 snapshot; that is why it was deleted
  which is the same two-field distinction the user fixed for the archived-page tier, and the reason
  "original" is never allowed to mean "present in our snapshot".

MULTI-LABEL RULE (locked)
  Where a node has labels in several languages, ALL are preserved in the evidence table and the
  display name is chosen deterministically: English if present, else lexicographically first
  language code.  Nothing is discarded.

OUTPUT (append-only)
  _acquisition/deleted_name_evidence/part_*.parquet   every attested (node, label, language)
  _acquisition/deleted_names_join/part_00000.parquet  one display row per node, cascade-ready
  V3_DELETED_NAME_JOIN.json
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
OUT = f"{ACQ}/deleted_names_join"
EVI = f"{ACQ}/deleted_name_evidence"
for d in (OUT, EVI):
    os.makedirs(d, exist_ok=True)
TIER = "FREEBASE_DELETED_NAME"
SRC = "deleted_freebase.tar.gz (md5 3a2b903862ea9d7d79f106a3821c3b02, 63,036,271 triples, " \
      "deletions through March 2013)"
NAME_CAP = 400
t0 = time.time()


def norm(s):
    """/m/040_1l9 -> m.040_1l9.  The whole finding turns on this one line."""
    return s.lstrip("/").replace("/", ".")


U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
HUNTC = pa.array(["LIKELY_REAL_ENTITY", "UNTYPED_CANDIDATE", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY",
                  "LIKELY_CREATIVE_WORK", "LIKELY_PERSON"])
H = np.sort(pc.filter(sk["node_uid"], pc.is_in(rc, value_set=HUNTC)).to_numpy())
del sk, rc

# everything already recovered, so the MARGINAL is measured and not inferred by adding reaches
prevdirs = ["fb2010_names", "fb2010_quad_names", "fb2010_quad_names_2008", "fb2010_tsv_names",
            "wex_names", "fb_page_names"]
ppaths = []
for d in prevdirs:
    ppaths += sorted(glob.glob(f"{ACQ}/{d}/*.parquet"))
PREV = (np.sort(pa.concat_tables([pq.read_table(x, columns=["node_uid"]) for x in ppaths])
                ["node_uid"].to_numpy()) if ppaths else np.zeros(0, np.int64))
print(f"residue {len(U):,}  hunt {len(H):,}  already recovered rows {len(PREV):,} "
      f"from {len(ppaths)} parts  ({time.time()-t0:.0f}s)", flush=True)

t = pq.read_table(f"{ACQ}/deleted_names.parquet")


def col(name):
    c = t[name]
    return (pc.cast(c, pa.string()) if pa.types.is_dictionary(c.type) else c).to_pylist()


sub, obj, lang = col("subject"), col("object"), col("lang")
print(f"deleted /type/object/name rows: {len(sub):,}  ({time.time()-t0:.0f}s)", flush=True)

nid = [norm(s) for s in sub]
uid = np.fromiter((hash(x) for x in nid), np.int64, len(nid))
pos = np.clip(np.searchsorted(U, uid), 0, len(U) - 1)
hit = U[pos] == uid
idx = np.flatnonzero(hit)
print(f"rows on unnamed residue nodes: {len(idx):,}  ({time.time()-t0:.0f}s)", flush=True)


def member(arr, ref):
    if len(ref) == 0 or len(arr) == 0:
        return np.zeros(len(arr), bool)
    p = np.clip(np.searchsorted(ref, arr), 0, len(ref) - 1)
    return ref[p] == arr


hu = uid[idx]
in_hunt = member(hu, H)
is_new = ~member(hu, PREV)

# evidence: every attested label, nothing discarded
pq.write_table(pa.table({"node_uid": pa.array(hu, pa.int64()),
                         "node_id": pa.array([nid[i] for i in idx.tolist()]),
                         "label": pa.array([obj[i][:NAME_CAP] for i in idx.tolist()]),
                         "language": pa.array([lang[i] for i in idx.tolist()]),
                         "in_hunt": pa.array(in_hunt, pa.bool_())}),
                f"{EVI}/part_00000.parquet", compression="zstd")

# display rule: English if present, else lexicographically first language code
by_node = collections.defaultdict(dict)
for j, i in enumerate(idx.tolist()):
    d = by_node[int(hu[j])]
    lg = lang[i]
    if lg not in d:                       # first label wins within a language, deterministically
        d[lg] = (obj[i][:NAME_CAP], nid[i], bool(in_hunt[j]), bool(is_new[j]))

rows = []
rule_count = collections.Counter()
lang_count = collections.Counter()
for u in sorted(by_node):
    d = by_node[u]
    if "en" in d:
        lg, rule = "en", "english_present"
    else:
        lg, rule = min(d), "lexicographically_first_language"
    nm, nodeid, inh, new = d[lg]
    rows.append((u, nodeid, nm, lg, len(d), inh, new))
    rule_count[rule] += 1
    lang_count[lg] += 1

n = len(rows)
pq.write_table(pa.table({
    "node_uid": pa.array([r[0] for r in rows], pa.int64()),
    "node_id": pa.array([r[1] for r in rows]),
    "display_name": pa.array([r[2] for r in rows]),
    "language": pa.array([r[3] for r in rows]),
    "n_attested_languages": pa.array([r[4] for r in rows], pa.int32()),
    "in_hunt": pa.array([r[5] for r in rows], pa.bool_()),
    "new_vs_everything_recovered": pa.array([r[6] for r in rows], pa.bool_()),
    "source": pa.array([TIER] * n),
    "source_artifact": pa.array(["deleted_freebase.tar.gz"] * n),
    "deletion_coverage": pa.array(["through 2013-03"] * n),
    "is_original_name": pa.array([True] * n, pa.bool_()),
    "is_current_snapshot_name": pa.array([False] * n, pa.bool_())}),
    f"{OUT}/part_00000.parquet", compression="zstd")

rec = {"schema": "DELETED_NAME_JOIN/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/; frozen graph and frozen overlay untouched.",
       "tier": TIER,
       "TIER_PLACEMENT": "NOT provisional. FREEBASE_DELETED_NAME is position 2 of the locked "
                         "provenance order (V3_NAME_PROVENANCE_ORDER_V1), above "
                         "FREEBASE_HISTORICAL_PAGE_NAME and FREEBASE_ALIAS. No ruling is needed.",
       "source": SRC,
       "WHY_IT_WAS_MISSED": "V3_TIER2_DELETED_EXTRACT measured these rows against unresolved "
                            "MEDIATORS only (3,532) and never against the full residue; "
                            "cascade_union.py carried the tier with an empty file list.",
       "THE_SLASH_DOT_TRAP": {
           "raw_subject_form": "/m/040_1l9",
           "canonical_node_id_form": "m.040_1l9",
           "unnormalised_join_result": 0,
           "normalised_join_result": int(len(idx)),
           "RULE": "a hard zero from a large source is a suspected bug, not a finding."},
       "deleted_name_rows_total": int(len(sub)),
       "rows_on_unnamed_residue_nodes": int(len(idx)),
       "distinct_residue_nodes_named": n,
       "HUNT_nodes_named": int(sum(1 for r in rows if r[5])),
       "MARGINAL_over_everything_recovered": int(sum(1 for r in rows if r[6])),
       "compared_against": prevdirs,
       "nodes_with_more_than_one_language": int(sum(1 for r in rows if r[4] > 1)),
       "display_rule_applied": dict(rule_count),
       "display_languages": dict(lang_count.most_common(25)),
       "SEMANTICS": {"is_original_name": True,
                     "is_current_snapshot_name": False,
                     "meaning": "Freebase asserted /type/object/name on this object and later "
                                "deleted the triple. Original, but deliberately not current."},
       "MULTI_LABEL_RULE": "all attested labels preserved in deleted_name_evidence/; display name "
                           "chosen English-first, else lexicographically first language code.",
       "TIMESTAMPS_ARE_RE_DERIVABLE": "deleted_ts/deletor/created_ts/creator exist in the verified "
                                      "tar.gz but were not carried into deleted_names.parquet; a "
                                      "second pass over the artifact can attach them.",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_DELETED_NAME_JOIN.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))
print("\nsample recovered deleted names:")
for r in rows[:20]:
    print(f"  {r[1]:<16} {r[2][:52]!r:<54} [{r[3]}] langs={r[4]} hunt={r[5]} new={r[6]}")
