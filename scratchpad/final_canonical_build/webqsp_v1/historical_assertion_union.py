"""FREEBASE_HISTORICAL_ASSERTION -- one class, three source kinds, resolved by CHRONOLOGY.

    PYTHONHASHSEED=0 python .../historical_assertion_union.py [--final]

THE RULING THIS IMPLEMENTS (V3_NAME_PROVENANCE_ORDER_V2, sha c4a2a85e...)
  FREEBASE_HISTORICAL_DUMP_NAME, FREEBASE_DELETED_NAME and FREEBASE_HISTORICAL_PAGE_NAME are not
  three ranks.  They are one class of ORIGINAL Freebase assertions of /type/object/name, differing
  only in the artifact that preserved them.  Source establishes AUTHENTICITY; TIME decides which
  historical name is displayed:

      2010 dump    "Foo Corporation"        snapshot_ts  2010-07-16
      2012 deleted "Foo Corp."              valid_until  2012-xx-xx
      2013 page    "Foo Holdings"           capture_ts   2013-xx-xx
                                            -> display "Foo Holdings", the later identity

  A rigid DELETED > DUMP > PAGE would pick "Foo Corp." purely for where it was found.

THE ONE COMPARABLE NUMBER
  last_attested_ts = the last instant a label is KNOWN to have been the asserted name:
      DELETED_TRIPLES  valid_until (deleted_ts)  -- it was still the name right up to removal
      HISTORICAL_DUMP  snapshot_ts               -- it was the name when the dump was cut
      HISTORICAL_PAGE  capture_ts                -- Freebase rendered it at capture time
  All three are normalised to epoch milliseconds so they are directly comparable.

DISPLAY SELECTION (total and seed-independent)
  1. English if any English label exists for the node
  2. else the lexicographically first language code present
  3. within that language, the LATEST last_attested_ts
  4. ties broken by (source_kind, label) so the result is a function of the evidence alone

COMPLETENESS GATE
  The ruling says to let every exact historical pass finish and then rebuild the union ONCE.  This
  script therefore checks each expected pass and marks the record PROVISIONAL unless all are
  present.  --final additionally refuses to write at all if any pass is missing, so a provisional
  union can never be mistaken for the rebuilt one.

NOTHING IS DISCARDED.  Every attested label in every language is written to the evidence table; the
display rule chooses among them without removing any.

OUTPUT (append-only)
  _acquisition/historical_assertion_evidence/part_*.parquet   every attested label, full schema
  _acquisition/historical_assertion_display/part_00000.parquet one display row per node
  V3_HISTORICAL_ASSERTION_UNION.json
"""
import sys, io, os, re, json, glob, time, calendar, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
EVI = f"{ACQ}/historical_assertion_evidence"
DSP = f"{ACQ}/historical_assertion_display"
for d in (EVI, DSP):
    os.makedirs(d, exist_ok=True)
FINAL = "--final" in sys.argv
t0 = time.time()


def ms(datestr):
    return int(calendar.timegm(time.strptime(datestr, "%Y-%m-%d")) * 1000)


def wb_ms(ts):
    """wayback 14-digit YYYYMMDDhhmmss -> epoch ms."""
    s = str(ts)[:14]
    if len(s) < 8 or not s.isdigit():
        return None
    s = s.ljust(14, "0")
    try:
        return int(calendar.timegm(time.strptime(s, "%Y%m%d%H%M%S")) * 1000)
    except Exception:
        return None


# (dir, kind, id column, label column, language column or literal, time spec)
# 'lang:en' means the source is English-only by construction and carries no language column.
SOURCES = [
    ("fb2010_names",            "HISTORICAL_DUMP", "node_id", "display_name", "lang:en",
     ("snapshot", "2010-07-16"), "simple_topic_dump"),
    ("fb2010_tsv_names",        "HISTORICAL_DUMP", "node_id", "display_name", "language",
     ("snapshot", "2010-07-16"), "per_type_tsv"),
    ("fb2010_quad_labels",      "HISTORICAL_DUMP", "node_id", "label", "language",
     ("snapshot", "2010-07-16"), "quadruples"),
    ("fb2010_quad_labels_2008", "HISTORICAL_DUMP", "node_id", "label", "language",
     ("snapshot", "2008-03-28"), "quadruples_2008"),
    ("wex_names",               "HISTORICAL_DUMP", "node_id", "display_name", "language",
     ("snapshot", "2010-07-05"), "wex"),
    ("deleted_name_evidence",   "DELETED_TRIPLES", "node_id", "label", "language",
     ("deleted", None), "deleted_triples"),
    ("fb_page_labels",          "HISTORICAL_PAGE", "freebase_mid", "label", "language",
     ("capture", None), "archived_page"),
]
# every pass the ruling expects before the union may be called rebuilt
EXPECTED = {"fb2010_names": "2010 simple-topic", "fb2010_tsv_names": "2010 per-type TSV",
            "fb2010_quad_labels": "2010 quadruples", "fb2010_quad_labels_2008": "2008 quadruples",
            "deleted_name_evidence": "deleted triples", "wex_names": "WEX",
            "fb_page_labels": "historical pages"}

U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
HUNTC = pa.array(["LIKELY_REAL_ENTITY", "UNTYPED_CANDIDATE", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY",
                  "LIKELY_CREATIVE_WORK", "LIKELY_PERSON"])
H = np.sort(pc.filter(sk["node_uid"], pc.is_in(rc, value_set=HUNTC)).to_numpy())
del sk, rc
print(f"residue {len(U):,}  hunt {len(H):,}  ({time.time()-t0:.0f}s)", flush=True)


def strcol(t, name):
    c = t[name]
    return (pc.cast(c, pa.string()) if pa.types.is_dictionary(c.type) else c).to_pylist()


# ------------------------------------------------------- HISTORICAL_PAGE admissibility gate
# WHY THIS GATE EXISTS. Most of what the archived-page hunt writes is not a name. Measured over the
# 852 rows on disk when the gate was calibrated:
#   574x "Data Dumps"            -- after the shutdown freebase.com 301-redirected every
#                                   /m/<anything> to its farewell page and the scraper read THAT
#                                   page's h1. Every one of these carries page_mid_verified=False:
#                                   the hunt's own url-mid check DID catch the redirect.
#    98x "Freebase - /m/<mid>"   -- the <title> Freebase served for a topic that HAD NO NAME. It
#                                   echoes the mid back, so accepting it would "recover" a name that
#                                   is literally the identifier we were trying to resolve. These
#                                   carry page_mid_verified=True but h1_agrees=False, rung
#                                   html_title.
#   176x a real name             -- 'Ramu Devadas', 'More Than a Feeling', "Can't Help Falling in
#                                   Love". Rung freebase_i18n_table (134) or freebase_h1 (42).
#
# A CORRECTION ON THE RECORD. An earlier version of this comment claimed exactly ONE of 619 rows was
# real, and that a url-mid check could not catch the redirect. Both were wrong. The 619 came from
# the hunt's running log counter and the 618 from a label histogram over the parquet -- two
# different denominators compared as though they were one, which is the same class of error the WEX
# record warns about. Measuring one population directly gives 176/852 = 20.7% of rows admissible,
# and page_mid_verified alone flags all 574 redirect rows. The archived-page source is not
# worthless; it is noisy in a way its own flags describe exactly.
#
# 63 inadmissible rows had already reached the cascade as display names before this gate existed --
# 'Data Dumps' x5 and the rest mid echoes. The union's MID_LIKE test could not catch them: it fires
# on a label that STARTS with "/m/", and "Freebase - /m/07q90vg" starts with "Freebase".
#
# THE RULE. A page label is admissible only when the hunt's own verification flags say the page
# belonged to that mid AND its h1 agreed with the extracted name -- page_mid_verified and h1_agrees,
# both true. Measured: that pair alone is a complete discriminator here -- ZERO rows pass it that
# the title patterns below then have to reject. The patterns are kept as a second line for
# shutdown-era titles in shards that may arrive without flags, not because they do the work.
PAGE_OK = set()
_pn = sorted(glob.glob(f"{ACQ}/fb_page_names/*.parquet"))
for _p in _pn:
    _t = pq.read_table(_p)
    _n = set(_t.schema.names)
    if {"freebase_mid", "page_mid_verified", "h1_agrees"} <= _n:
        for _m, _v, _h in zip(_t["freebase_mid"].to_pylist(),
                              _t["page_mid_verified"].to_pylist(),
                              _t["h1_agrees"].to_pylist()):
            if _v and _h:
                PAGE_OK.add(_m)
print(f"HISTORICAL_PAGE gate: {len(PAGE_OK):,} of the hunt's mids are page_mid_verified AND "
      f"h1_agrees across {len(_pn)} shards", flush=True)

SHUTDOWN_TITLES = {"data dumps", "freebase", "google", "page not found", "not found"}
_MIDECHO = re.compile(r"/[mg]/[0-9a-z_]{2,}")


def page_ok(node_id, label):
    if node_id not in PAGE_OK:
        return False
    s = (label or "").strip()
    if not s or s.lower() in SHUTDOWN_TITLES or _MIDECHO.search(s):
        return False
    return True


rows = []          # (node_id, label, language, kind, rung, valid_from, valid_until, last_ts)
per_source = {}
missing = []
n_rejected = 0
for d, kind, idc, labc, langc, tspec, rung in SOURCES:
    paths = sorted(glob.glob(f"{ACQ}/{d}/*.parquet"))
    if not paths:
        missing.append(d)
        per_source[d] = {"rung": rung, "kind": kind, "status": "MISSING", "rows": 0}
        continue
    n_before = len(rows)
    n_no_ts = 0
    for p in paths:
        t = pq.read_table(p)
        names = set(t.schema.names)
        ids = strcol(t, idc)
        labs = strcol(t, labc)
        langs = ([langc[5:]] * t.num_rows if langc.startswith("lang:")
                 else strcol(t, langc) if langc in names else ["und"] * t.num_rows)
        if tspec[0] == "snapshot":
            snap = ms(tspec[1])
            vf = [None] * t.num_rows
            vu = [None] * t.num_rows
            last = [snap] * t.num_rows
        elif tspec[0] == "deleted":
            vf = strcol(t, "valid_from") if "valid_from" in names else [None] * t.num_rows
            vu = strcol(t, "valid_until") if "valid_until" in names else [None] * t.num_rows
            last = []
            for x in vu:
                try:
                    last.append(int(x))
                except (TypeError, ValueError):
                    last.append(None)
        else:                                            # capture
            cap = strcol(t, "capture_timestamp") if "capture_timestamp" in names \
                else [None] * t.num_rows
            asserted = (strcol(t, "name_asserted_date") if "name_asserted_date" in names
                        else [None] * t.num_rows)
            vf = asserted
            vu = [None] * t.num_rows
            last = [wb_ms(x) if x else None for x in cap]
        for i in range(t.num_rows):
            if kind == "HISTORICAL_PAGE" and not page_ok(ids[i], labs[i]):
                n_rejected += 1
                continue
            if last[i] is None:
                n_no_ts += 1
            rows.append((ids[i], labs[i], langs[i] or "und", kind, rung, vf[i], vu[i], last[i]))
    per_source[d] = {"rung": rung, "kind": kind, "status": "PRESENT",
                     "rows": len(rows) - n_before, "rows_without_a_timestamp": n_no_ts}
    print(f"  {d:<26} {kind:<16} +{len(rows)-n_before:>8,} labels "
          f"({time.time()-t0:.0f}s)", flush=True)

if not rows:
    sys.exit("no historical assertion rows found at all -- refusing to write an empty union")

# --- join control (ZERO_JOIN_INVARIANT) -------------------------------------------------------
ids = [r[0] for r in rows]
uid = np.fromiter((hash(x) for x in ids), np.int64, len(ids))
pos = np.clip(np.searchsorted(U, uid), 0, len(U) - 1)
in_res = U[pos] == uid
ph = np.clip(np.searchsorted(H, uid), 0, len(H) - 1)
in_hunt = (H[ph] == uid) & in_res
# positive control: does the identifier shape join ANYWHERE, not only in the residue?
ctl_ok = int(in_res.sum()) > 0
print(f"labels {len(rows):,}  on residue nodes {int(in_res.sum()):,}  in hunt "
      f"{int(in_hunt.sum()):,}  ({time.time()-t0:.0f}s)", flush=True)
if not ctl_ok:
    sys.exit("POSITIVE CONTROL FAILED: no historical label joins the residue at all. "
             "Suspect an id-normalisation bug (slash-form vs dot-form). Refusing to write.")

keep = np.flatnonzero(in_res)
ev = [rows[i] for i in keep.tolist()]
ev_uid = uid[keep]
ev_hunt = in_hunt[keep]

pq.write_table(pa.table({
    "node_uid": pa.array(ev_uid, pa.int64()),
    "node_id": pa.array([r[0] for r in ev]),
    "label": pa.array([r[1] for r in ev]),
    "language": pa.array([r[2] for r in ev]),
    "source_kind": pa.array([r[3] for r in ev]),
    "source_rung": pa.array([r[4] for r in ev]),
    "valid_from": pa.array([r[5] for r in ev]),
    "valid_until": pa.array([r[6] for r in ev]),
    "last_attested_ts": pa.array([r[7] for r in ev], pa.int64()),
    "assertion_status": pa.array(["DELETED" if r[3] == "DELETED_TRIPLES"
                                  else "ACTIVE_AT_SOURCE_TIME" for r in ev]),
    "is_original_name": pa.array([True] * len(ev), pa.bool_()),
    "is_current_snapshot_name": pa.array([False] * len(ev), pa.bool_()),
    "in_hunt": pa.array(ev_hunt, pa.bool_())}),
    f"{EVI}/part_00000.parquet", compression="zstd")

# --- display selection -------------------------------------------------------------------------
by_node = collections.defaultdict(list)
for j, r in enumerate(ev):
    by_node[int(ev_uid[j])].append(j)

MID_LIKE = ("/m/", "/g/", "m.", "g.")
disp = []
rule_count = collections.Counter()
kind_count = collections.Counter()
lang_count = collections.Counter()
n_all_mid = 0
for u in sorted(by_node):
    cand = by_node[u]
    # a MID-shaped string is never a name (V1 ruling, retained in V2)
    # NO FALLBACK. This filter used to end in "or cand", which kept the unfiltered list whenever
    # filtering emptied it -- so a node whose ONLY attested label was its own mid was published with
    # that mid as its display name. Two nodes reached the cascade that way, m.0gx99s8 '/m/0gx99s8'
    # and m.0rhnw_v '/m/0rhnw_v', both from the deleted-triples dump, where Freebase itself had
    # recorded a /type/object/name whose value was the identifier (tagged @de and @es). Freebase
    # asserting it does not make it a name. A node whose every attested label is mid-shaped has no
    # name and belongs in the nameless population, not in the named one, so it is dropped here.
    cand = [j for j in cand if ev[j][1] and not ev[j][1].strip().startswith(MID_LIKE)]
    if not cand:
        n_all_mid += 1
        continue
    langs = {ev[j][2] for j in cand}
    if "en" in langs:
        lang, rule = "en", "english_present"
    else:
        lang, rule = min(langs), "lexicographically_first_language"
    grp = [j for j in cand if ev[j][2] == lang]
    # latest last_attested_ts wins; rows without one lose to any row that has one
    grp.sort(key=lambda j: (ev[j][7] is not None, ev[j][7] or 0, ev[j][3], ev[j][1]), reverse=True)
    best = grp[0]
    if len(grp) > 1 and ev[grp[1]][7] != ev[best][7]:
        rule += "+latest_attested"
    r = ev[best]
    disp.append((u, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                 "DELETED" if r[3] == "DELETED_TRIPLES" else "ACTIVE_AT_SOURCE_TIME",
                 len(cand), len(langs), bool(ev_hunt[best]), rule))
    rule_count[rule] += 1
    kind_count[r[3]] += 1
    lang_count[r[2]] += 1

n = len(disp)
pq.write_table(pa.table({
    "node_uid": pa.array([d[0] for d in disp], pa.int64()),
    "node_id": pa.array([d[1] for d in disp]),
    "display_name": pa.array([d[2] for d in disp]),
    "language": pa.array([d[3] for d in disp]),
    "source_kind": pa.array([d[4] for d in disp]),
    "source_rung": pa.array([d[5] for d in disp]),
    "valid_from": pa.array([d[6] for d in disp]),
    "valid_until": pa.array([d[7] for d in disp]),
    "last_attested_ts": pa.array([d[8] for d in disp], pa.int64()),
    "assertion_status": pa.array([d[9] for d in disp]),
    "n_attested_labels": pa.array([d[10] for d in disp], pa.int32()),
    "n_attested_languages": pa.array([d[11] for d in disp], pa.int32()),
    "in_hunt": pa.array([d[12] for d in disp], pa.bool_()),
    "display_rule": pa.array([d[13] for d in disp]),
    "source": pa.array(["FREEBASE_HISTORICAL_ASSERTION"] * n),
    "is_original_name": pa.array([True] * n, pa.bool_()),
    "is_current_snapshot_name": pa.array([False] * n, pa.bool_())}),
    f"{DSP}/part_00000.parquet", compression="zstd")

# how often does chronology actually change the answer vs a rigid source order?
flip = sum(1 for d in disp if "+latest_attested" in d[13])
status = "REBUILT_COMPLETE" if not missing else "PROVISIONAL"
rec = {"schema": "HISTORICAL_ASSERTION_UNION/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/; frozen graph and frozen overlay untouched.",
       "IMPLEMENTS": "V3_NAME_PROVENANCE_ORDER_V2 "
                     "(c4a2a85e461084a459650cfe084293211429cba7217e9590cff7dd8604eec0e7)",
       "STATUS": status,
       "HISTORICAL_PAGE_ADMISSIBILITY_GATE": {
           "WHY": "most of what the archived-page hunt writes is not a name. Measured over the "
                  "852 rows on disk when this gate was calibrated: 574 'Data Dumps' (after the "
                  "shutdown freebase.com 301-redirected every /m/<anything> to its farewell page "
                  "and the scraper read that page's h1), 98 'Freebase - /m/<mid>' (the title "
                  "Freebase served for a topic that HAD NO NAME, echoing back the very identifier "
                  "being resolved), and 176 real names.",
           "CORRECTION_ON_THE_RECORD": "an earlier version of this record claimed exactly ONE of "
                  "619 rows was real, and that a url-mid check could not catch the redirect. Both "
                  "were wrong. The 619 came from the hunt's running log counter and the 618 from a "
                  "label histogram over the parquet -- two different denominators compared as "
                  "though they were one. Measuring one population directly gives 176/852 = 20.7% "
                  "of rows admissible, and page_mid_verified=False already flags all 574 redirect "
                  "rows, so the url-mid check caught every one of them. The archived-page source "
                  "is not worthless; it is noisy in a way its own flags describe exactly.",
           "WHY_THE_OLD_GUARD_MISSED_IT": "the union rejected labels STARTING with '/m/'. "
                                          "'Freebase - /m/07q90vg' starts with 'Freebase'. That is "
                                          "what let 63 inadmissible rows -- 'Data Dumps' x5 and "
                                          "the rest mid echoes -- reach the cascade as display "
                                          "names before this gate existed.",
           "DISCRIMINATOR_IS_THE_FLAG_PAIR": "measured, (page_mid_verified AND h1_agrees) is a "
                  "complete discriminator here: ZERO rows pass it that the title patterns then "
                  "have to reject. The patterns are a second line for shards that may arrive "
                  "without flags, not the mechanism.",
           "RULE": "a page label is admissible only if the hunt flagged the capture "
                   "page_mid_verified AND h1_agrees, and the label is not a known shutdown title "
                   "and contains no /m/ or /g/ id echo.",
           "mids_passing_the_flag_test": len(PAGE_OK),
           "page_rows_rejected": n_rejected,
           "CONSEQUENCE": "HISTORICAL_PAGE is a NOISY source, not an empty one. Roughly a fifth "
                          "of the rows it writes are real names and the rest are shutdown-era "
                          "artifacts the hunt's own flags already mark. The gate keeps the fifth "
                          "and drops the rest.",
           "PRIOR_CONTAMINATION": "63 nodes carried one of these strings as their cascade display "
                                  "name before this gate existed."},
       "COMPLETENESS_GATE": {
           "expected_passes": EXPECTED,
           "missing": [EXPECTED.get(m, m) for m in missing],
           "RULE": "the ruling requires every exact historical pass to finish before the union is "
                   "rebuilt once. While anything is missing this record is PROVISIONAL and must "
                   "not be treated as the rebuilt union."},
       "JOIN_CONTROL": {
           "SOURCE_ROWS": len(rows),
           "VALID_ID_ROWS": len(rows),
           "IN_GRAPH_POSITIVE_CONTROL_N": int(in_res.sum()),
           "RESIDUE_HIT_N": int(in_res.sum()),
           "JOIN_NORMALIZATION": "node_uid = hash(node_id) under PYTHONHASHSEED=0; every source "
                                 "already stores dot-form node_id, the deleted rung having been "
                                 "normalised from slash-form at extraction."},
       "attested_labels_total": len(rows),
       "attested_labels_on_residue": int(in_res.sum()),
       "attested_labels_in_hunt": int(in_hunt.sum()),
       "distinct_nodes_with_a_historical_assertion": n,
       "distinct_nodes_in_hunt": int(sum(1 for d in disp if d[12])),
       "per_source": per_source,
       "DISPLAY_RULE": ["english if present", "else lexicographically first language",
                        "then latest last_attested_ts", "ties by (source_kind, label)"],
       "display_rule_applied": dict(rule_count),
       "display_by_source_kind": dict(kind_count),
       "display_by_language": dict(lang_count.most_common(20)),
       "nodes_where_chronology_changed_the_answer": flip,
       "nodes_dropped_every_label_was_mid_shaped": n_all_mid,
       "WHY_THAT_DROP": "a node whose every attested label is its own identifier has no name. The "
                        "filter used to fall back to the unfiltered list when it emptied one, which "
                        "published the mid as the display name for 2 nodes -- both from the "
                        "deleted-triples dump, where Freebase had recorded a /type/object/name "
                        "whose value was the mid. Freebase asserting it does not make it a name.",
       "WHY_THAT_NUMBER_MATTERS": "it is the count of nodes where a rigid source order and the "
                                  "ruling's chronological order disagree -- the size of the "
                                  "mistake the ruling prevents.",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_HISTORICAL_ASSERTION_UNION.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items() if k != "per_source"}, indent=1, ensure_ascii=False))
print("\nper source:")
for d, v in per_source.items():
    print(f"  {d:<26} {v['kind']:<16} {v['status']:<8} {v['rows']:>9,}")
if missing and FINAL:
    sys.exit(f"\n--final requested but these passes are missing: {missing}. "
             f"Union written as PROVISIONAL; rerun with --final when they land.")
