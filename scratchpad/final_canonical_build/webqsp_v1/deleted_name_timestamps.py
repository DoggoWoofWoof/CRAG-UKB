"""Attach deletion provenance to FREEBASE_DELETED_NAME -- now load-bearing for display selection.

    PYTHONHASHSEED=0 python .../deleted_name_timestamps.py

WHY THIS IS NO LONGER OPTIONAL
  Under the user ruling of 2026-09-08, FREEBASE_DELETED_NAME is not a rank below the dump tier: it
  is one source_kind inside a single FREEBASE_HISTORICAL_ASSERTION class, and the displayed name is
  chosen by CHRONOLOGY, not by source.  Without created_ts/deleted_ts these rows cannot take part in
  that comparison at all.  So the timestamps are the tier's admission ticket, not decoration.

THE FORMAT, WHICH THE FIRST VERSION OF THIS FILE GOT WRONG
  deletions.csv-NNNNN-of-00020 is COMMA delimited, and the columns are NOT (subject, predicate, ...):
      created_ts , creator , deleted_ts , deletor , subject , predicate , object , lang
          0           1          2           3         4          5         6       7
  The first version split on TAB and tested f[1] for the predicate.  Every one of 63,036,271 lines
  produced a single field, failed the length guard, and was skipped -- and the job reported
  provenance_attached = 0 with a straight face.  That is precisely the failure the new invariant
  exists to catch, so this version carries POSITIVE CONTROLS and refuses to write a zero it has not
  earned.

  Object literals contain commas ("Canciones amatorias, 7 songs for voice & piano ...").  So the
  first six fields are taken with a bounded split and the language is taken off the RIGHT end;
  everything between is the literal.  A plain split(",") would corrupt exactly the longest names.

WHAT IS ATTACHED
  For every one of the 44,404 attested (node, label, language) evidence rows -- not only the 43,033
  display rows -- created_ts, deleted_ts and deletor.  Under the ruling's schema these become
  valid_from and valid_until, and assertion_status = DELETED marks the name as original but
  superseded, never as non-original.

AND IT STILL TESTS THE MECHANISM CLAIM
  The join found the deleted dump is 85.93% English yet contributes ZERO English rows to the
  residue, explained as a mass non-English label cleanup.  A date histogram either shows that
  concentration or refutes it.  Both outcomes are written.

OUTPUT (append-only)
  _acquisition/deleted_names_join/part_00000.parquet     + valid_from/valid_until/deleted_by
  _acquisition/deleted_name_evidence/part_00000.parquet  + the same, per attested label
  V3_DELETED_NAME_TIMESTAMPS.json
"""
import sys, io, os, gzip, json, time, tarfile, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
SRC = f"{ACQ}/raw/deleted_freebase.tar.gz"
JOIN = f"{ACQ}/deleted_names_join/part_00000.parquet"
EVI = f"{ACQ}/deleted_name_evidence/part_00000.parquet"
NAME_P = "/type/object/name"
NAME_CAP = 400
t0 = time.time()
if not os.path.exists(SRC):
    sys.exit(f"missing {SRC}")


def parse(line):
    """created_ts,creator,deleted_ts,deletor,subject,predicate,object,lang -- object may contain
    commas, so bound the left split and take lang off the right."""
    p = line.split(",", 6)
    if len(p) < 7:
        return None
    obj, sep, lang = p[6].rpartition(",")
    if not sep:                       # no language field; the whole remainder is the literal
        obj, lang = p[6], ""
    return p[0], p[2], p[3], p[4], p[5], obj, lang


def load(path, cols):
    t = pq.read_table(path)
    out = {}
    for c in cols:
        col = t[c]
        out[c] = (pc.cast(col, pa.string()) if pa.types.is_dictionary(col.type) else col).to_pylist()
    return t, out


tj, J = load(JOIN, ["node_id", "display_name"])
te, E = load(EVI, ["node_id", "label"])
# key on (slash-form subject, literal): the same node can carry several labels and only the matching
# one may take that label's timestamps.
want = collections.defaultdict(list)
for i, (n, d) in enumerate(zip(E["node_id"], E["label"])):
    want[("/" + n.replace(".", "/", 1), d)].append(i)
print(f"evidence rows {te.num_rows:,}  display rows {tj.num_rows:,}  distinct (node,label) keys "
      f"{len(want):,}  ({time.time()-t0:.0f}s)", flush=True)

e_created = [None] * te.num_rows
e_deleted = [None] * te.num_rows
e_by = [None] * te.num_rows
found = 0
n_lines = n_name = n_name_en = 0
all_hist = collections.Counter()
res_hist = collections.Counter()
en_hist = collections.Counter()
deletors = collections.Counter()
probe = []


def ym(ms):
    try:
        return time.strftime("%Y-%m", time.gmtime(int(ms) / 1000.0))
    except Exception:
        return "unparsable"


fh = gzip.open(SRC, "rb")
tf = tarfile.open(fileobj=fh, mode="r|")
for m in tf:
    if not m.isfile():
        continue
    ex = tf.extractfile(m)
    if ex is None:
        continue
    for raw in ex:
        n_lines += 1
        r = parse(raw.decode("utf-8", "replace").rstrip("\n"))
        if r is None or r[4] != NAME_P:
            continue
        cts, dts, dby, sub, _, obj, lang = r
        n_name += 1
        b = ym(dts)
        all_hist[b] += 1
        if lang == "en":
            n_name_en += 1
            en_hist[b] += 1
        if len(probe) < 3:
            probe.append({"subject": sub, "object": obj[:70], "lang": lang,
                          "created_ts": cts, "deleted_ts": dts, "deleted_by": dby})
        for i in want.get((sub, obj[:NAME_CAP]), ()):
            if e_deleted[i] is None:
                e_created[i], e_deleted[i], e_by[i] = cts, dts, dby[:120]
                res_hist[b] += 1
                deletors[dby[:120]] += 1
                found += 1
    if n_lines % 20000000 < 1000000:
        print(f"  {n_lines:,} lines  name rows {n_name:,}  attached {found:,}/{te.num_rows:,} "
              f"({time.time()-t0:.0f}s)", flush=True)
tf.close()
fh.close()

# POSITIVE CONTROLS -- a zero must be earned, per the invariant added 2026-09-08
ctl = {"SOURCE_ROWS": n_lines,
       "VALID_ID_ROWS": n_name,
       "IN_GRAPH_POSITIVE_CONTROL_N": n_name,
       "RESIDUE_HIT_N": found,
       "JOIN_NORMALIZATION": "comma-delimited, columns (created_ts,creator,deleted_ts,deletor,"
                             "subject,predicate,object,lang); subject slash-form matched to "
                             "slash-form; literal capped at 400 chars; language taken off the right"}
if n_name == 0:
    sys.exit(f"POSITIVE CONTROL FAILED: 0 of {n_lines:,} lines parsed as {NAME_P}. "
             f"Refusing to write a zero. Check the delimiter and column order.")
if n_name != 5863639:
    print(f"NOTE: parsed {n_name:,} name rows; the extract recorded 5,863,639. "
          f"Difference must be explained before these timestamps are trusted.", flush=True)
print(f"lines {n_lines:,}  name rows {n_name:,} (en {n_name_en:,})  attached {found:,} of "
      f"{te.num_rows:,}  ({time.time()-t0:.0f}s)", flush=True)

ec = {k: te[k] for k in te.schema.names}
ec["valid_from"] = pa.array(e_created)
ec["valid_until"] = pa.array(e_deleted)
ec["deleted_by"] = pa.array(e_by)
ec["assertion_status"] = pa.array(["DELETED"] * te.num_rows)
ec["source_kind"] = pa.array(["DELETED_TRIPLES"] * te.num_rows)
pq.write_table(pa.table(ec), EVI, compression="zstd")

# carry the same onto the display rows by (node, chosen label)
pos = {}
for i, (n, d) in enumerate(zip(E["node_id"], E["label"])):
    pos.setdefault((n, d), i)
d_created, d_deleted, d_by = [], [], []
for n, d in zip(J["node_id"], J["display_name"]):
    i = pos.get((n, d))
    d_created.append(e_created[i] if i is not None else None)
    d_deleted.append(e_deleted[i] if i is not None else None)
    d_by.append(e_by[i] if i is not None else None)
jc = {k: tj[k] for k in tj.schema.names if k not in ("deleted_ts", "deleted_by", "created_ts")}
jc["valid_from"] = pa.array(d_created)
jc["valid_until"] = pa.array(d_deleted)
jc["deleted_by"] = pa.array(d_by)
jc["assertion_status"] = pa.array(["DELETED"] * tj.num_rows)
jc["source_kind"] = pa.array(["DELETED_TRIPLES"] * tj.num_rows)
# the ordering key the display rule compares across source kinds
jc["last_attested_ts"] = pa.array(d_deleted)
pq.write_table(pa.table(jc), JOIN, compression="zstd")
n_disp = sum(1 for x in d_deleted if x)

res_top = res_hist.most_common(12)
res_tot = sum(res_hist.values()) or 1
top3 = sum(v for _, v in res_top[:3]) / res_tot
bg_top = all_hist.most_common(12)
bg_tot = sum(all_hist.values()) or 1
bg_top3 = sum(v for _, v in bg_top[:3]) / bg_tot
concentrated = top3 >= 0.5 and top3 > bg_top3
verdict = ("MASS_CLEANUP_SUPPORTED" if concentrated else
           "MASS_CLEANUP_NOT_SUPPORTED -- residue deletions are no more concentrated than the "
           "background, so the earlier one-event explanation is not carried by the dates")

rec = {"schema": "DELETED_NAME_TIMESTAMPS/v2",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "adds provenance columns to the tier's own files; frozen artifacts untouched.",
       "SUPERSEDES": "DELETED_NAME_TIMESTAMPS/v1, which split on TAB against a COMMA-delimited "
                     "file, matched 0 of 63,036,271 lines and reported provenance_attached = 0.",
       "WHY_IT_MATTERS_NOW": "under the 2026-09-08 ruling the display name inside "
                             "FREEBASE_HISTORICAL_ASSERTION is chosen by chronology, so a row "
                             "without timestamps cannot compete at all.",
       "JOIN_CONTROL": ctl,
       "evidence_rows": te.num_rows,
       "evidence_rows_with_timestamps": found,
       "display_rows": tj.num_rows,
       "display_rows_with_timestamps": n_disp,
       "name_rows_in_source": n_name,
       "english_name_rows_in_source": n_name_en,
       "SCHEMA_ADDED": {"valid_from": "created_ts -- when Freebase asserted this name",
                        "valid_until": "deleted_ts -- when Freebase removed it",
                        "assertion_status": "DELETED (original, but later superseded)",
                        "source_kind": "DELETED_TRIPLES",
                        "last_attested_ts": "= valid_until; the last instant this label was still "
                                            "the asserted name. This is the key the display rule "
                                            "compares against a dump's snapshot_ts and a page's "
                                            "capture_ts."},
       "MECHANISM_TEST": {
           "claim": "the deleted dump is 85.93% English yet contributes zero English rows to the "
                    "residue because Freebase replaced deleted English names while "
                    "non-English-only objects lost theirs -- i.e. a mass label cleanup.",
           "residue_side_top_3_months_share": round(top3, 4),
           "background_top_3_months_share": round(bg_top3, 4),
           "VERDICT": verdict},
       "sample_parsed_rows": probe,
       "residue_deletions_by_month": dict(res_top),
       "all_deleted_names_by_month": dict(bg_top),
       "english_deleted_names_by_month": dict(en_hist.most_common(12)),
       "top_deletors_on_residue_names": dict(deletors.most_common(15)),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_DELETED_NAME_TIMESTAMPS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items()
                  if k not in ("residue_deletions_by_month", "all_deleted_names_by_month",
                               "english_deleted_names_by_month", "top_deletors_on_residue_names")},
                 indent=1, ensure_ascii=False))
print("\nresidue-side deletions by month:")
for k, v in res_top:
    print(f"  {k}  {v:>8,}")
print("\nbackground (all deleted names) by month:")
for k, v in bg_top:
    print(f"  {k}  {v:>9,}")
print("\ntop deletors on residue names:")
for k, v in deletors.most_common(10):
    print(f"  {k:<44} {v:>8,}")
