"""Item 7: restate package_bytes_modified, which reads 0 and cannot be 0.

WHERE THE ADOPTION RECORD IS

  Not in this repository.  `git log --all -S package_bytes_modified` returns no commits, and a
  scan of every text-ish file in the tree finds zero occurrences of the field name.  So this
  script cannot edit that record.  It produces the VALUE the record needs, measured, with each
  component kept separate, so that whichever definition of "modified" the record uses, the
  right number is derivable from this one file.

WHY 0 IS NOT A ROUNDING ERROR

  Two deletion events have been applied to the package since it was frozen, and both are
  recorded in the package's own files:

    A  CANONICAL_CLEANUP_PLAN.json, DRY_RUN false, 808 files.  Verified here by testing that
       every path it lists is absent from disk now -- not by trusting its own total.  Its
       per-file byte list then reconstructs the total, and the reconstruction is checked
       against the total the record claims.
    B  CANONICAL_DEDUP.json APPLIED_HISTORY, 13 paths / 105 files.  That record carries its own
       honesty note: it is RECONSTRUCTED, because a later dry run of the same script overwrote
       the applied record before APPLIED_HISTORY existed.  Its byte total is the run's reported
       output, not a measurement, and it is labelled that way below.  What IS re-verified here
       is that all 13 paths are absent.

  A + B is the 31.55 GB figure (30.80 + 0.75, each already rounded); the exact byte sum is
  reported.

  Deletions are only half of it.  The package also GAINED bytes: the two encoder patch trees,
  and the top-1000 retrieval caches, which existed for no dataset at freeze time.

WHAT "MODIFIED" HAS TO MEAN

  Bytes deleted and bytes written are different quantities, and one scalar hides which of them
  a reader is getting.  All three are reported.  package_bytes_modified is given as
  deleted + written, because the field exists to answer "is this still the package the record
  adopted", and both directions change that answer.

  Every write component carries its newest mtime, so a reader can see which components postdate
  their own adoption record and drop any that do not.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/package_bytes_ledger.py
"""

import datetime as _dt
import glob
import hashlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "PACKAGE_BYTES_LEDGER.json")
BS = chr(92)


def rel(p):
    return os.path.relpath(p, ROOT).replace(BS, "/")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def iso(ts):
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tree(paths):
    """(n_files, bytes, newest mtime) over any mix of files and directories."""
    n = b = 0
    mt = 0.0
    for p in paths:
        if os.path.isfile(p):
            n += 1
            b += os.path.getsize(p)
            mt = max(mt, os.path.getmtime(p))
        elif os.path.isdir(p):
            for r, _, fs in os.walk(p):
                for f in fs:
                    q = os.path.join(r, f)
                    n += 1
                    b += os.path.getsize(q)
                    mt = max(mt, os.path.getmtime(q))
    return n, b, mt


def deletions():
    """Event A re-verified path by path; event B's paths re-verified, its bytes quoted."""
    ev = []

    plan = json.load(io.open(os.path.join(FC, "CANONICAL_CLEANUP_PLAN.json"), encoding="utf-8"))
    gone_n = gone_b = still_n = still_b = 0
    bucket = {}
    for r in plan["reclaimable_files"]:
        p = os.path.join(ROOT, *r["path"].split("/"))
        e = bucket.setdefault(r.get("bucket", "?"),
                              {"deleted_files": 0, "deleted_bytes": 0,
                               "still_present_files": 0, "still_present_bytes": 0})
        if os.path.exists(p):
            still_n += 1
            still_b += r["bytes"]
            e["still_present_files"] += 1
            e["still_present_bytes"] += r["bytes"]
        else:
            gone_n += 1
            gone_b += r["bytes"]
            e["deleted_files"] += 1
            e["deleted_bytes"] += r["bytes"]
    ev.append({"event": "A_CANONICAL_CLEANUP_PLAN",
               "source_record": "data/final_canonical/CANONICAL_CLEANUP_PLAN.json",
               "DRY_RUN_in_record": plan.get("DRY_RUN"),
               "reclaimable_bytes_claimed_by_record": plan["reclaimable_bytes"],
               "measurement": "every listed path tested for absence on disk now; the total is "
                              "then summed from the record's own per-file byte list",
               "files_confirmed_deleted": gone_n,
               "bytes_confirmed_deleted": gone_b,
               "files_still_present": still_n,
               "bytes_still_present": still_b,
               "agrees_with_record_total": gone_b == plan["reclaimable_bytes"],
               "by_bucket": bucket})

    dedup = json.load(io.open(os.path.join(FC, "CANONICAL_DEDUP.json"), encoding="utf-8"))
    for h in dedup.get("APPLIED_HISTORY", []):
        ps = h.get("paths", [])
        absent = sum(1 for p in ps
                     if not os.path.exists(os.path.join(ROOT, *p["path"].split("/"))))
        ev.append({"event": "B_CANONICAL_DEDUP_APPLIED_HISTORY",
                   "source_record": "data/final_canonical/CANONICAL_DEDUP.json",
                   "applied_utc": h.get("applied_utc"),
                   "bytes_QUOTED_not_remeasured": h.get("bytes"),
                   "files_QUOTED_not_remeasured": h.get("files"),
                   "why_quoted": "the record states it is RECONSTRUCTED -- a later dry run "
                                 "overwrote the applied record before APPLIED_HISTORY existed, "
                                 "so its byte total is the run's reported output. The files are "
                                 "gone, so they cannot be weighed again. This is the one number "
                                 "in this ledger that is not a fresh measurement.",
                   "paths_declared": len(ps),
                   "paths_confirmed_absent_now": absent,
                   "all_paths_absent": absent == len(ps)})
    return ev


def writes():
    comp = []

    for name, sub in (("REV2_ENCODER_PATCH", "_rev2_encoder_patch"),
                      ("DENSE_REPAIR_PATCH", "_dense_repair_patch")):
        n, b, mt = tree([os.path.join(FC, sub)])
        comp.append({"component": name, "path": "data/final_canonical/" + sub,
                     "files": n, "bytes": b, "newest_mtime_utc": iso(mt) if mt else None,
                     "role": "pointer-resolvable patch store; POINTER_INDEX routes a subset of "
                             "canonical positions into it instead of into the base shards"})

    cache = sorted(glob.glob(os.path.join(FC, "*", "retrieval_cache", "*")))
    per = {}
    for p in cache:
        ds = os.path.basename(os.path.dirname(os.path.dirname(p)))
        e = per.setdefault(ds, {"files": 0, "bytes": 0, "newest_mtime_utc": None})
        e["files"] += 1
        e["bytes"] += os.path.getsize(p)
        m = iso(os.path.getmtime(p))
        if e["newest_mtime_utc"] is None or m > e["newest_mtime_utc"]:
            e["newest_mtime_utc"] = m
    n, b, mt = tree(cache)
    comp.append({"component": "RETRIEVAL_CACHE_TOP1000", "files": n, "bytes": b,
                 "newest_mtime_utc": iso(mt) if mt else None, "per_dataset": per,
                 "role": "the top-1000 candidate pool per (dataset, model). Existed for no "
                         "dataset at freeze time, so all of it is new bytes.",
                 "size_law": "exactly n_queries * 1000 * (4 byte int32 id + 2 byte float16 "
                             "score) plus a small npz header, so dense and splade are "
                             "byte-identical per dataset by format and not by duplication"})

    recs = sorted(glob.glob(os.path.join(FC, "*.json")))
    n, b, mt = tree(recs)
    comp.append({"component": "TOP_LEVEL_RECORDS", "files": n, "bytes": b,
                 "newest_mtime_utc": iso(mt) if mt else None,
                 "role": "the package's own JSON records, this one included; listed for "
                         "completeness, negligible against the other components"})
    return comp


def main():
    dels = deletions()
    wrs = writes()

    del_measured = sum(e.get("bytes_confirmed_deleted", 0) for e in dels)
    del_quoted = sum(e.get("bytes_QUOTED_not_remeasured", 0) for e in dels)
    del_total = del_measured + del_quoted
    wr_total = sum(c["bytes"] for c in wrs)

    pi = os.path.join(FC, "POINTER_INDEX.json")
    rec = {
        "RECORD": "PACKAGE_BYTES_LEDGER",
        "_what": "the correct value for the adoption record's package_bytes_modified field, "
                 "which currently reads 0.",
        "corrects": {"field": "package_bytes_modified", "current_value": 0,
                     "current_value_is": "FALSE"},
        "adoption_record_location": {
            "found_in_this_repository": False,
            "searched": ["git log --all -S package_bytes_modified -- 0 commits",
                         "every text-ish file in the tree -- 0 occurrences of the field name"],
            "consequence": "this script cannot edit the adoption record. It supplies the "
                           "measured value; carrying it in is a manual step. Under the standing "
                           "rule that a contract amendment needs a new record with its own hash "
                           "and never an edit, the correction belongs in a NEW adoption record "
                           "that cites this ledger's sha256."},
        "ANSWER": {
            "package_bytes_modified": del_total + wr_total,
            "definition": "bytes deleted + bytes written since the freeze. Both directions "
                          "change whether the package is still the thing the record adopted, "
                          "so one scalar has to carry both; the components are below for a "
                          "reader who means only one of them.",
            "bytes_deleted": del_total,
            "bytes_written": wr_total,
            "bytes_deleted_GB_decimal": round(del_total / 1e9, 3),
            "bytes_written_GB_decimal": round(wr_total / 1e9, 3),
            "package_bytes_modified_GB_decimal": round((del_total + wr_total) / 1e9, 3)},
        "measurement_status": {
            "bytes_deleted_remeasured_now": del_measured,
            "bytes_deleted_quoted_from_record": del_quoted,
            "why_not_all_remeasured": "deleted files cannot be weighed. Event A's total is "
                                      "summed from its own per-file byte list after confirming "
                                      "every path is absent, which is as good as a "
                                      "measurement; event B's path list carries no per-file "
                                      "bytes, so its total is quoted.",
            "all_bytes_written_remeasured_now": True},
        "deletion_events": dels,
        "write_components": wrs,
        "pointer_index_sha256_now": sha(pi),
        "note_on_the_31_55_GB_figure":
            "30.80 + 0.75, each already rounded to 2 dp, gives 31.55. The exact byte sum of "
            "the two events is %d, which is %.3f GB decimal / %.3f GiB."
            % (del_total, del_total / 1e9, del_total / 2 ** 30),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)

    print("DELETIONS")
    for e in dels:
        if "bytes_confirmed_deleted" in e:
            print("  %-34s files %4d  %16s  REMEASURED  agrees_with_record=%s"
                  % (e["event"], e["files_confirmed_deleted"],
                     format(e["bytes_confirmed_deleted"], ","), e["agrees_with_record_total"]))
            if e["files_still_present"]:
                print("      WARNING %d listed paths are STILL PRESENT (%s bytes)"
                      % (e["files_still_present"], format(e["bytes_still_present"], ",")))
        else:
            print("  %-34s files %4d  %16s  QUOTED      all_paths_absent=%s"
                  % (e["event"], e["files_QUOTED_not_remeasured"],
                     format(e["bytes_QUOTED_not_remeasured"], ","), e["all_paths_absent"]))
    print("  %-34s %22s (%.3f GB)"
          % ("TOTAL DELETED", format(del_total, ","), del_total / 1e9))
    print("")
    print("WRITES")
    for c in wrs:
        print("  %-34s files %4d  %16s  newest %s"
              % (c["component"], c["files"], format(c["bytes"], ","), c["newest_mtime_utc"]))
    print("  %-34s %22s (%.3f GB)"
          % ("TOTAL WRITTEN", format(wr_total, ","), wr_total / 1e9))
    print("")
    print("=" * 94)
    print("  package_bytes_modified: 0             <-- FALSE")
    print("  package_bytes_modified: %-13d <-- %.3f GB  (deleted %.3f + written %.3f)"
          % (del_total + wr_total, (del_total + wr_total) / 1e9,
             del_total / 1e9, wr_total / 1e9))
    print("=" * 94)
    print("wrote %s" % rel(OUT))
    print("      sha256 %s" % sha(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
