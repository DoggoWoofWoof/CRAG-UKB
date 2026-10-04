# -*- coding: utf-8 -*-
"""Cleanup pass 3: delete exactly the DELETE-verdict paths of CLEANUP_REFERENCE_CROSSCHECK.json.

DRY RUN BY DEFAULT. Nothing is deleted without --apply.

WHAT THIS PASS IS
-----------------
cleanup_plan.py (passes 1 and 2) reclaimed the superseded docs-side encodings and the build
staging that its own reference closure could prove dead. Everything it could not classify was
left in place and later inventoried by cleanup_reference_crosscheck.py, which checked every
remaining candidate against the records (by path AND by name), the live code, the
results/archive/docs tree and the pointer/cache resolution. This pass deletes the candidates
that check returned DELETE for, and nothing else. The verdicts are read from the record on
disk; the safety checks below are re-derived here, at delete time, from the source records,
so a bug in the cross-check cannot also pass the gate that guards it.

WHAT IT REFUSES, REGARDLESS OF THE VERDICT
------------------------------------------
  - anything under the PROTECTED prefixes (raw mirror, IDIR, FACC1, frozen Freebase layers)
  - anything a LOCKED_* record declares (verify_manifest.declared_index)
  - anything POINTER_INDEX or RETRIEVAL_CACHE resolves into
  - anything git tracks
  - any candidate whose precondition fails

PRECONDITIONS (evidence goes into the ledger)
---------------------------------------------
  _dense_repair_patch/<store>__pNNN   the parts must concatenate row-for-row (dense.npy bytes,
                                      dense_ids.json, dense_rows.json) into the merged <store>/
                                      directory; a store whose parts do not reproduce it keeps
                                      its parts
  metaqa_temp/**                      every file must be byte-identical (sha256) to the file at
                                      the same relative path under data/raw/metaqa
  crag_data_backup/**                 its four results/level_2 files are copied into
                                      archive/baseline_results_2026-04/level_2/from_crag_data_backup/
                                      and verified by sha256 before the backup goes; every other
                                      file is hashed and compared with the live file at the same
                                      relative path, so the ledger says what was a duplicate and
                                      what was a superseded substrate
  scratchpad/**                       only bulk binaries and files over 5 MB are deleted; the
                                      small json/md/txt/csv/log notes stay, because they are the
                                      results of those runs and results are what the audit keeps

LEDGER
------
data/final_canonical/CLEANUP_PASS_3.json: per-file path, bytes and sha256 of everything
deleted, everything refused and why, the precondition evidence and the results preserved.

Run:
  python src/dataset_canonical/cleanup_pass_3.py            # dry run, writes the ledger as DRY_RUN
  python src/dataset_canonical/cleanup_pass_3.py --apply    # deletes, writes the ledger
"""
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_manifest import declared_index, record_hash  # noqa: E402
from cleanup_plan import PROTECTED, PROTECTED_WHY, is_under  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = "data/final_canonical"
CROSSCHECK = FC + "/CLEANUP_REFERENCE_CROSSCHECK.json"
OUT = FC + "/CLEANUP_PASS_3.json"

RESULTS_DEST = "archive/baseline_results_2026-04/level_2/from_crag_data_backup"

BULK_EXT = {".npz", ".npy", ".pt", ".pth", ".pkl", ".bin", ".parquet", ".index", ".faiss",
            ".safetensors", ".arrow", ".h5", ".jsonl", ".tsv"}
SCRATCH_KEEP_MAX_BYTES = 5 * 1024 * 1024


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def norm(p):
    return p.replace("\\", "/").rstrip("/")


def walk_files(base):
    if os.path.isfile(base):
        return [norm(base)]
    out = []
    for dp, _dn, fn in os.walk(base):
        for f in fn:
            out.append(norm(os.path.join(dp, f)))
    return sorted(out)


# ---------------------------------------------------------------------------
def live_resolution_paths():
    """Every path POINTER_INDEX or RETRIEVAL_CACHE resolves into, as directory prefixes/files."""
    live = set()
    pi = rj(FC + "/POINTER_INDEX.json")
    for sec in ("datasets", "queries"):
        for _ds, d in (pi.get(sec) or {}).items():
            for ch in ("dense", "splade"):
                for st in ((d.get(ch) or {}).get("stores") or []):
                    p = norm(st["path"])
                    live.add(p)
                    live.add(os.path.dirname(p))
    rc = rj(FC + "/RETRIEVAL_CACHE.json")
    for _ds, d in (rc.get("caches") or {}).items():
        for ch in ("dense", "splade"):
            f = (d.get(ch) or {}).get("file")
            if f:
                live.add(norm(f))
    return live


def git_tracked():
    r = subprocess.run(["git", "ls-files", "-z"], capture_output=True, cwd=REPO)
    return set(norm(x) for x in r.stdout.decode("utf-8", "replace").split("\0") if x)


# ---------------------------------------------------------------------------
def check_parts_reproduce_store(store, parts):
    """The __pNNN part directories must be an exact row-concatenation of the merged store."""
    ev = {"store": store, "parts": parts, "ok": False}
    try:
        S = np.load(store + "/dense.npy", mmap_mode="r")
        s_ids = rj(store + "/dense_ids.json")
        s_rows = rj(store + "/dense_rows.json")
    except Exception as e:  # noqa: BLE001
        ev["error"] = "store unreadable: %s" % e
        return ev
    off = 0
    ids, rows = [], []
    for p in parts:
        try:
            A = np.load(p + "/dense.npy", mmap_mode="r")
            ids += rj(p + "/dense_ids.json")
            rows += rj(p + "/dense_rows.json")
        except Exception as e:  # noqa: BLE001
            ev["error"] = "%s unreadable: %s" % (p, e)
            return ev
        if A.dtype != S.dtype or A.shape[1:] != S.shape[1:] or off + A.shape[0] > S.shape[0]:
            ev["error"] = "%s shape/dtype %s %s does not fit store %s %s at row %d" % (
                p, A.shape, A.dtype, S.shape, S.dtype, off)
            return ev
        n = A.shape[0]
        for b in range(0, n, 8192):
            e = min(n, b + 8192)
            a = np.ascontiguousarray(A[b:e]).view(np.uint8)
            s = np.ascontiguousarray(S[off + b:off + e]).view(np.uint8)
            if not np.array_equal(a, s):
                ev["error"] = "%s rows %d..%d differ from store rows %d..%d" % (
                    p, b, e, off + b, off + e)
                return ev
        off += n
    ev["rows_store"] = int(S.shape[0])
    ev["rows_parts"] = int(off)
    ev["ids_equal"] = ids == s_ids
    ev["rows_equal"] = rows == s_rows
    ev["ok"] = (off == S.shape[0]) and ev["ids_equal"] and ev["rows_equal"]
    if not ev["ok"] and "error" not in ev:
        ev["error"] = "row count / ids / rows mismatch"
    return ev


def check_metaqa_temp(base="metaqa_temp", live="data/raw/metaqa"):
    ev = {"candidate": base, "live_tree": live, "files": [], "ok": True}
    for f in walk_files(base):
        rel = f[len(base) + 1:]
        lf = live + "/" + rel
        row = {"file": f, "live": lf, "bytes": os.path.getsize(f)}
        if not os.path.isfile(lf):
            row["identical"] = False
            row["error"] = "no live counterpart"
        else:
            row["sha256"] = sha256_file(f)
            row["identical"] = row["sha256"] == sha256_file(lf)
        ev["ok"] = ev["ok"] and bool(row["identical"])
        ev["files"].append(row)
    ev["n_files"] = len(ev["files"])
    return ev


def preserve_backup_results(apply_it, base="crag_data_backup"):
    """Copy the backup's results files into the archive (with a README) and hash the rest."""
    ev = {"candidate": base, "results_dest": RESULTS_DEST, "results": [], "other_files": [],
          "ok": True}
    files = walk_files(base)
    results = [f for f in files if f.startswith(base + "/results/")]
    for f in results:
        rel = f[len(base + "/results/"):]              # level_2/<name>
        name = rel.split("/")[-1]
        dest = RESULTS_DEST + "/" + name
        src_sha = sha256_file(f)
        row = {"file": f, "sha256": src_sha, "bytes": os.path.getsize(f), "dest": dest}
        live = "results/" + rel
        arch = "archive/baseline_results_2026-04/" + rel
        row["live_counterpart"] = {"path": live, "exists": os.path.isfile(live)}
        row["archive_counterpart"] = {"path": arch, "exists": os.path.isfile(arch)}
        if os.path.isfile(live):
            row["live_counterpart"]["identical"] = sha256_file(live) == src_sha
        if os.path.isfile(arch):
            row["archive_counterpart"]["identical"] = sha256_file(arch) == src_sha
        if apply_it:
            os.makedirs(RESULTS_DEST, exist_ok=True)
            if not (os.path.isfile(dest) and sha256_file(dest) == src_sha):
                shutil.copy2(f, dest)
            row["dest_sha256"] = sha256_file(dest)
            row["copied_ok"] = row["dest_sha256"] == src_sha
            ev["ok"] = ev["ok"] and row["copied_ok"]
        else:
            row["copied_ok"] = None
        ev["results"].append(row)
    for f in files:
        if f in results:
            continue
        rel = f[len(base) + 1:]
        row = {"file": f, "bytes": os.path.getsize(f), "sha256": sha256_file(f),
               "live_counterpart": rel, "live_exists": os.path.isfile(rel)}
        if row["live_exists"]:
            row["identical_to_live"] = sha256_file(rel) == row["sha256"]
        ev["other_files"].append(row)
    ev["n_results"] = len(results)
    ev["n_other"] = len(ev["other_files"])
    ev["n_other_identical_to_live"] = sum(1 for r in ev["other_files"]
                                          if r.get("identical_to_live"))
    if apply_it:
        readme = RESULTS_DEST + "/README.md"
        lines = [
            "# Results recovered from crag_data_backup/ (cleanup pass 3, %s)"
            % datetime.date.today().isoformat(),
            "",
            "`crag_data_backup/` was a pre-`_clean` snapshot of the legacy ukb_storage substrates,",
            "`data/processed/master_nodes.json`, the HNM-ablation checkpoints and four Level-2",
            "reranking result files. Everything but the result files was either byte-identical to",
            "the live copy or a superseded substrate, and is recorded (path, bytes, sha256) in",
            "`data/final_canonical/CLEANUP_PASS_3.json` before deletion. The result files are",
            "preserved here because results are what the audit keeps:",
            "",
        ]
        for r in ev["results"]:
            note = []
            if r["archive_counterpart"]["exists"]:
                note.append("archive copy %s" % (
                    "identical" if r["archive_counterpart"].get("identical") else "DIFFERS"))
            else:
                note.append("no archive copy existed")
            lines.append("- `%s` (%d bytes, sha256 %s) -- %s" % (
                r["dest"].split("/")[-1], r["bytes"], r["sha256"], "; ".join(note)))
        lines += ["", "Source paths were `crag_data_backup/results/level_2/<name>`."]
        with io.open(readme, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        ev["readme"] = readme
    return ev


# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    apply_it = "--apply" in args
    os.chdir(REPO)
    t0 = time.time()

    cc_raw = open(CROSSCHECK, "rb").read()
    cc = json.loads(cc_raw)
    cands = [c for c in cc["candidates"] if c["verdict"] == "DELETE"]
    print("cross-check %s  DELETE candidates %d  (%.2f GB claimed)"
          % (cc["created_utc"], len(cands), sum(c["bytes"] for c in cands) / 1e9))

    decl, _five, _six, _fam = declared_index()
    declared = set(norm(p) for p in decl)
    live = live_resolution_paths()
    tracked = git_tracked()

    # --- preconditions -------------------------------------------------------------------
    pre = {}
    part_groups = {}
    for c in cands:
        p = norm(c["path"])
        if "/_dense_repair_patch/" in p and "__p" in p.split("/")[-1]:
            store = p.rsplit("__p", 1)[0]
            part_groups.setdefault(store, []).append(p)
    pre["dense_repair_parts"] = []
    parts_ok = set()
    for store, parts in sorted(part_groups.items()):
        parts = sorted(parts, key=lambda s: int(s.rsplit("__p", 1)[1]))
        ev = check_parts_reproduce_store(store, parts)
        pre["dense_repair_parts"].append(ev)
        print("  parts %-60s %s" % (store, "REPRODUCE store" if ev["ok"]
                                    else "FAIL " + ev.get("error", "")))
        if ev["ok"]:
            parts_ok.update(parts)
    cand_paths = set(norm(c["path"]) for c in cands)
    if "metaqa_temp" in cand_paths:
        pre["metaqa_temp"] = check_metaqa_temp()
        print("  metaqa_temp: %d files, all identical to data/raw/metaqa: %s"
              % (pre["metaqa_temp"]["n_files"], pre["metaqa_temp"]["ok"]))
    if "crag_data_backup" in cand_paths:
        pre["crag_data_backup"] = preserve_backup_results(apply_it)
        b = pre["crag_data_backup"]
        print("  crag_data_backup: %d results files -> %s (%s); %d other files, %d identical to live"
              % (b["n_results"], RESULTS_DEST, "copied+verified" if apply_it else "dry run",
                 b["n_other"], b["n_other_identical_to_live"]))

    # --- per-file plan -------------------------------------------------------------------
    plan, refused, kept_notes = [], [], []
    for c in cands:
        p = norm(c["path"])
        if not os.path.exists(p):
            refused.append({"path": p, "why": "ABSENT"})
            continue
        why = None
        if is_under(p, PROTECTED):
            why = "PROTECTED"
        elif p in declared or any(d.startswith(p + "/") for d in declared):
            why = "DECLARED_BY_LOCKED_RECORD"
        elif p in live or any(l == p or l.startswith(p + "/") or p.startswith(l + "/")
                              for l in live):
            why = "RESOLVED_BY_POINTER_INDEX_OR_RETRIEVAL_CACHE"
        elif "__p" in p.split("/")[-1] and "/_dense_repair_patch/" in p and p not in parts_ok:
            why = "PARTS_DO_NOT_REPRODUCE_STORE"
        elif p == "metaqa_temp" and not pre["metaqa_temp"]["ok"]:
            why = "NOT_IDENTICAL_TO_data/raw/metaqa"
        elif p == "crag_data_backup" and apply_it and not pre["crag_data_backup"]["ok"]:
            why = "RESULTS_NOT_PRESERVED"
        if why:
            refused.append({"path": p, "why": why, "bytes": c["bytes"]})
            continue
        for f in walk_files(p):
            if f in tracked:
                refused.append({"path": f, "why": "GIT_TRACKED"})
                continue
            if f in declared or f in live:
                refused.append({"path": f, "why": "DECLARED_OR_RESOLVED_FILE"})
                continue
            sz = os.path.getsize(f)
            if p.startswith("scratchpad/") and os.path.isdir(p):
                ext = os.path.splitext(f)[1].lower()
                if ext not in BULK_EXT and sz <= SCRATCH_KEEP_MAX_BYTES:
                    kept_notes.append({"path": f, "bytes": sz})
                    continue
            plan.append({"candidate": p, "bucket": c["bucket"], "path": f, "bytes": sz})

    total = sum(x["bytes"] for x in plan)
    print("\nplan: %d files, %.2f GB   refused: %d   scratchpad notes kept: %d (%.1f MB)"
          % (len(plan), total / 1e9, len(refused), len(kept_notes),
             sum(k["bytes"] for k in kept_notes) / 1e6))
    for r in refused[:40]:
        print("  REFUSED %-45s %s" % (r["why"], r["path"]))

    # --- hash, then delete ---------------------------------------------------------------
    print("hashing %.2f GB ..." % (total / 1e9))
    for x in plan:
        x["sha256"] = sha256_file(x["path"])
    deleted = freed = 0
    failed = []
    if apply_it:
        for x in plan:
            try:
                os.remove(x["path"])
                x["deleted"] = True
                deleted += 1
                freed += x["bytes"]
            except OSError as e:
                x["deleted"] = False
                failed.append({"path": x["path"], "error": str(e)})
        # prune the directories the pass emptied -- only inside the candidates, never a parent
        removed_dirs = []
        for c in cands:
            p = norm(c["path"])
            if not os.path.isdir(p):
                continue
            for dp, dn, fn in os.walk(p, topdown=False):
                try:
                    if not os.listdir(dp):
                        os.rmdir(dp)
                        removed_dirs.append(norm(dp))
                except OSError:
                    pass
        print("deleted %d files, freed %.2f GB, removed %d empty dirs, %d failures"
              % (deleted, freed / 1e9, len(removed_dirs), len(failed)))
    else:
        removed_dirs = []
        print("DRY RUN -- nothing deleted. Re-run with --apply.")

    by_bucket = {}
    for x in plan:
        e = by_bucket.setdefault(x["bucket"], {"files": 0, "bytes": 0})
        e["files"] += 1
        e["bytes"] += x["bytes"]

    rec = {
        "RECORD": "CLEANUP_PASS_3",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "STATUS": "APPLIED" if apply_it else "DRY_RUN",
        "_what": "deleted exactly the DELETE-verdict candidates of CLEANUP_REFERENCE_CROSSCHECK "
                 "after re-deriving the safety gate from the source records; per-file sha256 "
                 "ledger of everything removed",
        "BUILDS_ON": {"record": "CLEANUP_REFERENCE_CROSSCHECK.json",
                      "file_sha256": hashlib.sha256(cc_raw).hexdigest(),
                      "created_utc": cc["created_utc"]},
        "gate": {"protected_prefixes": PROTECTED, "protected_why": PROTECTED_WHY,
                 "declared_paths_checked": len(declared),
                 "live_resolution_paths_checked": len(live),
                 "git_tracked_checked": len(tracked)},
        "scratchpad_rule": {"delete_extensions": sorted(BULK_EXT),
                            "delete_if_bytes_over": SCRATCH_KEEP_MAX_BYTES,
                            "kept_notes": kept_notes},
        "preconditions": pre,
        "refused": refused,
        "totals": {"files": len(plan), "bytes": total, "by_bucket": by_bucket,
                   "deleted_files": deleted, "freed_bytes": freed,
                   "failed": failed, "removed_empty_dirs": removed_dirs},
        "deleted": plan,
        "elapsed_s": round(time.time() - t0, 1),
    }
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote %s (%s) in %.0fs" % (OUT, rec["STATUS"], rec["elapsed_s"]))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
