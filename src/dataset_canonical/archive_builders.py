# -*- coding: utf-8 -*-
"""Snapshot the builder and tool sources into the package so the trees stay reproducible-in-principle
without the working tree.

    PYTHONHASHSEED=0 python src/dataset_canonical/archive_builders.py

Every *.py / *.sh under scratchpad/final_canonical_build/ (the Track-B / V1 / V3 builders that the
build_info.json and provenance records name; __pycache__ excluded) and every *.py under
src/dataset_canonical/ (the freeze, verification, handoff-check and Freebase tools) is copied to
    data/final_canonical/_history/builders/<its repo path>
and listed in _history/builders/BUILDERS.json (RECORD BUILDER_SNAPSHOT, self-hashed) with size and
sha256.  The record also states, per dataset, which builder revision its build_info.json names and
whether that exact revision is in the snapshot -- the one that is not (hotpotqa, build_kb.py
a46d745a...) is reported as LOST with the reason, never silently dropped.

Re-runnable: files are overwritten only when their bytes differ; nothing is deleted from an earlier
snapshot (a revision that was archived once stays archived under a versioned name).
freeze_canonical.py pins BUILDERS.json in the freeze; _history/INDEX.json digests every file.
"""
import glob
import hashlib
import io
import json
import os
import shutil
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FC = "data/final_canonical"
HIST = FC + "/_history"
DEST = HIST + "/builders"
SOURCES = [("scratchpad/final_canonical_build", (".py", ".sh")), ("src/dataset_canonical", (".py",))]
DATASETS = ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def record_hash(rec, crlf=True):
    d = dict(rec)
    d.pop("RECORD_SHA256", None)
    d.pop("RECORD_SHA256_LF", None)
    txt = json.dumps(d, indent=1)
    if crlf:
        txt = txt.replace("\n", "\r\n")
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def write_record(p, rec):
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))


def main():
    t0 = time.time()
    os.makedirs(DEST, exist_ok=True)
    files = []
    by_sha = {}
    copied = kept = versioned = 0
    for root, exts in SOURCES:
        for r, dirs, fs in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            for x in sorted(fs):
                if not x.endswith(exts):
                    continue
                src = os.path.join(r, x).replace(os.sep, "/")
                dst = DEST + "/" + src
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                h = sha_file(src)
                if os.path.exists(dst) and sha_file(dst) == h:
                    kept += 1
                elif os.path.exists(dst):
                    # keep the earlier revision under a versioned name, then place the current one
                    old = sha_file(dst)
                    vname = dst + ".rev_" + old[:12]
                    if not os.path.exists(vname):
                        shutil.copyfile(dst, vname)
                        versioned += 1
                    shutil.copyfile(src, dst)
                    copied += 1
                else:
                    shutil.copyfile(src, dst)
                    copied += 1
                e = {"source": src, "archived": dst.replace(os.sep, "/"), "bytes": os.path.getsize(src), "sha256": h}
                files.append(e)
                by_sha[h] = e["archived"]
    # earlier revisions kept under versioned names are part of the snapshot too
    for p in sorted(glob.glob(DEST + "/**/*.rev_*", recursive=True)):
        p = p.replace(os.sep, "/")
        h = sha_file(p)
        files.append({"source": None, "archived": p, "bytes": os.path.getsize(p), "sha256": h, "earlier_revision_of": p.split(".rev_")[0]})
        by_sha.setdefault(h, p)

    revisions = {}
    for ds in DATASETS:
        bi = FC + "/%s/build_info.json" % ds
        if not os.path.exists(bi):
            b = json.load(io.open(FC + "/%s/CANONICAL_V1_BUILD.json" % ds, encoding="utf-8")) if os.path.exists(FC + "/%s/CANONICAL_V1_BUILD.json" % ds) else None
            revisions[ds] = {"build_info": None, "build_record": ("%s/CANONICAL_V1_BUILD.json" % ds) if b else None,
                             "builder": "scratchpad/final_canonical_build/webqsp_v1/ (V1 render + V3 passes; the records under "
                                        "webqsp/ name the passes)" if ds == "webqsp" else None,
                             "status": "NO_BUILD_INFO: the build record names no builder file or digest; the builder sources are archived "
                                       "at their current revision, the revision used at build time was not recorded"}
            continue
        b = json.load(io.open(bi, encoding="utf-8"))
        want = b.get("builder_sha256")
        path = b.get("builder")
        hit = by_sha.get(want)
        entry = {"build_info": "%s/build_info.json" % ds, "builder": path, "builder_sha256": want, "git_commit": b.get("git_commit"),
                 "jsonstream_sha256": b.get("jsonstream_sha256"), "archived_revision": hit}
        if hit:
            entry["status"] = "ARCHIVED"
        else:
            cur = sha_file(path) if path and os.path.exists(path) else None
            entry["current_file_sha256"] = cur
            entry["status"] = "LOST"
            entry["why"] = ("the revision named by build_info.json no longer exists anywhere under the repo: the file was overwritten "
                            "in place by a later build (2wiki, 2026-09-05, revision %s) before any copy was kept; the tree remains "
                            "verified by its own records (nodes_jsonl_sha256 in build_info.json, integrity_report.json, "
                            "query_independence_test.json, the REV2 audits) but is not reproducible from an archived builder"
                            % ((cur or "")[:12]))
        revisions[ds] = entry
    lost = [ds for ds, e in revisions.items() if e.get("status") == "LOST"]
    rec = {
        "RECORD": "BUILDER_SNAPSHOT",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what": "byte copies of every builder / tool source the package's records name (scratchpad/final_canonical_build/**.py|.sh, "
                "src/dataset_canonical/**.py), each digested; per dataset, whether the builder revision its build_info.json names is "
                "in the snapshot",
        "files": len(files), "bytes": int(sum(e["bytes"] for e in files)),
        "copied_this_run": copied, "unchanged": kept, "earlier_revisions_kept": versioned,
        "build_info_revisions": revisions,
        "lost_revisions": lost,
        "inventory": files,
        "seconds": round(time.time() - t0, 1),
    }
    write_record(DEST + "/BUILDERS.json", rec)
    print("archived %d files (%.2f MB): copied %d, unchanged %d, earlier revisions kept %d; lost build_info revisions: %s"
          % (len(files), rec["bytes"] / 1e6, copied, kept, versioned, lost or "none"))
    for ds, e in revisions.items():
        print("  %-9s %s %s" % (ds, e.get("status"), (e.get("builder_sha256") or "")[:12]))


if __name__ == "__main__":
    main()
