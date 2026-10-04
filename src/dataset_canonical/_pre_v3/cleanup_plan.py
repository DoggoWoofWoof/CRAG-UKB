# -*- coding: utf-8 -*-
"""Classify every file under the data trees as KEEP or RECLAIMABLE, and print the plan.

DRY RUN BY DEFAULT. Nothing is deleted without --apply, and --apply refuses to touch
anything on the protected list no matter what the classifier concluded.

WHY THIS IS NOT A ONE-LINER
---------------------------
The obvious cleanup -- "data/canonical is the old tree, data/final_canonical is the new one,
delete the old one" -- is wrong and would destroy the package. data/final_canonical does NOT
contain its own vectors: pointer_index/<model>.npz stores (src, row) pairs that resolve into
data/canonical/<tree>/encodings/, so the two trees are one substrate and must travel
together. LOCKED_5_OF_5 says this outright under READ_THIS_FIRST.not_self_contained.

It is also wrong at tree level in both directions:

  data/canonical/2wiki is the SUPERSEDED 398,354-document tree -- its docs were replaced by
  2wiki_universe (5,989,847). But POINTER_INDEX.queries resolves 2wiki's QUERY vectors out
  of data/canonical/2wiki/encodings/*/queries. Deleting the tree deletes live query
  embeddings. Only its docs side is reclaimable.

  data/canonical/metaqa_{1,2,3}hop look like leftover per-hop splits, and they are not:
  metaqa has no query encodings of its own, and all 407,513 of its query vectors live in
  those three trees.

  data/final_canonical/_dense_repair_patch and _rev2_encoder_patch look like build staging,
  and they are not: POINTER_INDEX resolves src=1 (REV2_PATCH) and the repair redirect into
  exactly those files. They hold the only good copies of 232,085 dense rows -- 89,452 in
  _rev2_encoder_patch (2wiki 87,764 + hotpotqa 1,688) and 142,633 in _dense_repair_patch
  (10 stores, docs and queries; the figure 202,874 that stood here before was rev2 + the three
  docs-side repair stores only). HANDOFF_READINESS.repair_stores carries the per-store ledger.
  The <store>__pNNN/ part directories inside _dense_repair_patch ARE staging (byte-identical
  to the flat store, resolved by nothing) -- labelled in CANONICAL_DEDUP, protected here.

So classification is per-PATH and driven by the reference closure actually recorded in
data/final_canonical/*.json, never by directory name. A path is KEEP if any record mentions
it; RECLAIMABLE only if it is both unreferenced and has a named successor.

THE PROTECTED LIST
------------------
Some paths are never proposed regardless of reference status, because their value is not
reproducible:
  freebase_v3/_acquisition/raw/**        the 31.3 GB raw Freebase mirror = source of record
  freebase_v3/_acquisition/idir/**       the IDIR ZIP; deletion is explicitly still blocked
  freebase_v3/overlay_v1/**              CRAG_FREEBASE_RESOLUTION_OVERLAY_V1, frozen
  freebase_v3/canonical/**               CRAG_FREEBASE_CANONICAL, frozen, incl. metadata CVTs
"""
import fnmatch
import io
import json
import os
import re
import subprocess
import sys
import time

ROOT = "data/final_canonical"
CANON = "data/canonical"

PROTECTED = [
    "data/final_canonical/freebase_v3/_acquisition/raw",
    "data/final_canonical/freebase_v3/_acquisition/idir",
    "data/final_canonical/freebase_v3/_acquisition/facc1",
    "data/final_canonical/freebase_v3/overlay_v1",
    "data/final_canonical/freebase_v3/canonical",
    "data/final_canonical/freebase_v3/inference_overlay_v1",
    "data/final_canonical/freebase_v3/inferred_name_v1",
    "data/final_canonical/freebase_v3/semantic_v1",
]
PROTECTED_WHY = {
    "raw": "the 31.3 GB raw Freebase mirror is the SOURCE OF RECORD and must never be deleted",
    "idir": "the IDIR ZIP is the audit oracle and its deletion is explicitly still blocked",
    "facc1": "FACC1 mentions are an acquired external corpus, not a derived artifact",
    "overlay_v1": "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 is frozen and must not be modified",
    "canonical": "CRAG_FREEBASE_CANONICAL is frozen (35.3 GB, 9/9 integrity checks); includes "
                 "the 460K metadata CVTs which must NOT be removed",
    "inference_overlay_v1": "part of the frozen Freebase name-recovery chain",
    "inferred_name_v1": "part of the frozen Freebase name-recovery chain",
    "semantic_v1": "part of the frozen Freebase name-recovery chain",
}

# path prefix -> (successor, why it is safe to reclaim)
SUPERSEDED = [
    ("data/final_canonical/_superseded_textualization_rev1",
     "data/final_canonical/<ds>/nodes.jsonl (TEXTUALIZATION_REV 2)",
     "the REV1 nodes.jsonl for hotpotqa and 2wiki. REV2 is the substrate and is hash-pinned; "
     "REV1 is reproducible from the source corpora by re-running build.py at REV1. Kept only "
     "as a convenience during the REV2 transition."),
    ("data/final_canonical/_superseded_sidecars",
     "the current per-dataset sidecars",
     "sidecars displaced by a rebuild; each has a live replacement in <ds>/."),
    ("data/final_canonical/2wiki/_superseded_context_union_398354",
     "data/final_canonical/2wiki/nodes.jsonl (5,989,847 nodes)",
     "the 398,354-document context-union view, superseded when 2wiki was rebuilt to the full "
     "Wikipedia universe."),
    ("data/final_canonical/hotpotqa/_legacy_alias_hotpotqa_clean",
     "data/final_canonical/hotpotqa/",
     "an alias shim pointing the old hotpotqa_clean name at the canonical tree."),
    ("data/final_canonical/_stage_2wiki_rev2",
     "data/final_canonical/2wiki/nodes.jsonl",
     "the REV2 staging directory; its output was installed and hash-pinned."),
    ("data/canonical/2wiki/documents.jsonl",
     "data/canonical/2wiki_universe/documents.jsonl",
     "the 398,354-document Phase-C tree. Its DOCS side is superseded by 2wiki_universe. Its "
     "QUERY encodings are still live and are classified separately."),
    ("data/canonical/2wiki/encodings/dense/docs",
     "data/canonical/2wiki_universe/encodings/dense/docs",
     "docs-side dense shards of the superseded 398,354 tree. No pointer resolves here."),
    ("data/canonical/2wiki/encodings/splade/docs",
     "data/canonical/2wiki_universe/encodings/splade/docs",
     "docs-side splade shards of the superseded 398,354 tree. No pointer resolves here."),
    ("data/canonical/webqsp",
     "data/canonical/webqsp_rog_v1",
     "the Phase-C webqsp tree: a DIFFERENT node universe (1,316,466 nodes) sharing no ids with "
     "canonical webqsp, and one of the ten damaged dense channels. Its four keys in "
     "ENCODING_SHARD_HASHES.json are the documented stale ones; WEBQSP_ENCODING_SHARD_HASHES "
     "is the live record. No pointer resolves here."),
]

# reproducible staging: safe to reclaim, rebuildable by a named script
STAGING = [
    ("data/_family_v1/_knn_bundle", "scratchpad/modal_canonical_knn.py --stage compute",
     "the upload bundle for the remote kNN runs. Reproducible from the pointer index."),
    ("data/_family_v1/_knn_out", "scratchpad/modal_canonical_knn.py --stage merge",
     "downloaded kNN merge output. Already installed into <ds>/graph2/ and hash-pinned there."),
    ("data/_family_v1/_remote_census", "scratchpad/modal_store_hashcensus.py",
     "remote shard census JSON; re-derivable in minutes."),
    ("data/final_canonical/_work", None,
     "build scratch from the canonical build (reuse_kb, legacy, qi). Outputs were installed."),
]


def walk_files(base):
    out = {}
    if not os.path.isdir(base):
        return out
    for dp, dn, fn in os.walk(base):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fn:
            p = os.path.join(dp, f).replace("\\", "/")
            try:
                out[p] = os.path.getsize(p)
            except OSError:
                pass
    return out


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def reference_closure():
    """The paths the package actually DEPENDS on, read structurally, not by grepping.

    A mention is not a dependency. STALE_HASH_SWEEP.json alone names 153 files inside
    _superseded_textualization_rev1 -- it is the record whose whole job is to enumerate
    historical copies of a hash, and it classified every one of those hits as HISTORICAL.
    Treating its mentions as live would protect exactly the trees this tool exists to
    reclaim, so the closure is built from the records that DECLARE dependencies:

      tier 1, DEPENDENCY (protects a path):
        LOCKED_*                    the pinned artifact lists
        POINTER_INDEX.json          every store a doc or query pointer resolves into
        RETRIEVAL_CACHE.json        the materialised deep-K caches
        <ds>/graph{,2}/MANIFEST     each family's .npz and the source TSV it was built from

      tier 2, CENSUS (informative, does NOT protect):
        ENCODING_SHARD_HASHES.json / WEBQSP_ENCODING_SHARD_HASHES.json
        These fingerprint the CURRENT bytes of every channel they ever knew about,
        including four webqsp keys their own caveat calls the superseded ones. A census is
        a record of what exists, not a claim that it is needed.
    """
    dep_exact, dep_dirs, census_dirs = set(), set(), set()

    def add(p):
        if not p:
            return
        p = p.replace("\\", "/")
        (dep_dirs if os.path.isdir(p) else dep_exact).add(p.rstrip("/"))

    for fn in ("LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json",
               "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json",
               "LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json"):
        p = "%s/%s" % (ROOT, fn)
        if not os.path.exists(p):
            continue
        d = rj(p)
        for ds, v in (d.get("datasets") or {}).items():
            for a in (v.get("artifacts") or []):
                add(a.get("path"))
            for _f, a in (v.get("added") or {}).items():
                add(ROOT + "/" + a["file"])
        for _k, v in (d.get("package_level_artifacts") or {}).items():
            add(v.get("path"))
        if "webqsp" in d:
            for a in (d["webqsp"].get("artifacts") or []):
                add(a.get("path"))

    pi = rj(ROOT + "/POINTER_INDEX.json")
    for section in ("datasets", "queries"):
        for ds, v in (pi.get(section) or {}).items():
            for ch in ("dense", "splade"):
                for st in ((v.get(ch) or {}).get("stores") or []):
                    add(st.get("path"))

    rc = rj(ROOT + "/RETRIEVAL_CACHE.json") if os.path.exists(ROOT + "/RETRIEVAL_CACHE.json") else {}
    for ds, v in (rc.get("caches") or {}).items():
        for ch, e in (v or {}).items():
            if isinstance(e, dict):
                add(e.get("file"))

    for ds in sorted(os.listdir(ROOT)):
        for tree in ("graph", "graph2"):
            p = "%s/%s/%s/GRAPH_MANIFEST.json" % (ROOT, ds, tree)
            if not os.path.exists(p):
                continue
            for _f, v in (rj(p).get("families") or {}).items():
                if not v.get("present"):
                    continue
                add(v.get("source_tsv"))
                if v.get("file"):
                    add("%s/%s/%s" % (ROOT, ds, v["file"]))

    for fn in ("ENCODING_SHARD_HASHES.json", "WEBQSP_ENCODING_SHARD_HASHES.json"):
        p = "%s/%s" % (ROOT, fn)
        if not os.path.exists(p):
            continue
        for _k, v in (rj(p).get("channels") or {}).items():
            d2 = (v.get("dir") or v.get("path") or "").replace("\\", "/")
            if d2:
                census_dirs.add(d2.rstrip("/"))
    return dep_exact, dep_dirs, census_dirs


def is_under(path, bases):
    for b in bases:
        b = b.rstrip("/")
        if path == b or path.startswith(b + "/"):
            return b
    return None


def classify():
    dep_exact, dep_dirs, census_dirs = reference_closure()
    files = {}
    files.update(walk_files(ROOT))
    files.update(walk_files(CANON))
    files.update(walk_files("data/_family_v1"))

    sup_bases = [s[0] for s in SUPERSEDED]
    stg_bases = [s[0] for s in STAGING]
    sup_meta = {s[0]: s for s in SUPERSEDED}
    stg_meta = {s[0]: s for s in STAGING}

    out = {}
    for p, sz in files.items():
        prot = is_under(p, PROTECTED)
        if prot:
            out[p] = ("PROTECTED", sz, prot, PROTECTED_WHY[os.path.basename(prot)])
            continue
        sup = is_under(p, sup_bases)
        stg = is_under(p, stg_bases)
        depended = (p in dep_exact) or bool(is_under(p, dep_dirs))
        censused = bool(is_under(p, census_dirs))
        if depended:
            # a dependency wins over every superseded rule: if a pointer resolves here, the
            # tree containing it is not fully dead however its directory is named
            out[p] = ("KEEP_DEPENDENCY", sz, sup,
                      ("a POINTER_INDEX store / pinned artifact / graph source resolves here, "
                       "inside a tree marked superseded -- the SUPERSEDED rule for %s is "
                       "docs-side only" % sup) if sup else None)
        elif sup:
            _b, succ, why = sup_meta[sup]
            out[p] = ("RECLAIMABLE_SUPERSEDED", sz, succ,
                      why + ((" Present in a shard census (ENCODING_SHARD_HASHES) but nothing "
                              "depends on it; a census records what exists, not what is needed.")
                             if censused else ""))
        elif stg:
            _b, succ, why = stg_meta[stg]
            out[p] = ("RECLAIMABLE_STAGING", sz, succ, why)
        elif censused:
            out[p] = ("KEEP_CENSUSED", sz, None,
                      "fingerprinted by a shard census but not named by any dependency record "
                      "and not on a superseded path. Kept -- ambiguity is not evidence.")
        else:
            out[p] = ("KEEP_UNCLASSIFIED", sz, None,
                      "not named by any record and not on a known-superseded path. Kept: an "
                      "unclassified file is not evidence of a dead file.")
    return out, dep_exact, dep_dirs, census_dirs


def safety_gate(reclaimable):
    """Four independent proofs that the plan cannot delete anything live.

    The classifier already reasons its way to these, so the gate is redundant BY DESIGN --
    it re-derives the answer from the source records rather than from the classifier's own
    output, so a bug in the classification cannot also pass the check that guards it.
    --apply refuses to run unless all four are clean.
    """
    v = []
    pi = rj(ROOT + "/POINTER_INDEX.json")
    live = set()
    for sec in ("datasets", "queries"):
        for ds, d in (pi.get(sec) or {}).items():
            for ch in ("dense", "splade"):
                for st in ((d.get(ch) or {}).get("stores") or []):
                    live.add(st["path"].replace("\\", "/").rstrip("/"))
    for f in reclaimable:
        if is_under(f, live):
            v.append(("POINTER_STORE", f))

    decl = set()
    for fn in ("LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json",
               "LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json",
               "LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json"):
        d = rj("%s/%s" % (ROOT, fn))
        for ds, x in (d.get("datasets") or {}).items():
            for a in (x.get("artifacts") or []):
                decl.add(a["path"].replace("\\", "/"))
            for _f, a in (x.get("added") or {}).items():
                decl.add(ROOT + "/" + a["file"])
        for _k, x in (d.get("package_level_artifacts") or {}).items():
            decl.add(x["path"].replace("\\", "/"))
        if "webqsp" in d:
            for a in (d["webqsp"].get("artifacts") or []):
                decl.add(a["path"].replace("\\", "/"))
    for f in sorted(set(reclaimable) & decl):
        v.append(("PINNED_ARTIFACT", f))

    for f in reclaimable:
        if is_under(f, PROTECTED):
            v.append(("PROTECTED", f))

    # the three traps this tool exists to avoid, asserted by name
    for t in ("data/canonical/2wiki/encodings/dense/queries",
              "data/canonical/2wiki/encodings/splade/queries",
              "data/canonical/metaqa_1hop/encodings/dense/queries",
              "data/canonical/metaqa_2hop/encodings/dense/queries",
              "data/canonical/metaqa_3hop/encodings/dense/queries",
              "data/final_canonical/_dense_repair_patch",
              "data/final_canonical/_rev2_encoder_patch"):
        for f in reclaimable:
            if is_under(f, [t]):
                v.append(("NAMED_TRAP:" + t, f))
    return v


# ---------------------------------------------------------------------------------------
# repo-root clutter. Deliberately a SEPARATE accounting from the data trees: it is a
# tidiness question worth well under a gigabyte, and folding it into the headline number
# would dress up 0.5 MB of logs as part of a 30 GB reclaim.

# provably transient: a log or a scratch script, regenerable or worthless, and *.log is
# already gitignored so git has never seen most of these
ROOT_TRANSIENT = [
    ("_tmp_*", "scratch scripts and their output from interactive debugging"),
    ("modal_*.log", "Modal run logs; the runs they describe are recorded in results JSON"),
    ("eval_*.log", "eval run logs"),
    ("eval2wiki_full.log", "eval run log"),
    ("phase1_*.log", "Phase-1 router validation logs"),
    ("err*.txt", "captured stderr from a failed run"),
    ("nohup*", "nohup capture"),
]

# looks like clutter, is actually a result. A human decides, and the default is KEEP: an
# unlabelled results blob at the repo root is exactly the kind of file whose only copy this
# is, and a summary JSON may be the number quoted in a paper.
ROOT_REVIEW = [
    ("_pulled", "11.3 MB of scope statistics pulled from a run -- unlabelled, possibly the "
                "only copy. Identify it before deleting it."),
    ("summary_*.json", "run summaries; may be the source of a quoted number"),
    ("overall.json", "aggregate result"),
    ("probe_full_fp_exact.json", "probe result"),
    ("canonical_benchmarks.html", "generated report"),
    ("Review_Paper.zip", "paper submission bundle"),
    ("memory.md", "hand-written notes"),
]


def git_tracked():
    """The tracked set. A tracked file is never classified transient -- deleting it is a
    commit, not a cleanup, and that is the user's call to make explicitly."""
    try:
        out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             encoding="utf-8", timeout=120)
        return {l.strip() for l in out.stdout.splitlines() if l.strip()}
    except (OSError, subprocess.SubprocessError):
        return None      # unknown, so nothing may be called transient


def root_survey():
    tracked = git_tracked()
    rows = {}
    for f in sorted(os.listdir(".")):
        if not os.path.isfile(f):
            continue
        sz = os.path.getsize(f)
        is_tracked = None if tracked is None else (f.replace("\\", "/") in tracked)
        bucket, why = "KEEP", "not clutter, or unrecognised -- left alone"
        for pat, w in ROOT_REVIEW:
            if fnmatch.fnmatch(f, pat):
                bucket, why = "REVIEW", w
                break
        else:
            for pat, w in ROOT_TRANSIENT:
                if fnmatch.fnmatch(f, pat):
                    bucket, why = "TRANSIENT", w
                    break
        if bucket == "TRANSIENT" and is_tracked is not False:
            bucket = "REVIEW"
            why = ("matches a transient pattern but is git-tracked (or tracking could not be "
                   "determined): deleting it is a commit, not a cleanup")
        rows[f] = (bucket, sz, why, is_tracked)
    return rows

def main():
    args = sys.argv[1:]
    apply_it = "--apply" in args
    apply_root = "--apply-root" in args      # separate switch on purpose: --apply must never
    verbose = "--verbose" in args            # be able to sweep files at the repo root
    out_path = None
    for a in args:
        if a.startswith("--out="):
            out_path = a.split("=", 1)[1]

    cls, dep_exact, dep_dirs, census_dirs = classify()
    root = root_survey()
    buckets = {}
    for p, (b, sz, succ, why) in cls.items():
        e = buckets.setdefault(b, {"n": 0, "bytes": 0, "items": []})
        e["n"] += 1
        e["bytes"] += sz
        e["items"].append((p, sz, succ, why))

    print("dependency closure: %d exact paths + %d store dirs   (census-only dirs: %d)"
          % (len(dep_exact), len(dep_dirs), len(census_dirs)))
    print("%-30s %7s %12s" % ("bucket", "files", "GB"))
    order = ["KEEP_DEPENDENCY", "KEEP_CENSUSED", "KEEP_UNCLASSIFIED", "PROTECTED",
             "RECLAIMABLE_SUPERSEDED", "RECLAIMABLE_STAGING"]
    for b in order:
        e = buckets.get(b)
        if e:
            print("%-40s %7d %12.2f" % (b, e["n"], e["bytes"] / 1e9))
    recl = sum(buckets.get(b, {"bytes": 0})["bytes"]
               for b in ("RECLAIMABLE_SUPERSEDED", "RECLAIMABLE_STAGING"))
    print("\nRECLAIMABLE TOTAL: %.2f GB" % (recl / 1e9))

    for b in ("RECLAIMABLE_SUPERSEDED", "RECLAIMABLE_STAGING", "KEEP_DEPENDENCY"):
        e = buckets.get(b)
        if not e:
            continue
        if b == "KEEP_DEPENDENCY":
            # only the overlap with a superseded tree matters: those are the paths a cleanup
            # driven by directory names would have deleted. The other ~2500 are unremarkable.
            items = [it for it in e["items"] if it[2]]
            if not items:
                print("\n=== KEEP_DEPENDENCY INSIDE A SUPERSEDED TREE: none ===")
                print("  No dependency path falls inside a superseded prefix, because the "
                      "SUPERSEDED rules are scoped to the docs side only -- e.g. "
                      "data/canonical/2wiki/encodings/dense/docs is a rule, "
                      "data/canonical/2wiki is deliberately NOT, since its queries/ side "
                      "still serves 192,606 live query vectors.")
                continue
            print("\n=== KEEP_DEPENDENCY INSIDE A SUPERSEDED TREE (the traps) ===")
            trap = {}
            for pth, sz, sup, why in items:
                g = trap.setdefault(sup, {"n": 0, "bytes": 0, "why": why, "ex": []})
                g["n"] += 1
                g["bytes"] += sz
                if len(g["ex"]) < 2:
                    g["ex"].append(pth)
            for sup, g in sorted(trap.items(), key=lambda kv: -kv[1]["bytes"]):
                print("  %8.2f GB  %5d files  under %s" % (g["bytes"] / 1e9, g["n"], sup))
                print("      %s" % g["why"])
                for x in g["ex"]:
                    print("        e.g. %s" % x)
            continue
        print("\n=== %s ===" % b)
        by = {}
        for p, sz, succ, why in e["items"]:
            key = succ or "-"
            g = by.setdefault(key, {"n": 0, "bytes": 0, "why": why, "ex": []})
            g["n"] += 1
            g["bytes"] += sz
            if len(g["ex"]) < 3:
                g["ex"].append(p)
        for succ, g in sorted(by.items(), key=lambda kv: -kv[1]["bytes"]):
            print("  %8.2f GB  %5d files" % (g["bytes"] / 1e9, g["n"]))
            print("      successor: %s" % succ)
            print("      why: %s" % g["why"])
            for x in g["ex"]:
                print("        e.g. %s" % x)
            if verbose:
                for p, sz, s2, w2 in sorted(e["items"]):
                    if (s2 or "-") == succ:
                        print("           %12d %s" % (sz, p))

    report = {
        "RECORD": "CANONICAL_CLEANUP_PLAN",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "DRY_RUN": not apply_it,
        "dependency_closure": {"exact_paths": len(dep_exact), "store_dirs": len(dep_dirs),
                               "census_only_dirs": len(census_dirs)},
        "buckets": {b: {"n_files": e["n"], "bytes": e["bytes"]} for b, e in buckets.items()},
        "reclaimable_bytes": recl,
        "protected": PROTECTED,
        "superseded_rules": [{"prefix": a, "successor": b, "why": c} for a, b, c in SUPERSEDED],
        "staging_rules": [{"prefix": a, "rebuild_with": b, "why": c} for a, b, c in STAGING],
        "repo_root_survey": {
            "NOTE": ("Accounted SEPARATELY from the data trees and NOT included in "
                     "reclaimable_bytes. This is tidiness, not space: the whole repo root is "
                     "~12 MB and *.log is already gitignored. --apply does not touch it; "
                     "--apply-root does, and only the TRANSIENT bucket."),
            "buckets": {b: {"n_files": sum(1 for v in root.values() if v[0] == b),
                            "bytes": sum(v[1] for v in root.values() if v[0] == b)}
                        for b in ("TRANSIENT", "REVIEW", "KEEP")},
            "files": [{"path": f, "bucket": v[0], "bytes": v[1], "why": v[2],
                       "git_tracked": v[3]}
                      for f, v in sorted(root.items(), key=lambda kv: -kv[1][1])
                      if v[0] != "KEEP"],
        },
        "reclaimable_files": sorted(
            [{"path": p, "bytes": sz, "successor": succ, "bucket": b}
             for p, (b, sz, succ, why) in cls.items()
             if b.startswith("RECLAIMABLE")], key=lambda d: -d["bytes"]),
    }
    rt = [(f, v) for f, v in root.items() if v[0] == "TRANSIENT"]
    rr = [(f, v) for f, v in root.items() if v[0] == "REVIEW"]
    print("")
    print("=== REPO ROOT (separate accounting, NOT part of the total above) ===")
    print("  TRANSIENT %3d files %8.2f MB   logs and scratch scripts, all untracked"
          % (len(rt), sum(v[1] for _f, v in rt) / 1e6))
    print("  REVIEW    %3d files %8.2f MB   result blobs -- identify before deleting"
          % (len(rr), sum(v[1] for _f, v in rr) / 1e6))
    for f, v in sorted(rr, key=lambda kv: -kv[1][1])[:4]:
        print("      %8.2f MB  %-28s %s" % (v[1] / 1e6, f, v[2][:70]))
    print("  ~12 MB in total, so this is tidiness, not space. --apply does not touch it; "
          "--apply-root does, and only the TRANSIENT bucket.")

    targets = [p for p, (b, sz, succ, why) in cls.items() if b.startswith("RECLAIMABLE")]
    viol = safety_gate(targets)
    print("\nSAFETY GATE: %d violations across 4 independent checks" % len(viol))
    for kind, f in viol[:20]:
        print("  !! %-28s %s" % (kind, f))
    report["safety_gate"] = {"violations": len(viol),
                             "detail": [{"check": k, "path": f} for k, f in viol]}
    if out_path:
        with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(report, indent=1, ensure_ascii=False))
        print("wrote %s" % out_path)

    if apply_root:
        n = fr = 0
        for f, v in rt:
            if v[3] is not False:            # untracked only, re-checked at delete time
                print("  REFUSING (tracked or unknown): %s" % f)
                continue
            try:
                os.remove(f)
                n += 1
                fr += v[1]
            except OSError as e:
                print("  failed %s: %s" % (f, e))
        print("repo root: deleted %d transient files, freed %.2f MB" % (n, fr / 1e6))

    if not apply_it:
        print("\nDRY RUN -- nothing deleted. Re-run with --apply to reclaim the two "
              "RECLAIMABLE buckets.")
        return 0
    if viol:
        print("\nREFUSING TO APPLY: the safety gate found live paths in the plan.")
        return 2

    freed = n = 0
    for p, (b, sz, succ, why) in sorted(cls.items()):
        if not b.startswith("RECLAIMABLE"):
            continue
        if is_under(p, PROTECTED):          # belt and braces: never delete a protected path
            print("REFUSING (protected): %s" % p)
            continue
        try:
            os.remove(p)
            freed += sz
            n += 1
        except OSError as e:
            print("  failed %s: %s" % (p, e))
    print("deleted %d files, freed %.2f GB" % (n, freed / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
