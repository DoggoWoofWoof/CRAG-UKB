"""RETRIEVAL_CACHE.json understates the package, and it breaks a frozen size pin.

THREE THINGS ARE WRONG WITH THE RECORD AS IT STANDS

  1  NOT_MATERIALISED still lists hotpotqa and 2wiki.  Both now have caches, so the record
     asserts the absence of files that exist.  Its stated reason is superseded too: it says
     "a cost decision, not a technical obstacle ... ~13.9 h and ~29.1 h", and those two dense
     caches were in fact produced in 21 s and 37 s.  The estimate was not wrong about THIS
     machine -- 33.8 effective GFLOPS, no GPU -- it was an estimate about the wrong machine.

  2  verify_manifest.py --sizes reports RETRIEVAL_CACHE.json declared=9580 disk=10206,
     pinned_by LOCKED_5_OF_5:pkg.  That is a REAL failure and this work caused it: a webqsp
     entry was appended to a file whose bytes and sha256 a frozen record pins.

  3  The three post-freeze entries are not comparable to the frozen three.  metaqa, musique
     and squad each carry a sha256 and a verification block -- PASS, content_mismatches,
     exact_sequence_reproduced, order_only_differences.  webqsp's entry, appended by the
     builder, carries neither, so "webqsp is in the record" has not meant "webqsp was checked".

WHY THE PIN BREAK IS ADJUDICATED RATHER THAN TOLERATED

  The package already carries three additive drifts of this exact shape -- HANDOFF.md,
  ID_BRIDGE.json, POINTER_INDEX.json -- and each was adjudicated by DEMONSTRATION, not by
  adding a tolerance: strip what was appended, re-serialise at the frozen convention, and
  reproduce the frozen artifact.  The same demonstration runs here on every invocation:

     drop the caches entries for the three datasets added after the freeze, restore
     NOT_MATERIALISED and created_utc from _FROZEN_ORIGINAL -- which this update carries
     verbatim inside the file, so the recipe needs nothing that lives outside the artifact it
     reproduces -- drop the keys this update added, serialise at indent=1 with CRLF

  LOCKED_5_OF_5 pins bytes AND sha256, so this reproduces a HASH, not merely a length.  This
  script REFUSES to write if the hash does not come back: if it ever stops reproducing,
  something other than the additions changed and the drift is no longer benign.  That is the
  point of recording the recipe rather than the verdict.

WHAT IS TOUCHED AND WHAT IS NOT

  The three frozen datasets are re-measured from disk, COMPARED against the record, and then
  left byte-for-byte alone.  Rewriting them would break the reconstruction for a reason
  unrelated to the additions and would destroy the only thing the pin is good for.  A
  disagreement between record and disk is REPORTED, never silently repaired.

  The three post-freeze datasets are written in full from the artifacts themselves -- per this
  record's own RECORD_REBUILT_FROM_DISK rule, which exists because two builders once raced and
  rewrote this file from stale copies.  They get the sha256 the frozen three have, and their
  verification verdict is carried in from HEAVY_CACHE_VERIFICATION.json rather than restated,
  so there is one place where a cache's verification status lives.

  It REFUSES to write if any of the twelve cells is missing.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/retrieval_cache_record_update.py [--apply]
"""

import copy
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
REC = os.path.join(FC, "RETRIEVAL_CACHE.json")
LOCK5 = os.path.join(FC, "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json")
HEAVY = os.path.join(FC, "HEAVY_CACHE_VERIFICATION.json")
OUT = os.path.join(FC, "RETRIEVAL_CACHE_DRIFT_ADJUDICATION.json")
BS = chr(92)

MODELS = ["dense", "splade"]
FROZEN_DATASETS = ["metaqa", "musique", "squad"]      # in the 9,580-byte frozen file
ADDED_AFTER_FREEZE = ["webqsp", "hotpotqa", "2wiki"]  # appended since
DATASETS = FROZEN_DATASETS + ADDED_AFTER_FREEZE
FROZEN_NOT_MATERIALISED = ["hotpotqa", "2wiki"]


def rel(p):
    return os.path.relpath(p, ROOT).replace(BS, "/")


def shab(b):
    return hashlib.sha256(b).hexdigest()


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def frozen_pin():
    L = json.load(io.open(LOCK5, encoding="utf-8"))
    hit = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("path") == "data/final_canonical/RETRIEVAL_CACHE.json":
                hit.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(L)
    if not hit:
        raise SystemExit("LOCKED_5_OF_5 does not pin RETRIEVAL_CACHE.json -- refusing to "
                         "adjudicate a pin that cannot be found")
    return hit[0]


def serialise(obj):
    """The file's own convention, read off the file rather than assumed: indent=1, CRLF."""
    return json.dumps(obj, indent=1, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8")


def _backends_from_disk():
    """Read each cache's own meta for how it was built. Reported, never asserted.

    A merge record written by merge_splade_parts.py carries backend_by_part and
    backend_row_counts; a single-pass build carries one backend string. Either way the claim
    comes off the artifact, so this record cannot drift from what was actually run.
    """
    out = {}
    for ds in FROZEN_DATASETS + ADDED_AFTER_FREEZE:
        for m in MODELS:
            p = os.path.join(FC, ds, "retrieval_cache", "%s_top1000.meta.json" % m)
            if not os.path.isfile(p):
                continue
            try:
                d = json.load(io.open(p, encoding="utf-8"))
            except Exception as e:
                out["%s.%s" % (ds, m)] = {"unreadable": str(e)}
                continue
            cell = {k: d[k] for k in ("backend", "backend_row_counts", "merged_from")
                    if k in d}
            if "backend_by_part" in d:
                cell["n_parts_by_backend"] = {}
                for _, b in d["backend_by_part"].items():
                    cell["n_parts_by_backend"][b] = cell["n_parts_by_backend"].get(b, 0) + 1
            if cell:
                out["%s.%s" % (ds, m)] = cell
    return out


def reconstruct(cur, pin):
    """Strip the post-freeze additions and check the frozen HASH comes back."""
    e = copy.deepcopy(cur)
    dropped = [k for k in ADDED_AFTER_FREEZE if k in e.get("caches", {})]
    for k in dropped:
        e["caches"].pop(k)
    # The frozen NOT_MATERIALISED block and created_utc are carried verbatim inside the file
    # under _FROZEN_ORIGINAL, so this recipe needs no knowledge that lives outside the artifact
    # it is reproducing.  Restoring only the `datasets` list is not enough: the replacement
    # block has different keys, and those bytes count.
    orig = e.pop("_FROZEN_ORIGINAL", None)
    if orig:
        e["NOT_MATERIALISED"] = copy.deepcopy(orig["NOT_MATERIALISED"])
        e["created_utc"] = orig["created_utc"]
    else:
        e.setdefault("NOT_MATERIALISED", {})["datasets"] = list(FROZEN_NOT_MATERIALISED)
    for k in ("FROZEN_SIZE_DRIFT", "BACKENDS", "SUPERSEDES", "VERIFICATION_LIVES_IN",
              "updated_utc"):
        e.pop(k, None)
    b = serialise(e)
    return {"recipe": "drop caches entries %s; restore NOT_MATERIALISED and created_utc from "
                      "_FROZEN_ORIGINAL (carried verbatim in the file); drop the keys this "
                      "update added (_FROZEN_ORIGINAL, updated_utc, BACKENDS, "
                      "VERIFICATION_LIVES_IN, FROZEN_SIZE_DRIFT); serialise indent=1, CRLF"
                      % (ADDED_AFTER_FREEZE,),
            "restored_from_file": bool(orig),
            "dropped": dropped,
            "reconstructed_bytes": len(b),
            "reconstructed_sha256": shab(b),
            "frozen_bytes": pin.get("bytes"),
            "frozen_sha256": pin.get("sha256"),
            "bytes_match": len(b) == pin.get("bytes"),
            "sha256_match": shab(b) == pin.get("sha256"),
            "VERDICT": "ADDITIVE_ONLY" if shab(b) == pin.get("sha256") else "NOT_EXPLAINED"}


def cache_path(ds, model):
    return os.path.join(FC, ds, "retrieval_cache", "%s_top1000.npz" % model)


def measure(ds, model, prior, heavy, want_sha):
    p = cache_path(ds, model)
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    ids, sc = z["ids"], z["scores"]
    e = {"n_queries": int(ids.shape[0]), "K": int(ids.shape[1]),
         "score_min": float(sc.min()), "score_max": float(sc.max()),
         "top1_score_median": float(np.median(sc[:, 0].astype(np.float32))),
         "file": rel(p), "bytes": os.path.getsize(p),
         "ids_dtype": str(ids.dtype), "scores_dtype": str(sc.dtype)}
    del z, ids, sc

    mp = os.path.join(FC, ds, "retrieval_cache", "%s_top1000.meta.json" % model)
    rep = json.load(io.open(mp, encoding="utf-8")) if os.path.isfile(mp) else {}
    pe = (prior.get("caches", {}).get(ds, {}) or {}).get(model, {}) or {}

    nd = rep.get("n_docs", pe.get("n_docs"))
    if nd is None:
        sys.path.insert(0, FC)
        from pointer_resolver import CanonicalEmbeddings
        nd = len(CanonicalEmbeddings(ds, model, kind="docs", root=FC))
    e["n_docs"] = int(nd)

    if want_sha:
        e["sha256"] = shaf(p)
    e["seconds_REPORTED"] = rep.get("seconds", pe.get("seconds"))
    e["backend_REPORTED"] = rep.get("backend", pe.get("backend", "local cpu, numpy/scipy"))
    e["_reported_not_measured"] = ("seconds and backend cannot be read off an .npz; every "
                                   "other field here was")

    cell = (heavy.get("results", {}) or {}).get("%s.%s" % (ds, model), {}) or {}
    if cell.get("verdict"):
        e["verification"] = {
            "record": "data/final_canonical/HEAVY_CACHE_VERIFICATION.json",
            "verdict": cell.get("verdict"),
            "sampled_queries": cell.get("sampled"),
            "set_overlap_at_1000": cell.get("overlap_at_1000"),
            "containment_of_top10_in_cached_1000": cell.get(
                "containment_at_10_in_cached_1000"),
            "positionwise_exact_id_fraction": cell.get("positionwise_exact_id_fraction"),
            "max_abs_score_delta": cell.get("max_abs_score_delta"),
            "whole_artifact_invariants_PASS": (cell.get("whole_artifact_invariants") or {}
                                               ).get("PASS"),
            "_method": "the exact top-1000 for a seeded query sample recomputed LOCALLY over "
                       "the full corpus, streamed, and compared as a SET at depth 10/100/1000 "
                       "-- not positionwise, because float16 storage makes tie order "
                       "unrecoverable from the file"}
    else:
        e["verification"] = {"verdict": "NOT_VERIFIED",
                             "_why": "no cell for this cache in "
                                     "HEAVY_CACHE_VERIFICATION.json"}
    return e


def main():
    apply_ = "--apply" in sys.argv
    cur = json.loads(io.open(REC, "rb").read().decode("utf-8"))
    pin = frozen_pin()
    heavy = json.load(io.open(HEAVY, encoding="utf-8")) if os.path.isfile(HEAVY) else {}

    before = reconstruct(cur, pin)
    print("PRE-UPDATE reconstruction of the frozen file")
    print("   dropped %-28s %d bytes  sha %s"
          % (before["dropped"], before["reconstructed_bytes"], before["reconstructed_sha256"]))
    print("   frozen  %-28s %d bytes  sha %s"
          % ("", before["frozen_bytes"], before["frozen_sha256"]))
    print("   bytes_match=%s sha256_match=%s  %s"
          % (before["bytes_match"], before["sha256_match"], before["VERDICT"]))
    print("")

    new = copy.deepcopy(cur)
    caches, missing, disagree, kept, added = {}, [], [], [], []
    meas = {}
    for ds in DATASETS:
        frozen = ds in FROZEN_DATASETS
        blk = {}
        for m in MODELS:
            e = measure(ds, m, cur, heavy, want_sha=not frozen)
            if e is None:
                missing.append("%s/%s" % (ds, m))
                continue
            meas["%s.%s" % (ds, m)] = e
            old = (cur.get("caches", {}).get(ds, {}) or {}).get(m)
            if frozen and old is not None:
                for f in ("n_queries", "n_docs", "K", "bytes"):
                    if f in old and old[f] != e[f]:
                        disagree.append("%s/%s: %s recorded %r, disk %r"
                                        % (ds, m, f, old[f], e[f]))
                blk[m] = old
                kept.append("%s.%s" % (ds, m))
            else:
                blk[m] = e
                added.append("%s.%s" % (ds, m))
        if blk:
            caches[ds] = blk

    print("%-9s %-7s %9s %9s %5s %12s %14s %-18s %-11s %s"
          % ("dataset", "model", "queries", "docs", "K", "top1_median", "bytes", "backend",
             "verified", "entry"))
    for ds in DATASETS:
        for m in MODELS:
            e = meas.get("%s.%s" % (ds, m))
            if e is None:
                print("%-9s %-7s   MISSING" % (ds, m))
                continue
            iskept = "%s.%s" % (ds, m) in kept
            if iskept:
                # a frozen entry carries its verification INLINE, written by the resident-corpus
                # verifier; HEAVY_CACHE_VERIFICATION has no cell for it and reporting
                # NOT_VERIFIED here would be simply false
                old = cur["caches"][ds][m]
                vv = ("PASS(inline)" if old.get("PASS") is True
                      else "FAIL(inline)" if old.get("PASS") is False else "no verdict")
            else:
                vv = e["verification"]["verdict"]
            print("%-9s %-7s %9d %9d %5d %12.4f %14s %-18s %-12s %s"
                  % (ds, m, e["n_queries"], e["n_docs"], e["K"], e["top1_score_median"],
                     format(e["bytes"], ","), str(e["backend_REPORTED"])[:18], vv,
                     "kept" if iskept else "written"))
    print("")
    print("frozen entries kept verbatim: %d   post-freeze entries written: %d"
          % (len(kept), len(added)))
    if disagree:
        print("DISAGREEMENT between record and disk -- reported, NOT silently rewritten:")
        for x in disagree:
            print("   !! %s" % x)

    new["caches"] = caches
    # created_utc means when the record was created, so it stays; the edit gets its own field.
    new["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    new["_FROZEN_ORIGINAL"] = {
        "_what": "the two fields this update replaces, kept verbatim so the frozen artifact "
                 "can be reproduced from this file alone, without consulting anything else.",
        "created_utc": cur["created_utc"],
        "NOT_MATERIALISED": copy.deepcopy(cur.get("NOT_MATERIALISED", {}))}
    new["NOT_MATERIALISED"] = {
        "datasets": [],
        "was": FROZEN_NOT_MATERIALISED,
        "why_it_changed": "both now have caches. The frozen reason -- 'a cost decision, not a "
                          "technical obstacle ... hotpotqa ~13.9 h, 2wiki ~29.1 h' -- was an "
                          "estimate for THIS machine at 33.8 effective GFLOPS with no GPU, and "
                          "it was not wrong about this machine. The dense caches were produced "
                          "in 21 s and 37 s on a rented H100 and the splade caches on rented "
                          "CPU in 24 query shards.",
        "nothing_depends_on_them_STILL_HOLDS": "unchanged. The caches remain an accelerator "
                                               "over the frozen embeddings; any consumer can "
                                               "recompute the same exact top-K from the "
                                               "pointer index."}
    new["BACKENDS"] = {
        "note": "the corpus, the queries, the encoders and the ranking key are identical "
                "across backends; only the summation order of the dot product differs. Every "
                "stored operand is a float16 value, exactly representable in fp32 and in TF32, "
                "so the operands are identical and accumulation order is the only difference "
                "-- the same class of difference as numpy on another BLAS, which the builder's "
                "own docstring already concedes.",
        "measured_not_assumed": "HEAVY_CACHE_VERIFICATION.json recomputes a seeded query "
                                "sample LOCALLY against the full corpus for every "
                                "remotely-built cache and reports the set overlap. Whether "
                                "accumulation order reaches the artifact is measured there, "
                                "not asserted here.",
        "per_artifact": _backends_from_disk(),
        "MIXED_PROVENANCE_WARNING": "2wiki's splade cache is the one artifact in the package "
                                    "whose rows were not all produced in the same place: 13 of "
                                    "24 query shards on rented CPU before that workspace was "
                                    "disabled mid-run, the remaining 11 locally after every "
                                    "workspace in the pool was found to have exceeded its "
                                    "spend limit. The merge is sound: a part's [lo,hi) is a "
                                    "pure function of (qshard, n_qshards) and a score-matrix "
                                    "row depends on one query only, so shards built in "
                                    "different places tile the query axis with no overlap and "
                                    "no gap. Recorded because no later check can recover it -- "
                                    "the bytes are indistinguishable.",
        "WHAT_A_LOCAL_VERIFICATION_DOES_AND_DOES_NOT_PROVE":
            "verify_heavy_caches.py checks a cache by recomputing it HERE. For a "
            "remotely-built cache that is a cross-implementation check; for a locally-built "
            "one it is a reproducibility check, which is weaker. This is NOT special to 2wiki: "
            "read `per_artifact` above and the partition is remote-built = hotpotqa dense and "
            "splade, 2wiki dense, and 13 of 2wiki splade's 24 shards; locally built = metaqa, "
            "musique, squad (verified inline at build time under LOCKED_5_OF_5), webqsp both "
            "channels, and the other 11 of 2wiki splade's shards. The whole-artifact "
            "invariants -- no id out of range, no duplicate id in a row, scores monotone "
            "non-increasing across every row of the full artifact -- are independent of where "
            "a row was built and are checked over every row, not a sample."}
    new["VERIFICATION_LIVES_IN"] = {
        "frozen_three": "their verification fields are inline, written by "
                        "scratchpad/verify_retrieval_cache.py, which holds the whole corpus "
                        "resident and so cannot run on the heavy corpora",
        "post_freeze_three": "data/final_canonical/HEAVY_CACHE_VERIFICATION.json, whose "
                             "verifier streams the corpus in blocks; each entry's verification "
                             "block here is carried in from there rather than restated, so a "
                             "cache's verification status has one home"}
    new["FROZEN_SIZE_DRIFT"] = {
        "what": "LOCKED_5_OF_5 pins this file's bytes (9580) and sha256. Appending cache "
                "entries for the three datasets built after the freeze changes both.",
        "why_that_is_not_tampering": "the package already carries three drifts of this exact "
                                     "shape (HANDOFF.md, ID_BRIDGE.json, POINTER_INDEX.json), "
                                     "each adjudicated by reproducing the frozen artifact "
                                     "rather than by widening a tolerance.",
        "reconstruction_before_this_update": before,
        "how_to_re_verify": "src/dataset_canonical/retrieval_cache_record_update.py runs the "
                            "reconstruction on every invocation and refuses to write unless "
                            "the frozen sha256 comes back"}

    after = reconstruct(new, pin)
    new["FROZEN_SIZE_DRIFT"]["reconstruction_after_this_update"] = after
    print("")
    print("POST-UPDATE reconstruction: %d bytes  sha %s"
          % (after["reconstructed_bytes"], after["reconstructed_sha256"]))
    print("   bytes_match=%s sha256_match=%s  %s"
          % (after["bytes_match"], after["sha256_match"], after["VERDICT"]))

    adj = {"RECORD": "RETRIEVAL_CACHE_DRIFT_ADJUDICATION",
           "_what": "adjudicates the verify_manifest --sizes failure on "
                    "data/final_canonical/RETRIEVAL_CACHE.json by reproducing the frozen "
                    "artifact's sha256 from the current file, and records that "
                    "NOT_MATERIALISED no longer describes the package.",
           "pinned_by": pin,
           "reconstruction_before_this_update": before,
           "reconstruction_after_this_update": after,
           "cells_present": sorted(meas.keys()),
           "cells_missing": missing,
           "frozen_entries_kept_verbatim": sorted(kept),
           "post_freeze_entries_written": sorted(added),
           "record_vs_disk_disagreements": disagree,
           "why_frozen_entries_are_kept": "an entry already in the frozen file is checked "
                                          "against the .npz on disk and then left "
                                          "byte-for-byte alone. Rewriting it would break the "
                                          "reconstruction for a reason unrelated to the "
                                          "additions, and would destroy the only thing the "
                                          "frozen pin is good for.",
           "applied": bool(apply_ and not missing and after["sha256_match"]),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(adj, indent=1, ensure_ascii=False).replace("\n", "\r\n"))
    print("wrote %s" % rel(OUT))

    if missing:
        print("")
        print("REFUSING to write RETRIEVAL_CACHE.json: %d of %d cells missing -- %s"
              % (len(missing), len(DATASETS) * len(MODELS), ", ".join(missing)))
        print("   the record must not claim six datasets while a cell is absent.")
        return 1
    if not after["sha256_match"]:
        print("")
        print("REFUSING to write RETRIEVAL_CACHE.json: the reconstruction does NOT reproduce "
              "the frozen sha256, so the drift is not explained by the additions alone.")
        return 1
    if apply_:
        b = serialise(new)
        with io.open(REC, "wb") as f:
            f.write(b)
        print("wrote %s  %d bytes  sha %s" % (rel(REC), len(b), shab(b)))
    else:
        print("DRY RUN -- re-run with --apply to write RETRIEVAL_CACHE.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
