"""Fill the three absent retrieval_cache slots in UKB_COMMON_MANIFEST.json.

WHAT IS WRONG

  Three of the six retrieval_cache slots read:

      {"present": false, "reason": "no retrieval cache was built for this dataset. The cache
       is a convenience over the substrate, not part of it -- it can be rebuilt from the
       pointer index at any time."}

  The second sentence is still true and stays.  The first is now false: hotpotqa, 2wiki and
  webqsp all have caches at K=1000 for both channels.

THE PROVENANCE OF THESE THREE IS GENUINELY DIFFERENT, AND IS NOT PAPERED OVER

  This manifest's own rule is that it is DERIVED, not an authority: "every hash is copied from
  the freeze named in pinned_by, never recomputed, so the manifest cannot silently disagree
  with a freeze."  The three new caches are in NO freeze.  There is no freeze to copy from, so
  their hashes are necessarily computed here -- which is exactly the thing the rule exists to
  prevent, and writing them in as though they were freeze-backed would make three slots look
  like the other three when they are not.

  So each new slot carries an explicit `provenance` block naming what does vouch for it --
  RETRIEVAL_CACHE.json for the measurement, HEAVY_CACHE_VERIFICATION.json for the
  verification, FROZEN_CAVEAT_SUPERSESSION.json for the authority to add it after a freeze --
  and stating plainly that no LOCKED_* record pins it.  A reader can then tell the two classes
  of slot apart, which is the property the rule was protecting.

THE THREE EXISTING SLOTS ARE CHECKED, NOT REWRITTEN

  Their sha256 is recomputed from disk and COMPARED against the manifest.  A disagreement is
  reported and makes this exit non-zero; it is never silently repaired, because a manifest
  that disagrees with disk is evidence about one of them and guessing which would destroy it.
  That is ~7 GB of hashing and it is the point of the exercise.

  It REFUSES to write unless all twelve cells exist and all six new ones are verified
  CONSISTENT.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/manifest_cache_slots.py [--apply]
"""

import hashlib
import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
MAN = os.path.join(FC, "UKB_COMMON_MANIFEST.json")
HEAVY = os.path.join(FC, "HEAVY_CACHE_VERIFICATION.json")
OUT = os.path.join(FC, "MANIFEST_CACHE_SLOT_FILL.json")
BS = chr(92)

WAS_ABSENT = ["hotpotqa", "2wiki", "webqsp"]
WAS_PRESENT = ["metaqa", "musique", "squad"]
MODELS = ["dense", "splade"]


def rel(p):
    return os.path.relpath(p, ROOT).replace(BS, "/")


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def channel(ds, model):
    p = os.path.join(FC, ds, "retrieval_cache", "%s_top1000.npz" % model)
    if not os.path.isfile(p):
        return None
    z = np.load(p)
    ids = z["ids"]
    n, k = int(ids.shape[0]), int(ids.shape[1])
    del z, ids
    return {"file": rel(p), "bytes": os.path.getsize(p), "sha256": shaf(p),
            "K": k, "n_queries": n}


def main():
    apply_ = "--apply" in sys.argv
    raw = io.open(MAN, "rb").read()
    man = json.loads(raw.decode("utf-8"))
    heavy = json.load(io.open(HEAVY, encoding="utf-8")) if os.path.isfile(HEAVY) else {}
    res = heavy.get("results", {}) or {}

    missing, unverified, mismatch, filled, checked = [], [], [], [], []

    print("CHECKING the three slots that were already present (recomputing sha256 from disk)")
    for ds in WAS_PRESENT:
        slot = man["datasets"][ds]["retrieval_cache"]
        for m in MODELS:
            ch = channel(ds, m)
            if ch is None:
                missing.append("%s/%s" % (ds, m))
                print("   %-9s %-7s MISSING ON DISK" % (ds, m))
                continue
            rec = (slot.get("channels", {}) or {}).get(m, {}) or {}
            bad = [f for f in ("bytes", "sha256", "K", "n_queries")
                   if f in rec and rec[f] != ch[f]]
            if bad:
                mismatch.append("%s/%s: %s disagree (manifest %s, disk %s)"
                                % (ds, m, ",".join(bad),
                                   {f: rec[f] for f in bad}, {f: ch[f] for f in bad}))
            checked.append("%s.%s" % (ds, m))
            print("   %-9s %-7s sha %s  %s" % (ds, m, ch["sha256"][:16],
                                               "MISMATCH" if bad else "agrees with manifest"))

    print("")
    print("FILLING the three slots that read present=false")
    for ds in WAS_ABSENT:
        chans = {}
        for m in MODELS:
            ch = channel(ds, m)
            if ch is None:
                missing.append("%s/%s" % (ds, m))
                print("   %-9s %-7s MISSING ON DISK" % (ds, m))
                continue
            cell = res.get("%s.%s" % (ds, m), {}) or {}
            if cell.get("verdict") != "CONSISTENT":
                unverified.append("%s.%s" % (ds, m))
            chans[m] = ch
            print("   %-9s %-7s %14s  sha %s  K=%d nq=%d  verified=%s"
                  % (ds, m, format(ch["bytes"], ","), ch["sha256"][:16], ch["K"],
                     ch["n_queries"], cell.get("verdict", "ABSENT")))
        if len(chans) == len(MODELS):
            old = man["datasets"][ds]["retrieval_cache"]
            man["datasets"][ds]["retrieval_cache"] = {
                "present": True,
                "channels": chans,
                "provenance": {
                    "NOT_PINNED_BY_ANY_FREEZE": True,
                    "why": "this manifest copies hashes from the freeze named in pinned_by and "
                           "never recomputes them, so that it cannot silently disagree with a "
                           "freeze. These three caches postdate every freeze, so there is "
                           "nothing to copy and the hashes here were computed from disk. That "
                           "is a weaker guarantee than the other three slots carry, and it is "
                           "recorded rather than hidden.",
                    "measured_by": "data/final_canonical/RETRIEVAL_CACHE.json",
                    "verified_by": "data/final_canonical/HEAVY_CACHE_VERIFICATION.json",
                    "authority_to_add_after_a_freeze":
                        "data/final_canonical/FROZEN_CAVEAT_SUPERSESSION.json, which supersedes "
                        "the frozen caveats asserting these caches do not exist, by new record "
                        "rather than by editing them",
                    "verification_verdicts": {
                        m: (res.get("%s.%s" % (ds, m), {}) or {}).get("verdict")
                        for m in MODELS}},
                "superseded_slot_VERBATIM": old,
                "which_half_of_that_reason_still_holds":
                    "'The cache is a convenience over the substrate, not part of it -- it can "
                    "be rebuilt from the pointer index at any time' remains TRUE and is the "
                    "reason nothing in the package depends on these files. Only 'no retrieval "
                    "cache was built for this dataset' is superseded."}
            filled.append(ds)

    man["SLOT_UNIFORMITY"]["retrieval_cache_present_for_all_six"] = (len(filled)
                                                                     == len(WAS_ABSENT))
    man["SLOT_UNIFORMITY"]["retrieval_cache_provenance_is_not_uniform"] = (
        "metaqa/musique/squad are freeze-pinned (LOCKED_5_OF_5); hotpotqa/2wiki/webqsp are "
        "not pinned by any freeze -- see each slot's provenance block. Uniform KEYS, "
        "non-uniform AUTHORITY, stated rather than smoothed over.")

    rec = {"RECORD": "MANIFEST_CACHE_SLOT_FILL",
           "_what": "fills the three retrieval_cache slots in UKB_COMMON_MANIFEST.json that "
                    "read present=false, and records that their provenance is not the "
                    "freeze-copied provenance the other three have.",
           "slots_filled": filled,
           "slots_checked_against_disk": checked,
           "cells_missing_on_disk": missing,
           "cells_not_verified": unverified,
           "manifest_vs_disk_mismatches": mismatch,
           "applied": bool(apply_ and not missing and not unverified and not mismatch),
           "manifest_sha256_before": hashlib.sha256(raw).hexdigest(),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    print("")
    if mismatch:
        print("MANIFEST DISAGREES WITH DISK -- reported, not repaired:")
        for x in mismatch:
            print("   !! %s" % x)
    if missing:
        print("REFUSING: %d cells missing on disk -- %s" % (len(missing), ", ".join(missing)))
    if unverified:
        print("REFUSING: %d new cells not verified CONSISTENT -- %s"
              % (len(unverified), ", ".join(unverified)))

    ok = apply_ and not missing and not unverified and not mismatch
    if ok:
        # this file's own convention, read off the file: indent=1, LF
        b = json.dumps(man, indent=1, ensure_ascii=False).encode("utf-8")
        with io.open(MAN, "wb") as f:
            f.write(b)
        rec["manifest_sha256_after"] = hashlib.sha256(b).hexdigest()
        print("wrote %s  %d bytes  sha %s" % (rel(MAN), len(b), rec["manifest_sha256_after"]))
        print("   NOTE: UKB_COMMON_MANIFEST_VERIFICATION.json pins this file -- regenerate it:")
        print("   python src/dataset_canonical/verify_manifest.py --records --sizes --graphs "
              "--builders --out=data/final_canonical/UKB_COMMON_MANIFEST_VERIFICATION.json")
    elif not apply_:
        print("DRY RUN -- re-run with --apply to write the manifest")

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote %s" % rel(OUT))
    return 0 if (not missing and not unverified and not mismatch) else 1


if __name__ == "__main__":
    sys.exit(main())
