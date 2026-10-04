"""Three caveats inside FROZEN records are now false. This supersedes them without editing them.

WHAT IS FALSE

  LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json and LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json each
  carry limitations asserting that the retrieval caches cover three datasets, that hotpotqa and
  2wiki were left unbuilt on cost grounds, and that WebQSP has no cache at all.  All six
  datasets now have caches at K=1000 for both models, and every one of the three new ones has
  been verified against a local recomputation over its full corpus.

WHY THIS IS A NEW RECORD AND NOT AN EDIT

  The standing rule is that a contract amendment needs a new record with its own hash and never
  an edit.  These are LOCKED_* records: their self-hash is computed over the record with its
  own RECORD_SHA256 removed, so editing a caveat would invalidate the record and every record
  that BUILDS_ON it.  So nothing is touched.  Instead this quotes each superseded caveat
  VERBATIM, states what replaced it, names the measurement that establishes the replacement,
  and re-computes each frozen record's self-hash to PROVE it was not modified.

  A quote is matched by exact substring against the frozen record.  If a caveat's text cannot
  be found, this refuses rather than superseding a claim it cannot locate -- otherwise the
  supersession would drift away from the words it claims to supersede.

WHY IT REFUSES WITHOUT VERIFICATION

  "The cache exists" is a weaker claim than the caveat it replaces.  The caveat said the
  package had no such artifact; replacing it with an unverified file would trade a true
  statement for an unchecked one.  So a cache counts as superseding evidence only when
  HEAVY_CACHE_VERIFICATION.json holds a CONSISTENT verdict for that cell, and this exits
  non-zero if any of the six new cells is missing or unverified.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/frozen_caveat_supersession.py
"""

import hashlib
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
HEAVY = os.path.join(FC, "HEAVY_CACHE_VERIFICATION.json")
CACHEREC = os.path.join(FC, "RETRIEVAL_CACHE.json")
OUT = os.path.join(FC, "FROZEN_CAVEAT_SUPERSESSION.json")
BS = chr(92)

NEW_CELLS = [("webqsp", "dense"), ("webqsp", "splade"),
             ("hotpotqa", "dense"), ("hotpotqa", "splade"),
             ("2wiki", "dense"), ("2wiki", "splade")]

# Each entry: the frozen record, and a distinctive substring of the caveat being superseded.
# Substrings, not full sentences, so that a match is robust to surrounding text but still
# unambiguous -- each is checked to occur exactly once.
CAVEATS = [
    ("LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json",
     "WebQSP has no retrieval cache"),
    ("LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json",
     "retrieval caches are materialised for metaqa, squad and musique only"),
    ("LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json",
     "retrieval caches are materialised for metaqa, squad and musique only"),
]


def rel(p):
    return os.path.relpath(p, ROOT).replace(BS, "/")


# Do NOT reimplement the self-hash.  A first attempt here removed only RECORD_SHA256 and
# passed ensure_ascii=False.  It reproduced LOCKED_5_OF_5, which predates the LF field, and
# failed LOCKED_6_OF_6 -- which looks exactly like a tampered frozen record and was nothing of
# the kind.  The convention lives in verify_manifest.record_hash: it removes RECORD_SHA256 AND
# RECORD_SHA256_LF and serialises at the default ensure_ascii.  Import it, do not restate it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_manifest import record_hash  # noqa: E402


def self_hash(d):
    return record_hash(d, crlf=True), record_hash(d, crlf=False)


def find_quote(text, needle):
    n = text.count(needle)
    if n == 0:
        raise SystemExit("REFUSING: caveat text not found in the frozen record: %r" % needle)
    if n > 1:
        raise SystemExit("REFUSING: caveat text occurs %d times, so the supersession would be "
                         "ambiguous: %r" % (n, needle))
    i = text.index(needle)
    # widen to the enclosing string literal so the quote is the whole caveat, not a fragment
    a = text.rfind('"', 0, i) + 1
    b = text.index('"', i + len(needle))
    return text[a:b]


def main():
    heavy = json.load(io.open(HEAVY, encoding="utf-8")) if os.path.isfile(HEAVY) else {}
    res = heavy.get("results", {}) or {}

    evidence, unverified = {}, []
    for ds, m in NEW_CELLS:
        key = "%s.%s" % (ds, m)
        cell = res.get(key, {}) or {}
        p = os.path.join(FC, ds, "retrieval_cache", "%s_top1000.npz" % m)
        ex = os.path.isfile(p)
        ok = ex and cell.get("verdict") == "CONSISTENT"
        evidence[key] = {
            "file": rel(p), "file_exists": ex, "bytes": os.path.getsize(p) if ex else None,
            "verification_verdict": cell.get("verdict", "ABSENT_FROM_VERIFICATION_RECORD"),
            "set_overlap_at_1000": cell.get("overlap_at_1000"),
            "containment_of_top10_in_cached_1000": cell.get(
                "containment_at_10_in_cached_1000"),
            "max_abs_score_delta": cell.get("max_abs_score_delta"),
            "whole_artifact_invariants_PASS": (cell.get("whole_artifact_invariants") or {}
                                               ).get("PASS"),
            "counts_as_superseding_evidence": ok}
        if not ok:
            unverified.append(key)
        print("%-16s exists=%-5s verdict=%-12s ovl@1k=%-6s inv=%-5s evidence=%s"
              % (key, ex, cell.get("verdict", "ABSENT"),
                 cell.get("overlap_at_1000"),
                 (cell.get("whole_artifact_invariants") or {}).get("PASS"), ok))
    print("")

    sup = []
    for fname, needle in CAVEATS:
        p = os.path.join(FC, fname)
        d = json.load(io.open(p, encoding="utf-8"))
        text = json.dumps(d, ensure_ascii=False)
        quote = find_quote(text, needle)
        crlf, lf = self_hash(d)
        stamped = d.get("RECORD_SHA256")
        sup.append({
            "frozen_record": rel(p),
            "record_name": d.get("RECORD"),
            "superseded_caveat_VERBATIM": quote,
            "still_true_part": "'Caches are an accelerator over the frozen embeddings, not a "
                               "dependency' and 'nothing in the package depends on them' remain "
                               "TRUE and are not superseded. Only the coverage claim and the "
                               "cost verdict are.",
            "what_replaced_it": "all six datasets carry retrieval_cache/{dense,splade}"
                                "_top1000.npz at K=1000, and the three added after the freeze "
                                "are verified CONSISTENT against a local recomputation over "
                                "their full corpora.",
            "why_the_cost_verdict_was_not_wrong": "it measured THIS machine -- 33.8 effective "
                                                  "GFLOPS, torch 2.8.0+cpu, no GPU -- and on "
                                                  "this machine the estimate stands. The two "
                                                  "dense caches were produced in 21 s and 37 s "
                                                  "on a rented H100 and the two splade caches "
                                                  "on rented CPU in 24 query shards. A cost "
                                                  "verdict is only as portable as the hardware "
                                                  "it was measured on.",
            "record_was_NOT_edited": {
                "stamped_RECORD_SHA256": stamped,
                "stamped_RECORD_SHA256_LF": d.get("RECORD_SHA256_LF"),
                "recomputed_self_hash_crlf": crlf,
                "recomputed_self_hash_lf": lf,
                "self_hash_reproduces": (stamped == crlf
                                         or (d.get("RECORD_SHA256_LF") is not None
                                             and d.get("RECORD_SHA256_LF") == lf)),
                "_convention": "verify_manifest.record_hash, imported not restated: the record "
                               "re-serialised at indent=1 with BOTH RECORD_SHA256 and "
                               "RECORD_SHA256_LF removed. LOCKED_5_OF_5 predates the LF field, "
                               "so CRLF alone is its whole contract."}})
        print("%-46s caveat located, self_hash_reproduces=%s"
              % (fname[:46], stamped in (crlf, lf)))

    bad = [s for s in sup if not s["record_was_NOT_edited"]["self_hash_reproduces"]]

    rec = {"RECORD": "FROZEN_CAVEAT_SUPERSESSION",
           "_what": "supersedes three caveats inside frozen LOCKED_* records that asserted the "
                    "retrieval caches cover three of six datasets. It does not edit them.",
           "why_a_new_record": "a LOCKED_* record's self-hash is computed over itself with "
                               "RECORD_SHA256 removed, so editing a caveat would invalidate "
                               "the record and everything that BUILDS_ON it. The standing rule "
                               "is that a contract amendment gets a new record with its own "
                               "hash and never an edit.",
           "supersedes": sup,
           "evidence": evidence,
           "evidence_standard": "a cache counts only when HEAVY_CACHE_VERIFICATION.json holds "
                                "a CONSISTENT verdict for that cell. Replacing a true 'this "
                                "artifact does not exist' with an unchecked file would trade a "
                                "true statement for an unverified one.",
           "cells_not_yet_qualifying": unverified,
           "COMPLETE": not unverified and not bad,
           "downstream_records_to_reread": [
               "data/final_canonical/RETRIEVAL_CACHE.json (NOT_MATERIALISED now empty; see "
               "RETRIEVAL_CACHE_DRIFT_ADJUDICATION.json for its frozen-pin adjudication)",
               "data/final_canonical/HANDOFF.md sections on partial caches",
               "data/final_canonical/HANDOFF_PREPARTITION.md",
               "data/final_canonical/SUBSTRATE_HANDOFF.md ('the one genuinely universal hole "
               "is retrieval caches')"],
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    with io.open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False).replace("\n", "\r\n"))
    print("")
    print("wrote %s" % rel(OUT))

    if bad:
        print("REFUSING: a frozen record's self-hash does not reproduce -- it may have been "
              "modified, and a supersession must not be anchored to a record whose integrity "
              "is unproven.")
        return 1
    if unverified:
        print("INCOMPLETE: %d of %d new cells do not yet qualify as evidence -- %s"
              % (len(unverified), len(NEW_CELLS), ", ".join(unverified)))
        print("   the record is written but COMPLETE=false; re-run when they verify.")
        return 1
    print("COMPLETE: all %d new cells verified; three frozen caveats superseded without edit."
          % len(NEW_CELLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
