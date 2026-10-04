"""§F stale-hash sweep: find every surviving reference to the superseded 2wiki TEXTUALIZATION_REV 1 hashes.

    python scratchpad/final_canonical_build/_analysis/stale_hash_sweep.py

A plain recursive grep is not usable here: data/final_canonical holds ~11 GB of nodes.jsonl and reuse_map
shards, and ripgrep additionally honours .gitignore (which excludes data/ entirely), so it would read
everything slowly AND miss the tree that matters.  This walks the REPORT LAYER only -- small text files
under a size cap -- across both the gitignored data tree and the tracked repo.

Every hit is classified, because a hit is not automatically a defect.  A grep that reports 57 hits and calls
them all stale is useless: the directive requires BOTH "sweep for stale forward-looking references" AND
"preserve superseded hashes as historical records", so the sweep has to tell those two apart.

  HISTORICAL_DIR    the file lives in a preservation tree (_superseded_*/) whose whole job is to hold the
                    old build.  Rewriting these would falsify the record.
  HISTORICAL_FIELD  a LIVE file, but the hash sits in a field that names itself as past tense --
                    "old" / "*_before" / "_superseded_*" / "as rev1 was built" / an explicit HISTORICAL
                    comment -- normally beside the live value or a dated re-pin note.
  REGENERATED       a manifest or mirror copy that the pending re-pin rewrites from the live upstream
                    build_info/integrity_report.  Not hand-editable and not a defect; it is stale only
                    until manifest.py / mirror_audit.py run.  Listed so the claim can be re-checked after.
  STALE             a live, forward-looking artifact still asserting the old hash as current.  Defect.

Only STALE requires a hand edit.  STALE_HASH_REFERENCES_REMAINING is the number that matters; the
REGENERATED list is the promise this sweep must be re-run to discharge.
"""
import os, sys, json, time

# SEARCH BY 16-CHAR PREFIX, NOT BY THE FULL 64-CHAR DIGEST.  The report layer cites hashes both ways --
# json records carry the full digest, prose and tables carry `81fa7d1a5d4bbb24…` -- and a full-digest search
# silently misses every truncated citation.  That is not hypothetical: the first version of this sweep
# searched full digests, reported the 2wiki row of SOURCE_CONTRACTS_FOR_REVIEW.md as clean, and that row was
# in fact still asserting the rev1 hash as the live one.  The prefix is a strict superset of both forms.
OLD = {
    "81fa7d1a5d4bbb24": "2wiki REV1 CORPUS_HASH",
    "2653f6f35690ba31": "2wiki REV1 nodes_jsonl_sha256",
    "862e0b3c284fd12a": "REV1 build_kb.py builder_sha256",
    # hotpotqa's own rev1 -> rev2 re-pin (2026-09-05) is swept too: the same class of defect, and cheap to
    # check while we are here.  Its rev2 hash is 1b7eeac2bfbd7100...
    "fe139de9c7c2a14f": "hotpotqa REV1 CORPUS_HASH",
}
ROOTS = ("data/final_canonical", "results/data_audit/final_canonical_v1", "scratchpad/final_canonical_build")
EXT = {".json", ".md", ".txt", ".py", ".csv", ".log", ".yaml", ".yml"}
MAX_BYTES = 8 * 1024 * 1024        # report-layer files are KB-scale; this skips the node/reuse tables
SKIP_DIRS = {"_work", "reuse_map", "queries", "__pycache__", ".git"}
# This sweep's own output quotes every hit line verbatim, so it matches itself on the next run and would
# report ~75 phantom hits that grow each time.  Its two output paths are excluded from the scan.
SKIP_FILES = {"STALE_HASH_SWEEP.json"}

# A hit in one of these paths is the file doing its documented job.  Matched as substrings of the
# forward-slash relative path.
HISTORICAL_DIRS = (
    "_superseded_textualization_rev1/",   # the whole rev1 preservation tree, incl. its build_info/integrity
    "_superseded_mirror/",
    "_superseded_sidecars/",
    "_superseded_context_union_398354/",
    "SUPERSEDED.md",
    "TEXTUALIZATION_REV2_REPORT.md",      # the report whose subject IS the rev1 -> rev2 transition
    "REV2_DIFF_AUDIT.json",               # records old_* and new_* hashes side by side, by construction
    "REV2_VERIFY.json",
    "EMPTY_TEXT_AUDIT.md", "EMPTY_TEXT_SUMMARY.json",
    "ARTIFACT_REUSE_REPORT.md",           # carries the hash-history table
    "DOWNSTREAM_REBUILD_POLICY.md",
    "CANONICALIZATION_AUDIT.md", "CANONICALIZATION_AUDIT.json",
    "_build.log",                         # append-only build journal
    "stale_hash_sweep.py",                # this file
)
# Markers ON THE HIT LINE ITSELF that name the value as past tense.  Deliberately conservative: a bare
# mention of the hash with no such marker stays STALE even in a file that also carries a re-pin note
# elsewhere, because a reader grepping the file lands on the line, not on the note.
HISTORICAL_FIELD_MARKERS = (
    '"old"', "_superseded", "superseded", "_BEFORE", "_before", "CORPUS_HASH_before",
    "was re-pinned", "rev1", "REV1", "pre-REV2", "HISTORICAL", "_NOT_THE_LIVE_SHAS",
)
# Files whose content is not authored but derived: the pending re-pin rewrites them wholesale from the
# live upstream build_info/integrity_report, so a hand edit here would be overwritten anyway.
REGENERATED_SUFFIXES = ("MANIFEST.json", "dataset_manifest.json", "build_info.json",
                        "integrity_report.json", "query_independence_test.json")
MIRROR = "results/data_audit/final_canonical_v1/"
# The live values the re-pin installed.  A line that mentions an old hash AND the value that replaced it --
# or that mentions an old hash next to an arrow -- is a TRANSITION RECORD ("x -> y", "moved x -> y",
# "x -> see build_info"), which is the opposite of a stale claim: it exists to retire the old value.  This
# is a general rule about the shape of the line, not a per-file exemption.
CURRENT = ("94fe68b8d5f291f1",   # 2wiki rev2 CORPUS_HASH
           "73fb822f3e4dd255",   # 2wiki rev2 nodes_jsonl_sha256
           "e723192e739de388",   # rev2 build_kb.py
           "1b7eeac2bfbd7100")   # hotpotqa rev2 CORPUS_HASH
ARROWS = ("->", "→")


# Hits whose line carries no self-describing marker but which a REVIEWED adjacent record explains.  Each is
# named individually with its reason, rather than loosening the matcher above -- an exemption you have to
# write down and justify is auditable; a widened regex is not.  (path_suffix, line_substring, reason).
REVIEWED_EXEMPTIONS = (
    ("_WEBQSP_ACCEPTANCE_GATE.json", '"new": "862e0b3c',
     "the 'new' sha of the 2026-09-05 gold-semantics patch, i.e. a correct record of THAT event. The "
     "sibling key _NOT_THE_LIVE_SHAS_2026_09_06 in the same object states the current sha and the "
     "transition. Rewriting 'new' would falsify the patch record."),
    ("webqsp/WEBQSP_REPRESENTATION_AUDIT.md", "`862e0b3c284fd12a…` above is the",
     "part of the 2026-09-06 SECOND CORRECTION block that exists precisely to mark the value above as "
     "superseded; it quotes the old sha in order to retire it."),
    ("_audit/gold_semantics.json", '"2wiki": "81fa7d1a',
     "the CORPUS_HASH_BEFORE / CORPUS_HASH_AFTER pair of the gold-semantics audit, whose finding was that "
     "that fix did NOT move the hash. Both values are correct for that audit. _STATUS_2026_09_06 in the "
     "same file scopes them and gives the live value."),
    ("_audit/gold_semantics.json", '"hotpotqa": "fe139de9',
     "same CORPUS_HASH_BEFORE / CORPUS_HASH_AFTER pair for hotpotqa, superseded by its own rev2 re-pin on "
     "2026-09-05; scoped by the same _STATUS_2026_09_06 block."),
    ("webqsp/WEBQSP_REPRESENTATION_AUDIT.md", "**before and after** — unchanged",
     "a finding of the 2026-09-05 gold-semantics audit that was TRUE when written -- that fix ran "
     "--queries-only and did not move the hash. The 'SECOND CORRECTION 2026-09-06' block immediately below "
     "retires the quoted value. Rewriting the finding would falsify a dated audit result."),
    ("webqsp/WEBQSP_REPRESENTATION_AUDIT.md", "`81fa7d1a5d4bbb24…`.**]**",
     "the closing line of the 2026-09-05 CORRECTION block, retired by the 'SECOND CORRECTION 2026-09-06' "
     "block that directly follows it and names the value that replaced it."),
)


def classify(rel, line):
    if any(h in rel for h in HISTORICAL_DIRS):
        return "HISTORICAL_DIR"
    if rel.endswith(".log"):
        # append-only record of what a run actually printed; editing it would falsify the run
        return "HISTORICAL_RUNLOG"
    if any(c in line for c in CURRENT) or any(a in line for a in ARROWS):
        return "HISTORICAL_TRANSITION"
    if any(m in line for m in HISTORICAL_FIELD_MARKERS):
        return "HISTORICAL_FIELD"
    if any(rel.endswith(p) and s in line for p, s, _ in REVIEWED_EXEMPTIONS):
        return "HISTORICAL_REVIEWED"
    if rel.startswith(MIRROR) or any(rel.endswith(s) for s in REGENERATED_SUFFIXES):
        return "REGENERATED"
    return "STALE"


def main():
    t0 = time.time()
    hits, scanned = [], 0
    for root in ROOTS:
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS]
            for fn in fns:
                if os.path.splitext(fn)[1].lower() not in EXT or fn in SKIP_FILES:
                    continue
                p = os.path.join(dp, fn)
                try:
                    if os.path.getsize(p) > MAX_BYTES:
                        continue
                    txt = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                scanned += 1
                for h, what in OLD.items():
                    if h not in txt:
                        continue
                    rel = p.replace("\\", "/")
                    # classify EVERY occurrence line separately -- one file can hold both a preserved
                    # historical value and a live stale one, and collapsing to the first line hides that.
                    for ln in txt.splitlines():
                        if h in ln:
                            # CLASSIFY ON THE FULL LINE, DISPLAY THE TRUNCATION.  Classifying the 200-char
                            # display string was a real bug: SOURCE_CONTRACT.json serialises a multi-paragraph
                            # NORMALIZATION_RULE as ONE json line whose "rev1 ...; live value is
                            # build_info.json -> CORPUS_HASH" marker sits ~900 chars in, so the truncated
                            # string looked like a bare unexplained hash and the record was reported STALE.
                            hits.append({"path": rel, "hash": h, "what": what,
                                         "line": ln.strip()[:200], "class": classify(rel, ln)})
    by = lambda c: sorted([h for h in hits if h["class"] == c], key=lambda x: (x["path"], x["line"]))
    stale, regen = by("STALE"), by("REGENERATED")
    reviewed = by("HISTORICAL_REVIEWED")
    # EVERY class must appear in exactly one output list.  An earlier version of this file classified
    # HISTORICAL_TRANSITION / HISTORICAL_RUNLOG and then never added them to `hist`, so 15 hits were counted
    # in TOTAL_HITS and reported nowhere -- a sweep that loses hits is worse than no sweep.  The assert below
    # makes that failure loud instead of silent.
    hist = (by("HISTORICAL_DIR") + by("HISTORICAL_FIELD") + by("HISTORICAL_TRANSITION")
            + by("HISTORICAL_RUNLOG") + reviewed)
    assert len(stale) + len(regen) + len(hist) == len(hits), (
        f"classification lost hits: {len(stale)}+{len(regen)}+{len(hist)} != {len(hits)}; "
        f"classes seen = {sorted({h['class'] for h in hits})}")
    out = {"files_scanned": scanned, "roots": list(ROOTS),
           "hashes_searched": dict(OLD),
           "TOTAL_HITS": len(hits), "STALE_HITS": len(stale), "HISTORICAL_HITS": len(hist),
           "REGENERATED_HITS": len(regen),
           "hits_by_class": {c: sum(1 for h in hits if h["class"] == c)
                             for c in sorted({h["class"] for h in hits})},
           "REVIEWED_EXEMPTIONS": [{"path_suffix": p, "line": s, "reason": r}
                                   for p, s, r in REVIEWED_EXEMPTIONS],
           "reviewed_exemption_hits": reviewed,
           "STALE_HASH_REFERENCES_REMAINING": len(stale),
           "REGENERATED_FILES_PENDING": sorted({h["path"] for h in regen}),
           "_regenerated_note": "not hand-edited: manifest.py + mirror_audit.py rewrite these from the live "
                                "upstream build_info/integrity_report. Re-run this sweep after the re-pin; "
                                "REGENERATED_HITS must then be 0.",
           "stale": stale, "regenerated": regen,
           "historical": hist,
           "seconds": round(time.time() - t0, 1)}
    for op in ("data/final_canonical/STALE_HASH_SWEEP.json",
               "results/data_audit/final_canonical_v1/STALE_HASH_SWEEP.json"):
        os.makedirs(os.path.dirname(op), exist_ok=True)
        json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: out[k] for k in ("files_scanned", "TOTAL_HITS", "STALE_HITS", "REGENERATED_HITS",
                                          "HISTORICAL_HITS", "seconds")}, indent=1))
    print("\nSTALE -- live forward-looking claims, must be hand-fixed:")
    for h in stale:
        print(f"  {h['path']}  [{h['hash']}]\n      {h['line']}")
    if not stale:
        print("  (none)")
    print("\nREGENERATED -- rewritten by manifest.py / mirror_audit.py, re-check after the re-pin:")
    for f in out["REGENERATED_FILES_PENDING"]:
        print(f"  {f}  ({sum(1 for h in regen if h['path'] == f)} hits)")
    if not regen:
        print("  (none)")
    print(f"\nHISTORICAL -- preserved on purpose ({len(hist)} hits in "
          f"{len({h['path'] for h in hist})} files); listed in the json.")
    print("\nwrote", op)


if __name__ == "__main__":
    main()
