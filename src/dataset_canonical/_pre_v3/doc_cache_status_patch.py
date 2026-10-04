"""Update the four prose docs that say the retrieval caches cover three of six datasets.

WHY THESE FOUR CAN BE EDITED WHEN THE LOCKED_* RECORDS CANNOT

  HANDOFF.md is hashed by LOCKED_5_OF_5, and it has ALREADY diverged from that pin --
  FROZEN_ARTIFACT_RECHECK.json records it as DOCS_DRIFT and says why in as many words: sealing
  a document that is expected to be revised was a design mistake in freeze_five, and
  LOCKED_6_OF_6 deliberately does not repeat it.  So editing prose here is precedented and
  recorded, not a new violation.  The LOCKED_* records themselves are still never touched --
  their three now-false caveats are superseded by FROZEN_CAVEAT_SUPERSESSION.json instead.

  One consequence is recorded rather than hidden: FROZEN_ARTIFACT_RECHECK.json's
  `actual_sha256` for HANDOFF.md was a SNAPSHOT of a drifting file, not a pin, and it was
  already stale before this script ran (41,854 bytes recorded, more than that on disk).  This
  moves it again.  The new digests are written here so the trail does not go cold.

HOW A PATCH IS APPLIED

  By exact substring, required to occur EXACTLY ONCE, at the byte level so no line ending is
  rewritten as a side effect.  A patch whose text cannot be found -- or is found twice --
  REFUSES rather than guessing, because a prose edit that lands in the wrong paragraph is
  worse than one that does not land at all.  Nothing is applied unless every patch resolves.

  Struck-through claims are kept visible ("~~...~~ -- SUPERSEDED") wherever a reader might
  have quoted the old text, rather than deleted, so a stale citation still leads somewhere.

WHY IT REFUSES WITHOUT VERIFICATION

  Same standard as the records: a doc may not claim a cache exists until
  HEAVY_CACHE_VERIFICATION.json holds a CONSISTENT verdict for that cell.  Twelve cells, six
  datasets, two models, all twelve or nothing.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/doc_cache_status_patch.py [--apply]
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
OUT = os.path.join(FC, "DOC_CACHE_STATUS_PATCH.json")
BS = chr(92)

DATASETS = ["metaqa", "musique", "squad", "hotpotqa", "2wiki", "webqsp"]
MODELS = ["dense", "splade"]

H = os.path.join(FC, "HANDOFF.md")
P = os.path.join(FC, "HANDOFF_PREPARTITION.md")
S = os.path.join(FC, "SUBSTRATE_HANDOFF.md")
R = os.path.join(ROOT, "src", "dataset_canonical", "README.md")

# (path, label, old_exact, new_exact)
PATCHES = [
    (H, "layout_note_where_materialised",
     "   [where materialised]",
     "   [all six, K=1000]"),

    (H, "limitation_6_partial_caches",
     "6. **Retrieval caches are partial.** metaqa, squad and musique only. See §9.",
     "6. ~~**Retrieval caches are partial.** metaqa, squad and musique only.~~ **SUPERSEDED\n"
     "   2026-09-10** — all six now carry `retrieval_cache/{dense,splade}_top1000.npz`, and\n"
     "   the three added after the freeze are verified against a local recomputation over their\n"
     "   full corpora. The frozen wording in `LOCKED_5_OF_5` / `LOCKED_6_OF_6` was **not**\n"
     "   edited; see `FROZEN_CAVEAT_SUPERSESSION.json`. See §9."),

    (H, "limitation_12_webqsp_no_cache",
     "12. **WebQSP has no retrieval cache**, for the same throughput reason as hotpotqa and\n"
     "    2wiki. Caches are an accelerator, never a dependency.",
     "12. ~~**WebQSP has no retrieval cache**, for the same throughput reason as hotpotqa and\n"
     "    2wiki.~~ **SUPERSEDED 2026-09-10** — it has both channels at K=1000, verified\n"
     "    CONSISTENT. The second sentence stands unchanged and is the reason this was safe to\n"
     "    add: caches are an accelerator, never a dependency."),

    (H, "s9_heading_and_cost_framing",
     "**Retrieval caches for hotpotqa, 2wiki and webqsp.** Materialised: musique, squad, metaqa"
     " at K=1000\n(all of K = 50/100/250/500 are prefixes of it, so no K was ever selected from"
     " an outcome).\n\nThe remaining three are a *cost* decision, not a technical obstacle.",
     "**Retrieval caches — now all six.** Materialised: all six datasets at K=1000, both\n"
     "channels (all of K = 50/100/250/500 are prefixes of it, so no K was ever selected from an\n"
     "outcome). What K=1000 does and does not bound is a declared rule, filed before any\n"
     "ceiling was computed rather than discovered in a results table — see\n"
     "`K_SEMANTICS_AND_EVAL_SPLITS.json`: a ceiling measured over this cache is a\n"
     "**cache-limited** ceiling, not a corpus ceiling.\n\n"
     "The three that were missing were a *cost* decision, not a technical obstacle, and the"
     " cost was local.",),

    (H, "s9_cost_table_caption",
     "WebQSP's is cheap only because it has 4,737 queries; the FLOP figure is\n"
     "`2 × n_queries × n_docs × 1536` on the same measured throughput, not a timed run. It was\n"
     "**not** built, because `LOCKED_6_OF_6` is frozen with \"WebQSP has no retrieval cache\" as a"
     "\nrecorded limitation, and quietly adding an artifact after a freeze is how a freeze stops"
     "\nmeaning anything. Build it if you want it — then write a new record saying so.",
     "WebQSP's is cheap only because it has 4,737 queries; the FLOP figure is\n"
     "`2 × n_queries × n_docs × 1536` on the same measured throughput, not a timed run.\n\n"
     "Those estimates were never wrong — they measured **this** machine, and on this machine\n"
     "they still stand. Rented hardware made them obsolete; it did not correct them. The two\n"
     "dense caches took 21 s and 37 s on a rented H100. The SPLADE caches were split into 24\n"
     "query shards, which is exact rather than approximate because a score-matrix row depends\n"
     "on one query only, so shards built in different places tile the query axis with no\n"
     "overlap and no gap.\n\n"
     "**2wiki's SPLADE cache is the one mixed artifact in the package**, and the merge record\n"
     "says so per shard rather than averaging it away. Thirteen of its 24 shards were built on\n"
     "rented CPU; that workspace was then disabled mid-run, and every remaining workspace in\n"
     "the pool turned out to have exceeded its spend limit — storage still accepted a 4.72 GB\n"
     "upload on two of them, which is why liveness had to be probed with compute and not with\n"
     "storage. The other eleven shards were therefore built **here**, in one shared corpus pass\n"
     "carrying all 88,275 of their queries. Read `backend_by_part` and `backend_row_counts` in\n"
     "`2wiki/retrieval_cache/splade_top1000.meta.json` before quoting a provenance for it.\n\n"
     "The instruction this section gave — *\"Build it if you want it — then write a new record\n"
     "saying so\"* — is what was followed. `LOCKED_6_OF_6` is frozen with \"WebQSP has no\n"
     "retrieval cache\" as a recorded limitation and was **not** edited;\n"
     "`FROZEN_CAVEAT_SUPERSESSION.json` supersedes that caveat and two others by new record,\n"
     "and recomputes each frozen record's self-hash to prove it was not modified."),

    (H, "s9_gpu_paragraph",
     "On a GPU this is roughly an hour of compute, but it requires shipping ~34 GB of embeddings"
     " to\nthe runner first. Either path is fine; neither was worth spending unattended. "
     "**Nothing else\ndepends on these caches** — they are an accelerator over the frozen "
     "embeddings, and any\nconsumer can compute the same top-K exactly from the pointer index.",
     "The GPU path did cost what this section predicted: shipping the embeddings to the runner\n"
     "dominated, not the arithmetic. **Nothing else depends on these caches** — they are an\n"
     "accelerator over the frozen embeddings, and any consumer can compute the same top-K\n"
     "exactly from the pointer index. That is unchanged, and it is why adding them after a\n"
     "freeze changes no result: it is the reason the addition is safe, not a reason it was\n"
     "unnecessary."),

    (H, "s9_not_materialised_note",
     "One thing to know when reading `RETRIEVAL_CACHE.json` directly: its `NOT_MATERIALISED`\n"
     "list names hotpotqa and 2wiki only, because it was written before dataset 6 existed. It is"
     "\nhashed by `LOCKED_5_OF_5`, so it was not edited — webqsp's missing cache is recorded in"
     "\n`LOCKED_6_OF_6` → `KNOWN_LIMITATIONS[5]` instead. Amend by new record, never by edit.",
     "Two things to know when reading `RETRIEVAL_CACHE.json` directly. Its `NOT_MATERIALISED`\n"
     "list is now **empty**, and its entries for the three post-freeze datasets were appended —\n"
     "so the file no longer matches the 9,580-byte / `ae2b5c0f…` digest `LOCKED_5_OF_5` pins.\n"
     "That is adjudicated the way the package's three earlier drifts were, by reproduction and\n"
     "not by widening a tolerance: stripping the appended entries and re-serialising at the\n"
     "frozen convention reproduces those exact bytes and that exact digest, so no byte of the\n"
     "five's entries changed. The recipe is carried **inside** the file as `_FROZEN_ORIGINAL`,\n"
     "so it needs nothing external to check. See `RETRIEVAL_CACHE_DRIFT_ADJUDICATION.json`.\n"
     "`FROZEN_ARTIFACT_RECHECK.json` still lists this file as `MATCH` at 9,580 bytes; that\n"
     "verdict predates the append and is superseded by the adjudication record."),

    # Not a cache claim -- found while redoing the degree/hub analysis on the .npz families.
    # HANDOFF.md has ZERO mentions of graph2/ or LOCKED_6_OF_6_FAMILY_COMPLETION_V1, so a
    # reader of this list would conclude webqsp has one family and never learn that
    # pointer_resolver_v2 unions a second tree.  The freeze that added them is itself FROZEN,
    # so this is a doc that never caught up, not an unrecorded change.
    (H, "limitation_7_webqsp_structural_only",
     "7. **WebQSP carries the structural family only** — no NER (no document prose to run it\n"
     "   over) and no kNN (kNN needs embeddings, which the graph build deliberately did not\n"
     "   depend on).",
     "7. ~~**WebQSP carries the structural family only** — no NER (no document prose to run\n"
     "   it over) and no kNN (kNN needs embeddings, which the graph build deliberately did\n"
     "   not depend on).~~ **SUPERSEDED by `LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json`**, which\n"
     "   is itself FROZEN: webqsp now carries all three families — `ner` 2,994,802 edges and\n"
     "   `knn` 6,470,520 edges, both in `webqsp/graph2/`, with `structural` still the frozen\n"
     "   `graph/` copy. `graph/` is byte-identical; `pointer_resolver_v2.families(ds)` unions\n"
     "   `graph/` and `graph2/`, and `family_source(ds, fam)` says which tree answered. Read\n"
     "   the coverage before using them: webqsp `ner` touches only 30.67% of nodes, so its\n"
     "   mean degree over *touched* nodes (7.53) and over all nodes (2.31) differ by 3.3×.\n"
     "   Per-family degree and hub concentration for all 18 families are in\n"
     "   `GRAPH_DEGREE_HUB.json`."),

    (P, "prepart_stage6_row",
     "| 6 retrieval cache | `scratchpad/build_retrieval_cache.py` — squad, musique, metaqa"
     " only |",
     "| 6 retrieval cache | `scratchpad/build_retrieval_cache.py` — all six at K=1000 "
     "(the last three added 2026-09-10, off-freeze) |"),

    (P, "prepart_absent_slots",
     "indistinguishable from an unbuilt one. Three slots are explicitly absent today: the\n"
     "retrieval cache for hotpotqa, 2wiki and webqsp.",
     "indistinguishable from an unbuilt one. Those three slots — the retrieval cache for\n"
     "hotpotqa, 2wiki and webqsp — read `present: false` until 2026-09-10 and are now filled.\n"
     "They are **not** freeze-pinned the way the other three are: there is no freeze to copy a\n"
     "hash from, so each carries an explicit `provenance` block saying so rather than being\n"
     "written in as though it were. `SLOT_UNIFORMITY` records that the keys are uniform and the\n"
     "authority is not. See `MANIFEST_CACHE_SLOT_FILL.json`."),

    (P, "prepart_not_done",
     "- Retrieval caches for hotpotqa, 2wiki, webqsp.\n",
     "- ~~Retrieval caches for hotpotqa, 2wiki, webqsp.~~ **DONE 2026-09-10**, K=1000 both\n"
     "  channels, all three verified CONSISTENT against a local recomputation.\n"),

    (P, "prepart_record_index",
     "| `RETRIEVAL_CACHE.json` | deep-K caches (3 of 6) |",
     "| `RETRIEVAL_CACHE.json` | deep-K caches (**6 of 6** since 2026-09-10; drift from the\n"
     "5/5 pin is additive and adjudicated in `RETRIEVAL_CACHE_DRIFT_ADJUDICATION.json`) |"),

    (S, "substrate_universal_hole",
     "**(c) The one genuinely universal hole is retrieval caches.** No dense or SPLADE top-K"
     " candidate pool\nexists for any of the five. That is what actually blocks candidate/"
     "headroom analysis.",
     "**(c) ~~The one genuinely universal hole is retrieval caches.~~** **CLOSED 2026-09-10.**"
     " Dense and\nSPLADE top-1000 candidate pools now exist for all six. The thing this"
     " paragraph said was\nblocked — candidate/headroom analysis — is unblocked, with one"
     " declared caveat: a ceiling\nmeasured over a top-1000 pool is a **cache-limited** ceiling,"
     " not a corpus ceiling\n(`K_SEMANTICS_AND_EVAL_SPLITS.json`)."),

    (S, "substrate_step7",
     "7. **Retrieval caches** — the actual blocker for downstream work. Store one deep top-K"
     " and derive\n   shallower views by truncation; **do not pick K from test accuracy.** This"
     " needs GPU: 2wiki alone is\n   192,606 queries × 5,989,847 docs × 1536 dims.",
     "7. ~~**Retrieval caches** — the actual blocker for downstream work.~~ **DONE"
     " 2026-09-10**, exactly as\n   prescribed: one deep top-K (K=1000) stored, shallower views"
     " by truncation, and K **not** picked\n   from any accuracy. 2wiki's dense pass — 192,606"
     " queries × 5,989,847 docs × 1536 dims — did need a\n   GPU and took 37 s on a rented"
     " H100. Its SPLADE pass was split into 24 exact query shards and is\n   the package's one"
     " MIXED artifact: 13 shards on rented CPU before that workspace was disabled,\n   the other"
     " 11 built locally in a single shared corpus pass after every workspace in the pool\n   was"
     " found to have exceeded its spend limit. Per-shard provenance is in the merge record."),

    (R, "readme_stage6",
     "### Stage 6 — optional retrieval cache\n"
     "`scratchpad/build_retrieval_cache.py`, `verify_retrieval_cache.py`. Built for squad,\n"
     "musique, metaqa only; the other three carry the slot as explicitly absent.",
     "### Stage 6 — optional retrieval cache\n"
     "`scratchpad/build_retrieval_cache.py`, `verify_retrieval_cache.py`. Built for **all six**\n"
     "at K=1000, both channels. squad, musique and metaqa are pinned by `LOCKED_5_OF_5`;\n"
     "hotpotqa, 2wiki and webqsp were added 2026-09-10, after every freeze, and are pinned by\n"
     "no freeze — `verify_heavy_caches.py` is what vouches for them, by recomputing the top-K\n"
     "locally over the full corpus with the builder's own ranking key."),
]


def rel(p):
    return os.path.relpath(p, ROOT).replace(BS, "/")


def main():
    apply_ = "--apply" in sys.argv
    heavy = json.load(io.open(HEAVY, encoding="utf-8")) if os.path.isfile(HEAVY) else {}
    res = heavy.get("results", {}) or {}

    unverified = []
    for ds in DATASETS:
        for m in MODELS:
            p = os.path.join(FC, ds, "retrieval_cache", "%s_top1000.npz" % m)
            cell = res.get("%s.%s" % (ds, m), {}) or {}
            v = cell.get("verdict")
            # the frozen three are vouched for by LOCKED_5_OF_5 itself; the three added after
            # the freeze must be CONSISTENT in the verification record
            ok = os.path.isfile(p) and (ds in ("metaqa", "musique", "squad")
                                        or v == "CONSISTENT")
            if not ok:
                unverified.append("%s.%s(%s)" % (ds, m, "no file" if not os.path.isfile(p)
                                                 else v or "unverified"))
    if unverified:
        print("REFUSING: %d of 12 cells do not qualify -- %s"
              % (len(unverified), ", ".join(unverified)))
        return 1
    print("all 12 cache cells qualify")
    print("")

    files, notfound, dup, applied = {}, [], [], []
    for path in (H, P, S, R):
        raw = io.open(path, "rb").read()
        if b"\r\n" in raw:
            print("REFUSING: %s has CRLF line endings; the patch table is written in LF and "
                  "would produce mixed endings" % rel(path))
            return 1
        files[path] = {"before": raw, "buf": raw,
                       "sha_before": hashlib.sha256(raw).hexdigest(),
                       "bytes_before": len(raw)}

    for path, label, old, new in PATCHES:
        ob, nb = old.encode("utf-8"), new.encode("utf-8")
        n = files[path]["buf"].count(ob)
        if n == 0:
            notfound.append("%s :: %s" % (rel(path), label))
            print("   %-14s %-34s NOT FOUND" % (os.path.basename(path)[:14], label))
            continue
        if n > 1:
            dup.append("%s :: %s (%d occurrences)" % (rel(path), label, n))
            print("   %-14s %-34s AMBIGUOUS (%d)" % (os.path.basename(path)[:14], label, n))
            continue
        files[path]["buf"] = files[path]["buf"].replace(ob, nb)
        applied.append({"file": rel(path), "label": label,
                        "removed_bytes": len(ob), "added_bytes": len(nb)})
        print("   %-14s %-34s ok  %+d bytes"
              % (os.path.basename(path)[:14], label, len(nb) - len(ob)))

    print("")
    if notfound or dup:
        print("REFUSING: %d patch(es) unresolved -- nothing written. A prose edit that lands in "
              "the wrong paragraph is worse than one that does not land." % (len(notfound)
                                                                             + len(dup)))
        for x in notfound + dup:
            print("   !! %s" % x)
        return 1

    rec = {"RECORD": "DOC_CACHE_STATUS_PATCH",
           "_what": "updates the four prose docs that stated the retrieval caches cover three "
                    "of six datasets. The LOCKED_* records are not touched -- their now-false "
                    "caveats are superseded by FROZEN_CAVEAT_SUPERSESSION.json.",
           "why_prose_may_be_edited_here": "HANDOFF.md is hashed by LOCKED_5_OF_5 and had "
                                           "already diverged from that pin before this script "
                                           "ran. FROZEN_ARTIFACT_RECHECK.json records it as "
                                           "DOCS_DRIFT and states the reason: sealing a "
                                           "document that is expected to be revised was a "
                                           "design mistake in freeze_five, and LOCKED_6_OF_6 "
                                           "does not repeat it by not hashing HANDOFF.md.",
           "stale_snapshots_this_invalidates": {
               "FROZEN_ARTIFACT_RECHECK.json:HANDOFF.md.actual_sha256":
                   "a snapshot of a drifting file, not a pin; already stale before this run",
               "FROZEN_ARTIFACT_RECHECK.json:RETRIEVAL_CACHE.json.status":
                   "still reads MATCH at 9580 bytes; predates the post-freeze append and is "
                   "superseded by RETRIEVAL_CACHE_DRIFT_ADJUDICATION.json"},
           "patch_method": "exact substring, required to occur exactly once, applied at the "
                           "byte level so no line ending is rewritten as a side effect. A "
                           "patch that cannot be located refuses instead of guessing.",
           "patches": applied,
           "files": {},
           "applied": bool(apply_),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for path in (H, P, S, R):
        f = files[path]
        buf = f["buf"]
        rec["files"][rel(path)] = {
            "bytes_before": f["bytes_before"], "sha256_before": f["sha_before"],
            "bytes_after": len(buf), "sha256_after": hashlib.sha256(buf).hexdigest(),
            "changed": buf != f["before"]}
        print("%-46s %8d -> %8d bytes  sha %s"
              % (rel(path), f["bytes_before"], len(buf),
                 hashlib.sha256(buf).hexdigest()[:16]))
        if apply_ and buf != f["before"]:
            with io.open(path, "wb") as fh:
                fh.write(buf)

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("")
    print("%s %s" % ("WROTE" if apply_ else "DRY RUN (re-run with --apply) --", rel(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
