"""Prove each dataset has exactly ONE live substrate per role, then remove the dead skeletons.

The question this answers: "for all datasets there is only 1 nodes, queries whatever, only for
webqsp we should have 2 universes nothing else."

At the AUTHORITY level that is already true and always was: data/final_canonical/<ds>/ holds
exactly one nodes.jsonl per dataset. The 2-3 files under queries/ are train/dev/test SPLITS,
and graph/ + graph2/ is the documented family-completion split (graph/ is hash-pinned by
LOCKED_5_OF_5, so families built later install alongside it rather than mutating it). Neither
is duplication.

The confusing multiplicity is entirely on the ENCODING side, data/canonical/<tree>/, where the
tree name is a build-history artifact and does not have to match the dataset name. Three of
those trees are legitimately shared or split; two are dead skeletons.

TWO INDEPENDENT GATES, because either one alone gives a wrong answer here.

  G_RESOLVE  does anything actually READ this path?  Answered from POINTER_INDEX.json, which
             is the only authority on where a vector row comes from.  This is the gate with
             discriminating power.

  G_CITE     does any authority record NAME this path?  Answered by walking all ~494 json
             records under data/final_canonical.  On its own this gate is USELESS ON THIS
             TREE and it is worth saying why: data/canonical/2wiki/queries.jsonl and
             encodings/_src/queries are LIVE provenance sources and are cited by nothing,
             while data/canonical/webqsp has 231 citations and zero files.  A path can be
             uncited and load-bearing, or cited 231 times and dead.  So G_CITE is used only
             to separate a live pin from a historical mention, never as proof of death.

The citations that survive G_CITE on the dead set are all self-referential or historical, and
that has to be stated rather than filtered silently:

  CANONICAL_SUBSTRATE_MAP.json    this package's own SUPERSEDED registry.  Its job is to name
                                  dead paths so assert_not_superseded() can refuse them.  A
                                  hit here is the path being FORBIDDEN, not used.
  CANONICAL_CLEANUP_PLAN.json     this package's own deletion plan: 226 of webqsp's 231.
  ENCODING_SHARD_HASHES.json      fingerprints of shards that have since been superseded.
  WEBQSP_ENCODING_SHARD_HASHES    the hit is under the key THE_TRAP.its_webqsp_keys_point_at,
                                  i.e. the record documents its own stale keys.
  MANIFEST.json, webqsp/status    describe the METHOD of a past measurement ("exact, full
                                  streaming pass over .../documents.jsonl").
  V1_RETRIEVAL_ROLE_DECISION      cites two manifests as evidence they existed.

Frozen records are never edited to remove a stale mention.  A stale mention in a frozen record
is the correct historical state of that record; the contract is that an amendment gets a new
record with its own hash.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/canonical_dedup.py
  PYTHONHASHSEED=0 python src/dataset_canonical/canonical_dedup.py --apply
"""

import io
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
CAN = os.path.join(ROOT, "data", "canonical")
OUT = os.path.join(FC, "CANONICAL_DEDUP.json")

DATASETS = ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]

# A citation from one of these is not evidence of use.  Each entry says why, so the exclusion
# can be argued with instead of taken on faith.
NOT_A_USE = {
    "CANONICAL_SUBSTRATE_MAP.json": "this package's SUPERSEDED registry -- names it to forbid it",
    "CANONICAL_CLEANUP_PLAN.json": "this package's own deletion plan",
    "CANONICAL_DEDUP.json": "this file's own output",
    "ENCODING_SHARD_HASHES.json": "fingerprints of superseded shards",
    "WEBQSP_ENCODING_SHARD_HASHES.json": "the hit is under THE_TRAP.its_webqsp_keys_point_at",
    "CANONICALIZATION_AUDIT.json": "audit of a past state",
    "MANIFEST.json": "records the method of a past measurement",
    "status.json": "records the method of a past measurement",
    "V1_RETRIEVAL_ROLE_DECISION.json": "cites manifests as evidence they existed",
}

# The dead set.  (path, successor, why).  successor None means the artifact has no replacement
# because nothing needs one.
DEAD = [
    ("data/canonical/webqsp",
     "data/canonical/webqsp_rog_v1",
     "the pre-RoG webqsp encoding tree.  Its files were removed in the 30.80 GB pass; what is "
     "left is 15 empty directories that read as a second webqsp universe and contain nothing. "
     "POINTER_INDEX resolves every webqsp role into webqsp_rog_v1."),
    ("data/canonical/2wiki/graph_structural.tsv",
     "data/canonical/2wiki_universe/graph_structural.tsv",
     "359,549 edges over the superseded 398,354-doc tree.  The frozen family "
     "final_canonical/2wiki/graph/structural.npz (28,963,600 edges) cites the universe tsv as "
     "its source_tsv, not this one."),
    ("data/canonical/2wiki/graph_ner.tsv",
     "data/canonical/2wiki_universe/graph_ner.tsv",
     "3,274,931 edges over the 398k tree.  graph/ner.npz (34,067,241 edges) cites the universe "
     "tsv as its source_tsv."),
    ("data/canonical/2wiki/graph_knn.tsv",
     "data/final_canonical/2wiki/graph2/knn.npz",
     "908,911 kNN edges over the 398k tree.  The live kNN family was built EXHAUSTIVE from the "
     "universe vectors (13,643,063 edges) and never from a tsv -- the frozen graph/ manifest "
     "records knn as absent with the reason 'no graph_knn.tsv in the 2wiki_universe tree'."),
    ("data/canonical/2wiki/partition_map_C.json",
     None,
     "a 398k-tree partition map.  Partitioning has not been done on canonical_v1 at all, and "
     "the legacy L1/L2/L3 stack carries its own maps under data/ukb_storage.  Nothing reads it."),
    ("data/canonical/2wiki/partition_manifest_C.json",
     None,
     "the sidecar of the above."),
    ("data/canonical/2wiki/_ner_work",
     None,
     "64 bucket shards + 10 posting shards of NER staging for the dead 398k graph_ner.tsv.  "
     "Intermediate state of a superseded build."),
    ("data/canonical/2wiki/encodings/_src/docs",
     "data/canonical/2wiki_universe/encodings/_src/docs",
     "the source text of the 398k docs (_srcmeta n_items=398,354).  The universe successor "
     "carries 150 shards / 5,989,847 items.  2wiki doc vectors resolve into the universe tree, "
     "and resolved TEXT comes from final_canonical/2wiki/nodes.jsonl, not from _src."),
    ("data/canonical/2wiki/graph_manifest.json",
     "data/final_canonical/2wiki/graph/GRAPH_MANIFEST.json",
     "manifest of the dead 398k tsvs."),
    ("data/canonical/2wiki/ner_manifest.json",
     "data/final_canonical/2wiki/graph/GRAPH_MANIFEST.json",
     "manifest of the dead 398k NER tsv."),
    ("data/canonical/2wiki/knn_manifest.json",
     "data/final_canonical/2wiki/graph2/GRAPH_MANIFEST.json",
     "manifest of the dead 398k kNN tsv."),
    ("data/canonical/2wiki/graph_coverage_report.json",
     None,
     "coverage report over the dead 398k graph."),
    ("data/canonical/2wiki/document_manifest.json",
     "data/final_canonical/2wiki/dataset_manifest.json",
     "manifest of the 398k documents.jsonl, itself already removed."),
]

# Paths inside data/canonical/2wiki that are NOT dead, stated explicitly so the partial death
# of this tree is impossible to misread.  Deleting any of these destroys live query vectors.
LIVE_IN_2WIKI = [
    ("data/canonical/2wiki/encodings/dense/queries",
     "192,606 live 2wiki query vectors -- POINTER_INDEX.queries.2wiki.dense store PHASE_C"),
    ("data/canonical/2wiki/encodings/splade/queries",
     "live 2wiki SPLADE query vectors -- POINTER_INDEX.queries.2wiki.splade store PHASE_C"),
    ("data/canonical/2wiki/encodings/_src/queries",
     "the source text those query vectors were encoded from (provenance, uncited but live)"),
    ("data/canonical/2wiki/queries.jsonl",
     "the Phase-C query corpus the _src/queries shards hash back to"),
    ("data/canonical/2wiki/query_manifest.json",
     "its manifest"),
]


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def under(child, parent):
    """Is `child` at or below `parent`, matching on PATH COMPONENTS?

    A bare startswith() is wrong here and the failure is not hypothetical:
    "data/canonical/webqsp_rog_v1/encodings/dense/docs".startswith("data/canonical/webqsp")
    is True, so the LIVE RoG stores registered as readers of the DEAD webqsp tree and blocked
    its removal.  That direction fails safe.  The same bug on a live path -- a dead
    "…/2wiki" prefix swallowing "…/2wiki_universe" -- deletes 33.71 GB of live substrate.
    """
    return child == parent or child.startswith(parent + "/")


def norm(p):
    return p.replace("\\", "/")


def resolved_paths():
    """Every path POINTER_INDEX actually resolves a vector row out of."""
    pi = rj(os.path.join(FC, "POINTER_INDEX.json"))
    out = {}
    for section, node in (("docs", pi.get("datasets") or {}),
                          ("queries", pi.get("queries") or {})):
        for ds, d in node.items():
            if not isinstance(d, dict):
                continue
            for model in ("dense", "splade"):
                m = d.get(model)
                if not isinstance(m, dict):
                    continue
                for s in m.get("stores") or []:
                    p = s.get("path")
                    if p:
                        out.setdefault(norm(p), []).append("%s/%s/%s" % (ds, section, model))
    return out


def citations():
    """Every data/canonical path named by any authority record, with the naming records."""
    cited, unreadable = {}, []

    def walk(o, src):
        if isinstance(o, dict):
            for v in o.values():
                walk(v, src)
        elif isinstance(o, list):
            for v in o:
                walk(v, src)
        elif isinstance(o, str):
            s = norm(o)
            if s.startswith("data/canonical/"):
                cited.setdefault(s, set()).add(src)

    for dp, _dn, fn in os.walk(FC):
        if "freebase_v3" in norm(dp):
            continue
        for f in fn:
            if not f.endswith(".json"):
                continue
            p = os.path.join(dp, f)
            try:
                walk(rj(p), os.path.basename(p))
            except Exception as ex:
                # Named, not counted: a citation scan that skipped a record could have missed a
                # citation to a path it then deleted, so each skipped file is listed and then
                # raw-scanned in unreadable_raw_scan().
                unreadable.append({"file": norm(os.path.relpath(p, ROOT)),
                                   "bytes": os.path.getsize(p),
                                   "error": "%s: %s" % (type(ex).__name__, str(ex)[:80])})
    return cited, unreadable


def unreadable_raw_scan(unreadable, deleted):
    """What a record the JSON parser could not read might still cite, read as raw bytes.

    json.load fails on these (they are JSON-Lines, one object per line, hence "Extra data:
    line 2"), but a path string is a path string in any encoding, so stream every line and
    count occurrences of the two repository prefixes. A citation to a DELETED path would be
    the hazard; a citation to any repository path at all is what is reported.

    ZERO_JOIN_INVARIANT: a zero-hit result from a scanner is not evidence until the scanner is
    shown to hit on a record known to cite. POINTER_INDEX.json names data/canonical trees, so
    it is the positive control, and the verdict is VOID if the control does not fire.
    """
    needles = [b"data/canonical/", b"data/final_canonical/"]

    def scan(path):
        hits = {n.decode(): 0 for n in needles}
        lines = 0
        with open(path, "rb") as f:
            for line in f:
                lines += 1
                for n in needles:
                    if n in line:
                        hits[n.decode()] += line.count(n)
        return lines, hits

    _cl, chits = scan(os.path.join(FC, "POINTER_INDEX.json"))
    control_ok = sum(chits.values()) > 0
    rows = []
    for u in unreadable:
        path = os.path.join(ROOT, u["file"])
        lines, hits = scan(path)
        # a deleted path can only be cited if its prefix is; with a prefix count of 0 the
        # deleted-path count is 0 by containment -- reported as measured, never assumed
        cites_deleted = 0
        if sum(hits.values()):
            with open(path, "rb") as f:
                blob = f.read()
            cites_deleted = sum(blob.count(d.encode()) for d in deleted)
        rows.append(dict(u, lines=lines, repository_path_strings=hits,
                         deleted_path_citations=cites_deleted))
    any_repo = any(sum(r["repository_path_strings"].values()) for r in rows)
    if not control_ok:
        verdict = "VOID -- positive control did not fire"
    elif not any_repo:
        verdict = ("NO CITATION MISSED -- none of the unreadable records names any repository "
                   "path; they are upstream source files (NSM JSON-Lines), not authority records")
    else:
        verdict = "REPOSITORY PATHS PRESENT -- see files[].deleted_path_citations"
    return {
        "method": ("streamed byte search for the repository prefixes in every record json.load "
                   "rejected; deleted-path citations counted against APPLIED_HISTORY paths"),
        "positive_control": {"file": "POINTER_INDEX.json", "hits": chits, "fired": control_ok},
        "files": rows,
        "any_repository_path_cited": any_repo,
        "any_deleted_path_cited": any(r["deleted_path_citations"] for r in rows),
        "VERDICT": verdict,
    }


def repair_patch_staging():
    """Label the flat + sharded double copy under _dense_repair_patch that pass 2 never classified.

    consolidate_repair.py joined the Modal output parts (<store>__pNNN/) into one flat
    <store>/dense.npy per channel and proved the join; it did not remove the parts.
    POINTER_INDEX resolves the flat file only, so the parts are staging. They are checked here
    to be byte-identical to the flat store (not assumed) and are therefore a DEDUP CANDIDATE.
    They are NOT deleted by this script: cleanup_plan.py names _dense_repair_patch as a trap it
    never deletes under, and removing them is a decision to record, not to take in passing.
    """
    import glob
    import hashlib
    import numpy as np
    base = os.path.join(FC, "_dense_repair_patch")
    pi_text = io.open(os.path.join(FC, "POINTER_INDEX.json"), encoding="utf-8").read()
    out = []
    tot_bytes = 0
    for flat in sorted(d for d in glob.glob(os.path.join(base, "*"))
                       if os.path.isdir(d) and "__p" not in os.path.basename(d)):
        store = os.path.basename(flat)
        parts = sorted(glob.glob(os.path.join(base, store + "__p*")))
        if not parts:
            continue
        part_bytes = sum(os.path.getsize(os.path.join(q, f)) for q in parts for f in os.listdir(q))
        tot_bytes += part_bytes
        fa = np.load(os.path.join(flat, "dense.npy"), mmap_mode="r")
        h_flat = hashlib.sha256(np.ascontiguousarray(fa).tobytes()).hexdigest()
        h = hashlib.sha256()
        n = 0
        for q in parts:
            a = np.load(os.path.join(q, "dense.npy"), mmap_mode="r")
            h.update(np.ascontiguousarray(a).tobytes())
            n += int(a.shape[0])
        rows_flat = rj(os.path.join(flat, "dense_rows.json"))
        rows_parts = sum((rj(os.path.join(q, "dense_rows.json")) for q in parts), [])
        identical = (h.hexdigest() == h_flat) and (n == int(fa.shape[0])) and (rows_flat == rows_parts)
        out.append({
            "store": norm(os.path.relpath(flat, ROOT)),
            "parts": [norm(os.path.relpath(q, ROOT)) for q in parts],
            "part_bytes": part_bytes,
            "flat_rows": int(fa.shape[0]), "part_rows": n,
            "flat_resolved_by_pointer_index": ("_dense_repair_patch/%s/dense.npy" % store) in pi_text,
            "parts_resolved_by_pointer_index": ("%s__p" % store) in pi_text,
            "parts_concatenate_to_flat_byte_identical": bool(identical),
            "classification": "STAGING_PARTS_OF_CONSOLIDATED_STORE -- dedup candidate, NOT deleted",
        })
    return {"why_unclassified_until_now": (
                "pass 1 (cleanup_plan.py) protects _dense_repair_patch as a whole and pass 2 "
                "classified trees, not the parts inside a protected tree"),
            "stores": out, "total_part_bytes": tot_bytes,
            "all_parts_byte_identical_to_flat": all(o["parts_concatenate_to_flat_byte_identical"]
                                                    for o in out),
            "any_part_resolved_by_pointer_index": any(o["parts_resolved_by_pointer_index"]
                                                      for o in out),
            "decision_required": ("delete the parts (reclaims total_part_bytes) or keep them as "
                                  "the record of the Modal outputs -- either is fine once "
                                  "labelled; this record labels, it does not choose")}


def code_refs(rel):
    """Files under src/ that mention this path.  substrate_map.py is the registry that FORBIDS
    these paths, so a hit there is the opposite of a consumer."""
    try:
        r = subprocess.run(["grep", "-rl", "--include=*.py", rel, "src/"],
                           cwd=ROOT, capture_output=True, text=True)
        hits = [norm(x) for x in r.stdout.split() if x.strip()]
    except Exception:
        return [], []
    registry = [h for h in hits if "dataset_canonical/" in h]
    return [h for h in hits if h not in registry], registry


def du(p):
    if os.path.isfile(p):
        return 1, os.path.getsize(p)
    n = b = 0
    for dp, _dn, fn in os.walk(p):
        for f in fn:
            try:
                b += os.path.getsize(os.path.join(dp, f))
                n += 1
            except OSError:
                pass
    return n, b


def gb(b):
    return "%.2f GB" % (b / 1073741824.0) if b >= 1073741824 else "%.2f MB" % (b / 1048576.0)


def uniqueness():
    """One nodes.jsonl / one query corpus / one graph per dataset, at the authority level."""
    rows = []
    for ds in DATASETS:
        d = os.path.join(FC, ds)
        nodes = [f for f in ("nodes.jsonl",) if os.path.exists(os.path.join(d, f))]
        qd = os.path.join(d, "queries")
        splits = sorted(f for f in os.listdir(qd) if f.endswith(".jsonl")) if os.path.isdir(qd) else []
        trees = [t for t in ("graph", "graph2") if os.path.isdir(os.path.join(d, t))]
        rows.append({"dataset": ds, "nodes_files": len(nodes), "query_splits": splits,
                     "graph_dirs": trees})
    return rows


def main():
    apply = "--apply" in sys.argv
    res = resolved_paths()
    cited, unreadable = citations()

    print("=" * 96)
    print("PART 1 -- one nodes / one query corpus / one graph per dataset (authority level)")
    print("=" * 96)
    print("%-10s %7s  %-34s %s" % ("dataset", "nodes", "queries/ splits", "graph dirs"))
    bad = 0
    for r in uniqueness():
        if r["nodes_files"] != 1:
            bad += 1
        print("%-10s %7d  %-34s %s" % (r["dataset"], r["nodes_files"],
                                       ",".join(x.replace(".jsonl", "") for x in r["query_splits"]),
                                       "+".join(r["graph_dirs"])))
    print("\n  nodes.jsonl per dataset != 1: %d   (splits are train/dev/test, not duplicates;" % bad)
    print("  graph+graph2 is the family-completion split -- graph/ is pinned by LOCKED_5_OF_5)")

    print("\n" + "=" * 96)
    print("PART 2 -- which physical data/canonical tree each dataset actually reads")
    print("=" * 96)
    readers = {}
    for p, who in res.items():
        if p.startswith("data/canonical/"):
            readers.setdefault(p.split("/")[2], set()).update(who)
    trees = sorted(os.listdir(CAN))
    print("%-18s %7s %10s  %s" % ("tree", "files", "bytes", "read by"))
    dead_trees = []
    for t in trees:
        n, b = du(os.path.join(CAN, t))
        who = sorted(readers.get(t, []))
        if not who:
            dead_trees.append(t)
        print("%-18s %7d %10s  %s" % (t, n, gb(b), ", ".join(who) if who else "*** NOBODY ***"))
    print("\n  trees nothing resolves into: %s" % (", ".join(dead_trees) or "none"))

    print("\n" + "=" * 96)
    print("PART 3 -- the dead set, gated")
    print("=" * 96)
    plan, blocked, tot_n, tot_b = [], [], 0, 0
    for rel, succ, why in DEAD:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print("\n  [gone]    %s" % rel)
            continue
        n, b = du(p)
        g_resolve = not any(under(norm(k), rel) for k in res)
        consumers, registry = code_refs(rel)
        g_code = not consumers
        g_succ = succ is None or os.path.exists(os.path.join(ROOT, succ))
        naming = sorted({c for k, v in cited.items() if under(k, rel) for c in v})
        real = [c for c in naming if c not in NOT_A_USE]
        g_cite = not real
        ok = g_resolve and g_code and g_succ and g_cite
        print("\n  [%s] %s" % ("DELETE" if ok else "BLOCK ", rel))
        print("      %d files, %s" % (n, gb(b)))
        print("      G_RESOLVE nothing resolves into it ........ %s" % g_resolve)
        print("      G_CODE    no consumer in src/ ............. %s%s" % (
            g_code, "" if not registry else "  (registry-only: %s)" % ", ".join(
                os.path.basename(x) for x in registry)))
        print("      G_SUCC    successor present ............... %s  %s" % (g_succ, succ or "n/a"))
        print("      G_CITE    no live pin in a record ......... %s%s" % (
            g_cite, "" if not naming else "  (historical: %s)" % ", ".join(naming[:3])))
        print("      why: %s" % why)
        if ok:
            plan.append({"path": rel, "successor": succ, "why": why, "files": n, "bytes": b})
            tot_n += n
            tot_b += b
        else:
            blocked.append({"path": rel, "g_resolve": g_resolve, "g_code": g_code,
                            "g_succ": g_succ, "g_cite": g_cite, "live_referrers": real,
                            "src_consumers": consumers})

    print("\n" + "=" * 96)
    print("PART 4 -- what must NOT be touched in data/canonical/2wiki (it is half live)")
    print("=" * 96)
    for rel, why in LIVE_IN_2WIKI:
        n, b = du(os.path.join(ROOT, rel)) if os.path.exists(os.path.join(ROOT, rel)) else (0, 0)
        print("  KEEP %-46s %5d files %10s  %s" % (rel.replace("data/canonical/2wiki/", ""),
                                                   n, gb(b), why))

    print("\n" + "=" * 96)
    print("total deletable: %d paths, %d files, %s   blocked: %d" % (
        len(plan), tot_n, gb(tot_b), len(blocked)))
    print("=" * 96)

    if apply:
        if blocked:
            print("\nREFUSING --apply: %d path(s) failed a gate." % len(blocked))
            return 1
        # last-second re-check of the live set, then delete
        for rel, _why in LIVE_IN_2WIKI:
            if not os.path.exists(os.path.join(ROOT, rel)):
                print("\nREFUSING --apply: live path already missing: %s" % rel)
                return 1
        nd = 0
        for e in plan:
            p = os.path.join(ROOT, e["path"])
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            nd += 1
        print("\nAPPLIED: removed %d paths, freed %s" % (nd, gb(tot_b)))
    else:
        print("\ndry run -- pass --apply to delete")

    # A dry run must NOT erase the record of a completed deletion.  Re-running this script to
    # check idempotence overwrote applied=true with applied=false and an empty deleted list,
    # because after the deletion every path reports [gone] and drops out of `plan`.  The
    # deletion was real and the record then denied it.  So an applied record's payload is
    # carried forward under APPLIED_HISTORY and only ever appended to.
    prior = {}
    if os.path.isfile(OUT):
        try:
            prior = rj(OUT)
        except Exception:
            prior = {}
    history = list(prior.get("APPLIED_HISTORY") or [])
    if prior.get("applied") and prior.get("deleted_or_deletable"):
        history.append({"applied_record": prior.get("generated_utc") or "unrecorded",
                        "paths": prior["deleted_or_deletable"]})
    if apply and not blocked and plan:
        history.append({"applied_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                        "paths": plan, "files": tot_n, "bytes": tot_b})

    rec = {"RECORD": "CANONICAL_DEDUP_V1",
           "APPLIED_HISTORY": history,
           "APPLIED_HISTORY_NOTE":
               "every completed deletion, appended. A dry run adds nothing and erases nothing. "
               "Read deletion facts from here, never from deleted_or_deletable, which is the "
               "CURRENT dry-run plan and is correctly empty once the work is done.",
           "generated_by": "src/dataset_canonical/canonical_dedup.py",
           "applied": bool(apply and not blocked),
           "authority_uniqueness": uniqueness(),
           "tree_readers": {t: sorted(readers.get(t, [])) for t in trees},
           "trees_nothing_resolves_into": dead_trees,
           "deleted_or_deletable": plan,
           "blocked": blocked,
           "must_keep_in_2wiki": [{"path": p, "why": w} for p, w in LIVE_IN_2WIKI],
           "citations_not_treated_as_use": NOT_A_USE,
           "records_unreadable_during_citation_scan": len(unreadable),
           "records_unreadable_during_citation_scan_detail": unreadable_raw_scan(
               unreadable, sorted({norm(e["path"]) for h in history for e in h.get("paths", [])
                                   if isinstance(e, dict) and e.get("path")})),
           "repair_patch_staging_duplication": repair_patch_staging()}
    rec["generated_utc"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("wrote %s" % norm(os.path.relpath(OUT, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
