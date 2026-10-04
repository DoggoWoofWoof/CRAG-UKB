"""Items 2 and 4: file the K-semantics rule and the per-dataset evaluation split BEFORE any
ceiling number exists.

WHY THESE TWO GO IN ONE RECORD

  Both are preconditions on the same table, and both are the kind of thing that is discovered
  in a footnote if it is not declared first.  A headroom row is a triple -- (what pool, which
  queries, which gold) -- and K semantics fixes the first, the split fixes the second, and
  GOLD_FIELD_CHECK already fixed the third.

ITEM 2: THE TWO CEILINGS, AND WHY ONLY ONE OF THEM IS WORTH PRINTING

  corpus ceiling   the best any system could do if it could see the whole corpus.  It is NOT
                   1.0 for all six, and the first revision of this record was wrong to say so.
                   For the five text corpora it is 1.0 by construction -- the corpus was never
                   subsetted, every gold ref in every labelled split resolves to a corpus
                   position, and no labelled query has zero gold (GOLD_FIELD_CHECK, per split,
                   re-read here rather than quoted).  For webqsp it is a COLUMN: the corpus is
                   the RoG-standard subgraph, only 32,106 of the 51,438 upstream gold refs
                   survived the bridge into it (62.42%), and 143 of 4,737 queries have no gold
                   in the corpus at all, so no retriever can reach 1.0 there and any table that
                   includes webqsp carries its ceiling beside the row (the manifest's own
                   caveat: never average across the six without conditioning on this).
                   GOLD_FIELD_CHECK counts the refs already IN the query files, i.e. after the
                   bridge; it cannot see the 37.58% that did not survive, which is why "all
                   resolve" there was mistaken for "ceiling is 1.0".  The pre-bridge
                   denominator is measured from the NSM source files kept in the package.

  pool ceiling     the best any reranker, GNN or reader could do given the candidate pool it is
                   actually handed.  This is the informative quantity, and it is the one the
                   cache supports: recall of gold within the cached top-K.

  THE TRAP is reporting the second and calling it the first.  A pool ceiling measured over a
  top-1000 cache is bounded by that cache, so it is a CACHE-LIMITED ceiling; it answers "how
  much headroom is there above this retriever at this depth", not "how much headroom is there
  in this corpus".  The two differ by exactly the gold that the retriever failed to surface,
  which is the quantity under study -- so conflating them silently subtracts the finding.

  RULES, in force from this record onward:

  R1  Every ceiling number is reported with all five of (K, model, pool provenance, dataset,
      split).  A bare "ceiling" column is not admissible.
  R2  A ceiling computed from data/final_canonical/<ds>/retrieval_cache/<model>_top1000.npz is
      named `pool_ceiling@K` and never `candidate_ceiling` or `corpus_ceiling`.
  R3  `corpus_ceiling` is reserved for the whole-corpus quantity.  For metaqa, hotpotqa, 2wiki,
      musique and squad it is 1.0 by construction (evidence per split in
      K_SEMANTICS.corpus_ceiling) and is stated once, not columned.  For webqsp it is a COLUMN,
      per split, at both levels -- reference-level (gold refs in the corpus / upstream gold
      refs) and query-level (queries with >= 1 gold in the corpus / queries) -- and no table
      may average webqsp with the other five without conditioning on it.
  R4  Whether K=1000 BINDS is not assumed either way -- it is measured, by reporting the
      saturation curve recall@{1,5,10,20,100,1000} beside every pool ceiling.  If recall@1000
      is materially above recall@100 the pool is still filling at depth and the ceiling is
      cache-limited in a way that matters; if it has flattened, the cache depth is not the
      binding constraint.  Either way the reader sees which.
  R5  K is not selected.  The cache is materialised at ONE depth, 1000, and every shallower K
      is a prefix of it, so no depth was ever chosen against an outcome.  This restates the
      standing rule "Do not select K from test accuracy" as a property of the artifact rather
      than a promise about behaviour.
  R6  Membership is by canonical POSITION, not by row.  A gold at canonical position p is in
      the pool for query q iff p appears in ids[q].  The pointer index is many-to-one, so two
      distinct positions can share a vector and therefore a score; the cache resolves that by
      the score/position key, and a membership test over distinct source rows would over-count.

ITEM 4: THE EVALUATION SPLIT IS A DECLARED CHOICE, NOT A DEFAULT

  "Test" is not uniformly available and the standing rule bars test results, so the split is
  declared per dataset here, with the reason:

    metaqa    dev          a real labelled dev split exists
    hotpotqa  validation   the benchmark's name for its labelled held-out split; its `test`
                           split carries 7,405 queries with EMPTY gold arrays (GOLD_ABSENT)
    2wiki     dev          its `test` gold is hidden (12,576 queries, GOLD_ABSENT)
    musique   dev          its `test` gold is hidden (2,459 queries, GOLD_ABSENT)
    squad     dev          squad has no test split at all
    webqsp    train_holdout  THE ONE OPEN CASE.  webqsp has no dev split, and its only labelled
                           non-train split is `test` (1,639 queries), which collides with the
                           standing bar on test results.  Rather than take an exception, a
                           seed-pinned holdout is carved from train -- query subsetting is
                           explicitly allowed, corpus subsetting by query is not, and nothing
                           about the corpus changes.  The carve is PYTHONHASHSEED-independent:
                           it sorts query_ids and takes a deterministic stride, so it does not
                           depend on dict order, file order, or numpy's RNG version.

  webqsp's `test` split is left untouched and unread.  It is not deleted and not renamed; it is
  simply not the evaluation split, and this record is where that is written down.

WHAT IS MEASURED HERE RATHER THAN ASSERTED

  The cache is built over the FULL concatenated query pointer index -- one row per canonical
  query position across all splits -- so a per-split headroom row is a SLICE of it, and the
  slice indices have to be right.  POINTER_INDEX declares split_order ["train","dev",
  "validation","test"], but a declared order is not a measurement.  So the mapping is read from
  <ds>/queries/pointer_index/query_ids.json, which is position-ordered by construction and is
  the same file the RESOLUTION_TEXT_GATE used when it proved every query row's vector came from
  that query's canonical text.  Contiguity is CHECKED, not assumed: if a split's positions are
  not one contiguous range this record stores an explicit index array instead of a range, and
  says so.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/k_semantics_and_splits.py
"""

import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "K_SEMANTICS_AND_EVAL_SPLITS.json")
DATASETS = ["metaqa", "webqsp", "hotpotqa", "2wiki", "musique", "squad"]
SPLITS = ["train", "dev", "validation", "test"]

EVAL = {"metaqa": "dev", "hotpotqa": "validation", "2wiki": "dev",
        "musique": "dev", "squad": "dev", "webqsp": "train_holdout"}
EVAL_REASON = {
    "metaqa": "a real labelled dev split exists",
    "hotpotqa": "the benchmark's name for its labelled held-out split; its `test` split has "
                "empty gold arrays",
    "2wiki": "its `test` gold is hidden",
    "musique": "its `test` gold is hidden",
    "squad": "squad has no test split at all",
    "webqsp": "no dev split, and its only labelled non-train split is `test`, which the "
              "standing bar on test results excludes; a seed-pinned holdout is carved from "
              "train instead. Query subsetting is allowed; the corpus is untouched.",
}
REVISION_NOTE = ("R3 corrected: corpus ceiling is 1.0 by construction for the five text corpora "
                 "and a per-split COLUMN for webqsp (32,106 of 51,438 upstream gold refs in the "
                 "corpus; 143 zero-gold queries); the hardcoded basis total 3,850,471 replaced by "
                 "the measured per-split sum")
WEBQSP_HOLDOUT_STRIDE = 2      # every 2nd train query by sorted query_id -> 1,549 of 3,098


def corpus_ceiling_column():
    """Measure the corpus ceiling per dataset and split instead of asserting it.

    Source of the in-file counts: GOLD_FIELD_CHECK.json (queries, queries_with_gold, gold refs
    resolved per labelled split). Source of webqsp's pre-bridge denominator: the NSM source
    files under webqsp/_acquisition (answers[].kb_id per question, keyed by the same query ids
    as our split files), cross-checked against ID_BRIDGE.gold_refs and
    webqsp/CANONICAL_V1_BUILD.queries.gold_reference_level. A dataset is "1.0 by construction"
    only if every labelled split resolves every ref AND has no zero-gold query AND had no
    bridge loss; anything else is a column.
    """
    g = json.load(io.open(os.path.join(FC, "GOLD_FIELD_CHECK.json"), encoding="utf-8"))
    out = {"basis": "GOLD_FIELD_CHECK re-read per split; webqsp pre-bridge refs measured from "
                    "the NSM source files in the package",
           "datasets": {}, "total_gold_refs_resolved_in_files": 0}
    for ds, v in g["results"].items():
        splits = {}
        labelled_ok = True
        for sp, x in v["splits"].items():
            if x.get("verdict") == "GOLD_ABSENT":
                splits[sp] = {"gold": "ABSENT (hidden test labels); not an eval split"}
                continue
            q, qg = x["queries"], x["queries_with_gold"]
            r, rr = x["gold_refs"], x["gold_refs_resolved"]
            splits[sp] = {"queries": q, "queries_with_gold": qg, "queries_zero_gold": q - qg,
                          "gold_refs_in_file": r, "gold_refs_resolved": rr,
                          "reference_level_in_file": round(rr / r, 4) if r else None,
                          "query_level_any_gold": round(qg / q, 4) if q else None}
            out["total_gold_refs_resolved_in_files"] += rr
            if q != qg or r != rr:
                labelled_ok = False
        out["datasets"][ds] = {"splits": splits}
        if ds == "webqsp":
            out["datasets"][ds].update(_webqsp_pre_bridge(splits))
            out["datasets"][ds]["corpus_ceiling"] = "COLUMN"
        else:
            out["datasets"][ds]["corpus_ceiling"] = 1.0 if labelled_ok else "COLUMN"
            out["datasets"][ds]["why"] = (
                "corpus never subsetted; every gold ref in every labelled split resolves and no "
                "labelled query has zero gold" if labelled_ok else
                "a labelled split has unresolved refs or zero-gold queries -- see splits")
    return out


def _webqsp_pre_bridge(splits):
    """Per-split upstream gold refs for webqsp from the NSM source, plus the query-level
    coverage classes carried in our query files (FULL / PARTIAL / NONE_IN_CORPUS /
    NO_GOLD_GIVEN). NSM's dev_simple is part of our train split (3,098 = 2,848 + 250)."""
    base = os.path.join(FC, "webqsp", "_acquisition", "nsm", "extracted", "webqsp", "webqsp")
    src_files = {"train": ["train_simple.json", "dev_simple.json"], "test": ["test_simple.json"]}
    pre = {}
    for sp, files in src_files.items():
        refs = {}
        for fn in files:
            fp = os.path.join(base, fn)
            if not os.path.isfile(fp):
                return {"pre_bridge": {"error": "NSM source file missing: %s" % fn}}
            with io.open(fp, encoding="utf-8") as f:
                for line in f:
                    o = json.loads(line)
                    refs[o["id"]] = len(o.get("answers") or [])
        pre[sp] = refs
    cov = {}
    perq = {}   # query_id -> (n gold in corpus, n upstream refs, coverage class), for the holdout
    for sp in ("train", "test"):
        c = {}
        ids = set()
        with io.open(os.path.join(FC, "webqsp", "queries", "%s.jsonl" % sp), encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                ids.add(o["query_id"])
                c[o.get("gold_coverage")] = c.get(o.get("gold_coverage"), 0) + 1
                perq[o["query_id"]] = (len(o.get("gold_node_ids") or []),
                                       pre[sp].get(o["query_id"], 0), o.get("gold_coverage"))
        if sp == "train":
            train_ids = set(ids)
        n_pre = sum(pre[sp].get(i, 0) for i in ids)
        missing_ids = len(ids - set(pre[sp]))
        e = splits[sp]
        e["gold_refs_upstream"] = n_pre
        e["gold_refs_lost_at_bridge"] = n_pre - e["gold_refs_resolved"]
        e["reference_level_corpus_ceiling"] = round(e["gold_refs_resolved"] / n_pre, 4) if n_pre else None
        e["query_level_corpus_ceiling"] = e["query_level_any_gold"]
        e["gold_coverage_classes"] = c
        e["answerable_queries"] = e["queries"] - c.get("NO_GOLD_GIVEN", 0)
        e["query_level_corpus_ceiling_of_answerable"] = round(
            e["queries_with_gold"] / e["answerable_queries"], 4) if e["answerable_queries"] else None
        e["query_ids_missing_from_nsm_source"] = missing_ids
        cov[sp] = e
    # the declared eval split: every WEBQSP_HOLDOUT_STRIDE-th train query by sorted query_id --
    # the same carve main() records under results.webqsp.splits.train_holdout
    tr_ids = sorted(train_ids)          # membership in train.jsonl, exactly as main() carves it
    hold = tr_ids[::WEBQSP_HOLDOUT_STRIDE]
    import hashlib
    carve_sha = hashlib.sha256(",".join(hold).encode()).hexdigest()
    hq = len(hold)
    h_gold = sum(1 for q in hold if perq[q][0] > 0)
    h_refs = sum(perq[q][0] for q in hold)
    h_pre = sum(perq[q][1] for q in hold)
    h_cls = {}
    for q in hold:
        h_cls[perq[q][2]] = h_cls.get(perq[q][2], 0) + 1
    h_ans = hq - h_cls.get("NO_GOLD_GIVEN", 0)
    splits["train_holdout"] = {
        "queries": hq, "queries_with_gold": h_gold, "queries_zero_gold": hq - h_gold,
        "gold_refs_upstream": h_pre, "gold_refs_resolved": h_refs,
        "gold_refs_lost_at_bridge": h_pre - h_refs,
        "reference_level_corpus_ceiling": round(h_refs / h_pre, 4) if h_pre else None,
        "query_level_corpus_ceiling": round(h_gold / hq, 4) if hq else None,
        "gold_coverage_classes": h_cls, "answerable_queries": h_ans,
        "query_level_corpus_ceiling_of_answerable": round(h_gold / h_ans, 4) if h_ans else None,
        "carve": "train query_ids sorted, stride %d -- identical to results.webqsp.splits."
                 "train_holdout" % WEBQSP_HOLDOUT_STRIDE,
        "this_is_the_declared_eval_split": True,
        "carve_ids_sha256": carve_sha,
    }
    tot_pre = sum(cov[sp]["gold_refs_upstream"] for sp in cov)
    tot_res = sum(cov[sp]["gold_refs_resolved"] for sp in cov)
    bridge = json.load(io.open(os.path.join(FC, "ID_BRIDGE.json"), encoding="utf-8"))
    build = json.load(io.open(os.path.join(FC, "webqsp", "CANONICAL_V1_BUILD.json"), encoding="utf-8"))
    grl = ((build.get("queries") or {}).get("gold_reference_level") or {})
    bridge_refs = None
    txt = json.dumps(bridge)
    if '"gold_refs": %d' % tot_pre in txt:
        bridge_refs = tot_pre
    return {
        "pre_bridge": {
            "upstream_gold_refs": tot_pre, "gold_refs_in_corpus": tot_res,
            "reference_level_corpus_ceiling_overall": round(tot_res / tot_pre, 4) if tot_pre else None,
            "lost_at_bridge": tot_pre - tot_res,
            "lost_at_bridge_pct": round(100.0 * (tot_pre - tot_res) / tot_pre, 2) if tot_pre else None,
            "queries_zero_gold": sum(cov[sp]["queries_zero_gold"] for sp in cov),
            "cross_checks": {
                "ID_BRIDGE.gold_refs_equals_measured": bridge_refs == tot_pre,
                "CANONICAL_V1_BUILD.gold_reference_level": grl,
                "build_joined_equals_measured": grl.get("joined") == tot_res,
            },
            "note": "NO_GOLD_GIVEN queries have no gold upstream either (not a corpus gap); "
                    "NONE_IN_CORPUS queries have gold upstream that the corpus does not hold. "
                    "Both are zero-gold for any retriever; the answerable-conditioned ceiling "
                    "excludes only the first.",
        }
    }


def ranges(positions):
    """positions -> contiguous [lo,hi) if it is one run, else None."""
    p = np.sort(np.asarray(positions))
    if p.size and int(p[-1] - p[0]) == p.size - 1:
        return int(p[0]), int(p[-1]) + 1
    return None


def main():
    res, notes, fails = {}, [], []
    for ds in DATASETS:
        qd = os.path.join(FC, ds, "queries", "pointer_index")
        qids = json.load(io.open(os.path.join(qd, "query_ids.json"), encoding="utf-8"))
        pos_of = {q: i for i, q in enumerate(qids)}
        if len(pos_of) != len(qids):
            fails.append("%s: query_ids.json has %d duplicate ids"
                         % (ds, len(qids) - len(pos_of)))
        nrows = int(np.load(os.path.join(qd, "dense.npz"))["row"].shape[0])
        if nrows != len(qids):
            fails.append("%s: %d pointer rows but %d query_ids" % (ds, nrows, len(qids)))

        dsres = {"n_canonical_queries": len(qids), "pointer_rows": nrows,
                 "eval_split": EVAL[ds], "eval_split_reason": EVAL_REASON[ds], "splits": {}}
        for sp in SPLITS:
            p = os.path.join(FC, ds, "queries", sp + ".jsonl")
            if not os.path.isfile(p):
                continue
            ids, gold = [], 0
            with io.open(p, encoding="utf-8") as f:
                for ln in f:
                    o = json.loads(ln)
                    ids.append(o["query_id"])
                    if o.get("gold_node_ids"):
                        gold += 1
            miss = [q for q in ids if q not in pos_of]
            if miss:
                fails.append("%s.%s: %d query_ids absent from query_ids.json"
                             % (ds, sp, len(miss)))
                continue
            pos = [pos_of[q] for q in ids]
            rg = ranges(pos)
            e = {"queries": len(ids), "queries_with_gold": gold,
                 "contiguous": rg is not None,
                 "position_range": list(rg) if rg else None,
                 "position_min": int(min(pos)), "position_max": int(max(pos)),
                 "gold_absent": gold == 0}
            if rg is None:
                e["note"] = ("positions are NOT one contiguous run; a headroom row must slice "
                             "this split with an explicit index array, not a range")
                notes.append("%s.%s is non-contiguous in canonical query position" % (ds, sp))
            dsres["splits"][sp] = e
            print("%-9s %-11s q=%-7d gold_q=%-7d %s  %s"
                  % (ds, sp, len(ids), gold,
                     ("positions %d..%d" % (rg[0], rg[1] - 1)) if rg
                     else "NON-CONTIGUOUS %d..%d" % (min(pos), max(pos)),
                     "<= EVAL" if sp == EVAL[ds] else ("GOLD_ABSENT" if gold == 0 else "")),
                  flush=True)

        # webqsp: the carve, deterministic and stated in full
        if ds == "webqsp":
            tr = sorted(q for q in qids
                        if q in {json.loads(l)["query_id"] for l in
                                 io.open(os.path.join(FC, ds, "queries", "train.jsonl"),
                                         encoding="utf-8")})
            hold = tr[::WEBQSP_HOLDOUT_STRIDE]
            hp = sorted(pos_of[q] for q in hold)
            main_carve_sha = __import__("hashlib").sha256(",".join(hold).encode()).hexdigest()
            dsres["splits"]["train_holdout"] = {
                "queries": len(hold), "derived_from": "train",
                "rule": "sort train query_ids lexicographically, take every %dth"
                        % WEBQSP_HOLDOUT_STRIDE,
                "why_not_random": "a lexicographic stride needs no RNG, so it does not depend "
                                  "on PYTHONHASHSEED, numpy's Generator version, or file order",
                "contiguous": ranges(hp) is not None,
                "position_range": list(ranges(hp)) if ranges(hp) else None,
                "positions_sha256": __import__("hashlib").sha256(
                    (",".join(str(x) for x in hp)).encode()).hexdigest(),
                "first_5_positions": hp[:5], "last_5_positions": hp[-5:],
                "test_split_left_untouched": True}
            print("%-9s %-11s q=%-7d %s  <= EVAL (carved from train)"
                  % (ds, "train_holdout", len(hold),
                     "positions sha256 %s" % dsres["splits"]["train_holdout"]
                     ["positions_sha256"][:16]), flush=True)

        # does the retrieval cache cover these rows?
        rc = os.path.join(FC, ds, "retrieval_cache")
        cov = {}
        for model in ("dense", "splade"):
            f = os.path.join(rc, "%s_top1000.npz" % model)
            if os.path.isfile(f):
                z = np.load(f)
                cov[model] = {"rows": int(z["ids"].shape[0]), "K": int(z["ids"].shape[1]),
                              "covers_all_queries": int(z["ids"].shape[0]) == len(qids)}
                if not cov[model]["covers_all_queries"]:
                    fails.append("%s/%s: cache has %d rows, %d canonical queries"
                                 % (ds, model, z["ids"].shape[0], len(qids)))
            else:
                cov[model] = {"present": False}
        dsres["retrieval_cache"] = cov
        res[ds] = dsres
        print("")

    cc = corpus_ceiling_column()
    for ds_ in res:
        res[ds_]["corpus_ceiling"] = cc["datasets"].get(ds_)
    col_sha = cc["datasets"]["webqsp"]["splits"].get("train_holdout", {}).get("carve_ids_sha256")
    res["webqsp"]["splits"]["train_holdout"]["carve_ids_sha256"] = main_carve_sha
    res["webqsp"]["splits"]["train_holdout"]["carve_matches_corpus_ceiling_column"] = (
        col_sha == main_carve_sha)
    if col_sha != main_carve_sha:
        fails.append("webqsp: train_holdout carve differs between main() and the corpus-ceiling column")
    # revision chain: the previous bytes of this record, so a citation of the earlier hash
    # remains traceable after regeneration
    history = []
    if os.path.isfile(OUT):
        import hashlib
        prior_bytes = open(OUT, "rb").read()
        try:
            history = list(json.loads(prior_bytes).get("revision_history") or [])
        except Exception:
            history = []
        history.append({"sha256": hashlib.sha256(prior_bytes).hexdigest(),
                        "superseded_utc": __import__("datetime").datetime.now(
                            __import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "why": REVISION_NOTE})
    rec = {
        "RECORD": "K_SEMANTICS_AND_EVAL_SPLITS",
        "revision_history": history,
        "_what": "declares K semantics and the per-dataset evaluation split BEFORE any ceiling "
                 "number is computed. DECLARATION plus MEASUREMENT -- no artifact modified.",
        "supersedes_nothing": "this is additive; it pins no existing hash and edits no contract",
        "K_SEMANTICS": {
            "cache_depth": 1000,
            "shallower_K_is_a_prefix": True,
            "corpus_ceiling": dict(cc, **{
                "rule": "1.0 by construction for the five text corpora (stated once, not "
                        "columned); a COLUMN for webqsp, per split, reference-level and "
                        "query-level; never average webqsp with the five without conditioning "
                        "on it",
                "first_revision_error": "stated 1.0 for all six on the basis that GOLD_FIELD_CHECK "
                                        "resolved every ref -- but that check counts refs already "
                                        "in the query files, after the RoG bridge, and cannot see "
                                        "the upstream refs that did not survive it; its quoted "
                                        "total 3,850,471 was also hardcoded and does not "
                                        "reproduce (the per-split sum is %d)"
                                        % cc["total_gold_refs_resolved_in_files"]}),
            "pool_ceiling": {
                "definition": "recall of gold within the cached top-K for a given model",
                "name_to_use": "pool_ceiling@K",
                "names_forbidden": ["candidate_ceiling", "corpus_ceiling"],
                "why": "a ceiling measured over a top-1000 cache is bounded by that cache; "
                       "calling it a corpus ceiling silently subtracts the retriever's misses, "
                       "which are the quantity under study"},
            "R1_report_with": ["K", "model", "pool_provenance", "dataset", "split"],
            "R2_naming": "cache-derived ceilings are pool_ceiling@K",
            "R3_corpus_ceiling": "1.0 by construction for metaqa/hotpotqa/2wiki/musique/squad, "
                                 "stated here; a per-split COLUMN for webqsp (see corpus_ceiling)",
            "R4_binding_is_measured": "report recall@{1,5,10,20,100,1000} beside every pool "
                                      "ceiling so the reader sees whether depth binds",
            "R5_K_was_never_selected": "one depth materialised, all shallower K are prefixes",
            "R6_membership_by_position": "gold at canonical position p is in the pool for q iff "
                                         "p in ids[q]; never test membership over distinct "
                                         "source rows -- the pointer index is many-to-one",
        },
        "EVAL_SPLITS": {ds: {"split": EVAL[ds], "reason": EVAL_REASON[ds]} for ds in DATASETS},
        "webqsp_holdout_stride": WEBQSP_HOLDOUT_STRIDE,
        "split_order_declared_by_pointer_index": SPLITS,
        "split_order_verified_from": "<ds>/queries/pointer_index/query_ids.json, the same file "
                                     "RESOLUTION_TEXT_GATE used",
        "results": res, "notes": notes, "failures": fails,
    }
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)

    print("=" * 96)
    w = cc["datasets"]["webqsp"]["pre_bridge"]
    print("CORPUS CEILING: 1.0 by construction for the five text corpora; a COLUMN for webqsp -- "
          "%d of %d upstream gold refs in the corpus (%.4f), %d zero-gold queries"
          % (w["gold_refs_in_corpus"], w["upstream_gold_refs"],
             w["reference_level_corpus_ceiling_overall"], w["queries_zero_gold"]))
    print("EVAL SPLIT: " + "  ".join("%s=%s" % (d, EVAL[d]) for d in DATASETS))
    print("EVERY SPLIT CONTIGUOUS IN CANONICAL QUERY POSITION: %s"
          % ("YES" if not notes else "NO"))
    for x in notes:
        print("   %s" % x)
    print("FAILURES: %s" % ("NONE" if not fails else len(fails)))
    for x in fails:
        print("   %s" % x)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
