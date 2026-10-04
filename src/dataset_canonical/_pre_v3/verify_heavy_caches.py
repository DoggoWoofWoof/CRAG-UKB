"""Hold the three new retrieval caches to the same standard the first three were held to.

WHY THIS IS NOT cache_and_repair_check.py PART B

  That verifier does `X = np.asarray(D.gather(range(nd)), dtype=np.float32)` -- the whole
  corpus, fp32, resident.  For metaqa/musique/squad (43k / 118k / 20k docs) that is at most
  0.7 GB and fine.  For hotpotqa it is 32 GB and for 2wiki 37 GB, on a machine with 16.8 GB
  total.  So the corpus is STREAMED here in blocks, exactly as build_retrieval_cache.py itself
  streams it, and the comparison and the tolerances are otherwise identical -- same sample seed,
  same set-overlap-at-10/100/1000, same top-1 agreement, same max-abs-score delta, same
  0.05 / 0.5 tolerances.

WHY SET OVERLAP AND NOT POSITIONWISE ID EQUALITY

  Scores are stored as float16, so many near-neighbours land on the same stored value and their
  ORDER is not recoverable from the file.  Two runs that agree on the SET can disagree on the
  order inside a tie group without either being wrong.  Insisting on positionwise equality would
  therefore fail a correct artifact.  This was already the standard applied to the first three.

WHAT THIS TEST IS ACTUALLY FOR, GIVEN THE CACHE WAS BUILT ON A GPU

  The two heavy caches were produced by a float32/TF32 GEMM on an H100 and the first three by
  numpy on this CPU.  Operands are identical (every stored value is an fp16 value, exactly
  representable in both), so the only difference is summation order over dim=1536.  This
  measures whether that difference reaches the reported artifact.  It is a MEASUREMENT of the
  divergence, not an assumption that there is none: whatever it is, it is printed.

  webqsp was built locally by the same builder as the first three, so for webqsp this is a
  plain reproduction check.

THREE WHOLE-ARTIFACT INVARIANTS, CHECKED OVER EVERY ROW AND NOT JUST THE SAMPLE

  A 32-query sample can miss a systematic defect, so these run over the full cache:
    ids in range      every id in [0, n_docs); a negative id means a padded key survived the
                      merge, which is exactly how a short final block would fail
    no repeats        no id twice in one row; the merge is a concatenate-and-retake, so a
                      duplicate would mean a block was counted twice
    scores sorted     each row non-increasing; the key packs score above position, so a
                      descending key order must give a non-increasing score order

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/verify_heavy_caches.py [dataset ...]
"""

import io
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "HEAVY_CACHE_VERIFICATION.json")
sys.path.insert(0, FC)
sys.path.insert(0, os.path.join(ROOT, "scratchpad"))
from pointer_resolver import CanonicalEmbeddings  # noqa: E402
from build_retrieval_cache import order_key, take_top, unpack  # noqa: E402

DATASETS = ["webqsp", "hotpotqa", "2wiki"]
NQ = 32
TOPK = 1000
DOC_BLOCK = 131072          # 0.8 GB fp32 per block; the machine has ~5 GB free
TOL = {"dense": 0.05, "splade": 0.5}


def whole_artifact(cid, csc, nd):
    """Invariants over every row of the cache, not the sample."""
    bad_lo = int((cid < 0).sum())
    bad_hi = int((cid >= nd).sum())
    dup_rows = 0
    for a in range(0, cid.shape[0], 4096):
        blk = cid[a:a + 4096]
        s = np.sort(blk, axis=1)
        dup_rows += int((s[:, 1:] == s[:, :-1]).any(axis=1).sum())
    unsorted_rows = 0
    for a in range(0, csc.shape[0], 4096):
        blk = csc[a:a + 4096].astype(np.float32)
        unsorted_rows += int((blk[:, 1:] > blk[:, :-1]).any(axis=1).sum())
    return {"ids_below_zero": bad_lo, "ids_at_or_above_n_docs": bad_hi,
            "rows_with_duplicate_ids": dup_rows,
            "rows_with_nonmonotone_scores": unsorted_rows,
            "PASS": not (bad_lo or bad_hi or dup_rows or unsorted_rows)}


def _eval_span(ds):
    """The declared evaluation split's [lo, hi) query positions, or None.

    Read from K_SEMANTICS_AND_EVAL_SPLITS.json rather than restated here, because that record
    is where the per-dataset choice of split is DECLARED and a second copy of the boundaries
    could drift from it. Returns None when the split is not a contiguous position range --
    webqsp's `train_holdout` is a lexicographic stride, so it has no range -- and the eval
    probe is then skipped rather than guessed.

    This matters because the eval split is the pool a candidate ceiling is measured over. A
    cache can be assembled correctly and still be the wrong input if the rows the ceiling
    actually reads were never checked against the corpus.
    """
    p = os.path.join(FC, "K_SEMANTICS_AND_EVAL_SPLITS.json")
    if not os.path.isfile(p):
        return None
    try:
        d = json.load(io.open(p, encoding="utf-8"))
        r = d["results"][ds]
        rng = r["splits"][r["eval_split"]].get("position_range")
    except Exception:
        return None
    if not rng or len(rng) != 2:
        return None
    return int(rng[0]), int(rng[1])


def _merged_part_spans(ds, model, nq):
    """Per-part [lo, hi) query spans and backends for a cache assembled from query shards.

    WHY THIS EXISTS

      A seeded 32-query sample over 192,606 rows is a fine test of the ARITHMETIC but a weak
      test of the ASSEMBLY. 2wiki/splade is merged from 24 query shards, 13 of them computed on
      a Modal workspace that was later disabled and 11 locally; at 32 samples across 24 parts
      most parts draw zero exactly-recomputed rows. A whole part that is right-sized and wrong
      is not a hypothetical failure mode here -- it is the one already observed in this project,
      where 21 of 281 remote dense shards were the correct size and the wrong content, because
      `volume ls` under-lists and `volume put` skips on name.

      So every part that contributed rows gets at least one row recomputed from the corpus.

    The merge meta stores `backend_by_part` as {qshard: backend} and `merged_from` as the shard
    count, with no spans, so the span is reconstructed with the merge's own exact formula
    (lo = nq*s//N) rather than read from a second copy of the numbers that could disagree.
    Returns [] when the cache carries no part manifest -- the metas written before that field
    existed, hotpotqa's among them -- and the sample is then left exactly as it was.
    """
    p = os.path.join(FC, ds, "retrieval_cache", "%s_top%d.meta.json" % (model, TOPK))
    if not os.path.isfile(p):
        return []
    try:
        m = json.load(io.open(p, encoding="utf-8"))
    except Exception as e:
        print("   (part manifest unreadable: %s -- sample left unstratified)" % e)
        return []
    bb, n = m.get("backend_by_part"), m.get("merged_from")
    if not isinstance(bb, dict) or not n:
        return []
    n = int(n)
    out = []
    for key, be in bb.items():
        s = int(key)
        lo, hi = nq * s // n, nq * (s + 1) // n
        if hi > lo:
            out.append((s, lo, hi, str(be)))
    return sorted(out)


def recompute(ds, model, qs):
    """Exact top-1000 for the sampled queries over the FULL corpus, streamed."""
    D = CanonicalEmbeddings(ds, model, kind="docs", root=FC)
    Q = CanonicalEmbeddings(ds, model, kind="queries", root=FC)
    nd = len(D)
    k = min(TOPK, nd)
    dense = (model == "dense")
    if dense:
        q = np.ascontiguousarray(Q.gather(list(qs)), dtype=np.float32)
    else:
        import scipy.sparse as sp
        q = Q.gather(list(qs))
        q = (q if sp.issparse(q) else sp.csr_matrix(q)).tocsr().astype(np.float32)

    best = None
    t0 = time.time()
    for b0 in range(0, nd, DOC_BLOCK):
        b1 = min(b0 + DOC_BLOCK, nd)
        if dense:
            dm = np.ascontiguousarray(D.gather(list(range(b0, b1))), dtype=np.float32).T
            S = q @ dm
        else:
            import scipy.sparse as sp
            dm = D.gather(list(range(b0, b1)))
            dm = (dm if sp.issparse(dm) else sp.csr_matrix(dm)).tocsr().astype(np.float32)
            S = np.asarray((q @ dm.T).todense(), dtype=np.float32)
        del dm
        # rank on the BUILDER'S key, not on the bare score.  argpartition orders equal
        # scores arbitrarily while the cache orders them by ascending canonical position, so
        # a bare-score recomputation is a DIFFERENT function and disagrees with a correct
        # cache wherever a tie group straddles the cut.
        pos = np.broadcast_to(np.arange(b0, b1, dtype=np.int64), S.shape)
        kk = take_top(order_key(S, pos), min(k, b1 - b0))
        del S, pos
        best = kk if best is None else take_top(np.concatenate([best, kk], axis=1), k)
        del kk
        if (b0 // DOC_BLOCK) % 8 == 0 or b1 == nd:
            print("      docs %s/%s  %.0fs" % (format(b1, ","), format(nd, ","),
                                               time.time() - t0), flush=True)
    sc, ids = unpack(best)
    return ids.astype(np.int64), sc.astype(np.float32), nd


def main():
    argv = sys.argv[1:]
    models = tuple(m for m in ("dense", "splade") if m in argv) or ("dense", "splade")
    want = [a for a in argv if a not in ("dense", "splade")]

    # MERGE, don't overwrite.  Each cell here costs a full streamed pass over a multi-million
    # document corpus, so re-verifying one new cell must not delete the rows already
    # established for the others.  Prior cells are carried forward and labelled with the run
    # that produced them, so the record always says which cells this invocation actually
    # measured and which are inherited.
    res, fails = {}, []
    if os.path.isfile(OUT):
        try:
            prior = json.load(io.open(OUT, encoding="utf-8"))
            res = prior.get("results", {}) or {}
            for v in res.values():
                if isinstance(v, dict):
                    v.setdefault("measured_in_run", "earlier")
        except Exception as e:
            print("   (could not read prior %s: %s -- starting fresh)" % (OUT, e))
            res = {}
    this_run = []
    print("%-9s %-7s %5s %9s %7s %8s %8s %8s %8s %8s %9s %5s %11s"
          % ("dataset", "model", "nq", "docs", "top1", "ovl@10", "ovl@100", "ovl@1k",
             "cont@10", "exactPos", "maxdScr", "inv", "verdict"))
    for ds in [d for d in DATASETS if not want or d in want]:
        for model in models:
            cp = os.path.join(FC, ds, "retrieval_cache", "%s_top1000.npz" % model)
            if not os.path.isfile(cp):
                print("%-9s %-7s   (cache absent)" % (ds, model))
                res["%s.%s" % (ds, model)] = {"present": False}
                continue
            z = np.load(cp)
            cid, csc = z["ids"], z["scores"]
            nq = cid.shape[0]
            rng = np.random.RandomState(0)
            qs = sorted(rng.choice(nq, size=min(NQ, nq), replace=False).tolist())
            # The seeded 32 are kept verbatim so this cell stays comparable with the cells
            # already in the record. Per-part probes are ADDED to them, never substituted.
            spans = _merged_part_spans(ds, model, nq)
            ev = _eval_span(ds)
            probe = []
            base = set(qs)
            for s, lo, hi, be in spans:
                # One row per part establishes the ASSEMBLY. A second row inside the declared
                # eval span, where the part reaches it, establishes the rows a ceiling will
                # actually read -- the boundary parts' midpoints fall outside that span.
                rows = [("part_midpoint", lo + (hi - lo) // 2)]
                if ev and lo < ev[1] and hi > ev[0]:
                    a, b = max(lo, ev[0]), min(hi, ev[1])
                    rows.append(("eval_span_midpoint", a + (b - a) // 2))
                for why, r in rows:
                    probe.append({"qshard": s, "lo": lo, "hi": hi, "row": r,
                                  "chosen": why, "backend": be})
                    if r not in base:
                        qs.append(r)
                        base.add(r)
            qs = sorted(qs)
            if probe:
                print("   %s/%s  %d probe rows across %d parts (part midpoints, plus one "
                      "inside the eval span where a part reaches it)"
                      % (ds, model, len(probe),
                         len(set(e["qshard"] for e in probe))), flush=True)
            print("   %s/%s  recomputing %d queries over the full corpus ..."
                  % (ds, model, len(qs)), flush=True)
            gid, gsc, nd = recompute(ds, model, qs)
            inv = whole_artifact(cid, csc, nd)

            depth = gid.shape[1]
            t1 = float(np.mean([1.0 if gid[i, 0] == cid[q, 0] else 0.0
                                for i, q in enumerate(qs)]))
            ov = {}
            for d in (10, 100, 1000):
                dd = min(d, depth)
                ov[d] = float(np.mean([
                    len(set(gid[i, :dd].tolist()) & set(cid[q, :dd].tolist())) / float(dd)
                    for i, q in enumerate(qs)]))
            md = float(max(np.abs(gsc[i, :depth].astype(np.float32)
                                  - csc[q, :depth].astype(np.float32)).max()
                           for i, q in enumerate(qs)))
            # CONTAINMENT separates the two things a depth-10 set difference can mean.
            # If every one of the recomputed top-10 is somewhere in the cached top-1000, no
            # document is missing from the pool -- only its rank moved across the depth-10 cut,
            # which is what a tie group straddling that cut does and is not a defect.  A
            # document absent from the whole cached row IS a defect.
            cn = {}
            for d in (10, 100):
                dd = min(d, depth)
                cn[d] = float(np.mean([
                    len(set(gid[i, :dd].tolist()) & set(cid[q].tolist())) / float(dd)
                    for i, q in enumerate(qs)]))
            exact = float(np.mean([float((gid[i, :depth] == cid[q, :depth]).mean())
                                   for i, q in enumerate(qs)]))
            tol = TOL[model]
            # Per-part verdicts. A part whose probe row disagrees is reported by SHARD NUMBER
            # and by backend, so a systematic remote-build failure is localised rather than
            # averaged away into the whole-cache overlap figure.
            qi = {q: i for i, q in enumerate(qs)}
            bad_parts = []
            for e in probe:
                i = qi.get(e["row"])
                if i is None:
                    continue
                g, c = gid[i, :depth], cid[e["row"], :depth]
                e["overlap_at_1000"] = round(len(set(g.tolist()) & set(c.tolist()))
                                             / float(depth), 6)
                e["positionwise_exact"] = bool((g == c).all())
                e["top1_match"] = bool(g[0] == c[0])
                e["max_abs_score_delta"] = round(float(np.abs(
                    gsc[i, :depth].astype(np.float32)
                    - csc[e["row"], :depth].astype(np.float32)).max()), 6)
                if not (e["top1_match"] and e["overlap_at_1000"] >= 0.99
                        and e["max_abs_score_delta"] < tol):
                    bad_parts.append(e["qshard"])
            bad_parts = sorted(set(bad_parts))
            ok = (t1 == 1.0 and ov[1000] >= 0.99 and cn[10] == 1.0 and cn[100] == 1.0
                  and md < tol and inv["PASS"] and not bad_parts)
            if probe:
                print("   %s/%s  parts probed %d, parts DISAGREEING %s"
                      % (ds, model, len(probe),
                         (",".join(str(x) for x in bad_parts) if bad_parts else "none")),
                      flush=True)
            if bad_parts:
                fails.append("%s/%s: parts disagree: %s"
                             % (ds, model, ",".join(str(x) for x in bad_parts)))
            v = "CONSISTENT" if ok else "MISMATCH"
            if not ok:
                fails.append("%s/%s: %s" % (ds, model, v))
            print("%-9s %-7s %5d %9d %7.3f %8.4f %8.4f %8.4f %8.4f %8.4f %9.5f %5s %11s"
                  % (ds, model, len(qs), nd, t1, ov[10], ov[100], ov[1000],
                     cn[10], exact, md, "PASS" if inv["PASS"] else "FAIL", v), flush=True)
            res["%s.%s" % (ds, model)] = {
                "queries_in_cache": int(nq), "docs": int(nd), "sampled": len(qs),
                "cache_depth": int(cid.shape[1]), "recomputed_depth": int(depth),
                "top1_agreement": round(t1, 4),
                "overlap_at_10": round(ov[10], 4), "overlap_at_100": round(ov[100], 4),
                "overlap_at_1000": round(ov[1000], 4),
                "containment_at_10_in_cached_1000": round(cn[10], 6),
                "containment_at_100_in_cached_1000": round(cn[100], 6),
                "positionwise_exact_id_fraction": round(exact, 6),
                "max_abs_score_delta": round(md, 6), "score_tolerance": tol,
                "whole_artifact_invariants": inv,
                # probe ROWS, not parts: each part contributes its midpoint, and a part that
                # reaches the declared eval span contributes a second row inside it.
                "probe_rows": len(probe),
                "parts_covered": len(set(e["qshard"] for e in probe)),
                "eval_span_probed": _eval_span(ds),
                "parts_disagreeing": sorted(bad_parts),
                "per_part_probe": probe,
                "per_part_probe_note": ("one deterministically chosen row per merged part, "
                                        "recomputed from the corpus, ADDED to the seeded "
                                        "32-query sample; [] when the cache carries no part "
                                        "manifest"),
                "verdict": v,
                "measured_in_run": "this"}
            this_run.append("%s.%s" % (ds, model))
            with io.open(OUT, "w", encoding="utf-8") as f:
                json.dump({"RECORD": "HEAVY_CACHE_VERIFICATION", "results": res,
                           "failures": fails, "complete": False}, f, indent=1)

    # verdict over EVERY cell in the record, inherited ones included, not just this run's
    allfail = sorted("%s: %s" % (k, v.get("verdict"))
                     for k, v in res.items()
                     if isinstance(v, dict) and v.get("verdict") not in (None, "CONSISTENT"))
    absent = sorted(k for k, v in res.items()
                    if isinstance(v, dict) and v.get("present") is False)
    cells = sorted(k for k, v in res.items()
                   if isinstance(v, dict) and v.get("verdict") == "CONSISTENT")
    print("")
    print("=" * 108)
    print("cells measured in THIS run : %s" % (", ".join(this_run) or "(none)"))
    print("cells CONSISTENT in record : %d  (%s)" % (len(cells), ", ".join(cells)))
    if absent:
        print("cells with no cache yet    : %s" % ", ".join(absent))
    print("EVERY CACHE IN THE RECORD CONSISTENT WITH A LOCAL RECOMPUTATION: %s"
          % ("YES" if not allfail else "NO"))
    for x in allfail:
        print("   %s" % x)
    fails = allfail
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"RECORD": "HEAVY_CACHE_VERIFICATION",
                   "_what": "recomputes the exact top-1000 for a seeded 32-query sample over "
                            "the FULL corpus, locally, and compares it to the cached artifact. "
                            "MEASUREMENT ONLY.",
                   "method": {"sample": "numpy RandomState(0).choice, same as the first three",
                              "comparison": "set overlap at 10/100/1000, top-1 agreement, max "
                                            "abs score delta; not positionwise id equality, "
                                            "because float16 storage makes tie order "
                                            "unrecoverable",
                              "corpus": "streamed in %d-document blocks; never resident" %
                                        DOC_BLOCK,
                              "tolerances": TOL},
                   "record_is_cumulative": "each cell costs a full streamed pass over its "
                                           "corpus, so cells are merged across invocations "
                                           "rather than overwritten. measured_in_run says "
                                           "'this' for cells this invocation recomputed and "
                                           "'earlier' for cells inherited from a prior one.",
                   "cells_measured_in_last_run": this_run,
                   "results": res, "failures": fails, "complete": True}, f, indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
