"""Are the three existing retrieval caches VALID, and is the webqsp repair patch wired?

Section 4 of the readiness report answered existence only, because that is what was asked.
"present" is not "correct", so this asks the two validity questions existence cannot:

A  IS THE webqsp DENSE REPAIR PATCH WIRED?
   _dense_repair_patch/ holds ten patch stores.  Nine are referenced by POINTER_INDEX.  The
   tenth, webqsp__docs (12,814 rows starting at 62,381), is referenced by NOTHING.  Either
   the patch is a benign orphan (the store was fixed in place, or those rows were never
   damaged in webqsp) or the repair was never installed -- in which case webqsp's dense docs
   still serve damaged bytes for 12,814 rows and webqsp is NOT ready.

   The damage signature is not assumed.  It is derived from a patch that IS wired
   (musique__docs): read the same rows from the raw shard bytes, which the pointer index
   redirects AWAY from, and see what is actually wrong with them.  Then apply that same test
   to webqsp's live rows.

B  ARE THE CACHES CONSISTENT WITH THE SUBSTRATE THEY INDEX?
   mtimes put every cache after every repair patch, but mtime is not evidence -- it survives a
   copy and says nothing about content.  So recompute: sample queries, score them against the
   full doc set through the authoritative resolver, take an exact top-1000, and compare ids and
   scores against the cache.  A cache built before the repair, against a different
   textualization, or with different pooling would disagree here.

   Comparison is by SET OVERLAP at several depths plus a score delta, not by positionwise id
   equality: scores are stored fp16, so ties and near-ties permute freely and a positionwise
   test would report a correct cache as broken.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/cache_and_repair_check.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "CACHE_AND_REPAIR_CHECK.json")
sys.path.insert(0, FC)
from pointer_resolver import CanonicalEmbeddings  # noqa: E402

NQ = 32          # sampled queries per (dataset, model)
TOPK = 1000
CACHED = ["metaqa", "musique", "squad"]


def rj(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def raw_rows(store_dir, rows, shard_size):
    """Read rows from the RAW shard bytes, deliberately bypassing the pointer redirect."""
    out = np.zeros((len(rows), 1536), dtype=np.float32)
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r // shard_size, []).append((i, r % shard_size))
    names = sorted(f for f in os.listdir(store_dir)
                   if f.startswith("shard_") and f.endswith(".npy"))
    for sh, items in by.items():
        cand = [f for f in names if ("%05d" % sh) in f or ("%04d" % sh) in f]
        if not cand:
            return None
        a = np.load(os.path.join(store_dir, cand[0]), mmap_mode="r")
        for i, off in items:
            if off < a.shape[0]:
                out[i] = np.asarray(a[off], dtype=np.float32)
    return out


def describe(v, label):
    n = np.linalg.norm(v, axis=1)
    return {"label": label, "rows": int(v.shape[0]),
            "nonfinite_rows": int((~np.isfinite(v)).any(axis=1).sum()),
            "all_zero_rows": int((np.abs(v) < 1e-12).all(axis=1).sum()),
            "norm_min": round(float(n.min()), 6), "norm_max": round(float(n.max()), 6),
            "norm_mean": round(float(n.mean()), 6),
            "unit_norm_rows": int((np.abs(n - 1.0) < 2e-2).sum())}


def part_a(pi):
    print("=" * 96)
    print("A  IS THE webqsp DENSE REPAIR PATCH WIRED?")
    print("=" * 96)
    res = {}
    blob = json.dumps(pi)

    def phase_c(ds):
        node = ((pi.get("datasets") or {}).get(ds) or {}).get("dense") or {}
        for s in node.get("stores") or []:
            if s.get("store") == "PHASE_C" and s.get("path"):
                return s["path"].replace("\\", "/"), s.get("shard_size")
        return None, None

    for ds, patch in (("musique", "musique__docs"), ("webqsp", "webqsp__docs")):
        pd = os.path.join(FC, "_dense_repair_patch", patch)
        rows = rj(os.path.join(pd, "dense_rows.json"))
        good = np.load(os.path.join(pd, "dense.npy"), mmap_mode="r")
        sub = list(rows[:2048])
        gi = np.asarray(good[:len(sub)], dtype=np.float32)

        sdir, ss = phase_c(ds)
        raw = raw_rows(os.path.join(ROOT, sdir), sub, ss) if sdir and ss else None

        E = CanonicalEmbeddings(ds, "dense", kind="docs", root=FC)
        live = np.asarray(E.gather(sub), dtype=np.float32)

        wired = patch in blob
        d_raw = describe(raw, "raw PHASE_C bytes") if raw is not None else None
        d_live = describe(live, "live resolved")
        d_good = describe(gi, "patch bytes")
        eq_patch = float(np.abs(live - gi).max()) if live.shape == gi.shape else None
        eq_raw = float(np.abs(live - raw).max()) if raw is not None else None

        print("")
        print("%s   patch=%s  patch_rows=%d  sampled=%d  registered_in_POINTER_INDEX=%s"
              % (ds, patch, len(rows), len(sub), wired))
        print("   store=%s  shard_size=%s" % (sdir, ss))
        for d in (d_raw, d_good, d_live):
            if d:
                print("   %-20s nonfinite=%-6d all_zero=%-6d norm[min/mean/max]="
                      "%.4f/%.4f/%.4f  unit=%d/%d"
                      % (d["label"], d["nonfinite_rows"], d["all_zero_rows"], d["norm_min"],
                         d["norm_mean"], d["norm_max"], d["unit_norm_rows"], d["rows"]))
        print("   live vs patch  max abs delta = %s"
              % ("%.6f" % eq_patch if eq_patch is not None else "n/a"))
        print("   live vs raw    max abs delta = %s"
              % ("%.6f" % eq_raw if eq_raw is not None else "n/a"))

        clean = (d_live["all_zero_rows"] == 0 and d_live["nonfinite_rows"] == 0
                 and d_live["unit_norm_rows"] == d_live["rows"])
        verdict = "LIVE_ROWS_CLEAN" if clean else "LIVE_ROWS_DAMAGED"
        res[ds] = {"patch": patch, "patch_rows": len(rows), "sampled": len(sub),
                   "store": sdir, "shard_size": ss,
                   "registered_in_pointer_index": wired,
                   "raw_phase_c": d_raw, "patch_bytes": d_good, "live_resolved": d_live,
                   "max_abs_delta_live_vs_patch": eq_patch,
                   "max_abs_delta_live_vs_raw": eq_raw,
                   "verdict": verdict}
        print("   VERDICT: %s" % verdict)
    return res


def topk_dense(ds, qrows):
    D = CanonicalEmbeddings(ds, "dense", kind="docs", root=FC)
    Q = CanonicalEmbeddings(ds, "dense", kind="queries", root=FC)
    nd = len(D)
    X = np.asarray(D.gather(list(range(nd))), dtype=np.float32)
    q = np.asarray(Q.gather(list(qrows)), dtype=np.float32)
    S = q @ X.T
    return _rank(S, nd)


def topk_splade(ds, qrows):
    import scipy.sparse as sp
    D = CanonicalEmbeddings(ds, "splade", kind="docs", root=FC)
    Q = CanonicalEmbeddings(ds, "splade", kind="queries", root=FC)
    nd = len(D)
    Xd = D.gather(list(range(nd)))
    Xq = Q.gather(list(qrows))
    Xd = Xd if sp.issparse(Xd) else sp.csr_matrix(Xd)
    Xq = Xq if sp.issparse(Xq) else sp.csr_matrix(Xq)
    S = np.asarray((Xq @ Xd.T).todense(), dtype=np.float32)
    return _rank(S, nd)


def _rank(S, nd):
    k = min(TOPK, nd)
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    srt = np.take_along_axis(S, idx, 1)
    o = np.argsort(-srt, axis=1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(srt, o, 1), nd


def part_b():
    print("")
    print("=" * 96)
    print("B  ARE THE CACHES CONSISTENT WITH THE SUBSTRATE?  (recomputed, not trusted)")
    print("=" * 96)
    print("%-9s %-7s %5s %6s %7s %8s %8s %8s %11s %11s"
          % ("dataset", "model", "nq", "docs", "top1", "ovl@10", "ovl@100", "ovl@1k",
             "maxdScore", "verdict"))
    res = {}
    for ds in CACHED:
        for mdl in ("dense", "splade"):
            cp = os.path.join(FC, ds, "retrieval_cache", "%s_top1000.npz" % mdl)
            z = np.load(cp)
            cid, csc = z["ids"], z["scores"]
            nq = cid.shape[0]
            rng = np.random.RandomState(0)
            qs = sorted(rng.choice(nq, size=min(NQ, nq), replace=False).tolist())
            try:
                gid, gsc, nd = (topk_dense if mdl == "dense" else topk_splade)(ds, qs)
            except Exception as ex:
                print("%-9s %-7s   ERROR %s" % (ds, mdl, str(ex)[:70]))
                res["%s.%s" % (ds, mdl)] = {"error": "%s: %s" % (type(ex).__name__, str(ex)[:200])}
                continue
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
            tol = 0.05 if mdl == "dense" else 0.5   # splade scores run to ~75, fp16 step is coarser
            ok = ov[10] >= 0.99 and ov[1000] >= 0.99 and md < tol
            v = "CONSISTENT" if ok else "MISMATCH"
            print("%-9s %-7s %5d %6d %7.3f %8.4f %8.4f %8.4f %11.5f %11s"
                  % (ds, mdl, len(qs), nd, t1, ov[10], ov[100], ov[1000], md, v))
            res["%s.%s" % (ds, mdl)] = {
                "queries_in_cache": int(nq), "docs": int(nd), "sampled": len(qs),
                "cache_depth": int(cid.shape[1]), "recomputed_depth": int(depth),
                "top1_agreement": round(t1, 4),
                "overlap_at_10": round(ov[10], 4), "overlap_at_100": round(ov[100], 4),
                "overlap_at_1000": round(ov[1000], 4),
                "max_abs_score_delta": round(md, 6), "score_tolerance": tol,
                "verdict": v}
    return res


def main():
    pi = rj(os.path.join(FC, "POINTER_INDEX.json"))
    a = part_a(pi)
    b = part_b()
    rec = {"RECORD": "CACHE_AND_REPAIR_CHECK",
           "_what": "validity, not existence: (A) whether the unregistered webqsp dense repair "
                    "patch means webqsp serves damaged rows, and (B) whether the three built "
                    "retrieval caches reproduce from the currently-resolved substrate. "
                    "MEASUREMENT ONLY -- nothing was rebuilt or modified.",
           "method": {"sampled_queries_per_cell": NQ, "topk": TOPK,
                      "comparison": "set overlap at 10/100/1000 plus max abs score delta; "
                                    "positionwise id equality is NOT used because fp16 scores "
                                    "permute ties and would fail a correct cache",
                      "damage_signature": "derived from the wired musique__docs patch by reading "
                                          "the raw PHASE_C bytes the pointer redirects away from"},
           "A_repair_patch_wiring": a,
           "B_cache_consistency": b}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("")
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
