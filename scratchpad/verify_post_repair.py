"""
POST-REPAIR GATE
================
Proves that canonical_v1 no longer resolves any embedding to a corrupted Phase-C row.

The proof is a set operation rather than a 24 GB read, and it is exact because both sides are
known exactly: the integrity scan enumerated every damaged Phase-C row per channel, and the
pointer arrays enumerate every row canonical_v1 points at. So the gate is

    no pointer whose store is PHASE_C may name a row in that channel damaged set

plus, because a set argument only carries if the replacement is sound, a sampled numeric check
through the real resolver: dense finite and L2-normalised, SPLADE non-negative with no empty
row, and gather() order-preserving.

A residual is reported honestly rather than hidden: damaged Phase-C rows that no canonical node
points at are NOT repaired away by this, they are simply unreferenced. 2wiki has 922 such rows
and hotpotqa 16 -- exactly the rows whose canonical text was rewritten by TEXTUALIZATION_REV_2,
so canonical_v1 reads them from the REV2 patch instead. webqsp accounts for 12,814 more and is
not part of the five-dataset package.
"""
import io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "data/final_canonical")
from pointer_resolver import CanonicalEmbeddings  # noqa: E402

ROOT = "data/final_canonical"
RNG = np.random.default_rng(7)
DOC_TREE = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique",
            "hotpotqa": "hotpotqa", "squad": "squad"}


def damaged(scan, tree, kind, model):
    r = scan["BY_CHANNEL"].get("%s/%s/%s" % (tree, kind, model))
    if not r or not r.get("bad_rows"):
        return np.array([], dtype=np.int64)
    w = json.load(io.open(r["worklist"], encoding="utf-8"))
    return np.sort(np.asarray(w["bad_rows"], dtype=np.int64))


def main():
    t0 = time.time()
    scan = json.load(io.open("scratchpad/encoding_integrity.json", encoding="utf-8"))
    man = json.load(io.open("%s/POINTER_INDEX.json" % ROOT, encoding="utf-8"))
    res, ok = {}, True
    for ds, tree in DOC_TREE.items():
        for kind in ("docs", "queries"):
            for model in ("dense", "splade"):
                E = CanonicalEmbeddings(ds, model, kind, root=ROOT)
                trees = ([tree] if kind == "docs"
                         else man["queries"][ds]["trees"])
                # which store indices are PHASE_C, and which tree each one is
                stale = 0
                for i, s in enumerate(E.store_spec):
                    if s["store"] != "PHASE_C":
                        continue
                    bad = damaged(scan, s.get("tree", tree), kind, model)
                    if not bad.size:
                        continue
                    rows = E.row[E.src == i]
                    if rows.size:
                        p = np.searchsorted(bad, rows)
                        hit = (p < bad.size) & (bad[np.minimum(p, bad.size - 1)] == rows)
                        stale += int(hit.sum())
                k = min(500, len(E))
                pos = RNG.choice(len(E), size=k, replace=False)
                g = E.gather(pos)
                r = {"n": len(E), "stores": len(E.store_spec),
                     "pointers_into_damaged_rows": stale}
                if model == "dense":
                    a = np.asarray(g, dtype=np.float32)
                    nr = np.linalg.norm(a, axis=1)
                    r["finite"] = bool(np.isfinite(a).all())
                    r["l2_min"] = round(float(nr.min()), 6)
                    r["l2_max"] = round(float(nr.max()), 6)
                    r["l2_ok"] = bool(np.abs(nr - 1.0).max() < 1e-2)
                else:
                    r["splade_nonneg"] = bool((g.data >= 0).all())
                    r["empty_rows"] = int((np.diff(g.indptr) == 0).sum())
                    r["l2_ok"] = True
                sel = pos[:8]
                one = [E.get(int(i)) for i in sel]
                many = E.gather(sel)
                r["order_preserved"] = bool(all(
                    (np.array_equal(np.asarray(one[j]), np.asarray(many[j]))
                     if model == "dense" else (one[j] != many[j]).nnz == 0)
                    for j in range(len(sel))))
                r["PASS"] = bool(stale == 0 and r["l2_ok"] and r["order_preserved"]
                                 and r.get("finite", True) and r.get("empty_rows", 0) == 0
                                 and r.get("splade_nonneg", True))
                ok &= r["PASS"]
                res["%s/%s/%s" % (ds, kind, model)] = r
                print("  %-28s n=%-9d stores=%d stale=%-6d %s %s"
                      % ("%s/%s/%s" % (ds, kind, model), len(E), r["stores"], stale,
                         ("l2=[%.6f,%.6f]" % (r["l2_min"], r["l2_max"])) if model == "dense"
                         else "empty=%d" % r["empty_rows"],
                         "PASS" if r["PASS"] else "FAIL"), flush=True)
    res["ALL_PASS"] = bool(ok)
    res["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(res, io.open("scratchpad/verify_post_repair.json", "w", encoding="utf-8"), indent=1)
    print("\nALL_PASS =", ok, " %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
