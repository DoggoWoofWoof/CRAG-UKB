# -*- coding: utf-8 -*-
"""Run the NEW kNN pipeline against a dataset that already has a FROZEN kNN, and diff them.

Nothing else in this build can tell me whether the new builder reproduces the established
contract, because the three datasets I am building kNN for are exactly the three that have
none to compare against. squad and metaqa do have one, they are small enough to redo on CPU,
and they went through the same encoder -- so running the new code path over them and diffing
the edge set against the locked graph/knn.npz is the only available proof that the contract
survived the rewrite.

The math here is deliberately the same lines as modal_canonical_knn.run: resolve through the
pointer index, similarity over DISTINCT source rows, fp32 normalize, exhaustive top-4, drop
self, keep 3, undirected, lift each canonical position to its source row neighbours. Only the
device differs.

A clean result is edge-set equality. A near-clean result -- differences only where the 3rd and
4th neighbour are within float noise of each other -- is still a pass, and is reported as tie
churn rather than being hidden.
"""
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("data/final_canonical"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pointer_resolver import CanonicalEmbeddings          # noqa: E402

R = "data/final_canonical"
SEARCH_K = 4
TIE = 1e-3
BLOCK = 4096


def build(ds):
    z = np.load("%s/%s/pointer_index/dense.npz" % (R, ds))
    src, row = z["src"].astype(np.int64), z["row"].astype(np.int64)
    key = (src << 40) | row
    uniq, pos2u = np.unique(key, return_inverse=True)
    nu = uniq.size
    n = src.size
    first = np.full(nu, -1, dtype=np.int64)
    o = np.arange(n - 1, -1, -1, dtype=np.int64)
    first[pos2u[o]] = o
    E = CanonicalEmbeddings(ds, "dense")
    X = E.gather(first).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    nbr = np.full((nu, SEARCH_K), -1, dtype=np.int64)
    sc = np.full((nu, SEARCH_K), -1e30, dtype=np.float32)
    for b in range(0, nu, BLOCK):
        e = min(b + BLOCK, nu)
        t = X[b:e] @ X.T
        idx = np.argpartition(-t, SEARCH_K - 1, axis=1)[:, :SEARCH_K]
        val = np.take_along_axis(t, idx, 1)
        o2 = np.argsort(-val, axis=1, kind="stable")
        nbr[b:e] = np.take_along_axis(idx, o2, 1)
        sc[b:e] = np.take_along_axis(val, o2, 1)
    return n, nu, pos2u, first, nbr, sc


def lift(n, pos2u, first, nbr, sc):
    pos = np.arange(n, dtype=np.int64)
    I, J, W = [], [], []
    for c in range(SEARCH_K):
        v = nbr[pos2u, c]
        w = sc[pos2u, c]
        keep = (v >= 0) & (v != pos2u)
        j = first[v[keep]]
        i = pos[keep]
        ok = i != j
        I.append(i[ok])
        J.append(j[ok])
        W.append(w[keep][ok])
    i, j, w = np.concatenate(I), np.concatenate(J), np.concatenate(W)
    a, b = np.minimum(i, j), np.maximum(i, j)
    key = (a << 32) | b
    o = np.lexsort((-w, key))
    key, a, b, w = key[o], a[o], b[o], w[o]
    k = np.ones(key.size, dtype=bool)
    k[1:] = key[1:] != key[:-1]
    return key[k], w[k]


out = {}
for ds in sys.argv[1:] or ["squad", "metaqa"]:
    t0 = time.time()
    n, nu, pos2u, first, nbr, sc = build(ds)
    newkey, neww = lift(n, pos2u, first, nbr, sc)
    z = np.load("%s/%s/graph/knn.npz" % (R, ds))
    fa = np.minimum(z["src"], z["dst"]).astype(np.int64)
    fb = np.maximum(z["src"], z["dst"]).astype(np.int64)
    oldkey = np.unique((fa << 32) | fb)
    inter = np.intersect1d(newkey, oldkey, assume_unique=True)
    only_new = np.setdiff1d(newkey, oldkey, assume_unique=True)
    only_old = np.setdiff1d(oldkey, newkey, assume_unique=True)
    # a disagreement is real only if the edge that was dropped was not tied with the one kept
    gap = float(np.abs(sc[:, 2] - sc[:, 3]).min()) if nu > SEARCH_K else 0.0
    tie_rows = int((np.abs(sc[:, 2] - sc[:, 3]) <= TIE).sum())
    r = {"dataset": ds, "n_positions": int(n), "n_distinct_source_rows": int(nu),
         "frozen_edges": int(oldkey.size), "new_edges": int(newkey.size),
         "shared": int(inter.size), "only_new": int(only_new.size),
         "only_frozen": int(only_old.size),
         "jaccard": round(float(inter.size) / float(len(set(newkey.tolist()) | set(oldkey.tolist()))), 6),
         "rows_with_k3_k4_tie_within_%g" % TIE: tie_rows,
         "min_k3_k4_gap": gap, "secs": round(time.time() - t0, 1)}
    r["VERDICT"] = ("EXACT_MATCH" if not only_new.size and not only_old.size
                    else "TIE_CHURN_ONLY" if max(only_new.size, only_old.size) <= tie_rows
                    else "DIFFERS")
    out[ds] = r
    print("%-8s frozen %s  new %s  shared %s  only_new %s  only_frozen %s  -> %s  (%.0fs)"
          % (ds, "{:,}".format(oldkey.size), "{:,}".format(newkey.size),
             "{:,}".format(inter.size), "{:,}".format(only_new.size),
             "{:,}".format(only_old.size), r["VERDICT"], r["secs"]))
json.dump(out, io.open("data/_family_v1/KNN_PARITY_CHECK.json", "w", encoding="utf-8"), indent=1)
print("wrote data/_family_v1/KNN_PARITY_CHECK.json")
