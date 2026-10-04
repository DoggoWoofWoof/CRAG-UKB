# -*- coding: utf-8 -*-
"""Same self-consistency test as the frozen-kNN audit, on a graph2 family, by sampling.

Whole-corpus gathers do not fit 16.8 GB of RAM for the heavy datasets, and they do not need to:
if the search had run over the wrong vectors, every edge would be wrong, not a rare one -- the
frozen musique kNN failed this at 1.7% of edges with a p99 of 0.127. A random sample of edges
either sits at the fp16 floor or it does not.
"""
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, "data/final_canonical")
from pointer_resolver import CanonicalEmbeddings          # noqa: E402

NS = 200000
CHUNK = 25000
out_p = "data/_family_v1/NEW_KNN_WEIGHT_AUDIT.json"
out = json.load(io.open(out_p, encoding="utf-8")) if os.path.exists(out_p) else {}
for ds in sys.argv[1:]:
    z = np.load("data/final_canonical/%s/graph2/knn.npz" % ds)
    src, dst, w = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["weight"]
    rng = np.random.default_rng(20260909)
    pick = (np.arange(src.size) if src.size <= NS
            else np.sort(rng.choice(src.size, NS, replace=False)))
    s, d, wv = src[pick], dst[pick], w[pick].astype(np.float64)
    # gather in chunks: 200K sampled edges of a 12M-edge graph touch ~380K distinct rows, and
    # one gather of those is 1.1 GiB fp16 before the resolver's own reorder copy and the fp32
    # cast -- more than this 16.8 GB machine has spare.
    E = CanonicalEmbeddings(ds, "dense")
    errs, gathered = [], 0
    for c0 in range(0, s.size, CHUNK):
        cs, cd = s[c0:c0 + CHUNK], d[c0:c0 + CHUNK]
        need = np.unique(np.concatenate([cs, cd]))
        gathered += need.size
        X = E.gather(need).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        ix = {int(v): i for i, v in enumerate(need)}
        si = np.fromiter((ix[int(v)] for v in cs), np.int64, cs.size)
        di = np.fromiter((ix[int(v)] for v in cd), np.int64, cd.size)
        errs.append(np.abs(np.einsum("ij,ij->i", X[si], X[di]).astype(np.float64)
                           - wv[c0:c0 + CHUNK]))
        del X
    e = np.concatenate(errs)
    r = {"n_edges": int(src.size), "n_sampled": int(pick.size),
         "n_endpoint_rows_gathered": int(gathered),
         "max_abs_err": float(e.max()), "p99_abs_err": float(np.percentile(e, 99)),
         "n_gt_1e-3": int((e > 1e-3).sum()), "n_gt_1e-2": int((e > 1e-2).sum()),
         "weight_min": float(w.min()), "weight_max": float(w.max()),
         "n_zero_weight": int((w == 0).sum())}
    r["VERDICT"] = ("CONSISTENT_WITH_SUBSTRATE" if r["n_gt_1e-2"] == 0
                    else "STALE: %d of %d sampled edges do not reproduce"
                         % (r["n_gt_1e-2"], r["n_sampled"]))
    out[ds] = r
    print("%-9s n=%-9d sampled=%-7d max %.3e  p99 %.3e  >1e-2 %-5d  %s"
          % (ds, r["n_edges"], r["n_sampled"], r["max_abs_err"], r["p99_abs_err"],
             r["n_gt_1e-2"], r["VERDICT"]), flush=True)
    json.dump(out, io.open(out_p, "w", encoding="utf-8"), indent=1)
