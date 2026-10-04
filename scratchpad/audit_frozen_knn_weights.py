# -*- coding: utf-8 -*-
"""Does each frozen kNN edge's stored weight equal the cosine of its two endpoints TODAY?

An exact-kNN edge file is self-describing: weight IS the cosine of the two canonical vectors.
So the artifact can be checked against the substrate it claims to come from without recomputing
the search. squad and metaqa are the controls -- the new pipeline reproduces both edge-for-edge,
so whatever residual they show is the floor set by fp16 storage, and anything far above that
floor on another dataset is a stale artifact, not numerical noise.
"""
import io
import json
import sys

import numpy as np

sys.path.insert(0, "data/final_canonical")
from pointer_resolver import CanonicalEmbeddings          # noqa: E402

out = {}
for ds in sys.argv[1:]:
    z = np.load("data/final_canonical/%s/graph/knn.npz" % ds)
    src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)
    w = z["weight"].astype(np.float64)
    need = np.unique(np.concatenate([src, dst]))
    X = CanonicalEmbeddings(ds, "dense").gather(need).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ix = {int(v): i for i, v in enumerate(need)}
    si = np.fromiter((ix[int(v)] for v in src), np.int64, src.size)
    di = np.fromiter((ix[int(v)] for v in dst), np.int64, dst.size)
    d = np.abs(np.einsum("ij,ij->i", X[si], X[di]).astype(np.float64) - w)
    r = {"n_edges": int(src.size), "max_abs_err": float(d.max()),
         "mean_abs_err": float(d.mean()), "p99_abs_err": float(np.percentile(d, 99)),
         "n_gt_1e-3": int((d > 1e-3).sum()), "n_gt_1e-2": int((d > 1e-2).sum()),
         "frac_gt_1e-2": float((d > 1e-2).mean()),
         "n_zero_weight": int((w == 0).sum())}
    r["VERDICT"] = ("CONSISTENT_WITH_SUBSTRATE" if r["n_gt_1e-2"] == 0
                    else "STALE: %d edges cannot be reproduced from the frozen vectors"
                         % r["n_gt_1e-2"])
    out[ds] = r
    print("%-9s n=%-9d max %.3e  p99 %.3e  >1e-2 %-6d  %s"
          % (ds, r["n_edges"], r["max_abs_err"], r["p99_abs_err"], r["n_gt_1e-2"],
             r["VERDICT"]), flush=True)
json.dump(out, io.open("data/_family_v1/FROZEN_KNN_WEIGHT_AUDIT.json", "w", encoding="utf-8"),
          indent=1)
