# -*- coding: utf-8 -*-
"""Per-family node coverage and degree, for whichever of graph/ and graph2/ declares it.

Edge counts alone hide the thing worth knowing about a newly built family: how much of the
corpus it actually touches. metaqa node text is a bare KB entity name, so its NER graph is
expected to be nearly empty -- that expectation is exactly what the frozen ner_manifest.json
recorded when it set ner_available=false, and it should be measured rather than assumed once
the family exists.
"""
import io
import json
import sys

import numpy as np

sys.path.insert(0, "data/final_canonical")
import pointer_resolver_v2 as V                            # noqa: E402

R = "data/final_canonical"
out = {}
for ds in ["squad", "musique", "metaqa", "hotpotqa", "2wiki", "webqsp"]:
    n = sum(1 for _ in io.open("%s/%s/nodes.jsonl" % (R, ds), encoding="utf-8"))
    out[ds] = {"n_nodes": n, "families": {}}
    for fam in V.families(ds, R):
        g = V.CanonicalGraph(ds, fam, R)
        # bincount, not np.add.at: the unbuffered ufunc.at is ~30x slower and 2wiki
        # structural alone is tens of millions of edges.
        deg = np.bincount(np.concatenate([g.src, g.dst]).astype(np.int64), minlength=n)
        touched = int((deg > 0).sum())
        out[ds]["families"][fam] = {
            "source": V.family_source(ds, fam, R),
            "n_edges": int(g.src.size),
            "nodes_touched": touched,
            "node_coverage": round(touched / float(n), 6),
            "mean_degree_over_touched": round(float(deg.sum()) / max(1, touched), 3),
            "max_degree": int(deg.max()),
        }
        print("%-9s %-11s %-7s edges=%-10s coverage=%.4f  mean_deg=%.2f  max_deg=%d"
              % (ds, fam, out[ds]["families"][fam]["source"], format(int(g.src.size), ","),
                 out[ds]["families"][fam]["node_coverage"],
                 out[ds]["families"][fam]["mean_degree_over_touched"], int(deg.max())),
              flush=True)
json.dump(out, io.open("data/_family_v1/FAMILY_COVERAGE.json", "w", encoding="utf-8"), indent=1)
