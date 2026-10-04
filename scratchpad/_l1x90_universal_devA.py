"""DEV_A check of the universal selector on one cache: gates G1 and G2 vs BASE (DEV_B never printed).
Usage: python -u _l1x90_universal_devA.py <cache>"""
import json
import os
import sys

import numpy as np

import _l1x90_core as X
import _l1x90_universal as UN

name = sys.argv[1]
C = X.Cache(name)
A = C.split == "A"
T_rank = C.base_rank.astype(np.int64)
base_sel = X.fuse(C, T_rank, T_rank, "T")
r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
out = {"cache": name, "path": C.path, "cache_sha256": X.sha_file(C.path), "n_A": int(A.sum()), "BASE_A": r0["ALL_split"]["A"], "gates": {}}
print("%s A(n=%d): BASE ALL %.4f" % (name, int(A.sum()), r0["ALL_split"]["A"]["ALL"]))
for gate in ("G1", "G2"):
    sel, mode, info = UN.universal_select(C, gate, cache_tag=name)
    r, allv, anyv = X.evaluate(C, sel, gate)
    a = r["ALL_split"]["A"]
    g, l, p = X.mcnemar(base_all[A], allv[A])
    ident = bool(np.array_equal(sel, base_sel))
    hop = "".join(" h%d %.3f" % (h, a["hop%d" % h]["ALL"]) for h in (1, 2, 3) if ("hop%d" % h) in a)
    print("  %s: ALL %.4f (+%d/-%d p=%.2g)%s  ANY %.4f | typed_graph=%s n_rel=%d typed_rate(all q)=%s identical_to_BASE=%s" % (
        gate, a["ALL"], g, l, p, hop, r["ANY_split"]["A"]["ANY"] if "ANY_split" in r else float("nan"), info["typed_graph"], info["n_rel"],
        ("%.3f" % info["typed_rate"]) if "typed_rate" in info else "-", ident))
    out["gates"][gate] = {"A": a, "gained": g, "lost": l, "p": p, "info": info, "identical_to_BASE": ident, "scope_nodes": r["scope_nodes"]}
X.wj(os.path.join(X.OUT, "universal_A_%s.json" % name), out)
