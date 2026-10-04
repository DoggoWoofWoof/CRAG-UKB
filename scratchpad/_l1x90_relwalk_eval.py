"""DEV_A evaluation of the relation-decomposed walk: score = seed + ans + w * oth, several variants.
Usage: python -u _l1x90_relwalk_eval.py <cache> [key]"""
import json
import os
import sys
import time

import numpy as np

import _l1x90_core as X
import _l1x90_relwalk as RW
import _l1x90_typed as TY

name = sys.argv[1]
key = sys.argv[2] if len(sys.argv) > 2 else "wh"
C = X.Cache(name)
A = C.split == "A"
T = TY.Typed(C)
Q = TY.questions_of(C)
T_rank = C.base_rank.astype(np.int64)
base_sel = X.fuse(C, T_rank, T_rank, "T")
r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
SD = np.load(os.path.join(X.OUT, "_seeds_%s_anch5d1s1.npy" % name))
orders, anch = [], []
for i in range(C.nq):
    o, a = RW.schedule2(T, Q[i], int(SD[i, 0]), key=key) if T.has_rel else ([], False)
    orders.append(o)
    anch.append(a)
ks = np.array([len(o) for o in orders])
print("%s A: BASE %.4f | anchored %.3f  matched>=1 %.3f  k dist %s" % (name, r0["ALL_split"]["A"]["ALL"], float(np.mean(anch)), float(np.mean(ks > 0)),
                                                                        {int(k): int((ks == k).sum()) for k in np.unique(ks)}))
t0 = time.time()
V = {}
for tag, kw in (("hard", dict(mix=False)), ("mix", dict(mix=True)), ("hardperm", dict(mix=False, perm=True))):
    V[tag] = RW.relwalk(C, T, Q, SD, orders, H=3, **kw)
print("walks %.0fs" % (time.time() - t0))
np.savez_compressed(os.path.join(X.OUT, "_relwalk_%s_%s.npz" % (name, key)), **{"%s_%s" % (t, k): V[t][k] for t in V for k in V[t]}, k=ks, anchored=np.array(anch))


def ev(M, tag, rules=("S", "MINRANK")):
    S_rank = X.rank_from_scores(M)
    line = ["  %-28s" % tag]
    res = {}
    for rule in rules:
        sel = X.fuse(C, T_rank, S_rank, rule)
        r, allv, anyv = X.evaluate(C, sel, rule)
        a = r["ALL_split"]["A"]
        g, l, p = X.mcnemar(base_all[A], allv[A])
        hop = "".join(" h%d %.3f" % (h, a["hop%d" % h]["ALL"]) for h in (1, 2, 3) if ("hop%d" % h) in a)
        line.append(" %s %.4f(+%d/-%d)%s" % (rule, a["ALL"], g, l, hop))
        res[rule] = {"A": a, "gained": g, "lost": l, "p": p}
    print("".join(line), flush=True)
    return res


rows = {}
for comb in ("hard", "mix", "hard+mix", "hardperm+mix", "hardperm"):
    parts = comb.split("+")
    for w in (1.0, 0.5, 0.3, 0.15, 0.05):
        M = sum(V[p]["seed"] + V[p]["ans"] + w * V[p]["oth"] for p in parts)
        rows["%s w=%.2f" % (comb, w)] = ev(M, "%s w=%.2f" % (comb, w))
X.wj(os.path.join(X.OUT, "relwalk_A_%s_%s.json" % (name, key)), {"cache": name, "key": key, "BASE_A": r0["ALL_split"]["A"], "rows": rows})
