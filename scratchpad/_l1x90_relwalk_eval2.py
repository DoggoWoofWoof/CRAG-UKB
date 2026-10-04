"""DEV_A: hard answer channel + hedges (untyped walk / mix walk) with w down to the lexicographic limit.
Usage: python -u _l1x90_relwalk_eval2.py <cache> [key]"""
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
z = np.load(os.path.join(X.OUT, "_relwalk_%s_%s.npz" % (name, key)))
V = {t: {k: z["%s_%s" % (t, k)] for k in ("seed", "ans", "oth")} for t in ("hard", "mix", "hardperm")}
t0 = time.time()
U = RW.relwalk(C, T, Q, SD, [[] for _ in range(C.nq)], H=3, mix=False)   # plain untyped frontier walk
print("untyped walk %.0fs" % (time.time() - t0))
np.savez_compressed(os.path.join(X.OUT, "_untyped_%s.npz" % name), seed=U["seed"], oth=U["oth"])


def ev(M, tag, rules=("S", "MINRANK")):
    S_rank = X.rank_from_scores(M)
    line = ["  %-30s" % tag]
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
ws = (0.05, 0.02, 0.01, 0.001, 1e-6)
for w in ws:
    rows["HARD w=%g" % w] = ev(V["hard"]["seed"] + V["hard"]["ans"] + w * V["hard"]["oth"], "HARD w=%g" % w)
for w in ws:
    rows["HARD+UNTYPED w=%g" % w] = ev(V["hard"]["seed"] + V["hard"]["ans"] + w * (V["hard"]["oth"] + U["oth"]), "HARD+UNTYPED w=%g" % w)
for w in ws:
    rows["HARD+MIX w=%g" % w] = ev(V["hard"]["seed"] + V["hard"]["ans"] + V["mix"]["ans"] + w * (V["hard"]["oth"] + V["mix"]["oth"]), "HARD+MIX w=%g" % w)
for w in ws:
    rows["MIX w=%g" % w] = ev(V["mix"]["seed"] + V["mix"]["ans"] + w * V["mix"]["oth"], "MIX w=%g" % w)
# untyped hedge below the typed-other channel (two-level lexicographic)
for w in (0.05, 0.01):
    rows["HARD lex2 w=%g" % w] = ev(V["hard"]["seed"] + V["hard"]["ans"] + w * V["hard"]["oth"] + w * w * U["oth"], "HARD lex2 w=%g" % w)
X.wj(os.path.join(X.OUT, "relwalk2_A_%s_%s.json" % (name, key)), {"cache": name, "key": key, "BASE_A": r0["ALL_split"]["A"], "rows": rows})
