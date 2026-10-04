"""DEV_A evaluation: lexical seeds (fallback: anchored retrieval seeds) + schedule3 + relation-decomposed
hard walk + untyped hedge.  Usage: python -u _l1x90_eval3.py <cache>"""
import json
import os
import sys
import time

import numpy as np

import _l1x90_core as X
import _l1x90_relwalk as RW
import _l1x90_seeds as SE
import _l1x90_typed as TY

name = sys.argv[1]
C = X.Cache(name)
A = C.split == "A"
T = TY.Typed(C)
Q = TY.questions_of(C)
names = T.names()
T_rank = C.base_rank.astype(np.int64)
base_sel = X.fuse(C, T_rank, T_rank, "T")
r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
# seeds
idx = SE.build_name_index(names)
d1 = C.ret_dense[:, 0]
s1 = C.ret_splade[:, 0]
SD = np.full((C.nq, 5), -1, np.int64)
src = np.zeros(C.nq, np.int8)
for i in range(C.nq):
    lex = SE.lexical_seeds(Q[i], idx, names, T if T.has_rel else None)
    if lex:
        SD[i, :len(lex)] = lex[:5]
        src[i] = 1
    else:
        fb = [int(d1[i])] + ([int(s1[i])] if s1[i] != d1[i] else [])
        SD[i, :len(fb)] = fb
np.save(os.path.join(X.OUT, "_seeds_%s_lex.npy" % name), SD)
orders, anch = [], []
for i in range(C.nq):
    sd = [int(x) for x in SD[i] if x >= 0]
    o, a = SE.schedule3(T, Q[i], sd, key="wh") if T.has_rel else ([], False)
    orders.append(o)
    anch.append(a)
ks = np.array([len(o) for o in orders])
print("%s A: BASE %.4f | lexical seeds %.3f anchored %.3f matched>=1 %.3f  k dist %s" % (
    name, r0["ALL_split"]["A"]["ALL"], float(src.mean()), float(np.mean(anch)), float(np.mean(ks > 0)), {int(k): int((ks == k).sum()) for k in np.unique(ks)}))
t0 = time.time()
H = RW.relwalk(C, T, Q, SD, orders, H=3, mix=False)
U = RW.relwalk(C, T, Q, SD, [[] for _ in range(C.nq)], H=3, mix=False)
print("walks %.0fs" % (time.time() - t0))
np.savez_compressed(os.path.join(X.OUT, "_walk3_%s.npz" % name), seed=H["seed"], ans=H["ans"], oth=H["oth"], unt=U["oth"], k=ks, anchored=np.array(anch), src=src)


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
for w in (1.0, 0.3, 0.05, 1e-6):
    rows["HARD+UNTYPED w=%g" % w] = ev(H["seed"] + H["ans"] + w * (H["oth"] + U["oth"]), "HARD+UNTYPED w=%g" % w)
    rows["HARD w=%g" % w] = ev(H["seed"] + H["ans"] + w * H["oth"], "HARD w=%g" % w)
rows["UNTYPED"] = ev(U["seed"] + U["oth"], "UNTYPED only")
X.wj(os.path.join(X.OUT, "eval3_A_%s.json" % name), {"cache": name, "BASE_A": r0["ALL_split"]["A"], "rows": rows,
                                                       "lexical_seed_rate": float(src.mean()), "anchored_rate": float(np.mean(anch))})
