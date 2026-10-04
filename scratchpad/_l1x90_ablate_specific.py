"""How much of the metaqa gain depends on the dataset-fitted pieces?  DEV_A only (diagnostic ladder).
A0 full method | A1 closed-class stop words only | A2 + retrieval seeds (no exact-name seeds)
| A3 + order-free (permutation-averaged answer channel; no positional heuristics)
| A4 + exact-stem matching only (no prefix/substring/compound conflation)."""
import itertools
import json
import os
import sys

import numpy as np

import _l1x90_core as X
import _l1x90_relwalk as RW
import _l1x90_seeds as SE
import _l1x90_typed as TY

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
C = X.Cache(name)
A = C.split == "A"
T = TY.Typed(C)
Q = TY.questions_of(C)
names = T.names()
T_rank = C.base_rank.astype(np.int64)
base_sel = X.fuse(C, T_rank, T_rank, "T")
r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
CLOSED = set("a an the of in on by to for at from as with and or is are was were be been do does did has have had that this these those it its not also same who whom whose what which when where how why".split())
DOMAIN_STOP = set(TY.STOP)
idx = SE.build_name_index(names)
d1 = C.ret_dense[:, 0]
s1 = C.ret_splade[:, 0]


def lex_seeds():
    SD = np.full((C.nq, 5), -1, np.int64)
    for i in range(C.nq):
        lex = SE.lexical_seeds(Q[i], idx, names, T)
        if lex:
            SD[i, :len(lex)] = lex[:5]
        else:
            fb = [int(d1[i])] + ([int(s1[i])] if s1[i] != d1[i] else [])
            SD[i, :len(fb)] = fb
    return SD


def ret_seeds():
    SD = np.full((C.nq, 5), -1, np.int64)
    for i in range(C.nq):
        fb = [int(d1[i])] + ([int(s1[i])] if s1[i] != d1[i] else [])
        SD[i, :len(fb)] = fb
    return SD


def orders_of(SD):
    out = []
    for i in range(C.nq):
        sd = [int(x) for x in SD[i] if x >= 0]
        o, a = SE.schedule3(T, Q[i], sd, key="wh")
        out.append(o)
    return out


def run(tag, SD, orders, perm):
    Hh = RW.relwalk(C, T, Q, SD, orders, H=3, mix=False, perm=perm)
    U = RW.relwalk(C, T, Q, SD, [[] for _ in range(C.nq)], H=3, mix=False)
    M = Hh["seed"] + Hh["ans"] + 1e-6 * (Hh["oth"] + U["oth"])
    S_rank = X.rank_from_scores(M)
    anchored = np.array([len(o) >= 0 for o in orders])
    sel = X.fuse(C, T_rank, S_rank, "S")
    r, allv, anyv = X.evaluate(C, sel, tag)
    a = r["ALL_split"]["A"]
    g, l, p = X.mcnemar(base_all[A], allv[A])
    ks = np.array([len(o) for o in orders])
    print("  %-52s ALL %.4f (+%d/-%d) h1 %.3f h2 %.3f h3 %.3f | matched>=1 %.3f" % (
        tag, a["ALL"], g, l, a["hop1"]["ALL"], a["hop2"]["ALL"], a["hop3"]["ALL"], float((ks >= 1)[A].mean())), flush=True)
    return {"A": a, "gained": g, "lost": l, "p": p}


print("%s DEV_A n=%d BASE %.4f" % (name, int(A.sum()), r0["ALL_split"]["A"]["ALL"]))
res = {}
SDl = lex_seeds()
res["A0 full method"] = run("A0 full method", SDl, orders_of(SDl), False)
TY.STOP = CLOSED                                     # drop the domain words (film, movie, person, list, share, name ...)
SDl2 = lex_seeds()
res["A1 closed-class stop words only"] = run("A1 closed-class stop words only", SDl2, orders_of(SDl2), False)
SDr = ret_seeds()
res["A2 + retrieval seeds (dense/SPLADE top-1)"] = run("A2 + retrieval seeds (dense/SPLADE top-1)", SDr, orders_of(SDr), False)
res["A3 + order-free (permutation-averaged answer channel)"] = run("A3 + order-free (permutation-averaged answer channel)", SDr, orders_of(SDr), True)
_tm = TY.tok_match
TY.tok_match = lambda a, b: a == b                   # exact stem equality only
res["A4 + exact-stem matching only"] = run("A4 + exact-stem matching only", SDr, orders_of(SDr), True)
TY.tok_match = _tm
TY.STOP = DOMAIN_STOP
X.wj(os.path.join(X.OUT, "ablation_specific_A_%s.json" % name), {"cache": name, "BASE_A": r0["ALL_split"]["A"], "rows": res})
