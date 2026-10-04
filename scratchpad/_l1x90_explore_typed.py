"""DEV_A exploration of the relation-conditioned walk. Usage: python -u _l1x90_explore_typed.py <cache> [alpha ...]"""
import collections
import json
import os
import sys

import numpy as np

import _l1x90_core as X
import _l1x90_typed as TY

name = sys.argv[1]
alphas = [float(a) for a in sys.argv[2:]] or [0.3]
C = X.Cache(name)
A = C.split == "A"
T = TY.Typed(C)
Q = TY.questions_of(C)
X.log("%s: nq=%d A=%d  relation labels: %s (%d types)" % (name, C.nq, int(A.sum()), T.has_rel, len(T.vocab)))
T_rank = C.base_rank.astype(np.int64)
base_sel = X.fuse(C, T_rank, T_rank, "T")
r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
X.log("BASE A:", json.dumps(r0["ALL_split"]["A"]))

# ---- matcher diagnostic vs the annotated qtype relation path (metaqa only; labels never used by the method)
if C.ds == "metaqa" and T.has_rel:
    REL = {"director": "directed_by", "writer": "written_by", "actor": "starred_actors", "genre": "has_genre", "year": "release_year",
           "language": "in_language", "tags": "has_tags", "imdbrating": "has_imdb_rating", "imdbvotes": "has_imdb_votes",
           "tag": "has_tags"}
    qt = {}
    with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "eval_1998.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            qt[o["query_id"]] = o["qtype"]
    vidx = {v: i for i, v in enumerate(T.vocab)}
    exact = sup = sub = 0
    n = 0
    miss = collections.Counter()
    for i in range(C.nq):
        if not A[i]:
            continue
        q = qt.get(C.qids[i])
        if q is None:
            continue
        need = set(vidx[REL[p]] for p in q.split("_to_") if p in REL and p != "movie")
        got = T.match(Q[i], int(C.seeds[i, 0]))
        n += 1
        if got == need:
            exact += 1
        elif need <= got:
            sup += 1
        elif got < need:
            sub += 1
        for p in need - got:
            miss[T.vocab[p]] += 1
    X.log("MATCHER A (n=%d): exact %.3f  superset %.3f  strict-subset %.3f  other %.3f | missed relation counts %s"
          % (n, exact / n, sup / n, sub / n, 1 - (exact + sup + sub) / n, dict(miss.most_common())))

# ---- walks
w5 = np.repeat((1.0 / (X.K0 + np.arange(5)[None, :].astype(np.float64))), C.nq, 0)
rules = [("S", None), ("RRF", None), ("MINRANK", None), ("HOLD", None), ("CORE", 6), ("CORE", 12), ("CORE", 25)]
rows = []
for al in alphas:
    for hard in (False, True):
        M, matched = TY.typed_block_mass(C, T, Q, C.seeds[:, :5], alpha=al, iters=10, seed_weights=w5, hard=hard, log_every=0)
        S_rank = X.rank_from_scores(M)
        ps = X.rank_pos(S_rank, C.npart)
        reach = {K: float(np.array([all(ps[i, p] < K for p in C.gb[i]) for i in range(C.nq)])[A].mean()) for K in (50, 100, 200)}
        chan = "typed(%s) seeds=top5w a=%.2f" % ("hard" if hard else "mix", al)
        line = ["%-40s reach@50/100/200 %.4f/%.4f/%.4f  matched/q %.2f |" % (chan, reach[50], reach[100], reach[200],
                                                                             float(np.mean([len(m) for m in matched])))]
        for rule, B in rules:
            sel = X.fuse(C, T_rank, S_rank, rule, B)
            r, allv, anyv = X.evaluate(C, sel, rule)
            a = r["ALL_split"]["A"]
            g, l, p = X.mcnemar(base_all[A], allv[A])
            tag = rule if B is None else "%s%d" % (rule, B)
            hopstr = "".join(" h%d %.3f" % (h, a["hop%d" % h]["ALL"]) for h in (1, 2, 3) if ("hop%d" % h) in a)
            line.append(" %s %.4f(+%d/-%d)%s" % (tag, a["ALL"], g, l, hopstr))
            rows.append({"channel": chan, "alpha": al, "hard": hard, "rule": tag, "A": a, "gained": g, "lost": l, "p": p,
                         "scope": r["scope_nodes"], "reach_s": reach})
        X.log("\n".join(line))
X.wj(os.path.join(X.OUT, "explore_typed_A_%s.json" % name), {"cache": name, "path": C.path, "cache_sha256": X.sha_file(C.path),
                                                              "n_A": int(A.sum()), "BASE_A": r0["ALL_split"]["A"], "rows": rows})
