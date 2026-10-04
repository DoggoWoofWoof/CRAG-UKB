"""Exploration on DEV_A only: structural block channel (PPR from the retrieval seeds) x fusion rules.
Usage: python -u scratchpad/_l1x90_explore.py <cache> [quick]
Prints DEV_A numbers only; DEV_B is never printed here."""
import itertools
import json
import os
import sys

import numpy as np

import _l1x90_core as X

name = sys.argv[1]
quick = len(sys.argv) > 2 and sys.argv[2] == "quick"
C = X.Cache(name)
A = C.split == "A"
X.log("%s: nq=%d (A=%d B=%d) npart=%d N=%d  cache %s" % (name, C.nq, int(A.sum()), int((~A).sum()), C.npart, C.N, os.path.basename(C.path)))

# ------------------------------------------------------------------ 0. oracles / diagnostics (DEV_A)
ngb = np.array([len(g) for g in C.gb])
feas = ngb <= X.P_MAIN
T_rank = C.base_rank.astype(np.int64)
base_sel = X.fuse(C, T_rank, T_rank, "T")
r0, base_all, base_any = X.evaluate(C, base_sel, "BASE")
X.log("BASE      A:", json.dumps(r0["ALL_split"]["A"]))
X.log("ORACLE@50 A: feasible (gold blocks<=50) %.4f" % float(feas[A].mean()),
      " ".join("hop%d %.4f" % (h, float(feas[A & (C.hops == h)].mean())) for h in sorted(set(C.hops[C.hops >= 0].tolist()))) if (C.hops >= 0).any() else "")
# text channel reach: all gold blocks anywhere in the 200 ranked partitions
pt = X.rank_pos(T_rank, C.npart)
reach_t = np.array([all(pt[i, p] < 10 ** 6 for p in C.gb[i]) for i in range(C.nq)])
X.log("TEXT reach (all gold blocks within the 200 ranked) A: %.4f" % float(reach_t[A].mean()))

# seed quality vs annotated topic entity (diagnostic only, metaqa)
if C.ds == "metaqa":
    tq = {}
    with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "eval_1998.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            tq[o["query_id"]] = o.get("topic_entity_node_id")
    nid = {}
    with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "nodes.jsonl"), encoding="utf-8") as f:
        for k, line in enumerate(f):
            nid[json.loads(line)["node_id"]] = k
    top1 = top5 = have = 0
    for i in range(C.nq):
        if not A[i]:
            continue
        t = tq.get(C.qids[i])
        if t is None or t not in nid:
            continue
        have += 1
        pos = nid[t]
        top1 += int(C.seeds[i, 0] == pos)
        top5 += int(pos in set(C.seeds[i].tolist()))
    X.log("SEED quality A: topic entity == seed[0] %.4f, in seeds[:5] %.4f (n=%d)" % (top1 / max(have, 1), top5 / max(have, 1), have))

# ------------------------------------------------------------------ 1. structural channels
def seeds_of(kind):
    if kind == "top1":
        return C.seeds[:, :1], None
    if kind == "top5":
        return C.seeds[:, :5], None
    if kind == "top5w":
        w = 1.0 / (X.K0 + np.arange(5)[None, :].astype(np.float64))
        return C.seeds[:, :5], np.repeat(w, C.nq, 0)
    if kind == "top20w":
        w = 1.0 / (X.K0 + np.arange(20)[None, :].astype(np.float64))
        return C.ret_rrf[:, :20], np.repeat(w, C.nq, 0)
    raise ValueError(kind)


grid_seeds = ["top1", "top5", "top5w", "top20w"] if not quick else ["top5"]
grid_alpha = [0.15, 0.3, 0.5] if not quick else [0.3]
grid_hub = [None, "sqrt"] if not quick else [None]
grid_fam = ["struct", "sk"] if not quick else ["struct"]
rules = [("S", None), ("RRF", None), ("MINRANK", None), ("HOLD", None), ("CORE", 6), ("CORE", 12), ("CORE", 25)]

rows = []
for sk, al, hb, fm in itertools.product(grid_seeds, grid_alpha, grid_hub, grid_fam):
    sd, sw = seeds_of(sk)
    M = X.ppr_block_mass(C, sd, fam=fm, alpha=al, iters=10, hub=hb, seed_weights=sw)
    S_rank = X.rank_from_scores(M)
    ps = X.rank_pos(S_rank, C.npart)
    reach_s = {K: float(np.array([all(ps[i, p] < K for p in C.gb[i]) for i in range(C.nq)])[A].mean()) for K in (50, 100, 200)}
    chan = "seeds=%s a=%.2f hub=%s fam=%s" % (sk, al, hb, fm)
    line = ["%-40s reach@50/100/200 %.4f/%.4f/%.4f |" % (chan, reach_s[50], reach_s[100], reach_s[200])]
    for rule, B in rules:
        sel = X.fuse(C, T_rank, S_rank, rule, B)
        r, allv, anyv = X.evaluate(C, sel, rule)
        a = r["ALL_split"]["A"]
        g, l, p = X.mcnemar(base_all[A], allv[A])
        tag = rule if B is None else "%s%d" % (rule, B)
        hopstr = "".join(" h%d %.3f" % (h, a["hop%d" % h]["ALL"]) for h in (1, 2, 3) if ("hop%d" % h) in a)
        line.append(" %s %.4f(+%d/-%d)%s" % (tag, a["ALL"], g, l, hopstr))
        rows.append({"channel": chan, "seeds": sk, "alpha": al, "hub": hb, "fam": fm, "rule": tag,
                     "A": a, "gained": g, "lost": l, "p": p, "scope": r["scope_nodes"], "reach_s": reach_s})
    X.log("\n".join(line))

X.wj(os.path.join(X.OUT, "explore_A_%s.json" % name), {"cache": name, "path": C.path, "cache_sha256": X.sha_file(C.path),
                                                        "n_A": int(A.sum()), "BASE_A": r0["ALL_split"]["A"], "rows": rows})
X.log("wrote", os.path.join(X.OUT, "explore_A_%s.json" % name))
