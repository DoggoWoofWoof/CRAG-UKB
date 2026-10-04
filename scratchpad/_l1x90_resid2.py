"""residual categorisation on DEV_A under the lexicographic HARD+UNTYPED configuration."""
import collections, json, os, sys
import numpy as np
import _l1x90_core as X, _l1x90_relwalk as RW, _l1x90_typed as TY
name = "metaqa"; key = "wh"
C = X.Cache(name); A = C.split == "A"; T = TY.Typed(C); Q = TY.questions_of(C)
T_rank = C.base_rank.astype(np.int64)
SD = np.load(os.path.join(X.OUT, "_seeds_%s_anch5d1s1.npy" % name))
z = np.load(os.path.join(X.OUT, "_relwalk_%s_%s.npz" % (name, key))); U = np.load(os.path.join(X.OUT, "_untyped_%s.npz" % name))
M = z["hard_seed"] + z["hard_ans"] + 1e-6 * (z["hard_oth"] + U["oth"])
S_rank = X.rank_from_scores(M); sel = X.fuse(C, T_rank, S_rank, "S"); r, allv, anyv = X.evaluate(C, sel, "S")
ps = X.rank_pos(S_rank, C.npart)
nans = (z["hard_ans"] > 0).sum(1)
qt, te = {}, {}
with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "queries", "dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line); qt[o["query_id"]] = o["qtype"]; te[o["query_id"]] = o.get("topic_entity_node_id")
nid = {}
with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "nodes.jsonl"), encoding="utf-8") as f:
    for k, line in enumerate(f):
        nid[json.loads(line)["node_id"]] = k
REL = {"director": "directed_by", "writer": "written_by", "actor": "starred_actors", "genre": "has_genre", "year": "release_year",
       "language": "in_language", "tags": "has_tags", "imdbrating": "has_imdb_rating", "tag": "has_tags"}
cat = collections.defaultdict(collections.Counter); ex = collections.defaultdict(list)
for i in range(C.nq):
    if not A[i] or allv[i]:
        continue
    h = int(C.hops[i]); topic = nid.get(te[C.qids[i]]); seed_ok = topic in set(int(x) for x in SD[i] if x >= 0)
    order, _ = RW.schedule2(T, Q[i], int(SD[i, 0]), key=key); got = [T.vocab[r_] for r_ in order]
    parts = [REL[p] for p in qt[C.qids[i]].split("_to_") if p in REL]; exp = []
    for p in parts:
        if p not in exp:
            exp.append(p)
    sched = "exact" if got == exp else ("set" if set(got) == set(exp) else ("subset" if set(got) < set(exp) else ("superset" if set(got) > set(exp) else "other")))
    gb = sorted(C.gb[i]); ngb = len(gb); missed = [p for p in gb if ps[i, p] >= 50]
    if not seed_ok:
        c = "seed"
    elif sched != "exact":
        c = "sched_" + sched
    elif ngb > 49:
        c = "infeasible(>49 gold blocks)"
    elif all(z["hard_ans"][i, p] == 0 for p in missed):
        c = "gold_without_answer_mass"
    else:
        c = "ranking(ans blocks=%d)" % min(nans[i] // 25 * 25, 100)
    cat[h][c] += 1
    if len(ex[c]) < 4:
        ex[c].append((h, Q[i], got, exp, ngb, len(missed), int(nans[i]), [round(float(M[i, p]), 4) for p in missed[:4]], [int(ps[i, p]) for p in missed[:4]]))
for h in sorted(cat):
    print("hop%d misses %d / %d: %s" % (h, sum(cat[h].values()), int((A & (C.hops == h)).sum()), dict(cat[h].most_common())))
for c in ex:
    print("--", c)
    for e in ex[c]:
        print("   ", e)
