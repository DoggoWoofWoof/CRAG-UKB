"""DEV_A diagnostic of lexical seeds + schedule3 (metaqa annotations used only to measure)."""
import collections
import json
import os

import numpy as np

import _l1x90_core as X
import _l1x90_seeds as SE
import _l1x90_typed as TY

C = X.Cache("metaqa")
A = C.split == "A"
T = TY.Typed(C)
Q = TY.questions_of(C)
names = T.names()
idx = SE.build_name_index(names)
SD0 = np.load(os.path.join(X.OUT, "_seeds_metaqa_anch5d1s1.npy"))
qt, te = {}, {}
with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "queries", "dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        qt[o["query_id"]] = o["qtype"]
        te[o["query_id"]] = o.get("topic_entity_node_id")
nid = {}
with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "nodes.jsonl"), encoding="utf-8") as f:
    for k, line in enumerate(f):
        nid[json.loads(line)["node_id"]] = k
REL = {"director": "directed_by", "writer": "written_by", "actor": "starred_actors", "genre": "has_genre", "year": "release_year",
       "language": "in_language", "tags": "has_tags", "imdbrating": "has_imdb_rating", "tag": "has_tags"}
SD = np.full((C.nq, 5), -1, np.int64)
nlex = 0
hist = collections.Counter()
for i in range(C.nq):
    lex = SE.lexical_seeds(Q[i], idx, names, T)
    if lex:
        nlex += 1
        SD[i, :len(lex)] = lex[:5]
    else:
        SD[i] = SD0[i]
    hist[len(lex)] += 1
np.save(os.path.join(X.OUT, "_seeds_metaqa_lex.npy"), SD)
n = tp = tp0 = ok = sok = single = 0
bad = collections.Counter()
miss_ex = []
for i in range(C.nq):
    if not A[i]:
        continue
    n += 1
    topic = nid.get(te[C.qids[i]])
    sd = [int(x) for x in SD[i] if x >= 0]
    tp += topic in sd
    tp0 += topic in set(int(x) for x in SD0[i] if x >= 0)
    single += len(sd) == 1
    if topic not in sd and len(miss_ex) < 12:
        miss_ex.append((Q[i], names[topic], [names[x] for x in sd]))
    order, _ = SE.schedule3(T, Q[i], sd, key="wh")
    got = [T.vocab[r] for r in order]
    parts = [REL[p] for p in qt[C.qids[i]].split("_to_") if p in REL]
    exp = []
    for p in parts:
        if p not in exp:
            exp.append(p)
    ok += got == exp
    sok += set(got) == set(exp)
    if got != exp:
        bad[(tuple(exp), tuple(got))] += 1
print("lexical seeds found for %d/%d queries; seeds/query histogram %s" % (nlex, C.nq, dict(sorted(hist.items()))))
print("A: topic in seeds %.4f (retrieval-anchored: %.4f)  single-seed %.3f | schedule3 exact %.3f set %.3f" % (tp / n, tp0 / n, single / n, ok / n, sok / n))
for e in miss_ex:
    print("   MISS", e)
for k, c in bad.most_common(8):
    print("   %3d %s -> %s" % (c, k[0], k[1]))
