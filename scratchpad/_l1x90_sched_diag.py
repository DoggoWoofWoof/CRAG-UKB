"""DEV_A diagnostic: schedule2 accuracy vs the annotated qtype relation path (metaqa; labels never used
by the method) for both sort keys; also writes the anchored seed rows used by the lane."""
import collections
import json
import os
import sys

import numpy as np

import _l1x90_core as X
import _l1x90_relwalk as RW
import _l1x90_typed as TY

C = X.Cache("metaqa")
A = C.split == "A"
T = TY.Typed(C)
Q = TY.questions_of(C)
qt, te = {}, {}
with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "queries", "dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        qt[o["query_id"]] = o["qtype"]
        te[o["query_id"]] = o.get("topic_entity_node_id")
REL = {"director": "directed_by", "writer": "written_by", "actor": "starred_actors", "genre": "has_genre",
       "year": "release_year", "language": "in_language", "tags": "has_tags", "imdbrating": "has_imdb_rating",
       "tag": "has_tags"}
names = T.names()


def anchored(C, cands, fallback):
    out = np.full((C.nq, 5), -1, np.int64)
    for i in range(C.nq):
        c = [int(x) for x in cands[i] if x >= 0]
        qs = set(TY._positions(Q[i]))
        keep = []
        for x in c:
            ns = [t for t in TY._positions(names[x]) if t not in TY.STOP and len(t) >= 3]
            if ns and all(t in qs for t in ns):
                keep.append(x)
        if not keep:
            keep = [int(x) for x in fallback[i] if x >= 0]
        out[i, :len(keep)] = keep[:5]
    return out


d1 = C.ret_dense[:, 0]
s1 = C.ret_splade[:, 0]
seeds2 = np.stack([d1, np.where(d1 == s1, -1, s1)], 1)
SD = anchored(C, C.seeds[:, :5], seeds2)
np.save(os.path.join(X.OUT, "_seeds_metaqa_anch5d1s1.npy"), SD)


def expected(qid):
    parts = [REL[p] for p in qt[qid].split("_to_") if p in REL]
    exp = []
    for p in parts:
        if p not in exp:
            exp.append(p)
    return exp


for key in ("wh", "after"):
    n = ok = sok = 0
    bad = collections.Counter()
    per = collections.defaultdict(lambda: [0, 0])
    ex = collections.defaultdict(list)
    for i in range(C.nq):
        if not A[i]:
            continue
        order, _ = RW.schedule2(T, Q[i], int(SD[i, 0]), key=key)
        got = [T.vocab[r] for r in order]
        exp = expected(C.qids[i])
        n += 1
        ok += got == exp
        sok += set(got) == set(exp)
        per[int(C.hops[i])][0] += got == exp
        per[int(C.hops[i])][1] += 1
        if got != exp:
            bad[(tuple(exp), tuple(got))] += 1
            if len(ex[(tuple(exp), tuple(got))]) < 2:
                ex[(tuple(exp), tuple(got))].append(Q[i])
    print("key=%s: schedule exact %.3f set exact %.3f per hop %s" % (key, ok / n, sok / n, {h: round(a / b, 3) for h, (a, b) in sorted(per.items())}))
    for k, c in bad.most_common(10):
        print("   %3d %s -> %s  e.g. %s" % (c, k[0], k[1], ex[k][:2]))
