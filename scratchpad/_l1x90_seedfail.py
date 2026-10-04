import collections, json, os
import numpy as np
import _l1x90_core as X, _l1x90_typed as TY
C = X.Cache("metaqa"); A = C.split == "A"; T = TY.Typed(C); Q = TY.questions_of(C); names = T.names()
SD = np.load(os.path.join(X.OUT, "_seeds_metaqa_anch5d1s1.npy"))
te = {}
with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "queries", "dev.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line); te[o["query_id"]] = o.get("topic_entity_node_id")
nid = {}
with open(os.path.join(X.REPO, "data", "final_canonical", "metaqa", "nodes.jsonl"), encoding="utf-8") as f:
    for k, line in enumerate(f):
        nid[json.loads(line)["node_id"]] = k
n = 0
for i in range(C.nq):
    if not A[i]: continue
    topic = nid.get(te[C.qids[i]]); sd = [int(x) for x in SD[i] if x >= 0]
    if topic in sd: continue
    n += 1
    tn = names[topic]
    print(int(C.hops[i]), repr(Q[i]), "| topic:", repr(tn), "| verbatim:", tn.lower() in Q[i].lower(), "| seeds:", [names[x] for x in sd[:3]], "| dense top3:", [names[int(x)] for x in C.ret_dense[i, :3]], "| splade top3:", [names[int(x)] for x in C.ret_splade[i, :3]])
print("seed failures A:", n)
