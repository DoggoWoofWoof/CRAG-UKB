"""Does a richer inference-safe edge type (NER shares-entity, already integrated) recover
MuSiQue gold-chain reachability that title-mention edges miss (~31% ceiling)?

Loads the cached NER CSR (data/ukb_storage/musique_clean/ner_edges_w_df25.pkl) and the
title-mention struct graph (doc `neighbors`), and measures all-golds-<=4hop reachability
from question topics for: struct-only, NER-only, struct+NER union. Inference-safe.
"""
import json, re, collections, pickle, statistics as st
import numpy as np, scipy.sparse as sp

MFILE = "data/processed/master_nodes_musique_clean.json"
NERPKL = "data/ukb_storage/musique_clean/ner_edges_w_df25.pkl"
MAXHOP = 4; CAP = 64

nodes = json.load(open(MFILE, encoding="utf-8"))
docs = [n for n in nodes if n.get("metadata", {}).get("type") == "document"]
qs   = [n for n in nodes if n.get("metadata", {}).get("type") == "question"]
id2idx = {n["node_id"]: i for i, n in enumerate(docs)}
N = len(docs)
idx2title = {}; t2r = collections.defaultdict(list)
for i, n in enumerate(docs):
    ti = n.get("metadata", {}).get("title", "").strip()
    if ti:
        idx2title[i] = ti; t2r[ti].append(i)
ents = [e for e in t2r if e]

# struct (title-mention) adjacency lists
struct = [set() for _ in range(N)]
for i, n in enumerate(docs):
    for nb in n.get("neighbors", []):
        j = id2idx.get(nb)
        if j is not None and j != i:
            struct[i].add(j); struct[j].add(i)

# NER adjacency from cached CSR
A_ner = pickle.load(open(NERPKL, "rb")).tocsr()
print("NER CSR shape=%s nnz=%d (undirected edges=%d)  [N_docs=%d]" %
      (A_ner.shape, A_ner.nnz, A_ner.nnz // 2, N))
assert A_ner.shape[0] == N, "NER matrix row order does not match doc count — alignment unknown"
ner = [set(int(x) for x in A_ner.indices[A_ner.indptr[i]:A_ner.indptr[i + 1]] if int(x) != i) for i in range(N)]

def _toks(s): return re.findall(r"[a-z0-9]+", s.lower())
ttoks = {t: set(_toks(t)) for t in ents}
dfc = collections.Counter(w for ts in ttoks.values() for w in ts)
keyidx = collections.defaultdict(list)
for t in ents:
    if ttoks[t]:
        keyidx[min(ttoks[t], key=lambda w: dfc[w])].append(t)

def find_topics(q):
    ql = q.lower(); qt = set(_toks(q)); seen = set(); found = []
    for w in qt:
        for t in keyidx.get(w, ()):
            if t in seen:
                continue
            seen.add(t)
            if ttoks[t] and ttoks[t] <= qt and len(t) >= 3 and t.lower() in ql:
                found.append(t)
    found.sort(key=lambda s: -len(s)); topics = []
    for t in found:
        if not any(t != o and t.lower() in o.lower() for o in topics):
            topics.append(t)
    return topics[:4]

def make_bfs(adjlists):
    adj = [sorted(a) for a in adjlists]
    def bfs(seed_rows, targets):
        hop = {s: 0 for s in seed_rows}; frontier = set(seed_rows)
        for h in range(1, MAXHOP + 1):
            nxt = set()
            for s in frontier:
                for x in adj[s][:CAP]:
                    if x not in hop:
                        hop[x] = h; nxt.add(x)
            frontier = nxt
            if not frontier:
                break
        return {g: hop.get(g) for g in targets}
    n_edges = sum(len(a) for a in adj) // 2
    n_iso = sum(1 for a in adj if not a)
    return bfs, n_edges, n_iso

union = [struct[i] | ner[i] for i in range(N)]
GRAPHS = {"struct(title)": struct, "NER-only": ner, "struct+NER": union}

# precompute per-question topics + golds once
Q = []
for qn in qs:
    golds = [id2idx[g] for g in qn.get("neighbors", []) if g in id2idx]
    if not golds:
        continue
    topics = find_topics(qn["content"]); topic_rows = [r for tp in topics for r in t2r.get(tp, [])]
    Q.append((topic_rows, golds, len(golds)))

print("=" * 74)
for gname, adjl in GRAPHS.items():
    bfs, ne, niso = make_bfs(adjl)
    strat = collections.defaultdict(lambda: {"n": 0, "all": 0, "gt": 0, "unreach": 0})
    for topic_rows, golds, nk in Q:
        S = strat[f"{nk}gold"]; S["n"] += 1; S["gt"] += len(golds)
        reach = bfs(topic_rows, golds) if topic_rows else {g: None for g in golds}
        hs = []
        for g in golds:
            h = reach.get(g)
            if h is None:
                S["unreach"] += 1
            hs.append(h if h is not None else 99)
        if topic_rows and max(hs) <= MAXHOP:
            S["all"] += 1
    tq = sum(S["n"] for S in strat.values()); ta = sum(S["all"] for S in strat.values())
    print("\n### %-14s  edges=%d  isolated=%d/%d" % (gname, ne, niso, N))
    for b in sorted(strat):
        S = strat[b]
        print("   [%s] all-golds<=4hop %d/%d (%.0f%%)   gold-unreach %.0f%%" %
              (b, S["all"], S["n"], 100 * S["all"] / S["n"], 100 * S["unreach"] / S["gt"]))
    print("   OVERALL all-golds-reachable: %d/%d (%.1f%%)" % (ta, tq, 100 * ta / tq))
