"""Offline (no-encoder) MuSiQue coverage probe for the corpus-derived sentence-edge /
k-hop path construction used by kg_relsig.  Answers the PREMISE question before any GPU:

  are the gold supporting docs reachable from question-derived topics via the
  corpus doc<->doc title-mention graph at 2-4 hops, WITHOUT using gold decomposition?

Everything here is inference-safe: topics come ONLY from find_topics(question_text);
question->gold neighbors are used ONLY as labels (reachability / stratification),
never to CONSTRUCT the graph or the paths.  Mirrors the 2wiki_clean construction in
src/experiments/kg_relsig.py (find_topics / connect / bounded k-hop BFS).
"""
import json, re, collections, statistics as st

MFILE = "data/processed/master_nodes_musique_clean.json"
MAXHOP = 4
CAP = 64                    # per-node neighbor expansion cap (matches kg_relsig per_query cap)

nodes = json.load(open(MFILE, encoding="utf-8"))
docs = [n for n in nodes if n.get("metadata", {}).get("type") == "document"]
qs   = [n for n in nodes if n.get("metadata", {}).get("type") == "question"]

# ---- doc substrate (mirrors CoreEngine.nodes: questions excluded) ----
id2idx = {n["node_id"]: i for i, n in enumerate(docs)}
N = len(docs)
idx2title = {}; content_by_row = {}; t2r = collections.defaultdict(list)
for i, n in enumerate(docs):
    ti = n.get("metadata", {}).get("title", "").strip()
    content_by_row[i] = n.get("content", "")
    if ti:
        idx2title[i] = ti; t2r[ti].append(i)
ents = [e for e in t2r if e]

# ---- undirected doc<->doc adjacency A (struct edges = node.neighbors, doc-only) ----
adj = [[] for _ in range(N)]
edge_set = [set() for _ in range(N)]
for i, n in enumerate(docs):
    for nb in n.get("neighbors", []):
        j = id2idx.get(nb)
        if j is not None and j != i and j not in edge_set[i]:
            edge_set[i].add(j); adj[i].append(j)
# mirror (undirected)
for i in range(N):
    for j in adj[i]:
        if i not in edge_set[j]:
            edge_set[j].add(i); adj[j].append(i)
adj = [sorted(a) for a in edge_set]

# ---- topic finder: identical logic to kg_relsig 2wiki_clean find_topics ----
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

def bfs_hops(seed_rows, targets, maxhop=MAXHOP, cap=CAP):
    """min hop (1..maxhop) from any seed row to each target row; 0 if target is a seed."""
    hop = {}
    for s in seed_rows:
        hop[s] = 0
    frontier = set(seed_rows)
    for h in range(1, maxhop + 1):
        nxt = set()
        for s in frontier:
            for x in adj[s][:cap]:
                if x not in hop:
                    hop[x] = h; nxt.add(x)
        frontier = nxt
        if not frontier:
            break
    return {g: hop.get(g) for g in targets}

# ---- run over ALL questions, stratified by #golds (2/3/4 = intrinsic hop depth) ----
strata = collections.defaultdict(lambda: {
    "n": 0, "with_topic": 0, "n_topics": [], "topic_hits": 0,
    "gold_total": 0, "gold_reach": collections.Counter(),   # keyed by hop 0/1/2/3/4/'unreach'
    "all_golds_reached": 0,
})
examples = []
for qn in qs:
    golds = [id2idx[g] for g in qn.get("neighbors", []) if g in id2idx]
    if not golds:
        continue
    nk = len(golds)                                   # 2 / 3 / 4  -> intrinsic hop bucket
    bucket = f"{nk}gold"
    S = strata[bucket]; S["n"] += 1; S["gold_total"] += len(golds)
    q = qn["content"]
    topics = find_topics(q)
    topic_rows = [r for tp in topics for r in t2r.get(tp, [])]
    S["n_topics"].append(len(topics))
    if topics:
        S["with_topic"] += 1
    gset = set(golds)
    # golds directly named in the question (topic hits) vs reachable via graph
    topic_gold = gset & set(topic_rows)
    S["topic_hits"] += len(topic_gold)
    reach = bfs_hops(topic_rows, golds) if topic_rows else {g: None for g in golds}
    hops_seen = []
    for g in golds:
        h = reach.get(g)
        key = "unreach" if h is None else ("topic" if h == 0 else h)
        S["gold_reach"][key] += 1
        hops_seen.append(h if h is not None else 99)
    if topic_rows and max(hops_seen) <= MAXHOP:
        S["all_golds_reached"] += 1
    if len(examples) < 12 and nk >= 3 and topics:
        examples.append({
            "q": q[:110], "n_gold": nk, "topics": topics,
            "gold_titles": [idx2title.get(g, "?") for g in golds],
            "gold_hops": {idx2title.get(g, "?"): reach.get(g) for g in golds},
        })

# ---- report ----
print("=" * 78)
print("MuSiQue offline edge/path coverage probe  (N_docs=%d, edges=%d, isolated=%d)" %
      (N, sum(len(a) for a in adj) // 2, sum(1 for a in adj if not a)))
print("MAXHOP=%d  CAP=%d   inference-safe: topics from question text only" % (MAXHOP, CAP))
print("=" * 78)
tot_q = tot_allreach = 0
for bucket in sorted(strata):
    S = strata[bucket]; n = S["n"]; tot_q += n; tot_allreach += S["all_golds_reached"]
    gr = S["gold_reach"]; gt = S["gold_total"]
    def pc(k): return "%d (%.0f%%)" % (gr.get(k, 0), 100 * gr.get(k, 0) / gt) if gt else "0"
    print("\n[%s]  n_questions=%d" % (bucket, n))
    print("  topic found in q:     %d (%.0f%%)   avg #topics=%.2f" %
          (S["with_topic"], 100 * S["with_topic"] / n, st.mean(S["n_topics"])))
    print("  gold reachability (of %d gold docs):" % gt)
    print("    topic-named (0hop): %s" % pc("topic"))
    for h in (1, 2, 3, 4):
        print("    %d-hop:             %s" % (h, pc(h)))
    print("    UNREACH (>%dhop):    %s" % (MAXHOP, pc("unreach")))
    print("  ALL golds reachable (<=%dhop): %d/%d (%.0f%%)" %
          (MAXHOP, S["all_golds_reached"], n, 100 * S["all_golds_reached"] / n))
print("\n" + "-" * 78)
print("OVERALL: all-golds-reachable %d/%d (%.1f%%)" % (tot_allreach, tot_q, 100 * tot_allreach / tot_q))
print("-" * 78)
print("\nEXAMPLES (3-4 gold, topics found; gold_hops=min hop from topic set):")
for e in examples:
    print(json.dumps(e, ensure_ascii=False))
