"""Confound check for the MuSiQue coverage probe: is the low reachability caused by an
IMPOVERISHED edge set (known musique title-edge loader bug, see memory) rather than a
genuine property of the questions?

We re-induce the FULL inference-safe title-mention graph from doc CONTENT (doc A mentions
doc B's title -> edge A->B), globally, not just the substrate's stored `neighbors`, and
re-measure gold-chain reachability from question topics.  Still inference-safe: uses only
corpus text + question text; question->gold links used only as labels.
"""
import json, re, collections, statistics as st

MFILE = "data/processed/master_nodes_musique_clean.json"
MAXHOP = 4; CAP = 64

nodes = json.load(open(MFILE, encoding="utf-8"))
docs = [n for n in nodes if n.get("metadata", {}).get("type") == "document"]
qs   = [n for n in nodes if n.get("metadata", {}).get("type") == "question"]
id2idx = {n["node_id"]: i for i, n in enumerate(docs)}
N = len(docs)
idx2title = {}; content_by_row = {}; t2r = collections.defaultdict(list)
for i, n in enumerate(docs):
    ti = n.get("metadata", {}).get("title", "").strip()
    content_by_row[i] = n.get("content", "")
    if ti:
        idx2title[i] = ti; t2r[ti].append(i)
ents = [e for e in t2r if e]

def _toks(s): return re.findall(r"[a-z0-9]+", s.lower())
def _specific(t): return len(t) >= 4 and re.search(r"[A-Z][a-z]", t) is not None
ttoks = {t: set(_toks(t)) for t in ents}
dfc = collections.Counter(w for ts in ttoks.values() for w in ts)
keyidx = collections.defaultdict(list)
for t in ents:
    if ttoks[t]:
        keyidx[min(ttoks[t], key=lambda w: dfc[w])].append(t)

_wb = {}
def _mentions(title, text):
    p = _wb.get(title)
    if p is None:
        p = _wb[title] = re.compile(r"(?<![A-Za-z])" + re.escape(title) + r"(?![A-Za-z])")
    return p.search(text) is not None

# ---- induce full title-mention edges: for each doc, candidate titles via rare tokens in its content ----
adj = [set() for _ in range(N)]
for i in range(N):
    txt = content_by_row[i]
    if not txt:
        continue
    cand = set()
    for w in set(_toks(txt)):
        for t in keyidx.get(w, ()):
            cand.add(t)
    for t in cand:
        if not _specific(t):
            continue
        for j in t2r[t]:
            if j != i and idx2title.get(i, "") != t and _mentions(t, txt):
                adj[i].add(j)
# undirected
for i in range(N):
    for j in list(adj[i]):
        adj[j].add(i)
adj = [sorted(a) for a in adj]
n_edges = sum(len(a) for a in adj) // 2
n_iso = sum(1 for a in adj if not a)
print("INDUCED title-mention graph: edges=%d  isolated_docs=%d/%d  mean_deg=%.2f  median_deg=%d" %
      (n_edges, n_iso, N, st.mean(len(a) for a in adj), st.median([len(a) for a in adj])))

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
    hop = {s: 0 for s in seed_rows}; frontier = set(seed_rows)
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

strata = collections.defaultdict(lambda: {"n": 0, "with_topic": 0, "all_reach": 0,
                                          "gold_total": 0, "gr": collections.Counter()})
for qn in qs:
    golds = [id2idx[g] for g in qn.get("neighbors", []) if g in id2idx]
    if not golds:
        continue
    S = strata[f"{len(golds)}gold"]; S["n"] += 1; S["gold_total"] += len(golds)
    topics = find_topics(qn["content"]); topic_rows = [r for tp in topics for r in t2r.get(tp, [])]
    if topics:
        S["with_topic"] += 1
    reach = bfs_hops(topic_rows, golds) if topic_rows else {g: None for g in golds}
    hs = []
    for g in golds:
        h = reach.get(g)
        S["gr"]["unreach" if h is None else ("topic" if h == 0 else h)] += 1
        hs.append(h if h is not None else 99)
    if topic_rows and max(hs) <= MAXHOP:
        S["all_reach"] += 1

print("=" * 70)
tq = ta = 0
for b in sorted(strata):
    S = strata[b]; n = S["n"]; tq += n; ta += S["all_reach"]; gt = S["gold_total"]
    def pc(k): return "%.0f%%" % (100 * S["gr"].get(k, 0) / gt) if gt else "-"
    print("[%s] n=%d  topic%%=%.0f  gold reach: 0hop=%s 1=%s 2=%s 3=%s 4=%s UNREACH=%s | ALL<=4hop=%d/%d (%.0f%%)" %
          (b, n, 100 * S["with_topic"] / n, pc("topic"), pc(1), pc(2), pc(3), pc(4), pc("unreach"),
           S["all_reach"], n, 100 * S["all_reach"] / n))
print("-" * 70)
print("OVERALL all-golds-reachable: %d/%d (%.1f%%)" % (ta, tq, 100 * ta / tq))
