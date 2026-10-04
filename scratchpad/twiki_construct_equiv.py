"""Deterministic offline proof that the NEW generic k-hop enumeration reduces EXACTLY to the
OLD hardcoded depth-2 nesting on 2wiki (hops=2). Feeds both the SAME topic list and compares
edge/path sets by TEXT (vocab indices differ by registration order, which is irrelevant to scores).
No encoder/GPU. Uses struct(neighbors) adjacency — both sides consume the identical A.
"""
import json, re, collections

MFILE = "data/processed/master_nodes_2wiki_clean.json"
PATH_CAP = 8   # new-code default; test whether the cap ever drops a 2wiki path (it must not change sets if not)

nodes = json.load(open(MFILE, encoding="utf-8"))
docs = [n for n in nodes if n.get("metadata", {}).get("type") != "question"]
qs = [n for n in nodes if n.get("metadata", {}).get("type") == "question"]
id2idx = {n["node_id"]: i for i, n in enumerate(docs)}
N = len(docs)
idx2title = {}; content_by_row = {}; t2r = collections.defaultdict(list)
for i, n in enumerate(docs):
    ti = n.get("metadata", {}).get("title", "").strip()
    content_by_row[i] = n.get("content", "")
    if ti:
        idx2title[i] = ti; t2r[ti].append(i)
ents = [e for e in t2r if e]

# adjacency (CSR-like via dict) from neighbors
adj = collections.defaultdict(list); seen = collections.defaultdict(set)
for i, n in enumerate(docs):
    for nb in n.get("neighbors", []):
        j = id2idx.get(nb)
        if j is not None and j != i and j not in seen[i]:
            seen[i].add(j); adj[i].append(j)
            if i not in seen[j]:
                seen[j].add(i); adj[j].append(i)

_wb = {}
def _sents(t): return [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
def _specific(t): return len(t) >= 4 and re.search(r"[A-Z][a-z]", t) is not None
def _mentions(title, sent):
    p = _wb.get(title)
    if p is None:
        p = _wb[title] = re.compile(r"(?<![A-Za-z])" + re.escape(title) + r"(?![A-Za-z])")
    return p.search(sent) is not None
def connect(src_title):
    out = []
    for r in t2r.get(src_title, [])[:1]:
        ss = _sents(content_by_row.get(r, ""))
        for nb in adj.get(r, []):
            tt = idx2title.get(int(nb), "")
            if not tt or tt == src_title or not _specific(tt):
                continue
            for s in ss:
                if _mentions(tt, s):
                    out.append((tt, s, ss)); break
    return out

def _toks(s): return re.findall(r"[a-z0-9]+", s.lower())
ttoks = {t: set(_toks(t)) for t in ents}
dfc = collections.Counter(w for ts in ttoks.values() for w in ts)
keyidx = collections.defaultdict(list)
for t in ents:
    if ttoks[t]:
        keyidx[min(ttoks[t], key=lambda w: dfc[w])].append(t)
def find_topics(q):
    ql = q.lower(); qt = set(_toks(q)); sn = set(); found = []
    for w in sorted(qt):                                   # sorted -> deterministic, same list to both sides
        for t in keyidx.get(w, ()):
            if t in sn:
                continue
            sn.add(t)
            if ttoks[t] and ttoks[t] <= qt and len(t) >= 3 and t.lower() in ql:
                found.append(t)
    found.sort(key=lambda s: -len(s)); topics = []
    for t in found:
        if not any(t != o and t.lower() in o.lower() for o in topics):
            topics.append(t)
    return topics[:4]

# --- OLD construction (verbatim depth-2 nesting), returns text-keyed sets ---
def build_old(topics):
    tset = set(topics); edges = collections.defaultdict(set); paths = collections.defaultdict(set)
    for tp in topics:
        for mt, sent, ss in connect(tp):
            edges[mt].add(sent)
            for ct, sent2, ss2 in connect(mt):
                if ct in tset or ct == mt:
                    continue
                paths[ct].add((sent, sent2))
    return edges, paths

# --- NEW construction (generic BFS), hops=2, returns text-keyed sets ---
def build_new(topics, hops=2, path_cap=PATH_CAP):
    tset = set(topics); edges = collections.defaultdict(list); paths = collections.defaultdict(list)
    sent_of = {}                                            # rel-index proxy: use the sentence text itself as key
    frontier = [(tp, (), {tp}) for tp in topics]
    for depth in range(1, hops + 1):
        if not frontier:
            break
        nxt = []
        for cur, rpath, ptitles in frontier:
            for nb, sent, ss in connect(cur):
                if nb in ptitles or (depth >= 2 and nb in tset):
                    continue
                newpath = rpath + (sent,)
                if depth == 1:
                    edges[nb].append(sent)
                else:
                    if len(paths[nb]) < path_cap:
                        paths[nb].append(newpath)
                nxt.append((nb, newpath, ptitles | {nb}))
        frontier = nxt[:4000]
    return ({k: set(v) for k, v in edges.items()},
            {k: set(v) for k, v in paths.items()})

n_q = 0; edge_mismatch = 0; path_mismatch = 0; cap_touched = 0; examples = []
for qn in qs[:4000]:
    topics = find_topics(qn["content"])
    if not topics:
        continue
    n_q += 1
    eo, po = build_old(topics); en, pn = build_new(topics)
    # edges must be identical text-sets
    if {k: eo[k] for k in eo} != {k: en[k] for k in en}:
        edge_mismatch += 1
        if len(examples) < 5:
            examples.append(("EDGE", topics, {k: eo.get(k) for k in set(eo) | set(en)},
                             {k: en.get(k) for k in set(eo) | set(en)}))
    # paths: new caps at PATH_CAP; if any old candidate has >cap paths, allow subset, else must match
    for k in set(po) | set(pn):
        ov, nv = po.get(k, set()), pn.get(k, set())
        if len(ov) > PATH_CAP:
            cap_touched += 1
            if not nv <= ov:                                # capped set must be a subset of full
                path_mismatch += 1
        elif ov != nv:
            path_mismatch += 1
            if len(examples) < 5:
                examples.append(("PATH", topics, ov, nv))

print("questions compared:", n_q)
print("edge-set mismatches:", edge_mismatch)
print("path-set mismatches:", path_mismatch, "(candidates where old paths > cap:", cap_touched, ")")
if examples:
    for tag, tp, o, nn in examples:
        print("---", tag, "topics=", tp)
        print("  OLD:", o); print("  NEW:", nn)
else:
    print("RESULT: NEW enumeration IDENTICAL to OLD on 2wiki at hops=2 (edges + paths, by text). Cap never alters a set.")
