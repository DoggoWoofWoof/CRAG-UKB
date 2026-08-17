"""
Graph-native loader for RoG-webqsp (real-Freebase KGQA) as a KB dataset.
========================================================================
Real Freebase KGQA, ingested into the same text-graph substrate as the other datasets so ONE
system handles it (the KB paradigm, done properly):

  entities  -> document nodes, content = entity verbalized with its incident relations (<=max_rels)
  triples   -> the doc GRAPH (node.neighbors), ALL edges kept (Freebase hubs bounded at traversal time)
  questions -> question nodes, golds = the answer entities (a_entity)

Bypasses build_clean (no text-dedup / title-link rebuild) so the KG triple edges ARE the doc graph;
build_all adds gte-kNN on top, encoder-graph preserves node.neighbors in gte space. Writes
data/processed/master_nodes_{source}.json directly.
"""
import json
import logging
from collections import defaultdict, Counter

from src.pipeline.standardizer import StandardNode, save_nodes

log = logging.getLogger("pipeline.loader_webqsp")


def load_webqsp(parquet_paths, source, max_questions=None, max_rels=15, degree_cap=None):
    import re
    import pandas as pd
    df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    if max_questions:
        df = df.iloc[:max_questions].reset_index(drop=True)

    # The GRAPH is kept exactly as-is (all nodes + all edges). We ONLY clean the encoded TEXT: raw Freebase
    # MIDs (m.0abc / g.1xyz — unlabeled CVT/mediator plumbing) are noise the encoder can't use, so we DEDUCE
    # each MID to a real name via the graph (its named neighbour) and replace it. Zero IDs left in any content.
    idpat = re.compile(r'^[mg]\.[0-9a-z_]+$')
    is_mid = lambda x: bool(idpat.match(x))

    triples = set()
    for g in df["graph"]:
        for t in g:
            triples.add((str(t[0]), str(t[1]), str(t[2])))

    adj = defaultdict(list)                                    # undirected relation-adjacency for BOTH edges + deducing
    ents = set()
    for h, rel, tl in triples:
        adj[h].append((rel, tl)); adj[tl].append((rel, h)); ents.add(h); ents.add(tl)
    entities = sorted(ents)                                    # ALL entities kept as nodes (graph unchanged)
    ent2nid = {e: f"{source}_doc_{i}" for i, e in enumerate(entities)}

    def _rel(r):                                               # freebase 'a.b.official_language' -> 'official language'
        return r.split(".")[-1].replace("_", " ")

    named_of = {}                                              # MID -> a named neighbour (deduced), cached
    def deduce(x):                                             # return a NAME for x (itself if named; else a named neighbour)
        if not is_mid(x):
            return x
        if x in named_of:
            return named_of[x]
        nm = next((o for _, o in adj.get(x, []) if not is_mid(o)), None)
        named_of[x] = nm
        return nm

    nodes = []
    for e in entities:
        rels, seen = [], set()
        for rel, o in adj.get(e, []):
            ro = deduce(o)                                     # resolve MID targets to names
            if ro and ro != e and ro not in seen:
                rels.append((rel, ro)); seen.add(ro)
            if len(rels) >= max_rels:
                break
        name = e if not is_mid(e) else (deduce(e) or "")       # MID node: use a deduced name, never the raw id
        body = "; ".join(f"{_rel(r)} {o}" for r, o in rels)
        content = (f"{name}. {body}" if name and body else (name or body)) or f"entity {ent2nid[e]}"
        nodes.append(StandardNode(ent2nid[e], content,
                                  {"source": source, "type": "document", "title": (e if not is_mid(e) else name)}))
    id2node = {n.node_id: n for n in nodes}

    # KG edges — UNCHANGED: every original triple, exactly as before (do not modify the graph).
    deg = Counter()
    n_edges = 0
    for h, rel, tl in triples:
        if h in ent2nid and tl in ent2nid and (degree_cap is None or (deg[h] < degree_cap and deg[tl] < degree_cap)):
            id2node[ent2nid[h]].neighbors.append(ent2nid[tl])
            id2node[ent2nid[tl]].neighbors.append(ent2nid[h])
            deg[h] += 1; deg[tl] += 1; n_edges += 1

    zero = 0
    for _, r in df.iterrows():
        qid = str(r["id"]); question = str(r["question"])
        ans = list(r["answer"]) if r["answer"] is not None else []
        answers = [str(a) for a in ans]
        gold_ids = [ent2nid[str(a)] for a in r["a_entity"] if str(a) in ent2nid]
        if not gold_ids:
            zero += 1
        q_nid = f"{source}_q_{qid}"
        qn = StandardNode(q_nid, question, {"source": source, "type": "question",
                                            "answer": answers[0] if answers else "", "answers": answers})
        for g in gold_ids:
            qn.neighbors.append(g); id2node[g].neighbors.append(q_nid)
        nodes.append(qn)

    print(f"[load_webqsp/{source}] entities={len(entities)} triples={len(triples)} kg_edges={n_edges} "
          f"questions={len(df)} zero_gold_q={zero}")
    return nodes


def build_master(parquet_paths, source="webqsp", max_questions=None, out=None):
    nodes = load_webqsp(parquet_paths, source, max_questions=max_questions)
    out = out or f"data/processed/master_nodes_{source}.json"
    save_nodes(nodes, out)
    print(f"  -> {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="+", default=["data/raw/full/kb/webqsp_test0.parquet",
                                                  "data/raw/full/kb/webqsp_test1.parquet"])
    p.add_argument("--source", default="webqsp")
    p.add_argument("--max-questions", type=int, default=None)
    a = p.parse_args()
    build_master(a.paths, a.source, a.max_questions)
