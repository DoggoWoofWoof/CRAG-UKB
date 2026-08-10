"""
Loader for HippoRAG's released standard datasets (reproduce/dataset/*.json).
============================================================================
corpus.json    = [{"title": str, "text": str}]                       -> the retrieval corpus
questions.json = [{"id","question","answer","answer_aliases",
                   "paragraphs":[{"title","paragraph_text","is_supporting"}]}]  (musique/2wiki/hotpot)
                 (falls back to `supporting_facts` titles if `paragraphs` absent)

Doc nodes = corpus items. GOLD q->doc edges = corpus docs whose (title, text) matches a question's
`is_supporting` paragraphs (exact (title,text) match, title fallback). No doc-doc bridge edges
(label-free; kNN + title-mention edges are added downstream by the indexer / build_clean).
"""
import json

from src.pipeline.standardizer import StandardNode


def load_hipporag(corpus_path, questions_path, source):
    corpus = json.load(open(corpus_path, encoding="utf-8"))
    questions = json.load(open(questions_path, encoding="utf-8"))
    nodes = []
    by_title = {}                                              # title -> first node_id
    by_tt = {}                                                 # (title, text) -> node_id (exact)
    for i, c in enumerate(corpus):
        title = str(c.get("title", "")); text = str(c.get("text", ""))
        nid = f"{source}_doc_{i}"
        nodes.append(StandardNode(nid, text, {"source": source, "type": "document", "title": title}))
        by_title.setdefault(title, nid)
        by_tt.setdefault((title, text), nid)
    id2node = {n.node_id: n for n in nodes}

    zero_gold = 0
    for item in questions:
        qid = str(item.get("id", "unknown"))
        question = item.get("question", "")
        answer = item.get("answer", "") or ""
        aliases = item.get("answer_aliases", []) or []
        answers = list(dict.fromkeys([a for a in [answer, *aliases] if a]))
        gold_ids = []
        paras = item.get("paragraphs", [])
        if paras:
            for p in paras:
                if p.get("is_supporting"):
                    t = str(p.get("title", "")); tx = str(p.get("paragraph_text", ""))
                    nid = by_tt.get((t, tx)) or by_title.get(t)
                    if nid:
                        gold_ids.append(nid)
        else:                                                  # hotpot/2wiki supporting_facts fallback
            sf = item.get("supporting_facts", [])
            titles = sf.get("title", []) if isinstance(sf, dict) else \
                [x[0] for x in sf if isinstance(x, (list, tuple)) and x]
            for t in titles:
                nid = by_title.get(str(t))
                if nid:
                    gold_ids.append(nid)
        gold_ids = list(dict.fromkeys(gold_ids))
        if not gold_ids:
            zero_gold += 1
        q_nid = f"{source}_q_{qid}"
        qn = StandardNode(q_nid, question, {"source": source, "type": "question",
                                            "answer": answer, "answers": answers})
        for g in gold_ids:
            qn.neighbors.append(g)
            id2node[g].neighbors.append(q_nid)
        nodes.append(qn)

    n_docs = sum(1 for n in nodes if n.metadata["type"] == "document")
    print(f"[load_hipporag/{source}] docs={n_docs} questions={len(questions)} zero_gold_q={zero_gold}")
    return nodes
