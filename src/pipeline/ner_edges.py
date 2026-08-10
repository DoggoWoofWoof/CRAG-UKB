"""
NER / entity co-occurrence edges (STRUCTURAL role in the UKB).
=============================================================
Generalizes the title-mention edge: link two docs when they share a salient
named entity, not just when one's title appears in the other. This is the
content-derived relational structure L1/L3 need on entity-heavy corpora
(the title fix rescued musique; this is the general version).

Extraction is dependency-free by default (proper-noun / Capitalized multi-word
phrases, IDF-filtered so stopword-entities like "The"/"United States" don't
over-connect), and uses spaCy NER if it happens to be installed. Edges are
IDF-weighted and capped per doc so the graph stays sparse. Output is a
doc_id -> [neighbor_ids] JSON that l1l3_recall.py --extra_edges consumes for the
pool-matched A/B, and --bake writes them into the clean master as node.neighbors
(structural, exactly like _title_links) for a substrate rebuild.

Roles reminder (UKB): NER edges are STRUCTURAL (node.neighbors), folded into METIS
alongside title edges; kNN/SPLADE stay SYNTHETIC. Never links question nodes.
"""
import os
import re
import json
import math
import logging
import argparse
from collections import defaultdict, Counter

from src.core.engine import CoreEngine

log = logging.getLogger("pipeline.ner_edges")

# tokens that are Capitalized but not entities (sentence-initial / function words)
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "but",
         "this", "that", "these", "those", "it", "he", "she", "they", "we", "you",
         "his", "her", "their", "its", "was", "were", "is", "are", "been", "has",
         "have", "had", "by", "as", "with", "from", "after", "before", "when",
         "while", "during", "who", "what", "which", "where", "how", "also"}
_CONNECT = {"of", "the", "and", "for", "de", "van", "von", "da", "di"}
_PHRASE = re.compile(r"\b([A-Z][a-zA-Z0-9.'-]*(?:\s+(?:%s|[A-Z][a-zA-Z0-9.'-]*))*)"
                     % "|".join(_CONNECT))


def _spacy_nlp():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
        except Exception:
            return None
    except ImportError:
        return None


def _entities_regex(text):
    ents = set()
    for m in _PHRASE.finditer(text):
        phrase = m.group(1).strip()
        toks = phrase.split()
        # drop single sentence-initial common words / pure function-word phrases
        core = [t for t in toks if t.lower() not in _STOP]
        if not core:
            continue
        if len(toks) == 1 and toks[0].lower() in _STOP:
            continue
        ent = " ".join(toks).lower()
        if len(ent) >= 3:
            ents.add(ent)
    return ents


def _entities_spacy(nlp, texts, batch=256):
    keep = {"PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "EVENT", "FAC", "NORP", "PRODUCT"}
    out = []
    for doc in nlp.pipe(texts, batch_size=batch):
        out.append({e.text.lower().strip() for e in doc.ents
                    if e.label_ in keep and len(e.text.strip()) >= 3})
    return out


def build_edges(dataset, max_edges_per_doc=16, min_idf=1.5, use_spacy=True):
    engine = CoreEngine(source=dataset)
    docs = engine.nodes
    texts = [d.content for d in docs]
    ids = [d.node_id for d in docs]
    N = len(docs)

    nlp = _spacy_nlp() if use_spacy else None
    log.info(f"[{dataset}] extracting entities from {N} docs ({'spaCy' if nlp else 'regex proper-nouns'})")
    ent_sets = _entities_spacy(nlp, texts) if nlp else [_entities_regex(t) for t in texts]

    # document frequency -> IDF; drop over-common entities (stopword-like)
    df = Counter()
    for es in ent_sets:
        df.update(es)
    idf = {e: math.log(N / c) for e, c in df.items()}
    inv = defaultdict(list)                                  # entity -> doc indices (informative only)
    for i, es in enumerate(ent_sets):
        for e in es:
            if idf.get(e, 0) >= min_idf and df[e] >= 2:
                inv[e].append(i)

    # score candidate pairs by summed IDF of shared entities; keep top-K per doc
    pair_w = defaultdict(float)
    for e, members in inv.items():
        if len(members) > 200:                              # skip hub entities (still too common)
            continue
        w = idf[e]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pair_w[(members[a], members[b])] += w
    per_doc = defaultdict(list)
    for (i, j), w in pair_w.items():
        per_doc[i].append((w, j)); per_doc[j].append((w, i))

    edges = defaultdict(set)
    n_dir = 0
    for i, cand in per_doc.items():
        cand.sort(reverse=True)
        for w, j in cand[:max_edges_per_doc]:
            edges[ids[i]].add(ids[j]); edges[ids[j]].add(ids[i]); n_dir += 2
    edges = {k: sorted(v) for k, v in edges.items()}
    deg = (sum(len(v) for v in edges.values()) / N) if N else 0
    log.info(f"[{dataset}] {len(df)} entities, {n_dir} directed edges, avg doc degree {deg:.2f} "
             f"({len(edges)/N*100:.0f}% docs linked)")
    return edges


def run(dataset, out=None, bake=False, **kw):
    edges = build_edges(dataset, **kw)
    out = out or f"results/research/ner_edges_{dataset}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(edges, f)
    log.info(f"[{dataset}] wrote {out}")
    if bake:
        master = f"data/processed/master_nodes_{dataset}.json"
        nodes = json.load(open(master, encoding="utf-8"))
        for nd in nodes:
            if nd["metadata"].get("type") != "question" and nd["node_id"] in edges:
                nb = set(nd.get("neighbors", [])) | set(edges[nd["node_id"]])
                nd["neighbors"] = sorted(nb)
        with open(master, "w", encoding="utf-8") as f:
            json.dump(nodes, f)
        log.info(f"[{dataset}] BAKED NER edges into {master} (rebuild index next)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="NER / entity co-occurrence edges (structural).")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--max_edges_per_doc", type=int, default=16)
    p.add_argument("--min_idf", type=float, default=1.5)
    p.add_argument("--no_spacy", action="store_true")
    p.add_argument("--bake", action="store_true")
    a = p.parse_args(argv)
    for ds in a.datasets:
        run(ds, bake=a.bake, max_edges_per_doc=a.max_edges_per_doc,
            min_idf=a.min_idf, use_spacy=not a.no_spacy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
