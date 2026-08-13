"""NER 'shares-entity' edges as a UKB edge type (spaCy en_core_web_sm — NO LLM).

Two docs are linked if they share a named entity; the edge weight is Σ 1/entity_document_frequency, so
rare-entity BRIDGES dominate common-entity HUBS. This is the relational, multi-hop edge that synthetic
kNN edges are not (kNN just re-expresses dense similarity). Ablation: NER ≫ kNN everywhere, kNN redundant
once NER is in (results/L2/ner_edge_ablation.json). Edges are cached per dataset.
"""
import os
import pickle
import logging
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)
_NLP = None
ENT_LABELS = {"PERSON", "GPE", "ORG", "LOC", "FAC", "WORK_OF_ART", "EVENT", "PRODUCT", "NORP"}


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _NLP


def build_ner_edges(dataset, texts, n, maxdf=25, weighted=True, cache_dir="data/ukb_storage"):
    """Return an (n, n) CSR adjacency of NER shares-entity edges (weighted 1/df). Cached per dataset."""
    tag = f"ner_edges_{'w' if weighted else 'b'}_df{maxdf}.pkl"
    cache = os.path.join(cache_dir, dataset, tag) if dataset else None
    if cache and os.path.exists(cache):
        try:
            A = pickle.load(open(cache, "rb"))
            if A.shape[0] == n:
                log.info("[ner-edges] reused %s (%d edges)", os.path.basename(cache), A.nnz // 2)
                return A
        except Exception as e:
            log.warning("[ner-edges] cache load failed (%s); rebuilding", e)

    ent_docs = defaultdict(set)
    for i, doc in enumerate(_nlp().pipe([str(t) for t in texts], batch_size=256)):
        for e in doc.ents:
            if e.label_ in ENT_LABELS and len(e.text) > 2:
                ent_docs[e.text.lower().strip()].add(i)
    w = defaultdict(float)
    for _, ds in ent_docs.items():
        df = len(ds)
        if 2 <= df <= maxdf:                                        # skip singletons and ubiquitous hubs
            wt = (1.0 / df) if weighted else 1.0
            ds = sorted(ds)
            for a in range(len(ds)):
                for b in range(a + 1, len(ds)):
                    w[(ds[a], ds[b])] += wt
    r, c, v = [], [], []
    for (a, b), val in w.items():
        r += [a, b]; c += [b, a]; v += [val, val]
    A = (sp.csr_matrix((v, (r, c)), shape=(n, n), dtype=np.float32) if r
         else sp.csr_matrix((n, n), dtype=np.float32))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(A, open(cache, "wb"))
        log.info("[ner-edges] built + cached %s (%d edges)", os.path.basename(cache), len(w))
    return A
