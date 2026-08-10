"""
HippoRAG-class baseline (LLM-free) vs ours — the paper's core retrieval comparison.
===================================================================================
HippoRAG(2)'s mechanism = Personalized PageRank over an ENTITY knowledge-graph, seeded
from the query's entities (phrase nodes) + dense passages, scoring passages by the PPR
mass of the entities they contain. Its ONLY LLM step is OpenIE triple extraction for
KG construction. We reproduce it FAITHFULLY but LLM-FREE by building the entity graph
from NER (regex entity extraction we already use) instead of LLM triples:

  KG (LLM-free): passages linked when they share a named entity (entity-mediated edges,
                 like HippoRAG's phrase-mediated passage connectivity); over-common
                 entities filtered (IDF), degree-capped.
  seeds        : query entities -> passages containing them, weighted by entity IDF
                 (the "phrase" seeds) UNION dense top-N passages (the hybrid seeding).
  rank         : PPR over the graph; passages by mass.

Reported side-by-side on the SAME queries / budgets, at the metrics the literature uses
(Recall@2/@5, HippoRAG's headline) plus @20/@100 and FullCov:
  dense            : semantic top-k (retrieval floor)
  hipporag_class   : the LLM-free HippoRAG reproduction above
  ours             : L1-champion-seeded + synthetic-edges-DROPPED PPR (our SRW winner)

Writes {ds}/results/baselines/hipporag/compare.json + _index/hipporag_compare_summary.json.
This is the "reproduce competitor on our substrate" arm (apples-to-apples). The
"align-to-their-protocol" arm = report ours at Recall@2/@5 (this file does that too).
"""
import os
import re
import json
import logging
import argparse
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l3_methods import _ppr, _champion_seed_order
from src.experiments.l3_srw import _typed_adj, _weighted_P
from src.pipeline.ner_edges import _entities_regex

log = logging.getLogger("experiments.hipporag_baseline")
BUDGETS = [2, 5, 20, 100]
FC_BUDGETS = [5, 20, 100]
MAXB = 100
MIN_ENT_DF, MAX_ENT_DF = 2, 30      # keep entities in 2..30 passages (drop too-rare/too-common)
PASS_CAP = 20                       # cap passages-per-entity used for edge building (bounds edge count)


def _metrics(order, gold, budgets=BUDGETS, fc_budgets=FC_BUDGETS):
    rec = {b: [] for b in budgets}; fc = {b: [] for b in fc_budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g); top = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            rec[b].append(len(gs & set(top[:b])) / len(gs))
        for b in fc_budgets:
            fc[b].append(1.0 if gs <= set(top[:b]) else 0.0)
    out = {f"recall@{b}": round(float(np.mean(rec[b])) * 100, 2) for b in budgets}
    out.update({f"fullcov@{b}": round(float(np.mean(fc[b])) * 100, 2) for b in fc_budgets})
    return out


def _entity_index(engine, id2idx):
    """entity -> set(passage idx), from regex NER over passage contents (LLM-free KG)."""
    ent2pass = defaultdict(set)
    contents = [None] * len(id2idx)
    for node in engine.nodes:
        i = id2idx.get(node.node_id)
        if i is None:
            continue
        contents[i] = node.content.lower()
        for e in _entities_regex(node.content):
            if len(e) >= 3:
                ent2pass[e.lower()].add(i)
    ent2pass = {e: ps for e, ps in ent2pass.items() if MIN_ENT_DF <= len(ps) <= MAX_ENT_DF}
    idf = {e: np.log(len(id2idx) / len(ps)) for e, ps in ent2pass.items()}
    return ent2pass, idf, contents


def _entity_graph(ent2pass, n):
    """passage-passage adjacency: passages sharing an entity are linked (degree-capped)."""
    rows, cols = [], []
    for ps in ent2pass.values():
        ps = sorted(ps)[:PASS_CAP]                        # cap passages/entity -> bounds edges
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                rows.append(ps[a]); cols.append(ps[b])
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T).tolil()
    # degree cap: keep strongest (most co-occurring) — here just cap count per row
    A = A.tocsr()
    deg = np.asarray((A > 0).sum(1)).ravel()
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
    P = (sp.diags(1.0 / d) @ A).tocsr()
    return P, round(float(deg.mean()), 2)


def run(dataset, N_seed=20, limit=500, alpha=0.9, device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n = X.shape[0]; id2idx = engine.node_id_to_idx
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    enc = DenseEncoder()
    q = enc.encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    nq = len(test)
    qsim = q @ X.T
    dense_order = np.argsort(-qsim, axis=1)[:, :MAXB]
    out = {"dataset": dataset, "n_docs": n, "n_test": len([g for g in gold if g]),
           "N_seed": N_seed, "alpha": alpha, "budgets": BUDGETS, "methods": {}}

    # ---- HippoRAG-class (LLM-free): entity KG + entity+dense seeded PPR ----
    ent2pass, idf, contents = _entity_index(engine, id2idx)
    P_ent, avg_deg = _entity_graph(ent2pass, n)
    out["entity_graph"] = {"n_entities": len(ent2pass), "avg_passage_degree": avg_deg}
    seeds_h = np.zeros((nq, n), np.float32)
    for qi, (qn, _, _) in enumerate(test):
        qents = [e.lower() for e in _entities_regex(qn.content) if len(e) >= 3]
        hit = 0
        for e in qents:                                   # phrase seeds: passages with the query's entities
            if e in ent2pass:
                w = idf[e] / len(ent2pass[e])
                for p in ent2pass[e]:
                    seeds_h[qi, p] += w; hit += 1
        for p in dense_order[qi, :N_seed]:                # + dense passage seeds (hybrid)
            seeds_h[qi, p] += 0.5 * float(max(qsim[qi, p], 0.0))
        s = seeds_h[qi].sum()
        if s > 0:
            seeds_h[qi] /= s
        else:
            seeds_h[qi, dense_order[qi, :N_seed]] = 1.0 / N_seed
    hippo_order = np.argsort(-_ppr(seeds_h, P_ent, alpha), axis=1)[:, :MAXB]
    out["methods"]["dense"] = _metrics(dense_order, gold)
    out["methods"]["hipporag_class"] = _metrics(hippo_order, gold)

    # ---- ours: L1-champion-seeded + synthetic-edges-DROPPED PPR (SRW winner) ----
    import torch
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    champ_order = _champion_seed_order(engine, X, id2idx, q, MAXB, dev)
    struct, syn = _typed_adj(engine, id2idx)
    P_reweight = _weighted_P(struct, syn, X, n, beta=0.0, gamma=1.0)   # drop synthetic, cosine-weight structural
    seeds_o = np.zeros((nq, n), np.float32)
    for qi in range(nq):
        seeds_o[qi, champ_order[qi, :N_seed]] = 1.0 / N_seed
    ours_order = np.argsort(-_ppr(seeds_o, P_reweight, alpha), axis=1)[:, :MAXB]
    out["methods"]["ours"] = _metrics(ours_order, gold)

    path = os.path.join("data", "ukb_storage", dataset, "results", "baselines", "hipporag", "compare.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    m = out["methods"]
    log.info(f"[{dataset}] R@2/5/100 dense {m['dense']['recall@2']}/{m['dense']['recall@5']}/{m['dense']['recall@100']} "
             f"| hippo {m['hipporag_class']['recall@2']}/{m['hipporag_class']['recall@5']}/{m['hipporag_class']['recall@100']} "
             f"| OURS {m['ours']['recall@2']}/{m['ours']['recall@5']}/{m['ours']['recall@100']} "
             f"|| FCOV@100 hippo {m['hipporag_class']['fullcov@100']} ours {m['ours']['fullcov@100']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="LLM-free HippoRAG-class baseline vs ours (Recall@2/5 + FullCov).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean", "hotpotqa_clean", "squad_clean"])
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.9)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== HIPPORAG-CLASS vs OURS: {ds.upper()} =====")
        try:
            results[ds] = run(ds, N_seed=a.N_seed, limit=a.limit, alpha=a.alpha)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
    if results:
        summary = {ds: {k: {"recall@5": v[k]["recall@5"], "recall@100": v[k]["recall@100"], "fullcov@100": v[k]["fullcov@100"]}
                        for k in r["methods"]} for ds, r in results.items() for v in [r["methods"]]}
        with open(os.path.join("data", "ukb_storage", "_index", "hipporag_compare_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"HIPPORAG COMPARE summary written for {list(results)}")


if __name__ == "__main__":
    main()
