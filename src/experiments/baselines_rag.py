"""
External RAG baselines harness (#12) — the "different RAGs" for comparison.
==========================================================================
Flat retrieval over the same clean substrate (no partitioning), so our full
L1->L2->L3 system is compared apples-to-apples against what the field uses
(lit_review.md): sparse, dense, learned-sparse, hybrid, late-interaction, and
+cross-encoder reranking. Reports the field-standard metrics: Recall@{2,5,20,100},
nDCG@10, MRR, and FullCov (all-golds; = HippoRAG's headline).

Methods:
  bm25    : engine.search_lexical (Anserini/rank-bm25 index we already build)
  dense   : MiniLM bi-encoder (faiss node index) — batched q @ X.T
  hybrid  : z-normalized bm25 + dense fusion
  splade  : cached SPLADE doc vectors (from splade_edges) + SPLADE query encode  [if --splade]
  +ce     : cross-encoder rerank of the dense top-100                            [if --rerank]
  colbert : engine.search_colbert (skipped if no ColBERT index)
Writes data/ukb_storage/{ds}/results/baselines/flat_retrieval.json.
"""
import os
import json
import time
import math
import logging
import argparse

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.baselines_rag")
KS = [2, 5, 20, 100]
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _metrics(order, gold, ks=KS, ndcg_k=10):
    gs = set(gold)
    if not gs:
        return None
    rec = {k: len(gs & set(order[:k].tolist())) / len(gs) for k in ks}
    fcov = {k: 1.0 if gs <= set(order[:k].tolist()) else 0.0 for k in ks}
    # nDCG@10 (binary relevance)
    dcg = sum(1.0 / math.log2(r + 2) for r, d in enumerate(order[:ndcg_k].tolist()) if d in gs)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(gs), ndcg_k)))
    ndcg = dcg / idcg if idcg else 0.0
    rr = 0.0
    for r, d in enumerate(order.tolist()):
        if d in gs:
            rr = 1.0 / (r + 1); break
    return rec, fcov, ndcg, rr


def _agg(rows, ks=KS):
    recs = {k: [] for k in ks}; fcs = {k: [] for k in ks}; nd = []; mrr = []
    for m in rows:
        if m is None:
            continue
        rec, fcov, ndcg, rr = m
        for k in ks:
            recs[k].append(rec[k]); fcs[k].append(fcov[k])
        nd.append(ndcg); mrr.append(rr)
    return {**{f"recall@{k}": round(np.mean(recs[k]) * 100, 2) for k in ks},
            **{f"fullcov@{k}": round(np.mean(fcs[k]) * 100, 2) for k in ks},
            "ndcg@10": round(np.mean(nd) * 100, 2), "mrr": round(np.mean(mrr) * 100, 2)}


def run(dataset, limit=1000, rerank=False, splade=False, device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    texts = [qn.content for qn, _, _ in test]
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    q = DenseEncoder().encode(texts).astype("float32"); faiss.normalize_L2(q)
    maxk = max(KS)

    out = {"dataset": dataset, "n_docs": X.shape[0], "n_test": len(test), "methods": {}, "latency_ms": {}}

    def timed(fn):
        t0 = time.perf_counter(); orders = fn(); return orders, (time.perf_counter() - t0) / len(test) * 1000

    # dense
    dense_scores = q @ X.T
    def _dense(): return np.argsort(-dense_scores, axis=1)[:, :max(maxk, 100)]
    d_ord, d_ms = timed(_dense)
    out["methods"]["dense"] = _agg([_metrics(d_ord[i], gold[i]) for i in range(len(test))]); out["latency_ms"]["dense"] = round(d_ms, 3)

    # bm25 (per-query via engine.search_lexical)
    bm = []
    t0 = time.perf_counter()
    for qi in range(len(test)):
        nodes = engine.search_lexical(texts[qi], k=maxk)
        bm.append(np.array([id2idx[n.node_id] for n in nodes if n.node_id in id2idx][:maxk] +
                           [-1] * maxk)[:maxk])
    bm_ms = (time.perf_counter() - t0) / len(test) * 1000
    out["methods"]["bm25"] = _agg([_metrics(bm[i], gold[i]) for i in range(len(test))]); out["latency_ms"]["bm25"] = round(bm_ms, 3)

    # hybrid: z-norm dense + z-norm bm25 (bm25 as rank-based score since search_lexical hides scores)
    def _z(v):
        m, s = v.mean(1, keepdims=True), v.std(1, keepdims=True); s[s == 0] = 1; return (v - m) / s
    bm_score = np.full_like(dense_scores, -5.0)
    for qi in range(len(test)):
        for r, d in enumerate(bm[qi]):
            if d >= 0:
                bm_score[qi, d] = maxk - r                      # rank-derived score
    hyb = _z(dense_scores) + _z(bm_score)
    h_ord = np.argsort(-hyb, axis=1)[:, :max(maxk, 100)]
    out["methods"]["hybrid"] = _agg([_metrics(h_ord[i], gold[i]) for i in range(len(test))])

    if rerank:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(CE_MODEL, device=str(device) if device else None)
        contents = [n.content for n in engine.nodes]
        ce_ord = []
        t0 = time.perf_counter()
        for qi in range(len(test)):
            cand = d_ord[qi, :100].tolist()
            sc = ce.predict([(texts[qi], contents[c]) for c in cand], batch_size=64, show_progress_bar=False)
            ce_ord.append(np.array([cand[i] for i in np.argsort(-sc)]))
        ce_ms = (time.perf_counter() - t0) / len(test) * 1000
        out["methods"]["dense+ce"] = _agg([_metrics(ce_ord[i], gold[i]) for i in range(len(test))]); out["latency_ms"]["dense+ce"] = round(ce_ms, 3)

    if splade:
        cache = f"results/research/_splade_vecs_{dataset}.npz"
        if os.path.exists(cache):
            import scipy.sparse as sp
            from src.pipeline.splade_edges import _encode
            z = np.load(cache); D = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            Qs = _encode(texts, "naver/splade-cocondenser-ensembledistil", device=device)
            sc = (Qs @ D.T).toarray()
            s_ord = np.argsort(-sc, axis=1)[:, :max(maxk, 100)]
            out["methods"]["splade"] = _agg([_metrics(s_ord[i], gold[i]) for i in range(len(test))])
        else:
            log.info(f"[{dataset}] no cached SPLADE doc vecs — run splade_edges first")

    with open(rpath(dataset, "baselines", "flat_retrieval"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] " + " | ".join(f"{m}: R@20={v['recall@20']} nDCG={v['ndcg@10']} FCov@20={v['fullcov@20']}"
                                          for m, v in out["methods"].items()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="External flat RAG baselines.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--rerank", action="store_true", help="add cross-encoder rerank of dense top-100")
    p.add_argument("--splade", action="store_true", help="add SPLADE (needs cached doc vecs)")
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== BASELINES: {ds.upper()} =====")
        run(ds, limit=a.limit, rerank=a.rerank, splade=a.splade)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
