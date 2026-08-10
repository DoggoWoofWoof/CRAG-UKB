"""
SPLADE-kNN edges (SYNTHETIC role in the UKB).
=============================================
SPLADE (learned sparse; > BM25) captures exact-term + expansion matches that dense
kNN misses. These are SYNTHETIC edges (same partition-jumping role as the dense-kNN
edges the indexer adds) — an orthogonal semantic bridge, NOT structural. Output is
a doc_id -> [neighbor_ids] JSON for l1l3_recall.py --extra_edges (pool-matched A/B);
keep it only if it moves the frontier.

Pipeline: SPLADE-encode every doc -> sparse term-weight vectors (top-T terms kept)
-> blocked sparse kNN (top-k neighbors by dot product) -> edge file. CPU-encoding a
large corpus is slow — use --limit_docs to prototype, and it caches the sparse
vectors so kNN can be retuned without re-encoding.

Needs transformers (already present via sentence-transformers). Model defaults to
naver/splade-cocondenser-ensembledistil (downloaded on first run).
"""
import os
import json
import logging
import argparse

import numpy as np
import scipy.sparse as sp

from src.core.engine import CoreEngine

log = logging.getLogger("pipeline.splade_edges")

MODEL = "naver/splade-cocondenser-ensembledistil"


def _encode(texts, model_name, top_t=128, batch=16, device=None, max_len=256):
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    V = model.config.vocab_size
    rows, cols, vals = [], [], []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits                                  # (B,L,V)
            w = torch.log1p(torch.relu(logits))
            mask = enc["attention_mask"].unsqueeze(-1)                    # (B,L,1)
            vec = (w * mask).max(dim=1).values                           # (B,V) SPLADE max-pool
            for i in range(vec.shape[0]):
                v = vec[i]
                nz = torch.topk(v, min(top_t, V)).indices
                nzv = v[nz]
                keep = nzv > 0
                nz, nzv = nz[keep].cpu().numpy(), nzv[keep].cpu().numpy()
                r = s + i
                rows.extend([r] * len(nz)); cols.extend(nz.tolist()); vals.extend(nzv.tolist())
            if s % (batch * 50) == 0:
                log.info(f"    SPLADE-encoded {min(s + batch, len(texts))}/{len(texts)}")
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), V), dtype=np.float32)
    # L2-normalize rows so dot product ~ cosine
    norm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); norm[norm == 0] = 1
    return sp.diags(1.0 / norm) @ X


def _knn(X, k=3, block=512):
    n = X.shape[0]
    Xt = X.T.tocsr()
    nbrs = []
    for s in range(0, n, block):
        sims = (X[s:s + block] @ Xt).toarray()                          # (b, n)
        for i in range(sims.shape[0]):
            sims[i, s + i] = -1                                          # drop self
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for i in range(sims.shape[0]):
            top = idx[i][np.argsort(-sims[i, idx[i]])]
            nbrs.append([int(j) for j in top if sims[i, j] > 0])
    return nbrs


def run(dataset, k=3, top_t=128, limit_docs=0, model_name=MODEL, device=None):
    engine = CoreEngine(source=dataset)
    docs = engine.nodes
    if limit_docs:
        docs = docs[:limit_docs]
    ids = [d.node_id for d in docs]
    cache = f"results/research/_splade_vecs_{dataset}.npz"
    if os.path.exists(cache):
        z = np.load(cache); X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        log.info(f"[{dataset}] loaded cached SPLADE vecs {X.shape}")
    else:
        X = _encode([d.content for d in docs], model_name, top_t=top_t, device=device)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape)
        log.info(f"[{dataset}] encoded + cached SPLADE vecs {X.shape} -> {cache}")

    nbrs = _knn(X, k=k)
    edges = {ids[i]: [ids[j] for j in nb] for i, nb in enumerate(nbrs) if nb}
    deg = sum(len(v) for v in edges.values()) / max(len(ids), 1)
    out = f"results/research/splade_edges_{dataset}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(edges, f)
    log.info(f"[{dataset}] SPLADE-kNN k={k}: avg doc degree {deg:.2f}; wrote {out}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="SPLADE-kNN synthetic edges.")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--top_t", type=int, default=128)
    p.add_argument("--limit_docs", type=int, default=0, help="prototype on a subset")
    p.add_argument("--model", default=MODEL)
    a = p.parse_args(argv)
    for ds in a.datasets:
        run(ds, k=a.k, top_t=a.top_t, limit_docs=a.limit_docs, model_name=a.model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
