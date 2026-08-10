"""
Query-conditioned relational hop (phase 21, the right formulation).
===================================================================
The query names BOTH endpoints: the start entity (dense-findable, e.g. the actor)
and the target relation ("movies"). So learn a query-conditioned offset g(q) and
apply it AT THE DOC LEVEL from the semantic seed:
    start  = dense top-1 doc for q       (the actor doc — dense finds it)
    answer ~ start_vec + g(q)            (g(q) = the relation the query implies)
g is a tiny head trained on (query, start, gold) so start+g(q) lands on the gold.
This differs from the failed attempts: doc-space application (offsets validated
there), query-CONDITIONED relation (not all-K, not generic), trained to hit golds.

Reports recall/FullCov of: dense (baseline) vs offset-hop (start+g(q)) vs combined
(dense ∪ offset-hop). If offset-hop reaches golds dense misses (esp metaqa), the
king->queen idea finally pays off — at the level the query structure supports.
Writes data/ukb_storage/{ds}/results/L1/query_relation.json.
"""
import os
import json
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, INIT_SEED
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.query_relation")
KS = [5, 20, 50, 100, 200]


class OffsetHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, d))

    def forward(self, qn, seed):                            # qn, seed: (B, d) normalized
        return F.normalize(seed + self.net(qn), dim=-1)     # predicted answer position


def _recall(order, gold):
    out = {k: [] for k in KS}; fc = {k: [] for k in KS}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        for k in KS:
            hit = len(gs & set(order[qi][:k].tolist()))
            out[k].append(hit / len(gs)); fc[k].append(1.0 if hit == len(gs) else 0.0)
    return ({f"recall@{k}": round(np.mean(out[k]) * 100, 2) for k in KS},
            {f"fullcov@{k}": round(np.mean(fc[k]) * 100, 2) for k in KS})


def _seeds_and_q(engine_qs, X, encoder, device):
    q = encoder.encode([qn.content for qn, _, _ in engine_qs]).astype("float32"); faiss.normalize_L2(q)
    seed1 = np.argmax(q @ X.T, axis=1)                      # dense top-1 doc per query
    return q, seed1


def run(dataset, epochs=30, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    d = X.shape[1]; index = faiss.IndexFlatIP(d); index.add(X)
    id2idx = engine.node_id_to_idx
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}
    Xt = torch.tensor(X, device=device)
    data = {}
    for sp in ("train", "val", "test"):
        q, seed1 = _seeds_and_q(splits[sp], X, encoder, device)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in splits[sp]]
        data[sp] = (q, seed1, gold)

    # training triples (query, start-seed, one gold)
    trip = [(i, data["train"][1][i], g) for i, gl in enumerate(data["train"][2]) for g in gl]
    qtr = torch.tensor(data["train"][0], device=device)
    torch.manual_seed(INIT_SEED)
    model = OffsetHead(d).to(device); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 256
    for ep in range(epochs):
        model.train(); random.Random(ep).shuffle(trip)
        for s in range(0, len(trip), bs):
            b = trip[s:s + bs]
            qn = qtr[[t[0] for t in b]]
            seed = Xt[[int(t[1]) for t in b]]
            goldv = Xt[[int(t[2]) for t in b]]
            pred = model(qn, seed)
            sim = pred @ goldv.T / 0.05                     # in-batch InfoNCE, gold as positive
            loss = F.cross_entropy(sim, torch.arange(len(b), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    # eval on test: dense vs offset-hop vs combined
    q_te, seed_te, gold = data["test"]
    maxk = max(KS)
    dense_order = np.argsort(-(q_te @ X.T), axis=1)[:, :maxk]
    with torch.no_grad():
        pred = model(torch.tensor(q_te, device=device), Xt[[int(s) for s in seed_te]]).cpu().numpy()
    _, off_order = index.search(pred.astype("float32"), maxk)
    comb = np.array([list(dict.fromkeys(list(off_order[i][:maxk // 2]) + list(dense_order[i])))[:maxk]
                     for i in range(len(gold))])
    d_r, d_f = _recall(dense_order, gold)
    o_r, o_f = _recall(off_order, gold)
    c_r, c_f = _recall(comb, gold)
    out = {"dataset": dataset, "n_test": len(gold), "n_train_triples": len(trip),
           "dense_recall": d_r, "offset_recall": o_r, "combined_recall": c_r,
           "dense_fullcov": d_f, "offset_fullcov": o_f, "combined_fullcov": c_f,
           "combined_over_dense@100": round(c_r["recall@100"] - d_r["recall@100"], 2)}
    with open(rpath(dataset, "L1", "query_relation"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] recall@100 dense {d_r['recall@100']} | offset {o_r['recall@100']} | "
             f"combined {c_r['recall@100']} (+{out['combined_over_dense@100']})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Query-conditioned relational hop (king->queen, done right).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "2wiki_clean"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== QUERY-CONDITIONED RELATIONAL HOP: {ds.upper()} =====")
        run(ds, epochs=a.epochs, limit=a.limit)


if __name__ == "__main__":
    main()
