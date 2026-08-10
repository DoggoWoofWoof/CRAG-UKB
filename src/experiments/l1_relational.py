"""
L1 candidate generation with the relational hop (the win, at L1).
=================================================================
L2 only RERANKS L1's pool — it cannot add docs. So the relational-hop's KB win
(+37 recall) must live in L1 CANDIDATE GENERATION: the pool must actually contain
the answers. L1 retrieval = FUSE two retrievers per query:
  dense : q -> top docs by cosine                 (semantic — strong on text)
  rel   : dense-top1 seed + learned g(q) -> top docs (relational — strong on KB)
fused by Reciprocal Rank Fusion (robust: dense dominates where it wins, rel adds
the KB answers, never a fixed-budget trade). Metric = candidate-pool recall@B
(the recall L2 then reranks within). Shows dense vs rel vs FUSED per dataset.
Writes data/ukb_storage/{ds}/results/L1/relational_pool.json.
"""
import os
import json
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, INIT_SEED
from src.experiments.query_relation import OffsetHead
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.l1_relational")
BUDGETS = [50, 100, 200, 500]


def _rrf(orders, B, k0=60):
    nq = orders[0].shape[0]; out = []
    for qi in range(nq):
        sc = {}
        for od in orders:
            for r, d in enumerate(od[qi].tolist()):
                sc[d] = sc.get(d, 0.0) + 1.0 / (k0 + r)
        out.append(sorted(sc, key=lambda d: -sc[d])[:B])
    return np.array(out)


def _recall(order, gold, budgets=BUDGETS):
    out = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        for b in budgets:
            out[b].append(len(gs & set(order[qi][:b].tolist() if hasattr(order[qi], "tolist") else order[qi][:b])) / len(gs))
    return {f"recall@{b}": round(np.mean(out[b]) * 100, 2) for b in budgets}


def run(dataset, epochs=40, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    d = X.shape[1]; index = faiss.IndexFlatIP(d); index.add(X)
    id2idx = engine.node_id_to_idx
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}
    Xt = torch.tensor(X, device=device)
    maxb = max(BUDGETS)

    def prep(qs):
        q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        _, seed = index.search(q, 1)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed[:, 0], gold
    q_tr, seed_tr, gold_tr = prep(splits["train"])
    q_te, seed_te, gold_te = prep(splits["test"])

    # train relational offset head g (query -> relation direction), applied at doc level
    trip = [(i, seed_tr[i], g) for i, gl in enumerate(gold_tr) for g in gl]
    qtr = torch.tensor(q_tr, device=device)
    torch.manual_seed(INIT_SEED)
    g = OffsetHead(d).to(device); opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    for ep in range(epochs):
        random.Random(ep).shuffle(trip)
        for s in range(0, len(trip), 256):
            b = trip[s:s + 256]
            pred = g(qtr[[t[0] for t in b]], Xt[[int(t[1]) for t in b]])
            goldv = Xt[[int(t[2]) for t in b]]
            loss = F.cross_entropy(pred @ goldv.T / 0.05, torch.arange(len(b), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    g.eval()

    _, dense_order = index.search(q_te, maxb)                    # faiss top-k (no nq x n OOM)
    with torch.no_grad():
        rel_pos = g(torch.tensor(q_te, device=device), Xt[[int(s) for s in seed_te]]).cpu().numpy()
    _, rel_order = index.search(rel_pos.astype("float32"), maxb)
    fused_order = _rrf([dense_order, rel_order], maxb)

    out = {"dataset": dataset, "n_test": len(gold_te), "budgets": BUDGETS,
           "dense": _recall(dense_order, gold_te), "rel": _recall(rel_order, gold_te),
           "fused": _recall(fused_order, gold_te)}
    out["fused_over_dense"] = {b: round(out["fused"][b] - out["dense"][b], 2) for b in out["dense"]}
    with open(rpath(dataset, "L1", "relational_pool"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] recall@200 dense {out['dense']['recall@200']} | rel {out['rel']['recall@200']} | "
             f"FUSED {out['fused']['recall@200']} (+{out['fused_over_dense']['recall@200']})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L1 candidate generation with relational hop (RRF fusion).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--limit", type=int, default=20000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 RELATIONAL CANDIDATE GEN: {ds.upper()} =====")
        run(ds, epochs=a.epochs, limit=a.limit)


if __name__ == "__main__":
    main()
