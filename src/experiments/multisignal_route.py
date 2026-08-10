"""
Multi-signal L1 router (the unifying idea).
===========================================
Don't route on the pure semantic signal alone — feed the MLP every cheap signal
and let it learn to fuse them. The query-relational-hop showed the relational
signal is valuable; here it's one input feature among several:
  q     : dense query embedding                       (semantic — the current router)
  seed  : dense top-1 doc embedding                   (the start ENTITY the query names)
  nbr   : mean of dense top-k doc embeddings          (query neighbourhood / context)
  (splade: query learned-sparse summary — add when built)
Router = MLP(concat(selected signals)) -> partition centroids, trained with KL.
Compares signal sets [q] vs [q+seed] vs [q+seed+nbr] at matched partitions, so we
see how much each added signal helps routing (esp metaqa/KB, where pure semantic
fails). Writes data/ukb_storage/{ds}/results/L1/multisignal.json.
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
from src.experiments.overlap_retrain import (
    _reconstruct, _hard_membership, _centroids, _splits, membership_ref, TAU, HNK, INIT_SEED,
)
from src.alignment.train_mlp import kl_div_loss
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.multisignal_route")
KS = [5, 10, 20, 50, 100]


class MultiRouter(nn.Module):
    def __init__(self, d_in, d_out, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def _all_signals(qs, X, encoder, index, topk=10):
    """Compute the dense retrieval ONCE per split and derive all signals from it
    (q / seed=top1 / nbr=topk-mean). Uses faiss for top-k so it never materializes
    an (nq x n) score/index matrix (that was the metaqa/66k-doc OOM — even chunked
    argpartition allocates a full chunk x n int64 array). Configs then slice/concat."""
    q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
    _, order = index.search(q, topk)                            # (nq, topk) — memory-bounded
    return {"q": q, "seed": X[order[:, 0]], "nbr": X[order].mean(axis=1)}


def _concat(sigdict, signals):
    return np.concatenate([sigdict[s] for s in signals], axis=1).astype("float32")


def _cov(order, test, membership, npart):
    fc = {k: [] for k in KS}; gtr = {k: [] for k in KS}
    for j, (_, _, golds) in enumerate(test):
        golds = [g for g in golds if g in membership]
        if not golds:
            continue
        rank_of = {int(p): r for r, p in enumerate(order[j])}
        best = [min((rank_of.get(p, npart) for p in membership.get(d, ())), default=npart) for d in golds]
        for k in KS:
            cov = sum(1 for b in best if b < k)
            fc[k].append(1.0 if cov == len(best) else 0.0)
            gtr[k].append(cov / len(best))
    return ({k: round(np.mean(fc[k]) * 100, 2) for k in KS}, {k: round(np.mean(gtr[k]) * 100, 2) for k in KS})


def _train_router(feat_tr, tr, feat_va, va, Cg, D, device, tau, hn_k, epochs):
    torch.manual_seed(INIT_SEED)
    model = MultiRouter(feat_tr.shape[1], D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    bs = 64; best, best_state, noimp = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); order = list(range(len(tr))); random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            proj = model(torch.tensor(feat_tr[idx], device=device))
            loss = kl_div_loss(proj, [tr[i][1] for i in idx], Cg, temperature=tau, hn_k=hn_k)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), bs):
                proj = model(torch.tensor(feat_va[s:s + bs], device=device))
                tot += float(kl_div_loss(proj, [p for _, p, _ in va[s:s + bs]], Cg, temperature=tau, hn_k=hn_k)); nb += 1
        vl = tot / max(nb, 1)
        if vl < best:
            best, best_state, noimp = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            noimp += 1
        if noimp >= 12:
            break
    model.load_state_dict(best_state); model.eval()
    return model


def run(dataset, epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index); X = node_vecs.astype("float32"); faiss.normalize_L2(X)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    membership = _hard_membership(engine); membership_ref["hard"] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    D = C.shape[1]; Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1)
    test = splits["test"]

    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)          # for memory-safe top-k retrieval
    sig_tr = _all_signals(splits["train"], X, encoder, index)   # one dense retrieval per split
    sig_va = _all_signals(splits["val"], X, encoder, index)
    sig_te = _all_signals(test, X, encoder, index)
    configs = {"q": ["q"], "q+seed": ["q", "seed"], "q+seed+nbr": ["q", "seed", "nbr"]}
    out = {"dataset": dataset, "npart": npart, "n_test": len(test), "signal_sets": {}}
    for name, sig in configs.items():
        f_tr = _concat(sig_tr, sig); f_va = _concat(sig_va, sig); f_te = _concat(sig_te, sig)
        model = _train_router(f_tr, splits["train"], f_va, splits["val"], Cg, D, device, tau, hn_k, epochs)
        with torch.no_grad():
            scores = (model(torch.tensor(f_te, device=device)) @ Cg.T).cpu().numpy()
        fc, gt = _cov(np.argsort(-scores, axis=1), test, membership, npart)
        out["signal_sets"][name] = {"fullcov": fc, "gt_recall": gt}
        log.info(f"  [{dataset} {name:12}] FullCov {fc}")
    base = out["signal_sets"]["q"]["fullcov"]
    out["fullcov_lift_vs_q"] = {name: {k: round(v["fullcov"][k] - base[k], 2) for k in KS}
                                for name, v in out["signal_sets"].items() if name != "q"}
    with open(rpath(dataset, "L1", "multisignal"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] lift vs q: {out['fullcov_lift_vs_q']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Multi-signal L1 router (semantic + relational + neighbourhood).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "2wiki_clean", "musique_clean"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== MULTI-SIGNAL ROUTER: {ds.upper()} =====")
        run(ds, epochs=a.epochs, limit=a.limit)


if __name__ == "__main__":
    main()
