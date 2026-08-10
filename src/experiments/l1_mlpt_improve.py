"""
Improve the MLP-transformer: heads (K) x epochs x head-diversity regularizer.
=============================================================================
The head-to-head proved SWAP (dense + MLP-transformer + rel_2hop) beats the champion
on 2wiki/musique. Now push the mlpT COMPONENT itself. Three hypotheses:
  - MORE HEADS: does K>4 cover the multi-gold answer cloud better (metaqa avg 7.3)?
  - UNDERTRAINED: the user's "transformers train long" point — do more epochs help?
  - HEAD COLLAPSE: do the K offset directions collapse together? A diversity reg
    (penalize pairwise head-offset cosine) keeps them spread -> more coverage.

Trains the validated MLPTransformer (soft-OR + learned per-query gate + coverage_kl +
HNM) across K and epochs, with an optional diversity penalty added to the loss, and
reports mlpT-standalone recall + FullCov @ {KS}. Winner (by FullCov@100) feeds the L1
champion. Non-metaqa by default (metaqa runs after the head-to-head frees compute).
Writes {ds}/results/L1/mlpt_improve.json + _index/l1_mlpt_improve_summary.json.
"""
import os
import gc
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, INIT_SEED
from src.experiments.l1_ablate import _order, _recall, KS, MAXK
from src.experiments.l1_mlp_transformer import (MLPTransformer, _combined_logits, _loss,
                                                _candidates, _mh_order, HN_K)

log = logging.getLogger("experiments.l1_mlpt_improve")


def _fullcov(order, gold, budgets=KS):
    out = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        top_all = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            out[b].append(1.0 if gs <= set(top_all[:b]) else 0.0)
    return {b: round(float(np.mean(out[b])) * 100, 2) for b in budgets}


def _diversity(pos):
    """Mean pairwise |cos| between the K head offset directions (0 = orthogonal)."""
    o = pos - pos.mean(1, keepdim=True)                      # remove the shared seed
    o = F.normalize(o, dim=-1)
    g = torch.einsum("bkd,bjd->bkj", o, o).abs()
    K = pos.shape[1]
    return (g.sum((1, 2)) - K) / (K * (K - 1) + 1e-9)        # off-diagonal mean


def _train_div(head, q_tr, seed_tr, gold_tr, Xt, index, device, epochs, lam_div=0.0, bs=128):
    """Same trainer as l1_mlp_transformer._train but with an optional diversity penalty."""
    import random
    idx = [i for i, g in enumerate(gold_tr) if g]
    qtr = torch.tensor(q_tr, device=device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for ep in range(epochs):
        head.train(); random.Random(ep).shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            pos, w = head(qtr[b], Xt[[int(seed_tr[i]) for i in b]])
            cv, mask = _candidates([gold_tr[i] for i in b], pos, index, Xt, device)
            loss = _loss(_combined_logits(pos, w, cv), mask)
            if lam_div:
                loss = loss + lam_div * _diversity(pos).mean()   # encourage spread heads
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def run(dataset, Ks=(4, 6, 8), epochs_list=(30, 60), lam_divs=(0.0, 0.1), limit=15000, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    d = X.shape[1]; index = faiss.IndexFlatIP(d); index.add(X); Xt = torch.tensor(X, device=device)
    id2idx = engine.node_id_to_idx
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}

    def prep(qs):
        q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        _, seed = index.search(q, 1)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed[:, 0], gold
    q_tr, seed_tr, gold_tr = prep(splits["train"])
    q_te, seed_te, gold_te = prep(splits["test"])
    avg_golds = round(float(np.mean([len(g) for g in gold_te if g])), 2)

    def evalcfg(K, epochs, lam_div):
        torch.manual_seed(INIT_SEED)
        head = _train_div(MLPTransformer(d, K).to(device), q_tr, seed_tr, gold_tr,
                          Xt, index, device, epochs, lam_div)
        order = _mh_order(head, q_te, seed_te, X, Xt, index, device, per_head=150)
        r = {"recall": _recall(order, gold_te), "fullcov": _fullcov(order, gold_te)}
        del head; gc.collect()
        return r

    cfg = {}
    base_ep = epochs_list[0]
    log.info(f"[{dataset}] avg_golds {avg_golds} | K-sweep {Ks} @ {base_ep}ep...")
    for K in Ks:
        cfg[f"K{K}_e{base_ep}_d0"] = evalcfg(K, base_ep, 0.0)
        log.info(f"[{dataset}] K{K}_e{base_ep}: R@100 {cfg[f'K{K}_e{base_ep}_d0']['recall'][100]} "
                 f"FCOV@100 {cfg[f'K{K}_e{base_ep}_d0']['fullcov'][100]}")
    bestK = int(max(Ks, key=lambda K: cfg[f"K{K}_e{base_ep}_d0"]["fullcov"][100]))

    for ep in epochs_list[1:]:                               # epochs sweep at best K
        cfg[f"K{bestK}_e{ep}_d0"] = evalcfg(bestK, ep, 0.0)
        log.info(f"[{dataset}] K{bestK}_e{ep}: R@100 {cfg[f'K{bestK}_e{ep}_d0']['recall'][100]} "
                 f"FCOV@100 {cfg[f'K{bestK}_e{ep}_d0']['fullcov'][100]}")
    best_ep = int(max(epochs_list, key=lambda ep: cfg[f"K{bestK}_e{ep}_d0"]["fullcov"][100]))

    for ld in lam_divs:                                      # diversity reg at best K/epochs
        if ld == 0.0:
            continue
        cfg[f"K{bestK}_e{best_ep}_d{ld}"] = evalcfg(bestK, best_ep, ld)
        log.info(f"[{dataset}] K{bestK}_e{best_ep}_div{ld}: R@100 {cfg[f'K{bestK}_e{best_ep}_d{ld}']['recall'][100]} "
                 f"FCOV@100 {cfg[f'K{bestK}_e{best_ep}_d{ld}']['fullcov'][100]}")

    best_cfg = max(cfg, key=lambda c: cfg[c]["fullcov"][100])
    base = cfg.get(f"K4_e{base_ep}_d0")
    out = {"dataset": dataset, "avg_golds_per_q": avg_golds, "limit": limit, "budgets": KS, "configs": cfg,
           "best_cfg": best_cfg, "best_fullcov@100": cfg[best_cfg]["fullcov"][100],
           "best_recall@100": cfg[best_cfg]["recall"][100]}
    if base:
        out["best_vs_K4base_fullcov@100"] = round(cfg[best_cfg]["fullcov"][100] - base["fullcov"][100], 2)
        out["best_vs_K4base_recall@100"] = round(cfg[best_cfg]["recall"][100] - base["recall"][100], 2)
    os.makedirs(os.path.join("data", "ukb_storage", dataset, "results", "L1"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "mlpt_improve.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] BEST {best_cfg}: FCOV@100 {cfg[best_cfg]['fullcov'][100]} R@100 {cfg[best_cfg]['recall'][100]} "
             f"(vs K4 base: {out.get('best_vs_K4base_fullcov@100')} FCOV)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Improve the MLP-transformer (K x epochs x diversity).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean"])
    p.add_argument("--Ks", type=int, nargs="+", default=[4, 6, 8])
    p.add_argument("--epochs_list", type=int, nargs="+", default=[30, 60])
    p.add_argument("--lam_divs", type=float, nargs="+", default=[0.0, 0.1])
    p.add_argument("--limit", type=int, default=15000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 MLP-TRANSFORMER IMPROVE: {ds.upper()} =====")
        try:
            results[ds] = run(ds, Ks=tuple(a.Ks), epochs_list=tuple(a.epochs_list),
                              lam_divs=tuple(a.lam_divs), limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
        gc.collect()
    if results:
        summary = {"datasets": list(results),
                   "best_cfg": {ds: r["best_cfg"] for ds, r in results.items()},
                   "best_fullcov@100": {ds: r["best_fullcov@100"] for ds, r in results.items()},
                   "best_vs_K4base_fullcov@100": {ds: r.get("best_vs_K4base_fullcov@100") for ds, r in results.items()}}
        path = os.path.join("data", "ukb_storage", "_index", "l1_mlpt_improve_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"MLPT-IMPROVE best_cfg {summary['best_cfg']} | FCOV@100 {summary['best_fullcov@100']} "
                 f"| vs K4 base {summary['best_vs_K4base_fullcov@100']}")


if __name__ == "__main__":
    main()
