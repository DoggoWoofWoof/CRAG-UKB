"""
Attention-fusion L1: ONE model, doc-conditioned attention as the fusion mechanism.
==================================================================================
User's question: can a single mlpT with ATTENTION as the importance/fusion mech do
what the champion ensemble (dense + rel_hard + rel_2hop, RRF-fused) does? Prior
evidence says learned fusion loses to parameter-free RRF (mixture -6.8, MINIMAL worst,
universal-gate failed). This builds the strongest single-model version to test it fairly:

  AttnFusion: heads = [dense-direction (q itself), K learned offset arrows]      (H=K+1)
              per-doc head sim  s_hd = arrow_h . doc / tau
              DOC-CONDITIONED attention weight  a_hd = softmax_h( beta*s_hd + gate_h(q) )
              score(doc) = sum_h a_hd * s_hd
  -> each doc is scored mostly by its BEST-matching head (soft), modulated by a learned
     per-query head gate. Generalizes mlpT's fixed soft-OR (logsumexp) with a learnable
     temperature + an explicit dense head. Trained end-to-end on coverage_kl + HNM.

Compared head-to-head, SAME eval, vs: dense, CHAMPION (RRF), ADD (RRF+mlpT), mlpT
(query-gate). Reports recall + FullCov @ {KS}. Writes {ds}/results/L1/attn_fusion.json.
Honest question it answers: does doc-conditioned attention in ONE net match/beat the
ensemble, or does RRF still win?
"""
import os
import gc
import json
import math
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
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, _recall, KS, MAXK
from src.experiments.l1_dynamic import _train_hop2
from src.experiments.l1_mlp_transformer import MLPTransformer, _train as _train_mlpt, _mh_order, _loss, _candidates

log = logging.getLogger("experiments.l1_attn_fusion")
TAU_H = 0.05
HN_K = 16


class AttnFusion(nn.Module):
    def __init__(self, d, K=4, hidden=512):
        super().__init__()
        self.K, self.d = K, d
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.heads = nn.Linear(hidden, K * d)          # K learned offset arrows
        self.gate = nn.Linear(hidden, K + 1)           # per-query head bias (H = dense + K)
        self.log_beta = nn.Parameter(torch.zeros(1))   # learnable attention sharpness

    def forward(self, qn, seed):
        h = self.trunk(qn)
        arr = F.normalize(seed.unsqueeze(1) + self.heads(h).view(-1, self.K, self.d), dim=-1)  # (B,K,d)
        dense = F.normalize(qn, dim=-1).unsqueeze(1)   # (B,1,d) — the dense/semantic head
        arrows = torch.cat([dense, arr], dim=1)        # (B,H,d)
        return arrows, self.gate(h)                    # (B,H,d), (B,H)


def _attn_logits(arrows, gate_bias, log_beta, docs):
    """Doc-conditioned attention over heads: a_hd = softmax_h(beta*s_hd + gate_h); sum_h a_hd*s_hd."""
    beta = torch.exp(log_beta)
    s = torch.einsum("bhd,cd->bhc", arrows, docs) / TAU_H          # (B,H,C) head-doc sims
    a = torch.softmax(beta * s + gate_bias.unsqueeze(-1), dim=1)   # attention over heads, per doc
    return (a * s).sum(1)                                          # (B,C)


def _train_attn(head, q_tr, seed_tr, gold_tr, Xt, index, device, epochs, bs=128):
    idx = [i for i, g in enumerate(gold_tr) if g]
    qtr = torch.tensor(q_tr, device=device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for ep in range(epochs):
        head.train(); random.Random(ep).shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            arrows, gate_bias = head(qtr[b], Xt[[int(seed_tr[i]) for i in b]])
            cv, mask = _candidates([gold_tr[i] for i in b], arrows, index, Xt, device, hn_k=HN_K)
            loss = _loss(_attn_logits(arrows, gate_bias, head.log_beta, cv), mask)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def _attn_order(head, q_te, seed_te, Xt, device):
    with torch.no_grad():
        arrows, gate_bias = head(torch.tensor(q_te, device=device), Xt[[int(s) for s in seed_te]])
        beta = torch.exp(head.log_beta)
    out = []
    for i in range(arrows.shape[0]):                              # per-query score over all docs
        s = (arrows[i] @ Xt.T) / TAU_H                            # (H,n)
        a = torch.softmax(beta * s + gate_bias[i].unsqueeze(-1), dim=0)
        score = (a * s).sum(0)                                    # (n,)
        out.append(torch.argsort(-score)[:MAXK].cpu().numpy())
    return out


def _fullcov(order, gold, budgets=KS):
    o = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g); top = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            o[b].append(1.0 if gs <= set(top[:b]) else 0.0)
    return {b: round(float(np.mean(o[b])) * 100, 2) for b in budgets}


def _both(order, gold):
    return {"recall": _recall(order, gold), "fullcov": _fullcov(order, gold)}


def run(dataset, epochs=25, limit=12000, K=4, device=None):
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
    qte = torch.tensor(q_te, device=device)
    avg_golds = round(float(np.mean([len(g) for g in gold_te if g])), 2)

    log.info(f"[{dataset}] train g1,g_hard,g2 (champion), mlpT (ADD), AttnFusion (K={K})...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs)
    torch.manual_seed(INIT_SEED)
    mlpt = _train_mlpt(MLPTransformer(d, K).to(device), q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    torch.manual_seed(INIT_SEED)
    attn = _train_attn(AttnFusion(d, K).to(device), q_tr, seed_tr, gold_tr, Xt, index, device, epochs)

    def pos(headm):
        with torch.no_grad():
            return headm(qte, Xt[[int(s) for s in seed_te]]).cpu().numpy()
    dense = _order(q_te, index)
    rel_hard = _order(pos(g_hard), index)
    hop1 = _order(pos(g1), index); s1 = hop1[:, 0]
    with torch.no_grad():
        hop2 = _order(g2(qte, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
    rel_2hop = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    mlpt_order = _mh_order(mlpt, q_te, seed_te, X, Xt, index, device)
    attn_order = _attn_order(attn, q_te, seed_te, Xt, device)

    dmap, hmap, tmap, mmap = _ranks(dense), _ranks(rel_hard), _ranks(rel_2hop), _ranks(mlpt_order)
    cfg = {
        "dense": _both(dense, gold_te),
        "mlp_transformer": _both(mlpt_order, gold_te),
        "attn_fusion": _both(attn_order, gold_te),
        "CHAMPION": _both(_rrf_fuse([dmap, hmap, tmap], [1.0, 1.0, 1.0]), gold_te),
        "ADD_champion+mlpT": _both(_rrf_fuse([dmap, hmap, tmap, mmap], [1.0, 1.0, 1.0, 1.0]), gold_te),
    }

    def dr(a, b, m):
        return round(cfg[a][m][100] - cfg[b][m][100], 2)
    out = {"dataset": dataset, "K": K, "avg_golds_per_q": avg_golds, "limit": limit, "budgets": KS, "configs": cfg,
           "attn_vs_ADD@100": {"recall": dr("attn_fusion", "ADD_champion+mlpT", "recall"),
                               "fullcov": dr("attn_fusion", "ADD_champion+mlpT", "fullcov")},
           "attn_vs_CHAMPION@100": {"recall": dr("attn_fusion", "CHAMPION", "recall"),
                                    "fullcov": dr("attn_fusion", "CHAMPION", "fullcov")},
           "attn_vs_mlpT@100": {"recall": dr("attn_fusion", "mlp_transformer", "recall"),
                                "fullcov": dr("attn_fusion", "mlp_transformer", "fullcov")}}
    os.makedirs(os.path.join("data", "ukb_storage", dataset, "results", "L1"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "attn_fusion.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] FCOV@100 dense {cfg['dense']['fullcov'][100]} | mlpT {cfg['mlp_transformer']['fullcov'][100]} "
             f"| ATTN {cfg['attn_fusion']['fullcov'][100]} | CHAMP {cfg['CHAMPION']['fullcov'][100]} "
             f"| ADD {cfg['ADD_champion+mlpT']['fullcov'][100]} || attn vs ADD {out['attn_vs_ADD@100']['fullcov']:+} "
             f"vs CHAMP {out['attn_vs_CHAMPION@100']['fullcov']:+} vs mlpT {out['attn_vs_mlpT@100']['fullcov']:+}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Attention-fusion (one-model, doc-conditioned) vs champion/ADD ensemble.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=12000)
    p.add_argument("--K", type=int, default=4)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 ATTN-FUSION vs ENSEMBLE: {ds.upper()} =====")
        try:
            results[ds] = run(ds, epochs=a.epochs, limit=a.limit, K=a.K)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
        gc.collect()
    if results:
        def mean(cfg, m):
            return round(float(np.mean([r["configs"][cfg][m][100] for r in results.values()])), 2)
        cfgs = ["dense", "mlp_transformer", "attn_fusion", "CHAMPION", "ADD_champion+mlpT"]
        summary = {"datasets": list(results),
                   "fullcov_mean@100": {c: mean(c, "fullcov") for c in cfgs},
                   "recall_mean@100": {c: mean(c, "recall") for c in cfgs}}
        summary["attn_vs_ADD_fullcov@100"] = round(summary["fullcov_mean@100"]["attn_fusion"] - summary["fullcov_mean@100"]["ADD_champion+mlpT"], 2)
        with open(os.path.join("data", "ukb_storage", "_index", "l1_attn_fusion_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"ATTN-FUSION FCOV@100 {summary['fullcov_mean@100']} | attn vs ADD {summary['attn_vs_ADD_fullcov@100']:+}")


if __name__ == "__main__":
    main()
