"""
MLP-transformer: multi-head relational retriever, done the transformer way.
===========================================================================
The mixture head FAILED (−6.8) from three anti-transformer mistakes: hard
responsibility-assignment (each head trained on 1/K golds), a FIXED union/RRF
combiner, and single-positive-ish training. Real multi-head attention does the
opposite: every head sees the FULL input, a LEARNED combiner (W^O), soft mixing.
This rebuilds it that way — pure MLP, no GNN (GNNs oversmooth; APPNP already lost
to PPR):

  K heads:      pos_k = normalize(seed + g_k(q))            (K offset directions)
  head-gate:    w = softmax(gate(q))                        (LEARNED per-query weights)
  combine:      score(doc) = logsumexp_k( log w_k + pos_k·doc / tau_h )   (soft-OR, weighted)

Trained END-TO-END on coverage_kl (KL + CVaR weakest-positive) — the loss that beat
InfoNCE in the A/B — with HNM. No fragmentation: every head gets gradient from every
gold through the soft-OR, so heads specialize EMERGENTLY (transformer-style), not by
forced assignment. Compared head-to-head vs the single offset on metaqa (1-to-many).
Writes {ds}/results/L1/mlp_transformer.json.
"""
import os
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
from src.experiments.l1_ablate import _order, _ranks, _rrf_fuse, _recall, KS, MAXK

log = logging.getLogger("experiments.l1_mlp_transformer")
TAU_H = 0.05
HN_K = 16
LAM_COV = 0.5


class MLPTransformer(nn.Module):
    def __init__(self, d, K=4, hidden=512):
        super().__init__()
        self.K, self.d = K, d
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.heads = nn.Linear(hidden, K * d)              # K offset directions
        self.gate = nn.Linear(hidden, K)                   # learned per-query head weights (the "W^O" router)

    def forward(self, qn, seed):
        h = self.trunk(qn)
        pos = F.normalize(seed.unsqueeze(1) + self.heads(h).view(-1, self.K, self.d), dim=-1)  # (B,K,d)
        w = F.softmax(self.gate(h), dim=-1)                # (B,K)
        return pos, w


def _combined_logits(pos, w, docs, tau_h=TAU_H):
    """Weighted soft-OR over heads: logsumexp_k( log w_k + pos_k·doc / tau_h ). (B,C)"""
    s = torch.einsum("bkd,cd->bkc", pos, docs) / tau_h + torch.log(w + 1e-9).unsqueeze(-1)
    return torch.logsumexp(s, dim=1)


def _loss(logits, gold_mask, lam_cov=LAM_COV, cvar=0.25):
    logp = F.log_softmax(logits, dim=1)
    tgt = gold_mask.float(); tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1)
    kl = -(tgt * logp).sum(1).mean()                       # multi-positive KL
    cov = []                                               # CVaR weakest-positive (Jigsaw coverage)
    for i in range(logits.shape[0]):
        gp = logp[i][gold_mask[i]]
        if gp.numel() == 0:
            continue
        k = max(1, int(math.ceil(gp.numel() * cvar)))
        cov.append(-torch.topk(gp, k, largest=False).values.mean())   # push the weakest-covered golds up
    cov = torch.stack(cov).mean() if cov else logits.new_tensor(0.0)
    return kl + lam_cov * cov


def _candidates(gold_sets, pos, index, Xt, device, hn_k=HN_K):
    bg = set()
    for gs in gold_sets:
        bg.update(gs)
    if hn_k:                                               # mine hard negs near each head's prediction
        flat = pos.reshape(-1, pos.shape[-1]).detach().cpu().numpy().astype("float32")
        with torch.no_grad():
            _, nn_idx = index.search(np.ascontiguousarray(flat), hn_k)
        bg.update(int(x) for x in nn_idx.reshape(-1) if x >= 0)
    cand = sorted(bg)
    posmap = {c: j for j, c in enumerate(cand)}
    cv = Xt[torch.tensor(cand, device=device)]
    mask = torch.zeros(len(gold_sets), len(cand), dtype=torch.bool, device=device)
    for i, gs in enumerate(gold_sets):
        for g in gs:
            mask[i, posmap[g]] = True
    return cv, mask


def _train(head, q_tr, seed_tr, gold_tr, Xt, index, device, epochs, bs=128):
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
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def _mh_order(head, q_te, seed_te, X, Xt, index, device, per_head=100):
    with torch.no_grad():
        pos, w = head(torch.tensor(q_te, device=device), Xt[[int(s) for s in seed_te]])
    pos_np = pos.cpu().numpy(); w_np = w.cpu().numpy(); K = pos_np.shape[1]; nq = pos_np.shape[0]
    per = [_order(pos_np[:, k, :], index, per_head) for k in range(K)]
    out = []
    for i in range(nq):
        cand = list(dict.fromkeys(np.concatenate([per[k][i] for k in range(K)]).tolist()))
        cv = X[cand]                                       # (m,d)
        s = cv @ pos_np[i].T / TAU_H + np.log(w_np[i] + 1e-9)   # (m,K)
        score = np.logaddexp.reduce(s, axis=1)             # combined soft-OR
        out.append([cand[j] for j in np.argsort(-score)[:MAXK]])
    return out


def run(dataset, epochs=30, limit=8000, K=4, device=None):
    device = device or torch.device("cpu")
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

    from src.experiments.l1_ablate import _train_offset            # single-head baseline (base offset)
    g_single = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    torch.manual_seed(INIT_SEED)
    head = _train(MLPTransformer(d, K).to(device), q_tr, seed_tr, gold_tr, Xt, index, device, epochs)

    with torch.no_grad():
        single = _order(g_single(torch.tensor(q_te, device=device),
                                 Xt[[int(s) for s in seed_te]]).cpu().numpy(), index)
    mh = _mh_order(head, q_te, seed_te, X, Xt, index, device)
    dmap, mmap = _ranks(_order(q_te, index)), _ranks(mh)
    cfg = {"dense": _recall(_order(q_te, index), gold_te),
           "single_offset": _recall(single, gold_te),
           "mlp_transformer": _recall(mh, gold_te),
           "dense+mlp_transformer": _recall(_rrf_fuse([dmap, mmap], [1.0, 1.0]), gold_te)}
    out = {"dataset": dataset, "K": K, "avg_golds_per_q": avg_golds, "budgets": KS, "configs": cfg,
           "mlpT_vs_single@100": round(cfg["mlp_transformer"][100] - cfg["single_offset"][100], 2)}
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "mlp_transformer.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] K={K} avg_golds {avg_golds} | dense {cfg['dense'][100]} | single {cfg['single_offset'][100]} "
             f"| MLP-TRANSFORMER {cfg['mlp_transformer'][100]} (vs single {out['mlpT_vs_single@100']}) "
             f"| dense+mlpT {cfg['dense+mlp_transformer'][100]}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Transformer-style multi-head MLP relational retriever (soft-OR + coverage_kl).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--K", type=int, default=4)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 MLP-TRANSFORMER: {ds.upper()} (K={a.K}) =====")
        try:
            run(ds, epochs=a.epochs, limit=a.limit, K=a.K)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
