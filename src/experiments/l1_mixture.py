"""
Set-mixture head — multi-head "relation attention" for 1-to-many answers.
=========================================================================
The single offset seed+g(q) points at ONE region, so it structurally can't gather
a large answer set (metaqa: actor -> MANY movies). The old rel_mix produced K
directions but trained max-over-K with no diversity pressure, so the heads COLLAPSED
to one direction (8% @100 on metaqa). This fixes it the way multi-head attention does:

  K heads (separate projections)  ->  K query-conditioned directions g_k(q)
  answer positions:  pos_k = normalize(seed + g_k(q))          (signed: add OR subtract)

Training = set prediction with RESPONSIBILITY assignment (mixture-model E-step /
slot-attention / DETR): each gold is pulled ONLY toward its nearest head, so heads
specialize on different answer clusters; + a DIVERSITY regularizer (penalize pairwise
cosine of the K directions) so they cannot collapse.
Retrieval = union of the K heads' top docs, scored by MAX similarity to any head
(set coverage). Reports dense | rel_base (single offset) | MIXTURE | dense+mixture,
+ a diversity diagnostic (mean off-diagonal cosine of the K directions — low = spread).
Writes {ds}/results/L1/mixture.json.
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
from src.experiments.l1_ablate import _order, _ranks, _rrf_fuse, _recall, KS, MAXK, TAU


class SetMixtureHead(nn.Module):
    def __init__(self, d, K=8, hidden=512):
        super().__init__()
        self.K, self.d = K, d
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.heads = nn.Linear(hidden, K * d)                # K projection heads, like attention

    def offsets(self, qn):
        return self.heads(self.trunk(qn)).view(-1, self.K, self.d)   # (B,K,d)

    def forward(self, qn, seed):
        off = self.offsets(qn)
        return F.normalize(seed.unsqueeze(1) + off, dim=-1), off      # positions (B,K,d), raw offsets


def _train(head, q_tr, seed_tr, gold_tr, Xt, device, epochs, lam=0.5):
    trip = [(i, int(seed_tr[i]), int(g)) for i, gl in enumerate(gold_tr) for g in gl]
    qtr = torch.tensor(q_tr, device=device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3); bs = 256
    for ep in range(epochs):
        head.train(); random.Random(ep).shuffle(trip)
        for s in range(0, len(trip), bs):
            b = trip[s:s + bs]
            qn = qtr[[t[0] for t in b]]; seed = Xt[[t[1] for t in b]]; goldv = Xt[[t[2] for t in b]]
            pos, off = head(qn, seed)                                 # (B,K,d)
            sim_kg = (pos * goldv.unsqueeze(1)).sum(-1)               # (B,K) each head vs THIS gold
            resp = sim_kg.argmax(1)                                    # responsible head per gold
            resp_pos = pos[torch.arange(len(b), device=device), resp]  # (B,d)
            loss_nce = F.cross_entropy(resp_pos @ goldv.T / TAU, torch.arange(len(b), device=device))
            offn = F.normalize(off, dim=-1)
            gram = torch.bmm(offn, offn.transpose(1, 2))              # (B,K,K) pairwise cosine
            eye = torch.eye(head.K, device=device).unsqueeze(0)
            loss_div = (gram * (1 - eye)).pow(2).mean()              # push directions apart
            loss = loss_nce + lam * loss_div
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def _mixture_order(head, q, seed, X, Xt, index, device, per_head=100):
    """Union of the K heads' top docs, scored by MAX cosine to any head (set coverage)."""
    with torch.no_grad():
        pos, off = head(torch.tensor(q, device=device), Xt[[int(s) for s in seed]])
    pos = pos.cpu().numpy()                                          # (nq,K,d)
    K = pos.shape[1]; nq = pos.shape[0]; out = []
    per_head_orders = [_order(pos[:, k, :], index, per_head) for k in range(K)]  # K x (nq, per_head)
    for i in range(nq):
        cand = list(dict.fromkeys(np.concatenate([o[i] for o in per_head_orders]).tolist()))
        cv = X[cand]                                                 # (m,d)
        sims = cv @ pos[i].T                                         # (m,K)
        score = sims.max(axis=1)                                     # best head per candidate
        out.append([cand[j] for j in np.argsort(-score)[:MAXK]])
    # diversity diagnostic (mean off-diagonal cosine of the K directions)
    offn = F.normalize(off, dim=-1)
    gram = torch.bmm(offn, offn.transpose(1, 2))
    eye = torch.eye(K, device=device).unsqueeze(0)
    div = float((gram * (1 - eye)).abs().sum() / ((K * K - K) * nq))
    return out, div


def run(dataset, epochs=25, limit=20000, K=8, device=None):
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

    from src.experiments.l1_ablate import _train_offset            # single-offset baseline
    g_single = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    torch.manual_seed(INIT_SEED)
    head = _train(SetMixtureHead(d, K).to(device), q_tr, seed_tr, gold_tr, Xt, device, epochs)

    dense = _order(q_te, index)
    with torch.no_grad():
        single = _order(g_single(torch.tensor(q_te, device=device),
                                 Xt[[int(s) for s in seed_te]]).cpu().numpy(), index)
    mix, div = _mixture_order(head, q_te, seed_te, X, Xt, index, device)

    dmap, mmap = _ranks(dense), _ranks(mix)
    cfg = {"dense": _recall(dense, gold_te), "rel_base_single": _recall(single, gold_te),
           "mixture": _recall(mix, gold_te),
           "dense+mixture": _recall(_rrf_fuse([dmap, mmap], [1.0, 1.0]), gold_te)}
    out = {"dataset": dataset, "K": K, "avg_golds_per_q": avg_golds, "budgets": KS,
           "direction_offdiag_cos": round(div, 4), "configs": cfg,
           "mixture_vs_single@100": round(cfg["mixture"][100] - cfg["rel_base_single"][100], 2)}
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "mixture.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logging.getLogger("experiments.l1_mixture").info(
        f"[{dataset}] avg_golds {avg_golds} | div {div:.3f} | dense {cfg['dense'][100]} | "
        f"single {cfg['rel_base_single'][100]} | MIXTURE {cfg['mixture'][100]} "
        f"(vs single {out['mixture_vs_single@100']}) | dense+mix {cfg['dense+mixture'][100]}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Attention-style set-mixture head for 1-to-many relations.")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--K", type=int, default=8)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        logging.getLogger("experiments.l1_mixture").info(f"===== L1 SET-MIXTURE: {ds.upper()} =====")
        try:
            run(ds, epochs=a.epochs, limit=a.limit, K=a.K)
        except Exception as e:
            logging.getLogger("experiments.l1_mixture").exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
