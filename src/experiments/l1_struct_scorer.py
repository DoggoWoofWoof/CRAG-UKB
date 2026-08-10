"""
Offset-signal set-transformer scorer — dynamic, probabilistic ranking, OFFSET-ONLY.
===================================================================================
Scope: our L1 is offset-based (frozen encoder + learned offset heads) — NO graph, NO PPR,
NO KG. So the dynamic scorer's "rich signals" are all OFFSET-DERIVED. The attention-fusion
collapsed because it effectively had one scalar per head; here each candidate carries its
similarity to EVERY offset direction, and a set-transformer scores the pool context-aware:

  per-candidate features (all embedding/offset-space):
    - dense cosine            q . d
    - 1-hop hard offset sim   normalize(seed + g_hard(q)) . d
    - base 1-hop offset sim   normalize(seed + g1(q)) . d
    - 2-hop offset sim        normalize(s1 + g2(q)) . d        (offset COMPOSITION, not graph)
    - K multi-head offset sims normalize(seed + head_k(q)) . d  (the mlpT directions)

A small SET-TRANSFORMER (candidates attend to each other) emits one score per candidate;
softmax = probabilistic. Trained with the multi-positive coverage loss on the golds in the
pool. This is "learned dynamic fusion of the offset directions" — the question is whether a
context-aware learned scorer beats the parameter-free RRF ensemble on the SAME offset signals.
Pool = dense-top ∪ hard-offset-top ∪ per-head tops (offset/embedding only). Compared vs
dense / CHAMPION / ADD. Improvement search (relative); winner re-verified on frozen substrate.
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
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, _recall, KS, MAXK
from src.experiments.l1_dynamic import _train_hop2
from src.experiments.l1_mlp_transformer import MLPTransformer, _train as _train_mlpt

log = logging.getLogger("experiments.l1_struct_scorer")
POOL_DENSE, POOL_OFF = 150, 40      # candidate pool: dense-top + per-offset-direction tops
POOL_MAX = 256
K_HEADS = 4


class SetScorer(nn.Module):
    def __init__(self, n_feat, d_model=128, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_feat, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.score = nn.Linear(d_model, 1)

    def forward(self, feats, mask):                       # feats (B,P,F), mask (B,P) True=valid
        h = self.encoder(self.embed(feats), src_key_padding_mask=~mask)   # candidates attend to each other
        return self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))


def _coverage_kl(logits, gold_mask, lam=0.5, cvar=0.25):
    logp = F.log_softmax(logits, dim=1)
    tgt = gold_mask.float(); tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1)
    kl = -(tgt * torch.nan_to_num(logp, neginf=0.0)).sum(1).mean()
    cov = []
    for i in range(logits.shape[0]):
        gp = logp[i][gold_mask[i]]; gp = gp[torch.isfinite(gp)]
        if gp.numel():
            k = max(1, int(math.ceil(gp.numel() * cvar)))
            cov.append(-torch.topk(gp, k, largest=False).values.mean())
    cov = torch.stack(cov).mean() if cov else logits.new_tensor(0.0)
    return kl + lam * cov


def _fullcov(order, gold, budgets=KS):
    out = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g); top = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            out[b].append(1.0 if gs <= set(top[:b]) else 0.0)
    return {b: round(float(np.mean(out[b])) * 100, 2) for b in budgets}


def _both(order, gold):
    return {"recall": _recall(order, gold), "fullcov": _fullcov(order, gold)}


def run(dataset, epochs=25, limit=8000, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n, d = X.shape; index = faiss.IndexFlatIP(d); index.add(X); Xt = torch.tensor(X, device=device)
    id2idx = engine.node_id_to_idx
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}

    def prep(qs):
        q = encoder.encode([nd.content for nd, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        _, seed = index.search(q, 1)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed[:, 0], gold
    q_tr, seed_tr, gold_tr = prep(splits["train"])
    q_te, seed_te, gold_te = prep(splits["test"])
    n_feat = 4 + K_HEADS

    log.info(f"[{dataset}] training offset heads (base,hard,2hop,mlpT-K{K_HEADS}) for scorer features...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, 25)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, 25)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, 25)
    torch.manual_seed(INIT_SEED)
    mlpt = _train_mlpt(MLPTransformer(d, K_HEADS).to(device), q_tr, seed_tr, gold_tr, Xt, index, device, 25)

    def directions(q, seed):
        """All offset directions for each query: (pos_hard, pos1, pos2, head_k...) each (nq,d)."""
        qt = torch.tensor(q, device=device); sv = Xt[[int(s) for s in seed]]
        with torch.no_grad():
            pos_hard = F.normalize(g_hard(qt, sv), dim=-1).cpu().numpy()
            pos1 = F.normalize(g1(qt, sv), dim=-1).cpu().numpy()
            s1 = _order(pos1, index)[:, 0]
            pos2 = F.normalize(g2(qt, Xt[[int(s) for s in s1]]), dim=-1).cpu().numpy()
            heads, _w = mlpt(qt, sv)                        # (nq,K,d) already normalized in mlpT
            heads = heads.cpu().numpy()
        return pos_hard, pos1, pos2, heads

    def build(q, seed, gold):
        nq = len(q); qsim = q @ X.T
        pos_hard, pos1, pos2, heads = directions(q, seed)
        dense_top = np.argsort(-qsim, axis=1)[:, :POOL_DENSE]
        pool = np.zeros((nq, POOL_MAX), np.int64); mask = np.zeros((nq, POOL_MAX), bool)
        feats = np.zeros((nq, POOL_MAX, n_feat), np.float32); gmask = np.zeros((nq, POOL_MAX), bool)
        for i in range(nq):
            cand = dense_top[i].tolist()
            cand += _order(pos_hard[i][None], index, POOL_OFF)[0].tolist()
            for k in range(K_HEADS):
                cand += _order(heads[i, k][None], index, POOL_OFF // 2)[0].tolist()
            cand = list(dict.fromkeys(cand))[:POOL_MAX]
            gs = set(gold[i])
            for j, dc in enumerate(cand):
                xd = X[dc]
                feats[i, j, 0] = qsim[i, dc]
                feats[i, j, 1] = pos_hard[i] @ xd
                feats[i, j, 2] = pos1[i] @ xd
                feats[i, j, 3] = pos2[i] @ xd
                for k in range(K_HEADS):
                    feats[i, j, 4 + k] = heads[i, k] @ xd
                mask[i, j] = True
                if dc in gs:
                    gmask[i, j] = True
            pool[i, :len(cand)] = cand          # pool pre-zeroed; mask marks the valid slots
        return pool, feats, mask, gmask

    log.info(f"[{dataset}] building offset-signal pools (train {len(q_tr)}, test {len(q_te)})...")
    pool_tr, feat_tr, mask_tr, gmask_tr = build(q_tr, seed_tr, gold_tr)
    pool_te, feat_te, mask_te, gmask_te = build(q_te, seed_te, gold_te)

    torch.manual_seed(INIT_SEED)
    model = SetScorer(n_feat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    idx = [i for i in range(len(q_tr)) if gmask_tr[i].any()]
    ften = torch.tensor(feat_tr, device=device); mten = torch.tensor(mask_tr, device=device)
    gten = torch.tensor(gmask_tr, device=device)
    for ep in range(epochs):
        model.train(); random.Random(ep).shuffle(idx)
        for s in range(0, len(idx), 128):
            b = idx[s:s + 128]
            loss = _coverage_kl(model(ften[b], mten[b]), gten[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        sc = model(torch.tensor(feat_te, device=device), torch.tensor(mask_te, device=device)).cpu().numpy()
    scorer_order = []
    for i in range(len(q_te)):
        rank = np.argsort(-sc[i])
        scorer_order.append([int(pool_te[i, j]) for j in rank if mask_te[i, j]][:MAXK])

    dense = _order(q_te, index); qte = torch.tensor(q_te, device=device)

    def pos(h):
        with torch.no_grad():
            return h(qte, Xt[[int(s) for s in seed_te]]).cpu().numpy()
    rel_hard = _order(pos(g_hard), index)
    hop1 = _order(pos(g1), index); s1 = hop1[:, 0]
    with torch.no_grad():
        hop2 = _order(g2(qte, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
    rel_2hop = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    dmap, hmap, tmap = _ranks(dense), _ranks(rel_hard), _ranks(rel_2hop)
    cfg = {
        "dense": _both(dense, gold_te),
        "CHAMPION": _both(_rrf_fuse([dmap, hmap, tmap], [1.0, 1.0, 1.0]), gold_te),
        "offset_scorer": _both(scorer_order, gold_te),
        "scorer+CHAMPION": _both(_rrf_fuse([dmap, hmap, tmap, _ranks(scorer_order)], [1, 1, 1, 1]), gold_te),
    }
    out = {"dataset": dataset, "limit": limit, "n_feat": n_feat, "K_heads": K_HEADS, "budgets": KS,
           "avg_golds_per_q": round(float(np.mean([len(g) for g in gold_te if g])), 2), "configs": cfg,
           "scorer_vs_CHAMPION@100": {m: round(cfg["offset_scorer"][m][100] - cfg["CHAMPION"][m][100], 2)
                                      for m in ("recall", "fullcov")}}
    os.makedirs(os.path.join("data", "ukb_storage", dataset, "results", "L1"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "struct_scorer.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    c, ch = cfg["offset_scorer"], cfg["CHAMPION"]
    log.info(f"[{dataset}] FCOV@100 dense {cfg['dense']['fullcov'][100]} | CHAMP {ch['fullcov'][100]} | "
             f"SCORER {c['fullcov'][100]} (vs champ {out['scorer_vs_CHAMPION@100']['fullcov']:+}) | "
             f"scorer+champ {cfg['scorer+CHAMPION']['fullcov'][100]} || R@100 SCORER {c['recall'][100]}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Offset-signal set-transformer scorer (dynamic fusion of offset directions).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=8000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 OFFSET-SCORER: {ds.upper()} =====")
        try:
            run(ds, epochs=a.epochs, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
