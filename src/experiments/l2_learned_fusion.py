"""
L2 learned-fusion vs. parameter-free best-of.
=============================================
Tests the "why not one optimized learned model instead of fusing everything?" critique.
A LEARNED per-query gate reads the query embedding and outputs softmax weights over the four
orthogonal signal rankings (dense, rel_hard, mlpT, SPLADE); the fused score of a candidate doc is
  score(doc) = sum_s w_s(q) * 1/(rank_s(doc) + 60)   (learned weighted RRF).
The gate is DATASET-AGNOSTIC (one model across all datasets). To keep it honest, we split each
dataset's test queries 50/50: the gate trains on half A and is evaluated on half B (never sees B's
golds). We compare its Recall@20 on B to the parameter-free best-of (min-rank) over the same signals.

If best-of >= the learned gate, the multi-signal best-of is JUSTIFIED as near-optimal (not
un-optimized) — a learned combiner cannot beat it, so we keep the trivial, tuning-free fusion.
If the gate wins, we should adopt a learned combiner. Writes results/L2/learned_fusion_{subdir}.json.
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

from src.experiments.l2_seed import (_load, _train_universal, _scoped_order, _splade_scoped_order,
                                      _topP, _bestof, _recall, MAXK, DATASETS)

log = logging.getLogger(__name__)
RRF_K = 60
POOL_M = 100
SIGS = ["dense", "rel_hard", "mlpT", "splade"]


class FusionGate(nn.Module):
    """query embedding -> softmax weights over the signals (per-query learned RRF weighting)."""
    def __init__(self, d, nsig):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, nsig))

    def forward(self, q):
        return F.softmax(self.net(q), dim=-1)


def _rankfeat(orders, qi):
    """Candidate pool (union of each signal's top-POOL_M) and its RRF features 1/(rank_s+K)."""
    pool = set()
    for s in SIGS:
        pool.update(int(x) for x in orders[s][qi][:POOL_M] if x >= 0)
    pool = list(pool)
    feat = np.zeros((len(pool), len(SIGS)), dtype="float32")
    for si, s in enumerate(SIGS):
        pos = {int(doc): r for r, doc in enumerate(orders[s][qi][:POOL_M]) if doc >= 0}
        for ci, doc in enumerate(pool):
            feat[ci, si] = 1.0 / (pos.get(doc, POOL_M) + RRF_K)
    return pool, feat


def run(datasets=None, subdir="gte_qwen", limit=8000, tr_cap=3000, te_cap=2000, epochs=20, K=8,
        scope_topk=50, gate_epochs=40, device=None):
    datasets = datasets or DATASETS
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, per_ds = {}, {}
    for d in datasets:
        data[d] = _load(d, subdir, limit, tr_cap, te_cap)
        idx = faiss.IndexFlatIP(data[d]["X"].shape[1]); idx.add(data[d]["X"])
        per_ds[d] = {"train": data[d]["train"], "Xt": torch.tensor(data[d]["X"], device=device),
                     "index": idx, "faiss": idx}
        log.info("  loaded %s X%s", d, data[d]["X"].shape)

    heads = {k: _train_universal(k, per_ds, device, epochs, K=K) for k in ("hard", "mix_hard")}
    MIX = {"mlpT"}

    ds_orders, per_q, qdim = {}, [], None
    for d in datasets:
        X_t = per_ds[d]["Xt"]; hard = data[d]["hard"]; hard_t = torch.tensor(hard, device=device)
        mem_idx = data[d]["mem_idx"]; npart = data[d]["npart"]
        qte, ste, gte = data[d]["test"]; qdim = qte.shape[1]
        _, I = per_ds[d]["faiss"].search(qte, MAXK)
        topP = ([set()] * len(qte)) if not scope_topk else _topP(I, mem_idx, npart, scope_topk)
        with torch.no_grad():
            qt = torch.tensor(qte, device=device); sv = X_t[torch.tensor(ste, device=device)]
            pos = {"dense": qt, "rel_hard": heads["hard"](qt, sv), "mlpT": heads["mix_hard"](qt, sv)}
        orders = {m: _scoped_order(pos[m].cpu(), X_t, hard_t, topP, m in MIX, device)[0] for m in pos}
        sp = data[d].get("splade")
        orders["splade"] = (_splade_scoped_order(sp, data[d]["test_texts"], hard, topP, dataset=d)
                            if sp is not None else orders["dense"])
        ds_orders[d] = (orders, gte, len(qte))
        rng = np.random.RandomState(0); perm = rng.permutation(len(qte)); Aset = set(perm[:len(qte) // 2].tolist())
        for qi in range(len(qte)):
            if not gte[qi]:
                continue
            pool, feat = _rankfeat(orders, qi)
            if not pool:
                continue
            per_q.append({"q": qte[qi], "pool": pool, "feat": feat,
                          "gold": set(int(g) for g in gte[qi]), "d": d, "qi": qi, "A": qi in Aset})
        del hard_t
    log.info("pooled %d queries across %d datasets (gate is dataset-agnostic)", len(per_q), len(datasets))

    gate = FusionGate(qdim, len(SIGS)).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    A = [r for r in per_q if r["A"]]
    for ep in range(gate_epochs):
        random.Random(ep).shuffle(A); tot = 0.0; nb = 0
        for r in A:
            gold_mask = torch.tensor([1.0 if doc in r["gold"] else 0.0 for doc in r["pool"]], device=device)
            if gold_mask.sum() == 0:
                continue
            w = gate(torch.tensor(r["q"], device=device).unsqueeze(0))[0]
            score = torch.tensor(r["feat"], device=device) @ w
            loss = -(F.log_softmax(score, dim=0) * gold_mask).sum() / gold_mask.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % 10 == 0:
            log.info("  gate epoch %d loss=%.4f", ep, tot / max(nb, 1))

    gate.eval()
    out = {}
    mean_w = {d: np.zeros(len(SIGS)) for d in datasets}; cnt = {d: 0 for d in datasets}
    learned_by_ds = {d: {} for d in datasets}
    with torch.no_grad():
        for r in per_q:
            if r["A"]:
                continue
            w = gate(torch.tensor(r["q"], device=device).unsqueeze(0))[0]
            score = (torch.tensor(r["feat"], device=device) @ w).cpu().numpy()
            ranked = [r["pool"][j] for j in np.argsort(-score)]
            learned_by_ds[r["d"]][r["qi"]] = ranked
            mean_w[r["d"]] += w.cpu().numpy(); cnt[r["d"]] += 1

    for d in datasets:
        orders, gte, nq = ds_orders[d]
        Bset = set(learned_by_ds[d].keys())
        bo = _bestof([orders["dense"], orders["rel_hard"], orders["mlpT"], orders["splade"]])
        gte_B = [gte[qi] if qi in Bset else [] for qi in range(nq)]
        lo_B = [learned_by_ds[d].get(qi, []) for qi in range(nq)]
        bo_B = [bo[qi] if qi in Bset else [] for qi in range(nq)]
        out[d] = {"n_B": len(Bset),
                  "learned_gate": _recall(lo_B, gte_B),
                  "bestof": _recall(bo_B, gte_B),
                  "mean_gate_w": {SIGS[i]: round(float(mean_w[d][i] / max(cnt[d], 1)), 3) for i in range(len(SIGS))}}
        lg, bf = out[d]["learned_gate"][20], out[d]["bestof"][20]
        log.info("[%s] learned_gate R@20=%.2f vs best-of R@20=%.2f (delta %+.2f, nB=%d) w=%s",
                 d, lg, bf, lg - bf, len(Bset), out[d]["mean_gate_w"])

    os.makedirs("results/L2", exist_ok=True)
    path = f"results/L2/learned_fusion_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L2 learned-fusion gate vs parameter-free best-of.")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--scope-topk", type=int, default=50)
    p.add_argument("--gate-epochs", type=int, default=40)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, scope_topk=a.scope_topk, gate_epochs=a.gate_epochs)


if __name__ == "__main__":
    main()
