"""
Full-stack Level-1 combination — put every lever together and measure what stacks.
==================================================================================
Levers established so far (2wiki FCov@20): membership overlap+kNN is THE lever
(54 -> 83 graph -> 90 +knn1 -> 94 +knn3); KL+HNM is the best router loss (coverage
null); GNN is worse; adaptive-K is a minor tweak. The one interaction never tested
in combination is the REPRESENTATION lever: does fine-tuning the query encoder help
*on top of* the best overlap membership, or is it redundant/confounded?

This harness fixes ONE best membership config and races two front-ends on it with
an identical, overlap-correct (per-gold-doc any-hit) coverage eval:
  A. frozen encoder -> TextPartitionMLP -> centroids   (the current recommended stack)
  B. fine-tuned encoder (top-N layers) -> centroids     (representation lever)
Both trained with the same KL(+HNM) objective against the same rebuilt centroids.
Then it drops in the oracle adaptive-K point on the winner. Reports the assembled
best combination. Writes results/stack_best/{dataset}_{config}.json.
"""
import os
import json
import math
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_mlp import kl_div_loss
from src.experiments.overlap_retrain import _reconstruct, _centroids, _splits, TAU, HNK
from src.experiments.adaptive_k import _build, _reverse
from src.experiments.encoder_finetune import _Encoder, HF_MODEL
from src.evaluation.benchmark_partition_selection import COVERAGE_K_VALUES

log = logging.getLogger("experiments.stack_best")
SEED = 42


def _cov_from_ranked(ranked, gold_docs_list, membership, ks):
    """Per-gold-doc any-hit FullCov@k (the overlap-correct metric) for each k."""
    out = {}
    for k in ks:
        vals = []
        for ranks, golds in zip(ranked, gold_docs_list):
            topk = set(ranks[:k])
            gms = [membership[g] for g in golds if g in membership]
            vals.append(1.0 if gms and all(ms & topk for ms in gms) else 0.0)
        out[f"full_coverage@{k}"] = round(float(np.mean(vals)) * 100, 2)
    return out


def _rank_mlp(model, Cg, test_e, device, npart):
    ranked = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_e), 256):
            embs = torch.tensor(test_e[i:i + 256], dtype=torch.float32, device=device)
            proj = F.normalize(model(embs), dim=-1)
            top = torch.argsort(-(proj @ Cg.T), dim=1).cpu().tolist()
            ranked.extend(top)
    return ranked


def _train_mlp_router(C, splits, split_embs, tau, hn_k, epochs, device):
    Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1)
    D = C.shape[1]
    model = TextPartitionMLP(input_dim=D, hidden_dim=512, output_dim=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tr, tr_e = splits["train"], split_embs["train"]
    bs = 64
    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        order = list(range(len(tr))); random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            embs = torch.tensor(tr_e[idx], dtype=torch.float32, device=device)
            proj = F.normalize(model(embs), dim=-1)
            loss = kl_div_loss(proj, [tr[i][1] for i in idx], Cg, temperature=tau, hn_k=hn_k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # cheap val-loss early-model keep
        model.eval(); tot = 0.0; nb = 0
        va, va_e = splits["val"], split_embs["val"]
        with torch.no_grad():
            for s in range(0, len(va), bs):
                embs = torch.tensor(va_e[s:s + bs], dtype=torch.float32, device=device)
                proj = F.normalize(model(embs), dim=-1)
                tot += float(kl_div_loss(proj, [p for _, p, _ in va[s:s + bs]], Cg, temperature=tau, hn_k=hn_k)); nb += 1
        vl = tot / max(nb, 1)
        if vl < best_val:
            best_val = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, Cg


def _train_ft_encoder(C, splits, tau, hn_k, epochs, unfreeze, device):
    """Fine-tune the query encoder against the (overlap) centroids; return ranked test list."""
    from transformers import AutoTokenizer
    Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1)
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    enc = _Encoder(unfreeze=unfreeze).to(device)
    opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=2e-5)
    bs = 16

    def _batch(texts):
        t = tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        return t["input_ids"].to(device), t["attention_mask"].to(device)

    tr = splits["train"]
    for ep in range(epochs):
        enc.train()
        order = list(range(len(tr))); random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            ii, am = _batch([tr[i][0].content for i in idx])
            proj = enc(ii, am)
            loss = kl_div_loss(proj, [tr[i][1] for i in idx], Cg, temperature=tau, hn_k=hn_k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in enc.parameters() if p.requires_grad], 1.0); opt.step()
        log.info(f"    [ft-enc] epoch {ep+1}/{epochs} done")

    enc.eval()
    test = splits["test"]
    ranked = []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            ii, am = _batch([q.content for q, _, _ in test[i:i + 64]])
            top = torch.argsort(-(enc(ii, am) @ Cg.T), dim=1).cpu().tolist()
            ranked.extend(top)
    return ranked


def run_dataset(dataset, config="overlap1+knn1", epochs=100, ft_epochs=5, unfreeze=2, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    log.info(f"===== STACK-BEST: {dataset.upper()} config={config} (tau={tau:g}, hn_k={hn_k}) =====")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1

    membership = _build(engine, node_vecs, config)
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    test = splits["test"]
    gold_docs_list = [golds for _, _, golds in test]
    gcount = [len(pids) for _, pids, _ in test]
    rev = _reverse(membership, npart)
    ks = COVERAGE_K_VALUES

    # ── Router A: frozen encoder + MLP ──
    mlp, Cg = _train_mlp_router(C, splits, split_embs, tau, hn_k, epochs, device)
    ranked_mlp = _rank_mlp(mlp, Cg, split_embs["test"], device, npart)
    cov_mlp = _cov_from_ranked(ranked_mlp, gold_docs_list, membership, ks)

    # ── Router B: fine-tuned encoder ──
    ranked_ft = _train_ft_encoder(C, splits, tau, hn_k, ft_epochs, unfreeze, device)
    cov_ft = _cov_from_ranked(ranked_ft, gold_docs_list, membership, ks)

    # ── adaptive-K oracle on the better router (matched to fixed K=20 avg pool) ──
    better = "mlp" if cov_mlp["full_coverage@20"] >= cov_ft["full_coverage@20"] else "ft"
    ranked_best = ranked_mlp if better == "mlp" else ranked_ft

    def _pool_cov(kpq):
        pools, covs = [], []
        for qi, ranks in enumerate(ranked_best):
            topk = set(ranks[:max(1, int(kpq[qi]))])
            pool = set()
            for p in topk:
                pool |= rev[p]
            pools.append(len(pool))
            gms = [membership[g] for g in gold_docs_list[qi] if g in membership]
            covs.append(1.0 if gms and all(ms & topk for ms in gms) else 0.0)
        return float(np.mean(pools)), round(float(np.mean(covs)) * 100, 2)

    fixed20_pool, fixed20_cov = _pool_cov([20] * len(ranked_best))
    adaptive = []
    for s in [3, 5, 8, 12, 20, 30]:
        kpq = [min(npart, math.ceil(s * max(1, g))) for g in gcount]
        ap, fc = _pool_cov(kpq)
        adaptive.append({"scale": s, "avg_pool": round(ap, 1), "full_coverage": fc})
    # best adaptive point at <= fixed-K=20 pool (same-or-cheaper budget)
    cheaper = [a for a in adaptive if a["avg_pool"] <= fixed20_pool]
    adapt_at_budget = max(cheaper, key=lambda a: a["full_coverage"]) if cheaper else None

    out = {
        "dataset": dataset, "config": config, "tau": tau, "hn_k": hn_k,
        "avg_gold_partitions": round(float(np.mean(gcount)), 3), "npart": npart,
        "router_frozen_mlp": cov_mlp,
        "router_finetuned_encoder": {**cov_ft, "unfreeze": unfreeze, "ft_epochs": ft_epochs},
        "adaptive_k_on_best": {"better_router": better, "fixed20_pool": round(fixed20_pool, 1),
                               "fixed20_cov": fixed20_cov, "sweep": adaptive,
                               "best_at_fixed20_budget": adapt_at_budget},
    }
    out_dir = os.path.join("results", "stack_best")
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{dataset}_{config.replace('+', '_')}.json"
    with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"  [{config}] frozen+MLP FCov@20={cov_mlp['full_coverage@20']}%  "
             f"FT-enc FCov@20={cov_ft['full_coverage@20']}%  "
             f"adaptive@budget={adapt_at_budget}")
    log.info(f"Saved results/stack_best/{fn}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Full-stack best-combination harness.")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--configs", nargs="+", default=["overlap1+knn1"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--ft_epochs", type=int, default=5)
    p.add_argument("--unfreeze", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        for cfg in a.configs:
            run_dataset(ds, config=cfg, epochs=a.epochs, ft_epochs=a.ft_epochs,
                        unfreeze=a.unfreeze, limit=a.limit, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
