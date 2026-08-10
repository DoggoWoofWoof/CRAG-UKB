"""
Query-encoder fine-tuning for partition routing (representation lever).
=======================================================================
The biggest untapped lever: everything else freezes the query encoder and only
trains a linear MLP on top of its frozen 384-d embedding. Here we FINE-TUNE the
encoder itself (mean-pooled) end-to-end against the KL routing loss over the
frozen partition centroids — giving the representation, not just a projection,
room to move.

CPU-feasibility: full fine-tune of the transformer is GPU-bound; by default we
unfreeze only the top `--unfreeze` layers (2) so a bounded local probe runs on
CPU. The full run (all layers, all data) is the Modal escalation. Compares the
fine-tuned encoder's coverage against the frozen-encoder baseline number.
Writes results/finetune_ablation/{dataset}.json + per-epoch logs.
"""
import os
import json
import csv
import random
import logging
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.core.engine import CoreEngine
from src.alignment.train_mlp import kl_div_loss
from src.evaluation.benchmark_partition_selection import COVERAGE_K_VALUES

log = logging.getLogger("experiments.encoder_finetune")

HF_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
SPLIT_SEED, TRAIN_RATIO, VAL_RATIO = 42, 0.70, 0.20
TAU = {"metaqa": 0.01, "2wiki": 0.07, "musique": 0.05, "squad": 0.1,
       "2wiki_clean": 0.07, "musique_clean": 0.05, "hotpotqa_clean": 0.07}
HNK = {"metaqa": 400, "2wiki": 149, "musique": 33, "squad": 189,
       "2wiki_clean": 657, "musique_clean": 135, "hotpotqa_clean": 660}   # 100-docs/partition substrate


def _splits(engine):
    pairs = []
    for node in engine.all_nodes:
        if node.metadata.get("type") == "question":
            gp = {int(engine.partition_map[nb]) for nb in node.neighbors if nb in engine.partition_map}
            if gp:
                pairs.append((node.node_id, node.content, sorted(gp)))
    pairs.sort(key=lambda x: x[0])
    random.Random(SPLIT_SEED).shuffle(pairs)
    n = len(pairs)
    tr, va = int(n * TRAIN_RATIO), int(n * TRAIN_RATIO) + int(n * VAL_RATIO)
    f = lambda s: [(t, p) for _, t, p in s]
    return {"train": f(pairs[:tr]), "val": f(pairs[tr:va]), "test": f(pairs[va:])}


class _Encoder(torch.nn.Module):
    def __init__(self, unfreeze=2):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(HF_MODEL)
        for p in self.model.parameters():
            p.requires_grad = False
        layers = self.model.encoder.layer
        for lyr in layers[-unfreeze:]:            # unfreeze only the top layers (CPU-feasible)
            for p in lyr.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, dim=-1)


def run_dataset(dataset, epochs=3, limit=0, unfreeze=2, device=None):
    from transformers import AutoTokenizer
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    log.info(f"===== ENCODER FINE-TUNE: {dataset.upper()} (unfreeze top {unfreeze} layers, tau={tau:g}) =====")

    engine = CoreEngine(source=dataset)
    centroids = np.array([engine.centroid_index.reconstruct(i)
                          for i in range(engine.centroid_index.ntotal)], dtype=np.float32)
    Cg = F.normalize(torch.tensor(centroids, dtype=torch.float32, device=device), dim=-1)
    splits = _splits(engine)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    if not splits["train"]:
        return None

    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    enc = _Encoder(unfreeze=unfreeze).to(device)
    opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=2e-5)
    bs = 16

    def _batch(texts):
        t = tok(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        return t["input_ids"].to(device), t["attention_mask"].to(device)

    def _eval(split):
        enc.eval()
        fc = {k: [] for k in COVERAGE_K_VALUES}
        gtr = {k: [] for k in COVERAGE_K_VALUES}
        maxk = max(COVERAGE_K_VALUES)
        with torch.no_grad():
            for i in range(0, len(split), 64):
                chunk = split[i:i + 64]
                ii, am = _batch([t for t, _ in chunk])
                sims = enc(ii, am) @ Cg.T
                top = torch.topk(sims, min(maxk, Cg.shape[0]), dim=1).indices.cpu().tolist()
                for j, (_, gp) in enumerate(chunk):
                    gt = set(gp)
                    for k in COVERAGE_K_VALUES:
                        tk = set(top[j][:k])
                        fc[k].append(1.0 if gt.issubset(tk) else 0.0)
                        gtr[k].append(len(gt & tk) / len(gt))
        return ({f"full_coverage@{k}": round(float(np.mean(fc[k])) * 100, 2) for k in COVERAGE_K_VALUES},
                {f"gt_recall@{k}": round(float(np.mean(gtr[k])) * 100, 2) for k in COVERAGE_K_VALUES})

    logs_dir = os.path.join("logs", dataset, f"encoder_finetune_uf{unfreeze}")
    os.makedirs(logs_dir, exist_ok=True)
    hist = os.path.join(logs_dir, "history.csv")
    with open(hist, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "test_full_coverage@20"])

    tr = splits["train"]
    for ep in range(epochs):
        enc.train()
        order = list(range(len(tr))); random.Random(ep).shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            ii, am = _batch([tr[i][0] for i in idx])
            proj = enc(ii, am)
            loss = kl_div_loss(proj, [tr[i][1] for i in idx], Cg, temperature=tau, hn_k=hn_k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in enc.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss); nb += 1
        fc, _ = _eval(splits["test"])
        with open(hist, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep + 1, round(tot / max(nb, 1), 6), fc["full_coverage@20"]])
        log.info(f"  epoch {ep+1}: train_loss={tot/max(nb,1):.4f} test_FCov@20={fc['full_coverage@20']}%")

    fc, gtr = _eval(splits["test"])
    out = {"dataset": dataset, "unfreeze": unfreeze, "tau": tau, "hn_k": hn_k,
           "finetuned": {**fc, **gtr, "n_test": len(splits["test"])}}
    out_dir = os.path.join("results", "finetune_ablation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_encoder_finetune.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"  final test FCov@20={fc['full_coverage@20']}% FCov@50={fc['full_coverage@50']}% "
             f"gtR@20={gtr['gt_recall@20']}% -> saved results/finetune_ablation/{dataset}_encoder_finetune.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Query-encoder fine-tuning for partition routing.")
    p.add_argument("--datasets", nargs="+", default=["2wiki", "musique"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--unfreeze", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        run_dataset(ds, epochs=a.epochs, limit=a.limit, unfreeze=a.unfreeze, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
