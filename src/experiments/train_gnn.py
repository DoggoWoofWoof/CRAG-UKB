"""
Corrected GNN training for partition routing (fixes the no-grad conv bug).
==========================================================================
The legacy train_alignment.py GNN path computes partition embeddings under
`torch.no_grad()` and the query path (`project_text`) never calls the conv
layers, so the GIN/GCN/SAGE convolutions received ZERO gradient and stayed at
random init — the "GNN underperforms" numbers were an untrained-conv artifact,
not evidence of oversmoothing.

This trains the convolutions properly via Jigsaw-style LIVE POSITIVES: each step
re-encodes the batch's gold partition subgraphs WITH gradient and splices them
into a cached (detached) partition-embedding matrix used for the negatives. So
gradient flows through the convs (via the live positives) while cost stays
bounded. Node features = the same frozen text embeddings the MLP sees, so the
only added ingredient is message passing — a fair test of whether structure
helps. Trains gin/gcn/sage, evaluates coverage, writes results/gnn_ablation/.
"""
import os
import json
import csv
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.gnn_encoders import GINAlignmentEncoder, GCNAlignmentEncoder, SAGEAlignmentEncoder
from src.alignment.train_alignment import get_partition_subgraphs
from src.alignment.train_mlp import kl_div_loss
from src.evaluation.benchmark_partition_selection import compute_multi_gt_metrics, COVERAGE_K_VALUES

log = logging.getLogger("experiments.train_gnn")

ENCODERS = {"gin": GINAlignmentEncoder, "gcn": GCNAlignmentEncoder, "sage": SAGEAlignmentEncoder}
SPLIT_SEED, TRAIN_RATIO, VAL_RATIO = 42, 0.70, 0.20
TAU = {"metaqa": 0.01, "2wiki": 0.07, "musique": 0.05, "squad": 0.1,
       "2wiki_clean": 0.07, "musique_clean": 0.05, "hotpotqa_clean": 0.07}
HNK = {"metaqa": 400, "2wiki": 149, "musique": 33, "squad": 189,
       "2wiki_clean": 657, "musique_clean": 135, "hotpotqa_clean": 660}   # 100-docs/partition substrate


def _node_features(engine):
    idx = engine.node_index
    return torch.tensor(np.array([idx.reconstruct(i) for i in range(idx.ntotal)], dtype=np.float32))


def _partition_list(engine, npart):
    pl = [0] * len(engine.nodes)
    for i, node in enumerate(engine.nodes):
        p = engine.partition_map.get(node.node_id)
        if p is not None:
            pl[i] = int(p)
    return pl


def _splits(engine):
    pairs = []
    for node in engine.all_nodes:
        if node.metadata.get("type") == "question":
            gp = sorted({int(engine.partition_map[nb]) for nb in node.neighbors if nb in engine.partition_map})
            if gp:
                pairs.append((node.node_id, node.content, gp))
    pairs.sort(key=lambda x: x[0])
    random.Random(SPLIT_SEED).shuffle(pairs)
    n = len(pairs)
    tr, va = int(n * TRAIN_RATIO), int(n * TRAIN_RATIO) + int(n * VAL_RATIO)
    f = lambda s: [(c, p) for _, c, p in s]
    return {"train": f(pairs[:tr]), "val": f(pairs[tr:va]), "test": f(pairs[va:])}


def _all_partition_embs(model, subgraphs, device, grad=False):
    b = Batch.from_data_list(subgraphs).to(device)
    if grad:
        return F.normalize(model(b.x, b.edge_index, b.batch), dim=-1)
    with torch.no_grad():
        return F.normalize(model(b.x, b.edge_index, b.batch), dim=-1)


def _train_one(gnn_name, subgraphs, splits, split_embs, npart, device, tau, hn_k, epochs, logs_dir):
    model = ENCODERS[gnn_name](in_channels=subgraphs[0].x.shape[1], hidden_channels=256, out_channels=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=7)
    tr, va = splits["train"], splits["val"]
    tr_e, va_e = split_embs["train"], split_embs["val"]
    bs = 64
    os.makedirs(logs_dir, exist_ok=True)
    hist = os.path.join(logs_dir, "history.csv")
    with open(hist, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "is_best"])

    best_val, best_state, final_state = float("inf"), None, None
    no_imp = 0
    for ep in range(epochs):
        cached = _all_partition_embs(model, subgraphs, device, grad=False)  # detached negatives
        model.train()
        order = list(range(len(tr))); random.Random(ep).shuffle(order)
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            pids_list = [tr[i][1] for i in idx]
            gold = sorted({p for pids in pids_list for p in pids})
            live = _all_partition_embs(model, [subgraphs[p] for p in gold], device, grad=True)  # convs get grad
            part_mat = cached.clone()
            part_mat[torch.tensor(gold, device=device)] = live       # splice live positives
            q = F.normalize(model.project_text(torch.tensor(tr_e[idx], dtype=torch.float32, device=device), device), dim=-1)
            loss = kl_div_loss(q, pids_list, part_mat, temperature=tau, hn_k=hn_k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1

        # val loss (query vs freshly cached partition embs)
        model.eval()
        cached_v = _all_partition_embs(model, subgraphs, device, grad=False)
        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), bs):
                chunk = va[s:s + bs]
                q = F.normalize(model.project_text(torch.tensor(va_e[s:s + bs], dtype=torch.float32, device=device), device), dim=-1)
                vtot += float(kl_div_loss(q, [p for _, p in chunk], cached_v, temperature=tau, hn_k=hn_k)); vnb += 1
        vl = vtot / max(vnb, 1); sched.step(vl)
        improved = vl < best_val
        if improved:
            best_val = vl; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        with open(hist, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep + 1, round(tot / max(nb, 1), 6), round(vl, 6), improved])
        if no_imp >= 20:
            break
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, best_state, final_state


def _eval(model, state, subgraphs, test, test_e, npart, device):
    model.load_state_dict(state); model.eval()
    part = _all_partition_embs(model, subgraphs, device, grad=False)
    maxk = max(COVERAGE_K_VALUES)
    agg = {}
    rows = []
    with torch.no_grad():
        for s in range(0, len(test), 256):
            chunk = test[s:s + 256]
            q = F.normalize(model.project_text(torch.tensor(test_e[s:s + 256], dtype=torch.float32, device=device), device), dim=-1)
            sims = q @ part.T
            top = torch.argsort(-sims, dim=1)[:, :max(maxk, npart)].cpu().tolist()
            for j, (_, gp) in enumerate(chunk):
                rows.append(compute_multi_gt_metrics(top[j], gp, num_partitions=npart))
    keys = [f"full_coverage@{k}" for k in COVERAGE_K_VALUES] + [f"recall@{k}" for k in (1, 20)] + ["mrr", "weakest_positive_rank"]
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if not vals:
            continue
        agg[k] = round(float(np.mean(vals)) * (100 if k != "weakest_positive_rank" else 1), 2)
    agg["n_test"] = len(rows)
    return agg


def run_dataset(dataset, models=("gin", "gcn", "sage"), epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    log.info(f"===== GNN (fixed training) : {dataset.upper()} (tau={tau:g}, hn_k={hn_k}) =====")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    node_feats = _node_features(engine)
    part_list = _partition_list(engine, npart)
    subgraphs = get_partition_subgraphs(engine.graph.edge_index, node_feats, part_list, npart)
    splits = _splits(engine)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([c for c, _ in splits[s]]).astype("float32") for s in splits if splits[s]}

    results = {}
    for gnn_name in models:
        logs_dir = os.path.join("logs", dataset, f"gnn_fixed_{gnn_name}")
        model, best_state, final_state = _train_one(
            gnn_name, subgraphs, splits, split_embs, npart, device, tau, hn_k, epochs, logs_dir)
        m_final = _eval(model, final_state, subgraphs, splits["test"], split_embs["test"], npart, device)
        m_best = _eval(model, best_state, subgraphs, splits["test"], split_embs["test"], npart, device)
        results[gnn_name] = {"best": m_best, "final": m_final}
        log.info(f"  [{gnn_name}] final FCov@20={m_final.get('full_coverage@20')}% "
                 f"R@1={m_final.get('recall@1')}% MRR={m_final.get('mrr')}%")

    out_dir = os.path.join("results", "gnn_ablation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_gnn.json"), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "note": "convs trained via live-positive backprop (fixes no-grad bug)",
                   "tau": tau, "hn_k": hn_k, "results": results}, f, indent=2)
    log.info(f"Saved results/gnn_ablation/{dataset}_gnn.json")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="Corrected GNN training (live-positive backprop).")
    p.add_argument("--datasets", nargs="+", default=["2wiki", "musique"])
    p.add_argument("--models", nargs="+", default=["gin", "gcn", "sage"], choices=["gin", "gcn", "sage"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        run_dataset(ds, models=tuple(a.models), epochs=a.epochs, limit=a.limit, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
