"""
Relation-offset ROUTING integration (phase 21 payoff) — naive + learned.
========================================================================
Probe showed learned relation offsets recover relational neighbours (king->queen).
Goal: make the L1 router reach relationally-distant gold partitions WITHOUT
query-time PPR.

  mode=offset : NAIVE — at route time add ALL K offsets and take the max. FAILED
                (metaqa FullCov 42->11): broadcasting every relation direction to
                every query scatters routing. Kept as the negative baseline.
  mode=learned: the fix — a query-conditioned ATTENTION over the K directions plus
                a learned gate, trained end-to-end with the routing (KL) loss, so
                the router applies only the RIGHT relation shift per query (cheap:
                one Linear head + K fixed offsets, no heavy model). Compares a plain
                MLP router vs this relation-aware router at matched partitions.
Writes data/ukb_storage/{ds}/results/L1/relation_route[_learned].json.
"""
import os
import csv
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
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_mlp import kl_div_loss
from src.experiments.overlap_retrain import (
    _reconstruct, _hard_membership, _centroids, _splits, _train, membership_ref,
    TAU, HNK, INIT_SEED,
)
from src.experiments.l3_relation_probe import _rel_edges
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.relation_route")
KS = [5, 10, 20, 50, 100]


def _learn_offsets(X, engine, id2idx, K, sample=20000):
    A, B = _rel_edges(engine, id2idx)
    if len(A) == 0:
        return np.zeros((0, X.shape[1]), np.float32)
    rng = np.random.RandomState(0)
    sel = rng.choice(len(A), min(sample, len(A)), replace=False)
    deltas = (X[B[sel]] - X[A[sel]]).astype("float32")
    km = faiss.Kmeans(X.shape[1], K, niter=20, seed=0, verbose=False)
    km.train(deltas)
    return km.centroids.reshape(K, X.shape[1]).astype("float32")


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
    return ({k: round(float(np.mean(fc[k])) * 100, 2) for k in KS},
            {k: round(float(np.mean(gtr[k])) * 100, 2) for k in KS})


class AttnRelRouter(nn.Module):
    """Plain router + query-conditioned attention over K fixed relation offsets."""
    def __init__(self, d_in, d_out, R):
        super().__init__()
        self.base = TextPartitionMLP(input_dim=d_in, hidden_dim=512, output_dim=d_out)
        self.register_buffer("R", torch.tensor(R))                 # (K, d_out)
        self.att = nn.Linear(d_in, R.shape[0])
        self.gate = nn.Parameter(torch.tensor(-1.5))               # small initial shift (sigmoid ~0.18)

    def forward(self, q):
        proj = F.normalize(self.base(q), dim=-1)                   # (B, d_out)
        w = torch.softmax(self.att(q), dim=-1)                     # (B, K)
        shift = w @ self.R                                         # (B, d_out)
        return F.normalize(proj + torch.sigmoid(self.gate) * shift, dim=-1)


def _train_attn(model, splits, split_embs, Cg, device, tau, hn_k, epochs, name, membership):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tr, va = splits["train"], splits["val"]; tr_e, va_e = split_embs["train"], split_embs["val"]
    bs = 64; best, best_state, noimp = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); order = list(range(len(tr))); random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            embs = torch.tensor(tr_e[idx], dtype=torch.float32, device=device)
            proj = model(embs)
            loss = kl_div_loss(proj, [tr[i][1] for i in idx], Cg, temperature=tau, hn_k=hn_k)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), bs):
                embs = torch.tensor(va_e[s:s + bs], dtype=torch.float32, device=device)
                proj = model(embs)
                tot += float(kl_div_loss(proj, [p for _, p, _ in va[s:s + bs]], Cg, temperature=tau, hn_k=hn_k)); nb += 1
        vl = tot / max(nb, 1)
        if vl < best:
            best, best_state, noimp = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            noimp += 1
        if noimp >= 15:
            break
    model.load_state_dict(best_state); model.eval()
    return model


def _order(scores):
    return np.argsort(-scores, axis=1)


def run_learned(dataset, K=16, epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    X = node_vecs.astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    membership = _hard_membership(engine); membership_ref["hard"] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    D = C.shape[1]
    Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1)
    R = _learn_offsets(X, engine, id2idx, K)
    test, test_e = splits["test"], split_embs["test"]

    # baseline plain router (KL, full-softmax) — same as production L1
    _, base_best, _, Cg2 = _train(engine, C, splits, split_embs, device, tau, hn_k, epochs,
                                  os.path.join("data", "ukb_storage", dataset, "results", "L1", "_relroute_base"),
                                  "hard", "kl")
    base = TextPartitionMLP(input_dim=D, hidden_dim=512, output_dim=D).to(device)
    base.load_state_dict(base_best); base.eval()
    with torch.no_grad():
        p_scores = (F.normalize(base(torch.tensor(test_e, device=device)), dim=-1) @ Cg.T).cpu().numpy()
    p_fc, p_gt = _cov(_order(p_scores), test, membership, npart)

    # learned relation-aware router
    torch.manual_seed(INIT_SEED)
    model = AttnRelRouter(D, D, R).to(device)
    model = _train_attn(model, splits, split_embs, Cg, device, tau, hn_k, epochs, "rel", membership)
    with torch.no_grad():
        l_scores = (model(torch.tensor(test_e, device=device)) @ Cg.T).cpu().numpy()
    l_fc, l_gt = _cov(_order(l_scores), test, membership, npart)

    out = {"dataset": dataset, "npart": npart, "K_offsets": R.shape[0], "n_test": len(test),
           "gate": round(float(torch.sigmoid(model.gate).item()), 3),
           "plain_fullcov": p_fc, "learned_fullcov": l_fc,
           "plain_gt_recall": p_gt, "learned_gt_recall": l_gt,
           "fullcov_lift": {k: round(l_fc[k] - p_fc[k], 2) for k in KS}}
    with open(rpath(dataset, "L1", "relation_route_learned"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] LEARNED FullCov plain {p_fc} -> learned {l_fc} | lift {out['fullcov_lift']} | gate {out['gate']}")
    return out


def run_graphtrain(dataset, lam=0.3, epochs=100, limit=0, device=None):
    """Utilize the graph IN TRAINING: base router KL loss + an auxiliary graph-
    contrastive term (relationally-connected docs pushed together in the routing
    space vs other docs). Inference is PLAIN query routing (no graph/offsets at
    query time) — the point is the LEARNED space internalizes the relations. If the
    graph-regularized router beats the plain router (esp metaqa), the graph made the
    MLP learn better without a query-time crutch."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    X = node_vecs.astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    membership = _hard_membership(engine); membership_ref["hard"] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    D = C.shape[1]
    Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1)
    A, B = _rel_edges(engine, id2idx)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    test, test_e = splits["test"], split_embs["test"]

    # baseline: plain KL router
    _, base_best, _, _ = _train(engine, C, splits, split_embs, device, tau, hn_k, epochs,
                               os.path.join("data", "ukb_storage", dataset, "results", "L1", "_gt_base"), "hard", "kl")
    base = TextPartitionMLP(input_dim=D, hidden_dim=512, output_dim=D).to(device); base.load_state_dict(base_best); base.eval()
    with torch.no_grad():
        p_scores = (F.normalize(base(torch.tensor(test_e, device=device)), dim=-1) @ Cg.T).cpu().numpy()
    p_fc, p_gt = _cov(_order(p_scores), test, membership, npart)

    # graph-regularized router
    torch.manual_seed(INIT_SEED)
    model = TextPartitionMLP(input_dim=D, hidden_dim=512, output_dim=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    tr, va = splits["train"], splits["val"]; tr_e, va_e = split_embs["train"], split_embs["val"]
    bs, ebs = 64, 256; best, best_state, noimp = float("inf"), None, 0
    rng = np.random.RandomState(0)
    for ep in range(epochs):
        model.train(); order = list(range(len(tr))); random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            proj = F.normalize(model(torch.tensor(tr_e[idx], dtype=torch.float32, device=device)), dim=-1)
            loss = kl_div_loss(proj, [tr[i][1] for i in idx], Cg, temperature=tau, hn_k=hn_k)
            if len(A) and lam > 0:                                  # graph-contrastive aux (edges as positives)
                e = rng.choice(len(A), min(ebs, len(A)), replace=False)
                Pa = F.normalize(model(Xt[A[e]]), dim=-1); Pb = F.normalize(model(Xt[B[e]]), dim=-1)
                sim = Pa @ Pb.T / 0.1
                loss = loss + lam * F.cross_entropy(sim, torch.arange(len(e), device=device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va), bs):
                proj = F.normalize(model(torch.tensor(va_e[s:s + bs], dtype=torch.float32, device=device)), dim=-1)
                tot += float(kl_div_loss(proj, [p for _, p, _ in va[s:s + bs]], Cg, temperature=tau, hn_k=hn_k)); nb += 1
        vl = tot / max(nb, 1)
        if vl < best:
            best, best_state, noimp = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            noimp += 1
        if noimp >= 15:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        g_scores = (F.normalize(model(torch.tensor(test_e, device=device)), dim=-1) @ Cg.T).cpu().numpy()
    g_fc, g_gt = _cov(_order(g_scores), test, membership, npart)

    out = {"dataset": dataset, "npart": npart, "lambda": lam, "n_test": len(test), "n_rel_edges": int(len(A)),
           "plain_fullcov": p_fc, "graphreg_fullcov": g_fc, "plain_gt_recall": p_gt, "graphreg_gt_recall": g_gt,
           "fullcov_lift": {k: round(g_fc[k] - p_fc[k], 2) for k in KS},
           "gt_recall_lift": {k: round(g_gt[k] - p_gt[k], 2) for k in KS}}
    with open(rpath(dataset, "L1", "relation_graphtrain"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] GRAPHTRAIN FullCov plain {p_fc} -> graphreg {g_fc} | lift {out['fullcov_lift']}")
    return out


def run_offset(dataset, K=16, epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    X = node_vecs.astype("float32"); faiss.normalize_L2(X)
    id2idx = engine.node_id_to_idx
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    membership = _hard_membership(engine); membership_ref["hard"] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    model, best, _, Cg = _train(engine, C, splits, split_embs, device, tau, hn_k, epochs,
                                os.path.join("data", "ukb_storage", dataset, "results", "L1", "_relroute_train"),
                                "hard", "kl")
    model.load_state_dict(best); model.eval()
    R = _learn_offsets(X, engine, id2idx, K)
    test, test_e = splits["test"], split_embs["test"]
    Cgn = Cg.cpu().numpy()
    with torch.no_grad():
        proj = F.normalize(model(torch.tensor(test_e, device=device)), dim=-1).cpu().numpy()
    plain = proj @ Cgn.T; aug = plain.copy()
    for k in range(R.shape[0]):
        pk = proj + R[k]; pk = pk / (np.linalg.norm(pk, axis=1, keepdims=True) + 1e-9)
        aug = np.maximum(aug, pk @ Cgn.T)
    p_fc, p_gt = _cov(_order(plain), test, membership, npart)
    a_fc, a_gt = _cov(_order(aug), test, membership, npart)
    out = {"dataset": dataset, "npart": npart, "K_offsets": R.shape[0], "n_test": len(test),
           "plain_fullcov": p_fc, "offset_fullcov": a_fc, "plain_gt_recall": p_gt, "offset_gt_recall": a_gt,
           "fullcov_lift": {k: round(a_fc[k] - p_fc[k], 2) for k in KS}}
    with open(rpath(dataset, "L1", "relation_route"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] OFFSET(naive) FullCov plain {p_fc} -> {a_fc} | lift {out['fullcov_lift']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Relation-offset routing (naive vs learned attention).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "2wiki_clean"])
    p.add_argument("--mode", choices=["offset", "learned", "graphtrain"], default="graphtrain")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== RELATION ROUTING [{a.mode}]: {ds.upper()} =====")
        if a.mode == "graphtrain":
            run_graphtrain(ds, lam=a.lam, epochs=a.epochs, limit=a.limit)
        elif a.mode == "learned":
            run_learned(ds, K=a.K, epochs=a.epochs, limit=a.limit)
        else:
            run_offset(ds, K=a.K, epochs=a.epochs, limit=a.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
