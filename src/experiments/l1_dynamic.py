"""
L1 dynamic: trained multi-hop offset + regularized per-query fusion gate.
=========================================================================
Two levers the ablation flagged as the remaining high-value bets, and the two
components that build toward a per-query CONTROLLER (the end goal):

PART A — TRAINED 2-hop offset (attacks the `neither` gap: metaqa 70%, 2wiki 31%).
  The cheap iterated hop (reuse g) failed. Here a SECOND head g2 is trained to hop
  from the intermediate the first hop retrieves:
    hop1: pos1 = seed + g1(q)         ; s1 = top-1 doc near pos1   (intermediate)
    hop2: pos2 = X[s1] + g2(q)        ; retrieve near pos2         (the chain's end)
  g1 = trained 1-hop (curriculum, frozen); g2 trained InfoNCE(pos2, gold). Report
  whether hop1 UNION hop2 shrinks `neither` vs 1-hop alone (the compositional win).

PART B — REGULARIZED fusion gate (turns the fusion-weight LAW into a learned head).
  The plain gate saturates alpha->1 (drops dense's unique golds). Fix: train alpha on
  the ACTUAL fusion objective — a listwise loss over the dense∪rel candidate UNION,
  so dropping dense is penalized whenever dense holds a gold rel misses. alpha=sigmoid(gate(q))
  is then per-query and shouldn't saturate. Compare to the fixed wRRF sweep.

These are the pieces of the eventual dynamic controller: one MLP that, per query,
picks hop depth + fusion weights (+ later, partition routing). Writes ablation-style
JSON to data/ukb_storage/{ds}/results/L1/dynamic.json.
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
from src.experiments.query_relation import OffsetHead
from src.experiments.l1_ablate import _order, _ranks, _rrf_fuse, _recall, _overlap, _train_offset, KS, MAXK, TAU

log = logging.getLogger("experiments.l1_dynamic")


# --------------------------------------------------------------------------- Part A
def _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs):
    """Train g2 to hop from the intermediate g1 retrieves to the (chain-end) gold."""
    with torch.no_grad():
        pos1 = g1(torch.tensor(q_tr, device=device), Xt[[int(s) for s in seed_tr]]).cpu().numpy()
    _, s1 = index.search(np.ascontiguousarray(pos1.astype("float32")), 1)
    s1 = s1[:, 0]
    trip = [(i, int(s1[i]), int(g)) for i, gl in enumerate(gold_tr) for g in gl]
    qtr = torch.tensor(q_tr, device=device)
    torch.manual_seed(INIT_SEED + 1)
    g2 = OffsetHead(Xt.shape[1]).to(device); opt = torch.optim.Adam(g2.parameters(), lr=1e-3)
    for ep in range(epochs):
        g2.train(); random.Random(ep).shuffle(trip)
        for s in range(0, len(trip), 256):
            b = trip[s:s + 256]
            pred = g2(qtr[[t[0] for t in b]], Xt[[t[1] for t in b]])
            goldv = Xt[[t[2] for t in b]]
            loss = F.cross_entropy(pred @ goldv.T / TAU, torch.arange(len(b), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    g2.eval()
    return g2


def _multihop_orders(g1, g2, q_te, seed_te0, X, Xt, index, device):
    with torch.no_grad():
        pos1 = g1(torch.tensor(q_te, device=device), Xt[[int(s) for s in seed_te0]]).cpu().numpy()
    hop1 = _order(pos1, index)
    s1 = hop1[:, 0]
    with torch.no_grad():
        pos2 = g2(torch.tensor(q_te, device=device), Xt[[int(s) for s in s1]]).cpu().numpy()
    hop2 = _order(pos2, index)
    multihop = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    return hop1, hop2, multihop


# --------------------------------------------------------------------------- Part B
class FusionGate(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, qn):
        return torch.sigmoid(self.net(qn)).squeeze(-1)


def _prep_gate(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, K=50, M=100):
    """Precompute the dense∪rel candidate union + per-candidate rel/dense cosines (g1 frozen)."""
    with torch.no_grad():
        rel_pos = g1(torch.tensor(q_tr, device=device), Xt[[int(s) for s in seed_tr]]).cpu().numpy()
    _, dtop = index.search(np.ascontiguousarray(q_tr.astype("float32")), K)
    _, rtop = index.search(np.ascontiguousarray(rel_pos.astype("float32")), K)
    n = len(q_tr)
    C = np.full((n, M), -1, np.int64); rs = np.zeros((n, M), np.float32); ds = np.zeros((n, M), np.float32)
    gmask = np.zeros((n, M), bool); vmask = np.zeros((n, M), bool)
    for i in range(n):
        u = list(dict.fromkeys(rtop[i].tolist() + dtop[i].tolist()))[:M]
        C[i, :len(u)] = u; vmask[i, :len(u)] = True
        gset = set(gold_tr[i])
        for j, c in enumerate(u):
            gmask[i, j] = c in gset
        cv = X[u]                                             # (len,d)
        rs[i, :len(u)] = cv @ rel_pos[i]; ds[i, :len(u)] = cv @ q_tr[i]
    return rel_pos, C, rs, ds, gmask, vmask


def _train_gate(q_tr, rs, ds, gmask, vmask, device, epochs, tau=0.05):
    has_gold = gmask.any(1)
    idx = np.where(has_gold)[0]
    qtr = torch.tensor(q_tr, device=device)
    rs_t = torch.tensor(rs, device=device); ds_t = torch.tensor(ds, device=device)
    gm = torch.tensor(gmask, device=device); vm = torch.tensor(vmask, device=device)
    torch.manual_seed(INIT_SEED)
    gate = FusionGate(q_tr.shape[1]).to(device); opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    for ep in range(epochs):
        gate.train(); order = idx.copy(); random.Random(ep).shuffle(order)
        for s in range(0, len(order), 256):
            b = order[s:s + 256]
            a = gate(qtr[b]).unsqueeze(1)                     # (B,1)
            fused = (a * rs_t[b] + (1 - a) * ds_t[b]) / tau
            fused = fused.masked_fill(~vm[b], -1e9)
            logp = F.log_softmax(fused, dim=1)
            gold_logp = torch.where(gm[b], logp, torch.full_like(logp, -1e9))
            loss = -torch.logsumexp(gold_logp, dim=1).mean()  # listwise: mass on golds in the union
            opt.zero_grad(); loss.backward(); opt.step()
    gate.eval()
    return gate


# --------------------------------------------------------------------------- driver
def run(dataset, epochs=25, limit=8000, device=None):
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
    dense_te = _order(q_te, index)

    # g1 = trained 1-hop (shared by both parts)
    log.info(f"[{dataset}] training g1 (1-hop)...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)

    # ---- Part A: trained 2-hop
    log.info(f"[{dataset}] training g2 (2-hop)...")
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs)
    hop1, hop2, multihop = _multihop_orders(g1, g2, q_te, seed_te, X, Xt, index, device)
    fused_mh = _rrf_fuse([_ranks(dense_te), _ranks(multihop)], [1.0, 1.0])
    partA = {
        "dense": _recall(dense_te, gold_te),
        "hop1_1hop": _recall(hop1, gold_te),
        "hop2_alone": _recall(hop2, gold_te),
        "multihop_1u2": _recall(multihop, gold_te),
        "fused_dense_multihop": _recall(fused_mh, gold_te),
        "overlap_dense_vs_1hop": _overlap(dense_te, hop1, gold_te, 100),
        "overlap_dense_vs_multihop": _overlap(dense_te, multihop, gold_te, 100),
    }
    partA["neither_reduction@100"] = round(
        partA["overlap_dense_vs_1hop"]["neither"] - partA["overlap_dense_vs_multihop"]["neither"], 2)

    # ---- Part B: regularized gate (uses g1's 1-hop offset)
    log.info(f"[{dataset}] training regularized gate...")
    rel_pos_tr, C, rs, ds, gmask, vmask = _prep_gate(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device)
    gate = _train_gate(q_tr, rs, ds, gmask, vmask, device, epochs)
    with torch.no_grad():
        rel_te = _order(g1(torch.tensor(q_te, device=device),
                           Xt[[int(s) for s in seed_te]]).cpu().numpy(), index)
        alpha = gate(torch.tensor(q_te, device=device)).cpu().numpy()
    dmap, rmap = _ranks(dense_te), _ranks(rel_te)
    partB = {
        "dense": _recall(dense_te, gold_te),
        "rel_1hop": _recall(rel_te, gold_te),
        "rrf_equal": _recall(_rrf_fuse([dmap, rmap], [1.0, 1.0]), gold_te),
        "gate_learned": _recall(_rrf_fuse([dmap, rmap], [1 - alpha, alpha]), gold_te),
        "alpha_mean": round(float(alpha.mean()), 3), "alpha_std": round(float(alpha.std()), 3),
    }
    for a in (0.25, 0.5, 0.75):
        partB[f"wrrf_rel{a}"] = _recall(_rrf_fuse([dmap, rmap], [1 - a, a]), gold_te)
    fixed_best = max((partB[k][100], k) for k in partB if k.startswith("wrrf") or k == "rrf_equal")
    partB["fixed_best"] = {"method": fixed_best[1], "recall@100": fixed_best[0]}
    partB["gate_vs_fixed@100"] = round(partB["gate_learned"][100] - fixed_best[0], 2)

    out = {"dataset": dataset, "n_test": len([g for g in gold_te if g]), "budgets": KS,
           "A_multihop": partA, "B_gate": partB}
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "dynamic.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] 2HOP: 1hop@100 {partA['hop1_1hop'][100]} -> multihop {partA['multihop_1u2'][100]} "
             f"| neither {partA['overlap_dense_vs_1hop']['neither']}->{partA['overlap_dense_vs_multihop']['neither']} "
             f"(-{partA['neither_reduction@100']}) || GATE: learned@100 {partB['gate_learned'][100]} "
             f"(alpha {partB['alpha_mean']}+-{partB['alpha_std']}) vs fixed {fixed_best[0]} ({partB['gate_vs_fixed@100']})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Trained 2-hop offset + regularized per-query fusion gate.")
    p.add_argument("--datasets", nargs="+",
                   default=["metaqa", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=8000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 DYNAMIC (2-hop + gate): {ds.upper()} =====")
        try:
            run(ds, epochs=a.epochs, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
