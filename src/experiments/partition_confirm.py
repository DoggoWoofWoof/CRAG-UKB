"""
Trained-router confirmation of partition granularity (pool-matched).
====================================================================
The training-free partition ablation said finer is better; this confirms it with
the REAL KL+HNM router. For each target docs/partition it: builds the partition
in-memory (METIS on structural+kNN edges), trains the seeded MLP router
(frozen query emb -> MLP -> cosine centroids, KL+HNM), and evaluates FullCov at
FIXED POOL BUDGETS (2%/5%/10% of corpus) — the only fair way to compare
granularities, since more/smaller partitions need more top-K to fill the same
pool. Writes results/partition_ablation/{dataset}_trained_confirm.json.
"""
import os
import json
import random
import logging
import argparse

import numpy as np
import faiss
import torch
import torch.nn.functional as F

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_mlp import kl_div_loss
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, TAU, HNK, INIT_SEED

log = logging.getLogger("experiments.partition_confirm")


def _adjacency(engine, n, id2idx):
    adj = [set() for _ in range(n)]
    for i, node in enumerate(engine.nodes):
        for nb in list(node.neighbors) + list(node.metadata.get("synthetic_neighbors", [])):
            j = id2idx.get(nb)
            if j is not None and j != i:
                adj[i].add(j); adj[j].add(i)
    return adj


def _metis(adj, n, target):
    nparts = max(2, n // target)
    if sum(len(a) for a in adj) == 0:
        return np.array([i // max(1, n // nparts) for i in range(n)])
    import pymetis
    _, mem = pymetis.part_graph(nparts, adjacency=[sorted(a) for a in adj])
    return np.asarray(mem)


def _train_router(q_tr, gp_tr, q_va, gp_va, C, tau, hn_k, epochs, device):
    Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1)
    D = C.shape[1]
    torch.manual_seed(INIT_SEED)
    model = TextPartitionMLP(input_dim=D, hidden_dim=512, output_dim=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    bs, best, best_state, noimp = 64, 1e9, None, 0
    for ep in range(epochs):
        model.train(); order = list(range(len(q_tr))); random.Random(ep).shuffle(order)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            proj = F.normalize(model(torch.tensor(q_tr[idx], dtype=torch.float32, device=device)), dim=-1)
            loss = kl_div_loss(proj, [gp_tr[i] for i in idx], Cg, temperature=tau, hn_k=hn_k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); tot = 0.0
        with torch.no_grad():
            for s in range(0, len(q_va), bs):
                proj = F.normalize(model(torch.tensor(q_va[s:s + bs], dtype=torch.float32, device=device)), dim=-1)
                tot += float(kl_div_loss(proj, gp_va[s:s + bs], Cg, temperature=tau, hn_k=hn_k))
        if tot < best:
            best, best_state, noimp = tot, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            noimp += 1
        if noimp >= 15:
            break
    model.load_state_dict(best_state)
    return model, Cg


def _pool_fcov(model, Cg, q_te, gp_te, sizes, budgets, device):
    with torch.no_grad():
        proj = F.normalize(model(torch.tensor(q_te, dtype=torch.float32, device=device)), dim=-1)
        ranked = torch.argsort(-(proj @ Cg.T), dim=1).cpu().numpy()
    res = {b: [] for b in budgets}
    for qi in range(len(q_te)):
        gp = set(gp_te[qi])
        if not gp:
            continue
        r = ranked[qi]; cum = np.cumsum(sizes[r])
        for b in budgets:
            k = int(np.searchsorted(cum, b) + 1)     # partitions fitting in pool budget b
            res[b].append(1.0 if gp <= set(r[:k].tolist()) else 0.0)
    return {f"fcov@{int(b)}docs": round(float(np.mean(v)) * 100, 2) for b, v in res.items()}


def run(dataset, targets, epochs=80, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    engine = CoreEngine(source=dataset)
    nv = _reconstruct(engine.node_index).astype("float32")
    id2idx = engine.node_id_to_idx; n = len(engine.nodes)
    adj = _adjacency(engine, n, id2idx)
    sp = _splits(engine, _hard_membership(engine))
    enc = DenseEncoder()
    emb = {s: enc.encode([q.content for q, _, _ in sp[s]]).astype("float32") for s in ("train", "val", "test")}
    golds_idx = {s: [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in sp[s]] for s in sp}
    budgets = [round(n * f) for f in (0.02, 0.05, 0.10)]

    out = {"dataset": dataset, "n_docs": n, "pool_budgets": budgets, "hn_k": hn_k, "targets": {}}
    for tgt in targets:
        mem = _metis(adj, n, tgt); npart = int(mem.max()) + 1
        sizes = np.bincount(mem, minlength=npart).astype(np.float64)
        C = np.zeros((npart, nv.shape[1]), dtype=np.float32)
        for i, p in enumerate(mem):
            C[p] += nv[i]
        faiss.normalize_L2(C)
        gp = {s: [sorted({int(mem[i]) for i in gi}) for gi in golds_idx[s]] for s in sp}
        model, Cg = _train_router(emb["train"], gp["train"], emb["val"], gp["val"], C, tau, hn_k, epochs, device)
        fcov = _pool_fcov(model, Cg, emb["test"], gp["test"], sizes, budgets, device)
        out["targets"][str(tgt)] = {"npart": npart, "trained_fcov_at_pool": fcov}
        log.info(f"  [{dataset} tgt={tgt:>4} npart={npart:>4}] trained {fcov}")

    os.makedirs("results/partition_ablation", exist_ok=True)
    with open(f"results/partition_ablation/{dataset}_trained_confirm.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/partition_ablation/{dataset}_trained_confirm.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Trained-router partition-granularity confirmation.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean"])
    p.add_argument("--targets", nargs="+", type=int, default=[100, 250, 1000])
    p.add_argument("--epochs", type=int, default=80)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        log.info(f"===== PARTITION CONFIRM (trained): {ds.upper()} =====")
        run(ds, targets=tuple(a.targets), epochs=a.epochs, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
