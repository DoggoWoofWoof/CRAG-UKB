"""
L1 = PARTITION selection; mlpT = K-head PARTITION predictor (soft-OR over K partitions).
========================================================================================
L1's job is to route to the right PARTITIONS (candidate generation for L2/L3), not to
node-rank. The mlpT's K heads are meant to claim the K PARTITIONS a 1-to-many query's
golds span (actor->movies etc.), NOT K node positions. So each head emits one query-
conditioned direction in partition-CENTROID space and a partition's score is the soft-OR
    score(p) = max_k  ( head_k . centroid_p )
trained against the gold-partition SET with a KL objective + hard negatives, so distinct
heads specialize on distinct gold partitions -> the multi-partition predictor. K=1 recovers
the plain single-projection router, making single-vs-mlpT a controlled comparison.

Signals feeding the heads (concat): q (dense) + seed (top-1 doc) + nbr (top-k mean) and,
optionally, the champion's relational offset  off = normalize(seed + g_hard(q))  as one
more input. Headline metric = partition FullCov@{5,10,20,50,100} (fraction of queries whose
ALL golds' partitions fall in the top-P routed partitions), under HARD membership and
OVERLAP membership (own + 1-hop-neighbour partitions — the direct test that overlap makes
cross-boundary multi-hop golds partition-routable). Writes L1/partition_mlpt.json. Relative
within-run comparison; the winning config is re-verified on the frozen substrate via the
canonical harness before it becomes a paper number.
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
from src.experiments.overlap_retrain import (
    _reconstruct, _hard_membership, _onehop_membership, _centroids, _splits,
    TAU, HNK, INIT_SEED,
)
from src.experiments.multisignal_route import _concat, _cov, KS
from src.experiments.l1_ablate import _train_offset

log = logging.getLogger("experiments.l1_partition_mlpt")


class PartitionMLPT(nn.Module):
    """K query-conditioned directions in partition-centroid space; partition score = soft-OR."""
    def __init__(self, d_in, d_cent, K=8, hidden=512):
        super().__init__()
        self.K, self.d = K, d_cent
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, K * d_cent))

    def heads(self, x):
        return F.normalize(self.net(x).view(-1, self.K, self.d), dim=-1)      # (B,K,d)

    def logits(self, x, Cg):
        sim = torch.einsum("bkd,pd->bkp", self.heads(x), Cg)                  # (B,K,npart)
        return sim.max(dim=1).values                                          # (B,npart) soft-OR


def _softor_kl(logits, pids_list, tau, hn_k):
    """KL of the soft-OR partition logits toward the uniform gold-partition target (+ hard negs)."""
    sim = logits / tau
    B, npart = sim.shape
    pos = torch.zeros_like(sim, dtype=torch.bool)
    for i, pids in enumerate(pids_list):
        for p in pids:
            if p < npart:
                pos[i, p] = True
    valid = pos.any(dim=1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    sim, pos = sim[valid], pos[valid]
    if hn_k > 0:
        neg = sim.masked_fill(pos, -1e9)
        k = min(hn_k, npart - 1)
        _, hard_idx = neg.topk(k, dim=1)
        keep = pos.clone(); keep.scatter_(1, hard_idx, True)
        sim = sim.masked_fill(~keep, -1e9)
    student = F.log_softmax(sim, dim=1)
    teacher = pos.float(); teacher = teacher / teacher.sum(dim=1, keepdim=True)
    return F.kl_div(student, teacher, reduction="batchmean")


def _prep(qs, X, encoder, index, id2idx, topk=10):
    q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
    _, order = index.search(q, topk)
    seed_idx = order[:, 0]
    return {"q": q, "seed": X[seed_idx].astype("float32"), "nbr": X[order].mean(axis=1).astype("float32"),
            "seed_idx": seed_idx,
            "gold_idx": [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]}


def _offsig(head, q, seed_emb, device):
    with torch.no_grad():
        v = head(torch.tensor(q, device=device), torch.tensor(seed_emb, device=device))
    return v.cpu().numpy().astype("float32")


def _train_pmlpt(f_tr, tr_rows, f_va, va_rows, Cg, D, K, device, tau, hn_k, epochs):
    torch.manual_seed(INIT_SEED)
    model = PartitionMLPT(f_tr.shape[1], D, K=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    bs = 64; best, best_state, noimp = float("inf"), None, 0
    Ftr = torch.tensor(f_tr, device=device); Fva = torch.tensor(f_va, device=device)
    for ep in range(epochs):
        model.train(); idx = list(range(len(tr_rows))); random.Random(ep).shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            loss = _softor_kl(model.logits(Ftr[b], Cg), [tr_rows[i][1] for i in b], tau, hn_k)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(va_rows), bs):
                tot += float(_softor_kl(model.logits(Fva[s:s + bs], Cg),
                                        [p for _, p, _ in va_rows[s:s + bs]], tau, hn_k)); nb += 1
        vl = tot / max(nb, 1)
        if vl < best:
            best, best_state, noimp = vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            noimp += 1
        if noimp >= 12:
            break
    model.load_state_dict(best_state); model.eval()
    return model


def run(dataset, epochs=100, off_epochs=25, limit=8000, Ks=(1, 8), device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index); X = node_vecs.astype("float32"); faiss.normalize_L2(X)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    id2idx = engine.node_id_to_idx
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X); Xt = torch.tensor(X, device=device)

    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    sig = {s: _prep(splits[s], X, encoder, index, id2idx) for s in ("train", "val", "test")}

    tr = sig["train"]
    log.info(f"[{dataset}] training rel_hard offset head on {len(splits['train'])} train q...")
    ghard = _train_offset("hard", tr["q"], tr["seed_idx"], tr["gold_idx"], Xt, index, device, off_epochs)
    for s in ("train", "val", "test"):
        sig[s]["off"] = _offsig(ghard, sig[s]["q"], sig[s]["seed"], device)

    signal_sets = {"q+seed+nbr": ["q", "seed", "nbr"], "q+seed+nbr+off": ["q", "seed", "nbr", "off"]}
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    out = {"dataset": dataset, "npart": npart, "n_test": len(splits["test"]),
           "budgets": KS, "limit": limit, "Ks": list(Ks), "membership": {}}

    for mem_name, mem_fn in (("hard", _hard_membership), ("overlap", _onehop_membership)):
        membership = mem_fn(engine)
        C, _ = _centroids(engine, node_vecs, membership, npart)
        Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1); D = C.shape[1]

        def rows(split):
            r = []
            for (node, _, golds) in splits[split]:
                gp = set()
                for g in golds:
                    gp |= membership.get(g, set())
                r.append((node, sorted(gp), golds))
            return r
        tr_rows, va_rows, te_rows = rows("train"), rows("val"), rows("test")

        res = {}
        for sname, sset in signal_sets.items():
            f_tr = _concat(sig["train"], sset); f_va = _concat(sig["val"], sset); f_te = _concat(sig["test"], sset)
            Fte = torch.tensor(f_te, device=device)
            for K in Ks:
                tag = f"{'single' if K == 1 else f'mlpT_K{K}'}|{sname}"
                model = _train_pmlpt(f_tr, tr_rows, f_va, va_rows, Cg, D, K, device, tau, hn_k, epochs)
                with torch.no_grad():
                    scores = model.logits(Fte, Cg).cpu().numpy()
                fc, gt = _cov(np.argsort(-scores, axis=1), te_rows, membership, npart)
                res[tag] = {"fullcov": fc, "gt_recall": gt}
                log.info(f"  [{dataset} {mem_name} {tag:26}] partition FullCov {fc}")
        base = res[f"single|q+seed+nbr"]["fullcov"]
        lift = {tag: {k: round(v["fullcov"][k] - base[k], 2) for k in KS}
                for tag, v in res.items() if tag != "single|q+seed+nbr"}
        out["membership"][mem_name] = {"configs": res, "lift_vs_single|q+seed+nbr": lift}
        log.info(f"[{dataset} {mem_name}] lift vs single|q+seed+nbr: {lift}")

    # Exploratory output -> dedicated dir so pulling it can never overwrite canonical
    # paper files (results/L1/optimize*.json etc.) on the shared Modal volume.
    out_dir = os.path.join("data", "ukb_storage", dataset, "results", "L1_explore")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "partition_mlpt.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L1 K-head PARTITION predictor (mlpT soft-OR over partitions).")
    p.add_argument("--datasets", nargs="+", default=["musique_clean", "metaqa", "2wiki_clean"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--off-epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--Ks", nargs="+", type=int, default=[1, 8])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 K-HEAD PARTITION PREDICTOR (mlpT): {ds.upper()} =====")
        try:
            run(ds, epochs=a.epochs, off_epochs=a.off_epochs, limit=a.limit, Ks=tuple(a.Ks))
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
