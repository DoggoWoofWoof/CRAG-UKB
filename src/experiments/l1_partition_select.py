"""
L1 = PARTITION selection, but EXPLOIT node features to pick better partitions.
==============================================================================
The objective is partition coverage (route to the partitions holding the golds), but the
means is not restricted to coarse centroid routing. A partition centroid is a lossy average;
the strongest evidence for "route to partition p" is usually that a good NODE in p ranks high
under our node retrievers. So we let node retrieval VOTE for partitions and fuse that with
centroid routing — using every method we have (dense, rel_hard offset, mlpT, learned router).

For a ranked node list, partition score(p) = sum_r  w(r) * [p in membership(node_r)]  with
RRF weight w(r)=1/(k0+r) (a node votes for all its membership partitions; overlap => own +
neighbour partitions). Methods compared at partition FullCov@{5,10,20,50,100}, HARD vs OVERLAP:

  q_centroid        q . partition-centroid            (pure semantic route, baseline)
  route_mlp         MultiRouter(q+seed+nbr)->centroids (learned centroid route)
  vote_dense        dense node retrieval -> votes
  vote_relfuse      (dense (+) rel_hard) node retrieval -> votes   [champion node signal]
  vote_mlpT         mlpT K node directions (soft-OR union) -> votes
  fuse_route+relvote     RRF(route_mlp, vote_relfuse)
  fuse_all               RRF(route_mlp, vote_relfuse, vote_mlpT)

If node-evidence voting (esp. the champion signal) or its fusion with the learned router beats
pure centroid routing on partition FullCov, THAT is the L1 selector — node features exploited to
get better partitions. Node retrieval orders are membership-independent (computed once); votes and
the router are recomputed per membership. Writes L1_explore/partition_select.json (isolated dir so
the Modal pull can't clobber canonical results/L1). Relative within-run; winner re-verified on the
frozen substrate via the canonical harness before it is a paper number.
"""
import os
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import (
    _reconstruct, _hard_membership, _onehop_membership, _centroids, _splits, TAU, HNK,
)
from src.experiments.multisignal_route import MultiRouter, _concat, _cov, _train_router, KS
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, MAXK

log = logging.getLogger("experiments.l1_partition_select")
K0 = 60


def _prep(qs, X, encoder, index, id2idx, topk=10):
    q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
    _, order = index.search(q, topk)
    seed_idx = order[:, 0]
    return {"q": q, "seed": X[seed_idx].astype("float32"), "nbr": X[order].mean(axis=1).astype("float32"),
            "seed_idx": seed_idx,
            "gold_idx": [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]}


def _headpos(head, q, seed_emb, device):
    with torch.no_grad():
        return head(torch.tensor(q, device=device), torch.tensor(seed_emb, device=device)).cpu().numpy().astype("float32")


def _vote(node_lists, mem_idx, npart, topn, k0=K0):
    """Ranked node lists -> partition scores by RRF membership voting."""
    nq = len(node_lists); scores = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        row = node_lists[qi]
        for r, nd in enumerate(row[:topn]):
            w = 1.0 / (k0 + r)
            for p in mem_idx[int(nd)]:
                scores[qi, p] += w
    return scores


def _order_from_scores(scores):
    return np.argsort(-scores, axis=1)


def run(dataset, epochs=100, off_epochs=25, limit=8000, K=8, topn=200, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index); X = node_vecs.astype("float32"); faiss.normalize_L2(X)
    n = X.shape[0]; npart = max(int(p) for p in engine.partition_map.values()) + 1
    id2idx = engine.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
    hard = np.array([int(engine.partition_map.get(idx2id[i], -1)) for i in range(n)])
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X); Xt = torch.tensor(X, device=device)

    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    sig = {s: _prep(splits[s], X, encoder, index, id2idx) for s in ("train", "val", "test")}

    # node retrievers (membership-independent) — train on train, retrieve on test
    tr = sig["train"]
    log.info(f"[{dataset}] training rel_hard + mlpT node heads ({len(splits['train'])} train q)...")
    ghard = _train_offset("hard", tr["q"], tr["seed_idx"], tr["gold_idx"], Xt, index, device, off_epochs)
    gmix = _train_offset("mix", tr["q"], tr["seed_idx"], tr["gold_idx"], Xt, index, device, off_epochs, K=K)
    te = sig["test"]
    dense_order = _order(te["q"], index, MAXK)
    relhard_order = _order(_headpos(ghard, te["q"], te["seed"], device), index, MAXK)
    champ_order = _rrf_fuse([_ranks(dense_order), _ranks(relhard_order)], [1.0, 1.0], k=MAXK)
    with torch.no_grad():
        mixpos = gmix(torch.tensor(te["q"], device=device), torch.tensor(te["seed"], device=device)).cpu().numpy()
    T = max(topn, MAXK // K)
    head_orders = [_order(np.ascontiguousarray(mixpos[:, k, :]), index, T) for k in range(K)]
    mlpT_order = _rrf_fuse([_ranks(ho) for ho in head_orders], [1.0] * K, k=MAXK)

    node_orders = {"vote_dense": dense_order, "vote_relfuse": champ_order, "vote_mlpT": mlpT_order}
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    out = {"dataset": dataset, "npart": npart, "n_test": len(splits["test"]),
           "budgets": KS, "limit": limit, "topn": topn, "membership": {}}

    for mem_name, mem_fn in (("hard", _hard_membership), ("overlap", _onehop_membership)):
        membership = mem_fn(engine)
        mem_idx = [sorted(membership.get(idx2id[i], {int(hard[i])})) for i in range(n)]
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

        # learned centroid router (q+seed+nbr)
        sset = ["q", "seed", "nbr"]
        f_tr = _concat(sig["train"], sset); f_va = _concat(sig["val"], sset); f_te = _concat(sig["test"], sset)
        route_mlp = _train_router(f_tr, tr_rows, f_va, va_rows, Cg, D, device, tau, hn_k, epochs)
        with torch.no_grad():
            route_scores = (route_mlp(torch.tensor(f_te, device=device)) @ Cg.T).cpu().numpy()

        orders = {"q_centroid": _order_from_scores(te["q"] @ C.T),
                  "route_mlp": _order_from_scores(route_scores)}
        for name, no in node_orders.items():
            orders[name] = _order_from_scores(_vote(no, mem_idx, npart, topn))
        # fusions (RRF over partition rankings)
        orders["fuse_route+relvote"] = _rrf_fuse(
            [_ranks(orders["route_mlp"]), _ranks(orders["vote_relfuse"])], [1.0, 1.0], k=npart)
        orders["fuse_all"] = _rrf_fuse(
            [_ranks(orders["route_mlp"]), _ranks(orders["vote_relfuse"]), _ranks(orders["vote_mlpT"])],
            [1.0, 1.0, 1.0], k=npart)
        # node-only fusions (exclude the weak centroid router)
        orders["fuse_votes"] = _rrf_fuse(
            [_ranks(orders["vote_dense"]), _ranks(orders["vote_relfuse"]), _ranks(orders["vote_mlpT"])],
            [1.0, 1.0, 1.0], k=npart)
        orders["fuse_rel+mlpT"] = _rrf_fuse(
            [_ranks(orders["vote_relfuse"]), _ranks(orders["vote_mlpT"])], [1.0, 1.0], k=npart)
        orders["fuse_dense+rel"] = _rrf_fuse(
            [_ranks(orders["vote_dense"]), _ranks(orders["vote_relfuse"])], [1.0, 1.0], k=npart)

        res = {}
        for name, order in orders.items():
            fc, gt = _cov(order, te_rows, membership, npart)
            res[name] = {"fullcov": fc, "gt_recall": gt}
            log.info(f"  [{dataset} {mem_name} {name:20}] partition FullCov {fc}")
        base = res["route_mlp"]["fullcov"]
        lift = {name: {k: round(v["fullcov"][k] - base[k], 2) for k in KS}
                for name, v in res.items() if name != "route_mlp"}
        out["membership"][mem_name] = {"methods": res, "lift_vs_route_mlp": lift}
        log.info(f"[{dataset} {mem_name}] lift vs route_mlp: {lift}")

    # own dir (not L1_explore) so a concurrent pull can't race the partition-mlpt job
    out_dir = os.path.join("data", "ukb_storage", dataset, "results", "L1_select")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "partition_select.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L1 partition selection exploiting node features (voting + fusion).")
    p.add_argument("--datasets", nargs="+", default=["musique_clean", "metaqa", "2wiki_clean"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--off-epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--topn", type=int, default=200)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 PARTITION SELECT (node-evidence voting + fusion): {ds.upper()} =====")
        try:
            run(ds, epochs=a.epochs, off_epochs=a.off_epochs, limit=a.limit, K=a.K, topn=a.topn)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
