"""
Train + persist the locked per-dataset L1 champion (phase 5).
=============================================================
Once champion.py picks the pool-matched (structure, loss) winner, this trains that
single config properly (full-softmax; HNM is harmful) and SAVES the router model +
full metrics into the UKB store, so L2 can load a concrete L1 front-end.

Writes:
  data/ukb_storage/{ds}/results/L1/champion_model.pt    (router MLP state_dict + centroids)
  data/ukb_storage/{ds}/results/L1/champion_model.json  (config, loss, full metric suite)
Reuses overlap_retrain's membership/centroid/train/eval building blocks so it is
identical to the ablation pipeline (no drift).
"""
import os
import json
import logging
import argparse

import numpy as np
import torch

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import (
    _reconstruct, _hard_membership, _onehop_membership, _twohop_membership,
    _synhop_membership, _knn_membership, _centroids, _splits, _train, _eval,
    membership_ref, TAU, HNK,
)
from src.pipeline.ukb_results import rpath


def _atom(name, engine, node_vecs):
    if name == "hard":
        return _hard_membership(engine)
    if name == "overlap1":
        return _onehop_membership(engine)
    if name == "overlap2":
        return _twohop_membership(engine)
    if name == "syn1":
        return _synhop_membership(engine)
    if name.startswith("knn"):
        return _knn_membership(engine, node_vecs, int(name[3:]))
    raise ValueError(f"unknown atom {name!r}")


def _build(cfg, engine, node_vecs):
    atoms = [_atom(a, engine, node_vecs) for a in cfg.split("+")]
    if len(atoms) == 1:
        return atoms[0]
    keys = set().union(*[set(a) for a in atoms])
    return {k: set().union(*[a.get(k, set()) for a in atoms]) for k in keys}


def run(dataset, config, loss="kl", epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = logging.getLogger("experiments.train_champion")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    tau = TAU.get(dataset, 0.07)
    hn_k = HNK.get(dataset, npart - 1)                       # full-softmax (all negatives) = best

    membership = _build(config, engine, node_vecs)
    membership_ref[config] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    logs_dir = os.path.join("data", "ukb_storage", dataset, "results", "L1", "_champion_train")
    log.info(f"[{dataset}] training champion {config}/{loss} (tau={tau}, hn_k={hn_k}, {npart} parts)")
    model, best_state, _, Cg = _train(engine, C, splits, split_embs, device, tau, hn_k,
                                      epochs, logs_dir, config, loss_name=loss)
    metrics = _eval(model, best_state, Cg, splits["test"], split_embs["test"], membership, device)
    metrics.pop("_fc20_vec", None)

    torch.save({"state_dict": best_state, "centroids": Cg.cpu(), "config": config,
                "loss": loss, "npart": npart, "hn_k": hn_k},
               rpath(dataset, "L1", "champion_model.pt"))
    out = {"dataset": dataset, "config": config, "loss": loss, "hn_k": hn_k, "npart": npart,
           "mean_memberships_per_doc": round(float(np.mean(
               [len(membership.get(n.node_id, set())) for n in engine.nodes])), 3),
           "metrics": metrics}
    with open(rpath(dataset, "L1", "champion_model.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] champion trained: FCov@20={metrics.get('full_coverage@20')} "
             f"gtR@20={metrics.get('gt_recall@20')} -> {rpath(dataset,'L1','champion_model.json')}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Train + persist the locked L1 champion.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True, help="e.g. overlap1+knn3 or knn1")
    p.add_argument("--loss", default="kl", choices=["kl", "coverage"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(a.dataset, a.config, loss=a.loss, epochs=a.epochs, limit=a.limit)


if __name__ == "__main__":
    main()
