"""
REAL hard-negative-mining ablation (fixes the vacuous S1nohnm).
================================================================
The campaign's HNM ablation set hn_k = npart-1, but in kl_div_loss hn_k=npart-1
selects ALL negatives via topk -> identical to hn_k=0 (full softmax). So "HNM on
vs off" was all-negatives vs all-negatives: a tautology (bit-identical metrics).

This actually mines: for hn_k in {8,16,32,64,all}, restrict the KL softmax to the
positives + the hn_k HARDEST (highest-scoring) negative partitions, on the `hard`
config. Trains from the same pinned init (INIT_SEED) on the same split, so the
per-query FullCov vectors are paired -> McNemar of each hn_k vs `all` is valid.
If small hn_k never beats `all` (p>=0.05), HNM genuinely doesn't help and full
softmax is the right baseline. Writes results/overlap_ablation/{ds}_hnm_sweep.json.
"""
import os
import json
import logging
import argparse

import torch

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import (
    _reconstruct, _hard_membership, _centroids, _splits, _train, _eval, membership_ref, TAU,
)
from src.experiments.stats import paired

log = logging.getLogger("experiments.hnm_sweep")

LIMITS = {"metaqa": 40000}                      # cap huge query sets (matches campaign)
METRIC_KEYS = ("full_coverage@10", "full_coverage@20", "full_coverage@50",
               "gt_recall@20", "mrr", "weakest_positive_rank")


def run(dataset, hn_ks=(8, 16, 32, 64, -1), epochs=100, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    tau = TAU.get(dataset, 0.07)

    membership = _hard_membership(engine)
    membership_ref["hard"] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    limit = LIMITS.get(dataset, 0)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    test, test_e = splits["test"], split_embs["test"]
    logs_dir = os.path.join("results", "overlap_ablation", "_hnm_logs", dataset)

    results, vecs = {}, {}
    for hk in hn_ks:
        hn_k = (npart - 1) if hk == -1 else hk
        label = "all" if hk == -1 else str(hk)
        log.info(f"[{dataset}] training hard+KL with hn_k={hn_k} ({label}) ...")
        model, best_state, _, Cg = _train(engine, C, splits, split_embs, device, tau, hn_k,
                                           epochs, logs_dir, "hard", loss_name="kl")
        m = _eval(model, best_state, Cg, test, test_e, membership, device)
        vecs[label] = m.pop("_fc20_vec")
        results[f"hnk_{label}"] = {k: m.get(k) for k in METRIC_KEYS}
        results[f"hnk_{label}"]["n_test"] = m.get("n_test")
        log.info(f"  [{dataset} hn_k={label}] FCov@20={results[f'hnk_{label}']['full_coverage@20']} "
                 f"FCov@50={results[f'hnk_{label}']['full_coverage@50']}")

    mcnemar = {label: paired(v, vecs["all"]) for label, v in vecs.items() if label != "all"}
    out = {"dataset": dataset, "npart": npart, "tau": tau, "loss": "kl", "config": "hard",
           "hn_ks": [(npart - 1) if h == -1 else h for h in hn_ks],
           "note": "McNemar (FullCov@50) each hn_k vs all-negatives baseline",
           "results": results, "mcnemar_vs_all": mcnemar}
    os.makedirs("results/overlap_ablation", exist_ok=True)
    path = f"results/overlap_ablation/{dataset}_hnm_sweep.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved {path}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Real hard-negative-mining ablation (hn_k sweep).")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "metaqa", "hotpotqa_clean", "musique_clean", "squad_clean"])
    p.add_argument("--hn_ks", nargs="+", type=int, default=[8, 16, 32, 64, -1])
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== HNM SWEEP: {ds.upper()} =====")
        run(ds, hn_ks=tuple(a.hn_ks), epochs=a.epochs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
