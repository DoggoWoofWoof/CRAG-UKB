"""
L1 FULL — definitive EXHAUSTIVE combination ablation (lose nothing).
====================================================================
Greedy forward-selection can miss the global best. This searches the ENTIRE power
set of retrievers, so no combination is left untested, and adds a weight sweep on
the winner + a uniform (cross-dataset) champion.

Retrievers (trained once per dataset):
  dense, rel_base (1-hop), rel_hard (hard-neg), rel_mseed (multi-seed),
  rel_mix (mixture-of-K), rel_2hop (trained 2-hop, from l1_dynamic).

Ablations:
  A  — every retriever alone (recall@{50,100,200,500}).
  COMBO — ALL 2^6-1 = 63 subsets, equal-RRF fused, ranked by recall@100 (exhaustive).
  W  — on the winning subset, sweep the relational-family weight (dense=1, rel=w)
       for w in {0.5,1,1.5,2,3} — so we also don't lose the best *weighting*.
  overlap — dense vs best single rel (complementarity).
Per-dataset champion + UNIFORM champion (subset with best mean recall@100 across all
datasets — the generalizable config). Writes {ds}/results/L1/full.json + summary.
"""
import os
import gc
import json
import logging
import argparse
from itertools import combinations

import numpy as np
import torch
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, _recall, _overlap, KS, MAXK
from src.experiments.l1_dynamic import _train_hop2

log = logging.getLogger("experiments.l1_full")
RETRIEVERS = ["dense", "rel_base", "rel_hard", "rel_mseed", "rel_mix", "rel_2hop"]


def run(dataset, epochs=20, limit=5000, M=3, K=8, device=None):
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
        _, seed = index.search(q, M)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed, gold
    q_tr, seed_tr, gold_tr = prep(splits["train"])
    q_te, seed_te, gold_te = prep(splits["test"])
    qte = torch.tensor(q_te, device=device)

    # ---- train heads
    log.info(f"[{dataset}] training heads (base, hard, mix, 2hop)...")
    g1 = _train_offset("base", q_tr, seed_tr[:, 0], gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr[:, 0], gold_tr, Xt, index, device, epochs)
    g_mix = _train_offset("mix", q_tr, seed_tr[:, 0], gold_tr, Xt, index, device, epochs, K=K)
    g2 = _train_hop2(g1, q_tr, seed_tr[:, 0], gold_tr, X, Xt, index, device, epochs)

    # ---- build retriever orders
    def pos(head, col):
        with torch.no_grad():
            return head(qte, Xt[[int(s) for s in seed_te[:, col]]]).cpu().numpy()
    orders = {}
    orders["dense"] = _order(q_te, index)
    orders["rel_base"] = _order(pos(g1, 0), index)
    orders["rel_hard"] = _order(pos(g_hard, 0), index)
    orders["rel_mseed"] = _rrf_fuse([_ranks(_order(pos(g1, m), index)) for m in range(M)], [1.0] * M)
    with torch.no_grad():
        mixp = g_mix(qte, Xt[[int(s) for s in seed_te[:, 0]]]).cpu().numpy()          # (nq,K,d)
    orders["rel_mix"] = _rrf_fuse([_ranks(_order(mixp[:, k, :], index)) for k in range(K)], [1.0] * K)
    hop1 = orders["rel_base"]; s1 = hop1[:, 0]
    with torch.no_grad():
        hop2 = _order(g2(qte, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
    orders["rel_2hop"] = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    rmap = {n: _ranks(orders[n]) for n in RETRIEVERS}

    # ---- A: each alone
    A = {n: _recall(orders[n], gold_te) for n in RETRIEVERS}

    # ---- EXHAUSTIVE combination: every non-empty subset, equal-RRF, recall@100 for ranking
    sub100 = {}
    for r in range(1, len(RETRIEVERS) + 1):
        for c in combinations(RETRIEVERS, r):
            key = "+".join(c)
            sub100[key] = _recall(_rrf_fuse([rmap[n] for n in c], [1.0] * len(c)), gold_te, budgets=[100])[100]
    ranked = sorted(sub100.items(), key=lambda kv: -kv[1])
    best_key = ranked[0][0]; best_combo = best_key.split("+")
    best_full = _recall(_rrf_fuse([rmap[n] for n in best_combo], [1.0] * len(best_combo)), gold_te)

    # ---- W: relational-family weight sweep on the winning subset
    wsweep = {}
    for w in (0.5, 1.0, 1.5, 2.0, 3.0):
        weights = [1.0 if n == "dense" else w for n in best_combo]
        wsweep[str(w)] = _recall(_rrf_fuse([rmap[n] for n in best_combo], weights), gold_te)[100]
    best_w = max(wsweep, key=wsweep.get)

    best_single_rel = max([n for n in RETRIEVERS if n != "dense"], key=lambda n: A[n][100])
    ov = _overlap(orders["dense"], orders[best_single_rel], gold_te, 100)

    out = {"dataset": dataset, "n_test": len([g for g in gold_te if g]), "budgets": KS,
           "A_alone": A, "best_single_rel": best_single_rel,
           "combo_top10": [{"combo": k, "recall@100": v} for k, v in ranked[:10]],
           "best_combo": best_combo, "best_combo_recall": best_full,
           "weight_sweep@100": wsweep, "best_weight": best_w, "best_weighted@100": wsweep[best_w],
           "overlap_dense_vs_bestrel@100": ov, "dense@100": A["dense"][100],
           "all_subsets@100": sub100}     # full power set, for post-hoc cross-dataset aggregation
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "full.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] BEST combo = {best_key} @100 {best_full[100]} (dense {A['dense'][100]}) "
             f"| best weighted@100 {wsweep[best_w]} (w={best_w}) | best single rel {best_single_rel} {A[best_single_rel][100]}")
    return {"dataset": dataset, "sub100": sub100, "best_combo": best_combo,
            "best_combo@100": best_full[100], "dense@100": A["dense"][100]}


def main(argv=None):
    p = argparse.ArgumentParser(description="Exhaustive L1 combination ablation (full power set).")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--limit", type=int, default=5000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 FULL EXHAUSTIVE: {ds.upper()} =====")
        try:
            results[ds] = run(ds, epochs=a.epochs, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
        gc.collect()                                         # free the dataset's engine/index before the next

    if len(results) > 1:
        keys = set.intersection(*[set(r["sub100"]) for r in results.values()])
        uniform = {k: round(float(np.mean([results[ds]["sub100"][k] for ds in results])), 2) for k in keys}
        top_uniform = sorted(uniform.items(), key=lambda kv: -kv[1])[:10]
        summary = {
            "per_dataset_best": {ds: {"combo": r["best_combo"], "recall@100": r["best_combo@100"],
                                      "dense@100": r["dense@100"]} for ds, r in results.items()},
            "uniform_top10": [{"combo": k, "mean_recall@100": v} for k, v in top_uniform],
            "uniform_champion": top_uniform[0][0], "uniform_champion_mean@100": top_uniform[0][1],
            "dense_mean@100": round(float(np.mean([r["dense@100"] for r in results.values()])), 2),
        }
        path = os.path.join("data", "ukb_storage", "_index", "l1_full_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"UNIFORM CHAMPION = {summary['uniform_champion']} (mean@100 {summary['uniform_champion_mean@100']}, "
                 f"dense {summary['dense_mean@100']})")


if __name__ == "__main__":
    main()
