"""
Head-to-head: champion (dense+rel_hard+rel_2hop) vs the MLP-transformer, fused.
===============================================================================
The MLP-transformer beat a single offset head (+2.5 mean), but that was standalone
— it was never tested INSIDE the champion fusion. The champion is a 3-way RRF, so
the honest question is: does swapping its single rel_hard component for the K-head
MLP-transformer lift the FUSED L1? Same training, same eval set, four fusions:

  CHAMPION : RRF(dense, rel_hard, rel_2hop)                 (current, locked)
  SWAP     : RRF(dense, mlp_transformer, rel_2hop)          (mlpT replaces rel_hard)
  MINIMAL  : RRF(dense, mlp_transformer)                    (mlpT replaces whole rel stack)
  ADD      : RRF(dense, rel_hard, rel_2hop, mlp_transformer) (mlpT on top)

rel_hard/rel_2hop/mlp_transformer are all offset heads (normalize(seed + g(q))); the
MLP-transformer is just K of them soft-OR'd with a learned gate. Reports recall@{KS}
AND FullCov@{KS} for every config, per dataset + mean, and the swap/minimal/add deltas
vs CHAMPION on both metrics. Run on the 3 relational datasets (rel is neutral on
hotpot/squad). Writes {ds}/results/L1/headtohead.json + _index/l1_headtohead_summary.json.
"""
import os
import gc
import json
import logging
import argparse

import numpy as np
import torch
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, INIT_SEED
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, _recall, KS, MAXK
from src.experiments.l1_dynamic import _train_hop2
from src.experiments.l1_mlp_transformer import MLPTransformer, _train as _train_mlpT, _mh_order

log = logging.getLogger("experiments.l1_headtohead")


def _fullcov(order, gold, budgets=KS):
    """Fraction of queries whose ALL golds are within the top-b (deterministic coverage)."""
    out = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        top_all = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            out[b].append(1.0 if gs <= set(top_all[:b]) else 0.0)
    return {b: round(float(np.mean(out[b])) * 100, 2) for b in budgets}


def _both(order, gold):
    return {"recall": _recall(order, gold), "fullcov": _fullcov(order, gold)}


def run(dataset, epochs=25, limit=15000, K=4, device=None):
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
    qte = torch.tensor(q_te, device=device)
    avg_golds = round(float(np.mean([len(g) for g in gold_te if g])), 2)

    log.info(f"[{dataset}] train (n_tr={len(q_tr)}, n_te={len(q_te)}, avg_golds={avg_golds}): g1,g_hard,g2,mlpT(K={K})...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs)
    torch.manual_seed(INIT_SEED)
    mlpT = _train_mlpT(MLPTransformer(d, K).to(device), q_tr, seed_tr, gold_tr, Xt, index, device, epochs)

    def pos(head):
        with torch.no_grad():
            return head(qte, Xt[[int(s) for s in seed_te]]).cpu().numpy()
    dense = _order(q_te, index)
    rel_hard = _order(pos(g_hard), index)
    hop1 = _order(pos(g1), index); s1 = hop1[:, 0]
    with torch.no_grad():
        hop2 = _order(g2(qte, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
    rel_2hop = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    mlpT_order = _mh_order(mlpT, q_te, seed_te, X, Xt, index, device)

    dmap, hmap, tmap, mmap = _ranks(dense), _ranks(rel_hard), _ranks(rel_2hop), _ranks(mlpT_order)
    cfg = {
        "dense": _both(dense, gold_te),
        "rel_hard": _both(rel_hard, gold_te),
        "rel_2hop": _both(rel_2hop, gold_te),
        "mlp_transformer": _both(mlpT_order, gold_te),
        "CHAMPION": _both(_rrf_fuse([dmap, hmap, tmap], [1.0, 1.0, 1.0]), gold_te),
        "SWAP_dense+mlpT+rel_2hop": _both(_rrf_fuse([dmap, mmap, tmap], [1.0, 1.0, 1.0]), gold_te),
        "MINIMAL_dense+mlpT": _both(_rrf_fuse([dmap, mmap], [1.0, 1.0]), gold_te),
        "ADD_dense+hard+2hop+mlpT": _both(_rrf_fuse([dmap, hmap, tmap, mmap], [1.0, 1.0, 1.0, 1.0]), gold_te),
    }

    def dr(a, b, metric):     # delta of config a vs b at @100 on a metric
        return round(cfg[a][metric][100] - cfg[b][metric][100], 2)
    out = {"dataset": dataset, "K": K, "avg_golds_per_q": avg_golds, "limit": limit,
           "n_test": len([g for g in gold_te if g]), "n_train": len(q_tr), "budgets": KS, "configs": cfg,
           "swap_vs_champion@100": {"recall": dr("SWAP_dense+mlpT+rel_2hop", "CHAMPION", "recall"),
                                    "fullcov": dr("SWAP_dense+mlpT+rel_2hop", "CHAMPION", "fullcov")},
           "minimal_vs_champion@100": {"recall": dr("MINIMAL_dense+mlpT", "CHAMPION", "recall"),
                                       "fullcov": dr("MINIMAL_dense+mlpT", "CHAMPION", "fullcov")},
           "add_vs_champion@100": {"recall": dr("ADD_dense+hard+2hop+mlpT", "CHAMPION", "recall"),
                                   "fullcov": dr("ADD_dense+hard+2hop+mlpT", "CHAMPION", "fullcov")}}
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "headtohead.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] R@100 CHAMP {cfg['CHAMPION']['recall'][100]} | SWAP {cfg['SWAP_dense+mlpT+rel_2hop']['recall'][100]} "
             f"({out['swap_vs_champion@100']['recall']:+}) | MIN {cfg['MINIMAL_dense+mlpT']['recall'][100]} | "
             f"ADD {cfg['ADD_dense+hard+2hop+mlpT']['recall'][100]} ({out['add_vs_champion@100']['recall']:+}) || "
             f"FCOV@100 CHAMP {cfg['CHAMPION']['fullcov'][100]} | SWAP {cfg['SWAP_dense+mlpT+rel_2hop']['fullcov'][100]} "
             f"({out['swap_vs_champion@100']['fullcov']:+}) | ADD {cfg['ADD_dense+hard+2hop+mlpT']['fullcov'][100]} "
             f"({out['add_vs_champion@100']['fullcov']:+})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Champion vs MLP-transformer, fused (recall + FullCov).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=15000)
    p.add_argument("--K", type=int, default=4)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 HEAD-TO-HEAD (champion vs MLP-transformer): {ds.upper()} =====")
        try:
            results[ds] = run(ds, epochs=a.epochs, limit=a.limit, K=a.K)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
        gc.collect()

    if results:
        CFGS = ["dense", "rel_hard", "mlp_transformer", "CHAMPION",
                "SWAP_dense+mlpT+rel_2hop", "MINIMAL_dense+mlpT", "ADD_dense+hard+2hop+mlpT"]

        def mean(cfg, metric):
            return round(float(np.mean([r["configs"][cfg][metric][100] for r in results.values()])), 2)
        summary = {"datasets": list(results),
                   "recall_mean@100": {c: mean(c, "recall") for c in CFGS},
                   "fullcov_mean@100": {c: mean(c, "fullcov") for c in CFGS}}
        for metric in ("recall", "fullcov"):
            ch = summary[f"{metric}_mean@100"]["CHAMPION"]
            summary[f"{metric}_swap_vs_champion@100"] = round(summary[f"{metric}_mean@100"]["SWAP_dense+mlpT+rel_2hop"] - ch, 2)
            summary[f"{metric}_add_vs_champion@100"] = round(summary[f"{metric}_mean@100"]["ADD_dense+hard+2hop+mlpT"] - ch, 2)
            summary[f"best_{metric}"] = max(CFGS, key=lambda c: summary[f"{metric}_mean@100"][c])
        path = os.path.join("data", "ukb_storage", "_index", "l1_headtohead_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"HEAD-TO-HEAD RECALL@100 {summary['recall_mean@100']} -> best={summary['best_recall']} "
                 f"(swap {summary['recall_swap_vs_champion@100']:+}, add {summary['recall_add_vs_champion@100']:+})")
        log.info(f"HEAD-TO-HEAD FULLCOV@100 {summary['fullcov_mean@100']} -> best={summary['best_fullcov']} "
                 f"(swap {summary['fullcov_swap_vs_champion@100']:+}, add {summary['fullcov_add_vs_champion@100']:+})")


if __name__ == "__main__":
    main()
