"""
L1 champion — finalized fusion (weight-rule fix + score calibration).
=====================================================================
Champion retriever set = {dense, rel_hard, rel_2hop} (full-data confirmed
+12.9 mean@100). Equal-RRF's only blemish: it dilutes dense on the most
saturated dataset (hotpot -2.4). Fixes/improvements tested here:

  equal_rrf      : the confirmed baseline (equal weights).
  weight_rule    : per-query relational weight = f(dense confidence).
                   dense top-1 cosine is the saturation signal — high (dense sure)
                   -> down-weight rel; low (dense lost) -> up-weight rel. This IS the
                   complementarity law as a 1-line monotone rule (a,b fit on TRAIN,
                   not a learned MLP — the gate that failed). Reclaims hotpot.
(Next improvement to layer on: calibrated z-normalized COSINE-score fusion, borrowed
from Calibrated Graph-Vector Fusion — uses score magnitude RRF discards.)

Reports recall@{50,100,200,500} per dataset + mean, and the hotpot delta (did the
rule stop the dilution without touching the wins). Writes _index/l1_champion_summary.json.
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
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, _recall, KS, MAXK
from src.experiments.l1_dynamic import _train_hop2

log = logging.getLogger("experiments.l1_champion")


def _order_sims(pos, index, k=MAXK):
    D, I = index.search(np.ascontiguousarray(pos.astype("float32")), k)
    return I, D                                              # order, cosines (IP on normalized = cos)


def _weight_rule(conf, a, b, lo=0.1, hi=1.0):
    return np.clip(a * (b - conf), lo, hi)                  # high dense-conf -> low rel weight


def _fit_rule(conf_tr, dmap, hmap, tmap, gold_tr):
    """Grid-fit (a,b) on TRAIN maximizing fused recall@100 (2 params, not an MLP)."""
    best, best_ab = -1, (3.0, 0.8)
    for a in (2.0, 3.0, 4.0, 5.0):
        for b in (0.70, 0.75, 0.80, 0.85):
            w = _weight_rule(conf_tr, a, b)
            fused = _rrf_fuse([dmap, hmap, tmap], [np.ones_like(w), w, w])
            r = _recall(fused, gold_tr, budgets=[100])[100]
            if r > best:
                best, best_ab = r, (a, b)
    return best_ab


def run(dataset, epochs=25, limit=0, device=None):
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

    log.info(f"[{dataset}] train g1,g_hard,g2 (n_tr={len(q_tr)})...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs)

    def orders(q, seed):
        qt = torch.tensor(q, device=device); sv = Xt[[int(s) for s in seed]]
        with torch.no_grad():
            hard_pos = g_hard(qt, sv).cpu().numpy()
            hop1_pos = g1(qt, sv).cpu().numpy()
        dord, dsim = _order_sims(q, index)
        hard = _order(hard_pos, index)
        hop1 = _order(hop1_pos, index); s1 = hop1[:, 0]
        with torch.no_grad():
            hop2 = _order(g2(qt, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
        two = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
        return dord, dsim[:, 0], hard, two
    d_tr, conf_tr, h_tr, t_tr = orders(q_tr, seed_tr)
    d_te, conf_te, h_te, t_te = orders(q_te, seed_te)

    dmap_tr, hmap_tr, tmap_tr = _ranks(d_tr), _ranks(h_tr), _ranks(t_tr)
    dmap, hmap, tmap = _ranks(d_te), _ranks(h_te), _ranks(t_te)
    a, b = _fit_rule(conf_tr, dmap_tr, hmap_tr, tmap_tr, gold_tr)
    w_te = _weight_rule(conf_te, a, b)

    cfg = {
        "dense": _recall(d_te, gold_te),
        "equal_rrf": _recall(_rrf_fuse([dmap, hmap, tmap], [1.0, 1.0, 1.0]), gold_te),
        "weight_rule": _recall(_rrf_fuse([dmap, hmap, tmap], [np.ones_like(w_te), w_te, w_te]), gold_te),
    }
    out = {"dataset": dataset, "n_test": len([g for g in gold_te if g]), "n_train": len(q_tr),
           "budgets": KS, "rule_ab": [a, b], "dense_conf_mean": round(float(conf_te.mean()), 3),
           "rel_weight_mean": round(float(w_te.mean()), 3), "configs": cfg,
           "rule_vs_equal@100": round(cfg["weight_rule"][100] - cfg["equal_rrf"][100], 2),
           "rule_vs_dense@100": round(cfg["weight_rule"][100] - cfg["dense"][100], 2)}
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "champion.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] dense {cfg['dense'][100]} | equal_rrf {cfg['equal_rrf'][100]} | "
             f"WEIGHT_RULE {cfg['weight_rule'][100]} (rule a={a} b={b}, relw~{out['rel_weight_mean']}, "
             f"conf~{out['dense_conf_mean']}) | rule vs equal {out['rule_vs_equal@100']} vs dense {out['rule_vs_dense@100']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Finalized L1 champion: weight-rule fusion fix.")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 CHAMPION (weight-rule): {ds.upper()} =====")
        try:
            results[ds] = run(ds, epochs=a.epochs, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
        gc.collect()
    if results:
        def m(k):
            return round(float(np.mean([r["configs"][k][100] for r in results.values()])), 2)
        summary = {"datasets": list(results), "dense_mean@100": m("dense"),
                   "equal_rrf_mean@100": m("equal_rrf"), "weight_rule_mean@100": m("weight_rule"),
                   "per_dataset": {ds: {"dense": r["configs"]["dense"][100],
                                        "equal_rrf": r["configs"]["equal_rrf"][100],
                                        "weight_rule": r["configs"]["weight_rule"][100],
                                        "rule_vs_equal": r["rule_vs_equal@100"]} for ds, r in results.items()}}
        summary["rule_vs_equal_mean@100"] = round(summary["weight_rule_mean@100"] - summary["equal_rrf_mean@100"], 2)
        with open(os.path.join("data", "ukb_storage", "_index", "l1_champion_summary.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"CHAMPION SUMMARY: dense {summary['dense_mean@100']} | equal_rrf {summary['equal_rrf_mean@100']} "
                 f"| weight_rule {summary['weight_rule_mean@100']} (+{summary['rule_vs_equal_mean@100']} vs equal)")


if __name__ == "__main__":
    main()
