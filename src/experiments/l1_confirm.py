"""
Confirm the L1 uniform champion at FULL data.
=============================================
The exhaustive search picked champion = dense + rel_hard + rel_2hop at limit 4000.
This confirms it at full data (limit 0), training only the champion's heads
(g1 for the 2-hop's first hop + intermediate, g_hard, g2) — no 63-subset search,
so it's light enough to run at scale. Reports, per dataset and as a mean:
  dense (baseline) | rel_hard | rel_2hop | dense+rel_hard | CHAMPION(dense+rel_hard+rel_2hop)
to verify (a) the champion still beats dense by ~+9.5 mean, and (b) whether rel_2hop
still adds over dense+rel_hard once trained on all the data.
Writes {ds}/results/L1/confirm.json + _index/l1_confirm_summary.json.
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
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, _recall, _overlap, KS, MAXK
from src.experiments.l1_dynamic import _train_hop2

log = logging.getLogger("experiments.l1_confirm")


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
    qte = torch.tensor(q_te, device=device)

    log.info(f"[{dataset}] full-data train (n_tr={len(q_tr)}, n_te={len(q_te)}): g1, g_hard, g2...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs)

    def pos(head):
        with torch.no_grad():
            return head(qte, Xt[[int(s) for s in seed_te]]).cpu().numpy()
    dense = _order(q_te, index)
    rel_hard = _order(pos(g_hard), index)
    hop1 = _order(pos(g1), index); s1 = hop1[:, 0]
    with torch.no_grad():
        hop2 = _order(g2(qte, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
    rel_2hop = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])

    dmap, hmap, tmap = _ranks(dense), _ranks(rel_hard), _ranks(rel_2hop)
    cfg = {
        "dense": _recall(dense, gold_te),
        "rel_hard": _recall(rel_hard, gold_te),
        "rel_2hop": _recall(rel_2hop, gold_te),
        "dense+rel_hard": _recall(_rrf_fuse([dmap, hmap], [1.0, 1.0]), gold_te),
        "CHAMPION_dense+rel_hard+rel_2hop": _recall(_rrf_fuse([dmap, hmap, tmap], [1.0, 1.0, 1.0]), gold_te),
    }
    out = {"dataset": dataset, "n_test": len([g for g in gold_te if g]), "n_train": len(q_tr),
           "budgets": KS, "configs": cfg,
           "champion_over_dense@100": round(cfg["CHAMPION_dense+rel_hard+rel_2hop"][100] - cfg["dense"][100], 2),
           "2hop_adds_over_hard@100": round(cfg["CHAMPION_dense+rel_hard+rel_2hop"][100] - cfg["dense+rel_hard"][100], 2)}
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "confirm.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] CHAMPION@100 {cfg['CHAMPION_dense+rel_hard+rel_2hop'][100]} "
             f"(dense {cfg['dense'][100]}, +{out['champion_over_dense@100']}) | "
             f"dense+hard {cfg['dense+rel_hard'][100]} | 2hop adds {out['2hop_adds_over_hard@100']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Confirm the L1 uniform champion at full data.")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 CONFIRM (full data): {ds.upper()} =====")
        try:
            results[ds] = run(ds, epochs=a.epochs, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
        gc.collect()

    if results:
        def m(key):
            return round(float(np.mean([r["configs"][key][100] for r in results.values()])), 2)
        summary = {"datasets": list(results),
                   "dense_mean@100": m("dense"),
                   "dense+rel_hard_mean@100": m("dense+rel_hard"),
                   "champion_mean@100": m("CHAMPION_dense+rel_hard+rel_2hop"),
                   "per_dataset": {ds: {"dense": r["configs"]["dense"][100],
                                        "champion": r["configs"]["CHAMPION_dense+rel_hard+rel_2hop"][100],
                                        "champion_over_dense": r["champion_over_dense@100"],
                                        "2hop_adds": r["2hop_adds_over_hard@100"]} for ds, r in results.items()}}
        summary["champion_over_dense_mean@100"] = round(summary["champion_mean@100"] - summary["dense_mean@100"], 2)
        summary["2hop_adds_mean@100"] = round(summary["champion_mean@100"] - summary["dense+rel_hard_mean@100"], 2)
        path = os.path.join("data", "ukb_storage", "_index", "l1_confirm_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"CONFIRM: champion mean@100 {summary['champion_mean@100']} vs dense {summary['dense_mean@100']} "
                 f"(+{summary['champion_over_dense_mean@100']}) | 2hop adds {summary['2hop_adds_mean@100']} over dense+hard")


if __name__ == "__main__":
    main()
