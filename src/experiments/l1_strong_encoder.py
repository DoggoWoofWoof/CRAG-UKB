"""
Strong-encoder A/B — the SIGIR make-or-break experiment.
========================================================
Answers the #1 reviewer question two ways, on the SAME queries/metrics:
  (a) does STRONG-dense alone (bge-large / e5-large) already match our MiniLM relational
      champion?  -> if yes, our method is unnecessary.
  (b) does our relational L1 (dense + rel_hard + rel_2hop, RRF) STILL beat dense when
      trained on the STRONG encoder's geometry?  -> if yes, the gains are real, not a
      weak-encoder artifact.
Trains the champion offset heads on the chosen encoder (swapped via encoder_swap) and
reports dense / rel_hard / rel_2hop / CHAMPION at Recall@2/5/20/100 + FullCov@5/20/100.
Run per encoder (--encoder_subdir minilm|bge_large|e5_large). Writes
{ds}/results/L1/strong_encoder__{tag}.json + _index/l1_strong_encoder_summary.json.
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
from src.experiments.overlap_retrain import _splits, _hard_membership
from src.experiments.encoder_swap import load_docs_and_encoder, has_subdir
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse
from src.experiments.l1_dynamic import _train_hop2

log = logging.getLogger("experiments.l1_strong_encoder")
BUDGETS = (2, 5, 20, 100)
FC_BUDGETS = (5, 20, 100)


def _metrics(order, gold, budgets=BUDGETS, fcb=FC_BUDGETS):
    rec = {b: [] for b in budgets}; fc = {b: [] for b in fcb}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g); top = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            rec[b].append(len(gs & set(top[:b])) / len(gs))
        for b in fcb:
            fc[b].append(1.0 if gs <= set(top[:b]) else 0.0)
    return {**{f"recall@{b}": round(float(np.mean(rec[b])) * 100, 2) for b in budgets},
            **{f"fullcov@{b}": round(float(np.mean(fc[b])) * 100, 2) for b in fcb}}


SUBDIR = {"minilm": None, "bge_large": "bge_large", "e5_large": "e5_large"}


def run(dataset, encoder="bge_large", epochs=25, limit=12000, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subdir = SUBDIR[encoder]
    if subdir and not has_subdir(dataset, subdir):
        raise FileNotFoundError(f"{dataset}/{subdir}/nodes.npy missing — run reencode_ukb first")
    engine = CoreEngine(source=dataset)
    X, encode_q, tag = load_docs_and_encoder(engine, dataset, subdir)
    d = X.shape[1]; index = faiss.IndexFlatIP(d); index.add(X); Xt = torch.tensor(X, device=device)
    id2idx = engine.node_id_to_idx
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}

    def prep(qs):
        q = encode_q([n.content for n, _, _ in qs])
        _, seed = index.search(q, 1)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed[:, 0], gold
    q_tr, seed_tr, gold_tr = prep(splits["train"])
    q_te, seed_te, gold_te = prep(splits["test"])
    qte = torch.tensor(q_te, device=device)
    avg_golds = round(float(np.mean([len(g) for g in gold_te if g])), 2)

    log.info(f"[{dataset}/{encoder}={tag}] dim={d} n_tr={len(q_tr)} n_te={len(q_te)} avg_golds={avg_golds}: train g1,g_hard,g2...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs)

    def pos(h):
        with torch.no_grad():
            return h(qte, Xt[[int(s) for s in seed_te]]).cpu().numpy()
    dense = _order(q_te, index)
    rel_hard = _order(pos(g_hard), index)
    hop1 = _order(pos(g1), index); s1 = hop1[:, 0]
    with torch.no_grad():
        hop2 = _order(g2(qte, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
    rel_2hop = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    champ = _rrf_fuse([_ranks(dense), _ranks(rel_hard), _ranks(rel_2hop)], [1.0, 1.0, 1.0])

    cfg = {"dense": _metrics(dense, gold_te), "rel_hard": _metrics(rel_hard, gold_te),
           "rel_2hop": _metrics(rel_2hop, gold_te), "CHAMPION": _metrics(champ, gold_te)}
    out = {"dataset": dataset, "encoder": encoder, "model": tag, "dim": d, "avg_golds_per_q": avg_golds,
           "n_test": len([g for g in gold_te if g]), "budgets": list(BUDGETS), "configs": cfg,
           "champion_over_dense": {m: round(cfg["CHAMPION"][m] - cfg["dense"][m], 2) for m in cfg["dense"]}}
    os.makedirs(os.path.join("data", "ukb_storage", dataset, "results", "L1"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", f"strong_encoder__{encoder}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    c, dv = cfg["CHAMPION"], cfg["dense"]
    log.info(f"[{dataset}/{encoder}] R@5 dense {dv['recall@5']} -> champ {c['recall@5']} ({out['champion_over_dense']['recall@5']:+}) | "
             f"R@100 {dv['recall@100']}->{c['recall@100']} | FCOV@100 {dv['fullcov@100']}->{c['fullcov@100']} "
             f"({out['champion_over_dense']['fullcov@100']:+})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Strong-encoder A/B: does relational L1 survive bge/e5? (Recall@2/5 + FullCov)")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean"])
    p.add_argument("--encoders", nargs="+", default=["minilm", "bge_large", "e5_large"], choices=list(SUBDIR))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=12000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        for enc in a.encoders:
            log.info(f"===== STRONG-ENCODER A/B: {ds.upper()} / {enc} =====")
            try:
                results[f"{ds}/{enc}"] = run(ds, encoder=enc, epochs=a.epochs, limit=a.limit)
            except Exception as e:
                log.exception(f"[{ds}/{enc}] FAILED: {e}")
            gc.collect()
    if results:
        summary = {k: {"model": r["model"], "dense_R@5": r["configs"]["dense"]["recall@5"],
                       "champ_R@5": r["configs"]["CHAMPION"]["recall@5"],
                       "dense_FCOV@100": r["configs"]["dense"]["fullcov@100"],
                       "champ_FCOV@100": r["configs"]["CHAMPION"]["fullcov@100"],
                       "champ_over_dense_FCOV@100": r["champion_over_dense"]["fullcov@100"]}
                   for k, r in results.items()}
        with open(os.path.join("data", "ukb_storage", "_index", "l1_strong_encoder_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"STRONG-ENCODER SUMMARY: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
