"""
Loss A/B for the relational offset head (rel_hard).
===================================================
The champion's rel_hard is trained InfoNCE (single-positive) + HNM. But this repo's
router already found InfoNCE COLLAPSES under aggressive HNM while KL / the Jigsaw
coverage loss stay robust (see src/alignment/README.md). And rel_hard's golds are
MULTI-positive (metaqa avg 7.31/query), which single-positive InfoNCE ignores. So
A/B the offset-head loss, all with the SAME hard-negative mining:

  info_nce    : multi-positive InfoNCE  (-log [ sum_g e^s_g / sum_c e^s_c ])
  kl_div      : soft multi-hot target,  KL(target || softmax(s))
  coverage_kl : kl_div + lam * Jigsaw partition_coverage_loss (CVaR weakest-positive
                + FullCov@K barrier + hardest-negative margin) — the "cover the whole
                answer set" loss, the right objective for 1-to-many.

All query-batched (needed for multi-positive) with a per-batch candidate set =
(batch golds) ∪ (faiss-mined hard negs near the prediction). Retrieval eval =
docs nearest pred = normalize(seed + g(q)); metric recall@{50,100,200,500}.
Run on the relational datasets (metaqa, musique, 2wiki). Winner -> swap into champion.
Writes {ds}/results/L1/loss_ab.json + _index/l1_loss_ab_summary.json.
"""
import os
import json
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, INIT_SEED
from src.experiments.query_relation import OffsetHead
from src.experiments.l1_ablate import _order, _recall, KS, MAXK
from src.alignment.coverage_losses import partition_coverage_loss

log = logging.getLogger("experiments.l1_loss_ab")
TAU = 0.05
HN_K = 16          # hard negatives mined per query (same aggressive HNM for all losses)
LAM_COV = 0.5
LOSSES = ["info_nce", "kl_div", "coverage_kl"]


def _batch_candidates(gold_sets, preds, index, Xt, device, hn_k=HN_K):
    """Per-batch candidate set = union(batch golds) ∪ faiss-mined hard negs near preds."""
    bg = set()
    for gs in gold_sets:
        bg.update(gs)
    if hn_k:
        with torch.no_grad():
            _, nn = index.search(np.ascontiguousarray(preds.detach().cpu().numpy().astype("float32")), hn_k)
        bg.update(int(x) for x in nn.reshape(-1) if x >= 0)
    cand = sorted(bg)
    pos = {c: j for j, c in enumerate(cand)}
    cv = Xt[torch.tensor(cand, device=device)]                      # (C, d)
    mask = torch.zeros(len(gold_sets), len(cand), dtype=torch.bool, device=device)
    plist = []
    for i, gs in enumerate(gold_sets):
        p = [pos[g] for g in gs]
        mask[i, p] = True
        plist.append(p)
    return cv, mask, plist


def _compute_loss(loss_type, preds, cv, mask, plist):
    logits = preds @ cv.T / TAU                                     # (B, C)
    if loss_type == "info_nce":                                     # multi-positive InfoNCE
        num = torch.logsumexp(logits.masked_fill(~mask, float("-inf")), dim=1)
        den = torch.logsumexp(logits, dim=1)
        return -(num - den).mean()
    tgt = mask.float(); tgt = tgt / tgt.sum(1, keepdim=True).clamp(min=1)
    kl = -(tgt * F.log_softmax(logits, dim=1)).sum(1).mean()        # soft multi-hot KL
    if loss_type == "kl_div":
        return kl
    cov = partition_coverage_loss(preds, plist, cv, temperature=TAU)  # Jigsaw coverage (docs as centroids)
    return kl + LAM_COV * cov


def _train(loss_type, q_tr, seed_tr, gold_tr, Xt, index, device, epochs, bs=128):
    idx = [i for i, g in enumerate(gold_tr) if g]
    qtr = torch.tensor(q_tr, device=device)
    torch.manual_seed(INIT_SEED)
    head = OffsetHead(Xt.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for ep in range(epochs):
        head.train(); random.Random(ep).shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            qn = qtr[b]; seed = Xt[[int(seed_tr[i]) for i in b]]
            preds = head(qn, seed)
            cv, mask, plist = _batch_candidates([gold_tr[i] for i in b], preds, index, Xt, device)
            loss = _compute_loss(loss_type, preds, cv, mask, plist)
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def _fullcov(order, gold, budgets=KS):
    out = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        top_all = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            out[b].append(1.0 if gs <= set(top_all[:b]) else 0.0)   # ALL golds present
    return {b: round(float(np.mean(out[b])) * 100, 2) for b in budgets}


def run(dataset, epochs=25, limit=8000, device=None):
    device = device or torch.device("cpu")
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
    avg_golds = round(float(np.mean([len(g) for g in gold_te if g])), 2)
    qte = torch.tensor(q_te, device=device); seed_te_v = Xt[[int(s) for s in seed_te]]

    def _both(order):
        return {"recall": _recall(order, gold_te), "fullcov": _fullcov(order, gold_te)}
    cfg = {"dense": _both(_order(q_te, index))}
    for lt in LOSSES:
        log.info(f"[{dataset}] training rel_hard loss={lt} (+HNM {HN_K})...")
        head = _train(lt, q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
        with torch.no_grad():
            pred = head(qte, seed_te_v).cpu().numpy()
        cfg[lt] = _both(_order(pred, index))
    out = {"dataset": dataset, "avg_golds_per_q": avg_golds, "hn_k": HN_K, "budgets": KS, "configs": cfg,
           "best_recall_loss": max(LOSSES, key=lambda l: cfg[l]["recall"][100]),
           "best_fullcov_loss": max(LOSSES, key=lambda l: cfg[l]["fullcov"][100]),
           "cov_vs_infonce_recall@100": round(cfg["coverage_kl"]["recall"][100] - cfg["info_nce"]["recall"][100], 2),
           "cov_vs_infonce_fullcov@100": round(cfg["coverage_kl"]["fullcov"][100] - cfg["info_nce"]["fullcov"][100], 2),
           "cov_vs_kl_fullcov@100": round(cfg["coverage_kl"]["fullcov"][100] - cfg["kl_div"]["fullcov"][100], 2)}
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "loss_ab.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] avg_golds {avg_golds} || R@100 dense {cfg['dense']['recall'][100]} "
             f"in {cfg['info_nce']['recall'][100]} kl {cfg['kl_div']['recall'][100]} cov {cfg['coverage_kl']['recall'][100]} "
             f"|| FCOV@100 dense {cfg['dense']['fullcov'][100]} in {cfg['info_nce']['fullcov'][100]} "
             f"kl {cfg['kl_div']['fullcov'][100]} cov {cfg['coverage_kl']['fullcov'][100]} "
             f"|| cov-vs-kl FCOV {out['cov_vs_kl_fullcov@100']:+}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="A/B the rel_hard offset-head loss (info_nce/kl_div/coverage_kl, +HNM).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=8000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 LOSS A/B: {ds.upper()} =====")
        try:
            results[ds] = run(ds, epochs=a.epochs, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
    if results:
        def m(k, metric):
            return round(float(np.mean([r["configs"][k][metric][100] for r in results.values()])), 2)
        summary = {"datasets": list(results),
                   "recall_mean@100": {l: m(l, "recall") for l in ["dense"] + LOSSES},
                   "fullcov_mean@100": {l: m(l, "fullcov") for l in ["dense"] + LOSSES}}
        summary["best_recall"] = max(LOSSES, key=lambda l: summary["recall_mean@100"][l])
        summary["best_fullcov"] = max(LOSSES, key=lambda l: summary["fullcov_mean@100"][l])
        summary["cov_vs_kl_fullcov_mean@100"] = round(
            summary["fullcov_mean@100"]["coverage_kl"] - summary["fullcov_mean@100"]["kl_div"], 2)
        with open(os.path.join("data", "ukb_storage", "_index", "l1_loss_ab_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"LOSS A/B: RECALL@100 {summary['recall_mean@100']} | FULLCOV@100 {summary['fullcov_mean@100']} "
                 f"-> best_recall={summary['best_recall']} best_fullcov={summary['best_fullcov']} "
                 f"(coverage_kl − kl_div on FullCov: {summary['cov_vs_kl_fullcov_mean@100']:+})")


if __name__ == "__main__":
    main()
