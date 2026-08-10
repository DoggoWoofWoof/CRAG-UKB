"""
Learned multi-label partition router — predict the SET of gold partitions from the query.
=========================================================================================
The partition probe showed: a query's golds span only ~2 partitions and are partition-
REACHABLE (oracle FullCov 86-92 on musique/2wiki), but centroid-similarity PREDICTS those
partitions poorly (9-25), leaving a 60-70pt oracle-vs-predicted gap — biggest on 2wiki
(oracle 86 vs champion 49). So the lever is a LEARNED predictor of the gold-partition SET
(user's idea: train on the golds' partitions, incl multi-hop), not more overlap.

Model: query embedding -> Linear(d, n_partitions) -> sigmoid, multi-label BCE against the
golds' hard-partition set. Retrieve: top-P predicted partitions -> top quota/partition by
query cosine. Reported vs dense / oracle-partition / learned-router / (dense (RRF) router)
at Recall + FullCov. If the learned router closes the oracle gap (esp 2wiki), it's a real
L1 lever for the multi-hop golds — no traversal. Writes L1/partition_router.json.
"""
import os
import json
import math
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
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, INIT_SEED
from src.experiments.l1_ablate import _ranks, _rrf_fuse, _recall, KS, MAXK

log = logging.getLogger("experiments.l1_partition_router")
BUDGETS = KS
TOPP = 20


def _fullcov(order, gold, budgets=BUDGETS):
    out = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g); top = order[qi] if isinstance(order[qi], list) else order[qi].tolist()
        for b in budgets:
            out[b].append(1.0 if gs <= set(top[:b]) else 0.0)
    return {b: round(float(np.mean(out[b])) * 100, 2) for b in budgets}


def _both(order, gold):
    return {"recall": _recall(order, gold), "fullcov": _fullcov(order, gold)}


def _route_order(part_scores, qsim, docs_by_part, topp=TOPP, maxb=MAXK):
    """Per query: top-P partitions by score -> top quota/partition by query cosine -> order."""
    nq = part_scores.shape[0]; out = []
    for qi in range(nq):
        pp = np.argsort(-part_scores[qi])[:topp]
        per = max(1, maxb // topp); cand = []
        for p in pp:
            idxs = docs_by_part[int(p)]
            if len(idxs):
                cand += [int(x) for x in idxs[np.argsort(-qsim[qi, idxs])[:per]]]
        cand = list(dict.fromkeys(cand))
        seen = set(cand)
        cand += [int(d) for d in np.argsort(-qsim[qi]) if int(d) not in seen][:maxb - len(cand)]
        out.append(cand[:maxb])
    return out


def run(dataset, epochs=40, limit=8000, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); enc = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n, d = X.shape; index = faiss.IndexFlatIP(d); index.add(X)
    id2idx = engine.node_id_to_idx
    hard = np.array([int(engine.partition_map.get(nd.node_id, -1)) for nd in engine.nodes])
    npart = int(hard.max()) + 1
    docs_by_part = [np.where(hard == p)[0] for p in range(npart)]
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}

    def prep(qs):
        q = enc.encode([nd.content for nd, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        gparts = [sorted(set(int(hard[x]) for x in g if hard[x] >= 0)) for g in gold]
        return q, gold, gparts
    q_tr, gold_tr, gp_tr = prep(splits["train"])
    q_te, gold_te, gp_te = prep(splits["test"])

    # learned multi-label partition classifier: query -> partitions containing golds
    torch.manual_seed(INIT_SEED)
    clf = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, npart)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    Qtr = torch.tensor(q_tr, device=device)
    Ytr = torch.zeros(len(q_tr), npart, device=device)
    for i, ps in enumerate(gp_tr):
        for p in ps:
            Ytr[i, p] = 1.0
    idx = [i for i, ps in enumerate(gp_tr) if ps]
    log.info(f"[{dataset}] training partition classifier (npart={npart}, n_tr={len(idx)})...")
    for ep in range(epochs):
        clf.train(); random.Random(ep).shuffle(idx)
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            loss = F.binary_cross_entropy_with_logits(clf(Qtr[b]), Ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        part_scores = clf(torch.tensor(q_te, device=device)).cpu().numpy()

    qsim = q_te @ X.T
    dense_order = np.argsort(-qsim, axis=1)[:, :MAXK]
    router_order = _route_order(part_scores, qsim, docs_by_part)
    # oracle: route to the golds' true partitions
    oracle_scores = np.full((len(q_te), npart), -1e9, np.float32)
    for i, ps in enumerate(gp_te):
        for p in ps:
            oracle_scores[i, p] = 1.0
    oracle_order = _route_order(oracle_scores, qsim, docs_by_part)

    cfg = {
        "dense": _both(dense_order, gold_te),
        "oracle_partition": _both(oracle_order, gold_te),
        "learned_router": _both(router_order, gold_te),
        "dense+router": _both(_rrf_fuse([_ranks(dense_order), _ranks(router_order)], [1.0, 1.0]), gold_te),
    }
    # partition-prediction quality
    hit = []
    for i, ps in enumerate(gp_te):
        if not ps:
            continue
        pp = set(np.argsort(-part_scores[i])[:TOPP].tolist())
        hit.append(len(set(ps) & pp) / len(ps))
    out = {"dataset": dataset, "n_partitions": npart, "topP": TOPP, "budgets": BUDGETS,
           "avg_golds": round(float(np.mean([len(g) for g in gold_te if g])), 2),
           "pred_partition_recall@P": round(float(np.mean(hit)) * 100, 2),
           "configs": cfg,
           "router_vs_dense@100": {m: round(cfg["learned_router"][m][100] - cfg["dense"][m][100], 2) for m in ("recall", "fullcov")},
           "fused_vs_dense@100": {m: round(cfg["dense+router"][m][100] - cfg["dense"][m][100], 2) for m in ("recall", "fullcov")}}
    os.makedirs(os.path.join("data", "ukb_storage", dataset, "results", "L1"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "partition_router.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] pred-part-recall@{TOPP} {out['pred_partition_recall@P']}% | FCOV@100 dense {cfg['dense']['fullcov'][100]} "
             f"| learned_router {cfg['learned_router']['fullcov'][100]} (vs dense {out['router_vs_dense@100']['fullcov']:+}) "
             f"| dense+router {cfg['dense+router']['fullcov'][100]} | oracle {cfg['oracle_partition']['fullcov'][100]}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Learned multi-label partition router (predict gold-partition set).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--limit", type=int, default=8000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 LEARNED PARTITION ROUTER: {ds.upper()} =====")
        try:
            run(ds, epochs=a.epochs, limit=a.limit)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
