"""
Universal + residual relational retriever (the two novelty pillars).
====================================================================
Two ideas that push past the incremental single-hop result:

PILLAR 1 — ONE universal MLP across ALL datasets (generalization novelty).
  The offset g(q) maps a query to a relation direction in the SHARED frozen
  MiniLM space, so a single head can be trained on POOLED triples from every
  dataset and deployed on all of them. Competitors are per-corpus (HopRAG rebuilds
  a graph per corpus; HyDE calls an LLM per query). A single corpus-agnostic learned
  retriever — train once, deploy anywhere in the space — is the stronger claim.
  Tested: universal head vs per-dataset heads (does it retain the gains?).

PILLAR 2 — RESIDUAL / complementarity-driven training (overlap IS the novelty).
  The overlap analysis shows rel reaches golds dense MISSES. Turn that observation
  into the training objective: up-weight golds dense ranks poorly, so the offset
  spends its capacity covering dense's blind spots instead of redundantly re-finding
  what dense already has. This is boosting/residual-learning for retrieval — fit the
  second retriever to the first's errors -> maximally complementary -> bigger fusion.
  Tested: residual vs standard loss (rel recall, rel_only overlap, FUSED recall).

2x2 = {perdataset, universal} x {standard, residual}, each eval'd per dataset with
dense baseline + RRF fusion + overlap. Writes _index/l1_universal_summary.json.
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
from src.experiments.l1_ablate import _order, _ranks, _rrf_fuse, _recall, _overlap, KS, MAXK, TAU

log = logging.getLogger("experiments.l1_universal")
DENSE_MISS_K = 50          # a gold outside dense top-K counts as "dense missed it"
W_MISS, W_HIT = 1.0, 0.3   # residual loss weights: focus capacity on dense's misses


def load_ds(ds, encoder, limit, maxtrip, device):
    engine = CoreEngine(source=ds)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
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

    # dense rank of each train gold -> residual "dense missed it" flag
    _, dense_tr = index.search(q_tr, DENSE_MISS_K)
    miss_tr = [set(row.tolist()) for row in dense_tr]
    trip = [(i, int(seed_tr[i]), int(g), g not in miss_tr[i])
            for i, gl in enumerate(gold_tr) for g in gl]
    random.Random(INIT_SEED).shuffle(trip)
    trip = trip[:maxtrip]

    dense_te = _order(q_te, index)
    return {"X": X, "index": index, "q_tr": q_tr, "trip": trip,
            "q_te": q_te, "seed_te": seed_te, "gold_te": gold_te,
            "dense_te": dense_te, "dense_recall": _recall(dense_te, gold_te)}


def train(stores, scope, residual, epochs, device):
    d = stores[next(iter(stores))]["X"].shape[1]
    torch.manual_seed(INIT_SEED)
    if scope == "universal":
        head = OffsetHead(d).to(device); opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        heads, opts = None, None
    else:
        heads = {ds: OffsetHead(d).to(device) for ds in stores}
        opts = {ds: torch.optim.Adam(heads[ds].parameters(), lr=1e-3) for ds in stores}
        head, opt = None, None
    ds_list = list(stores)
    for ep in range(epochs):
        batches = []
        for di, ds in enumerate(ds_list):
            trip = list(stores[ds]["trip"]); random.Random(ep * 100 + di).shuffle(trip)
            for s in range(0, len(trip), 256):
                batches.append((ds, trip[s:s + 256]))
        random.Random(ep).shuffle(batches)                       # interleave datasets
        for ds, b in batches:
            st = stores[ds]
            h = head if scope == "universal" else heads[ds]
            o = opt if scope == "universal" else opts[ds]
            qn = torch.tensor(st["q_tr"][[t[0] for t in b]], device=device)
            seed = torch.tensor(st["X"][[t[1] for t in b]], device=device)
            gold = torch.tensor(st["X"][[t[2] for t in b]], device=device)
            pred = h(qn, seed)
            logits = pred @ gold.T / TAU                          # in-batch (within-dataset) negatives
            ce = F.cross_entropy(logits, torch.arange(len(b), device=device), reduction="none")
            if residual:
                w = torch.tensor([W_HIT if t[3] else W_MISS for t in b], device=device)
                loss = (ce * w).sum() / w.sum()
            else:
                loss = ce.mean()
            o.zero_grad(); loss.backward(); o.step()
    return head, heads


def evaluate(st, head, device):
    with torch.no_grad():
        pos = head(torch.tensor(st["q_te"], device=device),
                   torch.tensor(st["X"][[int(s) for s in st["seed_te"]]], device=device)).cpu().numpy()
    rel_order = _order(pos, st["index"])
    fused = _rrf_fuse([_ranks(st["dense_te"]), _ranks(rel_order)], [1.0, 1.0])
    return {"rel": _recall(rel_order, st["gold_te"]),
            "fused": _recall(fused, st["gold_te"]),
            "overlap": _overlap(st["dense_te"], rel_order, st["gold_te"], 100)}


def run(datasets, epochs=20, limit=5000, maxtrip=15000, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DenseEncoder()
    log.info(f"loading {len(datasets)} datasets (limit {limit})...")
    stores = {ds: load_ds(ds, encoder, limit, maxtrip, device) for ds in datasets}

    results = {ds: {"dense": stores[ds]["dense_recall"]} for ds in datasets}
    for scope in ("perdataset", "universal"):
        for residual in (False, True):
            tag = f"{scope}_{'residual' if residual else 'standard'}"
            log.info(f"training {tag}...")
            head, heads = train(stores, scope, residual, epochs, device)
            for ds in datasets:
                h = head if scope == "universal" else heads[ds]
                results[ds][tag] = evaluate(stores[ds], h, device)
            log.info(f"  {tag}: " + " | ".join(
                f"{ds} fused@100 {results[ds][tag]['fused'][100]}" for ds in datasets))

    # ---- summary: generalization gap (univ vs perds) + complementarity gain (residual vs standard)
    summary = {"datasets": datasets, "epochs": epochs, "limit": limit, "per_dataset": {}}
    for ds in datasets:
        r = results[ds]
        summary["per_dataset"][ds] = {
            "dense@100": r["dense"][100],
            "perds_std":  {"rel": r["perdataset_standard"]["rel"][100], "fused": r["perdataset_standard"]["fused"][100],
                           "rel_only": r["perdataset_standard"]["overlap"]["rel_only"]},
            "perds_res":  {"rel": r["perdataset_residual"]["rel"][100], "fused": r["perdataset_residual"]["fused"][100],
                           "rel_only": r["perdataset_residual"]["overlap"]["rel_only"]},
            "univ_std":   {"rel": r["universal_standard"]["rel"][100], "fused": r["universal_standard"]["fused"][100],
                           "rel_only": r["universal_standard"]["overlap"]["rel_only"]},
            "univ_res":   {"rel": r["universal_residual"]["rel"][100], "fused": r["universal_residual"]["fused"][100],
                           "rel_only": r["universal_residual"]["overlap"]["rel_only"]},
        }
    # aggregate signals
    def mean(sel):
        return round(float(np.mean([sel(summary["per_dataset"][ds]) for ds in datasets])), 2)
    summary["aggregate"] = {
        "univ_vs_perds_fused_gap@100": mean(lambda p: p["univ_std"]["fused"] - p["perds_std"]["fused"]),
        "residual_vs_standard_fused_gain@100(perds)": mean(lambda p: p["perds_res"]["fused"] - p["perds_std"]["fused"]),
        "residual_vs_standard_relonly_gain(perds)": mean(lambda p: p["perds_res"]["rel_only"] - p["perds_std"]["rel_only"]),
        "best_universal_fused@100": mean(lambda p: max(p["univ_std"]["fused"], p["univ_res"]["fused"])),
        "best_perds_fused@100": mean(lambda p: max(p["perds_std"]["fused"], p["perds_res"]["fused"])),
    }
    path = os.path.join("data", "ukb_storage", "_index", "l1_universal_summary.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"AGGREGATE {summary['aggregate']}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Universal cross-dataset head + residual complementarity training.")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--limit", type=int, default=5000)
    p.add_argument("--maxtrip", type=int, default=15000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("===== L1 UNIVERSAL + RESIDUAL =====")
    run(a.datasets, epochs=a.epochs, limit=a.limit, maxtrip=a.maxtrip)


if __name__ == "__main__":
    main()
