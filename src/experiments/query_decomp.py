"""
Query-decomposition routing (close the multi-ENTITY tail at L1).
================================================================
Multi-gold queries split into two kinds:
  - multi-ENTITY (2wiki: "director of film A and film B ...") — several distinct
    entities, each in its own partition. One averaged query vector ranks all of
    them mid; routing each entity SEPARATELY and unioning finds each precisely.
  - single-entity breadth (metaqa: "movies by [X]") — ONE entity, many scattered
    answers. Nothing to decompose; expected to NOT benefit (honest control).

This routes the full question (baseline) vs decomposed sub-queries (full + entity
spans, each -> top-k', unioned). Compares the two as (avg partitions retrieved,
FullCov) frontiers on the SAME trained router + membership, so a win = higher
FullCov at equal partitions retrieved. Entity extraction is a local no-LLM
heuristic (bracketed entities + Title-Case spans split on and/or/commas).
Writes results/query_decomp/{dataset}_{config}.json.
"""
import os
import re
import json
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _centroids, _splits, _train, membership_ref, TAU, HNK
from src.experiments.adaptive_k import _build

log = logging.getLogger("experiments.query_decomp")

_SEP = re.compile(r"\band\b|\bor\b|,|;|\bwhose\b|\bhave\b")
_ENT = re.compile(r"[A-Z][\w.'’&:\-]*(?:\s+(?:[A-Z][\w.'’&:\-]*|of|the|to|de|for|a|an|in|\([^)]*\)))*",
                  re.UNICODE)
_LEAD = re.compile(r"^(Are|Do|Does|Which|What|Who|Where|When|Is|Both|The)\b\s*")


def decompose(q):
    subs = [q]
    for b in re.findall(r"\[([^\]]+)\]", q):
        if b.strip():
            subs.append(b.strip())
    for frag in _SEP.split(q):
        for s in _ENT.findall(frag):
            s = _LEAD.sub("", s.strip()).strip()
            if len(s) > 2 and s.lower() not in ("film", "films", "director", "directors", "the"):
                subs.append(s)
    return list(dict.fromkeys(subs))


def run_dataset(dataset, config="hard", epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    log.info(f"===== QUERY-DECOMP: {dataset.upper()} config={config} =====")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1

    membership = _build(engine, node_vecs, config)
    membership_ref[config] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    model, best_state, final_state, Cg = _train(
        engine, C, splits, split_embs, device, tau, hn_k, epochs,
        os.path.join("logs", dataset, f"qdecomp_{config.replace('+','_')}"), config, "kl")
    model.load_state_dict(final_state); model.eval()

    test = splits["test"]
    gold_docs = [golds for _, _, golds in test]

    def _rank(embs):
        with torch.no_grad():
            proj = F.normalize(model(torch.tensor(embs, dtype=torch.float32, device=device)), dim=-1)
            return torch.argsort(-(proj @ Cg.T), dim=1).cpu().numpy()

    # baseline: full-question ranking
    full_rank = _rank(split_embs["test"])

    # decomposed: encode all sub-queries once, keep per-query rank lists
    sub_lists = [decompose(q.content) for q, _, _ in test]
    n_sub = [len(s) for s in sub_lists]
    flat = [s for subs in sub_lists for s in subs]
    flat_emb = encoder.encode(flat).astype("float32")
    flat_rank = _rank(flat_emb)
    # map back
    per_q_ranks, off = [], 0
    for k in n_sub:
        per_q_ranks.append(flat_rank[off:off + k]); off += k

    def _cov(retrieved_sets):
        fc, gtr, nparts = [], [], []
        for qi, rset in enumerate(retrieved_sets):
            golds = [g for g in gold_docs[qi] if g in membership]
            if not golds:
                continue
            cov = [g for g in golds if membership[g] & rset]
            fc.append(1.0 if len(cov) == len(golds) else 0.0)
            gtr.append(len(cov) / len(golds))
            nparts.append(len(rset))
        return (round(float(np.mean(fc)) * 100, 2), round(float(np.mean(gtr)) * 100, 2),
                round(float(np.mean(nparts)), 1))

    out = {"dataset": dataset, "config": config, "n_test": len(test),
           "avg_subqueries": round(float(np.mean(n_sub)), 2), "baseline": [], "decomposed": []}
    for K in [3, 5, 10, 20, 30]:
        rsets = [set(full_rank[qi][:K].tolist()) for qi in range(len(test))]
        fc, gtr, npt = _cov(rsets)
        out["baseline"].append({"K": K, "avg_parts": npt, "full_coverage": fc, "gt_recall": gtr})
    for kp in [2, 3, 5, 8, 12]:
        rsets = []
        for qi in range(len(test)):
            s = set()
            for r in per_q_ranks[qi]:
                s.update(r[:kp].tolist())
            rsets.append(s)
        fc, gtr, npt = _cov(rsets)
        out["decomposed"].append({"kp": kp, "avg_parts": npt, "full_coverage": fc, "gt_recall": gtr})

    out_dir = os.path.join("results", "query_decomp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}_{config.replace('+','_')}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"  avg sub-queries/q={out['avg_subqueries']}")
    log.info(f"  baseline  : {[(b['avg_parts'], b['full_coverage']) for b in out['baseline']]}")
    log.info(f"  decomposed: {[(d['avg_parts'], d['full_coverage']) for d in out['decomposed']]}")
    log.info(f"Saved results/query_decomp/{dataset}_{config.replace('+','_')}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Query-decomposition routing prototype.")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--configs", nargs="+", default=["hard"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        for cfg in a.configs:
            run_dataset(ds, config=cfg, epochs=a.epochs, limit=a.limit, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
