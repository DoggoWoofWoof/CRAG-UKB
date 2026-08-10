"""
Build STANDARD-setting (official DEV *distractor*) clean masters for 4 QA datasets.
==================================================================================
The existing ``*_clean`` substrates were built from TRAIN with a custom 70/20/10
split, and musique/squad are effectively *all-gold pools* (corpus ~= only the
answer docs). This module rebuilds each dataset in the STANDARD setting:

    corpus  = the union of the OFFICIAL DEV set's gold + distractor paragraphs
    queries = the dev questions

Outputs carry the ``_std_clean`` suffix so the existing ``*_clean`` files are NOT
overwritten:

    data/processed/master_nodes_musique_std_clean.json
    data/processed/master_nodes_2wiki_std_clean.json
    data/processed/master_nodes_squad_std_clean.json
    data/processed/master_nodes_hotpot_std_clean.json

CPU-ONLY. Pipeline per dataset:  loader  ->  raw master (json)  ->  build_clean
(dedup by (title, content-hash) + label-free title-mention edges).  NO encoding,
NO indexing, NO Modal.

Usage:
    python -m src.pipeline.build_std_masters                # all four
    python -m src.pipeline.build_std_masters --datasets musique_std squad_std
"""
import os
import json
import time
import logging
import argparse
import hashlib

from src.pipeline.loaders import (
    load_squad,
    load_hotpotqa,
    load_2wiki,
    load_musique_ans,
)
from src.pipeline.standardizer import save_nodes
from src.pipeline.build_clean import build_clean

log = logging.getLogger("pipeline.build_std_masters")

PROCESSED = "data/processed"

# ds_std name -> (loader, dev source path, base node-id/source prefix produced by loader)
JOBS = {
    "musique_std": (load_musique_ans, "data/raw/full/musique_ans_dev.jsonl", "musique"),
    "2wiki_std":   (load_2wiki,       "data/raw/full/2wiki_dev.jsonl",       "2wiki"),
    "squad_std":   (load_squad,       "data/raw/full/squad_dev.json",        "squad"),
    "hotpot_std":  (load_hotpotqa,    "data/raw/review_public/hotpot_dev_distractor.jsonl", "hotpot"),
}


def _restamp_std(nodes, base, ds_std):
    """Move a loader's ``base``-namespaced nodes into the ``ds_std`` namespace.

    build_clean(ds) keeps only nodes whose ``metadata['source'] == ds`` and writes
    ``master_nodes_{ds}_clean.json``; it also renames questions via
    ``node_id.replace(f'{ds}_q', f'{ds}_clean_q', 1)``. Stamping source + id prefix
    to ``ds_std`` therefore (a) makes the source filter match, (b) lands the output
    on the ``_std`` name instead of the base ``_clean`` file, and (c) fires the
    question-id rename. Neighbor references are rewritten through the same id map so
    gold q->doc edges survive.
    """
    id_map = {}
    for n in nodes:
        old = n.node_id
        new = (ds_std + old[len(base):]) if old.startswith(base + "_") else old
        id_map[old] = new
    for n in nodes:
        n.node_id = id_map.get(n.node_id, n.node_id)
        n.neighbors = [id_map.get(nb, nb) for nb in n.neighbors]
        n.metadata["source"] = ds_std
    return nodes


def build_one(ds_std):
    loader, src_path, base = JOBS[ds_std]
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"dev source not found for {ds_std}: {src_path}")
    t0 = time.time()
    log.info("[%s] loading %s ...", ds_std, src_path)
    nodes = loader(src_path)
    _restamp_std(nodes, base, ds_std)
    raw_master = f"{PROCESSED}/master_nodes_{ds_std}_raw.json"
    save_nodes(nodes, raw_master)
    log.info("[%s] raw master: %s (%d nodes, %.1fs)", ds_std, raw_master, len(nodes), time.time() - t0)
    out = build_clean(ds_std, master=raw_master)
    log.info("[%s] clean master -> %s (%.1fs total)", ds_std, out, time.time() - t0)
    return out


def _stats(ds_std, path):
    """corpus docs, questions, mean gold docs/q, all-gold ratio (unique golds/corpus)."""
    data = json.load(open(path, encoding="utf-8"))
    docs = [n for n in data if n["metadata"].get("type") != "question"]
    qs = [n for n in data if n["metadata"].get("type") == "question"]
    n_docs = len(docs)
    n_qs = len(qs)
    gold_counts = [len(q["neighbors"]) for q in qs]
    mean_gold = (sum(gold_counts) / n_qs) if n_qs else 0.0
    unique_golds = set()
    for q in qs:
        unique_golds.update(q["neighbors"])
    all_gold_ratio = (len(unique_golds) / n_docs) if n_docs else 0.0
    return {
        "ds": ds_std,
        "corpus_docs": n_docs,
        "questions": n_qs,
        "mean_gold_per_q": mean_gold,
        "unique_gold_docs": len(unique_golds),
        "all_gold_ratio": all_gold_ratio,
        "path": path,
    }


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build standard-setting (dev distractor) clean masters.")
    p.add_argument("--datasets", nargs="+", default=list(JOBS.keys()), choices=list(JOBS.keys()))
    a = p.parse_args(argv)

    results = []
    for ds_std in a.datasets:
        out = build_one(ds_std)
        results.append(_stats(ds_std, out))

    # ── stats table ─────────────────────────────────────────────────
    baseline = {  # current all-gold-pool _clean corpus doc counts (for comparison)
        "musique_std": 13672, "2wiki_std": 65865, "squad_std": 19029, "hotpot_std": 507494,
    }
    print("\n" + "=" * 96)
    print("STANDARD (official dev distractor) clean masters — stats")
    print("=" * 96)
    hdr = f"{'dataset':13s} {'corpus_docs':>12s} {'questions':>10s} {'mean_gold/q':>12s} {'uniq_golds':>11s} {'all_gold_ratio':>15s} {'old_clean_docs':>15s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['ds']:13s} {r['corpus_docs']:12d} {r['questions']:10d} "
              f"{r['mean_gold_per_q']:12.3f} {r['unique_gold_docs']:11d} "
              f"{r['all_gold_ratio']:15.4f} {baseline.get(r['ds'], 0):15d}")
    print("=" * 96)
    for r in results:
        print(f"  {r['ds']:13s} -> {r['path']}")


if __name__ == "__main__":
    main()
