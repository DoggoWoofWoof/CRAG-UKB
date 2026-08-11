"""
Build HippoRAG-standard (1,000-Q candidate corpus) clean masters for the multi-hop trio.
=========================================================================================
Uses HippoRAG's exact released data under data/raw/full/hipporag/ so our numbers sit
directly next to HippoRAG / HippoRAG 2. Corpus = their {title,text} candidate passages
(~6-12k), queries = their 1,000-question sample. Output carries the ``_hpr`` suffix.

CPU-ONLY:  load_hipporag -> raw master -> build_clean (dedup + label-free title edges).
NO encoding, NO indexing, NO Modal.

    python -m src.pipeline.build_hpr_masters
"""
import os
import json
import time
import logging
import argparse

from src.pipeline.loader_hipporag import load_hipporag
from src.pipeline.standardizer import save_nodes
from src.pipeline.build_clean import build_clean

log = logging.getLogger("pipeline.build_hpr_masters")
PROCESSED = "data/processed"
HR = "data/raw/full/hipporag"

# ds_hpr -> (corpus_json, questions_json). load_hipporag stamps source=ds_hpr directly, so
# build_clean(ds_hpr) filters source==ds_hpr, renames q-ids, and writes master_nodes_{ds}_clean.json.
JOBS = {
    "musique_hpr": (f"{HR}/musique_corpus.json",         f"{HR}/musique.json"),
    "2wiki_hpr":   (f"{HR}/2wikimultihopqa_corpus.json", f"{HR}/2wikimultihopqa.json"),
    "hotpot_hpr":  (f"{HR}/hotpotqa_corpus.json",        f"{HR}/hotpotqa.json"),
}


def build_one(ds):
    corpus_p, q_p = JOBS[ds]
    for p in (corpus_p, q_p):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{ds}: missing {p}")
    t0 = time.time()
    nodes = load_hipporag(corpus_p, q_p, ds)
    raw = f"{PROCESSED}/master_nodes_{ds}_raw.json"
    save_nodes(nodes, raw)
    out = build_clean(ds, master=raw)
    log.info("[%s] clean master -> %s (%.1fs)", ds, out, time.time() - t0)
    return out


def _stats(ds, path):
    data = json.load(open(path, encoding="utf-8"))
    docs = [n for n in data if n["metadata"].get("type") != "question"]
    qs = [n for n in data if n["metadata"].get("type") == "question"]
    n_docs = len(docs) or 1
    gold_counts = [len(q["neighbors"]) for q in qs]
    uniq = set()
    for q in qs:
        uniq.update(q["neighbors"])
    zero = sum(1 for c in gold_counts if c == 0)
    return {"ds": ds, "corpus_docs": len(docs), "questions": len(qs),
            "mean_gold_per_q": (sum(gold_counts) / (len(qs) or 1)),
            "unique_golds": len(uniq), "all_gold_ratio": len(uniq) / n_docs,
            "zero_gold_q": zero, "path": path}


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Build HippoRAG-standard (_hpr) clean masters for the multi-hop trio.")
    p.add_argument("--datasets", nargs="+", default=list(JOBS.keys()), choices=list(JOBS.keys()))
    a = p.parse_args(argv)
    results = [_stats(ds, build_one(ds)) for ds in a.datasets]

    exp = {"musique_hpr": 11656, "2wiki_hpr": 6119, "hotpot_hpr": 9811}   # HippoRAG released corpus sizes
    print("\n" + "=" * 100)
    print("HippoRAG-standard (_hpr) clean masters — stats")
    print("=" * 100)
    hdr = f"{'dataset':13s} {'corpus':>8s} {'questions':>10s} {'mean_gold/q':>12s} {'uniq_golds':>11s} {'all_gold_ratio':>15s} {'zero_gold_q':>12s} {'expected_corpus':>15s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['ds']:13s} {r['corpus_docs']:8d} {r['questions']:10d} {r['mean_gold_per_q']:12.3f} "
              f"{r['unique_golds']:11d} {r['all_gold_ratio']:15.4f} {r['zero_gold_q']:12d} {exp.get(r['ds'],0):15d}")
    print("=" * 100)
    for r in results:
        print(f"  {r['ds']:13s} -> {r['path']}")


if __name__ == "__main__":
    main()
