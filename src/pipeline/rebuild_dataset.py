"""
Reproducible clean-substrate rebuild for the two datasets whose golds were fixed.
================================================================================
Why this exists — the 2026-08 gold-count audit found two broken substrates:

  * squad_clean    : loaders.py back-linked the doc<->doc chain to ``nodes[-1]``
                     (the previous paragraph's last QUESTION node), giving ~14% of
                     questions a spurious 2nd gold. Fixed in loaders.py -> exactly
                     1 gold/question.
  * hotpotqa_clean : the old substrate was built from a FlashRAG dump whose
                     ``context`` was missing the gold articles for ~42% of
                     questions. Rebuilt from the OFFICIAL HotpotQA *distractor*
                     split (HuggingFace ``hotpotqa/hotpot_qa`` @ pinned revision,
                     train+dev merged) -> exactly 2 golds/question.

The other three sources (metaqa, musique_clean, 2wiki_clean) were audited sound
and are NOT touched here.

Two-stage rebuild (keeps the GPU work on Modal):
  1. LOCAL (this module): loaders -> per-source raw master -> build_clean ->
     data/processed/master_nodes_{src}_clean.json    (CPU only, no encoder).
  2. REMOTE (Modal 'rebuild-clean' task): build_all substrate + reencode
     (bge_large + gte_qwen) from the uploaded clean master.

Usage:
    python -m src.pipeline.rebuild_dataset --dataset squad
    python -m src.pipeline.rebuild_dataset --dataset hotpotqa
"""
import os
import json
import logging
import argparse

from src.pipeline.loaders import load_squad, load_hotpotqa
from src.pipeline.standardizer import save_nodes
from src.pipeline.build_clean import build_clean

log = logging.getLogger("pipeline.rebuild_dataset")

PROCESSED = "data/processed"
SQUAD_RAW = "data/raw/squad_v2.json"
HOTPOT_DEV = "data/raw/review_public/hotpot_dev_distractor.jsonl"
HOTPOT_TRAIN = "data/raw/review_public/hotpot_train_distractor.jsonl"
HOTPOT_COMBINED = "data/raw/review_public/hotpot_combined_distractor.jsonl"


def rebuild_squad(raw=SQUAD_RAW):
    """Official SQuAD v2 -> squad-only raw master -> master_nodes_squad_clean.json."""
    nodes = load_squad(raw)
    raw_master = f"{PROCESSED}/master_nodes_squad_raw.json"
    save_nodes(nodes, raw_master)
    log.info("squad raw master: %s (%d nodes)", raw_master, len(nodes))
    return build_clean("squad", master=raw_master)


def _merge_jsonl(sources, out):
    """Concatenate JSONL files (identical schema) into one, streaming line by line."""
    n = 0
    with open(out, "w", encoding="utf-8", newline="\n") as w:
        for src in sources:
            with open(src, "r", encoding="utf-8") as r:
                for line in r:
                    if line.strip():
                        w.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    log.info("merged %s -> %s (%d records)", sources, out, n)
    return out


def rebuild_hotpotqa(train=HOTPOT_TRAIN, dev=HOTPOT_DEV):
    """Official HotpotQA distractor (train+dev) -> hotpot raw master ->
    master_nodes_hotpotqa_clean.json. Both files share the HF distractor schema
    (context={title,sentences}, supporting_facts={title,sent_id}); merging them
    into one loader pass keeps a single article_cache so doc ids stay unique and
    titles dedup across splits."""
    sources = [p for p in (dev, train) if os.path.exists(p)]
    if not sources:
        raise FileNotFoundError(f"no hotpot distractor files found: {dev}, {train}")
    _merge_jsonl(sources, HOTPOT_COMBINED)
    nodes = load_hotpotqa(HOTPOT_COMBINED)
    raw_master = f"{PROCESSED}/master_nodes_hotpotqa_raw.json"
    save_nodes(nodes, raw_master)
    log.info("hotpot raw master: %s (%d nodes)", raw_master, len(nodes))
    return build_clean("hotpotqa", master=raw_master)


def main(argv=None):
    p = argparse.ArgumentParser(description="Rebuild a clean master for an audited-broken dataset.")
    p.add_argument("--dataset", required=True, choices=["squad", "hotpotqa"])
    a = p.parse_args(argv)
    out = rebuild_squad() if a.dataset == "squad" else rebuild_hotpotqa()
    log.info("clean master -> %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
