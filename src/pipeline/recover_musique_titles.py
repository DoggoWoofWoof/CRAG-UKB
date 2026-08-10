"""
Recover the REAL titles MuSiQue ships but our loader dropped, then build
title-mention edges (the relational structure L3 needs).
====================================================================
FlashRAG musique stores each supporting passage as a *stringified* Python dict
inside metadata.question_decomposition[*].support_paragraph, e.g.
  "{'idx': 5, 'title': 'Houston', 'paragraph_text': '...'}"
load_musique only kept paragraph_text, so the clean substrate has 0 title edges
(deg_struct=0.0) and L3 is inert on it. This:
  1. downloads musique train.jsonl if missing,
  2. parses support_paragraph (ast.literal_eval) -> {md5(paragraph_text): title},
  3. attaches titles to the existing master_nodes_musique_clean.json docs by
     content hash (reports coverage),
  4. builds label-free title-mention edges via build_clean._title_links,
  5. writes results/research/musique_title_edges.json (doc_id -> [neighbor ids])
     for the l1l3 A/B test, and data/processed/musique_title_map.json (hash->title)
     for the eventual clean rebuild.
Does NOT mutate the substrate; the rebuild is a separate, confirmed step.
"""
import os
import ast
import json
import hashlib
import logging
import urllib.request
from collections import defaultdict

from src.pipeline.build_clean import _title_links

log = logging.getLogger("pipeline.recover_musique_titles")

URL = "https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/train.jsonl"
RAW = "data/raw/musique.jsonl"
CLEAN = "data/processed/master_nodes_musique_clean.json"
EDGES_OUT = "results/research/musique_title_edges.json"
MAP_OUT = "data/processed/musique_title_map.json"


def _download():
    if os.path.exists(RAW) and os.path.getsize(RAW) > 1_000_000:
        log.info(f"raw present: {RAW} ({os.path.getsize(RAW)/1e6:.1f} MB)")
        return
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    log.info(f"downloading {URL} -> {RAW} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(RAW, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    log.info(f"  done ({os.path.getsize(RAW)/1e6:.1f} MB)")


def _norm(s):
    return " ".join(s.split())


def build_title_map():
    """md5(paragraph_text) -> title, from every support_paragraph in the source.
    Keys on both exact and whitespace-normalized text to survive minor reformatting."""
    tmap = {}
    n_lines = n_steps = n_ok = 0
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            for step in item.get("metadata", {}).get("question_decomposition", []):
                sp = step.get("support_paragraph")
                if not sp:
                    continue
                n_steps += 1
                try:
                    d = ast.literal_eval(sp) if isinstance(sp, str) else sp
                    title = (d.get("title") or "").strip()
                    text = d.get("paragraph_text") or d.get("content") or ""
                except (ValueError, SyntaxError, AttributeError):
                    continue
                if not title or not text:
                    continue
                n_ok += 1
                for key in (hashlib.md5(text.encode("utf-8")).hexdigest(),
                            hashlib.md5(_norm(text).encode("utf-8")).hexdigest()):
                    tmap.setdefault(key, title)
    log.info(f"  parsed {n_lines} questions, {n_steps} decomp steps, {n_ok} titled passages "
             f"-> {len(tmap)} content-hash keys")
    return tmap


def run(bake=False):
    _download()
    tmap = build_title_map()
    os.makedirs(os.path.dirname(MAP_OUT), exist_ok=True)
    with open(MAP_OUT, "w", encoding="utf-8") as f:
        json.dump(tmap, f)

    nodes = json.load(open(CLEAN, encoding="utf-8"))
    docs = [n for n in nodes if n["metadata"].get("type") != "question"]
    hit = 0
    for d in docs:
        c = d["content"]
        title = tmap.get(hashlib.md5(c.encode("utf-8")).hexdigest()) \
            or tmap.get(hashlib.md5(_norm(c).encode("utf-8")).hexdigest())
        d["metadata"]["title"] = title or ""
        d["neighbors"] = []                      # isolate title edges for the A/B test
        if title:
            hit += 1
    log.info(f"  matched titles to {hit}/{len(docs)} docs ({hit/len(docs)*100:.1f}%)")

    n_edges = _title_links(docs)
    linked = sum(1 for d in docs if d["neighbors"])
    deg = sum(len(d["neighbors"]) for d in docs) / max(len(docs), 1)
    log.info(f"  title-mention edges: {n_edges} directed adds, avg doc degree {deg:.2f} "
             f"({linked/len(docs)*100:.0f}% docs linked)")

    edges = {d["node_id"]: d["neighbors"] for d in docs if d["neighbors"]}
    os.makedirs(os.path.dirname(EDGES_OUT), exist_ok=True)
    with open(EDGES_OUT, "w", encoding="utf-8") as f:
        json.dump(edges, f)
    log.info(f"  wrote {EDGES_OUT} ({len(edges)} docs with edges) and {MAP_OUT}")

    if bake:
        # docs were mutated in place (title + title-edge neighbors); questions untouched.
        with open(CLEAN, "w", encoding="utf-8") as f:
            json.dump(nodes, f)
        log.info(f"  BAKED titles + title edges into {CLEAN} ({len(nodes)} nodes) — rebuild the index next")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", action="store_true", help="write titles + title edges back into the clean master")
    run(bake=ap.parse_args().bake)
