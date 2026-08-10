"""
Fast UKB build for a "{ds}_clean" source by REUSING existing embeddings.
========================================================================
The clean docs are deduplicated originals with IDENTICAL content, so their
MiniLM embeddings already exist in the original {ds}/nodes.index — re-encoding
(2+ hours for doc passages) is wasted. This maps each clean doc to its embedding
by content hash, writes nodes.index, then runs the indexer's (encoding-free)
graph / METIS / centroid / bm25 steps. ~2 min instead of ~2.3 h.

node.neighbors in the clean master = label-free title links (overlap1 substrate);
build_pyg_graph adds kNN edges -> graph.pt (so synthetic_neighbors = kNN = syn1);
centroids are degree-weighted by the (label-free) neighbor count.
"""
import os
import json
import hashlib
import logging
import argparse

import numpy as np
import faiss

from src.pipeline.standardizer import load_nodes
from src.core.engine import CoreEngine
from src.core import indexers

log = logging.getLogger("pipeline.build_clean_index")


def _content_hash_to_vec(orig_source):
    eng = CoreEngine(source=orig_source)
    idx = eng.node_index
    vecs = np.array([idx.reconstruct(i) for i in range(idx.ntotal)], dtype=np.float32)
    h2v = {}
    for i, n in enumerate(eng.nodes):
        h = hashlib.md5(n.content.encode("utf-8")).hexdigest()
        if h not in h2v:
            h2v[h] = vecs[i]
    log.info(f"  reuse map: {len(h2v)} unique content vectors from original '{orig_source}' index ({idx.ntotal} rows)")
    return h2v


def build(ds, orig_source, target_per_partition=1000):
    clean = f"{ds}_clean"
    master = f"data/processed/master_nodes_{clean}.json"
    out_dir = os.path.join("data", "ukb_storage", clean)
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Building UKB for '{clean}' (reusing '{orig_source}' embeddings) -> {out_dir}")

    nodes = load_nodes(master)
    docs = [n for n in nodes if n.metadata.get("type") != "question"]
    h2v = _content_hash_to_vec(orig_source)
    dim = len(next(iter(h2v.values())))
    embs = np.zeros((len(docs), dim), dtype=np.float32)
    miss = 0
    for i, d in enumerate(docs):
        v = h2v.get(hashlib.md5(d.content.encode("utf-8")).hexdigest())
        if v is None:
            miss += 1
        else:
            embs[i] = v
    log.info(f"  embeddings: {len(docs)} docs, {miss} content-hash misses (should be 0)")
    if miss:
        raise RuntimeError(f"{miss} clean docs had no matching original embedding — content mismatch")
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype("float32"))
    faiss.write_index(index, os.path.join(out_dir, "nodes.index"))
    log.info(f"  [1/5] nodes.index written ({index.ntotal} vecs, dim={dim}) — no encoding")

    indexers.build_bm25_index(nodes, out_dir)               # [2]
    G_nx = indexers.build_pyg_graph(nodes, embs, out_dir)   # [3] adds kNN -> graph.pt + synthetic
    parts = indexers.build_partition_map(nodes, G_nx, out_dir, target_per_partition)   # [4] METIS
    indexers.build_faiss_centroid_index(nodes, parts, embs, out_dir)  # [5] degree-weighted centroids

    npart = int(max(parts)) + 1 if parts else 0
    log.info(f"  DONE '{clean}': {index.ntotal} docs, {npart} partitions -> {out_dir}")
    return npart


def main(argv=None):
    p = argparse.ArgumentParser(description="Fast clean-UKB build via embedding reuse.")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--target_per_partition", type=int, default=1000,
                   help="~docs/partition for METIS (use ~300 for small deduped corpora like musique).")
    a = p.parse_args(argv)
    for ds in a.datasets:
        build(ds, orig_source=ds, target_per_partition=a.target_per_partition)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
