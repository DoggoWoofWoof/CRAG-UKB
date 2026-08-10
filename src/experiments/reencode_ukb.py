"""
Re-encode a UKB source with a STRONGER frozen encoder (non-destructive).
========================================================================
The whole Level-1 ceiling is set by the geometry of a small 384-d MiniLM-L6.
This caches doc embeddings from a stronger sentence encoder (default
BAAI/bge-base-en-v1.5, 768-d) + rebuilt degree-weighted hard centroids into a
SUBFOLDER of the same UKB (data/ukb_storage/{src}/{subdir}/), leaving the
original MiniLM indices untouched. encoder_upgrade.py then runs the overlap
stack against these for a matched A/B.

Doc side is encoded plain; queries later get the model's retrieval instruction
(stored in meta.json) — the documented bge/e5 usage. Saves nodes.npy (+ faiss
nodes.index), centroids.index, centroid_pids.json, meta.json.
"""
import os
import json
import logging
import argparse
from collections import defaultdict

import numpy as np

from src.core.engine import CoreEngine

# Big instruct encoders (gte-Qwen2) on long docs fragment CUDA memory; expandable segments avoids
# the "reserved-but-unallocated" OOM. Set before any CUDA context is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

log = logging.getLogger("experiments.reencode_ukb")

# model -> retrieval query instruction (docs are encoded plain). Empty = no instruction.
QUERY_INSTRUCTION = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
    "intfloat/e5-large-v2": "query: ",
    "sentence-transformers/all-mpnet-base-v2": "",
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct":
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "intfloat/e5-mistral-7b-instruct":
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
}
DOC_PREFIX = {"intfloat/e5-base-v2": "passage: ", "intfloat/e5-large-v2": "passage: "}
TRUST_REMOTE = ("Alibaba-NLP/gte-Qwen2-1.5B-instruct",)   # models needing trust_remote_code=True


def _centroids_hard(engine, node_vecs, npart):
    import faiss
    acc = defaultdict(list)
    for i, node in enumerate(engine.nodes):
        p = engine.partition_map.get(node.node_id)
        if p is None:
            continue
        w = float(len(node.neighbors)) + 1.0
        acc[int(p)].append((i, w))
    dim = node_vecs.shape[1]
    C = np.zeros((npart, dim), dtype=np.float32)
    pids = []
    for p in sorted(acc):
        idxs = [i for i, _ in acc[p]]
        ws = np.array([w for _, w in acc[p]], dtype=np.float64)
        C[p] = np.average(node_vecs[idxs], axis=0, weights=ws)
        pids.append(p)
    Cused = np.stack([C[p] for p in pids]).astype("float32")
    faiss.normalize_L2(Cused)
    return Cused, pids


def run(dataset, model_name="BAAI/bge-base-en-v1.5", subdir="bge_base", batch=64, max_seq=None):
    import faiss
    from sentence_transformers import SentenceTransformer

    out_dir = os.path.join("data", "ukb_storage", dataset, subdir)
    os.makedirs(out_dir, exist_ok=True)
    log.info(f"Re-encoding {dataset} with {model_name} -> {out_dir}")

    engine = CoreEngine(source=dataset)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    texts = [n.content for n in engine.nodes]
    doc_prefix = DOC_PREFIX.get(model_name, "")
    if doc_prefix:
        texts = [doc_prefix + t for t in texts]

    model = SentenceTransformer(model_name, trust_remote_code=(model_name in TRUST_REMOTE))
    if model_name in TRUST_REMOTE:                         # large instruct encoders (gte-Qwen2): FP16 engages
        import torch                                       # flash_attn + tensor cores -> ~5x over FP32 (which
        model = model.half()                              # otherwise runs ~10 docs/s -> intractable at 507k)
        log.info("  model -> FP16 (flash_attn + tensor cores)")
    if max_seq:                                            # cap sequence length: bounds attention memory on
        model.max_seq_length = int(max_seq)               # rare long docs (>99.9% of docs are far shorter)
        log.info(f"  max_seq_length capped to {max_seq}")
    dim = model.get_sentence_embedding_dimension()
    log.info(f"Encoding {len(texts)} docs (dim={dim})...")
    embs = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=True).astype("float32")

    np.save(os.path.join(out_dir, "nodes.npy"), embs)
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    faiss.write_index(idx, os.path.join(out_dir, "nodes.index"))

    # cache QUERY embeddings for the locked splits so reruns never re-encode (esp. big instruct
    # models can't be re-encoded on CPU each rerank). Aligns with rerank100's _splits order.
    from src.experiments.overlap_retrain import _splits, _hard_membership
    q_instr = QUERY_INSTRUCTION.get(model_name, "")
    splits = _splits(engine, _hard_membership(engine))
    for sp in ("train", "val", "test"):
        qtexts = [nd.content for nd, _, _ in splits[sp]]
        if not qtexts:
            continue
        qembs = model.encode([q_instr + t for t in qtexts], batch_size=batch,
                             normalize_embeddings=True, show_progress_bar=False).astype("float32")
        np.save(os.path.join(out_dir, f"queries_{sp}.npy"), qembs)
        log.info(f"  cached {len(qembs)} {sp} query embeddings")

    C, pids = _centroids_hard(engine, embs, npart)
    cidx = faiss.IndexFlatIP(C.shape[1]); cidx.add(C)
    faiss.write_index(cidx, os.path.join(out_dir, "centroids.index"))
    with open(os.path.join(out_dir, "centroid_pids.json"), "w") as f:
        json.dump(pids, f)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"model": model_name, "dim": dim, "n_docs": len(texts),
                   "query_instruction": QUERY_INSTRUCTION.get(model_name, ""),
                   "doc_prefix": doc_prefix}, f, indent=2)
    log.info(f"Saved {out_dir}: nodes.npy {embs.shape}, {len(pids)} centroids, dim={dim}")
    return out_dir


def build_encoder_graph(dataset, subdir, target_per_partition=100):
    """Build graph.pt + partition_map.json + centroids INSIDE {dataset}/{subdir}/ from that
    encoder's node embeddings — so kNN + METIS run in the encoder's own space (a per-encoder
    substrate). Title-link edges (node.neighbors) come from the master and are encoder-independent.
    Reuses the existing subdir nodes.npy (no re-encode). Aligns with the root build_all pipeline
    (build_pyg_graph -> build_partition_map -> degree-weighted centroids), only the vectors differ."""
    from src.core.engine import CoreEngine
    from src.core import indexers
    eng = CoreEngine(source=dataset)                          # root load: nodes + title-link neighbors
    sub = os.path.join("data", "ukb_storage", dataset, subdir)
    embs = np.load(os.path.join(sub, "nodes.npy")).astype("float32")
    if embs.shape[0] != len(eng.nodes):
        raise RuntimeError(f"{dataset}/{subdir}: nodes.npy has {embs.shape[0]} rows but "
                           f"{len(eng.nodes)} doc nodes — encoder/substrate misaligned")
    for nd in eng.nodes:
        nd.metadata.pop("synthetic_neighbors", None)          # rebuild synthetic kNN in THIS encoder's space
    G = indexers.build_pyg_graph(eng.nodes, embs, sub)                              # sub/graph.pt
    parts = indexers.build_partition_map(eng.nodes, G, sub, target_per_partition)   # sub/partition_map.json
    indexers.build_faiss_centroid_index(eng.nodes, parts, embs, sub)               # sub/centroids.index (+ pids)
    npart = int(max(parts)) + 1 if parts else 0
    log.info(f"[encoder-graph] {dataset}/{subdir}: {len(eng.nodes)} docs -> {npart} partitions "
             f"(~{target_per_partition}/part), kNN in {subdir} space")
    return npart


def main(argv=None):
    p = argparse.ArgumentParser(description="Re-encode a UKB source with a stronger encoder (non-destructive).")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--subdir", default="bge_base")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-seq", type=int, default=None, help="cap encoder sequence length (bounds memory on long docs)")
    a = p.parse_args(argv)
    for ds in a.datasets:
        run(ds, model_name=a.model, subdir=a.subdir, batch=a.batch, max_seq=a.max_seq)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
