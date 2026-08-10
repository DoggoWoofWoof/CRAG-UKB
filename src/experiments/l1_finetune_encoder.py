"""Fine-tune bge-large in-domain on a dataset's (query -> gold-doc) pairs, then encode docs + queries
with it into a reencode-compatible subdir (nodes.npy, queries_{split}.npy, centroids.index, meta.json).
A task-tuned encoder ranks the answer docs higher -> stronger partition votes -> the votable-but-
weakly-voted gold partition climbs into the top-20 (the ranking wall the laggards are stuck on).
Contrastive MultipleNegativesRankingLoss (in-batch negatives). Query embeddings are cached so the
rerank never re-encodes. Writes data/ukb_storage/{ds}/{subdir}/. Run on GPU."""
import os
import json
import logging
import argparse

import numpy as np

from src.core.engine import CoreEngine

log = logging.getLogger("experiments.l1_finetune_encoder")


FT_SEED = 42                                               # frozen: fine-tune must be run-to-run reproducible


def run(dataset, base="BAAI/bge-large-en-v1.5", subdir="ft_bge", epochs=1, batch=16, enc_batch=64,
        limit=0, max_seq=256, seed=FT_SEED):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce fragmentation OOMs
    import random as _random
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    from src.experiments.overlap_retrain import _splits, _hard_membership
    from src.experiments.reencode_ukb import _centroids_hard

    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # reproducible shuffle + init
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    eng = CoreEngine(source=dataset)
    npart = max(int(p) for p in eng.partition_map.values()) + 1
    id2content = {n.node_id: n.content for n in eng.all_nodes}
    splits = _splits(eng, _hard_membership(eng))
    tr = splits["train"][:limit] if limit else splits["train"]
    pairs = [InputExample(texts=[nd.content, id2content[g]])
             for nd, _, golds in tr for g in golds if g in id2content]
    log.info(f"[{dataset}] fine-tuning {base} on {len(pairs)} (query,gold) pairs, {epochs} ep, bs={batch}")

    model = SentenceTransformer(base)
    model.max_seq_length = max_seq                         # queries are short; truncate docs -> fits A10G in backprop
    g = torch.Generator().manual_seed(seed)                # deterministic batch order
    loader = DataLoader(pairs, batch_size=batch, shuffle=True, generator=g)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit([(loader, loss)], epochs=epochs, warmup_steps=min(100, len(loader) // 10),
              show_progress_bar=True, use_amp=torch.cuda.is_available())

    out = os.path.join("data", "ukb_storage", dataset, subdir); os.makedirs(out, exist_ok=True)
    texts = [n.content for n in eng.nodes]
    log.info(f"[{dataset}] encoding {len(texts)} docs with fine-tuned model...")
    embs = model.encode(texts, batch_size=enc_batch, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    np.save(os.path.join(out, "nodes.npy"), embs)
    idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs); faiss.write_index(idx, os.path.join(out, "nodes.index"))
    C, pids = _centroids_hard(eng, embs, npart)
    cidx = faiss.IndexFlatIP(C.shape[1]); cidx.add(C); faiss.write_index(cidx, os.path.join(out, "centroids.index"))
    json.dump(pids, open(os.path.join(out, "centroid_pids.json"), "w"))
    json.dump({"model": base, "dim": int(embs.shape[1]), "query_instruction": "", "finetuned": True},
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    for sp in ("train", "val", "test"):
        qt = [nd.content for nd, _, _ in splits[sp]]
        if not qt:
            continue
        qe = model.encode(qt, batch_size=enc_batch, normalize_embeddings=True, show_progress_bar=False).astype("float32")
        np.save(os.path.join(out, f"queries_{sp}.npy"), qe)
        log.info(f"  cached {len(qe)} {sp} query embeddings")
    # fail loudly if the artifacts did not actually land (a silent no-write would look like success)
    need = ["nodes.npy", "centroids.index", "meta.json", "queries_test.npy"]
    missing = [f for f in need if not os.path.exists(os.path.join(out, f))]
    if missing:
        raise RuntimeError(f"[{dataset}] fine-tune wrote nothing for {missing} in {out}")
    log.info(f"[{dataset}] saved fine-tuned subdir {out} ({', '.join(os.listdir(out))})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune bge-large in-domain -> reencode-compatible subdir.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean"])
    p.add_argument("--base", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--subdir", default="ft_bge")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    for ds in a.datasets:                                  # let run() throw natively -> Modal propagates the real traceback
        log.info(f"===== FINE-TUNE ENCODER: {ds.upper()} =====")
        run(ds, base=a.base, subdir=a.subdir, epochs=a.epochs, batch=a.batch, limit=a.limit, max_seq=a.max_seq)


if __name__ == "__main__":
    main()
