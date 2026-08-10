"""
Stronger-encoder A/B on the overlap stack (improve L1 at fixed overlap/pool).
=============================================================================
Runs the exact overlap+kNN stack (KL+HNM router, per-doc coverage) with either
the original MiniLM-L6 embeddings or the cached stronger-encoder embeddings
(reencode_ukb.py output in data/ukb_storage/{ds}/{subdir}/). Membership, centroid
rebuild, router training and eval are identical across encoders — the ONLY
difference is the frozen embedding — so the FCov delta isolates the encoder.

Node embeddings: MiniLM via engine.node_index; stronger via cached nodes.npy.
Queries: MiniLM via DenseEncoder; stronger via the model + its retrieval
instruction from meta.json. Writes results/encoder_upgrade/{dataset}.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import torch

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _centroids, _splits, _train, _eval, membership_ref, TAU, HNK
from src.evaluation.benchmark_partition_selection import COVERAGE_K_VALUES

log = logging.getLogger("experiments.encoder_upgrade")


def _membership(engine, node_vecs, npart, cfg):
    import faiss

    def atom(name):
        metis = {nid: int(p) for nid, p in engine.partition_map.items()}
        if name == "hard":
            return {nid: {p} for nid, p in metis.items()}
        if name == "overlap1":
            mem = {nid: {p} for nid, p in metis.items()}
            for node in engine.nodes:
                for nb in node.neighbors:
                    if nb in metis:
                        mem[node.node_id].add(metis[nb])
            return mem
        if name.startswith("knn"):
            m = int(name[3:])
            C, _ = _centroids(engine, node_vecs, {nid: {p} for nid, p in metis.items()}, npart)
            nv = node_vecs.copy(); faiss.normalize_L2(nv)
            cv = C.copy(); faiss.normalize_L2(cv)
            mem = {nid: {p} for nid, p in metis.items()}
            topm = np.argsort(-(nv @ cv.T), axis=1)[:, :m]
            for i, node in enumerate(engine.nodes):
                if node.node_id in mem:
                    mem[node.node_id].update(int(j) for j in topm[i])
            return mem
        raise ValueError(name)

    atoms = [atom(a) for a in cfg.split("+")]
    if len(atoms) == 1:
        return atoms[0]
    keys = set().union(*[set(a) for a in atoms])
    return {k: set().union(*[a.get(k, set()) for a in atoms]) for k in keys}


class _STEncoder:
    def __init__(self, model_name, instruction=""):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name)
        self.instr = instruction or ""

    def encode(self, texts):
        return self.m.encode([self.instr + t for t in texts], batch_size=64,
                             normalize_embeddings=True, show_progress_bar=False).astype("float32")


def _load_encoder(dataset, which, subdir):
    if which == "minilm":
        engine = CoreEngine(source=dataset)
        return engine, _reconstruct(engine.node_index), DenseEncoder(), {"model": "multi-qa-MiniLM-L6-cos-v1", "dim": 384}
    # stronger encoder from cache
    d = os.path.join("data", "ukb_storage", dataset, subdir)
    meta = json.load(open(os.path.join(d, "meta.json")))
    engine = CoreEngine(source=dataset)
    node_vecs = np.load(os.path.join(d, "nodes.npy")).astype("float32")
    q_enc = _STEncoder(meta["model"], meta.get("query_instruction", ""))
    return engine, node_vecs, q_enc, meta


def run_dataset(dataset, configs=("hard", "overlap1", "overlap1+knn1"),
                encoders=("minilm", "bge"), subdir="bge_base", epochs=100, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    out = {"dataset": dataset, "tau": tau, "hn_k": hn_k, "encoders": {}}

    for which in encoders:
        engine, node_vecs, q_enc, meta = _load_encoder(dataset, which, subdir)
        npart = max(int(p) for p in engine.partition_map.values()) + 1
        log.info(f"===== ENCODER-UPGRADE: {dataset.upper()} enc={which} ({meta['model']}, dim={meta['dim']}) =====")
        enc_res = {"model": meta["model"], "dim": meta["dim"], "configs": {}}
        for cfg in configs:
            membership = _membership(engine, node_vecs, npart, cfg)
            membership_ref[cfg] = membership
            C, _ = _centroids(engine, node_vecs, membership, npart)
            splits = _splits(engine, membership)
            split_embs = {s: q_enc.encode([q.content for q, _, _ in splits[s]]) for s in splits if splits[s]}
            model, best_state, final_state, Cg = _train(
                engine, C, splits, split_embs, device, tau, hn_k, epochs,
                os.path.join("logs", dataset, f"encup_{which}_{cfg.replace('+','_')}"), cfg, "kl")
            m = _eval(model, final_state, Cg, splits["test"], split_embs["test"], membership, device)
            enc_res["configs"][cfg] = m
            log.info(f"  [{which} {cfg}] FCov@10={m['full_coverage@10']}% FCov@20={m['full_coverage@20']}% "
                     f"gtR@20={m['gt_recall@20']}%")
        out["encoders"][which] = enc_res

    out_dir = os.path.join("results", "encoder_upgrade")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/encoder_upgrade/{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Stronger-encoder A/B on the overlap stack.")
    p.add_argument("--datasets", nargs="+", default=["2wiki"])
    p.add_argument("--configs", nargs="+", default=["hard", "overlap1", "overlap1+knn1"])
    p.add_argument("--encoders", nargs="+", default=["minilm", "bge"])
    p.add_argument("--subdir", default="bge_base")
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        run_dataset(ds, configs=tuple(a.configs), encoders=tuple(a.encoders),
                    subdir=a.subdir, epochs=a.epochs, device=dev)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
