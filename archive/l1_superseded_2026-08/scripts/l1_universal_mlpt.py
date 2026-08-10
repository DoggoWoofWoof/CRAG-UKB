"""
Universal single-model L1 router (one mlpT head, all datasets).
================================================================
The whole point: ONE model, not per-dataset. Because a frozen encoder (BGE-large) puts every
dataset's docs+queries in the SAME space, we pool (query -> gold) triples across all 5 datasets and
train ONE mlpT MixtureHead (K query-conditioned offset directions, soft-OR). That single head is then
applied to each dataset for retrieval -> partition votes -> FullCov@20. No per-dataset weights.

mlpT is the featured novelty: its K directions absorb what separate rel_hard / rel_2hop heads did, so
the retriever zoo collapses to one head. Fusion stays simple: dense + mlpT via equal-RRF (+ best-of).

Runs on frozen embeddings (subdir, default bge_large) so training is an MLP over cached vectors — fast.
Query embeddings are encoded once per dataset (cached in the subdir if present). Eval loads one corpus
at a time (memory-safe). Writes results/L1_universal_mlpt.json + a per-dataset breakdown.
"""
import os
import gc
import json
import logging
import argparse

import numpy as np
import torch

from src.core.engine import CoreEngine
from src.experiments.encoder_swap import load_docs_and_encoder
from src.experiments.overlap_retrain import _splits, _hard_membership, _onehop_membership
from src.experiments.l1_ablate import MixtureHead, MAXK, TAU, INIT_SEED
from src.experiments.l1_rerank100 import _feats, _rr, _fullcov, K0, TOPN

log = logging.getLogger("experiments.l1_universal_mlpt")

DATASETS = ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]


def _load_dataset(dataset, subdir, limit, tr_cap, te_cap, device):
    """Return everything needed for one dataset in the shared frozen space.
    Encodes ONLY train (<=tr_cap) + test (<=te_cap|limit); val is unused so we never encode it."""
    eng = CoreEngine(source=dataset)
    X, eq, tag = load_docs_and_encoder(eng, dataset, subdir)
    X = X.astype("float32")
    n = X.shape[0]
    npart = max(int(p) for p in eng.partition_map.values()) + 1
    id2idx = eng.node_id_to_idx
    idx2id = {v: k for k, v in id2idx.items()}
    hard = np.array([int(eng.partition_map.get(idx2id[i], -1)) for i in range(n)])
    mem = _onehop_membership(eng)
    mem_idx = [sorted(mem.get(idx2id[i], {int(hard[i])})) for i in range(n)]
    sp = _splits(eng, _hard_membership(eng))
    caps = {"train": tr_cap or limit, "test": te_cap or limit}

    def prep(qs, split):
        cache = os.path.join("data", "ukb_storage", dataset, subdir, f"queries_{split}.npy") if subdir else None
        if cache and os.path.exists(cache):
            q = np.load(cache)[:len(qs)].astype("float32")
        else:
            q = eq([nd.content for nd, _, _ in qs])
        seed = np.argmax(q @ X.T, axis=1).astype(np.int64)
        gold = [[id2idx[g] for g in gg if g in id2idx] for _, _, gg in qs]
        return q, seed, gold

    out = {"X": X, "npart": npart, "mem_idx": mem_idx, "tag": tag}
    for s in ("train", "test"):
        out[s] = prep(sp[s][:caps[s]], s)                  # cap BEFORE encoding; skip val (unused)
    return out


def _pool_train(data, cap):
    """Compact pooled training set across datasets: one Xt of the seed+gold vectors actually used,
    triples remapped into it, pooled query matrix. In-batch negatives => no global index needed."""
    q_rows, seed_ids, gold_ids, vecs = [], [], [], []
    vidx = {}                                              # (dataset, local_doc_idx) -> compact row

    def add_vec(dkey, X, li):
        key = (dkey, int(li))
        if key not in vidx:
            vidx[key] = len(vecs); vecs.append(X[int(li)])
        return vidx[key]

    qoff = 0
    for dkey, d in data.items():
        q, seed, gold = d["train"]
        q = q[:cap]; seed = seed[:cap]; gold = gold[:cap]
        X = d["X"]
        for i in range(len(q)):
            gl = [g for g in gold[i]]
            if not gl:
                continue
            q_rows.append(q[i])
            seed_ids.append(add_vec(dkey, X, seed[i]))
            gold_ids.append([add_vec(dkey, X, g) for g in gl])
        qoff += len(q)
    Q = np.stack(q_rows).astype("float32")
    Xt = torch.tensor(np.stack(vecs).astype("float32"))
    return Q, np.array(seed_ids), gold_ids, Xt


def _mlpt_order(head, q, seed_vec, X, device, k=MAXK, bs=256):
    """Retrieval order per query = argsort of soft-OR (max over K directions) similarity to docs."""
    Xd = torch.tensor(X, device=device)
    orders = np.empty((len(q), min(k, X.shape[0])), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(q), bs):
            qn = torch.tensor(q[s:s + bs], device=device)
            sd = torch.tensor(seed_vec[s:s + bs], device=device)
            pos = head(qn, sd)                              # (b, K, d)
            sim = torch.einsum("bkd,nd->bkn", pos, Xd).max(1).values   # (b, n) soft-OR
            orders[s:s + bs] = torch.topk(sim, min(k, X.shape[0]), dim=1).indices.cpu().numpy()
    del Xd
    return orders


def _dense_order(q, X, device, k=MAXK, bs=256):
    Xd = torch.tensor(X, device=device)
    orders = np.empty((len(q), min(k, X.shape[0])), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(q), bs):
            qn = torch.tensor(q[s:s + bs], device=device)
            sim = qn @ Xd.T
            orders[s:s + bs] = torch.topk(sim, min(k, X.shape[0]), dim=1).indices.cpu().numpy()
    del Xd
    return orders


def _votes(order, mem_idx, npart):
    S, M = _feats(order, mem_idx, npart)
    return _rr(S) + _rr(M)                                 # sum + max vote, RRF-ranked


def run(datasets=None, subdir="bge_large", limit=8000, tr_cap=4000, te_cap=0, epochs=25, K=8,
        device=None):
    datasets = datasets or DATASETS
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from src.experiments.l1_ablate import _train_offset

    log.info(f"loading {len(datasets)} datasets in '{subdir}' space...")
    data = {}
    for d in datasets:
        data[d] = _load_dataset(d, subdir, limit, tr_cap, te_cap, device)
        log.info(f"  {d}: X{data[d]['X'].shape} npart={data[d]['npart']} "
                 f"train={len(data[d]['train'][0])} test={len(data[d]['test'][0])} enc={data[d]['tag']}")

    Q, seed_ids, gold_ids, Xt = _pool_train(data, tr_cap)
    log.info(f"pooled train: {len(Q)} queries, {Xt.shape[0]} unique seed+gold vecs -> training ONE mlpT (K={K})")
    head = _train_offset("mix", Q, seed_ids, gold_ids, Xt.to(device), None, device, epochs, K=K)

    per = {}
    for d in datasets:
        dd = data[d]
        X = dd["X"]; mem_idx = dd["mem_idx"]; npart = dd["npart"]
        q, seed, gold = dd["test"]
        if te_cap:
            q, seed, gold = q[:te_cap], seed[:te_cap], gold[:te_cap]
        seed_vec = X[seed]
        gpl = [[mem_idx[g] for g in gg] for gg in gold]
        mo = _mlpt_order(head, q, seed_vec, X, device)
        do = _dense_order(q, X, device)
        v_mlpt = _votes(mo, mem_idx, npart)
        v_dense = _votes(do, mem_idx, npart)
        res = {
            "n_test": len(gpl), "npart": npart, "encoder": dd["tag"],
            "dense@20": _fullcov(v_dense, gpl, npart)[20],
            "mlpT@20": _fullcov(v_mlpt, gpl, npart)[20],
            "dense+mlpT@20": _fullcov(v_dense + v_mlpt, gpl, npart)[20],
        }
        per[d] = res
        log.info(f"[{d}] dense={res['dense@20']} mlpT={res['mlpT@20']} dense+mlpT={res['dense+mlpT@20']} (n={res['n_test']})")
        del X, mo, do, v_mlpt, v_dense; gc.collect()

    prim = "dense+mlpT@20"
    mean = round(float(np.mean([per[d][prim] for d in datasets])), 2)
    over95 = sum(1 for d in datasets if per[d][prim] >= 95)
    summary = {"model": "universal-mlpT", "encoder_subdir": subdir, "K": K, "epochs": epochs,
               "tr_cap": tr_cap, "primary": prim, "mean": mean, "over95": f"{over95}/{len(datasets)}",
               "per_dataset": per}
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open(os.path.join("results", "L1_universal_mlpt.json"), "w"), indent=2)
    log.info(f"UNIVERSAL mlpT: mean {prim} = {mean} | {over95}/{len(datasets)} over 95 -> results/L1_universal_mlpt.json")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Train ONE universal mlpT head across all datasets (frozen space).")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="bge_large")
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--tr-cap", type=int, default=4000)
    p.add_argument("--te-cap", type=int, default=0, help="cap test queries per dataset (0=all)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--K", type=int, default=8)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, limit=a.limit, tr_cap=a.tr_cap, te_cap=a.te_cap,
        epochs=a.epochs, K=a.K)


if __name__ == "__main__":
    main()
