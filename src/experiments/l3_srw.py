"""
L3 edge-reweighted PPR ("supervised random walk", training-free core).
=======================================================================
After 8 traversal methods x {dense,champion} seeds, PPR-class is the ceiling and the
verdict is "seeds >> traversal algorithm". The ONE lever left is the GRAPH ITSELF:
vanilla PPR weights every edge equally, but our substrate mixes STRUCTURAL title-edges
with SYNTHETIC kNN edges — if kNN edges are noise, down-weighting them lets the walk
follow real relations. We reweight each edge by (a) TYPE (synthetic weight beta vs
structural 1) and (b) SEMANTIC strength cos(u,v)^gamma, then run PPR on the reweighted
graph. (beta,gamma) are selected on a train subset by FullCov@100 and applied to test —
this is the training-free core of a supervised random walk (Backstrom-Leskovec), the
legit test of "can a better-weighted graph beat uniform PPR". Dense seeds by default
(isolate the graph lever), champion via --seed_source champion. Writes L3/srw.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import scipy.sparse as sp
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l3_methods import _ppr, _metrics, BUDGETS, _champion_seed_order

log = logging.getLogger("experiments.l3_srw")
MAXB = max(BUDGETS)
BETAS = [0.0, 0.25, 0.5, 1.0]        # synthetic-edge weight (1.0 = vanilla)
GAMMAS = [0.0, 1.0, 2.0]             # semantic edge-strength exponent (0 = unweighted)


def _typed_adj(engine, id2idx):
    """Structural (title/neighbor) vs synthetic (kNN) edge lists, separately."""
    docset = set(id2idx); n = len(id2idx)
    struct = [[] for _ in range(n)]; syn = [[] for _ in range(n)]
    for node in engine.nodes:
        i = id2idx[node.node_id]; seen = set()
        for nb in node.neighbors:
            if nb in docset and nb != node.node_id:
                j = id2idx[nb]
                if j not in seen:
                    struct[i].append(j); seen.add(j)
        for nb in node.metadata.get("synthetic_neighbors", ()):
            if nb in docset and nb != node.node_id:
                j = id2idx[nb]
                if j not in seen:
                    syn[i].append(j); seen.add(j)
    return struct, syn


def _weighted_P(struct, syn, X, n, beta, gamma):
    rows, cols, vals = [], [], []
    for i in range(n):
        xi = X[i]
        for j in struct[i]:
            w = (max(float(xi @ X[j]), 1e-3) ** gamma) if gamma > 0 else 1.0
            rows.append(i); cols.append(j); vals.append(w)
        if beta > 0:
            for j in syn[i]:
                w = beta * ((max(float(xi @ X[j]), 1e-3) ** gamma) if gamma > 0 else 1.0)
                rows.append(i); cols.append(j); vals.append(w)
    A = sp.csr_matrix((np.asarray(vals, np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1.0
    return (sp.diags(1.0 / d) @ A).tocsr()


def _seeds_mat(seed_order, N_seed, n):
    sm = np.zeros((seed_order.shape[0], n), np.float32)
    for qi in range(seed_order.shape[0]):
        sm[qi, seed_order[qi, :N_seed]] = 1.0 / N_seed
    return sm


def run(dataset, N_seed=20, limit=500, seed_source="dense", alpha=0.9, device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n = X.shape[0]; id2idx = engine.node_id_to_idx
    struct, syn = _typed_adj(engine, id2idx)
    n_struct = sum(len(s) for s in struct); n_syn = sum(len(s) for s in syn)
    splits = _splits(engine, _hard_membership(engine))
    enc = DenseEncoder()

    def prep(qs, cap):
        qs = qs[:cap]
        q = enc.encode([qn.content for qn, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, gold
    q_tr, gold_tr = prep(splits["train"], 200)              # small train subset for (beta,gamma) selection
    q_te, gold_te = prep(splits["test"], limit)

    def seed_order_for(q):
        if seed_source == "champion":
            import torch
            dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return _champion_seed_order(engine, X, id2idx, q, MAXB, dev)
        return np.argsort(-(q @ X.T), axis=1)[:, :MAXB]
    so_tr = seed_order_for(q_tr); so_te = seed_order_for(q_te)
    sm_tr = _seeds_mat(so_tr, N_seed, n); sm_te = _seeds_mat(so_te, N_seed, n)

    # select (beta,gamma) on train by FullCov@100
    results = {}
    best = None
    for beta in BETAS:
        for gamma in GAMMAS:
            P = _weighted_P(struct, syn, X, n, beta, gamma)
            o = np.argsort(-_ppr(sm_tr, P, alpha), axis=1)[:, :MAXB]
            _, fc = _metrics(o, gold_tr)
            key = f"b{beta}_g{gamma}"
            results[key] = {"train_fullcov@100": fc["fullcov@100"]}
            if best is None or fc["fullcov@100"] > best[0]:
                best = (fc["fullcov@100"], beta, gamma)
            log.info(f"[{dataset}] train {key}: FCOV@100={fc['fullcov@100']}")
    _, bb, bg = best

    # eval vanilla (b1,g0) vs best on test
    def test_eval(beta, gamma):
        P = _weighted_P(struct, syn, X, n, beta, gamma)
        o = np.argsort(-_ppr(sm_te, P, alpha), axis=1)[:, :MAXB]
        gr, fc = _metrics(o, gold_te)
        return {**gr, **fc}
    vanilla = test_eval(1.0, 0.0)
    learned = test_eval(bb, bg)
    out = {"dataset": dataset, "seed_source": seed_source, "n_docs": n, "N_seed": N_seed,
           "n_test": len([g for g in gold_te if g]), "n_struct_edges": n_struct, "n_syn_edges": n_syn,
           "alpha": alpha, "best_beta": bb, "best_gamma": bg, "train_grid": results,
           "vanilla_ppr": vanilla, "reweighted_ppr": learned,
           "reweighted_vs_vanilla_fullcov@100": round(learned["fullcov@100"] - vanilla["fullcov@100"], 2),
           "reweighted_vs_vanilla_recall@100": round(learned["gt_recall@100"] - vanilla["gt_recall@100"], 2)}
    tag = "srw_champion" if seed_source == "champion" else "srw"
    path = os.path.join("data", "ukb_storage", dataset, "results", "L3", f"{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] ({seed_source}) struct={n_struct} syn={n_syn} | best(beta,gamma)=({bb},{bg}) | "
             f"vanilla PPR FCOV@100={vanilla['fullcov@100']} -> reweighted={learned['fullcov@100']} "
             f"({out['reweighted_vs_vanilla_fullcov@100']:+}) | recall {out['reweighted_vs_vanilla_recall@100']:+}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 edge-reweighted PPR (supervised-random-walk core) vs vanilla PPR.")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean"])
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed_source", default="dense", choices=["dense", "champion"])
    p.add_argument("--alpha", type=float, default=0.9)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L3 SRW (edge-reweighted PPR): {ds.upper()} (seeds={a.seed_source}) =====")
        try:
            results[ds] = run(ds, N_seed=a.N_seed, limit=a.limit, seed_source=a.seed_source, alpha=a.alpha)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
    if results:
        summary = {ds: {"best_beta_gamma": [r["best_beta"], r["best_gamma"]],
                        "vanilla_fullcov@100": r["vanilla_ppr"]["fullcov@100"],
                        "reweighted_fullcov@100": r["reweighted_ppr"]["fullcov@100"],
                        "delta": r["reweighted_vs_vanilla_fullcov@100"]} for ds, r in results.items()}
        path = os.path.join("data", "ukb_storage", "_index", f"l3_srw_{a.seed_source}_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"SRW ({a.seed_source}) summary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
