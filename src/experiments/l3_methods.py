"""
L3 all-methods benchmark (#13) — the one L3 thing independent of the L1/L2 pipeline.
====================================================================================
Final L3 eval needs the champion->L2 seeds (pending), but the L3 METHOD comparison
on the fixed substrate from a common dense-seed front-end does NOT — so it runs in
parallel now. Compares every traversal method at matched doc budgets, with latency:
  dense    : top-B by query cosine (no graph)
  1hop/2hop: seeds + k-hop graph neighbours, cosine-ranked, dense-filled
  ppr      : personalized PageRank from seeds (= HippoRAG-class), best alpha
  appnp    : graph-diffused embeddings, retrieve q @ H.T (static propagation)
  ours     : bounded PPR-guided best-first (l3_traverse.traverse)
Seeds = dense top-N (common front-end). Graph = title UNION synthetic kNN.
Metrics: gt_recall@{20,50,100,200}, FullCov@{20,50,100}, median ms/query.
Writes data/ukb_storage/{ds}/results/L3/methods.json.
"""
import os
import time
import json
import logging
import argparse

import numpy as np
import scipy.sparse as sp
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l3_reachability import _adj
from src.experiments.l3_traverse import traverse
from src.pipeline.ner_edges import _entities_regex
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.l3_methods")
BUDGETS = [20, 50, 100, 200]


def _P(adj, n):
    rows, cols = [], []
    for i in range(n):
        for j in adj[i]:
            rows.append(i); cols.append(int(j))
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).maximum(
        sp.csr_matrix((np.ones(len(rows), np.float32), (cols, rows)), shape=(n, n)))
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return (sp.diags(1.0 / d) @ A).tocsr(), A.tocsr()


def _ppr(seeds, P, alpha, iters=20):
    p = seeds.copy()
    for _ in range(iters):
        p = (1 - alpha) * seeds + alpha * (p @ P)
    return p


def _metrics(order, gold, budgets=BUDGETS):
    gr = {b: [] for b in budgets}; fc = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        for b in budgets:
            hit = len(gs & set(order[qi][:b].tolist()))
            gr[b].append(hit / len(gs)); fc[b].append(1.0 if hit == len(gs) else 0.0)
    return ({f"gt_recall@{b}": round(np.mean(gr[b]) * 100, 2) for b in budgets},
            {f"fullcov@{b}": round(np.mean(fc[b]) * 100, 2) for b in budgets})


def _champion_seed_order(engine, X, id2idx, test_q, maxb, device):
    """L1-champion (dense + rel_hard + rel_2hop, RRF-fused) ranking for the test queries — used as
    the L3 seed front-end instead of pure dense (champion-seeded L3). Trains the champion heads on
    the train split (fast on GPU). This is the L1->L3 hand-off: better seeds -> better traversal."""
    import torch
    from src.experiments.l1_ablate import _train_offset, _ranks, _rrf_fuse, _order as _ford
    from src.experiments.l1_dynamic import _train_hop2
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)
    Xt = torch.tensor(X, device=device)
    enc = DenseEncoder()
    tr = _splits(engine, _hard_membership(engine))["train"][:20000]
    qtr = enc.encode([nd.content for nd, _, _ in tr]).astype("float32"); faiss.normalize_L2(qtr)
    _, s = index.search(qtr, 1); seed_tr = s[:, 0]
    gold_tr = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in tr]
    g1 = _train_offset("base", qtr, seed_tr, gold_tr, Xt, index, device, 25)
    gh = _train_offset("hard", qtr, seed_tr, gold_tr, Xt, index, device, 25)
    g2 = _train_hop2(g1, qtr, seed_tr, gold_tr, X, Xt, index, device, 25)
    _, ste = index.search(test_q, 1); seed_te = ste[:, 0]

    def pos(g):
        with torch.no_grad():
            return g(torch.tensor(test_q, device=device), Xt[[int(x) for x in seed_te]]).cpu().numpy()
    dord = _ford(test_q, index, maxb); hard = _ford(pos(gh), index, maxb)
    hop1 = _ford(pos(g1), index, maxb); s1 = hop1[:, 0]
    with torch.no_grad():
        hop2 = _ford(g2(torch.tensor(test_q, device=device), Xt[[int(x) for x in s1]]).cpu().numpy(), index, maxb)
    two = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    champ = _rrf_fuse([_ranks(dord), _ranks(hard), _ranks(two)], [1.0, 1.0, 1.0])
    arr = np.zeros((len(champ), maxb), np.int64)
    for i, r in enumerate(champ):
        r = r[:maxb]; arr[i, :len(r)] = r
    return arr


def run(dataset, N_seed=20, limit=500, seed_source="dense", device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n = X.shape[0]; id2idx = engine.node_id_to_idx
    adj, deg_s, deg_y = _adj(engine, id2idx)
    P, A = _P(adj, n)
    indptr, indices = A.indptr, A.indices
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    nq = len(test); maxb = max(BUDGETS)
    qsim = q @ X.T
    dense_order = np.argsort(-qsim, axis=1)[:, :maxb]
    if seed_source == "champion":
        import torch
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_order = _champion_seed_order(engine, X, id2idx, q, maxb, dev)   # L1 champion front-end
    else:
        seed_order = dense_order
    seeds_idx = seed_order[:, :N_seed]                                        # traversal seeds

    out = {"dataset": dataset, "n_docs": n, "N_seed": N_seed, "n_test": nq, "seed_source": seed_source,
           "deg_struct": deg_s, "deg_syn": deg_y, "budgets": BUDGETS, "methods": {}, "latency_ms": {}}

    def record(name, order, ms=None):
        gr, fc = _metrics(order, gold)
        out["methods"][name] = {**gr, **fc}
        if ms is not None:
            out["latency_ms"][name] = round(ms, 3)

    # dense
    t = time.perf_counter(); _ = np.argsort(-(q @ X.T), axis=1)[:, :maxb]; ms = (time.perf_counter()-t)/nq*1000
    record("dense", dense_order, ms)

    # k-hop (seeds + neighbours, cosine-ranked, dense-fill)
    def khop(hops):
        orders = np.zeros((nq, maxb), np.int64)
        for qi in range(nq):
            front = set(seeds_idx[qi].tolist()); reach = set(front)
            for _ in range(hops):
                nxt = set()
                for d in front:
                    nxt.update(int(x) for x in indices[indptr[d]:indptr[d+1]])
                front = nxt - reach; reach |= front
            cand = sorted(reach, key=lambda d: -qsim[qi, d])
            fill = [int(d) for d in seed_order[qi] if d not in reach]
            orders[qi] = (cand + fill)[:maxb]
        return orders
    t = time.perf_counter(); h1 = khop(1); record("1hop", h1, (time.perf_counter()-t)/nq*1000)
    t = time.perf_counter(); h2 = khop(2); record("2hop", h2, (time.perf_counter()-t)/nq*1000)

    # ppr (HippoRAG-class), best alpha over a small set
    seeds_mat = np.zeros((nq, n), np.float32)
    for qi in range(nq):
        seeds_mat[qi, seeds_idx[qi]] = 1.0 / N_seed
    best_ppr = None; best_a = None
    t = time.perf_counter()
    for a in (0.3, 0.5, 0.7, 0.9):
        o = np.argsort(-_ppr(seeds_mat, P, a), axis=1)[:, :maxb]
        gr, _ = _metrics(o, gold)
        if best_ppr is None or gr["gt_recall@100"] > best_ppr:
            best_ppr, best_a, best_o = gr["gt_recall@100"], a, o
    ppr_ms = (time.perf_counter()-t)/nq*1000/4
    record("ppr", best_o, ppr_ms); out["ppr_best_alpha"] = best_a

    # appnp (static diffused embeddings)
    t = time.perf_counter()
    H = X.copy()
    for _ in range(10):
        H = 0.5 * X + 0.5 * (P @ H)                          # row-normalized propagation (APPNP)
    Hn = H.copy(); faiss.normalize_L2(Hn)
    ap_order = np.argsort(-(q @ Hn.T), axis=1)[:, :maxb]
    record("appnp", ap_order, (time.perf_counter()-t)/nq*1000)

    # ours: bounded PPR-guided best-first
    content_lower = [nd.content.lower() for nd in engine.nodes]
    ours = np.zeros((nq, maxb), np.int64); ttot = 0.0
    for qi in range(nq):
        qents = {e for e in _entities_regex(test[qi][0].content) if len(e) >= 3}
        t0 = time.perf_counter()
        res = traverse(q[qi], qents, seeds_idx[qi], adj, X, content_lower, budget=maxb, alpha=0.5)
        ttot += time.perf_counter() - t0
        o = res["order"]; o = o + [int(d) for d in seed_order[qi] if d not in set(o)]
        ours[qi] = np.array(o[:maxb])
    record("ours_bestfirst", ours, ttot / nq * 1000)

    with open(rpath(dataset, "L3", "methods_champion" if seed_source == "champion" else "methods"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] " + " | ".join(
        f"{m}: R@100={v['gt_recall@100']}" for m, v in out["methods"].items()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 all-methods benchmark (dense-seed front-end).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "metaqa", "musique_clean", "hotpotqa_clean", "squad_clean"])
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed_source", default="dense", choices=["dense", "champion"])
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== L3 ALL-METHODS: {ds.upper()} (seeds={a.seed_source}) =====")
        run(ds, N_seed=a.N_seed, limit=a.limit, seed_source=a.seed_source)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
