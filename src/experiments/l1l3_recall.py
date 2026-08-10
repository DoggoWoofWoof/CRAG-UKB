"""
L1+L3 recall: the 3 recovery methods behind the REAL partition router.
=======================================================================
The why/ppr study compared dense / 1-hop / PPR seeded from FULL-CORPUS dense
top-N -- i.e. a *perfect fine-grained* L1. The deployed system is coarse
PARTITION routing (L1) -> traversal (L3). This measures whether each L3 method
still recovers the gold docs once the real (coarse) router picks the candidate
pool, i.e. end-to-end gold-doc recall, across datasets.

Two front-ends (anchors for L3), same recovery methods on each:
  ideal : anchors = global dense top-N over all docs        (upper bound / research_ppr)
  L1    : train the partition router (KL loss, hard membership), route the top-K
          partitions -> pool P; anchors = dense top-N WITHIN P (deployed pipeline)
Recovery -> final top-B docs:
  dense : rank the candidate pool by query-doc cosine (no graph)
  1hop  : anchors + their graph neighbours (partition-jumping), cosine-ranked, dense-filled
  ppr   : personalized PageRank from anchors over the doc graph (alpha swept, best kept)
Traversal graph = node.neighbours (structural title edges) UNION synthetic kNN
edges (the traversal-time partition-jumping substrate).

Lead metric is gt_recall (fraction of gold docs retrieved); FullCov (all-golds)
reported alongside. Also emits L1_pool recall = the ceiling L3 can reach from
that pool. Writes results/research/l1l3_{dataset}.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import (
    _reconstruct, _splits, _hard_membership, _centroids, _train, membership_ref, TAU, HNK,
)

log = logging.getLogger("experiments.l1l3_recall")

BUDGETS = (20, 50, 100, 200)
PPR_ALPHAS = (0.3, 0.5, 0.7, 0.9)


def _graph(engine, n, id2idx, sources=("struct", "syn"), extra_edges=None):
    """Row-normalized transition over the selected edge sources -- the L3 traversal
    substrate (partition-jumping). Sources: 'struct' (node.neighbors title edges),
    'syn' (synthetic kNN edges), 'extra' (an injected doc_id->[ids] edge set, e.g.
    recovered title-mention edges being A/B tested before a rebuild)."""
    rows, cols = [], []
    cnt = {"struct": 0, "syn": 0, "extra": 0}
    for i, node in enumerate(engine.nodes):
        if "struct" in sources:
            for nb in node.neighbors:
                j = id2idx.get(nb)
                if j is not None and j != i:
                    rows.append(i); cols.append(j); cnt["struct"] += 1
        if "syn" in sources:
            for nb in node.metadata.get("synthetic_neighbors", ()):
                j = id2idx.get(nb)
                if j is not None and j != i:
                    rows.append(i); cols.append(j); cnt["syn"] += 1
    if "extra" in sources and extra_edges:
        for src, nbrs in extra_edges.items():
            i = id2idx.get(src)
            if i is None:
                continue
            for nb in nbrs:
                j = id2idx.get(nb)
                if j is not None and j != i:
                    rows.append(i); cols.append(j); cnt["extra"] += 1
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)                                   # undirected
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1
    P = (sp.diags(1.0 / deg) @ A).tocsr()
    return P, A, {k: round(v / n, 2) for k, v in cnt.items()}


def _ppr(seeds, P, alpha, iters=20):
    p = seeds.copy()
    for _ in range(iters):
        p = (1 - alpha) * seeds + alpha * (p @ P)
    return p


def _metrics(order, gold_idx, budgets):
    """order: (nq, >=max(budgets)) doc-idx ranking. Returns gt_recall@B and fullcov@B."""
    gr = {b: [] for b in budgets}; fc = {b: [] for b in budgets}
    for qi, gset in enumerate(gold_idx):
        if not gset:
            continue
        gs = set(gset)
        row = order[qi]
        for b in budgets:
            hit = len(gs & set(row[:b].tolist()))
            gr[b].append(hit / len(gs))
            fc[b].append(1.0 if hit == len(gs) else 0.0)
    return ({f"gt_recall@{b}": round(float(np.mean(gr[b])) * 100, 2) for b in budgets},
            {f"fullcov@{b}": round(float(np.mean(fc[b])) * 100, 2) for b in budgets})


def _onehop_order(anchors_rows, qsim, P, cand_universe, maxb):
    """anchors + their 1-hop neighbours (cosine-ranked), then dense-fill from the
    candidate universe. cand_universe[qi] = array of doc idxs the front-end allows
    for the fill (pool for L1, all docs for ideal). Neighbours may lie OUTSIDE it
    (partition-jumping) and are always kept."""
    nq = qsim.shape[0]
    out = np.zeros((nq, maxb), dtype=np.int64)
    indptr, indices = P.indptr, P.indices
    for qi in range(nq):
        seeds = anchors_rows[qi].tolist()
        nbrs = set()
        for s in seeds:
            nbrs.update(indices[indptr[s]:indptr[s + 1]].tolist())
        cand = list(dict.fromkeys(seeds + sorted(nbrs, key=lambda d: -qsim[qi, d])))
        have = set(cand)
        fill = [d for d in cand_universe[qi] if d not in have]
        row = (cand + fill)[:maxb]
        if len(row) < maxb:                              # pad from global dense if short
            extra = [d for d in np.argsort(-qsim[qi]) if d not in set(row)]
            row = (row + extra)[:maxb]
        out[qi] = row[:maxb]
    return out


def run(dataset, pool_frac=0.10, N_anchor=20, epochs=100, limit=1500, device=None,
        edge_sources=("struct", "syn"), extra_edges_path=None, out_tag=""):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index).astype("float32")
    nv = node_vecs.copy(); faiss.normalize_L2(nv)
    n = len(engine.nodes)
    id2idx = engine.node_id_to_idx
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    extra_edges = json.load(open(extra_edges_path, encoding="utf-8")) if extra_edges_path else None
    P, A, deg = _graph(engine, n, id2idx, sources=tuple(edge_sources), extra_edges=extra_edges)
    maxb = max(BUDGETS)

    # partition -> member doc idxs (hard membership)
    pid_docs = [[] for _ in range(npart)]
    for nid, pid in engine.partition_map.items():
        j = id2idx.get(nid)
        if j is not None:
            pid_docs[int(pid)].append(j)
    pid_docs = [np.array(d, dtype=np.int64) for d in pid_docs]
    K = max(1, round(pool_frac * npart))

    # ---- train the L1 router (hard membership, KL) ----
    membership = _hard_membership(engine)
    membership_ref["hard"] = membership
    C, _ = _centroids(engine, node_vecs, membership, npart)
    splits = _splits(engine, membership)
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                  for s in splits if splits[s]}
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    logs_dir = os.path.join("results", "research", "_l1l3_logs", dataset)
    model, best_state, _, Cg = _train(engine, C, splits, split_embs, device, tau, hn_k,
                                       epochs, logs_dir, "hard", loss_name="kl")
    model.load_state_dict(best_state); model.eval()

    test = splits["test"]
    q_emb = encoder.encode([q.content for q, _, _ in test]).astype("float32")
    qn = q_emb.copy(); faiss.normalize_L2(qn)
    gold_idx = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    nq = len(test)

    qsim = qn @ nv.T                                     # (nq, n) dense query-doc cosine
    global_order = np.argsort(-qsim, axis=1)             # full dense ranking

    # router partition ranking
    with torch.no_grad():
        proj = F.normalize(model(torch.tensor(q_emb, device=device)), dim=-1)
        route = (proj @ Cg.T).cpu().numpy()              # (nq, npart)
    part_order = np.argsort(-route, axis=1)

    # ---- build candidate universes + anchors for both front-ends ----
    pool_docs = []                                        # L1 routed pool per query
    pool_sizes = []
    anchors_ideal = global_order[:, :N_anchor]
    anchors_L1 = np.zeros((nq, N_anchor), dtype=np.int64)
    for qi in range(nq):
        docs = np.concatenate([pid_docs[p] for p in part_order[qi, :K]]) if K else np.array([], np.int64)
        pool_docs.append(docs)
        pool_sizes.append(len(docs))
        if len(docs):
            loc = docs[np.argsort(-qsim[qi, docs])]       # pool docs by dense
            anchors_L1[qi] = loc[:N_anchor] if len(loc) >= N_anchor else \
                np.pad(loc, (0, N_anchor - len(loc)), constant_values=loc[-1])
    # L1 pool dense order (dense-rank the pool, pad from global if pool < maxb)
    L1_dense_order = np.zeros((nq, maxb), dtype=np.int64)
    for qi in range(nq):
        docs = pool_docs[qi]
        ranked = docs[np.argsort(-qsim[qi, docs])] if len(docs) else np.array([], np.int64)
        if len(ranked) < maxb:
            have = set(ranked.tolist())
            extra = np.array([d for d in global_order[qi] if d not in have], np.int64)
            ranked = np.concatenate([ranked, extra])
        L1_dense_order[qi] = ranked[:maxb]

    def ppr_order(anchor_rows):
        seeds = np.zeros((nq, n), np.float32)
        for qi in range(nq):
            a = anchor_rows[qi]
            seeds[qi, a] = 1.0 / max(len(set(a.tolist())), 1)
        best = None; best_key = None
        for alpha in PPR_ALPHAS:
            sc = _ppr(seeds, P, alpha)
            order = np.argsort(-sc, axis=1)[:, :maxb]
            gr, _fc = _metrics(order, gold_idx, BUDGETS)
            key = gr[f"gt_recall@{100 if 100 in BUDGETS else maxb}"]
            if best_key is None or key > best_key:
                best_key, best = key, (alpha, order)
        return best                                       # (alpha, order)

    out = {"dataset": dataset, "n_docs": n, "npart": npart, "K_partitions": K,
           "pool_frac": pool_frac, "avg_pool_docs": round(float(np.mean(pool_sizes)), 1),
           "N_anchor": N_anchor, "n_test": nq, "avg_golds": round(float(np.mean([len(g) for g in gold_idx if g])), 2),
           "edge_sources": list(edge_sources), "graph_deg": deg,
           "graph_deg_total": round(float(A.nnz) / n, 2),
           "budgets": list(BUDGETS), "ppr_alphas": list(PPR_ALPHAS), "results": {}}

    def record(name, order, extra=None):
        gr, fc = _metrics(order, gold_idx, BUDGETS)
        rec = {"gt_recall": gr, "fullcov": fc}
        if extra:
            rec.update(extra)
        out["results"][name] = rec
        log.info(f"  [{dataset} {name:16}] {gr}")

    # L1 pool ceiling (golds anywhere in routed pool)
    pool_reach = []
    for qi in range(nq):
        if not gold_idx[qi]:
            continue
        ps = set(pool_docs[qi].tolist())
        pool_reach.append(len(set(gold_idx[qi]) & ps) / len(gold_idx[qi]))
    out["results"]["L1_pool_ceiling"] = {"gt_recall_pool": round(float(np.mean(pool_reach)) * 100, 2)}
    log.info(f"  [{dataset} L1_pool_ceiling ] gt_recall_pool={out['results']['L1_pool_ceiling']['gt_recall_pool']}"
             f"  (avg_pool={out['avg_pool_docs']} docs)")

    # ---- ideal front-end (global dense anchors) ----
    record("ideal:dense", global_order[:, :maxb])
    record("ideal:1hop", _onehop_order(anchors_ideal, qsim, P,
                                        [global_order[qi] for qi in range(nq)], maxb))
    a_i, o_i = ppr_order(anchors_ideal)
    record("ideal:ppr", o_i, {"best_alpha": a_i})

    # ---- real L1 front-end (routed-pool anchors) ----
    record("L1:dense", L1_dense_order)
    record("L1:1hop", _onehop_order(anchors_L1, qsim, P, pool_docs, maxb))
    a_l, o_l = ppr_order(anchors_L1)
    record("L1:ppr", o_l, {"best_alpha": a_l})

    os.makedirs("results/research", exist_ok=True)
    fname = f"results/research/l1l3_{dataset}{out_tag}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved {fname}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L1+L3 gold-doc recall: dense/1hop/PPR behind ideal vs real router.")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "metaqa", "squad"])
    p.add_argument("--pool_frac", type=float, default=0.10)
    p.add_argument("--N_anchor", type=int, default=20)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--edge_sources", default="struct,syn",
                   help="comma list from {struct,syn,extra} defining the L3 traversal graph")
    p.add_argument("--extra_edges", default=None, help="json doc_id->[neighbor ids] for the 'extra' source")
    p.add_argument("--out_tag", default="", help="suffix for the output filename (avoid clobbering)")
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== L1+L3 RECALL: {ds.upper()}  (edges={a.edge_sources}) =====")
        run(ds, pool_frac=a.pool_frac, N_anchor=a.N_anchor, epochs=a.epochs, limit=a.limit,
            edge_sources=tuple(a.edge_sources.split(",")), extra_edges_path=a.extra_edges, out_tag=a.out_tag)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
