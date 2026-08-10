"""
Error analysis — WHY does the champion miss the golds it misses? (diagnostic -> arch)
=====================================================================================
Instead of guessing architectures, measure the misses. For the champion (dense+rel_hard+
rel_2hop RRF), take every gold NOT in top-100 and bucket it by the lever that would recover
it. Each bucket maps to a concrete fix — so the breakdown tells us where the gain is:

  fusion_recoverable : some single retriever ranked it < 100, but RRF drowned it
                       -> BETTER COMBINATION (weights / learned rerank of the pool)
  rerank_recoverable : best single-retriever rank in [100,300) -> bigger pool + rerank
  prf_embed_sibling  : embedding-close (cos >= T) to an ALREADY-FOUND gold
                       -> PSEUDO-RELEVANCE FEEDBACK / self-expansion (embedding)
  prf_graph_sibling  : within 1 graph hop of a found gold (diagnosis-only graph)
                       -> self-expansion (structural)  [graph used to DIAGNOSE, not retrieve]
  deep_multihop      : >=2 graph hops from the dense seed -> genuinely multi-hop (L3 territory)
  semantic_far       : low query cosine AND best rank > 300 -> offset didn't bridge (encoder?)
  rare_lowdeg        : degree <= 2 -> structurally hard / long-tail
Also: which retriever is 'best' for the misses (dense vs rel_hard vs rel_2hop), and the
semantic-gap / hop distributions. Writes {ds}/results/L1/error_analysis.json.
"""
import os
import json
import logging
import argparse
from collections import deque, defaultdict

import numpy as np
import torch
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse
from src.experiments.l1_dynamic import _train_hop2
from src.experiments.l3_reachability import _adj

log = logging.getLogger("experiments.l1_error_analysis")
PRF_COS = 0.6            # embedding-sibling threshold
SEM_FAR = 0.3            # query-cosine "far" threshold
MAXR = 2000             # cap when locating a gold's rank in a retriever


def _bfs_hop(seeds, targets, adj, max_hop=3):
    """min hop from any seed to each target (dict target->hop), capped."""
    want = set(int(t) for t in targets)
    got = {}
    seen = set(int(s) for s in seeds); frontier = set(seen)
    for h in range(1, max_hop + 1):
        nxt = set()
        for d in frontier:
            for j in adj[d]:
                jj = int(j)
                if jj not in seen:
                    seen.add(jj); nxt.add(jj)
                    if jj in want:
                        got[jj] = h
        frontier = nxt
        if not frontier or len(got) == len(want):
            break
    return got


def run(dataset, limit_test=500, limit_train=6000, N_seed=20, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); enc = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n, d = X.shape; index = faiss.IndexFlatIP(d); index.add(X); Xt = torch.tensor(X, device=device)
    id2idx = engine.node_id_to_idx
    adj, _, _ = _adj(engine, id2idx)                    # diagnosis-only graph
    deg = np.array([len(adj[i]) for i in range(n)], np.int64)
    splits = _splits(engine, _hard_membership(engine))
    tr = splits["train"][:limit_train]; te = splits["test"][:limit_test]

    def prep(qs):
        q = enc.encode([nd.content for nd, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        _, seed = index.search(q, 1)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed[:, 0], gold
    q_tr, seed_tr, gold_tr = prep(tr)
    q_te, seed_te, gold_te = prep(te)
    qte = torch.tensor(q_te, device=device)

    log.info(f"[{dataset}] train champion offsets (base,hard,2hop)...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, 25)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, 25)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, 25)

    def pos(h, seed=seed_te):
        with torch.no_grad():
            return h(qte, Xt[[int(s) for s in seed]]).cpu().numpy()
    P_hard = pos(g_hard)
    P1 = pos(g1); dense_ord = _order(q_te, index); hard_ord = _order(P_hard, index)
    hop1 = _order(P1, index); s1 = hop1[:, 0]
    with torch.no_grad():
        P2 = g2(qte, Xt[[int(s) for s in s1]]).cpu().numpy()
    hop2 = _order(P2, index)
    rel_2hop = _rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0])
    champ = _rrf_fuse([_ranks(dense_ord), _ranks(hard_ord), _ranks(rel_2hop)], [1.0, 1.0, 1.0])

    def rank_of(scores, node):
        s = float(scores[node]); r = int((scores > s).sum())
        return r if r < MAXR else MAXR

    buckets = defaultdict(int); best_ret = defaultdict(int)
    n_missed = 0; n_fullcov_fail = 0; n_q = 0; sem_gaps = []; hopseed_hist = defaultdict(int)
    for qi in range(len(te)):
        golds = gold_te[qi]
        if not golds:
            continue
        n_q += 1
        top100 = set(champ[qi][:100])
        missed = [g for g in golds if g not in top100]
        found = [g for g in golds if g in top100]
        if missed:
            n_fullcov_fail += 1
        if not missed:
            continue
        qs = q_te[qi]; d_row = qs @ X.T; h_row = P_hard[qi] @ X.T; t_row = P2[qi] @ X.T
        hop_seed = _bfs_hop([seed_te[qi]], missed, adj)
        hop_found = _bfs_hop(found, missed, adj) if found else {}
        Xf = X[found] if found else None
        for m in missed:
            n_missed += 1
            rd, rh, rt = rank_of(d_row, m), rank_of(h_row, m), rank_of(t_row, m)
            br = min(rd, rh, rt); best_ret[["dense", "rel_hard", "rel_2hop"][int(np.argmin([rd, rh, rt]))]] += 1
            qcos = float(d_row[m]); sem_gaps.append(qcos)
            maxf = float(np.max(Xf @ X[m])) if Xf is not None and len(Xf) else 0.0
            hs = hop_seed.get(m, 9); hf = hop_found.get(m, 9)
            hopseed_hist[min(hs, 4)] += 1
            if br < 100:
                buckets["fusion_recoverable"] += 1
            elif br < 300:
                buckets["rerank_recoverable"] += 1
            if maxf >= PRF_COS:
                buckets["prf_embed_sibling"] += 1
            if hf <= 1:
                buckets["prf_graph_sibling"] += 1
            if hs >= 2:
                buckets["deep_multihop"] += 1
            if qcos < SEM_FAR and br >= 300:
                buckets["semantic_far"] += 1
            if deg[m] <= 2:
                buckets["rare_lowdeg"] += 1

    pct = {k: round(v / max(n_missed, 1) * 100, 1) for k, v in buckets.items()}
    out = {"dataset": dataset, "n_test": n_q, "avg_golds": round(float(np.mean([len(g) for g in gold_te if g])), 2),
           "fullcov@100_fail_rate": round(n_fullcov_fail / max(n_q, 1) * 100, 1),
           "n_missed_golds": n_missed,
           "miss_buckets_pct": dict(sorted(pct.items(), key=lambda kv: -kv[1])),
           "best_retriever_for_misses_pct": {k: round(v / max(n_missed, 1) * 100, 1) for k, v in best_ret.items()},
           "missed_query_cos": {"mean": round(float(np.mean(sem_gaps)), 3) if sem_gaps else None,
                                "p10": round(float(np.percentile(sem_gaps, 10)), 3) if sem_gaps else None,
                                "p90": round(float(np.percentile(sem_gaps, 90)), 3) if sem_gaps else None},
           "hop_from_seed_to_miss_pct": {str(k): round(v / max(n_missed, 1) * 100, 1) for k, v in sorted(hopseed_hist.items())}}
    os.makedirs(os.path.join("data", "ukb_storage", dataset, "results", "L1"), exist_ok=True)
    with open(os.path.join("data", "ukb_storage", dataset, "results", "L1", "error_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] fullcov@100 fail {out['fullcov@100_fail_rate']}% | {n_missed} missed golds | "
             f"buckets {out['miss_buckets_pct']} | best-ret {out['best_retriever_for_misses_pct']} | "
             f"hop-from-seed {out['hop_from_seed_to_miss_pct']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Champion miss error-analysis (which lever recovers each missed gold).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "musique_clean", "2wiki_clean"])
    p.add_argument("--limit_test", type=int, default=500)
    p.add_argument("--limit_train", type=int, default=6000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for ds in a.datasets:
        log.info(f"===== L1 ERROR ANALYSIS: {ds.upper()} =====")
        try:
            run(ds, limit_test=a.limit_test, limit_train=a.limit_train)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
