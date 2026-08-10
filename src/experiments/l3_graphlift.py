"""
L3 graph-lift test — the graph's native job, tested honestly.
==============================================================
The candidate-generation defense of partitions failed (dense-top-N is a better pool). This tests the
DIFFERENT, structural claim: does walking the doc graph from dense seeds recover golds that dense
retrieval cannot reach — i.e., the 2nd-hop evidence that is NOT query-similar? That is the one thing a
graph does by construction and a similarity pool cannot.

Consistent with the rest of the pipeline: gte_qwen substrate, same engine as _load. The doc graph =
title/structural edges (node.neighbors) UNION synthetic kNN edges (the L3 traversal substrate, via
l1l3_recall._graph). For each query, at a matched budget B:

  dense_recall@B         : golds in the dense top-B pool.
  graph_recovered        : of the golds dense-top-B MISSES, the fraction within `hops` graph-steps of the
                           dense top-`n_seed` seeds -> golds the graph can reach that dense cannot.
  union_recall@B         : golds in (dense top-B  UNION  seed graph-neighborhood).

If graph_recovered is large on multi-hop/KB (metaqa/2wiki/hotpot/musique) and ~0 on single-hop (squad),
the graph earns its place at L3 for the right reason. Writes results/L2/graphlift_{subdir}.json.
"""
import os
import json
import logging
import argparse

import numpy as np

log = logging.getLogger(__name__)


def _neighbors(A, nodes, cap_per_node=64):
    """Union of graph neighbours of `nodes` from CSR adjacency A (cap per node to bound hubs)."""
    out = set()
    for s in nodes:
        nb = A.indices[A.indptr[s]:A.indptr[s + 1]]
        out.update(int(x) for x in nb[:cap_per_node])
    return out


def run(datasets=None, subdir="gte_qwen", n_seed=20, budget=100, hops=2, te_cap=2000):
    import torch
    import faiss
    from src.experiments.l1_universal_head import (CoreEngine, load_docs_and_encoder,
                                                    _splits, _hard_membership)
    from src.experiments.l1l3_recall import _graph

    datasets = datasets or ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/L2", exist_ok=True)
    out = {}
    for d in datasets:
        eng = CoreEngine(source=d, index_subdir=subdir)
        X, eq, _ = load_docs_and_encoder(eng, d, subdir)
        X = np.ascontiguousarray(X, dtype="float32"); n = X.shape[0]
        id2idx = eng.node_id_to_idx
        _, A, deg = _graph(eng, n, id2idx, sources=("struct", "syn"))   # doc graph: title UNION synthetic-kNN edges
        A = A.tocsr()

        test = _splits(eng, _hard_membership(eng))["test"][:te_cap]
        cache = os.path.join("data", "ukb_storage", d, subdir, "queries_test.npy")
        if os.path.exists(cache) and len(np.load(cache)) >= len(test):
            q = np.load(cache)[:len(test)].astype("float32")
        else:
            q = eq([nd.content for nd, _, _ in test]).astype("float32")
        gold_idx = [[id2idx[g] for g in gg if g in id2idx] for _, _, gg in test]

        Xn = X.copy(); faiss.normalize_L2(Xn); qn = q.copy(); faiss.normalize_L2(qn)
        Xt = torch.tensor(Xn, device=device)

        dense_r, union_r, dhit, uhit = [], [], [], []
        miss_tot, miss_reach = 0, 0                                     # of dense-missed golds, how many graph-reachable
        B = min(budget, n)
        with torch.no_grad():
            for qi in range(len(test)):
                g = set(gold_idx[qi])
                if not g:
                    continue
                sim = (torch.tensor(qn[qi:qi + 1], device=device) @ Xt.T)[0]
                order = torch.topk(sim, min(max(B, n_seed), n)).indices.cpu().numpy()
                dense_B = set(int(x) for x in order[:B])
                seeds = [int(x) for x in order[:n_seed]]
                reach = set(seeds)                                       # seeds + up-to-`hops` graph neighbours
                frontier = set(seeds)
                for _ in range(hops):
                    frontier = _neighbors(A, frontier) - reach
                    reach |= frontier
                    if not frontier:
                        break
                union = dense_B | reach
                dr = len(g & dense_B) / len(g); ur = len(g & union) / len(g)
                dense_r.append(dr); union_r.append(ur)
                dhit.append(1.0 if (g & dense_B) else 0.0); uhit.append(1.0 if (g & union) else 0.0)
                missed = g - dense_B                                     # golds dense-top-B did not retrieve
                miss_tot += len(missed); miss_reach += len(missed & reach)

        out[d] = {
            "corpus_N": int(n), "graph_deg": round(float(A.nnz) / max(1, n), 2), "n_seed": n_seed, "budget": B, "hops": hops,
            "dense_recall": round(100 * float(np.mean(dense_r)), 2),
            "union_recall": round(100 * float(np.mean(union_r)), 2),
            "traversal_lift": round(100 * float(np.mean(union_r) - np.mean(dense_r)), 2),
            "dense_hit": round(100 * float(np.mean(dhit)), 2),
            "union_hit": round(100 * float(np.mean(uhit)), 2),
            "dense_missed_golds": int(miss_tot),
            "graph_reachable_of_missed": round(100 * miss_reach / max(1, miss_tot), 2),
        }
        r = out[d]
        log.info("[graphlift/%s] N=%d deg=%.1f B=%d seeds=%d | dense R=%.1f -> union R=%.1f (lift %+.1f) | "
                 "of %d dense-missed golds, %.1f%% graph-reachable",
                 d, r["corpus_N"], r["graph_deg"], B, n_seed, r["dense_recall"], r["union_recall"],
                 r["traversal_lift"], r["dense_missed_golds"], r["graph_reachable_of_missed"])
        del Xt
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    path = f"results/L2/graphlift_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 graph-lift: does traversal from dense seeds recover golds dense misses?")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--n-seed", type=int, default=20)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--te-cap", type=int, default=2000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, n_seed=a.n_seed, budget=a.budget, hops=a.hops, te_cap=a.te_cap)


if __name__ == "__main__":
    main()
