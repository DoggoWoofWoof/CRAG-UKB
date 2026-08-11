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

EDGE ABLATION (--edge-ablation): the "why" — decompose the traversal lift by which edge TYPE does the
recovery, and by hop depth. Compares structural-only (real KG-triple / title edges) vs synthetic-kNN-only
(dense-similarity edges) vs both, at hops 1 and 2. If STRUCT carries the lift and SYN is ~0, the graph's
value is genuine relational structure, not dense-similarity re-expressed. Writes graphlift_edgeablation_{subdir}.json.
"""
import os
import json
import logging
import argparse

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DATASETS = ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]


def _neighbors(A, nodes, cap_per_node=64):
    """Union of graph neighbours of `nodes` from CSR adjacency A (cap per node to bound hubs)."""
    out = set()
    for s in nodes:
        nb = A.indices[A.indptr[s]:A.indptr[s + 1]]
        out.update(int(x) for x in nb[:cap_per_node])
    return out


def _reach(A, seeds, hops):
    """Seeds + up-to-`hops` graph neighbours."""
    reach = set(seeds); frontier = set(seeds)
    for _ in range(hops):
        frontier = _neighbors(A, frontier) - reach
        reach |= frontier
        if not frontier:
            break
    return reach


def _prep(d, subdir, te_cap, device):
    """Source-independent per-dataset setup: engine, normalized X (on device), queries, gold indices."""
    import torch
    import faiss
    from src.experiments.l1_universal_head import (CoreEngine, load_docs_and_encoder,
                                                    _splits, _hard_membership)
    eng = CoreEngine(source=d, index_subdir=subdir)
    X, eq, _ = load_docs_and_encoder(eng, d, subdir)
    X = np.ascontiguousarray(X, dtype="float32"); n = X.shape[0]
    id2idx = eng.node_id_to_idx
    test = _splits(eng, _hard_membership(eng))["test"][:te_cap]
    cache = os.path.join("data", "ukb_storage", d, subdir, "queries_test.npy")
    if os.path.exists(cache) and len(np.load(cache)) >= len(test):
        q = np.load(cache)[:len(test)].astype("float32")
    else:
        q = eq([nd.content for nd, _, _ in test]).astype("float32")
    gold_idx = [[id2idx[g] for g in gg if g in id2idx] for _, _, gg in test]
    Xn = X.copy(); faiss.normalize_L2(Xn); qn = q.copy(); faiss.normalize_L2(qn)
    Xt = torch.tensor(Xn, device=device)
    return eng, n, id2idx, qn, gold_idx, Xt


def _dense_order(qn, Xt, n, B, n_seed, device):
    """Per-query dense top-max(B,n_seed) index order (source-independent; computed once)."""
    import torch
    orders = []
    with torch.no_grad():
        for qi in range(len(qn)):
            sim = (torch.tensor(qn[qi:qi + 1], device=device) @ Xt.T)[0]
            orders.append(torch.topk(sim, min(max(B, n_seed), n)).indices.cpu().numpy())
    return orders


def _traverse_metrics(A, orders, gold_idx, n, n_seed, budget, hops):
    """Given a graph A and precomputed dense orders, compute dense/union recall + reachability at `hops`."""
    A = A.tocsr(); B = min(budget, n)
    dense_r, union_r = [], []
    miss_tot, miss_reach = 0, 0
    for qi in range(len(gold_idx)):
        g = set(gold_idx[qi])
        if not g:
            continue
        order = orders[qi]
        dense_B = set(int(x) for x in order[:B])
        reach = _reach(A, [int(x) for x in order[:n_seed]], hops)
        union = dense_B | reach
        dense_r.append(len(g & dense_B) / len(g)); union_r.append(len(g & union) / len(g))
        missed = g - dense_B
        miss_tot += len(missed); miss_reach += len(missed & reach)
    return {
        "graph_deg": round(float(A.nnz) / max(1, n), 2),
        "dense_recall": round(100 * float(np.mean(dense_r)), 2),
        "union_recall": round(100 * float(np.mean(union_r)), 2),
        "traversal_lift": round(100 * float(np.mean(union_r) - np.mean(dense_r)), 2),
        "graph_reachable_of_missed": round(100 * miss_reach / max(1, miss_tot), 2),
        "dense_missed_golds": int(miss_tot),
    }


def run(datasets=None, subdir="gte_qwen", n_seed=20, budget=100, hops=2, te_cap=2000, sources=("struct", "syn")):
    import torch
    from src.experiments.l1l3_recall import _graph
    datasets = datasets or DEFAULT_DATASETS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/L2", exist_ok=True)
    out = {}
    for d in datasets:
        eng, n, id2idx, qn, gold_idx, Xt = _prep(d, subdir, te_cap, device)
        _, A, _ = _graph(eng, n, id2idx, sources=tuple(sources))
        orders = _dense_order(qn, Xt, n, min(budget, n), n_seed, device)
        m = _traverse_metrics(A, orders, gold_idx, n, n_seed, budget, hops)
        out[d] = {"corpus_N": int(n), "n_seed": n_seed, "budget": min(budget, n), "hops": hops, **m}
        r = out[d]
        log.info("[graphlift/%s] N=%d deg=%.1f | dense R=%.1f -> union R=%.1f (lift %+.1f) | "
                 "%.1f%% of %d dense-missed golds graph-reachable",
                 d, r["corpus_N"], r["graph_deg"], r["dense_recall"], r["union_recall"],
                 r["traversal_lift"], r["graph_reachable_of_missed"], r["dense_missed_golds"])
        del Xt
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    path = f"results/L2/graphlift_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def run_edge_ablation(datasets=None, subdir="gte_qwen", n_seed=20, budget=100, te_cap=2000):
    """WHY does traversal work: decompose the lift by edge TYPE (struct vs synthetic-kNN vs both) and hop depth."""
    import torch
    from src.experiments.l1l3_recall import _graph
    datasets = datasets or DEFAULT_DATASETS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/L2", exist_ok=True)
    configs = [("struct", ("struct",)), ("syn", ("syn",)), ("both", ("struct", "syn"))]
    out = {}
    for d in datasets:
        eng, n, id2idx, qn, gold_idx, Xt = _prep(d, subdir, te_cap, device)
        orders = _dense_order(qn, Xt, n, min(budget, n), n_seed, device)   # dense is source-independent -> once
        rec = {"corpus_N": int(n)}
        dense_recall = None
        for name, srcs in configs:
            _, A, _ = _graph(eng, n, id2idx, sources=srcs)
            for hops in (1, 2):
                m = _traverse_metrics(A, orders, gold_idx, n, n_seed, budget, hops)
                if dense_recall is None:                              # source-independent -> capture once
                    dense_recall = m["dense_recall"]
                rec[f"{name}_h{hops}"] = {"deg": m["graph_deg"], "union_recall": m["union_recall"],
                                          "lift": m["traversal_lift"], "reach": m["graph_reachable_of_missed"]}
            log.info("[edge-abl/%s] %s: h1 lift %+.1f / h2 lift %+.1f (deg %.1f)",
                     d, name, rec[f"{name}_h1"]["lift"], rec[f"{name}_h2"]["lift"], rec[f"{name}_h2"]["deg"])
        rec["dense_recall"] = dense_recall
        out[d] = rec
        del Xt
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    path = f"results/L2/graphlift_edgeablation_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def run_budget_matched(datasets=None, subdir="gte_qwen", n_seed=20, budgets=(20, 50, 100), te_cap=2000):
    """AIRTIGHT L3 claim: at a MATCHED total budget B, does a hybrid (half dense-cosine + half
    structural-neighbourhood, hop-then-cosine ranked) beat spending all B on dense? Answers the
    reviewer critique that the union used a larger budget than dense. Writes graphlift_budgetmatched_{subdir}.json."""
    import torch
    from src.experiments.l1l3_recall import _graph
    datasets = datasets or DEFAULT_DATASETS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/L2", exist_ok=True)
    out = {}
    for d in datasets:
        eng, n, id2idx, qn, gold_idx, Xt = _prep(d, subdir, te_cap, device)
        _, A, _ = _graph(eng, n, id2idx, sources=("struct", "syn")); A = A.tocsr()
        maxB = max(budgets)
        acc = {B: {"dense": [], "hybrid": []} for B in budgets}
        with torch.no_grad():
            for qi in range(len(gold_idx)):
                g = set(gold_idx[qi])
                if not g:
                    continue
                sim = (torch.tensor(qn[qi:qi + 1], device=device) @ Xt.T)[0]
                order = torch.topk(sim, min(max(maxB, n_seed), n)).indices.cpu().numpy()
                simc = sim.cpu().numpy()
                seeds = [int(x) for x in order[:n_seed]]
                h1 = _neighbors(A, seeds) - set(seeds)                # 1-hop, then 2-hop; rank each by cosine
                h2 = _neighbors(A, h1) - h1 - set(seeds)
                graph_ranked = sorted(h1, key=lambda x: -simc[x]) + sorted(h2, key=lambda x: -simc[x])
                for B in budgets:
                    dense_B = set(int(x) for x in order[:B])
                    db = B // 2                                       # matched budget: half dense, half structural
                    hybrid = set(int(x) for x in order[:db]) | set(graph_ranked[:B - db])
                    acc[B]["dense"].append(len(g & dense_B) / len(g))
                    acc[B]["hybrid"].append(len(g & hybrid) / len(g))
        rec = {"corpus_N": int(n), "n_seed": n_seed}
        for B in budgets:
            dr = 100 * float(np.mean(acc[B]["dense"])); hr = 100 * float(np.mean(acc[B]["hybrid"]))
            rec[f"B{B}"] = {"dense": round(dr, 2), "hybrid": round(hr, 2), "lift": round(hr - dr, 2)}
        out[d] = rec
        msg = " | ".join(f"B{B}: dense {rec[f'B{B}']['dense']:.1f} vs hybrid {rec[f'B{B}']['hybrid']:.1f} ({rec[f'B{B}']['lift']:+.1f})" for B in budgets)
        log.info("[budget-matched/%s] N=%d | %s", d, n, msg)
        del Xt
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    path = f"results/L2/graphlift_budgetmatched_{subdir}.json"
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
    p.add_argument("--edge-ablation", action="store_true",
                   help="decompose the traversal lift by edge type (struct vs syn vs both) x hops")
    p.add_argument("--budget-matched", action="store_true",
                   help="airtight test: hybrid (half dense + half structural) vs all-dense at matched budget B")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if a.budget_matched:
        run_budget_matched(datasets=a.datasets, subdir=a.subdir, n_seed=a.n_seed, te_cap=a.te_cap)
    elif a.edge_ablation:
        run_edge_ablation(datasets=a.datasets, subdir=a.subdir, n_seed=a.n_seed, budget=a.budget, te_cap=a.te_cap)
    else:
        run(datasets=a.datasets, subdir=a.subdir, n_seed=a.n_seed, budget=a.budget, hops=a.hops, te_cap=a.te_cap)


if __name__ == "__main__":
    main()
