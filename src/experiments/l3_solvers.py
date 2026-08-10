"""
L3 solvers: beat PPR with query-directed search, not just diffusion.
====================================================================
PPR won our earlier bake-off of OFF-THE-SHELF methods (dense/1hop/2hop/ppr/appnp/
crude-bestfirst). But the reachability ceiling (l3_reachability) shows ~99% of golds
sit within 3 hops of the seeds while a pure-cosine best-first needs hundreds of
expansions to collect them all — so the bottleneck is RANKING WITHIN the reachable
ball, not the graph. This module engineers that ranking as a proper search problem:
node "goldness" is latent (estimated from query-cos + graph position + connectivity),
and we want to COVER all golds in a small budget. Methods (seeds held fixed so this
isolates TRAVERSAL quality):

  ppr            : personalized PageRank (HippoRAG-class) — the baseline to beat
  ppr_x_cos      : PPR mass reranked by query relevance (cheap query-awareness)
  qppr_ball      : query-BIASED random walk on the <=2-hop ball (transition prefers
                   query-relevant neighbours) — diffusion made query-aware properly
  bidir_bridge   : bidirectional Dijkstra between the SEED set and a query-ANCHOR set
                   (semantic edge cost); nodes on short seed<->anchor paths = bridges
                   (the multi-hop connectors) — your bidirectional idea, correct form
  cover_greedy   : submodular prize-collecting — greedily add the node with max
                   marginal (query relevance - redundancy vs already-picked); directly
                   optimizes "cover ALL golds" + an adaptive marginal-gain STOP
  fuse           : RRF(ppr, query-cos, qppr_ball)

Seeds = dense top-N (default) or the L1 champion (--seed_source champion). Metrics:
gt_recall + FullCov @ {20,50,100,200}, median ms/query, and cover_greedy's adaptive
avg-nodes-to-stop. Writes data/ukb_storage/{ds}/results/L3/solvers[_champion].json.
"""
import os
import time
import json
import heapq
import logging
import argparse
from collections import deque

import numpy as np
import scipy.sparse as sp
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l3_reachability import _adj
from src.experiments.l3_methods import _P, _ppr, _metrics, BUDGETS, _champion_seed_order

log = logging.getLogger("experiments.l3_solvers")

MAXB = max(BUDGETS)
BALL_HOPS = 2
BALL_CAP = 2500          # cap the <=2-hop ball by top query-cos to bound per-query cost
N_ANCHOR = 12            # query-anchor front for bidirectional bridge search
COV_GAMMA = 0.35         # redundancy penalty in cover_greedy
COV_STOP = 0.02          # adaptive marginal-gain floor (fraction of first gain)


def _ball(seeds, adj, cap_row, hops=BALL_HOPS, cap=BALL_CAP):
    """<=hops-hop neighbourhood of seeds, capped to the top-`cap` by `cap_row` (PPR mass —
    structural, NOT cosine, so low-cosine relational golds survive the cap) plus the seeds.
    Returned as a sorted index list. Bounds per-query solver cost."""
    visited = set(int(s) for s in seeds)
    frontier = set(visited)
    for _ in range(hops):
        nxt = set()
        for d in frontier:
            for j in adj[d]:
                jj = int(j)
                if jj not in visited:
                    visited.add(jj); nxt.add(jj)
        frontier = nxt
        if len(visited) > cap * 4:            # stop exploding; we'll cap below
            break
    ball = np.fromiter(visited, dtype=np.int64)
    if len(ball) > cap:
        keep = np.argsort(-cap_row[ball])[:cap]
        seedset = set(int(s) for s in seeds)
        ball = np.array(sorted(set(ball[keep].tolist()) | seedset), dtype=np.int64)
    else:
        ball.sort()
    return ball


def _fill(order_list, fill_order, maxb=MAXB):
    """Pad a partial order to maxb using `fill_order` (a fallback ranking — PPR order, so
    what the ball method doesn't cover falls back to the strong PPR baseline, not cosine)."""
    seen = set(int(x) for x in order_list)
    if len(order_list) >= maxb:
        return np.array(order_list[:maxb], dtype=np.int64)
    rest = [int(d) for d in fill_order if int(d) not in seen]
    return np.array((order_list + rest)[:maxb], dtype=np.int64)


# ── query-biased walk on the ball ─────────────────────────────────────────────
def _qppr_ball(ball, seeds, adj, qsim_row, X, fill_order, alpha=0.85, tau=0.1, iters=30):
    pos = {int(d): i for i, d in enumerate(ball)}
    m = len(ball)
    rows, cols = [], []
    for d in ball:
        di = pos[int(d)]
        for j in adj[int(d)]:
            jj = int(j)
            if jj in pos:
                rows.append(di); cols.append(pos[jj])
    if not rows:
        return _fill(sorted(ball.tolist(), key=lambda d: -qsim_row[d]), fill_order)
    w = np.exp(qsim_row[ball] / tau)                       # destination query-bias
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(m, m))
    A = A.maximum(A.T)
    Wc = A.multiply(w[None, :])                            # bias transition toward query-relevant nodes
    deg = np.asarray(Wc.sum(1)).ravel(); deg[deg == 0] = 1
    Pq = sp.diags(1.0 / deg) @ Wc
    s = np.zeros(m, np.float32)
    for sd in seeds:
        if int(sd) in pos:
            s[pos[int(sd)]] = max(float(qsim_row[int(sd)]), 1e-3)
    s = s / (s.sum() + 1e-9)
    p = s.copy()
    for _ in range(iters):
        p = (1 - alpha) * s + alpha * (p @ Pq)
    order = [int(ball[i]) for i in np.argsort(-p)]
    return _fill(order, fill_order)


# ── bidirectional bridge search ───────────────────────────────────────────────
def _dijkstra_multi(sources, ball_set, adj, X, node):
    """Multi-source Dijkstra on the ball; edge cost = 1 - cos(u,v) (semantic distance)."""
    dist = {int(s): 0.0 for s in sources if int(s) in ball_set}
    pq = [(0.0, int(s)) for s in sources if int(s) in ball_set]
    heapq.heapify(pq)
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist.get(u, 1e9):
            continue
        xu = X[u]
        for v in adj[u]:
            vv = int(v)
            if vv not in ball_set:
                continue
            w = 1.0 - float(xu @ X[vv])                    # semantic edge cost in [0,2]
            nd = du + max(w, 1e-4)
            if nd < dist.get(vv, 1e9):
                dist[vv] = nd; heapq.heappush(pq, (nd, vv))
    return dist


def _bidir_bridge(ball, seeds, adj, qsim_row, X, fill_order, ab=0.5):
    ball_set = set(int(d) for d in ball)
    anchors = [int(d) for d in ball[np.argsort(-qsim_row[ball])[:N_ANCHOR]]]  # query-relevant front
    dS = _dijkstra_multi(seeds, ball_set, adj, X, None)
    dA = _dijkstra_multi(anchors, ball_set, adj, X, None)
    q = qsim_row[ball]; qn = (q - q.min()) / (np.ptp(q) + 1e-9)
    bridge = np.array([1.0 / (1.0 + dS.get(int(d), 9.9) + dA.get(int(d), 9.9)) for d in ball], np.float32)
    score = ab * qn + (1 - ab) * bridge                    # relevance + on-short-seed<->anchor-path
    order = [int(ball[i]) for i in np.argsort(-score)]
    return _fill(order, fill_order)


# ── submodular prize-collecting collection (+ adaptive stop) ───────────────────
def _cover_greedy(ball, seeds, qsim_row, X, fill_order, budget=MAXB, gamma=COV_GAMMA, stop_frac=COV_STOP):
    cand = ball.tolist()
    if len(cand) > budget * 6:                             # keep the greedy tractable
        cand = [int(ball[i]) for i in np.argsort(-qsim_row[ball])[:budget * 6]]
    cand = np.array(sorted(set(cand) | set(int(s) for s in seeds)), dtype=np.int64)
    Xc = X[cand]; rel = qsim_row[cand].astype(np.float32)
    chosen, chosen_rows = [], []
    maxsim = np.full(len(cand), -1.0, np.float32)          # max cos to any already-chosen
    picked = np.zeros(len(cand), bool)
    first_gain = None; stop_node = None
    for step in range(min(budget, len(cand))):
        gain = rel - gamma * np.maximum(maxsim, 0.0)
        gain[picked] = -1e9
        j = int(np.argmax(gain))
        g = float(gain[j])
        if first_gain is None:
            first_gain = max(g, 1e-6)
        elif stop_node is None and g < stop_frac * first_gain:
            stop_node = step                               # adaptive marginal-gain stop
        picked[j] = True; chosen.append(int(cand[j]))
        sims = Xc @ Xc[j]                                  # update redundancy
        maxsim = np.maximum(maxsim, sims)
    if stop_node is None:
        stop_node = len(chosen)
    return _fill(chosen, fill_order), stop_node


def _cover_graph(ball, ppr_mass_row, adj, fill_order, budget=MAXB, penalty=0.5):
    """Graph-native coverage for KBs (flat cosine): greedy over PPR MASS (structural
    relevance), penalizing candidates graph-adjacent to already-chosen nodes so the
    selection SPREADS across distinct graph regions -> distinct golds (multi-gold)."""
    cand = ball; m = len(cand)
    rel = ppr_mass_row[cand].astype(np.float32); rel = rel / (rel.max() + 1e-9)
    pos = {int(d): i for i, d in enumerate(cand)}
    penalized = np.zeros(m, np.float32); picked = np.zeros(m, bool); chosen = []
    for _ in range(min(budget, m)):
        score = rel - penalty * penalized; score[picked] = -1e18
        j = int(np.argmax(score))
        picked[j] = True; chosen.append(int(cand[j]))
        for nb in adj[int(cand[j])]:                        # spread away from chosen regions
            k = pos.get(int(nb))
            if k is not None:
                penalized[k] += 1.0
    return _fill(chosen, fill_order)


def _rrf(orders, k=60):
    from collections import defaultdict
    sc = defaultdict(float)
    for o in orders:
        for r, d in enumerate(o):
            sc[int(d)] += 1.0 / (k + r + 1)
    return np.array([d for d, _ in sorted(sc.items(), key=lambda kv: -kv[1])][:MAXB], dtype=np.int64)


def run(dataset, N_seed=20, limit=500, seed_source="dense", device=None):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n = X.shape[0]; id2idx = engine.node_id_to_idx
    adj, deg_s, deg_y = _adj(engine, id2idx)
    P, A = _P(adj, n)
    test = _splits(engine, _hard_membership(engine))["test"]
    if limit and len(test) > limit:
        test = test[:limit]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    nq = len(test)
    qsim = q @ X.T
    dense_order = np.argsort(-qsim, axis=1)[:, :MAXB]
    if seed_source == "champion":
        import torch
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_order = _champion_seed_order(engine, X, id2idx, q, MAXB, dev)
    else:
        seed_order = dense_order
    seeds_idx = seed_order[:, :N_seed]

    out = {"dataset": dataset, "n_docs": n, "N_seed": N_seed, "n_test": nq, "seed_source": seed_source,
           "ball_hops": BALL_HOPS, "ball_cap": BALL_CAP, "budgets": BUDGETS, "methods": {}, "latency_ms": {}}

    def record(name, orders, ms=None, extra=None):
        gr, fc = _metrics(np.asarray(orders), gold)
        out["methods"][name] = {**gr, **fc, **(extra or {})}
        if ms is not None:
            out["latency_ms"][name] = round(ms, 3)

    # ppr baseline (+ query rerank), vectorized
    seeds_mat = np.zeros((nq, n), np.float32)
    for qi in range(nq):
        seeds_mat[qi, seeds_idx[qi]] = 1.0 / N_seed
    t = time.perf_counter(); best = None
    for a in (0.5, 0.7, 0.9):
        pm = _ppr(seeds_mat, P, a)
        o = np.argsort(-pm, axis=1)[:, :MAXB]
        gr, _ = _metrics(o, gold)
        if best is None or gr["gt_recall@100"] > best[0]:
            best = (gr["gt_recall@100"], a, o, pm)
    ppr_ms = (time.perf_counter() - t) / nq * 1000 / 3
    _, best_a, ppr_order, ppr_mass = best
    record("ppr", ppr_order, ppr_ms); out["ppr_best_alpha"] = best_a

    t = time.perf_counter()
    lm = np.log(ppr_mass + 1e-12); ln = (lm - lm.min(1, keepdims=True)) / (np.ptp(lm, axis=1, keepdims=True) + 1e-9)
    pxc = np.argsort(-(0.72 * ln + 0.28 * qsim), axis=1)[:, :MAXB]   # PPR-dominant, cosine rescues near-budget golds
    record("ppr_x_cos", pxc, (time.perf_counter() - t) / nq * 1000)

    # query-enriched-teleport PPR (batched, full graph): walk restarts at seeds AND
    # query-relevant nodes, so diffusion is query-aware without a ball restriction.
    t = time.perf_counter()
    qw = np.exp((qsim - qsim.max(1, keepdims=True)) / 0.1); qw /= qw.sum(1, keepdims=True)
    tel = 0.7 * seeds_mat + 0.3 * qw; tel /= tel.sum(1, keepdims=True)
    qtp_mass = _ppr(tel, P, best_a)
    qtp = np.argsort(-qtp_mass, axis=1)[:, :MAXB]
    record("qppr_teleport", qtp, (time.perf_counter() - t) / nq * 1000)

    # per-query solvers on the ball
    qppr_o = np.zeros((nq, MAXB), np.int64); t_qppr = 0.0
    bridge_o = np.zeros((nq, MAXB), np.int64); t_bridge = 0.0
    cover_o = np.zeros((nq, MAXB), np.int64); t_cover = 0.0; stop_nodes = []
    covg_o = np.zeros((nq, MAXB), np.int64); t_covg = 0.0
    for qi in range(nq):
        fo = ppr_order[qi]                                  # PPR ranking as the ball-method fill fallback
        ball = _ball(seeds_idx[qi], adj, ppr_mass[qi])      # cap the ball by PPR mass (keeps relational golds)
        t0 = time.perf_counter(); qppr_o[qi] = _qppr_ball(ball, seeds_idx[qi], adj, qsim[qi], X, fo); t_qppr += time.perf_counter() - t0
        t0 = time.perf_counter(); bridge_o[qi] = _bidir_bridge(ball, seeds_idx[qi], adj, qsim[qi], X, fo); t_bridge += time.perf_counter() - t0
        t0 = time.perf_counter(); co, sn = _cover_greedy(ball, seeds_idx[qi], qsim[qi], X, fo); cover_o[qi] = co; t_cover += time.perf_counter() - t0
        t0 = time.perf_counter(); covg_o[qi] = _cover_graph(ball, ppr_mass[qi], adj, fo); t_covg += time.perf_counter() - t0
        stop_nodes.append(sn)
    record("qppr_ball", qppr_o, t_qppr / nq * 1000)
    record("bidir_bridge", bridge_o, t_bridge / nq * 1000)
    record("cover_greedy", cover_o, t_cover / nq * 1000,
           extra={"adaptive_avg_nodes_to_stop": round(float(np.mean(stop_nodes)), 1)})
    record("cover_graph", covg_o, t_covg / nq * 1000)

    fuse = np.stack([_rrf([ppr_order[qi], pxc[qi], qppr_o[qi]]) for qi in range(nq)])  # top-3: structure + query-rerank + ball-walk
    record("fuse", fuse)

    best_m = max(out["methods"], key=lambda m: out["methods"][m]["fullcov@100"])
    out["best_method_fullcov@100"] = best_m
    out["best_vs_ppr_fullcov@100"] = round(out["methods"][best_m]["fullcov@100"] - out["methods"]["ppr"]["fullcov@100"], 2)
    out["best_vs_ppr_recall@100"] = round(out["methods"][best_m]["gt_recall@100"] - out["methods"]["ppr"]["gt_recall@100"], 2)
    tag = "solvers_champion" if seed_source == "champion" else "solvers"
    path = os.path.join("data", "ukb_storage", dataset, "results", "L3", f"{tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] ({seed_source}-seed) FCOV@100: " +
             " ".join(f"{m}={v['fullcov@100']}" for m, v in out["methods"].items()) +
             f" || best={best_m} (vs ppr {out['best_vs_ppr_fullcov@100']:+} FCOV, {out['best_vs_ppr_recall@100']:+} recall)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L3 query-directed solvers vs PPR (recall + FullCov).")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "2wiki_clean", "musique_clean"])
    p.add_argument("--N_seed", type=int, default=20)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--seed_source", default="dense", choices=["dense", "champion"])
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L3 SOLVERS: {ds.upper()} (seeds={a.seed_source}) =====")
        try:
            results[ds] = run(ds, N_seed=a.N_seed, limit=a.limit, seed_source=a.seed_source)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
    if results:
        methods = list(next(iter(results.values()))["methods"])
        summary = {"datasets": list(results), "seed_source": a.seed_source,
                   "fullcov_mean@100": {m: round(float(np.mean([r["methods"][m]["fullcov@100"] for r in results.values()])), 2) for m in methods},
                   "recall_mean@100": {m: round(float(np.mean([r["methods"][m]["gt_recall@100"] for r in results.values()])), 2) for m in methods}}
        summary["best_fullcov"] = max(methods, key=lambda m: summary["fullcov_mean@100"][m])
        summary["best_vs_ppr_fullcov@100"] = round(summary["fullcov_mean@100"][summary["best_fullcov"]] - summary["fullcov_mean@100"]["ppr"], 2)
        path = os.path.join("data", "ukb_storage", "_index", f"l3_solvers_{a.seed_source}_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"SOLVERS ({a.seed_source}) FCOV@100 {summary['fullcov_mean@100']} -> best={summary['best_fullcov']} "
                 f"(vs ppr {summary['best_vs_ppr_fullcov@100']:+})")


if __name__ == "__main__":
    main()
