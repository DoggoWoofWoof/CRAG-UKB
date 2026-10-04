"""C8c — TOP-50 RESIDUAL RERANKER. Within the already-good top-50 under the C7b soft-archetype fusion, learn
which candidates deserve the first five positions. Compact tabular LambdaMART (XGBRanker, rank:ndcg, NDCG@5).
Inference-safe candidate features only (no gold-derived, no new graph/relation feature banks). Candidates below
top-50 retain their base order. Reuses l2_c8 machinery."""
import os, sys, json
sys.path.insert(0, "scratchpad")
import numpy as np, l2_c8 as C8
import l2_c6 as C6
POOL = 50
FEATNAMES = ["base_rank", "base_score", "pool_pos",
             "rk_dense", "rk_splade", "rk_offset", "rk_mixture", "rk_relation",
             "c_dense", "c_splade", "c_offset", "c_mixture", "c_relation",
             "rel_mask", "rel_raw", "min_rk4", "mean_rk4", "max_contrib",
             "d_splade_diff", "off_mix_diff", "votes_top10",
             "w_dense", "w_splade", "w_offset", "w_mixture", "w_relation",
             "q_relany", "q_disagree"]


def _per_expert_ranks(C):
    n = C.shape[1]; rk = np.empty((5, n), np.float64)
    for i in range(5):
        rk[i] = C8.ranks_from_score(C[i])
    return rk


def build_pool(ds, split, qi_list, base_wts, relany, disagree, want_meta=False):
    """For each query: base fusion score, take top-POOL candidates, emit candidate feature rows + labels + group.
    base_wts[r] = (5,) C7b soft weights for query r. Returns X, y, groups, and (optional) per-query meta for eval."""
    R = C6._load_raw(ds, split); off = R["query_offsets"]
    X = []; Y = []; G = []; meta = []
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s; gold = set(np.where(R["labels"][s:e] == 1)[0].tolist())
        C = C8.contribs_of(R, s, e); w = base_wts[r]
        base = (w[:, None] * C).sum(0); brank = C8.ranks_from_score(base)
        rk = _per_expert_ranks(C); relraw = R["relation_qwen_score"][s:e].astype(np.float64); relmask = (R["relation_mask"][s:e] > 0).astype(np.float64)
        pool = np.argsort(-base, kind="stable")[:min(POOL, n)]
        min4 = rk[:4].min(0); mean4 = rk[:4].mean(0); maxc = C.max(0)
        votes = (rk < 10).sum(0)   # #experts ranking candidate in their top10
        for pp, li in enumerate(pool):
            row = [brank[li], base[li], pp,
                   rk[0, li], rk[1, li], rk[2, li], rk[3, li], rk[4, li],
                   C[0, li], C[1, li], C[2, li], C[3, li], C[4, li],
                   relmask[li], relraw[li], min4[li], mean4[li], maxc[li],
                   rk[0, li] - rk[1, li], rk[2, li] - rk[3, li], votes[li],
                   w[0], w[1], w[2], w[3], w[4], float(relany[r]), float(disagree[r])]
            X.append(row); Y.append(1 if int(li) in gold else 0)
        G.append(len(pool))
        if want_meta:
            meta.append({"qi": int(qi), "pool": pool.astype(np.int64), "brank": brank, "gold": np.array(sorted(gold), np.int64), "n": n})
    return np.asarray(X, np.float32), np.asarray(Y, np.int8), np.asarray(G, np.int64), meta


def rerank_eval(ds, split, qi_list, base_wts, model, relany, disagree, want_ranks=False):
    """Apply reranker within top-POOL; candidates below keep base order. Full top-5 + deep metric suite."""
    X, Y, G, meta = build_pool(ds, split, qi_list, base_wts, relany, disagree, want_meta=True)
    scores = model.predict(X); pos = 0
    acc = {k: [] for k in ("any5", "ndcg5", "ndcg50", "mrr", "all5", "all10", "all50")}
    g_in5 = 0; g_tot = 0; rec_macro = []; a5f = []; goldranks = {}
    pq = {k: [] for k in ("qi", "ndcg5", "ndcg50", "recall5", "any5", "mrr", "all5", "all10", "all50")}
    for m in meta:
        pool = m["pool"]; k = len(pool); sc = scores[pos:pos + k]; pos += k
        n = m["n"]; brank = m["brank"]
        # final ranks: pool reordered by reranker score desc -> 0..k-1 ; non-pool keep base order -> k..n-1
        final = np.empty(n, np.int64)
        pool_order = pool[np.argsort(-sc, kind="stable")]; final[pool_order] = np.arange(k)
        nonpool = np.setdiff1d(np.arange(n), pool, assume_unique=False)
        nonpool_sorted = nonpool[np.argsort(brank[nonpool], kind="stable")]; final[nonpool_sorted] = np.arange(k, n)
        mt = C8.topk_metrics(-final.astype(np.float64), m["gold"])
        for kk in acc: acc[kk].append(mt[kk])
        g_in5 += mt["golds_in5"]; g_tot += mt["ng"]; rec_macro.append(mt["recall5"])
        if mt["all5_feas"] is not None: a5f.append(mt["all5_feas"])
        for kk in ("any5", "ndcg5", "ndcg50", "mrr", "all5", "all10", "all50"): pq[kk].append(mt[kk])
        pq["recall5"].append(mt["recall5"]); pq["qi"].append(m["qi"])
        if want_ranks:
            for gl in m["gold"]: goldranks[(m["qi"], int(gl))] = int(final[gl])
    out = {"n": len(qi_list), "any5": float(np.mean(acc["any5"])), "recall5_micro": g_in5 / max(g_tot, 1),
           "recall5_macro": float(np.mean(rec_macro)), "ndcg5": float(np.mean(acc["ndcg5"])),
           "ndcg50": float(np.mean(acc["ndcg50"])), "mrr": float(np.mean(acc["mrr"])),
           "all5": float(np.mean(acc["all5"])), "all10": float(np.mean(acc["all10"])), "all50": float(np.mean(acc["all50"])),
           "all5_feas": float(np.mean(a5f)) if a5f else None, "n_all5_feas": len(a5f), "_gtot": g_tot,
           "_perq": {k: np.array(v) for k, v in pq.items()}}
    if want_ranks: out["_goldranks"] = goldranks
    return out


def base_weight_rows(head, ds, split, qi_list):
    return C8.soft_weights(*head[:3], ds, split, qi_list)


def train_ranker(Xtr, Ytr, Gtr, Xdev, Ydev, Gdev, seed=0, log=print):
    from xgboost import XGBRanker
    model = XGBRanker(objective="rank:ndcg", eval_metric=["ndcg@5"], n_estimators=400, learning_rate=0.05,
                      max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      random_state=seed, early_stopping_rounds=40, tree_method="hist")
    model.fit(Xtr, Ytr, group=Gtr, eval_set=[(Xdev, Ydev)], eval_group=[Gdev], verbose=False)
    log(f"  best_iteration={model.best_iteration} best_score(ndcg@5)={model.best_score:.4f}")
    return model
