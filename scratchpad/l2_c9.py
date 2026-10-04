"""C9 — TOP-5 BOUNDARY + EXPERT-CONSENSUS RERANKING. Extends the C8c top-50 residual reranker with
expert-consensus + query-regime×candidate-support interaction features, boundary-hard training, and
OUT-OF-FOLD C8c residual stacking. Same top-50 pool, same XGBRanker family, same inference-safe restriction.
No new experts, no pool enlargement, no TEST, no L1/L3.

Feature blocks:
  C8C28  : the 28 existing C8c features (reused verbatim; see l2_c8c.FEATNAMES)
  EXTRA  : consensus + regime×candidate interactions (Tasks 2/3/4) — only genuinely new information
  RESID  : C8c OOF residual features (Task 8) — appended by the driver after OOF prediction
"""
import os, sys, json
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_c8 as C8, l2_c8c as C8C
import l2_c6 as C6
EPS = 1e-9
C8C28 = C8C.FEATNAMES
EXTRA = ["best_rank5", "second_min_rank5", "mean_rank5", "median_rank5", "rank_std5",
         "second_contrib", "mean_contrib", "std_contrib", "best_minus_second_contrib",
         "max_mean_ratio", "max_second_ratio", "n_top3", "n_top5", "n_top20",
         "argmax_is_dense", "argmax_is_splade", "argmax_is_offset", "argmax_is_mixture", "argmax_is_relation",
         "wr_dense", "wr_splade", "wr_offset", "wr_mixture", "wr_relation",
         "weight_of_argmax_expert", "trusted_max_contrib", "regime_support_max", "regime_support_second",
         "contrib_entropy"]
RESID = ["c8c_oof_score", "c8c_oof_rank", "c8c_rank_minus5", "c8c_score_minus_rank5", "c8c_score_minus_rank1",
         "c8c_rank_percentile"]


def _extra_cols(Cp, rkp, w):
    """Cp,rkp = (5,P) contribs & full-scope ranks for pool candidates; w=(5,) query weights. -> (P,len(EXTRA))."""
    sr = np.sort(rkp, axis=0)
    best_rank5 = sr[0]; second_min = sr[1]; mean_rank5 = rkp.mean(0); median_rank5 = np.median(rkp, 0); rank_std5 = rkp.std(0)
    sc = np.sort(Cp, axis=0)[::-1]
    best_c = sc[0]; second_c = sc[1]; mean_c = Cp.mean(0); std_c = Cp.std(0)
    gap = best_c - second_c; max_mean = best_c / (mean_c + EPS); max_second = best_c / (second_c + EPS)
    n3 = (rkp < 3).sum(0); n5 = (rkp < 5).sum(0); n20 = (rkp < 20).sum(0)
    am = np.argmax(Cp, 0)
    oh = np.zeros((5, Cp.shape[1]), np.float32)
    oh[am, np.arange(Cp.shape[1])] = 1.0
    wr = w[:, None] * Cp
    wrs = np.sort(wr, axis=0)[::-1]; rss_max = wrs[0]; rss_second = wrs[1]
    w_argmax = w[am]; trusted = best_c * w_argmax
    p = Cp / (Cp.sum(0) + EPS); ent = -(p * np.log(p + EPS)).sum(0)
    cols = [best_rank5, second_min, mean_rank5, median_rank5, rank_std5,
            second_c, mean_c, std_c, gap, max_mean, max_second, n3, n5, n20,
            oh[0], oh[1], oh[2], oh[3], oh[4],
            wr[0], wr[1], wr[2], wr[3], wr[4],
            w_argmax, trusted, rss_max, rss_second, ent]
    return np.stack(cols, axis=1).astype(np.float32)


def build_bundle(ds, split, qi_list, base_head):
    """Full top-50 pools. Returns dict: X28,(m,28) Xex,(m,E) y,(m) groups,(Q) and meta list per query.
    Row order within a query = argsort(-base) (identical to C8c)."""
    R = C6._load_raw(ds, split); off = R["query_offsets"]
    W = C8.soft_weights(*base_head[:3], ds, split, qi_list)
    ra, di = _qmeta(ds, split, qi_list)
    X28 = []; Xex = []; Y = []; G = []; meta = []
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s; gold = set(np.where(R["labels"][s:e] == 1)[0].tolist())
        C = C8.contribs_of(R, s, e); w = W[r]
        base = (w[:, None] * C).sum(0); brank = C8.ranks_from_score(base)
        rk = np.stack([C8.ranks_from_score(C[i]) for i in range(5)]).astype(np.float64)
        relraw = R["relation_qwen_score"][s:e].astype(np.float64); relmask = (R["relation_mask"][s:e] > 0).astype(np.float64)
        pool = np.argsort(-base, kind="stable")[:min(C8C.POOL, n)]
        min4 = rk[:4].min(0); mean4 = rk[:4].mean(0); maxc = C.max(0); votes10 = (rk < 10).sum(0)
        for pp, li in enumerate(pool):
            X28.append([brank[li], base[li], pp, rk[0, li], rk[1, li], rk[2, li], rk[3, li], rk[4, li],
                        C[0, li], C[1, li], C[2, li], C[3, li], C[4, li], relmask[li], relraw[li],
                        min4[li], mean4[li], maxc[li], rk[0, li] - rk[1, li], rk[2, li] - rk[3, li], votes10[li],
                        w[0], w[1], w[2], w[3], w[4], float(ra[r]), float(di[r])])
            Y.append(1 if int(li) in gold else 0)
        Xex.append(_extra_cols(C[:, pool], rk[:, pool], w))
        G.append(len(pool))
        meta.append({"qi": int(qi), "pool": pool.astype(np.int64), "brank": brank,
                     "gold": np.array(sorted(gold), np.int64), "n": n})
    return {"X28": np.asarray(X28, np.float32), "Xex": np.concatenate(Xex).astype(np.float32),
            "y": np.asarray(Y, np.int8), "groups": np.asarray(G, np.int64), "meta": meta}


def _qmeta(ds, split, qi_list):
    M = np.load(f"{C8.CORP}/{ds}/{split}/expert_meta.npz")
    return (M["relation_any_signal_present"].astype(np.float32)[qi_list],
            M["dense_splade_disagreement"].astype(np.float32)[qi_list])


def resid_feats(scores, groups):
    """C8c OOF residual features per candidate, computed within each query's pool. -> (m,6)."""
    out = np.zeros((len(scores), 6), np.float32); pos = 0
    for g in groups:
        s = scores[pos:pos + g]; order = np.argsort(-s, kind="stable"); rankof = np.empty(g, np.int64); rankof[order] = np.arange(g)
        s5 = s[order[4]] if g > 4 else s[order[-1]]; s1 = s[order[0]]
        out[pos:pos + g, 0] = s; out[pos:pos + g, 1] = rankof; out[pos:pos + g, 2] = rankof - 5
        out[pos:pos + g, 3] = s - s5; out[pos:pos + g, 4] = s - s1; out[pos:pos + g, 5] = rankof / max(g, 1)
        pos += g
    return out


def assemble(bundle, resid):
    return np.concatenate([bundle["X28"], bundle["Xex"], resid], axis=1)


C9_FEATNAMES = C8C28 + EXTRA + RESID


def eval_ranking(bundle, cand_scores):
    """Rerank within top-50 by cand_scores; candidates below top50 keep base order. Full top-5 + deep suite +
    per-query arrays + per-gold final ranks. cand_scores aligned to bundle rows (pool order)."""
    acc = {k: [] for k in ("any5", "ndcg5", "ndcg50", "mrr", "all5", "all10", "all50")}
    g_in5 = 0; g_tot = 0; rec = []; a5f = []; pos = 0; goldranks = {}
    pq = {k: [] for k in ("qi", "ndcg5", "ndcg50", "recall5", "any5", "mrr", "all5", "all10", "all50", "ng")}
    for m, g in zip(bundle["meta"], bundle["groups"]):
        sc = cand_scores[pos:pos + g]; pos += g; pool = m["pool"]; n = m["n"]; brank = m["brank"]
        final = np.empty(n, np.int64); final[pool[np.argsort(-sc, kind="stable")]] = np.arange(g)
        nonpool = np.setdiff1d(np.arange(n), pool, assume_unique=False)
        final[nonpool[np.argsort(brank[nonpool], kind="stable")]] = np.arange(g, n)
        mt = C8.topk_metrics(-final.astype(np.float64), m["gold"])
        for k in acc: acc[k].append(mt[k])
        g_in5 += mt["golds_in5"]; g_tot += mt["ng"]; rec.append(mt["recall5"])
        if mt["all5_feas"] is not None: a5f.append(mt["all5_feas"])
        for k in ("any5", "ndcg5", "ndcg50", "mrr", "all5", "all10", "all50"): pq[k].append(mt[k])
        pq["recall5"].append(mt["recall5"]); pq["qi"].append(m["qi"]); pq["ng"].append(mt["ng"])
        for gl in m["gold"]: goldranks[(m["qi"], int(gl))] = int(final[gl])
    out = {"n": len(bundle["meta"]), "any5": float(np.mean(acc["any5"])), "recall5_micro": g_in5 / max(g_tot, 1),
           "recall5_macro": float(np.mean(rec)), "ndcg5": float(np.mean(acc["ndcg5"])), "ndcg50": float(np.mean(acc["ndcg50"])),
           "mrr": float(np.mean(acc["mrr"])), "all5": float(np.mean(acc["all5"])), "all10": float(np.mean(acc["all10"])),
           "all50": float(np.mean(acc["all50"])), "all5_feas": float(np.mean(a5f)) if a5f else None,
           "n_all5_feas": len(a5f), "_gtot": g_tot, "_perq": {k: np.array(v) for k, v in pq.items()}, "_goldranks": goldranks}
    return out
