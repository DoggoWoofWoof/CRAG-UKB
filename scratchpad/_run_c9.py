"""C9 end-to-end (Tasks 1-17). Nested C9_TRAIN/C9_DEV inside old TRAIN_INNER; select on C9_DEV; confirm ONCE on
old DEV_INNER. NO official VAL, NO TEST. Stacks C8c via 3-fold OOF (Task 9/10 anti-leakage). Writes
results/L2/L2_C9_TOP5.{md-data in json}. Base fusion = C7b soft-archetype fit on C9_TRAIN only."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_c8 as C8, l2_c8c as C8C, l2_c9 as C9
from xgboost import XGBRanker, XGBClassifier
OUT = C8.OUT; log = lambda *a: print(*a, flush=True); T0 = time.time()
DS = C8.DS; RB = ["5-9", "10-14", "15-19", "20-29", "30-49"]


def rbin(r):
    return "5-9" if r < 10 else "10-14" if r < 15 else "15-19" if r < 20 else "20-29" if r < 30 else "30-49"


def combined(split, qi_by_ds, base_head):
    bs = [C9.build_bundle(ds, split, qi_by_ds[ds], base_head) for ds in DS]
    out = {"X28": np.concatenate([b["X28"] for b in bs]), "Xex": np.concatenate([b["Xex"] for b in bs]),
           "y": np.concatenate([b["y"] for b in bs]), "groups": np.concatenate([b["groups"] for b in bs]),
           "meta": [dict(m, ds=ds) for ds, b in zip(DS, bs) for m in b["meta"]]}
    return out


def starts_of(groups):
    return np.concatenate([[0], np.cumsum(groups)]).astype(np.int64)


def rows_groups(qidx, groups, st):
    rows = np.concatenate([np.arange(st[i], st[i + 1]) for i in qidx]); gg = groups[qidx]
    return rows, gg


def fit_c8c(Xtr, ytr, gtr, Xdev, ydev, gdev, n_est=400, depth=6, seed=0):
    m = XGBRanker(objective="rank:ndcg", eval_metric=["ndcg@5"], n_estimators=n_est, learning_rate=0.05,
                  max_depth=depth, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                  tree_method="hist", random_state=seed, early_stopping_rounds=40)
    m.fit(Xtr, ytr, group=gtr, eval_set=[(Xdev, ydev)], eval_group=[gdev], verbose=False); return m


def fit_c9(Xtr, ytr, gtr, Xdev, ydev, gdev, depth, npair, seed=0):
    m = XGBRanker(objective="rank:ndcg", eval_metric=["ndcg@5"], lambdarank_pair_method="topk",
                  lambdarank_num_pair_per_sample=npair, n_estimators=600, learning_rate=0.05, max_depth=depth,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                  random_state=seed, early_stopping_rounds=40)
    m.fit(Xtr, ytr, group=gtr, eval_set=[(Xdev, ydev)], eval_group=[gdev], verbose=False); return m


def slim(p, keys=C8.AGG):
    return {k: (round(p[k], 4) if isinstance(p.get(k), float) else p.get(k)) for k in keys}


def paired_boot(a, b, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); dif = a - b
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(nboot)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(dif.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "significant": bool(lo > 0 or hi < 0)}


def rescue(base_gr, new_gr):
    src = ["5-9", "10-19", "20-49"]; resc = {b: 0 for b in src}; hurt = 0
    for key, ra in base_gr.items():
        rb = new_gr.get(key)
        if rb is None: continue
        if rb < 5 and ra >= 5:
            b = "5-9" if ra < 10 else "10-19" if ra < 20 else "20-49" if ra < 50 else "50+"
            resc[b] = resc.get(b, 0) + 1
        elif ra < 5 and rb >= 5: hurt += 1
    return {"rescued_by_bin": resc, "TOP5_RESCUE": sum(resc.values()), "TOP5_HURT": hurt,
            "NET_TOP5_GAIN": sum(resc.values()) - hurt}


def multigold(res):
    pq = res["_perq"]; ng = pq["ng"]; out = {}
    for lab, mask in (("1", ng == 1), ("2", ng == 2), ("3+", ng >= 3)):
        if mask.sum() == 0: out[lab] = None; continue
        out[lab] = {"n": int(mask.sum()), "recall5": round(float(pq["recall5"][mask].mean()), 4),
                    "all5": round(float(pq["all5"][mask].mean()), 4), "ndcg5": round(float(pq["ndcg5"][mask].mean()), 4)}
    return out


def main():
    c9tr = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9train"] for ds in DS}
    c9dev = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9dev"] for ds in DS}
    devinner = {ds: C8.load_split(ds)[1] for ds in DS}
    base = C8.fit_soft_head("ndcg50", c9tr)   # base C7b fusion fit on C9_TRAIN only
    log(f"[{time.time()-T0:.0f}s] base C7b fit on C9_TRAIN")

    B_tr = combined("train", c9tr, base); B_dev = combined("train", c9dev, base); B_di = combined("train", devinner, base)
    log(f"[{time.time()-T0:.0f}s] bundles: tr rows={len(B_tr['y'])} dev rows={len(B_dev['y'])} di rows={len(B_di['y'])}")

    # ---------- 3-fold OOF C8c on C9_TRAIN + full C8c for dev/di ----------
    st = starts_of(B_tr["groups"]); Q = len(B_tr["groups"])
    fold = np.array([int(C8._h(m["qi"] * 31 + (0 if m["ds"] == DS[0] else 7)) * 3) % 3 for m in B_tr["meta"]])
    oof = np.zeros(len(B_tr["y"]), np.float32)
    for f in range(3):
        tr_q = np.where(fold != f)[0]; te_q = np.where(fold == f)[0]
        rtr, gtr = rows_groups(tr_q, B_tr["groups"], st); rte, gte = rows_groups(te_q, B_tr["groups"], st)
        m = fit_c8c(B_tr["X28"][rtr], B_tr["y"][rtr], gtr, B_dev["X28"], B_dev["y"], B_dev["groups"])
        oof[rte] = m.predict(B_tr["X28"][rte])
        log(f"[{time.time()-T0:.0f}s] OOF fold {f}: best_iter={m.best_iteration}")
    c8c_full = fit_c8c(B_tr["X28"], B_tr["y"], B_tr["groups"], B_dev["X28"], B_dev["y"], B_dev["groups"])
    log(f"[{time.time()-T0:.0f}s] C8c_full best_iter={c8c_full.best_iteration}")
    s_dev = c8c_full.predict(B_dev["X28"]); s_di = c8c_full.predict(B_di["X28"])
    np.savez(f"{OUT}/_c9_oof.npz", oof=oof, fold=fold, s_dev=s_dev, s_di=s_di)

    # residual features + assembled C9 matrices
    R_tr = C9.resid_feats(oof, B_tr["groups"]); R_dev = C9.resid_feats(s_dev, B_dev["groups"]); R_di = C9.resid_feats(s_di, B_di["groups"])
    X_tr = C9.assemble(B_tr, R_tr); X_dev = C9.assemble(B_dev, R_dev); X_di = C9.assemble(B_di, R_di)

    # C8c baseline rankings (from full model scores) for comparison
    c8c_dev = C9.eval_ranking(B_dev, s_dev); c8c_di = C9.eval_ranking(B_di, s_di)
    log(f"[{time.time()-T0:.0f}s] C8c DEV ndcg5={c8c_dev['ndcg5']:.4f} recall5={c8c_dev['recall5_macro']:.4f}")

    # ---------- Task 1-4 audits on C9_DEV ----------
    audit = audit_dev(B_dev, X_dev, s_dev)

    # ---------- Task 7 boundary-hard training subset ----------
    oof_rank = R_tr[:, 1]   # c8c oof rank within pool
    bh = (B_tr["y"] == 1) | (oof_rank < 20)
    bg = (oof_rank >= 20) & (np.array([C8._h(i) for i in range(len(oof_rank))]) < 0.34)   # ~1/3 background
    keep = bh | bg
    # rebuild groups for the kept subset
    gtr_bh = []; rows_bh = []
    for i in range(Q):
        a, b = st[i], st[i + 1]; km = np.where(keep[a:b])[0] + a
        if (B_tr["y"][a:b] == 1).sum() == 0: continue          # skip queries with no gold in pool
        rows_bh.append(km); gtr_bh.append(len(km))
    rows_bh = np.concatenate(rows_bh); gtr_bh = np.array(gtr_bh)
    log(f"[{time.time()-T0:.0f}s] boundary-hard: kept {len(rows_bh)}/{len(B_tr['y'])} rows ({len(gtr_bh)} groups)")

    # ---------- Task 6 C9a grid (<=4 configs) ----------
    grid = [(4, 8), (4, 16), (6, 8), (6, 16)]; best = None; c9a_hist = {}
    for depth, npair in grid:
        m = fit_c9(X_tr[rows_bh], B_tr["y"][rows_bh], gtr_bh, X_dev, B_dev["y"], B_dev["groups"], depth, npair)
        r = C9.eval_ranking(B_dev, m.predict(X_dev)); key = (round(r["ndcg5"], 5), round(r["recall5_macro"], 5))
        c9a_hist[f"d{depth}_p{npair}"] = {"ndcg5": round(r["ndcg5"], 4), "recall5": round(r["recall5_macro"], 4),
                                          "all5_feas": round(r["all5_feas"], 4), "best_iter": int(m.best_iteration)}
        log(f"[{time.time()-T0:.0f}s] C9a d{depth} p{npair}: ndcg5={r['ndcg5']:.4f} recall5={r['recall5_macro']:.4f} all50={r['all50']:.4f}")
        if best is None or key > best[0]: best = (key, m, r, (depth, npair))
    c9a_model, c9a_dev, c9a_cfg = best[1], best[2], best[3]
    log(f"[{time.time()-T0:.0f}s] C9a selected d{c9a_cfg[0]} p{c9a_cfg[1]} ndcg5={c9a_dev['ndcg5']:.4f}")

    # ---------- Task 11 classifier diagnostic (top20 pointwise) ----------
    clf_dev = classifier_diag(B_tr, X_tr, oof_rank, B_dev, X_dev, R_dev[:, 1])

    # ---------- Task 9 C9b boundary residual over top20 (OOF C9a) ----------
    c9b = c9b_stage(B_tr, X_tr, B_dev, X_dev, rows_bh, gtr_bh, fold, st, oof_rank, R_dev[:, 1], c9a_model, c9a_dev)

    # ---------- select best RANKING model on C9_DEV (classifier is diagnostic only), confirm on DEV_INNER ----------
    cands = {"C8c": c8c_dev, "C9a": c9a_dev}
    if c9b: cands["C9b"] = c9b["dev"]
    sel = max(cands, key=lambda k: (cands[k]["ndcg5"], cands[k]["recall5_macro"]))
    log(f"[{time.time()-T0:.0f}s] SELECTED on C9_DEV = {sel}")

    # confirm selected + C8c on DEV_INNER (secondary held-out)
    conf = {"C8c": c8c_di, "C9a": C9.eval_ranking(B_di, c9a_model.predict(X_di))}
    if c9b: conf["C9b"] = c9b["apply_di"](X_di, B_di)
    sel_di = conf.get(sel, conf["C9a"])

    # ---------- diagnostics for the selected model vs C8c on C9_DEV ----------
    sel_dev = cands[sel]
    resc = rescue(c8c_dev["_goldranks"], sel_dev["_goldranks"])
    spec = specialist_rescue(B_dev, c8c_dev["_goldranks"], sel_dev["_goldranks"])
    mg = {"C8c": multigold(c8c_dev), sel: multigold(sel_dev)}
    ogap = oracle_gap(B_dev, sel_dev)

    boot = {m_: paired_boot(np.concatenate([sel_dev["_perq"][mk]]), np.concatenate([c8c_dev["_perq"][mk]]))
            for m_, mk in (("ndcg5", "ndcg5"), ("recall5", "recall5"), ("ndcg50", "ndcg50"), ("all50", "all50"))}

    res = {"phase": "C9 — top-5 boundary + expert-consensus reranking (C9_DEV selection; DEV_INNER confirm; NO VAL/TEST)",
           "split": {ds: {"c9train": int(len(c9tr[ds])), "c9dev": int(len(c9dev[ds])), "dev_inner": int(len(devinner[ds]))} for ds in DS},
           "n_features": X_tr.shape[1], "c9a_config": {"max_depth": c9a_cfg[0], "num_pair": c9a_cfg[1]},
           "c9a_grid": c9a_hist, "boundary_hard_kept_frac": round(len(rows_bh) / len(B_tr["y"]), 4),
           "AUDIT": audit,
           "C9_DEV": {"C8c": slim(c8c_dev), "C9a": slim(c9a_dev), **({"C9b": slim(c9b["dev"])} if c9b else {}),
                      "C9_classifier": slim(clf_dev["dev"])},
           "SELECTED_ON_C9DEV": sel,
           "DEV_INNER_CONFIRM": {k: slim(v) for k, v in conf.items()},
           "bootstrap_selected_vs_C8c_on_C9DEV": boot,
           "TOP5_RESCUE": resc, "SPECIALIST_RESCUE": spec, "MULTI_GOLD": mg, "ORACLE_GAP": ogap,
           "C9a_feature_importance_top": feat_imp(c9a_model),
           "classifier_diagnostic": {"dev": slim(clf_dev["dev"]), "note": clf_dev["note"]},
           }
    if c9b: res["c9b"] = {k: v for k, v in c9b.items() if k not in ("dev", "apply_di", "model")}
    # gates
    nd = sel_dev["ndcg5"]; rc = sel_dev["recall5_macro"]
    res["GATES"] = gates(nd, rc, sel, c8c_dev, sel_dev, audit, c9b, clf_dev, ogap)
    json.dump(res, open("results/L2/L2_C9_TOP5.json", "w"), indent=1, default=str)
    import joblib; joblib.dump(c9a_model, f"{OUT}/C9a_model.joblib")
    log(f"[{time.time()-T0:.0f}s] C9_DONE wrote results/L2/L2_C9_TOP5.json (selected={sel})")


# ============================================================ audits & sub-stages
def audit_dev(B, X, s):
    """Task 1-4: FN/FP audit + gold vs non-gold support-feature stats on C9_DEV under C8c ranking."""
    names = C9.C9_FEATNAMES; idx = {n: i for i, n in enumerate(names)}
    st = starts_of(B["groups"]); fn_bins = {b: 0 for b in RB}; n_fp = 0; n_fn = 0
    key = ["max_contrib", "n_top5", "n_top10" if "n_top10" in idx else "votes_top10", "best_rank5", "second_min_rank5",
           "best_minus_second_contrib", "max_second_ratio", "contrib_entropy", "trusted_max_contrib",
           "regime_support_max", "regime_support_second", "rank_std5", "n_top3"]
    key = [k for k in key if k in idx]
    acc = {grp: {k: [] for k in key} for grp in ("top5_gold", "fn_gold", "fp_nongold", "other_nongold")}
    pos = 0
    for m, g in zip(B["meta"], B["groups"]):
        sc = s[pos:pos + g]; rankof = np.empty(g, np.int64); rankof[np.argsort(-sc, kind="stable")] = np.arange(g)
        goldset = set(m["gold"].tolist()); pool = m["pool"]
        for j in range(g):
            li = int(pool[j]); r = int(rankof[j]); isg = li in goldset
            grp = ("top5_gold" if (isg and r < 5) else "fn_gold" if (isg and r >= 5) else
                   "fp_nongold" if ((not isg) and r < 5) else "other_nongold")
            if grp == "fn_gold": fn_bins[rbin(r)] += 1; n_fn += 1
            if grp == "fp_nongold": n_fp += 1
            for k in key: acc[grp][k].append(float(X[pos + j, idx[k]]))
        pos += g
    stats = {grp: {k: round(float(np.mean(v)), 4) if v else None for k, v in acc[grp].items()} for grp in acc}
    return {"fn_gold_c8c_rank_bins": fn_bins, "n_false_negative_golds": n_fn, "n_false_positive_nongold": n_fp,
            "support_feature_means_by_group": stats,
            "interpretation_keys": "compare fp_nongold (spikes?) vs fn_gold (specialist-strong?) vs top5_gold (consensus)"}


def classifier_diag(B_tr, X_tr, oof_rank_tr, B_dev, X_dev, dev_rank):
    """Task 11: pointwise gold classifier on top20; rank candidates by P(gold). DIAGNOSTIC ONLY (not auto-selected).
    Within-top20 rows ordered by P(gold); rows outside top20 pinned below in their C8c order."""
    mtr = oof_rank_tr < 20; mdev = dev_rank < 20
    clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, min_child_weight=5, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
                        scale_pos_weight=float((B_tr["y"][mtr] == 0).sum() / max((B_tr["y"][mtr] == 1).sum(), 1)))
    clf.fit(X_tr[mtr], B_tr["y"][mtr])
    p = np.where(mdev, 1.0 + clf.predict_proba(X_dev)[:, 1], -dev_rank).astype(np.float32)  # top20 above; rest by C8c rank
    r = C9.eval_ranking(B_dev, p)
    return {"dev": r, "note": "pointwise P(gold) on top20; rows outside top20 pinned below in C8c order; diagnostic only"}


def c9b_stage(B_tr, X_tr, B_dev, X_dev, rows_bh, gtr_bh, fold, st, oof_rank, dev_rank, c9a_model, c9a_dev):
    """Task 9/10: 2nd-stage boundary reranker over top20 under C9a, with OOF C9a score as a feature."""
    # OOF C9a on C9_TRAIN (3 folds) to get a clean stage-1 score feature
    Q = len(B_tr["groups"]); oof_c9a = np.zeros(len(B_tr["y"]), np.float32)
    for f in range(3):
        tr_q = np.where(fold != f)[0]; te_q = np.where(fold == f)[0]
        # boundary-hard rows restricted to this fold's train queries
        keep_rows = []; gg = []
        bh_set = set(rows_bh.tolist())    # reuse the same boundary-hard mask
        for qi in tr_q:
            a, b = st[qi], st[qi + 1]; km = [i for i in range(a, b) if i in bh_set]
            if (B_tr["y"][a:b] == 1).sum() == 0 or len(km) == 0: continue
            keep_rows += km; gg.append(len(km))
        if not keep_rows: continue
        keep_rows = np.array(keep_rows); gg = np.array(gg)
        m = fit_c9(X_tr[keep_rows], B_tr["y"][keep_rows], gg, X_dev, B_dev["y"], B_dev["groups"], 6, 16)
        rte, _ = rows_groups(te_q, B_tr["groups"], st); oof_c9a[rte] = m.predict(X_tr[rte])
    # stage-2 features = C9 features + oof C9a score/rank (within pool)
    def add_stage1(X, s1, groups):
        st2 = starts_of(groups); rank = np.zeros(len(s1), np.float32)
        for i in range(len(groups)):
            a, b = st2[i], st2[i + 1]; ro = np.empty(b - a, np.int64); ro[np.argsort(-s1[a:b], kind="stable")] = np.arange(b - a); rank[a:b] = ro
        return np.concatenate([X, s1.reshape(-1, 1), rank.reshape(-1, 1)], 1)
    X2_tr = add_stage1(X_tr, oof_c9a, B_tr["groups"])
    s1_dev = c9a_model.predict(X_dev); X2_dev = add_stage1(X_dev, s1_dev, B_dev["groups"])
    # stage-2 train only on top20 under stage-1 (boundary) ; label gold
    r1_tr = np.zeros(len(oof_c9a), np.float32); st1 = starts_of(B_tr["groups"])
    for i in range(len(B_tr["groups"])):
        a, b = st1[i], st1[i + 1]; ro = np.empty(b - a, np.int64); ro[np.argsort(-oof_c9a[a:b], kind="stable")] = np.arange(b - a); r1_tr[a:b] = ro
    mtr = r1_tr < 20; rows2 = []; g2 = []
    for i in range(len(B_tr["groups"])):
        a, b = st1[i], st1[i + 1]; km = np.where(mtr[a:b])[0] + a
        if (B_tr["y"][a:b][mtr[a:b]] == 1).sum() == 0 or len(km) == 0: continue
        rows2.append(km); g2.append(len(km))
    if not rows2: return None
    rows2 = np.concatenate(rows2); g2 = np.array(g2)
    m2 = fit_c9(X2_tr[rows2], B_tr["y"][rows2], g2, X2_dev, B_dev["y"], B_dev["groups"], 6, 16)
    # apply: within each pool, top20-under-stage1 ranked FIRST by stage-2 score; ranks 20-49 keep stage-1 order below.
    def apply(X2, s1, B):
        groups = B["groups"]; st_ = starts_of(groups); s2 = m2.predict(X2); fs = np.empty(len(s1), np.float64)
        for i in range(len(groups)):
            a, b = st_[i], st_[i + 1]; g = b - a
            r1 = np.empty(g, np.int64); r1[np.argsort(-s1[a:b], kind="stable")] = np.arange(g)
            in20 = r1 < 20
            # top20 rank by stage-2 (best=0); score = big - that rank  => all above the 20-49 block
            r2 = np.empty(g, np.int64); order2 = np.argsort(-np.where(in20, s2[a:b], -1e18), kind="stable"); r2[order2] = np.arange(g)
            fs[a:b] = np.where(in20, 1e6 - r2, -r1.astype(np.float64))
        return C9.eval_ranking(B, fs)
    dev = apply(X2_dev, s1_dev, B_dev)
    def apply_di(Xdi, Bdi):
        s1di = c9a_model.predict(Xdi); X2di = add_stage1(Xdi, s1di, Bdi["groups"]); return apply(X2di, s1di, Bdi)
    return {"dev": dev, "apply_di": apply_di, "model": m2, "dev_ndcg5": round(dev["ndcg5"], 4),
            "dev_recall5": round(dev["recall5_macro"], 4)}


def specialist_rescue(B, base_gr, new_gr):
    """Task 13: for each rescued gold, which expert gave the strongest RRF contribution."""
    from collections import Counter
    st = starts_of(B["groups"]); cnt = Counter(); exps = ["dense", "splade", "offset", "mixture", "relation"]
    idx28 = {n: i for i, n in enumerate(C9.C8C28)}
    ci = [idx28[f"c_{e}"] for e in exps]
    pos = 0
    for m, g in zip(B["meta"], B["groups"]):
        pool = m["pool"]; goldset = set(m["gold"].tolist())
        for j in range(g):
            li = int(pool[j])
            if li in goldset:
                key = (m["qi"], li)
                if new_gr.get(key, 99) < 5 and base_gr.get(key, 99) >= 5:
                    contribs = [B["X28"][pos + j, k] for k in ci]; cnt[exps[int(np.argmax(contribs))]] += 1
        pos += g
    return dict(cnt)


def oracle_gap(B, sel):
    """Task 17: selected recall5 vs perfect-rerank-of-top50 ceiling (given fixed pool)."""
    st = starts_of(B["groups"]); perf = []; ng_all = []
    for m in B["meta"]:
        ng = len(m["gold"]);
        if ng == 0: continue
        gp = ng   # all golds in pool by construction? count golds within pool
        golds_in_pool = len(set(m["gold"].tolist()) & set(m["pool"].tolist()))
        perf.append(min(golds_in_pool, 5) / ng)
    perfect = float(np.mean(perf))
    return {"selected_recall5": round(sel["recall5_macro"], 4), "perfect_rerank_top50_recall5": round(perfect, 4),
            "ORACLE_GAP_REMAINING_recall5": round(perfect - sel["recall5_macro"], 4)}


def feat_imp(model):
    fi = model.feature_importances_; names = C9.C9_FEATNAMES
    return dict(sorted({names[i]: round(float(fi[i]), 4) for i in range(len(names))}.items(), key=lambda kv: -kv[1])[:15])


def gates(nd, rc, sel, c8c, seld, audit, c9b, clf, ogap):
    return {"C9_NDCG5_ABOVE_090": "YES" if nd >= 0.90 else "NO",
            "C9_RECALL5_ABOVE_093": "YES" if rc >= 0.93 else "NO",
            "C9_RECALL5_ABOVE_095": "YES" if rc >= 0.95 else "NO",
            "EXPERT_CONSENSUS_FEATURES_HELP": "TBD_from_importance", "QUERY_REGIME_INTERACTIONS_HELP": "TBD_from_importance",
            "BOUNDARY_HARD_TRAINING_HELPS": "TBD", "OOF_STACKING_HELPS": "TBD",
            "SECOND_STAGE_RERANKER_HELPS": ("YES" if (c9b and c9b["dev"]["ndcg5"] > seld["ndcg5"]) else "NO") if c9b else "NOT_RUN",
            "C9_BEATS_C8C": "YES" if seld["ndcg5"] > c8c["ndcg5"] else "NO",
            "DEEP_RECALL_PRESERVED": "YES" if abs(seld["all50"] - c8c["all50"]) < 1e-9 else "NO",
            "TOP5_ORACLE_GAP_REMAINING": ogap["ORACLE_GAP_REMAINING_recall5"],
            "SAFE_TO_CONTINUE_TOP5_OPTIMIZATION": "YES", "SAFE_TO_FREEZE_L2": "NO"}


if __name__ == "__main__":
    main()
