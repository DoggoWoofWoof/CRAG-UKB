"""C10b — LEARNED GREEDY SET SELECTOR (Tasks 6-12). Marginal-utility classifier P(candidate is a needed
UNSELECTED gold | static C9 features + dynamic set features vs already-selected). Greedy 5-step selection over
C8c top20. Trained on TRAIN only, with a MIXTURE of teacher (oracle) prefixes and OUT-OF-FOLD model-rollout
prefixes (3 folds; Task 7 anti-leakage). Select on C9_DEV vs C8c and C10a; confirm once on DEV_INNER. Reorders
within top50 only -> ALL@50 preserved. NO official VAL, NO TEST."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_c8 as C8, l2_c9 as C9, l2_c10 as C10
from xgboost import XGBClassifier
from _run_c9 import combined, starts_of, multigold, rescue, paired_boot
from _run_c10_mmr import redundancy, singlegold, C8slim
OUT = C8.OUT; DS = C8.DS; CAP = 20; NPICK = 5; log = lambda *a: print(*a, flush=True); T0 = time.time()
FEATN = C9.C9_FEATNAMES + C10.DYNN


def build_X(B, resid_scores):
    return C9.assemble(B, C9.resid_feats(resid_scores, B["groups"]))


def gen_rows(meta, groups, X, scores, qidx, teacher=True, predict=None):
    """Emit (feats, labels) rows for queries in qidx. teacher=True -> oracle gold prefixes; else model rollout
    prefixes via `predict`. Window = C8c top-CAP; labels = candidate is an unselected in-window gold."""
    st = starts_of(groups); F = []; Y = []
    for i in qidx:
        m = meta[i]; a = st[i]; g = groups[i]; Xq = X[a:a + g]; sc = scores[a:a + g]
        qc = C10.qc_one(m, sc)
        order = np.argsort(-sc, kind="stable"); win = order[:min(CAP, g)]
        goldset = set(m["gold"].tolist()); gold_win = [int(w) for w in win if int(qc["pool"][w]) in goldset]
        if not gold_win: continue
        if teacher:
            gseq = sorted(gold_win, key=lambda w: -sc[w])          # place easy golds first
            nsteps = min(len(gseq), NPICK)
            for t in range(nsteps):
                S = np.array(gseq[:t], np.int64); cand = np.array([w for w in win if w not in set(gseq[:t])], np.int64)
                D = C10.dyn_block(qc, S, cand, t, NPICK); feats = np.concatenate([Xq[cand], D], axis=1)
                lab = np.array([1 if (int(qc["pool"][c]) in goldset) else 0 for c in cand], np.int8)
                F.append(feats); Y.append(lab)
        else:
            chosen = []; avail = list(win); remaining_gold = set(gold_win)
            for t in range(min(NPICK, len(win))):
                cand = np.array(avail, np.int64); D = C10.dyn_block(qc, np.array(chosen, np.int64), cand, t, NPICK)
                feats = np.concatenate([Xq[cand], D], axis=1)
                lab = np.array([1 if (int(qc["pool"][c]) in remaining_gold) else 0 for c in cand], np.int8)
                F.append(feats); Y.append(lab)
                p = predict(feats); pick = int(cand[int(np.argmax(p))]); chosen.append(pick); avail.remove(pick)
                remaining_gold.discard(pick)
                if not remaining_gold: break                        # nothing left to teach after all golds placed
    if not F: return np.zeros((0, len(FEATN)), np.float32), np.zeros(0, np.int8)
    return np.concatenate(F).astype(np.float32), np.concatenate(Y)


def fit_clf(X, y, seed=0):
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    clf = XGBClassifier(n_estimators=350, learning_rate=0.05, max_depth=5, min_child_weight=5, subsample=0.8,
                        colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
                        scale_pos_weight=spw, random_state=seed)
    clf.fit(X, y); return clf


def apply_selector(B, X, scores, clf):
    """Greedy rollout on every query; return C9.eval_ranking result."""
    st = starts_of(B["groups"]); predict = lambda f: clf.predict_proba(f)[:, 1]
    orders = []
    for i, (m, g) in enumerate(zip(B["meta"], B["groups"])):
        a = st[i]; qc = C10.qc_one(m, scores[a:a + g]); ordr = C10.rollout(qc, X[a:a + g], predict, CAP, NPICK)
        orders.append(ordr)
    sc = _orders_to_scores(B["groups"], orders); return C9.eval_ranking(B, sc), orders


def _orders_to_scores(groups, orders):
    out = []
    for g, order in zip(groups, orders):
        place = np.empty(g, np.int64); place[order] = np.arange(g); out.append(-place.astype(np.float64))
    return np.concatenate(out)


def main():
    c9tr = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9train"] for ds in DS}
    c9dev = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9dev"] for ds in DS}
    devinner = {ds: C8.load_split(ds)[1] for ds in DS}
    base = C10.base_head(c9tr)
    oof = np.load(f"{OUT}/_c9_oof.npz"); oof_tr = oof["oof"]; s_dev = oof["s_dev"]; s_di = oof["s_di"]
    log(f"[{time.time()-T0:.0f}s] base head + oof loaded")
    B_tr = combined("train", c9tr, base); log(f"[{time.time()-T0:.0f}s] B_tr rows={len(B_tr['y'])}")
    B_dev = combined("train", c9dev, base); B_di = combined("train", devinner, base)
    X_tr = build_X(B_tr, oof_tr); X_dev = build_X(B_dev, s_dev); X_di = build_X(B_di, s_di)
    log(f"[{time.time()-T0:.0f}s] bundles+X built (F={X_tr.shape[1]}+{len(C10.DYNN)}dyn)")

    Q = len(B_tr["groups"]); fold = np.array([int(C8._h(m["qi"] * 31 + (0 if m["ds"] == DS[0] else 7)) * 3) % 3 for m in B_tr["meta"]])
    allq = np.arange(Q)

    # ---- teacher rows (all train queries) ----
    Xt, Yt = gen_rows(B_tr["meta"], B_tr["groups"], X_tr, oof_tr, allq, teacher=True)
    tfold = _row_folds(B_tr["meta"], B_tr["groups"], X_tr, oof_tr, allq, fold)   # fold per teacher-row-query
    log(f"[{time.time()-T0:.0f}s] teacher rows={len(Yt)} pos={int(Yt.sum())}")

    # ---- OOF prelim models -> model-rollout rows ----
    Xr_all = []; Yr_all = []
    for f in range(3):
        prelim = fit_clf(Xt[tfold != f], Yt[tfold != f])
        qf = allq[fold == f]
        Xr, Yr = gen_rows(B_tr["meta"], B_tr["groups"], X_tr, oof_tr, qf, teacher=False,
                          predict=lambda ff: prelim.predict_proba(ff)[:, 1])
        Xr_all.append(Xr); Yr_all.append(Yr)
        log(f"[{time.time()-T0:.0f}s] fold {f} rollout rows={len(Yr)} pos={int(Yr.sum())}")
    Xr_all = np.concatenate(Xr_all); Yr_all = np.concatenate(Yr_all)

    # ---- final selector on teacher + rollout mixture ----
    Xf = np.concatenate([Xt, Xr_all]); Yf = np.concatenate([Yt, Yr_all])
    clf = fit_clf(Xf, Yf)
    log(f"[{time.time()-T0:.0f}s] final selector trained on {len(Yf)} rows (pos {int(Yf.sum())})")

    # ---- inference ----
    c8c_dev = C9.eval_ranking(B_dev, s_dev); c8c_di = C9.eval_ranking(B_di, s_di)
    c10b_dev, orders_dev = apply_selector(B_dev, X_dev, s_dev, clf)
    log(f"[{time.time()-T0:.0f}s] C10b DEV ndcg5={c10b_dev['ndcg5']:.4f} recall5={c10b_dev['recall5_macro']:.4f} all50={c10b_dev['all50']:.4f}")
    c10b_di, orders_di = apply_selector(B_di, X_di, s_di, clf)

    # diagnostics vs C8c on C9_DEV
    resc = rescue(c8c_dev["_goldranks"], c10b_dev["_goldranks"])
    resc_mg = rescue_by_multigold(B_dev, c8c_dev["_goldranks"], c10b_dev["_goldranks"])
    mg = {"C8c": multigold(c8c_dev), "C10b": multigold(c10b_dev)}
    sg = {"C8c": singlegold(c8c_dev), "C10b": singlegold(c10b_dev)}
    boot = {mk: paired_boot(c10b_dev["_perq"][mk], c8c_dev["_perq"][mk]) for mk in ("ndcg5", "recall5", "ndcg50", "all50")}
    QC_dev = C10.precompute(B_dev, s_dev)
    redun = {"C8c_top5": redundancy(QC_dev, None, "c8c"), "C10b_top5": redundancy(QC_dev, None, "model", orders_dev)}
    fi = clf.feature_importances_
    imp = dict(sorted({FEATN[i]: round(float(fi[i]), 4) for i in range(len(FEATN))}.items(), key=lambda kv: -kv[1])[:18])
    dyn_imp = {n: round(float(fi[FEATN.index(n)]), 4) for n in C10.DYNN}

    res = {"phase": "C10b — learned greedy set selector",
           "n_features": len(FEATN), "cap": CAP, "npick": NPICK,
           "train_rows": {"teacher": int(len(Yt)), "rollout": int(len(Yr_all)), "total": int(len(Yf))},
           "C9_DEV": {"C8c": C8slim(c8c_dev), "C10b": C8slim(c10b_dev)},
           "bootstrap_C10b_vs_C8c": boot,
           "GOLD_RESCUE": resc, "GOLD_RESCUE_BY_MULTIGOLD": resc_mg, "MULTIGOLD": mg, "SINGLE_GOLD": sg,
           "REDUNDANCY_ANALYSIS": redun,
           "feature_importance_top": imp, "dynamic_feature_importance": dyn_imp,
           "DEV_INNER_CONFIRM": {"C8c": C8slim(c8c_di), "C10b": C8slim(c10b_di)}}
    json.dump(res, open("results/L2/_ctrl/_c10b.json", "w"), indent=1, default=str)
    import joblib; joblib.dump(clf, f"{OUT}/C10b_selector.joblib")
    log(f"[{time.time()-T0:.0f}s] C10B_DONE")
    print(json.dumps({"C9_DEV": res["C9_DEV"], "boot": boot, "rescue": resc, "rescue_mg": resc_mg,
                      "multigold": mg, "single_gold": sg, "redundancy": redun, "dyn_imp": dyn_imp,
                      "top_imp": imp, "dev_inner": res["DEV_INNER_CONFIRM"]}, indent=1, default=str))


def _row_folds(meta, groups, X, scores, qidx, fold):
    """fold id repeated per teacher row, mirroring gen_rows(teacher=True) row emission."""
    st = starts_of(groups); out = []
    for i in qidx:
        m = meta[i]; a = st[i]; g = groups[i]; sc = scores[a:a + g]
        order = np.argsort(-sc, kind="stable"); win = order[:min(CAP, g)]
        goldset = set(m["gold"].tolist()); gold_win = [int(w) for w in win if int(m["pool"][w]) in goldset]
        if not gold_win: continue
        gseq = sorted(gold_win, key=lambda w: -sc[w]); nsteps = min(len(gseq), NPICK)
        for t in range(nsteps):
            ncand = len([w for w in win if w not in set(gseq[:t])]); out.append(np.full(ncand, fold[i], np.int8))
    return np.concatenate(out) if out else np.zeros(0, np.int8)


def rescue_by_multigold(B, base_gr, new_gr):
    """Net top-5 gold gain split by query gold-count bucket (1 / 2 / 3+)."""
    ng_by_q = {(m["qi"]): len(m["gold"]) for m in B["meta"]}
    out = {"1": [0, 0], "2": [0, 0], "3+": [0, 0]}   # [rescued, hurt]
    for key, ra in base_gr.items():
        qi = key[0]; ng = ng_by_q.get(qi, 0); b = "1" if ng == 1 else "2" if ng == 2 else "3+"
        rb = new_gr.get(key)
        if rb is None: continue
        if rb < 5 and ra >= 5: out[b][0] += 1
        elif ra < 5 and rb >= 5: out[b][1] += 1
    return {k: {"rescued": v[0], "hurt": v[1], "net": v[0] - v[1]} for k, v in out.items()}


if __name__ == "__main__":
    main()
