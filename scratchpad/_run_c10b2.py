"""C10b-v2 — RELEVANCE-PRESERVING learned greedy set selector. Fixes the C10b(pointwise) collapse: use a per-step
XGBRanker (rank:ndcg marginal utility) instead of pointwise logloss, AND seed position-0 with C8c's own top pick
so rank-1 relevance is never degraded; the learned set-aware model only decides positions 2-5 among the remaining
C8c top20. Teacher + 3-fold OOF rollout mixture (Task 7). Caches heavy invariants for cheap iteration.
Reorders within top50 only -> ALL@50 preserved. Select on C9_DEV; confirm DEV_INNER. NO VAL/TEST."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, joblib
import l2_c8 as C8, l2_c9 as C9, l2_c10 as C10
from xgboost import XGBRanker
from _run_c9 import combined, starts_of, multigold, rescue, paired_boot
from _run_c10_mmr import redundancy, singlegold, C8slim
from _run_c10b import rescue_by_multigold
OUT = C8.OUT; DS = C8.DS; CAP = 20; NPICK = 5; SEED0 = True
CACHE = f"{OUT}/_c10b_cache.joblib"; log = lambda *a: print(*a, flush=True); T0 = time.time()
FEATN = C9.C9_FEATNAMES + C10.DYNN


def build_cache():
    c9tr = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9train"] for ds in DS}
    c9dev = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9dev"] for ds in DS}
    devinner = {ds: C8.load_split(ds)[1] for ds in DS}
    base = C10.base_head(c9tr)
    oof = np.load(f"{OUT}/_c9_oof.npz"); oof_tr = oof["oof"]; s_dev = oof["s_dev"]; s_di = oof["s_di"]
    B_tr = combined("train", c9tr, base); log(f"[{time.time()-T0:.0f}s] B_tr rows={len(B_tr['y'])}")
    B_dev = combined("train", c9dev, base); B_di = combined("train", devinner, base)
    X_tr = C9.assemble(B_tr, C9.resid_feats(oof_tr, B_tr["groups"]))
    X_dev = C9.assemble(B_dev, C9.resid_feats(s_dev, B_dev["groups"]))
    X_di = C9.assemble(B_di, C9.resid_feats(s_di, B_di["groups"]))
    cache = {"B_tr": B_tr, "B_dev": B_dev, "B_di": B_di, "X_tr": X_tr, "X_dev": X_dev, "X_di": X_di,
             "oof_tr": oof_tr, "s_dev": s_dev, "s_di": s_di}
    joblib.dump(cache, CACHE); log(f"[{time.time()-T0:.0f}s] cache written"); return cache


def gen_rows(meta, groups, X, scores, qidx, teacher=True, predict=None):
    """Emit per-step candidate groups for greedy set selection. Returns (F rows, Y labels, glen per step-group,
    gfold query-fold per step-group). teacher=oracle gold prefix; else model rollout via predict."""
    st = starts_of(groups); F = []; Y = []; GL = []
    for i in qidx:
        m = meta[i]; a = st[i]; g = groups[i]; Xq = X[a:a + g]; sc = scores[a:a + g]
        qc = C10.qc_one(m, sc)
        order = np.argsort(-sc, kind="stable"); win = order[:min(CAP, g)]
        goldset = set(m["gold"].tolist()); gold_win = [int(w) for w in win if int(qc["pool"][w]) in goldset]
        if not gold_win: continue
        if teacher:
            gseq = sorted(gold_win, key=lambda w: -sc[w]); nsteps = min(len(gseq), NPICK)
            for t in range(nsteps):
                pref = set(gseq[:t]); cand = np.array([w for w in win if w not in pref], np.int64)
                if int((np.array([int(qc["pool"][c]) in goldset and c not in pref for c in cand])).sum()) == 0: continue
                D = C10.dyn_block(qc, np.array(gseq[:t], np.int64), cand, t, NPICK)
                F.append(np.concatenate([Xq[cand], D], axis=1))
                Y.append(np.array([1 if (int(qc["pool"][c]) in goldset and c not in pref) else 0 for c in cand], np.int8))
                GL.append((len(cand), i))
        else:
            chosen = []; avail = list(win); remaining_gold = set(gold_win)
            for t in range(min(NPICK, len(win))):
                cand = np.array(avail, np.int64)
                D = C10.dyn_block(qc, np.array(chosen, np.int64), cand, t, NPICK)
                feats = np.concatenate([Xq[cand], D], axis=1)
                if remaining_gold and t > 0:   # only rollout steps with a gold still to place carry signal
                    F.append(feats)
                    Y.append(np.array([1 if (int(qc["pool"][c]) in remaining_gold) else 0 for c in cand], np.int8))
                    GL.append((len(cand), i))
                p = predict(feats); pick = int(cand[int(np.argmax(p))]); chosen.append(pick); avail.remove(pick)
                remaining_gold.discard(pick)
                if not remaining_gold: break
    if not F:
        return (np.zeros((0, len(FEATN)), np.float32), np.zeros(0, np.int8), np.zeros(0, np.int64), np.zeros(0, np.int64))
    glen = np.array([x[0] for x in GL], np.int64); gfold = np.array([x[1] for x in GL], np.int64)
    return np.concatenate(F).astype(np.float32), np.concatenate(Y), glen, gfold


def fit_ranker(X, y, glen, seed=0):
    m = XGBRanker(objective="rank:ndcg", eval_metric=["ndcg@3"], lambdarank_pair_method="topk",
                  lambdarank_num_pair_per_sample=8, n_estimators=400, learning_rate=0.05, max_depth=6,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                  random_state=seed)
    m.fit(X, y, group=glen); return m


def rollout_seed(qc, Xq, predict, cap=CAP, npick=NPICK):
    """Greedy selection; position-0 = C8c top pick (SEED0), positions 1..4 by learned marginal utility."""
    g = qc["g"]; sc = qc["c8c"]; order = np.argsort(-sc, kind="stable")
    win = list(order[:min(cap, g)]); rest = list(order[min(cap, g):])
    chosen = []; avail = list(win)
    for t in range(min(npick, len(win))):
        if t == 0 and SEED0:
            pick = int(avail[0])                       # C8c's own top-ranked window candidate
        else:
            cand = np.array(avail, np.int64); D = C10.dyn_block(qc, np.array(chosen, np.int64), cand, t, npick)
            feats = np.concatenate([Xq[cand], D], axis=1); p = predict(feats); pick = int(cand[int(np.argmax(p))])
        chosen.append(pick); avail.remove(pick)
    return np.array(chosen + avail + rest, np.int64)


def apply_sel(B, X, scores, model):
    st = starts_of(B["groups"]); predict = lambda f: model.predict(f); orders = []
    for i, (m, g) in enumerate(zip(B["meta"], B["groups"])):
        a = st[i]; qc = C10.qc_one(m, scores[a:a + g]); orders.append(rollout_seed(qc, X[a:a + g], predict))
    out = []
    for g, order in zip(B["groups"], orders):
        place = np.empty(g, np.int64); place[order] = np.arange(g); out.append(-place.astype(np.float64))
    return C9.eval_ranking(B, np.concatenate(out)), orders


def main():
    cache = joblib.load(CACHE) if os.path.exists(CACHE) else build_cache()
    B_tr, B_dev, B_di = cache["B_tr"], cache["B_dev"], cache["B_di"]
    X_tr, X_dev, X_di = cache["X_tr"], cache["X_dev"], cache["X_di"]
    oof_tr, s_dev, s_di = cache["oof_tr"], cache["s_dev"], cache["s_di"]
    log(f"[{time.time()-T0:.0f}s] cache ready")
    Q = len(B_tr["groups"]); allq = np.arange(Q)
    fold = np.array([int(C8._h(m["qi"] * 31 + (0 if m["ds"] == DS[0] else 7)) * 3) % 3 for m in B_tr["meta"]])

    Xt, Yt, glt, gft = gen_rows(B_tr["meta"], B_tr["groups"], X_tr, oof_tr, allq, teacher=True)
    log(f"[{time.time()-T0:.0f}s] teacher groups={len(glt)} rows={len(Yt)} pos={int(Yt.sum())}")

    # OOF prelim ranker -> rollout groups
    Xr_all = []; Yr_all = []; glr_all = []
    for f in range(3):
        # build row-level train mask from teacher step-group folds (keep groups whose query is not in fold f)
        rows_keep = []; pos = 0
        for gl, qf in zip(glt, gft):
            if qf != f: rows_keep.append(np.arange(pos, pos + gl))
            pos += gl
        rk = np.concatenate(rows_keep); gl_keep = glt[gft != f]
        prelim = fit_ranker(Xt[rk], Yt[rk], gl_keep)
        qf_ids = allq[fold == f]
        Xr, Yr, glr, _ = gen_rows(B_tr["meta"], B_tr["groups"], X_tr, oof_tr, qf_ids, teacher=False,
                                  predict=lambda ff: prelim.predict(ff))
        Xr_all.append(Xr); Yr_all.append(Yr); glr_all.append(glr)
        log(f"[{time.time()-T0:.0f}s] fold {f} rollout groups={len(glr)} rows={len(Yr)} pos={int(Yr.sum())}")
    Xr_all = np.concatenate(Xr_all); Yr_all = np.concatenate(Yr_all); glr_all = np.concatenate(glr_all)

    Xf = np.concatenate([Xt, Xr_all]); Yf = np.concatenate([Yt, Yr_all]); glf = np.concatenate([glt, glr_all])
    model = fit_ranker(Xf, Yf, glf)
    log(f"[{time.time()-T0:.0f}s] final ranker trained groups={len(glf)} rows={len(Yf)}")

    c8c_dev = C9.eval_ranking(B_dev, s_dev); c8c_di = C9.eval_ranking(B_di, s_di)
    c10b_dev, orders_dev = apply_sel(B_dev, X_dev, s_dev, model)
    c10b_di, orders_di = apply_sel(B_di, X_di, s_di, model)
    log(f"[{time.time()-T0:.0f}s] C10b2 DEV ndcg5={c10b_dev['ndcg5']:.4f} recall5={c10b_dev['recall5_macro']:.4f} all50={c10b_dev['all50']:.4f}")

    resc = rescue(c8c_dev["_goldranks"], c10b_dev["_goldranks"])
    resc_mg = rescue_by_multigold(B_dev, c8c_dev["_goldranks"], c10b_dev["_goldranks"])
    mg = {"C8c": multigold(c8c_dev), "C10b2": multigold(c10b_dev)}
    sg = {"C8c": singlegold(c8c_dev), "C10b2": singlegold(c10b_dev)}
    boot = {mk: paired_boot(c10b_dev["_perq"][mk], c8c_dev["_perq"][mk]) for mk in ("ndcg5", "recall5", "ndcg50", "all50")}
    QC_dev = C10.precompute(B_dev, s_dev)
    redun = {"C8c_top5": redundancy(QC_dev, None, "c8c"), "C10b2_top5": redundancy(QC_dev, None, "model", orders_dev)}
    fi = model.feature_importances_
    imp = dict(sorted({FEATN[i]: round(float(fi[i]), 4) for i in range(len(FEATN))}.items(), key=lambda kv: -kv[1])[:18])
    dyn_imp = {n: round(float(fi[FEATN.index(n)]), 4) for n in C10.DYNN}

    res = {"phase": "C10b2 — relevance-preserving learned greedy set selector (per-step NDCG ranker, C8c-seeded pos0)",
           "seed0_c8c": SEED0, "n_features": len(FEATN),
           "train_groups": {"teacher": int(len(glt)), "rollout": int(len(glr_all)), "total": int(len(glf))},
           "C9_DEV": {"C8c": C8slim(c8c_dev), "C10b2": C8slim(c10b_dev)},
           "bootstrap_C10b2_vs_C8c": boot,
           "GOLD_RESCUE": resc, "GOLD_RESCUE_BY_MULTIGOLD": resc_mg, "MULTIGOLD": mg, "SINGLE_GOLD": sg,
           "REDUNDANCY_ANALYSIS": redun, "feature_importance_top": imp, "dynamic_feature_importance": dyn_imp,
           "DEV_INNER_CONFIRM": {"C8c": C8slim(c8c_di), "C10b2": C8slim(c10b_di)}}
    json.dump(res, open("results/L2/_ctrl/_c10b2.json", "w"), indent=1, default=str)
    joblib.dump(model, f"{OUT}/C10b2_selector.joblib")
    log(f"[{time.time()-T0:.0f}s] C10B2_DONE")
    print(json.dumps({"C9_DEV": res["C9_DEV"], "boot": boot, "rescue": resc, "rescue_mg": resc_mg,
                      "multigold": mg, "single_gold": sg, "dyn_imp": dyn_imp, "top_imp": imp,
                      "dev_inner": res["DEV_INNER_CONFIRM"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
