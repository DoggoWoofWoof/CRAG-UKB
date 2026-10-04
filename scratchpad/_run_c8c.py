"""C8c driver — top-50 residual reranker. Train on TRAIN_INNER (base=C7b_inner), early-stop on DEV_INNER,
report dev deltas + gold rescue. If it beats C7b on dev NDCG@5 with guardrails intact -> ONE official-VAL
milestone (base=C7b full-train): C0 / C7b / C8c, paired bootstrap C8c vs C7b. TEST untouched. Writes _c8c.json."""
import sys, os, json
sys.path.insert(0, "scratchpad")
import numpy as np, l2_c8 as C8, l2_c8c as C8C
OUT = C8.OUT; log = lambda *a: print(*a, flush=True)
GUARD = {"ndcg50": 0.005, "mrr": 0.010, "all50": 0.010}


def qmeta(ds, split, qi_list):
    M = np.load(f"{C8.CORP}/{ds}/{split}/expert_meta.npz")
    ra = M["relation_any_signal_present"].astype(np.float32); di = M["dense_splade_disagreement"].astype(np.float32)
    return ra[qi_list], di[qi_list]


def slim(p, keys=C8.AGG):
    return {k: (round(p[k], 4) if isinstance(p.get(k), float) else p.get(k)) for k in keys}


def paired_boot(a, b, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); dif = a - b
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(nboot)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(dif.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "frac_boot>0": round(float((bs > 0).mean()), 3), "significant": bool(lo > 0 or hi < 0)}


def rescue(base_ranks, new_ranks):
    src = ["5-9", "10-19", "20-49", "50-99", "100-499", "500+"]; resc = {b: 0 for b in src}; hurt = 0
    for key, ra in base_ranks.items():
        rb = new_ranks.get(key)
        if rb is None: continue
        if rb < 5 and ra >= 5: resc[C8.RBNAMES[C8.rankbin(int(ra))]] = resc.get(C8.RBNAMES[C8.rankbin(int(ra))], 0) + 1
        elif ra < 5 and rb >= 5: hurt += 1
    return {"rescued_into_top5_by_source_bin": {b: resc.get(b, 0) for b in src},
            "TOP5_GOLD_RESCUE": sum(resc.values()), "TOP5_GOLD_HURT": hurt, "NET_TOP5_GOLD_GAIN": sum(resc.values()) - hurt}


def build_split(inner, dev, base_head):
    Xtr = []; Ytr = []; Gtr = []; Xdv = []; Ydv = []; Gdv = []
    for ds in C8.DS:
        raI, diI = qmeta(ds, "train", inner[ds]); wI = C8C.base_weight_rows(base_head, ds, "train", inner[ds])
        x, y, g, _ = C8C.build_pool(ds, "train", inner[ds], wI, raI, diI); Xtr.append(x); Ytr.append(y); Gtr.append(g)
        raD, diD = qmeta(ds, "train", dev[ds]); wD = C8C.base_weight_rows(base_head, ds, "train", dev[ds])
        x, y, g, _ = C8C.build_pool(ds, "train", dev[ds], wD, raD, diD); Xdv.append(x); Ydv.append(y); Gdv.append(g)
    return (np.concatenate(Xtr), np.concatenate(Ytr), np.concatenate(Gtr),
            np.concatenate(Xdv), np.concatenate(Ydv), np.concatenate(Gdv))


def main():
    inner = {ds: C8.load_split(ds)[0] for ds in C8.DS}; dev = {ds: C8.load_split(ds)[1] for ds in C8.DS}
    base_inner = C8.fit_soft_head("ndcg50", inner)   # C7b_inner base fusion

    # ---- train reranker on inner, early-stop on dev ----
    Xtr, Ytr, Gtr, Xdv, Ydv, Gdv = build_split(inner, dev, base_inner)
    log(f"pool rows: train={len(Xtr)} groups={len(Gtr)} | dev={len(Xdv)} groups={len(Gdv)} | feat={Xtr.shape[1]}")
    model = C8C.train_ranker(Xtr, Ytr, Gtr, Xdv, Ydv, Gdv, log=log)

    # ---- dev evaluation: C7b base vs C8c ----
    rbB = {}; rbC = {}; rk_base = {}; rk_new = {}
    for ds in C8.DS:
        qi = dev[ds]; wB = C8C.base_weight_rows(base_inner, ds, "train", qi)
        rbB[ds] = C8.eval_policy(ds, "train", qi, wB, want_perq=True, want_ranks=True)
        rk_base[ds] = {(q, gl): rk for q, gl, rk in rbB[ds]["_goldranks"]}
        ra, di = qmeta(ds, "train", qi)
        rbC[ds] = C8C.rerank_eval(ds, "train", qi, wB, model, ra, di, want_ranks=True)
        rk_new[ds] = rbC[ds]["_goldranks"]
    pB = C8.pool(rbB); pC = C8.pool(rbC)
    log("DEV C7b : " + json.dumps(slim(pB)))
    log("DEV C8c : " + json.dumps(slim(pC)))
    harm = {g: round(pB[g] - pC[g], 4) for g in GUARD if (pB[g] - pC[g]) > GUARD[g]}
    dev_beats = (pC["ndcg5"] > pB["ndcg5"]) and (not harm)
    fi = {C8C.FEATNAMES[i]: round(float(v), 4) for i, v in enumerate(model.feature_importances_)}
    res = {"base_fusion": "C7b soft-archetype (U50)", "reranker": "XGBRanker rank:ndcg, top-50 pool",
           "best_iteration": int(model.best_iteration), "dev_best_ndcg@5_xgb": round(float(model.best_score), 4),
           "feature_importance": dict(sorted(fi.items(), key=lambda kv: -kv[1])),
           "guard_thresholds": GUARD,
           "DEV": {"C7b": {"pooled": slim(pB), "per_ds": {ds: slim(rbB[ds]) for ds in C8.DS}},
                   "C8c": {"pooled": slim(pC), "per_ds": {ds: slim(rbC[ds]) for ds in C8.DS}},
                   "delta_C8c_minus_C7b": {k: round(pC[k] - pB[k], 4) for k in C8.AGG if isinstance(pC[k], float)},
                   "guardrail_harm": harm, "C8c_beats_C7b_on_dev": bool(dev_beats)},
           "TOP5_GOLD_RESCUE_dev": {ds: rescue(rk_base[ds], rk_new[ds]) for ds in C8.DS}}
    res["TOP5_GOLD_RESCUE_dev"]["pooled"] = {
        m: sum(res["TOP5_GOLD_RESCUE_dev"][ds][m] for ds in C8.DS) for m in ("TOP5_GOLD_RESCUE", "TOP5_GOLD_HURT", "NET_TOP5_GOLD_GAIN")}
    log("DEV rescue pooled: " + json.dumps(res["TOP5_GOLD_RESCUE_dev"]["pooled"]))

    # ---- official VAL milestone (only if it earned it on dev) ----
    if dev_beats:
        full = {ds: C8.precompute_arch(ds, "train")["qi"] for ds in C8.DS}
        base_full = C8.fit_soft_head("ndcg50", full)
        XtrF, YtrF, GtrF, XdvF, YdvF, GdvF = build_split(full, dev, base_full)   # dev only for early stopping
        modelF = C8C.train_ranker(XtrF, YtrF, GtrF, XdvF, YdvF, GdvF, log=log)
        rb0 = {}; rvB = {}; rvC = {}
        for ds in C8.DS:
            qi = C8.precompute_arch(ds, "val")["qi"]
            rb0[ds] = C8.eval_policy(ds, "val", qi, C8.equal_weights(len(qi)), want_perq=True)
            wB = C8C.base_weight_rows(base_full, ds, "val", qi); rvB[ds] = C8.eval_policy(ds, "val", qi, wB, want_perq=True)
            ra, di = qmeta(ds, "val", qi); rvC[ds] = C8C.rerank_eval(ds, "val", qi, wB, modelF, ra, di)
        p0 = C8.pool(rb0); pvB = C8.pool(rvB); pvC = C8.pool(rvC)
        def cat(rbx, key): return np.concatenate([rbx[ds]["_perq"][key] for ds in C8.DS])
        boot = {m: paired_boot(cat(rvC, mk), cat(rvB, mk)) for m, mk in
                (("ndcg5", "ndcg5"), ("recall5_macro", "recall5"), ("ndcg50", "ndcg50"), ("all50", "all50"))}
        res["OFFICIAL_VAL"] = {"C0": {"pooled": slim(p0), "per_ds": {ds: slim(rb0[ds]) for ds in C8.DS}},
                               "C7b": {"pooled": slim(pvB), "per_ds": {ds: slim(rvB[ds]) for ds in C8.DS}},
                               "C8c": {"pooled": slim(pvC), "per_ds": {ds: slim(rvC[ds]) for ds in C8.DS}},
                               "delta_C8c_minus_C7b": {k: round(pvC[k] - pvB[k], 4) for k in C8.AGG if isinstance(pvC[k], float)},
                               "delta_C8c_minus_C0": {k: round(pvC[k] - p0[k], 4) for k in C8.AGG if isinstance(pvC[k], float)}}
        res["bootstrap_C8c_vs_C7b_VAL"] = boot
        log("VAL C0 : " + json.dumps(slim(p0))); log("VAL C7b: " + json.dumps(slim(pvB))); log("VAL C8c: " + json.dumps(slim(pvC)))
        log("VAL BOOT " + json.dumps(boot))
        import joblib; joblib.dump(modelF, f"{OUT}/C8c_reranker_fulltrain.joblib")
    else:
        res["OFFICIAL_VAL"] = "SKIPPED — C8c did not beat C7b on DEV_INNER under guardrails"
        log("VAL milestone SKIPPED (dev gate not passed)")

    json.dump(res, open(f"{OUT}/_c8c.json", "w"), indent=1, default=str)
    log("C8C_DONE wrote results/L2/_ctrl/_c8c.json")


if __name__ == "__main__":
    main()
