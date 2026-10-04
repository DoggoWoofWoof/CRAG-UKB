"""C10 Task 1/2/5: set-redundancy hypothesis audit (GATE), multi-gold breakdown, perfect-set oracles.
Reconstructs B_dev / B_di from the C9 base head (deterministic). Uses saved C8c scores (_c9_oof.npz).
Decides SET_AWARE_SIGNAL_PRESENT before any C10 training."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_c8 as C8, l2_c9 as C9, l2_c10 as C10
from _run_c9 import combined, starts_of, multigold
OUT = C8.OUT; DS = C8.DS; log = lambda *a: print(*a, flush=True); T0 = time.time()


def audit_split(B, scores, name):
    """For each query: C8c top5 (selected), missed golds (pool rank>=5), top5 FPs (nongold rank<5).
    Compute Qwen sim (max/mean/min) and expert-support cos (max/mean) of each to the selected top5, plus
    argmax-expert overlap. Compare missed-gold novelty vs FP novelty (pooled + query-conditioned)."""
    pos = 0
    G = {k: [] for k in ("mg_qmax", "mg_qmean", "mg_qmin", "mg_smax", "mg_smean", "mg_argov",
                          "fp_qmax", "fp_qmean", "fp_qmin", "fp_smax", "fp_smean", "fp_argov")}
    paired = {"mg_qmax": [], "fp_qmax": [], "mg_smax": [], "fp_smax": []}   # per-query means for queries with both
    n_q_both = 0; n_mg = 0; n_fp = 0
    for m, g in zip(B["meta"], B["groups"]):
        sc = scores[pos:pos + g]; pos += g; ds = m["ds"]; pool = m["pool"]
        rankof = np.empty(g, np.int64); rankof[np.argsort(-sc, kind="stable")] = np.arange(g)
        goldset = set(m["gold"].tolist())
        isg = np.array([int(pool[j]) in goldset for j in range(g)])
        sel_mask = rankof < 5
        V = C10.pool_embs(ds, m["qi"], pool)                     # (g,1536) unit
        # expert contrib vectors for the pool (5-dim)
        R = C6load(ds); s0, e0 = int(R["query_offsets"][m["qi"]]), int(R["query_offsets"][m["qi"] + 1])
        Cfull = C8.contribs_of(R, s0, e0)                        # (5, n)
        Cpool = Cfull[:, pool].T.astype(np.float32)              # (g,5)
        qmax, qmean, qmin = C10._cos_to_set(V, sel_mask)
        smax, smean = C10._supp_cos_to_set(Cpool, sel_mask)
        am = np.argmax(Cpool, axis=1); sel_am = am[sel_mask]
        argov = np.array([int((sel_am == am[j]).sum()) for j in range(g)]) / max(sel_mask.sum(), 1)
        mg = np.where(isg & (rankof >= 5))[0]; fp = np.where((~isg) & (rankof < 5))[0]
        for j in mg:
            G["mg_qmax"].append(qmax[j]); G["mg_qmean"].append(qmean[j]); G["mg_qmin"].append(qmin[j])
            G["mg_smax"].append(smax[j]); G["mg_smean"].append(smean[j]); G["mg_argov"].append(argov[j])
        for j in fp:
            G["fp_qmax"].append(qmax[j]); G["fp_qmean"].append(qmean[j]); G["fp_qmin"].append(qmin[j])
            G["fp_smax"].append(smax[j]); G["fp_smean"].append(smean[j]); G["fp_argov"].append(argov[j])
        n_mg += len(mg); n_fp += len(fp)
        if len(mg) and len(fp):
            n_q_both += 1
            paired["mg_qmax"].append(np.mean(qmax[mg])); paired["fp_qmax"].append(np.mean(qmax[fp]))
            paired["mg_smax"].append(np.mean(smax[mg])); paired["fp_smax"].append(np.mean(smax[fp]))
    # effect sizes (novelty = higher when sim is LOWER; hypothesis: missed golds MORE novel => LOWER qmax)
    d_qmax = C10.cohend(G["mg_qmax"], G["fp_qmax"]); d_smax = C10.cohend(G["mg_smax"], G["fp_smax"])
    # query-conditioned paired sign test on max Qwen sim
    a = np.array(paired["mg_qmax"]); b = np.array(paired["fp_qmax"])
    pd = a - b if len(a) else np.array([])
    sign = {"n_queries_with_both": n_q_both,
            "mean_missedgold_qmax": round(float(a.mean()), 4) if len(a) else None,
            "mean_fp_qmax": round(float(b.mean()), 4) if len(b) else None,
            "mean_paired_diff_mg_minus_fp": round(float(pd.mean()), 4) if len(pd) else None,
            "frac_queries_missedgold_more_novel": round(float((pd < 0).mean()), 4) if len(pd) else None}
    boot = None
    if len(pd):
        rng = np.random.default_rng(0); bs = np.array([pd[rng.integers(0, len(pd), len(pd))].mean() for _ in range(2000)])
        lo, hi = np.percentile(bs, [2.5, 97.5]); sign["paired_diff_ci95"] = [round(float(lo), 4), round(float(hi), 4)]
        sign["paired_diff_significant"] = bool(lo > 0 or hi < 0)
    return {"n_missed_golds": n_mg, "n_top5_false_positives": n_fp,
            "qwen_sim_to_selected_top5": {
                "missed_gold_MAX": C10.dist(G["mg_qmax"]), "fp_MAX": C10.dist(G["fp_qmax"]),
                "missed_gold_MEAN": C10.dist(G["mg_qmean"]), "fp_MEAN": C10.dist(G["fp_qmean"]),
                "missed_gold_MIN": C10.dist(G["mg_qmin"]), "fp_MIN": C10.dist(G["fp_qmin"])},
            "expert_support_cos_to_selected_top5": {
                "missed_gold_MAX": C10.dist(G["mg_smax"]), "fp_MAX": C10.dist(G["fp_smax"]),
                "missed_gold_MEAN": C10.dist(G["mg_smean"]), "fp_MEAN": C10.dist(G["fp_smean"])},
            "argmax_expert_overlap": {"missed_gold": C10.dist(G["mg_argov"]), "fp": C10.dist(G["fp_argov"])},
            "effect_sizes_cohend_mg_minus_fp": {"qwen_MAX_sim": round(d_qmax, 4) if d_qmax is not None else None,
                                                "support_MAX_cos": round(d_smax, 4) if d_smax is not None else None},
            "query_conditioned_paired": sign,
            "_hyp": "missed golds MORE novel than FPs => lower qwen_MAX_sim => NEGATIVE cohend & paired diff"}


_RAW = {}
def C6load(ds):
    if ds not in _RAW:
        import l2_c6 as C6; _RAW[ds] = C6._load_raw(ds, "train")
    return _RAW[ds]


def multigold_metrics(B, scores):
    """Task 2: current NDCG5/GOLD_RECALL5/ALL5 broken out by 1 / 2 / 3+ golds, under C8c ordering."""
    res = C9.eval_ranking(B, scores)
    return multigold(res)


def set_oracle(B, cap):
    """Task 5: perfect-set top-5 oracle over the first `cap` C8c candidates (cap in {20,50}). At each of 5 slots
    place an unselected gold if any remains within the cap window. Report recall5/all5/ndcg5 achievable."""
    # need C8c order to define the top-`cap` window; reuse eval order via base? Use pool order under scores.
    # We compute with the stored C8c scores passed by caller through B["_c8c"].
    scores = B["_c8c"]; pos = 0; rec = []; all5 = []; ndcg5 = []
    for m, g in zip(B["meta"], B["groups"]):
        sc = scores[pos:pos + g]; pos += g; ng = len(m["gold"])
        if ng == 0: continue
        order = np.argsort(-sc, kind="stable")[:min(cap, g)]        # top-cap pool candidates by C8c
        pool = m["pool"]; goldset = set(m["gold"].tolist())
        golds_in_window = [i for i in order if int(pool[i]) in goldset]
        placed = min(len(golds_in_window), 5)
        rec.append(placed / ng)
        if ng <= 5: all5.append(1.0 if placed == ng else 0.0)
        # ndcg5: golds placed at ranks 0..placed-1 ideally
        dcg = sum(1.0 / np.log2(r + 2) for r in range(placed)); idcg = sum(1.0 / np.log2(i + 2) for i in range(min(ng, 5)))
        ndcg5.append(dcg / idcg if idcg > 0 else 0.0)
    return {"recall5": round(float(np.mean(rec)), 4), "all5_feasible": round(float(np.mean(all5)), 4),
            "ndcg5": round(float(np.mean(ndcg5)), 4)}


def main():
    c9tr = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9train"] for ds in DS}
    c9dev = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9dev"] for ds in DS}
    devinner = {ds: C8.load_split(ds)[1] for ds in DS}
    base = C10.base_head(c9tr)
    log(f"[{time.time()-T0:.0f}s] base head refit")
    oof = np.load(f"{OUT}/_c9_oof.npz"); s_dev = oof["s_dev"]; s_di = oof["s_di"]
    B_dev = combined("train", c9dev, base); B_di = combined("train", devinner, base)
    B_dev["_c8c"] = s_dev; B_di["_c8c"] = s_di
    log(f"[{time.time()-T0:.0f}s] bundles rebuilt: dev={len(B_dev['y'])} di={len(B_di['y'])}")

    aud_dev = audit_split(B_dev, s_dev, "C9_DEV")
    log(f"[{time.time()-T0:.0f}s] audit C9_DEV done")
    aud_di = audit_split(B_di, s_di, "DEV_INNER")
    log(f"[{time.time()-T0:.0f}s] audit DEV_INNER done")

    mg_dev = multigold_metrics(B_dev, s_dev)
    oracles = {"C8c_top50_recall5": None,
               "top20_set_oracle": set_oracle(B_dev, 20), "top50_set_oracle": set_oracle(B_dev, 50)}

    # decide gate
    sg = aud_dev["query_conditioned_paired"]
    d = aud_dev["effect_sizes_cohend_mg_minus_fp"]["qwen_MAX_sim"]
    signal = bool((sg.get("paired_diff_significant") and sg.get("mean_paired_diff_mg_minus_fp", 0) < 0)
                  or (d is not None and d <= -0.15))
    res = {"phase": "C10 Task1/2/5 — set-redundancy audit (GATE), multigold breakdown, set oracles",
           "SET_REDUNDANCY_AUDIT": {"C9_DEV": aud_dev, "DEV_INNER": aud_di},
           "MULTIGOLD_BREAKDOWN_C8c_C9DEV": mg_dev,
           "SET_ORACLE_C9DEV": oracles,
           "GATE_SET_AWARE_SIGNAL_PRESENT": "YES" if signal else "NO",
           "gate_reasoning": ("missed golds significantly MORE novel (lower max Qwen sim) than top5 FPs on the "
                              "query-conditioned paired test and/or |cohend|>=0.15" )}
    json.dump(res, open("results/L2/_ctrl/_c10_audit.json", "w"), indent=1, default=str)
    log(f"[{time.time()-T0:.0f}s] C10_AUDIT_DONE gate={res['GATE_SET_AWARE_SIGNAL_PRESENT']}")
    print(json.dumps({"gate": res["GATE_SET_AWARE_SIGNAL_PRESENT"],
                      "paired": sg, "cohend": aud_dev["effect_sizes_cohend_mg_minus_fp"],
                      "mg_qmax": aud_dev["qwen_sim_to_selected_top5"]["missed_gold_MAX"],
                      "fp_qmax": aud_dev["qwen_sim_to_selected_top5"]["fp_MAX"],
                      "oracles": oracles, "multigold": mg_dev}, indent=1))


if __name__ == "__main__":
    main()
