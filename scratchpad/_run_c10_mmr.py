"""C10 Task 3/4/10/11/12: deterministic set-aware baselines (MMR C10a, support-diversity C10a2) over C8c top20.
Select lambda on C9_DEV only from a tiny predeclared set. Report redundancy analysis, gold rescue, single-gold
preservation. Reorders only within top50 -> ALL@50 preserved."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_c8 as C8, l2_c9 as C9, l2_c10 as C10
from _run_c9 import combined, multigold, rescue, paired_boot
OUT = C8.OUT; DS = C8.DS; log = lambda *a: print(*a, flush=True); T0 = time.time()


def evalcfg(B, QC, lam_sem, lam_exp, cap=20):
    orderings = [C10.greedy_order(qc, lam_sem, lam_exp, cap) for qc in QC]
    sc = C10.order_to_scores(QC, orderings)
    return C9.eval_ranking(B, sc)


def redundancy(QC, res_goldranks_unused, which="c8c", model_order=None):
    """Task 10: mean/max pairwise Qwen sim + support sim + unique-strongest-expert count over the final top5."""
    mp = []; xp = []; sp = []; ue = []
    for i, qc in enumerate(QC):
        if which == "c8c":
            order = np.argsort(-qc["c8c"], kind="stable")
        else:
            order = model_order[i]
        top5 = order[:min(5, qc["g"])]
        V = qc["V"][top5]; Cn = qc["Cn"][top5]
        if len(top5) >= 2:
            s = V @ V.T; iu = np.triu_indices(len(top5), 1); mp.append(float(s[iu].mean())); xp.append(float(s[iu].max()))
            ss = Cn @ Cn.T; sp.append(float(ss[iu].mean()))
        am = np.argmax(qc["Cn"][top5], axis=1); ue.append(len(set(am.tolist())))
    return {"mean_pairwise_qwen_sim": round(float(np.mean(mp)), 4), "max_pairwise_qwen_sim": round(float(np.mean(xp)), 4),
            "mean_pairwise_support_sim": round(float(np.mean(sp)), 4), "unique_strongest_expert_count": round(float(np.mean(ue)), 4)}


def singlegold(res):
    pq = res["_perq"]; ng = pq["ng"]; mask = ng == 1
    if mask.sum() == 0: return None
    return {"n": int(mask.sum()), "mrr": round(float(pq["mrr"][mask].mean()), 4),
            "ndcg5": round(float(pq["ndcg5"][mask].mean()), 4), "any5": round(float(pq["any5"][mask].mean()), 4)}


def main():
    c9tr = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9train"] for ds in DS}
    c9dev = {ds: np.load(f"{OUT}/_c9_split_{ds}.npz")["c9dev"] for ds in DS}
    devinner = {ds: C8.load_split(ds)[1] for ds in DS}
    base = C10.base_head(c9tr)
    oof = np.load(f"{OUT}/_c9_oof.npz"); s_dev = oof["s_dev"]; s_di = oof["s_di"]
    B_dev = combined("train", c9dev, base); B_di = combined("train", devinner, base)
    log(f"[{time.time()-T0:.0f}s] bundles rebuilt")
    QC_dev = C10.precompute(B_dev, s_dev); QC_di = C10.precompute(B_di, s_di)
    log(f"[{time.time()-T0:.0f}s] precompute done (dev {len(QC_dev)} di {len(QC_di)} queries)")

    # C8c baseline (lam=0 reproduces it — sanity check)
    c8c = C9.eval_ranking(B_dev, s_dev)
    base0 = evalcfg(B_dev, QC_dev, 0.0, 0.0)
    log(f"[{time.time()-T0:.0f}s] sanity lam0 ndcg5={base0['ndcg5']:.4f} vs c8c {c8c['ndcg5']:.4f} (should match)")

    # ---- Task 3: C10a MMR over top20, semantic novelty only; tiny predeclared lambda set ----
    grid = [0.0, 0.05, 0.10]; c10a = {}
    for lam in grid:
        r = evalcfg(B_dev, QC_dev, lam, 0.0)
        c10a[f"lam{lam}"] = {"ndcg5": round(r["ndcg5"], 4), "recall5": round(r["recall5_macro"], 4),
                             "all5_feas": round(r["all5_feas"], 4), "all50": round(r["all50"], 4),
                             "mrr": round(r["mrr"], 4), "_r": r}
        log(f"[{time.time()-T0:.0f}s] C10a lam={lam}: ndcg5={r['ndcg5']:.4f} recall5={r['recall5_macro']:.4f} all50={r['all50']:.4f}")
    best_lam = max(grid, key=lambda l: (c10a[f"lam{l}"]["ndcg5"], c10a[f"lam{l}"]["recall5"]))

    # ---- Task 4: C10a2 support+semantic diversity; ONE tiny predeclared setting ----
    r2 = evalcfg(B_dev, QC_dev, 0.05, 0.05)
    c10a2 = {"lam_sem0.05_lam_exp0.05": {"ndcg5": round(r2["ndcg5"], 4), "recall5": round(r2["recall5_macro"], 4),
             "all5_feas": round(r2["all5_feas"], 4), "all50": round(r2["all50"], 4), "mrr": round(r2["mrr"], 4)}}
    log(f"[{time.time()-T0:.0f}s] C10a2 sem0.05+exp0.05: ndcg5={r2['ndcg5']:.4f} recall5={r2['recall5_macro']:.4f}")

    # choose best deterministic config on C9_DEV among {C8c(lam0), C10a best, C10a2}
    cands = {"C8c": c8c, f"C10a_lam{best_lam}": c10a[f"lam{best_lam}"]["_r"], "C10a2_sem.05_exp.05": r2}
    sel = max(cands, key=lambda k: (cands[k]["ndcg5"], cands[k]["recall5_macro"]))
    sel_r = cands[sel]
    log(f"[{time.time()-T0:.0f}s] SELECTED deterministic = {sel}")

    # diagnostics vs C8c on C9_DEV
    resc = rescue(c8c["_goldranks"], sel_r["_goldranks"])
    mg = {"C8c": multigold(c8c), sel: multigold(sel_r)}
    sg = {"C8c": singlegold(c8c), sel: singlegold(sel_r)}
    # rescue by multigold bucket
    boot = {mk: paired_boot(sel_r["_perq"][mk], c8c["_perq"][mk]) for mk in ("ndcg5", "recall5", "ndcg50", "all50")}
    # redundancy (C8c vs selected)
    sel_orders = [C10.greedy_order(qc, *({"C8c": (0.0, 0.0), f"C10a_lam{best_lam}": (best_lam, 0.0),
                  "C10a2_sem.05_exp.05": (0.05, 0.05)}[sel])) for qc in QC_dev]
    redun = {"C8c_top5": redundancy(QC_dev, None, "c8c"), f"{sel}_top5": redundancy(QC_dev, None, "model", sel_orders)}

    # confirm on DEV_INNER (once)
    cfg_map = {"C8c": (0.0, 0.0), f"C10a_lam{best_lam}": (best_lam, 0.0), "C10a2_sem.05_exp.05": (0.05, 0.05)}
    conf = {"C8c": C9.eval_ranking(B_di, s_di), sel: evalcfg(B_di, QC_di, *cfg_map[sel])}

    res = {"phase": "C10 Task3/4 — deterministic set-aware baselines (MMR / support-diversity)",
           "sanity_lam0_matches_c8c": abs(base0["ndcg5"] - c8c["ndcg5"]) < 1e-6,
           "C10A_MMR_RESULT": {k: {kk: vv for kk, vv in v.items() if kk != "_r"} for k, v in c10a.items()},
           "C10A_best_lambda": best_lam,
           "C10A2_SUPPORT_DIVERSITY_RESULT": c10a2,
           "SELECTED_DETERMINISTIC_ON_C9DEV": sel,
           "C9_DEV_SELECTED_vs_C8c": {"C8c": C8slim(c8c), sel: C8slim(sel_r)},
           "bootstrap_selected_vs_C8c": boot,
           "GOLD_RESCUE": resc, "MULTIGOLD": mg, "SINGLE_GOLD": sg, "REDUNDANCY_ANALYSIS": redun,
           "DEV_INNER_CONFIRM": {k: C8slim(v) for k, v in conf.items()}}
    json.dump(res, open("results/L2/_ctrl/_c10_mmr.json", "w"), indent=1, default=str)
    log(f"[{time.time()-T0:.0f}s] C10_MMR_DONE selected={sel}")
    print(json.dumps({"selected": sel, "C10A": res["C10A_MMR_RESULT"], "C10A2": c10a2,
                      "sel_vs_c8c": res["C9_DEV_SELECTED_vs_C8c"], "boot": boot, "rescue": resc,
                      "multigold": mg, "single_gold": sg, "redundancy": redun,
                      "dev_inner": res["DEV_INNER_CONFIRM"]}, indent=1, default=str))


def C8slim(r):
    return {k: round(r[k], 4) for k in ("ndcg5", "recall5_macro", "any5", "all5_feas", "mrr", "ndcg50", "all10", "all50")}


if __name__ == "__main__":
    main()
