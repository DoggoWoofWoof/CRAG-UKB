"""Paired VAL bootstrap + failure analysis over saved controller per-query arrays. VAL only."""
import sys, os, json
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_controller as CT

OUT = "results/L2/_ctrl"; SHAP = "results/L2/_shapley"; CORP = "data/l2_corpus"
DS = CT.DS; EXPERTS = CT.EXPERTS


def load_perq(tag):
    z = np.load(f"{OUT}/_perq_{tag}.npz")
    # pooled arrays in fixed (ds, qi) order
    out = {}
    for k in ("ndcg", "mrr", "all10", "all50", "best", "relw", "qi"):
        out[k] = np.concatenate([z[f"{ds}__{k}"] for ds in DS])
    out["ds"] = np.concatenate([np.full(len(z[f"{ds}__ndcg"]), di) for di, ds in enumerate(DS)])
    out["meanw"] = np.concatenate([z[f"{ds}__meanw"] for ds in DS])  # (nq,5)
    return out


def paired_boot(a, b, nboot=2000, seed=0):
    """mean(a)-mean(b) with 95% CI via paired query resampling."""
    rng = np.random.default_rng(seed); n = len(a); idx = np.arange(n)
    d = a - b; obs = float(d.mean())
    boots = np.array([d[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_gt0 = float((boots > 0).mean())
    return {"delta": round(obs, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "frac_boot>0": round(p_gt0, 3), "significant": bool(lo > 0 or hi < 0)}


def bootstrap_table():
    P = {t: load_perq(t) for t in ("C0", "C1", "C2", "C3", "C4")}
    pairs = [("C0", "C1"), ("C0", "C2"), ("C2", "C3"), ("C3", "C4"), ("C0", "C3"), ("C0", "C4")]
    out = {}
    for hi, lo in pairs:
        out[f"{hi}_vs_{lo}"] = {m: paired_boot(P[hi][m], P[lo][m]) for m in ("ndcg", "all10", "all50", "mrr")}
    return out, P


def shapley_val(ds):
    z = np.load(f"{SHAP}/{ds}_val.npz")
    phi = {e: z[f"phi_ndcg50_{e}"] for e in EXPERTS}
    return phi, z["relation_gold_present"].astype(bool), z["relation_any_signal_present"].astype(bool), z["gold_count"], z["dense_splade_disagreement"]


def failure_analysis(P, best_tag="C3", nshow=12):
    """Queries where best model NDCG < C0 NDCG. Join Shapley/meta for diagnosis (analysis-only labels)."""
    rows = []
    # per-dataset to index shapley/meta by local qi
    off = {ds: len(P["C0"]["ndcg"]) for ds in DS}
    # rebuild per-ds slices
    counts = {}
    for ds in DS:
        z = np.load(f"{OUT}/_perq_C0.npz"); counts[ds] = len(z[f"{ds}__ndcg"])
    idx0 = 0; agg = {"n_worse": 0, "n_total": 0}
    buckets = {"relation_misuse": 0, "multigold_tradeoff": 0, "topk_shuffle": 0, "other": 0}
    worse_relgoldpresent = []; worse_disagree = []; worse_goldcount = []
    for di, ds in enumerate(DS):
        n = counts[ds]; sl = slice(idx0, idx0 + n); idx0 += n
        phi, relgold, relany, goldc, disag = shapley_val(ds)
        qis = P[best_tag]["qi"][sl]
        nd_best = P[best_tag]["ndcg"][sl]; nd_c0 = P["C0"]["ndcg"][sl]
        relw_best = P[best_tag]["relw"][sl]; relw_c0 = P["C0"]["relw"][sl]
        worse = nd_best < nd_c0 - 1e-6
        agg["n_worse"] += int(worse.sum()); agg["n_total"] += n
        for j in np.where(worse)[0]:
            qi = int(qis[j])
            rgp = bool(relgold[qi]); gc = int(goldc[qi]); dg = float(disag[qi])
            worse_relgoldpresent.append(rgp); worse_disagree.append(dg); worse_goldcount.append(gc)
            # crude cause attribution
            if (not rgp) and P[best_tag]["meanw"][sl][j][4] > 0.25:
                buckets["relation_misuse"] += 1
            elif gc >= 2:
                buckets["multigold_tradeoff"] += 1
            elif abs(nd_best[j] - nd_c0[j]) < 0.15:
                buckets["topk_shuffle"] += 1
            else:
                buckets["other"] += 1
    agg["frac_worse"] = round(agg["n_worse"] / agg["n_total"], 4)
    agg["cause_buckets"] = buckets
    agg["worse_pop"] = {"frac_relation_gold_present": round(float(np.mean(worse_relgoldpresent)), 3),
                        "mean_disagreement": round(float(np.mean(worse_disagree)), 3),
                        "mean_gold_count": round(float(np.mean(worse_goldcount)), 2)}
    return agg


def relation_learned_check(best_tag="C3"):
    """Does the gate raise relation weight where relation is actually useful (analysis-only: relation_gold_present)?"""
    out = {}
    P = load_perq(best_tag)
    idx0 = 0; counts = {}
    z0 = np.load(f"{OUT}/_perq_{best_tag}.npz")
    for ds in DS: counts[ds] = len(z0[f"{ds}__ndcg"])
    relw_gp = {True: [], False: []}; relw_sig = {True: [], False: []}
    for ds in DS:
        n = counts[ds]; sl = slice(idx0, idx0 + n); idx0 += n
        phi, relgold, relany, goldc, disag = shapley_val(ds)
        qis = P["qi"][sl]; relw = P["relw"][sl]
        for j in range(n):
            qi = int(qis[j])
            relw_gp[bool(relgold[qi])].append(float(relw[j]))
            relw_sig[bool(relany[qi])].append(float(relw[j]))
    return {"mean_relation_weight_by_relation_gold_present":
            {str(k): round(float(np.mean(v)), 4) for k, v in relw_gp.items()},
            "mean_relation_weight_by_any_signal":
            {str(k): round(float(np.mean(v)), 4) for k, v in relw_sig.items()}}


if __name__ == "__main__":
    boot, P = bootstrap_table()
    best = sys.argv[1] if len(sys.argv) > 1 else "C3"
    fail = failure_analysis(P, best)
    rel = relation_learned_check(best)
    res = {"bootstrap": boot, "failure_analysis_best=" + best: fail, "relation_learned_best=" + best: rel}
    json.dump(res, open(f"{OUT}/_analysis.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))
