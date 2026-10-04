"""C7 Track C — relation residual gating. BASE4 frozen equal RRF; learn only alpha multiplier on relation.
C7c0 parity(alpha=1)==C0; C7c1 alpha(q); C7c2 alpha(q,d). Full-P50 true-rank ΔNDCG@50 objective.
Saves metrics + per-query arrays + candidate-level alpha analysis + bootstrap (vs C0 and c2 vs c1). VAL only."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT, l2_c7 as C7
from _run_c7ab import slim, save_perq, concat, paired_boot, COLS

OUT = "results/L2/_ctrl"; log = lambda *a: print(*a, flush=True)


def alpha_stats(res_alpha):
    out = {}
    for ds in CT.DS:
        A = res_alpha[ds]
        def st(v):
            v = np.array(v)
            return None if len(v) == 0 else {"n": int(len(v)), "mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                                             "p10": round(float(np.percentile(v, 10)), 4), "p90": round(float(np.percentile(v, 90)), 4)}
        g = np.array(A["gold"]); ng = np.array(A["nongold"])
        sep = None
        if len(g) > 1 and len(ng) > 1:
            pooled_sd = np.sqrt((g.var() + ng.var()) / 2) + 1e-9; sep = round(float((g.mean() - ng.mean()) / pooled_sd), 4)  # Cohen's d
        out[ds] = {"all_eligible": st(A["all"]), "eligible_gold": st(A["gold"]), "eligible_nongold": st(A["nongold"]),
                   "gold_minus_nongold_cohend": sep}
    return out


def relrank_breakdown(res_alpha):
    out = {}
    for ds in CT.DS:
        A = res_alpha[ds]; al = np.array(A["all"]); rr = np.array(A["relrank"])
        if len(al) == 0: out[ds] = None; continue
        out[ds] = {"relrank0": round(float(al[rr == 0].mean()), 4) if (rr == 0).any() else None,
                   "relrank1": round(float(al[rr == 1].mean()), 4) if (rr == 1).any() else None,
                   "relrank2plus": round(float(al[rr >= 2].mean()), 4) if (rr >= 2).any() else None}
    return out


def main():
    va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
    out = {"parity": {}}
    # C7c0 parity
    r0 = C7.eval_generic(va, C7.relation_score_fn(None, "c0")); out["C7c0"] = slim(r0)
    out["parity"]["C7c0_eq_C0_ndcg"] = bool(abs(r0["POOLED"]["NDCG@50"] - 0.8400) < 1e-4)
    log(f"== C7c0 parity NDCG50={r0['POOLED']['NDCG@50']:.4f} (==C0: {out['parity']['C7c0_eq_C0_ndcg']})")
    boot = {}
    for kind, tag in (("c1", "C7c1"), ("c2", "C7c2")):
        model, meta, tb = C7.train_alpha(kind, epochs=8, lr=1e-3, balance_relsig=True, log=log)
        res_sel = C7.eval_generic(va, C7.relation_score_fn(model, kind)); save_perq(tag, res_sel)
        res_tb = C7.eval_generic(va, C7.relation_score_fn(tb, kind), want_alpha=True); save_perq(f"{tag}_trained", res_tb)
        torch.save(model.state_dict(), f"{OUT}/{tag}.pt"); torch.save(tb.state_dict(), f"{OUT}/{tag}_trained.pt")
        out[tag] = {"selected": slim(res_sel), "best_trained": slim(res_tb),
                    "meta": {k: meta[k] for k in ("kind", "epochs", "lr", "selected_epoch", "selected_is_init", "trained_best_epoch", "trained_best_ndcg")},
                    "history": meta["history"],
                    "alpha_analysis": alpha_stats(res_tb["_alpha"]), "alpha_by_relrank": relrank_breakdown(res_tb["_alpha"])}
        p = res_sel["POOLED"]; pt = res_tb["POOLED"]
        log(f"== {tag}: SELECTED NDCG50={p['NDCG@50']:.4f}(init={meta['selected_is_init']}) BEST-TRAINED NDCG50={pt['NDCG@50']:.4f} MRR={pt['MRR']:.4f} ALL@50={pt['ALL@50']:.4f}")
    # bootstrap: best-trained vs C0; and c2 vs c1 (trained)
    c0 = {k: concat("C0", k) for k in ("ndcg", "all10", "all50", "mrr")}
    for tag in ("C7c1", "C7c2"):
        t = {k: concat(f"{tag}_trained", k) for k in ("ndcg", "all10", "all50", "mrr")}
        boot[f"{tag}_trained_vs_C0"] = {m: paired_boot(t[m], c0[m]) for m in ("ndcg", "all10", "all50", "mrr")}
    t1 = {k: concat("C7c1_trained", k) for k in ("ndcg", "all10", "all50", "mrr")}
    t2 = {k: concat("C7c2_trained", k) for k in ("ndcg", "all10", "all50", "mrr")}
    boot["C7c2_vs_C7c1_trained"] = {m: paired_boot(t2[m], t1[m]) for m in ("ndcg", "all10", "all50", "mrr")}
    out["bootstrap"] = boot
    json.dump(out, open(f"{OUT}/_c7c.json", "w"), indent=1, default=str)
    log("BOOTSTRAP " + json.dumps(boot, indent=1))
    log("C7C_DONE wrote results/L2/_ctrl/_c7c.json")


if __name__ == "__main__":
    main()
