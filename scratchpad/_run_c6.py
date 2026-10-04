"""Drive C6a (depth negs + RankNet) and C6b (depth negs + NDCG@50 LambdaRank). Query-only gate.
Saves selected + best-trained per-query arrays and metrics; paired VAL bootstrap of strongest trained vs C0.
lambda_shapley=0, expert_dropout=0. Full-P50 VAL only. No TEST."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT
import l2_c6 as C6

OUT = "results/L2/_ctrl"; log = lambda *a: print(*a, flush=True)


def perq_arrays(res):
    d = {}
    for ds in CT.DS:
        for k, v in res["_perq"][ds].items(): d[f"{ds}__{k}"] = v
    return d


def paired_boot(a, b, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); dif = a - b; obs = float(dif.mean())
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(obs, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "frac_boot>0": round(float((bs > 0).mean()), 3), "significant": bool(lo > 0 or hi < 0)}


def eval_and_save(tag, model, va):
    res = CT.evaluate(model, "C2", va, want_gate=True, collect=True)
    np.savez(f"{OUT}/_perq_{tag}.npz", **perq_arrays(res))
    slim = {ds: {k: round(res[ds][k], 4) for k in res[ds] if k != "n"} | {"n": res[ds]["n"]} for ds in CT.DS}
    slim["POOLED"] = {k: round(res["POOLED"][k], 4) for k in res["POOLED"] if k != "n"} | {"n": res["POOLED"]["n"]}
    slim["gate"] = res["_gate"]
    return res, slim


def concat_perq(tag, key):
    z = np.load(f"{OUT}/_perq_{tag}.npz")
    return np.concatenate([z[f"{ds}__{key}"] for ds in CT.DS])


def main():
    va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
    T0 = time.time(); summary = {}
    for tag, loss_kind in (("C6a", "ranknet"), ("C6b", "lambdarank")):
        model, meta, tb_model = C6.train_c6(tag, loss_kind, "traindepth", epochs=10, lr=3e-4, log=log)
        # selected (init-inclusive) checkpoint
        res_sel, slim_sel = eval_and_save(tag, model, va)
        torch.save(model.state_dict(), f"{OUT}/{tag}.pt")
        # best TRAINED checkpoint (may differ from init) — for the honest bootstrap
        res_tb, slim_tb = eval_and_save(f"{tag}_trained", tb_model, va)
        torch.save(tb_model.state_dict(), f"{OUT}/{tag}_trained.pt")
        out = {"meta": {k: meta[k] for k in ("tag", "loss_kind", "train_name", "epochs", "lr", "best_sel",
                                             "selected_epoch", "selected_is_init", "trained_best_epoch", "trained_best_ndcg")},
               "history": meta["history"], "selected": slim_sel, "best_trained": slim_tb}
        json.dump(out, open(f"{OUT}/{tag}.json", "w"), indent=1, default=str)
        p = slim_sel["POOLED"]; pt = slim_tb["POOLED"]
        log(f"== {tag}: SELECTED NDCG50={p['NDCG@50']:.4f} (init={meta['selected_is_init']})  "
            f"BEST-TRAINED NDCG50={pt['NDCG@50']:.4f}@e{meta['trained_best_epoch']} MRR={pt['MRR']:.4f} ALL@50={pt['ALL@50']:.4f}")
        summary[tag] = out["meta"]
    # bootstrap: strongest trained checkpoint vs C0, on ndcg/all10/all50/mrr
    c0 = {k: concat_perq("C0", k) for k in ("ndcg", "all10", "all50", "mrr")}
    boot = {}
    for tag in ("C6a", "C6b"):
        t = {k: concat_perq(f"{tag}_trained", k) for k in ("ndcg", "all10", "all50", "mrr")}
        boot[f"{tag}_trained_vs_C0"] = {m: paired_boot(t[m], c0[m]) for m in ("ndcg", "all10", "all50", "mrr")}
    json.dump(boot, open(f"{OUT}/_c6_bootstrap.json", "w"), indent=1, default=str)
    log("BOOTSTRAP " + json.dumps(boot, indent=1))
    log(f"C6_DONE total={time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
