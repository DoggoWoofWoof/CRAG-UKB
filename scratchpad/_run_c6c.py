"""Drive C6c — full-P50-scope query-only gate, true-rank LambdaRank ΔNDCG@50. Runs only because C6b failed.
Saves selected + best-trained per-query arrays/metrics; paired VAL bootstrap of strongest trained vs C0."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT
import l2_c6 as C6
from _run_c6 import eval_and_save, paired_boot, concat_perq

OUT = "results/L2/_ctrl"; log = lambda *a: print(*a, flush=True)


def main():
    va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
    T0 = time.time()
    model, meta, tb_model = C6.train_c6c("C6c", epochs=8, lr=3e-4, log=log)
    res_sel, slim_sel = eval_and_save("C6c", model, va); torch.save(model.state_dict(), f"{OUT}/C6c.pt")
    res_tb, slim_tb = eval_and_save("C6c_trained", tb_model, va); torch.save(tb_model.state_dict(), f"{OUT}/C6c_trained.pt")
    out = {"meta": {k: meta[k] for k in ("tag", "loss_kind", "train_name", "epochs", "lr", "best_sel",
                                         "selected_epoch", "selected_is_init", "trained_best_epoch", "trained_best_ndcg")},
           "history": meta["history"], "selected": slim_sel, "best_trained": slim_tb}
    json.dump(out, open(f"{OUT}/C6c.json", "w"), indent=1, default=str)
    p = slim_sel["POOLED"]; pt = slim_tb["POOLED"]
    log(f"== C6c: SELECTED NDCG50={p['NDCG@50']:.4f} (init={meta['selected_is_init']})  "
        f"BEST-TRAINED NDCG50={pt['NDCG@50']:.4f}@e{meta['trained_best_epoch']} MRR={pt['MRR']:.4f} ALL@50={pt['ALL@50']:.4f}")
    c0 = {k: concat_perq("C0", k) for k in ("ndcg", "all10", "all50", "mrr")}
    t = {k: concat_perq("C6c_trained", k) for k in ("ndcg", "all10", "all50", "mrr")}
    boot = {"C6c_trained_vs_C0": {m: paired_boot(t[m], c0[m]) for m in ("ndcg", "all10", "all50", "mrr")}}
    # if selected beat init, also bootstrap selected vs C0
    if not meta["selected_is_init"]:
        ts = {k: concat_perq("C6c", k) for k in ("ndcg", "all10", "all50", "mrr")}
        boot["C6c_selected_vs_C0"] = {m: paired_boot(ts[m], c0[m]) for m in ("ndcg", "all10", "all50", "mrr")}
    json.dump(boot, open(f"{OUT}/_c6c_bootstrap.json", "w"), indent=1, default=str)
    log("BOOTSTRAP " + json.dumps(boot, indent=1))
    log(f"C6c_DONE total={time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
