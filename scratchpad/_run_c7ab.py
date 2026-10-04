"""C7 Track AB — archetype distillation (C7a hard, C7b soft, C7b2 utility). VAL only. Save + bootstrap vs C0."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT, l2_c7 as C7

OUT = "results/L2/_ctrl"; log = lambda *a: print(*a, flush=True)
COLS = ["MRR", "NDCG@50", "R@5", "R@10", "R@20", "R@50", "ANY@10", "ANY@20", "ANY@50", "ALL@10", "ALL@20", "ALL@50"]


def slim(res):
    o = {ds: {k: round(res[ds][k], 4) for k in COLS} | {"n": res[ds]["n"]} for ds in CT.DS}
    o["POOLED"] = {k: round(res["POOLED"][k], 4) for k in COLS} | {"n": res["POOLED"]["n"]}
    return o


def save_perq(tag, res):
    d = {}
    for ds in CT.DS:
        for k, v in res["_perq"][ds].items(): d[f"{ds}__{k}"] = v
    np.savez(f"{OUT}/_perq_{tag}.npz", **d)


def concat(tag, key):
    z = np.load(f"{OUT}/_perq_{tag}.npz"); return np.concatenate([z[f"{ds}__{key}"] for ds in CT.DS])


def paired_boot(a, b, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); dif = a - b
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(nboot)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(dif.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "frac_boot>0": round(float((bs > 0).mean()), 3), "significant": bool(lo > 0 or hi < 0)}


def main():
    va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
    clf, reg, mu, sd = C7.fit_archetype_heads(seed=0)
    out = {"archetypes": C7.ARCH, "archetype_order": C7.ANAMES}
    for tag, mode in (("C7a", "hard"), ("C7b", "soft"), ("C7b2", "utility")):
        wq = C7.val_weight_vectors(clf, reg, mu, sd, mode, tau=0.05)
        res = C7.eval_generic(va, C7.archetype_weight_fn(wq)); save_perq(tag, res)
        out[tag] = slim(res); p = res["POOLED"]
        log(f"== {tag}({mode}): NDCG50={p['NDCG@50']:.4f} MRR={p['MRR']:.4f} R@5={p['R@5']:.4f} ALL@10={p['ALL@10']:.4f} ALL@50={p['ALL@50']:.4f}")
    # bootstrap vs C0
    c0 = {k: concat("C0", k) for k in ("ndcg", "all10", "all50", "mrr")}
    boot = {}
    for tag in ("C7a", "C7b", "C7b2"):
        t = {k: concat(tag, k) for k in ("ndcg", "all10", "all50", "mrr")}
        boot[f"{tag}_vs_C0"] = {m: paired_boot(t[m], c0[m]) for m in ("ndcg", "all10", "all50", "mrr")}
    out["bootstrap_vs_C0"] = boot
    json.dump(out, open(f"{OUT}/_c7ab.json", "w"), indent=1, default=str)
    log("BOOTSTRAP " + json.dumps(boot, indent=1))
    log("wrote results/L2/_ctrl/_c7ab.json")


if __name__ == "__main__":
    main()
