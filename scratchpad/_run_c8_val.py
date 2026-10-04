"""C8 OFFICIAL VAL MILESTONE — Task 10. ONE evaluation on the official VAL, AFTER the C8 design is selected on
DEV_INNER. Refit C7b (U50) and the SELECTED C8 target on FULL TRAIN. Compare C0 / C7b / C8. Paired bootstrap
C8 vs C7b on NDCG@5, GOLD_RECALL@5, NDCG@50, ALL@50. TEST untouched. Writes _c8_val.json."""
import sys, os, json
sys.path.insert(0, "scratchpad")
import numpy as np, l2_c8 as C8
OUT = C8.OUT; log = lambda *a: print(*a, flush=True)
LAMBDA = 0.65
MET = ["ndcg5", "recall5_macro", "any5", "all5_feas", "mrr", "ndcg50", "all10", "all50"]


def fit_full(target):
    inner = {ds: C8.precompute_arch(ds, "train")["qi"] for ds in C8.DS}   # full train
    if target in ("ndcg5", "ndcg50"):
        return C8.fit_soft_head(target, inner)
    import torch, torch.nn as nn, torch.nn.functional as F
    Xs, ys = [], []
    for ds in C8.DS:
        A = C8.precompute_arch(ds, "train"); U = LAMBDA * A["ndcg5"] + (1 - LAMBDA) * A["ndcg50"]
        Xs.append(C8.feats_for(ds, "train", A["qi"])); ys.append(U.argmax(1))
    X = np.concatenate(Xs).astype(np.float32); y = np.concatenate(ys).astype(np.int64)
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)); yt = torch.from_numpy(y)
    torch.manual_seed(0); clf = nn.Linear(X.shape[1], 6)
    cnt = np.bincount(y, minlength=6); cw = torch.from_numpy((cnt.sum() / (6 * np.clip(cnt, 1, None))).astype(np.float32))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy(clf(Xn), yt, weight=cw).backward(); opt.step()
    return clf, mu, sd, {"lambda": LAMBDA}


def eval_val(Wfn):
    rb = {}; perq = {}
    for ds in C8.DS:
        qi = C8.precompute_arch(ds, "val")["qi"]; W = Wfn(ds, qi)
        r = C8.eval_policy(ds, "val", qi, W, want_perq=True); rb[ds] = r; perq[ds] = r["_perq"]
    return rb, C8.pool(rb), perq


def paired_boot(a, b, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(a); dif = a - b
    bs = np.array([dif[rng.integers(0, n, n)].mean() for _ in range(nboot)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": round(float(dif.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "frac_boot>0": round(float((bs > 0).mean()), 3), "significant": bool(lo > 0 or hi < 0)}


def slim(p):
    return {k: (round(p[k], 4) if isinstance(p.get(k), float) else p.get(k)) for k in MET}


def main():
    sel = json.load(open(f"{OUT}/_c8_train.json"))["SELECTED_ON_DEV"]
    target = {"C8a": "ndcg5", "C8b": "mix", "C7b": "ndcg50"}[sel]
    log(f"SELECTED_ON_DEV={sel} -> target={target}")

    c7b = fit_full("ndcg50"); c8 = fit_full(target)
    def Wc7b(ds, qi): return C8.soft_weights(*c7b[:3], ds, "val", qi)
    def Wc8(ds, qi): return C8.soft_weights(*c8[:3], ds, "val", qi)
    def Wc0(ds, qi): return C8.equal_weights(len(qi))

    rb0, p0, _ = eval_val(Wc0); rbB, pB, pqB = eval_val(Wc7b); rbC, pC, pqC = eval_val(Wc8)
    log("C0  val: " + json.dumps(slim(p0)))
    log("C7b val: " + json.dumps(slim(pB)))
    log(f"{sel} val: " + json.dumps(slim(pC)))

    # per-query pooled arrays for bootstrap (macro metrics)
    def cat(pq, key): return np.concatenate([pq[ds][key] for ds in C8.DS])
    keymap = {"ndcg5": "ndcg5", "recall5_macro": "recall5", "ndcg50": "ndcg50", "all50": "all50"}
    boot = {m: paired_boot(cat(pqC, keymap[m]), cat(pqB, keymap[m])) for m in ("ndcg5", "recall5_macro", "ndcg50", "all50")}

    out = {"selected": sel, "target": target,
           "OFFICIAL_VAL": {"C0": {"pooled": slim(p0), "per_ds": {ds: slim(rb0[ds]) for ds in C8.DS}},
                            "C7b": {"pooled": slim(pB), "per_ds": {ds: slim(rbB[ds]) for ds in C8.DS}},
                            sel: {"pooled": slim(pC), "per_ds": {ds: slim(rbC[ds]) for ds in C8.DS}}},
           "bootstrap_C8_vs_C7b": boot,
           "delta_C8_minus_C7b": {m: round(pC.get(m, 0) - pB.get(m, 0), 4) for m in MET if isinstance(pC.get(m), float)},
           "delta_C8_minus_C0": {m: round(pC.get(m, 0) - p0.get(m, 0), 4) for m in MET if isinstance(pC.get(m), float)}}
    json.dump(out, open(f"{OUT}/_c8_val.json", "w"), indent=1, default=str)
    log("BOOT " + json.dumps(boot, indent=1))
    log("C8_VAL_DONE wrote results/L2/_ctrl/_c8_val.json")


if __name__ == "__main__":
    main()
