"""Report the frozen-config hybrid benchmark: metric table + paired significance from per-question dumps.

Reads results/L2/kg_hybrid.json (aggregate metrics) and kg_hybrid_preds.pt (per-question ranked gold masks,
all seeds), plus the corrected-dropout ablation (kg_hybrid_fullfd.*). Produces:
  1. Benchmark-of-record table: R@{5,20,50} and Hit@{5,50}, mean +/- std over the fixed seeds, per dataset.
  2. flat -> hybrid and flat_drop -> hybrid paired comparison: seed-averaged per-question R@5, with a
     paired bootstrap 95% CI on the mean delta and a paired Wilcoxon test.  No reruns needed.

The per-question dumps make WebQSP (small) statistically defensible: we can bootstrap over questions and run
paired tests against the baseline directly from stored predictions.
"""
import json
import numpy as np

KS = (5, 20, 50); HS = (5, 50)


def _rk(mask, k):
    m = np.asarray(mask); return float(m[:k].sum())


def per_q_recall(preds, mode, ds, k):
    """Seed-averaged per-question recall@k -> array over questions (question order fixed across seeds/modes)."""
    seeds = sorted(preds[mode][ds].keys(), key=int)
    nq = len(preds[mode][ds][seeds[0]])
    out = np.zeros(nq)
    for s in seeds:
        rows = preds[mode][ds][s]
        for i, (mask, ng) in enumerate(rows):
            out[i] += _rk(mask, k) / ng
    return 100.0 * out / len(seeds)


def paired(preds, base, cand, ds, k=5, B=10000, seed=0):
    a = per_q_recall(preds, base, ds, k); b = per_q_recall(preds, cand, ds, k)
    d = b - a; n = len(d)
    rng = np.random.default_rng(seed)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # paired Wilcoxon signed-rank (two-sided) via scipy if available, else sign-test fallback
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        p = wilcoxon(nz).pvalue if len(nz) else 1.0
        test = "wilcoxon"
    except Exception:
        pos = int((d > 0).sum()); neg = int((d < 0).sum()); m = pos + neg
        # two-sided sign test, normal approx
        if m == 0:
            p = 1.0
        else:
            z = (pos - m / 2) / (0.5 * np.sqrt(m)); from math import erfc; p = erfc(abs(z) / np.sqrt(2))
        test = "sign-test"
    return dict(mean_delta=float(d.mean()), ci=(float(lo), float(hi)), p=float(p), test=test, n=n)


def table(path, title):
    d = json.load(open(path))
    modes = [m for m in d if not m.startswith("_")]
    dss = d["_config"]["datasets"]
    print(f"\n=== {title} ({d['_config']['seeds']} seeds, epochs={d['_config']['epochs']}) ===")
    cols = [f"R@{k}" for k in KS] + [f"Hit@{k}" for k in HS]
    for ds in dss:
        print(f"\n  [{ds}]")
        print("  %-10s " % "mode" + " ".join("%14s" % c for c in cols))
        for m in modes:
            print("  %-10s " % m + " ".join("%14s" % ("%.2f+/-%.2f" % tuple(d[m][ds][c])) for c in cols))
    return d


def main():
    d = table("results/L2/kg_hybrid.json", "BENCHMARK-OF-RECORD")
    try:
        table("results/L2/kg_hybrid_fullfd.json", "CORRECTED FEATURE-DROPOUT ABLATION")
    except FileNotFoundError:
        print("\n(kg_hybrid_fullfd.json not present yet)")

    try:
        import torch
        preds = torch.load("results/L2/kg_hybrid_preds.pt")
    except FileNotFoundError:
        print("\n(no preds file yet)"); return
    print("\n=== PAIRED SIGNIFICANCE (seed-averaged per-question R@5) ===")
    for ds in d["_config"]["datasets"]:
        for base in ("flat", "flat_drop"):
            if base in preds and "hybrid" in preds:
                r = paired(preds, base, "hybrid", ds, k=5)
                sig = "***" if r["p"] < 1e-3 else "**" if r["p"] < 1e-2 else "*" if r["p"] < 5e-2 else "ns"
                print("  %-8s %s->hybrid  delta=%+.2f  95%%CI[%+.2f,%+.2f]  p=%.2e (%s) %s  n=%d"
                      % (ds, base, r["mean_delta"], r["ci"][0], r["ci"][1], r["p"], r["test"], sig, r["n"]))


if __name__ == "__main__":
    main()
