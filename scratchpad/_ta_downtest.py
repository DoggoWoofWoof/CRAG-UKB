"""Paired bootstrap of each downstream config against BASE, on the per-query arrays dumped by _ta_down.py."""
import sys, glob, os, json
import numpy as np
DS = sys.argv[1] if len(sys.argv) > 1 else "2wiki_clean"
P = "results/GENERALIZATION"
base = np.load(f"{P}/_g2_tadown_{DS}_BASE_M0_perq.npz")
rng = np.random.default_rng(0)
def boot(a, b, n=10000):
    d = a - b; idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)
print(f"=== DOWNSTREAM paired bootstrap vs BASE  ({DS}, frozen C7b->C8c->C11a) ===")
print(f"  {'cfg':14s} {'metric':8s} {'delta':>9s} {'95% CI':>22s}  sig")
for f in sorted(glob.glob(f"{P}/_g2_tadown_{DS}_*_perq.npz")):
    tag = os.path.basename(f).replace(f"_g2_tadown_{DS}_", "").replace("_perq.npz", "")
    if tag == "BASE_M0":
        continue
    z = np.load(f)
    if len(z["eval_qi"]) != len(base["eval_qi"]) or not np.array_equal(z["eval_qi"], base["eval_qi"]):
        print(f"  {tag}: QUERY SET MISMATCH -- skipped"); continue
    for mk in ("pq_ndcg5", "pq_recall5", "pq_all50"):
        if mk not in z.files or mk not in base.files:
            continue
        d, lo, hi, sig = boot(z[mk], base[mk])
        print(f"  {tag:14s} {mk[3:]:8s} {d:+9.4f}   [{lo:+.4f}, {hi:+.4f}]  {'YES' if sig else 'no'}")
