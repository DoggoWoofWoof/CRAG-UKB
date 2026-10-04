"""Emit the MuSiQue TASK-7 markdown subsection from the downstream artefacts."""
import io, json, glob, os
import numpy as np

P = "results/GENERALIZATION"
DS = "musique_clean"
L1 = json.load(open(f"{P}/_g2_ta_{DS}.json"))["CONFIGS"]
ORDER = [("A1b_M0", "A1b M=0"), ("A1a_M32", "A1a M=32"), ("A2_M32", "A2 M=32"), ("A2dOnly_M32", "A2dOnly M=32")]
FILE = {"A1b_M0": "A1b_M0", "A1a_M32": "A1a_M32", "A2_M32": "A2_M32", "A2dOnly_M32": "A2dOnly_M32"}

b = np.load(f"{P}/_g2_tadown_{DS}_BASE_M0_perq.npz")
bj = json.load(open(f"{P}/_g2_tadown_{DS}_BASE_M0.json"))
rng = np.random.default_rng(0)


def boot(a, bb, n=10000):
    d = a - bb
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)


out = []
out.append(f"### musique_clean (n = {bj['n_queries']}, frozen valid universe)\n")
out.append("| cfg | L1 ALL@P50 net | scope Δ (cands) | scope Δ (golds) | zero-gold q | nDCG@5 | R@5 | ALL@5 | ALL@50 |")
out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
a = bj["AGG"]
out.append(f"| **BASE** (gate EXACT) | — | 0 / 0 | 0 / 0 | 0 | {a['ndcg5']:.4f} | {a['recall5_macro']:.4f} | "
           f"{a['all5']:.4f} | {a['all50']:.4f} |")
for key, lab in ORDER:
    f = f"{P}/_g2_tadown_{DS}_{FILE[key]}.json"
    if not os.path.exists(f):
        continue
    j = json.load(open(f)); a = j["AGG"]; s = j["SCOPE_DELTA"]
    net = L1[key]["paired_vs_BASE_ALL"]["net"]; p = L1[key]["paired_vs_BASE_ALL"]["mcnemar_p"]
    sig = f" (p={p:.3f})" if p < 0.05 else ""
    out.append(f"| {lab} | {net:+d}{sig} | −{s['evicted_cands']:,} / +{s['added_cands']:,} | "
               f"−{s['evicted_golds']} / +{s['added_golds']} | {j.get('ZERO_GOLD_QUERIES', 0)} | "
               f"{a['ndcg5']:.4f} | {a['recall5_macro']:.4f} | {a['all5']:.4f} | {a['all50']:.4f} |")

out.append("\nPaired bootstrap (10 000 resamples) against BASE:\n")
out.append("| cfg | ΔnDCG@5 | 95 % CI | ΔR@5 | 95 % CI | ΔALL@50 | 95 % CI |")
out.append("|---|--:|:--|--:|:--|--:|:--|")
for key, lab in ORDER:
    f = f"{P}/_g2_tadown_{DS}_{FILE[key]}_perq.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    cells = []
    for mk in ("pq_ndcg5", "pq_recall5", "pq_all50"):
        d, lo, hi, sg = boot(z[mk], b[mk])
        cells.append(f"**{d:+.4f}**" if sg else f"{d:+.4f}")
        cells.append(f"[{lo:+.4f}, {hi:+.4f}] {'**sig**' if sg else 'ns'}")
    out.append(f"| {lab} | " + " | ".join(cells) + " |")

out.append("\n### Stage decomposition — musique_clean (nDCG@5 units, split by what happened to gold coverage)\n")
out.append("| cfg | n gained | n lost | Δ on gained | Δ on lost | Δ on **unchanged** | total | share on unchanged |")
out.append("|---|--:|--:|--:|--:|--:|--:|--:|")
for key, lab in ORDER:
    f = f"{P}/_g2_tadown_{DS}_{FILE[key]}_perq.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f)
    d = z["pq_ndcg5"] - b["pq_ndcg5"]; aa, ab = z["pq_all50"], b["pq_all50"]
    G, L, S = (aa > ab), (aa < ab), (aa == ab); t = d.sum()
    out.append(f"| {lab} | {G.sum()} | {L.sum()} | {d[G].sum():+.2f} | {d[L].sum():+.2f} | "
               f"**{d[S].sum():+.2f}** | {t:+.2f} | **{100*d[S].sum()/t if t else 0:.0f} %** |")
print("\n".join(out))
