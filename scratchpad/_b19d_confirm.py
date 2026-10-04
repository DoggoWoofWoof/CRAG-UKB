"""B1.9 confirmation — is the negative LOCO result an artifact of the 1-NN policy, or is the corpus
descriptor genuinely uninformative about the reserve operating point?

Three checks, all cheap:
  A  SEPARABILITY (in-hindsight upper bound): can ANY single descriptor + ANY threshold separate the
     corpora that want R=16 from those that want R<=4? This is a best-case ceiling, fit on all 6 corpora
     with the labels visible -- if it fails here it cannot work under LOCO.
  B  FAIRER POLICIES: ordinal linear fit and rank-monotone fit (not just 1-NN), still selected in-fold.
  C  BASELINE DECOMPOSITION: how much does per-corpus tuning actually buy over ONE global constant?
"""
import os, sys, json, itertools
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from _b14_twochannel import load
from _b17_marginal import utilities, R_VALS

DESC = json.load(open("results/GENERALIZATION/_g2_b19_descriptors.json"))
POL = json.load(open("results/GENERALIZATION/_g2_b19c_corpus_policy.json"))
U = POL["STEP2_U_C"]
ALL = ["metaqa", "2wiki_clean", "squad_clean", "musique_clean", "hotpotqa_clean", "webqsp"]
FEATS = ["mean_degree", "neighbor_jaccard_redundancy", "StructSemOverlap@10", "StructSemOverlap@50",
         "StructSemOverlap@200", "edge_semantic_lift", "hop2_over_hop1", "edge_density",
         "degree_cv", "giant_component_frac", "isolated_frac", "hop2_unique_mean"]
OUT = "results/GENERALIZATION/_g2_b19d_confirm.json"


def main():
    URs = {ds: utilities(load(ds))[0] for ds in ALL}
    ri = {r: k for k, r in enumerate(R_VALS)}
    bestR = np.array([U[ds]["best_constant_R"] for ds in ALL], float)
    netq = np.array([U[ds]["NET_per_query_at_best"] for ds in ALL])
    hi = bestR >= 16                                        # the decision that dominates utility
    res = {}

    # ---------------- A. separability ceiling (labels VISIBLE -- an upper bound, not a policy)
    A = {}
    for f in FEATS:
        x = np.array([DESC[ds][f] for ds in ALL], float)
        best_acc, best_thr = 0.0, None
        order = np.sort(np.unique(x))
        cuts = [(order[i] + order[i + 1]) / 2 for i in range(len(order) - 1)]
        for t in cuts:
            for sgn in (1, -1):
                pred = (sgn * x > sgn * t)
                acc = float((pred == hi).mean())
                if acc > best_acc:
                    best_acc, best_thr = acc, (float(t), sgn)
        A[f] = {"best_separating_accuracy": round(best_acc, 4), "threshold": best_thr,
                "R16_values": {ds: DESC[ds][f] for ds in ALL if U[ds]["best_constant_R"] >= 16},
                "R_le4_values": {ds: DESC[ds][f] for ds in ALL if U[ds]["best_constant_R"] < 16}}
    A["_SUMMARY"] = {"n_corpora": len(ALL), "n_want_R16": int(hi.sum()),
                     "best_over_all_descriptors": round(max(A[f]["best_separating_accuracy"] for f in FEATS), 4),
                     "perfect_separator_exists": bool(any(A[f]["best_separating_accuracy"] > 0.999 for f in FEATS))}
    res["A_SEPARABILITY_CEILING"] = A

    # ---------------- B. fairer LOCO policies
    def spear(x, y):
        if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return None
        rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    B = {}
    for pol in ["1NN", "linear_ordinal", "rank_monotone"]:
        rows, tot = [], 0.0
        for tgt in ALL:
            src = [d for d in ALL if d != tgt]
            Rs = np.array([U[d]["best_constant_R"] for d in src], float)
            scored = []
            for f in FEATS:
                xs = np.array([DESC[d][f] for d in src], float)
                r = spear(xs, Rs)
                if r is not None:
                    scored.append((abs(r), f, r))
            scored.sort(key=lambda t: (-t[0], t[1]))
            _, fsel, rsel = scored[0]
            xs = np.array([DESC[d][fsel] for d in src], float); xt = float(DESC[tgt][fsel])
            if pol == "1NN":
                Rp = float(Rs[int(np.argmin(np.abs(xs - xt)))])
            elif pol == "linear_ordinal":
                z = (xs - xs.mean()) / (xs.std() + 1e-9)
                b = np.polyfit(z, Rs, 1)
                Rp = float(np.polyval(b, (xt - xs.mean()) / (xs.std() + 1e-9)))
            else:                                            # rank_monotone: map descriptor rank -> sorted R
                k = int((xs < xt).sum())
                Rp = float(np.sort(Rs)[::-1][k] if rsel < 0 else np.sort(Rs)[min(k, len(Rs) - 1)])
            Rq = min(R_VALS, key=lambda r: abs(r - Rp))       # snap to the frozen grid
            net = float(URs[tgt][:, ri[Rq]].sum()); tot += net
            rows.append({"target": tgt, "descriptor": fsel, "R_pred_raw": round(Rp, 2), "R": Rq,
                         "R_oracle": U[tgt]["best_constant_R"], "NET": round(net, 1),
                         "correct": bool(Rq == U[tgt]["best_constant_R"])})
        B[pol] = {"rows": rows, "TOTAL_NET": round(tot, 1),
                  "exact_match": round(float(np.mean([r["correct"] for r in rows])), 4)}
    res["B_FAIRER_POLICIES"] = B

    # ---------------- C. what per-corpus tuning is actually worth
    glob_net = {R: float(sum(URs[ds][:, ri[R]].sum() for ds in ALL)) for R in R_VALS}
    R_glob = max(glob_net, key=glob_net.get)
    orc = float(sum(URs[ds][:, ri[U[ds]["best_constant_R"]]].sum() for ds in ALL))
    perq_orc = float(sum(URs[ds].max(1).sum() for ds in ALL))
    res["C_BASELINES"] = {
        "NET_by_single_global_R": {str(R): round(glob_net[R], 1) for R in R_VALS},
        "best_global_R": R_glob, "NET_global_best": round(glob_net[R_glob], 1),
        "NET_per_corpus_oracle_constant": round(orc, 1),
        "NET_per_query_oracle": round(perq_orc, 1),
        "gain_from_perfect_per_corpus_tuning": round(orc - glob_net[R_glob], 1),
        "global_recovers_frac_of_corpus_oracle": round(glob_net[R_glob] / max(orc, 1e-9), 4),
        "best_label_free_policy_NET": max(B[p]["TOTAL_NET"] for p in B),
        "per_corpus_gain_breakdown": {ds: round(float(URs[ds][:, ri[U[ds]["best_constant_R"]]].sum()
                                                      - URs[ds][:, ri[R_glob]].sum()), 1) for ds in ALL}}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print("=== A. SEPARABILITY CEILING: split corpora wanting R=16 from R<=4 (labels VISIBLE) ===")
    print(f"  corpora wanting R=16: {[d for d in ALL if U[d]['best_constant_R']>=16]}")
    print(f"  {'descriptor':30s} {'best acc':>9s}   values(R16)  |  values(R<=4)")
    for f in FEATS:
        a = A[f]
        v16 = ",".join(f"{v:.4g}" for v in a["R16_values"].values())
        vlo = ",".join(f"{v:.4g}" for v in a["R_le4_values"].values())
        print(f"  {f:30s} {a['best_separating_accuracy']:9.3f}   [{v16}] | [{vlo}]")
    print(f"  --> perfect separator exists: {A['_SUMMARY']['perfect_separator_exists']}  "
          f"(best {A['_SUMMARY']['best_over_all_descriptors']:.3f})")

    print("\n=== B. fairer LOCO policies (descriptor selected in-fold) ===")
    for p in B:
        print(f"  {p:16s} TOTAL_NET {B[p]['TOTAL_NET']:8.1f}  exact-match {B[p]['exact_match']:.1%}  "
              + " ".join(f"{r['target'][:6]}:R{r['R']}(orc{r['R_oracle']})" for r in B[p]["rows"]))

    c = res["C_BASELINES"]
    print("\n=== C. what per-corpus tuning is worth ===")
    print(f"  NET by single global R: {c['NET_by_single_global_R']}  -> best global R={c['best_global_R']} "
          f"NET {c['NET_global_best']}")
    print(f"  per-corpus oracle constant NET {c['NET_per_corpus_oracle_constant']}  "
          f"(global recovers {c['global_recovers_frac_of_corpus_oracle']:.1%})")
    print(f"  gain from PERFECT per-corpus tuning: +{c['gain_from_perfect_per_corpus_tuning']}")
    print(f"  best label-free policy: {c['best_label_free_policy_NET']}")
    print(f"  per-corpus gain breakdown vs global: {c['per_corpus_gain_breakdown']}")
    print("\nwrote " + OUT)


if __name__ == "__main__":
    main()
