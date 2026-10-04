"""B1.8c — THE IN-DOMAIN CEILING (the test B1.4-B1.7 never ran).

Every B1.x failure has been attributed to CROSS-DOMAIN transfer (source coverage / conditional shift).
That attribution has never been checked against its own upper bound. This asks:

    If we CHEAT -- train the reserve policy on the SAME dataset it is evaluated on (50/50 query split),
    giving it the dataset identity, its label regime, and its feature distribution for free --
    can query-local features predict the reserve decision AT ALL?

This is an UPPER BOUND on every possible source-regime expansion: no amount of extra source data can
beat training on the target itself. If the in-domain policy also fails, the bottleneck is NOT coverage.

CHEATING BY CONSTRUCTION -- diagnostic only, never a deployable policy. TEST untouched; 0 encoder passes.
"""
import json, os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _b14_twochannel import load
from _b15_policy import query_matrix
from _b17_marginal import utilities, wridge, predict, seq_R, utility_policy_metrics, R_VALS, TAU_GRID
from _b18_complementarity import comp_features, shapley

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
OUT = "results/GENERALIZATION/_g2_b18c_ceiling.json"


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 1e-12 else None


def main():
    res = {"WHAT_THIS_IS": ("IN-DOMAIN (train==eval dataset, 50/50 query split) upper bound on the adaptive "
                            "reserve decision. CHEATS on purpose. Diagnostic only, never deployable.")}
    tab = {}
    for ds in DSES:
        d = load(ds)
        FE = comp_features(d); P0 = query_matrix(d, False)
        UR, D = utilities(d)
        _, pg, _ = shapley(d); pg = np.nan_to_num(pg, nan=0.0)
        nq = len(D); tr = np.arange(nq) % 2 == 0; te = ~tr
        row = {"n": nq, "n_train": int(tr.sum()), "n_eval": int(te.sum()),
               "true_D_mean": [round(float(D[:, k].mean()), 4) for k in range(3)],
               "frac_queries_with_any_positive_utility": round(float((UR.max(1) > 0).mean()), 4),
               "oracle_NET_eval_half": round(float(UR[te].max(1).sum()), 1),
               "mean_oracle_R_eval_half": round(float(np.mean(
                   [R_VALS[int(np.argmax(UR[i] >= UR[i].max() - 1e-12))] for i in np.where(te)[0]])), 3)}

        for tag, X in [("COMP", FE), ("P0", P0), ("P0+COMP", np.hstack([P0, FE]))]:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xn = (X - mu) / sd
            # (i) in-domain regression quality on the policy-relevant target D4 and on phi_structural
            b4 = wridge(Xn[tr], D[tr, 0], np.ones(int(tr.sum())), 1.0)
            r2_d4 = r2(D[te, 0], predict(Xn[te], b4))
            bp = wridge(Xn[tr], pg[tr], np.ones(int(tr.sum())), 1.0)
            r2_phi = r2(pg[te], predict(Xn[te], bp))
            # (ii) in-domain END-TO-END policy: same sequential ladder, taus chosen on the TRAIN half
            betas = [wridge(Xn[tr], D[tr, k], np.ones(int(tr.sum())), 1.0) for k in range(3)]
            pDtr = np.stack([predict(Xn[tr], b) for b in betas], 1)
            pDte = np.stack([predict(Xn[te], b) for b in betas], 1)
            URtr = UR[tr]
            best, bt = -1e18, None
            for taus in itertools.product(TAU_GRID, repeat=3):
                Rq = np.array([seq_R(pDtr[i], taus) for i in range(len(pDtr))])
                u = float(np.sum([URtr[i, R_VALS.index(int(Rq[i]))] for i in range(len(Rq))]))
                if u > best + 1e-9 or (abs(u - best) <= 1e-9 and bt is not None and sum(taus) > sum(bt)):
                    best, bt = u, taus
            Rte = np.array([seq_R(pDte[i], bt) for i in range(len(pDte))])
            met, _ = utility_policy_metrics(UR[te], Rte)
            net = float(np.sum([UR[te][i, R_VALS.index(int(Rte[i]))] for i in range(len(Rte))]))
            row[tag] = {"in_domain_R2_D4": None if r2_d4 is None else round(r2_d4, 4),
                        "in_domain_R2_phi_struct": None if r2_phi is None else round(r2_phi, 4),
                        "in_domain_NET": round(net, 1),
                        "capture_frac_of_oracle": round(net / max(row["oracle_NET_eval_half"], 1e-9), 4),
                        "mean_R": met["mean_chosen_R"], "RESERVE_WASTE_mean": met["RESERVE_WASTE_mean"],
                        "frac_R0": met["frac_R0"], "selected_taus": list(bt)}
        tab[ds] = row
        print(f"[b18c] {ds:13s} oracle {row['oracle_NET_eval_half']:.0f} | "
              + " | ".join(f"{t} NET {row[t]['in_domain_NET']:+.0f} ({row[t]['capture_frac_of_oracle']:.1%}) "
                           f"R2_D4 {row[t]['in_domain_R2_D4']}" for t in ["COMP", "P0", "P0+COMP"]), flush=True)

    res["IN_DOMAIN_CEILING"] = tab
    best_cap = {ds: max(tab[ds][t]["capture_frac_of_oracle"] for t in ["COMP", "P0", "P0+COMP"]) for ds in DSES}
    res["VERDICT"] = {
        "best_in_domain_capture_frac": {k: round(v, 4) for k, v in best_cap.items()},
        "IN_DOMAIN_POLICY_WORKS": {k: bool(v > 0.35) for k, v in best_cap.items()},
        "INTERPRETATION": (
            "If in-domain capture is low, the adaptive-reserve decision is NOT a learnable function of "
            "query-local features on that dataset, and NO source-regime expansion can fix it -- training on "
            "the target itself is the upper bound of any such expansion.")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print("\n=== IN-DOMAIN CEILING (cheating: train==eval dataset) ===")
    print(f"  {'dataset':14s} {'oracleNET':>9s} {'bestNET':>8s} {'capture':>8s} {'R2(D4)':>8s} {'meanR':>6s}")
    for ds in DSES:
        t = tab[ds]
        bt = max(["COMP", "P0", "P0+COMP"], key=lambda k: t[k]["capture_frac_of_oracle"])
        print(f"  {ds:14s} {t['oracle_NET_eval_half']:9.0f} {t[bt]['in_domain_NET']:8.0f} "
              f"{t[bt]['capture_frac_of_oracle']:7.1%} {str(t[bt]['in_domain_R2_D4']):>8s} {t[bt]['mean_R']:6.2f}   [{bt}]")
    print("\n[b18c] wrote", OUT)


if __name__ == "__main__":
    main()
