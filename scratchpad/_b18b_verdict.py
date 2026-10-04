"""B1.8b — CONTROLS + VERDICT for the complementarity representation.

The B1.8a diagnostics produced two results that CANNOT be interpreted without controls:
  (i)  STEP-4: every complementarity feature correlates with phi_structural about as strongly as
       `js_rank_null` -- a quantity that BY CONSTRUCTION depends only on universe size. If the null
       matches the real features, the "complementarity" correlations are an artifact.
  (ii) STEP-5: calibration error fell vs the B1.7 P0 baseline. But a 5-bin conditional mean is far more
       regularized than B1.7's 17-feature linear regressor, so the gain may be entirely the CONSTANT
       (source-pooled-mean) predictor -- i.e. exactly B1.7's failure mode with a better estimator.

Controls run here:
  A  CONSTANT source-pooled-mean calibration baseline (the honest STEP-5 control)
  B  collinearity: are the complementarity features proxies for expansion volume / universe size?
  C  IN-DOMAIN predictability CEILING for phi_structural (upper bound; if ~0 in-domain, cross-domain is moot)
  D  LODO cross-domain phi_structural prediction
  E  end-to-end LINEAR_COMPLEMENTARITY_POLICY (B1.7 machinery, new features) -- the decisive STEP-6 test
  F  identity / domain-shift diagnostic: dataset predictability from P0 vs COMP
  G  conditional-shift summary vs B1.7

Frozen: universe, BASE_FUSED, s_dir, two_channel_pool, TOP_POOL=50, R in {0,4,8,16}. 0 encoder passes.
No dataset id in any policy input, no target statistics, no target calibration, no TEST.
"""
import json, os, sys, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _b14_twochannel import load, per_query_idx, TOP_POOL
from _b15_policy import query_matrix
from _b17_marginal import utilities, wridge, predict, seq_R, utility_policy_metrics, R_VALS, BLOCKS, TAU_GRID, EPS
from _b18_complementarity import comp_features, shapley, CF_NAMES

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
OUT = "results/GENERALIZATION/_g2_b18b_verdict.json"
rng = np.random.default_rng(0)


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - p) ** 2).sum() / ss) if ss > 1e-12 else None


def spear(x, y):
    def rk(v):
        o = np.argsort(v, kind="stable"); r = np.empty(len(v), float); r[o] = np.arange(len(v)); return r
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


def main():
    DATA = {ds: load(ds) for ds in DSES}
    FE, P0, PHI_G, D, UR = {}, {}, {}, {}, {}
    for ds in DSES:
        d = DATA[ds]
        FE[ds] = comp_features(d)
        P0[ds] = query_matrix(d, False)
        _, pg, _ = shapley(d)
        PHI_G[ds] = np.nan_to_num(pg, nan=0.0)
        u, dd = utilities(d)
        UR[ds], D[ds] = u, dd
        print(f"[b18b] {ds}: comp {FE[ds].shape} p0 {P0[ds].shape}", flush=True)

    res = {"CONTRACT": {"encoder_passes": 0, "uses_dataset_id": False, "uses_target_stats": False,
                        "uses_TEST": False, "frozen": ["universe", "BASE_FUSED", "s_dir", "two_channel_pool",
                                                       "TOP_POOL=50", "R in {0,4,8,16}"]}}

    # ---------------------------------------------------------- A. CONSTANT calibration baseline
    print("[b18b] A constant baseline ...", flush=True)
    B17 = {"metaqa": [0.020, 0.003, 0.002], "2wiki_clean": [0.099, 0.054, 0.094],
           "squad_clean": [0.246, 0.148, 0.324]}
    A = {}
    tot_c, tot_17 = 0.0, 0.0
    for tgt in DSES:
        src = [s for s in DSES if s != tgt]
        errs_c, errs_17 = [], []
        for k in range(3):
            pooled = np.concatenate([D[s][:, k] for s in src]).mean()   # source-pooled mean, no query info
            true = D[tgt][:, k].mean()
            errs_c.append(abs(pooled - true)); errs_17.append(abs(B17[tgt][k] - true))
        A[tgt] = {"const_pred_D": [round(float(np.concatenate([D[s][:, k] for s in src]).mean()), 4) for k in range(3)],
                  "true_D": [round(float(D[tgt][:, k].mean()), 4) for k in range(3)],
                  "CONST_sum_abs_err": round(float(sum(errs_c)), 4),
                  "B17_P0_sum_abs_err": round(float(sum(errs_17)), 4),
                  "CONST_D4_err": round(float(errs_c[0]), 4), "B17_P0_D4_err": round(float(errs_17[0]), 4)}
        tot_c += sum(errs_c); tot_17 += sum(errs_17)
    A["TOTAL"] = {"CONST": round(tot_c, 4), "B17_P0": round(tot_17, 4)}
    A["NOTE"] = ("The constant source-pooled-mean predictor uses ZERO query information. Any representation that "
                 "does not beat it has added no query-conditional signal.")
    res["A_CONSTANT_BASELINE"] = A

    # ---------------------------------------------------------- B. collinearity / null control
    print("[b18b] B collinearity ...", flush=True)
    ja, jn = CF_NAMES.index("added_frac"), CF_NAMES.index("js_rank_null")
    B = {"corr_with_added_frac": {}, "corr_with_js_rank_null": {}, "NULL_CONTROL": {}}
    for j, nm in enumerate(CF_NAMES):
        B["corr_with_added_frac"][nm] = {ds: (None if np.std(FE[ds][:, j]) < 1e-12 else
                                              round(float(np.corrcoef(FE[ds][:, j], FE[ds][:, ja])[0, 1]), 3))
                                         for ds in DSES}
        B["corr_with_js_rank_null"][nm] = {ds: (None if np.std(FE[ds][:, j]) < 1e-12 else
                                                round(float(np.corrcoef(FE[ds][:, j], FE[ds][:, jn])[0, 1]), 3))
                                           for ds in DSES}
    for ds in DSES:
        rn = abs(np.corrcoef(FE[ds][:, jn], PHI_G[ds])[0, 1])
        best_real = max(abs(np.corrcoef(FE[ds][:, j], PHI_G[ds])[0, 1])
                        for j, nm in enumerate(CF_NAMES) if nm != "js_rank_null")
        B["NULL_CONTROL"][ds] = {"abs_corr_null_vs_phiG": round(float(rn), 4),
                                 "best_abs_corr_real_feature_vs_phiG": round(float(best_real), 4),
                                 "REAL_BEATS_NULL": bool(best_real > rn + 0.05)}
    res["B_COLLINEARITY_AND_NULL_CONTROL"] = B

    # ---------------------------------------------------------- C. in-domain predictability ceiling
    print("[b18b] C in-domain ceiling ...", flush=True)
    C = {}
    for ds in DSES:
        nq = len(PHI_G[ds]); tr = np.arange(nq) % 2 == 0; te = ~tr
        y = PHI_G[ds]
        row = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
               "phiG_std": round(float(y.std()), 4), "phiG_mean": round(float(y.mean()), 4)}
        for tag, X in [("COMP", FE[ds]), ("P0", P0[ds]), ("P0+COMP", np.hstack([P0[ds], FE[ds]]))]:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xs = (X - mu) / sd
            beta = wridge(Xs[tr], y[tr], np.ones(int(tr.sum())), 1.0)
            p = predict(Xs[te], beta)
            row[tag] = {"heldout_R2": None if r2(y[te], p) is None else round(r2(y[te], p), 4),
                        "heldout_spearman": None if spear(p, y[te]) is None else round(spear(p, y[te]), 4)}
        C[ds] = row
    C["NOTE"] = ("IN-DOMAIN held-out R2 is an UPPER BOUND on any cross-domain policy using these features. "
                 "If it is ~0 for a dataset, phi_structural is not representable there and the cross-domain "
                 "stability question is moot for that domain.")
    res["C_IN_DOMAIN_CEILING"] = C

    # ---------------------------------------------------------- D. LODO phi_structural prediction
    print("[b18b] D LODO phi ...", flush=True)
    Dd = {}
    for tgt in DSES:
        src = [s for s in DSES if s != tgt]
        row = {}
        for tag, get in [("COMP", lambda s: FE[s]), ("P0", lambda s: P0[s]),
                         ("P0+COMP", lambda s: np.hstack([P0[s], FE[s]]))]:
            Xs_ = np.vstack([get(s) for s in src]); ys = np.concatenate([PHI_G[s] for s in src])
            w = np.concatenate([np.full(len(PHI_G[s]), 1.0 / len(PHI_G[s])) for s in src]); w /= w.mean()
            mu, sd = Xs_.mean(0), Xs_.std(0) + 1e-9
            beta = wridge((Xs_ - mu) / sd, ys, w, 1.0)
            p = predict((get(tgt) - mu) / sd, beta); yt = PHI_G[tgt]
            row[tag] = {"pred_mean": round(float(p.mean()), 4), "true_mean": round(float(yt.mean()), 4),
                        "target_R2": None if r2(yt, p) is None else round(r2(yt, p), 4),
                        "target_spearman": None if spear(p, yt) is None else round(spear(p, yt), 4)}
        Dd[tgt] = row
    res["D_LODO_PHI_PREDICTION"] = Dd

    # ---------------------------------------------------------- E. end-to-end LINEAR_COMPLEMENTARITY_POLICY
    print("[b18b] E end-to-end policy ...", flush=True)
    E = {}
    for tag, get in [("COMP", lambda s: FE[s]), ("P0+COMP", lambda s: np.hstack([P0[s], FE[s]]))]:
        E[tag] = {}
        for tgt in DSES:
            src = [s for s in DSES if s != tgt]
            Xs_ = np.vstack([get(s) for s in src])
            w = np.concatenate([np.full(len(D[s]), 1.0 / len(D[s])) for s in src]); w /= w.mean()
            mu, sd = Xs_.mean(0), Xs_.std(0) + 1e-9
            Xn = (Xs_ - mu) / sd; Xt = (get(tgt) - mu) / sd
            betas = [wridge(Xn, np.concatenate([D[s][:, k] for s in src]), w, 1.0) for k in range(3)]
            pDs = np.stack([predict(Xn, b) for b in betas], 1)
            pDt = np.stack([predict(Xt, b) for b in betas], 1)
            URs = np.vstack([UR[s] for s in src])
            # tau selection on SOURCE TRAIN ONLY; conservative (largest tau) tie-break
            best, bt = -1e18, None
            for taus in itertools.product(TAU_GRID, repeat=3):
                Rq = np.array([seq_R(pDs[i], taus) for i in range(len(pDs))])
                u = float(np.sum([URs[i, R_VALS.index(int(Rq[i]))] * w[i] for i in range(len(Rq))]))
                if u > best + 1e-9 or (abs(u - best) <= 1e-9 and bt is not None and sum(taus) > sum(bt)):
                    best, bt = u, taus
            Rt = np.array([seq_R(pDt[i], bt) for i in range(len(pDt))])
            met, orc = utility_policy_metrics(UR[tgt], Rt)
            net = float(np.sum([UR[tgt][i, R_VALS.index(int(Rt[i]))] for i in range(len(Rt))]))
            E[tag][tgt] = {"selected_taus": list(bt), "NET_gold": round(net, 1),
                           "target_oracle_NET": round(float(UR[tgt].max(1).sum()), 1),
                           "pred_D_target_mean": [round(float(pDt[:, k].mean()), 4) for k in range(3)],
                           "true_D_target_mean": [round(float(D[tgt][:, k].mean()), 4) for k in range(3)],
                           "mean_R": met["mean_chosen_R"], "RESERVE_WASTE_mean": met["RESERVE_WASTE_mean"],
                           "UTILITY_REGRET_mean": met["UTILITY_REGRET_mean"], "frac_R0": met["frac_R0"],
                           "mean_oracle_R": met["mean_oracle_R"]}
    E["B15_REFERENCE"] = {"metaqa": {"NET": 51, "mean_R": None}, "2wiki_clean": {"NET": 69},
                          "squad_clean": {"NET": 0, "mean_R": 8.72}}
    E["B17_REFERENCE"] = {"metaqa": {"NET": 3, "mean_R": 0.13}, "2wiki_clean": {"NET": 73, "mean_R": 9.05},
                          "squad_clean": {"NET": 0, "mean_R": 10.12}}
    res["E_LINEAR_COMPLEMENTARITY_POLICY"] = E

    # ---------------------------------------------------------- F. identity / domain-shift diagnostic
    print("[b18b] F identity ...", flush=True)
    try:
        from sklearn.linear_model import LogisticRegression
        F = {}
        for tag, get in [("P0", lambda s: P0[s]), ("COMP", lambda s: FE[s]),
                         ("P0+COMP", lambda s: np.hstack([P0[s], FE[s]]))]:
            X = np.vstack([get(s) for s in DSES])
            y = np.concatenate([np.full(len(get(s)), i) for i, s in enumerate(DSES)])
            tr = np.arange(len(y)) % 2 == 0; te = ~tr
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
            clf.fit((X[tr] - mu) / sd, y[tr])
            acc = float(clf.score((X[te] - mu) / sd, y[te]))
            F[tag] = {"heldout_dataset_id_accuracy": round(acc, 4), "chance": round(1 / 3, 4)}
        F["NOTE"] = ("Dataset predictability is NOT itself disqualifying. The question is whether "
                     "P(structural utility | representation) becomes more stable -- see A/D/E.")
        res["F_IDENTITY_DIAGNOSTIC"] = F
    except Exception as e:
        res["F_IDENTITY_DIAGNOSTIC"] = {"error": str(e)}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    # ---------------------------------------------------------- console
    print("\n=== A  calibration: CONSTANT (no query info) vs B1.7 P0 ===")
    for tgt in DSES:
        a = A[tgt]
        print(f"  {tgt:13s} const_err {a['CONST_sum_abs_err']:.4f}   B17_P0_err {a['B17_P0_sum_abs_err']:.4f}   "
              f"const_pred {a['const_pred_D']} true {a['true_D']}")
    print(f"  TOTAL const {A['TOTAL']['CONST']:.4f}  vs  B17_P0 {A['TOTAL']['B17_P0']:.4f}")

    print("\n=== B  null control (|corr| vs phi_structural) ===")
    for ds in DSES:
        b = B["NULL_CONTROL"][ds]
        print(f"  {ds:13s} null {b['abs_corr_null_vs_phiG']:.4f}   best_real {b['best_abs_corr_real_feature_vs_phiG']:.4f}"
              f"   real_beats_null={b['REAL_BEATS_NULL']}")
    print("  corr(feature, added_frac):")
    for nm in CF_NAMES:
        print(f"    {nm:15s} " + " ".join(f"{k[:6]}:{v:+.2f}" if v is not None else f"{k[:6]}: None"
                                          for k, v in B["corr_with_added_frac"][nm].items()))

    print("\n=== C  IN-DOMAIN ceiling for phi_structural (held-out R2) ===")
    for ds in DSES:
        c = C[ds]
        print(f"  {ds:13s} std {c['phiG_std']:.3f} | COMP {c['COMP']['heldout_R2']} | P0 {c['P0']['heldout_R2']}"
              f" | P0+COMP {c['P0+COMP']['heldout_R2']}")

    print("\n=== D  LODO phi_structural prediction ===")
    for tgt in DSES:
        for tag in ["COMP", "P0+COMP"]:
            r = Dd[tgt][tag]
            print(f"  {tgt:13s} {tag:8s} pred {r['pred_mean']:+.4f} true {r['true_mean']:+.4f} R2 {r['target_R2']}")

    print("\n=== E  end-to-end LINEAR_COMPLEMENTARITY_POLICY ===")
    for tag in ["COMP", "P0+COMP"]:
        print(f"  -- {tag} --")
        for tgt in DSES:
            e = E[tag][tgt]
            print(f"    {tgt:13s} NET {e['NET_gold']:+8.1f} / orc {e['target_oracle_NET']:.0f}  meanR {e['mean_R']:5.2f}"
                  f"  waste {e['RESERVE_WASTE_mean']:5.2f}  fracR0 {e['frac_R0']:.3f}  regret {e['UTILITY_REGRET_mean']:.3f}")
    print(f"\n  B1.5 ref: metaqa NET +51 | 2wiki +69 | squad 0 (meanR 8.72)")
    print(f"  B1.7 ref: metaqa NET  +3 | 2wiki +73 | squad 0 (meanR 10.12)")

    if "heldout_dataset_id_accuracy" in str(res["F_IDENTITY_DIAGNOSTIC"]):
        print("\n=== F  dataset-identity predictability ===")
        for tag in ["P0", "COMP", "P0+COMP"]:
            print(f"  {tag:8s} acc {res['F_IDENTITY_DIAGNOSTIC'][tag]['heldout_dataset_id_accuracy']:.4f} (chance .3333)")
    print("\n[b18b] wrote", OUT)


if __name__ == "__main__":
    main()
