"""G2 B1.7 — MARGINAL-UTILITY RESERVE POLICY (3-target pilot; STOP before six-way).

B1.5/B1.6 used a CLASS-BALANCED 4-way R classifier for what is really a conservative resource-allocation problem — the
balancing itself may encourage over-reservation. B1.7 REFRAMES THE SUPERVISION: no class balancing, no new features, no
new architecture/scorer/encoder, no MLP. Everything frozen (universe, verification=BASE_FUSED, s_dir ordering,
two_channel_pool, TOP_POOL=50, R in {0,4,8,16}, the exact 17 P0 features).

Idea: instead of "which R?", ask "is spending the NEXT structural block expected to help?".
Per TRAIN query compute U0,U4,U8,U16 (same deterministic two-channel pool, TRAIN gold). Marginal utilities
  D4 = U4 - U0,  D8 = U8 - U4,  D16 = U16 - U8.
Three simple source-trained LINEAR regressors g4(P0)->D4, g8(P0)->D8, g16(P0)->D16 (no hidden layer, no class balance;
dataset-balanced QUERY weighting only, so one corpus can't dominate by count; natural +/0/- prevalence preserved).
Sequential inference: R=0 default; spend block k only if predicted D_k > tau_k (thresholds source-TRAIN-selected,
biased conservatively toward stopping; near-tie epsilon). Structural capacity must EARN admission.

UTILITY EQUATION (transparent, identical across targets):
  pool_R = two_channel_pool(fused, s_dir, added_mask, R)   # deterministic, NO gold used in construction
  U(R)   = #golds(pool_R) - #golds(pool_0)                 # = golds_gained(R) - golds_evicted(R)
  pool_0 = fused top-50 (the verification base). Every evicted gold is a retrieval-visible retention loss weighted 1 =>
  the retention penalty is built into U (eviction is subtracted 1:1). U0 = 0 by construction.

Reports quality metrics + REVISED policy metrics (utility regret, reserve waste, under/over-reserve cost) + a
SOURCE-COVERAGE AUDIT (kNN support in P0 space + per-regime bin support) to separate POLICY_OBJECTIVE_FAILURE from
SOURCE_SUPPORT_FAILURE. TEST untouched; 0 encoder passes; six-way NOT launched.
"""
import os, sys, json, time, itertools
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np
from _b14_twochannel import two_channel_pool, per_query_idx
from _b15_policy import (load, query_matrix, evaluate_adaptive, summarize, policy_vs_oracle, r_distribution,
                         P0_NAMES, R_VALUES, DSES)

R_VALS = list(R_VALUES)                     # [0,4,8,16]
BLOCKS = [(4, 0), (8, 1), (16, 2)]          # (R_after_block, marginal_index) : D4,D8,D16
EPS = 0.05                                  # TRAIN-frozen near-tie epsilon (expected-gold gain)
TAU_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
B15_JSON = "results/GENERALIZATION/_g2_b15_adaptive.json"
B14_JSON = "results/GENERALIZATION/_g2_b14_twochannel.json"
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)


# ---------------------------------------------------------------- per-query utilities U(R) and marginals
def utilities(d):
    """Return UR (nq,4) = U at R in {0,4,8,16}, and D (nq,3) = [D4,D8,D16]."""
    fused = d["fused"]; sdir = d["sdir"]; added = d["added"]; gold = d["gold"]
    idx = list(per_query_idx(d)); nq = len(idx)
    UR = np.zeros((nq, 4))
    for qi, (i, a, b) in enumerate(idx):
        scq = fused[a:b]; sdq = sdir[a:b]; amq = added[a:b]; yq = gold[a:b]
        base = None
        for ri, R in enumerate(R_VALS):
            pool, _comp = two_channel_pool(scq, sdq, amq, R)
            ng = float(yq[pool].sum())
            if ri == 0:
                base = ng
            UR[qi, ri] = ng - base
    D = np.stack([UR[:, 1] - UR[:, 0], UR[:, 2] - UR[:, 1], UR[:, 3] - UR[:, 2]], 1)
    return UR, D


# ---------------------------------------------------------------- weighted ridge (no class balance; query-balanced)
def wridge(X, y, w, l2=1.0):
    n, dm = X.shape
    Xa = np.concatenate([X, np.ones((n, 1))], 1)          # bias column last
    Wd = w[:, None]
    A = Xa.T @ (Wd * Xa); reg = l2 * np.eye(dm + 1); reg[dm, dm] = 0.0
    beta = np.linalg.solve(A + reg, Xa.T @ (w * y))
    return beta                                            # (dm+1,)


def predict(X, beta):
    return X @ beta[:-1] + beta[-1]


def seq_R(pD, taus):
    """pD = (3,) predicted [D4,D8,D16]; sequential spend-if->stop ladder. R=0 default."""
    R = 0
    for k, (Rk, mi) in enumerate(BLOCKS):
        if pD[mi] > taus[k]:
            R = Rk
        else:
            break
    return R


# ---------------------------------------------------------------- policy quality vs per-query utility oracle
def utility_policy_metrics(UR, Rq):
    """UR (nq,4), Rq (nq,) chosen R. Returns regret/waste/under/over + oracle R array."""
    ri_of = {r: i for i, r in enumerate(R_VALS)}
    maxU = UR.max(1)
    oracleR = np.array([R_VALS[int(np.argmax(UR[i] >= maxU[i] - 1e-12))] for i in range(len(UR))])  # smallest argmax
    # smallest near-optimal R (within EPS of max)
    nearR = np.array([R_VALS[int(np.argmax(UR[i] >= maxU[i] - EPS - 1e-12))] for i in range(len(UR))])
    chosenU = np.array([UR[i, ri_of[int(Rq[i])]] for i in range(len(UR))])
    regret = maxU - chosenU
    waste = np.maximum(0, Rq - nearR)
    under = np.where(Rq < oracleR, regret, 0.0)
    over = np.where(Rq > oracleR, regret, 0.0)
    return {"UTILITY_REGRET_mean": round(float(regret.mean()), 4), "UTILITY_REGRET_sum": round(float(regret.sum()), 1),
            "RESERVE_WASTE_mean": round(float(waste.mean()), 3),
            "UNDER_RESERVE_COST_mean": round(float(under.mean()), 4), "UNDER_RESERVE_COST_sum": round(float(under.sum()), 1),
            "OVER_RESERVE_COST_mean": round(float(over.mean()), 4), "OVER_RESERVE_COST_sum": round(float(over.sum()), 1),
            "mean_chosen_R": round(float(Rq.mean()), 3), "mean_oracle_R": round(float(oracleR.mean()), 3),
            "mean_nearopt_R": round(float(nearR.mean()), 3), "frac_R0": round(float((Rq == 0).mean()), 3)}, oracleR


# ---------------------------------------------------------------- source-coverage audit
def coverage_audit(FEATsrc, FEATtgt, dssrc, dstgt, mu, sd):
    Xs = (FEATsrc - mu) / sd; Xt = (FEATtgt - mu) / sd
    # kNN(10) support in P0 space
    k = 10
    def knn_dist(Q, Ref, self_=False):
        out = np.zeros(len(Q))
        for i in range(len(Q)):
            dd = np.sqrt(((Ref - Q[i]) ** 2).sum(1))
            if self_: dd = np.sort(dd)[1:k + 1]
            else: dd = np.sort(dd)[:k]
            out[i] = dd.mean()
        return out
    # subsample source ref for speed if huge
    ref = Xs
    if len(Xs) > 4000:
        ref = Xs[np.random.permutation(len(Xs))[:4000]]
    src_self = knn_dist(ref, ref, self_=True)
    p95 = float(np.percentile(src_self, 95))
    tgt_d = knn_dist(Xt, ref)
    oos = tgt_d > p95
    # per-regime bin support (axes computed from raw P0 cols)
    # RC=(rc_agree+rc_support+rc_pool_agree)/3 -> idx 0,1,4 ; SRED=1-idx15 ; SNOV=idx7 ; SEED=idx13
    def axes(F):
        RC = (F[:, 0] + F[:, 1] + F[:, 4]) / 3
        return {"retrieval_confidence": RC, "structural_redundancy": 1 - F[:, 15],
                "structural_novelty": F[:, 7], "seed_support": F[:, 13]}
    As, At = axes(FEATsrc), axes(FEATtgt)
    regime = {}
    for ax in As:
        edges = np.quantile(np.concatenate([As[ax], At[ax]]), [0, .2, .4, .6, .8, 1.0]); edges[-1] += 1e-6
        sb = np.clip(np.digitize(As[ax], edges[1:-1]), 0, 4)
        tb = np.clip(np.digitize(At[ax], edges[1:-1]), 0, 4)
        src_frac = [round(float((sb == j).mean()), 3) for j in range(5)]
        tgt_frac = [round(float((tb == j).mean()), 3) for j in range(5)]
        # target mass sitting in bins with <2% source support
        low = np.array([src_frac[j] < 0.02 for j in range(5)])
        tgt_in_lowsupport = round(float(np.mean([low[b] for b in tb])), 3)
        regime[ax] = {"src_frac_by_bin": src_frac, "tgt_frac_by_bin": tgt_frac,
                      "tgt_mass_in_<2%_source_bins": tgt_in_lowsupport}
    return {"src_self_kNN_p95": round(p95, 3), "tgt_frac_out_of_support": round(float(oos.mean()), 3),
            "regime_bins": regime}, oos


# ---------------------------------------------------------------- main
def main():
    log(f"=== B1.7 MARGINAL-UTILITY RESERVE POLICY  {DSES}  R={R_VALS} ===")
    DATA = {ds: load(ds) for ds in DSES}
    SC = {ds: DATA[ds]["fused"].copy() for ds in DSES}       # verification = BASE_FUSED
    FEAT = {ds: query_matrix(DATA[ds], use_p1=False) for ds in DSES}
    UR = {}; D = {}
    for ds in DSES:
        UR[ds], D[ds] = utilities(DATA[ds])
        log(f"utilities {ds}: nq={len(UR[ds])} meanU16={UR[ds][:,3].mean():.3f} "
            f"D>0 frac [D4,D8,D16]={[round(float((D[ds][:,j]>0).mean()),3) for j in range(3)]} "
            f"D=0 frac={[round(float((D[ds][:,j]==0).mean()),3) for j in range(3)]}")

    b15 = json.load(open(B15_JSON)); b14 = json.load(open(B14_JSON))
    FOLDS = {}; VERD = {}
    for target in DSES:
        src = [x for x in DSES if x != target]
        Xtr = np.concatenate([FEAT[s] for s in src]); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        Xs = (Xtr - mu) / sd; Xt = (FEAT[target] - mu) / sd
        Dtr = np.concatenate([D[s] for s in src])
        URtr = np.concatenate([UR[s] for s in src])
        # dataset-balanced query weights (no class balancing)
        cnt = {s: len(FEAT[s]) for s in src}; w = np.concatenate([np.full(cnt[s], 1.0 / cnt[s]) for s in src])
        w = w / w.mean()
        # three marginal-gain regressors
        betas = [wridge(Xs, Dtr[:, j], w, l2=1.0) for j in range(3)]
        pD_src = np.stack([predict(Xs, betas[j]) for j in range(3)], 1)
        pD_tgt = np.stack([predict(Xt, betas[j]) for j in range(3)], 1)

        # source-TRAIN threshold selection: maximize source realized utility, conservative tie-break (larger sum tau)
        ri_of = {r: i for i, r in enumerate(R_VALS)}
        best = None
        for taus in itertools.product(TAU_GRID, repeat=3):
            Rq = np.array([seq_R(pD_src[i], taus) for i in range(len(pD_src))])
            realized = sum(URtr[i, ri_of[int(Rq[i])]] for i in range(len(Rq)))
            key = (realized, sum(taus))                       # maximize utility, then larger taus (more conservative)
            if best is None or key > best[0]:
                best = (key, taus)
        taus = best[1]
        # apply frozen policy to target
        Rq = np.array([seq_R(pD_tgt[i], taus) for i in range(len(pD_tgt))])

        m = summarize(evaluate_adaptive(DATA[target], SC[target], Rq))
        polq, oracleR = utility_policy_metrics(UR[target], Rq)
        cov, oos = coverage_audit(Xtr := np.concatenate([FEAT[s] for s in src]), FEAT[target], src, target, mu, sd)

        # in-support vs out-of-support failure split (waste + regret)
        maxU = UR[target].max(1)
        chosenU = np.array([UR[target][i, ri_of[int(Rq[i])]] for i in range(len(Rq))])
        regret = maxU - chosenU
        nearR = np.array([R_VALS[int(np.argmax(UR[target][i] >= maxU[i] - EPS - 1e-12))] for i in range(len(Rq))])
        waste = np.maximum(0, Rq - nearR)
        def msplit(mask):
            if mask.sum() == 0: return {"n": 0}
            return {"n": int(mask.sum()), "reserve_waste_mean": round(float(waste[mask].mean()), 3),
                    "utility_regret_mean": round(float(regret[mask].mean()), 4),
                    "mean_R": round(float(Rq[mask].mean()), 3)}
        support_split = {"in_support": msplit(~oos), "out_of_support": msplit(oos)}

        FOLDS[target] = {"selected_taus": {"tau4": taus[0], "tau8": taus[1], "tau16": taus[2]},
                         "metrics": m, "R_distribution": r_distribution(Rq),
                         "policy_vs_oracle_discrete": policy_vs_oracle(Rq, oracleR),
                         "policy_utility_metrics": polq, "coverage_audit": cov,
                         "failure_by_support": support_split,
                         "regressor_bias_(intercept)": [round(float(b[-1]), 3) for b in betas],
                         "pred_D_target_mean": [round(float(pD_tgt[:, j].mean()), 3) for j in range(3)],
                         "true_D_target_mean": [round(float(D[target][:, j].mean()), 3) for j in range(3)]}
        VERD[target] = {"NET": m["NET_GOLD_GAIN@50"], "GO_adm": m["GRAPH_ONLY_GOLD_ADMISSION@50"],
                        "RV_ret": m["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"], "GOLD_RECALL@50": m.get("GOLD_RECALL@50"),
                        "ALL@50": m.get("ALL@50"), "POOL_CHURN": m.get("POOL_CHURN_mean"),
                        "mean_R": polq["mean_chosen_R"], "frac_R0": polq["frac_R0"],
                        "reserve_waste": polq["RESERVE_WASTE_mean"], "utility_regret": polq["UTILITY_REGRET_mean"]}
        log(f"[{target}] taus={taus} NET={m['NET_GOLD_GAIN@50']} GO={m['GRAPH_ONLY_GOLD_ADMISSION@50']:.4f} "
            f"RVret={m['RETRIEVAL_VISIBLE_GOLD_RETENTION@50']:.3f} meanR={polq['mean_chosen_R']} fracR0={polq['frac_R0']} "
            f"waste={polq['RESERVE_WASTE_mean']} regret={polq['UTILITY_REGRET_mean']} "
            f"OOS={cov['tgt_frac_out_of_support']}")

    # ---- comparison baselines pulled from B1.5/B1.4 ----
    baselines = {}
    for t in DSES:
        b15f = b15["FOLDS"][t]["QUERY_LOCAL_P0_LINEAR"]
        baselines[t] = {
            "FIXED_R0": {"NET": 0, "note": "reserve nothing (verification-only base)"},
            "B15_P0_CLASSIFIER": {"NET": b15f["metrics"]["NET_GOLD_GAIN@50"],
                                  "GO_adm": b15f["metrics"]["GRAPH_ONLY_GOLD_ADMISSION@50"],
                                  "RV_ret": b15f["metrics"]["RETRIEVAL_VISIBLE_GOLD_RETENTION@50"],
                                  "mean_R": b15f["policy_vs_target_oracle"]["mean_chosen_R"],
                                  "over_reserve": b15f["policy_vs_target_oracle"]["OVER_RESERVE_RATE"]}}

    # ---- DECISION (CASE A / B / C) ----
    def g(t, k): return VERD[t][k]
    # SQuAD: want lower mean R + lower reserve waste + more R=0, relevance preserved (RV_ret high, NET unchanged=0)
    sq_meanR_b15 = 8.72; sq_over_b15 = 0.614
    squad_waste_reduced = g("squad_clean", "mean_R") < sq_meanR_b15 - 1.0 and g("squad_clean", "reserve_waste") < 8.0
    squad_relevance_ok = g("squad_clean", "RV_ret") >= 0.99 and g("squad_clean", "NET") >= -2
    # MetaQA: want lower utility regret + >=B15 GO-adm/NET
    meta_better = (g("metaqa", "NET") >= baselines["metaqa"]["B15_P0_CLASSIFIER"]["NET"]) and \
                  (g("metaqa", "GO_adm") >= baselines["metaqa"]["B15_P0_CLASSIFIER"]["GO_adm"] - 1e-9)
    # 2Wiki: preserve
    wiki_ok = g("2wiki_clean", "NET") >= 0 and g("2wiki_clean", "RV_ret") >= 0.95
    caseA = bool(squad_waste_reduced and squad_relevance_ok and meta_better and wiki_ok)

    # failure-vs-support: does residual failure concentrate OOS?
    def waste_ratio(t):
        s = FOLDS[t]["failure_by_support"]
        io = s["in_support"].get("reserve_waste_mean", 0.0); oo = s["out_of_support"].get("reserve_waste_mean", 0.0)
        return io, oo, s["in_support"].get("n", 0), s["out_of_support"].get("n", 0)
    sq_io, sq_oo, sq_in, sq_on = waste_ratio("squad_clean")
    squad_fail_is_oos = (sq_on > 0) and (sq_oo > sq_io + 1.0) and (sq_in == 0 or sq_io < 4.0)
    squad_fail_in_support = (sq_in > 0) and (sq_io >= 4.0)

    if caseA:
        flag = "UTILITY_FORMULATION_WAS_THE_PRIMARY_BOTTLENECK=YES"
        interp = ("Marginal-utility (unbalanced) reframing materially cut SQuAD reserve waste and MetaQA utility regret "
                  "while preserving 2Wiki -> the class-balanced classifier objective was the primary bottleneck. B1 much "
                  "closer to solved. Do NOT enlarge capacity; six-way NOT auto-launched.")
        remaining = "OBJECTIVE"
    elif squad_fail_in_support or (not squad_waste_reduced and not squad_fail_is_oos):
        flag = "CURRENT_QUERY_LOCAL_FEATURES_ARE_INSUFFICIENT=YES"
        interp = ("Policy still over-reserves SQuAD even on target queries WITH good source support -> the deficiency is "
                  "not objective and not coverage but the P0 feature representation. Do NOT proceed to six-way blindly; "
                  "features cannot separate redundant-but-anchored from useful-novel even where the source covers the "
                  "region.")
        remaining = "FEATURE INSUFFICIENCY"
    else:
        flag = "SOURCE_COVERAGE_IS_THE_REMAINING_BOTTLENECK=YES"
        interp = ("Residual policy failure concentrates on target queries with little source feature-space support "
                  "(SQuAD's confident+redundant regime is under-represented in the graph-rich source pair). Objective "
                  "reframing helped but cannot manufacture the missing regime. Next justified step = broaden source "
                  "regimes with MuSiQue/Hotpot/WebQSP BEFORE concluding transfer impossible. Do NOT add model capacity; "
                  "six-way NOT auto-launched.")
        remaining = "SOURCE COVERAGE"

    verdict = {"B17": VERD, "baselines": baselines,
               "squad_diag": {"b15_mean_R": sq_meanR_b15, "b15_over_reserve": sq_over_b15,
                              "b17_mean_R": g("squad_clean", "mean_R"), "b17_frac_R0": g("squad_clean", "frac_R0"),
                              "b17_reserve_waste": g("squad_clean", "reserve_waste"),
                              "b17_RV_ret": g("squad_clean", "RV_ret"), "b17_NET": g("squad_clean", "NET"),
                              "waste_reduced": bool(squad_waste_reduced), "relevance_ok": bool(squad_relevance_ok)},
               "metaqa_diag": {"b15_NET": baselines["metaqa"]["B15_P0_CLASSIFIER"]["NET"], "b17_NET": g("metaqa", "NET"),
                               "b15_GO": baselines["metaqa"]["B15_P0_CLASSIFIER"]["GO_adm"], "b17_GO": g("metaqa", "GO_adm"),
                               "b17_utility_regret": g("metaqa", "utility_regret"),
                               "b17_3hop": FOLDS["metaqa"]["metrics"]["PER_HOP"].get("3", {}),
                               "better_than_b15": bool(meta_better)},
               "support_split_squad": FOLDS["squad_clean"]["failure_by_support"],
               "CASE_A_objective_was_bottleneck": caseA, "REMAINING_FAILURE_CLASS": remaining,
               "FLAG": flag, "INTERPRETATION": interp,
               "B1_marked_solved": False, "six_way_launched": False}

    out = {"phase": "G2 B1.7 marginal-utility reserve policy; 3-target source-only pilot; TEST untouched; STOP before six-way",
           "datasets": DSES, "R_values": R_VALS, "verification_channel_fixed": "BASE_FUSED",
           "utility_equation": "U(R)=#golds(two_channel_pool(fused,s_dir,added,R)) - #golds(fused_top50); U0=0; "
                               "eviction of retrieval-visible golds subtracted 1:1 (retention penalty built in)",
           "supervision": "three linear marginal-gain regressors g4,g8,g16: P0->D_block; NO class balancing; "
                          "dataset-balanced query weights only; natural +/0/- prevalence preserved",
           "inference": "sequential ladder R=0 default; spend block k iff predicted D_k > tau_k; taus source-TRAIN "
                        "selected (maximize source realized utility, conservative tie-break toward larger tau); "
                        "near-tie epsilon=%.2f" % EPS,
           "P0_features": P0_NAMES, "tau_grid": TAU_GRID,
           "leakage_checks": {"target_excluded_from_training_and_threshold_selection": True,
                              "features_norm_regressors_taus_source_train_only": True,
                              "gold_only_for_TRAIN_utility_labels_and_eval_oracle": True,
                              "no_gold_in_pool_construction": True, "no_class_balancing": True,
                              "no_dataset_id_or_hop_or_metadata_feature": True, "verification_channel_frozen": True,
                              "structural_orderer_frozen_s_dir": True, "no_new_feature_arch_scorer_encoder_mlp": True,
                              "coverage_audit_is_analysis_only_no_target_info_in_policy": True},
           "FOLDS": FOLDS, "VERDICT": verdict, "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO",
           "SIX_WAY_LODO_LAUNCHED": "NO"}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open("results/GENERALIZATION/_g2_b17_marginal.json", "w"), indent=1, default=str)
    log(f"VERDICT flag={flag} caseA={caseA} remaining={remaining} | squad meanR "
        f"{sq_meanR_b15}->{g('squad_clean','mean_R')} fracR0={g('squad_clean','frac_R0')} waste={g('squad_clean','reserve_waste')} "
        f"| meta NET {baselines['metaqa']['B15_P0_CLASSIFIER']['NET']}->{g('metaqa','NET')} regret={g('metaqa','utility_regret')} "
        f"| 2wiki NET {g('2wiki_clean','NET')} RVret={g('2wiki_clean','RV_ret')}")
    log("B17_DONE -> results/GENERALIZATION/_g2_b17_marginal.json")


if __name__ == "__main__":
    main()
