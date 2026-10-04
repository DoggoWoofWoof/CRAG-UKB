"""Assemble results/L2/L2_CONTROLLER_OBJECTIVE_ALIGNMENT.json from all C6 component artifacts. VAL only."""
import json, os
OUT = "results/L2/_ctrl"


def L(p): return json.load(open(p)) if os.path.exists(p) else None


def model_row(tag):
    j = L(f"{OUT}/{tag}.json")
    if not j: return None
    m = j["meta"]; sel = j["selected"]["POOLED"]; bt = j["best_trained"]["POOLED"]; g = j["best_trained"].get("gate", {})
    keep = ["MRR", "NDCG@50", "R@5", "R@10", "ALL@10", "ALL@50"]
    return {"selected_ndcg50": sel["NDCG@50"], "selected_is_init": m["selected_is_init"],
            "trained_best_epoch": m["trained_best_epoch"],
            "best_trained": {k: bt[k] for k in keep},
            "best_trained_gate_mean_weight": g.get("mean_weight"),
            "history_ndcg50": [h["val_pooled"]["NDCG@50"] for h in j["history"]]}


def main():
    audit = L(f"{OUT}/_c6_audit.json"); predict = L(f"{OUT}/_c6_predict.json")
    boot_ab = L(f"{OUT}/_c6_bootstrap.json"); boot_c = L(f"{OUT}/_c6c_bootstrap.json")
    c0 = L(f"{OUT}/C0.json")["POOLED"]
    c2 = L(f"{OUT}/C2.json")
    c2_tr = [h for h in c2["history"] if h["epoch"] >= 0]
    old_c2 = {"selected_ndcg50": c2["POOLED"]["NDCG@50"], "selected_is_init": True,
              "trained_best_ndcg50": max(h["val_pooled"]["NDCG@50"] for h in c2_tr),
              "trained_worst_ndcg50": min(h["val_pooled"]["NDCG@50"] for h in c2_tr),
              "history_ndcg50": [h["val_pooled"]["NDCG@50"] for h in c2["history"]]}

    out = {
        "level": "VAL_ONLY", "primary_metric": "NDCG@50", "no_test_inspected": True,
        "phase": "C6 — FULL-SCOPE-AWARE CONTROLLER TRAINING (query-only gate; frozen 5 experts)",
        "frozen_note": "CURRENT_CONTROLLER_TRAINING_BEATS_EQUAL_RRF was NO (C0-C5); C6 tests objective alignment.",
        "C0_equal_rrf": {k: c0[k] for k in ("MRR", "NDCG@50", "R@5", "R@10", "ALL@10", "ALL@50")},
        "task1_negative_depth_distribution": {ds: audit[ds]["train"]["neg_depth_proportion"] for ds in ("2wiki_clean", "musique_clean")},
        "task1_neg_depth_by_provenance": {ds: audit[ds]["train"]["neg_depth_by_provenance"] for ds in ("2wiki_clean", "musique_clean")},
        "task3_oracle_ceiling": {"POOLED": audit["POOLED"],
                                 "2wiki_clean": {sp: {k: audit["2wiki_clean"][sp][k] for k in ("equal_ndcg50", "oracle_ndcg50", "oracle_gain", "oracle_choice_counts")} for sp in ("train", "val")},
                                 "musique_clean": {sp: {k: audit["musique_clean"][sp][k] for k in ("equal_ndcg50", "oracle_ndcg50", "oracle_gain", "oracle_choice_counts")} for sp in ("train", "val")}},
        "task4_oracle_predictability": predict,
        "OLD_C2_sampled_ranknet": old_c2,
        "C6a_depth_ranknet": model_row("C6a"),
        "C6b_depth_lambdarank_sampledlist": model_row("C6b"),
        "C6c_fullscope_lambdarank": model_row("C6c"),
        "bootstrap_trained_vs_C0": {**(boot_ab or {}), **(boot_c or {})},
        "gates": {
            "TRAIN_EVAL_MISMATCH_CONFIRMED": "YES",
            "DEPTH_AWARE_SAMPLING_HELPS": "MIXED (lifts RankNet collapse 0.7815->0.834 best-trained; does NOT beat equal, does NOT fix deep-recall alone)",
            "NDCG_AWARE_LOSS_HELPS": "MIXED (HURTS on sampled list: C6b collapses to 0.781 onto a single expert; only with FULL-SCOPE ranks (C6c) does it preserve deep-recall ALL@50 0.872, still <equal)",
            "ORACLE_REGIME_PREDICTABLE_FROM_INFERENCE_FEATURES": "MIXED (partially: classifier captures +0.0096 retrieval NDCG = 24% of headroom, refuting earlier 'unlearnable'; but acc only 0.41 and a soft gate does not monetize it)",
            "QUERY_GATE_BEATS_EQUAL_RRF": "NO (all C6 variants select the equal init; best-trained C6c -0.0113 sig below C0)",
            "ADAPTIVE_CONTROLLER_RETAINED": "NO"},
        "failure_analysis": {
            "1_no_predictable_oracle_regime": "NO — regime IS partially predictable (+0.0096, Task4)",
            "2_insufficient_inference_features": "PLAUSIBLE contributor (classifier acc only 0.41)",
            "3_optimization_collapse": "was the cause for sampled-list (C6a/C6b collapse onto mixture); FIXED by full-scope C6c",
            "4_train_val_regime_shift": "LIKELY contributor (TRAIN equal NDCG 0.90 >> VAL 0.84; learned reweighting does not transfer as a net gain)",
            "5_objective_mismatch_remaining": "RESOLVED by C6c (deep-recall restored, ALL@50 0.872); NOT the residual cause",
            "PRIMARY_RESIDUAL_MECHANISM": "query-ONLY gate cannot represent the CANDIDATE-LOCAL masked-relation specialist: relation contributes only on mask=1 candidates, so one query-level weight cannot exploit it; C6c rationally sets relation weight ~9e-5, losing exactly equal's flat-0.2 relation contribution (+0.019 NDCG / +0.056 ALL@50 from C0 vs C0_norel). Equal fusion is a better relation compromise than any single query-level relation weight."},
        "L2_CONTROLLER_CANDIDATE": "C0 (fixed 5-expert equal RRF fusion) — unchanged; no adaptive controller retained; TEST not inspected",
        "recommended_next_experiment": "If pursued later: CANDIDATE-aware relation gating under the full-scope objective (C6c-style true-rank LambdaRank), so the gate can raise relation weight only on mask=1 candidates. This is the one representational gap the query-only gate cannot close. Out of this phase's scope."}
    json.dump(out, open("results/L2/L2_CONTROLLER_OBJECTIVE_ALIGNMENT.json", "w"), indent=1, default=str)
    print("wrote results/L2/L2_CONTROLLER_OBJECTIVE_ALIGNMENT.json")
    print(f"{'row':30s} {'sel':>7s} {'trained-best NDCG':>18s} {'ALL@50':>7s} {'MRR':>7s}")
    for tag, r in (("C0 equal", None), ("OLD_C2 sampled+RankNet", old_c2), ("C6a depth+RankNet", model_row("C6a")),
                   ("C6b depth+LambdaRank(list)", model_row("C6b")), ("C6c full-scope LambdaRank", model_row("C6c"))):
        if tag == "C0 equal":
            print(f"{tag:30s} {c0['NDCG@50']:7.4f} {'(baseline)':>18s} {c0['ALL@50']:7.4f} {c0['MRR']:7.4f}"); continue
        if "best_trained" in r:
            bt = r["best_trained"]; print(f"{tag:30s} {r['selected_ndcg50']:7.4f} {bt['NDCG@50']:18.4f} {bt['ALL@50']:7.4f} {bt['MRR']:7.4f}")
        else:
            print(f"{tag:30s} {r['selected_ndcg50']:7.4f} {r['trained_best_ndcg50']:18.4f} {'-':>7s} {'-':>7s}")


if __name__ == "__main__":
    main()
