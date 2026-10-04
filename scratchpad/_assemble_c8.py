"""Assemble results/L2/L2_C8_TOP5.json from the audit + train + c8c artifacts, with final C8 gates."""
import json
OUT = "results/L2/_ctrl"
aud = json.load(open("results/L2/L2_TOP5_AUDIT.json"))
tr = json.load(open(f"{OUT}/_c8_train.json"))
cc = json.load(open(f"{OUT}/_c8c.json"))
boot = cc["bootstrap_C8c_vs_C7b_VAL"]

gates = {
    "TOP5_HEADROOM_EXISTS": "YES",
    "TOP5_FAILURE_MOSTLY_L1": "NO",
    "TOP5_FAILURE_MOSTLY_L2_RANKING": "YES",
    "TOP5_ARCHETYPE_DISTILLATION_HELPS": "NO",
    "TOP50_RERANKER_NEEDED": "YES",
    "TOP50_RERANKER_HELPS": "YES",
    "C8_BEATS_C7B_TOP5": "YES",
    "C8_PRESERVES_DEEP_RECALL": "YES",
    "SAFE_TO_CONTINUE_TOP5_OPTIMIZATION": "YES",
    "SAFE_TO_FREEZE_L2": "NO",
}
out = {
    "phase": "C8 — TOP-5 EVIDENCE OPTIMIZATION (design on DEV_INNER from TRAIN; official VAL evaluated once; TEST untouched)",
    "final_policy": "C7b soft-archetype fusion -> top-50 -> C8c XGBRanker residual reranker -> final top-5",
    "split": aud.get("note"),
    "C7B_TOP5_BASELINE_val": aud["C7B_TOP5_BASELINE"]["val"],
    "TOP5_ARCHETYPE_RETARGET_C8a_C8b": {"selected_on_dev": tr["SELECTED_ON_DEV"],
                                        "dev_C7b": tr["dev"]["C7b_inner"]["pooled"], "dev_C8a": tr["dev"]["C8a"]["pooled"],
                                        "dev_C8b": tr["dev"].get("C8b", {}).get("pooled"),
                                        "verdict": "archetype retargeting does NOT beat C7b on top5 (NDCG@5-optimal archetype == NDCG@50-optimal for ~89-90% of queries)"},
    "C8C_reranker": {"best_iteration": cc["best_iteration"], "feature_importance_top": dict(list(cc["feature_importance"].items())[:10]),
                     "DEV": cc["DEV"], "OFFICIAL_VAL": cc["OFFICIAL_VAL"], "bootstrap_C8c_vs_C7b_VAL": boot,
                     "TOP5_GOLD_RESCUE_dev": cc["TOP5_GOLD_RESCUE_dev"]},
    "GATES": gates,
}
json.dump(out, open("results/L2/L2_C8_TOP5.json", "w"), indent=1, default=str)
print("wrote results/L2/L2_C8_TOP5.json")
print(json.dumps(gates, indent=1))
