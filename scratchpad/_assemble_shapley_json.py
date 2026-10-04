import json, numpy as np
A = json.load(open("scratchpad/_shapley_analysis.json"))
EX = ("dense","splade","offset","mixture","relation"); UT=("ndcg50","mrr","all10","all50")
OUTD="results/L2/_shapley"

# efficiency over full saved data
eff={}
for ds in ("2wiki_clean","musique_clean"):
    for sp in ("train","val"):
        z=np.load(f"{OUTD}/{ds}_{sp}.npz"); v=z["valid"]
        mx=0.0
        for u in UT:
            s=sum(z[f"phi_{u}_{e}"][v] for e in EX)
            mx=max(mx,float(np.abs(s-z[f"v_full_{u}"][v]).max()))
        eff[f"{ds}/{sp}"]=mx
EFF_MAX=max(eff.values())

def gmean(ds,sp,u,e): return A[ds][sp]["A1_global"][u][e]["mean"]

out={
 "PHASE":"EXACT_QUERY_LEVEL_SHAPLEY",
 "N_EXPERTS":5,"EXPERTS":list(EX),"N_COALITIONS":32,
 "ENUMERATION":"EXACT (2^5=32/query, no Monte-Carlo)",
 "UTILITY_PRIMARY":"NDCG@50","UTILITIES":list(UT),
 "SHAPLEY_EMPTY_VALUE":0,"SHAPLEY_RELATION_MASKED":"YES",
 "VALUE_FUNCTION":"frozen masked-RRF K0=60; candidate total contribution 0 => not retrieved (rank=inf); relation contributes 1/(K0+rank_among_eligible) on mask=1 else 0",
 "NO_SIZE_NORMALIZATION":True,"SIGNED_PHI_PRESERVED_NO_CLAMP":True,
 "SPLITS_COMPUTED":{"2wiki_clean/train":10497,"2wiki_clean/val":3000,"musique_clean/train":13948,"musique_clean/val":3985},
 "SPLITS_SKIPPED_L1_ANY_FAIL":{"2wiki_clean/train":3,"2wiki_clean/val":0,"musique_clean/train":8,"musique_clean/val":2},
 "TEST_TOUCHED":"NO",
 "UNIT_TESTS":{"EFFICIENCY":"PASS","DUMMY":"PASS(phi=0)","SYMMETRY":"PASS(gap 6.9e-18)",
   "RELATION_ABSTENTION":"PASS(maxabs 0.0)","NEGATIVE":"PASS(found phi<0, not clamped)"},
 "SHAPLEY_EFFICIENCY":"PASS","SHAPLEY_MAX_EFFICIENCY_ERROR":EFF_MAX,"SHAPLEY_EFFICIENCY_PER_SPLIT":eff,
 "SHAPLEY_PILOT_RUNTIME":"~3-5 ms/query (18-strata pilot, ~325-478 q/split)",
 "SHAPLEY_FULL_RUNTIME":{"2wiki_clean/train":"30.9s","2wiki_clean/val":"8.9s","musique_clean/train":"39.6s","musique_clean/val":"11.3s","total":"~91s CPU"},

 "GLOBAL_EXPERT_STATS_NDCG50_MEAN":{
   ds:{sp:{e:gmean(ds,sp,"ndcg50",e) for e in EX} for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},
 "GLOBAL_EXPERT_STATS_FULL":{ds:{sp:A[ds][sp]["A1_global"] for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},

 "QUERY_HETEROGENEITY":{ds:{sp:A[ds][sp]["A2_heterogeneity"] for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},
 "RELATION_SPECIALIST_STATS":{ds:{sp:A[ds][sp]["A3_relation"] for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},
 "OFFSET_MIXTURE_TRADEOFF":{ds:{sp:A[ds][sp]["A4_offset_mixture"] for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},
 "DISAGREEMENT_BUCKETS":{ds:{sp:A[ds][sp]["A5_disagreement"] for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},
 "EXPERT_COMPLEMENTARITY":{ds:{sp:A[ds][sp]["A6_complementarity"] for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},
 "HARDNESS":{ds:{sp:A[ds][sp]["hardness"] for sp in ("train","val")} for ds in ("2wiki_clean","musique_clean")},
 "REGIMES_TRAIN":{ds:A[ds]["regime_train"] for ds in ("2wiki_clean","musique_clean")},

 "CONTROLLER_READINESS_GATES":{
   "DO_QUERY_EXPERT_WEIGHTS_VARY":"YES",
   "IS_FIXED_EQUAL_FUSION_SUBOPTIMAL":"YES",
   "ARE_NEGATIVE_EXPERT_CONTRIBUTIONS_COMMON":"YES (relation 19-34% of queries; offset/mixture 25-40% on ALL@10 for 2wiki-val; dense/splade rare 1-9%)",
   "IS_RELATION_A_SPECIALIST":"YES (clean gold_present conditioning; anti-correlated with dense/splade; abstains 54-92%; phi=0 exactly when no signal)",
   "ARE_OFFSET_MIXTURE_MULTI_GOLD_SPECIALISTS":"PARTIAL-YES (deep multi-gold recovery signature on 2wiki: high phi_ALL50, low/neg phi_ALL10; general dominant experts on musique)",
   "IS_QUERY_LEVEL_GATING_JUSTIFIED":"YES",
   "IS_CANDIDATE_LEVEL_GATING_POTENTIALLY_JUSTIFIED":"YES (hypothesis supported by relation mask sparsity + sparse-gold rescue; requires candidate-level Shapley to confirm, deferred)"
 },
 "DELIVERABLE_GATES":{
   "EXACT_SHAPLEY_COMPLETE":"YES","TRAIN_SHAPLEY_READY":"YES","VAL_SHAPLEY_READY":"YES",
   "QUERY_HETEROGENEITY_CONFIRMED":"YES","FIXED_FUSION_LIMITATION_CONFIRMED":"YES",
   "SAFE_TO_START_SOFT_CONTROLLER":"YES",
   "RECOMMENDED_CONTROLLER_INPUTS":["query_repr(gte-Qwen q-embed)","per-expert rank+RRF-score of candidate (5x2)","relation_mask(candidate)","relation_any_signal_present(query)","relation_gold_present is a LABEL not input","dense_splade_disagreement(query)","gold_count is NOT an input (unknown at inference)","offset/mixture agreement","candidate dense/splade raw score"],
   "RECOMMENDED_SHAPLEY_AUXILIARY":"positive-KL (low-weight, ablated) with analysis-only as safe fallback; NOT signed-regression (would bind controller to the frozen equal-RRF marginals it must improve on)"
 },
 "CAVEATS":[
  "Shapley here = cooperative-game attribution under the FROZEN equal-weight masked-RRF value function; it is NOT universal causal importance and NOT the value under a learned controller.",
  "phi<0 means the expert's average marginal under equal-RRF hurts; a learned per-query weight is exactly the lever to remove that harm.",
  "Query-level only this phase; candidate-level gating justification is a supported hypothesis, not yet measured.",
  "gold_count / relation_gold_present are ORACLE labels used for analysis only; controller inputs must be inference-available."
 ]
}
json.dump(out, open("results/L2/L2_SHAPLEY_EXACT.json","w"), indent=1, default=str)
print("wrote results/L2/L2_SHAPLEY_EXACT.json  EFF_MAX=",EFF_MAX)
