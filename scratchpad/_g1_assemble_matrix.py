#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Assemble G1_KB_TRANSFER.{md,json} + G1_MATRIX.{md,json} from the per-dataset
G1 eval JSONs. Reads exact numbers; performs NO new modeling. VAL-only; TEST untouched."""
import json, os
R = "results/GENERALIZATION"

def load(p): return json.load(open(p, encoding="utf-8"))

# ---- sources (in-distribution: C11a trained+validated here; no zeroshot/refit split) ----
c12 = load("results/L2/L2_C12_OFFICIAL_VAL.json")["OFFICIAL_VAL_MAIN_TABLE"]["per_ds"]
text = load(f"{R}/G1_TEXT_TRANSFER.json")["datasets"]
web  = load(f"{R}/_g1_eval_webqsp.json")
meta = load(f"{R}/_g1_eval_metaqa.json")

def g(d, *ks, default=None):
    for k in ks:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return default
    return d

# Unified row schema. Metrics: ndcg5, recall5(=gold_recall@5 macro), any5, all5, mrr, all50(deep recall)
def row(name, kind, interp, c8c, zs, rf, extra):
    return {"row": name, "kind": kind, "interpretation": interp,
            "C8c": c8c, "ZERO_SHOT_C11A": zs, "REFIT_C11A": rf, **extra}

def pick(m, ndcg5="ndcg5", r5="recall5_macro", any5="any5", all5="all5", mrr="mrr", all50="all50"):
    if m is None: return None
    return {"ndcg5": m.get(ndcg5), "recall5": m.get(r5), "any5": m.get(any5),
            "all5": m.get(all5, m.get("all5_feas")), "mrr": m.get(mrr), "all50": m.get(all50)}

rows = []

# 1-2 SOURCE (in-distribution). ZERO_SHOT/REFIT == C11a itself (fitted on this distribution).
for ds, disp in [("2wiki_clean", "2Wiki (source)"), ("musique_clean", "MuSiQue (source)")]:
    c8c = pick(c12[ds]["C8c"]); c11 = pick(c12[ds]["C11a"])
    rows.append(row(disp, "SOURCE_TEXT", "IN_DISTRIBUTION (C11a fitted here)",
                    c8c, None, c11,
                    {"note": "source backbone; C11a is the in-distribution fitted model (no zero-shot/refit split)"}))

# 3 SQuAD (text target)
sq = text["squad_clean"]["MAIN_TABLE"]
rows.append(row("SQuAD", "TARGET_TEXT", text["squad_clean"]["GATES"]["INTERPRETATION"],
                pick(sq["C8c"]), pick(sq["ZERO_SHOT_C11A"]), pick(sq["REFIT_C11A"]),
                {"zs_vs_c8c_ndcg5_delta": text["squad_clean"]["ZERO_SHOT_vs_C8c"]["ndcg5"]["delta"],
                 "zs_vs_c8c_sig": text["squad_clean"]["ZERO_SHOT_vs_C8c"]["ndcg5"]["significant"],
                 "refit_vs_c8c_ndcg5_delta": text["squad_clean"]["REFIT_vs_C8c"]["ndcg5"]["delta"]}))

# 4 HotpotQA (text target) — uppercase schema
hp = text["hotpotqa_clean"]["MAIN_TABLE"]
def pick_hp(m):
    return {"ndcg5": m["NDCG@5"], "recall5": m["GOLD_RECALL@5"], "any5": m["ANY@5"],
            "all5": m["ALL@5"], "mrr": m["MRR"], "all50": m["ALL@50"]}
rows.append(row("HotpotQA", "TARGET_TEXT", text["hotpotqa_clean"]["GATES"]["INTERPRETATION"],
                pick_hp(hp["C8c_ZERO_SHOT"]), pick_hp(hp["ZERO_SHOT_C11A"]), pick_hp(hp["REFIT_25K_C11A"]),
                {"zs_vs_c8c_ndcg5_delta": text["hotpotqa_clean"]["ZERO_SHOT_vs_C8c"]["ndcg5"]["delta"],
                 "refit_vs_c8c_ndcg5_delta": text["hotpotqa_clean"]["REFIT_vs_C8c"]["ndcg5"]["delta"],
                 "L1_P50_CEILING": text["hotpotqa_clean"]["L1_P50_CEILING_val"]}))

# 5 WebQSP (KB target)
wt = web["MAIN_TABLE"]
rows.append(row("WebQSP", "TARGET_KB", web.get("INTERPRETATION", web.get("GATES", {}).get("INTERPRETATION")),
                pick(wt["C8c"]), pick(wt["ZERO_SHOT_C11A"]), pick(wt["REFIT_C11A"]),
                {"P50_SCOPE_COVERAGE": web.get("P50_SCOPE_COVERAGE"),
                 "FAILURE_DECOMPOSITION": {k: {kk: v[kk] for kk in v if kk in
                    ("in_scope_golds","top5","below50","FAIL_below_top20","COND_ON_P50_gold_recall5_micro")}
                    for k, v in web.get("FAILURE_DECOMPOSITION", {}).items()}}))

# 6-8 MetaQA per-hop (KB target)
for hop in ["1", "2", "3"]:
    ph = meta["PER_HOP"][hop]; sysd = ph["systems"]
    def pick_hop(s):
        u = s["UNCONDITIONAL"]
        return {"ndcg5": u["ndcg5"], "recall5": u["recall5_macro"], "any5": u["any5"],
                "all5": u["all5"], "mrr": u["mrr"], "all50": u["all50"]}
    rows.append(row(f"MetaQA-{hop}hop", "TARGET_KB", "PER_HOP",
                    pick_hop(sysd["C8c"]), pick_hop(sysd["ZERO_SHOT_C11A"]), pick_hop(sysd["REFIT_C11A"]),
                    {"P50_ANY_coverage": ph["P50_ANY_coverage"], "P50_ALL_coverage": ph["P50_ALL_coverage"],
                     "golds_expected_in_corpus": ph["golds_expected_in_corpus"],
                     "golds_in_P50_scope": ph["golds_in_P50_scope"], "L1_MISSING_golds": ph["L1_MISSING_golds"],
                     "REFIT_failure_decomp": sysd["REFIT_C11A"]["FAILURE_DECOMP"]}))

matrix = {
 "phase": "G1 cross-dataset generalization matrix. Source backbone=2wiki_clean+musique_clean. VAL only; TARGET TEST NEVER TOUCHED.",
 "locked_architecture": "L1 topology C / P50 (K0=60,K=100,P_MAIN=50) -> 5 frozen experts {Dense,SPLADE,Offset,Mixture,Masked-Relation} -> C7b soft-archetype fusion -> C8c XGBRanker -> C11a interaction-MLP residual. NO dataset-ID/LLM/cross-encoder/Transformer/attention/GNN.",
 "two_questions": {
   "PARAMETER_generalization": "ZERO_SHOT_C11A = exact source-trained C11a weights + source E-standardization, NO target fitting",
   "ARCHITECTURE_generalization": "REFIT_C11A = same arch/hparams/loss/opt, trained on TARGET TRAIN (cap 25000), selected on TARGET VAL"},
 "metric_key": "ndcg5, recall5=GOLD_RECALL@5(macro), any5=ANY@5, all5=ALL@5, mrr=MRR, all50=ALL@50(deep-recall ceiling; identical across systems since all rerank within top-50)",
 "ROWS": rows,
 "TARGET_TEST_TOUCHED": "NO", "NEW_ENCODER_FORWARD_PASSES": 0, "NEW_LLM_COMPONENTS": 0,
}
os.makedirs(R, exist_ok=True)
json.dump(matrix, open(f"{R}/G1_MATRIX.json", "w", encoding="utf-8"), indent=1)
print("wrote G1_MATRIX.json,", len(rows), "rows")

# ---------- Markdown matrix ----------
def f(x): return "—" if x is None else f"{x:.4f}"
def line(r, model, m):
    if m is None: return None
    return f"| {r['row']} | {r['kind']} | {model} | {f(m['ndcg5'])} | {f(m['recall5'])} | {f(m['any5'])} | {f(m['all5'])} | {f(m['mrr'])} | {f(m['all50'])} |"

L = []
L.append("# G1 — Cross-Dataset Generalization Matrix\n")
L.append("**Source backbone:** 2wiki_clean + musique_clean. **VAL only; TARGET TEST NEVER TOUCHED.** No new encoder passes, no LLM.\n")
L.append("Frozen architecture: L1 topology-C / P50 → 5 frozen experts → C7b fusion → C8c XGBRanker → **C11a interaction-MLP** (the transfer object).\n")
L.append("Two questions: **ZERO_SHOT_C11A** = source weights, no target fit (PARAMETER transfer). **REFIT_C11A** = same arch, target-trained ≤25k (ARCHITECTURE transfer).\n")
L.append("\n## Main table (NDCG@5 / GoldRecall@5 / ANY@5 / ALL@5 / MRR / ALL@50)\n")
L.append("| Dataset | Kind | Model | NDCG@5 | GR@5 | ANY@5 | ALL@5 | MRR | ALL@50 |")
L.append("|---|---|---|--:|--:|--:|--:|--:|--:|")
for r in rows:
    for model, key in [("C8c", "C8c"), ("ZERO_SHOT_C11A", "ZERO_SHOT_C11A"), ("REFIT_C11A", "REFIT_C11A")]:
        ln = line(r, model, r[key])
        if ln: L.append(ln)
    L.append("|  |  |  |  |  |  |  |  |  |")

L.append("\n## Interpretation per row\n")
L.append("| Dataset | Kind | Interpretation | Deep-recall ceiling (ALL@50) |")
L.append("|---|---|---|--:|")
for r in rows:
    a50 = r["REFIT_C11A"]["all50"] if r["REFIT_C11A"] else None
    L.append(f"| {r['row']} | {r['kind']} | {r['interpretation']} | {f(a50)} |")

# KB scope diagnostics
L.append("\n## KB scope diagnostics (why deep recall is the ceiling)\n")
L.append("MetaQA per-hop P50 coverage — ANY stays high, ALL collapses with hop-count (multi-hop chain endpoints reachable individually but not co-scoped):\n")
L.append("| Hop | P50 ANY cov | P50 ALL cov | golds expected | golds in P50 | L1-missing golds | REFIT NDCG@5 | REFIT ALL@50 |")
L.append("|---|--:|--:|--:|--:|--:|--:|--:|")
for hop in ["1","2","3"]:
    ph = meta["PER_HOP"][hop]; rf = ph["systems"]["REFIT_C11A"]["UNCONDITIONAL"]
    L.append(f"| {hop}-hop | {ph['P50_ANY_coverage']:.4f} | {ph['P50_ALL_coverage']:.4f} | "
             f"{ph['golds_expected_in_corpus']} | {ph['golds_in_P50_scope']} | {ph['L1_MISSING_golds']} | "
             f"{rf['ndcg5']:.4f} | {rf['all50']:.4f} |")
# WebQSP query_meta lacks the RAW/mapped keys -> P50_SCOPE_COVERAGE raw/mapped are 0/negative and NOT usable.
# Authoritative source for WebQSP = FAILURE_DECOMPOSITION (from _goldranks).
wfd = web["FAILURE_DECOMPOSITION"]["REFIT_C11A"]
L.append(f"\nWebQSP (n_val={web['n_val']}): of {wfd['in_scope_golds']} in-P50-scope golds, "
         f"{wfd['below50']} ({100*wfd['below50']/wfd['in_scope_golds']:.1f}%) rank below top-50 even after rerank — "
         f"i.e. present in the ~5k P50 scope but never lifted into the deep top-50. "
         f"REFIT top-5 golds = {wfd['top5']} (vs C8c {web['FAILURE_DECOMPOSITION']['C8c']['top5']}). "
         f"REFIT ALL@50 = {web['MAIN_TABLE']['REFIT_C11A']['all50']:.4f} — the deep-recall ceiling is set upstream at L1/P50, not by L2 ranking. "
         f"(P50_SCOPE_COVERAGE raw/mapped keys are absent from WebQSP query_meta, so only the decomposition above is authoritative.)\n")

open(f"{R}/G1_MATRIX.md", "w", encoding="utf-8").write("\n".join(L))
print("wrote G1_MATRIX.md")

# ================= G1_KB_TRANSFER.{json,md} — KB-specific failure analysis =================
webfd = web["FAILURE_DECOMPOSITION"]
kb = {
 "phase": "G1 KB transfer (WebQSP -> MetaQA). Same FROZEN architecture as text pilots. VAL only; TARGET TEST NEVER TOUCHED.",
 "source_backbone": ["2wiki_clean", "musique_clean"],
 "integrity": {"TARGET_TEST_TOUCHED": "NO", "NEW_ENCODER_FORWARD_PASSES": 0, "NEW_LLM_COMPONENTS": 0,
               "PATH_OR_GRAPH_ADDED": "NO", "ARCHITECTURE_CHANGED": "NO"},
 "WEBQSP": {
   "n_val": web["n_val"],
   "MAIN_TABLE": web["MAIN_TABLE"],
   "FAILURE_DECOMPOSITION": webfd,
   "INTERPRETATION": web.get("INTERPRETATION"),
   "DIAGNOSIS": (
     "Golds are IN the P50 scope but L2 deep-recall is the bottleneck: "
     f"{webfd['REFIT_C11A']['below50']}/{webfd['REFIT_C11A']['in_scope_golds']} "
     f"({100*webfd['REFIT_C11A']['below50']/webfd['REFIT_C11A']['in_scope_golds']:.1f}%) in-scope golds never enter the top-50. "
     "REFIT_C11A roughly doubles top-5 golds vs C8c (143 vs 67) => ARCHITECTURE_GENERALIZES, but the absolute "
     "ceiling (ALL@50=0.2525) is fixed upstream by the dense+splade experts' inability to surface KB-entity golds into "
     "the deep list. This is a REPRESENTATION_SHIFT at the expert level, not an L2 fusion/ranking failure."),
   "FAILURE_SPLIT": {
     "candidate_absent_from_P50": "not separable from decomp (webqsp query_meta lacks RAW/mapped keys); _goldranks covers in-scope only",
     "present_but_below_top20": webfd["REFIT_C11A"]["FAIL_below_top20"],
     "top20_but_below_top5": webfd["REFIT_C11A"].get("FAIL_top20_below_top5"),
     "top5": webfd["REFIT_C11A"]["top5"]},
 },
 "METAQA": {
   "n_val": meta["n_val"],
   "MAIN_TABLE_pooled": meta["MAIN_TABLE"],
   "POOLED_INTERPRETATION": meta["INTERPRETATION"],
   "POOLED_INTERPRETATION_CAVEAT": (
     "Pooled INTERPRETATION=PARAMETERS_GENERALIZE is an ARTIFACT of hop-mixing: ZERO_SHOT beats C8c only marginally "
     "(NDCG@5 +0.0041) because 2-/3-hop are near-floor for every system so source weights cannot hurt, while 1-hop "
     "shares the source offset structure. The genuine, large, consistent lever everywhere is REFIT (architecture), "
     "not parameter reuse. Read PER-HOP, not pooled."),
   "PER_HOP": {h: {
       "P50_ANY_coverage": meta["PER_HOP"][h]["P50_ANY_coverage"],
       "P50_ALL_coverage": meta["PER_HOP"][h]["P50_ALL_coverage"],
       "golds_expected_in_corpus": meta["PER_HOP"][h]["golds_expected_in_corpus"],
       "golds_in_P50_scope": meta["PER_HOP"][h]["golds_in_P50_scope"],
       "L1_MISSING_golds": meta["PER_HOP"][h]["L1_MISSING_golds"],
       "REFIT_UNCONDITIONAL": meta["PER_HOP"][h]["systems"]["REFIT_C11A"]["UNCONDITIONAL"],
       "REFIT_CONDITIONAL_ON_P50": meta["PER_HOP"][h]["systems"]["REFIT_C11A"]["CONDITIONAL_ON_P50"],
       "REFIT_FAILURE_DECOMP": meta["PER_HOP"][h]["systems"]["REFIT_C11A"]["FAILURE_DECOMP"],
     } for h in ["1", "2", "3"]},
   "DIAGNOSIS": (
     "1-hop is a clean architecture win (REFIT NDCG@5 0.694, ANY@5 0.758, ALL@5 0.626; P50 ANY/ALL cov 99.85/99.59). "
     "2-hop and 3-hop COLLAPSE, and the mechanism is UPSTREAM SCOPE, not L2 ranking: P50 ANY coverage stays high "
     "(96.3/94.7%) but P50 ALL coverage falls to 71.4/27.0% — the multi-hop chain endpoints are each individually "
     "reachable yet not CO-SCOPED into the same ~5k P50 window, so ALL@k is unattainable at any rerank depth "
     "(3-hop ALL@50 is only 0.0537). The REFIT failure decomp confirms the residual is not ranking: at 3-hop only 210 "
     "golds are top20-but-below-top5 vs 73216 in-scope-below-top20 and 111398 L1-missing."),
   "MISSING_RELATIONAL_STRUCTURE": "LIKELY (2-hop and 3-hop): evidence present-but-not-co-scoped; recorded, NOT repaired (no path/graph added, per authorization).",
 },
 "CROSS_CUTTING_CONCLUSION": (
   "PARAMETER transfer does not hold as a real effect on KB (WebQSP neutral; MetaQA pooled +0.0041 is a hop-mixing "
   "artifact). ARCHITECTURE transfer (REFIT_C11A) is the consistent lever across BOTH KB targets exactly as on text: "
   "it significantly beats C8c on WebQSP (NDCG@5 0.0847->0.1924) and on MetaQA-1hop (0.323->0.694). The KB ceiling is "
   "set UPSTREAM (L1/P50 scope + expert representation), not by the L2 reranker: WebQSP 73.5% of in-scope golds never "
   "reach top-50, and MetaQA multi-hop ALL-coverage collapses with hop-count. G1 verdict matches the text pilots: "
   "the C11a interaction-MLP ARCHITECTURE is a transferable retrieval principle; its PARAMETERS are distribution-specific."),
}
json.dump(kb, open(f"{R}/G1_KB_TRANSFER.json", "w", encoding="utf-8"), indent=1)
print("wrote G1_KB_TRANSFER.json")

K = []
K.append("# G1 — KB Transfer (WebQSP → MetaQA)\n")
K.append("Same **frozen** architecture as the text pilots (L1-C/P50 → 5 experts → C7b → C8c → C11a). "
         "**No path/graph added. No architecture change. VAL only; TARGET TEST NEVER TOUCHED. 0 encoder passes, 0 LLM.**\n")
K.append("\n## WebQSP\n")
K.append(f"n_val = {web['n_val']}. Interpretation: **{web.get('INTERPRETATION')}**.\n")
K.append("| Model | NDCG@5 | GR@5 | ANY@5 | ALL@5 | MRR | NDCG@50 | ALL@50 |")
K.append("|---|--:|--:|--:|--:|--:|--:|--:|")
for m in ["C7b_fusion","C8c","ZERO_SHOT_C11A","REFIT_C11A"]:
    r = web["MAIN_TABLE"][m]
    K.append(f"| {m} | {r['ndcg5']:.4f} | {r['recall5_macro']:.4f} | {r['any5']:.4f} | {r['all5']:.4f} | "
             f"{r['mrr']:.4f} | {r['ndcg50']:.4f} | {r['all50']:.4f} |")
K.append(f"\n**Failure split (REFIT_C11A, {webfd['REFIT_C11A']['in_scope_golds']} in-scope golds):** "
         f"top5={webfd['REFIT_C11A']['top5']}, present-but-below-top20={webfd['REFIT_C11A']['FAIL_below_top20']}, "
         f"below-top50={webfd['REFIT_C11A']['below50']} "
         f"({100*webfd['REFIT_C11A']['below50']/webfd['REFIT_C11A']['in_scope_golds']:.1f}%).\n")
K.append(f"\n{kb['WEBQSP']['DIAGNOSIS']}\n")
K.append("\n## MetaQA (per hop — the critical breakdown)\n")
K.append("| Hop | P50 ANY | P50 ALL | REFIT NDCG@5 | REFIT ANY@5 | REFIT ALL@5 | REFIT MRR | ALL@50 | L1-missing golds |")
K.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for h in ["1","2","3"]:
    ph = meta["PER_HOP"][h]; u = ph["systems"]["REFIT_C11A"]["UNCONDITIONAL"]
    K.append(f"| {h}-hop | {ph['P50_ANY_coverage']:.4f} | {ph['P50_ALL_coverage']:.4f} | {u['ndcg5']:.4f} | "
             f"{u['any5']:.4f} | {u['all5']:.4f} | {u['mrr']:.4f} | {u['all50']:.4f} | {ph['L1_MISSING_golds']} |")
K.append(f"\n{kb['METAQA']['DIAGNOSIS']}\n")
K.append(f"\n**MISSING_RELATIONAL_STRUCTURE:** {kb['METAQA']['MISSING_RELATIONAL_STRUCTURE']}\n")
K.append(f"\n**Pooled-interpretation caveat:** {kb['METAQA']['POOLED_INTERPRETATION_CAVEAT']}\n")
K.append("\n## Cross-cutting conclusion\n")
K.append(kb["CROSS_CUTTING_CONCLUSION"] + "\n")
open(f"{R}/G1_KB_TRANSFER.md", "w", encoding="utf-8").write("\n".join(K))
print("wrote G1_KB_TRANSFER.md")
