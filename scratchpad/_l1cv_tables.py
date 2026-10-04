"""Emit TABLES.md for the conversion phase straight from the diag JSON (no hand transcription)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1cv_core as CV

D = f"{CV.CVD}/diag"
M = json.load(open(f"{D}/main.json"))
W = json.load(open(f"{D}/why.json"))
F = json.load(open(f"{D}/fin.json"))
G = json.load(open(f"{D}/gate.json"))
NAME = {"metaqa": "metaqa", "webqsp": "webqsp", "2wiki_clean": "2wiki",
        "musique_clean": "musique", "hotpotqa_clean": "hotpot", "squad_clean": "squad"}
OD = [d for d in CV.DSETS if d in M]
L = []
w = L.append


def tbl(hdr, rows):
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        assert len(r) == len(hdr), (len(r), len(hdr), r)
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


def sg(x):
    return "**sig**" if x else "ns"


def f4(x):
    return f"{x:.4f}"


w("# L1 CONVERSION PHASE -- TABLES\n")
w("Generated from `diag/gate.json`, `diag/main.json`, `diag/why.json`, `diag/fin.json`. "
  "`SAFE` = the frozen `R0 / B6_S4_F6_Ms64_Mr32` reference.\n")

w("## T1 -- STEP 1 symmetric U_RANK\n")
tbl(["corpus", "mean len(U)", "incumbents carrying u_rank", "U_PC5 proposals/query",
     "truly-new offered/query"],
    [[NAME[d], M[d]["mean_U_len"], M[d]["mean_U_inside_base50"], M[d]["mean_U_PC5_proposals"],
      M[d]["mean_new_proposals_offered"]] for d in OD])
w("Determinism (MetaQA, two independent builds): `" + str(G["deterministic"]) + "`. "
  "All 50 incumbents receive a u_rank, so incumbents and challengers are scored under identical "
  "rank semantics.\n")

w("## T2 -- STEP 5 bookends (MetaQA)\n")
BK = ["SAFE", "CURRENT_POOL_ORACLE", "U_PC5_POOL_ORACLE", "UNLIMITED_POOL_ORACLE",
      "FULL_UNIVERSE_P50_ORACLE"]
tbl(["slice"] + [b.replace("_", " ").lower() for b in BK],
    [[k] + [f4(M["metaqa"]["BOOKENDS"][k][b]) for b in BK]
     for k in ["hop1", "hop2", "hop3", "ALL"]])

w("## T3 -- STEP 5 MetaQA primary, per hop\n")
rows = []
for r in CV.RULES:
    e = M["metaqa"]["RULES"][r]
    for k in ["hop1", "hop2", "hop3", "ALL"]:
        v = e[k]
        rows.append([r, k, f4(v["ACTUAL"]), f"{v['delta_vs_SAFE']:+.4f}",
                     v["newly_covered"], v["newly_uncovered"], f"{v['net']:+d}",
                     f"{v['mcnemar_p']:.4g}", sg(v["sig"])])
tbl(["rule", "slice", "ACTUAL", "delta vs SAFE", "newly covered", "newly uncovered", "net",
     "McNemar p", "sig"], rows)

w("## T4 -- STEP 5 conversion bookkeeping (MetaQA)\n")
tbl(["rule", "mean churn", "new U_PC5 partitions selected", "per query", "queries with 1 or more",
     "gold-bearing U proposals admitted", "gold-bearing incumbents evicted",
     "queries whose P50 changes"],
    [[r, M["metaqa"]["RULES"][r]["mean_churn"],
      M["metaqa"]["RULES"][r]["NEW_PROPOSALS_SELECTED"],
      M["metaqa"]["RULES"][r]["new_selected_per_query"],
      M["metaqa"]["RULES"][r]["queries_with_a_new_proposal_selected"],
      M["metaqa"]["RULES"][r]["gold_bearing_U_proposals_admitted"],
      M["metaqa"]["RULES"][r]["gold_bearing_incumbents_evicted"],
      M["metaqa"]["RULES"][r]["queries_whose_P50_changes"]] for r in CV.RULES])

w("## T5 -- STEP 6 CONVERSION_EFFICIENCY = (ACTUAL - SAFE) / (U_PC5_POOL_ORACLE - SAFE)\n")
rows = []
for r in CV.RULES[1:]:
    for d, k in [("metaqa", "hop2"), ("metaqa", "hop3"), ("metaqa", "ALL"), ("webqsp", "ALL")]:
        v = M[d]["RULES"][r][k]
        b = M[d]["BOOKENDS"][k]
        ce = v["CONVERSION_EFFICIENCY"]
        rows.append([r, NAME[d] + " " + k, f4(b["SAFE"]), f4(b["U_PC5_POOL_ORACLE"]),
                     f"{b['U_PC5_POOL_ORACLE'] - b['SAFE']:+.4f}", f4(v["ACTUAL"]),
                     f"{v['delta_vs_SAFE']:+.4f}", "n/a" if ce is None else f"{ce:+.2%}"])
tbl(["rule", "slice", "SAFE", "U_PC5 pool oracle", "available gap", "ACTUAL", "converted",
     "CONVERSION_EFFICIENCY"], rows)

w("## T6 -- STEP 7 all six corpora, ALL coverage\n")
rows = []
for r in CV.RULES:
    for d in OD:
        v = M[d]["RULES"][r]["ALL"]
        rows.append([r, NAME[d], f4(M[d]["BOOKENDS"]["ALL"]["SAFE"]), f4(v["ACTUAL"]),
                     f"{v['delta_vs_SAFE']:+.4f}", v["newly_covered"], v["newly_uncovered"],
                     f"{v['net']:+d}", f"{v['mcnemar_p']:.4g}", sg(v["sig"]),
                     M[d]["RULES"][r]["mean_churn"]])
tbl(["rule", "corpus", "SAFE", "ACTUAL", "delta", "gained", "lost", "net", "McNemar p", "sig",
     "mean churn"], rows)

w("## T7 -- STEP 7 pooled over all six corpora\n")
tbl(["rule", "pooled delta", "macro delta", "worst corpus", "corpora improved", "gained", "lost",
     "net", "McNemar p", "sig"],
    [[r, f"{F['POOLED'][r]['pooled_delta']:+.5f}", f"{F['POOLED'][r]['macro_delta']:+.5f}",
      f"{F['POOLED'][r]['worst_corpus_delta']:+.5f}",
      str(F["POOLED"][r]["corpora_improved"]) + "/6",
      F["POOLED"][r]["gained"], F["POOLED"][r]["lost"], f"{F['POOLED'][r]['net']:+d}",
      f"{F['POOLED'][r]['mcnemar_p']:.3g}", sg(F["POOLED"][r]["sig"])] for r in CV.RULES[1:]])

w("## T8 -- why the admitted gold does not convert\n")
rows = []
for d, k in [("metaqa", "hop3"), ("webqsp", "ALL")]:
    for r in CV.RULES:
        v = W[d]["RULES"][r][k]
        rows.append([NAME[d] + " " + k, r, f4(v["feasible"]), f4(v["covered"]),
                     v["feasible_but_uncovered"], v["mean_n_miss_on_gap"],
                     f"{v['mean_frac_of_miss_selected']:.3f}",
                     f"{v['mean_frac_of_miss_selected_on_gap']:.3f}"])
tbl(["slice", "rule", "feasible (pool oracle)", "covered (actual)", "feasible-but-uncovered q",
     "mean size of miss on those q", "frac of miss selected (all q)",
     "frac of miss selected (gap q)"], rows)

w("## T9 -- where the needed-but-unselected partitions rank in the score order of the rule\n")
rows = []
for d, k in [("metaqa", "hop3"), ("webqsp", "ALL")]:
    for r in CV.RULES:
        u = W[d]["RULES"][r]["UNSELECTED_NEEDED_PARTITIONS"]
        s = u["source"]
        rows.append([NAME[d] + " " + k, r, u["n"], u["rank_in_rule_order_p50"],
                     u["rank_in_rule_order_p90"], u["within_top_12"], u["within_top_50"],
                     s["in_base50"], s["in_chal"], s["truly_new"]])
tbl(["slice", "rule", "n", "rank p50", "rank p90", "in top-12", "in top-50", "src: base50",
     "src: challenger", "src: truly new"], rows)
w("Rank is the position among all scored candidates (92-290 of them, see T10); only the top 6 are "
  "taken.\n")

w("## T10 -- STEP 10 latency and cache accounting\n")
tbl(["corpus", "u_rank lookup+merge (ms/query)", "selector SAFE (ms/query)",
     "selector S1_CSU_RAW (ms/query)", "total added (ms/query)", "mean candidates scored",
     "U cache (MB)", "partition-graph cache (MB)"],
    [[NAME[d], f"{F['LATENCY'][d]['u_rank_lookup_ms']:.3f}",
      f"{F['LATENCY'][d]['selector_S0_SAFE_ms']:.3f}",
      f"{F['LATENCY'][d]['selector_S1_CSU_RAW_ms']:.3f}",
      f"{F['LATENCY'][d]['u_rank_lookup_ms'] + F['LATENCY'][d]['selector_S1_CSU_RAW_ms'] - F['LATENCY'][d]['selector_S0_SAFE_ms']:.3f}",
      F["LATENCY"][d]["mean_candidates"], f"{F['LATENCY'][d]['u_cache_bytes'] / 1e6:.2f}",
      f"{F['LATENCY'][d]['partition_graph_cache_bytes'] / 1e6:.2f}"] for d in OD])
w("`ONLINE_GRAPH_EDGES_TOUCHED = " + str(F["ONLINE_GRAPH_EDGES_TOUCHED"]) + "`. No encoder work "
  "and no graph traversal at query time: `G_GRAPH_NBR_RAW` reads precomputed depth-1 neighbour "
  "rows and merges them arithmetically.\n")

os.makedirs(CV.CVD, exist_ok=True)
open(f"{CV.CVD}/TABLES.md", "w", encoding="utf-8").write("\n".join(L))
print(f"wrote {CV.CVD}/TABLES.md  ({len(L)} lines)")
