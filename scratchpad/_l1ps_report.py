"""Assemble results/GENERALIZATION/G2_L1_PARTITION_SEARCH/FINAL_REPORT.md (17 deliverables).

  python scratchpad/_l1ps_report.py <final_run_name>
      e.g. python scratchpad/_l1ps_report.py B4_S4_F6_Ms32_Mr32
"""
import os, sys, json, glob
import numpy as np

ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
NAME = sys.argv[1]
DS = [("metaqa", "MetaQA", "KB"), ("webqsp", "WebQSP", "KB"),
      ("2wiki_clean", "2Wiki", "text"), ("musique_clean", "MuSiQue", "text"),
      ("hotpotqa_clean", "HotpotQA", "text"), ("squad_clean", "SQuAD", "text")]
MAN = json.load(open(f"{ROOT}/manifest.json"))
ORA = json.load(open(f"{ROOT}/oracle/fixed_p50_ceiling.json"))
FIN = json.load(open(f"{ROOT}/runs/final_{NAME}.json"))
CMB = {k: json.load(open(f"results/GENERALIZATION/_g2_comb_{k}.json")) for k, _, _ in DS}
SB = {}
for f in sorted(glob.glob(f"{ROOT}/scoreboard_round*.json")):
    SB[os.path.basename(f)[11:-5]] = json.load(open(f))

L = []
w = L.append
PIPE = "\\|"


def sig(p):
    return "**sig**" if p < 0.05 else "ns"


def sgn(x):
    return "--" if x is None else f"{x:+.4f}"


w("# G2 -- UNIVERSAL FIXED-P50 PARTITION ROUTER SEARCH")
w("\n*L1 output = exactly 50 canonical C partitions. No stray nodes. No learned parameters, "
  "no dataset identity, no gold at inference, no TEST inspection.*\n")

# ---------------- 1 substrate / BASE
w("\n## 1. Six-dataset BASE (STEP 0)\n")
w("| corpus | family | dev n | sample rule | docs | partitions | BASE ANY@P50 | BASE ALL@P50 | "
  "BASE scope (nodes) | scope % corpus | parity |")
w("|---|---|--:|---|--:|--:|--:|--:|--:|--:|:--|")
for k, la, fa in DS:
    m = MAN[k]
    w(f"| **{la}** | {fa} | {m['n_dev_queries']} | {m['sample_rule']} | {m['n_docs']:,} | "
      f"{m['npart']:,} | {m['BASE_ANY_P50']:.4f} | {m['BASE_ALL_P50']:.4f} | "
      f"{m['BASE_SCOPE_NODES']:.0f} | {100.0 * m['BASE_SCOPE_NODES'] / m['n_docs']:.1f}% | "
      f"`{m['BASE_PARTITION_PARITY']['status']}` |")
w(f"\nFrozen contract on every corpus: `{json.dumps(MAN['metaqa']['CONTRACT'])}`.")
w("\nWebQSP reports both ANY (0.9436) and ALL (0.7618); ALL is never silently substituted by ANY "
  "anywhere in this report.")

# ---------------- 2 direct-node diagnostic
w("\n## 2. Direct structural-NODE evidence (what the partition router has to reproduce)\n")
w("Additive node-level residual arms measured *before* the partition-only constraint was imposed "
  "(`BASE P50 + node residual`, BASE never evicted). These are the STEP-11 denominators.\n")
w("| corpus | BASE ALL | RET32 | STRUCT32 | RET64 | STRUCT64 | COMBINED 32+32 | "
  "struct newly-cov q | ret newly-cov q |")
w("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for k, la, fa in DS:
    c = CMB[k]["RESULTS"]

    def d(a, c=c):
        return f"{c[a]['ALL']:.4f} ({c[a]['dALL_vs_BASE']:+.4f})"

    w(f"| **{la}** ({fa}) | {CMB[k]['BASE_ALL_P50']:.4f} | {d('RET32')} | {d('STRUCT32')} | "
      f"{d('RET64')} | {d('STRUCT64')} | **{d('COMBINED_32_32')}** | "
      f"{c['STRUCT32']['queries_newly_ALL_covered']} | {c['RET32']['queries_newly_ALL_covered']} |")

# ---------------- 3 oracle
w("\n## 3. Fixed-P50 ORACLE ceiling (STEP 2 -- the gate)\n")
w("A query is oracle-covered at budget B iff `need = gold_parts \\ protected` has `|need| <= B` "
  "and `need` is contained in `boundary u challengers`. Evaluation only: gold is used **only** to "
  "compute this ceiling, never as an inference feature.\n")
BG = ORA["B_GRID"]
w("| corpus | BASE | direct STRUCT_NODE32 | " + " | ".join(f"B={b}" for b in BG) +
  " | UNBOUNDED@B=4 | mean need @B=1 | frac need<=1 |")
w("|---|--:|--:|" + "--:|" * (len(BG) + 3))
for k, la, _ in DS:
    r = ORA["RESULTS"][k]
    cells = " | ".join(f"{r['B'][str(b)]['COMBINED']['ALL']:.4f}" for b in BG)
    w(f"| **{la}** | {r['BASE_ALL']:.4f} | {r['direct_node_diagnostic']['STRUCT_NODE32']:.4f} | "
      f"{cells} | {r['B']['4']['UNBOUNDED']['ALL']:.4f} | {r['B']['1']['mean_need']:.3f} | "
      f"{r['B']['1']['frac_need_le_B']:.3f} |")
w("\nChallenger-family ceilings at B=4 -- does structure add *reach* over retrieval continuation?\n")
w("| corpus | STRUCT only | RET only | COMBINED | COMBINED - best single |")
w("|---|--:|--:|--:|--:|")
for k, la, _ in DS:
    b = ORA["RESULTS"][k]["B"]["4"]
    mx = max(b["STRUCT"]["ALL"], b["RET"]["ALL"])
    w(f"| **{la}** | {b['STRUCT']['ALL']:.4f} | {b['RET']['ALL']:.4f} | "
      f"**{b['COMBINED']['ALL']:.4f}** | {b['COMBINED']['ALL'] - mx:+.4f} |")

# ---------------- 4 families / rounds
w("\n## 4. Search rounds and surviving families (STEPS 4-8)\n")
RND = {"round1": ("F1 F2 F3 F4(T1-3) F5, asymmetric", "B 1-12, M=32, S1-S4"),
       "round2": ("F6 F7 F8, SYMMETRIC competition", "B 1-24, M=32, S1-S4"),
       "round3": ("F6, widened challenger source", "B 2-8, M_s/M_r in {32,64,128}, S1/S4"),
       "round4": ("FA, symmetric + multi-node support gate T (F6 control)",
                  "B 4-8, M_s in {64,128}, M_r in {32,64}, S1/S4")}
w("| round | families | grid | configs/corpus | runtime s | best universal config | worst dALL | "
  "macro dALL | sig regressions |")
w("|---|---|---|--:|--:|---|--:|--:|--:|")
for r in sorted(SB):
    sb = SB[r]
    t = sb["RANKING"][0]
    dd = RND.get(r, ("", ""))
    w(f"| {r} | {dd[0]} | {dd[1]} | {len(sb['SCOREBOARD']['metaqa']['configs'])} | "
      f"{sb['runtime_sec']:.0f} | `{t['config']}` | {t['worst_dALL']:+.4f} | "
      f"{t['macro_dALL']:+.4f} | {t['n_sig_regressions']} |")

if "round2" in SB:
    w("\n**Family separation (round 2, S4, B=4)** -- the result that decides the architecture:\n")
    w("| corpus | F7 struct-only | F8 ret-only | F6 struct+ret |")
    w("|---|--:|--:|--:|")
    for k, la, _ in DS:
        c = SB["round2"]["SCOREBOARD"][k]["configs"]
        w(f"| **{la}** | {c['B4_S4_F7_Ms32_Mr32']['dALL']:+.4f} | "
          f"{c['B4_S4_F8_Ms32_Mr32']['dALL']:+.4f} | **{c['B4_S4_F6_Ms32_Mr32']['dALL']:+.4f}** |")
    w("\nStructure-only helps the KB corpora and hurts the text corpora; retrieval-only does the "
      "opposite. Only the union is non-negative on all six -- which is why a *universal* router "
      "needs both channels and cannot be either one alone.")

if "round4" in SB:
    w("\n**Refinement H rejected on evidence (round 4, B=6, S4, M_s=64, M_r=32).** Requiring a "
      "challenger to have >= T structural nodes in the same partition before it may compete:\n")
    w("| corpus | F6 ungated | FAT2 (T=2) | FAT3 (T=3) | churn F6 / FAT2 / FAT3 |")
    w("|---|--:|--:|--:|--:|")
    for k, la, fa in DS:
        c = SB["round4"]["SCOREBOARD"][k]["configs"]

        def g(f, c=c):
            return c[f"B6_S4_{f}_Ms64_Mr32"]

        w(f"| **{la}** ({fa}) | {g('F6')['dALL']:+.4f} | {g('FAT2')['dALL']:+.4f} | "
          f"{g('FAT3')['dALL']:+.4f} | {g('F6')['churn_per_query']:.2f} / "
          f"{g('FAT2')['churn_per_query']:.2f} / {g('FAT3')['churn_per_query']:.2f} |")
    w("\nThe gate improves **all four text corpora** and simultaneously destroys **both KB "
      "corpora**. The mechanism is direct: on a KB substrate the decisive structural evidence is "
      "frequently a *single* node that a second hop reaches inside the right partition, so a "
      "multi-node requirement discards exactly the signal that makes structure worth having; on "
      "text substrates single-node structural evidence is mostly noise, so the same rule acts as "
      "a clean filter. Adopting it would therefore be a corpus-specific rule, which the "
      "architecture contract forbids -- so it is rejected despite winning on 4 of 6 corpora. "
      "Round 4 produced **no Pareto improvement**: its universal winner is bit-identical to "
      "round 3's.")

# ---------------- 5 final config full DEV
F = FIN["DATASETS"]
u = FIN["UNIVERSAL"]
w("\n## 5. FINAL universal configuration -- full DEV (STEP 10)\n")
w("```json")
w(json.dumps({k: FIN["config"][k] for k in
              ("name", "B", "M_struct", "M_ret", "agg", "fusion", "T", "K0", "P_MAIN",
               "learned_parameters", "uses_gold_at_inference", "uses_dataset_identity",
               "output")}, indent=1))
w("```\n")
w("| corpus | BASE ANY | ANY | BASE ALL | ALL | dALL | newly ALL-cov | newly uncov | net | "
  "McNemar p | |")
w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
for k, la, _ in DS:
    v = F[k]
    w(f"| **{la}** | {v['BASE_ANY']:.4f} | {v['ANY']:.4f} | {v['BASE_ALL']:.4f} | {v['ALL']:.4f} | "
      f"**{v['dALL']:+.4f}** | {v['newly_covered']} | {v['newly_uncovered']} | {v['net']:+} | "
      f"{v['mcnemar_p']} | {sig(v['mcnemar_p'])} |")
w(f"\nworst-dataset dALL **{u['worst_dALL']:+.4f}** | macro dALL **{u['macro_dALL']:+.4f}** | "
  f"positive on {u['n_positive']}/6 | significant improvements {u['n_sig_improvements']}/6 | "
  f"**significant regressions {u['n_sig_regressions']}/6** | net queries "
  f"{u['total_net_queries']:+} | mean churn {u['mean_churn']:.2f}")

# ---------------- 6 metaqa per hop
if "per_hop" in F["metaqa"]:
    w("\n## 6. MetaQA per-hop ALL\n")
    w("| hop | n | BASE ALL | final ALL | delta |")
    w("|---|--:|--:|--:|--:|")
    for h, v in F["metaqa"]["per_hop"].items():
        w(f"| {h} | {v['n']} | {v['BASE_ALL']:.4f} | {v['ALL']:.4f} | "
          f"{v['ALL'] - v['BASE_ALL']:+.4f} |")

# ---------------- 7 admitted / evicted / churn
w("\n## 7. Gold-bearing partitions admitted vs evicted, and churn (STEP 10)\n")
w("| corpus | admitted | evicted | net | churn/query | churn max | queries with 0 churn | "
  "final scope nodes | BASE scope | scope growth |")
w("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
for k, la, _ in DS:
    v = F[k]
    w(f"| **{la}** | {v['gold_parts_admitted']} | {v['gold_parts_evicted']} | "
      f"{v['net_gold_parts']:+} | {v['churn_per_query']:.3f} | {v['churn_max']} | "
      f"{v['queries_with_zero_churn']} ({100.0 * v['queries_with_zero_churn'] / v['n']:.0f}%) | "
      f"{v['final_scope_nodes_mean']:.0f} | {v['BASE_hard_union_nodes_mean']:.0f} | "
      f"{v['scope_growth_pct']:+.2f}% |")
w("\nScope is *not* the source of the gain: the final hard union stays at the BASE budget because "
  "the output is always exactly 50 partitions. Growth is only the difference in partition sizes "
  "between the evicted and admitted slots.")

# ---------------- 8 complementarity
w("\n## 8. Structural vs retrieval challenger complementarity (STEP 12)\n")
w("Which channel supplied the partition that actually flipped each newly ALL-covered query:\n")
w("| corpus | struct only | ret only | both | boundary reorder |")
w("|---|--:|--:|--:|--:|")
for k, la, _ in DS:
    s = F[k]["decisive_channel_for_newly_covered"]
    w(f"| **{la}** | {s['struct_only']} | {s['ret_only']} | {s['both']} | {s['boundary_reorder']} |")
w("\nNode-level marginal complementarity from the direct experiment, both directions:\n")
w(f"| corpus | STRUCT after RET | RET after STRUCT | overlap {PIPE}RET32 n STRUCT32{PIPE}/32 |")
w("|---|--:|--:|--:|")
for k, la, _ in DS:
    m = CMB[k]["MARGINAL"]
    o = CMB[k]["OVERLAP"]
    a = m["paired_COMBINED_vs_RET32"]
    b = m["paired_COMBINED_vs_STRUCT32"]
    w(f"| **{la}** | {m['STRUCT_MARGINAL_AFTER_RET']:+.4f} (net {a['net']:+}, p={a['mcnemar_p']}) | "
      f"{m['RET_MARGINAL_AFTER_STRUCT']:+.4f} (net {b['net']:+}, p={b['mcnemar_p']}) | "
      f"{o['overlap_fraction_of_32']:.1%} |")

# ---------------- 9 recovered fraction
w("\n## 9. STEP 11 -- fraction of the direct structural-node gain recovered at exactly P50\n")
w("`RECOVERED_STRUCTURAL_GAIN_FRACTION = dALL(best fixed-P50) / dALL(direct STRUCT_NODE32)`\n")
w("| corpus | direct STRUCT_NODE32 dALL | direct COMBINED node dALL | fixed-P50 dALL | "
  "**RECOVERED** | vs direct COMBINED | oracle dALL @B | frac of oracle captured |")
w("|---|--:|--:|--:|--:|--:|--:|--:|")
BSEL = str(FIN["config"]["B"])
for k, la, _ in DS:
    v = F[k]
    # read the ceiling from the oracle artifact rather than the value cached in the run json, so
    # this column stays correct when the oracle grid is extended after a run.
    ob = ORA["RESULTS"][k]["B"].get(BSEL)
    v["oracle_dALL_at_this_B"] = ob["COMBINED"]["dALL"] if ob else None
    f3 = (round(v["dALL"] / v["oracle_dALL_at_this_B"], 3)
          if v["oracle_dALL_at_this_B"] and v["oracle_dALL_at_this_B"] > 1e-9 else None)
    v["fraction_of_oracle_captured"] = f3
    f1, f2 = v["RECOVERED_STRUCTURAL_GAIN_FRACTION"], v["recovered_of_direct_COMBINED"]
    w(f"| **{la}** | {v['direct_STRUCT_NODE32_dALL']:+.4f} | "
      f"{v['direct_COMBINED_NODE_32_32_dALL']:+.4f} | {v['dALL']:+.4f} | "
      f"**{'--' if f1 is None else f'{f1:.2f}x'}** | {'--' if f2 is None else f'{f2:.2f}x'} | "
      f"{sgn(v['oracle_dALL_at_this_B'])} | {'--' if f3 is None else f'{f3:.1%}'} |")
w(f"\nMacro RECOVERED_STRUCTURAL_GAIN_FRACTION = "
  f"**{u['macro_recovered_structural_gain_fraction']:.2f}x**")
w("\n*Read the ratio with its denominator.* STEP 11 defines the denominator as the direct "
  "STRUCT_NODE32 gain, which on the four text corpora is nearly zero (+0.0005 to +0.0035, i.e. "
  "1-7 queries), so a ratio above 1 there means only that the router's *retrieval* channel "
  "contributes more than structure ever did on those corpora -- not that structure was amplified. "
  "The informative cells are the two KB corpora, where the structural signal is real, and the "
  "'vs direct COMBINED' column, which compares against the full node-level residual gain that the "
  "50-partition constraint had to give up.")

# ---------------- 10 compute
w("\n## 10. Compute, memory, Modal (STEP 10)\n")
w("| corpus | fp16 path | predicted peak GB | measured peak RSS GB | BASE s | struct traversal s | "
  "x BASE | edges traversed | cache build s | route ms/query | executed on |")
w("|---|:--|--:|--:|--:|--:|--:|--:|--:|--:|---|")
for k, la, _ in DS:
    m = MAN[k]
    v = F[k]
    w(f"| **{la}** | {'yes' if m['fp16_path'] else 'no'} | {m['predicted_peak_gb']:.2f} | "
      f"{v['peak_rss_gb']} | {m['COST']['BASE_sec']} | {m['COST']['STRUCT_sec']} | "
      f"{m['COST']['STRUCT_x_BASE']}x | {m['COST']['STRUCT_edges']:,} | {v['cache_build_sec']} | "
      f"{v['route_ms_per_query']:.3f} | local |")
w("\nNo corpus required Modal. The two corpora whose fp32 node matrices do not fit in 15.7 GB "
  "(WebQSP 781,485x1536 = 4.8 GB, HotpotQA 507,494x1536 = 3.1 GB) ran through the frozen "
  "`EXHAUSTIVE_SHARDED_FP16_IP` path (fp16 storage, fp32 compute), predicted peak 2.43 / 1.58 GB, "
  "well under the 70% trigger. `NEW_ENCODER_PASSES = 0` throughout.")

# ---------------- 11 universal Pareto frontier
w("\n## 11. Universal Pareto frontier (STEP 9)\n")
ALLR = []
for r in sorted(SB):
    for x in SB[r]["RANKING"]:
        ALLR.append(dict(x, round=r))
seen = set()
zero = []
for r in ALLR:                      # a config can appear in more than one round; keep the first
    if r["n_sig_regressions"] == 0 and r["config"] not in seen:
        seen.add(r["config"])
        zero.append(r)
par = [a for a in zero if not any(
    (b["worst_dALL"] >= a["worst_dALL"] and b["macro_dALL"] >= a["macro_dALL"] and
     (b["worst_dALL"] > a["worst_dALL"] or b["macro_dALL"] > a["macro_dALL"])) for b in zero)]
par.sort(key=lambda r: (-r["worst_dALL"], -r["macro_dALL"]))
w(f"{len(ALLR)} configurations evaluated across all rounds; {len(zero)} have zero significant "
  f"regressions on any of the six corpora. Non-dominated on (worst-dataset dALL, macro dALL):\n")
w("| round | config | worst dALL | macro dALL | churn | " +
  " | ".join(la for _, la, _ in DS) + " |")
w("|---|---|--:|--:|--:|" + "--:|" * len(DS))
for p in par[:10]:
    w(f"| {p['round']} | `{p['config']}` | **{p['worst_dALL']:+.4f}** | {p['macro_dALL']:+.4f} | "
      f"{p['churn']:.2f} | " + " | ".join(f"{p['per_ds'][k]:+.4f}" for k, _, _ in DS) + " |")
w("\nThe frontier is a genuine trade-off, not a hidden regression: raising B past the selected "
  "value buys macro-average dALL by spending WebQSP, which is the binding constraint. The "
  "lexicographic rule in STEP 9 (no significant regression anywhere, then maximise the worst "
  "dataset) selects the top row.")

# ---------------- 12 mechanism / cross-dataset analysis
w("\n## 12. Cross-dataset mechanism analysis (STEP 12)\n")
w("| corpus | family | struct newly-cov (direct nodes) | ret newly-cov (direct nodes) | "
  "struct:ret ratio | mean need @B=1 | frac need<=1 | decisive struct-only at P50 |")
w("|---|---|--:|--:|--:|--:|--:|--:|")
for k, la, fa in DS:
    c = CMB[k]["RESULTS"]
    s = c["STRUCT32"]["queries_newly_ALL_covered"]
    r = c["RET32"]["queries_newly_ALL_covered"]
    o = ORA["RESULTS"][k]["B"]["1"]
    w(f"| **{la}** | {fa} | {s} | {r} | {'inf' if r == 0 else f'{s / r:.1f}x'} | "
      f"{o['mean_need']:.3f} | {o['frac_need_le_B']:.3f} | "
      f"{F[k]['decisive_channel_for_newly_covered']['struct_only']} |")
w("\nDataset identity is never an inference feature; this table is post-hoc explanation only.")

# ---------------- 13 recommendation + verdict
cfgd = FIN["config"]
FOUND = (u["n_sig_regressions"] == 0 and u["worst_dALL"] > 0)
w("\n## 13. Recommendation, canonical configuration, verdict\n")
w("**Recommendation: PROTECTED-CORE + COMBINED STRUCTURAL/RETRIEVAL BOUNDARY.**\n")
w("Rejected alternatives, on evidence: *KEEP BASE* is dominated (the selected router is "
  "non-negative on all six corpora and significantly positive on several); "
  "*PROTECTED-CORE + STRUCTURAL BOUNDARY only* (F7) helps the two KB corpora and significantly "
  "hurts the text corpora, so it is not universal.\n")
w("Exact canonical configuration:\n")
w("```")
w(f"FINAL_PARTITIONS      = {cfgd['P_MAIN']}            # exactly 50 canonical C partitions, "
  f"no stray nodes")
w(f"B (replaceable slots) = {cfgd['B']}             # protected core = base_rank[:{50 - cfgd['B']}]")
w(f"M_struct              = {cfgd['M_struct']}            # structural residual nodes -> partition "
  f"evidence")
w(f"M_ret                 = {cfgd['M_ret']}            # retrieval-continuation nodes -> partition "
  f"evidence")
w(f"aggregation           = {cfgd['agg']}            # MULTI_SIGNAL_RRF over 4 partition rankings")
w("                                     #   best structural node rank, node-count support,")
w("                                     #   min hop, best directional s_dir -- fixed RRF, no weights")
w(f"boundary fusion       = {cfgd['fusion']}{'' if cfgd['T'] == 1 else ' T=%d' % cfgd['T']}"
  f"            # SYMMETRIC boundary competition")
w("                                     #   incumbents and challengers scored on the same three")
w("                                     #   channels (canonical, structural, retrieval), so B is a")
w("                                     #   CAP on churn rather than a forced churn")
w(f"K0                    = {cfgd['K0']}            # canonical RRF constant, unchanged")
w("structural discovery  = universal RRF seeds, beam 64, MAX_HOPS 3, DEG_CAP 300, SEED_K 5")
w("learned parameters    = 0        dataset branches = 0        gold at inference = no")
w("```\n")
kb = [F["metaqa"], F["webqsp"]]
w("### What this does *not* claim -- the measured limit\n")
w(f"The router is universally positive, but on the two corpora where structural evidence is "
  f"genuinely informative it recovers only "
  f"{kb[0]['RECOVERED_STRUCTURAL_GAIN_FRACTION']:.2f}x (MetaQA) and "
  f"{kb[1]['RECOVERED_STRUCTURAL_GAIN_FRACTION']:.2f}x (WebQSP) of the direct structural-node "
  f"gain -- {kb[0]['fraction_of_oracle_captured']:.1%} and "
  f"{kb[1]['fraction_of_oracle_captured']:.1%} of its own fixed-P50 oracle ceiling.\n")
w("STEP 2 already settled where that loss comes from, and it is **not** the partition "
  "abstraction: at B=1 the combined-challenger oracle already exceeds the entire direct-node "
  "gain on every corpus (MetaQA 0.7072 vs 0.6977; WebQSP 0.7956 vs 0.7759), and capacity "
  "saturates by B=2. Reach and capacity are sufficient. What binds is the **eviction cost of a "
  "parameter-free scoring rule**: on MetaQA the selected router admits 257 gold-bearing "
  "partitions and evicts 183 to pay for them (net +74); on WebQSP 91 admitted against 90 "
  "evicted (net +1). Every promotion is funded by a demotion, and across 368 configurations and "
  "four mechanism-guided rounds no parameter-free rule separated the two well enough to keep "
  "more than a small fraction of the ceiling on a KB substrate.\n")
w("The one rule that separates them better -- the multi-node support gate of section 4 -- is "
  "regime-specific in the opposite direction (it helps all four text corpora and destroys both "
  "KB corpora), so it cannot be adopted without a dataset branch. That is the honest boundary "
  "of this result.\n")
w(f"**UNIVERSAL_FIXED_P50_PARTITION_ROUTER_FOUND = {'YES' if FOUND else 'NO'}**\n")
if FOUND:
    w(f"One global configuration, no dataset selector, improves ALL-coverage on all six corpora "
      f"(worst {u['worst_dALL']:+.4f}, macro {u['macro_dALL']:+.4f}, "
      f"{u['n_sig_improvements']}/6 significant, 0 significant regressions, "
      f"{u['total_net_queries']:+} net queries) while emitting exactly 50 partitions.")
else:
    w("No single global configuration is non-negative on all six corpora; see the Pareto frontier "
      "in section 11 for the trade-off rather than a hidden per-dataset choice.")

w("\n## 14. Contract and stop-condition compliance\n")
w("| requirement | status |")
w("|---|:--|")
for a, b in [
    ("L1 output = partitions only, exactly P=50",
     "asserted per query in every evaluation (`len(set(final)) == 50`)"),
    ("No stray nodes bypass partitions",
     "structural nodes are used only as EVIDENCE; the emitted object is a list of partition ids"),
    ("No MLP / learned router / attention / learned fusion / learned lambda",
     "none; scoring is fixed RRF at the canonical K0=60"),
    ("No fitted score / target calibration", "no value is fitted to any target"),
    ("No dataset-specific branch or corpus-specific hyperparameter",
     "one global config for all six corpora"),
    ("No gold at inference",
     "gold is used only for the STEP-2 ceiling and for scoring the metrics"),
    ("No benchmark-specific entity annotations / special MetaQA logic",
     "universal RRF seeds only; `metaqa_ent_*` and bracket entity seeds never used"),
    ("No TEST inspection", "DEV only; WebQSP extended with TRAIN, never TEST"),
    ("No L2 / L3 run, no SQuAD-L2 repair, no MetaQA-L2 build", "not run"),
    ("No training", "no model was trained"),
    ("No topology / repartition / encoder / K / P change",
     "MASTER_TOPOLOGY=C, K=100, P_MAIN=50 unchanged; NEW_ENCODER_PASSES=0"),
    ("Single owner per config, unique run dirs, manifest",
     "lock file per cache; one scoreboard per round; a duplicate owner was detected and killed"),
]:
    w(f"| {a} | {b} |")

out = f"{ROOT}/FINAL_REPORT.md"
os.makedirs(ROOT, exist_ok=True)
open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"wrote {out}  ({len(L)} lines)")
