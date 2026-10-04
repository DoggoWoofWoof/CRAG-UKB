"""Render TABLES.md for the L1 FINAL SCORING PHASE from the diag/*.json artifacts."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cal_core as CC
import _l1pp_core as PP

D = f"{CC.CALD}/diag"
S0 = json.load(open(f"{D}/step0_capacity.json"))
S1 = json.load(open(f"{D}/step1_failure.json"))
MN = json.load(open(f"{D}/main_R0R6.json"))
VA = json.load(open(f"{D}/step10_validation.json"))
SHORT = {"metaqa": "MetaQA", "webqsp": "WebQSP", "2wiki_clean": "2Wiki",
         "musique_clean": "MuSiQue", "hotpotqa_clean": "HotpotQA", "squad_clean": "SQuAD"}
DS = CC.DSETS
R = CC.RULES
L = []
w = L.append


def sig(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


IND = {}
for ds in DS:
    z = np.load(f"{D}/ind_{ds}.npz")
    IND[ds] = {k: z[k].astype(np.int8) for k in z.files}
    IND[ds]["disc"], IND[ds]["val"] = CC.folds(ds, len(z["base"]))

w("# L1 FINAL SCORING PHASE — TABLES\n")
w("`*` p<0.05, `**` p<0.01 (exact McNemar, paired, DEV only). SAFE_F6 = `B6_S4_F6_Ms64_Mr32` = R0.\n")

# ---------------------------------------------------------------- T1
w("\n## T1 — STEP 0. The capacity claim with its candidate universe attached (B=6)\n")
w("| corpus | CAP_P50 (C) | POOL_UNREACHABLE current | POOL_UNREACHABLE PPR-expanded | "
  "CAP_B6 current pool (A) | CAP_B6 PPR-expanded pool (B) | SOLVABLE current | SOLVABLE PPR-exp |")
w("|---|---|---|---|---|---|---|---|")
for ds in DS:
    c = S0[ds]["CLASSES"]["B6"]
    w(f"| {SHORT[ds]} | {c['CAP_P50']} | {c['POOL_UNREACHABLE_CURRENT']} | "
      f"{c['POOL_UNREACHABLE_PPR_EXPANDED']} | {c['CAP_B_CURRENT_POOL']} | "
      f"{c['CAP_B_PPR_EXPANDED_POOL']} | {c['SOLVABLE_CURRENT']} | {c['SOLVABLE_PPR_EXPANDED']} |")

w("\n## T2 — STEP 0. CAP_B by boundary width, both universes\n")
w("| corpus | " + " | ".join(f"B={b} cur / ppr-exp" for b in (6, 8, 12)) + " |")
w("|---|" + "---|" * 3)
for ds in DS:
    cells = []
    for b in (6, 8, 12):
        c = S0[ds]["CLASSES"][f"B{b}"]
        cells.append(f"{c['CAP_B_CURRENT_POOL']} / {c['CAP_B_PPR_EXPANDED_POOL']}")
    w(f"| {SHORT[ds]} | " + " | ".join(cells) + " |")

w("\n## T3 — STEP 0 DIAGNOSTIC. PPR admitted as a fourth co-equal channel (ALL coverage, Δ vs BASE)\n")
w("| corpus | B | SAFE_F6 Δ | F6+PPR Δ | Δ(F6+PPR − F6) | McNemar p |")
w("|---|---|---|---|---|---|")
for ds in DS:
    for b in (6, 8, 12):
        e = S0[ds]["PPR_DIAGNOSTIC"][f"B{b}"]
        w(f"| {SHORT[ds]} | {b} | {e['F6_dBASE']:+.4f} | {e['PPR_dBASE']:+.4f} | "
          f"{e['PPR_dF6']:+.4f}{sig(e['vs_F6']['mcnemar_p'])} | {e['vs_F6']['mcnemar_p']:.4f} |")

# ---------------------------------------------------------------- T4/T5
w("\n## T4 — STEP 1. The current failure distribution (frozen R0 selector, B=6)\n")
w("| corpus | swaps | GOOD admissions | BAD evictions | GOOD/BAD | queries with the decisive "
  "cross-list pattern | of those, costing a gold partition |")
w("|---|---|---|---|---|---|---|")
for ds in DS:
    v = S1[ds]
    w(f"| {SHORT[ds]} | {v['swaps']} | {v['GOOD_admissions']} | {v['BAD_evictions']} | "
      f"{v['GOOD_over_BAD']:.2f} | {v['queries_with_decisive_cross_list_pattern']} | "
      f"{v['of_those_evicting_a_gold_partition']} |")

w("\n## T5 — STEP 1. Channel profile of every evicted incumbent / admitted challenger\n")
w("`C` = has canonical partition rank, `S` = has S4 structural rank, `R` = has node-retrieval rank. "
  "`gold` = of those, how many carried a gold partition.\n")
w("| corpus | side | profile | n | gold |")
w("|---|---|---|---|---|")
for ds in DS:
    for side, key in (("evicted", "evicted_profiles"), ("admitted", "admitted_profiles")):
        for k, v in sorted(S1[ds][key].items(), key=lambda kv: -kv[1]["n"]):
            w(f"| {SHORT[ds]} | {side} | `{k}` | {v['n']} | {v['gold']} |")

w("\n## T6 — STEP 1. Counterfactual: what each rule does with R0's own gold decisions\n")
w("| corpus | rule | R0 BAD evictions now RETAINED | still evicted | R0 GOOD admissions still "
  "ADMITTED | now missed | net gold slots vs R0 |")
w("|---|---|---|---|---|---|---|")
for ds in DS:
    for r in R:
        v = S1[ds]["COUNTERFACTUAL"][r]
        w(f"| {SHORT[ds]} | {r} | {v['R0_BAD_evictions_now_RETAINED']} | {v['still_evicted']} | "
          f"{v['R0_GOOD_admissions_still_ADMITTED']} | {v['now_missed']} | "
          f"{v['net_gold_slots_vs_R0']:+d} |")

# ---------------------------------------------------------------- T7 main
w("\n## T7 — STEP 8. R0–R6 on all six corpora (ALL coverage, full DEV)\n")
w("| corpus | method | ALL | Δ vs BASE | Δ vs SAFE_F6 | McNemar vs BASE | McNemar vs SAFE_F6 | "
  "newly covered | newly uncovered | churn | swaps | gold admitted | gold evicted | GOOD | BAD | "
  "GOOD/BAD |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for ds in DS:
    m = MN[ds]
    b = m["BASE"]
    w(f"| {SHORT[ds]} | BASE | {b['ALL']:.4f} | +0.0000 | {b['dSAFE_F6']:+.4f} | — | — | — | — | "
      f"0.00 | 0 | — | — | — | — | — |")
    for r in R:
        v = m[r]
        w(f"| {SHORT[ds]} | {r} | {v['ALL']:.4f} | {v['dBASE']:+.4f} | "
          f"{v['dSAFE_F6']:+.4f}{sig(v['vs_SAFE_F6']['mcnemar_p'])} | "
          f"{v['vs_BASE']['mcnemar_p']:.4f} | {v['vs_SAFE_F6']['mcnemar_p']:.4f} | "
          f"{v['newly_covered_vs_F6']} | {v['newly_uncovered_vs_F6']} | {v['churn_mean']:.2f} | "
          f"{v['swaps']} | {v['gold_admitted']} | {v['gold_evicted']} | {v['GOOD']} | {v['BAD']} | "
          f"{v['GOOD_over_BAD']:.2f} |")
    o = m["ORACLE_B6"]
    w(f"| {SHORT[ds]} | ORACLE_B6 | {o['ALL']:.4f} | {o['dBASE']:+.4f} | {o['dSAFE_F6']:+.4f} | "
      f"— | — | — | — | — | — | — | — | — | — | — |")

w("\n## T8 — STEP 8. Δ vs SAFE_F6, compact\n")
w("| rule | " + " | ".join(SHORT[d] for d in DS) + " | macro | worst |")
w("|---|" + "---|" * (len(DS) + 2))
for r in R:
    ds_ = [MN[d][r]["dSAFE_F6"] for d in DS]
    cells = [f"{MN[d][r]['dSAFE_F6']:+.4f}{sig(MN[d][r]['vs_SAFE_F6']['mcnemar_p'])}" for d in DS]
    w(f"| {r} | " + " | ".join(cells) + f" | {np.mean(ds_):+.5f} | {min(ds_):+.4f} |")

# ---------------------------------------------------------------- T9 hops
w("\n## T9 — STEP 9. MetaQA per-hop (mandatory), 666 queries per hop\n")
w("| method | hop1 | Δ hop1 | hop2 | Δ hop2 | hop3 | Δ hop3 |")
w("|---|---|---|---|---|---|---|")
m = MN["metaqa"]
ph = m["BASE"]["per_hop"]
w("| BASE | " + " | ".join(f"{ph[h]['ALL']:.4f} | — " for h in "123") + "|")
for r in R:
    p = m[r]["per_hop"]
    cells = []
    for h in "123":
        e = p[h]
        cells.append(f"{e['ALL']:.4f}")
        cells.append(f"{e['dSAFE_F6']:+.4f}{sig(e['vs_SAFE_F6']['mcnemar_p'])}")
    w(f"| {r} | " + " | ".join(cells) + " |")
ph = m["ORACLE_B6"]["per_hop"]
w("| ORACLE_B6 | " + " | ".join(f"{ph[h]['ALL']:.4f} | — " for h in "123") + "|")

# ---------------------------------------------------------------- T10/T11 validation
w("\n## T10 — STEP 10. Nested discovery → validation (sha1 parity folds, query-disjoint)\n")
w(f"Criterion, fixed in advance: {VA['criterion']}\n")
w("| rule | DISCOVERY macro | DISCOVERY worst | DISCOVERY sig-regressions | selected |")
w("|---|---|---|---|---|")
for r in R:
    v = VA["NESTED"]["discovery"][r]
    sr = ", ".join(SHORT[x] for x in v["sig_regressions"]) or "none"
    w(f"| {r} | {v['macro']:+.5f} | {v['worst']:+.5f} | {sr} | "
      f"{'**YES**' if r == VA['NESTED']['discovery_winner'] else ''} |")

w("\n## T11 — STEP 10. Per-half stability of the two calibrated rules\n")
w("| rule | corpus | DISCOVERY Δ | p | VALIDATION Δ | p | FULL Δ | p |")
w("|---|---|---|---|---|---|---|---|")
for r in ("R1_RRF_LIFT", "R2_RRF_UNIT"):
    macros = {}
    for ds in DS:
        cells = []
        for k in ("disc", "val", None):
            msk = IND[ds][k] if k else np.ones(len(IND[ds]["base"]), bool)
            cur, ref = IND[ds][r][msk], IND[ds]["R0_RAW_RRF"][msk]
            d = float(cur.mean() - ref.mean()); p = PP.mcnemar(cur, ref)["mcnemar_p"]
            cells += [f"{d:+.4f}{sig(p)}", f"{p:.4f}"]
            macros.setdefault(k, []).append(d)
        w(f"| {r} | {SHORT[ds]} | " + " | ".join(cells) + " |")
    w(f"| {r} | **macro** | **{np.mean(macros['disc']):+.5f}** | | "
      f"**{np.mean(macros['val']):+.5f}** | | **{np.mean(macros[None]):+.5f}** | |")

w("\n## T12 — STEP 10. Leave-one-dataset-out selection\n")
w("| held-out corpus | rule selected on the other 5 | train macro | held-out Δ vs SAFE_F6 | "
  "McNemar p | significant regression |")
w("|---|---|---|---|---|---|")
for ds in DS:
    v = VA["LODO"]["folds"][ds]
    w(f"| {SHORT[ds]} | {v['selected_on_other_5']} | {v['train_macro']:+.5f} | "
      f"{v['heldout_delta_vs_SAFE_F6']:+.4f} | {v['heldout_mcnemar_p']:.4f} | "
      f"{'YES' if v['heldout_regression_significant'] else 'no'} |")
lo = VA["LODO"]
w(f"\n`SELECTION_STABLE = {lo['SELECTION_STABLE']}` — distinct selections across folds: "
  f"{', '.join(lo['distinct_selections'])}. Held-out macro {lo['heldout_macro']:+.5f}.\n")
w(f"`SURVIVES_VALIDATION = {VA['NESTED']['SURVIVES_VALIDATION']}` · "
  f"`PROMOTABLE = {VA['PROMOTABLE']}` · `WINNER = {VA['WINNER']}`\n")

# ---------------------------------------------------------------- T13 latency
w("\n## T13 — STEP 11. Cost accounting\n")
w("| corpus | ONLINE_GRAPH_EDGES_TOUCHED | new encoder passes | additional cached bytes/query | "
  "ranks read/query | candidates scored/query | extra float ops/query vs R0 |")
w("|---|---|---|---|---|---|---|")
for ds in DS:
    v = MN[ds]["_LATENCY"]
    w(f"| {SHORT[ds]} | {v['ONLINE_GRAPH_EDGES_TOUCHED']} | {v['new_encoder_passes']} | "
      f"{v['additional_cached_bytes_per_query']} | {v['ranks_read_per_query']} | "
      f"{v['mean_candidates_scored_per_query']:.1f} | {v['float_ops_per_query_extra_vs_R0']} |")

w("\n## T14 — STEP 11. Query routing latency by rule (µs/query, mean over the six corpora)\n")
w("| rule | µs/query | vs R0 |")
w("|---|---|---|")
r0 = np.mean([MN[d]["R0_RAW_RRF"]["sec_per_query"] for d in DS]) * 1e6
for r in R:
    t = np.mean([MN[d][r]["sec_per_query"] for d in DS]) * 1e6
    w(f"| {r} | {t:.1f} | {t - r0:+.1f} |")

open(f"{CC.CALD}/TABLES.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote TABLES.md", len(L), "lines")
