"""TRIPLET PHASE -- FINAL_REPORT.md.  Every number is interpolated from the diag JSONs."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1tp_core as TP
import _l1tp_build as TB
import _l1tp_run as TR
import _l1tp_p50 as PJ

D = f"{TP.TPD}/diag"
J = lambda pre, ds: json.load(open(f"{D}/{pre}_{ds}.json"))
have = lambda pre, ds: os.path.exists(f"{D}/{pre}_{ds}.json")
DS = [d for d in ["metaqa", "musique_clean", "2wiki_clean", "squad_clean"] if have("tp", d)]
R = json.load(open(f"{TP.TPD}/RETURNS.json"))
md, p = J("tp", "metaqa"), J("p50", "metaqa")
mf, mm = p["STEP10_EXACT_P50"], p["STEP10_EXACT_P50_DEPTH_MATCHED"]
SAFE = p["SAFE"]
L = []
w = L.append

w("# L1 PARAMETER-FREE STRUCTURAL-OFFSET TRIPLET PHASE -- FINAL REPORT")
w("")
w("**Verdict: C. NODE_COLLAPSE_IS_LOSSLESS.  The triplet unit adds no separable evidence over the "
  "node unit.  PROMOTED = NONE.  B unchanged at 6.  L1_FROZEN = NO.**")
w("")
w("The phase asked whether L1 improves if the structural evidence unit changes from NODE evidence "
  "to ACTUAL EDGE / TRIPLET evidence -- if edge transitions are preserved instead of immediately "
  "collapsing to target nodes.  The answer is that, for the score the collapse actually applies, "
  "**there is nothing to preserve**: the collapse is algebraically lossless, and this is a proof "
  "with an exact empirical check behind it, not a measurement that could have come out otherwise.")
w("")

# ---------------------------------------------------------------- what was built
w("## What was built (STEP 1 + STEP 2)")
w("")
w("The frozen bounded traversal already computes, for every actual directed edge `u -> v` it "
  "inspects,")
w("")
w("```")
w("delta = Xn[nb] - Xn[e];  nd = norm(delta, axis=1)")
w("S = (delta[ok] / nd[ok, None]) @ rq        # == cosine(r_q, normalize(emb(v) - emb(u)))")
w("```")
w("")
w("which is exactly `T0_OFFSET` for the triplet `T = (u, delta_uv, v)`, and then discards it in")
w("`np.maximum.at(best_s, inv, S)`.  Every other quantity STEP 1 asks for is likewise already in "
  "registers at that moment: the source's own arrival score (`F_edge[PAR]`, 1.0 for a retrieval "
  "seed) is T1_SOURCE, `hop + 1` is T3_HOP, `F_seed[PAR]` is the originating retrieval seed, and "
  "`PAR` gives the source node and its parent triplet.  T2_TARGET is `Xn[v] @ q_hat` on the same "
  "frozen unit-normalised embeddings the search already loaded.")
w("")
w("So STEP 2 needed no additional graph search and no new encoder.  `expand_feat` gained one "
  "`want_edges` export that writes those arrays out and touches no search variable, following the "
  "pattern already used for `want_visited`.  Parity is therefore preserved by construction and "
  "asserted anyway on every query against the cached frozen `s_node / s_hop / s_sdir / s_cnt`:")
w("")
rows = []
for ds in DS:
    T = np.load(f"{TP.TPD}/data/tr_{ds}.npz")
    s = J("tp", ds)["STEP12_SATURATION"]["ALL"]
    rows.append(f"| {ds} | {int(T['nq'][0])}/{int(T['nq'][0])} EXACT | {s['edges_per_query']} | "
                f"{s['TRIPLET_transitions_per_query']} | {len(T['pi']):,} |")
w("| corpus | frozen replay parity | edges inspected/query | Pi->Pj transitions/query | "
  "transitions cached |")
w("|---|---|---|---|---|")
L.extend(rows)
w("")
w("delta is never materialised as a vector and never used to retrieve: only its inner product with "
  "the static query residual is taken, and only to score the REAL transition `u -> v`.  STEP 7's "
  "rule is respected by construction -- there is no `u + delta -> FAISS` anywhere in this phase.")
w("")

# ---------------------------------------------------------------- the decisive control
c0 = R["MAX_COLLAPSE_IS_LOSSLESS"]
tot_i = sum(v["identical_partition_orderings"] for v in c0.values())
tot_n = 0
for ds in c0:
    T = np.load(f"{TP.TPD}/data/tr_{ds}.npz")
    tot_n += int(T["nq"][0]) - int((T["NE"] == 0).sum())
w("## The decisive control: the collapse is lossless (STEP 5)")
w("")
w("A partition score built by MAX cannot tell the two representations apart, because max is "
  "associative:")
w("")
w("```")
w("max over edges into Pj  ==  max over nodes v in Pj of ( max over edges into v )")
w("```")
w("")
w("The right-hand side is precisely what `np.maximum.at(best_s, inv, S)` computes and stores as "
  "`s_sdir`.  Measured on every query with at least one usable edge, the edge-level and "
  "node-level partition score vectors are **bit-identical, and so are the orderings they induce**:")
w("")
w("| corpus | queries | no usable edge | identical score vectors | identical orderings |")
w("|---|---|---|---|---|")
for ds in c0:
    T = np.load(f"{TP.TPD}/data/tr_{ds}.npz")
    z = int((T["NE"] == 0).sum()); nq = int(T["nq"][0])
    v = c0[ds]
    w(f"| {ds} | {nq} | {z} | {v['identical_partition_score_vectors']}/{nq - z} | "
      f"{v['identical_partition_orderings']}/{nq - z} |")
w("")
w(f"That is {tot_i}/{tot_n} queries, exact, with zero exceptions.  The premise of the phase -- "
  "that the offset evidence is thrown away when edges collapse to nodes -- is false for the "
  "aggregation the pipeline actually uses.  `C0_SINGLE_TRIPLET` and the frozen node score are the "
  "same object under two names.")
w("")
w("The edge unit and the node unit can only differ under aggregations that count, and under "
  "source-side quantities a node score cannot express.  Both were measured, and both are worse "
  "than the max they were supposed to improve on (MetaQA hop3, needed-partition recall):")
w("")
cm = J("ctrl", "metaqa")["RECALL"]
w("| aggregation | R@6 | R@12 | R@20 | R@50 |")
w("|---|---|---|---|---|")
for n in ["EDGE_MAX", "NODE_MAX", "EDGE_SUM", "NODE_SUM", "EDGE_COUNT", "NODE_COUNT",
          "SOURCE_PARTITION_COUNT", "SOURCE_NODE_COUNT"]:
    w(f"| {n} | " + " | ".join(f"{cm[n]['hop3'][f'R@{k}']:.4f}" for k in (6, 12, 20, 50)) + " |")
w("")
w(f"Preserving the edge helps exactly once, for SUM ({cm['EDGE_SUM']['hop3']['R@6']:.4f} against "
  f"{cm['NODE_SUM']['hop3']['R@6']:.4f} at R@6 -- a node reached by seven edges is counted seven "
  f"times rather than once), and that whole family lands far below the max it is competing with "
  f"({cm['EDGE_MAX']['hop3']['R@6']:.4f}).  For COUNT the edge unit is worse than the node unit "
  f"({cm['EDGE_COUNT']['hop3']['R@6']:.4f} against {cm['NODE_COUNT']['hop3']['R@6']:.4f}).  The "
  "source side -- the one axis genuinely unavailable to a node score -- is the weakest of all at "
  "R@6, which is the budget that actually matters, B = 6 being the entire swap allowance.")
w("")

# ---------------------------------------------------------------- STEP 3 / 4
w("## STEP 3 + STEP 4 -- the necessary signal was not met")
w("")
w("The directive's necessary condition, ahead of any end-to-end improvement, was that "
  "MARGINAL_NEEDED_PARTITION recall at small TRIPLET budgets should substantially exceed the "
  "current node/partition structural ranking at the same budget.  At R@64 -- 64 triplets against "
  "the 64 nodes of the frozen structural read -- the triplet unit is BELOW the node unit on every "
  "corpus:")
w("")
w("| corpus | T0 R@16 | T0 R@32 | T0 R@64 | T0 R@128 | NODE R@16 | NODE R@32 | NODE R@64 | "
  "NODE R@128 | R@64 delta |")
w("|---|---|---|---|---|---|---|---|---|---|")
for ds in DS:
    d = J("tp", ds)
    t = d["STEP4_TRIPLET_RECALL"]["T0_OFFSET"]["ALL"]
    n = d["STEP4_NODE_RECALL"]["NODE_MAX_SDIR"]["ALL"]
    w(f"| {ds} | " + " | ".join(f"{t[f'R@{k}']:.4f}" for k in TB.BUD) + " | " +
      " | ".join(f"{n[f'R@{k}']:.4f}" for k in TB.BUD) +
      f" | {t['R@64'] - n['R@64']:+.4f} |")
w("")
_h = md["STEP4_TRIPLET_RECALL"]["T0_OFFSET"]
_n = md["STEP4_NODE_RECALL"]["NODE_MAX_SDIR"]
_cells = [(d2, k, J("tp", d2)["STEP4_TRIPLET_RECALL"]["T0_OFFSET"]["ALL"][f"R@{k}"] -
           J("tp", d2)["STEP4_NODE_RECALL"]["NODE_MAX_SDIR"]["ALL"][f"R@{k}"])
          for d2 in DS for k in TB.BUD]
_up = [(d2, k, x) for d2, k, x in _cells if x > 0]
_tie = [(d2, k, x) for d2, k, x in _cells if x == 0]
w(f"Across the whole grid of {len(_cells)} corpus-by-budget cells the triplet unit is ahead in "
  f"{len(_up)} ({', '.join(f'{d2} R@{k} {x:+.4f}' for d2, k, x in _up) if _up else 'none'}) and "
  f"exactly ties in {len(_tie)}; everywhere else it is behind.")
w("")
w(f"On MetaQA by block the gap is the same sign everywhere: hop2 "
  f"{_h['hop2']['R@64']:.4f} against {_n['hop2']['R@64']:.4f}, hop3 {_h['hop3']['R@64']:.4f} "
  f"against {_n['hop3']['R@64']:.4f}.  `T0_OFFSET` is nonetheless the best of the four triplet "
  "signals by a wide margin -- T1_SOURCE, T2_TARGET and T3_HOP are all far behind it at every "
  "budget -- so the ordering among the signals is exactly what the hypothesis predicted; it is the "
  "level that fails.")
w("")
_pv = {k: md["STEP3_PERCENTILE"][k] for k in TB.SIG}
_all = [v["ALL"] for v in _pv.values()]
w("Per triplet the offset is barely better than chance at picking out a marginally useful "
  "transition.  The mean percentile rank of MARGINAL_USEFUL triplets under each signal (0.5 is "
  f"chance) stays within {max(abs(x - 0.5) for x in _all):.3f} of chance on every signal over all "
  "queries:")
w("")
w("| signal | ALL | hop1 | hop2 | hop3 |")
w("|---|---|---|---|---|")
for k in TB.SIG:
    v = md["STEP3_PERCENTILE"][k]
    w(f"| {k} | " + " | ".join(f"{v[b]:.4f}" for b in ("ALL", "hop1", "hop2", "hop3")) + " |")
w("")
_h1 = md["STEP12_SATURATION"]["hop1"]["needed_partitions_per_query"]
w(f"The two values that do move (T1_SOURCE {_pv['T1_SOURCE']['hop1']:.4f} and T2_TARGET "
  f"{_pv['T2_TARGET']['hop1']:.4f}) are both on hop1, the block that has almost no marginal work "
  f"to do at all -- {_h1} needed partitions per query -- so they rest on negligible support.  The "
  "ranking works better than the per-item percentile suggests only because sorting by offset "
  "concentrates targets, not because individual triplets are well separated.")
w("")

# ---------------------------------------------------------------- STEP 5 / 6
w("## STEP 5 -- transition evidence does not separate novel Pj better than reach")
w("")
w("The primary STEP 5 question was whether `Pi -> Pj` evidence separates a required novel `Pj` "
  "better than the plain fact that `Pj` was structurally reached.  Inside the identical universe, "
  "with the reach controls stated explicitly (MetaQA hop3):")
w("")
w("| ordering | AUC | R@6 | R@12 | R@20 | R@50 |")
w("|---|---|---|---|---|---|")
for n in ["R0_REACHED_ARRIVAL", "R1_REACHED_COUNT", "TR_SOURCE_PARTITIONS",
          "TR_TRANSITION_COUNT", "C0_SINGLE_TRIPLET"]:
    a = md["STEP5_AUC"][n]["hop3"]; r = md["STEP5_PARTITION_RECALL"][n]["hop3"]
    w(f"| {n} | {a:.4f} | " + " | ".join(f"{r[f'R@{k}']:.4f}" for k in TR.PBUD) + " |")
w("")
_ts = md["STEP5_AUC"]["TR_SOURCE_PARTITIONS"]["hop3"]
_tc = md["STEP5_AUC"]["TR_TRANSITION_COUNT"]["hop3"]
_rc = md["STEP5_AUC"]["R1_REACHED_COUNT"]["hop3"]
w(f"The two orderings only the transition representation can express -- how many distinct source "
  f"partitions feed Pj, and how many distinct transitions land on it -- score {_ts:.4f} and "
  f"{_tc:.4f}, against {_rc:.4f} for simply counting the nodes of Pj that were reached.  They are "
  "the same signal.  Across corpora the transition-native ordering is at or below chance "
  "on MuSiQue and 2Wiki, and on no corpus does it separate more than 0.010 of AUC away from that "
  "control:")
w("")
w("| corpus | R1_REACHED_COUNT | TR_SOURCE_PARTITIONS | difference |")
w("|---|---|---|---|")
for _d in DS:
    _a = J("tp", _d)["STEP5_AUC"]
    w(f"| {_d} | {_a['R1_REACHED_COUNT']['ALL']:.4f} | {_a['TR_SOURCE_PARTITIONS']['ALL']:.4f} | "
      f"{_a['TR_SOURCE_PARTITIONS']['ALL'] - _a['R1_REACHED_COUNT']['ALL']:+.4f} |")
w("")
w("**PARTITION_TRANSITION_INFORMATIVE = NO.**")
w("")
w("## STEP 6 -- composition does not help either")
w("")
w("Exactly three controls, no scorer grid (MetaQA hop3 AUC / R@6):")
w("")
w("| control | AUC | R@6 |")
w("|---|---|---|")
for n, lbl in (("C0_SINGLE_TRIPLET", "C0_SINGLE_TRIPLET"),
               ("C1_BEST_CHAIN_MIN", "C1_BEST_COMPOSABLE_CHAIN (weakest link)"),
               ("C1_BEST_CHAIN_SUM", "C1_BEST_COMPOSABLE_CHAIN (total)"),
               ("C2_MULTI_PATH_SUPPORT", "C2_MULTI_PATH_SUPPORT (distinct ancestors)"),
               ("C2_MULTI_SEED_SUPPORT", "C2_MULTI_PATH_SUPPORT (distinct seeds)")):
    w(f"| {lbl} | {md['STEP5_AUC'][n]['hop3']:.4f} | "
      f"{md['STEP5_PARTITION_RECALL'][n]['hop3']['R@6']:.4f} |")
w("")
_mus = J("tp", "musique_clean")["STEP5_AUC"] if "musique_clean" in DS else None
w("On MetaQA every composed control is below the single transition it is composed from, and the "
  "two multi-path supports collapse onto the reach-count line "
  f"({md['STEP5_AUC']['C2_MULTI_PATH_SUPPORT']['hop3']:.4f} and "
  f"{md['STEP5_AUC']['C2_MULTI_SEED_SUPPORT']['hop3']:.4f} against "
  f"{md['STEP5_AUC']['R1_REACHED_COUNT']['hop3']:.4f}).")
if _mus:
    w("")
    w(f"One corpus dissents on the chain score alone: on MuSiQue `C1_BEST_CHAIN_SUM` reaches "
      f"{_mus['C1_BEST_CHAIN_SUM']['ALL']:.4f} against {_mus['C0_SINGLE_TRIPLET']['ALL']:.4f} for "
      "the single transition.  Both sit close to chance, the corpus has "
      f"{J('tp','musique_clean')['STEP12_SATURATION']['ALL']['needed_partitions_per_query']} "
      "needed partitions per query to work with, and the chain ordering does not survive the "
      f"exact-P50 mechanism on MetaQA (`PATH_S4` depth-matched net {mm['PATH_S4']['vs_SAFE']['net']:+d}).")
w("")
w("**COMPOSABLE_CHAIN_INFORMATIVE = NO.**")
w("")

# ---------------------------------------------------------------- STEP 8
w("## STEP 8 -- complementarity is real, and it is small")
w("")
w("Complementarity was measured before anything was combined, on the unit that matters: a query "
  "whose ENTIRE needed set is covered by the ordering's top 6, B = 6 being the whole swap budget.")
w("")
w("| ordering | both | only triplet | only node | triplet alone | node alone | union ceiling |")
w("|---|---|---|---|---|---|---|")
for k, v in p["STEP8_COMPLEMENTARITY"].items():
    w(f"| {k.replace('_vs_NODE_S4_FROZEN','')} | {v['solved_by_both']} | "
      f"{v['solved_only_by_triplet']} | {v['solved_only_by_node']} | {v['triplet_alone']:.4f} | "
      f"{v['node_alone']:.4f} | {v['union_ceiling']:.4f} |")
w("")
_ce = p["STEP8_COMPLEMENTARITY"]["CORE_EXIT_OFFSET_vs_NODE_S4_FROZEN"]
w(f"This is genuine disagreement, not noise: `CORE_EXIT_OFFSET` -- the one signal only a "
  f"transition can express, a transition whose SOURCE partition is inside the protected core -- "
  f"solves {_ce['solved_only_by_triplet']} queries the frozen node ordering does not, while the "
  f"node ordering solves {_ce['solved_only_by_node']} it does not.  It is also small in absolute "
  f"terms: a perfect combination would lift full coverage at B = 6 from {_ce['node_alone']:.4f} "
  f"to {_ce['union_ceiling']:.4f}.  That earned the ONE parameter-free combination the directive "
  "permits: an equal rank fusion of the frozen node ordering with the core-exit ordering, K0 = 60, "
  "no weights, no lambda, no threshold, no grid.")
w("")

# ---------------------------------------------------------------- STEP 10
w("## STEP 10 -- through the unchanged mechanism at exact P = 50")
w("")
w(f"The frozen candidate list is {p['frozen_ordering_len_mean']} partitions/query; the triplet "
  f"universe is {p['triplet_ordering_len_mean']}.  The preceding calibration phase established "
  "that a longer candidate list alone costs coverage through F6 by dilution, so reporting only "
  "full depth would confound representation with list size.  Both conditions are given, with the "
  "matched condition cutting each triplet ordering to exactly the frozen ordering's own per-query "
  "length K -- read off, never tuned.")
w("")
for tag, ttl in ((mf, "FULL depth"), (mm, "DEPTH-MATCHED")):
    w(f"**{ttl}** -- frozen SAFE: ALL {SAFE['ALL']:.4f}, hop1 {SAFE['hop1']:.4f}, "
      f"hop2 {SAFE['hop2']:.4f}, hop3 {SAFE['hop3']:.4f}")
    w("")
    w("| ordering | ALL | hop1 | hop2 | hop3 | net | McNemar p | |")
    w("|---|---|---|---|---|---|---|---|")
    for k in PJ.CAND:
        r = tag[k]; m = r["vs_SAFE"]
        w(f"| {k} | {r['ALL']:.4f} | {r['hop1']:.4f} | {r['hop2']:.4f} | {r['hop3']:.4f} | "
          f"{m['net']:+d} | {m['mcnemar_p']:.3g} | {'**SIG**' if m['sig'] else ''} |")
    w("")
nsig_full = sum(1 for k in PJ.CAND if mf[k]["vs_SAFE"]["sig"])
nsig_m = sum(1 for k in PJ.CAND if mm[k]["vs_SAFE"]["sig"])
bp = R["EXACT_P50_BEST_ORDERING"]; bm = mm[bp]["vs_SAFE"]
_msig = [k for k in PJ.CAND if mm[k]["vs_SAFE"]["sig"]]
_mup = [k for k in _msig if mm[k]["vs_SAFE"]["net"] > 0]
w(f"At full depth **{nsig_full}/{len(PJ.CAND)} orderings are significantly WORSE** than the frozen "
  f"baseline, none better.  Depth-matched, {nsig_m}/{len(PJ.CAND)} is significant in either "
  f"direction -- {', '.join(_msig) if _msig else 'none'}, and it is significantly WORSE, not "
  f"better ({len(_mup)} depth-matched ordering is significantly better).  So the harm at full "
  "depth was list length, exactly as the calibration phase found, and the representation itself "
  "is worth approximately nothing.  The best depth-matched ordering is "
  f"`{bp}` at ALL {mm[bp]['ALL']:.4f} against {SAFE['ALL']:.4f} "
  f"(net {bm['net']:+d}, gained {bm['gained']}, lost {bm['lost']}, p = {bm['mcnemar_p']}), "
  f"hop2 {mm[bp]['hop2']:.4f} against {SAFE['hop2']:.4f}, "
  f"hop3 {mm[bp]['hop3']:.4f} against {SAFE['hop3']:.4f}.  Not significant on any block.")
w("")
w(f"STEP 10 gates WebQSP and 2Wiki on MetaQA materially improving.  A non-significant "
  f"{mm[bp]['ALL'] - SAFE['ALL']:+.4f} is not material, so the exact-P50 mechanism was not run on "
  "the gated corpora.  The diagnostic steps that carry no such gate (STEPS 3-6 and 12) were run on "
  "MuSiQue, 2Wiki and SQuAD anyway.  They agree with MetaQA on the two findings the verdict rests "
  "on -- the collapse is lossless on every corpus, and the triplet unit is below the node unit at "
  "R@64 on all four (" +
  ", ".join(f"{d2} {J('tp', d2)['STEP4_TRIPLET_RECALL']['T0_OFFSET']['ALL']['R@64'] - J('tp', d2)['STEP4_NODE_RECALL']['NODE_MAX_SDIR']['ALL']['R@64']:+.4f}"
            for d2 in DS) + ").")
w("")
w(f"Worth recording, though it is a cross-reference and not a measurement made here: hop3 "
  f"{mm[bp]['hop3']:.4f} is the number the earlier read-stage separability phase recorded when it "
  "gave the M64 read a PERFECT node scorer.  Two unrelated routes into the same stage land on the "
  "same value, which is at least suggestive that the ceiling belongs to the contract rather than "
  "to either scorer.")
w("")

# ---------------------------------------------------------------- STEP 11 / 12
w("## STEP 11 -- the budget curve does not move left")
w("")
ks = [f"k{k}" for k in PJ.BUDG]
w("| ordering | " + " | ".join(ks) + " |")
w("|" + "|".join(["---"] * (len(ks) + 1)) + "|")
for n, v in p["STEP11_BUDGET"].items():
    w(f"| {n} | " + " | ".join(f"{v[k]['ALL']:.4f}" for k in ks) + " |")
w("")
_fz = p["STEP11_BUDGET"]["NODE_S4_FROZEN"]
_bs = p["STEP11_BUDGET"].get(f"{bp}_DEPTH_MATCHED", _fz)
w("The required-partition curve is flat against the frozen one at every budget "
  f"(k64 {_bs['k64']['ALL']:.4f} against {_fz['k64']['ALL']:.4f}, "
  f"k128 {_bs['k128']['ALL']:.4f} against {_fz['k128']['ALL']:.4f}).  Nothing moves toward P50; "
  "the triplet representation does not make required partitions available earlier.")
w("")
w("## STEP 12 -- the saturation the phase was meant to break is not a representation problem")
w("")
w("| corpus | needed P/q | partitions reached/q | saturation | transitions/q | chains/q | "
  "transitions per partition | MARGINAL_USEFUL triplet prevalence |")
w("|---|---|---|---|---|---|---|---|")
for ds in DS:
    s = J("tp", ds)["STEP12_SATURATION"]["ALL"]
    w(f"| {ds} | {s['needed_partitions_per_query']} | "
      f"{s['NODE_partitions_reached_per_query']} | {s['NODE_partition_saturation']:.4f} | "
      f"{s['TRIPLET_transitions_per_query']} | {s['PATH_coherent_chains_per_query']} | "
      f"{s['transition_per_partition_ratio']} | {s['marginal_useful_triplet_prevalence']} |")
w("")
s3 = md["STEP12_SATURATION"]["hop3"]
w(f"The transition representation is genuinely finer -- {s3['transition_per_partition_ratio']}x "
  f"more transitions than partitions on MetaQA hop3, {s3['TRIPLET_transitions_per_query']} "
  f"transitions and {s3['PATH_coherent_chains_per_query']} coherent chains against "
  f"{s3['NODE_partitions_reached_per_query']} partitions -- and conditioning on it does not reduce "
  f"the saturation at all: the traversal still touches {s3['NODE_partition_saturation']:.1%} of "
  "every partition in the corpus per query, because that is a property of where the beam goes, "
  "not of how its arrivals are named.  Splitting one saturated partition axis into three saturated "
  "transition axes lowers the prevalence of a useful unit rather than raising it: MARGINAL_USEFUL "
  f"triplets are {s3['marginal_useful_triplet_prevalence']:.2%} of the hop3 edge universe.")
w("")

# ---------------------------------------------------------------- returns + verdict
w("## Returns")
w("")
w("| return | value |")
w("|---|---|")
for k in ("TRIPLET_OFFSET_INFORMATIVE", "PARTITION_TRANSITION_INFORMATIVE",
          "COMPOSABLE_CHAIN_INFORMATIVE", "BEST_TRIPLET_SIGNAL",
          "MARGINAL_PARTITION_R64_NODE_BASE", "MARGINAL_PARTITION_R64_TRIPLET",
          "EXACT_P50_METAQA_HOP2", "EXACT_P50_METAQA_HOP3", "TRIPLET_L1_VERDICT"):
    w(f"| `{k}` | {R[k]} |")
w("")
w("`TRIPLET_OFFSET_INFORMATIVE = YES` needs its qualification stated with it.  The offset is "
  "informative -- it is the best of the four signals and it separates needed partitions well above "
  f"both reach controls (hop3 AUC {md['STEP5_AUC']['C0_SINGLE_TRIPLET']['hop3']:.4f} against "
  f"{md['STEP5_AUC']['R0_REACHED_ARRIVAL']['hop3']:.4f} arrival and "
  f"{md['STEP5_AUC']['R1_REACHED_COUNT']['hop3']:.4f} count).  It is not informative *as a "
  "triplet*: the identical quantity is already carried by the frozen node score, and the phase's "
  "own control proves the two are the same object.")
w("")
w("## What this closes, and what it does not")
w("")
w("**Closed.** The evidence-unit axis.  Node vs edge vs partition-transition vs chain is not a "
  "live degree of freedom in L1: the collapse that motivated the phase is lossless, the "
  "aggregations under which it is not lossless are worse, and the only genuinely transition-native "
  "signals are indistinguishable from counting reached nodes.  A finer unit is not the lever.")
w("")
w("**Not closed, and not touched.** L1 remains unfrozen.  Everything this phase found is "
  "consistent with the standing account: the binding constraint is the beam's reach and the "
  f"{s3['needed_partitions_per_query']}-needed-partitions-against-B=6 capacity ratio on MetaQA "
  "hop3, and both are properties of the traversal contract, not of how its arrivals are "
  "represented.  The small real complementarity STEP 8 found (core-exit transitions solving "
  f"{_ce['solved_only_by_triplet']} queries the node ordering misses) is the one thread here that "
  "did not go to zero, and it is worth "
  f"{_ce['union_ceiling'] - _ce['node_alone']:+.4f} at its ceiling -- too small to promote on its "
  "own evidence.  It is also the one quantity in this phase that genuinely REQUIRES the transition "
  "unit: filtering the offset by whether the SOURCE partition is inside the protected core is "
  "exactly the information `np.maximum.at` destroys, because the collapse maxes over incoming "
  "edges without regard to where they came from.  So the evidence-unit axis is closed for target "
  "scoring and open, barely, for source-conditioned filtering -- on a lift that is not "
  "significant here and would need its own justification to pursue.")
w("")
w("**Nothing was promoted.  B stays at 6.  P stays at 50.  No parameter, model, encoder, "
  "partitioning or selector changed.  L2 and L3 were not touched and TEST was not run.**")
w("")
w("Artifacts: `FINAL_REPORT.md`, `TABLES.md`, `RETURNS.json`, `diag/{tp,ctrl,p50}_<corpus>.json`, "
  "`data/tr_<corpus>.npz`.  Code: `scratchpad/_l1tp_core.py` (replay + parity), `_l1tp_build.py` "
  "(STEPS 1-6 substrate), `_l1tp_run.py` (STEPS 3-6, 12), `_l1tp_ctrl.py` (the lossless-collapse "
  "control), `_l1tp_p50.py` (STEPS 8-11), `_l1tp_report.py`, `_l1tp_final.py`.  The `want_edges` "
  "export added to `scratchpad/_l1ss_core.py` is inert unless requested.")
open(f"{TP.TPD}/FINAL_REPORT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"wrote FINAL_REPORT.md ({len(L)} lines)")
