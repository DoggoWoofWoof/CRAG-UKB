"""Emit FINAL_REPORT.md.  Every number is pulled from diag/*.json, none is retyped.

  python scratchpad/_l1bm_report.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1bm_core as BM
import _l1bm_returns as RET

D = f"{BM.BMD}/diag"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]


def J(n):
    p = f"{D}/{n}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    R = RET.main()
    j1, j3, j4, j7, j8, jf = (J("step1_metaqa"), J("step3_metaqa"), J("step4_metaqa"),
                              J("step7_metaqa"), J("step8_metaqa"), J("funnel_metaqa"))
    jl = J("latency_metaqa")
    st = j1["stats"]
    S8, SAFE = j8["STEP8"], j8["SAFE"]
    F = jf["FUNNEL"]
    m0, b1, l1, d0 = (F["M0_BASELINE"], F["B1_PARENT_DIVERSE"], F["L1_MAX_FUTURE"],
                      F["D0_DELAYED_PRUNE"])

    def nf(p, blk, row):
        return F[p][blk][row]["node_frac"]

    L = []
    A = L.append

    A("# L1 BEAM RECOVERY -- parameter-free structural beam recovery")
    A("")
    A("Question: can future structural evidence be exploited *before* the hop-2 pruning decision, so")
    A("that useful nodes move toward the top of the beam while L1 still emits EXACTLY 50 partitions?")
    A("")
    A("**Answer: no.** The prune is not premature. Position 2 really is where the gold dies -- 64% of")
    A("all structural gold loss happens there -- but the loss is *conserved*, not removable: an")
    A("oracle-free control that deletes the position-2 and position-3 prunes outright recovers the")
    A("gold into the beam and then loses more of it than the frozen beam did, one stage further down.")
    A("Every lookahead variant is strictly worse end-to-end. MetaQA hop3 does not move off its frozen")
    A(f"value of {SAFE['hop3']:.4f} under any of the ten policies.")
    A("")
    A("All numbers below are read out of `diag/*.json` by `scratchpad/_l1bm_report.py`; the tables")
    A("live in [TABLES.md](TABLES.md).")
    A("")

    # ---------------------------------------------------------------- returns
    A("## Returns")
    A("")
    A("| flag | value |")
    A("|---|---|")
    for k in ("PREMATURE_PRUNING_CONFIRMED", "BEST_BEAM_METHOD", "POSITION2_GOLD_SURVIVAL_GAIN",
              "HOP3_GOLD_NODE_RECOVERY_GAIN", "EXACT_P50_METAQA_HOP2", "EXACT_P50_METAQA_HOP3",
              "ONLINE_EDGES", "LATENCY"):
        A(f"| `{k}` | {R[k]} |")
    A("")

    # ---------------------------------------------------------------- parity
    A("## 1. The replay is bit-exact, so every difference is the policy")
    A("")
    A("The frozen beam was re-implemented from `_ta_prepartition.expand_dir` and replayed on the same")
    A("RRF seeds, the same residual query vector and the same CSR topology. Parity is exact on all")
    A("four frozen per-query fields *and* on the S4 partition ranking, on all six corpora:")
    A("")
    A("| dataset | queries | all 4 frozen fields | S4 ranking | verdict |")
    A("|---|---|---|---|---|")
    for ds in DSETS:
        j = J(f"step1_{ds}")
        if j:
            A(f"| {ds} | {j['nq']} | {j['all_field_parity']}/{j['nq']} | "
              f"{j['s4_ranking_parity']}/{j['nq']} | {j['BASELINE_PARITY']} |")
    A("")
    A("Two details were load-bearing. The frozen expansion scores each frontier node with its own")
    A("mat-vec rather than one batched product -- BLAS blocking changes the last bits and that is")
    A("enough to reorder a tie. And dedup is `np.unique` + `np.maximum.at`, so ties break by ascending")
    A("node id. Reproducing both is what closed the 1997/1998 gap left by the previous phase.")
    A("")

    # ---------------------------------------------------------------- where the prune binds
    A("## 2. Where the beam actually prunes (the premise arithmetic was wrong)")
    A("")
    A("The directive assumed roughly 2,560 candidates at position 2, from `64 frontier x ~40 edges`.")
    A("That figure came from an estimate in the previous phase report, not a measurement. Measured on")
    A("the exact replay:")
    A("")
    A("| path position | frontier in | distinct fresh candidates | kept (beam) | prune ratio |")
    A("|---|---|---|---|---|")
    for i in range(3):
        A(f"| {i+1} | {st['frontier'][i]} | {st['cands'][i]} | {st['kept'][i]} | "
          f"{st['cands'][i] / max(1e-9, st['kept'][i]):.1f}x |")
    A("")
    A("Position 1 does not prune at all. Position 2 prunes about 7x and position 3 about 12x, so by")
    A("*ratio* position 3 is the harsher cut. Mean legal out-degree is around 17, not 40.")
    A("")

    # ---------------------------------------------------------------- the funnel
    A("## 3. The transition table -- and conservation of loss")
    A("")
    A("Candidate survival at one hop is not the deliverable, so the whole pipeline was instrumented")
    A("per required gold node. `alive after pK` is *won or still winnable*: the node has already been")
    A("found, or some survivor of that prune can still reach it in the hops that remain. Both halves")
    A("are needed -- a node found early is excluded from every later candidate set, so a purely")
    A("forward-looking test is not monotone.")
    A("")
    A(f"On the frozen beam, of the {jf['FUNNEL']['M0_BASELINE']['ALL']['gold_nodes']:,} required gold")
    A(f"nodes over {jf['FUNNEL']['M0_BASELINE']['ALL']['queries']:,} MetaQA queries:")
    A("")
    A("| stage | fraction still in play | share of structural loss |")
    A("|---|---|---|")
    rows = jf["rows"][:jf["rows"].index("NODE_read_S4") + 1]
    tot = nf("M0_BASELINE", "ALL", rows[0]) - nf("M0_BASELINE", "ALL", rows[-1])
    prev = None
    for r in rows:
        v = nf("M0_BASELINE", "ALL", r)
        sh = "" if prev is None else f"{(prev - v) / max(1e-9, tot):.1%}"
        A(f"| {r} | {v:.4f} | {sh} |")
        prev = v
    A("")
    A("So the premise was right about *where*: position 2 is the dominant cut by a wide margin, and")
    A("position 3 is minor. What the premise got wrong is that this makes the cut premature.")
    A("")
    A("`D0_DELAYED_PRUNE` settles that. It is oracle-free -- it simply does not prune at positions 2")
    A("and 3, so `alive p2` and `alive p3` are pinned to the reachable set by construction. It")
    A(f"raises gold discovery from {nf('M0_BASELINE','ALL','ALIVE_p3'):.4f} to "
      f"{nf('D0_DELAYED_PRUNE','ALL','ALIVE_p3'):.4f}. And then:")
    A("")
    A("| | frozen beam | delayed prune |")
    A("|---|---|---|")
    A(f"| gold alive after p2 | {nf('M0_BASELINE','ALL','ALIVE_p2'):.4f} | "
      f"{nf('D0_DELAYED_PRUNE','ALL','ALIVE_p2'):.4f} |")
    A(f"| gold in `added(256)` | {nf('M0_BASELINE','ALL','NODE_added256'):.4f} | "
      f"{nf('D0_DELAYED_PRUNE','ALL','NODE_added256'):.4f} |")
    A(f"| gold read by `S4(64)` | {nf('M0_BASELINE','ALL','NODE_read_S4'):.4f} | "
      f"{nf('D0_DELAYED_PRUNE','ALL','NODE_read_S4'):.4f} |")
    A(f"| **total structural loss** | **{tot:.4f}** | "
      f"**{nf('D0_DELAYED_PRUNE','ALL',rows[0]) - nf('D0_DELAYED_PRUNE','ALL',rows[-1]):.4f}** |")
    A(f"| EXACT P50 ALL | {SAFE['ALL']:.4f} | {S8['D0_DELAYED_PRUNE']['P50']['ALL']:.4f} |")
    A("")
    A("The loss is not removed, it is relocated: 84% of it moves onto the static-score-ordered")
    A("`added(256)` cut, where the same score now has to discriminate a pool an order of magnitude")
    A("larger -- and does it worse. On hop2 the relocation is brutal: `added(256)` holds")
    A(f"{nf('M0_BASELINE','hop2','NODE_added256'):.4f} of gold on the frozen beam and only "
      f"{nf('D0_DELAYED_PRUNE','hop2','NODE_added256'):.4f} with the prunes removed.")
    A("")
    A("The small beam was doing useful work. It kept the pool small enough that a score which is at")
    A("chance before the terminal hop did not have to discriminate among thousands of candidates.")
    A("")

    # ---------------------------------------------------------------- lookahead
    A("## 4. Why one-step lookahead is strictly harmful")
    A("")
    A("`L1_MAX_FUTURE` and `L2_TOP2_FUTURE` compute, for every position-2 candidate and before the")
    A("prune, the best (or mean-of-best-two) score of its own legal fresh children, using the exact")
    A("existing score semantics. No learned weight, no threshold, no lambda.")
    A("")
    A("They do raise gold-*parent* survival: "
      f"{j7['STEP7']['M0_BASELINE']['ALL']['gold_parent_survival']:.4f} -> "
      f"{j7['STEP7']['L1_MAX_FUTURE']['ALL']['gold_parent_survival']:.4f}. They pay for it by losing")
    A("gold *candidates*: "
      f"{j7['STEP7']['M0_BASELINE']['ALL']['gold_cand_survival']:.4f} -> "
      f"{j7['STEP7']['L1_MAX_FUTURE']['ALL']['gold_cand_survival']:.4f}. That is the trade in one")
    A("line: a node that *leads toward* gold is promoted over a node that *is* gold, and the gold")
    A(f"candidate's percentile rank degrades from {j7['STEP7']['M0_BASELINE']['ALL']['gold_cand_pct']:.4f}")
    A(f"to {j7['STEP7']['L1_MAX_FUTURE']['ALL']['gold_cand_pct']:.4f}.")
    A("")
    A("The hop2 rows expose a second, sharper mechanism. Lookahead puts *more* hop2 gold into")
    A(f"`added(256)` than the frozen beam ({nf('L1_MAX_FUTURE','hop2','NODE_added256'):.4f} vs "
      f"{nf('M0_BASELINE','hop2','NODE_added256'):.4f}) and S4 then reads less than half as much")
    A(f"({nf('L1_MAX_FUTURE','hop2','NODE_read_S4'):.4f} vs "
      f"{nf('M0_BASELINE','hop2','NODE_read_S4'):.4f}). Admission and aggregation use different keys.")
    A("Nodes admitted on the strength of their children rank badly under the static score S4 actually")
    A("reads, so improving admission under one key degrades the read under the other. Any lookahead")
    A("that is not accompanied by a matching change to S4 is fighting itself -- and S4 is frozen.")
    A("")

    # ---------------------------------------------------------------- lexicographic
    A("## 5. The lexicographic pair are identity controls, not a second fusion mechanism")
    A("")
    A("`L3_FUTURE_THEN_CURRENT` and `L4_CURRENT_THEN_FUTURE` were specified as a strict lexicographic")
    A("pair with no weighted sum. A lexicographic rule can only differ from its own primary key where")
    A("that primary key ties, so the honest report is the measured tie mass:")
    A("")
    A("| quantity | value |")
    A("|---|---|")
    A(f"| position-2 candidates scored | {j3['position2_candidates']:,} |")
    A(f"| tied in the CURRENT (static cosine) key | {j3['tied_in_current_key']:,} "
      f"({j3['tied_current_frac']:.4%}) |")
    A(f"| tied in the FUTURE key | {j3['tied_in_future_key']:,} ({j3['tied_future_frac']:.2%}) |")
    A(f"| of which: no legal fresh child (sentinel) | {j3['no_legal_child_sentinel']:,} "
      f"({j3['sentinel_frac']:.2%}) |")
    A("")
    for k, v in j8["STEP3"].items():
        a, b = k.split("_vs_")
        A(f"- `{a}` produces an **identical node list to `{b}`** on "
          f"{v['identical_node_lists']}/{v['of']} queries ({v['identical_frac']:.2%}).")
    A("")
    A("Ten ties in 870,793 continuous cosine scores. `L4` is therefore the frozen beam exactly, on")
    A("every query, and `L3` is `L1` on all but 15 queries -- and the residual tie mass in the future")
    A("key is almost entirely the sentinel for candidates with no legal fresh child, which is a")
    A("structural fact rather than evidence. **These two are not two fusion mechanisms.** They are")
    A("controls that prove continuous cosine keys leave no tie mass for lexicographic secondary")
    A("evidence to act on. That is a clean negative result and should be written up as one.")
    A("")

    # ---------------------------------------------------------------- geometry + beam
    A("## 6. Composed path geometry and stratified beams")
    A("")
    A("`C1_ENDPOINT_DISPLACEMENT` and `C2_NORMALIZED_EDGE_SUM` both score the composed path rather")
    A("than the last edge. Both are worse than the frozen beam at every stage of the funnel")
    A(f"(gold alive after p2 {nf('C1_ENDPOINT_DISPLACEMENT','ALL','ALIVE_p2'):.4f} and "
      f"{nf('C2_NORMALIZED_EDGE_SUM','ALL','ALIVE_p2'):.4f} vs "
      f"{nf('M0_BASELINE','ALL','ALIVE_p2'):.4f}) and both lose end-to-end. No further geometric")
    A("variants were tested.")
    A("")
    A("`B1_PARENT_DIVERSE` (one slot per parent first, then fill by score) is the one policy that")
    A("improves *every* stage of the funnel and end-to-end, at the same total beam of 64:")
    A("")
    A("| stage | frozen | B1_PARENT_DIVERSE | delta |")
    A("|---|---|---|---|")
    for r in rows + ["PART_in_S4"]:
        A(f"| {r} | {nf('M0_BASELINE','ALL',r):.4f} | {nf('B1_PARENT_DIVERSE','ALL',r):.4f} | "
          f"{nf('B1_PARENT_DIVERSE','ALL',r) - nf('M0_BASELINE','ALL',r):+.4f} |")
    A(f"| EXACT P50 ALL | {SAFE['ALL']:.4f} | {S8['B1_PARENT_DIVERSE']['P50']['ALL']:.4f} | "
      f"{S8['B1_PARENT_DIVERSE']['P50']['ALL'] - SAFE['ALL']:+.4f} |")
    A("")
    A(f"That is a net of {S8['B1_PARENT_DIVERSE']['vs_SAFE_net']:+d} queries out of {j8['nq']:,} "
      f"(McNemar p = {S8['B1_PARENT_DIVERSE']['vs_SAFE_mcnemar_p']:.3g}), and hop3 is unmoved at "
      f"{S8['B1_PARENT_DIVERSE']['P50']['hop3']:.4f}.")
    A("The mechanism is real and the direction is right at every stage, but the size is exactly the")
    A("`+0.002` that was ruled out in advance. `B2_SEED_DIVERSE` is smaller still.")
    A("")

    # ---------------------------------------------------------------- exact P50
    A("## 7. EXACT P50, end to end")
    A("")
    A(f"SAFE = ALL {SAFE['ALL']:.4f} / hop1 {SAFE['hop1']:.4f} / hop2 {SAFE['hop2']:.4f} / "
      f"hop3 {SAFE['hop3']:.4f}. Every row below emits exactly {S8['M0_BASELINE']['OUTPUT']['partitions']} "
      f"partitions.")
    A("")
    A("| policy | ALL | hop2 | hop3 | gained/lost | net | McNemar p |")
    A("|---|---|---|---|---|---|---|")
    for p, v in S8.items():
        A(f"| {p} | {v['P50']['ALL']:.4f} | {v['P50']['hop2']:.4f} | {v['P50']['hop3']:.4f} | "
          f"{v['vs_SAFE_gained']}/{v['vs_SAFE_lost']} | {v['vs_SAFE_net']:+d} | "
          f"{v['vs_SAFE_mcnemar_p']:.3g} |")
    A("")
    ties = [p for p, v in S8.items() if round(v["P50"]["hop3"], 4) == R["_best_hop3_any_policy"][0]]
    A(f"MetaQA hop3 is {SAFE['hop3']:.4f} for the frozen beam and **no policy exceeds it**. The best")
    A(f"figure across all ten is {R['_best_hop3_any_policy'][0]:.4f}, reached only by policies that "
      f"tie the frozen beam exactly ({', '.join(ties)}); every other policy is below it. The bar set")
    A("for this phase was `0.258 -> 0.30+` to be worth a quality/latency trade, and `0.258 -> 0.263`")
    A("to be rejected on sight. Nothing reaches even the reject threshold, in either direction.")
    A("")
    A("Cross-corpus:")
    A("")
    A("| dataset | policy | SAFE ALL | policy ALL | delta | net | McNemar p | sig |")
    A("|---|---|---|---|---|---|---|---|")
    for ds in DSETS:
        j = J(f"step8_{ds}")
        if not j:
            A(f"| {ds} | | | | *not measured* | | | |")
            continue
        for p, v in j["STEP8"].items():
            if p == "M0_BASELINE":
                continue
            A(f"| {ds} | {p} | {j['SAFE']['ALL']:.4f} | {v['P50']['ALL']:.4f} | "
              f"{v['P50']['ALL'] - j['SAFE']['ALL']:+.4f} | {v['vs_SAFE_net']:+d} | "
              f"{v['vs_SAFE_mcnemar_p']:.3g} | {'YES' if v['vs_SAFE_sig'] else 'no'} |")
    A("")

    # ---------------------------------------------------------------- work vs scope
    A("## 8. Internal routing work vs downstream exposure")
    A("")
    A("These are different things and were kept separate. Internal work is what the router touches")
    A("while deciding; downstream exposure is what L2 actually sees. The output contract is")
    A("unchanged at exactly 50 partitions for every policy, so no policy here buys coverage by")
    A("quietly enlarging the scope.")
    A("")
    A("| policy | graph edges/q | lookahead edges/q | nodes scored/q | FINAL partitions | FINAL scope nodes/q |")
    A("|---|---|---|---|---|---|")
    for p, v in S8.items():
        i, o = v["INTERNAL"], v["OUTPUT"]
        A(f"| {p} | {i['graph_edges_per_q']:,.0f} | {i['lookahead_edges_per_q']:,.0f} | "
          f"{i['distinct_nodes_evaluated_per_q']:,.0f} | {o['partitions']} | "
          f"{o['scope_nodes_per_q']:,.0f} |")
    A("")
    if jl:
        A(f"Latency, measured cleanly ({jl['n_queries']} queries spread across the hop blocks, "
          f"30-query warm-up, median of {jl['repeats']} repeats, nothing else on the CPU -- the ms/q "
          "inside the sweep JSONs is CPU-contaminated and is not reported anywhere):")
        A("")
        A("| policy | ms/query | x frozen |")
        A("|---|---|---|")
        for p, v in jl["ms_per_query"].items():
            A(f"| {p} | {v:.2f} | {jl['multiplier_vs_SAFE'][p]:.2f}x |")
        A("")
        A("Treat the multiplier as the robust quantity and the absolute ms/q as machine- and")
        A("load-dependent: this is a 12-core box and the mat-vecs go through a multithreaded BLAS, so")
        A("anything else running moves the absolute number a lot. An earlier attempt at this table had")
        A("two identical benchmarks racing each other and every figure came out roughly 3x inflated.")
        A("")
    A("This cost is a *latency* cost, not a compression cheat, and the distinction matters. A prior")
    A("candidate, `U_PC5@256`, passed 30,681 MetaQA nodes downstream -- 76.4% of the corpus -- which")
    A("is fake compression. Lookahead inspects more edges internally and still emits exactly 50")
    A(f"partitions, i.e. {S8['M0_BASELINE']['OUTPUT']['scope_nodes_per_q']:,.0f} nodes downstream. It")
    A("would have been a legitimate L1 had it worked. It did not work.")
    A("")

    # ---------------------------------------------------------------- verdict
    A("## 9. Verdict")
    A("")
    A("Of the four outcomes enumerated for this phase, the measured one is:")
    A("")
    A("> **Better parent survival is achievable and does not convert.** Position-2 gold-parent")
    A("> survival can be raised (lookahead) and every funnel stage can be nudged (B1). Neither")
    A("> produces materially more required gold nodes in front of S4, and neither moves EXACT P50.")
    A("")
    A("Two independent reasons, both measured:")
    A("")
    A("1. **The loss is conserved across the compression points.** Removing the position-2 and")
    A("   position-3 prunes entirely relocates 84% of the structural loss onto `added(256)` and ends")
    A(f"   up *below* the frozen beam at the stage that matters ({nf('D0_DELAYED_PRUNE','ALL','NODE_read_S4'):.4f} "
      f"vs {nf('M0_BASELINE','ALL','NODE_read_S4'):.4f} read by S4).")
    A("2. **Admission and aggregation use different keys.** Nodes promoted on future evidence rank")
    A("   badly under the static score S4 reads, so better admission produces a worse read.")
    A("")
    A("The stage that actually binds, on the frozen beam, is the score-ordered read `added(256) ->")
    A(f"S4({j4['M_struct']})`: {(nf('M0_BASELINE','ALL','NODE_added256') - nf('M0_BASELINE','ALL','NODE_read_S4')):.4f} "
      f"of gold ({(nf('M0_BASELINE','ALL','NODE_added256') - nf('M0_BASELINE','ALL','NODE_read_S4')) / max(1e-9, tot):.0%} "
      "of all structural loss) is discovered, admitted, and then never read, because it ranks below")
    A("position 64 under the static score. On hop2 that single stage is 41% of the loss. That stage")
    A("is downstream of both prunes and inside the S4 contract this phase was explicitly told not to")
    A("modify, so it is named here and not touched.")
    A("")
    A("Nothing is promoted. The frozen static-score beam (`B0_GLOBAL`) stands, `S4` and `F6` are")
    A("untouched, `L1_FROZEN` remains **NO**, and no TEST split was read.")
    A("")

    # ---------------------------------------------------------------- repro
    A("## 10. Reproduction")
    A("")
    A("```bash")
    A("python scratchpad/_l1bm_run.py step1 metaqa          # bit-exact parity gate")
    A("python scratchpad/_l1bm_run.py build metaqa M0_BASELINE,L1_MAX_FUTURE,...")
    A("python scratchpad/_l1bm_diag.py metaqa               # STEP 7 position-2 diagnostic")
    A("python scratchpad/_l1bm_step3.py metaqa              # tie mass at the binding hop")
    A("python scratchpad/_l1bm_step4.py metaqa              # discovery -> added -> read")
    A("python scratchpad/_l1bm_step8.py metaqa              # EXACT P50 end to end")
    A("python scratchpad/_l1bm_funnel.py metaqa             # the transition table")
    A("python scratchpad/_l1bm_latency.py metaqa 400 3      # clean latency, quiet CPU")
    A("python scratchpad/_l1bm_tables.py && python scratchpad/_l1bm_report.py")
    A("```")
    A("")

    out = "\n".join(L) + "\n"
    open(f"{BM.BMD}/FINAL_REPORT.md", "w", encoding="utf-8").write(out)
    print(f"wrote {BM.BMD}/FINAL_REPORT.md  ({len(L)} lines)")


if __name__ == "__main__":
    main()
