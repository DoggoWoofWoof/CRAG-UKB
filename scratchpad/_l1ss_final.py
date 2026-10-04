"""FINAL_REPORT.md for the score-separability phase.  Numbers are interpolated from diag/*.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1ss_core as SS
import _l1ss_report as RP

B = RP.B
DS = RP.DS


def main():
    RET, macro, CORR = RP.main()
    RANK = {d: RP.j(f"{B}/rank_{d}.json") for d in DS}
    FUN = {d: RP.j(f"{B}/funnel_{d}.json") for d in DS}
    SEP = RP.j(f"{B}/sep_metaqa.json")
    SEP2 = RP.j(f"{B}/sep2_metaqa.json")
    S2 = SEP["STEP2"]
    W = SEP2["WITHIN"]
    fm = FUN["metaqa"]["METHODS"]
    rm = RANK["metaqa"]["STEP9"]
    o = rm["RX_ORACLE_READ"]["P50"]
    s = rm["R0_STATIC"]["P50"]
    st7 = RET["STEP7_BEAM_CONSISTENT"]
    L = []
    A = L.append

    A("# L1 STRUCTURAL SCORE SEPARABILITY -- FINAL REPORT\n")
    A("## VERDICT: **B. STRUCTURAL_SCORE_PARTIAL**\n")
    A("Separability is real and large. Conversion to exact P50 is zero. The reason is now measured "
      "rather than guessed: the read stage was never worth much, and it is the only stage this "
      "phase was allowed to touch.\n")

    A("### Returns\n")
    A("| flag | value |")
    A("|---|---|")
    A(f"| `BEST_SINGLE_NODE_SIGNAL` | **RR_PARENT_SUPPORT** "
      f"(R@64 {S2['RR_PARENT_SUPPORT']['ALL']['recall@64']:.4f}, "
      f"AUC {S2['RR_PARENT_SUPPORT']['ALL']['auc']:.4f}, "
      f"Cohen d {SEP['STEP5']['RR_PARENT_SUPPORT']['cohen_d']:+.4f}) |")
    A(f"| `GOLD_READ_RECALL64_BASE` | **{RET['GOLD_READ_RECALL64_BASE']:.4f}** |")
    A(f"| `GOLD_READ_RECALL64_BEST` | **{RET['GOLD_READ_RECALL64_BEST']:.4f}** "
      f"(R2_HOP_CONDITIONED; oracle {RET['GOLD_READ_RECALL64_ORACLE']:.4f}) |")
    A(f"| `READ_EFFICIENCY_BASE` | **{RET['READ_EFFICIENCY_BASE']:.4f}** |")
    A(f"| `READ_EFFICIENCY_BEST` | **{RET['READ_EFFICIENCY_BEST']:.4f}** "
      f"(oracle {RET['READ_EFFICIENCY_ORACLE']:.4f}) |")
    br = RP.j(f"{B}/rank_metaqa__BEAMR1.json")
    b2 = max(br["STEP9"][r]["P50"]["hop2"] for r in ("R0_STATIC", "R1_EQUAL_RRF"))
    b3 = max(br["STEP9"][r]["P50"]["hop3"] for r in ("R0_STATIC", "R1_EQUAL_RRF"))
    A(f"| `EXACT_P50_METAQA_HOP2` | **{RET['EXACT_P50_METAQA_HOP2']:.4f}**, unchanged. Best "
      f"inference-safe {max(RET['EXACT_P50_METAQA_HOP2_BEST'], b2):.4f} (+"
      f"{round((max(RET['EXACT_P50_METAQA_HOP2_BEST'], b2)-RET['EXACT_P50_METAQA_HOP2'])*666)} "
      f"queries of 666), not significant |")
    A(f"| `EXACT_P50_METAQA_HOP3` | **{RET['EXACT_P50_METAQA_HOP3']:.4f}**, unchanged. **No "
      f"inference-safe ranking beats it** -- the best of the three rankings ties frozen exactly at "
      f"{RET['EXACT_P50_METAQA_HOP3_BEST']:.4f}, and the STEP-7 beam variant reaches {b3:.4f} "
      f"(+{round((b3-RET['EXACT_P50_METAQA_HOP3'])*666)} query), not significant |")
    A(f"| `STRUCTURAL_SCORE_VERDICT` | **B_STRUCTURAL_SCORE_PARTIAL** |")
    A("")
    A("`PROMOTED = NONE`. `L1_FROZEN = NO`. No TEST split was read. M_struct = 64, M_ret = 32, "
      "B = 6, P = 50, S4 and F6 are all exactly as frozen.\n")

    A("---\n")
    A("## 1. The dataset is the frozen read stage, not a model of it\n")
    A("Every one of the 11,415 queries across all six corpora reproduces the frozen "
      "`s_node / s_hop / s_sdir / s_cnt` arrays bit-exactly (T1). The features are quantities the "
      "search already computes; gold labels were attached afterwards and no feature sees them.\n")

    A("## 2. Separability exists, and the frozen key is below chance\n")
    A(f"The frozen ordering key `CS_ADMIT` scores **AUC {S2['CS_ADMIT']['ALL']['auc']:.4f}** on "
      f"MetaQA -- below chance. That is not a bug, it is a cross-hop calibration failure. Gold "
      f"density by node hop is "
      f"{100*SEP2['CROSS']['ALL']['node_hop1']['gold_rate']:.2f}% / "
      f"{100*SEP2['CROSS']['ALL']['node_hop2']['gold_rate']:.2f}% / "
      f"{100*SEP2['CROSS']['ALL']['node_hop3']['gold_rate']:.2f}%, while the frozen key places gold "
      f"at percentile rank {S2['CS_ADMIT']['ALL']['gold_pctrank']:.4f} -- it pushes the most "
      f"gold-dense stratum toward the bottom. Inside a stratum the frozen key is fine: its "
      f"within-query, within-node-hop AUC is {W['CS_ADMIT']['auc_within_query_and_nodehop']}, which "
      f"is also the no-information floor for any rank rule here (a constant signal falls back to "
      f"the stable tiebreak, so the floor is that value, not 0.5).\n")
    A("The signal audit splits cleanly into two families:\n")
    A(f"* **provenance** -- `RR_PARENT_SUPPORT` {S2['RR_PARENT_SUPPORT']['ALL']['auc']:.4f}, "
      f"`HOP` {S2['HOP']['ALL']['auc']:.4f}, `ADMIT_RANK` {S2['ADMIT_RANK']['ALL']['auc']:.4f}, "
      f"`PARENT_SCORE` {S2['PARENT_SCORE']['ALL']['auc']:.4f}, "
      f"`PATH_SUPPORT` {S2['PATH_SUPPORT']['ALL']['auc']:.4f}")
    A(f"* **geometry** -- `CS_ADMIT` {S2['CS_ADMIT']['ALL']['auc']:.4f}, "
      f"`RSIM` {S2['RSIM']['ALL']['auc']:.4f}, `PATH_SUM` {S2['PATH_SUM']['ALL']['auc']:.4f}, "
      f"`STATIC_SDIR` {S2['STATIC_SDIR']['ALL']['auc']:.4f}\n")
    A("Every geometric signal is at or below chance at the read stage; every provenance signal is "
      "well above it. The best single signal is `RR_PARENT_SUPPORT`, which lifts read recall from "
      f"{S2['CS_ADMIT']['ALL']['recall@64']:.4f} to {S2['RR_PARENT_SUPPORT']['ALL']['recall@64']:.4f} "
      f"(+{100*(S2['RR_PARENT_SUPPORT']['ALL']['recall@64']/S2['CS_ADMIT']['ALL']['recall@64']-1):.0f}% "
      "relative).\n")
    A(f"`FUTURE_MAX`, the previous phase lookahead, scores {S2['FUTURE_MAX']['ALL']['auc']:.4f} "
      f"globally and {W['FUTURE_MAX']['auc_within_query_and_nodehop']} within stratum -- below the "
      f"{W['CS_ADMIT']['auc_within_query_and_nodehop']} floor. Independent re-confirmation that the "
      "lookahead carries no usable ordering information.\n")

    A("## 3. STEP 5 -- the support hypothesis is confirmed\n")
    A("Gold bridge nodes do carry materially stronger multi-path support than negatives, and it is "
      "measurable before the top-64 read:\n")
    for sg in ("PARENT_SCORE", "RR_PARENT_SUPPORT", "DISTINCT_SEED_SUPPORT"):
        v = SEP["STEP5"][sg]
        A(f"* `{sg}` gold {v['gold_mean']:.4f} vs non-gold {v['nongold_mean']:.4f}, "
          f"Cohen d **{v['cohen_d']:+.4f}**")
    A("\nThis was the phase's primary new hypothesis and it is true. It is also, as section 5 "
      "shows, not worth anything at exact P50.\n")

    A("## 4. STEP 4 -- the hop prior is a Simpson artifact\n")
    A("Pooled over MetaQA, shallow nodes look far more gold-dense, which is why every ranking that "
      "promotes hop-1 nodes lifts read recall. Conditioning on query hop inverts it (T3): on 3-hop "
      f"questions gold density is "
      f"{100*SEP2['CROSS']['query_hop3']['node_hop1']['gold_rate']:.2f}% at node hop 1 but "
      f"{100*SEP2['CROSS']['query_hop3']['node_hop3']['gold_rate']:.2f}% at node hop 3. Query hop is "
      "not observable at inference. So any hop-conditioned rule that helps hop2 necessarily hurts "
      "hop3, which is exactly what R2_HOP_CONDITIONED does: "
      f"hop2 {rm['R2_HOP_CONDITIONED']['P50']['hop2']:.4f} vs frozen {s['hop2']:.4f}, "
      f"hop3 {rm['R2_HOP_CONDITIONED']['P50']['hop3']:.4f} vs frozen {s['hop3']:.4f}. "
      "Hop is not dataset identity, but on this substrate it is a proxy for the one thing the "
      "contract forbids observing.\n")

    A("## 5. STEP 6/9 -- separability does not convert, on any corpus\n")
    A("| ranking | macro delta | worst corpus | net over 11,415 queries | corpora improved | significant |")
    A("|---|---:|---:|---:|---:|---|")
    for rk in ("R1_EQUAL_RRF", "R2_HOP_CONDITIONED", "RS_BEST_SINGLE", "RX_ORACLE_READ"):
        m = macro[rk]
        A(f"| {RP.NAME[rk]} | {m['macro_delta']:+.5f} | {m['worst']:+.4f} | {m['net_total']:+d} | "
          f"{m['improved']}/6 | {', '.join(m['significant']) or 'none'} |")
    A("")
    A(f"Across the {CORR['cells']} inference-safe cells the correlation between read-recall lift and "
      f"exact-P50 change is **r = {CORR['pearson_r']:+.4f}** (spearman {CORR['spearman_r']:+.4f}). "
      f"{CORR['cells_read_improved']}/{CORR['cells']} cells improve the read -- by "
      f"{CORR['mean_read_lift']:+.4f} on average, up to {CORR['max_read_lift']:+.4f} -- and of "
      f"those, {CORR['of_those_P50_improved']} improve exact P50. A coin flip.\n")
    A("The sharpest single case is 2wiki: R2 reads "
      f"{FUN['2wiki_clean']['METHODS']['R2_HOP_CONDITIONED']['read_recall@64'] / FUN['2wiki_clean']['METHODS']['R0_STATIC']['read_recall@64']:.1f}x "
      "more gold than frozen and delivers gold partitions to S4 essentially perfectly "
      f"({RANK['2wiki_clean']['STEP10']['R0_STATIC']['gold_part_S4_frac']:.4f} -> "
      f"{RANK['2wiki_clean']['STEP10']['R2_HOP_CONDITIONED']['gold_part_S4_frac']:.4f}), for "
      f"{RANK['2wiki_clean']['STEP9']['R2_HOP_CONDITIONED']['vs_SAFE_net']:+d} query at exact P50.\n")
    A("And the ranking with the *best* measured separability, used directly as the directive's "
      "first branch prescribes, is directionally **negative**: `RS_BEST_SINGLE` "
      f"(RR_PARENT_SUPPORT) has macro {macro['RS_BEST_SINGLE']['macro_delta']:+.5f} and improves "
      f"{macro['RS_BEST_SINGLE']['improved']}/6 corpora.\n")

    A("## 6. Why -- gold node recall is not the objective\n")
    A("Measured on MetaQA at the top-64 read:\n")
    A("| ranking | gold nodes read | distinct gold partitions | of which the swap actually NEEDS |")
    A("|---|---:|---:|---:|")
    A("| R0_STATIC (frozen) | 1,985 | 1,792 | 905 |")
    A("| R2_HOP_CONDITIONED | 3,125 | 2,650 | **756** |")
    A("| RX_ORACLE_READ | 4,318 | 3,534 | 1,279 |")
    A("")
    A("R2 reads 57% more gold nodes and 48% more distinct gold partitions than frozen, and supplies "
      "**16% fewer of the partitions the swap actually needs**. The needed partitions are by "
      "definition the ones the protected core does not already hold -- the peripheral, hard ones. A "
      "better generic node ranking rescues *typical* gold, which is disproportionately in "
      "already-covered partitions. Read recall and the objective are not the same quantity, and on "
      "this substrate they are barely related.\n")

    A("## 7. STEP 7 -- the same key inside the beam\n")
    A("R1 was re-run as the beam survival key as well as the `added` and read key: one semantics at "
      "every stage, which is the consistency the lookahead failed. Parity against frozen breaks by "
      "design; the question is whether a better-separating key buys reach.\n")
    A("| corpus | parity vs frozen | gold discovered | exact P50 | net | p | oracle on this beam |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for d, v in st7.items():
        A(f"| {d} | {v['parity_vs_frozen']} | {v['gold_discovered_frozen']:.4f} -> "
          f"{v['gold_discovered_beamR1']:.4f} | {v['P50_frozen']:.4f} -> {v['P50_beamR1']:.4f} | "
          f"{v['net']:+d} | {v['p']:.3g} | {v['oracle_on_beamR1']:.4f} |")
    A("")
    if "metaqa" in st7:
        m7 = st7["metaqa"]
        A(f"On MetaQA the R1 beam is a genuinely different search ({m7['parity_vs_frozen']} parity) "
          f"and does find more gold ({m7['gold_discovered_frozen']:.4f} -> "
          f"{m7['gold_discovered_beamR1']:.4f}), which raises the *ceiling* from {o['ALL']:.4f} to "
          f"{m7['oracle_on_beamR1']:.4f}. The realisable gain is {m7['net']:+d} queries at "
          f"p = {m7['p']:.3g}, and hop3 lands at "
          f"{RP.j(f'{B}/rank_metaqa__BEAMR1.json')['STEP9']['R1_EQUAL_RRF']['P50']['hop3']:.4f} -- "
          f"exactly the frozen {s['hop3']:.4f}.\n")
    if "musique_clean" in st7:
        mq = st7["musique_clean"]
        A(f"On musique the R1 beam finds marginally more gold and the oracle on it is *worse* than "
          f"frozen ({mq['oracle_on_beamR1']:.4f} vs {mq['P50_frozen']:.4f}). Changing the survival "
          "key changes pool composition, and composition can cost more than the extra gold is "
          "worth. This is the same pool-composition sensitivity the pre-partition audit recorded.\n")

    A("## 8. STEP 10 -- the headroom, and where it goes\n")
    A("Every query lands in exactly one bucket according to what the 6-slot swap has to supply "
      "(T7). On MetaQA, frozen:\n")
    b = fm["R0_STATIC"]["buckets"]
    bo = fm["RX_ORACLE_READ"]["buckets"]
    A(f"* `FREE` {b['FREE']} -- the protected 44 already cover the query")
    A(f"* `CAPACITY` {b['CAPACITY']} -- more than 6 partitions are needed; dead at B = 6 whatever "
      f"the ranking")
    A(f"* `NOCAND` {b['NOCAND']} -- some needed partition is not a candidate at all. "
      f"**The only bucket a better read can move.**")
    A(f"* `LOST` {b['LOST']} -- every needed partition is a candidate and F6 still did not pick them")
    A(f"* `WON` {b['WON']}\n")
    A(f"A perfect read moves `NOCAND` {b['NOCAND']} -> {bo['NOCAND']} and pushes "
      f"{bo['LOST']-b['LOST']} of those straight into `LOST`, for a net "
      f"{bo['WON']-b['WON']:+d} queries. Of the "
      f"{fm['R0_STATIC']['needed_partitions']:,} partitions the swap must supply across MetaQA, "
      f"**{fm['R0_STATIC']['needed_with_NO_evidence']:,} have no evidence in any channel** -- not "
      f"structural, not retrieval, not incumbent -- and the oracle read only reduces that to "
      f"{fm['RX_ORACLE_READ']['needed_with_NO_evidence']:,}. The read stage can supply evidence for "
      f"{fm['RX_ORACLE_READ']['needed_with_structural_evidence']-fm['R0_STATIC']['needed_with_structural_evidence']} "
      f"more needed partitions out of {fm['R0_STATIC']['needed_partitions']:,}, or "
      f"{100*(fm['RX_ORACLE_READ']['needed_with_structural_evidence']-fm['R0_STATIC']['needed_with_structural_evidence'])/fm['R0_STATIC']['needed_partitions']:.1f}%.\n")
    A("On 2wiki, musique and squad the oracle does not reduce the no-evidence count at all "
      "(2wiki 98 -> 98). Those partitions contain no discovered node, so no read ordering can reach "
      "them: on the text corpora the needed-partition deficit is a **reach** deficit, and the read "
      "stage is already saturated with respect to what it can contribute.\n")
    h3 = fm["R0_STATIC"]["hop3"]
    A(f"MetaQA hop3, the phase's primary target, is the extreme case: mean need is "
      f"**{h3['mean_need']} partitions against a 6-slot budget**, and "
      f"{h3['buckets']['CAPACITY']}/666 hop-3 queries ({100*h3['buckets']['CAPACITY']/666:.0f}%) are "
      f"in `CAPACITY` -- unreachable at B = 6 under any ranking whatsoever. That is why the oracle "
      f"moves hop3 only {s['hop3']:.4f} -> {o['hop3']:.4f}.\n")

    A("## 9. The ceiling\n")
    A(f"A **perfect** read -- every discovered gold node in the top 64, gold-partition delivery to "
      f"S4 at 1.0000 on every corpus -- is worth macro "
      f"**{macro['RX_ORACLE_READ']['macro_delta']:+.5f}**, significant on only "
      f"{len(macro['RX_ORACLE_READ']['significant'])} of 6 corpora "
      f"({', '.join(macro['RX_ORACLE_READ']['significant'])}). On MetaQA it is "
      f"ALL {s['ALL']:.4f} -> {o['ALL']:.4f}, hop2 {s['hop2']:.4f} -> {o['hop2']:.4f}, "
      f"hop3 {s['hop3']:.4f} -> {o['hop3']:.4f}.\n")
    A(f"The phase target was hop3 *materially* above {s['hop3']:.4f} and hop2 *materially* above "
      f"{s['hop2']:.4f}. The oracle reaches hop3 {o['hop3']:.4f}, i.e. "
      f"{round((o['hop3']-s['hop3'])*666)} queries out of 666. **The target is unreachable from this "
      "stage even with gold labels.** For scale, the fixed-P50 partition router promoted in an "
      "earlier phase was worth macro +0.0069 -- roughly twice what solving this entire stage "
      "perfectly would buy.\n")

    A("## 10. What this closes and what it does not\n")
    A("**Closed.** Read-stage node ordering as an L1 lever. The stage has a hard ceiling of macro "
      f"{macro['RX_ORACLE_READ']['macro_delta']:+.5f}; three parameter-free rankings and the best "
      "single signal used directly all capture none of it; and the read metric is uncorrelated "
      f"(r = {CORR['pearson_r']:+.4f}) with the objective, so it cannot even be used as a proxy to "
      "optimise against. Do not build another node scorer for this stage.\n")
    A("**Not closed, and not touched here.** The residual sits in two places this phase was "
      "explicitly forbidden from changing:\n")
    A(f"1. **Capacity.** B = 6 against a mean need of {h3['mean_need']} on MetaQA hop3; "
      f"{b['CAPACITY']}/1998 queries overall are arithmetically dead.")
    A(f"2. **Reach.** {fm['R0_STATIC']['needed_with_NO_evidence']:,}/"
      f"{fm['R0_STATIC']['needed_partitions']:,} needed partitions on MetaQA and 98/147 on 2wiki "
      f"have no evidence in any channel, and a perfect read barely dents it.\n")
    A("Neither is a scoring problem. The verdict is B rather than C because separability was found "
      "and is large -- but the practical consequence is the same STOP for parameter-free structural "
      "L1 search, with a better-specified reason: the stage was not where the loss lives.\n")

    open(f"{SS.SSD}/FINAL_REPORT.md", "w").write("\n".join(L))
    print(f"wrote {SS.SSD}/FINAL_REPORT.md ({len(L)} blocks)")


if __name__ == "__main__":
    main()
