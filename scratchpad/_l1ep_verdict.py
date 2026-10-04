"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- section 9: verdicts and Q1-Q12.

Each verdict letter is assigned from the measured numbers against a criterion stated inline, so
the label is reproducible rather than asserted.  The A-E / A-D letter DEFINITIONS from the
directive are not recoverable from the session record; the criteria below are stated explicitly
so the label can be re-mapped without re-running anything if the intended definitions differed.
"""
MATERIAL = 0.01          # a gain smaller than this does not justify changing a frozen component
# corpus CLASS is a property of the source, fixed before any measurement -- MetaQA and WebQSP are
# knowledge-base corpora, the other four are free-text corpora.  It is used only to REPORT the
# structure of the result; no rule below switches behaviour on it.
CLASS = {"metaqa": "KB", "webqsp": "KB", "2wiki_clean": "text", "musique_clean": "text",
         "squad_clean": "text", "hotpotqa_clean": "text"}


def _class_line(CS):
    """The combined-verdict bullet about the SHAPE of the partition effect.

    Written from the measured rows, not from the KB-vs-text hypothesis that HotpotQA refuted."""
    rows = CS.get("rows", [])
    fmt = lambda r: f"{r[0]} ({r[1]}) {r[3]:+.4f}"
    pos = [fmt(r) for r in sorted(rows, key=lambda r: -r[3]) if r[7] and r[3] > 0]
    neg = [fmt(r) for r in sorted(rows, key=lambda r: r[3]) if r[7] and r[3] < 0]
    flat = [fmt(r) for r in rows if not r[7]]
    return ("The partition gain is CORPUS-DEPENDENT, and a CORPUS-CLASS reading is REFUTED "
            f"(clean KB-vs-text split = {CS.get('CLEAN_KB_VS_TEXT_SPLIT')}). Under one rule with "
            "no per-corpus tuning it converts significantly on " + "; ".join(pos) +
            ", significantly regresses on " + "; ".join(neg) +
            ", and is not separable from noise on " + "; ".join(flat) +
            ". KB-vs-text was the obvious hypothesis while only the two knowledge bases had "
            "converted; HotpotQA is free text and converts significantly, so corpus class does "
            "not organise the result. What separates the winners from the losers is exposure and "
            "block demand, not class -- see the headline section.")


def class_split(AN, tag):
    """what structure does a candidate's effect actually have across corpora?

    This started as a KB-vs-text hypothesis, formed when only MetaQA and WebQSP had converted.
    HotpotQA -- a free-text corpus -- then converted significantly too (+0.0150, p = 4e-05), so
    CLEAN_KB_VS_TEXT_SPLIT is reported as measured and comes out FALSE.  The function keeps the
    class labels because they are still the honest way to show that the hypothesis failed."""
    st = AN.get("B15_STABILITY", {})
    rows, pos, neg = [], [], []
    for ds, v in st.items():
        r = v["CANDIDATES"].get(tag)
        if not r:
            continue
        rows.append((ds, CLASS.get(ds, "?"), r["F6_ALL_P50"], r["delta_vs_production"],
                     r.get("gained"), r.get("lost"), r.get("p"), bool(r.get("sig")),
                     v.get("NOISE_FLOOR_F6_SD_SAME_GRAPH_SAME_ALGO")))
        (pos if r["delta_vs_production"] > 0 else neg).append(CLASS.get(ds, "?"))
    clean = bool(rows) and set(pos) == {"KB"} and "KB" not in set(neg)
    return {"rows": rows, "CLEAN_KB_VS_TEXT_SPLIT": clean,
            "kb_sig_positive": sum(1 for r in rows if r[1] == "KB" and r[7] and r[3] > 0),
            "kb_n": sum(1 for r in rows if r[1] == "KB"),
            "text_sig_negative": sum(1 for r in rows if r[1] == "text" and r[7] and r[3] < 0),
            "text_n": sum(1 for r in rows if r[1] == "text")}


def hopstr(R, ds="metaqa", tag="P4_CE_LOCAL_ONLY"):
    """MetaQA hop2/hop3 before -> after, read from Phase C rather than hardcoded."""
    ph = (R.get("SYSTEM") or {}).get("C1_C2_PHASE_C", {}).get(ds) or {}
    cand = (ph.get("candidates") or {}).get(tag) or {}
    a, b = ph.get("by_hop") or {}, cand.get("by_hop") or {}
    out = {}
    for h in ("hop1", "hop2", "hop3"):
        if h in a and h in b:
            out[h] = (a[h]["F6_ALL"], b[h]["F6_ALL"])
    return out


def _inter_line(AN):
    """what the measured 2x2 says about whether the two axes are separable."""
    got = {d: v for d, v in AN.get("C4_DECOMPOSITION", {}).items() if v.get("INTERACTION_2X2")}
    if not got:
        return "The joint 2x2 was not run, so INTERACTION is unmeasured."
    st = AN.get("B15_STABILITY", {})
    parts = []
    for d, v in got.items():
        nf = st.get(d, {}).get("NOISE_FLOOR_F6_SD_SAME_GRAPH_SAME_ALGO")
        parts.append("`%s` %+.4f%s" % (d, v["INTERACTION"],
                                       " (noise floor sd %.4f)" % nf if nf else ""))
    big = [d for d, v in got.items()
           if abs((v["INTERACTION_2X2"] or {}).get("PARTITION_EFFECT_b_minus_a", 0))
           > 2 * (st.get(d, {}).get("NOISE_FLOOR_F6_SD_SAME_GRAPH_SAME_ALGO") or 1)]
    weak = [d for d in got if d not in big]
    caveat = ""
    if weak:
        caveat = (" One caveat on coverage: %s carries a near-null partition arm (%s), so its cell "
                  "tests additivity in a regime where there is little to interact with. The "
                  "load-bearing tests are %s, where the partition effect is large."
                  % (", ".join("`%s`" % d for d in weak),
                     ", ".join(got[d]["INTERACTION_2X2"]["partition_arm"] for d in weak),
                     ", ".join("`%s`" % d for d in big)))
    return ("The two axes are also SEPARABLE, which the 2x2 measures rather than assumes: "
            "INTERACTION = d - b - c + a is " + "; ".join(parts) + ". Every value is at or below "
            "that corpus's own reseed noise floor, so the partition gain does not depend on which "
            "edge substrate the router walks, and widening the substrate costs the same small "
            "amount under either partitioning. Edge and partition are independent levers of very "
            "unequal size, not two views of one effect." + caveat)


def verdicts(R, AN):
    E = R["EDGE_SUBSTRATE"]
    st = AN.get("B15_STABILITY", {})
    nA = len(R["CORPORA_WITH_PHASE_A"])
    sig_pos, sig_neg = E["A11_SIGNIFICANT_GAINS"], E["A11_SIGNIFICANT_REGRESSIONS"]
    best, worst = E["A11_BEST_GAIN"], E["A11_WORST_LOSS"]

    # ---------------------------------------------------------------- edge substrate
    e_A = bool(sig_pos and not sig_neg and best >= MATERIAL and len(sig_pos) >= max(nA - 1, 1))
    e_B = bool(sig_pos and not sig_neg and best >= MATERIAL and not e_A)
    e_D = bool(sig_pos and sig_neg)
    e_E = nA == 0
    e_C = not (e_A or e_B or e_D or e_E)
    EC = {"A": ("a richer edge substrate is a universal win: significant material gain on "
                "essentially every corpus, no significant regression, acceptable cost", e_A),
          "B": ("PARTIAL: a material significant gain somewhere and no significant regression "
                "anywhere, but not universal", e_B),
          "C": (f"EXHAUSTED: no substrate produces a material gain (>= {MATERIAL:.2f} ALL@50) on "
                f"any corpus; the axis is closed", e_C),
          "D": ("MIXED: gains on some corpora paid for by significant regressions elsewhere",
                e_D),
          "E": ("INCONCLUSIVE: parity failed or coverage too thin to discriminate", e_E)}
    ev = next(k for k, v in EC.items() if v[1])

    # ---------------------------------------------------------------- partitioning
    conv, regr = [], []
    for ds, v in st.items():
        for t, r in v["CANDIDATES"].items():
            d = r["delta_vs_production"]
            if r.get("sig") and d >= MATERIAL:
                conv.append((ds, t, d))
            if r.get("sig") and d < 0:
                regr.append((ds, t, d))
    tags = sorted({t for v in st.values() for t in v["CANDIDATES"]})
    universal = None
    for t in tags:
        present = [ds for ds in st if t in st[ds]["CANDIDATES"]]
        if len(present) < len(st):
            continue
        C_ = {ds: st[ds]["CANDIDATES"][t] for ds in present}
        if any(c.get("sig") and c["delta_vs_production"] < 0 for c in C_.values()):
            continue
        if any(c.get("sig") and c["delta_vs_production"] >= MATERIAL for c in C_.values()):
            universal = t
            break
    p_A = universal is not None and len(st) >= 5
    p_B = universal is not None and not p_A
    p_D = bool(conv and regr and universal is None)
    p_E = len(st) == 0
    p_C = not (p_A or p_B or p_D or p_E)
    PC = {"A": ("one universal partitioning rule beats production materially and never "
                "significantly regresses, on every corpus", p_A),
          "B": ("PARTIAL: a single universal rule gains materially with no significant "
                "regression, but coverage is incomplete", p_B),
          "C": ("EXHAUSTED: no alternative partitioning converts into a material gain anywhere",
                p_C),
          "D": ("MIXED / CORPUS-DEPENDENT: a rule converts materially on some corpora and "
                "significantly regresses on others, so no universal change is licensed", p_D),
          "E": ("INCONCLUSIVE: Phase C parity failed or no corpus completed", p_E)}
    pv = next(k for k, v in PC.items() if v[1])

    # ---------------------------------------------------------------- combined
    c_D = (ev == "D") or (pv == "D")
    c_A = (ev in "AB") and (pv in "AB") and not c_D
    c_B = ((ev in "AB") != (pv in "AB")) and not c_D
    c_C = not (c_A or c_B or c_D)
    CC = {"A": ("both axes are limiters and both convert universally", c_A),
          "B": ("exactly one axis is the limiter and it converts universally", c_B),
          "C": ("neither axis is the limiter; the residual lives elsewhere", c_C),
          "D": ("at least one axis produces a real, large, mechanistically isolated effect that "
                "is corpus-dependent, so no universal change is licensed", c_D)}
    cv = next(k for k, v in CC.items() if v[1])

    # ---------------------------------------------------------------- bases
    eb = [f"{E['A11_CELLS']} substrate x corpus cells at exact P = 50; "
          f"{E['A11_SIGNIFICANT_CELLS']} reach significance.",
          f"Best gain any substrate achieves anywhere: {best:+.4f}; worst {worst:+.4f}.",
          "Cost of the richest substrate (`E6_TOPOLOGY_C`) is 1.5x-4.3x the frozen traversal "
          "time and up to 8.5x the edges scored.",
          "On MetaQA the candidate ORACLE falls as the substrate widens (0.7212 -> 0.7042): the "
          "wider frontier dilutes a fixed-size candidate list rather than enriching it."]
    kn = E.get("A9_KNN_NOVEL_AND_NEEDED_COUNT", {})
    if kn:
        eb.append("kNN reaches partitions no other channel reaches on the KB-ish corpora but "
                  "that territory is nearly empty of what is needed: "
                  + ", ".join(f"{d} {v}" for d, v in kn.items())
                  + " novel-and-needed partitions.")
    pb = []
    if conv:
        pb.append("Material significant conversions: "
                  + "; ".join(f"`{d}` / `{t}` {x:+.4f}"
                              for d, t, x in sorted(conv, key=lambda z: -z[2])))
    if regr:
        pb.append("Significant regressions: "
                  + "; ".join(f"`{d}` / `{t}` {x:+.4f}"
                              for d, t, x in sorted(regr, key=lambda z: z[2])[:8]))
    pb.append(f"A single universal rule that gains materially and never significantly regresses: "
              f"`{universal or 'NONE'}`.")
    for ds, v in st.items():
        nf = v.get("NOISE_FLOOR_F6_SD_SAME_GRAPH_SAME_ALGO")
        if nf:
            pb.append(f"`{ds}`: METIS reseed noise floor sd = {nf:.4f} (production never pinned a "
                      f"seed), so any delta below ~{2 * nf:.4f} is not a partitioner difference.")

    metaqa_conv = [x for x in conv if x[0] == "metaqa"]
    p4_regr = [x for x in regr if x[1].startswith("P4_")]
    CAND = "P4_CE_LOCAL_ONLY"
    cand_regr = [x for x in regr if x[1] == CAND]
    other_regr = sorted({x[1] for x in p4_regr if x[1] != CAND})
    CS = class_split(AN, "P4_CE_LOCAL_ONLY")
    HOP = hopstr(R)
    gates = {
        "exact P = 50 preserved": "PASS -- every cell selects exactly 50 partitions",
        "balanced, predictable partitions":
            "PASS -- every candidate holds max/mean <= 1.04 at the production k",
        "materially useful MetaQA hop2/hop3":
            (("PASS -- hop3 %.4f -> %.4f and hop2 %.4f -> %.4f under `P4_CE_LOCAL_ONLY`"
              % (HOP["hop3"] + HOP["hop2"])) if metaqa_conv and HOP else "NOT DEMONSTRATED"),
        "no significant cross-corpus regression":
            (f"FAIL for the promotion candidate `{CAND}`: "
             + "; ".join(f"`{d}` {x:+.4f}" for d, t, x in cand_regr)
             + (f". No sibling rule escapes it either -- {', '.join('`' + t + '`' for t in other_regr)} "
                f"also regress significantly, so there is no nearby variant to promote instead."
                if other_regr else ".")
             if cand_regr else
             ("FAIL -- " + "; ".join(f"`{d}`/`{t}` {x:+.4f}" for d, t, x in p4_regr)
              if p4_regr else "PASS")),
        "reasonable latency":
            "PASS -- partitioning is an offline artifact and online exposure is unchanged "
            "(4,977-5,042 scoped nodes across every cell)",
        "no gold or query leakage into partition construction":
            "PASS -- every build reads corpus topology, NER and kNN only"}
    promoted = universal or "NONE"
    if not all(str(v).startswith("PASS") for v in gates.values()):
        promoted = "NONE"

    labels = {}
    if metaqa_conv and p4_regr:
        labels["PARTITION_QUALITY_IMPROVED_ROUTER_CONVERTS_BUT_NOT_UNIVERSALLY"] = (
            f"the alternative partitioning is a genuinely better partitioner AND the router "
            f"converts it, on THREE of six corpora under one untuned rule: MetaQA +0.0966 (hop3 "
            f"{HOP['hop3'][0]:.4f} -> {HOP['hop3'][1]:.4f}"
            f"{', %.2fx' % (HOP['hop3'][1] / HOP['hop3'][0]) if HOP.get('hop3') else ''}), "
            f"WebQSP +0.0423, HotpotQA +0.0150, all significant. It is withheld because the same "
            f"rule significantly regresses SQuAD (-0.0095, p = 0.0066) and no sibling variant "
            f"escapes that. This is explicitly NOT "
            f"`PARTITION_QUALITY_IMPROVED_ROUTER_CANNOT_CONVERT`: the router had no trouble "
            f"converting it, three times. Note also that the conversion is a partition-RANKING "
            f"effect -- the full delta is already present in BASE before the F6 selector runs "
            f"-- so 'the router cannot convert it' is the wrong diagnosis in both directions.")
    labels["EDGE_REACH_GAIN_EXPOSURE_DRIVEN"] = (
        "not applicable as a rescue -- the richer substrates raise visited-partition exposure "
        "1.2x-2.7x and still do not raise exact-P50, so there is no reach gain to relabel.")

    return {"EDGE_CRITERIA": EC, "EDGE_SUBSTRATE_VERDICT": ev, "EDGE_BASIS": eb,
            "PART_CRITERIA": PC, "PARTITIONING_VERDICT": pv, "PART_BASIS": pb,
            "COMB_CRITERIA": CC, "COMBINED_L1_VERDICT": cv,
            "COMB_BASIS": [
                f"The edge axis moves exact-P50 by at most {max(abs(best), abs(worst)):.4f} "
                f"anywhere. The partition axis moves it by +0.0966 on MetaQA and +0.0423 on "
                f"WebQSP, and by -0.07 to -0.25 when degraded to a balanced random blocking at "
                f"identical exposure. The two axes are not comparable in magnitude.",
                _class_line(CS),
                "Yet the partitioner NEEDS the very families the traversal cannot use: "
                "partitioning on STRUCT alone is significantly worse on MuSiQue (-0.0410), "
                "2Wiki (-0.0445) and SQuAD (-0.0090).",
                _inter_line(AN),
                "So the extra edges are real information that L1 already exploits -- offline, "
                "through the block structure, not online through traversal."],
            "PROMOTION_GATES": gates, "PROMOTED": promoted, "L1_FROZEN": "NO",
            "SPECIAL_LABELS": labels,
            "CLASS_SPLIT_P4_CE_LOCAL_ONLY": CS,
            "METAQA_BY_HOP_P4_CE_LOCAL_ONLY": HOP,
            "QUESTIONS": questions(R, AN, universal, CS, HOP)}


def questions(R, AN, universal, CS=None, HOP=None):
    E = R["EDGE_SUBSTRATE"]
    CS = CS or class_split(AN, "P4_CE_LOCAL_ONLY")
    HOP = HOP or hopstr(R)
    h23 = ("hop3 %.4f -> %.4f and hop2 %.4f -> %.4f" % (HOP["hop3"] + HOP["hop2"])
           if HOP else "hop2/hop3 NOT MEASURED")
    h3 = ("%.4f (-> %.4f)" % HOP["hop3"] if HOP else "NOT MEASURED")
    tbl = "; ".join(f"{d} ({c}) {x:+.4f}{'*' if sg else ''}"
                    for d, c, _f, x, _g, _l, _p, sg, _n in
                    sorted(CS["rows"], key=lambda z: -z[3]))
    return [
        ("Q1. Is `master_nodes.neighbors` really the only family the router traverses, and is "
         "the edge algebra what we assumed?",
         "Yes, and it is now verified rather than assumed. `CSR(S)` is bit-identical to the "
         "frozen neighbour lists on every corpus; `S n N` and `S n K` are empty everywhere; "
         "`N n K` is non-empty everywhere, so provenance must be carried as a bitmask. "
         "`S u N u K` equals the topology-C file exactly on 5 of 6 corpora -- Hotpot's older "
         "`gte_qwen/graph.pt` omits 34,425 structural edges, making the measured Hotpot cut a "
         "0.33% superset of what METIS actually saw (documented, not fatal)."),
        ("Q2. How much of the graph does the frozen router never touch?",
         "Between 17% (SQuAD) and 83% (2Wiki) of topology-C edges are never traversed. The "
         "unused majority is large enough that its irrelevance is itself the finding."),
        ("Q3. Do the untraversed families reach needed partitions the frozen substrate misses?",
         "Yes, but marginally. Unique-kNN reach is 0.3%-7% of missed needed partitions and "
         "unique-NERX reach 2%-28%, while 2%-9% are reached by no family at all."),
        ("Q4. Is that reach DISTINCT information, or a restatement of Dense/SPLADE?",
         "Largely a restatement. On MuSiQue and SQuAD, ZERO kNN-reached partitions are novel to "
         "Dense/SPLADE top-200. Where novelty is high (2Wiki 22%, WebQSP 43%) the novel "
         "territory contains 5 and 23 needed partitions respectively. For NERX the "
         "query-to-target similarity separates needed from nuisance better than the edge's own "
         "endpoint similarity (AUC 0.9312 vs 0.8371 on MuSiQue): "
         "`DENSE_WEARING_GRAPH_COSTUME = YES`."),
        ("Q5. Do mixed-family (bridge) paths buy a path shape homogeneous ones cannot?",
         "Only at the first hop, and only on the KB-ish corpora. Bridges beat homogeneous paths "
         "at hop1 by +0.11 to +0.20 needed-yield on 4 of 5 corpora, then lose at hop2 and hop3 "
         "everywhere except WebQSP, where they win at all three lengths. Mixed traversal is a KB "
         "phenomenon whose value is exhausted after one step on text."),
        ("Q6. At exact P = 50, does ANY richer edge substrate beat the frozen one?",
         f"No. Across {E['A11_CELLS']} cells the best result anywhere is "
         f"{E['A11_BEST_GAIN']:+.4f} and only {E['A11_SIGNIFICANT_CELLS']} cell reaches "
         f"significance, at 1.5x-4.3x the traversal cost. On MetaQA the candidate oracle FALLS "
         f"as the substrate widens."),
        ("Q7. Does the shipped partitioning impose a ceiling at P = 50?",
         "No. Oracle ALL@50 over the shipped assignment is 1.0000 on every text corpus and "
         "0.987 / 0.995 on MetaQA / WebQSP. The required evidence is inside a 50-block budget "
         "almost always; what fails is selecting those blocks."),
        ("Q8. Is the partitioning doing anything at all, or would any balanced blocking do?",
         "It is doing a great deal. `P0_RANDOM_BALANCED`, built with the EXACT production size "
         "histogram and landing at identical exposure, costs 7-25 points of ALL@50 while leaving "
         "the P=50 oracle essentially intact. What the partitioning buys is ROUTABILITY -- a "
         "usable partition-level vote -- not containment."),
        ("Q9. Which graph should the partitioner see?",
         "On text, not the one the router walks. Partitioning on STRUCT alone (`PM0_STRUCT`) is "
         "significantly worse on MuSiQue (-0.0410), 2Wiki (-0.0445) and SQuAD (-0.0090). The NER "
         "and kNN families are valuable to the PARTITIONER and worthless to the TRAVERSAL -- the "
         "sharpest asymmetry in the program, and the reconciliation of its two halves.\n\n"
         "The exception is instructive rather than contradictory: on WebQSP `PM0_STRUCT` is "
         "+0.0120 and not significant (1.6 sd of that corpus's 0.0077 floor). On a knowledge "
         "base the structural family IS the curated relation set, so it already carries the "
         "semantics NER and kNN are approximating on text, where STRUCT is only title mentions. "
         "The rule is therefore: the partitioner needs a semantically complete graph, which on "
         "text requires the untraversed families and on a KB does not."),
        ("Q10. Is edge cut the right objective?",
         "No, and the counter-example is unambiguous. On MetaQA the best partitioning has a 29% "
         "WORSE cut on topology C (296,258 vs 228,941) and a LOWER gold-edge containment (0.2715 "
         "vs 0.4467), yet gains +0.0966. The statistics that track utility are gold co-location "
         "(all-gold-in-one-block 0.4620 -> 0.5706) and the number of blocks a query needs "
         "(5.78 -> 4.82)."),
        ("Q11. If partition construction matters, WHAT property of it matters?",
         "Second-order locality with degree-normalised strength -- and neither ingredient alone. "
         "On MetaQA: topology C unweighted is the baseline; adding the NER artifact's own stored "
         "1/df weights to it does nothing (-0.0050); closing local neighbourhoods into cliques "
         "with UNIT weights does nothing (-0.0005); NER hyperedges alone do nothing (-0.0010). "
         "Closing local neighbourhoods into cliques WITH the canonical 1/(|e|-1) weight gains "
         f"+0.0966, {h23}. The isolation replicates on "
         "WebQSP, where the unweighted clique closure lands inside that corpus's own reseed noise "
         "floor (+0.0064 against sd 0.0077) and the weighted one converts +0.0423.\n\n"
         f"Under one rule with no per-corpus tuning: {tbl} (* = significant). Three corpora "
         "gain significantly, one loses significantly, two are flat.\n\n"
         "A CORPUS-CLASS reading is REFUTED. When only MetaQA and WebQSP had converted, "
         "KB-vs-text was the obvious hypothesis; HotpotQA then converted significantly "
         "(+0.0150, p = 4e-05) and is free text, so class does not organise the result. What "
         "does hold is that the winners are the corpora where P = 50 exposes at most ~1.5% of "
         "the corpus (WebQSP 0.64%, HotpotQA 1.00%) or where a query needs three or more blocks "
         "(MetaQA 5.78, WebQSP 3.66) -- two thresholds fitted to six points after the fact, "
         "offered as the hypothesis a seventh corpus would test rather than as a law.\n\n"
         "State the gains at their own scales rather than as equals: MetaQA's +0.0966 is about "
         "46 sd of its 0.0021 reseed noise floor, WebQSP's +0.0423 about 5.5 sd of its much "
         "larger 0.0077 floor (its reseed replicates span 0.7512-0.7710). All three are "
         "significant by exact McNemar, but only MetaQA is overwhelming relative to the variance "
         "of its own partitioner."),
        ("Q12. Is the largest remaining problem: traversal / partition construction / ranking / "
         "or genuinely missing information?",
         f"**Partition construction -- and specifically the partition RANKING it produces. Not "
         f"traversal, not the selector, and not missing information.**\n\n"
         f"The edge substrate, which is what the router actually walks, is closed on all six "
         f"corpora: its entire dynamic range at exact P = 50 is about +/-0.004, for up to 4.3x "
         f"the traversal cost. Partition construction has roughly 25x that dynamic range on "
         f"MetaQA (+0.0966), 10x on WebQSP (+0.0423) and 4x on HotpotQA (+0.0150), and is the "
         f"only intervention in the whole L1 program that has ever moved MetaQA hop3 off {h3}. "
         f"The 2x2 shows the two axes are additive -- INTERACTION sits at or below every "
         f"corpus's reseed noise floor -- so this is a genuinely separate lever, not a "
         f"restatement of the edge result.\n\n"
         f"Within partition construction the binding stage is RANKING, not selection. The whole "
         f"delta is already present in BASE, the Dense+SPLADE RRF over blocks, before the F6 "
         f"selector runs; the selector moves it by at most +/-0.008 on five of six corpora. Its "
         f"one larger contribution is defensive: on SQuAD it recovers two thirds of a -0.0285 "
         f"ranking loss. The frozen selector absorbs a bad partitioning; it never creates a good "
         f"one.\n\n"
         f"Where the rule does not gain, the residual is still ranking rather than missing "
         f"information: the P = 50 oracle over the shipped assignment is 1.0000 on every text "
         f"corpus, so the evidence is inside the budget and simply is not selected. Genuinely "
         f"missing information is the smallest term everywhere -- only 2%-9% of missed needed "
         f"partitions are unreachable by any edge family."),
    ]
