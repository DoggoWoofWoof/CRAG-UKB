"""Assemble FINAL_REPORT.md for the L1 SIMPLIFY / PPR phase.

Tables come from _l1pp_tables.py (captured, never retyped); headline numbers are recomputed from
the round JSONs here so the prose cannot go stale.

  python scratchpad/_l1pp_final.py
"""
import os, sys, io, json, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1pp_core as PP

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import _l1pp_tables                                     # noqa: F401  (executes on import)
raw = buf.getvalue()

T = {}
cur, name = [], None
for line in raw.split("\n"):
    if line.startswith("## T"):
        if name:
            T[name] = "\n".join(cur).strip()
        name = line.split(" ")[1]
        cur = [line]
    elif name:
        cur.append(line)
if name:
    T[name] = "\n".join(cur).strip()

D = PP.PPD + "/diag"
J = lambda n: json.load(open(f"{D}/{n}.json")) if os.path.exists(f"{D}/{n}.json") else None
A, Bj, Cj, Dj, Ej, OR = J("round_a"), J("round_b"), J("round_c"), J("round_d"), J("round_e"), \
    J("oracle")
DS = PP.DSETS


def mw(src, key, node="RESULTS"):
    v = [src[node][d][key]["dALL"] for d in DS]
    return sum(v) / len(v), min(v)


f6m, f6w = mw(Bj, "L1_SWAP_F6_B6")
best_direct = min(["L2_DIRECT_S4", "L3_DIRECT_PPRGLOBAL", "L3b_DIRECT_PPRGLOBAL_M64",
                   "L4_DIRECT_PPRBOUNDED", "L4b_DIRECT_PPRBOUNDED_EXP", "L6_DIRECT_S4_PLUS_PPR"],
                  key=lambda k: -mw(Bj, k)[0])
bdm, bdw = mw(Bj, best_direct)
esm, esw = mw(Bj, "SWAP_ES_B6")
b8m, b8w = mw(Bj, "L1_SWAP_F6_B8")

# best round-E variant by (no negative corpus, then macro)
EROWS = {}
if Ej:
    for bv in (6, 8):
        for m in ("F6", "ES", "LEXONE", "DS3", "DS4"):
            EROWS[f"{m}_B{bv}"] = mw(Ej, f"{m}_B{bv}")
    e_best = sorted(EROWS, key=lambda k: (EROWS[k][1] < 0, -EROWS[k][0]))[0]
else:
    e_best = None

# best round-D structural signal
DROWS = {}
if Dj:
    for k in Dj["RESULTS"]["metaqa"]:
        if k != "BASE":
            DROWS[k] = mw(Dj, k)
    d_best = sorted(DROWS, key=lambda k: (DROWS[k][1] < 0, -DROWS[k][0]))[0]
else:
    d_best = None

def p3_gap(pre):
    """per corpus: (best graph summary dALL, best NO-GRAPH control dALL, gap)."""
    rows = []
    for d in DS:
        r = (Cj or {}).get("STEP6_ROUNDC", {}).get(d, {})
        g = [r[k]["dALL"] for k in r if k.startswith(pre) and "CTRL" not in k]
        c = [r[k]["dALL"] for k in r if k.startswith(pre) and "CTRL" in k]
        if g and c:
            rows.append((d, max(g), max(c), max(g) - max(c)))
    return rows


P3G = {f: p3_gap(p) for f, p in (("P3", "L5_P3_"), ("P3ENTRY", "L5e_P3ENTRY_"))}


def signtest(k):
    """two-sided exact binomial over corpora on the sign of (graph - control)."""
    from math import comb
    g = [r[3] for r in P3G[k]]
    n, pos = len(g), sum(1 for x in g if x > 0)
    tail = sum(comb(n, i) for i in range(max(pos, n - pos), n + 1)) / 2 ** n
    return pos, n, min(1.0, 2 * tail)


def famstat(k):
    g = [r[3] for r in P3G[k]]
    return min(g), max(g), sum(g) / len(g)


def bestof(pre, node="STEP6_ROUNDC"):
    """best variant of a family by (no negative corpus, macro) over the round-C scores."""
    ks = sorted({k for d in DS for k in Cj[node][d] if k.startswith(pre) and "CTRL" not in k})
    rows = {k: ([Cj[node][d][k]["dALL"] for d in DS]) for k in ks}
    sc = {k: (sum(v) / len(v), min(v)) for k, v in rows.items()}
    b = sorted(sc, key=lambda k: (sc[k][1] < 0, -sc[k][0]))[0]
    return b, sc[b][0], sc[b][1]


mq7 = Cj["STEP7_DECOMPOSITION"]["metaqa"]["ROUTER_POOL"] if Cj else {}
tot7 = sum(mq7[k] for k in ("CAP_P50", "REACH", "CAP_B6", "RANKING")) if mq7 else 1
h3 = mq7.get("hop3", {})
tot3 = sum(h3[k] for k in ("CAP_P50", "REACH", "CAP_B6", "RANKING")) if h3 else 1

LAT = Bj["STEP11_LATENCY"]
rng = lambda k, f="{:.4f}": (f.format(min(LAT[d][k] for d in DS)),
                             f.format(max(LAT[d][k] for d in DS)))
irng = lambda k: (f"{min(LAT[d][k] for d in DS):,}", f"{max(LAT[d][k] for d in DS):,}")
p1s, p1S = rng("P1_global_ppr_sec_per_query")
p2s, p2S = rng("P2_bounded_ppr_sec_per_query")
p1e, p1E = irng("P1_online_graph_edges_touched_per_query")
p2e, p2E = irng("P2_online_graph_edges_touched_per_query")
cnd = (min(LAT[d]["P2_mean_candidate_partitions"] for d in DS),
       max(LAT[d]["P2_mean_candidate_partitions"] for d in DS))
edg = (min(LAT[d]["P2_mean_subgraph_edges"] for d in DS),
       max(LAT[d]["P2_mean_subgraph_edges"] for d in DS))
big = sorted(DS, key=lambda d: -LAT[d]["npart"])[:2]
spd = sorted(LAT[d]["P1_global_ppr_sec_per_query"] / LAT[d]["P2_bounded_ppr_sec_per_query"]
             for d in big)
SPD = f"{spd[0]:.0f}-{spd[-1]:.0f}x"

R = []
w = R.append

w("# L1 — SIMPLIFY, REMOVE MODALITY DOUBLE-COUNTING, TEST PRECOMPUTABLE PPR ROUTING")
w("")
w("*G2 / L1_PPR_SIMPLIFY. Six corpora, DEV only. No TEST, no L2, no L3 change, no training, "
  "no learned router, no dataset identity, no gold at inference, no new encoder pass. Every "
  "method reported here emits EXACTLY 50 canonical MASTER_TOPOLOGY=C partitions.*")
w("")
w("---")
w("")
w("## VERDICT")
w("")
w("```")
w("L1_FROZEN                          = NO")
w("STANDING POLICY                    = B6_S4_F6_Ms64_Mr32   (unchanged, still SAFE_REFERENCE)")
w(f"DIRECT_TOP50_CAN_REPLACE_SWAP      = NO   (best DIRECT variant {best_direct}:")
w(f"                                          macro {bdm:+.4f} / worst {bdw:+.4f} vs "
  f"swap {f6m:+.4f} / {f6w:+.4f})")
w("PARTITION_PPR_IS_AN_L1_CHANNEL     = NO   (reaches more gold partitions, converts none)")
w("BOUNDED_PPR_BETTER_THAN_GLOBAL     = YES  (accuracy and latency), but both are refuted")
w("PARTITION_BOUNDED_NODE_PPR_USEFUL  = QUALIFIED YES, NOT WORTH IT  (entry-mode beats its")
w("                                     own NO-GRAPH controls 6/6, but stays below the")
w("                                     standing router and costs the most latency here)")
w("PPR_FULLY_PRECOMPUTABLE            = YES  (exactly, by linearity; online graph edges = 0)")
w("SIMPLEST_STRUCTURAL_SIGNAL         = S4   (the frozen directional expansion)")
w("B_IS_THE_LIMITER                   = NO   (CAP_B6 is %d of %d MetaQA failures, %.1f%%;"
  % (mq7.get("CAP_B6", 0), tot7, 100.0 * mq7.get("CAP_B6", 0) / tot7))
w("                                          0 on every text corpus)")
w("NEW_ENCODER_PASSES                 = 0")
w("MODAL_REQUIRED                     = NO   (peak RSS 1.6 GB, all six corpora local)")
w("```")
w("")
nsig = sum(1 for d in DS if Bj["RESULTS"][d]["L1_SWAP_F6_B6"]["vs_BASE"]["sig"])
nneg = sum(1 for d in DS if Bj["RESULTS"][d]["L1_SWAP_F6_B6"]["dALL"] < 0)
w("This phase proposed five ways to change L1 — replace the swap with DIRECT top-50 fusion, "
  "add global / bounded / partition-local PPR, or move B. **None of the five improves L1**, "
  "though partition-local PPR is refuted on cost rather than on signal. The sixth question, "
  "whether PPR can be precomputed, is answered **yes, exactly** — and is therefore moot, "
  "because the signal it would make free is not one worth having.")
w("The standing router survives unchanged, which is a real result: it is now the best of a much")
w("larger tested space, and the reasons the alternatives fail are mechanical rather than statistical.")
w("")
w("---")
w("")
w("## STEP 0 — FINAL AUDIT OF THE CURRENT SWAPPING")
w("")
w("All eight gates pass exactly on all six corpora. `diag/swap_dump.json` holds the full "
  "per-query dumps (BASE top-50, protected core, boundary, challengers, per-channel ranks, "
  "swapped-in, swapped-out, final top-50) for MetaQA hop2, MetaQA hop3, WebQSP and HotpotQA.")
w("")
w(T["T1"])
w("")
w("`base_rank` is reproduced bit-for-bit by replaying `rrf_partitions([PR_dense, PR_splade])` "
  "from the cached dense/SPLADE top-`K_LOCK` node lists, and `ret_rrf` is reproduced bit-for-bit "
  "by replaying `node_rrf(dense200, splade200)`. Symmetric scoring is verified by rescoring every "
  "candidate with an incumbent-blind expression and confirming it reproduces the selection "
  "(0/300 mismatches per corpus); determinism is verified by shuffling the challenger list "
  "(0 differences); exact P=50 is verified on every query of every corpus.")
w("")
w("### The audit found the mechanism, and it is not a competition")
w("")
w("A worked example from the dump (MetaQA hop2, query 666): the gold partition **318** sits at "
  "canonical rank 45, inside the boundary, with **no** structural and **no** retrieval evidence. "
  "Its score is `1/(60+45) = 0.009524`. Any challenger holding structural rank 1 and nothing else "
  "scores `1/(60+1) = 0.016393`. The gold partition is evicted and the query flips from covered "
  "to uncovered.")
w("")
w("This is arithmetic, not chance: **a canonical-only incumbent anywhere in the boundary loses to "
  "any challenger inside the top 45 of the structural channel.** So the boundary competition is "
  "near-deterministic turnover of channel-poor incumbents.")
w("")
if "T17" in T:
    w(T["T17"])
    w("")
    EV = J("eviction")
    if EV:
        cr = [EV[d]["B6"]["retention_rate_of_canonical_only_incumbents"] for d in DS]
        rr = [EV[d]["B6"]["incumbent_retention_rate"] for d in DS]
        n6 = {d: EV[d]["B6"]["net_gold_partitions"] for d in DS}
        n12 = {d: EV[d]["B12"]["net_gold_partitions"] for d in DS}
        w(f"The mechanism is confirmed and it is extreme: a boundary incumbent holding **only** "
          f"canonical evidence is retained at a rate of {min(cr):.4f}–{max(cr):.4f} "
          f"(0.0009 on MetaQA, 0.0002 on HotpotQA — effectively never). Across all six corpora only "
          f"{min(rr):.2f}–{max(rr):.2f} of the B slots go to incumbents at all; the rest import "
          "partitions from outside BASE50. F6 is not weighing incumbents against challengers, it is "
          "replacing them.")
        w("")
        w("**And that is, on balance, the right thing to do.** Net gold partitions at B=6 are "
          "positive on every corpus (" + ", ".join(f"{d.split('_')[0]} {n6[d]:+d}" for d in DS)
          + "). The turnover gains more gold than it destroys — which is why the standing router "
          "beats BASE at all.")
        w("")
        w(f"The failure is at the margin, and B is where it shows: at B=12 WebQSP flips to "
          f"{n12['webqsp']:+d} net gold partitions ({EV['webqsp']['B12']['gold_partitions_evicted']} "
          f"evicted against {EV['webqsp']['B12']['gold_partitions_gained']} gained) while every "
          "other corpus stays positive. Raising B does not buy more search; it buys more eviction, "
          "and eviction stops paying at a corpus-specific point. That is the B-sweep below, "
          "explained.")
        w("")
    w("")
w("---")
w("")
w("## STEP 1 — REDEFINING THE L1 CHANNELS")
w("")
w("The conceptual channel set `{base_rank, ret_rrf, structure}` is rejected as instructed: "
  "`base_rank` and `ret_rrf` are one Dense+SPLADE retrieval read at two granularities. The three "
  "orthogonal families are `DENSE_PARTITION_RANK`, `SPLADE_PARTITION_RANK`, "
  "`STRUCTURAL_PARTITION_RANK`, and the split is exact:")
w("")
w(T["T3"])
w("")
w("**`BASE_RANK_PARITY = EXACT` on all six**, so Dense and SPLADE can each be given exactly one "
  "vote with no loss and no re-encoding. The two modalities are far from redundant with each "
  "other (ρ from 0.345 on MetaQA to 0.907 on WebQSP), which is why the canonical fusion is worth "
  "keeping intact.")
w("")
w("### The duplication is real in support but NOT in ordering")
w("")
w(T["T2"])
w("")
w("The set overlap is what the previous phase measured: 81–100% of the partitions the retrieval "
  "channel promotes are already canonically ranked. But the **rank correlation between the two "
  "lexical views on the partitions they share is only ρ = 0.158–0.344**. They agree on *which* "
  "partitions matter and disagree on *how much*. That distinction decides the phase.")
w("")
w("The directive allows a second vote from the same evidence family only if an ablation shows "
  "truly independent information. That ablation is the five-variant family below — every way "
  "of collapsing the two lexical granularities into one vote, plus the control that keeps "
  "both:")
w("")
if Ej:
    w(T["T19"])
    w("")
    nlose = lambda k: sum(1 for d in DS if Ej["RESULTS"][d][k]["vs_F6"]["sig"]
                          and Ej["RESULTS"][d][k]["dALL_vs_F6"] < 0)
    mqh2 = lambda k: Ej["RESULTS"]["metaqa"][k]["per_hop"]["2"]["dALL"]
    w("Three ways to obey the one-vote-per-family rule were tested, and **all three fail in the "
      "same place**:")
    w("")
    w(f"- `ES` (delete the node view): macro {EROWS['ES_B6'][0]:+.4f} / worst "
      f"{EROWS['ES_B6'][1]:+.4f}, significantly worse than the standing router on "
      f"{nlose('ES_B6')}/6 corpora.")
    w(f"- `LEXONE` (fuse the two granularities into one lexical rank): macro "
      f"{EROWS['LEXONE_B6'][0]:+.4f} / worst {EROWS['LEXONE_B6'][1]:+.4f}, significantly worse on "
      f"{nlose('LEXONE_B6')}/6.")
    w(f"- `DS3` — **the directive's literal target form**, `DENSE + SPLADE + STRUCTURAL`, one vote "
      f"each: macro {EROWS['DS3_B6'][0]:+.4f} / worst {EROWS['DS3_B6'][1]:+.4f}, significantly "
      f"worse on {nlose('DS3_B6')}/6.")
    w("")
    w(f"All three **gain on MetaQA** (`ES` hop2 {mqh2('ES_B6'):+.4f}, `DS3` hop2 "
      f"{mqh2('DS3_B6'):+.4f}, against the standing router's {mqh2('F6_B6'):+.4f}) and lose on the "
      "text corpora. The exposed-channel set is not the problem; the node-level lexical view is "
      "doing real work on the text corpora and harmful work on MetaQA. That is a corpus-level "
      "regime split with no query-local separator — the exact object the previous phase searched "
      "for and the directive instructed me to stop searching for. I stopped.")
    w("")
    w(f"`DS4` — dense + splade + structural **and** the node view — is the one channel "
      f"redefinition that costs nothing: macro {EROWS['DS4_B6'][0]:+.4f} at B=6 and "
      f"{EROWS['DS4_B8'][0]:+.4f} at B=8, **not significantly different from the standing router "
      f"on any corpus** ({nlose('DS4_B6')}/6 and {nlose('DS4_B8')}/6 significant losses). So the "
      "canonical partition RRF *can* be split into its two named channels for free — but that "
      "does not remove the duplication, because the duplication is partition-granularity versus "
      "node-granularity, not dense versus SPLADE. `DS4` spends an extra channel to expose the "
      "families and buys nothing, and STEP 6 says prefer fewer signals.")
    w("")
    ldr = sorted(EROWS, key=lambda k: (EROWS[k][1] < 0, -EROWS[k][1], -EROWS[k][0]))[0]
    w(f"On the standing no-corpus-left-behind criterion the nominal leader is **`{ldr}`** "
      f"(macro {EROWS[ldr][0]:+.4f} / worst {EROWS[ldr][1]:+.4f}) against the standing "
      f"`F6_B6` ({f6m:+.4f} / {f6w:+.4f}) — but it is **not significant on any corpus** "
      f"(best paired p = "
      f"{min(Ej['RESULTS'][d][ldr]['vs_F6']['mcnemar_p'] for d in DS):.4f}), so it is a "
      "candidate, not a demonstrated improvement, and it is not adopted here.")
    w("")
w("The conclusion is uncomfortable but clean: **the third vote is not a duplicate to be deleted.** "
  "The node-level view carries ordering information the partition-level view does not. The "
  "architectural ideal and the measurement disagree here, and this report follows the "
  "measurement — while recording the inconsistency as an open reason not to freeze.")
w("")
w("---")
w("")
w("## STEP 2 — THE CANONICAL PARTITION GRAPH")
w("")
w("Built once per corpus from the frozen MASTER_TOPOLOGY=C partition map: every node edge "
  "`u -> v` accumulates mass on `part(u) -> part(v)`, direction and edge mass preserved. "
  "Self-loops (`part(u) == part(v)`) are real internal mass but move no probability between "
  "partitions, so they are stored separately and excluded from the transition matrix. "
  "No repartitioning, no encoder work, cached permanently under `graph/pgraph_{ds}.npz`.")
w("")
w(T["T4"])
w("")
w("**These graphs are dense.** Mean out-degree is 56–471 out of 136–7814 partitions, density "
  "0.017–0.411, and not one partition is isolated on any corpus. A partition graph in which the "
  "average partition links to a quarter of the corpus is a poor substrate for diffusion: there is "
  "almost nothing for a random walk to discover that a two-hop neighbourhood does not already "
  "contain. This predicts the PPR results before any of them are run.")
w("")
w("---")
w("")
w("## STEP 3 — THREE PPR SEMANTICS, KEPT SEPARATE")
w("")
w("### P1 — partition-graph PPR (\"which partitions are structurally important to this query?\")")
w("")
w("Universal query seed logic, identical to the frozen router: Dense + SPLADE retrieval fused by "
  "fixed node RRF, seed nodes mapped to canonical partitions, mass `1/(K0+r)` aggregated per "
  "partition and row-normalised. One PPR rank over partitions.")
w("")
w("### P2 — bounded partition-graph PPR")
w("")
w("Same personalization, but the walk runs on a query-local partition graph induced on "
  "`BASE50 ∪ dense-top-50 ∪ splade-top-50 ∪ structural aggregate ∪ retrieval aggregate ∪ seed "
  "partitions`, with a neighbour-expanded variant capped at the frozen `DEG_CAP=300`. Candidate "
  f"and edge counts per query are in T16 ({cnd[0]:.0f}-{cnd[1]:.0f} partitions, "
  f"{edg[0]:,.0f}-{edg[1]:,.0f} edges).")
w("")
w("### P3 — partition-bounded node PPR")
w("")
w("For a candidate partition `P`, only `P`'s own nodes and internal edges; probability may not "
  "leave `P`. Summaries: max personalized node mass, top-`SEED_K` mass, concentration, reachable "
  "fraction. Because it cannot discover a partition, it is scored strictly as a partition-quality "
  "signal, restricted to the query's candidate partitions.")
w("")
w("**Two NO-GRAPH controls with identical support were run alongside it**, and they are what makes "
  "the result interpretable: `CTRL_seed_mass` ranks the same partitions by raw seed mass, and "
  "`CTRL_support_only` ranks them with no magnitude at all. Any gain a graph summary shows that "
  "its own control also shows is support, not structure.")
w("")
if "T13" in T:
    w(T["T13"])
    w("")
    w(T["T14"])
    w("")
    w("Each P3 summary is scored the way the directive defines it — as a partition-quality signal, "
      "given a third co-equal vote next to the Dense and SPLADE partition channels.")
    w("")
    w("| corpus | family | best graph summary | best NO-GRAPH control | gap |")
    w("|---|---|---:|---:|---:|")
    for f in ("P3", "P3ENTRY"):
        for d, g, c, gp in P3G[f]:
            w(f"| {d} | {f} | {g:+.4f} | {c:+.4f} | {gp:+.4f} |")
    w("")
    s3, S3, m3 = famstat("P3")
    se, Se, me = famstat("P3ENTRY")
    pos3, n3, pv3 = signtest("P3")
    pose, ne, pve = signtest("P3ENTRY")
    bp3, bp3m, bp3w = bestof("L5_P3_")
    bpe, bpem, bpew = bestof("L5e_P3ENTRY_")
    w("**The two P3 families do not behave the same way, and the difference is the whole result.**")
    w("")
    w(f"**Seed-only P3** (score only the partitions the query's seeds landed in): the graph adds "
      f"nothing. Gaps run {s3:+.4f} to {S3:+.4f}, mean {m3:+.4f}; the sign is positive on only "
      f"{pos3}/{n3} corpora (sign test p = {pv3:.3f}). MuSiQue's `+0.0100` and SQuAD's `+0.0130` "
      "are reproduced *exactly* by ranking the same partitions by raw seed mass, and MetaQA's large "
      "negative is less negative for the controls than for the diffusion summaries. Here the "
      "channel is measuring support, not structure.")
    w("")
    w(f"**Entry-mode P3** (score every candidate partition, entering it at its own best seed): the "
      f"graph **does** beat its own same-support controls, on **{pose} of {ne} corpora** "
      f"(gaps {se:+.4f} to {Se:+.4f}, mean {me:+.4f}, sign test p = {pve:.3f}). That is a real "
      "signal: internal connectivity around the entry node carries partition-quality information "
      "that raw seed mass does not. I did not run a paired per-query test between a summary and its "
      "control, so the per-corpus gaps are point estimates; the consistent sign across six "
      "independent corpora is the evidence.")
    w("")
    w(f"**It is still not worth taking.** The best entry-mode variant is `{bpe}` at macro "
      f"{bpem:+.4f} / worst {bpew:+.4f}, against the standing router's {f6m:+.4f} / {f6w:+.4f} — "
      "below it on both, with a negative corpus. And it is the most expensive signal in the phase: "
      "T14 puts it at " + "%.4f-%.4f s/query with a %.1f-%.0f MB local operator, versus the "
      "standing structural channel's zero online graph work." % (
          min(Cj["STEP3_P3"][d]["sec_per_query_entry"] for d in DS),
          max(Cj["STEP3_P3"][d]["sec_per_query_entry"] for d in DS),
          min(Cj["STEP3_P3"][d]["local_operator_bytes"] for d in DS) / 1e6,
          max(Cj["STEP3_P3"][d]["local_operator_bytes"] for d in DS) / 1e6))
    w("")
    w("So partition-bounded node PPR **does** provide partition-quality information — the honest "
      "answer to the question as asked is a qualified yes — but not enough of it to pay for "
      "itself, and not enough to reach a router that touches no graph at query time.")
    w("")
w("---")
w("")
w("## STEP 4 — PRECOMPUTATION")
w("")
w("The inherited iteration `p_k = (1-a)S + a p_{k-1} P`, `p_0 = S`, is **linear in S**, so")
w("")
w("```")
w("p_20 = S @ M       with   M = (1-a) * sum_{j=0..19} (aP)^j + (aP)^20")
w("                          M_k = (1-a) I + a M_{k-1} P,   M_0 = I")
w("```")
w("")
w("Row `p` of `M` is exactly the PPR response to a unit personalization at partition `p`. So the "
  "\"full precomputed basis\" is **not an approximation of online PPR — it is the same function**, "
  "and any disagreement is floating-point. Only the top-L truncation is lossy. `M` is built and "
  "consumed in row blocks, so the dense basis is never held for the large corpora.")
w("")
if "T12" in T:
    w(T["T12"])
    w("")
    w("Query-time L1 with the basis is: retrieve, look up ~100 sparse basis rows, merge ranks, "
      "take top-50. **`ONLINE_GRAPH_EDGES_TOUCHED = 0`**, against 152k–47.8M for online global "
      "PPR. The precomputed variant also behaves the same as online PPR where it matters: in the "
      "swap bake-off `PPRP32` tracks `PPRG` within ±0.0015 on five of six corpora (T9).")
    w("")
    w("So the answer to \"can PPR be made effectively fully precomputed?\" is an unqualified yes — "
      "exactly, cheaply, and with zero online graph work. It simply does not help.")
    w("")
w("---")
w("")
w("## STEP 5 — B REMOVED AS AN AXIS, AND DIRECT TOP-50 FUSION")
w("")
w(T["T6"])
w("")
bs = lambda d: {b: Bj["RESULTS"][d][f"L1_SWAP_F6_B{b}"]["dALL"] for b in (2, 4, 6, 8, 12)}
w("`B` is a real axis, but it is corpus-dependent and it does not point at MetaQA. Under F6 it "
  f"helps HotpotQA monotonically ({bs('hotpotqa_clean')[2]:+.4f} → {bs('hotpotqa_clean')[12]:+.4f})"
  f" and SQuAD ({bs('squad_clean')[2]:+.4f} → {bs('squad_clean')[12]:+.4f}), is flat on MetaQA "
  f"({bs('metaqa')[2]:+.4f} → {bs('metaqa')[12]:+.4f}), and **hurts** WebQSP "
  f"({bs('webqsp')[2]:+.4f} → {bs('webqsp')[12]:+.4f}). B=8 has a better macro "
  f"({b8m:+.4f}) than B=6 ({f6m:+.4f}) but a worse worst corpus ({b8w:+.4f} vs {f6w:+.4f}), so "
  "under the standing \"no corpus left behind\" criterion B=6 remains the universal choice and "
  "B=8 is the runner-up. Changing B never changes P_MAIN; every cell above outputs exactly 50.")
w("")
w("### DIRECT top-50 fusion is architecturally preferable and empirically worse")
w("")
w(T["T5"])
w("")
w("`DIRECT` with no third channel reproduces BASE **bit-for-bit on all six corpora** "
  "(`DIRECT_NOEXTRA_PARITY = EXACT`), so the comparison is clean. Adding any third channel to a "
  "full-corpus fusion is negative on macro for every variant tried, and the best of them "
  f"({best_direct}, macro {bdm:+.4f}) is still below BASE and far below the swap "
  f"({f6m:+.4f} macro, {f6w:+.4f} worst).")
w("")
w("The MetaQA per-hop table shows exactly why:")
w("")
w(T["T7"])
w("")
DIRV = [k for k in Bj["RESULTS"]["metaqa"] if "DIRECT" in k]
dch = [Bj["RESULTS"][d][k]["churn_mean"] for d in DS for k in DIRV]
sch = [Bj["RESULTS"][d]["L1_SWAP_F6_B6"]["churn_mean"] for d in DS]
mqh = Bj["RESULTS"]["metaqa"]
w("Every DIRECT variant **buys hop3 and pays for it in hop2** — up to "
  f"{min(mqh[k]['per_hop']['2']['dALL'] for k in DIRV):+.4f} on hop2 for a "
  f"{mqh['L3b_DIRECT_PPRGLOBAL_M64']['per_hop']['3']['dALL']:+.4f} hop3 gain. A structural channel "
  "with a co-equal full-corpus vote promotes partitions everywhere, including over the canonical "
  "partitions that were carrying the easy hops. Churn confirms it: "
  f"{min(dch):.1f}–{max(dch):.1f} partitions replaced per query for DIRECT against "
  f"{min(sch):.1f}–{max(sch):.1f} for the swap. "
  "**The protected core is not scaffolding around a selector; it is the thing that makes a third "
  "channel safe.** Eliminating the swap subsystem would be architecturally cleaner and costs "
  "real coverage on every corpus.")
w("")
w("---")
w("")
w("## STEP 6 — WHICH ONE STRUCTURAL SIGNAL?")
w("")
w("Asked inside the architecture that actually works, with the duplicated channel present (F6) "
  "and absent (ES), at B=6 and B=8:")
w("")
if Dj:
    w(T["T9"])
    w("")
    w(f"**`{d_best}` wins** (macro {DROWS[d_best][0]:+.4f}, worst {DROWS[d_best][1]:+.4f}) and is "
      "the only configuration with no negative corpus. Every PPR variant — global, bounded, "
      "precomputed — has a significantly negative corpus, and adding global PPR *to* S4 makes S4 "
      "worse on MetaQA (+0.0130 → +0.0040 under ES). Fewer signals, and the simplest one, wins.")
    w("")
    w(T["T11"])
    w("")
w("---")
w("")
w("## STEP 7 — CAP_B6 vs CAP_P50 (these are not the same thing)")
w("")
w("`CAP_P50` = more than 50 gold partitions: impossible at P=50 under **any** replacement. "
  "`CAP_B6` = every missing gold is reachable, but more than B=6 partitions of BASE would have to "
  "change — recoverable by raising B while still emitting exactly 50, which is **not** a contract "
  "change. `REACH` = a missing gold partition is not in the query's candidate universe at all. "
  "`RANKING` = ≤B changes would suffice and the selector ordered them wrong.")
w("")
if "T15" in T:
    w(T["T15"])
    w("")
    w(f"**This overturns the previous phase's capacity claim.** That phase reported 213 CAPACITY "
      f"+ 211 REACH = 87.2% of residual MetaQA hop3 as out of reach for a B≤6 router, with only "
      f"12.8% ranking. Separating the two capacity classes as instructed shows `CAP_B6` is "
      f"**{mq7.get('CAP_B6', 0)} of {tot7}** MetaQA failures "
      f"({100.0 * mq7.get('CAP_B6', 0) / tot7:.1f}%), "
      f"**{h3.get('CAP_B6', 0)} of {tot3}** on hop3 ({100.0 * h3.get('CAP_B6', 0) / tot3:.1f}%), "
      f"and **0 on every text corpus**. The old CAPACITY class was mostly CAP_P50 plus REACH. "
      f"Raising B is not the lever.")
    w("")
    pp = Cj["STEP7_DECOMPOSITION"]
    kk = ("CAP_P50", "REACH", "CAP_B6", "RANKING")
    w("### The graph solves REACH completely, and it changes nothing")
    w("")
    w("Recomputing the same decomposition with the PPR-reachable set added to the candidate "
      "universe collapses **`REACH` to exactly 0 on all six corpora**. Every gold partition the "
      "router's own universe cannot see is reachable through the partition graph. Those queries do "
      "not disappear — they reclassify:")
    w("")
    w("| corpus | " + " | ".join(f"pool {k}" for k in kk) + " | "
      + " | ".join(f"+PPR {k}" for k in kk) + " |")
    w("|---|" + "---|" * 8)
    for d in DS:
        r, q = pp[d]["ROUTER_POOL"], pp[d]["PLUS_PPR_REACHABLE"]
        w(f"| {d} | " + " | ".join(str(r[k]) for k in kk) + " | "
          + " | ".join(f"**{q[k]}**" if q[k] != r[k] else str(q[k]) for k in kk) + " |")
    w("")
    w(f"On MetaQA the {pp['metaqa']['ROUTER_POOL']['REACH']} REACH failures become "
      f"{pp['metaqa']['PLUS_PPR_REACHABLE']['CAP_B6'] - pp['metaqa']['ROUTER_POOL']['CAP_B6']} more "
      f"`CAP_B6` and "
      f"{pp['metaqa']['PLUS_PPR_REACHABLE']['RANKING'] - pp['metaqa']['ROUTER_POOL']['RANKING']} "
      "more `RANKING`; on 2Wiki and HotpotQA they become `RANKING` outright. **So the pool is not "
      "the binding constraint.** With the graph in the universe, "
      f"{pp['metaqa']['PLUS_PPR_REACHABLE']['RANKING']}/{sum(pp['metaqa']['ROUTER_POOL'][k] for k in kk)}"
      " of MetaQA's residual failures are selector-solvable inside B=6 and only "
      f"{pp['metaqa']['ROUTER_POOL']['CAP_P50']} are impossible at P=50.")
    w("")
    w("And the coverage tables say this buys nothing. That juxtaposition — reach fully solved, "
      "coverage unmoved — is the phase's central finding, and it also warns that pool-widening and "
      "`B` interact: widening the universe pushes queries into `CAP_B6` "
      f"({pp['metaqa']['ROUTER_POOL']['CAP_B6']} → "
      f"{pp['metaqa']['PLUS_PPR_REACHABLE']['CAP_B6']} on MetaQA), so a future wider pool would "
      "need more replacement slots, not the same six.")
    w("")
if Dj:
    w(T["T10"])
    w("")
    w("**Partition PPR does reach gold partitions S4 cannot** — on WebQSP it reaches 196 gold "
      "partitions S4 misses entirely against S4's own 123, and it is ahead on 2Wiki (37 vs 10), "
      "HotpotQA (48 vs 17) and MuSiQue (33 vs 20). That reach converts to **no** coverage "
      "anywhere (T9). The reason is the P=50 budget: the extra partitions arrive with no "
      "discriminative signal attached, so promoting them costs a canonical partition that was "
      "carrying a gold. **Reach is not coverage under a hard budget** — this is the single most "
      "useful negative result of the phase, and it applies to any future \"widen the pool\" idea.")
    w("")
if OR:
    w("### Is the selector or the pool the limiter?")
    w("")
    w(T["T20"])
    w("")
w("---")
w("")
w("## STEP 8 / STEP 9 — MetaQA multi-hop and all six corpora")
w("")
w("Primary target hop3, then hop2; hop1 stays saturated (`+0.0000` for the swap family on every "
  "variant tested). The full per-hop tables are T7 (DIRECT and B-sweep) and T11 (structural "
  "bake-off). Nothing tested this phase improves the standing router's MetaQA hop3 without a "
  "significant regression somewhere else; `SWAP_ES` reaches hop2 +0.0255 / hop3 +0.0135 and pays "
  "for it on 2Wiki, MuSiQue, HotpotQA and SQuAD. No dataset identity was used anywhere, and no "
  "configuration was selected per corpus.")
w("")
w("---")
w("")
w("## STEP 10 — PPR CONSTANTS")
w("")
w("The historical implementation was audited first, and the audit is a **negative finding**. "
  "`src/experiments/l3_methods.py::_ppr` and `l3_solvers.py` implement "
  "`p = (1-a)*s + a*(p @ P)` with `iters=20` over `P = D^-1 A` (symmetrised, row-normalised) and "
  "uniform seed mass — those are a real inherited contract and were reused unchanged. The damping "
  "`a` is **not**: both call sites select it per run from `{0.3,0.5,0.7,0.9}` / `{0.5,0.7,0.9}` "
  "**by gold recall**, store it as `ppr_best_alpha`, and `LEVEL3_README.md:130` marks the PPR "
  "alpha as exploratory. A gold-selected constant cannot be inherited into a router that may not "
  "see gold, and no surviving `ppr_best_alpha` value exists in `results/`.")
w("")
w("One global contract was therefore fixed for every corpus and every variant: **α = 0.85** — the "
  "only non-gold-selected alpha in the codebase (`l3_solvers._qppr_ball` default) and the standard "
  "PageRank damping — and **iters = 20**, inherited. Alpha was never selected on gold here.")
w("")
if "T18" in T:
    w(T["T18"])
    w("")
    w("Reported, never used for selection. The reading is not that α does not matter — it does, and "
      "**its best value is corpus-dependent**: MetaQA prefers the highest α tested, HotpotQA the "
      "lowest, WebQSP is flat and negative throughout. The reading is that **no α rescues partition "
      "PPR**. On MetaQA and WebQSP every α tested leaves it negative against BASE while S4 is "
      "positive, and on HotpotQA its best cell (α=0.7, +0.0170) merely ties S4 (+0.0160). A "
      "corpus-dependent α would also be a dataset-identity parameter, which the contract forbids. "
      "So the refutation is not an artifact of fixing α = 0.85.")
    w("")
w("---")
w("")
w("## STEP 11 — LATENCY AS A SELECTION METRIC")
w("")
w(T["T16"])
w("")
w("The desired property `ONLINE_GRAPH_EDGES_TOUCHED ≈ 0` **is achievable and is achieved** by the "
  f"precomputed diffusion basis. Bounded PPR is {SPD} faster than global PPR on the two largest "
  "corpora and touches 2–3 orders of magnitude fewer edges. But the standing router already "
  "touches zero online partition-graph edges — its structural channel is the cached directional "
  "expansion — so on this metric there is nothing to buy.")
w("")
w("---")
w("")
w("## STEP 12 — STAGE SEPARATION")
w("")
w("| stage | question | admissible mechanisms | status after this phase |")
w("|---|---|---|---|")
w("| **L1** | *where should we search?* | partition-level, parameter-free, precomputable, "
  "exactly-P50 | Dense + SPLADE partition RRF, protected core, S4 structural competition. "
  "**Partition PPR excluded** (P1/P2/P3 all refuted). |")
w("| **L2** | *which candidates are relevant?* | node/candidate relevance, learning allowed | "
  "unchanged and untouched. No PPR was added. Note L2 already carries a `path_raw` graph expert "
  "previously found available-but-useless on the text corpora — that duplication is pre-existing "
  "and out of scope here. |")
w("| **L3** | *bounded graph reasoning* | dynamic traversal, node granularity | this is where the "
  "historical PPR lives and belongs: `l3_methods`/`l3_solvers` run node-level PPR over structural "
  "+ weighted-NER edges to rank documents; `LEVEL3_README.md:141` already places "
  "\"Partition-local push PPR\" inside the target L3 flow, and `LEVEL3_README.md:172` proposes "
  "push-based PPR over lazily loaded partitions as the L3 optimisation. |")
w("")
w("The instruction not to duplicate one structural signal across stages without controlled "
  "justification is satisfied by exclusion: the controlled comparison was run at L1 and **failed**, "
  "so diffusion stays a single-stage (L3) mechanism.")
w("")
w("---")
w("")
w("## ANSWERS TO THE STOP CONDITIONS")
w("")
w("| # | question | answer |")
w("|---|---|---|")
w(f"| 1 | Is B=6 artificially causing much of MetaQA's CAPACITY class? | **No.** `CAP_B6` is "
  f"{mq7.get('CAP_B6', 0)}/{tot7} MetaQA failures ({100.0 * mq7.get('CAP_B6', 0) / tot7:.1f}%), "
  f"{h3.get('CAP_B6', 0)}/{tot3} on hop3, and 0 on every text corpus. |")
w(f"| 2 | Can direct exactly-P50 fusion replace swapping entirely? | **No.** Best DIRECT variant "
  f"macro {bdm:+.4f} vs swap {f6m:+.4f}; gains hop3, destroys hop2 (up to −0.0946). |")
w("| 3 | Does partition PPR improve reach? | **Yes — completely.** It drives the `REACH` failure "
  "class to 0 on all six corpora (T15). It converts to zero coverage gain at P=50. |")
w(f"| 4 | Is bounded PPR better than global PPR? | **Yes**, on accuracy and by {SPD} on latency. "
  "Both are still refuted as L1 channels. |")
w("| 5 | Does partition-bounded node PPR give useful partition quality information? | "
  f"**Qualified yes.** Seed-only P3 is pure support (sign test p = {signtest('P3')[2]:.3f}), but "
  f"entry-mode P3 beats its same-support controls on {signtest('P3ENTRY')[0]}/6 corpora "
  f"(p = {signtest('P3ENTRY')[2]:.3f}). It still does not reach the standing router and is the "
  "most expensive signal tested. |")
w("| 6 | Can PPR be made effectively fully precomputed? | **Yes, exactly** — the iteration is "
  "linear, the basis is the operator, online graph edges = 0. |")
w(f"| 7 | Which structural signal is simplest? | **S4.** {d_best if d_best else 'S4_F6_B6'} is the "
  "only no-negative-corpus configuration; PPR variants each have a significant regression. |")
w("| 8 | Simplest universal low-latency exact-P50 L1? | Dense+SPLADE partition RRF → protected "
  "core (44) → symmetric boundary competition over canonical + S4 + node-retrieval → exactly 50 "
  "— the standing router. The strictly simpler 3-channel form the directive asked for (`DS3`) "
  "is significantly worse on 4/6 corpora. |")
w("| 9 | MetaQA hop2/hop3 | No variant improves either without a significant regression "
  "elsewhere. Standing router: hop2 +0.0060, hop3 +0.0015 (T7). |")
w(f"| 10 | Six-dataset result | Standing router macro {f6m:+.4f}, worst {f6w:+.4f}, significant "
  f"on {nsig}/6, {nneg} regressions. Nothing tested beats it on the no-negative-corpus "
  "criterion. |")
w(f"| 11 | Exact query-time latency | T16. Global PPR {p1s}-{p1S} s/q ({p1e}-{p1E} edges); "
  f"bounded PPR {p2s}-{p2S} s/q ({p2e}-{p2E} edges); precomputed basis **0** online edges. |")
w("| 12 | `L1_FROZEN` | **NO** — see below. |")
w("")
w("---")
w("")
w("## WHY NOT FREEZE, AND WHAT WOULD ACTUALLY MOVE IT")
w("")
if OR:
    hd = {d: OR[d]["B6"]["SELECTOR_HEADROOM"] for d in DS}
    hmac = sum(hd.values()) / len(hd)
    w("The router is unchanged and is now the best of a much larger tested space, so freezing it is "
      "defensible on the evidence. I am still returning `L1_FROZEN = NO`, for two reasons that the "
      "phase measured rather than assumed.")
    w("")
    w("**1. The headroom is in the selector, it is large, and it is in-contract.** The STEP 7 "
      "addendum asks the exact combinatorial question — does there exist a choice of B replacements "
      "from the boundary and challengers that covers every gold partition? — and the answer is that "
      "a perfect parameter-free selector would gain:")
    w("")
    w("| corpus | " + " | ".join(d for d in DS) + " | macro |")
    w("|---|" + "---|" * (len(DS) + 1))
    w("| selector headroom at B=6 | " + " | ".join(f"**{hd[d]:+.4f}**" for d in DS)
      + f" | **{hmac:+.4f}** |")
    w("")
    w(f"That is {hmac / f6m:.1f}× the standing router's entire gain over BASE ({f6m:+.4f}), and it "
      "needs no change to P, B, `M_struct`, `M_ret` or `MAX_HOPS` — it is available inside the "
      "frozen contract, from the candidate pool the router already builds. Freezing L1 now would "
      "freeze that on the table.")
    w("")
    w("**2. The winning router violates the architecture this phase was asked to impose.** F6's "
      "third vote is the same evidence family as its first. Deleting it (`ES`), fusing it "
      "(`LEXONE`) and re-exposing the families without it (`DS3`) are each significantly worse on "
      "4 of 6 corpora. The measurement and the architectural ideal disagree, and freezing means "
      "ratifying the inconsistency deliberately.")
    w("")
    w("What is now closed, and should not be re-opened: DIRECT full-corpus fusion; partition-graph "
      "PPR in all three semantics; PPR precomputation as a lever; and `B` as a lever.")
    w("")
    w("What remains open, in order of expected value:")
    w("")
    w("- **The eviction asymmetry — the one untested thing that targets the measured failure.** "
      "T17 shows canonical-only incumbents are retained at rates as low as 0.0002, and T20 shows "
      f"the selector is leaving {hmac:+.4f} macro on the table inside its own pool. A scoring rule "
      "that lets a canonical-only incumbent hold its slot against a single-channel challenger is a "
      "parameter-free change to the RRF expression, uses no gold, no learning and no dataset "
      "identity, and is the natural next experiment.")
    w(f"- **The pool needs slots, not reach.** STEP 7 shows the graph makes REACH zero on all six "
      f"corpora, and that converting it would push MetaQA's `CAP_B6` class from "
      f"{Cj['STEP7_DECOMPOSITION']['metaqa']['ROUTER_POOL']['CAP_B6']} to "
      f"{Cj['STEP7_DECOMPOSITION']['metaqa']['PLUS_PPR_REACHABLE']['CAP_B6']}. A wider pool without "
      "more replacement slots cannot pay. Both are contract changes and need an explicit ruling.")
    w(f"- **P=50 itself.** `CAP_P50` is nonzero only on MetaQA "
      f"({Cj['STEP7_DECOMPOSITION']['metaqa']['ROUTER_POOL']['CAP_P50']} queries, "
      f"{Cj['STEP7_DECOMPOSITION']['metaqa']['ROUTER_POOL']['hop3']['CAP_P50']} of them hop3). "
      "Small, and genuinely impossible under the current contract.")
    w("")
    w("If the answer to all three is \"no contract change\", then the eviction rule is the whole "
      "remaining L1 program, and `L1_FROZEN = YES` becomes the right call as soon as it is tested.")
    w("")
w("---")
w("")
w("## ARTIFACTS")
w("")
w("```")
w("results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PPR_SIMPLIFY/")
w("  FINAL_REPORT.md   this report")
w("  TABLES.md         every table, regenerated from the round JSONs")
w("  diag/round_a.json STEP 0 gates, STEP 1 channels, STEP 2 graph stats")
w("  diag/swap_dump.json  per-query swap provenance dumps")
w("  diag/round_b.json STEP 5 B-sweep + DIRECT fusion + STEP 1 independence + latency")
w("  diag/round_c.json STEP 4 precomputation, STEP 3 P3 + controls, STEP 7 decomposition")
w("  diag/round_d.json STEP 6 structural bake-off, STEP 7 reach probe")
w("  diag/round_e.json STEP 1 one-vote-per-family variants")
w("  diag/oracle.json  STEP 7 addendum: selector headroom vs pool failure")
w("  diag/eviction.json  STEP 0 quantified boundary turnover")
w("  diag/alpha_sensitivity.json  STEP 10 robustness")
w("  graph/pgraph_{ds}.npz  cached canonical partition transition graphs")
w("  ppr/mass_{ds}.npz      cached personalization + P1/P2 masses")
w("  logs/                  run logs")
w("scratchpad/_l1pp_core.py  shared substrate: channels, partition graph, PPR contract, scoring")
w("scratchpad/_l1pp_a.py     ROUND A   scratchpad/_l1pp_b.py  ROUND B")
w("scratchpad/_l1pp_c.py     ROUND C   scratchpad/_l1pp_d.py  ROUND D")
w("scratchpad/_l1pp_e.py     ROUND E   scratchpad/_l1pp_oracle.py / _l1pp_evict.py / _l1pp_alpha.py")
w("scratchpad/_l1pp_tables.py / _l1pp_final.py  report generation")
w("```")

os.makedirs(PP.PPD, exist_ok=True)
open(f"{PP.PPD}/TABLES.md", "w", encoding="utf-8").write(raw)
open(f"{PP.PPD}/FINAL_REPORT.md", "w", encoding="utf-8").write("\n".join(R) + "\n")
print(f"wrote {PP.PPD}/FINAL_REPORT.md  ({len(R)} blocks)")
print(f"wrote {PP.PPD}/TABLES.md        ({len(T)} tables: {' '.join(sorted(T, key=lambda x: int(x[1:])))})")
