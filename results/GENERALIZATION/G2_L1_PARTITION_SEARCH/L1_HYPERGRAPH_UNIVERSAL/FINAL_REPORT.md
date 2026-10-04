# UNIVERSAL TRUE-HYPERGRAPH + HALO -- final report

Continuation of [[l1-edge-closed-partition-weighting]] Part 1 (`O4_FULL_C_b0.5` halo, METIS core).
That program closed with two open questions: does a TRUE hypergraph partitioner (not a hard
partitioner fed a clique-expanded graph) beat METIS as the halo's core, and does any of this
survive frozen L2? This phase answers the first in full and scopes the second.

Frozen L1 contract, unchanged from Part 1: Mt-KaHyPar `DETERMINISTIC_QUALITY`, `KM1` objective,
`k ~ N/100`, `epsilon=0.03`, `seed=0`, identical node universe. Only the hypergraph
REPRESENTATION (how multi-way structural/NER/kNN relations become hyperedges) varies.

## STAGE 1-2 -- screening and the finalist

Nine representations (`H0_FIXED_CAP` .. `H4_SPLIT_PRESERVE`) screened on pin retention and a
fast accuracy pass. `H4_SPLIT_PRESERVE` won: oversized hyperedges are split into overlapping
anchor-preserving sub-hyperedges, weighted `1/(|e_i|-1)`, so **no pin is ever discarded**
(pin retention = 1.0000 on all six corpora, the only rule with that property alongside the
size-inflating `H1_FULL_WEIGHTED`, which keeps everything but never trims a degenerate giant
hyperedge either). See `pin_retention/A1_PRECAP_DISTRIBUTION.json`, `TABLES.md` A0/A1.

## STAGE 3 -- universal L1 gate (Phase A7-A10)

`H4_SPLIT_PRESERVE__SK` replayed through the frozen L1 contract on all six corpora, both lanes
(BASE = Dense+SPLADE partition ranking at P=50 with no selector; SAFE = the frozen
`B6_S4_F6_Ms64_Mr32` selector, mechanically reaggregated). Two corpora — WebQSP and HotpotQA —
structurally exceed this machine's 15.7 GB RAM to partition (Mt-KaHyPar's `create_hypergraph`
has no lower-memory bulk constructor and multilevel partitioning is not shardable — coarsening /
initial-partition / uncoarsening is a global optimisation over the whole graph; both facts were
verified directly, not assumed). Both ran under a user-approved narrow Modal exception, one job
at a time on a single account: WebQSP peaked at 125.0 GB RSS / 83 min, HotpotQA at 245.7 GB RSS
/ 13 min, against requested ceilings Modal allowed both to burst past (Modal's `memory=` is
soft/burstable, not a hard cgroup limit — worth knowing before sizing the next one). Pure CPU
throughout, `MODAL_NO_GPU_ANYWHERE = true`.

| corpus | SAFE delta vs METIS | McNemar p | sig? | reseed floor |
|---|---:|---:|---|---:|
| MetaQA | **+0.1276** | (exact, see gate) | YES | 0.0021 |
| WebQSP | **+0.0430** | (exact, see gate) | YES | 0.0077 |
| HotpotQA | **+0.0115** | (exact, see gate) | YES | 0.0025 |
| SQuAD | +0.0040 | -- | no | 0.0013 |
| 2Wiki | +0.0005 | -- | no | 0.0025 |
| MuSiQue | -0.0045 | -- | no | 0.0029 |

`UNIVERSAL_HYPERGRAPH_GATE_PASSED = [H4_SPLIT_PRESERVE__SK]`, the only representation with zero
significant regressions on all six. `HYPER_CROSS_CORPUS_SAFE = true`. Full per-representation
grid (all 9 rules x 6 corpora, both lanes) in `TABLES.md` A8/A9; balance envelope
(`max/mean <= 1.05`, all blocks used) holds for every eligible cell in `TABLES.md` A7.

**`HYPERGRAPH_VERDICT`: the true hypergraph representation, split rather than truncated, is
universally safe and recovers real accuracy where the earlier clique-expansion / hard-partitioner
line topped out.** MetaQA's multi-hop tail is the clearest single number: hop3 SAFE 0.2583 (Part 1
METIS baseline) -> 0.5901 here. (Original spec option labels are not verbatim-recoverable in this
session after compaction -- see note at the end of this report -- so this verdict is stated in
prose rather than quoting a specific letter.)

## STAGE 4 -- does a better core compose with the halo? (Phase B0-B9)

Two cores registered (`halo/CORES.json`): `C0_METIS` (production, unchanged) and
`C1_HYPER_UNIVERSAL` (`H4_SPLIT_PRESERVE__SK`, the Stage-3 winner). `C2_HYPER_BEST_DIAGNOSTIC`
was deliberately NOT registered -- `H2_Q99__SK` is the strongest single-corpus diagnostic cell,
but Stage 2 already settled on one finalist, and giving the diagnostic core a fair 6-corpus B-phase
would mean partitioning WebQSP/HotpotQA a second time under a second representation, which is new
large-corpus compute Stage 2's decision didn't call for. The halo itself (FULL_C one-hop, static
boundary-mass ranking, `|H_j| <= beta*|C_j|`) is untouched from Part 1.

**B3 hard gate (halo must never touch core ranking) -- passes on all 6 corpora x 2 cores.**
`core_only_identical_across_beta = true` and `replay_agrees = true` everywhere: the beta=0 core
coverage is bit-identical to the independently-computed Stage-3 replay, and every beta's
core-only coverage matches beta=0 exactly. Verified, not assumed (`halo/CORES.json`'s `b3` check).

**B6 -- additivity.** `A` = METIS no halo, `B` = hypergraph no halo, `C` = METIS + halo(0.5),
`D` = hypergraph + halo(0.5). `JOINT_EFFECT = D-A` is positive on all six (+0.0045 SQuAD to
+0.1466 MetaQA). `INTERACTION = D-B-C+A` is negative-to-zero on all six (-0.0630 MetaQA to
0.0000 MuSiQue) -- the two mechanisms are **sub-additive, never harmful**: they partly rescue the
same queries (a better core needs less halo repair, not a repair-resistant one), and stacking them
never costs anything. Full table: `TABLES.md` B6, `interaction/B6_INTERACTION.json`.

**B9 -- joint system (hypergraph core + halo@0.5) vs production METIS, paired McNemar** on the
per-query indicators computed for B6 (not point estimates):

| corpus | delta | gained | lost | p | sig |
|---|---:|---:|---:|---:|---|
| MetaQA | +0.1466 | 335 | 42 | 7.9e-58 | YES |
| WebQSP | +0.0747 | 123 | 17 | 5.2e-21 | YES |
| HotpotQA | +0.0130 | 41 | 15 | 6.9e-04 | YES |
| MuSiQue | +0.0095 | 53 | 34 | 0.053 | no |
| 2Wiki | +0.0075 | 58 | 43 | 0.163 | no |
| SQuAD | +0.0045 | 15 | 6 | 0.078 | no |

**Zero significant regressions, three significant gains, worst cell still +0.0045.** This is
the strongest of the three L1 lines this program has now run through the universal gate
(edge axis closed / partition-ranking-is-the-lever / this hypergraph+halo result).

**B5 depth** (required-node coverage at P1/5/10/25/50, both cores, core-only and core+halo):
`TABLES.md` B5. The hypergraph core front-loads recovery earlier in the ranking almost everywhere
-- e.g. 2Wiki P5 core-only 0.365 (METIS) vs 0.621 (hypergraph); HotpotQA P5 0.341 vs 0.523 -- even
on corpora where the P50 endpoint gain is small or not significant, which is a separate, useful
fact from the gate itself (a shallower selector would show these two cores diverging more, not
less, than the P50 numbers suggest).

**B7/B8 exposure.** Both cores use the identical beta=0.5 formula, so exposure multipliers land
close by construction (METIS 1.20-1.39x, hypergraph 1.20-1.39x) rather than being interpolated to
an exact match. Efficiency (`required_per_1k_extra_exposed`) is markedly LOWER for the hypergraph
core on MetaQA (0.160 vs METIS's 0.454) and WebQSP (0.100 vs 0.183) -- consistent with B6's
negative interaction: once the core itself recovers more, the halo has less marginal repair work
left, so it "spends" its exposure budget less efficiently in absolute terms even though the total
`D` still beats `C`. Full table: `TABLES.md` B7/B8, `exposure/B7_B8_EXPOSURE.json`.

**`JOINT_VERDICT`: hypergraph core + halo passes the universal gate against production METIS,
significantly on half the corpora and safely (non-negative, sub-floor) on the rest.** (Same
verbatim-recoverability caveat as Stage 3's verdict -- see note at the end.)

## PHASE C -- L2 survival: SCOPED, NOT RUN

The spec's own caution from [[l1-edge-closed-partition-weighting]] applies with more force here,
not less: a sibling L1 program already found `L1_GAIN_SURVIVES_FROZEN_L2 = NO` for a different L1
change (54-78% of the loss fell on queries whose L1 coverage was UNCHANGED -- pool-composition
sensitivity, not a real reranking failure). That is exactly the risk Phase C exists to check
before anything here gets promoted.

Investigating the frozen L2 policy (`C7b` archetype-distillation -> `C8c` XGBRanker -> `C11a`
interaction-MLP, per [[l2-c11-mlp-interaction]] and [[l2-c7-relation-gating]]) found two facts
that were not visible before opening the code:

1. **The frozen L2 substrate exists for exactly two corpora**: `2wiki_clean` and `musique_clean`
   (`l2_c8.DS = ('2wiki_clean', 'musique_clean')`; every cached feature file under
   `results/L2/_ctrl/` is keyed by only those two names). L2 was never built for MetaQA, WebQSP,
   HotpotQA or SQuAD -- a separate, already-tracked thread ([[g1-cross-dataset-generalization]])
   is mid-flight on exactly that gap and was not something this phase's spec asked for.
2. **The candidate pool each L2 stage reads is a static, pre-materialised artifact**
   (`data/l2_corpus/{ds}/{split}/cand_ids.npy` etc, built once by `build_l2_corpus.py` from the
   production METIS `hard` assignment). None of `C6` (relation gating) through `C11a`
   (interaction MLP) take a partition assignment as a runtime argument -- each is a frozen model
   trained against that one fixed pool. Running `L2_1`/`L2_2`/`L2_3` (METIS+halo / hypergraph-core
   / hypergraph+halo, per the spec's own definition) as anything more than a pool-membership check
   means rebuilding the candidate pool from each new core/halo assignment and re-deriving every
   downstream feature through all five intermediate stages (C6, C7, C8/C8c, C9, C10b) before C11a
   can even score it -- forward-pass only, no retraining, but five unfamiliar caching layers that
   took their own dedicated sessions to build originally, for a comparison that would still only
   cover 2 of the 6 corpora.

Those two corpora are also, pointedly, the two where B9 found the L1 gain NOT significant
(2Wiki +0.0075 p=0.163, MuSiQue +0.0095 p=0.053). A Phase C run scoped to what already exists
would test pool-composition sensitivity on the weakest part of this phase's result and leave the
headline MetaQA/WebQSP/HotpotQA gains -- the ones actually worth protecting from an L2 inversion
-- untested regardless. That combination (large new build, partial coverage, wrong two corpora)
is why Phase C stopped at scoping rather than executing; see the accompanying message for the
options being put to the user.

`l2_survival/` is left as an empty artifact directory pending that decision.

## STAGE 5 (PART 0B) -- benefit attribution + compute-constrained optimization

Given Stage 4's win, a follow-on spec asked where the benefit enters the system, which parts of
H4+halo are actually necessary, and whether the combined score can be improved at MATCHED
exposure -- all under a binding "Modal credits are scarce" constraint (LOCAL FIRST / CACHE FIRST /
MODAL LAST). A cache/sync audit (`LOCAL_CACHE_MANIFEST.json`, 602 files / 6.3 GB) found nothing
needed syncing from Modal; **every result below ran at compute tier L0-L2, zero Modal jobs.**
Drivers `scratchpad/_l1hu_{attr,global,e,g,h}.py`; artifacts under `attribution/`, `global_halo/`,
`halo_family/`, `support_multiplicity/`, `residual/`.

### Part F -- global dedup-aware halo (the primary optimization experiment)

Shipped mechanism (F0): each of a query's 50 selected cores gets its OWN independently-capped
beta=0.5 boundary-mass budget; a node supported by several cores gets no credit for that. Tested
at IDENTICAL per-query exposure (K_q = F0's own new-halo-node count, not a global constant):
`F1_MAX_BOUNDARY` (score = best single supporting core's mass) and `F2_SUM_NORMALIZED` (score =
sum of mass/that-core's-own-max-mass over every supporting core).

| corpus | F0 | F1 | F2 | F1 vs F0 (p, sig) | F2 vs F0 (p, sig) |
|---|---:|---:|---:|---|---|
| MetaQA | 0.8078 | 0.8083 | **0.8148** | ns (1.0) | **+0.0070 (p=.049, YES)** |
| WebQSP | 0.8393 | 0.8330 | **0.8548** | ns, below floor (p=.0039) | **+0.0155 (p=6.0e-05, YES)** |
| HotpotQA | 0.9635 | 0.9635 | **0.9665** | ns (1.0) | **+0.0030 (p=.031, YES)** |
| 2Wiki | 0.9510 | 0.9530 | 0.9475 | ns (.125) | -0.0035, ns (p=.092) |
| MuSiQue | 0.9730 | 0.9725 | 0.9735 | ns (1.0) | +0.0005, ns |
| SQuAD | 0.9920 | 0.9920 | 0.9920 | ns | ns |

**`F2_SUM_NORMALIZED` clears the phase's own promotion bar** (zero significant regressions, and
improves both worst-performing corpora, MetaQA/WebQSP). **`F1_MAX_BOUNDARY` does not** (flat or
negative everywhere -- max-of-supporters throws away exactly the signal that matters).
`PART_F_PROMOTION_CANDIDATE = F2_SUM_NORMALIZED`, not yet shipped (gated on Parts G/H below and
the still-open Phase C question).

### Part G -- support-multiplicity diagnostic (why F2 beats F1)

Confirms the mechanism directly: for every required node, count how many of the query's selected
cores halo it ("support count"), then compare that count across nodes F2 uniquely rescues vs F1
uniquely rescues vs both vs neither, against an ambient candidate-pool baseline.

| corpus | pool baseline mean | F2_ONLY mean (n) | F1_ONLY mean (n) | BOTH mean (n) |
|---|---:|---:|---:|---:|
| MetaQA | 3.16 | **10.13** (619) | 3.43 (252) | 7.88 (298) |
| WebQSP | 1.06 | **5.43** (134) | 3.78 (51) | 7.73 (165) |
| 2Wiki | 1.93 | **6.25** (4) | 4.67 (15) | 4.25 (4) |
| MuSiQue | 3.88 | **10.30** (10) | 5.67 (9) | 11.30 (23) |
| HotpotQA | 1.28 | **7.14** (7) | 0 (0) | 7.33 (3) |
| SQuAD | 4.92 | -- (0) | -- (0) | 35.0 (1) |

**F2_ONLY's mean support count exceeds F1_ONLY's on all 6/6 corpora where both groups are
non-empty** (SQuAD's residual population is too small to have either). This is not a coincidental
correlate: F2's edge over F1 is specifically, and universally, about correctly rewarding nodes
supported by MULTIPLE selected cores. The net per-corpus win/loss in Part F tracks the
F2_ONLY-vs-F1_ONLY count ratio exactly (MetaQA 619 vs 252 -> big win; WebQSP 134 vs 51 -> big win;
HotpotQA 7 vs 0 -> clean win; 2Wiki 4 vs 15 -> the one net loss, though still ns).

### Part A/B -- attribution ladder + per-query rescue classification

Pure tabulation of the 8-cell ladder (METIS/H4 core x no-halo/halo@0.5 x BASE/SAFE) -- every
number already sat in Stage 4's own `COVERAGE_<ds>.json`, zero new retrieval evaluation. Confirms
Stage 4 as a clean decomposition: core-swap effect (`HYPERGRAPH_BASE_EFFECT`, no halo) is sig on
MetaQA (+0.1276), WebQSP (+0.0416), HotpotQA (+0.016); adding halo@0.5 is sig positive with
**gained>0, lost=0 on every corpus x core cell except SQuAD's H4-vs-H4+halo** (ns, 1 gained) --
on the BASE (no-selector) lane the halo is a clean monotone add, exactly as designed.

Per-query classification (SAFE lane) into 8 exhaustive/mutually-exclusive buckets surfaces two
things not visible in the aggregate gate:
- **A real per-query REGRESSED population exists on every corpus** (H4 alone would have lost this
  query and the halo does NOT repair it): MetaQA 42/1998 (2.1%), 2Wiki 43/2000 (2.15%), MuSiQue
  34/2000 (1.7%), SQuAD 6/2000 (0.3%), WebQSP 17/1419 (1.2%), HotpotQA 15/2000 (0.75%). This does
  **not** contradict Stage 4's "zero significant regressions" (an aggregate McNemar call on net
  win/loss -- MetaQA alone nets 346 gained vs 42 lost) -- but the aggregate claim was never "zero
  queries regress," and this is the first place that distinction is made explicit.
- **STILL_UNSOLVED (neither core swap nor halo helps) is the dominant residual bucket on the two
  hardest corpora**: MetaQA 342/1998 (17.1%, mean 26.9 required nodes, 238/342 at hop3), WebQSP
  211/1419 (14.9%, mean 17.9 required nodes) -- far exceeding every other bucket combined on those
  corpora. This is the population Part H classifies next.

### Part E -- halo family / ranking-rule necessity

Stays inside the shipped per-core mechanism, varies only which edges populate the candidate pool
or how they're ranked (same beta=0.5 cap throughout). **No edge family is universally necessary
or harmful.** Dropping NERX from the halo (keep STRUCT+KNN) is never significant anywhere
(-0.0055..+0.002) -- its marginal HALO contribution is small everywhere, though NERX remains
critical at the CORE/partition-weighting level (a narrower claim, don't conflate the two).
**WebQSP is the one sharp, large outlier**: dropping STRUCT costs -0.0225 (sig, p=4.4e-07, 5
gained/37 lost); dropping KNN **gains +0.0303** (sig, p=2.6e-12, 44 gained/1 lost) -- semantic-kNN
edges are actively harmful in webqsp's halo, consistent with webqsp being the one KB-structured
corpus of the six (schema edges > embedding-similarity edges there). Random-vs-boundary-mass
ranking (same pool, same cap) is only significantly worse than boundary-mass ranking on MetaQA
(-0.009, sig); every other corpus is a wash.

### Part H -- residual headroom classification (best config = F2_SUM_NORMALIZED)

Classifies every node still missing under F2 into: H0 (sanity/bug, should be 0) / H1 (1-hop
candidate, cut by budget, would be rescued at 2x -- budget-limited) / H2 (1-hop candidate, still
cut even at 2x -- fundamentally low-ranked) / H3 (2-hop from a selected core, invisible to the
1-hop mechanism) / H4 (connected elsewhere in the corpus graph) / H5 (zero degree anywhere in
STRUCT u NERX u KNN -- no graph mechanism could ever reach it).

| corpus | missing nodes | H1 (budget) | H2 (low-ranked) | H3 (2-hop) | H4/H5 |
|---|---:|---:|---:|---:|---:|
| MetaQA | 4,440 | 13% | **65%** | 22% | 0% |
| WebQSP | 1,204 | 12% | **81%** | 6% | 0.5% |
| HotpotQA | 71 | 7% | **92%** | 1% | 0% |
| 2Wiki | 108 | 4% | **93%** | 3% | 1% |
| MuSiQue | 58 | 40% | **52%** | 9% | 0% |
| SQuAD | 16 | 6% | **81%** | 12% | 0% |

**H0=0 on all six (sanity passes). H2 dominates everywhere (52-93%): when F2 fails to fetch a
required node, it is overwhelmingly NOT because the budget was too small -- it's because the
ranking put that node far too low even at 2x budget.** H1 (genuinely budget-limited) is a
minority everywhere. H4/H5 are ~0 everywhere -- almost nothing is structurally stranded; residual
misses are close by, the mechanism just doesn't surface them.

### Part I -- 2-hop oracle (diagnostic only, no production routing)

Reuses Part H's classification directly: a query is "2-hop-oracle-solved" iff F2 already solves
it, or every one of its remaining missing nodes is bucket H3 (the only bucket a 2-hop mechanism
could reach that F2's 1-hop mechanism can't).

| corpus | F2 | 2-hop oracle ceiling | upside |
|---|---:|---:|---:|
| MetaQA | 0.8148 | 0.8183 | +0.0035 |
| WebQSP | 0.8548 | 0.8584 | +0.0036 |
| HotpotQA | 0.9665 | 0.9670 | +0.0005 |
| 2Wiki | 0.9475 | 0.9485 | +0.0010 |
| MuSiQue | 0.9735 | 0.9755 | +0.0020 |
| SQuAD | 0.9920 | 0.9930 | +0.0010 |

**Upside is negligible everywhere (+0.05 to +0.36pt). `PART_I_VERDICT: do not pursue 2-hop
routing`** -- consistent with H3's small share of an already-small residual population in Part H.

### Residual bottleneck (per the spec's own request to name one before opening a new axis)

**The ranking, not the budget or the candidate graph's reach, is the residual bottleneck.** H2
(low-ranked despite being a valid 1-hop candidate) dominates every corpus's remaining misses, the
2-hop oracle has almost nothing left to give, and H4/H5 (structurally absent) are ~0 -- reach is
not the problem. A plausible (not yet tested) unifying hypothesis: Part E's WebQSP-specific
finding (KNN edges actively harmful in the halo there) and Part H's universal H2-dominance may
share a root cause -- low-quality candidates crowding the ranking rather than being cleanly
outscored. This is exactly the kind of new ranking-signal question the spec's stop conditions
say not to chase without sign-off (no selector/ranking-feature search restart) -- named here as
the next axis, not acted on.

`PART_0B_PROMOTED = NONE`. `F2_SUM_NORMALIZED` is a validated candidate, not yet shipped -- next
gated on the same frozen-L2 question Phase C above already scoped and left open (see PROMOTION).

### Part 0C -- local residual 1-hop ranking phase (R0-R6)

Authorized explicitly, narrowly, with verbatim stop conditions: no parameter sweep, no learned
model, no new embeddings, no Modal, no 2-hop, no corpus-specific family rules. Reference =
F2_SUM_NORMALIZED. Four fixed rank constructions tested, one pass, gated exactly like every prior
stage. `scratchpad/_l1hu_r.py`; full numbers in TABLES.md Stage 6.

**R0 (parity).** This script's independent re-derivation of F2 is bit-identical to the shipped
Stage-5 F2 on all 6 corpora (0 diffs) -- everything built on top inherits that confidence.

**R1/R3 (enrichment diagnostics).** Both support-count and family-count correlate strongly and
mostly-monotonically with required-node likelihood on 5/6 corpora (dynamic range from 3x on
MetaQA's support-count up to 720x on WebQSP's) -- SQuAD is inconclusive (single-digit `req` counts
per bucket at its near-ceiling F2 rate) rather than contradictory. Both diagnostics passed, which
is what the spec required before building F3 and F5 at all.

**F3_SUPPORT_FIRST** (lexicographic support_count/F2/node-id) and **F5_FAMILY_SUPPORT**
(lexicographic family_count/support_count/F2) were built per R1/R3 passing. **F4_QUERY_RRF**
(RRF of F2's own within-pool rank and the *already-cached* node-level Dense+SPLADE query-relevance
rank `ret_rrf` -- no new encoder call, no new embeddings) was built per R2. All four cleared R5's
gate (zero significant regressions, macro delta positive) independently, so R4's own condition
(F3 AND F4 must each clear R5 before F6 is authorized) was comfortably satisfied, and
**F6_SUPPORT_QUERY_RRF** (RRF of F3's structural rank and the query-relevance rank) was built.

**R5 (gate, vs F2).**

| method | worst delta | macro delta | sig regressions | sig gains |
|---|---:|---:|---:|---:|
| F3_SUPPORT_FIRST | -0.0010 | +0.0091 | 0 | 1 |
| F4_QUERY_RRF | -0.0005 | +0.0064 | 0 | 4 |
| F5_FAMILY_SUPPORT | -0.0028 | +0.0014 | 0 | 0 |
| **F6_SUPPORT_QUERY_RRF** | **+0.0010** | **+0.0143** | **0** | **4** |

F6 wins on every axis: it is the *only* method with a non-negative delta on all 6 corpora, has the
best macro delta, and matches F4's sig-gain count while adding the single largest result in the
whole phase (WebQSP +0.0585 -- bigger than F3 alone's +0.0522 there, so the two structural/query
signals are complementary on WebQSP, not redundant). Significant gains: MuSiQue +0.0070, SQuAD
+0.0045, WebQSP +0.0585, HotpotQA +0.0120. MetaQA (+0.0010) and 2Wiki (+0.0025) are directionally
positive but not significant at this n.

**R6 (H2-targeted).** Re-derived inline (no re-run of Part H's expensive per-corpus scan) for the
exact nodes Part H classified as low-ranked-not-budget-limited. F6 has the best macro H2-query-
rescue rate (0.2986) and wins outright on 4/6 corpora; F5 edges it narrowly on MetaQA's and
MuSiQue's H2 subpopulation specifically, but both are tiny-n slices (27 and 329 queries) and F5
never posts a significant *aggregate* gain anywhere, so this reads as noise rather than a real
mechanism to chase. The underlying split is clean: WebQSP/MetaQA's H2 rescue leans structural
(F3-heavy), SQuAD/HotpotQA/MuSiQue's leans query-relevance (F4-heavy) -- F6's RRF construction
lands at or near the best of both per corpus rather than diluting either.

**Cumulative effect.** Extending the Production-METIS-baseline table from this phase's own
authorization message with F6 in place of F2:

| corpus | Production | H4+F2 | H4+F6 | net vs Production |
|---|---:|---:|---:|---:|
| MetaQA | 0.6612 | 0.8148 | 0.8158 | +0.1546 |
| WebQSP | 0.7646 | 0.8548 | 0.9133 | +0.1487 |
| HotpotQA | 0.9505 | 0.9665 | 0.9785 | +0.0280 |
| 2Wiki | 0.9435 | 0.9475 | 0.9500 | +0.0065 |
| MuSiQue | 0.9635 | 0.9735 | 0.9805 | +0.0170 |
| SQuAD | 0.9875 | 0.9920 | 0.9965 | +0.0090 |

WebQSP's net-vs-production gain grows from +0.0902 (H4+F2) to +0.1487 (H4+F6) -- roughly 39% of
WebQSP's entire cumulative gain over production METIS traces to this one local ranking-refinement
phase alone, on top of everything Stages 1-5 already contributed.

**Return values** (as requested): `BEST_ONEHOP_RANKER = F6_SUPPORT_QUERY_RRF`,
`METAQA_GAIN_OVER_F2 = +0.0010`, `WEBQSP_GAIN_OVER_F2 = +0.0585`, `WORST_DELTA_OVER_F2 = +0.0010`,
`MACRO_DELTA_OVER_F2 = +0.0143`, `H2_RESCUE_RATE` (F6, macro) `= 0.2986` (per-corpus: MetaQA
0.0182, 2Wiki 0.1020, MuSiQue 0.2963, SQuAD 0.6154, WebQSP 0.3886, HotpotQA 0.3710).

**Stop-condition compliance.** No parameter sweep (4 fixed constructions only), no learned model,
no new embeddings (reused cached `ret_rrf`), no Modal (100% local), no 2-hop, no corpus-specific
family rules (F5's family_count is a generic ranking signal, not a per-corpus drop rule; the
WebQSP "drop KNN" finding from Part E was **not** touched or shipped this phase, per the explicit
caution in the authorization). `PART_0C_PROMOTED = NONE` -- same frozen-L2 hold as Part 0B. Per
the spec's own "Then stop," this phase concludes here; no new optimization axis was opened.

## Stage 7 (Part 0D): L1 HIGH-CEILING RESIDUAL RANKING

Targeted follow-on with two explicit numeric targets: MetaQA >= 0.90 (from F6's 0.8158) and WebQSP
>= 0.95 (from F6's 0.9133). Framed by the user as WebQSP-realistic / MetaQA-uncertain, with an
explicit binding instruction that the very first thing to run is an oracle ceiling check with its
own STOP RULE, before any new ranker is built -- and an explicit instruction not to increase halo
size or start 2-hop; the lever, if any, had to be a better query-conditioned structural ranker on
the *existing* candidate pool.

**Part A -- oracle ceilings.** For the identical H4 cores/selected-50/FULL_C pool/K_q as F6, three
oracles: O1_FIXED_K (same pool, same K_q, required nodes ranked first -- the ranking-alone
ceiling), O2_FULL_1HOP (no K_q cutoff -- the reachability ceiling), O3_SELECTED_CORE_PLUS_1HOP
(core + every candidate, budget ignored). O2 and O3 were verified numerically identical on every
one of the 6 corpora, not merely assumed equal by definition -- confirming core members and FULL_C
candidates are exactly complementary in this codebase, so there is no third distinct ceiling here.
Both reduce to plain set arithmetic on quantities F6 already computes (`got_core`, `uniq_v`, K_q),
so `oracle()` is a separate, cheap, fast pass (0.2-1.6s per corpus, all 6 corpora in under 15s
total) run before any of the expensive G1-G6 machinery.

`METAQA_FIXED_K_CEILING = 0.9279` (>= .90, **PASS**). `WEBQSP_FIXED_K_CEILING = 0.9859` (>= .95,
**PASS**). Both STOP RULE checks pass -- ranking alone can in principle reach both targets, so
Parts B-I were authorized to proceed. The universal finding underneath both: **zero
budget-limited queries in all 6 corpora** -- O1 exactly equals O2 everywhere, meaning K_q never
binds anywhere, for any corpus. The entire F6-vs-ceiling gap, in every corpus, is a pure ranking
problem within the existing pool, not a budget or reach problem -- direct confirmation that
investing in a better ranker (not more exposure, not 2-hop) was the right call. MetaQA's own
shortfall is concentrated almost entirely in hop3 (O1: hop1=1.0000, hop2=0.9985, hop3=**0.7853**,
n=666 each) -- even a perfect one-hop ranker caps at 0.9279 for MetaQA; the remaining 0.0721 needs
deeper reach (2-hop), which the user explicitly said not to pursue in this phase.

**Parts B-D -- three new structural signals**, all reusing cached/static data, no new embeddings,
no Modal, no hypergraph rebuild: `G1_QUERY_WEIGHTED_SUPPORT` (raw union-graph neighbour support,
weighted by the supporting node's own cached query-relevance rank divided by its degree -- built
from `KS.keysets`/`KS._csr`, the same raw-edge primitive `_l1hu_h.py` already used for H-bucket
classification), `G2_CORE_RANK_SUPPORT` (F2's boundary-mass ingredient reweighted by the supporting
core's own selection rank, tracked in parallel with core identity during the same gather loop that
already builds F2), `G3_HYPER_COHERENCE` (H4_SPLIT_PRESERVE's own `SK`-famset hyperedge membership,
confirmed via `BUILD_MANIFEST.json` and a grep for what feeds the shipped `C1_HYPER_UNIVERSAL`
partition to be the correct source file -- query-weighted shared-hyperedge support, computed via a
two-stage source-side/candidate-side scatter-add over a reusable per-corpus scratch array with
touched-index-only resets, avoiding the ~9-billion-operation naive triple-nested enumeration a
candidate x hyperedge x member loop would require on MetaQA alone).

**Part E -- RRF combination**, gated on G1/G2/G3 showing independent value (they did, except G2):
`G4_STRUCT_RRF = RRF(G1,G2,G3)`, `G5_STRUCT_QUERY_RRF = RRF(G4, cached query-relevance rank)`.

**Part F -- universal STRUCT-vs-KNN precedence** (`SP0`-`SP3`), testing whether WebQSP's earlier
Part-E KNN-noise finding generalises as a query-local rule rather than a corpus-conditioned drop,
per the explicit prohibition on `if dataset == 'webqsp': remove_knn()`: `SP0` = `G5` unchanged
(alias, the no-tiering control). `SP1` tiers STRUCT-supported candidates first, tie-broken by G5.
`SP2` requires multi-core STRUCT support (own overall support count >= 2, not a STRUCT-only
recount -- a documented simplification) for the first tier. `SP3` demotes a KNN-only candidate
below the STRUCT-supported tier whenever its support count does not exceed the best STRUCT-
supported candidate's -- "comparable-or-less support cannot outrank STRUCT," operationalised
without introducing a new tunable. All four are the *same* rule on all 6 corpora; none inspects
`ds`.

**Gate (Part H), 5/6 corpora complete** (hotpotqa_clean's run was still in progress at write time;
its F6 is already 0.9785, near-saturated, so it is very unlikely to overturn the verdict below):

`G2_CORE_RANK_SUPPORT` is **refuted** -- significant regression on musique_clean/squad_clean/webqsp
(worst=-0.0564 on webqsp, macro=-0.0145), never a significant win anywhere. Every other method
(G1,G3,G4,G5/SP0,SP1,SP2,SP3) has zero significant regressions and a positive macro delta. `G5`/
`SP0` and `SP3` are the cleanest universal candidates -- worst_delta never negative on any corpus,
even non-significantly (+0.0005 and +0.0000). **`SP1` wins outright on both primary target
corpora** -- MetaQA 0.8669 (best of all 9 tested methods) and WebQSP 0.9626 (best of all 9,
p=3.09e-20) -- at the cost of one non-significant -0.0010 dip on squad_clean (already at 99.65%,
p=0.625). `PART_0D_BEST_UNIVERSAL_METHOD_5OF6 = SP1_STRUCT_PRECEDENCE`.

MetaQA: F6 0.8158 -> SP1 0.8669, **+0.0511**, closing 46% of the 0.1121 headroom to the 0.9279
ceiling -- real, significant progress (p=2.72e-25), target (0.90) not reached. WebQSP: F6 0.9133 ->
SP1 0.9626, **+0.0493**, closing 68% of the 0.0726 headroom to the 0.9859 ceiling -- **target
(0.95) reached.**

**Part G -- MetaQA hop-stratified rescue of nodes still missing under F6.** hop2's residual (809
missing nodes, 118 queries) is almost entirely rescuable by structure: `G3` alone converts 79.7% of
previously-failing hop2 queries into fully-solved queries, closing hop2 to within 0.036 of its own
0.9985 ceiling. hop3's residual (2606 missing nodes, 243 queries) is not: even the best method
converts only 8.2% of missing hop3 queries -- roughly a tenth of hop2's rescue rate. This is what
gates the 0.90 target (hop3 is 1/3 of the corpus) and is what motivated Part I.

**Part I -- feature-separability audit, MetaQA hop3 residual** (conditional trigger met: oracle
0.9279 >= .90 but best method 0.8669 falls meaningfully short; gold used only to measure
separability, no model trained). Pooled (query, candidate) rows over all 666 hop3 queries, label =
still-missing required node after core resolution -- 3,003 positive / 6,251,514 negative rows.
Twelve parameter-free features tested: F2 score, support count, `G1`/`G2`/`G3` scores, Dense rank,
SPLADE rank, RRF rank, candidate degree, family count, `has_struct`, best supporting-core rank.
**No feature separates strongly.** Best is support count (AUC=0.611, 1.79x top-decile enrichment)
-- weak by any standard threshold (0.5=chance, 0.7+=usually the floor for "acceptable"). F2 has the
best top-decile enrichment (2.17x) despite a lower AUC (0.597). Direct query relevance sits at pure
chance (Dense 0.508, RRF 0.503, SPLADE 0.498) -- quantitative confirmation of the motivating
hypothesis that direct lexical/dense relevance carries no signal for 3rd-hop answers. `has_struct`
and best-core-rank are mildly *anti*-correlated with the missing-node label (AUC<0.5): among the
hop3 residual specifically, structural support has by definition already been exhausted by the
time a query reaches this failure mode, so its presence stops being informative. Per the pre-
committed decision rule stated in the authorization ("if no parameter-free feature separates them
strongly, then stop L1 ranking work and hand this discrimination problem to learned L2"): **stop
further L1 ranking work on MetaQA's hop3 residual; hand it to learned L2.**

**Return values** (as requested): `METAQA_FIXED_K_CEILING = 0.9279`, `METAQA_1HOP_CEILING =
0.9279`, `WEBQSP_FIXED_K_CEILING = 0.9859`, `WEBQSP_1HOP_CEILING = 0.9859`,
`BEST_UNIVERSAL_METHOD_5OF6 = SP1_STRUCT_PRECEDENCE`, `METAQA_TARGET_090_MET = false`,
`WEBQSP_TARGET_095_MET = true`, `PART_I_VERDICT` = hand MetaQA hop3 to learned L2 (no strong
parameter-free feature found).

**Stop-condition compliance.** Oracle run first, STOP RULE evaluated and reported before any
ranker was built. No parameter sweep (G1-G5 and SP0-SP3 are fixed constructions; SP2/SP3's
thresholds are documented interpretations, not tuned). No learned model (Part I explicitly
diagnostic only). No new embeddings (`ret_dense`/`ret_splade`/`ret_rrf` all reused from cache). No
Modal (100% local, per-corpus profiled on 50 queries before any full run, exactly as instructed).
No 2-hop, no halo-size increase. No corpus-conditional rule -- SP1-SP3 are the same function on
every corpus, gated only by each candidate's own support/family/rank values. `PART_0D_PROMOTED =
NONE` -- same frozen-L2 hold as Stages 5-6; SP1/G5 are validated candidates on top of F6, not
shipped.

## PROMOTION

`PROMOTED = NONE`. `L1_FROZEN = NO`. Stages 1-4 are complete, gated and safe to cite as an L1
coverage result; Phase C's finding above is the reason promotion is still open, exactly as
[[l1-edge-closed-partition-weighting]] flagged in advance. Stage 5 adds one promotion candidate
(`F2_SUM_NORMALIZED` global halo) on top, gated the same way. Stage 6 adds a second, strictly
better candidate (`F6_SUPPORT_QUERY_RRF`, superseding F2 -- F6 is built directly on top of F2, not
a replacement mechanism) gated identically. Stage 7 adds a third, strictly better candidate
(`SP1_STRUCT_PRECEDENCE`, built directly on top of F6/G5 -- superseding F6 the same non-replacement
way) gated identically, 5/6 corpora (hotpotqa_clean pending, unlikely to change the verdict). All
three remain unshipped pending the same frozen-L2 question.

**Note on verdict labels.** The original spec's enumerated VERDICT option lists (the lettered
choices for `HYPERGRAPH_VERDICT` / `JOINT_VERDICT`) were pasted by the user in full earlier in
this session but are not verbatim-recoverable from the post-compaction transcript. Rather than
guess at letter/option codes that might not match what was actually offered, the verdicts above
are stated in full prose, backed by the numbers in this report and in `RETURNS.json`.

Artifacts: this file, `TABLES.md`, `RETURNS.json`, `hypergraph_build/`, `pin_retention/`,
`partitions/`, `halo/`, `interaction/`, `exposure/`, `l2_survival/` (empty, see above), `ceiling/`
(Part 0D: `O_*.json` oracles, `S_*.json` per-corpus G1-G6/SP0-3 results, `S_GATE.json`, `I_*.json`
Part I audits) under `results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_HYPERGRAPH_UNIVERSAL/`.
