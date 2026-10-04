# FULL_VISITED PARTITION CALIBRATION AUDIT -- FINAL REPORT

Corpora: MetaQA, WebQSP, 2Wiki, MuSiQue, HotpotQA, SQuAD.  Artifacts: `TABLES.md` (T1-T13), `RETURNS.json`, `diag/calib_<corpus>.json`, `diag/ledger_<corpus>.json`, `diag/trunc_<corpus>.json`.  Code: `scratchpad/_l1pc_core.py` (substrate), `_l1pc_run.py` (STEPS 1-8), `_l1pc_ledger.py` (STEP 7 ledger), `_l1pc_trunc.py` (STEP 7c depth-matched control), `_l1pc_report.py`, `_l1pc_final.py`.

**DECISION: C. PARAMETER_FREE_PARTITION_CALIBRATION_EXHAUSTED**  --  PROMOTED = NONE, L1_FROZEN = NO.

## What was held fixed

FINAL P = **50**, exact, on every row of every table. B = **6**, untouched (STEP 6). The frozen
bounded structural search: no new traversal, no new edge, no node scorer, no learned parameter,
no threshold grid, no dataset branch, no TEST split. The FULL_VISITED universe is a diagnostic
built by replaying the frozen search and reading the statistics it already accumulated over ALL
arrivals before its own prune -- the replay reproduces the frozen structural arrays bit-exactly
on every corpus (MetaQA 1998/1998, WebQSP 1419/1419, 2Wiki 2000/2000, MuSiQue 2000/2000, HotpotQA 2000/2000, SQuAD 2000/2000).
The only thing any variant changes is the ORDER of the structural partition ranking handed to
the unchanged F6 boundary competition.

## STEP 1 -- the challenger universe is enormous and the signal inside it is ~1%

Challengers are the FULL_VISITED partitions outside the canonical top-50; NEEDED are the gold
partitions among them (labels used for evaluation only, never as a feature).

On MetaQA the bounded search visits **76.2% of the entire
corpus partition set per query** -- 100.0% of
partitions are visited by some query and
95.8% are visited by more than
half of them. That leaves 261.2 challengers per query carrying
2.71 needed partitions: a prevalence of
**1.038%**. On hop3, where the headroom is, it is
2.565% -- about one needed row in
39.

Reachability confirms the contract audit: MetaQA 92.0%, WebQSP 36.0%, 2Wiki 37.4%, MuSiQue 85.6%, HotpotQA 55.1%, SQuAD 61.5% of the gold partitions outside the canonical top-50 are inside this universe.
**FULL_VISITED_SIGNAL_SATURATION = YES.**

## STEP 2 -- the composite is worse at B = 6 than two of its own four channels

Seventeen recorded quantities plus five coherence statistics were scored as challenger orderings.
Discrimination is real but weak: the best AUC on MetaQA hop3 is
0.6741. What matters is the top of the list, because
only six partitions can ever be swapped in:

- `BEST_STRUCT_NODE_SCORE` -- **NEEDED_RECALL@6 = 0.0899** (the best of any single signal)
- `S4_RANK_FULL_VISITED`, the composite actually in the contract -- **0.0752**
- attainable ceiling at that budget -- **0.4116**

So the frozen composite reaches 18.3% of what a perfect ordering could reach, and the best single
signal reaches 21.8%. The composite is beaten by its own parts:
S4 is an equal-weight RRF over four channels, and on this universe they split cleanly --
  first arrival 0.0897; node count 0.0497; minimum hop 0.0352; best node score 0.0899.
Two strong channels are averaged against two weak ones, and the RRF gives away +0.0147 at B = 6 for it.

One naming point the directive's list makes worth stating: on this substrate a node enters the
frontier at exactly one hop, so distinct parent STATES, distinct parent NODES and the arrival
count coincide -- `DISTINCT_PARENT_COUNT` and a path count are the SAME recorded quantity here,
not two independent ones. The independent parent-side axes are `RANK_WEIGHTED_PARENT_SUPPORT`
(0.5871) and `BEST_PARENT_SCORE` (0.5452), both weaker than the node-score channel.

The retrieval channels are near chance on this population (Dense 0.5319, SPLADE 0.5101, fused 0.5217), which is expected: these are exactly the
partitions dense retrieval failed to surface.

## STEP 3 -- exposure bias is real, is CAUSED by opening the universe, and is not the fault

Correlating static corpus-side geometry with how well the raw structural ranking places a
partition shows strong exposure dominance on MetaQA: expected visitation
+0.775, P(visited) +0.691, boundary degree +0.570, partition size only +0.211.

The control that matters is the same measurement on the FROZEN M64 universe, where the identical
correlations are pexp -0.093, pvis -0.181, pbdeg -0.096. The dominance is therefore
**created by widening the universe**, not inherited from the frozen ranking (delta +0.868 on expected visitation,
+0.872 on P(visited)).
**DEGREE_EXPOSURE_BIAS = YES on the saturated corpora, and it TRACKS universe saturation (strong on MetaQA/SQuAD/2Wiki/MuSiQue, near zero on WebQSP, negative on HotpotQA); but correcting for it is harmful, not helpful.**

But the diagnosis that would follow from it is wrong, and STEP 3's needed-vs-nuisance split says
so before any normalisation is built. On MetaQA the NEEDED challengers are themselves the
high-degree partitions: pbdeg x1.42, pdeg x1.30, pnodedeg x1.30, padj x1.26, while size is flat (x1.00).
A gold partition that a multi-hop chain has to traverse *is* a well-connected one. Dividing the
evidence by degree therefore penalises the target, not the distractor.

Across all six corpora the bias is not a constant, and both halves of that matter:

- The exposure correlation **tracks universe saturation** (Spearman +0.600 across the six): MetaQA sat 0.76 -> rho +0.775, WebQSP sat 0.05 -> rho +0.196, 2Wiki sat 0.25 -> rho +0.630, MuSiQue sat 0.82 -> rho +0.525, HotpotQA sat 0.12 -> rho -0.226, SQuAD sat 0.61 -> rho +0.697. Visiting most of a corpus is what makes reachability dominate the ranking; visiting 5% of it
  does not.
- Whether the needed partitions are themselves hubs is corpus-specific: boundary-degree ratio MetaQA x1.42, WebQSP x1.84, 2Wiki x0.98, MuSiQue x0.98, HotpotQA x0.90, SQuAD x1.00. It is above 1 exactly where the multi-hop chains are (MetaQA, WebQSP), so degree
  normalisation is actively harmful there and merely inert elsewhere.

The honest answer to the directive's question is therefore: the bias is real and is caused by the
wider universe, but it is not the failure mechanism, and correcting it makes things worse.

## STEP 4 -- the normalisation family confirms it (nothing beats N0_RAW)

One division each, no exponent, no weight, no threshold, no dataset branch:

- `N0_RAW` -- AUC 0.6738, NEEDED_RECALL@6 0.0752
- `N1_DEGREE_NORMALIZED` -- AUC 0.6497, NEEDED_RECALL@6 0.0695
- `N2_SIZE_NORMALIZED` -- AUC 0.6727, NEEDED_RECALL@6 0.0756
- `N3_EXPOSURE_NORMALIZED` -- AUC 0.6639, NEEDED_RECALL@6 0.0726

Every normalisation is at or below the raw support on AUC. Only `N2_SIZE_NORMALIZED` edges ahead
on NEEDED_RECALL@6, by +0.0004, and
it loses that back through F6: on MetaQA all three normalisations are significantly WORSE than
the frozen baseline (net N1 -16, N2 -16, N3 -16) and none is better than raw on any corpus. The exposure correlation is real and the
correction for it does not pay -- consistent only because, where the bias is strongest, the bias
and the signal are the same quantity.

## STEP 5 -- coherence, and three direct orderings

Coherence statistics (seeds per node, arrivals per node, best-node share of the partition's RRF
mass, admitted fraction, seeds x best node score) were audited on the same population. Only one
separates at all: `COH_SEED_TIMES_SDIR`, AUC 0.6741 -- the best
AUC in the whole table -- but its NEEDED_RECALL@6 is 0.0877, below the best single signal. Diffuse
support is not the discriminator: the normalised coherence ratios (`COH_SEEDS_PER_NODE` 0.4128, `COH_BEST_NODE_RRF_SHARE` 0.4502) are BELOW chance,
i.e. needed partitions have *less* concentrated support than nuisance ones.

Three direct orderings were then built from what STEP 2 found -- D1 best node score, D2 seeds x
best node score, D3 the S4 RRF with its two weak channels dropped. These were specified after
reading the table, so they are diagnostics, and they were held to the full six-corpus promotion
gate rather than to the MetaQA number that motivated them.

## STEP 7 -- through the unchanged F6, and why the net is negative

Every ordering was fed into the same boundary competition at the same B = 6. On MetaQA:

- frozen M64 + F6 (baseline): ALL 0.6612, hop3 0.2583
- FULL_VISITED + `N0_RAW`: ALL 0.6537, hop3 0.2508, net -15 (p = 0.00027, SIG)
- FULL_VISITED + `N1_DEGREE_NORMALIZED`: ALL 0.6532, hop3 0.2492, net -16 (p = 0.00014, SIG)
- FULL_VISITED + `N2_SIZE_NORMALIZED`: ALL 0.6532, hop3 0.2492, net -16 (p = 0.00014, SIG)
- FULL_VISITED + `N3_EXPOSURE_NORMALIZED`: ALL 0.6532, hop3 0.2508, net -16 (p = 0.00014, SIG)
- FULL_VISITED + `D1_BEST_NODE_SCORE`: ALL 0.6542, hop3 0.2523, net -14 (p = 0.00131, SIG)
- FULL_VISITED + `D2_SEED_TIMES_SDIR`: ALL 0.6542, hop3 0.2553, net -14 (p = 0.00258, SIG)
- FULL_VISITED + `D3_S4_TWO_CHANNEL`: ALL 0.6542, hop3 0.2523, net -14 (p = 0.00131, SIG)

The ledger (T11) opens the selection and shows the mechanism, which recall alone cannot:

- MetaQA hop3 has **5183 needed partitions across 666 queries** (7.78 per query) competing for **3996 swap slots** (1.30 needed per available slot).
- The frozen baseline already spends **93.9% of those slots on non-gold partitions** and lands 203 novel golds.
- Opening the universe makes that *worse*, not better: N0_RAW lands 146 novel golds (-57) and spends 95.7% of its slots on nuisance.
- Of the 5 queries it loses, **5 are lost because a needed partition the baseline's own six slots already held fell out** -- displaced by a challenger that exists
  only in the enlarged universe. Only 0 of those was a canonical boundary incumbent.

This is dilution, measured directly, and it holds on every corpus with a ledger: MetaQA 5/5; WebQSP 2/2; 2Wiki 6/6; MuSiQue 7/7; HotpotQA 4/4; SQuAD 2/2 of lost queries lost this way.

The best ordering in the whole family, `D2_SEED_TIMES_SDIR`, moves MetaQA hop3 from 0.2583 (frozen) to 0.2553 -- i.e. it recovers part of the damage that
opening the universe does, and still ends -0.0030 BELOW the frozen baseline, against a FULL_VISITED
B = 6 oracle of 0.5916.

## STEP 7c -- the depth-matched control: it is the LIST SIZE, not the ORDER

STEP 7 changes two things at once. The calibrated ordering prefers different partitions, AND it
hands F6 a far longer candidate list -- MetaQA 48 -> 306, WebQSP 37 -> 417, 2Wiki 38 -> 167, MuSiQue 28 -> 112, HotpotQA 46 -> 585, SQuAD 16 -> 117 partitions per query. F6 fuses structural rank with canonical and retrieval rank, so a
nuisance partition with a mediocre structural rank but a strong canonical rank can outscore a
needed one that the shorter list would have protected.

Cutting each calibrated ordering, per query, to exactly the length the frozen ordering produced
for that same query separates the two. K is read off the frozen ordering; it is not tuned.

- MetaQA (hop3): frozen 0.2583, best depth-matched 0.2583 (`D3_S4_TWO_CHANNEL`, net -1, p = 1) -- the same ordering at full depth was net -14
- WebQSP (ALL): frozen 0.7646, best depth-matched 0.7689 (`N0_RAW`, net +6, p = 0.109) -- the same ordering at full depth was net +6
- 2Wiki (ALL): frozen 0.9435, best depth-matched 0.9430 (`N3_EXPOSURE_NORMALIZED`, net -1, p = 1) -- the same ordering at full depth was net -3
- MuSiQue (ALL): frozen 0.9635, best depth-matched 0.9630 (`N3_EXPOSURE_NORMALIZED`, net -1, p = 1) -- the same ordering at full depth was net -1
- HotpotQA (ALL): frozen 0.9505, best depth-matched 0.9510 (`D3_S4_TWO_CHANNEL`, net +1, p = 1) -- the same ordering at full depth was net +0
- SQuAD (ALL): frozen 0.9875, best depth-matched 0.9875 (`N3_EXPOSURE_NORMALIZED`, net +0, p = 1) -- the same ordering at full depth was net +1

**Depth-matching removes the harm on the corpus that had it, and produces no gain anywhere.**
On MetaQA, where all 7 full-depth orderings were significantly worse, not one depth-matched ordering is: the worst is
net -6 and the best lands
at hop3 0.2583, exactly the frozen 0.2583 -- it ties, it
does not beat.

The significance ledger across the whole grid is the cleanest statement of the result: at full
depth 7 of 42 corpus x ordering cells are significant, all of
them MetaQA and all of them WORSE; depth-matched, 1 of 42 is (MuSiQue D2_SEED_TIMES_SDIR), and it is also worse. In neither condition is a single significant IMPROVEMENT produced, on any corpus.

The dose-response confirms the mechanism rather than assuming it. Feeding F6 the SAME `N0_RAW`
ordering cut at increasing depth on MetaQA:

-  1xK:   47.5 partitions/q -> hop3 0.2553, net -3 (p = 0.607)
-  2xK:   94.9 partitions/q -> hop3 0.2538, net -9 (p = 0.0225, SIG)
-  4xK:  188.1 partitions/q -> hop3 0.2508, net -13 (p = 0.00235, SIG)
- FULL:  305.5 partitions/q -> hop3 0.2508, net -15 (p = 0.00027, SIG)

Monotone in list length. A mis-scoring mechanism would not produce that curve; dilution does.

## STEP 8 -- exact P50, all six corpora

| corpus | frozen M64 + F6 | best FULL_VISITED ordering | delta | net | significant? |
|---|---|---|---|---|---|
| MetaQA | 0.6612 | 0.6542 (`D3_S4_TWO_CHANNEL`) | -0.0070 | -14 | **worse, SIG** |
| WebQSP | 0.7646 | 0.7689 (`N2_SIZE_NORMALIZED`) | +0.0043 | +6 | no |
| 2Wiki | 0.9435 | 0.9420 (`N3_EXPOSURE_NORMALIZED`) | -0.0015 | -3 | no |
| MuSiQue | 0.9635 | 0.9630 (`N3_EXPOSURE_NORMALIZED`) | -0.0005 | -1 | no |
| HotpotQA | 0.9505 | 0.9505 (`N1_DEGREE_NORMALIZED`) | +0.0000 | +0 | no |
| SQuAD | 0.9875 | 0.9880 (`N3_EXPOSURE_NORMALIZED`) | +0.0005 | +1 | no |

Two corpora move numerically upward (WebQSP +0.0043, SQuAD +0.0005), HotpotQA exactly ties, and
neither movement is significant -- the larger, WebQSP net +6, is p = 0.109. WebQSP is also the
least saturated corpus in the set (universe saturation 0.053 against MetaQA's 0.762), which is consistent with dilution being
the mechanism, but it is a trend and not a result.

The promotion gate requires a meaningful MetaQA hop3 improvement AND no significant cross-corpus
regression AND exact P50. Every MetaQA hop3 number in the phase is at or below the frozen
0.2583, so the gate is failed by all seven orderings at both depths, and
`ANY_ORDERING_PASSES_PROMOTION_GATE = False`.
Exact P50 held on every row: every configuration returns exactly 50 partitions.

## Returns

```
FULL_VISITED_SIGNAL_SATURATION     = YES
DEGREE_EXPOSURE_BIAS               = YES on the saturated corpora, and it TRACKS universe saturation (strong on MetaQA/SQuAD/2Wiki/MuSiQue, near zero on WebQSP, negative on HotpotQA); but correcting for it is harmful, not helpful
BEST_NEEDED_PARTITION_SIGNAL       = BEST_STRUCT_NODE_SCORE
NEEDED_RECALL_B6_BASE              = 0.0752
NEEDED_RECALL_B6_BEST              = 0.0899
NEEDED_RECALL_B6_CEILING           = 0.4116
EXACT_P50_METAQA_HOP3              = 0.2583 frozen  ->  0.2553 best  (-0.003)
EXACT_P50_ALL                      = MetaQA 0.6612 -> 0.6542, WebQSP 0.7646 -> 0.7689, 2Wiki 0.9435 -> 0.9420, MuSiQue 0.9635 -> 0.9630, HotpotQA 0.9505 -> 0.9505, SQuAD 0.9875 -> 0.9880
VERDICT                            = C. PARAMETER_FREE_PARTITION_CALIBRATION_EXHAUSTED
PROMOTED                           = NONE
B_CHANGED                          = False
L1_FROZEN                          = NO
```

## What this closes, and what it does not

**Closed.** Partition-level evidence calibration, in the parameter-free form the directive
specifies, is exhausted. The premise it was built on -- that FULL_VISITED's 0.9195 novel-gold
reachability and 0.5916 B = 6 oracle are unexploited because the *evidence is mis-scored* -- is
refuted for this family. Re-scoring, re-normalising and re-composing the same evidence moves
MetaQA hop3 across a range of 0.2492 to 0.2553, entirely below the frozen 0.2583.

**Not closed, and now sharper.** The binding constraint is the ratio the ledger names: on MetaQA
hop3 there are 7.78 needed partitions per query and B = 6 slots, and 93.9% of the slots that exist
are already spent on non-gold partitions by the frozen selector itself. A better ordering of a
261-challenger universe with 1% prevalence cannot fix that; a ranking that is right 9% of the
time at B = 6 spends most of a 6-slot budget on distractors no matter how it is normalised.

The depth-matched control (STEP 7c) is what makes this a closure rather than a null result. It
separates the two things STEP 7 confounds and finds them BOTH negative: the harm belongs entirely
to candidate-list inflation (removing it removes every significant result on MetaQA, the only
corpus that had one, and the dose-response is monotone in list length), and once
that is removed the calibration axis itself is worth exactly zero -- the best depth-matched
ordering ties the frozen one and never beats it. Neither half of the hypothesis survives.

STEP 6's condition is therefore now testable and was deliberately not tested here: B stayed at 6
throughout, and no ranking materially improved B = 6, so the directive's own gate for revisiting
B = 12 / 20 is not met by this phase. What the contract audit already showed is that B is dormant
under frozen evidence (exactly +0.0000 on four of six corpora) and only becomes worth +0.1081 on
MetaQA hop3 once the evidence universe is opened -- and this phase shows opening it costs more
through the frozen selector than it pays. The two moves are coupled and neither works alone.
