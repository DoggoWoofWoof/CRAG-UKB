# L1 BACKWARD CAUSAL OPTIMIZATION PHASE — FINAL REPORT

**Verdict B — `GLOBAL_SET_OBJECTIVE_PARTIAL`.**  The joint-set view is not empty: it converts
**18.6 % of the MetaQA hop3 selection headroom** (0.2583 → 0.2778, net +13, McNemar p = 0.004), and
the mechanism is identified down to the single evidence family responsible for every changed query.
It fails the universality gate: the same mechanism **significantly regresses MuSiQue (−7, p = 0.039)
and HotpotQA (−11, p = 0.027)**.  `PROMOTED = NONE`.  `L1_FROZEN = NO`.

The backward chain also answers the question it was built to answer, and the answer is that neither
of the two candidate limiters is the limiter.  Ranking each partition independently and optimising
the whole P50 set jointly are both operating on a candidate universe that **does not contain 65 % of
the partitions the query needs**.

Artifacts: `TABLES.md` (T1–T13), `RETURNS.json`, `diag/{ledger,diag,meth,attr,lat}_*.json`,
`data/bc_<corpus>.npz`.  Code: `scratchpad/_l1bc_{core,ledger,methods,diag,attr,report}.py`.
Frozen replay parity **EXACT on every query of all six corpora** (metaqa 1998, webqsp 1419,
musique 2000, 2wiki 2000, hotpot 2000, squad 2000).  Gold is read for evaluation only.

> **CORRECTION, 2026-09-03 — reach bug, re-run.**  The `O1 <= O3` monotonicity check run at the start
> of the follow-up admission phase failed on metaqa, webqsp and 2wiki.  Cause: `base_rank` is already
> a *partition* ranking while every other array is a *node* ranking, and the identity test separating
> them (`arr is not z["base_rank"][qi]`) is always true because re-indexing builds a fresh view — so
> canonical partition ids were mapped through `hard[]` as document ids.  `reach` therefore held
> spurious partitions and dropped offered candidates (metaqa 251, webqsp 2476, 2wiki 2311,
> musique 243).  Fixed with an explicit flag plus an assertion that `reach` is a superset of
> everything offered; **all six substrates rebuilt and the chain is now monotone per query on all
> six.**  Every `O3`, `O4`, `CAND_UNIVERSE`, `B_CAPACITY`, `REACH` and `UNREACHABLE` figure in this
> report is the corrected one.  **No verdict changes**: `NOT_IN_CANDIDATE_UNIVERSE` remains the
> largest label on all six corpora, and the G4 result is untouched (it never used `reach`).  The two
> materially revised numbers are WebQSP `REACH` 0.1184 -> **0.0620** and WebQSP `UNREACHABLE`
> 44.5 % -> **26.1 %**.
>
> **`B_CAPACITY` was also two questions under one name.**  `O2-O1` (B headroom under the *current*
> universe) is 0.0010 on metaqa and 0.0000 elsewhere; `O4-O3` (B headroom *after* the universe is
> repaired) is 0.0771 on metaqa ALL and **0.1862 on metaqa hop3**.  Both were computed correctly;
> only the second was tabulated, under a name that reads like the first.  Full reconciliation and the
> B ladder: `../L1_ADMISSION/FINAL_REPORT.md` STEP 0.
>
> Superseded by `../L1_ADMISSION/` (verdict C, `CURRENT_ADMISSION_EVIDENCE_EXHAUSTED`).

---

## 1.  STEP 2 — the backward headroom ledger

Five mutually exclusive terms that sum to the whole loss `1 − O0`, on the chain
`O0 ≤ O1 ≤ O3 ≤ O4 ≤ O5` (T1, T2, T3).

| corpus | SELECTION `O1−O0` | CAND_UNIVERSE `O3−O1` | B_CAPACITY `O4−O3` | REACH `O5−O4` | P50_CAPACITY `1−O5` | total |
|---|---|---|---|---|---|---|
| metaqa | 0.0600 | 0.1467 | 0.0770 | 0.0421 | 0.0130 | 0.3388 |
| webqsp | 0.0240 | 0.1275 | 0.0170 | 0.0620 | 0.0049 | 0.2354 |
| musique | 0.0115 | 0.0250 | 0.0000 | 0.0000 | 0.0000 | 0.0365 |
| 2wiki | 0.0080 | 0.0400 | 0.0000 | 0.0085 | 0.0000 | 0.0565 |
| hotpot | 0.0140 | 0.0275 | 0.0000 | 0.0080 | 0.0000 | 0.0495 |
| squad | 0.0070 | 0.0055 | 0.0000 | 0.0000 | 0.0000 | 0.0125 |

Independent cross-check on both ends of the harness: `O0` reproduces the frozen SAFE values
0.7297 / 0.2583 / 0.6612 exactly, and `O1` on MetaQA hop3 is **0.3634** — bit-identical to the
frozen-pool oracle the candidate-generation phase measured by a completely different route.

As a share of MetaQA's own loss: CANDIDATE_UNIVERSE **43.3 %**, B_CAPACITY 22.7 %, SELECTION
**17.7 %**, REACH 14.5 %, P50_CAPACITY 3.8 %.  On hop3 the split is 37.4 / 24.9 / 14.2 / 18.6 / 4.9.

Three of the six decision questions are settled by this table alone.

* **`P50_CAPACITY_LIMITED = NO`.**  `1 − O5` is 0.0130 on MetaQA, 0.0049 on WebQSP, **0.0000 on the
  other four**.  P = 50 is enough slots almost everywhere.
* **`B_LIMITED = NO`** — *inside the frozen contract*.  This is the test the directive asked for
  explicitly, and the off-chain rung answers it: **`O2 − O1` is 0.0000 on five of six corpora and
  0.0010 on MetaQA.**  Given the frozen candidate universe, relaxing B from 6 to 50 — allowing the
  protected core itself to be evicted — buys one query in a thousand.  B only starts to bind once
  the universe is *widened*: over everything the machinery visited, `O4 − O3` is 0.0771 on MetaQA ALL and **0.1862 on hop3**.
  So B is a downstream consequence of the candidate universe, not an independent limiter.
* **`REACH_LIMITED = YES`, but it is a KB phenomenon.**  `O5 − O4` is 0.0620 on WebQSP (26.3 % of
  its whole loss) and 0.0421 on MetaQA, against ≤ 0.0095 on the four text corpora.  In the
  per-partition ledger `UNREACHABLE` is 26.1 % of WebQSP's missing partitions.

## 2.  STEP 1 — the per-partition loss ledger

One primary label per missing required partition, assigned by walking the chain and stopping at the
first stage that prevents recovery (T4; the secondary co-occurrence matrix is T5).

| corpus | missing/q | P50_CAP | UNREACH | **NOT_IN_CAND_UNIV** | B_CAP | POINTWISE | REDUND | SET_SEL | FUSION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa hop2 | 1.31 | 0.9 | 0.7 | **70.4** | 1.3 | 7.7 | 5.4 | 10.8 | 2.9 |
| metaqa hop3 | 7.42 | 7.2 | 2.9 | **63.2** | 4.5 | 6.7 | 6.2 | 7.0 | 2.2 |
| metaqa ALL | 2.91 | 6.2 | 2.4 | 64.5 | 4.0 | 6.9 | 6.0 | 7.6 | 2.3 |
| webqsp | 1.19 | 5.6 | 26.1 | 59.3 | 0.0 | 1.4 | 0.5 | 4.5 | 2.6 |
| musique | 0.04 | 0.0 | 0.0 | 68.0 | 0.0 | 1.3 | 4.0 | 20.0 | 6.7 |
| 2wiki | 0.06 | 0.0 | 14.5 | 69.2 | 0.0 | 4.3 | 2.6 | 6.8 | 2.6 |
| hotpot | 0.05 | 0.0 | 16.3 | 54.8 | 0.0 | 3.8 | 7.7 | 8.7 | 8.7 |
| squad | 0.01 | 0.0 | 0.0 | 44.0 | 0.0 | 0.0 | 4.0 | 32.0 | 20.0 |

`NOT_IN_CANDIDATE_UNIVERSE` is the largest label on all six corpora (44.0 – 70.4 %).  These
partitions were **touched by the bounded machinery and then never offered to the selector**.

Restricting to the residual the selection stage can actually address — the last four labels, 1 328
partitions on MetaQA — the split is POINTWISE 30.1 %, REDUNDANCY 26.4 %, SET_SELECTION 33.3 %,
FUSION 10.2 %.  So **59.7 % of the addressable residual is set-shaped and 40.3 % is not**, which is
what justified building the four set methods rather than another scalar.

## 3.  STEP 4 / STEP 5 — do the useful partitions have distinct evidence?

**Pareto (T6).**  `MISSING_GOLD_ON_PARETO_FRONT = 0.6671` on MetaQA (0.5667 – 0.7083 across
corpora): two thirds of the missing partitions are non-dominated, so no monotone rule over the ten
rank columns is *forbidden* from rescuing them.  The price is that the front is not selective —
`nuisance_on_front` is 0.5468 on MetaQA, the front holds 29.4 of 49.2 candidates, and the AUC
separating missing-required from nuisance by front rank is **0.5661**.  It is more selective on the
two large corpora (hotpot 0.7151, webqsp 0.6696) and least on squad (0.5330).  This is exactly why
`G1_PARETO` moves almost nothing.

**Novelty (T7) is dead, and unambiguously so.**  Against the current P50, the fraction of missing
required partitions carrying **zero** evidence atoms the final set does not already cover is
**1.0000 on five corpora and 0.9994 on MetaQA**.  `AUC_nov_atoms` is 0.4995 – 0.5002, macro
**0.4998 — exact chance**.  The 44 protected partitions already cover every evidence atom in the
query, so "what is new" carries no information at B = 6.  `G2_NOVELTY` consequently reproduces SAFE
**exactly on all six corpora** at DEPTH_MATCHED — gained 0 and lost 0 on every one, not a net-zero
average — and moves at all only at FULL_AVAILABLE, by one lost query on hotpot.  That is also the
harness gate working: a method carrying no information is required to fall through to the frozen
tie-break and reproduce SAFE, and it does.

`REDUNDANCY_LIMITED = NO` follows from this plus `O2 − O1 ≈ 0`: there is no slack in the set to
reclaim.

## 4.  STEP 10 / STEP 8 — exact P50, depth-matched

Frozen contract, B = 6, exactly 50 partitions out, McNemar against SAFE (T8a, T9).

| corpus | G0_SAFE | G1_PARETO | G2_NOVELTY | G3_SUBMODULAR | G4_ASSIGNMENT |
|---|---|---|---|---|---|
| metaqa hop2 | 0.7297 | 0.7312 (+1) | 0.7297 (0) | 0.7192 (−7, p=.092) | 0.7222 (−5, p=.302) |
| **metaqa hop3** | 0.2583 | 0.2598 (+1) | 0.2583 (0) | 0.2733 (+10, p=.031) **SIG** | **0.2778 (+13, p=.004) SIG** |
| metaqa ALL | 0.6612 | 0.6622 (+2) | 0.6612 (0) | 0.6627 (+3, p=.720) | 0.6652 (+8, p=.230) |
| webqsp | 0.7646 | 0.7660 (+2) | 0.7646 (0) | 0.7632 (−2) | 0.7632 (−2) |
| musique | 0.9635 | 0.9645 (+2) | 0.9635 (0) | 0.9600 (−7, p=.039) **SIG** | 0.9600 (−7, p=.039) **SIG** |
| 2wiki | 0.9435 | 0.9430 (−1) | 0.9435 (0) | 0.9435 (0) | 0.9435 (0) |
| hotpot | 0.9505 | 0.9510 (+1) | 0.9505 (0) | 0.9440 (−13, p=.007) **SIG** | 0.9450 (−11, p=.027) **SIG** |
| squad | 0.9875 | 0.9865 (−2) | 0.9875 (0) | 0.9870 (−1) | 0.9870 (−1) |

Read as a fraction of each corpus's own `O1 − O0` selection headroom, `G4_ASSIGNMENT` converts
**+18.6 % on MetaQA hop3** and **+6.7 % on MetaQA ALL**, against −10.2 % on MetaQA hop2, −5.8 %
WebQSP, −30.4 % MuSiQue, 0.0 % 2Wiki, −39.3 % Hotpot, −7.1 % SQuAD.  One corpus and one hop.

**The depth control held (T8b).**  Every method was run first at exactly the frozen candidate depth.
Widening to the whole visited universe (MetaQA 49.2 → 329.8 candidates/query, hotpot 55.4 → 915.0)
is worse in 13 of 24 corpus × method cells, tied in 10, and ahead in exactly one by +0.0005.  The
largest single depth penalty is hotpot `G3_SUBMODULAR` −0.0095, which takes it from net −13 to net
−32 against SAFE (p = 0.000).  The dilution law from
the calibration and triplet phases reproduces here; without this control the set methods would look
worse than they are and the comparison would be confounded with list length.

**B50 (T10) never rescues anything.**  Choosing all 50 freely is at best a tie and usually a
collapse: `G2_NOVELTY` falls to 0.2768 on MetaQA and `G3_SUBMODULAR` to 0.4259, because pure
coverage maximisation abandons the canonical core, which is where nearly all the accuracy lives.
`G4_ASSIGNMENT` at B50 (0.6642) does not beat itself at B6 (0.6652).  The directive's discriminating
case — "B6 no gain, B50 huge gain" — does not occur on any corpus.

## 5.  STEP 9 — REPAIR

`REPAIR@6` for SAFE is 0.0000 by construction: SAFE's best six *are* the selection, so anything
missing is by definition not among them.  `G4_ASSIGNMENT` reaches **0.0528** (ALL) and **0.0535**
(hop3); `G3_SUBMODULAR` 0.0471; `G1_PARETO` 0.0026 (T11).

Deeper in, the methods and SAFE converge: at k = 50 SAFE recovers 0.2696 and G4 0.2703.  **The set
methods reorder the head of the same list; they add no reach.**  That is the correct behaviour for a
selection experiment and it also bounds what any of them could ever do.

## 6.  STEP 11 — why each MetaQA query changed

34 queries changed under `G4_ASSIGNMENT` (21 gained, 13 lost).  Per-query attribution (T12):

| | mechanism | count |
|---|---|---|
| GAIN | `INDEPENDENT_EVIDENCE_COVERAGE` — won ≥ 1 atom outright in the assignment | 13 |
| GAIN | `ASSIGNMENT_DIVERSIFICATION` — won atoms of a family no SAFE pick won | 8 |
| GAIN | `DOMINATED_INCUMBENT_REMOVED` / `NOVELTY` / `ORDER_ONLY` | 0 / 0 / 0 |
| LOSS | `DISPLACED_REQUIRED_WON_NO_ATOM` — the lost partition won zero atoms | 13 |
| LOSS | `DISPLACED_BY_LARGER_MASS` / `DISPLACED_BY_ORDER` | 0 / 0 |

By hop: hop3 gains 16 / losses 3; hop2 gains 5 / losses 10.  The hop3 win and the hop2 loss are the
same mechanism pointed at different query populations.

**Every one of the 34 changed queries is driven by the `SRC_*` atom family** (33 by `SRC` alone, 1
by `CH + SRC`) — one atom per *protected-core source partition*, membership given by the actual
`Pi → Pj` graph transitions out of that partition, ranked by the offset score already computed
inside the frozen beam.  Nothing else in the evidence table moved a single query.

This is the same quantity the triplet phase isolated as `CORE_EXIT_OFFSET` and called "the one live
thread": the only evidence the node collapse genuinely destroys, because `np.maximum.at` maxes over
incoming edges without regard to where they came from.  There it was fused by equal RRF and gave
MetaQA hop3 +0.0030, not significant.  Placed in an assignment objective instead — where an atom
can be won by only one partition — the same evidence gives **+0.0195, p = 0.004**.  The evidence was
never the problem; the aggregation was.  It still does not generalise.

The loss side is equally clean and explains the failure: **all 13 losses are required partitions
that won no atom at all**, so the assignment could not see them.  A partition with no graph
transition out of the protected core is invisible to this objective, and on hop2 — where the frozen
selector is already right 73 % of the time — that costs more than the diversification gains.

## 7.  STEP 12 — latency

MetaQA, 120 queries, selection stage timed alone against the frozen L1 work it sits on (T13).

| stage | ms/query |
|---|---|
| L1 residual (frozen) | 0.301 |
| L1 bounded traversal (frozen) | 15.175 |
| L1 SAFE selector (frozen) | 0.089 |
| **L1 frozen total** | **15.565** |

`G4_ASSIGNMENT` **0.099 ms = 0.63 % of L1**; `G1_PARETO` 0.828 ms (5.05 %); `G3_SUBMODULAR`
1.057 ms (6.36 %); `G2_NOVELTY` 1.246 ms (7.41 %).  All four are inside the < 10 % target, and the
one method that actually wins is the cheapest of the four.  Latency is not a constraint here.

## 8.  DECISION RETURNS

| return | value |
|---|---|
| `P50_CAPACITY_LIMITED` | **NO** — `1−O5` = 0.0130 metaqa, 0.0000 on four corpora |
| `REACH_LIMITED` | **YES** — 0.0620 webqsp (26 % of its loss), 0.0421 metaqa; ≤ 0.0085 on text |
| `B_LIMITED` | **NO under the CURRENT universe** — `O2−O1` = 0.0010 metaqa, 0.0000 elsewhere; B50 never gains.  AFTER repair, `O4−O3` = 0.0771 metaqa ALL / **0.1862 hop3**, 0.0169 webqsp, 0.0000 on all four text corpora |
| `POINTWISE_RANKING_LIMITED` | **NO** — 6.9 % of missing, 30.1 % of the addressable residual; 66.7 % of missing partitions are Pareto non-dominated, so they are visible to some column |
| `REDUNDANCY_LIMITED` | **NO** — 6.0 % of missing; novelty AUC 0.4998 (chance), 99.94–100 % of missing carry zero new atoms |
| `GLOBAL_SET_SELECTION_LIMITED` | **NO** — set-shaped is 59.7 % of the addressable residual, but the best set method converts 18.6 % of the MetaQA hop3 headroom and 6.7 % of ALL, and regresses two corpora significantly |
| `PRIMARY_LIMITER` | **`NOT_IN_CANDIDATE_UNIVERSE`** — 64.5 % of missing partitions on MetaQA, largest label on all six corpora, 43.3 % of MetaQA's total loss |
| `BEST_SET_METHOD` | `G4_ASSIGNMENT` |
| `SAFE_METAQA_HOP2` / `BEST_METAQA_HOP2` | 0.7297 / 0.7222 (best under any method: 0.7312, `G1_PARETO`, ns) |
| `SAFE_METAQA_HOP3` / `BEST_METAQA_HOP3` | 0.2583 / **0.2778** (p = 0.0044, significant) |
| `REPAIR6_SAFE` / `REPAIR6_BEST` | 0.0000 / 0.0528 (hop3 0.0535) |
| `SET_SELECTION_OVERHEAD_MS` | 0.099 ms = 0.63 % of L1 (target < 10 %) |
| `SIGNIFICANT_REGRESSIONS` | musique G3 −7 p=.039, musique G4 −7 p=.039, hotpot G3 −13 p=.007, hotpot G4 −11 p=.027 |
| `SET_OBJECTIVE_VERDICT` | **B. `GLOBAL_SET_OBJECTIVE_PARTIAL`** |
| `PROMOTED` | **NONE** |
| `L1_FROZEN` | **NO** |

## 9.  What this phase settles

The two fundamentally different mathematical views have now both been tested on the same frozen
substrate, and neither is the limiter.  Ranking partitions independently leaves 59.7 % of the
addressable residual on the table; optimising the whole set jointly recovers a real, significant,
mechanistically explained slice of it on MetaQA hop3 and loses more than it wins on two other
corpora.  Both are working inside a candidate universe that omits 64 % of what the query needs, and
inside a selection headroom (`O1 − O0`) worth only 0.0600 on MetaQA and 0.0070 – 0.0240 everywhere
else.

That is a much stronger basis for a freeze decision than the previous phases gave, because it is now
quantified from the objective backwards rather than inferred from a series of failed scorers.  The
directive's own closing condition — "if joint optimization still cannot materially improve MetaQA
hop3 despite the oracle headroom" — was *not* met: it did improve it, significantly.  What failed
was universality, and the per-query attribution says precisely why: the winning evidence family
exists only where the graph has real `Pi → Pj` structure out of the protected core, and where it
does not, the objective is blind to partitions that carry no such transition.

The next lever that is *not* a selection lever, if L1 is not frozen here, is candidate admission —
getting the 64 % into the pool at all — which prior work has shown is solvable
(`U_PC5` raised the MetaQA hop3 pool oracle from 0.3634 to 0.6351) and whose conversion has so far
been zero.  This phase changes that picture in one specific way: it identifies an objective
(`G4_ASSIGNMENT` over `SRC_*` atoms) that *does* convert headroom once a partition is in the pool,
and it names the exact reason it is unsafe (blindness to partitions with no core-exit transition).
Those two facts together are testable against a widened pool; nothing else here is.
