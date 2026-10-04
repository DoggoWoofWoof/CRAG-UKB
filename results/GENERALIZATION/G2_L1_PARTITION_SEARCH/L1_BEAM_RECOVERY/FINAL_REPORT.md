# L1 BEAM RECOVERY -- parameter-free structural beam recovery

Question: can future structural evidence be exploited *before* the hop-2 pruning decision, so
that useful nodes move toward the top of the beam while L1 still emits EXACTLY 50 partitions?

**Answer: no.** The prune is not premature. Position 2 really is where the gold dies -- 64% of
all structural gold loss happens there -- but the loss is *conserved*, not removable: an
oracle-free control that deletes the position-2 and position-3 prunes outright recovers the
gold into the beam and then loses more of it than the frozen beam did, one stage further down.
Every lookahead variant is strictly worse end-to-end. MetaQA hop3 does not move off its frozen
value of 0.2583 under any of the ten policies.

All numbers below are read out of `diag/*.json` by `scratchpad/_l1bm_report.py`; the tables
live in [TABLES.md](TABLES.md).

## Returns

| flag | value |
|---|---|
| `PREMATURE_PRUNING_CONFIRMED` | DISCOVERY_ONLY -- delaying the prune raises gold DISCOVERY 0.3240 -> 0.8547 but LOWERS what S4 reads 0.1334 -> 0.1139 and lowers exact-P50 ALL 0.6612 -> 0.6582; the answer to the phase question is therefore NO |
| `BEST_BEAM_METHOD` | B0_GLOBAL (the frozen static-score beam) -- nothing cleared promotion |
| `POSITION2_GOLD_SURVIVAL_GAIN` | +0.0000 for the promoted method; gold-parent survival stays 0.1521. The largest gain any bounded policy achieved is +0.0510 (0.1521 -> 0.2031, L1_MAX_FUTURE), but it pays for that by losing gold CANDIDATES (-0.1737, 0.3756 -> 0.2019) and it does not convert -- see HOP3_GOLD_NODE_RECOVERY_GAIN. |
| `HOP3_GOLD_NODE_RECOVERY_GAIN` | +0 gold nodes read by S4 (1428 -> 1428 of 9718, +0.0000) for the promoted method. The position-2 survival maximiser L1_MAX_FUTURE gives -97 gold nodes read by S4 (1428 -> 1331 of 9718, -0.0099); the oracle-free delayed-prune control D0_DELAYED_PRUNE, which removes the position-2 and position-3 prunes entirely, gives +32 gold nodes read by S4 (1428 -> 1460 of 9718, +0.0033). |
| `EXACT_P50_METAQA_HOP2` | 0.7297 |
| `EXACT_P50_METAQA_HOP3` | 0.2583 |
| `ONLINE_EDGES` | 1,506 graph-edge inspections/query (1,506 beam + 0 lookahead), 1,241 distinct nodes scored, emitting EXACTLY 50 partitions = 5,036 nodes downstream (unchanged). The one-step lookahead costs 6,479 edges/query (4.3x) for the same 50-partition output -- a latency cost, not a compression cheat. |
| `LATENCY` | 22.8 ms/query structural routing for the promoted method (clean benchmark: 400 queries, median of 3 repeats, nothing else on the CPU). Rejected alternatives: L1_MAX_FUTURE 5.83x, L2_TOP2_FUTURE 6.41x, D0_DELAYED_PRUNE 3.81x, B1_PARENT_DIVERSE 1.20x. |

## 1. The replay is bit-exact, so every difference is the policy

The frozen beam was re-implemented from `_ta_prepartition.expand_dir` and replayed on the same
RRF seeds, the same residual query vector and the same CSR topology. Parity is exact on all
four frozen per-query fields *and* on the S4 partition ranking, on all six corpora:

| dataset | queries | all 4 frozen fields | S4 ranking | verdict |
|---|---|---|---|---|
| metaqa | 1998 | 1998/1998 | 1998/1998 | EXACT |
| webqsp | 1419 | 1419/1419 | 1419/1419 | EXACT |
| 2wiki_clean | 2000 | 2000/2000 | 2000/2000 | EXACT |
| musique_clean | 2000 | 2000/2000 | 2000/2000 | EXACT |
| hotpotqa_clean | 2000 | 2000/2000 | 2000/2000 | EXACT |
| squad_clean | 2000 | 2000/2000 | 2000/2000 | EXACT |

Two details were load-bearing. The frozen expansion scores each frontier node with its own
mat-vec rather than one batched product -- BLAS blocking changes the last bits and that is
enough to reorder a tie. And dedup is `np.unique` + `np.maximum.at`, so ties break by ascending
node id. Reproducing both is what closed the 1997/1998 gap left by the previous phase.

## 2. Where the beam actually prunes (the premise arithmetic was wrong)

The directive assumed roughly 2,560 candidates at position 2, from `64 frontier x ~40 edges`.
That figure came from an estimate in the previous phase report, not a measurement. Measured on
the exact replay:

| path position | frontier in | distinct fresh candidates | kept (beam) | prune ratio |
|---|---|---|---|---|
| 1 | 5.0 | 25.2 | 21.5 | 1.2x |
| 2 | 21.5 | 438.8 | 61.8 | 7.1x |
| 3 | 61.8 | 778.3 | 63.8 | 12.2x |

Position 1 does not prune at all. Position 2 prunes about 7x and position 3 about 12x, so by
*ratio* position 3 is the harsher cut. Mean legal out-degree is around 17, not 40.

## 3. The transition table -- and conservation of loss

Candidate survival at one hop is not the deliverable, so the whole pipeline was instrumented
per required gold node. `alive after pK` is *won or still winnable*: the node has already been
found, or some survivor of that prune can still reach it in the hops that remain. Both halves
are needed -- a node found early is excluded from every later candidate set, so a purely
forward-looking test is not monotone.

On the frozen beam, of the 14,878 required gold
nodes over 1,998 MetaQA queries:

| stage | fraction still in play | share of structural loss |
|---|---|---|
| REACHABLE | 0.8556 |  |
| ALIVE_p1 | 0.8548 | 0.1% |
| ALIVE_p2 | 0.3954 | 63.6% |
| ALIVE_p3 | 0.3240 | 9.9% |
| NODE_added256 | 0.2902 | 4.7% |
| NODE_read_S4 | 0.1334 | 21.7% |

So the premise was right about *where*: position 2 is the dominant cut by a wide margin, and
position 3 is minor. What the premise got wrong is that this makes the cut premature.

`D0_DELAYED_PRUNE` settles that. It is oracle-free -- it simply does not prune at positions 2
and 3, so `alive p2` and `alive p3` are pinned to the reachable set by construction. It
raises gold discovery from 0.3240 to 0.8547. And then:

| | frozen beam | delayed prune |
|---|---|---|
| gold alive after p2 | 0.3954 | 0.8547 |
| gold in `added(256)` | 0.2902 | 0.2335 |
| gold read by `S4(64)` | 0.1334 | 0.1139 |
| **total structural loss** | **0.7222** | **0.7417** |
| EXACT P50 ALL | 0.6612 | 0.6582 |

The loss is not removed, it is relocated: 84% of it moves onto the static-score-ordered
`added(256)` cut, where the same score now has to discriminate a pool an order of magnitude
larger -- and does it worse. On hop2 the relocation is brutal: `added(256)` holds
0.4330 of gold on the frozen beam and only 0.1072 with the prunes removed.

The small beam was doing useful work. It kept the pool small enough that a score which is at
chance before the terminal hop did not have to discriminate among thousands of candidates.

## 4. Why one-step lookahead is strictly harmful

`L1_MAX_FUTURE` and `L2_TOP2_FUTURE` compute, for every position-2 candidate and before the
prune, the best (or mean-of-best-two) score of its own legal fresh children, using the exact
existing score semantics. No learned weight, no threshold, no lambda.

They do raise gold-*parent* survival: 0.1521 -> 0.2031. They pay for it by losing
gold *candidates*: 0.3756 -> 0.2019. That is the trade in one
line: a node that *leads toward* gold is promoted over a node that *is* gold, and the gold
candidate's percentile rank degrades from 0.3376
to 0.6223.

The hop2 rows expose a second, sharper mechanism. Lookahead puts *more* hop2 gold into
`added(256)` than the frozen beam (0.4470 vs 0.4330) and S4 then reads less than half as much
(0.0490 vs 0.1134). Admission and aggregation use different keys.
Nodes admitted on the strength of their children rank badly under the static score S4 actually
reads, so improving admission under one key degrades the read under the other. Any lookahead
that is not accompanied by a matching change to S4 is fighting itself -- and S4 is frozen.

## 5. The lexicographic pair are identity controls, not a second fusion mechanism

`L3_FUTURE_THEN_CURRENT` and `L4_CURRENT_THEN_FUTURE` were specified as a strict lexicographic
pair with no weighted sum. A lexicographic rule can only differ from its own primary key where
that primary key ties, so the honest report is the measured tie mass:

| quantity | value |
|---|---|
| position-2 candidates scored | 870,793 |
| tied in the CURRENT (static cosine) key | 10 (0.0000%) |
| tied in the FUTURE key | 17,220 (1.98%) |
| of which: no legal fresh child (sentinel) | 17,420 (2.00%) |

- `L3_FUTURE_THEN_CURRENT` produces an **identical node list to `L1_MAX_FUTURE`** on 1983/1998 queries (99.25%).
- `L4_CURRENT_THEN_FUTURE` produces an **identical node list to `M0_BASELINE`** on 1998/1998 queries (100.00%).

Ten ties in 870,793 continuous cosine scores. `L4` is therefore the frozen beam exactly, on
every query, and `L3` is `L1` on all but 15 queries -- and the residual tie mass in the future
key is almost entirely the sentinel for candidates with no legal fresh child, which is a
structural fact rather than evidence. **These two are not two fusion mechanisms.** They are
controls that prove continuous cosine keys leave no tie mass for lexicographic secondary
evidence to act on. That is a clean negative result and should be written up as one.

## 6. Composed path geometry and stratified beams

`C1_ENDPOINT_DISPLACEMENT` and `C2_NORMALIZED_EDGE_SUM` both score the composed path rather
than the last edge. Both are worse than the frozen beam at every stage of the funnel
(gold alive after p2 0.3481 and 0.3457 vs 0.3954) and both lose end-to-end. No further geometric
variants were tested.

`B1_PARENT_DIVERSE` (one slot per parent first, then fill by score) is the one policy that
improves *every* stage of the funnel and end-to-end, at the same total beam of 64:

| stage | frozen | B1_PARENT_DIVERSE | delta |
|---|---|---|---|
| REACHABLE | 0.8556 | 0.8556 | +0.0000 |
| ALIVE_p1 | 0.8548 | 0.8548 | +0.0000 |
| ALIVE_p2 | 0.3954 | 0.4177 | +0.0223 |
| ALIVE_p3 | 0.3240 | 0.3310 | +0.0070 |
| NODE_added256 | 0.2902 | 0.2973 | +0.0071 |
| NODE_read_S4 | 0.1334 | 0.1400 | +0.0066 |
| PART_in_S4 | 0.4018 | 0.4250 | +0.0232 |
| EXACT P50 ALL | 0.6612 | 0.6622 | +0.0010 |

That is a net of +2 queries out of 1,998 (McNemar p = 0.625), and hop3 is unmoved at 0.2583.
The mechanism is real and the direction is right at every stage, but the size is exactly the
`+0.002` that was ruled out in advance. `B2_SEED_DIVERSE` is smaller still.

## 7. EXACT P50, end to end

SAFE = ALL 0.6612 / hop1 0.9955 / hop2 0.7297 / hop3 0.2583. Every row below emits exactly 50 partitions.

| policy | ALL | hop2 | hop3 | gained/lost | net | McNemar p |
|---|---|---|---|---|---|---|
| M0_BASELINE | 0.6612 | 0.7297 | 0.2583 | 0/0 | +0 | 1 |
| L1_MAX_FUTURE | 0.6567 | 0.7237 | 0.2508 | 8/17 | -9 | 0.108 |
| L2_TOP2_FUTURE | 0.6567 | 0.7222 | 0.2523 | 8/17 | -9 | 0.108 |
| L3_FUTURE_THEN_CURRENT | 0.6567 | 0.7237 | 0.2508 | 8/17 | -9 | 0.108 |
| L4_CURRENT_THEN_FUTURE | 0.6612 | 0.7297 | 0.2583 | 0/0 | +0 | 1 |
| D0_DELAYED_PRUNE | 0.6582 | 0.7252 | 0.2538 | 8/14 | -6 | 0.286 |
| C1_ENDPOINT_DISPLACEMENT | 0.6602 | 0.7297 | 0.2553 | 8/10 | -2 | 0.815 |
| C2_NORMALIZED_EDGE_SUM | 0.6597 | 0.7267 | 0.2568 | 9/12 | -3 | 0.664 |
| B1_PARENT_DIVERSE | 0.6622 | 0.7327 | 0.2583 | 3/1 | +2 | 0.625 |
| B2_SEED_DIVERSE | 0.6612 | 0.7297 | 0.2583 | 0/0 | +0 | 1 |

MetaQA hop3 is 0.2583 for the frozen beam and **no policy exceeds it**. The best
figure across all ten is 0.2583, reached only by policies that tie the frozen beam exactly (M0_BASELINE, L4_CURRENT_THEN_FUTURE, B1_PARENT_DIVERSE, B2_SEED_DIVERSE); every other policy is below it. The bar set
for this phase was `0.258 -> 0.30+` to be worth a quality/latency trade, and `0.258 -> 0.263`
to be rejected on sight. Nothing reaches even the reject threshold, in either direction.

Cross-corpus:

| dataset | policy | SAFE ALL | policy ALL | delta | net | McNemar p | sig |
|---|---|---|---|---|---|---|---|
| metaqa | L1_MAX_FUTURE | 0.6612 | 0.6567 | -0.0045 | -9 | 0.108 | no |
| metaqa | L2_TOP2_FUTURE | 0.6612 | 0.6567 | -0.0045 | -9 | 0.108 | no |
| metaqa | L3_FUTURE_THEN_CURRENT | 0.6612 | 0.6567 | -0.0045 | -9 | 0.108 | no |
| metaqa | L4_CURRENT_THEN_FUTURE | 0.6612 | 0.6612 | +0.0000 | +0 | 1 | no |
| metaqa | D0_DELAYED_PRUNE | 0.6612 | 0.6582 | -0.0030 | -6 | 0.286 | no |
| metaqa | C1_ENDPOINT_DISPLACEMENT | 0.6612 | 0.6602 | -0.0010 | -2 | 0.815 | no |
| metaqa | C2_NORMALIZED_EDGE_SUM | 0.6612 | 0.6597 | -0.0015 | -3 | 0.664 | no |
| metaqa | B1_PARENT_DIVERSE | 0.6612 | 0.6622 | +0.0010 | +2 | 0.625 | no |
| metaqa | B2_SEED_DIVERSE | 0.6612 | 0.6612 | +0.0000 | +0 | 1 | no |
| webqsp | B1_PARENT_DIVERSE | 0.7646 | 0.7660 | +0.0014 | +2 | 0.5 | no |
| 2wiki_clean | B1_PARENT_DIVERSE | 0.9435 | 0.9435 | +0.0000 | +0 | 1 | no |
| musique_clean | B1_PARENT_DIVERSE | 0.9635 | 0.9635 | +0.0000 | +0 | 1 | no |
| hotpotqa_clean | B1_PARENT_DIVERSE | 0.9505 | 0.9510 | +0.0005 | +1 | 1 | no |
| squad_clean | B1_PARENT_DIVERSE | 0.9875 | 0.9875 | +0.0000 | +0 | 1 | no |

## 8. Internal routing work vs downstream exposure

These are different things and were kept separate. Internal work is what the router touches
while deciding; downstream exposure is what L2 actually sees. The output contract is
unchanged at exactly 50 partitions for every policy, so no policy here buys coverage by
quietly enlarging the scope.

| policy | graph edges/q | lookahead edges/q | nodes scored/q | FINAL partitions | FINAL scope nodes/q |
|---|---|---|---|---|---|
| M0_BASELINE | 1,506 | 0 | 1,241 | 50 | 5,036 |
| L1_MAX_FUTURE | 1,379 | 5,100 | 1,097 | 50 | 5,036 |
| L2_TOP2_FUTURE | 1,472 | 5,114 | 1,175 | 50 | 5,037 |
| L3_FUTURE_THEN_CURRENT | 1,379 | 5,100 | 1,097 | 50 | 5,036 |
| L4_CURRENT_THEN_FUTURE | 1,506 | 5,076 | 1,241 | 50 | 5,036 |
| D0_DELAYED_PRUNE | 5,077 | 0 | 3,097 | 50 | 5,036 |
| C1_ENDPOINT_DISPLACEMENT | 1,410 | 0 | 1,158 | 50 | 5,036 |
| C2_NORMALIZED_EDGE_SUM | 1,380 | 0 | 1,129 | 50 | 5,036 |
| B1_PARENT_DIVERSE | 1,520 | 0 | 1,258 | 50 | 5,036 |
| B2_SEED_DIVERSE | 1,506 | 0 | 1,242 | 50 | 5,036 |

Latency, measured cleanly (400 queries spread across the hop blocks, 30-query warm-up, median of 3 repeats, nothing else on the CPU -- the ms/q inside the sweep JSONs is CPU-contaminated and is not reported anywhere):

| policy | ms/query | x frozen |
|---|---|---|
| M0_BASELINE | 22.83 | 1.00x |
| L1_MAX_FUTURE | 133.07 | 5.83x |
| L2_TOP2_FUTURE | 146.44 | 6.41x |
| L3_FUTURE_THEN_CURRENT | 140.17 | 6.14x |
| L4_CURRENT_THEN_FUTURE | 147.82 | 6.47x |
| D0_DELAYED_PRUNE | 87.05 | 3.81x |
| C1_ENDPOINT_DISPLACEMENT | 54.92 | 2.41x |
| C2_NORMALIZED_EDGE_SUM | 66.97 | 2.93x |
| B1_PARENT_DIVERSE | 27.36 | 1.20x |
| B2_SEED_DIVERSE | 26.86 | 1.18x |

Treat the multiplier as the robust quantity and the absolute ms/q as machine- and
load-dependent: this is a 12-core box and the mat-vecs go through a multithreaded BLAS, so
anything else running moves the absolute number a lot. An earlier attempt at this table had
two identical benchmarks racing each other and every figure came out roughly 3x inflated.

This cost is a *latency* cost, not a compression cheat, and the distinction matters. A prior
candidate, `U_PC5@256`, passed 30,681 MetaQA nodes downstream -- 76.4% of the corpus -- which
is fake compression. Lookahead inspects more edges internally and still emits exactly 50
partitions, i.e. 5,036 nodes downstream. It
would have been a legitimate L1 had it worked. It did not work.

## 9. Verdict

Of the four outcomes enumerated for this phase, the measured one is:

> **Better parent survival is achievable and does not convert.** Position-2 gold-parent
> survival can be raised (lookahead) and every funnel stage can be nudged (B1). Neither
> produces materially more required gold nodes in front of S4, and neither moves EXACT P50.

Two independent reasons, both measured:

1. **The loss is conserved across the compression points.** Removing the position-2 and
   position-3 prunes entirely relocates 84% of the structural loss onto `added(256)` and ends
   up *below* the frozen beam at the stage that matters (0.1139 vs 0.1334 read by S4).
2. **Admission and aggregation use different keys.** Nodes promoted on future evidence rank
   badly under the static score S4 reads, so better admission produces a worse read.

The stage that actually binds, on the frozen beam, is the score-ordered read `added(256) ->
S4(64)`: 0.1568 of gold (22% of all structural loss) is discovered, admitted, and then never read, because it ranks below
position 64 under the static score. On hop2 that single stage is 41% of the loss. That stage
is downstream of both prunes and inside the S4 contract this phase was explicitly told not to
modify, so it is named here and not touched.

Nothing is promoted. The frozen static-score beam (`B0_GLOBAL`) stands, `S4` and `F6` are
untouched, `L1_FROZEN` remains **NO**, and no TEST split was read.

## 10. Reproduction

```bash
python scratchpad/_l1bm_run.py step1 metaqa          # bit-exact parity gate
python scratchpad/_l1bm_run.py build metaqa M0_BASELINE,L1_MAX_FUTURE,...
python scratchpad/_l1bm_diag.py metaqa               # STEP 7 position-2 diagnostic
python scratchpad/_l1bm_step3.py metaqa              # tie mass at the binding hop
python scratchpad/_l1bm_step4.py metaqa              # discovery -> added -> read
python scratchpad/_l1bm_step8.py metaqa              # EXACT P50 end to end
python scratchpad/_l1bm_funnel.py metaqa             # the transition table
python scratchpad/_l1bm_latency.py metaqa 400 3      # clean latency, quiet CPU
python scratchpad/_l1bm_tables.py && python scratchpad/_l1bm_report.py
```

