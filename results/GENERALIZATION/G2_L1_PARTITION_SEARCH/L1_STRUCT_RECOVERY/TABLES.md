# PARAMETER-FREE EXACT-P50 STRUCTURAL RECOVERY — tables

All P50 rows are exactly-50-partition outputs. Significance is exact McNemar against the stated
baseline. `NOPE` entries (partition absent from a list) are excluded from percentiles and counted
separately, never imputed.

---

## T1 — STEP 1 rank transfer, MetaQA gold partitions SAFE misses

`n` is how many of the missed partitions have that rank defined at all; the gap between `n` and
`n_missed` is itself the finding.

### hop2 (872 missed gold partitions)

| stage | n | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| gold node structural rank | 222 | 39.0 | 92.0 | 118.8 | 133.0 |
| any node structural rank | 355 | 24.5 | 57.0 | 97.0 | 121.6 |
| S4 partition rank | 192 | 11.0 | 24.0 | 36.0 | 42.0 |
| F6 boundary rank | 245 | 12.0 | 22.0 | 32.0 | 43.0 |
| canonical partition rank | 654 | 63.0 | 90.5 | 130.0 | 166.4 |

### hop3 (4,939 missed gold partitions)

| stage | n | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| gold node structural rank | 858 | 10.0 | 26.5 | 61.0 | 96.3 |
| any node structural rank | 1,779 | 16.0 | 42.0 | 81.0 | 113.0 |
| S4 partition rank | 1,154 | 11.0 | 21.0 | 33.0 | 42.0 |
| F6 boundary rank | 1,379 | 15.0 | 25.0 | 36.0 | 44.0 |
| canonical partition rank | 2,925 | 74.0 | 109.0 | 150.0 | 178.0 |

### classification (thresholds = frozen operational depths `M_struct = 64`, `B = 6`)

| | n missed | A NODE_DISCOVERY | B AGGREGATION | C SWAP |
|---|---|---|---|---|
| hop2 | 872 | 799 (91.6 %) | 61 (7.0 %) | 12 (1.4 %) |
| hop3 | 4,939 | 4,281 (86.7 %) | 555 (11.2 %) | 103 (2.1 %) |

### concrete traces

| q | hop | partition | gold node rank | any node rank | min hop | S4 rank | F6 rank | canon rank | class |
|---|---|---|---|---|---|---|---|---|---|
| 667 | 2 | 9 | 99 | 99 | 2 | absent | absent | 114 | A |
| 1333 | 3 | 240 | absent | absent | — | absent | absent | 92 | A |
| 666 | 2 | 318 | absent | absent | — | absent | 42 | 45 | A (evicted) |
| 690 | 2 | 52 | 22 | 22 | 1 | 13 | 7 | 50 | B |
| 1334 | 3 | 60 | 18 | 18 | 2 | 13 | 38 | absent | B |
| 846 | 2 | 69 | 21 | 21 | 2 | 3 | 9 | 102 | C |
| 1348 | 3 | 103 | 10 | 10 | 2 | 3 | 22 | absent | C |

The first two A rows are the shape of the dominant class: the partition never enters the candidate
pool at all (`S4 absent`, `F6 absent`), so no boundary rule can reach it — its gold node is either
deep in the structural list (rank 99, past `M_struct = 64`) or absent from it entirely. The third row
is the minority A sub-case where the partition *was* a boundary incumbent (canonical rank 45) and was
evicted for having canonical evidence only.

**A-class sub-split** (over the 4,000-record dump): eviction is a small minority of it —

| | evicted from the boundary | never in base50 |
|---|---|---|
| hop2 A | 47 (5.9 %) | 752 (94.1 %) |
| hop3 A | 84 (3.1 %) | 2,653 (96.9 %) |

So the dominant failure is genuinely *never seeing* the partition, not losing it at the swap.

(Trace rows are drawn verbatim from `diag/step1_metaqa.json` → `records`, which caps at 4,000
records; the sub-split percentages are over that cap.)

## T2 — STEP 1b reachable vs beam-pruned

| | total | A1 UNREACHABLE | A2 BEAM_PRUNED |
|---|---|---|---|
| hop2 + hop3 | 5,594 | 585 (10.5 %) | 5,009 (89.5 %) |
| hop2 | 744 | 48 (6.5 %) | 696 (93.5 %) |
| hop3 | 4,850 | 537 (11.1 %) | 4,313 (88.9 %) |

Frozen-constant closure with the beam removed (565 queries measured):

| quantity | value |
|---|---|
| mean nodes reached in 3 hops | 3,774.5 |
| fraction of corpus | 9.4 % |
| new nodes per hop | 27.0 → 578.6 → 3,163.9 |
| S4 list length at `M_struct = 64` | 47.5 |
| S4 list length at depth 256 | 84.5 |

## T3 — STEP 2 `s_dir` audit (executable code, `_ta_prepartition.py:179`)

| # | question | answer |
|---|---|---|
| 1 | hop1 query representation | `r_q = residual(q, seeds, Xn)` |
| 2 | hop2 query representation | same `r_q` |
| 3 | hop3 query representation | same `r_q` |
| 4 | same residual reused? | YES — assigned once before the hop loop, never reassigned |
| 5 | hop2 depends on the hop1 edge chosen? | topologically only; the score has no dependence on it |
| 6 | hop3 depends on the full preceding path? | NO |
| 7 | path provenance preserved to partition scoring? | NO |
| 8 | where discarded? | `np.maximum.at(best_s, inv, S)`, then the flat `added` list |

**`CURRENT_SDIR_IS_TRULY_SEQUENTIAL = NO`.**

## T4 — STEP 3 mechanism: percentile rank of the gold edge (MetaQA, 9,044 gold-path edges)

0 = gold edge ranked first among its node's out-edges; **0.5 = chance**.

| path position | n | mean out-edges | STATIC (frozen) | SEQUENTIAL | qsim control | rsim control | seq changes rank |
|---|---|---|---|---|---|---|---|
| 1 | 3,636 | 8.3 | 0.5151 | 0.5151 | 0.4736 | 0.5132 | 0.0 % |
| 2 | 3,127 | 39.9 | 0.4177 | 0.4245 | 0.4521 | 0.4174 | 19.6 % |
| 3 | 2,281 | 10.9 | 0.1718 | 0.1637 | 0.2924 | 0.1778 | 14.2 % |

By query hop, positions ≥ 2 only:

| | n | STATIC | SEQUENTIAL | qsim | rsim |
|---|---|---|---|---|---|
| hop2 | 765 | 0.4305 | 0.4304 | 0.4143 | 0.4235 |
| hop3 | 4,643 | 0.2948 | 0.2954 | 0.3798 | 0.2987 |

### T4b — partial replication, 2wiki (278 gold-path edges; positions 2-3 too sparse to interpret)

| path position | n | mean out-edges | STATIC | SEQUENTIAL | qsim | rsim |
|---|---|---|---|---|---|---|
| 1 | 267 | 3.6 | 0.4538 | 0.4538 | 0.2364 | 0.4478 |
| 2 | 8 | 50.6 | 0.5743 | 0.6074 | 0.3979 | 0.5743 |
| 3 | 3 | 60.0 | 0.0463 | 0.0463 | 0.0926 | 0.0648 |

## T5 — STEPS 4-6, the full 4 × 5 grid at EXACT P50 (MetaQA)

Baseline SAFE = 0.6612 / 0.9955 / 0.7297 / 0.2583. Harness parity: N0 × P0 reproduces SAFE exactly.

| node scorer | aggregation | ALL | hop1 | hop2 | hop3 | net vs SAFE | p | sig |
|---|---|---|---|---|---|---|---|---|
| N0 static | **P0_S4** | **0.6612** | 0.9955 | **0.7297** | **0.2583** | +0 | 1 | |
| N0 static | P1_MAX_NODE | 0.6607 | 0.9955 | 0.7282 | 0.2583 | −1 | 1 | |
| N0 static | P2_TOP2_SUPPORT | 0.6597 | 0.9955 | 0.7282 | 0.2553 | −3 | 0.508 | |
| N0 static | P3_BEST_PATH | 0.6557 | 0.9955 | 0.7192 | 0.2523 | −11 | 0.0433 | ✗ |
| N0 static | P4_MULTI_SEED | 0.6607 | 0.9955 | 0.7282 | 0.2583 | −1 | 1 | |
| N1 terminal | P0_S4 | 0.6597 | 0.9955 | 0.7252 | 0.2583 | −3 | 0.664 | |
| N1 terminal | P1_MAX_NODE | 0.6587 | 0.9955 | 0.7237 | 0.2568 | −5 | 0.383 | |
| N1 terminal | P2_TOP2_SUPPORT | 0.6587 | 0.9955 | 0.7252 | 0.2553 | −5 | 0.383 | |
| N1 terminal | P3_BEST_PATH | 0.6567 | 0.9955 | 0.7207 | 0.2538 | −9 | 0.122 | |
| N1 terminal | P4_MULTI_SEED | 0.6587 | 0.9955 | 0.7237 | 0.2568 | −5 | 0.383 | |
| N2 path-min | P0_S4 | 0.6552 | 0.9955 | 0.7177 | 0.2523 | −12 | 0.0428 | ✗ |
| N2 path-min | P1_MAX_NODE | 0.6552 | 0.9955 | 0.7192 | 0.2508 | −12 | 0.0428 | ✗ |
| N2 path-min | P2_TOP2_SUPPORT | 0.6557 | 0.9955 | 0.7207 | 0.2508 | −11 | 0.0708 | |
| N2 path-min | P3_BEST_PATH | 0.6562 | 0.9955 | 0.7207 | 0.2523 | −10 | 0.110 | |
| N2 path-min | P4_MULTI_SEED | 0.6552 | 0.9955 | 0.7192 | 0.2508 | −12 | 0.0428 | ✗ |
| N3 path-geo | P0_S4 | 0.6567 | 0.9955 | 0.7207 | 0.2538 | −9 | 0.150 | |
| N3 path-geo | P1_MAX_NODE | 0.6567 | 0.9955 | 0.7222 | 0.2523 | −9 | 0.163 | |
| N3 path-geo | P2_TOP2_SUPPORT | 0.6572 | 0.9955 | 0.7237 | 0.2523 | −8 | 0.200 | |
| N3 path-geo | P3_BEST_PATH | 0.6567 | 0.9955 | 0.7222 | 0.2523 | −9 | 0.163 | |
| N3 path-geo | P4_MULTI_SEED | 0.6567 | 0.9955 | 0.7222 | 0.2523 | −9 | 0.163 | |

### T5b — node-level discovery at matched compute (200 queries/hop)

Gold NODES found inside the beam, and the edge budget each beam spent:

| scorer | hop1 | hop2 | hop3 | edges scored (full MetaQA) |
|---|---|---|---|---|
| N0 static | 125 | **503** | **625** | 3,008,656 |
| N1 terminal | 125 | 484 | 541 | 2,970,383 |
| N2 path-min | 125 | 472 | 439 | 2,594,241 |
| N3 path-geo | 125 | **506** | 494 | 2,698,468 |

(of 361 / 1,097 / 3,078 gold nodes respectively; hop1 is identical by construction — both scorers
use `r_0` at the first hop.)

## T6 — STEP 5/6 aggregations across all six corpora, EXACT P50

Computed from the frozen structural cache, so P0 is the frozen S4 by construction (parity verified
on musique: identical ranking on every query). P3/P4 need path provenance the frozen cache never
stored and are MetaQA-only (T5).

| aggregation | metaqa | webqsp | 2wiki | musique | hotpot | squad |
|---|---|---|---|---|---|---|
| **P0_S4 (frozen)** | **0.6612** | **0.7646** | **0.9435** | **0.9635** | 0.9505 | **0.9875** |
| P1_MAX_NODE | 0.6607 | 0.7646 | 0.9430 | 0.9625 | 0.9510 | 0.9870 |
| P2_TOP2_SUPPORT | 0.6597 | 0.7639 | 0.9430 | 0.9625 | 0.9510 | 0.9875 |

net vs P0 (exact McNemar; no cell is significant in either direction):

| aggregation | metaqa | webqsp | 2wiki | musique | hotpot | squad |
|---|---|---|---|---|---|---|
| P1_MAX_NODE | −1 | +0 | −1 | −2 | +1 | −1 |
| P2_TOP2_SUPPORT | −3 | −1 | −1 | −2 | +1 | +0 |

## T7 — STEP 7 candidate-budget curve, MetaQA

`final_k = prot(44) ∪ top-(k−44)` of the same score order; k = 50 is exactly the P50 output.

| k | partitions actually output | ALL | hop1 | hop2 | hop3 |
|---|---|---|---|---|---|
| 50 | 50.0 | 0.6612 | 0.9955 | 0.7297 | 0.2583 |
| 64 | 64.0 | 0.6842 | 0.9955 | 0.7673 | 0.2898 |
| 80 | 80.0 | 0.7047 | 0.9970 | 0.7838 | 0.3333 |
| 100 | 100.0 | 0.7377 | 0.9985 | 0.8258 | 0.3889 |
| 128 | 128.0 | 0.7718 | 0.9985 | 0.8874 | 0.4294 |
| 256 † | 215.8 | 0.8278 | 0.9985 | 0.9339 | 0.5511 |

† candidate pool exhausted — an upper bound, not a 256-partition operating point.

Marginal improvement per increment:

| increment | ALL | hop1 | hop2 | hop3 |
|---|---|---|---|---|
| 50 → 64 | +0.0230 | +0.0000 | +0.0376 | +0.0315 |
| 64 → 80 | +0.0205 | +0.0015 | +0.0165 | +0.0435 |
| 80 → 100 | +0.0330 | +0.0015 | +0.0420 | +0.0556 |
| 100 → 128 | +0.0341 | +0.0000 | +0.0616 | +0.0405 |
| 128 → 256 † | +0.0560 | +0.0000 | +0.0465 | +0.1217 |

**No knee.** Increments are flat to increasing, so the gain available at 128 does not become
available at 64-80.

### T7b — the same curve on all six corpora (ALL)

| corpus | k=50 | k=64 | k=80 | k=100 | k=128 | k=256 † (partitions) |
|---|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.6842 | 0.7047 | 0.7377 | 0.7718 | 0.8278 (215.8) |
| webqsp | 0.7646 | 0.7808 | 0.7970 | 0.8295 | 0.8576 | 0.9056 (221.5) |
| 2wiki | 0.9435 | 0.9475 | 0.9500 | 0.9575 | 0.9630 | 0.9755 (219.2) |
| musique | 0.9635 | 0.9745 | 0.9840 | 0.9915 | 0.9990 | 1.0000 (136.0) |
| hotpot | 0.9505 | 0.9615 | 0.9625 | 0.9640 | 0.9665 | 0.9750 (233.6) |
| squad | 0.9875 | 0.9945 | 0.9955 | 0.9970 | 0.9990 | 1.0000 (190.0) |

Total 50 → 128 gain: metaqa **+0.1106**, webqsp **+0.0930**, 2wiki +0.0195, musique +0.0355,
hotpot +0.0160, squad +0.0115. The two KBs are the budget-elastic corpora; the four text corpora
are near-saturated at k = 50.

## T8 — STEP 9 swap rules, EXACT P50, all six corpora

Frozen structural ranking; only the boundary rule changes. Baseline is A_F6.

| corpus | A_F6 | B_STRUCT_DIRECT | net | p | C_PAIRWISE_EVIDENCE | net | p |
|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | **0.6712** | **+20** | 0.00032 ✓ | **0.6667** | **+11** | 0.0192 ✓ |
| webqsp | 0.7646 | 0.7533 | −16 | 0.00014 ✗ | 0.7625 | −3 | 0.508 |
| 2wiki | 0.9435 | 0.9340 | −19 | 7e-05 ✗ | 0.9370 | −13 | 0.00235 ✗ |
| musique | 0.9635 | 0.9505 | −26 | <1e-06 ✗ | 0.9530 | −21 | 5e-05 ✗ |
| hotpot | 0.9505 | 0.9285 | −44 | <1e-06 ✗ | 0.9365 | −28 | <1e-06 ✗ |
| squad | 0.9875 | 0.9790 | −17 | 8e-05 ✗ | 0.9810 | −13 | 0.00098 ✗ |

MetaQA per-hop for the two challengers:

| rule | hop1 | hop2 | hop3 |
|---|---|---|---|
| A_F6 | 0.9955 | 0.7297 | 0.2583 |
| B_STRUCT_DIRECT | 0.9955 | 0.7432 | 0.2748 |
| C_PAIRWISE_EVIDENCE | 0.9955 | 0.7342 | 0.2703 |

**Not promoted.** The MetaQA gain is corpus-specific and costs significantly more on the other five
corpora, WebQSP — the other KB — included.

## T9 — STEP 8 node → partition transfer (corrected)

The first measurement was invalid: `struct_aggregate_full` reads only `M_struct = 64` nodes, so the
S4 list is ≤ 47.5 entries and "in S4 top-64/80/100" was trivially true. Re-measured against S4 over
the full 256-node list and against the final P50.

| | n | median partition rank | p90 | partition absent |
|---|---|---|---|---|
| gold node → S4@64 | 4,318 | 9.0 | 33.0 | 1,004 |
| gold node → S4@256 | 4,318 | **12.0** | 54.0 | **0** |

| top-8 structural gold nodes | n | S4@256 top-50 | top-64 | top-80 | top-100 | **in final P50** |
|---|---|---|---|---|---|---|
| | 598 | 100.0 % | 100.0 % | 100.0 % | 100.0 % | **68.1 %** |

Displacement by node-rank band (S4@256):

| node rank band | n | median S4@256 partition rank | absent |
|---|---|---|---|
| 0-7 | 598 | 6.0 | 0 |
| 8-15 | 370 | 10.0 | 0 |
| 16-31 | 471 | 17.0 | 0 |
| 32-63 | 546 | 20.0 | 0 |
| 64-127 | 1,050 | 19.0 | 0 |
| 128-255 | 1,283 | 9.0 | 0 |

Transfer is lossless (0 absent in every band) and compressive: a gold node at structural rank 128-255
still lands at median partition rank 9, because many deep nodes fall into partitions that shallower
nodes already support. The stage that loses gold is neither the aggregation nor the node→partition
map.

## T10 — returns

| flag | value |
|---|---|
| `NODE_DISCOVERY_LIMITED` | **YES** — 91.6 % hop2 / 86.7 % hop3, of which 89.5 % is reachable-but-pruned |
| `AGGREGATION_LIMITED` | **NO** — 7.0 % / 11.2 %, and no alternative aggregation converts any of it |
| `SWAP_LIMITED` | **NO** — 1.4 % / 2.1 %; the MetaQA-winning rules regress significantly on 5/6 corpora |
| `CURRENT_SDIR_IS_TRULY_SEQUENTIAL` | **NO** |
| `BEST_STRUCTURAL_NODE_METHOD` | **N0_STATIC** (the existing score) |
| `BEST_NODE_TO_PARTITION_METHOD` | **P0_S4** (the existing aggregation) |
| `EXACT_P50_METAQA_HOP2` | **0.7297** (unchanged from SAFE) |
| `EXACT_P50_METAQA_HOP3` | **0.2583** (unchanged from SAFE) |
