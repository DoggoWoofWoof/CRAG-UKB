# L1 QWEN-KNN EDGE SUBSTRATE AUDIT -- TABLES

Corpora with a completed replay: **MuSiQue** (INCOMPLETE).


## STEP 1 -- EDGE SUBSTRATES

### 1a. The three precomputed families, and how much of topology C the traversal never sees

| corpus | N | STRUCT (traversed) | KNN | NERX | KNN&NERX | edges of C never traversed |
|---|--:|--:|--:|--:|--:|--:|
| MetaQA | 40,151 | 109,480 | 55,207 | 232,505 | 0 | **72.4%** |
| WebQSP | 781,485 | 1,638,066 | 1,672,731 | 975,995 | 0 | **61.8%** |
| 2Wiki | 65,865 | 126,070 | 134,737 | 494,304 | 0 | **83.3%** |
| MuSiQue | 13,672 | 51,543 | 27,406 | 93,500 | 0 | **70.1%** |
| Hotpot | 507,494 | 3,426,945 | 1,166,652 | 4,058,707 | 0 | **60.4%** |
| SQuAD | 19,029 | 694,580 | 28,276 | 126,488 | 0 | **18.2%** |

### 1b. The traversal substrates

| corpus | substrate | undirected edges | mean deg | max deg | isolated | over DEG_CAP |
|---|---|--:|--:|--:|--:|--:|
| MetaQA | `T0_FROZEN` | 109,480 | 5.45 | 4,180 | 0 | 39 |
| MetaQA | `T1_KNN_ONLY` | 55,207 | 2.75 | 41 | 2,010 | 0 |
| MetaQA | `T2_FULL_UNION` | 164,687 | 8.20 | 4,183 | 0 | 40 |
| MetaQA | `T3_MATCHED_HYBRID` (kNN slots 21.1%) | 109,480 | 5.45 | 4,180 | 0 | 39 |
| MetaQA | `T4_NERX_ONLY` | 232,505 | 11.58 | 107 | 10,503 | 0 |
| MetaQA | `T5_TOPOLOGY_C` | 387,788 | 19.32 | 4,199 | 0 | 41 |
| WebQSP | `T0_FROZEN` | 1,638,066 | 4.19 | 41,605 | 0 | 665 |
| WebQSP | `T1_KNN_ONLY` | 1,672,731 | 4.28 | 180 | 8,378 | 0 |
| WebQSP | `T2_FULL_UNION` | 3,310,797 | 8.47 | 41,607 | 0 | 669 |
| WebQSP | `T3_MATCHED_HYBRID` (kNN slots 15.6%) | 1,638,066 | 4.19 | 41,605 | 0 | 665 |
| WebQSP | `T4_NERX_ONLY` | 975,995 | 2.50 | 129 | 595,103 | 0 |
| WebQSP | `T5_TOPOLOGY_C` | 4,154,576 | 10.63 | 41,607 | 0 | 688 |
| 2Wiki | `T0_FROZEN` | 126,070 | 3.83 | 38,592 | 11,398 | 26 |
| 2Wiki | `T1_KNN_ONLY` | 134,737 | 4.09 | 30 | 357 | 0 |
| 2Wiki | `T2_FULL_UNION` | 260,807 | 7.92 | 38,595 | 0 | 27 |
| 2Wiki | `T3_MATCHED_HYBRID` (kNN slots 23.5%) | 126,070 | 3.83 | 38,592 | 11,398 | 26 |
| 2Wiki | `T4_NERX_ONLY` | 494,304 | 15.01 | 336 | 14,057 | 1 |
| 2Wiki | `T5_TOPOLOGY_C` | 728,689 | 22.13 | 38,595 | 0 | 28 |
| MuSiQue | `T0_FROZEN` | 51,543 | 7.54 | 1,486 | 3,792 | 17 |
| MuSiQue | `T1_KNN_ONLY` | 27,406 | 4.01 | 26 | 182 | 0 |
| MuSiQue | `T2_FULL_UNION` | 78,949 | 11.55 | 1,490 | 0 | 17 |
| MuSiQue | `T3_MATCHED_HYBRID` (kNN slots 15.3%) | 51,543 | 7.54 | 1,486 | 3,792 | 17 |
| MuSiQue | `T4_NERX_ONLY` | 93,500 | 13.68 | 248 | 2,038 | 0 |
| MuSiQue | `T5_TOPOLOGY_C` | 163,723 | 23.95 | 1,494 | 0 | 20 |
| Hotpot | `T0_FROZEN` | 3,426,945 | 13.51 | 46,891 | 3,520 | 1,180 |
| Hotpot | `T1_KNN_ONLY` | 1,166,652 | 4.60 | 260,449 | 3,562 | 1 |
| Hotpot | `T2_FULL_UNION` | 4,593,597 | 18.10 | 260,464 | 30 | 1,194 |
| Hotpot | `T3_MATCHED_HYBRID` (kNN slots 20.3%) | 3,426,945 | 13.51 | 46,891 | 3,520 | 1,180 |
| Hotpot | `T4_NERX_ONLY` | 4,058,707 | 15.99 | 588 | 86,204 | 2 |
| Hotpot | `T5_TOPOLOGY_C` | 8,468,338 | 33.37 | 260,481 | 14 | 1,239 |
| SQuAD | `T0_FROZEN` | 694,580 | 73.00 | 2,334 | 6,082 | 595 |
| SQuAD | `T1_KNN_ONLY` | 28,276 | 2.97 | 22 | 3,352 | 0 |
| SQuAD | `T2_FULL_UNION` | 722,856 | 75.97 | 2,334 | 0 | 617 |
| SQuAD | `T3_MATCHED_HYBRID` (kNN slots 2.2%) | 694,580 | 73.00 | 2,334 | 6,082 | 595 |
| SQuAD | `T4_NERX_ONLY` | 126,488 | 13.29 | 171 | 4,870 | 0 |
| SQuAD | `T5_TOPOLOGY_C` | 841,351 | 88.43 | 2,334 | 0 | 691 |

## STEP 2 -- PARITY

`T0_FROZEN` must reproduce frozen L1 exactly: the structural rank map `spos` identical on every query, and the exact-P50 outcome identical on every query.

| corpus | nq | `spos` exact | P50 exact | T0_PARITY | seeds/q | edges/q | nodes/q | partitions/q | hops |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|
| MuSiQue | 100 | 100/100 | 100/100 | **EXACT** | 5.00 | 2,847 | 968 | 116.0 | 3 |

## STEP 3 -- UNIQUE REACH

Every gold partition MISSED by the frozen SAFE output, classified by which substrate reaches it. `VISITED` = anything the bounded search scored; `READ` = what the M_struct=64 read actually hands the S4 aggregation.  `KNN_UNIQUE` is the primary number.


**VISITED**

| corpus / slice | missed needed | STRUCT only | **KNN only** | both | neither | NERX only | **KNN_UNIQUE_NEEDED_RECALL** |
|---|--:|--:|--:|--:|--:|--:|--:|
| MuSiQue | 5 | 2 | **0** | 3 | 0 | 0 | **0.0000** |

**READ (M_struct=64)**

| corpus / slice | missed needed | STRUCT only | **KNN only** | both | neither | NERX only | **KNN_UNIQUE_NEEDED_RECALL** |
|---|--:|--:|--:|--:|--:|--:|--:|
| MuSiQue | 5 | 1 | **2** | 0 | 2 | 1 | **0.4000** |

### 3c. What turning the family ON actually buys (needed-partition recall vs `T0`)

`T1` runs at a FRACTION of `T0`'s edge budget (the kNN graph is sparse), so its standalone reach understates the family. `T2`/`T5` are the operational question: does adding the family to the frozen traversal move needed recall?

| corpus / slice | level | T0 | T1 kNN-only | T2 +kNN | T3 matched | T4 NERX-only | T5 topology C | **T2 - T0** |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| MuSiQue | VIS | 1.0000 | 0.9497 | 1.0000 | 0.9944 | 1.0000 | 1.0000 | **+0.0000** |
| MuSiQue | READ | 0.5642 | 0.7821 | 0.7486 | 0.5978 | 0.7430 | 0.7598 | **+0.1844** |

## STEP 4 -- DENSE REDUNDANCY

Every partition the kNN-only traversal reaches, tested against the channels the frozen system already has. `novel` = in none of dense / SPLADE / canonical-200 / retrieval continuation / structural-T0.

| corpus | kNN partitions | in Dense | in SPLADE | in canon-200 | in RF | in struct T0 | **novel** | needed | needed&novel | P(needed \| novel) | P(needed \| dup) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MuSiQue | 5,102 | 68.8% | 71.0% | 100.0% | 26.9% | 90.5% | **0.0%** | 170 | 0 | 0.0000 | 0.0333 |

## STEP 5 -- EDGE-FAMILY SATURATION

| corpus | substrate | nodes visited/q | partitions visited/q | % of corpus partitions | admitted partitions/q | read partitions/q |
|---|---|--:|--:|--:|--:|--:|
| MuSiQue | `T0_FROZEN` | 968 | 116.0 | **85.3%** | 48.1 | 29.3 |
| MuSiQue | `T1_KNN_ONLY` | 167 | 51.0 | **37.5%** | 42.3 | 30.5 |
| MuSiQue | `T2_FULL_UNION` | 1,292 | 125.2 | **92.1%** | 54.1 | 30.5 |
| MuSiQue | `T3_MATCHED_HYBRID` | 1,045 | 117.7 | **86.6%** | 49.8 | 28.5 |
| MuSiQue | `T4_NERX_ONLY` | 1,484 | 124.5 | **91.6%** | 54.8 | 30.2 |
| MuSiQue | `T5_TOPOLOGY_C` | 2,349 | 132.8 | **97.6%** | 56.4 | 30.1 |

## STEPS 6 + 9 -- MATCHED WORK AND CANDIDATE ORACLES

`T3` spends the SAME per-node edge budget as `T0` (same `adj_ptr`, same `deg`, same DEG_CAP), reallocated between families. Needed-partition recall is per gold partition; the candidate oracle asks whether ANY B=6 subset of that substrate's F6 pool could cover the query.

| corpus / slice | substrate | edges/q | needed recall VIS | needed recall READ | candidate oracle | partitions/q | exact P50 |
|---|---|--:|--:|--:|--:|--:|--:|
| MuSiQue | `T0_FROZEN` | 2,847 | 1.0000 | 0.5642 | 0.9700 | 116.0 | 0.9500 |
| MuSiQue | `T1_KNN_ONLY` | 288 | 0.9497 | 0.7821 | 0.9700 | 51.0 | 0.9500 |
| MuSiQue | `T2_FULL_UNION` | 3,483 | 1.0000 | 0.7486 | 0.9800 | 125.2 | 0.9500 |
| MuSiQue | `T3_MATCHED_HYBRID` | 2,733 | 0.9944 | 0.5978 | 0.9700 | 117.7 | 0.9400 |
| MuSiQue | `T4_NERX_ONLY` | 3,637 | 1.0000 | 0.7430 | 0.9700 | 124.5 | 0.9500 |
| MuSiQue | `T5_TOPOLOGY_C` | 5,747 | 1.0000 | 0.7598 | 0.9600 | 132.8 | 0.9500 |

## STEP 7 -- PATH PROVENANCE

Family sequence of every path `T2_FULL_UNION` actually explored (the `parent_tid` chain of each scored edge). `needed-target rate` = fraction of that sequence's edges whose target node lies in a gold partition.


**MuSiQue** (341,851 scored edges)

| family sequence | paths | share | needed-target rate | distinct needed partitions |
|---|--:|--:|--:|--:|
| `STRUCT>STRUCT` | 136,007 | 39.8% | 0.14539 | 92 |
| `STRUCT>STRUCT>STRUCT` | 83,706 | 24.5% | 0.06759 | 79 |
| `KNN>STRUCT>STRUCT` | 40,623 | 11.9% | 0.02385 | 54 |
| `STRUCT>STRUCT>KNN` | 12,356 | 3.6% | 0.04799 | 73 |
| `KNN+NERX>STRUCT>STRUCT` | 10,011 | 2.9% | 0.03216 | 22 |
| `KNN>STRUCT` | 8,837 | 2.6% | 0.10286 | 50 |
| `STRUCT>KNN>STRUCT` | 7,286 | 2.1% | 0.03637 | 37 |
| `STRUCT>KNN` | 6,239 | 1.8% | 0.10659 | 58 |
| `STRUCT` | 5,216 | 1.5% | 0.32189 | 67 |
| `KNN>KNN` | 4,605 | 1.3% | 0.19501 | 84 |
| `STRUCT>STRUCT>KNN+NERX` | 4,570 | 1.3% | 0.08096 | 43 |
| `STRUCT>KNN+NERX>STRUCT` | 2,230 | 0.7% | 0.03049 | 15 |
| `KNN+NERX>STRUCT` | 2,102 | 0.6% | 0.18221 | 42 |
| `STRUCT>KNN>KNN` | 2,037 | 0.6% | 0.12371 | 45 |

## STEP 8 -- KNN AS ITS OWN EVIDENCE FAMILY

| corpus | queries with a kNN edge | kNN partitions/q | needed reached | missed-needed reached |
|---|--:|--:|--:|--:|
| MuSiQue | 10 | 88.4 | 17/17 (100.0%) | 0/0 (0.0%) |

| corpus | signal | mean needed | mean nuisance | AUC needed vs nuisance | AUC missed vs nuisance |
|---|---|--:|--:|--:|--:|
| MuSiQue | `K1_BEST_KNN_SIM` | 0.6359 | 0.5324 | **0.8193** | None |
| MuSiQue | `K1_QUERY_SIM` | 0.5022 | 0.2786 | **0.9644** | None |
| MuSiQue | `K2_SEEDS` | 3.2941 | 1.5433 | **0.8744** | None |
| MuSiQue | `K2_SRC_PARTS` | 7.5294 | 2.5848 | **0.9221** | None |
| MuSiQue | `K2_EDGES` | 22.2941 | 4.421 | **0.9552** | None |

## STEP 11 -- COST

Uniform instrumentation (`want_edges=True` for every substrate, so the timings are comparable). `edges/q` is the adjacency volume inspected; `scored/q` the edges that survived DEG_CAP and entered the matvec.

| corpus | substrate | edges/q | scored/q | scope/q | ms/q | x T0 edges | x T0 ms |
|---|---|--:|--:|--:|--:|--:|--:|
| MuSiQue | `T0_FROZEN` | 2,564 | 2,515 | 160 | 80.43 | 1.00 | 1.00 |
| MuSiQue | `T1_KNN_ONLY` | 268 | 268 | 123 | 37.30 | 0.10 | 0.46 |
| MuSiQue | `T2_FULL_UNION` | 3,246 | 3,189 | 175 | 126.18 | 1.27 | 1.57 |
| MuSiQue | `T3_MATCHED_HYBRID` | 2,428 | 2,399 | 162 | 81.44 | 0.95 | 1.01 |
| MuSiQue | `T4_NERX_ONLY` | 3,786 | 3,786 | 190 | 85.34 | 1.48 | 1.06 |
| MuSiQue | `T5_TOPOLOGY_C` | 5,867 | 5,766 | 195 | 108.19 | 2.29 | 1.35 |

## STEP 10 -- EXACT P50

The frozen channels are untouched; the kNN structural ranking is added as a FOURTH symmetric RRF channel. `K_CHAN_VOTE` lets it re-score the frozen candidate list only; `K_CHAN_FULL` also lets it contribute challengers. `*` = McNemar significant.

| corpus / slice | FROZEN | K_CHAN_VOTE | K_CHAN_FULL | T2 substitution | T3 substitution |
|---|--:|--:|--:|--:|--:|
| MuSiQue | 0.9500 | 0.9500 (+0) | 0.9500 (+0) | 0.9500 (+0) | 0.9400 (-1) |
