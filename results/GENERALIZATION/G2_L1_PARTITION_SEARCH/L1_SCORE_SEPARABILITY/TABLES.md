# L1 STRUCTURAL SCORE SEPARABILITY -- TABLES

## T1. STEP 1 -- the read-stage dataset (parity gate)

| corpus | queries | rows | rows/query | frozen-array parity | gold discovered | edges/q |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 1998 | 293,624 | 147.0 | **1998/1998** | 0.2902 | 1,505.8 |
| webqsp | 1419 | 196,355 | 138.4 | **1419/1419** | 0.2734 | 1,661.8 |
| 2wiki_clean | 2000 | 211,481 | 105.7 | **2000/2000** | 0.2596 | 601.1 |
| musique_clean | 2000 | 305,927 | 153.0 | **2000/2000** | 0.1400 | 3,196.0 |
| hotpotqa_clean | 2000 | 301,986 | 151.0 | **2000/2000** | 0.1872 | 1,892.9 |
| squad_clean | 2000 | 317,597 | 158.8 | **2000/2000** | 0.0265 | 14,181.3 |

Every corpus reproduces the frozen `s_node/s_hop/s_sdir/s_cnt` arrays exactly on every query (11,417 queries total), so the feature dataset IS the frozen read stage rather than a re-implementation of it.

## T2. STEP 2/3 -- single-signal separability at the read bottleneck (MetaQA)

Absolute recall = gold in top-k / all required gold nodes. `within` = Mann-Whitney AUC computed inside one query AND one node hop; its no-information floor is the frozen order's own 0.6034, not 0.5, because a constant signal falls back to the stable tiebreak.

| signal | R@16 | R@32 | **R@64** | R@128 | cond@64 | gold pctrank | AUC | within |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RR_PARENT_SUPPORT | 0.1033 | 0.1483 | **0.2061** | 0.2688 | 0.7101 | 0.3011 | 0.7792 | 0.6243 |
| ADMIT_RANK | 0.0833 | 0.1321 | **0.2042** | 0.2719 | 0.7036 | 0.3113 | 0.7538 | 0.6049 |
| HOP | 0.1019 | 0.1534 | **0.2010** | 0.2754 | 0.6927 | 0.2977 | 0.7974 | 0.6034 |
| PATH_SUPPORT | 0.0875 | 0.1377 | **0.2004** | 0.2603 | 0.6906 | 0.3343 | 0.6866 | 0.5793 |
| PARTITION_CANONICAL_PRIOR | 0.0864 | 0.1405 | **0.1977** | 0.2762 | 0.6811 | 0.3093 | 0.7734 | 0.676 |
| PARENT_SCORE | 0.0985 | 0.1435 | **0.1957** | 0.2632 | 0.6744 | 0.3239 | 0.7335 | 0.5336 |
| PARTITION_RETRIEVAL_PRIOR | 0.0755 | 0.1204 | **0.1951** | 0.2516 | 0.6723 | 0.3592 | 0.6938 | 0.6547 |
| SEED_RANK | 0.0807 | 0.1215 | **0.1776** | 0.2581 | 0.6119 | 0.3706 | 0.6712 | 0.7512 |
| QSIM | 0.0616 | 0.1036 | **0.1671** | 0.2650 | 0.5757 | 0.3883 | 0.6693 | 0.6326 |
| PATH_MIN | 0.0738 | 0.1044 | **0.1544** | 0.2457 | 0.5320 | 0.4293 | 0.5940 | 0.5444 |
| DISTINCT_SEED_SUPPORT | 0.0680 | 0.1085 | **0.1491** | 0.2153 | 0.5139 | 0.4697 | 0.5048 | 0.615 |
| STATIC_SDIR | 0.0688 | 0.1012 | **0.1383** | 0.2115 | 0.4766 | 0.4891 | 0.4940 | 0.6285 |
| PATH_SUM | 0.0501 | 0.0858 | **0.1343** | 0.2108 | 0.4627 | 0.5151 | 0.4485 | 0.5804 |
| CS_ADMIT **<- frozen key** | 0.0651 | 0.0967 | **0.1334** | 0.2040 | 0.4597 | 0.5066 | 0.4541 | 0.6034 |
| FUTURE_MAX | 0.0528 | 0.0828 | **0.1291** | 0.2695 | 0.4449 | 0.4226 | 0.6017 | 0.4624 |
| RSIM | 0.0530 | 0.0881 | **0.1280** | 0.2096 | 0.4412 | 0.5223 | 0.4453 | 0.5848 |

## T3. STEP 4 -- CROSS: gold density by (query hop x node hop), MetaQA

| | node_hop1 | node_hop2 | node_hop3 |
|---|---:|---:|---:|
| query_hop1 | 5.22% | 0.00% | 0.00% |
| query_hop2 | 7.70% | 1.39% | 0.09% |
| query_hop3 | 1.80% | 1.47% | 2.44% |
| ALL | 4.80% | 0.96% | 0.84% |

## T4. STEP 5 -- support mechanism audit (MetaQA, gold vs non-gold)

| signal | gold mean | non-gold mean | Cohen d |
|---|---:|---:|---:|
| PARENT_SCORE | 0.4935 | 0.1578 | +0.9556 |
| RR_PARENT_SUPPORT | 0.5080 | 0.1885 | +0.9154 |
| HOP | 1.7714 | 2.2933 | -0.7441 |
| ADMIT_RANK | 16.8710 | 28.9571 | -0.6468 |
| SEED_RANK | 0.9940 | 1.8908 | -0.6155 |
| QSIM | 0.4126 | 0.3699 | +0.5835 |
| DISTINCT_SEED_SUPPORT | 1.1881 | 1.0568 | +0.5175 |
| PARTITION_RETRIEVAL_PRIOR | 580594.4896 | 786279.4660 | -0.5001 |
| PATH_MIN | -0.0143 | -0.0402 | +0.3954 |
| PARTITION_CANONICAL_PRIOR | 97527.9521 | 263549.8599 | -0.3784 |

## T5. STEP 6/9 -- exact P50 on all six corpora

`read@64` is gold read recall; `->S4` is the fraction of the query's gold partitions reaching the S4 ranking. `net` is the McNemar net query change against frozen.

| corpus | ranking | exact P50 | net | p | sig | read@64 | ->S4 |
|---|---|---:|---:|---:|:-:|---:|---:|
| metaqa | R0_STATIC (frozen) | 0.6612 | +0 | 1 |  | 0.1334 | 0.7368 |
| metaqa | R1_EQUAL_RRF | 0.6617 | +1 | 1 |  | 0.1933 | 0.8246 |
| metaqa | R2_HOP_CONDITIONED | 0.6607 | -1 | 1 |  | 0.2100 | 0.8314 |
| metaqa | RS_BEST_SINGLE (direct) | 0.6577 | -7 | 0.248 |  | 0.2061 | 0.7841 |
| metaqa | RX_ORACLE_READ (ceiling) | 0.6707 | +19 | 0 | YES | 0.2902 | 1.0000 |
| webqsp | R0_STATIC (frozen) | 0.7646 | +0 | 1 |  | 0.1299 | 0.7289 |
| webqsp | R1_EQUAL_RRF | 0.7681 | +5 | 0.0625 |  | 0.2081 | 0.8654 |
| webqsp | R2_HOP_CONDITIONED | 0.7653 | +1 | 1 |  | 0.2043 | 0.8811 |
| webqsp | RS_BEST_SINGLE (direct) | 0.7696 | +7 | 0.143 |  | 0.2369 | 0.9113 |
| webqsp | RX_ORACLE_READ (ceiling) | 0.7752 | +15 | 6e-05 | YES | 0.2734 | 1.0000 |
| 2wiki_clean | R0_STATIC (frozen) | 0.9435 | +0 | 1 |  | 0.0751 | 0.5758 |
| 2wiki_clean | R1_EQUAL_RRF | 0.9430 | -1 | 1 |  | 0.2340 | 0.9250 |
| 2wiki_clean | R2_HOP_CONDITIONED | 0.9430 | -1 | 1 |  | 0.2590 | 0.9992 |
| 2wiki_clean | RS_BEST_SINGLE (direct) | 0.9425 | -2 | 0.5 |  | 0.2571 | 0.9912 |
| 2wiki_clean | RX_ORACLE_READ (ceiling) | 0.9435 | +0 | 1 |  | 0.2596 | 1.0000 |
| musique_clean | R0_STATIC (frozen) | 0.9635 | +0 | 1 |  | 0.0876 | 0.8797 |
| musique_clean | R1_EQUAL_RRF | 0.9645 | +2 | 0.688 |  | 0.1118 | 0.9288 |
| musique_clean | R2_HOP_CONDITIONED | 0.9625 | -2 | 0.754 |  | 0.1009 | 0.9066 |
| musique_clean | RS_BEST_SINGLE (direct) | 0.9615 | -4 | 0.388 |  | 0.1064 | 0.8813 |
| musique_clean | RX_ORACLE_READ (ceiling) | 0.9640 | +1 | 1 |  | 0.1400 | 1.0000 |
| hotpotqa_clean | R0_STATIC (frozen) | 0.9505 | +0 | 1 |  | 0.0767 | 0.7037 |
| hotpotqa_clean | R1_EQUAL_RRF | 0.9510 | +1 | 1 |  | 0.1615 | 0.9147 |
| hotpotqa_clean | R2_HOP_CONDITIONED | 0.9485 | -4 | 0.344 |  | 0.1732 | 0.9702 |
| hotpotqa_clean | RS_BEST_SINGLE (direct) | 0.9490 | -3 | 0.375 |  | 0.1817 | 0.9851 |
| hotpotqa_clean | RX_ORACLE_READ (ceiling) | 0.9515 | +2 | 0.5 |  | 0.1872 | 1.0000 |
| squad_clean | R0_STATIC (frozen) | 0.9875 | +0 | 1 |  | 0.0195 | 0.9245 |
| squad_clean | R1_EQUAL_RRF | 0.9880 | +1 | 1 |  | 0.0190 | 0.9245 |
| squad_clean | R2_HOP_CONDITIONED | 0.9855 | -4 | 0.125 |  | 0.0165 | 0.8679 |
| squad_clean | RS_BEST_SINGLE (direct) | 0.9880 | +1 | 1 |  | 0.0185 | 0.8302 |
| squad_clean | RX_ORACLE_READ (ceiling) | 0.9875 | +0 | 1 |  | 0.0265 | 1.0000 |

## T6. MetaQA per hop -- the phase primary target

Target: hop3 materially above 0.2583 and hop2 materially above 0.7297.

| ranking | ALL | hop1 | hop2 | hop3 |
|---|---:|---:|---:|---:|
| R0_STATIC (frozen) | 0.6612 | 0.9955 | 0.7297 | 0.2583 |
| R1_EQUAL_RRF | 0.6617 | 0.9955 | 0.7327 | 0.2568 |
| R2_HOP_CONDITIONED | 0.6607 | 0.9955 | 0.7312 | 0.2553 |
| RS_BEST_SINGLE (direct) | 0.6577 | 0.9970 | 0.7192 | 0.2568 |
| RX_ORACLE_READ (ceiling) | 0.6707 | 0.9985 | 0.7523 | 0.2613 |
| STEP7 consistent beam + R0_STATIC | 0.6632 | 0.9955 | 0.7342 | 0.2598 |
| STEP7 consistent beam + R1_EQUAL_RRF | 0.6632 | 0.9955 | 0.7357 | 0.2583 |
| STEP7 consistent beam + RX_ORACLE_READ | 0.6812 | 0.9985 | 0.7763 | 0.2688 |

## T7. STEP 10 -- where the read gain is lost (query buckets)

Every query lands in exactly one bucket. `need` = the gold partitions the protected core does not already hold; the swap has B = 6 slots. FREE = need is empty. CAPACITY = need > 6, dead at B = 6 whatever the ranking. NOCAND = some needed partition is not a candidate at all. LOST = every needed partition is a candidate but F6 did not pick them all. **NOCAND is the only bucket a better read can move.**

| corpus | ranking | FREE | CAPACITY | NOCAND | LOST | WON | P50 |
|---|---|---:|---:|---:|---:|---:|---:|
| metaqa | R0_STATIC (frozen) | 1294 | 246 | 311 | 120 | 27 | 0.6612 |
| metaqa | R1_EQUAL_RRF | 1294 | 246 | 309 | 121 | 28 | 0.6617 |
| metaqa | R2_HOP_CONDITIONED | 1294 | 246 | 316 | 116 | 26 | 0.6607 |
| metaqa | RS_BEST_SINGLE (direct) | 1294 | 246 | 346 | 92 | 20 | 0.6577 |
| metaqa | RX_ORACLE_READ (ceiling) | 1294 | 246 | 276 | 136 | 46 | 0.6707 |
| webqsp | R0_STATIC (frozen) | 1046 | 65 | 235 | 34 | 39 | 0.7646 |
| webqsp | R1_EQUAL_RRF | 1046 | 65 | 233 | 31 | 44 | 0.7681 |
| webqsp | R2_HOP_CONDITIONED | 1046 | 65 | 232 | 36 | 40 | 0.7653 |
| webqsp | RS_BEST_SINGLE (direct) | 1046 | 65 | 237 | 25 | 46 | 0.7696 |
| webqsp | RX_ORACLE_READ (ceiling) | 1046 | 65 | 222 | 32 | 54 | 0.7752 |
| 2wiki_clean | R0_STATIC (frozen) | 1862 | 0 | 97 | 16 | 25 | 0.9435 |
| 2wiki_clean | R1_EQUAL_RRF | 1862 | 0 | 98 | 16 | 24 | 0.9430 |
| 2wiki_clean | R2_HOP_CONDITIONED | 1862 | 0 | 95 | 19 | 24 | 0.9430 |
| 2wiki_clean | RS_BEST_SINGLE (direct) | 1862 | 0 | 97 | 18 | 23 | 0.9425 |
| 2wiki_clean | RX_ORACLE_READ (ceiling) | 1862 | 0 | 97 | 16 | 25 | 0.9435 |
| musique_clean | R0_STATIC (frozen) | 1878 | 0 | 50 | 23 | 49 | 0.9635 |
| musique_clean | R1_EQUAL_RRF | 1878 | 0 | 54 | 17 | 51 | 0.9645 |
| musique_clean | R2_HOP_CONDITIONED | 1878 | 0 | 56 | 19 | 47 | 0.9625 |
| musique_clean | RS_BEST_SINGLE (direct) | 1878 | 0 | 58 | 19 | 45 | 0.9615 |
| musique_clean | RX_ORACLE_READ (ceiling) | 1878 | 0 | 50 | 22 | 50 | 0.9640 |
| hotpotqa_clean | R0_STATIC (frozen) | 1848 | 0 | 71 | 28 | 53 | 0.9505 |
| hotpotqa_clean | R1_EQUAL_RRF | 1848 | 0 | 69 | 29 | 54 | 0.9510 |
| hotpotqa_clean | R2_HOP_CONDITIONED | 1848 | 0 | 69 | 34 | 49 | 0.9485 |
| hotpotqa_clean | RS_BEST_SINGLE (direct) | 1848 | 0 | 69 | 33 | 50 | 0.9490 |
| hotpotqa_clean | RX_ORACLE_READ (ceiling) | 1848 | 0 | 69 | 28 | 55 | 0.9515 |
| squad_clean | R0_STATIC (frozen) | 1951 | 0 | 11 | 14 | 24 | 0.9875 |
| squad_clean | R1_EQUAL_RRF | 1951 | 0 | 11 | 13 | 25 | 0.9880 |
| squad_clean | R2_HOP_CONDITIONED | 1951 | 0 | 12 | 17 | 20 | 0.9855 |
| squad_clean | RS_BEST_SINGLE (direct) | 1951 | 0 | 14 | 10 | 25 | 0.9880 |
| squad_clean | RX_ORACLE_READ (ceiling) | 1951 | 0 | 10 | 15 | 24 | 0.9875 |

MetaQA per hop, frozen vs the oracle read:

| hop | ranking | FREE | CAPACITY | NOCAND | LOST | WON | mean need | P50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hop1 | R0_STATIC (frozen) | 663 | 0 | 2 | 1 | 0 | 0.01 | 0.9955 |
| hop1 | RX_ORACLE_READ (ceiling) | 663 | 0 | 1 | 0 | 2 | 0.01 | 0.9985 |
| hop2 | R0_STATIC (frozen) | 468 | 33 | 98 | 49 | 18 | 1.42 | 0.7297 |
| hop2 | RX_ORACLE_READ (ceiling) | 468 | 33 | 73 | 59 | 33 | 1.42 | 0.7523 |
| hop3 | R0_STATIC (frozen) | 163 | 213 | 211 | 70 | 9 | 7.78 | 0.2583 |
| hop3 | RX_ORACLE_READ (ceiling) | 163 | 213 | 202 | 77 | 11 | 7.78 | 0.2613 |

## T8. STEP 10 -- evidence available for the partitions the swap must supply

| corpus | ranking | needed | in SF | in RF | incumbent | **NO evidence** | picked |
|---|---|---:|---:|---:|---:|---:|---:|
| metaqa | R0_STATIC (frozen) | 6132 | 1617 | 272 | 243 | **4190** | 317 |
| metaqa | R1_EQUAL_RRF | 6132 | 1571 | 272 | 243 | **4239** | 335 |
| metaqa | R2_HOP_CONDITIONED | 6132 | 1364 | 272 | 243 | **4437** | 290 |
| metaqa | RS_BEST_SINGLE (direct) | 6132 | 1113 | 272 | 243 | **4665** | 235 |
| metaqa | RX_ORACLE_READ (ceiling) | 6132 | 1867 | 272 | 243 | **3958** | 415 |
| webqsp | R0_STATIC (frozen) | 1811 | 150 | 44 | 125 | **1535** | 126 |
| webqsp | R1_EQUAL_RRF | 1811 | 164 | 44 | 125 | **1525** | 132 |
| webqsp | R2_HOP_CONDITIONED | 1811 | 160 | 44 | 125 | **1527** | 124 |
| webqsp | RS_BEST_SINGLE (direct) | 1811 | 142 | 44 | 125 | **1547** | 125 |
| webqsp | RX_ORACLE_READ (ceiling) | 1811 | 209 | 44 | 125 | **1488** | 165 |
| 2wiki_clean | R0_STATIC (frozen) | 147 | 13 | 33 | 16 | **98** | 30 |
| 2wiki_clean | R1_EQUAL_RRF | 147 | 12 | 33 | 16 | **99** | 29 |
| 2wiki_clean | R2_HOP_CONDITIONED | 147 | 14 | 33 | 16 | **96** | 29 |
| 2wiki_clean | RS_BEST_SINGLE (direct) | 147 | 12 | 33 | 16 | **98** | 28 |
| 2wiki_clean | RX_ORACLE_READ (ceiling) | 147 | 13 | 33 | 16 | **98** | 30 |
| musique_clean | R0_STATIC (frozen) | 128 | 36 | 42 | 38 | **51** | 53 |
| musique_clean | R1_EQUAL_RRF | 128 | 24 | 42 | 38 | **55** | 55 |
| musique_clean | R2_HOP_CONDITIONED | 128 | 22 | 42 | 38 | **57** | 51 |
| musique_clean | RS_BEST_SINGLE (direct) | 128 | 18 | 42 | 38 | **59** | 49 |
| musique_clean | RX_ORACLE_READ (ceiling) | 128 | 38 | 42 | 38 | **51** | 54 |
| hotpotqa_clean | R0_STATIC (frozen) | 161 | 25 | 78 | 23 | **74** | 57 |
| hotpotqa_clean | R1_EQUAL_RRF | 161 | 26 | 78 | 23 | **72** | 58 |
| hotpotqa_clean | R2_HOP_CONDITIONED | 161 | 23 | 78 | 23 | **72** | 52 |
| hotpotqa_clean | RS_BEST_SINGLE (direct) | 161 | 27 | 78 | 23 | **72** | 52 |
| hotpotqa_clean | RX_ORACLE_READ (ceiling) | 161 | 31 | 78 | 23 | **72** | 59 |
| squad_clean | R0_STATIC (frozen) | 49 | 11 | 32 | 10 | **11** | 24 |
| squad_clean | R1_EQUAL_RRF | 49 | 9 | 32 | 10 | **11** | 25 |
| squad_clean | R2_HOP_CONDITIONED | 49 | 8 | 32 | 10 | **12** | 20 |
| squad_clean | RS_BEST_SINGLE (direct) | 49 | 6 | 32 | 10 | **14** | 25 |
| squad_clean | RX_ORACLE_READ (ceiling) | 49 | 12 | 32 | 10 | **10** | 24 |

## T9. STEP 10 -- READ_EFFICIENCY = gold read / gold in scope

| corpus | frozen | R1 | R2 | RS | oracle |
|---|---:|---:|---:|---:|---:|
| metaqa | 0.4118 | 0.5967 | 0.6483 | 0.6361 | 0.8959 |
| webqsp | 0.3996 | 0.6402 | 0.6284 | 0.7286 | 0.8410 |
| 2wiki_clean | 0.0826 | 0.2574 | 0.2849 | 0.2829 | 0.2856 |
| musique_clean | 0.1108 | 0.1413 | 0.1275 | 0.1346 | 0.1770 |
| hotpotqa_clean | 0.0839 | 0.1765 | 0.1893 | 0.1986 | 0.2046 |
| squad_clean | 0.0214 | 0.0209 | 0.0181 | 0.0203 | 0.0291 |

## T10. STEP 7 -- the same ranking inside the beam

R1 as the beam survival key AND the `added`/read key: one semantics at every stage, which is what STEP 7 requires. Parity against frozen is broken by design here; the question is whether a better-separating key buys reach.

| corpus | frozen parity | gold discovered frozen -> beam-R1 | P50 frozen | P50 beam-R1 | net | p | oracle on beam-R1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| metaqa | 12/1998 | 0.2902 -> 0.3261 | 0.6612 | 0.6632 | +4 | 0.541 | 0.6812 |
| webqsp | 64/1419 | 0.2734 -> 0.3048 | 0.7646 | 0.7660 | +2 | 0.791 | 0.7794 |
| 2wiki_clean | 448/2000 | 0.2596 -> 0.2600 | 0.9435 | 0.9430 | -1 | 1 | 0.9435 |
| musique_clean | 48/2000 | 0.1400 -> 0.1423 | 0.9635 | 0.9625 | -2 | 0.727 | 0.9610 |
| hotpotqa_clean | 5/2000 | 0.1872 -> 0.1958 | 0.9505 | 0.9505 | +0 | 1 | 0.9510 |
| squad_clean | 295/2000 | 0.0265 -> 0.0240 | 0.9875 | 0.9875 | +0 | 1 | 0.9875 |

## T11. Does read separability predict the objective?

18 inference-safe cells (6 corpora x 3 rankings), read@64 lift against exact-P50 delta:

* pearson r = **+0.0419**, spearman = **+0.0320**
* mean read lift **+0.0740** (max +0.1839), mean exact-P50 delta **-0.00013**
* 15/18 cells improve the read; of those, 6 improve exact-P50

| ranking | macro delta | worst corpus | net total | corpora improved | significant |
|---|---:|---:|---:|---:|---|
| R1_EQUAL_RRF | +0.00092 | -0.0005 | +9 | 5/6 | none |
| R2_HOP_CONDITIONED | -0.00088 | -0.0020 | -11 | 1/6 | none |
| RS_BEST_SINGLE (direct) | -0.00042 | -0.0035 | -8 | 2/6 | none |
| RX_ORACLE_READ (ceiling) | +0.00360 | +0.0000 | +37 | 4/6 | metaqa, webqsp |
