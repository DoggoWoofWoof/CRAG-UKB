# L1 BEAM RECOVERY -- tables

All values are read directly out of `diag/*.json`.  `SAFE` is the frozen incumbent
(N0_STATIC beam 64 -> S4 -> F6 -> exact P50).

## T1  STEP 1 -- baseline parity gate

| dataset | queries | all 4 frozen fields | S4 ranking | SAFE ALL | replay ALL | verdict |
|---|---|---|---|---|---|---|
| metaqa | 1998 | 1998/1998 | 1998/1998 | 0.6612 | 0.6612 | EXACT |
| webqsp | 1419 | 1419/1419 | 1419/1419 | 0.7646 | 0.7646 | EXACT |
| 2wiki_clean | 2000 | 2000/2000 | 2000/2000 | 0.9435 | 0.9435 | EXACT |
| musique_clean | 2000 | 2000/2000 | 2000/2000 | 0.9635 | 0.9635 | EXACT |
| hotpotqa_clean | 2000 | 2000/2000 | 2000/2000 | 0.9505 | 0.9505 | EXACT |
| squad_clean | 2000 | 2000/2000 | 2000/2000 | 0.9875 | 0.9875 | EXACT |

## T2  Where the beam actually prunes (MetaQA, measured on the exact replay)

| path position | frontier in | distinct fresh candidates | kept (beam) | prune ratio |
|---|---|---|---|---|
| 1 | 5.0 | 25.2 | 21.5 | 1.2x |
| 2 | 21.5 | 438.8 | 61.8 | 7.1x |
| 3 | 61.8 | 778.3 | 63.8 | 12.2x |

The directive's premise put position 2 at ~2,560 candidates.  That figure came from an
estimate in the previous phase report (64 frontier x ~40 edges), not from a measurement.
Measured, position 1 does not prune at all, position 2 prunes ~7x, and position 3 is the
*harder* prune at ~12x.  Mean legal out-degree is ~17, not ~40.

## T3  STEP 3 -- tie mass available to a lexicographic rule (MetaQA, position 2)

| quantity | value |
|---|---|
| position-2 candidates scored | 870,793 |
| no legal fresh child (future = sentinel) | 17,420 (2.00%) |
| candidates tied in the FUTURE key | 17,220 (1.98%) |
| candidates tied in the CURRENT key | 10 (0.00%) |
| sentinel candidates entering beam 64 | 776 (0.39/query) |

| lexicographic rule | identical node list to its primary key | fraction |
|---|---|---|
| L3_FUTURE_THEN_CURRENT | 1983/1998 | 0.9925 |
| L4_CURRENT_THEN_FUTURE | 1998/1998 | 1.0000 |

## T4  STEP 7 -- position-2 diagnostic (MetaQA ALL)

| policy | cands @p2 | gold cand survived | gold cand survival | gold parent survived | gold parent survival | gold cand pctrank | gold parent pctrank |
|---|---|---|---|---|---|---|---|
| M0_BASELINE | 438.4 | 1178/3136 | 0.3756 | 2201/14474 | 0.1521 | 0.3376 | 0.3866 |
| L1_MAX_FUTURE | 435.8 | 636/3150 | 0.2019 | 2933/14440 | 0.2031 | 0.6223 | 0.3126 |
| L2_TOP2_FUTURE | 437.1 | 657/3148 | 0.2087 | 2852/14439 | 0.1975 | 0.6076 | 0.3226 |
| L3_FUTURE_THEN_CURRENT | 435.8 | 635/3150 | 0.2016 | 2933/14440 | 0.2031 | 0.6223 | 0.3126 |
| L4_CURRENT_THEN_FUTURE | 438.4 | 1178/3136 | 0.3756 | 2201/14474 | 0.1521 | 0.3376 | 0.3866 |
| D0_DELAYED_PRUNE | 438.4 | 3136/3136 | 1.0000 | 14474/14474 | 1.0000 | 0.3376 | 0.3866 |
| C1_ENDPOINT_DISPLACEMENT | 438.4 | 976/3136 | 0.3112 | 1553/14474 | 0.1073 | 0.4452 | 0.5058 |
| C2_NORMALIZED_EDGE_SUM | 438.4 | 950/3136 | 0.3029 | 1538/14474 | 0.1063 | 0.4609 | 0.5080 |
| B1_PARENT_DIVERSE | 438.5 | 1313/3138 | 0.4184 | 2250/14470 | 0.1555 | 0.3061 | 0.3819 |
| B2_SEED_DIVERSE | 438.5 | 1198/3138 | 0.3818 | 2201/14470 | 0.1521 | 0.3322 | 0.3865 |

## T4.hop2  STEP 7 -- position-2 diagnostic (MetaQA hop2)

| policy | cands @p2 | gold cand survived | gold cand survival | gold parent survived | gold parent survival | gold cand pctrank | gold parent pctrank |
|---|---|---|---|---|---|---|---|
| M0_BASELINE | 397.1 | 560/2242 | 0.2498 | 323/1209 | 0.2672 | 0.4782 | 0.2534 |
| L1_MAX_FUTURE | 397.1 | 591/2242 | 0.2636 | 162/1194 | 0.1357 | 0.4613 | 0.4041 |
| L2_TOP2_FUTURE | 398.5 | 605/2242 | 0.2698 | 179/1193 | 0.1500 | 0.4509 | 0.4532 |
| L3_FUTURE_THEN_CURRENT | 397.1 | 590/2242 | 0.2632 | 162/1194 | 0.1357 | 0.4613 | 0.4041 |
| L4_CURRENT_THEN_FUTURE | 397.1 | 560/2242 | 0.2498 | 323/1209 | 0.2672 | 0.4782 | 0.2534 |
| D0_DELAYED_PRUNE | 397.1 | 2242/2242 | 1.0000 | 1209/1209 | 1.0000 | 0.4782 | 0.2534 |
| C1_ENDPOINT_DISPLACEMENT | 397.1 | 457/2242 | 0.2038 | 311/1209 | 0.2572 | 0.5995 | 0.3123 |
| C2_NORMALIZED_EDGE_SUM | 397.1 | 432/2242 | 0.1927 | 299/1209 | 0.2473 | 0.6130 | 0.3134 |
| B1_PARENT_DIVERSE | 397.0 | 664/2243 | 0.2960 | 298/1207 | 0.2469 | 0.4541 | 0.2609 |
| B2_SEED_DIVERSE | 397.0 | 576/2243 | 0.2568 | 323/1207 | 0.2676 | 0.4707 | 0.2545 |

## T4.hop3  STEP 7 -- position-2 diagnostic (MetaQA hop3)

| policy | cands @p2 | gold cand survived | gold cand survival | gold parent survived | gold parent survival | gold cand pctrank | gold parent pctrank |
|---|---|---|---|---|---|---|---|
| M0_BASELINE | 533.2 | 618/893 | 0.6920 | 1878/13262 | 0.1416 | 0.0504 | 0.4006 |
| L1_MAX_FUTURE | 531.3 | 45/906 | 0.0497 | 2763/13229 | 0.2089 | 0.9131 | 0.3068 |
| L2_TOP2_FUTURE | 531.5 | 52/905 | 0.0575 | 2665/13227 | 0.2015 | 0.9057 | 0.3154 |
| L3_FUTURE_THEN_CURRENT | 531.3 | 45/906 | 0.0497 | 2763/13229 | 0.2089 | 0.9117 | 0.3068 |
| L4_CURRENT_THEN_FUTURE | 533.2 | 618/893 | 0.6920 | 1878/13262 | 0.1416 | 0.0504 | 0.4006 |
| D0_DELAYED_PRUNE | 533.2 | 893/893 | 1.0000 | 13262/13262 | 1.0000 | 0.0504 | 0.4006 |
| C1_ENDPOINT_DISPLACEMENT | 533.2 | 518/893 | 0.5801 | 1242/13262 | 0.0937 | 0.1031 | 0.5212 |
| C2_NORMALIZED_EDGE_SUM | 533.2 | 517/893 | 0.5789 | 1239/13262 | 0.0934 | 0.1064 | 0.5222 |
| B1_PARENT_DIVERSE | 533.2 | 649/894 | 0.7260 | 1952/13262 | 0.1472 | 0.0448 | 0.3938 |
| B2_SEED_DIVERSE | 533.2 | 622/894 | 0.6957 | 1878/13262 | 0.1416 | 0.0516 | 0.4000 |

## T5  STEP 4 -- the three cuts after discovery (MetaQA ALL)

| policy | nodes visited/q | edges/q | gold DISCOVERED | gold in added(256) | gold READ by S4(64) | gold total | ms/q |
|---|---|---|---|---|---|---|---|
| M0_BASELINE | 152.0 | 1505.8 | 0.3240 | 0.2902 | 0.1334 | 14878 | 13.34 |
| D0_DELAYED_PRUNE | 3098.6 | 5077.1 | 0.8547 | 0.2335 | 0.1139 | 14878 | 40.63 |
| L1_MAX_FUTURE | 152.0 | 6478.8 | 0.3035 | 0.2698 | 0.1038 | 14878 | 76.33 |

## T5.hop2  STEP 4 -- the three cuts after discovery (MetaQA hop2)

| policy | nodes visited/q | edges/q | gold DISCOVERED | gold in added(256) | gold READ by S4(64) | gold total | ms/q |
|---|---|---|---|---|---|---|---|
| M0_BASELINE | 152.0 | 1505.8 | 0.4354 | 0.4330 | 0.1134 | 3861 | 13.34 |
| D0_DELAYED_PRUNE | 3098.6 | 5077.1 | 0.8946 | 0.1072 | 0.0578 | 3861 | 40.63 |
| L1_MAX_FUTURE | 152.0 | 6478.8 | 0.4494 | 0.4470 | 0.0490 | 3861 | 76.33 |

## T5.hop3  STEP 4 -- the three cuts after discovery (MetaQA hop3)

| policy | nodes visited/q | edges/q | gold DISCOVERED | gold in added(256) | gold READ by S4(64) | gold total | ms/q |
|---|---|---|---|---|---|---|---|
| M0_BASELINE | 152.0 | 1505.8 | 0.2062 | 0.1987 | 0.1469 | 9718 | 13.34 |
| D0_DELAYED_PRUNE | 3098.6 | 5077.1 | 0.8360 | 0.3081 | 0.1502 | 9718 | 40.63 |
| L1_MAX_FUTURE | 152.0 | 6478.8 | 0.1710 | 0.1635 | 0.1370 | 9718 | 76.33 |

## T6  STEP 8 -- EXACT P50 end-to-end (MetaQA)

SAFE = ALL 0.6612 / hop1 0.9955 / hop2 0.7297 / hop3 0.2583

| policy | ALL | hop1 | hop2 | hop3 | gained/lost | net | McNemar p | sig |
|---|---|---|---|---|---|---|---|---|
| M0_BASELINE | 0.6612 | 0.9955 | 0.7297 | 0.2583 | 0/0 | +0 | 1 | no |
| L1_MAX_FUTURE | 0.6567 | 0.9955 | 0.7237 | 0.2508 | 8/17 | -9 | 0.108 | no |
| L2_TOP2_FUTURE | 0.6567 | 0.9955 | 0.7222 | 0.2523 | 8/17 | -9 | 0.108 | no |
| L3_FUTURE_THEN_CURRENT | 0.6567 | 0.9955 | 0.7237 | 0.2508 | 8/17 | -9 | 0.108 | no |
| L4_CURRENT_THEN_FUTURE | 0.6612 | 0.9955 | 0.7297 | 0.2583 | 0/0 | +0 | 1 | no |
| D0_DELAYED_PRUNE | 0.6582 | 0.9955 | 0.7252 | 0.2538 | 8/14 | -6 | 0.286 | no |
| C1_ENDPOINT_DISPLACEMENT | 0.6602 | 0.9955 | 0.7297 | 0.2553 | 8/10 | -2 | 0.815 | no |
| C2_NORMALIZED_EDGE_SUM | 0.6597 | 0.9955 | 0.7267 | 0.2568 | 9/12 | -3 | 0.664 | no |
| B1_PARENT_DIVERSE | 0.6622 | 0.9955 | 0.7327 | 0.2583 | 3/1 | +2 | 0.625 | no |
| B2_SEED_DIVERSE | 0.6612 | 0.9955 | 0.7297 | 0.2583 | 0/0 | +0 | 1 | no |

## T7  STEP 9 -- internal routing work vs final downstream scope (MetaQA)

| policy | graph edges/q | lookahead edges/q | total edges/q | distinct nodes evaluated/q | beam scope nodes/q | ms/q | FINAL partitions | FINAL scope nodes/q |
|---|---|---|---|---|---|---|---|---|
| M0_BASELINE | 1,506 | 0 | 1,506 | 1,241 | 152 | 35.1 | 50 | 5,036 |
| L1_MAX_FUTURE | 1,379 | 5,100 | 6,479 | 1,097 | 152 | 142.1 | 50 | 5,036 |
| L2_TOP2_FUTURE | 1,472 | 5,114 | 6,586 | 1,175 | 152 | 149.6 | 50 | 5,037 |
| L3_FUTURE_THEN_CURRENT | 1,379 | 5,100 | 6,479 | 1,097 | 152 | 139.7 | 50 | 5,036 |
| L4_CURRENT_THEN_FUTURE | 1,506 | 5,076 | 6,581 | 1,241 | 152 | 135.0 | 50 | 5,036 |
| D0_DELAYED_PRUNE | 5,077 | 0 | 5,077 | 3,097 | 3,099 | 74.8 | 50 | 5,036 |
| C1_ENDPOINT_DISPLACEMENT | 1,410 | 0 | 1,410 | 1,158 | 152 | 42.7 | 50 | 5,036 |
| C2_NORMALIZED_EDGE_SUM | 1,380 | 0 | 1,380 | 1,129 | 152 | 47.4 | 50 | 5,036 |
| B1_PARENT_DIVERSE | 1,520 | 0 | 1,520 | 1,258 | 152 | 14.7 | 50 | 5,036 |
| B2_SEED_DIVERSE | 1,506 | 0 | 1,506 | 1,242 | 152 | 14.7 | 50 | 5,036 |

Internal work is what the router touches while deciding.  Final scope is what L2 sees.
The output contract is unchanged at exactly 50 partitions for every row.

## T9  Transition table -- where a REQUIRED gold node is lost (MetaQA ALL)

Fraction of the 14,878 required gold nodes over 1,998 queries still in play at each stage of the structural search.
This funnel is nested, so it is monotone by construction.

| policy | reachable | alive p1 | alive p2 | alive p3 | in added(256) | read by S4(64) |
|---|---|---|---|---|---|---|
| M0_BASELINE | 0.8556 | 0.8548 | 0.3954 | 0.3240 | 0.2902 | 0.1334 |
| L1_MAX_FUTURE | 0.8556 | 0.8547 | 0.3726 | 0.3035 | 0.2698 | 0.1038 |
| L2_TOP2_FUTURE | 0.8556 | 0.8546 | 0.3849 | 0.3045 | 0.2707 | 0.1028 |
| L3_FUTURE_THEN_CURRENT | 0.8556 | 0.8547 | 0.3725 | 0.3035 | 0.2697 | 0.1038 |
| L4_CURRENT_THEN_FUTURE | 0.8556 | 0.8548 | 0.3954 | 0.3240 | 0.2902 | 0.1334 |
| D0_DELAYED_PRUNE | 0.8556 | 0.8548 | 0.8547 | 0.8547 | 0.2335 | 0.1139 |
| C1_ENDPOINT_DISPLACEMENT | 0.8556 | 0.8548 | 0.3481 | 0.2793 | 0.2456 | 0.1124 |
| C2_NORMALIZED_EDGE_SUM | 0.8556 | 0.8548 | 0.3457 | 0.2795 | 0.2458 | 0.1119 |
| B1_PARENT_DIVERSE | 0.8556 | 0.8548 | 0.4177 | 0.3310 | 0.2973 | 0.1400 |
| B2_SEED_DIVERSE | 0.8556 | 0.8548 | 0.3974 | 0.3256 | 0.2919 | 0.1338 |

The two partition rows are NOT nested inside that funnel -- a gold partition can
also be reached through a different gold node, or through the canonical and
retrieval channels, which is why they sit above the node rows.  The last column is
the query-level all-gold metric, i.e. exactly the reported EXACT-P50 number.

| policy | partition in S4 | partition in P50 | EXACT P50 (queries) |
|---|---|---|---|
| M0_BASELINE | 0.4018 | 0.1881 | 0.6612 |
| L1_MAX_FUTURE | 0.3473 | 0.1875 | 0.6567 |
| L2_TOP2_FUTURE | 0.3408 | 0.1875 | 0.6567 |
| L3_FUTURE_THEN_CURRENT | 0.3473 | 0.1875 | 0.6567 |
| L4_CURRENT_THEN_FUTURE | 0.4018 | 0.1881 | 0.6612 |
| D0_DELAYED_PRUNE | 0.3708 | 0.1877 | 0.6582 |
| C1_ENDPOINT_DISPLACEMENT | 0.3604 | 0.1860 | 0.6602 |
| C2_NORMALIZED_EDGE_SUM | 0.3560 | 0.1864 | 0.6597 |
| B1_PARENT_DIVERSE | 0.4250 | 0.1889 | 0.6622 |
| B2_SEED_DIVERSE | 0.4019 | 0.1881 | 0.6612 |

## T9.hop2  Transition table -- where a REQUIRED gold node is lost (MetaQA hop2)

Fraction of the 3,861 required gold nodes over 666 queries still in play at each stage of the structural search.
This funnel is nested, so it is monotone by construction.

| policy | reachable | alive p1 | alive p2 | alive p3 | in added(256) | read by S4(64) |
|---|---|---|---|---|---|---|
| M0_BASELINE | 0.8951 | 0.8948 | 0.4900 | 0.4354 | 0.4330 | 0.1134 |
| L1_MAX_FUTURE | 0.8951 | 0.8951 | 0.4556 | 0.4494 | 0.4470 | 0.0490 |
| L2_TOP2_FUTURE | 0.8951 | 0.8951 | 0.4639 | 0.4540 | 0.4517 | 0.0492 |
| L3_FUTURE_THEN_CURRENT | 0.8951 | 0.8951 | 0.4553 | 0.4491 | 0.4468 | 0.0490 |
| L4_CURRENT_THEN_FUTURE | 0.8951 | 0.8948 | 0.4900 | 0.4354 | 0.4330 | 0.1134 |
| D0_DELAYED_PRUNE | 0.8951 | 0.8948 | 0.8946 | 0.8946 | 0.1072 | 0.0578 |
| C1_ENDPOINT_DISPLACEMENT | 0.8951 | 0.8948 | 0.4693 | 0.4118 | 0.4095 | 0.1303 |
| C2_NORMALIZED_EDGE_SUM | 0.8951 | 0.8948 | 0.4600 | 0.4069 | 0.4046 | 0.1285 |
| B1_PARENT_DIVERSE | 0.8951 | 0.8948 | 0.5123 | 0.4623 | 0.4600 | 0.1287 |
| B2_SEED_DIVERSE | 0.8951 | 0.8948 | 0.4944 | 0.4395 | 0.4372 | 0.1140 |

The two partition rows are NOT nested inside that funnel -- a gold partition can
also be reached through a different gold node, or through the canonical and
retrieval channels, which is why they sit above the node rows.  The last column is
the query-level all-gold metric, i.e. exactly the reported EXACT-P50 number.

| policy | partition in S4 | partition in P50 | EXACT P50 (queries) |
|---|---|---|---|
| M0_BASELINE | 0.4893 | 0.3271 | 0.7297 |
| L1_MAX_FUTURE | 0.3947 | 0.3258 | 0.7237 |
| L2_TOP2_FUTURE | 0.3807 | 0.3253 | 0.7222 |
| L3_FUTURE_THEN_CURRENT | 0.3947 | 0.3258 | 0.7237 |
| L4_CURRENT_THEN_FUTURE | 0.4893 | 0.3271 | 0.7297 |
| D0_DELAYED_PRUNE | 0.4268 | 0.3261 | 0.7252 |
| C1_ENDPOINT_DISPLACEMENT | 0.4789 | 0.3199 | 0.7297 |
| C2_NORMALIZED_EDGE_SUM | 0.4657 | 0.3206 | 0.7267 |
| B1_PARENT_DIVERSE | 0.5258 | 0.3294 | 0.7327 |
| B2_SEED_DIVERSE | 0.4905 | 0.3271 | 0.7297 |

## T9.hop3  Transition table -- where a REQUIRED gold node is lost (MetaQA hop3)

Fraction of the 9,718 required gold nodes over 666 queries still in play at each stage of the structural search.
This funnel is nested, so it is monotone by construction.

| policy | reachable | alive p1 | alive p2 | alive p3 | in added(256) | read by S4(64) |
|---|---|---|---|---|---|---|
| M0_BASELINE | 0.8376 | 0.8360 | 0.2939 | 0.2062 | 0.1987 | 0.1469 |
| L1_MAX_FUTURE | 0.8376 | 0.8363 | 0.2741 | 0.1710 | 0.1635 | 0.1370 |
| L2_TOP2_FUTURE | 0.8376 | 0.8362 | 0.2896 | 0.1706 | 0.1631 | 0.1355 |
| L3_FUTURE_THEN_CURRENT | 0.8376 | 0.8363 | 0.2741 | 0.1710 | 0.1635 | 0.1370 |
| L4_CURRENT_THEN_FUTURE | 0.8376 | 0.8360 | 0.2939 | 0.2062 | 0.1987 | 0.1469 |
| D0_DELAYED_PRUNE | 0.8376 | 0.8360 | 0.8360 | 0.8360 | 0.3081 | 0.1502 |
| C1_ENDPOINT_DISPLACEMENT | 0.8376 | 0.8360 | 0.2296 | 0.1471 | 0.1396 | 0.1023 |
| C2_NORMALIZED_EDGE_SUM | 0.8376 | 0.8360 | 0.2297 | 0.1494 | 0.1419 | 0.1021 |
| B1_PARENT_DIVERSE | 0.8376 | 0.8361 | 0.3191 | 0.2062 | 0.1987 | 0.1471 |
| B2_SEED_DIVERSE | 0.8376 | 0.8361 | 0.2950 | 0.2070 | 0.1995 | 0.1473 |

The two partition rows are NOT nested inside that funnel -- a gold partition can
also be reached through a different gold node, or through the canonical and
retrieval channels, which is why they sit above the node rows.  The last column is
the query-level all-gold metric, i.e. exactly the reported EXACT-P50 number.

| policy | partition in S4 | partition in P50 | EXACT P50 (queries) |
|---|---|---|---|
| M0_BASELINE | 0.3419 | 0.0326 | 0.2583 |
| L1_MAX_FUTURE | 0.3225 | 0.0322 | 0.2508 |
| L2_TOP2_FUTURE | 0.3182 | 0.0324 | 0.2523 |
| L3_FUTURE_THEN_CURRENT | 0.3225 | 0.0322 | 0.2508 |
| L4_CURRENT_THEN_FUTURE | 0.3419 | 0.0326 | 0.2583 |
| D0_DELAYED_PRUNE | 0.3386 | 0.0325 | 0.2538 |
| C1_ENDPOINT_DISPLACEMENT | 0.2877 | 0.0323 | 0.2553 |
| C2_NORMALIZED_EDGE_SUM | 0.2853 | 0.0326 | 0.2568 |
| B1_PARENT_DIVERSE | 0.3567 | 0.0329 | 0.2583 |
| B2_SEED_DIVERSE | 0.3412 | 0.0326 | 0.2583 |

### Conservation of loss -- where each cut spends it (MetaQA ALL)

| transition | M0_BASELINE | B1_PARENT_DIVERSE | L1_MAX_FUTURE | D0_DELAYED_PRUNE |
|---|---|---|---|---|
| reachable -> alive p1 | 0.0008 (0%) | 0.0008 (0%) | 0.0009 (0%) | 0.0008 (0%) |
| alive p1 -> alive p2 | 0.4594 (64%) | 0.4371 (61%) | 0.4821 (64%) | 0.0001 (0%) |
| alive p2 -> alive p3 | 0.0714 (10%) | 0.0867 (12%) | 0.0691 (9%) | 0.0000 (0%) |
| alive p3 -> in added(256) | 0.0338 (5%) | 0.0337 (5%) | 0.0337 (4%) | 0.6212 (84%) |
| in added(256) -> read by S4(64) | 0.1568 (22%) | 0.1573 (22%) | 0.1660 (22%) | 0.1196 (16%) |
| TOTAL structural loss | 0.7222 | 0.7156 | 0.7518 | 0.7417 |

### Conservation of loss -- where each cut spends it (MetaQA hop2)

| transition | M0_BASELINE | B1_PARENT_DIVERSE | L1_MAX_FUTURE | D0_DELAYED_PRUNE |
|---|---|---|---|---|
| reachable -> alive p1 | 0.0003 (0%) | 0.0003 (0%) | 0.0000 (0%) | 0.0003 (0%) |
| alive p1 -> alive p2 | 0.4048 (52%) | 0.3825 (50%) | 0.4395 (52%) | 0.0002 (0%) |
| alive p2 -> alive p3 | 0.0546 (7%) | 0.0500 (7%) | 0.0062 (1%) | 0.0000 (0%) |
| alive p3 -> in added(256) | 0.0024 (0%) | 0.0023 (0%) | 0.0024 (0%) | 0.7874 (94%) |
| in added(256) -> read by S4(64) | 0.3196 (41%) | 0.3313 (43%) | 0.3980 (47%) | 0.0494 (6%) |
| TOTAL structural loss | 0.7817 | 0.7664 | 0.8461 | 0.8373 |

`D0_DELAYED_PRUNE` is the oracle-free control: it removes the position-2 and
position-3 prunes entirely, so its `alive p2` and `alive p3` rows are pinned to the
reachable set by construction.  It does not remove the loss -- it relocates all of it
onto the static-score-ordered `added(256)` cut, where the same score now has to
discriminate a pool an order of magnitude larger, and does so worse.

## T10  Clean latency benchmark (MetaQA)

400 queries spread across the hop blocks, 30-query warm-up, median of
3 repeats, nothing else on the CPU.  The ms/q column inside the sweep
JSONs is contaminated (jobs shared the CPU and finished partway through) and is not
reported anywhere.

| policy | ms/query | x SAFE |
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

## T8  STEP 8 -- promotion check on all six corpora (EXACT P50)

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

