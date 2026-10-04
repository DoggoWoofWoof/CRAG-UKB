# L1 BACKWARD CAUSAL OPTIMIZATION -- TABLES

All six corpora.  Frozen contract throughout: MASTER_TOPOLOGY=C, P_MAIN=50, B=6, M_struct=64,
M_ret=32, K0=60.  Replay parity EXACT on every query of every corpus.  Gold is used for
evaluation only; no method reads it.

## T1  STEP 2 -- oracle ladder (exact P50, ALL queries)

`O0` frozen SAFE.  `O1` perfect selection over the frozen candidate universe at B=6.
`O2` same universe, B=50.  `O3` perfect selection over everything VISITED at B=6.  `O4` visited,
B=50.  `O5` |REQUIRED| <= 50.

| corpus | O0 SAFE | O1 | O2 | O3 | O4 | O5 |
|---|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.7212 | 0.7222 | 0.8609 | 0.9379 | 0.9870 |
| webqsp | 0.7646 | 0.7886 | 0.7886 | 0.8739 | 0.8767 | 0.9951 |
| musique_clean | 0.9635 | 0.9750 | 0.9750 | 0.9995 | 0.9995 | 1.0000 |
| 2wiki_clean | 0.9435 | 0.9515 | 0.9515 | 0.9905 | 0.9905 | 1.0000 |
| hotpotqa_clean | 0.9505 | 0.9645 | 0.9645 | 0.9910 | 0.9910 | 1.0000 |
| squad_clean | 0.9875 | 0.9945 | 0.9945 | 1.0000 | 1.0000 | 1.0000 |

## T2  STEP 2 -- mutually exclusive loss chain  (O0 <= O1 <= O3 <= O4 <= O5, sums to 1-O0)

| corpus | SELECTION O1-O0 | CAND_UNIVERSE O3-O1 | B_CAPACITY O4-O3 | REACH O5-O4 | P50_CAPACITY 1-O5 | TOTAL 1-O0 | off-chain B50-B6 O2-O1 |
|---|---|---|---|---|---|---|---|
| metaqa | 0.0600 | 0.1397 | 0.0770 | 0.0491 | 0.0130 | 0.3388 | 0.0010 |
| webqsp | 0.0240 | 0.0853 | 0.0028 | 0.1184 | 0.0049 | 0.2354 | 0.0000 |
| musique_clean | 0.0115 | 0.0245 | 0.0000 | 0.0005 | 0.0000 | 0.0365 | 0.0000 |
| 2wiki_clean | 0.0080 | 0.0390 | 0.0000 | 0.0095 | 0.0000 | 0.0565 | 0.0000 |
| hotpotqa_clean | 0.0140 | 0.0265 | 0.0000 | 0.0090 | 0.0000 | 0.0495 | 0.0000 |
| squad_clean | 0.0070 | 0.0055 | 0.0000 | 0.0000 | 0.0000 | 0.0125 | 0.0000 |

## T3  STEP 2 -- MetaQA oracle ladder and loss chain by hop

| hop | O0 | O1 | O2 | O3 | O4 | O5 | SELECTION | CAND_UNIV | B_CAP | REACH | P50_CAP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hop1 | 0.9955 | 0.9970 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 0.0015 | 0.0030 | 0.0000 | 0.0000 | 0.0000 |
| hop2 | 0.7297 | 0.8033 | 0.8033 | 0.9414 | 0.9880 | 0.9970 | 0.0736 | 0.1381 | 0.0466 | 0.0090 | 0.0030 |
| hop3 | 0.2583 | 0.3634 | 0.3664 | 0.6411 | 0.8258 | 0.9640 | 0.1051 | 0.2777 | 0.1847 | 0.1382 | 0.0360 |
| ALL | 0.6612 | 0.7212 | 0.7222 | 0.8609 | 0.9379 | 0.9870 | 0.0600 | 0.1397 | 0.0770 | 0.0491 | 0.0130 |

## T4  STEP 1 -- backward loss ledger, % of MISSING required partitions (one primary label each)

| corpus / hop | missing/q | required/q | p50 capacity | unreachable | not in candidate universe | b capacity | pointwise ranking | redundancy | set selection | final fusion |
|---|---|---|---|---|---|---|---|---|---|---|
| metaqa hop1 | 0.01 | 1.61 | 0.0 | 0.0 | 75.0 | 0.0 | 0.0 | 0.0 | 0.0 | 25.0 |
| metaqa hop2 | 1.31 | 3.99 | 0.9 | 0.7 | 70.4 | 1.3 | 7.7 | 5.4 | 10.8 | 2.9 |
| metaqa hop3 | 7.42 | 11.75 | 7.2 | 2.9 | 63.2 | 4.5 | 6.7 | 6.2 | 7.0 | 2.2 |
| metaqa ALL | 2.91 | 5.78 | 6.2 | 2.6 | 64.3 | 4.0 | 6.9 | 6.0 | 7.6 | 2.3 |
| webqsp | 1.19 | 3.66 | 5.6 | 44.5 | 42.8 | 0.0 | 1.4 | 0.5 | 4.0 | 1.3 |
| musique_clean | 0.04 | 1.77 | 0.0 | 1.3 | 66.7 | 0.0 | 1.3 | 4.0 | 20.0 | 6.7 |
| 2wiki_clean | 0.06 | 1.98 | 0.0 | 16.2 | 68.4 | 0.0 | 4.3 | 2.6 | 6.8 | 1.7 |
| hotpotqa_clean | 0.05 | 1.66 | 0.0 | 18.3 | 52.9 | 0.0 | 3.8 | 7.7 | 8.7 | 8.7 |
| squad_clean | 0.01 | 1.00 | 0.0 | 0.0 | 44.0 | 0.0 | 0.0 | 4.0 | 32.0 | 20.0 |

## T5  STEP 1 -- secondary co-occurrence on MetaQA (rows: primary label, cells: conditions also true)

| primary label | also true |
|---|---|
| P50_CAPACITY | POINTWISE_RANKING 328, REDUNDANCY 38, SET_SELECTION 33, FINAL_FUSION 8 |
| UNREACHABLE | POINTWISE_RANKING 147, SET_SELECTION 1, FINAL_FUSION 2 |
| NOT_IN_CANDIDATE_UNIVERSE | POINTWISE_RANKING 3740, REDUNDANCY 324 |
| B_CAPACITY | POINTWISE_RANKING 178, REDUNDANCY 54, SET_SELECTION 122, FINAL_FUSION 14 |
| POINTWISE_RANKING | REDUNDANCY 102, SET_SELECTION 181 |
| REDUNDANCY | SET_SELECTION 305, FINAL_FUSION 46 |
| SET_SELECTION | - |
| FINAL_FUSION | - |

## T6  STEP 4 -- Pareto diagnostic over the frozen candidate pool (ALL)

Ten rank columns, all smaller-is-better.  A DOMINATED missing partition cannot be rescued by any
monotone rule over these columns.  `AUC front-rank` separates missing-required from nuisance.

| corpus | MISSING_GOLD_ON_PARETO_FRONT | nuisance on front | selected on front | front size/q | pool/q | AUC front-rank |
|---|---|---|---|---|---|---|
| metaqa | 0.6671 | 0.5468 | 0.9594 | 29.4 | 49.2 | 0.5661 |
| webqsp | 0.5667 | 0.2985 | 0.8423 | 13.8 | 34.4 | 0.6696 |
| musique_clean | 0.7083 | 0.5967 | 0.9475 | 13.0 | 18.6 | 0.5562 |
| 2wiki_clean | 0.6842 | 0.4165 | 0.9218 | 19.5 | 41.8 | 0.6454 |
| hotpotqa_clean | 0.6000 | 0.2980 | 0.9007 | 20.3 | 55.4 | 0.7151 |
| squad_clean | 0.6429 | 0.5962 | 0.9067 | 10.9 | 15.8 | 0.5330 |

## T7  STEP 5 -- novelty diagnostic against the current P50 (ALL)

| corpus | new atoms missing | new atoms nuisance | AUC new-atoms | AUC novelty-mass | frac missing with ZERO new atoms | frac nuisance with ZERO |
|---|---|---|---|---|---|---|
| metaqa | 0.0010 | 0.0000 | 0.5002 | 0.5657 | 0.9994 | 0.9997 |
| webqsp | 0.0000 | 0.0000 | 0.5000 | 0.4903 | 1.0000 | 1.0000 |
| musique_clean | 0.0000 | 0.0010 | 0.4995 | 0.5175 | 1.0000 | 0.9990 |
| 2wiki_clean | 0.0000 | 0.0010 | 0.4997 | 0.5625 | 1.0000 | 0.9993 |
| hotpotqa_clean | 0.0000 | 0.0010 | 0.4997 | 0.5631 | 1.0000 | 0.9994 |
| squad_clean | 0.0000 | 0.0000 | 0.5000 | 0.4984 | 1.0000 | 1.0000 |

## T8a  STEP 10 / STEP 8 -- exact P50 at B=6, DEPTH_MATCHED

| corpus | pool/q | G0_SAFE | G1_PARETO | G2_NOVELTY | G3_SUBMODULAR | G4_ASSIGNMENT |
|---|---|---|---|---|---|---|
| metaqa | 49.2 | 0.6612 | 0.6622 (+2, p=0.625) | 0.6612 (+0, p=1.000) | 0.6627 (+3, p=0.720) | 0.6652 (+8, p=0.230) |
| webqsp | 34.4 | 0.7646 | 0.7660 (+2, p=0.688) | 0.7646 (+0, p=1.000) | 0.7632 (-2, p=0.625) | 0.7632 (-2, p=0.625) |
| musique_clean | 18.6 | 0.9635 | 0.9645 (+2, p=0.625) | 0.9635 (+0, p=1.000) | 0.9600 (-7, p=0.039)  SIG | 0.9600 (-7, p=0.039)  SIG |
| 2wiki_clean | 41.8 | 0.9435 | 0.9430 (-1, p=1.000) | 0.9435 (+0, p=1.000) | 0.9435 (+0, p=1.000) | 0.9435 (+0, p=1.000) |
| hotpotqa_clean | 55.4 | 0.9505 | 0.9510 (+1, p=1.000) | 0.9505 (+0, p=1.000) | 0.9440 (-13, p=0.007)  SIG | 0.9450 (-11, p=0.027)  SIG |
| squad_clean | 15.8 | 0.9875 | 0.9865 (-2, p=0.500) | 0.9875 (+0, p=1.000) | 0.9870 (-1, p=1.000) | 0.9870 (-1, p=1.000) |

## T8b  STEP 10 / STEP 8 -- exact P50 at B=6, FULL_AVAILABLE

| corpus | pool/q | G0_SAFE | G1_PARETO | G2_NOVELTY | G3_SUBMODULAR | G4_ASSIGNMENT |
|---|---|---|---|---|---|---|
| metaqa | 329.8 | 0.6612 | 0.6622 (+2, p=0.625) | 0.6612 (+0, p=1.000) | 0.6612 (+0, p=1.000) | 0.6647 (+7, p=0.324) |
| webqsp | 715.8 | 0.7646 | 0.7660 (+2, p=0.688) | 0.7646 (+0, p=1.000) | 0.7611 (-5, p=0.180) | 0.7611 (-5, p=0.180) |
| musique_clean | 92.0 | 0.9635 | 0.9645 (+2, p=0.625) | 0.9635 (+0, p=1.000) | 0.9580 (-11, p=0.003)  SIG | 0.9580 (-11, p=0.003)  SIG |
| 2wiki_clean | 372.0 | 0.9435 | 0.9435 (+0, p=1.000) | 0.9435 (+0, p=1.000) | 0.9430 (-1, p=1.000) | 0.9430 (-1, p=1.000) |
| hotpotqa_clean | 915.0 | 0.9505 | 0.9510 (+1, p=1.000) | 0.9500 (-1, p=1.000) | 0.9345 (-32, p=0.000)  SIG | 0.9350 (-31, p=0.000)  SIG |
| squad_clean | 146.0 | 0.9875 | 0.9865 (-2, p=0.500) | 0.9875 (+0, p=1.000) | 0.9860 (-3, p=0.375) | 0.9860 (-3, p=0.375) |

## T9  STEP 10 -- MetaQA by hop, DEPTH_MATCHED, B=6

| hop | G0_SAFE | G1_PARETO | G2_NOVELTY | G3_SUBMODULAR | G4_ASSIGNMENT |
|---|---|---|---|---|---|
| hop1 | 0.9955 | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) |
| hop2 | 0.7297 | 0.7312 (+1, p=1.000) | 0.7297 (+0, p=1.000) | 0.7192 (-7, p=0.092) | 0.7222 (-5, p=0.302) |
| hop3 | 0.2583 | 0.2598 (+1, p=1.000) | 0.2583 (+0, p=1.000) | 0.2733 (+10, p=0.031)  SIG | 0.2778 (+13, p=0.004)  SIG |
| ALL | 0.6612 | 0.6622 (+2, p=0.625) | 0.6612 (+0, p=1.000) | 0.6627 (+3, p=0.720) | 0.6652 (+8, p=0.230) |

## T10  STEP 6/7 -- B50 diagnostic: all 50 chosen freely from prot | pool (ALL)

`G0_SAFE` at B50 is the plain canonical top-50, i.e. L1 with no swap at all.

| corpus | G0_SAFE | G1_PARETO | G2_NOVELTY | G3_SUBMODULAR | G4_ASSIGNMENT |
|---|---|---|---|---|---|
| metaqa | 0.6587 | 0.6627 | 0.2768 | 0.4259 | 0.6642 |
| webqsp | 0.7618 | 0.7625 | 0.5476 | 0.5765 | 0.7597 |
| musique_clean | 0.9565 | 0.9545 | 0.9395 | 0.9420 | 0.9540 |
| 2wiki_clean | 0.9375 | 0.9360 | 0.5005 | 0.7605 | 0.9385 |
| hotpotqa_clean | 0.9345 | 0.9390 | 0.5395 | 0.8280 | 0.9360 |
| squad_clean | 0.9805 | 0.9840 | 0.9755 | 0.9835 | 0.9845 |

## T11  STEP 9 -- REPAIR@k: SAFE-missing required partitions recovered in the top-k challengers

| corpus | missing | G0_SAF@6 | G0_SAF@50 | G1_PAR@6 | G1_PAR@50 | G2_NOV@6 | G2_NOV@50 | G3_SUB@6 | G3_SUB@50 | G4_ASS@6 | G4_ASS@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | 5815 | 0.0000 | 0.2696 | 0.0026 | 0.2714 | 0.0002 | 0.2696 | 0.0471 | 0.2703 | 0.0528 | 0.2703 |
| webqsp | 1685 | 0.0000 | 0.0866 | 0.0053 | 0.0872 | 0.0000 | 0.0866 | 0.0042 | 0.0866 | 0.0042 | 0.0866 |
| musique_clean | 75 | 0.0000 | 0.3200 | 0.0400 | 0.3200 | 0.0000 | 0.3200 | 0.0400 | 0.3200 | 0.0400 | 0.3200 |
| 2wiki_clean | 117 | 0.0000 | 0.1538 | 0.0000 | 0.1624 | 0.0000 | 0.1538 | 0.0256 | 0.1538 | 0.0256 | 0.1538 |
| hotpotqa_clean | 104 | 0.0000 | 0.2596 | 0.0096 | 0.2692 | 0.0000 | 0.2596 | 0.0577 | 0.2596 | 0.0673 | 0.2596 |
| squad_clean | 25 | 0.0000 | 0.5600 | 0.0000 | 0.5600 | 0.0000 | 0.5600 | 0.0400 | 0.5600 | 0.0400 | 0.5600 |

MetaQA by hop, G4_ASSIGNMENT vs SAFE:

| hop | missing | SAFE@6 | G4@6 | SAFE@20 | G4@20 | SAFE@50 | G4@50 |
|---|---|---|---|---|---|---|---|
| hop2 | 872 | 0.0000 | 0.0493 | 0.1250 | 0.1330 | 0.2752 | 0.2764 |
| hop3 | 4939 | 0.0000 | 0.0535 | 0.0978 | 0.1160 | 0.2687 | 0.2693 |
| ALL | 5815 | 0.0000 | 0.0528 | 0.1018 | 0.1185 | 0.2696 | 0.2703 |

## T12  STEP 11 -- per-query causal attribution on MetaQA (every changed query)

**G4_ASSIGNMENT** -- 0.6612 -> 0.6652, gained 21, lost 13, net 8, p=0.2295

| mechanism | count |
|---|---|
| ASSIGNMENT_DIVERSIFICATION | 8 |
| INDEPENDENT_EVIDENCE_COVERAGE | 13 |
| DISPLACED_REQUIRED_WON_NO_ATOM | 13 |

**G3_SUBMODULAR** -- 0.6612 -> 0.6627, gained 17, lost 14, net 3, p=0.7201

| mechanism | count |
|---|---|
| ASSIGNMENT_DIVERSIFICATION | 5 |
| INDEPENDENT_EVIDENCE_COVERAGE | 12 |
| DISPLACED_REQUIRED_WON_NO_ATOM | 13 |
| DISPLACED_BY_ORDER | 1 |

Evidence family behind the changed queries (MetaQA, G4_ASSIGNMENT):

| atom family | changed queries |
|---|---|
| SRC | 33 |
| CH + SRC | 1 |

Worked examples:

| q | hop | kind | mechanism | detail |
|---|---|---|---|---|
| 813 | 2 | GAIN | ASSIGNMENT_DIVERSIFICATION | {"entered": [62], "entered_required": [62], "won_families": ["SRC"], "won_mass": 0.01667, "won_atoms": 1, "safe_pick_families": [], "displaced": [278, 46, 325, 77, 36, 95], "safe_order_index": 10} |
| 846 | 2 | GAIN | INDEPENDENT_EVIDENCE_COVERAGE | {"entered": [77, 69], "entered_required": [77, 69], "won_families": ["SRC"], "won_mass": 0.03333, "won_atoms": 2, "safe_pick_families": ["SRC"], "displaced": [204, 72, 40, 315], "safe_order_index": 7} |
| 1035 | 2 | GAIN | INDEPENDENT_EVIDENCE_COVERAGE | {"entered": [377], "entered_required": [377], "won_families": ["SRC"], "won_mass": 0.01667, "won_atoms": 1, "safe_pick_families": ["CH", "SRC"], "displaced": [197, 303, 70], "safe_order_index": 8} |
| 1066 | 2 | GAIN | INDEPENDENT_EVIDENCE_COVERAGE | {"entered": [260], "entered_required": [260], "won_families": ["SRC"], "won_mass": 0.01667, "won_atoms": 1, "safe_pick_families": ["SRC"], "displaced": [226, 307, 344], "safe_order_index": 27} |
| 672 | 2 | LOSS | DISPLACED_REQUIRED_WON_NO_ATOM | {"lost_required": [315], "lost_won_mass": 0, "lost_won_atoms": 0, "lost_on_front": true, "displaced_by": [20, 15, 345], "taker_families": ["SRC"], "taker_mass": 0.01667, "safe_order_index": 1} |
| 674 | 2 | LOSS | DISPLACED_REQUIRED_WON_NO_ATOM | {"lost_required": [52, 142], "lost_won_mass": 0, "lost_won_atoms": 0, "lost_on_front": true, "displaced_by": [46, 262, 60], "taker_families": ["SRC"], "taker_mass": 0.03333, "safe_order_index": 1} |
| 713 | 2 | LOSS | DISPLACED_REQUIRED_WON_NO_ATOM | {"lost_required": [300], "lost_won_mass": 0, "lost_won_atoms": 0, "lost_on_front": false, "displaced_by": [61, 12, 27], "taker_families": ["SRC"], "taker_mass": 0.01667, "safe_order_index": 2} |
| 725 | 2 | LOSS | DISPLACED_REQUIRED_WON_NO_ATOM | {"lost_required": [319], "lost_won_mass": 0, "lost_won_atoms": 0, "lost_on_front": true, "displaced_by": [321, 318, 324], "taker_families": ["SRC"], "taker_mass": 0.01667, "safe_order_index": 4} |

## T13  STEP 12 -- latency, MetaQA, 120 queries, selection stage alone

| stage | ms / query | % of L1 |
|---|---|---|
| L1 residual (frozen) | 0.301 | - |
| L1 bounded traversal (frozen) | 15.175 | - |
| L1 SAFE selector (frozen) | 0.089 | - |
| **L1 frozen total** | 15.565 | - |
| G0_SAFE | 0.028 | 0.18% |
| G1_PARETO | 0.828 | 5.05% |
| G2_NOVELTY | 1.246 | 7.41% |
| G3_SUBMODULAR | 1.057 | 6.36% |
| G4_ASSIGNMENT | 0.099 | 0.63% |
