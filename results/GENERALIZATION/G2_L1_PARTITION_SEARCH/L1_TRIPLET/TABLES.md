# STRUCTURAL-OFFSET TRIPLET PHASE -- TABLES

Every table is measured on the frozen bounded traversal replayed bit-exactly (`want_edges` exports the per-edge quantities the search already computes; parity asserted on every query).  The target label everywhere is MARGINAL_USEFUL: a required gold partition the canonical/protected core does not already hold.  Gold is never an input to any score.

## T1. STEP 12 -- what each representation retains per query (saturation)

| corpus | needed P/q | NODE partitions reached/q | partition saturation | TRIPLET transitions/q | PATH chains/q | transitions per partition | MARGINAL_USEFUL triplet prevalence | all needed reachable as a transition target |
|---|---|---|---|---|---|---|---|---|
| metaqa | 3.069 | 305.5 | 0.7619 | 929.8 | 990.1 | 3.043 | 0.01132 | 0.6903 |
| musique_clean | 0.064 | 111.7 | 0.8216 | 403.5 | 782.7 | 3.611 | 0.00032 | 0.877 |
| 2wiki_clean | 0.073 | 167.2 | 0.2541 | 276.7 | 323.0 | 1.655 | 0.00013 | 0.3623 |
| squad_clean | 0.025 | 116.8 | 0.6147 | 420.4 | 1709.0 | 3.599 | 0.00038 | 0.6531 |

MetaQA by hop:

| block | needed P/q | NODE partitions reached/q | partition saturation | TRIPLET transitions/q | PATH chains/q | transitions per partition | MARGINAL_USEFUL triplet prevalence | all needed reachable as a transition target |
|---|---|---|---|---|---|---|---|---|
| hop1 | 0.009 | 301.2 | 0.7510 | 914.7 | 973.4 | 3.037 | 3e-05 | 1.0 |
| hop2 | 1.416 | 303.9 | 0.7578 | 933.5 | 996.7 | 3.072 | 0.00718 | 0.9444 |
| hop3 | 7.782 | 311.5 | 0.7769 | 941.3 | 1000.1 | 3.021 | 0.02648 | 0.5885 |

## T2. STEP 3 -- each triplet signal on its own: percentile rank of MARGINAL_USEFUL triplets

0.5 is chance.  No signal is combined with any other in this table.

| signal | ALL | hop1 | hop2 | hop3 |
|---|---|---|---|---|
| T0_OFFSET | 0.5323 | 0.4891 | 0.5184 | 0.5382 |
| T1_SOURCE | 0.4898 | 0.6004 | 0.4839 | 0.4915 |
| T2_TARGET | 0.5069 | 0.4467 | 0.5211 | 0.5015 |
| T3_HOP | 0.4995 | 0.5336 | 0.5414 | 0.4825 |
| C1_PATH_MIN | 0.4926 | 0.5234 | 0.479 | 0.4979 |
| C1_PATH_SUM | 0.5225 | 0.5164 | 0.4855 | 0.5374 |

## T3. STEP 4 -- needed-partition recall at a TRIPLET budget, against the NODE unit at the same budget

**MetaQA ALL**

| signal | unit | R@16 | R@32 | R@64 | R@128 |
|---|---|---|---|---|---|
| T0_OFFSET | triplet | 0.1612 | 0.2488 | 0.3427 | 0.4435 |
| T1_SOURCE | triplet | 0.0238 | 0.0798 | 0.1643 | 0.2686 |
| T2_TARGET | triplet | 0.004 | 0.0305 | 0.118 | 0.2719 |
| T3_HOP | triplet | 0.0265 | 0.067 | 0.1515 | 0.2717 |
| C1_PATH_MIN | triplet | 0.0645 | 0.1094 | 0.1898 | 0.3109 |
| C1_PATH_SUM | triplet | 0.0951 | 0.1707 | 0.2767 | 0.3927 |
| NODE_FROZEN_ARRIVAL | NODE | 0.1749 | 0.2603 | 0.3504 | 0.4583 |
| NODE_MAX_SDIR | NODE | 0.1754 | 0.2611 | 0.3506 | 0.4581 |

**MetaQA hop2**

| signal | unit | R@16 | R@32 | R@64 | R@128 |
|---|---|---|---|---|---|
| T0_OFFSET | triplet | 0.1878 | 0.2777 | 0.3837 | 0.5041 |
| T1_SOURCE | triplet | 0.0674 | 0.1629 | 0.2746 | 0.3841 |
| T2_TARGET | triplet | 0.0099 | 0.0414 | 0.1429 | 0.3422 |
| T3_HOP | triplet | 0.0778 | 0.1718 | 0.3089 | 0.452 |
| C1_PATH_MIN | triplet | 0.1237 | 0.1591 | 0.251 | 0.3563 |
| C1_PATH_SUM | triplet | 0.0702 | 0.1292 | 0.2565 | 0.3778 |
| NODE_FROZEN_ARRIVAL | NODE | 0.2134 | 0.3019 | 0.3896 | 0.5531 |
| NODE_MAX_SDIR | NODE | 0.2133 | 0.3019 | 0.3892 | 0.5529 |

**MetaQA hop3**

| signal | unit | R@16 | R@32 | R@64 | R@128 |
|---|---|---|---|---|---|
| T0_OFFSET | triplet | 0.1516 | 0.2388 | 0.3266 | 0.4203 |
| T1_SOURCE | triplet | 0.0058 | 0.0456 | 0.1179 | 0.2198 |
| T2_TARGET | triplet | 0.0016 | 0.0265 | 0.1089 | 0.2448 |
| T3_HOP | triplet | 0.0055 | 0.0242 | 0.0865 | 0.1984 |
| C1_PATH_MIN | triplet | 0.0406 | 0.0896 | 0.1648 | 0.2918 |
| C1_PATH_SUM | triplet | 0.1055 | 0.187 | 0.2844 | 0.3989 |
| NODE_FROZEN_ARRIVAL | NODE | 0.1608 | 0.2455 | 0.3351 | 0.4217 |
| NODE_MAX_SDIR | NODE | 0.1615 | 0.2466 | 0.3355 | 0.4215 |

Cross-corpus, ALL queries, best triplet signal against the node unit:

| corpus | T0 R@16 | T0 R@32 | T0 R@64 | T0 R@128 | NODE R@16 | NODE R@32 | NODE R@64 | NODE R@128 |
|---|---|---|---|---|---|---|---|---|
| metaqa | 0.1612 | 0.2488 | 0.3427 | 0.4435 | 0.1754 | 0.2611 | 0.3506 | 0.4581 |
| musique_clean | 0.0984 | 0.1393 | 0.1803 | 0.2992 | 0.123 | 0.1721 | 0.2623 | 0.3689 |
| 2wiki_clean | 0.0435 | 0.0652 | 0.087 | 0.1473 | 0.0435 | 0.0652 | 0.0894 | 0.1401 |
| squad_clean | 0.0408 | 0.0612 | 0.1224 | 0.2245 | 0.102 | 0.1633 | 0.2245 | 0.2449 |

R@64 triplet minus node: metaqa -0.0079, musique_clean -0.0820, 2wiki_clean -0.0024, squad_clean -0.1021.

## T4. STEP 5 decisive control -- is the node collapse lossy?

`np.maximum.at(best_s, inv, S)` is the collapse this phase set out to undo.  For a partition score built by MAX it is algebraically lossless, because max is associative.  Measured, not argued:

| corpus | queries | queries with no usable edge | identical score vectors | identical orderings | verdict over queries with >=1 edge |
|---|---|---|---|---|---|
| metaqa | 1998 | 0 | 1998 | 1998 | EXACT |
| musique_clean | 2000 | 19 | 1981 | 1981 | EXACT |
| 2wiki_clean | 2000 | 86 | 1914 | 1914 | EXACT |
| squad_clean | 2000 | 295 | 1705 | 1705 | EXACT |

The only aggregations under which the edge unit and the node unit differ at all are SUM and COUNT, and the source-side counts the node unit cannot express:

**metaqa (hop3)**

| aggregation | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| EDGE_MAX | 0.0906 | 0.1441 | 0.2084 | 0.3412 |
| NODE_MAX | 0.0906 | 0.1441 | 0.2084 | 0.3412 |
| EDGE_SUM | 0.052 | 0.0873 | 0.1273 | 0.2413 |
| NODE_SUM | 0.0387 | 0.0695 | 0.1117 | 0.2215 |
| EDGE_COUNT | 0.0198 | 0.0445 | 0.0867 | 0.2219 |
| NODE_COUNT | 0.0277 | 0.0526 | 0.0875 | 0.2237 |
| SOURCE_PARTITION_COUNT | 0.0142 | 0.0393 | 0.079 | 0.2202 |
| SOURCE_NODE_COUNT | 0.0142 | 0.037 | 0.0702 | 0.2224 |

**musique_clean (ALL)**

| aggregation | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| EDGE_MAX | 0.0984 | 0.1639 | 0.1967 | 0.4016 |
| NODE_MAX | 0.0984 | 0.1639 | 0.1967 | 0.4016 |
| EDGE_SUM | 0.0779 | 0.123 | 0.1639 | 0.4016 |
| NODE_SUM | 0.0328 | 0.0615 | 0.1762 | 0.4139 |
| EDGE_COUNT | 0.0492 | 0.0861 | 0.1475 | 0.2951 |
| NODE_COUNT | 0.0246 | 0.0615 | 0.1148 | 0.3156 |
| SOURCE_PARTITION_COUNT | 0.0082 | 0.0328 | 0.0492 | 0.3402 |
| SOURCE_NODE_COUNT | 0.0082 | 0.0615 | 0.1025 | 0.3525 |

**2wiki_clean (ALL)**

| aggregation | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| EDGE_MAX | 0.0362 | 0.0362 | 0.0652 | 0.0966 |
| NODE_MAX | 0.0362 | 0.0362 | 0.0652 | 0.0966 |
| EDGE_SUM | 0.0217 | 0.029 | 0.0362 | 0.1039 |
| NODE_SUM | 0.0145 | 0.029 | 0.0362 | 0.1111 |
| EDGE_COUNT | 0.0072 | 0.0072 | 0.029 | 0.0942 |
| NODE_COUNT | 0.0072 | 0.0145 | 0.029 | 0.0942 |
| SOURCE_PARTITION_COUNT | 0.0 | 0.029 | 0.0507 | 0.0942 |
| SOURCE_NODE_COUNT | 0.0072 | 0.029 | 0.058 | 0.0978 |

**squad_clean (ALL)**

| aggregation | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| EDGE_MAX | 0.0612 | 0.1429 | 0.2041 | 0.3265 |
| NODE_MAX | 0.0612 | 0.1429 | 0.2041 | 0.3265 |
| EDGE_SUM | 0.0204 | 0.0408 | 0.0816 | 0.2245 |
| NODE_SUM | 0.0612 | 0.0816 | 0.102 | 0.2245 |
| EDGE_COUNT | 0.1224 | 0.1224 | 0.2245 | 0.2653 |
| NODE_COUNT | 0.0408 | 0.102 | 0.2041 | 0.3061 |
| SOURCE_PARTITION_COUNT | 0.102 | 0.1224 | 0.1837 | 0.3469 |
| SOURCE_NODE_COUNT | 0.0816 | 0.102 | 0.1429 | 0.2449 |

## T5. STEP 5 + STEP 6 -- transition evidence against reach, at the PARTITION budget

`R0_REACHED_ARRIVAL` and `R1_REACHED_COUNT` are the controls: they encode only that Pj was structurally reached.  `TR_*` are the two orderings only the transition representation can express.  `C0/C1/C2` are the three STEP 6 controls; there is no scorer grid.

**metaqa (hop3)**

| ordering | AUC | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|---|
| R0_REACHED_ARRIVAL | 0.5133 | 0.0008 | 0.0093 | 0.0343 | 0.1334 |
| R1_REACHED_COUNT | 0.5909 | 0.0222 | 0.0462 | 0.088 | 0.2241 |
| C0_SINGLE_TRIPLET | 0.6481 | 0.0906 | 0.1441 | 0.2084 | 0.3412 |
| T1_SOURCE | 0.5216 | 0.0005 | 0.0118 | 0.049 | 0.1512 |
| T2_TARGET | 0.5787 | 0.0154 | 0.0429 | 0.0745 | 0.2123 |
| T3_HOP | 0.5133 | 0.0008 | 0.0093 | 0.0343 | 0.1334 |
| C1_BEST_CHAIN_MIN | 0.5735 | 0.0181 | 0.0463 | 0.0795 | 0.2075 |
| C1_BEST_CHAIN_SUM | 0.6194 | 0.0543 | 0.1021 | 0.1652 | 0.3126 |
| C2_MULTI_PATH_SUPPORT | 0.5921 | 0.0152 | 0.0371 | 0.0702 | 0.2224 |
| C2_MULTI_SEED_SUPPORT | 0.5911 | 0.0119 | 0.0423 | 0.0819 | 0.2223 |
| TR_SOURCE_PARTITIONS | 0.5905 | 0.0142 | 0.0393 | 0.079 | 0.2202 |
| TR_TRANSITION_COUNT | 0.5905 | 0.0142 | 0.0393 | 0.079 | 0.2202 |
| NODE_S4_FROZEN (incumbent) | -- | 0.0711 | 0.1382 | 0.1971 | 0.3317 |

**musique_clean (ALL)**

| ordering | AUC | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|---|
| R0_REACHED_ARRIVAL | 0.5001 | 0.0082 | 0.0287 | 0.1148 | 0.3648 |
| R1_REACHED_COUNT | 0.4908 | 0.0328 | 0.0656 | 0.0984 | 0.3115 |
| C0_SINGLE_TRIPLET | 0.5345 | 0.0984 | 0.1639 | 0.1967 | 0.4016 |
| T1_SOURCE | 0.467 | 0.0123 | 0.0369 | 0.082 | 0.3443 |
| T2_TARGET | 0.6004 | 0.0902 | 0.1434 | 0.1926 | 0.4795 |
| T3_HOP | 0.5001 | 0.0082 | 0.0287 | 0.1148 | 0.3648 |
| C1_BEST_CHAIN_MIN | 0.5032 | 0.0164 | 0.082 | 0.1434 | 0.3811 |
| C1_BEST_CHAIN_SUM | 0.5572 | 0.0656 | 0.1025 | 0.1885 | 0.4467 |
| C2_MULTI_PATH_SUPPORT | 0.4864 | 0.0082 | 0.0615 | 0.1025 | 0.3566 |
| C2_MULTI_SEED_SUPPORT | 0.4834 | 0.0082 | 0.0328 | 0.0574 | 0.3525 |
| TR_SOURCE_PARTITIONS | 0.4869 | 0.0082 | 0.0328 | 0.0492 | 0.3402 |
| TR_TRANSITION_COUNT | 0.4869 | 0.0082 | 0.0328 | 0.0492 | 0.3402 |
| NODE_S4_FROZEN (incumbent) | -- | 0.1025 | 0.1557 | 0.1803 | 0.2787 |

**2wiki_clean (ALL)**

| ordering | AUC | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|---|
| R0_REACHED_ARRIVAL | 0.4864 | 0.0 | 0.0217 | 0.0362 | 0.1159 |
| R1_REACHED_COUNT | 0.4745 | 0.0072 | 0.0072 | 0.029 | 0.0942 |
| C0_SINGLE_TRIPLET | 0.4983 | 0.0362 | 0.0362 | 0.0652 | 0.0966 |
| T1_SOURCE | 0.4908 | 0.0072 | 0.029 | 0.0725 | 0.1159 |
| T2_TARGET | 0.6378 | 0.0 | 0.0217 | 0.0507 | 0.1365 |
| T3_HOP | 0.4864 | 0.0 | 0.0217 | 0.0362 | 0.1159 |
| C1_BEST_CHAIN_MIN | 0.4477 | 0.0072 | 0.029 | 0.0362 | 0.0942 |
| C1_BEST_CHAIN_SUM | 0.4729 | 0.029 | 0.0435 | 0.0507 | 0.1087 |
| C2_MULTI_PATH_SUPPORT | 0.4873 | 0.0072 | 0.029 | 0.058 | 0.0978 |
| C2_MULTI_SEED_SUPPORT | 0.4833 | 0.0072 | 0.029 | 0.0507 | 0.0942 |
| TR_SOURCE_PARTITIONS | 0.4838 | 0.0 | 0.029 | 0.0507 | 0.0942 |
| TR_TRANSITION_COUNT | 0.4838 | 0.0 | 0.029 | 0.0507 | 0.0942 |
| NODE_S4_FROZEN (incumbent) | -- | 0.0217 | 0.0435 | 0.0507 | 0.0894 |

**squad_clean (ALL)**

| ordering | AUC | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|---|
| R0_REACHED_ARRIVAL | 0.5821 | 0.102 | 0.1633 | 0.2245 | 0.3061 |
| R1_REACHED_COUNT | 0.5783 | 0.102 | 0.1224 | 0.2041 | 0.2857 |
| C0_SINGLE_TRIPLET | 0.5942 | 0.0612 | 0.1429 | 0.2041 | 0.3265 |
| T1_SOURCE | 0.5424 | 0.102 | 0.1429 | 0.1837 | 0.2449 |
| T2_TARGET | 0.6726 | 0.1429 | 0.1837 | 0.2245 | 0.4082 |
| T3_HOP | 0.5821 | 0.102 | 0.1633 | 0.2245 | 0.3061 |
| C1_BEST_CHAIN_MIN | 0.5998 | 0.1633 | 0.1837 | 0.2245 | 0.3265 |
| C1_BEST_CHAIN_SUM | 0.5834 | 0.0408 | 0.1429 | 0.1429 | 0.3061 |
| C2_MULTI_PATH_SUPPORT | 0.5591 | 0.0816 | 0.102 | 0.1429 | 0.2449 |
| C2_MULTI_SEED_SUPPORT | 0.5774 | 0.102 | 0.1224 | 0.1837 | 0.3265 |
| TR_SOURCE_PARTITIONS | 0.5765 | 0.102 | 0.1224 | 0.1837 | 0.3469 |
| TR_TRANSITION_COUNT | 0.5765 | 0.102 | 0.1224 | 0.1837 | 0.3469 |
| NODE_S4_FROZEN (incumbent) | -- | 0.0816 | 0.102 | 0.2041 | 0.2245 |

## T6. STEP 8 -- complementarity, measured BEFORE any combination

Unit: a query whose ENTIRE set of needed partitions is covered by the ordering's top 6 (B = 6 is the whole swap budget).

| ordering | both | only triplet | only node | triplet alone | node alone | union ceiling |
|---|---|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 17 | 9 | 13 | 0.013 | 0.015 | 0.0195 |
| TR_EDGE_SUM | 5 | 11 | 25 | 0.008 | 0.015 | 0.0205 |
| CORE_EXIT_OFFSET | 19 | 10 | 11 | 0.0145 | 0.015 | 0.02 |
| CORE_EXIT_ONLY | 19 | 10 | 11 | 0.0145 | 0.015 | 0.02 |
| TRIPLET_S4 | 6 | 3 | 24 | 0.0045 | 0.015 | 0.0165 |
| PATH_S4 | 5 | 2 | 25 | 0.0035 | 0.015 | 0.016 |
| FUSE_NODE_S4_X_CORE_EXIT | 23 | 10 | 7 | 0.0165 | 0.015 | 0.02 |

## T7. STEP 9 -- triplet evidence mapped onto the EXISTING C partitions

**MetaQA ALL**

| ordering | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 0.0943 | 0.1577 | 0.217 | 0.3585 |
| TR_EDGE_SUM | 0.0578 | 0.0891 | 0.1276 | 0.2372 |
| CORE_EXIT_OFFSET | 0.0914 | 0.1569 | 0.2154 | 0.3356 |
| CORE_EXIT_ONLY | 0.0914 | 0.1569 | 0.2154 | 0.3356 |
| TRIPLET_S4 | 0.0177 | 0.0442 | 0.1054 | 0.291 |
| PATH_S4 | 0.0162 | 0.0438 | 0.0919 | 0.2588 |
| FUSE_NODE_S4_X_CORE_EXIT | 0.0963 | 0.1596 | 0.217 | 0.3571 |
| NODE_S4_FROZEN | 0.0845 | 0.1578 | 0.2139 | 0.3479 |

**MetaQA hop2**

| ordering | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 0.1053 | 0.1946 | 0.2419 | 0.4028 |
| TR_EDGE_SUM | 0.0733 | 0.0952 | 0.1301 | 0.2253 |
| CORE_EXIT_OFFSET | 0.1117 | 0.1918 | 0.2575 | 0.3935 |
| CORE_EXIT_ONLY | 0.1117 | 0.1918 | 0.2575 | 0.3935 |
| TRIPLET_S4 | 0.0496 | 0.1034 | 0.2018 | 0.4284 |
| PATH_S4 | 0.0406 | 0.0943 | 0.176 | 0.3757 |
| FUSE_NODE_S4_X_CORE_EXIT | 0.1244 | 0.1961 | 0.262 | 0.4023 |
| NODE_S4_FROZEN | 0.1198 | 0.2099 | 0.26 | 0.3893 |

**MetaQA hop3**

| ordering | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 0.0906 | 0.1441 | 0.2084 | 0.3412 |
| TR_EDGE_SUM | 0.052 | 0.0873 | 0.1273 | 0.2413 |
| CORE_EXIT_OFFSET | 0.0839 | 0.1441 | 0.2001 | 0.3128 |
| CORE_EXIT_ONLY | 0.0839 | 0.1441 | 0.2001 | 0.3128 |
| TRIPLET_S4 | 0.0052 | 0.0211 | 0.0672 | 0.2356 |
| PATH_S4 | 0.0067 | 0.0232 | 0.0583 | 0.2113 |
| FUSE_NODE_S4_X_CORE_EXIT | 0.0858 | 0.1462 | 0.1996 | 0.3395 |
| NODE_S4_FROZEN | 0.0711 | 0.1382 | 0.1971 | 0.3317 |

## T8. STEP 10 -- through the unchanged boundary mechanism, exact P = 50, B = 6

Frozen candidate list 47.5 partitions/query; the triplet universe is 305.5.  The previous phase established that a longer list alone costs coverage through F6, so both conditions are reported: FULL depth, and DEPTH-MATCHED, where each triplet ordering is cut to exactly the frozen ordering's own per-query length K (read off, never tuned).

**FULL depth** (SAFE = ALL 0.6612, hop1 0.9955, hop2 0.7297, hop3 0.2583)

| ordering | ALL | hop1 | hop2 | hop3 | churn | net | McNemar p |  |
|---|---|---|---|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 0.6542 | 0.9955 | 0.7147 | 0.2523 | 4.939 | -14 | 0.00131 | SIG |
| TR_EDGE_SUM | 0.6512 | 0.9955 | 0.7087 | 0.2492 | 4.975 | -20 | 4e-05 | SIG |
| CORE_EXIT_OFFSET | 0.6537 | 0.9955 | 0.7162 | 0.2492 | 4.929 | -15 | 0.00073 | SIG |
| CORE_EXIT_ONLY | 0.6537 | 0.9955 | 0.7162 | 0.2492 | 4.927 | -15 | 0.00073 | SIG |
| TRIPLET_S4 | 0.6542 | 0.9970 | 0.7132 | 0.2523 | 4.876 | -14 | 0.00661 | SIG |
| PATH_S4 | 0.6522 | 0.9970 | 0.7072 | 0.2523 | 4.879 | -18 | 0.00091 | SIG |
| FUSE_NODE_S4_X_CORE_EXIT | 0.6542 | 0.9955 | 0.7162 | 0.2508 | 4.927 | -14 | 0.00052 | SIG |

**DEPTH-MATCHED** (SAFE = ALL 0.6612, hop1 0.9955, hop2 0.7297, hop3 0.2583)

| ordering | ALL | hop1 | hop2 | hop3 | churn | net | McNemar p |  |
|---|---|---|---|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 0.6607 | 0.9955 | 0.7282 | 0.2583 | 4.678 | -1 | 1.0 |  |
| TR_EDGE_SUM | 0.6547 | 0.9955 | 0.7147 | 0.2538 | 4.78 | -13 | 0.01062 | SIG |
| CORE_EXIT_OFFSET | 0.6622 | 0.9955 | 0.7312 | 0.2598 | 4.697 | +2 | 0.77441 |  |
| CORE_EXIT_ONLY | 0.6622 | 0.9955 | 0.7312 | 0.2598 | 4.696 | +2 | 0.77441 |  |
| TRIPLET_S4 | 0.6602 | 0.9955 | 0.7312 | 0.2538 | 4.49 | -2 | 0.85055 |  |
| PATH_S4 | 0.6587 | 0.9955 | 0.7267 | 0.2538 | 4.507 | -5 | 0.47313 |  |
| FUSE_NODE_S4_X_CORE_EXIT | 0.6627 | 0.9955 | 0.7312 | 0.2613 | 4.696 | +3 | 0.45312 |  |

## T9. STEP 11 -- candidate-availability budget curve

k = 50 is the contract.  The question is whether the required-partition curve moves LEFT toward it; 256 is reference only.

| ordering | k50 | k64 | k80 | k100 | k128 | k256 |
|---|---|---|---|---|---|---|
| NODE_S4_FROZEN | 0.6612 | 0.6842 | 0.7047 | 0.7377 | 0.7718 | 0.8278 |
| FUSE_NODE_S4_X_CORE_EXIT_DEPTH_MATCHED | 0.6627 | 0.6857 | 0.7052 | 0.7372 | 0.7718 | 0.8273 |
| C0_SINGLE_TRIPLET_DEPTH_MATCHED | 0.6607 | 0.6852 | 0.7057 | 0.7377 | 0.7718 | 0.8278 |

MetaQA hop3 only:

| ordering | k50 | k64 | k80 | k100 | k128 | k256 |
|---|---|---|---|---|---|---|
| NODE_S4_FROZEN | 0.2583 | 0.2898 | 0.3333 | 0.3889 | 0.4294 | 0.5511 |
| FUSE_NODE_S4_X_CORE_EXIT_DEPTH_MATCHED | 0.2613 | 0.2943 | 0.3348 | 0.3874 | 0.4294 | 0.5495 |
| C0_SINGLE_TRIPLET_DEPTH_MATCHED | 0.2583 | 0.2898 | 0.3363 | 0.3889 | 0.4294 | 0.5511 |

