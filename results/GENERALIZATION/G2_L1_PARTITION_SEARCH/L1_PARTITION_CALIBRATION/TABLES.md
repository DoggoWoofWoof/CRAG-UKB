# FULL_VISITED PARTITION CALIBRATION AUDIT -- TABLES

Contract held fixed in every row: FINAL **P = 50** (exact), **B = 6**, the frozen bounded
structural search (no new traversal, no new edge, no node scorer), the frozen F6 boundary
competition, no learned parameter, no threshold grid, no dataset branch, no TEST split.
The only thing any variant changes is the ORDER of the structural partition ranking.

Corpora present: MetaQA, WebQSP, 2Wiki, MuSiQue, HotpotQA, SQuAD.

## T1. STEP 1 -- the exact challenger universe

Challengers = every FULL_VISITED partition outside the canonical top-50. NEEDED = a gold
partition that must displace an incumbent; NUISANCE = everything else. Gold is used only to
label rows for evaluation and never enters a feature.

| corpus | nq | partitions | replay parity | cand/q | needed/q | prevalence | needed reach | universe saturation | empty universes |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA | 1998 | 401 | 1998/1998 | 261.2 | 2.71 | 0.01038 | 0.9195 | 0.7619 | 0 |
| WebQSP | 1419 | 7814 | 1419/1419 | 387.8 | 0.428 | 0.00110 | 0.3600 | 0.0534 | 14 |
| 2Wiki | 2000 | 658 | 2000/2000 | 144.4 | 0.025 | 0.00017 | 0.3740 | 0.2541 | 86 |
| MuSiQue | 2000 | 136 | 2000/2000 | 66.7 | 0.038 | 0.00058 | 0.8556 | 0.8216 | 19 |
| HotpotQA | 2000 | 5074 | 2000/2000 | 561.2 | 0.038 | 0.00007 | 0.5507 | 0.1153 | 0 |
| SQuAD | 2000 | 190 | 2000/2000 | 80.0 | 0.012 | 0.00015 | 0.6154 | 0.6147 | 295 |

`universe saturation` = mean fraction of ALL partitions the bounded search visits for one query.

## T2. STEP 1 by hop (MetaQA -- the only corpus with real hop labels)

| block | cand/q | needed/q | prevalence |
|---|---|---|---|
| hop1 | 257.7 | 0.006 | 0.000023 |
| hop2 | 259.8 | 1.296 | 0.004989 |
| hop3 | 266.2 | 6.829 | 0.025649 |

On hop1 there is essentially nothing to find, which is why hop1 is flat in every table below.

## T3. STEP 2 + STEP 5 -- raw structural evidence, MetaQA hop3

Every row is a quantity the frozen beam already accumulated over ALL arrivals before its prune.
`NEEDED_RECALL@B` is the primary metric: the fraction of needed partitions that land in the top B
of the challenger ordering, i.e. that get the chance to displace a canonical incumbent.

| signal | AUC | needed pctl | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|---|---|
| BEST_STRUCT_NODE_SCORE | 0.6722 | 0.6676 | 0.0899 | 0.1489 | 0.2058 | 0.3681 |
| FIRST_ARRIVAL_INDEX | 0.6724 | 0.6678 | 0.0897 | 0.1482 | 0.2056 | 0.3676 |
| COH_SEED_TIMES_SDIR | 0.6741 | 0.6693 | 0.0877 | 0.1464 | 0.2139 | 0.3841 |
| S4_RANK_FULL_VISITED | 0.6738 | 0.6687 | 0.0752 | 0.1330 | 0.1996 | 0.3832 |
| BEST_PATH_SCORE | 0.6396 | 0.6355 | 0.0646 | 0.1255 | 0.1900 | 0.3575 |
| RRF_MASS | 0.6712 | 0.6661 | 0.0563 | 0.1119 | 0.1785 | 0.3824 |
| DISTINCT_PARENT_COUNT | 0.6278 | 0.6235 | 0.0526 | 0.1027 | 0.1596 | 0.3417 |
| STRUCT_NODE_COUNT | 0.6190 | 0.6147 | 0.0497 | 0.0994 | 0.1599 | 0.3329 |
| BEST_PATH_MIN_SCORE | 0.6026 | 0.5995 | 0.0479 | 0.0871 | 0.1365 | 0.2951 |
| RANK_WEIGHTED_PARENT_SUPPORT | 0.5871 | 0.5835 | 0.0475 | 0.0860 | 0.1350 | 0.3012 |
| ADMITTED_NODE_COUNT | 0.6303 | 0.6271 | 0.0460 | 0.0943 | 0.1482 | 0.3421 |
| DISTINCT_SEED_COUNT | 0.5896 | 0.5863 | 0.0455 | 0.0893 | 0.1447 | 0.3153 |
| CANONICAL_RANK | 0.6375 | 0.6346 | 0.0402 | 0.0818 | 0.1343 | 0.3067 |
| BEST_PARENT_SCORE | 0.5452 | 0.5430 | 0.0361 | 0.0688 | 0.1163 | 0.2533 |
| COH_ADMITTED_FRACTION | 0.6239 | 0.6209 | 0.0356 | 0.0726 | 0.1218 | 0.3210 |
| MIN_HOP | 0.5508 | 0.5494 | 0.0352 | 0.0671 | 0.1137 | 0.2946 |
| DENSE_RANK | 0.5319 | 0.5306 | 0.0328 | 0.0633 | 0.1009 | 0.2445 |
| COH_ARRIVALS_PER_NODE | 0.5669 | 0.5652 | 0.0323 | 0.0719 | 0.1271 | 0.2949 |
| RETRIEVAL_RRF_RANK | 0.5217 | 0.5207 | 0.0259 | 0.0512 | 0.0888 | 0.2216 |
| COH_SEEDS_PER_NODE | 0.4128 | 0.4156 | 0.0224 | 0.0484 | 0.0743 | 0.1447 |
| COH_BEST_NODE_RRF_SHARE | 0.4502 | 0.4530 | 0.0204 | 0.0369 | 0.0552 | 0.1137 |
| SPLADE_RANK | 0.5101 | 0.5097 | 0.0187 | 0.0429 | 0.0748 | 0.1983 |
| **ATTAINABLE CEILING** | -- | -- | **0.4116** | **0.6295** | **0.7891** | **0.9813** |

`DISTINCT_PARENT_COUNT` is also the path count: a node enters the frontier at exactly one hop on
this substrate, so distinct parent states, distinct parent nodes and the arrival count coincide.

The ceiling row is `sum min(k, n_needed) / sum n_needed` -- what a perfect ordering could reach
at that budget, given that many queries need more than k partitions.

## T4. STEP 2 -- the S4 composite vs its own best component, all corpora (ALL block)

| corpus | S4 AUC | S4 R@6 | best single signal | best R@6 | delta | ceiling R@6 |
|---|---|---|---|---|---|---|
| MetaQA | 0.7023 | 0.0790 | BEST_STRUCT_NODE_SCORE | 0.0886 | +0.0096 | 0.4334 |
| WebQSP | 0.6912 | 0.0890 | CANONICAL_RANK | 0.1631 | +0.0741 | 0.8699 |
| 2Wiki | 0.5221 | 0.0612 | RETRIEVAL_RRF_RANK | 0.2245 | +0.1633 | 1.0000 |
| MuSiQue | 0.5858 | 0.1039 | RETRIEVAL_RRF_RANK | 0.3766 | +0.2727 | 1.0000 |
| HotpotQA | 0.7050 | 0.1053 | RETRIEVAL_RRF_RANK | 0.4342 | +0.3289 | 1.0000 |
| SQuAD | 0.6718 | 0.2500 | RETRIEVAL_RRF_RANK | 0.6250 | +0.3750 | 1.0000 |


## T5. STEP 3 -- is the ranking rewarding partitions that are merely easy to reach?

Ratio of the mean static corpus-side quantity over NEEDED vs NUISANCE challengers. A ratio > 1
means the needed partitions are themselves the high-degree / high-exposure ones, so dividing
the evidence by that quantity removes signal rather than noise.

| corpus | partition size | partition degree | boundary degree | adjacent partitions | degree per node | expected visitation | P(visited) |
|---|---|---|---|---|---|---|---|
| MetaQA | 1.004 | 1.303 | 1.420 | 1.260 | 1.298 | 1.101 | 1.036 |
| WebQSP | 1.003 | 1.735 | 1.843 | 1.563 | 1.725 | 1.709 | 1.295 |
| 2Wiki | 1.004 | 1.032 | 0.978 | 0.963 | 1.030 | 1.004 | 1.056 |
| MuSiQue | 1.006 | 0.968 | 0.977 | 1.016 | 0.966 | 0.996 | 1.007 |
| HotpotQA | 0.997 | 0.902 | 0.899 | 0.990 | 0.907 | 1.097 | 1.051 |
| SQuAD | 1.004 | 0.975 | 1.000 | 1.037 | 0.969 | 1.015 | 0.983 |


## T6. STEP 3 -- exposure dominance, and whether FULL_VISITED CAUSES it

Spearman between a static corpus-side quantity and how well the raw structural ranking places a
partition (1.0 = ranked best). The M64 column is the identical measurement on the FROZEN
evidence universe -- the control that turns a correlation into a claim about the universe.

**MetaQA**

| static quantity | FULL_VISITED | frozen M64 | caused by FULL_VISITED | vs P(visited) |
|---|---|---|---|---|
| partition size | +0.211 | -0.036 | **+0.248** | +0.302 |
| partition degree | +0.539 | -0.082 | **+0.621** | +0.719 |
| boundary degree | +0.570 | -0.096 | **+0.666** | +0.745 |
| adjacent partitions | +0.508 | -0.168 | **+0.676** | +0.789 |
| degree per node | +0.544 | -0.082 | **+0.626** | +0.723 |
| expected visitation | +0.775 | -0.093 | **+0.868** | +0.891 |
| P(visited) | +0.691 | -0.181 | **+0.872** | +1.000 |

**WebQSP**

| static quantity | FULL_VISITED | frozen M64 | caused by FULL_VISITED | vs P(visited) |
|---|---|---|---|---|
| partition size | +0.060 | +0.054 | **+0.006** | -0.081 |
| partition degree | +0.001 | -0.067 | **+0.068** | +0.360 |
| boundary degree | -0.016 | -0.079 | **+0.064** | +0.388 |
| adjacent partitions | -0.126 | -0.105 | **-0.021** | +0.615 |
| degree per node | +0.000 | -0.069 | **+0.069** | +0.367 |
| expected visitation | +0.196 | +0.007 | **+0.189** | +0.752 |
| P(visited) | -0.023 | -0.116 | **+0.093** | +1.000 |

**2Wiki**

| static quantity | FULL_VISITED | frozen M64 | caused by FULL_VISITED | vs P(visited) |
|---|---|---|---|---|
| partition size | +0.034 | +0.006 | **+0.028** | +0.070 |
| partition degree | +0.413 | +0.447 | **-0.034** | +0.727 |
| boundary degree | +0.373 | +0.379 | **-0.006** | +0.754 |
| adjacent partitions | +0.500 | +0.509 | **-0.009** | +0.835 |
| degree per node | +0.416 | +0.452 | **-0.036** | +0.733 |
| expected visitation | +0.630 | +0.561 | **+0.069** | +0.914 |
| P(visited) | +0.604 | +0.491 | **+0.113** | +1.000 |

**MuSiQue**

| static quantity | FULL_VISITED | frozen M64 | caused by FULL_VISITED | vs P(visited) |
|---|---|---|---|---|
| partition size | +0.087 | +0.056 | **+0.031** | +0.206 |
| partition degree | +0.325 | +0.070 | **+0.254** | +0.617 |
| boundary degree | +0.252 | -0.101 | **+0.354** | +0.751 |
| adjacent partitions | +0.318 | +0.024 | **+0.294** | +0.770 |
| degree per node | +0.327 | +0.073 | **+0.255** | +0.615 |
| expected visitation | +0.525 | +0.196 | **+0.329** | +0.712 |
| P(visited) | +0.467 | -0.011 | **+0.478** | +1.000 |

**HotpotQA**

| static quantity | FULL_VISITED | frozen M64 | caused by FULL_VISITED | vs P(visited) |
|---|---|---|---|---|
| partition size | +0.055 | +0.120 | **-0.065** | -0.085 |
| partition degree | -0.089 | -0.026 | **-0.063** | +0.307 |
| boundary degree | -0.062 | -0.028 | **-0.035** | +0.356 |
| adjacent partitions | -0.217 | -0.139 | **-0.078** | +0.540 |
| degree per node | -0.095 | -0.034 | **-0.061** | +0.319 |
| expected visitation | -0.226 | -0.173 | **-0.053** | +0.729 |
| P(visited) | -0.356 | -0.327 | **-0.029** | +1.000 |

**SQuAD**

| static quantity | FULL_VISITED | frozen M64 | caused by FULL_VISITED | vs P(visited) |
|---|---|---|---|---|
| partition size | +0.111 | +0.145 | **-0.034** | +0.135 |
| partition degree | +0.416 | +0.312 | **+0.104** | +0.526 |
| boundary degree | +0.465 | +0.245 | **+0.221** | +0.499 |
| adjacent partitions | +0.500 | +0.194 | **+0.306** | +0.599 |
| degree per node | +0.417 | +0.311 | **+0.105** | +0.529 |
| expected visitation | +0.697 | +0.431 | **+0.266** | +0.773 |
| P(visited) | +0.704 | +0.325 | **+0.379** | +1.000 |

## T7. STEP 4 -- the normalisation family (one division each, no exponent, no weight)

| corpus | block | AUC N0 | AUC N1 | AUC N2 | AUC N3 | R@6 N0 | R@6 N1 | R@6 N2 | R@6 N3 |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA | hop3 | 0.6738 | 0.6497 | 0.6727 | 0.6639 | 0.0752 | 0.0695 | 0.0756 | 0.0726 |
| WebQSP | ALL | 0.6912 | 0.6514 | 0.6938 | 0.6608 | 0.0890 | 0.0824 | 0.0906 | 0.0857 |
| 2Wiki | ALL | 0.5221 | 0.4980 | 0.5157 | 0.4984 | 0.0612 | 0.0612 | 0.0612 | 0.0816 |
| MuSiQue | ALL | 0.5858 | 0.5862 | 0.5861 | 0.5901 | 0.1039 | 0.1169 | 0.1039 | 0.1429 |
| HotpotQA | ALL | 0.7050 | 0.7130 | 0.7135 | 0.6942 | 0.1053 | 0.0921 | 0.0921 | 0.1053 |
| SQuAD | ALL | 0.6718 | 0.6723 | 0.6716 | 0.6819 | 0.2500 | 0.2917 | 0.2500 | 0.2500 |


## T8. STEP 5 -- the three direct orderings built from what STEP 2 found

D1/D2/D3 were specified AFTER reading T3, so they are diagnostics, not candidates: only survival
on all six corpora under the STEP 8 promotion gate would make one a candidate.
D1 = best node score first; D2 = distinct seeds x best node score; D3 = the S4 RRF with its two
weak channels (node count, min hop) dropped.

| corpus | block | R@6 raw S4 | R@6 D1 | R@6 D2 | R@6 D3 |
|---|---|---|---|---|---|
| MetaQA | hop3 | 0.0752 | 0.0899 | 0.0877 | 0.0897 |
| WebQSP | ALL | 0.0890 | 0.0675 | 0.0675 | 0.0675 |
| 2Wiki | ALL | 0.0612 | 0.0816 | 0.0816 | 0.0816 |
| MuSiQue | ALL | 0.1039 | 0.1688 | 0.1429 | 0.1688 |
| HotpotQA | ALL | 0.1053 | 0.0789 | 0.1053 | 0.0789 |
| SQuAD | ALL | 0.2500 | 0.3333 | 0.2917 | 0.3333 |


## T9. STEP 7/8 -- exact P50 through the UNCHANGED frozen F6

Every row feeds one structural partition order into the same boundary competition at the same
B = 6 and reports exact 50-partition coverage. `net` and `p` are McNemar against the frozen
M64 + F6 baseline on the same queries.

**MetaQA**

| structural ordering | exact P50 (ALL) | churn | delta | net | McNemar p |
|---|---|---|---|---|---|
| FROZEN M64 + F6 (baseline) | 0.6612 | 4.659 | -- | -- | -- |
| FULL_VISITED + N0_RAW | 0.6537 | 4.895 | -0.0075 | -15 | 0.00027 **SIG** |
| FULL_VISITED + N1_DEGREE_NORMALIZED | 0.6532 | 4.911 | -0.0080 | -16 | 0.00014 **SIG** |
| FULL_VISITED + N2_SIZE_NORMALIZED | 0.6532 | 4.894 | -0.0080 | -16 | 0.00014 **SIG** |
| FULL_VISITED + N3_EXPOSURE_NORMALIZED | 0.6532 | 4.901 | -0.0080 | -16 | 0.00014 **SIG** |
| FULL_VISITED + D1_BEST_NODE_SCORE | 0.6542 | 4.939 | -0.0070 | -14 | 0.00131 **SIG** |
| FULL_VISITED + D2_SEED_TIMES_SDIR | 0.6542 | 4.916 | -0.0070 | -14 | 0.00258 **SIG** |
| FULL_VISITED + D3_S4_TWO_CHANNEL | 0.6542 | 4.935 | -0.0070 | -14 | 0.00131 **SIG** |

**WebQSP**

| structural ordering | exact P50 (ALL) | churn | delta | net | McNemar p |
|---|---|---|---|---|---|
| FROZEN M64 + F6 (baseline) | 0.7646 | 4.968 | -- | -- | -- |
| FULL_VISITED + N0_RAW | 0.7689 | 4.721 | +0.0043 | +6 | 0.109 |
| FULL_VISITED + N1_DEGREE_NORMALIZED | 0.7681 | 4.711 | +0.0035 | +5 | 0.227 |
| FULL_VISITED + N2_SIZE_NORMALIZED | 0.7689 | 4.723 | +0.0043 | +6 | 0.146 |
| FULL_VISITED + N3_EXPOSURE_NORMALIZED | 0.7674 | 4.72 | +0.0028 | +4 | 0.344 |
| FULL_VISITED + D1_BEST_NODE_SCORE | 0.7667 | 4.826 | +0.0021 | +3 | 0.375 |
| FULL_VISITED + D2_SEED_TIMES_SDIR | 0.7667 | 4.801 | +0.0021 | +3 | 0.453 |
| FULL_VISITED + D3_S4_TWO_CHANNEL | 0.7653 | 4.82 | +0.0007 | +1 | 1 |

**2Wiki**

| structural ordering | exact P50 (ALL) | churn | delta | net | McNemar p |
|---|---|---|---|---|---|
| FROZEN M64 + F6 (baseline) | 0.9435 | 4.259 | -- | -- | -- |
| FULL_VISITED + N0_RAW | 0.9415 | 4.348 | -0.0020 | -4 | 0.289 |
| FULL_VISITED + N1_DEGREE_NORMALIZED | 0.9415 | 4.367 | -0.0020 | -4 | 0.289 |
| FULL_VISITED + N2_SIZE_NORMALIZED | 0.9415 | 4.354 | -0.0020 | -4 | 0.289 |
| FULL_VISITED + N3_EXPOSURE_NORMALIZED | 0.9420 | 4.355 | -0.0015 | -3 | 0.453 |
| FULL_VISITED + D1_BEST_NODE_SCORE | 0.9415 | 4.38 | -0.0020 | -4 | 0.289 |
| FULL_VISITED + D2_SEED_TIMES_SDIR | 0.9420 | 4.359 | -0.0015 | -3 | 0.453 |
| FULL_VISITED + D3_S4_TWO_CHANNEL | 0.9415 | 4.378 | -0.0020 | -4 | 0.289 |

**MuSiQue**

| structural ordering | exact P50 (ALL) | churn | delta | net | McNemar p |
|---|---|---|---|---|---|
| FROZEN M64 + F6 (baseline) | 0.9635 | 4.312 | -- | -- | -- |
| FULL_VISITED + N0_RAW | 0.9625 | 4.327 | -0.0010 | -2 | 0.774 |
| FULL_VISITED + N1_DEGREE_NORMALIZED | 0.9620 | 4.391 | -0.0015 | -3 | 0.581 |
| FULL_VISITED + N2_SIZE_NORMALIZED | 0.9625 | 4.322 | -0.0010 | -2 | 0.774 |
| FULL_VISITED + N3_EXPOSURE_NORMALIZED | 0.9630 | 4.367 | -0.0005 | -1 | 1 |
| FULL_VISITED + D1_BEST_NODE_SCORE | 0.9630 | 4.413 | -0.0005 | -1 | 1 |
| FULL_VISITED + D2_SEED_TIMES_SDIR | 0.9610 | 4.328 | -0.0025 | -5 | 0.332 |
| FULL_VISITED + D3_S4_TWO_CHANNEL | 0.9630 | 4.418 | -0.0005 | -1 | 1 |

**HotpotQA**

| structural ordering | exact P50 (ALL) | churn | delta | net | McNemar p |
|---|---|---|---|---|---|
| FROZEN M64 + F6 (baseline) | 0.9505 | 5.098 | -- | -- | -- |
| FULL_VISITED + N0_RAW | 0.9500 | 5.055 | -0.0005 | -1 | 1 |
| FULL_VISITED + N1_DEGREE_NORMALIZED | 0.9505 | 5.058 | +0.0000 | +0 | 1 |
| FULL_VISITED + N2_SIZE_NORMALIZED | 0.9500 | 5.061 | -0.0005 | -1 | 1 |
| FULL_VISITED + N3_EXPOSURE_NORMALIZED | 0.9500 | 5.044 | -0.0005 | -1 | 1 |
| FULL_VISITED + D1_BEST_NODE_SCORE | 0.9505 | 5.17 | +0.0000 | +0 | 1 |
| FULL_VISITED + D2_SEED_TIMES_SDIR | 0.9500 | 5.127 | -0.0005 | -1 | 1 |
| FULL_VISITED + D3_S4_TWO_CHANNEL | 0.9505 | 5.168 | +0.0000 | +0 | 1 |

**SQuAD**

| structural ordering | exact P50 (ALL) | churn | delta | net | McNemar p |
|---|---|---|---|---|---|
| FROZEN M64 + F6 (baseline) | 0.9875 | 3.974 | -- | -- | -- |
| FULL_VISITED + N0_RAW | 0.9880 | 4.24 | +0.0005 | +1 | 1 |
| FULL_VISITED + N1_DEGREE_NORMALIZED | 0.9880 | 4.271 | +0.0005 | +1 | 1 |
| FULL_VISITED + N2_SIZE_NORMALIZED | 0.9880 | 4.242 | +0.0005 | +1 | 1 |
| FULL_VISITED + N3_EXPOSURE_NORMALIZED | 0.9880 | 4.281 | +0.0005 | +1 | 1 |
| FULL_VISITED + D1_BEST_NODE_SCORE | 0.9875 | 4.295 | +0.0000 | +0 | 1 |
| FULL_VISITED + D2_SEED_TIMES_SDIR | 0.9875 | 4.248 | +0.0000 | +0 | 1 |
| FULL_VISITED + D3_S4_TWO_CHANNEL | 0.9875 | 4.306 | +0.0000 | +0 | 1 |

## T10. STEP 8 -- MetaQA by hop, exact P50, against the FULL_VISITED oracle ceilings

| structural ordering | hop1 | hop2 | hop3 | ALL | net vs baseline |
|---|---|---|---|---|---|
| FROZEN M64 + F6 (baseline) | 0.9955 | 0.7297 | 0.2583 | 0.6612 | -- |
| FULL_VISITED + N0_RAW | 0.9955 | 0.7147 | 0.2508 | 0.6537 | -5 hop3 |
| FULL_VISITED + N1_DEGREE_NORMALIZED | 0.9955 | 0.7147 | 0.2492 | 0.6532 | -6 hop3 **SIG** |
| FULL_VISITED + N2_SIZE_NORMALIZED | 0.9955 | 0.7147 | 0.2492 | 0.6532 | -6 hop3 **SIG** |
| FULL_VISITED + N3_EXPOSURE_NORMALIZED | 0.9955 | 0.7132 | 0.2508 | 0.6532 | -5 hop3 |
| FULL_VISITED + D1_BEST_NODE_SCORE | 0.9955 | 0.7147 | 0.2523 | 0.6542 | -4 hop3 |
| FULL_VISITED + D2_SEED_TIMES_SDIR | 0.9955 | 0.7117 | 0.2553 | 0.6542 | -2 hop3 |
| FULL_VISITED + D3_S4_TWO_CHANNEL | 0.9955 | 0.7147 | 0.2523 | 0.6542 | -4 hop3 |
| *FULL_VISITED oracle, B=6* | -- | -- | *0.5916* | -- | *ceiling* |
| *FULL_VISITED oracle, B=50* | -- | -- | *0.6997* | -- | *ceiling* |

Oracle rows carry over from the contract audit's exact coverability test on the same universe;
they are what a perfect selector could reach, not anything measured here.

## T11. STEP 7 ledger -- where the six swap slots actually go

Exact-P50 is a set test, so recall cannot say why a query flips. This opens the frozen F6
selection itself. `slots on nuisance` is the fraction of the B x nq swap slots that hold a
non-gold partition; `gold evicted` counts gold partitions sitting at boundary ranks 44-49 that
a challenger displaces. `lost by drop` is the number of queries the baseline covered and this
ordering does not BECAUSE a needed partition the baseline's own six slots held fell out.

**MetaQA** (hop3: 666 queries x B = 6 slots)

| structural ordering | needed | needed admitted | novel gold admitted | gold evicted | slots on nuisance | lost / gained | lost by drop |
|---|---|---|---|---|---|---|---|
| FROZEN_M64/S4 (SAFE) | 5183 | 244 | 203 | 132 | 0.9389 | -- | -- |
| FV + N0_RAW | 5183 | 171 | 146 | 148 | 0.9572 | 5 / 0 | 5 (0 incumbent) |
| FV + N1_DEGREE_NORMALIZED | 5183 | 162 | 138 | 149 | 0.9595 | 6 / 0 | 6 (1 incumbent) |
| FV + N2_SIZE_NORMALIZED | 5183 | 167 | 143 | 149 | 0.9582 | 6 / 0 | 6 (1 incumbent) |
| FV + N3_EXPOSURE_NORMALIZED | 5183 | 164 | 140 | 149 | 0.9590 | 5 / 0 | 5 (0 incumbent) |
| FV + D1_BEST_NODE_SCORE | 5183 | 167 | 142 | 148 | 0.9582 | 5 / 1 | 5 (1 incumbent) |
| FV + D2_SEED_TIMES_SDIR | 5183 | 178 | 151 | 146 | 0.9555 | 4 / 2 | 4 (0 incumbent) |
| FV + D3_S4_TWO_CHANNEL | 5183 | 168 | 143 | 148 | 0.9580 | 5 / 1 | 5 (1 incumbent) |

**WebQSP** (ALL: 1419 queries x B = 6 slots)

| structural ordering | needed | needed admitted | novel gold admitted | gold evicted | slots on nuisance | lost / gained | lost by drop |
|---|---|---|---|---|---|---|---|
| FROZEN_M64/S4 (SAFE) | 1811 | 126 | 91 | 90 | 0.9852 | -- | -- |
| FV + N0_RAW | 1811 | 144 | 103 | 84 | 0.9831 | 2 / 8 | 2 (0 incumbent) |
| FV + N1_DEGREE_NORMALIZED | 1811 | 139 | 100 | 86 | 0.9837 | 3 / 8 | 3 (0 incumbent) |
| FV + N2_SIZE_NORMALIZED | 1811 | 147 | 107 | 85 | 0.9827 | 3 / 9 | 3 (0 incumbent) |
| FV + N3_EXPOSURE_NORMALIZED | 1811 | 136 | 98 | 87 | 0.9840 | 3 / 7 | 3 (0 incumbent) |
| FV + D1_BEST_NODE_SCORE | 1811 | 134 | 97 | 88 | 0.9843 | 1 / 4 | 1 (0 incumbent) |
| FV + D2_SEED_TIMES_SDIR | 1811 | 140 | 101 | 86 | 0.9836 | 2 / 5 | 2 (0 incumbent) |
| FV + D3_S4_TWO_CHANNEL | 1811 | 133 | 97 | 89 | 0.9844 | 2 / 3 | 2 (0 incumbent) |

**2Wiki** (ALL: 2000 queries x B = 6 slots)

| structural ordering | needed | needed admitted | novel gold admitted | gold evicted | slots on nuisance | lost / gained | lost by drop |
|---|---|---|---|---|---|---|---|
| FROZEN_M64/S4 (SAFE) | 147 | 30 | 20 | 6 | 0.9975 | -- | -- |
| FV + N0_RAW | 147 | 26 | 17 | 7 | 0.9978 | 6 / 2 | 6 (1 incumbent) |
| FV + N1_DEGREE_NORMALIZED | 147 | 26 | 17 | 7 | 0.9978 | 6 / 2 | 6 (1 incumbent) |
| FV + N2_SIZE_NORMALIZED | 147 | 26 | 17 | 7 | 0.9978 | 6 / 2 | 6 (1 incumbent) |
| FV + N3_EXPOSURE_NORMALIZED | 147 | 27 | 18 | 7 | 0.9978 | 5 / 2 | 5 (1 incumbent) |
| FV + D1_BEST_NODE_SCORE | 147 | 26 | 17 | 7 | 0.9978 | 6 / 2 | 6 (1 incumbent) |
| FV + D2_SEED_TIMES_SDIR | 147 | 27 | 18 | 7 | 0.9978 | 5 / 2 | 5 (1 incumbent) |
| FV + D3_S4_TWO_CHANNEL | 147 | 26 | 17 | 7 | 0.9978 | 6 / 2 | 6 (1 incumbent) |

**MuSiQue** (ALL: 2000 queries x B = 6 slots)

| structural ordering | needed | needed admitted | novel gold admitted | gold evicted | slots on nuisance | lost / gained | lost by drop |
|---|---|---|---|---|---|---|---|
| FROZEN_M64/S4 (SAFE) | 128 | 53 | 26 | 11 | 0.9956 | -- | -- |
| FV + N0_RAW | 128 | 52 | 27 | 13 | 0.9957 | 7 / 5 | 7 (3 incumbent) |
| FV + N1_DEGREE_NORMALIZED | 128 | 51 | 27 | 14 | 0.9958 | 8 / 5 | 8 (4 incumbent) |
| FV + N2_SIZE_NORMALIZED | 128 | 52 | 27 | 13 | 0.9957 | 7 / 5 | 7 (3 incumbent) |
| FV + N3_EXPOSURE_NORMALIZED | 128 | 53 | 28 | 13 | 0.9956 | 6 / 5 | 6 (3 incumbent) |
| FV + D1_BEST_NODE_SCORE | 128 | 53 | 29 | 14 | 0.9956 | 6 / 5 | 6 (4 incumbent) |
| FV + D2_SEED_TIMES_SDIR | 128 | 49 | 26 | 15 | 0.9959 | 11 / 6 | 11 (6 incumbent) |
| FV + D3_S4_TWO_CHANNEL | 128 | 53 | 29 | 14 | 0.9956 | 6 / 5 | 6 (4 incumbent) |

**HotpotQA** (ALL: 2000 queries x B = 6 slots)

| structural ordering | needed | needed admitted | novel gold admitted | gold evicted | slots on nuisance | lost / gained | lost by drop |
|---|---|---|---|---|---|---|---|
| FROZEN_M64/S4 (SAFE) | 161 | 57 | 39 | 5 | 0.9952 | -- | -- |
| FV + N0_RAW | 161 | 57 | 39 | 5 | 0.9952 | 4 / 3 | 4 (0 incumbent) |
| FV + N1_DEGREE_NORMALIZED | 161 | 58 | 39 | 4 | 0.9952 | 4 / 4 | 4 (0 incumbent) |
| FV + N2_SIZE_NORMALIZED | 161 | 57 | 39 | 5 | 0.9952 | 4 / 3 | 4 (0 incumbent) |
| FV + N3_EXPOSURE_NORMALIZED | 161 | 57 | 38 | 4 | 0.9952 | 5 / 4 | 5 (0 incumbent) |
| FV + D1_BEST_NODE_SCORE | 161 | 57 | 39 | 5 | 0.9952 | 4 / 4 | 4 (0 incumbent) |
| FV + D2_SEED_TIMES_SDIR | 161 | 55 | 37 | 5 | 0.9954 | 5 / 4 | 5 (0 incumbent) |
| FV + D3_S4_TWO_CHANNEL | 161 | 57 | 39 | 5 | 0.9952 | 4 / 4 | 4 (0 incumbent) |

**SQuAD** (ALL: 2000 queries x B = 6 slots)

| structural ordering | needed | needed admitted | novel gold admitted | gold evicted | slots on nuisance | lost / gained | lost by drop |
|---|---|---|---|---|---|---|---|
| FROZEN_M64/S4 (SAFE) | 49 | 24 | 17 | 3 | 0.9980 | -- | -- |
| FV + N0_RAW | 49 | 25 | 17 | 2 | 0.9979 | 2 / 3 | 2 (0 incumbent) |
| FV + N1_DEGREE_NORMALIZED | 49 | 25 | 17 | 2 | 0.9979 | 2 / 3 | 2 (0 incumbent) |
| FV + N2_SIZE_NORMALIZED | 49 | 25 | 17 | 2 | 0.9979 | 2 / 3 | 2 (0 incumbent) |
| FV + N3_EXPOSURE_NORMALIZED | 49 | 25 | 17 | 2 | 0.9979 | 2 / 3 | 2 (0 incumbent) |
| FV + D1_BEST_NODE_SCORE | 49 | 24 | 17 | 3 | 0.9980 | 2 / 2 | 2 (0 incumbent) |
| FV + D2_SEED_TIMES_SDIR | 49 | 24 | 17 | 3 | 0.9980 | 2 / 2 | 2 (0 incumbent) |
| FV + D3_S4_TWO_CHANNEL | 49 | 24 | 17 | 3 | 0.9980 | 2 / 2 | 2 (0 incumbent) |

## T12. STEP 7c -- the depth-matched control (is it the ORDER, or the LIST SIZE?)

STEP 7 changes two things at once: which partitions the structural ranking prefers, and how
many it hands F6 (~40 entries under M64 vs ~110-420 under FULL_VISITED). Here each calibrated
ordering is cut, per query, to EXACTLY the length the frozen M64 ordering produced for that
same query. K is read off the frozen ordering, not tuned. Everything else is unchanged, so
the only remaining difference is WHICH partitions occupy the slots -- the calibration question
in isolation.

| corpus | block | frozen len | FULL_VISITED len | frozen P50 | best depth-matched | net | p | net at full depth |
|---|---|---|---|---|---|---|---|---|
| MetaQA | hop3 | 47.5 | 305.5 | 0.2583 | 0.2583 (`D3_S4_TWO_CHANNEL`) | -1 | 1 | -14 |
| WebQSP | ALL | 36.6 | 417.1 | 0.7646 | 0.7689 (`N0_RAW`) | +6 | 0.109 | +6 |
| 2Wiki | ALL | 37.7 | 167.2 | 0.9435 | 0.9430 (`N3_EXPOSURE_NORMALIZED`) | -1 | 1 | -3 |
| MuSiQue | ALL | 28.2 | 111.7 | 0.9635 | 0.9630 (`N3_EXPOSURE_NORMALIZED`) | -1 | 1 | -1 |
| HotpotQA | ALL | 45.5 | 585.0 | 0.9505 | 0.9510 (`D3_S4_TWO_CHANNEL`) | +1 | 1 | +0 |
| SQuAD | ALL | 16.5 | 116.8 | 0.9875 | 0.9875 (`N3_EXPOSURE_NORMALIZED`) | +0 | 1 | +1 |

Depth-matching removes the significant harm and does not produce a gain: at the frozen list
length no re-scoring of the same evidence beats the frozen ordering on any corpus.

## T13. Dilution dose-response (`N0_RAW`, the same ordering cut at increasing depth)

| corpus | block | frozen | 1xK | 2xK | 4xK | FULL |
|---|---|---|---|---|---|---|
| MetaQA | hop3 | 0.2583 | 0.2553 / -3 | 0.2538 / -9* | 0.2508 / -13* | 0.2508 / -15* |
| WebQSP | ALL | 0.7646 | 0.7689 / +6 | 0.7696 / +7* | 0.7689 / +6 | 0.7689 / +6 |
| 2Wiki | ALL | 0.9435 | 0.9425 / -2 | 0.9415 / -4 | 0.9420 / -3 | 0.9415 / -4 |
| MuSiQue | ALL | 0.9635 | 0.9610 / -5 | 0.9625 / -2 | 0.9625 / -2 | 0.9625 / -2 |
| HotpotQA | ALL | 0.9505 | 0.9505 / +0 | 0.9500 / -1 | 0.9500 / -1 | 0.9500 / -1 |
| SQuAD | ALL | 0.9875 | 0.9875 / +0 | 0.9870 / -1 | 0.9880 / +1 | 0.9880 / +1 |

Cells are `exact P50 / net vs frozen`; `*` marks McNemar significance. On MetaQA the harm is
monotone in list length, which is what a dilution mechanism predicts and a mis-scoring
mechanism does not.
