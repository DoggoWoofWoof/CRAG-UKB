# L1 CANDIDATE ADMISSION -- TABLES

Frozen contract unchanged: P = exactly 50, B = 6, protected core 44, same substrate, no new
traversal.  Every production pool holds exactly the frozen per-query candidate count K(q).
`F6_PARITY_ON_SAFE_POOL` gates the harness: the re-expressed frozen selector must reproduce
`KB.f6_select` on the SAFE pool for every query.

## T0  STEP 0 -- the two B questions, and the B ladder at fixed P = 50

| corpus / hop | O0 | O1 | O2 | O3 | O4 | O5 | B now (O2-O1) | B after (O4-O3) |
|---|---|---|---|---|---|---|---|---|
| metaqa/hop1 | 0.9955 | 0.9970 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| metaqa/hop2 | 0.7297 | 0.8033 | 0.8033 | 0.9489 | 0.9940 | 0.9970 | 0.0000 | 0.0450 |
| metaqa/hop3 | 0.2583 | 0.3634 | 0.3664 | 0.6547 | 0.8408 | 0.9640 | 0.0030 | 0.1862 |
| metaqa/ALL | 0.6612 | 0.7212 | 0.7222 | 0.8679 | 0.9449 | 0.9870 | 0.0010 | 0.0771 |
| webqsp | 0.7646 | 0.7886 | 0.7886 | 0.9161 | 0.9331 | 0.9951 | 0.0000 | 0.0169 |
| musique_clean | 0.9635 | 0.9750 | 0.9750 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| 2wiki_clean | 0.9435 | 0.9515 | 0.9515 | 0.9915 | 0.9915 | 1.0000 | 0.0000 | 0.0000 |
| hotpotqa_clean | 0.9505 | 0.9645 | 0.9645 | 0.9920 | 0.9920 | 1.0000 | 0.0000 | 0.0000 |
| squad_clean | 0.9875 | 0.9945 | 0.9945 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

B ladder: protect `base_rank[:50-b]`, choose the remaining b perfectly from the universe.

| universe / hop | b=6 | b=8 | b=10 | b=14 | b=20 | b=30 | b=50 |
|---|---|---|---|---|---|---|---|
| SAFE universe, metaqa/hop2 | 0.8033 | 0.8033 | 0.8033 | 0.8033 | 0.8033 | 0.8033 | 0.8033 |
| SAFE universe, metaqa/hop3 | 0.3634 | 0.3634 | 0.3634 | 0.3664 | 0.3664 | 0.3664 | 0.3664 |
| SAFE universe, metaqa/ALL | 0.7212 | 0.7212 | 0.7212 | 0.7222 | 0.7222 | 0.7222 | 0.7222 |
| FULL_VISITED, metaqa/hop2 | 0.9489 | 0.9535 | 0.9625 | 0.9640 | 0.9760 | 0.9850 | 0.9940 |
| FULL_VISITED, metaqa/hop3 | 0.6547 | 0.6832 | 0.7027 | 0.7523 | 0.7838 | 0.8168 | 0.8408 |
| FULL_VISITED, metaqa/ALL | 0.8679 | 0.8789 | 0.8884 | 0.9054 | 0.9199 | 0.9339 | 0.9449 |

## T1  STEP 1/2 -- what the missing required partitions look like at the admission stage

| corpus / hop | missing | visited | has SRC atom | already admitted |
|---|---|---|---|---|
| metaqa/hop2 | 872 | 0.9977 | 0.9702 | 0.2810 |
| metaqa/hop3 | 4939 | 0.9674 | 0.8012 | 0.2792 |
| metaqa/ALL | 5815 | 0.9720 | 0.8263 | 0.2794 |
| webqsp | 1685 | 0.6896 | 0.2493 | 0.0890 |
| musique_clean | 75 | 1.0000 | 0.8133 | 0.3200 |
| 2wiki_clean | 117 | 0.8547 | 0.2479 | 0.1624 |
| hotpotqa_clean | 104 | 0.8365 | 0.3846 | 0.2885 |
| squad_clean | 25 | 1.0000 | 0.5200 | 0.5600 |

## T2  STEP 2/3/4 -- pool composition (all production pools are exactly K(q))

| corpus | K(q) | A1 filled | A2 retained SAFE | A2 new vs SAFE | UNION (diagnostic) |
|---|---|---|---|---|---|
| metaqa | 49.2 | 48.7 | 0.4 | 29.0 | 78.2 |
| webqsp | 34.4 | 33.2 | 0.2 | 20.5 | 55.0 |
| musique_clean | 18.6 | 18.4 | 0.1 | 9.1 | 27.6 |
| 2wiki_clean | 41.8 | 35.3 | 0.3 | 17.2 | 59.0 |
| hotpotqa_clean | 55.4 | 54.4 | 0.5 | 35.4 | 90.9 |
| squad_clean | 15.8 | 14.5 | 0.1 | 8.3 | 24.1 |

## T3  STEP 1 -- MISSING_REQUIRED_ADMISSION_RECALL by pool

| corpus / hop | SAFE_POOL | A1_SRC_ASSIGN | A2_HYBRID_ADMIT | A3_INTERLEAVE | A4_SUBSUME | SAFE_U_SRC |
|---|---|---|---|---|---|---|
| metaqa/hop2 | 0.2810 | 0.3956 | 0.3945 | 0.3796 | 0.3991 | 0.5092 |
| metaqa/hop3 | 0.2792 | 0.3140 | 0.3144 | 0.3152 | 0.3076 | 0.4355 |
| metaqa/ALL | 0.2794 | 0.3261 | 0.3262 | 0.3248 | 0.3212 | 0.4464 |
| webqsp | 0.0890 | 0.0944 | 0.1039 | 0.1181 | 0.1056 | 0.1519 |
| musique_clean | 0.3200 | 0.1733 | 0.1733 | 0.2800 | 0.2133 | 0.4133 |
| 2wiki_clean | 0.1624 | 0.1368 | 0.1368 | 0.1453 | 0.1538 | 0.2308 |
| hotpotqa_clean | 0.2885 | 0.1923 | 0.2308 | 0.1827 | 0.2019 | 0.3750 |
| squad_clean | 0.5600 | 0.1600 | 0.1600 | 0.2800 | 0.1600 | 0.5600 |

## T4  STEP 5 -- candidate-pool oracle, perfect selection at B = 6, exact P50

| corpus / hop | SAFE_POOL | A1_SRC_ASSIGN | A2_HYBRID_ADMIT | A3_INTERLEAVE | A4_SUBSUME | SAFE_U_SRC |
|---|---|---|---|---|---|---|
| metaqa/hop1 | 0.9970 | 0.9955 | 0.9955 | 0.9955 | 0.9955 | 0.9970 |
| metaqa/hop2 | 0.8033 | 0.8468 | 0.8468 | 0.8378 | 0.8498 | 0.8769 |
| metaqa/hop3 | 0.3634 | 0.3754 | 0.3754 | 0.3649 | 0.3589 | 0.4189 |
| metaqa/ALL | 0.7212 | 0.7392 | 0.7392 | 0.7327 | 0.7347 | 0.7643 |
| webqsp | 0.7886 | 0.7801 | 0.7822 | 0.7935 | 0.7900 | 0.8048 |
| musique_clean | 0.9750 | 0.9580 | 0.9590 | 0.9695 | 0.9695 | 0.9785 |
| 2wiki_clean | 0.9515 | 0.9400 | 0.9455 | 0.9465 | 0.9510 | 0.9555 |
| hotpotqa_clean | 0.9645 | 0.9420 | 0.9460 | 0.9490 | 0.9510 | 0.9690 |
| squad_clean | 0.9945 | 0.9820 | 0.9855 | 0.9880 | 0.9855 | 0.9945 |

## T5  STEP 6 -- the 2x2 causal matrix, exact P50 (net / p vs SAFE_POOL+F6)

**metaqa by hop**

| row | SAFE_POOL+F6 | SAFE_POOL+G4 | A2_HYBRID_ADMIT+F6 | A2_HYBRID_ADMIT+G4 | A3_INTERLEAVE+F6 | A3_INTERLEAVE+G4 | A4_SUBSUME+F6 | A4_SUBSUME+G4 |
|---|---|---|---|---|---|---|---|---|
| hop1 | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) | 0.9955 (+0, p=1.000) |
| hop2 | 0.7297 (+0, p=1.000) | 0.7222 (-5, p=0.302) | 0.7372 (+5, p=0.180) | 0.7237 (-4, p=0.455) | 0.7372 (+5, p=0.062) | 0.7237 (-4, p=0.455) | 0.7342 (+3, p=0.250) | 0.7207 (-6, p=0.210) |
| hop3 | 0.2583 (+0, p=1.000) | 0.2778 (+13, p=0.004)  SIG | 0.2703 (+8, p=0.008)  SIG | 0.2808 (+15, p=0.002)  SIG | 0.2673 (+6, p=0.031)  SIG | 0.2778 (+13, p=0.007)  SIG | 0.2643 (+4, p=0.125) | 0.2793 (+14, p=0.004)  SIG |
| ALL | 0.6612 (+0, p=1.000) | 0.6652 (+8, p=0.230) | 0.6677 (+13, p=0.002)  SIG | 0.6667 (+11, p=0.099) | 0.6667 (+11, p=0.001)  SIG | 0.6657 (+9, p=0.188) | 0.6647 (+7, p=0.016)  SIG | 0.6652 (+8, p=0.256) |

**all corpora, ALL**

| row | SAFE_POOL+F6 | SAFE_POOL+G4 | A2_HYBRID_ADMIT+F6 | A2_HYBRID_ADMIT+G4 | A3_INTERLEAVE+F6 | A3_INTERLEAVE+G4 | A4_SUBSUME+F6 | A4_SUBSUME+G4 |
|---|---|---|---|---|---|---|---|---|
| metaqa/ALL | 0.6612 (+0, p=1.000) | 0.6652 (+8, p=0.230) | 0.6677 (+13, p=0.002)  SIG | 0.6667 (+11, p=0.099) | 0.6667 (+11, p=0.001)  SIG | 0.6657 (+9, p=0.188) | 0.6647 (+7, p=0.016)  SIG | 0.6652 (+8, p=0.256) |
| webqsp/ALL | 0.7646 (+0, p=1.000) | 0.7632 (-2, p=0.625) | 0.7618 (-4, p=0.388) | 0.7604 (-6, p=0.146) | 0.7653 (+1, p=1.000) | 0.7639 (-1, p=1.000) | 0.7653 (+1, p=1.000) | 0.7611 (-5, p=0.227) |
| musique_clean/ALL | 0.9635 (+0, p=1.000) | 0.9600 (-7, p=0.039)  SIG | 0.9545 (-18, p=0.000)  SIG | 0.9535 (-20, p=0.000)  SIG | 0.9600 (-7, p=0.039)  SIG | 0.9575 (-12, p=0.002)  SIG | 0.9630 (-1, p=1.000) | 0.9600 (-7, p=0.092) |
| 2wiki_clean/ALL | 0.9435 (+0, p=1.000) | 0.9435 (+0, p=1.000) | 0.9385 (-10, p=0.006)  SIG | 0.9390 (-9, p=0.035)  SIG | 0.9385 (-10, p=0.002)  SIG | 0.9395 (-8, p=0.057) | 0.9430 (-1, p=1.000) | 0.9420 (-3, p=0.508) |
| hotpotqa_clean/ALL | 0.9505 (+0, p=1.000) | 0.9450 (-11, p=0.027)  SIG | 0.9400 (-21, p=0.001)  SIG | 0.9365 (-28, p=0.000)  SIG | 0.9420 (-17, p=0.001)  SIG | 0.9335 (-34, p=0.000)  SIG | 0.9445 (-12, p=0.029)  SIG | 0.9335 (-34, p=0.000)  SIG |
| squad_clean/ALL | 0.9875 (+0, p=1.000) | 0.9870 (-1, p=1.000) | 0.9845 (-6, p=0.109) | 0.9845 (-6, p=0.109) | 0.9850 (-5, p=0.125) | 0.9850 (-5, p=0.180) | 0.9835 (-8, p=0.008)  SIG | 0.9835 (-8, p=0.021)  SIG |

## T6  STEP 7 -- does hybrid admission preserve what plain G4 lost?

| corpus | partitions plain G4 lost | of which no SRC atom | preserved by A2+G4 | no-SRC ones preserved |
|---|---|---|---|---|
| metaqa | 191 | 10 | 0.0262 | 0.0000 |
| webqsp | 19 | 8 | 0.0526 | 0.0000 |
| musique_clean | 9 | 0 | 0.2222 | 0.0000 |
| 2wiki_clean | 3 | 3 | 0.0000 | 0.0000 |
| hotpotqa_clean | 18 | 10 | 0.0000 | 0.0000 |
| squad_clean | 2 | 2 | 0.0000 | 0.0000 |

## T7  STEP 7 -- changed-query audit, A2_HYBRID+G4 against each control

**metaqa**

| comparison | gains | losses | gain from a NEWLY admitted partition | losses with SRC | losses without SRC | lost but still in pool | lost partition evidence (dense/splade/struct/retcont) |
|---|---|---|---|---|---|---|---|
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+G4 | 4 | 1 | 3 | 1 | 0 | 1 | 0/0/1/0 |
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+F6 | 24 | 13 | 11 | 13 | 0 | 11 | 5/3/12/1 |
| A2_HYBRID_ADMIT+F6 vs SAFE_POOL+F6 | 15 | 2 | 0 | 2 | 0 | 0 | 0/0/2/0 |
| A3_INTERLEAVE+F6 vs SAFE_POOL+F6 | 11 | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 |
| A4_SUBSUME+F6 vs SAFE_POOL+F6 | 7 | 0 | 0 | 0 | 0 | 0 | 0/0/0/0 |
| A4_SUBSUME+G4 vs SAFE_POOL+F6 | 23 | 15 | 10 | 15 | 0 | 15 | 6/3/14/1 |

**webqsp**

| comparison | gains | losses | gain from a NEWLY admitted partition | losses with SRC | losses without SRC | lost but still in pool | lost partition evidence (dense/splade/struct/retcont) |
|---|---|---|---|---|---|---|---|
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+G4 | 2 | 6 | 2 | 4 | 2 | 1 | 2/2/5/1 |
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+F6 | 3 | 9 | 3 | 6 | 3 | 1 | 3/3/7/2 |
| A2_HYBRID_ADMIT+F6 vs SAFE_POOL+F6 | 4 | 8 | 0 | 5 | 3 | 0 | 3/3/6/2 |
| A3_INTERLEAVE+F6 vs SAFE_POOL+F6 | 4 | 3 | 0 | 2 | 1 | 0 | 2/2/1/2 |
| A4_SUBSUME+F6 vs SAFE_POOL+F6 | 3 | 2 | 0 | 1 | 1 | 0 | 0/1/2/0 |
| A4_SUBSUME+G4 vs SAFE_POOL+F6 | 3 | 8 | 2 | 6 | 2 | 6 | 2/4/6/2 |

**musique_clean**

| comparison | gains | losses | gain from a NEWLY admitted partition | losses with SRC | losses without SRC | lost but still in pool | lost partition evidence (dense/splade/struct/retcont) |
|---|---|---|---|---|---|---|---|
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+G4 | 4 | 17 | 2 | 13 | 4 | 2 | 17/15/5/14 |
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+F6 | 3 | 23 | 1 | 19 | 4 | 3 | 22/21/7/18 |
| A2_HYBRID_ADMIT+F6 vs SAFE_POOL+F6 | 2 | 20 | 1 | 16 | 4 | 0 | 20/18/4/18 |
| A3_INTERLEAVE+F6 vs SAFE_POOL+F6 | 1 | 8 | 0 | 6 | 2 | 0 | 8/7/0/8 |
| A4_SUBSUME+F6 vs SAFE_POOL+F6 | 2 | 3 | 0 | 1 | 2 | 0 | 3/2/0/3 |
| A4_SUBSUME+G4 vs SAFE_POOL+F6 | 3 | 10 | 0 | 8 | 2 | 7 | 9/9/4/6 |

**2wiki_clean**

| comparison | gains | losses | gain from a NEWLY admitted partition | losses with SRC | losses without SRC | lost but still in pool | lost partition evidence (dense/splade/struct/retcont) |
|---|---|---|---|---|---|---|---|
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+G4 | 0 | 9 | 0 | 2 | 7 | 1 | 6/8/2/8 |
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+F6 | 3 | 12 | 1 | 2 | 10 | 1 | 9/10/2/11 |
| A2_HYBRID_ADMIT+F6 vs SAFE_POOL+F6 | 1 | 11 | 0 | 2 | 9 | 0 | 8/10/1/11 |
| A3_INTERLEAVE+F6 vs SAFE_POOL+F6 | 0 | 10 | 0 | 3 | 7 | 0 | 8/9/0/10 |
| A4_SUBSUME+F6 vs SAFE_POOL+F6 | 1 | 2 | 0 | 0 | 2 | 0 | 0/2/0/2 |
| A4_SUBSUME+G4 vs SAFE_POOL+F6 | 3 | 6 | 2 | 0 | 6 | 4 | 4/4/1/5 |

**hotpotqa_clean**

| comparison | gains | losses | gain from a NEWLY admitted partition | losses with SRC | losses without SRC | lost but still in pool | lost partition evidence (dense/splade/struct/retcont) |
|---|---|---|---|---|---|---|---|
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+G4 | 3 | 20 | 2 | 7 | 13 | 3 | 19/19/5/20 |
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+F6 | 8 | 36 | 5 | 14 | 22 | 5 | 34/33/6/35 |
| A2_HYBRID_ADMIT+F6 vs SAFE_POOL+F6 | 10 | 31 | 0 | 10 | 21 | 0 | 30/29/5/30 |
| A3_INTERLEAVE+F6 vs SAFE_POOL+F6 | 4 | 21 | 0 | 7 | 14 | 0 | 20/18/0/21 |
| A4_SUBSUME+F6 vs SAFE_POOL+F6 | 7 | 19 | 0 | 4 | 15 | 0 | 18/16/1/19 |
| A4_SUBSUME+G4 vs SAFE_POOL+F6 | 3 | 37 | 2 | 13 | 24 | 18 | 35/33/6/36 |

**squad_clean**

| comparison | gains | losses | gain from a NEWLY admitted partition | losses with SRC | losses without SRC | lost but still in pool | lost partition evidence (dense/splade/struct/retcont) |
|---|---|---|---|---|---|---|---|
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+G4 | 1 | 6 | 0 | 2 | 4 | 0 | 6/6/1/6 |
| A2_HYBRID_ADMIT+G4 vs SAFE_POOL+F6 | 2 | 8 | 0 | 2 | 6 | 0 | 8/8/1/8 |
| A2_HYBRID_ADMIT+F6 vs SAFE_POOL+F6 | 2 | 8 | 0 | 2 | 6 | 0 | 8/8/1/8 |
| A3_INTERLEAVE+F6 vs SAFE_POOL+F6 | 1 | 6 | 0 | 2 | 4 | 0 | 6/5/1/6 |
| A4_SUBSUME+F6 vs SAFE_POOL+F6 | 0 | 8 | 0 | 2 | 6 | 0 | 8/8/0/8 |
| A4_SUBSUME+G4 vs SAFE_POOL+F6 | 1 | 9 | 0 | 3 | 6 | 1 | 9/9/1/8 |

## T8  STEP 8 -- query-local applicability, inference-safe statistics (no gate, no threshold)

**metaqa** -- gains 24, losses 13, unchanged 1961

| statistic | gain queries | loss queries | unchanged | AUC gain vs loss |
|---|---|---|---|---|
| src_atoms | 15.9167 | 19.7692 | 17.8409 | 0.2228 |
| src_targets | 237.4583 | 253.0769 | 233.5441 | 0.391 |
| src_conc | 0.187 | 0.1713 | 0.1858 | 0.5641 |
| src_safe_overlap | 0.1595 | 0.1684 | 0.1784 | 0.4199 |
| src_dense_agree | 0.3394 | 0.362 | 0.3472 | 0.3622 |
| src_splade_agree | 0.3717 | 0.3554 | 0.3639 | 0.633 |
| src_targets_per_source | 28.3425 | 24.1811 | 24.7485 | 0.6282 |
| frac_miss_with_src | 1.0 | 0.0 | 0.2701 | 1.0 |

**webqsp** -- gains 3, losses 9, unchanged 1407

| statistic | gain queries | loss queries | unchanged | AUC gain vs loss |
|---|---|---|---|---|
| src_atoms | 13.6667 | 21.0 | 15.0007 | 0.2963 |
| src_targets | 204.3333 | 259.1111 | 209.1891 | 0.4074 |
| src_conc | 0.2587 | 0.1643 | 0.2544 | 0.8519 |
| src_safe_overlap | 0.1207 | 0.0826 | 0.1249 | 0.7778 |
| src_dense_agree | 0.1417 | 0.1737 | 0.1831 | 0.4074 |
| src_splade_agree | 0.1375 | 0.1147 | 0.1437 | 0.4444 |
| src_targets_per_source | 16.8531 | 18.3915 | 18.8564 | 0.4444 |
| frac_miss_with_src | 1.0 | 0.0 | 0.0923 | 1.0 |

**musique_clean** -- gains 3, losses 23, unchanged 1974

| statistic | gain queries | loss queries | unchanged | AUC gain vs loss |
|---|---|---|---|---|
| src_atoms | 21.0 | 22.0435 | 18.8627 | 0.4638 |
| src_targets | 102.0 | 116.087 | 104.9767 | 0.3333 |
| src_conc | 0.1277 | 0.1206 | 0.1621 | 0.5942 |
| src_safe_overlap | 0.1541 | 0.1549 | 0.1628 | 0.4638 |
| src_dense_agree | 0.5638 | 0.5436 | 0.4953 | 0.6232 |
| src_splade_agree | 0.5525 | 0.5866 | 0.5438 | 0.4928 |
| src_targets_per_source | 13.3086 | 17.9555 | 16.1988 | 0.2754 |
| frac_miss_with_src | 1.0 | 0.0 | 0.0284 | 1.0 |

**2wiki_clean** -- gains 3, losses 12, unchanged 1985

| statistic | gain queries | loss queries | unchanged | AUC gain vs loss |
|---|---|---|---|---|
| src_atoms | 13.6667 | 11.0833 | 9.9693 | 0.8333 |
| src_targets | 83.3333 | 99.3333 | 105.063 | 0.5 |
| src_conc | 0.238 | 0.345 | 0.3675 | 0.2778 |
| src_safe_overlap | 0.2753 | 0.2065 | 0.2514 | 0.7778 |
| src_dense_agree | 0.3676 | 0.2901 | 0.3491 | 0.8056 |
| src_splade_agree | 0.3981 | 0.3246 | 0.3602 | 0.6944 |
| src_targets_per_source | 8.6101 | 12.0607 | 12.1154 | 0.4444 |
| frac_miss_with_src | 1.0 | 0.0 | 0.0126 | 1.0 |

**hotpotqa_clean** -- gains 8, losses 36, unchanged 1956

| statistic | gain queries | loss queries | unchanged | AUC gain vs loss |
|---|---|---|---|---|
| src_atoms | 11.375 | 13.2222 | 12.4279 | 0.401 |
| src_targets | 275.625 | 344.5278 | 293.9269 | 0.3802 |
| src_conc | 0.2307 | 0.2358 | 0.2639 | 0.5174 |
| src_safe_overlap | 0.1294 | 0.1212 | 0.1263 | 0.559 |
| src_dense_agree | 0.1719 | 0.1289 | 0.1457 | 0.7083 |
| src_splade_agree | 0.1606 | 0.1308 | 0.1397 | 0.559 |
| src_targets_per_source | 34.9971 | 34.8854 | 30.9589 | 0.4896 |
| frac_miss_with_src | 0.625 | 0.0 | 0.0174 | 0.8125 |

**squad_clean** -- gains 2, losses 8, unchanged 1990

| statistic | gain queries | loss queries | unchanged | AUC gain vs loss |
|---|---|---|---|---|
| src_atoms | 6.0 | 14.25 | 11.507 | 0.0625 |
| src_targets | 118.5 | 109.125 | 96.3819 | 0.7188 |
| src_conc | 0.2543 | 0.1378 | 0.1291 | 1.0 |
| src_safe_overlap | 0.1223 | 0.1237 | 0.1045 | 0.375 |
| src_dense_agree | 0.3462 | 0.4206 | 0.3407 | 0.0 |
| src_splade_agree | 0.3034 | 0.4608 | 0.3864 | 0.0625 |
| src_targets_per_source | 34.5429 | 19.97 | 20.1027 | 1.0 |
| frac_miss_with_src | 1.0 | 0.0 | 0.0055 | 1.0 |

## T9  STEP 10 -- admission latency (ms/query)

| corpus | frozen admission | A1 SRC assignment | A2 hybrid | F6 selector | G4 selector |
|---|---|---|---|---|---|
| metaqa | 0.0893 | 0.3459 | 0.1049 | 0.1456 | 0.5056 |
| webqsp | 0.0506 | 0.224 | 0.0768 | 0.0789 | 0.3248 |
| musique_clean | 0.0572 | 0.5925 | 0.1334 | 0.0774 | 0.5507 |
| 2wiki_clean | 0.1033 | 0.3523 | 0.143 | 0.1661 | 0.6173 |
| hotpotqa_clean | 0.136 | 0.5017 | 0.1942 | 0.2302 | 0.8558 |
| squad_clean | 0.039 | 0.2734 | 0.0966 | 0.0526 | 0.3772 |

MetaQA frozen L1 total **15.565 ms/q**; admission overhead (A1+A2) **0.4508 ms/q = 2.81%** of L1.
