# L1 CONVERSION PHASE -- TABLES

Generated from `diag/gate.json`, `diag/main.json`, `diag/why.json`, `diag/fin.json`. `SAFE` = the frozen `R0 / B6_S4_F6_Ms64_Mr32` reference.

## T1 -- STEP 1 symmetric U_RANK

| corpus | mean len(U) | incumbents carrying u_rank | U_PC5 proposals/query | truly-new offered/query |
|---|---|---|---|---|
| metaqa | 306.0 | 50.0 | 256.0 | 218.6 |
| webqsp | 306.0 | 50.0 | 256.0 | 245.8 |
| 2wiki | 306.0 | 50.0 | 256.0 | 233.0 |
| musique | 136.0 | 50.0 | 86.0 | 73.4 |
| hotpot | 306.0 | 50.0 | 256.0 | 234.6 |
| squad | 190.0 | 50.0 | 140.0 | 130.2 |

Determinism (MetaQA, two independent builds): `True`. All 50 incumbents receive a u_rank, so incumbents and challengers are scored under identical rank semantics.

## T2 -- STEP 5 bookends (MetaQA)

| slice | safe | current pool oracle | u pc5 pool oracle | unlimited pool oracle | full universe p50 oracle |
|---|---|---|---|---|---|
| hop1 | 0.9955 | 0.9970 | 0.9985 | 1.0000 | 1.0000 |
| hop2 | 0.7297 | 0.8033 | 0.9474 | 0.9505 | 0.9970 |
| hop3 | 0.2583 | 0.3634 | 0.6366 | 0.6802 | 0.9640 |
| ALL | 0.6612 | 0.7212 | 0.8609 | 0.8769 | 0.9870 |

## T3 -- STEP 5 MetaQA primary, per hop

| rule | slice | ACTUAL | delta vs SAFE | newly covered | newly uncovered | net | McNemar p | sig |
|---|---|---|---|---|---|---|---|---|
| S0_SAFE | hop1 | 0.9955 | +0.0000 | 0 | 0 | +0 | 1 | ns |
| S0_SAFE | hop2 | 0.7297 | +0.0000 | 0 | 0 | +0 | 1 | ns |
| S0_SAFE | hop3 | 0.2583 | +0.0000 | 0 | 0 | +0 | 1 | ns |
| S0_SAFE | ALL | 0.6612 | +0.0000 | 0 | 0 | +0 | 1 | ns |
| S1_CSU_RAW | hop1 | 0.9955 | +0.0000 | 0 | 0 | +0 | 1 | ns |
| S1_CSU_RAW | hop2 | 0.7402 | +0.0105 | 8 | 1 | +7 | 0.03906 | **sig** |
| S1_CSU_RAW | hop3 | 0.2658 | +0.0075 | 7 | 2 | +5 | 0.1797 | ns |
| S1_CSU_RAW | ALL | 0.6672 | +0.0060 | 15 | 3 | +12 | 0.00754 | **sig** |
| S2_CSU_LIFT | hop1 | 0.9955 | +0.0000 | 0 | 0 | +0 | 1 | ns |
| S2_CSU_LIFT | hop2 | 0.7267 | -0.0030 | 6 | 8 | -2 | 0.7905 | ns |
| S2_CSU_LIFT | hop3 | 0.2568 | -0.0015 | 3 | 4 | -1 | 1 | ns |
| S2_CSU_LIFT | ALL | 0.6597 | -0.0015 | 9 | 12 | -3 | 0.6636 | ns |
| S3_CU_ABLATION | hop1 | 0.9955 | +0.0000 | 0 | 0 | +0 | 1 | ns |
| S3_CU_ABLATION | hop2 | 0.7117 | -0.0180 | 6 | 18 | -12 | 0.02266 | **sig** |
| S3_CU_ABLATION | hop3 | 0.2598 | +0.0015 | 7 | 6 | +1 | 1 | ns |
| S3_CU_ABLATION | ALL | 0.6557 | -0.0055 | 13 | 24 | -11 | 0.09887 | ns |

## T4 -- STEP 5 conversion bookkeeping (MetaQA)

| rule | mean churn | new U_PC5 partitions selected | per query | queries with 1 or more | gold-bearing U proposals admitted | gold-bearing incumbents evicted | queries whose P50 changes |
|---|---|---|---|---|---|---|---|
| S0_SAFE | 4.659 | 0 | 0.0 | 0 | 0 | 183 | 0 |
| S1_CSU_RAW | 5.133 | 18 | 0.009 | 13 | 0 | 183 | 1990 |
| S2_CSU_LIFT | 4.461 | 1423 | 0.712 | 1045 | 38 | 165 | 1978 |
| S3_CU_ABLATION | 4.407 | 2380 | 1.191 | 1419 | 66 | 183 | 1994 |

## T5 -- STEP 6 CONVERSION_EFFICIENCY = (ACTUAL - SAFE) / (U_PC5_POOL_ORACLE - SAFE)

| rule | slice | SAFE | U_PC5 pool oracle | available gap | ACTUAL | converted | CONVERSION_EFFICIENCY |
|---|---|---|---|---|---|---|---|
| S1_CSU_RAW | metaqa hop2 | 0.7297 | 0.9474 | +0.2177 | 0.7402 | +0.0105 | +4.83% |
| S1_CSU_RAW | metaqa hop3 | 0.2583 | 0.6366 | +0.3783 | 0.2658 | +0.0075 | +1.98% |
| S1_CSU_RAW | metaqa ALL | 0.6612 | 0.8609 | +0.1997 | 0.6672 | +0.0060 | +3.01% |
| S1_CSU_RAW | webqsp ALL | 0.7646 | 0.9027 | +0.1381 | 0.7632 | -0.0014 | -1.02% |
| S2_CSU_LIFT | metaqa hop2 | 0.7297 | 0.9474 | +0.2177 | 0.7267 | -0.0030 | -1.38% |
| S2_CSU_LIFT | metaqa hop3 | 0.2583 | 0.6366 | +0.3783 | 0.2568 | -0.0015 | -0.40% |
| S2_CSU_LIFT | metaqa ALL | 0.6612 | 0.8609 | +0.1997 | 0.6597 | -0.0015 | -0.75% |
| S2_CSU_LIFT | webqsp ALL | 0.7646 | 0.9027 | +0.1381 | 0.7604 | -0.0042 | -3.06% |
| S3_CU_ABLATION | metaqa hop2 | 0.7297 | 0.9474 | +0.2177 | 0.7117 | -0.0180 | -8.28% |
| S3_CU_ABLATION | metaqa hop3 | 0.2583 | 0.6366 | +0.3783 | 0.2598 | +0.0015 | +0.40% |
| S3_CU_ABLATION | metaqa ALL | 0.6612 | 0.8609 | +0.1997 | 0.6557 | -0.0055 | -2.76% |
| S3_CU_ABLATION | webqsp ALL | 0.7646 | 0.9027 | +0.1381 | 0.7519 | -0.0127 | -9.18% |

## T6 -- STEP 7 all six corpora, ALL coverage

| rule | corpus | SAFE | ACTUAL | delta | gained | lost | net | McNemar p | sig | mean churn |
|---|---|---|---|---|---|---|---|---|---|---|
| S0_SAFE | metaqa | 0.6612 | 0.6612 | +0.0000 | 0 | 0 | +0 | 1 | ns | 4.659 |
| S0_SAFE | webqsp | 0.7646 | 0.7646 | +0.0000 | 0 | 0 | +0 | 1 | ns | 4.968 |
| S0_SAFE | 2wiki | 0.9435 | 0.9435 | +0.0000 | 0 | 0 | +0 | 1 | ns | 4.259 |
| S0_SAFE | musique | 0.9635 | 0.9635 | +0.0000 | 0 | 0 | +0 | 1 | ns | 4.312 |
| S0_SAFE | hotpot | 0.9505 | 0.9505 | +0.0000 | 0 | 0 | +0 | 1 | ns | 5.098 |
| S0_SAFE | squad | 0.9875 | 0.9875 | +0.0000 | 0 | 0 | +0 | 1 | ns | 3.974 |
| S1_CSU_RAW | metaqa | 0.6612 | 0.6672 | +0.0060 | 15 | 3 | +12 | 0.00754 | **sig** | 5.133 |
| S1_CSU_RAW | webqsp | 0.7646 | 0.7632 | -0.0014 | 6 | 8 | -2 | 0.7905 | ns | 4.903 |
| S1_CSU_RAW | 2wiki | 0.9435 | 0.9400 | -0.0035 | 4 | 11 | -7 | 0.1185 | ns | 5.117 |
| S1_CSU_RAW | musique | 0.9635 | 0.9535 | -0.0100 | 6 | 26 | -20 | 0.00054 | **sig** | 4.785 |
| S1_CSU_RAW | hotpot | 0.9505 | 0.9445 | -0.0060 | 4 | 16 | -12 | 0.01182 | **sig** | 5.102 |
| S1_CSU_RAW | squad | 0.9875 | 0.9820 | -0.0055 | 1 | 12 | -11 | 0.00342 | **sig** | 4.845 |
| S2_CSU_LIFT | metaqa | 0.6612 | 0.6597 | -0.0015 | 9 | 12 | -3 | 0.6636 | ns | 4.461 |
| S2_CSU_LIFT | webqsp | 0.7646 | 0.7604 | -0.0042 | 10 | 16 | -6 | 0.3269 | ns | 4.417 |
| S2_CSU_LIFT | 2wiki | 0.9435 | 0.9445 | +0.0010 | 5 | 3 | +2 | 0.7266 | ns | 4.316 |
| S2_CSU_LIFT | musique | 0.9635 | 0.9600 | -0.0035 | 9 | 16 | -7 | 0.2295 | ns | 4.048 |
| S2_CSU_LIFT | hotpot | 0.9505 | 0.9540 | +0.0035 | 11 | 4 | +7 | 0.1185 | ns | 4.817 |
| S2_CSU_LIFT | squad | 0.9875 | 0.9885 | +0.0010 | 5 | 3 | +2 | 0.7266 | ns | 4.332 |
| S3_CU_ABLATION | metaqa | 0.6612 | 0.6557 | -0.0055 | 13 | 24 | -11 | 0.09887 | ns | 4.407 |
| S3_CU_ABLATION | webqsp | 0.7646 | 0.7519 | -0.0127 | 12 | 30 | -18 | 0.00792 | **sig** | 4.281 |
| S3_CU_ABLATION | 2wiki | 0.9435 | 0.9465 | +0.0030 | 7 | 1 | +6 | 0.07031 | ns | 4.215 |
| S3_CU_ABLATION | musique | 0.9635 | 0.9610 | -0.0025 | 12 | 17 | -5 | 0.4583 | ns | 3.941 |
| S3_CU_ABLATION | hotpot | 0.9505 | 0.9535 | +0.0030 | 13 | 7 | +6 | 0.2632 | ns | 4.808 |
| S3_CU_ABLATION | squad | 0.9875 | 0.9895 | +0.0020 | 7 | 3 | +4 | 0.3438 | ns | 4.338 |

## T7 -- STEP 7 pooled over all six corpora

| rule | pooled delta | macro delta | worst corpus | corpora improved | gained | lost | net | McNemar p | sig |
|---|---|---|---|---|---|---|---|---|---|
| S1_CSU_RAW | -0.00350 | -0.00340 | -0.01000 | 1/6 | 36 | 76 | -40 | 0.0002 | **sig** |
| S2_CSU_LIFT | -0.00044 | -0.00062 | -0.00423 | 3/6 | 49 | 54 | -5 | 0.694 | ns |
| S3_CU_ABLATION | -0.00158 | -0.00212 | -0.01268 | 3/6 | 64 | 82 | -18 | 0.159 | ns |

## T8 -- why the admitted gold does not convert

| slice | rule | feasible (pool oracle) | covered (actual) | feasible-but-uncovered q | mean size of miss on those q | frac of miss selected (all q) | frac of miss selected (gap q) |
|---|---|---|---|---|---|---|---|
| metaqa hop3 | S0_SAFE | 0.3634 | 0.2583 | 70 | 1.61 | 0.067 | 0.085 |
| metaqa hop3 | S1_CSU_RAW | 0.6366 | 0.2658 | 247 | 2.4 | 0.106 | 0.089 |
| metaqa hop3 | S2_CSU_LIFT | 0.6366 | 0.2568 | 253 | 2.38 | 0.073 | 0.064 |
| metaqa hop3 | S3_CU_ABLATION | 0.6366 | 0.2598 | 251 | 2.39 | 0.055 | 0.040 |
| webqsp ALL | S0_SAFE | 0.7886 | 0.7646 | 34 | 1.15 | 0.143 | 0.044 |
| webqsp ALL | S1_CSU_RAW | 0.9027 | 0.7632 | 198 | 1.97 | 0.145 | 0.063 |
| webqsp ALL | S2_CSU_LIFT | 0.9027 | 0.7604 | 202 | 1.95 | 0.141 | 0.069 |
| webqsp ALL | S3_CU_ABLATION | 0.9027 | 0.7519 | 214 | 1.9 | 0.108 | 0.061 |

## T9 -- where the needed-but-unselected partitions rank in the score order of the rule

| slice | rule | n | rank p50 | rank p90 | in top-12 | in top-50 | src: base50 | src: challenger | src: truly new |
|---|---|---|---|---|---|---|---|---|---|
| metaqa hop3 | S0_SAFE | 162 | 15 | 36 | 68 | 158 | 29 | 133 | 0 |
| metaqa hop3 | S1_CSU_RAW | 787 | 53 | 183 | 81 | 368 | 66 | 237 | 484 |
| metaqa hop3 | S2_CSU_LIFT | 828 | 51 | 180 | 84 | 400 | 58 | 303 | 467 |
| metaqa hop3 | S3_CU_ABLATION | 869 | 62 | 214 | 60 | 378 | 65 | 341 | 463 |
| webqsp ALL | S0_SAFE | 36 | 17 | 36 | 10 | 36 | 23 | 13 | 0 |
| webqsp ALL | S1_CSU_RAW | 351 | 44 | 146 | 45 | 185 | 40 | 36 | 275 |
| webqsp ALL | S2_CSU_LIFT | 351 | 50 | 163 | 27 | 175 | 38 | 52 | 261 |
| webqsp ALL | S3_CU_ABLATION | 365 | 43 | 140 | 35 | 202 | 46 | 61 | 258 |

Rank is the position among all scored candidates (92-290 of them, see T10); only the top 6 are taken.

## T10 -- STEP 10 latency and cache accounting

| corpus | u_rank lookup+merge (ms/query) | selector SAFE (ms/query) | selector S1_CSU_RAW (ms/query) | total added (ms/query) | mean candidates scored | U cache (MB) | partition-graph cache (MB) |
|---|---|---|---|---|---|---|---|
| metaqa | 0.020 | 0.040 | 0.249 | 0.229 | 267.7 | 1.48 | 0.06 |
| webqsp | 0.026 | 0.025 | 0.706 | 0.707 | 280.1 | 1.26 | 1.54 |
| 2wiki | 0.022 | 0.034 | 0.258 | 0.246 | 275.1 | 1.51 | 0.10 |
| musique | 0.011 | 0.013 | 0.069 | 0.067 | 92.0 | 0.55 | 0.02 |
| hotpot | 0.022 | 0.051 | 0.285 | 0.256 | 289.8 | 1.81 | 0.91 |
| squad | 0.012 | 0.011 | 0.120 | 0.121 | 146.0 | 0.77 | 0.03 |

`ONLINE_GRAPH_EDGES_TOUCHED = 0`. No encoder work and no graph traversal at query time: `G_GRAPH_NBR_RAW` reads precomputed depth-1 neighbour rows and merges them arithmetically.
