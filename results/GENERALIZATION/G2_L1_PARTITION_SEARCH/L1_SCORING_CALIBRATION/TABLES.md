# L1 FINAL SCORING PHASE — TABLES

`*` p<0.05, `**` p<0.01 (exact McNemar, paired, DEV only). SAFE_F6 = `B6_S4_F6_Ms64_Mr32` = R0.


## T1 — STEP 0. The capacity claim with its candidate universe attached (B=6)

| corpus | CAP_P50 (C) | POOL_UNREACHABLE current | POOL_UNREACHABLE PPR-expanded | CAP_B6 current pool (A) | CAP_B6 PPR-expanded pool (B) | SOLVABLE current | SOLVABLE PPR-exp |
|---|---|---|---|---|---|---|---|
| MetaQA | 26 | 529 | 440 | 2 | 3 | 1441 | 1529 |
| WebQSP | 7 | 293 | 265 | 0 | 0 | 1119 | 1147 |
| 2Wiki | 0 | 97 | 81 | 0 | 0 | 1903 | 1919 |
| MuSiQue | 0 | 50 | 31 | 0 | 0 | 1950 | 1969 |
| HotpotQA | 0 | 71 | 68 | 0 | 0 | 1929 | 1932 |
| SQuAD | 0 | 11 | 8 | 0 | 0 | 1989 | 1992 |

## T2 — STEP 0. CAP_B by boundary width, both universes

| corpus | B=6 cur / ppr-exp | B=8 cur / ppr-exp | B=12 cur / ppr-exp |
|---|---|---|---|
| MetaQA | 2 / 3 | 2 / 2 | 0 / 0 |
| WebQSP | 0 / 0 | 0 / 0 | 0 / 0 |
| 2Wiki | 0 / 0 | 0 / 0 | 0 / 0 |
| MuSiQue | 0 / 0 | 0 / 0 | 0 / 0 |
| HotpotQA | 0 / 0 | 0 / 0 | 0 / 0 |
| SQuAD | 0 / 0 | 0 / 0 | 0 / 0 |

## T3 — STEP 0 DIAGNOSTIC. PPR admitted as a fourth co-equal channel (ALL coverage, Δ vs BASE)

| corpus | B | SAFE_F6 Δ | F6+PPR Δ | Δ(F6+PPR − F6) | McNemar p |
|---|---|---|---|---|---|
| MetaQA | 6 | +0.0025 | -0.0070 | -0.0095** | 0.0006 |
| MetaQA | 8 | +0.0050 | -0.0080 | -0.0130** | 0.0000 |
| MetaQA | 12 | +0.0060 | -0.0085 | -0.0145** | 0.0001 |
| WebQSP | 6 | +0.0028 | -0.0049 | -0.0078* | 0.0127 |
| WebQSP | 8 | +0.0000 | -0.0092 | -0.0092* | 0.0106 |
| WebQSP | 12 | -0.0063 | -0.0092 | -0.0028 | 0.4807 |
| 2Wiki | 6 | +0.0060 | +0.0055 | -0.0005 | 1.0000 |
| 2Wiki | 8 | +0.0055 | +0.0055 | +0.0000 | 1.0000 |
| 2Wiki | 12 | +0.0045 | +0.0040 | -0.0005 | 1.0000 |
| MuSiQue | 6 | +0.0070 | -0.0010 | -0.0080** | 0.0037 |
| MuSiQue | 8 | +0.0090 | -0.0010 | -0.0100** | 0.0003 |
| MuSiQue | 12 | +0.0085 | +0.0005 | -0.0080** | 0.0037 |
| HotpotQA | 6 | +0.0160 | +0.0105 | -0.0055* | 0.0266 |
| HotpotQA | 8 | +0.0175 | +0.0100 | -0.0075* | 0.0107 |
| HotpotQA | 12 | +0.0210 | +0.0105 | -0.0105** | 0.0001 |
| SQuAD | 6 | +0.0070 | +0.0055 | -0.0015 | 0.3750 |
| SQuAD | 8 | +0.0080 | +0.0055 | -0.0025 | 0.1250 |
| SQuAD | 12 | +0.0095 | +0.0065 | -0.0030 | 0.1460 |

## T4 — STEP 1. The current failure distribution (frozen R0 selector, B=6)

| corpus | swaps | GOOD admissions | BAD evictions | GOOD/BAD | queries with the decisive cross-list pattern | of those, costing a gold partition |
|---|---|---|---|---|---|---|
| MetaQA | 9309 | 257 | 183 | 1.40 | 1 | 0 |
| WebQSP | 7049 | 91 | 90 | 1.01 | 544 | 28 |
| 2Wiki | 8518 | 20 | 6 | 3.33 | 43 | 0 |
| MuSiQue | 8624 | 26 | 11 | 2.36 | 0 | 0 |
| HotpotQA | 10197 | 39 | 5 | 7.80 | 80 | 0 |
| SQuAD | 7948 | 17 | 3 | 5.67 | 0 | 0 |

## T5 — STEP 1. Channel profile of every evicted incumbent / admitted challenger

`C` = has canonical partition rank, `S` = has S4 structural rank, `R` = has node-retrieval rank. `gold` = of those, how many carried a gold partition.

| corpus | side | profile | n | gold |
|---|---|---|---|---|
| MetaQA | evicted | `C--` | 7906 | 145 |
| MetaQA | evicted | `CS-` | 977 | 35 |
| MetaQA | evicted | `C-R` | 426 | 3 |
| MetaQA | admitted | `C-R` | 3383 | 32 |
| MetaQA | admitted | `CS-` | 3064 | 143 |
| MetaQA | admitted | `CSR` | 2753 | 79 |
| MetaQA | admitted | `-SR` | 108 | 3 |
| MetaQA | admitted | `-S-` | 1 | 0 |
| WebQSP | evicted | `C--` | 6954 | 90 |
| WebQSP | evicted | `CS-` | 83 | 0 |
| WebQSP | evicted | `C-R` | 12 | 0 |
| WebQSP | admitted | `CS-` | 3487 | 55 |
| WebQSP | admitted | `C-R` | 1837 | 24 |
| WebQSP | admitted | `-S-` | 1467 | 4 |
| WebQSP | admitted | `CSR` | 248 | 8 |
| WebQSP | admitted | `--R` | 6 | 0 |
| WebQSP | admitted | `-SR` | 4 | 0 |
| 2Wiki | evicted | `C--` | 7833 | 5 |
| 2Wiki | evicted | `CS-` | 354 | 1 |
| 2Wiki | evicted | `C-R` | 331 | 0 |
| 2Wiki | admitted | `C-R` | 4049 | 19 |
| 2Wiki | admitted | `CS-` | 3246 | 0 |
| 2Wiki | admitted | `CSR` | 1139 | 1 |
| 2Wiki | admitted | `-S-` | 81 | 0 |
| 2Wiki | admitted | `-SR` | 3 | 0 |
| MuSiQue | evicted | `C--` | 7971 | 9 |
| MuSiQue | evicted | `CS-` | 552 | 1 |
| MuSiQue | evicted | `C-R` | 101 | 1 |
| MuSiQue | admitted | `CS-` | 6535 | 7 |
| MuSiQue | admitted | `C-R` | 1434 | 14 |
| MuSiQue | admitted | `CSR` | 655 | 5 |
| HotpotQA | evicted | `C--` | 9770 | 2 |
| HotpotQA | evicted | `CS-` | 301 | 2 |
| HotpotQA | evicted | `C-R` | 126 | 1 |
| HotpotQA | admitted | `C-R` | 5390 | 26 |
| HotpotQA | admitted | `CS-` | 2499 | 0 |
| HotpotQA | admitted | `CSR` | 1978 | 12 |
| HotpotQA | admitted | `-SR` | 167 | 1 |
| HotpotQA | admitted | `-S-` | 163 | 0 |
| SQuAD | evicted | `C--` | 7704 | 3 |
| SQuAD | evicted | `CS-` | 150 | 0 |
| SQuAD | evicted | `C-R` | 94 | 0 |
| SQuAD | admitted | `CS-` | 4218 | 1 |
| SQuAD | admitted | `C-R` | 3088 | 9 |
| SQuAD | admitted | `CSR` | 642 | 7 |

## T6 — STEP 1. Counterfactual: what each rule does with R0's own gold decisions

| corpus | rule | R0 BAD evictions now RETAINED | still evicted | R0 GOOD admissions still ADMITTED | now missed | net gold slots vs R0 |
|---|---|---|---|---|---|---|
| MetaQA | R0_RAW_RRF | 0 | 183 | 257 | 0 | +0 |
| MetaQA | R1_RRF_LIFT | 18 | 165 | 206 | 51 | -33 |
| MetaQA | R2_RRF_UNIT | 9 | 174 | 217 | 40 | -31 |
| MetaQA | R3_SIGNED_RET_RESIDUAL | 2 | 181 | 181 | 76 | -74 |
| MetaQA | R4_POSITIVE_RET_RESIDUAL | 11 | 172 | 201 | 56 | -45 |
| MetaQA | R5_PAIRWISE_RESIDUAL | 6 | 177 | 197 | 60 | -54 |
| MetaQA | R6_PARETO_SWAP | 183 | 0 | 0 | 257 | -74 |
| WebQSP | R0_RAW_RRF | 0 | 90 | 91 | 0 | +0 |
| WebQSP | R1_RRF_LIFT | 31 | 59 | 61 | 30 | +1 |
| WebQSP | R2_RRF_UNIT | 2 | 88 | 59 | 32 | -30 |
| WebQSP | R3_SIGNED_RET_RESIDUAL | 0 | 90 | 43 | 48 | -48 |
| WebQSP | R4_POSITIVE_RET_RESIDUAL | 2 | 88 | 54 | 37 | -35 |
| WebQSP | R5_PAIRWISE_RESIDUAL | 4 | 86 | 53 | 38 | -34 |
| WebQSP | R6_PARETO_SWAP | 90 | 0 | 0 | 91 | -1 |
| 2Wiki | R0_RAW_RRF | 0 | 6 | 20 | 0 | +0 |
| 2Wiki | R1_RRF_LIFT | 1 | 5 | 14 | 6 | -5 |
| 2Wiki | R2_RRF_UNIT | 0 | 6 | 16 | 4 | -4 |
| 2Wiki | R3_SIGNED_RET_RESIDUAL | 0 | 6 | 11 | 9 | -9 |
| 2Wiki | R4_POSITIVE_RET_RESIDUAL | 0 | 6 | 4 | 16 | -16 |
| 2Wiki | R5_PAIRWISE_RESIDUAL | 0 | 6 | 2 | 18 | -18 |
| 2Wiki | R6_PARETO_SWAP | 6 | 0 | 0 | 20 | -14 |
| MuSiQue | R0_RAW_RRF | 0 | 11 | 26 | 0 | +0 |
| MuSiQue | R1_RRF_LIFT | 0 | 11 | 21 | 5 | -5 |
| MuSiQue | R2_RRF_UNIT | 0 | 11 | 24 | 2 | -2 |
| MuSiQue | R3_SIGNED_RET_RESIDUAL | 0 | 11 | 24 | 2 | -2 |
| MuSiQue | R4_POSITIVE_RET_RESIDUAL | 0 | 11 | 17 | 9 | -9 |
| MuSiQue | R5_PAIRWISE_RESIDUAL | 0 | 11 | 16 | 10 | -10 |
| MuSiQue | R6_PARETO_SWAP | 11 | 0 | 0 | 26 | -15 |
| HotpotQA | R0_RAW_RRF | 0 | 5 | 39 | 0 | +0 |
| HotpotQA | R1_RRF_LIFT | 3 | 2 | 27 | 12 | -9 |
| HotpotQA | R2_RRF_UNIT | 0 | 5 | 30 | 9 | -9 |
| HotpotQA | R3_SIGNED_RET_RESIDUAL | 0 | 5 | 17 | 22 | -22 |
| HotpotQA | R4_POSITIVE_RET_RESIDUAL | 2 | 3 | 14 | 25 | -23 |
| HotpotQA | R5_PAIRWISE_RESIDUAL | 2 | 3 | 17 | 22 | -20 |
| HotpotQA | R6_PARETO_SWAP | 5 | 0 | 0 | 39 | -34 |
| SQuAD | R0_RAW_RRF | 0 | 3 | 17 | 0 | +0 |
| SQuAD | R1_RRF_LIFT | 0 | 3 | 14 | 3 | -3 |
| SQuAD | R2_RRF_UNIT | 0 | 3 | 15 | 2 | -2 |
| SQuAD | R3_SIGNED_RET_RESIDUAL | 0 | 3 | 14 | 3 | -3 |
| SQuAD | R4_POSITIVE_RET_RESIDUAL | 0 | 3 | 12 | 5 | -5 |
| SQuAD | R5_PAIRWISE_RESIDUAL | 0 | 3 | 13 | 4 | -4 |
| SQuAD | R6_PARETO_SWAP | 3 | 0 | 0 | 17 | -14 |

## T7 — STEP 8. R0–R6 on all six corpora (ALL coverage, full DEV)

| corpus | method | ALL | Δ vs BASE | Δ vs SAFE_F6 | McNemar vs BASE | McNemar vs SAFE_F6 | newly covered | newly uncovered | churn | swaps | gold admitted | gold evicted | GOOD | BAD | GOOD/BAD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | BASE | 0.6587 | +0.0000 | -0.0025 | — | — | — | — | 0.00 | 0 | — | — | — | — | — |
| MetaQA | R0_RAW_RRF | 0.6612 | +0.0025 | +0.0000 | 0.5224 | 1.0000 | 0 | 0 | 4.66 | 9309 | 257 | 183 | 257 | 183 | 1.40 |
| MetaQA | R1_RRF_LIFT | 0.6717 | +0.0130 | +0.0105** | 0.0011 | 0.0000 | 21 | 0 | 4.72 | 9436 | 362 | 174 | 362 | 174 | 2.08 |
| MetaQA | R2_RRF_UNIT | 0.6697 | +0.0110 | +0.0085** | 0.0053 | 0.0001 | 18 | 1 | 5.07 | 10126 | 373 | 192 | 373 | 192 | 1.94 |
| MetaQA | R3_SIGNED_RET_RESIDUAL | 0.6692 | +0.0105 | +0.0080** | 0.0098 | 0.0037 | 22 | 6 | 5.62 | 11219 | 408 | 219 | 408 | 219 | 1.86 |
| MetaQA | R4_POSITIVE_RET_RESIDUAL | 0.6722 | +0.0135 | +0.0110** | 0.0009 | 0.0000 | 23 | 1 | 5.41 | 10809 | 438 | 193 | 438 | 193 | 2.27 |
| MetaQA | R5_PAIRWISE_RESIDUAL | 0.6707 | +0.0120 | +0.0095** | 0.0032 | 0.0006 | 24 | 5 | 5.55 | 11080 | 448 | 208 | 448 | 208 | 2.15 |
| MetaQA | R6_PARETO_SWAP | 0.6587 | +0.0000 | -0.0025 | 1.0000 | 0.5224 | 17 | 22 | 0.00 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| MetaQA | ORACLE_B6 | 0.7212 | +0.0626 | +0.0601 | — | — | — | — | — | — | — | — | — | — | — |
| WebQSP | BASE | 0.7618 | +0.0000 | -0.0028 | — | — | — | — | 0.00 | 0 | — | — | — | — | — |
| WebQSP | R0_RAW_RRF | 0.7646 | +0.0028 | +0.0000 | 0.6587 | 1.0000 | 0 | 0 | 4.97 | 7049 | 91 | 90 | 91 | 90 | 1.01 |
| WebQSP | R1_RRF_LIFT | 0.7646 | +0.0028 | +0.0000 | 0.5847 | 1.0000 | 8 | 8 | 3.16 | 4486 | 66 | 61 | 66 | 61 | 1.08 |
| WebQSP | R2_RRF_UNIT | 0.7604 | -0.0014 | -0.0042 | 0.8714 | 0.0703 | 1 | 7 | 5.04 | 7151 | 63 | 93 | 63 | 93 | 0.68 |
| WebQSP | R3_SIGNED_RET_RESIDUAL | 0.7555 | -0.0063 | -0.0092** | 0.1996 | 0.0010 | 1 | 14 | 5.57 | 7910 | 51 | 100 | 51 | 100 | 0.51 |
| WebQSP | R4_POSITIVE_RET_RESIDUAL | 0.7604 | -0.0014 | -0.0042 | 0.8714 | 0.0703 | 1 | 7 | 5.14 | 7299 | 59 | 95 | 59 | 95 | 0.62 |
| WebQSP | R5_PAIRWISE_RESIDUAL | 0.7604 | -0.0014 | -0.0042 | 0.8714 | 0.1094 | 2 | 8 | 5.19 | 7366 | 59 | 95 | 59 | 95 | 0.62 |
| WebQSP | R6_PARETO_SWAP | 0.7618 | +0.0000 | -0.0028 | 1.0000 | 0.6587 | 21 | 25 | 0.00 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| WebQSP | ORACLE_B6 | 0.7886 | +0.0268 | +0.0240 | — | — | — | — | — | — | — | — | — | — | — |
| 2Wiki | BASE | 0.9375 | +0.0000 | -0.0060 | — | — | — | — | 0.00 | 0 | — | — | — | — | — |
| 2Wiki | R0_RAW_RRF | 0.9435 | +0.0060 | +0.0000 | 0.0169 | 1.0000 | 0 | 0 | 4.26 | 8518 | 20 | 6 | 20 | 6 | 3.33 |
| 2Wiki | R1_RRF_LIFT | 0.9420 | +0.0045 | -0.0015 | 0.0636 | 0.5078 | 3 | 6 | 4.03 | 8054 | 16 | 7 | 16 | 7 | 2.29 |
| 2Wiki | R2_RRF_UNIT | 0.9415 | +0.0040 | -0.0020 | 0.1338 | 0.2891 | 2 | 6 | 4.69 | 9371 | 18 | 9 | 18 | 9 | 2.00 |
| 2Wiki | R3_SIGNED_RET_RESIDUAL | 0.9385 | +0.0010 | -0.0050* | 0.8238 | 0.0129 | 2 | 12 | 5.26 | 10516 | 13 | 12 | 13 | 12 | 1.08 |
| 2Wiki | R4_POSITIVE_RET_RESIDUAL | 0.9360 | -0.0015 | -0.0075** | 0.5810 | 0.0007 | 2 | 17 | 4.83 | 9671 | 6 | 11 | 6 | 11 | 0.55 |
| 2Wiki | R5_PAIRWISE_RESIDUAL | 0.9345 | -0.0030 | -0.0090** | 0.1460 | 0.0001 | 2 | 20 | 4.92 | 9836 | 4 | 12 | 4 | 12 | 0.33 |
| 2Wiki | R6_PARETO_SWAP | 0.9375 | +0.0000 | -0.0060* | 1.0000 | 0.0169 | 5 | 17 | 0.00 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| 2Wiki | ORACLE_B6 | 0.9515 | +0.0140 | +0.0080 | — | — | — | — | — | — | — | — | — | — | — |
| MuSiQue | BASE | 0.9565 | +0.0000 | -0.0070 | — | — | — | — | 0.00 | 0 | — | — | — | — | — |
| MuSiQue | R0_RAW_RRF | 0.9635 | +0.0070 | +0.0000 | 0.0288 | 1.0000 | 0 | 0 | 4.31 | 8624 | 26 | 11 | 26 | 11 | 2.36 |
| MuSiQue | R1_RRF_LIFT | 0.9605 | +0.0040 | -0.0030 | 0.2295 | 0.0703 | 1 | 7 | 3.50 | 7009 | 22 | 13 | 22 | 13 | 1.69 |
| MuSiQue | R2_RRF_UNIT | 0.9620 | +0.0055 | -0.0015 | 0.1081 | 0.4531 | 2 | 5 | 4.42 | 8834 | 26 | 14 | 26 | 14 | 1.86 |
| MuSiQue | R3_SIGNED_RET_RESIDUAL | 0.9605 | +0.0040 | -0.0030 | 0.2912 | 0.1460 | 3 | 9 | 4.93 | 9862 | 27 | 18 | 27 | 18 | 1.50 |
| MuSiQue | R4_POSITIVE_RET_RESIDUAL | 0.9565 | +0.0000 | -0.0070** | 1.0000 | 0.0013 | 2 | 16 | 4.43 | 8865 | 19 | 19 | 19 | 19 | 1.00 |
| MuSiQue | R5_PAIRWISE_RESIDUAL | 0.9555 | -0.0010 | -0.0080** | 0.8714 | 0.0004 | 2 | 18 | 4.51 | 9020 | 18 | 20 | 18 | 20 | 0.90 |
| MuSiQue | R6_PARETO_SWAP | 0.9565 | +0.0000 | -0.0070* | 1.0000 | 0.0288 | 11 | 25 | 0.00 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| MuSiQue | ORACLE_B6 | 0.9750 | +0.0185 | +0.0115 | — | — | — | — | — | — | — | — | — | — | — |
| HotpotQA | BASE | 0.9345 | +0.0000 | -0.0160 | — | — | — | — | 0.00 | 0 | — | — | — | — | — |
| HotpotQA | R0_RAW_RRF | 0.9505 | +0.0160 | +0.0000 | 0.0000 | 1.0000 | 0 | 0 | 5.10 | 10197 | 39 | 5 | 39 | 5 | 7.80 |
| HotpotQA | R1_RRF_LIFT | 0.9475 | +0.0130 | -0.0030 | 0.0000 | 0.2379 | 6 | 12 | 4.74 | 9481 | 30 | 3 | 30 | 3 | 10.00 |
| HotpotQA | R2_RRF_UNIT | 0.9475 | +0.0130 | -0.0030 | 0.0000 | 0.1094 | 2 | 8 | 5.33 | 10654 | 32 | 6 | 32 | 6 | 5.33 |
| HotpotQA | R3_SIGNED_RET_RESIDUAL | 0.9395 | +0.0050 | -0.0110** | 0.0872 | 0.0000 | 2 | 24 | 5.68 | 11357 | 19 | 11 | 19 | 11 | 1.73 |
| HotpotQA | R4_POSITIVE_RET_RESIDUAL | 0.9380 | +0.0035 | -0.0125** | 0.2295 | 0.0000 | 4 | 29 | 5.54 | 11089 | 16 | 11 | 16 | 11 | 1.46 |
| HotpotQA | R5_PAIRWISE_RESIDUAL | 0.9390 | +0.0045 | -0.0115** | 0.1361 | 0.0000 | 4 | 27 | 5.64 | 11287 | 19 | 12 | 19 | 12 | 1.58 |
| HotpotQA | R6_PARETO_SWAP | 0.9345 | +0.0000 | -0.0160** | 1.0000 | 0.0000 | 4 | 36 | 0.00 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| HotpotQA | ORACLE_B6 | 0.9645 | +0.0300 | +0.0140 | — | — | — | — | — | — | — | — | — | — | — |
| SQuAD | BASE | 0.9805 | +0.0000 | -0.0070 | — | — | — | — | 0.00 | 0 | — | — | — | — | — |
| SQuAD | R0_RAW_RRF | 0.9875 | +0.0070 | +0.0000 | 0.0026 | 1.0000 | 0 | 0 | 3.97 | 7948 | 17 | 3 | 17 | 3 | 5.67 |
| SQuAD | R1_RRF_LIFT | 0.9860 | +0.0055 | -0.0015 | 0.0127 | 0.2500 | 0 | 3 | 2.40 | 4809 | 14 | 3 | 14 | 3 | 4.67 |
| SQuAD | R2_RRF_UNIT | 0.9875 | +0.0070 | +0.0000 | 0.0026 | 1.0000 | 2 | 2 | 3.79 | 7589 | 17 | 3 | 17 | 3 | 5.67 |
| SQuAD | R3_SIGNED_RET_RESIDUAL | 0.9870 | +0.0065 | -0.0005 | 0.0106 | 1.0000 | 4 | 5 | 4.37 | 8730 | 18 | 5 | 18 | 5 | 3.60 |
| SQuAD | R4_POSITIVE_RET_RESIDUAL | 0.9840 | +0.0035 | -0.0035* | 0.1671 | 0.0391 | 1 | 8 | 3.46 | 6930 | 13 | 6 | 13 | 6 | 2.17 |
| SQuAD | R5_PAIRWISE_RESIDUAL | 0.9835 | +0.0030 | -0.0040* | 0.2863 | 0.0215 | 1 | 9 | 3.48 | 6954 | 14 | 8 | 14 | 8 | 1.75 |
| SQuAD | R6_PARETO_SWAP | 0.9805 | +0.0000 | -0.0070** | 1.0000 | 0.0026 | 3 | 17 | 0.00 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| SQuAD | ORACLE_B6 | 0.9945 | +0.0140 | +0.0070 | — | — | — | — | — | — | — | — | — | — | — |

## T8 — STEP 8. Δ vs SAFE_F6, compact

| rule | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| R0_RAW_RRF | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.00000 | +0.0000 |
| R1_RRF_LIFT | +0.0105** | +0.0000 | -0.0015 | -0.0030 | -0.0030 | -0.0015 | +0.00025 | -0.0030 |
| R2_RRF_UNIT | +0.0085** | -0.0042 | -0.0020 | -0.0015 | -0.0030 | +0.0000 | -0.00037 | -0.0042 |
| R3_SIGNED_RET_RESIDUAL | +0.0080** | -0.0092** | -0.0050* | -0.0030 | -0.0110** | -0.0005 | -0.00345 | -0.0110 |
| R4_POSITIVE_RET_RESIDUAL | +0.0110** | -0.0042 | -0.0075** | -0.0070** | -0.0125** | -0.0035* | -0.00395 | -0.0125 |
| R5_PAIRWISE_RESIDUAL | +0.0095** | -0.0042 | -0.0090** | -0.0080** | -0.0115** | -0.0040* | -0.00453 | -0.0115 |
| R6_PARETO_SWAP | -0.0025 | -0.0028 | -0.0060* | -0.0070* | -0.0160** | -0.0070** | -0.00688 | -0.0160 |

## T9 — STEP 9. MetaQA per-hop (mandatory), 666 queries per hop

| method | hop1 | Δ hop1 | hop2 | Δ hop2 | hop3 | Δ hop3 |
|---|---|---|---|---|---|---|
| BASE | 0.9955 | —  | 0.7237 | —  | 0.2568 | — |
| R0_RAW_RRF | 0.9955 | +0.0000 | 0.7297 | +0.0000 | 0.2583 | +0.0000 |
| R1_RRF_LIFT | 0.9955 | +0.0000 | 0.7477 | +0.0180** | 0.2718 | +0.0135** |
| R2_RRF_UNIT | 0.9955 | +0.0000 | 0.7462 | +0.0165** | 0.2673 | +0.0090 |
| R3_SIGNED_RET_RESIDUAL | 0.9955 | +0.0000 | 0.7372 | +0.0075 | 0.2748 | +0.0165** |
| R4_POSITIVE_RET_RESIDUAL | 0.9955 | +0.0000 | 0.7477 | +0.0180** | 0.2733 | +0.0150** |
| R5_PAIRWISE_RESIDUAL | 0.9955 | +0.0000 | 0.7402 | +0.0105 | 0.2763 | +0.0180** |
| R6_PARETO_SWAP | 0.9955 | +0.0000 | 0.7237 | -0.0060 | 0.2568 | -0.0015 |
| ORACLE_B6 | 0.9970 | —  | 0.8033 | —  | 0.3634 | — |

## T10 — STEP 10. Nested discovery → validation (sha1 parity folds, query-disjoint)

Criterion, fixed in advance: argmax macro mean dSAFE_F6 s.t. no corpus with negative delta at McNemar p<0.05; ties broken by worst-corpus delta; fallback = R0_RAW_RRF (= SAFE_F6)

| rule | DISCOVERY macro | DISCOVERY worst | DISCOVERY sig-regressions | selected |
|---|---|---|---|---|
| R0_RAW_RRF | +0.00000 | +0.00000 | none | **YES** |
| R1_RRF_LIFT | -0.00014 | -0.00601 | none |  |
| R2_RRF_UNIT | -0.00037 | -0.00406 | none |  |
| R3_SIGNED_RET_RESIDUAL | -0.00478 | -0.01201 | 2Wiki, HotpotQA |  |
| R4_POSITIVE_RET_RESIDUAL | -0.00509 | -0.01702 | 2Wiki, HotpotQA |  |
| R5_PAIRWISE_RESIDUAL | -0.00701 | -0.01624 | 2Wiki, MuSiQue, HotpotQA |  |
| R6_PARETO_SWAP | -0.00840 | -0.01802 | 2Wiki, MuSiQue, HotpotQA |  |

## T11 — STEP 10. Per-half stability of the two calibrated rules

| rule | corpus | DISCOVERY Δ | p | VALIDATION Δ | p | FULL Δ | p |
|---|---|---|---|---|---|---|---|
| R1_RRF_LIFT | MetaQA | +0.0118** | 0.0005 | +0.0092** | 0.0039 | +0.0105** | 0.0000 |
| R1_RRF_LIFT | WebQSP | +0.0015 | 1.0000 | -0.0013 | 1.0000 | +0.0000 | 1.0000 |
| R1_RRF_LIFT | 2Wiki | -0.0051 | 0.0625 | +0.0020 | 0.6250 | -0.0015 | 0.5078 |
| R1_RRF_LIFT | MuSiQue | -0.0020 | 0.5000 | -0.0039 | 0.2188 | -0.0030 | 0.0703 |
| R1_RRF_LIFT | HotpotQA | -0.0060 | 0.0703 | +0.0000 | 1.0000 | -0.0030 | 0.2379 |
| R1_RRF_LIFT | SQuAD | -0.0010 | 1.0000 | -0.0020 | 0.5000 | -0.0015 | 0.2500 |
| R1_RRF_LIFT | **macro** | **-0.00014** | | **+0.00065** | | **+0.00025** | |
| R2_RRF_UNIT | MetaQA | +0.0088* | 0.0117 | +0.0082** | 0.0078 | +0.0085** | 0.0001 |
| R2_RRF_UNIT | WebQSP | -0.0030 | 0.6250 | -0.0053 | 0.1250 | -0.0042 | 0.0703 |
| R2_RRF_UNIT | 2Wiki | -0.0041 | 0.1250 | +0.0000 | 1.0000 | -0.0020 | 0.2891 |
| R2_RRF_UNIT | MuSiQue | -0.0010 | 1.0000 | -0.0020 | 0.6875 | -0.0015 | 0.4531 |
| R2_RRF_UNIT | HotpotQA | -0.0040 | 0.2188 | -0.0020 | 0.6250 | -0.0030 | 0.1094 |
| R2_RRF_UNIT | SQuAD | +0.0010 | 1.0000 | -0.0010 | 1.0000 | +0.0000 | 1.0000 |
| R2_RRF_UNIT | **macro** | **-0.00037** | | **-0.00035** | | **-0.00037** | |

## T12 — STEP 10. Leave-one-dataset-out selection

| held-out corpus | rule selected on the other 5 | train macro | held-out Δ vs SAFE_F6 | McNemar p | significant regression |
|---|---|---|---|---|---|
| MetaQA | R0_RAW_RRF | +0.00000 | +0.0000 | 1.0000 | no |
| WebQSP | R2_RRF_UNIT | +0.00040 | -0.0042 | 0.0703 | no |
| 2Wiki | R1_RRF_LIFT | +0.00060 | -0.0015 | 0.5078 | no |
| MuSiQue | R1_RRF_LIFT | +0.00090 | -0.0030 | 0.0703 | no |
| HotpotQA | R1_RRF_LIFT | +0.00090 | -0.0030 | 0.2379 | no |
| SQuAD | R1_RRF_LIFT | +0.00060 | -0.0015 | 0.2500 | no |

`SELECTION_STABLE = False` — distinct selections across folds: R0_RAW_RRF, R1_RRF_LIFT, R2_RRF_UNIT. Held-out macro -0.00220.

`SURVIVES_VALIDATION = True` · `PROMOTABLE = False` · `WINNER = R0_RAW_RRF`


## T13 — STEP 11. Cost accounting

| corpus | ONLINE_GRAPH_EDGES_TOUCHED | new encoder passes | additional cached bytes/query | ranks read/query | candidates scored/query | extra float ops/query vs R0 |
|---|---|---|---|---|---|---|
| MetaQA | 0 | 0 | 0 | 272 | 49.2 | 816 |
| WebQSP | 0 | 0 | 0 | 250 | 34.4 | 752 |
| 2Wiki | 0 | 0 | 0 | 260 | 41.8 | 782 |
| MuSiQue | 0 | 0 | 0 | 181 | 18.6 | 545 |
| HotpotQA | 0 | 0 | 0 | 266 | 55.4 | 798 |
| SQuAD | 0 | 0 | 0 | 222 | 15.8 | 666 |

## T14 — STEP 11. Query routing latency by rule (µs/query, mean over the six corpora)

| rule | µs/query | vs R0 |
|---|---|---|
| R0_RAW_RRF | 36.7 | +0.0 |
| R1_RRF_LIFT | 36.7 | -0.0 |
| R2_RRF_UNIT | 36.8 | +0.1 |
| R3_SIGNED_RET_RESIDUAL | 34.7 | -2.0 |
| R4_POSITIVE_RET_RESIDUAL | 38.9 | +2.2 |
| R5_PAIRWISE_RESIDUAL | 40.6 | +3.9 |
| R6_PARETO_SWAP | 38.9 | +2.2 |
