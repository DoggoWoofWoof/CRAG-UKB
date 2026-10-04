# UNIVERSAL TRUE-HYPERGRAPH + HALO -- tables

### A1  raw hyperedge-size distribution, before any cap

Hyperedge = closed neighbourhood {u} u N(u), so raw size = deg(u)+1.  No truncation.

| corpus | family | hyperedges | pins | mean | p75 | p90 | p95 | p99 | max | pins kept at the fixed cap 25 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | STRUCT | 40,151 | 259,111 | 6.45 | 7 | 11 | 14 | 28 | 4,181 | 0.8028 |
| MetaQA | KNN | 38,141 | 148,555 | 3.89 | 4 | 6 | 7 | 10 | 42 | 0.9984 |
| MetaQA | NER | 29,648 | 494,658 | 16.68 | 24 | 38 | 48 | 67 | 108 | 0.4581 |
| WebQSP | STRUCT | 781,485 | 4,057,617 | 5.19 | 4 | 6 | 11 | 36 | 41,606 | 0.6398 |
| WebQSP | KNN | 773,107 | 4,118,569 | 5.33 | 6 | 8 | 10 | 14 | 181 | 0.9948 |
| WebQSP | NER | 186,382 | 2,138,372 | 11.47 | 17 | 23 | 28 | 44 | 130 | 0.8005 |
| 2Wiki | STRUCT | 54,467 | 306,607 | 5.63 | 5 | 7 | 10 | 21 | 38,593 | 0.7196 |
| 2Wiki | KNN | 65,508 | 334,982 | 5.11 | 6 | 8 | 9 | 12 | 31 | 0.9997 |
| 2Wiki | NER | 51,808 | 1,040,416 | 20.08 | 27 | 44 | 58 | 92 | 337 | 0.3944 |
| MuSiQue | STRUCT | 9,880 | 112,966 | 11.43 | 11 | 23 | 40 | 121 | 1,487 | 0.4715 |
| MuSiQue | KNN | 13,490 | 68,302 | 5.06 | 6 | 7 | 8 | 11 | 27 | 0.9996 |
| MuSiQue | NER | 11,634 | 198,634 | 17.07 | 24 | 35 | 44 | 67 | 249 | 0.5438 |
| HotpotQA | STRUCT | 503,974 | 7,357,864 | 14.60 | 11 | 16 | 21 | 62 | 46,892 | 0.5651 |
| HotpotQA | KNN | 503,932 | 2,837,236 | 5.63 | 6 | 8 | 9 | 13 | 260,450 | 0.9071 |
| HotpotQA | NER | 421,290 | 8,538,704 | 20.27 | 28 | 42 | 53 | 78 | 589 | 0.4183 |
| SQuAD | STRUCT | 12,947 | 1,402,107 | 108.30 | 128 | 218 | 300 | 514 | 2,335 | 0.0087 |
| SQuAD | KNN | 15,677 | 72,229 | 4.61 | 5 | 7 | 8 | 10 | 23 | 1.0000 |
| SQuAD | NER | 14,159 | 267,135 | 18.87 | 26 | 41 | 51 | 75 | 172 | 0.4224 |

### A0/A2-A5  pin retention by rule (STRUCT + KNN, the family set the previous build used)

| corpus | H0_FIXED_CAP | H1_FULL_WEIGHTED | H2_Q90 | H2_Q95 | H2_Q99 | H3_B05 | H3_B10 | H3_B20 | H4_SPLIT_PRESERVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 0.8741 | 1.0000 | 0.8314 | 0.8570 | 0.8989 | 0.9215 | 0.9365 | 0.9526 | 1.0000 |
| WebQSP | 0.8186 | 1.0000 | 0.7802 | 0.8052 | 0.8526 | 0.8830 | 0.9052 | 0.9245 | 1.0000 |
| 2Wiki | 0.8658 | 1.0000 | 0.8123 | 0.8422 | 0.8709 | 0.8880 | 0.8944 | 0.9012 | 1.0000 |
| MuSiQue | 0.6705 | 1.0000 | 0.6949 | 0.7647 | 0.9192 | 0.8709 | 0.9431 | 0.9809 | 1.0000 |
| HotpotQA | 0.6602 | 1.0000 | 0.6547 | 0.6749 | 0.7112 | 0.7237 | 0.7450 | 0.7671 | 1.0000 |
| SQuAD | 0.0573 | 1.0000 | 0.7658 | 0.8537 | 0.9392 | 0.4411 | 0.6688 | 0.8487 | 1.0000 |

Numeric cap the identical rule produced on each corpus:

| corpus | H0_FIXED_CAP | H1_FULL_WEIGHTED | H2_Q90 | H2_Q95 | H2_Q99 | H3_B05 | H3_B10 | H3_B20 | H4_SPLIT_PRESERVE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 25 | None | 9 | 11 | 21 | 50 | 100 | 200 | 100 |
| WebQSP | 25 | None | 8 | 10 | 22 | 50 | 100 | 200 | 100 |
| 2Wiki | 25 | None | 7 | 9 | 16 | 50 | 100 | 200 | 100 |
| MuSiQue | 25 | None | 12 | 20 | 80 | 50 | 101 | 202 | 101 |
| HotpotQA | 25 | None | 13 | 16 | 34 | 50 | 100 | 200 | 100 |
| SQuAD | 25 | None | 139 | 205 | 376 | 50 | 100 | 200 | 100 |

### A8  BASE -- Dense+SPLADE partition ranking at P=50, no selector

Delta against the production METIS core.  `*` = exact McNemar p<0.05 AND the delta exceeds that corpus's measured reseed noise floor.

| representation | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | worst | macro | sig reg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H2_Q99 | +0.1236\* | - | -0.0040 | -0.0125\* | - | -0.0005 | -0.0125 | +0.0267 | 1 |
| H3_B05 | +0.1236\* | - | -0.0045 | -0.0090 | - | +0.0070\* | -0.0090 | +0.0293 | 0 |
| H3_B10 | +0.1221\* | - | +0.0000 | -0.0065 | - | +0.0070\* | -0.0065 | +0.0307 | 0 |
| H4_SPLIT_PRESERVE | +0.1276\* | +0.0416\* | -0.0025 | -0.0085 | +0.0160\* | +0.0015 | -0.0085 | +0.0293 | 0 |
| H2_Q90 | +0.1191\* | - | -0.0040 | -0.0065 | - | +0.0025 | -0.0065 | +0.0278 | 0 |
| H1_FULL_WEIGHTED | +0.1221\* | - | +0.0010 | -0.0050 | - | +0.0010 | -0.0050 | +0.0298 | 0 |
| H2_Q95 | +0.1301\* | - | -0.0035 | -0.0010 | - | +0.0075\* | -0.0035 | +0.0333 | 0 |
| H3_B20 | +0.1246\* | - | -0.0045 | +0.0015 | - | +0.0010 | -0.0045 | +0.0307 | 0 |
| H0_FIXED_CAP | +0.1326\* | +0.0564\* | +0.0020 | -0.0015 | - | -0.0440\* | -0.0440 | +0.0291 | 1 |

### A9  SAFE -- frozen B6_S4_F6_Ms64_Mr32, mechanically reaggregated

Delta against the production METIS core.  `*` = exact McNemar p<0.05 AND the delta exceeds that corpus's measured reseed noise floor.

| representation | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | worst | macro | sig reg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H2_Q99 | +0.1236\* | - | -0.0030 | -0.0025 | - | +0.0030 | -0.0030 | +0.0303 | 0 |
| H3_B05 | +0.1221\* | - | -0.0040 | +0.0000 | - | +0.0030 | -0.0040 | +0.0303 | 0 |
| H3_B10 | +0.1186\* | - | -0.0040 | -0.0030 | - | +0.0030 | -0.0040 | +0.0286 | 0 |
| H4_SPLIT_PRESERVE | +0.1276\* | +0.0430\* | +0.0005 | -0.0045 | +0.0115\* | +0.0040 | -0.0045 | +0.0303 | 0 |
| H2_Q90 | +0.1171\* | - | -0.0045 | -0.0035 | - | +0.0030 | -0.0045 | +0.0280 | 0 |
| H1_FULL_WEIGHTED | +0.1211\* | - | +0.0000 | -0.0050 | - | +0.0010 | -0.0050 | +0.0293 | 0 |
| H2_Q95 | +0.1271\* | - | -0.0060 | -0.0020 | - | +0.0030 | -0.0060 | +0.0305 | 0 |
| H3_B20 | +0.1226\* | - | -0.0075 | +0.0010 | - | +0.0030 | -0.0075 | +0.0298 | 0 |
| H0_FIXED_CAP | +0.1286\* | +0.0599\* | -0.0020 | -0.0010 | - | -0.0130\* | -0.0130 | +0.0345 | 1 |


### A7  partition balance envelope

| corpus | representation | blocks | min | median | mean | p90 | p99 | max | max/mean | CV | eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MetaQA | CURRENT | 401 | 96 | 100 | 100.1 | 103 | 103 | 103 | 1.029 | 0.022 | yes |
| MetaQA | H0_FIXED_CAP | 401 | 1 | 102 | 100.1 | 103 | 104 | 104 | 1.039 | 0.129 | yes |
| MetaQA | H1_FULL_WEIGHTED | 401 | 2 | 102 | 100.1 | 104 | 104 | 104 | 1.039 | 0.116 | yes |
| MetaQA | H2_Q90 | 401 | 3 | 102 | 100.1 | 104 | 104 | 104 | 1.039 | 0.104 | yes |
| MetaQA | H2_Q95 | 401 | 2 | 102 | 100.1 | 104 | 104 | 104 | 1.039 | 0.113 | yes |
| MetaQA | H2_Q99 | 401 | 2 | 102 | 100.1 | 104 | 104 | 104 | 1.039 | 0.096 | yes |
| MetaQA | H3_B05 | 401 | 1 | 102 | 100.1 | 104 | 104 | 104 | 1.039 | 0.106 | yes |
| MetaQA | H3_B10 | 401 | 4 | 102 | 100.1 | 104 | 104 | 104 | 1.039 | 0.114 | yes |
| MetaQA | H3_B20 | 401 | 2 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.113 | yes |
| MetaQA | H4_SPLIT_PRESERVE | 401 | 3 | 102 | 100.1 | 104 | 104 | 104 | 1.039 | 0.103 | yes |
| WebQSP | CURRENT | 7814 | 86 | 100 | 100.0 | 103 | 103 | 103 | 1.030 | 0.020 | yes |
| WebQSP | H0_FIXED_CAP | 7814 | 1 | 101 | 100.0 | 104 | 104 | 104 | 1.040 | 0.061 | yes |
| WebQSP | H4_SPLIT_PRESERVE | 7814 | 1 | 103 | 100.0 | 104 | 104 | 104 | 1.040 | 0.115 | yes |
| 2Wiki | CURRENT | 658 | 97 | 100 | 100.1 | 103 | 103 | 103 | 1.029 | 0.020 | yes |
| 2Wiki | H0_FIXED_CAP | 658 | 2 | 101 | 100.1 | 104 | 104 | 104 | 1.039 | 0.082 | yes |
| 2Wiki | H1_FULL_WEIGHTED | 658 | 2 | 104 | 100.1 | 104 | 104 | 104 | 1.039 | 0.125 | yes |
| 2Wiki | H2_Q90 | 658 | 1 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.112 | yes |
| 2Wiki | H2_Q95 | 658 | 3 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.141 | yes |
| 2Wiki | H2_Q99 | 658 | 4 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.117 | yes |
| 2Wiki | H3_B05 | 658 | 1 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.127 | yes |
| 2Wiki | H3_B10 | 658 | 3 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.125 | yes |
| 2Wiki | H3_B20 | 658 | 3 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.111 | yes |
| 2Wiki | H4_SPLIT_PRESERVE | 658 | 2 | 103 | 100.1 | 104 | 104 | 104 | 1.039 | 0.122 | yes |
| MuSiQue | CURRENT | 136 | 96 | 101 | 100.5 | 103 | 103 | 103 | 1.025 | 0.024 | yes |
| MuSiQue | H0_FIXED_CAP | 136 | 85 | 102 | 100.5 | 104 | 104 | 104 | 1.034 | 0.036 | yes |
| MuSiQue | H1_FULL_WEIGHTED | 136 | 33 | 103 | 100.5 | 104 | 104 | 104 | 1.034 | 0.080 | yes |
| MuSiQue | H2_Q90 | 136 | 56 | 103 | 100.5 | 104 | 104 | 104 | 1.034 | 0.070 | yes |
| MuSiQue | H2_Q95 | 136 | 5 | 104 | 100.5 | 104 | 104 | 104 | 1.034 | 0.123 | yes |
| MuSiQue | H2_Q99 | 136 | 34 | 104 | 100.5 | 104 | 104 | 104 | 1.034 | 0.097 | yes |
| MuSiQue | H3_B05 | 136 | 4 | 103 | 100.5 | 104 | 104 | 104 | 1.034 | 0.095 | yes |
| MuSiQue | H3_B10 | 136 | 28 | 104 | 100.5 | 104 | 104 | 104 | 1.034 | 0.085 | yes |
| MuSiQue | H3_B20 | 136 | 36 | 103 | 100.5 | 104 | 104 | 104 | 1.034 | 0.086 | yes |
| MuSiQue | H4_SPLIT_PRESERVE | 136 | 22 | 104 | 100.5 | 104 | 104 | 104 | 1.034 | 0.097 | yes |
| HotpotQA | CURRENT | 5074 | 71 | 100 | 100.0 | 103 | 103 | 103 | 1.030 | 0.023 | yes |
| HotpotQA | H4_SPLIT_PRESERVE | 5074 | 1 | 104 | 100.0 | 104 | 104 | 104 | 1.040 | 0.149 | yes |
| SQuAD | CURRENT | 190 | 82 | 102 | 100.2 | 103 | 103 | 103 | 1.028 | 0.031 | yes |
| SQuAD | H0_FIXED_CAP | 190 | 86 | 100 | 100.2 | 102 | 104 | 104 | 1.038 | 0.022 | yes |
| SQuAD | H1_FULL_WEIGHTED | 190 | 30 | 102 | 100.2 | 104 | 104 | 104 | 1.038 | 0.078 | yes |
| SQuAD | H2_Q90 | 190 | 70 | 101 | 100.2 | 104 | 104 | 104 | 1.038 | 0.042 | yes |
| SQuAD | H2_Q95 | 190 | 36 | 102 | 100.2 | 104 | 104 | 104 | 1.038 | 0.070 | yes |
| SQuAD | H2_Q99 | 190 | 53 | 101 | 100.2 | 103 | 104 | 104 | 1.038 | 0.058 | yes |
| SQuAD | H3_B05 | 190 | 73 | 101 | 100.2 | 104 | 104 | 104 | 1.038 | 0.047 | yes |
| SQuAD | H3_B10 | 190 | 67 | 101 | 100.2 | 104 | 104 | 104 | 1.038 | 0.047 | yes |
| SQuAD | H3_B20 | 190 | 64 | 101 | 100.2 | 104 | 104 | 104 | 1.038 | 0.047 | yes |
| SQuAD | H4_SPLIT_PRESERVE | 190 | 56 | 103 | 100.2 | 104 | 104 | 104 | 1.038 | 0.071 | yes |

### B6  additivity of core and halo (F6 lane, ALL_REQUIRED_FETCHED)

A = METIS no halo, B = hypergraph no halo, C = METIS + beta 0.5, D = hypergraph + beta 0.5.
INTERACTION = D - B - C + A: negative means the two mechanisms rescue the same queries.

| corpus | A | B | C | D | CORE | HALO | JOINT | INTERACTION |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 0.6612 | 0.7888 | 0.7432 | 0.8078 | +0.1276 | +0.0820 | +0.1466 | -0.0630 |
| WebQSP | 0.7646 | 0.8076 | 0.8118 | 0.8393 | +0.0430 | +0.0472 | +0.0747 | -0.0155 |
| 2Wiki | 0.9435 | 0.9440 | 0.9540 | 0.9510 | +0.0005 | +0.0105 | +0.0075 | -0.0035 |
| MuSiQue | 0.9635 | 0.9590 | 0.9775 | 0.9730 | -0.0045 | +0.0140 | +0.0095 | +0.0000 |
| HotpotQA | 0.9505 | 0.9620 | 0.9575 | 0.9635 | +0.0115 | +0.0070 | +0.0130 | -0.0055 |
| SQuAD | 0.9875 | 0.9915 | 0.9910 | 0.9920 | +0.0040 | +0.0035 | +0.0045 | -0.0030 |

### B5  required-node coverage by prefix depth (O0_CORE vs O4_FULL_C beta=0.5)

ALL_REQUIRED_FETCHED at each partition-list prefix P.  Depth is defined on the BASE block ranking (the selector is set-valued at exactly P=50, so it has no shorter prefix of its own).

| corpus | core | cell | P1 | P5 | P10 | P25 | P50 |
|---|---|---|---:|---:|---:|---:|---:|
| MetaQA | METIS | O0_CORE | 0.1226 | 0.3539 | 0.4780 | 0.5966 | 0.6587 |
| MetaQA | METIS | O4_FULL_C_b0.5 | 0.1912 | 0.4499 | 0.5701 | 0.6732 | 0.7392 |
| WebQSP | METIS | O0_CORE | 0.1113 | 0.3714 | 0.5053 | 0.6603 | 0.7618 |
| WebQSP | METIS | O4_FULL_C_b0.5 | 0.1931 | 0.4778 | 0.6047 | 0.7230 | 0.8076 |
| WebQSP | hypergraph | O0_CORE | 0.1219 | 0.4087 | 0.5518 | 0.7033 | 0.8034 |
| WebQSP | hypergraph | O4_FULL_C_b0.5 | 0.1910 | 0.5011 | 0.6180 | 0.7435 | 0.8316 |
| 2Wiki | METIS | O0_CORE | 0.0170 | 0.3650 | 0.6910 | 0.9005 | 0.9375 |
| 2Wiki | METIS | O4_FULL_C_b0.5 | 0.0215 | 0.4060 | 0.7165 | 0.9130 | 0.9475 |
| 2Wiki | hypergraph | O0_CORE | 0.0095 | 0.6210 | 0.8280 | 0.9080 | 0.9350 |
| 2Wiki | hypergraph | O4_FULL_C_b0.5 | 0.0110 | 0.6415 | 0.8380 | 0.9165 | 0.9425 |
| MuSiQue | METIS | O0_CORE | 0.2210 | 0.5695 | 0.7125 | 0.8855 | 0.9565 |
| MuSiQue | METIS | O4_FULL_C_b0.5 | 0.2710 | 0.6285 | 0.7645 | 0.9180 | 0.9730 |
| MuSiQue | hypergraph | O0_CORE | 0.1780 | 0.5320 | 0.7040 | 0.8780 | 0.9480 |
| MuSiQue | hypergraph | O4_FULL_C_b0.5 | 0.2480 | 0.6010 | 0.7600 | 0.9060 | 0.9660 |
| HotpotQA | METIS | O0_CORE | 0.0625 | 0.3410 | 0.5775 | 0.8525 | 0.9345 |
| HotpotQA | METIS | O4_FULL_C_b0.5 | 0.0695 | 0.3830 | 0.6155 | 0.8680 | 0.9410 |
| HotpotQA | hypergraph | O0_CORE | 0.0335 | 0.5225 | 0.7740 | 0.9175 | 0.9505 |
| HotpotQA | hypergraph | O4_FULL_C_b0.5 | 0.0440 | 0.5530 | 0.7900 | 0.9230 | 0.9535 |
| SQuAD | METIS | O0_CORE | 0.2675 | 0.6355 | 0.8045 | 0.9435 | 0.9805 |
| SQuAD | METIS | O4_FULL_C_b0.5 | 0.3165 | 0.6945 | 0.8450 | 0.9555 | 0.9850 |
| SQuAD | hypergraph | O0_CORE | 0.3575 | 0.7040 | 0.8635 | 0.9525 | 0.9820 |
| SQuAD | hypergraph | O4_FULL_C_b0.5 | 0.3945 | 0.7465 | 0.8840 | 0.9580 | 0.9825 |

### B7/B8  halo exposure cost at beta=0.5 (F6 lane)

Same beta budget applied to both cores -> exposure multipliers land close by construction (not interpolated to an exact match).  Efficiency = newly-required nodes rescued per 1000 extra unique nodes the halo exposes -- the marginal return on the halo's exposure cost.

| corpus | core | expo x | new required | queries rescued | required/1k extra exposed |
|---|---|---:|---:|---:|---:|
| MetaQA | METIS | x1.3582 | 1637 | 164 | 0.4542 |
| MetaQA | hypergraph | x1.3368 | 548 | 38 | 0.1602 |
| WebQSP | METIS | x1.3940 | 514 | 67 | 0.1831 |
| WebQSP | hypergraph | x1.3869 | 279 | 45 | 0.1001 |
| 2Wiki | METIS | x1.3488 | 21 | 21 | 0.0060 |
| 2Wiki | hypergraph | x1.3219 | 15 | 14 | 0.0046 |
| MuSiQue | METIS | x1.2037 | 29 | 28 | 0.0141 |
| MuSiQue | hypergraph | x1.2002 | 32 | 28 | 0.0157 |
| HotpotQA | METIS | x1.2893 | 14 | 14 | 0.0048 |
| HotpotQA | hypergraph | x1.2569 | 3 | 3 | 0.0011 |
| SQuAD | METIS | x1.2294 | 7 | 7 | 0.0030 |
| SQuAD | hypergraph | x1.2057 | 1 | 1 | 0.0005 |

### B9  joint system (hypergraph core + beta=0.5 halo) vs production METIS, paired McNemar

| corpus | delta | gained | lost | p | significant |
|---|---:|---:|---:|---:|---|
| MetaQA | +0.1466 | 335 | 42 | 7.92e-58 | YES |
| WebQSP | +0.0747 | 123 | 17 | 5.17e-21 | YES |
| 2Wiki | +0.0075 | 58 | 43 | 0.163 | no |
| MuSiQue | +0.0095 | 53 | 34 | 0.053 | no |
| HotpotQA | +0.0130 | 41 | 15 | 0.000686 | YES |
| SQuAD | +0.0045 | 15 | 6 | 0.0784 | no |

## STAGE 5 (PART 0B) -- attribution + optimization, full numbers

Prose summary and promotion status in `FINAL_REPORT.md`'s STAGE 5. Source JSON:
`global_halo/F_REPORT.json` (F), `attribution/A_LADDER.json` + `B_QUERY_CLASSES.json` (A/B),
`halo_family/E_REPORT.json` (E), `support_multiplicity/G_REPORT.json` (G),
`residual/H_REPORT.json` (H/I).

### F  global dedup-aware halo, matched exposure (K_q identical to F0 per query)

| corpus | F0 | F1 | F2 | F0 mean exposure | F1 vs F0 p | F2 vs F0 p |
|---|---:|---:|---:|---:|---:|---:|
| MetaQA | 0.8078 | 0.8083 | 0.8148 | 1712.5 | 1.0 | 0.0488 |
| 2Wiki | 0.9510 | 0.9530 | 0.9475 | 1636.3 | 0.125 | 0.0923 |
| MuSiQue | 0.9730 | 0.9725 | 0.9735 | 1019.3 | 1.0 | 1.0 |
| SQuAD | 0.9920 | 0.9920 | 0.9920 | 1047.9 | 1.0 | 1.0 |
| WebQSP | 0.8393 | 0.8330 | 0.8548 | 1965.2 | 0.00391 | 5.95e-05 |
| HotpotQA | 0.9635 | 0.9635 | 0.9665 | 1320.0 | 1.0 | 0.0312 |

### E  halo family leave-one-out + ranking-rule control (F6 lane, delta vs E0_FULL_C)

| corpus | -STRUCT (E1) | -NERX (E2) | -KNN (E3) | random rank (E4) |
|---|---:|---:|---:|---:|
| MetaQA | +0.0010 ns | -0.0055 ns | +0.0035 ns | **-0.0090 sig** |
| 2Wiki | **+0.0035 sig** | -0.0025 ns | -0.0005 ns | -0.0015 ns |
| MuSiQue | **+0.0045 sig** | -0.0045 ns | -0.0015 ns | -0.0025 ns |
| SQuAD | 0.0 ns | +0.0005 ns | 0.0 ns | +0.0005 ns |
| WebQSP | **-0.0225 sig** | +0.0014 ns | **+0.0303 sig** | +0.0014 ns |
| HotpotQA | **+0.0045 sig** | +0.0020 ns | -0.0005 ns | +0.0010 ns |

### G  support-multiplicity by rescue group (mean support count, n in parens)

| corpus | pool baseline | BOTH | F2_ONLY | F1_ONLY | NEITHER |
|---|---:|---:|---:|---:|---:|
| MetaQA | 3.156 | 7.879 (298) | 10.134 (619) | 3.425 (252) | 4.210 (3213) |
| 2Wiki | 1.931 | 4.250 (4) | 6.250 (4) | 4.667 (15) | 3.101 (89) |
| MuSiQue | 3.876 | 11.304 (23) | 10.300 (10) | 5.667 (9) | 4.682 (44) |
| SQuAD | 4.919 | 35.000 (1) | -- (0) | -- (0) | 3.857 (14) |
| WebQSP | 1.061 | 7.733 (165) | 5.425 (134) | 3.784 (51) | 2.638 (1080) |
| HotpotQA | 1.278 | 7.333 (3) | 7.143 (7) | -- (0) | 3.314 (70) |

### H  residual node classification under F2 (best config), counts

| corpus | total missing | H0 | H1 (budget) | H2 (low-rank) | H3 (2-hop) | H4 (far) | H5 (isolated) |
|---|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 4440 | 0 | 569 | 2896 | 975 | 0 | 0 |
| 2Wiki | 108 | 0 | 4 | 100 | 3 | 1 | 0 |
| MuSiQue | 58 | 0 | 23 | 30 | 5 | 0 | 0 |
| SQuAD | 16 | 0 | 1 | 13 | 2 | 0 | 0 |
| WebQSP | 1204 | 0 | 150 | 981 | 67 | 6 | 0 |
| HotpotQA | 71 | 0 | 5 | 65 | 1 | 0 | 0 |

### I  2-hop oracle ceiling (diagnostic -- every H3 node hypothetically rescued, unbounded)

| corpus | F2 | 2-hop oracle | upside |
|---|---:|---:|---:|
| MetaQA | 0.8148 | 0.8183 | +0.0035 |
| 2Wiki | 0.9475 | 0.9485 | +0.0010 |
| MuSiQue | 0.9735 | 0.9755 | +0.0020 |
| SQuAD | 0.9920 | 0.9930 | +0.0010 |
| WebQSP | 0.8548 | 0.8584 | +0.0036 |
| HotpotQA | 0.9665 | 0.9670 | +0.0005 |

## Stage 6 (Part 0C) -- local residual 1-hop ranking phase (R0-R6)

Reference = F2_SUM_NORMALIZED (Stage 5). All four candidate rankers computed in one pass,
gated with the same exact two-sided McNemar used throughout. `scratchpad/_l1hu_r.py`;
`ranking_refine/R_<ds>.json` + `R_GATE.json`.

### R0 parity (this script's F2 vs the shipped global_halo F2)

EXACT on all 6 corpora -- 0 diffs, ref_rate == mine_rate to 4dp everywhere.

### R5 gate -- all four candidates vs F2 reference

| method | worst delta | macro delta | sig regressions | sig gains | clears R5 |
|---|---:|---:|---:|---:|---:|
| F3_SUPPORT_FIRST | -0.0010 | +0.0091 | 0 | 1 (webqsp) | yes |
| F4_QUERY_RRF | -0.0005 | +0.0064 | 0 | 4 (musique,squad,webqsp,hotpot) | yes |
| F5_FAMILY_SUPPORT | -0.0028 | +0.0014 | 0 | 0 | yes (weak) |
| **F6_SUPPORT_QUERY_RRF** | **+0.0010** | **+0.0143** | **0** | **4 (musique,squad,webqsp,hotpot)** | **yes (best)** |

F6 is the only method with a non-negative delta on every corpus. R4's own gate (F3 AND F4 must
each independently clear R5 before F6 is authorized) is satisfied comfortably -- in fact all four
candidates clear R5 independently.

### F6_SUPPORT_QUERY_RRF vs F2, per corpus

| corpus | F2 | F6 | delta | gained | lost | p | sig |
|---|---:|---:|---:|---:|---:|---:|---:|
| MetaQA | 0.8148 | 0.8158 | +0.0010 | 17 | 15 | 0.860 | no |
| 2Wiki | 0.9475 | 0.9500 | +0.0025 | 11 | 6 | 0.332 | no |
| MuSiQue | 0.9735 | 0.9805 | +0.0070 | 19 | 5 | 0.00661 | **yes** |
| SQuAD | 0.9920 | 0.9965 | +0.0045 | 9 | 0 | 0.00391 | **yes** |
| WebQSP | 0.8548 | 0.9133 | +0.0585 | 87 | 4 | 2.26e-21 | **yes** |
| HotpotQA | 0.9665 | 0.9785 | +0.0120 | 26 | 2 | 3.03e-06 | **yes** |

### R1 support-count enrichment (req_per_10k by support-count quartile, ascending)

| corpus | q1 | q2 | q3 | q4 | monotone? |
|---|---:|---:|---:|---:|---|
| MetaQA | 0.386 | 0.636 | 0.931 | 1.775 | yes |
| 2Wiki | 0.0028 | 0.0206 | 0.0175 | 0.0412 | mostly (dip q3) |
| MuSiQue | 0.0201 | 0.0204 | 0.0472 | 0.1315 | yes |
| SQuAD | 0.0068 | 0.0112 | 0.0034 | 0.0060 | no (n=1-7, noise) |
| WebQSP | 0.0108 | 0.3174 | 1.6031 | 7.7889 | yes, 720x range |
| HotpotQA | 0.0002 | 0.0010 | 0.0060 | 0.0260 | yes, 130x range |

### R3 family-count enrichment (req_per_10k by family-count tercile, ascending)

| corpus | t1 | t2 | t3 | monotone? |
|---|---:|---:|---:|---|
| MetaQA | 0.688 | 1.125 | 2.021 | yes |
| 2Wiki | 0.0032 | 0.0245 | 0.0507 | yes |
| MuSiQue | 0.0227 | 0.0764 | 0.2653 | yes |
| SQuAD | 0.0054 | 0.0081 | 0.0098 | yes (n=1-7, weak) |
| WebQSP | 0.0304 | 0.8222 | 2.5328 | yes, 83x range |
| HotpotQA | 0.0004 | 0.0040 | 0.0299 | yes, 75x range |

squad_clean is inconclusive on both R1 and R3 (single-digit `req` counts per bucket) -- near-ceiling
F2 performance leaves almost no required nodes in the residual candidate pool to enrich against.
Both diagnostics pass cleanly on the other 5/6 corpora, which is what authorized building F3/F5.

### R6 -- H2-targeted rank movement and query rescue (bucket H2 = low-ranked, not budget-limited)

`frac_moved_into_Kq` = fraction of H2 candidate nodes moved into the retrieved set under each
ranker. `query_rescue_conversion` = fraction of H2-affected queries fully resolved.

| corpus | n H2 nodes | n H2 queries | F3 rescue | F4 rescue | F5 rescue | F6 rescue |
|---|---:|---:|---:|---:|---:|---:|
| MetaQA | 2896 | 329 | 0.0152 | 0.0061 | **0.0274** | 0.0182 |
| 2Wiki | 100 | 98 | 0.0306 | 0.0714 | 0.0918 | **0.1020** |
| MuSiQue | 30 | 27 | 0.0370 | 0.2963 | **0.4074** | 0.2963 |
| SQuAD | 13 | 13 | 0.0 | 0.6154 | 0.0769 | **0.6154** |
| WebQSP | 981 | 175 | 0.3371 | 0.0857 | 0.1086 | **0.3886** |
| HotpotQA | 65 | 62 | 0.0645 | 0.3548 | 0.0968 | **0.3710** |
| **macro** | | | 0.0807 | 0.2383 | 0.1348 | **0.2986** |

F6 has the best macro H2-rescue rate (0.2986) and wins outright on 4/6 corpora; F5 edges it on
MetaQA and MuSiQue's H2 subpopulation specifically (both tiny-n: 27-329 queries), even though F5
is the weakest method in aggregate -- attributed to noise at that n, not a real mechanism, since
F5 never posts a significant aggregate gain anywhere.

### Cumulative: Production METIS -> H4+F2 (Stage 5) -> H4+F6 (Stage 6)

| corpus | Production | H4+F2 | H4+F6 | net vs Production |
|---|---:|---:|---:|---:|
| MetaQA | 0.6612 | 0.8148 | 0.8158 | +0.1546 |
| WebQSP | 0.7646 | 0.8548 | 0.9133 | +0.1487 |
| HotpotQA | 0.9505 | 0.9665 | 0.9785 | +0.0280 |
| 2Wiki | 0.9435 | 0.9475 | 0.9500 | +0.0065 |
| MuSiQue | 0.9635 | 0.9735 | 0.9805 | +0.0170 |
| SQuAD | 0.9875 | 0.9920 | 0.9965 | +0.0090 |

`BEST_ONEHOP_RANKER = F6_SUPPORT_QUERY_RRF`. `PART_0C_PROMOTED = NONE` -- same frozen-L2 hold as
Stage 5 (see FINAL_REPORT.md PROMOTION section). Per the spec's own stop conditions, this phase
concludes here.

## Stage 7 (Part 0D): L1 HIGH-CEILING RESIDUAL RANKING -- oracle-gated structural signals on F6

Targeted follow-on: can query-conditioned *structure* push MetaQA past 0.90 and WebQSP past 0.95,
or is F6 already near the wall? Part A settles this before any new ranker is built.

### Part A: oracle ceilings (run first, binding STOP RULE)

Same H4 cores, same selected-50, same FULL_C candidate universe as F6. O1 = same K_q, required
nodes ranked first (ranking-alone ceiling). O2 = no K_q cutoff, every candidate counts as fetched
(reachability ceiling). O3 = core + every candidate, budget ignored -- verified identical to O2 in
this codebase (core members are already unconditionally fetched, and the candidate pool literally
*is* "every 1-hop candidate"), not merely assumed.

| corpus | O1_FIXED_K | O2_FULL_1HOP | O2==O3 | mean K_q | budget-limited queries |
|---|---:|---:|:---:|---:|---:|
| MetaQA | **0.9279** | 0.9279 | True | 360.1 | 0 |
| WebQSP | **0.9859** | 0.9859 | True | 380.8 | 0 |
| 2Wiki | 0.9980 | 0.9980 | True | 92.4 | 0 |
| MuSiQue | 0.9975 | 0.9975 | True | 42.1 | 0 |
| SQuAD | 0.9990 | 0.9990 | True | 9.2 | 0 |
| HotpotQA | 0.9995 | 0.9995 | True | 51.8 | 0 |

MetaQA by hop (n=666 each): hop1 O1=1.0000, hop2 O1=0.9985, hop3 O1=**0.7853**.

**STOP RULE verdict**: MetaQA 0.9279 >= .90 -- PASS. WebQSP 0.9859 >= .95 -- PASS. Both targets are
ranking-reachable in principle. Universal finding across all 6 corpora: O1==O2==O3 and zero
budget-limited queries everywhere -- K_q never binds, so the entire F6-vs-ceiling gap is a pure
ranking problem, not a budget/reach problem. This is what authorized Parts B-I instead of
2-hop/exposure work. MetaQA's shortfall is almost entirely hop3 (0.7853 ceiling vs ~1.0 for
hop1/hop2) -- even a perfect ranker caps at 0.9279; closing the remaining 0.0721 needs deeper
reach, not ranking, consistent with the instruction not to chase 2-hop now.

### Parts B-E: new structural rankers, all reusing cached/static data (no new embeddings, no Modal)

G1_QUERY_WEIGHTED_SUPPORT (raw-edge neighbour support weighted by supporter's own query rank /
supporter's degree). G2_CORE_RANK_SUPPORT (F2's boundary mass reweighted by the supporting core's
own selection rank). G3_HYPER_COHERENCE (H4_SPLIT_PRESERVE's own hyperedge membership, no rebuild
-- query-weighted shared-hyperedge support). G4_STRUCT_RRF = RRF(G1,G2,G3). G5_STRUCT_QUERY_RRF =
RRF(G4, cached query-relevance rank). SP0-SP3 (Part F): universal STRUCT-vs-KNN precedence tiers on
top of G5 (SP0=G5 alias, SP1=STRUCT-supported first, SP2=multi-core-STRUCT first, SP3=KNN-only
demoted below comparable-or-less-supported STRUCT candidates) -- same rule on all 6 corpora, no
`if dataset==` conditionals.

ALL_REQUIRED_FETCHED vs F6, 5/6 corpora complete (hotpotqa_clean still running, F6 already 0.9785
there so unlikely to change the verdict):

| method | metaqa | 2wiki | musique | squad | webqsp | worst_delta | macro_delta | sig regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F6 (ref) | 0.8158 | 0.9500 | 0.9805 | 0.9965 | 0.9133 | -- | -- | -- |
| G1 | 0.8614 | 0.9800 | 0.9870 | 0.9945 | 0.9521 | -0.0020 | +0.0238 | none |
| G2 | 0.8138 | 0.9475 | 0.9735 | 0.9920 | 0.8569 | **-0.0564** | -0.0145 | musique, squad, webqsp |
| G3 | 0.8644 | 0.9545 | 0.9800 | 0.9935 | 0.9528 | -0.0030 | +0.0178 | none |
| G4 | 0.8664 | 0.9795 | 0.9865 | 0.9945 | 0.9542 | -0.0020 | +0.0250 | none |
| G5 / SP0 | 0.8664 | 0.9815 | 0.9905 | 0.9970 | 0.9570 | **+0.0005** | +0.0273 | none |
| SP1 | **0.8669** | 0.9775 | 0.9810 | 0.9955 | **0.9626** | -0.0010 (ns) | +0.0255 | none |
| SP2 | 0.8609 | 0.9765 | 0.9810 | 0.9955 | 0.9380 | -0.0010 (ns) | +0.0192 | none |
| SP3 | 0.8664 | 0.9810 | 0.9890 | 0.9965 | 0.9612 | +0.0000 | +0.0276 | none |

G2 is refuted (significant regression on 3/5 corpora, never a significant win) -- excluded going
forward. G5/SP0 and SP3 are the cleanest universal candidates (never negative on any corpus, even
non-significantly). **SP1 wins outright on both primary target corpora** (metaqa 0.8669, webqsp
0.9626) with its only dip a non-significant -0.0010 on squad (already at 99.65%, p=0.625).

MetaQA: 0.8158 -> 0.8669 (SP1), +0.0511, closing **46%** of the 0.1121 headroom to ceiling (0.9279)
-- real progress, target (0.90) not reached. WebQSP: 0.9133 -> 0.9626 (SP1), +0.0493, closing **68%**
of the 0.0726 headroom to ceiling (0.9859) -- **target (0.95) reached.**

### Part G: MetaQA hop-stratified rescue of nodes still missing under F6

| hop | n missing nodes | n missing queries | F6 rescue | best rescue (method) |
|---|---:|---:|---:|---:|
| hop1 | 1 | 1 | 0.0 | 0.0 (n=1, noise) |
| hop2 | 809 | 118 | 0.0 | **0.7966 (G3)** |
| hop3 | 2606 | 243 | 0.0 | 0.0823 (G4/G5/SP0/SP1/SP3, tied) |

hop2's residual is almost entirely rescuable by structure (G3 alone recovers 79.7% of previously-
failing hop2 queries). hop3's residual is not: even the best method only converts 8.2% of missing
hop3 queries, an order of magnitude weaker than hop2 -- this gap motivated Part I below.

### Part I: feature-separability audit, MetaQA hop3 residual (conditional, gold used diagnostically only)

Pooled (query, candidate) rows over all 666 hop3 queries, label = still-missing required node after
core resolution (3,003 positive / 6,251,514 negative rows). AUC: 0.5 = chance, 1.0 = perfect.

| feature | AUC | top-decile enrichment |
|---|---:|---:|
| support count (cnt) | 0.611 | 1.79x |
| F2 score | 0.597 | **2.17x** |
| G2 (core-rank support) | 0.576 | 2.05x |
| G3 (hyperedge coherence) | 0.567 | 1.07x |
| G1 (query-weighted support) | 0.548 | 1.00x |
| degree | 0.528 | 0.85x |
| n_families | 0.526 | 1.05x |
| Dense rank | 0.508 | 1.00x |
| RRF rank | 0.503 | 1.00x |
| SPLADE rank | 0.498 | 1.00x |
| best supporting-core rank | 0.430 | 1.05x |
| has_struct | 0.393 | 0.70x |

No feature separates strongly (best AUC 0.611, weak by any standard threshold). Direct query
relevance sits at pure chance (0.498-0.508) -- confirms the motivating hypothesis that direct
lexical/dense relevance carries no signal for 3rd-hop answers. `has_struct` and best-core-rank are
mildly *anti*-correlated (AUC<0.5): among the hop3 residual specifically, structural support has
already been exhausted by definition, so its presence stops being informative. Per the pre-
committed decision rule: **stop L1 ranking work on MetaQA's hop3 residual; hand it to learned L2.**

### Cumulative: Production METIS -> H4+F2 -> H4+F6 -> H4+SP1 (5/6 corpora; hotpot pending)

| corpus | Production | H4+F2 | H4+F6 | H4+SP1 | net vs Production |
|---|---:|---:|---:|---:|---:|
| MetaQA | 0.6612 | 0.8148 | 0.8158 | 0.8669 | +0.2057 |
| WebQSP | 0.7646 | 0.8548 | 0.9133 | 0.9626 | +0.1980 |
| 2Wiki | 0.9435 | 0.9475 | 0.9500 | 0.9775 | +0.0340 |
| MuSiQue | 0.9635 | 0.9735 | 0.9805 | 0.9810 | +0.0175 |
| SQuAD | 0.9875 | 0.9920 | 0.9965 | 0.9955 | +0.0080 |
| HotpotQA | 0.9505 | 0.9665 | 0.9785 | *pending* | *pending* |

`PART_0D_BEST_UNIVERSAL_METHOD_5OF6 = SP1_STRUCT_PRECEDENCE`. `PART_0D_PROMOTED = NONE` -- same
frozen-L2 hold as Stages 5-6. WebQSP's 0.95 target is met; MetaQA's 0.90 target is not, and Part I
says further L1 ranking work on its hop3 residual specifically is not promising -- that piece
belongs to learned L2, not another L1 ranker.
