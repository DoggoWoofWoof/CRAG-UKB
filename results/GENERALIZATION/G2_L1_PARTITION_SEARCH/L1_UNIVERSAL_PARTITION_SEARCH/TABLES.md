# TABLES -- universal balanced partition + overlap search

## P8  EXPLOSION AUDIT -- unbounded 1-hop overlap replication factor

`R = sum_j |C_j u H_j| / |V|`.  R=1 is a hard partition.

| corpus   | core | STRUCT | NERX | KNN  | FULL_C | FULLC max mult | FULLC p99 |
|----------|------|--------|------|------|--------|----------------|-----------|
| MetaQA   | 1.00 | 4.07   | 5.32 | 2.68 | 9.18   | 397            | 40.0      |
| WebQSP   | 1.00 | 3.18   | 1.32 | 2.06 | 4.42   | 4964           | 23.0      |
| 2Wiki    | 1.00 | 3.02   | 6.29 | 3.25 | 9.97   | 658            | 46.0      |
| MuSiQue  | 1.00 | 2.74   | 4.56 | 2.68 | 7.29   | 131            | 36.0      |
| SQuAD    | 1.00 | 9.08   | 7.39 | 2.68 | 15.86  | 190            | 96.0      |
| HotpotQA | 1.00 | 9.43   | 6.61 | 3.68 | 16.93  | 5074           | 61.0      |

## P10/P12  ALL_REQUIRED_FETCHED and selected-P50 exposure

Coverage is measured on required NODES, never on a canonical partition id.

### unbounded

| corpus   | core   | O1_STRUCT       | O2_NERX        | O3_KNN         | O4_FULL_C       |
|----------|--------|-----------------|----------------|----------------|-----------------|
| MetaQA   | 0.6612 | 0.9234 (x3.85)  | 0.8128 (x3.16) | 0.7362 (x2.31) | 0.9384 (x4.92)  |
| WebQSP   | 0.7646 | 0.9803 (x10.76) | 0.7829 (x1.40) | 0.8062 (x2.11) | 0.9817 (x11.75) |
| 2Wiki    | 0.9435 | 0.9915 (x8.84)  | 0.9770 (x3.84) | 0.9705 (x2.69) | 0.9970 (x9.80)  |
| MuSiQue  | 0.9635 | 0.9820 (x1.58)  | 0.9915 (x1.81) | 0.9935 (x1.70) | 0.9980 (x2.24)  |
| SQuAD    | 0.9875 | 0.9955 (x2.34)  | 0.9980 (x2.52) | 0.9955 (x1.94) | 1.0000 (x3.32)  |
| HotpotQA | 0.9505 | 0.9975 (x48.13) | 0.9700 (x5.62) | 0.9740 (x3.42) | 0.9980 (x50.43) |

### beta=0.25

| corpus   | core   | O1_STRUCT      | O2_NERX        | O3_KNN         | O4_FULL_C      |
|----------|--------|----------------|----------------|----------------|----------------|
| MetaQA   | 0.6612 | 0.7668 (x1.13) | 0.6912 (x1.19) | 0.6662 (x1.20) | 0.7032 (x1.18) |
| WebQSP   | 0.7646 | 0.8062 (x1.18) | 0.7752 (x1.15) | 0.7759 (x1.22) | 0.8013 (x1.18) |
| 2Wiki    | 0.9435 | 0.9445 (x1.11) | 0.9520 (x1.19) | 0.9520 (x1.19) | 0.9500 (x1.16) |
| MuSiQue  | 0.9635 | 0.9650 (x1.05) | 0.9775 (x1.12) | 0.9735 (x1.12) | 0.9730 (x1.10) |
| SQuAD    | 0.9875 | 0.9880 (x1.07) | 0.9890 (x1.15) | 0.9905 (x1.15) | 0.9895 (x1.12) |
| HotpotQA | 0.9505 | 0.9505 (x1.08) | 0.9545 (x1.23) | 0.9575 (x1.22) | 0.9550 (x1.13) |

### beta=0.5

| corpus   | core   | O1_STRUCT      | O2_NERX        | O3_KNN         | O4_FULL_C      |
|----------|--------|----------------|----------------|----------------|----------------|
| MetaQA   | 0.6612 | 0.8028 (x1.24) | 0.7047 (x1.37) | 0.6702 (x1.40) | 0.7432 (x1.36) |
| WebQSP   | 0.7646 | 0.8280 (x1.35) | 0.7780 (x1.23) | 0.7822 (x1.44) | 0.8118 (x1.39) |
| 2Wiki    | 0.9435 | 0.9480 (x1.25) | 0.9545 (x1.38) | 0.9580 (x1.39) | 0.9540 (x1.35) |
| MuSiQue  | 0.9635 | 0.9670 (x1.10) | 0.9795 (x1.22) | 0.9770 (x1.24) | 0.9775 (x1.20) |
| SQuAD    | 0.9875 | 0.9885 (x1.14) | 0.9910 (x1.29) | 0.9910 (x1.29) | 0.9910 (x1.23) |
| HotpotQA | 0.9505 | 0.9520 (x1.18) | 0.9585 (x1.46) | 0.9605 (x1.45) | 0.9575 (x1.29) |

### beta=1.0

| corpus   | core   | O1_STRUCT      | O2_NERX        | O3_KNN         | O4_FULL_C      |
|----------|--------|----------------|----------------|----------------|----------------|
| MetaQA   | 0.6612 | 0.8293 (x1.51) | 0.7267 (x1.70) | 0.6992 (x1.77) | 0.7968 (x1.70) |
| WebQSP   | 0.7646 | 0.8689 (x1.69) | 0.7815 (x1.32) | 0.7963 (x1.80) | 0.8414 (x1.81) |
| 2Wiki    | 0.9435 | 0.9510 (x1.51) | 0.9610 (x1.72) | 0.9625 (x1.76) | 0.9600 (x1.71) |
| MuSiQue  | 0.9635 | 0.9680 (x1.20) | 0.9845 (x1.38) | 0.9860 (x1.44) | 0.9820 (x1.38) |
| SQuAD    | 0.9875 | 0.9890 (x1.26) | 0.9935 (x1.53) | 0.9935 (x1.55) | 0.9915 (x1.42) |
| HotpotQA | 0.9505 | 0.9540 (x1.40) | 0.9610 (x1.92) | 0.9670 (x1.92) | 0.9625 (x1.63) |

## P15  MATCHED-NODE-EXPOSURE CONTROL (the decisive table)

Overlap vs simply selecting MORE hard core partitions until the unique-node budget
matches.  Values are difference-in-differences: the cell's advantage over its
matched hard depth, MINUS the advantage O0_CORE already has at P'=50 (which is the
F6-vs-BASE selector gap, not an overlap effect).  `*` = paired bootstrap 95% CI
excludes zero.  `(unmatched)` = the hard control saturated the 200-block ranking
and read FEWER nodes, so no matched claim is possible.

| cell            | MetaQA      | WebQSP      | 2Wiki       | MuSiQue  | SQuAD    | HotpotQA    |
|-----------------|-------------|-------------|-------------|----------|----------|-------------|
| O1_STRUCT       | +0.1271*    | (unmatched) | (unmatched) | -0.0080* | -0.0075* | (unmatched) |
| O1_STRUCT_b0.25 | +0.0916*    | +0.0275*    | -0.0020     | -0.0030  | -0.0020  | -0.0020*    |
| O1_STRUCT_b0.5  | +0.1166*    | +0.0303*    | -0.0045     | -0.0025  | -0.0030  | -0.0040*    |
| O1_STRUCT_b1.0  | +0.1246*    | +0.0416*    | -0.0085*    | -0.0080* | -0.0055* | -0.0075*    |
| O2_NERX         | +0.0385*    | -0.0176*    | -0.0015     | -0.0025  | -0.0070* | (unmatched) |
| O2_NERX_b0.25   | +0.0085     | -0.0007     | +0.0015     | +0.0075* | -0.0030  | -0.0025     |
| O2_NERX_b0.5    | +0.0080     | -0.0092     | -0.0020     | +0.0025  | -0.0040  | -0.0050     |
| O2_NERX_b1.0    | +0.0100     | -0.0141*    | -0.0010     | +0.0010  | -0.0025  | -0.0135*    |
| O3_KNN          | -0.0075     | -0.0458*    | -0.0015     | +0.0020  | -0.0050  | -0.0105*    |
| O3_KNN_b0.25    | -0.0165*    | -0.0091     | +0.0015     | +0.0035  | -0.0010  | +0.0005     |
| O3_KNN_b0.5     | -0.0275*    | -0.0205*    | +0.0015     | -0.0010  | -0.0040  | -0.0030     |
| O3_KNN_b1.0     | -0.0195*    | -0.0374*    | +0.0000     | +0.0015  | -0.0030  | -0.0075     |
| O4_FULL_C       | (unmatched) | (unmatched) | (unmatched) | -0.0035  | -0.0070* | (unmatched) |
| O4_FULL_C_b0.25 | +0.0230*    | +0.0226*    | +0.0000     | +0.0035  | -0.0020  | +0.0000     |
| O4_FULL_C_b0.5  | +0.0480*    | +0.0113     | -0.0005     | +0.0015  | -0.0035  | -0.0020     |
| O4_FULL_C_b1.0  | +0.0801*    | +0.0056     | -0.0020     | -0.0015  | -0.0045  | -0.0075*    |

## P14  FAMILY CONTRIBUTION TO THE HALO (evaluation-only provenance)

| corpus   | budget    | NERX-STRUCT | KNN-STRUCT | best single | FULLC-best |
|----------|-----------|-------------|------------|-------------|------------|
| MetaQA   | unbounded | -0.1106     | -0.1872    | O1_STRUCT   | +0.0150    |
| MetaQA   | b0.25     | -0.0756     | -0.1006    | O1_STRUCT   | -0.0636    |
| MetaQA   | b0.5      | -0.0981     | -0.1326    | O1_STRUCT   | -0.0596    |
| MetaQA   | b1.0      | -0.1026     | -0.1301    | O1_STRUCT   | -0.0325    |
| WebQSP   | unbounded | -0.1974     | -0.1741    | O1_STRUCT   | +0.0014    |
| WebQSP   | b0.25     | -0.0310     | -0.0303    | O1_STRUCT   | -0.0049    |
| WebQSP   | b0.5      | -0.0500     | -0.0458    | O1_STRUCT   | -0.0162    |
| WebQSP   | b1.0      | -0.0874     | -0.0726    | O1_STRUCT   | -0.0275    |
| 2Wiki    | unbounded | -0.0145     | -0.0210    | O1_STRUCT   | +0.0055    |
| 2Wiki    | b0.25     | +0.0075     | +0.0075    | O2_NERX     | -0.0020    |
| 2Wiki    | b0.5      | +0.0065     | +0.0100    | O3_KNN      | -0.0040    |
| 2Wiki    | b1.0      | +0.0100     | +0.0115    | O3_KNN      | -0.0025    |
| MuSiQue  | unbounded | +0.0095     | +0.0115    | O3_KNN      | +0.0045    |
| MuSiQue  | b0.25     | +0.0125     | +0.0085    | O2_NERX     | -0.0045    |
| MuSiQue  | b0.5      | +0.0125     | +0.0100    | O2_NERX     | -0.0020    |
| MuSiQue  | b1.0      | +0.0165     | +0.0180    | O3_KNN      | -0.0040    |
| SQuAD    | unbounded | +0.0025     | +0.0000    | O2_NERX     | +0.0020    |
| SQuAD    | b0.25     | +0.0010     | +0.0025    | O3_KNN      | -0.0010    |
| SQuAD    | b0.5      | +0.0025     | +0.0025    | O2_NERX     | +0.0000    |
| SQuAD    | b1.0      | +0.0045     | +0.0045    | O2_NERX     | -0.0020    |
| HotpotQA | unbounded | -0.0275     | -0.0235    | O1_STRUCT   | +0.0005    |
| HotpotQA | b0.25     | +0.0040     | +0.0070    | O3_KNN      | -0.0025    |
| HotpotQA | b0.5      | +0.0065     | +0.0085    | O3_KNN      | -0.0030    |
| HotpotQA | b1.0      | +0.0070     | +0.0130    | O3_KNN      | -0.0045    |

## P3  HYPEREDGE SIZES BEFORE THE UNIVERSAL CAP

Cap = 25, identical on every corpus and family.  A closed neighbourhood over the cap is dropped whole.

| corpus   | family         | hyperedges | mean  | p90   | p99   | p99.9  | max     | over cap | pins over cap |
|----------|----------------|------------|-------|-------|-------|--------|---------|----------|---------------|
| MetaQA   | H_STRUCT_LOCAL | 40,151     | 6.45  | 11.0  | 28.0  | 280.5  | 4,181   | 1.24%    | 19.71%        |
| MetaQA   | H_KNN_LOCAL    | 38,141     | 3.89  | 6.0   | 10.0  | 17.0   | 42      | 0.02%    | 0.16%         |
| WebQSP   | H_STRUCT_LOCAL | 781,485    | 5.19  | 6.0   | 36.0  | 258.5  | 41,606  | 1.56%    | 36.02%        |
| WebQSP   | H_KNN_LOCAL    | 773,107    | 5.33  | 8.0   | 14.0  | 24.0   | 181     | 0.08%    | 0.52%         |
| 2Wiki    | H_STRUCT_LOCAL | 54,467     | 5.63  | 7.0   | 21.0  | 114.0  | 38,593  | 0.66%    | 28.04%        |
| 2Wiki    | H_KNN_LOCAL    | 65,508     | 5.11  | 8.0   | 12.0  | 18.0   | 31      | 0.01%    | 0.03%         |
| MuSiQue  | H_STRUCT_LOCAL | 9,880      | 11.43 | 23.0  | 121.0 | 320.0  | 1,487   | 8.93%    | 52.84%        |
| MuSiQue  | H_KNN_LOCAL    | 13,490     | 5.06  | 7.0   | 11.0  | 16.0   | 27      | 0.01%    | 0.04%         |
| SQuAD    | H_STRUCT_LOCAL | 12,947     | 108.3 | 218.0 | 514.0 | 2270.0 | 2,335   | 95.57%   | 99.13%        |
| SQuAD    | H_KNN_LOCAL    | 15,677     | 4.61  | 7.0   | 10.0  | 14.0   | 23      | 0.00%    | 0.00%         |
| HotpotQA | H_STRUCT_LOCAL | 503,974    | 14.6  | 16.0  | 62.0  | 826.1  | 46,892  | 3.19%    | 43.49%        |
| HotpotQA | H_KNN_LOCAL    | 503,932    | 5.63  | 8.0   | 13.0  | 20.0   | 260,450 | 0.02%    | 9.29%         |

## P16  BEST HARD CORE x HALO

The one hard partition that moved a corpus (H4 true hypergraph) under the SAME
halo recipe as the production METIS core.  Only this pairing is run.

| corpus  | halo           | METIS core | H4 core | delta   | METIS R | H4 R  | METIS expo | H4 expo |
|---------|----------------|------------|---------|---------|---------|-------|------------|---------|
| MetaQA  | O0_CORE        | 0.6612     | 0.7898  | +0.1286 | 1.00    | 1.00  | x1.00      | x1.00   |
| MetaQA  | O1_STRUCT_b1.0 | 0.8293     | 0.8323  | +0.0030 | 1.99    | 2.00  | x1.51      | x1.49   |
| MetaQA  | O4_FULL_C_b0.5 | 0.7432     | 0.8093  | +0.0661 | 1.50    | 1.50  | x1.36      | x1.34   |
| MetaQA  | O4_FULL_C_b1.0 | 0.7968     | 0.8233  | +0.0265 | 2.00    | 2.00  | x1.70      | x1.67   |
| MetaQA  | O4_FULL_C      | 0.9384     | 0.9324  | -0.0060 | 9.18    | 10.98 | x4.92      | x5.30   |
| 2Wiki   | O0_CORE        | 0.9435     | 0.9415  | -0.0020 | 1.00    | 1.00  | x1.00      | x1.00   |
| 2Wiki   | O1_STRUCT_b1.0 | 0.9510     | 0.9435  | -0.0075 | 1.77    | 1.64  | x1.51      | x1.36   |
| 2Wiki   | O4_FULL_C_b0.5 | 0.9540     | 0.9510  | -0.0030 | 1.50    | 1.50  | x1.35      | x1.31   |
| 2Wiki   | O4_FULL_C_b1.0 | 0.9600     | 0.9595  | -0.0005 | 2.00    | 2.00  | x1.71      | x1.62   |
| 2Wiki   | O4_FULL_C      | 0.9970     | 0.9985  | +0.0015 | 9.97    | 11.53 | x9.80      | x10.26  |
| MuSiQue | O0_CORE        | 0.9635     | 0.9625  | -0.0010 | 1.00    | 1.00  | x1.00      | x1.00   |
| MuSiQue | O1_STRUCT_b1.0 | 0.9680     | 0.9705  | +0.0025 | 1.91    | 1.91  | x1.20      | x1.16   |
| MuSiQue | O4_FULL_C_b0.5 | 0.9775     | 0.9775  | +0.0000 | 1.50    | 1.50  | x1.20      | x1.19   |
| MuSiQue | O4_FULL_C_b1.0 | 0.9820     | 0.9860  | +0.0040 | 2.00    | 2.00  | x1.38      | x1.35   |
| MuSiQue | O4_FULL_C      | 0.9980     | 0.9985  | +0.0005 | 7.29    | 8.80  | x2.24      | x2.26   |
| SQuAD   | O0_CORE        | 0.9875     | 0.9745  | -0.0130 | 1.00    | 1.00  | x1.00      | x1.00   |
| SQuAD   | O1_STRUCT_b1.0 | 0.9890     | 0.9810  | -0.0080 | 2.00    | 2.00  | x1.26      | x1.23   |
| SQuAD   | O4_FULL_C_b0.5 | 0.9910     | 0.9760  | -0.0150 | 1.50    | 1.50  | x1.23      | x1.15   |
| SQuAD   | O4_FULL_C_b1.0 | 0.9915     | 0.9800  | -0.0115 | 2.00    | 2.00  | x1.42      | x1.30   |
| SQuAD   | O4_FULL_C      | 1.0000     | 0.9980  | -0.0020 | 15.86   | 36.69 | x3.32      | x3.24   |

## P11  MIN_BLOCK_COVER -- blocks needed to cover all required nodes

| corpus   | cell            | median | p90  | p95  | max | exact frac |
|----------|-----------------|--------|------|------|-----|------------|
| MetaQA   | O0_CORE         | 2.0    | 15.0 | 28.1 | 104 | 1.0        |
| MetaQA   | O1_STRUCT_b0.25 | 1.0    | 9.0  | 23.0 | 102 | 0.982      |
| MetaQA   | O4_FULL_C_b0.25 | 1.0    | 12.0 | 23.1 | 98  | 0.9765     |
| MetaQA   | O4_FULL_C       | 1.0    | 3.0  | 5.0  | 36  | 0.9675     |
| WebQSP   | O0_CORE         | 1.0    | 7.0  | 15.0 | 69  | 1.0        |
| WebQSP   | O1_STRUCT_b0.25 | 1.0    | 5.0  | 10.0 | 66  | 0.9831     |
| WebQSP   | O4_FULL_C_b0.25 | 1.0    | 5.0  | 10.1 | 66  | 0.9859     |
| WebQSP   | O4_FULL_C       | 1.0    | 1.0  | 1.0  | 17  | 0.9972     |
| 2Wiki    | O0_CORE         | 2.0    | 3.0  | 4.0  | 4   | 1.0        |
| 2Wiki    | O1_STRUCT_b0.25 | 2.0    | 3.0  | 3.0  | 4   | 1.0        |
| 2Wiki    | O4_FULL_C_b0.25 | 2.0    | 3.0  | 3.0  | 4   | 1.0        |
| 2Wiki    | O4_FULL_C       | 1.0    | 2.0  | 2.0  | 3   | 1.0        |
| MuSiQue  | O0_CORE         | 2.0    | 3.0  | 3.0  | 4   | 1.0        |
| MuSiQue  | O1_STRUCT_b0.25 | 2.0    | 3.0  | 3.0  | 4   | 1.0        |
| MuSiQue  | O4_FULL_C_b0.25 | 2.0    | 2.0  | 3.0  | 4   | 1.0        |
| MuSiQue  | O4_FULL_C       | 1.0    | 2.0  | 2.0  | 3   | 1.0        |
| SQuAD    | O0_CORE         | 1.0    | 1.0  | 1.0  | 1   | 1.0        |
| SQuAD    | O1_STRUCT_b0.25 | 1.0    | 1.0  | 1.0  | 1   | 1.0        |
| SQuAD    | O4_FULL_C_b0.25 | 1.0    | 1.0  | 1.0  | 1   | 1.0        |
| SQuAD    | O4_FULL_C       | 1.0    | 1.0  | 1.0  | 1   | 1.0        |
| HotpotQA | O0_CORE         | 2.0    | 2.0  | 2.0  | 2   | 1.0        |
| HotpotQA | O1_STRUCT_b0.25 | 2.0    | 2.0  | 2.0  | 2   | 1.0        |
| HotpotQA | O4_FULL_C_b0.25 | 2.0    | 2.0  | 2.0  | 2   | 1.0        |
| HotpotQA | O4_FULL_C       | 1.0    | 1.0  | 2.0  | 2   | 1.0        |

## P1-P5  HARD PARTITIONER x REPRESENTATION MATRIX

Retrieval utility under the FROZEN L1 contract, delta of F6 ALL@50 vs the
production METIS partition.  Significance is exact McNemar AND above the
measured METIS reseed noise floor for that corpus.

| partitioner x graph                              | MetaQA  | WebQSP  | 2Wiki   | MuSiQue | SQuAD   | HotpotQA | worst   | macro   | sig reg | sig gains |
|--------------------------------------------------|---------|---------|---------|---------|---------|----------|---------|---------|---------|-----------|
| H2_KAHIP_CONNECTED__G0_TOPOLOGY_C                | --      | --      | --      | -0.0005 | -0.0015 | --       | -0.0015 | -0.0010 | 0       | 0         |
| H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | -0.0020 | --      | --      | +0.0005 | +0.0010 | --       | -0.0020 | -0.0002 | 0       | 0         |
| H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | +0.0030 | +0.0078 | +0.0015 | -0.0050 | +0.0005 | +0.0005  | -0.0050 | +0.0014 | 0       | 0         |
| H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | +0.1201 | +0.0571 | -0.0045 | -0.0005 | -0.0120 | +0.0140  | -0.0120 | +0.0290 | 1       | 3         |
| H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | +0.1286 | +0.0599 | -0.0020 | -0.0010 | -0.0130 | +0.0150  | -0.0130 | +0.0312 | 1       | 3         |

### P4  balance, cut and boundary

| corpus   | partition                                        | max/mean | size CV | min | edge cut | boundary | F6 ALL@50 | BASE ALL@50 |
|----------|--------------------------------------------------|----------|---------|-----|----------|----------|-----------|-------------|
| MetaQA   | CURRENT                                          | 1.0287   | 0.0216  | 96  | 0.5904   | 0.9235   | 0.6612    | 0.6587      |
| MetaQA   | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 1.0387   | 0.0571  | 56  | 0.5792   | 0.9197   | 0.6642    | 0.6582      |
| MetaQA   | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 1.0387   | 0.1486  | 2   | 0.7983   | 0.8679   | 0.7813    | 0.7718      |
| MetaQA   | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 1.0387   | 0.1289  | 1   | 0.7761   | 0.889    | 0.7898    | 0.7913      |
| MetaQA   | H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | 1.0287   | 0.0675  | 26  | 0.5867   | 0.9205   | 0.6592    | 0.6577      |
| WebQSP   | CURRENT                                          | 1.0299   | 0.0203  | 86  | 0.4444   | 0.8838   | 0.7646    | 0.7618      |
| WebQSP   | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 1.0399   | 0.062   | 1   | 0.4334   | 0.8821   | 0.7724    | 0.7653      |
| WebQSP   | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 1.0399   | 0.0956  | 1   | 0.5373   | 0.9061   | 0.8217    | 0.8125      |
| WebQSP   | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 1.0399   | 0.0612  | 1   | 0.53     | 0.8873   | 0.8245    | 0.8182      |
| 2Wiki    | CURRENT                                          | 1.029    | 0.02    | 97  | 0.5981   | 0.9781   | 0.9435    | 0.9375      |
| 2Wiki    | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 1.039    | 0.0801  | 10  | 0.5884   | 0.9762   | 0.945     | 0.9355      |
| 2Wiki    | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 1.039    | 0.1013  | 6   | 0.7607   | 0.9664   | 0.939     | 0.934       |
| 2Wiki    | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 1.039    | 0.082   | 2   | 0.7698   | 0.9652   | 0.9415    | 0.9395      |
| MuSiQue  | CURRENT                                          | 1.0246   | 0.0235  | 96  | 0.4532   | 0.9201   | 0.9635    | 0.9565      |
| MuSiQue  | H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | 1.0246   | 0.0438  | 79  | 0.447    | 0.9117   | 0.964     | 0.955       |
| MuSiQue  | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 1.0345   | 0.0567  | 75  | 0.4387   | 0.9122   | 0.9585    | 0.948       |
| MuSiQue  | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 1.0345   | 0.0697  | 70  | 0.6221   | 0.891    | 0.963     | 0.95        |
| MuSiQue  | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 1.0345   | 0.0361  | 85  | 0.6452   | 0.9011   | 0.9625    | 0.955       |
| MuSiQue  | H2_KAHIP_CONNECTED__G0_TOPOLOGY_C                | 1.0345   | 0.0731  | 48  | 0.451    | 0.9153   | 0.963     | 0.9555      |
| SQuAD    | CURRENT                                          | 1.0284   | 0.0313  | 82  | 0.6077   | 0.9837   | 0.9875    | 0.9805      |
| SQuAD    | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 1.0384   | 0.1153  | 1   | 0.5772   | 0.9713   | 0.988     | 0.9795      |
| SQuAD    | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 1.0384   | 0.0388  | 83  | 0.9138   | 0.9231   | 0.9755    | 0.9465      |
| SQuAD    | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 1.0384   | 0.0219  | 86  | 0.9188   | 0.9255   | 0.9745    | 0.9365      |
| SQuAD    | H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | 1.0384   | 0.0905  | 34  | 0.5765   | 0.9719   | 0.9885    | 0.9825      |
| SQuAD    | H2_KAHIP_CONNECTED__G0_TOPOLOGY_C                | 1.0384   | 0.0836  | 37  | 0.5756   | 0.9719   | 0.986     | 0.9805      |
| HotpotQA | CURRENT                                          | 1.0298   | 0.0229  | 71  | 0.6909   | 0.9985   | 0.9505    | 0.9345      |
| HotpotQA | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 1.0398   | 0.1001  | 1   | 0.6834   | 0.9985   | 0.951     | 0.9325      |
| HotpotQA | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 1.0398   | 0.142   | 1   | 0.8203   | 0.9981   | 0.9645    | 0.952       |
| HotpotQA | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 1.0398   | 0.1124  | 1   | 0.8318   | 0.9979   | 0.9655    | 0.956       |

### P4  minimum blocks required by the gold/reference nodes

min_blocks = number of DISTINCT partitions the required nodes of one query fall into; the fewest blocks any selector would have to pick.  reported per the program and deliberately NOT used as the winning objective.  On this matrix the association with utility is real (see PREDICTIVENESS) but not sufficient: it cannot vary at all on a one-gold-node corpus whose utility does vary, and the previous program has a standing counter-example where every containment statistic worsened while F6 ALL@50 gained significantly.

| corpus   | partition                                        | required nodes | min blocks mean | median | max | single-block queries |
|----------|--------------------------------------------------|----------------|-----------------|--------|-----|----------------------|
| MetaQA   | CURRENT                                          | 7.446          | 5.784           | 2.0    | 104 | 46.2%                |
| MetaQA   | H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | 7.446          | 5.815           | 2.0    | 110 | 45.5%                |
| MetaQA   | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 7.446          | 5.816           | 2.0    | 108 | 45.9%                |
| MetaQA   | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 7.446          | 4.67            | 1.0    | 92  | 60.6%                |
| MetaQA   | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 7.446          | 4.569           | 1.0    | 100 | 61.0%                |
| WebQSP   | CURRENT                                          | 5.965          | 3.664           | 1.0    | 69  | 52.1%                |
| WebQSP   | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 5.965          | 3.574           | 1.0    | 70  | 53.0%                |
| WebQSP   | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 5.965          | 3.264           | 1.0    | 68  | 55.4%                |
| WebQSP   | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 5.965          | 3.08            | 1.0    | 68  | 56.3%                |
| 2Wiki    | CURRENT                                          | 2.417          | 1.98            | 2.0    | 4   | 23.6%                |
| 2Wiki    | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 2.417          | 1.974           | 2.0    | 4   | 24.1%                |
| 2Wiki    | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 2.417          | 2.09            | 2.0    | 4   | 20.3%                |
| 2Wiki    | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 2.417          | 1.954           | 2.0    | 4   | 26.7%                |
| MuSiQue  | CURRENT                                          | 2.34           | 1.775           | 2.0    | 4   | 37.2%                |
| MuSiQue  | H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | 2.34           | 1.758           | 2.0    | 4   | 39.0%                |
| MuSiQue  | H2_KAHIP_CONNECTED__G0_TOPOLOGY_C                | 2.34           | 1.768           | 2.0    | 4   | 37.1%                |
| MuSiQue  | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 2.34           | 1.736           | 2.0    | 4   | 40.6%                |
| MuSiQue  | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 2.34           | 1.858           | 2.0    | 4   | 29.8%                |
| MuSiQue  | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 2.34           | 1.881           | 2.0    | 4   | 29.0%                |
| SQuAD    | CURRENT                                          | 1.0            | 1.0             | 1.0    | 1   | 100.0%               |
| SQuAD    | H1_KAHIP_STRONG__G0_TOPOLOGY_C                   | 1.0            | 1.0             | 1.0    | 1   | 100.0%               |
| SQuAD    | H2_KAHIP_CONNECTED__G0_TOPOLOGY_C                | 1.0            | 1.0             | 1.0    | 1   | 100.0%               |
| SQuAD    | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 1.0            | 1.0             | 1.0    | 1   | 100.0%               |
| SQuAD    | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 1.0            | 1.0             | 1.0    | 1   | 100.0%               |
| SQuAD    | H4_MTKAHYPAR_TRUE_HYPERGRAPH__G2_TRUE_HYPERGRAPH | 1.0            | 1.0             | 1.0    | 1   | 100.0%               |
| HotpotQA | CURRENT                                          | 2.0            | 1.664           | 2.0    | 2   | 33.6%                |
| HotpotQA | H3_MTKAHYPAR_GRAPH__G0_TOPOLOGY_C                | 2.0            | 1.66            | 2.0    | 2   | 34.0%                |
| HotpotQA | H3_MTKAHYPAR_GRAPH__G1_LOCAL_HYPER_CLIQUE        | 2.0            | 1.718           | 2.0    | 2   | 28.2%                |

Across the 22 partitioner-by-corpus cells, the change in mean min-blocks and the change in F6 ALL@50 are monotonically associated: Spearman rho = -0.490, p = 0.0205.  Pearson reads -0.980 but is dominated by two MetaQA cells, so Spearman is the honest summary.  Containment still cannot be the objective: on SQuAD every query has exactly one required node, so min-blocks is 1 for EVERY partition and cannot vary -- yet its utility spans 0.0140, and it is the corpus that blocks the universal gate.
