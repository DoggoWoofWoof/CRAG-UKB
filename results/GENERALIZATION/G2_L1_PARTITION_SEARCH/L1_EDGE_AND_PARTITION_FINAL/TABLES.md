# L1 EDGE SUBSTRATE + PARTITION UTILITY -- TABLES

Frozen contract everywhere below: MASTER_TOPOLOGY=C, P=50 exactly, B=6, K0=60, M_struct=64, M_ret=32, MAX_HOPS=3, BEAM=64, DEG_CAP=300, SEED_K=5. No learning, no LLM, no encoder call, no TEST split.

## A0 -- edge algebra (verified from source artifacts)

`S = E_STRUCT` (master_nodes.neighbors, the ONLY family the frozen traversal walks), `N = E_NERX = NER \ S`, `K = E_KNN = A \ S` where `A = gte_qwen/graph.pt`.

| quantity | metaqa | 2wiki | musique | squad | hotpotqa | webqsp |
|---|---|---|---|---|---|---|
| `N` | 40,151 | 65,865 | 13,672 | 19,029 | 507,494 | 781,485 |
| `|E_STRUCT|` | 109,480 | 126,070 | 51,543 | 694,580 | 3,426,945 | 1,638,066 |
| `|E_NERX|` | 232,505 | 494,304 | 93,500 | 126,488 | 4,058,707 | 975,995 |
| `|E_KNN|` | 55,207 | 134,737 | 27,406 | 28,276 | 1,166,652 | 1,672,731 |
| `|S n N|` | 0 | 0 | 0 | 0 | 0 | 0 |
| `|S n K|` | 0 | 0 | 0 | 0 | 0 | 0 |
| `|N n K|` | 9,404 | 26,422 | 8,726 | 7,993 | 183,966 | 132,216 |
| `|S u N|` | 341,985 | 620,374 | 145,043 | 821,068 | 7,485,652 | 2,614,061 |
| `|S u K|` | 164,687 | 260,807 | 78,949 | 722,856 | 4,593,597 | 3,310,797 |
| `|S u N u K|` | 387,788 | 728,689 | 163,723 | 841,351 | 8,468,338 | 4,154,576 |
| `|topology_C_file|` | 387,788 | 728,689 | 163,723 | 841,351 | 8,440,280 | 4,154,576 |
| `SuNuK_equals_C_file` | YES | YES | YES | YES | NO | YES |
| `frac_C_never_traversed` | 0.7177 | 0.8270 | 0.6852 | 0.1744 | 0.5953 | 0.6057 |

`|S n N| = |S n K| = 0` holds by construction on all six.  `|N n K| != 0` on all six, so family provenance is a **bitmask**, never a partition of the edges.

## A1 / A2 -- the canonical substrates

| substrate (undirected edges) | metaqa | 2wiki | musique | squad | hotpotqa | webqsp |
|---|---|---|---|---|---|---|
| `E0_STRUCT` | 109,480 | 126,070 | 51,543 | 694,580 | 3,426,945 | 1,638,066 |
| `E1_NERX_ONLY` | 232,505 | 494,304 | 93,500 | 126,488 | 4,058,707 | 975,995 |
| `E2_KNN_ONLY` | 55,207 | 134,737 | 27,406 | 28,276 | 1,166,652 | 1,672,731 |
| `E3_STRUCT_NERX` | 341,985 | 620,374 | 145,043 | 821,068 | 7,485,652 | 2,614,061 |
| `E4_STRUCT_KNN` | 164,687 | 260,807 | 78,949 | 722,856 | 4,593,597 | 3,310,797 |
| `E5_NERX_KNN` | 278,308 | 602,619 | 112,180 | 146,771 | 5,041,393 | 2,516,510 |
| `E6_TOPOLOGY_C` | 387,788 | 728,689 | 163,723 | 841,351 | 8,468,338 | 4,154,576 |
| `M0_STRUCT` | 109,480 | 126,070 | 51,543 | 694,580 | 3,426,945 | 1,638,066 |
| `M1_STRUCT_NERX_MATCHED` | 109,480 | 126,070 | 51,543 | 694,580 | 3,426,945 | 1,638,066 |
| `M2_STRUCT_KNN_MATCHED` | 109,480 | 126,070 | 51,543 | 694,580 | 3,426,945 | 1,638,066 |
| `M3_TOPOLOGY_C_MATCHED` | 109,480 | 126,070 | 51,543 | 694,580 | 3,426,945 | 1,638,066 |

The `M*` matched controls carry EXACTLY `E0_STRUCT`'s `adj_ptr` and `deg`, so DEG_CAP behaves identically and every frontier node costs the same number of edge inspections -- matched by construction, not by tuning.

## A3 -- parity gate (hard gate, every query, both passes)

| corpus | nq | pass1 E0 parity | spos exact | P50 exact | pass2 E0 parity | pass1==pass2 (p50/orac) |
|---|---|---|---|---|---|---|
| metaqa | 1998 | EXACT | 1998/1998 | 1998/1998 | EXACT | YES/YES |
| 2wiki_clean | 2000 | EXACT | 2000/2000 | 2000/2000 | EXACT | YES/YES |
| musique_clean | 2000 | EXACT | 2000/2000 | 2000/2000 | EXACT | YES/YES |
| squad_clean | 2000 | EXACT | 2000/2000 | 2000/2000 | EXACT | YES/YES |
| hotpotqa_clean | 2000 | EXACT | 2000/2000 | 2000/2000 | EXACT | YES/YES |
| webqsp | 1419 | EXACT | 1419/1419 | 1419/1419 | EXACT | YES/YES |

`CSR(E_STRUCT keys) == frozen master_nodes CSR` is asserted bit-exact inside the substrate builder before any run starts; `spos` and the exact P50 set are then re-verified per query.

## A4 -- needed partitions the frozen substrate MISSES, by family bitmask (VIS level)

| corpus | missed needed | S only | S and K | K only (not S) | N only (not S) | K and N (not S) | none | UNIQUE_KNN | UNIQUE_NERX | unreached by any |
|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | 5,815 | 4,039 | 1,291 | 15 | 272 | 56 | 142 | 0.0026 | 0.0468 | 0.0244 |
| 2wiki_clean | 117 | 22 | 23 | 8 | 22 | 31 | 11 | 0.0684 | 0.1880 | 0.0940 |
| musique_clean | 75 | 25 | 39 | 0 | 6 | 4 | 1 | 0.0000 | 0.0800 | 0.0133 |
| squad_clean | 25 | 8 | 7 | 0 | 7 | 3 | 0 | 0.0000 | 0.2800 | 0.0000 |
| hotpotqa_clean | 104 | 25 | 27 | 7 | 12 | 8 | 25 | 0.0673 | 0.1154 | 0.2404 |
| webqsp | 1,685 | 505 | 69 | 97 | 28 | 4 | 982 | 0.0576 | 0.0166 | 0.5828 |


### A4 -- same, READ level (what the M_struct=64 aggregation actually sees)

| corpus | missed needed | UNIQUE_KNN | UNIQUE_NERX | UNIQUE_K or N | unreached by any |
|---|---|---|---|---|---|
| metaqa | 5,815 | 0.0831 | 0.0841 | 0.1826 | 0.5859 |
| 2wiki_clean | 117 | 0.2479 | 0.1197 | 0.4017 | 0.5128 |
| musique_clean | 75 | 0.2000 | 0.1733 | 0.5067 | 0.3733 |
| squad_clean | 25 | 0.2400 | 0.3600 | 0.6400 | 0.2800 |
| hotpotqa_clean | 104 | 0.1923 | 0.0865 | 0.3173 | 0.6250 |
| webqsp | 1,685 | 0.0451 | 0.0178 | 0.0653 | 0.9015 |

## A5 -- incremental attribution and interaction (full work)

**needed-partition reach, VIS**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.9409 | 0.9716 | 0.9662 | 0.9777 | +0.0307 | +0.0253 | +0.0061 | +0.0115 | -0.0192 |
| 2wiki_clean | 0.9548 | 0.9866 | 0.9742 | 0.9894 | +0.0318 | +0.0194 | +0.0028 | +0.0152 | -0.0167 |
| musique_clean | 0.9896 | 1.0000 | 0.9986 | 1.0000 | +0.0104 | +0.0090 | +0.0000 | +0.0014 | -0.0090 |
| squad_clean | 0.9785 | 0.9980 | 0.9965 | 0.9990 | +0.0195 | +0.0180 | +0.0010 | +0.0025 | -0.0170 |
| hotpotqa_clean | 0.9694 | 0.9802 | 0.9811 | 0.9856 | +0.0108 | +0.0117 | +0.0054 | +0.0045 | -0.0063 |
| webqsp | 0.6353 | 0.6538 | 0.7180 | 0.7255 | +0.0185 | +0.0827 | +0.0717 | +0.0075 | -0.0110 |

**needed-partition reach, READ**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.3587 | 0.3334 | 0.3445 | 0.3291 | -0.0253 | -0.0142 | -0.0043 | -0.0154 | +0.0099 |
| 2wiki_clean | 0.3621 | 0.4758 | 0.5583 | 0.5235 | +0.1136 | +0.1962 | +0.0477 | -0.0348 | -0.1485 |
| musique_clean | 0.5969 | 0.7332 | 0.7378 | 0.7792 | +0.1363 | +0.1409 | +0.0459 | +0.0414 | -0.0949 |
| squad_clean | 0.5115 | 0.7285 | 0.7005 | 0.7700 | +0.2170 | +0.1890 | +0.0415 | +0.0695 | -0.1475 |
| hotpotqa_clean | 0.5379 | 0.6028 | 0.5931 | 0.6316 | +0.0649 | +0.0553 | +0.0289 | +0.0385 | -0.0264 |
| webqsp | 0.3170 | 0.3001 | 0.3703 | 0.3510 | -0.0169 | +0.0533 | +0.0510 | -0.0192 | -0.0023 |

**candidate-pool oracle**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.7212 | 0.7132 | 0.7202 | 0.7042 | -0.0080 | -0.0010 | -0.0090 | -0.0160 | -0.0080 |
| 2wiki_clean | 0.9515 | 0.9590 | 0.9590 | 0.9585 | +0.0075 | +0.0075 | -0.0005 | -0.0005 | -0.0080 |
| musique_clean | 0.9750 | 0.9765 | 0.9755 | 0.9760 | +0.0015 | +0.0005 | -0.0005 | +0.0005 | -0.0010 |
| squad_clean | 0.9945 | 0.9945 | 0.9945 | 0.9945 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| hotpotqa_clean | 0.9645 | 0.9695 | 0.9680 | 0.9690 | +0.0050 | +0.0035 | -0.0005 | +0.0010 | -0.0040 |
| webqsp | 0.7886 | 0.7872 | 0.7865 | 0.7872 | -0.0014 | -0.0021 | +0.0000 | +0.0007 | +0.0021 |

**exact P50 ALL@50**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.6627 | 0.6617 | 0.6597 | +0.0015 | +0.0005 | -0.0030 | -0.0020 | -0.0035 |
| 2wiki_clean | 0.9435 | 0.9445 | 0.9450 | 0.9465 | +0.0010 | +0.0015 | +0.0020 | +0.0015 | +0.0005 |
| musique_clean | 0.9635 | 0.9650 | 0.9640 | 0.9655 | +0.0015 | +0.0005 | +0.0005 | +0.0015 | +0.0000 |
| squad_clean | 0.9875 | 0.9880 | 0.9875 | 0.9870 | +0.0005 | +0.0000 | -0.0010 | -0.0005 | -0.0010 |
| hotpotqa_clean | 0.9505 | 0.9470 | 0.9510 | 0.9505 | -0.0035 | +0.0005 | +0.0035 | -0.0005 | +0.0030 |
| webqsp | 0.7646 | 0.7689 | 0.7681 | 0.7681 | +0.0042 | +0.0035 | -0.0007 | +0.0000 | -0.0042 |

## A5 -- incremental attribution and interaction (matched work)

**needed-partition reach, VIS**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.9409 | 0.9164 | 0.9373 | 0.9264 | -0.0245 | -0.0036 | +0.0100 | -0.0108 | +0.0137 |
| 2wiki_clean | 0.9548 | 0.9538 | 0.9571 | 0.9538 | -0.0010 | +0.0023 | +0.0000 | -0.0033 | -0.0023 |
| musique_clean | 0.9896 | 0.9944 | 0.9918 | 0.9955 | +0.0048 | +0.0022 | +0.0011 | +0.0037 | -0.0011 |
| squad_clean | 0.9785 | 0.9930 | 0.9895 | 0.9945 | +0.0145 | +0.0110 | +0.0015 | +0.0050 | -0.0095 |
| hotpotqa_clean | 0.9694 | 0.9594 | 0.9657 | 0.9594 | -0.0099 | -0.0036 | +0.0000 | -0.0063 | +0.0036 |
| webqsp | 0.6353 | 0.6165 | 0.6230 | 0.6128 | -0.0188 | -0.0123 | -0.0037 | -0.0102 | +0.0087 |

**needed-partition reach, READ**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.3587 | 0.3207 | 0.3210 | 0.3177 | -0.0380 | -0.0377 | -0.0030 | -0.0033 | +0.0347 |
| 2wiki_clean | 0.3621 | 0.4311 | 0.4194 | 0.4414 | +0.0689 | +0.0573 | +0.0103 | +0.0220 | -0.0470 |
| musique_clean | 0.5969 | 0.6273 | 0.6276 | 0.6330 | +0.0304 | +0.0307 | +0.0056 | +0.0053 | -0.0251 |
| squad_clean | 0.5115 | 0.6455 | 0.6225 | 0.6765 | +0.1340 | +0.1110 | +0.0310 | +0.0540 | -0.0800 |
| hotpotqa_clean | 0.5379 | 0.5706 | 0.5529 | 0.5592 | +0.0328 | +0.0150 | -0.0114 | +0.0063 | -0.0264 |
| webqsp | 0.3170 | 0.3064 | 0.3239 | 0.3168 | -0.0106 | +0.0069 | +0.0104 | -0.0071 | +0.0035 |

**candidate-pool oracle**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.7212 | 0.7037 | 0.7062 | 0.7012 | -0.0175 | -0.0150 | -0.0025 | -0.0050 | +0.0125 |
| 2wiki_clean | 0.9515 | 0.9550 | 0.9510 | 0.9545 | +0.0035 | -0.0005 | -0.0005 | +0.0035 | +0.0000 |
| musique_clean | 0.9750 | 0.9750 | 0.9740 | 0.9760 | +0.0000 | -0.0010 | +0.0010 | +0.0020 | +0.0020 |
| squad_clean | 0.9945 | 0.9935 | 0.9945 | 0.9935 | -0.0010 | +0.0000 | +0.0000 | -0.0010 | +0.0000 |
| hotpotqa_clean | 0.9645 | 0.9650 | 0.9660 | 0.9660 | +0.0005 | +0.0015 | +0.0010 | +0.0000 | -0.0005 |
| webqsp | 0.7886 | 0.7907 | 0.7886 | 0.7900 | +0.0021 | +0.0000 | -0.0007 | +0.0014 | -0.0007 |

**exact P50 ALL@50**

| corpus | M(S) | M(SuN) | M(SuK) | M(SuNuK) | DELTA_N | DELTA_K | DELTA_K|N | DELTA_N|K | INTERACTION |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.6602 | 0.6602 | 0.6572 | -0.0010 | -0.0010 | -0.0030 | -0.0030 | -0.0020 |
| 2wiki_clean | 0.9435 | 0.9455 | 0.9435 | 0.9445 | +0.0020 | +0.0000 | -0.0010 | +0.0010 | -0.0010 |
| musique_clean | 0.9635 | 0.9620 | 0.9620 | 0.9610 | -0.0015 | -0.0015 | -0.0010 | -0.0010 | +0.0005 |
| squad_clean | 0.9875 | 0.9880 | 0.9880 | 0.9870 | +0.0005 | +0.0005 | -0.0010 | -0.0010 | -0.0015 |
| hotpotqa_clean | 0.9505 | 0.9505 | 0.9520 | 0.9520 | +0.0000 | +0.0015 | +0.0015 | +0.0000 | +0.0000 |
| webqsp | 0.7646 | 0.7653 | 0.7660 | 0.7653 | +0.0007 | +0.0014 | +0.0000 | -0.0007 | -0.0014 |

## A6 -- is the family Dense wearing a graph costume?

For partitions a family reaches, `EDGE_SIM` is the neighbour similarity the family itself asserts and `QUERY_SIM` is what Dense already knows. AUC separates needed from nuisance.

| corpus | KNN EDGE_SIM AUC | KNN QUERY_SIM AUC | NERX EDGE_SIM AUC | NERX QUERY_SIM AUC |
|---|---|---|---|---|
| metaqa | 0.6028 | 0.7167 | 0.6469 | 0.7497 |
| 2wiki_clean | 0.7550 | 0.9413 | 0.8579 | 0.9302 |
| musique_clean | 0.8037 | 0.9424 | 0.8371 | 0.9312 |
| squad_clean | 0.8198 | 0.9319 | 0.8509 | 0.9287 |
| hotpotqa_clean | 0.8338 | 0.9740 | 0.9267 | 0.9736 |
| webqsp | 0.7931 | 0.8625 | 0.8023 | 0.9006 |

## A7 -- saturation (no oracle gets credit for exposure)

| corpus | substrate | visited parts/q | frac of corpus | read parts/q | cand pool/q | candidate oracle |
|---|---|---|---|---|---|---|
| metaqa | E0_STRUCT | 305.5 | 0.7619 | 47.5 | 49.2 | 0.7212 |
| metaqa | E1_NERX_ONLY | 288.6 | 0.7197 | 38.6 | 43.0 | 0.6832 |
| metaqa | E2_KNN_ONLY | 72.9 | 0.1818 | 44.2 | 48.0 | 0.6712 |
| metaqa | E3_STRUCT_NERX | 361.8 | 0.9023 | 42.6 | 44.5 | 0.7132 |
| metaqa | E4_STRUCT_KNN | 345.6 | 0.8618 | 48.1 | 48.8 | 0.7202 |
| metaqa | E6_TOPOLOGY_C | 370.0 | 0.9227 | 43.6 | 44.9 | 0.7042 |
| metaqa | M1_STRUCT_NERX_MATCHED | 310.4 | 0.7739 | 46.4 | 48.6 | 0.7037 |
| metaqa | M2_STRUCT_KNN_MATCHED | 324.9 | 0.8102 | 48.9 | 50.5 | 0.7062 |
| metaqa | M3_TOPOLOGY_C_MATCHED | 320.3 | 0.7988 | 47.5 | 49.4 | 0.7012 |
| 2wiki_clean | E0_STRUCT | 168.0 | 0.2554 | 37.7 | 41.8 | 0.9515 |
| 2wiki_clean | E1_NERX_ONLY | 397.9 | 0.6047 | 38.0 | 41.0 | 0.9565 |
| 2wiki_clean | E2_KNN_ONLY | 93.6 | 0.1423 | 40.3 | 39.8 | 0.9630 |
| 2wiki_clean | E3_STRUCT_NERX | 432.9 | 0.6579 | 38.7 | 40.9 | 0.9590 |
| 2wiki_clean | E4_STRUCT_KNN | 266.1 | 0.4044 | 43.5 | 43.2 | 0.9590 |
| 2wiki_clean | E6_TOPOLOGY_C | 450.8 | 0.6852 | 39.7 | 40.5 | 0.9585 |
| 2wiki_clean | M1_STRUCT_NERX_MATCHED | 187.1 | 0.2843 | 40.8 | 44.1 | 0.9550 |
| 2wiki_clean | M2_STRUCT_KNN_MATCHED | 189.2 | 0.2875 | 40.4 | 43.0 | 0.9510 |
| 2wiki_clean | M3_TOPOLOGY_C_MATCHED | 194.1 | 0.2950 | 41.1 | 44.1 | 0.9545 |
| musique_clean | E0_STRUCT | 111.9 | 0.8228 | 28.2 | 18.6 | 0.9750 |
| musique_clean | E1_NERX_ONLY | 123.6 | 0.9086 | 30.0 | 19.7 | 0.9790 |
| musique_clean | E2_KNN_ONLY | 48.1 | 0.3540 | 29.5 | 18.1 | 0.9775 |
| musique_clean | E3_STRUCT_NERX | 131.5 | 0.9667 | 28.9 | 17.8 | 0.9765 |
| musique_clean | E4_STRUCT_KNN | 123.9 | 0.9107 | 29.9 | 18.2 | 0.9755 |
| musique_clean | E6_TOPOLOGY_C | 132.7 | 0.9756 | 29.6 | 17.6 | 0.9760 |
| musique_clean | M1_STRUCT_NERX_MATCHED | 116.3 | 0.8551 | 29.0 | 19.3 | 0.9750 |
| musique_clean | M2_STRUCT_KNN_MATCHED | 114.2 | 0.8397 | 28.4 | 18.4 | 0.9740 |
| musique_clean | M3_TOPOLOGY_C_MATCHED | 117.6 | 0.8646 | 29.5 | 19.4 | 0.9760 |
| squad_clean | E0_STRUCT | 117.6 | 0.6191 | 16.5 | 15.8 | 0.9945 |
| squad_clean | E1_NERX_ONLY | 162.6 | 0.8558 | 35.3 | 29.3 | 0.9960 |
| squad_clean | E2_KNN_ONLY | 30.3 | 0.1594 | 23.5 | 18.7 | 0.9950 |
| squad_clean | E3_STRUCT_NERX | 177.9 | 0.9363 | 25.6 | 20.5 | 0.9945 |
| squad_clean | E4_STRUCT_KNN | 142.2 | 0.7486 | 21.5 | 17.5 | 0.9945 |
| squad_clean | E6_TOPOLOGY_C | 180.0 | 0.9473 | 26.6 | 20.5 | 0.9945 |
| squad_clean | M1_STRUCT_NERX_MATCHED | 163.6 | 0.8610 | 23.5 | 19.7 | 0.9935 |
| squad_clean | M2_STRUCT_KNN_MATCHED | 134.6 | 0.7082 | 19.7 | 17.0 | 0.9945 |
| squad_clean | M3_TOPOLOGY_C_MATCHED | 166.0 | 0.8734 | 24.3 | 19.9 | 0.9935 |
| hotpotqa_clean | E0_STRUCT | 585.1 | 0.1153 | 45.5 | 55.4 | 0.9645 |
| hotpotqa_clean | E1_NERX_ONLY | 710.2 | 0.1400 | 37.4 | 49.2 | 0.9675 |
| hotpotqa_clean | E2_KNN_ONLY | 99.7 | 0.0197 | 42.2 | 50.6 | 0.9705 |
| hotpotqa_clean | E3_STRUCT_NERX | 1021.4 | 0.2013 | 41.0 | 50.2 | 0.9695 |
| hotpotqa_clean | E4_STRUCT_KNN | 690.8 | 0.1362 | 45.9 | 54.5 | 0.9680 |
| hotpotqa_clean | E6_TOPOLOGY_C | 1083.9 | 0.2136 | 41.6 | 49.9 | 0.9690 |
| hotpotqa_clean | M1_STRUCT_NERX_MATCHED | 556.1 | 0.1096 | 44.2 | 54.1 | 0.9650 |
| hotpotqa_clean | M2_STRUCT_KNN_MATCHED | 575.2 | 0.1134 | 45.9 | 55.4 | 0.9660 |
| hotpotqa_clean | M3_TOPOLOGY_C_MATCHED | 571.6 | 0.1126 | 44.9 | 54.7 | 0.9660 |
| webqsp | E0_STRUCT | 417.3 | 0.0534 | 36.6 | 34.4 | 0.7886 |
| webqsp | E1_NERX_ONLY | 60.0 | 0.0077 | 15.5 | 20.5 | 0.7703 |
| webqsp | E2_KNN_ONLY | 40.0 | 0.0051 | 23.1 | 23.5 | 0.7759 |
| webqsp | E3_STRUCT_NERX | 498.7 | 0.0638 | 33.5 | 32.2 | 0.7872 |
| webqsp | E4_STRUCT_KNN | 470.7 | 0.0602 | 36.0 | 32.4 | 0.7865 |
| webqsp | E6_TOPOLOGY_C | 523.5 | 0.0670 | 33.1 | 30.2 | 0.7872 |
| webqsp | M1_STRUCT_NERX_MATCHED | 409.6 | 0.0524 | 36.0 | 35.0 | 0.7907 |
| webqsp | M2_STRUCT_KNN_MATCHED | 398.2 | 0.0510 | 35.8 | 34.1 | 0.7886 |
| webqsp | M3_TOPOLOGY_C_MATCHED | 404.7 | 0.0518 | 36.1 | 35.2 | 0.7900 |

## A11 -- EXACT P=50 (ALL@50), the only number that decides anything

| corpus | FROZEN STRUCT | +NERX | +KNN | +NERX+KNN | MATCHED +NERX | MATCHED +KNN | MATCHED FULL C |
|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.6627 (+0.0015) | 0.6617 (+0.0005) | 0.6597 (-0.0015) | 0.6602 (-0.0010) | 0.6602 (-0.0010) | 0.6572 (-0.0040) |
| 2wiki_clean | 0.9435 | 0.9445 (+0.0010) | 0.9450 (+0.0015) | 0.9465 (+0.0030 SIG) | 0.9455 (+0.0020) | 0.9435 (+0.0000) | 0.9445 (+0.0010) |
| musique_clean | 0.9635 | 0.9650 (+0.0015) | 0.9640 (+0.0005) | 0.9655 (+0.0020) | 0.9620 (-0.0015) | 0.9620 (-0.0015) | 0.9610 (-0.0025) |
| squad_clean | 0.9875 | 0.9880 (+0.0005) | 0.9875 (+0.0000) | 0.9870 (-0.0005) | 0.9880 (+0.0005) | 0.9880 (+0.0005) | 0.9870 (-0.0005) |
| hotpotqa_clean | 0.9505 | 0.9470 (-0.0035) | 0.9510 (+0.0005) | 0.9505 (+0.0000) | 0.9505 (+0.0000) | 0.9520 (+0.0015) | 0.9520 (+0.0015) |
| webqsp | 0.7646 | 0.7689 (+0.0042) | 0.7681 (+0.0035) | 0.7681 (+0.0035) | 0.7653 (+0.0007) | 0.7660 (+0.0014) | 0.7653 (+0.0007) |

Significance is McNemar on the paired per-query ALL@50 indicator against the frozen substrate on the same queries.

### A11 -- MetaQA by hop

| hop | FROZEN STRUCT | +NERX | +KNN | +NERX+KNN | MATCHED +NERX | MATCHED +KNN | MATCHED FULL C |
|---|---|---|---|---|---|---|---|
| hop1 | 0.9955 | 0.9955 (+0.0000) | 0.9955 (+0.0000) | 0.9955 (+0.0000) | 0.9955 (+0.0000) | 0.9955 (+0.0000) | 0.9955 (+0.0000) |
| hop2 | 0.7297 | 0.7297 (+0.0000) | 0.7312 (+0.0015) | 0.7207 (-0.0090) | 0.7267 (-0.0030) | 0.7282 (-0.0015) | 0.7252 (-0.0045) |
| hop3 | 0.2583 | 0.2628 (+0.0045) | 0.2583 (+0.0000) | 0.2628 (+0.0045) | 0.2583 (+0.0000) | 0.2568 (-0.0015) | 0.2508 (-0.0075) |

## A12 -- cost

| corpus | substrate | edges/query | vs frozen | ms/query | vs frozen |
|---|---|---|---|---|---|
| metaqa | E0_STRUCT | 1,506 | 1.00x | 41.6 | 1.00x |
| metaqa | E1_NERX_ONLY | 3,497 | 2.32x | 63.1 | 1.51x |
| metaqa | E2_KNN_ONLY | 177 | 0.12x | 11.8 | 0.28x |
| metaqa | E3_STRUCT_NERX | 5,236 | 3.48x | 39.9 | 0.96x |
| metaqa | E4_STRUCT_KNN | 2,126 | 1.41x | 54.1 | 1.30x |
| metaqa | E6_TOPOLOGY_C | 5,434 | 3.61x | 106.1 | 2.55x |
| metaqa | M1_STRUCT_NERX_MATCHED | 1,685 | 1.12x | 17.0 | 0.41x |
| metaqa | M2_STRUCT_KNN_MATCHED | 1,717 | 1.14x | 47.9 | 1.15x |
| metaqa | M3_TOPOLOGY_C_MATCHED | 1,843 | 1.22x | 18.3 | 0.44x |
| 2wiki_clean | E0_STRUCT | 601 | 1.00x | 21.7 | 1.00x |
| 2wiki_clean | E1_NERX_ONLY | 4,476 | 7.45x | 83.1 | 3.83x |
| 2wiki_clean | E2_KNN_ONLY | 357 | 0.59x | 18.9 | 0.87x |
| 2wiki_clean | E3_STRUCT_NERX | 5,005 | 8.33x | 31.2 | 1.44x |
| 2wiki_clean | E4_STRUCT_KNN | 1,165 | 1.94x | 34.0 | 1.57x |
| 2wiki_clean | E6_TOPOLOGY_C | 5,131 | 8.54x | 93.8 | 4.33x |
| 2wiki_clean | M1_STRUCT_NERX_MATCHED | 624 | 1.04x | 7.9 | 0.36x |
| 2wiki_clean | M2_STRUCT_KNN_MATCHED | 625 | 1.04x | 22.6 | 1.04x |
| 2wiki_clean | M3_TOPOLOGY_C_MATCHED | 640 | 1.06x | 7.2 | 0.33x |
| musique_clean | E0_STRUCT | 3,196 | 1.00x | 66.5 | 1.00x |
| musique_clean | E1_NERX_ONLY | 3,494 | 1.09x | 63.2 | 0.95x |
| musique_clean | E2_KNN_ONLY | 274 | 0.09x | 15.9 | 0.24x |
| musique_clean | E3_STRUCT_NERX | 5,815 | 1.82x | 66.7 | 1.00x |
| musique_clean | E4_STRUCT_KNN | 3,705 | 1.16x | 81.2 | 1.22x |
| musique_clean | E6_TOPOLOGY_C | 5,927 | 1.85x | 110.5 | 1.66x |
| musique_clean | M1_STRUCT_NERX_MATCHED | 2,538 | 0.79x | 33.3 | 0.50x |
| musique_clean | M2_STRUCT_KNN_MATCHED | 2,896 | 0.91x | 60.7 | 0.91x |
| musique_clean | M3_TOPOLOGY_C_MATCHED | 2,477 | 0.78x | 32.0 | 0.48x |
| squad_clean | E0_STRUCT | 14,181 | 1.00x | 237.1 | 1.00x |
| squad_clean | E1_NERX_ONLY | 3,273 | 0.23x | 54.0 | 0.23x |
| squad_clean | E2_KNN_ONLY | 158 | 0.01x | 9.9 | 0.04x |
| squad_clean | E3_STRUCT_NERX | 14,309 | 1.01x | 155.6 | 0.66x |
| squad_clean | E4_STRUCT_KNN | 14,487 | 1.02x | 236.0 | 0.99x |
| squad_clean | E6_TOPOLOGY_C | 14,082 | 0.99x | 212.4 | 0.90x |
| squad_clean | M1_STRUCT_NERX_MATCHED | 12,530 | 0.88x | 145.5 | 0.61x |
| squad_clean | M2_STRUCT_KNN_MATCHED | 13,980 | 0.99x | 228.2 | 0.96x |
| squad_clean | M3_TOPOLOGY_C_MATCHED | 12,296 | 0.87x | 142.9 | 0.60x |
| hotpotqa_clean | E0_STRUCT | 1,893 | 1.00x | 41.5 | 1.00x |
| hotpotqa_clean | E1_NERX_ONLY | 4,039 | 2.13x | 73.2 | 1.76x |
| hotpotqa_clean | E2_KNN_ONLY | 314 | 0.17x | 13.4 | 0.32x |
| hotpotqa_clean | E3_STRUCT_NERX | 5,487 | 2.90x | 45.7 | 1.10x |
| hotpotqa_clean | E4_STRUCT_KNN | 2,445 | 1.29x | 48.8 | 1.18x |
| hotpotqa_clean | E6_TOPOLOGY_C | 5,761 | 3.04x | 97.9 | 2.36x |
| hotpotqa_clean | M1_STRUCT_NERX_MATCHED | 1,810 | 0.96x | 20.1 | 0.49x |
| hotpotqa_clean | M2_STRUCT_KNN_MATCHED | 1,805 | 0.95x | 41.0 | 0.99x |
| hotpotqa_clean | M3_TOPOLOGY_C_MATCHED | 1,824 | 0.96x | 19.9 | 0.48x |
| webqsp | E0_STRUCT | 1,662 | 1.00x | 40.9 | 1.00x |
| webqsp | E1_NERX_ONLY | 1,092 | 0.66x | 23.0 | 0.56x |
| webqsp | E2_KNN_ONLY | 245 | 0.15x | 10.7 | 0.26x |
| webqsp | E3_STRUCT_NERX | 2,857 | 1.72x | 30.8 | 0.75x |
| webqsp | E4_STRUCT_KNN | 2,068 | 1.25x | 44.5 | 1.09x |
| webqsp | E6_TOPOLOGY_C | 3,059 | 1.84x | 61.8 | 1.51x |
| webqsp | M1_STRUCT_NERX_MATCHED | 1,724 | 1.04x | 20.4 | 0.50x |
| webqsp | M2_STRUCT_KNN_MATCHED | 1,610 | 0.97x | 36.7 | 0.90x |
| webqsp | M3_TOPOLOGY_C_MATCHED | 1,694 | 1.02x | 20.8 | 0.51x |

## B0 -- how the production partitions were actually built (read from source)

| item | value |
|---|---|
| build script | `scratchpad/build_canonical_topo.py` |
| graph handed to METIS | `C = A_QWEN u NER = (STRUCT u KNN) u NER` -- **full topology C** |
| partitioning happens | AFTER full topology-C construction (not before, not per-family) |
| directed / symmetrised / dedup | undirected, symmetrised, deduplicated (python `set`), self-loops excluded |
| vertex weights | NONE (unweighted) |
| edge weights | NONE (unweighted -- the NER 1/df weights are DISCARDED at this step) |
| library | `pymetis.part_graph(n_parts, adjacency=...)` (METIS 5.x) |
| recursive vs k-way | k-way (pymetis uses recursive only for k <= 8; k is 136..7814) |
| objective | METIS default `OBJTYPE_CUT` (edge cut) |
| balance tolerance | METIS default `ufactor=30` for k-way, i.e. 1.03 |
| seed | METIS default (`seed=-1`); no seed was pinned by the build |
| k | `max(1, N // 100)` |
| target size | 100 documents per partition |

| corpus | N | k | min | max | mean | median | CV | max/mean | empty |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 40,151 | 401 | 96 | 103 | 100.13 | 100.0 | 0.0216 | 1.0287 | 0 |
| 2wiki_clean | 65,865 | 658 | 97 | 103 | 100.1 | 100.0 | 0.0200 | 1.0290 | 0 |
| musique_clean | 13,672 | 136 | 96 | 103 | 100.53 | 101.0 | 0.0235 | 1.0246 | 0 |
| squad_clean | 19,029 | 190 | 82 | 103 | 100.15 | 102.0 | 0.0313 | 1.0284 | 0 |
| hotpotqa_clean | 507,494 | 5,074 | 71 | 103 | 100.02 | 100.0 | 0.0229 | 1.0298 | 0 |
| webqsp | 781,485 | 7,814 | 86 | 103 | 100.01 | 100.0 | 0.0203 | 1.0299 | 0 |

Measured imbalance is 1.025-1.030 on every corpus, exactly the METIS `ufactor=30` guarantee -- independent confirmation of the configuration read from source.

## B1-B8 -- partition utility of the SHIPPED assignment

| metric | metaqa | 2wiki | musique | squad | hotpotqa | webqsp |
|---|---|---|---|---|---|---|
| B1 gold compression | 0.9003 | 0.8317 | 0.7593 | 1.0000 | 0.8320 | 0.8997 |
| B1 fetch saving | 0.0997 | 0.1683 | 0.2407 | 0.0000 | 0.1680 | 0.1003 |
| B2 pair coloc | 0.0374 | 0.2147 | 0.3491 | N/A | 0.3360 | 0.2630 |
| B2 expected (random) | 0.0025 | 0.0015 | 0.0073 | N/A | 0.0002 | 0.0001 |
| B2 adjusted | 0.0350 | 0.2135 | 0.3443 | N/A | 0.3359 | 0.2630 |
| B3 all-gold in 1 block | 0.4620 | 0.2365 | 0.3725 | 1.0000 | 0.3360 | 0.5208 |
| B3 block density | 0.6354 | 0.6054 | 0.6879 | 1.0000 | 0.6680 | 0.7065 |
| B4 seed->required | 0.3252 | 0.8245 | 0.8405 | 0.9195 | 0.8410 | 0.3173 |
| B4 nontrivial | -- | 0.2037 | -- | -- | 0.4015 | -- |
| B5 gold-edge containment | 0.4467 | 0.5132 | 0.6756 | 0.0000 | 0.4571 | 0.7662 |
| B8 oracle ALL@50 | 0.9870 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9951 |
| B7 min partitions (mean) | 5.784 | 1.980 | 1.775 | 1.000 | 1.664 | 3.664 |
| B7 min partitions (max) | 104 | 4 | 4 | 1 | 2 | 69 |

### B1-B8 -- MetaQA by hop (the primary multi-hop case)

| metric | hop1 | hop2 | hop3 |
|---|---|---|---|
| B1_GOLD_COMPRESSION | 0.9336 | 0.8677 | 0.8997 |
| B2_PAIR_COLOCATION | 0.1893 | 0.1170 | 0.0159 |
| B3_ALL_GOLD_CONTAINED | 0.7177 | 0.4865 | 0.1817 |
| B4_SEED_REQUIRED_COLOCATION | 0.5649 | 0.2893 | 0.1214 |
| B4_SEED_REQUIRED_COLOCATION_NONTRIVIAL | -- | -- | -- |
| B5_GOLD_EDGE_CONTAINMENT | 0.4426 | 0.5405 | 0.2762 |
| B7_MIN_PARTITIONS_REQUIRED_mean | 1.6081 | 3.9895 | 11.7538 |
| B8_ORACLE_ALL_AT_50 | 1.0000 | 0.9970 | 0.9640 |

## B8 -- full-universe oracle ALL@P (the ceiling the ASSIGNMENT imposes, no router)

| corpus | P=1 | P=2 | P=3 | P=5 | P=10 | P=20 | P=50 | P=100 |
|---|---|---|---|---|---|---|---|---|
| metaqa | 0.4620 | 0.6091 | 0.6917 | 0.7688 | 0.8604 | 0.9234 | 0.9870 | 0.9995 |
| 2wiki_clean | 0.2365 | 0.8350 | 0.9485 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| musique_clean | 0.3725 | 0.8625 | 0.9900 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| squad_clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hotpotqa_clean | 0.3360 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| webqsp | 0.5208 | 0.6716 | 0.7717 | 0.8739 | 0.9316 | 0.9669 | 0.9951 | 1.0000 |

## B9-B15 -- partitioner leaderboard (utility metrics, all built without gold/queries)

**metaqa**

| partitioner | B1 compress | B2 adj coloc | B3 all-in-1 | B5 edge contain | B8 oracle@50 | B7 min parts | B12 cut (C) | B12 modularity | balance | B14 top-50 nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_METIS_CURRENT | 0.9003 | 0.0350 | 0.4620 | 0.4467 | 0.9870 | 5.784 | 228,941 | 0.4066 | 1.029 | 5,150 |
| PM0_STRUCT | 0.8953 | 0.0330 | 0.4595 | 0.2192 | 0.9890 | 5.665 | 290,048 | 0.2493 | 1.029 | 5,139 |
| PM1_STRUCT_NERX | 0.8984 | 0.0363 | 0.4600 | 0.4359 | 0.9875 | 5.760 | 232,609 | 0.3974 | 1.029 | 5,150 |
| PM2_STRUCT_KNN | 0.9172 | 0.0241 | 0.4439 | 0.1924 | 0.9865 | 5.925 | 296,788 | 0.2318 | 1.029 | 5,131 |
| PM3_TOPOLOGY_C | 0.9018 | 0.0356 | 0.4580 | 0.4495 | 0.9860 | 5.796 | 228,986 | 0.4065 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed1 | 0.9044 | 0.0333 | 0.4570 | 0.4321 | 0.9860 | 5.875 | 228,391 | 0.4080 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed2 | 0.9027 | 0.0352 | 0.4580 | 0.4486 | 0.9865 | 5.814 | 228,764 | 0.4071 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed3 | 0.9007 | 0.0355 | 0.4590 | 0.4586 | 0.9865 | 5.834 | 229,030 | 0.4064 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed4 | 0.9008 | 0.0347 | 0.4580 | 0.4531 | 0.9860 | 5.807 | 229,134 | 0.4061 | 1.029 | 5,150 |
| P2_METIS_STRONG | 0.9053 | 0.0373 | 0.4545 | 0.4541 | 0.9865 | 5.833 | 228,522 | 0.4077 | 1.029 | 5,150 |
| P3_FENNEL | 0.9273 | 0.0217 | 0.4389 | 0.3248 | 0.9845 | 6.193 | 264,256 | 0.3156 | 1.039 | 5,200 |
| P4_HYPERGRAPH_CE | 0.7852 | 0.0753 | 0.5706 | 0.2715 | 0.9860 | 4.820 | 296,258 | 0.2314 | 1.029 | 5,150 |
| P4_CE_LOCAL_ONLY | 0.8026 | 0.0727 | 0.5495 | 0.2660 | 0.9860 | 4.882 | 312,356 | 0.1900 | 1.029 | 5,150 |
| P4_CE_UNWEIGHTED | 0.8905 | 0.0460 | 0.4499 | 0.4239 | 0.9885 | 5.439 | 255,989 | 0.3364 | 1.029 | 5,150 |
| P4_CE_NER_ONLY | 0.8876 | 0.0232 | 0.4745 | 0.2442 | 0.9865 | 5.802 | 273,657 | 0.2914 | 1.029 | 5,150 |
| PM4_TOPOLOGY_C_NERW | 0.9020 | 0.0360 | 0.4570 | 0.4512 | 0.9880 | 5.811 | 228,761 | 0.4071 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s0 | 0.9913 | -0.0000 | 0.4064 | 0.0022 | 0.9740 | 7.128 | 386,817 | -0.0001 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s1 | 0.9925 | -0.0001 | 0.4064 | 0.0034 | 0.9735 | 7.140 | 386,840 | -0.0001 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s2 | 0.9917 | 0.0001 | 0.4054 | 0.0023 | 0.9735 | 7.120 | 386,854 | -0.0002 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s3 | 0.9930 | 0.0001 | 0.4059 | 0.0017 | 0.9725 | 7.125 | 386,812 | -0.0001 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s4 | 0.9915 | 0.0001 | 0.4059 | 0.0015 | 0.9735 | 7.112 | 386,845 | -0.0002 | 1.029 | 5,150 |

**2wiki_clean**

| partitioner | B1 compress | B2 adj coloc | B3 all-in-1 | B5 edge contain | B8 oracle@50 | B7 min parts | B12 cut (C) | B12 modularity | balance | B14 top-50 nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_METIS_CURRENT | 0.8317 | 0.2135 | 0.2365 | 0.5132 | 1.0000 | 1.980 | 435,819 | 0.3994 | 1.029 | 5,150 |
| PM0_STRUCT | 0.7289 | 0.3466 | 0.3790 | 0.8254 | 1.0000 | 1.711 | 620,747 | 0.1457 | 1.029 | 5,150 |
| PM1_STRUCT_NERX | 0.8097 | 0.2419 | 0.2665 | 0.5807 | 1.0000 | 1.922 | 456,210 | 0.3715 | 1.029 | 5,150 |
| PM2_STRUCT_KNN | 0.7894 | 0.2657 | 0.3020 | 0.6310 | 1.0000 | 1.875 | 550,707 | 0.2418 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C | 0.8267 | 0.2211 | 0.2410 | 0.5317 | 1.0000 | 1.964 | 435,383 | 0.4000 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed1 | 0.8364 | 0.2089 | 0.2255 | 0.5030 | 1.0000 | 1.988 | 435,807 | 0.3994 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed2 | 0.8313 | 0.2140 | 0.2360 | 0.5120 | 1.0000 | 1.978 | 435,904 | 0.3993 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed3 | 0.8305 | 0.2150 | 0.2390 | 0.5102 | 1.0000 | 1.978 | 435,878 | 0.3993 | 1.029 | 5,150 |
| PM3_TOPOLOGY_C_seed4 | 0.8304 | 0.2133 | 0.2405 | 0.5108 | 1.0000 | 1.979 | 436,488 | 0.3985 | 1.029 | 5,150 |
| P2_METIS_STRONG | 0.8276 | 0.2209 | 0.2380 | 0.5287 | 1.0000 | 1.965 | 434,708 | 0.4009 | 1.029 | 5,150 |
| P3_FENNEL | 0.8731 | 0.1610 | 0.1760 | 0.3882 | 1.0000 | 2.086 | 498,888 | 0.3128 | 1.039 | 5,200 |
| P4_HYPERGRAPH_CE | 0.8751 | 0.1534 | 0.1845 | 0.3618 | 1.0000 | 2.102 | 490,919 | 0.3219 | 1.029 | 5,150 |
| P4_CE_LOCAL_ONLY | 0.8754 | 0.1495 | 0.1915 | 0.3577 | 1.0000 | 2.110 | 560,100 | 0.2270 | 1.029 | 5,150 |
| P4_CE_UNWEIGHTED | 0.8772 | 0.1498 | 0.1830 | 0.3517 | 1.0000 | 2.109 | 486,743 | 0.3294 | 1.029 | 5,150 |
| P4_CE_NER_ONLY | 0.8151 | 0.2358 | 0.2570 | 0.5610 | 1.0000 | 1.935 | 525,836 | 0.2759 | 1.039 | 5,151 |
| PM4_TOPOLOGY_C_NERW | 0.8311 | 0.2130 | 0.2385 | 0.5138 | 1.0000 | 1.980 | 437,526 | 0.3971 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s0 | 0.9990 | -0.0000 | 0.0010 | 0.0024 | 1.0000 | 2.414 | 727,607 | -0.0008 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s1 | 0.9991 | -0.0003 | 0.0010 | 0.0018 | 1.0000 | 2.414 | 727,574 | -0.0007 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s2 | 0.9991 | -0.0003 | 0.0010 | 0.0012 | 1.0000 | 2.414 | 727,571 | -0.0007 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s3 | 0.9991 | -0.0003 | 0.0010 | 0.0018 | 1.0000 | 2.414 | 727,570 | -0.0007 | 1.029 | 5,150 |
| P0_RANDOM_BALANCED_s4 | 0.9986 | 0.0004 | 0.0015 | 0.0024 | 1.0000 | 2.413 | 727,518 | -0.0006 | 1.029 | 5,150 |

**musique_clean**

| partitioner | B1 compress | B2 adj coloc | B3 all-in-1 | B5 edge contain | B8 oracle@50 | B7 min parts | B12 cut (C) | B12 modularity | balance | B14 top-50 nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_METIS_CURRENT | 0.7593 | 0.3443 | 0.3725 | 0.6756 | 1.0000 | 1.775 | 74,205 | 0.5378 | 1.025 | 5,150 |
| PM0_STRUCT | 0.8205 | 0.2599 | 0.2575 | 0.4866 | 1.0000 | 1.907 | 112,281 | 0.3054 | 1.034 | 5,152 |
| PM1_STRUCT_NERX | 0.7615 | 0.3344 | 0.3670 | 0.6840 | 1.0000 | 1.783 | 77,697 | 0.5167 | 1.025 | 5,150 |
| PM2_STRUCT_KNN | 0.7737 | 0.3199 | 0.3430 | 0.5868 | 1.0000 | 1.808 | 93,394 | 0.4206 | 1.025 | 5,147 |
| PM3_TOPOLOGY_C | 0.7610 | 0.3315 | 0.3755 | 0.6526 | 1.0000 | 1.787 | 73,748 | 0.5408 | 1.025 | 5,150 |
| PM3_TOPOLOGY_C_seed1 | 0.7540 | 0.3463 | 0.3810 | 0.6779 | 1.0000 | 1.766 | 74,393 | 0.5366 | 1.025 | 5,150 |
| PM3_TOPOLOGY_C_seed2 | 0.7520 | 0.3536 | 0.3870 | 0.6718 | 1.0000 | 1.760 | 74,303 | 0.5374 | 1.025 | 5,150 |
| PM3_TOPOLOGY_C_seed3 | 0.7595 | 0.3310 | 0.3715 | 0.6718 | 1.0000 | 1.782 | 74,110 | 0.5386 | 1.025 | 5,150 |
| PM3_TOPOLOGY_C_seed4 | 0.7561 | 0.3376 | 0.3835 | 0.6679 | 1.0000 | 1.775 | 74,386 | 0.5368 | 1.025 | 5,150 |
| P2_METIS_STRONG | 0.7554 | 0.3391 | 0.3815 | 0.6595 | 1.0000 | 1.774 | 73,622 | 0.5416 | 1.025 | 5,150 |
| P3_FENNEL | 0.8420 | 0.2100 | 0.2485 | 0.4767 | 1.0000 | 1.978 | 98,511 | 0.3899 | 1.034 | 5,200 |
| P4_HYPERGRAPH_CE | 0.7778 | 0.3231 | 0.3350 | 0.5907 | 1.0000 | 1.812 | 99,649 | 0.3830 | 1.025 | 5,150 |
| P4_CE_LOCAL_ONLY | 0.8091 | 0.2703 | 0.2780 | 0.5295 | 1.0000 | 1.887 | 104,124 | 0.3557 | 1.025 | 5,150 |
| P4_CE_UNWEIGHTED | 0.7722 | 0.3194 | 0.3585 | 0.6067 | 1.0000 | 1.812 | 86,531 | 0.4628 | 1.025 | 5,150 |
| P4_CE_NER_ONLY | 0.8108 | 0.2718 | 0.2735 | 0.5907 | 1.0000 | 1.887 | 101,185 | 0.3736 | 1.025 | 5,147 |
| PM4_TOPOLOGY_C_NERW | 0.7529 | 0.3475 | 0.3855 | 0.6725 | 1.0000 | 1.765 | 75,075 | 0.5327 | 1.025 | 5,146 |
| P0_RANDOM_BALANCED_s0 | 0.9944 | 0.0014 | 0.0045 | 0.0115 | 1.0000 | 2.325 | 162,514 | -0.0001 | 1.025 | 5,150 |
| P0_RANDOM_BALANCED_s1 | 0.9947 | 0.0014 | 0.0045 | 0.0069 | 1.0000 | 2.325 | 162,481 | 0.0001 | 1.025 | 5,150 |
| P0_RANDOM_BALANCED_s2 | 0.9968 | -0.0024 | 0.0030 | 0.0046 | 1.0000 | 2.332 | 162,525 | -0.0002 | 1.025 | 5,150 |
| P0_RANDOM_BALANCED_s3 | 0.9938 | 0.0022 | 0.0065 | 0.0107 | 1.0000 | 2.324 | 162,541 | -0.0002 | 1.025 | 5,150 |
| P0_RANDOM_BALANCED_s4 | 0.9936 | 0.0022 | 0.0070 | 0.0145 | 1.0000 | 2.324 | 162,494 | 0.0001 | 1.025 | 5,150 |

**squad_clean**

| partitioner | B1 compress | B2 adj coloc | B3 all-in-1 | B5 edge contain | B8 oracle@50 | B7 min parts | B12 cut (C) | B12 modularity | balance | B14 top-50 nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_METIS_CURRENT | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 511,295 | 0.3847 | 1.028 | 5,150 |
| PM0_STRUCT | 1.0000 | -0.0053 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 542,559 | 0.3479 | 1.038 | 5,151 |
| PM1_STRUCT_NERX | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 523,917 | 0.3697 | 1.028 | 5,150 |
| PM2_STRUCT_KNN | 1.0000 | -0.0053 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 517,694 | 0.3770 | 1.028 | 5,150 |
| PM3_TOPOLOGY_C | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 511,247 | 0.3846 | 1.028 | 5,150 |
| PM3_TOPOLOGY_C_seed1 | 1.0000 | -0.0053 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 508,568 | 0.3877 | 1.028 | 5,150 |
| PM3_TOPOLOGY_C_seed2 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 513,894 | 0.3815 | 1.028 | 5,150 |
| PM3_TOPOLOGY_C_seed3 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 511,502 | 0.3842 | 1.028 | 5,150 |
| PM3_TOPOLOGY_C_seed4 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 515,451 | 0.3796 | 1.028 | 5,150 |
| P2_METIS_STRONG | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 506,247 | 0.3905 | 1.028 | 5,150 |
| P3_FENNEL | 1.0000 | -0.0053 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 642,480 | 0.2285 | 1.038 | 5,200 |
| P4_HYPERGRAPH_CE | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 744,327 | 0.1092 | 1.028 | 5,150 |
| P4_CE_LOCAL_ONLY | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 769,766 | 0.0787 | 1.028 | 5,145 |
| P4_CE_UNWEIGHTED | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 715,731 | 0.1432 | 1.028 | 5,150 |
| P4_CE_NER_ONLY | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 732,633 | 0.1233 | 1.028 | 5,150 |
| PM4_TOPOLOGY_C_NERW | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 518,066 | 0.3766 | 1.028 | 5,150 |
| P0_RANDOM_BALANCED_s0 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 836,937 | -0.0001 | 1.028 | 5,150 |
| P0_RANDOM_BALANCED_s1 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 837,001 | -0.0002 | 1.028 | 5,150 |
| P0_RANDOM_BALANCED_s2 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 837,079 | -0.0003 | 1.028 | 5,150 |
| P0_RANDOM_BALANCED_s3 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 836,952 | -0.0001 | 1.028 | 5,150 |
| P0_RANDOM_BALANCED_s4 | 1.0000 | -0.0052 | 1.0000 | 0.0000 | 1.0000 | 1.000 | 836,922 | -0.0001 | 1.028 | 5,150 |

**hotpotqa_clean**

| partitioner | B1 compress | B2 adj coloc | B3 all-in-1 | B5 edge contain | B8 oracle@50 | B7 min parts | B12 cut (C) | B12 modularity | balance | B14 top-50 nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_METIS_CURRENT | 0.8320 | 0.3359 | 0.3360 | 0.4571 | 1.0000 | 1.664 | 5,851,058 | 0.3085 | 1.030 | 5,150 |
| PM0_STRUCT | 0.7970 | 0.4059 | 0.4060 | 0.5594 | 1.0000 | 1.594 | 7,113,255 | 0.1595 | 1.030 | 5,150 |
| PM1_STRUCT_NERX | 0.8290 | 0.3419 | 0.3420 | 0.4757 | 1.0000 | 1.658 | 5,916,057 | 0.3009 | 1.030 | 5,150 |
| PM2_STRUCT_KNN | 0.8090 | 0.3819 | 0.3820 | 0.5086 | 1.0000 | 1.618 | 6,852,759 | 0.1902 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C | 0.8310 | 0.3379 | 0.3380 | 0.4564 | 1.0000 | 1.662 | 5,850,943 | 0.3085 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed1 | 0.8297 | 0.3404 | 0.3405 | 0.4607 | 1.0000 | 1.659 | 5,848,295 | 0.3088 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed2 | 0.8295 | 0.3409 | 0.3410 | 0.4592 | 1.0000 | 1.659 | 5,847,937 | 0.3089 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed3 | 0.8333 | 0.3334 | 0.3335 | 0.4478 | 1.0000 | 1.667 | 5,848,990 | 0.3088 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed4 | 0.8263 | 0.3474 | 0.3475 | 0.4742 | 1.0000 | 1.653 | 5,847,620 | 0.3089 | 1.030 | 5,150 |
| P2_METIS_STRONG | 0.8310 | 0.3379 | 0.3380 | 0.4592 | 1.0000 | 1.662 | 5,839,702 | 0.3098 | 1.030 | 5,150 |
| P3_FENNEL | 0.8730 | 0.2539 | 0.2540 | 0.3534 | 1.0000 | 1.746 | 6,458,722 | 0.2367 | 1.040 | 5,200 |
| P4_HYPERGRAPH_CE | 0.8492 | 0.3014 | 0.3015 | 0.3920 | 1.0000 | 1.698 | 6,287,558 | 0.2524 | 1.030 | 5,150 |
| P4_CE_LOCAL_ONLY | 0.8470 | 0.3059 | 0.3060 | 0.3984 | 1.0000 | 1.694 | 6,981,818 | 0.1716 | 1.030 | 5,150 |
| P4_CE_UNWEIGHTED | 0.8550 | 0.2899 | 0.2900 | 0.3863 | 1.0000 | 1.710 | 6,299,143 | 0.2556 | 1.030 | 5,150 |
| P4_CE_NER_ONLY | 0.8688 | 0.2624 | 0.2625 | 0.3627 | 1.0000 | 1.738 | 6,669,346 | 0.2119 | 1.040 | 5,157 |
| PM4_TOPOLOGY_C_NERW | 0.8270 | 0.3459 | 0.3460 | 0.4714 | 1.0000 | 1.654 | 5,857,004 | 0.3078 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s0 | 1.0000 | -0.0002 | 0.0000 | 0.0000 | 1.0000 | 2.000 | 8,466,741 | -0.0003 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s1 | 0.9998 | 0.0003 | 0.0005 | 0.0007 | 1.0000 | 2.000 | 8,466,680 | -0.0003 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s2 | 0.9998 | 0.0003 | 0.0005 | 0.0007 | 1.0000 | 2.000 | 8,466,707 | -0.0003 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s3 | 0.9995 | 0.0008 | 0.0010 | 0.0007 | 1.0000 | 1.999 | 8,466,655 | -0.0003 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s4 | 1.0000 | -0.0002 | 0.0000 | 0.0000 | 1.0000 | 2.000 | 8,466,679 | -0.0003 | 1.030 | 5,150 |

**webqsp**

| partitioner | B1 compress | B2 adj coloc | B3 all-in-1 | B5 edge contain | B8 oracle@50 | B7 min parts | B12 cut (C) | B12 modularity | balance | B14 top-50 nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| P1_METIS_CURRENT | 0.8997 | 0.2630 | 0.5208 | 0.7662 | 0.9951 | 3.664 | 1,846,207 | 0.5554 | 1.030 | 5,150 |
| PM0_STRUCT | 0.9254 | 0.1173 | 0.5074 | 0.4179 | 0.9944 | 3.975 | 2,802,557 | 0.3252 | 1.030 | 5,150 |
| PM1_STRUCT_NERX | 0.9276 | 0.1416 | 0.5095 | 0.6088 | 0.9937 | 3.996 | 2,442,697 | 0.4118 | 1.030 | 5,150 |
| PM2_STRUCT_KNN | 0.8755 | 0.2816 | 0.5447 | 0.7400 | 0.9951 | 3.417 | 2,133,608 | 0.4862 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C | 0.9016 | 0.2363 | 0.5201 | 0.7555 | 0.9951 | 3.668 | 1,846,666 | 0.5553 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed1 | 0.9015 | 0.2516 | 0.5208 | 0.7488 | 0.9951 | 3.670 | 1,847,816 | 0.5550 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed2 | 0.8963 | 0.2093 | 0.5243 | 0.7382 | 0.9951 | 3.643 | 1,849,120 | 0.5547 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed3 | 0.9044 | 0.1992 | 0.5215 | 0.6963 | 0.9958 | 3.679 | 1,847,015 | 0.5552 | 1.030 | 5,150 |
| PM3_TOPOLOGY_C_seed4 | 0.9038 | 0.2018 | 0.5208 | 0.7456 | 0.9958 | 3.683 | 1,847,423 | 0.5551 | 1.030 | 5,150 |
| P2_METIS_STRONG | 0.9014 | 0.2551 | 0.5194 | 0.7632 | 0.9958 | 3.653 | 1,841,119 | 0.5566 | 1.030 | 5,150 |
| P3_FENNEL | 0.9222 | 0.1372 | 0.5074 | 0.5966 | 0.9944 | 3.977 | 2,454,917 | 0.4088 | 1.040 | 5,200 |
| P4_HYPERGRAPH_CE | 0.8837 | 0.3147 | 0.5243 | 0.7696 | 0.9958 | 3.451 | 2,083,175 | 0.4975 | 1.030 | 5,150 |
| P4_CE_LOCAL_ONLY | 0.8691 | 0.2837 | 0.5447 | 0.7088 | 0.9958 | 3.402 | 2,282,372 | 0.4496 | 1.030 | 5,150 |
| P4_CE_UNWEIGHTED | 0.8941 | 0.2592 | 0.5243 | 0.7425 | 0.9958 | 3.536 | 2,078,346 | 0.4995 | 1.030 | 5,150 |
| P4_CE_NER_ONLY | 0.9307 | 0.1044 | 0.5208 | 0.6155 | 0.9930 | 4.331 | 3,072,409 | 0.2602 | 1.110 | 5,164 |
| PM4_TOPOLOGY_C_NERW | 0.9022 | 0.2403 | 0.5208 | 0.7649 | 0.9951 | 3.667 | 1,855,032 | 0.5533 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s0 | 0.9956 | 0.0186 | 0.4820 | 0.0001 | 0.9817 | 5.483 | 4,154,078 | -0.0001 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s1 | 0.9955 | 0.0187 | 0.4820 | 0.0001 | 0.9817 | 5.481 | 4,154,005 | -0.0001 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s2 | 0.9953 | 0.0186 | 0.4820 | 0.0001 | 0.9817 | 5.483 | 4,154,055 | -0.0001 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s3 | 0.9955 | 0.0187 | 0.4820 | 0.0000 | 0.9817 | 5.483 | 4,154,046 | -0.0001 | 1.030 | 5,150 |
| P0_RANDOM_BALANCED_s4 | 0.9955 | 0.0187 | 0.4820 | 0.0001 | 0.9817 | 5.486 | 4,154,021 | -0.0001 | 1.030 | 5,150 |

## C -- frozen L1 replayed on each partitioning (P=50, B=6, S4, F6; nothing retuned)

**metaqa** (parity: `EXACT`)

| partitioning | BASE ALL@50 | delta | F6 ALL@50 | delta | exposure (nodes) |
|---|---|---|---|---|---|
| PM_CURRENT_EXACT | 0.6587 | +0.0000 | 0.6612 | +0.0000 | 5,038 |
| PM0_STRUCT | 0.6481 | -0.0106 | 0.6557 | -0.0055 | 5,023 |
| PM1_STRUCT_NERX | 0.6632 | +0.0045 | 0.6617 | +0.0005 | 5,040 |
| PM2_STRUCT_KNN | 0.6431 | -0.0156 | 0.6461 | -0.0151 | 5,024 |
| PM3_TOPOLOGY_C | 0.6557 | -0.0030 | 0.6582 | -0.0030 | 5,041 |
| PM3_TOPOLOGY_C_seed1 | 0.6567 | -0.0020 | 0.6612 | +0.0000 | 5,034 |
| PM3_TOPOLOGY_C_seed2 | 0.6607 | +0.0020 | 0.6627 | +0.0015 | 5,034 |
| PM3_TOPOLOGY_C_seed3 | 0.6567 | -0.0020 | 0.6572 | -0.0040 | 5,039 |
| PM3_TOPOLOGY_C_seed4 | 0.6572 | -0.0015 | 0.6597 | -0.0015 | 5,037 |
| P2_METIS_STRONG | 0.6577 | -0.0010 | 0.6567 | -0.0045 | 5,035 |
| P3_FENNEL | 0.6341 | -0.0246 | 0.6356 | -0.0256 | 5,064 |
| P4_HYPERGRAPH_CE | 0.7467 | +0.0880 | 0.7548 | +0.0936 | 4,986 |
| P4_CE_LOCAL_ONLY | 0.7548 | +0.0961 | 0.7578 | +0.0966 | 4,977 |
| P4_CE_UNWEIGHTED | 0.6627 | +0.0040 | 0.6607 | -0.0005 | 5,038 |
| P4_CE_NER_ONLY | 0.6547 | -0.0040 | 0.6602 | -0.0010 | 4,999 |
| PM4_TOPOLOGY_C_NERW | 0.6542 | -0.0045 | 0.6562 | -0.0050 | 5,034 |
| P0_RANDOM_BALANCED_s0 | 0.5811 | -0.0776 | 0.5796 | -0.0816 | 5,003 |
| P0_RANDOM_BALANCED_s1 | 0.5806 | -0.0781 | 0.5786 | -0.0826 | 5,022 |
| P0_RANDOM_BALANCED_s2 | 0.5871 | -0.0716 | 0.5856 | -0.0756 | 5,012 |
| P0_RANDOM_BALANCED_s3 | 0.5841 | -0.0746 | 0.5766 | -0.0846 | 5,005 |
| P0_RANDOM_BALANCED_s4 | 0.5801 | -0.0786 | 0.5806 | -0.0806 | 5,016 |

**2wiki_clean** (parity: `EXACT`)

| partitioning | BASE ALL@50 | delta | F6 ALL@50 | delta | exposure (nodes) |
|---|---|---|---|---|---|
| PM_CURRENT_EXACT | 0.9375 | +0.0000 | 0.9435 | +0.0000 | 5,028 |
| PM0_STRUCT | 0.8945 | -0.0430 | 0.8990 | -0.0445 | 5,066 |
| PM1_STRUCT_NERX | 0.9380 | +0.0005 | 0.9455 | +0.0020 | 5,048 |
| PM2_STRUCT_KNN | 0.9255 | -0.0120 | 0.9315 | -0.0120 | 5,027 |
| PM3_TOPOLOGY_C | 0.9415 | +0.0040 | 0.9485 | +0.0050 | 5,028 |
| PM3_TOPOLOGY_C_seed1 | 0.9390 | +0.0015 | 0.9490 | +0.0055 | 5,031 |
| PM3_TOPOLOGY_C_seed2 | 0.9370 | -0.0005 | 0.9445 | +0.0010 | 5,031 |
| PM3_TOPOLOGY_C_seed3 | 0.9430 | +0.0055 | 0.9490 | +0.0055 | 5,022 |
| PM3_TOPOLOGY_C_seed4 | 0.9355 | -0.0020 | 0.9460 | +0.0025 | 5,031 |
| P2_METIS_STRONG | 0.9390 | +0.0015 | 0.9475 | +0.0040 | 5,031 |
| P3_FENNEL | 0.9255 | -0.0120 | 0.9345 | -0.0090 | 5,042 |
| P4_HYPERGRAPH_CE | 0.9395 | +0.0020 | 0.9410 | -0.0025 | 4,990 |
| P4_CE_LOCAL_ONLY | 0.9395 | +0.0020 | 0.9380 | -0.0055 | 4,994 |
| P4_CE_UNWEIGHTED | 0.9330 | -0.0045 | 0.9400 | -0.0035 | 5,011 |
| P4_CE_NER_ONLY | 0.9395 | +0.0020 | 0.9470 | +0.0035 | 5,002 |
| PM4_TOPOLOGY_C_NERW | 0.9410 | +0.0035 | 0.9485 | +0.0050 | 5,026 |
| P0_RANDOM_BALANCED_s0 | 0.8645 | -0.0730 | 0.8685 | -0.0750 | 5,014 |
| P0_RANDOM_BALANCED_s1 | 0.8625 | -0.0750 | 0.8660 | -0.0775 | 5,005 |
| P0_RANDOM_BALANCED_s2 | 0.8695 | -0.0680 | 0.8750 | -0.0685 | 5,008 |
| P0_RANDOM_BALANCED_s3 | 0.8620 | -0.0755 | 0.8680 | -0.0755 | 5,009 |
| P0_RANDOM_BALANCED_s4 | 0.8655 | -0.0720 | 0.8705 | -0.0730 | 5,007 |

**musique_clean** (parity: `EXACT`)

| partitioning | BASE ALL@50 | delta | F6 ALL@50 | delta | exposure (nodes) |
|---|---|---|---|---|---|
| PM_CURRENT_EXACT | 0.9565 | +0.0000 | 0.9635 | +0.0000 | 5,042 |
| PM0_STRUCT | 0.9165 | -0.0400 | 0.9225 | -0.0410 | 5,056 |
| PM1_STRUCT_NERX | 0.9445 | -0.0120 | 0.9515 | -0.0120 | 5,041 |
| PM2_STRUCT_KNN | 0.9430 | -0.0135 | 0.9465 | -0.0170 | 5,047 |
| PM3_TOPOLOGY_C | 0.9505 | -0.0060 | 0.9555 | -0.0080 | 5,039 |
| PM3_TOPOLOGY_C_seed1 | 0.9505 | -0.0060 | 0.9605 | -0.0030 | 5,049 |
| PM3_TOPOLOGY_C_seed2 | 0.9500 | -0.0065 | 0.9560 | -0.0075 | 5,038 |
| PM3_TOPOLOGY_C_seed3 | 0.9515 | -0.0050 | 0.9575 | -0.0060 | 5,048 |
| PM3_TOPOLOGY_C_seed4 | 0.9510 | -0.0055 | 0.9605 | -0.0030 | 5,049 |
| P2_METIS_STRONG | 0.9510 | -0.0055 | 0.9605 | -0.0030 | 5,039 |
| P3_FENNEL | 0.9045 | -0.0520 | 0.9205 | -0.0430 | 5,063 |
| P4_HYPERGRAPH_CE | 0.9535 | -0.0030 | 0.9620 | -0.0015 | 5,027 |
| P4_CE_LOCAL_ONLY | 0.9460 | -0.0105 | 0.9565 | -0.0070 | 5,033 |
| P4_CE_UNWEIGHTED | 0.9465 | -0.0100 | 0.9550 | -0.0085 | 5,033 |
| P4_CE_NER_ONLY | 0.9185 | -0.0380 | 0.9420 | -0.0215 | 5,023 |
| PM4_TOPOLOGY_C_NERW | 0.9460 | -0.0105 | 0.9560 | -0.0075 | 5,038 |
| P0_RANDOM_BALANCED_s0 | 0.7090 | -0.2475 | 0.7550 | -0.2085 | 5,028 |
| P0_RANDOM_BALANCED_s1 | 0.7060 | -0.2505 | 0.7595 | -0.2040 | 5,035 |
| P0_RANDOM_BALANCED_s2 | 0.7090 | -0.2475 | 0.7445 | -0.2190 | 5,032 |
| P0_RANDOM_BALANCED_s3 | 0.7105 | -0.2460 | 0.7520 | -0.2115 | 5,029 |
| P0_RANDOM_BALANCED_s4 | 0.7135 | -0.2430 | 0.7495 | -0.2140 | 5,026 |

**squad_clean** (parity: `EXACT`)

| partitioning | BASE ALL@50 | delta | F6 ALL@50 | delta | exposure (nodes) |
|---|---|---|---|---|---|
| PM_CURRENT_EXACT | 0.9805 | +0.0000 | 0.9875 | +0.0000 | 5,039 |
| PM0_STRUCT | 0.9655 | -0.0150 | 0.9785 | -0.0090 | 5,090 |
| PM1_STRUCT_NERX | 0.9745 | -0.0060 | 0.9840 | -0.0035 | 5,030 |
| PM2_STRUCT_KNN | 0.9760 | -0.0045 | 0.9835 | -0.0040 | 5,076 |
| PM3_TOPOLOGY_C | 0.9790 | -0.0015 | 0.9855 | -0.0020 | 5,035 |
| PM3_TOPOLOGY_C_seed1 | 0.9820 | +0.0015 | 0.9860 | -0.0015 | 5,060 |
| PM3_TOPOLOGY_C_seed2 | 0.9800 | -0.0005 | 0.9850 | -0.0025 | 5,029 |
| PM3_TOPOLOGY_C_seed3 | 0.9815 | +0.0010 | 0.9875 | +0.0000 | 5,031 |
| PM3_TOPOLOGY_C_seed4 | 0.9775 | -0.0030 | 0.9850 | -0.0025 | 5,025 |
| P2_METIS_STRONG | 0.9780 | -0.0025 | 0.9840 | -0.0035 | 5,036 |
| P3_FENNEL | 0.9555 | -0.0250 | 0.9755 | -0.0120 | 5,028 |
| P4_HYPERGRAPH_CE | 0.9565 | -0.0240 | 0.9785 | -0.0090 | 4,989 |
| P4_CE_LOCAL_ONLY | 0.9520 | -0.0285 | 0.9780 | -0.0095 | 5,010 |
| P4_CE_UNWEIGHTED | 0.9485 | -0.0320 | 0.9730 | -0.0145 | 4,990 |
| P4_CE_NER_ONLY | 0.9255 | -0.0550 | 0.9635 | -0.0240 | 4,993 |
| PM4_TOPOLOGY_C_NERW | 0.9785 | -0.0020 | 0.9855 | -0.0020 | 5,037 |
| P0_RANDOM_BALANCED_s0 | 0.7285 | -0.2520 | 0.8240 | -0.1635 | 5,015 |
| P0_RANDOM_BALANCED_s1 | 0.7355 | -0.2450 | 0.8150 | -0.1725 | 5,016 |
| P0_RANDOM_BALANCED_s2 | 0.7495 | -0.2310 | 0.8300 | -0.1575 | 5,016 |
| P0_RANDOM_BALANCED_s3 | 0.7450 | -0.2355 | 0.8290 | -0.1585 | 5,006 |
| P0_RANDOM_BALANCED_s4 | 0.7630 | -0.2175 | 0.8420 | -0.1455 | 5,006 |

**hotpotqa_clean** (parity: `EXACT`)

| partitioning | BASE ALL@50 | delta | F6 ALL@50 | delta | exposure (nodes) |
|---|---|---|---|---|---|
| PM_CURRENT_EXACT | 0.9345 | +0.0000 | 0.9505 | +0.0000 | 5,082 |
| PM0_STRUCT | 0.9260 | -0.0085 | 0.9490 | -0.0015 | 5,100 |
| PM1_STRUCT_NERX | 0.9250 | -0.0095 | 0.9470 | -0.0035 | 5,081 |
| PM2_STRUCT_KNN | 0.9435 | +0.0090 | 0.9615 | +0.0110 | 5,084 |
| PM3_TOPOLOGY_C | 0.9320 | -0.0025 | 0.9515 | +0.0010 | 5,078 |
| PM3_TOPOLOGY_C_seed1 | 0.9355 | +0.0010 | 0.9545 | +0.0040 | 5,080 |
| PM3_TOPOLOGY_C_seed2 | 0.9320 | -0.0025 | 0.9510 | +0.0005 | 5,081 |
| PM3_TOPOLOGY_C_seed3 | 0.9320 | -0.0025 | 0.9485 | -0.0020 | 5,078 |
| PM3_TOPOLOGY_C_seed4 | 0.9350 | +0.0005 | 0.9540 | +0.0035 | 5,082 |
| P2_METIS_STRONG | 0.9330 | -0.0015 | 0.9520 | +0.0015 | 5,082 |
| P3_FENNEL | 0.9030 | -0.0315 | 0.9335 | -0.0170 | 5,083 |
| P4_HYPERGRAPH_CE | 0.9475 | +0.0130 | 0.9585 | +0.0080 | 5,043 |
| P4_CE_LOCAL_ONLY | 0.9525 | +0.0180 | 0.9655 | +0.0150 | 5,052 |
| P4_CE_UNWEIGHTED | 0.9285 | -0.0060 | 0.9505 | +0.0000 | 5,072 |
| P4_CE_NER_ONLY | 0.9065 | -0.0280 | 0.9315 | -0.0190 | 4,998 |
| PM4_TOPOLOGY_C_NERW | 0.9320 | -0.0025 | 0.9500 | -0.0005 | 5,078 |
| P0_RANDOM_BALANCED_s0 | 0.8305 | -0.1040 | 0.8760 | -0.0745 | 4,994 |
| P0_RANDOM_BALANCED_s1 | 0.8310 | -0.1035 | 0.8755 | -0.0750 | 5,012 |
| P0_RANDOM_BALANCED_s2 | 0.8275 | -0.1070 | 0.8760 | -0.0745 | 5,005 |
| P0_RANDOM_BALANCED_s3 | 0.8255 | -0.1090 | 0.8700 | -0.0805 | 5,008 |
| P0_RANDOM_BALANCED_s4 | 0.8280 | -0.1065 | 0.8735 | -0.0770 | 4,994 |

**webqsp** (parity: `EXACT`)

| partitioning | BASE ALL@50 | delta | F6 ALL@50 | delta | exposure (nodes) |
|---|---|---|---|---|---|
| PM_CURRENT_EXACT | 0.7618 | +0.0000 | 0.7646 | +0.0000 | 5,023 |
| PM0_STRUCT | 0.7752 | +0.0134 | 0.7766 | +0.0120 | 5,049 |
| PM1_STRUCT_NERX | 0.7667 | +0.0049 | 0.7696 | +0.0050 | 5,043 |
| PM2_STRUCT_KNN | 0.7907 | +0.0289 | 0.7808 | +0.0162 | 5,025 |
| PM3_TOPOLOGY_C | 0.7533 | -0.0085 | 0.7590 | -0.0056 | 5,026 |
| PM3_TOPOLOGY_C_seed1 | 0.7519 | -0.0099 | 0.7512 | -0.0134 | 5,023 |
| PM3_TOPOLOGY_C_seed2 | 0.7632 | +0.0014 | 0.7681 | +0.0035 | 5,024 |
| PM3_TOPOLOGY_C_seed3 | 0.7526 | -0.0092 | 0.7604 | -0.0042 | 5,026 |
| PM3_TOPOLOGY_C_seed4 | 0.7597 | -0.0021 | 0.7710 | +0.0064 | 5,026 |
| P2_METIS_STRONG | 0.7512 | -0.0106 | 0.7576 | -0.0070 | 5,026 |
| P3_FENNEL | 0.7209 | -0.0409 | 0.7294 | -0.0352 | 5,046 |
| P4_HYPERGRAPH_CE | 0.7851 | +0.0233 | 0.7886 | +0.0240 | 5,017 |
| P4_CE_LOCAL_ONLY | 0.8055 | +0.0437 | 0.8069 | +0.0423 | 5,014 |
| P4_CE_UNWEIGHTED | 0.7752 | +0.0134 | 0.7710 | +0.0064 | 5,022 |
| P4_CE_NER_ONLY | 0.6920 | -0.0698 | 0.6970 | -0.0676 | 5,002 |
| PM4_TOPOLOGY_C_NERW | 0.7632 | +0.0014 | 0.7681 | +0.0035 | 5,022 |
| P0_RANDOM_BALANCED_s0 | 0.5821 | -0.1797 | 0.5863 | -0.1783 | 4,998 |
| P0_RANDOM_BALANCED_s1 | 0.5835 | -0.1783 | 0.5884 | -0.1762 | 5,003 |
| P0_RANDOM_BALANCED_s2 | 0.5800 | -0.1818 | 0.5856 | -0.1790 | 5,000 |
| P0_RANDOM_BALANCED_s3 | 0.5863 | -0.1755 | 0.5913 | -0.1733 | 5,002 |
| P0_RANDOM_BALANCED_s4 | 0.5828 | -0.1790 | 0.5899 | -0.1747 | 5,002 |

## A8 -- path-family analysis: do BRIDGE paths carry needed partitions?

A signature is HOMOGENEOUS when every hop is carried by one single family (`STRUCT>STRUCT`), and a BRIDGE when the hops mix families (`STRUCT>KNN`, `KNN+NERX>STRUCT`). `needed yield` = needed targets / all targets reached by that signature class -- the precision of the path shape, not its volume.

| corpus | path length | homogeneous yield | bridge yield | delta | bridge better |
|---|---|---|---|---|---|
| metaqa | hop1 | 0.1235 | 0.2368 | +0.1133 | YES |
| metaqa | hop2 | 0.0581 | 0.0500 | -0.0080 | NO |
| metaqa | hop3 | 0.0350 | 0.0276 | -0.0074 | NO |
| 2wiki_clean | hop1 | 0.1920 | 0.3937 | +0.2016 | YES |
| 2wiki_clean | hop2 | 0.1050 | 0.1042 | -0.0008 | NO |
| 2wiki_clean | hop3 | 0.0474 | 0.0415 | -0.0059 | NO |
| musique_clean | hop1 | 0.3225 | 0.5085 | +0.1860 | YES |
| musique_clean | hop2 | 0.1558 | 0.1491 | -0.0067 | NO |
| musique_clean | hop3 | 0.0632 | 0.0437 | -0.0195 | NO |
| squad_clean | hop1 | 0.3182 | 0.2974 | -0.0208 | NO |
| squad_clean | hop2 | 0.1541 | 0.0853 | -0.0688 | NO |
| squad_clean | hop3 | 0.0273 | 0.0202 | -0.0072 | NO |
| hotpotqa_clean | hop1 | 0.2585 | 0.5200 | +0.2615 | YES |
| hotpotqa_clean | hop2 | 0.0757 | 0.1164 | +0.0407 | YES |
| hotpotqa_clean | hop3 | 0.0320 | 0.0334 | +0.0014 | YES |
| webqsp | hop1 | 0.1890 | 0.3793 | +0.1903 | YES |
| webqsp | hop2 | 0.0996 | 0.1283 | +0.0287 | YES |
| webqsp | hop3 | 0.0373 | 0.0578 | +0.0205 | YES |

## A9 -- distinct evidence per family

`K3 novel` counts partitions the kNN family reaches that NO other channel reaches (frozen structural traversal, Dense top-200, SPLADE top-200, retrieval frontier, context). A family is not credited from AUC alone -- it must contribute a needed partition nothing else supplies.

| corpus | K reached | K0 also STRUCT | K1 also Dense/SPLADE | K3 novel | K3 novel frac | K3 novel AND needed | distinct |
|---|---|---|---|---|---|---|---|
| metaqa | 145,678 | 122,499 | 91,234 | 7,704 | 0.0529 | 14 | YES |
| 2wiki_clean | 187,205 | 78,583 | 101,025 | 41,409 | 0.2212 | 5 | YES |
| musique_clean | 96,276 | 85,293 | 96,276 | 0 | 0.0000 | 0 | NO |
| squad_clean | 60,570 | 42,268 | 60,570 | 0 | 0.0000 | 0 | NO |
| hotpotqa_clean | 199,493 | 87,946 | 56,386 | 79,160 | 0.3968 | 2 | YES |
| webqsp | 56,767 | 23,840 | 21,087 | 24,644 | 0.4341 | 23 | YES |

| corpus | N unique among missed needed | fraction | AUC edge-sim | AUC query-sim | dense in costume | N0/N1 |
|---|---|---|---|---|---|---|
| metaqa | 272 | 0.0468 | 0.6469 | 0.7497 | YES | NOT_MEASURED |
| 2wiki_clean | 22 | 0.1880 | 0.8579 | 0.9302 | YES | NOT_MEASURED |
| musique_clean | 6 | 0.0800 | 0.8371 | 0.9312 | YES | NOT_MEASURED |
| squad_clean | 7 | 0.2800 | 0.8509 | 0.9287 | YES | NOT_MEASURED |
| hotpotqa_clean | 12 | 0.1154 | 0.9267 | 0.9736 | YES | NOT_MEASURED |
| webqsp | 28 | 0.0166 | 0.8023 | 0.9006 | YES | NOT_MEASURED |

## B15 -- stability and significance

`PM3_TOPOLOGY_C_seed*` re-runs METIS on the IDENTICAL graph with a different RNG. Production never pinned a seed, so this spread is the noise floor every partitioner delta must clear; a candidate is only credited when it exceeds two of these standard deviations AND is significant by exact McNemar on the same queries.

| corpus | production F6 | METIS reseed sd | reseed min | reseed max | noise floor sd | random mean | random sd |
|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.0023 | 0.6572 | 0.6627 | 0.0021 | 0.5802 | 0.0034 |
| 2wiki_clean | 0.9435 | 0.0022 | 0.9445 | 0.9490 | 0.0025 | 0.8696 | 0.0034 |
| musique_clean | 0.9635 | 0.0023 | 0.9560 | 0.9605 | 0.0029 | 0.7521 | 0.0057 |
| squad_clean | 0.9875 | 0.0012 | 0.9850 | 0.9875 | 0.0013 | 0.8280 | 0.0098 |
| hotpotqa_clean | 0.9505 | 0.0028 | 0.9485 | 0.9545 | 0.0025 | 0.8742 | 0.0026 |
| webqsp | 0.7646 | 0.0089 | 0.7512 | 0.7710 | 0.0077 | 0.5883 | 0.0024 |

**metaqa** -- candidates vs production (exact McNemar, same 1998 queries)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_CE_LOCAL_ONLY | 0.7578 | +0.0966 | 193 | 0.0000 | SIG | 0.988 | YES |
| P4_HYPERGRAPH_CE | 0.7548 | +0.0936 | 187 | 0.0000 | SIG | 0.990 | YES |
| PM1_STRUCT_NERX | 0.6617 | +0.0005 | 1 | 1.0000 | ns | 1.000 | NO |
| P4_CE_UNWEIGHTED | 0.6607 | -0.0005 | -1 | 1.0000 | ns | 1.000 | NO |
| P4_CE_NER_ONLY | 0.6602 | -0.0010 | -2 | 0.9350 | ns | 0.992 | NO |
| PM3_TOPOLOGY_C | 0.6582 | -0.0030 | -6 | 0.6536 | ns | 1.001 | NO |
| P2_METIS_STRONG | 0.6567 | -0.0045 | -9 | 0.4394 | ns | 0.999 | YES |
| PM4_TOPOLOGY_C_NERW | 0.6562 | -0.0050 | -10 | 0.4075 | ns | 0.999 | YES |
| PM0_STRUCT | 0.6557 | -0.0055 | -11 | 0.4096 | ns | 0.997 | YES |
| PM2_STRUCT_KNN | 0.6461 | -0.0151 | -30 | 0.0168 | SIG | 0.997 | YES |
| P3_FENNEL | 0.6356 | -0.0256 | -51 | 0.0000 | SIG | 1.005 | YES |

**2wiki_clean** -- candidates vs production (exact McNemar, same 2000 queries)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| PM3_TOPOLOGY_C | 0.9485 | +0.0050 | 10 | 0.2529 | ns | 1.000 | NO |
| PM4_TOPOLOGY_C_NERW | 0.9485 | +0.0050 | 10 | 0.2604 | ns | 1.000 | NO |
| P2_METIS_STRONG | 0.9475 | +0.0040 | 8 | 0.3409 | ns | 1.001 | NO |
| P4_CE_NER_ONLY | 0.9470 | +0.0035 | 7 | 0.5154 | ns | 0.995 | NO |
| PM1_STRUCT_NERX | 0.9455 | +0.0020 | 4 | 0.7376 | ns | 1.004 | NO |
| P4_HYPERGRAPH_CE | 0.9410 | -0.0025 | -5 | 0.6718 | ns | 0.992 | NO |
| P4_CE_UNWEIGHTED | 0.9400 | -0.0035 | -7 | 0.5203 | ns | 0.997 | NO |
| P4_CE_LOCAL_ONLY | 0.9380 | -0.0055 | -11 | 0.3291 | ns | 0.993 | YES |
| P3_FENNEL | 0.9345 | -0.0090 | -18 | 0.0535 | ns | 1.003 | YES |
| PM2_STRUCT_KNN | 0.9315 | -0.0120 | -24 | 0.0210 | SIG | 1.000 | YES |
| PM0_STRUCT | 0.8990 | -0.0445 | -89 | 0.0000 | SIG | 1.008 | YES |

**musique_clean** -- candidates vs production (exact McNemar, same 2000 queries)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_HYPERGRAPH_CE | 0.9620 | -0.0015 | -3 | 0.8482 | ns | 0.997 | NO |
| P2_METIS_STRONG | 0.9605 | -0.0030 | -6 | 0.5323 | ns | 0.999 | NO |
| P4_CE_LOCAL_ONLY | 0.9565 | -0.0070 | -14 | 0.2150 | ns | 0.998 | YES |
| PM4_TOPOLOGY_C_NERW | 0.9560 | -0.0075 | -15 | 0.1053 | ns | 0.999 | YES |
| PM3_TOPOLOGY_C | 0.9555 | -0.0080 | -16 | 0.0599 | ns | 1.000 | YES |
| P4_CE_UNWEIGHTED | 0.9550 | -0.0085 | -17 | 0.1285 | ns | 0.998 | YES |
| PM1_STRUCT_NERX | 0.9515 | -0.0120 | -24 | 0.0149 | SIG | 1.000 | YES |
| PM2_STRUCT_KNN | 0.9465 | -0.0170 | -34 | 0.0024 | SIG | 1.001 | YES |
| P4_CE_NER_ONLY | 0.9420 | -0.0215 | -43 | 0.0005 | SIG | 0.996 | YES |
| PM0_STRUCT | 0.9225 | -0.0410 | -82 | 0.0000 | SIG | 1.003 | YES |
| P3_FENNEL | 0.9205 | -0.0430 | -86 | 0.0000 | SIG | 1.004 | YES |

**squad_clean** -- candidates vs production (exact McNemar, same 2000 queries)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| PM3_TOPOLOGY_C | 0.9855 | -0.0020 | -4 | 0.5235 | ns | 0.999 | NO |
| PM4_TOPOLOGY_C_NERW | 0.9855 | -0.0020 | -4 | 0.5413 | ns | 1.000 | NO |
| PM1_STRUCT_NERX | 0.9840 | -0.0035 | -7 | 0.1671 | ns | 0.998 | YES |
| P2_METIS_STRONG | 0.9840 | -0.0035 | -7 | 0.2100 | ns | 0.999 | YES |
| PM2_STRUCT_KNN | 0.9835 | -0.0040 | -8 | 0.1338 | ns | 1.007 | YES |
| PM0_STRUCT | 0.9785 | -0.0090 | -18 | 0.0014 | SIG | 1.010 | YES |
| P4_HYPERGRAPH_CE | 0.9785 | -0.0090 | -18 | 0.0114 | SIG | 0.990 | YES |
| P4_CE_LOCAL_ONLY | 0.9780 | -0.0095 | -19 | 0.0066 | SIG | 0.994 | YES |
| P3_FENNEL | 0.9755 | -0.0120 | -24 | 0.0007 | SIG | 0.998 | YES |
| P4_CE_UNWEIGHTED | 0.9730 | -0.0145 | -29 | 0.0000 | SIG | 0.990 | YES |
| P4_CE_NER_ONLY | 0.9635 | -0.0240 | -48 | 0.0000 | SIG | 0.991 | YES |

**hotpotqa_clean** -- candidates vs production (exact McNemar, same 2000 queries)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_CE_LOCAL_ONLY | 0.9655 | +0.0150 | 30 | 0.0000 | SIG | 0.994 | YES |
| PM2_STRUCT_KNN | 0.9615 | +0.0110 | 22 | 0.0032 | SIG | 1.000 | YES |
| P4_HYPERGRAPH_CE | 0.9585 | +0.0080 | 16 | 0.0226 | SIG | 0.992 | YES |
| P2_METIS_STRONG | 0.9520 | +0.0015 | 3 | 0.7608 | ns | 1.000 | NO |
| PM3_TOPOLOGY_C | 0.9515 | +0.0010 | 2 | 0.8776 | ns | 0.999 | NO |
| P4_CE_UNWEIGHTED | 0.9505 | +0.0000 | 0 | 1.0000 | ns | 0.998 | NO |
| PM4_TOPOLOGY_C_NERW | 0.9500 | -0.0005 | -1 | 1.0000 | ns | 0.999 | NO |
| PM0_STRUCT | 0.9490 | -0.0015 | -3 | 0.8099 | ns | 1.004 | NO |
| PM1_STRUCT_NERX | 0.9470 | -0.0035 | -7 | 0.3604 | ns | 1.000 | NO |
| P3_FENNEL | 0.9335 | -0.0170 | -34 | 0.0000 | SIG | 1.000 | YES |
| P4_CE_NER_ONLY | 0.9315 | -0.0190 | -38 | 0.0000 | SIG | 0.983 | YES |

**webqsp** -- candidates vs production (exact McNemar, same 1419 queries)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_CE_LOCAL_ONLY | 0.8069 | +0.0423 | 60 | 0.0000 | SIG | 0.998 | YES |
| P4_HYPERGRAPH_CE | 0.7886 | +0.0240 | 34 | 0.0011 | SIG | 0.999 | YES |
| PM2_STRUCT_KNN | 0.7808 | +0.0162 | 23 | 0.0265 | SIG | 1.000 | YES |
| PM0_STRUCT | 0.7766 | +0.0120 | 17 | 0.1038 | ns | 1.005 | NO |
| P4_CE_UNWEIGHTED | 0.7710 | +0.0064 | 9 | 0.3682 | ns | 1.000 | NO |
| PM1_STRUCT_NERX | 0.7696 | +0.0050 | 7 | 0.5203 | ns | 1.004 | NO |
| PM4_TOPOLOGY_C_NERW | 0.7681 | +0.0035 | 5 | 0.6201 | ns | 1.000 | NO |
| PM3_TOPOLOGY_C | 0.7590 | -0.0056 | -8 | 0.3891 | ns | 1.000 | NO |
| P2_METIS_STRONG | 0.7576 | -0.0070 | -10 | 0.2750 | ns | 1.001 | NO |
| P3_FENNEL | 0.7294 | -0.0352 | -50 | 0.0000 | SIG | 1.005 | YES |
| P4_CE_NER_ONLY | 0.6970 | -0.0676 | -96 | 0.0000 | SIG | 0.996 | YES |

## C4 -- effect decomposition

| corpus | frozen cell F6 | best edge substrate | edge effect | sig | best partitioning | partition effect | interaction |
|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | E3_STRUCT_NERX | +0.0015 | ns | P4_CE_LOCAL_ONLY | +0.0966 | MEASURED |
| 2wiki_clean | 0.9435 | E6_TOPOLOGY_C | +0.0030 | SIG | PM3_TOPOLOGY_C | +0.0050 | MEASURED |
| musique_clean | 0.9635 | E6_TOPOLOGY_C | +0.0020 | ns | P4_HYPERGRAPH_CE | -0.0015 | NOT_RUN |
| squad_clean | 0.9875 | E3_STRUCT_NERX | +0.0005 | ns | PM3_TOPOLOGY_C | -0.0020 | NOT_RUN |
| hotpotqa_clean | 0.9505 | M2_STRUCT_KNN_MATCHED | +0.0015 | ns | P4_CE_LOCAL_ONLY | +0.0150 | MEASURED |
| webqsp | 0.7646 | E3_STRUCT_NERX | +0.0043 | ns | P4_CE_LOCAL_ONLY | +0.0423 | MEASURED |


### C3 -- the joint 2x2 cells behind those interactions

Cell `a` must reproduce the frozen scoreboard exactly; that gate is what licenses reading `b`, `c` and `d`. `PARITY_vs_PHASE_C` additionally checks `a` against the independently written Phase-C driver, and `b` lands on the Phase-C value for the same partitioning.

| corpus | edge arm | partition arm | a | b | c | d | EDGE c-a | PARTITION b-a | JOINT d-a | INTERACTION | C3 parity | vs Phase C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | E6_TOPOLOGY_C | P4_CE_LOCAL_ONLY | 0.6612 | 0.7578 | 0.6597 | 0.7553 | -0.0015 | +0.0966 | +0.0941 | -0.0010 | EXACT | EXACT |
| 2wiki_clean | E6_TOPOLOGY_C | PM3_TOPOLOGY_C | 0.9435 | 0.9485 | 0.9465 | 0.9520 | +0.0030 | +0.0050 | +0.0085 | +0.0005 | EXACT | EXACT |
| hotpotqa_clean | E6_TOPOLOGY_C | P4_CE_LOCAL_ONLY | 0.9505 | 0.9655 | 0.9505 | 0.9670 | +0.0000 | +0.0150 | +0.0165 | +0.0015 | EXACT | EXACT |
| webqsp | E6_TOPOLOGY_C | P4_CE_LOCAL_ONLY | 0.7646 | 0.8069 | 0.7681 | 0.8090 | +0.0035 | +0.0423 | +0.0444 | -0.0014 | EXACT | EXACT |

