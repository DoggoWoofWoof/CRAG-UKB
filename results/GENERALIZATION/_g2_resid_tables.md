## 1. BASE coverage (STEP 0) — authoritative canonical L1, parity-gated

| corpus | n dev q | docs | partitions | BASE_ANY_P50 | BASE_ALL_P50 | BASE_SCOPE_NODES | scope % of corpus | BASE_PARTITION_PARITY |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| **MetaQA** | 1998 | 40,151 | 401 | 0.9745 | 0.6587 | 5038 | 12.5 % | `EXACT` |
| **2wiki** | 2000 | 65,865 | 658 | 1.0000 | 0.9375 | 5028 | 7.6 % | `EXACT` |
| **MuSiQue** | 2000 | 13,672 | 136 | 0.9990 | 0.9565 | 5042 | 36.9 % | `EXACT` |
| SQuAD | — | — | — | — | — | — | — | NOT RUN |

MetaQA per-hop BASE ALL: hop1 0.9955 (n=666) · hop2 0.7237 (n=666) · hop3 0.2568 (n=666)

## 2/3/5. NODE arms — matched added-node budget M (STEPS 3B, 4, 6)

Every arm adds **exactly M nodes** on top of the same BASE P50 union; BASE is never evicted, so ΔALL ≥ 0 by construction. Bold = McNemar p<0.05 vs BASE. `(+n)` = net queries newly ALL-covered.


### MetaQA — BASE ALL 0.6587

| arm | M=8 | M=16 | M=32 | M=64 |
|---|--:|--:|--:|--:|
| RANDOM_NODE (5 seeds) | +0.0002 (+0) | +0.0002 (+1) | +0.0001 (+0) | +0.0001 (+0) |
| RETRIEVAL_NODE dense (exact) | +0.0010 (+2) | +0.0020 (+4) | **+0.0040 (+8)** | **+0.0070 (+14)** |
| RETRIEVAL_NODE splade200 | +0.0000 (+0) | +0.0005 (+1) | +0.0005 (+1) | +0.0010 (+2) |
| RETRIEVAL_NODE rrf200 | +0.0010 (+2) | +0.0010 (+2) | +0.0025 (+5) | **+0.0045 (+9)** |
| **STRUCT_NODE** RRF seeds | **+0.0215 (+43)** | **+0.0295 (+59)** | **+0.0390 (+78)** | **+0.0546 (+109)** |
| STRUCT_NODE dense seeds | **+0.0160 (+32)** | **+0.0215 (+43)** | **+0.0290 (+58)** | **+0.0475 (+95)** |
| STRUCT_NODE splade seeds | **+0.0195 (+39)** | **+0.0300 (+60)** | **+0.0360 (+72)** | **+0.0475 (+95)** |
| STRUCT_NODE RRF, beam 256 | **+0.0115 (+23)** | **+0.0230 (+46)** | **+0.0345 (+69)** | **+0.0450 (+90)** |

### 2wiki — BASE ALL 0.9375

| arm | M=8 | M=16 | M=32 | M=64 |
|---|--:|--:|--:|--:|
| RANDOM_NODE (5 seeds) | +0.0000 (+0) | +0.0000 (+0) | +0.0000 (+0) | +0.0001 (+0) |
| RETRIEVAL_NODE dense (exact) | **+0.0030 (+6)** | **+0.0050 (+10)** | **+0.0065 (+13)** | **+0.0075 (+15)** |
| RETRIEVAL_NODE splade200 | **+0.0070 (+14)** | **+0.0075 (+15)** | **+0.0090 (+18)** | **+0.0100 (+20)** |
| RETRIEVAL_NODE rrf200 | **+0.0065 (+13)** | **+0.0075 (+15)** | **+0.0105 (+21)** | **+0.0125 (+25)** |
| **STRUCT_NODE** RRF seeds | +0.0005 (+1) | +0.0010 (+2) | +0.0015 (+3) | +0.0015 (+3) |
| STRUCT_NODE dense seeds | +0.0005 (+1) | +0.0005 (+1) | +0.0005 (+1) | +0.0010 (+2) |
| STRUCT_NODE splade seeds | +0.0005 (+1) | +0.0005 (+1) | +0.0010 (+2) | +0.0015 (+3) |
| STRUCT_NODE RRF, beam 256 | +0.0000 (+0) | +0.0010 (+2) | +0.0010 (+2) | +0.0010 (+2) |

### MuSiQue — BASE ALL 0.9565

| arm | M=8 | M=16 | M=32 | M=64 |
|---|--:|--:|--:|--:|
| RANDOM_NODE (5 seeds) | +0.0001 (+1) | +0.0000 (+0) | +0.0002 (+0) | +0.0003 (+2) |
| RETRIEVAL_NODE dense (exact) | **+0.0100 (+20)** | **+0.0150 (+30)** | **+0.0190 (+38)** | **+0.0220 (+44)** |
| RETRIEVAL_NODE splade200 | **+0.0085 (+17)** | **+0.0110 (+22)** | **+0.0130 (+26)** | **+0.0170 (+34)** |
| RETRIEVAL_NODE rrf200 | **+0.0120 (+24)** | **+0.0150 (+30)** | **+0.0205 (+41)** | **+0.0240 (+48)** |
| **STRUCT_NODE** RRF seeds | +0.0010 (+2) | +0.0020 (+4) | +0.0020 (+4) | +0.0025 (+5) |
| STRUCT_NODE dense seeds | +0.0015 (+3) | +0.0020 (+4) | +0.0020 (+4) | +0.0025 (+5) |
| STRUCT_NODE splade seeds | +0.0010 (+2) | +0.0015 (+3) | +0.0025 (+5) | **+0.0030 (+6)** |
| STRUCT_NODE RRF, beam 256 | +0.0015 (+3) | **+0.0030 (+6)** | **+0.0035 (+7)** | **+0.0035 (+7)** |

## 6. PARTITION arms — additive out-of-P50 partitions (STEP 3C)

`STRUCT_PART` maps each out-of-BASE structural node to its canonical C partition, converts node rank j to the fixed RRF contribution 1/(K0+j), sums per partition and takes the top-Ps. BASE P50 is never evicted. Added-node counts are ~Ps × mean partition size, so the node-matched controls below are the load-bearing comparison.


### MetaQA — BASE ALL 0.6587, mean partition 100 nodes

| arm | Ps=1 | Ps=2 | Ps=4 | Ps=8 |
|---|--:|--:|--:|--:|
| RANDOM_PART (5 seeds) | +0.0003 (+0) | +0.0006 (+1) | +0.0007 (+1) | +0.0015 (+3) |
| RETRIEVAL_PART (fused rank 50..50+Ps) | **+0.0030 (+6)** | **+0.0045 (+9)** | **+0.0080 (+16)** | **+0.0160 (+32)** |
| **STRUCT_PART** RRF seeds | +0.0025 (+5) | **+0.0075 (+15)** | **+0.0105 (+21)** | **+0.0145 (+29)** |
| STRUCT_PART RRF, beam 256 | +0.0025 (+5) | **+0.0050 (+10)** | **+0.0070 (+14)** | **+0.0125 (+25)** |
| · node-matched RANDOM | +0.0002 (+0) | +0.0003 (+1) | +0.0012 (+0) | +0.0019 (+4) |
| · node-matched RETRIEVAL dense | **+0.0105 (+21)** | **+0.0155 (+31)** | **+0.0245 (+49)** | **+0.0370 (+74)** |
| · node-matched STRUCT_NODE | **+0.0666 (+133)** | **+0.0676 (+135)** | **+0.0676 (+135)** | **+0.0676 (+135)** |
| *added nodes* | 100 | 201 | 401 | 801 |
| *scope growth* | 2.0 % | 4.0 % | 8.0 % | 15.9 % |

### 2wiki — BASE ALL 0.9375, mean partition 100 nodes

| arm | Ps=1 | Ps=2 | Ps=4 | Ps=8 |
|---|--:|--:|--:|--:|
| RANDOM_PART (5 seeds) | +0.0000 (+0) | +0.0005 (+1) | +0.0004 (+0) | +0.0003 (+1) |
| RETRIEVAL_PART (fused rank 50..50+Ps) | +0.0000 (+0) | +0.0020 (+4) | **+0.0030 (+6)** | **+0.0065 (+13)** |
| **STRUCT_PART** RRF seeds | +0.0005 (+1) | +0.0005 (+1) | +0.0015 (+3) | +0.0020 (+4) |
| STRUCT_PART RRF, beam 256 | +0.0000 (+0) | +0.0005 (+1) | +0.0015 (+3) | +0.0025 (+5) |
| · node-matched RANDOM | +0.0001 (+0) | +0.0001 (+0) | +0.0004 (+0) | +0.0006 (+1) |
| · node-matched RETRIEVAL dense | **+0.0100 (+20)** | **+0.0135 (+27)** | **+0.0160 (+32)** | **+0.0200 (+40)** |
| · node-matched STRUCT_NODE | +0.0015 (+3) | +0.0015 (+3) | +0.0015 (+3) | +0.0015 (+3) |
| *added nodes* | 89 | 177 | 351 | 692 |
| *scope growth* | 1.8 % | 3.5 % | 7.0 % | 13.8 % |

### MuSiQue — BASE ALL 0.9565, mean partition 100 nodes

| arm | Ps=1 | Ps=2 | Ps=4 | Ps=8 |
|---|--:|--:|--:|--:|
| RANDOM_PART (5 seeds) | +0.0008 (+2) | +0.0007 (+1) | +0.0014 (+2) | **+0.0037 (+8)** |
| RETRIEVAL_PART (fused rank 50..50+Ps) | +0.0015 (+3) | **+0.0035 (+7)** | **+0.0055 (+11)** | **+0.0090 (+18)** |
| **STRUCT_PART** RRF seeds | +0.0005 (+1) | +0.0025 (+5) | **+0.0055 (+11)** | **+0.0085 (+17)** |
| STRUCT_PART RRF, beam 256 | +0.0020 (+4) | **+0.0030 (+6)** | **+0.0045 (+9)** | **+0.0085 (+17)** |
| · node-matched RANDOM | +0.0008 (+1) | +0.0006 (+2) | +0.0016 (+4) | **+0.0041 (+11)** |
| · node-matched RETRIEVAL dense | **+0.0260 (+52)** | **+0.0290 (+58)** | **+0.0340 (+68)** | **+0.0355 (+71)** |
| · node-matched STRUCT_NODE | +0.0025 (+5) | +0.0025 (+5) | +0.0025 (+5) | +0.0025 (+5) |
| *added nodes* | 100 | 199 | 396 | 782 |
| *scope growth* | 2.0 % | 3.9 % | 7.8 % | 15.5 % |

## 7. STEP 7 — structure-specific gain at matched scope

| corpus | budget | ΔALL STRUCT | ΔALL RANDOM | ΔALL RETRIEVAL | over RANDOM | over RETRIEVAL |
|---|---|--:|--:|--:|--:|--:|
| MetaQA | NODE M=8 | +0.0215 | +0.0002 | +0.0010 (dense) | +0.0213 | +0.0205 |
| MetaQA | NODE M=16 | +0.0295 | +0.0002 | +0.0020 (dense) | +0.0293 | +0.0275 |
| MetaQA | NODE M=32 | +0.0390 | +0.0001 | +0.0040 (dense) | +0.0389 | +0.0350 |
| MetaQA | NODE M=64 | +0.0546 | +0.0001 | +0.0070 (dense) | +0.0545 | +0.0476 |
| MetaQA | PART Ps=1 | +0.0025 | +0.0003 | +0.0030 | +0.0022 | -0.0005 |
| MetaQA | PART Ps=2 | +0.0075 | +0.0006 | +0.0045 | +0.0069 | +0.0030 |
| MetaQA | PART Ps=4 | +0.0105 | +0.0007 | +0.0080 | +0.0098 | +0.0025 |
| MetaQA | PART Ps=8 | +0.0145 | +0.0015 | +0.0160 | +0.0130 | -0.0015 |
| 2wiki | NODE M=8 | +0.0005 | +0.0000 | +0.0070 (splade200) | +0.0005 | -0.0065 |
| 2wiki | NODE M=16 | +0.0010 | +0.0000 | +0.0075 (splade200) | +0.0010 | -0.0065 |
| 2wiki | NODE M=32 | +0.0015 | +0.0000 | +0.0105 (rrf200) | +0.0015 | -0.0090 |
| 2wiki | NODE M=64 | +0.0015 | +0.0001 | +0.0125 (rrf200) | +0.0014 | -0.0110 |
| 2wiki | PART Ps=1 | +0.0005 | +0.0000 | +0.0000 | +0.0005 | +0.0005 |
| 2wiki | PART Ps=2 | +0.0005 | +0.0005 | +0.0020 | +0.0000 | -0.0015 |
| 2wiki | PART Ps=4 | +0.0015 | +0.0004 | +0.0030 | +0.0011 | -0.0015 |
| 2wiki | PART Ps=8 | +0.0020 | +0.0003 | +0.0065 | +0.0017 | -0.0045 |
| MuSiQue | NODE M=8 | +0.0010 | +0.0001 | +0.0120 (rrf200) | +0.0009 | -0.0110 |
| MuSiQue | NODE M=16 | +0.0020 | +0.0000 | +0.0150 (dense) | +0.0020 | -0.0130 |
| MuSiQue | NODE M=32 | +0.0020 | +0.0002 | +0.0205 (rrf200) | +0.0018 | -0.0185 |
| MuSiQue | NODE M=64 | +0.0025 | +0.0003 | +0.0240 (rrf200) | +0.0022 | -0.0215 |
| MuSiQue | PART Ps=1 | +0.0005 | +0.0008 | +0.0015 | -0.0003 | -0.0010 |
| MuSiQue | PART Ps=2 | +0.0025 | +0.0007 | +0.0035 | +0.0018 | -0.0010 |
| MuSiQue | PART Ps=4 | +0.0055 | +0.0014 | +0.0055 | +0.0041 | +0.0000 |
| MuSiQue | PART Ps=8 | +0.0085 | +0.0037 | +0.0090 | +0.0048 | -0.0005 |

## 8. The central MetaQA hop table (STEP 6)

| arm | added nodes | hop1 ALL | hop2 ALL | hop3 ALL |
|---|--:|--:|--:|--:|
| BASE | 0.0 | 0.9955 | 0.7237 | 0.2568 |
| PREPARTITION (A2_M32) | 0 (substitution) | 0.9955 | 0.7417 | 0.2643 |
| RANDOM_NODE +32 | 31.988 | 0.9955 | 0.7237 | 0.2571 |
| RETRIEVAL_NODE +32 (dense) | 32.0 | 0.9955 | 0.7297 | 0.2628 |
| **STRUCT_NODE +32** | 31.85 | 0.9955 | 0.7778 | 0.3198 |
| RANDOM_PART +4 | 400.152 | 0.9955 | 0.7246 | 0.2580 |
| RETRIEVAL_PART +4 | 401.96 | 0.9970 | 0.7372 | 0.2658 |
| **STRUCT_PART +4** | 401.05 | 0.9970 | 0.7417 | 0.2688 |
| *PRIVILEGED_OLD (UNBOUNDED)* | 144 | 0.9970 | 0.8228 | 0.4805 |
| *PRIVILEGED_OLD (M32)* | 28 | 0.9970 | 0.7808 | 0.3348 |

## 9. Cross-dataset table at one global operating point (STEP 10)

| corpus | BASE ALL | PREPARTITION best | STRUCT_NODE +32 | STRUCT_PART +4 | RETRIEVAL_NODE +32 | RANDOM_NODE +32 |
|---|--:|--:|--:|--:|--:|--:|
| **MetaQA** | 0.6587 | 0.6672 (+17, A2_M32) | **+0.0390 (+78)** | **+0.0105 (+21)** | **+0.0040 (+8)** | +0.0001 (+0) |
| **2wiki** | 0.9375 | 0.9360 (-3, A2dOnly_M32) | +0.0015 (+3) | +0.0015 (+3) | **+0.0065 (+13)** | +0.0000 (+0) |
| **MuSiQue** | 0.9565 | 0.9580 (+3, A2sOnly_M32) | +0.0020 (+4) | **+0.0055 (+11)** | **+0.0190 (+38)** | +0.0002 (+0) |

## 10. Cost table (STEP 9)

| corpus | method | edges traversed | structural nodes evaluated | wall s | × BASE | residual available/q |
|---|---|--:|--:|--:|--:|--:|
| **MetaQA** | BASE router | 0 | 0 | 9.4 | 1.0 | — |
| | STRUCT_RRF_beam64 | 3,008,656 | 293,624 | 129.9 | 13.8× | 89.3 |
| | STRUCT_DENSE_beam64 | 2,993,565 | 285,392 | 96.6 | 10.3× | 87.0 |
| | STRUCT_SPLADE_beam64 | 3,136,702 | 304,195 | 92.3 | 9.8× | 94.9 |
| | STRUCT_RRF_beam256 | 6,300,017 | 507,042 | 138.1 | 14.7× | 188.3 |
| **2wiki** | BASE router | 0 | 0 | 14.0 | 1.0 | — |
| | STRUCT_RRF_beam64 | 1,202,171 | 211,481 | 58.5 | 4.2× | 68.3 |
| | STRUCT_DENSE_beam64 | 1,195,215 | 211,553 | 48.8 | 3.5× | 70.0 |
| | STRUCT_SPLADE_beam64 | 1,252,023 | 219,504 | 40.5 | 2.9× | 72.9 |
| | STRUCT_RRF_beam256 | 2,136,210 | 389,490 | 71.5 | 5.1× | 146.6 |
| **MuSiQue** | BASE router | 0 | 0 | 23.2 | 1.0 | — |
| | STRUCT_RRF_beam64 | 6,392,042 | 305,927 | 188.7 | 8.1× | 39.6 |
| | STRUCT_DENSE_beam64 | 6,460,605 | 304,583 | 144.3 | 6.2× | 40.2 |
| | STRUCT_SPLADE_beam64 | 6,432,472 | 307,086 | 118.1 | 5.1× | 40.8 |
| | STRUCT_RRF_beam256 | 14,614,407 | 498,114 | 170.2 | 7.3× | 89.7 |

## 11. Exact decomposition of the old MetaQA gain (STEP 5)

**Part 1 — exact reproduction of the published run.** `MATCHES_PUBLISHED = True`

| hop | mine P50→UNION | published P50→UNION | added nodes (mine / published) |
|---|--:|--:|--:|
| hop1 | 1.0000 → 1.0000 | 1.0000 → 1.0000 | 402 / 402 |
| hop2 | 0.7080 → 0.9280 | 0.7080 → 0.9280 | 316 / 316 |
| hop3 | 0.2880 → 0.4240 | 0.2880 → 0.4240 | 436 / 436 |

**Part 2 — controlled decomposition** (same traversal implementation, main dev sample n=1998, BASE ALL 0.6587). Only the seed source and the adjacency vary.

| seeds | adjacency | M=8 | M=16 | M=32 | M=64 | UNBOUNDED | added (unb.) | hop2 | hop3 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| — | BASE | 0.6587 | 0.6587 | 0.6587 | 0.6587 | 0.6587 | 0 | 0.7237 | 0.2568 |
| PRIVILEGED_bracket | KB_privileged | 0.6857 | 0.6952 | 0.7042 | 0.7247 | 0.7668 | 144 | 0.8228 | 0.4805 |
| UNIVERSAL_RRF | KB_privileged | 0.6702 | 0.6817 | 0.6932 | 0.7037 | 0.7227 | 188 | 0.7943 | 0.3784 |
| PRIVILEGED_bracket | MASTER_universal | 0.6857 | 0.6952 | 0.7042 | 0.7247 | 0.7668 | 144 | 0.8228 | 0.4805 |
| UNIVERSAL_RRF | MASTER_universal | 0.6702 | 0.6817 | 0.6932 | 0.7037 | 0.7227 | 188 | 0.7943 | 0.3784 |
| UNIVERSAL_DENSE | MASTER_universal | 0.6717 | 0.6772 | 0.6867 | 0.6937 | 0.7097 | 186 | 0.7658 | 0.3679 |
| UNIVERSAL_SPLADE | MASTER_universal | 0.6687 | 0.6787 | 0.6892 | 0.7032 | 0.7217 | 191 | 0.7883 | 0.3814 |