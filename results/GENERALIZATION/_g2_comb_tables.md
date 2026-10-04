# G2 FINAL L1 EXPERIMENT — COMBINED PARAMETER-FREE RESIDUAL (six corpora)

`BASE P50 + retrieval residual + structural residual`, all additive (BASE never evicted).


## 0. Substrate + protocol

| corpus | family | dev n | sample | docs | parts | BASE scope | scope % corpus | golds/q | parity |
|---|---|--:|---|--:|--:|--:|--:|--:|:--|
| MetaQA | KB | 1998 | val, hop-balanced | 40,151 | 401 | 5038 | 12.5% |  | EXACT |
| WebQSP | KB | 1419 | val+train | 781,485 | 7,814 | 5023 | 0.6% |  | NO_REFERENCE |
| 2Wiki | text | 2000 | val | 65,865 | 658 | 5028 | 7.6% |  | EXACT |
| MuSiQue | text | 2000 | val | 13,672 | 136 | 5042 | 36.9% |  | EXACT |
| HotpotQA | text | 2000 | val | 507,494 | 5,074 | 5082 | 1.0% |  | NO_REFERENCE |
| SQuAD | text | 2000 | val | 19,029 | 190 | 5039 | 26.5% |  | EXACT |

## 1. THE SIX-DATASET TABLE (ALL@scope, Δ vs BASE)

| corpus | BASE | RET32 | STRUCT32 | RET64 | STRUCT64 | COMBINED 32+32 | uniq added | scope growth |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **MetaQA** (KB) | 0.6587 | 0.6612 (+0.0025) | 0.6977 (+0.0390) | 0.6632 (+0.0045) | 0.7132 (+0.0546) | **0.7002 (+0.0415)** | 63.2 | 1.25% |
| **WebQSP** (KB) | 0.7618 | 0.7787 (+0.0169) | 0.7759 (+0.0141) | 0.7815 (+0.0197) | 0.7829 (+0.0211) | **0.7914 (+0.0296)** | 61.2 | 1.22% |
| **2Wiki** (text) | 0.9375 | 0.9480 (+0.0105) | 0.9390 (+0.0015) | 0.9500 (+0.0125) | 0.9390 (+0.0015) | **0.9490 (+0.0115)** | 58.4 | 1.16% |
| **MuSiQue** (text) | 0.9565 | 0.9770 (+0.0205) | 0.9585 (+0.0020) | 0.9805 (+0.0240) | 0.9590 (+0.0025) | **0.9775 (+0.0210)** | 58.3 | 1.16% |
| **HotpotQA** (text) | 0.9345 | 0.9630 (+0.0285) | 0.9380 (+0.0035) | 0.9680 (+0.0335) | 0.9395 (+0.0050) | **0.9635 (+0.0290)** | 63.0 | 1.24% |
| **SQuAD** (text) | 0.9805 | 0.9955 (+0.0150) | 0.9810 (+0.0005) | 0.9970 (+0.0165) | 0.9810 (+0.0005) | **0.9955 (+0.0150)** | 56.5 | 1.12% |

## 2. Queries newly ALL-covered / gold occurrences recovered

| corpus | RET32 | STRUCT32 | RET64 | STRUCT64 | COMBINED | RANDOM matched |
|---|--:|--:|--:|--:|--:|--:|
| **MetaQA** | 5 / 97 | 78 / 826 | 9 / 159 | 109 / 1092 | **83 / 886** | 0.2 / 15.4 |
| **WebQSP** | 24 / 135 | 20 / 75 | 28 / 194 | 30 / 122 | **42 / 194** | 0.0 / 0.4 |
| **2Wiki** | 21 / 32 | 3 / 3 | 25 / 36 | 3 / 3 | **23 / 34** | 0.2 / 0.2 |
| **MuSiQue** | 41 / 45 | 4 / 6 | 48 / 53 | 5 / 7 | **42 / 46** | 0.4 / 0.4 |
| **HotpotQA** | 57 / 63 | 7 / 7 | 67 / 73 | 10 / 11 | **58 / 64** | 0.0 / 0.0 |
| **SQuAD** | 30 / 30 | 1 / 1 | 33 / 33 | 1 / 1 | **30 / 30** | 0.0 / 0.0 |

## 3. Overlap + marginal complementarity

| corpus | \|RET32∩STRUCT32\| | overlap frac | ret-only | struct-only | unique | STRUCT after RET | RET after STRUCT | additivity gap |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **MetaQA** | 0.60 | 1.9% | 31.4 | 31.2 | 63.2 | +0.0390 (+78, p=0.0) | +0.0025 (+5, p=0.0625) | +0.0000 |
| **WebQSP** | 1.29 | 4.0% | 30.7 | 29.3 | 61.2 | +0.0127 (+18, p=1e-05) | +0.0155 (+22, p=0.0) | -0.0014 |
| **2Wiki** | 0.22 | 0.7% | 31.7 | 26.4 | 58.4 | +0.0010 (+2, p=0.5) | +0.0100 (+20, p=0.0) | -0.0005 |
| **MuSiQue** | 1.22 | 3.8% | 30.6 | 26.5 | 58.3 | +0.0005 (+1, p=1.0) | +0.0190 (+38, p=0.0) | -0.0015 |
| **HotpotQA** | 0.89 | 2.8% | 31.1 | 31.0 | 63.0 | +0.0005 (+1, p=1.0) | +0.0255 (+51, p=0.0) | -0.0030 |
| **SQuAD** | 0.96 | 3.0% | 31.0 | 24.5 | 56.5 | +0.0000 (+0, p=1.0) | +0.0145 (+29, p=0.0) | -0.0005 |

## 4. COMBINED vs the single-family 64-node budgets (matched total budget)

| corpus | COMBINED − RET64 | paired | COMBINED − STRUCT64 | paired |
|---|--:|:--|--:|:--|
| **MetaQA** | +0.0370 | net +74, p=0.0 **sig** | -0.0130 | net -26, p=1e-05 **sig** |
| **WebQSP** | +0.0099 | net +14, p=0.00258 **sig** | +0.0085 | net +12, p=0.0501 |
| **2Wiki** | -0.0010 | net -2, p=0.6875 | +0.0100 | net +20, p=0.0 **sig** |
| **MuSiQue** | -0.0030 | net -6, p=0.03125 **sig** | +0.0185 | net +37, p=0.0 **sig** |
| **HotpotQA** | -0.0045 | net -9, p=0.01172 **sig** | +0.0240 | net +48, p=0.0 **sig** |
| **SQuAD** | -0.0015 | net -3, p=0.25 | +0.0145 | net +29, p=0.0 **sig** |

## 5. ANY@scope + availability + monotonicity check

| corpus | BASE ANY | COMBINED ANY | struct avail/q | ret avail/q (rrf200) | q with <64 ret | worsened (all arms) |
|---|--:|--:|--:|--:|--:|--:|
| **MetaQA** | 0.9745 | 0.9855 | 89.3 | 262.2 | 1 | 0.0 |
| **WebQSP** | 0.9436 | 0.9556 | 75.2 | 175.1 | 70 | 0.0 |
| **2Wiki** | 1.0000 | 1.0000 | 68.3 | 212.4 | 111 | 0.0 |
| **MuSiQue** | 0.9990 | 0.9995 | 39.6 | 91.2 | 474 | 0.0 |
| **HotpotQA** | 0.9950 | 0.9980 | 106.3 | 249.4 | 28 | 0.0 |
| **SQuAD** | 0.9805 | 0.9955 | 45.9 | 126.3 | 159 | 0.0 |

## 6. Cost

| corpus | BASE s | RET build s | STRUCT traversal s | ×BASE | edges traversed | COMBINED increment over STRUCT32 |
|---|--:|--:|--:|--:|--:|--:|
| **MetaQA** | 11.1 | 1.57 | 71.1 | 6.4× | 3,008,656 | +1.57s (2.2%) |
| **WebQSP** | 22.9 | 2.04 | 47.6 | 2.1× | 2,358,159 | +2.04s (4.29%) |
| **2Wiki** | 14.3 | 2.02 | 21.9 | 1.5× | 1,202,171 | +2.02s (9.23%) |
| **MuSiQue** | 4.2 | 0.68 | 66.1 | 15.6× | 6,392,042 | +0.68s (1.03%) |
| **HotpotQA** | 9.9 | 1.24 | 51.5 | 5.2× | 3,785,828 | +1.24s (2.4%) |
| **SQuAD** | 6.2 | 0.76 | 203.8 | 32.7× | 28,362,633 | +0.76s (0.37%) |

## 7. MetaQA per-hop ALL

| arm | hop1 | hop2 | hop3 |
|---|--:|--:|--:|
| BASE | 0.9955 | 0.7237 | 0.2568 |
| RET32 | 0.9970 | 0.7267 | 0.2598 |
| STRUCT32 | 0.9955 | 0.7778 | 0.3198 |
| RET64 | 0.9970 | 0.7297 | 0.2628 |
| STRUCT64 | 0.9970 | 0.8078 | 0.3348 |
| COMBINED_32_32 | 0.9970 | 0.7808 | 0.3228 |

## 8. Retrieval-channel sensitivity (is rrf200 the right canonical channel?)

| corpus | RET32 rrf200 | RET32 dense200 | RET32 splade200 | COMBINED rrf200 | COMBINED dense200 |
|---|--:|--:|--:|--:|--:|
| **MetaQA** | +0.0025 | +0.0040 | +0.0005 | +0.0415 | +0.0420 |
| **WebQSP** | +0.0169 | +0.0078 | +0.0134 | +0.0296 | +0.0204 |
| **2Wiki** | +0.0105 | +0.0060 | +0.0090 | +0.0115 | +0.0075 |
| **MuSiQue** | +0.0205 | +0.0180 | +0.0130 | +0.0210 | +0.0190 |
| **HotpotQA** | +0.0285 | +0.0260 | +0.0180 | +0.0290 | +0.0280 |
| **SQuAD** | +0.0150 | +0.0130 | +0.0125 | +0.0150 | +0.0130 |
