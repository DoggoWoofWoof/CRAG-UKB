# L1 CORE-EXIT PROVENANCE AUDIT -- TABLES

Corpora with a completed audit pass: **6/6** (MetaQA, WebQSP, 2Wiki, MuSiQue, Hotpot, SQuAD).


## STEP 1 -- EDGE FAMILY LEDGER

### 1a. Corpus-level: which families exist in the FROZEN traversal adjacency

The traversal adjacency is `master_nodes_{ds}.json` `neighbors`, i.e. **STRUCT**. `A = gte_qwen/graph.pt` is `STRUCT u KNN`; `KNN = A \ STRUCT`; `NER = ner_edges_w_df25.pkl`. Undirected canonical keys, the existing `ac_edge_recon` convention.

| corpus | nodes | traversal edges | also NER | frac NER | also KNN | not in A | STRUCT provenance (canonical manifest) |
|---|--:|--:|--:|--:|--:|--:|---|
| MetaQA | 40,151 | 109,480 | 59,403 | **54.26%** | 0 | 0 | `structural_native` / 9 rel |
| WebQSP | 781,485 | 1,638,066 | 240,485 | **14.68%** | 0 | 0 | `structural_derived_from_freebase` / 6094 rel |
| 2Wiki | 65,865 | 126,070 | 28,890 | **22.92%** | 0 | 0 | `structural_native` / `hyperlink` / 1 rel |
| MuSiQue | 13,672 | 51,543 | 6,156 | **11.94%** | 0 | 0 | `structural_derived` / `title_mention` / 1 rel |
| Hotpot | 507,494 | 3,426,945 | 283,658 | **8.28%** | 0 | 34,425 | `structural_native` / `hyperlink` / 1 rel |
| SQuAD | 19,029 | 694,580 | 19,027 | **2.74%** | 0 | 0 | `structural_derived` / `title_mention` / 1 rel |

**KNN is structurally absent from L1 traversal on all six corpora (0 edges everywhere).** The frozen beam expands `nd.neighbors` only, so the semantic/Qwen-kNN family can never appear in a core-exit transition. The only live provenance axis is `STRUCT_ONLY` vs `STRUCT_AND_NER`.


### 1b. Per-query core-exit TRANSITION counts, by family

| corpus | queries | parity | edges/q | core-exit/q | core-exit NER/q | core-exit STRUCT_ONLY/q | frac NER (mean) | frac NER (median) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| MetaQA | 1998 | 1998/1998 | 1,411 | 725 | 233 | 492 | **0.3710** | 0.3361 |
| WebQSP | 1405 | 1419/1419 | 1,494 | 737 | 109 | 628 | **0.1737** | 0.1465 |
| 2Wiki | 1914 | 2000/2000 | 537 | 304 | 85 | 220 | **0.3557** | 0.2966 |
| MuSiQue | 1981 | 2000/2000 | 3,167 | 2,766 | 207 | 2,559 | **0.0935** | 0.0768 |
| Hotpot | 2000 | 2000/2000 | 1,404 | 710 | 90 | 620 | **0.1456** | 0.1241 |
| SQuAD | 1705 | 2000/2000 | 14,088 | 11,179 | 326 | 10,852 | **0.0308** | 0.0273 |

### 1c. Core-exit transitions by hop, and their NER share

| corpus | hop1 | hop2 | hop3 | NER frac hop1 | NER frac hop2 | NER frac hop3 |
|---|--:|--:|--:|--:|--:|--:|
| MetaQA | 48,649 | 816,379 | 583,738 | 0.6709 | 0.2579 | 0.3824 |
| WebQSP | 42,987 | 544,706 | 447,313 | 0.2179 | 0.1187 | 0.1762 |
| 2Wiki | 21,859 | 327,738 | 232,324 | 0.5761 | 0.2022 | 0.3569 |
| MuSiQue | 107,762 | 3,110,039 | 2,261,407 | 0.1450 | 0.0679 | 0.0806 |
| Hotpot | 65,744 | 1,059,702 | 294,539 | 0.2595 | 0.0983 | 0.1964 |
| SQuAD | 399,983 | 11,229,042 | 7,430,863 | 0.0434 | 0.0286 | 0.0293 |

MetaQA per-query, split by question hop:

| question hop | core-exit/q | core-exit NER/q |
|---|--:|--:|
| hop1 | 695 | 220 |
| hop2 | 693 | 236 |
| hop3 | 787 | 245 |

## STEP 2 -- GOOD/BAD SWAP BY EDGE FAMILY

Pool `A1_SRC`. GOOD/BAD definitions verbatim from the prior phase (`_l1kt_partb.run`): at identical K(q) the SRC pool evicts SAFE candidates and admits new ones; a labelled pair has exactly one required-and-missing side. Every family is scored on the **same** pair population -- the pools are family-independent, only the score changes. `AUC > 0.5` means the family separates GOOD from BAD in the right direction.


### SRC_ALL

| row | n GOOD | n BAD | mean GOOD margin | mean BAD margin | AUC | sign acc | AUC (both have evidence) |
|---|--:|--:|--:|--:|--:|--:|--:|
| MetaQA hop2 | 5399 | 2908 | 0.8576 | 0.1563 | **0.8429** | 0.7999 | -- |
| MetaQA hop3 | 21721 | 18998 | 0.7238 | 0.5809 | **0.6081** | 0.5657 | -- |
| WebQSP | 2826 | 2597 | 1.9292 | 1.6410 | **0.6693** | 0.5916 | 0.8402 |
| 2Wiki | 268 | 560 | 1.8486 | 1.8017 | **0.5066** | 0.3587 | 0.5417 |
| MuSiQue | 65 | 417 | 0.5056 | 0.2350 | **0.5781** | 0.3651 | 0.5345 |
| Hotpot | 384 | 2116 | 1.5236 | 1.6847 | **0.4644** | 0.2432 | 0.6408 |
| SQuAD | 0 | 300 | -- | 1.6784 | **--** | 0.0633 | -- |

### SRC_NER

| row | n GOOD | n BAD | mean GOOD margin | mean BAD margin | AUC | sign acc | AUC (both have evidence) |
|---|--:|--:|--:|--:|--:|--:|--:|
| MetaQA hop2 | 5399 | 2908 | 1.4456 | 0.5341 | **0.7140** | 0.6084 | -- |
| MetaQA hop3 | 21721 | 18998 | 1.2112 | 0.8323 | **0.5852** | 0.4295 | -- |
| WebQSP | 2826 | 2597 | 0.5161 | 0.3137 | **0.5422** | 0.1422 | 0.6764 |
| 2Wiki | 268 | 560 | 1.2106 | 0.5128 | **0.6268** | 0.2041 | 0.0000 |
| MuSiQue | 65 | 417 | 0.3176 | 0.2374 | **0.5203** | 0.1680 | 0.9032 |
| Hotpot | 384 | 2116 | 1.0115 | 0.2569 | **0.6444** | 0.1172 | 0.4032 |
| SQuAD | 0 | 300 | -- | 0.4259 | **--** | 0.0300 | -- |

### SRC_STRUCT_ONLY

| row | n GOOD | n BAD | mean GOOD margin | mean BAD margin | AUC | sign acc | AUC (both have evidence) |
|---|--:|--:|--:|--:|--:|--:|--:|
| MetaQA hop2 | 5399 | 2908 | 0.2881 | -0.1256 | **0.6570** | 0.5494 | -- |
| MetaQA hop3 | 21721 | 18998 | 0.4153 | 0.2498 | **0.5674** | 0.4810 | -- |
| WebQSP | 2826 | 2597 | 1.5679 | 1.4043 | **0.6251** | 0.5148 | 0.8179 |
| 2Wiki | 268 | 560 | 0.8819 | 1.3459 | **0.3895** | 0.1944 | 0.3088 |
| MuSiQue | 65 | 417 | 0.4893 | 0.2715 | **0.5283** | 0.3465 | 0.4751 |
| Hotpot | 384 | 2116 | 0.9692 | 1.5182 | **0.4088** | 0.1924 | 0.5997 |
| SQuAD | 0 | 300 | -- | 1.6776 | **--** | 0.0633 | -- |

### ALL_NER

| row | n GOOD | n BAD | mean GOOD margin | mean BAD margin | AUC | sign acc | AUC (both have evidence) |
|---|--:|--:|--:|--:|--:|--:|--:|
| MetaQA hop2 | 5399 | 2908 | 0.5908 | -0.0533 | **0.6920** | 0.6645 | -- |
| MetaQA hop3 | 21721 | 18998 | 0.3099 | 0.1449 | **0.5624** | 0.5179 | -- |
| WebQSP | 2826 | 2597 | 0.5581 | 0.0569 | **0.6104** | 0.3019 | 0.7698 |
| 2Wiki | 268 | 560 | 1.3403 | 0.5073 | **0.6494** | 0.2971 | 0.5714 |
| MuSiQue | 65 | 417 | 0.3052 | 0.1205 | **0.5288** | 0.2614 | 0.5882 |
| Hotpot | 384 | 2116 | 0.6416 | 0.1450 | **0.5688** | 0.1896 | 0.6248 |
| SQuAD | 0 | 300 | -- | 0.3055 | **--** | 0.1267 | -- |

### ALL_STRUCT_ONLY

| row | n GOOD | n BAD | mean GOOD margin | mean BAD margin | AUC | sign acc | AUC (both have evidence) |
|---|--:|--:|--:|--:|--:|--:|--:|
| MetaQA hop2 | 5399 | 2908 | -0.0200 | -0.1155 | **0.6487** | 0.5784 | -- |
| MetaQA hop3 | 21721 | 18998 | 0.2088 | 0.0481 | **0.5833** | 0.5161 | -- |
| WebQSP | 2826 | 2597 | 0.2877 | 0.9806 | **0.4920** | 0.5584 | 0.8971 |
| 2Wiki | 268 | 560 | -0.1491 | 0.7193 | **0.3890** | 0.3116 | 0.6797 |
| MuSiQue | 65 | 417 | 0.2108 | 0.2298 | **0.4381** | 0.3278 | 0.4253 |
| Hotpot | 384 | 2116 | 0.4678 | 1.1277 | **0.4119** | 0.3320 | 0.6661 |
| SQuAD | 0 | 300 | -- | 1.0537 | **--** | 0.2067 | -- |

### IS ANY EDGE FAMILY SIGN-CONSISTENT ACROSS CORPORA?

A family is sign-consistent if its AUC lands on the same side of 0.5 on every eligible row (a row is eligible when it has at least one GOOD and one BAD pair, so the AUC is defined).

| family | eligible rows | AUC > 0.5 | AUC < 0.5 | min AUC | max AUC | SIGN-CONSISTENT? |
|---|--:|--:|--:|--:|--:|---|
| SRC_ALL | 6 | 5 | 1 | 0.4644 | 0.8429 | NO |
| SRC_NER | 6 | 6 | 0 | 0.5203 | 0.7140 | **YES** |
| SRC_STRUCT_ONLY | 6 | 4 | 2 | 0.3895 | 0.6570 | NO |
| ALL_NER | 6 | 6 | 0 | 0.5288 | 0.6920 | **YES** |
| ALL_STRUCT_ONLY | 6 | 2 | 4 | 0.3890 | 0.6487 | NO |

## STEP 3 -- EXCLUSIVE SUPPORT

Each admitted candidate is bucketed by which families supply its core-exit evidence. `gain/loss` is `good_swaps / bad_swaps` -- above 1 means candidates in that bucket are more often the required-and-missing side than the collateral side.

| corpus | ONLY_NER<br>good / bad / ratio | ONLY_STRUCT<br>good / bad / ratio | MULTI_FAMILY<br>good / bad / ratio | NO_CORE_EXIT_EVIDENCE<br>good / bad / ratio |
|---|---|---|---|---|
| MetaQA | 5104 / 4241 / 1.204 | 7769 / 8278 / 0.939 | 14247 / 9421 / 1.512 | 0 / 0 / -- |
| WebQSP | 434 / 277 / 1.567 | 2172 / 2117 / 1.026 | 220 / 203 / 1.084 | 0 / 0 / -- |
| 2Wiki | 111 / 110 / 1.009 | 126 / 407 / 0.310 | 31 / 43 / 0.721 | 0 / 0 / -- |
| MuSiQue | 0 / 3 / 0.000 | 52 / 292 / 0.178 | 13 / 122 / 0.107 | 0 / 0 / -- |
| Hotpot | 98 / 229 / 0.428 | 197 / 1762 / 0.112 | 89 / 125 / 0.712 | 0 / 0 / -- |
| SQuAD | 0 / 0 / -- | 0 / 233 / 0.000 | 0 / 67 / 0.000 | 0 / 0 / -- |

## STEP 4 -- SOURCE QUALITY (diagnostic only)

`p_required` = P(admitted candidate is required-and-missing | bin), attributed to its best supporting edge in that family. Bin edges are fixed by the frozen contract (`prot` = canonical ranks 0..43 in four equal blocks, `bnd` = 44..49), not searched. A universal lever would need the same monotone direction on every corpus.


### core_source -- source in protected core (1) vs not (0)

| corpus | family | 0<br>p_req (n) | 1<br>p_req (n) |
|---|---|---|---|
| MetaQA | NER | 0.0162 (14403) | 0.0202 (28730) |
| MetaQA | STRUCT_ONLY | 0.0140 (15517) | 0.0181 (34959) |
| WebQSP | NER | 0.0079 (2397) | 0.0039 (5398) |
| WebQSP | STRUCT_ONLY | 0.0039 (2045) | 0.0035 (24015) |
| 2Wiki | NER | 0.0003 (3436) | 0.0006 (8739) |
| 2Wiki | STRUCT_ONLY | 0.0002 (4111) | 0.0001 (25821) |
| MuSiQue | NER | 0.0009 (1146) | 0.0005 (4372) |
| MuSiQue | STRUCT_ONLY | 0.0014 (2073) | 0.0003 (15811) |
| Hotpot | NER | 0.0003 (3755) | 0.0004 (11383) |
| Hotpot | STRUCT_ONLY | 0.0003 (6079) | 0.0001 (58920) |
| SQuAD | NER | 0.0000 (1406) | 0.0000 (2846) |
| SQuAD | STRUCT_ONLY | 0.0000 (3230) | 0.0000 (13395) |

### hop -- hop of the supporting edge

| corpus | family | 1<br>p_req (n) | 2<br>p_req (n) | 3<br>p_req (n) |
|---|---|---|---|---|
| MetaQA | NER | 0.0210 (524) | 0.0291 (13687) | 0.0140 (28922) |
| MetaQA | STRUCT_ONLY | 0.0109 (917) | 0.0226 (22742) | 0.0122 (26817) |
| WebQSP | NER | 0.0000 (40) | 0.0038 (2346) | 0.0057 (5409) |
| WebQSP | STRUCT_ONLY | 0.0190 (263) | 0.0023 (11958) | 0.0043 (13839) |
| 2Wiki | NER | 0.0000 (39) | 0.0006 (3213) | 0.0004 (8923) |
| 2Wiki | STRUCT_ONLY | 0.0000 (56) | 0.0001 (14369) | 0.0001 (15507) |
| MuSiQue | NER | 0.0000 (14) | 0.0013 (1520) | 0.0003 (3984) |
| MuSiQue | STRUCT_ONLY | 0.0000 (192) | 0.0002 (6310) | 0.0005 (11382) |
| Hotpot | NER | 0.0000 (325) | 0.0006 (8491) | 0.0000 (6322) |
| Hotpot | STRUCT_ONLY | 0.0000 (1818) | 0.0002 (44976) | 0.0001 (18205) |
| SQuAD | NER | 0.0000 (14) | 0.0000 (1022) | 0.0000 (3216) |
| SQuAD | STRUCT_ONLY | 0.0000 (466) | 0.0000 (4961) | 0.0000 (11198) |

### src_cpos_quartile -- source partition canonical rank

| corpus | family | BND<br>p_req (n) | OUT_OF_P50<br>p_req (n) | PROT_Q0<br>p_req (n) | PROT_Q1<br>p_req (n) | PROT_Q2<br>p_req (n) | PROT_Q3<br>p_req (n) |
|---|---|---|---|---|---|---|---|
| MetaQA | NER | 0.0204 (393) | 0.0161 (14010) | 0.0258 (11628) | 0.0218 (7706) | 0.0108 (5358) | 0.0134 (4038) |
| MetaQA | STRUCT_ONLY | 0.0115 (1132) | 0.0143 (14385) | 0.0195 (15429) | 0.0205 (9453) | 0.0143 (6009) | 0.0128 (4068) |
| WebQSP | NER | 0.0000 (22) | 0.0080 (2375) | 0.0036 (2809) | 0.0047 (1290) | 0.0050 (796) | 0.0020 (503) |
| WebQSP | STRUCT_ONLY | 0.0000 (61) | 0.0040 (1984) | 0.0045 (13432) | 0.0031 (5494) | 0.0016 (3117) | 0.0010 (1972) |
| 2Wiki | NER | 0.0000 (50) | 0.0003 (3386) | 0.0008 (3766) | 0.0004 (2546) | 0.0000 (1421) | 0.0010 (1006) |
| 2Wiki | STRUCT_ONLY | 0.0000 (156) | 0.0003 (3955) | 0.0002 (12068) | 0.0000 (7465) | 0.0000 (3789) | 0.0000 (2499) |
| MuSiQue | NER | 0.0000 (82) | 0.0009 (1064) | 0.0009 (2231) | 0.0000 (1006) | 0.0000 (650) | 0.0000 (485) |
| MuSiQue | STRUCT_ONLY | 0.0040 (248) | 0.0011 (1825) | 0.0000 (8833) | 0.0011 (3496) | 0.0000 (2029) | 0.0000 (1453) |
| Hotpot | NER | 0.0116 (86) | 0.0000 (3669) | 0.0003 (3559) | 0.0000 (3146) | 0.0011 (2617) | 0.0000 (2061) |
| Hotpot | STRUCT_ONLY | 0.0000 (337) | 0.0003 (5742) | 0.0001 (20556) | 0.0001 (15582) | 0.0001 (12591) | 0.0000 (10191) |
| SQuAD | NER | 0.0000 (72) | 0.0000 (1334) | 0.0000 (1346) | 0.0000 (737) | 0.0000 (433) | 0.0000 (330) |
| SQuAD | STRUCT_ONLY | 0.0000 (341) | 0.0000 (2889) | 0.0000 (6288) | 0.0000 (3317) | 0.0000 (2282) | 0.0000 (1508) |

### src_conf_quartile -- source retrieval confidence quartile (T1_SOURCE)

| corpus | family | 0<br>p_req (n) | 1<br>p_req (n) | 2<br>p_req (n) | 3<br>p_req (n) |
|---|---|---|---|---|---|
| MetaQA | NER | 0.0137 (12089) | 0.0195 (11460) | 0.0208 (10526) | 0.0226 (9058) |
| MetaQA | STRUCT_ONLY | 0.0234 (8116) | 0.0190 (11238) | 0.0164 (11665) | 0.0132 (19457) |
| WebQSP | NER | 0.0039 (2075) | 0.0043 (2102) | 0.0054 (1869) | 0.0074 (1749) |
| WebQSP | STRUCT_ONLY | 0.0035 (6805) | 0.0016 (5736) | 0.0030 (5248) | 0.0052 (8271) |
| 2Wiki | NER | 0.0007 (2698) | 0.0000 (2700) | 0.0003 (3196) | 0.0008 (3581) |
| 2Wiki | STRUCT_ONLY | 0.0002 (4954) | 0.0000 (6891) | 0.0001 (7842) | 0.0002 (10245) |
| MuSiQue | NER | 0.0011 (920) | 0.0000 (1303) | 0.0013 (1559) | 0.0000 (1736) |
| MuSiQue | STRUCT_ONLY | 0.0000 (2004) | 0.0003 (3397) | 0.0004 (4821) | 0.0005 (7662) |
| Hotpot | NER | 0.0013 (3113) | 0.0000 (4054) | 0.0002 (4111) | 0.0000 (3860) |
| Hotpot | STRUCT_ONLY | 0.0000 (12302) | 0.0001 (18007) | 0.0002 (17284) | 0.0002 (17406) |
| SQuAD | NER | 0.0000 (911) | 0.0000 (1092) | 0.0000 (1146) | 0.0000 (1103) |
| SQuAD | STRUCT_ONLY | 0.0000 (1863) | 0.0000 (2887) | 0.0000 (4153) | 0.0000 (7722) |

## STEPS 5-7 -- FAMILY-RESTRICTED ASSIGNMENT AT EXACT P50

Fixed-K replacement, `K = K0`, `B = 6`, `P = 50`; the ordering key is the family-restricted provenance score and nothing else. `SRC_ALL` is the parity anchor: it is `V6_EXIT_SRC_SIM` from the prior phase and must reproduce it exactly.


### Candidate coverage -- what fraction of the K(q) universe the family can score at all

| corpus | SRC_ALL | SRC_NER | SRC_STRUCT_ONLY | ALL_NER | ALL_STRUCT_ONLY |
|---|--:|--:|--:|--:|--:|
| MetaQA | 0.8610 | 0.4316 | 0.7143 | 0.7213 | 0.8560 |
| WebQSP | 0.7362 | 0.1273 | 0.6686 | 0.2865 | 0.8435 |
| 2Wiki | 0.6869 | 0.1382 | 0.6017 | 0.2946 | 0.7799 |
| MuSiQue | 0.9371 | 0.2255 | 0.9263 | 0.3376 | 0.9514 |
| Hotpot | 0.7442 | 0.1024 | 0.6903 | 0.2616 | 0.8418 |
| SQuAD | 0.8217 | 0.1579 | 0.8192 | 0.2795 | 0.8723 |

### Exact P50 coverage, selector `F6` (net = gained - lost vs FROZEN)

| corpus | slice | FROZEN | SRC_ALL | SRC_NER | SRC_STRUCT_ONLY | ALL_NER | ALL_STRUCT_ONLY |
|---|---|--:|--:|--:|--:|--:|--:|
| MetaQA | hop1 | 0.9955 | 0.9955 +0 | 0.9955 +0 | 0.9955 +0 | 0.9955 +0 | 0.9955 +0 |
| MetaQA | hop2 | 0.7297 | 0.7387 +6 | 0.7297 +0 | 0.7267 -2 | 0.7357 +4 | 0.7342 +3 |
| MetaQA | hop3 | 0.2583 | 0.2643 +4 | 0.2598 +1 | 0.2613 +2 | 0.2613 +2 | 0.2613 +2 |
| MetaQA | ALL | 0.6612 | 0.6662 +10 **SIG** | 0.6617 +1 | 0.6612 +0 | 0.6642 +6 | 0.6637 +5 |
| WebQSP | ALL | 0.7646 | 0.7660 +2 | 0.7646 +0 | 0.7646 +0 | 0.7646 +0 | 0.7667 +3 |
| 2Wiki | ALL | 0.9435 | 0.9410 -5 | 0.9435 +0 | 0.9420 -3 | 0.9435 +0 | 0.9420 -3 |
| MuSiQue | ALL | 0.9635 | 0.9585 -10 **SIG** | 0.9635 +0 | 0.9585 -10 **SIG** | 0.9630 -1 | 0.9595 -8 **SIG** |
| Hotpot | ALL | 0.9505 | 0.9405 -20 **SIG** | 0.9505 +0 | 0.9405 -20 **SIG** | 0.9505 +0 | 0.9415 -18 **SIG** |
| SQuAD | ALL | 0.9875 | 0.9805 -14 **SIG** | 0.9875 +0 | 0.9805 -14 **SIG** | 0.9875 +0 | 0.9820 -11 **SIG** |

### Exact P50 coverage, selector `G4` (net = gained - lost vs FROZEN)

| corpus | slice | FROZEN | SRC_ALL | SRC_NER | SRC_STRUCT_ONLY | ALL_NER | ALL_STRUCT_ONLY |
|---|---|--:|--:|--:|--:|--:|--:|
| MetaQA | hop1 | 0.9955 | 0.9955 +0 | 0.9955 +0 | 0.9955 +0 | 0.9955 +0 | 0.9955 +0 |
| MetaQA | hop2 | 0.7297 | 0.7237 -4 | 0.7207 -6 | 0.7192 -7 | 0.7222 -5 | 0.7207 -6 |
| MetaQA | hop3 | 0.2583 | 0.2823 +16 **SIG** | 0.2748 +11 **SIG** | 0.2733 +10 **SIG** | 0.2733 +10 **SIG** | 0.2703 +8 |
| MetaQA | ALL | 0.6612 | 0.6672 +12 | 0.6637 +5 | 0.6627 +3 | 0.6637 +5 | 0.6622 +2 |
| WebQSP | ALL | 0.7646 | 0.7639 -1 | 0.7625 -3 | 0.7632 -2 | 0.7625 -3 | 0.7653 +1 |
| 2Wiki | ALL | 0.9435 | 0.9400 -7 | 0.9435 +0 | 0.9415 -4 | 0.9435 +0 | 0.9420 -3 |
| MuSiQue | ALL | 0.9635 | 0.9565 -14 **SIG** | 0.9595 -8 **SIG** | 0.9565 -14 **SIG** | 0.9590 -9 **SIG** | 0.9575 -12 **SIG** |
| Hotpot | ALL | 0.9505 | 0.9345 -32 **SIG** | 0.9425 -16 **SIG** | 0.9335 -34 **SIG** | 0.9430 -15 **SIG** | 0.9345 -32 **SIG** |
| SQuAD | ALL | 0.9875 | 0.9810 -13 **SIG** | 0.9870 -1 | 0.9810 -13 **SIG** | 0.9870 -1 | 0.9820 -11 **SIG** |
