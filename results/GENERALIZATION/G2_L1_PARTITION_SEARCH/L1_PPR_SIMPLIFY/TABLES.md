# L1 SIMPLIFY / PPR PHASE — TABLES

`*` marks p < 0.05 (exact McNemar, paired, vs BASE unless stated). Every method in every table emits EXACTLY 50 partitions.

## T1 — STEP 0 verification gates

| corpus | base_rank prov. | ret_rrf prov. | symmetric scoring | deterministic ties | exact P=50 | absent-channel | gold in ranking |
|---|---|---|---|---|---|---|---|
| MetaQA | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| WebQSP | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| 2Wiki | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| MuSiQue | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| HotpotQA | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| SQuAD | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |

## T2 — STEP 0/1 measured modality redundancy (the reason for the channel redefinition)

| corpus | frac. of retrieval challengers already canonically ranked | Spearman(canonical, retrieval) on shared partitions | queries with ≥5 shared |
|---|---|---|---|
| MetaQA | 0.9534 | 0.2488 | 1795 |
| WebQSP | 0.9378 | 0.3154 | 233 |
| 2Wiki | 0.9986 | 0.3442 | 1331 |
| MuSiQue | 1.0000 | 0.158 | 221 |
| HotpotQA | 0.8148 | 0.3243 | 1618 |
| SQuAD | 1.0000 | 0.2378 | 672 |

## T3 — STEP 1 the three orthogonal channels

| corpus | npart | base_rank == RRF(PR_dense, PR_splade) | ρ(DENSE, SPLADE) | ρ(DENSE, BASE) | ρ(SPLADE, BASE) |
|---|---|---|---|---|---|
| MetaQA | 401 | **EXACT** | 0.3451 | 0.7713 | 0.7985 |
| WebQSP | 7814 | **EXACT** | 0.9069 | 0.9519 | 0.9626 |
| 2Wiki | 658 | **EXACT** | 0.6862 | 0.8850 | 0.8890 |
| MuSiQue | 136 | **EXACT** | 0.5805 | 0.8774 | 0.8757 |
| HotpotQA | 5074 | **EXACT** | 0.7617 | 0.8935 | 0.9042 |
| SQuAD | 190 | **EXACT** | 0.6975 | 0.9104 | 0.9133 |

## T4 — STEP 2 canonical partition transition graph (built once, cached permanently)

| corpus | partitions | nodes | inter-partition edges | density | out-deg mean | out-deg max | isolated | internal (self) mass frac | cache bytes | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | 401 | 40151 | 63388 | 0.3942 | 158.07 | 398 | 0 | 0.309 | 84,681 | 0.1 |
| WebQSP | 7814 | 781485 | 1008926 | 0.0165 | 129.12 | 4966 | 0 | 0.268 | 2,110,755 | 0.9 |
| 2Wiki | 658 | 65865 | 56140 | 0.1297 | 85.32 | 657 | 0 | 0.199 | 95,827 | 0.1 |
| MuSiQue | 136 | 13672 | 7596 | 0.4107 | 55.85 | 131 | 0 | 0.536 | 13,880 | 0.0 |
| HotpotQA | 5074 | 507494 | 2389756 | 0.0928 | 470.98 | 4986 | 0 | 0.119 | 4,931,073 | 2.0 |
| SQuAD | 190 | 19029 | 12808 | 0.3548 | 67.41 | 189 | 0 | 0.425 | 26,548 | 0.1 |

## T5 — STEP 5B/6 DIRECT exactly-P50 fusion vs the protected-core swap (ΔALL vs BASE)

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| L0  BASE = Dense+SPLADE partition RRF | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L2  DIRECT + S4 structural | -0.0065 | -0.0063 | -0.0175\* | -0.0125\* | -0.0095\* | -0.0045 | -0.0095 | -0.0175 |
| L3  DIRECT + global partition PPR (P1) | -0.0140\* | -0.0120 | +0.0000 | -0.0040 | +0.0050 | -0.0005 | -0.0043 | -0.0140 |
| L3b DIRECT + global PPR, top-64 only | -0.0380\* | -0.0324\* | -0.0085\* | -0.0190\* | +0.0115\* | -0.0115\* | -0.0163 | -0.0380 |
| L4  DIRECT + bounded partition PPR (P2) | -0.0085\* | +0.0035 | -0.0025 | -0.0035 | -0.0020 | -0.0005 | -0.0023 | -0.0085 |
| L4b DIRECT + bounded PPR, neighbour-expanded | -0.0145\* | +0.0021 | -0.0030 | -0.0040 | -0.0015 | -0.0005 | -0.0036 | -0.0145 |
| L6  DIRECT + S4 + global PPR | -0.0010 | -0.0028 | -0.0205\* | -0.0180\* | +0.0020 | -0.0030 | -0.0072 | -0.0205 |
| **L1  SAFE_REFERENCE_ROUTER B6_S4_F6** | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| SWAP de-duplicated (no retrieval channel), B=6 | +0.0130\* | +0.0007 | -0.0030 | -0.0045 | -0.0020 | -0.0005 | +0.0006 | -0.0045 |

## T6 — STEP 5A B-sweep. B is removed as an axis; output is still exactly 50 (ΔALL vs BASE)

| corpus | F6 B=2 | F6 B=4 | F6 B=6 | F6 B=8 | F6 B=12 | ES B=2 | ES B=4 | ES B=6 | ES B=8 | ES B=12 |
|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | +0.0030 | +0.0025 | +0.0025 | +0.0050 | +0.0060 | +0.0100\* | +0.0125\* | +0.0130\* | +0.0140\* | +0.0130\* |
| WebQSP | +0.0028 | +0.0042 | +0.0028 | +0.0000 | -0.0063 | +0.0049 | +0.0035 | +0.0007 | -0.0049 | -0.0106 |
| 2Wiki | +0.0050\* | +0.0065\* | +0.0060\* | +0.0055\* | +0.0045 | -0.0005 | -0.0010 | -0.0030 | -0.0055\* | -0.0100\* |
| MuSiQue | +0.0010 | +0.0060\* | +0.0070\* | +0.0090\* | +0.0085\* | +0.0005 | -0.0040 | -0.0045 | -0.0065\* | -0.0075\* |
| HotpotQA | +0.0085\* | +0.0130\* | +0.0160\* | +0.0175\* | +0.0210\* | +0.0005 | -0.0010 | -0.0020 | -0.0045 | -0.0130\* |
| SQuAD | +0.0040\* | +0.0065\* | +0.0070\* | +0.0080\* | +0.0095\* | +0.0000 | +0.0000 | -0.0005 | -0.0010 | -0.0030 |

- **F6** — B=2: macro +0.0041 / worst +0.0010; B=4: macro +0.0065 / worst +0.0025; B=6: macro +0.0069 / worst +0.0025; B=8: macro +0.0075 / worst +0.0000; B=12: macro +0.0072 / worst -0.0063
- **ES** — B=2: macro +0.0026 / worst -0.0005; B=4: macro +0.0017 / worst -0.0040; B=6: macro +0.0006 / worst -0.0045; B=8: macro -0.0014 / worst -0.0065; B=12: macro -0.0052 / worst -0.0130

## T7 — STEP 8 MetaQA per-hop (ΔALL vs BASE); hop1 n=666 hop2 n=666 hop3 n=666

| variant | hop1 | hop2 | hop3 |
|---|---|---|---|
| L0_BASE | +0.0000 | +0.0000 | +0.0000 |
| L2_DIRECT_S4 | -0.0045 | -0.0390\* | +0.0240 |
| L3_DIRECT_PPRGLOBAL | -0.0060 | -0.0526\* | +0.0165 |
| L3b_DIRECT_PPRGLOBAL_M64 | -0.0240\* | -0.0946\* | +0.0045 |
| L4_DIRECT_PPRBOUNDED | -0.0030 | -0.0300\* | +0.0075 |
| L4b_DIRECT_PPRBOUNDED_EXP | -0.0060 | -0.0511\* | +0.0135 |
| L6_DIRECT_S4_PLUS_PPR | -0.0165\* | -0.0315\* | +0.0450\* |
| L1_SWAP_F6_B6 | +0.0000 | +0.0060 | +0.0015 |
| SWAP_ES_B6 | +0.0000 | +0.0255\* | +0.0135 |
| L1_SWAP_F6_B8 | +0.0000 | +0.0090 | +0.0060 |
| L1_SWAP_F6_B12 | +0.0000 | +0.0090 | +0.0090 |
| SWAP_ES_B8 | +0.0000 | +0.0255\* | +0.0165 |
| SWAP_ES_B12 | +0.0000 | +0.0165 | +0.0225\* |

## T8 — STEP 1 independence gate: does a third channel carry NEW information?

`dup` = of the partitions the channel promotes that are outside the canonical top-50, the fraction that is nevertheless already inside the canonical top-200. `1.000` = the channel says nothing Dense+SPLADE does not already say.

| corpus | S4 dup | S4 new parts/q | global PPR dup | PPR new parts/q | bounded PPR dup | Jaccard(PPR top50, BASE50) | Jaccard(S4 top50, BASE50) |
|---|---|---|---|---|---|---|---|
| MetaQA | 0.555 | 34.57 | 0.856 | 32.1 | 0.899 | 0.392 | 0.154 |
| WebQSP | 0.186 | 26.29 | 0.438 | 35.5 | 0.713 | 0.341 | 0.142 |
| 2Wiki | 0.354 | 29.76 | 0.834 | 28.33 | 0.918 | 0.479 | 0.099 |
| MuSiQue | 1.000 | 11.09 | 1.000 | 25.56 | 1.000 | 0.490 | 0.281 |
| HotpotQA | 0.180 | 38.35 | 0.630 | 37.05 | 0.787 | 0.324 | 0.086 |
| SQuAD | 1.000 | 6.59 | 1.000 | 30.01 | 1.000 | 0.403 | 0.173 |

## T9 — STEP 6 structural-signal bake-off inside the swap (ΔALL vs BASE)

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| **S4_F6_B6** | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| PPRG_F6_B6 | -0.0075\* | -0.0113\* | +0.0065\* | +0.0035 | +0.0160\* | +0.0060\* | +0.0022 | -0.0113 |
| PPRB_F6_B6 | -0.0065\* | +0.0028 | +0.0050 | +0.0045 | +0.0105\* | +0.0065\* | +0.0038 | -0.0065 |
| PPRP32_F6_B6 | -0.0080\* | -0.0099\* | +0.0070\* | +0.0040 | +0.0160\* | +0.0075\* | +0.0028 | -0.0099 |
| S4+PPRG_F6_B6 | -0.0065\* | -0.0007 | +0.0055\* | +0.0030 | +0.0165\* | +0.0045 | +0.0037 | -0.0065 |
| S4_ES_B6 | +0.0130\* | +0.0007 | -0.0030 | -0.0045 | -0.0020 | -0.0005 | +0.0006 | -0.0045 |
| PPRG_ES_B6 | -0.0015 | -0.0099\* | +0.0000 | -0.0100\* | +0.0010 | -0.0005 | -0.0035 | -0.0100 |
| PPRB_ES_B6 | +0.0005 | +0.0028 | -0.0020 | -0.0075\* | -0.0045 | +0.0010 | -0.0016 | -0.0075 |
| PPRP32_ES_B6 | +0.0000 | -0.0106\* | -0.0005 | -0.0110\* | +0.0015 | -0.0005 | -0.0035 | -0.0110 |
| S4+PPRG_ES_B6 | +0.0040 | +0.0007 | -0.0015 | -0.0070\* | +0.0020 | +0.0005 | -0.0002 | -0.0070 |
| S4_F6_B8 | +0.0050 | +0.0000 | +0.0055\* | +0.0090\* | +0.0175\* | +0.0080\* | +0.0075 | +0.0000 |
| PPRG_F6_B8 | -0.0095\* | -0.0113\* | +0.0065\* | +0.0045 | +0.0190\* | +0.0065\* | +0.0026 | -0.0113 |
| PPRB_F6_B8 | -0.0080\* | +0.0021 | +0.0045 | +0.0045 | +0.0125\* | +0.0070\* | +0.0038 | -0.0080 |
| S4+PPRG_F6_B8 | -0.0075\* | -0.0007 | +0.0055 | +0.0040 | +0.0185\* | +0.0060\* | +0.0043 | -0.0075 |

## T10 — STEP 7 reach probe: PPR *does* reach gold partitions S4 cannot

| corpus | gold partitions outside BASE50 | reached by S4 | reached ONLY by global PPR | ONLY by bounded PPR | ONLY by precomputed PPR | new partitions/q vs S4 (G/B/P32) |
|---|---|---|---|---|---|---|
| MetaQA | 5889 | 1536 | 867 | 611 | 827 | 26.97 / 20.05 / 27.3 |
| WebQSP | 1686 | 123 | 196 | 262 | 209 | 33.99 / 17.87 / 32.68 |
| 2Wiki | 131 | 10 | 37 | 34 | 42 | 25.93 / 19.84 / 26.23 |
| MuSiQue | 90 | 20 | 33 | 37 | 40 | 20.89 / 12.47 / 19.35 |
| HotpotQA | 138 | 17 | 48 | 36 | 56 | 35.4 / 22.03 / 36.43 |
| SQuAD | 39 | 10 | 11 | 14 | 20 | 27.67 / 16.0 / 27.28 |

## T11 — MetaQA per-hop for the bake-off (ΔALL vs BASE)

| variant | hop1 | hop2 | hop3 |
|---|---|---|---|
| S4_F6_B6 | +0.0000 | +0.0060 | +0.0015 |
| PPRG_F6_B6 | +0.0000 | -0.0180\* | -0.0045 |
| PPRB_F6_B6 | +0.0000 | -0.0165\* | -0.0030 |
| PPRP32_F6_B6 | +0.0000 | -0.0195\* | -0.0045 |
| S4+PPRG_F6_B6 | +0.0000 | -0.0135\* | -0.0060 |
| S4_ES_B6 | +0.0000 | +0.0255\* | +0.0135 |
| PPRG_ES_B6 | +0.0000 | -0.0135\* | +0.0090 |
| PPRB_ES_B6 | +0.0000 | -0.0060 | +0.0075 |
| PPRP32_ES_B6 | +0.0000 | -0.0120 | +0.0120 |
| S4+PPRG_ES_B6 | +0.0000 | +0.0000 | +0.0120 |
| S4_F6_B8 | +0.0000 | +0.0090 | +0.0060 |
| PPRG_F6_B8 | +0.0000 | -0.0240\* | -0.0045 |
| PPRB_F6_B8 | +0.0000 | -0.0225\* | -0.0015 |
| S4+PPRG_F6_B8 | +0.0000 | -0.0180\* | -0.0045 |

## T12 — STEP 4 precomputation. (B) full basis is EXACT by linearity; only (C) is lossy

| corpus | basis precompute s | full basis bytes | (A) online s/query | (A) online graph edges/query | (B) top50 set agreement | (B) max abs Δmass | (C) L=8 | (C) L=16 | (C) L=32 | (C) L=64 | (C) L=32 bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | 0.3 | 643,204 | 0.000425 | 1,267,760 | 1.0000 | 5.20e-17 | 0.0000 | 0.0000 | 0.0005 | 0.0000 | 102,656 |
| WebQSP | 110.1 | 244,234,384 | 0.013502 | 20,178,520 | 1.0000 | 3.47e-16 | 0.0000 | 0.0000 | 0.0000 | 0.0395 | 2,000,384 |
| 2Wiki | 0.4 | 1,731,856 | 0.000488 | 1,122,800 | 1.0000 | 4.16e-16 | 0.0000 | 0.0000 | 0.0180 | 0.0040 | 168,448 |
| MuSiQue | 0.0 | 73,984 | 0.000066 | 151,920 | 1.0000 | 6.25e-17 | 0.0000 | 0.0000 | 0.0030 | 0.0165 | 34,816 |
| HotpotQA | 144.7 | 102,981,904 | 0.030566 | 47,795,120 | 1.0000 | 9.71e-17 | 0.0000 | 0.0000 | 0.0015 | 0.0590 | 1,298,944 |
| SQuAD | 0.0 | 144,400 | 0.000103 | 256,160 | 1.0000 | 6.94e-17 | 0.0000 | 0.0000 | 0.0000 | 0.0300 | 48,640 |

## T13 — STEP 3 P3 partition-bounded node PPR, with the two NO-GRAPH controls

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| L0_BASE | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L1_SWAP_F6_B6 | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| L3_DIRECT_PPRGLOBAL | -0.0140\* | -0.0120 | +0.0000 | -0.0040 | +0.0050 | -0.0005 | -0.0043 | -0.0140 |
| L3p_DIRECT_PPR_PRECOMP_L32 | -0.0295\* | -0.0197\* | -0.0045 | +0.0000 | +0.0075 | +0.0025 | -0.0073 | -0.0295 |
| L5_P3_CTRL_seed_mass_NO_GRAPH | -0.0435\* | -0.0056 | +0.0030 | +0.0100\* | +0.0235\* | +0.0130\* | +0.0001 | -0.0435 |
| L5_P3_CTRL_support_only_NO_GRAPH | -0.0390\* | -0.0049 | +0.0040 | +0.0095\* | +0.0240\* | +0.0130\* | +0.0011 | -0.0390 |
| L5_P3_concentration | -0.0455\* | -0.0070 | +0.0040 | +0.0100\* | +0.0240\* | +0.0130\* | -0.0002 | -0.0455 |
| L5_P3_max_mass | -0.0455\* | -0.0049 | +0.0040 | +0.0100\* | +0.0240\* | +0.0130\* | +0.0001 | -0.0455 |
| L5_P3_reachable_frac | -0.0435\* | -0.0056 | +0.0035 | +0.0100\* | +0.0245\* | +0.0125\* | +0.0002 | -0.0435 |
| L5_P3_topk_mass | -0.0445\* | -0.0049 | +0.0040 | +0.0095\* | +0.0245\* | +0.0125\* | +0.0002 | -0.0445 |
| L5e_P3ENTRY_CTRL_seed_mass_NO_GRAPH | -0.0095\* | +0.0028 | +0.0030 | +0.0000 | +0.0060\* | +0.0005 | +0.0005 | -0.0095 |
| L5e_P3ENTRY_CTRL_support_only_NO_GRAPH | +0.0000 | +0.0014 | +0.0035\* | +0.0000 | +0.0000 | +0.0000 | +0.0008 | +0.0000 |
| L5e_P3ENTRY_concentration | -0.0015 | +0.0000 | +0.0050\* | +0.0020 | +0.0120\* | +0.0050\* | +0.0037 | -0.0015 |
| L5e_P3ENTRY_max_mass | -0.0010 | +0.0106\* | +0.0030 | -0.0010 | -0.0025 | +0.0055\* | +0.0024 | -0.0025 |
| L5e_P3ENTRY_reachable_frac | +0.0010 | +0.0070\* | +0.0045\* | -0.0035 | -0.0030 | -0.0005 | +0.0009 | -0.0035 |
| L5e_P3ENTRY_topk_mass | -0.0015 | +0.0113\* | +0.0025 | -0.0020 | -0.0040 | +0.0040\* | +0.0017 | -0.0040 |

## T14 — STEP 3 P3 cost and support

| corpus | local operator bytes | s/query (seed-only) | s/query (entry) | partitions scored/query (seed-only) | (entry) |
|---|---|---|---|---|---|
| MetaQA | 16,088,316 | 0.005770 | 0.012200 | 39.57 | 82.18 |
| WebQSP | 276,252,016 | 0.007343 | 0.012555 | 26.67 | 56.08 |
| 2Wiki | 26,382,532 | 0.004987 | 0.008667 | 34.92 | 61.06 |
| MuSiQue | 5,500,776 | 0.004989 | 0.008528 | 34.15 | 56.73 |
| HotpotQA | 202,855,272 | 0.006173 | 0.012891 | 27.7 | 73.08 |
| SQuAD | 7,630,684 | 0.005308 | 0.013748 | 28.09 | 58.21 |

## T15 — STEP 7 CAP_B6 vs CAP_P50. These are NOT the same thing.

Decomposition of the queries the SAFE_REFERENCE_ROUTER (B6_S4_F6) leaves uncovered. `CAP_P50` = more than 50 gold partitions, impossible at P=50 under ANY replacement. `REACH` = a missing gold partition is not in the query's candidate universe at all. `CAP_B6` = every missing gold IS reachable but more than B=6 partitions of BASE would have to change. `RANKING` = ≤6 changes would suffice and the selector simply ordered them wrong.

| corpus | CAP_P50 | REACH | CAP_B6 | RANKING | REACH after adding PPR-reachable | mean gold partitions/query | mean gold outside BASE50 |
|---|---|---|---|---|---|---|---|
| MetaQA | 26 | 318 | 21 | 312 | 0 | 5.78 | 2.947 |
| WebQSP | 7 | 127 | 13 | 187 | 0 | 3.66 | 1.188 |
| 2Wiki | 0 | 49 | 0 | 64 | 0 | 1.98 | 0.066 |
| MuSiQue | 0 | 0 | 0 | 73 | 0 | 1.77 | 0.045 |
| HotpotQA | 0 | 50 | 0 | 49 | 0 | 1.66 | 0.069 |
| SQuAD | 0 | 0 | 0 | 25 | 0 | 1.0 | 0.019 |

MetaQA by hop (router pool):

| hop | CAP_P50 | REACH | CAP_B6 | RANKING |
|---|---|---|---|---|
| hop2 | 2 | 42 | 7 | 129 |
| hop3 | 24 | 275 | 14 | 181 |

## T16 — STEP 11 latency and online graph work

| corpus | npart | P1 global PPR s/q | P1 online edges/q | P2 bounded PPR s/q | P2 candidate partitions | P2 subgraph edges | P2 online edges/q | precomputed basis online edges/q |
|---|---|---|---|---|---|---|---|---|
| MetaQA | 401 | 0.0005 | 1,267,760 | 0.0013 | 141.1 | 10702.5 | 214,049 | **0** |
| WebQSP | 7814 | 0.0149 | 20,178,520 | 0.0010 | 116.5 | 1646.5 | 32,930 | **0** |
| 2Wiki | 658 | 0.0005 | 1,122,800 | 0.0014 | 127.4 | 5054.4 | 101,089 | **0** |
| MuSiQue | 136 | 0.0001 | 151,920 | 0.0011 | 80.5 | 3271.7 | 65,434 | **0** |
| HotpotQA | 5074 | 0.0385 | 47,795,120 | 0.0031 | 151.4 | 9343.3 | 186,866 | **0** |
| SQuAD | 190 | 0.0001 | 256,160 | 0.0014 | 83.5 | 3572.2 | 71,445 | **0** |

## T17 — STEP 0 (quantified) the boundary competition is near-deterministic turnover

| corpus | B | incumbents retained / B | retention rate | boundary incumbents with canonical evidence ONLY (/q) | their retention rate | gold partitions evicted | gained | net |
|---|---|---|---|---|---|---|---|---|
| MetaQA | 6 | 1.341 | 0.2235 | 3.96 | 0.0009 | 183 | 257 | +74 |
| MetaQA | 12 | 3.624 | 0.3020 | 7.767 | 0.0012 | 356 | 438 | +82 |
| WebQSP | 6 | 1.032 | 0.1721 | 5.042 | 0.0280 | 90 | 91 | +1 |
| WebQSP | 12 | 2.484 | 0.2070 | 9.971 | 0.0489 | 196 | 115 | -81 |
| 2Wiki | 6 | 1.741 | 0.2902 | 4.295 | 0.0882 | 6 | 20 | +14 |
| 2Wiki | 12 | 4.424 | 0.3687 | 8.397 | 0.1190 | 10 | 23 | +13 |
| MuSiQue | 6 | 1.688 | 0.2813 | 4.258 | 0.0641 | 11 | 26 | +15 |
| MuSiQue | 12 | 4.776 | 0.3980 | 8.24 | 0.1439 | 14 | 33 | +19 |
| HotpotQA | 6 | 0.901 | 0.1502 | 4.886 | 0.0002 | 5 | 39 | +34 |
| HotpotQA | 12 | 2.213 | 0.1844 | 9.718 | 0.0015 | 6 | 52 | +46 |
| SQuAD | 6 | 2.026 | 0.3377 | 4.814 | 0.1999 | 3 | 17 | +14 |
| SQuAD | 12 | 5.617 | 0.4681 | 9.454 | 0.3278 | 3 | 22 | +19 |

## T18 — STEP 10 PPR damping robustness (reported, NOT selected)

| corpus | S4_F6_B6 | DIRECT α=0.5 | DIRECT α=0.7 | DIRECT α=0.85 | DIRECT α=0.95 | SWAP α=0.5 | SWAP α=0.7 | SWAP α=0.85 | SWAP α=0.95 |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA | +0.0025 | -0.0295 | -0.0250 | -0.0140 | -0.0075 | -0.0085 | -0.0085 | -0.0075 | -0.0040 |
| HotpotQA | +0.0160 | +0.0135 | +0.0120 | +0.0050 | -0.0110 | +0.0165 | +0.0170 | +0.0160 | +0.0120 |
| WebQSP | +0.0028 | -0.0113 | -0.0127 | -0.0120 | -0.0120 | -0.0092 | -0.0085 | -0.0113 | -0.0120 |

## T19 — STEP 1 executed literally: ONE vote per evidence family (ΔALL vs BASE)

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| B=6 · F6      canonical + structural + retrieval  (3 votes, lexical twice) | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| B=6 · ES      canonical + structural              (2, node view DELETED) | +0.0130\* | +0.0007 | -0.0030 | -0.0045 | -0.0020 | -0.0005 | +0.0006 | -0.0045 |
| B=6 · LEXONE  RRF(canonical, retrieval) + structural (2, ONE lexical vote) | +0.0100\* | +0.0014 | -0.0005 | -0.0035 | +0.0010 | +0.0015 | +0.0017 | -0.0035 |
| B=6 · DS3     dense + splade + structural         (the directive's literal set) | +0.0120\* | +0.0035 | -0.0035 | -0.0050 | -0.0010 | +0.0005 | +0.0011 | -0.0050 |
| B=6 · DS4     dense + splade + structural + retrieval | +0.0015 | +0.0042 | +0.0050\* | +0.0065\* | +0.0160\* | +0.0065\* | +0.0066 | +0.0015 |
| B=8 · F6      canonical + structural + retrieval  (3 votes, lexical twice) | +0.0050 | +0.0000 | +0.0055\* | +0.0090\* | +0.0175\* | +0.0080\* | +0.0075 | +0.0000 |
| B=8 · ES      canonical + structural              (2, node view DELETED) | +0.0140\* | -0.0049 | -0.0055\* | -0.0065\* | -0.0045 | -0.0010 | -0.0014 | -0.0065 |
| B=8 · LEXONE  RRF(canonical, retrieval) + structural (2, ONE lexical vote) | +0.0115\* | -0.0042 | -0.0030 | -0.0045 | +0.0000 | +0.0010 | +0.0001 | -0.0045 |
| B=8 · DS3     dense + splade + structural         (the directive's literal set) | +0.0135\* | +0.0007 | -0.0055\* | -0.0055 | -0.0030 | -0.0010 | -0.0001 | -0.0055 |
| B=8 · DS4     dense + splade + structural + retrieval | +0.0035 | +0.0035 | +0.0055\* | +0.0080\* | +0.0170\* | +0.0075\* | +0.0075 | +0.0035 |

MetaQA per hop (ΔALL vs BASE):

| variant | hop1 | hop2 | hop3 |
|---|---|---|---|
| F6_B6 | +0.0000 | +0.0060 | +0.0015 |
| ES_B6 | +0.0000 | +0.0255\* | +0.0135 |
| LEXONE_B6 | +0.0000 | +0.0165 | +0.0135 |
| DS3_B6 | +0.0000 | +0.0285\* | +0.0075 |
| DS4_B6 | +0.0000 | +0.0030 | +0.0015 |

## T20 — STEP 7 addendum: is the SELECTOR or the POOL the limiter?

`ORACLE_B` = exists X with |X|=B drawn from (boundary + challengers) such that the protected core plus X covers all gold partitions. It is the best any parameter-free selector could do at that B without changing the contract.

| corpus | B | actual ALL | oracle ALL | selector headroom | uncovered even by oracle | of those: POOL failure | of those: need > B swaps | mean partitions needing replacement |
|---|---|---|---|---|---|---|---|---|
| MetaQA | 6 | 0.6612 | 0.7212 | **+0.0601** | 557 | 555 | 2 | 3.069 |
| MetaQA | 8 | 0.6637 | 0.7212 | **+0.0576** | 557 | 555 | 2 | 3.119 |
| MetaQA | 12 | 0.6647 | 0.7222 | **+0.0576** | 555 | 555 | 0 | 3.211 |
| WebQSP | 6 | 0.7646 | 0.7886 | **+0.0240** | 300 | 300 | 0 | 1.276 |
| WebQSP | 8 | 0.7618 | 0.7886 | **+0.0268** | 300 | 300 | 0 | 1.31 |
| WebQSP | 12 | 0.7555 | 0.7886 | **+0.0331** | 300 | 300 | 0 | 1.386 |
| 2Wiki | 6 | 0.9435 | 0.9515 | **+0.0080** | 97 | 97 | 0 | 0.073 |
| 2Wiki | 8 | 0.9430 | 0.9515 | **+0.0085** | 97 | 97 | 0 | 0.077 |
| 2Wiki | 12 | 0.9420 | 0.9515 | **+0.0095** | 97 | 97 | 0 | 0.083 |
| MuSiQue | 6 | 0.9635 | 0.9750 | **+0.0115** | 50 | 50 | 0 | 0.064 |
| MuSiQue | 8 | 0.9655 | 0.9750 | **+0.0095** | 50 | 50 | 0 | 0.069 |
| MuSiQue | 12 | 0.9650 | 0.9750 | **+0.0100** | 50 | 50 | 0 | 0.078 |
| HotpotQA | 6 | 0.9505 | 0.9645 | **+0.0140** | 71 | 71 | 0 | 0.081 |
| HotpotQA | 8 | 0.9520 | 0.9645 | **+0.0125** | 71 | 71 | 0 | 0.086 |
| HotpotQA | 12 | 0.9555 | 0.9645 | **+0.0090** | 71 | 71 | 0 | 0.1 |
| SQuAD | 6 | 0.9875 | 0.9945 | **+0.0070** | 11 | 11 | 0 | 0.025 |
| SQuAD | 8 | 0.9885 | 0.9945 | **+0.0060** | 11 | 11 | 0 | 0.026 |
| SQuAD | 12 | 0.9900 | 0.9945 | **+0.0045** | 11 | 11 | 0 | 0.032 |

MetaQA per hop at B=6:

| hop | n | actual | oracle | selector headroom | POOL failures |
|---|---|---|---|---|---|
| hop1 | 666 | 0.9955 | 0.9970 | **+0.0015** | 2 |
| hop2 | 666 | 0.7297 | 0.8033 | **+0.0736** | 131 |
| hop3 | 666 | 0.2583 | 0.3634 | **+0.1051** | 422 |
