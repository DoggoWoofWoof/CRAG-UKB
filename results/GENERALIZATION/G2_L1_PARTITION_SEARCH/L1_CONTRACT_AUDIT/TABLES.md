# EXACT-P50 STRUCTURAL CONTRACT AUDIT -- TABLES

## T1. STEP 1 -- the three nested evidence universes

`S4_M64` is a strict prefix of `S4_FULL_ADMITTED`, which is a strict prefix of `S4_FULL_VISITED`. Node ordering is held fixed across all three, so the only thing that varies is how much evidence reaches the aggregation. The gate is that the M64 prefix reproduces the frozen structural ranking exactly.

| corpus | queries | replay parity | M64 == frozen SF | nodes/q M64 | nodes/q admitted | nodes/q visited | partitions/q M64 | partitions/q admitted | partitions/q visited |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metaqa | 1998 | **1998/1998** | **1998/1998** | 64 | 147.0 | 1233.3 | 47.5 | 84.5 | 305.5 |
| webqsp | 1419 | **1419/1419** | **1419/1419** | 64 | 138.4 | 1199.0 | 36.6 | 67.6 | 417.1 |
| 2wiki_clean | 2000 | **2000/2000** | **2000/2000** | 64 | 105.7 | 381.1 | 37.7 | 60.2 | 167.2 |
| musique_clean | 2000 | **2000/2000** | **2000/2000** | 64 | 153.0 | 971.3 | 28.2 | 46.9 | 111.7 |
| hotpotqa_clean | 2000 | **2000/2000** | **2000/2000** | 64 | 151.0 | 1158.9 | 45.5 | 93.8 | 585.0 |
| squad_clean | 2000 | **2000/2000** | **2000/2000** | 64 | 158.8 | 1741.6 | 16.5 | 32.2 | 116.8 |

## T2. STEP 2 -- CORE vs NOVEL partition accounting

CORE = inside the canonical top-50, NOVEL = outside it. Classified purely on inference-time state; gold is used only to count. `evidence on CORE` is the share of all partitions receiving structural evidence that were already in the canonical top-50, i.e. the share of the mechanism's output that is pure confirmation.

| corpus | universe | NOVEL_GOLD_PARTITION_RECALL | CORE gold recall | gold nodes CORE | gold nodes NOVEL | evidence on CORE |
|---|---|---:|---:|---:|---:|---:|
| metaqa | S4_M64 | **0.2608** | 0.4604 | 1066 | 919 | 0.2728 |
| metaqa | S4_FULL_ADMITTED | **0.3892** | 0.7244 | 3017 | 1301 | 0.2768 |
| metaqa | S4_FULL_VISITED | **0.9195** | 0.9629 | 4932 | 2666 | 0.1450 |
| webqsp | S4_M64 | **0.0730** | 0.4341 | 940 | 73 | 0.2812 |
| webqsp | S4_FULL_ADMITTED | **0.1198** | 0.5938 | 1996 | 136 | 0.2579 |
| webqsp | S4_FULL_VISITED | **0.3600** | 0.7615 | 3422 | 409 | 0.0701 |
| 2wiki_clean | S4_M64 | **0.0763** | 0.3719 | 360 | 3 | 0.2112 |
| 2wiki_clean | S4_FULL_ADMITTED | **0.1298** | 0.5748 | 1252 | 3 | 0.2120 |
| 2wiki_clean | S4_FULL_VISITED | **0.3740** | 0.8007 | 3491 | 4 | 0.1362 |
| musique_clean | S4_M64 | **0.2222** | 0.6066 | 404 | 6 | 0.6061 |
| musique_clean | S4_FULL_ADMITTED | **0.3889** | 0.7514 | 648 | 7 | 0.5616 |
| musique_clean | S4_FULL_VISITED | **0.8556** | 0.9555 | 3064 | 22 | 0.4031 |
| hotpotqa_clean | S4_M64 | **0.1232** | 0.5558 | 298 | 9 | 0.1578 |
| hotpotqa_clean | S4_FULL_ADMITTED | **0.2174** | 0.7897 | 735 | 14 | 0.1511 |
| hotpotqa_clean | S4_FULL_VISITED | **0.5507** | 0.9683 | 3565 | 45 | 0.0406 |
| squad_clean | S4_M64 | **0.2564** | 0.5166 | 38 | 1 | 0.6004 |
| squad_clean | S4_FULL_ADMITTED | **0.2821** | 0.5793 | 50 | 3 | 0.4999 |
| squad_clean | S4_FULL_VISITED | **0.6154** | 0.7537 | 978 | 8 | 0.3147 |

MetaQA per hop:

| hop | universe | NOVEL_GOLD_PARTITION_RECALL | CORE gold recall |
|---|---|---:|---:|
| hop1 | S4_M64 | **0.2500** | 0.5323 |
| hop1 | S4_FULL_ADMITTED | **0.5000** | 0.8735 |
| hop1 | S4_FULL_VISITED | **1.0000** | 0.9756 |
| hop2 | S4_M64 | **0.2640** | 0.5017 |
| hop2 | S4_FULL_ADMITTED | **0.4297** | 0.7722 |
| hop2 | S4_FULL_VISITED | **0.9863** | 0.9663 |
| hop3 | S4_M64 | **0.2603** | 0.4070 |
| hop3 | S4_FULL_ADMITTED | **0.3820** | 0.6377 |
| hop3 | S4_FULL_VISITED | **0.9078** | 0.9560 |

## T3. STEP 3 -- what the NEEDED partitions actually have (MetaQA)

`needed` = gold partitions outside the protected 44. Fractions are per needed partition.

| block | needed | visited_universe | admitted | S4_M64 | S4_FULL_ADMITTED | S4_FULL_VISITED | frozen_candidate_set | boundary | RF | canonical_top50 | NOTHING_frozen_contract | NOTHING |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 6132 | 0.9206 | 0.3951 | 0.2637 | 0.3951 | 0.9206 | 0.3167 | 0.0396 | 0.0444 | 0.0396 | 0.6833 | 0.0406 |
| hop1 | 6 | 1.0000 | 0.6667 | 0.3333 | 0.6667 | 1.0000 | 0.5000 | 0.3333 | 0.1667 | 0.3333 | 0.5000 | 0.0000 |
| hop2 | 943 | 0.9862 | 0.4422 | 0.2651 | 0.4422 | 0.9862 | 0.3351 | 0.0721 | 0.0361 | 0.0721 | 0.6649 | 0.0032 |
| hop3 | 5183 | 0.9085 | 0.3863 | 0.2634 | 0.3863 | 0.9085 | 0.3131 | 0.0334 | 0.0457 | 0.0334 | 0.6869 | 0.0475 |

## T4. STEP 4 -- how many replacements the swap actually needs

| corpus | block | queries | mean need at B=6 | >6 | >8 | >12 | >20 |
|---|---|---:|---:|---:|---:|---:|---:|
| metaqa | ALL | 1998 | 3.07 | 246 | 219 | 159 | 88 |
| metaqa | hop1 | 666 | 0.01 | 0 | 0 | 0 | 0 |
| metaqa | hop2 | 666 | 1.42 | 33 | 30 | 23 | 11 |
| metaqa | hop3 | 666 | 7.78 | 213 | 189 | 136 | 77 |
| webqsp | ALL | 1419 | 1.28 | 65 | 47 | 31 | 17 |
| 2wiki_clean | ALL | 2000 | 0.07 | 0 | 0 | 0 | 0 |
| musique_clean | ALL | 2000 | 0.06 | 0 | 0 | 0 | 0 |
| hotpotqa_clean | ALL | 2000 | 0.08 | 0 | 0 | 0 | 0 |
| squad_clean | ALL | 2000 | 0.02 | 0 | 0 | 0 | 0 |

## T5. STEP 4/5 -- the ORACLE ceiling matrix (FINAL P = 50 throughout)

Each cell is an oracle exact-P50 ceiling: coverable iff `need = goldp - prot_B` fits inside the candidate universe AND `|need| <= B`. Diagnostic only.

**metaqa / ALL** (SAFE 0.6612)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.7212 | 0.7212 | 0.7222 | 0.7222 | 0.7222 |
| S4_FULL_ADMITTED | 0.7447 | 0.7452 | 0.7467 | 0.7467 | 0.7467 |
| S4_FULL_VISITED | 0.8433 | 0.8509 | 0.8679 | 0.8809 | 0.8939 |

**metaqa / hop1** (SAFE 0.9955)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.9970 | 0.9970 | 0.9970 | 0.9970 | 0.9970 |
| S4_FULL_ADMITTED | 0.9985 | 0.9985 | 0.9985 | 0.9985 | 0.9985 |
| S4_FULL_VISITED | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**metaqa / hop2** (SAFE 0.7297)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.8033 | 0.8033 | 0.8033 | 0.8033 | 0.8033 |
| S4_FULL_ADMITTED | 0.8453 | 0.8453 | 0.8453 | 0.8453 | 0.8453 |
| S4_FULL_VISITED | 0.9384 | 0.9429 | 0.9520 | 0.9640 | 0.9820 |

**metaqa / hop3** (SAFE 0.2583)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.3634 | 0.3634 | 0.3664 | 0.3664 | 0.3664 |
| S4_FULL_ADMITTED | 0.3904 | 0.3919 | 0.3964 | 0.3964 | 0.3964 |
| S4_FULL_VISITED | 0.5916 | 0.6096 | 0.6517 | 0.6787 | 0.6997 |

**webqsp / ALL** (SAFE 0.7646)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.7886 | 0.7886 | 0.7886 | 0.7886 | 0.7886 |
| S4_FULL_ADMITTED | 0.7992 | 0.7992 | 0.7992 | 0.7992 | 0.7992 |
| S4_FULL_VISITED | 0.8591 | 0.8598 | 0.8598 | 0.8598 | 0.8598 |

**2wiki_clean / ALL** (SAFE 0.9435)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.9515 | 0.9515 | 0.9515 | 0.9515 | 0.9515 |
| S4_FULL_ADMITTED | 0.9545 | 0.9545 | 0.9545 | 0.9545 | 0.9545 |
| S4_FULL_VISITED | 0.9670 | 0.9670 | 0.9670 | 0.9670 | 0.9670 |

**musique_clean / ALL** (SAFE 0.9635)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.9750 | 0.9750 | 0.9750 | 0.9750 | 0.9750 |
| S4_FULL_ADMITTED | 0.9795 | 0.9795 | 0.9795 | 0.9795 | 0.9795 |
| S4_FULL_VISITED | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.9945 |

**hotpotqa_clean / ALL** (SAFE 0.9505)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.9645 | 0.9645 | 0.9645 | 0.9645 | 0.9645 |
| S4_FULL_ADMITTED | 0.9680 | 0.9680 | 0.9680 | 0.9680 | 0.9680 |
| S4_FULL_VISITED | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800 |

**squad_clean / ALL** (SAFE 0.9875)

| universe | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.9945 | 0.9945 | 0.9945 | 0.9945 | 0.9945 |
| S4_FULL_ADMITTED | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 |
| S4_FULL_VISITED | 0.9975 | 0.9975 | 0.9975 | 0.9975 | 0.9975 |

## T6. STEP 6 -- mutually exclusive attribution of the headroom

Five rungs, each relaxing exactly one restriction, each per-query monotone over the one below, so the differences partition the total gap.

| corpus | block | SAFE | +selection | +read trunc | +beam trunc | +B capacity | +reach | = FULL UNIVERSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| metaqa | ALL | 0.6612 | +0.0600 | +0.0235 | +0.0986 | +0.0506 | +0.0931 | **0.9870** |
| metaqa | hop1 | 0.9955 | +0.0015 | +0.0015 | +0.0015 | +0.0000 | +0.0000 | **1.0000** |
| metaqa | hop2 | 0.7297 | +0.0736 | +0.0420 | +0.0931 | +0.0436 | +0.0150 | **0.9970** |
| metaqa | hop3 | 0.2583 | +0.1051 | +0.0270 | +0.2012 | +0.1081 | +0.2643 | **0.9640** |
| webqsp | ALL | 0.7646 | +0.0240 | +0.0106 | +0.0599 | +0.0007 | +0.1353 | **0.9951** |
| 2wiki_clean | ALL | 0.9435 | +0.0080 | +0.0030 | +0.0125 | +0.0000 | +0.0330 | **1.0000** |
| musique_clean | ALL | 0.9635 | +0.0115 | +0.0045 | +0.0150 | +0.0000 | +0.0055 | **1.0000** |
| hotpotqa_clean | ALL | 0.9505 | +0.0140 | +0.0035 | +0.0120 | +0.0000 | +0.0200 | **1.0000** |
| squad_clean | ALL | 0.9875 | +0.0070 | +0.0005 | +0.0025 | +0.0000 | +0.0025 | **1.0000** |

Share of the total gap owned by each restriction:

| corpus | block | total gap | selection | read trunc | beam trunc | B capacity | reach | largest |
|---|---|---:|---:|---:|---:|---:|---:|---|
| metaqa | ALL | 0.3258 | 18.4% | 7.2% | 30.3% | 15.5% | 28.6% | **beam_truncation** |
| metaqa | hop1 | 0.0045 | 33.3% | 33.3% | 33.3% | 0.0% | 0.0% | **selection** |
| metaqa | hop2 | 0.2673 | 27.5% | 15.7% | 34.8% | 16.3% | 5.6% | **beam_truncation** |
| metaqa | hop3 | 0.7057 | 14.9% | 3.8% | 28.5% | 15.3% | 37.5% | **reach** |
| webqsp | ALL | 0.2305 | 10.4% | 4.6% | 26.0% | 0.3% | 58.7% | **reach** |
| 2wiki_clean | ALL | 0.0565 | 14.2% | 5.3% | 22.1% | 0.0% | 58.4% | **reach** |
| musique_clean | ALL | 0.0365 | 31.5% | 12.3% | 41.1% | 0.0% | 15.1% | **beam_truncation** |
| hotpotqa_clean | ALL | 0.0495 | 28.3% | 7.1% | 24.2% | 0.0% | 40.4% | **reach** |
| squad_clean | ALL | 0.0125 | 56.0% | 4.0% | 20.0% | 0.0% | 20.0% | **selection** |

The 3-way split the directive asked for (coarser: `selection` here bundles the two truncations, as `FULL_VISITED_B6 - SAFE`):

| corpus | block | reach gap | B capacity gap | ranking/selection gap | total |
|---|---|---:|---:|---:|---:|
| metaqa | ALL | +0.0931 | +0.0506 | +0.1821 | 0.3258 |
| metaqa | hop1 | +0.0000 | +0.0000 | +0.0045 | 0.0045 |
| metaqa | hop2 | +0.0150 | +0.0436 | +0.2087 | 0.2673 |
| metaqa | hop3 | +0.2643 | +0.1081 | +0.3333 | 0.7057 |
| webqsp | ALL | +0.1353 | +0.0007 | +0.0945 | 0.2305 |
| 2wiki_clean | ALL | +0.0330 | +0.0000 | +0.0235 | 0.0565 |
| musique_clean | ALL | +0.0055 | +0.0000 | +0.0310 | 0.0365 |
| hotpotqa_clean | ALL | +0.0200 | +0.0000 | +0.0295 | 0.0495 |
| squad_clean | ALL | +0.0025 | +0.0000 | +0.0100 | 0.0125 |

## T7. Is B a binding constraint? (B sweep under each evidence universe)

What raising B alone buys, holding the evidence universe fixed:

| corpus | block | universe | B=6 | B=50 | B6 -> B50 |
|---|---|---|---:|---:|---:|
| metaqa | ALL | S4_M64 | 0.7212 | 0.7222 | **+0.0010** |
| metaqa | ALL | S4_FULL_ADMITTED | 0.7447 | 0.7467 | **+0.0020** |
| metaqa | ALL | S4_FULL_VISITED | 0.8433 | 0.8939 | **+0.0506** |
| metaqa | hop3 | S4_M64 | 0.3634 | 0.3664 | **+0.0030** |
| metaqa | hop3 | S4_FULL_ADMITTED | 0.3904 | 0.3964 | **+0.0060** |
| metaqa | hop3 | S4_FULL_VISITED | 0.5916 | 0.6997 | **+0.1081** |
| webqsp | ALL | S4_M64 | 0.7886 | 0.7886 | **+0.0000** |
| webqsp | ALL | S4_FULL_ADMITTED | 0.7992 | 0.7992 | **+0.0000** |
| webqsp | ALL | S4_FULL_VISITED | 0.8591 | 0.8598 | **+0.0007** |
| 2wiki_clean | ALL | S4_M64 | 0.9515 | 0.9515 | **+0.0000** |
| 2wiki_clean | ALL | S4_FULL_ADMITTED | 0.9545 | 0.9545 | **+0.0000** |
| 2wiki_clean | ALL | S4_FULL_VISITED | 0.9670 | 0.9670 | **+0.0000** |
| musique_clean | ALL | S4_M64 | 0.9750 | 0.9750 | **+0.0000** |
| musique_clean | ALL | S4_FULL_ADMITTED | 0.9795 | 0.9795 | **+0.0000** |
| musique_clean | ALL | S4_FULL_VISITED | 0.9945 | 0.9945 | **+0.0000** |
| hotpotqa_clean | ALL | S4_M64 | 0.9645 | 0.9645 | **+0.0000** |
| hotpotqa_clean | ALL | S4_FULL_ADMITTED | 0.9680 | 0.9680 | **+0.0000** |
| hotpotqa_clean | ALL | S4_FULL_VISITED | 0.9800 | 0.9800 | **+0.0000** |
| squad_clean | ALL | S4_M64 | 0.9945 | 0.9945 | **+0.0000** |
| squad_clean | ALL | S4_FULL_ADMITTED | 0.9950 | 0.9950 | **+0.0000** |
| squad_clean | ALL | S4_FULL_VISITED | 0.9975 | 0.9975 | **+0.0000** |

Failure-mode split (mutually exclusive by construction: a query is a capacity failure when `|need| > B`, and an evidence failure only when it clears capacity and still cannot be assembled, so raising B moves queries out of capacity into covered-or-evidence):

| corpus | block | B | capacity failures | evidence failures |
|---|---|---:|---:|---:|
| metaqa | ALL | 6 | 246 | 67 |
| metaqa | ALL | 8 | 220 | 78 |
| metaqa | ALL | 12 | 163 | 101 |
| metaqa | ALL | 20 | 105 | 133 |
| metaqa | ALL | 50 | 26 | 186 |
| metaqa | hop3 | 6 | 213 | 59 |
| metaqa | hop3 | 8 | 190 | 70 |
| metaqa | hop3 | 12 | 140 | 92 |
| metaqa | hop3 | 20 | 90 | 124 |
| metaqa | hop3 | 50 | 24 | 176 |
| webqsp | ALL | 6 | 65 | 135 |
| webqsp | ALL | 8 | 48 | 151 |
| webqsp | ALL | 12 | 34 | 165 |
| webqsp | ALL | 20 | 20 | 179 |
| webqsp | ALL | 50 | 7 | 192 |
| 2wiki_clean | ALL | 6 | 0 | 66 |
| 2wiki_clean | ALL | 8 | 0 | 66 |
| 2wiki_clean | ALL | 12 | 0 | 66 |
| 2wiki_clean | ALL | 20 | 0 | 66 |
| 2wiki_clean | ALL | 50 | 0 | 66 |
| musique_clean | ALL | 6 | 0 | 11 |
| musique_clean | ALL | 8 | 0 | 11 |
| musique_clean | ALL | 12 | 0 | 11 |
| musique_clean | ALL | 20 | 0 | 11 |
| musique_clean | ALL | 50 | 0 | 11 |
| hotpotqa_clean | ALL | 6 | 0 | 40 |
| hotpotqa_clean | ALL | 8 | 0 | 40 |
| hotpotqa_clean | ALL | 12 | 0 | 40 |
| hotpotqa_clean | ALL | 20 | 0 | 40 |
| hotpotqa_clean | ALL | 50 | 0 | 40 |
| squad_clean | ALL | 6 | 0 | 5 |
| squad_clean | ALL | 8 | 0 | 5 |
| squad_clean | ALL | 12 | 0 | 5 |
| squad_clean | ALL | 20 | 0 | 5 |
| squad_clean | ALL | 50 | 0 | 5 |

## T8. STEP 7 -- internal work (downstream output is EXACTLY 50 partitions throughout)

| corpus | edges/q | candidates scored/q | visited/q | admitted/q | read/q (M64) | partition accumulations/q M64 -> FULL_VISITED | ms/q (replay) |
|---|---:|---:|---:|---:|---:|---:|---:|
| metaqa | 1505.8 | 1241.0 | 1233.3 | 147.0 | 64 | 64 -> 1233.3 (19.3x) | 25.07 |
| webqsp | 1661.8 | 1213.7 | 1199.0 | 138.4 | 64 | 64 -> 1199.0 (18.7x) | 63.12 |
| 2wiki_clean | 601.1 | 382.6 | 381.1 | 105.7 | 64 | 64 -> 381.1 (6.0x) | 9.14 |
| musique_clean | 3196.0 | 1051.2 | 971.3 | 153.0 | 64 | 64 -> 971.3 (15.2x) | 32.92 |
| hotpotqa_clean | 1892.9 | 1181.9 | 1158.9 | 151.0 | 64 | 64 -> 1158.9 (18.1x) | 23.0 |
| squad_clean | 14181.3 | 2187.2 | 1741.6 | 158.8 | 64 | 64 -> 1741.6 (27.2x) | 250.45 |

## T9. REALISABILITY -- the UNCHANGED frozen F6 selector fed each evidence universe

The oracle rungs above are ceilings. This table asks the separate question the previous phase forced on us: does a *real*, unchanged, parameter-free selector convert any of the extra evidence? Same F6 selector, same B=6, same exact P=50; only the structural evidence universe it is shown differs. `net` is the McNemar net query flip against SAFE.

| corpus | universe | ALL | net vs SAFE | p | sig | churn |
|---|---|---:|---:|---:|---|---:|
| metaqa | SAFE (frozen) | 0.6612 | 0 | - | - | - |
| metaqa | S4_M64 | 0.6612 | +0 | 1 | ns | 4.659 |
| metaqa | S4_FULL_ADMITTED | 0.6582 | -6 | 0.146 | ns | 4.741 |
| metaqa | S4_FULL_VISITED | 0.6537 | -15 | 0.00027 | **SIG** | 4.895 |
| webqsp | SAFE (frozen) | 0.7646 | 0 | - | - | - |
| webqsp | S4_M64 | 0.7646 | +0 | 1 | ns | 4.968 |
| webqsp | S4_FULL_ADMITTED | 0.7646 | +0 | 1 | ns | 4.826 |
| webqsp | S4_FULL_VISITED | 0.7689 | +6 | 0.109 | ns | 4.721 |
| 2wiki_clean | SAFE (frozen) | 0.9435 | 0 | - | - | - |
| 2wiki_clean | S4_M64 | 0.9435 | +0 | 1 | ns | 4.259 |
| 2wiki_clean | S4_FULL_ADMITTED | 0.9430 | -1 | 1 | ns | 4.26 |
| 2wiki_clean | S4_FULL_VISITED | 0.9415 | -4 | 0.289 | ns | 4.348 |
| musique_clean | SAFE (frozen) | 0.9635 | 0 | - | - | - |
| musique_clean | S4_M64 | 0.9635 | +0 | 1 | ns | 4.312 |
| musique_clean | S4_FULL_ADMITTED | 0.9630 | -1 | 1 | ns | 4.321 |
| musique_clean | S4_FULL_VISITED | 0.9625 | -2 | 0.774 | ns | 4.327 |
| hotpotqa_clean | SAFE (frozen) | 0.9505 | 0 | - | - | - |
| hotpotqa_clean | S4_M64 | 0.9505 | +0 | 1 | ns | 5.098 |
| hotpotqa_clean | S4_FULL_ADMITTED | 0.9510 | +1 | 1 | ns | 5.053 |
| hotpotqa_clean | S4_FULL_VISITED | 0.9500 | -1 | 1 | ns | 5.055 |
| squad_clean | SAFE (frozen) | 0.9875 | 0 | - | - | - |
| squad_clean | S4_M64 | 0.9875 | +0 | 1 | ns | 3.974 |
| squad_clean | S4_FULL_ADMITTED | 0.9870 | -1 | 1 | ns | 4.186 |
| squad_clean | S4_FULL_VISITED | 0.9880 | +1 | 1 | ns | 4.24 |

MetaQA per hop (the block where the oracle gain is largest):

| universe | hop1 | hop2 | hop3 | ALL |
|---|---:|---:|---:|---:|
| SAFE (frozen) | 0.9955 | 0.7297 | 0.2583 | 0.6612 |
| S4_M64 | 0.9955 | 0.7297 | 0.2583 | 0.6612 |
| S4_FULL_ADMITTED | 0.9955 | 0.7237 | 0.2553 | 0.6582 |
| S4_FULL_VISITED | 0.9955 | 0.7147 | 0.2508 | 0.6537 |

## T10. STEP 7 -- clean cost isolation (idle CPU, 400 queries/corpus)

The bounded graph search is IDENTICAL across the three universes -- the pruned nodes are already visited, scored and accumulated, so `S4_FULL_VISITED` adds no search and no edges. Only retention and aggregation differ. Downstream exposure is EXACTLY 50 partitions in every row.

| corpus | search ms/q | keep-visited overhead ms/q | aggregate ms/q M64 | aggregate ms/q FULL_VISITED | accumulations/q M64 -> FULL_VISITED | transient bytes/q | partitions/q M64 -> FULL_VISITED | downstream P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| metaqa | 37.2 | +8.0 | 0.287 | 4.007 | 64 -> 1234 (19.4x) | 39,496 | 46 -> 304 | **50** |
| webqsp | 52.7 | +0.5 | 0.295 | 4.906 | 62 -> 1134 (18.3x) | 36,277 | 36 -> 395 | **50** |
| 2wiki_clean | 21.9 | -3.9 | 0.208 | 1.644 | 54 -> 388 (7.2x) | 12,414 | 38 -> 169 | **50** |
| musique_clean | 77.0 | -6.0 | 0.320 | 2.528 | 64 -> 1047 (16.5x) | 33,514 | 28 -> 114 | **50** |
| hotpotqa_clean | 68.1 | -0.2 | 0.133 | 6.231 | 64 -> 1143 (17.9x) | 36,577 | 45 -> 580 | **50** |
| squad_clean | 235.3 | -6.8 | 0.097 | 4.520 | 54 -> 1708 (31.4x) | 54,653 | 16 -> 116 | **50** |

The `keep-visited overhead` column is two independent timed passes differenced, so it carries the full run-to-run noise of the search itself; it is negative on four corpora, i.e. indistinguishable from zero, which is the expected result since nothing is re-scored. The real added cost of the widest universe is the aggregation column: +3.7 ms/q on MetaQA (+10% of search), +6.1 ms/q on Hotpot.
