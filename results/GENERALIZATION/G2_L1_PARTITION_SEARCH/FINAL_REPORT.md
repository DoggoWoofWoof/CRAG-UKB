# G2 -- UNIVERSAL FIXED-P50 PARTITION ROUTER SEARCH

*L1 output = exactly 50 canonical C partitions. No stray nodes. No learned parameters, no dataset identity, no gold at inference, no TEST inspection.*


## 1. Six-dataset BASE (STEP 0)

| corpus | family | dev n | sample rule | docs | partitions | BASE ANY@P50 | BASE ALL@P50 | BASE scope (nodes) | scope % corpus | parity |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|:--|
| **MetaQA** | KB | 1998 | val, hop-balanced (identical to _ta_resid.py / _ta_comb.py) | 40,151 | 401 | 0.9745 | 0.6587 | 5038 | 12.5% | `EXACT` |
| **WebQSP** | KB | 1419 | val+train (val<1000; TEST never touched) | 781,485 | 7,814 | 0.9436 | 0.7618 | 5023 | 0.6% | `EXACT` |
| **2Wiki** | text | 2000 | val | 65,865 | 658 | 1.0000 | 0.9375 | 5028 | 7.6% | `EXACT` |
| **MuSiQue** | text | 2000 | val | 13,672 | 136 | 0.9990 | 0.9565 | 5042 | 36.9% | `EXACT` |
| **HotpotQA** | text | 2000 | val | 507,494 | 5,074 | 0.9950 | 0.9345 | 5082 | 1.0% | `EXACT` |
| **SQuAD** | text | 2000 | val | 19,029 | 190 | 0.9805 | 0.9805 | 5039 | 26.5% | `EXACT` |

Frozen contract on every corpus: `{"K0": 60, "K": 100, "P_MAIN": 50, "SEED_K": 5, "BEAM": 64, "MAX_HOPS": 3, "DEG_CAP": 300, "SMAX": 256, "TOPP": 200}`.

WebQSP reports both ANY (0.9436) and ALL (0.7618); ALL is never silently substituted by ANY anywhere in this report.

## 2. Direct structural-NODE evidence (what the partition router has to reproduce)

Additive node-level residual arms measured *before* the partition-only constraint was imposed (`BASE P50 + node residual`, BASE never evicted). These are the STEP-11 denominators.

| corpus | BASE ALL | RET32 | STRUCT32 | RET64 | STRUCT64 | COMBINED 32+32 | struct newly-cov q | ret newly-cov q |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **MetaQA** (KB) | 0.6587 | 0.6612 (+0.0025) | 0.6977 (+0.0390) | 0.6632 (+0.0045) | 0.7132 (+0.0546) | **0.7002 (+0.0415)** | 78 | 5 |
| **WebQSP** (KB) | 0.7618 | 0.7787 (+0.0169) | 0.7759 (+0.0141) | 0.7815 (+0.0197) | 0.7829 (+0.0211) | **0.7914 (+0.0296)** | 20 | 24 |
| **2Wiki** (text) | 0.9375 | 0.9480 (+0.0105) | 0.9390 (+0.0015) | 0.9500 (+0.0125) | 0.9390 (+0.0015) | **0.9490 (+0.0115)** | 3 | 21 |
| **MuSiQue** (text) | 0.9565 | 0.9770 (+0.0205) | 0.9585 (+0.0020) | 0.9805 (+0.0240) | 0.9590 (+0.0025) | **0.9775 (+0.0210)** | 4 | 41 |
| **HotpotQA** (text) | 0.9345 | 0.9630 (+0.0285) | 0.9380 (+0.0035) | 0.9680 (+0.0335) | 0.9395 (+0.0050) | **0.9635 (+0.0290)** | 7 | 57 |
| **SQuAD** (text) | 0.9805 | 0.9955 (+0.0150) | 0.9810 (+0.0005) | 0.9970 (+0.0165) | 0.9810 (+0.0005) | **0.9955 (+0.0150)** | 1 | 30 |

## 3. Fixed-P50 ORACLE ceiling (STEP 2 -- the gate)

A query is oracle-covered at budget B iff `need = gold_parts \ protected` has `|need| <= B` and `need` is contained in `boundary u challengers`. Evaluation only: gold is used **only** to compute this ceiling, never as an inference feature.

| corpus | BASE | direct STRUCT_NODE32 | B=1 | B=2 | B=4 | B=6 | B=8 | B=12 | UNBOUNDED@B=4 | mean need @B=1 | frac need<=1 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **MetaQA** | 0.6587 | 0.6977 | 0.7072 | 0.7172 | 0.7202 | 0.7222 | 0.7222 | 0.7227 | 0.8509 | 2.964 | 0.753 |
| **WebQSP** | 0.7618 | 0.7759 | 0.7956 | 0.7999 | 0.8013 | 0.8013 | 0.8013 | 0.8013 | 0.9401 | 1.200 | 0.861 |
| **2Wiki** | 0.9375 | 0.9390 | 0.9610 | 0.9610 | 0.9610 | 0.9610 | 0.9610 | 0.9610 | 1.0000 | 0.067 | 0.997 |
| **MuSiQue** | 0.9565 | 0.9585 | 0.9905 | 0.9920 | 0.9920 | 0.9920 | 0.9920 | 0.9920 | 1.0000 | 0.049 | 0.998 |
| **HotpotQA** | 0.9345 | 0.9380 | 0.9695 | 0.9700 | 0.9700 | 0.9700 | 0.9700 | 0.9700 | 1.0000 | 0.070 | 0.996 |
| **SQuAD** | 0.9805 | 0.9810 | 0.9975 | 0.9975 | 0.9975 | 0.9975 | 0.9975 | 0.9975 | 1.0000 | 0.021 | 1.000 |

Challenger-family ceilings at B=4 -- does structure add *reach* over retrieval continuation?

| corpus | STRUCT only | RET only | COMBINED | COMBINED - best single |
|---|--:|--:|--:|--:|
| **MetaQA** | 0.7122 | 0.6687 | **0.7202** | +0.0080 |
| **WebQSP** | 0.7851 | 0.7822 | **0.8013** | +0.0162 |
| **2Wiki** | 0.9420 | 0.9575 | **0.9610** | +0.0035 |
| **MuSiQue** | 0.9695 | 0.9895 | **0.9920** | +0.0025 |
| **HotpotQA** | 0.9410 | 0.9685 | **0.9700** | +0.0015 |
| **SQuAD** | 0.9850 | 0.9960 | **0.9975** | +0.0015 |

## 4. Search rounds and surviving families (STEPS 4-8)

| round | families | grid | configs/corpus | runtime s | best universal config | worst dALL | macro dALL | sig regressions |
|---|---|---|--:|--:|---|--:|--:|--:|
| round1 | F1 F2 F3 F4(T1-3) F5, asymmetric | B 1-12, M=32, S1-S4 | 140 | 637 | `B2_S4_F1_Ms32_Mr32` | -0.0005 | +0.0022 | 0 |
| round2 | F6 F7 F8, SYMMETRIC competition | B 1-24, M=32, S1-S4 | 84 | 1386 | `B4_S4_F6_Ms32_Mr32` | +0.0021 | +0.0063 | 0 |
| round3 | F6, widened challenger source | B 2-8, M_s/M_r in {32,64,128}, S1/S4 | 72 | 3897 | `B6_S4_F6_Ms64_Mr32` | +0.0025 | +0.0069 | 0 |
| round4 | FA, symmetric + multi-node support gate T (F6 control) | B 4-8, M_s in {64,128}, M_r in {32,64}, S1/S4 | 72 | 2571 | `B6_S4_F6_Ms64_Mr32` | +0.0025 | +0.0069 | 0 |

**Family separation (round 2, S4, B=4)** -- the result that decides the architecture:

| corpus | F7 struct-only | F8 ret-only | F6 struct+ret |
|---|--:|--:|--:|
| **MetaQA** | +0.0110 | -0.0060 | **+0.0030** |
| **WebQSP** | +0.0000 | -0.0021 | **+0.0021** |
| **2Wiki** | -0.0005 | +0.0085 | **+0.0075** |
| **MuSiQue** | -0.0045 | +0.0050 | **+0.0045** |
| **HotpotQA** | -0.0005 | +0.0150 | **+0.0145** |
| **SQuAD** | -0.0010 | +0.0085 | **+0.0060** |

Structure-only helps the KB corpora and hurts the text corpora; retrieval-only does the opposite. Only the union is non-negative on all six -- which is why a *universal* router needs both channels and cannot be either one alone.

**Refinement H rejected on evidence (round 4, B=6, S4, M_s=64, M_r=32).** Requiring a challenger to have >= T structural nodes in the same partition before it may compete:

| corpus | F6 ungated | FAT2 (T=2) | FAT3 (T=3) | churn F6 / FAT2 / FAT3 |
|---|--:|--:|--:|--:|
| **MetaQA** (KB) | +0.0025 | -0.0045 | -0.0060 | 4.66 / 4.54 / 4.39 |
| **WebQSP** (KB) | +0.0028 | +0.0000 | +0.0000 | 4.97 / 4.08 / 3.04 |
| **2Wiki** (text) | +0.0060 | +0.0070 | +0.0075 | 4.26 / 4.08 / 3.75 |
| **MuSiQue** (text) | +0.0070 | +0.0095 | +0.0090 | 4.31 / 3.06 / 2.20 |
| **HotpotQA** (text) | +0.0160 | +0.0170 | +0.0175 | 5.10 / 4.92 / 4.71 |
| **SQuAD** (text) | +0.0070 | +0.0075 | +0.0080 | 3.97 / 3.30 / 2.95 |

The gate improves **all four text corpora** and simultaneously destroys **both KB corpora**. The mechanism is direct: on a KB substrate the decisive structural evidence is frequently a *single* node that a second hop reaches inside the right partition, so a multi-node requirement discards exactly the signal that makes structure worth having; on text substrates single-node structural evidence is mostly noise, so the same rule acts as a clean filter. Adopting it would therefore be a corpus-specific rule, which the architecture contract forbids -- so it is rejected despite winning on 4 of 6 corpora. Round 4 produced **no Pareto improvement**: its universal winner is bit-identical to round 3's.

## 5. FINAL universal configuration -- full DEV (STEP 10)

```json
{
 "name": "B6_S4_F6_Ms64_Mr32",
 "B": 6,
 "M_struct": 64,
 "M_ret": 32,
 "agg": "S4",
 "fusion": "F6",
 "T": 1,
 "K0": 60,
 "P_MAIN": 50,
 "learned_parameters": 0,
 "uses_gold_at_inference": false,
 "uses_dataset_identity": false,
 "output": "exactly 50 canonical C partitions, no stray nodes"
}
```

| corpus | BASE ANY | ANY | BASE ALL | ALL | dALL | newly ALL-cov | newly uncov | net | McNemar p | |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| **MetaQA** | 0.9745 | 0.9785 | 0.6587 | 0.6612 | **+0.0025** | 22 | 17 | +5 | 0.5224 | ns |
| **WebQSP** | 0.9436 | 0.9457 | 0.7618 | 0.7646 | **+0.0028** | 25 | 21 | +4 | 0.65874 | ns |
| **2Wiki** | 1.0000 | 1.0000 | 0.9375 | 0.9435 | **+0.0060** | 17 | 5 | +12 | 0.0169 | **sig** |
| **MuSiQue** | 0.9990 | 0.9995 | 0.9565 | 0.9635 | **+0.0070** | 25 | 11 | +14 | 0.02882 | **sig** |
| **HotpotQA** | 0.9950 | 0.9970 | 0.9345 | 0.9505 | **+0.0160** | 36 | 4 | +32 | 0.0 | **sig** |
| **SQuAD** | 0.9805 | 0.9875 | 0.9805 | 0.9875 | **+0.0070** | 17 | 3 | +14 | 0.00258 | **sig** |

worst-dataset dALL **+0.0025** | macro dALL **+0.0069** | positive on 6/6 | significant improvements 4/6 | **significant regressions 0/6** | net queries +81 | mean churn 4.54

## 6. MetaQA per-hop ALL

| hop | n | BASE ALL | final ALL | delta |
|---|--:|--:|--:|--:|
| 1 | 666 | 0.9955 | 0.9955 | +0.0000 |
| 2 | 666 | 0.7237 | 0.7297 | +0.0060 |
| 3 | 666 | 0.2568 | 0.2583 | +0.0015 |

## 7. Gold-bearing partitions admitted vs evicted, and churn (STEP 10)

| corpus | admitted | evicted | net | churn/query | churn max | queries with 0 churn | final scope nodes | BASE scope | scope growth |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **MetaQA** | 257 | 183 | +74 | 4.659 | 6 | 0 (0%) | 5036 | 5038 | -0.04% |
| **WebQSP** | 91 | 90 | +1 | 4.968 | 6 | 12 (1%) | 5022 | 5023 | -0.03% |
| **2Wiki** | 20 | 6 | +14 | 4.259 | 6 | 59 (3%) | 5027 | 5028 | -0.02% |
| **MuSiQue** | 26 | 11 | +15 | 4.312 | 6 | 31 (2%) | 5042 | 5042 | -0.01% |
| **HotpotQA** | 39 | 5 | +34 | 5.098 | 6 | 0 (0%) | 5074 | 5082 | -0.17% |
| **SQuAD** | 17 | 3 | +14 | 3.974 | 6 | 150 (8%) | 5037 | 5039 | -0.03% |

Scope is *not* the source of the gain: the final hard union stays at the BASE budget because the output is always exactly 50 partitions. Growth is only the difference in partition sizes between the evicted and admitted slots.

## 8. Structural vs retrieval challenger complementarity (STEP 12)

Which channel supplied the partition that actually flipped each newly ALL-covered query:

| corpus | struct only | ret only | both | boundary reorder |
|---|--:|--:|--:|--:|
| **MetaQA** | 20 | 0 | 2 | 0 |
| **WebQSP** | 19 | 5 | 1 | 0 |
| **2Wiki** | 0 | 16 | 1 | 0 |
| **MuSiQue** | 7 | 13 | 5 | 0 |
| **HotpotQA** | 0 | 24 | 12 | 0 |
| **SQuAD** | 1 | 9 | 7 | 0 |

Node-level marginal complementarity from the direct experiment, both directions:

| corpus | STRUCT after RET | RET after STRUCT | overlap \|RET32 n STRUCT32\|/32 |
|---|--:|--:|--:|
| **MetaQA** | +0.0390 (net +78, p=0.0) | +0.0025 (net +5, p=0.0625) | 1.9% |
| **WebQSP** | +0.0127 (net +18, p=1e-05) | +0.0155 (net +22, p=0.0) | 4.0% |
| **2Wiki** | +0.0010 (net +2, p=0.5) | +0.0100 (net +20, p=0.0) | 0.7% |
| **MuSiQue** | +0.0005 (net +1, p=1.0) | +0.0190 (net +38, p=0.0) | 3.8% |
| **HotpotQA** | +0.0005 (net +1, p=1.0) | +0.0255 (net +51, p=0.0) | 2.8% |
| **SQuAD** | +0.0000 (net +0, p=1.0) | +0.0145 (net +29, p=0.0) | 3.0% |

## 9. STEP 11 -- fraction of the direct structural-node gain recovered at exactly P50

`RECOVERED_STRUCTURAL_GAIN_FRACTION = dALL(best fixed-P50) / dALL(direct STRUCT_NODE32)`

| corpus | direct STRUCT_NODE32 dALL | direct COMBINED node dALL | fixed-P50 dALL | **RECOVERED** | vs direct COMBINED | oracle dALL @B | frac of oracle captured |
|---|--:|--:|--:|--:|--:|--:|--:|
| **MetaQA** | +0.0390 | +0.0415 | +0.0025 | **0.06x** | 0.06x | +0.0636 | 3.9% |
| **WebQSP** | +0.0141 | +0.0296 | +0.0028 | **0.20x** | 0.10x | +0.0395 | 7.1% |
| **2Wiki** | +0.0015 | +0.0115 | +0.0060 | **4.00x** | 0.52x | +0.0235 | 25.5% |
| **MuSiQue** | +0.0020 | +0.0210 | +0.0070 | **3.50x** | 0.33x | +0.0355 | 19.7% |
| **HotpotQA** | +0.0035 | +0.0290 | +0.0160 | **4.57x** | 0.55x | +0.0355 | 45.1% |
| **SQuAD** | +0.0005 | +0.0150 | +0.0070 | **14.00x** | 0.47x | +0.0170 | 41.2% |

Macro RECOVERED_STRUCTURAL_GAIN_FRACTION = **4.39x**

*Read the ratio with its denominator.* STEP 11 defines the denominator as the direct STRUCT_NODE32 gain, which on the four text corpora is nearly zero (+0.0005 to +0.0035, i.e. 1-7 queries), so a ratio above 1 there means only that the router's *retrieval* channel contributes more than structure ever did on those corpora -- not that structure was amplified. The informative cells are the two KB corpora, where the structural signal is real, and the 'vs direct COMBINED' column, which compares against the full node-level residual gain that the 50-partition constraint had to give up.

## 10. Compute, memory, Modal (STEP 10)

| corpus | fp16 path | predicted peak GB | measured peak RSS GB | BASE s | struct traversal s | x BASE | edges traversed | cache build s | route ms/query | executed on |
|---|:--|--:|--:|--:|--:|--:|--:|--:|--:|---|
| **MetaQA** | no | 0.25 | 0.13 | 5.0 | 31.9 | 6.4x | 3,008,656 | 188.7 | 0.080 | local |
| **WebQSP** | yes | 2.43 | 0.2 | 6.6 | 27.9 | 4.2x | 2,358,159 | 116.1 | 0.081 | local |
| **2Wiki** | no | 0.41 | 0.2 | 3.8 | 10.6 | 2.8x | 1,202,171 | 197.4 | 0.083 | local |
| **MuSiQue** | no | 0.08 | 0.2 | 4.0 | 45.9 | 11.5x | 6,392,042 | 209.8 | 0.042 | local |
| **HotpotQA** | yes | 1.60 | 0.2 | 6.9 | 34.3 | 5.0x | 3,785,828 | 219.0 | 0.169 | local |
| **SQuAD** | no | 0.12 | 0.2 | 3.8 | 187.6 | 49.4x | 28,362,633 | 253.5 | 0.092 | local |

No corpus required Modal. The two corpora whose fp32 node matrices do not fit in 15.7 GB (WebQSP 781,485x1536 = 4.8 GB, HotpotQA 507,494x1536 = 3.1 GB) ran through the frozen `EXHAUSTIVE_SHARDED_FP16_IP` path (fp16 storage, fp32 compute), predicted peak 2.43 / 1.58 GB, well under the 70% trigger. `NEW_ENCODER_PASSES = 0` throughout.

## 11. Universal Pareto frontier (STEP 9)

368 configurations evaluated across all rounds; 193 have zero significant regressions on any of the six corpora. Non-dominated on (worst-dataset dALL, macro dALL):

| round | config | worst dALL | macro dALL | churn | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| round3 | `B6_S4_F6_Ms64_Mr32` | **+0.0025** | +0.0069 | 4.54 | +0.0025 | +0.0028 | +0.0060 | +0.0070 | +0.0160 | +0.0070 |
| round3 | `B8_S4_F6_Ms64_Mr64` | **+0.0021** | +0.0077 | 5.81 | +0.0030 | +0.0021 | +0.0070 | +0.0090 | +0.0175 | +0.0075 |
| round3 | `B8_S4_F6_Ms32_Mr64` | **-0.0021** | +0.0079 | 5.75 | +0.0040 | -0.0021 | +0.0070 | +0.0110 | +0.0195 | +0.0080 |

The frontier is a genuine trade-off, not a hidden regression: raising B past the selected value buys macro-average dALL by spending WebQSP, which is the binding constraint. The lexicographic rule in STEP 9 (no significant regression anywhere, then maximise the worst dataset) selects the top row.

## 12. Cross-dataset mechanism analysis (STEP 12)

| corpus | family | struct newly-cov (direct nodes) | ret newly-cov (direct nodes) | struct:ret ratio | mean need @B=1 | frac need<=1 | decisive struct-only at P50 |
|---|---|--:|--:|--:|--:|--:|--:|
| **MetaQA** | KB | 78 | 5 | 15.6x | 2.964 | 0.753 | 20 |
| **WebQSP** | KB | 20 | 24 | 0.8x | 1.200 | 0.861 | 19 |
| **2Wiki** | text | 3 | 21 | 0.1x | 0.067 | 0.997 | 0 |
| **MuSiQue** | text | 4 | 41 | 0.1x | 0.049 | 0.998 | 7 |
| **HotpotQA** | text | 7 | 57 | 0.1x | 0.070 | 0.996 | 0 |
| **SQuAD** | text | 1 | 30 | 0.0x | 0.021 | 1.000 | 1 |

Dataset identity is never an inference feature; this table is post-hoc explanation only.

## 13. Recommendation, canonical configuration, verdict

**Recommendation: PROTECTED-CORE + COMBINED STRUCTURAL/RETRIEVAL BOUNDARY.**

Rejected alternatives, on evidence: *KEEP BASE* is dominated (the selected router is non-negative on all six corpora and significantly positive on several); *PROTECTED-CORE + STRUCTURAL BOUNDARY only* (F7) helps the two KB corpora and significantly hurts the text corpora, so it is not universal.

Exact canonical configuration:

```
FINAL_PARTITIONS      = 50            # exactly 50 canonical C partitions, no stray nodes
B (replaceable slots) = 6             # protected core = base_rank[:44]
M_struct              = 64            # structural residual nodes -> partition evidence
M_ret                 = 32            # retrieval-continuation nodes -> partition evidence
aggregation           = S4            # MULTI_SIGNAL_RRF over 4 partition rankings
                                     #   best structural node rank, node-count support,
                                     #   min hop, best directional s_dir -- fixed RRF, no weights
boundary fusion       = F6            # SYMMETRIC boundary competition
                                     #   incumbents and challengers scored on the same three
                                     #   channels (canonical, structural, retrieval), so B is a
                                     #   CAP on churn rather than a forced churn
K0                    = 60            # canonical RRF constant, unchanged
structural discovery  = universal RRF seeds, beam 64, MAX_HOPS 3, DEG_CAP 300, SEED_K 5
learned parameters    = 0        dataset branches = 0        gold at inference = no
```

### What this does *not* claim -- the measured limit

The router is universally positive, but on the two corpora where structural evidence is genuinely informative it recovers only 0.06x (MetaQA) and 0.20x (WebQSP) of the direct structural-node gain -- 3.9% and 7.1% of its own fixed-P50 oracle ceiling.

STEP 2 already settled where that loss comes from, and it is **not** the partition abstraction: at B=1 the combined-challenger oracle already exceeds the entire direct-node gain on every corpus (MetaQA 0.7072 vs 0.6977; WebQSP 0.7956 vs 0.7759), and capacity saturates by B=2. Reach and capacity are sufficient. What binds is the **eviction cost of a parameter-free scoring rule**: on MetaQA the selected router admits 257 gold-bearing partitions and evicts 183 to pay for them (net +74); on WebQSP 91 admitted against 90 evicted (net +1). Every promotion is funded by a demotion, and across 368 configurations and four mechanism-guided rounds no parameter-free rule separated the two well enough to keep more than a small fraction of the ceiling on a KB substrate.

The one rule that separates them better -- the multi-node support gate of section 4 -- is regime-specific in the opposite direction (it helps all four text corpora and destroys both KB corpora), so it cannot be adopted without a dataset branch. That is the honest boundary of this result.

**UNIVERSAL_FIXED_P50_PARTITION_ROUTER_FOUND = YES**

One global configuration, no dataset selector, improves ALL-coverage on all six corpora (worst +0.0025, macro +0.0069, 4/6 significant, 0 significant regressions, +81 net queries) while emitting exactly 50 partitions.

## 14. Contract and stop-condition compliance

| requirement | status |
|---|:--|
| L1 output = partitions only, exactly P=50 | asserted per query in every evaluation (`len(set(final)) == 50`) |
| No stray nodes bypass partitions | structural nodes are used only as EVIDENCE; the emitted object is a list of partition ids |
| No MLP / learned router / attention / learned fusion / learned lambda | none; scoring is fixed RRF at the canonical K0=60 |
| No fitted score / target calibration | no value is fitted to any target |
| No dataset-specific branch or corpus-specific hyperparameter | one global config for all six corpora |
| No gold at inference | gold is used only for the STEP-2 ceiling and for scoring the metrics |
| No benchmark-specific entity annotations / special MetaQA logic | universal RRF seeds only; `metaqa_ent_*` and bracket entity seeds never used |
| No TEST inspection | DEV only; WebQSP extended with TRAIN, never TEST |
| No L2 / L3 run, no SQuAD-L2 repair, no MetaQA-L2 build | not run |
| No training | no model was trained |
| No topology / repartition / encoder / K / P change | MASTER_TOPOLOGY=C, K=100, P_MAIN=50 unchanged; NEW_ENCODER_PASSES=0 |
| Single owner per config, unique run dirs, manifest | lock file per cache; one scoreboard per round; a duplicate owner was detected and killed |
