# L1 ARCHITECTURAL CEILING AUDIT — TABLES

Gold is used for oracle evaluation only; nothing here is an inference method. `base_rank` replay parity vs the cached canonical ranking is **EXACT on all six corpora**.


## T1 — Full-universe oracle ALL by partition budget (= the representational ceiling)

An oracle free to pick any P partitions from the complete canonical universe covers a query exactly when its gold evidence fits in P partitions, so this column *is* `GOLD_PARTITION_COUNT <= P`. ANY is budget-independent at P>=1.

| block | npart | P=10 | P=25 | P=50 | P=75 | P=100 | P=150 | ANY |
|---|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 401 | 0.9955 | 0.9985 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MetaQA hop2 | 401 | 0.9234 | 0.9745 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MetaQA hop3 | 401 | 0.6622 | 0.8498 | 0.9640 | 0.9940 | 0.9985 | 1.0000 | 1.0000 |
| MetaQA aggregate | 401 | 0.8604 | 0.9409 | 0.9870 | 0.9980 | 0.9995 | 1.0000 | 1.0000 |
| WebQSP | 7814 | 0.9316 | 0.9803 | 0.9951 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2Wiki | 658 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| MuSiQue | 136 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| HotpotQA | 5074 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SQuAD | 190 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## T2 — Required gold-partition count: how many partitions must be selected at all

| block | mean | median | p75 | p90 | p95 | p99 | max | =1 | =2 | =3 | =4 | >=5 | >=10 | >25 | >50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 1.61 | 1 | 2 | 3 | 4 | 7 | 34 | 0.718 | 0.150 | 0.066 | 0.035 | 0.032 | 0.004 | 0.002 | 0.000 |
| MetaQA hop2 | 3.99 | 2 | 3 | 8 | 17 | 43 | 57 | 0.486 | 0.179 | 0.092 | 0.048 | 0.195 | 0.084 | 0.025 | 0.003 |
| MetaQA hop3 | 11.75 | 6 | 14 | 32 | 44 | 68 | 104 | 0.182 | 0.113 | 0.090 | 0.057 | 0.559 | 0.354 | 0.150 | 0.036 |
| MetaQA aggregate | 5.78 | 2 | 5 | 15 | 28 | 55 | 104 | 0.462 | 0.147 | 0.083 | 0.046 | 0.262 | 0.148 | 0.059 | 0.013 |
| WebQSP | 3.66 | 1 | 3 | 7 | 15 | 37 | 69 | 0.521 | 0.151 | 0.100 | 0.064 | 0.164 | 0.072 | 0.020 | 0.005 |
| 2Wiki | 1.98 | 2 | 2 | 3 | 4 | 4 | 4 | 0.236 | 0.599 | 0.114 | 0.051 | 0.000 | 0.000 | 0.000 | 0.000 |
| MuSiQue | 1.77 | 2 | 2 | 3 | 3 | 3 | 4 | 0.372 | 0.490 | 0.128 | 0.010 | 0.000 | 0.000 | 0.000 | 0.000 |
| HotpotQA | 1.66 | 2 | 2 | 2 | 2 | 2 | 2 | 0.336 | 0.664 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| SQuAD | 1.00 | 1 | 1 | 1 | 1 | 1 | 1 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## T3 — Canonical Dense+SPLADE ranking gap

`worst required-gold rank` is the variable that governs ALL coverage: a top-P canonical cut covers the query iff worst < P. Ranks are 0-indexed over the full universe.

| block | npart | best med | best p90 | **worst med** | worst p75 | worst p90 | worst p95 | worst p99 | worst max |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 401 | 1 | 8 | **2** | 6 | 12 | 17 | 29 | 365 |
| MetaQA hop2 | 401 | 2 | 20 | **14** | 57 | 145 | 309 | 396 | 400 |
| MetaQA hop3 | 401 | 2 | 19 | **218** | 360 | 393 | 397 | 400 | 400 |
| MetaQA aggregate | 401 | 2 | 14 | **11** | 125 | 356 | 389 | 399 | 400 |
| WebQSP | 7814 | 2 | 29 | **9** | 46 | 202 | 1373 | 7500 | 7812 |
| 2Wiki | 658 | 2 | 6 | **6** | 11 | 24 | 68 | 361 | 647 |
| MuSiQue | 136 | 0 | 4 | **3** | 11 | 28 | 47 | 99 | 133 |
| HotpotQA | 5074 | 3 | 11 | **7** | 16 | 35 | 75 | 2811 | 4857 |
| SQuAD | 190 | 3 | 16 | **3** | 7 | 16 | 27 | 79 | 166 |

## T4 — Coverage from simply taking the canonical top-P (no router at all)

| block | P=25 | P=50 | P=75 | P=100 | P=150 | full-universe oracle P=50 | ranking gap at P=50 |
|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 0.9835 | 0.9955 | 0.9985 | 0.9985 | 0.9985 | 1.0000 | **+0.0045** |
| MetaQA hop2 | 0.6021 | 0.7237 | 0.8078 | 0.8574 | 0.9009 | 0.9970 | **+0.2733** |
| MetaQA hop3 | 0.2042 | 0.2568 | 0.2973 | 0.3243 | 0.3994 | 0.9640 | **+0.7072** |
| MetaQA aggregate | 0.5966 | 0.6587 | 0.7012 | 0.7267 | 0.7663 | 0.9870 | **+0.3283** |
| WebQSP | 0.6603 | 0.7618 | 0.8069 | 0.8407 | 0.8795 | 0.9951 | **+0.2333** |
| 2Wiki | 0.9005 | 0.9375 | 0.9535 | 0.9590 | 0.9670 | 1.0000 | **+0.0625** |
| MuSiQue | 0.8855 | 0.9565 | 0.9795 | 0.9905 | 1.0000 | 1.0000 | **+0.0435** |
| HotpotQA | 0.8525 | 0.9345 | 0.9495 | 0.9595 | 0.9680 | 1.0000 | **+0.0655** |
| SQuAD | 0.9435 | 0.9805 | 0.9890 | 0.9940 | 0.9990 | 1.0000 | **+0.0195** |

## T5 — MANDATORY MetaQA ceiling comparison

| method / ceiling | hop1 | hop2 | hop3 | aggregate |
|---|---|---|---|---|
| BASE P50 | 0.9955 | 0.7237 | 0.2568 | 0.6587 |
| SAFE F6 | 0.9955 | 0.7297 | 0.2583 | 0.6612 |
| R1 LIFT | 0.9955 | 0.7477 | 0.2718 | 0.6717 |
| current-pool oracle B6 | 0.9970 | 0.8033 | 0.3634 | 0.7212 |
| PPR-expanded-pool oracle B6 | 0.9970 | 0.8273 | 0.4715 | 0.7653 |
| full-universe oracle P25 | 0.9985 | 0.9745 | 0.8498 | 0.9409 |
| **full-universe oracle P50** | **1.0000** | **0.9970** | **0.9640** | **0.9870** |
| full-universe oracle P75 | 1.0000 | 1.0000 | 0.9940 | 0.9980 |
| full-universe oracle P100 | 1.0000 | 1.0000 | 0.9985 | 0.9995 |
| full-universe oracle P150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## T6 — Constraint ladder: which frozen constraint actually binds

Each row keeps the partition abstraction and switches ONE constraint off. Gold-evaluated.

| ladder step | MetaQA hop2 | MetaQA hop3 | MetaQA agg | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD |
|---|---|---|---|---|---|---|---|---|
| actual router (SAFE F6) | 0.7297 | 0.2583 | 0.6612 | 0.7646 | 0.9435 | 0.9635 | 0.9505 | 0.9875 |
| + perfect selector, current pool, B=6 | 0.8033 | 0.3634 | 0.7212 | 0.7886 | 0.9515 | 0.9750 | 0.9645 | 0.9945 |
| + perfect selector, current pool, B unlimited | 0.8033 | 0.3664 | 0.7222 | 0.7886 | 0.9515 | 0.9750 | 0.9645 | 0.9945 |
| + perfect selector, unlimited pool, B=6 | 0.9505 | 0.6802 | 0.8769 | 0.9542 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| + abstraction only (gold fits in 50 partitions) | 0.9970 | 0.9640 | 0.9870 | 0.9951 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## T7 — WebQSP control (full-universe oracle)

| budget | ALL | ANY |
|---|---|---|
| P=25 | 0.9803 | 1.0000 |
| P=50 | 0.9951 | 1.0000 |
| P=75 | 1.0000 | 1.0000 |
| P=100 | 1.0000 | 1.0000 |
| P=150 | 1.0000 | 1.0000 |

Required gold-partition count: mean 3.66, median 1, p90 7, p99 37, max 69; 52.1% need 1, 16.4% need >=5, 0.5% need >50. npart = 7814.


## T8 — Text controls

| corpus | npart | BASE P50 | SAFE router | full-universe oracle P50 | full-universe oracle P100 | ranking gap |
|---|---|---|---|---|---|---|
| 2Wiki | 658 | 0.9375 | 0.9435 | 1.0000 | 1.0000 | **+0.0565** |
| MuSiQue | 136 | 0.9565 | 0.9635 | 1.0000 | 1.0000 | **+0.0365** |
| HotpotQA | 5074 | 0.9345 | 0.9505 | 1.0000 | 1.0000 | **+0.0495** |
| SQuAD | 190 | 0.9805 | 0.9875 | 1.0000 | 1.0000 | **+0.0125** |

## T9 — MetaQA hop3 architectural decomposition

All 494 hop3 queries the SAFE router fails, assigned to exactly one category. Precedence: A (uncoverable at any P) -> B (|gold|>50) -> D (required partition outside boundary+challengers) -> C (in pool but >B=6 outside the protected core) -> E (in pool and fits B, selector chose wrong)

| category | definition | count | % of uncovered hop3 |
|---|---|---|---|
| **A_REPRESENTATION_LIMIT** | not coverable at any budget (gold unmapped / no gold) | 0 | 0.00% |
| **B_P50_CAPACITY_LIMIT** | more than 50 gold partitions — needs a larger P | 24 | 4.86% |
| **D_CURRENT_POOL_LIMIT** | a required partition is outside boundary ∪ challengers | 398 | 80.57% |
| **C_RANKING_LIMIT** | all required in pool, but >B=6 of them sit outside the protected core | 2 | 0.40% |
| **E_SELECTOR_LIMIT** | all required in pool and fits B=6 — the selector chose wrong | 70 | 14.17% |

Overlap disclosure: of the 398 category-D queries, **187 (47.0%)** would *also* fail on the B=6 swap cap even with an unlimited candidate pool. Widening the pool alone does not fix them.


## T10 — R1 ruling support: gain against each available headroom (MetaQA)

| level | SAFE F6 | R1 | R1 gain | full-universe P50 oracle | R1 gain / P50-oracle headroom | current-pool B6 oracle | R1 gain / selector headroom |
|---|---|---|---|---|---|---|---|
| hop2 | 0.7297 | 0.7477 | +0.0180 | 0.9970 | **6.7%** | 0.8033 | 24.5% |
| hop3 | 0.2583 | 0.2718 | +0.0135 | 0.9640 | **1.9%** | 0.3634 | 12.8% |
| aggregate | 0.6612 | 0.6717 | +0.0105 | 0.9870 | **3.2%** | 0.7212 | 17.5% |
