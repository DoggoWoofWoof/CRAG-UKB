# L1 ARCHITECTURAL CEILING AUDIT

**Question.** Is further L1 selector work scientifically worth doing at all?

**Scope.** DEV only, six corpora. Gold used for oracle evaluation only. No new routing method, no
RRF variants, no PPR, no set-aware routing, no new structural features, no L2/L3, no training, no
TEST. `R0 / B6_S4_F6_Ms64_Mr32` kept as SAFE_REFERENCE_L1; `R1_RRF_LIFT` kept as a diagnostic
candidate, neither promoted nor rejected.

Tables: [TABLES.md](TABLES.md) (T1–T10). Raw: `ceiling_audit.json`. Script:
`scratchpad/_l1ac_audit.py`, `scratchpad/_l1ac_tables.py`. Runtime 33 s locally, no Modal.

**Verdict up front.**

```
RANKING_LIMITED                                   (with a 4.86% hop3 capacity tail)
IS_FURTHER_L1_SELECTOR_OPTIMIZATION_JUSTIFIED  =  NO
```

The partition abstraction is not the problem and P=50 is not the problem. The canonical ranking is.
But within that ranking failure, the *selector* — the boundary arithmetic the last two phases
optimised — accounts for **14.17%** of MetaQA hop3 failures and has a total remaining ceiling of
**+0.1051**. **80.57%** are queries whose required partitions never enter the candidate pool at all.

---

## 1–2. Full-universe oracle, and what "oracle P50" means

An oracle free to choose any P partitions from the complete canonical universe covers a query for
ALL exactly when the query's gold evidence fits inside P partitions. So the STEP 1 oracle and the
STEP 2 "pure representational ceiling" `GOLD_PARTITION_COUNT <= P` are **the same quantity**. It is
computed once and named as such; no graph traversal is involved.

Gold mapping is verified total, not assumed: `hard` assigns every corpus doc to exactly one
partition and gold partitions are built as `hard[g]` over gold rows, so no gold doc can be unmapped.
**0 queries on all six corpora have zero gold partitions**, and `ANY = 1.0000` everywhere. Canonical
`base_rank` replay parity vs cache: **EXACT on all six**.

| block | npart | P=10 | P=25 | **P=50** | P=75 | P=100 | P=150 |
|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 401 | 0.9955 | 0.9985 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| MetaQA hop2 | 401 | 0.9234 | 0.9745 | **0.9970** | 1.0000 | 1.0000 | 1.0000 |
| MetaQA hop3 | 401 | 0.6622 | 0.8498 | **0.9640** | 0.9940 | 0.9985 | 1.0000 |
| MetaQA agg | 401 | 0.8604 | 0.9409 | **0.9870** | 0.9980 | 0.9995 | 1.0000 |
| WebQSP | 7814 | 0.9316 | 0.9803 | **0.9951** | 1.0000 | 1.0000 | 1.0000 |
| 2Wiki | 658 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| MuSiQue | 136 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| HotpotQA | 5074 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| SQuAD | 190 | 1.0000 | 1.0000 | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

**P=50 is not remotely restrictive.** It is exactly achievable on 96.4% of MetaQA hop3, 99.7% of
hop2, 99.5% of WebQSP and 100% of every text corpus. Raising P to 150 buys +0.036 on hop3 and
**nothing anywhere else**. The entire budget question is worth at most 3.6 points on the hardest
slice of one corpus.

## 3. Required gold-partition count

| block | mean | median | p75 | p90 | p95 | p99 | max | =1 | =2 | =3 | =4 | ≥5 | ≥10 | >25 | >50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 1.61 | 1 | 2 | 3 | 4 | 7 | 34 | .718 | .150 | .066 | .035 | .032 | .004 | .002 | .000 |
| MetaQA hop2 | 3.99 | 2 | 3 | 8 | 17 | 43 | 57 | .486 | .179 | .092 | .048 | .195 | .084 | .025 | .003 |
| MetaQA hop3 | 11.75 | 6 | 14 | 32 | 44 | 68 | 104 | .182 | .113 | .090 | .057 | .559 | .354 | .150 | **.036** |
| MetaQA agg | 5.78 | 2 | 5 | 15 | 28 | 55 | 104 | .462 | .147 | .083 | .046 | .262 | .148 | .059 | .013 |
| WebQSP | 3.66 | 1 | 3 | 7 | 15 | 37 | 69 | .521 | .151 | .100 | .064 | .164 | .072 | .020 | .005 |
| 2Wiki | 1.98 | 2 | 2 | 3 | 4 | 4 | **4** | .236 | .599 | .114 | .051 | .000 | .000 | .000 | .000 |
| MuSiQue | 1.77 | 2 | 2 | 3 | 3 | 3 | **4** | .372 | .490 | .128 | .010 | .000 | .000 | .000 | .000 |
| HotpotQA | 1.66 | 2 | 2 | 2 | 2 | 2 | **2** | .336 | .664 | .000 | .000 | .000 | .000 | .000 | .000 |
| SQuAD | 1.00 | 1 | 1 | 1 | 1 | 1 | **1** | 1.000 | .000 | .000 | .000 | .000 | .000 | .000 | .000 |

MetaQA hop3 genuinely is a multi-evidence problem — median 6 partitions, 35.4% need ≥10 — but only
**3.6%** exceed 50. The text corpora never need more than 4; SQuAD needs exactly 1 on every query.
A 50-partition budget is 12× the largest requirement any text query has.

## 4. The canonical ranking gap

ALL coverage under a top-P canonical cut is governed by one variable: the **worst** required-gold
rank. Ranks 0-indexed over the full universe.

| block | npart | best med | **worst med** | worst p75 | worst p90 | worst p95 | worst p99 |
|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 401 | 1 | **2** | 6 | 12 | 17 | 29 |
| MetaQA hop2 | 401 | 2 | **14** | 57 | 145 | 309 | 396 |
| MetaQA hop3 | 401 | 2 | **218** | 360 | 393 | 397 | 400 |
| WebQSP | 7814 | — | **9** | 46 | 202 | 1373 | 7500 |
| 2Wiki | 658 | — | **6** | 11 | 24 | 68 | 361 |
| MuSiQue | 136 | — | **3** | 11 | 28 | 47 | 99 |
| HotpotQA | 5074 | — | **7** | 16 | 35 | 75 | 2811 |
| SQuAD | 190 | — | **3** | 7 | 16 | 27 | 79 |

**This is the finding of the audit.** On MetaQA hop3 the canonical Dense+SPLADE ranking places the
*best* required partition at median rank **2** and the *worst* at median rank **218** — out of 401.
It finds the entry point immediately and scatters the rest of the chain across the entire universe.
At p90 the worst required partition sits at rank 393, i.e. in the last 2% of the whole corpus.

Coverage from simply taking the canonical top-P, no router:

| block | P=25 | P=50 | P=75 | P=100 | P=150 | oracle P=50 | gap |
|---|---|---|---|---|---|---|---|
| MetaQA hop1 | 0.9835 | 0.9955 | 0.9985 | 0.9985 | 0.9985 | 1.0000 | +0.0045 |
| MetaQA hop2 | 0.6021 | 0.7237 | 0.8078 | 0.8574 | 0.9009 | 0.9970 | **+0.2733** |
| MetaQA hop3 | 0.2042 | 0.2568 | 0.2973 | 0.3243 | 0.3994 | 0.9640 | **+0.7072** |
| WebQSP | 0.6603 | 0.7618 | 0.8069 | 0.8407 | 0.8795 | 0.9951 | **+0.2333** |
| 2Wiki | 0.9005 | 0.9375 | 0.9535 | 0.9590 | 0.9670 | 1.0000 | +0.0625 |
| MuSiQue | 0.8855 | 0.9565 | 0.9795 | 0.9905 | 1.0000 | 1.0000 | +0.0435 |
| HotpotQA | 0.8525 | 0.9345 | 0.9495 | 0.9595 | 0.9680 | 1.0000 | +0.0655 |
| SQuAD | 0.9435 | 0.9805 | 0.9890 | 0.9940 | 0.9990 | 1.0000 | +0.0195 |

Tripling the canonical budget from 50 to 150 buys MetaQA hop3 **+0.143**; an oracle choosing 50 partitions freely would buy **+0.707**. Budget is the wrong axis.

## 5. Mandatory MetaQA ceiling table

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

The bottleneck is visually obvious, and one row makes it unarguable: the **full-universe oracle at
P=25 — half the current budget — scores 0.8498 on hop3 against the router's 0.2583 at P=50.**

### The constraint ladder — which frozen constraint actually binds

Each step keeps the partition abstraction and switches one constraint off (T6):

| ladder step | MetaQA hop2 | MetaQA hop3 | MetaQA agg | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD |
|---|---|---|---|---|---|---|---|---|
| actual router (SAFE F6) | 0.7297 | 0.2583 | 0.6612 | 0.7646 | 0.9435 | 0.9635 | 0.9505 | 0.9875 |
| + perfect selector, current pool, B=6 | 0.8033 | 0.3634 | 0.7212 | 0.7886 | 0.9515 | 0.9750 | 0.9645 | 0.9945 |
| + perfect selector, current pool, **B unlimited** | 0.8033 | 0.3664 | 0.7222 | 0.7886 | 0.9515 | 0.9750 | 0.9645 | 0.9945 |
| + perfect selector, **unlimited pool**, B=6 | 0.9505 | 0.6802 | 0.8769 | 0.9542 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| + abstraction only (gold fits in 50) | 0.9970 | 0.9640 | 0.9870 | 0.9951 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Three things fall out:

1. **B is not a constraint at all.** Removing the swap cap entirely, with the current pool, moves
   MetaQA hop3 by **+0.0030** and every other corpus by **0.0000**. This confirms the previous
   phase's finding on a much stronger test.
2. **Pool reach is the dominant constraint.** Removing it (unlimited pool, B still 6) moves hop3
   from 0.3664 to **0.6802** and takes all four text corpora to **exactly 1.0000**.
3. **The selector's entire share is the first step**: +0.0736 hop2, +0.1051 hop3, +0.0240 WebQSP,
   +0.0070 to +0.0140 on text. That is the whole prize for perfect boundary arithmetic.

## 6. WebQSP control — this is not MetaQA-specific

| budget | ALL | ANY |
|---|---|---|
| P=25 | 0.9803 | 1.0000 |
| P=50 | 0.9951 | 1.0000 |
| P=75 | 1.0000 | 1.0000 |
| P=100 | 1.0000 | 1.0000 |
| P=150 | 1.0000 | 1.0000 |

Gold-partition count: mean 3.66, median 1, p90 7, p99 37, max 69; 52.1% need 1, 16.4% need ≥5, 0.5%
need >50. npart = 7814.

WebQSP shows the **same shape**: abstraction ceiling 0.9951, router 0.7646, and the loss decomposes
the same way — pool reach −0.2065, B given the current pool −0.0000 (B in isolation −0.0409),
selector −0.0240. So this is a **general KB /
multi-evidence partition-scoping issue**, not a MetaQA artifact. What is MetaQA-specific is only the
*severity*: hop3 demands a median of 6 partitions whose canonical ranks span 2 to 218.

## 7. Text controls

| corpus | npart | BASE P50 | SAFE router | oracle P50 | oracle P100 | ranking gap |
|---|---|---|---|---|---|---|
| 2Wiki | 658 | 0.9375 | 0.9435 | 1.0000 | 1.0000 | +0.0565 |
| MuSiQue | 136 | 0.9565 | 0.9635 | 1.0000 | 1.0000 | +0.0365 |
| HotpotQA | 5074 | 0.9345 | 0.9505 | 1.0000 | 1.0000 | +0.0495 |
| SQuAD | 190 | 0.9805 | 0.9875 | 1.0000 | 1.0000 | +0.0125 |

Text is **100% coverable at P=50 by construction** (max 4 gold partitions), and even at P=10. The
remaining 1–6 points are pure ranking, and the ladder shows all of it is pool reach — with an
unlimited pool and B=6 every text corpus hits 1.0000. Text is near-saturated; there is no
architectural question here.

## 8. MetaQA hop3 decomposition

All 494 hop3 queries the SAFE router fails, one category each. Strict precedence:
**A** (uncoverable at any P) → **B** (>50 gold partitions) → **D** (a required partition is outside
boundary ∪ challengers) → **C** (in pool, but >B=6 of them outside the protected core) → **E**
(in pool and fits B=6 — the selector chose wrong).

| category | count | % |
|---|---|---|
| **A_REPRESENTATION_LIMIT** | 0 | **0.00%** |
| **B_P50_CAPACITY_LIMIT** | 24 | 4.86% |
| **D_CURRENT_POOL_LIMIT** | 398 | **80.57%** |
| **C_RANKING_LIMIT** (protected-core placement) | 2 | 0.40% |
| **E_SELECTOR_LIMIT** | 70 | **14.17%** |

**Overlap disclosed:** of the 398 category-D queries, **187 (47.0%)** would *also* fail the B=6 cap
even with an unlimited candidate pool. The categories are mutually exclusive by precedence, but the
underlying constraints are not, and widening the pool alone would fix at most 53% of D directly.
That is consistent with the ladder: unlimited pool at B=6 reaches 0.6802, not 0.9640.

## 9. Decision

- **RANKING_LIMITED** — full-universe P50 (hop2 0.9970, hop3 0.9640) is dramatically above the
  current router (0.7297, 0.2583). Confirmed.
- **CAPACITY_LIMITED** — refuted. P=50 already achieves 0.9640 on hop3; P=150 adds 0.0360 and adds
  nothing on any other corpus.
- **REPRESENTATION_LIMITED** — refuted. Large-P oracle reaches exactly 1.0000 everywhere, `ANY` is
  1.0000, and 0 queries have unmapped gold.

Not MIXED in any material sense, but the exact percentages of the hop3 failure mass are:
representation **0.00%**, budget capacity **4.86%**, candidate reach **80.57%**, protected-core
placement **0.40%**, selector **14.17%**.

## 10. R1 ruling support

| level | SAFE F6 | R1 | R1 gain | oracle P50 | **R1 / P50-oracle headroom** | current-pool B6 oracle | R1 / selector headroom |
|---|---|---|---|---|---|---|---|
| MetaQA hop2 | 0.7297 | 0.7477 | +0.0180 | 0.9970 | **6.7%** | 0.8033 | 24.5% |
| MetaQA hop3 | 0.2583 | 0.2718 | +0.0135 | 0.9640 | **1.9%** | 0.3634 | 12.8% |
| aggregate | 0.6612 | 0.6717 | +0.0105 | 0.9870 | **3.2%** | 0.7212 | 17.5% |

By the directive's own example interpretation — oracle hop3 is 0.9640, far above the 0.70 threshold
given for "R1 is only scratching a huge ranking problem" — **R1 is scratching a huge ranking
problem.** But the second reading matters just as much: measured against what a *selector* can
possibly achieve, R1 has already captured **12.8–24.5%**. Those two facts are not in tension. They
say R1 is a good selector and the selector is a small lever. Even a **perfect** selector over the
current pool adds only +0.0916 more on hop3 beyond R1.

The R1 promotion decision remains yours. This audit contributes one input: whichever way it goes,
the decision is worth at most ~0.10 of MetaQA hop3, and the 0.71 sitting above it is untouched by
any selector.

---

## Answer

```
RANKING_LIMITED
IS_FURTHER_L1_SELECTOR_OPTIMIZATION_JUSTIFIED = NO
```

**Justification, and the one distinction that matters.** The directive's decision rule maps
RANKING_LIMITED to "selector/routing research is still justified". The decomposition it also
required splits that in two, and the two halves point opposite ways:

- **Selector research — NO.** The boundary arithmetic optimised across the last two phases governs
  14.17% of hop3 failures and 0.40% via protected-core placement. Its total remaining ceiling is
  +0.1051 hop3, +0.0736 hop2, +0.0240 WebQSP, +0.0070–0.0140 on text. R1 has already taken 12.8–24.5%
  of it. Continuing to search scoring rules optimises a lever worth at most a seventh of the gap
  while the other six-sevenths sits untouched.
- **Routing research — YES, redirected to candidate generation.** 80.57% of hop3 failures are
  queries whose required partitions never enter `boundary ∪ challengers`. The abstraction supports
  covering them (96.4% fit in 50 partitions; oracle at P=25 already scores 0.8498), and the ladder
  shows an unlimited pool at the *current* B=6 reaches 0.6802 on hop3 and 1.0000 on all four text
  corpora. That is a +0.42 hop3 lever versus the selector's +0.10.

Two constraints on any such follow-up, both established here rather than assumed: **B must not be
enlarged** (unlimited B moves hop3 by +0.0030 and everything else by 0.0000), and the failure is
**not** fixable by pool widening alone — 47% of the pool-blind queries also exceed the B=6 cap, so
reach and protected-core composition have to move together.

Stopping here as instructed.
