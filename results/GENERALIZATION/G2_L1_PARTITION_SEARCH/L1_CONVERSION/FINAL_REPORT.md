# L1 CONVERSION PHASE — open the selector, minimal change only

**Verdict: C. `SELECTOR_STILL_BLOCKED`.**

| flag | value |
|---|---|
| `U_PC5_PROMOTED` | **NO** |
| `SELECTOR_PROMOTED` | **NO** |
| `L1_FROZEN` | **NO** |
| `CONVERSION_GATE_PASSED` (STEP 4) | YES |
| `ONLINE_GRAPH_EDGES_TOUCHED` | 0 |

The algebraic blocker identified last phase is **gone** — truly-new proposals are now selectable,
and gold-bearing ones are actually admitted. But coverage does not follow. The best rule converts
**2.0 %** of the MetaQA hop3 headroom and is **significantly worse than SAFE on three of six
corpora**; the only rule with no significant regression anywhere is statistically indistinguishable
from SAFE everywhere. Nothing is promoted. SAFE remains `R0 / B6_S4_F6_Ms64_Mr32`.

Tables: [TABLES.md](TABLES.md) (T1–T10). Raw: `diag/{gate,main,why,fin}.json`.
Code: `scratchpad/_l1cv_{core,gate,main,why,fin,tables}.py`.

---

## What changed, and what did not

Frozen and untouched: the swap contract `final = base_rank[:44] ∪ X, |X| = B = 6`; P = 50; K0 = 60;
the U_PC5 proposer (five sources, round-robin, dedup, M_TOTAL = 256, zero learned parameters, no
dataset identity, one global budget). Changed: **only the boundary score**, and only to collapse the
proposal mechanism into one vote.

| signal | meaning | list length |
|---|---|---|
| `C` | canonical Dense+SPLADE partition rank (`cpos`) | 200 |
| `S` | S4 structural rank (`spos`) | M_struct = 64 |
| `U` | U_PC5 proposal rank (`u_rank`) | ≤ 306 (T1) |

`ret_rrf` is removed as a co-equal scoring signal — U already carries the node-level retrieval
continuation as two of its five sources — but is retained as candidate **membership**, so ACTUAL and
the U_PC5 pool oracle are measured over the same candidate set.

### STEP 1 — one symmetric ranking, and a budget correction that had to be made

u_rank is built over the five sources **before** the base50 exclusion, so incumbents carry proposal
evidence too. Determinism verified by two independent builds (T1).

Building it that way collides with U_PC5's frozen budget semantics, which charge M_TOTAL to *real*
proposals (each family is filtered to non-incumbents before the top-M cut). Cutting the symmetric
merge at 256 spends 50 slots on incumbents, and that is not a bookkeeping detail — measured on
MetaQA it shrinks the proposal set 256 → 172.6/query and drops the hop3 pool oracle
**0.6351 → 0.5871**, i.e. it would silently *alter U_PC5*, which the directive forbids.

Resolution: the merge is built to `M_TOTAL + P` so u_rank is defined across the whole competition,
and the 256 budget is applied to the non-incumbent entries only. At most P of the first M_TOTAL + P
entries can be incumbents, so ≥ 256 real proposals are always available. This reproduces the frozen
proposer's power: MetaQA hop2 pool oracle **0.9474** (frozen: 0.9474, exact), ALL **0.8609** (0.8604),
hop3 **0.6366** (0.6351); WebQSP **0.9027** (0.9027, exact). U_PC5 is preserved, not weakened.

### STEP 2 — four rules, no grid

`S0_SAFE` = C + S + ret_rrf under R0/F6, unchanged (reference).
`S1_CSU_RAW` = `rrf60(C) + rrf60(S) + rrf60(U)`, absent → 0 (primary hypothesis).
`S2_CSU_LIFT` = the same three channels with the finite-list lift `1/(60+r) − 1/(60+L_x)`, present
only, absent → 0.
`S3_CU_ABLATION` = C + U raw (mechanism only, not promotable).
No weights, no thresholds, no tuning, no search over a family.

---

## STEP 4 — the gate: PASSED

The closure theorem is broken. Under SAFE, 2,129,285 truly-new candidates over six corpora produced
**0** selections; the new rules select them (MetaQA: 18 / 1,423 / 2,380 for S1 / S2 / S3), and S2 and
S3 admit gold-bearing ones (38 and 66). Both S1 and S2 clear `NEW_PROPOSALS_SELECTED > 0`, so the
phase proceeded.

## STEPS 5–6 — MetaQA primary (T2–T5)

Bookends, hop3: SAFE **0.2583** → current-pool oracle 0.3634 → **U_PC5 pool oracle 0.6366** →
unlimited-pool oracle 0.6802 → full-universe P50 oracle 0.9640.

| rule | hop1 | hop2 | hop3 | ALL | new selected | gold admitted | churn |
|---|---|---|---|---|---|---|---|
| S0_SAFE | 0.9955 | 0.7297 | 0.2583 | 0.6612 | 0 | 0 | 4.66 |
| S1_CSU_RAW | 0.9955 | **0.7402** | 0.2658 | **0.6672** | 18 | **0** | 5.13 |
| S2_CSU_LIFT | 0.9955 | 0.7267 | 0.2568 | 0.6597 | 1,423 | 38 | 4.46 |
| S3_CU_ABLATION | 0.9955 | **0.7117** | 0.2598 | 0.6557 | 2,380 | 66 | 4.41 |

Bold = significant against SAFE (McNemar). `CONVERSION_EFFICIENCY` for S1: hop2 **+4.83 %**,
hop3 **+1.98 %**, ALL **+3.01 %**, WebQSP **−1.02 %**. S2 on hop3: **−0.40 %**.

Two observations decide the phase.

**The one positive result is not a conversion result.** S1 gains +0.0060 ALL / +0.0075 hop3 while
admitting **zero** gold-bearing new proposals (T4). Its 15 newly-covered queries come entirely from
re-ordering candidates SAFE already had — that is, from dropping `ret_rrf`, not from the new
evidence. It converts none of the +0.3783 hop3 headroom.

**The rules that do convert gold lose more than they gain.** S2 admits 38 gold proposals and lands
at −0.0015 ALL; S3 admits 66 and lands at −0.0055. Coverage is conjunctive — a query counts only
when its *whole* missing set is swapped in — so partition-level admissions do not add up.

## STEP 7 — all six corpora (T6, T7)

| rule | metaqa | webqsp | 2wiki | musique | hotpot | squad | pooled | macro |
|---|---|---|---|---|---|---|---|---|
| S1_CSU_RAW | **+0.0060** | −0.0014 | −0.0035 | **−0.0100** | **−0.0060** | **−0.0055** | **−0.00350** | −0.00340 |
| S2_CSU_LIFT | −0.0015 | −0.0042 | +0.0010 | −0.0035 | +0.0035 | +0.0010 | −0.00044 | −0.00062 |
| S3_CU_ABLATION | −0.0055 | **−0.0127** | +0.0030 | −0.0025 | +0.0030 | +0.0020 | −0.00158 | −0.00212 |

Bold = significant. Against the directive's promotion gate — *no significant regression against SAFE
is allowed*:

- **S1_CSU_RAW is disqualified.** Significant regressions on musique (p = 5.4e-4), hotpot
  (p = 0.012) and squad (p = 0.0034); improves on 1/6 corpora; **pooled significantly worse than
  SAFE** (−0.00350, +36/−76, p = 2e-4). The MetaQA win is a single-corpus effect.
- **S2_CSU_LIFT is not promotable either.** It has no significant regression anywhere — and no
  significant gain anywhere. Pooled −0.00044, p = 0.694, 3/6 corpora up. It is noise.
- **S3_CU_ABLATION does not dominate** (significantly worse on webqsp), so it stays unpromoted as
  specified. Its value is diagnostic: dropping S is *worse* than keeping it, so the S4 structural
  channel is **not** subsumed by U even though U contains a graph-neighbour source.

## STEPS 8, 9 — not reached

STEP 8 (nested discovery/validation + LODO) gates on a rule appearing promotable. None does: S1 fails
the regression gate outright, S2 has no effect to validate. A LODO would only re-measure noise.

STEP 9 (B = 6/8/12) is explicitly conditioned on a winning selector having been identified. There is
none, so it was **not run**. B remains overturned-but-untested against an open pool; that finding
from the previous phase stands unchanged.

## STEP 10 — cost (T10)

`ONLINE_GRAPH_EDGES_TOUCHED = 0`, verified: `G_GRAPH_NBR_RAW` reads precomputed depth-1 neighbour
rows and merges them arithmetically. No encoder work at query time.

u_rank lookup + merge is 0.011–0.026 ms/query; the selector goes from 0.011–0.051 ms (SAFE) to
0.069–0.706 ms (S1) because it scores 92–290 candidates instead of ~50. Total added L1 routing
latency **0.067–0.707 ms/query**. Cached bytes: 7.4 MB of U lists plus 2.7 MB of partition graphs
across all six corpora. The cost is negligible — it simply buys nothing.

---

## Why conversion fails (T8, T9)

The blocker has moved, and the new one is not a scoring-arithmetic problem.

|  | MetaQA hop3 | WebQSP ALL |
|---|---|---|
| feasible under SAFE pool → covered | 0.3634 → 0.2583 | 0.7886 → 0.7646 |
| feasible under U_PC5 pool → covered (S1) | **0.6366 → 0.2658** | **0.9027 → 0.7632** |
| feasible-but-uncovered queries | 70 → 247 | 34 → 198 |
| mean size of the missing set on those queries | 1.61 → **2.40** | 1.15 → **1.97** |
| fraction of the missing set actually selected | 0.085 → 0.089 | 0.044 → 0.063 |

The queries U_PC5 makes newly feasible are structurally harder: they need **~2.4 partitions swapped
in simultaneously** out of 6 slots, and the selector picks under 9 % of what each one needs. Coverage
requires all of them at once, so a partition-level hit rate at this level cannot convert regardless
of how many candidates are offered.

T9 shows why the hit rate is that low. The needed-but-unselected partitions sit at **median rank 53
of ~268 scored candidates** on MetaQA hop3 (81 of 787 inside the top 12) and **median rank 44** on
WebQSP (45 of 351 in the top 12). They are not near-misses. Contrast the previous phase, where the
blocker was a margin of exactly `1/109 − 1/110 = 8.34e-05` — a tie-break artefact that a rule change
could and did remove. Here the gold-bearing partitions are simply **ranked in the middle of the
field** by every combination of C, S and U.

That is the finding: **C, S and U carry no discriminative signal about which boundary partition is
gold-bearing.** U_PC5 solved discovery — it puts the right partitions in the room (0.2583 → 0.6366
feasible on hop3) — but none of the three available rank signals can pick them out once they are
there, and adding U as a third vote mostly perturbs an ordering that was already better tuned.

Per the directive: selector micro-tuning stops here, and no further rule family is invented.

---

## Standing state

- SAFE reference unchanged: `R0 / B6_S4_F6_Ms64_Mr32`. Parity verified exactly this phase
  (S0 reproduces ALL 0.6612 / hop3 0.2583 on MetaQA and every corpus scoreboard).
- `U_PC5_PROMOTED = NO` — it remains a validated *discovery* result with no way to spend it. It is
  not refuted; nothing that can consume it exists.
- `SELECTOR_PROMOTED = NO`. `L1_FROZEN = NO`.
- Untouched, as instructed: L2, L3, TEST, training, PPR, candidate-generation search, new proposal
  families, new structural signals, learned L1 components, channel weights, threshold search.
