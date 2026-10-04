
---

## Mechanism isolation — is the small positive signal structural at all?

`A1b_M0` is the clean control the sweep happens to contain: node-level RRF with **M=0, i.e. zero structural
expansion, zero graph edges traversed**. Comparing it against `A1b` at its best M isolates what the *graph*
contributes on top of merely relocating RRF from the partition level to the node level.

| corpus | BASE ALL@P50 | A1b M=0 — node-RRF, **no structure** | A1b best M — **with** structure | Δ from structure |
|---|--:|--:|--:|--:|
| MetaQA | 0.6587 | 0.6662 (**+15**, p=.058) | M=64: 0.6667 (+16) | **+0.0005** (1 query) |
| 2wiki | 0.9375 | 0.9445 (**+14**, p=.013) | M=64: 0.9425 (+10) | **−0.0020** |
| MuSiQue | 0.9565 | 0.9505 (−12, p=.073) | M=32: 0.9485 (−16) | **−0.0020** |
| SQuAD | 0.9850 | 0.9810 (−4) | M=32: 0.9780 (−7) | **−0.0030** |

**Inside this family, structural propagation adds nothing on top of the RRF relocation on any corpus** —
one query on MetaQA, and strictly negative on the other three. Every positive number in the A1b column is
produced with the graph switched off.

This does *not* say MetaQA's gain is graph-free, and the distinction matters. In the parity-gated A2 family
the M=0 row **is** BASE by construction, so A2dOnly's +15 there is unambiguously produced by the graph. What
the table shows is that the two routes are **not additive**: node-level RRF alone reaches +15, graph
expansion alone reaches +15, and doing both reaches +16. Whichever mechanism runs first absorbs the gain.
The honest reading is that MetaQA has roughly one query in 130 whose gold-bearing partition sits just below
the P50 cut, and either perturbation is enough to push it in — not that the graph is contributing
independent multi-hop evidence at the partition level.

Within the parity-gated family (which is the one that is architecturally admissible, since A1b changes the
object at M=0):

| corpus | BASE | best parity-gated config |
|---|--:|--:|
| MetaQA | 0.6587 | **A2_M32 0.6672 (+17, p=.050)** — and A2dOnly_M32 +15 at p=.024 |
| 2wiki | 0.9375 | A2dOnly_M32 0.9360 (−3, ns) |
| MuSiQue | 0.9565 | A2sOnly_M32 0.9580 (+3, ns) |
| SQuAD | 0.9850 | A2dOnly_M32 0.9840 (−1, ns) |

So the entire positive result of the corrected architecture is **one corpus, +15 to +17 queries out of
1998 (+0.8 pt ALL@P50)**, bought with 21× the BASE compute at the top of the M sweep.

---

## The four DECISION questions, answered

**1. Does parameter-free structural propagation improve partition-level co-scoping?**

**Essentially no.** One corpus out of four shows a significant gain (MetaQA, A2dOnly M=32: +15 queries,
p=0.024, ALL@P50 0.6587 → 0.6662). 2wiki, MuSiQue and SQuAD are all ≤ BASE under every parity-gated
configuration. MetaQA's gain is real and parity-gated, but the mechanism-isolation control above shows it is
**not separable from a graph-free perturbation of the same size**: node-level RRF with the graph switched
off reaches the same +15, and combining the two reaches +16.

The audit explains why, and the explanation is structural rather than empirical: **`mem_idx` already is a
1-hop structural propagation.** The canonical router votes every retrieved node into its own partition *and
its directed 1-hop neighbours' partitions*. Track-A's hop-1 evidence is therefore largely re-counting what
the frozen router already counted. The `in_BASE_partition` diagnostic makes this quantitative — 25–46 % of
expanded nodes on MetaQA/2wiki and **64–78 % on MuSiQue and SQuAD** land in partitions BASE had already
selected.

**2. Is RRF better BEFORE or AFTER structural propagation?**

**AFTER — decisively, on all four corpora, at every M.** A1a (structure as a co-equal third RRF channel) is
significantly *harmful* on 2wiki (−31, p<0.001) and MuSiQue (−22, p=0.002), and degrades monotonically in M
on MetaQA (−14 → −19 → −35). The mechanism is legible in the churn columns: A1a **evicts far more
gold-bearing partitions than it admits** (2wiki M=32: 47 evicted vs 7 admitted; downstream, 79 golds
evicted vs 14 added). Granting structural evidence a full-weight vote displaces the retrieval evidence that
was carrying the golds.

A2's retrieval-subordinate damping — structural nodes appended *after* the K retrieval nodes, so they enter
at weight `1/(K0+K+j)` — is the parameter-free way to express "structure is corroboration, not evidence of
the same kind", and it is what makes the difference. This answer is robust and is the clearest result in the
whole exercise.

**3. Does improved P50 co-scoping survive through the frozen L2?**

**No — it inverts.** On 2wiki every configuration is significantly worse than BASE downstream, including
`A1b M=0`, the *only* config that significantly improved L1 (+14 ALL@P50, p=0.013; +41 golds into scope
against 6 lost). It is the worst downstream row: nDCG@5 −0.0094, R@5 −0.0109, ALL@50 −0.0167, all
significant. The stage decomposition locates the loss precisely: **54–78 % of it falls on queries whose gold
coverage in scope did not change at all**, i.e. it is the frozen reranker reacting to a perturbed pool, not
to lost golds. L1 buys coverage in units of a handful of queries and pays in units of thousands of
perturbed pools.

**One honest limitation:** MetaQA — the only corpus where L1 improved — has no frozen L2 corpus, so this
question could only be answered on corpora where the L1 effect was neutral-to-negative. What has been shown
is that a *significant L1 co-scoping gain does not survive* (the A1b/2wiki case). What has *not* been shown
is that MetaQA's specific gain would fail; that test is not runnable without a new build.

**4. After the corrected L1, is a new B1 admission stage necessary?**

**No — and the evidence argues it would be counterproductive.** The premise of a B1-style admission stage is
that structural candidates are worth having in the pool if only one could decide *which* ones. The
downstream diagnostic tests exactly that premise in its most favourable form — structural candidates enter
as ordinary partition members, fully scored by all five experts, with real golds among them — and the answer
is that adding them is net-harmful **even when they include golds**, because the dilution cost on the fixed
top-50 pool exceeds the coverage benefit. That is the same wall B1.1–B1.9 hit from the other side, and it is
now visible without any admission policy at all.

Independently, Q2's three blind gates already showed that structurally-recovered multi-hop golds are
un-poolable by every expert (median best-expert rank ~440–610; exact Relation abstains on 100 %; forced into
a C11a window, promoted to top-5 in **0 / 1025**). Corrected Track-A does not change those documents'
expert scores, so it does not change that verdict.

---

## Recorded flags

```
TRACK_A_CANONICAL_POSITION        = PRE_PARTITION_VOTING          (correction established by this audit)
POST_P50_STRUCTURAL_APPEND        = DIAGNOSTIC_ONLY
M0_PARTITION_PARITY               = EXACT     (A1a, A2, A2dOnly, A2sOnly; all dev corpora)
DOWNSTREAM_M0_CANDIDATE_PARITY    = EXACT     (2wiki, musique; 0 evicted / 0 added)
RRF_POSITION                      = AFTER_STRUCTURAL_PROPAGATION  (A2 > A1, all corpora, all M)
STRUCTURAL_PROPAGATION_HELPS_L1   = MARGINAL  (1 of 4 corpora, +0.8pt; non-additive with a graph-free
                                               control that reaches the same +15 on its own)
L1_GAIN_SURVIVES_FROZEN_L2        = NO        (inverts; 54-78% of loss on unchanged-coverage queries)
B1_ADMISSION_STAGE_REQUIRED       = NO
L1_PARAMETER_FREE                 = YES       (0 learned parameters; verified by grep, below)
SQUAD_L2_CORPUS_STALE             = YES       (pre-existing defect found by this audit; not repaired)
SAFE_TO_RUN_LOCKED_TEST           = NO
```

### `L1_PARAMETER_FREE` — how it was verified, not asserted

The whole corrected L1 path is `scratchpad/_ta_prepartition.py` + `scratchpad/_ta_run.py`. Searching both
for every way a parameter could enter:

```
grep -nE "torch\.load|joblib\.load|\.pt\b|\.pkl|Head\(|nn\.|xgb|Booster|sklearn|fit\(|train"  ->  NONE
grep -nE "gold" _ta_prepartition.py                                                           ->  NONE
```

There is no checkpoint load, no module, no fit, no gold in the primitives. In `_ta_run.py` gold appears in
exactly two places: selecting the evaluation universe (`_ta_run.py:51-56`, queries with ≥1 in-corpus gold —
applied identically to BASE and every variant, **before** any scope is built) and scoring the resulting
partitions (`_ta_run.py:174-183`). **No gold enters scope construction.** No dataset identity is read; the
compute contract (`K0=60, K=100, P=50, SEED_K=5, M_MAX, MAX_HOPS=3, DEG_CAP=300, MAX_EDGES_SCORED=400k,
MAX_FRONTIER=1500, PPR_CAP=4000`) is one global set of integers with no per-corpus branch. Every constant is
declared in TASK 3's parameter inventory and **none was tuned against target labels** — M is swept and
reported in full precisely so that no single value is silently selected.

## Directive stops honoured

Not started, not run, not touched:

- no new learned admission model, no reserve `R`, no special graph candidate slot
- no attention, no MLP, no GNN, no learned router / fusion weight / λ anywhere in L1
- no B2 redesign
- no capacity enlargement
- no encoder pass (added candidates scored deterministically from frozen embeddings and frozen heads)
- **TEST never inspected** on any corpus; every number above is dev VAL
- no B1.1–B1.9 artefact deleted or rewritten — reclassified in place, below
- the SQuAD stale-corpus defect is **reported, not repaired** (repair = rebuilding the corpus, out of scope)
