# L2 — C10: Set-Aware Multi-Evidence Top-5 Selection

**Goal.** C8c/C9 score candidates essentially independently and sort. For multi-gold retrieval the marginal value
of a candidate should depend on **what evidence is already selected** — a redundant look-alike is worth less than a
different required item. C10 tests whether **dynamic set-aware complementarity** can push top-5 past C8c, using only
existing canonical Qwen node embeddings and the existing 5-dim expert-contribution vectors. No re-encoding, no new
experts, no pool enlargement, no graph traversal, no TEST, no L1/L3. Designed/selected on **C9_DEV**, confirmed once
on **DEV_INNER**.

> **Scale note.** C9_DEV / DEV_INNER are TRAIN-derived nested-development splits (≈0.96) and are **not comparable**
> to the official-VAL C8c milestone (0.8824). Every number below measures a C10 method **vs C8c on the same split**.

## Headline — the hypothesis is TRUE but NOT CONVERTIBLE; standing policy stays C8c

The set-redundancy signal is real and strong as a **diagnostic**, yet no set-aware selector beats C8c: the best
deterministic method ties it (not significant) and every learned greedy selector is *worse*.

## 1. Set-redundancy audit (Task 1 — GATE: **SET_AWARE_SIGNAL_PRESENT = YES**)

For each query, under C8c's own ranking, we compared the **missed golds** (gold in top-50, C8c rank ≥ 5) against the
**top-5 false positives** (non-golds occupying top-5), by their similarity to the already-selected top-5:

| C9_DEV | missed golds | top-5 FPs |
|---|:---:|:---:|
| max **Qwen** sim to selected top-5 (mean) | **0.477** | 0.569 |
| max **expert-support** cos to selected (mean) | **0.895** | 0.951 |
| argmax-expert overlap with selected (mean) | **0.191** | 0.369 |

- **Cohen's d** (missed − FP): Qwen max-sim **−0.52**, support max-cos **−0.67** (both medium-large; replicated on
  DEV_INNER: −0.51 / −0.45).
- **Query-conditioned paired test** (queries with both a missed gold and an FP): mean paired diff **−0.194**,
  95% CI [−0.227, −0.161], **significant**; the missed gold is more novel than the FP in **~80%** of such queries.

So the FPs currently in top-5 are **redundant** with already-selected evidence (they duplicate the selected set's
dominant expert nearly 2× as often), while the missed golds carry **different** evidence. The hypothesis holds.

## 2. Multi-gold breakdown (Task 2) & set oracles (Task 5)

All headroom is in the multi-gold tail; single-gold is already perfect:

| C8c on C9_DEV | 1 gold (n=79) | 2 gold (n=2448) | 3+ gold (n=776) |
|---|:---:|:---:|:---:|
| GOLD_RECALL@5 | 1.000 | 0.987 | **0.940** |
| ALL@5 | 1.000 | 0.973 | **0.799** |

**Perfect-set oracle** over the C8c window: top-20 recall5 **0.9956** ≈ top-50 **0.9969** → top-20 already contains
essentially all recoverable golds, so C10 operates over **top-20** (Task 9). The recoverable top-5 gap is ~0.02,
entirely in the 3+-gold tail.

## 3. C10a — deterministic MMR / support-diversity (Tasks 3, 4): **ties C8c, not significant**

Greedy relevance − λ·max-similarity over the C8c top-20 (λ=0 reproduces C8c exactly — verified). Tiny predeclared
λ∈{0, 0.05, 0.10}; select on C9_DEV only.

| C9_DEV | C8c | C10a λ=0.05 | C10a λ=0.10 | C10a2 (sem .05 + expert .05) |
|---|:---:|:---:|:---:|:---:|
| NDCG@5 | 0.9658 | 0.9660 | **0.9662** | 0.9662 |
| GOLD_RECALL@5 | 0.9762 | 0.9761 | 0.9762 | 0.9764 |
| ALL@50 | 0.9897 | 0.9897 | 0.9897 | 0.9897 |

Best (λ=0.10) is **+0.0004 NDCG@5**, paired bootstrap CI95 [−0.0002, +0.0010] — **not significant**; net top-5 gold
rescue **−2**. It genuinely diversifies (mean pairwise Qwen sim 0.403→0.399) and leaves single-gold **identical**,
but the effect is negligible. On DEV_INNER it is marginally larger (+0.0016 NDCG@5, +0.0019 recall5) but was not the
selection split. **MMR_HELPS = NO.**

## 4. C10b / C10b2 — learned greedy set selector (Tasks 6–8, 11, 12): **worse than C8c**

A marginal-utility model P(candidate is a needed *unselected* gold | static C9 features + **dynamic set features**:
Qwen/support similarity to selected, same-argmax overlap, novelty rank, step/remaining), greedy 5-step over top-20,
trained on TRAIN with a **teacher + 3-fold OOF-rollout** prefix mixture (Task 7 anti-leakage).

| C9_DEV | C8c | C10b (pointwise) | C10b2 (NDCG ranker + C8c-seeded pos0) |
|---|:---:|:---:|:---:|
| NDCG@5 | **0.9658** | 0.9074 | 0.9532 |
| GOLD_RECALL@5 | **0.9762** | 0.9020 | 0.9538 |
| ALL@50 | 0.9897 | 0.9897 | 0.9897 |

- **C10b (pointwise)** collapses: a logloss P(gold) model is a far weaker top-5 ranker than C8c's listwise
  LambdaMART, so it forfeits relevance (net rescue −618).
- **C10b2** fixes that (per-step `rank:ndcg` ranker; rank-1 seeded with C8c's own top pick; feature importance now
  healthy — `c8c_oof_score` 0.28 dominant, dynamic features small), yet is **still significantly worse**: NDCG@5
  **−0.0126** (CI [−0.015, −0.010]), recall5 **−0.0224** (CI [−0.026, −0.019]). The damage lands in the very tail it
  targeted: **3+-gold recall 0.9404 → 0.8947, ALL@5 0.799 → 0.662**. Net rescue **−212** (45 in, 257 out).
  DEV_INNER confirms (0.9637 → 0.9483). **LEARNED_SET_SELECTOR_HELPS = NO.**

## 5. Why the signal doesn't convert

Novelty separates missed-golds from FPs **on average** (cohend −0.5), but the **novel population is dominated by
non-gold distractors** — there are ~200 missed golds against thousands of equally-novel non-golds. At the
per-candidate decision boundary the signal is therefore **diagnostic, not discriminative**: acting on it (MMR *or*
greedy) swaps in more noise than evidence. Greedy set-selection additionally **forfeits the joint listwise
optimization C8c already performs**, which is why even a relevance-preserving learned ranker loses. Deep recall is
untouched throughout (ALL@50 identical — reorder within top-50 only).

## Final gates

| Gate | Answer |
|------|--------|
| SET_AWARE_SIGNAL_PRESENT | **YES** (Qwen cohend −0.52, support −0.67; paired diff significant both splits) |
| MMR_HELPS | **NO** (+0.0004 NDCG@5, not significant; net rescue −2) |
| LEARNED_SET_SELECTOR_HELPS | **NO** (C10b collapses; C10b2 significantly −0.0126) |
| MULTIGOLD_RECALL_IMPROVES | **NO** (MMR flat; C10b2 3+-gold 0.940→0.895) |
| SINGLE_GOLD_PRESERVED | **YES** (MMR identical; C10b2 −0.001) |
| C10_BEATS_C8C | **NO** |
| DEEP_RECALL_PRESERVED | **YES** (ALL@50 identical 0.9897 for every variant) |
| SET_AWARE_AXIS_EXHAUSTED | **YES** (best set-aware method ties C8c non-significantly; learned greedy hurts) |
| SAFE_TO_RUN_OFFICIAL_VAL | **NO** (nothing beats C8c on C9_DEV — do not spend the VAL milestone) |
| SAFE_TO_FREEZE_L2 | **NO** (held per instruction) |

**Conclusion.** The remaining ~0.0208 top-50 oracle gap (the 3+-gold tail) requires a genuinely **new discriminative
signal or a larger retrieval/graph scope** — not set-aware reselection of the current pool. **Standing top-5 policy
remains C8c.**

**STOP (C10 complete):** audit gate + MMR + learned selector + set oracles done; selected on C9_DEV, confirmed once
on DEV_INNER. Official VAL **not** run, TEST **not** inspected, pool not enlarged, no cross-encoder/LLM/graph added,
L1 unchanged, L3 not started, L2 not frozen. Returned for review before any official-VAL milestone.

Artifacts: `results/L2/L2_C10_SET_AWARE.json`, `results/L2/_ctrl/{_c10_audit,_c10_mmr,_c10b,_c10b2}.json`,
selectors `results/L2/_ctrl/{C10b_selector,C10b2_selector}.joblib`, cache `results/L2/_ctrl/_c10b_cache.joblib`.
