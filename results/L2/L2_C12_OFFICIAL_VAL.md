# L2 — C12 Official-VAL Milestone (C8c / C11a / C12a)

**Date:** 2026-08-27 · **Datasets:** 2wiki_clean VAL (n=3000) + musique_clean VAL (n=3985), pooled n=**6985** ·
**One evaluation, three pre-declared policies, TEST untouched.** Backbone (C7b soft-archetype fusion + C8c
XGBRanker residual reranker) refit on FULL TRAIN, early-stop DEV_INNER, then frozen; C11a/C12a applied
UNCHANGED. No refit on VAL, no retune, no threshold-chasing. Reorder within top-20 → **ALL@50 preserved by
construction**. `NEW_ENCODER_FORWARD=0`, `NEW_LLM_COMPONENT=0`. Freeze manifest written **before** any VAL
access (`L2_PREVAL_FREEZE_MANIFEST.json`).

## TL;DR — the protected VAL did its job

**C11a generalizes; C12a does not.** On official VAL, the per-candidate interaction MLP (**C11a**) beats the
C8c reranker significantly and on *both* datasets. The DeepSets list-context model (**C12a**) — which won on
C9_DEV by moving the 3+-gold tail — **fails to replicate**: it is *significantly below C11a* on NDCG@5,
GOLD_RECALL@5, and ALL@5, and its 3+-gold advantage inverts. The C12a dev gain was development-set-specific.

> **`FINAL_TEXT_L2_POLICY = C11a`.** C12a is **not** promoted.

- `C8C_VAL_PARITY = PASS` (exact — all five reference metrics reproduced to 4 dp, Δ=0.0)
- `C11A_BEATS_C8C_ON_VAL = YES` (sig on NDCG@5 / GOLD_RECALL@5 / ALL@5_feas; replicates on both datasets)
- `C12A_BEATS_C11A_ON_VAL = **NO**` (significantly *worse* on NDCG@5 / recall@5 / ALL@5)
- `THREE_PLUS_GOLD_GAIN_REPLICATES = **NO**` (the dev mechanism reverses on VAL)
- `DEEP_RECALL_PRESERVED = YES` (ALL@50 identical across all three, both datasets)
- `SAFE_TO_BEGIN_CROSS_DATASET_GENERALIZATION = YES` · `SAFE_TO_RUN_LOCKED_TEST = NO`

## Main table — official VAL (pooled, n-weighted)

| Metric | C8c | **C11a** | C12a | C11a−C8c | C12a−C11a |
|---|---|---|---|---|---|
| **NDCG@5** | 0.8824 | **0.8914** | 0.8888 | **+0.0090** | −0.0026 |
| GOLD_RECALL@5 | 0.9030 | **0.9093** | 0.9039 | **+0.0063** | −0.0054 |
| ANY@5 | 0.9941 | 0.9941 | 0.9940 | 0.0000 | −0.0001 |
| ALL@5 (feasible) | 0.7771 | **0.7924** | 0.7810 | **+0.0153** | −0.0114 |
| MRR | 0.9569 | 0.9591 | 0.9613 | +0.0022 | +0.0022 |
| NDCG@50 | 0.9114 | 0.9176 | 0.9176 | +0.0062 | 0.0000 |
| ALL@10 | 0.8736 | 0.8763 | 0.8805 | +0.0027 | +0.0042 |
| **ALL@50** | **0.9406** | **0.9406** | **0.9406** | **0.0000** | **0.0000** |

### Per dataset

| | NDCG@5 | GOLD_R@5 | ALL@5f | ALL@50 |
|---|---|---|---|---|
| **2wiki** C8c / C11a / C12a | 0.8834 / **0.9021** / 0.8999 | 0.9044 / **0.9184** / 0.9142 | 0.7617 / **0.7943** / 0.7857 | 0.9217 (all) |
| **musique** C8c / C11a / C12a | 0.8816 / **0.8834** / 0.8805 | 0.9019 / **0.9023** / 0.8961 | 0.7887 / **0.7910** / 0.7774 | 0.9548 (all) |

C11a is the top NDCG@5 on both datasets; C12a is below C11a on both. Deep recall (ALL@50) is bit-identical
across all three policies per dataset — the reorder-within-top-20 invariant held exactly.

## C8c VAL parity (reproduces the frozen reference)

| Metric | C8c (this run) | Reference | Δ |
|---|---|---|---|
| NDCG@5 | 0.8824 | 0.8824 | 0.0000 |
| GOLD_RECALL@5 | 0.9030 | 0.9030 | 0.0000 |
| MRR | 0.9569 | 0.9569 | 0.0000 |
| NDCG@50 | 0.9114 | 0.9114 | 0.0000 |
| ALL@50 | 0.9406 | 0.9406 | 0.0000 |

**`C8C_VAL_PARITY = PASS`** — exact to 4 dp. The full-train backbone here *is* the canonical C8c; the C11a/C12a
deltas are measured against the correct reference.

## Paired bootstrap (2000 resamples, pooled per-query, CI95)

| Comparison | NDCG@5 | GOLD_R@5 | ALL@5_feas | MRR |
|---|---|---|---|---|
| **C11a vs C8c** | **+0.0090** [0.0071, 0.0110] ✓ | **+0.0063** [0.0036, 0.0088] ✓ | **+0.0153** [0.0096, 0.0210] ✓ | +0.0022 [−0.0002, 0.0046] ns |
| **C12a vs C11a** | **−0.0026** [−0.0044, −0.0008] ✓worse | **−0.0054** [−0.0078, −0.0029] ✓worse | **−0.0115** [−0.0168, −0.0059] ✓worse | +0.0022 [0.0001, 0.0042] ✓ |
| C12a vs C8c | +0.0065 [0.0045, 0.0085] ✓ | +0.0009 [−0.0018, 0.0035] ns | +0.0039 [−0.0019, 0.0099] ns | +0.0044 [0.0022, 0.0066] ✓ |

C11a's advantage over C8c is significant on the three retrieval-quality metrics. C12a is *significantly worse
than C11a* on all three of those metrics (it wins only a small MRR reshuffle). C12a beats C8c only on NDCG@5 /
MRR — a strictly weaker, dominated improvement.

## Multi-gold breakdown — the dev mechanism reverses

On C9_DEV, C12a's whole story was the **3+-gold** (multi-evidence) tail. On VAL that inverts:

| bin (pooled) | n | C8c R@5 / ALL@5 | C11a R@5 / ALL@5 | C12a R@5 / ALL@5 |
|---|---|---|---|---|
| 1-gold | 161 | 0.9503 / 0.9503 | 0.9503 / 0.9503 | **0.9565 / 0.9565** |
| 2-gold | 5177 | 0.9265 / 0.8590 | **0.9306 / 0.8671** | 0.9256 / 0.8573 |
| **3+-gold** | 1647 | 0.8245 / 0.5027 | **0.8380 / 0.5422** | 0.8307 / 0.5240 |

**C11a**, not C12a, is what lifts the 2-gold and 3+-gold tails on VAL. C12a instead nudges the small 1-gold
bin up and gives back the 2-/3+-gold gains — the opposite of its hypothesized (and dev-observed) mechanism.
`THREE_PLUS_GOLD_GAIN_REPLICATES = NO`.

## Gold-rescue (C12a relative to C11a) — net negative

| ds | rescued into top-5 (5–9 / 10–19) | pushed out | **NET** |
|---|---|---|---|
| 2wiki | 121 (111 / 10) | 154 | **−33** |
| musique | 85 (80 / 5) | 146 | **−61** |

Relative to C11a, C12a's list-context reshuffle evicts more golds from the top-5 than it rescues on both
datasets. (Against C8c, C12a is net +87 on 2wiki but −53 on musique — again dominated by, and less consistent
than, C11a.) The context-effect term is directionally sane — rescued golds carry a positive mean Δ_context
(2wiki 11.40, musique 8.06) — but it also promotes non-golds at least as strongly (12.50 / 9.46), so on VAL
the net set-context effect is not gold-selective. This is the concrete failure: the learned list context does
not generalize as a gold-discriminative signal.

## Why C12a didn't replicate (interpretation, no new experiments)

The C11a per-candidate interaction — query↔candidate matching over the frozen Dense/SPLADE/offset reps — is a
**local, well-identified** signal and it generalizes cleanly (+0.90pt NDCG@5, both datasets, all sig). The C12
addition — a permutation-invariant leave-one-out set context over the top-20 — added ~0 parameters of *new
identifiable structure* but a lot of capacity to fit the **specific composition of the C9_DEV top-20 windows**.
The 3+-gold tail is exactly the lowest-support, highest-variance regime (dev 3+ n=776; the effect there was
~+0.006 NDCG@5), so a set-context head that helped it on dev had the most room to be noise — and on the
independent VAL windows it reverses. C11a's gain, by contrast, sits in the high-support 2-gold bulk and holds.
**Consistent with the C10 finding** (`l2-c10-set-aware-diagnostic-only`): set-novelty is diagnostic but not
reliably *convertible* into a discriminative top-5 signal. C12 converted it on dev; VAL says that conversion
was development-specific.

## Deep-recall invariant

`ALL@50`: C8c = C11a = C12a to machine precision — pooled 0.9406, 2wiki 0.9217, musique 0.9548.
**`DEEP_RECALL_PRESERVED = YES`.** All three policies only reorder inside the top-20 window; the deep pool is
untouched, so no candidate-generation change is smuggled in.

## Decision

This is the **"C11a stands, C12a not promoted"** branch: C11a beats C8c on protected VAL and generalizes to
both datasets; C12a fails to beat C11a (significantly worse on the retrieval metrics) and its dev mechanism
does not replicate. Per the pre-registered rule (select the simplest policy that is VAL-validated and not
dominated), the standing L2 text top-5 policy is **C11a** — the per-candidate query↔candidate interaction MLP,
a zero-init residual on the frozen C8c reranker, one shared network, no dataset ID, no new encoder/LLM. C12a
is retained only as a negative-result ablation arm.

**No retraining, no C13, no retune, no TEST** was performed after seeing VAL, and none follows from this
milestone.

## Gates

| Gate | Value |
|---|---|
| C8C_VAL_PARITY | **PASS** (exact) |
| C11A_BEATS_C8C_ON_VAL | **YES** (sig, both datasets) |
| C12A_BEATS_C11A_ON_VAL | **NO** (sig worse on NDCG@5/R@5/ALL@5) |
| C12A_BEATS_C8C_ON_VAL | YES (NDCG@5/MRR only; dominated by C11a) |
| THREE_PLUS_GOLD_GAIN_REPLICATES | **NO** |
| LEARNED_LIST_CONTEXT_GENERALIZES_TO_VAL | **NO** |
| DEEP_RECALL_PRESERVED | **YES** (ALL@50 identical) |
| **FINAL_TEXT_L2_POLICY** | **C11a** |
| C12_OFFICIAL_VAL_REPLICATES | **NO** |
| SAFE_TO_BEGIN_CROSS_DATASET_GENERALIZATION | **YES** (on C11a) |
| SAFE_TO_RUN_LOCKED_TEST | **NO** |
| NEW_ENCODER_FORWARD / NEW_LLM_COMPONENT | PASS / PASS |

## Generalization prep note

The VAL-validated interface to carry into cross-dataset work is **C11a**: inputs = frozen gte_qwen query rep
`queries_all[row_all]` + node reps `nodes.npy` (unit-norm) + the 18 expert/RRF features `E` standardized with
the *saved* training `emean/estd`; a top-20 window over C8c scores; output = zero-init residual on the z-scored
C8c window score. This is **parameter generalization** (one frozen net, no per-dataset fitting, no dataset ID) —
the correct object to test on a held-out corpus. C12a is **not** carried forward: its official-VAL result shows
its extra list-context capacity does not transfer even in-distribution.

## KB / dependency note

C11a depends on no dataset ID, no gold count, no hand-coded thresholds, and no dataset-specific rule — it is a
per-candidate matching residual, identical code and weights across 2wiki and musique. That is what makes its
VAL result a genuine generalization signal and makes it safe to begin cross-dataset (KB) generalization from.

**Artifacts:** `results/L2/L2_C12_OFFICIAL_VAL.json` · `results/L2/L2_PREVAL_FREEZE_MANIFEST.json` · models
`_ctrl/C11_models.joblib` (C11a) · driver `scratchpad/_run_c12_val.py`.
