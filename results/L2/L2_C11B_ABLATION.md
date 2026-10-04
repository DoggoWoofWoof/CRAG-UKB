# L2 C11b — Multi-Offset Interaction MLP: ablation vs C11a

**Date:** 2026-08-27 · **Datasets:** 2wiki_clean + musique_clean (one network, no dataset ID, K=4) ·
**Select:** C9_DEV · **Confirm:** DEV_INNER (once) · **No VAL/TEST · no new encoder/cross-encoder/attention.**
Embedding space validated (see `L2_C11_ALIGNMENT_AUDIT.md`): all reps are the exact frozen Dense space.

## Scientific question & verdict

*Does explicit multi-offset query–candidate interaction improve over the already-successful single
interaction MLP C11a?* **No.** C11b retains the full C11a path and adds a K=4 offset branch (mixture +
elementwise-max, shared matching MLP, zero-init residual, small offset-norm penalty). It **does not beat
C11a** (C9_DEV NDCG@5 −0.0006, ns) and the offsets, though genuinely diverse (no collapse), **do not
specialize by gold** (gold spread == random non-gold spread). **C11_FINAL_POLICY = C11a.**

## Ablation — C8c / C11a / C11b (C9_DEV, selection)

| Metric | C8c | **C11a** | C11b |
|---|---|---|---|
| NDCG@5 | 0.9658 | **0.9685** | 0.9679 |
| GOLD_RECALL@5 (macro) | 0.9762 | **0.9799** | 0.9792 |
| ALL@5 | 0.9997 | 0.9997 | 0.9997 |
| ALL@5 (feasible) | 0.9331 | **0.9425** | 0.9413 |
| MRR | 0.9893 | 0.9894 | 0.9887 |
| NDCG@50 | 0.9748 | 0.9761 | 0.9757 |
| **ALL@50** | **0.9897** | **0.9897** | **0.9897** |

DEV_INNER confirmation: C8c 0.9637 / C11a 0.9672 / C11b 0.9693 NDCG@5; ALL@50 identical (0.9836).
(C11b edges C11a on DEV_INNER but loses on the pre-registered selection split C9_DEV — selection stands.)

## Paired bootstrap

| Comparison | NDCG@5 Δ [CI95] | sig | GOLD_RECALL@5 Δ [CI95] | sig |
|---|---|---|---|---|
| C11a vs C8c | **+0.0027** [0.0007, 0.0047] | **YES** | **+0.0037** [0.0013, 0.0063] | **YES** |
| **C11b vs C11a** | **−0.0006** [−0.0022, 0.0010] | **NO** | **−0.0007** [−0.0026, 0.0013] | **NO** |
| C11b vs C8c | +0.0021 [0.0000, 0.0040] | YES | +0.0031 [0.0004, 0.0056] | YES |

C11b's gain over C8c is entirely inherited from the shared C11a path; the offset branch contributes
nothing over C11a (CI straddles 0, sign negative).

## Multi-gold breakdown (the hypothesis's target regime)

The premise for multi-offset is that different latent query directions support different evidence pieces —
so the win, if any, should appear on 2-gold / 3+-gold queries. It does not.

| bin | n | C8c R@5 / ALL@5 | C11a R@5 / ALL@5 | C11b R@5 / ALL@5 |
|---|---|---|---|---|
| 1-gold | 79 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| 2-gold | 2448 | 0.9867 / 0.9734 | **0.9904 / 0.9812** | 0.9888 / 0.9779 |
| 3+-gold | 776 | 0.9404 / 0.7990 | 0.9447 / 0.8144 | 0.9471 / 0.8196 |

C11b is **worse than C11a on 2-gold** and only trivially higher on 3+-gold (R@5 +0.0024, ALL@5 +0.0052) —
within noise and not significant in aggregate. No coherent multi-offset advantage in the target regime.
(Both C11a and C11b improve the multi-gold tail over C8c — that gain is the single-interaction MLP's,
`MULTIGOLD_RECALL_IMPROVES = YES` for the standing policy C11a vs C8c.)

## Offset diagnostics — diverse but not collapsed, and not gold-aligned

- **No collapse:** mean pairwise offset cosine **0.2174** (`MULTI_OFFSET_COLLAPSE = NO`); per-offset mean
  norms [0.449, 0.428, 0.454, 0.437], std across offsets 0.010 (balanced, none vanishing). The K=4
  offsets are genuinely distinct directions.
- **No gold specialization:** per-candidate winning offset = argmax_k max f(m_k). Among 3,215 queries with
  ≥2 golds in-window, golds span ≥2 distinct offsets in **56.05%** of queries (mean distinct offsets
  **1.608**). Size-matched **non-gold control: 56.64%, mean 1.619** — statistically identical. Gold offset
  win-counts [1633, 1620, 2579, 1749] mirror the non-gold shape [12343, 12785, 19489, 13862]. **Offsets
  emerge as generic diversity that carries no gold-specific structure** (offsets were never assigned to
  golds — labels are ranking supervision only). `MULTI_OFFSET_SPECIALIZATION_RATE = 0.5605`.

This is the mechanistic reason the ablation is null: the multi-offset branch fires the same way on golds
and distractors, so it cannot add discriminative top-5 signal beyond the single interaction.

## C11c (diversity auxiliary) — NOT run, not justified

The spec authorizes at most one C11c **only if** C11b improves C11a *but* offsets show collapse / poor
specialization. Here **C11b does not improve C11a** (non-significant, negative point estimate) **and there
is no collapse** to repair (offsets already diverse at cos 0.22). A diversity penalty targets collapse;
the failure here is gold-agnostic diversity, which more diversity cannot fix. Precondition unmet on both
counts → **C11c not run.**

## Model size / cost

| | C11a | C11b | Δ |
|---|---|---|---|
| params | 1,387,778 | 1,754,502 | +366,724 |
| train (s) | 815.6 | 1076.9 | +261.3 |
| inference (ms/query, top-20, CPU) | 0.205 | 0.333 | +0.128 |

Multi-offset costs +26% params, +32% train, +62% inference/query for no accuracy gain — the intended
research story (small MLP + frozen retrieval reps, not a large reranker) favors the single interaction.

## Final C11 decision

| Gate | Value |
|---|---|
| MLP_INTERACTION_ADDS_VALUE | **YES** |
| MULTI_OFFSET_ADDS_VALUE | **NO** |
| MULTI_OFFSET_COLLAPSE | NO |
| MULTI_OFFSET_SPECIALIZATION_RATE | 0.5605 (≈ non-gold 0.5664 → not gold-aligned) |
| MULTIGOLD_RECALL_IMPROVES | YES (C11a vs C8c; C11b-vs-C11a is a wash) |
| C11B_BEATS_C11A | **NO** |
| **C11_FINAL_POLICY** | **C11a** |
| DEEP_RECALL_PRESERVED | YES (ALL@50 identical) |
| NO_NEW_ENCODER_FORWARD | PASS |
| NO_NEW_LLM_COMPONENT | PASS |

**C11_FINAL_POLICY = C11a** (single interaction MLP). C11b is rejected — not preferred despite being more
complex. Per the stop rule we STOP after C11b + one DEV_INNER confirmation (C11c not justified) and return
**before official VAL**. No VAL/TEST run, L1 unchanged, no L3, no new model families. `SAFE_TO_FREEZE_L2`
remains NO (held; official VAL not yet run).

**Artifacts:** `results/L2/L2_C11B_ABLATION.json` · analysis `scratchpad/_run_c11b_analysis.py` ·
models `_ctrl/C11_models.joblib` · engine `scratchpad/l2_c11.py`.
