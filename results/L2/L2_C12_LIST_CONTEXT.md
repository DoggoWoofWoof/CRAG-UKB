# L2 C12 — Learned List-Context (DeepSets) Residual MLP

**Date:** 2026-08-27 · **Datasets:** 2wiki_clean + musique_clean (one shared net, no dataset ID) ·
**Select:** C9_DEV · **Confirm:** DEV_INNER (once) · **No VAL/TEST · no attention/transformer/GNN/LLM/encoder.**
Reps validated as the exact frozen Dense space (`L2_C11_ALIGNMENT_AUDIT.md`).

## TL;DR

**Learned list context beats C11a — significantly — and the gain lands exactly where predicted: the
multi-evidence (3+-gold) tail.** A DeepSets-style permutation-invariant leave-one-out set context over the
C8c top-20, fused with the C11a local interaction as a zero-init residual on C8c, lifts C9_DEV NDCG@5 to
**0.9712** (C11a 0.9685, C8c 0.9658). The **mean** context carries all the value; adding **max** does
nothing. Deep recall exactly preserved. **New standing candidate: C12a.**

- `LEARNED_LIST_CONTEXT_ADDS_VALUE = YES` · `DEEPSETS_MEAN_HELPS = YES` · `DEEPSETS_MAX_HELPS = NO`
- `C12_BEATS_C11A = YES` · `C12_BEATS_C8C = YES` · `THREE_PLUS_GOLD_IMPROVES = YES`
- `DEEP_RECALL_PRESERVED = YES` · `NO_NEW_LLM_COMPONENT = PASS` · `NO_NEW_ENCODER_FORWARD = PASS`
- **`FINAL_PRE_VAL_POLICY = C12a`** · `SAFE_TO_RUN_OFFICIAL_VAL = YES` · **`SAFE_TO_FREEZE_L2 = NO`** (held)

## Method

Per candidate: `hq=Pq(q)`, `hd=Pd(d)`, local interaction `z_i = LocalMLP([hq,hd,hq*hd,|hq−hd|, E])` (the
useful C11a representation, Z=128). Over the 20-candidate window, a **permutation-invariant leave-one-out**
set context (exact, no self-leakage): `mean_{j≠i} z_j` (C12a), plus `max_{j≠i} z_j` (C12b). Candidate×list
interaction `[z_i, c_i, z_i·c_i, |z_i−c_i|]` + query condition `hq` + a compact 15-dim expert-set summary
(per-expert mean/max/std of the 5 RRF contribs over the window) → `ContextMLP` → `Δ_i`. Residual
`S_i = S_C8c + β·Δ_i` with **zero-init output** (init ranking == C8c; ALL@50 preserved by construction).
Listwise multi-positive top-5-weighted RankNet loss (up-weights pos@rank5–19 vs neg@rank0–4). Pure MLP +
mean/max pooling — no attention, transformer, GNN, recurrence, or new encoder. Leave-one-out max is exact
(ordinary max self-leakage measured at 5% of dims — removed by the LOO construction).

## Results — C9_DEV (selection)

| Metric | C8c | C11a | **C12a (mean)** | C12b (mean+max) |
|---|---|---|---|---|
| NDCG@5 | 0.9658 | 0.9685 | **0.9712** | 0.9713 |
| GOLD_RECALL@5 | 0.9762 | 0.9799 | **0.9816** | 0.9815 |
| ALL@5 (feas) | 0.9331 | 0.9425 | **0.9476** | 0.9470 |
| MRR | 0.9893 | 0.9894 | 0.9913 | 0.9917 |
| NDCG@50 | 0.9748 | 0.9761 | 0.9780 | 0.9781 |
| **ALL@50** | **0.9897** | **0.9897** | **0.9897** | **0.9897** |

### Paired bootstrap

| Comparison | NDCG@5 Δ [CI95] | sig | GOLD_RECALL@5 Δ [CI95] | sig |
|---|---|---|---|---|
| **C12a vs C11a** | **+0.0027** [0.0011, 0.0043] | **YES** | +0.0017 [−0.0003, 0.0035] | no |
| C12b vs C11a | +0.0028 [0.0010, 0.0048] | YES | +0.0016 [−0.0003, 0.0036] | no |
| **C12a vs C8c** | **+0.0053** [0.0031, 0.0076] | **YES** | **+0.0054** [0.0027, 0.0081] | **YES** |
| C12b vs C8c | +0.0055 [0.0033, 0.0078] | YES | +0.0053 [0.0027, 0.0079] | YES |
| **C12b vs C12a** | **+0.0002** [−0.0015, 0.0019] | **NO** | −0.0001 [−0.002, 0.002] | no |

C12 significantly beats C11a on NDCG@5 (the primary test). **Max context does not beat mean** (Δ+0.0002,
ns) → policy is the simpler **C12a**; C12b is retained only as the ablation arm, not preferred for its extra
complexity (parsimony per spec). *(The pre-registered selector picked C12b by a +0.0001 tie; the C12b-vs-C12a
bootstrap shows that is noise, so the final policy is corrected to C12a.)*

## Confirmation — DEV_INNER (one confirmation)

| | C8c | C11a | C12a | C12b |
|---|---|---|---|---|
| NDCG@5 | 0.9637 | 0.9672 | **0.9680** | 0.9688 |
| GOLD_RECALL@5 | 0.9748 | 0.9795 | 0.9761 | 0.9787 |
| ALL@50 | 0.9836 | 0.9836 | 0.9836 | 0.9836 |

Both C12 variants exceed C11a on NDCG@5 on the independent split (C12a +0.0008, C12b +0.0016), confirming
direction; margins are smaller than on C9_DEV (expected). Deep recall identical.

## Multi-gold breakdown — the mechanism (this is the story)

The hypothesis: list context should barely move single-gold and lift 2-/3+-gold. **Confirmed.**

| bin | n | C11a R@5 / ALL@5 / NDCG@5 | **C12a** R@5 / ALL@5 / NDCG@5 |
|---|---|---|---|
| 1-gold | 79 | 1.000 / 1.000 / 0.9766 | 1.000 / 1.000 / 0.9766 *(unchanged)* |
| 2-gold | 2448 | 0.9904 / 0.9812 / 0.9750 | 0.9906 / 0.9820 / 0.9767 |
| **3+-gold** | 776 | 0.9447 / 0.8144 / 0.9472 | **0.9511 / 0.8338 / 0.9532** |

3+-gold is where it moves: **RECALL@5 +0.0064, ALL@5 +0.0194, NDCG@5 +0.0060** over C11a (C12b similar:
0.9527 / 0.8363 / 0.9555). Single-gold is untouched. This is the first method in the C10→C12 line to shift
the multi-evidence tail — C10's engineered set-novelty failed here, C11's per-candidate interaction barely
touched it. **Learned list context is what the 3+-gold regime needed.** `THREE_PLUS_GOLD_IMPROVES = YES`.

## Context-effect & specialist analysis

- Rescued golds (into C12 top-5, out of C11a top-5): C12a n=63 / C12b n=65, mean Δ_context > 0 (the
  context term raises them). Rescued golds are **spread across all five specialists** (C12a: splade 17,
  relation 14, mixture 13, dense 11, offset 8) — the list context is not just re-amplifying one axis.
- Rescued-gold expert **dominance** (max−second contrib) ≈ 0.002 (tiny) ⇒ rescued golds are *not*
  single-expert-dominated; the set context validates multi-evidence candidates, not lone-specialist spikes.
- The top-5 boundary reshuffles heavily (removed/promoted FP counts ~2.2k each across 3,303 queries), but
  the net is a significant NDCG@5 / 3+-gold gain — the ranking metrics, not the raw boundary churn, are the
  ground truth. (Absolute Δ_context magnitudes are in the z-scored window-score space and not directly
  comparable across models; only signs/counts are read here.)

## Model size / cost

| | C11a | **C12a** | C12b |
|---|---|---|---|
| params | 1,387,778 | **1,321,602** | 1,419,906 |
| train (s) | 815.6 | **345.0** | 379.5 |

C12a is **smaller than C11a** and trains faster, yet beats it — the win is architectural (list context),
not scale. Consistent with the "small MLP + frozen retrieval reps" research story. Top-20 pool, CPU.

## Gates

| Gate | Value |
|---|---|
| LEARNED_LIST_CONTEXT_ADDS_VALUE | **YES** |
| DEEPSETS_MEAN_HELPS | **YES** |
| DEEPSETS_MAX_HELPS | **NO** |
| MULTIGOLD_RECALL_IMPROVES | YES |
| THREE_PLUS_GOLD_IMPROVES | **YES** |
| C12_BEATS_C11A | **YES** |
| C12_BEATS_C8C | YES |
| DEEP_RECALL_PRESERVED | YES (ALL@50 identical) |
| NO_NEW_LLM_COMPONENT | PASS |
| NO_NEW_ENCODER_FORWARD | PASS |
| **FINAL_PRE_VAL_POLICY** | **C12a** |
| SAFE_TO_RUN_OFFICIAL_VAL | YES |
| **SAFE_TO_FREEZE_L2** | **NO** (held; official VAL not yet run) |

## Decision & next step

Learned list context **adds real value** (significant over C11a; the gain is the 3+-gold multi-evidence
tail). **Standing top-5 candidate advances C11a → C12a** (mean context; max rejected as non-additive; C12c/d
not invented per spec). Per the stop rule we STOP after C12a + C12b + one DEV_INNER confirmation and return
**before official VAL** — no VAL/TEST, L1 unchanged, no L3, no attention/transformer/GNN, no new pretrained
model, pool still top-20. Promotion of C12a to frozen L2 policy needs a single official-VAL check
(C12a vs C11a vs C8c) — not run here, awaiting go-ahead.

**Artifacts:** `results/L2/L2_C12_LIST_CONTEXT.json` · models `_ctrl/C12_models.joblib` · engine
`scratchpad/l2_c12.py` · driver `scratchpad/_run_c12.py`.
