# L2 C11 — Learned Query–Candidate Interaction MLP (residual on C8c)

**Date:** 2026-08-27 · **Datasets:** 2wiki_clean + musique_clean (joint, no dataset ID) ·
**Selection split:** C9_DEV · **Confirmation split:** DEV_INNER (one confirmation) ·
**No official VAL. No TEST. No L1 change. No L3. No pool enlargement (top-20). No new expert.**

## TL;DR

C11a — a **pure interaction MLP** scoring `S_final = S_C8c + β·Δ_MLP` on frozen gte_qwen
query/node embeddings + compact retrieval evidence — is **the first method in the C9/C10/C11
line to significantly beat C8c**, with deep recall left **exactly** intact. The multi-offset
variant C11b adds nothing over C11a. Standing top-5 candidate becomes **C11a**, pending official
VAL (not run, per instruction).

- `MLP_INTERACTION_ADDS_VALUE = YES` · `C11_BEATS_C8C = YES` · `MULTIGOLD_RECALL_IMPROVES = YES`
- `MULTI_OFFSET_ADDS_VALUE = NO` · `MULTI_OFFSET_COLLAPSE = NO`
- `DEEP_RECALL_PRESERVED = YES` (ALL@50 identical) · `NO_NEW_LLM_COMPONENT = PASS`
- `NEW_ENCODER_FORWARD_PASSES = 0` · `NEW_LLM_COMPONENTS = 0`
- `SAFE_TO_RUN_OFFICIAL_VAL = YES` · **`SAFE_TO_FREEZE_L2 = NO`** (held per instruction; VAL not yet run)

## Method

- **Representation (frozen, reused — no new inference):** gte_qwen query embedding (recovered via
  the cached `qmap` corpus_qi→queries_train permutation; dense_score is affine in cosine, corr≈1.0)
  and gte_qwen candidate node embeddings, unit-norm. This IS the dense expert's encoder — nothing
  new is encoded.
- **Learned model = MLP only.** Learned projections `Pq, Pd : D=1536 → H=256` (GELU); interaction
  `x = [hq, hd, hq*hd, |hq−hd|, E]`; scorer `Linear(→512)→LayerNorm→GELU→Dropout(0.1)→Linear(→128)
  →GELU→Linear(→1)`. **No attention, no transformer, no GNN, no recurrence, no cross-encoder.**
- **Compact evidence E (18 dims), not the 63 C9 features:** C8c OOF score/rank, C7b fused/base
  score, 5 expert RRF contribs, 5 expert ranks, relation mask, max/second contrib, n_top5,
  votes_top10.
- **Residual scoring with zero-init output layer** ⇒ at init the ranking == C8c exactly (avoids the
  C10b relearn-from-scratch collapse). Learned scalar `β` (converged ≈1.10).
- **Training:** pool = C8c **top-20** per query; **pairwise RankNet**, multi-positive (every gold vs
  every non-gold), pair weights **up-weighting the exact repair case** (a negative currently ranked
  <5 paired against a positive currently ranked 5–19). C8c backbone score is **OOF on TRAIN**
  (anti-leakage). Trained on C9_TRAIN, selected on C9_DEV, confirmed once on DEV_INNER.
- **C11b (multi-offset):** from `hq`, predict K=4 latent offsets `o_k`; `u_k=hq+o_k`;
  `m_k=[u_k*hd, |u_k−hd|]`; small `f(m_k)`; query-dependent softmax mixture `a_k(q)` **plus** an
  elementwise max-over-k specialist path; offset L2 reg + collapse check.

## Results — C9_DEV (selection) vs C8c

| Metric | C8c | **C11a** | C11b | C11a Δ vs C8c | boot CI95 | sig |
|---|---|---|---|---|---|---|
| NDCG@5 | 0.9658 | **0.9685** | 0.9679 | **+0.0027** | [0.0007, 0.0047] | **YES** |
| RECALL@5 (macro) | 0.9762 | **0.9799** | 0.9792 | **+0.0037** | [0.0013, 0.0063] | **YES** |
| ALL@5 (feasible) | 0.9331 | **0.9425** | 0.9413 | +0.0094 | — | — |
| MRR | 0.9893 | 0.9894 | 0.9887 | +0.0001 | — | — |
| NDCG@50 | 0.9748 | 0.9761 | 0.9757 | +0.0013 | [−0.0001, 0.0027] | no |
| ALL@10 | 0.9749 | 0.9791 | 0.9785 | +0.0042 | — | — |
| **ALL@50 (deep recall)** | **0.9897** | **0.9897** | **0.9897** | **0.0000** | [0, 0] | — (preserved by construction) |

## Confirmation — DEV_INNER (one confirmation)

| Metric | C8c | **C11a** | C11b |
|---|---|---|---|
| NDCG@5 | 0.9637 | **0.9672** (+0.0035) | 0.9693 |
| RECALL@5 | 0.9748 | **0.9795** (+0.0047) | 0.9800 |
| ALL@5 (feas) | 0.9312 | 0.9419 | 0.9447 |
| ALL@50 | 0.9836 | 0.9836 | 0.9836 |

C11a's win **replicates on the independent DEV_INNER split** (+0.0035 NDCG@5, +0.0047 RECALL@5),
deep recall again identical. C11b scores marginally higher than C11a on DEV_INNER, but selection was
pre-registered on **C9_DEV** where C11a wins and C11b's extra offset machinery costs; we honor the
selection split. C11b is **not** the standing candidate.

> All splits are TRAIN-derived (metrics ≈0.96) and are **not** comparable to official VAL
> (C8c VAL ≈0.8824). The claim is the *consistent, significant delta over C8c on the same held-out
> splits*, not the absolute level.

## Why the MLP wins where C9 and C10 failed

- **C9** (63 static per-candidate features) reconstructed C8c (0.89 feat-importance) and was neutral.
- **C10** (set-aware MMR / greedy selectors) found a real novelty *diagnostic* but it was not
  *discriminative* — the novel population is dominated by non-gold distractors.
- **C11** does not add features or reselect; it learns a **query-conditioned interaction** between the
  frozen query and candidate embeddings that the fixed dense-cosine / RRF experts cannot express, and
  applies it as a **bounded residual** on top of C8c's joint listwise ordering. Residual + zero-init +
  top-5-focused pairwise loss is exactly the design that avoids C10b's collapse.

## Multi-gold tail (where the headroom is)

| bin | n | C11a RECALL@5 | C11a ALL@5 | C11a NDCG@5 |
|---|---|---|---|---|
| 1-gold | 79 | 1.000 | 1.000 | 0.977 |
| 2-gold | 2448 | 0.990 | 0.981 | 0.975 |
| 3+-gold | 776 | 0.945 | **0.814** | 0.947 |

Single-gold preserved (RECALL@5 = 1.0). Gains land in the multi-gold tail: 3+-gold ALL@5 0.814
(vs the ~0.799 C8c baseline recorded in C10) → `MULTIGOLD_RECALL_IMPROVES = YES`.

## Gold-rescue / false-positive analysis (C11a)

- **Net top-5 rescue +32** (84 golds moved into top-5, 52 pushed out). By multi-gold bin the net is
  positive everywhere: single +3, 2-gold +21, 3+-gold +8. Rescues come almost entirely from the
  reachable boundary (rank 5–9: 73; 10–19: 11; 20–49: 0).
- **Rescued golds are spread across specialist experts** (strongest expert of each rescued gold:
  relation 25, mixture 23, splade 15, dense 14, offset 7) — the MLP is not just re-amplifying the
  dense axis it was built from.
- **`SPECIALIST_RESCUE_VALIDATION_LEARNED = NO`:** the strict signature Δ(gold)>0 ∧ Δ(FP)<0 does not
  hold in raw units — both mean deltas are negative (rescued-gold −7.40, removed-FP −19.43). The
  *relative* direction is nonetheless correct (FPs are demoted ~2.6× harder than golds), which is what
  produces the net rescue; the strict per-item gate simply isn't met. Diagnostic, not a blocker.

## Multi-offset (C11b) verdict

`MULTI_OFFSET_ADDS_VALUE = NO`. Offsets are genuinely diverse (mean pairwise offset cosine 0.2248 ⇒
`MULTI_OFFSET_COLLAPSE = NO`), so the null result is **not** a degenerate collapse — the K=4 latent
offsets + max-specialist path simply do not beat the single interaction MLP on the selection split
(C9_DEV NDCG@5 0.9679 < C11a 0.9685) at +367k params and +260s train. Single interaction is the
right complexity for this regime.

## Cost / KB-friendliness

- C11a: **1,387,778 params**; 20 candidate evaluations/query (top-20 pool); CPU.
- No dataset ID input; identical architecture on both datasets ⇒ representation-general, KB-ready
  (abstract query-vec / candidate-vec / evidence interface).
- `NEW_ENCODER_FORWARD_PASSES = 0`, `NEW_LLM_COMPONENTS = 0`, `NO_NEW_LLM_COMPONENT = PASS`.

## Gates

| Gate | Value |
|---|---|
| MLP_INTERACTION_ADDS_VALUE | **YES** |
| MULTI_OFFSET_ADDS_VALUE | NO |
| MULTI_OFFSET_COLLAPSE | NO |
| C11_BEATS_C8C | **YES** |
| MULTIGOLD_RECALL_IMPROVES | YES |
| SPECIALIST_RESCUE_VALIDATION_LEARNED | NO (relative direction correct; strict Δ gate unmet) |
| DEEP_RECALL_PRESERVED | YES (ALL@50 identical) |
| NO_NEW_LLM_COMPONENT | PASS |
| SAFE_TO_RUN_OFFICIAL_VAL | **YES** |
| **SAFE_TO_FREEZE_L2** | **NO** (held per instruction; official VAL not yet run) |

## Standing decision & next step

**Standing top-5 candidate is now C11a** (was C8c) — the first method to significantly and
reproducibly beat C8c while preserving deep recall exactly. Per the C11 stop rule, we **STOP after
C9_DEV selection + one DEV_INNER confirmation** and **return before official VAL**. The one remaining
step to promote C11a from *candidate* to *frozen L2 top-5 policy* is a single official-VAL evaluation
(C11a vs C8c) — **not run here**; awaiting go-ahead. Do not run VAL/TEST, add a cross-encoder/LLM/new
encoder, change L1, start L3, or enlarge the pool.

**Artifacts:** `results/L2/L2_C11_MLP_INTERACTION.json` · models `_ctrl/C11_models.joblib` ·
engine `scratchpad/l2_c11.py` · prep `scratchpad/_run_c11_prep.py` + `_c11_qmap.py` · driver
`scratchpad/_run_c11.py`.
