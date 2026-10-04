# L2 Soft-Controller Phase — Results (VAL only)

**Scientific question.** Can a learned soft controller outperform fixed equal fusion by learning
*when* each expert should matter — and does candidate-conditioned gating beat a query-only gate?

**Answer (for THIS training regime): NO.** Under the declared protocol (RRF expert-contribution interface,
multi-positive RankNet loss on the canonical hard-negative TRAIN subset, full-P50 VAL evaluation,
init-inclusive selection on NDCG@50), **no learned controller — global, query-only, query+candidate,
+Shapley auxiliary, or +expert-dropout — beats the fixed 5-expert equal RRF fusion (C0).** Every trained
model's best epoch falls short of the equal-weight init, so selection reverts to C0 in all five rows.

> **FROZEN STATEMENT (do not overclaim).** The correct conclusion is
> `CURRENT_CONTROLLER_TRAINING_BEATS_EQUAL_RRF = NO`, **not** `ADAPTIVE_GATING_IS_IMPOSSIBLE = YES`. A
> per-query oracle ceiling of **+0.0395 NDCG@50** over equal fusion still exists (§4), so meaningful
> adaptive headroom is present; the tested inference features + tested training objectives failed to
> capture it — impossibility is **not** proven. The follow-on phase (C6, full-scope-aware controller
> training) tests whether that headroom is capturable when TRAIN supervision matches the full-P50
> multi-gold evaluation objective. See `L2_CONTROLLER_OBJECTIVE_ALIGNMENT.md`.

- **FROZEN INPUT:** MASTER_TOPOLOGY=C, L1=Dense+SPLADE, K=100, P=50. Experts E1 Dense, E2 Exact SPLADE,
  E3 Offset, E4 Mixture, E5 Masked-Qwen Relation. Interface = RRF contribution rᵢ(q,d)=1/(K0+rankᵢ),
  K0=60; r_relation=0 whenever relation_mask=0 (hard structural abstention, never overridden by gate weight).
- **Data:** 2Wiki TRAIN + MuSiQue TRAIN (balanced), VAL = 6985 queries (2wiki 3000 + musique 3985 in-scope).
  TEST never inspected. POSITIVE_COLLISIONS=0 in all runs.
- **Selection rule (declared pre-training):** primary NDCG@50, tie-break ALL@10 then ALL@50; init-inclusive
  (equal-fusion init is epoch −1, guaranteeing no regression below C0).

---

## 1. Controlled progression (POOLED VAL, selected model)

Each row adds exactly one capability over the row above. `sel-NDCG@50` is the **selected** model
(init-inclusive); `trained-best` / `trained-worst` show what the *trained* epochs actually did.

| Row | Capability | sel-NDCG@50 | MRR | ALL@50 | trained-best NDCG | trained-worst NDCG | selected |
|-----|------------|:-----------:|:---:|:------:|:-----------------:|:------------------:|:--------:|
| **C0** | fixed **equal** RRF (5 experts) | **0.8400** | 0.9064 | 0.9283 | — | — | (baseline) |
| C0_norel | equal RRF, 4 experts (no relation) | 0.8206 | 0.9243 | 0.8726 | — | — | (baseline) |
| C1 | learned **global** weights | 0.8400 | 0.9064 | 0.9283 | 0.8341 @e0 | 0.7962 @e7 | init (=C0) |
| C2 | **query-only** gate wᵢ(q) | 0.8400 | 0.9064 | 0.9283 | 0.7884 @e8 | 0.7814 @e1 | init (=C0) |
| C3 | **query+candidate** gate wᵢ(q,d) | 0.8400 | 0.9064 | 0.9283 | 0.7815 @e0 | 0.7814 @e1 | init (=C0) |
| C4 | C3 + **Shapley** positive-KL aux (λ=0.05) | 0.8400 | 0.9064 | 0.9283 | 0.8267 @e4 | 0.8039 @e2 | init (=C0) |
| C5 | C4 + **expert dropout** (p=0.3) | 0.8400 | 0.9064 | 0.9283 | 0.8383 @e4 | 0.8114 @e0 | init (=C0) |

**The repair ladder.** The bare adaptive gate (C2/C3) collapses to ~0.781 NDCG@50 on the full scope even
as MRR *rises* to ~0.931 — it learns to over-weight the top-precision experts (dense/splade) and suppress
the deep-recall specialists (relation/offset). Each regularizer partially repairs the collapse by pulling
the gate back toward uniform: bare 0.781 → +Shapley-KL 0.827 → +expert-dropout 0.838. The ladder
**monotonically approaches equal fusion (0.840) from below and never crosses it.** Every mechanism that
helps does so by *undoing* the gate's adaptivity, which is the mechanistic statement that adaptivity itself
does not help here.

## 2. C0 vs C0_norel — the relation specialist (sanity, matches Shapley)

Relation (E5) is a positive but narrow specialist, exactly as exact-Shapley found:

| Scope | metric | C0 (with relation) | C0_norel (no relation) | Δ (relation adds) |
|-------|--------|:---:|:---:|:---:|
| POOLED | NDCG@50 | 0.8400 | 0.8206 | **+0.0194** |
| POOLED | R@5 | 0.7933 | 0.7451 | +0.0482 |
| POOLED | ALL@10 | 0.7396 | 0.6726 | +0.0670 |
| POOLED | ALL@50 | 0.9283 | 0.8726 | +0.0557 |
| POOLED | MRR | 0.9064 | 0.9243 | −0.0179 |
| 2wiki | NDCG@50 | 0.8198 | 0.7679 | **+0.0519** |
| musique | NDCG@50 | 0.8553 | 0.8602 | **−0.0049** |

Relation lifts recall/completeness (ALL@K) at a small top-1 (MRR) cost — a deep-recall specialist. It is a
**2wiki specialist** (+0.052) and marginally *negative* on musique (−0.005). This is precisely the
per-query heterogeneity a controller *should* be able to exploit — and which the tested features + tested
objectives did not capture (Section 4).

## 3. Paired VAL bootstrap (2000 resamples, 95% CI) — SELECTED_CHECKPOINT_EQ_C0

**Label: `SELECTED_CHECKPOINT_EQ_C0`.** Init-inclusive selection chose the *exact* C0 state for every
learned row, so the selected-model per-query metrics are **byte-identical** to C0 and all deltas are
trivially zero. This is **not** evidence that the trained trajectories are statistically equivalent to C0
— they are significantly *worse* (§1 trajectory column). The zeros only record that selection fell back to
init; a genuine trained-vs-C0 bootstrap requires a checkpoint that differs from init (deferred to C6).

| Pair | NDCG@50 Δ [CI] | ALL@10 Δ | ALL@50 Δ | MRR Δ | significant |
|------|:---:|:---:|:---:|:---:|:---:|
| C0 vs C1 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | no |
| C0 vs C2 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | no |
| C2 vs C3 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | no |
| C3 vs C4 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | no |
| C0 vs C4 | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | no |

No pair shows a significant improvement. (The *trained* checkpoints differ — and are significantly
**worse**, per the trajectory column in §1 — but none is ever selected.)

## 4. Real headroom; tested features + objectives did not capture it

| gate | POOLED NDCG@50 | 2wiki | musique |
|------|:---:|:---:|:---:|
| equal fusion (C0) | 0.8400 | 0.8198 | 0.8553 |
| **per-query oracle** (best of 6 archetypes, uses VAL golds — ceiling only) | **0.8795 (+0.0395)** | 0.8741 (+0.0543) | 0.8836 (+0.0283) |
| inference-feature heuristic (relation_any_signal → up_relation) | 0.8175 (**−0.0225**) | 0.8215 | 0.8146 |

There **is** latent per-query headroom (+0.0395 pooled). The **tested** inference features + **tested**
training objectives did not capture it (impossibility is NOT claimed): the one honest heuristic tried
(raise relation weight when the relation signal fires) *hurts* by −0.0225, because
`relation_any_signal_present ≠ relation_gold_present` (the signal fires on candidates that are structurally
connected but not gold). Oracle archetype choice: equal is best for **3418/6985 (48.9%)** of queries; the
rest split up_offmix 1235 / up_splade 1033 / up_relation 746 / up_dense 339 / down_relation 214 — no single
non-equal archetype dominates. Whether a richer inference-feature set / objective-aligned training can
recover this headroom is exactly what phase C6 tests — it is an open question, not a settled negative.

## 5. Relation gating learned?  NO

Mean relation gate weight of the **selected** controller, split by the analysis-only label
`relation_gold_present` and by the observable `relation_any_signal_present`:

| split | relation weight |
|-------|:---:|
| relation_gold_present = True | 0.200 |
| relation_gold_present = False | 0.200 |
| any_signal = True | 0.200 |
| any_signal = False | 0.200 |

The selected model is uniform (0.2 each), so it does not differentiate — and the *trained* gates that did
differentiate moved relation the wrong way (down, chasing MRR), which is why they were not selected. Under
the **tested** features + objective, the controller did not learn *when* relation is useful: the strongest
discriminating feature (`relation_gold_present`) is forbidden (gold-derived) and the observable proxy tried
was non-discriminative (§4). Whether a richer feature set makes it learnable is open (C6, Task 4).

## 6. Failure analysis

With the selected controller equal to C0, **0 / 6985** queries are worse than C0 (by construction of
init-inclusive selection). The informative failure population is the *trained* gate: on the hard-negative
TRAIN subset the RankNet optimum is a top-precision (MRR) solution, which is **not** the full-scope
NDCG@50 optimum — a train/eval objective mismatch. Cause of the collapse (from the trajectory + Shapley):
the gate suppresses the deep-recall specialists (relation, offset) that carry the ALL@K / NDCG@50 gains,
trading them for a small MRR bump. Shapley-KL and expert-dropout each counteract this suppression and
recover most of the loss, but only up to the equal-fusion point.

---

## Deliverable gates

| Gate | Answer | Evidence |
|------|--------|----------|
| GLOBAL_CALIBRATION_ADDS_VALUE | **NO** | C1 trained-best 0.8341 < 0.8400; equal is the global NDCG@50 optimum. |
| QUERY_ONLY_GATE_ADDS_VALUE | **NO** | C2 collapses to 0.7884, selects init. |
| CANDIDATE_GATE_ADDS_VALUE | **NO** | C3 ≈ C2 (0.7815); candidate conditioning adds nothing over query-only. |
| SHAPLEY_AUXILIARY_ADDS_VALUE | **MIXED** | Repairs the collapse (0.781→0.827) but does not beat equal; helpful as a *regularizer toward uniform*, not as a source of gain. |
| EXPERT_DROPOUT_NEEDED | **TESTED — best repair, still no gain** | Justified by observed collapse; C5 reaches 0.8383 (closest to 0.8400) but never exceeds it. |
| RELATION_GATING_LEARNED | **NO** | Selected relation weight 0.200 regardless of signal; proxy feature non-discriminative. |
| MULTI_GOLD_RECOVERY_IMPROVES | **NO** | ALL@10/ALL@50 not improved over C0 by any selected controller. |
| TOP_PRECISION_PRESERVED | **YES (but not a win)** | Trained gates *raise* MRR to ~0.931, but at deep-recall cost; selected model keeps C0's MRR 0.9064. |
| FIXED_EQUAL_FUSION_BEATEN | **NO** | All five learned rows select the equal-fusion init. |
| QUERY_ONLY_VS_CANDIDATE | **TIE** | C3 ≈ C2; prefer the simpler query-only gate — candidate conditioning is not justified. |

## Selected model

**L2_CONTROLLER_CANDIDATE = C0 (fixed 5-expert equal RRF fusion).** Selected on VAL only; no learned
controller beats it. TEST not inspected.

## Interpretation & next phase (C6)

The negative result is **regime-specific — `CURRENT_CONTROLLER_TRAINING_BEATS_EQUAL_RRF = NO`, not a claim
that adaptivity is impossible.** The hypothesized binding constraint is a **train/eval objective mismatch**:
training ranks a ~130-candidate hard-negative subset (RankNet → top-precision/MRR surrogate) while selection
and deployment score the full P50 scope (NDCG@50 → deep-recall). The gate optimizes the training objective
faithfully and lands on a solution that is worse on the deployment objective. The oracle (+0.0395) shows
real per-query headroom that the tested features + tested objectives did not capture.

The follow-on phase — **C6 (full-scope-aware controller training)** — isolates whether the failure is
sampling/objective mismatch: depth-aware negatives (Task 2) + NDCG@50-aware LambdaRank loss (Task 6) on the
simpler **query-only** gate, with a TRAIN-oracle predictability diagnostic (Tasks 3–4) to test whether the
oracle regime is predictable from inference features at all. See `L2_CONTROLLER_OBJECTIVE_ALIGNMENT.md`.

**STOP (this phase).** TEST not inspected; datasets not extended; L3 not started; L1 unchanged; no regime
classifier trained (beyond the C6 diagnostic); no bounded expansion run. C1–C5 artifacts preserved as
completed negative ablations (not overwritten).
