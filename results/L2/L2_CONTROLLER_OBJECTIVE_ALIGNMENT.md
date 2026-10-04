# L2 Controller — Objective Alignment (C6 phase, VAL only)

**Question.** The C0–C5 phase established `CURRENT_CONTROLLER_TRAINING_BEATS_EQUAL_RRF = NO` (not
impossibility) with a real +0.0395 per-query oracle headroom left uncaptured. C6 isolates **whether that
failure was a TRAIN candidate-distribution / objective mismatch** — by aligning TRAIN supervision with the
full-P50 multi-gold NDCG@50 objective, on the simpler **query-only** gate (frozen 5 experts;
λ_shapley = 0; expert_dropout = 0). Not an architecture search.

**Answer.** Objective alignment **removes the collapse** but **does not beat equal fusion**. The full-scope
objective (C6c) restores deep-recall (ALL@50 0.872 vs the sampled-list collapse 0.68) and produces a
*balanced* gate instead of a single-expert collapse — yet still lands significantly below equal on NDCG@50.
`QUERY_GATE_BEATS_EQUAL_RRF = NO`; **L2 candidate remains C0 (equal RRF)**. TEST never inspected.

---

## 1. Task 1 — negative depth distribution (train/eval mismatch, CONFIRMED)

Full-scope C0 rank of every current-sampler TRAIN negative. The decisive band for NDCG@50/ALL@50 is
ranks 0–50; the sampler concentrates negatives in the trivial deep tail instead:

| rank band | 2wiki (neg share) | musique (neg share) |
|-----------|:---:|:---:|
| 0–10 | 0.045 | 0.055 |
| 11–20 | 0.038 | 0.048 |
| 21–50 | 0.093 | 0.115 |
| 51–100 | 0.116 | 0.142 |
| 101–500 | 0.163 | 0.228 |
| **>500** | **0.546** | **0.413** |

Only **17.6% (2wiki) / 21.8% (musique)** of negative mass sits in ranks 0–50; the tail (>500) dominates.
Training therefore spends most of its gradient separating golds from trivially-easy negatives, not on the
rank-20–50 boundary that decides the deployment metric. `TRAIN_EVAL_MISMATCH_CONFIRMED = YES`.

## 2. Task 3 — weight-archetype oracle ceiling (fixed archetype set, TRAIN and VAL)

Same fixed 6-archetype set as the C0–C5 ceiling (archetypes NOT created from VAL; VAL golds ceiling-only):

| split | equal NDCG@50 | oracle NDCG@50 | oracle gain |
|-------|:---:|:---:|:---:|
| TRAIN (24 445 q) | 0.9001 | 0.9308 | **+0.0307** |
| VAL (6 985 q) | 0.8401 | 0.8796 | **+0.0395** |

Headroom exists on **both** splits and the archetype-choice shape is similar (equal ≈ 59–62%, with
up_relation / up_splade / up_offmix as the live alternatives; musique's dominant specialist is up_offmix).
The adaptive regime is structurally present in TRAIN, not a VAL artifact.

## 3. Task 4 — is the oracle regime predictable from inference-safe features?

Tiny class-balanced multinomial-logistic classifier (query embedding + 5 query diagnostics + 15 expert
top-k summary stats; **no gold-derived features**), trained on TRAIN, evaluated on VAL:

| metric | value |
|--------|:---:|
| classification accuracy | 0.414 |
| macro-F1 | 0.340 |
| balanced accuracy | 0.385 |
| **retrieval NDCG@50, predicted archetype applied** | **0.8497** |
| retrieval NDCG@50, equal (C0) | 0.8401 |
| retrieval NDCG@50, oracle ceiling | 0.8796 |
| **predicted − equal** | **+0.0096** (24.4% of the oracle headroom) |

**This refutes the earlier "unlearnable" overclaim.** A simple classifier on inference-safe features
extracts +0.0096 NDCG@50 over equal — modest but positive and significant in direction — whereas the naive
relation-signal heuristic *hurt* (−0.0225). The regime is **partially predictable**; it is the soft gate
(below) that fails to monetize it, not the features that are devoid of signal.

## 4. Controlled training progression (POOLED VAL; init-inclusive selection)

Query-only gate throughout. `trained-best` = best epoch ≥ 0 (may differ from the init-inclusive selection).

| Row | TRAIN candidates | Loss | selected NDCG@50 | trained-best NDCG@50 | ALL@50 | MRR |
|-----|------------------|------|:---:|:---:|:---:|:---:|
| **C0** | — (fixed equal RRF) | — | **0.8400** | (baseline) | 0.9283 | 0.9064 |
| OLD C2 | canonical hard-neg sub | RankNet | 0.8400 | 0.7884 | ~0.70 | ~0.932 |
| **C6a** | **depth-aware** | RankNet | 0.8400 | 0.8342 @e3 | 0.7928 | 0.9410 |
| **C6b** | depth-aware | **LambdaRank (sampled list)** | 0.8400 | 0.7814 @e0 | 0.6766 | 0.9305 |
| **C6c** | **full P50 scope** | **LambdaRank true-rank ΔNDCG@50** | 0.8400 | 0.8287 @e7 | **0.8723** | 0.9306 |

All learned rows select the equal init. Bootstrap of each **trained** checkpoint vs C0 (paired, 2000×,
95% CI) — all significant:

| trained vs C0 | NDCG@50 Δ | ALL@10 Δ | ALL@50 Δ | MRR Δ |
|---------------|:---:|:---:|:---:|:---:|
| C6a | −0.0059 | −0.064 | −0.135 | +0.035 |
| C6b | −0.0586 | −0.188 | −0.252 | +0.024 |
| C6c | −0.0113 | −0.048 | −0.056 | +0.024 |

## 5. Mechanism — gate mean-weights of the best-trained checkpoints

| Row | dense | splade | offset | mixture | relation | regime |
|-----|:---:|:---:|:---:|:---:|:---:|--------|
| C0 (equal) | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | balanced |
| C6a | 0.01 | 0.15 | 0.00 | **0.84** | 0.00 | single-expert collapse (mixture) |
| C6b | 0.00 | 0.00 | 0.00 | **0.996** | 0.00 | hard single-expert collapse |
| C6c | 0.17 | 0.34 | 0.20 | 0.30 | **9e-5** | balanced, but **relation zeroed** |

- **Sampled-list objectives (C6a/C6b) collapse onto a single top-precision expert (mixture)** — the
  within-list ranks reward putting golds at the very top of a ~124-candidate list, which is a top-1 (MRR)
  solution. LambdaRank concentrates on the list top even harder than RankNet → C6b is the worst.
- **The full-scope objective (C6c) keeps the gate balanced** (near-equal, no collapse; ALL@50 restored to
  0.872) — objective alignment works as intended for deep-recall. **But it drives the relation weight to
  ~0** in this jointly-trained 5-expert query-only setup. Note the precise limitation (corrected — the
  earlier "query-only cannot represent candidate-local relation" was too strong): because
  `relation_mask(q,d)=0 ⇒ contribution 0`, a query-only weight `w_R(q)` **already affects only
  relation-eligible candidates**. What `w_R(q)` cannot do is **distinguish between multiple eligible
  candidates within the same query** — that requires a candidate-aware `w_R(q,d)`. Here the jointly-trained
  gate *chose* to suppress relation (weight ~9e-5), losing the flat-0.2 relation contribution equal keeps
  (**+0.019 NDCG@50 / +0.056 ALL@50**, C0 vs C0_norel). Whether this is representational or merely an
  optimization choice is exactly what C7c isolates: C7c1 (a query-only relation scalar with the 4 base
  experts frozen) tests whether a query-only `w_R(q)` *can* learn to keep relation; C7c2 tests whether
  candidate-aware `w_R(q,d)` adds value beyond it.

## 6. Failure analysis (which of the five causes)

1. **No predictable oracle regime?** NO — it *is* partially predictable (+0.0096, §3).
2. **Insufficient inference features?** Plausible contributor (classifier acc only 0.41).
3. **Optimization collapse?** Was the cause for sampled-list training (C6a/C6b collapse onto mixture);
   **FIXED by full-scope C6c**.
4. **TRAIN/VAL regime shift?** Likely contributor — TRAIN equal NDCG 0.90 ≫ VAL 0.84; the learned
   reweighting does not transfer as a net gain.
5. **Objective mismatch remaining?** **RESOLVED by C6c** (deep-recall restored); not the residual cause.
6. **PRIMARY RESIDUAL MECHANISM (newly identified; wording corrected):** the jointly-trained query-only
   gate *chose* to zero the masked-relation specialist (§5), losing a deficit on the order of equal's
   relation contribution. The open question — representational vs optimization — is isolated in C7c: a
   query-only `w_R(q)` still cannot distinguish *between* eligible candidates within a query (only a
   candidate-aware `w_R(q,d)` can), but whether that degree of freedom is actually needed is untested until
   C7c2 vs C7c1.

---

## Deliverable gates

| Gate | Answer |
|------|--------|
| TRAIN_EVAL_MISMATCH_CONFIRMED | **YES** (§1; sampled-list collapse vs full-scope repair) |
| DEPTH_AWARE_SAMPLING_HELPS | **MIXED** — lifts the RankNet collapse (0.7815→0.834 best-trained) but does not beat equal and does not fix deep-recall alone |
| NDCG_AWARE_LOSS_HELPS | **MIXED** — *hurts* on the sampled list (C6b hard-collapse 0.781); only with full-scope ranks (C6c) does it preserve deep-recall, still < equal |
| ORACLE_REGIME_PREDICTABLE_FROM_INFERENCE_FEATURES | **MIXED** — partially: classifier captures +0.0096 (24% of headroom), refuting "unlearnable"; acc only 0.41 and the soft gate does not monetize it |
| QUERY_GATE_BEATS_EQUAL_RRF | **NO** — all C6 select the equal init; best-trained C6c is −0.0113 (sig) below C0 |
| ADAPTIVE_CONTROLLER_RETAINED | **NO** |

## L2 candidate

**L2_CONTROLLER_CANDIDATE = C0 (fixed 5-expert equal RRF fusion)** — unchanged. No adaptive controller
retained. Selected on VAL only; TEST not inspected.

## Recommended next experiment (deferred — out of C6 scope)

The one representational gap the query-only gate cannot close is candidate-local relation gating. If pursued
later: **candidate-aware relation gating under the full-scope (C6c-style, true-rank ΔNDCG@50) objective**, so
the gate may raise relation weight *only on `mask=1` candidates* — combining the deep-recall-preserving
objective of C6c with the candidate conditioning that the masked relation specialist requires. Every other
knob (depth sampling, NDCG-aware loss, full-scope training) was tested here and does not, on its own, beat
equal fusion.

**STOP (this phase).** No Shapley auxiliary re-enabled; no dropout re-enabled; no candidate-aware gate
trained; TEST not inspected; datasets not extended; L3 not started; L1 unchanged. C1–C5 artifacts preserved.
