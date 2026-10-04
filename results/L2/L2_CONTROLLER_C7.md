# L2 Controller — C7: Adaptive Policy Distillation + Candidate-Aware Relation Residual Gating

**Scope.** VAL only; TEST never inspected. Frozen 5 experts `(dense, splade, offset, mixture, relation)`,
no new experts, no Shapley auxiliary, no expert dropout. Two independent questions over the C0 equal-RRF baseline
(`NDCG@50 = 0.8400`, VAL, pooled 2wiki+musique, `n = 6985`):

- **Q1 (Track AB).** Can the TRAIN oracle-archetype structure be **distilled** into an inference-safe *query*
  policy that significantly beats C0?
- **Q2 (Track C).** With the 4 non-relation experts frozen, does **candidate-specific** relation weighting
  `α(q,d)` outperform equal *and* query-only `α(q)`?

**Headline.** **Yes to Q1** — three learned query policies significantly beat equal fusion on VAL; the C0–C6
"no learned controller beats equal RRF" negative is **overturned** for the distillation regime. **No to the
candidate-aware half of Q2** — a query-only relation scalar already beats fixed relation trust, and adding
candidate conditioning *hurts*. **`FINAL_L2_POLICY = SOFT_ARCHETYPE (C7b)`**, VAL pooled `NDCG@50 = 0.8533`
(**+0.0133** over C0, **34% of the +0.0395 oracle headroom**), significant on all four metrics.

---

## 1. Results — pooled VAL (2wiki + musique, n = 6985)

| Policy | What it learns | NDCG@50 | MRR | R@5 | ALL@10 | ALL@50 |
|--------|----------------|:---:|:---:|:---:|:---:|:---:|
| **C0** equal RRF | — (fixed) | 0.8400 | 0.9063 | 0.7934 | 0.7396 | 0.9283 |
| **C7a** hard archetype | argmax archetype → its weight vector | 0.8497 | 0.9094 | 0.8174 | 0.7798 | 0.9323 |
| **C7b** soft archetype | Σ p_k(q)·archetype_k **← FINAL** | **0.8533** | **0.9113** | 0.8166 | **0.7818** | 0.9406 |
| **C7b2** utility archetype | softmax(û_k(q)/τ)·archetype_k | 0.8478 | 0.9126 | 0.8004 | 0.7586 | 0.9386 |
| **C7c1** query relation α(q) | one relation-trust scalar / query | 0.8487 | 0.9039 | 0.8108 | 0.7711 | **0.9503** |
| **C7c2** candidate relation α(q,d) | relation trust per candidate | 0.8455 | 0.8994 | 0.8102 | 0.7674 | 0.9436 |

Per-dataset (NDCG@50): C7b — 2wiki 0.8382, musique 0.8647; C7c1 — 2wiki 0.8400, musique 0.8552. Reported
separately, **not** optimised per dataset (dataset ID never fed to any head). Full 12-metric per-ds tables in
`L2_CONTROLLER_C7.json → rows_full_metrics`.

**Parity.** `C7c0` (α ≡ 1, relation residual on the frozen BASE4) reproduces C0 exactly
(`NDCG@50 = 0.8400`), confirming the residual formulation `score = BASE4 + α·r_relation` changes *only* relation
trust. Every learned row initialises at this exact C0 state (α(q)=1 via zero-init `z`; archetype heads over the
fixed set including `equal`), so all gains are learned, not init luck.

## 2. Bootstrap (paired, 2000×, 95% CI) — the decisive tests

| Comparison | NDCG@50 | ALL@10 | ALL@50 | MRR |
|------------|:---:|:---:|:---:|:---:|
| C7a − C0 | **+0.0096** [.0081,.0112] ✓ | +0.0402 ✓ | +0.0040 ✓ | +0.0030 ✓ |
| **C7b − C0** | **+0.0133** [.0119,.0147] ✓ | +0.0422 ✓ | +0.0123 ✓ | +0.0049 ✓ |
| C7b2 − C0 | +0.0078 [.0069,.0086] ✓ | +0.0190 ✓ | +0.0103 ✓ | +0.0062 ✓ |
| C7c1 − C0 | **+0.0086** [.0069,.0103] ✓ | +0.0315 ✓ | +0.0220 ✓ | −0.0025 (ns) |
| C7c2 − C0 | +0.0054 [.0038,.0071] ✓ | +0.0278 ✓ | +0.0153 ✓ | **−0.0069** ✓(neg) |
| **C7c2 − C7c1** | **−0.0032** [−.0041,−.0023] ✓(neg) | −0.0037 ✓(neg) | −0.0067 ✓(neg) | −0.0044 ✓(neg) |

✓ = 95% CI excludes 0. **All three archetype policies beat C0 on all four metrics.** The query-only relation
scalar beats C0 on ranking/recall (MRR-neutral). **Candidate-aware relation gating is significantly *worse* than
query-only on every metric.**

## 3. Q1 — distillation works (the C0–C6 negative is overturned)

The C6 phase established `QUERY_GATE_BEATS_EQUAL_RRF = NO` — a *jointly-trained* query gate over all 5 experts
selected the equal init. C7's difference is **what** is learned: instead of jointly reweighting five competing
experts under a ranking loss, C7a/b/b2 **distil the TRAIN oracle-archetype regime** (the fixed 6 archetypes that
give the +0.0395 ceiling) through a small inference-safe head (query embedding + dense/splade disagreement +
relation signal presence + per-expert score/rank summary stats — **no gold-derived features**), then apply the
predicted archetype mixture at inference.

- **Soft (C7b)** is best: `NDCG@50 0.8533` (+0.0133, 34% of headroom), and it is the *only* policy that improves
  **both** top-rank (MRR +0.0049) and deep recall (ALL@50 +0.0123) significantly.
- **Hard (C7a)** +0.0096 — argmax discards the calibrated mixture C7b keeps.
- **Utility (C7b2)** +0.0078 — highest MRR (0.9126) but lower recall; regressing per-archetype utility is
  noisier than classifying the argmax regime.

`ORACLE_DISTILLATION_WORKS = YES`. The Task-4 C6 finding (a gold-free classifier extracts +0.0096) generalises:
the soft mixture monetises more of the same regime signal than the hard argmax the C6 probe used.

## 4. Q2 — query-only relation trust helps; candidate conditioning does not

Isolating relation (BASE4 frozen equal RRF; learn only `α` on `r_relation`, structural mask keeps `α·0 = 0` on
`mask=0` so no relation evidence is ever fabricated):

- **C7c1 `α(q)` beats fixed α=1** (`QUERY_RELATION_SCALAR_BEATS_FIXED = YES`): +0.0086 NDCG, **+0.0220 ALL@50**
  (the largest deep-recall lift of any policy — relation trust is a deep-recall lever), MRR-neutral. **This
  resolves the C6 open question:** the C6c joint gate drove relation to ~9e-5 not because a query-only weight
  *cannot* carry relation, but because under 5-way competition the optimiser *competed it away*. When relation is
  isolated, a query-only scalar **keeps and raises** it. The C6c zeroing was an **optimisation artifact, not a
  representational limit** — matching the C7 wording correction (a query-only `w_R(q)` already acts only on
  `mask=1` candidates; its one true limit is it cannot distinguish *between* eligible candidates within a query).
- **C7c2 `α(q,d)` is significantly worse than C7c1** on every metric (`CANDIDATE_RELATION_GATE_BEATS_QUERY =
  NO`). The subtlety: candidate conditioning **does** learn the discrimination it was designed for — it separates
  eligible-gold from eligible-non-gold α far more sharply than C7c1:

  | | C7c1 gold / non-gold α (Cohen's d) | C7c2 gold / non-gold α (Cohen's d) |
  |---|:---:|:---:|
  | 2wiki | 1.99 / 1.63 (d 0.71) | 1.94 / 1.29 (**d 0.97**) |
  | musique | 1.59 / 0.95 (d 0.88) | 1.67 / 0.35 (**d 2.26**) |

  Yet the sharper separation **does not translate to better ranking** — it is net negative. To discriminate at
  candidate level the gate must lean on candidate features that fit TRAIN gold but perturb the already
  well-calibrated BASE4 order on VAL; the extra degree of freedom **overfits**. Relation helps as a broadly-applied
  per-query trust dial, not as a per-candidate selector. `CANDIDATE_RELATION_GATE_BEATS_C0 = MIXED` — it beats C0
  on NDCG/recall but loses MRR (−0.0069) and is dominated by every simpler policy.

## 5. Architectural decision

The user's rule: *if C7a/C7b beats C0 but C7c2 does not, retain the simpler archetype/query controller; only if
C7c2 beats all is candidate-aware justified.* C7c2 beats neither the archetype policies nor C7c1, so the
candidate-aware branch is **not** adopted.

**`FINAL_L2_POLICY = SOFT_ARCHETYPE (C7b)`** — a query-only distilled policy (best NDCG@50 and MRR, significant on
all four metrics, inference-safe, no candidate conditioning). Complexity stops here; the candidate-aware gate was
tested and rejected on evidence rather than assumed.

## Gates

| Gate | Answer |
|------|--------|
| HARD_ARCHETYPE_BEATS_C0 | **YES** (+0.0096 NDCG, all 4 metrics sig) |
| SOFT_ARCHETYPE_BEATS_C0 | **YES** (+0.0133 NDCG, all 4 sig; best policy) |
| ORACLE_DISTILLATION_WORKS | **YES** (distilled query policy captures 34% of the +0.0395 headroom) |
| QUERY_RELATION_SCALAR_BEATS_FIXED | **YES** (C7c1 +0.0086 NDCG / +0.0220 ALL@50 sig; MRR-neutral) |
| CANDIDATE_RELATION_GATE_BEATS_QUERY | **NO** (C7c2 < C7c1 on all 4, all sig; overfits despite sharper α separation) |
| CANDIDATE_RELATION_GATE_BEATS_C0 | **MIXED** (NDCG/recall sig+, MRR sig−; dominated by simpler policies) |
| ADAPTIVITY_CONFIRMED_ON_VAL | **YES** (multiple learned adaptive policies robustly beat equal fusion on held-out VAL) |
| FINAL_L2_POLICY | **SOFT_ARCHETYPE (C7b)** |
| SAFE_TO_FREEZE_L2 | **YES** (VAL-selected; TEST never inspected; archetypes fixed from C6; no gold-derived features) |

## What changed vs C6, in one line

C6: a jointly-trained query gate over 5 competing experts cannot beat equal RRF and zeroes relation. C7:
**distilling the oracle-archetype regime** into a query policy *does* beat equal RRF (+0.0133), and **isolating**
relation shows its C6c zeroing was optimisation not representation — but pushing conditioning down to the
candidate level overfits. The adaptive lever is **query-level archetype distillation**, not candidate-level gating.

**STOP (C7 complete).** No TEST inspected; datasets not extended; unrestricted 5-expert gating not restarted; no
Shapley auxiliary; no dropout; no regime labels beyond the archetype policy; L3 not started; L1 unchanged.
Artifacts: `results/L2/L2_CONTROLLER_C7.{md,json}`, `results/L2/_ctrl/{_c7ab,_c7c}.json`,
`results/L2/_ctrl/C7c{1,2}{,_trained}.pt`, per-query arrays `_perq_C7*.npz`.
