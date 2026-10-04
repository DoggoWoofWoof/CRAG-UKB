# L2 — Exact query-level Shapley over the 5 frozen experts

**Scope.** Exact cooperative-game Shapley for the frozen expert set
`{E1 Dense, E2 Exact-SPLADE, E3 Offset, E4 Mixture, E5 Masked-Qwen-Relation}` on
**2wiki_clean** and **musique_clean**, **TRAIN + VAL** (no TEST). Purpose: decide whether a soft
per-query controller is *justified* and what it should condition on. **Controller NOT trained. Expert
dropout NOT trained. TEST never inspected. L1 untouched. L3 not started.** Code:
`scratchpad/l2_shapley.py` (engine), `scratchpad/l2_shapley_analysis.py` (analyses). Arrays:
`results/L2/_shapley/<ds>_<split>.npz`. Machine-checked numbers: `results/L2/L2_SHAPLEY_EXACT.json`.

## Method (locked)

- **Exact enumeration:** n=5 → 2⁵=32 coalitions/query; φᵢ(q)=Σ_{S⊆N\{i}} |S|!(n−|S|−1)!/n! · [v_q(S∪{i})−v_q(S)]. No Monte-Carlo.
- **Value function v_q(S):** utility of the ranking from the **frozen masked-RRF fusion** of coalition S
  (K0=60, contribution 1/(K0+rank)). A candidate whose *total* coalition contribution is 0 is **not
  retrieved** (rank=∞). This makes **v_q(∅)=0** and **relation-only-with-no-signal=0** automatically — no
  tie-order artifact. **No coalition-size normalization.** φ kept **signed, never clamped**.
- **Relation is masked (ABSTAIN):** contributes 1/(K0+rank_among_eligible) on mask=1, **exactly 0** on
  mask=0. Rank defined only over eligible candidates.
- **Utilities (4 separate Shapley views, never combined into one scalar):** **nDCG@50 (PRIMARY, multi-positive)**,
  MRR=1/(1+best_gold_rank) (diagnostic — under-credits deep/multi-gold experts), ALL@10, ALL@50 (binary all-golds-within-K).
- **L1 handling:** query with no in-scope gold → `SHAPLEY_VALID=False`, skipped (2wiki 3+0, musique 8+2).
  Multi-gold: nDCG/ALL use all in-scope golds; MRR uses best only.

## Validity — all gates PASS

| check | result |
|---|---|
| **EFFICIENCY** (Σφᵢ = v_full, v_∅=0) on **full** data, all 4 splits × 4 utils | **PASS, max error 4.44e-16** (machine precision) |
| **DUMMY** (zero-contribution 6th expert) | φ = 0 |
| **SYMMETRY** (duplicated expert) | gap 6.9e-18 |
| **RELATION ABSTENTION** (all-mask-false) | φ_relation maxabs **exactly 0.0** — verified on every no-signal query in all 4 splits |
| **NEGATIVE** (adversarial expert) | φ<0 found, **not clamped** |
| runtime | ~3–5 ms/q pilot; full = **~91 s CPU** (2wiki 30.9+8.9 s, musique 39.6+11.3 s) |

`SHAPLEY_VALID` distribution and `SANITY_no_signal_relation_phi_maxabs = 0.0` confirm the masked/empty
semantics hold on real data, not just unit tests.

---

## A1 — Global expert value (φ mean, PRIMARY nDCG@50)

| | dense | splade | offset | mixture | relation |
|---|---|---|---|---|---|
| **2wiki train** | 0.186 | **0.205** | 0.189 | **0.217** | 0.090 |
| 2wiki val | 0.198 | **0.218** | 0.145 | 0.156 | 0.103 |
| **musique train** | 0.169 | 0.172 | **0.288** | **0.301** | **−0.021** |
| musique val | 0.185 | 0.190 | **0.244** | **0.252** | **−0.015** |

**The expert-value ordering flips across datasets** — SPLADE/Mixture lead on 2wiki; Offset/Mixture
dominate on musique; **Relation is a positive contributor on 2wiki (+0.09..0.10) but net-negative on
musique (−0.02)**. `frac_pos` for the four base experts is 0.93–1.00 everywhere; Relation is
`frac_pos≈0.54 / frac_zero≈0.26 / frac_neg≈0.19` on 2wiki and `0.12 / 0.54 / 0.34` on musique (it abstains on
most musique queries and, at equal weight, mildly hurts the rest). A single fixed weight vector cannot be
right for both datasets, and — from the per-query signs below — cannot be right for many queries *within* a
dataset.

## A2 — Query heterogeneity → **CONFIRMED**

- **No query is a one-expert query:** `frac_single_expert_dominant(≥90% mass in one expert) ≈ 0.000` on all
  splits. **96–99 %** of queries need **≥3 experts** to reach 90 % of positive Shapley mass;
  positive-mass entropy ≈ **0.81–0.87** (1.0 = uniform over 5).
- **The dominant expert is query-dependent and its distribution is dataset-dependent.** Largest-positive
  contributor share:
  - 2wiki train — mixture 31 %, splade 24 %, dense 18 %, relation 15 %, offset 12 % (genuinely spread).
  - musique train — mixture 43 %, offset 32 %, dense 16 %, splade 9 %, relation 1 %.
- **Negative contributions are common and expert-specific:** relation φ<0 on **19.5 %** (2wiki tr) / **34 %**
  (musique tr) of queries; dense/splade φ<0 only 1–9 %. → the harm of a fixed weight is concentrated in
  relation (and, per A4, offset/mixture on top-10).

## A3 — Relation is a clean **specialist** (gate: IS_RELATION_A_SPECIALIST = YES)

φ_relation conditioned on whether a relation-eligible **gold** exists (nDCG@50):

| | signal FALSE | gold_present FALSE | **gold_present TRUE** |
|---|---|---|---|
| 2wiki train | φ=0 (n=2763) | −0.039 (n=4616) | **+0.192, frac_pos 0.97** (n=5881) |
| 2wiki val | φ=0 (n=769) | −0.035 | **+0.207, frac_pos 0.98** (n=1710) |
| musique train | φ=0 (n=7531) | −0.043 (n=12024) | **+0.115, frac_pos 0.88** (n=1924) |
| musique val | φ=0 (n=2153) | −0.038 | **+0.125, frac_pos 0.88** (n=566) |

Relation is **correctly positive exactly when its bridge gold exists** — on *both* datasets — and mildly
negative (equal-weight over-trust) when it does not. The difference between datasets is *base rate*: 2wiki
has a relation gold on 56 % of queries, musique on only 14 %, so at a fixed equal weight Relation is a net
win on 2wiki and a net loss on musique. This is the textbook case for **gated, not fixed** inclusion. Sanity:
`φ_relation = 0.0` exactly on every no-signal query (SANITY maxabs = 0.0).

## A4 — Offset / Mixture: deep multi-gold specialists (2wiki), general experts (musique)

The predicted MRR-vs-deep-recall tension is **real but low-frequency** as a strict sign flip:
`frac(φ_MRR≤0 & φ_nDCG50>0)` is only 1–2.5 % (2wiki) / 0.3–1 % (musique). The clearer signature is in the
binary utilities on **2wiki-val**: offset/mixture have **high φ_ALL50 but LOW φ_ALL10** (mean 0.06 / 0.08,
`frac_neg` 0.40 / 0.38) — they recover *deep* (rank 10–50) multi-gold members but can **displace the easy
first gold out of the top-10**. On musique offset/mixture are simply the dominant experts on every utility
(φ_ALL10 mean 0.37 / 0.39, frac_pos ~0.99). → they are **multi-gold deep-recovery specialists on text
(2wiki)** and **general workhorses on musique**; a controller that could down-weight them for top-heavy
queries would recover the small top-10 cost.

## A5 — Dense/SPLADE disagreement is a **weak** conditioning signal

Bucketing by TRAIN quantiles of `1−Jaccard@20(dense,splade)`, φ (nDCG@50) shifts only mildly low→high:
relation 0.078→0.099 (2wiki), −0.032→−0.011 (musique, less negative); splade 0.194→0.212 (2wiki). Monotone
but small — disagreement is a *useful auxiliary* input, **not** a strong standalone gate.

## A6 — Expert complementarity → **three latent axes**

Correlation of φ vectors (nDCG@50), consistent across both datasets:

- **dense ~ splade = +0.71 / +0.42** (redundant semantic+lexical base).
- **offset ~ mixture = +0.64 / +0.68** (mixture ⊇ offset; redundant relational-offset pair).
- **relation anti-correlated** with dense (−0.56 / −0.24) and splade (−0.54 / −0.26); dense/splade
  anti-correlated with offset/mixture (−0.2…−0.5).

→ ~**3 orthogonal axes**: `{dense,splade}` base · `{offset,mixture}` relational-offset · `{relation}`
graph-bridge. A controller only needs to trade **between axes**, not among 5 independent knobs — this shrinks
the effective decision and argues for a small gating net.

## Hardness

`frac(low|φ| & high v_full)=0.0` everywhere — **no solved query was "free"**; every success used expert
cooperation. `frac(low|φ| & low v_full)=0.16–0.19` — a genuinely-hard tail no expert can help (front-end /
graph-reachability limited, not a controller problem). v_full mean 0.82–0.91.

## Regime discovery (TRAIN φ, k=6, primary utility) — interpretable, **not** forced

- **2wiki** separates into six readable regimes: *offset/mixture-dominant* (22 %), *splade/dense-dominant,
  relation-negative* (21 %+16 %), *balanced multi-expert, relation-positive* (18 %), *offset-mixture,
  relation≈0* (13 %), and a distinct **RELATION_RESCUE** regime (10 %, centroid relation **+0.277**, the top
  expert). So BASE_DOMINANT / LEXICAL_DOMINANT / OFFSET_RESCUE / RELATION_RESCUE / MULTI_EXPERT all appear.
- **musique** has mixture as argmax in every cluster, but the clusters split cleanly on the **secondary
  structure**: a **relation-HURT** regime (17 %, relation −0.149), a **relation-HELP** regime (8 %, relation
  +0.183), a splade-dead regime, and a dense+splade-both-dead offset/mixture-only regime (7 %). Mixture is a
  strong universal baseline; the *decision* that varies is the {dense,splade} vs {offset,mixture} balance and
  the relation sign.

Regimes are reported descriptively; **no controller/classifier was trained on them.**

---

## Controller-readiness gates (answered from distributions, not expectation)

| gate | verdict | evidence |
|---|---|---|
| `DO_QUERY_EXPERT_WEIGHTS_VARY` | **YES** | dominant-expert spread (A2); 6 regimes (both ds); per-expert φ std ≈ its mean |
| `IS_FIXED_EQUAL_FUSION_SUBOPTIMAL` | **YES** | relation net-negative on musique overall and φ<0 on 19–34 % of queries; offset/mixture φ_ALL10<0 on 38–40 % of 2wiki-val — equal weight demonstrably leaves value on the table |
| `ARE_NEGATIVE_EXPERT_CONTRIBUTIONS_COMMON` | **YES** | relation 19–34 %; offset/mixture 25–40 % on ALL@10; dense/splade rare (1–9 %) |
| `IS_RELATION_A_SPECIALIST` | **YES** | A3 gold_present conditioning; anti-correlation; 54–92 % abstain; φ=0 with no signal |
| `ARE_OFFSET_MIXTURE_MULTI_GOLD_SPECIALISTS` | **PARTIAL-YES** | deep multi-gold signature on 2wiki (high φ_ALL50 / low φ_ALL10); general dominant on musique |
| `IS_QUERY_LEVEL_GATING_JUSTIFIED` | **YES** | all of the above |
| `IS_CANDIDATE_LEVEL_GATING_POTENTIALLY_JUSTIFIED` | **YES (hypothesis)** | relation mask is per-candidate and rescues a sparse gold subset → per-candidate weight plausibly helps; **needs candidate-level Shapley to confirm** (deferred, not this phase) |

**Deliverable gates:** `EXACT_SHAPLEY_COMPLETE=YES` · `TRAIN_SHAPLEY_READY=YES` · `VAL_SHAPLEY_READY=YES`
· `QUERY_HETEROGENEITY_CONFIRMED=YES` · `FIXED_FUSION_LIMITATION_CONFIRMED=YES` ·
`SAFE_TO_START_SOFT_CONTROLLER=YES`.

---

## PROPOSED soft controller (design only — NOT trained)

A per-(query, candidate) **gating network** producing expert weights, fused by weighted RRF; **no hard
router** (no discrete regime switch — regimes are diagnostic, and A2 shows most queries mix ≥3 experts).

```
inputs (all inference-available):
  query-level:   gte-Qwen2 q-embed; relation_any_signal_present; dense_splade_disagreement(Jaccard@20);
                 offset/mixture top-k agreement
  candidate-level (per expert e in 5): rank_e, rrf_score_e = 1/(K0+rank_e); relation_mask(candidate)
gate:  small self-attention / MLP over the 5 expert tokens (rank+score+mask), conditioned on the
       query-level vector  ->  w_dense, w_splade, w_offset, w_mixture, w_relation  in R+ (softplus)
final score(d) = Σ_e w_e(q,d) · 1/(K0 + rank_e(d)),  with w_relation(q,d) ≡ 0 wherever relation_mask=0
                 (hard-abstain preserved from the frozen value function)
```

Design choices forced by the analysis: (i) weights are **per-query at minimum, per-candidate for relation**
(A3 + candidate-level hypothesis); (ii) the gate acts on **3 axes** not 5 free knobs (A6) — regularize toward
axis-tied weights; (iii) relation's hard abstain (mask=0 ⇒ w=0) is kept structurally, never learned; (iv)
**`gold_count` / `relation_gold_present` are oracle labels — inputs only, forbidden.** Optimize the primary
retrieval metric directly (nDCG@50 / ALL@50 surrogate), VAL-selected.

## RECOMMENDED Shapley auxiliary — **positive-contribution KL (low weight, ablated)**, analysis-only fallback

Ranked among the four options the user listed:

1. **positive-contribution KL** *(recommended, optional, small λ)* — match the controller's normalized
   *positive* weight distribution to the normalized positive-φ distribution per query. Injects the specialist
   priors (relation only where its gold is present; the 3-axis structure) **without** forcing magnitudes, and
   it is easy to ablate to prove it is not load-bearing.
2. **analysis-only** *(safe fallback)* — use Shapley purely to justify architecture + choose inputs; add no
   loss term. Recommended if the KL term shows no VAL gain.
3. contribution-**ranking** loss — plausible but couples training to the frozen equal-RRF ordering.
4. **signed regression onto φ** *(not recommended)* — φ is the marginal under the **frozen equal-weight**
   value function, i.e. the very fusion the controller must *improve on*; regressing onto it binds the
   controller to reproduce equal-RRF and defeats the purpose.

`RECOMMENDED_SHAPLEY_AUXILIARY = positive-KL (ablated, low-weight) → analysis-only if no VAL lift`.

## Caveats (documented)

- Shapley here is a **cooperative-game attribution under the FROZEN equal-weight masked-RRF value
  function** — **not** universal causal importance, and **not** the value under a learned controller.
- φ<0 means an expert's *average marginal under equal-RRF* hurts; a learned per-query weight is precisely the
  lever to remove that harm — but the magnitude of achievable gain is not read off φ directly.
- **Query-level only** this phase; candidate-level gating is a *supported hypothesis*, not yet measured.
- `gold_count`, `relation_gold_present` are oracle analysis labels; excluded from controller inputs.

## STOP

Exact Shapley TRAIN+VAL complete (both datasets, 4 utilities, efficiency machine-exact) · six analyses
complete · regimes discovered (descriptive) · controller architecture and Shapley-auxiliary **proposed**.
**Not done (by instruction):** controller training, expert-dropout training, TEST inspection, dataset
extension, L3, any L1 change.
