# G2 · B1.8 — Cross-channel complementarity, and the in-domain ceiling that reframes B1

**Frozen throughout:** candidate universe (P50 ∪ parameter-free structural directional expansion, `M_MAX=256`,
admit `TOP_POOL=50`), verification channel `BASE_FUSED`, structural orderer `s_dir` (`S2old[:,0]`),
`two_channel_pool`, `R ∈ {0,4,8,16}`. **0 encoder passes.** No dataset ID in any policy input, no target
statistics, no target calibration, no hop label as input, no gold at inference, TEST untouched.

Scripts `scratchpad/_b18_complementarity.py` (STEP 0–5), `_b18b_verdict.py` (controls), `_b18c_ceiling.py`
(in-domain ceiling), `_b18d_decompose.py` (headroom decomposition). JSON `_g2_b18_complementarity.json`,
`_g2_b18b_verdict.json`, `_g2_b18c_ceiling.json`, `_g2_b18d_decompose.json`.

---

## Headline

The complementarity representation **did not work**, and the reason is not the one B1.4–B1.7 assumed. A control
that the whole B1 program had never run — training the reserve policy **on the same dataset it is evaluated on** —
shows that **70–82% of all available structural-reserve utility is carried by a single per-domain scalar**, and that
a learned query-conditional policy captures *less* than that scalar even when it is allowed to cheat on the domain.
The reserve decision is **not a query-level decision**. Every B1.x failure has been an attempt to solve a
domain-level problem with a query-level model.

---

## STEP 0 — Channel inventory: one family is missing, and the verification channel is censored

| family | signal | status | per-query modal-tie share |
|---|---|---|---|
| **SEMANTIC** | `dense_pct` | **AVAILABLE** | 0.003 |
| | `dense_splade_agree` | AVAILABLE | 0.015 |
| | `fused_rrf_pct` | **DEGENERATE** | **0.783** |
| | `retriever_support_frac` | DEGENERATE | 0.818 |
| | `splade_pct` | **DEGENERATE** | **0.858** |
| **RELATIONAL** | offset / mixture / relation | **MISSING over the expanded universe** | — |
| **STRUCTURAL** | `s_dir_pct`, `dir_margin`, `dir_support_max_pct`, `dir_exp_rank_pct`, `degree_local_ratio` | AVAILABLE | 0.12–0.20 |
| | `ppr_share_pct`, `seed_support_frac`, `struct_dist_pct`, `min_exp_hop_norm`, `n_struct_anchors_pct` | DEGENERATE | 0.69–0.85 |

Modal-tie share = fraction of a query's candidates sitting on the single most common value; 1.0 is fully
uninformative. Two consequences:

- **The system's own verification channel is censored.** `splade_pct` and `fused_rrf_pct` derive from ranks pinned
  at the `TOP200=200` sentinel, so ~78–86% of each query's universe collapses to one tied value. `BASE_FUSED`
  cannot discriminate among the large majority of the candidates it is asked to admit. Only `dense_pct` — a
  percentile of the *exact* cosine `Xn[universe] @ q̂`, computed over the whole expanded universe — is dense.
- **RELATIONAL is absent, not weak.** The `_b12` builder emits S0 (semantic) and S1/S2 (structural) only; no
  offset/mixture/relation scores exist over the expanded universe. Per the directive this is handled by
  **deterministic abstention** — φ_relational is *not identifiable here* and is reported as such, not as zero
  evidence of irrelevance. All channels that are present cover the identical candidate universe.

## STEP 1 — Why JS / KL / entropy are ill-posed on these artifacts (demonstrated, not asserted)

The cache is **percentile-normalized by design**: order-preserving, scale-destroying. It stores no raw channel
scores and no candidate row IDs. Both natural constructions of `p_sem(v|q)`, `p_struct(v|q)` therefore carry no
query-level evidence, which I measured rather than assumed:

| construction | mean H | **std of H across queries** |
|---|---|---|
| percentile column as a mass function | 5.19–5.49 | 0.15–0.61 (tracks universe size, σ=75 on mean 271) |
| rank-softmax (τ=10, as STEP 1 proposes) | 3.298–3.303 | **0.00016 – 0.013** |

A rank-softmax's entropy is a deterministic function of universe size alone — across 1,998 MetaQA queries its
standard deviation is **1.6 × 10⁻⁴**. Entropy, top-k probability mass, top1−top2 margin, JS and KL over these
distributions are all scale-dependent quantities computed on objects whose scale has been destroyed. Under the
directive's AUTONOMY clause I therefore substituted **scale-free, order-based** complementarity features
(top-k overlap, Rank-Biased Overlap p=0.9, structural mass outside the semantic top-50, structural concentration
via `dir_margin`, expansion volume) and kept the rank-softmax **JS as an explicit null control**.

## STEP 2/4 — The complementarity features are an expansion-volume artifact

The null control settles it. `js_rank_null` carries no query evidence by construction, yet:

| dataset | \|corr(null, φ_struct)\| | best \|corr(real feature, φ_struct)\| | real beats null? |
|---|---|---|---|
| MetaQA | 0.0035 | 0.1088 | yes (but tiny) |
| 2Wiki | 0.1279 | 0.2021 | yes |
| SQuAD | 0.3499 | **0.3505** | **NO** |

And the features are not an independent axis — they are near-perfect monotone transforms of expansion volume:

| `corr(feature, added_frac)` | MetaQA | 2Wiki | SQuAD |
|---|---|---|---|
| `ovl10` / `ovl50` / `rbo` | −0.43 | −0.65 | **−1.00** |
| `str_out_sem50` | +0.43 | +0.65 | **+1.00** |
| `sem_core10` | **−1.00** | **−1.00** | **−1.00** |
| `sem_blind_frac` | +0.81 | +0.99 | +0.98 |
| `js_rank_null` | +0.79 | +0.95 | **+1.00** |

On SQuAD every "complementarity" feature *is* `added_frac`. What STEP 4 measured as cross-channel complementarity
is how much geometry expansion happened — an old B1.x quantity already inside `P0` (`sn_*`), not a new axis. The
directive's distinction between *confident complementarity* and *diffuse disagreement* is not resolvable on these
artifacts, because neither is measurable on them.

## STEP 3 — Exact TRAIN channel Shapley: the structural channel's value **flips sign** across domains

Two identifiable players (S = semantic, G = structural), **budget fixed at 50 for every coalition**,
`U(∅) = 50·n_gold/n_universe` (expected golds under uniform admission), `U(S,G)` = RRF(K=60) of the two orderings.
TRAIN relevance labels only; never computed at inference.

| dataset | U(∅) | U(S) | U(G) | U(S,G) | φ_semantic | **φ_structural** | **U(SG)−U(S)** | frac φ_G>0 |
|---|---|---|---|---|---|---|---|---|
| **MetaQA** | 0.652 | 1.087 | **1.359** | 1.835 | +0.455 | **+0.728** | **+0.748** | 0.355 |
| **2Wiki** | 0.617 | 1.737 | 0.473 | 1.810 | +1.229 | **−0.035** | +0.073 | 0.221 |
| **SQuAD** | 0.293 | 0.996 | 0.150 | 0.963 | +0.758 | **−0.087** | **−0.033** | 0.001 |

On MetaQA the structural channel **alone outranks the semantic channel alone** (1.359 vs 1.087) and contributes
more Shapley value than semantics. On 2Wiki — also multi-hop, also graph-rich — structural Shapley is ≈0. On SQuAD
adding structure is **net negative**. This is the B1 instability at the *label* level, under the new representation:
the quantity we are trying to predict does not merely change magnitude across domains, it **changes sign**, and it
does so even between the two graph-rich domains. It also independently confirms B1.3's finding that MetaQA's
structural signal is diffuse while 2Wiki's is surgical.

## STEP 5 + control A — B1.7's learned model was worse than using no query information at all

STEP 5's binned calibration appeared to improve on B1.7. The control shows the improvement was entirely
regularization. A **constant source-pooled-mean predictor**, which uses zero query information:

| target | CONSTANT predictor | B1.7 `P0` linear | complementarity bins (D4) |
|---|---|---|---|
| MetaQA | 0.2728 | 0.2928 | 0.081–0.093 |
| 2Wiki | 0.1443 | 0.1795 | 0.024–0.039 |
| SQuAD | **0.1926** | **0.7180** | 0.068–0.081 |
| **TOTAL** | **0.6097** | **1.1903** | — |

(sum of \|predicted − true\| marginal utility over the three blocks; complementarity column is D4-only, whose
constant baseline is 0.193 — matched to within noise by every complementarity feature, 0.180–0.197.)

**B1.7's 17-feature regressor was ~2× worse than predicting a constant**, and 3.7× worse on SQuAD. The
complementarity features exactly reproduce the constant baseline and add nothing.

## STEP 6 — The linear complementarity policy, run end-to-end

Same B1.7 machinery (three ridge marginal-gain regressors, no class balancing, conservative `R=0` default,
τ selected on source TRAIN only), new features, true 3-fold LODO:

| features | target | NET | mean R | waste | frac R0 |
|---|---|---|---|---|---|
| COMP | MetaQA | **+3** / 740 | 0.12 | 0.11 | 0.984 |
| COMP | 2Wiki | **+94** / 179 | 11.64 | 11.16 | 0.144 |
| COMP | SQuAD | 0 | **10.02** | 10.01 | 0.159 |
| P0+COMP | MetaQA | +3 | 0.15 | 0.14 | 0.978 |
| P0+COMP | 2Wiki | +79 | 8.74 | 8.37 | 0.320 |
| P0+COMP | SQuAD | 0 | 9.68 | 9.66 | 0.354 |

*(reference: B1.5 MetaQA +51 / 2Wiki +69 / SQuAD mean R 8.72; B1.7 +3 / +73 / 10.12)*

2Wiki improves (+94, best so far). **Both required sides fail:** MetaQA capture stays at +3 (0.4% of oracle, vs
B1.5's +51) and SQuAD reserve waste stays at ~10 — worse than B1.5's 8.72. And LODO φ_structural prediction has the
**wrong sign on all three folds** (MetaQA pred −0.096 vs true +0.425; 2Wiki +0.377 vs −0.035; SQuAD +0.256 vs
−0.086; target R² −0.06, −2.37, −15.4). Identity diagnostic: dataset predictability P0 0.688 → COMP 0.612 →
P0+COMP 0.714 (chance 0.333) — marginally *less* dataset-identifying, but `P(structural utility | representation)`
did not become more stable, which is the criterion that matters.

---

## The decisive control — the in-domain ceiling

Every B1.x post-mortem attributed failure to cross-domain transfer (source coverage, conditional shift). That
attribution was never checked against its own upper bound. So: **train the policy on the same dataset it is
evaluated on** (50/50 query split), handing it the dataset identity, its label regime and its feature distribution
for free. This cheats deliberately and is not deployable — it is the ceiling of *any* source-regime expansion,
because no amount of extra source data can beat training on the target itself.

| dataset | oracle NET (eval half) | learned in-domain NET | **capture** | in-domain R²(D4) | mean R |
|---|---|---|---|---|---|
| MetaQA | 355 | 260 | **73.2%** | **0.0035** | 12.98 |
| 2Wiki | 75 | 29 | 38.7% | 0.0041 | 2.17 |
| SQuAD | 1 | 0 | 0.0% | n/a (D≡0) | **0.00** |

Against cross-domain LODO — MetaQA **0.4%** capture (3/740), SQuAD mean R **10.02** — the in-domain policy captures
**73.2%** on MetaQA and correctly spends **nothing** on SQuAD. The representation is not the problem.

But note the dissociation: **in-domain R²(D4) is ≈0.003 while capture is 73%.** The policy predicts essentially
nothing per query, yet performs well. It is not discriminating queries — it is finding an operating point.

## What the headroom actually decomposes into

| dataset | NET by constant R = 0 / 4 / 8 / 16 | best constant R | NET(const) | NET(learned) | NET(oracle) | **learned − const** | query share of headroom |
|---|---|---|---|---|---|---|---|
| MetaQA | 0 / 106 / 174 / **290** | 16 | **290** | 260 | 355 | **−30** | **18.3%** |
| 2Wiki | 0 / 31 / 44 / **52** | 16 | **52** | 29 | 75 | **−23** | 30.7% |
| SQuAD | **0** / 0 / 0 / 1 | 0 | 0 | 0 | 1 | 0 | (headroom = 1 gold) |

**A single per-domain constant beats the learned query-conditional policy on both datasets that have headroom** —
by +30 on MetaQA and +23 on 2Wiki — while training in-domain with every advantage. Query-conditional modelling is
not merely unnecessary here; it is actively harmful relative to the scalar. **69–82% of all available structural
reserve utility is a per-domain constant**; the query-level residual is 18–31%, and no method tested across
B1.4–B1.8 has captured any of it.

(The best constant is itself chosen using target labels, so it is a diagnostic decomposition, not a deployable
policy. Note also MetaQA's curve is still rising at R=16 — the `R` grid, not the method, caps MetaQA.)

---

## What this means

The B1 admission problem has been mis-specified, and the mis-specification explains every prior result:

- **B1.4** (fixed reserve) failed because *one global* constant cannot serve both regimes — MetaQA wants R=16,
  SQuAD wants R=0. Correct diagnosis, but the conclusion drawn was "make it query-adaptive."
- **B1.5–B1.7** then tried query-adaptive models of increasing sophistication. All of them were solving a
  domain-level problem with a query-level model. Their outputs tracked the source pool's average regime because
  **that is the only thing in the labels** — the per-query component is 18–31% of a signal no model has ever
  reached, and the domain component is what actually moves the metric.
- **B1.7's "conditional shift + coverage" diagnosis is now superseded.** It is true but shallow: `P(D | P0)` differs
  across domains because `D` is dominated by a domain constant that `P0` cannot observe — not because query
  features are insufficient (they reach 73% in-domain) and not because the source *query* regimes are miscovered.

**The exact remaining bottleneck:** inferring **one scalar per corpus** — how much structural reserve this corpus
warrants — for a corpus never seen during training. That quantity is a property of the corpus and its retrieval
regime, not of any individual query. The current inference contract (**first-unseen-query, no target-wide
statistics**) forbids observing precisely the quantity that carries 70–82% of the utility. **B1 as currently
specified is therefore close to unachievable in principle**, and that is a finding about the problem statement
rather than a failure of method.

## The relational channel — answered, without forcing three-way symmetry

φ_relational is **not identifiable** on this universe: relational scores are not cached over it, and the directive
forbids fabricating them. What the surrounding evidence says, without overclaiming:

- Offset and mixture are re-parameterizations of the *same* gte-Qwen embedding space that produces `dense_pct`,
  which is a strong prior for redundancy with SEMANTIC rather than an independent axis.
- Q2 established that the Relation expert fires on **0 / 5,352** graph-only recovered golds — it abstains exactly
  on the population admission is about.

So the data support the user's own hypothesis: **SEMANTIC and STRUCTURAL are the admission channels; relational
evidence belongs to B2** (ordering/verification within an admitted pool), not to B1. Three-way symmetry is not
imposed. Measuring φ_relational properly would require emitting relational scores over the expanded universe — a
deterministic, parity-checkable rebuild with **zero encoder passes** (the embeddings are cached) — and is only
worth doing if B2 needs it.

## Verdict

| flag | value |
|---|---|
| `COMPLEMENTARITY_REPRESENTATION` | **NOT PROMISING** — features are collinear with expansion volume; do not beat a null control or a constant predictor |
| JS / KL / entropy on cached artifacts | **ILL-POSED** (scale destroyed; rank-softmax H has σ = 1.6×10⁻⁴) |
| `phi_structural` cross-domain | **SIGN-UNSTABLE** (+0.728 MetaQA / −0.035 2Wiki / −0.087 SQuAD) |
| conditional shift reduced vs B1.7? | **NO** — LODO φ prediction wrong-signed on all 3 folds; constant predictor beats B1.7 P0 2× |
| `LINEAR_COMPLEMENTARITY_POLICY` | **INSUFFICIENT** — MetaQA +3 (unchanged), SQuAD mean R 10.02 (still worse than B1.5's 8.72) |
| **B1.8 CASE** | **CASE C** — `CURRENT_SOURCE_REGIMES_DO_NOT_DEFINE_AN_INVARIANT_POLICY` |
| tiny channel attention | **NOT RUN** — CASE B's precondition ("linear direction correct") is refuted; running it would be blind capacity |
| **NEW — `RESERVE_IS_A_DOMAIN_LEVEL_SCALAR`** | **YES** — 69–82% of headroom is a per-domain constant; best constant beats the in-domain learned policy by +30 / +23 |
| **NEW — representation insufficiency (B1.7's CASE-C component)** | **REFUTED** — in-domain capture 73.2% MetaQA, mean R 0.00 SQuAD |
| **NEW — source *query*-regime coverage as the primary lever** | **REFUTED** — in-domain training is the ceiling of any expansion, and it does not need query discrimination |
| six-way LODO | **NOT launched** · B2 **NOT started** · capacity **NOT enlarged** · TEST untouched · 0 encoder passes |

## Recommended next experiment — and the scoping decision it needs first

The single high-value question is now: **can the per-corpus reserve scalar be estimated from unlabeled evidence
about a new corpus** (e.g. expansion-volume distribution, dense-score mass profile, graph density/redundancy
statistics) — quantities that involve **no relevance labels and no gold**, computed once at ingest?

This is blocked on a scoping ruling I should not make unilaterally, because it changes the inference contract:
the current constraint says **"no target-wide statistics, no target calibration."** An unlabeled index-time corpus
statistic is not target *calibration* in the leakage sense — no relevance signal is involved — but it **is** a
target-wide statistic and therefore currently prohibited. Two readings:

- **If unlabeled index-time corpus statistics are permitted:** B1 becomes tractable and probably easy — the target
  is one scalar per corpus with 70–82% of the utility behind it, and the diagnostics above already show which
  corpus properties separate MetaQA (R=16) from SQuAD (R=0).
- **If they remain prohibited:** B1's adaptive-reserve goal should be **retired**, not iterated. Under a strict
  first-unseen-query contract there is no observable that carries the dominant term, and the honest fallback is a
  fixed conservative `R` with its cost stated explicitly (SQuAD waste) — B1.4's answer, now with a principled
  reason rather than an empirical one.

Pursuing the residual 18–31% query-level headroom with more capacity is **not** justified: in-domain learned
policies capture *less* of it than a constant.

## STOP

STOP after B1.8 as directed. The tiny channel-gating experiment was **not** run — its CASE-B precondition is
refuted, and the directive forbids attention as blind capacity. Six-way LODO not launched; B2 not started; capacity
not enlarged; verification channel and `s_dir` orderer frozen; TEST untouched; 0 encoder passes. Awaiting the
contract-scoping ruling above before any further B1 work.
