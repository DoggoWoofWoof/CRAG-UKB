# G2 · B1.7 — Marginal-utility reserve policy (3-target pilot before six-way)

**What changed vs [B1.5](G2_B15_ADAPTIVE.md)/[B1.6](G2_B16_INTERACT.md): only the SUPERVISION FORM.** Everything
else is frozen from B1.4: candidate universe (P50 ∪ parameter-free structural directional expansion, `M_MAX=256`,
admit `TOP_POOL=50`), verification channel (`BASE_FUSED`), structural orderer (frozen `s_dir` = `S2old[:,0]`),
`two_channel_pool`, `R ∈ {0,4,8,16}`, the 17 query-local `P0` features. No new features, no interaction terms, no
encoder/graph/C11a/LLM, no candidate rescorer. TEST untouched; 0 encoder passes. Script
`scratchpad/_b17_marginal.py`; JSON `results/GENERALIZATION/_g2_b17_marginal.json`.

**The B1.7 hypothesis (user's entering premise).** B1.5/B1.6 posed the reserve decision as a *class-balanced 4-way
classification* of R. That framing may itself manufacture over-reservation. So B1.7 tests
`SOURCE_COVERAGE_IS_THE_ONLY_BOTTLENECK` by **removing the classification framing entirely** and replacing it with
an unbalanced marginal-utility regression that respects the natural prevalence of "spending the next block does
nothing."

---

## Method — marginal utility, unbalanced, sequential spend

**Utility (documented, target-invariant).** For every TRAIN query, using the identical deterministic
`two_channel_pool(fused, s_dir, added, R)` and TRAIN gold:

```
U(R) = #golds( two_channel_pool(fused, s_dir, added, R) ) − #golds( fused_top50 )     ; U(0) ≜ 0
```

Eviction of a retrieval-visible gold is subtracted 1:1 (it lowers the pool gold count). **Marginal utilities**
`D4 = U4−U0`, `D8 = U8−U4`, `D16 = U16−U8` — "is the NEXT structural capacity block expected to add net gold?"

**Model — three linear marginal-gain regressors** `g4,g8,g16 : P0 → D_block` (ridge, unpenalized bias, `l2=1.0`).
**No class balancing.** Dataset-balanced *query* weighting only (`w = 1/n_ds`, normalized) so one source can't
dominate by count. Natural +/0/− prevalence preserved.

**Sequential inference (conservative, R=0 default).** `R=0`; spend block *k* iff `pred D_k > τ_k`, else stop:
`R:0 → 4 (if D̂4>τ4) → 8 (if D̂8>τ8) → 16 (if D̂16>τ16)`. Thresholds selected on SOURCE TRAIN only (grid
`{.05,.10,.20,.35,.50,.75}³`, maximize realized source utility, conservative larger-τ tie-break); near-tie
`EPS=0.05` prefers the smaller reserve. **No target statistic touches any decision.** True 3-fold LODO
(MetaQA←2Wiki+SQuAD, 2Wiki←MetaQA+SQuAD, SQuAD←MetaQA+2Wiki).

---

## Result — the marginal reframing did NOT help; on 2 of 3 folds it made things WORSE

| target | source pair | NET (B17) | NET (B15) | mean R | over-reserve / waste | util-regret | GO-adm | RV-ret |
|---|---|---|---|---|---|---|---|---|
| **MetaQA** | 2Wiki + **SQuAD** | **+3** | +51 | 0.13 | 0.13 | **0.369** | 0.001 | 1.000 |
| **2Wiki** | MetaQA + SQuAD | +73 | +69 | 9.05 | 8.70 | 0.053 | 0.228 | 0.995 |
| **SQuAD** | MetaQA + 2Wiki | 0 | 0 | **10.12** | **10.11** | 0.001 | 0 | 0.999 |

- **SQuAD over-reservation got WORSE, not better.** Mean R **8.72 → 10.12** (target was *lower*), waste 10.11,
  `frac_R0` only 0.322. The user's SQuAD success criterion (lower reserve waste, lower mean R, more R=0, no relevance
  harm) is **failed** on every axis except the vacuous relevance one (RV-ret 0.999 — SQuAD golds sit at fused rank
  0–1 and can't be evicted, exactly as in B1.5).
- **MetaQA capture COLLAPSED.** NET **+51 → +3** (of target-oracle +740), utility-regret 0.369, `frac_R0` 0.978 —
  the policy now *under*-reserves the very deep multi-hop queries where graph-only golds live. 3-hop post-hoc
  NET **+1** (vs B1.5 +44).
- **2Wiki preserved** (NET +73, RV-ret 0.995) — the one fold whose source pair spans both regimes.

`CASE A` (objective/formulation was the bottleneck) is **REFUTED**: reframing supervision from class-balanced
classification to unbalanced marginal regression did not reduce SQuAD over-reservation and destroyed MetaQA
capture.

---

## Crux — the predicted marginal utility tracks the SOURCE PAIR, not the query

Read predicted vs. **true** target marginal utility against each fold's LODO source pair:

| target | source pair | pred D̄ = [D̂4,D̂8,D̂16] | **true D̄** | consequence |
|---|---|---|---|---|
| MetaQA | 2Wiki + **SQuAD (D≡0)** | [.020, .003, .002] | **[.112, .078, .129]** | predicts ≈0 → **under**-reserves (NET 3) |
| 2Wiki | MetaQA + SQuAD | [.099, .054, .094] | [.042, .016, .009] | mild over-predict → NET +73 (tolerated) |
| SQuAD | **MetaQA + 2Wiki (graph-rich)** | [.246, .148, .324] | **[0, 0, 0]** | predicts large → **over**-reserves (mean R 10.1) |

The regressor's output is essentially the **average D-regime of whatever two datasets are in the source pool**,
barely modulated by the query:

- **MetaQA's** source pair *contains SQuAD*, whose true marginal utility is **identically 0** (D>0 on only
  0.0/0.0/0.1% of SQuAD queries). SQuAD's mass of D=0 examples drags the regressor's intercept and slope toward
  predicting ≈0, so MetaQA — whose true D is large (0.11–0.13) — is told "don't spend." Under-reservation.
- **SQuAD's** source pair is *both graph-rich* (MetaQA+2Wiki, high D). SQuAD's `P0` features map into that pair's
  spend-positive region, so it is told "spend the whole budget," while its true D is exactly 0. Over-reservation.

This is the **same source-composition failure as B1.4/B1.5**, now exposed at the *label* level rather than the
decision level. The unbalanced marginal formulation did not remove it — it made it more legible.

---

## Honest CASE classification — a CONDITIONAL SHIFT, not clean feature-insufficiency

The automated verdict flagged `FEATURE INSUFFICIENCY` (CASE C) because SQuAD over-reserves **even on in-support
queries**:

| SQuAD split (kNN in P0 space) | n | reserve-waste | mean R |
|---|---|---|---|
| in-support | 562 | 6.66 | 6.66 |
| out-of-support | 438 | 14.53 | 14.57 |

That flag is **defensible but incomplete, and I am correcting it.** The support measure is kNN distance in `P0`
*feature* space — it certifies that a source query with similar features exists, **not** that the source's
`P0 → D` *label mapping* holds for SQuAD. The pred/true gap proves it does not: at SQuAD's feature coordinates the
nearby source points carry D>0 labels (from graph-rich data), which are simply *wrong* for SQuAD (D=0). So a SQuAD
query being "in-support" by feature-kNN does **not** mean the label transfers — the in-support over-reservation is
evidence of a **conditional (posterior) shift** `P(D | P0)` that differs by domain, not of features that are
insufficient given correct labels.

The two axes are therefore **entangled**, and both are real:

- **Coverage component (CASE B):** the discriminating regime — *confident retrieval AND structurally redundant ⇒
  D=0* — is **absent from the graph-rich source labels** (source structural-redundancy 0.14–0.17 vs SQuAD 0.44; same
  OOD regime B1.6 identified). OOS queries are 2× worse (waste 14.5 vs 6.66), so coverage is strongly implicated.
- **Representation component (CASE C):** `P0` carries no *domain-invariant* variable that separates SQuAD's
  redundant D=0 from the sources' productive D>0 *at the same feature location* — which is exactly why the in-support
  half still over-reserves.

**Decisive caveat that resolves the "just add sources" reflex.** CASE B's letter says "broaden source regimes using
MuSiQue/Hotpot/WebQSP." But those three are **all multi-hop graph-rich** — the *same* regime as MetaQA+2Wiki. They
would supply more D>0 examples, not the missing *confident-retrieval + redundant-structure ⇒ D=0* regime that SQuAD
needs. Adding them would help 2Wiki/MetaQA-like targets and leave SQuAD over-reservation intact. The specific
missing thing is a **retrieval-easy / single-hop-redundant source**, which none of the proposed expansions provide.

---

## Verdict

| flag | value |
|---|---|
| `CASE_A_objective_formulation_was_bottleneck` | **REFUTED** (marginal reframing did not help; worse on 2/3) |
| `MARGINAL_UTILITY_LINEAR_POLICY` | **INSUFFICIENT** (SQuAD mean R 8.72→10.12, MetaQA NET 51→3) |
| remaining-failure class | **CONDITIONAL SHIFT** `P(D∣P0)` domain-specific = entangled **SOURCE/REGIME COVERAGE (primary, actionable) + representation** |
| `SOURCE_COVERAGE_IS_THE_ONLY_BOTTLENECK` (entering premise) | **NO** — coverage is primary but not *only*; a domain-invariant `P0→D` separator is also absent |
| proposed CASE-B remedy (MuSiQue/Hotpot/WebQSP) | **would NOT fix SQuAD** — all multi-hop graph-rich, do not supply the redundant single-hop regime |
| six-way LODO | **NOT launched** · B1 **not** marked solved |
| capacity | **NOT enlarged** (per standing directive) |

**Bottom line.** B1.7's job was to falsify the "the class-balanced formulation caused the over-reservation"
hypothesis. It falsified it: the unbalanced marginal-utility reformulation did not lower SQuAD reserve waste (it
raised it) and it lost almost all of MetaQA's capture, because the linear marginal regressor predicts the **source
pair's average D-regime**, not a query-conditioned utility. What remains is a **conditional shift** — the
`P0 → marginal-utility` map is genuinely different across domains (SQuAD D≡0 where the graph-rich sources have D>0
at the same features). That is simultaneously a **source/regime-coverage** gap (the redundant-retrieval regime is
missing from the source labels) and a **representation** gap (`P0` lacks a domain-invariant separator), and it is
**not** an objective-formulation gap. Neither blind six-way (adds capacity CASE-C already argues against) nor the
proposed multi-hop-source expansion (adds the wrong regime) is justified as the next step.

## STOP

STOP after B1.7. Six-way LODO **not** launched; capacity **not** enlarged; verification channel and `s_dir` orderer
frozen; TEST untouched; 0 encoder passes. The three datasets remain LODO *development* evidence, not external-
generalization proof.
