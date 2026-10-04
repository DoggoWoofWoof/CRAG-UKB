# G2 · B1.1 — Expanded-universe admission pilot (TRUE 3-way LODO)

**Targets:** MetaQA (geometry-recovered multi-hop), 2Wiki (structural regime), SQuAD (weak-structure negative control).
**Universe:** original P50 ∪ parameter-free structural directional expansion, `M_MAX=256`, admit `TOP_POOL=50`.
**LODO:** for each target, TRAIN = the other two only; TRAIN-frozen normalization; TRAIN-selected `beta`; target VAL scored once; **TEST untouched; 0 new encoder passes.**
**Admission = base-anchored residual:** `sc = fused_rrf_pct + beta · zscore_q(learned_raw)` — protect the retrieval ranking, let the model swap in only what it is confident about. `beta` grid-selected on TRAIN to maximize mean source `NET_GOLD_GAIN@50`.
**Feature ladder:** S0 retrieval-relative (5) → S1 +structural (10) → S2 +directional-geometry (17) → small MLP on the per-target best rung.
**Inference-safety audit:** `inference_safety_audit()` **PASS** (17 contract-safe cols, no banned substrings, dim=17, `qhop` per-query-only). No dataset ID / gold / relation-type / bracket / target stats in any model feature.

Canonical result: `results/GENERALIZATION/_g2_b11_pilot_extgrid.json`. Coarse-grid (documented failure mode): `_g2_b11_pilot_coarsegrid.json`.

---

## Headline verdict

**Partial pass.** A single shared, inference-safe admission rule achieves **positive net gold gain on MetaQA (+122) and 2Wiki (+42) with negligible SQuAD damage** — *once the residual step is fine enough to express "barely perturb."* But the **winning feature set is regime-dependent** (MetaQA wants retrieval-relative S0; 2Wiki wants structural S1), no single fixed rung is simultaneously optimal, and the **directional-geometry layer (S2) is not universally safe** (destroys SQuAD, −508). The one universally non-negative choice is **S1** (retrieval+structural, *no* directional geometry). The MetaQA multi-hop recovery is driven by **retrieval-confidence reweighting, not the geometry features** — under LODO the geometry signal does not transfer and self-limits to base.

---

## The methodological finding that unlocked the pilot (beta grid)

The first full LODO used a coarse residual grid `beta ∈ {0.1 … 1.0}` with **no zero**. `select_beta` was *forced* to perturb, and even the floor (0.1) shifts a `[0,1]` percentile by ≈±0.3 — reordering ~30 % of the pool. Result: catastrophic, uniform collapse (RETAIN 0.13–0.49, `NET` −114 … −2259 everywhere). This was an **artifact of the grid, not of the method**.

Re-running with `beta ∈ {0.0, 0.02, 0.05, 0.1, 0.2, 0.35}` (0.0 = base null; finer low end) let `select_beta` find safe operating points. Every headline number below is from this corrected grid. **Lesson for B2/6-way: the residual scale must be able to select *no perturbation*, and the useful regime is `beta ≤ 0.05`.**

---

## Main results (corrected grid) — target × ladder

`RECOVERED_GOLD_ADMISSION@50` (REC_ADM) · `GOLD_RECALL@50` (R@50) · `ORIGINAL_GOLD_RETENTION@50` (RETAIN) · `NET_GOLD_GAIN@50` (NET = new − evicted) · TRAIN-selected `β`. **Bold = per-target best linear.**

### MetaQA — target, TRAIN = {2Wiki, SQuAD} · BASE R@50 = 0.280
| rung | β | REC_ADM | R@50 | RETAIN | NET |
|---|---|---|---|---|---|
| **S0-linear** | **0.05** | **0.050** | **0.307** | **0.968** | **+122** |
| S1-linear | 0.0 | 0.000 | 0.280 | 1.000 | 0 |
| S2-linear | 0.0 | 0.000 | 0.280 | 1.000 | 0 |
| MLP (S0) | 0.02 | 0.014 | 0.288 | 0.993 | +35 |

**S0 lifts R@50 by +2.7 pt and nets +122 golds.** S1/S2 self-limit to base-null: with MetaQA's geometry lesson held out of TRAIN, structural/geometry features learned on {2Wiki, SQuAD} do not help MetaQA, and TRAIN correctly picks "don't perturb."

MetaQA per-hop at the S0 operating point (the Q2-repair signal):
| hop | nq | BASE R@50 | S0 R@50 | recovered admitted | NET |
|---|---|---|---|---|---|
| 1 (dense-reachable) | 490 | 0.971 | 0.957 | 7 / 27 | −14 |
| 2 | 219 | 0.259 | **0.314** | 29 / 312 | **+23** |
| 3 | 457 | 0.077 | **0.113** | **126 / 2921** | **+113** |

→ multi-hop golds are genuinely admitted (hop-3 +113, hop-2 +23) with only minor single-hop collateral (−14). **But this is retrieval-relative confidence, not `s_dir` geometry** — the geometry rungs sit at base.

### 2Wiki — target, TRAIN = {MetaQA, SQuAD} · BASE R@50 = 0.896 (all hop-1 labeled)
| rung | β | REC_ADM | R@50 | RETAIN | NET |
|---|---|---|---|---|---|
| S0-linear | 0.05 | 0.129 | 0.885 | 0.972 | −44 |
| **S1-linear** | **0.02** | **0.124** | **0.907** | **0.998** | **+42** |
| S2-linear | 0.05 | 0.144 | 0.906 | 0.994 | +36 |
| MLP (S1) | 0.05 | 0.085 | 0.898 | 0.992 | +5 |

**Structure helps here:** S1 nets +42 and lifts R@50 +1.1 pt; S0 (retrieval-only) is *negative* (−44). This is the opposite regime to MetaQA.

### SQuAD — negative control, TRAIN = {MetaQA, 2Wiki} · BASE R@50 = 0.996
| rung | β | REC_ADM | R@50 | RETAIN | NET |
|---|---|---|---|---|---|
| S0-linear | 0.05 | 0.50 | 0.992 | 0.994 | −4 |
| **S1-linear** | **0.02** | **0.50** | **0.998** | **1.000** | **+2** |
| S2-linear | 0.2 | 1.00 | **0.480** | 0.478 | **−508** |
| MLP (S1) | 0.02 | 1.00 | 0.999 | 0.999 | +3 |

**Negative control passes at S0/S1/MLP** (NET −4 / +2 / +3; near-perfect base recall preserved). **S2 is toxic (−508):** the directional-geometry layer, with only 4 recoverable golds to chase, evicts ~half of SQuAD's original golds. Geometry is *not* universally safe.

---

## Central tradeoff — recovered-gold admission vs original-gold retention

The base can *never* admit a recovered gold (`REC_ADM_base = 0` by definition). The pilot buys recovered golds by spending retention. At the corrected operating point the trade is favorable on the two structured targets and neutral on the control:

- MetaQA S0: +283→ (net +122 after evictions), retention 0.968.
- 2Wiki S1: 74 recovered admitted, retention 0.998, net +42.
- SQuAD S1: 2 of 4 recovered admitted, retention 1.000, net +2.

The coarse grid showed the failure mode of over-spending retention (churn 0.6–0.85, retention 0.13–0.49). The signal is real but **only extractable with a near-base residual**.

---

## Universal single-rule analysis

No fixed rung is simultaneously optimal; the safe floor and the high-upside choice differ:

| universal choice | MetaQA | 2Wiki | SQuAD | verdict |
|---|---|---|---|---|
| **S1** (retr+struct) | 0 | +42 | +2 | **never negative → safe universal floor** |
| S0 (retr only) | +122 | −44 | −4 | high MetaQA upside, mild 2Wiki/SQuAD cost |
| S2 (+geometry) | 0 | +36 | −508 | **toxic on SQuAD — reject as universal** |

**S1 is the defensible universal admission feature set** (NET ≥ 0 everywhere, positive on 2Wiki). The larger MetaQA prize (+122) requires the S0 regime, which structure-heavy 2Wiki penalizes. This is precisely the *query-local regime signal* the directive anticipated: the rule that should fire is "lean on retrieval-confidence for MetaQA-like multi-hop, on structure for 2Wiki-like single-hop" — but it must be **selected per query**, and a single fixed rung cannot.

---

## Linear vs MLP verdict — **prefer linear**

The MLP (2 hidden layers, class-balanced BCE, fit on the per-target best rung, same residual admission + TRAIN-selected β) **underperforms the best linear on every target**: MetaQA +35 vs +122, 2Wiki +5 vs +42, SQuAD +3 vs +2 (≈tie). It did **not** learn the intended nonlinear regime gate ("trust structure when retrievers disagree and structural confidence is high"); it converged to a gentler, lower-yield version of the linear rule. **Do not enlarge the MLP** (per directive) — the ceiling here is *fusion/gating representation*, not model capacity.

---

## Dataset-identity leak diagnostic (analysis only — NOT a model input)

3-way classifier predicting dataset from the feature block (chance = 0.333):

| block | accuracy | dominant identity-carrying features (|w|) |
|---|---|---|
| S0 | **0.393** | ≈ chance — retrieval-relative features do not fingerprint |
| S1 | 0.572 | `struct_support` 2.55, `degree` 1.65, `min_hop` 1.30 |
| S2 | 0.625 | + `min_exp_hop` 1.43, `dir_exp_rank` 1.37, `s_dir` 1.30 |

**The structural/geometry features carry dataset-identity signal** (0.57–0.63 ≫ chance) even though dataset ID is never fed and the shared linear/MLP map cannot act as a router. This is a real carried risk: at 6-way scale a higher-capacity model *could* exploit structural-feature magnitude as an implicit dataset fingerprint. S0 is identity-clean. **Mitigation already in the contract:** per-query percentile normalization + relation-type/bracket/kNN-presence exclusion; monitor this accuracy as a gate in the full LODO.

---

## Success-condition scorecard (directive)

1. **Substantially increase MetaQA recovered-gold admission** — ✅ *partial*: +122 net, hop-3 +113 / hop-2 +23, R@50 +2.7 pt — **via retrieval-confidence, not geometry** (geometry does not transfer under LODO).
2. **Preserve/improve 2Wiki** — ✅ +42 net, R@50 +1.1 pt, retention 0.998 (needs the S1 structural rung).
3. **Avoid meaningful SQuAD degradation** — ✅ at S0/S1/MLP (−4/+2/+3); ❌ at S2 (−508).
4. **No dataset ID / target fitting / target normalization / target thresholds** — ✅ audit PASS; β and normalization TRAIN-only.

**Net:** the shared rule works, but its universal form is **S1 (no directional geometry)**, and the headline MetaQA lever is retrieval-relative confidence rather than the structural geometry the pilot was built to test. Within the tested five-expert + embedding-interaction contract, **directional structure remains the only demonstrated discriminative signal for geometry-recovered evidence *in-domain*, but it does not survive LODO transfer** — so it cannot yet be the universal lever.

---

## Recommendation (STOP point per directive)

Do **not** auto-start the six-way LODO. Before extending:
- The directive's failure-branch checklist applies: geometry's non-transfer is a **feature-availability/normalization + incompatible-graph-semantics** issue, not a capacity issue — **do not enlarge the MLP.**
- Carry **S1 as the universal-safe admission rung**; treat S0 as the MetaQA-regime high-upside variant; **drop S2 from the mainline universal rule** (SQuAD-toxic, most identity-leaking).
- The real open lever is a **query-local regime gate** that a fixed rung cannot express — the proper next experiment, not a bigger network.
- Keep `beta ≤ 0.05` and always include `beta = 0` in any residual grid.

*The three datasets are LODO development evidence, not external-generalization proof; a fresh external dataset is still required before any universal claim.*
