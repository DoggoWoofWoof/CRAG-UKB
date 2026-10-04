# G2 · B1.4 — Source-selected two-channel admission (final 3-target pilot before six-way)

**Design.** A deterministic **two-channel** admission interface, no learned directional verifier, no dataset ID, no
new features. **CHANNEL A (verification)** = one of `{BASE_FUSED, S0, S1_PRIME}` (the already-valid source-trained
base-anchored-residual definitions from B1.2; `S1_PRIME` = B1.2 `S1new` ladder). **CHANNEL B (structural reserve)** =
geometry-added candidates ordered **only by the frozen `s_dir`** (`S2old[:,0]`; the one cross-dataset-consistent
orderer from B1.3). Pool = top-(50−R) from V ∪ up-to-R highest-`s_dir` **unique** structural discoveries; unused slots
fall back to V. Exactly 50, no gold in construction. Reserve grid **R ∈ {0,4,8,16}** (R=32 excluded per B1.3).

**Source-only selection rule (reported verbatim).** For held-out target D, SOURCE = the other two datasets. Fit each
verification channel on SOURCE (logreg + train-only μ/σ; **β** selected on SOURCE single-channel NET) and **freeze**.
Grid = 3 channels × 4 R = 12 configs. **FEASIBLE** = configs whose `RETRIEVAL_VISIBLE_GOLD_RETENTION@50 ≥ 0.95` on
**both** source datasets (the strong retention constraint: may not buy graph-only golds by destroying retrieval-visible
golds). Among FEASIBLE, maximize **mean-over-source `NET_GOLD_GAIN@50`**; tie-break → smaller R → simpler channel
(`BASE_FUSED < S0 < S1_PRIME`). D is never used to select channel / R / β / threshold / normalization.
`RET_MIN=0.95` fixed a priori. JSON: `results/GENERALIZATION/_g2_b14_twochannel.json`.

---

## Source-selected pair per fold + held-out target metrics

| target | SOURCE | **selected (β)** | R@50 | ALL@50 | ANY@50 | **GO-adm** (avail) | **RV-ret** | NEW / EVICT | **NET** | churn |
|---|---|---|---|---|---|---|---|---|---|---|
| **MetaQA** | 2Wiki+SQuAD | **BASE_FUSED · R16** | 0.420 | 0.543 | 0.772 | **0.210** (2906) | 0.897 | 765 / 130 | **+635** | 0.32 |
| **2Wiki** | MetaQA+SQuAD | **BASE_FUSED · R8** | 0.926 | 0.867 | 1.000 | **0.313** (316) | **0.995** | 135 / 19 | **+116** | 0.14 |
| **SQuAD** | MetaQA+2Wiki | **S1_PRIME · R8** (.05) | 0.976 | 0.976 | 0.976 | — (0) | 0.978 | 2 / 22 | **−20** | 0.25 |

`GO-adm` = graph-only recovered-gold admission fraction; `RV-ret` = retrieval-visible-gold retention vs BASE fused-top50;
`NET` = new−evicted golds vs BASE fused-top50; MetaQA admits **611** and 2Wiki **99** genuinely graph-only golds that
**no learned verification channel reached** (B1.2: S0/S1'/S2' admitted 0 % of graph-only).

**Channel composition (mean / median / p95 per query):**

| target | unique structural inserted | overlap w/ V | unused slots → V | frac pool structural |
|---|---|---|---|---|
| MetaQA (R16) | 15.99 / 16 / 16 | 0 / 0 / 0 | 0.02 / 0 / 0 | 0.32 |
| 2Wiki (R8) | 7.22 / 8 / 8 | 0 / 0 / 0 | **0.78 / 0 / 8** | 0.14 |
| SQuAD (R8) | 6.81 / 8 / 8 | **0.80 / 0 / 3** | **1.20 / 0 / 8** | 0.14 |

The hoped-for **natural adaptation through overlap/exhaustion is real but insufficient**: the reserve auto-returns some
slots to verification on the graph-poorer folds (2Wiki unused 0.78 p95=8; SQuAD unused 1.20 + overlap 0.80) — but SQuAD
still has 210 903 geometry-added *non-gold* candidates, so a fixed R keeps inserting ~6.8 structural **noise** items/query,
evicting 22 retrieval-visible golds to admit 2 → **NET −20**. A fixed reserve does not shrink to zero where it should.

---

## MetaQA per-hop (post-hoc; evaluation-only labels)

| hop | nq | graph-only avail | GO-adm | R@50 | NET |
|---|---|---|---|---|---|
| 1-hop | 490 | 7 | 0.571 | 0.932 | **−37** |
| 2-hop | 219 | 243 | 0.226 | 0.380 | +51 |
| 3-hop | 457 | 2656 | 0.208 | 0.273 | **+621** |

The reserve's entire MetaQA gain is **multi-hop graph-only discovery** (3-hop +621, where 2656 of the 2906 graph-only
golds live); it costs a little on easy 1-hop (−37) by displacing retrieval-visible golds. Consistent with B1.3:
MetaQA graph-only evidence is **diffuse under all currently tested inference-safe parameter-free orderers**, so the
reserve buys the deep golds at the price of some churn and 1-hop retention.

---

## Target-oracle gap (DIAGNOSTIC · DEVELOPMENT-ONLY · NOT VALID INFERENCE SELECTION)

| target | source-selected NET | **MAXNET ceiling** (config) | NET left on table | constrained-oracle (ret≥0.95) |
|---|---|---|---|---|
| MetaQA | +635 | **+635** (BASE_FUSED R16) | **0** | BASE_FUSED R8 → +378 @ ret 0.953 |
| 2Wiki | +116 | +137 (S0 R16) | 21 | S0 R16 → +137 @ ret 0.987 |
| SQuAD | −20 | **0** (BASE_FUSED R0) | 20 | BASE_FUSED R0 → 0 @ ret 1.000 |

Source-selection finds the **absolute NET ceiling** on MetaQA (gap 0) and near-ceiling on 2Wiki (gap 21). On SQuAD the
oracle is trivially **R=0** (reserve nothing → NET 0, perfect retention): the two-channel *candidate scoring* is
sound — it is only the *policy* (choosing R without seeing the target) that fails. Note MetaQA's source-selected R16
also **breaches the target retention floor** (0.897 < 0.95; the constrained oracle would cap at R8) — the same
policy-generalization symptom: R16 kept retention ≥ 0.95 on the source pair (2Wiki 0.987 / SQuAD 0.999) but MetaQA's
retrieval-visible golds sit deeper and get evicted.

---

## Verdict — PASS / FAIL

| success condition | source-selected | target-oracle |
|---|---|---|
| MetaQA: GO-adm > 0 **and** NET > 0 | ✅ (0.210 / +635) | ✅ |
| 2Wiki: NET ≥ 0 **and** RV-ret ≥ 0.95 | ✅ (+116 / 0.995) | ✅ |
| SQuAD: NET ≥ −5 **and** RV-ret ≥ 0.99 | ❌ (**−20** / 0.978) | ✅ (R=0) |
| **overall** | **FAIL** | **PASS (all three)** |

> **`RESERVE_POLICY_GENERALIZATION = NO`.** The **target-oracle** pairs satisfy every condition, but the
> **source-selected** policy does not (SQuAD). Per the directive's failure table, when oracle passes but source-selection
> fails, the open problem is **policy selection** — choosing channel/R *without target information* — **not candidate
> scoring**. The two-channel discovery bridge is sound (MetaQA +635 / 611 graph-only golds; 2Wiki +116 / 99; oracle
> recovers SQuAD perfectly at R=0); what fails to generalize is the **fixed source-selected reserve size**.

**Why it fails, precisely.** With three datasets under LODO, the SQuAD fold's SOURCE is MetaQA + 2Wiki — **both
graph-rich**, both unanimously rewarding R>0, both keeping source retention ≥ 0.95 at R≥8. Nothing observable on the
source reveals that SQuAD's **graph-only population is empty**, so a fixed source-selected R over-reserves and inserts
structural noise. The signal that would set R→0 (target has no graph-only golds) is exactly a **target property**, and a
fixed uniform R cannot represent it. This is the same structural fact B1.3 flagged (SQuAD = negative control, 0
graph-only golds) surfacing as a *policy*-selection failure rather than a scoring one.

**Per directive:** do **NOT** enlarge the MLP (the deficiency is not candidate scoring); do **NOT** launch six-way LODO.
The next problem — if pursued under a fresh directive — is a **policy that can shrink the reserve where the graph adds
nothing**, which is precisely the *adaptive per-query R / regime signal* the current directive holds out of scope
("DO NOT DO YET: no adaptive per-query R, no dataset classifier, no learned structural gate").

---

## STOP

STOP after B1.4 three-target source-selected evaluation, per directive. `TWO_CHANNEL_UNIVERSAL_ADMISSION` = **NOT YET**
(`RESERVE_POLICY_GENERALIZATION = NO`). Structural-reserve orderer stays frozen `s_dir`; no six-way LODO launched; no
directional MLP; no MLP enlargement; TEST untouched; 0 encoder passes. The three datasets remain LODO development
evidence, not external-generalization proof.
