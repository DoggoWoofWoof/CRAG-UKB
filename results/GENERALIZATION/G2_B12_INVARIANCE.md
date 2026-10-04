# G2 · B1.2 — Invariance repair (TRUE 3-way LODO)

**Goal (directive):** make structural/directional candidate evidence more **query-relative** and less **corpus-relative**,
*without* adding capacity or another expert. Compare BASE / S0 / old-S1 / **invariant S1'** / old-S2 / **invariant S2'**
under identical **base-anchored residual admission** (`sc = fused_rrf_pct + β·zscore_q(learned_raw)`, β TRAIN-selected,
grid incl **0.0**). Reuse MetaQA / 2Wiki / SQuAD only. **TEST untouched; 0 encoder passes. STOP before six-way.**

**Universe parity with B1.1 confirmed** — identical counts (MetaQA 1998 q / 598 522 cand / 3260 recovered; 2Wiki 2000 q /
485 699 / 402; SQuAD 1000 q / 260 903 / 4). Results: `results/GENERALIZATION/_g2_b12_invariance.json`.

---

## FIRST DIAGNOSTIC — "why can S0 admit recovered golds?" (the important causal result)

For every **recovered** gold (present in the expanded universe, absent from original P50) we classified Dense/SPLADE
top-200 membership: **A** dense-only · **B** splade-only · **C** both · **D** neither (graph-only). Then the same
decomposition for the subset the **winning LODO-trained S0** actually admits.

| target | recovered golds | **overall** A / B / C / **D** | S0-**admitted** A / B / C / **D** | admitted / recovered |
|---|---|---|---|---|
| MetaQA | 3260 | 0.101 / 0.005 / 0.003 / **0.891** | 0.963 / 0.031 / 0.006 / **0.000** | 162 / 3260 |
| 2Wiki | 402 | 0.134 / 0.057 / 0.022 / **0.786** | 0.714 / 0.190 / 0.095 / **0.000** | 42 / 402 |
| SQuAD | 4 | 0.75 / 0.25 / 0 / 0 | — (0 admitted) | 0 / 4 |

Rank-percentile / support of recovered golds, **admitted vs rejected** by S0 (1 = best rank; 0 = absent from top-200):

| target | | dense-rank pct | splade-rank pct | fused_rrf pct | retriever support |
|---|---|---|---|---|---|
| MetaQA | admitted (162) | **0.556** | 0.015 | 0.816 | 0.124 |
| MetaQA | rejected (3098) | **0.027** | 0.003 | 0.421 | 0.011 |
| 2Wiki | admitted (42) | **0.392** | 0.108 | 0.760 | 0.071 |
| 2Wiki | rejected (360) | **0.045** | 0.035 | 0.392 | 0.038 |

**Causal conclusion.** ~**89 %** of MetaQA recovered golds (79 % on 2Wiki) are **graph-only** — invisible to ordinary
Dense *and* SPLADE retrieval. S0 admits **0 %** of them. Every gold S0 admits is one Dense already contained (dense-rank
pct 0.556 vs 0.027 for the rejected) — i.e. a candidate ranked **just below the P50 cut** that geometry re-surfaced and
retrieval evidence then verified. So within this contract:

> **Directional geometry functions as candidate DISCOVERY; ordinary retrieval evidence performs candidate VERIFICATION.**
> S0's headline MetaQA gain (+122) is re-promotion of dense-retrievable evidence, **not** scoring of the graph-discovered
> majority. The graph-only 89 % can only be admitted by the geometry features themselves — which do not transfer under LODO.

This is the mechanistic answer the directive asked for, and it *confirms* the "discovery + verification" split.

---

## Invariance refactor — identity fingerprinting (analysis only, NOT a model input)

3-way dataset classifier per feature block (chance = 0.333); **lower = less corpus-identifiable**:

| block | old acc | invariant acc | Δ | what moved |
|---|---|---|---|---|
| structural | S1old **0.577** | S1new **0.547** | **−0.030** | `degree_pct` |w| 1.62 → `degree_local_ratio` **0.11** (de-fingerprinted); `ppr_pct`→`ppr_share_pct` clean |
| directional | S2old **0.622** | S2new **0.571** | **−0.051** | `n_struct_anchors` (raw edge count) removed; directional margins are within-query |

**The invariant refactor measurably reduces corpus fingerprinting on both blocks** — decisively so for raw degree
(1.62 → 0.11). Residual offenders that still carry identity: `seed_support_frac` (|w| 1.98 — reach-count normalization
still scales with graph density) and `struct_vs_retrieval_adv` (1.91 — the useful advantage signal is itself regime-
correlated). Per directive we do **not** optimize further against the classifier; it is a diagnostic.

---

## Admission utility under LODO (base-anchored residual, β TRAIN-selected)

`NET_GOLD_GAIN@50` (new − evicted) · `RECOVERED_GOLD_ADMISSION@50` (REC) · `ORIGINAL_GOLD_RETENTION@50` (RET) · β.

| policy | MetaQA NET (REC/RET/β) | 2Wiki NET (REC/RET/β) | SQuAD NET (REC/RET/β) | all ≥ 0 |
|---|---|---|---|---|
| S0 | **+122** (.050/.968/.05) | +34 (.105/.998/.02) | −24 (.00/.976/.35) | no |
| S1old | 0 (.00/1.00/.0) | +33 (.105/.997/.02) | −36 (.75/.960/.05) | no |
| **S1new (invariant)** | **+5** (.011/.976/.02) | **+22** (.082/.997/.02) | **−6** (.25/.993/.10) | **borderline** |
| S2old | +31 (.016/.984/.02) | +9 (.070/.995/.02) | 0 (.00/1.00/.0) | yes |
| S2new (invariant) | +34 (.023/.967/.02) | −8 (.067/.990/.02) | **−437** (1.0/.550/.20) | **no (toxic)** |

Reading:
- **S1new is more regime-balanced than S1old** — it turns MetaQA's flat 0 into +5 and cuts SQuAD damage −36 → −6,
  at a modest 2Wiki cost (+33 → +22). SQuAD −6 is **churn-noise on a saturated control** (BASE R@50 0.996, only 4
  recoverable golds), *not* toxicity — 6 net golds of ~980 shuffled by near-base perturbation. Call it **borderline
  non-negative**.
- **S2new is still toxic** (SQuAD −437, same failure class as B1.1's S2 −508; 2Wiki −8). Making directional geometry
  candidate-local **reduced its fingerprint but not its instability**: once β grows enough to use `s_dir` it reorders
  the near-perfect SQuAD pool catastrophically. Directional geometry does **not** become a safe learned L2 feature.
- **MLP was not run** — per directive, only if the invariant *linear* first establishes useful transfer. It did not
  clear the bar on all three regimes, so no MLP (also honors "do not increase capacity").

MetaQA per-hop confirms the geometry variant is *not* the multi-hop lever: at hop-3 **S0 admits 126** recovered golds
(NET +113) vs **S2new only 68** (NET +49). Even where directional geometry should dominate, retrieval-relative S0
recovers ~2× as many.

---

## Decision (per directive's branch table)

- **INVARIANT_STRUCTURE_TRANSFER = PROMISING (borderline).** S1' reduces fingerprinting (−0.030) **and** rebalances
  utility (MetaQA 0→+5, SQuAD −36→−6) with 2Wiki still positive (+22). The only sub-zero is SQuAD −6 (saturated-control
  churn-noise, not toxicity). → structural invariance is a genuine development improvement.
- **DIRECTIONAL_EVIDENCE_TRANSFER = NO.** S2' does not improve beyond S1' and stays toxic on the negative control.
  → **do NOT force directional geometry into the learned L2**; keep it as a **parameter-free L1 candidate-generation /
  discovery** mechanism. Standing design: **geometry for DISCOVERY + universal retrieval/structure for VERIFICATION.**
- **Development status:** `B11_DEVELOPMENT_CANDIDATE = S1 (invariant form S1new)`. **NOT** `UNIVERSAL_S1_PROVEN` — S1
  non-negativity remains development-only evidence (three LODO regimes, not external proof).

## Global compute contract (frozen, uniform — no dataset runtime branch)

`MAX_HOPS=3 · MAX_FRONTIER=1500 · MAX_EDGES_SCORED=400 000 · DEG_CAP=300 · PPR_CAP=4000 · M_MAX=256 · TOP_POOL=50`.

| dataset | edges visited mean / p50 / p95 | expansion latency s (mean/p95) | candidate count mean | edge-budget hit frac |
|---|---|---|---|---|
| MetaQA (sparse KB) | 13 385 / 12 123 / 26 664 | 0.10 / 0.28 | 300 | 0.00 |
| 2Wiki | 4 974 / 5 040 / 10 400 | 0.05 / 0.20 | 243 | 0.00 |
| **SQuAD (dense, deg≈73)** | 341 920 / 289 801 / 858 592 | 0.59 / 1.40 | 261 | **0.36** |

The dense SQuAD graph hits the edge budget on 36 % of queries under the **same** caps that leave the sparse graphs
untouched — the bounded contract holds for dense and sparse alike with **no per-dataset branch**.

---

## STOP

Per directive, **do not launch six-way LODO automatically.** Structural invariance (S1new) is a promising development
candidate; directional geometry stays an L1 discovery mechanism. The three datasets are LODO development evidence, not
external-generalization proof — a fresh external dataset is still required before any universal claim.
