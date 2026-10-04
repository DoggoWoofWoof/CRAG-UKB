# G2 · B1.3 — Graph-only admission bridge (diagnostic-only)

**Question.** Can the GRAPH_ONLY discoveries (recovered golds absent from *both* Dense and SPLADE top-200 — the 89 %
of MetaQA recovered golds that S0 admitted 0 % of) be admitted through a small **parameter-free structural reserve**,
with **no learned directional L2**? Diagnostic only: no training, no rebuild, no encoder passes, no TEST. Gold labels
evaluate the rankings; they never build any score/ranking/reserve. Reuses B1.2 artifacts. JSON:
`results/GENERALIZATION/_g2_b13_bridge.json`.

---

## Population (recovered golds → visible vs graph-only)

| target | recovered | RETRIEVAL_VISIBLE | **GRAPH_ONLY** | graph-only frac | discovery-pool non-gold cands |
|---|---|---|---|---|---|
| MetaQA | 3260 | 354 | **2906** | **0.891** | 470 925 |
| 2Wiki | 402 | 86 | **316** | **0.786** | 371 014 |
| SQuAD (control) | 4 | 4 | **0** | 0.000 | 187 637 |

SQuAD has **no** graph-only population — exactly the negative control the directive predicted.

---

## Graph-only rank diagnostic — where do graph-only golds sit in the discovery pool?

Rank of each graph-only gold **among its query's geometry-added candidates**, by each parameter-free signal. `R@k` =
fraction with within-additions rank `< k` (diagnostic ceiling). **The concentration is sharply regime-split:**

**2Wiki — structural signals concentrate golds almost surgically:**

| signal | R@16 | R@32 | median rank |
|---|---|---|---|
| min_exp_hop_norm | **0.987** | 0.994 | **3** |
| ppr_share_pct | **0.987** | 0.987 | **2** |
| struct_vs_retrieval_adv | 0.889 | 0.946 | 2 |
| n_struct_anchors_pct | 0.788 | 0.845 | 3 |
| s_dir_pct (directional) | 0.427 | 0.582 | 21 |

**MetaQA — irreducibly diffuse under *every* signal:**

| signal | R@16 | R@32 | median rank |
|---|---|---|---|
| struct_vs_retrieval_adv | 0.229 | 0.369 | 53 |
| seed_support_frac | 0.225 | 0.372 | 51 |
| s_dir_pct (directional) | 0.210 | 0.338 | 58 |
| n_struct_anchors_pct | 0.210 | 0.340 | 61 |
| min_exp_hop_norm | 0.172 | 0.272 | 72 |

No parameter-free orderer puts MetaQA's graph-only golds' median below rank ~51, or gets R@16 above 0.23. On 2Wiki the
*best* structural signals reach median rank 2–3. **The discovery signal that concentrates 2Wiki (min-exp-hop, PPR) is
the *worst* on MetaQA** — and its gold/non-gold ordering is *inverted* there (AUC 0.20).

---

## Gold-vs-negative separability (graph-only gold vs graph-only non-gold), cross-dataset consistency

AUC (gold ranked above non-gold); **only the directional signals separate gold-up CONSISTENTLY on both**:

| signal | MetaQA AUC | 2Wiki AUC | both gold-up? |
|---|---|---|---|
| **dir_margin** | 0.688 | 0.693 | **yes** |
| **s_dir_pct** | 0.683 | 0.704 | **yes** |
| **dir_support_max_pct** | 0.682 | 0.705 | **yes** |
| struct_vs_retrieval_adv | 0.633 | **0.968** | yes (asymmetric) |
| n_struct_anchors_pct | 0.629 | 0.909 | yes (asymmetric) |
| ppr_share_pct | 0.512 | 0.985 | borderline |
| min_exp_hop_norm | **0.201** | 0.990 | **no (inverted on MetaQA)** |
| seed_support_frac | 0.474 | 0.240 | **no (gold lower on both)** |

**The only cross-dataset-consistent orderer is the directional discovery score** (`dir_margin` / `s_dir` /
`dir_support_max`, AUC ≈ 0.68–0.70 on both) — candidate-local, query-relative, parameter-free, but **moderate**. Every
strongly-separating structural signal (PPR, hop, anchors, struct-adv) is 2Wiki-specific and near-chance or inverted on
MetaQA.

---

## Fixed-reserve ceiling — TOP50 = top(50−R) BASE fused-RRF ∪ top-R parameter-free directional discovery

`R` uniform across all datasets (no per-target tuning); discovery ordering = the cross-dataset-consistent `s_dir`.
`GO_ADM` = graph-only golds admitted; `RV_RET` = retrieval-visible-gold retention; NET / R@50 / ALL@50 / churn.

| target | R | GO_ADM / avail | RV_RET | NET | R@50 | ALL@50 | churn |
|---|---|---|---|---|---|---|---|
| MetaQA | 0 | 0 / 2906 | 0.782 | 0 | 0.280 | 0.493 | 0.00 |
| MetaQA | 4 | 193 | 0.800 | **+223** | 0.329 | 0.524 | 0.08 |
| MetaQA | 8 | 339 | 0.806 | **+378** | 0.363 | 0.534 | 0.16 |
| MetaQA | 16 | 611 | 0.796 | **+635** | 0.420 | **0.543** | 0.32 |
| MetaQA | 32 | 982 | 0.695 | **+841** | 0.466 | 0.503 | 0.64 |
| 2Wiki | 0 | 0 / 316 | 0.976 | 0 | 0.896 | 0.818 | 0.00 |
| 2Wiki | 4 | 72 | 0.979 | **+84** | 0.918 | 0.854 | 0.07 |
| 2Wiki | 8 | 99 | 0.981 | **+116** | 0.926 | 0.867 | 0.14 |
| 2Wiki | 16 | 135 | 0.976 | **+135** | 0.931 | **0.875** | 0.28 |
| 2Wiki | 32 | 184 | 0.961 | +131 | 0.930 | 0.873 | 0.56 |
| **SQuAD** | 0 | 0 / 0 | 0.996 | 0 | 0.996 | 0.996 | 0.00 |
| **SQuAD** | 4 | 0 | 0.996 | **0** | 0.996 | 0.996 | 0.07 |
| **SQuAD** | 8 | 0 | 0.996 | **0** | 0.996 | 0.996 | 0.14 |
| **SQuAD** | 16 | 0 | 0.996 | **0** | 0.996 | 0.996 | 0.27 |
| SQuAD | 32 | 0 | 0.975 | **−21** | 0.975 | 0.975 | 0.54 |

**Reading:**
- A small fixed structural reserve **nets positive on both structured targets** (MetaQA +378 at R=8, +635 at R=16;
  2Wiki +116 at R=8, +135 at R=16) — admitting graph-only golds that **no learned retrieval/verification channel could
  reach** (S0/S1'/S2' all admitted 0 % of them). This is the bridge working.
- **SQuAD (negative control) pays exactly 0 through R=16** and only −21 at R=32: reserving structural slots costs
  nothing when the graph adds nothing, until the reserve grows large enough to displace retrieval-visible gold.
- The reserve is **surgical on 2Wiki** (RV retention ≈ 0.98, ALL@50 peaks R=16) but **coarse on MetaQA** — NET keeps
  climbing with R only because the diffuse discovery ordering forces you to spend churn (0.64 at R=32) and RV retention
  starts falling (0.782 → 0.695) to buy the deeper golds.

---

## Interpretation (directive's case table)

The result is a **principled blend of CASE A and CASE B**, split by regime:

- **PARAMETER_FREE_STRUCTURAL_ADMISSION_BRIDGE = PROMISING.** A small fixed structural discovery reserve yields
  **positive NET on both MetaQA and 2Wiki while preserving SQuAD** (0 cost ≤ R=16) — the CASE-A *operational* test is
  met. It admits precisely the graph-only golds that every learned verification channel rejected. The candidate
  architecture **(retrieval/S1' verification + small fixed structural discovery reserve, no directional MLP)** is
  supported.
- **L1_DISCOVERY_HIGH_RECALL_BUT_LOW_PRECISION = YES (on MetaQA).** The CASE-A *precision* condition ("concentrate
  strongly on MetaQA **and** 2Wiki") fails: MetaQA's graph-only golds are diffuse under **every** parameter-free
  orderer (median rank ~51–58, R@16 ≤ 0.23), so the reserve admits them only by spending churn. 2Wiki, by contrast,
  is high-precision under structural ordering (median rank 2–3). A **transferable graph-only *verifier*** — one that
  would keep the reserve small and precise on MetaQA too — **remains unsolved**, consistent with B1.2's S2' non-transfer.
- **Discovery ordering is regime-specific.** No single parameter-free structural signal concentrates golds on both
  (the 2Wiki champion, min-exp-hop/PPR, is inverted/near-chance on MetaQA). The **only cross-dataset-consistent**
  orderer is the directional discovery score (`dir_margin`/`s_dir`, AUC ≈ 0.69 on both) — moderate, not sharp. Per
  directive, this is **not yet a general precision solution**; we do not introduce a dataset-specific reserve ordering.

**Net:** the parameter-free reserve is a real, SQuAD-safe admission bridge for graph-only discoveries — promising as a
*discovery* channel — but MetaQA exposes it as high-recall / low-precision, and a transferable verifier is still the
open problem. Per directive (CASE B present), **do not proceed to six-way LODO.**

---

## STOP

STOP after B1.3 diagnostics and fixed-reserve ceilings, per directive. No R selected. No six-way LODO launched.
Directional geometry stays in the parameter-free L1 discovery mechanism; no directional MLP; no MLP enlargement. The
three datasets remain LODO development evidence, not external-generalization proof.
