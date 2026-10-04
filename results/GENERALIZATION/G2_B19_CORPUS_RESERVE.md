# G2 · B1.9 — Corpus-level reserve generalization under label-free index-time substrate adaptation

**Frozen:** candidate universe (P50 ∪ parameter-free structural directional expansion, `M_MAX=256`, admit
`TOP_POOL=50`), verification channel `BASE_FUSED`, structural orderer `s_dir`, `two_channel_pool`,
`R ∈ {0,4,8,16}`. **0 encoder passes.** No dataset ID, no target query statistics, no target calibration, no
benchmark hop label as input, no gold at inference, **TEST untouched**.

Scripts `scratchpad/_b19_descriptors.py` (STEP 1), `_b19b_environments.py` (STEP 3/4), `_b19c_corpus_policy.py`
(STEP 2/4/5/6), `_b19d_confirm.py` (controls). JSON `_g2_b19*.json`.

## Verdict

**`CORPUS_LEVEL_RESERVE_GENERALIZATION = NO`** — and the result is stronger than a null: every label-free
corpus-configured policy is **~2× worse than a single global constant**.

| policy | total NET (6 corpora) |
|---|---|
| best label-free corpus-configured R (in-fold selection) | **300** |
| best-case post-hoc descriptor, honest LOCO | 348 |
| source-selected fixed R | 260 |
| **single GLOBAL fixed R = 16** | **672** |
| per-corpus oracle constant (invalid: uses target labels) | 792 |

Per-corpus tuning is worth only **+120 (15.1%)** over one global constant, and no label-free descriptor
captures any of it. B1.4's fixed reserve — rejected on 3 corpora — is **vindicated on 6**.

---

## STEP 1 — Label-free descriptors (all six substrates)

Computed from corpus substrate alone: documents + node embeddings + the **same** adjacency `_b12_build.py`
uses (`kb.txt` for MetaQA, `master_nodes.neighbors` otherwise — *not* `graph.pt`, which describes a different
graph). Every descriptor: `uses_corpus_only=YES · uses_target_queries=NO · uses_labels=NO · uses_dataset_id=NO`,
cacheable at ingest, available before the first unseen query.

| descriptor | MetaQA | 2Wiki | SQuAD | MuSiQue | HotpotQA | WebQSP |
|---|---|---|---|---|---|---|
| n_nodes | 40,151 | 65,865 | 19,029 | 13,672 | 507,494 | 781,485 |
| mean_degree | 5.45 | 3.83 | **73.00** | 7.54 | 13.51 | 4.19 |
| neighbor_jaccard_redundancy | **0.0003** | 0.0602 | **0.2981** | 0.0962 | 0.0208 | 0.0135 |
| StructSemOverlap@10 | **0.593** | 0.169 | 0.051 | 0.166 | 0.114 | 0.223 |
| edge_semantic_lift | **0.952** | 0.333 | 0.352 | 0.497 | 0.414 | 0.654 |
| hop2_unique_mean | 70.8 | 29.9 | 451.6 | 122.7 | 154.8 | 78.0 |

## STEP 2 — Reserve curves, now measured for all six (MuSiQue / Hotpot / WebQSP are new)

| corpus | NET/query at R = 0 / 4 / 8 / 16 | best constant R | curve shape |
|---|---|---|---|
| **MetaQA** | 0 / 0.1116 / 0.1892 / **0.3178** | **16** | monotone up |
| **2Wiki** | 0 / 0.0420 / 0.0580 / **0.0675** | **16** | monotone up |
| **SQuAD** | 0 / 0 / 0 / 0 | **0** | flat |
| **MuSiQue** | 0 / **0.0025** / −0.0045 / **−0.0260** | **4** | peak then **negative** |
| **HotpotQA** | 0 / **0.0040** / 0.0030 / **−0.0115** | **4** | peak then **negative** |
| **WebQSP** | 0 / **0.0286** / −0.0190 / **−0.0730** | **4** | peak then **negative** |

Two findings B1.x had no way to see with three corpora:

- **Three corpora go actively negative at R=16.** Prior evidence knew only monotone-up (MetaQA, 2Wiki) and
  flat (SQuAD). Over-reserving is not merely wasteful on MuSiQue/Hotpot/WebQSP — it *destroys* golds.
- **WebQSP is a KB like MetaQA but wants R=4, not 16.** "KB ⇒ high structural reserve" is refuted. This also
  contradicts the expectation carried from the L3/PPR work; the reserve operating point does not follow
  substrate kind.

## STEP 4 — The stated hypothesis is refuted, with the sign inverted

The directive's hypothesis was *high semantic-structural redundancy ⇒ low reserve utility; low overlap ⇒ high
complementarity ⇒ high utility*. The data say the opposite:

| descriptor | ρ vs NET/query | ρ vs best R | LOCO sign-stable |
|---|---|---|---|
| **StructSemOverlap@10** | **+0.886** | +0.600 | yes |
| neighbor_jaccard_redundancy | −0.829 | −0.600 | yes |
| mean_degree | −0.771 | −0.714 | yes |
| StructSemOverlap@50 | +0.714 | +0.371 | yes |
| edge_semantic_lift | +0.371 | −0.029 | no |

Overlap correlates **positively** with reserve utility (ρ = +0.886). High overlap is not redundancy — it means
the structural edges are *semantically coherent*, i.e. the edge means something. Low overlap (SQuAD: 0.051,
mean degree 73, jaccard 0.298) means the edges are **noise**: NER "shares-entity" links joining paragraphs that
merely co-mention a name. So `STRUCTURAL_NOVELTY_INDEX = 1 − overlap` is **anti**-correlated with utility;
"novelty" is the wrong frame and "edge semantic coherence" is the right one. Neither, however, transfers (below).

**MuSiQue was the decisive falsification en route.** On four corpora, `neighbor_jaccard_redundancy` inverted to
the exact true order and looked compelling. Adding WebQSP and Hotpot broke it (2Wiki lands 4th by descriptor,
2nd by truth). The 4-point "perfect match" was descriptor selection on too few points — which is why the LOCO
protocol below selects inside each fold.

## STEP 3 — Coherent sub-environments cannot be constructed here

Deterministic graph-Voronoi regions (multi-source BFS, connected, label-free; regions partition the *sample*,
not the graph, so no edge is cut). The feasibility check fails on the **utility** side:

| corpus | usable environments (≥25 queries) | NET/query mean | **std** |
|---|---|---|---|
| MetaQA | 7 | 0.365 | 0.135 |
| 2Wiki | **1** | 0.068 | — |
| SQuAD | 11 | 0.000 | **0.000** |

SQuAD's utility is identically zero (no variance to explain); 2Wiki's queries concentrate into a single region.
Only MetaQA has usable within-corpus variation. Worse, where variation exists the relationship **inverts**:
within-MetaQA vs pooled Spearman is +0.357 vs −0.665 (`log_mean_degree`), +0.286 vs −0.730 (jaccard), −0.321 vs
+0.709 (overlap@10) — 5 of 6 descriptors flip sign. The strong pooled correlations (|ρ| 0.61–0.76) are a
**Simpson's-paradox artifact of three corpus clusters**, not evidence of a within-substrate law. Per the
directive, the analysis stays descriptive; no high-capacity model was fit.

## STEP 5/6 — Leave-one-CORPUS-out, with selection inside the fold

Descriptor selection happens on training corpora only, so choosing a descriptor after seeing the full table
cannot leak. `R_D` is frozen before any target query is scored; no target label or query statistic touches it.

| target | descriptor (in-fold) | R configured | R oracle | NET cfg | NET oracle |
|---|---|---|---|---|---|
| MetaQA | degree_cv | **4** | 16 | **223** | 635 |
| 2Wiki | jaccard redundancy | **4** | 16 | **84** | 135 |
| SQuAD | StructSemOverlap@10 | 4 | 0 | 0 | 0 |
| MuSiQue | StructSemOverlap@10 | **16** | 4 | **−52** | 5 |
| HotpotQA | StructSemOverlap@10 | 4 | 4 | 8 | 8 |
| WebQSP | StructSemOverlap@10 | **16** | 4 | **−23** | 9 |

Exact-match 16.7%; recovers 30.3% of the constant-oracle. The failure is not the policy class — 1-NN 240,
linear-ordinal 300, rank-monotone 297, all ≤33% exact-match, and **all three assign R=4 to MetaQA and 2Wiki**,
forfeiting the two largest wins (635→223, 135→84).

### The one apparent positive, and why it does not survive

`hop2_unique_mean` **perfectly separates** the R=16 corpora in hindsight (70.8, 29.9 vs 78.0, 122.7, 154.8,
451.6; threshold ≈ 74). Two controls kill it:

- **Chance.** P(a random descriptor places the two R=16 corpora at one extreme) = 2/C(6,2) = 0.133. Across 12
  descriptors, **1.60 perfect separators are expected by chance; exactly 1 was observed.**
- **LOCO.** Fixing that descriptor a priori and fitting only the threshold per fold gives **348**, still far
  below the global constant's 672. MetaQA (70.83) and WebQSP (78.04) straddle the boundary with ~7 units of
  margin on a scale spanning 30–452, so held-out thresholds flip them: hold MetaQA out and the source threshold
  lands at 54.0, assigning it R=4 and losing 412 golds.

## What per-corpus tuning is actually worth

| | NET |
|---|---|
| single global R = 0 / 4 / 8 / **16** | 0 / 329 / 485 / **672** |
| per-corpus oracle constant | 792 |
| **global recovers** | **84.9%** |
| gain from perfect per-corpus tuning | **+120** |

Gain breakdown vs global: MetaQA 0, 2Wiki 0, SQuAD 0, **MuSiQue +57, Hotpot +31, WebQSP +32**. All of the
per-corpus value is *avoiding harm* on the three corpora where R=16 is destructive — none of it is extra gain
where the reserve works.

This **revises B1.8's headline.** On three corpora the per-domain scalar looked essential (MetaQA 16 vs SQuAD 0).
On six, one global constant captures 84.9%, per-corpus adaptation is worth 15.1%, and no label-free method
reaches it. Honest caveat in the other direction: R=16 is optimal in aggregate but **strictly harmful on 3 of 6
corpora** — the aggregate is carried by MetaQA (635 of 672).

## The exact remaining bottleneck

**2Wiki is the counterexample that breaks every descriptor.** It sits inside the low-utility group's range on
*every* label-free statistic — overlap@10 0.169 vs the R≤4 group's [0.051, 0.223]; jaccard 0.060 vs [0.013,
0.096]; mean degree 3.83 vs [4.19, 73.0] — yet its reserve behaves like MetaQA's. Nothing about 2Wiki's graph
marks it as a high-reserve corpus.

What separates them is not the graph but **whether the query distribution's gold chains are actually traversable
in that graph**. MuSiQue makes this concrete: its graph is dense and well-connected, but its corpus title-mention
graph does not contain the reasoning chains, so expansion adds noise and R=16 costs −52. That is a **joint
property of (corpus, query distribution)**, and the contract forbids target query statistics — which is exactly
the information required.

So the bottleneck has moved once more:

- **B1.7:** conditional shift `P(D | P0)` — superseded.
- **B1.8:** the reserve is a per-domain scalar the first-unseen-query contract can't observe — **partly
  superseded**: it is mostly a *global* constant (84.9%), and the per-corpus residual is only 15.1%.
- **B1.9:** that 15.1% residual is not a function of the corpus substrate at all. It is a function of
  corpus×query-distribution fit, which label-free index-time adaptation cannot see **by construction**.

The scoping ruling was applied in full and was **not** the blocker — label-free index-time statistics are
computable, cheap and clean; they simply do not carry the signal.

## Recommendation

1. **Ship the global constant.** `R = 16` recovers 84.9% of the achievable per-corpus-constant utility with zero
   configuration, zero labels and zero risk of the 2× degradation every adaptive policy showed. If the harm on
   MuSiQue/Hotpot/WebQSP matters more than MetaQA's gain, `R = 4` is the conservative alternative (329, positive
   or zero on all six). This is a value judgement about which corpora matter and is yours, not mine.
2. **Do not add capacity.** Every negative here is a *transfer* failure with 6 corpora and 2 positives, not an
   under-fitting failure. An MLP, attention, or a richer descriptor set would fit 6 points perfectly and
   generalize no better.
3. **If B1 is revisited, the lever is more corpora, not more model** — specifically corpora that break the
   current confound: text corpora that want R=16 (to separate 2Wiki from MuSiQue/Hotpot) and KB corpora that
   want R=4 (WebQSP is currently the only one). With 2 positives out of 6, no method can be validated.
4. **The 15.1% residual is only reachable by relaxing the contract** to permit a small labelled probe set on a
   new corpus (a few hundred queries with relevance judgements would identify `R_C` directly and trivially).
   That is a genuine contract change and is your call — I have not assumed it.

## STOP

STOP after B1.9 as directed. No attention, no query MLP, no enlarged model, no B2, TEST untouched, 0 encoder
passes. The one small corpus-level configuration model permitted by STEP 5 was run (three policy classes) and
failed; nothing beyond it was attempted.

**`CORPUS_LEVEL_RESERVE_GENERALIZATION = NO`**
