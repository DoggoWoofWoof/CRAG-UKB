# A-vs-C Graph Diagnostic — Tasks 5–11 (P50 primary, K100 fusion router)

**Populations:** 2wiki 1500 (full), musique 1995 (full), hotpot 2500 (strided sample of 9786), squad 2500 (strided sample of 13033).
**Rule:** `GRAPH_SCOPE_RULE = INDUCED_SUBGRAPH_OVER_L1_CANDIDATE_IDS`. Seeds = (dense top-20 ∪ splade top-20) ∩ scope. Traversal ≤3 hops.
**Raw metrics:** `results/L1/ac_graph_diagnostic.json`. Scope/overlap: `ac_scope_analysis.json` + `ac_scope_hotpot.json`. Edge parity: `ac_edge_recon.json`. Runtime: 2wiki 10.8s/100q, musique 5.8s, squad 14.3s, hotpot 67.7s/100q (est. full hotpot ~2.3 h — sampled instead).

---

## STEP 1 — Edge-family reconstruction / parity: **PASS on all 4 datasets**

All edges canonicalized undirected `(min,max)`. `EDGE_FAMILY_RECONSTRUCTION=PASS` (alignment gate + C-parity + B-parity) everywhere.

| dataset | N_nodes | STRUCT | KNN | NER | A=S⊔K | B=S∪NER | C=A∪NER | C_par | B_par |
|---|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|
| 2wiki | 65 865 | 126 070 | 134 737 | 523 194 | 260 807 | 620 374 | 728 689 | PASS | PASS |
| musique | 13 672 | 51 543 | 27 406 | 99 656 | 78 949 | 145 043 | 163 723 | PASS | PASS |
| hotpot | 507 494 | 3 426 945 | 1 166 652 | 4 342 365 | 4 559 172 | 7 485 652 | 8 440 280 | PASS | PASS |
| squad | 19 029 | 694 580 | 28 276 | 145 515 | 722 856 | 821 068 | 841 351 | PASS | PASS |

- **STRUCT ∩ KNN = 0** everywhere → `A = STRUCT ⊔ KNN` is a clean disjoint union (Qwen-kNN edges are entirely non-structural).
- **C == A ∪ NER** and **B == STRUCT ∪ NER**, bit-exact (verified against `variant_C/graph.pt`, `variant_B/graph.pt`).
- **NER ∩ KNN overlap:** 2wiki 26 422 (20% of KNN), musique 8 726 (32%), hotpot 183 966 (16%), squad 7 993 (28%). C's *unique* edges over B (= KNN − NER∩KNN) are only the non-NER kNN: 2wiki 108 315, musique 18 680, hotpot 982 686, squad 20 283.

---

## STEP 2 — Where A vs C actually enters the pipeline (from the code)

Verified in `src/experiments/e2e_pipeline.py` + `l1_universal_head.py` + `l2_seed.py`:

| Stage | Signals / graph actually used | Edge family | Topology dependence |
|---|---|---|---|
| **L1 scope** | partition `hard` from `eng.partition_map` (`gte_qwen`) | — | **This is the only place A/C differ.** Default = **A**; C's map is not wired in. |
| **L2 signals** | `dense`, `rel_hard` (OffsetHead), `mlpT` (MixtureHead-K8), `SPLADE`, [`adapter`] | **none** | `mem_idx = _onehop_membership(eng)` is **structural-only** (identical A/C). Topology enters only via `hard`-scope when `scope_topk>0`. Default `scope_topk=0` ⇒ L2 runs **unscoped**. |
| **L3 traversal** | `_ner_compose` → PPR (`α=0.5`) over `P = _transition(A_str + A_ner)`; `n_seed=2`, `w=3.0`; `expand=100` = go-outward | **B (struct + weighted-NER)** — line 505 comment: *"kNN dropped"* | Built from `eng` struct + `build_ner_edges(maxdf=25,w)` = `ner_edges_w_df25.pkl`. **Topology-independent.** |

**`CURRENT_PIPELINE_SCOPE_RULE`:** partition scope is an *optional* filter (`scope_topk>0`, mem_idx = structural one-hop); the `anchor_docs` path is the KGQA-subgraph protocol. The diagnostic's induced-subgraph rule models the `scope_topk>0` path. The pipeline's L3 graph **B** is bit-identical to this report's B family.

**Consequence:** the kNN edges that distinguish A from C are **consumed nowhere downstream** — dropped at L3, absent at L2. Their only footprint is on the METIS partitioning that sets L1 scope.

---

## STEP 3 — Graph-family ablation (fixed **A-scope**, P50) — the edge-necessity test

**CONDITIONAL_ON_GOLD_PRESENT** reach (present golds reached from seeds), the metric that isolates graph topology from L1 pruning:

reach@1 (1-hop — most discriminative):

| dataset | STRUCT | B=S+NER | A=S+kNN | C=S+NER+kNN |
|---|--:|--:|--:|--:|
| 2wiki | 0.9711 | **0.9909** | 0.9796 | **0.9929** |
| musique | 0.9074 | **0.9583** | 0.9348 | **0.9669** |
| hotpot | 0.9787 | 0.9855 | 0.9859 | **0.9896** |
| squad | 0.9934 | 0.9963 | 0.9951 | **0.9963** |

reach@3 (≤3-hop — the operating point; **saturates**):

| dataset | STRUCT | B | A | C |
|---|--:|--:|--:|--:|
| 2wiki | 0.9983 | 0.9997 | 0.9997 | 1.0000 |
| musique | 0.9697 | 0.9952 | 0.9969 | 1.0000 |
| hotpot | 0.9996 | 0.9998 | 0.9998 | 1.0000 |
| squad | 0.9980 | 0.9996 | 1.0000 | 1.0000 |

Efficiency — `gold_reach_per_1k_expanded` / `mean_expanded`:

| dataset | STRUCT | B | A | C | read |
|---|--:|--:|--:|--:|---|
| musique | 0.705 / 3142 | **0.514 / 4426** | 0.552 / 4126 | 0.477 / 4794 | C expands most, yields least per-1k |
| hotpot | 0.412 / 4689 | 0.401 / 4820 | 0.392 / 4924 | 0.389 / 4964 | flat; kNN adds expansion, not yield |

**Findings**
1. **STRUCT alone is the weakest** everywhere (musique @1 0.907, @3 0.970). Edges matter.
2. **NER is the dominant lever.** STRUCT→B is the big jump (musique @3 +2.55pp, @1 +5.1pp; 2wiki @1 +2.0pp). B ≈ C at @3.
3. **NER > kNN as a downstream edge family.** B (struct+NER) beats A (struct+kNN) at reach@1 on 2wiki (0.991 vs 0.980) and musique (0.958 vs 0.935); equal on hotpot/squad.
4. **kNN's marginal value over NER is tiny.** B→C: @1 +0.13–0.20pp (text), +0.41 (hotpot); @3 +0.02–0.48pp. **kNN is not noise** (C ≥ B everywhere — semantic-kNN noise test = negative), but it is largely *redundant* with NER (16–32% literal overlap) and its unique contribution is marginal while it raises expansion cost (lower gold-per-1k).

---

## STEP 4 — Conditions COND_1..4 (scope × graph) + connectivity

**Once a gold survives L1 scope, both graphs connect it to the seeds ≈always** (conditional reach@3 = 0.9996–1.0 in every A/C-graph condition). So **unconditional reach@3 tracks `gold_present_rate` almost exactly** — downstream success is governed by L1 scope, not graph connectivity.

`gold_present_rate` (= L1 scope quality, the real differentiator):

| dataset | A-scope | C-scope | winner |
|---|--:|--:|:--|
| 2wiki | 0.9700 | 0.9719 | C +0.19 |
| musique | 0.9764 | 0.9824 | C +0.60 |
| hotpot | **0.9658** | 0.9526 | **A +1.32** |
| squad | 0.9768 | 0.9792 | C +0.24 |

→ C's partition marginally wins 3/4 text sets; A wins hotpot by more. **Mutual, small, no dominance** (matches Tasks 2–4 oracle ceilings +0.9 to +2.5pp).

**Connectivity — COND_4 (C-scope + A-graph) is the coherence tell:**

| cond (scope+graph) | 2wiki ncomp / largest | musique | hotpot | squad |
|---|--:|--:|--:|--:|
| COND_1 A+A | 2.0 / .999 | 2.7 / .998 | 22 / .995 | 4.8 / .997 |
| COND_2 A+C | 1.8 / .999 | 1.3 / .9997 | 14 / .997 | 2.3 / .999 |
| COND_3 C+C | 2.5 / .999 | 1.2 / .9996 | 5.4 / .998 | 2.2 / .9995 |
| **COND_4 C+A** | **172 / .957** | **101 / .973** | **226 / .949** | **139 / .964** |

A's graph (struct+kNN, **no NER**) laid over C's NER-shaped scope **fragments** (100–226 components). Golds still sit in the giant component (reach@3 stays ~1.0), but this shows **A-graph is a poor structural match for a NER-built scope** — if you ever adopt C partitioning you must carry NER in the L3 graph (i.e., B/C graph), which the pipeline already does.

**LITERAL_SCOPE_UNION cost** (A-scope ∪ C-scope, a real candidate cost, *not* a free oracle): union/A = 1.75× / 1.46× / 1.83× / 1.45× (2wiki/musique/hotpot/squad); Jaccard 0.14 / 0.37 / 0.09 / 0.38. It buys only the +0.9…+2.5pp oracle ceiling (≈0.4–1.1pp per extra 1k candidates). **Not worth ~1.5–1.8× candidate cost.** `ORACLE_CHOOSE_A_OR_C` (the magic per-query selector) is not implementable; `LITERAL_SCOPE_UNION` is the only realizable form and is a poor trade.

---

## Answers (Q6–Q10) & decision variables

- **Q6 — current pipeline scope rule?** Optional partition scope (`scope_topk>0`, structural mem_idx); default unscoped L2; L3 = PPR over struct+NER. Induced-subgraph-over-candidates is the right model.
- **Q7 — does the graph discriminate A vs C on present golds?** **No.** Conditional reach@3 saturates (~1.0) for any NER-bearing graph; even STRUCT reaches 0.97–1.0. A vs C differ only in scope.
- **Q8 — which edge family carries downstream value?** **NER** (B). It is the dominant lever (+2.5pp conditional @3 over STRUCT on musique) and is topology-independent.
- **Q9 — is kNN downstream-valuable / noisy?** **Marginal, not noisy.** C ≥ B everywhere (no noise), but the unique kNN gain over NER is +0.1–0.5pp at higher traversal cost, and kNN is dropped at L3 today → effective downstream value ≈ 0.
- **Q10 — is C partitioning required?** **No.** C-scope is scope-neutral-to-slightly-better on text but worse on hotpot; net near-neutral. No rebuild justified.

| Decision variable | Value |
|---|---|
| `P_MAIN` | **50** |
| `L1_PARTITION_TOPOLOGY` | **A** (canonical, already wired; scope near-neutral vs C, wins hotpot; no rebuild) |
| `L2_GRAPH` | **none** (L2 signals are embedding/lexical; no edge family) |
| `L3_GRAPH` | **B = struct + weighted-NER** (already the pipeline default; the graph's real home) |
| `C_PARTITIONING_REQUIRED` | **NO** |
| `NER_EDGES_DOWNSTREAM_VALUE` | **HIGH** — dominant L3 lever, topology-independent |
| `KNN_EDGES_DOWNSTREAM_VALUE` | **LOW** — not noise, but redundant with NER, unused at L3, marginal unique gain, higher cost |

**Bottom line:** Keep **A** as the L1 partition (no C rebuild), keep **B (struct+NER)** as the L3 graph (already default), and treat kNN as droppable. The A-vs-C question resolves to a near-tie on L1 scope; the graph value that matters downstream is NER, which neither A nor C uniquely provides.
