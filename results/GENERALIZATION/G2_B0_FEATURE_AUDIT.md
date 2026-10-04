# G2 · B0 — Universal feature-availability audit (6 datasets)

**Goal.** Before any Track-B training, verify every proposed universal feature has the **same inference meaning** across 2Wiki, MuSiQue, SQuAD, HotpotQA, WebQSP, MetaQA — with no dataset IDs, no gold-derived entity links, no supporting-fact / target labels. Classify each feature per dataset **AVAILABLE / MISSING / DEGENERATE / DISTRIBUTION**, and — critically — check whether a feature's *missingness* trivially reveals dataset identity. Machine-readable: `G2_B0_FEATURE_AUDIT.json`.

## Node & graph model (interface is uniform; provenance is not)
| | 2wiki | hotpotqa | musique | squad | webqsp | metaqa |
|---|---|---|---|---|---|---|
| node | document | document | document | document | doc titled by entity | KB entity |
| graph provenance | hyperlink | hyperlink | title-mention | title-mention | freebase rel | KB triple |
| typed? (#rel) | untyped (1) | untyped (1) | untyped (1) | untyped (1) | typed (6094) | typed (9) |
| #edges | 359 k | 15.4 M | 2.74 M | 874 k | 3.79 M | 133 k |

Row-space adjacency is available **uniformly** for all six via `master_nodes.neighbors` (the interface `l2_relation` already uses); the canonical `graph_*.tsv` live in a hash/curid id-space and would need a title/curid crosswalk — avoided by using `master_nodes.neighbors`. kNN graph is missing for hotpot, NER graph is missing for metaqa/webqsp — **both excluded from the mainline** (their presence would itself leak identity).

## The decisive finding — query anchor (val coverage, n≈315–400)
| anchor mechanism | 2wiki | hotpotqa | musique | squad | webqsp | metaqa |
|---|--:|--:|--:|--:|--:|--:|
| **bracketed entity** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.00** |
| **title-match (deterministic)** | 1.00 | 0.93 | 0.67 | **0.24** | 0.99 | 1.00 |
| **retrieval seed (dense/splade top-k)** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

- **Bracketed entity is MetaQA-only** (1.0 vs 0.0). Using it — or any `is-bracketed` signal — as a feature **perfectly reveals the dataset** → **EXCLUDED**.
- **Title-match coverage is dataset-correlated** (1.00/0.93/0.67/**0.24**/0.99/1.00). SQuAD queries almost never name a document title (DEGENERATE); MuSiQue is partial (DISTRIBUTION). If directional/structural features were anchored on title-match *alone*, their abstention rate would track dataset identity → identity leak via missingness.
- **Retrieval seeds are the only ~100 %-uniform anchor.** ⇒ the contract **mandates retrieval-seeded anchors** as the universal base, with title-match allowed only as an on-top refinement flag. This keeps `*_anchor_present` ≈ 1.0 everywhere and non-revealing.

## Per-feature availability (all six) — see JSON for the full matrix
- **S0 retrieval-relative** (dense/splade percentiles, RRF, agreement, P50 membership): **AVAILABLE all six.** Percentile-normalized ⇒ corpus-scale-free.
- **Structural graph, seed-distance, seed-connectivity, PPR, degree**: **AVAILABLE all six** (row-space `neighbors`; anchor-conditional features are retrieval-seeded ⇒ ~1.0 coverage; degree is anchor-free). Absolute graph scale differs hugely (hotpot 15 M vs metaqa 133 k edges) ⇒ **must** be per-query percentile-normalized, never raw.
- **Directional geometry `s_dir`, expansion provenance/hop**: **AVAILABLE all six** via retrieval-seed anchors — but **DISCRIMINATIVE value is demonstrated on MetaQA only** (Track-A/Q2, entity-anchored). Text/WebQSP efficacy through retrieval-seed anchors is **UNTESTED** and is an explicit B1 / Track-A-extension question. On a pure-P50 (no-expansion) scope, `expansion_hop` is degenerate (all 0).

## Missingness → identity-leak analysis (explicit, per the directive)
| signal | reveals dataset id? | action |
|---|---|---|
| bracketed-entity indicator | **YES, perfectly** (metaqa only) | **EXCLUDED** |
| title-match-alone anchor | **YES, partially** (squad 0.24) | not a sole anchor; retrieval-seeded base neutralizes it |
| relation TYPE / count | **YES** (9 / 6094 / 1) | relation type EXCLUDED; only `s_dir` displacement used |
| raw degree/hop/PPR scale | **YES** if absolute | neutralized by per-query percentile normalization |
| ner/knn graph presence | **YES** (metaqa/webqsp no-ner; hotpot no-knn) | both EXCLUDED from mainline |

Under the three rules — (a) retrieval-seeded anchors, (b) per-query percentile normalization of every graph feature, (c) exclude relation-type / bracket / ner-knn-presence — **feature missingness does not reveal dataset identity.**

## Data-availability caveats (for the smoke plan)
L2 corpora local: **2wiki_clean, musique_clean, squad_clean**. Modal-only: metaqa (197 M-row P50 val), hotpotqa (507 k docs), webqsp (781 k). ⇒ a **local B1 LODO smoke runs 3-way on {2wiki, musique, squad}**; the full 6-way LODO needs Modal for metaqa/hotpot/webqsp.

## Q2 population caveat (preserved)
This audit's coverage numbers and the Q2 decomposition refer to the **valid-query population** of the frozen G1/C6 path (≥1 in-scope gold). Do **not** generalize to fully-L1-failed queries until that population is separately measured.

## Bottom line
All S0/S1/S2 features are AVAILABLE across all six under a uniform interface **iff** the anchor is retrieval-seeded, every graph feature is per-query percentile-normalized, and relation-type/bracket/ner-knn-presence are excluded. **Open risk carried into B1:** directional-geometry discriminative value is MetaQA-demonstrated only, and **SQuAD is the weakest structural/directional regime** (title-match 0.24, derived title-mention graph) — the dataset most likely to expose an S1/S2 over-reliance under LODO.

---

## HARD inference-safety / cache contract (v2 — applies to L1, B1, B2)
Every inference feature for a completely unseen query Q must be computable from ONLY: **(1)** Q itself, **(2)** frozen model parameters, **(3)** corpus/graph artifacts built before Q arrived and independent of relevance labels, **(4)** intermediate results produced while processing Q. If it cannot be computed when Q is the *first ever* query after deployment, it is **not inference-safe**. Full contract (offline-cache allow-list, online-computation allow-list, TRAIN-frozen vs query-local normalization, anchor policy, hop/path safety, provenance pass-through, PPR restart-from-current-seeds, repeated-query-cache caveat, dataset-missingness rule) is in **`G2_UNIVERSAL_FEATURE_CONTRACT.json` → `inference_safety_cache_contract`**.

**Hard gate (every inference feature):** `uses_gold_labels=NO` ∧ `uses_supporting_fact_annotations=NO` ∧ `uses_dataset_identity=NO` ∧ `uses_target_statistics=NO` ∧ `available_for_first_unseen_query=YES`.

**Feature-provenance manifest** for the 17 actually-implemented B1.1 features (family ∈ {STATIC_CORPUS, STATIC_GRAPH, TRAIN_FROZEN, QUERY_RETRIEVAL, QUERY_STRUCTURE, QUERY_GEOMETRY}, offline/online deps, and the five hard-gate booleans) is in the same JSON → `feature_provenance_manifest_B1_1_implemented`. **All 17 pass the gate.** The benchmark question-hop (`qhop`) is stored per-query for **post-hoc breakdown ONLY** and is asserted **absent** from the model feature matrix.

**Automated leakage test.** `scratchpad/_b11_lodo.py::inference_safety_audit()` runs **before any training/eval** and hard-fails (assert) if: feature columns deviate from the 17 contract-safe names, any column name contains a banned substring (`gold/label/dataset/corpus/hop_annotation/supporting/relation_type/bracket/answer/target`), feature dim ≠ 17, or `qhop` is per-candidate rather than per-query. It also reports per-dataset abstention/missingness rates (`s_dir_abstain_frac`, `geometry_added_frac`, `struct_unreachable_frac`) so missingness fingerprinting stays visible; `IDENTITY_LEAK_DIAGNOSTIC` (3-way classifier accuracy from features vs 0.333 chance) quantifies residual fingerprinting. TRAIN-frozen normalization is enforced structurally (target excluded from the fold; mean/std from source rows only; percentile features are query-local).

## Correction to the data-availability caveat
The earlier "MetaQA Modal-only" note applies to the **full 5-expert L2 backbone** (197 M-row P50 corpus). For the **B1.1 admission pilot** MetaQA is **fully local**: `dense_top200_all` / `splade_top200_all` (query-online retrieval results), `nodes.npy` (node embeddings), `queries_all.npy`, and the **kb.txt structural graph** suffice to rebuild the candidate universe + all S0/S1/S2 features. ⇒ the 3-way pilot {**MetaQA, 2Wiki, SQuAD**} runs **entirely local**; Modal is only needed to extend to hotpot/webqsp in the full six-way LODO. Mainline graph provenance is native-structural, kNN excluded: metaqa = `kb.txt` (its `master_nodes.neighbors` mixes in kNN per edge counts); 2wiki/squad = `master_nodes.neighbors` (edge counts confirm structural-dominated, not the large NER/kNN sets).
