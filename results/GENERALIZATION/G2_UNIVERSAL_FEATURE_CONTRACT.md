# G2 · Universal feature contract (v2) — dataset-agnostic L2 + hard inference-safety / cache contract

Machine-readable source of truth: **`G2_UNIVERSAL_FEATURE_CONTRACT.json`**. This note is the human summary.

## Candidate universe & admission
- **UNIVERSE** = original **P50** ∪ parameter-free **structural directional expansion** (M_MAX = **256**, an upper *generation* budget, not the pool size; no per-dataset M).
- **Mainline graph** = native-structural only, **synthetic kNN excluded** (kNN = ablation). Row-native per dataset: metaqa = `kb.txt` structural; 2wiki/squad = `master_nodes.neighbors` (edge counts confirm structural-dominated, not NER/kNN).
- **Final admission budget** TOP_POOL = **50** (B1 selects 50 from the universe).

## Anchors
Canonical universal anchor = **retrieval-seeded** (dense/splade top-k). Bracketed-entity (MetaQA-only) and title-match-alone are **forbidden as mainline** — their coverage is dataset-correlated (metaqa 1.00 / 2wiki 1.00 / musique 0.67 / squad 0.24) and would leak dataset identity through missingness. A generic entity linker is allowed only as an *optional auxiliary* source (same interface across datasets, inference-safe, may abstain); with no valid anchor, **fall back to retrieval-seeded**.

## Feature ladder (all bounded/relative, per-query normalized)
- **S0 — retrieval-relative** (always available): dense/splade percentiles, fused RRF percentile, retriever support, dense–splade agreement.
- **S1 — + structural**: min-hop-from-seed, seed-connectivity fraction, structural support, degree percentile, **query-local PPR** percentile.
- **S2 — + directional geometry**: `s_dir` percentile (residual-vs-edge-displacement), directional-expansion rank/hop (**our own traversal hop**, never a benchmark hop label), structural-anchor count, structural-distance percentile, directional support, `geometry_added` provenance flag (included **with** confidence/context features — provenance alone never defines relevance).

Excluded everywhere: dataset id, bracket indicator, **relation TYPE** (9/6094/1 → fingerprints identity), raw corpus-scale scores, gold relations/supporting-facts/paths, KB predicate names.

---

## HARD inference-safety / cache contract (applies to L1, B1, B2 — overrides implementation convenience)

**Core rule.** For a completely unseen query Q, every inference feature must be computable from ONLY: (1) Q, (2) frozen parameters, (3) corpus/graph artifacts built before Q and independent of relevance labels, (4) intermediates produced while processing Q. If it cannot be computed when Q is the *first ever* query after deployment, it is **not inference-safe**. **Caching does not make leaked information safe.**

**Offline cache (allowed, must be query-independent & label-independent):** node/entity embeddings, SPLADE doc vectors, dense index/shards, structural adjacency, partition maps, neighbor lists, node degree / static graph stats, corpus-construction edge provenance, graph transition operators, *optional* precomputed edge-displacement vectors, frozen B1/B2 weights, normalization/calibration stats fit **only** on permitted TRAIN.

**Online per-query (allowed):** query embedding, query SPLADE rep, dense/SPLADE retrieval, retrieval seeds, RRF/agreement, bounded traversal, seed-conditioned BFS distances, seed connectivity/support, query-local PPR (restart = current seeds), seed residual, directional compatibility, expansion provenance/hop, query-local rank/percentile features, B1/B2 forward pass.

**Normalization — two safe forms only:** **(A) TRAIN-frozen** — fit on the fold's TRAIN datasets, frozen before target eval; **(B) query-local** — over the current query's universe (rank/degree/PPR/direction percentiles). **Forbidden:** target-dataset-wide VAL/TEST statistics; any target calibration.

**Hop/path safety.** Allowed as model input: our own online `expansion_hop`, min distance from current-query seeds, paths from current-query BFS/PPR. **Forbidden as model input:** benchmark question-hop, gold decomposition/supporting-path/relation-chain. Benchmark hop labels are used **only for post-hoc evaluation breakdowns**.

**Provenance pass-through.** L1 must not emit only candidate IDs — it passes each candidate's inference-generated evidence (entered_via_dense/splade/structural_expansion, dense/splade rank percentiles, min_expansion_hop, structural support, directional score/rank, supporting-anchor count, query-local PPR) to B1/B2. (Q2 showed discarding *why* a candidate was retrieved destroys signal.)

**Cache implementation.** Distinguish CAN_BE_CACHED from MUST_BE_MATERIALIZED. Normalized edge displacement is safe to cache but do **not** blindly materialize `all_edges × 1536`; cache node embeddings + adjacency and compute `x_v − x_u` on demand for traversed edges only. Profile storage/latency first.

**PPR.** Cache adjacency / degree normalization / transition operator offline, but the **restart vector must come from the current query's retrieval seeds** — no target-wide precomputed query relevance.

**Repeated-query cache.** May cache query hash/embedding/retrieval/universe/features/B1-top50/B2-top5, but **correctness must never depend on it**; scientific evaluation assumes cache-miss/unseen queries; never report canonical latency from repeated-query hits.

**Dataset missingness.** No silent dataset-specific replacement; each universal feature has deterministic abstention semantics; **no `if dataset==…` anywhere in the inference path**; audit whether missingness fingerprints identity.

## Feature-provenance manifest (hard gate)
Every inference feature carries: `feature_name`, `feature_family` ∈ {STATIC_CORPUS, STATIC_GRAPH, TRAIN_FROZEN, QUERY_RETRIEVAL, QUERY_STRUCTURE, QUERY_GEOMETRY}, `offline_dependencies`, `online_dependencies`, and the five booleans. **Hard gate — every inference feature must satisfy:** `uses_gold_labels=NO` ∧ `uses_supporting_fact_annotations=NO` ∧ `uses_dataset_identity=NO` ∧ `uses_target_statistics=NO` ∧ `available_for_first_unseen_query=YES`. The full manifest for the **17 implemented B1.1 features** is in the JSON (`feature_provenance_manifest_B1_1_implemented`); **all 17 pass**.

## Automated leakage test (runs before every LODO/eval)
`scratchpad/_b11_lodo.py::inference_safety_audit()` hard-fails if any inference tensor depends on target labels, target VAL/TEST aggregate statistics, benchmark annotations unavailable for a real unseen query, dataset ID, gold entities/relations/paths, or future queries — checked structurally (column-name allow-list + banned substrings + feature-dim + `qhop`-is-per-query-only + TRAIN-frozen normalization enforced in the fold). It emits per-dataset abstention rates and a 3-way identity-leak classifier accuracy (vs 0.333 chance) so residual fingerprinting is quantified.

## Guardrails
True LODO (train all-except-D; hyperparameters + normalization from TRAIN only; evaluate target VAL once); no target refit/calibration/model-selection (target-refit = oracle/control only); the six datasets are **development/LODO evidence, not final external-generalization proof** — a fresh external dataset is required before any universal claim.
