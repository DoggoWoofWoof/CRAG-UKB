# L2 Canonical Corpus — Plan & Format Spec

`CORPUS_INTERFACE = C / Dense+SPLADE fusion (RRF K0=60) / K=100 / P=50` (the frozen L1→L2 interface).
Builder: `scratchpad/build_l2_corpus.py`. Routing is **bit-identical** to the L1 evaluator (verified: pilot ANY/ALL match `phase1_*` C/K100/P50 exactly). **No Qwen/SPLADE re-encode** — all scores derive from cached embeddings/top200/partition maps.

---

## Three separate objects (never conflated)

| object | what | size | truncated? |
|---|---|---|---|
| **(1) FULL SCOPE** | every query's complete P50 candidate universe (~5k), CSR arrays | large | **NEVER** — required for eval, failure analysis, full reranking, L3 interface |
| **(2) TRAIN SUBSET** | all in-scope positives + ≤128 tagged sampled negatives/query (index arrays into (1)) | small | training only |
| **(3) EXPENSIVE FEATURES** | relation / path / graph-heavy features | deferred | built only after Step-9 profiling, only where justified |

---

## On-disk format (CSR, compact, memmap-friendly)

Per `(dataset, split)` directory under `data/l2_corpus/<dataset>/<split>/`:

**(1) Full scope** — aligned arrays, one row per (query, candidate) pair, in **canonical dense-desc order** within each query's scope:
| file | dtype | bytes/pair | meaning |
|---|---|--:|---|
| `query_offsets.npy` | int64 [Nq+1] | — | CSR row pointers |
| `cand_ids.npy` | int32 | 4 | doc index (eng/doc order, aligned to `nodes.npy`, `hard`, graph) |
| `labels.npy` | uint8 | 1 | 1 = gold-in-scope |
| `part_id.npy` | int16 | 2 | candidate's C hard-partition id |
| `part_rank.npy` | int16 | 2 | fusion rank (0..49) of that partition among the P50 selected |
| `dense_score.npy` | float16 | 2 | cosine `q·doc` from **cached** embeddings (all candidates) |
| `dense_rank.npy` | int16 | 2 | global dense rank (0..199) if in top-200 cache, else −1 |
| `splade_rank.npy` | int16 | 2 | global SPLADE rank (0..199) if in top-200 cache, else −1 |
| `query_meta.json` | — | — | per-query record (below) |

Measured **≈15.6 bytes/pair** total (incl. subset arrays + meta). `dense_score` is the only per-pair float we compute; it is cheap (cached-embedding matmul, GPU). **SPLADE score** for the full scope needs query-side SPLADE encoding → **deferred to Step-9 profiling**; only SPLADE *rank* (top-200 membership) is stored now.

**Per-query record** (`query_meta.json`): `query_id, dataset, split, row_all, N_SCOPE, N_GOLD_EXPECTED_RAW, N_GOLD_EXPECTED_INCORP, N_GOLD_IN_SCOPE, ANY_GOLD_PRESENT, ALL_GOLD_PRESENT, hop, L1_STATUS`.

**(2) Train subset** — `train_sub_offsets.npy` (int64 [Nq+1]), `train_sub_local.npy` (int32, local idx into that query's scope), `train_sub_source.json` (provenance tag per row). Every positive kept; negatives tagged by bucket.

**Canonical candidate order = dense score descending** (reproducible primary retrievability signal within scope). Partition order is fully recoverable from `part_rank`. Nothing about the scope *set* depends on the order — it matches the L1 union exactly.

---

## L1 failure label (Step 4) — survives all later L2 eval

Per query, from `N_GOLD_IN_SCOPE` vs `N_GOLD_EXPECTED_INCORP`:
- `L1_ANY_FAIL` — `N_GOLD_IN_SCOPE == 0` (no evidence survived; L2 must never be blamed).
- `L1_PARTIAL` — `0 < N_GOLD_IN_SCOPE < N_GOLD_EXPECTED` (partial evidence; ANY-coverage opportunity).
- `L1_ALL_SUCCESS` — `N_GOLD_IN_SCOPE == N_GOLD_EXPECTED`.

`N_GOLD_EXPECTED_RAW` (pre doc-map) is kept separately so gold-ids that don't map into the corpus are not silently blamed on L1.

---

## Negative sampler (Step 5) — deterministic, tagged, over the FULL scope

Always keep **all positives**. Then ≤128 negatives/query from buckets (seed=1234), each retaining `NEGATIVE_SOURCE`:
`DENSE_HARD` (best dense rank), `SPLADE_HARD` (best splade rank), `DENSE_SPLADE_DISAGREEMENT` (|drank−srank| large, both present), `HIGH_PARTITION_VOTE` (top selected partitions), `SAME_PARTITION` (shares a positive's partition), `RANDOM` (fill).
**Only buckets whose information exists now are used.** `NER_CONFUSER / KNN_CONFUSER / RELATION_CONFUSER / PATH_CONFUSER` are **deferred** until those features exist (do not invent confusers before the feature).

---

## Storage audit (Step 6) — realistic dtypes, ~15.6 B/pair (full scope, train+val+test)

| dataset | N_train | N_val | N_test | mean_scope | total cand rows | est. base storage | relation (dense f16) | path (dense f16) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 2wiki_clean | 10 500 | 3 000 | 1 500 | ~5 029 | 75.4 M | **1.18 GB** | +0.15 GB | +0.15 GB |
| musique_clean | 13 956 | 3 987 | 1 995 | ~5 042 | 100.5 M | **1.57 GB** | +0.20 GB | +0.20 GB |
| hotpotqa_clean | 68 496 | 19 570 | 9 786 | ~5 081 | 497.2 M | **7.76 GB** | +0.99 GB | +0.99 GB |
| squad_clean | 91 223 | 26 063 | 13 033 | ~5 039 | 656.6 M | **10.24 GB** | +1.31 GB | +1.31 GB |
| **metaqa** | 329 282 | 39 138 | 39 093 | ~5 038 | **2 053 M** | **⚠ 32.0 GB** | +4.1 GB | +4.1 GB |
| webqsp | 1 104 | 315 | 159 | ~5 023 | 7.9 M | **0.12 GB** | +0.02 GB | +0.02 GB |
| **all six** | | | | | ~3.39 B | **≈52.9 GB** | +6.8 GB | +6.8 GB |

Relation/path *dense* worst-case shown; **actual will be sparse** (mostly missing on text sets → COO ~10 B per present entry, far smaller). **MetaQA is pathological** (32 GB base; 329k train queries, gold-multiplicity mean 7.5 / max 285) → **do not materialize full**; cap/stratify at build time, handle separately (Step 13). This is exactly why the pilot is 2wiki+musique only (~2.75 GB).

---

## Build order (Steps 8, 13)
1. **Pilot: 2wiki + musique** (multi-gold/multi-hop, manageable, prior head history) — verify parity, integrity, sampler, storage. ← *this deliverable*
2. Validate transfer build: **hotpot + squad**.
3. **metaqa + webqsp** separately, with caps + substrate/protocol caveats (metaqa native splits; webqsp no invented labels, train=1104).

## Reuse rule (Step 10)
Reuse *components* (OffsetHead, MixtureHead-K8, ResidualAdapter, InfoNCE+hard-neg loop, crag masked-controller/gates/dropout/LODO, XGBRanker, L3 PPR) — **replace their old candidate pools** (dense∪NER, A/full-corpus) with this C/P50 corpus. Never compare models trained on different pools and call it a signal ablation.

## Regime data (Step 11) — collected, not labeled
The corpus stores per-candidate `dense_score/rank`, `splade_rank`, `part_rank` now; once offset/mixture/relation/path are computed on-scope, per-query **gold-rank-by-expert** is derivable → later data-derived regimes (LEXICAL/SEMANTIC/RELATIONAL/PATH/CONFLICT/RESCUE/REDUNDANT/HARD) on **TRAIN only**. No labels assigned yet; no TEST used.

## L2→L3 contract (Step 12) — room left, nothing trained
Feature/label arrays are CSR-extensible; future per-candidate outputs (`relevance_score, expert_weights, regime_probs, node_expand_priority, edge_transition_score, edge_family scores`) attach as new aligned arrays. No speculative heads built.

## Expensive-feature profiling (Step 9) — REQUIRED before E4/E5
Offset/Mixture are cheap (cached head forward → same matmul family as dense). Relation/Path/graph require edge-relation-text / path enumeration per candidate → profile `SECONDS/1000 candidates`, memory, cache size, batchability, precomputability on the C/P50 pool **before** building them over full scope. Report deferred until pilot accepted.

---

## Status (updated after pilot completes)
`FULL_SCOPE_BUILT`, `TRAIN_SAMPLER_BUILT`, `BASE_FEATURE_ALIGNMENT`, pilot counts and storage → see `L2_CORPUS_MANIFEST.json` and the summary in the accompanying report.
**STOP before model training** — awaiting review of: corpus format, storage estimate, pilot integrity, negative sampler, expensive-feature cost.
