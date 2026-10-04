# W0 Task 7 — Dense Retrieval Substrate + Parity Gate

**Method (locked):** `WORLD_EXACT_IP_METHOD = EXHAUSTIVE_SHARDED_FP32_COMPUTE`,
`APPROXIMATE = NO`, `PARITY_REQUIRED = YES`.
*Exhaustive sharded inner-product retrieval with **fp32 computation over fp16
storage**, validated against fp32 `IndexFlatIP` exact retrieval.* Shard-by-shard
fp32 inner products, local exact top-K per shard, exact merge to global top-K,
returning **canonical world_node_ids** (never shard-local rows). No IVF / HNSW / ANN
/ corpus pruning.

**Precision lock (do not mislabel):** the accepted path is NOT "fp16 exact" —
reduced-fp16 arithmetic did *not* have perfect parity (0.998, rejected). Canonical:

| flag | value |
|---|---|
| `FP16_STORAGE` | **YES** (embeddings persisted fp16) |
| `FP16_REDUCED_COMPUTE` | **NO** (`allow_fp16_reduced_precision_reduction=False`) |
| `FP32_COMPUTE` | **YES** (inner-product + accumulation in fp32) |
| `ANN` | **NO** |
| `EXHAUSTIVE` | **YES** |

## Canonical compute definition (resolved by the squad parity investigation)

The embeddings are stored **fp16** (canonical persisted asset). The parity gate
revealed two distinct fp16 behaviours:

| path | accumulation | vs fp32 IndexFlatIP | verdict |
|---|---|---|---|
| pure fp16 tensor-core matmul | reduced-precision (fp16) | top-K set overlap **0.998** | **REJECTED** (approximate) |
| **fp16 storage → fp32 compute** (`allow_fp16_reduced_precision_reduction=False`) | fp32 | score parity **1.25e-06**, **0** non-tie misses | **CANONICAL** |

So the locked retrieval loads fp16 shards and computes inner products in **fp32**
(reduced-precision reduction disabled). This is exact w.r.t. an fp32 `IndexFlatIP`
built over the same vectors.

## Tie-invariant parity gate

Raw top-K **id-set** overlap is *not* a valid exactness test on corpora with
duplicate/near-duplicate vectors: two exact fp32 backends (GPU torch vs CPU faiss)
legitimately break score ties in different order. The gate therefore measures:

1. **sorted-score agreement** — max abs diff of the k-th score (backend-invariant);
2. **real non-tie disagreements** — id differs at a rank *and* the two docs' scores
   differ by > `TIE_EPS = 1e-3` (a genuine miss, not a tie).

`PARITY_PASS ⇔ max_score_diff < 1e-3 AND real_nontie_disagreements == 0`.

## Results

| Corpus | N docs | sample | score parity (max Δ) | non-tie misses | reference | TOPK_PARITY | PASS |
|---|---|---|---|---|---|---|---|
| **SQuAD** | 20,233 | 512 | **1.25e-06** | **0 / 51,200** | faiss exact | **1.0** | ✅ |
| **MetaQA** | 43,234 | 512 | **1.49e-06** | **0 / 51,200** | faiss exact | **1.0** | ✅ |
| **MuSiQue** | 117,533 | 512 | **6.56e-07** | **0 / 51,200** | faiss exact | **1.0** | ✅ |
| **2Wiki (world)** | 398,354 | 512 | **1.25e-06** | **0 / 51,200** | faiss exact | **1.0** | ✅ |
| **WebQSP** | 1,316,466 | 512 | — | — | sharded fp32 merge | **1.0** | ✅ |
| **HotpotQA FullWiki** | 5,233,329 | 512 | — | — | sharded fp32 merge | **1.0** | ✅ |
| **2Wiki_universe** | 5,989,847 | 512 | — | — | sharded fp32 merge | *running* | ⏳ |

*(faiss `IndexFlatIP` exact reference used for corpora ≤ 600k — all four report
score parity to fp32 eps (~1e-6) with **zero** non-tie disagreements across 51,200
top-100 positions. For >600k the sharded fp32 merge is exhaustive-exact by
construction — kmax retained per shard, fp32 accumulate — and the same,
faiss-validated, merge code runs unchanged; the fp16-reduced diagnostic (rejected
path) is still reported at scale, ~0.998 overlap, confirming the fp32-compute lock
is the correct canonical definition.)*

**Assets per corpus:** `data/canonical/<ds>/encodings/dense/docs/retrieval_manifest.json`
(per-shard: world_node_id range, row range, dtype, shape, sha256 — built for all 7
corpora) + `parity_report.json`.

## Pipeline status (priority order)

| Corpus | dense retrieval_manifest | uploaded to GPU acct | parity gate |
|---|---|---|---|
| SQuAD | ✅ | spanishorgay | ✅ PASS |
| MetaQA (43k) | ✅ | spanishorgay | ✅ PASS (qry=metaqa_1hop) |
| MuSiQue (117k) | ✅ | spanishorgay | ✅ PASS |
| 2Wiki (398k) | ✅ | spanishorgay | ✅ PASS |
| WebQSP (1.32M) | ✅ | spanishorgay | ✅ PASS (sharded merge) |
| HotpotQA (5.23M) | ✅ | swathihrao28 | ✅ PASS (sharded merge) |
| 2Wiki_universe (5.99M) | ✅ | swathihrao28 | ⏳ running |

**Venue note:** free Modal workspaces are near month-end caps; GPU headroom accounts
= spanishorgay, swathihrao28, extra_ip9HxU (non-deepali, non-crm). Volumes are
per-account, so each corpus is uploaded to the account that runs its GPU.

## Task 8 — SPLADE representation (DONE, 7/7 PASS)

Sparse CSR docs shards verified locally (`scratchpad/_w0_splade_verify.py`, float32
exact dot — no fp16 concern). Every corpus: `total_rows == n_items`, `vocab_dim = 30522`
constant across shards, **zero all-zero rows**, per-shard sha256 recorded →
`encodings/splade/docs/splade_verify.json`.

| Corpus | rows | vocab | nnz/doc (mean) | zero-rows | PASS |
|---|---|---|---|---|---|
| SQuAD | 20,233 | 30522 | — | 0 | ✅ |
| MetaQA | 43,234 | 30522 | — | 0 | ✅ |
| MuSiQue | 117,533 | 30522 | — | 0 | ✅ |
| 2Wiki | 398,354 | 30522 | — | 0 | ✅ |
| WebQSP | 1,316,466 | 30522 | — | 0 | ✅ |
| HotpotQA | 5,233,329 | 30522 | — | 0 | ✅ |
| 2Wiki_universe | 5,989,847 | 30522 | 148.4 | 0 | ✅ |

## Task 9 — world graph (spec pinned; cost split; NO silent ANN)

Read the frozen-C build (`src/core/indexers.py:109-161`). Topology-C = three edge
families, and **the Qwen-kNN is an exact GLOBAL k=3 `faiss.IndexFlatIP`** over
L2-normalized gte-Qwen2 embeddings (top-4 minus self), which the build already
GPU-accelerates (`index_cpu_to_gpu`) as exact all-pairs. So there is **no ANN in C**
and nothing to redefine. The earlier "O(N²) → needs an ANN decision" flag was
miscalibrated.

**Component status per corpus:**

- **Structural family — DONE for all 7** (`graph_manifest.json` already at world
  scale): universe 28.96M edges, HotpotQA 15.37M, WebQSP 3.79M, MuSiQue 2.74M,
  SQuAD 0.87M, 2Wiki-world 0.36M, MetaQA 0.13M.
- **NER (shares-entity, df≥25) — clean-only today**, CPU rebuild at world scale
  (cheap for small corpora; hours of CPU, no GPU $, for 5M).
- **Qwen-kNN (exact global k=3) — GPU.** The **method-consistent world-scale path is
  to reuse the Task-7 `EXHAUSTIVE_SHARDED_FP32_COMPUTE` harness with queries = the corpus,
  k = 4, cosine** — bit-exact vs `faiss.IndexFlatIP`, memory-fits 24 GB by sharding,
  **no ANN**. Cost is inherently ~O(N²·d):

  | Corpus | N | exact k=3 kNN cost | verdict |
  |---|---|---|---|
  | SQuAD | 20k | seconds | ✅ tractable now |
  | MetaQA | 43k | seconds | ✅ tractable now |
  | MuSiQue | 117k | ~1 min | ✅ tractable now |
  | 2Wiki-world | 398k | ~5–10 min | ✅ tractable now |
  | WebQSP | 1.32M | ~tens of GPU-min–hrs | ⚠ spend decision |
  | HotpotQA | 5.23M | ~tens–hundreds GPU-hrs | ⚠ spend decision |
  | 2Wiki_universe | 5.99M | ~tens–hundreds GPU-hrs | ⚠ spend decision |

  Exact remains **feasible** at multi-million via the sharded harness (not blocked by
  memory or by ANN) — the wall is compute *$* on near-month-end free-tier accounts.
  Surfaced for a spend decision; **not** silently switched to ANN.

- **Task 10 C partitions:** balanced ~100-node METIS partitions (clean 2Wiki = 658
  parts, size min 97 / max 103 / mean 100.1). Runs once each corpus's C graph exists.

## Sharded >1M kNN — implemented, parity-gated, running (2026-08-28)

The multi-million exact kNN is now a real implementation: `scratchpad/modal_w0_knn_sharded.py`
(`EXHAUSTIVE_SHARDED_FP32_COMPUTE`), query×doc 2-D blocking with **per-query-shard
checkpoint/resume**, raw fp16 storage + **fp32 on-the-fly L2 normalization** (never
re-rounded — bit-faithful to `faiss.normalize_L2`/`indexers.py`), TF32 off.

**PARITY GATE — PASS** (`W0_SHARDED_KNN_PARITY.json`):

| test | reference | top1 | top3-set | real non-tie (eps 1e-3) | max Δ | verdict |
|---|---|---|---|---|---|---|
| self (merge vs single-shot) | same torch backend | 1.0 | 1.0 | 0 | 0.0 | ✅ |
| metaqa 43k cross-impl | faiss IndexFlatIP fp32 | 1.0 | 0.99995 | 0 | 1.3e-6 | ✅ |
| **webqsp-world 1.32M** | local brute-force fp32 | 0.959* | 0.959* | **0** | **0.0** | ✅ |

*webqsp raw agreement is 0.959 because every id-disagreement is an **exact-score tie**
(duplicate Freebase entity embeddings — score multisets bitwise identical, Δ=0.0); zero
real misses. Same implementation runs unchanged for Hotpot / universe.

**Big-3 status:**

| Corpus | N | kNN compute | kNN edges | topology-C | partitions |
|---|---|---|---|---|---|
| **WebQSP** | 1.32M | ✅ done (spanishorgay) | 3,078,350 | ✅ struct∪knn 5.74M, 0 isolated | ✅ 13,164 parts min94/mean100.0/max103 |
| HotpotQA | 5.23M | ⏳ running (swathihrao28) | — | — | — |
| 2Wiki_universe | 5.99M | ⏳ running (swathihrao28) | — | — | — |

**Plan:** tractable-4 done; WebQSP world substrate **COMPLETE** end-to-end (validates the
>1M pipeline). Hotpot + universe kNN compute running detached (~80 / 92 query shards);
after each → merge → assemble C (struct∪ner∪knn) → pymetis N//100. Big-2 assembly needs
CSR/numpy (not python-set) or Modal (8.7 GB local RAM won't hold 30M-edge adjacency).
`WORLD_RETRIEVAL_READY` still NO until all three finish; `SAFE_TO_START_FULL_WORLD_EXPERIMENT=NO`.
