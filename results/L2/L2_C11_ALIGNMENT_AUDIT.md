# C11 Embedding Alignment Audit — REPRESENTATION ALIGNMENT (rank parity, not score equality)

**Date:** 2026-08-27 · **Datasets:** 2wiki_clean, musique_clean · **Sample:** 150 queries/dataset
(per-query rank parity) + 10 queries/dataset (full-corpus) · **No new encoder forward pass.**

## Verdict

**PASS.** The cached representations reproduce the frozen Dense ranking to (essentially exact) parity.
C11 may proceed / stands validated on the correct representation.

| Gate | Result |
|---|---|
| QUERY_EMBED_ALIGNMENT | **PASS** |
| NODE_EMBED_ALIGNMENT | **PASS** |
| DENSE_RANK_PARITY | **PASS** |
| DENSE_SCORE_IS_TRANSFORMED | **NO** (raw cosine ≈ dense_score up to float16) |
| NO_NEW_ENCODER_FORWARD | **PASS** |

## What was audited (all cached artifacts, zero encoder passes)

- cached query emb: `gte_qwen/queries_train.npy[qmap[qi]]` (what C11 uses) **and** independently
  `gte_qwen/queries_all.npy[row_all]` (canonical, from `query_meta.json`).
- cached node emb: `gte_qwen/nodes.npy`.
- L2/P50 scope + frozen dense expert: `l2_corpus/{ds}/train/{cand_ids,query_offsets,dense_rank,dense_score}.npy`
  (scope ≈ 5,000 candidates/query; frozen dense rank taken **directly** from the corpus artifact).
- frozen full-corpus retrieval: `gte_qwen/dense_top200_all.npy` (row via `row_all`).

## Per-query rank parity vs frozen dense (raw_sim = q·D over the scope)

| Metric | 2wiki_clean | musique_clean |
|---|---|---|
| Spearman(raw_sim, dense_score) mean / min | 0.99999996 / 0.99999921 | 0.99999964 / 0.99999933 |
| Kendall τ mean / min | 0.99969 / 0.99957 | 0.99972 / 0.99959 |
| Pearson(raw_sim, dense_score) mean | 0.99999982 | 0.99999982 |
| top-1 agreement | **1.00** | **1.00** |
| top-5 overlap | **1.00** | **1.00** |
| top-10 overlap | **1.00** | **1.00** |
| top-50 overlap | 0.99987 | **1.00** |

Spearman ≈ Pearson ≈ 1.0 ⇒ raw cosine and `dense_score` are the **same ranking and effectively the same
values** (no non-trivial monotone transform): `DENSE_SCORE_IS_TRANSFORMED = NO`.

## Query-embedding provenance cross-check

`qmap`-recovered query vector vs canonical `queries_all[row_all]`: **cosine = 1.0** (mean 1.0, min 1.0)
on all 150×2 sampled queries. The `qmap` permutation recovers **exactly** the canonical query row — the
C11 query representation is the frozen dense query representation, confirmed independently of the dense
scores it was originally fit against.

## Full-corpus sanity (cached-vector top-200 over ALL nodes vs frozen `dense_top200_all`)

| Overlap | 2wiki_clean | musique_clean |
|---|---|---|
| top-10 | **1.00** | **1.00** |
| top-50 | **1.00** | **1.00** |
| top-200 | **1.00** | **1.00** |

This is the strongest end-to-end check: dotting the cached query vector against **all** canonical node
embeddings reproduces the frozen Qwen dense retrieval top-200 exactly. `NODE_EMBED_ALIGNMENT = PASS`.

## Note on the `dense_rank` artifact (why an early pass mis-scored)

`l2_corpus/{ds}/train/dense_rank.npy` is a **truncated** dense ranking within scope: range −1…198,
rank 0 = best, and **−1 is a sentinel** for candidates outside the dense top-~199 (≈95% of a ~5,000
scope). A naïve `argsort(dense_rank)` sorts the −1 tail (the worst candidates) to the front and yields
0.0 overlap despite perfect score correlation. Mapping −1 → +∞ (tail last) restores parity (all top-K
overlaps 1.0). `dense_rank_eq_argsort_score_frac` = 0.90 / 0.83 (not 1.0) reflects **float16
quantization ties** reshuffling a few candidates exactly at the top-199 truncation boundary — benign,
not misalignment (the raw-value Spearman/Pearson are ≈1.0).

## Consequence for C11

The representation C11 trained on — `queries_train[qmap]` + `nodes` — **is** the frozen dense
representation (query cosine 1.0 to canonical; full-corpus top-200 identical; in-scope rank parity
≈ exact). The C11a result (C9_DEV NDCG@5 +0.0027 / RECALL@5 +0.0037 sig, replicated on DEV_INNER,
ALL@50 preserved) is therefore **not** an artifact of mismatched embeddings. Proceed with C11 exactly as
specified: cached representations only, projection MLP, query–candidate interaction, residual over C8c;
no new encoder, no cross-encoder, no LLM reranker.

**Artifacts:** `results/L2/_ctrl/_c11_align_audit.json` · script `scratchpad/_c11_align_audit.py`.
