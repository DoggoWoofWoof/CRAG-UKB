# L1 / Phase-1 — STOP OUTPUTS (verified 2026-08-25)

Exact ceiling only (no ANN/HNSW/IVF/PQ). Dense = gte-Qwen2-1.5B FAISS IndexFlatIP top200; sparse = SPLADE exact top200.
Eval sweep = 108 cells/dataset (K{25,50,100,200} x router{dense,splade,dense+splade} x topo{A,B,C} x P{20,50,100}).
Source JSONs: results/L1/phase1_<ds>.json. All numbers below re-extracted directly from those files.

## 1. Best config @ K100/P50 (primary metric per dataset)

| dataset | n_test | topo | metric | dense | splade | fusion | winner | fusion-dense |
|---|--:|:--:|---|--:|--:|--:|--:|--:|
| 2wiki | 1500 | B | ALL | 90.67 | 93.47 | 94.07 | **94.07 (fusion)** | +3.4 |
| musique | 1995 | C | ALL | 95.24 | 93.73 | 96.14 | **96.14 (fusion)** | +0.9 |
| webqsp | 1578 | A | ANY | 93.28 | 92.14 | 95.31 | **95.31 (fusion)** | +2.03 |
| hotpot | 9786 | A | ALL | 91.67 | 90.88 | 93.48 | **93.48 (fusion)** | +1.81 |
| squad | 13033 | A | ANY | 97.12 | 97.2 | 97.96 | **97.96 (fusion)** | +0.84 |
| metaqa | 39093 | A | ANY | 89.84 | 96.32 | 95.52 | **96.32 (splade)** | +5.68 |

## 2. Full K100/P50 grid (all topologies, ANY + ALL)

| dataset | topo | ANY dense/splade/fusion | ALL dense/splade/fusion | mean_scope(fusion) | reduction |
|---|:--:|---|---|--:|--:|
| 2wiki | A | 100.0/99.93/100.0 | 90.33/92.53/93.27 | 5028 | 13.1x |
| 2wiki | B | 99.87/99.93/100.0 | 90.67/93.47/94.07 | 5044 | 13.1x |
| 2wiki | C | 100.0/100.0/100.0 | 90.27/93.4/93.6 | 5029 | 13.1x |
| musique | A | 99.85/99.85/99.95 | 93.73/92.78/94.99 | 5041 | 2.7x |
| musique | B | 99.9/99.8/99.9 | 93.68/91.83/94.94 | 5044 | 2.7x |
| musique | C | 99.95/99.9/99.95 | 95.24/93.73/96.14 | 5042 | 2.7x |
| webqsp | A | 93.28/92.14/95.31 | 75.03/72.94/79.28 | 5027 | 155.4x |
| webqsp | B | 93.22/90.62/94.3 | 73.89/70.85/76.43 | 5058 | 154.5x |
| webqsp | C | 92.33/91.25/94.3 | 72.75/68.88/76.24 | 5023 | 155.6x |
| hotpot | A | 98.55/98.9/99.09 | 91.67/90.88/93.48 | 5078 | 100.0x |
| hotpot | B | 98.0/98.39/98.81 | 89.42/89.01/91.46 | 5083 | 99.8x |
| hotpot | C | 98.03/98.54/98.92 | 89.83/89.83/92.2 | 5081 | 99.9x |
| squad | A | 97.12/97.2/97.96 | 97.12/97.2/97.96 | 5052 | 3.8x |
| squad | B | 96.54/96.48/97.34 | 96.54/96.48/97.34 | 5030 | 3.8x |
| squad | C | 97.15/97.16/97.95 | 97.15/97.16/97.95 | 5039 | 3.8x |
| metaqa | A | 89.84/96.32/95.52 | 50.46/62.75/59.65 | 5024 | 8.0x |
| metaqa | B | 90.09/96.51/95.8 | 51.99/64.3/61.39 | 5031 | 8.0x |
| metaqa | C | 91.27/97.1/96.52 | 52.12/64.79/61.76 | 5038 | 8.0x |

## 3. K sensitivity (topo A, fusion, P50) — ANY / ALL

| dataset | K25 | K50 | K100 | K200 |
|---|---|---|---|---|
| 2wiki | 100.0/91.47 | 100.0/92.4 | 100.0/93.27 | 100.0/93.4 |
| musique | 99.9/93.98 | 100.0/94.79 | 99.95/94.99 | 99.95/94.94 |
| webqsp | 87.77/67.62 | 93.47/76.3 | 95.31/79.28 | 96.13/80.29 |
| hotpot | 99.28/93.59 | 99.12/93.64 | 99.09/93.48 | 99.06/93.25 |
| squad | 97.96/97.96 | 97.99/97.99 | 97.96/97.96 | 97.86/97.86 |
| metaqa | 95.8/60.49 | 95.49/59.8 | 95.52/59.65 | 95.46/59.56 |

## 4. P sensitivity (topo A, fusion, K100) — ANY / ALL / reduction

| dataset | P20 | P50 | P100 |
|---|---|---|---|
| 2wiki | 99.8/87.6 (33x) | 100.0/93.27 (13x) | 100.0/95.27 (7x) |
| musique | 99.3/85.71 (7x) | 99.95/94.99 (3x) | 100.0/98.95 (1x) |
| webqsp | 88.28/65.15 (388x) | 95.31/79.28 (155x) | 97.34/86.76 (78x) |
| hotpot | 97.46/83.83 (249x) | 99.09/93.48 (100x) | 99.58/96.3 (50x) |
| squad | 92.43/92.43 (9x) | 97.96/97.96 (4x) | 99.44/99.44 (2x) |
| metaqa | 88.95/49.18 (20x) | 95.52/59.65 (8x) | 98.25/68.08 (4x) |

## 5. MetaQA per-hop breakdown (topo A, fusion, K100/P50)

| hop | N | ANY | ALL |
|---|--:|--:|--:|
| 1-hop | 9947 | 99.78 | 99.53 |
| 2-hop | 14872 | 94.63 | 66.93 |
| 3-hop | 14274 | 93.48 | 24.27 |

## 6. Verdict

- L1_PIPELINE_CORRECT = YES; CACHE_INTEGRITY = PASS; n_test verified per dataset.
- DENSE_SPLADE_BETTER = YES everywhere (fusion beats dense on primary metric); metaqa winner = splade-alone (KB lexical).
- BEST_TOPOLOGY: 2wiki=B, musique=C, webqsp/hotpot/squad=A, metaqa=A (B/C LEGACY_EXPLORATORY for KB entity nodes).
- BEST_K/P = 100/50 (universal knee).
- SPLADE_SCORING_BACKEND = torch_sparse_gpu (exact; GPU-parity 1.00000 vs scipy; scipy-CPU auto-fallback).
- All caches local: 2wiki_clean, musique_clean, webqsp, hotpotqa_clean, squad_clean, metaqa (dense/splade top200 + queries_all).