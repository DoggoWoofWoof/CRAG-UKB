# L1 — LOCKED (frozen record)

**STATUS:** `L1_FROZEN = YES` · `L1_DOCUMENTED = YES` · `L1_RECOMPUTE_REQUIRED = NO`
Machine-readable manifest with all hashes/shapes/stats: **`results/L1/L1_LOCKED_MANIFEST.json`**.

## Architecture (locked)

| field | value |
|---|---|
| `MASTER_TOPOLOGY` | **C** = STRUCT + NER + QWEN_KNN |
| `L1_PARTITION_TOPOLOGY` | **C** (`variant_C/partition_map.json`) |
| `ROUTER` | **Dense + SPLADE** |
| `DENSE_MODEL` | Alibaba-NLP/gte-Qwen2-1.5B-instruct |
| `DENSE_EXACT_CEILING` | FAISS IndexFlatIP (top200) |
| `SPARSE` | SPLADE exact (naver/splade-cocondenser-ensembledistil, top200) |
| `FUSION` | partition-level RRF, `K0 = 60` |
| `K` | **100** |
| `P_MAIN` | **50** (~5k candidates/query; the frozen L1→L2 interface) |
| `P20` | aggressive-pruning / efficiency ablation only |
| `P100` | high-recall ceiling only |

## Why C (do NOT paraphrase as "C beats A everywhere" — that is false)

- A and C have **approximately equal aggregate L1 quality**; differences at K100/fusion/P50 are ~±1pp.
- **C wins some, A wins others** (see table below).
- Scope sizes are essentially identical (~5k candidates); reduction factor set by corpus size.
- A and C **candidate sets differ strongly**: Jaccard 0.09–0.38, ~0% identical scopes; they **mutually rescue different queries**.
- **C is the superset graph** (STRUCT+NER+QWEN_KNN) → one unified representation from which downstream stages **selectively consume** edge families. L3 need not blindly traverse all C edges; families can be masked/scored separately.
- ∴ C is selected for **ARCHITECTURAL CONSISTENCY + FUTURE L2/L3 SIGNAL AVAILABILITY**, not universal L1 dominance.

## Canonical operating point — C, Dense+SPLADE, K100, P50 (primary metric)

| dataset | primary | **C** primary | A primary (ref) | ANY | ALL | mean scope | reduction |
|---|:--:|--:|--:|--:|--:|--:|--:|
| 2wiki_clean | ALL | **93.60** | 93.27 | 100.0 | 93.60 | 5029 | 13.1× |
| musique_clean | ALL | **96.14** | 94.99 | 99.95 | 96.14 | 5042 | 2.71× |
| hotpotqa_clean | ALL | 92.20 | **93.48** | 98.92 | 92.20 | 5081 | 99.9× |
| squad_clean | ANY | 97.95 | 97.96 | 97.95 | 97.95 | 5039 | 3.78× |
| metaqa †LEGACY | ANY | 96.52 | 95.52 | 96.52 | 61.76 | 5038 | 7.97× |
| webqsp †LEGACY | ANY | 94.30 | **95.31** | 94.30 | 76.24 | 5023 | 155.6× |

Full 648-cell sweep (6 datasets × 108 cells) is preserved and **not recomputed** — see `L1_FULL_TABLE.md`, `L1_STOP_OUTPUTS.md`, and `phase1_<dataset>.json`.

## P50 decision

P20 tested and too aggressive as the L1→L2 boundary. At K100/fusion, **P20→P50 recovered ~5–10pp of primary gold coverage for ~2.5× candidate scope**. `P_MAIN = 50` is canonical. **Do not retune P during L2 development.**

## ANY vs ALL (keep explicit)

- `ANY_GOLD_COV` = ≥1 gold survives L1. `ALL_GOLD_COV` = all supporting golds survive.
- High ANY / lower ALL = partial evidence: L2/L3 may still progress from one strong surviving anchor.
- **Under strict induced-P50, a gold pruned entirely by L1 is UNRECOVERABLE.** Distinguish (A) within-scope reasoning failure from (B) L1 pruning failure. L3 must **not** silently reintroduce the full corpus. Bounded frontier expansion outside P50 = explicit **future** L3 experiment.

## Datasets (exact)

| dataset | N_docs | N_all_queries | N_eval | eval population | primary | C status |
|---|--:|--:|--:|---|:--:|:--:|
| 2wiki_clean | 65 865 | 15 000 | 1 500 | DERIVED_TEST | ALL | CANONICAL |
| musique_clean | 13 672 | 19 938 | 1 995 | DERIVED_TEST | ALL | CANONICAL |
| hotpotqa_clean | 507 494 | 97 852 | 9 786 | DERIVED_TEST | ALL | CANONICAL |
| squad_clean | 19 029 | 130 319 | 13 033 | DERIVED_TEST | ANY | CANONICAL |
| metaqa | 40 151 | 407 513 | 39 093 | NATIVE_TEST | ANY | **LEGACY_EXPLORATORY** |
| webqsp | 781 485 | 1 578 | 1 578 | ALL_EVALUABLE | ANY | **LEGACY_EXPLORATORY** |

## Integrity

`CACHE_INTEGRITY = PASS` · `EDGE_FAMILY_RECONSTRUCTION = PASS` · `C_PARITY = PASS` (C == A ∪ NER, 4 text substrates, bit-exact) · `B_PARITY = PASS` · `ALIGNMENT_GATE = PASS` (eng.nodes order == load_nodes doc order) · `STRUCT ∩ KNN = 0` (A = STRUCT ⊔ KNN). Per-file sha256 (partition maps, C graphs, query-id caches, dense/splade `.npy` with shape+dtype, NER pkls, result JSONs, scripts) recorded in the manifest.

## Caveats

1. **C is architecturally selected; C is NOT a universal empirical L1 winner** (A wins hotpot/webqsp; C wins 2wiki/musique; squad tie).
2. **MetaQA/WebQSP B/C NER variants remain LEGACY_EXPLORATORY** — their entity-node substrate makes current NER provenance noncanonical. Canonical L1 for metaqa = SPLADE-alone (96.32), webqsp = A-fusion (95.31). C partition_maps exist for them but are not the canonical operating topology.
3. **Hotpot/SQuAD A/C graph diagnostic used strided sampling** (2500 of 9786/13033); 2wiki/musique full-N. The 648-cell L1 sweep itself is full-population.

## Freeze rule

After this lock, do **NOT**: rerun the 648 L1 cells · rebuild current C topology · recompute Dense caches · recompute SPLADE caches · re-encode Qwen documents · retune K/P · reopen A/C over tiny L1 differences — **unless an actual correctness bug is found.**
