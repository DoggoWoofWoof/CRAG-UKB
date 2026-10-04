# G1 — Cross-Dataset Generalization Matrix

**Source backbone:** 2wiki_clean + musique_clean. **VAL only; TARGET TEST NEVER TOUCHED.** No new encoder passes, no LLM.

Frozen architecture: L1 topology-C / P50 → 5 frozen experts → C7b fusion → C8c XGBRanker → **C11a interaction-MLP** (the transfer object).

Two questions: **ZERO_SHOT_C11A** = source weights, no target fit (PARAMETER transfer). **REFIT_C11A** = same arch, target-trained ≤25k (ARCHITECTURE transfer).


## Main table (NDCG@5 / GoldRecall@5 / ANY@5 / ALL@5 / MRR / ALL@50)

| Dataset | Kind | Model | NDCG@5 | GR@5 | ANY@5 | ALL@5 | MRR | ALL@50 |
|---|---|---|--:|--:|--:|--:|--:|--:|
| 2Wiki (source) | SOURCE_TEXT | C8c | 0.8834 | 0.9044 | 0.9990 | 0.7617 | 0.9597 | 0.9217 |
| 2Wiki (source) | SOURCE_TEXT | REFIT_C11A | 0.9021 | 0.9184 | 0.9977 | 0.7943 | 0.9653 | 0.9217 |
|  |  |  |  |  |  |  |  |  |
| MuSiQue (source) | SOURCE_TEXT | C8c | 0.8816 | 0.9019 | 0.9905 | 0.7887 | 0.9548 | 0.9548 |
| MuSiQue (source) | SOURCE_TEXT | REFIT_C11A | 0.8834 | 0.9023 | 0.9915 | 0.7910 | 0.9545 | 0.9548 |
|  |  |  |  |  |  |  |  |  |
| SQuAD | TARGET_TEXT | C8c | 0.7958 | 0.8705 | — | 0.8705 | 0.7810 | 0.9621 |
| SQuAD | TARGET_TEXT | ZERO_SHOT_C11A | 0.7868 | 0.8706 | — | 0.8706 | 0.7682 | 0.9621 |
| SQuAD | TARGET_TEXT | REFIT_C11A | 0.8229 | 0.9085 | — | 0.9085 | 0.7995 | 0.9621 |
|  |  |  |  |  |  |  |  |  |
| HotpotQA | TARGET_TEXT | C8c | 0.7735 | 0.8099 | 0.9528 | 0.6670 | 0.8960 | 0.9907 |
| HotpotQA | TARGET_TEXT | ZERO_SHOT_C11A | 0.7627 | 0.7995 | 0.9526 | 0.6464 | 0.8825 | 0.9907 |
| HotpotQA | TARGET_TEXT | REFIT_C11A | 0.8508 | 0.8918 | 0.9688 | 0.8148 | 0.9101 | 0.9907 |
|  |  |  |  |  |  |  |  |  |
| WebQSP | TARGET_KB | C8c | 0.0847 | 0.0924 | 0.1462 | 0.0598 | 0.1190 | 0.2525 |
| WebQSP | TARGET_KB | ZERO_SHOT_C11A | 0.0863 | 0.0949 | 0.1528 | 0.0598 | 0.1194 | 0.2525 |
| WebQSP | TARGET_KB | REFIT_C11A | 0.1924 | 0.1970 | 0.2890 | 0.1362 | 0.2317 | 0.2525 |
|  |  |  |  |  |  |  |  |  |
| MetaQA-1hop | TARGET_KB | C8c | 0.3229 | 0.4787 | 0.5733 | 0.4031 | 0.3087 | 0.6929 |
| MetaQA-1hop | TARGET_KB | ZERO_SHOT_C11A | 0.3306 | 0.4676 | 0.5589 | 0.3962 | 0.3233 | 0.6929 |
| MetaQA-1hop | TARGET_KB | REFIT_C11A | 0.6940 | 0.6939 | 0.7578 | 0.6256 | 0.7235 | 0.6929 |
|  |  |  |  |  |  |  |  |  |
| MetaQA-2hop | TARGET_KB | C8c | 0.0260 | 0.0385 | 0.0759 | 0.0241 | 0.0437 | 0.0902 |
| MetaQA-2hop | TARGET_KB | ZERO_SHOT_C11A | 0.0258 | 0.0399 | 0.0767 | 0.0259 | 0.0428 | 0.0902 |
| MetaQA-2hop | TARGET_KB | REFIT_C11A | 0.0861 | 0.0788 | 0.1516 | 0.0515 | 0.1264 | 0.0902 |
|  |  |  |  |  |  |  |  |  |
| MetaQA-3hop | TARGET_KB | C8c | 0.0391 | 0.0533 | 0.1101 | 0.0334 | 0.0592 | 0.0537 |
| MetaQA-3hop | TARGET_KB | ZERO_SHOT_C11A | 0.0450 | 0.0573 | 0.1174 | 0.0359 | 0.0657 | 0.0537 |
| MetaQA-3hop | TARGET_KB | REFIT_C11A | 0.1061 | 0.0840 | 0.1946 | 0.0477 | 0.1772 | 0.0537 |
|  |  |  |  |  |  |  |  |  |

## Interpretation per row

| Dataset | Kind | Interpretation | Deep-recall ceiling (ALL@50) |
|---|---|---|--:|
| 2Wiki (source) | SOURCE_TEXT | IN_DISTRIBUTION (C11a fitted here) | 0.9217 |
| MuSiQue (source) | SOURCE_TEXT | IN_DISTRIBUTION (C11a fitted here) | 0.9548 |
| SQuAD | TARGET_TEXT | ARCHITECTURE_GENERALIZES | 0.9621 |
| HotpotQA | TARGET_TEXT | ARCHITECTURE_GENERALIZES | 0.9907 |
| WebQSP | TARGET_KB | ARCHITECTURE_GENERALIZES | 0.2525 |
| MetaQA-1hop | TARGET_KB | PER_HOP | 0.6929 |
| MetaQA-2hop | TARGET_KB | PER_HOP | 0.0902 |
| MetaQA-3hop | TARGET_KB | PER_HOP | 0.0537 |

## KB scope diagnostics (why deep recall is the ceiling)

MetaQA per-hop P50 coverage — ANY stays high, ALL collapses with hop-count (multi-hop chain endpoints reachable individually but not co-scoped):

| Hop | P50 ANY cov | P50 ALL cov | golds expected | golds in P50 | L1-missing golds | REFIT NDCG@5 | REFIT ALL@50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 1-hop | 0.9985 | 0.9959 | 18985 | 18908 | 77 | 0.6940 | 0.6929 |
| 2-hop | 0.9631 | 0.7138 | 93898 | 68026 | 25872 | 0.0861 | 0.0902 |
| 3-hop | 0.9468 | 0.2701 | 188097 | 76699 | 111398 | 0.1061 | 0.0537 |

WebQSP (n_val=301): of 1367 in-P50-scope golds, 1005 (73.5%) rank below top-50 even after rerank — i.e. present in the ~5k P50 scope but never lifted into the deep top-50. REFIT top-5 golds = 143 (vs C8c 67). REFIT ALL@50 = 0.2525 — the deep-recall ceiling is set upstream at L1/P50, not by L2 ranking. (P50_SCOPE_COVERAGE raw/mapped keys are absent from WebQSP query_meta, so only the decomposition above is authoritative.)
