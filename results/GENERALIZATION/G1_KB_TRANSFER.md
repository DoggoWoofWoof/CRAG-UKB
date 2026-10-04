# G1 — KB Transfer (WebQSP → MetaQA)

Same **frozen** architecture as the text pilots (L1-C/P50 → 5 experts → C7b → C8c → C11a). **No path/graph added. No architecture change. VAL only; TARGET TEST NEVER TOUCHED. 0 encoder passes, 0 LLM.**


## WebQSP

n_val = 301. Interpretation: **ARCHITECTURE_GENERALIZES**.

| Model | NDCG@5 | GR@5 | ANY@5 | ALL@5 | MRR | NDCG@50 | ALL@50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| C7b_fusion | 0.0893 | 0.0937 | 0.1528 | 0.0598 | 0.1212 | 0.1593 | 0.2525 |
| C8c | 0.0847 | 0.0924 | 0.1462 | 0.0598 | 0.1190 | 0.1609 | 0.2525 |
| ZERO_SHOT_C11A | 0.0863 | 0.0949 | 0.1528 | 0.0598 | 0.1194 | 0.1611 | 0.2525 |
| REFIT_C11A | 0.1924 | 0.1970 | 0.2890 | 0.1362 | 0.2317 | 0.2250 | 0.2525 |

**Failure split (REFIT_C11A, 1367 in-scope golds):** top5=143, present-but-below-top20=1162, below-top50=1005 (73.5%).


Golds are IN the P50 scope but L2 deep-recall is the bottleneck: 1005/1367 (73.5%) in-scope golds never enter the top-50. REFIT_C11A roughly doubles top-5 golds vs C8c (143 vs 67) => ARCHITECTURE_GENERALIZES, but the absolute ceiling (ALL@50=0.2525) is fixed upstream by the dense+splade experts' inability to surface KB-entity golds into the deep list. This is a REPRESENTATION_SHIFT at the expert level, not an L2 fusion/ranking failure.


## MetaQA (per hop — the critical breakdown)

| Hop | P50 ANY | P50 ALL | REFIT NDCG@5 | REFIT ANY@5 | REFIT ALL@5 | REFIT MRR | ALL@50 | L1-missing golds |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1-hop | 0.9985 | 0.9959 | 0.6940 | 0.7578 | 0.6256 | 0.7235 | 0.6929 | 77 |
| 2-hop | 0.9631 | 0.7138 | 0.0861 | 0.1516 | 0.0515 | 0.1264 | 0.0902 | 25872 |
| 3-hop | 0.9468 | 0.2701 | 0.1061 | 0.1946 | 0.0477 | 0.1772 | 0.0537 | 111398 |

1-hop is a clean architecture win (REFIT NDCG@5 0.694, ANY@5 0.758, ALL@5 0.626; P50 ANY/ALL cov 99.85/99.59). 2-hop and 3-hop COLLAPSE, and the mechanism is UPSTREAM SCOPE, not L2 ranking: P50 ANY coverage stays high (96.3/94.7%) but P50 ALL coverage falls to 71.4/27.0% — the multi-hop chain endpoints are each individually reachable yet not CO-SCOPED into the same ~5k P50 window, so ALL@k is unattainable at any rerank depth (3-hop ALL@50 is only 0.0537). The REFIT failure decomp confirms the residual is not ranking: at 3-hop only 210 golds are top20-but-below-top5 vs 73216 in-scope-below-top20 and 111398 L1-missing.


**MISSING_RELATIONAL_STRUCTURE:** LIKELY (2-hop and 3-hop): evidence present-but-not-co-scoped; recorded, NOT repaired (no path/graph added, per authorization).


**Pooled-interpretation caveat:** Pooled INTERPRETATION=PARAMETERS_GENERALIZE is an ARTIFACT of hop-mixing: ZERO_SHOT beats C8c only marginally (NDCG@5 +0.0041) because 2-/3-hop are near-floor for every system so source weights cannot hurt, while 1-hop shares the source offset structure. The genuine, large, consistent lever everywhere is REFIT (architecture), not parameter reuse. Read PER-HOP, not pooled.


## Cross-cutting conclusion

PARAMETER transfer does not hold as a real effect on KB (WebQSP neutral; MetaQA pooled +0.0041 is a hop-mixing artifact). ARCHITECTURE transfer (REFIT_C11A) is the consistent lever across BOTH KB targets exactly as on text: it significantly beats C8c on WebQSP (NDCG@5 0.0847->0.1924) and on MetaQA-1hop (0.323->0.694). The KB ceiling is set UPSTREAM (L1/P50 scope + expert representation), not by the L2 reranker: WebQSP 73.5% of in-scope golds never reach top-50, and MetaQA multi-hop ALL-coverage collapses with hop-count. G1 verdict matches the text pilots: the C11a interaction-MLP ARCHITECTURE is a transferable retrieval principle; its PARAMETERS are distribution-specific.
