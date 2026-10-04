# G1 — Cross-Dataset Generalization: the TEXT conclusion (SQuAD + HotpotQA)

**Question.** Does the frozen C11a text-pilot retrieval *principle* generalize to new
text distributions? We separate two orthogonal questions per target:

- **PARAMETER generalization** — `ZERO_SHOT_C11A`: the exact source-trained C11a
  weights + source E-standardization, applied to the target with **no fitting**.
- **ARCHITECTURE generalization** — `REFIT_C11A`: the *same* architecture / hparams /
  loss / optimizer, retrained on **target TRAIN** (cap 25 000), selected on target VAL.

**Locked architecture (unchanged across targets).** L1 topology **C / P50**
(Dense+SPLADE partition-RRF, K0=60 / K=100 / P_MAIN=50) → 5 frozen experts
{Dense, SPLADE, universal Offset, universal Mixture, Masked-Relation} → **C7b** soft-archetype
fusion → **C8c** XGBRanker → **C11a** interaction-MLP residual. **No** dataset-ID, LLM,
cross-encoder, Transformer, attention, or GNN. Source backbone = `2wiki_clean + musique_clean`.
**VAL only — the target TEST split was never inspected.**

## Main table (target VAL)

| System | SQuAD NDCG@5 | R@5 | MRR | HotpotQA NDCG@5 | R@5 | ALL@5 | MRR |
|---|---|---|---|---|---|---|---|
| C7b fusion | 0.7941 | 0.8645 | 0.7793 | 0.7368 | 0.7792 | 0.6112 | 0.8414 |
| **C8c** (baseline) | 0.7958 | 0.8705 | 0.7810 | 0.7735 | 0.8099 | 0.6670 | 0.8960 |
| ZERO_SHOT_C11A | 0.7868 | 0.8706 | 0.7682 | 0.7627 | 0.7995 | 0.6464 | 0.8825 |
| **REFIT_C11A** | **0.8229** | **0.9085** | **0.7995** | **0.8508** | **0.8918** | **0.8148** | **0.9101** |

*(SQuAD n_val 25 489 / refit-train 24 456; HotpotQA n_val 19 376 / refit-train 24 751.
HotpotQA is 2-hop — every query has 2 golds, so ALL@5 = both golds in top-5.)*

## The two questions, answered — with paired-bootstrap significance

**PARAMETER transfer FAILS on both** (ZERO_SHOT_C11A vs C8c, all significant):

| Δ vs C8c | SQuAD | HotpotQA |
|---|---|---|
| NDCG@5 | **−0.0090** (sig) | **−0.0107** (sig) |
| R@5 | +0.0001 (ns) | **−0.0104** (sig) |
| ALL@5 | — | **−0.0205** (sig) |
| MRR | **−0.0127** (sig) | **−0.0135** (sig) |

The source-trained C11a **weights** do not transfer — they are at best neutral (SQuAD
recall) and mostly a significant *regression* on the target.

**ARCHITECTURE transfer GENERALIZES on both** (REFIT_C11A vs C8c, all significant):

| Δ vs C8c | SQuAD | HotpotQA |
|---|---|---|
| NDCG@5 | **+0.0271** [+0.024, +0.030] | **+0.0773** [+0.075, +0.080] |
| R@5 | **+0.0380** [+0.035, +0.041] | **+0.0819** [+0.078, +0.085] |
| ALL@5 | — | **+0.1478** [+0.142, +0.154] |
| MRR | **+0.0186** [+0.016, +0.021] | **+0.0142** [+0.012, +0.017] |

The *same* C11a design, retrained on the target, significantly beats the strong C8c
baseline on every metric — and the lift is **larger on HotpotQA** (the harder 2-hop set).

## Guardrails (both datasets)

- **Deep recall preserved identically** across all four systems — C11a reranks strictly
  within the top-50 (SQuAD ALL@50 = 0.9621; HotpotQA ALL@50 = 0.9907, equal for every system).
- **HotpotQA L1 ceiling (val):** GOLD_RECALL_IN_SCOPE 0.9567, ALL-in-scope 0.9232 —
  the P50 front-end is not the bottleneck; the win is in the rerank.
- **Relation expert is informative when present (Hotpot val):** 84.7% of queries carry a
  relation signal, it touches a gold on 32.7%, and when a gold is relation-eligible its
  median rank is 0 (51.8% at rank 0, 92.9% in top-5).
- **Integrity:** target TEST untouched; **0** new encoder forward passes; **0** new LLM
  components; both datasets report `INTERPRETATION = ARCHITECTURE_GENERALIZES`.

## Verdict

> **The C11a interaction-MLP is a transferable retrieval *architecture* across text
> distributions; its learned *parameters* are distribution-specific.** Zero-shot weight
> transfer is significantly negative on both SQuAD and HotpotQA; the same architecture
> refit on the target is significantly positive on both, with a bigger margin on the
> harder multi-hop set. Deep recall is untouched — the effect is a pure top-5 rerank.

**Next:** STOP here for review of the TEXT conclusion before the KB targets
(WebQSP → MetaQA). No target TEST, no L3, no architecture change in scope.
