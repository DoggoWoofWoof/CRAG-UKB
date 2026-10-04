# CRAG benchmark state (2026-08-18)

The single source of truth for the paper's comparison tables. All numbers are reader-free retrieval
**Recall@k over gold nodes** unless a row is explicitly QA. CRAG = frozen gte-Qwen2-1.5B, **no LLM**.

Interactive version: the `canonical_benchmarks.html` artifact. Machine-readable: `baselines/full_matrix.json`,
`results/L2/unified_metrics.json`.

## 1. The one metric, all six datasets (open-corpus, joint head)

| Dataset | gold | R@5 | R@20 | R@50 | hit@5 | hit@50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MuSiQue | passages | 74.6 | 89.5 | 93.7 | 96.4 | 99.8 |
| HotpotQA | passages | 75.0 | 82.5 | 86.4 | 93.2 | 97.4 |
| 2Wiki | passages | 75.2 | 81.5 | 83.4 | 99.5 | 100 |
| SQuAD | passage | 89.7 | 96.4 | 98.1 | 89.7 | 98.1 |
| MetaQA | KB entities | 68.8 | 74.0 | 77.5 | 84.3 | 88.9 |
| WebQSP | KB entities | 29.7 | 45.6 | 57.0 | 48.4 | 76.1 |

One `_recall` function scores all rows; gold type is the only per-dataset difference. `hit@k` ≫ `R@k`
means coverage is high but ranking has headroom — **the L3 tuning (n_seed 10→2, w 0.2→3.0) targets exactly
this gap: +3.3 mean R@5, MetaQA +12.1, 2Wiki/HotpotQA/WebQSP +3** (small strong-dense dips on MuSiQue/SQuAD
that recover by @20). See LEVEL3_README.

## 2. Centralized SOTA comparison — Recall@5 (HippoRAG-2-paper corpora)

| System | mean R@5 | compute |
| --- | ---: | --- |
| HippoRAG 2 | 87.1 | 7B dense + 70B LLM |
| **CRAG (trained)** | **82.4** | **1.5B, no LLM** |
| GFM-RAG | 80.3 | GNN + GPT-4o |
| NV-Embed-v2 | 80.2 | 7B dense |
| **CRAG (zero-shot)** | **74.0** | **1.5B, no LLM, no training** |
| HippoRAG-v1 | 73.6 | KG+PPR + 70B LLM |
| RAPTOR | 70.3 | 70B LLM |
| GTR / Contriever | 63.6 / 59.8 | dense |

CRAG is **#2 of 9**, behind only the 70B-LLM SOTA, at ~1/50 the model footprint. Zero-shot already
beats the two 70B-LLM systems below it.

## 3. KB — anchored (KGQA-comparable) answer-entity coverage, WebQSP

| Retriever (topic-entity-anchored) | coverage | uses |
| --- | ---: | --- |
| SubgraphRAG | 86.5 | MLP + GPT-4o supervision |
| **CRAG (full pipeline, hit@50)** | **80.5** | **1.5B, no LLM** |
| cosine | 71.9 | dense |
| GNN-RAG | 40.5 | GNN |
| RoG | 38.8 | LLM paths |

CRAG beats GNN-RAG and RoG decisively, trails only the GPT-4o-supervised SubgraphRAG (budget caveat:
CRAG @50 entities vs their @100 triples). File: `results/L2/webqsp_anchored_fullstrength.json`.

## 4. QA (secondary — reader-confounded, internal ablation only)

Fixed reader Qwen2.5-1.5B on CRAG top-5 (EM/F1): MuSiQue 20.0/28.4 · 2Wiki 41.4/43.9 · SQuAD 26.2/34.1
· MetaQA 40.8/60.4 · HotpotQA 46.0/53.2 · WebQSP 38.4/52.8. **Not comparable** to published F1 (70B /
GPT-4o readers) or KGQA Hits@1 (topic oracle + big reader) — the gaps are reader-size + oracle, not
retrieval. Use only to measure CRAG's own end-to-end deltas (e.g. NER-L3's effect).

## 5. Why the table has N/As (centralization finding)

The field shares **no common metric**: retrievers report Recall/coverage, RAPTOR & SiReRAG report *only*
QA F1 ("retrieval recall unfair, different pool"), KGQA reports Hits@1, KG2RAG reports "Hit Rate", and
every QA number uses a different reader. A fully dense cross-system table is therefore impossible from
published numbers, and cross-family cells (a Freebase KGQA system on HotpotQA, a text retriever on
Freebase) do not apply without redesign. The centralized comparison is **reader-free retrieval Recall**
for the retriever class (§2, §3); other cells are honest N/As. Disposition of every cell:
`baselines/full_matrix.json`.

## 6. Gap summary (CRAG's real gaps)

- **−4.7** to the 70B-LLM SOTA on multi-hop Recall (the headline gap; cheap to defend on cost).
- **2Wiki −13** vs GFM-RAG — the one genuine retrieval soft spot (KG-structured), the target to improve.
- KB anchored **−6** (budget-caveated) to GPT-4o-supervised SubgraphRAG, while **+24–26 over** the
  non-supervised KB retrievers.
- QA / Hits@1 gaps are **not retrieval gaps** — reader + oracle artifacts.
