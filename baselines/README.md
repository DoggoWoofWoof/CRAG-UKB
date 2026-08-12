# Baseline graph-RAG systems — C-RAG head-to-head

Staging for the paper comparison (Next Steps #4). Every established system below builds its graph with
an **LLM** and/or reasons with a **GNN**; C-RAG uses neither. The head-to-head must therefore report
**indexing cost and latency** alongside retrieval quality, since our differentiator is competitive
quality with no LLM in the loop.

## Systems (all cloned under `baselines/repos/`, gitignored — code confirmed present)

| System | Official repo | Venue | Graph construction | Reasoning core | LLM? | GNN? |
|---|---|---|---|---|---|---|
| HippoRAG 2 | `OSU-NLP-Group/HippoRAG` (arXiv 2502.14802) | 2025 | LLM OpenIE triples | Personalized PageRank | yes | no |
| GFM-RAG | `RManLuo/gfm-rag` | NeurIPS'25 | LLM (GPT-4o) extraction | **GNN foundation model** (dataset-agnostic) | yes | yes |
| HopRAG | `LIU-Hao-2002/HopRAG` | ACL Findings'25 | passage graph | LLM logic-aware multi-hop walk | yes | no |
| SiReRAG | `SalesforceAIResearch/SiReRAG` | ICLR'25 | similar + related trees (RAPTOR-based) | dual similarity/relatedness retrieval | yes | no |
| KG2RAG | `nju-websoft/KG2RAG` | NAACL'25 | KG-guided expansion | KG chains + rerank | yes | no |

Confirmed by inspection: HippoRAG's repo *is* HippoRAG 2 (uses `OpenIE`); GFM-RAG ships a 302-line
retriever + graph_indexer; HopRAG/SiReRAG/KG2RAG each ship runnable eval scripts (SiReRAG also bundles
notebooks + corpus/KG JSONs; KG2RAG ships `code/` + data + run commands).

`GFM-RAG` is the closest competitor to our **dataset-agnostic** claim — but it is dataset-agnostic
*via a GNN over an LLM-built KG*. C-RAG is dataset-agnostic with neither (§4.8: a GNN underperforms
dense/offset and does not scale to our 500k+-node graphs).

## Head-to-head protocol (directly comparable)

1. **Corpora:** the HippoRAG-standard **1,000-question candidate corpora** — MuSiQue 11,656 / 2Wiki 6,119 /
   HotpotQA 9,811 — reserved at `results/L2/_hpr_paper_backup/`. **HippoRAG 2 and SiReRAG use exactly
   these corpora and metrics** (SiReRAG bundles `*_corpus.json`), so the comparison is apples-to-apples.
2. **Retrieval metric:** Recall@2 / Recall@5 (HippoRAG's protocol). C-RAG entry = the full pipeline
   (dense + SPLADE candgen → best-of fusion rerank → traversal), frozen gte-Qwen2 substrate.
3. **Downstream:** EM / F1 with a **matched reader** (same generator + prompt for all), once the C-RAG
   end-to-end QA path (Next Steps #1) is wired.
4. **Cost columns (the point):** per-corpus **indexing cost** (LLM tokens / $ for OpenIE/GPT-4o
   construction — zero for C-RAG) and **query latency**, so "competitive quality, no LLM" is quantified.

## Setup

Each baseline runs in its own venv (conflicting torch / LLM-client pins). The LLM-based ones need an
OpenAI (or local-LLM) key **for graph construction** — the step C-RAG removes; track its token cost.

```bash
python -m venv baselines/.venv-hipporag && baselines/.venv-hipporag/bin/pip install -r baselines/repos/HippoRAG/requirements.txt
# build KG on our _hpr corpora -> run retrieval -> export per-query recall -> compare vs results/L2/_hpr_paper_backup
```

## Head-to-head vs HippoRAG 2 (measured, no LLM) — `results/L2/hpr_headtohead.json`

Our full pipeline on the reserved `_hpr` corpora (100-query test split each), Recall@5, vs HippoRAG 2's
**published** numbers. Ours: frozen gte-Qwen2-**1.5B** encoder, **zero LLM**. This **supersedes an earlier
draft that compared against HippoRAG v1** (51.9 / 89.1) — those are obsolete; HippoRAG 2 is the live target.

| R@5 | MuSiQue | 2Wiki | HotpotQA | mean | compute |
|---|---|---|---|---|---|
| Our dense (gte-1.5B) | 56.7 | 70.2 | 88.5 | 71.8 | 1.5B enc, no LLM |
| **Ours: multi-task head + fusion + L3** | **68.2** | **79.2** | **91.0** | **79.5** | 1.5B enc, no LLM |
| NV-Embed-v2 dense | 69.7 | 76.5 | 94.5 | 80.2 | **7B** enc, no LLM |
| **HippoRAG 2** (full) | 74.7 | 90.4 | 96.3 | 87.1 | 7B enc **+ Llama-70B** |

Honest reading (this is not a "we win" table):
- **We do not beat HippoRAG 2** (87.1). It uses a **7B** embedder + a **70B-LLM** OpenIE knowledge graph + PPR.
- **Most of the deficit is the encoder**, not the method: our 1.5B dense (71.8) vs their 7B dense baseline
  (80.2) is an 8.4pt *encoder* gap. Our fusion on a 1.5B encoder (**79.5**) essentially **matches their 7B
  dense** (80.2) and **beats it on 2Wiki** (79.2 vs 76.5).
- **Positioning is efficiency / no-LLM:** competitive recall at ~1/5 the encoder size and **zero LLM indexing
  cost**, ~8pt below full HippoRAG 2 — the gap their 70B-LLM graph buys.
- **L3 traversal** is neutral at R@5 on text (golds are dense-reachable; the reranker owns the tight budget),
  adds at R@50 (mean 94.0→95.9; 2Wiki +4.25), and is decisive on KB (WebQSP graphlift +62).
- **Open limitation (surfaced here):** applied *zero-shot* (head trained on `_clean`, not refit), the offset
  head is unreliable — MuSiQue drops to 55.2, **below dense**, because min-rank fusion isn't robust to a weak
  OOD signal. The head helps in-suite; robust cross-corpus transfer is unsolved. HotpotQA zero-shot holds (90.0).
- **Caveat:** 100-query test split (small sample). Remaining four systems (metrics differ — recall vs EM/F1)
  still to run.

## Status

- **All 5 official repos cloned and code-confirmed** (`baselines/repos/`, gitignored).
- Corrections from staging: `ianliuwd/HippoRAG2` dropped (OSU repo *is* HippoRAG 2); KG2RAG is NAACL'25.
- **TODO:** common corpus adapter (`master_nodes_{ds}_hpr_clean.json` → each system's ingest); per-venv
  install; run + tabulate (recall + indexing cost + latency) against `results/L2/_hpr_paper_backup`.
