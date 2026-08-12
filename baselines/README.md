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

## First comparison point (measured, no LLM) — `results/L2/hpr_headtohead.json`

Our pipeline's L2 rerank on the reserved `_hpr` corpora, Recall@5 (no LLM, no Modal):

| dataset | dense R@5 | our best-rerank R@5 | HippoRAG v1 (published R@5) |
|---|---|---|---|
| MuSiQue | 56.7 | **70.5** | 51.9 → ours **+18.6** |
| 2Wiki | 70.3 | **80.5** | 89.1 → HippoRAG +8.6 |
| HotpotQA | 88.5 | **91.0** | (not in hand) |

Honest reading: this is our **rerank alone** (dense + offset + SPLADE best-of), **no L3 traversal composed in**, and **no LLM**. It already beats HippoRAG v1 on MuSiQue (+18.6) and trails on 2Wiki — where our L3 traversal adds +15–20 (§4.5), not yet composed here. To complete the head-to-head: (1) compose rerank → traversal for our final recall; (2) target HippoRAG **2** (current) numbers, not v1; (3) run/extract the remaining four (their metrics differ — recall vs EM/F1).

## Status

- **All 5 official repos cloned and code-confirmed** (`baselines/repos/`, gitignored).
- Corrections from staging: `ianliuwd/HippoRAG2` dropped (OSU repo *is* HippoRAG 2); KG2RAG is NAACL'25.
- **TODO:** common corpus adapter (`master_nodes_{ds}_hpr_clean.json` → each system's ingest); per-venv
  install; run + tabulate (recall + indexing cost + latency) against `results/L2/_hpr_paper_backup`.
