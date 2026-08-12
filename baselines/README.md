# Baseline graph-RAG systems — C-RAG head-to-head

Staging for the paper comparison (Next Steps #4). Every established system below builds its graph with
an **LLM** and/or reasons with a **GNN**; C-RAG uses neither. The head-to-head must therefore report
**indexing cost and latency** alongside retrieval quality, since our differentiator is competitive
quality with no LLM in the loop.

## Systems to compare

| System | Repo | Venue | Graph construction | Reasoning core | Uses LLM? | Uses GNN? |
|---|---|---|---|---|---|---|
| HippoRAG | `OSU-NLP-Group/HippoRAG` | NeurIPS'24 | LLM OpenIE triples | Personalized PageRank | yes | no |
| HippoRAG 2 | `ianliuwd/HippoRAG2` | NeurIPS'24/25 | LLM OpenIE | PPR + better associativity | yes | no |
| GFM-RAG | `RManLuo/gfm-rag` | NeurIPS'25 | LLM (GPT-4o) extraction | **GNN foundation model** (dataset-agnostic) | yes | yes |
| HopRAG | arXiv 2502.12442 | 2025 | passage graph | LLM logic-aware multi-hop walk | yes | no |
| KG2RAG | (search for code) | WWW'25 | KG-guided expansion | KG chains + rerank | yes | no |
| SiReRAG | (search for code) | 2025 | similar + related indexing | dual similarity/relatedness trees | yes | no |

`GFM-RAG` is the closest competitor to our **dataset-agnostic** claim — but it is dataset-agnostic
*via a GNN over an LLM-built KG*. C-RAG is dataset-agnostic with neither (and §4.8 shows a GNN
underperforms dense/offset and does not scale to our 500k+-node graphs).

## Head-to-head protocol

1. **Corpora:** the HippoRAG-standard **1,000-question candidate corpora** (MuSiQue 11,656 / 2Wiki 6,119 /
   HotpotQA 9,811) — reserved in this project for exactly this comparison and backed up at
   `results/L2/_hpr_paper_backup/`. Same corpora, same splits for every system.
2. **Retrieval metric:** Recall@2 / Recall@5 (HippoRAG's protocol). C-RAG entry = the full pipeline
   (dense + SPLADE candgen → best-of fusion rerank → traversal), frozen gte-Qwen2 substrate.
3. **Downstream:** EM / F1 with a **matched reader** (same generator + prompt for all systems), once the
   C-RAG end-to-end QA path (Next Steps #1) is wired.
4. **Cost columns (the point):** report per-corpus **indexing cost** (LLM tokens / $ for OpenIE-style
   construction — zero for C-RAG) and **query latency**, so "competitive quality, no LLM" is quantified.

## Setup

Each baseline runs in its own venv (they pin conflicting torch / LLM-client versions). The LLM-based
ones need an OpenAI (or local-LLM) key **for graph construction** — that is exactly the step C-RAG
removes, so track its token cost as a baseline expense.

```bash
# cloned (gitignored) under baselines/repos/
python -m venv baselines/.venv-hipporag && baselines/.venv-hipporag/bin/pip install -r baselines/repos/HippoRAG/requirements.txt
# build their KG on our _hpr corpora -> run their retrieval -> export per-query recall
# compare against results/L2/_hpr_paper_backup/*.json
```

## Status

- **Cloned** (`baselines/repos/`, gitignored): HippoRAG, HippoRAG2, gfm-rag.
- **TODO:** locate HopRAG / KG2RAG / SiReRAG code; write a common corpus adapter (our
  `master_nodes_{ds}_hpr_clean.json` → each system's ingest format); run and tabulate.
