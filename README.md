# Unified Cognitive Graph-RAG (C-RAG)

> **Branch:** `unified-crag-architecture` · **Phase:** Capstone Phase-2 (PW25_BJD_01)  
> A modular, factory-based RAG laboratory where **five retrieval architectures** share one Unified Knowledge Base.

---

## Overview

C-RAG (Cognitive Retrieval-Augmented Generation) is a research framework that unifies several retrieval paradigms—from simple vector search to an agentic, graph-walking multi-hop reasoner—into a single, benchmarkable system. All architectures are isolated as pluggable strategies that receive identical inputs from the same knowledge base, making comparisons scientifically rigorous.

```
                  ┌─────────────────────────────────────┐
                  │        Unified Knowledge Base        │
                  │  ┌─────────────┐ ┌───────────────┐  │
                  │  │  FAISS Node │ │ FAISS Centroid│  │
                  │  │    Index    │ │     Index     │  │
                  │  └─────────────┘ └───────────────┘  │
                  │         ┌─────────────────┐          │
                  │         │   PyG Graph.pt  │          │
                  │         └─────────────────┘          │
                  └──────────────────┬──────────────────┘
                                     │ shared
              ┌──────────────────────┼─────────────────────┐
              ▼          ▼           ▼          ▼           ▼
         VectorRAG   GraphRAG  CRAGStandard CRAGColBERT CRAGAgent
                                                           (v4)
```

---

## Five Retrieval Strategies

| Strategy | File | Key Mechanism | Target Category |
|---|---|---|---|
| **Vector RAG** | `src/retrievers/vector_rag.py` | FAISS top-K semantic search | Broad semantic queries |
| **Graph RAG** | `src/retrievers/graph_rag.py` | 2-hop BFS from entry node | Multi-hop entity queries |
| **C-RAG Standard** | `src/retrievers/crag_standard.py` | Epistemic evaluator + web fallback | Unanswerable / false premise |
| **C-RAG ColBERT** | `src/retrievers/crag_colbert.py` | ColBERT late-interaction re-rank top-50→5 | Exact lexical / jargon |
| **C-RAG Agent v4** | `src/retrievers/crag_agent.py` | Teleport → Stitch → ColBERT Traverse | Complex hybrid routing |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build the Unified Knowledge Base (offline, one-time)
python -m src.ingestion.kb_builder \
  --squad   data/raw/squad_v2.json \
  --webqsp  data/raw/webqsp.json \
  --cora    data/raw/cora/ \
  --output  data/kg_store/

# 3. Run METIS partitioning
python -m src.ingestion.partitioner --input data/kg_store/graph.pt --n-partitions 50

# 4. Train InfoNCE Bi-Encoder (query → partition alignment)
python -m src.alignment.infonce_loss --train --epochs 30

# 5. Run any pipeline via the PipelineFactory
python -m src.router.factory \
  --strategy crag_agent \
  --benchmark benchmark_400.csv \
  --output results/crag_agent.json

# 6. Evaluate results (P@1, R@10, MRR)
python -m src.evaluation.metrics \
  --results results/ \
  --groundtruth data/processed/ground_truth.json
```

---

## Directory Structure

```
unified-crag/
├── data/               → Raw datasets, processed nodes/edges, serialized KB
├── src/
│   ├── common/         → Shared singletons: GraphEngine, VectorStore, LLMClient
│   ├── ingestion/      → Offline KB construction: kb_builder, partitioner, embedder
│   ├── alignment/      → InfoNCE bi-encoder: query-to-partition alignment model
│   ├── retrievers/     → Five retrieval strategies implementing BaseRetriever
│   ├── router/         → PipelineFactory: instantiate any strategy with one call
│   └── evaluation/     → P@1, R@10, MRR metrics + random-walk benchmark synthesizer
├── configs/            → unified.yaml — single config for all strategies
├── results/            → JSON outputs from each pipeline run
└── benchmark_400.csv   → 400-query stratified benchmark (5 categories × 80 queries)
```

---

## Datasets Used

| Dataset | Nodes Added | Edge Type | Purpose |
|---|---|---|---|
| **SQuAD v2** (142k+) | Document chunks (answerable) | `NEXT` (sequential) | Semantic & fallback categories |
| **SQuAD v2 unanswerable** | Document chunks (unanswerable) | `NEXT` | Epistemic evaluator trigger |
| **WebQSP** (4k+) | Wikidata entities / triples | `SUBJECT_OF`, `OBJECT_OF` | Multi-hop entity reasoning |
| **Cora** (2.7k papers) | Academic paper nodes | `CITES` | Partition validation + graph traversal |
| **Entity Linking** | Bridge edges | `MENTIONS` | Connects SQuAD text ↔ WebQSP entities |

---

## Known Issues Fixed vs. Legacy Branches

| Bug | Legacy Branch | Fix Applied |
|---|---|---|
| `ImportError: GraphPartitioner` | `v3`, `c-rag-colbert-query` | Replaced with `SemanticPartitioner` everywhere |
| `MockLLMClient TypeError` | all branches | Added `**kwargs` to `__init__` |
| ColBERT receives node dicts not strings | `c-rag-colbert-query` | Added `serialize_node_to_passage()` utility |
| Empty bridge edges after METIS | `v4-query-graph-selection` | BFS boundary union fallback added |
| Baselines queried different KBs | all branches | Single `SharedComponents` injected via factory |

---

## Requirements

```
Python >= 3.10
torch >= 2.0
torch-geometric
faiss-cpu         # or faiss-gpu
networkx
colbert-ai
sentence-transformers
ragas
scikit-learn
pandas
tqdm
```

---

## License

MIT — see [LICENSE](LICENSE).
