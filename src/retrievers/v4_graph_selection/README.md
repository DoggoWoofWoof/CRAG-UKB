# `src/retrievers/v4_graph_selection/` — v4 Agentic Graph Selection (Standalone Component)

This directory is a **fully self-contained port** of the `v4-query-graph-selection` branch from [github.com/suryap3105/C-RAG](https://github.com/suryap3105/C-RAG/tree/v4-query-graph-selection).

It can be run independently **or** called transparently through `PipelineFactory` as the `"v4_graph_selection"` strategy — exactly like any other retriever.

---

## Architecture: Teleport → Stitch → Traverse

```
Query
  │
  ▼ [Teleport] ColBERT MaxSim Partition Router
  │  Encodes query → scores against partition token matrices
  │  → returns Top-3 Partition IDs + scores
  │
  ▼ [Stitch] GraphEngine Subgraph Assembly
  │  Calls graph_engine.get_bridge_edges(partition_ids)
  │  FALLBACK: If METIS cut all inter-partition edges →
  │            bfs_boundary_union(partitions, hops=1)
  │
  ▼ [Traverse] NeuroHybridRetrievalModule
  │  Parallel vector + graph retrieval on the stitched subgraph
  │  AdaptiveGatingNetwork blends results (α auto-tuned)
  │  NeuralSubgraphMatcher re-scores candidate paths
  │  HybridReranker (cross-encoder) finalizes ranking
  │
  ▼ Token Budget Truncation → LLM Generate → Answer
```

---

## Internal Components

| File | Purpose | Source (GitHub Branch) |
|---|---|---|
| `adapter.py` | `BaseRetriever`-compatible wrapper + standalone entry point | New (this repo) |
| `pipeline.py` | Assembles `NeuroHybridRetrievalModule` from `SharedComponents` | New (this repo) |
| `neural_hybrid.py` | `NeuroHybridRetrievalModule` + `AdaptiveGatingNetwork` | `src/crag/retrieval/neural_hybrid.py` |
| `routing/colbert.py` | `ColBERTPartitionRouter` — MaxSim late-interaction | `src/crag/routing/colbert.py` |
| `routing/structural.py` | `StructuralAligner` — node/edge type fingerprint scoring | `src/crag/routing/structural.py` |
| `model/gnn.py` | `NeuralSubgraphMatcher` — GNN for subgraph scoring | `src/crag/model/gnn.py` |
| `model/query_graph.py` | `QueryGraphGenerator` — LLM-based query decomposition | `src/crag/model/query_graph.py` |
| `model/cross_encoder.py` | `HybridReranker` — final cross-encoder reranking | `src/crag/model/cross_encoder.py` |

---

## Independent Testing (No PipelineFactory required)

```python
import yaml
from src.retrievers.v4_graph_selection import build_v4_pipeline

config = yaml.safe_load(open("configs/unified.yaml"))

# Build the pipeline
pipeline = build_v4_pipeline(config)

# Run a query
result = pipeline.retrieve("Which company acquired GitHub in 2018 and who is its CEO?")

print("Answer:", result.answer)
print("Partitions selected:", result.partitions_selected)
print("ColBERT scores:", result.colbert_scores)
print("Reasoning path:", result.reasoning_path)
print(f"Latency: {result.latency_seconds:.2f}s")
```

---

## Via PipelineFactory

```python
from src.router.factory import PipelineFactory
import yaml

config = yaml.safe_load(open("configs/unified.yaml"))
pipeline = PipelineFactory.get_pipeline("v4_graph_selection", config)
result = pipeline.retrieve("your query here")
```

```bash
# Or via CLI
python -m src.router.factory \
  --strategy v4_graph_selection \
  --benchmark benchmark_400.csv \
  --output results/v4_graph_selection.json
```

---

## Prerequisites

Before running, these must be complete:

1. **METIS partitioning** — `python -m src.ingestion.partitioner` (creates `graph.data.part_id`)
2. **Embeddings** — `python -m src.ingestion.embedder` (creates `graph.data.x`)
3. **GNN checkpoint** *(optional)* — set `gnn.checkpoint_path` in `configs/unified.yaml`

---

## Key Differences vs. `crag_agent.py`

| Aspect | `crag_agent.py` | `v4_graph_selection/` |
|---|---|---|
| Partition selection | MLP Bi-Encoder (InfoNCE-trained) | ColBERT MaxSim (token matrices) |
| Traversal | ColBERT node pruning loop | `NeuroHybridRetrievalModule` (parallel + GNN) |
| Fusion | Deterministic (pruning) | Adaptive Gating Network (learned α) |
| GNN dependency | No | Yes (`NeuralSubgraphMatcher`) |
| Source | New implementation | Ported from v4 GitHub branch |
