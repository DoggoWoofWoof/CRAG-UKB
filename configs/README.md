# `configs/` — Configuration Files

All system parameters for the unified C-RAG architecture live here. A single `unified.yaml` drives every pipeline. Override specific keys with `--config-override` on the CLI without touching this file.

---

## `unified.yaml`

```yaml
# ─── Knowledge Base ───────────────────────────────────────────────────────────
kg_store_path: data/kg_store/

# ─── LLM Provider ─────────────────────────────────────────────────────────────
llm:
  provider: mock            # Options: mock | ollama | openai
  model: llama3.2           # Ollama model tag (ignored if provider = mock)
  base_url: http://localhost:11434
  temperature: 0.3
  max_tokens: 512
  token_budget: 3000        # Max tokens passed to LLM context window

# ─── Embedding Model ──────────────────────────────────────────────────────────
embedding:
  model: sentence-transformers/all-mpnet-base-v2
  dim: 768
  batch_size: 128

# ─── Partitioning ─────────────────────────────────────────────────────────────
partitioning:
  n_partitions: 50
  max_partition_size: 200   # Hard cap per partition (nodes)
  validate_on_cora: true    # Run METIS on Cora first and assert edge-cut < 15%

# ─── Alignment (InfoNCE Bi-Encoder) ───────────────────────────────────────────
alignment:
  checkpoint: checkpoints/mlp_encoder.pt
  hidden_dim: 256           # Projection target dimension
  temperature: 0.07         # InfoNCE temperature τ
  epochs: 30
  batch_size: 64
  lr: 1e-4

# ─── Retrieval Strategy Parameters ────────────────────────────────────────────
retrieval:
  vector_top_k: 10          # VectorRAG / GraphRAG entry node search
  colbert_candidate_k: 50   # Pre-filter before ColBERT rerank
  colbert_final_k: 5        # Nodes kept after ColBERT rerank
  bfs_hops: 2               # GraphRAG BFS expansion depth
  agent_top_partitions: 3   # CRAGAgent: number of partitions to Teleport into
  agent_prune_threshold: 0.3 # ColBERT score below this → prune node from subgraph

# ─── ColBERT ──────────────────────────────────────────────────────────────────
colbert:
  model: colbert-ir/colbertv2.0
  device: cpu               # Options: cpu | cuda

# ─── Evaluation ───────────────────────────────────────────────────────────────
evaluation:
  benchmark_csv: benchmark_400.csv
  output_dir: results/
  ground_truth: data/processed/ground_truth.json
  use_ragas: false          # Set true + provide OPENAI_API_KEY for LLM-based metrics
```

---

## Overriding Config at Runtime

```bash
# Use Ollama instead of Mock LLM
python -m src.router.factory --strategy vector --config-override llm.provider=ollama

# Use GPU for ColBERT
python -m src.router.factory --strategy crag_colbert --config-override colbert.device=cuda

# Change number of partitions (requires re-running partitioner.py)
python -m src.ingestion.partitioner --config-override partitioning.n_partitions=100
```
