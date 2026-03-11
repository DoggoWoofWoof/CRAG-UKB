# `src/router/` — PipelineFactory (The Super Model)

The **single entry point** for instantiating any of the five retrieval architectures. All shared infrastructure is loaded once inside `SharedComponents` and injected into every pipeline — this is what guarantees fair benchmarking.

---

## `factory.py`

```python
class SharedComponents:
    """
    Loaded once. Passed by reference to every retriever.
    Guarantees all pipelines use the exact same KG, vector store, and LLM.
    """
    def __init__(self, config: dict):
        self.graph_engine  = GraphEngine(config["kg_store_path"])
        self.vector_store  = UnifiedVectorStore(config["kg_store_path"])
        self.llm           = build_llm_client(config["llm"])  # Ollama or Mock

class PipelineFactory:
    @staticmethod
    def get_pipeline(strategy: str, config: dict) -> BaseRetriever:
        """
        Instantiate any retrieval strategy by name.
        
        Valid strategy names:
          "vector"         → VectorRAG
          "graph"          → GraphRAG
          "crag_standard"  → CRAGStandard
          "crag_colbert"   → CRAGColBERT
          "crag_agent"     → CRAGAgent (v4, requires trained encoder checkpoint)
        
        Raises ValueError for unknown strategy names.
        """
        shared = SharedComponents(config)
        ...
```

---

## CLI Usage

The factory can be invoked directly from the command line to run a full benchmark:

```bash
# Single strategy run
python -m src.router.factory \
  --strategy crag_agent \
  --config   configs/unified.yaml \
  --benchmark benchmark_400.csv \
  --output   results/crag_agent.json \
  --llm      mock          # Use --llm ollama for a live model

# Run all 5 strategies in sequence
for strategy in vector graph crag_standard crag_colbert crag_agent; do
  python -m src.router.factory \
    --strategy $strategy \
    --config configs/unified.yaml \
    --benchmark benchmark_400.csv \
    --output results/${strategy}.json
done
```

---

## Adding a New Strategy

1. Create `src/retrievers/my_strategy.py` inheriting from `BaseRetriever`
2. Implement `retrieve(self, query: str) -> RetrievalResult`
3. Add one line to `factory.py`:
   ```python
   "my_strategy": MyStrategy(shared),
   ```

No other files need to change.
