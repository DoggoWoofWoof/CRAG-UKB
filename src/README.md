# `src/` — Python Package Root

Top-level package containing all C-RAG source modules. Import paths start with `src.`:

```python
from src.common.vector_store import UnifiedVectorStore
from src.retrievers.crag_agent import CRAGAgent
from src.router.factory import PipelineFactory
```

---

## Module Hierarchy

```
src/
├── common/      → Shared singletons & utilities (load once, use everywhere)
├── ingestion/   → OFFLINE ONLY: KB construction, partitioning, embedding
├── alignment/   → OFFLINE ONLY: KL Divergence & InfoNCE training
├── retrievers/  → QUERY TIME: Five retrieval strategy implementations
├── router/      → QUERY TIME: PipelineFactory — the single entry point
└── evaluation/  → POST-RUN: IR metrics + benchmark synthesizer
```

**Dependencies flow strictly downward:**
- `retrievers/` depends on `common/` — never on `ingestion/` or `alignment/`
- `router/` depends on `retrievers/` and `common/`
- `evaluation/` depends only on result JSON files — never on live pipelines

---

## Key Design Invariants

1. **No circular imports.** `common/` never imports from `retrievers/`.
2. **All retrievers are stateless at query time.** State (indices, graph) lives in `SharedComponents`.
3. **Token budget is always enforced.** Every retriever calls `truncate_to_token_budget()` inside `_prepare_prompt()` before touching the LLM.
4. **`data/kg_store/` is read-only at query time.** Only `ingestion/` scripts write to it.
