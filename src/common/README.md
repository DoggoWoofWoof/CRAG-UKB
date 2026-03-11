# `src/common/` — Shared Core Components

The **central nervous system** of the unified C-RAG architecture. Every retrieval strategy imports directly from here. Nothing in this directory is pipeline-specific — these are pure, stateless utilities and singletons that guarantee all 5 retrievers operate on identical infrastructure.

> **Rule:** If you find yourself copy-pasting a utility into a retriever file, it belongs here instead.

---

## Files

### `graph_engine.py` — Graph Topology Singleton

Wraps the PyTorch Geometric `Data` object loaded from `data/kg_store/graph.pt`. Provides high-level traversal methods so retrievers never touch the raw tensor API.

```python
class GraphEngine:
    def get_neighbors(self, node_id: str, hops: int = 1) -> List[Node]:
        """BFS traversal up to N hops. Returns deduplicated node list."""

    def get_bridge_edges(self, partition_ids: List[int]) -> List[Edge]:
        """
        Returns edges that cross partition boundaries.
        FALLBACK: If METIS cut all inter-partition edges (common), performs
        a 1-hop BFS from each partition's boundary nodes and returns the
        union of their neighborhoods as the stitched subgraph.
        """

    def bfs_boundary_union(self, partition_ids: List[int], hops: int = 1) -> Subgraph:
        """Explicit BFS fallback for the Stitch phase in crag_agent.py."""
```

**Key design note:** `get_bridge_edges` never raises an exception on empty results. It silently falls back to `bfs_boundary_union`. This prevents the v4 agent from silently returning nothing when METIS aggressively cuts cross-partition connections.

---

### `vector_store.py` — Dual FAISS Index Wrapper

Maintains **two** FAISS indices:
- `nodes.index` — 768-dim embeddings for every node in the UKB (~150k vectors)
- `centroids.index` — 768-dim centroid embeddings for every partition (~50 vectors)

```python
class UnifiedVectorStore:
    def search_nodes(self, query_emb: np.ndarray, top_k: int = 10) -> List[Node]:
        """Search the node-level index. Used by VectorRAG, GraphRAG entry point, ColBERT pre-filter."""

    def search_partitions(self, query_emb: np.ndarray, top_k: int = 3) -> List[int]:
        """Search the centroid index. Used exclusively by CRAGAgent Teleport step."""
```

**Why two indices?** The centroid index is tiny (~50 vectors) and enables sub-millisecond partition selection — the entire point of the METIS partitioning strategy. Mixing it with the full node index would ruin the two-stage retrieval design.

---

### `llm_client.py` — SLM Connector with Token Buffer

Abstract `LLMClient` base class + concrete implementations. All implementations expose the same `generate(prompt) → str` interface.

```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str: ...

class OllamaClient(LLMClient):
    """Production: local Ollama instance (Llama-3, Phi-3, etc.)"""

class MockLLMClient(LLMClient):
    """Testing: deterministic keyword-based responses. Accepts **kwargs."""
```

**Critical utility — Token Truncation Buffer:**

```python
def truncate_to_token_budget(nodes: List[Node], budget: int = 3000) -> List[Node]:
    """
    Drops the lowest-ranked nodes until the total token count fits
    within the SLM's context window. Always called before generate().
    Prevents OOM errors on small models (Phi-3, TinyLlama, etc.).
    """
```

Every retriever **must** call `truncate_to_token_budget(nodes)` before passing context to the LLM. This is enforced in `BaseRetriever._prepare_prompt()`.

---

### `utils.py` — Logging, Config, and Token Counting

```python
# Config
def load_config(path: str = "configs/unified.yaml") -> dict: ...

# Logging — structured, timestamped
def get_logger(name: str) -> logging.Logger: ...

# Token counting — uses tiktoken for OpenAI-compatible counts
def count_tokens(text: str) -> int: ...

# Node serialization — shared across ColBERT and the agent
def serialize_node_to_passage(node: Node) -> str:
    """
    Converts a node dict to a plain string passage for ColBERT.
    Format: "{TYPE}: {text_content}"
    REQUIRED: ColBERT operates on string passages only, not dicts.
    """
```

---

## Design Principles

1. **Singleton pattern** — `GraphEngine` and `UnifiedVectorStore` load their indices once at startup and are shared across all pipeline instances via `SharedComponents` in `factory.py`.
2. **No side effects** — Methods here never write to disk. Writing belongs in `src/ingestion/`.
3. **Framework-agnostic** — No retriever-specific logic. These modules don't know about ColBERT, InfoNCE, or the agent loop.
