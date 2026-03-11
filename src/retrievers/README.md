# `src/retrievers/` — Five Retrieval Strategies

Each file implements one retrieval architecture. All inherit from `BaseRetriever` and expose a single `retrieve(query: str) -> RetrievalResult` method. Every strategy receives the **identical** `SharedComponents` bundle (same `GraphEngine`, `UnifiedVectorStore`, `LLMClient`) via the factory, guaranteeing fair benchmarking.

---

## Shared Interface: `base.py`

```python
@dataclass
class RetrievalResult:
    answer: str
    retrieved_nodes: List[Node]     # Nodes used for generation
    reasoning_path: List[str]       # Empty for flat retrievers, populated by agent
    latency_seconds: float
    metadata: dict                  # Strategy-specific diagnostics

class BaseRetriever(ABC):
    def __init__(self, shared: SharedComponents): ...
    
    @abstractmethod
    def retrieve(self, query: str) -> RetrievalResult: ...
    
    def _prepare_prompt(self, query: str, nodes: List[Node]) -> str:
        """
        Shared by all subclasses. Calls truncate_to_token_budget() before
        building the prompt. Never call llm.generate() without going
        through this method.
        """
```

---

## Strategy Files

### `vector_rag.py` — Baseline 1: Pure Semantic Search

**Mechanism:** Embeds the query and retrieves the top-K nearest nodes from `nodes.index`.

```
Query → embed → search_nodes(top_K=10) → truncate → generate
```

**Strengths:** Very fast (~10ms), excellent recall on SQuAD-style semantic questions.  
**Weaknesses:** Cannot cross node boundaries; multi-hop queries fail completely.

**Expected categories:** Semantic (1–80) — strong. All others — weak.

---

### `graph_rag.py` — Baseline 2: Static Multi-Hop BFS

**Mechanism:** Extracts keywords (or uses vector search for an entry node), then performs a 2-hop BFS expansion across the graph.

```
Query → keyword extract → find entry node → BFS(hops=2) → truncate → generate
```

**Strengths:** Preserves relational structure, capable of multi-hop reasoning.  
**Weaknesses:** BFS is indiscriminate — pulls in many irrelevant neighbor nodes. Latency scales with graph density.

**Expected categories:** Multi-hop (81–160) — competitive. Others — poor (noisy).

---

### `crag_standard.py` — C-RAG Standard: Epistemic Evaluator

**Mechanism:** Runs vector search, then asks the LLM to grade the retrieved context as `CORRECT`, `AMBIGUOUS`, or `INCORRECT`.

```
Query → search_nodes → Evaluator SLM grades context
  CORRECT   → generate directly
  AMBIGUOUS → expand neighbors (1-hop) → re-generate
  INCORRECT → discard context entirely → Web Search fallback OR refuse
```

**Strengths:** Prevents hallucination on unanswerable questions; handles false premises gracefully.  
**Weaknesses:** Extra LLM call per query doubles latency; web search requires internet access.

**Expected categories:** Fallback/Web (161–240) and False Premise — strong. Others — mediocre.

---

### `crag_colbert.py` — C-RAG with ColBERT Late-Interaction Re-Ranking

**Mechanism:** Broad semantic search returns top-50 candidates; ColBERT cross-encoder re-ranks them by exact token-level interaction; top-5 are kept for generation.

```
Query → search_nodes(top_K=50)
      → serialize each node to passage string   ← REQUIRED: ColBERT operates on strings
      → ColBERT.rerank(query, passages)
      → keep top-5 → truncate → generate
```

> **Critical:** The `serialize_node_to_passage(node)` utility from `src/common/utils.py` **must** be called before feeding nodes to ColBERT. ColBERT's API does not accept dict objects.

**Strengths:** Exceptional at recovering exact part numbers, legal clauses, jargon, and precise quotes that dense embeddings over-smooth.  
**Weaknesses:** ColBERT reranking adds ~200–500ms. Overkill for broad semantic queries.

**Expected categories:** Exact Lexical (241–320) — best performer. Semantic — competitive.

---

### `crag_agent.py` — v4 Agentic Graph Selection (Crown Jewel)

**Mechanism:** Three-phase agentic loop operating on the METIS-partitioned graph.

```
Phase 1 — TELEPORT (Partition Selection via Bi-Encoder):
  Query → mlp_encoder.encode() → search_partitions(top_k=3) → [Partition A, B, C]

Phase 2 — STITCH (Subgraph Assembly):
  graph_engine.get_bridge_edges([A, B, C])
  ↳ FALLBACK if empty: bfs_boundary_union([A, B, C], hops=1)
  → Assembled Subgraph

Phase 3 — TRAVERSE (ColBERT Think-Act-Observe Loop):
  for node in subgraph.walk():
      score = colbert.score(query, serialize_node_to_passage(node))
      if score < prune_threshold:
          subgraph.prune(node)         # Aggressively remove irrelevant branches
  → Curated reasoning path → truncate → generate
```

**Strengths:** Produces highly targeted multi-hop reasoning paths; resistant to noise.  
**Weaknesses:** Most complex pipeline; highest latency (~2–6s). Requires trained Bi-Encoder checkpoint.

**Expected categories:** Routing/Graph-Selection (321–400) — dominant. Multi-hop — strong.

---

## Comparative Summary

| Strategy | Avg Latency | P@1 (Semantic) | R@10 (Multi-hop) | Fallback | Exact-Lexical |
|---|---|---|---|---|---|
| Vector RAG | ~10ms | ★★★★★ | ★☆☆☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| Graph RAG | ~300ms | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| C-RAG Standard | ~600ms | ★★★☆☆ | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ |
| C-RAG ColBERT | ~400ms | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| C-RAG Agent v4 | ~3000ms | ★★★★☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ |

*All values are projections from architectural analysis, not measured results.*
