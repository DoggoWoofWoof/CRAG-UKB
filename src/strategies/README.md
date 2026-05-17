# Retrieval Strategies Architecture
This directory houses the execution frameworks comprising the scientific benchmark tests. Each Strategy operates off the singular `CoreEngine` (`src/core/engine.py`).

## Director Structure & File Mechanics

### `base.py`
The Strategy interface.
*   **Purpose**: Defines a common protocol `execute(query: str) -> dict`. Forces all 5 Models to accept the same input and output metrics identically for the SuperModel routing.

### `vector_rag.py` (Baseline 1: Dense-Lexical RRF)
The standard vector-similarity baseline.
*   **Purpose**: Retrieves contexts globally with no structural awareness.
*   **Mechanics**: Computes semantic similarity (Dense) and exact keyword occurrences (Lexical) simultaneously. Merges both result rankings using Reciprocal Rank Fusion (RRF). Provides a normalized, context-rich subset to the LLM generator.

### `graph_rag.py` (Baseline 2: Static Graph Traversal)
The standard semantic sub-graph baseline.
*   **Purpose**: Retrieves contexts physically tied to topological boundaries.
*   **Mechanics**: Finds an exact entry node, walks its edges using `CoreEngine.get_neighbors()`, and feeds the collected subgraph cluster into the generative prompt. Inherently structural, mathematically rigid.

### `crag_standard.py` (Advanced 1: Corrective Agent)
The base C-RAG evaluation metric variant.
*   **Purpose**: Introduces critique-based epistemic reasoning.
*   **Mechanics**: After a retrieval pass, it queries a secondary "Evaluator" prompt. This explicit evaluator classifies the retrieved chunk as `CORRECT`, `INCORRECT`, or `AMBIGUOUS`. If `CORRECT`, proceeds; if `INCORRECT`, actively discards the chunk; if `AMBIGUOUS`, triggers localized fallback logic (in this framework, graph expansion).

### `crag_colbert.py` (Advanced 2: ColBERT Precision)
A High-Precision variant leveraging Late Interaction.
*   **Purpose**: Maximizes strict word-association accuracy at the cost of latency.
*   **Mechanics**: Bypasses the FAISS Dense single-vector similarity completely. Instead, passes the query string straight to Ragatouille. It aligns query tokens computationally against individual stored document tokens (e.g., matching the literal context of "George Washington" structurally rather than a generic vector representation of "presidents").

### `crag_v4_agent.py` (Advanced 3: The Unified Zenith Strategy)
The ultimate execution framework integrating the "Multi-Modal Retrieval Engine" capabilities linearly.
*   **Purpose**: Solves massive context bloat by teleporting between Dense and Structural dimensions.
*   **Mechanics**:
    1.  **Teleport (Dense -> Hierarchical)**: Uses the trained **MLP Bi-Encoder** (or FAISS Centroid Index) to project the query vector into partition space. The Engine returns ONLY the `partition_id`s that mathematically align closest to the prompt intention.
    2.  **Stitch (Hierarchical -> Structural)**: Immediately fetches all `node_ids` belonging to the top K winning partitions mathematically.
    3.  **Observe (Structural -> Late-Interaction)**: Because these nodes would crush the LLM context window, they are fed aggressively into a Level 2 Reranker (like ColBERT or BGE). It physically discards 95% of the graph, returning the top perfectly precise exact-match topological tokens.
    4.  **Generate**: Feeds the aggressively scrubbed, context-perfect structural data to the generator.
