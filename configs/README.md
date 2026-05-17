# Centralized Configuration Architecture
This directory houses the global variables and tunable hyperparameters used by the C-RAG Multi-Modal system.

## Director Structure & File Mechanics

### `config.yaml`
The application's single source of truth for dynamic parameters. 
*   **Purpose**: Prevents hardcoding paths or inference variables inside specific `.py` files, allowing researchers to quickly tune the framework for different benchmarks without rewriting the logic layers.
*   **Mechanics**:
    *   **`storage:`**: Hard links to the Data Lake (`data/raw`, `data/processed/master_nodes.json`, etc). Any pipeline scripts or evaluators inherently trust these paths.
    *   **`retrieval:`**: Focuses on hyperparameters modifying context fetch limits:
        *   `top_k`: How many dense vector arrays FAISS should retrieve initially (default: 5).
        *   `graph_hops`: How deep the topological BFS (`vector_to_graph_search`) should crawl before terminating (default: 2 hops).
        *   `max_context_tokens`: The `tiktoken` budget enforcing input limitations on standard API calls, preventing LLM failure (default: 3000 tokens).
        *   `models`: Defines the specific HuggingFace dense `encoder`, the textual generic `generator` (e.g., `gpt-3.5-turbo`), and the `colbert` late-interaction checkpoint string.
    *   **`partitioning:`**: Defines integer constraints for PyMETIS graph partitioning (e.g., grouping topological structures into communities of ~1000 nodes) to prevent `Centroid` arrays from becoming too dispersed while strictly preserving logical multi-hop boundaries.
