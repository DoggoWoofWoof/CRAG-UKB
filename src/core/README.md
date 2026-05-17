# Core Service Architecture
This directory is the foundational backend supporting the Multi-Modal Retrieval Engine. It handles Vectorization, Indexing, and the primary Singleton API.

## Director Structure & File Mechanics

### `encoders.py`
The Dense Mathematical Embedder.
*   **Purpose**: Transforms natural language strings into high-dimensional numerical arrays.
*   **Mechanics**: Initializes a HuggingFace `SentenceTransformer` model (default: `all-MiniLM-L6-v2` loaded from `config.yaml`). The `encode()` method converts arrays of texts into `float32` numpy arrays of shape `[N, 384]`. Crucial for Vector-RAG similarity.

### `indexers.py`
The UKB Unified Factory Builder.
*   **Purpose**: Consumes the `StandardNode` list and mathematically compiles all 6 index views simultaneously. 
*   **Mechanics**:
    *   `build_faiss_node_index`: Uses `encoders.py` to embed all ~215k document/entity texts, injecting the matrix into a `faiss.IndexFlatL2` object.
    *   `build_bm25_index`: Uses `rank_bm25` Okapi algorithm to compute Term-Frequency/Inverse-Document-Frequency occurrences. 
    *   `build_pyg_graph`: Extracts raw string `neighbors` into numerical `[node_src, node_dst]` tensors. Weaves synthetic KNN edges for topologically isolated nodes using a fast `faiss.IndexIVFFlat` approximate clusterer to guarantee full connectivity. Outputs PyTorch `graph.pt`.
    *   `build_partition_map`: Runs PyMETIS (with naive chunking fallback) over the graph. Clusters the nodes into distinct topological community partitions (targeting ~1000 nodes each).
    *   `build_faiss_centroid_index`: Calculates the **degree-weighted** average mathematical vector (centroid) of each distinct community. Hub nodes exert higher geometric gravity on the partition representation.
    *   `build_colbert_index`: Passes nodes to `ragatouille` for localized per-token Late-Interaction weights.

### `llm_manager.py`
The Token-Aware Language Model Interface.
*   **Purpose**: Routes structured prompts and retrieved context string arrays securely to the Generator Model (e.g., GPT-3.5 or Llama3).
*   **Mechanics**: Implements `tiktoken` to dynamically count prompt characters. If the retrieved context exceeds `max_context_tokens`, it strictly truncates the string to prevent `ContextWindowExceeded` API exceptions.

### `engine.py`
The Application Singleton API (`CoreEngine`).
*   **Purpose**: Loads all multi-modal indices (`.index`, `.pkl`, `.pt`) from disk to RAM simultaneously, ensuring strategies do not waste memory recompiling data. Exposes unified search methods.
*   **Mechanics**:
    *   `search_dense` / `search_lexical`: Queries the FAISS and BM25 implementations directly.
    *   `get_neighbors`: Retrieves topological arrays using PyG graphs statically.
    *   **Unified Graph <-> Vector Interfacing**:
        *   `vector_to_graph_search`: Queries FAISS for semantic similarity (k=3), then physically extracts the target's node_id and executes `BFS` over the PyG graph up to N-hops, capturing the topological surrounding context.
        *   `graph_to_vector_embedding`: Accepts an array of topological `node_ids` (e.g., a clustered community), looks up their original `[384]` dimensional dense vectors from FAISS, stacks them, and runs `np.mean(axis=0)` to generate a dynamic Centroid vector representing that community mathematically.
