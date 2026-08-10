# Core Service Architecture

This directory contains the runtime backend for the C-RAG indexes and search APIs.

## Director Structure & File Mechanics

### `encoders.py`
The Dense Mathematical Embedder.
*   **Purpose**: Transforms natural language strings into high-dimensional numerical arrays.
*   **Mechanics**: Initializes a HuggingFace `SentenceTransformer` model. Current indexes were built with the `DenseEncoder` default, `multi-qa-MiniLM-L6-cos-v1`, unless explicitly rebuilt with a different config. The `encode()` method returns normalized `float32` vectors.

### `indexers.py`
The UKB Unified Factory Builder.
*   **Purpose**: Consumes the `StandardNode` list and mathematically compiles all 6 index views simultaneously. 
*   **Mechanics**:
    *   `build_faiss_node_index`: embeds document/entity texts only and stores normalized vectors in a `faiss.IndexFlatIP` cosine-similarity index.
    *   `build_bm25_index`: Uses `rank_bm25` Okapi algorithm to compute Term-Frequency/Inverse-Document-Frequency occurrences. 
    *   `build_pyg_graph`: Extracts raw string `neighbors` into numerical `[node_src, node_dst]` tensors. Weaves synthetic KNN edges for topologically isolated nodes using a fast `faiss.IndexIVFFlat` approximate clusterer to guarantee full connectivity. Outputs PyTorch `graph.pt`.
    *   `build_partition_map`: runs PyMETIS (with naive chunking fallback) over the graph. Current builds target about 1,000 document/entity nodes per partition.
    *   `build_faiss_centroid_index`: Calculates the **degree-weighted** average mathematical vector (centroid) of each distinct community. Hub nodes exert higher geometric gravity on the partition representation.
    *   `build_colbert_index`: Passes nodes to `ragatouille` for localized per-token Late-Interaction weights.

### `llm_manager.py`
The Token-Aware Language Model Interface.
*   **Purpose**: Routes structured prompts and retrieved context string arrays securely to the Generator Model (e.g., GPT-3.5 or Llama3).
*   **Mechanics**: Implements `tiktoken` token counting and truncation. It accepts either a list of context strings or a pre-joined context string.

### `engine.py`
The Application Singleton API (`CoreEngine`).
*   **Purpose**: Loads all multi-modal indices (`.index`, `.pkl`, `.pt`) from disk to RAM simultaneously, ensuring strategies do not waste memory recompiling data. Exposes unified search methods.
*   **Mechanics**:
    *   `search_dense` / `search_lexical`: Queries the FAISS and BM25 implementations directly.
    *   `get_neighbors`: Retrieves topological arrays using PyG graphs statically.
    *   **Unified Graph <-> Vector Interfacing**:
        *   `vector_to_graph_search`: Queries FAISS for semantic similarity (k=3), then physically extracts the target's node_id and executes `BFS` over the PyG graph up to N-hops, capturing the topological surrounding context.
        *   `graph_to_vector_embedding`: accepts an array of `node_ids`, reconstructs their FAISS vectors, and mean-pools them into a dynamic centroid. Persisted partition centroids are built in `indexers.py` with degree weighting.
