# Retrieval Strategies

This directory contains the runtime retrievers used by `SuperModel` and system-level evaluation.

## Files

- `base.py`: common `RetrievalResult` dataclass and `BaseRetriever` interface.
- `vector_rag.py`: dense FAISS plus BM25 Reciprocal Rank Fusion baseline.
- `graph_rag.py`: dense seed retrieval followed by fixed-hop graph BFS.
- `crag.py`: current C-RAG strategy with Level 1 partition routing, Level 2 partition entry/reranking, Level 3 traversal, and context formatting.
- `query_graph_gnn.py`: experimental query-graph selector that currently falls back to centroid selection unless a GIN checkpoint is wired in.

## Current CRAG Modes

`CRAG` supports these Level 1 selectors:

- `faiss_centroid`: dense query vector against partition centroids.
- `colbert_centroid`: ColBERT search over centroid representations when the index is available.
- `mlp`: trained `TextPartitionMLP` projection followed by centroid search.

It supports these Level 2 entry modes:

- `faiss`: dense cosine reranking inside selected partitions.
- `cross_encoder`: pairwise cross-encoder scoring inside selected partitions.
- `colbert`: legacy global ColBERT search filtered by selected partitions; prefer the Level 2 benchmark implementation in `src/evaluation/level2.py` (run via `python experiments.py run bench-level2`) for paper results.

## Level 3 Traversal

`CRAG` now uses deterministic priority/beam traversal rather than FIFO expansion. It keeps a scored frontier, selects nodes above `score_threshold`, expands only nodes above `expand_threshold`, caps frontier size with `beam_width`, and records traversal stats in `RetrievalResult.metadata["traversal"]`.

Important Level 3 knobs:

- `max_traverse_steps`: maximum graph-pop operations.
- `max_context_nodes`: maximum nodes retained for generation.
- `expand_top_neighbors`: number of highest-scoring neighbors queued per expansion.
- `exclude_synthetic_edges`: whether KNN index-time edges are skipped.
- `min_context_nodes`: fallback count from Level 2 seeds if traversal prunes too aggressively.

## Paper 2 Caveat

The strategy code is a runtime integration path. Publication-grade Paper 2 numbers should come from deterministic benchmark runners that export CSV/JSON with fixed splits, metrics, and checkpoint paths.
