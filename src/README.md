# `src/` Package Map

The current package root is organized around the implemented C-RAG pipeline:

```text
src/
  alignment/   Offline MLP/GNN training and ablation losses
  core/        Runtime indexes, encoders, CoreEngine, and LLM manager
  evaluation/  Level 1, Level 2, and generation evaluation utilities
  experiments/ Component studies and unified experiment task bodies
  pipeline/    Dataset loaders and StandardNode normalization
  strategies/  Runtime retrieval strategies
```

## Dependency Direction

- `pipeline/` creates normalized data and should not depend on retrievers.
- `core/` loads persisted artifacts and exposes search APIs.
- `alignment/` trains query-to-partition models from `core` artifacts.
- `experiments/` may train direct document candidate generators and evaluate
  graph diffusion independently of the integrated runtime.
- `strategies/` use `core` APIs at query time.
- `evaluation/` may use `core`, `alignment`, and `strategies` depending on the benchmark.

## Important Runtime Invariants

- `CoreEngine.nodes` contains document/entity nodes only and aligns with FAISS, BM25, graph, and partition indexes.
- `CoreEngine.all_nodes` includes question nodes and is used for training/evaluation ground truth.
- Question nodes are filtered out before indexing and partitioning.
- The active research definition of Level 1 is broad candidate generation:
  partition routing plus dense, sparse, and learned relational document
  retrievers. Several July `l1_*` experiments rank documents directly.
- `experiments/l1_optimize.py` is the validation-locked Level 1 runner. It
  combines direct document signals with bounded partition-router quotas,
  selects model/fusion settings on validation, exports an exact top-100 test
  pool, and persists fingerprinted caches/checkpoints under each dataset UKB.
- `src/experiments/l3_srw.py` is a full-graph component harness. It does not
  demonstrate partition-local dynamic execution or the complete runtime.
- Paper 2 system claims require one frozen Level 1 candidate export, a matched
  Level 2 reranker comparison, Level 3, and real generation metrics.
- Current structured July component results live primarily under
  `data/ukb_storage/{dataset}/results/`, not only top-level `results/`.
