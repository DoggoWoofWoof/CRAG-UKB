# Evaluation

This directory contains benchmark and metric utilities for both paper tracks.

The complete full-ingestion SOTA protocol, pinned baselines, common reader, and
end-to-end metrics are documented in
[`docs/sota_end_to_end_protocol.md`](../../docs/sota_end_to_end_protocol.md).
Its matched track is the source of truth for whole-system paper claims.

The deterministic integrated benchmarks in this package are distinct from the
July component studies in `src/experiments/`. Current MLP-offset and SRW results
live under `data/ukb_storage/{dataset}/results/` and do not yet pass through one
frozen Level 1 -> Level 2 -> Level 3 -> generation run.

## Files

- `benchmark_partition_selection.py`: Level 1 partition-routing benchmark. It compares centroid, dense-vote, **bm25-vote (lexical)**, **selectivity_route (offline IDF routing: lexical / dense / hybrid-RRF)**, MLP, MLP-topology, and GNN selectors on deterministic 70/20/10 splits. Methods retrieve the full partition ranking so FullCov@K and the weakest-positive-rank diagnostic are well-defined.
- `benchmark_reranking.py`: older/local Level 2 reranking benchmark for no-rerank, dense, and cross-encoder variants.
- `benchmark_level3.py`: Level 3 traversal/context benchmark. It compares Level 2-only contexts against traversal with and without synthetic KNN edges.
- `benchmark_generation.py`: Paper 2 generation-evaluation scaffold. It can build retrieved-context JSONL files and score prediction files with EM/F1.
- `sota_contract.py`: immutable full-corpus adapters plus the common retrieval, generation, efficiency, and bootstrap-CI evaluation contract.
- `external_sota_adapter.py`: resumable adapters executed inside pinned HippoRAG/GFM-RAG environments.
- `metrics.py`: legacy strategy-level metrics helper around `SuperModel`.
- `benchmark_gen.py`: compatibility wrapper for generation evaluation utilities.
- `ground_truth.py`: legacy synthetic benchmark helper, superseded by question-node ground truth in `master_nodes.json`.

## Metrics

- `Recall@K`: whether at least one correct partition/document appears in the top K.
- `GT Recall@K`: fraction of all required evidence partitions/documents found.
- `Full Coverage@K`: whether **all** annotated evidence items are found by K.
  It is a useful retrieval precondition for multi-hop answering, but its
  relationship to answer correctness still has to be measured with a real
  reader/generator.
- `weakest_positive_rank`: 1-indexed rank of the **worst-ranked** required partition (the depth at which full coverage is reached) — the Jigsaw "worst-required-item" diagnostic that the coverage loss is designed to shrink. Uses a uniform `num_partitions+1` miss sentinel so it is comparable across methods.
- `route_lexical` / `route_hybrid`: for `selectivity_route`, the % of queries routed to the lexical arm / hybrid fusion.
- `MRR`: reciprocal rank of the first correct hit.
- `NDCG@K`: rank-sensitive relevance over multi-evidence queries.
- `EM/F1`: downstream answer metrics for Paper 2 generation evaluation.

For SIGIR/KDD claims, prefer exported CSV/JSON artifacts over terminal logs.
Do not select configurations on the test split. Several July exploration
scripts currently do this and must be converted to train/dev selection followed
by a single untouched-test evaluation before their numbers are publishable.

## Coverage-loss ablation (Level 1)

The `train-coverage` task (body in `src/experiments/coverage.py`) trains the coverage-aware loss over a lambda sweep against the frozen best KL+HNM baseline and runs a **paired exact McNemar test on FullCov@20**:

```bash
python experiments.py run train-coverage -- --datasets 2wiki musique --lambdas 0.1 0.25 0.5 1.0 --epochs 100
# fast end-to-end smoke run locally (caps training AND eval, isolated _lim checkpoints):
python experiments.py run train-coverage --backend local -- --datasets 2wiki --lambdas 0.5 --limit 500 --epochs 5
```

Outputs: `results/coverage_ablation/{dataset}_coverage_ablation_results.csv` and `comparison_{dataset}_coverage.json` (with a `significance` block).

## Level 3 traversal benchmark

Run a small smoke benchmark:

```bash
python -m src.evaluation.benchmark_level3 --dataset squad --seed-reranker auto --limit 100
```

Full outputs are written to `results/level_3/{dataset}_level_3_traversal.json` and `.csv`.

## Paper 2 context export

Build generator-ready retrieval contexts:

```bash
python -m src.evaluation.benchmark_generation build-context --dataset squad --output results/generation/squad_contexts.jsonl --limit 100 --use-level3
```

Score a JSONL prediction file with `prediction` and `answers` fields:

```bash
python -m src.evaluation.benchmark_generation score --predictions results/generation/squad_predictions.jsonl --output results/generation/squad_scores.json
```
