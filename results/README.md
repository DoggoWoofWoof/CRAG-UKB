# Result Artifacts

Current results are split across two roots.

## Per-Dataset Research Results

The July 2026 component studies write to:

```text
data/ukb_storage/{dataset}/results/
  L1/          candidate-generation and partition studies
  L3/          reachability, traversal, solver, and SRW studies
  baselines/   flat BM25/dense/hybrid/cross-encoder baselines
  cross/       L1-to-L3 bottleneck studies
```

Cross-dataset summaries are in `data/ukb_storage/_index/`, including:

- `l1_headtohead_summary.json`
- `l1_mlpt_improve_summary.json`
- `l3_solvers_{dense,champion}_summary.json`
- `l3_srw_{dense,champion}_summary.json`

These files are the source for the current exploratory numbers.

## Top-Level Results

The top-level `results/` tree contains coverage/overlap ablations, research
probes, legacy Level 1 exports, and logs produced by the unified task runner.
The intended integrated pipeline outputs are:

| Subdirectory | Intended producer |
| --- | --- |
| `level_1/` | `experiments.py run bench-level1` |
| `level_2/` | `experiments.py run bench-level2` |
| `level_3/` | `experiments.py run bench-level3` |
| `generation/` | `src.evaluation.benchmark_generation` |

The validation-locked Level 1 runner now exports
`data/ukb_storage/{dataset}/results/L1/{run_id}/candidates_test.jsonl`.
The active `l1opt_v1` sweeps are not yet frozen paper results: their winning
configurations require multi-seed confirmation, followed by a matched Level 2
reranker sweep and Level 3/generation evaluation. Until those stages share the
same frozen candidate IDs, no result bundle should be described as a complete
end-to-end C-RAG run.

Level 1 reusable artifacts live beside the dataset:

| Path | Contents |
| --- | --- |
| `cache/L1/{data_fingerprint}/` | Query tensors, dense seeds, gold IDs, and BM25 rankings |
| `checkpoints/L1/{run_id}/` | Signature-checked partition routers and relational models |
| `results/L1/{run_id}/` | Validation selection, test metrics, histories, and candidate export |

## Provenance

- `archive/baseline_results_2026-04/`: pre-coverage baseline results.
- `archive/pre_repartition_2026-07/`: results from the earlier partitioning.
- `archive/leaked_2026-07/`: invalidated outputs produced on leaked substrates.

Do not mix metrics across these eras. Match dataset name, clean-substrate
version, split, query limit, seed source, candidate budget, and metric
definition before comparing runs.

## Publication Guardrail

The current July tables are exploratory because methods were iterated while
observing the internal test split and several runs use only 500 queries. Final
paper tables must:

- tune on validation data only;
- use untouched official test sets;
- compare identical per-query candidate pools and budgets;
- include query-level uncertainty/significance;
- include multiple training seeds for learned models;
- report retrieval, answer quality, cost, and scalability together.
