# results/L2 archive — superseded benchmark results

Moved here 2026-08-18 during repo cleanup. **Nothing deleted** — these are older/superseded versions
of results that have been replaced by the current canonical set (see `results/L2/*.json`). Kept for
provenance. Safe to consult; do not cite in the paper (use the current files).

## Why each was archived

- **`L2_seed_*.json`** — early L2 rerank-seed sweeps (top-k / K variants). Superseded by the composed
  L2 inside `e2e_full6_universal_gte_qwen.json` (dense+SPLADE+offset head+adapter).
- **`candgen_*.json`** — L1 candidate-generation sweeps. Superseded; candgen folded into the pipeline.
- **`dense_adapter_*.json`** (text / gte / webqsp) — standalone adapter ablations. The adapter is now a
  live L2 signal in the full-6 run.
- **`e2e_multitask_*`, `e2e_transfer_*`, `learned_fusion_*`** — exploratory multitask/fusion variants,
  superseded by the min-rank + NER-L3 composition.
- **`e2e_ner_gte_qwen.json`, `e2e_ner_webqsp_zeroshot_*`** — earlier NER-edge e2e runs.
- **`graphlift_*.json`** — L3 graph-lift experiments on the **old ID-polluted WebQSP substrate** and
  earlier text runs; superseded by the tuned L3 (n_seed=2, w=3.0) + clean WebQSP.
- **`overlap_test_*`, `mem_bench_*`, `l3_traverse_summary.json`, `hpr_headtohead.json`** — older
  partition/overlap/memory/head-to-head probes, superseded.
- **`webqsp_anchored_gte_qwen.json`** — the first *dense-only, local* anchored WebQSP eval; superseded
  by the full-pipeline `webqsp_anchored_fullstrength.json`.

## Current canonical results (kept in results/L2/)

- `e2e_full6_universal_gte_qwen.json` — the one joint head+adapter, all 6 datasets, open-corpus Recall@k.
- `e2e_pipeline_anchored_gte_qwen.json` + `webqsp_anchored_fullstrength.json` — anchored (KGQA-comparable) WebQSP.
- `e2e_pipeline_qa_gte_qwen.json` — fixed-reader QA (EM/F1) across all 6.
- `unified_metrics.json` — the one-metric-all-datasets table.
- `crag_variants.json` — zero-shot vs trained vs SOTA (Recall@5).
- `zeroshot_transfer.json` — leave-one-out head transfer.
- `ner_edge_ablation.json` — NER vs kNN edge ablation (the NER-edge win).

Backups (pre-cleanup / pre-L3-tune / paper-number snapshots) live in `results/L2/_backups/`.
