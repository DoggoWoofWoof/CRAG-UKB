# Archive

Superseded / provenance material, bundled out of the working root when the
project pivoted to the **coverage-aware Level-1** era (Jigsaw FullCov transfer).
Nothing here is deleted — it is kept for reference and reproducibility. Paper
claims for the new era should be grounded in freshly regenerated results under
`results/`, not these.

## Contents

| Path | What it is | Why archived |
| --- | --- | --- |
| `baseline_results_2026-04/` | The pre-coverage Level-1 ablation results: `level_1/`, `loss_ablation/`, `temp_ablation/`, `hnm_ablation/`, plus the SQuAD-only `level_2/`, and the old report helpers (`benchmark_analysis.md`, `generate_full_report.py`, `hnm_report.txt`, `read_metrics.py`). | These are the **frozen KL+HNM baseline**. Once the coverage loss lands the Level-1 leaderboard changes, so these no longer represent "current best" — they are the comparison baseline, not headline numbers. Regenerate the new-era tables into `results/`. |
| `legacy_scripts/` | One-off analysis/scratch scripts from earlier sessions (`audit_*.py`, `check_hnm_best.py`, `compare_infonce_logic.py`, `final_total_comparison.py`, `gen_full_hnm_tables.py`, `get_data*.py`, `get_kl_sweep.py`, `_check_best.py`, `download_volume.py`). | Ad-hoc, not part of the pipeline. Kept for provenance of how the baseline numbers were produced. |
| `legacy_reports/` | Plain-text report dumps (`audit_*.txt`, `configs_check.txt`, `final_report.txt`, `hnm_exhaustive_tables.txt`, `temp_ablation_summary.txt`, `_best_hnm.txt`). | Terminal-dump summaries; superseded by CSV/JSON artifacts. |
| `logs/` | All historical pipeline logs (`*.log`, ~60 MB incl. the 28 MB `metaqa.log`). | Noise in the root. New runs write `*.log` to the root but those are now git-ignored. |
| `recovered_chats/` | Recovered chat transcripts. | Provenance only. |
| `legacy_runners/` | The old per-experiment scripts: `run_pipeline.py`, `run_modal.py`, `run_loss_eval.py`, `run_temp_eval.py`, `run_hnm_eval.py`, `run_level2_eval.py`, `run_coverage_eval.py`, `run_coverage_modal.py`. | Superseded by the single `experiments.py` runner. Their compute bodies were extracted into `src/experiments/` (coverage, ablations) and `src/evaluation/level2.py`, and their Modal/Lightning orchestration into `src/experiments/backends.py`. |

## Live pipeline

Single entrypoint: **`experiments.py`** (repo root) → `experiments.py run <task>`.
Task bodies live in `src/experiments/` and `src/evaluation/`; compute backends
(Modal / Lightning AI) + account rotation in `src/experiments/backends.py`.

Not archived (still needed): `checkpoints/` (the frozen KL+HNM baseline checkpoints
that the coverage loss trains on top of), `data/`, `src/`, `configs/`.
