# L1 Partition-Routing — Method, Universal Direction, Reproducibility

_Living doc. Live numbers are regenerated into `results/L1_benchmark.md` by
`python experiments.py run l1-benchmark` (reads the canonical per-cell result files).
This doc holds the method, the design decisions, and the exact reproduction recipe._

## What L1 does
Query → rank the **partitions**; metric = **partition FullCov@20** (fraction of test queries whose
*every* gold doc is covered by some partition in the top-20). L1 is the pipeline ceiling: L2/L3 operate
*within* L1's top-20 partitions and cannot recover a partition L1 missed. Oracle@20 (greedy min-cover
over voted partitions) = 100 on all 5 datasets ⇒ every gap is pure ranking, nothing substrate-bound.

Mechanism: retrieve nodes → each node votes (RRF weight 1/(60+rank)) for its **overlap membership**
partitions (own + 1-hop-neighbor partitions); aggregate votes (sum, max); rank partitions.

## Encoders (frozen unless noted), partition FullCov@20
See `results/L1_benchmark.md` for the live table. Current canonical values (`eqrrf6+bestof@20`):

| encoder | musique | 2wiki | squad | metaqa | hotpot | mean |
|---|---|---|---|---|---|---|
| MiniLM-L6 (frozen) | 90.68 | 89.60 | 89.97 | 98.45 | 94.33 | 92.61 |
| BGE-large (frozen)  | 93.23 | 90.80 | 92.42 | 99.53 | 97.41 | 94.68 |
| BGE-large (per-dataset ft) | 95.89 | 96.07 | _pending_ | — | — | — |

- **Frozen BGE + offset stack = best *universal-recipe* number (mean 94.68, 2/5 ≥95).** Uniform recipe,
  but the offset heads are trained **per-dataset** ⇒ one recipe, five weight-sets, NOT one model.
- **Per-dataset fine-tune (ft)** clears 95 (musique +2.7, 2wiki +5.3) but = a separate encoder per
  dataset ⇒ "multiple models" ⇒ **kept only as an ablation, not the method.**

## Direction: ONE universal model (not per-dataset)
Goal = a single model that clears 95 on all 5, simple, mlpT-featured. Design:
- **One frozen encoder** shared across datasets (BGE-large now; gte-Qwen2-1.5B as the stronger option,
  encodings via a prebuilt flash_attn wheel — see below).
- **One universal `mlpT` offset head** (`MixtureHead`, K query-conditioned directions, soft-OR) trained
  **once on pooled (query→gold) pairs across all 5 datasets** in the shared frozen space. Its K directions
  absorb what separate rel_hard/rel_2hop heads did ⇒ collapses the retriever zoo into one head.
- **Simple fusion:** dense + mlpT via equal-RRF (+ best-of). Two signals, not five.
- Honest risk: a *simple universal* model clearing 95 everywhere is a high bar (frozen BGE + per-dataset
  offsets only hit 2/5); prior universal-MLP probe cost ≈ −1.5 vs per-dataset (FullCov@100, MiniLM).

## Reproducibility (paper numbers must not move)
- Query splits: `overlap_retrain.SPLIT_SEED` (locked train/val/test; official metadata respected).
- Offset-head init + batch order: `overlap_retrain.INIT_SEED`.
- Encoder fine-tune (ablation): `l1_finetune_encoder.FT_SEED=42` (seeded shuffle + init).
- Frozen encodings (BGE / gte-Qwen2): model-deterministic (no training).
- **Integrity, enforced by the benchmark:** every encoder for a dataset must share `npart` + `n_test`
  (same frozen UKB + same locked split); a mismatch is flagged, not averaged. Verified: all datasets
  consistent (2wiki MiniLM re-run at the full 1500 split; was a stale 300-query subsample).

## Artifacts (all local under `data/ukb_storage/{dataset}/`)
- Substrate: `partition_map.json`, `nodes.index`, `centroids.index`, `graph.pt`, `bm25.pkl`.
- Frozen encoders: `bge_large/` (all 5), `gte_qwen/` (in progress).
- Per-dataset ft (ablation): `ft_bge/` (musique, 2wiki; squad pending).
- Results: `results/L1_select/rerank100_{dataset}[_{subdir}].json`.

## Reproduce
```bash
# frozen encoder encoding (once per dataset; model-deterministic)
python experiments.py run reencode-ukb --backend modal --gpu --account <a> -- --datasets <d> --model BAAI/bge-large-en-v1.5 --subdir bge_large --batch 64
# per-cell rerank (offset stack -> FullCov@20)
python experiments.py run l1-rerank100 --backend modal --cpu --account <a> -- --datasets <d> --subdir bge_large
# aggregate everything into one table
python experiments.py run l1-benchmark
# universal single-model (pooled mlpT across all datasets) — see l1_universal_mlpt.py
```
