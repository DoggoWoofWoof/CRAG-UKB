# Dataset-Agnostic Level 1 Protocol

## Research Question

Can one Level 1 checkpoint generate high-coverage document candidates across
heterogeneous multi-hop QA corpora without receiving a dataset identifier and
without losing more than two FullCov@100 points relative to independently
selected per-dataset models?

This is the preferred main-model direction. Per-dataset `l1-optimize` runs are
diagnostic upper bounds and component ablations, not five models to combine
into the headline system.

## Shared Architecture

`src/experiments/l1_unified.py` trains one `DatasetAgnosticRouter` in the common
MiniLM embedding space:

1. An immutable dense skip head uses the normalized query embedding.
2. A shared MLP produces `K` relational offsets around the nearest dense
   document.
3. A query-conditioned softmax gate weights dense and relational heads.
4. Weighted soft-OR scoring ranks direct document candidates.
5. The same positions score each corpus's immutable UKB centroids.
6. Fixed partition quotas (`20x5`, `10x10`, `5x20`) return at most 100
   documents and never materialize complete partitions.
7. One RRF policy per candidate budget is selected by macro validation across
   all training corpora; policies never vary by corpus.

The corpus index, centroids, and document-to-partition map are data structures,
not learned dataset-specific parameters. The checkpoint receives no dataset ID,
has no dataset embedding, and has no per-dataset temperature, head, or gate.

## Objective And Sampling

The training objective is:

```text
multi-positive KL
+ lambda_coverage * weakest-positive top-100 barrier
+ lambda_diversity * relational-head diversity
+ lambda_balance * gate-load balance
```

Each epoch gives every corpus the same number of optimizer updates. Smaller
training splits are deterministically oversampled, so MetaQA or another large
corpus cannot dominate merely through row count. Model configuration is
selected by macro-averaged mean FullCov across K=20/50/100, then mean recall.
The RRF policy at each K is selected by FullCov@K and Recall@K. Test data is not
used for selection.

## Experiment Ladder

### 1. Engineering smoke

```bash
python experiments.py smoke l1-unified
```

The smoke uses two corpora, 32 queries per split, one epoch, one relational
head, and one seed. It validates data staging, training, centroid routing,
global fusion, checkpoint persistence, candidate export, and result pullback.
Its metrics are never paper evidence.

### 2. Shared-model screening

```bash
python experiments.py run l1-unified --backend modal --account 1 -- \
  --datasets 2wiki_clean musique_clean hotpotqa_clean squad_clean metaqa \
  --run-id l1_unified_v1 --limit 15000 --epochs 40 \
  --relational-heads 1 4 8 --coverage-lambdas 0 0.25 \
  --seeds 42 --hard-negative-k 32 --eval-every 5 --patience 3 \
  --batch-size 128
```

This run screens head count and coverage loss using one seed. It selects one
architecture and three budget-specific global fusion policies over the five
validation sets.

### 3. Multi-seed confirmation

Rerun only the validation-selected architecture with seeds `42 43 44`.
Do not repeat the full grid. Report mean, standard deviation, paired bootstrap
confidence intervals, and paired McNemar tests against dense retrieval.

### 4. Leave-one-dataset-out transfer

For each target corpus, pass `--holdout-dataset TARGET`. The target contributes
neither training updates nor validation selection and is evaluated only after
the shared model and fusion are locked. These five runs distinguish actual
dataset transfer from ordinary multi-corpus training.

## Acceptance Gates

The shared model is the main CRAG Level 1 model only if:

- each in-domain corpus is within two FullCov@100 points of its
  validation-selected per-dataset `l1opt_v1` upper bound;
- macro FullCov@100 exceeds dense retrieval with paired significance or a
  practically meaningful confidence interval;
- no corpus has a severe regression hidden by the macro average;
- dense gate use, relational gate use, and partition-fusion selection are
  reported rather than treated as an opaque ensemble;
- candidate count remains exactly within the frozen 20/50/100 budgets;
- the same checkpoint and fusion policy are used for every corpus.

If one corpus fails the two-point gate, keep one shared checkpoint but allow a
validation-locked fallback to the dense skip head. Do not introduce a
dataset-specific learned model unless it is explicitly reported as an oracle
upper bound.

## Artifacts

Shared outputs:

```text
data/ukb_storage/_shared/checkpoints/L1/<run_id>/
data/ukb_storage/_shared/results/L1/<run_id>/summary.json
data/ukb_storage/_shared/results/L1/<run_id>/training_history.json
```

Frozen per-corpus candidates:

```text
data/ukb_storage/<dataset>/results/L1/<run_id>_unified/
```

The summary records corpus fingerprints, split manifests, selected model,
global fusion at each budget, dense baselines, per-seed metrics, gate
statistics, paired significance, and available per-dataset upper bounds.

## Cache And Recovery Contract

Remote staging sends immutable top-level UKB files and only the largest
locally compatible query-cache fingerprint. Cache fingerprints include the
document manifest, split IDs, encoder, and limit. Completed query caches and
checkpoints are atomically written and committed to the Modal volume at
artifact boundaries, so a preempted worker can resume rather than repeat
encoding or completed configurations.

## Claim Boundary

The model establishes a dataset-agnostic candidate generator, not a complete
RAG result. The final paper claim still requires freezing its top-100 files,
selecting Level 2 on identical candidates, running Level 3 behind the frozen
reranker, and comparing complete ingestion-to-answer systems with one reader,
corpus, budget, and evaluation contract.
