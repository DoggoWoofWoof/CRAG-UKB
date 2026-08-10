# REALM metric-divergence experiment

This artifact tests one narrow claim: mean supporting-passage recall can look healthy while a multi-hop
retriever fails to recover the complete evidence set. It is independent of CRAG and makes no architecture
or SOTA claim.

## Audited data

| Dataset | Official split | Queries | Deduplicated split corpus | Supports/query |
|---|---|---:|---:|---|
| 2Wiki | dev | 12,576 | 56,661 | 2 or 4 |
| MuSiQue | answerable dev | 2,417 | 21,099 | 2, 3, or 4 |
| HotpotQA | distractor validation | 7,405 | 66,581 | 2 |

The runner downloads pinned official sources, verifies fixed SHA-256 checksums, maps every annotated
support to a candidate passage, and fails rather than dropping malformed questions. The corpus for each
dataset is the title/text-deduplicated union of candidate contexts supplied in that split.

## Retrieval protocol

- Encoder: `BAAI/bge-large-en-v1.5`, revision `d4aa6901d3a41ba39fb536a557fa166f842b0e09`.
- Query instruction: `Represent this sentence for searching relevant passages: `.
- Ranking: exact inner product over normalized 1,024-dimensional embeddings.
- Budgets: `K=10,20`; weakest supporting-passage rank is measured to depth 2,000.
- Control: fixed pseudo-relevance feedback `normalize(query + top1_document)`, fused with the original
  depth-100 ranking using RRF with `k0=60`.
- Uncertainty: 5,000 paired bootstrap resamples, seed `20260803`.
- Paired test: exact McNemar test over single-shot versus feedback joint-recall outcomes.

## Reproduce

Install `requirements_experiment.txt`, then run:

```powershell
python review_paper/experiment.py --datasets 2wiki musique hotpotqa --no-reuse-existing
python review_paper/validate_experiment.py --results review_paper/experiment_results.json --per-query review_paper/experiment_per_query.jsonl
```

For the repository's isolated A10G path:

```powershell
python experiments.py run review-metric --backend modal --gpu --account 1 -- --datasets 2wiki musique hotpotqa --output-dir results/review_metric_v2 --device cuda --no-reuse-existing
```

Stages live under `data/ukb_storage/_review/realm_metric_v2/<dataset>/<view-fingerprint>/`. Sources,
embeddings, and exact rankings are content-addressed; matching completed stages are reused.

## Validated headline results

| Dataset | Aggregate recall@20 | Joint recall@20 | Paired gap (95% CI) |
|---|---:|---:|---:|
| 2Wiki | 73.17 | 45.25 | 27.92 [27.47, 28.37] |
| MuSiQue | 65.51 | 34.63 | 30.88 [29.80, 31.97] |
| HotpotQA | 91.02 | 82.70 | 8.32 [7.89, 8.75] |

The validation manifest reconstructs every estimate and paired contingency cell from all 22,398
per-query records and records SHA-256 hashes for both released result artifacts.
