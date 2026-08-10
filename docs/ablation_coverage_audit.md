# CRAG Ablation Coverage Audit

Audit date: 2026-07-25

This document answers two separate questions:

1. Has the research idea already been implemented and tested somewhere?
2. Is the available evidence valid for a final paper table?

Those are not equivalent. CRAG has explored most of the Level 1 design space,
but much of the July component work was developed while observing the internal
test split. It is useful directional evidence, not final model-selection
evidence.

## Evidence Classes

| Class | Meaning | Permitted use |
| --- | --- | --- |
| A | Clean substrate, validation-selected, test evaluated after selection | Main paper table |
| B | Clean substrate, but configurations were iterated or selected using test results | Method screening and motivation only |
| C | Pre-repartition or old non-clean substrate | Historical context only |
| D | Explicitly leaked or invalidated substrate | Negative directional evidence only |
| E | Code or cache exists, but no compatible completed result | Not evidence |

The active validation-locked `l1-optimize` runs are the first intended Class A
Level 1 results. Until their selected configurations are confirmed over
multiple seeds, all earlier Level 1 result files remain Class B or lower.

## Known Protocol Traps

These details are easy to miss when reading filenames or summary JSON alone:

- historical overlap-retraining McNemar vectors named `_fc20_vec` were actually
  computed at `K=50`; the source label is now corrected for future runs, while
  existing significance results must be described as FullCov@50;
- `adaptive_k.py` uses the true number of gold partitions to set each query's
  budget, so it is an oracle-ceiling probe rather than a deployable router;
- `query_decomp.py` evaluates the final checkpoint, not the
  validation-selected best checkpoint;
- `l1_full.py` chooses the best retriever subset and relational RRF weight using
  test Recall@100;
- `l1_mlpt_improve.py` selects head count, epoch count, and diversity
  configuration using test FullCov@100;
- `l1_universal.py` reports the better standard/residual variant after comparing
  them on test;
- the historical temperature values came from old-substrate sweeps, and no
  clean validation-only temperature sweep currently exists;
- query-encoder fine-tuning evaluates the test split after every epoch.

None of these invalidate the corresponding exploration, but all prevent those
artifacts from being promoted directly into a final Class A paper table.

## Executive Verdict

The architecture search is largely saturated. We should not start another
open-ended encoder, loss, gating, decomposition, overlap, or multi-vector
campaign. The useful findings are already stable:

- frozen encoder replacement did not help;
- query-encoder fine-tuning did not beat the structural and relational stacks;
- KL-family multi-positive training is substantially safer than the tested
  multi-positive InfoNCE formulation;
- coverage-enhanced KL is a small, dataset-dependent lever;
- full-softmax training is generally best or tied with HNM;
- overlap, kNN membership, and NER edges can inflate partition FullCov by
  expanding document membership, but violate the intended 20-100 document
  budget when whole partitions are materialized;
- multi-prototype partition vectors are mixed rather than transformative;
- direct relational document candidates are the strongest Level 1 addition;
- trained two-hop and soft-OR multi-head offsets help, while naive mixtures,
  universal heads, residual weighting, learned fusion gates, and query
  decomposition do not;
- typed PPR is the strongest robust Level 3 family, but the historical solver
  comparisons are not validation-locked.

The remaining work is experimental hygiene and integration, not another broad
method search.

## Encoder And Representation Ablations

| Family | Coverage | Main evidence | Verdict | Class |
| --- | --- | --- | --- | --- |
| Frozen MiniLM vs BGE-base | 2Wiki; hard, overlap1, overlap1+knn1 | `archive/leaked_2026-07/results/encoder_upgrade/2wiki.json` | BGE is lower at FullCov@20 by 0.87, 1.14, and 0.93 points respectively. Do not reopen as a CRAG optimization. | D |
| Query-encoder fine-tuning | 2Wiki, MuSiQue, HotpotQA | `data/ukb_storage/2wiki_clean/results/L1/encoder_ft.json`; `results/finetune_ablation/*_encoder_finetune.json` | Clean outputs exist, but `encoder_finetune.py` evaluates test every epoch. FullCov@20 is 28.13/71.38/63.05 on 2Wiki/MuSiQue/HotpotQA and is below stronger structural or relational stacks. | B |
| Multi-prototype partition vectors | 2Wiki, MuSiQue, HotpotQA; 1/2/4 prototypes | `data/ukb_storage/2wiki_clean/results/L1/multiproto__*.json`; `results/multiproto/*.json` | Mixed. At FullCov@100, c1 to c4 changes 76.40 to 83.73 on 2Wiki overlap1, 97.79 to 98.05 on MuSiQue, and 92.87 to 94.17 on HotpotQA. Gains are small or negative at tight budgets/configurations. | B |
| Multi-signal router vectors | All five datasets; q, q+seed, q+seed+neighbour | `data/ukb_storage/*/results/L1/multisignal.json` | Modest routing gain on four datasets and a small MuSiQue loss. FullCov@100 lift for q+seed+neighbour is +1.67 2Wiki, -0.50 MuSiQue, +3.73 HotpotQA, +1.09 SQuAD, and +3.60 MetaQA. Keep as an optional partition prior. | B |
| Multi-head document offsets | 2Wiki, MuSiQue, MetaQA; K=4 and partial K=6/8 sweep | `data/ukb_storage/*/results/L1/mlp_transformer.json`; `mlpt_improve.json`; `headtohead.json` | Works. K=4 beats the single offset at Recall@100 by +6.08 2Wiki, +1.48 MuSiQue, and +5.84 MetaQA. K=6/8 adds little except +1.2 FullCov@100 on MetaQA. The improvement script selects head/epoch/diversity settings on test, so only the direction is settled. More heads are low ROI. | B |
| Set-mixture vectors with diversity | MetaQA and SQuAD | `data/ukb_storage/{metaqa,squad_clean}/results/L1/mixture.json` | Diversity prevents literal collapse, but MetaQA Recall@100 is 46.29 versus 53.05 for the single head. Drop. | B |
| Token-level ColBERT vectors | Old 2Wiki/MuSiQue/SQuAD indexes and MetaQA caches | `data/ukb_storage/{2wiki,musique,squad,metaqa}/colbert_*`; archived Level 2 files | Implementation and caches exist, but no matched clean Level 1 top-100 result exists. Old SQuAD ColBERT FullCov@20 was 80.26 over a 19,035-document pool on the old substrate. It cannot be used as a current result. | C/E |

Conclusion: encoder and multi-vector ideas have been covered at the partition,
document-head, and token-interaction levels. The only justified ColBERT work is
a clean matched-pool baseline, not a new architecture campaign.

## Objective And Optimization Ablations

| Family | Coverage | Main evidence | Verdict | Class |
| --- | --- | --- | --- | --- |
| BCE, single/multi InfoNCE, KL | Historical all-dataset sweep plus current corrected studies | `archive/baseline_results_2026-04/loss_ablation/`; `data/ukb_storage/*/results/L1/loss_ab*.json` | Multi-positive KL is the reliable baseline. The tested multi-positive InfoNCE collapses on several datasets; single-positive InfoNCE is only an ablation. | B/C |
| Coverage-enhanced KL | 2Wiki, MuSiQue, MetaQA corrected document-level A/B; broader partition studies | `data/ukb_storage/_index/l1_loss_ab_summary.json` | Mean FullCov@100 is 50.63 versus 50.42 for KL: +2.28 MetaQA, +1.01 MuSiQue, -2.67 2Wiki. Keep in the validation grid, but do not make it the sole objective. | B |
| Temperature | Old four-dataset sweep | `archive/baseline_results_2026-04/temp_ablation/` | Extensively swept, but values were selected on the old substrate. Existing values are reasonable screening defaults, not final evidence. | C |
| Hard-negative count | Clean sweeps on all five datasets | `data/ukb_storage/{2wiki_clean,metaqa}/results/L1/hnm_sweep.json`; `results/overlap_ablation/*_hnm_sweep.json` | Full softmax is best or effectively tied on 2Wiki, MetaQA, HotpotQA, and MuSiQue. SQuAD has a small tight-budget peak around 32-64 negatives but no durable broad-budget advantage. Do not spend another sweep on HNM. | B |
| Residual/missed-positive weighting | All five datasets | `data/ukb_storage/_index/l1_universal_summary.json` | Residual training reduces mean fused Recall@100 by 0.57 and relational-only recall by 0.70. Drop. | B |
| Graph-regularized router training | 2Wiki, MuSiQue, HotpotQA, MetaQA | `data/ukb_storage/*/results/L1/relation_graphtrain.json` | Approximately null: +1.07 FullCov@100 on 2Wiki, -0.25 on MetaQA, with similarly small changes elsewhere. Not a main lever. | B |

## Partition And Membership Ablations

| Family | Coverage | Main evidence | Verdict | Class |
| --- | --- | --- | --- | --- |
| Target partition size | 2Wiki and MetaQA detailed grids, plus current 100-doc substrates | `data/ukb_storage/{2wiki_clean,metaqa}/results/L1/{partition,overlap_partsize}.json` | Around 100 documents per partition gives the strongest equal-pool frontier in the representative text and KB datasets. A five-dataset final efficiency check remains useful, but another architecture sweep is unnecessary. | B |
| Structural-only vs structural+kNN METIS | 2Wiki and MetaQA detailed grid | Same partition files | Semantic kNN edges improve partition quality in text data; structural edges are essential in MetaQA. Keep typed edge provenance. | B |
| Hard, overlap1/2, syn1, kNN1/2/3, combinations | 2Wiki, MuSiQue, HotpotQA, MetaQA; SQuAD only partial | `structure_sweep.json`; `results/overlap_ablation/*S2*.json` | Ranking metrics rise strongly, but membership rises too. Example: 2Wiki overlap1+knn3 has 5.43 memberships/doc; overlap2 has 425.8. These cannot be interpreted as 20-100 document retrieval. | B |
| NER membership edges | 2Wiki, MuSiQue, HotpotQA, MetaQA | `results/overlap_ablation/*_Sner.json` | Raw FullCov rises, but memberships reach about 12-17 per document. This is evidence that entity structure is useful, not evidence that materializing NER-overlapped partitions is efficient. Use NER as typed traversal edges or a bounded score, not unrestricted membership. | B |
| SPLADE membership edges | Implementation only | `src/pipeline/splade_edges.py` | No `splade_edges_*.json` result exists. This is the main membership atom that was planned but not actually completed. It should not block Level 1 because unrestricted sparse overlap would have the same pool-explosion risk. | E |
| Adaptive K | 2Wiki, MuSiQue, HotpotQA | `results/adaptive_k/*.json`; current 2Wiki UKB copy | Oracle upper bound: it uses the true gold-partition count. Even with that information it improves FullCov only by spending thousands of documents, so a learned predictor is not a priority for the intended 20-100 document budget. | B |
| Query decomposition | 2Wiki, MuSiQue, HotpotQA | `data/ukb_storage/2wiki_clean/results/L1/query_decomp__*.json`; `results/query_decomp/*.json` | Dominated at matched average partition count. Example: 2Wiki hard baseline FullCov 24.0 at 20 parts versus decomposition 23.07 at 26.4 parts. The script also evaluates the final rather than validation-best checkpoint. Drop. | B |

## Direct Candidate Generation And Fusion

| Family | Coverage | Main evidence | Verdict | Class |
| --- | --- | --- | --- | --- |
| Dense document retrieval | All five datasets | `relational_pool.json`, `confirm.json`, and current flat baselines | Necessary control and complementary source. It saturates HotpotQA/SQuAD but fails badly on MetaQA. | B; current flat baseline A-protocol |
| Learned relational offset | All five datasets | `data/ukb_storage/*/results/L1/relational_pool.json` | Strongest new candidate source. At Recall@100, equal fusion adds +31.61 MetaQA, +13.58 MuSiQue, +3.30 2Wiki, is near-neutral on saturated datasets. | B |
| Hard-negative relational head | All five datasets | `ablation.json`, `full.json`, `confirm.json` | Most robust relational variant and part of the uniform champion. Keep. | B |
| Multi-seed relational head | All five datasets | Same files | Useful mainly on MuSiQue; not a uniform replacement. Keep only if validation selects it. | B |
| Cheap repeated multi-hop | All five datasets | `ablation.json` (`rel_mhop`) | Neutral or worse because it reuses the same offset. Drop. | B |
| Trained two-hop head | All five datasets | `dynamic.json`, `full.json`, `confirm.json` | Real but dataset-dependent gain. On full confirmation it adds +6.10 Recall@100 over dense+hard on MetaQA and roughly +1 point on 2Wiki/MuSiQue. Keep in the validation grid. | B |
| Exhaustive retriever subsets | All five datasets | `full.json`; `_index/l1_full_summary.json` | All 63 subsets were tested, but the winning subset and RRF weight were selected on test Recall@100. `dense+rel_hard+rel_2hop` is a strong screening recipe; it must be reselected on validation before final use. No need for another open-ended subset search. | B |
| Fixed weighted RRF | All five datasets | `ablation.json`, `full.json`, `confirm.json` | Robust. Best relational weight tracks source complementarity, but validation-selected RRF is safer than a learned gate. | B |
| Learned/confidence/overlap gates | All five datasets | `dynamic.json`, `fusion.json`, `_index/l1_universal_gate_summary.json` | Failed to beat equal RRF on average. Universal gate mean Recall@100 is 80.52 versus 80.73 equal RRF; listwise learned gate loses on all five. Drop as a main contribution. | B |
| Universal cross-dataset head | All five datasets | `_index/l1_universal_summary.json` | Costs 1.46 mean Recall@100 versus per-dataset training. The summary chooses standard versus residual after comparing test results. Useful practicality ablation, not the champion. | B |

## Sparse, Hybrid, And Multi-Vector Baselines

The new validation-locked flat-retrieval harness is
`src/experiments/sota_baselines.py`. It uses the exact clean split manifests,
selects hybrid weights on validation, evaluates test once, stores candidate IDs,
and reports McNemar tests.

Completed `sota_core_v1` test results:

| Dataset | Method | FullCov@20 | FullCov@100 | Recall@100 |
| --- | --- | ---: | ---: | ---: |
| 2Wiki | dense | 27.93 | 36.93 | 67.72 |
| 2Wiki | validation-tuned BM25+dense RRF | 31.13 | 39.13 | 69.33 |
| MuSiQue | dense | 35.49 | 52.58 | 77.04 |
| MuSiQue | validation-tuned BM25+dense RRF | 39.10 | 54.99 | 78.50 |
| HotpotQA | dense | 82.01 | 89.79 | 92.22 |
| HotpotQA | equal/tuned BM25+dense RRF | 94.98 | 97.57 | 98.14 |
| SQuAD | dense | 77.78 | 87.45 | 90.50 |
| SQuAD | validation-tuned BM25+dense RRF | 81.06 | 89.73 | 93.03 |
| MetaQA | dense | 6.00 | 9.23 | 12.66 |
| MetaQA | validation-tuned BM25+dense RRF | 6.13 | 9.23 | 12.66 |

Artifacts are under
`data/ukb_storage/{dataset}/results/baselines/sota_core_v1/`. All five core
runs are complete. SQuAD used 13,033 held-out test queries; its tuned
dense:BM25 RRF weight of 1:0.5 was selected on 15,000 validation queries.

Historical SPLADE and ColBERT evidence is not compatible:

- old SQuAD used the non-clean `squad` substrate and a roughly 19,000-document
  pool;
- old 2Wiki and MuSiQue files are quarantined under `archive/leaked_2026-07/`;
- the deleted historical MetaQA result in git contains only `no_rerank`; its
  log stopped during Level 1 precomputation and never evaluated ColBERT or
  SPLADE;
- caches are stored under a mix of old aliases and lack the new run
  fingerprint;
- no current clean top-100 candidate pool has been reranked by SPLADE, ColBERT,
  or a cross-encoder.

Therefore sparse+dense fusion is now covered with BM25, while SPLADE, ColBERT,
and cross-encoder are implemented historically but still missing as valid
matched-pool paper baselines.

## Artifact And Runner Hygiene

Two aggregate indexes are incomplete even though their per-dataset artifacts
exist:

- `_index/l1_fusion_summary.json` contains only MetaQA despite five
  per-dataset `fusion.json` files;
- `_index/l1_mlpt_improve_summary.json` contains only MetaQA despite completed
  files for 2Wiki, MuSiQue, and MetaQA.

The standalone scripts overwrite these aggregates with only the datasets in the
current invocation. The individual result files remain usable, but claims
should be regenerated from them rather than trusting the stale indexes.

Experiment registration is also fragmented: the central task catalogue covers
the principal overlap, encoder, multiprototype, decomposition, SOTA, and Level
2 runners, while several relational, fusion, universal, multisignal, and
partition-size ablations remain standalone CLIs. This is an operational cleanup
issue, not an untested research direction.

## Level 3 Ablations

Level 3 has broad method coverage:

- dense seeds;
- one-hop and two-hop expansion;
- PPR with alpha sweeps;
- APPNP;
- query-weighted PPR variants;
- graph-ball and bridge methods;
- cover-greedy/cover-graph;
- bounded best-first traversal;
- typed-edge reweighted PPR;
- relational-offset reachability;
- dense versus relational/champion seeds;
- reachability, pool ceiling, recovery, and latency probes.

Primary artifacts:

- `data/ukb_storage/*/results/L3/methods*.json`
- `data/ukb_storage/*/results/L3/solvers*.json`
- `data/ukb_storage/*/results/L3/srw*.json`
- `data/ukb_storage/*/results/L3/traverse.json`
- `data/ukb_storage/*/results/L3/reachability.json`
- `data/ukb_storage/*/results/cross/l1l3.json`

The robust finding is that PPR-class diffusion benefits relationally reachable
datasets and better restart seeds. Examples at FullCov@100 include MetaQA
dense 10.4 versus PPR 43.0, and champion-seeded PPR 55.8; MuSiQue dense 51.8
versus champion-seeded PPR 74.6. HotpotQA and SQuAD are already dense-saturated.

These are Class B because most use 500 internal test queries and several scripts
select PPR alpha or solver settings on test. They prove the mechanism, but they
must be rerun behind one frozen Level 1/2 pool with validation-selected
parameters.

## Do Not Reopen

The following directions have enough negative or diminishing-return evidence:

- BGE/E5-style encoder replacement as a CRAG optimization;
- further query-encoder fine-tuning before the pipeline is frozen;
- another temperature or HNM sweep;
- naive multi-positive InfoNCE;
- unrestricted overlap/NER/SPLADE partition membership;
- more partition prototypes without a new matched-budget hypothesis;
- more than 4-8 offset heads;
- naive mixture-of-directions;
- cheap repeated offset hops;
- residual missed-positive weighting;
- learned fusion gates using the current objectives;
- heuristic query decomposition;
- graph-regularized Level 1 training;
- CPU GNN sweeps as a priority.

## Remaining Publication-Critical Work

Only the following experiments are genuinely missing:

1. Finish `l1-optimize` as per-dataset upper bounds, screen `l1-unified`, then
   confirm its globally validation-selected configuration with three seeds and
   leave-one-dataset-out transfer.
2. Freeze one top-100 candidate-ID file per dataset.
3. Run every Level 2 method on those identical IDs: dense, BM25, SPLADE,
   ColBERT, cross-encoder, and calibrated fusion.
4. Select the Level 2 reranker and seed budget on validation only.
5. Rerun typed PPR and required Level 3 ablations behind the frozen Level 2
   seeds, with no test-time alpha or solver selection.
6. Compare partition-local streaming PPR with full-graph PPR for equivalence,
   partitions loaded, latency, memory, and throughput.
7. Add current named multi-hop/graph-RAG baselines under matched corpora and a
   common reader.
8. Run answer generation and report EM/F1, support F1, faithfulness, context
   tokens, latency, and cost.

This is a much narrower program than inventing another Level 1 architecture.
The evidence says to consolidate the best existing components and make the
evaluation protocol defensible.
