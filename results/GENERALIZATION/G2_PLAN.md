# G2 — Dataset-Agnostic Retrieval: Interface Audit + Experimental Plan

**Status:** audit complete; plan frozen for smoke-test phase. No large runs until smoke tests validate every metric + leakage guard. C11a result is the frozen diagnostic baseline. Nothing existing is destroyed.

**Two bottlenecks (from G1):** (1) L1 multi-hop **co-scoping** — evidence individually reachable (ANY high) but not jointly scoped (ALL collapses 99.6→71.4→27.0% over MetaQA hops); (2) L2 **parameter** transfer fails zero-shot though the **architecture** transfers on refit.

---

## 1. INTERFACE AUDIT (what actually exists, in the frozen row space)

### 1.1 Row space / substrate (per dataset)
- Doc row space = `master_nodes_{ds}.json` filtered `type!=question`, file order == `data/ukb_storage/{ds}/gte_qwen/nodes.npy` rows (`load_partition_topology`, `l1_eval_phase1.py:19`).
- Frozen L1 topology C = `scratchpad/ablation_qwen/{ds}/variant_C/partition_map.json` (+ `graph.pt`, `stats.json`). P_MAIN=50.
- Frozen dense/splade candidate signals: `data/ukb_storage/{ds}/gte_qwen/{dense_top200_all,splade_top200_all}.npy`; query embs `queries_all.npy`; eval keys `query_ids_all.json = {ids, golds, hops, split_indices, hash}`.

### 1.2 Query-entity linking (inference-safe) — **differs sharply by dataset kind**
| Kind | Datasets | Entity = | Seed source (inference-safe) | Status |
|---|---|---|---|---|
| KB-bracketed | **MetaQA** | node (`metaqa_ent_<name>`) | **literal `[...]` in question** | ✅ 100% resolve (500/500); cleanest |
| KB-topic | WebQSP | node (Freebase entity doc) | topic-entity annotation / NER match — **not in query text** | ⚠️ needs linker; defer to phase 2 |
| Text | 2Wiki, HotpotQA, MuSiQue, SQuAD | ≈ document (title) | NER on question → title/alias match | ⚠️ NER-based; no separate entity node |

MetaQA is the primary Track-A proving ground: seed given, edges real+typed, and it is the exact diagnostic that exposed the collapse.

### 1.3 Edge provenance — **frozen graph.pt is UNTYPED/MIXED** (the mandated control needs a rebuild)
- `variant_C/graph.pt` = `Data(num_nodes=40151, edge_index=[2,775576])`, **no `edge_type`/`edge_attr`** → structural and kNN are indistinguishable in it. `master_nodes.neighbors` (219,083) is likewise mixed.
- Provenance-pure sources DO exist:
  - **structural**: `data/canonical/metaqa/graph_structural.tsv` (133,582 edges, **9 relation labels**, directed) — but integer-id space (`metaqa_ent_0`). Cleaner: rebuild from official `data/original/metaqa/kb.txt` (`Entity|relation|Entity`, name-space) → **typed adjacency directly in nodes.npy row space**, no crosswalk.
  - **kNN**: `graph_knn.tsv` (97,920, cosine) — or recompute exact IndexFlatIP k=3 from `nodes.npy` (bit-faithful to `indexers.py`).
- **Provenance control** = every A/B experiment runs 3 ways: `structural_only` / `knn_only` / `structural+knn`. Tells us whether gains are real relational topology or reintroduced embedding similarity. (metaqa NER unavailable by KB design.)

### 1.4 L2 feature contract (current C11a, from `_g1_eval_metaqa.json` `E_feat_names`, 18 feats)
`c8c_oof_score, c8c_oof_rank, base_score, c_dense, c_splade, c_offset, c_mixture, c_relation, rk_dense, rk_splade, rk_offset, rk_mixture, rk_relation, rel_mask, max_contrib, second_contrib, n_top5, votes_top10`.
- **rk_*** = ranks (candidate-relative, more transferable). **c_*** = raw contributions (distribution-specific — likely the transfer failure). `c_relation`/`rel_mask` come from the masked-relation expert (KB-meaningful; but derived from corpus text, not gold relations — allowed). Track B normalizes/rebuilds this contract to be invariant.

### 1.5 LODO harness — **not clean.** No true leave-one-dataset-out implementation found (`crag_lodo.json` exists as output; logic unclear/rotating). Track B builds true LODO: train on all-except-D, freeze, eval once on D.

### 1.6 Leakage guards (must hold in every smoke test)
- NEVER use: question-node `neighbors` (they are gold answers), `query_ids_all.golds`, `hops` as a feature, dataset id, gold relations/supporting facts, target test split.
- Splits: dev/train only for development; `split_indices` defines test — untouched.
- Seed = only the inference-safe query-entity link (brackets for MetaQA).

---

## 2. TRACK A — Parameter-free L1 co-scoping (MetaQA first)

**Goal:** raise **ALL-gold co-scope coverage** for 2/3-hop without exploding candidate count. No trained L1.

**A0 (build, provenance-pure):** typed structural adjacency from `kb.txt` in name space; exact k=3 kNN from `nodes.npy`; assemble `{structural_only, knn_only, both}`. Seeds from brackets. All frozen before eval.

**A1 residual:** `r_q = q − proj_span(E) q` (E = seed-entity embeddings). Robust to 1/many/collinear/no entity (fallback: r_q=q).

**A2 real displacement scoring (parameter-free):** for seed e, neighbor v: `delta(e,v)=norm(x_v−x_e)`, `s_dir=cos(r_q, delta)`. Two variants: DIRECT-NEIGHBOR (top-M by s_dir) and OFFSET-RETRIEVAL (retrieve around `z=norm(x_e+λ·delta)`; λ,M fixed/predeclared, tuned on dev only).

**A3 direction voting:** soft `delta* = norm(Σ softmax(τ·s_dir_i)·delta_i)`; also geometric-median / cosine-clustered variants. Label-free.

**A4 multi-hop residual expansion:** after choosing delta_1, `r_1 = r_q − proj_{delta1} r_q`; pick next direction on r_1; bounded beam + hop budget + total candidate budget. Compare vs reusing r_q each hop.

**A5 structural/hybrid baseline:** seeds → bounded graph-neighbor expansion → score = fixed combo(semantic sim, directional compat, PPR/distance). Isolates whether vector shifting beats plain graph expansion.

**A6 metrics (per hop 1/2/3, per provenance):** ANY@scope, **ALL@scope**, FullCov@scope, candidate count + ceiling, recall@50/@200, graph-expansion count, latency. **Success = large ALL@2/3-hop gain at bounded scope; ANY rising alone is NOT success.** Then run **frozen C11a** on the new scope (answers Q1+Q2: was co-scoping the bottleneck?).

---

## 3. TRACK B — Dataset-agnostic L2 (no refit, no dataset id)

**B1 universal feature contract:** only cross-dataset-identical families — semantic (dense/splade sim+rank, RRF, agree/disagree), retrieval prior (seed membership, rank-vs-seed, multi-retriever support), query-local structure (seed distance, bounded walk counts, #/frac seeds connected, seed-conditioned PPR, local subgraph degree), optional geometric (A-track residual/dir compat). **Exclude** KB-only relation labels, dataset name, gold decomposition.

**B2 baselines (same pools/splits/metrics):** Dense, SPLADE, equal-RRF, val-weighted RRF, PPR-only, distance-only, RRF+PPR, linear universal, small universal MLP, current C11a.

**B3 normalization = likely the real transfer lever:** compare source-global / rank-percentile / per-query / robust median-IQR / bounded-monotonic / pool-relative. Prefer invariant transforms (candidate percentile > raw sim; degree-percentile-within-query-graph > raw degree; PPR-rank/normalized-mass > raw PPR).

**B4 TRUE LODO:** for each D — train on all-except-D, freeze, eval once on D. Compare {zero-shot universal, target-refit control, source-C11a, simple baselines}. No target-val tuning (except a labeled few-shot/refit control).

**B5 DRO (only if pooled overfits identity):** dataset-balanced / per-ds loss balance / GroupDRO worst-group / feature dropout / retriever corruption / budget+density+seed-quality variation. No dataset classifier.

**B6 linear-vs-MLP:** if normalized linear ≈ C11a in transfer, that's the scientific headline. Don't force neural complexity.

---

## 4. TRACK C — Joint (after A,B independently work) + external confirmation

Four configs for attribution: (1) old-L1+old-L2, (2) new-L1+refit-L2, (3) old-L1+universal-L2, (4) new-L1+universal-L2.

**Generalization protocol:** the 6 datasets are for mechanism/ablation/LODO only. Freeze {L1 algo, feature contract, normalization, hparams, L2 recipe, thresholds, budgets} BEFORE touching a **fresh external dataset** (unseen queries+corpus+labels+label-free frozen graph). External run is one-shot.

---

## 5. SMOKE-TEST GATE (before any scale)
S1 (Track A, MetaQA, ≤300 q/hop dev): metric harness validates ANY/ALL/FullCov match a brute-force recomputation; leakage asserts (no gold, no question-neighbors, seed∈inference-safe); A5 vs A2/A3 ALL@scope delta at fixed budget; 3-provenance split. S2 (Track B): feature builder on 2 datasets, assert identical schema/shape, leakage asserts, one linear LODO fold end-to-end. Only after S1+S2 green do we scale.

## Deliverables
`G2_A_METAQA_L1.{md,json}` (per-hop, per-provenance, +frozen-C11a-on-new-scope), `G2_B_UNIVERSAL_L2.{md,json}` (LODO matrix + normalization ablation + linear-vs-MLP), `G2_C_JOINT.{md,json}` (4-config + external). All VAL/dev; external one-shot; TEST untouched.
