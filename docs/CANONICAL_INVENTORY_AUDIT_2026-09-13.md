# CANONICAL DATASET INVENTORY AUDIT — READ ONLY (2026-09-13)

Nothing was built, encoded, retrieved, indexed, partitioned, patched, deleted or overwritten. Every number below
was read from disk in this window (`audit_six.py`, `webqsp_extra.py`, record reads); no TEST result was used.
Governing records: `CANONICAL_FREEZE.json` `58958f33a3af74c2…` (STATUS FROZEN, 2026-09-12T22:27:25Z, self-hash OK)
and `VERIFICATION.json` (2026-09-13T00:10:16Z, mode full, **686 checks / 0 failed, PASS**, for that freeze).
`_history/logs/HANDOFF_CLAIMS_CHECK.json`: ALL_PASS. **Not one pinned file changed size or mtime after the
verification run** (metaqa 52 / 2wiki 345 / musique 34 / hotpotqa 302 / squad 35 / webqsp 217 pinned files re-checked).

Disk warning: free space fell from 27 GB to **9.9 GB (99 % full) during this audit** — not from this session
(it wrote ~1 MB of JSON). Three foreign python processes are running (`scripts/m3b_contract.py`,
`scripts/m3b_relations.py`, and a `dry_compile.py` from the *message-passing-retrieval* session). Left untouched.

---

## §1 Dataset roots (exact paths, followed from the manifests)

**Served package (the only canonical copy)** — `data/final_canonical/` pinned by `CANONICAL_FREEZE.json`:

| tree | DATASET.json (RECORD_SHA256) | frozen | tree GB |
|---|---|---|---:|
| `data/final_canonical/metaqa/` | `8ab4506933d0…` | 2026-09-12T00:11:15Z | 6.88 |
| `data/final_canonical/2wiki/` | `ddb083a8f8cc…` | 2026-09-12T00:12:57Z | 34.88 |
| `data/final_canonical/musique/` | `67fa637183fb…` | 2026-09-12T00:11:29Z | 1.13 |
| `data/final_canonical/hotpotqa/` | `2f778679a52f…` | 2026-09-12T00:12:08Z | 28.08 |
| `data/final_canonical/squad/` | `29976120cf08…` | 2026-09-12T00:11:26Z | 2.43 |
| `data/final_canonical/webqsp/` | `56c8ed2352c5…` | 2026-09-12T00:13:18Z | 15.46 |
| `data/final_canonical/freebase/` (§8 only) | `d3b9324b1c28…` | 2026-09-12T13:02:26Z | 79.22 |

Layout per dataset (identical): `nodes.jsonl` (line i = position i) · `queries/<split>.jsonl` + `queries/query_ids.json`
(+ `queries/lanes/` on the five) · `embeddings/{dense,splade}/{docs,queries}/shard_%05d.{npy,npz}` + `manifest.json`
· `graph/{structural,ner,knn}.npz` + `GRAPH_MANIFEST.json` · `retrieval_cache/{dense,splade}_top1000.npz`
(+ `.meta.json` on 2wiki/hotpotqa) · `DATASET.json`. Loader `data/final_canonical/canonical.py` (`Dataset`, `Embeddings`,
`Graph`, `Freebase`), gates `src/dataset_canonical/{verify_canonical,check_handoff,freeze_canonical}.py`.
Totals: 13,997,071 nodes / 877,119 queries / 147,251,785 edges (18 families) / 24 embedding channels (60.0 GB) / 88.85 GB.

**Legacy substrate — ⚠ stale/conflicting universes (what every L1 script reads; NOT deleted, NOT served):**

| root | what | universe |
|---|---|---|
| `data/ukb_storage/{metaqa,2wiki_clean,musique_clean,hotpotqa_clean,squad_clean,webqsp}/gte_qwen/` | `nodes.npy` (fp32 1536), `queries_all.npy`, `dense_top200_all.npy`, `splade_top200_all.npy` (int64 [nq,200]), `query_ids_all.json` (ids/split_indices/golds/hops), `partition_map.json` (topology A), `graph.pt`, `meta.json` | docs 40,151 / 65,865 / 13,672 / 507,494 / 19,029 / 781,485; queries 407,513 / 15,000 / 19,938 / 97,852 / 130,319 / 1,578 |
| `data/ukb_storage/<legacy>/{ner_edges_w_df25.pkl, splade_doc_embs.pkl, bm25.pkl, results/}` | legacy NER / SPLADE / BM25 stores + L1/L2/L3 result dirs | same |
| `data/processed/master_nodes_<legacy>.json` (+ `_std_*`, `_hpr_*`, `webqsp_small/tiny` variants) | node order + neighbours the L1 code aligns to (`src.pipeline.standardizer.load_nodes` doc order; `_l1ep_c.dir_out` builds the out-CSR from it) | legacy |
| `scratchpad/ablation_qwen/<legacy>/variant_{A,B,C}/{partition_map.json, graph.pt}` | the frozen topology-C P50 partition maps (401 / 658 / 136 / 5,074 / 190 / 7,814 parts) | legacy |
| `scratchpad/_l1hu/graphs/`, `scratchpad/_l1ep/parts/<legacy>__<TAG>.npy`, `scratchpad/_l1kn/` | H4 hypergraphs, Mt-KaHyPar hard assignments (H4FAM_S/SK/SN/SKN, METIS, FENNEL, …) | legacy |
| `results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_<legacy>.npz` | the frozen-L1 replay caches (2,000-query val samples; webqsp 1,419 val+train) | legacy |
| `results/L1/L1_LOCKED_MANIFEST.json` (`L1_FROZEN: true, L1_RECOMPUTE_REQUIRED: false`) | pins the legacy paths above by sha | legacy → ⚠ stale vs the canonical package (also KNOWN_LIMITATIONS[7]: pre-2026-09-11 numbers used corrupted dense rows) |
| `data/l2_corpus/{2wiki_clean,musique_clean,squad_clean}/` | L2 corpus from the legacy C/P50 assignment | legacy |
| `data/_family_v1/` (`metaqa/`, `webqsp/` NER work + kNN audits), `data/original/`, `data/raw/`, `crag_data_backup/` | build workspaces / official sources / backups | n/a |

`legacy_comparison.json` (five datasets): every legacy node maps into the canonical corpus (`id_mapping_coverage` 1.0);
canonical is a strict superset (extra nodes: metaqa 3,083; 2wiki 5,925,732; musique 103,862; hotpotqa 4,725,835; squad 1,204).
There is **no** legacy comparison for webqsp (see §7). No script of the L1 chain references `data/final_canonical/`
(the 56 non-gate `.py` files that mention it are builders and audit tools under `scratchpad/`); 182 `.py` files under
`src/` + `scratchpad/` reference `data/ukb_storage`, including `_l1ps_cache.py`, `_l1ep_sub.py`, `_l1kn_sub.py`, `l1_eval_phase1.py`.

---

## §2 Per-dataset inventory (independent re-checks of this window; ✅ = verified now)

Common to all six: `DATASET.json` self-hash OK; `nodes.jsonl` line count == n_nodes; **0 duplicate node ids, 0 empty
texts, 0 rows without a text field**; every split's row count == recorded; **0 duplicate query ids**;
`query_ids.json` == concatenation of the splits in `splits_in_row_order`; every `gold_node_ids` entry and every topic
reference resolves to a node id (0 unresolved on every labelled split; blind tests are GOLD_ABSENT by design);
**every embedding channel's row count re-read from the shard headers == corpus/query rows** (dense fp16 dim 1536;
splade CSR float32); every graph family: edge count == manifest, **all endpoints in range, 0 nodes with degree 0 across
the three families**, 0 NaN weights; both caches `[n_queries, 1000]` int32 ids + float16 scores, ids in range, 1,000
distinct ids per row and non-increasing scores on a 2,000-row sample.

| | A queries (train/eval/test) | B nodes | C graph (structural / ner / knn; deg-0 per family) | D relations | E doc emb (rows) | F query emb (rows) | G caches | H bridge | I manifest |
|---|---|---|---|---|---|---|---|---|---|
| **metaqa** | 329,282 / dev 39,138 / test 39,093 (all labelled; gold refs 2,446,161 / 300,980 / 302,779 resolve) | 43,234 | 133,582 (11 self loops, 8,902 pairs with >1 relation, 0 dup triples; deg-0 0) / 5,949 (deg-0 37,552) / 97,920 (deg-0 0) | 9 relations, vocab 9, ids < vocab | dense 43,234 · splade 43,234 | dense 407,513 · splade 407,513 | dense+splade [407,513×1000] | `node_id_map_legacy.json` 40,151→40,151 (1:1, 0 ambiguous, all resolve); eval_1998 gold differs on 46 rows (recorded in `legacy_comparison.json`) | ✅ frozen |
| **2wiki** | 167,454 / dev 12,576 / test 12,576 (**blind**) | 5,989,847 | 28,963,600 hyperlink (0 loops, 0 dup, 754,743 mutual pairs; deg-0 515,830) / 34,067,241 (deg-0 2,114,594) / 13,643,063 (deg-0 0) | 1 relation (hyperlink) | 5,989,847 · 5,989,847 (150 shards) | 192,606 · 192,606 | [192,606×1000] + meta.json | 65,865 legacy → 64,115 distinct canonical (1,750 legacy nodes collapse onto shared canonical nodes) + 2 ambiguous; coverage 1.0 | ✅ frozen |
| **musique** | 19,938 / dev 2,417 / test 2,459 (**blind**) | 117,534 | 2,744,076 title_mention (deg-0 9,858) / 800,278 (deg-0 20,677) / 265,366 (deg-0 0) | 1 relation | 117,534 · 117,534 | 24,814 · 24,814 | [24,814×1000] | 13,672→13,672 (1:1, all resolve) | ✅ frozen |
| **hotpotqa** | 90,447 / validation 7,405 / test 7,405 (**blind**) | 5,233,329 | 15,367,541 hyperlink (deg-0 551,848) / 20,369,074 (deg-0 2,136,890) / 11,946,289 (deg-0 0) | 1 relation | 5,233,329 · 5,233,329 (131 shards) | 105,257 · 105,257 | [105,257×1000] + meta.json | 507,494→507,494 (1:1) | ✅ frozen (builder revision a46d745a… LOST, recorded) |
| **squad** | 130,319 / dev 11,873 (no test) | 20,233 | 874,190 title_mention (deg-0 6,349) / 156,130 (deg-0 4,391) / 42,969 (deg-0 0) | 1 relation | 20,233 · 20,233 | 142,192 · 142,192 | [142,192×1000] | 19,029→19,029 (1:1, all resolve) | ✅ frozen |
| **webqsp** | 3,098 / eval = train_holdout (stride 2 → 1,549) / test 1,639 (labelled, barred); zero-gold rows 88 train + 55 test (= 37 unanswerable + 106 gold-absent) | 2,592,894 (id == position on every row) | 8,309,195 kb_relation (93,692 self loops, 333,124 pairs with >1 relation, 0 dup triples, 4,351,530 reverse rows; deg-0 0) / 2,994,802 (deg-0 1,797,723) / 6,470,520 (deg-0 0) | **7,058 qualified relations, 1:1 with rel ids** (see §7) | 2,592,894 · 2,592,894 (65 shards) | 4,737 · 4,737 | [4,737×1000] | no legacy bridge (N/A — §7); canonical: positions + `webqsp:n<pos>` + `source_mid` | ✅ frozen; governance records ⚠ stale wording (§7) |

Suspected gaps refuted: **MetaQA and 2Wiki query embeddings exist** in both channels (407,513 and 192,606 rows, read
from the shard headers), as do all 24 channels and all 12 caches.

---

## §3 Endpoint / alignment checks (low-memory verifiers, all six)

| check | result |
|---|---|
| missing_src / missing_dst (endpoint ∉ [0, N)) | **0** in all 18 families |
| duplicate node ids | **0** (6/6); duplicate query ids **0** (6/6) |
| duplicate rows | 0 duplicate (src,dst,rel) triples in every structural family; 0 duplicate rows in every ner/knn family; the (src,dst) repeats in metaqa (8,902) and webqsp (333,124) are multi-relation pairs |
| orphan nodes (no edge in any family) | **0** (6/6) — kNN k=3 covers every position; musique position 74545 has 3 kNN edges only (KNOWN_LIMITATIONS[3]) |
| zero-degree per family | listed in §2-C; NER is sparse by construction on metaqa (37,552 of 43,234 without an NER edge) and webqsp (1,797,723 of 2,592,894) |
| gold / topic resolution | 0 unresolved on every labelled split; webqsp `gold_positions` == `gold_node_ids` on every row |
| cache alignment | rows == n_queries, ids == positions, K = 1000 (shallower K is a prefix) |
| webqsp components (structural, this audit) | 23 weakly connected components; largest 2,592,826 (99.9974 %); others of size 2–12; 0 isolates; 1,084,103 strongly connected components |

---

## §4 Primary status (exactly one) and §5 readiness levels

| dataset | PRIMARY STATUS | L1 canonical | L2 representation | L3 L1-ready (inputs) |
|---|---|---|---|---|
| metaqa | READY_FOR_L1 | ✅ | ✅ | ✅ inputs; L1 stage artefacts not built (§6) |
| 2wiki | READY_FOR_L1 | ✅ | ✅ | ✅ inputs; partitioning at 5.99M nodes not feasible here (§6) |
| musique | READY_FOR_L1 | ✅ | ✅ | ✅ inputs |
| hotpotqa | READY_FOR_L1 | ✅ | ✅ | ✅ inputs; partitioning at 5.23M nodes not feasible here (§6) |
| squad | READY_FOR_L1 | ✅ | ✅ | ✅ inputs |
| webqsp | READY_FOR_L1 (as the WEBQSP_ROG_STANDARD lane; identity/independence items are rulings, not artefacts — §7) | ✅ | ✅ | ✅ inputs; partitioning at 2.59M nodes not feasible here (§6) |

No dataset is MISSING_* / ALIGNMENT_BLOCKED / MANIFEST_ONLY_BLOCKED: the package already passed
LOCKED_5_OF_5_BENCHMARK_SUBSTRATES (2026-09-08T16:57Z) → LOCKED_6_OF_6_BENCHMARK_SUBSTRATES (2026-09-08T18:28Z, record
`015b71b9…`, now in `_history/records/`) → FAMILY_COMPLETION V1/V2 → **CANONICAL_FREEZE (current)**. What is not ready
is the L1 side, not the datasets.

---

## §6 L1 requirements audit — what the frozen L1 ACTUALLY consumes

Current canonical L1 = the frozen contract replayed by the hypergraph core + halo line
(`scratchpad/_l1ps_cache.py` → `_ta_prepartition`; `_l1ep_*`, `_l1kn_*`, `_l1hu_*`; frozen selector
`runs/final_B6_S4_F6_Ms64_Mr32.json`; `l1_eval_phase1.py` for the C/P50 evaluator). CONTRACT read from
`runs/cache_<ds>.npz` meta: **K0 60, K 100 (TOPP 200 cached), P_MAIN 50, SEED_K 5, BEAM 64, MAX_HOPS 3, DEG_CAP 300, SMAX 256**.

| L1 needs | read from (today) | canonical equivalent | status |
|---|---|---|---|
| query embeddings | `gte_qwen/queries_all.npy` (fp32; normalised at load) | `embeddings/dense/queries` (fp16, unit norm) | ✅ exists |
| doc embeddings (directional expansion `s_dir`) | `gte_qwen/nodes.npy` (fp32/fp16 path) | `embeddings/dense/docs` | ✅ exists |
| Dense top-K (K=100 of 200) | `dense_top200_all.npy` | `retrieval_cache/dense_top1000.npz[:, :200]` (prefix rule) | ✅ exists |
| SPLADE top-K | `splade_top200_all.npy` | `retrieval_cache/splade_top1000.npz[:, :200]` | ✅ exists |
| STRUCT / KNN / NER edge families (H4 hypergraph build, FULL_C halo, struct traversal) | `gte_qwen/graph.pt` (A = STRUCT ⊔ KNN), `ner_edges_w_df25.pkl`, `variant_C/graph.pt` | `graph/{structural,ner,knn}.npz` (positions) | ✅ exists (not the same graphs: canonical corpora are supersets; NER weights are the same 1/df family) |
| node order + directed out-CSR (`_l1ep_c.dir_out` over `master_nodes_<legacy>.json`) | `data/processed/master_nodes_*.json` | `nodes.jsonl` order + `graph/structural.npz` | ✅ exists |
| gold, hops, eval population | `query_ids_all.json` (val sample 2,000; metaqa hop-balanced; webqsp val+train, TEST never touched) | `queries/<EVAL_SPLIT>.jsonl` (`gold_node_ids`, `hop`), EVAL_SPLITS: metaqa dev, 2wiki dev, musique dev, hotpotqa validation, squad dev, webqsp train_holdout (stride 2, carve sha `a818f461…`) | ✅ exists |
| partition assignment (`hard`, k ≈ N/100) — H4_SK → Mt-KaHyPar → P50 cores | `scratchpad/_l1ep/parts/<legacy>__*.npy`, `variant_C/partition_map.json` | **none over canonical positions** | ❌ missing (L1-stage artefact) |
| replay cache (`cache_<ds>.npz`) | `results/…/runs/` (legacy) | none | ❌ missing (derived, cheap once the above exist) |
| relation metadata / labels | not read | — | N/A for L1 |
| corpus hashes, node text, lanes, top-1000 depth, ANN indexes | not read | — | N/A for L1 |

Two honest constraints: (1) **no L1 script reads the canonical layout** — every path is hard-coded to
`data/ukb_storage/{ds}/gte_qwen/`, `data/processed/master_nodes_{ds}.json`, `scratchpad/ablation_qwen/{ds}/variant_C/`;
(2) the frozen partitioner is in-memory Mt-KaHyPar (`DETERMINISTIC_QUALITY`, KM1, k ≈ N/100, ε 0.03, seed 0): the legacy
webqsp (781k) needed 125 GB RSS and hotpotqa (507k) 245.7 GB on Modal; canonical webqsp / hotpotqa / 2wiki are 2.59M /
5.23M / 5.99M nodes (k ≈ 25,929 / 52,333 / 59,898 partitions) — Modal is dead and this machine has 16 GB.
Do not build graph families for symmetry: L1 needs S/K/N only, and all three exist on all six.

---

## §7 WebQSP

| item | tag | fact |
|---|---|---|
| canonical queries | EXISTS_AND_VERIFIED | 4,737 (train 3,098 / test 1,639), 0 duplicate strings/ids, gold resolves on every row; eval = train_holdout (1,549: 1,503 with gold, 46 zero-gold); query-level ceiling 0.9703 (0.9792 of answerable), reference-level 0.5255 |
| node table | EXISTS_AND_VERIFIED | `nodes.jsonl` 2,592,894 rows, `node_id == webqsp:n<position>` on every row, 0 dup, 0 empty; `v1/nodes.parquet` (14 cols: node_uid, node_key, source_rog_endpoint, resolution_status, node_kind, …, source_mid, display_name(_disambiguated), display_source) pinned; kinds CVT_MEDIATOR 1,588,085 / READABLE_ENTITY 925,761 / MID_NAMED_ENTITY 62,375 / VALUE_LITERAL 14,515 / UNRESOLVED_OTHER 2,158 |
| NAME_ONLY text | EXISTS_AND_VERIFIED | `text_name_only`; display_source STRUCTURAL_SCHEMA_LABEL 1,652,618 / ROG_SURFACE 925,761 / LITERAL_VERBATIM 14,515; 1,791,533 distinct encoder texts; 801,361 positions share text (identical vectors); CANONICAL_V1_VERIFICATION 2026-09-08 N1–N4, E1–E2, M1–M3, G, Q1–Q3 all PASS |
| typed edges | EXISTS_AND_VERIFIED | 8,309,195 `(src,dst,rel int16)`, 0 out-of-range, 0 duplicate triples, 93,692 self loops, 333,124 multi-relation pairs |
| relation vocabulary | EXISTS_AND_VERIFIED | `v1/relations.parquet`: relation_uid, **relation_key (7,058 distinct = identity)**, relation_label_short (4,657 distinct → 34 % collide), relation_label_qualified (7,058 distinct, TYPE_PROPERTY); the served `relation_vocabulary` is the 7,058 qualified labels, 1:1 with rel ids → **identity is qualified and unambiguous** |
| direction | EXISTS_AND_VERIFIED | directed as found in RoG (subject→object), never symmetrised; 4,351,530 rows have a stored reverse; manifest `directed: true` |
| node / MID identity | BLOCKED_BY_RULING | served identity = RoG endpoint (`node_key`); `source_mid` exists for the 1,652,618 bare-MID endpoints, absent for the 925,761 surface-only endpoints whose MIDs RoG discarded upstream (not recoverable from the RoG files); lane WEBQSP_ROG_MID "STILL UNRESOLVED"; NSM union = sibling extraction, NOT RESOLVED (v1 corpus unchanged) |
| RoG exactness | EXISTS_AND_VERIFIED | ROG_UNION_REBUILD: triples 8,309,195 and relations 7,058 EXACT; endpoints 2,592,894 vs paper 2,566,291 (+26,603, REPORTED NOT FORCED; GRAPH_ACCEPTED by user 2026-09-05) |
| serialization issue (5 backslash endpoints) | EXISTS_AND_VERIFIED | node table is JSON (lossless); N2/N3 pass on all rows; the gate's `SERIALIZATION_BLOCKER` text was never updated (⚠) |
| CORPUS_HASH | EXISTS_AND_VERIFIED | `nodes.jsonl` sha256 `b39c615122ad55f5…` — pinned in DATASET.json and recomputed now (match) |
| NODE_ORDER_HASH | EXISTS_AND_VERIFIED (implied) | order = line = position, ids embed the position; no separate hash record (the gate field says PENDING — ⚠ stale) |
| connected components / largest / isolates / self loops | EXISTS_AND_VERIFIED (computed now) | 23 weak components, largest 2,592,826 (99.9974 %), 0 isolates, 93,692 self loops; 1,084,103 SCCs |
| query independence | BLOCKED_BY_RULING | membership is question-derived upstream (RoG per-question subgraphs, all 4,700 WebQSP questions + CWQ) — accepted as the standard benchmark KG lane, never "full Freebase"; no `query_independence_test.json` (the five have one, all CORPUS_QUERY_INDEPENDENT true); a query-independent universe now exists as `freebase/` (302M nodes; bridge resolves 1,618,950 of the 1,652,618 MID nodes, −1 for the 973,944 others) but has no embeddings (INFEASIBLE_HERE) |
| node embeddings | EXISTS_AND_VERIFIED | dense 2,592,894 rows (65 fp16 shards), splade 2,592,894 rows; attention-masked pooling (differs from the five) |
| query embeddings | EXISTS_AND_VERIFIED | 4,737 rows, both channels |
| Dense retrieval | EXISTS_AND_VERIFIED | `dense_top1000.npz` [4,737×1000], ids in range, 1,000 distinct/row, monotone scores; per-split pool ceilings recorded |
| SPLADE retrieval | EXISTS_AND_VERIFIED | same shape/checks |
| alignment | EXISTS_AND_VERIFIED | id == position; gold_positions == gold_node_ids; topic positions in range; legacy bridge N/A by design (legacy 781,485-doc substrate was TEST-derived with 458,872 fabricated names — LEGACY_FABRICATION_BUG_VERIFIED) |
| manifest / status | STALE (wording) | `_WEBQSP_ACCEPTANCE_GATE.json` (`_last_updated` 2026-09-06: 7 checklist fields PENDING, NODE_ID "CONFLICT — USER DECISION REQUIRED", next_phase "NOT STARTED"), `webqsp/SOURCE_CONTRACT.json` ("BLOCKED_PENDING_FREEBASE_SOURCE — NOTHING BUILT AND NOTHING MAY BE BUILT"; `QUERY_DEPENDENT_CORPUS: false` beside prose calling the source query-derived), `_APPROVALS.json` (READY_6_OF_6_SOURCE_CONTRACTS, "_do_not_mark_locked_until"). All three predate the 2026-09-08 build, are pinned by the freeze as governance history, and must not be edited; the served status is the freeze |

---

## §8 Freebase — confirm only (nothing reopened)

| record | path | status |
|---|---|---|
| canonical graph (served) | `freebase/DATASET.json` `d3b9324b1c287f10…` (file `f3b40c16…`, self-hash OK): 301,977,131 nodes, 2,062,430,072 edges (out+in CSR on positions), 784,928 relations, flipped 0 | EXISTS_AND_VERIFIED (163 freebase checks in VERIFICATION, 0 failed) |
| served names | census ORIGINAL 157,420,227 / RECOVERED_ORIGINAL 287,703 / RECOVERED_EXTERNAL 246,829 / URI_SELF 74,454,117 / KEY_SEMANTIC 324,820 / INFERRED 62,022,081 / GENERATED_FLOOR 7,221,354; actual 157,954,759; empty 0 | EXISTS_AND_VERIFIED |
| actual / inferred hierarchy | `name_kind` column + `FREEBASE.names` in the freeze; KNOWN_LIMITATIONS[11] | EXISTS_AND_VERIFIED |
| closure status | `freebase_v3/FREEBASE_V3_CLOSURE_STATUS_V2.json` `ab8ae4d6fc0a008e…` (file `3e653477…`, self-hash OK, TERMINAL by DISPOSITION), builds on V1 `bff1453f…` (file `b1da1f0f…`, byte-unchanged) | EXISTS_AND_VERIFIED |
| canonical freeze | `CANONICAL_FREEZE.json` `58958f33…` FREEBASE.source_layer pins V2 + deletion log + R2 census (all sha-match) | EXISTS_AND_VERIFIED |
| deletion log | `_history/logs/FREEBASE_V3_SUBLAYER_DELETION.json` `a28a2791…` (file `866daff9…`): APPLIED, 810 files / 20.73 GB, 328 pinned, 0 mismatch | EXISTS_AND_VERIFIED |
| source of record | raw mirror `freebase_v3/_acquisition/raw/freebase-rdf-latest.gz` 31,305,093,084 B present; `_acquisition/idir/idirlab-freebases.zip` + oracle/ + metadata/ present; `_acquisition/facc1` present; `freebase_v3/canonical/CANONICAL_MANIFEST.json` present (37.76 GB layer, user's call); `inference_admissibility_v1/` 32 files | PRESERVED |
| not included | embeddings / queries / retrieval cache: none (recorded, INFEASIBLE_HERE) | as recorded |

---

## §9 Master matrix

| Dataset | Canonical queries | Corpus/nodes | Graph | Node/doc emb | Query emb | Dense/SPLADE retrieval | ID alignment | Manifest | L1 ready |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 inputs ✅ / L1 stage ❌ |
| 2Wiki | ✅ (test blind, N/A) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 inputs ✅ / partition ❌ (external) |
| MuSiQue | ✅ (test blind) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 inputs ✅ / L1 stage ❌ |
| HotpotQA | ✅ (test blind) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 inputs ✅ / partition ❌ (external) |
| SQuAD | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 inputs ✅ / L1 stage ❌ |
| WebQSP | ✅ (test barred) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ (legacy N/A) | ✅ / ⚠ stale governance wording | 🟡 inputs ✅ / partition ❌ (external) + rulings |

**A. EXISTING AND VERIFIED** — all six trees end to end (nodes, splits, lanes on the five, 24 embedding channels,
18 graph families, 12 caches, DATASET.json, GRAPH_MANIFEST.json, loader, freeze, full verification 686/0, claims check);
legacy→canonical bridges for the five; webqsp v1 tables, RoG exactness, relation identity; Freebase served tree +
terminal records + protected sources.

**B. EXISTS BUT NEEDS VERIFICATION** — (1) cache *exactness* vs the served embeddings was verified at build
(builder's tie-break key, 202 ok / 0 bad) and by the freeze's digests, but not re-derived in this audit;
(2) the webqsp NER/kNN families (built 2026-09-09 over NAME_ONLY texts) are verified structurally, not against a
re-run; (3) 2wiki bridge: 1,750 legacy nodes collapse many-to-one onto canonical nodes (paired analyses only).

**C. ACTUALLY MISSING** (all L1-stage, none dataset-level):
1. an L1 reader for the canonical layout (positions, `canonical.py`, top-1000 prefixes, EVAL_SPLITS) — every L1 script hard-codes the legacy paths;
2. H4_SK hypergraph + P50 hard partition assignments over canonical positions, 6/6 (k ≈ N/100);
3. frozen-contract replay caches (`cache_<ds>.npz` equivalents) over canonical positions and EVAL_SPLITS populations;
4. webqsp `query_independence_test.json` / legacy comparison — N/A by contract (not buildable, not required by L1).

**D. STALE / CONFLICTING** — `results/L1/L1_LOCKED_MANIFEST.json` and everything under `data/ukb_storage`,
`data/processed/master_nodes_*`, `scratchpad/ablation_qwen`, `scratchpad/_l1hu|_l1ep|_l1kn`,
`results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs`, `data/l2_corpus` (legacy universes, pre-2026-09-11 dense rows);
webqsp governance wording in `_WEBQSP_ACCEPTANCE_GATE.json`, `webqsp/SOURCE_CONTRACT.json`, `_APPROVALS.json`
(pre-build statuses pinned as history — supersede by a new record only if a ruling wants it; never edit);
`CANONICAL_V1_VERIFICATION.json` says ner/knn "not built" (true on 2026-09-08; both were added 2026-09-09 and are served).

**E. MINIMUM WORK TO RETURN TO L1** (dependency order; LOCKED_5_OF_5 and LOCKED_6_OF_6 are already behind us —
the freeze is the lock; no dataset item is on the path):
1. VERIFY_ONLY — none pending (686/0 stands; this audit found no drift).
2. Canonical-layout L1 adapter (read positions/caches/graphs/EVAL_SPLITS through `canonical.py`; top-200 = prefix of top-1000; eval population = 2,000-query sample of the EVAL_SPLIT, seed-pinned; TEST never read).
3. H4_SK hypergraph build + Mt-KaHyPar P50 partitions over canonical positions: metaqa, squad, musique first (local), then webqsp, hotpotqa, 2wiki (need a machine with ≥ 250 GB RAM or a ruling to change the partitioner — Modal is dead).
4. Replay caches + frozen L1 gate (BASE / SAFE) on the canonical substrate → RETURN_TO_L1.
5. Rulings, not builds: webqsp lane wording (ROG_STANDARD vs ROG_MID), superseding governance note, disk (9.9 GB free), `freebase_v3/canonical/` 37.8 GB.

**F. WORK CLASS** — item 2 CHEAP_CPU (code); item 3 metaqa/squad/musique HEAVY_CPU (local), webqsp/hotpotqa/2wiki
EXTERNAL/RULING_REQUIRED; item 4 CHEAP_CPU–HEAVY_CPU (traversal caches on 5–6M-node graphs); item 5 RULING_REQUIRED;
B(1) cache re-derivation GPU_SMALL (H100 minutes) or HEAVY_CPU; freebase embeddings GPU_LARGE (out of scope).

STOP_FOR_REVIEW
