# `_pre_v3/` — tools of the earlier layouts (historical)

Every script here wrote or checked a record of the pointer-indexed package
(`LOCKED_5_OF_5` → `LOCKED_6_OF_6` → `FAMILY_COMPLETION_V2` → `CONSOLIDATION_V3`). Those
records are kept unchanged under `data/final_canonical/_history/records/` and their data
copies were deleted on 2026-09-12 (see `data/final_canonical/CANONICAL_FREEZE.json`,
`LINEAGE`). The scripts are kept so the records can be read against the code that wrote
them; they are **not** runnable against the tree as it is now (they look for
`data/canonical/`, `POINTER_INDEX.json`, `graph2/` and root-level records that no longer
exist), and they must not be "fixed" to run: a record is superseded by a new record, never
regenerated.

They were moved here from `src/dataset_canonical/` on 2026-09-12 without edits; relative
imports between them (`from verify_manifest import record_hash`, …) still resolve because
each one puts its own directory on `sys.path`.

| script | wrote / checked |
|---|---|
| `ukb_manifest.py`, `verify_manifest.py` | `UKB_COMMON_MANIFEST.json` (+ `_VERIFICATION`) — the V1 common manifest and its verifier; `record_hash` convention |
| `gold_field_check.py` | `GOLD_FIELD_CHECK.json` |
| `k_semantics_and_splits.py` | `K_SEMANTICS_AND_EVAL_SPLITS.json` (reads `webqsp/_acquisition/nsm/extracted/webqsp/webqsp/*_simple.json`) |
| `retrieval_cache_record_update.py`, `verify_heavy_caches.py`, `manifest_cache_slots.py`, `cache_and_repair_check.py`, `doc_cache_status_patch.py`, `repair_coverage_check.py` | `RETRIEVAL_CACHE*.json`, `HEAVY_CACHE_VERIFICATION.json`, `MANIFEST_CACHE_SLOT_FILL.json`, `CACHE_AND_REPAIR_CHECK.json`, `DOC_CACHE_STATUS_PATCH.json`, `REPAIR_COVERAGE_CHECK.json` |
| `musique_knn_graph2.py`, `freeze_family_completion_v2.py`, `graph_degree_hub.py` | the `graph2/` families and `LOCKED_6_OF_6_FAMILY_COMPLETION_V{1,2}.json`, `GRAPH_DEGREE_HUB.json` |
| `frozen_caveat_supersession.py`, `superseded_statements_index.py` | `FROZEN_CAVEAT_SUPERSESSION.json`, `SUPERSEDED_STATEMENTS_INDEX.json` |
| `webqsp_pointer_amendment.py`, `webqsp_v3_*.py`, `webqsp_antijoin_test.py`, `webqsp_name_join_control.py` | `WEBQSP_POINTER_AMENDMENT.json` and the WebQSP name-join controls |
| `consolidate_v3.py`, `consolidate_v3_layout.py` | the 2026-09-11 in-place heal and fold (`CONSOLIDATION_V3_LOG.json`) |
| `cleanup_plan.py`, `cleanup_reference_crosscheck.py`, `cleanup_pass_3.py`, `canonical_dedup.py`, `canonical_names.py`, `postcleanup_check.py`, `package_bytes_ledger*.py`, `stack_census.py`, `substrate_map.py`, `handoff_readiness.py` | the cleanup classifier passes (`CANONICAL_CLEANUP_PLAN`, `CLEANUP_REFERENCE_CROSSCHECK`, `CLEANUP_PASS_3`, …) and package inventories |
| `build_splade_parts_local.py` | the local SPLADE cache shards of 2wiki (see `2wiki/retrieval_cache/splade_top1000.meta.json`) |

`freebase_v3_closure_status.py` stayed in the parent directory: it belongs to the Freebase
layer, which is not part of the freeze.
