# PHG_SCALE_WEBQSP -- webqsp (WEBQSP_ROG_RESOLVED / BENCHMARK_CONDITIONED / query_independent NOT CLAIMED / MID completeness NOT REQUIRED FOR V1; NAME_ONLY) under the experimental PHG contract (H4_SK_ZOLTAN_PHG_CONNECTIVITY_NP4 + PHG_NONEMPTY_REPAIR_V1)

Generated 2026-09-14T20:44:56Z -- STATUS: **STOP_FOR_REVIEW** -- DECISION: **WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED**

Question (preregistered): Can the frozen PHG contract produce a valid partition of the exact canonical WebQSP H4_SK and carry it through unchanged canonical L1 with sane retrieval behaviour at a memory footprint that makes the scale lane practical?

Reference policy: no canonical Mt-KaHyPar webqsp reference exists (the frozen recipe is NOT_FEASIBLE_AT_250GB: None GB linear in pins / None GB pins x k), so this is a scale / absolute-L1 result -- no equivalence claim, no comparative quality claim.  The legacy webqsp cells (NAME + verbalised facts corpus) are historical context only and are not compared numerically.

The host never met the preregistered clean-host criterion within the wait budget (775 samples, 0 clean, host available [0.0, 4.21] GB, pages in/s [816.4, 138831.9]); nothing was launched and nothing was killed.  This is a host-resource outcome, not evidence about PHG.

Preflight immediately before each launch (criterion: host available >= 6.0 GB, Pages Input/sec <= 300, WSL available >= 5000 MB, disk >= 8.0 GB, swap not growing; 3 consecutive samples):

- history_summary: 2026-09-14T20:44:43Z CONTENDED  host avail 1.16/15.69 GB  pages_in/s 37056.8  swap 9.57 GB (+0.27)  wsl avail 7174/7789 MB  disk 17.3 GB  foreign pid gone rss None MB  ranks 4
- log: {"path": "results/L1_LOWMEM/logs/phg_webqsp_preflight.jsonl", "bytes": 851171, "sha256": "07fc17d1cfe5bc31078643118a310ab58bf0e9f07c4f05f21256088a36402e96"}

Stages:

| Stage | Completed | Wall s | Memory | Note |
|---|---|---|---|---|

Partition (diagnostics; no canonical Mt-KaHyPar value exists to compare the KM1 with):

| Arm | KM1 (weighted) | cut nets | STRUCT cut | KNN cut | blocks min/p50/p95/max (bound) | empty | peak RSS | wall | status |
|---|---|---|---|---|---|---|---|---|---|

Canonical L1 on the frozen train_holdout population (absolute; canonical NAME_ONLY encoding; test never read):

| Arm | nq | BASE_ALL | SAFE_ALL | BASE_ANY | SAFE_ANY | scope BASE / SAFE | SAFE additions (+/-) | BND_ALL | balance max/mean | ELIGIBLE |
|---|---|---|---|---|---|---|---|---|---|---|

Historical context (NOT a baseline; different corpus encoding, N, k, family set and population; never compared numerically): legacy webqsp LEGACY_EXPLORATORY, 781485 docs, npart 7814, ALL_EVALUABLE x 1578 queries, operating point C_dense+splade_K100_P50 = {"ANY_GOLD_COV": 94.3, "ALL_GOLD_COV": 76.24, "GOLD_FRAC": 87.36, "mean_scope": 5023.3, "reduction": 155.57, "PRIMARY": "ANY", "PRIMARY_VALUE": 94.3}; legacy Mt-KaHyPar SKN resource point 10,900,573 pins, k 7,814, peak RSS 140.6 GB (EXTERNAL_LANE_MEMORY_EXPECTATION.json legacy_points).

Host: before null; after {"utc": "2026-09-14T20:44:54Z", "purpose": "not_run", "host_total_gb": 15.69, "host_available_gb": 0.59, "host_percent_used": 96.2, "swap_used_gb": 9.6, "swap_total_gb": 18.71, "pages_input_per_sec_samples": [54619.388878, 14047.012111, 46872.889184, 17994.588751, 16025.952727], "pages_input_per_sec_mean": 29912.0, "typeperf_available_mbytes": [957.0, 912.0, 929.0, 828.0, 894.0], "typeperf_error": null, "wsl": {"wsl_total_mb": 7789, "wsl_used_mb": 446, "wsl_free_mb": 6874, "wsl_buff_cache_mb": 468, "wsl_available_mb": 7112, "wsl_swap_total_mb": 2048, "wsl_swap_used_mb": 0, "wsl_loadavg_1_5_15": [1.06, 0.87, 0.89]}, "disk_free_gb": 17.31, "foreign_processes": [{"pid": 6620, "name": "python.exe", "rss_mb": 7144, "private_mb": 14207, "started": "2026-09-14T14:54:56"}], "foreign_pid": 8628, "foreign_pid_present": false, "foreign_pid_rss_mb": null, "mpi_ranks": 4, "swap_growth_gb_since_previous": null, "checks": {"host_available_ge_min": false, "not_paging": false, "wsl_available_ge_min": true, "disk_free_ge_min": true, "swap_not_growing": true}, "CLEAN": false}.

Integrity: frozen records changed = none; input/data pins changed = none; musique files changed = none; canonical webqsp partition still absent = True; canonical webqsp replay still absent = True; experimental files written = [].

Caveats: 4 local MPI ranks inside one WSL VM validate feasibility on this host only (no aggregate-RAM claim at larger scale); the rank count is a contract element and was not changed; PHG_NONEMPTY_REPAIR_V1 is part of the contract (no-op when no block is empty; no minimum-part-size condition exists); the preflight is execution hygiene, not a PHG parameter.

HotpotQA / 2Wiki / the 302M Freebase lane not run (unauthorized); no promotion; no tuning; NP unchanged; MuSiQue not rerun; the canonical webqsp partition slot stays empty; FREIGHT stays FREIGHT_CLOSED; the squad / metaqa / musique PHG records untouched; the foreign process untouched
