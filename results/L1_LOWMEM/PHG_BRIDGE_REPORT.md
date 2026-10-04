# PHG_SCALE_BRIDGE -- musique under the experimental PHG contract (H4_SK_ZOLTAN_PHG_CONNECTIVITY_NP4 + PHG_NONEMPTY_REPAIR_V1)

Generated 2026-09-13T19:16:20Z -- STATUS: **STOP_FOR_REVIEW** -- DECISION: **PHG_SCALE_BRIDGE_COMPLETE**

Question (preregistered): does exact H4_SK -> PHG -> NONEMPTY_REPAIR_V1 if needed -> P50 -> canonical L1 complete locally with sane partition / L1 behaviour?

No Mt-KaHyPar reference exists for this corpus (RESOURCE_INFEASIBLE_LOCAL -- FAILED_MEMORY_CAP (threads 4, cap 6.0 GB as recorded by partition.py, RSS at kill 5862 MB, wall 1132 s)), so this is a feasibility record: absolute L1 numbers, no equivalence claim, no comparative quality claim against Mt-KaHyPar.

Stages:

| Stage | Completed | Wall s | Memory | Note |
|---|---|---|---|---|
| 1 H4_SK input verified | True | 5.4 | py peak 317.3 MB | N 117534 M 250732 P 6014938 k 1175; digest recomputed == ORIGINAL |
| 2 structure gate (gate-mode PHG run) | PASS | 53.7 | peak RSS/rank max 145 MB | PASS; weight reception rel diff 2.1e-08; removal/warning lines none |
| 3 PHG partition R1 (+ repeat) | True | 167.6 (partition 34.82, job 76.87) | peak RSS/rank [177.9, 163.1, 165.8, 164.1] MB (max 178, sum 671) | raw validity PARTITION_INVALID: max 104 (bound 104), empty 9; repeat identical True; Python KM1 vs Zoltan cutl rel diff 4e-08 |
| 3b PHG_NONEMPTY_REPAIR_V1 | PASS | 43.7 | py peak 591.9 MB | empty blocks [330, 358, 365, 366, 367, 1164, 1165, 1171, 1173] -> 9 move(s) node 663: 866 -> 330 (delta +0, incident nets 68, ties 396); node 692: 1041 -> 358 (delta +0, incident nets 12, ties 395); node 800: 684 -> 365 (delta +0, incident nets 22, ties 394); node 823: 2 -> 366 (delta +0, incident nets 10, ties 393); node 1205: 189 -> 367 (delta +0, incident nets 4, ties 392); node 1225: 411 -> 1164 (delta +0, incident nets 5, ties 391); node 1391: 68 -> 1165 (delta +0, incident nets 9, ties 390); node 1713: 412 -> 1171 (delta +0, incident nets 4, ties 389); node 2051: 593 -> 1173 (delta +0, incident nets 12, ties 388); KM1 85311179 -> 85311179 (+0); STRUCT cut delta 0.0; KNN cut delta 0.0; postconditions PASS; validity after PASS |
| 4/5 import + P50 replay cache + canonical L1 | BASELINE_ABSENT | 468.2 (eval 4.0) | builder predicted peak 0.75 GB; py peak 1498.8 MB | cache 6183023 bytes; BASE 22.5 s / STRUCT 415.1 s; status BASELINE_ABSENT |

Partition (diagnostics; the KM1 has no Mt-KaHyPar value to be compared with):

| Arm | KM1 | STRUCT cut | KNN cut | blocks min/p50/p95/max (bound) | empty | peak RSS | wall | status |
|---|---|---|---|---|---|---|---|---|
| PHG C1 (R1 + PHG_NONEMPTY_REPAIR_V1, NP=4) | 85311179 | 0.9663 | 0.4158 | 1/103/103/104 (104) | 0 | 178 MB max rank / 671 MB sum | 34.82 s partition / 76.87 s job | PHG_SCALE_BRIDGE_COMPLETE |
| FREIGHT 1-pass (frozen, CLOSED; context) | 167482101 | 0.9781 | 0.7583 | 57/104/104/104 | 0 | (frozen record) | (frozen record) | FREIGHT_CLOSED |
| Mt-KaHyPar (frozen recipe, local) | - | - | - | - | - | 5862 MB at kill (cap 6.0 GB) | 1132.0 s at kill | RESOURCE_INFEASIBLE_LOCAL |

Canonical L1 on the frozen dev sample (absolute; the FREIGHT row is context from a closed arm, not a reference):

| Arm | nq | BASE_ALL | SAFE_ALL | BASE_ANY | SAFE_ANY | scope BASE / SAFE | SAFE additions (+/-) | BND_ALL | balance max/mean | ELIGIBLE |
|---|---|---|---|---|---|---|---|---|---|---|
| PHG C1 (absolute) | 2000 | 0.6415 | 0.6965 | 0.9905 | 0.9920 | 5115.5 / 5112.8 | +121/-11 | 0.6415 | 1.0397 | True |
| FREIGHT 1-pass (frozen, CLOSED; context) | 2000 | 0.3710 | 0.4730 | 0.9045 | 0.9600 | 5176.0 / 5169.1 | +217/-13 | 0.3710 | 1.0397 | True |

Diagnostic only (no rule, no claim): PHG vs the closed FREIGHT arm on the identical 2000 ids -- SAFE 0.4730 -> 0.6965 (+0.2235, exact p 9.92e-99, +487/-40); BASE 0.3710 -> 0.6415 (+0.2705, exact p 4.43e-124, +581/-40).

Hop-wise: the canonical musique replay carries no hop labels (by_hop absent), as in the frozen FREIGHT record.

Sanity checks (structural, preregistered; no numeric retrieval threshold): S1_downstream_executed = True; S2_population = True; S3_canonical_eligibility = True; S4_definitional_consistency = True; S5_no_numeric_retrieval_threshold = True; ALL_PASS = True

Blocker (resources, not quality): the frozen Mt-KaHyPar recipe was killed at 5862 MB RSS after 1132.0 s (cap 6.0 GB as recorded, 4 threads); PHG's peak was 177.9 MB on the largest rank (670.9 MB summed over 4 ranks, an upper bound) and the partition took 34.824 s.

Host: before {"utc": "2026-09-13T19:03:45Z", "host_total_mb": 16069, "host_available_mb": 1759, "foreign_python_processes_over_1GB_rss_untouched": [{"pid": 8628, "name": "python.exe", "rss_mb": 4374, "started": "2026-09-13T20:29:08"}]}; after {"utc": "2026-09-13T19:16:06Z", "host_total_mb": 16069, "host_available_mb": 1451, "foreign_python_processes_over_1GB_rss_untouched": [{"pid": 8628, "name": "python.exe", "rss_mb": 6080, "started": "2026-09-13T20:29:08"}]}.

Integrity: frozen records changed = none; input/data pins changed = none; canonical musique partition still absent = True; experimental files written = ['data/l1_canonical/musique/parts/LOWMEM__PHG_C1_con.json', 'data/l1_canonical/musique/parts/LOWMEM__PHG_C1_con.npy', 'data/l1_lowmem/musique/phg_repair1/repaired.npy', 'data/l1_lowmem/musique/replay_cache__LOWMEM__PHG_C1_con.json', 'data/l1_lowmem/musique/replay_cache__LOWMEM__PHG_C1_con.npz', 'data/l1_lowmem/musique/phg/{G,R1,R1_repeat,STAGE*.json}'].

Caveats: 4 local MPI ranks validate feasibility on this host only (no aggregate-RAM claim at larger scale); the rank count is a contract element and was not changed; PHG_NONEMPTY_REPAIR_V1 is part of the contract (no-op when no block is empty).

WebQSP / HotpotQA / 2Wiki not run (separate authorization required); no promotion; no tuning; PHG not rerun; FREIGHT stays FREIGHT_CLOSED; squad / metaqa PHG records untouched
