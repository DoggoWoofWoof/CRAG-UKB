# H4_SK low-memory partitioning validation (FREIGHT vs canonical Mt-KaHyPar)

Generated 2026-09-13T16:38:49Z -- CLASSIFICATION: **LOW_MEMORY_BUT_QUALITY_FAIL**

| Dataset | Method | Structure exact | Peak RSS | Time | Objective (KM1) | Imbalance (max/ceil(N/k)-1) | STRUCT cut | KNN cut | BASE | SAFE |
|---|---|---|---|---|---|---|---|---|---|---|
| squad | Mt-KaHyPar (canonical) | yes | 540 MB | 81.0 s | 6525934 | 0.0297 | 0.9011 | 0.1943 | 0.9885 | 0.9930 |
| squad | FREIGHT con (sharded+ckpt) | yes | 7 MB | 4.9 s | 18062025 | 0.0297 | 0.9470 | 0.6491 | 0.9490 | 0.9720 |
| metaqa | Mt-KaHyPar (canonical) | yes | 966 MB | 185.7 s | 41244100 | 0.0297 | 0.8190 | 0.6308 | 0.6376 | 0.6401 |
| metaqa | FREIGHT con (sharded+ckpt) | yes | 10 MB | 1.8 s | 57790340 | 0.0297 | 0.8031 | 0.9373 | 0.6356 | 0.6371 |
| musique | Mt-KaHyPar (canonical) | yes | >5862 MB (killed at 6.0 GB cap, 2 attempts) | 1132.0 s (killed) | RESOURCE_INFEASIBLE_LOCAL | | | | | |
| musique | FREIGHT con (sharded+ckpt) | yes | 44 MB | 23.5 s | 167482101 | 0.0297 | 0.9781 | 0.7583 | 0.3710 | 0.4730 |

P50 post-processing statistics (canonical replay + l1_eval numerics; nothing tuned):

| Dataset | Method | Blocks used/k | Block size min/p50/p95/max | max/mean | BASE ALL | BASE ANY | BASE scope | SAFE ALL | SAFE ANY | SAFE scope | SAFE additions +g/-l (net) | SAFE p | churn/q |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| squad | Mt-KaHyPar (canonical) | 202/202 | 49/102/104/104 | 1.0383 | 0.9885 | 0.9885 | 5089.2 | 0.9930 | 0.9930 | 5085.8 | +9/-0 (+9) | 0.00391 | 3.725 |
| squad | FREIGHT con (sharded+ckpt) | 202/202 | 72/104/104/104 | 1.0383 | 0.9490 | 0.9490 | 5149.2 | 0.9720 | 0.9720 | 5140.4 | +52/-6 (+46) | 0.0 | 4.431 |
| metaqa | Mt-KaHyPar (canonical) | 432/432 | 14/102/104/104 | 1.0392 | 0.6376 | 0.9094 | 5023.3 | 0.6401 | 0.9134 | 5023.6 | +11/-6 (+5) | 0.33231 | 4.559 |
| metaqa | FREIGHT con (sharded+ckpt) | 432/432 | 29/104/104/104 | 1.0392 | 0.6356 | 0.9299 | 5176.3 | 0.6371 | 0.9309 | 5166.5 | +7/-4 (+3) | 0.54883 | 4.419 |
| musique | Mt-KaHyPar (canonical) | RESOURCE_INFEASIBLE_LOCAL | | | | | | | | | | | |
| musique | FREIGHT con (sharded+ckpt) | 1175/1175 | 57/104/104/104 | 1.0397 | 0.3710 | 0.9045 | 5176.0 | 0.4730 | 0.9600 | 5169.1 | +217/-13 (+204) | 0.0 | 5.188 |

Rule: STRUCTURE_MISMATCH if any local dataset fails gate 1; OBJECTIVE_INCOMPATIBLE if the FREIGHT objective family is not connectivity/KM1 (it is: freight_con; the Python KM1 recomputation equals FREIGHT's connectivity wherever FREIGHT's evaluator runs); LOW_MEMORY_BUT_QUALITY_FAIL if gates 1-3 pass and any paired dataset shows a SIGNIFICANT (exact McNemar p<0.05) SAFE_ALL_P50 or BASE_ALL_P50 loss; LOW_MEMORY_L1_EQUIVALENT if gates 1-3 and exact restart equivalence pass on all three local datasets, all three are paired, and no paired loss is significant; NEEDS_MORE_EVIDENCE otherwise (a pending stage, an unpaired dataset, or an unregistered threshold decision left to the user).  The 5 % objective-degradation threshold is not preregistered and does not gate.

Not promoted: FREIGHT stays EXPERIMENTAL (parts/LOWMEM__FREIGHT_con.*); canonical H4_SK manifests and partitions are unchanged. STOP_FOR_REVIEW before WebQSP / HotpotQA / 2Wiki.

Review items:

- squad: paired on 2000 identical query IDs; SAFE -0.0210 (p=0.0, sig True, +6/-48), BASE -0.0395 (p=0.0, sig True, +12/-91) -- this is the classification trigger
- squad objective diagnostic: FREIGHT KM1 = 2.768x Mt-KaHyPar (STRUCT cut delta +0.0459, KNN cut delta +0.4548); not a gate
- metaqa: paired on 1998 identical query IDs; SAFE -0.0030 (p=0.69808, sig False, +80/-86), BASE -0.0020 (p=0.82017, sig False, +85/-89)
- metaqa hop-wise (reported, not gated by the preregistered rule): significant hops = ['hop1 SAFE -0.0225 p=0.00149']; others n.s.
- metaqa objective diagnostic: FREIGHT KM1 = 1.401x Mt-KaHyPar (STRUCT cut delta -0.0159, KNN cut delta +0.3065); not a gate
- musique: BASELINE_ABSENT -- FREIGHT-arm numbers only (BASE 0.3710, SAFE 0.4730); the Mt-KaHyPar baseline is RESOURCE_INFEASIBLE_LOCAL
- structure exact on ['squad', 'metaqa', 'musique']; exact restart equivalence on ['squad', 'metaqa', 'musique']; refusals all refused on ['squad', 'metaqa', 'musique']; peak RSS FREIGHT 7 MB / 10 MB / 44 MB vs Mt-KaHyPar 540 MB / 966 MB / >5862 MB (killed)
- classification LOW_MEMORY_BUT_QUALITY_FAIL follows the rule as written before the downstream numbers existed; no threshold was added or moved afterwards

Canonical untouched (hashes re-read at report time): {'squad': {'H4_SK_partition_sha256': 'a45195b825fa2f2fed58a64b08aa5471a20ad946f2b47831db3689b1348f54ce', 'H4_SK_partition_matches_production_replay_pin': True, 'production_replay_cache_matches_manifest': True, 'experimental_files_written': ['parts/LOWMEM__FREIGHT_con.npy', 'parts/LOWMEM__FREIGHT_con.json', 'replay_cache__LOWMEM__FREIGHT_con.npz', 'replay_cache__LOWMEM__FREIGHT_con.json']}, 'metaqa': {'H4_SK_partition_sha256': '454875e83d467f1c33d0a99add91d1edd81dd96e640bc597dfe6e658543a8062', 'H4_SK_partition_matches_production_replay_pin': True, 'production_replay_cache_matches_manifest': True, 'experimental_files_written': ['parts/LOWMEM__FREIGHT_con.npy', 'parts/LOWMEM__FREIGHT_con.json', 'replay_cache__LOWMEM__FREIGHT_con.npz', 'replay_cache__LOWMEM__FREIGHT_con.json']}, 'musique': {'H4_SK_partition': 'ABSENT (FAILED_MEMORY_CAP)', 'production_replay_cache_matches_manifest': None, 'experimental_files_written': ['parts/LOWMEM__FREIGHT_con.npy', 'parts/LOWMEM__FREIGHT_con.json', 'replay_cache__LOWMEM__FREIGHT_con.npz', 'replay_cache__LOWMEM__FREIGHT_con.json']}}

## squad
- structure: ORIGINAL 4fd825932a3ae758, shards gate PASS, official-bytes gate PASS
- experiment 1 (exact restart equivalence): EXACT_RESTART_EQUIVALENCE; runs identical to A: {'A': True, 'A_forklib': True, 'B': True, 'B0': True, 'C_kill': True, 'C_abort': True}; refusals: True; kill test: KILLED_AND_RESUMED; 2-pass: EXACT
- gates: {'1_structure_exact': True, '2_feasibility_peak_rss_le_local_ram': True, '2b_exact_restart_equivalence': True, '3_partition_valid': True, '5_downstream': {'SAFE_delta': -0.021, 'SAFE_p': 0.0, 'SAFE_sig': True, 'BASE_delta': -0.0395, 'BASE_p': 0.0, 'BASE_sig': True}}
- quality (reported, not gated): {'objective_ratio_B_over_A': 2.7677, 'struct_cut_delta': 0.0459, 'knn_cut_delta': 0.4548, 'threshold': 'NOT PREREGISTERED (5 % proposed, not frozen)'}
- P50 post-processing Mt-KaHyPar: blocks used 202/202, size min/median/mean/p90/p99/max 49/102.0/100.16/104.0/104.0/104, cv 0.0648, max/mean 1.0383, ELIGIBLE True; BASE ALL/ANY 0.9885/0.9885 scope 5089.2; SAFE ALL/ANY 0.9930/0.9930 scope 5085.8; BND 0.9885; SAFE additions +9/-0 net 9 p=0.00391 sig=True, gold admitted/evicted 9/0, churn/q 3.725 (max 6, zero-churn queries 229)
- P50 post-processing FREIGHT: blocks used 202/202, size min/median/mean/p90/p99/max 72/104.0/100.16/104.0/104.0/104, cv 0.0927, max/mean 1.0383, ELIGIBLE True; BASE ALL/ANY 0.9490/0.9490 scope 5149.2; SAFE ALL/ANY 0.9720/0.9720 scope 5140.4; BND 0.9490; SAFE additions +52/-6 net 46 p=0.0 sig=True, gold admitted/evicted 52/6, churn/q 4.431 (max 6, zero-churn queries 21)
- downstream paired (nq 2000): SAFE A 0.9930 B 0.9720 d -0.0210 p=0.0 sig=True (+6/-48); BASE A 0.9885 B 0.9490 d -0.0395 p=0.0
- candidate coverage A vs B: BASE ANY 0.9885 vs 0.9490, SAFE ANY 0.9930 vs 0.9720, BASE scope 5089.2 vs 5149.2, SAFE scope 5085.8 vs 5140.4; SAFE additions A +9/-0 (net 9) vs B +52/-6 (net 46)

## metaqa
- structure: ORIGINAL fc7c4f0fc158c220, shards gate PASS, official-bytes gate PASS
- experiment 1 (exact restart equivalence): EXACT_RESTART_EQUIVALENCE; runs identical to A: {'A': True, 'A_forklib': True, 'B': True, 'B0': True, 'C_kill': True, 'C_abort': True}; refusals: True; kill test: KILLED_AND_RESUMED; 2-pass: None
- gates: {'1_structure_exact': True, '2_feasibility_peak_rss_le_local_ram': True, '2b_exact_restart_equivalence': True, '3_partition_valid': True, '5_downstream': {'SAFE_delta': -0.003, 'SAFE_p': 0.69808, 'SAFE_sig': False, 'BASE_delta': -0.002, 'BASE_p': 0.82017, 'BASE_sig': False}}
- quality (reported, not gated): {'objective_ratio_B_over_A': 1.4012, 'struct_cut_delta': -0.0159, 'knn_cut_delta': 0.3065, 'threshold': 'NOT PREREGISTERED (5 % proposed, not frozen)'}
- P50 post-processing Mt-KaHyPar: blocks used 432/432, size min/median/mean/p90/p99/max 14/102.0/100.08/104.0/104.0/104, cv 0.0617, max/mean 1.0392, ELIGIBLE True; BASE ALL/ANY 0.6376/0.9094 scope 5023.3; SAFE ALL/ANY 0.6401/0.9134 scope 5023.6; BND 0.6376; SAFE additions +11/-6 net 5 p=0.33231 sig=False, gold admitted/evicted 132/113, churn/q 4.559 (max 6, zero-churn queries 0)
  - hop-wise Mt-KaHyPar: hop1 (n 666) BASE 0.8979 SAFE 0.8979; hop2 (n 666) BASE 0.4880 SAFE 0.5015; hop3 (n 666) BASE 0.5270 SAFE 0.5210
- P50 post-processing FREIGHT: blocks used 432/432, size min/median/mean/p90/p99/max 29/104.0/100.08/104.0/104.0/104, cv 0.1518, max/mean 1.0392, ELIGIBLE True; BASE ALL/ANY 0.6356/0.9299 scope 5176.3; SAFE ALL/ANY 0.6371/0.9309 scope 5166.5; BND 0.6356; SAFE additions +7/-4 net 3 p=0.54883 sig=False, gold admitted/evicted 135/138, churn/q 4.419 (max 6, zero-churn queries 0)
  - hop-wise FREIGHT: hop1 (n 666) BASE 0.8754 SAFE 0.8754; hop2 (n 666) BASE 0.5105 SAFE 0.5135; hop3 (n 666) BASE 0.5210 SAFE 0.5225
- downstream paired (nq 1998): SAFE A 0.6401 B 0.6371 d -0.0030 p=0.69808 sig=False (+80/-86); BASE A 0.6376 B 0.6356 d -0.0020 p=0.82017
- candidate coverage A vs B: BASE ANY 0.9094 vs 0.9299, SAFE ANY 0.9134 vs 0.9309, BASE scope 5023.3 vs 5176.3, SAFE scope 5023.6 vs 5166.5; SAFE additions A +11/-6 (net 5) vs B +7/-4 (net 3)
  - hop1 (n 666): SAFE A 0.8979 B 0.8754 d -0.0225 p=0.00149; BASE d -0.0225 p=0.00149
  - hop2 (n 666): SAFE A 0.5015 B 0.5135 d +0.0120 p=0.43971; BASE d +0.0225 p=0.1284
  - hop3 (n 666): SAFE A 0.5210 B 0.5225 d +0.0015 p=1.0; BASE d -0.0060 p=0.7163

## musique
- structure: ORIGINAL 27628d1c0319dcbb, shards gate PASS, official-bytes gate PASS
- Mt-KaHyPar baseline: RESOURCE_INFEASIBLE_LOCAL -- exactly one isolated Mt-KaHyPar retry after host contention ended; if it fails again on memory -> RESOURCE_INFEASIBLE_LOCAL, no further retries; attempts (threads / wall s / peak RSS kB at kill / host avail GB at start): ['8 / 801.2 / 6002284 / 7.71', '4 / 1132.0 / 6002748 / 0.779247616']; cap 6.0 GB
- experiment 1 (exact restart equivalence): EXACT_RESTART_EQUIVALENCE; runs identical to A: {'A': True, 'A_forklib': True, 'B': True, 'B0': True, 'C_kill': True, 'C_abort': True}; refusals: True; kill test: KILLED_AND_RESUMED; 2-pass: None
- gates: {'1_structure_exact': True, '2_feasibility_peak_rss_le_local_ram': True, '2b_exact_restart_equivalence': True, '3_partition_valid': True, '5_downstream': None}
- quality (reported, not gated): {'objective_ratio_B_over_A': None, 'struct_cut_delta': None, 'knn_cut_delta': None, 'threshold': 'NOT PREREGISTERED (5 % proposed, not frozen)'}
- P50 post-processing FREIGHT: blocks used 1175/1175, size min/median/mean/p90/p99/max 57/104.0/100.03/104.0/104.0/104, cv 0.111, max/mean 1.0397, ELIGIBLE True; BASE ALL/ANY 0.3710/0.9045 scope 5176.0; SAFE ALL/ANY 0.4730/0.9600 scope 5169.1; BND 0.3710; SAFE additions +217/-13 net 204 p=0.0 sig=True, gold admitted/evicted 557/46, churn/q 5.188 (max 6, zero-churn queries 1)
- downstream: BASELINE_ABSENT
