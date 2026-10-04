# H4_SK_FREIGHT_RESTREAM_VCYCLE_2PASS -- follow-up arm (one fixed configuration, no sweeps)

Generated 2026-09-13T17:04:36Z -- DECISION: **FREIGHT_CLOSED**

Reason (preregistered): single-pass streaming produced substantially worse partition objective/cut quality; this follow-up tests FREIGHT's documented restreaming refinement mechanism

| Dataset | Arm | Exactness | Peak RSS | Time | KM1 | ratio vs Mt-KaHyPar | ratio vs 1-pass | STRUCT cut | KNN cut | max block (bound) | pass-1 nodes moved | Downstream |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| squad | 2-pass + restream_vcycle | EXACT_RESTART_EQUIVALENCE | 7 MB | 8.2 s | 18062025 | 2.7677 | 1.0000 | 0.9470 | 0.6491 | 104 (104) | 11346 | identical partition -> frozen paired result stands: SAFE -0.0210 p=0.0, BASE -0.0395 p=0.0 |
| squad | Mt-KaHyPar (frozen) | - | - | - | 6525934 | 1.0 | - | 0.9011 | 0.1943 | 104 (104) | - | - |
| squad | 1-pass (frozen) | - | 7 MB | 4.9 s | 18062025 | 2.7677 | 1.0 | 0.9470 | 0.6491 | 104 (104) | - | frozen |
| metaqa | 2-pass + restream_vcycle | EXACT_RESTART_EQUIVALENCE | 11 MB | 4.3 s | 55390296 | 1.343 | 0.9585 | 0.8144 | 0.8943 | 104 (104) | 13722 | PAIRED: SAFE -0.0275 p=0.00011, BASE -0.0245 p=0.00063 |
| metaqa | Mt-KaHyPar (frozen) | - | - | - | 41244100 | 1.0 | - | 0.8190 | 0.6308 | 104 (104) | - | - |
| metaqa | 1-pass (frozen) | - | 10 MB | 1.8 s | 57790340 | 1.4012 | 1.0 | 0.8031 | 0.9373 | 104 (104) | - | frozen |

Stop rule (preregistered): FREIGHT_CLOSED if, on squad or metaqa, (a) the restream partition is byte-identical to the frozen 1-pass FREIGHT partition (the frozen paired result then stands unchanged), or (b) the canonical paired downstream evaluation shows a SIGNIFICANT (exact two-sided McNemar p<0.05) SAFE_ALL_P50 or BASE_ALL_P50 loss vs Mt-KaHyPar, or (c) the output violates the imbalance contract (max block > ceil(1.03 N/k)).  FREIGHT_RESTREAM_QUALITY_PASS if both datasets are paired with no significant loss.  EXACTNESS_FAIL (engineering, separate from quality) if A != A_forklib or the sharded / checkpointed / killed / aborted runs differ from A.  Objective and cut ratios vs Mt-KaHyPar are reported as diagnostics; no numeric objective threshold is preregistered (whether the gap is 'materially closed' is the reviewer's call).  No pass-count, seed or order sweep follows a failure.

Reasons:

- squad: restream partition byte-identical to the frozen 1-pass partition (rule a)
- metaqa: significant paired SAFE_ALL_P50 loss -0.0275 (p=0.00011) (rule b)
- metaqa: significant paired BASE_ALL_P50 loss -0.0245 (p=0.00063) (rule b)

Per-pass facts (fork checkpoints META, B0 run):
- squad: {'pass0': {'checkpoint': 'checkpoint_0_00004.bin', 'next_node': '20000', 'nodes_moved': '0', 'best_objective': '1.7976931348623157e+308', 'stream_total_upperbound': '105', 'block0_capacity': '104', 'processed_pins': '1573561'}, 'pass1': {'checkpoint': 'checkpoint_1_00004.bin', 'next_node': '20000', 'nodes_moved': '11346', 'best_objective': '18062025', 'stream_total_upperbound': '104', 'block0_capacity': '104', 'processed_pins': '1573561'}}
- metaqa: {'pass0': {'checkpoint': 'checkpoint_0_00008.bin', 'next_node': '40000', 'nodes_moved': '0', 'best_objective': '1.7976931348623157e+308', 'stream_total_upperbound': '105', 'block0_capacity': '104', 'processed_pins': '496097'}, 'pass1': {'checkpoint': 'checkpoint_1_00008.bin', 'next_node': '40000', 'nodes_moved': '13722', 'best_objective': '57790340', 'stream_total_upperbound': '104', 'block0_capacity': '104', 'processed_pins': '496097'}}

Integrity: frozen records changed since preregistration = none; canonical H4_SK partitions unchanged = True; new files written by this arm: data/l1_lowmem/squad/freight_restream2/{A,A_forklib,B,B0,C_kill,C_abort}/, data/l1_canonical/metaqa/parts/LOWMEM__FREIGHT_con_restream2.npy, data/l1_canonical/metaqa/parts/LOWMEM__FREIGHT_con_restream2.json, data/l1_lowmem/metaqa/replay_cache__LOWMEM__FREIGHT_con_restream2.npz, data/l1_lowmem/metaqa/replay_cache__LOWMEM__FREIGHT_con_restream2.json, data/l1_lowmem/metaqa/freight_restream2/{A,A_forklib,B,B0,C_kill,C_abort}/, results/L1_LOWMEM/FREIGHT_RESTREAM_PREREG.json, results/L1_LOWMEM/FREIGHT_RESTREAM_RUNS_squad.json, results/L1_LOWMEM/FREIGHT_RESTREAM_RUNS_metaqa.json, results/L1_LOWMEM/L1_DOWNSTREAM_metaqa__restream2.json, results/L1_LOWMEM/L1_REPLAY_metaqa__LOWMEM__FREIGHT_con_restream2.json

Post-hoc observation (not a gate): metaqa: pass 1 lowered the H4-weighted KM1 by 4.2 % (57,790,340 -> 55,390,296) by cutting fewer KNN nets (KNN edge-cut 0.937 -> 0.894) at the price of more STRUCT cut (0.803 -> 0.814), and L1 coverage fell significantly, concentrated in hop3 (SAFE -0.066, BASE -0.069, sig; hop1/hop2 n.s.).  H4 weights max(1, rint(1000/(|e|-1))) make the small KNN nets worth ~1000 per cut and the large STRUCT nets a few units, so the streaming objective is dominated by KNN locality while the P50 coverage of multi-hop queries tracks STRUCT locality; Mt-KaHyPar reaches both (KNN 0.631, STRUCT 0.819).  Recorded as an observation after seeing the numbers; it changes no rule and motivates no re-run.

Frozen classification unchanged: LOW_MEMORY_BUT_QUALITY_FAIL.  Next per the decision tree: close FREIGHT permanently; next candidate = Zoltan-PHG on SQuAD / MetaQA (AUDIT_ZOLTAN_PHG.json: FEASIBLE_NOT_EXECUTED)
