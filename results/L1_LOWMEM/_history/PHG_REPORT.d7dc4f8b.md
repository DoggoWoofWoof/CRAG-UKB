# H4_SK_ZOLTAN_PHG_CONNECTIVITY_NP4 -- Zoltan-PHG substitution validation (SQuAD + MetaQA)

Generated 2026-09-13T17:59:28Z -- STATUS: **STOP_FOR_REVIEW** -- decisions: squad = **PHG_L1_NONINFERIOR**, metaqa = **PHG_PARTITION_INVALID**

Primary question (preregistered): does canonical L1 with the PHG partition reach at least Mt-KaHyPar-level final SAFE recall on the identical frozen H4_SK input and identical dev query ids?  Partition equality is not required; KM1 / cuts are diagnostics.

| Dataset | Arm | Ranks | Partition wall | Peak RSS max rank / sum | KM1 | KM1 ratio vs MtK | STRUCT cut | KNN cut | blocks min/p50/p95/max (bound) | empty | repeat identical | SAFE (paired) | BASE (paired) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| squad | PHG (PHG_L1_NONINFERIOR) | 4 | 4.24 s | 53 / 211 MB | 7538754 | 1.1552 | 0.9282 | 0.2172 | 60/102/103/103 (104) | 0 | True | 0.9930 -> 0.9900 (-0.0030, p=0.23788, +6/-12) | 0.9885 -> 0.9845 (-0.0040, p=0.20049, +11/-19) |
| squad | Mt-KaHyPar (canonical, frozen) | - | 81.0 s | 539.5 MB | 6525934 | 1.0 | 0.9011 | 0.1943 | 49/102/104/104 | 0 | - | (reference) | (reference) |
| squad | FREIGHT 1-pass (frozen, CLOSED) | 1 | - | - | 18062025 | 2.7677 | 0.9470 | 0.6491 | 72/104/104/104 | 0 | - | (frozen record) | (frozen record) |
| metaqa | PHG (PHG_PARTITION_INVALID) | 4 | 2.62 s | 35 / 137 MB | 45623771 | 1.1062 | 0.8318 | 0.7320 | 0/103/103/104 (104) | 2 | True | - | - |
| metaqa | Mt-KaHyPar (canonical, frozen) | - | 185.7 s | 966.4 MB | 41244100 | 1.0 | 0.8190 | 0.6308 | 14/102/104/104 | 0 | - | (reference) | (reference) |
| metaqa | FREIGHT 1-pass (frozen, CLOSED) | 1 | - | - | 57790340 | 1.4012 | 0.8031 | 0.9373 | 29/104/104/104 | 0 | - | (frozen record) | (frozen record) |

Hop-wise (diagnostic, no rule):

| Dataset | hop | n | SAFE MtK -> PHG | BASE MtK -> PHG |
|---|---|---|---|---|

Coverage / SAFE additions / P50 (diagnostic):

| Dataset | SAFE_ANY MtK / PHG | BASE_ANY MtK / PHG | SAFE scope nodes | SAFE additions MtK / PHG | BND_ALL MtK / PHG | balance max/mean MtK / PHG |
|---|---|---|---|---|---|---|
| squad | 0.9930 / 0.9900 | 0.9885 / 0.9845 | 5085.8 / 5085.5 | {'gained': 9, 'lost': 0, 'net': 9} / {'gained': 12, 'lost': 1, 'net': 11} | 0.9885 / 0.9845 | 1.0383 / 1.0283 |

Structure gates:
- squad: PASS -- N 20233 M 39146 P 1589832 weight sum 6879755; digest from the query functions 4fd825932a3ae758 == ORIGINAL 4fd825932a3ae758; partition-run dumps identical: True; Zoltan removal/warning lines: none
- metaqa: PASS -- N 43234 M 86796 P 531006 weight sum 32317710; digest from the query functions fc7c4f0fc158c220 == ORIGINAL fc7c4f0fc158c220; partition-run dumps identical: True; Zoltan removal/warning lines: none

Stage 3 (memory / timing / cross-check):
- squad: 4 ranks; peak RSS per rank [53.1, 52.3, 53.0, 53.0] MB (max 53, sum 211 -- sum is an upper bound); partition wall 4.24 s, job wall 11.76 s; Python KM1 vs Zoltan cutl: {'python_km1': 7538754, 'zoltan_cutl_float': 7538754.0, 'relative_diff': 0.0, 'consistent_within_float32': True}
- metaqa: 4 ranks; peak RSS per rank [34.2, 33.3, 34.0, 35.4] MB (max 35, sum 137 -- sum is an upper bound); partition wall 2.62 s, job wall 6.52 s; Python KM1 vs Zoltan cutl: {'python_km1': 45623771, 'zoltan_cutl_float': 45623772.0, 'relative_diff': 2e-08, 'consistent_within_float32': True}

Post-hoc observation (metaqa, NOT a gate, not acted on): empty blocks [215, 430]; blocks below half the mean size {212: 47, 215: 0, 429: 25, 430: 0, 431: 4} (Mt-KaHyPar min block 14).  Zoltan's balance constraint is one-sided (max part weight <= IMBALANCE_TOL x average); an empty or tiny part violates nothing PHG enforces, and PHG_OUTPUT_LEVEL=1 shows recursive bisection (p=2) of k=432 with that upper-only bound at every split.  The canonical partition.py marks a vector with unused blocks BUILT_WITH_WARNINGS, not OK, so the preregistered 'no empty block' gate is the canonical contract, not a new threshold.

Gates and decision rule (preregistered): per dataset: PHG_STRUCTURE_MISMATCH | PHG_PARTITION_INVALID | PHG_L1_NONINFERIOR | PHG_L1_SIG_LOSS (from gates 1-3 in that order).  SQuAD runs first; MetaQA runs only if SQuAD is structurally valid (gate 1 PASS) -- with the identical contract.  After both: STOP_FOR_REVIEW.  No tuning, no second configuration, no other rank count, no other dataset.

Integrity: frozen records changed since preregistration = none; canonical H4_SK partitions unchanged = True

Caveats: PHG's 2D layout and coarsening depend on the rank count; partitions are not assumed identical across rank counts; NP is part of the contract.  Multiple LOCAL ranks validate implementation and quality only -- no claim about lower aggregate host RAM at large scale is made from this run..  no checkpoint/restart guarantee is claimed for PHG (FREIGHT's proven exact restart does not transfer); stage-boundary artifacts only: H4_SK input verified -> PHG input/callback manifest -> completed PHG partition -> P50 -> replay cache.  Same-rank-count repeatability is measured by one identical repeat run (diagnostic; the FIRST run is the arm's partition).

MuSiQue / WebQSP / HotpotQA / 2Wiki not run; no promotion; no tuning; FREIGHT stays FREIGHT_CLOSED
