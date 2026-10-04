# H4_SK_ZOLTAN_PHG_CONNECTIVITY_NP4 -- Zoltan-PHG substitution validation (SQuAD + MetaQA) + PHG_NONEMPTY_REPAIR_V1

Generated 2026-09-13T18:32:01Z (supersedes results/L1_LOWMEM/_history/PHG_REPORT.618883c9.json) -- STATUS: **STOP_FOR_REVIEW** -- final decisions: squad = **PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS**, metaqa = **PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS**

Terminology amendment: `PHG_L1_NONINFERIOR` -> `PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS` (rule identical: SIG_LOSS iff delta < 0 and p < 0.05; no non-inferiority margin was preregistered, so no non-inferiority claim is made; the superseded records are in results/L1_LOWMEM/_history/).

Primary question: does canonical L1 with the PHG partition show no significant paired SAFE loss against Mt-KaHyPar on the identical frozen H4_SK input and identical dev query ids?  Partition equality is not required; KM1 / cuts are diagnostics.

| Dataset | Arm | KM1 | KM1 / MtK | STRUCT cut | KNN cut | blocks min/p50/max (bound) | empty | SAFE (paired, dev) | BASE (paired, dev) | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| squad | PHG R1 (NP=4) | 7538754 | 1.1552 | 0.9282 | 0.2172 | 60/102/103 (104) | 0 | 0.9930 -> 0.9900 (-0.0030, p=0.23788, +6/-12) | 0.9885 -> 0.9845 (-0.0040, p=0.20049, +11/-19) | PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS |
| squad | PHG R1 + PHG_NONEMPTY_REPAIR_V1 | (no-op: 0 moves, vector byte-identical) | | | | | | | | REPAIR_NOOP_IDENTICAL |
| squad | Mt-KaHyPar (canonical, frozen) | 6525934 | 1.0 | 0.9011 | 0.1943 | 49/102/104 | 0 | (reference) | (reference) | |
| squad | FREIGHT 1-pass (frozen, CLOSED) | 18062025 | 2.7677 | 0.9470 | 0.6491 | 72/104/104 | 0 | (frozen record) | (frozen record) | FREIGHT_CLOSED |
| metaqa | PHG R1 (NP=4) | 45623771 | 1.1062 | 0.8318 | 0.7320 | 0/103/104 (104) | 2 | - | - | PHG_PARTITION_INVALID |
| metaqa | PHG R1 + PHG_NONEMPTY_REPAIR_V1 (2 moves) | 45623771 | 1.1062 | 0.8318 | 0.7320 | 1/103/104 (104) | 0 | 0.6401 -> 0.6712 (+0.0310, p=0.0, +112/-50) | 0.6376 -> 0.6707 (+0.0330, p=0.0, +113/-47) | PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS |
| metaqa | Mt-KaHyPar (canonical, frozen) | 41244100 | 1.0 | 0.8190 | 0.6308 | 14/102/104 | 0 | (reference) | (reference) | |
| metaqa | FREIGHT 1-pass (frozen, CLOSED) | 57790340 | 1.4012 | 0.8031 | 0.9373 | 29/104/104 | 0 | (frozen record) | (frozen record) | FREIGHT_CLOSED |

Objective gap vs L1 outcome (FREIGHT records frozen, PHG this lane):

| Arm | Dataset | Objective gap (KM1 / Mt-KaHyPar) | Paired L1 vs Mt-KaHyPar (same dev ids) | Outcome |
|---|---|---|---|---|
| FREIGHT 1-pass | squad | 2.768x | SAFE -0.0210 (p=0.0, sig), BASE -0.0395 (p=0.0, sig) | significant loss (LOW_MEMORY_BUT_QUALITY_FAIL) |
| PHG R1 | squad | 1.155x | SAFE -0.0030 (p=0.23788), BASE -0.0040 (p=0.20049) | PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS |
| FREIGHT 1-pass | metaqa | 1.401x | SAFE -0.0030 (p=0.69808), BASE -0.0020 (p=0.82017) | n.s. overall (hop1 sig loss) |
| FREIGHT 2-pass restream | metaqa | 1.343x | SAFE -0.0275 (p=0.00011, sig), BASE -0.0245 (p=0.00063, sig) | significant loss (FREIGHT_CLOSED) |
| PHG R1 | metaqa | 1.106x | not evaluated (2 empty blocks) | PHG_PARTITION_INVALID |
| PHG R1 + PHG_NONEMPTY_REPAIR_V1 | metaqa | 1.106x | SAFE +0.0310 (p=0.0, sig), BASE +0.0330 (p=0.0, sig) | PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS |

PHG_NONEMPTY_REPAIR_V1 (preregistered adapter; PHG not rerun):

| Dataset | empty blocks | moves | moved node -> block (from donor, donor size) | delta KM1 per move | KM1 before -> after | STRUCT cut before -> after | KNN cut before -> after | vector sha in -> out | postconditions | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| squad | [] | 0 | - | - | 7538754 -> 7538754 (+0, +0.00e+00 rel) | 0.9282 -> 0.9282 | 0.2172 -> 0.2172 | c37be0971cf9 -> c37be0971cf9 | PASS | REPAIR_NOOP_IDENTICAL |
| metaqa | [215, 430] | 2 | 69 -> 215 (from 104, size 103); 223 -> 430 (from 50, size 103) | +0, +0 | 45623771 -> 45623771 (+0, +0.00e+00 rel) | 0.8318 -> 0.8318 | 0.7320 -> 0.7320 | b516fbbab59b -> 29ccfc18b184 | PASS | PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS |

Hop-wise (diagnostic, no rule):

| Dataset | Arm | hop | n | SAFE MtK -> PHG | BASE MtK -> PHG |
|---|---|---|---|---|---|
| metaqa | PHG R1 + repair | hop1 | 666 | 0.8979 -> 0.8904 (-0.0075, p=0.40487) | 0.8979 -> 0.8904 (-0.0075, p=0.40487) |
| metaqa | PHG R1 + repair | hop2 | 666 | 0.5015 -> 0.5315 (+0.0300, p=0.01866) | 0.4880 -> 0.5345 (+0.0465, p=0.00019) |
| metaqa | PHG R1 + repair | hop3 | 666 | 0.5210 -> 0.5916 (+0.0706, p=0.0) | 0.5270 -> 0.5871 (+0.0601, p=0.0) |

Coverage / SAFE additions / balance (diagnostic):

| Dataset | Arm | SAFE_ANY MtK / PHG | BASE_ANY MtK / PHG | SAFE scope nodes MtK / PHG | SAFE additions MtK / PHG (gained/lost) | balance max/mean MtK / PHG |
|---|---|---|---|---|---|---|
| squad | PHG R1 | 0.9930 / 0.9900 | 0.9885 / 0.9845 | 5085.8 / 5085.5 | 9/0 / 12/1 | 1.0383 / 1.0283 |
| metaqa | PHG R1 + repair | 0.9134 / 0.9204 | 0.9094 / 0.9219 | 5023.6 / 5086.1 | 11/6 / 6/5 | 1.0392 / 1.0392 |

Structure gates (exact):
- squad: PASS -- N 20233 M 39146 P 1589832 weight sum 6879755; digest from the query functions 4fd825932a3ae758 == ORIGINAL 4fd825932a3ae758; weight reception (Zoltan cutl vs Python KM1 of the rank ownership): 0.0; partition-run dumps identical: True; Zoltan removal/warning lines: none
- metaqa: PASS -- N 43234 M 86796 P 531006 weight sum 32317710; digest from the query functions fc7c4f0fc158c220 == ORIGINAL fc7c4f0fc158c220; weight reception (Zoltan cutl vs Python KM1 of the rank ownership): 3.5e-08; partition-run dumps identical: True; Zoltan removal/warning lines: none

PHG runs (4 local MPI ranks; validation of implementation/quality only -- no aggregate-RAM claim at scale):
- squad: peak RSS per rank [53.1, 52.3, 53.0, 53.0] MB (max 53, sum 211 = upper bound); partition wall 4.24 s, job wall 11.76 s; validity PASS (max 103, bound 104, empty 0); Python KM1 vs Zoltan cutl 0.0; repeat run identical True
- metaqa: peak RSS per rank [34.2, 33.3, 34.0, 35.4] MB (max 35, sum 137 = upper bound); partition wall 2.62 s, job wall 6.52 s; validity PARTITION_INVALID (max 104, bound 104, empty 2); Python KM1 vs Zoltan cutl 2e-08; repeat run identical True

Gates / decision rule (PHG_PREREG.json, label renamed): per dataset: PHG_STRUCTURE_MISMATCH | PHG_PARTITION_INVALID | PHG_L1_NONINFERIOR | PHG_L1_SIG_LOSS (from gates 1-3 in that order).  SQuAD runs first; MetaQA runs only if SQuAD is structurally valid (gate 1 PASS) -- with the identical contract.  After both: STOP_FOR_REVIEW.  No tuning, no second configuration, no other rank count, no other dataset.

Repair algorithm (PHG_REPAIR_PREREG.json): input: the frozen PHG R1 k-way vector of the dataset (assembled from data/l1_lowmem/<ds>/phg/R1/part_rank*.txt, sha-pinned in PHG_RUNS_<ds>.json) and the frozen H4_SK hypergraph (weighted nets over canonical positions) | for each empty block id b, in ascending order: |   1. candidates = nodes v whose current (donor) block a = part[v] has size >= 2 |   2. delta(v) = exact change of the weighted KM1 = sum over nets e containing v of w_e * (1 - [count(e, a) == 1]) -- b is empty so every net of v gains one block; a net loses one block iff v was its only pin in a |   3. choose the candidate with the minimum delta |   4. ties: smaller canonical node position, then smaller donor block id (a node has exactly one donor block, so the third key can never decide between distinct nodes; implemented as a lexicographic sort on (delta, position, donor) for fidelity) |   5. move exactly that node: part[v] = b |   6. recompute block sizes and the per-(net, block) pin counts / connectivity before the next empty block (exact, from scratch) | moves == number of empty blocks of the input (one move per empty block); no other node changes block | postconditions: same node universe (length N, every position assigned exactly once), same k, block ids in [0, k), 0 empty blocks, max block <= ceil(1.03 N/k) (the canonical bound; a move can only shrink a donor of size >= 2 and create a singleton), exact H4_SK unchanged (npz sha + H4_SK_STRUCTURE_V1 digest == ORIGINAL), KM1_after == KM1_before + sum of the recorded deltas (full recomputation) | must NOT read: queries, answers, hop labels, BASE, SAFE, coverage, any replay/eval result; no parameter search; no minimum-block-size rule

Integrity: frozen records changed = none; canonical H4_SK partitions unchanged = True; superseded originals: {'PHG_RUNS_squad.json': 'results/L1_LOWMEM/_history/PHG_RUNS_squad.30559e1c.json', 'PHG_REPORT.json': 'results/L1_LOWMEM/_history/PHG_REPORT.618883c9.json', 'PHG_REPORT.md': 'results/L1_LOWMEM/_history/PHG_REPORT.d7dc4f8b.md'}

Caveats: PHG's 2D layout and coarsening depend on the rank count; partitions are not assumed identical across rank counts; NP is part of the contract.  Multiple LOCAL ranks validate implementation and quality only -- no claim about lower aggregate host RAM at large scale is made from this run.  no checkpoint/restart guarantee is claimed for PHG (FREIGHT's proven exact restart does not transfer); stage-boundary artifacts only: H4_SK input verified -> PHG input/callback manifest -> completed PHG partition -> P50 -> replay cache.  Same-rank-count repeatability is measured by one identical repeat run (diagnostic; the FIRST run is the arm's partition).

MuSiQue / WebQSP / HotpotQA / 2Wiki not run; no promotion; no tuning; PHG not rerun; FREIGHT stays FREIGHT_CLOSED
