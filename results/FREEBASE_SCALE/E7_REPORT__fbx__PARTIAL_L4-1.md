# E7 -- Freebase STRUCT-only V-cycle (VCQ512L4N4FM2), systems measurement -- PARTIAL (levels [4, 3, 2, 1] done; level 0 pending)

N = 301,977,131 nodes, S = 25, per-block cap 12,199,876 (ceil(1.03 N / 25)).  Nothing here is a recall claim.

## Levels (exact quotients)

| level | V | M (nets) | P (pins) | V reduction | P reduction |
|---|---|---|---|---|---|
| 0 | 301,977,131 | 301,926,705 | 4,253,534,391 | 1.00x | 1.00x |
| 1 | 178,333,613 | 176,520,867 | 1,690,890,784 | 1.69x | 2.52x |
| 2 | 59,213,072 | 121,250,920 | 1,251,065,170 | 5.10x | 3.40x |
| 3 | 23,474,979 | 103,777,069 | 937,114,098 | 12.86x | 4.54x |
| 4 | 11,664,088 | 84,930,375 | 712,838,071 | 25.89x | 5.97x |

## Surrogate (cap 4, level 4) vs budget

pins 21,714,432 = 5.6% of the 0.39e9 budget (stop rule 0.45e9: NOT triggered); Zoltan NP 4 wall 948 s, peak RSS 9.32 GB summed over ranks (3.64 GB max rank); surrogate cut 2,247,078,656, imbalance 1.0100.

## KM1 trajectory (full STRUCT-only hypergraph, projected top refined level by level)

| level | V | km1 in | km1 out | moves | load max / min after |
|---|---|---|---|---|---|
| 4 | 11,664,088 | 28,478,946,950 | 20,237,287,414 | 1,390,265 | 12,199,876 / 10,138,875 |
| 3 | 23,474,979 | 20,237,287,414 | 19,746,631,790 | 1,071,200 | 12,199,876 / 10,151,540 |
| 2 | 59,213,072 | 19,746,631,790 | 19,109,099,473 | 773,478 | 12,199,876 / 10,126,697 |
| 1 | 178,333,613 | 19,109,099,473 | 17,903,248,768 | 2,640,732 | 12,199,876 / 10,154,795 |

km1 last out / first in = 0.6286 (37.14% reduction through 4 refined levels).

## Peaks

max stage RSS 40.5 GB (budget 70 GB: ok); max crag work-dir disk 20.1 GB (budget 55 GB: ok).

## Not yet measured

Level 0 (the full 301,977,131-node FM2 refinement, peak RSS ~41 GB at level 1 so reserved 41 GB) and the final block-validity check are pending; they need a host capacity slot that mpr jobs currently hold.

## Not claimed

no recall at Freebase scale (no Freebase gold; the WebQSP bridge proxy is a later addendum), no K-way sub-partition, no SK arm; the verdict transferred from WebQSP is ON the pass line (7 rows, F = 8), one Zoltan seed
