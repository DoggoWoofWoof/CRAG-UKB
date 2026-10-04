# EXACT-P50 STRUCTURAL CONTRACT AUDIT -- FINAL REPORT

Diagnostic phase. No selector was invented, nothing was promoted, FINAL P stayed at exactly 50 in every measurement, and no TEST split was touched.

## DECISION: **D. MIXED** -- but with a strict ordering, and B is *dormant*

No single restriction owns the exact-P50 headroom, so the honest verdict is D. But D here is not a shrug: the audit produces a strict ordering of the restrictions, and it *overturns* both of the phase's own starting hypotheses in their literal form.

| rung (each relaxes exactly ONE restriction) | MetaQA hop3 | share | MetaQA ALL |
|---|---:|---:|---:|
| oracle selection instead of frozen F6 | +0.1051 | 14.9% | +0.0600 |
| restriction **A** as stated: M_struct 64 -> all admitted | +0.0270 | 3.8% | +0.0235 |
| the beam prune: admitted -> all visited (same search) | +0.2012 | 28.5% | +0.0986 |
| restriction **B**: B=6 -> B=50, P still exactly 50 | +0.1081 | 15.3% | +0.0506 |
| the bounded search itself -> whole corpus | +0.2643 | 37.5% | +0.0931 |
| **total** (SAFE 0.2583 -> full-universe P50 0.9640) | **+0.7057** | 100% | **+0.3258** |

The same ladder on all six corpora (ALL block), which is what makes the verdict D rather than a single-corpus story:

| corpus / block | SAFE | +selection | +read trunc | +beam trunc | +B capacity | +reach | = full universe | largest |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| metaqa / ALL | 0.6612 | +0.0600 | +0.0235 | +0.0986 | +0.0506 | +0.0931 | 0.9870 | beam_truncation |
| metaqa / hop1 | 0.9955 | +0.0015 | +0.0015 | +0.0015 | +0.0000 | +0.0000 | 1.0000 | selection |
| metaqa / hop2 | 0.7297 | +0.0736 | +0.0420 | +0.0931 | +0.0436 | +0.0150 | 0.9970 | beam_truncation |
| metaqa / hop3 | 0.2583 | +0.1051 | +0.0270 | +0.2012 | +0.1081 | +0.2643 | 0.9640 | reach |
| webqsp / ALL | 0.7646 | +0.0240 | +0.0106 | +0.0599 | +0.0007 | +0.1353 | 0.9951 | reach |
| 2wiki_clean / ALL | 0.9435 | +0.0080 | +0.0030 | +0.0125 | +0.0000 | +0.0330 | 1.0000 | reach |
| musique_clean / ALL | 0.9635 | +0.0115 | +0.0045 | +0.0150 | +0.0000 | +0.0055 | 1.0000 | beam_truncation |
| hotpotqa_clean / ALL | 0.9505 | +0.0140 | +0.0035 | +0.0120 | +0.0000 | +0.0200 | 1.0000 | reach |
| squad_clean / ALL | 0.9875 | +0.0070 | +0.0005 | +0.0025 | +0.0000 | +0.0025 | 1.0000 | selection |

`read_truncation` -- restriction **A** exactly as the directive states it -- is the smallest rung on **every corpus and every block**, without exception. `B_capacity` -- restriction **B** -- is *exactly* 0.0000 on 4 of 6 corpora (2wiki_clean, musique_clean, hotpotqa_clean, squad_clean) and +0.0007 on WebQSP, even with the widest evidence universe; it is nonzero only on MetaQA. The largest rung is `beam_truncation` on 3 blocks, `reach` on 4, `selection` on 2 -- no restriction dominates.

### What this overturns

**Restriction A, as literally stated, is nearly refuted.** The directive suspected the node-level *read* (top-`M_struct=64`) before partition aggregation. Removing it entirely -- aggregating every admitted node -- is the **smallest** rung on the ladder: +0.0270 hop3, +0.0235 ALL. The truncation that actually binds is one stage *earlier*: the beam prune, worth +0.2012 hop3 -- 7.5x more. The frozen search visits 1233 nodes per query and admits only 147 of them; the M64 read then discards a further 83. Evidence dies at the prune, not at the read.

**Restriction B is dormant, not absent.** Under the frozen M64 evidence universe, raising B from 6 to 50 buys +0.0030 on MetaQA hop3, +0.0010 on ALL, and exactly +0.0000 on the WebQSP control. B is not binding because the candidate universe cannot fill even six slots. Once the evidence truncation is lifted, the same B6->B50 move is worth +0.1081 hop3. **The two suspected restrictions are not independent: B is gated behind evidence.** Any reading of the B-sweep that ignores which evidence universe it was measured in is wrong.

The mechanical reason is in the need distribution: on 4 of 6 corpora (2wiki_clean, musique_clean, hotpotqa_clean, squad_clean) **not a single query** needs more than six replacements (mean need at B=6 is 0.02-0.08). B=6 only bites on MetaQA hop3 (213/666 queries, mean need 7.78) and mildly on WebQSP (65/1419).

**The previous phase's `needed partition has no evidence` bucket was a measurement artifact.** Measured through the M64 read, only 26.37% of needed partitions carry structural evidence. Measured against what the *same bounded search* already visited, 92.06% do (hop3 90.85%), and only 4.06% are in nothing at all. The search reaches the partitions; the contract throws them away before they are ever scored.

### What this does NOT license

Every rung above is an ORACLE ceiling. The audit also ran the **unchanged frozen F6 selector** on each evidence universe -- same selector, same B=6, same exact P=50, only the evidence differs. On the M64 universe it reproduces SAFE exactly on all six corpora (net +0 everywhere), which is the gate. On the full visited universe it is **significantly WORSE on MetaQA**: ALL 0.6612 -> 0.6537, hop3 0.2583 -> 0.2508, net -15 queries, p = 0.00027. On the other five it is null (net +6, -4, -2, -1, +1, none significant).

So the largest oracle rung is not merely unconverted by the incumbent selector -- it is *actively harmful* to it, and harmful precisely on the block where its ceiling is biggest. The extra evidence is dominated by distractors that dilute F6's structural aggregation. This does not prove the headroom is unreachable; it does prove that relaxing the truncation is not a free win and that S4_FULL_VISITED must not be promoted on the strength of its ceiling. Nothing was promoted.

## STEP 1 -- removing the node read as an analytic bottleneck

Three nested evidence universes, built as strict prefixes of ONE node order so that node ordering cannot confound the comparison; the aggregation is the frozen S4 verbatim.

```
S4_M64            first 64 admitted nodes, frozen `added` order      == frozen, gated
S4_FULL_ADMITTED  every admitted node,     frozen `added` order      read trunc removed
S4_FULL_VISITED   + every VISITED-but-pruned node, static score desc  beam trunc removed
```

Nothing here re-searches. The pruned nodes were *already* scored by the bounded beam and their statistics already accumulated; the beam simply discards them. No learned weights, no new graph search, no new edges.

**Gate:** the M64 prefix reproduces the frozen structural ranking on 1998/1998 MetaQA queries, and the frozen replay itself is bit-exact (1998/1998). Every corpus passes both. Without this the ladder would not be an audit of the frozen system.

Partitions receiving structural evidence per query:

| corpus | M64 | all admitted | all visited | growth |
|---|---:|---:|---:|---:|
| metaqa | 47.5 | 84.5 | 305.5 | 6.4x |
| webqsp | 36.6 | 67.6 | 417.1 | 11.4x |
| 2wiki_clean | 37.7 | 60.2 | 167.2 | 4.4x |
| musique_clean | 28.2 | 46.9 | 111.7 | 4.0x |
| hotpotqa_clean | 45.5 | 93.8 | 585.0 | 12.9x |
| squad_clean | 16.5 | 32.2 | 116.8 | 7.1x |

## STEP 2 -- CORE vs NOVEL: how much of the mechanism is confirmation

Classified purely on inference-time state (CORE = inside the canonical top-50, NOVEL = outside); gold is used only to count, never to classify.

| universe | NOVEL_GOLD_PARTITION_RECALL | CORE gold recall | share of evidence spent on CORE |
|---|---:|---:|---:|
| S4_M64 | **0.2608** | 0.4604 | 0.2728 |
| S4_FULL_ADMITTED | **0.3892** | 0.7244 | 0.2768 |
| S4_FULL_VISITED | **0.9195** | 0.9629 | 0.1450 |

The frozen read is disproportionately *confirmatory*: at M64 it recovers 0.4604 of gold partitions that were already in the canonical top-50 but only 0.2608 of the ones that are not -- and only the latter can ever change an exact-P50 outcome, because a CORE partition needs no swap. 27.3% of the partitions the frozen mechanism lights up are partitions the canonical channel already had. Under the visited universe NOVEL recall reaches **0.9195** and the confirmation share drops to 14.5%.

This is the cleanest statement of why the previous phase's read-recall improvements did not move exact-P50: generic gold-node read recall is dominated by CORE nodes, which are free.

## STEP 3 -- the needed-partition evidence ceiling (MetaQA)

`needed` = gold partitions outside the protected 44 -- exactly the partitions a swap must supply.

| | visited by the search | admitted by the beam | given S4_M64 evidence | given S4_FULL_VISITED evidence | any canonical/retrieval evidence | in NOTHING |
|---|---:|---:|---:|---:|---:|---:|
| hop2 (n=943) | 0.9862 | 0.4422 | 0.2651 | 0.9862 | 0.7688 | 0.0032 |
| hop3 (n=5183) | 0.9085 | 0.3863 | 0.2634 | 0.9085 | 0.6130 | 0.0475 |
| ALL (n=6132) | 0.9206 | 0.3951 | 0.2637 | 0.9206 | 0.6371 | 0.0406 |

The funnel is: visited 90.85% -> admitted 38.63% -> read at M64 26.34% on hop3. The bounded structural search is *not* the thing that fails to reach the needed partitions. The beam prune and the read discard them afterwards.

## STEP 4 -- how much headroom B blocks

| corpus | block | queries | mean need at B=6 | >6 | >8 | >12 | >20 |
|---|---|---:|---:|---:|---:|---:|---:|
| metaqa | ALL | 1998 | 3.07 | 246 | 219 | 159 | 88 |
| metaqa | hop2 | 666 | 1.42 | 33 | 30 | 23 | 11 |
| metaqa | hop3 | 666 | 7.78 | 213 | 189 | 136 | 77 |
| webqsp | ALL | 1419 | 1.28 | 65 | 47 | 31 | 17 |
| 2wiki_clean | ALL | 2000 | 0.07 | 0 | 0 | 0 | 0 |
| musique_clean | ALL | 2000 | 0.06 | 0 | 0 | 0 | 0 |
| hotpotqa_clean | ALL | 2000 | 0.08 | 0 | 0 | 0 | 0 |
| squad_clean | ALL | 2000 | 0.02 | 0 | 0 | 0 | 0 |

| MetaQA hop3 oracle | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.3634 | 0.3634 | 0.3664 | 0.3664 | 0.3664 |
| S4_FULL_ADMITTED | 0.3904 | 0.3919 | 0.3964 | 0.3964 | 0.3964 |
| S4_FULL_VISITED | 0.5916 | 0.6096 | 0.6517 | 0.6787 | 0.6997 |

| WebQSP control (ALL) | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.7886 | 0.7886 | 0.7886 | 0.7886 | 0.7886 |
| S4_FULL_ADMITTED | 0.7992 | 0.7992 | 0.7992 | 0.7992 | 0.7992 |
| S4_FULL_VISITED | 0.8591 | 0.8598 | 0.8598 | 0.8598 | 0.8598 |

## STEP 5 -- crossing the two ceilings

MetaQA hop3, oracle exact-P50, FINAL P = 50 in every cell:

| | B=6 | B=8 | B=12 | B=20 | B=50 |
|---|---:|---:|---:|---:|---:|
| S4_M64 | 0.3634 | 0.3634 | 0.3664 | 0.3664 | 0.3664 |
| S4_FULL_ADMITTED | 0.3904 | 0.3919 | 0.3964 | 0.3964 | 0.3964 |
| S4_FULL_VISITED | 0.5916 | 0.6096 | 0.6517 | 0.6787 | 0.6997 |

Reading the matrix along its two axes is the whole answer to the directive's question:

- along B, at frozen evidence: +0.0030
- along B, at full visited evidence: +0.1081
- along evidence, at B=6: +0.2282
- along evidence, at B=50: +0.3333

So **both** must change to bank the joint gain, but they are strictly ordered: evidence first, B second. Raising B alone is a no-op.

## STEP 6 -- the full-universe bookend

`FULL_UNIVERSE_P50_HOP3 = 0.9640`, `FULL_UNIVERSE_P50_ALL = 0.9870`. P = 50 is confirmed as not the binding constraint.

Mutually exclusive 3-way decomposition the directive asked for (MetaQA hop3):

| term | value |
|---|---:|
| reach gap (FULL_P50 - FULL_VISITED+B50) | +0.2643 |
| protected-core/B gap (FULL_VISITED+B50 - FULL_VISITED+B6) | +0.1081 |
| ranking/selection gap (FULL_VISITED+B6 - SAFE) | +0.3333 |
| **total** | **0.7057** |

The 5-rung ladder at the top of this report splits that third term further, which is where the real ordering shows up: it is not one 'selection' gap but selection +0.1051, read truncation +0.0270, beam truncation +0.2012.

## STEP 7 -- internal work

Downstream exposure is **exactly 50 partitions** in every configuration measured here. Only internal bookkeeping grows, and the graph search is byte-identical across the three universes -- the pruned nodes were already visited, scored and accumulated.

| corpus | edges touched/q | candidates scored/q | nodes visited/q | nodes admitted/q | nodes read/q (M64) | partition accumulations/q, M64 -> FULL_VISITED |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 1505.8 | 1241.0 | 1233.3 | 147.0 | 64 | 64 -> 1233 (19.3x) |
| webqsp | 1661.8 | 1213.7 | 1199.0 | 138.4 | 64 | 64 -> 1199 (18.7x) |
| 2wiki_clean | 601.1 | 382.6 | 381.1 | 105.7 | 64 | 64 -> 381 (6.0x) |
| musique_clean | 3196.0 | 1051.2 | 971.3 | 153.0 | 64 | 64 -> 971 (15.2x) |
| hotpotqa_clean | 1892.9 | 1181.9 | 1158.9 | 151.0 | 64 | 64 -> 1159 (18.1x) |
| squad_clean | 14181.3 | 2187.2 | 1741.6 | 158.8 | 64 | 64 -> 1742 (27.2x) |

Latency and memory, measured cleanly on an idle CPU over 400 queries per corpus:

| corpus | search ms/q | aggregate ms/q M64 | aggregate ms/q FULL_VISITED | added ms/q | transient bytes/q | downstream P |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 37.2 | 0.287 | 4.007 | +3.72 | 39,496 | **50** |
| webqsp | 52.7 | 0.295 | 4.906 | +4.61 | 36,277 | **50** |
| 2wiki_clean | 21.9 | 0.208 | 1.644 | +1.44 | 12,414 | **50** |
| musique_clean | 77.0 | 0.320 | 2.528 | +2.21 | 33,514 | **50** |
| hotpotqa_clean | 68.1 | 0.133 | 6.231 | +6.10 | 36,577 | **50** |
| squad_clean | 235.3 | 0.097 | 4.520 | +4.42 | 54,653 | **50** |

The graph search is byte-identical across the three universes, so the widest universe adds no edges and no candidate scoring -- only retention and aggregation. Retention was timed as two independent search passes differenced and came out negative on four of six corpora, i.e. indistinguishable from zero, which is the expected result when nothing is re-scored. The real added cost is the aggregation: +3.72 ms/q on MetaQA (10% of search) against a transient 39 KB/query of node arrays. This is not the reason to reject the wider universe.

## Realisability

The rungs above are ORACLE ceilings. The previous phase's hard-won lesson was that a bigger ceiling does not imply a bankable gain, so the audit also runs the **unchanged frozen F6 selector** on each evidence universe -- same selector, same B=6, same exact P=50, only the evidence it is shown differs. See `TABLES.md` T9.

| universe | MetaQA hop2 | hop3 | ALL | net vs SAFE | p |
|---|---:|---:|---:|---:|---:|
| SAFE (frozen) | 0.7297 | 0.2583 | 0.6612 | 0 | - |
| S4_M64 | 0.7297 | 0.2583 | 0.6612 | +0 | 1 |
| S4_FULL_ADMITTED | 0.7237 | 0.2553 | 0.6582 | -6 | 0.146 |
| S4_FULL_VISITED | 0.7147 | 0.2508 | 0.6537 | -15 | 0.00027 |


## Returns

```json
{
 "DECISION": "D_MIXED",
 "FULL_VISITED_NOVEL_PARTITION_RECALL": 0.9195,
 "NOVEL_PARTITION_RECALL_M64": 0.2608,
 "ORACLE_B6_HOP3": {
  "S4_M64": 0.3634,
  "S4_FULL_ADMITTED": 0.3904,
  "S4_FULL_VISITED": 0.5916
 },
 "ORACLE_B12_HOP3": {
  "S4_M64": 0.3664,
  "S4_FULL_ADMITTED": 0.3964,
  "S4_FULL_VISITED": 0.6517
 },
 "ORACLE_B20_HOP3": {
  "S4_M64": 0.3664,
  "S4_FULL_ADMITTED": 0.3964,
  "S4_FULL_VISITED": 0.6787
 },
 "ORACLE_B50_HOP3": {
  "S4_M64": 0.3664,
  "S4_FULL_ADMITTED": 0.3964,
  "S4_FULL_VISITED": 0.6997
 },
 "FULL_VISITED_B50_HOP3": 0.6997,
 "FULL_UNIVERSE_P50_HOP3": 0.964,
 "SAFE_HOP3": 0.2583,
 "B_IS_DORMANT_UNDER_FROZEN_EVIDENCE": {
  "hop3_M64_B6": 0.3634,
  "hop3_M64_B50": 0.3664,
  "gain": 0.003
 },
 "PROMOTED": "NONE",
 "L1_FROZEN": "NO"
}
```

Full machine-readable returns in `RETURNS.json`; all tables in `TABLES.md`; per-corpus raw in `diag/audit_<corpus>.json`.
