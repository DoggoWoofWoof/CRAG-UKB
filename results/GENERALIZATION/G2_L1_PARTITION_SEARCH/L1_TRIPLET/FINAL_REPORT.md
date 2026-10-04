# L1 PARAMETER-FREE STRUCTURAL-OFFSET TRIPLET PHASE -- FINAL REPORT

**Verdict: C. NODE_COLLAPSE_IS_LOSSLESS.  The triplet unit adds no separable evidence over the node unit.  PROMOTED = NONE.  B unchanged at 6.  L1_FROZEN = NO.**

The phase asked whether L1 improves if the structural evidence unit changes from NODE evidence to ACTUAL EDGE / TRIPLET evidence -- if edge transitions are preserved instead of immediately collapsing to target nodes.  The answer is that, for the score the collapse actually applies, **there is nothing to preserve**: the collapse is algebraically lossless, and this is a proof with an exact empirical check behind it, not a measurement that could have come out otherwise.

## What was built (STEP 1 + STEP 2)

The frozen bounded traversal already computes, for every actual directed edge `u -> v` it inspects,

```
delta = Xn[nb] - Xn[e];  nd = norm(delta, axis=1)
S = (delta[ok] / nd[ok, None]) @ rq        # == cosine(r_q, normalize(emb(v) - emb(u)))
```

which is exactly `T0_OFFSET` for the triplet `T = (u, delta_uv, v)`, and then discards it in
`np.maximum.at(best_s, inv, S)`.  Every other quantity STEP 1 asks for is likewise already in registers at that moment: the source's own arrival score (`F_edge[PAR]`, 1.0 for a retrieval seed) is T1_SOURCE, `hop + 1` is T3_HOP, `F_seed[PAR]` is the originating retrieval seed, and `PAR` gives the source node and its parent triplet.  T2_TARGET is `Xn[v] @ q_hat` on the same frozen unit-normalised embeddings the search already loaded.

So STEP 2 needed no additional graph search and no new encoder.  `expand_feat` gained one `want_edges` export that writes those arrays out and touches no search variable, following the pattern already used for `want_visited`.  Parity is therefore preserved by construction and asserted anyway on every query against the cached frozen `s_node / s_hop / s_sdir / s_cnt`:

| corpus | frozen replay parity | edges inspected/query | Pi->Pj transitions/query | transitions cached |
|---|---|---|---|---|
| metaqa | 1998/1998 EXACT | 1410.9 | 929.8 | 1,857,787 |
| musique_clean | 2000/2000 EXACT | 3136.9 | 403.5 | 807,006 |
| 2wiki_clean | 2000/2000 EXACT | 514.3 | 276.7 | 553,331 |
| squad_clean | 2000/2000 EXACT | 12009.7 | 420.4 | 840,705 |

delta is never materialised as a vector and never used to retrieve: only its inner product with the static query residual is taken, and only to score the REAL transition `u -> v`.  STEP 7's rule is respected by construction -- there is no `u + delta -> FAISS` anywhere in this phase.

## The decisive control: the collapse is lossless (STEP 5)

A partition score built by MAX cannot tell the two representations apart, because max is associative:

```
max over edges into Pj  ==  max over nodes v in Pj of ( max over edges into v )
```

The right-hand side is precisely what `np.maximum.at(best_s, inv, S)` computes and stores as `s_sdir`.  Measured on every query with at least one usable edge, the edge-level and node-level partition score vectors are **bit-identical, and so are the orderings they induce**:

| corpus | queries | no usable edge | identical score vectors | identical orderings |
|---|---|---|---|---|
| metaqa | 1998 | 0 | 1998/1998 | 1998/1998 |
| musique_clean | 2000 | 19 | 1981/1981 | 1981/1981 |
| 2wiki_clean | 2000 | 86 | 1914/1914 | 1914/1914 |
| squad_clean | 2000 | 295 | 1705/1705 | 1705/1705 |

That is 7598/7598 queries, exact, with zero exceptions.  The premise of the phase -- that the offset evidence is thrown away when edges collapse to nodes -- is false for the aggregation the pipeline actually uses.  `C0_SINGLE_TRIPLET` and the frozen node score are the same object under two names.

The edge unit and the node unit can only differ under aggregations that count, and under source-side quantities a node score cannot express.  Both were measured, and both are worse than the max they were supposed to improve on (MetaQA hop3, needed-partition recall):

| aggregation | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|
| EDGE_MAX | 0.0906 | 0.1441 | 0.2084 | 0.3412 |
| NODE_MAX | 0.0906 | 0.1441 | 0.2084 | 0.3412 |
| EDGE_SUM | 0.0520 | 0.0873 | 0.1273 | 0.2413 |
| NODE_SUM | 0.0387 | 0.0695 | 0.1117 | 0.2215 |
| EDGE_COUNT | 0.0198 | 0.0445 | 0.0867 | 0.2219 |
| NODE_COUNT | 0.0277 | 0.0526 | 0.0875 | 0.2237 |
| SOURCE_PARTITION_COUNT | 0.0142 | 0.0393 | 0.0790 | 0.2202 |
| SOURCE_NODE_COUNT | 0.0142 | 0.0370 | 0.0702 | 0.2224 |

Preserving the edge helps exactly once, for SUM (0.0520 against 0.0387 at R@6 -- a node reached by seven edges is counted seven times rather than once), and that whole family lands far below the max it is competing with (0.0906).  For COUNT the edge unit is worse than the node unit (0.0198 against 0.0277).  The source side -- the one axis genuinely unavailable to a node score -- is the weakest of all at R@6, which is the budget that actually matters, B = 6 being the entire swap allowance.

## STEP 3 + STEP 4 -- the necessary signal was not met

The directive's necessary condition, ahead of any end-to-end improvement, was that MARGINAL_NEEDED_PARTITION recall at small TRIPLET budgets should substantially exceed the current node/partition structural ranking at the same budget.  At R@64 -- 64 triplets against the 64 nodes of the frozen structural read -- the triplet unit is BELOW the node unit on every corpus:

| corpus | T0 R@16 | T0 R@32 | T0 R@64 | T0 R@128 | NODE R@16 | NODE R@32 | NODE R@64 | NODE R@128 | R@64 delta |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.1612 | 0.2488 | 0.3427 | 0.4435 | 0.1754 | 0.2611 | 0.3506 | 0.4581 | -0.0079 |
| musique_clean | 0.0984 | 0.1393 | 0.1803 | 0.2992 | 0.1230 | 0.1721 | 0.2623 | 0.3689 | -0.0820 |
| 2wiki_clean | 0.0435 | 0.0652 | 0.0870 | 0.1473 | 0.0435 | 0.0652 | 0.0894 | 0.1401 | -0.0024 |
| squad_clean | 0.0408 | 0.0612 | 0.1224 | 0.2245 | 0.1020 | 0.1633 | 0.2245 | 0.2449 | -0.1021 |

Across the whole grid of 16 corpus-by-budget cells the triplet unit is ahead in 1 (2wiki_clean R@128 +0.0072) and exactly ties in 2; everywhere else it is behind.

On MetaQA by block the gap is the same sign everywhere: hop2 0.3837 against 0.3892, hop3 0.3266 against 0.3355.  `T0_OFFSET` is nonetheless the best of the four triplet signals by a wide margin -- T1_SOURCE, T2_TARGET and T3_HOP are all far behind it at every budget -- so the ordering among the signals is exactly what the hypothesis predicted; it is the level that fails.

Per triplet the offset is barely better than chance at picking out a marginally useful transition.  The mean percentile rank of MARGINAL_USEFUL triplets under each signal (0.5 is chance) stays within 0.032 of chance on every signal over all queries:

| signal | ALL | hop1 | hop2 | hop3 |
|---|---|---|---|---|
| T0_OFFSET | 0.5323 | 0.4891 | 0.5184 | 0.5382 |
| T1_SOURCE | 0.4898 | 0.6004 | 0.4839 | 0.4915 |
| T2_TARGET | 0.5069 | 0.4467 | 0.5211 | 0.5015 |
| T3_HOP | 0.4995 | 0.5336 | 0.5414 | 0.4825 |
| C1_PATH_MIN | 0.4926 | 0.5234 | 0.4790 | 0.4979 |
| C1_PATH_SUM | 0.5225 | 0.5164 | 0.4855 | 0.5374 |

The two values that do move (T1_SOURCE 0.6004 and T2_TARGET 0.4467) are both on hop1, the block that has almost no marginal work to do at all -- 0.009 needed partitions per query -- so they rest on negligible support.  The ranking works better than the per-item percentile suggests only because sorting by offset concentrates targets, not because individual triplets are well separated.

## STEP 5 -- transition evidence does not separate novel Pj better than reach

The primary STEP 5 question was whether `Pi -> Pj` evidence separates a required novel `Pj` better than the plain fact that `Pj` was structurally reached.  Inside the identical universe, with the reach controls stated explicitly (MetaQA hop3):

| ordering | AUC | R@6 | R@12 | R@20 | R@50 |
|---|---|---|---|---|---|
| R0_REACHED_ARRIVAL | 0.5133 | 0.0008 | 0.0093 | 0.0343 | 0.1334 |
| R1_REACHED_COUNT | 0.5909 | 0.0222 | 0.0462 | 0.0880 | 0.2241 |
| TR_SOURCE_PARTITIONS | 0.5905 | 0.0142 | 0.0393 | 0.0790 | 0.2202 |
| TR_TRANSITION_COUNT | 0.5905 | 0.0142 | 0.0393 | 0.0790 | 0.2202 |
| C0_SINGLE_TRIPLET | 0.6481 | 0.0906 | 0.1441 | 0.2084 | 0.3412 |

The two orderings only the transition representation can express -- how many distinct source partitions feed Pj, and how many distinct transitions land on it -- score 0.5905 and 0.5905, against 0.5909 for simply counting the nodes of Pj that were reached.  They are the same signal.  Across corpora the transition-native ordering is at or below chance on MuSiQue and 2Wiki, and on no corpus does it separate more than 0.010 of AUC away from that control:

| corpus | R1_REACHED_COUNT | TR_SOURCE_PARTITIONS | difference |
|---|---|---|---|
| metaqa | 0.6218 | 0.6203 | -0.0015 |
| musique_clean | 0.4908 | 0.4869 | -0.0039 |
| 2wiki_clean | 0.4745 | 0.4838 | +0.0093 |
| squad_clean | 0.5783 | 0.5765 | -0.0018 |

**PARTITION_TRANSITION_INFORMATIVE = NO.**

## STEP 6 -- composition does not help either

Exactly three controls, no scorer grid (MetaQA hop3 AUC / R@6):

| control | AUC | R@6 |
|---|---|---|
| C0_SINGLE_TRIPLET | 0.6481 | 0.0906 |
| C1_BEST_COMPOSABLE_CHAIN (weakest link) | 0.5735 | 0.0181 |
| C1_BEST_COMPOSABLE_CHAIN (total) | 0.6194 | 0.0543 |
| C2_MULTI_PATH_SUPPORT (distinct ancestors) | 0.5921 | 0.0152 |
| C2_MULTI_PATH_SUPPORT (distinct seeds) | 0.5911 | 0.0119 |

On MetaQA every composed control is below the single transition it is composed from, and the two multi-path supports collapse onto the reach-count line (0.5921 and 0.5911 against 0.5909).

One corpus dissents on the chain score alone: on MuSiQue `C1_BEST_CHAIN_SUM` reaches 0.5572 against 0.5345 for the single transition.  Both sit close to chance, the corpus has 0.064 needed partitions per query to work with, and the chain ordering does not survive the exact-P50 mechanism on MetaQA (`PATH_S4` depth-matched net -5).

**COMPOSABLE_CHAIN_INFORMATIVE = NO.**

## STEP 8 -- complementarity is real, and it is small

Complementarity was measured before anything was combined, on the unit that matters: a query whose ENTIRE needed set is covered by the ordering's top 6, B = 6 being the whole swap budget.

| ordering | both | only triplet | only node | triplet alone | node alone | union ceiling |
|---|---|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 17 | 9 | 13 | 0.0130 | 0.0150 | 0.0195 |
| TR_EDGE_SUM | 5 | 11 | 25 | 0.0080 | 0.0150 | 0.0205 |
| CORE_EXIT_OFFSET | 19 | 10 | 11 | 0.0145 | 0.0150 | 0.0200 |
| CORE_EXIT_ONLY | 19 | 10 | 11 | 0.0145 | 0.0150 | 0.0200 |
| TRIPLET_S4 | 6 | 3 | 24 | 0.0045 | 0.0150 | 0.0165 |
| PATH_S4 | 5 | 2 | 25 | 0.0035 | 0.0150 | 0.0160 |
| FUSE_NODE_S4_X_CORE_EXIT | 23 | 10 | 7 | 0.0165 | 0.0150 | 0.0200 |

This is genuine disagreement, not noise: `CORE_EXIT_OFFSET` -- the one signal only a transition can express, a transition whose SOURCE partition is inside the protected core -- solves 10 queries the frozen node ordering does not, while the node ordering solves 11 it does not.  It is also small in absolute terms: a perfect combination would lift full coverage at B = 6 from 0.0150 to 0.0200.  That earned the ONE parameter-free combination the directive permits: an equal rank fusion of the frozen node ordering with the core-exit ordering, K0 = 60, no weights, no lambda, no threshold, no grid.

## STEP 10 -- through the unchanged mechanism at exact P = 50

The frozen candidate list is 47.5 partitions/query; the triplet universe is 305.5.  The preceding calibration phase established that a longer candidate list alone costs coverage through F6 by dilution, so reporting only full depth would confound representation with list size.  Both conditions are given, with the matched condition cutting each triplet ordering to exactly the frozen ordering's own per-query length K -- read off, never tuned.

**FULL depth** -- frozen SAFE: ALL 0.6612, hop1 0.9955, hop2 0.7297, hop3 0.2583

| ordering | ALL | hop1 | hop2 | hop3 | net | McNemar p | |
|---|---|---|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 0.6542 | 0.9955 | 0.7147 | 0.2523 | -14 | 0.00131 | **SIG** |
| TR_EDGE_SUM | 0.6512 | 0.9955 | 0.7087 | 0.2492 | -20 | 4e-05 | **SIG** |
| CORE_EXIT_OFFSET | 0.6537 | 0.9955 | 0.7162 | 0.2492 | -15 | 0.00073 | **SIG** |
| CORE_EXIT_ONLY | 0.6537 | 0.9955 | 0.7162 | 0.2492 | -15 | 0.00073 | **SIG** |
| TRIPLET_S4 | 0.6542 | 0.9970 | 0.7132 | 0.2523 | -14 | 0.00661 | **SIG** |
| PATH_S4 | 0.6522 | 0.9970 | 0.7072 | 0.2523 | -18 | 0.00091 | **SIG** |
| FUSE_NODE_S4_X_CORE_EXIT | 0.6542 | 0.9955 | 0.7162 | 0.2508 | -14 | 0.00052 | **SIG** |

**DEPTH-MATCHED** -- frozen SAFE: ALL 0.6612, hop1 0.9955, hop2 0.7297, hop3 0.2583

| ordering | ALL | hop1 | hop2 | hop3 | net | McNemar p | |
|---|---|---|---|---|---|---|---|
| C0_SINGLE_TRIPLET | 0.6607 | 0.9955 | 0.7282 | 0.2583 | -1 | 1 |  |
| TR_EDGE_SUM | 0.6547 | 0.9955 | 0.7147 | 0.2538 | -13 | 0.0106 | **SIG** |
| CORE_EXIT_OFFSET | 0.6622 | 0.9955 | 0.7312 | 0.2598 | +2 | 0.774 |  |
| CORE_EXIT_ONLY | 0.6622 | 0.9955 | 0.7312 | 0.2598 | +2 | 0.774 |  |
| TRIPLET_S4 | 0.6602 | 0.9955 | 0.7312 | 0.2538 | -2 | 0.851 |  |
| PATH_S4 | 0.6587 | 0.9955 | 0.7267 | 0.2538 | -5 | 0.473 |  |
| FUSE_NODE_S4_X_CORE_EXIT | 0.6627 | 0.9955 | 0.7312 | 0.2613 | +3 | 0.453 |  |

At full depth **7/7 orderings are significantly WORSE** than the frozen baseline, none better.  Depth-matched, 1/7 is significant in either direction -- TR_EDGE_SUM, and it is significantly WORSE, not better (0 depth-matched ordering is significantly better).  So the harm at full depth was list length, exactly as the calibration phase found, and the representation itself is worth approximately nothing.  The best depth-matched ordering is `FUSE_NODE_S4_X_CORE_EXIT` at ALL 0.6627 against 0.6612 (net +3, gained 5, lost 2, p = 0.45312), hop2 0.7312 against 0.7297, hop3 0.2613 against 0.2583.  Not significant on any block.

STEP 10 gates WebQSP and 2Wiki on MetaQA materially improving.  A non-significant +0.0015 is not material, so the exact-P50 mechanism was not run on the gated corpora.  The diagnostic steps that carry no such gate (STEPS 3-6 and 12) were run on MuSiQue, 2Wiki and SQuAD anyway.  They agree with MetaQA on the two findings the verdict rests on -- the collapse is lossless on every corpus, and the triplet unit is below the node unit at R@64 on all four (metaqa -0.0079, musique_clean -0.0820, 2wiki_clean -0.0024, squad_clean -0.1021).

Worth recording, though it is a cross-reference and not a measurement made here: hop3 0.2613 is the number the earlier read-stage separability phase recorded when it gave the M64 read a PERFECT node scorer.  Two unrelated routes into the same stage land on the same value, which is at least suggestive that the ceiling belongs to the contract rather than to either scorer.

## STEP 11 -- the budget curve does not move left

| ordering | k50 | k64 | k80 | k100 | k128 | k256 |
|---|---|---|---|---|---|---|
| NODE_S4_FROZEN | 0.6612 | 0.6842 | 0.7047 | 0.7377 | 0.7718 | 0.8278 |
| FUSE_NODE_S4_X_CORE_EXIT_DEPTH_MATCHED | 0.6627 | 0.6857 | 0.7052 | 0.7372 | 0.7718 | 0.8273 |
| C0_SINGLE_TRIPLET_DEPTH_MATCHED | 0.6607 | 0.6852 | 0.7057 | 0.7377 | 0.7718 | 0.8278 |

The required-partition curve is flat against the frozen one at every budget (k64 0.6857 against 0.6842, k128 0.7718 against 0.7718).  Nothing moves toward P50; the triplet representation does not make required partitions available earlier.

## STEP 12 -- the saturation the phase was meant to break is not a representation problem

| corpus | needed P/q | partitions reached/q | saturation | transitions/q | chains/q | transitions per partition | MARGINAL_USEFUL triplet prevalence |
|---|---|---|---|---|---|---|---|
| metaqa | 3.069 | 305.5 | 0.7619 | 929.8 | 990.1 | 3.043 | 0.01132 |
| musique_clean | 0.064 | 111.7 | 0.8216 | 403.5 | 782.7 | 3.611 | 0.00032 |
| 2wiki_clean | 0.073 | 167.2 | 0.2541 | 276.7 | 323.0 | 1.655 | 0.00013 |
| squad_clean | 0.025 | 116.8 | 0.6147 | 420.4 | 1709.0 | 3.599 | 0.00038 |

The transition representation is genuinely finer -- 3.021x more transitions than partitions on MetaQA hop3, 941.3 transitions and 1000.1 coherent chains against 311.5 partitions -- and conditioning on it does not reduce the saturation at all: the traversal still touches 77.7% of every partition in the corpus per query, because that is a property of where the beam goes, not of how its arrivals are named.  Splitting one saturated partition axis into three saturated transition axes lowers the prevalence of a useful unit rather than raising it: MARGINAL_USEFUL triplets are 2.65% of the hop3 edge universe.

## Returns

| return | value |
|---|---|
| `TRIPLET_OFFSET_INFORMATIVE` | YES |
| `PARTITION_TRANSITION_INFORMATIVE` | NO |
| `COMPOSABLE_CHAIN_INFORMATIVE` | NO |
| `BEST_TRIPLET_SIGNAL` | T0_OFFSET |
| `MARGINAL_PARTITION_R64_NODE_BASE` | 0.3506 |
| `MARGINAL_PARTITION_R64_TRIPLET` | 0.3427 |
| `EXACT_P50_METAQA_HOP2` | 0.7312 |
| `EXACT_P50_METAQA_HOP3` | 0.2613 |
| `TRIPLET_L1_VERDICT` | C. NODE_COLLAPSE_IS_LOSSLESS -- THE TRIPLET UNIT ADDS NO SEPARABLE EVIDENCE OVER THE NODE UNIT |

`TRIPLET_OFFSET_INFORMATIVE = YES` needs its qualification stated with it.  The offset is informative -- it is the best of the four signals and it separates needed partitions well above both reach controls (hop3 AUC 0.6481 against 0.5133 arrival and 0.5909 count).  It is not informative *as a triplet*: the identical quantity is already carried by the frozen node score, and the phase's own control proves the two are the same object.

## What this closes, and what it does not

**Closed.** The evidence-unit axis.  Node vs edge vs partition-transition vs chain is not a live degree of freedom in L1: the collapse that motivated the phase is lossless, the aggregations under which it is not lossless are worse, and the only genuinely transition-native signals are indistinguishable from counting reached nodes.  A finer unit is not the lever.

**Not closed, and not touched.** L1 remains unfrozen.  Everything this phase found is consistent with the standing account: the binding constraint is the beam's reach and the 7.782-needed-partitions-against-B=6 capacity ratio on MetaQA hop3, and both are properties of the traversal contract, not of how its arrivals are represented.  The small real complementarity STEP 8 found (core-exit transitions solving 10 queries the node ordering misses) is the one thread here that did not go to zero, and it is worth +0.0050 at its ceiling -- too small to promote on its own evidence.  It is also the one quantity in this phase that genuinely REQUIRES the transition unit: filtering the offset by whether the SOURCE partition is inside the protected core is exactly the information `np.maximum.at` destroys, because the collapse maxes over incoming edges without regard to where they came from.  So the evidence-unit axis is closed for target scoring and open, barely, for source-conditioned filtering -- on a lift that is not significant here and would need its own justification to pursue.

**Nothing was promoted.  B stays at 6.  P stays at 50.  No parameter, model, encoder, partitioning or selector changed.  L2 and L3 were not touched and TEST was not run.**

Artifacts: `FINAL_REPORT.md`, `TABLES.md`, `RETURNS.json`, `diag/{tp,ctrl,p50}_<corpus>.json`, `data/tr_<corpus>.npz`.  Code: `scratchpad/_l1tp_core.py` (replay + parity), `_l1tp_build.py` (STEPS 1-6 substrate), `_l1tp_run.py` (STEPS 3-6, 12), `_l1tp_ctrl.py` (the lossless-collapse control), `_l1tp_p50.py` (STEPS 8-11), `_l1tp_report.py`, `_l1tp_final.py`.  The `want_edges` export added to `scratchpad/_l1ss_core.py` is inert unless requested.
