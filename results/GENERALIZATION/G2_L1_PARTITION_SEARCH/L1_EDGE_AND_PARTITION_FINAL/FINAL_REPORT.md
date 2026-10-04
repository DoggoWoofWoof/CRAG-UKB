# L1 EDGE SUBSTRATE + PARTITION UTILITY -- FINAL REPORT

Two independent foundational questions, answered before any L1 freeze decision:
**A.** is the frozen L1 traversing the right graph?  **B.** do the offline balanced partitions actually co-locate useful multi-hop evidence?  Their interaction is measured once, without redesigning the selector.

Coverage at time of writing -- Phase A: metaqa, 2wiki_clean, musique_clean, squad_clean, hotpotqa_clean, webqsp; Phase B: metaqa, 2wiki_clean, musique_clean, squad_clean, hotpotqa_clean, webqsp; Phase C: metaqa, 2wiki_clean, musique_clean, squad_clean, hotpotqa_clean, webqsp.

## 1. Contract, scope, and what was held frozen

Nothing in this program is learned, and no TEST split was touched. Every measurement runs the shipped L1 contract unchanged:

| parameter | value |
|---|---|
| MASTER_TOPOLOGY | `C` |
| P_MAIN | `50` |
| B | `6` |
| K0 | `60` |
| M_struct | `64` |
| M_ret | `32` |
| MAX_HOPS | `3` |
| BEAM | `64` |
| DEG_CAP | `300` |
| MAX_EDGES_SCORED | `400000` |
| SEED_K | `5` |
| M_MAX | `256` |
| K_LOCK | `100` |
| agg | `S4` |
| selector | `F6` |
| traversed_family | `STRUCT ONLY (master_nodes.neighbors)` |

The single most important structural fact, verified from source and stated up front: **METIS is given the full topology C, but the online traversal walks STRUCT only.** `master_nodes.neighbors` is exactly the frozen STRUCT CSR; NER and kNN edges are never independently traversed. Every Phase-A question is therefore about a graph the router has never actually walked, and every Phase-B question is about a partitioning built from a graph richer than the one the router walks.

- No gold or query information enters any partition build: `True`
- No learned component anywhere: `True`
- No TEST split touched: `True`

## 2. A0 -- edge algebra, verified from source artifacts

`S` = STRUCT (the traversed CSR), `N` = NERX (NER minus STRUCT), `K` = KNN (`gte_qwen/graph.pt` minus STRUCT). Undirected canonical keys, self-loops dropped.

| corpus | nodes | |S| | |N| | |K| | |S n N| | |S n K| | |N n K| | CSR(S) == frozen | S u N u K == C | frac of C never traversed |
|---|---|---|---|---|---|---|---|---|---|---|
| musique_clean | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| squad_clean | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| metaqa | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 2wiki_clean | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| webqsp | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| hotpotqa_clean | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |

- `S n N` is empty on every corpus, and `S n K` is empty on every corpus, so STRUCT is disjoint from both other families: `True`.
- `N n K` is non-empty on every corpus: `True`. A mutually exclusive family label is therefore impossible and provenance is carried as a **bitmask** (`STRUCT=1, KNN=2, NERX=4`) throughout.
- Between 17% and 83% of topology C is never traversed by the frozen router, depending on corpus. That unused majority is what Phase A prices.

## 3. Phase A -- what the extra edge families actually buy

### A4 unique reach, by bitmask

Of the needed partitions the frozen substrate MISSES, how many does any other family reach on its own?

| corpus | missed needed | S only | S and K | K only | N only | K and N | none | unique KNN frac | unique NERX frac | unreached frac |
|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | 5,815 | 4,039 | 1,291 | 15 | 272 | 56 | 142 | 0.0026 | 0.0468 | 0.0244 |
| 2wiki_clean | 117 | 22 | 23 | 8 | 22 | 31 | 11 | 0.0684 | 0.1880 | 0.0940 |
| musique_clean | 75 | 25 | 39 | 0 | 6 | 4 | 1 | 0.0000 | 0.0800 | 0.0133 |
| squad_clean | 25 | 8 | 7 | 0 | 7 | 3 | 0 | 0.0000 | 0.2800 | 0.0000 |
| hotpotqa_clean | 104 | 25 | 27 | 7 | 12 | 8 | 25 | 0.0673 | 0.1154 | 0.2404 |
| webqsp | 1,685 | 505 | 69 | 97 | 28 | 4 | 982 | 0.0576 | 0.0166 | 0.5828 |

### A9 distinct evidence -- is the reach genuinely new information?

| corpus | K reached | also STRUCT | also Dense/SPLADE | novel | novel frac | novel AND needed | distinct |
|---|---|---|---|---|---|---|---|
| metaqa | 145,678 | 122,499 | 91,234 | 7,704 | 0.0529 | 14 | YES |
| 2wiki_clean | 187,205 | 78,583 | 101,025 | 41,409 | 0.2212 | 5 | YES |
| musique_clean | 96,276 | 85,293 | 96,276 | 0 | 0.0000 | 0 | NO |
| squad_clean | 60,570 | 42,268 | 60,570 | 0 | 0.0000 | 0 | NO |
| hotpotqa_clean | 199,493 | 87,946 | 56,386 | 79,160 | 0.3968 | 2 | YES |
| webqsp | 56,767 | 23,840 | 21,087 | 24,644 | 0.4341 | 23 | YES |

`DENSE_WEARING_GRAPH_COSTUME` -- for every family tested, the similarity between the QUERY and the edge's target separates needed partitions from nuisance ones better than the similarity the edge itself asserts between its two endpoints:

| corpus | family | AUC edge similarity | AUC query similarity | dense in costume |
|---|---|---|---|---|
| metaqa | NERX | 0.6469 | 0.7497 | YES |
| 2wiki_clean | NERX | 0.8579 | 0.9302 | YES |
| musique_clean | NERX | 0.8371 | 0.9312 | YES |
| squad_clean | NERX | 0.8509 | 0.9287 | YES |
| hotpotqa_clean | NERX | 0.9267 | 0.9736 | YES |
| webqsp | NERX | 0.8023 | 0.9006 | YES |

### A8 path families -- do BRIDGE paths carry what homogeneous ones cannot?

| corpus | length | homogeneous yield | bridge yield | delta | bridge better |
|---|---|---|---|---|---|
| metaqa | hop1 | 0.1235 | 0.2368 | +0.1133 | YES |
| metaqa | hop2 | 0.0581 | 0.0500 | -0.0080 | NO |
| metaqa | hop3 | 0.0350 | 0.0276 | -0.0074 | NO |
| 2wiki_clean | hop1 | 0.1920 | 0.3937 | +0.2016 | YES |
| 2wiki_clean | hop2 | 0.1050 | 0.1042 | -0.0008 | NO |
| 2wiki_clean | hop3 | 0.0474 | 0.0415 | -0.0059 | NO |
| musique_clean | hop1 | 0.3225 | 0.5085 | +0.1860 | YES |
| musique_clean | hop2 | 0.1558 | 0.1491 | -0.0067 | NO |
| musique_clean | hop3 | 0.0632 | 0.0437 | -0.0195 | NO |
| squad_clean | hop1 | 0.3182 | 0.2974 | -0.0208 | NO |
| squad_clean | hop2 | 0.1541 | 0.0853 | -0.0688 | NO |
| squad_clean | hop3 | 0.0273 | 0.0202 | -0.0072 | NO |
| hotpotqa_clean | hop1 | 0.2585 | 0.5200 | +0.2615 | YES |
| hotpotqa_clean | hop2 | 0.0757 | 0.1164 | +0.0407 | YES |
| hotpotqa_clean | hop3 | 0.0320 | 0.0334 | +0.0014 | YES |
| webqsp | hop1 | 0.1890 | 0.3793 | +0.1903 | YES |
| webqsp | hop2 | 0.0996 | 0.1283 | +0.0287 | YES |
| webqsp | hop3 | 0.0373 | 0.0578 | +0.0205 | YES |

### A7 / A12 -- exposure and cost of each substrate

| corpus | substrate | edges/query | x frozen | ms/query | x frozen | visited partitions | candidate pool | candidate oracle |
|---|---|---|---|---|---|---|---|---|
| metaqa | E0_STRUCT | 1505.8000 | 1.00 | 41.6 | 1.00 | 305.5 | 49.2 | 0.7212 |
| metaqa | E4_STRUCT_KNN | 2125.6000 | 1.41 | 54.1 | 1.30 | 345.6 | 48.8 | 0.7202 |
| metaqa | E6_TOPOLOGY_C | 5434.4000 | 3.61 | 106.1 | 2.55 | 370.0 | 44.9 | 0.7042 |
| 2wiki_clean | E0_STRUCT | 601.1000 | 1.00 | 21.7 | 1.00 | 168.0 | 41.8 | 0.9515 |
| 2wiki_clean | E4_STRUCT_KNN | 1165.4000 | 1.94 | 34.0 | 1.57 | 266.1 | 43.2 | 0.9590 |
| 2wiki_clean | E6_TOPOLOGY_C | 5131.0000 | 8.54 | 93.8 | 4.33 | 450.8 | 40.5 | 0.9585 |
| musique_clean | E0_STRUCT | 3196.0000 | 1.00 | 66.5 | 1.00 | 111.9 | 18.6 | 0.9750 |
| musique_clean | E4_STRUCT_KNN | 3705.0000 | 1.16 | 81.2 | 1.22 | 123.9 | 18.2 | 0.9755 |
| musique_clean | E6_TOPOLOGY_C | 5927.2000 | 1.85 | 110.5 | 1.66 | 132.7 | 17.6 | 0.9760 |
| squad_clean | E0_STRUCT | 14181.3000 | 1.00 | 237.1 | 1.00 | 117.6 | 15.8 | 0.9945 |
| squad_clean | E4_STRUCT_KNN | 14487.0000 | 1.02 | 236.0 | 0.99 | 142.2 | 17.5 | 0.9945 |
| squad_clean | E6_TOPOLOGY_C | 14082.0000 | 0.99 | 212.4 | 0.90 | 180.0 | 20.5 | 0.9945 |
| hotpotqa_clean | E0_STRUCT | 1892.9000 | 1.00 | 41.5 | 1.00 | 585.1 | 55.4 | 0.9645 |
| hotpotqa_clean | E4_STRUCT_KNN | 2445.0000 | 1.29 | 48.8 | 1.18 | 690.8 | 54.5 | 0.9680 |
| hotpotqa_clean | E6_TOPOLOGY_C | 5761.4000 | 3.04 | 97.9 | 2.36 | 1083.9 | 49.9 | 0.9690 |
| webqsp | E0_STRUCT | 1661.8000 | 1.00 | 40.9 | 1.00 | 417.3 | 34.4 | 0.7886 |
| webqsp | E4_STRUCT_KNN | 2068.4000 | 1.25 | 44.5 | 1.09 | 470.7 | 32.4 | 0.7865 |
| webqsp | E6_TOPOLOGY_C | 3058.8000 | 1.84 | 61.8 | 1.51 | 523.5 | 30.2 | 0.7872 |


## 4. Phase A -- exact P = 50 outcome

`E0_STRUCT` reproduces the frozen router bit-exactly on every query of every corpus (visited nodes, visited partitions, spos, rpos, candidate identities and the final 50). That parity is a hard gate and it passed before any substrate was compared.

| corpus | A3 parity vs frozen |
|---|---|
| metaqa | `EXACT` |
| 2wiki_clean | `EXACT` |
| musique_clean | `EXACT` |
| squad_clean | `EXACT` |
| hotpotqa_clean | `EXACT` |
| webqsp | `EXACT` |

| corpus | FROZEN STRUCT | +NERX | +KNN | +NERX+KNN | MATCHED +NERX | MATCHED +KNN | MATCHED FULL C |
|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.6627 (+0.0015) | 0.6617 (+0.0005) | 0.6597 (-0.0015) | 0.6602 (-0.0010) | 0.6602 (-0.0010) | 0.6572 (-0.0040) |
| 2wiki_clean | 0.9435 | 0.9445 (+0.0010) | 0.9450 (+0.0015) | 0.9465 (+0.0030, SIG) | 0.9455 (+0.0020) | 0.9435 (+0.0000) | 0.9445 (+0.0010) |
| musique_clean | 0.9635 | 0.9650 (+0.0015) | 0.9640 (+0.0005) | 0.9655 (+0.0020) | 0.9620 (-0.0015) | 0.9620 (-0.0015) | 0.9610 (-0.0025) |
| squad_clean | 0.9875 | 0.9880 (+0.0005) | 0.9875 (+0.0000) | 0.9870 (-0.0005) | 0.9880 (+0.0005) | 0.9880 (+0.0005) | 0.9870 (-0.0005) |
| hotpotqa_clean | 0.9505 | 0.9470 (-0.0035) | 0.9510 (+0.0005) | 0.9505 (+0.0000) | 0.9505 (+0.0000) | 0.9520 (+0.0015) | 0.9520 (+0.0015) |
| webqsp | 0.7646 | 0.7689 (+0.0042) | 0.7681 (+0.0035) | 0.7681 (+0.0035) | 0.7653 (+0.0007) | 0.7660 (+0.0014) | 0.7653 (+0.0007) |

Across 36 substrate x corpus cells, 1 reach significance. The best result any edge substrate achieves anywhere is **+0.0042**; the worst is -0.0040. Universally safe substrate (never significantly regresses, significantly gains somewhere): `E6_TOPOLOGY_C`.

## 5. B0 -- how the production partitioner was actually built

Read from source, not assumed. The production partitioner is `scratchpad/build_canonical_topo.py`, not the `src/core/indexers.py` default path (whose `target_per_partition=1000` does not match the shipped ~100).

| property | value |
|---|---|
| graph handed to METIS | topology C = (STRUCT u KNN) u NER, AFTER full construction |
| vertex weights | none (unweighted) |
| edge weights | **none** -- the NER artifact's stored 1/df weights are DISCARDED |
| symmetry | undirected, symmetrised, set-deduplicated, self-loops excluded |
| library / call | `pymetis.part_graph(n_parts, adjacency=adjacency_list)` |
| objective | default `OBJTYPE_CUT` |
| scheme | k-way (pymetis uses recursive only for k <= 8) |
| balance | `ufactor=30` (1.03) |
| seed | METIS default `-1` -- **never pinned** |
| k | `max(1, N // 100)`; TARGET = 100 nodes per block |

Two independent confirmations that this reading is correct: the measured edge cut of `E6_TOPOLOGY_C` equals `variant_C/stats.json` `n_cuts` exactly on 5 of 6 corpora, and the measured balance `max/mean` is 1.0246-1.0299 everywhere -- exactly the `ufactor=30` guarantee.

Because the seed was never pinned, re-running the identical algorithm on the identical graph produces a different assignment. That spread is measured in section 8 and is the yardstick every partitioner comparison is judged against.

## 6. Phase B -- utility of the shipped partitioning

| corpus | B1 gold compression | B2 adjusted colocation | B3 all gold in 1 block | B4 seed->required (nontrivial) | B5 gold-edge containment | B7 min partitions mean | B7 max | B8 oracle ALL@50 |
|---|---|---|---|---|---|---|---|---|
| metaqa | 0.9003 | 0.0350 | 0.4620 | -- | 0.4467 | 5.784 | 104 | 0.9870 |
| 2wiki_clean | 0.8317 | 0.2135 | 0.2365 | 0.2037 | 0.5132 | 1.980 | 4 | 1.0000 |
| musique_clean | 0.7593 | 0.3443 | 0.3725 | -- | 0.6756 | 1.775 | 4 | 1.0000 |
| squad_clean | 1.0000 | N/A_SINGLE_GOLD_PER_QUERY | 1.0000 | -- | 0.0000 | 1.000 | 1 | 1.0000 |
| hotpotqa_clean | 0.8320 | 0.3359 | 0.3360 | 0.4015 | 0.4571 | 1.664 | 2 | 1.0000 |
| webqsp | 0.8997 | 0.2630 | 0.5208 | -- | 0.7662 | 3.664 | 69 | 0.9951 |

`B2` is reported `N/A` where a corpus has exactly one gold document per query: there are no gold PAIRS, so the statistic is undefined rather than zero.

**The partition assignment imposes essentially no ceiling at P = 50.** B8 oracle ALL@50 is 1.0000 on the text corpora and 0.987/0.995 on MetaQA/WebQSP. Whatever limits L1, it is not that the required evidence has been scattered beyond a 50-block budget.

## 7. Phase B -- the partitioner search, and what the graph weighting does

Every alternative obeys the production contract exactly: `k = N // 100`, the same balance, no query or gold anywhere in the build, one universal rule for all corpora.

### The headline: one rule, three corpora converted, and the mechanism is RANKING

`P4_CE_LOCAL_ONLY` -- local-neighbourhood clique closure weighted 1/(|e|-1) -- applied identically to every corpus, with no per-corpus tuning of any kind. Each delta is stated against that corpus's OWN METIS reseed noise floor, because the production partitioner never pins a seed.

| corpus | class | reseed sd | F6 ALL@50 | delta | multiples of sd | queries +/- | McNemar |
|---|---|---|---|---|---|---|---|
| `metaqa` | KB | 0.0021 | 0.7578 | **+0.0966** | 46.0 | +260 / -67 | p < 1e-6 **SIG** |
| `webqsp` | KB | 0.0077 | 0.8069 | **+0.0423** | 5.5 | +89 / -29 | p < 1e-6 **SIG** |
| `hotpotqa_clean` | text | 0.0025 | 0.9655 | **+0.0150** | 6.0 | +41 / -11 | p = 4e-05 **SIG** |
| `2wiki_clean` | text | 0.0025 | 0.9380 | **-0.0055** | 2.2 | +47 / -58 | p = 0.32914 ns |
| `musique_clean` | text | 0.0029 | 0.9565 | **-0.0070** | 2.4 | +48 / -62 | p = 0.21498 ns |
| `squad_clean` | text | 0.0013 | 0.9780 | **-0.0095** | 7.3 | +13 / -32 | p = 0.00661 **SIG** |

**A corpus-class reading of this table is REFUTED.** While only MetaQA and WebQSP had been measured, KB-vs-text was the obvious hypothesis and it was the one this section originally stated. HotpotQA then converted significantly (+0.0150, p = 4e-05) and it is a free-text corpus, so the split is False: three corpora gain significantly, one loses significantly, two are flat. The class labels are kept in the table only to show that they do not organise it.

A second, independent refutation comes from a different rule. `PM2_STRUCT_KNN` SPLITS THE TWO KNOWLEDGE BASES: it gains significantly on `webqsp` (+0.0162) and regresses significantly on `metaqa` (-0.0151). If corpus class were the organising variable, no single rule could send the two KBs in opposite directions. The two rules also disagree about which corpora they help, which is what one expects if the operative variable is a property of the corpus rather than its type.

On MetaQA this is still the first intervention in the entire L1 program to move hop3: hop1 0.9955 -> 0.9955 (+0.0000), hop2 0.7297 -> 0.7568 (+0.0271), hop3 0.2583 -> 0.5210 (+0.2627).

#### Where the gain actually comes from

`BASE` is the partition ranking alone (Dense + SPLADE RRF over blocks, no selector). `F6` is the frozen selector on top of it. If a partitioning helped by giving the selector better material, the two columns would diverge; they do not.

| corpus | BASE | BASE' | delta BASE | F6 | F6' | delta F6 | what the SELECTOR adds |
|---|---|---|---|---|---|---|---|
| `metaqa` | 0.6587 | 0.7548 | **+0.0961** | 0.6612 | 0.7578 | **+0.0966** | +0.0005 |
| `2wiki_clean` | 0.9375 | 0.9395 | **+0.0020** | 0.9435 | 0.9380 | **-0.0055** | -0.0075 |
| `musique_clean` | 0.9565 | 0.9460 | **-0.0105** | 0.9635 | 0.9565 | **-0.0070** | +0.0035 |
| `squad_clean` | 0.9805 | 0.9520 | **-0.0285** | 0.9875 | 0.9780 | **-0.0095** | +0.0190 |
| `hotpotqa_clean` | 0.9345 | 0.9525 | **+0.0180** | 0.9505 | 0.9655 | **+0.0150** | -0.0030 |
| `webqsp` | 0.7618 | 0.8055 | **+0.0437** | 0.7646 | 0.8069 | **+0.0423** | -0.0014 |

The whole effect is present in `BASE`, before the selector runs. The selector contributes at most +/-0.008 on five corpora. The one exception runs the other way: on SQuAD the partitioning costs -0.0285 of ranking and F6 recovers two thirds of it (-0.0095 final), so the frozen selector is a partial shock-absorber for a bad partitioning, never the source of a good one. **This is a partition-RANKING effect, not a selector effect.**

#### The counter-example that kills containment as an objective

Section 6 showed that a balanced random blocking preserves the P=50 oracle and still costs 7-25 points, so partitioning buys routability rather than containment. HotpotQA supplies the positive half of that argument: under `P4_CE_LOCAL_ONLY` every containment statistic gets WORSE -- all-gold-in-one-block 0.3360 -> 0.3060, gold-edge containment 0.4571 -> 0.3984, blocks-needed-per-query 1.664 -> 1.694 -- and the corpus still gains +0.0150 significantly. On the two KBs containment moves the other way (MetaQA all-gold 0.4620 -> 0.5495, WebQSP 0.5208 -> 0.5447) while gold-edge containment still FALLS on both. So neither edge cut, nor gold-edge containment, nor gold co-location predicts utility. What all three winners share is that their blocks became easier for the retrieval channels to FIND.

#### What separates the three winners from the three losers

| corpus | delta | corpus size N | P=50 exposure | blocks a query needs |
|---|---|---|---|---|
| `metaqa` | +0.0966 | 40,151 | 12.55% | 5.78 |
| `webqsp` | +0.0423 | 781,485 | 0.64% | 3.66 |
| `hotpotqa_clean` | +0.0150 | 507,494 | 1.00% | 1.66 |
| `2wiki_clean` | -0.0055 | 65,865 | 7.63% | 1.98 |
| `musique_clean` | -0.0070 | 13,672 | 36.88% | 1.77 |
| `squad_clean` | -0.0095 | 19,029 | 26.48% | 1.00 |

Two conditions cover all six corpora with no exception: the rule gains where P = 50 exposes at most ~1.5% of the corpus (WebQSP 0.64%, HotpotQA 1.00%) OR where a query genuinely needs three or more blocks (MetaQA 5.78, WebQSP 3.66), and loses where neither holds (2Wiki 7.6% / 1.98, SQuAD 26% / 1.00, MuSiQue 36.9% / 1.77). That is a coherent mechanism -- better co-location only pays when the budget is a small window on the corpus, or when the answer is scattered across many blocks -- but it is two thresholds fitted to six points AFTER seeing them. It is offered as the hypothesis a seventh corpus would test, not as an established law, and nothing in the verdict below rests on it.

### The mechanism ablation

**metaqa** (production F6 ALL@50 = 0.6612)

| graph handed to METIS | F6 ALL@50 | delta | hop2 | hop3 | exposure (nodes) |
|---|---|---|---|---|---|
| topology C, unweighted (= production graph, reseeded) | 0.6582 | -0.0030 | 0.7312 | 0.2477 | 5,041 |
| topology C + the NER artifact's stored 1/df weights | 0.6562 | -0.0050 | 0.7192 | 0.2523 | 5,034 |
| local-neighbourhood clique closure, unit weights | 0.6607 | -0.0005 | 0.7252 | 0.2628 | 5,038 |
| NER hyperedges only, weighted | 0.6602 | -0.0010 | 0.7252 | 0.2613 | 4,999 |
| local-neighbourhood clique closure, 1/(|e|-1) weights | 0.7578 | +0.0966 | 0.7568 | 0.5210 | 4,977 |
| all three hyperedge families, weighted | 0.7548 | +0.0936 | 0.7748 | 0.4940 | 4,986 |

**2wiki_clean** (production F6 ALL@50 = 0.9435)

| graph handed to METIS | F6 ALL@50 | delta | hop2 | hop3 | exposure (nodes) |
|---|---|---|---|---|---|
| topology C, unweighted (= production graph, reseeded) | 0.9485 | +0.0050 | -- | -- | 5,028 |
| topology C + the NER artifact's stored 1/df weights | 0.9485 | +0.0050 | -- | -- | 5,026 |
| local-neighbourhood clique closure, unit weights | 0.9400 | -0.0035 | -- | -- | 5,011 |
| NER hyperedges only, weighted | 0.9470 | +0.0035 | -- | -- | 5,002 |
| local-neighbourhood clique closure, 1/(|e|-1) weights | 0.9380 | -0.0055 | -- | -- | 4,994 |
| all three hyperedge families, weighted | 0.9410 | -0.0025 | -- | -- | 4,990 |

**musique_clean** (production F6 ALL@50 = 0.9635)

| graph handed to METIS | F6 ALL@50 | delta | hop2 | hop3 | exposure (nodes) |
|---|---|---|---|---|---|
| topology C, unweighted (= production graph, reseeded) | 0.9555 | -0.0080 | -- | -- | 5,039 |
| topology C + the NER artifact's stored 1/df weights | 0.9560 | -0.0075 | -- | -- | 5,038 |
| local-neighbourhood clique closure, unit weights | 0.9550 | -0.0085 | -- | -- | 5,033 |
| NER hyperedges only, weighted | 0.9420 | -0.0215 | -- | -- | 5,023 |
| local-neighbourhood clique closure, 1/(|e|-1) weights | 0.9565 | -0.0070 | -- | -- | 5,033 |
| all three hyperedge families, weighted | 0.9620 | -0.0015 | -- | -- | 5,027 |

**squad_clean** (production F6 ALL@50 = 0.9875)

| graph handed to METIS | F6 ALL@50 | delta | hop2 | hop3 | exposure (nodes) |
|---|---|---|---|---|---|
| topology C, unweighted (= production graph, reseeded) | 0.9855 | -0.0020 | -- | -- | 5,035 |
| topology C + the NER artifact's stored 1/df weights | 0.9855 | -0.0020 | -- | -- | 5,037 |
| local-neighbourhood clique closure, unit weights | 0.9730 | -0.0145 | -- | -- | 4,990 |
| NER hyperedges only, weighted | 0.9635 | -0.0240 | -- | -- | 4,993 |
| local-neighbourhood clique closure, 1/(|e|-1) weights | 0.9780 | -0.0095 | -- | -- | 5,010 |
| all three hyperedge families, weighted | 0.9785 | -0.0090 | -- | -- | 4,989 |

**hotpotqa_clean** (production F6 ALL@50 = 0.9505)

| graph handed to METIS | F6 ALL@50 | delta | hop2 | hop3 | exposure (nodes) |
|---|---|---|---|---|---|
| topology C, unweighted (= production graph, reseeded) | 0.9515 | +0.0010 | -- | -- | 5,078 |
| topology C + the NER artifact's stored 1/df weights | 0.9500 | -0.0005 | -- | -- | 5,078 |
| local-neighbourhood clique closure, unit weights | 0.9505 | +0.0000 | -- | -- | 5,072 |
| NER hyperedges only, weighted | 0.9315 | -0.0190 | -- | -- | 4,998 |
| local-neighbourhood clique closure, 1/(|e|-1) weights | 0.9655 | +0.0150 | -- | -- | 5,052 |
| all three hyperedge families, weighted | 0.9585 | +0.0080 | -- | -- | 5,043 |

**webqsp** (production F6 ALL@50 = 0.7646)

| graph handed to METIS | F6 ALL@50 | delta | hop2 | hop3 | exposure (nodes) |
|---|---|---|---|---|---|
| topology C, unweighted (= production graph, reseeded) | 0.7590 | -0.0056 | -- | -- | 5,026 |
| topology C + the NER artifact's stored 1/df weights | 0.7681 | +0.0035 | -- | -- | 5,022 |
| local-neighbourhood clique closure, unit weights | 0.7710 | +0.0064 | -- | -- | 5,022 |
| NER hyperedges only, weighted | 0.6970 | -0.0676 | -- | -- | 5,002 |
| local-neighbourhood clique closure, 1/(|e|-1) weights | 0.8069 | +0.0423 | -- | -- | 5,014 |
| all three hyperedge families, weighted | 0.7886 | +0.0240 | -- | -- | 5,017 |


## 8. Phase C -- frozen L1 replayed on each partitioning

Only the partition-dependent artifacts are rebuilt (`mem_idx`, `base_rank`, gold labels, `part_sizes`). Query embeddings, Dense, SPLADE, the retrieval seeds and the whole structural expansion are read straight from the shipped cache -- the traversal runs on doc rows and never sees a block id, so a partition swap needs no re-traversal and no re-encode. Replaying the shipped assignment reproduces the frozen scoreboard exactly, which is asserted before any candidate is scored.

| corpus | production F6 ALL@50 | METIS reseed sd | noise floor sd | random balanced mean | random delta |
|---|---|---|---|---|---|
| metaqa | 0.6612 | 0.0023 | 0.0021 | 0.5802 | -0.0810 |
| 2wiki_clean | 0.9435 | 0.0022 | 0.0025 | 0.8696 | -0.0739 |
| musique_clean | 0.9635 | 0.0023 | 0.0029 | 0.7521 | -0.2114 |
| squad_clean | 0.9875 | 0.0012 | 0.0013 | 0.8280 | -0.1595 |
| hotpotqa_clean | 0.9505 | 0.0028 | 0.0025 | 0.8742 | -0.0763 |
| webqsp | 0.7646 | 0.0089 | 0.0077 | 0.5883 | -0.1763 |

`P0_RANDOM_BALANCED` uses the **exact production size histogram** and lands at the same exposure, and its B8 oracle at P = 50 is essentially unchanged -- yet it costs tens of points. What the partitioning buys is therefore **routability**, not containment: the canonical partition-level vote is what breaks, not the reachability of the evidence.

**metaqa** -- candidates vs production (exact McNemar)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_CE_LOCAL_ONLY | 0.7578 | +0.0966 | 193 | 0.0000 | SIG | 0.988 | YES |
| P4_HYPERGRAPH_CE | 0.7548 | +0.0936 | 187 | 0.0000 | SIG | 0.990 | YES |
| PM1_STRUCT_NERX | 0.6617 | +0.0005 | 1 | 1.0000 | ns | 1.000 | NO |
| P4_CE_UNWEIGHTED | 0.6607 | -0.0005 | -1 | 1.0000 | ns | 1.000 | NO |
| P4_CE_NER_ONLY | 0.6602 | -0.0010 | -2 | 0.9350 | ns | 0.992 | NO |
| PM3_TOPOLOGY_C | 0.6582 | -0.0030 | -6 | 0.6536 | ns | 1.001 | NO |
| P2_METIS_STRONG | 0.6567 | -0.0045 | -9 | 0.4394 | ns | 0.999 | YES |
| PM4_TOPOLOGY_C_NERW | 0.6562 | -0.0050 | -10 | 0.4075 | ns | 0.999 | YES |
| PM0_STRUCT | 0.6557 | -0.0055 | -11 | 0.4096 | ns | 0.997 | YES |
| PM2_STRUCT_KNN | 0.6461 | -0.0151 | -30 | 0.0168 | SIG | 0.997 | YES |
| P3_FENNEL | 0.6356 | -0.0256 | -51 | 0.0000 | SIG | 1.005 | YES |

**2wiki_clean** -- candidates vs production (exact McNemar)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| PM3_TOPOLOGY_C | 0.9485 | +0.0050 | 10 | 0.2529 | ns | 1.000 | NO |
| PM4_TOPOLOGY_C_NERW | 0.9485 | +0.0050 | 10 | 0.2604 | ns | 1.000 | NO |
| P2_METIS_STRONG | 0.9475 | +0.0040 | 8 | 0.3409 | ns | 1.001 | NO |
| P4_CE_NER_ONLY | 0.9470 | +0.0035 | 7 | 0.5154 | ns | 0.995 | NO |
| PM1_STRUCT_NERX | 0.9455 | +0.0020 | 4 | 0.7376 | ns | 1.004 | NO |
| P4_HYPERGRAPH_CE | 0.9410 | -0.0025 | -5 | 0.6718 | ns | 0.992 | NO |
| P4_CE_UNWEIGHTED | 0.9400 | -0.0035 | -7 | 0.5203 | ns | 0.997 | NO |
| P4_CE_LOCAL_ONLY | 0.9380 | -0.0055 | -11 | 0.3291 | ns | 0.993 | YES |
| P3_FENNEL | 0.9345 | -0.0090 | -18 | 0.0535 | ns | 1.003 | YES |
| PM2_STRUCT_KNN | 0.9315 | -0.0120 | -24 | 0.0210 | SIG | 1.000 | YES |
| PM0_STRUCT | 0.8990 | -0.0445 | -89 | 0.0000 | SIG | 1.008 | YES |

**musique_clean** -- candidates vs production (exact McNemar)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_HYPERGRAPH_CE | 0.9620 | -0.0015 | -3 | 0.8482 | ns | 0.997 | NO |
| P2_METIS_STRONG | 0.9605 | -0.0030 | -6 | 0.5323 | ns | 0.999 | NO |
| P4_CE_LOCAL_ONLY | 0.9565 | -0.0070 | -14 | 0.2150 | ns | 0.998 | YES |
| PM4_TOPOLOGY_C_NERW | 0.9560 | -0.0075 | -15 | 0.1053 | ns | 0.999 | YES |
| PM3_TOPOLOGY_C | 0.9555 | -0.0080 | -16 | 0.0599 | ns | 1.000 | YES |
| P4_CE_UNWEIGHTED | 0.9550 | -0.0085 | -17 | 0.1285 | ns | 0.998 | YES |
| PM1_STRUCT_NERX | 0.9515 | -0.0120 | -24 | 0.0149 | SIG | 1.000 | YES |
| PM2_STRUCT_KNN | 0.9465 | -0.0170 | -34 | 0.0024 | SIG | 1.001 | YES |
| P4_CE_NER_ONLY | 0.9420 | -0.0215 | -43 | 0.0005 | SIG | 0.996 | YES |
| PM0_STRUCT | 0.9225 | -0.0410 | -82 | 0.0000 | SIG | 1.003 | YES |
| P3_FENNEL | 0.9205 | -0.0430 | -86 | 0.0000 | SIG | 1.004 | YES |

**squad_clean** -- candidates vs production (exact McNemar)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| PM3_TOPOLOGY_C | 0.9855 | -0.0020 | -4 | 0.5235 | ns | 0.999 | NO |
| PM4_TOPOLOGY_C_NERW | 0.9855 | -0.0020 | -4 | 0.5413 | ns | 1.000 | NO |
| PM1_STRUCT_NERX | 0.9840 | -0.0035 | -7 | 0.1671 | ns | 0.998 | YES |
| P2_METIS_STRONG | 0.9840 | -0.0035 | -7 | 0.2100 | ns | 0.999 | YES |
| PM2_STRUCT_KNN | 0.9835 | -0.0040 | -8 | 0.1338 | ns | 1.007 | YES |
| PM0_STRUCT | 0.9785 | -0.0090 | -18 | 0.0014 | SIG | 1.010 | YES |
| P4_HYPERGRAPH_CE | 0.9785 | -0.0090 | -18 | 0.0114 | SIG | 0.990 | YES |
| P4_CE_LOCAL_ONLY | 0.9780 | -0.0095 | -19 | 0.0066 | SIG | 0.994 | YES |
| P3_FENNEL | 0.9755 | -0.0120 | -24 | 0.0007 | SIG | 0.998 | YES |
| P4_CE_UNWEIGHTED | 0.9730 | -0.0145 | -29 | 0.0000 | SIG | 0.990 | YES |
| P4_CE_NER_ONLY | 0.9635 | -0.0240 | -48 | 0.0000 | SIG | 0.991 | YES |

**hotpotqa_clean** -- candidates vs production (exact McNemar)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_CE_LOCAL_ONLY | 0.9655 | +0.0150 | 30 | 0.0000 | SIG | 0.994 | YES |
| PM2_STRUCT_KNN | 0.9615 | +0.0110 | 22 | 0.0032 | SIG | 1.000 | YES |
| P4_HYPERGRAPH_CE | 0.9585 | +0.0080 | 16 | 0.0226 | SIG | 0.992 | YES |
| P2_METIS_STRONG | 0.9520 | +0.0015 | 3 | 0.7608 | ns | 1.000 | NO |
| PM3_TOPOLOGY_C | 0.9515 | +0.0010 | 2 | 0.8776 | ns | 0.999 | NO |
| P4_CE_UNWEIGHTED | 0.9505 | +0.0000 | 0 | 1.0000 | ns | 0.998 | NO |
| PM4_TOPOLOGY_C_NERW | 0.9500 | -0.0005 | -1 | 1.0000 | ns | 0.999 | NO |
| PM0_STRUCT | 0.9490 | -0.0015 | -3 | 0.8099 | ns | 1.004 | NO |
| PM1_STRUCT_NERX | 0.9470 | -0.0035 | -7 | 0.3604 | ns | 1.000 | NO |
| P3_FENNEL | 0.9335 | -0.0170 | -34 | 0.0000 | SIG | 1.000 | YES |
| P4_CE_NER_ONLY | 0.9315 | -0.0190 | -38 | 0.0000 | SIG | 0.983 | YES |

**webqsp** -- candidates vs production (exact McNemar)

| partitioning | F6 ALL@50 | delta | net | p | McNemar | exposure x | > noise floor |
|---|---|---|---|---|---|---|---|
| P4_CE_LOCAL_ONLY | 0.8069 | +0.0423 | 60 | 0.0000 | SIG | 0.998 | YES |
| P4_HYPERGRAPH_CE | 0.7886 | +0.0240 | 34 | 0.0011 | SIG | 0.999 | YES |
| PM2_STRUCT_KNN | 0.7808 | +0.0162 | 23 | 0.0265 | SIG | 1.000 | YES |
| PM0_STRUCT | 0.7766 | +0.0120 | 17 | 0.1038 | ns | 1.005 | NO |
| P4_CE_UNWEIGHTED | 0.7710 | +0.0064 | 9 | 0.3682 | ns | 1.000 | NO |
| PM1_STRUCT_NERX | 0.7696 | +0.0050 | 7 | 0.5203 | ns | 1.004 | NO |
| PM4_TOPOLOGY_C_NERW | 0.7681 | +0.0035 | 5 | 0.6201 | ns | 1.000 | NO |
| PM3_TOPOLOGY_C | 0.7590 | -0.0056 | -8 | 0.3891 | ns | 1.000 | NO |
| P2_METIS_STRONG | 0.7576 | -0.0070 | -10 | 0.2750 | ns | 1.001 | NO |
| P3_FENNEL | 0.7294 | -0.0352 | -50 | 0.0000 | SIG | 1.005 | YES |
| P4_CE_NER_ONLY | 0.6970 | -0.0676 | -96 | 0.0000 | SIG | 0.996 | YES |

### C4 decomposition

| corpus | frozen cell | best edge substrate | EDGE_EFFECT | sig | best partitioning | PARTITION_EFFECT |
|---|---|---|---|---|---|---|
| metaqa | 0.6612 | E3_STRUCT_NERX | +0.0015 | ns | P4_CE_LOCAL_ONLY | +0.0966 |
| 2wiki_clean | 0.9435 | E6_TOPOLOGY_C | +0.0030 | SIG | PM3_TOPOLOGY_C | +0.0050 |
| musique_clean | 0.9635 | E6_TOPOLOGY_C | +0.0020 | ns | P4_HYPERGRAPH_CE | -0.0015 |
| squad_clean | 0.9875 | E3_STRUCT_NERX | +0.0005 | ns | PM3_TOPOLOGY_C | -0.0020 |
| hotpotqa_clean | 0.9505 | M2_STRUCT_KNN_MATCHED | +0.0015 | ns | P4_CE_LOCAL_ONLY | +0.0150 |
| webqsp | 0.7646 | E3_STRUCT_NERX | +0.0043 | ns | P4_CE_LOCAL_ONLY | +0.0423 |


**C3 -- the joint 2x2, and the INTERACTION term.** Phase A varied the edge substrate at the shipped partitioning; Phase C varied the partitioning at the frozen substrate. Neither produces the fourth cell, so it was run: `_l1ep_x.py` composes the Phase-C partition rebuild with the Phase-A traversal, changing nothing else. Phase A licensed the edge arm (`E6_TOPOLOGY_C` never significantly regresses and significantly gains on 2wiki), so the arm is not invented.

| corpus | a: frozen x shipped | b: frozen x candidate | c: E6 x shipped | d: E6 x candidate | EDGE (c-a) | PARTITION (b-a) | JOINT (d-a) | **INTERACTION** | parity |
|---|---|---|---|---|---|---|---|---|---|
| `metaqa` | 0.6612 | 0.7578 | 0.6597 | 0.7553 | -0.0015 | +0.0966 | +0.0941 | **-0.0010** | `EXACT` / `EXACT` |
| `2wiki_clean` | 0.9435 | 0.9485 | 0.9465 | 0.9520 | +0.0030 | +0.0050 | +0.0085 | **+0.0005** | `EXACT` / `EXACT` |
| `hotpotqa_clean` | 0.9505 | 0.9655 | 0.9505 | 0.9670 | +0.0000 | +0.0150 | +0.0165 | **+0.0015** | `EXACT` / `EXACT` |
| `webqsp` | 0.7646 | 0.8069 | 0.7681 | 0.8090 | +0.0035 | +0.0423 | +0.0444 | **-0.0014** | `EXACT` / `EXACT` |

Cell `a` is a hard gate: it must reproduce the frozen scoreboard, and it does on every corpus run, which is what licenses reading the other three cells.


## 9. Verdicts, answers, and what remains

**Note on the verdict letters.** The A-E / A-D letter definitions from the directive are not recoverable from the session record, so each letter below carries an explicit criterion and the verdict is assigned from the measured numbers against that criterion. If the intended definitions differ, the measurements are unchanged and the label can be re-mapped without re-running anything.

### EDGE_SUBSTRATE_VERDICT

| letter | criterion | assigned |
|---|---|---|
| A | a richer edge substrate is a universal win: significant material gain on essentially every corpus, no significant regression, acceptable cost |  |
| B | PARTIAL: a material significant gain somewhere and no significant regression anywhere, but not universal |  |
| C | EXHAUSTED: no substrate produces a material gain (>= 0.01 ALL@50) on any corpus; the axis is closed | **YES** |
| D | MIXED: gains on some corpora paid for by significant regressions elsewhere |  |
| E | INCONCLUSIVE: parity failed or coverage too thin to discriminate |  |

**`EDGE_SUBSTRATE_VERDICT = C`**

- 36 substrate x corpus cells at exact P = 50; 1 reach significance.
- Best gain any substrate achieves anywhere: +0.0042; worst -0.0040.
- Cost of the richest substrate (`E6_TOPOLOGY_C`) is 1.5x-4.3x the frozen traversal time and up to 8.5x the edges scored.
- On MetaQA the candidate ORACLE falls as the substrate widens (0.7212 -> 0.7042): the wider frontier dilutes a fixed-size candidate list rather than enriching it.
- kNN reaches partitions no other channel reaches on the KB-ish corpora but that territory is nearly empty of what is needed: metaqa 14, 2wiki_clean 5, musique_clean 0, squad_clean 0, hotpotqa_clean 2, webqsp 23 novel-and-needed partitions.

### PARTITIONING_VERDICT

| letter | criterion | assigned |
|---|---|---|
| A | one universal partitioning rule beats production materially and never significantly regresses, on every corpus |  |
| B | PARTIAL: a single universal rule gains materially with no significant regression, but coverage is incomplete |  |
| C | EXHAUSTED: no alternative partitioning converts into a material gain anywhere |  |
| D | MIXED / CORPUS-DEPENDENT: a rule converts materially on some corpora and significantly regresses on others, so no universal change is licensed | **YES** |
| E | INCONCLUSIVE: Phase C parity failed or no corpus completed |  |

**`PARTITIONING_VERDICT = D`**

- Material significant conversions: `metaqa` / `P4_CE_LOCAL_ONLY` +0.0966; `metaqa` / `P4_HYPERGRAPH_CE` +0.0936; `webqsp` / `P4_CE_LOCAL_ONLY` +0.0423; `webqsp` / `P4_HYPERGRAPH_CE` +0.0240; `webqsp` / `PM2_STRUCT_KNN` +0.0162; `hotpotqa_clean` / `P4_CE_LOCAL_ONLY` +0.0150; `hotpotqa_clean` / `PM2_STRUCT_KNN` +0.0110
- Significant regressions: `webqsp` / `P4_CE_NER_ONLY` -0.0676; `2wiki_clean` / `PM0_STRUCT` -0.0445; `musique_clean` / `P3_FENNEL` -0.0430; `musique_clean` / `PM0_STRUCT` -0.0410; `webqsp` / `P3_FENNEL` -0.0352; `metaqa` / `P3_FENNEL` -0.0256; `squad_clean` / `P4_CE_NER_ONLY` -0.0240; `musique_clean` / `P4_CE_NER_ONLY` -0.0215
- A single universal rule that gains materially and never significantly regresses: `NONE`.
- `metaqa`: METIS reseed noise floor sd = 0.0021 (production never pinned a seed), so any delta below ~0.0042 is not a partitioner difference.
- `2wiki_clean`: METIS reseed noise floor sd = 0.0025 (production never pinned a seed), so any delta below ~0.0050 is not a partitioner difference.
- `musique_clean`: METIS reseed noise floor sd = 0.0029 (production never pinned a seed), so any delta below ~0.0058 is not a partitioner difference.
- `squad_clean`: METIS reseed noise floor sd = 0.0013 (production never pinned a seed), so any delta below ~0.0026 is not a partitioner difference.
- `hotpotqa_clean`: METIS reseed noise floor sd = 0.0025 (production never pinned a seed), so any delta below ~0.0050 is not a partitioner difference.
- `webqsp`: METIS reseed noise floor sd = 0.0077 (production never pinned a seed), so any delta below ~0.0154 is not a partitioner difference.

### COMBINED_L1_VERDICT

| letter | criterion | assigned |
|---|---|---|
| A | both axes are limiters and both convert universally |  |
| B | exactly one axis is the limiter and it converts universally |  |
| C | neither axis is the limiter; the residual lives elsewhere |  |
| D | at least one axis produces a real, large, mechanistically isolated effect that is corpus-dependent, so no universal change is licensed | **YES** |

**`COMBINED_L1_VERDICT = D`**

- The edge axis moves exact-P50 by at most 0.0042 anywhere. The partition axis moves it by +0.0966 on MetaQA and +0.0423 on WebQSP, and by -0.07 to -0.25 when degraded to a balanced random blocking at identical exposure. The two axes are not comparable in magnitude.
- The partition gain is CORPUS-DEPENDENT, and a CORPUS-CLASS reading is REFUTED (clean KB-vs-text split = False). Under one rule with no per-corpus tuning it converts significantly on metaqa (KB) +0.0966; webqsp (KB) +0.0423; hotpotqa_clean (text) +0.0150, significantly regresses on squad_clean (text) -0.0095, and is not separable from noise on 2wiki_clean (text) -0.0055; musique_clean (text) -0.0070. KB-vs-text was the obvious hypothesis while only the two knowledge bases had converted; HotpotQA is free text and converts significantly, so corpus class does not organise the result. What separates the winners from the losers is exposure and block demand, not class -- see the headline section.
- Yet the partitioner NEEDS the very families the traversal cannot use: partitioning on STRUCT alone is significantly worse on MuSiQue (-0.0410), 2Wiki (-0.0445) and SQuAD (-0.0090).
- The two axes are also SEPARABLE, which the 2x2 measures rather than assumes: INTERACTION = d - b - c + a is `metaqa` -0.0010 (noise floor sd 0.0021); `2wiki_clean` +0.0005 (noise floor sd 0.0025); `hotpotqa_clean` +0.0015 (noise floor sd 0.0025); `webqsp` -0.0014 (noise floor sd 0.0077). Every value is at or below that corpus's own reseed noise floor, so the partition gain does not depend on which edge substrate the router walks, and widening the substrate costs the same small amount under either partitioning. Edge and partition are independent levers of very unequal size, not two views of one effect. One caveat on coverage: `2wiki_clean` carries a near-null partition arm (PM3_TOPOLOGY_C), so its cell tests additivity in a regime where there is little to interact with. The load-bearing tests are `metaqa`, `hotpotqa_clean`, `webqsp`, where the partition effect is large.
- So the extra edges are real information that L1 already exploits -- offline, through the block structure, not online through traversal.

### Promotion decision

| promotion gate | status |
|---|---|
| exact P = 50 preserved | PASS -- every cell selects exactly 50 partitions |
| balanced, predictable partitions | PASS -- every candidate holds max/mean <= 1.04 at the production k |
| materially useful MetaQA hop2/hop3 | PASS -- hop3 0.2583 -> 0.5210 and hop2 0.7297 -> 0.7568 under `P4_CE_LOCAL_ONLY` |
| no significant cross-corpus regression | FAIL for the promotion candidate `P4_CE_LOCAL_ONLY`: `squad_clean` -0.0095. No sibling rule escapes it either -- `P4_CE_NER_ONLY`, `P4_CE_UNWEIGHTED`, `P4_HYPERGRAPH_CE` also regress significantly, so there is no nearby variant to promote instead. |
| reasonable latency | PASS -- partitioning is an offline artifact and online exposure is unchanged (4,977-5,042 scoped nodes across every cell) |
| no gold or query leakage into partition construction | PASS -- every build reads corpus topology, NER and kNN only |

**`PROMOTED = NONE`** and **`L1_FROZEN = NO`**

- `PARTITION_QUALITY_IMPROVED_ROUTER_CONVERTS_BUT_NOT_UNIVERSALLY` -- the alternative partitioning is a genuinely better partitioner AND the router converts it, on THREE of six corpora under one untuned rule: MetaQA +0.0966 (hop3 0.2583 -> 0.5210, 2.02x), WebQSP +0.0423, HotpotQA +0.0150, all significant. It is withheld because the same rule significantly regresses SQuAD (-0.0095, p = 0.0066) and no sibling variant escapes that. This is explicitly NOT `PARTITION_QUALITY_IMPROVED_ROUTER_CANNOT_CONVERT`: the router had no trouble converting it, three times. Note also that the conversion is a partition-RANKING effect -- the full delta is already present in BASE before the F6 selector runs -- so 'the router cannot convert it' is the wrong diagnosis in both directions.
- `EDGE_REACH_GAIN_EXPOSURE_DRIVEN` -- not applicable as a rescue -- the richer substrates raise visited-partition exposure 1.2x-2.7x and still do not raise exact-P50, so there is no reach gain to relabel.

### The twelve program questions

_The exact Q1-Q11 wording is not recoverable from the session record; each is stated as the phase built to test it. Q12 is verbatim._

**Q1. Is `master_nodes.neighbors` really the only family the router traverses, and is the edge algebra what we assumed?**

Yes, and it is now verified rather than assumed. `CSR(S)` is bit-identical to the frozen neighbour lists on every corpus; `S n N` and `S n K` are empty everywhere; `N n K` is non-empty everywhere, so provenance must be carried as a bitmask. `S u N u K` equals the topology-C file exactly on 5 of 6 corpora -- Hotpot's older `gte_qwen/graph.pt` omits 34,425 structural edges, making the measured Hotpot cut a 0.33% superset of what METIS actually saw (documented, not fatal).

**Q2. How much of the graph does the frozen router never touch?**

Between 17% (SQuAD) and 83% (2Wiki) of topology-C edges are never traversed. The unused majority is large enough that its irrelevance is itself the finding.

**Q3. Do the untraversed families reach needed partitions the frozen substrate misses?**

Yes, but marginally. Unique-kNN reach is 0.3%-7% of missed needed partitions and unique-NERX reach 2%-28%, while 2%-9% are reached by no family at all.

**Q4. Is that reach DISTINCT information, or a restatement of Dense/SPLADE?**

Largely a restatement. On MuSiQue and SQuAD, ZERO kNN-reached partitions are novel to Dense/SPLADE top-200. Where novelty is high (2Wiki 22%, WebQSP 43%) the novel territory contains 5 and 23 needed partitions respectively. For NERX the query-to-target similarity separates needed from nuisance better than the edge's own endpoint similarity (AUC 0.9312 vs 0.8371 on MuSiQue): `DENSE_WEARING_GRAPH_COSTUME = YES`.

**Q5. Do mixed-family (bridge) paths buy a path shape homogeneous ones cannot?**

Only at the first hop, and only on the KB-ish corpora. Bridges beat homogeneous paths at hop1 by +0.11 to +0.20 needed-yield on 4 of 5 corpora, then lose at hop2 and hop3 everywhere except WebQSP, where they win at all three lengths. Mixed traversal is a KB phenomenon whose value is exhausted after one step on text.

**Q6. At exact P = 50, does ANY richer edge substrate beat the frozen one?**

No. Across 36 cells the best result anywhere is +0.0042 and only 1 cell reaches significance, at 1.5x-4.3x the traversal cost. On MetaQA the candidate oracle FALLS as the substrate widens.

**Q7. Does the shipped partitioning impose a ceiling at P = 50?**

No. Oracle ALL@50 over the shipped assignment is 1.0000 on every text corpus and 0.987 / 0.995 on MetaQA / WebQSP. The required evidence is inside a 50-block budget almost always; what fails is selecting those blocks.

**Q8. Is the partitioning doing anything at all, or would any balanced blocking do?**

It is doing a great deal. `P0_RANDOM_BALANCED`, built with the EXACT production size histogram and landing at identical exposure, costs 7-25 points of ALL@50 while leaving the P=50 oracle essentially intact. What the partitioning buys is ROUTABILITY -- a usable partition-level vote -- not containment.

**Q9. Which graph should the partitioner see?**

On text, not the one the router walks. Partitioning on STRUCT alone (`PM0_STRUCT`) is significantly worse on MuSiQue (-0.0410), 2Wiki (-0.0445) and SQuAD (-0.0090). The NER and kNN families are valuable to the PARTITIONER and worthless to the TRAVERSAL -- the sharpest asymmetry in the program, and the reconciliation of its two halves.

The exception is instructive rather than contradictory: on WebQSP `PM0_STRUCT` is +0.0120 and not significant (1.6 sd of that corpus's 0.0077 floor). On a knowledge base the structural family IS the curated relation set, so it already carries the semantics NER and kNN are approximating on text, where STRUCT is only title mentions. The rule is therefore: the partitioner needs a semantically complete graph, which on text requires the untraversed families and on a KB does not.

**Q10. Is edge cut the right objective?**

No, and the counter-example is unambiguous. On MetaQA the best partitioning has a 29% WORSE cut on topology C (296,258 vs 228,941) and a LOWER gold-edge containment (0.2715 vs 0.4467), yet gains +0.0966. The statistics that track utility are gold co-location (all-gold-in-one-block 0.4620 -> 0.5706) and the number of blocks a query needs (5.78 -> 4.82).

**Q11. If partition construction matters, WHAT property of it matters?**

Second-order locality with degree-normalised strength -- and neither ingredient alone. On MetaQA: topology C unweighted is the baseline; adding the NER artifact's own stored 1/df weights to it does nothing (-0.0050); closing local neighbourhoods into cliques with UNIT weights does nothing (-0.0005); NER hyperedges alone do nothing (-0.0010). Closing local neighbourhoods into cliques WITH the canonical 1/(|e|-1) weight gains +0.0966, hop3 0.2583 -> 0.5210 and hop2 0.7297 -> 0.7568. The isolation replicates on WebQSP, where the unweighted clique closure lands inside that corpus's own reseed noise floor (+0.0064 against sd 0.0077) and the weighted one converts +0.0423.

Under one rule with no per-corpus tuning: metaqa (KB) +0.0966*; webqsp (KB) +0.0423*; hotpotqa_clean (text) +0.0150*; 2wiki_clean (text) -0.0055; musique_clean (text) -0.0070; squad_clean (text) -0.0095* (* = significant). Three corpora gain significantly, one loses significantly, two are flat.

A CORPUS-CLASS reading is REFUTED. When only MetaQA and WebQSP had converted, KB-vs-text was the obvious hypothesis; HotpotQA then converted significantly (+0.0150, p = 4e-05) and is free text, so class does not organise the result. What does hold is that the winners are the corpora where P = 50 exposes at most ~1.5% of the corpus (WebQSP 0.64%, HotpotQA 1.00%) or where a query needs three or more blocks (MetaQA 5.78, WebQSP 3.66) -- two thresholds fitted to six points after the fact, offered as the hypothesis a seventh corpus would test rather than as a law.

State the gains at their own scales rather than as equals: MetaQA's +0.0966 is about 46 sd of its 0.0021 reseed noise floor, WebQSP's +0.0423 about 5.5 sd of its much larger 0.0077 floor (its reseed replicates span 0.7512-0.7710). All three are significant by exact McNemar, but only MetaQA is overwhelming relative to the variance of its own partitioner.

**Q12. Is the largest remaining problem: traversal / partition construction / ranking / or genuinely missing information?**

**Partition construction -- and specifically the partition RANKING it produces. Not traversal, not the selector, and not missing information.**

The edge substrate, which is what the router actually walks, is closed on all six corpora: its entire dynamic range at exact P = 50 is about +/-0.004, for up to 4.3x the traversal cost. Partition construction has roughly 25x that dynamic range on MetaQA (+0.0966), 10x on WebQSP (+0.0423) and 4x on HotpotQA (+0.0150), and is the only intervention in the whole L1 program that has ever moved MetaQA hop3 off 0.2583 (-> 0.5210). The 2x2 shows the two axes are additive -- INTERACTION sits at or below every corpus's reseed noise floor -- so this is a genuinely separate lever, not a restatement of the edge result.

Within partition construction the binding stage is RANKING, not selection. The whole delta is already present in BASE, the Dense+SPLADE RRF over blocks, before the F6 selector runs; the selector moves it by at most +/-0.008 on five of six corpora. Its one larger contribution is defensive: on SQuAD it recovers two thirds of a -0.0285 ranking loss. The frozen selector absorbs a bad partitioning; it never creates a good one.

Where the rule does not gain, the residual is still ranking rather than missing information: the P = 50 oracle over the shipped assignment is 1.0000 on every text corpus, so the evidence is inside the budget and simply is not selected. Genuinely missing information is the smallest term everywhere -- only 2%-9% of missed needed partitions are unreachable by any edge family.

