# L1 — SIMPLIFY, REMOVE MODALITY DOUBLE-COUNTING, TEST PRECOMPUTABLE PPR ROUTING

*G2 / L1_PPR_SIMPLIFY. Six corpora, DEV only. No TEST, no L2, no L3 change, no training, no learned router, no dataset identity, no gold at inference, no new encoder pass. Every method reported here emits EXACTLY 50 canonical MASTER_TOPOLOGY=C partitions.*

---

## VERDICT

```
L1_FROZEN                          = NO
STANDING POLICY                    = B6_S4_F6_Ms64_Mr32   (unchanged, still SAFE_REFERENCE)
DIRECT_TOP50_CAN_REPLACE_SWAP      = NO   (best DIRECT variant L4_DIRECT_PPRBOUNDED:
                                          macro -0.0023 / worst -0.0085 vs swap +0.0069 / +0.0025)
PARTITION_PPR_IS_AN_L1_CHANNEL     = NO   (reaches more gold partitions, converts none)
BOUNDED_PPR_BETTER_THAN_GLOBAL     = YES  (accuracy and latency), but both are refuted
PARTITION_BOUNDED_NODE_PPR_USEFUL  = QUALIFIED YES, NOT WORTH IT  (entry-mode beats its
                                     own NO-GRAPH controls 6/6, but stays below the
                                     standing router and costs the most latency here)
PPR_FULLY_PRECOMPUTABLE            = YES  (exactly, by linearity; online graph edges = 0)
SIMPLEST_STRUCTURAL_SIGNAL         = S4   (the frozen directional expansion)
B_IS_THE_LIMITER                   = NO   (CAP_B6 is 21 of 677 MetaQA failures, 3.1%;
                                          0 on every text corpus)
NEW_ENCODER_PASSES                 = 0
MODAL_REQUIRED                     = NO   (peak RSS 1.6 GB, all six corpora local)
```

This phase proposed five ways to change L1 — replace the swap with DIRECT top-50 fusion, add global / bounded / partition-local PPR, or move B. **None of the five improves L1**, though partition-local PPR is refuted on cost rather than on signal. The sixth question, whether PPR can be precomputed, is answered **yes, exactly** — and is therefore moot, because the signal it would make free is not one worth having.
The standing router survives unchanged, which is a real result: it is now the best of a much
larger tested space, and the reasons the alternatives fail are mechanical rather than statistical.

---

## STEP 0 — FINAL AUDIT OF THE CURRENT SWAPPING

All eight gates pass exactly on all six corpora. `diag/swap_dump.json` holds the full per-query dumps (BASE top-50, protected core, boundary, challengers, per-channel ranks, swapped-in, swapped-out, final top-50) for MetaQA hop2, MetaQA hop3, WebQSP and HotpotQA.

## T1 — STEP 0 verification gates

| corpus | base_rank prov. | ret_rrf prov. | symmetric scoring | deterministic ties | exact P=50 | absent-channel | gold in ranking |
|---|---|---|---|---|---|---|---|
| MetaQA | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| WebQSP | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| 2Wiki | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| MuSiQue | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| HotpotQA | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |
| SQuAD | EXACT | EXACT | EXACT (0/300) | EXACT (0 diffs) | EXACT (0 viol.) | 0 candidates with no channel | none |

`base_rank` is reproduced bit-for-bit by replaying `rrf_partitions([PR_dense, PR_splade])` from the cached dense/SPLADE top-`K_LOCK` node lists, and `ret_rrf` is reproduced bit-for-bit by replaying `node_rrf(dense200, splade200)`. Symmetric scoring is verified by rescoring every candidate with an incumbent-blind expression and confirming it reproduces the selection (0/300 mismatches per corpus); determinism is verified by shuffling the challenger list (0 differences); exact P=50 is verified on every query of every corpus.

### The audit found the mechanism, and it is not a competition

A worked example from the dump (MetaQA hop2, query 666): the gold partition **318** sits at canonical rank 45, inside the boundary, with **no** structural and **no** retrieval evidence. Its score is `1/(60+45) = 0.009524`. Any challenger holding structural rank 1 and nothing else scores `1/(60+1) = 0.016393`. The gold partition is evicted and the query flips from covered to uncovered.

This is arithmetic, not chance: **a canonical-only incumbent anywhere in the boundary loses to any challenger inside the top 45 of the structural channel.** So the boundary competition is near-deterministic turnover of channel-poor incumbents.

## T17 — STEP 0 (quantified) the boundary competition is near-deterministic turnover

| corpus | B | incumbents retained / B | retention rate | boundary incumbents with canonical evidence ONLY (/q) | their retention rate | gold partitions evicted | gained | net |
|---|---|---|---|---|---|---|---|---|
| MetaQA | 6 | 1.341 | 0.2235 | 3.96 | 0.0009 | 183 | 257 | +74 |
| MetaQA | 12 | 3.624 | 0.3020 | 7.767 | 0.0012 | 356 | 438 | +82 |
| WebQSP | 6 | 1.032 | 0.1721 | 5.042 | 0.0280 | 90 | 91 | +1 |
| WebQSP | 12 | 2.484 | 0.2070 | 9.971 | 0.0489 | 196 | 115 | -81 |
| 2Wiki | 6 | 1.741 | 0.2902 | 4.295 | 0.0882 | 6 | 20 | +14 |
| 2Wiki | 12 | 4.424 | 0.3687 | 8.397 | 0.1190 | 10 | 23 | +13 |
| MuSiQue | 6 | 1.688 | 0.2813 | 4.258 | 0.0641 | 11 | 26 | +15 |
| MuSiQue | 12 | 4.776 | 0.3980 | 8.24 | 0.1439 | 14 | 33 | +19 |
| HotpotQA | 6 | 0.901 | 0.1502 | 4.886 | 0.0002 | 5 | 39 | +34 |
| HotpotQA | 12 | 2.213 | 0.1844 | 9.718 | 0.0015 | 6 | 52 | +46 |
| SQuAD | 6 | 2.026 | 0.3377 | 4.814 | 0.1999 | 3 | 17 | +14 |
| SQuAD | 12 | 5.617 | 0.4681 | 9.454 | 0.3278 | 3 | 22 | +19 |

The mechanism is confirmed and it is extreme: a boundary incumbent holding **only** canonical evidence is retained at a rate of 0.0002–0.1999 (0.0009 on MetaQA, 0.0002 on HotpotQA — effectively never). Across all six corpora only 0.15–0.34 of the B slots go to incumbents at all; the rest import partitions from outside BASE50. F6 is not weighing incumbents against challengers, it is replacing them.

**And that is, on balance, the right thing to do.** Net gold partitions at B=6 are positive on every corpus (metaqa +74, webqsp +1, 2wiki +14, musique +15, hotpotqa +34, squad +14). The turnover gains more gold than it destroys — which is why the standing router beats BASE at all.

The failure is at the margin, and B is where it shows: at B=12 WebQSP flips to -81 net gold partitions (196 evicted against 115 gained) while every other corpus stays positive. Raising B does not buy more search; it buys more eviction, and eviction stops paying at a corpus-specific point. That is the B-sweep below, explained.


---

## STEP 1 — REDEFINING THE L1 CHANNELS

The conceptual channel set `{base_rank, ret_rrf, structure}` is rejected as instructed: `base_rank` and `ret_rrf` are one Dense+SPLADE retrieval read at two granularities. The three orthogonal families are `DENSE_PARTITION_RANK`, `SPLADE_PARTITION_RANK`, `STRUCTURAL_PARTITION_RANK`, and the split is exact:

## T3 — STEP 1 the three orthogonal channels

| corpus | npart | base_rank == RRF(PR_dense, PR_splade) | ρ(DENSE, SPLADE) | ρ(DENSE, BASE) | ρ(SPLADE, BASE) |
|---|---|---|---|---|---|
| MetaQA | 401 | **EXACT** | 0.3451 | 0.7713 | 0.7985 |
| WebQSP | 7814 | **EXACT** | 0.9069 | 0.9519 | 0.9626 |
| 2Wiki | 658 | **EXACT** | 0.6862 | 0.8850 | 0.8890 |
| MuSiQue | 136 | **EXACT** | 0.5805 | 0.8774 | 0.8757 |
| HotpotQA | 5074 | **EXACT** | 0.7617 | 0.8935 | 0.9042 |
| SQuAD | 190 | **EXACT** | 0.6975 | 0.9104 | 0.9133 |

**`BASE_RANK_PARITY = EXACT` on all six**, so Dense and SPLADE can each be given exactly one vote with no loss and no re-encoding. The two modalities are far from redundant with each other (ρ from 0.345 on MetaQA to 0.907 on WebQSP), which is why the canonical fusion is worth keeping intact.

### The duplication is real in support but NOT in ordering

## T2 — STEP 0/1 measured modality redundancy (the reason for the channel redefinition)

| corpus | frac. of retrieval challengers already canonically ranked | Spearman(canonical, retrieval) on shared partitions | queries with ≥5 shared |
|---|---|---|---|
| MetaQA | 0.9534 | 0.2488 | 1795 |
| WebQSP | 0.9378 | 0.3154 | 233 |
| 2Wiki | 0.9986 | 0.3442 | 1331 |
| MuSiQue | 1.0000 | 0.158 | 221 |
| HotpotQA | 0.8148 | 0.3243 | 1618 |
| SQuAD | 1.0000 | 0.2378 | 672 |

The set overlap is what the previous phase measured: 81–100% of the partitions the retrieval channel promotes are already canonically ranked. But the **rank correlation between the two lexical views on the partitions they share is only ρ = 0.158–0.344**. They agree on *which* partitions matter and disagree on *how much*. That distinction decides the phase.

The directive allows a second vote from the same evidence family only if an ablation shows truly independent information. That ablation is the five-variant family below — every way of collapsing the two lexical granularities into one vote, plus the control that keeps both:

## T19 — STEP 1 executed literally: ONE vote per evidence family (ΔALL vs BASE)

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| B=6 · F6      canonical + structural + retrieval  (3 votes, lexical twice) | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| B=6 · ES      canonical + structural              (2, node view DELETED) | +0.0130\* | +0.0007 | -0.0030 | -0.0045 | -0.0020 | -0.0005 | +0.0006 | -0.0045 |
| B=6 · LEXONE  RRF(canonical, retrieval) + structural (2, ONE lexical vote) | +0.0100\* | +0.0014 | -0.0005 | -0.0035 | +0.0010 | +0.0015 | +0.0017 | -0.0035 |
| B=6 · DS3     dense + splade + structural         (the directive's literal set) | +0.0120\* | +0.0035 | -0.0035 | -0.0050 | -0.0010 | +0.0005 | +0.0011 | -0.0050 |
| B=6 · DS4     dense + splade + structural + retrieval | +0.0015 | +0.0042 | +0.0050\* | +0.0065\* | +0.0160\* | +0.0065\* | +0.0066 | +0.0015 |
| B=8 · F6      canonical + structural + retrieval  (3 votes, lexical twice) | +0.0050 | +0.0000 | +0.0055\* | +0.0090\* | +0.0175\* | +0.0080\* | +0.0075 | +0.0000 |
| B=8 · ES      canonical + structural              (2, node view DELETED) | +0.0140\* | -0.0049 | -0.0055\* | -0.0065\* | -0.0045 | -0.0010 | -0.0014 | -0.0065 |
| B=8 · LEXONE  RRF(canonical, retrieval) + structural (2, ONE lexical vote) | +0.0115\* | -0.0042 | -0.0030 | -0.0045 | +0.0000 | +0.0010 | +0.0001 | -0.0045 |
| B=8 · DS3     dense + splade + structural         (the directive's literal set) | +0.0135\* | +0.0007 | -0.0055\* | -0.0055 | -0.0030 | -0.0010 | -0.0001 | -0.0055 |
| B=8 · DS4     dense + splade + structural + retrieval | +0.0035 | +0.0035 | +0.0055\* | +0.0080\* | +0.0170\* | +0.0075\* | +0.0075 | +0.0035 |

MetaQA per hop (ΔALL vs BASE):

| variant | hop1 | hop2 | hop3 |
|---|---|---|---|
| F6_B6 | +0.0000 | +0.0060 | +0.0015 |
| ES_B6 | +0.0000 | +0.0255\* | +0.0135 |
| LEXONE_B6 | +0.0000 | +0.0165 | +0.0135 |
| DS3_B6 | +0.0000 | +0.0285\* | +0.0075 |
| DS4_B6 | +0.0000 | +0.0030 | +0.0015 |

Three ways to obey the one-vote-per-family rule were tested, and **all three fail in the same place**:

- `ES` (delete the node view): macro +0.0006 / worst -0.0045, significantly worse than the standing router on 4/6 corpora.
- `LEXONE` (fuse the two granularities into one lexical rank): macro +0.0017 / worst -0.0035, significantly worse on 4/6.
- `DS3` — **the directive's literal target form**, `DENSE + SPLADE + STRUCTURAL`, one vote each: macro +0.0011 / worst -0.0050, significantly worse on 4/6.

All three **gain on MetaQA** (`ES` hop2 +0.0255, `DS3` hop2 +0.0285, against the standing router's +0.0060) and lose on the text corpora. The exposed-channel set is not the problem; the node-level lexical view is doing real work on the text corpora and harmful work on MetaQA. That is a corpus-level regime split with no query-local separator — the exact object the previous phase searched for and the directive instructed me to stop searching for. I stopped.

`DS4` — dense + splade + structural **and** the node view — is the one channel redefinition that costs nothing: macro +0.0066 at B=6 and +0.0075 at B=8, **not significantly different from the standing router on any corpus** (0/6 and 0/6 significant losses). So the canonical partition RRF *can* be split into its two named channels for free — but that does not remove the duplication, because the duplication is partition-granularity versus node-granularity, not dense versus SPLADE. `DS4` spends an extra channel to expose the families and buys nothing, and STEP 6 says prefer fewer signals.

On the standing no-corpus-left-behind criterion the nominal leader is **`DS4_B8`** (macro +0.0075 / worst +0.0035) against the standing `F6_B6` (+0.0069 / +0.0025) — but it is **not significant on any corpus** (best paired p = 0.7266), so it is a candidate, not a demonstrated improvement, and it is not adopted here.

The conclusion is uncomfortable but clean: **the third vote is not a duplicate to be deleted.** The node-level view carries ordering information the partition-level view does not. The architectural ideal and the measurement disagree here, and this report follows the measurement — while recording the inconsistency as an open reason not to freeze.

---

## STEP 2 — THE CANONICAL PARTITION GRAPH

Built once per corpus from the frozen MASTER_TOPOLOGY=C partition map: every node edge `u -> v` accumulates mass on `part(u) -> part(v)`, direction and edge mass preserved. Self-loops (`part(u) == part(v)`) are real internal mass but move no probability between partitions, so they are stored separately and excluded from the transition matrix. No repartitioning, no encoder work, cached permanently under `graph/pgraph_{ds}.npz`.

## T4 — STEP 2 canonical partition transition graph (built once, cached permanently)

| corpus | partitions | nodes | inter-partition edges | density | out-deg mean | out-deg max | isolated | internal (self) mass frac | cache bytes | build s |
|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | 401 | 40151 | 63388 | 0.3942 | 158.07 | 398 | 0 | 0.309 | 84,681 | 0.1 |
| WebQSP | 7814 | 781485 | 1008926 | 0.0165 | 129.12 | 4966 | 0 | 0.268 | 2,110,755 | 0.9 |
| 2Wiki | 658 | 65865 | 56140 | 0.1297 | 85.32 | 657 | 0 | 0.199 | 95,827 | 0.1 |
| MuSiQue | 136 | 13672 | 7596 | 0.4107 | 55.85 | 131 | 0 | 0.536 | 13,880 | 0.0 |
| HotpotQA | 5074 | 507494 | 2389756 | 0.0928 | 470.98 | 4986 | 0 | 0.119 | 4,931,073 | 2.0 |
| SQuAD | 190 | 19029 | 12808 | 0.3548 | 67.41 | 189 | 0 | 0.425 | 26,548 | 0.1 |

**These graphs are dense.** Mean out-degree is 56–471 out of 136–7814 partitions, density 0.017–0.411, and not one partition is isolated on any corpus. A partition graph in which the average partition links to a quarter of the corpus is a poor substrate for diffusion: there is almost nothing for a random walk to discover that a two-hop neighbourhood does not already contain. This predicts the PPR results before any of them are run.

---

## STEP 3 — THREE PPR SEMANTICS, KEPT SEPARATE

### P1 — partition-graph PPR ("which partitions are structurally important to this query?")

Universal query seed logic, identical to the frozen router: Dense + SPLADE retrieval fused by fixed node RRF, seed nodes mapped to canonical partitions, mass `1/(K0+r)` aggregated per partition and row-normalised. One PPR rank over partitions.

### P2 — bounded partition-graph PPR

Same personalization, but the walk runs on a query-local partition graph induced on `BASE50 ∪ dense-top-50 ∪ splade-top-50 ∪ structural aggregate ∪ retrieval aggregate ∪ seed partitions`, with a neighbour-expanded variant capped at the frozen `DEG_CAP=300`. Candidate and edge counts per query are in T16 (80-151 partitions, 1,646-10,702 edges).

### P3 — partition-bounded node PPR

For a candidate partition `P`, only `P`'s own nodes and internal edges; probability may not leave `P`. Summaries: max personalized node mass, top-`SEED_K` mass, concentration, reachable fraction. Because it cannot discover a partition, it is scored strictly as a partition-quality signal, restricted to the query's candidate partitions.

**Two NO-GRAPH controls with identical support were run alongside it**, and they are what makes the result interpretable: `CTRL_seed_mass` ranks the same partitions by raw seed mass, and `CTRL_support_only` ranks them with no magnitude at all. Any gain a graph summary shows that its own control also shows is support, not structure.

## T13 — STEP 3 P3 partition-bounded node PPR, with the two NO-GRAPH controls

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| L0_BASE | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L1_SWAP_F6_B6 | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| L3_DIRECT_PPRGLOBAL | -0.0140\* | -0.0120 | +0.0000 | -0.0040 | +0.0050 | -0.0005 | -0.0043 | -0.0140 |
| L3p_DIRECT_PPR_PRECOMP_L32 | -0.0295\* | -0.0197\* | -0.0045 | +0.0000 | +0.0075 | +0.0025 | -0.0073 | -0.0295 |
| L5_P3_CTRL_seed_mass_NO_GRAPH | -0.0435\* | -0.0056 | +0.0030 | +0.0100\* | +0.0235\* | +0.0130\* | +0.0001 | -0.0435 |
| L5_P3_CTRL_support_only_NO_GRAPH | -0.0390\* | -0.0049 | +0.0040 | +0.0095\* | +0.0240\* | +0.0130\* | +0.0011 | -0.0390 |
| L5_P3_concentration | -0.0455\* | -0.0070 | +0.0040 | +0.0100\* | +0.0240\* | +0.0130\* | -0.0002 | -0.0455 |
| L5_P3_max_mass | -0.0455\* | -0.0049 | +0.0040 | +0.0100\* | +0.0240\* | +0.0130\* | +0.0001 | -0.0455 |
| L5_P3_reachable_frac | -0.0435\* | -0.0056 | +0.0035 | +0.0100\* | +0.0245\* | +0.0125\* | +0.0002 | -0.0435 |
| L5_P3_topk_mass | -0.0445\* | -0.0049 | +0.0040 | +0.0095\* | +0.0245\* | +0.0125\* | +0.0002 | -0.0445 |
| L5e_P3ENTRY_CTRL_seed_mass_NO_GRAPH | -0.0095\* | +0.0028 | +0.0030 | +0.0000 | +0.0060\* | +0.0005 | +0.0005 | -0.0095 |
| L5e_P3ENTRY_CTRL_support_only_NO_GRAPH | +0.0000 | +0.0014 | +0.0035\* | +0.0000 | +0.0000 | +0.0000 | +0.0008 | +0.0000 |
| L5e_P3ENTRY_concentration | -0.0015 | +0.0000 | +0.0050\* | +0.0020 | +0.0120\* | +0.0050\* | +0.0037 | -0.0015 |
| L5e_P3ENTRY_max_mass | -0.0010 | +0.0106\* | +0.0030 | -0.0010 | -0.0025 | +0.0055\* | +0.0024 | -0.0025 |
| L5e_P3ENTRY_reachable_frac | +0.0010 | +0.0070\* | +0.0045\* | -0.0035 | -0.0030 | -0.0005 | +0.0009 | -0.0035 |
| L5e_P3ENTRY_topk_mass | -0.0015 | +0.0113\* | +0.0025 | -0.0020 | -0.0040 | +0.0040\* | +0.0017 | -0.0040 |

## T14 — STEP 3 P3 cost and support

| corpus | local operator bytes | s/query (seed-only) | s/query (entry) | partitions scored/query (seed-only) | (entry) |
|---|---|---|---|---|---|
| MetaQA | 16,088,316 | 0.005770 | 0.012200 | 39.57 | 82.18 |
| WebQSP | 276,252,016 | 0.007343 | 0.012555 | 26.67 | 56.08 |
| 2Wiki | 26,382,532 | 0.004987 | 0.008667 | 34.92 | 61.06 |
| MuSiQue | 5,500,776 | 0.004989 | 0.008528 | 34.15 | 56.73 |
| HotpotQA | 202,855,272 | 0.006173 | 0.012891 | 27.7 | 73.08 |
| SQuAD | 7,630,684 | 0.005308 | 0.013748 | 28.09 | 58.21 |

Each P3 summary is scored the way the directive defines it — as a partition-quality signal, given a third co-equal vote next to the Dense and SPLADE partition channels.

| corpus | family | best graph summary | best NO-GRAPH control | gap |
|---|---|---:|---:|---:|
| metaqa | P3 | -0.0435 | -0.0390 | -0.0045 |
| webqsp | P3 | -0.0049 | -0.0049 | +0.0000 |
| 2wiki_clean | P3 | +0.0040 | +0.0040 | +0.0000 |
| musique_clean | P3 | +0.0100 | +0.0100 | +0.0000 |
| hotpotqa_clean | P3 | +0.0245 | +0.0240 | +0.0005 |
| squad_clean | P3 | +0.0130 | +0.0130 | +0.0000 |
| metaqa | P3ENTRY | +0.0010 | +0.0000 | +0.0010 |
| webqsp | P3ENTRY | +0.0113 | +0.0028 | +0.0085 |
| 2wiki_clean | P3ENTRY | +0.0050 | +0.0035 | +0.0015 |
| musique_clean | P3ENTRY | +0.0020 | +0.0000 | +0.0020 |
| hotpotqa_clean | P3ENTRY | +0.0120 | +0.0060 | +0.0060 |
| squad_clean | P3ENTRY | +0.0055 | +0.0005 | +0.0050 |

**The two P3 families do not behave the same way, and the difference is the whole result.**

**Seed-only P3** (score only the partitions the query's seeds landed in): the graph adds nothing. Gaps run -0.0045 to +0.0005, mean -0.0007; the sign is positive on only 1/6 corpora (sign test p = 0.219). MuSiQue's `+0.0100` and SQuAD's `+0.0130` are reproduced *exactly* by ranking the same partitions by raw seed mass, and MetaQA's large negative is less negative for the controls than for the diffusion summaries. Here the channel is measuring support, not structure.

**Entry-mode P3** (score every candidate partition, entering it at its own best seed): the graph **does** beat its own same-support controls, on **6 of 6 corpora** (gaps +0.0010 to +0.0085, mean +0.0040, sign test p = 0.031). That is a real signal: internal connectivity around the entry node carries partition-quality information that raw seed mass does not. I did not run a paired per-query test between a summary and its control, so the per-corpus gaps are point estimates; the consistent sign across six independent corpora is the evidence.

**It is still not worth taking.** The best entry-mode variant is `L5e_P3ENTRY_concentration` at macro +0.0037 / worst -0.0015, against the standing router's +0.0069 / +0.0025 — below it on both, with a negative corpus. And it is the most expensive signal in the phase: T14 puts it at 0.0085-0.0137 s/query with a 5.5-276 MB local operator, versus the standing structural channel's zero online graph work.

So partition-bounded node PPR **does** provide partition-quality information — the honest answer to the question as asked is a qualified yes — but not enough of it to pay for itself, and not enough to reach a router that touches no graph at query time.

---

## STEP 4 — PRECOMPUTATION

The inherited iteration `p_k = (1-a)S + a p_{k-1} P`, `p_0 = S`, is **linear in S**, so

```
p_20 = S @ M       with   M = (1-a) * sum_{j=0..19} (aP)^j + (aP)^20
                          M_k = (1-a) I + a M_{k-1} P,   M_0 = I
```

Row `p` of `M` is exactly the PPR response to a unit personalization at partition `p`. So the "full precomputed basis" is **not an approximation of online PPR — it is the same function**, and any disagreement is floating-point. Only the top-L truncation is lossy. `M` is built and consumed in row blocks, so the dense basis is never held for the large corpora.

## T12 — STEP 4 precomputation. (B) full basis is EXACT by linearity; only (C) is lossy

| corpus | basis precompute s | full basis bytes | (A) online s/query | (A) online graph edges/query | (B) top50 set agreement | (B) max abs Δmass | (C) L=8 | (C) L=16 | (C) L=32 | (C) L=64 | (C) L=32 bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | 0.3 | 643,204 | 0.000425 | 1,267,760 | 1.0000 | 5.20e-17 | 0.0000 | 0.0000 | 0.0005 | 0.0000 | 102,656 |
| WebQSP | 110.1 | 244,234,384 | 0.013502 | 20,178,520 | 1.0000 | 3.47e-16 | 0.0000 | 0.0000 | 0.0000 | 0.0395 | 2,000,384 |
| 2Wiki | 0.4 | 1,731,856 | 0.000488 | 1,122,800 | 1.0000 | 4.16e-16 | 0.0000 | 0.0000 | 0.0180 | 0.0040 | 168,448 |
| MuSiQue | 0.0 | 73,984 | 0.000066 | 151,920 | 1.0000 | 6.25e-17 | 0.0000 | 0.0000 | 0.0030 | 0.0165 | 34,816 |
| HotpotQA | 144.7 | 102,981,904 | 0.030566 | 47,795,120 | 1.0000 | 9.71e-17 | 0.0000 | 0.0000 | 0.0015 | 0.0590 | 1,298,944 |
| SQuAD | 0.0 | 144,400 | 0.000103 | 256,160 | 1.0000 | 6.94e-17 | 0.0000 | 0.0000 | 0.0000 | 0.0300 | 48,640 |

Query-time L1 with the basis is: retrieve, look up ~100 sparse basis rows, merge ranks, take top-50. **`ONLINE_GRAPH_EDGES_TOUCHED = 0`**, against 152k–47.8M for online global PPR. The precomputed variant also behaves the same as online PPR where it matters: in the swap bake-off `PPRP32` tracks `PPRG` within ±0.0015 on five of six corpora (T9).

So the answer to "can PPR be made effectively fully precomputed?" is an unqualified yes — exactly, cheaply, and with zero online graph work. It simply does not help.

---

## STEP 5 — B REMOVED AS AN AXIS, AND DIRECT TOP-50 FUSION

## T6 — STEP 5A B-sweep. B is removed as an axis; output is still exactly 50 (ΔALL vs BASE)

| corpus | F6 B=2 | F6 B=4 | F6 B=6 | F6 B=8 | F6 B=12 | ES B=2 | ES B=4 | ES B=6 | ES B=8 | ES B=12 |
|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | +0.0030 | +0.0025 | +0.0025 | +0.0050 | +0.0060 | +0.0100\* | +0.0125\* | +0.0130\* | +0.0140\* | +0.0130\* |
| WebQSP | +0.0028 | +0.0042 | +0.0028 | +0.0000 | -0.0063 | +0.0049 | +0.0035 | +0.0007 | -0.0049 | -0.0106 |
| 2Wiki | +0.0050\* | +0.0065\* | +0.0060\* | +0.0055\* | +0.0045 | -0.0005 | -0.0010 | -0.0030 | -0.0055\* | -0.0100\* |
| MuSiQue | +0.0010 | +0.0060\* | +0.0070\* | +0.0090\* | +0.0085\* | +0.0005 | -0.0040 | -0.0045 | -0.0065\* | -0.0075\* |
| HotpotQA | +0.0085\* | +0.0130\* | +0.0160\* | +0.0175\* | +0.0210\* | +0.0005 | -0.0010 | -0.0020 | -0.0045 | -0.0130\* |
| SQuAD | +0.0040\* | +0.0065\* | +0.0070\* | +0.0080\* | +0.0095\* | +0.0000 | +0.0000 | -0.0005 | -0.0010 | -0.0030 |

- **F6** — B=2: macro +0.0041 / worst +0.0010; B=4: macro +0.0065 / worst +0.0025; B=6: macro +0.0069 / worst +0.0025; B=8: macro +0.0075 / worst +0.0000; B=12: macro +0.0072 / worst -0.0063
- **ES** — B=2: macro +0.0026 / worst -0.0005; B=4: macro +0.0017 / worst -0.0040; B=6: macro +0.0006 / worst -0.0045; B=8: macro -0.0014 / worst -0.0065; B=12: macro -0.0052 / worst -0.0130

`B` is a real axis, but it is corpus-dependent and it does not point at MetaQA. Under F6 it helps HotpotQA monotonically (+0.0085 → +0.0210) and SQuAD (+0.0040 → +0.0095), is flat on MetaQA (+0.0030 → +0.0060), and **hurts** WebQSP (+0.0028 → -0.0063). B=8 has a better macro (+0.0075) than B=6 (+0.0069) but a worse worst corpus (+0.0000 vs +0.0025), so under the standing "no corpus left behind" criterion B=6 remains the universal choice and B=8 is the runner-up. Changing B never changes P_MAIN; every cell above outputs exactly 50.

### DIRECT top-50 fusion is architecturally preferable and empirically worse

## T5 — STEP 5B/6 DIRECT exactly-P50 fusion vs the protected-core swap (ΔALL vs BASE)

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| L0  BASE = Dense+SPLADE partition RRF | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| L2  DIRECT + S4 structural | -0.0065 | -0.0063 | -0.0175\* | -0.0125\* | -0.0095\* | -0.0045 | -0.0095 | -0.0175 |
| L3  DIRECT + global partition PPR (P1) | -0.0140\* | -0.0120 | +0.0000 | -0.0040 | +0.0050 | -0.0005 | -0.0043 | -0.0140 |
| L3b DIRECT + global PPR, top-64 only | -0.0380\* | -0.0324\* | -0.0085\* | -0.0190\* | +0.0115\* | -0.0115\* | -0.0163 | -0.0380 |
| L4  DIRECT + bounded partition PPR (P2) | -0.0085\* | +0.0035 | -0.0025 | -0.0035 | -0.0020 | -0.0005 | -0.0023 | -0.0085 |
| L4b DIRECT + bounded PPR, neighbour-expanded | -0.0145\* | +0.0021 | -0.0030 | -0.0040 | -0.0015 | -0.0005 | -0.0036 | -0.0145 |
| L6  DIRECT + S4 + global PPR | -0.0010 | -0.0028 | -0.0205\* | -0.0180\* | +0.0020 | -0.0030 | -0.0072 | -0.0205 |
| **L1  SAFE_REFERENCE_ROUTER B6_S4_F6** | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| SWAP de-duplicated (no retrieval channel), B=6 | +0.0130\* | +0.0007 | -0.0030 | -0.0045 | -0.0020 | -0.0005 | +0.0006 | -0.0045 |

`DIRECT` with no third channel reproduces BASE **bit-for-bit on all six corpora** (`DIRECT_NOEXTRA_PARITY = EXACT`), so the comparison is clean. Adding any third channel to a full-corpus fusion is negative on macro for every variant tried, and the best of them (L4_DIRECT_PPRBOUNDED, macro -0.0023) is still below BASE and far below the swap (+0.0069 macro, +0.0025 worst).

The MetaQA per-hop table shows exactly why:

## T7 — STEP 8 MetaQA per-hop (ΔALL vs BASE); hop1 n=666 hop2 n=666 hop3 n=666

| variant | hop1 | hop2 | hop3 |
|---|---|---|---|
| L0_BASE | +0.0000 | +0.0000 | +0.0000 |
| L2_DIRECT_S4 | -0.0045 | -0.0390\* | +0.0240 |
| L3_DIRECT_PPRGLOBAL | -0.0060 | -0.0526\* | +0.0165 |
| L3b_DIRECT_PPRGLOBAL_M64 | -0.0240\* | -0.0946\* | +0.0045 |
| L4_DIRECT_PPRBOUNDED | -0.0030 | -0.0300\* | +0.0075 |
| L4b_DIRECT_PPRBOUNDED_EXP | -0.0060 | -0.0511\* | +0.0135 |
| L6_DIRECT_S4_PLUS_PPR | -0.0165\* | -0.0315\* | +0.0450\* |
| L1_SWAP_F6_B6 | +0.0000 | +0.0060 | +0.0015 |
| SWAP_ES_B6 | +0.0000 | +0.0255\* | +0.0135 |
| L1_SWAP_F6_B8 | +0.0000 | +0.0090 | +0.0060 |
| L1_SWAP_F6_B12 | +0.0000 | +0.0090 | +0.0090 |
| SWAP_ES_B8 | +0.0000 | +0.0255\* | +0.0165 |
| SWAP_ES_B12 | +0.0000 | +0.0165 | +0.0225\* |

Every DIRECT variant **buys hop3 and pays for it in hop2** — up to -0.0946 on hop2 for a +0.0045 hop3 gain. A structural channel with a co-equal full-corpus vote promotes partitions everywhere, including over the canonical partitions that were carrying the easy hops. Churn confirms it: 4.5–19.4 partitions replaced per query for DIRECT against 4.0–5.1 for the swap. **The protected core is not scaffolding around a selector; it is the thing that makes a third channel safe.** Eliminating the swap subsystem would be architecturally cleaner and costs real coverage on every corpus.

---

## STEP 6 — WHICH ONE STRUCTURAL SIGNAL?

Asked inside the architecture that actually works, with the duplicated channel present (F6) and absent (ES), at B=6 and B=8:

## T9 — STEP 6 structural-signal bake-off inside the swap (ΔALL vs BASE)

| variant | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| **S4_F6_B6** | +0.0025 | +0.0028 | +0.0060\* | +0.0070\* | +0.0160\* | +0.0070\* | +0.0069 | +0.0025 |
| PPRG_F6_B6 | -0.0075\* | -0.0113\* | +0.0065\* | +0.0035 | +0.0160\* | +0.0060\* | +0.0022 | -0.0113 |
| PPRB_F6_B6 | -0.0065\* | +0.0028 | +0.0050 | +0.0045 | +0.0105\* | +0.0065\* | +0.0038 | -0.0065 |
| PPRP32_F6_B6 | -0.0080\* | -0.0099\* | +0.0070\* | +0.0040 | +0.0160\* | +0.0075\* | +0.0028 | -0.0099 |
| S4+PPRG_F6_B6 | -0.0065\* | -0.0007 | +0.0055\* | +0.0030 | +0.0165\* | +0.0045 | +0.0037 | -0.0065 |
| S4_ES_B6 | +0.0130\* | +0.0007 | -0.0030 | -0.0045 | -0.0020 | -0.0005 | +0.0006 | -0.0045 |
| PPRG_ES_B6 | -0.0015 | -0.0099\* | +0.0000 | -0.0100\* | +0.0010 | -0.0005 | -0.0035 | -0.0100 |
| PPRB_ES_B6 | +0.0005 | +0.0028 | -0.0020 | -0.0075\* | -0.0045 | +0.0010 | -0.0016 | -0.0075 |
| PPRP32_ES_B6 | +0.0000 | -0.0106\* | -0.0005 | -0.0110\* | +0.0015 | -0.0005 | -0.0035 | -0.0110 |
| S4+PPRG_ES_B6 | +0.0040 | +0.0007 | -0.0015 | -0.0070\* | +0.0020 | +0.0005 | -0.0002 | -0.0070 |
| S4_F6_B8 | +0.0050 | +0.0000 | +0.0055\* | +0.0090\* | +0.0175\* | +0.0080\* | +0.0075 | +0.0000 |
| PPRG_F6_B8 | -0.0095\* | -0.0113\* | +0.0065\* | +0.0045 | +0.0190\* | +0.0065\* | +0.0026 | -0.0113 |
| PPRB_F6_B8 | -0.0080\* | +0.0021 | +0.0045 | +0.0045 | +0.0125\* | +0.0070\* | +0.0038 | -0.0080 |
| S4+PPRG_F6_B8 | -0.0075\* | -0.0007 | +0.0055 | +0.0040 | +0.0185\* | +0.0060\* | +0.0043 | -0.0075 |

**`S4_F6_B8` wins** (macro +0.0075, worst +0.0000) and is the only configuration with no negative corpus. Every PPR variant — global, bounded, precomputed — has a significantly negative corpus, and adding global PPR *to* S4 makes S4 worse on MetaQA (+0.0130 → +0.0040 under ES). Fewer signals, and the simplest one, wins.

## T11 — MetaQA per-hop for the bake-off (ΔALL vs BASE)

| variant | hop1 | hop2 | hop3 |
|---|---|---|---|
| S4_F6_B6 | +0.0000 | +0.0060 | +0.0015 |
| PPRG_F6_B6 | +0.0000 | -0.0180\* | -0.0045 |
| PPRB_F6_B6 | +0.0000 | -0.0165\* | -0.0030 |
| PPRP32_F6_B6 | +0.0000 | -0.0195\* | -0.0045 |
| S4+PPRG_F6_B6 | +0.0000 | -0.0135\* | -0.0060 |
| S4_ES_B6 | +0.0000 | +0.0255\* | +0.0135 |
| PPRG_ES_B6 | +0.0000 | -0.0135\* | +0.0090 |
| PPRB_ES_B6 | +0.0000 | -0.0060 | +0.0075 |
| PPRP32_ES_B6 | +0.0000 | -0.0120 | +0.0120 |
| S4+PPRG_ES_B6 | +0.0000 | +0.0000 | +0.0120 |
| S4_F6_B8 | +0.0000 | +0.0090 | +0.0060 |
| PPRG_F6_B8 | +0.0000 | -0.0240\* | -0.0045 |
| PPRB_F6_B8 | +0.0000 | -0.0225\* | -0.0015 |
| S4+PPRG_F6_B8 | +0.0000 | -0.0180\* | -0.0045 |

---

## STEP 7 — CAP_B6 vs CAP_P50 (these are not the same thing)

`CAP_P50` = more than 50 gold partitions: impossible at P=50 under **any** replacement. `CAP_B6` = every missing gold is reachable, but more than B=6 partitions of BASE would have to change — recoverable by raising B while still emitting exactly 50, which is **not** a contract change. `REACH` = a missing gold partition is not in the query's candidate universe at all. `RANKING` = ≤B changes would suffice and the selector ordered them wrong.

## T15 — STEP 7 CAP_B6 vs CAP_P50. These are NOT the same thing.

Decomposition of the queries the SAFE_REFERENCE_ROUTER (B6_S4_F6) leaves uncovered. `CAP_P50` = more than 50 gold partitions, impossible at P=50 under ANY replacement. `REACH` = a missing gold partition is not in the query's candidate universe at all. `CAP_B6` = every missing gold IS reachable but more than B=6 partitions of BASE would have to change. `RANKING` = ≤6 changes would suffice and the selector simply ordered them wrong.

| corpus | CAP_P50 | REACH | CAP_B6 | RANKING | REACH after adding PPR-reachable | mean gold partitions/query | mean gold outside BASE50 |
|---|---|---|---|---|---|---|---|
| MetaQA | 26 | 318 | 21 | 312 | 0 | 5.78 | 2.947 |
| WebQSP | 7 | 127 | 13 | 187 | 0 | 3.66 | 1.188 |
| 2Wiki | 0 | 49 | 0 | 64 | 0 | 1.98 | 0.066 |
| MuSiQue | 0 | 0 | 0 | 73 | 0 | 1.77 | 0.045 |
| HotpotQA | 0 | 50 | 0 | 49 | 0 | 1.66 | 0.069 |
| SQuAD | 0 | 0 | 0 | 25 | 0 | 1.0 | 0.019 |

MetaQA by hop (router pool):

| hop | CAP_P50 | REACH | CAP_B6 | RANKING |
|---|---|---|---|---|
| hop2 | 2 | 42 | 7 | 129 |
| hop3 | 24 | 275 | 14 | 181 |

**This overturns the previous phase's capacity claim.** That phase reported 213 CAPACITY + 211 REACH = 87.2% of residual MetaQA hop3 as out of reach for a B≤6 router, with only 12.8% ranking. Separating the two capacity classes as instructed shows `CAP_B6` is **21 of 677** MetaQA failures (3.1%), **14 of 494** on hop3 (2.8%), and **0 on every text corpus**. The old CAPACITY class was mostly CAP_P50 plus REACH. Raising B is not the lever.

### The graph solves REACH completely, and it changes nothing

Recomputing the same decomposition with the PPR-reachable set added to the candidate universe collapses **`REACH` to exactly 0 on all six corpora**. Every gold partition the router's own universe cannot see is reachable through the partition graph. Those queries do not disappear — they reclassify:

| corpus | pool CAP_P50 | pool REACH | pool CAP_B6 | pool RANKING | +PPR CAP_P50 | +PPR REACH | +PPR CAP_B6 | +PPR RANKING |
|---|---|---|---|---|---|---|---|---|
| metaqa | 26 | 318 | 21 | 312 | 26 | **0** | **215** | **436** |
| webqsp | 7 | 127 | 13 | 187 | 7 | **0** | **53** | **274** |
| 2wiki_clean | 0 | 49 | 0 | 64 | 0 | **0** | 0 | **113** |
| musique_clean | 0 | 0 | 0 | 73 | 0 | 0 | 0 | 73 |
| hotpotqa_clean | 0 | 50 | 0 | 49 | 0 | **0** | 0 | **99** |
| squad_clean | 0 | 0 | 0 | 25 | 0 | 0 | 0 | 25 |

On MetaQA the 318 REACH failures become 194 more `CAP_B6` and 124 more `RANKING`; on 2Wiki and HotpotQA they become `RANKING` outright. **So the pool is not the binding constraint.** With the graph in the universe, 436/677 of MetaQA's residual failures are selector-solvable inside B=6 and only 26 are impossible at P=50.

And the coverage tables say this buys nothing. That juxtaposition — reach fully solved, coverage unmoved — is the phase's central finding, and it also warns that pool-widening and `B` interact: widening the universe pushes queries into `CAP_B6` (21 → 215 on MetaQA), so a future wider pool would need more replacement slots, not the same six.

## T10 — STEP 7 reach probe: PPR *does* reach gold partitions S4 cannot

| corpus | gold partitions outside BASE50 | reached by S4 | reached ONLY by global PPR | ONLY by bounded PPR | ONLY by precomputed PPR | new partitions/q vs S4 (G/B/P32) |
|---|---|---|---|---|---|---|
| MetaQA | 5889 | 1536 | 867 | 611 | 827 | 26.97 / 20.05 / 27.3 |
| WebQSP | 1686 | 123 | 196 | 262 | 209 | 33.99 / 17.87 / 32.68 |
| 2Wiki | 131 | 10 | 37 | 34 | 42 | 25.93 / 19.84 / 26.23 |
| MuSiQue | 90 | 20 | 33 | 37 | 40 | 20.89 / 12.47 / 19.35 |
| HotpotQA | 138 | 17 | 48 | 36 | 56 | 35.4 / 22.03 / 36.43 |
| SQuAD | 39 | 10 | 11 | 14 | 20 | 27.67 / 16.0 / 27.28 |

**Partition PPR does reach gold partitions S4 cannot** — on WebQSP it reaches 196 gold partitions S4 misses entirely against S4's own 123, and it is ahead on 2Wiki (37 vs 10), HotpotQA (48 vs 17) and MuSiQue (33 vs 20). That reach converts to **no** coverage anywhere (T9). The reason is the P=50 budget: the extra partitions arrive with no discriminative signal attached, so promoting them costs a canonical partition that was carrying a gold. **Reach is not coverage under a hard budget** — this is the single most useful negative result of the phase, and it applies to any future "widen the pool" idea.

### Is the selector or the pool the limiter?

## T20 — STEP 7 addendum: is the SELECTOR or the POOL the limiter?

`ORACLE_B` = exists X with |X|=B drawn from (boundary + challengers) such that the protected core plus X covers all gold partitions. It is the best any parameter-free selector could do at that B without changing the contract.

| corpus | B | actual ALL | oracle ALL | selector headroom | uncovered even by oracle | of those: POOL failure | of those: need > B swaps | mean partitions needing replacement |
|---|---|---|---|---|---|---|---|---|
| MetaQA | 6 | 0.6612 | 0.7212 | **+0.0601** | 557 | 555 | 2 | 3.069 |
| MetaQA | 8 | 0.6637 | 0.7212 | **+0.0576** | 557 | 555 | 2 | 3.119 |
| MetaQA | 12 | 0.6647 | 0.7222 | **+0.0576** | 555 | 555 | 0 | 3.211 |
| WebQSP | 6 | 0.7646 | 0.7886 | **+0.0240** | 300 | 300 | 0 | 1.276 |
| WebQSP | 8 | 0.7618 | 0.7886 | **+0.0268** | 300 | 300 | 0 | 1.31 |
| WebQSP | 12 | 0.7555 | 0.7886 | **+0.0331** | 300 | 300 | 0 | 1.386 |
| 2Wiki | 6 | 0.9435 | 0.9515 | **+0.0080** | 97 | 97 | 0 | 0.073 |
| 2Wiki | 8 | 0.9430 | 0.9515 | **+0.0085** | 97 | 97 | 0 | 0.077 |
| 2Wiki | 12 | 0.9420 | 0.9515 | **+0.0095** | 97 | 97 | 0 | 0.083 |
| MuSiQue | 6 | 0.9635 | 0.9750 | **+0.0115** | 50 | 50 | 0 | 0.064 |
| MuSiQue | 8 | 0.9655 | 0.9750 | **+0.0095** | 50 | 50 | 0 | 0.069 |
| MuSiQue | 12 | 0.9650 | 0.9750 | **+0.0100** | 50 | 50 | 0 | 0.078 |
| HotpotQA | 6 | 0.9505 | 0.9645 | **+0.0140** | 71 | 71 | 0 | 0.081 |
| HotpotQA | 8 | 0.9520 | 0.9645 | **+0.0125** | 71 | 71 | 0 | 0.086 |
| HotpotQA | 12 | 0.9555 | 0.9645 | **+0.0090** | 71 | 71 | 0 | 0.1 |
| SQuAD | 6 | 0.9875 | 0.9945 | **+0.0070** | 11 | 11 | 0 | 0.025 |
| SQuAD | 8 | 0.9885 | 0.9945 | **+0.0060** | 11 | 11 | 0 | 0.026 |
| SQuAD | 12 | 0.9900 | 0.9945 | **+0.0045** | 11 | 11 | 0 | 0.032 |

MetaQA per hop at B=6:

| hop | n | actual | oracle | selector headroom | POOL failures |
|---|---|---|---|---|---|
| hop1 | 666 | 0.9955 | 0.9970 | **+0.0015** | 2 |
| hop2 | 666 | 0.7297 | 0.8033 | **+0.0736** | 131 |
| hop3 | 666 | 0.2583 | 0.3634 | **+0.1051** | 422 |

---

## STEP 8 / STEP 9 — MetaQA multi-hop and all six corpora

Primary target hop3, then hop2; hop1 stays saturated (`+0.0000` for the swap family on every variant tested). The full per-hop tables are T7 (DIRECT and B-sweep) and T11 (structural bake-off). Nothing tested this phase improves the standing router's MetaQA hop3 without a significant regression somewhere else; `SWAP_ES` reaches hop2 +0.0255 / hop3 +0.0135 and pays for it on 2Wiki, MuSiQue, HotpotQA and SQuAD. No dataset identity was used anywhere, and no configuration was selected per corpus.

---

## STEP 10 — PPR CONSTANTS

The historical implementation was audited first, and the audit is a **negative finding**. `src/experiments/l3_methods.py::_ppr` and `l3_solvers.py` implement `p = (1-a)*s + a*(p @ P)` with `iters=20` over `P = D^-1 A` (symmetrised, row-normalised) and uniform seed mass — those are a real inherited contract and were reused unchanged. The damping `a` is **not**: both call sites select it per run from `{0.3,0.5,0.7,0.9}` / `{0.5,0.7,0.9}` **by gold recall**, store it as `ppr_best_alpha`, and `LEVEL3_README.md:130` marks the PPR alpha as exploratory. A gold-selected constant cannot be inherited into a router that may not see gold, and no surviving `ppr_best_alpha` value exists in `results/`.

One global contract was therefore fixed for every corpus and every variant: **α = 0.85** — the only non-gold-selected alpha in the codebase (`l3_solvers._qppr_ball` default) and the standard PageRank damping — and **iters = 20**, inherited. Alpha was never selected on gold here.

## T18 — STEP 10 PPR damping robustness (reported, NOT selected)

| corpus | S4_F6_B6 | DIRECT α=0.5 | DIRECT α=0.7 | DIRECT α=0.85 | DIRECT α=0.95 | SWAP α=0.5 | SWAP α=0.7 | SWAP α=0.85 | SWAP α=0.95 |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA | +0.0025 | -0.0295 | -0.0250 | -0.0140 | -0.0075 | -0.0085 | -0.0085 | -0.0075 | -0.0040 |
| HotpotQA | +0.0160 | +0.0135 | +0.0120 | +0.0050 | -0.0110 | +0.0165 | +0.0170 | +0.0160 | +0.0120 |
| WebQSP | +0.0028 | -0.0113 | -0.0127 | -0.0120 | -0.0120 | -0.0092 | -0.0085 | -0.0113 | -0.0120 |

Reported, never used for selection. The reading is not that α does not matter — it does, and **its best value is corpus-dependent**: MetaQA prefers the highest α tested, HotpotQA the lowest, WebQSP is flat and negative throughout. The reading is that **no α rescues partition PPR**. On MetaQA and WebQSP every α tested leaves it negative against BASE while S4 is positive, and on HotpotQA its best cell (α=0.7, +0.0170) merely ties S4 (+0.0160). A corpus-dependent α would also be a dataset-identity parameter, which the contract forbids. So the refutation is not an artifact of fixing α = 0.85.

---

## STEP 11 — LATENCY AS A SELECTION METRIC

## T16 — STEP 11 latency and online graph work

| corpus | npart | P1 global PPR s/q | P1 online edges/q | P2 bounded PPR s/q | P2 candidate partitions | P2 subgraph edges | P2 online edges/q | precomputed basis online edges/q |
|---|---|---|---|---|---|---|---|---|
| MetaQA | 401 | 0.0005 | 1,267,760 | 0.0013 | 141.1 | 10702.5 | 214,049 | **0** |
| WebQSP | 7814 | 0.0149 | 20,178,520 | 0.0010 | 116.5 | 1646.5 | 32,930 | **0** |
| 2Wiki | 658 | 0.0005 | 1,122,800 | 0.0014 | 127.4 | 5054.4 | 101,089 | **0** |
| MuSiQue | 136 | 0.0001 | 151,920 | 0.0011 | 80.5 | 3271.7 | 65,434 | **0** |
| HotpotQA | 5074 | 0.0385 | 47,795,120 | 0.0031 | 151.4 | 9343.3 | 186,866 | **0** |
| SQuAD | 190 | 0.0001 | 256,160 | 0.0014 | 83.5 | 3572.2 | 71,445 | **0** |

The desired property `ONLINE_GRAPH_EDGES_TOUCHED ≈ 0` **is achievable and is achieved** by the precomputed diffusion basis. Bounded PPR is 12-14x faster than global PPR on the two largest corpora and touches 2–3 orders of magnitude fewer edges. But the standing router already touches zero online partition-graph edges — its structural channel is the cached directional expansion — so on this metric there is nothing to buy.

---

## STEP 12 — STAGE SEPARATION

| stage | question | admissible mechanisms | status after this phase |
|---|---|---|---|
| **L1** | *where should we search?* | partition-level, parameter-free, precomputable, exactly-P50 | Dense + SPLADE partition RRF, protected core, S4 structural competition. **Partition PPR excluded** (P1/P2/P3 all refuted). |
| **L2** | *which candidates are relevant?* | node/candidate relevance, learning allowed | unchanged and untouched. No PPR was added. Note L2 already carries a `path_raw` graph expert previously found available-but-useless on the text corpora — that duplication is pre-existing and out of scope here. |
| **L3** | *bounded graph reasoning* | dynamic traversal, node granularity | this is where the historical PPR lives and belongs: `l3_methods`/`l3_solvers` run node-level PPR over structural + weighted-NER edges to rank documents; `LEVEL3_README.md:141` already places "Partition-local push PPR" inside the target L3 flow, and `LEVEL3_README.md:172` proposes push-based PPR over lazily loaded partitions as the L3 optimisation. |

The instruction not to duplicate one structural signal across stages without controlled justification is satisfied by exclusion: the controlled comparison was run at L1 and **failed**, so diffusion stays a single-stage (L3) mechanism.

---

## ANSWERS TO THE STOP CONDITIONS

| # | question | answer |
|---|---|---|
| 1 | Is B=6 artificially causing much of MetaQA's CAPACITY class? | **No.** `CAP_B6` is 21/677 MetaQA failures (3.1%), 14/494 on hop3, and 0 on every text corpus. |
| 2 | Can direct exactly-P50 fusion replace swapping entirely? | **No.** Best DIRECT variant macro -0.0023 vs swap +0.0069; gains hop3, destroys hop2 (up to −0.0946). |
| 3 | Does partition PPR improve reach? | **Yes — completely.** It drives the `REACH` failure class to 0 on all six corpora (T15). It converts to zero coverage gain at P=50. |
| 4 | Is bounded PPR better than global PPR? | **Yes**, on accuracy and by 12-14x on latency. Both are still refuted as L1 channels. |
| 5 | Does partition-bounded node PPR give useful partition quality information? | **Qualified yes.** Seed-only P3 is pure support (sign test p = 0.219), but entry-mode P3 beats its same-support controls on 6/6 corpora (p = 0.031). It still does not reach the standing router and is the most expensive signal tested. |
| 6 | Can PPR be made effectively fully precomputed? | **Yes, exactly** — the iteration is linear, the basis is the operator, online graph edges = 0. |
| 7 | Which structural signal is simplest? | **S4.** S4_F6_B8 is the only no-negative-corpus configuration; PPR variants each have a significant regression. |
| 8 | Simplest universal low-latency exact-P50 L1? | Dense+SPLADE partition RRF → protected core (44) → symmetric boundary competition over canonical + S4 + node-retrieval → exactly 50 — the standing router. The strictly simpler 3-channel form the directive asked for (`DS3`) is significantly worse on 4/6 corpora. |
| 9 | MetaQA hop2/hop3 | No variant improves either without a significant regression elsewhere. Standing router: hop2 +0.0060, hop3 +0.0015 (T7). |
| 10 | Six-dataset result | Standing router macro +0.0069, worst +0.0025, significant on 4/6, 0 regressions. Nothing tested beats it on the no-negative-corpus criterion. |
| 11 | Exact query-time latency | T16. Global PPR 0.0001-0.0385 s/q (151,920-47,795,120 edges); bounded PPR 0.0010-0.0031 s/q (32,930-214,049 edges); precomputed basis **0** online edges. |
| 12 | `L1_FROZEN` | **NO** — see below. |

---

## WHY NOT FREEZE, AND WHAT WOULD ACTUALLY MOVE IT

The router is unchanged and is now the best of a much larger tested space, so freezing it is defensible on the evidence. I am still returning `L1_FROZEN = NO`, for two reasons that the phase measured rather than assumed.

**1. The headroom is in the selector, it is large, and it is in-contract.** The STEP 7 addendum asks the exact combinatorial question — does there exist a choice of B replacements from the boundary and challengers that covers every gold partition? — and the answer is that a perfect parameter-free selector would gain:

| corpus | metaqa | webqsp | 2wiki_clean | musique_clean | hotpotqa_clean | squad_clean | macro |
|---|---|---|---|---|---|---|---|
| selector headroom at B=6 | **+0.0601** | **+0.0240** | **+0.0080** | **+0.0115** | **+0.0140** | **+0.0070** | **+0.0208** |

That is 3.0× the standing router's entire gain over BASE (+0.0069), and it needs no change to P, B, `M_struct`, `M_ret` or `MAX_HOPS` — it is available inside the frozen contract, from the candidate pool the router already builds. Freezing L1 now would freeze that on the table.

**2. The winning router violates the architecture this phase was asked to impose.** F6's third vote is the same evidence family as its first. Deleting it (`ES`), fusing it (`LEXONE`) and re-exposing the families without it (`DS3`) are each significantly worse on 4 of 6 corpora. The measurement and the architectural ideal disagree, and freezing means ratifying the inconsistency deliberately.

What is now closed, and should not be re-opened: DIRECT full-corpus fusion; partition-graph PPR in all three semantics; PPR precomputation as a lever; and `B` as a lever.

What remains open, in order of expected value:

- **The eviction asymmetry — the one untested thing that targets the measured failure.** T17 shows canonical-only incumbents are retained at rates as low as 0.0002, and T20 shows the selector is leaving +0.0208 macro on the table inside its own pool. A scoring rule that lets a canonical-only incumbent hold its slot against a single-channel challenger is a parameter-free change to the RRF expression, uses no gold, no learning and no dataset identity, and is the natural next experiment.
- **The pool needs slots, not reach.** STEP 7 shows the graph makes REACH zero on all six corpora, and that converting it would push MetaQA's `CAP_B6` class from 21 to 215. A wider pool without more replacement slots cannot pay. Both are contract changes and need an explicit ruling.
- **P=50 itself.** `CAP_P50` is nonzero only on MetaQA (26 queries, 24 of them hop3). Small, and genuinely impossible under the current contract.

If the answer to all three is "no contract change", then the eviction rule is the whole remaining L1 program, and `L1_FROZEN = YES` becomes the right call as soon as it is tested.

---

## ARTIFACTS

```
results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PPR_SIMPLIFY/
  FINAL_REPORT.md   this report
  TABLES.md         every table, regenerated from the round JSONs
  diag/round_a.json STEP 0 gates, STEP 1 channels, STEP 2 graph stats
  diag/swap_dump.json  per-query swap provenance dumps
  diag/round_b.json STEP 5 B-sweep + DIRECT fusion + STEP 1 independence + latency
  diag/round_c.json STEP 4 precomputation, STEP 3 P3 + controls, STEP 7 decomposition
  diag/round_d.json STEP 6 structural bake-off, STEP 7 reach probe
  diag/round_e.json STEP 1 one-vote-per-family variants
  diag/oracle.json  STEP 7 addendum: selector headroom vs pool failure
  diag/eviction.json  STEP 0 quantified boundary turnover
  diag/alpha_sensitivity.json  STEP 10 robustness
  graph/pgraph_{ds}.npz  cached canonical partition transition graphs
  ppr/mass_{ds}.npz      cached personalization + P1/P2 masses
  logs/                  run logs
scratchpad/_l1pp_core.py  shared substrate: channels, partition graph, PPR contract, scoring
scratchpad/_l1pp_a.py     ROUND A   scratchpad/_l1pp_b.py  ROUND B
scratchpad/_l1pp_c.py     ROUND C   scratchpad/_l1pp_d.py  ROUND D
scratchpad/_l1pp_e.py     ROUND E   scratchpad/_l1pp_oracle.py / _l1pp_evict.py / _l1pp_alpha.py
scratchpad/_l1pp_tables.py / _l1pp_final.py  report generation
```
