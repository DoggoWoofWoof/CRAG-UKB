# PARAMETER-FREE EXACT-P50 STRUCTURAL RECOVERY

```
NODE_DISCOVERY_LIMITED            = YES   (dominant: 91.6 % hop2, 86.7 % hop3)
AGGREGATION_LIMITED               = NO    (7.0 % / 11.2 %; no aggregation beat the current one)
SWAP_LIMITED                      = NO    (1.4 % / 2.1 %; the rules that gain on MetaQA
                                           regress significantly on all five other corpora)
CURRENT_SDIR_IS_TRULY_SEQUENTIAL  = NO
BEST_STRUCTURAL_NODE_METHOD       = N0_STATIC   (the existing score; N1/N2/N3 all lose)
BEST_NODE_TO_PARTITION_METHOD     = P0_S4       (the existing aggregation; P1-P4 all lose)
EXACT_P50_METAQA_HOP2             = 0.7297      (unchanged)
EXACT_P50_METAQA_HOP3             = 0.2583      (unchanged)
```

The audit conclusion is recorded and stands: **U_PC5@256 is not a valid final L1 solution** — on MetaQA
it exposes 76.4 % of corpus nodes, so MetaQA hop3 selective routing remains FAILED. Every number
below is an **exactly-50-partition** output. No 256-partition result is offered as an outcome; the
one place a larger budget appears (STEP 7) is labelled a diagnostic.

Tables: [TABLES.md](TABLES.md). Raw: `diag/*.json`.
Code: `scratchpad/_l1sr_{diag,reach,seq,path,eval,run,step79,cache6,swap6,curve6}.py`.

**Bottom line.** Sequential residual reasoning was implemented exactly as specified and it does not
work — but the mechanism diagnostic shows *why*, and the reason is not the one STEP 3 assumed. The
frozen score is static, but making it sequential is close to a no-op (it changes the gold edge's rank
in only 14-20 % of cases, by ±0.008 in percentile rank). The binding constraint is that **the
displacement-cosine score is at chance at the hop where the beam has to commit** — the gold edge sits
at percentile rank 0.5151 at path position 1 and 0.4177 at position 2, against a 0.5 chance
baseline. It only becomes informative at position 3 (0.1718), by which point the beam has already
discarded the path.

---

## STEP 1 — rank-transfer diagnostic (no new algorithm)

Gold NODE rows were reconstructed from `query_ids_all.json` via `id2row` (the frozen cache stores
only gold *partitions*); the reconstruction reproduces the cached gold partition sets on
**1998/1998 queries exactly**. Classification uses the frozen operational depths, not gold-tuned
cuts: `M_struct = 64` (where `struct_aggregate_full` stops reading `s_node`) and `B = 6` (the swap
slots `f6_select` fills).

| MetaQA | missed gold partitions | A NODE_DISCOVERY | B AGGREGATION | C SWAP |
|---|---|---|---|---|
| hop2 | 872 | **799 (91.6 %)** | 61 (7.0 %) | 12 (1.4 %) |
| hop3 | 4,939 | **4,281 (86.7 %)** | 555 (11.2 %) | 103 (2.1 %) |

Rank distributions for those same partitions (T1). The decisive column is the first: on hop3,
**4,081 of 4,939 missed partitions have no gold node anywhere in the 256-deep structural list** — not
deep in it, absent from it. And the A class is dominated by partitions that never enter the candidate
pool at all (94.1 % hop2, 96.9 % hop3); only 3-6 % of it is a gold boundary incumbent evicted for
having canonical evidence only.

### STEP 1b — is that reachable or unreachable?

"No gold node in the list" is only actionable if the node was reachable and the beam dropped it. The
frozen traversal was re-run with the beam removed and every other frozen constant kept
(`SEED_K=5`, `MAX_HOPS=3`, `DEG_CAP=300`):

| | total | A1 UNREACHABLE | A2 BEAM_PRUNED |
|---|---|---|---|
| both hops | 5,594 | 585 (10.5 %) | **5,009 (89.5 %)** |
| hop2 | 744 | 48 (6.5 %) | 696 (93.5 %) |
| hop3 | 4,850 | 537 (11.1 %) | 4,313 (88.9 %) |

The 3-hop closure is only **3,774.5 nodes (9.4 % of the corpus)**, growing 27 → 579 → 3,164 per hop.
So the reachable set is small, it contains the gold, and the beam is throwing it away. That is what
justified STEP 3.

## STEP 2 — audit of the current `s_dir` implementation

Answered from executable code (`scratchpad/_ta_prepartition.py:179`, `expand_dir`), not documentation.

| # | question | answer |
|---|---|---|
| 1 | query representation at hop1 | `r_q = residual(q, seeds, Xn)` — q minus the SVD span of the seed embeddings, renormalised |
| 2 | at hop2 | **the same `r_q`** |
| 3 | at hop3 | **the same `r_q`** |
| 4 | is the same residual reused at every hop? | **YES** — `rq` is assigned once before `for hop in range(MAX_HOPS)` and never reassigned |
| 5 | does hop2 depend on which hop1 edge was selected? | only *topologically* (frontier membership). The score `(delta/‖delta‖) @ rq` has no dependence on the incoming edge |
| 6 | does hop3 depend on the complete preceding path? | **NO** |
| 7 | is path provenance preserved until partition scoring? | **NO** |
| 8 | where exactly is path information discarded? | twice: `np.maximum.at(best_s, inv, S)` keeps the max over incoming edges and drops the parent; then `added = [v for _, v in sorted(order, ...)]` returns a flat node list. `_l1kb_paths.expand_dir_prov` does keep parent pointers, but it is not what `_l1ps_cache.py` writes |

**`CURRENT_SDIR_IS_TRULY_SEQUENTIAL = NO`.** The gate for STEP 3 was met.

## STEP 3 — true sequential residual reasoning

Implemented exactly as specified: `r_0 = residual(q, seeds, Xn)`;
`directional_h = cos(r_{h-1}, delta_h)` per path; on accepting an edge,
`r_h = r_{h-1} − proj_{delta_h}(r_{h-1})`, renormalised. Every beam path carries its own current
node, residual, path, per-hop directional scores and path confidence. No learning, no dataset
identity, no gold at inference. Frozen constants unchanged (beam 64, `MAX_HOPS=3`, `DEG_CAP=300`,
`SMAX=256`), and compute is matched — edges scored 3.01 M (N0) vs 2.97 M / 2.59 M / 2.70 M.

**Control.** With the residual update switched off, the implementation reproduces the frozen static
beam's node set on **300/300** queries in the pilot and **1997/1998** in the full build; the single
exception (query 1647, hop3) differs by exactly one node out of 168 — a tie-break, not a mechanism.

### Why it does not work (T4)

Measured on real gold paths, with no beam and no ranking in the way: BFS from the frozen seeds gives
one shortest seed→gold path per reachable gold node, and each edge is scored under both residuals.
The reported quantity is the **percentile rank of the gold edge among its node's competitors** —
exactly what the beam prunes on. 0 = best, **0.5 = chance**.

| path position | n | mean out-edges | STATIC | SEQUENTIAL | qsim (control) | rsim (control) |
|---|---|---|---|---|---|---|
| 1 | 3,636 | 8.3 | **0.5151** | 0.5151 | 0.4736 | 0.5132 |
| 2 | 3,127 | 39.9 | **0.4177** | 0.4245 | 0.4521 | 0.4174 |
| 3 | 2,281 | 10.9 | 0.1718 | 0.1637 | 0.2924 | 0.1778 |

Three things follow.

1. **The residual update is nearly a no-op.** It changes the gold edge's rank in 19.6 % of cases at
   position 2 and 14.2 % at position 3, and the mean percentile shift is ±0.008. STEP 3's premise —
   that staticness is what loses the path — is **refuted**.
2. **The score is at chance where the beam commits.** At position 1 there are only ~8 out-edges and
   the beam keeps 64, so nothing is pruned and a chance-level rank costs nothing. Pruning happens at
   position 2, where the frontier is ~64 nodes × ~40 edges ≈ 2,560 candidates competing for 64 slots
   — and there the gold edge sits at percentile 0.4177. That is the mechanism behind STEP 1b's
   89.5 % beam-pruned.
3. **The controls do not rescue it.** Plain query-similarity is better at position 1 (0.4736) but
   *worse* at positions 2 and 3 (0.4521, 0.2924), and residual-similarity is indistinguishable from
   the displacement score. Neither is a drop-in improvement.

A partial cross-corpus replication on 2wiki (T4b) shows the same position-1 behaviour — static
0.4538, qsim 0.2364 — but with only 8 and 3 edges at positions 2 and 3 it cannot speak to the
pruning hop, and is reported as suggestive only.

## STEPS 4-6 — the scorer × aggregation grid at EXACT P50

Four node scorers × five aggregations, one shared beam per scorer, path provenance preserved until
aggregation. Only the structural ranking `SF` changes; the protected core (44), `B = 6`, the
retrieval channel, the canonical channel and the exactly-50 output are the frozen contract.
**Harness parity: N0 × P0 reproduces SAFE exactly** (verified query-by-query before any comparison).

MetaQA, all 20 cells (T5) — the frozen cell is the best of all twenty:

| | P0_S4 | P1_MAX_NODE | P2_TOP2 | P3_BEST_PATH | P4_MULTI_SEED |
|---|---|---|---|---|---|
| **N0 static** | **0.6612** | 0.6607 | 0.6597 | 0.6557 ᶜ | 0.6607 |
| N1 terminal | 0.6597 | 0.6587 | 0.6587 | 0.6567 | 0.6587 |
| N2 path-min | 0.6552 ᶜ | 0.6552 ᶜ | 0.6557 | 0.6562 | 0.6552 ᶜ |
| N3 path-geo | 0.6567 | 0.6567 | 0.6572 | 0.6567 | 0.6567 |

ᶜ significantly worse than SAFE (exact McNemar, p < 0.05). Nothing is better; nothing is even equal.
At the node level the sequential beams find *fewer* gold nodes at matched compute — MetaQA hop3
gold nodes discovered: N0 625, N1 541, N2 439, N3 494.

Across all six corpora (T6), the aggregations computable from the frozen structural cache are
neutral-to-worse everywhere, largest deviation −0.0015:

| | metaqa | webqsp | 2wiki | musique | hotpot | squad |
|---|---|---|---|---|---|---|
| P0_S4 (frozen) | **0.6612** | **0.7646** | **0.9435** | **0.9635** | 0.9505 | **0.9875** |
| P1_MAX_NODE | 0.6607 | 0.7646 | 0.9430 | 0.9625 | 0.9510 | 0.9870 |
| P2_TOP2_SUPPORT | 0.6597 | 0.7639 | 0.9430 | 0.9625 | 0.9510 | **0.9875** |

**`AGGREGATION_LIMITED = NO`**, on two independent grounds: it is 7-11 % of the failures by the STEP
1 classification, and no alternative aggregation converts even that.

## STEP 7 — candidate-budget diagnostic (does the curve move left?)

Same contract shape, only the budget varied: `final_k = prot(44) ∪ top-(k−44)` of the same score
order, so k = 50 is exactly the P50 output. The candidate pool is extended with the canonical
continuation purely so k = 128 can be filled — a continuation entry has canonical rank ≥ 50 and
therefore always scores below every boundary incumbent, so it cannot alter k = 50. At k = 256 the
pool is exhausted (215.8 partitions available on average), so that row is an upper bound, not a
budget.

MetaQA (T7):

| k | ALL | hop1 | hop2 | hop3 |
|---|---|---|---|---|
| **50** | **0.6612** | 0.9955 | **0.7297** | **0.2583** |
| 64 | 0.6842 | 0.9955 | 0.7673 | 0.2898 |
| 80 | 0.7047 | 0.9970 | 0.7838 | 0.3333 |
| 100 | 0.7377 | 0.9985 | 0.8258 | 0.3889 |
| 128 | 0.7718 | 0.9985 | 0.8874 | 0.4294 |
| (256) | (0.8278) | (0.9985) | (0.9339) | (0.5511) |

Per-step ALL improvement: 50→64 **+0.0230**, 64→80 +0.0205, 80→100 +0.0330, 100→128 +0.0341,
128→256 +0.0561. On hop3: +0.0315, +0.0435, +0.0556, +0.0405, +0.1217.

**The curve did not move left — and it has no knee to move.** The increments are flat to
*increasing*, so the structural gain available at 128/256 does not become available at 64-80; each
budget increment buys roughly what the previous one bought. This is the same "coverage tracks
exposure" shape the equal-exposure audit found, now measured on the partition-budget axis instead of
the node-exposure axis. The two KBs are the elastic corpora (metaqa 0.6612→0.7718, webqsp
0.7646→0.8576 by k=128); the four text corpora are nearly saturated at k=50 and gain +0.004 to
+0.011 over the same range (T7b).

## STEP 8 — direct node → partition transfer

The first pass of this measurement was wrong and is corrected here. `struct_aggregate_full` reads
only the first `M_struct = 64` structural nodes, so the S4 list is at most ~47.5 entries long and
"partition in S4 top-64/80/100" was trivially true. Re-measured against S4 aggregated over the full
256-node list (mean list length 84.5) and against the final P50:

| | n | median partition rank | p90 | absent |
|---|---|---|---|---|
| gold node → S4@256 partition rank | 4,318 | **12.0** | 54.0 | **0** |

| top-8 structural gold nodes (n = 598) | S4@256 top-50 | top-64 | top-80 | top-100 | **in final P50** |
|---|---|---|---|---|---|
| | 100.0 % | 100.0 % | 100.0 % | 100.0 % | **68.1 %** |

Node → partition transfer is clean: every gold node's partition is present, at median rank 12, and
every top-8 gold node's partition is inside the structural top-50. The loss is not in this step.

## STEP 9 — swap rules (run last, and the reason it does not promote)

Three rules on the frozen ranking: **A** the current F6 control, **B** direct strong-structural
challenger replacement, **C** one deterministic pairwise incumbent-vs-challenger evidence rule (a
candidate outranks another iff it carries strictly more evidence *channels*). No RRF search, no
learned weights, no threshold grid.

On MetaQA both challengers beat F6 significantly:

| rule | ALL | hop1 | hop2 | hop3 | net vs F6 | p |
|---|---|---|---|---|---|---|
| A_F6 | 0.6612 | 0.9955 | 0.7297 | 0.2583 | — | — |
| **B_STRUCT_DIRECT** | **0.6712** | 0.9955 | **0.7432** | **0.2748** | **+20** | 0.00032 |
| C_PAIRWISE_EVIDENCE | 0.6667 | 0.9955 | 0.7342 | 0.2703 | +11 | 0.0192 |

On the other five corpora both are significantly **worse** (T8):

| net vs F6 | metaqa | webqsp | 2wiki | musique | hotpot | squad |
|---|---|---|---|---|---|---|
| B_STRUCT_DIRECT | **+20** ✓ | −16 ✗ | −19 ✗ | −26 ✗ | −44 ✗ | −17 ✗ |
| C_PAIRWISE_EVIDENCE | **+11** ✓ | −3 | −13 ✗ | −21 ✗ | −28 ✗ | −13 ✗ |

✓/✗ = significant gain/loss. Letting structure decide the boundary outright is a **MetaQA-specific**
win that costs more everywhere else — including on WebQSP, the *other* KB, which rules out a
"KB versus text" reading. This reproduces the finding that made F6 the fixed-P50 router in the first
place: symmetric competition is universal, single-channel dominance is not.

**`SWAP_LIMITED = NO`.** No promotion. F6 stays.

---

## Verdict

The success criterion was "substantially better MetaQA hop2/hop3 at exact P50". The delivered change
is **0.0000** on both, because every mechanism in the specified families is neutral or worse than
what is already frozen. Reported as such rather than dressed up: this is a negative phase, and the
directive's own instruction not to celebrate +0.002 applies with more force to +0.0000.

What the phase did establish, and what it costs to ignore:

- The failure is **node discovery**, it is **89.5 % addressable in principle** (the gold nodes are
  inside a 3,774-node closure the beam already visits), and it is **not** caused by the score being
  static. It is caused by the displacement-cosine score being **at chance at path position 1-2**,
  which is where the beam must prune ~2,560 candidates down to 64.
- Neither of the two natural geometric alternatives measured as controls (query similarity,
  residual similarity) is better at the pruning hop, so this is not a one-line substitution.
- The node→partition and swap stages are **not** the bottleneck, and the one swap rule that gains on
  MetaQA is corpus-specific and regresses on the other five corpora, the other KB included.
- The partition-budget curve has **no knee**: MetaQA hop3 buys roughly +0.03 to +0.05 coverage per
  16-28 partition increment all the way out, so no reordering of the existing evidence makes the
  128-budget gain available at 64-80.

Nothing was promoted, nothing retired, no frozen constant changed. `L1_FROZEN` is unchanged and the
L1→L2 handoff stays paused. No TEST, no L2, no L3, no encoder change, no new proposal family, no
U_PC5 redesign, no dataset-specific branch. `ONLINE_GRAPH_EDGES_TOUCHED` at selection time = 0 for
every mechanism reported. The frozen mechanism reads the precomputed expansion cache; N1/N2/N3
rebuild that expansion, but at the same offline/index-time stage as the frozen `expand_dir` and with
matched edge budgets, so no rule scores a candidate by walking the graph at query-selection time.
