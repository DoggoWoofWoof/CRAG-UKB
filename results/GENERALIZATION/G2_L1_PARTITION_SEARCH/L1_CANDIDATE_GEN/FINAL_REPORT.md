# L1 CANDIDATE GENERATION PHASE — FINAL REPORT

Companion tables: [TABLES.md](TABLES.md) (T1–T17). Raw JSON under `diag/`.

Selector frozen at **R0 / B6_S4_F6_Ms64_Mr32** throughout. Nothing was promoted. No selector scoring
was modified. No L2, L3, TEST, training or new encoder was touched. No gold quantity enters any
proposal family — every family is built from cached retrieval rankings, the cached node lists, the
cached S4 aggregation, cached PPR mass, or corpus-side partition-graph tables.

---

## VERDICT

**`CANDIDATE_GENERATION_BOTTLENECK_RESOLVED = NO`** — but not for the reason the phase was set up to
test, and the failure is now located precisely.

Candidate generation as a *discovery* problem is **solved**. One parameter-free universal proposer at
a single global budget lifts the MetaQA hop3 candidate-pool oracle from 0.3634 to **0.6351**, which
is **85.8%** of the entire pool-attributable headroom (ceiling 0.6802 at B=6), and lifts hop2 to
97.9% of its ceiling. Every text corpus reaches 0.9825–1.0000. It touches **zero graph edges at query
time**.

Candidate generation as a *conversion* problem is **blocked, provably**. Under the frozen R0/F6
selector the actual result is bit-identical to SAFE on **every** configuration tested — 2,129,285 new
candidates offered across 11,417 queries on all six corpora, **0 ever selected**, 11,417/11,417
selections identical (T16). This is not a tuning shortfall; it is an algebraic property of the frozen
scoring rule, verified empirically and derived below.

The bottleneck named in the directive's motivation (`candidate-pool headroom ≈ +0.317`) is real and
now reachable, but it is **not spendable without a selector change**, which this phase is forbidden
to make. The phase therefore ends with a ruling request rather than a promotion.

---

## The blocker, stated exactly

For a candidate `q` outside `base50` that is not in the frozen challenger set `SF ∪ RF`, the frozen
selector assigns

```
s(q) = 1/(K0 + cpos[q])   if q is ranked canonically, else 0        (spos, rpos absent ⇒ 0.0 each)
```

`q ∉ base50` forces `cpos[q] ≥ P = 50`, so `s(q) ≤ 1/(K0+P) = 1/110 = 0.00909091`. Every boundary
incumbent `p ∈ bnd = base_rank[P−B:P]` has `cpos[p] ≤ P−1 = 49`, so `s(p) ≥ 1/(K0+P−1) = 1/109 =
0.00917431`, before any structural or retrieval support it may also carry. There are exactly `B`
incumbents competing for exactly `B` slots, so every such `q` is strictly dominated — **for any B, on
any corpus, at any proposal budget**.

The only way for a proposal to carry non-canonical evidence is to be in `SF ∪ RF`, and anything in
`SF ∪ RF` is already in the pool. The frozen candidate pool is therefore **closed**: expanding it
cannot change the selector's output.

T16 confirms this empirically rather than by assertion. The realised minimum score margin on
2Wiki, MuSiQue, SQuAD and WebQSP is `8.34e-05`, which is exactly `1/109 − 1/110`; on MetaQA and
HotpotQA it is larger because their incumbents also carry structural or retrieval support.

I also tested the one candidate-generation knob that *is* convertible — the generator's aggregation
depth `M_struct` / `M_ret`, which extends the `spos` / `rpos` rank lists themselves so that new
candidates arrive with channel evidence, while leaving the scoring rule and all existing ranks
untouched (T17). It converts nothing either. On MetaQA the pool oracle rises 0.3634 → 0.4144 while
ACTUAL moves 0.2583 → 0.2583 (+4 gained / −4 lost); on WebQSP the oracle rises 0.7886 → 0.8125 while
ACTUAL *falls* 0.7646 → 0.7597 (+8/−15); on 2Wiki the oracle rises 0.9515 → 0.9690 while ACTUAL falls
0.9435 → 0.9410 (+4/−9). No configuration is positive and none is significant. Deeper channels admit
and evict at the same rate.

Two independent mechanisms, one conclusion: **the residual MetaQA gap is a selector problem, not a
pool problem.** That is consistent with the ceiling audit's `RANKING_LIMITED` verdict, and it
converts that verdict from a diagnosis into a constraint.

---

## STEP 1–3 — where the missing partitions actually live

Ten inference-safe proposal families were built (T2). Eight are the directive's A–H; a ninth and
tenth were added after STEP 4 measured what the failures were made of:

- **I_CANON_CONT** — the *fused* canonical ranking beyond the top-50, `base_rank[P:]`. STEP 4 showed
  77.08% of MetaQA hop3's missing partitions already sit inside the canonical top-200 and are simply
  not in the pool, so the fused continuation had to be a family in its own right. It is the single
  best proposer on WebQSP.
- **G_GRAPH_NBR_NORM** — the volume-normalised variant of the neighbour table, to separate hub
  effects from real structural adjacency.

**Measurement caveat that must travel with every recall number.** On MetaQA (npart = 401) families A,
B and F have all 351 non-base50 partitions available, so at M = 256 they propose 73% of the universe;
G/H propose 66–85. Recall is not comparable across families without T2's `frac universe` column. The
union recommended below is deliberately built from selective families.

**Primary metric (ALL_REQUIRED_PARTITIONS_PROPOSED@M, T3/T4).** The best single proposer differs by
corpus: F_PPR_REACH on MetaQA hop3 (0.643 @256); I_CANON_CONT on WebQSP (0.648);
B_SPLADE_PARTITION on 2Wiki (0.711); C_NODE_DENSE_CONT on HotpotQA (0.465); A_DENSE_PARTITION on
SQuAD. **No single family is universal.**

**Complementarity (T6/T7).** The greedy discovery orders are corpus-specific and share almost no
prefix — MetaQA `B(+147) → F(+61) → G(+14)`, WebQSP `I(+110) → B(+23) → D(+13)`, 2Wiki `D(+36) →
C(+15) → G(+8)`, HotpotQA `C(+24) → D(+8)`, MuSiQue `F(+45)`, SQuAD `C(+8)`. Pairwise Jaccard on
MetaQA confirms these are genuinely different signals: B vs F 0.34, B vs G 0.26, B vs A 0.19, B vs C
0.15 (B vs I is 0.60, the one strongly redundant pair — the fused canonical ranking is largely SPLADE
on this corpus). This is the empirical justification for a *union* proposer rather than a best-family
proposer.

---

## STEP 4 — anatomy of the 398 MetaQA hop3 pool-limited queries (T8/T9)

398 queries, 2,596 required gold partitions the current pool cannot see. Grouped strictly from the
measurements, not from categories chosen in advance:

| group | n | frac |
|---|---|---|
| inside canonical top-200 but outside the pool | 2001 | 0.7708 |
| proposed, but only outside canonical top-200 | 563 | 0.2169 |
| graph-reachable yet proposed by no family at M=256 | 32 | 0.0123 |
| unreachable in the partition graph | **0** | **0.0000** |

Two facts dominate.

1. **Every one of the 2,596 partitions is graph-reachable** — 2,415 at distance 1 from a retrieval
   seed partition and 181 at distance 2. There is no reachability failure at all.
2. **77% of them are already canonically ranked inside the top-200.** They are not undiscovered; they
   are unadmitted. The current pool (`bnd` + S4 + ret continuation) simply never offers canonical
   ranks 50–199.

Channel ranks on those partitions (T8): Dense and SPLADE are full rankings so they list every
partition, but they list these ones *deep* — median rank 189 and 167 of 401, and p10 67 and 52, so
even the best decile is outside the P=50 budget. The sparse channels barely see them at all: S4 ranks
16.3%, node-dense 30.2%, node-SPLADE 33.3%. That is the shape of a ranking failure, not a coverage
failure, and it matches the ceiling audit's finding that the *worst* required partition sits at median
canonical rank 218 of 401.

---

## STEP 5 — PPR as a proposer only (T10)

PPR contributes candidate membership only; no PPR score reaches F6/R0, and PPR scoring was not
re-opened. As a **proposer** it is genuinely strong, in direct contrast to its behaviour as a selector
vote: on MetaQA hop3 it is the best single family at every budget (pool oracle 0.4279 / 0.4745 /
0.5090 / 0.5601 at M_ppr = 16/32/64/128 against a current-pool 0.3634), and it is the family greedy
discovery selects first on MuSiQue (+45 queries, T6). The architectural separation the directive insisted on is therefore confirmed by
measurement: **PPR is harmful as a ranking signal and useful as a discovery signal.**

It is, however, **not necessary**. The recommended fully-precomputed union reaches 0.6351 on MetaQA
hop3 without PPR, above PPR-alone at any budget and equal to the best union containing it. Since PPR
requires query-time propagation and every alternative family is a cache lookup, PPR is excluded from
the recommendation on latency grounds, not on quality grounds.

Actual result under the frozen selector at every M_ppr: **unchanged, +0/−0.**

---

## STEP 6 — can the generator be a pure cache lookup? (T11)

Yes for the structural channel, and the answer comes with an important negative.

`ONLINE_GRAPH_EDGES_TOUCHED = 0` is achieved by construction for every family in the recommendation:
A/B/I are cached partition rankings, C/D are cached node lists mapped through `hard`, and G is a
lookup of ≤ 5 seed rows in a corpus-side top-K neighbour table followed by an integer merge. The
partition graph `G = Mᵀ A M`, the neighbour tables, and the depth-2/3 compositions are all built
corpus-side and cached.

The negative: **structural reachability is close to vacuous on these corpora.** The partition graphs
are dense — mean partition degree 158 of 401 on MetaQA, 471 of 5,074 on HotpotQA — so an online
depth-2 frontier reaches 100% of the missing partitions on MetaQA, WebQSP, 2Wiki *and* HotpotQA while
proposing essentially the whole universe (351/351 on MetaQA, 5,024/5,074 on HotpotQA) and touching
55,933 / 110,095 / 30,733 / 1,079,499 edges respectively. Precomputed tables reproduce that recall
exactly at zero edges (K=128, depth 2: 1.0000 / 0.9915 / 1.0000 / 1.0000) but inherit the same
uselessly-large proposal set. At a *selective* budget the tables are weak: K=32 depth-1 recall is
0.3790 on MetaQA, 0.1396 on WebQSP, 0.0135 on HotpotQA.

So the graph's contribution to candidate generation is an **ordering** contribution, not a
reachability one, and the top-K weight ordering is a mediocre ordering. This is why G enters the
recommended union as one voice among five rather than as the backbone.

---

## STEP 7 / STEP 12 — the recommended universal proposer

```
U_PC5  =  round-robin interleave, deduplicated, cut at M_TOTAL
          [ B_SPLADE_PARTITION, I_CANON_CONT, C_NODE_DENSE_CONT,
            D_NODE_SPLADE_CONT, G_GRAPH_NBR_RAW ]
M_TOTAL = 256          one global budget, identical on all six corpora
```

Zero learned parameters, no dataset identity, no per-corpus budget, no learned selector.
`ONLINE_GRAPH_EDGES_TOUCHED = 0`; query-time cost is five cached-list reads, ≤ 5 × 32 integer reads
from the neighbour table, and a merge bounded by 256.

Selected under the STEP 12 criteria in the stated priority order (T12): it is tied-best on MetaQA
hop3 (0.6351, tied with U_ALL), outright best on MetaQA hop2 (0.9474, where U_ALL is 0.9399), and is
the only union that is simultaneously top-of-table and fully precomputed. Its one concession is
WebQSP, where U_RET2_NODE reaches 0.9098 against U_PC5's 0.9027 — a 10-query difference on criterion
3, taken in exchange for criteria 1 and 2 which rank above it, and for full precomputability.

| corpus | slice | SAFE | current-pool oracle | **U_PC5 pool oracle @256** | ACTUAL frozen |
|---|---|---|---|---|---|
| metaqa | hop3 | 0.2583 | 0.3634 | **0.6351** | 0.2583 |
| metaqa | hop2 | 0.7297 | 0.8033 | **0.9474** | 0.7297 |
| metaqa | ALL | 0.6612 | 0.7212 | **0.8604** | 0.6612 |
| webqsp | ALL | 0.7646 | 0.7886 | **0.9027** | 0.7646 |
| 2wiki_clean | ALL | 0.9435 | 0.9515 | **0.9890** | 0.9435 |
| musique_clean | ALL | 0.9635 | 0.9750 | **1.0000** | 0.9635 |
| hotpotqa_clean | ALL | 0.9505 | 0.9645 | **0.9825** | 0.9505 |
| squad_clean | ALL | 0.9875 | 0.9945 | **1.0000** | 0.9875 |

Text behaviour is preserved in the strict sense the directive asked for: the ACTUAL column is
untouched everywhere, and the pool oracle only ever rises.

---

## STEP 11 — B revisited against the expanded pool (T15)

The ceiling audit measured B as non-binding, but it measured it against the *old* pool. Re-run once,
as instructed, against U_PC5 at M=256:

| corpus | B=6 | B=8 | B=12 |
|---|---|---|---|
| metaqa hop3, expanded-pool oracle | 0.6351 | 0.6562 | 0.6937 |
| webqsp, expanded-pool oracle | 0.9027 | 0.9084 | 0.9147 |
| 2wiki / musique / hotpot / squad | unchanged | unchanged | unchanged |

**The earlier conclusion is overturned exactly where the directive warned it might be.** Against the
old pool B bought +0.0030 on MetaQA hop3 in total; against the expanded pool it buys +0.0586 from
B=6 to B=12, and +0.0120 on WebQSP. B is non-binding only while the pool is starved. It remains
irrelevant on all four text corpora.

Two cautions. First, ACTUAL is still invariant at every B, for the reason in the blocker section.
Second, SAFE itself *degrades* as B grows on the graph corpora (WebQSP 0.7646 → 0.7618 → 0.7555;
2Wiki 0.9435 → 0.9430 → 0.9420), because a larger B shrinks the protected core and a weak selector
pays for the extra freedom. Raising B is only safe *jointly* with a selector that can use it.

---

## STOP CONDITION — the eight questions

**1. Which source actually finds the missing KB partitions?**
No single one. MetaQA hop3: PPR reach (ALL@256 = 0.643), then SPLADE partition ranking (0.533) and
the fused canonical continuation (0.538). WebQSP: the fused canonical continuation (0.648), then
SPLADE (0.594). Structure-derived families (E, G, H) are never the best on any corpus. The dominant
finding is that on MetaQA hop3, 77% of the missing partitions were never lost — they sit in canonical
ranks 50–199 and the pool simply does not reach that far.

**2. Is there complementarity, or is one source enough?**
Real complementarity. Greedy discovery orders share almost no prefix across corpora, and pairwise
Jaccard on MetaQA is 0.15–0.40 between the productive families (the one redundant pair is
SPLADE vs fused-canonical at 0.60). A union beats every single family on five of six corpora.

**3. How much of MetaQA's remaining gap closes at M ≤ 256?**
Of the pool-attributable headroom, **85.8% on hop3** (0.3634 → 0.6351 against a 0.6802 ceiling),
**97.9% on hop2**, **89.4% aggregate**. Of the *end-to-end* gap to SAFE, **0.0%** — none of it
converts.

**4. Does PPR belong in the architecture, but only as a proposer?**
As a proposer it is validated: best single family on MetaQA hop3 at every budget, and greedy
discovery's first pick on MuSiQue,
which is the opposite of its behaviour as a selector vote. But it is not required — a
zero-online-edge union matches or beats it — so on the STEP 12 latency and precomputability criteria
it stays out of the recommendation. The architectural separation the directive asserted is confirmed;
the practical answer is that we do not need to pay for it.

**5. Can candidate generation be fully precomputed?**
Yes. The recommendation is cache-lookup-plus-merge with `ONLINE_GRAPH_EDGES_TOUCHED = 0`. The caveat
from T11 is that the *structural* part of it is nearly information-free on these corpora, because the
partition graphs are dense enough that depth-2 reachability is universal; the precomputed tables
reproduce online discovery exactly, but online discovery was not worth much.

**6. Does the frozen selector convert the new evidence into real gains?**
**No — zero, everywhere, provably.** 2,129,285 new candidates, 11,417 queries, six corpora, 0
selected, all selections bit-identical. The convertible depth knob converts nothing either. See the
blocker section.

**7. Does B become the limiting factor once candidate generation improves?**
Yes on the two KB-like corpora, and this reverses the earlier reading. MetaQA hop3 pool oracle
0.6351 → 0.6937 and WebQSP 0.9027 → 0.9147 from B=6 to B=12, against +0.0030 / 0.0000 under the old
pool. No on all four text corpora. B must not be raised on its own: SAFE degrades with B on the graph
corpora under the current selector.

**8. What is the simplest universal candidate contract?**
`U_PC5` at `M_TOTAL = 256`: round-robin over SPLADE partition ranking, fused canonical continuation,
node-dense continuation, node-SPLADE continuation, and the precomputed depth-1 partition-neighbour
table. Five cached lists, one integer merge, one global budget, zero learned parameters, zero online
graph edges, identical on all six corpora.

---

## What this phase does NOT claim

- Nothing is promoted. SAFE remains `R0 / B6_S4_F6_Ms64_Mr32`; `R1_LIFT` remains a documented
  selector ablation and was not touched.
- No end-to-end gain is claimed. Every ACTUAL number in this report equals SAFE.
- The recall figures for families A, B and F on MetaQA are inflated by near-exhaustive proposal sets
  and must be read alongside T2.
- MuSiQue (npart 136) and SQuAD (npart 190) reach 1.0000 pool oracle partly because M=256 approaches
  their whole universe; their agreement is weak evidence.
- The pool oracle is a perfect-selector bookend, not an achievable score.
- `B=8/12` was measured once, as instructed, and is reported, not recommended.

---

## Ruling requested

Candidate generation has done its job and cannot do more inside the current contract. The measured
+0.2717 of MetaQA hop3 pool oracle (and +0.1441 hop2, +0.1141 WebQSP) is sitting in the pool,
unspendable, because a candidate that carries only canonical evidence is arithmetically incapable of
displacing a boundary incumbent under R0/F6.

Three mutually exclusive directions follow, and the choice is yours:

1. **Open the selector.** The minimal change consistent with everything measured is to let a
   proposal's canonical rank compete on equal terms rather than against an incumbent's structurally
   guaranteed floor. This is a scoring change and is out of scope here.
2. **Ship the generator and leave B and the selector alone.** Zero risk, zero gain today; the pool
   work is banked for whenever the selector is opened.
3. **Retire the pool line for L1** and accept `RANKING_LIMITED` as final, treating the +0.317 as
   unreachable under the frozen contract.

`CANDIDATE_GENERATION_BOTTLENECK_RESOLVED = NO`
`L1_FROZEN = NO`
`DISCOVERY_SOLVED = YES` — `CONVERSION_BLOCKED_BY_FROZEN_SELECTOR = YES`

STOP.
