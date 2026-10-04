# L1 EQUAL-EXPOSURE AUDIT — coverage vs unique nodes exposed

**`L1_ROUTING_VERDICT = B. L1_ROUTING_PARTIAL`**
with an explicit sub-ruling: **on MetaQA hop3, `C. L1_ROUTING_FAILED` holds** — reaching useful
multi-hop coverage requires exposing most of the graph.

**The L1→L2 handoff stays paused.** The critique that prompted it is correct and is now quantified:
MetaQA has 40,151 nodes in 401 partitions, and **U_PC5@256 exposes 30,681 of them — 76.4 % of the
corpus, a node compression ratio of 1.31×.** Its headline pool oracle was measured under
near-exhaustive exposure and is not evidence of a routing layer.

Tables: [TABLES.md](TABLES.md) (T1–T10). Raw: `diag/exposure.json`.
Code: `scratchpad/_l1ex_{core,run,null,tables}.py`. No new router, no new proposal family, no new
scoring — existing methods only, re-measured on the axis that was never controlled.

---

## Method

Exposure is the **real node union over the hard partition membership** (`np.bincount(z["hard"])`),
never partitions × a nominal size. Partitions are disjoint, so the node count of a partition set is
the sum of its member counts. Coverage is one metric for every method — a query is covered iff
**all** its gold partitions are inside the exposed set — which makes a router output and a candidate
pool directly comparable on a single axis.

Methods audited, each truncated only by its own existing ranking (no gold-based truncation):
`CANON` (fused canonical Dense+SPLADE partition ranking), `SAFE` (the frozen `R0/B6_S4_F6` router
output, 50 partitions), `SAFE_POOL` (SAFE plus its F6 challengers), `U_PC5` (base50 then the U_PC5
proposals in u_rank order). A **random-exposure null** is included so "real routing" is measured
rather than asserted.

---

## 1. The partition layer is a fixed absolute node budget, not a compression ratio

| corpus | nodes | partitions | mean partition size | SAFE exposure | compression |
|---|---|---|---|---|---|
| musique | 13,672 | 136 | 100.5 | **36.87 %** | 2.71× |
| squad | 19,029 | 190 | 100.2 | **26.47 %** | 3.78× |
| metaqa | 40,151 | 401 | 100.1 | 12.54 % | 7.97× |
| 2wiki | 65,865 | 658 | 100.1 | 7.63 % | 13.10× |
| hotpot | 507,494 | 5,074 | 100.0 | 1.00 % | 100.02× |
| webqsp | 781,485 | 7,814 | 100.0 | 0.64 % | 155.61× |

Every corpus is partitioned at ~100 nodes per partition, so `P_MAIN = 50` is **~5,000 nodes on every
corpus** (measured range 5,022–5,074). The compression ratio is therefore decided entirely by corpus
size — the router has no say in it. That single fact explains most of what follows.

## 2. MetaQA — routing works at hop1, degrades at hop2, fails at hop3 (T3, T9)

| exposed nodes needed for coverage ≥ | hop1 | hop2 | hop3 |
|---|---|---|---|
| 0.30 | 417 (1.0 %) | 916 (2.3 %) | 5,936 (14.8 %) |
| 0.40 | 417 (1.0 %) | 916 (2.3 %) | **19,961 (49.7 %)** |
| 0.50 | 417 (1.0 %) | 1,920 (4.8 %) | **24,961 (62.2 %)** |
| 0.60 | 916 (2.3 %) | 2,927 (7.3 %) | **29,963 (74.6 %)** |

hop1 reaches 0.9955 at 12.5 % exposure; hop2 reaches 0.8574 at 25 %. hop3 needs **32× more of the
corpus than hop2** for the same 0.60 coverage. Per the stated criterion: **L1 routing has failed on
MetaQA hop3.**

There is no knee to find. Marginal return along the canonical curve is roughly proportional across
the whole usable range — 0.95 coverage per unit exposure from 4 %→12.6 %, then **0.54** from
12.6 %→25 % and **0.58** from 25 %→64 %, rising again only in the exhaustive tail. Coverage tracks
exposure; it does not outrun it.

The ranking is not *uninformative* — at 50 partitions the random-exposure null is hop3 **0.0246**
against SAFE's 0.2583, a 10.5× lift. The signal exists. It is simply nowhere near strong enough:
MetaQA hop3 queries need a mean of 5.78 gold partitions, and no existing method concentrates that
many into a small budget.

## 3. Equal-node-budget comparison (T5) — the routers do not beat plain canonical retrieval

MetaQA hop3, primary:

| node budget | CANON | SAFE | SAFE_POOL | U_PC5 |
|---|---|---|---|---|
| 2,500 | 0.2012 (6.0 %) | 0.2012 | 0.2012 | 0.2012 |
| 5,000 | 0.2553 (12.3 %) | 0.2568 | 0.2568 | 0.2553 |
| 10,000 | 0.3243 (24.8 %) | 0.2583 (12.5 %) | **0.3529** (22.1 %) | 0.3183 (24.8 %) |
| 20,000 | 0.4670 (49.7 %) | 0.2583 (12.5 %) | 0.3544 (22.2 %) | **0.4910** (49.7 %) |

WebQSP ALL, primary:

| node budget | CANON | SAFE | SAFE_POOL | U_PC5 |
|---|---|---|---|---|
| 2,500 | 0.6526 (0.3 %) | 0.6526 | 0.6526 | 0.6526 |
| 5,000 | 0.7576 (0.6 %) | 0.7618 | 0.7618 | 0.7576 |
| 10,000 | **0.8393** (1.3 %) | 0.7646 (0.6 %) | 0.7738 (0.9 %) | 0.7992 (1.3 %) |
| 20,000 | **0.8992** (2.5 %) | 0.7646 (0.6 %) | 0.7738 (0.9 %) | 0.8661 (2.5 %) |

On WebQSP, **plain canonical retrieval beats U_PC5 at every matched budget** — 0.8393 vs 0.7992 at
10 k, 0.8992 vs 0.8661 at 20 k. SAFE and SAFE_POOL simply run out of partitions at ~5 k and ~7.4 k
nodes and flat-line; they are fixed-size methods, not budget-elastic ones.

### What the routing apparatus itself is worth (T6)

At **identical** 50-partition exposure, SAFE against just taking canonical top-50:

| corpus | SAFE | CANON@50 | router buys |
|---|---|---|---|
| metaqa | 0.6612 | 0.6587 | +0.0025 |
| webqsp | 0.7646 | 0.7618 | +0.0028 |
| 2wiki | 0.9435 | 0.9375 | +0.0060 |
| musique | 0.9635 | 0.9565 | +0.0070 |
| squad | 0.9875 | 0.9805 | +0.0070 |
| hotpot | 0.9505 | 0.9345 | +0.0160 |

The whole S4/F6 apparatus is worth **+0.0025 to +0.0160** over reading the canonical ranking down to
the same node count. Whatever compression exists on the large corpora is delivered by canonical
Dense+SPLADE retrieval and the partitioning, not by the router.

## 4. Is U_PC5@256 Pareto-optimal, or a near-exhaustive high-recall point? (T7, T8)

Technically it is non-dominated on all six corpora. That fact is close to meaningless on four of
them:

| corpus | U_PC5@256 exposure | compression | ALL | CANON@256 ALL | diff for +20 % nodes |
|---|---|---|---|---|---|
| musique | **100.00 %** | 1.00× | 1.0000 | 1.0000 | +0.0000 |
| squad | **100.00 %** | 1.00× | 1.0000 | 1.0000 | +0.0000 |
| metaqa | **76.41 %** | 1.31× | 0.8789 | 0.8293 | +0.0496 |
| 2wiki | 46.55 % | 2.15× | 0.9880 | 0.9795 | +0.0085 |
| hotpot | 6.07 % | 16.47× | 0.9825 | 0.9740 | +0.0085 |
| webqsp | 3.93 % | 25.47× | 0.9147 | 0.9112 | +0.0035 |

On **musique and squad, U_PC5@256 is literally the entire corpus** — it ties CANON@256 because both
are "read everything". On MetaQA it is a 76 % scan. On WebQSP its frontier membership rests on
**+0.0035 coverage for +20 % nodes**, and below that point canonical dominates it outright
(CANON@128 reaches 0.8640 at 12,830 nodes; U_PC5@128 reaches 0.8555 at 17,860). Frontier
composition: WebQSP 9/13 points are CANON and only `U_PC5@256`/`U_PC5@full` survive; MetaQA 9/18 are
CANON. Only on **hotpot** is U_PC5 on the frontier at every budget — that is the one corpus where it
is a genuine coverage-per-node improvement rather than a scan.

**Answer: U_PC5@256 is a near-exhaustive high-recall point on 4 of 6 corpora, a marginal frontier
point on WebQSP, and a real frontier method only on hotpot.**

## 5. The global M budget is not universal in exposure semantics (T10)

The same `M_TOTAL = 256`, one global budget with zero learned parameters, exposes:

`musique 100.00 %` · `squad 100.00 %` · `metaqa 76.41 %` · `2wiki 46.55 %` · `hotpot 6.07 %` ·
`webqsp 3.93 %`

— a **25× spread in compression ratio** (1.00× to 25.47×) for a nominally identical budget. A budget
that means "read the whole corpus" on one dataset and "read 4 %" on another is universal only in its
parameter value, not in its compute or its exposure. It should not be described as a universal
budget without this qualification.

---

## Decision

**`L1_ROUTING_VERDICT = B. L1_ROUTING_PARTIAL`**

- **Large corpora — real routing.** hotpot ALL **0.9505 at 1.00 % exposure (100× compression)** and
  webqsp ALL **0.7646 at 0.64 % (156×)**, both far above their random-exposure nulls (0.0034 and
  0.0033). Extending to ~2–4 % lifts hotpot to 0.9790 and webqsp to 0.8992. These are useful
  coverage-vs-compression operating points.
- **Small corpora — no meaningful compression.** SAFE already exposes 36.9 % of musique and 26.5 %
  of squad because 50 partitions is a fixed ~5,000 nodes; U_PC5@256 exposes 100 % of both.
- **MetaQA hop3 — failed.** 0.50 coverage needs 62.2 % of the corpus, 0.60 needs 74.6 %. Sub-ruling
  **C** applies on this slice: multi-hop coverage requires exposing too much of the graph.
- **The router is not the source of what does work.** At identical exposure it is worth +0.0025 to
  +0.0160 over plain canonical truncation; on WebQSP canonical retrieval dominates U_PC5 at every
  budget except the last.

The success criterion is not redefined around pool oracle. Restated on the audited axis: **no current
parameter-free L1 method achieves a useful coverage-vs-compression tradeoff on MetaQA, and on
musique and squad the partition layer does not compress at all.** Where the tradeoff is genuinely
good — hotpot, and WebQSP at low budgets — it is carried by canonical retrieval plus the fixed ~5 k
node budget, not by the routing logic layered on top.

Nothing promoted, nothing retired, no method changed. L2, L3, TEST, training, PPR, encoders and new
proposal families all untouched.
