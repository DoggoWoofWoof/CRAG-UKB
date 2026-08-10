# L1 Universal Single-Model — Findings, Failure Analysis, and the L1/L2 Repositioning

_Partition FullCov@20 unless noted. "Single model" = ONE frozen encoder + ONE offset head trained
once on pooled (query→gold) triples across all 5 datasets (no per-dataset weights). Runner:
`src/experiments/l1_universal_head.py`. Seeds: splits `SPLIT_SEED=42`, head-init `INIT_SEED=42`._

## 1. Universal single-head comparison (frozen BGE-large)

| head | standalone | dense+head (fused mean) | ≥95 | musique | 2wiki | squad | metaqa | hotpot |
|---|---|---|---|---|---|---|---|---|
| dense only | — | 93.50 | — | 85.9 | 91.9 | 91.8 | 98.9 | 99.0 |
| **base** (plain offset) | 92.00 | **93.88** | 2/5 | 89.1 | 89.9 | 92.7 | 99.8 | 97.9 |
| hard (offset + hard-neg) | 91.74 | 93.61 | 2/5 | 88.4 | 89.7 | 92.0 | 99.8 | 98.2 |
| mix = **mlpT** (K-mixture) | 91.73 | 93.54 | 2/5 | 88.8 | 88.9 | 92.6 | 99.8 | 97.7 |

**Findings:**
- **`base` wins; mlpT is *last*.** So the answer to "is mlpT the best single head" is **no** — a plain learned
  offset beats the K-mixture universally. "It might not be mlpT" → confirmed, mlpT is the weakest.
- **Hard negatives don't transfer universally** (hard 93.61 < base 93.88). Their MVP status in the roadmap was
  a *per-dataset* effect; pooled across domains the hard-neg signal doesn't help.
- A single universal head adds only **+0.4 over dense** (93.5→93.88) and stays at **2/5 ≥95**.
- **Equal-RRF fusion actively dilutes the saturated datasets** — 2wiki 91.9→89.9, hotpot 99.0→97.9 — while
  helping musique (+3.2) and squad (+0.9). → fusion must be **best-of / confidence-gated**, not equal.

## 2. Failure analysis — the gap to 95 is 100% RANKING

From the per-dataset `rerank100` top-N sweep (BGE): **`voted_oracle@20 = 100` at N=200/350/500 on all 5.**
The gold's partition is ALWAYS in the vote pool with a min-cover ≤ 20 — so every miss is the correct
partition getting **voted but ranked below 20** (~5–10% of queries).

Per-dataset gap to 95 (BGE, best per-dataset config = best-of):
| dataset | best@20 | gap | failure character |
|---|---|---|---|
| 2wiki | 90.8 | **+4.2** | comparison Qs → 2 gold entities in *different* partitions; one weakly voted |
| squad | 92.4 | +2.6 | single-hop but semantically-far golds → gold nodes retrieved low → weak vote |
| musique | 93.2 | +1.8 | deep multi-hop → gold nodes 2+ hops from top retrievals → weak vote |

**Levers that DON'T close it (ruled out by data + prior work):**
- **Deeper voting HURTS** — eqrrf@20 drops with N (musique 93.2→92.0): more nodes → more noise partitions voted → gold ranks lower.
- **Learned rerankers fail** — attn/mlp over the vote features lose to plain RRF.
- **Membership / PPR dead** — more overlap dilutes; PPR over the near-complete partition graph dilutes.

**Levers that CAN (all single-model-preserving):**
1. **Stronger encoder (gte-Qwen2, 1536-d)** — the only untapped lever with headroom. Since the gold partition
   is voted but too *weakly*, a stronger encoder retrieves the gold's nodes higher → its partition out-votes the
   noise into top-20. *(gte L1 run pending — the decisive test of the encoder ceiling.)*
2. **best-of fusion (+ confidence gate)** — min-rank across dense+head keeps each retriever's strong votes
   instead of averaging → recovers the 2wiki/hotpot dilution.

## 3. THE MECHANISM — why offset/mlpT is huge at node-level but flat at L1 partition-level

- **Node/doc level (where offset WINS big):** the offset's job is to recover golds that are *relationally far*
  from the query in embedding space (KB hops, multi-hop). At the doc level that's the whole game — dense misses
  the gold doc, `seed + g(q)` lands on it. Measured wins: metaqa doc recall@100 13.7→50.9 (**+37**),
  musique 77→91.8 (**+15**); l1_relational recall@200 metaqa 16→51.5, musique 81.5→93.9.
- **L1 partition level (where offset is FLAT):** the **overlap-voting already performs the relational hop —
  structurally, via graph edges.** Each retrieved node votes for its own partition AND its 1-hop neighbors'
  partitions. So a node that is a relation-hop *away* from the gold (but a graph *neighbor* of it — which
  relational golds usually are) **already covers the gold's partition**. The graph edge substitutes for the
  offset's embedding-space hop. Plus the metric is coarse (gold's *partition* in top-20, partition ≈ 100 nodes).
- **Result:** the offset's precise doc-recovery is *redundant* at L1 — the partition was already votable via a
  neighbor. Confirmed empirically: `voted_oracle@20 = 100` everywhere; the offset adds nothing to *coverage*,
  only to *ranking*, where it is weak.

**One line:** *L1's overlap-voting IS a relational-hop mechanism, so it makes the learned offset redundant.
The offset only shines where there is no graph-voting fallback — i.e., when you need the actual doc.*

## 4. Architectural conclusion — mlpT/offset belongs at L2, not L1

- **L1 = dense + overlap-voting → fast partition routing.** Offset redundant here (drop, or keep as a tiny
  multi-hop nudge). This is *why* dense looked "singlehandedly good" — the overlap-voting (a core L1 design
  element) does the relational work.
- **L2 = relational-offset / mlpT SEED retrieval** within the routed top-20 partitions → recover the
  multi-hop/KB gold *docs* dense misses. **This is where the offset novelty actually lives** — a doc-level task
  with no partition-voting fallback, exactly the +15/+37 regime.
- **L3 = traverse** from those seeds.

The roadmap half-anticipated this ("the query-relational-hop win is a doc retriever → belongs at L2"); the L1
partition results now confirm it empirically: **offset redundant with overlap-voting at L1, irreplaceable at L2.**

**Next test (to confirm):** within each query's top-20 partitions, measure gold-**doc** recall@{5,20,50} for
`dense` vs `dense + offset/mlpT` (seed-finding). If the +15/+37 lift reappears in the partition-scoped pool,
mlpT is confirmed as the L2 seed retriever — a stronger, more defensible contribution than forcing it into L1.

## Artifacts
- Universal comparison: `results/L1_universal_head_bge_large.json` (+ `_gte_qwen.json` when the gte run lands).
- Per-dataset stack + top-N sweep + oracle: `data/ukb_storage/{d}/results/L1_select/rerank100_{d}_bge_large.json`.
- Runner `src/experiments/l1_universal_head.py`; benchmark aggregator `src/experiments/l1_benchmark.py`.
