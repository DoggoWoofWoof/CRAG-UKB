# L1_STATIC — a purely static, parameter-free improvement of the P50 router

Lane: `results/L1_STATIC/` (scratch code `scratchpad/_l1s_*.py`, all untracked). Date: 2026-09-14.
Governing request: *no dataset-specific method, no walk (that belongs in L3); with what we have, improve the
partitioning or find something else that improves the retrievals at P50, as a global improvement for all datasets.*

Research question: **how much of the gap between the frozen static router (BASE, metaqa ALL 0.64) and the
dynamic-reasoning ceiling (typed walk 0.97, `results/L1_P90_EXPLOIT/`, diagnostic only) can a purely static,
parameter-free L1 router recover — uniformly across datasets?**

Rules kept throughout: served dense/SPLADE top-100 hits only, frozen numerics (`K0=60, K_LOCK=100, P_MAIN=50`,
`partition_ranking`, `rrf_partitions` imported from `scratchpad/_ta_prepartition.py`, never edited), frozen H4_SK
partitions, no graph walked at query time in any arm, no parameter fitted, no dataset-specific rule, TEST never read.
Exploration read **DEV_A** only (`sha1(qid)[:8] & 1 == 0`); one mechanism was frozen and pre-registered, then
**DEV_B** was read once. Significance: exact McNemar (paired, `scipy.stats.binomtest`).

## 1. Answer

| metaqa (NAME_ONLY dev, ALL@P50)                       | ALL    |
|-------------------------------------------------------|--------|
| BASE (frozen L1 selector)                              | 0.638  |
| **static universal candidate (this lane, full dev)**   | **0.677** |
| KB-only static tables (R2 alone, DEV_A; loses on text) | 0.71–0.74 |
| typed walk (L1_P90_EXPLOIT, diagnostic upper bound)    | 0.97   |
| P50 feasibility ceiling (all gold blocks fit in 50)    | 0.98   |

A static, parameter-free router recovers **about 4 points of the 33-point gap on metaqa (~12%)** and **6 points on
musique**, with no significant loss anywhere — and that is close to the end of what "static" can do: every other
static mechanism examined here is refuted (§4), and the remaining gap is reach, not ranking (§3). The rest of the
gap needs query-conditioned traversal, i.e. L3.

## 2. The confirmed mechanism — multi-radius static node→blocks tables

BASE already votes each dense / SPLADE hit into a static table: *own block + blocks of the directed structural
out-neighbours* (the legacy `load_topology` semantics reproduced bit-exactly, 1998/1998). The candidate keeps that
and adds three more static tables of different radius, each a query-independent `node → blocks` list computed once
from the partition and the structural family:

| table | definition | mean span (metaqa / squad / musique) |
|---|---|---|
| R0 | own block | 1.0 / 1.0 / 1.0 |
| RL | own + directed-out neighbours' blocks (= BASE's table) | 2.5 / 5.0 / 3.1 |
| R1 | own + undirected neighbours' blocks | 4.0 / 5.0 / 3.9 |
| R2 | R1 ∪ R1-tables of undirected neighbours whose own span ≤ P50 (*budget rule*: an intermediate that already spans more than the 50-block budget cannot be followed, so it is not passed through — the constant is the frozen budget, nothing was fitted) | 24.9 / 9.9 / 14.1 |

Each table gives a dense channel and a SPLADE channel through the frozen `partition_ranking` (rank votes
`1/(K0+r)`, `rr(sum)+rr(max)`); the channels are fused by the frozen `rrf_partitions` (RL-dense first = the frozen
tie-break). **PRIMARY = RL+R0+R1+R2 (8 channels)**, SECONDARY = RL+R0+R1. Nothing is walked at query time; the
tables are precomputed sparse lists (R2 is ~25 ints/node on metaqa). Module: `scratchpad/_l1s_candidate.py`.

### Pre-registered DEV_B confirmation (`PREREGISTRATION_DEV_B.json` → `DEV_B_CONFIRMATION.json`)

Rule, stated before DEV_B was read: no population with a significant loss (lost > gained, p < 0.05) on any of
metaqa / squad / musique DEV_B, and a significant gain (gained > lost, p < 0.01) on at least one. The confirmation
script verifies the code and cache hashes, reproduces the recorded DEV_A numbers exactly, and refuses to run twice.

| cache | population | n | BASE | PRIMARY | gained / lost | p |
|---|---|---|---|---|---|---|
| metaqa | DEV_A (exploration) | 1002 | 0.6397 | 0.6766 | +43 / −6 | 5.7e-8 |
| metaqa | **DEV_B (confirmation)** | 996 | 0.6355 | **0.6767** | +48 / −7 | 1.3e-8 |
| metaqa | full dev | 1998 | 0.6376 | 0.6767 | +91 / −13 | 1.4e-15 |
| squad | DEV_A | 994 | 0.9859 | 0.9889 | +4 / −1 | 0.38 |
| squad | **DEV_B** | 1008 | 0.9911 | **0.9921** | +2 / −1 | 1.0 |
| squad | full dev | 2002 | 0.9885 | 0.9905 | +6 / −2 | 0.29 |
| musique | DEV_A | 996 | 0.6596 | 0.7239 | +71 / −7 | 1.9e-14 |
| musique | **DEV_B** | 1004 | 0.6235 | **0.6803** | +69 / −12 | 7.0e-11 |
| musique | full dev | 2000 | 0.6416 | 0.7020 | +140 / −19 | 5.7e-24 |

**Outcome: `PRIMARY_CONFIRMED`** (significant gains on metaqa and musique DEV_B, no significant loss anywhere; squad
is at ceiling, +2/−1). Per hop on metaqa DEV_B: hop1 0.950, hop2 0.559, hop3 0.512 (BASE 0.879 / 0.491 / 0.527) — the gain is hop-1/hop-2 reach, hop-3 is untouched. Scope stays ~5,100 nodes (P50 unchanged).
The SECONDARY also passed (musique-only gain, +91/−10, p=1.7e-17; metaqa ±0).

### Partition-independence (DEV_A, metaqa)

| partition | BASE | PRIMARY | Δ |
|---|---|---|---|
| frozen H4_SK (Mt-KaHyPar) | 0.6397 | 0.6766 | +43/−6, p=6e-8 |
| PHG (Zoltan, `replay_cache__LOWMEM__PHG_REPAIR1_con`) | 0.6727 | 0.7186 | +48/−2, p=2e-12 |
| STRUCT-only H4 (`parts/metaqa__S`, ruling-gated) | 0.6796 | 0.7186 | +40/−1, p=4e-11 |

The mechanism composes with whatever partition is served; it is not a property of one partition.

## 3. Why this is (nearly) the static limit — the reach diagnostic

"Reach" = the fraction of queries whose gold blocks *all* receive at least one vote from the served top-100 hits
through a given table (a rank-free upper bound on what any re-weighting of those votes can achieve).

| table (metaqa, DEV_A, K=100) | reach ALL | hop1 | hop2 | hop3 |
|---|---|---|---|---|
| R0 own block | 0.26 | 0.51 | 0.19 | 0.09 |
| RL (BASE's table) | 0.71 | 0.94 | 0.59 | 0.61 |
| R1 undirected | 0.86 | 1.00 | 0.85 | 0.74 |
| R2 (unbounded undirected 2-hop) | 1.00 | 1.00 | 1.00 | 1.00 |

Under BASE's own table only 71% of metaqa queries *can* be fully covered by any ordering of the votes; gold blocks
BASE misses sit at median rank 231 of 432 — they are not narrowly missed, they are unreached. That is why every
aggregation / weighting variant (§4, A1) is flat: the votes are right, the evidence is absent. Wider static tables
restore reach (R1 0.86, R2 1.0) but dilute precision — R2 alone reaches everything yet covers 0.73 (metaqa) and
*loses* on text corpora (squad 0.91, musique 0.49) because a 2-hop table of a text graph spans too many blocks.
The RRF over all radii is the parameter-free way to take the reach of the wide tables and the precision of the
narrow ones; it is what survives on all three corpora. On squad reach is already 0.999 under RL — there is nothing
static left to recover, which is exactly what the squad numbers show. What remains on metaqa (hop-3 0.51, hop-2
0.56) needs the *query* to decide which neighbours to follow — that is a walk, and it belongs in L3.

## 4. Refuted mechanisms (all DEV_A; numbers in the `*_A_<ds>.json` files)

* **A1 node→block aggregation** (`agg_A_*.json`): sum / max / count / top-m / mean / size-normalised votes; rank
  vs steep vs score vs relative-score weights; K ∈ {20, 50, 100, 200, 500, 1000}; hard vs legacy membership. Nothing
  beats BASE on metaqa (±0.005); count / mean are far worse; K=20 is much worse everywhere, K≥500 worse on musique.
  A steeper vote kernel (1/(1+r)) is +16/−2 on musique (p=1e-3) but flat on metaqa/squad — a contract constant, not
  adopted (no tuning on results). *Hypothesis "sum-mass lets mediocre hits win" refuted; the reach diagnostic explains
  the flatness.*
* **A2 direct block signatures** (block dense centroid, SPLADE mean/max pooling, scored against the query without
  the node hits): useless alone (metaqa 0.13 / 0.39 / 0.17; squad 0.91 / 0.91 / 0.98; musique 0.41 / 0.27 / 0.53);
  the SPLADE mean-pool ranking is half query-independent (mean top-50 overlap between different queries 0.49) and
  tracks block term mass. Fused with BASE: metaqa −17/+7 (p=0.06), squad ±0, musique +62/−19 (p=2e-6) — not
  universal (musique-only, at a metaqa cost).
* **Graph-derived embedding offsets** (`offsets_diag_metaqa.json`, `offsets_q_A_metaqa.json`): the NAME_ONLY
  corpus embeddings do not encode entity relations — held-out offset hit@100 for directed_by / written_by /
  starred_actors is 0.08–0.10 (attribute relations like genre / year / rating are the only coherent ones, 0.83–1.0),
  the three entity Δ_r are collinear (cos 0.85–0.97), the global operator G has trace 1 with top eigenvalue 0.09
  (effective rank 230). The query ladder O0–O3 (raw offset, G-projected offset, typed consensus, ±Δ) is ≈ BASE
  everywhere (0.634–0.643 alone, 0.636–0.643 fused). Even the **oracle** relation offset is 0.60 alone / 0.636
  fused, and the **oracle answer-centroid point** gives 0.81 alone / 0.74 fused — the one-point family is capped below
  the walk even with perfect information. *Family refuted.*
* **D block-graph propagation / static SAFE ordering** (`static2_A_*.json`): one step of block-graph smoothing
  `S0 + S0·W̃` (struct or SK cut weights), or BASE head-25 + cut-weight tail, hurts on every corpus (metaqa 0.49 /
  0.61 fused; squad 0.96 / 0.98; musique 0.20 / 0.60) — the block graph is 36–59% dense on metaqa (13–17% on
  musique); cut weights carry no query information.
* **Query-independent priors** (`prior_A_metaqa.json`): block degree-mass alone 0.44 (hop3 0.52 ≈ BASE's 0.53 — a
  benchmark artefact: 28% of gold-block incidences fall in the 50 highest-mass blocks), fused with BASE +15/−7 (ns).
  No lever.
* **Single-radius refinements** (`mem1_A_*.json`): hub rule (span > P50 → own block only), mass conservation
  (vote / table size), vote-level summation across tables — none beats the plain RRF over radii on all three corpora;
  the mass-conserved / hub variants trade metaqa against musique.
* **KB-only static tables**: R2 alone (or R2+BASE) reaches 0.71–0.74 on metaqa (τ=25: 0.7425) but loses on squad
  (−1.1 pts, p=0.003) and musique (−3.7 pts, p=6e-5). Dataset-specific by the user's criterion → not a candidate; kept
  as the static KB ceiling in the hierarchy.

## 5. The partition lever (`partition_A_{metaqa,squad}.json`, `parts/`)

Single global rule changes to the frozen H4_SPLIT_PRESERVE hypergraph, same worker (Mt-KaHyPar DETERMINISTIC_QUALITY,
KM1, eps 0.03, seed 0, k = N//100), scratch outputs only (`results/L1_STATIC/parts/`, nothing under `data/`):

| variant | metaqa BASE (DEV_A) | vs frozen | squad BASE | vs frozen |
|---|---|---|---|---|
| frozen H4_SK | 0.6397 | — | 0.9859 | — |
| W1 (every hyperedge weight 1) | 0.5519 | +14/−102, p=1e-17 | 0.9919 | +11/−5, ns |
| WSQRT (weight 1000/√(|e|−1)) | 0.6018 | +25/−63, p=6e-5 | — | — |
| S (STRUCT-only hypergraph, no KNN family) | **0.6796** | +68/−28, p=5.5e-5 | 0.9859 | +8/−8, ns |

The frozen equal-weight-per-neighbourhood rule is confirmed (both re-weightings lose). The only partition gain is
the STRUCT-only hypergraph on metaqa (gold blocks/query 5.20 → 4.45, feasible-P50 0.980 → 0.990; the candidate on top
of it reaches 0.7186). It is neutral on squad and could not be tested on musique (Mt-KaHyPar RESOURCE_INFEASIBLE_LOCAL).
**This is ruling-gated:** the standing edge-family audit closed with `SK_STAYS` (S/SN refuted on the shipped cell),
so a STRUCT-only partition would reopen a closed audit — reported, not proposed.

## 6. What adopting the candidate means

* It is a **new L1 selector**, not a cache rebuild: it reads only the served dense/SPLADE top-100 caches, the frozen
  partition and the structural family (directed + undirected CSR), and precomputes four sparse `node → blocks`
  tables. No embedding, partition or replay cache changes.
* Adoption changes the L1 contract (selector module + contract hash) → **needs a ruling**; nothing frozen was edited
  in this lane (`_ta_prepartition.py`, `src/l1_canonical/*` imported only).
* Standing caution (`g2-trackA-prepartition-audit`): an L1 coverage gain has previously not survived the frozen L2
  (pool-composition sensitivity). This lane measures ALL@P50 coverage; downstream survival is a separate question and
  is out of scope until the L1 re-freeze.
* WebQSP was not part of this lane (its L1 caches are the PHG scale run, still preflight-gated; `WEBQSP_ROG_RESOLVED`,
  BENCHMARK_CONDITIONED).

## 7. Files

`_l1s_check.py` sanity / BASE reproduction · `_l1s_agg.py` A1+A2 · `_l1s_offsets_diag.py`, `_l1s_offsets_q.py`
offsets · `_l1s_static2.py` tables + reach + D · `_l1s_prior.py` · `_l1s_mem1.py` · `_l1s_radius.py` subset ladder
(optional second arg = scratch partition tag) · `_l1s_partition.py` partition variants · `_l1s_candidate.py` the
frozen mechanism · `_l1s_prereg.py` → `PREREGISTRATION_DEV_B.json` (+ `code_snapshot/`) · `_l1s_devB_confirm.py` →
`DEV_B_CONFIRMATION.json`. Core: `_l1s_core.py` (read-only over the replay caches and the canonical adapter).
