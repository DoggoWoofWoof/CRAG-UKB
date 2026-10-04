# L1_GEOM — partition · fusion · query geometry under the fixed P50 budget

Lane `results/L1_GEOM/` (2026-09-14). Objective as ruled: **maximise P(all gold evidence inside L1's 50 blocks)** with
three levers only — (1) partition quality, (2) parameter-free query-time fusion of the Dense/SPLADE block channels,
(3) a universal parameter-free query transform derived from q and its frozen SEED_K = 5 seeds — plus the directional
form of (3): the orientation of the seed → query displacement scored over real nodes (D0/D1/D2). Everything else is
frozen: K0 = 60, K_LOCK = 100, P50, SEED_K = 5, the legacy node → blocks table, the served Dense/SPLADE top-100 hits,
the canonical embeddings. No relation labels, no traversal, no learned weights, no fitted constants, no dataset rule.
Exploration on DEV_A (sha1(qid) parity 0); one pre-registered DEV_B look (§8). TEST never read.

Populations: metaqa under the frozen Mt-KaHyPar H4_SK partition and under the validated Zoltan-PHG partition of the
same corpus; squad under both; musique under PHG only (Mt-KaHyPar musique is RESOURCE_INFEASIBLE_LOCAL).
DEV_A n = 1002 / 1002 / 992 / 992 / 996. Metric: ALL-gold P50 coverage, paired exact McNemar vs BASE on the same
queries (`+gained/−lost p`). Bold = p < 0.05 in either direction.

## 0. Verdict

| question | answer (DEV_A) |
|---|---|
| How much comes from partitioning? | **metaqa +3.3 pts** (0.640 → 0.673, paired +52/−19, p = 1e-4; the full-set ruling saw the same, 0.638 → 0.671). squad −0.3 (ns). PHG cuts *more* edges than Mt-KaHyPar (directed cut 0.817 vs 0.802; km1 ratio 1.106) and still covers more: the partitioner's objective is not the coverage objective. |
| How much comes from fusion? | **metaqa / squad: nothing** — five parameter-free fusions all within ±0.5 pt, and the union-of-top-50 ceiling leaves only 1.0–1.2 pts to any fusion. **musique: +1.1 … +1.8** (interleave, agreement blends; p 0.002–0.03) out of a 7.4-pt fusion-fixable mass. |
| Does query geometry give genuinely new recall? | **Point transforms (μ, 2q−μ, q−μ, q̂, shifted seeds): no on metaqa and squad** — every arm within ±1 pt, 2q−μ is +0.4 / +0.4 / −0.5 / 0.0 → *killed* per the pre-stated rule. On musique 2q−μ is +1.8 (p = 0.04), +4.2 with the interleave (p = 1e-5), but it does not stack on the directional signal (below). |
| Does the *orientation* (D2) carry relational information? | **metaqa: no** — gold nodes sit at median rank 2,521 of 43,234 under the directional score; gold-in-top-100 0.171 vs 0.143 for Dense(q); D2 = BASE ± 1 pt on both partitions. **musique: yes, but most of the D2 gain is aggregation** — D2d 0.660 → **0.746** (+92/−6, p = 7e-21); the exhaustive dense block-max *without any seed or direction* already gives 0.732 (+77/−5); reversed / random directions *lose* (−3.6 / −1.9). Orientation ≈ +1.4 pts on top (paired +27/−13, p = 0.04). |
| Did it survive the held-out look? | **Yes — PRIMARY_CONFIRMED** (§8). DEV_B musique 0.6235 → **0.7181** (+100/−5, p = 5e-24; full dev 0.642 → 0.732, +192/−11); no significant loss on any of the five populations (metaqa MtK +8/−6, metaqa PHG +9/−2, squad +3/−0, squad PHG +6/−1). The aggregation-only decomposition arm reaches 0.7122 on the same DEV_B (+92/−3): the orientation's own share is ≈ +0.6 pt. The point-geometry SECONDARY (2q−μ + interleave) did not pass the stricter threshold (musique +59/−36, p = 0.02). |
| Where is the rest? | metaqa: 27 of the 33–36 failure points are **UNREACHED** — no served hit votes for at least one gold block (6.3 unreached gold blocks per failed query); only structure (the shelved static tables, or L3 walks) reaches them. musique: 11 unreached + 15.6 reached-but-weak + 7.4 fusion-fixable — a ranking problem, which is why query-time signals work there. |

## 1. The factorial

P0/P1 = the frozen selector on each partition (Mt-KaHyPar columns = P0, PHG columns = P1). F = the same two served
channels under another parameter-free fusion (F1a evidence-masked RRF, F1b RRF × MNZ, F1c round-robin interleave,
F1d/F1e agreement blend of the RRF and interleave positions by A_q = set overlap / reciprocal-rank agreement of the
two channels' top-50). G = one more retrieval point through the same dense index → one more block channel (G1 μ = mean
seed embedding, G2 2q−μ, G3 q−μ, G4 q̂ = q + P(q−μ) with P the Moore-Penrose projector onto span(e_i − μ), G5 seeds
shifted by P(q−μ)), fused with the two served channels. D = the directional signal D_i(v) = cos(q − e_{s_i}, e_v − e_{s_i})
over *all* real nodes (D1a mean over seeds → top-100 → table; D1b per-seed node RRF → top-100 → table; D1c block max of
the mean over actual nodes; D1d per-seed block max → RRF over seeds), each as a third channel through the frozen RRF.

{{factorial}}

Notes. G4/G5 on metaqa MtK are the re-run with the pseudo-inverse guard (`pinv(G, rcond=1e-5)`): Σ_i(e_i − μ) = 0 makes
the 5×5 Gram rank ≤ 4 and the first run inverted the null direction (‖Δ‖ ≈ 2,100); with the guard ‖Δ‖/‖r‖ = 0.30 on
metaqa, 0.23 on musique. Numerical hygiene, not a fitted parameter. "G alone" rows (the point's channel without the
served channels) are in the JSON records; every point alone is worse than BASE everywhere (μ alone 0.607 / 0.542,
2q−μ alone 0.636 / 0.499 on metaqa PHG / musique) — the points are weaker retrieval queries than q itself.

## 2. Ceilings, headroom and the agreement signal

Ceilings under the frozen evidence (served top-100 hits through the legacy table): *feasible* = ≤ 50 gold blocks;
*reach* = every gold block gets at least one vote (the bound for any re-weighting of the votes); *union* = every gold
block is inside the union of the two channels' top-50 (the bound for any fusion of the two rankings). Failure mass of
BASE decomposed accordingly.

{{headroom}}

Fusion headroom is 1.0–1.2 pts on metaqa, 0.8–0.9 on squad, 7.4 on musique — which is exactly where the fusions moved.
The agreement A_q (top-50 overlap of the two channels) is a *confidence* signal, not a selector: the top quartile of
agreement runs 8–15 pts above the others (metaqa PHG 0.754 vs 0.61–0.68; musique 0.705 vs 0.60–0.70), and on metaqa
the unreached fraction is lowest there (0.22 vs 0.24–0.36). Usable for an L1 → L3 gate; it cannot fix the P50 set.

## 3. Lever 1 — partition quality

Same DEV_A queries, same served hits, same numerics, two partitions of the same corpus:

{{partition}}

PHG gives metaqa +3.3 pts on every arm (BASE, F1e, G12, G3 all +2.5…+3.3, p ≤ 3e-3) and squad nothing (−0.3, ns).
The gain is not cut quality — PHG's cut is *worse* (directed cut-edge fraction 0.817 vs 0.802; the validation ruling
recorded km1 ratio 1.106 on metaqa, 1.155 on squad) — and not feasibility (0.984 vs 0.982) or fewer gold blocks per
query (5.09 vs 5.15); it is a different co-location of gold blocks with the served hits. Block sizes are also less
even under PHG (min 1 vs 14; blocks of size 1–3 exist). Consequence for "better block geometry": the objective the
partitioner optimises (connectivity / cut) is only loosely coupled to P50 coverage; a partition objective that scores
coverage directly would be the lever, but it needs a ruling (the frozen contract is Mt-KaHyPar H4_SK; PHG is authorised
by the validation ruling only).

## 4. Lever 2 — parameter-free fusion

Nothing on metaqa or squad (all five fusions within ±0.5 pt, none significant, consistent with the 1-pt ceiling). On
musique the interleave (+1.8, p = 0.03), the set-overlap agreement blend (+1.8, p = 0.002) and the reciprocal-rank
blend (+1.1, p = 0.03) all gain; the evidence-masked RRF and RRF × MNZ do not (−0.3 / −0.5). Mechanism: on musique the
two channels disagree more often (A_q mean 0.59 vs 0.70 on squad) and the interleave / agreement blends guarantee each
channel its top-17/25 blocks, which the RRF's rank-sum denies to a block ranked well by one channel only. On metaqa the
channels' disagreements are not about gold blocks — the gold blocks are outside both lists (§2).

## 5. Lever 3 — point geometry (μ, 2q−μ, q−μ, q̂, shifted seeds)

The pre-stated rule was "if 2q−μ does nothing, kill it immediately; if it adds 3–5 points, investigate further".
2q−μ as a third channel through the frozen RRF: metaqa +0.4 / +0.4, squad −0.5 / 0.0 (all ns) → **killed for the KB
and for squad**. On musique it adds +1.8 (p = 0.04), +4.2 under the interleave (0.702, +66/−24, p = 1e-5), so the
"investigate" branch was taken there: q̂ (Moore-Penrose, the user's Δ_q) +1.5 (p = 0.04), μ *hurts* (−2.9, p = 8e-5),
q−μ nothing, μ + 2q−μ + interleave 0.694. New-evidence diagnostics say why the points are weak on metaqa: 2q−μ brings
new hits for half the queries but reaches an *unreached* gold block for 10–12 % of them and lifts reach only
0.713 → 0.733 / 0.728 → 0.746 — the points re-rank what is already reached, they do not open the unreached mass.

{{newevidence}}

## 6. The directional signal (D0 / D1 / D2), its orientation test and its controls

Orientation test — where do gold nodes rank under the directional score vs under Dense(q)?

{{orientation}}

On metaqa the orientation is real but weak: gold nodes are in the top 5.8 % of the directional ranking (median 2,521 of
43,234), yet almost never in the top-100 (0.147 vs 0.143 for Dense(q); 0.171 with the per-seed RRF; hop3 0.205 vs
0.172); D2 through any aggregation is BASE ± 1 pt on both partitions (D2c even loses on Mt-KaHyPar, −1.0, p = 0.02).
This is the stronger evidence the ruling asked for: **the orientation of NAME_ONLY embedding displacement does not carry
enough relational information** to find the multi-hop gold entities' blocks — the unreached 27 pts stay unreached.

On musique D2 is large and one-sided: D2d 0.660 → 0.746 (+92/−6), D2c 0.727, D2b 0.697. Controls (DEV_A, third channel
through the frozen RRF, or alone):

{{controls}}

Reading: (i) the *direction* matters — reversed (e_s − q) and random unit directions turn the same block-max
aggregation into a significant loss (−3.6 / −1.9); (ii) but the *aggregation* carries most of the gain — the exhaustive
dense block max (score(B) = max_{v∈B} q·e_v over all nodes, no seed, no direction) alone beats BASE (0.701 vs 0.660) and
as a third channel gives 0.732 (+77/−5); the directional per-seed max adds +1.4 on top (paired D2d vs QMAX +27/−13,
p = 0.04); (iii) the pooled variant (candidates = served dense ∪ SPLADE top-1000, as proposed for scalability) is
indistinguishable from the exact all-node version (0.745 vs 0.746) — the signal is cheap; (iv) block size is not the
mechanism on musique (size ranking loses −3.4), while on metaqa the size prior *alone* covers 0.456 (the 50 largest of
432 blocks hold all gold for 46 % of queries) — more than any embedding-similarity block score (D1d 0.26, qmax 0.20).

Why the aggregation helps on musique and not on metaqa: the served channel votes only for blocks touched by the top-100
hits, and on a paragraph corpus those hits pile into a few blocks of near-duplicate paragraphs; the exhaustive per-block
max scores every block once by its best paragraph, so a second-hop paragraph at dense rank 300 still ranks its block
(reached-weak mass 15.6 pts). On metaqa NAME_ONLY the best entity-name similarity of a block is simply not related to
whether the answer entity is in it (qmax alone 0.20).

Combinations (DEV_A only, to decide what to carry to DEV_B):

{{combos}}

Stacking does not help: 2q−μ on top of the directional channel *loses* on musique (X1 −1.8 vs D2d, p = 0.005; X3
p = 0.004); Cmax + Cdir is not better than D2d on musique (+19/−14, ns) and hurts metaqa (−2.0, p = 9e-4); giving the
BASE ranking one vote and the directional channel one vote (X4) gains +0.9 on musique but loses −1.4 on metaqa
(p = 0.009). One extra channel through the frozen RRF, no re-weighting, is the universal form.

## 7. Candidate and pre-registration

Fixed after the DEV_A factorial, before any DEV_B look (`_l1g_candidate.py`, `PREREGISTRATION_DEV_B.json`, code and cache
sha256 pinned, snapshot in `code_snapshot/`):

* PRIMARY `D2d` = frozen RRF of {Dense(q) block channel, SPLADE(q) block channel, directional per-seed block max → RRF
  over the 5 frozen seeds} → P50. The D2 cell exactly as specified. Zero fitted parameters.
* SECONDARY `G2_IL` = round-robin interleave of {Dense(q), SPLADE(q), Dense(2q−μ)} block channels → P50.
* decomposition arms, reported on DEV_B but never decided on: `IL` (fusion alone), `G2_F0` (point alone, frozen RRF),
  `QMAX_F0` (exhaustive dense block max as the third channel — how much of the PRIMARY is orientation vs aggregation).
* rule: no significant loss (p < 0.05) on any of the five populations; a significant gain on at least one at
  **p < 0.005** — stricter than L1_STATIC's 0.01 because **DEV_B was already read once** (by the L1_STATIC confirmation,
  a different, shelved candidate family); prediction from DEV_A: PRIMARY gains on musique, neutral elsewhere.

## 8. DEV_B confirmation (one shot)

{{confirmation}}

## 9. What this means for L1

* The three levers behave differently on the two regimes. On the **NAME_ONLY KB** (metaqa) only the partition moved
  (+3.3) and the P50 problem is *reach*: 27 pts of queries have a gold block no served hit votes for, 6 such blocks per
  failed query, hop2/hop3 unreached 0.37/0.38. No query-time signal derived from names — points, orientation, exhaustive
  block max, agreement — touches that mass; the earlier lanes put the same conclusion the other way round (static 2-hop
  tables reach 1.0 and confirmed +4; the typed-path selector reaches 0.97). The 0.67 → 0.98 gap on metaqa is structural
  and belongs to L3 (or to a partition objective that scores coverage, by ruling).
* On the **text corpus** (musique) the problem is ranking, and a zero-parameter query-time signal recovers a quarter to
  a third of the failure mass — DEV_A 0.660 → 0.746, **DEV_B 0.624 → 0.718 (confirmed), full dev 0.642 → 0.732** — with
  no significant loss anywhere. Most of it is the *aggregation* (score every block by its best node: 0.712 on DEV_B
  without any seed or direction); the orientation of the seed → query displacement adds ≈ +0.6 on DEV_B (+1.4 on
  DEV_A). It is the first L1 gain in this program that needs no table, no walk and no parameter; cost = 5 exact
  scorings of the node matrix per query for the confirmed form (the pooled top-1000 variant matched it on DEV_A but
  was not confirmed), or one exact scoring for the aggregation-only form (a decomposition arm, not the candidate).
  Adoption = a contract ruling (new selector channel); nothing is rebuilt.
* squad is at its ceiling (0.98–0.99; feasible 1.0, reach 0.999); nothing here should be read from it except "no loss".
* Agreement A_q is a usable L1 confidence signal for the L1 → L3 gate (top quartile +8–15 pts), not a selector.

## 10. Records, code, reproduction

`results/L1_GEOM/`: `ladder_A_<cache>.json` + `vecs_A_<cache>.npz` (P/F/G arms, ceilings, agreement, new-evidence, DEV_A ALL
vectors), `dir_A_<cache>.json` + `dirvecs_A_<cache>.npz` (orientation test, D arms), `headroom_A_<cache>.json`,
`partition_effect_A.json`, `dircontrol_A_{musique,metaqa_phg}.json`, `combo_A_{musique,metaqa_phg}.json`,
`PREREGISTRATION_DEV_B.json`, `DEV_B_CONFIRMATION.json`, `code_snapshot/`. Caches: metaqa/squad Mt-KaHyPar
(`replay_cache__H4_SK…`), metaqa `PHG_REPAIR1_con`, squad `PHG_con`, musique `PHG_C1_con` (sha256 in the pre-registration).
Code (`scratchpad/`): `_l1g_core.py` (channels, fusions, exact retrieval over the fp16 shards, geometry, ceilings),
`_l1g_ladder.py`, `_l1g_dir.py`, `_l1g_headroom.py`, `_l1g_partition_effect.py`, `_l1g_dir_control.py`, `_l1g_combo.py`,
`_l1g_candidate.py`, `_l1g_prereg.py`, `_l1g_devB_confirm.py`, `_l1g_tables.py` + `_l1g_report_assemble.py` (this
report's tables are generated from the records). Frozen numerics imported from `_ta_prepartition.py` /
`_l1s_core.py` / `_l1x90_core.py` (unchanged; `_l1s_core.py` is also sha-pinned by L1_STATIC). Run with
`PYTHONHASHSEED=0`; every arm is deterministic (stable sorts, fixed chunking; the random-direction control uses
`default_rng(0)`).

Disclosures: (a) ≈ 45 arms per cache were read on DEV_A — the p-values in §1–§6 are exploratory; only §8 is
confirmatory. (b) DEV_B is on its second look (first: L1_STATIC); the stricter gain threshold is the mitigation, and no
further candidate should be confirmed on it. (c) musique has no Mt-KaHyPar cache, so the partition lever is measured on
metaqa and squad only. (d) The exhaustive-block-max and pooled controls were run after the D2 results were seen; they
are diagnostics of the D2 mechanism, carried into the pre-registration only as a reported decomposition arm.
(e) Nothing under `data/`, no canonical manifest, partition, production cache or L1 result was modified.
