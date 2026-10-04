# L1 STRUCTURAL SCORE SEPARABILITY -- FINAL REPORT

## VERDICT: **B. STRUCTURAL_SCORE_PARTIAL**

Separability is real and large. Conversion to exact P50 is zero. The reason is now measured rather than guessed: the read stage was never worth much, and it is the only stage this phase was allowed to touch.

### Returns

| flag | value |
|---|---|
| `BEST_SINGLE_NODE_SIGNAL` | **RR_PARENT_SUPPORT** (R@64 0.2061, AUC 0.7792, Cohen d +0.9154) |
| `GOLD_READ_RECALL64_BASE` | **0.1334** |
| `GOLD_READ_RECALL64_BEST` | **0.2100** (R2_HOP_CONDITIONED; oracle 0.2902) |
| `READ_EFFICIENCY_BASE` | **0.4118** |
| `READ_EFFICIENCY_BEST` | **0.6483** (oracle 0.8959) |
| `EXACT_P50_METAQA_HOP2` | **0.7297**, unchanged. Best inference-safe 0.7357 (+4 queries of 666), not significant |
| `EXACT_P50_METAQA_HOP3` | **0.2583**, unchanged. **No inference-safe ranking beats it** -- the best of the three rankings ties frozen exactly at 0.2583, and the STEP-7 beam variant reaches 0.2598 (+1 query), not significant |
| `STRUCTURAL_SCORE_VERDICT` | **B_STRUCTURAL_SCORE_PARTIAL** |

`PROMOTED = NONE`. `L1_FROZEN = NO`. No TEST split was read. M_struct = 64, M_ret = 32, B = 6, P = 50, S4 and F6 are all exactly as frozen.

---

## 1. The dataset is the frozen read stage, not a model of it

Every one of the 11,415 queries across all six corpora reproduces the frozen `s_node / s_hop / s_sdir / s_cnt` arrays bit-exactly (T1). The features are quantities the search already computes; gold labels were attached afterwards and no feature sees them.

## 2. Separability exists, and the frozen key is below chance

The frozen ordering key `CS_ADMIT` scores **AUC 0.4541** on MetaQA -- below chance. That is not a bug, it is a cross-hop calibration failure. Gold density by node hop is 4.80% / 0.96% / 0.84%, while the frozen key places gold at percentile rank 0.5066 -- it pushes the most gold-dense stratum toward the bottom. Inside a stratum the frozen key is fine: its within-query, within-node-hop AUC is 0.6034, which is also the no-information floor for any rank rule here (a constant signal falls back to the stable tiebreak, so the floor is that value, not 0.5).

The signal audit splits cleanly into two families:

* **provenance** -- `RR_PARENT_SUPPORT` 0.7792, `HOP` 0.7974, `ADMIT_RANK` 0.7538, `PARENT_SCORE` 0.7335, `PATH_SUPPORT` 0.6866
* **geometry** -- `CS_ADMIT` 0.4541, `RSIM` 0.4453, `PATH_SUM` 0.4485, `STATIC_SDIR` 0.4940

Every geometric signal is at or below chance at the read stage; every provenance signal is well above it. The best single signal is `RR_PARENT_SUPPORT`, which lifts read recall from 0.1334 to 0.2061 (+54% relative).

`FUTURE_MAX`, the previous phase lookahead, scores 0.6017 globally and 0.4624 within stratum -- below the 0.6034 floor. Independent re-confirmation that the lookahead carries no usable ordering information.

## 3. STEP 5 -- the support hypothesis is confirmed

Gold bridge nodes do carry materially stronger multi-path support than negatives, and it is measurable before the top-64 read:

* `PARENT_SCORE` gold 0.4935 vs non-gold 0.1578, Cohen d **+0.9556**
* `RR_PARENT_SUPPORT` gold 0.5080 vs non-gold 0.1885, Cohen d **+0.9154**
* `DISTINCT_SEED_SUPPORT` gold 1.1881 vs non-gold 1.0568, Cohen d **+0.5175**

This was the phase's primary new hypothesis and it is true. It is also, as section 5 shows, not worth anything at exact P50.

## 4. STEP 4 -- the hop prior is a Simpson artifact

Pooled over MetaQA, shallow nodes look far more gold-dense, which is why every ranking that promotes hop-1 nodes lifts read recall. Conditioning on query hop inverts it (T3): on 3-hop questions gold density is 1.80% at node hop 1 but 2.44% at node hop 3. Query hop is not observable at inference. So any hop-conditioned rule that helps hop2 necessarily hurts hop3, which is exactly what R2_HOP_CONDITIONED does: hop2 0.7312 vs frozen 0.7297, hop3 0.2553 vs frozen 0.2583. Hop is not dataset identity, but on this substrate it is a proxy for the one thing the contract forbids observing.

## 5. STEP 6/9 -- separability does not convert, on any corpus

| ranking | macro delta | worst corpus | net over 11,415 queries | corpora improved | significant |
|---|---:|---:|---:|---:|---|
| R1_EQUAL_RRF | +0.00092 | -0.0005 | +9 | 5/6 | none |
| R2_HOP_CONDITIONED | -0.00088 | -0.0020 | -11 | 1/6 | none |
| RS_BEST_SINGLE (direct) | -0.00042 | -0.0035 | -8 | 2/6 | none |
| RX_ORACLE_READ (ceiling) | +0.00360 | +0.0000 | +37 | 4/6 | metaqa, webqsp |

Across the 18 inference-safe cells the correlation between read-recall lift and exact-P50 change is **r = +0.0419** (spearman +0.0320). 15/18 cells improve the read -- by +0.0740 on average, up to +0.1839 -- and of those, 6 improve exact P50. A coin flip.

The sharpest single case is 2wiki: R2 reads 3.4x more gold than frozen and delivers gold partitions to S4 essentially perfectly (0.5758 -> 0.9992), for -1 query at exact P50.

And the ranking with the *best* measured separability, used directly as the directive's first branch prescribes, is directionally **negative**: `RS_BEST_SINGLE` (RR_PARENT_SUPPORT) has macro -0.00042 and improves 2/6 corpora.

## 6. Why -- gold node recall is not the objective

Measured on MetaQA at the top-64 read:

| ranking | gold nodes read | distinct gold partitions | of which the swap actually NEEDS |
|---|---:|---:|---:|
| R0_STATIC (frozen) | 1,985 | 1,792 | 905 |
| R2_HOP_CONDITIONED | 3,125 | 2,650 | **756** |
| RX_ORACLE_READ | 4,318 | 3,534 | 1,279 |

R2 reads 57% more gold nodes and 48% more distinct gold partitions than frozen, and supplies **16% fewer of the partitions the swap actually needs**. The needed partitions are by definition the ones the protected core does not already hold -- the peripheral, hard ones. A better generic node ranking rescues *typical* gold, which is disproportionately in already-covered partitions. Read recall and the objective are not the same quantity, and on this substrate they are barely related.

## 7. STEP 7 -- the same key inside the beam

R1 was re-run as the beam survival key as well as the `added` and read key: one semantics at every stage, which is the consistency the lookahead failed. Parity against frozen breaks by design; the question is whether a better-separating key buys reach.

| corpus | parity vs frozen | gold discovered | exact P50 | net | p | oracle on this beam |
|---|---:|---:|---:|---:|---:|---:|
| metaqa | 12/1998 | 0.2902 -> 0.3261 | 0.6612 -> 0.6632 | +4 | 0.541 | 0.6812 |
| webqsp | 64/1419 | 0.2734 -> 0.3048 | 0.7646 -> 0.7660 | +2 | 0.791 | 0.7794 |
| 2wiki_clean | 448/2000 | 0.2596 -> 0.2600 | 0.9435 -> 0.9430 | -1 | 1 | 0.9435 |
| musique_clean | 48/2000 | 0.1400 -> 0.1423 | 0.9635 -> 0.9625 | -2 | 0.727 | 0.9610 |
| hotpotqa_clean | 5/2000 | 0.1872 -> 0.1958 | 0.9505 -> 0.9505 | +0 | 1 | 0.9510 |
| squad_clean | 295/2000 | 0.0265 -> 0.0240 | 0.9875 -> 0.9875 | +0 | 1 | 0.9875 |

On MetaQA the R1 beam is a genuinely different search (12/1998 parity) and does find more gold (0.2902 -> 0.3261), which raises the *ceiling* from 0.6707 to 0.6812. The realisable gain is +4 queries at p = 0.541, and hop3 lands at 0.2583 -- exactly the frozen 0.2583.

On musique the R1 beam finds marginally more gold and the oracle on it is *worse* than frozen (0.9610 vs 0.9635). Changing the survival key changes pool composition, and composition can cost more than the extra gold is worth. This is the same pool-composition sensitivity the pre-partition audit recorded.

## 8. STEP 10 -- the headroom, and where it goes

Every query lands in exactly one bucket according to what the 6-slot swap has to supply (T7). On MetaQA, frozen:

* `FREE` 1294 -- the protected 44 already cover the query
* `CAPACITY` 246 -- more than 6 partitions are needed; dead at B = 6 whatever the ranking
* `NOCAND` 311 -- some needed partition is not a candidate at all. **The only bucket a better read can move.**
* `LOST` 120 -- every needed partition is a candidate and F6 still did not pick them
* `WON` 27

A perfect read moves `NOCAND` 311 -> 276 and pushes 16 of those straight into `LOST`, for a net +19 queries. Of the 6,132 partitions the swap must supply across MetaQA, **4,190 have no evidence in any channel** -- not structural, not retrieval, not incumbent -- and the oracle read only reduces that to 3,958. The read stage can supply evidence for 250 more needed partitions out of 6,132, or 4.1%.

On 2wiki, musique and squad the oracle does not reduce the no-evidence count at all (2wiki 98 -> 98). Those partitions contain no discovered node, so no read ordering can reach them: on the text corpora the needed-partition deficit is a **reach** deficit, and the read stage is already saturated with respect to what it can contribute.

MetaQA hop3, the phase's primary target, is the extreme case: mean need is **7.78 partitions against a 6-slot budget**, and 213/666 hop-3 queries (32%) are in `CAPACITY` -- unreachable at B = 6 under any ranking whatsoever. That is why the oracle moves hop3 only 0.2583 -> 0.2613.

## 9. The ceiling

A **perfect** read -- every discovered gold node in the top 64, gold-partition delivery to S4 at 1.0000 on every corpus -- is worth macro **+0.00360**, significant on only 2 of 6 corpora (metaqa, webqsp). On MetaQA it is ALL 0.6612 -> 0.6707, hop2 0.7297 -> 0.7523, hop3 0.2583 -> 0.2613.

The phase target was hop3 *materially* above 0.2583 and hop2 *materially* above 0.7297. The oracle reaches hop3 0.2613, i.e. 2 queries out of 666. **The target is unreachable from this stage even with gold labels.** For scale, the fixed-P50 partition router promoted in an earlier phase was worth macro +0.0069 -- roughly twice what solving this entire stage perfectly would buy.

## 10. What this closes and what it does not

**Closed.** Read-stage node ordering as an L1 lever. The stage has a hard ceiling of macro +0.00360; three parameter-free rankings and the best single signal used directly all capture none of it; and the read metric is uncorrelated (r = +0.0419) with the objective, so it cannot even be used as a proxy to optimise against. Do not build another node scorer for this stage.

**Not closed, and not touched here.** The residual sits in two places this phase was explicitly forbidden from changing:

1. **Capacity.** B = 6 against a mean need of 7.78 on MetaQA hop3; 246/1998 queries overall are arithmetically dead.
2. **Reach.** 4,190/6,132 needed partitions on MetaQA and 98/147 on 2wiki have no evidence in any channel, and a perfect read barely dents it.

Neither is a scoring problem. The verdict is B rather than C because separability was found and is large -- but the practical consequence is the same STOP for parameter-free structural L1 search, with a better-specified reason: the stage was not where the loss lives.
