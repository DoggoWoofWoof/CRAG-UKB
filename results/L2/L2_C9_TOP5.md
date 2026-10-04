# L2 — C9: Top-5 Boundary + Expert-Consensus Reranking

**Goal.** Push top-5 evidence quality *beyond* the C8c reranker (VAL NDCG@5 0.8824, GOLD_RECALL@5 0.9030)
**without** changing the top-50 candidate set — via expert-consensus features, query-regime × candidate
interactions, boundary-hard training, and out-of-fold C8c residual stacking. Same top-50 pool, same XGBRanker
family, no new experts, no pool enlargement. Designed on a nested **C9_TRAIN (85%) / C9_DEV (15%)** carved from
the old TRAIN_INNER; selected on **C9_DEV only**; confirmed **once** on the old **DEV_INNER**. Official VAL not
touched; TEST never inspected; L1 unchanged; L3 not started.

## Headline — C9 is NEUTRAL: it does not beat C8c

The consensus + regime-interaction axis is **exhausted at the top-50 level**. On held-out C9_DEV the selected
C9b ties C8c within bootstrap noise, and net top-5 gold movement is slightly negative.

| C9_DEV (pooled, n=3303) | C8c (base) | C9a | **C9b (selected)** | C9-classifier |
|---|:---:|:---:|:---:|:---:|
| **NDCG@5** | 0.9658 | 0.9660 | **0.9660** | 0.9657 |
| **GOLD_RECALL@5** (macro) | 0.9762 | 0.9763 | **0.9762** | 0.9756 |
| ALL@5_FEASIBLE | 0.9331 | 0.9331 | 0.9331 | 0.9316 |
| MRR | 0.9893 | 0.9898 | 0.9898 | 0.9898 |
| NDCG@50 | 0.9748 | 0.9749 | 0.9750 | 0.9750 |
| **ALL@50** | 0.9897 | 0.9897 | **0.9897** | 0.9897 |

> **Note on scale.** C9_DEV is drawn from TRAIN_INNER, so it is *in-distribution* and sits ~0.966 — **not
> comparable** to the official-VAL 0.8824 milestone. These numbers measure **C9 vs C8c on the same split**, which
> is the only question C9 was allowed to ask. On that comparison C9 is flat.

**Paired bootstrap (2000×), selected C9b − C8c on C9_DEV:**

| metric | Δ | CI95 | significant |
|---|:---:|:---:|:---:|
| NDCG@5 | +0.0002 | [−0.0004, +0.0008] | ✗ |
| GOLD_RECALL@5 | +0.0000 | [−0.0010, +0.0010] | ✗ |
| NDCG@50 | +0.0002 | [−0.0001, +0.0005] | ✗ |
| ALL@50 | +0.0000 | [0, 0] | — (identical by construction) |

**DEV_INNER confirmation** (secondary held-out, n=2442): C8c NDCG@5 0.9637 → C9b 0.9642 (+0.0005); GOLD_RECALL@5
0.9748 → 0.9752. Same story — a hair up, inside noise. ALL@50 identical (0.9836).

## Why it's flat — the new features carry no information C8c lacks

C9a feature importance is dominated by the **out-of-fold C8c stack itself**:

| feature | importance |
|---|:---:|
| `c8c_oof_score` | 0.468 |
| `c8c_oof_rank` | 0.308 |
| `c8c_score_minus_rank1` | 0.111 |
| `c8c_rank_percentile` | 0.026 |
| `c8c_rank_minus5` | 0.024 |
| — all 29 consensus/regime EXTRA features combined — | **< 0.02** |

The strongest genuinely-new feature is `max_mean_ratio` at 0.0022; `regime_support_max` 0.0018, `contrib_entropy`
0.0015, `w_splade` 0.0013. In other words the reranker learns to **reconstruct C8c** and finds nothing to add
from the expert-consensus or query-regime × candidate interaction blocks. This is consistent with C8c already
having `max_contrib` (0.67 importance there) — the union-of-experts signal these consensus features were meant to
supply is *already captured* by C8c.

## Audit — where the residual top-5 errors actually live (C9_DEV)

- **False-negative golds** (gold ranked ≥5): 207 total, overwhelmingly shallow — **155 at C8c rank 5–9**, 31 at
  10–14, only 21 deeper. Their support profile is *specialist-strong, consensus-weak*: mean `n_top5` = 0.99 (vs
  3.71 for top-5 golds), `best_rank5` = 4.79 (vs 0.42), `rank_std5` = 427 (vs 96). One expert likes them; the
  rest bury them.
- **False-positive non-golds** in top-5 (9128) sit *between* the two: `n_top5` 1.99, `best_rank5` 2.02 — moderate
  multi-expert support but not the broad consensus the true top-5 golds enjoy (`n_top5` 3.71).
- The separating axis (consensus breadth) is therefore real — but it is **already the axis C8c ranks on**, so
  adding explicit consensus features is redundant.

## Diagnostics

- **Multi-gold** is the hard tail: single-gold queries recall5 = 1.00, 2-gold = 0.987, **3+-gold = 0.940**
  (ALL@5 only 0.799). C9b vs C8c on 3+-gold: 0.9391 vs 0.9404 — flat/marginally down. This is where the oracle
  gap lives.
- **Error-conditioned rescue** (C9b vs C8c): 8 golds rescued into top-5 (all bin 5–9), 11 pushed out →
  **net −3**. The second-stage boundary reranker (C9b) does not improve on the single-stage.
- **Specialist rescue**: the handful of moves come from mixture (3), relation (3), splade (2) — no systematic
  specialist the fusion is missing.
- **Oracle gap**: a *perfect* rerank of the existing top-50 would reach GOLD_RECALL@5 = 0.9969; the selected
  policy reaches 0.9762 → **0.0208 of recoverable headroom remains inside top-50**, but it is concentrated in the
  3+-gold cases whose golds need signal outside this feature family (or a larger pool).

## Decision & gates

**Selected on C9_DEV = C9b**, but it is a **tie with C8c, not an improvement** — reported for completeness.
**The standing top-5 policy remains C8c.** No design in C9 (improved LambdaMART grid, boundary-hard training,
OOF stacking, second-stage boundary reranker, or the pointwise gold classifier) produced a significant lift.

| Gate | Answer |
|------|--------|
| C9_NDCG5_ABOVE_090 | YES* (in-distribution C9_DEV; C8c already 0.9658 there) |
| C9_RECALL5_ABOVE_093 / _095 | YES* (C9_DEV; C8c baseline already 0.9762) |
| EXPERT_CONSENSUS_FEATURES_HELP | **NO** (all EXTRA consensus features < 0.02 importance) |
| QUERY_REGIME_INTERACTIONS_HELP | **NO** (regime/interaction features ≈ 0 importance) |
| BOUNDARY_HARD_TRAINING_HELPS | **NO** (kept 60% rows; C9a = C8c within noise) |
| OOF_STACKING_HELPS | **DOMINANT BUT CIRCULAR** (stack = 0.89 importance; reconstructs C8c, no net gain) |
| SECOND_STAGE_RERANKER_HELPS | **NO** (C9b net top-5 rescue −3) |
| C9_BEATS_C8C | **NO** (Δ NDCG@5 +0.0002, not significant) |
| DEEP_RECALL_PRESERVED | **YES** (ALL@50 identical 0.9897; rerank within top-50 only) |
| TOP5_ORACLE_GAP_REMAINING | 0.0208 (3+-gold tail, inside top-50) |
| SAFE_TO_CONTINUE_TOP5_OPTIMIZATION | **NO within current pool** (consensus/regime axis exhausted) |
| SAFE_TO_FREEZE_L2 | **NO** (held per instruction) |

\* True on the in-distribution development split, but not the point — the C9-vs-C8c comparison is the real test,
and it is flat.

**STOP (C9 complete):** selection done on C9_DEV, single confirmation on DEV_INNER. Official VAL **not** run,
TEST **not** inspected, L1 unchanged, L3 not started, pool not enlarged, no new expert families, L2 not frozen.
Returned for review **before** any official-VAL milestone.

Artifacts: `results/L2/L2_C9_TOP5.json`, `results/L2/L2_C9_FEATURE_MANIFEST.json`,
`results/L2/_ctrl/C9a_model.joblib`, `results/L2/_ctrl/_c9_oof.npz`, splits `_c9_split_*.npz`.
