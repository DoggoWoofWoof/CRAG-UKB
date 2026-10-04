# L2 — C8: Top-5 Evidence Optimization

**Goal.** Given that the relevant evidence already survives L1/P50, concentrate more of it into the **first five**
retrieval positions **without** sacrificing deeper multi-gold completeness. Design decisions made on `DEV_INNER`
(from TRAIN); the official VAL evaluated **once**; TEST untouched; L1 unchanged; L3 not started; no new experts;
no unrestricted controller search.

**Result.** A **top-50 residual reranker (C8c)** over the frozen C7b fusion lifts VAL **NDCG@5 +0.090**,
**GOLD_RECALL@5 +0.086**, **ALL@5_FEASIBLE +0.159**, **MRR +0.046** — while leaving deep recall **exactly
unchanged** (ALL@50 identical; NDCG@50 in fact +0.058). All gains bootstrap-significant on held-out VAL. The
simpler archetype-retargeting route (C8a/C8b) **did not help** and was rejected on DEV.

**Final L2 policy (candidate, not frozen):** `C7b soft-archetype fusion → top-50 → C8c XGBRanker residual
reranker → final top-5`. `SAFE_TO_FREEZE_L2 = NO` (per instruction; headroom remains).

---

## 1. Why not just retarget the archetype policy? (C8a / C8b — rejected on DEV)

Smallest extension first: reuse C7b's classifier family, features, and archetype set; change **only** the TRAIN
utility target from NDCG@50 to a top-5 target. Both trained on `TRAIN_INNER`, compared on `DEV_INNER`:

| DEV_INNER (pooled) | NDCG@5 | GOLD_RECALL@5 | ALL@5 | NDCG@50 | ALL@50 |
|---|:---:|:---:|:---:|:---:|:---:|
| C7b (NDCG@50 target) | **0.8721** | 0.9059 | 0.7793 | 0.9087 | 0.9832 |
| C8a (NDCG@5 target) | 0.8717 | 0.9047 | 0.7752 | 0.9084 | 0.9824 |
| C8b (0.65·NDCG@5 + 0.35·NDCG@50) | 0.8718 | 0.9053 | 0.7772 | 0.9085 | 0.9824 |

Neither beats C7b — both are marginally *worse*. `SELECTED_ON_DEV = C7b`; `TOP5_ARCHETYPE_DISTILLATION_HELPS =
NO`. Cause (audit Task 3): the NDCG@5-optimal archetype equals the NDCG@50-optimal one for **89–90%** of queries,
so retargeting the label barely changes the policy — and per-query *global* reweighting cannot reorder *within*
a query, which is what the category-D multi-gold failures require.

## 2. C8c — top-50 residual reranker

**Pipeline.** `C7b archetype fusion → take top-50 → small residual reranker → final top-5`; candidates below
top-50 keep their base order. Because the reranker only reorders *within* the top-50, **the top-50 set is
preserved**, so ALL@50 / R@50 cannot degrade — deep recall is protected by construction.

**Model.** XGBRanker (LambdaMART, `rank:ndcg`, NDCG@5), trained on `TRAIN_INNER` top-50 pools (~1.1M rows,
22k query groups), early-stopped on `DEV_INNER`. **28 inference-safe candidate features only** — per-expert
rank & RRF contribution (dense/splade/offset/mixture/relation), relation score/rank/mask, C7 soft-archetype
weights, C7 fused score/rank, dense–splade disagreement, expert rank differences, top-10 vote count, pool
position. **No gold-derived features; no new graph/relation feature banks.**

**What it learns (feature importance).** `max_contrib` (0.67) — the single strongest expert RRF contribution for
a candidate — dominates, followed by `min_rank_across_base4` (0.16) and `pool_pos` (0.06). This is precisely the
**union-of-experts** signal that audit Task 2 found: a candidate any one expert ranks highly deserves top-5, but
linear archetype fusion averages that signal away. The reranker recovers it non-linearly.

## 3. Official VAL milestone (evaluated once; base refit on full TRAIN)

| VAL (pooled, n=6985) | C0 equal | C7b | **C8c** | C8c − C7b |
|---|:---:|:---:|:---:|:---:|
| **NDCG@5** | 0.7735 | 0.7926 | **0.8824** | **+0.0898** |
| **GOLD_RECALL@5** | 0.7935 | 0.8166 | **0.9030** | **+0.0864** |
| ANY@5 | 0.9840 | 0.9867 | 0.9941 | +0.0074 |
| **ALL@5_FEASIBLE** | 0.5817 | 0.6183 | **0.7771** | +0.1588 |
| MRR | 0.9064 | 0.9114 | **0.9569** | +0.0455 |
| NDCG@50 | 0.8401 | 0.8533 | **0.9114** | +0.0581 |
| ALL@10 | 0.7397 | 0.7820 | 0.8736 | +0.0916 |
| **ALL@50** | 0.9283 | 0.9406 | 0.9406 | **+0.0000** |

Per-dataset C8c NDCG@5: 2wiki 0.8834, musique 0.8816 (balanced; dataset ID never fed to the model).

**Paired bootstrap (2000×, 95% CI), C8c vs C7b on VAL:**

| metric | Δ | CI95 | significant |
|---|:---:|:---:|:---:|
| NDCG@5 | **+0.0898** | [+0.0861, +0.0937] | ✓ |
| GOLD_RECALL@5 | **+0.0864** | [+0.0815, +0.0911] | ✓ |
| NDCG@50 | +0.0581 | [+0.0556, +0.0607] | ✓ |
| ALL@50 | +0.0000 | [0, 0] | — (identical by construction) |

Deep recall is **provably preserved** (ALL@50 byte-identical), and the truncated NDCG@50 *rises* because
concentrating golds into the top-5 also lifts the top of the @50 ranking. Every guardrail is satisfied — nothing
is materially worse; everything is better or equal.

## 4. Gold-level rescue (DEV_INNER, C8c vs C7b)

| | rescued into top-5 | pushed out of top-5 | net |
|---|:---:|:---:|:---:|
| 2wiki | 197 | 12 | +185 |
| musique | 276 | 15 | +261 |
| **pooled** | **473** | **27** | **+446** |

Rescues come almost entirely from source bins **5–9** (358) and **10–19** (103), a few from 20–49 (12), none
from 50+ — exactly the shallow, in-scope population the audit predicted.

## 5. Remaining headroom (why L2 is not yet frozen)

- The union-of-experts oracle reaches GOLD_RECALL@5 ≈ 0.98 (2wiki) / 0.997 (musique) vs C8c's 0.90 — a
  reranker with richer cross-candidate features or a larger pool could close more.
- 2–3.5% of golds fall **outside top-50** on VAL — unrecoverable by any reranker; that is the separate
  L1-scope / L3-routing question (Task 9), deliberately untouched here.

`SAFE_TO_CONTINUE_TOP5_OPTIMIZATION = YES`.

---

## Gates

| Gate | Answer |
|------|--------|
| TOP5_HEADROOM_EXISTS | **YES** (golds sit in ranks 5–49, inside top-50 and other experts' top-k) |
| TOP5_FAILURE_MOSTLY_L1 | **NO** (full L1 coverage by construction; only 2–3.5% of golds miss top-50) |
| TOP5_FAILURE_MOSTLY_L2_RANKING | **YES** (77–98% of C7b's missed golds are in another expert's top-5) |
| TOP5_ARCHETYPE_DISTILLATION_HELPS | **NO** (C8a/C8b ≤ C7b on DEV; top-5 & top-50 archetype regimes coincide 89–90%) |
| TOP50_RERANKER_NEEDED | **YES** (archetype cannot reach it; evidence is present within top-50) |
| TOP50_RERANKER_HELPS | **YES** (VAL NDCG@5 +0.090, GOLD_RECALL@5 +0.086, both sig) |
| C8_BEATS_C7B_TOP5 | **YES** |
| C8_PRESERVES_DEEP_RECALL | **YES** (ALL@50 identical; NDCG@50 +0.058) |
| SAFE_TO_CONTINUE_TOP5_OPTIMIZATION | **YES** (union oracle ≈0.98–0.997 vs C8c 0.90) |
| SAFE_TO_FREEZE_L2 | **NO** (held open per instruction) |

**STOP (C8 complete):** failure audit + C8a/C8b + justified C8c + one official-VAL milestone done. No TEST
inspected, L2 not frozen, L3 not started, L1 unchanged, no new expensive expert families, no unrestricted
5-expert controller search. Results returned for review before any further top-5 experiment or TEST.

Artifacts: `results/L2/L2_C8_TOP5.json`, `results/L2/L2_TOP5_AUDIT.{md,json}`,
`results/L2/_ctrl/{_c8_train,_c8c}.json`, reranker `results/L2/_ctrl/C8c_reranker_fulltrain.joblib`,
splits `_c8_split_*.npz`, per-archetype caches `_c8_arch_*.npz`.
