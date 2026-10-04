# L2 Top-5 Failure Audit (C8, pre-training)

**Scope.** Audits the deployed **C7b** soft-archetype policy for where recoverable top-5 evidence currently
sits, on a deterministic development split carved from TRAIN (the official VAL has been used heavily across
C0–C7 and is held for the single C8 milestone). No TEST inspected.

**Development split.** Per dataset, valid TRAIN queries are split **90/10** into `TRAIN_INNER` / `DEV_INNER`,
deterministically (stable per-qi hash) and **stratified** by (gold multiplicity ∈{1,2,3+}) × (relation signal
present) × (dense/splade disagreement above/below dataset median) — 12 strata each. Sizes: 2wiki
inner 9449 / dev 1048; musique inner 12554 / dev 1394.

> **L1 vs L2 terminology — a structural fact about this corpus.** In `l2_corpus`, `gold_count` **equals the
> in-scope gold count for every query** (verified across all splits): the L2 working corpus is **full-L1-coverage
> by construction**. The pre-P50 L1 scope-coverage failures were filtered out upstream. Therefore Task-1
> categories **A (L1_ANY_FAIL)** and **B (L1_PARTIAL)** are **empty here** — every audited top-5 failure is the
> *recoverable* L2-ranking population (C/D/E). The `L1_SCOPE_COVERAGE` question (golds absent from P50 entirely)
> is a separate routing/L3 matter and is **out of C8 scope** (Task 9). Where golds do fall outside top-50, we
> report it (Task 8) but do not act on it.

---

## Task 1 — top-5 failure buckets (DEV_INNER, C7b)

| dataset | E: all in-scope gold in top5 | D: some top5, others below | C: none of the in-scope gold in top5 |
|---|:---:|:---:|:---:|
| 2wiki | 0.773 | 0.225 | 0.002 |
| musique | 0.812 | 0.186 | 0.002 |

Category **C is ~0.2%** — almost every query already has *some* gold in top5 (ANY@5 ≈ 0.998). The live failure
is category **D (18–23%)**: **multi-gold** queries where one gold is in the top5 but its siblings sit just
below. Per-gold C7b rank distribution confirms the misses are shallow:

| gold rank bin | 2wiki | musique |
|---|:---:|:---:|
| 0–4 | 0.881 | 0.913 |
| **5–9** | **0.070** | **0.064** |
| 10–19 | 0.024 | 0.021 |
| 20–49 | 0.013 | 0.002 |
| 50+ | 0.013 | 0.0006 |

~9–12% of golds sit in ranks 5–49 — present in scope, just outside top-5. This is the recoverable population.

## Task 2 — where do the missed golds live? (fusion vs scoring; DEV_INNER)

Per-expert standalone GOLD_RECALL@5, the single-best-expert-per-query oracle, and the **union** of experts'
top-k — the decisive fusion-vs-scoring diagnostic:

| | 2wiki | musique |
|---|:---:|:---:|
| C7b GOLD_RECALL@5 | 0.906 | 0.906 |
| best per-expert standalone | relation 0.83 | mixture 0.99 |
| **single-expert-per-query oracle** | 0.928 | 0.994 |
| **union of experts' top-5** | **0.977** | **0.997** |
| union of experts' top-10 | 0.988 | 0.999 |

**Of the golds C7b misses from its top-5:**

| | in *some* expert's top-5 | in some expert's top-10 (only) | deep under **every** expert |
|---|:---:|:---:|:---:|
| 2wiki (293 missed) | **77.5%** | 7.8% | 14.7% |
| musique (277 missed) | **97.8%** | 1.1% | 1.1% |

**Verdict: the top-5 failure is a fusion/ranking problem, not candidate scoring / expert weakness.** 77–98% of
C7b's missed golds are already inside *another* expert's top-5, and ≥85% within some expert's top-10. The
evidence is present; the linear archetype fusion just fails to place it in *its* top-5. Only 1–15% are deep
under every expert (a scoring/L1 residual).

## Task 3 — archetype oracle under a top-5 objective (TRAIN_INNER)

Re-deriving the oracle-winning archetype under **NDCG@5** instead of NDCG@50:

| | 2wiki | musique |
|---|:---:|:---:|
| same winner (NDCG@5 vs NDCG@50) | **0.890** | **0.900** |
| equal→oracle NDCG@5 gain (ceiling) | +0.0495 | +0.0424 |
| equal→oracle GOLD_RECALL@5 | 0.880→0.925 | 0.899→0.941 |

The NDCG@5-optimal archetype **is the same** as the NDCG@50-optimal one for **89–90%** of queries. The
archetype family cannot separate a top-5 regime from a top-50 regime — so **retargeting the distillation label
to NDCG@5 cannot move top-5** (confirmed by C8a/C8b below). Per-query *global* reweighting also cannot reorder
*within* a query, which is exactly what category-D failures require.

## Task 8 — top-50 pool (maximum reach of a residual reranker)

| | DEV_INNER 2wiki | DEV_INNER musique | VAL 2wiki | VAL musique |
|---|:---:|:---:|:---:|:---:|
| fraction of golds inside C7b top-50 | 0.987 | 0.999 | 0.965 | 0.979 |
| of the top-50 pool: present-but-not-top5 | 10.8% | 8.6% | 22.1% | 15.2% |
| golds missing top-50 (unrecoverable here) | 31 / 2460 | 2 / 3196 | 250 / 7066 | 193 / 9123 |

**96.5–99.9% of golds are already inside C7b's top-50**, and a large slice of that pool (15–22% on VAL) is
present-but-not-top5 — the exact population a top-50 reranker can rescue. Only **2–3.5% of golds miss top-50**
entirely (the L1/L3 residual, out of C8 scope).

---

## Audit conclusions (feed C8 design)

1. **Headroom exists and is recoverable** — golds sit just below the top-5 cutoff, inside the top-50 and inside
   other experts' top-k. `TOP5_HEADROOM_EXISTS = YES`.
2. **The failure is L2 ranking, not L1 coverage or expert weakness** — full L1 coverage by construction; 77–98%
   of misses are in another expert's top-5. `TOP5_FAILURE_MOSTLY_L2_RANKING = YES`, `TOP5_FAILURE_MOSTLY_L1 = NO`.
3. **Archetype reweighting is the wrong tool** — the top-5 and top-50 archetype regimes coincide (89–90%), and
   global per-query weights cannot reorder within a query. → **build the Task-7 top-50 residual reranker (C8c)**
   rather than a retargeted archetype policy.

Artifacts: `results/L2/L2_TOP5_AUDIT.json`, caches `results/L2/_ctrl/_c8_arch_*.npz`, splits
`results/L2/_ctrl/_c8_split_*.npz`.
