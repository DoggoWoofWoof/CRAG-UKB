# G2 · Track B — universal pool-admission + ranker (design; NO training yet)

Motivated by **Q2/Q2.1/Q2.2** (`G2_Q2_METAQA_C11A_ON_EXPANDED.md`): parameter-free geometry recovers multi-hop golds into scope, but every current gate — fusion/pool admission, C8c pool-internal ranking, and the C11a reranker even under forced exposure (0/1025 top-5) — is blind to the graph edge that co-scoped them. The fix is architectural, not a retune.

## Conceptual architecture change
**OLD** (frozen G1 pipeline):
```
L1 (topo-C P50 partition-RRF)  →  5 similarity/lexical/relational experts  →  top-50 POOL (base-fused score)
                                →  C8c XGBRanker (pool-internal)  →  C11a interaction-MLP (top-20 rerank)
```
The pool gate and both rankers see only similarity-shaped features → geometry-recovered evidence is rejected at every stage.

**NEW** (Track B target):
```
L1 UNIVERSAL CANDIDATE SCOPE  =  P50  ∪  parameter-free directional geometry (Track-A, inference-safe)
        →  UNIVERSAL POOL ADMISSION  (inference-safe graph/geometry-aware scorer; decides the top-K pool)
        →  UNIVERSAL CANDIDATE RANKER  (same feature family; produces final order)
```
The new, load-bearing stage is **UNIVERSAL POOL ADMISSION**: a dataset-agnostic scorer that can admit a candidate on **graph/geometric** evidence, not only similarity — so evidence that similarity experts rank ~500th can still enter the pool and reach the ranker.

## Inference-safe candidate feature contract (no dataset ID, no gold relation, no target-test tuning)
Every feature is per-candidate, relative/normalized, and computable at inference:
- **Retrieval-relative**: normalized retrieval rank & percentile per expert (dense/splade/offset/mixture), cross-expert agreement (how many experts place it in their top-k), min/mean normalized rank.
- **Graph / geometric** (the missing axis): directional compatibility `s_dir = cos(r_q, Δ(e→v))`; graph seed-distance (hop from nearest query seed); fraction of query seeds connected to the candidate; normalized structural support / PPR mass; expansion provenance (P50 vs geometry) and expansion hop.
- Explicitly **excluded**: dataset identifier, gold relation / supporting-fact labels, any target-specific rule, raw distribution-specific contributions used as absolute values.

## First Track-B experiment — POOL ADMISSION first (the question is recovery, not "beat C11a")
Feature ladder, **pool-admission** objective (does the recovered graph-relevant gold enter the top-K pool?):
- **S0** — retrieval-relative features only (relative rank/percentile/agreement). Reproduces the current similarity-only gate as a learned baseline.
- **S1** — S0 + normalized **structural** support (seed-distance, fraction-seeds-connected, structural/PPR mass).
- **S2** — S1 + **directional geometry** (`s_dir`, expansion provenance/hop).

Compare a **linear** admission scorer vs a **small MLP**, under **TRUE LODO** (train on all datasets except D, freeze, evaluate once on D — build the real leave-one-dataset-out split; the rotating variant is not LODO). Primary metric: **recovered-gold pool-admission rate** and downstream top-5/ALL after admission, vs the S0 baseline. The first question is **NOT "can we beat C11a?"** but **"can a universal inference-safe admission rule recover graph-relevant candidates that the similarity experts systematically reject?"** — i.e. does S2 > S1 > S0 on recovered-gold admission.

## Guardrails (unchanged from the G2 directive)
No dataset IDs; no tuning on held-out target TEST; no gold relations / supporting facts as features; do not retrain another offset MLP; do not modify C11a before an improved L1 is tested with frozen C11a; do not destroy existing frozen results / tags / artifacts / code. **HOLD large training** until the S0/S1/S2 pool-admission smoke validates the mechanism and the LODO split is built and leakage-checked.
