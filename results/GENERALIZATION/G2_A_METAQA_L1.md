# G2 Track-A — Parameter-Free L1 Co-Scoping (MetaQA) — smoke findings

**Status:** smoke-validated on dev/val (≤250 q/hop), parameter-free, leakage-guarded, TEST untouched. Not yet scaled to full val; frozen-C11a-on-new-scope (Q2) pending.

**Question Q1:** *Can parameter-free query-entity geometry repair multi-hop L1 co-scoping?* → **YES for 2-hop (large), PARTIAL for 3-hop.**

## Method (all parameter-free, no learned weights, no gold, no hop label, no dataset id)
- **Seed** = bracketed entity in the MetaQA question (inference-safe, 100% resolve).
- **A1 residual** `r_q = q − proj_span(E) q` (E = seed-entity embeddings; SVD basis; fallback q).
- **A2 directional score** for edge e→v: `s_dir = cos(r_q, normalize(x_v − x_e))` over REAL graph edges.
- **EXPAND_DIR**: BFS from seeds, per-hop keep top-M edges by s_dir, hop budget H=3.
- **Provenance-pure adjacency** in the exact nodes.npy row space: structural from official `kb.txt` (134,741 typed KB edges, 0 unmapped, all 40,151 entities); kNN recomputed exact k=3 from nodes.npy.

## Result 1 — directional pruning >> plain expansion; mechanism = real topology (S1b)
At matched budget, `EXPAND_DIR` beats `EXPAND_ALL`: 1-hop ALL **1.000 at ~61 candidates** (vs P50's ~5038); 2-hop ALL 0.696 vs 0.34–0.47. **Beam M binds, not budget B** (DIR saturates ~61–67 cands).
**Edge-provenance control:** structural-only ≈ structural+kNN (2-hop 0.696 vs 0.652 — kNN marginally *hurts*). → the gain is REAL relational KB structure, **not reintroduced embedding similarity**. Sequential-residual (A4) ≈ single-residual (A2): no lift at this scale.

## Result 2 — P50 ∪ bounded directional expansion (the money metric, S1c)
Adding a small directionally-chosen set to the frozen P50 scope:

| Hop | P50 ALL | +M32 | +M64 | +M128 | +M256 | added cands (M32→M256) |
|---|--:|--:|--:|--:|--:|--|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | +61 → +402 |
| **2** | **0.708** | 0.868 | 0.900 | 0.916 | **0.928** | +54 → +316 |
| **3** | **0.288** | 0.316 | 0.340 | 0.388 | **0.424** | +67 → +436 |

- **2-hop: +22pt ALL (0.71→0.93)** at bounded cost — strong, cheap co-scoping win.
- **3-hop: +14pt ALL (0.29→0.42)**, monotonic in M but costlier; deep multi-gold chains only partially co-scoped by a single-residual beam.

## Reading
The G1 collapse ("evidence individually reachable but not jointly scoped") is **directly repairable** for 2-hop by parameter-free graph+embedding geometry, and partially for 3-hop. The frozen dense+SPLADE P50 router misses co-scoped multi-hop evidence that lies along **real relation edges the query residual points to** — recoverable without any learned L1.

## Open items (next, in order)
1. **Q2:** run frozen C11a L2 on the P50∪expansion scope — does the reranker recover downstream ALL@5/NDCG now that the chain is co-scoped? (heavier build: rebuild L2 features on expanded candidates)
2. Ablate A1 residual vs raw-q directional scoring (isolate the residual's contribution).
3. 3-hop: intermediate-entity re-linking (condition hop-2 direction on the actually-reached hop-1 node) vs single global residual.
4. Scale to full val; then WebQSP (needs topic-entity linker) and text datasets (NER→title seeds).

## Artifacts
`scratchpad/g2_l1_geo.py` (harness), `scratchpad/g2_smoke_s1{,b,c}.py`, `results/GENERALIZATION/_g2_smoke_s1{,b,c}.json`. Constants: K0=60, K=100, P_MAIN=50 (frozen L1); H=3, M∈{32..256} (predeclared, dev-tuned only).
