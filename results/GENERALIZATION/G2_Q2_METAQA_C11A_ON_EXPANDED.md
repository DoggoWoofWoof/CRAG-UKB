# G2 · Q2 — Does frozen C11a convert geometry-recovered multi-hop chains into better ranking? (MetaQA)

**Question.** Track-A showed parameter-free directional geometry repairs multi-hop L1 co-scoping (2-hop ALL 0.71→0.93, 3-hop 0.29→0.42). Q2: does the **EXISTING FROZEN C11a** reranker (ZERO_SHOT source weights, unchanged) turn those newly co-scoped chains into better downstream top-5 ranking? Q2.1 (exact Relation control) and Q2.2 (oracle exposure) then localise *which* stage fails.

**Answer — NO, and the failure is now localised to THREE independent gates, every one of them blind to the graph edge.** Better L1 does not convert to downstream gain. It is *not only* the pool-admission gate: even when a recovered gold is force-admitted to the pool **and** force-exposed inside the C11a scoring window, the frozen reranker promotes it to top-5 in **0 of 1025** cases. Within the entire tested feature contract — 5 similarity/lexical/relational experts *plus* C11a's query–node embedding interaction — **no feature distinguishes a geometry-recovered multi-hop gold from a low-similarity distractor.** The one signal that identified these golds is the directional geometry / KB edge that co-scoped them, and that signal is absent from L2.

> Status: **full-val confirmed** (MetaQA VAL, 37,814 valid queries = frozen valid-query universe = all L1_PARTIAL cases; TEST untouched). C11a NOT modified, expert set NOT enlarged, no training. 18-feature contract re-expressed exactly; only the candidate SET is augmented. Q2.2 oracle is a **NOT-INFERENCE-SAFE diagnostic** (gold identity used only to *construct* the forced windows; never fed to any scorer).

---

## Result 1 — co-scoping works at full-val scale, downstream is flat→slightly down (M = 0 → 64 → 256)
Aggregate over all 37,814 valid VAL queries (frozen backbone C7b→C8c→C11a, ZERO_SHOT):

| Scope | golds added | ndcg5 | all5 | recall5 | all50 | hop2 ALL-cov | hop3 ALL-cov |
|---|--:|--:|--:|--:|--:|--:|--:|
| **M=0** (frozen P50) | 0 | **0.1145** | 0.1284 | 0.1609 | **0.2389** | 0.741 | 0.285 |
| M=64 | 25,427 | 0.1142 | 0.1276 | 0.1600 | 0.2345 | 0.888 | 0.341 |
| M=256 | 47,517 | **0.1139** | 0.1270 | 0.1592 | **0.2323** | **0.914** | **0.417** |

- **M=0 reproduces the frozen G1 ZERO_SHOT_C11A path EXACTLY** (ndcg5 0.1145, all5 0.1284, recall5 0.1609, all50 0.2389, mrr 0.1271) — the gate. The approximations (conservative splade, relation-abstain for added rows) touch added rows only.
- Coverage climbs monotonically and hugely (hop3 ALL 0.285→**0.417**, +13pt; hop2 →0.914). **Downstream ndcg5/all5/recall5 do not move**, and **all50 monotonically DECLINES** (0.2389→0.2323): naive scope expansion slightly *hurts* deep recall by displacing marginal in-pool golds. The A/B decomposition is exact — A_L1_MISSING falls by the recovered count and B_L2_RANKFAIL@5 rises by the same amount: **every recovered gold enters the scope but lands below rank 5.**

## Result 2 — WHY (crux): recovered golds are un-poolable by *every* existing expert, incl. exact Relation
Per-expert rank of the recovered golds *within the augmented ~5300-candidate scope* (dense/offset/mixture **real**; full-val M=256):

| Hop | recovered golds | **% in top-50 pool** | best-expert rank (median) | dense | offset | mixture |
|---|--:|--:|--:|--:|--:|--:|
| 2 | 11,484 | **4.4%** | 437 | 1086 | 950 | 952 |
| 3 | 35,981 | **0.3%** | 612 | 665 | 3049 | 3002 |

The frozen reranker operates on the **top-50 pool**; recovered golds sit at median rank **~440–610** under the *best* of any similarity expert, so **0–4% ever enter the pool.** The universal **relational** heads (offset/mixture) rank them *no better than, and at hop-3 far worse than, raw dense.*

### Q2.1 — exact Relation expert control (CPU, no encoder pass; full 1500 q/hop)
The KB **Relation** expert fires on a candidate only if its title is a title-mention target of the query's topics (`title(d) ∈ query_edges(q)`); otherwise it ABSTAINS (RRF contribution exactly 0 → cannot admit). The expert **is alive** on MetaQA — it fires on **85.6 %** of hop-1 in-scope golds and 7.6 % at hop-3 — yet it fires on **0 of 5,352** recovered multi-hop golds (hop2 0/1214, hop3 0/4136; frac 0.0000).

→ **CURRENT_EXISTING_EXPERT_SET_LACKS_RECOVERY_SIGNAL = YES.** The one KB-derived expert also carries no signal for the recovered evidence; the structural conclusion does not depend on a cosine pass, because an abstaining expert cannot rank an abstained gold above candidates that do fire.

## Result 3 — Q2.2 oracle exposure: even *forced* exposure to C11a does not promote them
**NOT INFERENCE SAFE** (subset 300 q/hop at M=256; gold identity used only to build the forced windows). Two oracles per recovered gold:

| Hop | recovered | natural in pool | **ORACLE A** (force→pool, normal window): reach C11a window / top-20 / top-5 | **ORACLE B** (force→C11a window vs top-19): **top-5** |
|---|--:|--:|--:|--:|
| 2 | 173 | 5 | 2 / 2 / **0** | **0 / 173** |
| 3 | 852 | 1 | 20 / 20 / **0** | **0 / 852** |

- **Oracle A** (force the gold into the top-50 pool, then run the *normal* C8c→C11a top-20 window): C8c immediately re-buries it at the bottom of the pool — only **2/173 (hop2)** and **20/852 (hop3)** even reach the 20-candidate C11a window, and **0** reach top-5.
- **Oracle B** (force the gold *into* a 20-candidate C11a window = {gold} ∪ top-19-by-C8c, so C11a must score it against the strongest competitors): C11a ranks it top-5 in **0 / 1025**. (`top-20 = 100 %` is trivial — the window has only 20 candidates; **top-5 is the informative metric**, and it is 0.) C11a systematically places the recovered gold 6th–20th out of 20, *below* the similarity-strong candidates, every time — because its features (the same 5 expert contributions/ranks + the query–node embedding dot-product) all read "low-similarity candidate."

### Classification of recovered golds (per hop; the decisive partition is Oracle-B outcome)
| Category | hop2 | hop3 |
|---|--:|--:|
| **1. SCOPE_RECOVERED_BUT_POOL_REJECTED** (not naturally poolable) | 168/173 | 851/852 |
| **2. POOL_EXPOSED_BUT_C11A_REJECTED** (force-exposed, C11a still ranks it out of top-5) | 173 | 852 |
| **3. C11A_EXPOSED_AND_PROMOTED_TOP5** (fixing pooling alone would recover it) | **0** | **0** |

Category 3 = 0 is the load-bearing number: **fixing pool admission alone recovers nothing** — the reranker itself has no signal to rank the exposed evidence. The bottleneck is the *feature contract*, not any single stage.

## The three gates (all downstream-blind to the graph edge)
1. **Fusion / pool-admission gate** — 99.6 % of recovered golds never enter the top-50 pool (best similarity expert ranks them ~440–610; exact Relation abstains on 100 %).
2. **C8c pool-internal gate** — force-admitted to the pool, C8c re-buries them; only ~1–2 % reach the top-20 C11a window.
3. **C11a reranker gate** — force-exposed inside the C11a window, 0/1025 reach top-5.

## Decision & implication for Track B
**Track-A parameter-free co-scoping is validated as necessary but not sufficient. Do NOT tune C11a; do NOT enlarge the current expert set** — the exact Relation control already shows the *entire* current expert family (similarity + lexical + KB-title-mention) is silent on this evidence, and the oracle shows the reranker is too. Track B's universal L2 must **inject the graph-relational signal as an inference-safe candidate feature** — the directional score `s_dir = cos(r_q, Δ(e→v))`, graph seed-distance / hop, fraction-of-seeds-connected, normalized structural support — **upstream of pooling** (a *universal pool-admission* stage), so the pool can *admit* and the reranker can *promote* graph-recovered evidence. Similarity-only fusion, and a similarity-only reranker, structurally cannot.

**Wording now supported by the finished diagnostics:** within the tested feature contract, the graph edge that co-scoped a recovered multi-hop gold is the only evidence of its relevance — no similarity, lexical, relational-head, KB-title-mention, or query–node-embedding feature separates it from a low-similarity distractor.

### Scope caveat
Universe = frozen valid-query set (≥1 in-scope gold under P50 = all L1_PARTIAL cases; M=0 == G1). Fully-L1-failed queries (0 in-scope golds) need a C6-feature-cache extension — a separate follow-up; they contribute 0 uniformly at M=0 and are a minority. The Q2.2 oracle is a 300 q/hop diagnostic and is explicitly NOT-INFERENCE-SAFE.

## Artifacts
`scratchpad/_q2_eval.py` (harness incl. `run_oracle`), `scratchpad/modal_q2.py` (crm job, `--oracle 1`), `scratchpad/_q2_1_relation_coverage.py` (Q2.1 CPU control), `results/GENERALIZATION/_q2_metaqa.json` (full-val M∈{0,64,256}), `results/GENERALIZATION/_q2_oracle_metaqa.json` (Q2.2 oracle), `results/GENERALIZATION/_q2_1_relation_coverage.json` (Q2.1). NEW_ENCODER_FORWARD_PASSES=0; TARGET_TEST_TOUCHED=NO.
