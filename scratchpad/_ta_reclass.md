
---

## Reclassification of B1.1 – B1.9 (nothing deleted, nothing rewritten)

Per the directive, the prior line is relabelled, not discarded:

```
POST_P50_STRUCTURAL_ADMISSION_DIAGNOSTICS = {Track-A smokes S1/S1b/S1c, Q2, B1.1 … B1.9}
```

Every one of them measured the same object: **a fixed candidate pool, plus structural candidates appended
after selection, with a policy deciding how many appended candidates to admit.** None of them recomputed a
partition vote. Their internal validity is untouched; what changes is what they are evidence *about*.

### Which conclusions survive, and why

**Survive fully — they are statements about the substrate or about L2, not about where structure enters.**

| conclusion | why it is independent of the architecture correction |
|---|---|
| **Q2's three blind gates** — recovered multi-hop golds are un-poolable by every expert (median best-expert rank ~440–610; exact Relation abstains on 100 %; forced into a C11a window, promoted to top-5 in **0 / 1025**) | This is a property of the *documents*, not of how they arrived. Under corrected Track-A the same golds arrive as ordinary partition members with full expert scores — their dense/SPLADE/offset/mixture/relation values are unchanged, so the ranking failure is identical. If anything the corrected architecture **strengthens** this: it removes the append mechanism that Q2 might have been blamed for, and the gates still stand. |
| **`musique-graph-reachability-ceiling`** — MuSiQue's corpus graph does not contain the gold chains (~31 % all-golds-reachable ceiling) | Pure substrate reachability. No admission policy involved. |
| **B1.2's discovery-vs-verification observation** — the graph reaches golds that retrieval does not | Substrate-level, and independently re-confirmed here by the expansion diagnostics (a large fraction of expanded nodes land in partitions BASE did *not* select). |
| **B1.9's methodological negatives** — the Simpson's-paradox inversion across corpus clusters, the chance control for perfect separators (expected 1.60 of 12, observed 1), LOCO with in-fold descriptor selection | These are statistical lessons about fitting six corpora, valid for any policy family. |

**Survive only as diagnostics of the surrogate — they no longer bear on Track-A.**

| conclusion | status now |
|---|---|
| `RESERVE_POLICY_GENERALIZATION = NO` (B1.4) | valid **for the post-P50 reserve**; the corrected architecture has no reserve, so this cannot be cited as "structure does not generalise" |
| B1.5 / B1.6 query-local adaptive-R, interaction gates | valid for the reserve; moot for Track-A |
| B1.7 conditional-shift `P(D|P0)` diagnosis | valid for the reserve; superseded within its own line by B1.8 |
| B1.8 "the reserve is a per-domain scalar", in-domain ceiling | valid for the reserve; revised by B1.9 to "mostly a *global* scalar" |
| B1.9 `CORPUS_LEVEL_RESERVE_GENERALIZATION = NO`, global R=16 recovers 84.9 % | valid for the reserve |

**Does not survive as stated.**

| claim | correction |
|---|---|
| the Track-A headline **"MetaQA 2-hop ALL +22pt / 3-hop +14pt co-scoping"** | Measured as a **node-level set union** `in_p50 OR in_expansion` against a P50 computed *before* expansion, and **seeded from bracketed entity strings** (`metaqa_ent_<name>`) that the B0 feature audit had already ruled dataset-identifying (coverage 1.0 on MetaQA, 0.0 elsewhere). It is neither a partition-level result nor a retrieval-seeded one. The corrected, retrieval-seeded, partition-level measurement is reported above and is far smaller. |
| B1.1–B1.9 "operate at L1" | They do not. Their `P_MAIN=50` selects **50 documents** from a node-level RRF; the genuine partition P50 is ~5000 documents. B1.x is an **L2 pool-admission** experiment throughout. |
