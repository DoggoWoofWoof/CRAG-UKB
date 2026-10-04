# G2 L1 — KB MULTI-HOP / SET-AWARE PARTITION ROUTING — FINAL REPORT

**Verdict: `FREEZE_L1 = NO`. The standing universal policy remains `B6_S4_F6_Ms64_Mr32`, unchanged.**
No candidate cleared promotion. The parameter-free router search axis is, however, **exhausted**:
the one mechanism that materially moves MetaQA hop3 is statistically incompatible with the text
corpora, no query-local signal separates the two regimes, and 87% of the residual hop3 gap is
outside what the frozen contract permits any B≤6 boundary router to fix.

Contract honoured throughout: exactly 50 canonical C partitions on every query, no stray nodes,
no learned model / MLP / attention / fitted weight or threshold, no dataset identity or
corpus-specific branch, no benchmark hop label or gold at inference, no TEST, no L2/L3, no
training, `NEW_ENCODER_PASSES = 0`, `MAX_HOPS = 3`.

---

## 1. Headline

| | MetaQA hop2 | MetaQA hop3 | worst corpus vs F6 | promotable |
|---|---|---|---|---|
| BASE (un-routed P50) | 0.7237 | 0.2568 | — | — |
| **`F6` safe baseline (B6_S4_F6_Ms64_Mr32)** | 0.7297 | 0.2583 | — | standing |
| `ALT_DEEP_B6` (best that clears every gate) | 0.7372 | 0.2628 | −0.0035 (ns) | see §9 |
| `CANON_DEEP_B6` (best on the primary diagnostic) | **0.7492** | **0.2703** | **−0.0175 (p<1e-4)** | **no** |
| direct COMBINED (reach ceiling) | 0.7808 | 0.3228 | — | — |
| oracle B6 | 0.8093 | 0.3589 | — | — |

The phase found a real, fold-replicated, mechanistically explained lever worth **4× on hop2 and
9× on hop3** over the safe baseline — and then established, with paired significance, that it
cannot be shipped universally.

---

## 2. STEP 1 — hop-stratified diagnosis

`diag/hop_diagnosis.json`. ALL-gold coverage, oracle over B ∈ {1,2,4,6}, and per-query failure
mode for every uncovered query.

| corpus | BASE | B6 | oracle B1 | oracle B6 | dominant failure modes |
|---|---|---|---|---|---|
| metaqa | 0.6587 | 0.6612 | 0.7072 | 0.7222 | CAPACITY 246, REACH 226, REACH_AT_ROUTER_M 85, RANKING 78, EVICTION 26, CO_SELECT 16 |
| webqsp | 0.7618 | 0.7646 | 0.7956 | 0.8013 | REACH 188, CAPACITY 65, REACH_AT_ROUTER_M 47, EVICTION 21, RANKING 12, CO_SELECT 1 |
| 2wiki | 0.9375 | 0.9435 | 0.9610 | 0.9610 | REACH 52, REACH_AT_ROUTER_M 45, RANKING 11, EVICTION 5 |
| musique | 0.9565 | 0.9635 | 0.9905 | 0.9920 | REACH_AT_ROUTER_M 39, EVICTION 11, REACH 11, RANKING 10, CO_SELECT 2 |
| hotpot | 0.9345 | 0.9505 | 0.9695 | 0.9700 | REACH 37, REACH_AT_ROUTER_M 34, RANKING 23, EVICTION 5 |
| squad | 0.9805 | 0.9875 | 0.9975 | 0.9975 | RANKING 11, REACH_AT_ROUTER_M 10, EVICTION 3, REACH 1 |

**Answer to the STEP-1 question.** On the KB corpora the dominant modes are *capacity* and
*reach*, not ranking: on MetaQA only 120 of 682 uncovered queries (17.6%) are
ranking/eviction/co-selection failures a boundary router could address. WebQSP is post-hoc
bucketed only (analysis-only; never an inference feature).

## 3. STEP 2 — path provenance cache

`paths/path_{ds}.npz` (6 corpora), `paths_manifest.json`. The frozen expansion was replayed with
parent pointers, recording for every scored node its originating seed slot, its ordered node
chain `seed → a → b → node`, chain length, and the deduplicated ordered canonical-partition
chain. **Parity EXACT on all six corpora** (node-identity mismatch 0 of 196k–318k nodes).

One honest caveat found and reported rather than suppressed: the frozen `vmeta` records the first
hop at which a node was *scored*, while the parent chain records *scope-entry* depth. A node can
be scored at hop h, lose the beam, and re-enter scope later, so the two disagree on a minority of
nodes. Parity is therefore asserted on node identity, and the depth disagreement is reported as a
statistic.

| corpus | chains/query | depth histogram {1,2,3} | distinct partitions/chain |
|---|---|---|---|
| metaqa | 63.8 | 6 749 / 46 359 / 74 423 | 2.837 |
| webqsp | 62.2 | 8 745 / 33 547 / 45 945 | 2.716 |
| 2wiki | 54.4 | 4 703 / 45 979 / 58 096 | 2.936 |
| musique | 63.0 | 17 420 / 79 763 / 28 783 | 2.356 |
| hotpot | 63.9 | 7 836 / 98 082 / 21 869 | 2.657 |
| squad | 54.6 | 41 167 / 53 404 / 14 549 | 1.980 |

## 4. STEP 3 — set oracles (`oracle/set_completion.json`)

`A` independent M32 · `A64` independent M64 · `D` chain-union independent · `P` path-set atomic ·
`C` path groups + retrieval singletons.

| corpus | BASE | A | **A64** | D | P | C |
|---|---|---|---|---|---|---|
| metaqa | 0.6587 | 0.7222 | **0.7377** | 0.7317 | 0.7247 | 0.7317 |
| webqsp | 0.7618 | 0.8013 | **0.8055** | 0.8041 | 0.7879 | 0.8041 |
| 2wiki | 0.9375 | 0.9610 | **0.9620** | 0.9615 | 0.9435 | 0.9615 |
| musique | 0.9565 | 0.9920 | **0.9925** | 0.9915 | 0.9665 | 0.9915 |
| hotpot | 0.9345 | 0.9700 | **0.9710** | 0.9705 | 0.9470 | 0.9705 |
| squad | 0.9805 | 0.9975 | 0.9975 | 0.9975 | 0.9855 | 0.9975 |

**`A64 ≥ D ≥ C` on every corpus.** Path intermediates add *no reach* beyond a slightly deeper
independent challenger pool, and `C = D` exactly everywhere means atomicity costs nothing at the
oracle level. An earlier reading of `C > A` was an artifact of comparing M=64 chains against M=32
independent challengers; the `A64` control removes it. **Path provenance does not extend reach.**

## 5. STEPS 4–5, 7 — joint subset selection and the J-families

Exhaustive DFS-pruned search over replacement subsets (combinatorial, not learned), families J1
path-complete promotion, J2 frontier coverage, J3 incumbent-aware group swap with abstention, J4
structural chain + retrieval complement, J5 path-stable singleton. `KEEP BASE` (X = boundary) is
representable in every family, so full abstention is always available.

Two structural results:

- **Modular-redundancy theorem, confirmed empirically.** Under a sum-of-RRF objective an atomic
  group can never change a sum-maximising selection whenever its members are individually
  available. `J1SR_B6` reproduced `F6` *exactly* on all six corpora. Path structure can therefore
  only act by *restricting* singleton promotion or through a gate — never as an additional option.
- **J1's KB gain is not atomicity.** The struct-only control decomposes it:

  | MetaQA | hop2 | hop3 |
  |---|---|---|
  | `F6` (3 channels) | +0.0060 | +0.0015 |
  | `ES` (identical router, retrieval channel removed) | +0.0180 | +0.0105 |
  | `J1` (`ES` + atomic path groups) | +0.0195 | +0.0120 |

  **87.5% of the recoverable hop3 gain is channel exclusion; atomicity contributes +0.0015.**
  This corrected the round-2 reading that hop3 was a co-selection/atomicity phenomenon.

## 6. Rounds — observed failure → hypothesis → change → result

**Round 1 (J1/J3 + reach/atomicity controls).** J1 wins MetaQA (+0.0105) precisely by forbidding
retrieval singletons, which is what makes it lose on text (webqsp −0.0014, musique −0.0005);
J1R restores text and loses the KB gain; J1SR collapses onto F6.

**Round 2 (J4 repair operator).** *Hypothesis:* the mechanism must be a repair on top of F6, not
a rival objective — complete a chain F6 already anchored. *Result:* nearly a no-op at B=6 (most
J4 variants collapse to F6) because F6 has already spent all six slots on retrieval singletons,
so chains are rarely anchored with missing members. Only at B=4 does completion beat F6 on the KB
axis (`J4_ANCHOR_B4`, worst +0.0035, macro +0.0062).

**Round 3 (modality de-duplication).** *Observed failure:* the struct-only control shows the
retrieval channel is what suppresses hop2/hop3. *Mechanism found at the source:*
`_l1ps_router.ret_challengers` reads `z["ret_rrf"]`, the dense+SPLADE **rrf200 node
continuation** — and `z["base_rank"]` is that *same* evidence aggregated to partitions. The
canonical and retrieval channels are one modality at two granularities, so F6 gives lexical
evidence two correlated votes and structure one. *Measured* (`diag/channel_redundancy.json`):
the fraction of retrieval challengers the canonical channel already ranks is **1.0000** on
musique and squad, 0.9986 on 2wiki, 0.9534 metaqa, 0.9378 webqsp, and **0.8148** on hotpot.
*Change:* replace the summed lexical term with `MINRANK` = 1/(K0+min(cpos,rpos)), `CANON` =
canonical precedence, or `NOVEL` = retrieval votes only when it has no canonical rank, each
optionally gated to fall back to exactly F6. *Result:* the KB axis moves as predicted —
MetaQA hop3 +0.0015 → **+0.0135**, hop2 +0.0060 → **+0.0255** — beating both ES and J1. But
**every** de-dup variant pays on HotpotQA (F6 +0.0160 → MINRANK +0.0025 → CANON −0.0015), the one
corpus where the retrieval channel is genuinely non-redundant. `CANON ≡ NOVEL` bit-for-bit on all
six corpora, a free self-consistency check that passed. Hard structural priority (`PRIO`) is a
disaster (webqsp −0.0204) and was dropped.

**Round 4 (modality-balanced slot allocation).** *Observed failure:* re-weighting the score cannot
be right for corroboration and redundancy at once. *Hypothesis:* share **slots**, not score
weight — leave the frozen F6 score untouched and let the two modalities alternate as they claim
the B boundary slots; 1:1 is the parameter-free fair split. *Result:* text is protected —
`ALT_STRLEAD_SEED_B6` reaches hotpot **+0.0165 > F6 +0.0160** with 2wiki and squad exactly at F6 —
but hop3 only reaches +0.0030. *New failure exposed:* MetaQA wants essentially all six slots
structural and HotpotQA wants them lexical, so **any fixed split, at any ratio, lands between the
two regimes and gets neither**. Fixed rules are exhausted.

**Round 5 (is there a query-local allocation signal?).** This round is a *test*, not a family. Nine
parameter-free rank/count/percentile features (channel concordance, retrieval redundancy, channel
Jaccard, deep structure-only count and fraction, seed support, structural top-rank gap, retrieval
top canonical rank, boundary structural support) were scored by how well they separate the queries
where de-duplication *helps* from those where it *hurts*. Gold labels the outcome only; it is
never a feature. The first pass was underpowered (the ALL-coverage label moves on only 17–41
queries per corpus), so the test was re-run with a per-query gold-partition delta — an order of
magnitude more support (pooled n = 359).

*Result — a clean, well-powered null.*

| feature | pooled AUC | mean \|AUC−0.5\| within corpus | sign agreement | fold-stable |
|---|---|---|---|---|
| f_concord | 0.511 | 0.122 | 2/6 | 1/6 |
| f_ret_redund | 0.545 | 0.073 | 5/6 | 2/6 |
| f_jaccard | 0.577 | 0.140 | 4/6 | 2/6 |
| f_deep | 0.517 | 0.104 | 2/6 | 3/6 |
| f_deep_frac | 0.475 | 0.050 | 0/6 | 2/6 |
| f_seed | 0.515 | 0.019 | 3/6 | 1/6 |
| f_topgap | 0.447 | 0.095 | 3/6 | 1/6 |
| f_ret_top_c | 0.419 | 0.151 | 0/6 | 2/6 |
| f_bnd_struct | 0.566 | 0.117 | 1/6 | 1/6 |

No feature has both a consistent sign across corpora and fold stability. `f_ret_redund` has the
most consistent direction (5/6) and the *least* discriminative power (0.073). The reason is visible
in the label distribution: de-duplication is **corpus-constant, not query-varying** — 80–96% of
every corpus's outcome sign is predicted by the corpus alone (metaqa 165 up / 40 down; 2wiki 2/24;
musique 1/25; hotpot 3/41; squad 1/16). Within a corpus there is almost nothing left to separate.
A rule keyed on any of these features would be dataset identity in disguise, which the contract
forbids. This independently reproduces the B1.8/B1.9 finding that the useful quantity here is a
per-domain scalar rather than a query-level one.

## 7. STEP 12 — mandatory MetaQA comparison table

| row | hop1 | hop2 | hop3 |
|---|---|---|---|
| BASE | 0.9955 | 0.7237 | 0.2568 |
| **B6 safe baseline (`F6`)** | 0.9955 | 0.7297 | 0.2583 |
| direct STRUCT32 | 0.9955 | 0.7778 | 0.3198 |
| direct STRUCT64 | 0.9970 | 0.8078 | 0.3348 |
| direct RET32 | 0.9970 | 0.7267 | 0.2598 |
| direct COMBINED_32_32 | 0.9970 | 0.7808 | 0.3228 |
| oracle B1 | 0.9985 | 0.7958 | 0.3273 |
| oracle B2 | 0.9985 | 0.8018 | 0.3514 |
| oracle B4 | 0.9985 | 0.8078 | 0.3544 |
| oracle B6 | 0.9985 | 0.8093 | 0.3589 |
| `ALT_STRLEAD_SEED_B6` | 0.9955 | 0.7327 | 0.2598 |
| `ALT_DEEP_B6` | 0.9955 | 0.7372 | 0.2628 |
| `ES_B6` | 0.9955 | 0.7417 | 0.2673 |
| `J1_B6` | 0.9955 | 0.7432 | 0.2688 |
| `MINRANK_DEEP_B6` | 0.9955 | 0.7417 | 0.2703 |
| `CANON_DEEP_B6` | 0.9955 | **0.7492** | **0.2703** |

hop1 is unchanged (+0.0000) for every candidate, so no configuration here is the STEP-8 failure
mode of buying aggregate ALL with hop1 while leaving hop3 alone.

## 8. STEP 13 — RECOVERED_HOP_HEADROOM

`(NEW_h − BASE_h) / (DIRECT_COMBINED_h − BASE_h)`; hop2 headroom 0.0571, hop3 headroom 0.0660.

| config | hop2 recovered | hop3 recovered |
|---|---|---|
| `F6` safe baseline | 10.5% | 2.3% |
| `ALT_STRLEAD_SEED_B6` | 15.8% | 4.5% |
| `ALT_DEEP_B6` | 23.6% | 9.1% |
| `ES_B6` | 31.5% | 15.9% |
| `MINRANK_DEEP_B6` | 31.5% | 20.5% |
| `J1_B6` | 34.2% | 18.2% |
| `CANON_DEEP_B6` | **44.7%** | **20.5%** |

Oracle-gap fraction closed (ALL, vs oracle B6): metaqa `F6` 3.9% → `ALT_DEEP` 10.2% →
`CANON_DEEP` 20.5%; but on hotpot the same ordering runs 45.1% → 35.2% → **−4.2%**.

WebQSP (ALL / ANY, BASE 0.7618 / 0.9436): `F6` 0.7646 / 0.9457 · `ALT_DEEP_B6` 0.7653 / 0.9471 ·
`ALT_STRLEAD_SEED_B6` 0.7653 / 0.9471 · `CANON_DEEP_B6` 0.7625 / 0.9471 · `J1_B6` 0.7604 / 0.9450.

## 9. STEPS 9–10 — selection, nested folds, LODO

Deterministic sha1 DISCOVERY/VALIDATION folds; promotion judged on VALIDATION; LODO re-runs the
whole selection on five corpora and reads off the held-out sixth. **No TEST was touched.**

A correction that decided the phase: the per-round scoreboards pair every config against BASE
only. Criterion 1 must also be enforced against the **safe baseline** — a rule that beats the
un-routed P50 while significantly regressing the router we already trust is not a safe promotion.
All 90 configs were therefore re-scored against both (`scoreboard_consolidated.json`).

| config | sig. reg. vs BASE | sig. reg. vs F6 | sig. **wins** vs F6 | mq_h3 | mq_h2 | webqsp | worst text vs F6 |
|---|---|---|---|---|---|---|---|
| **`ALT_DEEP_B6`** | 0 | **0** | **1** | +0.0060 | +0.0135 | +0.0035 | −0.0035 |
| `ALT_STRLEAD_SEED_B6` | 0 | 0 | 0 | +0.0030 | +0.0090 | +0.0035 | −0.0005 |
| `F6_B6` (standing) | 0 | 0 | 0 | +0.0015 | +0.0060 | +0.0028 | ±0.0000 |
| `CANON_DEEP_B6` | 0 | **2** | 1 | +0.0135 | +0.0255 | +0.0007 | −0.0175 |

39 of 90 configs are in the safe set (zero significant regression against either baseline).

`ALT_DEEP_B6` per corpus, paired vs F6: metaqa **+0.0040 (p = 0.0078)**, webqsp +0.0007 (ns),
2wiki −0.0015 (p = 0.375), musique +0.0000, hotpot −0.0035 (p = 0.065), squad +0.0000.

`CANON_DEEP_B6` per corpus, paired vs F6: metaqa **+0.0105 (p < 1e-4)**, hop2 +0.0195
(p = 0.0002), hop3 +0.0120 (p = 0.0215); 2wiki **−0.0070 (p = 0.0013)**, hotpot **−0.0175
(p < 1e-4)**.

**Fold replication is what separates them.** `CANON_DEEP_B6` replicates on MetaQA across folds
(DISCOVERY +0.0109 p = 0.052, VALIDATION +0.0151 p = 0.017). `ALT_DEEP_B6` does not
(DISCOVERY +0.0030 p = 0.63, VALIDATION +0.0101 p = 0.099), and its hop2/hop3 gains are
individually non-significant (p = 0.0625, p = 0.250). With 90 configs compared, a single
p = 0.0078 on the selected maximum is exactly what the nested folds exist to discount.

**LODO under the corrected criterion:**

| held out | selected on the other five | held-out vs BASE | held-out vs F6 |
|---|---|---|---|
| metaqa | `J4_ANCHOR_FRONT_B6` | +0.0020 (p 0.64) | −0.0005 (ns) |
| webqsp | `ALT_DEEP_B6` *(= global)* | +0.0035 (p 0.55) | +0.0007 (ns) |
| 2wiki | `ALT_DEEP_B6` *(= global)* | +0.0045 (p 0.06) | −0.0015 (ns) |
| musique | `ALT_DEEP_B6` *(= global)* | +0.0070 (p 0.03) | +0.0000 |
| hotpot | `CANON_SEED_B6` | +0.0035 (p 0.28) | **−0.0125 (p < 1e-4)** |
| squad | `ALT_DEEP_B6` *(= global)* | +0.0070 (p 0.003) | +0.0000 |

4 of 6 folds reselect the global winner — but the two *informative* held-outs both diverge. With
MetaQA removed the hop3 criterion goes vacuous and the procedure picks something else; with
HotpotQA removed it picks a rule that then significantly damages HotpotQA. The selection is not
LODO-stable in the directions that matter.

## 10. STEP 14 — failure analysis of MetaQA hop3

Every hop3 query whose selection changed vs `F6`, classified; and the still-uncovered hop3 queries
decomposed.

| config | RECOVERED | DAMAGED | PARTIAL | NEUTRAL |
|---|---|---|---|---|
| `CANON_DEEP_B6` | 9 | 1 | 150 | 501 |
| `MINRANK_DEEP_B6` | 9 | 1 | 117 | 508 |
| `J1_B6` | 8 | 1 | 156 | 497 |
| `ES_B6` | 6 | 0 | 88 | 463 |
| `MINRANK_SEED_B6` | 6 | 1 | 65 | 303 |
| `ALT_STRLEAD_DEEP_B6` | 1 | 0 | 45 | 263 |
| `ALT_STRLEAD_SEED_B6` | 1 | 0 | 24 | 152 |

Still-uncovered hop3 queries under `CANON_DEEP_B6` (486 of 666):

| mode | count | share |
|---|---|---|
| CAPACITY (more than B=6 partitions missing) | 213 | 43.8% |
| REACH (gold not in the candidate pool at all) | 211 | 43.4% |
| RANKING | 44 | 9.1% |
| EVICTION | 9 | 1.9% |
| CO_SELECT | 9 | 1.9% |

**424 of 486 (87.2%) are capacity or reach failures** — unreachable for *any* B≤6 boundary router
under exactly-50 partitions and MAX_HOPS = 3. Only 12.8% remain as ranking/eviction/co-selection.
The large PARTIAL counts (150 for `CANON_DEEP`) confirm the router is admitting gold partitions
that simply cannot complete a set within six slots.

## 11. Churn and abstention (STEP 4 / message-1 STEP 4)

B is a cap, never forced churn. MetaQA churn histograms: `F6` {1:2, 2:36, 3:192, 4:570, 5:809,
6:389} mean 4.66; `ALT_DEEP_B6` {1:1, 2:22, 3:157, 4:529, 5:867, 6:422} mean 4.75;
`CANON_DEEP_B6` {1:1, 2:5, 3:59, 4:321, 5:817, 6:795} mean 5.17. Full-abstention rate (churn = 0)
is corpus-driven and identical across families: squad 0.075, 2wiki 0.029, musique 0.015,
webqsp 0.009, metaqa 0.000, hotpot 0.000. The `MAJ` consensus gate abstains almost always
(mean churn 0.64) and is uniformly worse — abstention is cheap to obtain and does not help.

## 12. Resources

Single owner per run, lock files per corpus in the path builder, distinct output file per job, no
duplicate writers. **Modal was not required and was not used**: measured peak RSS 0.08–2.43 GB,
far below the 70% trigger. `NEW_ENCODER_PASSES = 0` — the path build reuses the existing gte_qwen
embeddings, and substrates are memoised to `ctx/*.pkl` (~340 s per corpus once, then ~30 s per
round). Total artifacts 81 MB. Wall clock: paths ~9 min, oracles ~14 min, diagnosis ~7 min,
rounds 1–5 ~4 min, consolidation ~3 min.

Artifacts: `paths/`, `oracle/set_completion.json`, `diag/{hop_diagnosis,channel_redundancy,
query_local_separability}.json`, `scoreboard_round{1,2,3,4}.json`,
`scoreboard_consolidated.json`, `consolidated_ranking.json`, `selection.json`,
`selection_corrected.json`, `final_analysis.json`, `logs/`.
Code: `scratchpad/_l1kb_{core,paths,setoracle,diag,router,round3,round4,round5,final,consolidate,
select,select2}.py`.

## 13. Final rule and verdict

**Exact standing universal rule (unchanged):** `B6_S4_F6_Ms64_Mr32` —
`final = base_rank[:44] ∪ X`, `|X| = 6`, `X` chosen by symmetric F6 boundary competition
(incumbents and challengers scored identically by RRF over the canonical, S4-structural and
retrieval channels with fixed K0 = 60), `M_struct = 64`, `M_ret = 32`, S4 = MULTI_SIGNAL_RRF.

**`FREEZE_L1 = NO`** — no candidate cleared promotion:

- `CANON_DEEP_B6` is the only rule with a fold-replicated, significant gain on the primary
  diagnostic (hop3 +0.0120 vs F6, p = 0.0215; hop2 +0.0195, p = 0.0002), and it significantly
  regresses two corpora vs the safe baseline (2wiki p = 0.0013, hotpot p < 1e-4). Fails
  criterion 1.
- `ALT_DEEP_B6` clears every gate and significantly beats F6 on MetaQA overall (+0.0040,
  p = 0.0078), but its hop2/hop3 gains are individually non-significant and it does not replicate
  across the nested folds. It is a selected maximum over 90 configs, not an established gain.
  It is available as a low-risk optional upgrade — never significantly worse anywhere, macro
  +0.0068 vs F6's +0.0069 — but the evidence does not meet the bar this phase set.

**What is now established, and why the axis is closed.** The KB multi-hop deficit is not a
set-selection or path-atomicity problem: path provenance adds no reach (`A64 ≥ D ≥ C`), atomic
groups are provably redundant under a modular objective (`J1SR ≡ F6`), and atomicity contributes
+0.0015 of J1's +0.0120. It is a **channel-redundancy** problem — the canonical and retrieval
channels are one modality at two granularities, 81–100% redundant — and correcting that is worth
4×/9× on hop2/hop3. But the correction is right for MetaQA and wrong for HotpotQA, the difference
is corpus-constant rather than query-varying (well-powered null across nine candidate signals),
and 87% of the residual hop3 gap is capacity/reach that the frozen contract forbids fixing.

Further progress requires a contract change, not another router family — raising B beyond 6 or the
P50 capacity (addresses the 43.8% CAPACITY class), or raising MAX_HOPS / M_struct (addresses the
43.4% REACH class). Both are outside this phase's mandate and are the user's call.

**Not run, as instructed:** no TEST, no L2/L3, no training, no learned routing.
