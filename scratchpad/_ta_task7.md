
---

## TASK 7 — downstream diagnostic through the FROZEN L2

Each corrected P50 hard union is fed to the **unmodified** G1 backbone (C7b soft-archetype fusion → C8c
XGBRanker → C11a interaction MLP, loaded from `results/GENERALIZATION/_g1_backbone`). Nothing is trained,
retrained or tuned. **No structural reserve `R`, no special graph candidate slot, no new admission model**
— the directive's prohibition is honoured: structural nodes arrive as ordinary partition members and are
scored by the same five experts as everything else.

### How the changed candidate set is handled (and why a loss here cannot be an artefact)

The corrected Track-A *replaces* partitions, so the candidate set changes by **addition and eviction**.

| class | treatment |
|---|---|
| **kept** (in stored corpus ∩ new union) | every stored expert score reused **bit-exactly** |
| **evicted** (in stored corpus, not in new union) | dropped — exact |
| **added** (in new union, not in stored corpus) | scored deterministically, **no encoder pass**: dense `= cos(q, node)` exact; offset `= node · universal_offset_head(q, d_top1)` exact; mixture `= max_k node · head_k` exact; **splade `=` per-query `MIN − 1`** (conservative); **relation `=` ABSTAIN** (mask False → RRF contribution exactly 0) |

Only SPLADE and relation are approximate, and both **under-credit** added candidates. A downstream *gain*
would therefore be a lower bound; a downstream *loss* cannot be manufactured by the approximation making
added candidates look too good. A control on the approximation is reported below.

**Gold labelling.** Added candidates are labelled from the *true* gold set (`query_ids_all.json`), not from
the stored corpus — the stored corpus contains only golds that were already inside the BASE scope, so
labelling from it would silently mark a newly-admitted gold as a non-gold and bias every metric against the
new scope. Gold is used **only to label for evaluation**, never in scope construction. A label-consistency
assertion on the kept candidates passes 3000/3000.

**Corpus coverage and its one real limitation.** The frozen L2 corpora that exist are `2wiki_clean`,
`musique_clean` and `squad_clean`; SQuAD is excluded for the staleness defect established in TASK 5.
**MetaQA has no frozen L2 corpus at all** (`data/l2_corpus/` contains only the three text sets, and the
backbone was developed on 2wiki + MuSiQue). MetaQA is the **only** corpus where corrected Track-A improved
L1 co-scoping, so the downstream question can be answered only where the L1 effect was neutral-to-negative.
This is stated as a limitation rather than papered over; building a MetaQA L2 corpus is a new build and
outside this directive.

### Gate

```
DOWNSTREAM_M0_CANDIDATE_PARITY = EXACT
```
BASE reproduces the stored frozen baseline with **0 evicted and 0 added** over the full 3000-query universe
on both corpora.

### 2wiki_clean (n = 3000, frozen valid universe)

| cfg | L1 ALL@P50 net | scope Δ (cands) | scope Δ (golds) | nDCG@5 | R@5 | ALL@5 | ALL@50 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **BASE** (gate EXACT) | — | 0 / 0 | 0 / 0 | 0.9021 | 0.9184 | 0.7943 | 0.9217 |
| A1b M=0 | **+14 (p=.013)** | −2,134,516 / +2,128,591 | −6 / **+41** | 0.8926 | 0.9076 | 0.7743 | 0.9050 |
| A1a M=32 | −31 (p<.001) | −3,322,581 / +3,331,599 | −79 / +14 | 0.8986 | 0.9163 | 0.7917 | 0.9143 |
| A2 M=32 | −9 | −2,082,593 / +2,090,627 | −31 / +19 | 0.8989 | 0.9148 | 0.7870 | 0.9143 |
| A2dOnly M=32 | −3 | −1,217,149 / +1,221,175 | −19 / +8 | 0.9002 | 0.9163 | 0.7920 | 0.9173 |

Paired bootstrap (10 000 resamples) against BASE:

| cfg | ΔnDCG@5 | 95 % CI | ΔR@5 | 95 % CI | ΔALL@50 | 95 % CI |
|---|--:|:--|--:|:--|--:|:--|
| A1b M=0 | **−0.0094** | [−0.0119, −0.0071] **sig** | **−0.0109** | [−0.0143, −0.0077] **sig** | **−0.0167** | [−0.0217, −0.0117] **sig** |
| A1a M=32 | **−0.0035** | [−0.0060, −0.0010] **sig** | −0.0021 | [−0.0054, +0.0011] ns | **−0.0073** | [−0.0127, −0.0020] **sig** |
| A2 M=32 | **−0.0032** | [−0.0053, −0.0011] **sig** | **−0.0037** | [−0.0065, −0.0008] **sig** | **−0.0073** | [−0.0117, −0.0030] **sig** |
| A2dOnly M=32 | **−0.0019** | [−0.0038, −0.0000] **sig** | −0.0021 | [−0.0047, +0.0004] ns | **−0.0043** | [−0.0077, −0.0010] **sig** |

**Every configuration is significantly worse than BASE downstream. There are no exceptions.**

The critical row is **A1b M=0**: at L1 it is the *only* 2wiki config that improved, and it improved
significantly (ALL@P50 +14, p=0.013; +41 golds into scope against 6 lost). Through the frozen L2 it is the
**worst** row on every metric. The L1 gain does not merely fail to survive — it inverts.

### Stage decomposition (nDCG@5 units, summed over queries, split by what happened to gold coverage)

| cfg | n gained | n lost | Δ on gained | Δ on lost | Δ on **unchanged** | total | share on unchanged |
|---|--:|--:|--:|--:|--:|--:|--:|
| A1b M=0 | 5 | 55 | +0.72 | −9.81 | **−19.24** | −28.33 | **68 %** |
| A1a M=32 | 23 | 45 | +3.49 | −5.77 | **−8.12** | −10.41 | **78 %** |
| A2 M=32 | 11 | 33 | +1.28 | −5.65 | **−5.21** | −9.59 | **54 %** |
| A2dOnly M=32 | 6 | 19 | +1.20 | −2.91 | **−4.00** | −5.70 | **70 %** |

**54–78 % of the downstream loss falls on queries whose gold coverage in scope did not change at all.**
Those queries have exactly the same golds inside the scope before and after; the reranker got worse purely
because the *composition* of the ~5000-node pool changed around them. C8c and C11a consume rank- and
percentile-normalised features computed over the pool, so swapping ~2 M candidates moves every candidate's
features even when the golds are untouched. That is the stage: **L1 buys coverage in units of a handful of
queries and pays for it in units of thousands of perturbed pools.**

**Control on the SPLADE approximation.** If the unchanged-group loss were an artefact of conservatively
scoring added candidates, it would scale with the number of added candidates. It does not: A1b M=0 and
A2 M=32 add almost identical volumes (2,128,591 vs 2,090,627) yet differ nearly 4× in unchanged-group loss
(−19.24 vs −5.21) — a 3.7× gap at a 1.8 % difference in volume. A2dOnly, which adds 43 % fewer candidates
than A2, still puts the *largest share* of its (smaller) loss on the unchanged group, 70 % vs 54 %. The loss
tracks *which* partitions were swapped, not *how many* candidates the swap moved — i.e. it is
pool-composition sensitivity, not the scoring approximation.
