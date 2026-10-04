# L1 FINAL SCORING PHASE — CALIBRATE THE SWAP, DO NOT ADD FEATURES

**Scope.** DEV only, six corpora (MetaQA, WebQSP, 2Wiki, MuSiQue, HotpotQA, SQuAD).
No TEST, no training, no L2, no L3, no new features, no PPR architecture search, no new encoder pass.
`SAFE_REFERENCE = B6_S4_F6_Ms64_Mr32` held throughout.
Tables: [TABLES.md](TABLES.md) (T1–T14). Raw: `diag/*.json`, `diag/ind_*.npz`.
Scripts: `scratchpad/_l1cal_{core,s0,s1,main,val,tables}.py`.

**Headline.** Seven boundary-scoring rules were run. One of them — `R1_RRF_LIFT`, the per-channel
finite-list floor subtraction — produces the largest MetaQA multi-hop gain this program has found
inside the frozen contract (**hop2 +0.0180, hop3 +0.0135, both p<0.01**, hop1 untouched, 21 MetaQA
queries newly covered and **zero** lost). It is nonetheless **not promotable**: its macro advantage
over SAFE_F6 is +0.00025 on the full DEV but **−0.00014 on the pre-registered discovery half**, and
the leave-one-dataset-out winner flips across three different rules. It fails promotion criteria (1)
and (4). **`L1_FROZEN = NO`.**

---

## 1. STEP 0 — the capacity statement, with its candidate universe attached

The previous phase's two capacity numbers were about different universes and were quoted without
one. Restated as three explicitly named classes at B=6 (T1, T2):

| class | definition | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD |
|---|---|---|---|---|---|---|---|
| **A `CAP_B6_CURRENT_POOL`** | gold reachable in boundary ∪ challengers, but needs >6 swaps | **2** | 0 | 0 | 0 | 0 | 0 |
| **B `CAP_B6_PPR_EXPANDED_POOL`** | same, with the PPR-reached set added to the universe | **3** | 0 | 0 | 0 | 0 | 0 |
| **C `CAP_P50`** | more than 50 gold partitions — impossible under any rule | **26** | 7 | 0 | 0 | 0 | 0 |

Classification is ordered `CAP_P50 → POOL_UNREACHABLE → CAP_B → SOLVABLE`, over all queries; a query
is counted in the first class it falls into. That ordering is why these numbers differ from the
previous phase's `CAP_B6 = 21/677`, which was computed over failures only and did not split out
`CAP_P50` first.

**A is 2 queries. B is 3 queries.** Boundary width is not the limiter under either universe, and
adding PPR reach does not make it one — the extra reach converts queries out of
`POOL_UNREACHABLE`, not into `CAP_B`: MetaQA 529→440, WebQSP 293→265, 2Wiki 97→81, MuSiQue 50→31,
HotpotQA 71→68, SQuAD 11→8. The reach is real and it is worth 89 MetaQA queries of *reachability*.

**The diagnostic (T3).** PPR was then admitted as a fourth co-equal voting channel (M_PPR = 64, from
the cached global partition PPR) under the protected core at B ∈ {6, 8, 12}, on all six corpora.
Result: **negative or zero at every B on every corpus** — MetaQA −0.0095/−0.0130/−0.0145,
WebQSP −0.0078/−0.0092/−0.0028, MuSiQue −0.0080/−0.0100/−0.0080, HotpotQA −0.0055/−0.0075/−0.0105,
SQuAD −0.0015/−0.0025/−0.0030, 2Wiki −0.0005/0.0000/−0.0005. On four corpora it gets *worse* as B
grows. More boundary capacity lets PPR do more damage, not less.

> **Answer to the question STEP 0 poses:** PPR did **not** fail because its newly reached partitions
> could not fit through B6. They fit — capacity absorbs them (A=2 → B=3, one query). It failed
> because **its ranking is poor**: given a vote, it displaces better-evidenced partitions.
>
> **PPR is closed for L1.**

---

## 2. STEP 1 — the exact current failure, measured

Under the frozen R0 selector, per corpus (T4, T5):

| corpus | swaps | GOOD admissions | BAD evictions | GOOD/BAD | queries with the decisive cross-list pattern | of those, costing gold |
|---|---|---|---|---|---|---|
| MetaQA | 9309 | 257 | 183 | 1.40 | **1** | 0 |
| WebQSP | 7049 | 91 | 90 | 1.01 | **544** | 28 |
| 2Wiki | 8518 | 20 | 6 | 3.33 | 43 | 0 |
| MuSiQue | 8624 | 26 | 11 | 2.36 | **0** | 0 |
| HotpotQA | 10197 | 39 | 5 | 7.80 | 80 | 0 |
| SQuAD | 7948 | 17 | 3 | 5.67 | **0** | 0 |

**The eviction half of the hypothesis is confirmed at scale.** Evicted incumbents are overwhelmingly
canonical-only: `C--` accounts for 7906/9309 MetaQA and 6954/7049 WebQSP evictions, carrying gold
145 and 90 times respectively. A boundary incumbent holding canonical evidence alone loses its slot
constantly.

**The admission half is refuted.** The named failure requires the *winner* to hold only one
non-canonical channel. It almost never does, because the challenger pool is drawn from TOP200 — so
nearly every challenger carries a canonical rank too. Of MetaQA's 9309 admissions, 9200 have
canonical evidence; only 109 do not. The pure pattern exists at scale on exactly one corpus, WebQSP
(1467 admitted `-S-` challengers), and there it is nearly sterile: those 1467 admissions carry gold
**4** times.

So the decisive cross-list comparison the phase was convened to fix occurs on **1 MetaQA query, 0
MuSiQue, 0 SQuAD**, and costs a gold partition on 28 WebQSP queries and nowhere else. And in every
corpus the swap is **net-positive on gold** (GOOD/BAD from 1.01 to 7.80). Retaining incumbents more
aggressively should therefore be expected to cost more than it saves — which is exactly what the
counterfactual (T6) shows: every rule other than R0 has a **negative** net-gold-slot balance on 6/6
corpora, the single exception being R1 on WebQSP (+1).

The RRF-scale distortion is nevertheless real and quantifiable. Each channel's floor is set by its
own list length:

| channel | L | floor `1/(K0+L)` | span `1/K0 − floor` |
|---|---|---|---|
| canonical partition | 200 | 0.003846 | 0.012821 |
| S4 structural | ~48 | 0.009259 | 0.007407 |
| node retrieval | ~25 | 0.011765 | 0.004902 |

Merely *appearing* at the bottom of the 25-long node list is worth 0.011765, three times what
appearing at the bottom of the 200-long canonical list is worth. Worked case — incumbent at
canonical rank 47 against a challenger holding only a node-retrieval rank of 20:

```
RAW  : incumbent 0.009346   challenger 0.012500   -> challenger wins by 0.003154
LIFT : incumbent 0.005500   challenger 0.000735   -> incumbent  wins by 0.004764
```

That is the arithmetic R1 corrects.

---

## 3. STEPS 2–7 — the seven rules

All seven consume only the three cached rankings the frozen router already builds. No learned
constant, no fitted weight, no threshold other than zero, K0 = 60 throughout, absence contributes
exactly 0.0, output is exactly 50 partitions.

| rule | expression (per candidate `p`) |
|---|---|
| **R0** `RAW_RRF` | `raw_c + raw_s + raw_r`, `raw(r)=1/(K0+r)` — **control, untouched** |
| **R1** `RRF_LIFT` | `lift_c + lift_s + lift_r`, `lift_x(r)=1/(K0+r) − 1/(K0+L_x)` |
| **R2** `RRF_UNIT` | `unit_c + unit_s + unit_r`, `unit_x = lift_x / (1/K0 − 1/(K0+L_x))` |
| **R3** `SIGNED_RET_RESIDUAL` | `unit_c + unit_s + (unit_r − unit_c)` |
| **R4** `POSITIVE_RET_RESIDUAL` | `unit_c + unit_s + max(0, unit_r − unit_c)` |
| **R5** `PAIRWISE_RESIDUAL` | replace iff `Δunit_c + Δunit_s + Δresid⁺ > 0`, else keep incumbent |
| **R6** `PARETO_SWAP` | replace only on a dominating evidence vector (non-additive control) |

**R0 parity is exact**: R0 reproduces `KB.f6_select` bit-for-bit on every query of every corpus
(**0 / 11417 mismatches**), so
the control is the frozen selector, not a re-implementation of it. Old artifacts were not modified.

Two structural facts found while implementing, reported rather than silently patched:

- **R3 is algebraically degenerate.** `unit_c + unit_s + (unit_r − unit_c) ≡ unit_s + unit_r`. The
  signed residual cancels the canonical channel identically, so R3 as specified is not "canonical
  plus a residual" — it is **canonical retrieval deleted**. It was run as written. R4's `max(0,·)`
  is non-linear, so there the canonical term survives.
- **R6 can never swap, by construction.** Boundary incumbents are `base_rank[44:50]`; every
  challenger is canonically worse or absent, so `Δcanonical ≥ 0` is unsatisfiable. R6 executes 0
  swaps on all six corpora and is therefore identically BASE (T7: `ΔBASE = +0.0000` everywhere). A
  dominance rule over a pool the incumbents already dominate on one axis is vacuous. It remains
  reported as the control it was asked to be.

---

## 4. STEP 8 — all six corpora

Δ vs SAFE_F6, `*` p<0.05, `**` p<0.01, exact McNemar (T7 has the full table with churn, swaps,
admitted/evicted gold-bearing partitions, and GOOD/BAD per cell):

| rule | MetaQA | WebQSP | 2Wiki | MuSiQue | HotpotQA | SQuAD | macro | worst |
|---|---|---|---|---|---|---|---|---|
| R0 (= SAFE_F6) | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.00000 | +0.0000 |
| **R1 RRF_LIFT** | **+0.0105\*\*** | +0.0000 | −0.0015 | −0.0030 | −0.0030 | −0.0015 | **+0.00025** | −0.0030 |
| R2 RRF_UNIT | +0.0085\*\* | −0.0042 | −0.0020 | −0.0015 | −0.0030 | +0.0000 | −0.00037 | −0.0042 |
| R3 SIGNED_RET_RESIDUAL | +0.0080\*\* | −0.0092\*\* | −0.0050\* | −0.0030 | −0.0110\*\* | −0.0005 | −0.00345 | −0.0110 |
| R4 POSITIVE_RET_RESIDUAL | +0.0110\*\* | −0.0042 | −0.0075\*\* | −0.0070\*\* | −0.0125\*\* | −0.0035\* | −0.00395 | −0.0125 |
| R5 PAIRWISE_RESIDUAL | +0.0095\*\* | −0.0042 | −0.0090\*\* | −0.0080\*\* | −0.0115\*\* | −0.0040\* | −0.00453 | −0.0115 |
| R6 PARETO_SWAP (≡ BASE) | −0.0025 | −0.0028 | −0.0060\* | −0.0070\* | −0.0160\*\* | −0.0070\*\* | −0.00688 | −0.0160 |

Reading down the table, calibration helps and residualization hurts, monotonically. R1 (subtract
each list's own floor) is the only rule with a positive macro and no significant regression
anywhere. R2 (rescale each channel to span [0,1]) re-inflates the short lists with a different
constant and gives back most of the gain. R3/R4/R5 over-correct: they discount the node channel so
hard that they start evicting text golds, and R5's strict pairwise gate is the worst additive
variant. R6 is BASE.

**R1's coverage change is strictly one-directional on MetaQA**: 21 queries newly covered, **0**
newly uncovered. Elsewhere it is a near-wash (WebQSP 8/8, 2Wiki 3/6, MuSiQue 1/7, HotpotQA 6/12,
SQuAD 0/3).

**R1 self-selects its aggression without seeing dataset identity.** Swap counts vs R0: MetaQA
**+1.4%**, HotpotQA −7.0%, 2Wiki −5.4%, MuSiQue −18.7%, WebQSP **−36.4%**, SQuAD **−39.5%**. The
same fixed expression swaps slightly more where swapping pays and a third less where it does not —
because the floors are computed from each query's own cached list lengths.

---

## 5. STEP 9 — MetaQA per hop (mandatory)

666 queries per hop. Δ vs SAFE_F6 (T9):

| method | hop1 | Δ | hop2 | Δ | hop3 | Δ |
|---|---|---|---|---|---|---|
| BASE | 0.9955 | — | 0.7237 | — | 0.2568 | — |
| R0 (SAFE_F6) | 0.9955 | +0.0000 | 0.7297 | +0.0000 | 0.2583 | +0.0000 |
| **R1 RRF_LIFT** | 0.9955 | +0.0000 | **0.7477** | **+0.0180\*\*** | **0.2718** | **+0.0135\*\*** |
| R2 RRF_UNIT | 0.9955 | +0.0000 | 0.7462 | +0.0165\*\* | 0.2673 | +0.0090 |
| R3 SIGNED_RET_RESIDUAL | 0.9955 | +0.0000 | 0.7372 | +0.0075 | 0.2748 | +0.0165\*\* |
| R4 POSITIVE_RET_RESIDUAL | 0.9955 | +0.0000 | 0.7477 | +0.0180\*\* | 0.2733 | +0.0150\*\* |
| R5 PAIRWISE_RESIDUAL | 0.9955 | +0.0000 | 0.7402 | +0.0105 | **0.2763** | **+0.0180\*\*** |
| R6 PARETO_SWAP | 0.9955 | +0.0000 | 0.7237 | −0.0060 | 0.2568 | −0.0015 |
| ORACLE_B6 | 0.9970 | — | 0.8033 | — | 0.3634 | — |

The stated primary goal — improve hop3, then hop2, without sacrificing the text datasets — is met by
exactly one rule. R5 buys more hop3 (+0.0180) and R3 more hop3 than R1, but both sacrifice the text
corpora significantly (R5 on four of five, R3 on three). **R1 is the only rule that moves both hop2 and
hop3 significantly while regressing no text corpus significantly.** It closes 24.5% of the hop2
oracle gap and 12.8% of the hop3 gap. Hop labels were not used at inference.

---

## 6. STEP 10 — nested validation and leave-one-dataset-out

The selection criterion was fixed before any held-out number was read: *argmax macro mean Δ vs
SAFE_F6, subject to no corpus regressing significantly (McNemar p<0.05); ties broken by
worst-corpus; fallback = SAFE_F6.* Folds are sha1(`ds:qi`) parity — query-disjoint and reproducible.
Selection optimises one pooled number and never sees dataset identity.

**Nested (T10, T11).** On the discovery half the criterion selects **R0 — SAFE_F6 itself**:

| rule | discovery macro | discovery worst | discovery sig-regressions |
|---|---|---|---|
| R0 | +0.00000 | +0.00000 | none |
| R1 | **−0.00014** | −0.00601 | none |
| R2 | −0.00037 | −0.00406 | none |
| R3 | −0.00478 | −0.01201 | 2Wiki, HotpotQA |
| R4 | −0.00509 | −0.01702 | 2Wiki, HotpotQA |
| R5 | −0.00701 | −0.01624 | 2Wiki, MuSiQue, HotpotQA |
| R6 | −0.00840 | −0.01802 | 2Wiki, MuSiQue, HotpotQA |

R1's macro is **negative on discovery (−0.00014) and positive on validation (+0.00065)**; the full-DEV
figure of +0.00025 sits between them. The macro difference between R1 and R0 is **fold noise**.

What is *not* noise is the split beneath it (T11). R1's MetaQA gain reproduces on both halves —
**+0.0118 (p=0.0005) discovery, +0.0092 (p=0.0039) validation** — while every text-corpus loss is
sub-noise and **sign-unstable across halves**: 2Wiki −0.0051→+0.0020, HotpotQA −0.0060→+0.0000,
WebQSP +0.0015→−0.0013, none significant on either half. R1 is a reproducible single-corpus effect
sitting inside a macro that cannot resolve it.

**LODO (T12).** Selecting on five corpora and reporting on the sixth:

| held-out | selected on the other 5 | held-out Δ | p | sig. regression |
|---|---|---|---|---|
| MetaQA | R0_RAW_RRF | +0.0000 | 1.0000 | no |
| WebQSP | R2_RRF_UNIT | −0.0042 | 0.0703 | no |
| 2Wiki | R1_RRF_LIFT | −0.0015 | 0.5078 | no |
| MuSiQue | R1_RRF_LIFT | −0.0030 | 0.0703 | no |
| HotpotQA | R1_RRF_LIFT | −0.0030 | 0.2379 | no |
| SQuAD | R1_RRF_LIFT | −0.0015 | 0.2500 | no |

`SELECTION_STABLE = False` — three different rules win across six folds. Held-out macro **−0.0022**.
No held-out regression is significant, but the *choice itself* moves with which corpora are in the
training set, and it moves in the obvious way: drop MetaQA and the criterion stops choosing R1.

**Against the four promotion conditions:**

| # | condition | R1 |
|---|---|---|
| 1 | improves the universal Pareto frontier on discovery | **NO** (−0.00014) |
| 2 | survives validation | n/a — never selected on discovery |
| 3 | no significant regression vs SAFE_F6 | **YES** (none, on either half or full) |
| 4 | does not depend on dataset identity | **NO** (LODO selection unstable) |

`PROMOTABLE = False`. `WINNER = R0_RAW_RRF`.

---

## 7. STEP 11 — cost

| quantity | value |
|---|---|
| **`ONLINE_GRAPH_EDGES_TOUCHED`** | **0** (all six corpora, all seven rules) |
| new encoder passes | 0 |
| new PPR / traversal at query time | none |
| additional cached bytes per query | **0** — the floors are `1/(K0+len(list))` on lists already in memory |
| additional operations per query | one subtraction per rank read: 181–272 ranks/query, 545–816 extra float ops |
| candidates scored per query | 15.8 (SQuAD) – 55.4 (HotpotQA) |
| routing latency, R0 | 36.7 µs/query |
| routing latency, R1 | **36.7 µs/query (+0.0)** |

R1 is free. R5 is the most expensive rule at 40.6 µs (+3.9). Everything remains query-time rank
arithmetic over cached lists.

---

## 8. STEP 12 — the ruling, and the honest reading

STEP 12 asks that a calibrated rule matching or improving F6 be preferred even for a modest gain.
STEP 10 is the gate that decides what "matching or improving" means, and it does not admit R1: the
macro edge is noise on the protected half and the LODO winner flips. **The standing policy stays
`B6_S4_F6_Ms64_Mr32` (= R0).** Nothing was promoted; no artifact of a previous phase was modified.

Applying STEP 12's fallback clause — the node retrieval channel is documented as follows:

> **The node-level `ret_rrf` channel is a fine-grained retrieval confirmation view, not an
> independent modality.** Its list is ~25 long against canonical's 200, so under raw RRF its mere
> presence is worth 3× canonical's, and it wins boundary contests on floor value rather than
> evidence. Discounting that inflation (R1) is safe and helps KB multi-hop. *Residualizing* the
> channel against the partition view (R3/R4/R5) over-corrects and costs text coverage on three to
> four corpora. The channel should be kept, and kept cheap; it should not be re-weighted, deleted,
> or promoted to a third independent vote.

**What this phase actually establishes, stated plainly:**

1. The failure it was convened to fix is **rare in its pure form** — 1 query on MetaQA, 0 on
   MuSiQue and SQuAD, and gold-costly on 28 WebQSP queries. The eviction side of the story is real
   at scale; the admission side is not, because challengers almost always carry canonical evidence
   too.
2. The swap is **net-positive on gold in every corpus** (GOOD/BAD 1.01–7.80), so rules that retain
   incumbents more aggressively lose more than they save — confirmed by every counterfactual in T6.
3. **R1 is a genuine, reproducible KB multi-hop win that the universal criterion cannot see.**
   +0.0180 hop2 and +0.0135 hop3 at zero cost and zero risk to text is the largest in-contract
   MetaQA movement found so far, and it replicates on both folds — but five text corpora each
   contribute a −0.0015 to −0.0030 of noise to the macro, and six noise terms drown one signal.
   This is the same corpus-level regime split this program has hit repeatedly, now at the scoring
   layer rather than the routing layer.
4. **PPR is closed for L1** on a capacity-controlled diagnostic, not on an assumption.
5. The in-contract selector headroom identified last phase (macro +0.0208) remains almost entirely
   unclaimed. R1 takes 1.2% of it.

**A decision for the user, not for me.** R1 clears three of the four promotion conditions and fails
the two that are about the *macro*, not about harm — it regresses nothing significantly, anywhere,
on either fold. If the universal criterion is meant to protect against harm, R1 passes. If it is
meant to require pooled improvement, R1 fails. I did not promote it, because the contract as written
says pooled improvement and forbids dataset identity. Resolving this needs either a corpus of the
missing kind (a second KB-style multi-hop corpus, which would move the macro) or an explicit ruling
that a harm-free rule with a single-corpus benefit is acceptable. Both are contract changes.

```
L1_FROZEN = NO
STANDING POLICY = B6_S4_F6_Ms64_Mr32   (unchanged)
PROMOTED = none
PPR_FOR_L1 = CLOSED
ONLINE_GRAPH_EDGES_TOUCHED = 0
```
