# L1 CANDIDATE ADMISSION CAUSAL PHASE

**VERDICT: C. CURRENT_ADMISSION_EVIDENCE_EXHAUSTED**
**PROMOTED = NONE. L1_FROZEN = NO. CROSS_CORPUS_SAFE = NO.**

Contract held throughout: no LLM, no MLP, no learning, no new encoder, no new graph traversal,
final `P = 50` exactly, `B = 6`, and every production admission variant emits exactly the frozen
`K(q)` candidates. `F6_PARITY_ON_SAFE_POOL` is exact on all six corpora
(1998/1419/2000/2000/2000/2000), so every comparison below is against a bit-identical replay of the
frozen system.

---

## STEP 0 — the B accounting, reconciled

The previous report labelled one number `B_CAPACITY` while answering a different question with
`B_LIMITED = NO`. Both were computed correctly; they are two different quantities and only one of
them was named. Separated:

| | `B_CURRENT_HEADROOM` = O2−O1 | `B_AFTER_FULL_VISITED_HEADROOM` = O4−O3 |
|---|---|---|
| metaqa hop2 | 0.0000 | 0.0450 |
| metaqa hop3 | 0.0030 | **0.1862** |
| metaqa ALL | 0.0010 | 0.0771 |
| webqsp | 0.0000 | 0.0169 |
| musique_clean | 0.0000 | **0.0000** |
| 2wiki_clean | 0.0000 | **0.0000** |
| hotpotqa_clean | 0.0000 | **0.0000** |
| squad_clean | 0.0000 | **0.0000** |

**Your reading is confirmed, with one restriction.** B is dormant under the current candidate
universe and becomes a substantial secondary bottleneck once the universe is repaired — but only on
metaqa, and marginally on webqsp. On the four text corpora `O4−O3` is exactly 0.0000: even with a
perfect universe and perfect selection, B never binds, because O3 is already 0.99–1.00 there.

The ladder makes it a curve rather than a two-point claim. At fixed `P = 50`, protect
`base_rank[:50−b]` and choose `b` perfectly:

| metaqa hop3, universe | b=6 | b=8 | b=10 | b=14 | b=20 | b=30 | b=50 |
|---|---|---|---|---|---|---|---|
| SAFE (current) | 0.3634 | 0.3634 | 0.3634 | 0.3664 | 0.3664 | 0.3664 | 0.3664 |
| FULL_VISITED | 0.6547 | 0.6832 | 0.7027 | 0.7523 | 0.7838 | 0.8168 | 0.8408 |

Flat on the current universe, monotone with no knee on a repaired one. Opening replacement capacity
before the candidates are good is worth +0.003; after, it is worth up to +0.186 on hop3.

### A bug was found and fixed while checking this

The bookkeeping check surfaced a real defect: `O1 <= O3` failed per query on metaqa, webqsp and
2wiki. Cause — `base_rank` is already a *partition* ranking while every other array is a *node*
ranking, and the identity test that distinguished them (`arr is not z["base_rank"][qi]`) is always
true, because re-indexing builds a fresh view. Canonical partition ids were being mapped through
`hard[]` as document ids, so `reach` silently contained spurious partitions and omitted offered
candidates (metaqa 251, webqsp 2476, 2wiki 2311, musique 243). Fixed with an explicit flag plus a
hard assertion that `reach` is a superset of everything offered; all six substrates rebuilt.
**The chain is now monotone per query on all six.** Corrected Phase-6 numbers are in
`../L1_BACKWARD/` — the qualitative conclusions there are unchanged and
`NOT_IN_CANDIDATE_UNIVERSE` remains the largest label on all six (44.0–69.2%), but webqsp
`UNREACHABLE` drops from 44.5% to 26.1% and its `REACH` chain term from 0.1184 to 0.0620.

---

## STEPS 1–4 — the admission target, and a degenerate safeguard

Of the required partitions the frozen SAFE P50 misses (metaqa, n=5815): **97.2% were visited,
82.6% already carry an SRC core-exit atom, and only 27.9% are admitted.** So the evidence exists and
the pool is where it is being lost — which is what motivated this phase.

STEP 4 as written turned out to be degenerate. "Retain candidates with unique existing SAFE
evidence" tests uniqueness against the protected core, and the core already owns nearly every
non-SRC atom, so retention is **0.11–0.46 candidates per query** — SRC silently takes the entire
budget and `A2_HYBRID_ADMIT` collapses onto `A1_SRC_ASSIGN`. The safeguard you specified was never
actually exercised. Two non-degenerate forms were built and run alongside it:

- **`A3_INTERLEAVE`** — equal 1:1 alternation of the frozen admission and the SRC assignment into
  one fixed-K pool. The head of the frozen pool always survives; SRC only takes slots the frozen
  channel has not claimed. Equal alternation is the same parameter-free choice the frozen equal-RRF
  fusion already makes between channels.
- **`A4_SUBSUME`** — a candidate is evictable only if some candidate the frozen admission already
  ranks *above* it carries a **superset** of its evidence. No evidence in the frozen pool is ever
  lost; SRC fills only provably redundant slots. This is the strongest parameter-free safeguard the
  contract permits.

### `MISSING_REQUIRED_ADMISSION_RECALL` (ALL), at identical K(q)

| | SAFE | A1 | A2 | A3 | A4 | UNION (diag) | K(q) |
|---|---|---|---|---|---|---|---|
| metaqa | 0.2794 | 0.3261 | **0.3262** | 0.3248 | 0.3212 | 0.4464 | 49.2 |
| webqsp | 0.0890 | 0.0944 | 0.1039 | **0.1181** | 0.1056 | 0.1519 | 34.4 |
| musique_clean | **0.3200** | 0.1733 | 0.1733 | 0.2800 | 0.2133 | 0.4133 | 18.6 |
| 2wiki_clean | 0.1624 | 0.1368 | 0.1368 | 0.1453 | 0.1538 | 0.2308 | 41.8 |
| hotpotqa_clean | **0.2885** | 0.1923 | 0.2308 | 0.1827 | 0.2019 | 0.3750 | 55.4 |
| squad_clean | **0.5600** | 0.1600 | 0.1600 | 0.2800 | 0.1600 | 0.5600 | 15.8 |

The SRC channel is a **better** admission signal than the frozen admission on exactly two corpora —
metaqa and webqsp, the two graph/KB substrates — and a **worse** one on all four text corpora. That
is the first negative result of the phase, and it sits upstream of any selector.

The `SAFE_U_SRC` union diagnostic (never used as a production selector) beats the frozen pool on
**all six**. The information SRC carries is genuinely additive everywhere. What fails is buying it
at fixed K.

---

## STEP 5 — admission oracle, perfect selection at B=6

| pool oracle (ALL) | SAFE | A2 | A3 | A4 | UNION |
|---|---|---|---|---|---|
| metaqa | 0.7212 | **0.7392** | 0.7327 | 0.7347 | 0.7643 |
| metaqa hop3 | 0.3634 | **0.3754** | 0.3649 | 0.3589 | 0.4189 |
| webqsp | 0.7886 | 0.7822 | **0.7935** | 0.7900 | 0.8048 |
| musique_clean | **0.9750** | 0.9590 | 0.9695 | 0.9695 | 0.9785 |
| 2wiki_clean | **0.9515** | 0.9455 | 0.9465 | 0.9510 | 0.9555 |
| hotpotqa_clean | **0.9645** | 0.9460 | 0.9490 | 0.9510 | 0.9690 |
| squad_clean | **0.9945** | 0.9855 | 0.9880 | 0.9855 | 0.9945 |

Same shape: the pool genuinely improves on metaqa and webqsp and genuinely degrades on the four
text corpora.

---

## STEPS 6 + 9 — the 2x2, all six corpora, exact P50 vs frozen SAFE

`acc / net / (gained/lost)`, McNemar, depth-matched by construction (every pool is exactly K(q)).

| ALL | SAFE+F6 | A2+F6 | A3+F6 | A4+F6 |
|---|---|---|---|---|
| metaqa | 0.6612 | **0.6677 +13 (15/2) SIG** | **0.6667 +11 (11/0) SIG** | **0.6647 +7 (7/0) SIG** |
| webqsp | 0.7646 | 0.7618 −4 (4/8) | 0.7653 +1 (4/3) | 0.7653 +1 (3/2) |
| musique_clean | 0.9635 | 0.9545 −18 (2/20) **SIG** | 0.9600 −7 (1/8) **SIG** | 0.9630 −1 (2/3) |
| 2wiki_clean | 0.9435 | 0.9385 −10 (1/11) **SIG** | 0.9385 −10 (0/10) **SIG** | 0.9430 −1 (1/2) |
| hotpotqa_clean | 0.9505 | 0.9400 −21 (10/31) **SIG** | 0.9420 −17 (4/21) **SIG** | 0.9445 −12 (7/19) **SIG** |
| squad_clean | 0.9875 | 0.9845 −6 (2/8) | 0.9850 −5 (1/6) | 0.9835 −8 (0/8) **SIG** |

| ALL | SAFE+G4 | A2+G4 | A3+G4 | A4+G4 |
|---|---|---|---|---|
| metaqa | 0.6652 +8 (21/13) | 0.6667 +11 (24/13) | 0.6657 +9 (23/14) | 0.6652 +8 (23/15) |
| webqsp | 0.7632 −2 | 0.7604 −6 | 0.7639 −1 | 0.7611 −5 |
| musique_clean | 0.9600 −7 **SIG** | 0.9535 −20 **SIG** | 0.9575 −12 **SIG** | 0.9600 −7 |
| 2wiki_clean | 0.9435 +0 | 0.9390 −9 **SIG** | 0.9395 −8 | 0.9420 −3 |
| hotpotqa_clean | 0.9450 −11 **SIG** | 0.9365 −28 **SIG** | 0.9335 −34 **SIG** | 0.9335 −34 **SIG** |
| squad_clean | 0.9870 −1 | 0.9845 −6 | 0.9850 −5 | 0.9835 −8 **SIG** |

MetaQA by hop:

| metaqa | SAFE+F6 | A2+F6 | A3+F6 | A4+F6 | SAFE+G4 | A2+G4 |
|---|---|---|---|---|---|---|
| hop2 | 0.7297 | 0.7372 +5 (7/2) | 0.7372 +5 (5/0) | 0.7342 +3 (3/0) | 0.7222 −5 (5/10) | 0.7237 −4 (6/10) |
| hop3 | 0.2583 | **0.2703 +8 (8/0) SIG** | **0.2673 +6 (6/0) SIG** | 0.2643 +4 (4/0) ns | **0.2778 +13 SIG** | **0.2808 +15 SIG** |

**The 2x2 does separate admission from selection, cleanly.** Under the frozen F6 selector, admission
is monotone on metaqa — 15/2, 11/0, 7/0 — and it repairs G4 hop2 damage (G4 costs 10 hop2 queries;
every A-variant with F6 costs 0–2). Under G4, every column re-imports the G4 hop2 losses regardless
of which pool feeds it. So on metaqa the gains attach to admission and the losses attach to the
selector. **That separation does not survive contact with the other five corpora**, where admission
losses dominate on their own.

---

## STEP 7 — the no-atom loss audit, and the finding that decides the phase

**Your primary question — does A2 preserve the NO_SRC_ATOM partitions G4 previously lost?
Answer: NO, on all six corpora.** `frac_no_src_preserved_by_A2_G4 = 0.0000` everywhere
(191/19/9/3/18/2 partitions lost by plain G4; 0 of the no-SRC ones recovered by the hybrid). The
coverage hole in the assignment objective is not addressed by feeding it a better pool.

The exchange ledger is more damaging still. Across **all six corpora x all three variants under the
frozen F6 selector**:

- **75 gains total — of which 1 came from a newly admitted partition.** The other 74 were already
  in the frozen SAFE pool and rose into the top-6 only because a competitor was evicted.
- **162 losses total — `loss_still_in_pool = 0` in every single cell.** Every loss is an eviction;
  not one is a ranking miss.

Gains and losses are therefore the *same* mechanism — eviction — pointing in opposite directions.
Admission did not convert. The metaqa pool oracle does improve (0.7212 → 0.7392, hop3 0.3634 →
0.3754), so required partitions genuinely were admitted; the frozen selector then picked none of
them. That is `l1-conversion-selector-blind` reproducing exactly, one stage earlier.

Loss composition confirms the eviction reading: on the text corpora most losses carry no SRC atom at
all (2wiki 9/11, hotpot 21/31, squad 6/8) but do carry dense, SPLADE and retrieval-continuation
evidence — ordinary well-supported candidates that the SRC channel cannot see and therefore
displaces.

**`A4_SUBSUME` is the decisive control.** It is provably lossless in evidence terms — nothing is
evicted unless a better-ranked candidate carries a superset of its evidence — and it behaves exactly
as designed: musique losses fall 20 → 3, 2wiki 11 → 2, webqsp turns net positive, and metaqa keeps a
significant 7/0 gain. And it *still* significantly regresses hotpot (−12) and squad (−8).

That is the mechanism, stated precisely: **evidence subsumption does not imply requirement
subsumption.** A required partition whose atom set is a subset of another candidate's is still
required. The frozen admission ordering carries information about which partitions are needed that
is not present anywhere in the atom matrix — the same conclusion
`l1-calibration-dilution-not-scoring` reached from the opposite direction.

---

## STEP 8 — query-local applicability (diagnostic only; no gate built, no threshold search)

AUC(gain vs loss) over inference-safe per-query statistics, macro |AUC − 0.5| across corpora:

| src_atoms | src_targets | src_conc | src_safe_overlap | src_dense_agree | src_splade_agree | src_targets_per_source |
|---|---|---|---|---|---|---|
| 0.1356 | 0.2408 | 0.1785 | 0.1663 | 0.1419 | 0.1981 | 0.2309 |

Nothing separates. More importantly **the sign of every effect flips across corpora** — `src_targets`
is 0.267 on metaqa and 0.875 on squad; `src_conc` is 0.182 on 2wiki and 0.813 on webqsp — so no
universal-direction rule exists in this family, let alone a universal threshold.

The one statistic that separates perfectly (**AUC = 1.0000 on five of six**) is
`frac_of_missing_required_that_carry_an_SRC_atom`. **It reads gold and is reported here only as an
oracle control.** It says the applicability signal is real and is exactly "does the partition I am
missing happen to have a core-exit atom" — unavailable at inference time. There is no inference-safe
proxy for it in the current evidence.

---

## STEP 10 — latency

Graph work is byte-identical; only admission and selection differ. ms/query, metaqa:

| frozen_admit | A1 | A2 | A3 | A4 | F6 | G4 |
|---|---|---|---|---|---|---|
| 0.0893 | 0.3459 | 0.1049 | 0.0270 | 0.2843 | 0.1456 | 0.5056 |

Against the frozen L1 total of 15.565 ms/q: **A2 2.90%, A3 2.40%, A4 4.05%** — all inside the <5%
preferred target, well inside the 10% ceiling. Latency is not a blocker for anything here.

---

## Decision returns

| return | value |
|---|---|
| `B_CURRENT_HEADROOM` | 0.0010 ALL / 0.0000 hop2 / 0.0030 hop3 (0.0000 on 5 of 6 corpora) |
| `B_AFTER_ADMISSION_HEADROOM` | 0.0771 ALL / 0.0450 hop2 / **0.1862 hop3**; webqsp 0.0169; **0.0000 on all four text corpora** |
| `SAFE_POOL_MISSING_RECALL` | 0.2794 (hop3 0.2792) |
| `SRC_POOL_MISSING_RECALL` | 0.3261 (hop3 0.3140) |
| `HYBRID_POOL_MISSING_RECALL` | 0.3262 (hop3 0.3144) — but **lower than SAFE on 4 of 6 corpora** |
| `SAFE_POOL_ORACLE_HOP3` | 0.3634 |
| `HYBRID_POOL_ORACLE_HOP3` | 0.3754 |
| `F6_HYBRID_HOP3` | 0.2703 (+8, p=0.0078, SIG) |
| `G4_HYBRID_HOP3` | 0.2808 (+15, p=0.0015, SIG) |
| `CROSS_CORPUS_SAFE` | **NO** — 15 significant regressions across variants; every variant loses on at least 2 corpora |
| `PROMOTED` | **NONE** |
| `L1_FROZEN` | **NO** |

**VERDICT — C. CURRENT_ADMISSION_EVIDENCE_EXHAUSTED.**

1. The cross-corpus gate fails outright. The MetaQA gain is real, significant and (with the frozen
   selector) loss-free — and it is the only one.
2. The SRC channel is a *worse* admission signal than the frozen admission on four of six corpora.
   The failure is upstream of the selector, not a tuning issue.
3. 1 of 75 realized gains came from a newly admitted partition. Admission as a mechanism converted
   nothing; every other gain and all 162 losses are eviction effects.
4. Even the strongest parameter-free safeguard the contract allows (`A4_SUBSUME`, provably lossless
   in evidence terms) still significantly regresses two corpora.
5. No inference-safe query-local statistic distinguishes the gain regime from the loss regime; the
   only separator requires gold.

## What remains true, and what it implies for the next ruling

- **The evidence is additive.** `SAFE_U_SRC` beats the frozen pool on missing-required recall on all
  six corpora and on the B=6 pool oracle on all six. SRC is not noise.
- **The binding constraint is `K(q)`.** A fixed pool forces an exchange, and nothing measurable
  predicts which side of that exchange is required. Every negative in this phase traces to that
  exchange, not to the evidence itself.
- **B stays conditional, and now narrowly so.** It is dormant now, and it only ever becomes a real
  bottleneck on metaqa (+0.186 hop3) and marginally webqsp. On musique, 2wiki, hotpot and squad,
  `O4 − O3 = 0.0000` exactly — B is not a lever there under any repair.

## Artifacts

- `RETURNS.json` — all decision returns plus the full per-variant matrices
- `TABLES.md` — T0–T9
- `diag/admit_<ds>.json`, `diag/audit_<ds>.json`, `diag/stat_<ds>.npz`
- code: `scratchpad/_l1ca_{step0,admit,audit,report}.py`, substrate `scratchpad/_l1bc_core.py`
- corrected Phase-6 ledger: `../L1_BACKWARD/diag/ledger_*.json`,
  `../L1_BACKWARD/diag/step0_ALL.json`
