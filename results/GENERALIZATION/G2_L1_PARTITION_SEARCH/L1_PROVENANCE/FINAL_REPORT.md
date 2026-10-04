# L1 CORE-EXIT PROVENANCE AUDIT — FINAL REPORT

**Question.** The transformation phase closed with the finding that the one significant MetaQA gain
was produced by the **core-exit restriction** (which partitions get evidence at all), not by the
transformation geometry that ranks them — `V4_SRC_EXIT`, `V6_EXIT_SRC_SIM` and the `V0` identity
control all landed within 2–3 queries at exact P50 despite producing the identical candidate pool in
**0 of 182** MetaQA queries. This audit asks the surviving question:

> Why does SRC/core-exit evidence improve MetaQA but regress the text corpora? Is the answer
> **existing edge provenance / type**?

No new graph work, no transformation experiment, no learning, no threshold search.

**Frozen contract.** `MASTER_TOPOLOGY = C`, `P_MAIN = 50`, `B = 6`, `M_struct = 64`, `M_ret = 32`,
`K0 = 60`, `MAX_HOPS = 3`, `BEAM = 64`. `final = prot(P−B) ∪ X`, `|X| = B = 6`, exactly 50
partitions. Replay parity was **exact on every corpus** (`1998/1998`, `1419/1419`, `2000/2000`,
`2000/2000`, `2000/2000`, `2000/2000`).

**Artifacts.**

| file | contents |
|---|---|
| `TABLES.md` | every table referenced below |
| `RETURNS.json` | the named decision fields |
| `diag/step1_families.json` | corpus-level family decomposition of the traversal adjacency |
| `diag/pv_<ds>.json` | STEPS 1–4 per corpus |
| `diag/pvswap_<ds>.json` | STEPS 5–7 exact-P50 per corpus |
| `scratchpad/_l1pv_fam.py` | family labeller (STRUCT / NER / KNN, undirected canonical keys) |
| `scratchpad/_l1pv_audit.py` | STEPS 1–4 |
| `scratchpad/_l1pv_swap.py` | STEPS 5–7, fixed-K replacement at exact P50 |
| `scratchpad/_l1pv_{report,returns}.py` | assembly only |

**Correctness anchor.** `SRC_ALL` in the STEP 5–7 harness is definitionally `V6_EXIT_SRC_SIM` from
the prior phase. It reproduces that phase's exact-P50 numbers **cell for cell** on `(acc, net)` —
`metaqa 8/8`, `webqsp 2/2`, `2wiki 2/2`, `musique 2/2`, `hotpot 2/2` (SQuAD had no prior V6 run, so
`0/0`) — so the new harness is measuring the same thing on the same substrate.

---

## VERDICTS

| field | value | why |
|---|---|---|
| `BEST_SRC_EDGE_FAMILY` | **`STRUCT_AND_NER`** (score `SRC_NER`) | the only family sign-consistent across corpora, and the only key that is above chance where the generic control fails |
| `EDGE_FAMILY_SIGN_CONSISTENT` | **YES** | `SRC_NER` 6/6 eligible rows AUC > 0.5 (min 0.5203); `ALL_NER` 6/6 (min 0.5288) |
| generic control `SRC_ALL` sign-consistent | **NO** | 5/6 — Hotpot **0.4644**, below chance |
| `SRC_STRUCT_ONLY` / `ALL_STRUCT_ONLY` sign-consistent | **NO** | 4/6 and 2/6; minima 0.3895 / 0.3890 |
| `NOISIEST_FAMILY` | **`STRUCT_ONLY`** | worst gain/loss ratio in every corpus with NER support (MetaQA 0.939 vs 1.204, WebQSP 1.026 vs 1.567, 2Wiki 0.310 vs 1.009, Hotpot 0.112 vs 0.428) |
| `STEP4_UNIVERSAL_MONOTONE_TREND` | **NO**, on all four axes | direction flips by corpus *and* by family; `p_required` is 0.0000–0.0013 on every text corpus, so the bins carry no usable signal there |
| `METAQA_HOP2` | **`0.7297 → 0.7297`, net +0** (F6) | the family restriction is a no-op on the promoted selector; G4 is −6 ns. The control gets +6 |
| `METAQA_HOP3` | `0.2583 → 0.2598`, net **+1** ns (F6); **+11 SIG** (G4) | the one significant gain, and it is smaller than the control's +16 |
| `CROSS_CORPUS_SAFE` | **NO** | MuSiQue `G4 −8*` and Hotpot `G4 −16*` survive. On `F6` alone it *is* safe — 0 significant regressions on all six, down from the control's 6 |
| `PROMOTED` | **NONE** | the bar was material MetaQA hop2/hop3 improvement AND no significant regression anywhere; both legs fail |
| `L1_FROZEN` | **NO** | the directive mandates freezing only under classification C |
| **CLASSIFICATION** | **B. PROVENANCE_CHANNEL_PARTIAL** | the channel is recovered as a *presence* signal (sign-consistent, removes all significant `F6` harm) but not as an *evidence channel* (cannot rank; delivers no MetaQA improvement) |

---

## STEP 1 — EDGE FAMILY LEDGER

### The families that actually exist in the frozen traversal

The beam expands `master_nodes_{ds}.json` `neighbors` and nothing else
(`_ta_prepartition.load_topology` → `_l1bm_run.topology` → `_l1tp_core.ctx` → `SS.expand_feat`).
Families are taken from disk in that same node-index space, with the existing `ac_edge_recon`
convention: `A = data/ukb_storage/{ds}/gte_qwen/graph.pt` = `STRUCT ∪ KNN`, `KNN = A \ STRUCT`,
`NER = ner_edges_w_df25.pkl`. No relation label was invented.

| corpus | nodes | traversal edges | also NER | frac NER | also KNN | STRUCT provenance |
|---|--:|--:|--:|--:|--:|---|
| MetaQA | 40,151 | 109,480 | 59,403 | **54.26 %** | **0** | `structural_native`, 9 relations |
| WebQSP | 781,485 | 1,638,066 | 240,485 | **14.68 %** | **0** | `structural_derived_from_freebase`, 6,094 relations |
| 2Wiki | 65,865 | 126,070 | 28,890 | **22.92 %** | **0** | `structural_native` / `hyperlink` |
| MuSiQue | 13,672 | 51,543 | 6,156 | **11.94 %** | **0** | `structural_derived` / `title_mention` |
| Hotpot | 507,494 | 3,426,945 | 283,658 | **8.28 %** | **0** | `structural_native` / `hyperlink` |
| SQuAD | 19,029 | 694,580 | 19,027 | **2.74 %** | **0** | `structural_derived` / `title_mention` |

**Result 1 — the semantic/Qwen-kNN family is structurally absent from L1.** `traversal ∩ KNN = 0`
on all six corpora. The frozen beam has never traversed a kNN edge, so kNN cannot be the source of
the cross-corpus regressions and cannot be the lever either. **The only live provenance axis is
`STRUCT_ONLY` vs `STRUCT_AND_NER`.**

**Result 2 — NER corroboration is a 20× corpus gradient that tracks the phenomenon.** MetaQA, the
one corpus where core-exit evidence helps, is the one corpus where a *majority* of traversal edges
are NER-corroborated (54.26 %). SQuAD, where the operation is worst, sits at 2.74 %. The ordering
MetaQA ≫ 2Wiki > WebQSP > MuSiQue > Hotpot > SQuAD is the same ordering as the corpus-level tolerance
for SRC evidence observed in the prior phase.

*Data note.* On Hotpot 34,425 traversal edges (1.00 %) are absent from `gte_qwen/graph.pt`, so that
corpus's `A` is a slightly older build than its `master_nodes`. This does not affect the audit: NER
membership is tested independently of `A`, and the `KNN` label (`A \ STRUCT`) has zero traversal hits
anyway. Recorded rather than silently dropped.

### Per-query core-exit transition counts, by family

| corpus | edges/q | **core-exit/q** | core-exit NER/q | core-exit STRUCT_ONLY/q | frac NER |
|---|--:|--:|--:|--:|--:|
| MetaQA | 1,411 | 725 | 233 | 492 | **0.3710** |
| WebQSP | 1,494 | 737 | 109 | 628 | 0.1737 |
| 2Wiki | 537 | 304 | 85 | 220 | 0.3557 |
| MuSiQue | 3,167 | 2,766 | 207 | 2,559 | 0.0935 |
| Hotpot | 1,404 | 710 | 90 | 620 | 0.1456 |
| SQuAD | 14,088 | 11,179 | 326 | 10,852 | 0.0308 |

Two things to note. First, the beam is **not** a uniform sample of the corpus: MetaQA's per-query
core-exit NER share (0.3710) is below its corpus share (0.5426) while 2Wiki's (0.3557) is well above
its corpus share (0.2292), so the traversal preferentially reaches NER-corroborated edges on the
hyperlink corpora. Second, SQuAD emits **11,179 core-exit transitions per query** — 15× MetaQA — and
only 3 % of them carry NER; that corpus drowns any provenance signal in raw structural volume.

By hop, the NER share is highest at hop 1 everywhere (MetaQA 0.6709, 2Wiki 0.5761, Hotpot 0.2595)
and drops sharply at hop 2, which is exactly where the beam does most of its work.

---

## STEP 2 — GOOD/BAD SWAP BY EDGE FAMILY

GOOD/BAD are the prior phase's definitions verbatim (`_l1kt_partb.run`): at identical `K(q)` the SRC
pool evicts SAFE candidates and admits new ones; a labelled pair has exactly one required-and-missing
side. The pools are family-independent, so **every family is scored on an identical pair
population** — only the ordering score changes. Full tables (including mean GOOD/BAD margins and
sign accuracy) are in `TABLES.md`.

### AUC by row and family — the universality question

| row | `SRC_ALL` (control) | **`SRC_NER`** | `SRC_STRUCT_ONLY` | `ALL_NER` | `ALL_STRUCT_ONLY` |
|---|--:|--:|--:|--:|--:|
| MetaQA hop2 | **0.8429** | 0.7140 | 0.6570 | 0.6920 | 0.6487 |
| MetaQA hop3 | **0.6081** | 0.5852 | 0.5674 | 0.5624 | 0.5833 |
| WebQSP | **0.6693** | 0.5422 | 0.6251 | 0.6104 | 0.4920 |
| 2Wiki | 0.5066 | 0.6268 | **0.3895** | 0.6494 | **0.3890** |
| MuSiQue | **0.5781** | 0.5203 | 0.5283 | 0.5288 | **0.4381** |
| Hotpot | **0.4644** | **0.6444** | **0.4088** | 0.5688 | **0.4119** |
| SQuAD | — | — | — | — | — |
| **sign-consistent?** | **NO** (5/6) | **YES** (6/6) | NO (4/6) | **YES** (6/6) | NO (2/6) |
| worst case | 0.4644 | **0.5203** | 0.3895 | **0.5288** | 0.3890 |
| mean | 0.6116 | 0.6055 | 0.5294 | 0.6020 | 0.4938 |

SQuAD is ineligible: it has **0 GOOD swaps and 300 BAD swaps** — every labelled forced swap there
loses a required partition, exactly as in the prior phase, so no AUC is defined.

### IS ANY EDGE FAMILY SIGN-CONSISTENT ACROSS CORPORA? — **YES**

**`SRC_NER` and `ALL_NER` are above chance on all six eligible rows; the generic core-exit control
is not, and neither STRUCT-only key is.** This is the first sign-consistent statistic found anywhere
in the L1 search program — the prior phase's answer was `0 of 33` keys.

Two honest qualifications, both of which matter for what follows:

1. **NER is not better on average, it is better in the worst case.** Mean AUC is a wash
   (0.6055 vs the control's 0.6116) and NER is *worse* than the control on four of six rows. What it
   does is raise the minimum from 0.4644 (below chance) to 0.5203. Universality here means uniformity
   of direction, not sharpness.
2. **The two keys are sign-consistent for opposite reasons.** See the decomposition below.

### The mechanism: PRESENCE vs RANKING

Splitting each AUC into the pairs where **both** candidates have evidence under that score (a pure
ranking comparison) versus the full population (which also includes pairs decided by the floor,
i.e. by *presence* of evidence):

| corpus | `SRC_ALL` raw → both-evidence | `SRC_NER` raw → both-evidence (support) |
|---|--:|--:|
| MetaQA | 0.6446 → **0.6850** | 0.6075 → 0.6485 (n=3,929) |
| WebQSP | 0.6693 → **0.8402** | 0.5422 → 0.6764 (n=29) |
| 2Wiki | 0.5066 → **0.5417** | 0.6268 → 0.0000 (n=1) |
| MuSiQue | 0.5781 → **0.5345** | 0.5203 → 0.9032 (n=1) |
| Hotpot | **0.4644** → **0.6408** | 0.6444 → **0.4032** (n=8) |

This is the cleanest statement the audit produces:

- **Generic core-exit is a good *ranker* and a bad *presence* signal.** Its both-evidence AUC is
  above chance on all five eligible corpora (0.5345–0.8402) — including Hotpot, where its raw AUC is
  0.4644. Hotpot does not break the core-exit score's ranking; it breaks its *coverage*.
- **NER is a good *presence* signal and an unreliable *ranker*.** On three of five corpora fewer than
  30 labelled pairs have NER evidence on both sides, and where there is any support (Hotpot, n=8) the
  ranking is **below chance** (0.4032). Its 6/6 sign-consistency is carried by the floor: candidates
  with NER-corroborated core-exit evidence are better bets than candidates without.

The candidate-coverage numbers confirm it directly: `SRC_NER` can score only **12.7 %–43.2 %** of the
`K(q)` universe (WebQSP 0.1273, 2Wiki 0.1382, MuSiQue 0.2255, MetaQA 0.4316) against `SRC_ALL`'s
68.7 %–93.7 %. Under `SRC_NER`, most of the pool ties at the floor and keeps the frozen order.

---

## STEP 3 — EXCLUSIVE SUPPORT

Admitted candidates bucketed by which families supply their core-exit evidence; `ratio` is
`good_swaps / bad_swaps`.

| corpus | ONLY_NER | ONLY_STRUCT | MULTI_FAMILY |
|---|--:|--:|--:|
| MetaQA | 5104 / 4241 / **1.204** | 7769 / 8278 / 0.939 | 14247 / 9421 / **1.512** |
| WebQSP | 434 / 277 / **1.567** | 2172 / 2117 / 1.026 | 220 / 203 / 1.084 |
| 2Wiki | 111 / 110 / **1.009** | 126 / 407 / **0.310** | 31 / 43 / 0.721 |
| MuSiQue | 0 / 3 / — | 52 / 292 / 0.178 | 13 / 122 / 0.107 |
| Hotpot | 98 / 229 / **0.428** | 197 / 1762 / **0.112** | 89 / 125 / 0.712 |
| SQuAD | 0 / 0 / — | 0 / 233 / 0.000 | 0 / 67 / 0.000 |

**Yes — one family is responsible for a disproportionate share of the collateral damage, and it is
the plain-structural one.** `ONLY_STRUCT` has the worst gain/loss ratio of the three buckets on
**every corpus that has any NER-exclusive support** (MetaQA, WebQSP, 2Wiki, Hotpot: 4/4). The
starkest case is Hotpot, where structural-only candidates are admitted 1,959 times and are the
required-and-missing side only 197 of them (0.112) — while NER-exclusive candidates run at 0.428 and
multi-family at 0.712. MuSiQue is the one apparent inversion and it has no support in the NER bucket
(3 candidates total). SQuAD has zero good swaps in any bucket, consistent with STEP 2.

*Uncontrolled confound, stated rather than hidden.* Buckets are assigned by which family supports a
candidate, so this is an association, not an intervention: NER-supported candidates could differ from
structural-only ones for reasons other than provenance. STEP 4 was the intended control and finds no
universal source-quality trend that would explain it; the properly controlled comparison is the
`ALL_NER` vs `ALL_STRUCT_ONLY` pair in STEP 2, which shares the pair population exactly and shows the
same direction (6/6 above chance vs 2/6).

---

## STEP 4 — SOURCE QUALITY (diagnostic)

Within each family, `p_required = P(admitted candidate is required-and-missing | bin)`, over four
axes of information the substrate already carries. Bin edges come from the frozen contract
(`prot` = canonical ranks 0–43 split into four equal blocks, `bnd` = 44–49); nothing was searched.

**No universal monotone trend exists on any axis.** Directions flip by corpus *and* by family:

| axis | what happens |
|---|---|
| `core_source` (protected core vs not) | MetaQA both families **up**; WebQSP NER **down** (0.0079 → 0.0039); MuSiQue both **down**; Hotpot NER up / STRUCT down |
| `hop` | MetaQA peaks at hop 2 for both families (0.029 / 0.023) — **non-monotone**; WebQSP STRUCT peaks at hop 1 (0.019) and collapses (0.002) |
| `src_cpos_quartile` | MetaQA decreasing across the protected core; WebQSP STRUCT decreasing; 2Wiki/Hotpot flat at 0.000x |
| `src_conf_quartile` (`T1_SOURCE`) | MetaQA is the clearest signal and it **points opposite ways in the two families**: NER rises monotonically 0.0137 → 0.0226, STRUCT_ONLY falls monotonically 0.0234 → 0.0132 |

The MetaQA `src_conf` inversion is the most interesting single row in this step — ancestor trust
predicts usefulness positively for NER-corroborated edges and negatively for plain-structural ones —
but it does not replicate: WebQSP NER also rises (0.0039 → 0.0074) while WebQSP STRUCT is U-shaped,
and Hotpot NER falls. On every text corpus the base rate is 0.0000–0.0013, so the bins there are
noise, not weak signal.

`STEP4_UNIVERSAL_MONOTONE_TREND = NO` on all six (family × axis) combinations tested.

---

## STEP 5 — IS ONE MECHANISM JUSTIFIED?

The gate has two legs. They disagree, so both are reported.

| leg | verdict |
|---|---|
| same GOOD-vs-BAD direction across all eligible corpora | **PASS**, and uniquely: `SRC_NER` and `ALL_NER` are the only keys of the five that are sign-consistent, and they are sign-consistent where the generic control is not |
| materially better separation than generic SRC | **SPLIT**. NER beats the control only on the two corpora where the control is weakest (2Wiki 0.6268 vs 0.5066, Hotpot 0.6444 vs 0.4644) and loses on the other four. Mean AUC is a wash (0.6055 vs 0.6116). Worst-case AUC is materially better: **0.5203 vs 0.4644**, i.e. above chance versus below it |

The gate was treated as **passed on the universality reading** — that is what leg 1 measures and what
the directive's primary STEP-2 question asks — and `TRUSTED_SRC_ASSIGNMENT` was built from the
`STRUCT_AND_NER` family alone. No weight, no learned gate, no dataset branch, no threshold: a family
either supplies evidence for a candidate or it does not, and candidates with none keep the frozen
order. Both readings are recorded so the ruling is auditable; had the gate been read as failed, the
STEP 6–7 measurement below is what would have had to justify it anyway.

## STEPS 6–7 — EXACT P50, ALL SIX CORPORA

`K = K0`, `B = 6`, `P = 50`, fixed-K replacement, no extra candidate depth. `net = gained − lost`
against the FROZEN pool; `*` = McNemar significant. `SRC_ALL` is the parity anchor and reproduces
the prior phase exactly (`8/8`, `2/2`, `2/2`, `2/2`, `2/2`; SQuAD had no prior V6 run).

| corpus / slice | sel | FROZEN | `SRC_ALL` (control) | **`SRC_NER`** | `SRC_STRUCT_ONLY` | `ALL_NER` | `ALL_STRUCT_ONLY` |
|---|---|--:|--:|--:|--:|--:|--:|
| MetaQA hop1 | F6 | 0.9955 | +0 | +0 | +0 | +0 | +0 |
| MetaQA **hop2** | F6 | 0.7297 | +6 | **+0** | −2 | +4 | +3 |
| MetaQA **hop2** | G4 | 0.7297 | −4 | **−6** | −7 | −5 | −6 |
| MetaQA **hop3** | F6 | 0.2583 | +4 | **+1** | +2 | +2 | +2 |
| MetaQA **hop3** | G4 | 0.2583 | **+16\*** | **+11\*** | +10\* | +10\* | +8 |
| MetaQA ALL | F6 | 0.6612 | +10\* | +1 | +0 | +6 | +5 |
| MetaQA ALL | G4 | 0.6612 | +12 | +5 | +3 | +5 | +2 |
| WebQSP | F6 | 0.7646 | +2 | +0 | +0 | +0 | +3 |
| WebQSP | G4 | 0.7646 | −1 | −3 | −2 | −3 | +1 |
| 2Wiki | F6 | 0.9435 | −5 | **+0** | −3 | +0 | −3 |
| 2Wiki | G4 | 0.9435 | −7 | **+0** | −4 | +0 | −3 |
| MuSiQue | F6 | 0.9635 | **−10\*** | **+0** | −10\* | −1 | −8\* |
| MuSiQue | G4 | 0.9635 | **−14\*** | **−8\*** | −14\* | −9\* | −12\* |
| Hotpot | F6 | 0.9505 | **−20\*** | **+0** | −20\* | +0 | −18\* |
| Hotpot | G4 | 0.9505 | **−32\*** | **−16\*** | −34\* | −15\* | −32\* |
| SQuAD | F6 | 0.9875 | **−14\*** | **+0** | −14\* | +0 | −11\* |
| SQuAD | G4 | 0.9875 | **−13\*** | **−1** | −13\* | −1 | −11\* |

**Significant-cell ledger.**

| key | significant gains | significant regressions |
|---|---|---|
| `SRC_ALL` (control) | 2 (MetaQA ALL/F6, hop3/G4) | **6** (MuSiQue ×2, Hotpot ×2, SQuAD ×2) |
| **`SRC_NER`** | 1 (MetaQA hop3/G4) | **2** (MuSiQue/G4, Hotpot/G4) |
| `ALL_NER` | 1 (MetaQA hop3/G4) | **2** (MuSiQue/G4, Hotpot/G4) |
| `SRC_STRUCT_ONLY` | 1 (MetaQA hop3/G4) | **6** |
| `ALL_STRUCT_ONLY` | 0 | **6** |

### What actually happened

**The provenance channel converts as a safety filter and nothing else.** On `F6` — the promoted
selector — `SRC_NER` is `+0` on five of six corpora and `+1` on the sixth, while it removes *every*
significant `F6` regression the control incurs: MuSiQue `−10*` → `+0`, Hotpot `−20*` → `+0`,
SQuAD `−14*` → `+0`, 2Wiki `−5` → `+0`. That is a real and complete removal of the cross-corpus harm.

But it is the safety of doing almost nothing. `SRC_NER` can score only **10.2 %–43.2 %** of the
`K(q)` universe (Hotpot 0.1024, WebQSP 0.1273, 2Wiki 0.1382, SQuAD 0.1579, MuSiQue 0.2255, MetaQA
0.4316), so 57–90 % of every pool ties at the floor and is re-ordered by the frozen rank. The pool
changes on 1,358–1,995 queries per corpus and the exact-P50 outcome moves on essentially none of
them. **`SRC_NER` on `F6` is, to within one query, the frozen selector.**

**And where the family had its best separation, it converted worst.** Hotpot is the corpus where
NER's AUC advantage over the control was largest — 0.6444 against a below-chance 0.4644 — and it is
where NER still leaves a `−16*` `G4` regression. Separation measured on forced swaps did not predict
exact-P50 conversion, for the fourth time in this program (see the read-stage audit, the
calibration/dilution phase, and the transformation algebra).

---

## DECISION

| return | value |
|---|---|
| `BEST_SRC_EDGE_FAMILY` | **`STRUCT_AND_NER`** — traversal edges that also carry an existing NER "shares-entity" edge (score `SRC_NER`) |
| `EDGE_FAMILY_SIGN_CONSISTENT` | **YES** — `SRC_NER` and `ALL_NER`, 6/6 eligible rows AUC > 0.5, minima 0.5203 / 0.5288. The generic core-exit control is **not** (5/6, Hotpot 0.4644), and neither STRUCT-only key is (4/6, 2/6) |
| `METAQA_HOP2` | `0.7297 → 0.7297`, net **+0** (F6); `→ 0.7207`, net −6 ns (G4). **No improvement.** |
| `METAQA_HOP3` | `0.2583 → 0.2598`, net **+1** ns (F6); `→ 0.2748`, net **+11 SIG** (G4) |
| `CROSS_CORPUS_SAFE` | **NO** — MuSiQue `G4 −8*` and Hotpot `G4 −16*` survive. (On `F6` alone it *is* safe: 0 significant regressions on all six.) |
| `PROMOTED` | **NONE** — the bar was "material MetaQA hop2/hop3 improvement AND no significant regression anywhere"; hop2 is +0 and two significant `G4` regressions remain |
| `L1_FROZEN` | **NO** |

### CLASSIFICATION: **B. PROVENANCE_CHANNEL_PARTIAL**

Not **A**: the channel delivers no material MetaQA improvement (hop2 `+0`, hop3 `+1` on the promoted
selector) and is not cross-corpus safe on `G4`.

Not **C**: this phase produced the **first sign-consistent statistic found anywhere in the L1 search
program**. The prior phase's answer was `0 of 33` score keys; here two keys are above chance on every
eligible row, and they are above chance precisely where the incumbent control drops below it. The
channel also converts — completely — as a safety filter on `F6`. Calling that "exhausted" would
misreport a reproducible, directionally universal signal. *(Row-definition caveat: the prior phase's
`0/33` was over five corpus-level slices; this phase's rows are the seven the directive names, of
which six are eligible. The comparison is indicative, not exact.)*

**What "partial" means here, precisely.** The provenance channel is recovered as a **presence signal**
and not as an **evidence channel**. NER membership tells you *which candidates are worth trusting*
consistently across corpora; it cannot tell you *how to order them* — its both-evidence AUC is
unsupported on three corpora (n = 1, 1, 29) and **below chance** where it has any support at all
(Hotpot 0.4032, n = 8). The complementary decomposition is the phase's main scientific result:

- **generic core-exit = good ranker, bad presence signal** (both-evidence AUC 0.5345–0.8402 on all
  five eligible corpora, including Hotpot's 0.6408, yet raw AUC 0.4644 there);
- **NER = good presence signal, unreliable ranker.**

Neither half is a universal admission rule on its own, and the audit found no parameter-free way to
combine them: STEP 4 tested the four axes of existing source information the substrate carries
(protected-core source, canonical-rank quartile, `T1_SOURCE` confidence quartile, hop) and **none has
a universal monotone direction** — MetaQA's clearest trend, `T1_SOURCE`, points in opposite
directions for the two families and does not replicate.

### What this closes

1. **The kNN hypothesis is closed structurally, not statistically.** The frozen beam expands
   `master_nodes.neighbors`, so `traversal ∩ KNN = 0` on all six corpora. The semantic family has
   never been in an L1 core-exit transition and cannot explain either the MetaQA win or the text
   regressions. Do not test it again.
2. **The provenance axis itself is enumerated and measured.** Only two families exist in the
   traversal index space (STRUCT, NER); both were measured, alone and in combination, with and
   without the core-exit restriction, on identical pair populations, on all six corpora, at exact
   P50 under both selectors. There is no third family to try.
3. **The noisy family is identified.** `STRUCT_ONLY` has the worst gain/loss ratio on 4/4 corpora
   with NER support, and both STRUCT-only keys carry all six significant regressions. The prior
   phase's cross-corpus harm is attributable to plain-structural core-exit evidence.

### What it does not close

The safety half converts and the gain half does not, and the reason is coverage: the family that is
trustworthy is present on too few candidates to rank with. That is a statement about the *density* of
NER corroboration in the frozen substrate (2.74 %–54.26 % of traversal edges), not a proof that no
provenance-aware admission rule exists. It does mean that no rule built from **these two families
alone, at this coverage, without a learned combination** can clear the promotion bar — every such
rule was tested here.

**Constraints honoured.** No transformation experiment was run. No new graph work, no new traversal,
no new encoder. `P` stayed at 50, `B` at 6, `K` at the frozen `K(q)`. Nothing was learned, fitted,
weighted, thresholded, or switched on dataset identity. No relation label was invented — every family
came from a file already on disk. No L2, no L3, no TEST.
