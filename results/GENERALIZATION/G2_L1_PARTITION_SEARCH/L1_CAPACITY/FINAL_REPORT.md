# L1 CAPACITY + TRANSFORMATION EVICTION — FINAL REPORT

**Phase:** two directives run as one program — *L1 QUERY-CONDITIONED TRANSFORMATION ALGEBRA*
(replace `x' = x + delta` with parameter-free graph-derived `x' = T(x)`) and *L1 CAPACITY +
TRANSFORMATION EVICTION* (test the only two explanations the admission phase left standing:
**A. K is slightly too small**, **B. we need new mathematical information to decide safe evictions**).

**Contract held throughout.** `MASTER_TOPOLOGY=C`, `P_MAIN=50` exactly, `B=6`, `M_struct=64`,
`M_ret=32`, `K0=60`, `MAX_HOPS=3`, `BEAM=64`. No LLM, no MLP, no learning, no new encoder, no new
graph traversal, no dataset-specific switch, no threshold grid, no L2, no L3, no TEST. Every
transformation is derived from edges the frozen beam **already** scored. Graph parity is exact on
every corpus and every run (`parity` = nq/nq in each `diag/*.json`).

**Artifacts.** `TABLES.md` (all numbers, machine-generated), `RETURNS.json`,
`diag/{parta,tf,pb,swap}_*.json`, `diag/closure.json`, `diag/gates.json`,
`diag/orthogonal_degeneracy.json`. Code `scratchpad/_l1kt_{parta,tf,partb,swap,closure,gates,report}.py`.

---

## VERDICTS

| directive | classification | the single fact that decides it |
|---|---|---|
| L1 QUERY-CONDITIONED TRANSFORMATION ALGEBRA | **C. TRANSFORMATION_GEOMETRY_EXHAUSTED** | 0 of 16 (state x family x mode) cells beat the `NO_TRANSFORM` identity control on a majority of the six corpora, and every cell has negative mean lift |
| L1 CAPACITY + TRANSFORMATION EVICTION | **D. CURRENT_L1_SUBSTRATE_EXHAUSTED** | `BEST_K_DELTA = 0` (F6 closure theorem) and `GOOD_BAD_SWAP_SIGN_CONSISTENT = NO` (0 of 33 keys), with the one significant P50 gain reproduced by a no-transformation control |

`PROMOTED = NONE`. `L1_FROZEN = NO`.

Named returns, all machine-derived, in `RETURNS.json`:

| return | value |
|---|---|
| `BEST_K_DELTA` | **0** |
| `BEST_K_METAQA_HOP3` | 0.2583 under F6 at every delta (unchanged from frozen) |
| `BEST_K_CROSS_CORPUS_SAFE` | vacuously YES under F6 (nothing changes); NO under G4 |
| `CAPACITY_PRESENT_SELECTOR_CANNOT_CONVERT` | **YES** |
| `BEST_TRANSFORM_FAMILY` | **NONE** (least-bad cell `r_q/F3_RANK1_TRANSPORT/chain`, mean AUC lift −0.0074, beats identity on 3/6) |
| `TRANSLATION_ADMISSION_R6 / R50` | 0.0725 / 0.3113 (metaqa); best @6 of the two on 4 of 6 corpora |
| `TRANSFORM_ADMISSION_R6 / R50` | 0.0757 / 0.3610 (metaqa); best @50 of the two on 4 of 6 corpora |
| `CHAIN_HOP2_GAIN / CHAIN_HOP3_GAIN` | +0.0309 / +0.0989 mean, but only for `F0_TRANSLATION` and only as the additive artefact; orthogonal families flat to −0.012 |
| `TRANSFORM_F6_METAQA_HOP2` | 0.7417, net **+8 (+8/−0)**, *p*=0.0078 **SIG** |
| `TRANSFORM_F6_METAQA_HOP3` | 0.2643, net +4, *p*=0.2188 ns |
| `TRANSFORM_G4_METAQA_HOP3` | 0.2778, net **+13 (+16/−3)**, *p*=0.0044 **SIG** |
| `TRANSFORM_FIXED_K_METAQA_HOP3` | 0.2643 (F6) |
| `TRANSFORM_CROSS_CORPUS_SAFE` | **NO** — significant regressions on musique, hotpot and squad |
| `GOOD_BAD_SWAP_AUC` | best key `r_q/F1_HOUSEHOLDER/V4_SRC_EXIT` = 0.5683 / 0.5475 / 0.6608 / 0.5763 / **0.4925** |
| `GOOD_BAD_SWAP_SIGN_CONSISTENT` | **NO** (0 of 33 keys; 5 of 33 on the both-evidence subset, one of which is the identity control) |
| `TRANSFORM_OVERHEAD_MS` | 2.90–67.44 ms/q for one family x one state; 46–1,079 ms/q for the full algebra; 27.7–352.9 ms/q end-to-end for the replacement |

---

## PART A — MICRO K CAPACITY LADDER (explanation A: "K is slightly too small")

The entire frozen SAFE pool is preserved in frozen order and SRC-assignment candidates are
**appended**, never substituted, at K(q)+{0,4,8,16,32}. `delta = 0` reproduces the frozen pool
exactly (asserted per query) and F6 parity against the frozen system is exact on all six corpora.

### A1 — the oracle does rise, on all six

| corpus/slice | K(q) | K0+0 | K0+32 | delta |
|---|---|---|---|---|
| metaqa/hop2 | 49.2 | 0.8033 | 0.8739 | **+0.0706** |
| metaqa/hop3 | 49.2 | 0.3634 | 0.4204 | **+0.0570** |
| metaqa/ALL | 49.2 | 0.7212 | 0.7638 | +0.0426 |
| webqsp/ALL | 34.4 | 0.7886 | 0.8104 | +0.0218 |
| musique_clean/ALL | 18.6 | 0.9750 | 0.9860 | +0.0110 |
| 2wiki_clean/ALL | 41.8 | 0.9515 | 0.9565 | +0.0050 |
| hotpotqa_clean/ALL | 55.4 | 0.9645 | 0.9685 | +0.0040 |
| squad_clean/ALL | 15.8 | 0.9945 | 0.9955 | +0.0010 |

**But the curve does not left-saturate.** Marginal oracle gain *per added candidate*:

| corpus/slice | +0→+4 | +4→+8 | +8→+16 | +16→+32 |
|---|---|---|---|---|
| metaqa/hop2 | 0.00375 | 0.00225 | 0.00281 | 0.00151 |
| metaqa/hop3 | 0.00075 | 0.00038 | **0.00263** | 0.00197 |
| webqsp/ALL | 0.00140 | 0.00052 | 0.00089 | 0.00044 |
| 2wiki_clean/ALL | 0.00012 | 0.00000 | 0.00019 | 0.00019 |
| musique_clean/ALL | 0.00038 | 0.00037 | 0.00031 | 0.00034 |
| hotpotqa_clean/ALL | 0.00012 | 0.00000 | 0.00038 | 0.00003 |
| squad_clean/ALL | 0.00000 | 0.00000 | 0.00000 | 0.00006 |

Only **webqsp** decays like a corpus whose K was slightly too small. MetaQA hop3 — the target — is
*non-monotone and largest in the middle* (0.00263 at +8→+16), and four corpora are flat-to-rising.
The missing candidates are spread deep, not bunched just past the cut. **The premise of explanation A
is not satisfied.**

### A2 — the actual selectors convert exactly none of it

`capacity-only` holds the selector fixed and moves only the depth, isolating what the extra
candidates are worth:

| corpus/slice | F6 capacity-only | G4 capacity-only |
|---|---|---|
| metaqa/hop2 | **0** | −1 |
| metaqa/hop3 | **0** | **0** |
| metaqa/ALL | **0** | −1 |
| webqsp/ALL | **0** | −3 |
| 2wiki_clean/ALL | **0** | −1 |
| musique_clean/ALL | **0** | −4 |
| hotpotqa_clean/ALL | **0** | −20 |
| squad_clean/ALL | **0** | −2 |

F6 converts **exactly zero at every depth on every corpus** (net 0, gained 0, lost 0 — not a wash,
literally no query changes). G4 converts **negatively everywhere** and gets worse with K. G4's
MetaQA hop3 `+13` is present already at `K0+0`: it is a selector effect on the frozen pool, not
capacity.

`CAPACITY_PRESENT_SELECTOR_CANNOT_CONVERT = YES`.

### A2 — and F6's zero is a theorem, not a measurement

The SAFE pool is `bnd ++ chal`, and `chal` is *defined* as every partition carrying struct (`spos`)
or ret (`rpos`) evidence outside base50. An appendable partition therefore has **neither**, so its
F6 score is at most `1/(K0+cpos) ≤ 1/110 = 0.009091`, while all six `bnd` members (cpos 44..49)
score at least `1/(60+49) = 0.009174`. The sixth-best pool score strictly dominates every
appendable candidate at every depth.

| corpus | appended candidates | with spos\|rpos | could enter top-6 | max appended score | min 6th pool score |
|---|---|---|---|---|---|
| metaqa | 62,776 | **0** | **0** | 0.009091 | 0.009346 |
| webqsp | 42,121 | **0** | **0** | 0.009091 | 0.009259 |
| 2wiki_clean | 42,824 | **0** | **0** | 0.009091 | 0.009524 |
| musique_clean | 59,892 | **0** | **0** | 0.009091 | 0.009174 |
| hotpotqa_clean | 63,214 | **0** | **0** | 0.009091 | 0.009615 |
| squad_clean | 53,545 | **0** | **0** | 0.009091 | 0.009174 |

324,372 appended candidates, zero with any struct or ret evidence, zero that could ever enter the
top-6, and the bound is **tight** (equality on musique and squad). Appending cannot reach the frozen
selector's output. This is why the only way transformation evidence can act at all is **replacement**
— which is what PART B does.

### A3 — cost

Admission overhead at +32 is +0.017 to +0.064 ms/query on top of a 15.565 ms L1 — negligible, and
irrelevant, because there is no gain to buy. **There is no gain/cost knee: the numerator is zero.**

`BEST_K_DELTA = 0` (no K increase is justified). `BEST_K_METAQA_HOP3 = 0.2583` (unchanged from
frozen at every delta under F6). `BEST_K_CROSS_CORPUS_SAFE = YES` only in the vacuous sense that
appending changes nothing under F6; under G4 it is `NO` (significant regressions on musique −11 and
hotpot −31).

**Explanation A is refuted.**

---

## THE TRANSFORMATION ALGEBRA (STEPS 1-6)

### STEP 1-2 — four parameter-free operators, all gated on real edges

For a real explored edge `u -> v` (unit rows of `Xn`) acting on a query state `z`, all matrix-free
and `O(d)`; no `d x d` matrix is ever formed and nothing is fitted:

| family | definition |
|---|---|
| `F0_TRANSLATION` | `z' = z + (v - u)` |
| `F1_HOUSEHOLDER` | `w = normalize(u - v)`, `z' = z - 2 w (w^T z)` |
| `F2_MIN_ROTATION` | minimal rotation inside `span(u,v)` carrying `u` to `v`, identity on the complement |
| `F3_RANK1_TRANSPORT` | `z' = z + (v - u)(u^T z)/(u^T u)` |

Gates on 18,511 real replayed edges (`diag/gates.json`):

| family | `max abs T(u)-v` | `norm(Tz)/norm(z)` on random unit z | chain == manual | scalar == vector |
|---|---|---|---|---|
| F0_TRANSLATION | 1.5e-08 | 1.570656 +- 0.070539 | **0.0** | 1.2e-06 |
| F1_HOUSEHOLDER | 2.2e-07 | **1.000000 +- 0.000000** | **0.0** | 3.2e-07 |
| F2_MIN_ROTATION | 7.5e-08 | **1.000000 +- 0.000000** | **0.0** | 3.2e-07 |
| F3_RANK1_TRANSPORT | 1.5e-08 | 1.000004 +- 0.000619 | **0.0** | 3.8e-07 |

F2 is the identity on `span(u,v)^perp` (5.6e-09), F1 on `(u-v)^perp` (5.6e-09), F1 is an involution
(7.5e-09), and composition along the recorded parent chain equals sequential manual application
**exactly** for all four. A closed-form `O(1)` scalar transport (derived so that `cos(T(z),v)`,
`norm(T(z))` and `<T(z),z0>` never need the transformed vector) matches the `O(d)` vector path to
1.2e-06; only the ~3% of edges that are the parent of another edge ever materialise a vector. That
took the measurement from 3.58 s/query to 226 ms/query.

### The orthogonal branch is algebraically degenerate

`F1_HOUSEHOLDER` and `F2_MIN_ROTATION` are genuinely different operators (max abs elementwise
difference **0.692** on random states) yet their single-hop target compatibility is **identical to
3.4e-08 on every corpus** — see the STEP 3 table, where the two columns agree to four decimals
everywhere. The reason is a one-line proof, not a coincidence:

> For any **orthogonal** `T` with `T(u) = v`:
> `<T z, v> = <T z, T u> = <z, u>` and `norm(T z) = norm(z)`,
> hence `cos(T(z), v) = cos(z, u)` **exactly**.

Verified numerically to 3e-08 (`diag/orthogonal_degeneracy.json`). So single-hop target
compatibility under **any** norm-preserving transformation is just the query's similarity to the
**source** node: the transformation contributes literally nothing beyond re-reading `u`. This closes
**STEP 11 (orthogonal Procrustes group operator) algebraically** for the primary measurement — a
fitted group-level orthogonal map would collapse to the same scalar. The gate on STEP 11 ("only if
an edge-level family shows clear signal") was therefore never opened, and would have been vacuous
for single-hop transport even if it had been.

### STEP 3 + STEP 5 — no transformation beats doing nothing, on any majority of corpora

The identity control `NO_TF = cos(z0, x_v)` is evaluated on the **same** edge population, with the
same positives (edge target partition is REQUIRED and MISSING from the frozen P50). Lift over that
control, `AUC_transform - AUC_NO_TRANSFORM`:

```
cell                                   metaqa     webqsp  2wiki_cle  musique_c  hotpotqa_  squad_cle  up/down
  q_raw/F0_TRANSLATION/1hop           -0.0612    +0.0520    -0.0782    +0.0206    -0.2131    +0.0043     3/3
  q_raw/F1_HOUSEHOLDER/1hop           -0.0418    +0.0735    -0.1406    +0.0003    -0.0419    -0.1480     2/4
  q_raw/F2_MIN_ROTATION/1hop          -0.0418    +0.0735    -0.1405    +0.0003    -0.0419    -0.1480     2/4
  q_raw/F3_RANK1_TRANSPORT/1hop       -0.0061    +0.0500    -0.0416    -0.0031    -0.0734    -0.0764     1/5
  q_raw/F0_TRANSLATION/chain          +0.0217    +0.0179    -0.0852    -0.0500    -0.2010    -0.1395     2/4
  q_raw/F1_HOUSEHOLDER/chain          +0.0254    -0.0638    -0.1103    -0.0987    -0.1354    -0.4034     1/5
  q_raw/F2_MIN_ROTATION/chain         +0.0254    -0.0638    -0.1104    -0.0987    -0.1356    -0.4033     1/5
  q_raw/F3_RANK1_TRANSPORT/chain      +0.0291    +0.0904    -0.0683    -0.0133    -0.1805    -0.2180     2/4
  r_q/F0_TRANSLATION/1hop             -0.0662    +0.0077    -0.0653    +0.0112    -0.1022    -0.0742     2/4
  r_q/F1_HOUSEHOLDER/1hop             -0.0666    +0.0750    -0.1708    +0.0441    +0.0598    -0.2451     3/3
  r_q/F2_MIN_ROTATION/1hop            -0.0666    +0.0750    -0.1708    +0.0441    +0.0598    -0.2451     3/3
  r_q/F3_RANK1_TRANSPORT/1hop         -0.0126    +0.0443    -0.0596    +0.0337    +0.0100    -0.1020     3/3
  r_q/F0_TRANSLATION/chain            -0.0843    -0.0126    -0.1002    -0.0175    -0.0558    +0.0113     1/5
  r_q/F1_HOUSEHOLDER/chain            -0.0928    -0.1727    -0.1360    -0.1505    +0.1627    -0.3169     1/5
  r_q/F2_MIN_ROTATION/chain           -0.0780    -0.1848    -0.1513    -0.1113    +0.1249    -0.1141     1/5
  r_q/F3_RANK1_TRANSPORT/chain        -0.0138    +0.0760    -0.0431    +0.0444    +0.0300    -0.1379     3/3
```

**Not one of the 16 cells beats the identity control on more than 3 of 6 corpora**, and every cell
has a negative mean lift. Which state wins flips by corpus (metaqa needs `q_raw`, webqsp / musique /
squad need `r_q`); which family wins flips by corpus (F3 chain on metaqa / webqsp, F1/F2 1hop on
webqsp / musique / hotpot, F0 chain on squad, **nothing** on 2wiki). The one clean regime signal is
that webqsp — the KB substrate — likes transformations at every setting, exactly like the offset
channel this algebra generalises: it is a KB-regime effect, and that is what disqualifies it as a
universal L1 mechanism.

**Primary STEP 3 question, answered literally.** *Does any family rank missing/needed target
partitions better than translation?* **Yes** — `F3_RANK1_TRANSPORT` beats `F0_TRANSLATION` in
single-hop mode on 4 of 6 corpora and on the metaqa partition rank (0.3867/0.3858 for F1/F2 vs
0.4464 for F0), and translation is the worst or near-worst family almost everywhere. **But the
comparison that decides the phase is against `NO_TRANSFORM`, and there the whole family loses.**

Translation also has a specific artefact worth recording: its mean `cos(z_h, x_v)` *rises* with hop
(metaqa 0.6750 -> 0.6863 -> 0.7168) purely because `z + (v-u)` literally adds `v` to the state. It
inflates the score of **every** target equally, positive and negative alike — which is why F0 has the
highest mean compatibility and one of the **worst** AUCs. The orthogonal families cannot inflate and
their per-hop means are flat (0.5649 / 0.5649 / 0.5617).

### STEP 6 — composition stability

| family | `norm(z1..z3)/norm(z0)` (metaqa) | `cos(z_h, z0)` h=1,2,3 |
|---|---|---|
| F0_TRANSLATION | 1.240 / 1.262 / 1.335 | 0.692 / 0.625 / **0.570** |
| F1_HOUSEHOLDER | **1.000 / 1.000 / 1.000** | 0.925 / 0.890 / 0.865 |
| F2_MIN_ROTATION | **1.000 / 1.000 / 1.000** | 0.855 / 0.802 / 0.769 |
| F3_RANK1_TRANSPORT | 1.048 / 1.068 / 1.109 | 0.873 / 0.812 / 0.769 |

Same ordering on every corpus. **Nothing explodes** at `MAX_HOPS = 3`, but translation drifts
materially — norm +24% to +43% by hop 3 (musique 1.426) and direction cosine down to 0.50-0.57 —
while the two orthogonal families are exact isometries by construction and rank-1 transport is
near-isometric. So the STEP-6 question is answered: **translation composition drifts (it does not
explode), rank-1 transport is stable, and the orthogonal families are exactly stable** — and none of
that stability converts into ranking quality.

### STEP 7 — transformed states as an admission ranker

`MISSING_REQUIRED_ADMISSION_RECALL` = of the required partitions that the frozen P50 **misses**,
what fraction does a score put inside its own top-6 / top-50 of `FULL_VISITED`? Transformed states
are used **only to rank partitions the frozen beam already visited** — no new expansion.

```
corpus             n     FROZEN_SAFE       SRC_ASSIGN       OFFSET_RAW   V0_NOTRANSFORM
                            @6 / @50         @6 / @50         @6 / @50        @6 / @50
metaqa           467   0.0000/0.3222   0.0853/0.4158   0.0629/0.3191   0.0311/0.2844
webqsp           132   0.0000/0.3077   0.0695/0.3097   0.0188/0.2288   0.0642/0.2596
2wiki_clean       22   0.0000/0.4318   0.0455/0.3864   0.0000/0.1591   0.0909/0.2955
musique_clean     26   0.0000/0.8462   0.0769/0.8077   0.0769/0.8462   0.0385/0.8846
hotpotqa_clean    30   0.0000/0.5500   0.0333/0.3167   0.0000/0.0333   0.1333/0.3167
squad_clean       10   0.0000/1.0000   0.0000/0.1000   0.0000/0.2000   0.1000/0.1000
```

| corpus | best **translation** @6 / @50 | best **transformation** @6 / @50 | winning transformation cell |
|---|---|---|---|
| metaqa | 0.0725 / 0.3113 | **0.0757 / 0.3610** | `q_raw/F1_HOUSEHOLDER/V1_SINGLE` |
| webqsp | **0.1676** / 0.3865 | 0.1112 / **0.4131** | `q_raw/F3_RANK1_TRANSPORT/V1_SINGLE` |
| 2wiki_clean | **0.0909 / 0.4318** | 0.0455 / 0.4091 | `q_raw/F2_MIN_ROTATION/V1_SINGLE` |
| musique_clean | **0.0769** / 0.7692 | 0.0385 / **0.8846** | `r_q/F1_HOUSEHOLDER/V3_MAXVIEW` |
| hotpotqa_clean | **0.1500** / 0.2167 | 0.1333 / **0.3167** | `r_q/F1_HOUSEHOLDER/V3_MAXVIEW` |
| squad_clean | 0.0000 / 0.2000 | 0.0000 / 0.2000 | `r_q/F1_HOUSEHOLDER/V2_CHAIN` |

`TRANSFORM_ADMISSION_R50 > TRANSLATION_ADMISSION_R50` on 4 of 6 — so, read literally, **STEP 3's
primary question is answered YES: transformation ranks missing/needed targets better than
translation.** But that answer does not survive contact with the two controls it has to beat:

- **at the depth that matters (@6, the actual `B = 6` budget) translation is the better of the two on
  4 of 6 corpora**, and the transformation only wins on metaqa by 0.0032 (1.5 queries of 467);
- **the winning transformation cell is a different cell on every corpus** — four distinct
  `state/family/view` combinations across six corpora. There is no universal operator to promote.

The reference rows are more informative than either. `SRC_ASSIGN` is the best @6 admission signal on
metaqa / webqsp / 2wiki and the best @50 on metaqa / webqsp; `FROZEN_SAFE` — the frozen ordering
itself, restricted to the SAFE pool — has the **best @50 on 2wiki / musique / hotpot / squad**, on
squad by a factor of 10 (1.0000 vs 0.1000). Two facts follow, and they frame the whole phase:

1. `FROZEN_SAFE @6 = 0.0000` on all six **by construction** — the frozen top-6 is exactly the set
   that already failed, so any repair has to come from re-ordering, never from re-scoring alone.
2. On four of six corpora the frozen SAFE ordering **already ranks the missing-required partitions
   better, deep in its list, than any transformation does**. The evidence is not absent from the
   frozen system; it is present and un-promoted. This is the same picture the read-stage audit found
   (separability real, correlation with exact-P50 nil).

### STEP 8 — multi-state coverage, without collapsing to one scalar

The four query states `q0..q3` are kept apart and each view scored on its own, as instructed.
Best key per view, `@6 / @50`:

| view | metaqa | webqsp | 2wiki | musique | hotpot | squad |
|---|---|---|---|---|---|---|
| `T_SINGLE` (1 hop, no composition) | **0.076 / 0.361** | **0.111 / 0.413** | 0.045 / 0.409 | 0.077 / 0.769 | **0.133** / 0.233 | 0.000 / 0.200 |
| `T_CHAIN` (composed along the real path) | 0.074 / 0.349 | 0.086 / 0.294 | **0.136** / 0.341 | 0.038 / 0.692 | 0.067 / 0.200 | 0.000 / 0.200 |
| `T_MAX_VIEW` (max over q0..q3) | 0.074 / 0.349 | 0.086 / 0.294 | 0.000 / 0.295 | 0.038 / **0.885** | **0.133 / 0.317** | 0.000 / 0.100 |
| `T_ASSIGN` (source in the protected core) | 0.076 / 0.357 | 0.065 / 0.309 | 0.091 / **0.432** | 0.038 / 0.731 | 0.100 / 0.283 | 0.000 / 0.200 |

**Composition never wins at @50 on any corpus** (it ties squad). The uncomposed single-hop view is
the best or joint-best on four of six. `T_MAX_VIEW` — the multi-state union the directive asked us
not to collapse prematurely — is best on exactly one corpus at @50 (musique 0.885) and is *worse*
than `T_SINGLE` on three. Keeping the states apart therefore buys nothing that a single state does
not already give: **the multi-state hypothesis is measured and negative.**

---

## PART B — TRANSFORMATION EVICTION AUDIT (explanation B: "we need new mathematical information to decide safe evictions")

### B4 — the forced-swap margin, and the universality bar the directive set

Forced pairs are the labelled cross product of *evicted SAFE candidate* × *admitted SRC candidate*
at fixed `K(q)`. `GOOD_SWAP` = the admitted one is required and the evicted one is not;
`BAD_SWAP` = the reverse. Labels are used **for evaluation only** — nothing is trained on them.

| corpus | GOOD_SWAP | BAD_SWAP | unlabelled (tie) |
|---|---|---|---|
| metaqa | 27,120 | 21,940 | 1,741,882 |
| webqsp | 2,826 | 2,597 | 768,499 |
| 2wiki_clean | 268 | 560 | 1,003,572 |
| musique_clean | 65 | 417 | 185,464 |
| hotpotqa_clean | 384 | 2,116 | 2,911,384 |
| **squad_clean** | **0** | 300 | 183,518 |

**squad_clean has no GOOD_SWAP pairs at all.** Every labelled forced swap on squad is a swap that
loses a required partition. There is nothing there for any eviction signal to win — which is itself
an answer to the phase's question on that corpus. The universality test below therefore runs on the
five corpora where both classes exist, which is the most favourable reading available.

`AUC` of the swap margin `transform_score(admitted) - transform_score(evicted)`, same score key on
every corpus, best five of 33 keys by worst-corpus AUC:

```
key                                        metaqa    webqsp     2wiki   musique    hotpot     min  consistent
r_q/F1_HOUSEHOLDER/V4_SRC_EXIT             0.5683    0.5475    0.6608    0.5763    0.4925  0.4925     no
q_raw/F2_MIN_ROTATION/V4_SRC_EXIT          0.6441    0.6040    0.6618    0.5072    0.4883  0.4883     no
q_raw/F1_HOUSEHOLDER/V4_SRC_EXIT           0.6429    0.6056    0.6620    0.5175    0.4882  0.4882     no
q_raw/F3_RANK1_TRANSPORT/V4_SRC_EXIT       0.6207    0.6467    0.6269    0.4913    0.4822  0.4822     no
q_raw/F3_RANK1_TRANSPORT/V1_SINGLE         0.6226    0.5287    0.4792    0.5102    0.5128  0.4792     no
V0_NOTRANSFORM  (identity control)         0.5898    0.4504    0.4755    0.5161    0.5560  0.4504     no
```

**`GOOD_BAD_SWAP_SIGN_CONSISTENT = NO`. Zero of 33 score keys point the same way on all five
corpora.** The four best keys are all the `V4_SRC_EXIT` view and all of them look strong on
metaqa / webqsp / 2wiki (AUC 0.60–0.66) and then land **below chance on hotpotqa** (0.482–0.493).
The directive's own bar — *"a signal that flips sign by corpus is NOT usable"* — is not met, so the
B5 gate (*"fixed-K admission ONLY if universal separation shown"*) does not open.

**The honest subset makes the same point differently.** Restricting to pairs where **both**
candidates actually carry transformation evidence (outside that subset the margin is decided by the
`FLOOR`, i.e. by *presence*, not by geometry) five keys do become sign-consistent — and the identity
control is one of them, at values three transformation keys reproduce **exactly**:

```
key (BOTH-evidence pairs only)             metaqa    webqsp     2wiki   musique    hotpot     min  consistent
r_q/F1_HOUSEHOLDER/V4_SRC_EXIT             0.5552    0.5465    0.8049    0.5168    0.5342  0.5168    YES
r_q/F2_MIN_ROTATION/V3_MAXVIEW             0.5821    0.6983    0.6448    0.5081    0.8638  0.5081    YES
r_q/F1_HOUSEHOLDER/V3_MAXVIEW              0.5821    0.6983    0.6448    0.5081    0.8638  0.5081    YES
V0_NOTRANSFORM  (identity control)         0.5821    0.6983    0.6448    0.5081    0.8638  0.5081    YES
q_raw/F3_RANK1_TRANSPORT/V1_SINGLE         0.6199    0.8652    0.6791    0.5025    0.8032  0.5025    YES
```

The two `V3_MAXVIEW` rows are numerically identical to `V0_NOTRANSFORM` to four decimals on all five
corpora, because `V3_MAXVIEW = max(V0, chain)` and the composed term never wins the max. So on the
only population where the margin is a genuine comparison of geometry, **the thing that separates
good evictions from bad ones is plain retrieval compatibility with the target node — with no
transformation applied at all.**

### B4 by hop — where the separation is large, and what it is worth

On MetaQA the swap margin separates GOOD from BAD *strongly* on the hop-2 slice, and the
transformation views are far above the identity control there:

| score (MetaQA, A1_SRC pairs) | hop2 AUC (5,399 / 2,908) | hop3 AUC (21,721 / 18,998) |
|---|---|---|
| `V0_NOTRANSFORM` (identity control) | 0.5975 | 0.5917 |
| `q_raw/F2_MIN_ROTATION/V4_SRC_EXIT` | **0.8321** | 0.6119 |
| `q_raw/F1_HOUSEHOLDER/V4_SRC_EXIT` | 0.8202 | 0.6128 |
| `q_raw/F2_MIN_ROTATION/V1_SINGLE` | 0.8204 | 0.6065 |
| `q_raw/F2_MIN_ROTATION/V2_CHAIN` | 0.8099 | **0.6701** |
| `q_raw/F0_TRANSLATION/V4_SRC_EXIT` | 0.7482 | 0.6025 |
| `q_raw/F3_RANK1_TRANSPORT/V4_SRC_EXIT` | 0.7373 | 0.6029 |

`hop1` has **zero** GOOD pairs on MetaQA (0 / 34): at one hop there is nothing to repair.

This is the sharpest quantitative statement of the pattern this program keeps hitting. On MetaQA
hop2 the transformation margin separates good evictions from bad ones at **AUC 0.832 against an
identity-control 0.598** — a 0.23 gap over 8,307 labelled pairs — and that separation converts into
**+8 queries out of 666 versus the control's +6**. A large, real, significant separation on the
pair population is worth two queries at exact P50. Separability is not the binding constraint.

### B5 / STEPS 9-10 — fixed-K replacement at exact P50, and what actually causes the one gain

Because F6 is algebraically closed to *appended* candidates, the only route by which transformation
evidence can reach the output is **replacement** at the frozen depth:

```
universe = SAFE pool ++ SRC-assignment candidates not already in it
pool     = top K(q) of the universe ordered by (-transform_score, frozen_safe_rank, pid)
```

`|pool|` is exactly `K(q)`, the ordering is total and deterministic, nothing is learned or
thresholded, and the output is still exactly `P = 50` with `B = 6`. Graph parity is exact
(1998/1998, 2000/2000).

Three scores are run side by side on the same pools — the winning transformation cell, and **two
controls that use no transformation at all**:

- `V0_NOTRANSFORM(p) = max over visited edges into p of cos(q, x_v)` — plain target compatibility;
- `V6_EXIT_SRC_SIM(p) = max over **core-exit** edges into p of cos(q, x_u)` — plain *source*
  compatibility, i.e. pure provenance: how well the query matches the protected-core node that
  points at `p`. This is `V4_SRC_EXIT` with the geometry deleted and only the restriction kept.

**MetaQA (n=1998, the primary corpus):**

| slice | frozen | `F2_MIN_ROTATION/V4_SRC_EXIT` | `V6_EXIT_SRC_SIM` (no transform) | `V0_NOTRANSFORM` |
|---|---|---|---|---|
| ALL / F6 | 0.6612 | **0.6672** +12 (+13/−1) *p*=.0018 **SIG** | 0.6662 +10 (+13/−3) *p*=.0213 **SIG** | 0.6657 +9 (+13/−4) *p*=.0490 **SIG** |
| ALL / G4 | 0.6612 | 0.6667 +11 *p*=.0989 | **0.6672** +12 *p*=.0807 | 0.6642 +6 *p*=.3915 |
| hop2 / F6 | 0.7297 | **0.7417** +8 (+8/−0) *p*=.0078 **SIG** | 0.7387 +6 (+9/−3) *p*=.1460 | 0.7387 +6 (+8/−2) *p*=.1094 |
| hop3 / F6 | 0.2583 | 0.2643 +4 *p*=.2188 | 0.2643 +4 (+4/−0) *p*=.1250 | 0.2628 +3 *p*=.4531 |
| hop3 / G4 | 0.2583 | 0.2778 +13 (+16/−3) *p*=.0044 **SIG** | **0.2823 +16 (+19/−3) *p*=.0009 SIG** | 0.2748 +11 *p*=.0192 **SIG** |
| hop1 / both | 0.9955 | 0.9955 ±0 | 0.9955 ±0 | 0.9955 ±0 |

The MetaQA gain is real and reproducible — and **the no-transformation provenance control matches it
everywhere and beats it on the largest slice** (hop3/G4: +16 vs +13). Every cell separating the two
is 2–3 queries out of 666.

**The attribution is not a coincidence of close numbers.** `diag/attrib_*.json` measures how
different the two scores actually are, on the population that decides the pool (1-in-11 query
sample, ~180 queries per corpus, same replay and same core-exit edge set as the swap runs):

| | metaqa | 2wiki_clean | musique_clean |
|---|---|---|---|
| core-exit edges at hop 1 / 2 / 3 | 3.2% / 58.2% / 38.7% | 3.7% / 53.9% / 42.4% | 2.1% / 55.6% / 42.4% |
| the per-partition **argmax edge** sits at hop 1 | 1.9% | 1.8% | 1.3% |
| `V4` and `V6` pick the **same argmax edge** | 63.9% | 74.3% | 35.6% |
| Spearman(`V4`, `V6`) over partitions, mean / min | **0.267** / −0.456 | **0.207** / −0.600 | **0.395** / −0.489 |
| queries where the two produce the **identical pool** | **0 / 182** | 16 / 174 (9.2%) | 2 / 179 (1.1%) |

Only ~2-4% of core-exit edges sit at hop 1, and the per-partition argmax edge sits at hop 1 under
2% of the time, so the orthogonal-degeneracy identity `cos(T z, v) = cos(z, u)` — which is exact
only at hop 1 — does **not** equate the two. They are genuinely different scores, correlated at
ρ ≈ 0.21–0.40, disagreeing on the argmax edge 26–64% of the time, and agreeing on the pool in
**0 of 182** MetaQA queries. **Two scores that never produce the same candidate pool produce the
same exact-P50 outcome to within 2–3 queries on every slice.** On 2wiki the two land on *identical*
aggregate outcomes in all four cells (−5/−5 F6, −7/−7 G4, same gained and lost counts) despite
disagreeing on 91% of pools.

On **webqsp** — the one corpus that preferred transformations at every STEP-3 setting, and whose
STEP-7 admission recall @6 under translation (0.1676) is 2.4x `SRC_ASSIGN` (0.0695) — the same
replacement converts **+1 (ns)** at exact P50, with the no-transformation provenance control again
slightly ahead (+2, ns). The KB regime is not an exception; it is one more instance of separation
that does not convert.

The conclusion is forced: **what converts is the core-exit restriction — which partitions receive
any evidence at all — not the geometry that ranks them.** The transformation algebra contributes
nothing measurable on top of a plain query-to-source cosine. This is the same shape as the
read-stage audit's finding (separability real, correlation with exact-P50 nil), now reproduced at
the eviction stage.

---

## STEP 12 / A3 — cost

All transport is closed-form and matrix-free. `cos(T(z), x_v)`, `‖T(z)‖` and `⟨T(z), z0⟩` each close
in `O(1)` per edge from seven scalars, so only the ~3% of edges that are a parent of another edge
ever materialise a `d`-dimensional vector (3.58 s/q → 226 ms/q when that was put in). Even so:

| corpus | edges/q | 1 family × 1 state (ms/q) | all 4 families × 2 states (ms/q) | frozen admission stage (ms/q) |
|---|---|---|---|---|
| metaqa | 1,386 | 7.97 | 127.6 | 0.0201 |
| webqsp | 1,506 | 9.09 | 145.4 | 0.0151 |
| 2wiki_clean | 547 | 2.90 | 46.4 | 0.0156 |
| musique_clean | 3,251 | 15.09 | 241.4 | 0.0110 |
| hotpotqa_clean | 1,389 | 8.62 | 138.0 | 0.0154 |
| squad_clean | 14,123 | 67.44 | 1,079.0 | 0.0045 |

**The cheapest single operator costs 190× the entire admission stage it would sit in; the full
algebra costs 2,000–240,000×.** The micro-K ladder by contrast is nearly free — appending 32
candidates costs +0.005 to +0.024 ms/q — which is exactly why PART A's failure is a *conversion*
failure and not a cost failure. `A3`'s gain/cost knee does not exist because the gain is zero, not
because the cost is high; the cost matters only for the transformation branch, where it disqualifies
even the corpus-local MetaQA win on its own terms.

### B6 / STEP 10 — exact P50, all six corpora

`ALL` slice, `acc` and net vs the frozen system; `*` = significant (McNemar).

| corpus | frozen | transform F6 | transform G4 | `V6` provenance F6 | `V6` provenance G4 | `V0` identity F6 | `V0` identity G4 | ms/q |
|---|---|---|---|---|---|---|---|---|
| metaqa | 0.6612 | **0.6672 +12\*** | 0.6667 +11 | 0.6662 +10\* | 0.6672 +12 | 0.6657 +9\* | 0.6642 +6 | 116.6 |
| webqsp | 0.7646 | 0.7653 +1 | 0.7625 −3 | 0.7660 +2 | 0.7639 −1 | 0.7639 −1 | 0.7632 −2 | 72.9 |
| 2wiki_clean | 0.9435 | 0.9410 −5 | 0.9400 −7 | 0.9410 −5 | 0.9400 −7 | 0.9405 −6 | 0.9410 −5 | 27.7 |
| musique_clean | 0.9635 | **0.9560 −15\*** | **0.9545 −18\*** | 0.9585 −10\* | 0.9565 −14\* | 0.9605 −6 | 0.9590 −9\* | 118.9 |
| hotpotqa_clean | 0.9505 | **0.9410 −19\*** | **0.9345 −32\*** | 0.9405 −20\* | 0.9345 −32\* | 0.9420 −17\* | 0.9340 −33\* | 66.1 |
| squad_clean | 0.9875 | **0.9810 −13\*** | **0.9815 −12\*** | — | — | 0.9835 −8\* | 0.9840 −7\* | 352.9 |

**One gain, one neutral, four losses — three of them significant, on both selectors.**
`TRANSFORM_CROSS_CORPUS_SAFE = NO`. Two further facts settle the attribution for good:

1. **Where the replacement hurts, the transformation makes it hurt *more* than the plain provenance
   control** (musique −15 vs −10, squad −13 vs −8, hotpot −19 vs −20/−17 — a wash), while where it
   helps it delivers nothing the control does not. The transformation is a pure amplifier of the
   damage and a non-contributor to the gain.
2. On hotpot all three scores land within 3 queries of each other on F6 (−19 / −20 / −17) and within
   1 on G4 (−32 / −32 / −33), on 2,000 queries. Whatever the replacement is doing, it is not
   sensitive to the score.


---

## What this phase establishes

**1. The frozen selector is closed to appended candidates — provably.** `F6`'s score for a partition
with neither struct nor ret evidence is bounded above by `1/110`, and every boundary member it would
have to displace is bounded below by `1/109`. That is not a measurement that could come out
differently on another corpus; it is the definition of the SAFE pool colliding with the definition of
the score. **Every "widen K / append candidates / go deeper" experiment against this contract is dead
before it runs.** The only live operation is replacement at fixed depth.

**2. Norm-preserving transformations cannot carry single-hop information.** `cos(T(z), v) = cos(z, u)`
exactly, for any orthogonal `T` with `T(u) = v`. Householder and minimal rotation are different maps
(elementwise difference 0.692) with identical target compatibility to 3.4e-08 on all six corpora. A
fitted group-level orthogonal operator — STEP 11 — would collapse to the same scalar, so that branch
is closed algebraically rather than empirically.

**3. Where the transformation branch looked alive, it was the incumbent in disguise.** webqsp is the
only corpus that prefers transformations at every setting, and its best admission cell is
`F0_TRANSLATION` — the very operator this directive set out to replace, in the KB regime where the
offset channel was already known to be strong. That is not evidence for a new algebra.

**4. The one real gain is a provenance effect, and it is not universal.** Five structurally different
scores — a composed min-rotation over core-exit edges, a Householder chain, a translation chain, a
pure query-to-source cosine, and a pure query-to-target cosine — produce MetaQA `ALL/F6` nets of
**+12, +9, +7, +10, +9** and `hop3/G4` nets of **+13, +12, +11, +16, +11**. The spread across scores
that disagree on the candidate pool in ~100% of queries is smaller than the effect itself. The effect
is "replace weak pool members using *some* evidence over already-visited targets"; which evidence is
close to irrelevant. And the same operation is **significantly harmful** on musique and squad, and
negative on 2wiki.

**5. Separation is not the binding constraint, again.** MetaQA hop2: swap-margin AUC **0.832** for the
transformation against **0.598** for the identity control, over 8,307 labelled pairs — worth **+8
queries versus +6** at exact P50. This is the third stage of the pipeline at which this program has
measured large, real, significant separability that converts to nothing (see the read-stage audit and
the calibration/dilution phase).

## What this phase does not close

Explanation **B** was *directionally* right and should not be recorded as refuted in spirit: evictions
**are** repairable, and the repair is worth a significant gain on MetaQA (`+12` ALL, `+8` hop2, up to
`+16` hop3). What is refuted is the specific hypothesis that the missing ingredient is **new
mathematical information**. The information that converts is ordinary retrieval evidence the substrate
already holds; what is missing is an **inference-safe rule that knows when to apply it**, and every
statistic tested here flips sign across corpora. That is the same wall as
`l1-admission-eviction-not-admission` and `l1-exposure-audit-partial`, reached from a new direction.

`PROMOTED = NONE`. `L1_FROZEN = NO`. No TEST was run, no L2 or L3 work was touched, `P` stayed at 50,
`B` stayed at 6, `K` never exceeded `K0+32`, no graph traversal was added, and nothing was learned,
fitted, thresholded or switched on dataset identity.
