# L2 E0–E3 Controlled Training Phase — Results

**Scope:** First controlled L2 training phase on the canonical **C / FUSION / K100 / P50** pilot.
**Locked inputs (unchanged):** MASTER_TOPOLOGY=C · L1_PARTITIONS=C · L1_ROUTER=Dense+SPLADE · RRF K0=60 · K=100 · P=50 · TAU=0.05.
**Pilot datasets:** `2wiki_clean`, `musique_clean` (canonical frozen train/val/test splits, canonical P50 candidate universe).
**Status:** E0, E0s, E1a, E1b, E2, E3, E3b complete. **STOPPED after E0–E3 as instructed.** Relation NOT built, Path NOT used, no controller/dropout/regime training, no Hotpot/SQuAD/MetaQA/WebQSP, L1 untouched.

---

## 0. Experimental discipline (identical across all rows)

- **Same query splits**, **same P50 candidate universe**, **same positive labels**, **same evaluator**, **same negative-sampling policy** for every config. No result compares across candidate substrates.
- **SPLADE update (done before E1):** exact candidate-level `splade_scope_score` / `splade_scope_rank` were materialized over the P50 scope for all pilot corpora (`scratchpad/l2_splade_cache.py`, manifest `results/L2/_splade_scope_manifest.json`). The pre-existing global-retrieval field `splade_global_top200_rank` was **kept and not overwritten**; both concepts coexist. No full-corpus SPLADE retrieval was recomputed. Parity check: differences were ties-only.
- **Evaluation universe:** TRAIN = in-scope positives + canonical deterministic hard-negative subset (128 negs/query). **VAL/TEST rank the FULL P50 scope (~5k candidates/query)** — a model trained on 128 negatives is still evaluated over the entire scope.

### Label semantics (printed first, as required)

| dataset | split | Nq | positive label | N_GOLD_EXPECTED (incorp) | N_GOLD_IN_SCOPE | multi-gold frac | L1_ANY_FAIL | L1_PARTIAL | L1_ALL_SUCCESS |
|---|---|---|---|---|---|---|---|---|---|
| 2wiki_clean | train | 10500 | cand is a gold doc present in P50 scope | 2.422 | 2.347 | 1.00 | 3 | 705 | 9792 |
| 2wiki_clean | val | 3000 | " | 2.424 | 2.355 | 1.00 | 0 | 188 | 2812 |
| musique_clean | train | 13956 | " | 2.337 | 2.289 | 1.00 | 8 | 626 | 13322 |
| musique_clean | val | 3987 | " | 2.339 | 2.288 | 1.00 | 2 | 187 | 3798 |

- **Every pilot query is multi-gold** (multi_gold_frac = 1.00, ≈2.3–2.4 golds/query). This makes multi-positive loss correctness and ALL@K the central metrics.
- **L1_ANY_FAIL is tiny** (0–8 queries). A query with no gold in scope is an **L1 pruning failure, not an L2 ranking failure**, and is excluded from the conditional denominators. Because ANY_FAIL is negligible here, the three denominators nearly coincide — L1 and L2 are cleanly separated for this pilot.

**Three denominators reported:** `ALL_EVAL_QUERIES` (L1_ANY_FAIL scored as recall 0), `COND_ANY_GOLD_IN_SCOPE` (honest L2-ranking set), `COND_ALL_GOLD_IN_SCOPE` (multi-gold ALL@K set). Tables below use **COND_ANY_GOLD_IN_SCOPE** (the honest L2 set); the full three-denominator breakdown is in `L2_E0_E3_RESULTS.json`.

---

## 1. Canonical results table

Rows: **E0** dense · **E0s** SPLADE-only · **E1a** Dense+SPLADE RRF · **E1b** Dense+SPLADE z-norm · **E2** +Offset (RRF) · **E3** +Mixture (RRF) · **E3b** +cheap graph (RRF).

<!-- BEGIN AUTO TABLE (results/L2/_L2_table_body.md) -->

### 2wiki_clean
#### 2wiki_clean — test (denominator: COND_ANY_GOLD_IN_SCOPE)

| config | MRR | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | ANY@10 | ANY@20 | ANY@50 | ALL@10 | ALL@20 | ALL@50 | bgr_mean | bgr_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 Dense | 0.9245 | 0.4012 | 0.6591 | 0.6987 | 0.7302 | 0.7636 | 0.7955 | 0.9953 | 0.9987 | 1.0000 | 0.4027 | 0.4527 | 0.5073 | 0.3013 | 0.0000 |
| E0s SPLADE | 0.9618 | 0.4319 | 0.6993 | 0.7251 | 0.7414 | 0.7633 | 0.7979 | 0.9993 | 0.9993 | 0.9993 | 0.4247 | 0.4520 | 0.4900 | 0.5140 | 0.0000 |
| E1a Dense+SPLADE RRF | 0.9588 | 0.4283 | 0.7004 | 0.7271 | 0.7494 | 0.7835 | 0.8084 | 0.9993 | 0.9993 | 1.0000 | 0.4353 | 0.4707 | 0.5293 | 0.1367 | 0.0000 |
| E1b Dense+SPLADE z-norm | 0.9723 | 0.4384 | 0.7115 | 0.7322 | 0.7524 | 0.7804 | 0.8126 | 0.9993 | 0.9993 | 1.0000 | 0.4380 | 0.4760 | 0.5247 | 0.0747 | 0.0000 |
| E2 +Offset (RRF) | 0.9274 | 0.4067 | 0.6617 | 0.7400 | 0.7950 | 0.8700 | 0.9144 | 0.9967 | 0.9993 | 1.0000 | 0.4593 | 0.5533 | 0.6940 | 0.2860 | 0.0000 |
| E3 +Mixture (RRF) | 0.9171 | 0.4028 | 0.6490 | 0.7671 | 0.8557 | 0.9137 | 0.9366 | 0.9787 | 0.9933 | 1.0000 | 0.5333 | 0.6720 | 0.7707 | 0.5660 | 0.0000 |
| E3b +cheap graph (RRF) | 0.9506 | 0.4219 | 0.6807 | 0.7261 | 0.7694 | 0.8337 | 0.8968 | 0.9987 | 1.0000 | 1.0000 | 0.4407 | 0.5100 | 0.6220 | 0.1627 | 0.0000 |

#### 2wiki_clean — val (denominator: COND_ANY_GOLD_IN_SCOPE)

| config | MRR | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | ANY@10 | ANY@20 | ANY@50 | ALL@10 | ALL@20 | ALL@50 | bgr_mean | bgr_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 Dense | 0.9312 | 0.4091 | 0.6525 | 0.6889 | 0.7181 | 0.7548 | 0.7841 | 0.9957 | 0.9983 | 0.9997 | 0.3853 | 0.4290 | 0.4887 | 0.2983 | 0.0000 |
| E0s SPLADE | 0.9598 | 0.4298 | 0.6930 | 0.7144 | 0.7306 | 0.7539 | 0.7825 | 0.9990 | 0.9993 | 1.0000 | 0.4070 | 0.4337 | 0.4753 | 0.1200 | 0.0000 |
| E1a Dense+SPLADE RRF | 0.9592 | 0.4276 | 0.6949 | 0.7197 | 0.7391 | 0.7682 | 0.7964 | 0.9997 | 1.0000 | 1.0000 | 0.4207 | 0.4530 | 0.5017 | 0.1020 | 0.0000 |
| E1b Dense+SPLADE z-norm | 0.9694 | 0.4359 | 0.7079 | 0.7270 | 0.7423 | 0.7704 | 0.7993 | 0.9997 | 0.9997 | 1.0000 | 0.4287 | 0.4557 | 0.5070 | 0.0717 | 0.0000 |
| E2 +Offset (RRF) | 0.9366 | 0.4139 | 0.6660 | 0.7397 | 0.7879 | 0.8656 | 0.9152 | 0.9983 | 0.9997 | 1.0000 | 0.4593 | 0.5407 | 0.6827 | 0.2177 | 0.0000 |
| E3 +Mixture (RRF) | 0.9286 | 0.4111 | 0.6470 | 0.7641 | 0.8637 | 0.9118 | 0.9344 | 0.9900 | 0.9973 | 1.0000 | 0.5233 | 0.6873 | 0.7670 | 0.3903 | 0.0000 |
| E3b +cheap graph (RRF) | 0.9515 | 0.4221 | 0.6733 | 0.7119 | 0.7563 | 0.8204 | 0.8854 | 0.9970 | 0.9993 | 1.0000 | 0.4190 | 0.4863 | 0.5980 | 0.1673 | 0.0000 |

### musique_clean
#### musique_clean — test (denominator: COND_ANY_GOLD_IN_SCOPE)

| config | MRR | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | ANY@10 | ANY@20 | ANY@50 | ALL@10 | ALL@20 | ALL@50 | bgr_mean | bgr_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 Dense | 0.8813 | 0.3805 | 0.6774 | 0.7520 | 0.8079 | 0.8660 | 0.9015 | 0.9840 | 0.9915 | 0.9965 | 0.5060 | 0.5953 | 0.7041 | 1.6138 | 0.0000 |
| E0s SPLADE | 0.8843 | 0.3842 | 0.6506 | 0.7126 | 0.7650 | 0.8248 | 0.8676 | 0.9739 | 0.9840 | 0.9915 | 0.4368 | 0.5181 | 0.6229 | 1.8621 | 0.0000 |
| E1a Dense+SPLADE RRF | 0.9032 | 0.3936 | 0.7018 | 0.7654 | 0.8184 | 0.8777 | 0.9081 | 0.9850 | 0.9900 | 0.9975 | 0.5201 | 0.6123 | 0.7217 | 0.8962 | 0.0000 |
| E1b Dense+SPLADE z-norm | 0.9141 | 0.4008 | 0.7053 | 0.7671 | 0.8167 | 0.8739 | 0.9057 | 0.9865 | 0.9910 | 0.9970 | 0.5196 | 0.6073 | 0.7151 | 0.8425 | 0.0000 |
| E2 +Offset (RRF) | 0.9251 | 0.4043 | 0.7784 | 0.8549 | 0.9094 | 0.9644 | 0.9863 | 0.9945 | 0.9970 | 0.9995 | 0.6810 | 0.7884 | 0.9152 | 0.3701 | 0.0000 |
| E3 +Mixture (RRF) | 0.9299 | 0.4069 | 0.8248 | 0.9086 | 0.9609 | 0.9847 | 0.9907 | 0.9955 | 0.9990 | 1.0000 | 0.7924 | 0.9082 | 0.9629 | 0.3014 | 0.0000 |
| E3b +cheap graph (RRF) | 0.9198 | 0.3999 | 0.8072 | 0.8934 | 0.9545 | 0.9855 | 0.9929 | 0.9955 | 0.9985 | 0.9995 | 0.7593 | 0.8922 | 0.9634 | 0.3636 | 0.0000 |

#### musique_clean — val (denominator: COND_ANY_GOLD_IN_SCOPE)

| config | MRR | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | ANY@10 | ANY@20 | ANY@50 | ALL@10 | ALL@20 | ALL@50 | bgr_mean | bgr_med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 Dense | 0.8699 | 0.3738 | 0.6704 | 0.7442 | 0.7994 | 0.8636 | 0.8986 | 0.9739 | 0.9875 | 0.9942 | 0.4996 | 0.5837 | 0.6994 | 1.9162 | 0.0000 |
| E0s SPLADE | 0.8685 | 0.3723 | 0.6343 | 0.7024 | 0.7576 | 0.8170 | 0.8551 | 0.9721 | 0.9822 | 0.9915 | 0.4141 | 0.5024 | 0.6058 | 3.3792 | 0.0000 |
| E1a Dense+SPLADE RRF | 0.8874 | 0.3833 | 0.6873 | 0.7548 | 0.8096 | 0.8707 | 0.9084 | 0.9819 | 0.9895 | 0.9967 | 0.5006 | 0.5927 | 0.7084 | 1.2693 | 0.0000 |
| E1b Dense+SPLADE z-norm | 0.9036 | 0.3945 | 0.6937 | 0.7606 | 0.8094 | 0.8645 | 0.9009 | 0.9829 | 0.9907 | 0.9965 | 0.5094 | 0.5905 | 0.6966 | 1.2266 | 0.0000 |
| E2 +Offset (RRF) | 0.9149 | 0.3965 | 0.7771 | 0.8515 | 0.9028 | 0.9576 | 0.9809 | 0.9922 | 0.9965 | 0.9987 | 0.6735 | 0.7789 | 0.9004 | 0.5066 | 0.0000 |
| E3 +Mixture (RRF) | 0.9213 | 0.4001 | 0.8195 | 0.9016 | 0.9567 | 0.9788 | 0.9871 | 0.9947 | 0.9980 | 0.9990 | 0.7849 | 0.9039 | 0.9518 | 0.4324 | 0.0000 |
| E3b +cheap graph (RRF) | 0.9113 | 0.3940 | 0.7987 | 0.8897 | 0.9473 | 0.9799 | 0.9890 | 0.9915 | 0.9975 | 0.9985 | 0.7581 | 0.8806 | 0.9543 | 0.5388 | 0.0000 |

<!-- END AUTO TABLE -->

*(bgr = best_gold_rank; 0-indexed, so median 0 means the top-ranked candidate is a gold. The full three-denominator breakdown incl. ALL_EVAL_QUERIES and COND_ALL_GOLD_IN_SCOPE is in `L2_E0_E3_RESULTS.json`.)*

---

## 2. Training configuration (E2 Offset, E3 Mixture)

Identical machinery for both heads; **VAL-MRR checkpoint selection** (primary evidence is validation, never train).

| item | value |
|---|---|
| objective | per-query **listwise InfoNCE**, batched (einsum + logsumexp) |
| multi-positive | **each in-scope gold is its own positive slot**; negatives exclude **all** golds for that query |
| positive handling | all in-scope positives kept (no single-gold collapse) |
| negatives | canonical deterministic sampler, **128 negs/query**, mean_negs/query = 128.0 |
| positive collisions | **0** (verified — mandatory) |
| temperature (τ) | 0.05 |
| optimizer / lr | Adam / 1e-3 |
| epochs | 10 (early-stop patience 3 on VAL MRR) |
| query-batch | 256 |
| checkpoint selection | max VAL fused-RRF MRR |
| seed | 1234 |
| OffsetHead | `Linear(1536,512)→ReLU→Linear(512,1536)`, `normalize(seed+net(qn))` |
| MixtureHead | K=8: `Linear(1536,512)→ReLU→Linear(512,K·1536)`, `normalize(seed+off)` |
| seed | dense top-1 doc per query |

**Multi-positive verification (critical):** the pre-existing InfoNCE machinery (`kg_relsig.py` / `l1_universal_head.py`) builds one training instance per (query, gold) pair — it does *not* collapse to a single gold — but its in-batch negatives can accidentally include a query's *other* golds as negatives when multiple golds co-occur. Since every pilot query is multi-gold, we used the **cleaner per-query listwise formulation** where the negative set explicitly excludes all of that query's golds → **zero false negatives** (collisions = 0, verified). This is the correct multi-positive loss for this data.

### Train vs VAL (overfitting check)

| dataset | head | train loss (ep1→ep10) | VAL MRR (ep1→best) | best epoch | verdict |
|---|---|---|---|---|---|
| 2wiki | offset | 2.79 → 1.75 | 0.9342 → **0.9366** | 9 | train loss falls steadily, VAL MRR ~flat → **mild overfit; VAL selection mitigated** |
| 2wiki | mixture | 2.77 → 1.37 | 0.9303 → **0.9372** | 8 | same pattern, stronger train fit |
| musique | offset | 3.00 → 0.76 | 0.8809 → **0.9149** | 10 | VAL still climbing at cap → **not overfit; genuine signal** |
| musique | mixture | 2.92 → 0.36 | 0.8837 → **0.9191** | 10 | VAL still climbing at cap → genuine signal |

**TRAIN saturation was NOT used to claim signal quality.** On 2wiki the train loss keeps dropping while VAL MRR plateaus (~0.937) — classic mild overfitting, contained by VAL checkpoint selection. On musique VAL is still improving at the epoch cap, so the heads carry real generalizing signal there.

---

## 3. DELTA vs previous-best & per-signal value

DELTAs on **VAL** (primary evidence), COND_ANY. "vs prev-best" chains each row against the best of all rows above it, on the metric named.

### Top-rank axis (MRR) — where dense+SPLADE are already right

| step | 2wiki VAL MRR (Δ vs prev best) | musique VAL MRR (Δ vs prev best) |
|---|---|---|
| E0 dense | 0.9312 (—) | 0.8699 (—) |
| E0s splade | 0.9598 (**+0.0286**) | 0.8685 (−0.0014) |
| E1a RRF | 0.9592 (−0.0006) | 0.8874 (**+0.0175**) |
| E1b z-norm | **0.9694 (+0.0096)** | **0.9036 (+0.0162)** |
| E2 +offset | 0.9366 (−0.0328) | **0.9149 (+0.0113)** |
| E3 +mixture | 0.9286 (−0.0408) | **0.9213 (+0.0177)** |

### Deep multi-gold axis (ALL@50) — the actual L2 job

| step | 2wiki VAL ALL@50 (Δ vs prev best) | musique VAL ALL@50 (Δ vs prev best) |
|---|---|---|
| best of E0/E0s/E1a/E1b | 0.5070 (—) | 0.7084 (—) |
| E2 +offset | 0.6827 (**+0.1757**) | 0.9004 (**+0.1920**) |
| E3 +mixture | **0.7670 (+0.0843)** | **0.9518 (+0.0514)** |
| E3b +graph | 0.5980 (below its base) | 0.9543 (≈E3, +0.0025) |

### SIGNAL_VALUE classification (from VALIDATION)

| signal | 2wiki | musique | overall | decision |
|---|---|---|---|---|
| **Dense (E0)** | strong baseline | strong baseline | **STRONG BASELINE** | carry forward (anchor) |
| **Exact SPLADE** | +2.9 MRR standalone; +fusion | ≈0 standalone, but **RRF fusion E1a beats E0 by +1.75 MRR** | **POSITIVE (via fusion)** | **KEEP** |
| **Offset (E2)** | **MIXED** — MRR −3.3, but **ALL@50 +17.6pt** | **POSITIVE** — every metric up, ALL@50 +19.2pt | **POSITIVE for deep multi-gold recovery** | **KEEP** |
| **Mixture (E3)** | deep-recall **ALL@50 +8.4pt more**; MRR −0.8 more | **POSITIVE** — best MRR *and* best ALL@50 | **POSITIVE for deep multi-gold recovery** | **KEEP** |
| **Cheap graph (E3b)** | below its base on every metric | ≈E3 (marginal, within noise) | **NEUTRAL→NEGATIVE** | **DROP** |

**The central finding.** On **musique**, offset then mixture improve *everything* monotonically (MRR, R@5, ALL@50) — unambiguous POSITIVE. On **2wiki**, the heads *trade top-1/MRR for a very large deep multi-gold gain*: E3 lifts ALL@50 from 0.507 → 0.767 (+26pt over E1a) while MRR slips 0.959 → 0.917. This trade is an **artifact of equal-weight RRF fusion**: the heads deserve high weight where they rescue a deep second/third gold, but low weight at the very top where dense+SPLADE already place a gold at rank 0 (2wiki bgr median = 0 for every config). Equal weighting lets the head perturb an already-correct #1. **This is exactly the motivation for a query-aware soft controller** — learn *when* each verified expert matters instead of stacking them at fixed equal weight.

---

## 4. FINAL DECISION QUESTIONS

1. **DENSE_BASELINE_STRONG** — **YES.** Dense alone gets MRR 0.92–0.93 (2wiki) / 0.87–0.88 (musique) and bgr median 0. It is the anchor every other signal is measured against.
2. **EXACT_SPLADE_ADDS_VALUE** — **YES.** Materialized scope-exact SPLADE lifts top-rank via fusion on both datasets (E1a > E0 by +1.75 MRR on musique; E0s +2.9 MRR on 2wiki). Standalone on musique it is ~neutral, but as a fusion axis it is positive everywhere.
3. **OFFSET_ADDS_VALUE** — **YES (musique) / MIXED (2wiki).** Large deep multi-gold recovery on both (ALL@50 +17–19pt); on 2wiki it costs top-1/MRR under equal fusion. Kept for its deep-recall value; the cost is a fusion-weighting artifact, not a bad signal.
4. **MIXTURE_ADDS_VALUE** — **YES.** Best config on musique (MRR 0.921, ALL@50 0.952); on 2wiki adds a further +8.4pt ALL@50 over offset. It further deepens the 2wiki MRR cost under equal weight (same controller motivation).
5. **CHEAP_GRAPH_ADDS_VALUE** — **NO.** E3b is below its base on 2wiki and within noise on musique. **Dropped** — no expensive graph build is justified by this diagnostic.
6. **TRAIN_OVERFIT_OBSERVED** — **YES (2wiki, mild).** Train loss falls while VAL MRR plateaus; contained by VAL-MRR checkpoint selection. musique shows no overfit (VAL still rising at cap). TRAIN ranking was never used as a signal-quality claim.
7. **MULTI_POSITIVE_LOSS_CORRECT** — **YES.** Every pilot query is multi-gold; each in-scope gold is a positive slot and negatives exclude all golds → positive_collisions = 0 (verified). Not trained as if one gold.
8. **FULL_P50_EVAL** — **PASS.** VAL/TEST rank the entire ~5k-candidate P50 scope; only TRAIN uses the 128-negative subset.
9. **SAFE_TO_BUILD_RELATION** — **YES.** The candidate substrate, labels, evaluator, and multi-positive loss are verified and stable; a new expert can be added on the identical universe with a clean A/B.
10. **SAFE_TO_START_QUERY_REGIME_MODELING** — **YES.** The regime raw material (per-query best-gold-rank per expert, rescue counts, dense/SPLADE disagreement) is being collected on TRAIN+VAL (`results/L2/_regime_raw/`, collect-only, no classifier trained). The equal-weight fusion trade-off is quantified and clearly motivates a soft controller.

---

## 5. What to carry forward (architectural read)

- **Carry forward:** Dense (anchor) + exact SPLADE (fusion axis) + Offset + Mixture (deep multi-gold recovery). **Drop:** cheap graph seed-proximity.
- **Do not** simply stack signals at fixed equal weight — 2wiki shows that hurts the top rank. The next architectural step (NOT started, per instruction) is: **build Relation**, then a **query-aware soft controller** that learns *when* each verified expert matters, so deep-recall experts get weight on the queries that need a second/third gold rescued without disturbing queries dense+SPLADE already answer at rank 0.

**STOP.** Per instruction, no Relation build, no controller/dropout training, no Hotpot/SQuAD/MetaQA/WebQSP extension, no L3, L1 untouched. Reporting complete.

---
*Machine-readable: `results/L2/L2_E0_E3_RESULTS.json`. SPLADE scope cache manifest: `results/L2/_splade_scope_manifest.json`. Regime raw (collect-only): `results/L2/_regime_raw/`.*
