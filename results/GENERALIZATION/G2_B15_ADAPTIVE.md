# G2 · B1.5 — Query-local adaptive reserve policy (3-target pilot before six-way)

**One new component only:** a map **query → reserve size R ∈ {0,4,8,16}**. Everything else is frozen from B1.4/B1.2 —
candidate universe, verification-channel definitions `{BASE_FUSED, S0, S1_PRIME}`, structural orderer `s_dir`,
`two_channel_pool` composition, `TOP_POOL=50`. No new scorer/encoder/graph/LLM; TEST untouched; 0 encoder passes.

**Verification channel fixed = `BASE_FUSED`** for the policy and all `FIXED_R` baselines (the directive says the *only*
new component is query→R, so the channel is held constant to isolate R; `BASE_FUSED` is simplest and won 2/3 B1.4
source-selections). B1.4's own source-selected channel+R is carried as a baseline row, verbatim from its JSON.

**Model:** linear **4-way softmax** (formulation A), class-balanced, source-**TRAIN**-only normalization/weights, frozen
before target; a-priori smaller-R margin (0.10) breaks ties toward smaller R. **Not an MLP.**

**Labels (TRAIN only, gold allowed):** per source query, utility(R) = golds in pool_R − golds in fused-top50;
conservative preferred R = smallest R at max utility, and R>0 only if it *strictly* beats R=0 ⇒ **R=0 is legitimate and
common** (source label dist ≈ 88–94 % R=0). No target gold used for labels or selection.

**Features** (query-local, inference-safe, bounded — no dataset ID, no hop, no gold at runtime):
`P0` = retrieval-confidence + structural-novelty + structural-confidence (17 dims). `P1` = P0 + available
runtime/graph-regime proxies (candidate count, structural-generation exhaustion, local neighborhood density) — an
**ablation**; true per-query frontier/edge-budget counters are **not persisted** (stat JSON stores aggregates only), so
P1 uses the available proxies and this limitation is stated. JSON: `results/GENERALIZATION/_g2_b15_adaptive.json`.

---

## Main results — per fold (verification = BASE_FUSED)

| target | config | R@50 | GO-adm (avail) | RV-ret | **NET** | notes |
|---|---|---|---|---|---|---|
| **MetaQA** | FIXED R0 / R4 / R8 / R16 | .280/.329/.363/.420 | 0/.066/.117/.210 | 1.0/.971/.953/.897 | 0 / 223 / 378 / **635** | monotone in R |
| | B1.4 src-selected (BASE_FUSED R16) | .420 | .210 (2906) | .897 | **+635** | |
| | **QUERY_LOCAL_P0** | .291 | .015 | .996 | **+51** | ✅ letter (NET>0, GO>0) — but hollow |
| | QUERY_LOCAL_P1 | .294 | .016 | .996 | +62 | |
| | *oracle-adaptive (eval-only)* | *.443* | *.207* | *.991* | *+740* | *per-query optimum* |
| **2Wiki** | FIXED R0 / R4 / R8 / R16 | .896/.918/.926/.931 | 0/.228/.313/.427 | 1.0/.997/.995/.987 | 0 / 84 / 116 / 135 | |
| | B1.4 src-selected (BASE_FUSED R8) | .926 | .313 | .995 | +116 | |
| | **QUERY_LOCAL_P0** | .914 | .174 | .997 | **+69** | ✅ |
| | QUERY_LOCAL_P1 | .917 | .206 | .998 | +81 | |
| | *oracle-adaptive* | *.943* | *.427* | *1.000* | *+179* | |
| **SQuAD** | FIXED R0 / R4 / R8 / R16 | .996 | 0 (**0 avail**) | 1.0/1.0/1.0/.999 | **0 / 0 / 0 / 0** | **NET R-insensitive** |
| | B1.4 src-selected (S1_PRIME R8) | — | 0 | .978 | **−20** | (damage was the S1_PRIME channel) |
| | **QUERY_LOCAL_P0** | .996 | 0 | .999 | **0** | ✅ letter — but vacuous (see below) |
| | QUERY_LOCAL_P1 | .996 | 0 | .999 | 0 | |
| | *oracle-adaptive* | *.997* | *0* | *1.000* | *+1* | *R=0 for 99.9 %* |

`GO-adm` = graph-only recovered-gold admission fraction; `RV-ret` = retrieval-visible-gold retention vs fused-top50;
`NET` = new − evicted golds vs fused-top50. **P1 ≥ P0** by a hair on the graph-rich targets but fingerprints more and
over-reserves SQuAD slightly more (below) → **prefer P0** per directive.

---

## R distribution & the critical policy diagnostic (chosen R vs eval-only target-oracle R)

| target | P0 R-dist (0/4/8/16) | mean chosen / oracle R | **OVER-reserve** | UNDER-reserve | R-acc | MAE(R) |
|---|---|---|---|---|---|---|
| MetaQA | .84 / .09 / .07 / .00 | 0.96 / 1.69 | 0.142 | **0.137** | 0.721 | 2.41 |
| 2Wiki | .33 / .41 / .16 / .10 | 4.48 / 0.66 | **0.620** | 0.049 | 0.332 | 4.57 |
| **SQuAD** | .39 / .01 / .13 / **.48** | **8.72 / 0.016** | **0.614** | 0.000 | 0.386 | **8.70** |

**SQuAD is the crux.** Its source (MetaQA+2Wiki) is labeled 88 % R=0, yet the policy sends SQuAD to R=0 only 39 % of the
time and to **R=16 on 48 %** (mean chosen R = 8.72 vs oracle 0.016, MAE 8.7). Over-reservation is **materially reduced
vs B1.4 (0.999 → 0.614)** but still a **majority** of queries.

---

## Why the letter-pass is a **QUALIFIED** pass — three caveats

**(a) SQuAD's NET-pass is a `BASE_FUSED` artifact, not policy skill — the SQuAD-NET criterion is VACUOUS.**
Under `BASE_FUSED`, `FIXED_R` at *every* R (0/4/8/16) gives SQuAD **NET = 0, RV-ret ≈ 1.0**: SQuAD golds sit at fused
rank 0–1, so reserving the bottom slots never evicts a gold and (0 graph-only golds) never admits one. NET therefore
**cannot discriminate policies on SQuAD**; the only meaningful SQuAD metric is over-reservation (0.614, still high).
B1.4's −20 came specifically from the `S1_PRIME` channel reshuffling RV golds downward — a channel effect, not a reserve
effect.

**(b) MetaQA capture is hollow.** P0 NET = +51 of the target-oracle **+740** (≈ **7 %** of headroom). Per-hop:

| hop | GO avail | P0 GO-adm | P0 NET | oracle NET |
|---|---|---|---|---|
| 1-hop | 7 | 0.000 | −2 | +10 |
| 2-hop | 243 | 0.037 | +9 | +77 |
| 3-hop | **2656** | **0.012** | **+44** | **+653** |

The policy correctly learns ~85 % of MetaQA is R=0 but **under-reserves the deep multi-hop minority** (3-hop) where the
graph-only golds actually live — capturing +44 of +653 available NET there.

**(c) It over-reserves the graph-rich targets too** (2Wiki over-reserve 0.62), winning there only because
over-reservation is *tolerated* when the graph is rich. The policy's behaviour is driven by the LODO **source
composition**, now at query granularity: MetaQA fold (source includes graph-poor SQuAD) → conservative → under-reserves;
SQuAD fold (both sources graph-rich) → over-reserves.

---

## Which inference-safe quantities fail to separate useful novelty from noise (directive-requested)

Per-dataset mean P0 features (SQuAD − graph-rich-mean in the last column):

| signal | metaqa | 2wiki | squad | Δ(squad − rich) | reads as |
|---|---|---|---|---|---|
| `sc_multiseed` (struct multi-seed support) | 0.230 | 0.161 | **0.568** | **+0.372** | **CONFOUND** ↑ |
| `sc_seed_supp` | 0.264 | 0.228 | 0.457 | +0.211 | **CONFOUND** ↑ |
| `rc_pool_agree` (dense/splade agree) | 0.189 | 0.205 | 0.433 | +0.236 | correct ↑ (retrieval easy) |
| `rc_support` | 0.741 | 0.770 | 0.929 | +0.173 | correct ↑ |
| `sc_weak_support` (struct w/ weak D/S) | 0.862 | 0.830 | 0.563 | **−0.284** | correct ↓ (already visible) |
| `sc_dirvsret` (struct beats retrieval) | 0.700 | 0.667 | 0.570 | −0.113 | correct ↓ |
| `sc_dirmargin_topk` | 0.875 | 0.775 | 0.742 | −0.083 | correct ↓ |

**The confound is seed/anchor support.** On retrieval-easy SQuAD, structural expansion is *well-anchored* (high
`sc_multiseed`/`sc_seed_supp`), which on the graph-rich sources correlates with *useful* discovery and carries a positive
R>0 weight — so the linear policy reserves. The genuinely discriminative signals **do exist and point the right way**
(SQuAD's structural discoveries are already retrieval-visible `sc_weak_support`↓, don't beat retrieval `sc_dirvsret`↓,
and its queries are retrieval-confident `rc_support`↑), but a **linear** model cannot represent the required
**interaction** "retrieval-confident **AND** structurally-redundant ⇒ R=0" — it sees high seed-support and reserves. The
identity diagnostic confirms the raw separability is present (**dataset-prediction acc P0 = 0.715, P1 = 0.726, chance
0.333**) but it lives in an interaction the R-label mapping does not — and must not (no dataset ID) — exploit.

---

## Verdict — QUALIFIED PASS, signal PARTIAL (decision deferred to you)

| success condition (P0 linear) | met (letter)? |
|---|---|
| MetaQA: GO-adm > 0 **and** NET > 0 | ✅ (0.015 / +51) |
| 2Wiki: NET ≥ 0 **and** RV-ret ≥ 0.95 | ✅ (+69 / 0.997) |
| SQuAD: NET ≥ −5 **and** RV-ret ≥ 0.99 | ✅ (0 / 0.999) — **but SQuAD NET is vacuous** |
| SQuAD over-reservation materially reduced vs B1.4 | ✅ (0.999 → **0.614**, still majority) |

> **The P0 linear policy clears the LETTER of all four conditions, but the query-local reserve signal is only PARTIAL.**
> It can be read two ways, and the call is yours:
> - **LITERAL PASS** → `ADAPTIVE_TWO_CHANNEL_ADMISSION = PROMISING`; prefer P0 (P0 ≥ P1, less fingerprinting), **stop
>   model complexity**, proceed (separately) to six-way LODO.
> - **`QUERY_LOCAL_RESERVE_SIGNAL = PARTIAL / INSUFFICIENT`** → the SQuAD over-reservation (0.614, majority; mean R 8.72
>   vs oracle 0.016) and the ~7 % MetaQA headroom capture say the signal doesn't yet cleanly separate useful structural
>   novelty from noise. The mechanism is identified (seed-support confound × missing interaction), and the target-oracle
>   adaptive R shows **large headroom on all three folds** — the contingency where a *tiny nonlinear gate* (interaction
>   of retrieval-confidence and structural-redundancy, **not** a bigger MLP) would be the next lever.

**Per directive:** do **NOT** increase model capacity on the basis of the vacuous SQuAD-NET pass; the deficiency is
**policy signal/representation**, not candidate scoring. **Six-way LODO NOT launched.**

---

## STOP

STOP after the B1.5 three-target evaluation, per directive. No six-way LODO launched; no MLP; verification channel and
`s_dir` orderer frozen; TEST untouched; 0 encoder passes. The three datasets remain LODO development evidence, not
external-generalization proof.
