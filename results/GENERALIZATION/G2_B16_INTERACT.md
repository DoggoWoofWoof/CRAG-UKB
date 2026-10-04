# G2 · B1.6 — Query-local interaction gate (3-target pilot before six-way)

**Only the reserve-policy REPRESENTATION changed.** Frozen from B1.4/B1.5: candidate universe, verification channel
(`BASE_FUSED`), structural orderer `s_dir`, `two_channel_pool`, `TOP_POOL=50`, `R ∈ {0,4,8,16}`. No new
retriever/encoder/graph/C11a/LLM; source-only supervision identical to B1.5 (per-query utility → conservative preferred
R, smaller-R tie-break); TEST untouched; 0 encoder passes. JSON: `results/GENERALIZATION/_g2_b16_gate.json`.

**Step 1 — explicit interaction features (`P0_INTERACT` = P0 (17) + 6 bounded, hypothesis-driven interactions, no
polynomial expansion).** Composite scalars in [0,1] from the raw P0 values: `RC`=retrieval confidence,
`SNOV`=structural novelty, `SRED`=structural retrieval-redundancy (=1−`sc_weak_support`), `SEED`=seed/anchor support,
`WEAKRET`=graph-only fraction of top structural, `DVR`=directional advantage. Interactions + intended sign:

| term | hypothesis | expected reserve weight |
|---|---|---|
| `RC×SNOV` | novelty given confident retrieval | ambiguous |
| `RC×SRED` | **CASE A**: confident retrieval AND redundant structure ⇒ R=0 | **NEG** |
| `RC×SEED` | de-confound: seed support is redundant when retrieval confident | **NEG** |
| `SNOV×SEED` | **CASE B**: novel AND well-anchored ⇒ genuine discovery | **POS** |
| `DVR×RC` | structure beats retrieval despite confident retrieval | POS |
| `SEED×WEAKRET` | **CASE B core / de-confounder**: anchored AND genuinely graph-only (LOW on SQuAD) | **POS** |

**Model 1 = the SAME linear 4-way softmax over `P0_INTERACT`** (still linear over explicit interactions), compared
directly against B1.5 `P0_LINEAR`. Tiny nonlinear gate = ONE config (1 hidden tanh layer, width 8, L2 5e-3,
class-balanced, same inputs; no architecture search), run only because the linear interactions were directionally weak
yet far below oracle headroom.

---

## Main comparison — per target (verification = `BASE_FUSED`)

| target | policy | **NET** | GO-adm | RV-ret | **over-reserve** | mean R | headroom |
|---|---|---|---|---|---|---|---|
| **MetaQA** | P0_LINEAR (B1.5) | +51 | 0.0145 | 0.996 | 0.142 | 0.96 | 7 % of +740 |
| | **P0_INTERACT** | +51 | 0.0145 | 0.996 | 0.133 | 0.94 | **7 %** (flat) |
| | TINY_GATE | +102 | 0.0265 | 0.991 | 0.230 | 1.69 | **14 %** |
| | *oracle-adaptive* | *+740* | *0.207* | *0.991* | — | — | — |
| **2Wiki** | P0_LINEAR | +69 | 0.174 | 0.997 | 0.620 | 4.48 | |
| | **P0_INTERACT** | +69 | 0.174 | 0.998 | 0.584 | 4.32 | |
| | TINY_GATE | +60 | 0.149 | 0.998 | 0.453 | 3.67 | |
| | *oracle-adaptive* | *+179* | *0.427* | *1.000* | — | — | |
| **SQuAD** | P0_LINEAR | 0 | 0 | 0.999 | **0.614** | **8.72** | |
| | **P0_INTERACT** | 0 | 0 | 0.999 | **0.605** | 8.74 | |
| | TINY_GATE | 0 | 0 | 0.999 | **0.587** | 8.78 | |
| | *oracle-adaptive* | *+1* | *0* | *1.000* | — | *0.016* | |

`NET` = new − evicted golds vs fused-top50; `GO-adm` = graph-only recovered-gold admission; `RV-ret` =
retrieval-visible-gold retention; over-reserve = P(chosen R > eval-only oracle R). MetaQA 3-hop: P0 NET **+44** / GO
0.0124 → P0_INTERACT +47 / 0.0139 → TINY **+94** / 0.0264 (oracle +653).

---

## Result — `QUERY_LOCAL_INTERACTION_SIGNAL = PARTIAL` (SQuAD hard gate NOT cleared)

**The explicit interactions did essentially nothing to the known failure.** P0_INTERACT moved SQuAD over-reservation
0.614 → **0.605** (noise; mean R 8.72 → **8.74**, the wrong way) and MetaQA capture stayed **7 %**. It is not a
meaningful change over the B1.5 linear.

**Why the de-confounders backfire — the crux diagnostic.** Per-dataset mean of each interaction term (want the
reserve-UP terms LOW on SQuAD):

| interaction | metaqa | 2wiki | **squad** | squad − rich | verdict |
|---|---|---|---|---|---|
| `RC×SNOV` | 0.576 | 0.559 | 0.650 | +0.082 | ↑ SQuAD |
| `RC×SRED` (→R=0) | 0.080 | 0.101 | **0.328** | **+0.238** | CASE-A regime **absent from sources** |
| `RC×SEED` (→R down) | 0.132 | 0.098 | **0.436** | **+0.320** | **contaminated by seed** |
| `SNOV×SEED` (→R up) | 0.230 | 0.161 | **0.568** | **+0.372** | **contaminated by seed** |
| `DVR×RC` | 0.404 | 0.399 | 0.435 | +0.033 | flat |
| `SEED×WEAKRET` (→R up, want LOW SQuAD) | 0.188 | 0.139 | **0.348** | **+0.185** | **backfires — HIGH on SQuAD** |

Two structural facts kill the representation:

1. **The seed-support confound contaminates every `SEED`-bearing interaction.** SQuAD's raw seed/anchor support is
   extreme (`SEED`=0.568 vs 0.16–0.23 on the sources). Its graph-only fraction `WEAKRET`=0.563 is only *moderately*
   lower than the sources' (0.83–0.86) — not low enough to overcome the huge `SEED`. So `SEED×WEAKRET` (the intended
   de-confounder) lands **higher** on SQuAD (0.348 vs 0.14–0.19), and `RC×SEED`, `SNOV×SEED` are +0.32/+0.37 higher —
   **all three push R UP on SQuAD**, exactly backwards.
2. **The CASE-A regime is out-of-distribution for source-only supervision.** The rule that should fire R=0
   (confident retrieval **and** redundant structure, `RC×SRED`) needs high-redundancy training examples — but the
   graph-rich source pair has almost none (source `SRED` 0.14–0.17 vs SQuAD 0.44). A policy trained on MetaQA+2Wiki
   never sees "structure redundant despite high anchor support," so it cannot learn to down-weight it. The missing
   signal is a **target property under-represented in the sources** — no amount of interaction cleverness or policy
   capacity manufactures it from source data alone.

**The tiny nonlinear gate is REJECTED (pooled-average gain, not a fix of the failure).** It doubled MetaQA capture
(7 %→14 %, 3-hop NET 44→94) and preserved 2Wiki (NET +60, RV-ret 0.998) — but it achieved this by reserving **more
aggressively on graph-rich deep queries** (mean R up on both MetaQA 0.96→1.69 and SQuAD 8.72→8.78), **not** by learning
the SQuAD redundancy rule. SQuAD over-reservation moved only 0.614 → **0.587** with mean R *rising* to 8.78 — far short
of the SQuAD hard gate's required *material* move toward R=0. The gate improves one side of the known failure
(MetaQA) while leaving the other (SQuAD) intact ⇒ **acceptance rule fails** (must improve **both** sides materially).

---

## Identity diagnostic — NOT a routing proxy

Dataset-prediction accuracy: P0 = **0.7147**, P0_INTERACT = **0.7107** (chance 0.333) — the interactions **did not**
increase fingerprinting (it fell slightly), so `INTERACTION_POLICY_MAY_BE_DATASET_ROUTING_PROXY = NO`. But this cuts
the other way: the interactions added no usable separation *and* the dominant classifier features remain the raw
confound (`sn_unique_out`, `rc_pool_agree`, `rc_agree`, `sc_seed_supp` — whose weight actually *rose* 0.55→0.71). The
separability exists in the raw features; the interactions neither exploit it for routing (good) nor convert it into the
inference-safe reserve rule (the goal).

---

## Verdict

| flag | value |
|---|---|
| `QUERY_LOCAL_INTERACTION_SIGNAL` | **PARTIAL** (SQuAD hard gate not cleared) |
| `LINEAR_RESERVE_POLICY` | still **INSUFFICIENT** |
| `TINY_GATE` | **REJECTED** (pooled-average gain; SQuAD side not material) |
| `ADAPTIVE_TWO_CHANNEL_ADMISSION` | **NOT YET PROVEN** |
| `INTERACTION_POLICY_MAY_BE_DATASET_ROUTING_PROXY` | NO |
| preferred policy | `P0_LINEAR` (P0_INTERACT ties it; tiny gate not accepted) |
| six-way LODO | **NOT launched** · B1 **not** marked solved |

**Bottom line.** The B1.6 hypothesis — that an explicit query-local interaction between retrieval confidence and
structural redundancy would de-confound seed-support and set R→0 on SQuAD — is **refuted for source-only supervision**.
The de-confounding interactions are contaminated by SQuAD's extreme anchor support, and the very regime they target is
out-of-distribution in the graph-rich training pair. A tiny nonlinear gate lifts the graph-rich side but does not touch
the SQuAD over-reservation, so it is rejected under the strict two-sided rule. Per directive, capacity is **not**
enlarged further and six-way LODO is **not** launched; the failure is a **source-coverage / target-property** problem,
not a policy-representation one that more parameters would solve.

## STOP

STOP after the B1.6 three-target evaluation. No six-way LODO; no MLP kept; verification channel and `s_dir` orderer
frozen; TEST untouched; 0 encoder passes. The three datasets remain LODO development evidence, not external-
generalization proof.
