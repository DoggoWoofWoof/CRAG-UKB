# G1 — Cross-Dataset Generalization: Dataset & Expert-Availability Manifest

**Date:** 2026-08-27 · **Status:** AUDIT ONLY (no substrate built, no head trained, no C11a run). This is the
gating deliverable the spec requires *before* any build: *"first audit which frozen upstream components are
legitimately available … do NOT silently invent missing upstream experts."*

## Locked text-pilot architecture (frozen for all of G1)

L1 topology **C**, Dense+SPLADE partition-RRF (K0=60, K=100, **P=50**) → 5 frozen experts {Dense, SPLADE,
Offset, Mixture, Masked-Relation} → **C7b** soft-archetype fusion → **C8c** XGBRanker reranker → **C11a**
interaction-MLP residual. No dataset ID, no LLM/cross-encoder/transformer/attention/GNN. Backbone trained on
**2wiki+musique TRAIN only**, dataset-agnostic → applied unchanged for ZERO_SHOT. **Not modified in G1.**

## The core finding

**The assembled C11a substrate (`data/l2_corpus/…` — the 5 aligned expert arrays + `expert_meta.npz`) exists
only for the two DEV datasets.** Every target has the *raw ingredients* (cached gte_qwen reps, dense/splade
top-200, partition maps, `graph.pt`, NER edges) but **no assembled substrate**. So G1 is gated on rebuilding
each expert *by a frozen, dataset-agnostic method* — and one expert (Mixture) has no such frozen checkpoint yet.

## Corpora

| Dataset | Family | Nodes | Queries | Role |
|---|---|---:|---:|---|
| 2wiki_clean | TEXT | 65,865 | 15,000 | DEV / source |
| musique_clean | TEXT | 13,672 | 19,938 | DEV / source |
| **squad_clean** | TEXT | 19,029 | 130,319 | target 1 (single-evidence) |
| **hotpotqa_clean** | TEXT | 507,494 | 97,852 | target 2 (large corpus) |
| **webqsp** | KB | 781,485 | 1,578 | target 3 (tiny queries, huge KB) |
| **metaqa** | KB | 40,151 | 407,513 | target 4 (hop 1/2/3) |

## Splits (TEST locked everywhere, including the DEV pair)

| Dataset | train | val | test | val cache | dev note |
|---|---:|---:|---:|---|---|
| squad_clean | 91,223 | 26,063 | 13,033 | **not materialized** | val = `queries_all[val_idx]` — legitimate split, **no synthetic dev** |
| hotpotqa_clean | 68,496 | 19,570 | 9,786 | materialized | — |
| webqsp | 1,104 | 315 | 159 | materialized | tiny; `build_l2_corpus` currently special-cases webqsp→`all`, must switch to train/val/test |
| metaqa | 329,282 | 39,138 | 39,093 | **not materialized** | val sliceable; **per-query hop label present** → 1/2/3-hop breakdown |

All four targets have a **legitimate val split** in `split_indices`; squad/metaqa merely lack the separate
`queries_val.npy` cache (sliceable from `queries_all.npy`). No TRAIN-derived synthetic dev is required.

## Expert-availability matrix

Legend: **AV** = assembled on disk · **RB** = rebuildable with a frozen dataset-agnostic method (no target
learning) · **RB\*** = frozen *trainer* exists but no universal checkpoint on disk — decision required · **DIFF**
= meaningful but different substrate.

| Expert | 2wiki | musique | squad | hotpot | webqsp | metaqa | frozen method |
|---|---|---|---|---|---|---|---|
| **Dense** | AV | AV | RB | RB | RB | RB | cosine over cached nodes (`build_l2_corpus`) |
| **SPLADE** | AV | AV | RB | RB | RB | RB | `splade_top200_all` + `l2_splade_cache` scope |
| **Offset** | AV | AV | RB | RB | RB | RB | **universal** offset head (full-6, no refit) |
| **Mixture** | AV | AV | **RB\*** | **RB\*** | **RB\*** | **RB\*** | MixtureHead (mlpT) — universal ckpt **missing** |
| **Masked-Relation** | AV | AV | RB | RB | **DIFF** | **DIFF** | title-mention mask + gte-Qwen2 sentence encode |

### Notes that change the numbers

- **Offset — distribution-shift caveat.** The DEV substrate used **per-dataset** offset heads
  (`_heads/{2wiki,musique}_offset_cp50.pt`). Targets have no per-dataset head, so they must use the **universal
  offset head** (`src/experiments/l1_universal_head`, one head trained on the broad-6 mix, already applied to all
  6 in `e2e_full6_universal_gte_qwen.json`). Legitimate and frozen — but the offset feature on targets comes from
  a *different head* than C11a was trained on. The **Zero-Shot Feature Compatibility** step must measure this.

- **Mixture — the one real gap (decision required).** Only per-dataset mixture heads (2wiki/musique) exist; no
  universal mixture checkpoint is on disk. A universal mixture head is producible by the *identical* frozen method
  that made the universal offset head (`l1_universal_head` `kind="mix"`, trained once on the broad-6 mix, applied
  without refit) — the exact mirror of Offset. That is the consistent, non-inventing path, but it is one training
  run and a **substrate-defining choice**. Mixture is part of the C11a 5-expert input interface and cannot be
  dropped without changing the interface (disallowed). **This must be decided before assembling any target
  substrate.**

- **Relation — encode cost / KB semantics.** The mask is pure title-mention topology (from `graph.pt` /
  `ner_edges_w_df25.pkl`, present for every target). The score needs a **gte-Qwen2 encode** of connecting
  sentences (Modal GPU; heavy for hotpot 507k / metaqa). `relsig_feats_{ds}.pt` exist for all six and might
  shortcut this, but memory flags the old relsig as capped/stale — validate before reuse. On **KB** (webqsp,
  metaqa) the relation is native *typed graph edges*, not NL connecting sentences: the mask is trivially rich but
  the Qwen-sentence score has different semantics — reported as a substrate difference, **not forced**.

## KB path / graph signal (reported, NOT added to C11a in G1)

`graph.pt` and `ner_edges_w_df25.pkl` are present for both webqsp and metaqa, so a Path/structural signal is
available from frozen infrastructure. Per the spec it is **reserved**: the first KB test measures how far the
*text* architecture transfers. If KB failures persist despite candidate presence and the evidence points at
missing relation/path structure, that is reported as `MISSING_RELATIONAL_STRUCTURE` — **not patched during G1.**

## Backbone applicability

C7b / C8c / C11a carry no dataset ID and apply unchanged to any assembled substrate.
**ZERO_SHOT_C11A** = exact saved C11a weights + **original 2wiki+musique TRAIN** E-standardization (`emean/estd`
from the joblib) + original top-20 policy; **no** target-fitting/recalibration/threshold change, **never** target
VAL/TEST statistics. **REFIT_C11A** = same architecture/hyperparameters/loss/optimizer, trained on target TRAIN
only, selected on target VAL.

## Build-plan order (subsequent G1 steps)

0. **DECISION — universal Mixture head** (close the gap the same frozen way as Offset). *Blocks step 1.*
1. **TEXT targets** (squad → hotpot): `build_l2_corpus` (Dense+SPLADE scope) → splade_scope cache → universal
   offset+mixture scores → relation mask + Qwen sentence encode → `expert_meta.npz`.
2. **Zero-Shot Feature Compatibility**: distribution diagnostics (query/candidate emb norms, expert ranks, RRF
   contribs, C8c scores, C11 evidence) target vs source — original-train standardization only.
3. **ZERO_SHOT_C11A + REFIT_C11A** per target, with all four baselines.
4. **KB targets** (webqsp → metaqa): representation-mismatch audit first (candidate = entity/relation/path/node);
   same interface; MetaQA 1/2/3-hop + the conditional-on-coverage ceiling (the central L3 question).
5. **G1_MATRIX** + interpretation categories + final gates.

## Decisions surfaced (must not be resolved by inventing an expert)

1. **MIXTURE_UNIVERSAL_HEAD** — build one universal mixture head to preserve the 5-expert interface on targets
   (recommended; the only alternative is an interface change, which is disallowed)?
2. **RELATION_ENCODE** — reuse `relsig_feats_*.pt` (fast, possibly capped/stale) vs a fresh gte-Qwen2
   connecting-sentence encode on Modal (canonical, heavy)?
3. **KB_RELATION_SEMANTICS** — keep text-Qwen relation semantics on KB (honest transfer measurement) vs a native
   typed-edge score (changes the expert)?

## No test leakage

2wiki/musique TEST not inspected. Targets use TRAIN/VAL (legitimate `split_indices`) only. All TEST splits remain
locked. **This step performed no builds, no training, and no C11a evaluation.**

**Artifacts:** `results/GENERALIZATION/G1_DATASET_MANIFEST.json` (structured matrix) · this file.
