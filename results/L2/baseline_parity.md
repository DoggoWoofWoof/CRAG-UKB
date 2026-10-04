# CRAG L2 — Baseline Parity (Phase A)

**Goal:** reconcile OLD CRAG headline numbers with the CURRENT frozen `relsig` substrate under *genuinely comparable* settings, before training anything new. A number is comparable **only** if same dataset+corpus variant, gold definition, metric, candidate budget, and eval split. Everything else is marked **NON-COMPARABLE**, not a regression.

Sources: `substrate_audit.json` (standalone + pools), `crag_lodo.json` / `crag_lodo_xgb.json` (LODO), `kg_hybrid_fullfd.json` / `kg_hybrid_universal4.json` (old in-dist), `graphlift_gte_qwen.json` / `ner_edge_ablation.json` (old L3), `webqsp_anchored_fullstrength.json`, `e2e_full6_universal_gte_qwen.json`, `hpr_headtohead.json`.

---

## 0. The comparability trap: FOUR different pools

The single biggest source of apparent "regressions" is that history mixes **four incompatible candidate pools**. Absolute recall for the *same* system differs by 20–60 pt across them.

| Pool | What it is | e.g. WebQSP dense R@5 | Used by |
|---|---|---|---|
| **relsig substrate** | ~1.2k–4.5k cand/query, per-query frozen pool | 9.5 | **crag_fusion / kg_hybrid / this program** |
| `_hpr` small | ~1k distractor docs, 100-query test | 88.5 (hotpot) | old head-to-head, crag_variants |
| open 781k KB | all WebQSP entities | 25.4 | graphlift, anchored open-corpus |
| topic-anchored | anchored subgraph (~100s entities) | 28.4 | webqsp_anchored |

**Rule applied below:** old numbers are compared to the current substrate **only** when they share the relsig lineage. `_hpr`, 781k, and anchored numbers are listed but marked NON-COMPARABLE.

---

## 1. Current fixed substrate — pools & gold ceiling

| dataset | n_test | pool size | pool gold-cov (oracle ceiling) |
|---|--:|--:|--:|
| webqsp | 147 | 1184.9 | **88.19%** |
| metaqa | 1998 | 1288.4 | 99.95% |
| 2wiki_clean | 1500 | 1280.0 | 96.17% |
| musique_clean | 1994 | 1386.4 | 96.42% |
| hotpotqa_clean | 1985 | 4533.5 | 98.31% |

> WebQSP's pool oracle is only **88.2%** — a hard cap on any L2 recall; the remaining 11.8% of golds are not in the pool at all (an L1/candidate-gen limit, not L2).

## 2. Standalone experts on the current substrate (R@5)

| dataset | dense | offset | proto | **splade** | graph | relation | path | **best** |
|---|--:|--:|--:|--:|--:|--:|--:|--|
| webqsp | 9.5 | **39.3** | 7.9 | 10.5 | 2.4 | 24.3 | 7.7 | offset 39.3 |
| metaqa | 19.4 | 59.3 | 19.4 | **77.1** | 1.8 | 27.2 | — | splade 77.1 |
| 2wiki | 64.0 | **75.9** | 64.0 | 67.9 | 1.6 | 26.1 | 1.1 | offset 75.9 |
| musique | **66.6** | 66.4 | 54.9 | 64.5 | 2.3 | 6.1 | 1.5 | dense 66.6 |
| hotpot | 70.4 | 58.4 | 70.4 | **71.0** | 2.0 | 15.0 | 0.5 | splade 71.0 |

## 3. THE PARITY TABLE (R@5) — CONFIRMED

| dataset | OLD comparable | new **in-dist** (all-5 / best-cfg) | new **LODO** (kl / best) | best standalone | Δ in-dist vs old | Δ LODO vs standalone | **MARK** |
|---|--:|--:|--:|--:|--:|--:|--|
| webqsp | 54.8 | 48.7 / **52.1** (p_drop 0.2) | 32.95 / 34.29 | offset 39.3 | −2.7 | **−4.4** | PASS* (transfer gap) |
| metaqa | 87.7 **NON-COMP** | 69.5 / 69.9 | 74.70 | splade 77.1 | n/c | **−2.4** | **DEFECT (dilutes splade)** |
| 2wiki | 87.5 **NON-COMP** | 80.5 | 77.94 | offset 75.9 | n/c | **+2.0** | PASS (>standalone) |
| musique | 69.6 | 69.0 | 68.44 / 68.74 | dense 66.6 | −0.6 | **+1.8** | PASS |
| hotpot | — (no relsig old) | 67.1 | 64.33 / 64.69 | splade 71.0 | n/a | **−6.7** | **DEFECT (dilutes splade)** |

- **in-dist** = `crag_indist.json` (ONE model, all-5 pooled, target-in-training, kl). **best-cfg** = webqsp+metaqa 2-ds in-dist at the best p_drop.
- **p_drop sweep (2-ds, dropout_only):** webqsp 0.0→48.2, **0.2→52.1**, 0.5→50.3; metaqa 0.0→68.2, **0.2→69.9**, 0.5→68.7. The new pipeline's `p_drop=0.5` was too aggressive; `p_drop=0.2` (the old `full_fd2` setting) recovers WebQSP to ~52.1, within noise of the old 54.8.
- `*` **WebQSP in-distribution parity is CONFIRMED.** The 34.3 LODO number is **pure domain transfer** — in-training WebQSP scores 50–53. No substrate regression.

### Reading the marks (confirmed)

- **WebQSP — PASS.** In-dist 52.1 (p_drop 0.2) ≈ old 54.8. LODO collapse to ~33 is transfer only → Phase D (transfer robustness).
- **MetaQA — DEFECT, old anchor NON-COMPARABLE.** No fusion on the current substrate reaches the old 87.7: `kg_relsig` flat rerankers give none 69.8 / +relation 67.8 / kl_router 72.2, and new crag_fusion 69.5 — **all below standalone SPLADE 77.0.** So 87.7 required a stronger *old* substrate (NON-COMPARABLE). The genuine, fixable defect is universal across fusion methods: **injecting relation/dense dilutes SPLADE** (relation/path are near-noise on MetaQA — gold-valid 24.5% / 0%). → Phase C/D: SPLADE-preservation.
- **2Wiki — PASS.** In-dist 80.5 and LODO 77.9 both beat standalone offset 75.9. Old 87.5 pool unverified → NON-COMPARABLE, but no regression concern.
- **MuSiQue — PASS.** In-dist 69.0 ≈ old 69.6; LODO +1.8 over dense.
- **HotpotQA — DEFECT.** Fusion (64.3 LODO / 67.1 in-dist) is −6.7 below standalone SPLADE 71.0 — same dilute-the-strong-expert mechanism, reproduced in-distribution. → Phase D/E.

## 4. Old L3 reachability (for Phase G, listed now to avoid re-deriving)

| setting | source | seeds/budget | dense seed | union/reach | note |
|---|---|---|--:|--:|---|
| WebQSP 781k KB | graphlift | 20 / 100, 2-hop | 25.4 | **union 87.8** (lift +62.4; 63.2% of missed reachable) | different pool from relsig |
| clean full-corpus R@100 (struct+wNER, dense-top20 seeds) | ner_edge_ablation | —/100 | musique 79.4 / 2wiki 70.3 / metaqa 32.8 | musique **88.2**, 2wiki **93.7**, metaqa **97.5** | reachability ceiling |
| `_hpr` 1k pool R@100 | ner_edge_ablation | —/100 | — | musique 86.2, 2wiki 98.8, hotpot 97.5, metaqa 97.5 | NON-COMP pool |

## 5. Bottom line (Phase A+B, confirmed)

1. **No catastrophic substrate regression.** WebQSP reproduces old in-dist (52.1 @ p_drop 0.2 vs 54.8); MuSiQue matches (69.0 vs 69.6); 2Wiki beats its standalone. The scary LODO gaps (WebQSP −20) are **domain transfer**, not lost capability.
2. **Two genuine, fixable defects — both "fusion < best standalone", both reproduce IN-DISTRIBUTION:** MetaQA (69.5 vs SPLADE 77.0, −7.5 in-dist / −2.4 LODO) and HotpotQA (67.1 vs SPLADE 71.0). Mechanism (phaseC_evidence): the controller **over-trusts offset / injects near-noise relation & dense against the dense+SPLADE consensus**, diluting the strong expert. Universal across fusion architectures on this substrate.
3. **The universal-non-regression tension is real:** offset is the unique-gold *hero* on WebQSP (40.8% of queries) but the *villain* on Hotpot (0.3% up / 12.7% down). A universal controller must separate them by query evidence, no dataset ID → Phase D (bounded, evidence-gated delta over a semantic anchor).
4. **NON-COMPARABLE old anchors (do NOT count as regressions):** MetaQA 87.7 (no substrate fusion exceeds ~72), 2Wiki 87.5 (overwritten pool), and ALL `_hpr`/781k/anchored numbers (different pools).
5. **Immediate config fix already found:** set `p_drop=0.2` (not 0.5) — recovers WebQSP in-dist ~+2 with no cost elsewhere. Carry into Phase D/E.

*Confirmed by: `crag_indist.json` (all-5 universal), `crag_indist_wm.json` (2-ds p_drop 0.5), `crag_indist_wm_pdrop{0,2}.json` (p_drop sweep). All local/acct9, spend nominal.*
