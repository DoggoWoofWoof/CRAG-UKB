# G2 · B1 — universal pool-admission LODO **smoke** (mechanics validation)

**Purpose (per directive): validate MECHANICS before the full six-way LODO** — feature alignment, no leakage, normalization, candidate budget, pool metrics — with a *small* linear admission experiment, not a sweep. Scope is the **original P50 candidate universe** (no geometry expansion in this smoke). Learned admission selects **TOP_POOL = 50** from the ~5000 P50 candidates. TRUE 3-way LODO over the three local datasets **{2wiki, musique, squad}** (val, 700 q/ds). Linear scorer (logistic regression) only; MLP deferred. Artifacts: `_g2_b1_smoke.json`, `scratchpad/_b1_smoke.py`.

## Mechanics — all validated ✓
- **Feature alignment**: identical columns across all datasets — S0 = [dense_rank_pct, splade_rank_pct, fused_rrf_pct, retriever_support_frac, dense_splade_agreement]; S1 adds [min_hop_from_seed_norm, seed_connectivity_frac, struct_support_count_norm, degree_pct]. All per-query bounded/relative.
- **No leakage**: target excluded from train; normalization (mean/std) fit on **TRAIN datasets only** and applied to target; no gold/label/dataset-id in features. (asserted in code + recorded in JSON `leakage_checks`.)
- **Candidate budget**: TOP_POOL=50 enforced; scope ~5000/query.
- **Anchor**: retrieval-seeded (dense top-5), per the contract — no bracket rule, no dataset branch.
- **Structural graph**: row-space `master_nodes.neighbors` (uniform interface); H≤2 BFS, degree-capped.

## Results — GOLD_RECALL@50 / ALL@50 (learned admission vs BASE_FUSED reference)
| Target (train = other two) | BASE R@50 | S0 R@50 | **S1 R@50** | BASE ALL@50 | **S1 ALL@50** | NET_GOLD_GAIN(S1) | churn(S1) |
|---|--:|--:|--:|--:|--:|--:|--:|
| **2wiki** (multi-hop, hyperlink) | 0.744 | 0.741 | **0.768** | 0.503 | **0.556** | **+0.057** | 0.24 |
| musique (title-mention) | 0.941 | 0.916 | 0.905 | 0.886 | 0.836 | −0.070 | 0.32 |
| squad (single-hop, title-mention) | 0.994 | 0.983 | **0.899** | 0.994 | **0.899** | **−0.096** | 0.39 |

## Reading (honest, per-dataset — NOT pooled)
- **Structure is a regime-specific lever, not a universal one.** S1 **beats** S0 and BASE on **2wiki** (+2.4 pt recall, +5.3 pt ALL, NET +0.057) — the multi-hop / low-base / real-hyperlink regime where graph structure carries signal. S1 **hurts** on **squad** (NET −0.096) and slightly on **musique** — exactly the **B0-predicted weak regime** (squad title-match anchor 0.24, derived title-mention graph = structural noise).
- **A single global linear LODO admission does not uniformly beat fused-RRF.** On high-base datasets (musique 0.94, squad 0.99) fused retrieval is already near-ceiling, so any learned reweighting churns golds out (retention 0.90–0.97). This is expected on the **pure-P50 scope**, where golds are largely already reachable and there is **nothing to recover**.

## Decision-gate read (with the required per-dataset caveat)
- `UNIVERSAL_STRUCTURE_ADDS_TRANSFERABLE_VALUE` (S1 > S0): **YES on 2wiki, NO on musique/squad → MIXED, not uniform.** Per the "require per-dataset LODO reporting; do not claim from pooled averages" gate, the honest verdict is: **structure transfers positively only in the multi-hop regime; a flat global structural weight is wrong** — the full model must **gate** structural evidence by regime (retriever-agreement / base-recall context), which a linear term cannot express. This is a concrete design signal for the S2 features and the small MLP.
- Directional geometry (S2), `RECOVERED_GOLD_ADMISSION`, PPR, the small MLP, and metaqa/hotpot/webqsp are **deferred to the full run** (recorded in JSON).

## The key limitation (why this is a smoke, not the science)
This smoke runs on the **pure-P50 scope**, so there are **no geometry-recovered golds** — it therefore does **not** yet test B1's real question ("can a universal rule admit graph-relevant candidates similarity rejects?"). It validates the pipeline and gives an early transfer signal. **Next (full run):** build the geometry-expanded universe (P50 ∪ ≤256 structural directional expansion, retrieval-seeded) so recovered golds exist, add S2, run the true six-way LODO on Modal, and report `RECOVERED_GOLD_ADMISSION@50` + `NET_GOLD_GAIN@50` per dataset. **STOP here per directive.**

## Q2 population caveat (preserved)
All numbers are on the frozen G1/C6 valid-query population (≥1 in-scope gold). Not generalized to fully-L1-failed queries.
