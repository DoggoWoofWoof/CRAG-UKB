# L2 Readiness Audit (READ-ONLY) — pre-training

**Scope:** what is *actually* in the repo for L2, classified so we never again call a feature "used" because a helper exists. No training was run. Terminology used throughout:

- **IMPLEMENTED** — code exists and runs.
- **WIRED** — actually invoked by a canonical pipeline entry point (not just an experiment script).
- **TRAINED** — a fitted checkpoint exists on disk.
- **VALIDATED** — measured on held-out eval with a recorded result.
- **PROPOSED** — described/intended, not built.

> **Headline:** there are **two parallel L2 codebases**, and **neither is currently built on the just-locked C/P50 interface.** That mismatch — not missing model code — is the main thing standing between us and a clean L2 v2.

---

## 0. The two existing L2 lines

| line | entry point | what it is | candidate pool it uses | status |
|---|---|---|---|---|
| **L2-A: offset/mixture heads → fusion → L3** | `src/experiments/e2e_pipeline.py` `run()` | dense + offset + mixture + SPLADE [+adapter], parameter-free fusion, then PPR L3 (`_ner_compose`) | **full corpus** (`scope_topk=0` default) or optional partition-`topP`; **topology A** (`eng.partition_map` from `gte_qwen`) | IMPLEMENTED+WIRED+TRAINED+VALIDATED |
| **L2-B: candidate-level learned fusion** | `src/experiments/kg_hybrid.py` + `crag_fusion.py` / `crag_gates.py` / `crag_ranker.py` | 5-expert masked substrate + query-state controller (KL-α / cooperative sigmoid gates / XGBRanker) scoring candidates | **dense-top-200 ∪ NER-2-hop**, capped `tr1104/te2000` (`kg_relsig._run`: `n_dense=200,n_seed=20,hops=2`) | IMPLEMENTED+WIRED+TRAINED+VALIDATED(LODO) |

**Neither pool is the locked interface** (C partition_map, Dense+SPLADE fusion, K100, **P50 ≈ 5k candidates**). L2-A uses full-corpus/A-partition; L2-B uses a dense+NER pool on a stale cap. **Rebuilding the L2 training substrate on the C/P50 candidate set is prerequisite work item #1.**

---

## 1–12 required items

**1. Current implemented L2 architecture.** Two lines as above. The most target-shaped one is **L2-B** (candidate-level, query-state soft controller over per-expert weights, authoritative masks). L2-A is the head+end-to-end+L3 line and is what feeds the current L3 traversal.

**2. Actual signal list** (the real feature stack — see full table in §13).
- L2-A signals (`e2e_pipeline.level2_order`): `dense`, `rel_hard` (OffsetHead), `mlpT` (MixtureHead-K8), `SPLADE`, optional `adapter` (ResidualAdapter). **No relation/path/graph feature in L2-A.**
- L2-B experts (`kg_hybrid.EXPERTS = [dense, offset, splade, relation, path]`; 7 raw cols incl. unused `prototype`, `graph`). Relation/path/graph come from `kg_relsig`.

**3. Existing checkpoints.**
- `data/ukb_storage/_head_cache/head_*.pt` — many OffsetHead (~6.3 MB, dim 1536) and MixtureHead-K8 (~28 MB) heads, content-fingerprinted; `adapter_*.pt` (~25 MB) ResidualAdapters.
- `results/L2/*.pt` — `relsig_feats_<ds>.pt` (cached L2-B feature tensors, 86 MB–843 MB), `kg_hybrid_*_preds.pt`, `l2_gate.pt`, `_squad_metric_ckpts.pt`.

**4. Existing training data.** No frozen "L2 training corpus" file. L2-A builds triples `(query, dense-top1 seed, gold)` on the fly in `_train_universal` (head_datasets = musique+2wiki). L2-B caches per-query feature tensors (`relsig_feats_*.pt`) over the dense+NER pool (capped). **A frozen L2 corpus on C/P50 does not yet exist.**

**5. Existing feature caches.** `dense_top200_all.npy`, `splade_top200_all.npy` (per dataset, in the L1 manifest, hashed); `relsig_feats_<ds>.pt` (L2-B, stale pool/cap); head/adapter caches. Dense/SPLADE caches are canonical and reusable; relsig caches must be rebuilt on C/P50.

**6. Relation implementation status.** IMPLEMENTED + TRAINED (`kg_relsig`: `relation_raw` = cos over relation-text; `relation_trained` = InfoNCE head with sibling-relation hard negs). WIRED only in L2-B. **VALIDATED as weak/absent on text corpora** (relation needs edge relation-text; genuinely informative mainly on KB/webqsp). Requires the graph (edge → relation text).

**7. Path implementation status.** IMPLEMENTED (`kg_relsig`: `path_raw` = max path-text similarity along multi-hop). WIRED only in L2-B. **VALIDATED as "available-but-useless on text sets."** Requires the graph. Pair/path-level.

**8. Mixture implementation status.** IMPLEMENTED+WIRED(L2-A)+TRAINED+VALIDATED (`l1_ablate.MixtureHead`, K=8; `mix_hard` variant adds hard-neg mining). Query-conditioned offset mixture; **not** an expert in L2-B.

**9. Current losses.**
- L2-A `_train_universal`: **InfoNCE / cross-entropy** with in-batch negatives + **hard-negative mining** (GPU top-k over `Xt`: 16 negs offset, 8 negs mixture), temperature `TAU`. No regime/edge/transition loss.
- L2-B: **ListNet candidate loss + λ·KL(π*‖α)** (crag_fusion), where π* = softmax(−bounded-InfoNCE expert loss); crag_gates adds cooperative KL (G1) + pairwise hinge (G2) + corroboration; crag_ranker adds XGBRanker (grouped LTR). Expert/structure **dropout** already implemented.

**10. Current negative sampling.**
- L2-A: hard negs = current head's top-k over the corpus, own-gold masked. No diverse-source sampler.
- L2-B: negatives subsampled for **train only, full pool at eval** (crag_ranker); pool itself is dense+NER-2hop. No explicit Dense/SPLADE-disagreement / partition-vote / kNN-confusable negative buckets (the user's target sampler is **PROPOSED**, not built).

**11. Current train/val/test loaders.**
- L2-A: `l1_universal_head._load` (per-dataset, `_splits` train/val/test via `overlap_retrain`), multi-dataset head training over `head_datasets`. Query caps supported.
- L2-B: `kg_hybrid._load` (multi-dataset, pooled train mu/sd), `crag_fusion.run_lodo` = leave-one-dataset-out. **Multi-dataset + LODO exist.** Dataset-balanced/hop-balanced/regime-balanced samplers are **PARTIAL** (regime work exists as audits: `results/L2/_query_regime.json`, `_regime_ablation_smoke.json`).

**12. Current GPU requirements.** L2-A head training dominated by hotpot hard-neg mining (~13 min historically; now GPU-topk). Adapter ~40 epochs. L2-B substrate build (`kg_relsig`) is the heavy step (encode + NER-hop expansion + relation/path features; the 843 MB hotpot cache reflects it). Controllers themselves (crag_fusion/gates/ranker) are small/fast once the substrate is cached. A single A10/A100 suffices for all model fitting; the cost is **substrate construction**, not training.

**13. Signal table**

| SIGNAL | IMPL? | CODE PATH | USED (canonical)? | TRAINABLE? | CACHED? | QUERY | CAND | PAIR/EDGE | NEEDS C GRAPH | classification | notes |
|---|:--:|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|---|
| Dense | ✔ | gte-Qwen2 / FAISS; both lines | ✔ | via adapter | ✔ `dense_top200` | ✔ | ✔ | – | no | IMPL+WIRED+VALID | anchor signal (`sigs[0]`), `FEAT_NAMES[0]` |
| SPLADE | ✔ | `l2_seed._splade_scoped_order`; `core.splade_scorer` | ✔ | frozen | ✔ `splade_top200` | ✔ | ✔ | – | no | IMPL+WIRED+VALID | learned-lexical; strong on metaqa/hotpot |
| Offset (rel_hard) | ✔ | `query_relation.OffsetHead` | ✔ (L2-A) / expert (L2-B) | ✔ | ✔ head cache | ✔ | ✔ | – | no | IMPL+WIRED+TRAINED+VALID | `norm(seed+MLP(q))`; hard-neg mined |
| Mixture (mlpT) | ✔ | `l1_ablate.MixtureHead` K8 | ✔ (L2-A) | ✔ | ✔ head cache | ✔ | ✔ | – | no | IMPL+WIRED+TRAINED+VALID | K offset directions; not an L2-B expert |
| Relation | ✔ | `kg_relsig` relation_raw/trained | L2-B only | ✔ | ✔ `relsig_feats` | – | ✔ | ✔ | **yes** (edge rel-text) | IMPL+TRAINED, WIRED(L2-B), VALID=weak-on-text | KB signal; redundant on text (Phase-C/D) |
| Path | ✔ | `kg_relsig` path_raw | L2-B only | derived | ✔ `relsig_feats` | – | ✔ | ✔ | **yes** | IMPL, WIRED(L2-B), VALID="useless-on-text" | multi-hop path-text sim |
| Graph distance | ✔ | `kg_relsig` `graph = -hopcount` | ✘ (dropped from EXPERTS) | – | ✔ (raw col) | – | ✔ | – | **yes** | IMPL, NOT WIRED as expert | hop distance from seeds |
| NER (edges) | ✔ | `pipeline.ner_edges.build_ner_edges` (df25,w) | ✔ (L3 graph; L2-B pool exp.) | 1/df weights | ✔ `ner_edges_w_df25.pkl` | – | – | ✔ | is a C family | IMPL+WIRED(L3)+VALID | dominant L3 lever (see L1 diag) |
| Qwen-kNN (edges) | ✔ | C graph family | **✘ unused downstream** | – | ✔ (in C graph) | – | – | ✔ | is a C family | IMPL, UNUSED | dropped at L3; marginal per L1 diag |
| Structural connectivity | ◑ | struct edges → `mem_idx` vote (L1), `A_str` in L3 graph | ✔ (L1 route + L3) | – | ✔ | ✔(vote) | – | ✔ | is a C family | PARTIAL | no dedicated L2 connectivity score |
| Learned heads | ✔ | OffsetHead, MixtureHead, ResidualAdapter, crag controllers, FusionGate, GateModel, XGBRanker | mixed | ✔ | ✔ | ✔ | ✔ | – | no | IMPL+TRAINED | see §3 |
| Adapter features | ✔ | `dense_adapter.ResidualAdapter` | optional (L2-A `use_adapter`) | ✔ | ✔ `adapter_*.pt` | ✔ | ✔ | – | no | IMPL+TRAINED+VALID | regime-specific win (KB/hard-text) |
| Prototype | ✔ | `kg_relsig` prototype col | ✘ (dropped) | – | ✔ (raw col) | – | ✔ | – | no | IMPL, NOT WIRED | class-prototype sim |

**14. What can be reused (no rebuild).** Dense/SPLADE top-200 caches (canonical, hashed in L1 manifest); OffsetHead + MixtureHead-K8 architectures & training loop (`_train_universal`, InfoNCE+hard-neg); ResidualAdapter; the entire L2-B controller zoo (KL-α, cooperative gates, XGBRanker) and its mask/dropout/teacher machinery; multi-dataset + LODO loaders; the L3 PPR (`_ner_compose`) consuming struct+NER. The **C master graph + edge families** (parity PASS) are the substrate for any graph feature.

**15. What must be newly built for a C/P50 all-signals L2.**
- **(a) A frozen L2 training corpus on the locked interface:** for each TRAIN query, the C/P50 candidate set (~5k) from cached Dense+SPLADE fusion, with **all in-scope golds preserved** and `N_GOLD_EXPECTED` / `N_GOLD_IN_SCOPE` / `ANY_GOLD_PRESENT` / `ALL_GOLD_PRESENT` recorded per query (separates L1-pruning from L2-ranking failure).
- **(b) Feature recomputation on that pool:** relation/path/graph features currently live on the dense+NER pool (capped) — recompute on C/P50 (uncapped) or explicitly restrict to it.
- **(c) A diverse hard-negative sampler** (top-Dense, top-SPLADE, Dense/SPLADE-disagreement, high C-partition-vote, same-partition, NER-confusable, kNN-confusable, random) — currently PROPOSED.
- **(d) The L2→L3 output contract** (relevance + regime_probs + expert_weights [+ optional edge/transition scores]) — currently the two lines don't emit a shared schema.
- **(e) Optional multi-task heads** (edge/transition/expand/regime) — only where valid supervision exists (see loss note below).

---

## Deliverable 2 — Proposed **L2 v2** architecture (design only, not implemented)

**Substrate:** frozen corpus on **C partition / Dense+SPLADE / K100 / P50** (~5k cand/query), all in-scope golds kept, per-query scope-accounting fields (a).

**Verified experts to carry (start here, don't over-reach):** `dense`, `splade`, `offset`, `mixture`. **Graph experts staged in** only where they pay: `relation`/`path` are KB-leaning (weak on text) → include but expect the controller to down-weight on text; `graph`(hop)/kNN → candidates for edge-selection features, not standalone rankers.

**Model (reuse L2-B shape):** authoritative per-candidate masks → per-expert features `[score, rank_pct, top-k]` → **query-state soft controller** emitting **non-exclusive** `h_e(q)` (cooperative sigmoid gates, *not* softmax simplex — avoids forced competition / collapse) → candidate MLP scoring `[raw φ, gated z, h, cross-expert corroboration, masks]`. Keep `w_i(q)` now; design the interface so `w_i(q, candidate)` is a later drop-in.

**Query-regime controller (auxiliary, interpretable):** derive regimes on **TRAIN only** from per-expert gold-rank behavior (which expert rescues, agreement/disagreement), cluster into SEMANTIC / LEXICAL / RELATIONAL / MULTIHOP / CONFLICT / RESCUE / EASY / HARD — **do not hard-freeze labels**; expose `regime_probs` as an auxiliary head, never a hard router.

**L2→L3 interface (design now, emit later):** `{query_id, candidate_id, relevance_score, regime_probs, expert_weights[, edge_usefulness/transition_score, expand_priority]}`. This is what lets L3 do **selective fan-out over C edge families** (top-M edges/node by a query-aware transition score) instead of blind traversal.

**Loss:** `L_rank` (ListNet/InfoNCE, keep) `+ λ_regime·L_regime` (weak, TRAIN-derived) `+ λ_edge·L_edge` / `λ_transition·L_transition` **only** where supervision is real — derive weak edge/transition labels from gold supporting-evidence transitions / shortest useful paths and **label them weak/derived**; do not fabricate relation/path targets. Anti-collapse: expert dropout + usage/entropy regularization — introduce **one at a time**, validate each.

**Balancing & safety:** dataset- and hop-balanced sampling (don't let hotpot/squad size dominate); TRAIN-only fitting, VAL for hyperparams/checkpoint/thresholds/regime tuning, TEST locked; MetaQA native splits; WebQSP no invented training labels; MetaQA/WebQSP topology caveats stay. Checkpoint selection on **aggregate + per-dataset + worst-dataset + multi-hop + regime** metrics (robustness, not mean).

**Explicitly deferred (documented, not built):** `w_i(q,candidate)`; edge/transition/expand multi-task heads beyond weak supervision; **L3 MODE 2 BOUNDED_FRONTIER_EXPANSION** (traversal outside P50) — remains a future L3 experiment; L3 must not silently reintroduce the full corpus.

---

## Deliverable 3 — Cheapest ablation sequence (does each signal help?)

Modified to what is actually implemented. All on **C/P50 substrate**, one universal dataset-agnostic model, ranking the **full P50 scope at eval**, reporting **MRR / Recall@{5,20,50} / Hits(full-gold)@k / NDCG**, broken out by **dataset / hop / L1 gold-present status**. Stop condition = no worst-dataset regression vs previous rung beyond noise.

| exp | model | new cost | why |
|---|---|---|---|
| **E0** | Dense only | none (cached) | floor |
| **E1** | + SPLADE (param-free fusion) | none (cached) | orthogonal-lexical lift; matches locked router |
| **E2** | + Offset | reuse head cache | does relational offset survive on C/P50? |
| **E3** | + Mixture | reuse head cache | K-direction gain over single offset |
| **E4** | + Relation | **rebuild relsig on C/P50** | KB lift; expect ≈0 on text |
| **E5** | + Path | (same rebuild) | multi-hop path lift; expect ≈0 on text |
| **E6** | all verified signals, param-free best-of | none beyond above | ceiling of non-learned fusion |
| **E7** | + query-aware cooperative-gate controller | small (controller only) | does learned gating beat best-of? (L2-B machinery) |
| **E8** | E7 + expert dropout / usage-reg | small | anti-collapse; guard splade-dominance |

**Gate:** the only non-trivial new compute is the **C/P50 substrate build** (E0–E3 reuse existing caches/heads; E4–E5 need the relsig rebuild). Recommend running **E0–E3 first** (nearly free) to confirm the offset/mixture signals hold on the locked interface before paying for the relation/path rebuild.

---

## Status flags
`L2_AUDIT_COMPLETE = YES` · `L2_TRAINING_STARTED = NO` · `L2_TARGET_ARCHITECTURE = PROPOSED` · `L3_BOUNDED_EXPANSION = FUTURE_EXPERIMENT`

**No GPU training will start until you review this audit.**
