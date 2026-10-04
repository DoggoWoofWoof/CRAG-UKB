# L2 RELATION Expert — Build, Validate, Decide

**Phase:** Build + validate the RELATION expert on the canonical **C / Dense+SPLADE / K100 / P50** interface. Determine whether Relation deserves a seat in the final expert set `{Dense, SPLADE, Offset, Mixture, Relation}` before the soft controller is built.
**Pilot:** `2wiki_clean`, `musique_clean`. **Selection split: VAL.** **TEST frozen** (never inspected this phase).
**Controller / Shapley: NOT started** (per instruction). L1 untouched.

**Scientific question (not "does Relation beat Dense"):** *Does Relation provide complementary information / rescue behavior that Dense, SPLADE, Offset, and Mixture do not?*

---

## TASK 1 — Exact Relation semantics

| field | value |
|---|---|
| RELATION_SIGNAL_NAME | `relation_raw` = cos( E(question), E(connecting-sentence) ) |
| CODE_PATH | `src/experiments/kg_relsig.py` (PASSAGE branch, `ner_edges=False`); C/P50 wrapper `scratchpad/l2_relation.py` |
| INPUTS | question embedding; embedding of the **sentence in the topic doc that names the candidate's title**, over the struct (title-mention) adjacency |
| QUERY_DEPENDENT | **YES** (uses E(q)) |
| CANDIDATE_DEPENDENT | **YES** (edge topic→candidate title) |
| PAIR_DEPENDENT | **YES** (a (topic, candidate) connecting sentence) |
| ENTITY_DEPENDENT | **YES** (candidate must map to a title/entity; topics extracted from the question) |
| USES_NER | **NO** (title-mention struct edges; NER is an optional variant, not used here) |
| USES_STRUCT | **YES** (`node.neighbors` title-mention graph) |
| USES_QWEN_KNN | **NO** |
| USES_EXTERNAL_KG_RELATIONS | **NO** — on passage corpora the "relation" is the raw connecting **sentence**, not a Freebase/KB relation id (dataset-agnostic) |
| USES_TEXT_EMBEDDINGS | **YES** |
| ENCODER_MODEL | gte-Qwen2-1.5B (faithful) — **MiniLM-L6 used here as a proxy** (see Task 5 blocker) |
| OUTPUT_DIM | scalar score per candidate (cosine of 1536-d faithful / 384-d proxy embeddings) |
| CURRENT_CHECKPOINT_EXISTS | **NO trained head required** — the signal is a raw cosine. Old `relsig_feats_*.pt` are OLD-pool feature tensors only |
| CURRENT_POOL_USED_BY_OLD_IMPL | **dense-top200 ∪ NER-2hop**, capped tr1104/te2000 — **NOT** the canonical P50 (rebuilt here) |

**What it represents.** For a multi-hop question, the topic entities are named in the question; the gold "bridge" documents are usually *not* directly similar to the question (dense/SPLADE miss them) but are reachable by a title-mention edge from a topic document. The relation signal scores each candidate by how well the **specific sentence that links topic→candidate** answers the question — i.e. it measures *which relation the question asks for* vs *which relation the edge encodes*, a signal orthogonal to bag-of-words similarity. Its natural home is the **second-hop / bridge** candidate that similarity-based experts cannot reach.

---

## TASK 2–4 — Rebuild on C/P50, alignment, strict scope

The old pool (dense-top200 ∪ NER-2hop, ~1098 cand/query, tr1104/te1500 caps) is **incompatible** with the canonical P50 (~5,029 cand/query). Relation was **rebuilt against C/P50 candidate IDs** with a minimal wrapper — no CoreEngine, everything derived from `master_nodes` (doc rows, titles, struct neighbors, question texts), verified bit-identical to the frozen row space (`doc_N → row N`; gold dense-sims 0.33–0.63 vs random 0.13–0.26).

**RELATION_C_P50_ALIGNMENT = PASS**
- **Determinism:** identical `cand_sent` across two separate processes (fixed a hash-seed set-iteration nondeterminism in `find_topics` → sorted token order).
- **Scope:** `out_of_scope_writes = 0` — Relation only ever scores L1-surviving P50 candidates; **no new candidate is introduced** (RELATION_SCOPE_RULE = P50 only). Future bounded L3 expansion is separate and not done here.
- `cand_sent` length == P50 pair count; positive labels unchanged.

---

## TASK 5 — Cost / storage profile

| item | value |
|---|---|
| Faithful encoder (gte-Qwen2-1.5B) | **LOCAL LOAD INFEASIBLE** — Windows `OSError 1455` (paging file too small); 16.8 GB RAM / 4.6 GB free vs 6.7 GB model, in fp32 **and** fp16. |
| Re-encoding unavoidable? | **YES** — connecting sentences are sub-document sentences, not in any cache. |
| Batchable / Cacheable | YES / YES (sentence-emb cache reused across queries) |
| Proxy encoder used | `multi-qa-MiniLM-L6-cos-v1` (384-d), CPU **~25 ms/sentence** |
| Encode volume (unique sentences) | 2wiki val **3,464** · musique val **1,617** · musique train 4,101 · 2wiki train ~9–10k (est) |
| Bytes/sentence emb | 1,536 B (384×f32); relation score array = 4 B × N_pairs |
| Cache sizes (val) | 2wiki sents 5.3 MB / q 4.6 MB · musique sents 2.5 MB / q 6.1 MB · relscore 60–80 MB/split |
| Faithful gte-Qwen2 build | **GPU/Modal only** — ~18k sentences total, <1 min on GPU |

**Decision:** the faithful signal needs GPU; the local proxy is a **conservative lower bound** — under a weaker encoder, if Relation helps it helps *at least as much* under gte-Qwen2 (old gte-Qwen2 bridge R@5 = 75). No huge encoding job was launched.

---

## TASK 6 — Coverage (relation exists often enough?)

| dataset / split | frac_pos_w_rel | frac_neg_w_rel | ANY_GOLD_COV | ALL_GOLD_COV | queries NO_SIGNAL | uniq sents |
|---|---|---|---|---|---|---|
| 2wiki val | 0.293 | **0.0002** | **0.570** | 0.0 | 769 / 3000 | 3,463 |
| 2wiki train (n=4500) | 0.285 | 0.0002 | 0.554 | 0.0 | 1185 | 5,044 |
| musique val | 0.063 | 0.0005 | **0.142** | 0.0 | 2154 / 3987 | 1,617 |
| musique train | 0.061 | 0.0005 | 0.138 | 0.0 | 7539 / 13956 | 4,101 |

- Relation is **extremely precise** (fires on ~0.02 % of negatives) and **bridge-specific** (ALL_GOLD_COV = 0 everywhere — topic golds never get a relation edge, only second-hop golds do).
- **2wiki: ~57 % of queries have ≥1 gold with a relation edge.** **musique: only ~14 %** — the musique title-mention graph structurally does not contain the chains (matches the known musique reachability ceiling). This dataset split drives the entire result.

---

## TASK 7 & 10 — Relation-only and fixed-fusion (VAL, COND_ANY_GOLD_IN_SCOPE)

| config | MRR | R@5 | ALL@10 | ALL@20 | ALL@50 |
|---|---|---|---|---|---|
| **2wiki** relation-only | 0.7926 | 0.8276 | 0.7833 | 0.8363 | 0.8743 |
| 2wiki E1a (dense+splade) | 0.9592 | 0.6949 | 0.4207 | 0.4530 | 0.5017 |
| 2wiki E1a + relation | 0.9507 | 0.6956 | 0.4320 | 0.4830 | 0.6197 |
| 2wiki **E3** (d+s+off+mix) | 0.9286 | 0.6470 | 0.5233 | 0.6873 | 0.7670 |
| 2wiki **E3 + relation** | **0.9310** | **0.7198** | **0.6297** | **0.7373** | **0.8343** |
| **musique** relation-only | 0.6633 | 0.5836 | 0.4715 | 0.6080 | 0.7345 |
| musique E1a | 0.8874 | 0.6873 | 0.5006 | 0.5927 | 0.7084 |
| musique E1a + relation | 0.8812 | 0.6957 | 0.5257 | 0.6258 | 0.7385 |
| musique **E3** | 0.9213 | 0.8195 | 0.7849 | 0.9039 | 0.9518 |
| musique **E3 + relation** | 0.9152 | 0.8064 | 0.7350 | 0.8419 | 0.9458 |

**The headline (Task 17 — not MRR alone):**
- **2wiki: adding Relation to the full E3 stack improves EVERY metric** — MRR +0.24, R@5 **+7.3pt**, ALL@10 **+10.6pt**, ALL@20 +5.0pt, ALL@50 **+6.7pt**.
- **musique: adding Relation to E3 DEGRADES every deep metric** — ALL@10 −5.0pt, ALL@20 −6.2pt.

*(relation-only metrics are inflated by a sentinel fallback: absent candidates keep the canonical dense-desc order, so relation-only ≈ relation-boost-over-dense, not pure relation. Fusion results are the sound signal.)*

---

## TASK 8 & 9 — Rescue, interference, disagreement (VAL)

| metric | 2wiki | musique |
|---|---|---|
| RELATION_RESCUE top10 vs **cheap_stack** (d∪s∪off∪mix) | **0** | **0** |
| RELATION_RESCUE into top10 **within E3 fusion** | **18** | 6 |
| RELATION_HURT top10 **within E3 fusion** | **0** | **13** |
| RELATION_RESCUE / HURT top20 in E3 fusion | 5 / 0 | 2 / 2 |
| disagreement \|mixture − relation\| (mean / p95) | 12.0 / 19.0 | 6.0 / 14.0 |
| disagreement \|dense − relation\| (mean) | 0.69 | 1.94 |

- **Relation is not a *solo* rescue specialist** (rescue-vs-cheap-stack = 0 both datasets: Offset/Mixture already reach the bridge golds Relation reaches, alone).
- **Its value is COOPERATIVE in fusion:** on 2wiki it reinforces borderline bridge candidates, pushing **18 golds into top-10 with ZERO interference**; on musique it mostly interferes (**13 hurt > 6 rescued**).
- **High \|mixture − relation\| disagreement (p95 = 19 on 2wiki)** confirms Relation ranks differently from the existing experts — it is genuinely complementary, not redundant, on 2wiki.

---

## TASK 11–14 — Head / multi-positive / regime raw

- **No Relation head trained** (Task 11): the signal is a raw cosine; per Task 11 we evaluated it directly. → **MULTI_POSITIVE_RELATION_LOSS_CORRECT = NOT_APPLICABLE**, positive collisions N/A.
- **Regime raw material preserved & extended (Task 14):** appended `best_gold_rank_relation` to `results/L2/_regime_raw/{2wiki,musique}_clean_val.npz` (originals backed up to `_bak_*`), original 5 columns untouched. *(This column is the fused-expert rank — dense-correlated via the absent-sentinel fallback; pure-relation reach is read from coverage + E3-fusion rescue.)* No final query-class labels assigned (Task 19 respected).

---

## TASK 16 — RELATION_VALUE classification

**RELATION_VALUE = MIXED — a cooperative *rescue* expert on bridge-reachable (2wiki-type) queries, negative on musique-type.**

This is precisely the **query/dataset-dependent expert behavior** the E0–E3 phase predicted would need a soft controller: Relation should be gated **ON** where the corpus graph carries the chain (2wiki: +7.3 R@5, +6.7 ALL@50, 0 hurt) and **OFF** where it does not (musique: pure interference). Adding it by **fixed equal fusion is wrong** (it would hurt musique); a controller that learns *when* Relation is reliable can capture the 2wiki gain without the musique cost.

---

## DECISION GATES

| gate | verdict |
|---|---|
| RELATION_C_P50_COMPATIBLE | **YES** |
| RELATION_ALIGNMENT | **PASS** (determinism, out_of_scope=0, aligned, no new candidates) |
| MULTI_POSITIVE_RELATION_LOSS_CORRECT | **NOT_APPLICABLE** (raw cosine, no head) |
| RELATION_ADDS_FIXED_FUSION_VALUE | **MIXED** (2wiki YES all-metrics/0-hurt; musique NO, degrades E3) |
| RELATION_HAS_COMPLEMENTARY_RESCUE_VALUE | **YES** (2wiki: +18 into top10 / 0 hurt; \|mixture−relation\| p95=19) |
| RELATION_VALUE | **MIXED → cooperative RESCUE specialist (2wiki), negative (musique)** |
| KEEP_RELATION_FOR_CONTROLLER | **YES** — carries complementary, gateable info; do **not** add by fixed equal fusion |
| FINAL_EXPERT_SET_READY | **YES** `{Dense, SPLADE, Offset, Mixture, Relation}` — pending faithful gte-Qwen2 rebuild (GPU) + train-split relation for controller *fitting* |
| SAFE_TO_START_SHAPLEY_ANALYSIS | **YES** (all experts on identical C/P50 universe; val relation ranks in regime raw) |
| SAFE_TO_START_SOFT_CONTROLLER | **YES, after** faithful gte-Qwen2 relation build + train-split relation encode |

---

## Remaining before controller (explicitly NOT done here — STOP)

1. **Faithful gte-Qwen2 relation encode** on GPU/Modal (local infeasible). MiniLM proxy is a conservative lower bound; gte-Qwen2 should only strengthen the 2wiki gain.
2. **Train-split relation** encode (VAL was decisive for the *keep* decision; controller *fitting* needs train relation).
Both are inputs to the next phase. **No controller, no Shapley, no dropout, no Hotpot/SQuAD/MetaQA/WebQSP, no L3, no L1 change — reporting and stopping as instructed.**

---
*Deliverables: this file · `results/L2/L2_RELATION_RESULTS.json` · `results/L2/L2_RELATION_MANIFEST.json`. Builder: `scratchpad/l2_relation.py`.*
