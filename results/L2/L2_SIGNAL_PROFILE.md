# L2 Signal Profile — C/P50 corpus (Step-9)

**Scope of this document:** a *profile only*. No model was trained; no encoder was re-run; no full-corpus
recompute; no Hotpot/SQuAD/full-MetaQA; no L3 expansion; no topology/routing change. All numbers derive from
the frozen C/P50 corpus (`data/l2_corpus/{2wiki_clean,musique_clean}/train`) and cached artifacts.

- Executed profile (DENSE/SPLADE/OFFSET/MIXTURE): **300 train queries/dataset**.
- Graph + relation/path structural probe: **120 train queries/dataset**.
- Ran on **CPU** (no CUDA in this env) → all timings are conservative upper bounds.
- Machine-readable: `results/L2/L2_SIGNAL_PROFILE.json` (full per-signal records + decision gates).

---

## 1. SPLADE signal completeness — **RESOLVED, no re-encode**

The user's central question. **Reusable sparse query vectors already exist** — no re-encoding was required.

| artifact | where | what |
|---|---|---|
| doc matrix | `data/ukb_storage/<ds>/splade_doc_embs.pkl` | CSR `n_docs × 30522` + `id_to_idx`; `DOC_ROW_ORDER_ALIGNMENT=PASS` ⇒ `doc_mat[cand_id]` is the correct row |
| query vectors | `data/ukb_storage/<ds>/gte_qwen/splade_q_shards/q_shard_*.npz` | scipy CSR, sharded by 8192; **concatenation == all-order EXACTLY** (verified both datasets) ⇒ `Q_all[row_all]` is the query's sparse vector |

**Method (exactly the user's preferred path):** `SPLADE_SCORE(q,d) = doc_mat[cand] · Q_all[row_all]ᵀ`, restricted to
the ~5k P50 candidates. Not a full-corpus recompute.

**Alignment parity gate** (`argmax(doc_mat·q) == splade_top200_all[row_all,0]`):
- musique: **60/60**.
- 2wiki: **59/60** — the single "miss" is an **exact score tie** (`27.249336 == 27.249336`; my `argmax` vs the
  cache's `argpartition` break ties differently). Alignment is **bit-exact**.

**Completeness finding (the bias the user flagged).** The corpus currently stores `splade_rank` = *global top-200
membership only*. Consequently **~97–98% of every ~5k scope had `splade_rank == −1`** (no SPLADE information):

| dataset | scope frac with rank==−1 (no info before) | scope frac now nonzero exact score |
|---|--:|--:|
| 2wiki | 0.985 | 0.960 |
| musique | 0.971 | 0.944 |

Exact scope scoring gives **every** candidate a real score → removes the "not-in-top-200 ⇒ 0" bias. Gold sits at
**median within-scope SPLADE rank 0** on both datasets. **Cost ≈ 0.0045 s / 1000 candidates** (CPU); full
train scope for both pilots ≈ 13 min CPU, cache at 2 B/pair (f16 score + int16 within-scope rank).

> **`SPLADE_P50_SCORE_COMPLETE = YES`** — and it is the first real information lift unlocked by this profile,
> because ~97% of the scope previously carried no lexical signal at all.

---

## 2. Per-signal profile (300 train queries/dataset)

Gold "best-rank" = lowest within-scope rank of any in-scope gold. All four cheap experts place gold at
**median rank 0** on 2wiki/musique train (train is easy/saturated for these experts — consistent with
`[[crag-valproxy-shapley-findings]]`; the discrimination that matters lives on the *negatives* and on
harder datasets).

| signal | impl path | inputs | out dim | device | sec/1k | B/pair | cache | coverage | gold best-rank (med / p90) |
|---|---|---|--:|---|--:|--:|:--:|---|--:|
| **DENSE** | cached `nodes·queries_all` cosine (in corpus) | dense embeds | 1 | GPU-precomp | 0.0003 | 2 | ✓ | 100% scope | 0 / 0–2 |
| **SPLADE** | exact sparse dot `doc_mat[cand]·Q[row]` (§1) | cached SPLADE CSRs | 1 | CPU sparse | 0.0045 | 2 | ✓ | 100% scope (exact) | 0 / 0–2 |
| **OFFSET** | cached head → `pred·docemb[cand]` | dense embeds + head | 1 | GPU | 0.003 | 2 | ✓ | 100% scope | 0 / 0–1 |
| **MIXTURE-K8** | cached head → `max_k pred_k·docemb[cand]` | dense embeds + head | 8 | GPU | 0.004–0.008 | 2 | ✓ | 100% scope | 0 / 0–1 |
| **GRAPH (cheap)** | C families: degree / 2-hop / components | reconstructed CSRs | ~5 | CPU sparse | 0.04 | 2 | ✓ | 100% scope | — (structural) |
| **RELATION** | `cos(E(q), E(connecting-sentence))` | master_nodes + A + **gte-Qwen encode** | 1 | GPU encode | *build* | sparse | build-once | see §3 | — |
| **PATH** | `max_p cos(E(q), E(path-ctx))` | + bounded path enum + **encode** | 1 | GPU encode | *build* | sparse | build-once | ~0 @P50 | — |

**OFFSET/MIXTURE checkpoints** used here (`head_06a9fd3a…`, `head_32404bf9…`) are *available caches for the
compat/runtime probe only* — **not** the canonical L2 checkpoint. The mechanism (`pred · docemb[cand]`) is a
dense-family dot ⇒ **pool-agnostic**, so both are C/P50-compatible by construction; the empirical run confirms
runtime and gold reachability. Per the reuse-rule, when they enter an ablation they must be **retrained on the
C/P50 pool**, never carried over as a signal claim from the old dense∪NER pool.

---

## 3. RELATION / PATH — exact semantics + C/P50 feasibility (encode-free probe)

Both are **candidate-indexed** (doc → title-entity → edge/path from a detected query topic) ⇒ **pool-agnostic**:
they *can* score any candidate set including C/P50. What they need that the cheap experts don't is a
**construction + gte-Qwen text-encode** of the relation-/path-sentence vocabulary — the expensive, not-yet-cached
step. That is the whole reason they are the "deferred expensive features."

**RELATION (passage corpora).** `relation_raw(q,d) = cos(E(q), E(sentence in the topic passage that names the
candidate title))` — the title-mention edge *is* the relation; it is a sentence, **not** a KB label. (KB variants:
webqsp = Freebase relation text; metaqa = parsed doc facts — out of scope for this text pilot.)
Structural coverage upper bound at C/P50 (frac of scope with ≥1 in-scope title-mention edge):
**2wiki 0.970, musique 0.983.** High → relation *could* score almost the whole scope, but only after the encode.
→ **`CAN_RELATION_RUN_ON_C_P50 = YES, with a build`** (gated behind E4; train-subset first).

**PATH.** `path_raw(q,d) = max over bounded k-hop paths of cos(E(q), E(path-context))`, scored only for
*path-only* candidates (no 1-hop edge). Structural coverage at C/P50 = **0.000 on both datasets**: ~97–98% of the
scope is already 1-hop reachable within scope, so 2-hop title-mention adds **essentially no new scorable
candidates**. Matches `[[musique-graph-reachability-ceiling]]`. → **`CAN_PATH_RUN_ON_C_P50 = YES mechanically,
but near-empty at P50`** — **defer/drop on the text pilot**; revisit only on KB (webqsp/metaqa) where multi-hop is
the actual regime.

---

## 4. Cheap GRAPH features (C families)

Reconstructed C edge families load in ~5 s (musique) / ~29 s (2wiki), then per-scope features are cheap
(~0.04 s/1k). At P50 the scope is almost fully inside the C graph:

| dataset | mean global deg (struct / NER / C) | C in-scope membership | C largest component |
|---|---|--:|--:|
| 2wiki | 3.8 / 15.9 / 22.1 | 1.000 | 0.999 |
| musique | 7.5 / 14.6 / 24.0 | 1.000 | 1.000 |

Cheap, cacheable, C/P50-compatible: per-candidate degree (per family), in-scope degree, 2-hop reach, induced
component id, seed geodesic. Use as **reranker input features**, not as a standalone expert.

---

## 5. Full-scope vs train-subset feature policy

| expert | policy | why |
|---|---|---|
| DENSE | **A** full scope now | already in corpus, 2 B/pair |
| SPLADE | **A** full scope now | exact, cheap (~13 min CPU); replaces top-200-only rank |
| OFFSET | **A** full scope (from canonical retrained head) | GPU-cheap dot |
| MIXTURE | **A** full scope | same |
| GRAPH (cheap) | **A** full scope | ~0.04 s/1k |
| RELATION | **C** train-subset first | encode is the cost; materialize full only if E4 lifts |
| PATH | **D** defer/drop (text pilot) | ~0% coverage at P50 |

*(A=full now, C=train-subset first, D=defer/drop.)*

---

## 6. Decision gates

| flag | value |
|---|---|
| `SPLADE_P50_SCORE_COMPLETE` | **YES** |
| `OFFSET_C_P50_COMPATIBLE` | **YES** |
| `MIXTURE_C_P50_COMPATIBLE` | **YES** |
| `RELATION_C_P50_COMPATIBLE` | **YES — with construction+encode build** |
| `PATH_C_P50_COMPATIBLE` | **YES mechanically — but near-empty at P50 (defer to KB)** |
| `GRAPH_FEATURES_C_P50_COMPATIBLE` | **YES (cheap)** |
| `EXPENSIVE_FEATURE_PROFILE_COMPLETE` | **YES** |
| `SAFE_TO_BEGIN_E0_E3` | **YES** |
| `SAFE_TO_BEGIN_RELATION_PATH_ABLATIONS` | RELATION = **YES after E4 build (train-subset)**; PATH = **NO on text pilot** |

---

## 7. Proposed training order (proposal only — not executed)

1. **E0** DENSE-only reranker on C/P50 train-subset — sanity that the corpus trains.
2. **E1** DENSE + **exact-scope SPLADE** fusion — the first real lever (SPLADE completeness now unlocked).
3. **E2** + OFFSET — **retrained on the C/P50 pool** (never reuse an old-pool head as a signal).
4. **E3** + MIXTURE-K8 (retrained on C/P50). *(E0–E3 = cheap cached-family experts.)*
5. **E3b** + cheap GRAPH features (degree/component) as reranker inputs.
6. **E4** BUILD RELATION (construction+encode) on train-subset → add relation expert; keep only on real lift over E3.
7. **E5** PATH — **skip on text pilot**; schedule only when the corpus extends to webqsp/metaqa.
8. **Gating** soft per-query controller over retained experts (crag_fusion/gates), learned on TRAIN regimes,
   checkpoint-selected on VAL, reported on TEST once.

**Reuse-rule reminder:** every expert must be (re)trained on the C/P50 pool before entering an ablation — never
mix candidate pools and call the difference a signal effect.

---

**STOP after profile.** No training started; no expensive feature built. Awaiting review before E0.
