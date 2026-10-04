# L2 Relation — Canonical gte-Qwen2 build (TRAIN+VAL) + 5-expert matrix freeze

Builds the **canonical** masked Relation expert with `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (MiniLM was
proxy/diagnostic only), on **TRAIN+VAL** for `2wiki_clean` + `musique_clean`, then freezes the five-expert
matrix for exact Shapley. **VAL-only decision. TEST never touched. L1 untouched. Controller NOT trained,
Shapley NOT run.** Code: `scratchpad/l2_relation_qwen.py`, `scratchpad/modal_encode_relsents.py`.

The tie-leak fix from the previous phase is LOCKED: `RELATION_MASK_SEMANTICS`, `MASKED_RELATION_RRF`; the
legacy unmasked path is HISTORICAL only. The encoder ablation there is **`RELATION_ENCODER_SENSITIVITY =
LOW_ON_TESTED_ORDERINGS`** (sensitivity evidence over 4 probes, *not* a universal proof) — this phase settles
it empirically with the real encoder.

---

## 1. Query embeddings reused — `RELATION_Q_EMBED_REUSE = PASS`

`encoder_swap.eq()` encodes `normalize(gteQwen(query_instruction + text))`; `queries_all.npy` is exactly that
over the split's queries. Verified against `meta.json`:

| field | value |
|---|---|
| `Q_MODEL` | Alibaba-NLP/gte-Qwen2-1.5B-instruct |
| `Q_PROMPT` | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ` |
| `Q_POOLING` | last-token (gte-Qwen2 default) |
| `Q_NORMALIZATION` | L2 (unit) |
| `Q_DIM` | **1536** |

Queries were **not** re-encoded.

## 2. Only unique connecting sentences encoded (canonical document semantics)

`meta.json` has `doc_prefix = ""` → canonical **passages are encoded RAW**. The connecting sentence is a
passage, so:

- `QUERY_ENCODING_TEMPLATE` = `normalize(gteQwen(Q_PROMPT + query))` (instructed; reused cache).
- `RELATION_SENTENCE_ENCODING_TEMPLATE` = `normalize(gteQwen(sentence))` — **RAW, no instruction** (matches
  `nodes.npy` document encoding).

**Corrected historical bug:** the old `kg_relsig._enc` applied the *query* instruction to the connecting
sentence. That is non-canonical (gte-Qwen2 is asymmetric: instructed query ↔ raw passage) and was **not**
preserved — sentences are encoded raw here, consistent with the canonical document side.

`N_QWEN_SENTENCES_ENCODED = 19,235` unique (dedup across all 4 splits; 2wiki train 11,388 / val 3,464;
musique train 4,101 / val 1,617). One deduplicated Modal job, gte-Qwen2 fp16 on A10G, `max_seq_length=256`:
**60.3 s encode (3.1 ms/sent)**, 35 s model load. Deterministic `sentence_text → sha1 → global id` map.

## 3. Mask is encoder-independent — `QWEN_RELATION_MASK_PARITY = PASS`

Eligibility is pure topology (title-mention adjacency + `find_topics`); the encoder never changes it.
Verified bit-identical to the MiniLM-phase mask on VAL: 2wiki 5,284 / musique 10,633 eligible pairs, exact
array equality. `mask=0 ⇒ ABSTAIN` (no rank, no RRF vote) is preserved.

---

## 4. Canonical Qwen VAL result (COND_ANY_GOLD_IN_SCOPE), E3+relation vs E3

| metric | 2wiki E3 | **2wiki E3+Qwen-rel** | Δ | musique E3 | **musique E3+Qwen-rel** | Δ |
|---|---|---|---|---|---|---|
| MRR | 0.929 | 0.910 | −1.9 | 0.921 | 0.904 | −1.7 |
| R@5 | 0.647 | **0.747** | **+10.0** | 0.820 | 0.829 | +0.9 |
| R@10 | 0.764 | 0.848 | +8.4 | 0.902 | 0.907 | +0.5 |
| R@50 | 0.912 | 0.965 | +5.3 | 0.979 | 0.980 | +0.2 |
| ALL@10 | 0.523 | **0.666** | **+14.2** | 0.785 | 0.795 | +1.0 |
| ALL@20 | 0.687 | 0.793 | +10.6 | 0.904 | 0.908 | +0.4 |
| ALL@50 | 0.767 | **0.891** | **+12.4** | 0.952 | 0.956 | +0.4 |

- **2wiki:** large deep-recall / multi-gold lift; small top-MRR cost (equal-weight-RRF artifact → soft
  controller). **musique:** NEUTRAL-positive.

### 4a. `QWEN_VS_MINILM_DELTA` — the encoder barely matters, now confirmed with the real encoder

| | max |Δ(Qwen − MiniLM-masked)| over all 8 metrics |
|---|---|
| 2wiki | **0.0014** (ALL@50) |
| musique | **0.0001** |

Every metric agrees to ≤0.14pp — **even though** the two encoders order the eligible sets differently
(Kendall-τ of eligible ordering ≈ **0.61** 2wiki / **0.70** musique; 747 / 563 VAL queries are *sole*-eligible
and thus encoder-invariant by construction). This is the empirical confirmation of the prior sensitivity
evidence: the **mask (topology) carries relation's value**, not the cosine tiebreak among ~2 candidates. The
`LOW_ON_TESTED_ORDERINGS` note is now backed by the canonical encoder — still stated as an empirical result,
not a universal proof.

## 5. GOLD-level analysis (primary; query-best is confounded by the easy first-hop gold)

| | 2wiki val | musique val |
|---|---|---|
| eligible golds | 2073 | 573 |
| `GOLD_HURT@{5,10,20,50}` | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |
| eligible golds base-missed (min base rank ≥50) | 24.1% | 2.8% |
| `GOLD_RESCUE@50` (of which base-missed ≥50) | 503 (**319**) | 16 (13) |
| `GOLD_RESCUE@10` (base-missed ≥50) | 698 (66) | 68 (2) |

On **2wiki**, canonical Qwen relation recovers **319 second-hop bridge golds that all four base experts place
at rank ≥50** into the top-50 (66 into top-10), with **zero gold demotions at any K**. On **musique** the
eligible golds are already easy for the base (median base rank 1) and the hard multi-hop chains are not
relation-eligible (graph lacks them) → little to add, no harm.

For every relation-eligible gold the matrix records `dense_rank, splade_rank, offset_rank, mixture_rank,
E3_rank, relation_masked_rank, E3+relation_rank` (for the controller's future rescue supervision).

---

## 6. `RELATION_VALUE_FINAL = MIXED` → RESCUE_SPECIALIST (2wiki) / NEUTRAL (musique)

Preserved under the canonical encoder. Relation abstains on 99.8% of candidates, so it cannot rank a list
alone — its sparse vote lifts eligible **bridge** golds *inside* the fusion (cooperative realization of a
genuine rescue). Gate it **ON** where the query has a graph bridge (2wiki), **OFF** where it does not
(musique). Fixed equal-weight fusion is not the way to add it (it costs top-MRR on already-solved queries).

## 7. Frozen five-expert set (`FINAL_EXPERT_SET_READY = YES`)

`EXPERT_1 = Dense` · `EXPERT_2 = Exact SPLADE` · `EXPERT_3 = Offset` · `EXPERT_4 = Mixture` ·
`EXPERT_5 = Masked Qwen Relation`. Cheap graph stays dropped from L2 ranking; Path stays deferred on the text
pilot. Matrix + manifest: `results/L2/L2_EXPERT_MATRIX_MANIFEST.json`.

---

## Decision gates

| gate | verdict |
|---|---|
| `RELATION_Q_EMBED_REUSE` | **PASS** |
| `N_QWEN_SENTENCES_ENCODED` | **19,235** |
| `QWEN_RELATION_MASK_PARITY` | **PASS** (both VAL) |
| `QWEN_RELATION_VAL_RESULT` | 2wiki R@5 +10.0 / ALL@10 +14.2 / ALL@50 +12.4 / MRR −1.9; musique neutral; GOLD_HURT=0 both |
| `QWEN_VS_MINILM_DELTA` | ≤ **0.0014** every metric (τ_order ≈ 0.6 → mask carries value) |
| `QWEN_GOLD_RESCUE` | 2wiki @50 = 503 (319 base-missed), @10 = 698 (66) ; musique @50 = 16 (13) |
| `QWEN_GOLD_HURT` | **0 at every K, both datasets** |
| `RELATION_VALUE_FINAL` | **MIXED** — RESCUE_SPECIALIST (2wiki) / NEUTRAL (musique) |
| `FINAL_EXPERT_SET_READY` | **YES** {Dense, Exact-SPLADE, Offset, Mixture, Masked-Qwen-Relation} |
| `EXPERT_MATRIX_TRAIN_READY` / `EXPERT_MATRIX_VAL_READY` | see manifest |
| `SAFE_TO_START_EXACT_SHAPLEY` | **YES** once matrix frozen (TRAIN+VAL relation built) |
| `SAFE_TO_START_CONTROLLER` | **NO** (waits for Shapley target phase) |
