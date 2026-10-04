# W0 — Full / Open-World Corpus Substrate Manifest

**Scope:** preparation-only inventory. No evaluation, no model selection, no TEST
inspection, no architecture change. Runs in parallel with G1; touches no clean
`l2_corpus` / C11a / universal-head / L1 artifact.

**Machine-readable twin:** `results/WORLD/W0_CORPUS_MANIFEST.json` (regenerable via
`scratchpad/_w0_build_manifest.py`, which re-reads every manifest below — no hand
transcription).

---

## 0. Clean vs World vs Universe — verified counts

Three distinct corpus layers are recorded explicitly (never conflated). All counts
read from the actual `document_manifest.json` / `variant_C/stats.json` files.

| Dataset | Kind | **Clean** (controlled substrate) | **World / Open** (retrieval corpus) | **Universe** (full graph) |
|---|---|---|---|---|
| 2Wiki | TEXT | 65,865 | **398,354** | **5,989,847** |
| HotpotQA | TEXT | 507,494 | **5,233,329** (FullWiki) | — |
| MuSiQue | TEXT | 13,672 | **117,533** | — |
| SQuAD | TEXT | 19,029 | **20,233** | — |
| WebQSP | KB | 781,485 (legacy) | **1,316,466** (RoG) | — |
| MetaQA | KB | 40,151 | **43,234** | — |

Every number matches the expected inventory in the task spec (2Wiki 65,865 / ≈398k /
≈5.99M; Hotpot 507,494 / ≈5.23M; MuSiQue 13.7k / ≈117.5k; SQuAD 19.0k / ≈20.2k;
WebQSP 781,485 / ≈1.316M; MetaQA 40,151 / ≈43.2k). **Counts LOCKED.**

Clean paths: `scratchpad/ablation_qwen/<clean>/variant_C/`.
World paths: `data/canonical/<ds>/`. Universe: `data/canonical/2wiki_universe/`.

---

## 1. World-corpus ingredient status (per Task 1)

Legend: ✅ present & complete · ⬜ not built · (n) = n_items.

| Dataset (world) | docs materialized | query/gold map | **Qwen dense** | **SPLADE** | Dense FAISS index | struct graph | NER graph | Qwen-kNN edges | topo-C partitions |
|---|---|---|---|---|---|---|---|---|---|
| 2Wiki (398,354) | ✅ | ✅ | ✅ (398,354) | ✅ (398,354) | ⬜ | ✅ 359,549 | ✅ | ⬜ | ⬜ |
| 2Wiki_universe (5,989,847) | ✅ | n/a (graph univ) | ✅ (5,989,847) | ✅ (5,989,847) | ⬜ | ✅ 28,963,600 | ✅ | ⬜ | ⬜ |
| HotpotQA (5,233,329) | ✅ | ✅ | ✅ (5,233,329) | ✅ (5,233,329) | ⬜ | ✅ 15,367,541 | ✅ | ⬜ | ⬜ |
| MuSiQue (117,533) | ✅ | ✅ | ✅ (117,533) | ✅ (117,533) | ⬜ | ✅ 2,744,076 | ✅ | ⬜ | ⬜ |
| SQuAD (20,233) | ✅ | ✅ | ✅ (20,233) | ✅ (20,233) | ⬜ | ✅ 874,190 | ✅ | ⬜ | ⬜ |
| WebQSP (1,316,466) | ✅ | ✅ | ✅ (1,316,466) | ✅ (1,316,466) | ⬜ | ✅ 3,791,303 | n/a (KB) | ⬜ | ⬜ |
| MetaQA (43,234) | ✅ | ✅ | ✅ (43,234) | ✅ (43,234) | ⬜ | ✅ 133,582 | n/a (KB) | ⬜ | ⬜ |

**Encoders (frozen, unchanged):** dense = `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
(dim 1536, fp16); SPLADE = `naver/splade-cocondenser-ensembledistil`. Every dense
and SPLADE `docs` manifest reports `complete: true` with `rows_covered == n_items`
and `source_sha256` matching the corpus `documents.jsonl`. Query dense encodings are
complete for all datasets (2Wiki 192,606 · Hotpot 105,257 · MuSiQue 24,814 · SQuAD
142,192 · WebQSP 4,737 · MetaQA per-hop 116,045 / 148,724 / 142,744).

> **Headline:** the expensive GPU work — encoding **5.99M + 5.23M + 1.32M** node
> corpora with Qwen dense **and** SPLADE — is **already done and complete on disk.**
> W0's remaining compute is comparatively light: FAISS/exact index, Qwen-kNN edge
> family + combined C graph, and C partitions.

---

## 2. Local preservation (Task 2)

All raw world-corpus sources required to rebuild the substrate are **local**:

| Dataset | `data/original/` size | Key raw source (referenced by manifest) |
|---|---|---|
| 2Wiki | 2.8 GB | `v1.0_ids_april2021/para_with_hyperlink.zip` (1.9 GB) |
| HotpotQA | 1.9 GB | `fullwiki_corpus/enwiki-20171001-...withlinks-abstracts.tar.bz2` (1.55 GB) |
| MetaQA | 135 MB | `kb.txt`, `entity/kb_entity_dict.txt`, `{1,2,3}-hop/` |
| MuSiQue | 287 MB | official train/dev/test paragraphs |
| SQuAD | 45 MB | official train+dev contexts |
| WebQSP | 523 MB | `rog_webqsp/{train,validation,test}-*.parquet` |

Derived world artifacts (documents.jsonl, graph TSVs, Qwen/SPLADE shards) are also on
local disk under `data/canonical/`. Modal volumes are compute cache only.

**`WORLD_RAW_INPUTS_LOCAL = YES`**

---

## 3. Canonical world IDs (Task 3)

Each world `documents.jsonl` carries an explicit canonical `id` (row order is **not**
the ID system). Schemes recorded in each `document_manifest.json`:

- **2Wiki (open):** title-level dedup id over context paragraphs (unique_ids 398,354).
- **2Wiki_universe:** `2wu:<curid>` (curid-keyed, 5,989,847 unique curids).
- **HotpotQA:** wiki curid (5,233,329 unique curids).
- **MuSiQue / SQuAD:** paragraph text-hash-dedup ids.
- **WebQSP:** Freebase entity id (RoG subgraph node), 1,316,466 unique.
- **MetaQA:** KB entity id (name), 43,234 unique.

The shard `ids_*.json` files (alongside `shard_*.npy`) preserve **row → world_node_id**
alignment for every encoder; `docs/manifest.json` records shard layout + row coverage.
`STATUS: canonical IDs frozen for all 7 corpora.`

---

## 4. Clean ↔ World mapping (Task 4)

- **2Wiki:** `data/canonical/2wiki_universe/crosswalk_to_retrieval_view.json` maps the
  398k open view into the 5.99M universe by title — **398,354 / 398,354 titles found
  (coverage 1.0)**. Clean 65,865 ⊂ open 398,354 (title identity).
- **Other datasets:** the clean substrate was built from the same canonical
  `documents.jsonl`, so `clean_node_id → world_node_id` is a **subset/identity by
  canonical id**, but an explicit materialized crosswalk file does **not yet exist**
  (cheap, local, CPU-only to emit). This is the one gap in Task 4.

`CLEAN_TO_WORLD_MAPPING_COVERAGE`: 2Wiki = 1.0 (materialized). Others = derivable
(identity on canonical id) — **materialize before clean-vs-world comparison.**

---

## 5. Query / gold alignment (Task 5)

From each `query_manifest.json` (train/dev; test gold hidden or reserved):

| Dataset | split | n (answerable) | all-gold mapped | partial | none |
|---|---|---|---|---|---|
| 2Wiki | train / dev | 167,454 / 12,576 | 100% / 100% | 0 | 0 |
| HotpotQA | train / dev | 90,447 / 7,405 | 100% / 100% | 0 | 0 |
| MuSiQue | train / dev | 19,938 / 2,417 | 100% / 100% | 0 | 0 |
| SQuAD | train / dev | 130,319 / 11,873 | 100% / 100% | 0 | 0 |
| WebQSP | train / test | 3,072 / 1,628 (answerable) | 100% / 100% | 0 | 0 |
| MetaQA 1hop | train/dev/test | 96,106 / 9,992 / 9,947 | 100% | 0 | 0 |
| MetaQA 2hop | train/dev/test | 118,980 / 14,872 / 14,872 | 100% | 0 | 0 |
| MetaQA 3hop | train/dev/test | 114,196 / 14,274 / 14,274 | 100% | 0 | 0 |

Gold is **unambiguous** (all answerable gold resolves to a canonical world doc id;
zero partial, zero unmapped). `GOLD_WORLD_MAP_READY = YES` for every dataset.
*(2Wiki/MuSiQue/Hotpot test gold hidden by design; never inspected.)*

---

## 6. Topology-C definition (for Tasks 9–10)

`variant_C/stats.json` records `"synthetic_qwen_edges": "reused from A"` →
**C = structural_native ⊕ NER (text only) ⊕ Qwen semantic-kNN**, then balanced
~100-node partitioning (2Wiki clean: 65,865 nodes → 658 parts, sizes 97–103,
edge-cut ratio 0.299). This definition is **not redefined** here.

Edge families that genuinely exist per world corpus:
- **Text (2Wiki, Hotpot, MuSiQue, SQuAD):** structural + NER present on disk; Qwen-kNN
  **derivable from the already-complete world dense embeddings** (not yet emitted).
- **KB (WebQSP, MetaQA):** structural KB triples present (WebQSP 6,094 relations;
  MetaQA 9 relations); no NER family (consistent with clean KB C); Qwen-kNN derivable.

So building world-C needs only: (a) emit Qwen-kNN edges from existing dense shards,
(b) fuse into `graph.pt`, (c) partition. No re-encoding.

---

## 7. Dense index — cost report (Task 7, STOP-gate)

No FAISS index exists yet for any world corpus. Exact `IndexFlatIP` stores float32
internally → RAM = N × 1536 × 4 B:

| Corpus | N | Exact FlatIP RAM (fp32) | fp16 sharded-topk RAM (streamed) |
|---|---|---|---|
| 2Wiki_universe | 5,989,847 | **36.8 GB** | 18.4 GB on disk, streamed in shards |
| HotpotQA | 5,233,329 | **32.2 GB** | 16.1 GB |
| WebQSP | 1,316,466 | 8.1 GB | 4.0 GB |
| 2Wiki open | 398,354 | 2.45 GB | 1.2 GB |
| MetaQA | 43,234 | 0.27 GB | 0.13 GB |
| MuSiQue | 117,533 | 0.72 GB | 0.36 GB |
| SQuAD | 20,233 | 0.12 GB | 0.06 GB |

**Feasibility:** exact search is feasible for **all** corpora. The two multi-million
corpora (5.99M, 5.23M) exceed a single 24 GB GPU as a monolithic fp32 flat index, but
the project's established retrieval path is **sharded GPU torch top-k over the fp16
shards** (parity 1.0 vs exact; see prior mining work) — which streams shard-by-shard
and never materializes the full matrix, so exact top-k is achievable within one A10G.

**Decision required before building (no silent algorithm switch):**
- **Option A (recommended):** exact sharded fp16 GPU top-k for the two 5M corpora +
  exact `IndexFlatIP` for the ≤1.32M corpora. Keeps exact semantics everywhere.
- **Option B:** `IndexFlatIP` everywhere on a high-RAM CPU box (needs ~37 GB for
  2Wiki_universe).
- **Option C (only if A/B rejected):** approximate (IVF/HNSW) — this **changes
  methodology** and is *not* adopted without explicit sign-off.

Per Task 7 I am **stopping here for the index approach** rather than silently choosing.

---

## Final gates (per dataset)

| Gate | 2Wiki | Hotpot | MuSiQue | SQuAD | WebQSP | MetaQA |
|---|---|---|---|---|---|---|
| WORLD_RAW_READY | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| WORLD_ID_MAP_READY | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CLEAN_WORLD_MAP_READY | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| GOLD_WORLD_MAP_READY | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| WORLD_QWEN_READY | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| WORLD_DENSE_INDEX_READY | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| WORLD_SPLADE_READY | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| WORLD_GRAPH_C_READY | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| WORLD_PARTITIONS_READY | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| WORLD_RETRIEVAL_READY | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

**`WORLD_RAW_INPUTS_LOCAL = YES`**
**`SAFE_TO_START_FULL_WORLD_EXPERIMENT = NO`** (dense index, Qwen-kNN/​C graph, and C
partitions absent for every corpus) — and per spec, **STOP before starting** regardless.

---

## Remaining W0 work to reach retrieval-ready (Task 11 boundary)

Ordered by the spec's expense priority. Nothing below builds L2 P50 matrices or
Offset/Mixture/Relation scores (those stay lazy until the world experiment is
authorized).

1. **Dense index** — after approach sign-off (§7): exact sharded top-k / FlatIP.
2. **Qwen-kNN edge family** — derive from existing world dense shards (GPU, no re-encode).
3. **World C `graph.pt`** — fuse structural ⊕ NER ⊕ Qwen-kNN per the frozen C recipe.
4. **World C partitions** — balanced ~100-node partition; record node/edge/part
   stats, orphans, time, memory (Task 10).
5. **Clean↔world crosswalks** for Hotpot/MuSiQue/SQuAD/WebQSP/MetaQA (cheap, local).
6. **Priority order:** Hotpot FullWiki 5.23M → WebQSP 1.316M → 2Wiki open/universe →
   MuSiQue → MetaQA → SQuAD.

**No interference with G1:** no clean `l2_corpus`, C11a, universal-head, L1, or TEST
artifact is touched by any step above.
