# DOWNSTREAM REBUILD POLICY — after canonical_v1 locks

**Record only. Nothing in this document has been built, and nothing may be built from it until a separate task
authorises it.** `FULL_CANONICAL_V1_STATUS` is currently `READY_5_OF_6_SOURCE_CONTRACTS`
(`WEBQSP_STATUS = WEBQSP_BLOCKED_ON_FREEBASE_SOURCE`), so no lock exists.

---

## 2026-09-05 — 2WIKI DOWNSTREAM ARTIFACTS ARE INVALIDATED (nothing rebuilt)

The 2wiki canonical corpus was **replaced**: the 398,354-node pooled question-context union
(`corpus_hash ec7fbbe8cd783040…`) was rejected and superseded by the full official article universe,
**5,989,847 nodes**, `corpus_hash 81fa7d1a5d4bbb24…`, node ids `2wiki:c<curid>` instead of
`2wiki:<sha256(title)[:24]>`. Every 2wiki artifact whose definition depends on the node set, the node ids, or
the row ordering is therefore **INVALID**. None of them was rebuilt in this task, by instruction.

> **Re-pin 2026-09-06 — the 2wiki corpus hash quoted in this section has since moved, by a second and much
> smaller change.** `TEXTUALIZATION_REV 2` retextualized the 87,765 records whose canonical body was empty to
> their source-native title (the same general rule hotpotqa received on 2026-09-05), moving
> `corpus_hash 81fa7d1a5d4bbb24… → 94fe68b8d5f291f1…`. **Membership, node ids and node order did not change**,
> so nothing in the invalidation table below changes class: what was INVALID because of the 398,354 → 5,989,847
> replacement stays INVALID, and nothing newly becomes invalid. The one row this *does* touch is the dense /
> SPLADE reuse row: 87,765 encoder inputs changed from `""` to a title, so their rows must be re-encoded — the
> exact bill is measured in the Phase-3 table below, not inferred. The rev1 table is preserved at
> `data/final_canonical/_superseded_textualization_rev1/2wiki/`.

| 2wiki artifact family | State after the corpus change | Action (NOT taken here) |
|---|---|---|
| dense doc embeddings | **valid rows exist at full scale** — `data/canonical/2wiki_universe/encodings/dense/docs` covers all 5,989,847 with byte-identical encoder inputs (measured `phase_c_divergence = 0`) | remap row → `2wiki:c<curid>` via `reuse_map/`; no encode |
| SPLADE doc rows | same | same |
| KNN graph | **INVALID** — 15× more points; new points displace old neighbours | REBUILD globally |
| STRUCT / hyperlink adjacency | **INVALID** — endpoints resolved to old title-hash ids; internal-vs-external link status changes with membership | REBUILD |
| NER / NERX adjacency | **INVALID** — `1/df` weights depend on the whole corpus | REBUILD (per-doc postings for the full universe exist under `data/canonical/2wiki_universe/_ner_work/`) |
| H4_SK hypergraph, partition maps/aggregates | **INVALID** | REBUILD |
| router caches (`cache_2wiki*.npz`, `dense_top200_*`, `partition_map*.json`) | **INVALID** — they store integer row indices into the old 398,354 ordering | REGENERATE from the canonical ordering; never index-translate in place |
| halo / SP1 / SAFE caches | **INVALID** | REBUILD |
| query embeddings | **VALID** — query text is unchanged by the corpus change | REUSE |

**Node id remapping is possible and lossless in one direction only:** every one of the 398,354 old title-hash
nodes has a corresponding curid node in the new table (`legacy_comparison.id_mapping_coverage = 1.0` against
the 65,865-node legacy substrate; the superseded 398,354 table itself is preserved at
`data/final_canonical/2wiki/_superseded_context_union_398354/` precisely so this remap stays possible). The
reverse is not: 5,591,493 new nodes have no old counterpart.

The same invalidation statement applies to **hotpotqa**, which had no canonical_v1 downstream artifacts at all
(its legacy substrate was the 507,494-node distractor-context union, a different corpus entirely).

## The rule

```
NODE-LOCAL artifacts  -> REUSE aggressively:
    Dense embeddings, SPLADE rows, query embeddings, NER annotations,
    source structural facts
CORPUS-GLOBAL artifacts -> REBUILD:
    ANN/index, KNN graph, STRUCT adjacency, NER/NERX adjacency,
    H4_SK, partition aggregates, SAFE caches, halo caches, SP1 caches
KNN must be rerun globally even if every old embedding is reused,
because new points can alter old nearest neighbours.
```

## Why the KNN clause is not optional

A canonical corpus is strictly larger than every legacy substrate it replaces (musique 13,672 → 117,534;
2wiki 65,865 → **5,989,847**; squad 19,029 → 20,233; metaqa 40,151 → 43,234; hotpotqa 507,494 → **5,233,329**;
webqsp 781,485 → **BLOCKED, no canonical corpus**). k-NN is a *ranking among all points*: inserting a new point can displace an
existing neighbour of an old point, so an old edge list is wrong even when every vector in it is bit-identical.
The same argument applies to anything that aggregates over the whole node set — partitions, halo/core sets,
SAFE and SP1 caches, and any df-weighted edge family (NER `1/df` weights change when the document frequency
denominator changes).

## Applying it, given what Phase 3 measured

Phase 3 has now been **measured for all five built corpora** (`ARTIFACT_REUSE_REPORT.md`), by frozen-tokenizer
token-ID equality, with the memoization independently re-verified on 20,000 sampled inputs per heavy corpus
(0 mismatches):

| dataset | canonical N | dense rows reusable | dense rows needing encode | SPLADE rows needing encode |
|---|---:|---:|---:|---:|
| metaqa | 43,234 | 43,234 | 0 | 0 |
| 2wiki (TEXTUALIZATION_REV 2) | 5,989,847 | 5,902,083 | **87,764** (1.465 %) | **87,763** (1.465 %) |
| musique | 117,534 | 117,534 | 0 | 0 |
| squad | 20,233 | 20,233 | 0 | 0 |
| hotpotqa (TEXTUALIZATION_REV 2) | 5,233,329 | 5,231,641 | **1,688** (0.032 %) | **94** (0.0018 %) |
| **total** | **11,404,177** | **11,314,725** | **89,452** (0.784 %) | **87,857** (0.770 %) |

So the expensive half of a rebuild is still almost entirely paid for: **89,452 dense rows and 87,857 SPLADE rows**
are the *entire* encoder requirement across 11.4 M nodes — under 0.8 % on either side — and even those are
reported, not run. **No encoder was executed in this task.**

Both figures were re-measured on 2026-09-06, not adjusted arithmetically. Where they come from:

* **2wiki, 87,764 dense / 87,763 SPLADE.** The 87,765 nodes that `TEXTUALIZATION_REV 2` rewrote from `""` to
  their source-native title. Attribution was measured by partitioning the per-node map: the 5,902,082 nodes the
  rule did not touch need **0** rows on either side, so the entire demand is the revision's own
  (`ALL_ENCODE_DEMAND_COMES_FROM_THE_CHANGED_ROWS = true`). Two rows escape the bill because a token-identical
  input already exists elsewhere in the store — 1 on the dense side, 2 on the SPLADE side.
* **hotpotqa, 1,688 dense / 94 SPLADE.** 1,594 abstracts where Phase-C applied `.strip()` and canonical_v1 does
  not, plus the 94 empty-abstract nodes rewritten to their title. Those 94 are the whole SPLADE requirement:
  the 1,594 whitespace deltas are invisible to its uncased WordPiece, but a title where there was previously an
  empty string is not.

In **distinct forward passes** — the quantity that actually bounds GPU cost, since rows with equal token-ID
sequences collapse into one pass — 2wiki's bill is `87,764` dense and `87,655` SPLADE (108 of its new titles fold
onto another's token IDs under SPLADE's lowercasing and accent-stripping; none fold under dense's BPE, because all
87,765 titles are distinct strings).

The webqsp 1,316,466-row store also exists but is attached to a REJECTED corpus and is not reusable — see
`webqsp/FREEBASE_SOURCE_PROPOSAL.md` §5.

| artifact | class | action | note |
|---|---|---|---|
| dense doc embeddings (`data/canonical/<ds>/encodings/dense/docs`) | node-local | **REUSE** | remap row → canonical `node_id` via `reuse_map/shard_*.jsonl` (`phase_c_row_index`, `dense_reuse_source`) |
| SPLADE doc rows (`encodings/splade/docs`) | node-local | **REUSE** | same remap; SPLADE truncation 256 already matched in the reuse test |
| query embeddings (`encodings/{dense,splade}/queries`) | node-local | **REUSE** where the query text is unchanged | metaqa/webqsp query encodings are absent for some lanes — check per lane, and remember queries carry the `GTE_QINSTR` prefix while docs do not |
| NER per-doc entity postings | node-local | **REUSE** where per-doc output survives (2wiki, hotpotqa `_ner_work/`); re-run for musique/squad (edges only were kept — cheap, 2wiki's full pass was 29.4 s) | |
| source structural facts (2wiki hyperlinks, hotpot `text_with_links`, RoG triples, metaqa `kb.txt`) | node-local | **REUSE** the extracted facts | but see next row for the adjacency built from them |
| ANN / FAISS index | corpus-global | **REBUILD** | |
| KNN graph (`graph_knn.tsv`) | corpus-global | **REBUILD** | new points change old neighbours |
| STRUCT adjacency (`graph_structural.tsv`) | corpus-global | **REBUILD** | edge endpoints must resolve to canonical node_ids; membership changes which links are internal |
| NER / NERX adjacency (`graph_ner.tsv`) | corpus-global | **REBUILD** | `1/df` weights depend on the whole corpus |
| H4_SK hypergraph, partition aggregates/maps, SAFE caches, halo caches, SP1 caches | corpus-global | **REBUILD** | |

## Ordering constraint

Node id order is `lexicographic node_id`, and node_id is a content hash of the identity key — so the row order of
every rebuilt corpus-global artifact differs from the legacy ordering. Any cache that stores integer row indices
(the G2 `cache_<ds>.npz` family, `partition_map*.json`, `dense_top200_*`) is invalid across the boundary and must be
regenerated from the canonical ordering, never index-translated in place.

## Non-negotiables carried forward

- Write only under `data/final_canonical/`, `scratchpad/final_canonical_build/`,
  `results/data_audit/final_canonical_v1/`. Never touch `data/canonical/`, `data/processed/`,
  `data/ukb_storage/`, `results/GENERALIZATION/`, `scratchpad/ablation_qwen/`, `data/original/`.
- Reuse is decided by frozen-tokenizer token-ID equality, never by raw-text equality alone.
- No downstream build starts before a separate task authorises it. `LOCKED_6_OF_6` must NOT be written while
  webqsp is `BLOCKED_PENDING_FREEBASE_SOURCE`; the current status is `READY_5_OF_6_SOURCE_CONTRACTS`.
- 2wiki's downstream artifacts are INVALID (see the 2026-09-05 block at the top). They were not rebuilt.
