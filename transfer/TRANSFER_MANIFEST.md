# CRAG canonical_v1 — transfer package

Every number below was measured in this session against the artifacts on disk, not copied from a
manifest. Where a shipped manifest disagrees it is called out as stale.

## 1. Verdict per dataset

| dataset | nodes | corpus_hash | graph edges / relations | dense | SPLADE | queries | transferable |
|---|---|---|---|---|---|---|---|
| **metaqa**   | 43,234 | `5f55719a` | 133,582 / **9** | 43,234 ✔ | ✔ | train+dev+test | **YES, complete** |
| **squad**    | 20,233 | `f592d4ab` | 874,190 / 1 *(derived)* | 20,233 ✔ | ✔ | train+dev | **YES, complete** |
| **musique**  | 117,534 | `dc00da21` | 2,744,076 / 1 *(derived)* | 117,533 ✔ | ✔ | train+dev+test | **YES, complete** |
| **hotpotqa** | 5,233,329 | `1b7eeac2` | 15,367,541 / 1 | 5,233,235 ✔ + **94 stale** | same | train+val+test | YES after 94-row patch |
| **2wiki**    | 5,989,847 | `94fe68b8` | 28,963,600 / 1 | 5,902,082 ✔ + **87,765 stale** | same | train+dev+test | YES after 87,765-row patch |
| **webqsp v1**| 2,592,894 | *not computed* | 8,309,195 / **7,058** | **none** | **none** | not in v1 | graph+text yes, **embeddings no** |

`musique` 117,533 vs 117,534 is benign and explained in §3.

## 2. What "complete and clean" rests on

* `dataset_manifest.json` per dataset records source sha256 + byte size for every input file,
  `dropped_records: 0`, and the dedup policy.
* `REV2_VERIFY.json` (2wiki, hotpotqa) recomputes `CORPUS_HASH` and `NODE_ORDER_HASH` over every
  node and re-derives the content hash for all 5.99M / 5.23M rows: `CONTENT_HASH_ALL_CORRECT: true`,
  `EMPTY_TEXT_N: 0`.
* `MANIFEST.json` carries `CORPUS_QUERY_INDEPENDENT: true` and `all_eval_gold_nodes_present: true`
  for all five text/KB corpora.
* **Stale in the shipped records, do not trust:** `data/final_canonical/MANIFEST.json` still says
  `WEBQSP_STATUS: WEBQSP_BLOCKED_ON_FREEBASE_SOURCE` and `FULL_CANONICAL_V1_STATUS:
  READY_5_OF_6`, and `webqsp/status.json` still says `BLOCKED_PENDING_FREEBASE_SOURCE`. Both
  predate the WebQSP∪CWQ build that is on disk at `webqsp/v1/`. Re-derive, don't quote them.

## 3. Embeddings: 100% reusable, via an id rewrite

The encodings under `data/canonical/<dir>/encodings/` were computed 2026-08-23, before the
canonical_v1 id scheme existed. The **row sets are identical**; only the id strings differ.
Measured canonical-node coverage:

| dataset | encoding dir | rule | canonical nodes covered |
|---|---|---|---|
| 2wiki | `2wiki_universe` | `2wu:<curid>` → `2wiki:c<curid>` | 5,989,847 / 5,989,847 |
| hotpotqa | `hotpotqa` | `hotpot_<curid>` → `hotpotqa:c<curid>` | 5,233,329 / 5,233,329 |
| metaqa | `metaqa` | `metaqa_ent_<i>` → `metaqa:e<i:05d>` | 43,234 / 43,234 |
| musique | `musique` | exact text join (table emitted) | 117,534 / 117,534 |
| squad | `squad` | exact text join (table emitted) | 20,233 / 20,233 |

musique and squad need a table rather than a rewrite because the legacy id hashed the **text
alone** while canonical hashes **(title, text)**. That difference also explains musique's off-by-one:
one paragraph text is carried by two different titles (`Etz Efraim`, `Adei Ad`), so canonical keeps
two nodes where legacy kept one. Both canonical nodes share that single vector, which is correct —
the encoder input was the text alone. Emitted by `transfer/embedding_id_bridge.py` to
`transfer/id_bridge/`.

Encoder config (from `src/experiments/canonical_encode.py`, both models present in the local HF cache):

* dense — `Alibaba-NLP/gte-Qwen2-1.5B-instruct`, `trust_remote_code`, FP16, `normalize_embeddings=True`,
  dim 1536 stored float16, shard size 40,000. **Documents get no prefix**; queries get
  `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`.
* SPLADE — `naver/splade-cocondenser-ensembledistil`, `log(1+relu(logits))` max-pooled to CSR,
  vocab 30,522, `max_len` 256.

## 4. The only real staleness: 87,859 rows that encoded the empty string

TEXTUALIZATION_REV2 gave empty-bodied articles their title as text. The Aug-23 encodings predate it.
Measured exactly, not estimated — every one of these rows encoded `""`:

| dataset | stale rows | of total | status |
|---|---|---|---|
| 2wiki | 87,765 | 5,989,847 (1.47%) | patch |
| hotpotqa | 94 | 5,233,329 (0.002%) | patch |
| **total** | **87,859** | 11,223,176 (0.78%) | |

`transfer/encode_rev2_delta.py` re-encodes only these and writes a **patch**, leaving the 25 GB of
base shards untouched: `transfer/rev2_patch/<ds>/{dense.npy,splade.npz,ids.json,status.json}`.
Apply by overriding the listed `node_id`s after loading the base shards.

## 5. Relations: nothing was discarded

Audited because the graphs carry a single link type for four corpora — that is the source's
vocabulary, not a loss.

* **metaqa — proven exactly lossless.** Source `kb.txt` = 134,741 lines / **133,582 distinct**
  triples (1,159 exact duplicates); canonical = **133,582** edges. Distinct-source − canonical = **0**.
  All 9 relations present. `graph_manifest.json`: `unmapped_endpoint_edges_dropped: 0`.
* **webqsp — 7,058 relations, exact match to the published RoG figure**, and 8,309,195 triples,
  also exact (`ROG_UNION_REBUILD.json`, `HARD_CHECK_EXACT: true`, 0 malformed, 0 dangling).
  metaqa's 9 relations, recounted from the shipped graph: `starred_actors` 33,718, `has_tags`
  28,719, `written_by` 19,127, `release_year` 16,715, `directed_by` 15,957, `has_genre` 15,485,
  `in_language` 3,428, `has_imdb_rating` 314, `has_imdb_votes` 119 — sums to 133,582.
* **2wiki / hotpotqa** — `structural_native` hyperlinks from the official
  `para_with_hyperlink.zip` and `enwiki-20171001` withlinks abstracts. One edge type because a
  hyperlink is one edge type.
* **musique / squad** — `structural_derived` `title_mention`; the manifests state plainly
  *"no native graph available"*. There was no source relation vocabulary to discard.

**Use the right 2wiki graph.** `data/canonical/2wiki/graph_structural.tsv` (359,549 edges) belongs
to the superseded 398,354-node corpus. The graph for the canonical 5,989,847-node corpus is
`data/canonical/2wiki_universe/graph_structural.tsv` — **28,963,600 edges**, 80× larger.

**The one honest gap, and it is the source's.** 2wiki's graph manifest records
`mentions_without_ref_ids: 6,194,218` — hyperlink mentions in `para_with_hyperlink` that carry no
resolvable target curid, so they cannot become edges. Also `targets_out_of_universe: 310` and
`self_loops_dropped: 3,328`. Nothing there was discarded by choice; the source does not name a
target. Worth stating in any writeup rather than implying the hyperlink graph is complete.

**Joinability is solved.** The graph endpoint id space is the *same* space as the encoding row ids,
so the §3 bridge maps graphs too. Measured, distinct endpoints that fail to map: metaqa **0**,
musique **0**, squad **0** (hotpot/2wiki measured separately, see `graph_join.json`). The shipped
`node_id_map_legacy.json` is a red herring for this purpose — it bridges the *document* space and
covers essentially no graph endpoints. Use the rewrites, not that file.

## 6. What to copy

**Minimum working set — five complete datasets, ~64 GB:**

```
data/final_canonical/{metaqa,squad,musique,hotpotqa,2wiki}/          19.5 GB  corpora, queries, manifests
data/canonical/{metaqa,squad,musique,hotpotqa,2wiki_universe}/encodings/   43 GB  dense + SPLADE
data/canonical/{metaqa,squad,musique,hotpotqa,2wiki_universe}/*.tsv       1.7 GB  graph_structural.tsv
data/canonical/{metaqa,squad,musique,hotpotqa,2wiki_universe}/graph_manifest.json  + its manifest
transfer/                                                             small  bridge, patch, this manifest
src/experiments/canonical_encode.py                                   small  the encoder, to extend or re-run
```

Per-corpus encoding sizes, `_src/` excluded: metaqa 137 MB, squad 544 MB, musique 525 MB,
hotpotqa 19.5 GB, 2wiki_universe 22.2 GB. Dropping `_src/` (the encoder input shards) saves a
further 4.5 GB; keep it for 2wiki/hotpotqa if you want to verify the REV2 patch.

**Do not copy:** `data/final_canonical/_superseded_textualization_rev1/` (18 GB, superseded),
`data/final_canonical/_work/` (3 GB, build scratch), `data/final_canonical/freebase_v3/` (125 GB,
a separate project), `data/canonical/2wiki/` entirely — both its `encodings/` and its `graph_structural.tsv`
belong to the superseded 398,354-node corpus; `2wiki_universe` replaces both.

**webqsp** adds `data/final_canonical/webqsp/v1/` (239 MB: nodes, edges, relations, entity_text,
cvt_text, name_resolvable_band) — graph and text complete, embeddings absent.
