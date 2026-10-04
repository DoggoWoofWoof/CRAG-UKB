# HotpotQA — empty-canonical-text forensics (94 records)

Derived **from source**, not from a supplied list: `scratchpad/final_canonical_build/hotpot_empty_forensics.py`
replays `build_kb.py:432-439` verbatim (`_flat_sentences` + `' '.join`, **no** `.strip()`) over the approved
tarball and reports every record whose canonical text is empty.

* source: `data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2`
  (sha256 `1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729`, 1,553,565,403 bytes)
* scan: 15,517 tar members, **5,233,329 raw records**, 955.9 s
* machine-readable per-record table: `EMPTY_TEXT_94_AUDIT.jsonl` (94 lines, all source fields)

```
identified_empty_text_nodes = 94        ASSERTION PASSES (expected 94)
```

## §2 — why each became empty

All 94 land in **one** combination. There is no second path:

```
abstract_empty + links_nonempty + title_present + other_present   94   (100%)
```

| measure | count |
|---|---:|
| `abstract_empty` | 94 |
| `abstract_nonempty` | 0 |
| `text_with_links_nonempty` (raw field) | 94 |
| `text_with_links_empty` (raw field) | 0 |
| `title_nonempty` | 94 |
| `title_empty` | 0 |
| `other_source_text_nonempty` (charoffset arrays) | 94 |
| **`truly_no_readable_source_text`** | **0** |
| `exact_empty_string` (`body == ""`) | 94 |
| `whitespace_only` (`body.strip()==""` but `body != ""`) | **0** |

The path is identical for all 94:

```
raw 'text' == ['']                    (a list holding one EMPTY sentence)
  -> _flat_sentences yields nothing   ('' is falsy, build_kb.py:397)
  -> body = ' '.join([]) = ''
  -> not body.strip()                 -> counted empty, node KEPT
  -> final canonical encoder input = ''
```

Because every one is *exactly* `""` and none is whitespace-only, the `!= ""` vs `.strip()` distinction in the
proposed fallback rule is moot here — both predicates select the same 94 records.

**Phase-C's divergent path.** `c1_hotpot_fullwiki.py:38` fell back to `text_with_links` when `text` was falsy.
That fallback fired for all 94 (matching `n_records_where_phase_c_fell_back_to_text_with_links = 94`).

## §3 — is `text_with_links` a usable alternative? NO — measured

`text_with_links` is non-empty for all 94, but it holds **only empty anchor tags**:

```
c10158121  <a href="https%3A//commons.wikimedia.org/...jpg"></a>
c10186388  <a href="https%3A//en.wikipedia.org/wiki/Michael_R._Burns"></a>
c1086109   <a href="http%3A//www.nswrail.net/lines/sydney-lines.html"></a>
```

Stripping the markup leaves nothing:

```
records with ANY readable text after removing <...> tags:  0 / 94
```

Corroborated by the source's own offsets: **`charoffset == [[]]` for all 94** — the dump itself records zero
text spans for the plain-text field. So Phase-C's stored encoder input for these rows was **URL markup, not
prose**, which is why the canonical contract rejects `text_with_links` outright
(`NORMALIZATION_RULE`: *"Plain 'text' is used, NOT 'text_with_links'"*). It is not a fallback candidate.

## §3 — title provenance: `TITLE_QUERY_INDEPENDENT = true`

| requirement | evidence |
|---|---|
| source path | the approved tarball above — the only corpus input |
| source key | `title`, read at `build_kb.py:451` as `r.get("title","")` from the **same JSON line** as `text` |
| deterministic rule | exact field read, no transform, no normalization |
| query independence | the record schema is `{id, title, url, text, text_with_links, charoffset, charoffset_with_links}` — there is **no** question, answer, `supporting_facts`, or split field anywhere in the record; `build_corpus_hotpotqa` receives no query input at all |
| reproducibility | 94/94 recovered on a clean re-scan from the hashed tarball |

It is **not** from query text, supporting facts, answers, eval metadata, retrieval output, neighbour records,
or manual aliases. All 94 URLs are canonical `https://en.wikipedia.org/wiki?curid=<id>`.

## §4 — what these records are

| category (from the source title alone) | count |
|---|---:|
| ordinary article | 86 |
| list page (`List of …`) | 2 |
| year-prefixed article | 6 |
| disambiguation | 0 |

| identity check | value |
|---|---|
| unique titles | **94** |
| duplicate title groups | **0** |
| empty titles | **0** |
| duplicate curids | **0** |

Title length: min 8, max 47, mean 19.9 chars. Examples: `Windmill Hill, Kent`,
`List of Arizona State University alumni`, `USS Mapiro (SS-376)`, `Pardada Pardadi Educational Society`,
`Pandavaiar River`. These are real, meaningful article names — not placeholders.

No node is dropped. Corpus membership stays source-defined (94 unique curids, all retained).

## §5 — measured effect of the proposed fallback

```
EMPTY_BEFORE                        = 94
TITLE_RECOVERABLE                   = 94
STILL_EMPTY_AFTER_PROPOSED_FALLBACK = 0
```

## Why the current state is not acceptable

`Alibaba-NLP/gte-Qwen2-1.5B-instruct` tokenizes `""` to **`[]` — a zero-length token sequence** (verified
directly; `" "` gives `[220]`, so the two are distinct inputs). The 94 nodes therefore do not merely share an
embedding: they share the output of a **degenerate forward pass over an empty input**, credited from a single
Phase-C row (`PHASE_C:44755`). Reuse is sound in the token-identity sense — identical input, identical output —
but the *input itself* carries no information about the article, while the source record plainly holds a
usable human-readable name.
