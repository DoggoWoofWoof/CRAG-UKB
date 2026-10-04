# 2Wiki — empty-canonical-text forensics (87,765 records)

> **STATUS UPDATE 2026-09-06 — THE PROPOSAL IN THIS DOCUMENT HAS SINCE BEEN APPLIED.**
> Everything below is preserved verbatim as the *pre-decision* forensic record, which is why it still speaks
> in the future tense and still cites `81fa7d1a5d4bbb24…` as current. On 2026-09-06 the user approved the
> revision and `TEXTUALIZATION_REV 2` was built and installed for 2wiki: all 87,765 records here were
> retextualized to their source-native title, `CORPUS_HASH` moved
> `81fa7d1a5d4bbb24… → 94fe68b8d5f291f1…`, `NODE_ORDER_HASH` and membership did **not** move, and
> `empty_text_nodes` is now `0`. The rev1 table is preserved at
> `data/final_canonical/_superseded_textualization_rev1/2wiki/`. See
> `_superseded_textualization_rev1/SUPERSEDED.md`, `REV2_DIFF_AUDIT.json` and `REV2_VERIFY.json`.
> Read the paragraph below as history, not as the current state of the corpus.

**REPORT ONLY. Nothing was applied.** No encoder, KNN, H4, SAFE, halo or SP1 was run; no builder was run; no
node table, corpus hash or gold mapping was touched. `CORPUS_HASH` is still `81fa7d1a5d4bbb24…`.

Derived **from source**, not from a supplied list:
`scratchpad/final_canonical_build/twowiki_empty_forensics.py` replays `build_kb.py:269-277` verbatim
(`sents = r.get("sentences") or []` → `body = " ".join(sents)`, **no** `.strip()`) over the approved archive
and reports every record whose canonical text is empty. Method mirrors the HotpotQA precedent
(`../hotpotqa/EMPTY_TEXT_94_AUDIT.md`, `scratchpad/final_canonical_build/hotpot_empty_forensics.py`).

* source: `data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip`
  (sha256 `a585bdc3c39425446e4b2701a5f7f30051cb6c100d179322055d53dc0a71a723`, 1,900,740,270 bytes;
  single member `para_with_hyperlink.jsonl`, 7,023,046,781 bytes uncompressed)
* pass 1 (text rule + encoder-input identity): **5,989,847 records**, 385 s
* pass 2 (in-link structure + query roles): 5,989,847 records, 453 s
* counts: `EMPTY_TEXT_SUMMARY.json` (this directory)
* machine-readable per-record tables, **87,765 lines each**:
  * `data/final_canonical/2wiki/EMPTY_TEXT_AUDIT.jsonl` — **full fidelity**, 32 columns, every source field
    (106.7 MB, gitignored tree)
  * `EMPTY_TEXT_AUDIT.jsonl` (this directory) — **compact**, 8.6 MB, the four columns that actually vary:
    `id` = canonical_node_id, `title` = source-native title (= the proposed fallback text),
    `cat` = category, `inlinks` = incoming hyperlink mentions. Every dropped column is either constant
    across all 87,765 rows (and recorded in `EMPTY_TEXT_SUMMARY.json`) or present in the full table.
    Reason for the split, not a preference: the full table is 106.7 MB, the largest **tracked** file anywhere
    under `results/` is 2.85 MB, and GitHub refuses files over 100 MB — so it goes where this repo's own rule
    already puts bulky tables (`mirror_audit.py`: *"bulky tables … stay only under data/final_canonical/"*).
    HotpotQA needed no split: its table is 94 rows / 115 KB.

```
records_scanned              = 5,989,847   ASSERTION PASSES (expected 5,989,847)
identified_empty_text_nodes  =    87,765   ASSERTION PASSES (expected 87,765 = 1.4652% of the corpus)
```

## §1 — why each became empty

All 87,765 land in **one** combination. There is no second path:

```
sentences_empty + title_present + mentions_empty + no_other_fields    87,765   (100%)
```

| measure | count |
|---|---:|
| `sentences` yields no non-empty sentence | 87,765 |
| `sentences` non-empty | 0 |
| **`sentences` is literally `[]`** (an empty list, `n_sentences == 0`) | **87,765** |
| `sentences == [""]` or any other all-empty-strings shape | 0 |
| `title` non-empty | 87,765 |
| `title` empty | 0 |
| `mentions` non-empty | 0 |
| any other source field carrying text | 0 (no other field exists) |
| **`truly_no_readable_source_text`** | **0** |
| `exact_empty_string` (`body == ""`) | **87,765** |
| `whitespace_only` (`body.strip()==""` but `body != ""`) | **0** |

The path is identical for all 87,765:

```
raw 'sentences' == []                 (an EMPTY list -- not a list holding an empty string)
  -> body = ' '.join([]) = ''
  -> not body.strip()                 -> counted empty, node KEPT
  -> final canonical encoder input = ''
```

Every one is **exactly `""`**; none is whitespace-only. Two independent measurements say so: (a) `sentences`
is the empty list in all 87,765, and `" ".join([])` is `""` by construction — a whitespace-only body would
need a list of blank strings, of which there are **0**; (b) the corpus-wide `strip_delta` is **0** here and
in `integrity_report.corpus_accounting_detail.phase_c_divergence.n_records_where_strip_changes_text`, i.e.
no record anywhere in 2wiki produces leading or trailing whitespace. So the `!= ""` vs `.strip()` predicate
in the proposed rule selects the same 87,765 records either way — but the two strings are **not**
interchangeable to the encoders (`""` → `[]`, `" "` → `[220]`, §7), so the rule below is written on
`.strip()`, which is correct under both.

> Difference from HotpotQA worth recording: Hotpot's 94 had `text: ['']` (a list holding one empty string)
> and a non-empty `text_with_links`. 2wiki's 87,765 have an **empty list and nothing else at all**.

## §2 — full key inventory, and is there any alternative text field? **NO — measured**

The 2wiki record schema has exactly four keys, and all 87,765 empty records carry all four and nothing else:

| source key | present on hits | usable as text? |
|---|---:|---|
| `id` | 87,765 | no — the Wikipedia curid (identity, not prose) |
| `title` | 87,765 | **YES — the only candidate** |
| `sentences` | 87,765 | no — empty list by definition of this set |
| `mentions` | 87,765 | no — **empty list in 100% of them** |

```
ALT_TEXT_NONEMPTY, per alternate field:
  sentences (any non-empty leaf)              0 / 87,765
  mentions  (any element at all)              0 / 87,765
  mentions[].ref_url readable residue         0 / 87,765   (vacuous: there are no mentions)
  any other key                               0 / 87,765   (no other key exists in the schema)
```

`mentions` is not a fallback candidate even in principle: its elements are
`{id, start, end, ref_url, ref_ids, sent_idx}` — character offsets into `sentences` plus a link-target slug,
never prose. Here the question is moot, because the list is empty for every one of the 87,765. There is no
2wiki analogue of Hotpot's `text_with_links`.

**Consequence: `title` is not merely the best fallback, it is the only one that exists.**

## §3 — title provenance: `TITLE_QUERY_INDEPENDENT = true`

| requirement | evidence |
|---|---|
| source path | `para_with_hyperlink.zip` (hash above) — the only corpus input |
| source key | `title`, read at `build_kb.py:282` as `r.get("title","")` from the **same JSON line** as `sentences` |
| deterministic rule | exact field read, no transform, no normalization, no lookup |
| query independence (schema) | the record schema is `{id, title, sentences, mentions}` — there is **no** question, answer, `supporting_facts`, split or any other 2Wiki annotation field anywhere in a record (verified over all 87,765: `source_key_inventory_over_hits` has exactly those four keys) |
| query independence (code) | `build_corpus_2wiki` opens `zipfile.ZipFile(S["zip"])` and nothing else; no question file is read in the corpus pass, and `--eval-subset` never reaches it (`build_kb.py:24-30`) |
| query independence (constructive) | already demonstrated for the corpus that contains these titles: three builds under different `--eval-subset` (legacy / random_held:7 / none) produced an identical `CORPUS_HASH` (`query_independence_test.json`, `CORPUS_QUERY_INDEPENDENT: true`) |
| not from neighbours | the fallback reads one field of the **same** record; no cross-record, no graph, no kNN, no retrieval output |
| reproducibility | 87,765/87,765 recovered on a clean re-scan from the hashed archive |

It is **not** from query text, supporting facts, answers, eval metadata, retrieval output, neighbour records
or manual aliases. The titles are ordinary Wikipedia article names — see §4.

## §4 — what these records are

| category (from the source title alone) | count | share |
|---|---:|---:|
| ordinary article | 81,236 | 92.56% |
| list page (`List of …` / `Lists of …`) | 6,173 | 7.03% |
| disambiguation (`… (disambiguation)`) | 202 | 0.23% |
| index / outline / timeline | 154 | 0.18% |
| namespace page (`Category:`, `Template:`, `File:`, …) | **0** | 0% |

| identity check | value |
|---|---|
| unique titles | **87,765** (every title distinct) |
| duplicate title groups | **0** |
| empty titles | **0** |
| duplicate curids | **0** |
| titles that also title a **non-empty** record | **0** |

Title length: min 1, p25 12, median 16, p75 23, max 141, mean 18.93 chars. curids span 728 … 62,717,334.
Examples: `Toshiro Akamatsu`, `Ministry of Justice (Afghanistan)`, `Sawara-ku, Fukuoka`,
`(39546) 1992 DT5`, `Kamisakaemachi Station`, `List of national anthems`, `Manet (disambiguation)`,
`Timeline of computing 1990–1999`. 7,220 carry a parenthetical qualifier; 440 are purely numeric.
These are real, meaningful article names, not placeholders.

**Redirect vs malformed extraction.** The source carries no redirect flag, so this cannot be settled from
the archive alone — stated as a limit, not glossed over. What *is* measured: these records were emitted by
the same extractor as the other 5.9M, with a well-formed id and title and an empty paragraph list, which is
the signature of *extraction yielding no lead paragraph* (redirect pages, disambiguation stubs, list pages
whose content is entirely a table/list, and articles whose lead is markup-only all produce exactly this).

The category mix supports that reading, but only partly, and the mismatch is worth stating. Corpus-wide base
rates (MEASURED, `rg -c -F` line counts over the 5,989,847-line `nodes.jsonl`):

| title shape | in the 87,765 empties | corpus-wide | enrichment |
|---|---:|---:|---:|
| `List of …` / `Lists of …` | 7.03% | 108,915 = 1.82% | **3.9×** |
| index / outline / timeline | 0.18% | 3,674 = 0.06% | 2.9× |
| `… (disambiguation)` | 0.23% | ~51,390 = 0.86% | **0.27× (depleted)** |

List pages are strongly over-represented — consistent with "the article body is a table, so no lead
paragraph was extracted". Disambiguation pages are *under*-represented, so an empty lead is **not** simply
"this page is a stub kind". The remaining 92.6% are ordinary article names, which the archive gives no way
to classify further.

### §4b — how the rest of the corpus refers to them (second full source pass, MEASURED)

`mentions` is 2wiki's only structural signal, so pass 2 re-scanned all 5,989,847 records and counted
hyperlinks **into** each empty node (their outgoing edge count is 0 by construction — no sentences, no
mentions).

| in-links | empty nodes | share |
|---|---:|---:|
| 0 | **67,241** | 76.6% |
| 1 | 10,487 | 12.0% |
| 2–9 | 8,481 | 9.7% |
| 10–99 | 1,487 | 1.7% |
| ≥100 | **69** | 0.08% |

Totals: 98,077 incoming mentions from 96,482 distinct linking articles; max 2,685; median 0.

So **76.6% are fully isolated** in the mention graph (no in-edge, no out-edge) — for those the empty text is
the *only* thing the node has, and there is no structural path by which any retriever could reach them.
The other 23.4% are the opposite problem: they are link targets with a zero-length encoder input. The
top of that tail is not obscure:

```
2,685  2wiki:c5878373   Terrestrial animal
2,185  2wiki:c448061    Isfahan Province
1,550  2wiki:c297809    Socialist Federal Republic of Yugoslavia
1,482  2wiki:c186932    Japanese people
1,468  2wiki:c24751     Punjab, Pakistan
1,196  2wiki:c19261     Monaco
  886  2wiki:c1409922   Incertae sedis
  548  2wiki:c58092     Hokkaido
  535  2wiki:c498108    TV Tokyo
  515  2wiki:c5422      Capcom
```

These are hub entities the SK/H4 topology will route through, and today every one of them presents the dense
encoder with a zero-length sequence.

### §4c — none of them is ever a gold or context node (MEASURED)

Read from the **already built** query layer (impact assessment only — nothing about the corpus depends on
it): over all 192,606 official questions of all three splits, which reference 398,354 distinct context nodes
and 188,350 distinct gold nodes:

```
empty-text nodes appearing as GOLD     :  0 / 87,765
empty-text nodes appearing as CONTEXT  :  0 / 87,765
```

Positive control: the same scan does resolve 398,354 context and 188,350 gold node ids, so the zero is a
real property, not a matching failure. The explanation is structural — an official 2wiki context paragraph
must *have* text, so an article with no lead paragraph can never be one — and it is corroborated by
`hits_whose_title_also_titles_a_NONEMPTY_record = 0` (no empty node's title is reachable through some other
curid either).

Two consequences, and they point in opposite directions:

* **Safety.** The proposed fallback cannot change gold coverage, gold resolution, or any recall number
  through the gold path. `ALL_GOLD_REFS_RESOLVE` stays true by construction.
* **What it *does* affect.** These 87,765 are pure distractor mass — 1.47% of the corpus that currently
  occupies one point in embedding space. Every query's neighbourhood, every KNN edge list, every partition
  and every H4 split is computed against that degenerate block. That is the actual reason to fix it, and it
  is a *representation* argument, not a recall-gain argument.

## §5 — required counts

```
EMPTY_BEFORE           = 87,765
TITLE_NONEMPTY         = 87,765
ALT_TEXT_NONEMPTY      =      0     (sentences 0, mentions 0, mentions[].ref_url 0, other fields: none exist)
TRULY_TEXTLESS         =      0     (no record lacks BOTH text and title)

exactly ""             = 87,765     <- all of them; distinct token sequence [] under gte
whitespace-only        =      0     <- none; " " would tokenize to [220], a DIFFERENT input
```

```
EMPTY_BEFORE                        = 87,765
TITLE_RECOVERABLE                   = 87,765
STILL_EMPTY_AFTER_PROPOSED_FALLBACK =      0
```

## §6 — proposed rule (**NOT APPLIED**)

General, deterministic, source-level, fallback-only, no hardcoded ids. It is the same rule already accepted
for HotpotQA (`TEXTUALIZATION_REV 2`, `build_kb.py:457-471`), transposed to 2wiki's `sentences` field.

```python
# ---- build_kb.py::build_corpus_2wiki : line 253, add two counters ----
    raw = 0; dropped = []; empty_text = 0; strip_delta = 0; strip_examples = []
    title_fallback = 0; still_empty = 0                      # NEW

# ---- replace line 277 (`if not body.strip(): empty_text += 1`) with: ----
            if not body.strip():
                empty_text += 1
                # TEXTUALIZATION_REV 2 (2wiki) -- identical in form to the hotpotqa rule at build_kb.py:457-471.
                # A record whose 'sentences' list yields nothing is otherwise handed to the dense encoder as ""
                # -- gte-Qwen2 tokenizes "" to the ZERO-LENGTH sequence [] (measured), so all 87,765 such nodes
                # collapse onto ONE degenerate forward pass; SPLADE maps every one of them to [CLS][SEP].  The
                # same source record carries a human-readable 'title'.  Source-native (same JSON line as
                # 'sentences'), deterministic, and query-independent: the record schema is
                # {id, title, sentences, mentions} -- no question/answer/supporting_facts field exists.
                # There is NO alternative text field: 'mentions' is EMPTY for every one of these records
                # (measured 87,765/87,765) and in any case holds only offsets and link slugs, never prose
                # (results/data_audit/final_canonical_v1/2wiki/EMPTY_TEXT_AUDIT.md).
                # FALLBACK ONLY: a record that already yields non-empty text is untouched, and no title is
                # prepended anywhere else in the corpus.  A record with an empty title would stay empty.
                tfb = r.get("title", "")
                if tfb.strip():
                    body = tfb
                    title_fallback += 1
                else:
                    still_empty += 1
```

Ordering matters and is deliberate:

* the `strip_delta` check at lines 272-276 stays **before** the fallback — it measures Phase-C divergence on
  the original join and must not see the substituted text;
* `content_hash` at line 283 is computed from `body` *after* the substitution, so the node's hash follows its
  final text (same as hotpot, asserted by `test_build_kb.py:179`);
* `n_sentences` stays `len(sents)` = 0 — the source fact is preserved, not overwritten.

Accompanying `acc` changes (mirroring `build_kb.py:489-520`), so the accounting stays honest:

```python
"text_construction": "text = ' '.join(official 'sentences' list); if that join is empty the source-native "
                     "'title' of the SAME record is used instead (TEXTUALIZATION_REV 2, fallback only -- no "
                     "title is prepended to any record that already has text)",
"textualization_fallback": {
    "TEXTUALIZATION_REV": 2,
    "rule": "if ' '.join(sentences) is empty -> canonical_text = record['title'] (unchanged otherwise)",
    "empty_before_fallback": empty_text,          # 87,765
    "title_fallback_applied": title_fallback,     # 87,765 (projected)
    "still_empty_after_fallback": still_empty,    # 0      (projected)
    "field_source": "record['title'] -- same JSON line as 'sentences'; schema carries no question/answer field",
    "query_independent": True,
    "membership_impact": "NONE -- no node added, removed or reordered; identity remains the curid"},
"phase_c_divergence": {
    ...,
    "n_records_with_a_different_phase_c_encoder_input": strip_delta + title_fallback},   # 0 + 87,765
```

That last line is the one that must not be forgotten: `phase_c_divergence` currently reports **0** records
differing from the already-encoded 2wiki_universe input, and after this change it is **87,765**.

Suggested test assertions, mirroring `test_build_kb.py:166-182` / `215-231`:
`2wiki: empty article gets its TITLE`, `2wiki: non-empty body UNCHANGED (no title prepended)`,
`2wiki: empty title stays empty`, `2wiki: content_hash follows the post-fallback text`,
`2wiki: fallback is query-independent (different query set -> identical CORPUS_HASH)`.

## §7 — why the current state is not acceptable (MEASURED tokenizer evidence)

Both frozen tokenizers were loaded (offline, tokenizer only — **no model, no forward pass**) and probed:

| input | `gte-Qwen2-1.5B-instruct` (dense, BPE) | `splade-cocondenser-ensembledistil` (WordPiece) |
|---|---|---|
| `""` | **`[]` — zero-length** | `[101, 102]` — `[CLS][SEP]` only |
| `" "` | `[220]` | `[101, 102]` |
| `"  "` | `[256]` | `[101, 102]` |
| `"Masaki Kobayashi"` | `[43649, 14624, 59148, 352, 30378]` | `[101, 16137, 8978, 28930, 102]` |
| `" Masaki Kobayashi "` | `[19868, 14624, 59148, 352, 30378, 220]` | `[101, 16137, 8978, 28930, 102]` |

So today **87,765 nodes (1.47% of the corpus) share one degenerate dense forward pass over a zero-length
token sequence**, and the same 87,765 share one SPLADE forward pass over a bare `[CLS][SEP]`. Whatever
vector that produces, it is identical for all of them and carries no information about the article — while
the source record plainly holds a usable human-readable name. At 94 records (hotpot) this was a curiosity;
at 87,765 it is a block of identical vectors 934× larger, which is why it has to be settled before Dense,
KNN, SK and H4 are built on top of it.

The table also shows why the two bills differ (§8): gte's BPE preserves whitespace **and** case, while
SPLADE's WordPiece is whitespace-, case- **and** accent-insensitive, so SPLADE's token-identity classes are
strictly coarser.

## §8 — feature-bill projection (no encoder was run)

Decision rule (unchanged, `scratchpad/final_canonical_build/reuse_map_kb.py`): a dense/SPLADE row is
**reusable iff its frozen-tokenizer token-ID sequence equals an already-encoded row's**. Byte-identical
inputs are a *sufficient* condition (the tokenizer is a deterministic pure function; independently verified
there with k=20,000, 0 mismatches), so byte-identity membership is measurable **without tokenizing or
encoding anything**.

Already-encoded pool used for the test (both families the reuse map credits):
`PHASE_C_UNIVERSE` 5,989,847 inputs + `PHASE_C_VIEW398` 398,354 inputs → 5,894,962 distinct strings.
The universe inputs were re-derived from the archive as `strip(" ".join(sentences))`
(`scratchpad/build_2wiki_universe.py:42`); cross-checked against the **stored** encoder-input shards:
40,000/40,000 of `data/canonical/2wiki_universe/encodings/_src/docs/shard_00000.jsonl` present.

| quantity | value | status |
|---|---:|---|
| dense rows reusable **today** (whole corpus) | 5,989,847 / 5,989,847 | MEASURED (`reuse_map/index.json`) |
| SPLADE rows reusable **today** | 5,989,847 / 5,989,847 | MEASURED (`reuse_map/index.json`) |
| rows whose input the proposal changes | 87,765 | MEASURED |
| of those, title already a stored encoder input (byte-identical) | **1** | MEASURED |
| **new dense rows required** | **≤ 87,764** | MEASURED upper bound |
| distinct new dense forward passes | 87,764 (all titles distinct) | MEASURED |
| **new SPLADE rows required** | **≤ 87,764** | MEASURED upper bound |
| distinct new SPLADE forward passes | **87,748** | MEASURED (16 title pairs collide once whitespace/case are folded away, e.g. `party pooper`, `border poll`) |
| rows unaffected | 5,902,082 | MEASURED |

The single reusable row is real, not an artefact: node `2wiki:c320778` has title `Kagawa`, and the stored
encoder input of `2wu:28138851` is exactly `"Kagawa"` (grepped directly out of
`.../encodings/_src/docs/shard_00080.jsonl`).

Why the two bills are not the same number, and why both are *upper* bounds:

* **row counts coincide** at 87,764 because the same single title (`Kagawa`) is the only one already present
  in the encoded pool under either tokenizer's equivalence;
* **distinct forward passes differ** — 87,764 dense vs 87,748 SPLADE — because SPLADE's uncased WordPiece
  folds 16 title pairs together that gte's BPE keeps apart;
* **dense** is effectively tight — byte-identity is essentially necessary as well as sufficient for a
  whitespace- and case-preserving BPE;
* **SPLADE** is only an upper bound: the membership test folded whitespace and case, but *not* accent
  stripping or punctuation splitting, which the uncased WordPiece tokenizer also does (`Tó Neinilii` and
  `To Neinilii` measured identical). Any additional collision only lowers the bill.

PROJECTED, not measured — cost of paying that bill: 87,764 rows of ~5-40 characters, median title 16 chars
→ order 1.3M dense tokens total (≈15 tokens/title), against the 5.99M full-paragraph rows already encoded.
On the recorded Modal per-token pricing ($0.086–0.129 / Mtok) that is roughly **$0.11–0.17**, i.e. ~0.1% of
a full re-encode; SPLADE is capped at 256 tokens and is cheaper still. Treat these as order-of-magnitude.

**What the change does *not* cost:** no node is added, removed or reordered, so `NODE_ORDER_HASH`
(`57dd809a…`) and every `node_id`, the legacy id map, the gold title→curid resolution and all query files are
untouched — and per §4c not one of the 87,765 is a gold or context node, so gold coverage cannot move at all.
`CORPUS_HASH` **does** change (87,765 `content_hash` values change), exactly as it did for hotpot.
`dataset_manifest.json`, `integrity_report.json`, `build_info.json`, `query_independence_test.json` (all four
runs), `SOURCE_CONTRACT.NORMALIZATION_RULE` and `reuse_map/` would need regeneration. Per
`dataset_manifest.downstream_artifacts_built = "NONE (no embeddings / SPLADE / graphs / partitions /
indexes)"`, **no downstream 2wiki artifact exists yet to invalidate** — which is precisely why now is the
cheap moment.

## §9 — honest limits

1. **Redirect vs failed extraction cannot be separated from this archive** (§4). The fallback does not
   depend on the distinction: it uses the title either way, and a redirect page's title is exactly as
   legitimate a retrieval surface as a stub's.
2. **A title is a weak document.** 16 median characters is a much thinner encoder input than a lead
   paragraph. The claim here is only that it is strictly better than a zero-length sequence and that it is
   source-native; it is not a claim that these 87,765 nodes become good retrieval targets. For the 67,241
   that are also structurally isolated (§4b), the title is the *only* signal they will ever have, and it
   will usually not be enough. What the fix removes is a degenerate collapse, not a retrieval deficit.
3. **Nothing here is genuinely textless.** `TRULY_TEXTLESS = 0`: every one of the 87,765 has a usable
   source-native title. That is the finding, and it is why the decision is clean — but it is worth stating
   plainly that the *fallback* is a name, not prose, for all 87,765 of them.
4. **The SPLADE bill is an upper bound**, not an exact count (§8).
5. **Set membership was tested with 64-bit sha256 prefixes.** With 5.9M keys and 87,765 queries the expected
   number of false positives is ~3e-8; the one positive was additionally verified by direct grep.
6. **Mirror caveat (unrelated to the fix, but worth one line).** `scratchpad/final_canonical_build/mirror_audit.py`
   moves any file under `results/data_audit/final_canonical_v1/` that has no counterpart under
   `data/final_canonical/` into `_superseded_mirror/`. `EMPTY_TEXT_AUDIT.jsonl` now has an upstream
   counterpart and survives; **`EMPTY_TEXT_AUDIT.md` and `EMPTY_TEXT_SUMMARY.json` do not**, and neither do
   hotpotqa's `EMPTY_TEXT_94_*`, so the next `mirror_audit.py` run would exile them (moved, never deleted).
   The durable fix is to add these names to that script's `DS_FILES` list. Not done here: editing
   `mirror_audit.py` is outside this task's write scope, and `build_kb.py` / `test_build_kb.py` are being
   edited concurrently by another job.
