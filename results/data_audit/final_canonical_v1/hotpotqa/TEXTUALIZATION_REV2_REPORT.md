# HotpotQA — TEXTUALIZATION_REV 2: forensics, fix, and re-pin

Final report for the empty-canonical-text task. Everything below is measured; each claim names the script that
produced it. No encoder, ANN/KNN, graph rebuild, H4, SAFE, halo or SP1 was run.

---

## A. Scope and outcome

94 of HotpotQA's 5,233,329 canonical nodes carried empty text. The task was to find out **why**, to decide
whether query-independent source text existed, and to correct the textualization only if the source supported a
principled deterministic fallback.

It did. The rule adopted is a source-native title fallback, applied as a general rule over the source records —
never as a special case over 94 ids.

```
MEMBERSHIP change : NONE      IDENTITY change : NONE      ORDER change : NONE
TEXT       change : 94 nodes, every one of which was previously exactly ""
CORPUS_HASH       : fe139de9c7c2a14f... -> 1b7eeac2bfbd7100...      (moved, as it must)
NODE_ORDER_HASH   : 8f043c35cbf629a2... -> 8f043c35cbf629a2...      (unchanged, as it must)
EMPTY_TEXT_N      : 94 -> 0
```

## B. Why the 94 were empty — forensics

`hotpot_empty_forensics.py` replayed `build_kb.py:432-439` verbatim over the approved tarball
(sha256 `1acca1c5…`, 1,553,565,403 B): 15,517 tar members, **5,233,329 records**, 955.9 s. It re-derived the
empty set from source rather than accepting a supplied list, and found **exactly 94** — the expected count.

All 94 follow one path; there is no second cause:

```
raw 'text' == ['']  ->  _flat_sentences yields nothing  ->  body = ' '.join([]) = ''
```

Every one is *exactly* `""`; **0** are whitespace-only, so the `!= ""` vs `.strip()` distinction is moot here.

**`text_with_links` was rejected by measurement, not preference.** It is non-empty for all 94 but holds only
empty anchor tags. Readable residue after stripping markup: **0 / 94**. The dump's own offsets agree —
`charoffset == [[]]` for all 94. Phase-C's stored encoder input for these rows was therefore URL markup, not
prose.

**Why the old state was a real defect, not cosmetic.** `gte-Qwen2` tokenizes `""` to the **zero-length sequence
`[]`** (verified directly; `" "` gives `[220]`, so the two are distinct inputs). The 94 nodes did not merely
share an embedding — they shared the output of a degenerate forward pass over an empty input, credited from a
single Phase-C row.

Full writeup: `EMPTY_TEXT_94_AUDIT.md`; per-record table: `EMPTY_TEXT_94_AUDIT.jsonl`.

## C. The rule

```python
if ' '.join(non-empty sentences of record['text']) is empty:
        canonical_text = record['title']          # same JSON line, exact field read, no transform
```

Fallback only: no title is prepended to any record that already has text, and a record with an empty title
stays empty. **Query-independent by construction** — the record schema is
`{id, title, url, text, text_with_links, charoffset, charoffset_with_links}`, with no question, answer,
`supporting_facts` or split field anywhere, and `build_corpus_hotpotqa` receives no query input at all.

Titles are real article names, not placeholders: 94 unique, 0 empty, 0 duplicates, length min 8 / max 47 /
mean 19.9 chars (`Windmill Hill, Kent`, `USS Mapiro (SS-376)`, `Pardada Pardadi Educational Society`).

`test_build_kb.py` pins the behaviour and passes **78/78** (was 69/69), including: nonempty bodies unchanged,
fallback applied, empty title stays empty, no title prepended, membership unchanged, exact counters,
`content_hash` follows the post-fallback text, and a build under a *different* query parquet producing an
identical `(N, NODE_ORDER_HASH, CORPUS_HASH)`.

## D. What actually changed in the table

Rebuild: 1270.8 s, peak RSS 757.4 MB. `hotpot_rev2_diff.py` then streamed the old and new tables in lockstep
(`REV2_DIFF_AUDIT.json`):

```
N_BEFORE 5233329   N_AFTER 5233329   N_UNCHANGED True
NODE_ORDER_IDENTICAL True            ORDERED_NODE_ID_HASH_EQUAL True
CHANGED_NODE_COUNT 94                fields_that_changed {'content_hash': 94, 'text': 94}
ONLY_TEXT_AND_CONTENT_HASH_CHANGED True
all_old_texts_were_empty True        all_new_texts_equal_title True
```

`nodes.jsonl` grew by exactly **1,872 bytes**; 1872 / 94 = 19.9 chars, matching the independently measured mean
title length. The size delta is fully accounted for.

`n_sentences` is deliberately left at **0** on all 94: it counts *source abstract* sentences, so it remains a
faithful provenance marker that these rows came from an empty abstract.

**Independent verification** (`hotpot_rev2_verify.py`, `REV2_VERIFY.json`) recomputed everything from the
installed table, trusting no builder counter:

| check | result |
|---|---|
| recomputed `CORPUS_HASH` / `NODE_ORDER_HASH` vs stored | **match** |
| every one of 5,233,329 `content_hash` re-derived as `sha(curid + US + text)` | **0 mismatches** |
| strict lexicographic node ordering | **holds** |
| `EMPTY_TEXT_N` / `WHITESPACE_ONLY_TEXT_N` / `EMPTY_TITLE_N` | **0 / 0 / 0** |
| 94 audited ids: present, `text == title`, title non-empty | **94 / 94** |

That `NODE_ORDER_HASH` held while `CORPUS_HASH` moved is positive structural evidence, not a coincidence:
`NODE_ORDER_HASH` hashes node_ids alone, `CORPUS_HASH` hashes `node_id \t content_hash`. A text-only change is
*required* to move exactly one of them.

## E. Query independence and determinism (re-run end to end)

`qi_test.py hotpotqa` rebuilt the corpus from source three more times, 5037.4 s
(`query_independence_test.json`):

| run | eval subset | N | CORPUS_HASH | NODE_ORDER_HASH | nodes.jsonl sha256 |
|---|---|---:|---|---|---|
| FINAL | legacy | 5,233,329 | `1b7eeac2bfbd7100` | `8f043c35cbf629a2` | `c777ab4b7a64eaaa` |
| A | legacy | 5,233,329 | `1b7eeac2bfbd7100` | `8f043c35cbf629a2` | `c777ab4b7a64eaaa` |
| B | random_held:7 | 5,233,329 | `1b7eeac2bfbd7100` | `8f043c35cbf629a2` | `c777ab4b7a64eaaa` |
| none | none | 5,233,329 | `1b7eeac2bfbd7100` | `8f043c35cbf629a2` | `c777ab4b7a64eaaa` |

Node-id **sets** compared directly (streamed, not just hashed): identical, 0 only-in-FINAL / 0 only-in-run for
all three. The test is not vacuous — the A and B eval subsets overlap in only **33** of 2,000 questions.

```
CORPUS_QUERY_INDEPENDENT = True        DETERMINISTIC_REBUILD = True
```

The hash differs from the pre-fix `fe139de9…` because the text changed. What matters is that it does not vary
by query lane, and it does not.

## F. Gold and legacy integrity

`queries/{train,validation,test}.jsonl` and `eval_2000.jsonl` are **sha256-identical** to the pre-fix versions —
expected, since query→node resolution keys on `title`, which the fix never touched.

```
GOLD_REFS_TOTAL 195,704   RESOLVED 195,704   MISSING 0   AMBIGUOUS 0   ALL_GOLD_REFS_RESOLVE true
  train      180,894 / 180,894          validation 14,810 / 14,810
  eval_2000  n=2000, 4,000 refs, all resolve
titles_needed 535,840   matched 535,840   unmatched 0   ambiguous 0
lanes  LEGACY_CONTINUITY 2,000 | QUALITY_LOCKED 7,405 | SCALE_ALL 105,257 | overlap(LC in QL) 158
```

All three lane files are byte-identical to the pre-fix versions, and the lane corpus-hash assertion re-passes.

`legacy_comparison.json`, regenerated: **all 21 fields identical** to the pre-fix run. Independently confirmed by
direct set intersection — **0 of the 94 changed nodes are among the 507,494 legacy-mapped node ids**, so legacy
continuity is untouched rather than merely assumed to be.

## G. Feature reuse — recounted, not inferred

Recomputed with `reuse_map_kb.py` against the rev-2 text (token-ID equality under the frozen tokenizers;
memoization independently verified, k=20,000, **0 mismatches**), then checked per node against the 94 audited ids:

| | strip-delta | empty-abstract 94 | **total** | was (rev 1) |
|---|---:|---:|---:|---:|
| dense rows needing encode | 1,594 | 94 | **1,688** (0.0323 %) | 1,594 |
| SPLADE rows needing encode | 0 | 94 | **94** (0.0018 %) | 0 |

All 94 return `dense_reusable=false, splade_reusable=false, reuse_source=null`.

**Why each number moved.** Under rev 1 the 94 stored `""` and the matching Phase-C rows also stored `""`
(`text_with_links` was empty markup for all 94), so they were token-identical and counted as reusable — that is
why dense then read 1,594 rather than the builder's source-derived bound of 1,688. Under rev 2 they carry real
content and match nothing already encoded.

A collision was possible in principle and was tested for: **114,187 nodes corpus-wide have text exactly equal to
their own title**, so a fallback title could have matched some other article's stored encoder input. **0 of 94**
did. The arithmetic `1,594 + 94` turns out to be right for dense, but it is now measured rather than assumed —
and no arithmetic on the dense side would have predicted the SPLADE result.

**SPLADE reuse is no longer 100 %** — the first non-zero SPLADE requirement in canonical_v1. The 1,594
strip-deltas remain free there (uncased WordPiece discards leading/trailing whitespace) but the 94 are a genuine
content change no tokenizer normalization folds away.

## H. Artifacts, provenance, and what was deliberately not done

**§11 census** (`REV2_VERIFY.json`):

```
TOTAL_N 5,233,329   NONEMPTY_TEXT_N 5,233,329   EMPTY_TEXT_N 0   SEMANTIC_KNN_ELIGIBLE 5,233,329
```

Every node now yields a non-degenerate token sequence, so none is excluded from semantic KNN on the grounds of
having no encodable text. No node was dropped at any point; membership stayed source-defined throughout.

**Superseded, not deleted.** The complete rev-1 artifact set — including its `nodes.jsonl`, `reuse_map/`, and its
own query-independence test — is preserved at `data/final_canonical/_superseded_textualization_rev1/` with a
`SUPERSEDED.md` recording the full hash history. Artifact parity was checked: nothing that existed in rev 1 is
missing from the live directory.

**Re-pinned:** `SOURCE_CONTRACT.json` (the `NORMALIZATION_RULE` now states the second construction step —
previously it claimed the join was "the only construction step", which had become false), `dataset_manifest.json`,
`integrity_report.json`, `build_info.json`, `legacy_comparison.json`, `node_id_map_legacy.json`,
`query_independence_test.json`, `reuse_map/`, `queries/lanes/`, top-level `MANIFEST.json`,
`CANONICALIZATION_AUDIT.{json,md}`, `SOURCE_CONTRACTS_FOR_REVIEW.md`, `ARTIFACT_REUSE_REPORT.md`,
`DOWNSTREAM_REBUILD_POLICY.md`, `_WEBQSP_ACCEPTANCE_GATE.json`, and the `results/` mirror. Regenerating the
other five datasets' manifests changed **only** their `manifest_written` timestamp; their contracts are
byte-identical.

Three documents had asserted *"the actual figure is lower: 1,594, not 1,688"*. That statement was true of rev 1
and is false of rev 2; it has been corrected everywhere rather than left to age, and the reason it changed is
recorded alongside it.

**Builder identity:** `build_kb.py` `862e0b3c284fd12a…` → `a46d745af5e77331…`. 2wiki's table was built by the
earlier sha and its `build_corpus_2wiki` path is untouched, so `CORPUS_HASH 81fa7d1a5d4bbb24…` stands.

**One defect found and fixed in the tooling.** `reuse_map_kb.py` returned a cached `.npz` when one existed, but
its step-4 cleanup deleted only the `.jsonl` spill files — so the canonical-side digest caches survived a run.
Re-running after a text change would have handed the 94 rows the digest of `""` and reported **rev 1's reuse
counts as freshly measured**. It now purges canonical-side caches per run while keeping the family caches, which
are keyed to immutable encoder-input shards (and which cut this run from ~60 min to 41 s). The stale files were
preserved, not deleted. This would have affected the 2wiki re-run too.

**Not done, by instruction:** no dense or SPLADE encoding, no ANN/KNN, no graph rebuild, no H4, SAFE, halo or
SP1. The 1,688 dense and 94 SPLADE rows are a *reported requirement*, not scheduled work. 2wiki's 87,765
empty-text nodes are a separate and larger decision: its forensics are complete
(`results/data_audit/final_canonical_v1/2wiki/EMPTY_TEXT_AUDIT.md`, `TRULY_TEXTLESS = 0`) and a rule is
proposed, but **nothing has been applied** to it.

written 2026-09-06

---

## Addendum, later the same day (2026-09-06) — 2wiki received the same rule

Two forward-looking statements above have since been overtaken by events and are corrected here rather than
edited in place, so the record of what was true when this report was written stays intact:

* *"`CORPUS_HASH 81fa7d1a5d4bbb24…` stands"* — no longer true. The user approved applying the identical
  textualization rule to 2wiki, and `TEXTUALIZATION_REV 2` was built and installed for it:
  `81fa7d1a5d4bbb24… → 94fe68b8d5f291f1…`, with `NODE_ORDER_HASH`, membership, node ids and node order
  unchanged and `empty_text_nodes` `87,765 → 0`.
* *"nothing has been applied to it"* — no longer true, for the same reason.

The builder moved again with that change, `a46d745af5e77331… → e723192e739de388…`; only the module docstring
and `build_corpus_2wiki` were touched, and hotpotqa's corpus path — including its own title fallback — is
covered by the 89-test suite that passes on the new sha. **hotpotqa's own hashes did not move:**
`CORPUS_HASH 1b7eeac2bfbd7100…` and `NODE_ORDER_HASH 8f043c35cbf629a2…` are still current.

The stale-cache defect described above was confirmed live on the 2wiki side before the re-run: the rev1
`verify_0.npz` / `verify_10000.npz` files were still sitting in `data/final_canonical/_work/reuse_kb/2wiki/`,
keyed to the pre-rev2 text, and a re-run without the patch would have reported rev1's 2wiki reuse counts as
freshly measured. The patched purge **deleted** them at the start of the 2wiki re-run, which is what the
instruction *"do not trust any pre-existing canonical-side `.npz` cache"* asks for; unlike hotpotqa's, they
were not copied aside first, because the patch now runs the purge automatically before any cache can be moved.
The files under `_superseded_textualization_rev1/_reuse_kb_rev1_canonical_side_caches/` are **hotpotqa's**
(`spill_todo_0.npz` n=1594, `verify_*.npz` idx≤5,233,304), preserved by hand on 2026-09-05 when the defect was
first found — they are not 2wiki's. Nothing of value was lost: these are derived digests, fully recomputable
from `nodes.jsonl` plus the frozen tokenizers, and the run that replaced them recomputes every one of them.

Full record: `data/final_canonical/_superseded_textualization_rev1/SUPERSEDED.md`.
