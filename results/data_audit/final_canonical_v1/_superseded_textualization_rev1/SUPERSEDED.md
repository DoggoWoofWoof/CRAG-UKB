# SUPERSEDED — `canonical_v1` TEXTUALIZATION_REV 1 (HotpotQA 2026-09-05, 2Wiki 2026-09-06)

This directory holds the **complete, unmodified** `canonical_v1` artifact sets for **both** heavy corpora as they
stood before `TEXTUALIZATION_REV 2` was applied. Nothing was deleted; in each case the rev1 build outputs were
moved here by filesystem renames on one volume, and the rev2 build was renamed into
`data/final_canonical/<ds>/` in their place.

The old builds are kept so the hash history is auditable and so any downstream artifact derived from
`fe139de9…` (hotpotqa) or `81fa7d1a…` (2wiki) can be traced to the exact table it was derived from.

```
data/final_canonical/_superseded_textualization_rev1/
  hotpotqa/                              the rev1 artifact set, verbatim (nodes.jsonl, reuse_map/, manifests, queries/, ...)
  2wiki/                                 the rev1 artifact set, verbatim  + INSTALL_RECORD.json
  _qi_rev1/                              hotpotqa's rev1 query-independence test: three build reports + verdict json
  _reuse_kb_rev1_canonical_side_caches/  HOTPOTQA's rev1 canonical-side digest caches (see "the cache trap" below)
  SUPERSEDED.md                          this file
```

## The rule — one rule, two datasets, applied at the source

Both datasets were revised by the **same general, deterministic, source-native, query-independent** rule. It is
not a per-dataset policy and it is not a list of node ids:

```
if canonical body is nonempty:              keep unchanged
elif source-native title is nonempty:       canonical_text = title
else:                                       canonical_text = ""
```

No title is prepended to any record that already has text. A record with an empty title stays empty. Membership
is never changed: a record that produces no text is **kept**, not dropped. Because the fallback reads a field of
the record's own source line, the rule is a pure function of the source and is unaffected by which queries are
evaluated — which is what the query-independence harness then re-establishes constructively rather than by
argument.

Recorded in each build's accounting as `TEXTUALIZATION_REV = 2`, together with `empty_before_fallback`,
`title_fallback_applied`, `still_empty_after_fallback`, `field_source` and `same_rule_as`.

---

# 1. HotpotQA — superseded 2026-09-05

## Why it was superseded

94 of the 5,233,329 nodes carried **empty canonical text** (`""`). `Alibaba-NLP/gte-Qwen2-1.5B-instruct`
tokenizes `""` to the **zero-length sequence `[]`**, so all 94 nodes collapsed onto a single degenerate
forward pass over an empty input — while the same source record plainly carried a usable human-readable
`title`. Full forensics, including the measured rejection of `text_with_links` as an alternative
(0/94 readable residue after tag stripping, `charoffset == [[]]` for all 94), are in
`results/data_audit/final_canonical_v1/hotpotqa/EMPTY_TEXT_94_AUDIT.md`.

## Hash history

| quantity | TEXTUALIZATION_REV 1 | TEXTUALIZATION_REV 2 | moved? |
|---|---|---|---|
| `canonical_node_count` | 5233329 | 5233329 | **no** |
| `NODE_ORDER_HASH` | `8f043c35cbf629a23e6acf2c2708b33124e3aa1a0044a7a9c179de18da3a823a` | `8f043c35cbf629a23e6acf2c2708b33124e3aa1a0044a7a9c179de18da3a823a` | **no** |
| `CORPUS_HASH` | `fe139de9c7c2a14f428ed98deb336269bff0cbb0befbb51bf8797ae8cce5ccac` | `1b7eeac2bfbd7100107f3eb252c947af333bf018c1dd9f7bdfbc3a5ea807653f` | **yes** |
| `nodes_jsonl_sha256` | `109377a6284c44d5393f16a5d43163fca5b854f0545efc41f388a5df5cbebf9f` | `c777ab4b7a64eaaa5af852265daf6ba9f9debb4658a67f0c96595662e43f182a` | **yes** |
| `nodes.jsonl` bytes | 3967224402 | 3967226274 | +1872 |
| `builder_sha256` | `862e0b3c284fd12a08a97e7b1cefd1de1467effffdd62025fe09fa81eef1c3b1` | `a46d745af5e77331d424ae06cd73fc721b046d59c66332d331c4b1794237ec72` | **yes** |
| `integrity_report.empty_text_nodes` | 94 | 0 | **yes** |
| `git_commit` at build time | `ba6bd71bcc9f367dc28d3e4cde777a3931501406` | `ba6bd71bcc9f367dc28d3e4cde777a3931501406` | no |

### Impact classification

```
MEMBERSHIP  change : NONE   -- 0 nodes added, 0 removed; identity is still the official Wikipedia curid
IDENTITY    change : NONE   -- every node_id is byte-identical
ORDER       change : NONE   -- ordered node-id hash equal, verified row-by-row in lockstep
TEXT        change : 94 nodes (all of which were previously exactly "")
```

`+1872 bytes / 94 nodes = 19.9 chars per node`, which equals the independently measured mean title length
of those 94 records (min 8, max 47). The size delta is fully accounted for.

### Measured diff (not asserted)

`scratchpad/final_canonical_build/hotpot_rev2_diff.py` streamed both tables in lockstep. Full output:
`results/data_audit/final_canonical_v1/hotpotqa/REV2_DIFF_AUDIT.json`.

```
N_BEFORE 5233329   N_AFTER 5233329   N_UNCHANGED True
NODE_ORDER_IDENTICAL True            ORDERED_NODE_ID_HASH_EQUAL True
CHANGED_NODE_COUNT 94
fields_that_changed {'content_hash': 94, 'text': 94}
ONLY_TEXT_AND_CONTENT_HASH_CHANGED True
all_old_texts_were_empty True        all_new_texts_equal_title True
```

The queries were **not** rebuilt in substance: `queries/{train,validation,test}.jsonl` and `eval_2000.jsonl`
are byte-identical (sha256) across the two revisions, because query→node resolution keys on `title`, which
the fix never touched.

### Encoder bill

`1,688 dense` and `94 SPLADE` rows, measured by re-running `reuse_map_kb.py` — never inferred by adding 94 to
the pre-rev2 figure of 1,594. The dense figure is 1,594 hotpot abstracts where Phase-C applied `.strip()` and
`canonical_v1` does not, **plus** the 94 rewritten rows. Those same 94 are the *whole* SPLADE requirement: the
1,594 whitespace deltas are invisible to SPLADE's uncased, whitespace-insensitive WordPiece, but a title where
there was previously an empty string is not.

---

# 2. 2Wiki — superseded 2026-09-06

## Why it was superseded

**87,765** of the 5,989,847 nodes carried empty canonical text — 937× hotpotqa's count, and 1.47 % of the
corpus. The forensic pass over the official archive
(`scratchpad/final_canonical_build/twowiki_empty_forensics.py`, 5,989,847 records read twice, 385 s + 453 s;
`results/data_audit/final_canonical_v1/2wiki/EMPTY_TEXT_AUDIT.md`) established every fact the decision rested
on, from source rather than from a supplied list:

```
TITLE_NONEMPTY            = 87,765 / 87,765     every one has a usable source-native title
ALT_TEXT_NONEMPTY         = 0                   the schema has exactly 4 keys; no other text field exists
TRULY_TEXTLESS            = 0                   none is genuinely without a human-readable name
empty_titles              = 0                   duplicate_title_groups = 0, unique_titles = 87,765
whitespace_only_text      = 0                   all 87,765 are exactly "", not " "
official_query_roles      = [] for all 87,765   never gold, never context -> no gold mapping can move
incoming hyperlinks       = 20,524 nodes (23.4 %) are the target of ≥1 link; 98,077 mentions from 96,482 articles
```

That last line is why it mattered structurally and not only cosmetically: nearly a quarter of these nodes are
**graph link targets**. Leaving them semantically empty puts 20,524 degenerate points into any Dense/KNN/H4
construction built over this corpus — and the decision was taken *before* those artifacts exist, so no
downstream rebuild was triggered by it.

## Hash history

| quantity | TEXTUALIZATION_REV 1 | TEXTUALIZATION_REV 2 | moved? |
|---|---|---|---|
| `canonical_node_count` | 5989847 | 5989847 | **no** |
| `NODE_ORDER_HASH` | `57dd809af573b6b531256b2a65e4b4efd1e26794c4fea81e0b43311fdcb24c86` | `57dd809af573b6b531256b2a65e4b4efd1e26794c4fea81e0b43311fdcb24c86` | **no** |
| `CORPUS_HASH` | `81fa7d1a5d4bbb24b0925e8ec65a4b59cb2416e87d6c5b7c20f74ac41a6d76d4` | `94fe68b8d5f291f1849d752fcc2349b024a1026a9e673d7f813ceae5fd9aa185` | **yes** |
| `nodes_jsonl_sha256` | `2653f6f35690ba31eeab5ea105ada6a85bad27a0ba07d4d3f943ed36e6185118` | `73fb822f3e4dd255c2555af717f4664c78a01fec4b5dce892fdec25d5de2426a` | **yes** |
| `nodes.jsonl` bytes | 5316499998 | 5318179753 | +1,679,755 |
| `builder_sha256` | `862e0b3c284fd12a08a97e7b1cefd1de1467effffdd62025fe09fa81eef1c3b1` | `e723192e739de38880f04d92fdfc788604d4cfd95c17529c5bd4a52633e326f6` | **yes** |
| `integrity_report.empty_text_nodes` | 87765 | 0 | **yes** |
| `git_commit` at build time | `ba6bd71bcc9f367dc28d3e4cde777a3931501406` | `ba6bd71bcc9f367dc28d3e4cde777a3931501406` | no |

`NODE_ORDER_HASH` is unchanged **because it hashes node_ids only**, while `CORPUS_HASH` hashes
`node_id \t content_hash` with `content_hash = sha256(cid + SEP + text)` (`SEP = "\x1f"`, `build_kb.py:78`).
A text-only change is therefore *required* to move `CORPUS_HASH` and *required* to leave `NODE_ORDER_HASH`
fixed. Observing exactly that, on both datasets, is positive structural evidence that only text moved.

### Impact classification

```
MEMBERSHIP  change : NONE   -- 0 nodes added, 0 removed; identity is still the official Wikipedia curid
IDENTITY    change : NONE   -- every node_id is byte-identical
ORDER       change : NONE   -- ordered node-id hash equal, verified row-by-row in lockstep
TEXT        change : 87,765 nodes (all of which were previously exactly "")
```

`+1,679,755 bytes / 87,765 nodes = 19.14 chars per node` against an independently measured
`title_len_mean = 18.93` (min 1, max 141); the 0.21-char residue is JSON escaping. The size delta is fully
accounted for.

### Measured diff (not asserted)

`scratchpad/final_canonical_build/rev2_diff.py 2wiki` — the dataset-general successor to the hotpot-specific
script — streamed both tables in lockstep for 275.1 s. Full output:
`results/data_audit/final_canonical_v1/2wiki/REV2_DIFF_AUDIT.json`; the complete changed-id list is at
`data/final_canonical/2wiki/REV2_CHANGED_NODE_IDS.txt`.

```
N_BEFORE 5989847   N_AFTER 5989847   N_UNCHANGED True
NODE_ORDER_IDENTICAL True   ORDERED_NODE_ID_HASH_EQUAL True   NODE_ID_SET_IDENTICAL True
CHANGED_NODE_COUNT 87765
fields_that_changed {'content_hash': 87765, 'text': 87765}
ONLY_TEXT_AND_CONTENT_HASH_CHANGED True
all_old_texts_were_empty True   all_new_texts_equal_title True   all_new_texts_nonempty True
```

### Independent re-verification of the installed table

`scratchpad/final_canonical_build/rev2_verify.py 2wiki` recomputed `NODE_ORDER_HASH`, `CORPUS_HASH` and
**every one of the 5,989,847 `content_hash` values** from the installed table (181.8 s), rather than trusting
the builder's own report. `results/data_audit/final_canonical_v1/2wiki/REV2_VERIFY.json`:

```
TOTAL_N 5989847   EMPTY_TEXT_N 0   WHITESPACE_ONLY_TEXT_N 0   EMPTY_TITLE_N 0
STRICTLY_ORDERED True
NODE_ORDER_HASH_MATCHES True   CORPUS_HASH_MATCHES True
content_hash_recomputed_for_every_node True   content_hash_mismatches []   CONTENT_HASH_ALL_CORRECT True
changed set: expected 87765 / seen_in_table 87765 / text_equals_title_and_title_nonempty 87765 / violations 0
```

`n_sentences` is deliberately left at 0 on the 87,765 fallback rows: it counts **source** sentences, so it
stays a faithful provenance marker of the empty source body.

### Queries and gold

`queries/{train,dev,test}.jsonl` and `eval_2000.jsonl` are **byte-identical** (sha256) across the two
revisions — 234.1 MB verified, not assumed — because 2wiki query→node resolution keys on `title`, which the
rule never touches. Gold semantics re-verified on the rev2 table: `GOLD_REFS_TOTAL 434,824`,
`MISSING_GOLD_REFS 0`, `AMBIGUOUS_GOLD_REFS 0`, `ALL_GOLD_REFS_RESOLVE true`, and all **8** `"Unconquered"`
question–title pairs still resolve uniquely to `2wiki:c8073502` (the 1947 DeMille film) by source-context
prefix agreement, 578 chars vs 15.

### Legacy continuity

`legacy_compare_kb.py 2wiki` was re-run end-to-end against the rev2 table (123.4 s) rather than inherited,
and every field came back identical to the rev1 result: `legacy_count 65,865`, `overlap_legacy_nodes_mapped
65,865`, `missing_in_canonical 0`, `ambiguous_legacy_nodes 2`, `extra_canonical_nodes_not_in_legacy
5,925,732`, `id_mapping_coverage 1.0`, `changed_text {exact_equal 62,136, whitespace_only_diff 153,
real_text_diff 3,576}`, and `eval_subset_gold_consistency 2000/2000 equal, 0 differ, 0 unmapped`. That is the
measured form of the claim that none of the 87,765 rewritten nodes belongs to the legacy substrate — expected,
since all 87,765 are empty-body articles and the legacy substrate is a question-context union, but executed
rather than argued.

---

## The cache trap — why the reuse recount is trustworthy

`reuse_map_kb.py` returned a cached `.npz` whenever one existed, and its step-4 cleanup deleted only the
`.jsonl` spill files — so the **canonical-side** digest caches (`spill_todo_*.npz`, `verify_*.npz`) survived a
run. Those digests are keyed to the canonical text. Re-running after a textualization change would therefore
have handed the changed rows the digest of `""` and reported the **previous revision's reuse counts as freshly
measured**. The patch purges canonical-side caches at the start of every run while keeping the family caches
(`<FAM>_*.npz` / `.npy`), which are keyed to immutable already-encoded encoder-input shards.

The trap was live on both datasets. HotpotQA's rev1 caches were moved aside by hand when the defect was found
and are preserved here in `_reuse_kb_rev1_canonical_side_caches/` (`spill_todo_0.npz` n=1594,
`verify_0.npz` / `verify_10000.npz` n=10000 each, max row index 5,233,304 — hotpot's row space). 2wiki's rev1
`verify_0.npz` / `verify_10000.npz` were still present under `data/final_canonical/_work/reuse_kb/2wiki/` and
were **deleted** by the patched purge at the start of the rev2 recount, which is exactly what the instruction
*"do not trust any pre-existing canonical-side `.npz` cache"* requires. Nothing of value was lost: these are
derived digests, recomputable from `nodes.jsonl` plus the two frozen tokenizers, and the run that removed them
recomputed every one.

## What downstream artifacts this invalidates

Anything keyed to `CORPUS_HASH = fe139de9…` (hotpotqa) or `81fa7d1a…` (2wiki) must be re-pinned to
`1b7eeac2…` / `94fe68b8…`. Within each dataset that is `dataset_manifest.json`, `integrity_report.json`,
`build_info.json`, `SOURCE_CONTRACT.json`, `legacy_comparison.json`, `query_independence_test.json`,
`reuse_map/`, and the dataset's entries in `data/final_canonical/MANIFEST.json` and the audit reports. All of
these are regenerated for rev2; the rev1 copies stay here.

**No encoder output is invalidated for an unchanged node** — its encoder input is byte-identical, hence
token-identical, so its dense/SPLADE reuse is unaffected. That is 5,233,235 of hotpotqa's nodes and 5,902,082
of 2wiki's. Only the changed rows can lose reuse, and the exact number is **measured** by re-running
`reuse_map_kb.py`, never inferred by addition.

### The measured encoder bill

| dataset | canonical N | dense reusable | dense needs encode | SPLADE reusable | SPLADE needs encode |
|---|---:|---:|---:|---:|---:|
| hotpotqa | 5,233,329 | 5,231,641 | 1,688 | 5,233,235 | 94 |
| 2wiki | 5,989,847 | 5,902,083 | 87,764 | 5,902,084 | 87,763 |

Attribution measured by partitioning 2wiki's per-node map on the changed-id list
(`_analysis/rev2_reuse_attribution.py`): of the 5,902,082 nodes the rule did **not** touch, **0** need a dense or
SPLADE row — `ALL_ENCODE_DEMAND_COMES_FROM_THE_CHANGED_ROWS = true`. Of the 87,765 it did touch, 1 keeps its
dense row and 2 keep their SPLADE rows, because a token-identical input already exists elsewhere in the store.

Neither bill is the arithmetic you would guess. Adding 87,765 to zero gives the wrong answer on both sides, and
the two sides differ from each other, for the same reason hotpotqa's dense figure went 1,594 → 1,688 while its
SPLADE figure went 0 → 94: the frozen tokenizers induce different equivalence classes. gte's BPE preserves case,
accents and whitespace; SPLADE's WordPiece lowercases, strips accents and normalises whitespace, so its classes
are strictly coarser and it can reuse rows dense cannot. In **distinct forward passes**, 2wiki's bill is 87,764
dense and 87,655 SPLADE — 108 of the new titles fold onto another's token IDs under SPLADE and none under dense,
because all 87,765 titles are distinct strings (`unique_titles = 87,765`, `duplicate_title_groups = 0`).

**No encoder was run for either dataset.** These are reported requirements.

No membership-, graph-, index-, partition-, halo- or SP1-level artifact existed for either corpus when these
revisions were applied, which is precisely why they were applied now rather than later.

written 2026-09-05 (hotpotqa), rewritten 2026-09-06 to cover both datasets
