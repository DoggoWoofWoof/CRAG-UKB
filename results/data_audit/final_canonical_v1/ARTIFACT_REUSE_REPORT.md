# ARTIFACT REUSE REPORT — canonical_v1 (Track B, Phase 3)

Regenerated 2026-09-05 after the 2wiki full-universe rebuild and the hotpotqa fullwiki build, git `ba6bd71`;
**2wiki re-measured 2026-09-06** after `TEXTUALIZATION_REV 2` (`CORPUS_HASH 81fa7d1a… → 94fe68b8…`), which is the
first change in this project to make a non-trivial number of stored rows unusable.
Builders: `scratchpad/final_canonical_build/reuse_map.py` (metaqa, musique, squad) and
`scratchpad/final_canonical_build/reuse_map_kb.py` (2wiki, hotpotqa — streaming/parallel, same decision rule).
Per-node maps: `data/final_canonical/<ds>/reuse_map/shard_NNN.jsonl` + `reuse_map/index.json`. Nothing downsampled.

**No encoder has been run in this task.** Everything below is metadata: SHA-256 over **frozen-tokenizer token-ID
sequences**. Where an encode *is* required it is reported as a requirement and left unexecuted.

## The reuse rule actually applied

> User's rule: *"Do not decide reuse from raw-text equality alone."*

A node's dense/SPLADE row is reusable **iff the frozen tokenizer's exact token-ID sequence for its encoder input is
identical** to that of an already-computed row. Raw-text equality is recorded (`status`) but never decides.

| | frozen model | doc encoder input | truncation | citation |
|---|---|---|---|---|
| dense | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | the `text` field **verbatim** — no prefix, no title (only *queries* get `GTE_QINSTR`) | `max_seq_length` 32768 → none at these lengths | `src/experiments/canonical_encode.py:176-190`; `sentence_bert_config.json`; legacy UKB agrees (`data/ukb_storage/<ds>/gte_qwen/meta.json` → `doc_prefix: ""`) |
| splade | `naver/splade-cocondenser-ensembledistil` | same text | `truncation=True, max_length=256` | `src/experiments/canonical_encode.py:209`; `data/canonical/<ds>/encodings/splade/docs/status/shard_*.json` → `"max_len": 256` |

Tokenizers were loaded from the exact local HF snapshots
(`~/.cache/huggingface/hub/models--Alibaba-NLP--gte-Qwen2-1.5B-instruct/…`,
`models--naver--splade-cocondenser-ensembledistil/…`), offline.

**Memoization and its independent check.** At 5–6 M nodes each *distinct* encoder input is tokenized once and its
digest shared by every byte-identical input — memoization of a deterministic pure function, not a text-equality
shortcut. It was verified rather than asserted: `--verify 20000` re-tokenizes 20,000 randomly chosen shared inputs
in independent worker processes and compares digests.

| dataset | verify k | mismatches | result |
|---|---:|---:|---|
| 2wiki | 20,000 | 0 | PASS |
| hotpotqa | 20,000 | 0 | PASS |

## THE TABLE — measured, all five non-blocked datasets

| Dataset | Canonical N | Dense rows reusable | **Dense rows needing encode** | SPLADE rows reusable | **SPLADE rows needing encode** | source family |
|---|---:|---:|---:|---:|---:|---|
| metaqa | 43,234 | 43,234 | **0** | 43,234 | **0** | PHASE_C |
| **2wiki** (TEXTUALIZATION_REV 2) | **5,989,847** | **5,902,083** | **87,764** | **5,902,084** | **87,763** | PHASE_C_UNIVERSE |
| musique | 117,534 | 117,534 | **0** | 117,534 | **0** | PHASE_C |
| squad | 20,233 | 20,233 | **0** | 20,233 | **0** | PHASE_C |
| **hotpotqa** (TEXTUALIZATION_REV 2) | **5,233,329** | **5,231,641** | **1,688** | **5,233,235** | **94** | PHASE_C |
| **total (5 built)** | **11,404,177** | **11,314,725** | **89,452** | **11,316,320** | **87,857** | |
| webqsp | — BLOCKED, nothing built — | — | — | — | — | — |

> **2wiki row re-measured 2026-09-06.** It previously read `5,989,847 / 0 / 5,989,847 / 0`. That was correct for
> `TEXTUALIZATION_REV 1`, in which 87,765 nodes carried the empty string and the Phase-C universe store carried
> the empty string for the same curids, so the inputs were byte-identical and every row was reusable — *reusable,
> but reusing a degenerate forward pass over an empty input*. `TEXTUALIZATION_REV 2` replaced those 87,765 texts
> with their source-native titles, which is exactly the change that makes their stored rows unusable. The new
> figures were produced by re-running `reuse_map_kb.py` end-to-end after the patched purge removed the rev1
> canonical-side digest caches — **not** by adding 87,765 to the old total, which would have given the wrong
> answer on both sides (see the two off-by-one paragraphs below).

`webqsp` has no canonical node table and none may be built (`_APPROVALS.json` →
`BLOCKED_PENDING_FREEBASE_SOURCE`); its reuse question is analysed in `webqsp/FREEBASE_SOURCE_PROPOSAL.md` §5 and the
short answer there is *essentially no reuse*, because the existing 1,316,466 Phase-C rows were encoded over a
query-derived corpus whose text is 59.01 % bare Freebase MIDs.

## 2WIKI — the five numbers the brief asked for (re-measured under TEXTUALIZATION_REV 2, 2026-09-06)

```
FULL_2WIKI_N              = 5,989,847
DENSE_ROWS_REUSED         = 5,902,083     (98.5348 %)
SPLADE_ROWS_REUSED        = 5,902,084     (98.5348 %)
NEW_DENSE_ROWS_REQUIRED   =    87,764     ( 1.4652 %)
NEW_SPLADE_ROWS_REQUIRED  =    87,763     ( 1.4650 %)
```

`reuse_map_kb.py 2wiki --workers 4`, 1223.5 s, no encoder, no model loaded beyond the two frozen tokenizers.
Memoization re-verified in the same run: `k = 20,000, mismatches = 0`.

**Where each number comes from.**

1. `data/canonical/2wiki_universe/` — checked first, because it was the single largest cost question — already holds
   a **complete full-scale store**: `encodings/dense/docs/manifest.json` and `encodings/splade/docs/manifest.json`
   both report `complete: true`, `n_items = rows_covered = 5,989,847`, `shard_size 40,000`, 150 shards
   (dense gte-Qwen2-1.5B-instruct, dim 1536, fp16; splade splade-cocondenser-ensembledistil).
2. Its encoder inputs are `encodings/_src/docs/shard_*.jsonl`, keyed `2wu:<curid>`. The canonical identity is the same
   curid, so the join is an identity join, not a text join: **5,989,847 / 5,989,847 curids matched**.
3. Of those, **5,902,082 encoder inputs are byte-identical** and 87,765 are not — and the 87,765 are exactly the
   nodes `TEXTUALIZATION_REV 2` rewrote from `""` to their source-native title. Two independent measurements agree
   that nothing else diverges: `scratchpad/build_2wiki_universe.py:42`'s `.strip()` is still the only other rule
   difference between the pipelines and the builder measures its effect at
   `phase_c_divergence.n_records_where_strip_changes_text = 0`, and the reuse map's own count of canonical rows
   needing their own tokenization is **87,765 — not one more.**
4. Token-ID join against the full-universe family: `dense_from_PHASE_C_UNIVERSE = 5,902,083`,
   `splade_from_PHASE_C_UNIVERSE = 5,902,084`.

**Two off-by-one results that arithmetic would have got wrong.** 87,765 rows changed, but the dense bill is
87,764 and the SPLADE bill is 87,763 — because reuse is decided by token-ID equality against *the whole already-
encoded pool*, not against the row that happens to share the node's curid:

* **Dense, +1 reusable.** `2wiki:c320778`'s new text is the title `"Kagawa"`, which is byte-identical to a
  different stored encoder input (`2wu:28138851`). Its own stored row is unusable, but a token-identical row
  exists elsewhere in the store, so the node is served without a new forward pass. Predicted before the build by
  the forensic pass (`title_is_already_encoded_input_exact` true for exactly 1 of 87,765) and confirmed here.
* **SPLADE, +2 reusable.** One *more* than dense, in the direction the measured tokenizer asymmetry predicts:
  gte's BPE is whitespace- and case-preserving, while SPLADE's WordPiece lowercases, strips accents and
  normalises whitespace, so its equivalence classes are strictly coarser and it can match a stored input that
  dense cannot. The pre-build projection modelled this with a whitespace+lowercase proxy and predicted 1; the
  measurement found 2. **This is why the bill is measured and not projected** — the projection was right to
  within one row on the dense side and wrong by one on the SPLADE side, in the direction the coarser tokenizer
  makes possible.

**The legacy-side status counts did not move**, which independently confirms that none of the 87,765 rewritten
nodes is a legacy node: `EXACT 61,576`, `TEXT_CHANGED 2,539`, `NEW 5,925,732`, `LEGACY_ONLY 1,750` — identical to
the rev1 recount, as expected given the forensic finding that 0 of the 87,765 is ever a gold or context node.

### Attributing the bill to the rows the revision actually changed

The corpus totals above do not by themselves establish *that the revision is the whole cause*. That was measured
separately, by partitioning the per-node map on `REV2_CHANGED_NODE_IDS.txt`
(`scratchpad/final_canonical_build/_analysis/rev2_reuse_attribution.py`, 93.1 s, read-only, no tokenizer loaded;
full output `results/data_audit/final_canonical_v1/2wiki/REV2_REUSE_ATTRIBUTION.json`):

| partition | rows | dense reusable | dense needs encode | SPLADE reusable | SPLADE needs encode |
|---|---:|---:|---:|---:|---:|
| **changed by TEXTUALIZATION_REV 2** | 87,765 | 1 | **87,764** | 2 | **87,763** |
| **untouched by it** | 5,902,082 | 5,902,082 | **0** | 5,902,082 | **0** |
| corpus | 5,989,847 | 5,902,083 | 87,764 | 5,902,084 | 87,763 |

`ALL_ENCODE_DEMAND_COMES_FROM_THE_CHANGED_ROWS = true`, and all 87,765 changed ids were found in the map
(`ALL_CHANGED_NODES_PRESENT_IN_MAP = true`). Every changed row's `status` is `NEW`, and the only reuse credit any
of them draws is `PHASE_C_UNIVERSE` (1 dense, 2 SPLADE). So the 5,902,082 nodes the rule never touched cost
nothing — the claim "a text-only revision cannot invalidate an unchanged node's encoder row" is here a
measurement, not an argument.

**Distinct forward passes — the number that actually bounds the cost.** Rows are not passes: rows whose frozen
token-ID sequences are equal collapse into one. Counted from the map's own `dense_input_hash` /
`splade_input_hash`:

```
DENSE   87,764 rows -> 87,764 distinct forward passes   (no two of the new titles tokenize identically)
SPLADE  87,763 rows -> 87,655 distinct forward passes   (108 rows fold onto another row's token IDs)
```

The 108-pass saving is the same tokenizer asymmetry again: the 87,765 titles are *all distinct strings* (the
forensics measured `unique_titles = 87,765`, `duplicate_title_groups = 0`), so nothing folds on the dense side,
while SPLADE's lowercasing, accent-stripping and whitespace normalisation merges 108 of them. The pre-build
projection bounded this at ≤ 87,748 distinct SPLADE passes; the measurement came in below that bound, at 87,655.

**The pre-existing 398,354 rows were not discarded.** `data/canonical/2wiki/encodings` (the 398,354-row view that
matched the superseded context-union corpus) was mapped in as a second family, `PHASE_C_VIEW398`, and offered to the
same token-ID join. Measured compatibility (`_work/2wiki_view398_compatibility.json`):

| | rows | token-ID-equal to a canonical node's encoder input | not compatible |
|---|---:|---:|---:|
| dense | 398,354 | **398,354 (100 %)** | 0 |
| splade | 398,354 | **398,354 (100 %)** | 0 |

So every one of those rows *is* an exact compatible row. Their **marginal** contribution is nevertheless 0, because
the full-universe family is strictly larger and already supplies a token-identical row for the same node — the map
credits the first family that covers a node and records which one (`dense_reuse_source` / `splade_reuse_source`).
Neither store is deleted; both remain on disk and either can serve the 398,354 overlapping nodes.

Re-checked under `TEXTUALIZATION_REV 2`: `VIEW398`'s marginal contribution is **still 0**, and this is measured
rather than assumed — the run prints the running totals family by family, and adding `PHASE_C_VIEW398` after
`PHASE_C_UNIVERSE` left them at `dense 5,902,083 / splade 5,902,084`, unchanged. The 398,354-node view is a
*subset* of the universe by construction, and the rows the revision broke are empty-body articles that the
question-context union never contained, so there was no reason to expect it to rescue any of them — but the
number is reported because it was executed, not because it was expected.

**Not reusable, and not re-run here:** everything downstream of the embeddings (KNN, structural/NER graphs, H4,
partitions, router caches, halo, SP1) is invalidated by a 15× larger node set. See `DOWNSTREAM_REBUILD_POLICY.md`;
nothing downstream was rebuilt in this task.

## HOTPOTQA — the re-encode requirement, measured not assumed

Re-measured against the **TEXTUALIZATION_REV 2** node table (`CORPUS_HASH 1b7eeac2…`):

```
HOTPOT_N                  = 5,233,329
DENSE_ROWS_REUSED         = 5,231,641    (99.9677 %)
NEW_DENSE_ROWS_REQUIRED   = 1,688        (0.0323 %)     was 1,594 under rev 1
SPLADE_ROWS_REUSED        = 5,233,235    (99.9982 %)
NEW_SPLADE_ROWS_REQUIRED  = 94           (0.0018 %)     was 0 under rev 1
```

`data/canonical/hotpotqa/encodings/{dense,splade}/docs` is complete at 5,233,329 rows (131 shards), keyed
`hotpot_<curid>` — an identity join again, 5,233,329/5,233,329 curids matched.

canonical_v1's text can differ from the Phase-C encoder input in exactly two ways, both counted by the builder
(`integrity_report.json → corpus_accounting_detail.phase_c_divergence`):

| divergence | source | n | dense re-encode? | SPLADE re-encode? |
|---|---|---:|---|---|
| Phase-C applied `.strip()` to the joined abstract; canonical_v1 does not | `scratchpad/c1_hotpot_fullwiki.py:39` | 1,594 | **yes** | **no** |
| empty abstract: Phase-C fell back to `text_with_links`, canonical_v1 (rev 2) uses the source `title` | `c1_hotpot_fullwiki.py:38` | 94 | **yes** | **yes** |
| builder's `n_records_with_a_different_phase_c_encoder_input` | | 1,688 | | |

**Why both figures moved, and why neither was inferred.** Under TEXTUALIZATION_REV 1 the 94 empty-abstract nodes
stored `""`; the corresponding Phase-C rows *also* stored `""` (`text_with_links` was empty markup for all 94), so
they were token-identical and counted as reusable — that is why the dense figure then read 1,594 rather than the
builder's source-derived upper bound of 1,688. Under rev 2 those nodes carry their source `title`, which is a real
content change, so they match nothing already encoded. The recount was **executed** against the rev-2 text rather
than obtained by adding 94 to the old number, and then checked per node: all 94 audited ids come back
`dense_reusable=false, splade_reusable=false, reuse_source=null`. A collision was possible in principle — 114,187
nodes corpus-wide have text exactly equal to their own title — and was tested for; **0 of 94** matched any
already-encoded input.

**SPLADE is no longer 100 %.** The 1,594 strip-deltas still cost nothing there: they differ only in leading or
trailing whitespace, which SPLADE's uncased WordPiece discards, so all 1,594 keep an identical token-ID sequence.
The 94 are different in kind — actual content — and no tokenizer normalization folds them away. This is the first
non-zero SPLADE requirement in canonical_v1.

**Requirement reported, not executed** *(revised 2026-09-06, when 2wiki received the same rule)*: across the five
built datasets the encoder requirement is **89,452 dense rows (0.784 %)** with
`Alibaba-NLP/gte-Qwen2-1.5B-instruct` and **87,857 SPLADE rows (0.770 %)** with
`naver/splade-cocondenser-ensembledistil` — 87,764 / 87,763 of them 2wiki's and 1,688 / 94 hotpotqa's. Before
2026-09-06 this line read *1,688 dense and 94 SPLADE*; that was the whole bill only while 2wiki's 87,765
empty-text nodes were still empty. **No encoder was run**, for either dataset.

## Already-computed row families

| family | rows keyed to | dense | splade | complete |
|---|---|---|---|---|
| **PHASE_C** (metaqa, musique, squad, hotpotqa) | `data/canonical/<ds>/documents.jsonl` → `encodings/{dense,splade}/docs` | 1536-d fp16 | CSR vocab 30522 | yes |
| **PHASE_C_UNIVERSE** (2wiki) | `data/canonical/2wiki_universe/documents.jsonl` (5,989,847) | 1536-d fp16, 150 shards | 150 shards | yes |
| **PHASE_C_VIEW398** (2wiki) | `data/canonical/2wiki/documents.jsonl` (398,354) | 10 shards | 10 shards | yes |
| **UKB (legacy)** | `data/processed/master_nodes_<legacy>.json` row order → `data/ukb_storage/<legacy>/gte_qwen/nodes.npy` | 1536-d fp32 | `splade_top200_all.npy` (scores, not rows) | per substrate |

## Legacy-substrate status columns (a different question from reuse)

`EXACT / TEXT_CHANGED / NEW / LEGACY_ONLY` are measured against the **legacy UKB substrate** through
`node_id_map_legacy.json`. They describe continuity with the old experiments, not the encoder requirement.

| dataset | EXACT | TEXT_CHANGED | NEW | LEGACY_ONLY | dense rows needing encode |
|---|---:|---:|---:|---:|---:|
| metaqa | 0 | 40,151 | 3,083 | 0 | 0 |
| 2wiki | 61,576 | 2,539 | 5,925,732 | 1,750 | 87,764 |
| musique | 13,672 | 0 | 103,862 | 0 | 0 |
| squad | 19,029 | 0 | 1,204 | 0 | 0 |
| hotpotqa | 503,084 | 4,410 | 4,725,835 | 0 | 1,688 |

The two views disagree by design and that is the point: 10,759,716 nodes are `NEW` relative to the legacy substrates
and only 89,452 of them need a dense encoder pass, because Phase-C already covers the corpora at full canonical scale.
Neither dataset's `EXACT`/`TEXT_CHANGED`/`NEW`/`LEGACY_ONLY` columns are changed by TEXTUALIZATION_REV 2: they are
measured against the *legacy* substrate, and none of the rewritten nodes belongs to one. For hotpotqa that was
verified by direct set intersection against all 507,494 legacy-mapped node ids; for 2wiki the four columns came back
byte-identical to the rev1 recount from an independent full re-run of `reuse_map_kb.py`, which is the same statement
arrived at from the other direction, and is consistent with the forensic finding that 0 of the 87,765 is ever a gold
or context node.

### The 2wiki UKB detokenization cases (unchanged finding, re-stated for the new corpus)

All 65,865 legacy nodes vs their mapped canonical node
(`data/final_canonical/_work/reuse_targeted_cases.json`): 62,137 byte-exact; 3,575 punctuation-spacing diffs
(`"Rosenberg (August"` vs `"Rosenberg( August"`) of which **0** are dense-token-identical and 3,387 are
SPLADE-token-identical; 153 whitespace-only, 0 dense-identical, 153 SPLADE-identical. This costs nothing, because the
canonical rows are served by Phase-C rows encoded over the official text, not by the UKB rows.

### MetaQA — 40,151 legacy → 43,234 canonical

All 40,151 legacy entity ids resolve into the exact-case identity space (0 missing, 0 ambiguous); 3,083 canonical
nodes are `NEW` — the case variants the legacy `strip().lower().replace('_',' ')` collapse had merged. **0 of 40,151**
UKB dense rows are reusable, and not because of the case collapse: the legacy encoder input was the verbalized triple
bag (`"$. $ directed by Richard Brooks | …"`), while canonical text is the bare entity name — a different string by
construction. All 43,234 are covered by Phase-C rows.

## NER (node-local) reuse

NER is `spaCy en_core_web_sm` over `documents.jsonl` text (`data/canonical/<ds>/ner_manifest.json`), a node-local
function of the text, reusable under the same text identity.

| dataset | state | per-doc output on disk | `ner_reusable` in map |
|---|---|---|---|
| metaqa | NOT_APPLICABLE | — (`ner_available: false`, KB entity names) | false |
| 2wiki | PER_DOC_ON_DISK | `data/canonical/2wiki_universe/_ner_work/` (364 files, 2.6 GB, full universe) | **5,989,847 true** |
| musique | EDGES_ONLY | only `graph_ner.tsv` | false |
| squad | EDGES_ONLY | only `graph_ner.tsv` | false |
| hotpotqa | PER_DOC_ON_DISK | `data/canonical/hotpotqa/_ner_work/` (326 files, 1.8 GB) | **5,233,329 true** |
| webqsp | NOT_APPLICABLE | — (blocked; no corpus) | — |

(The older 398,354-node `data/canonical/2wiki/_ner_work/` (84 files, 326 MB) also survives and is a subset view.)
Re-running NER where only edges were kept is cheap. **No NER run was performed in this task.**

## Caveats a reviewer should weigh

1. `dense_reusable` means *the encoder input is token-identical to an already-encoded row*. It does **not** assert the
   stored vector is bit-identical to what a re-run would produce (fp16 storage, GPU nondeterminism, library
   versions). For retrieval this is the right criterion; for a bit-exactness claim it is not.
2. Reuse credit is assigned to the first family that covers a node (`dense_reuse_source` records which), so a family's
   *marginal* contribution can be 0 even when 100 % of its rows are compatible — exactly what happened to
   PHASE_C_VIEW398 above. Read the per-node map, not just the totals, before retiring a store.
3. In-memory lookups key on a 128-bit prefix of the token-ID digest; the full 64-hex digests are written to every row.
4. Dense hashes use the untruncated token sequence (max_seq_length 32768 ≫ any document here), so they are exact, not
   merely conservative. SPLADE hashes use the real 256-token truncation.
5. `data/final_canonical/2wiki/reuse_map/` has 5,991,597 rows for 5,989,847 nodes: the extra 1,750 are `LEGACY_ONLY`
   rows (legacy nodes with no canonical counterpart, kept so the map is a complete bipartite record).
