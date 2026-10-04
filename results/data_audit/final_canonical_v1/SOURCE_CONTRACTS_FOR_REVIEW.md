# SOURCE CONTRACTS — canonical_v1 (Track B)

Rewritten 2026-09-05 after the user's per-dataset approval decision. Full contracts:
`data/final_canonical/<ds>/SOURCE_CONTRACT.json` (six files, identical field set).
Generator: `scratchpad/final_canonical_build/source_contracts.py`.

**Approval mechanism.** `data/final_canonical/_APPROVALS.json` is the authority and is read per dataset by
`scratchpad/final_canonical_build/build_kb.py::require_gates`. The old all-or-nothing file
`_APPROVED_SOURCE_CONTRACTS` is **obsolete, is never consulted, and was deliberately not created.** The heavy
gate `_GATE_HOTPOT_HEAVY_OK` still applies, and now gates the 2wiki full build as well
(`HEAVY = {"hotpotqa", "2wiki"}`). Each build records the gate state it saw in its `build_info.json`.

```
FULL_CANONICAL_V1_STATUS = READY_5_OF_6_SOURCE_CONTRACTS
WEBQSP_STATUS            = WEBQSP_BLOCKED_ON_FREEBASE_SOURCE
```

`LOCKED_6_OF_6` is **not** written and must not be while webqsp is blocked.

Invariant policed throughout:
> QUERY SUBSETTING IS ALLOWED. CORPUS SUBSETTING BY QUERY IS FORBIDDEN.
> FULL SOURCE CORPUS ≠ union of subgraphs/contexts associated with benchmark queries.

---

## 1. The table (DERIVED counts, not predictions)

| DATASET | FULL_SOURCE | RAW_N | CANONICAL_ID_RULE | EXPECTED_N | **DERIVED_N** | QUERY_INDEP. | STATUS |
|---|---|---:|---|---:|---:|---|---|
| **metaqa** | `entity/kb_entity_dict.txt` (+ `kb.txt` cross-check) | 43,234 | official dict index → `metaqa:e<idx:05d>` | 43,234 | **43,234** | YES | APPROVED as-is, unchanged |
| **2wiki** | `para_with_hyperlink.zip` → `para_with_hyperlink.jsonl` (**full article universe**) | 5,989,847 | official wiki **curid** → `2wiki:c<curid>` | 5,989,847 | **5,989,847** | YES | APPROVED_WITH_CONTRACT_CHANGE_REQUIRED → **contract changed, rebuilt** |
| **musique** | `musique_ans_v1.0_{train,dev,test}.jsonl` → all 20 paragraphs/question | 496,194 | `(title, paragraph_text)` exact | 117,534 | **117,534** | YES | APPROVED as-is, unchanged |
| **squad** | `train-v2.0.json` + `dev-v2.0.json` → all `context` | 20,239 | `(title, context)` exact | 20,233 | **20,233** | YES | APPROVED as-is, unchanged |
| **hotpotqa** | `enwiki-20171001-…-withlinks-abstracts.tar.bz2` | 5,233,329 | official wiki **curid** → `hotpotqa:c<curid>` | 5,233,329 | **5,233,329** | YES | APPROVED → **built** |
| **webqsp** | — none — | — | — | — | **none built** | — | **BLOCKED_PENDING_FREEBASE_SOURCE** |

**Total across the five built corpora: 11,404,177 nodes.**

> **Arithmetic discrepancy found in the authority file, not corrected.** `_APPROVALS.json` →
> `expected_final_corpus_table._subtotal_excluding_webqsp` says **11,403,177**, but its own five per-dataset
> targets sum to **11,404,177** (43,234 + 5,989,847 + 117,534 + 20,233 + 5,233,329). Every per-dataset target
> was met **exactly**; only the subtotal field is off, by 1,000. `_APPROVALS.json` was **not edited** — it is
> the user's authority file, and silently rewriting it would destroy the audit trail.

Full source SHA-256 + byte counts are in each `SOURCE_CONTRACT.json` → `SOURCE_HASHES` and in each
`build_info.json` → `source_files`. The two new corpus archives:

| file | sha256 | bytes |
|---|---|---:|
| `data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip` | `a585bdc3c39425446e4b2701a5f7f30051cb6c100d179322055d53dc0a71a723` | 1,900,740,270 |
| `data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2` | `1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729` | 1,553,565,403 |

### 1b. Query independence — executed, not asserted

Every corpus was rebuilt from scratch three extra times with a *different* `--eval-subset`
(`legacy`, `random_held:7`, `none`) and the three results compared against the shipped build. All four runs of each
dataset produced the **same CORPUS_HASH, the same ordered_node_id_hash, the same nodes.jsonl sha256 and the same node
count**, and the node-id sets were compared element-wise (`only_in_final = only_in_run = 0`).

| dataset | runs compared | CORPUS_HASH (first 16) | node count | CORPUS_QUERY_INDEPENDENT | DETERMINISTIC_REBUILD | lane hash assertion |
|---|---|---|---:|---|---|---|
| metaqa | FINAL + 3 | `5f55719a08c10e54` | 43,234 | **true** | **true** | PASS |
| 2wiki | FINAL + 3 | `94fe68b8d5f291f1` | 5,989,847 | **true** | **true** | PASS |
| musique | FINAL + 3 | `dc00da21ea8a0fd5` | 117,534 | **true** | **true** | PASS |
| squad | FINAL + 3 | `f592d4ab61956c1b` | 20,233 | **true** | **true** | PASS |
| hotpotqa | FINAL + 3 | `1b7eeac2bfbd7100` | 5,233,329 | **true** | **true** | PASS |
| webqsp | — blocked, nothing built — | — | — | — | — | — |

> **2wiki's row was re-executed from scratch on 2026-09-06** (5,083 s, three full rebuilds) after
> `TEXTUALIZATION_REV 2` moved its `CORPUS_HASH`. It was not re-pinned by editing the hash in this table:
> `legacy`, `random_held:7` and `none` were each rebuilt from the official sources under builder
> `e723192e…`, and all three independently reproduced `94fe68b8d5f291f1…`, the same `NODE_ORDER_HASH`
> `57dd809af573b6b5…`, the same `nodes.jsonl` sha256 `73fb822f3e4dd255…` and the same 5,989,847 node ids
> (`only_in_final = only_in_run = 0`, compared element-wise). The A/B eval subsets were disjoint
> (2,000 each, overlap 0), so the comparison is not vacuous. That reproduction is also the evidence that
> the rev-2 retextualization is a **general source-level rule inside the builder** rather than a patch
> applied to the installed table: a post-hoc patch would have left A/B/`none` regenerating
> the superseded rev1 `81fa7d1a5d4bbb24…` corpus, and the test would have failed. Record:
> `2wiki/query_independence_test.json`.

The lane assertion is `corpus_hash(LEGACY_CONTINUITY) == corpus_hash(QUALITY_LOCKED) == corpus_hash(SCALE_ALL)`,
recorded per dataset in `dataset_manifest.json → corpus_hash_equality_assertion` with both its bookkeeping form (one
`nodes.jsonl` serves all lanes; its hash recomputed from disk) and its constructive form (the rebuilds above).

Query-lane sizes (queries may be sampled; the corpus never is):

| dataset | LEGACY_CONTINUITY | QUALITY_LOCKED | SCALE_ALL | overlap LC∩QL |
|---|---:|---:|---:|---:|
| metaqa | 1,998 | 39,093 (official test) | 407,513 | — |
| 2wiki | 2,000 | 12,576 (official dev) | 192,606 | 0 |
| musique | 2,000 | 2,417 (official dev) | 24,814 | — |
| squad | 2,000 | 11,873 (official dev) | 142,192 | — |
| hotpotqa | 2,000 | 7,405 (fullwiki dev) | 105,257 | 158 |

### 1c. Gold-node coverage is reported as a measurement

`all_eval_gold_nodes_present` is **true** for metaqa, musique, squad and hotpotqa, and **false** for 2wiki. The 2wiki
value was diagnosed rather than accepted or quietly redefined (`dataset_manifest.json → gold_shortfall_diagnostic`):
all **8** affected questions are ones whose supporting-fact title `"Unconquered"` matches **two** distinct curids, so
`len(gold_node_ids) = len(gold_refs) + 1` and the strict equality test fails. `unresolved_gold_refs.counts` is empty
and `n_any_gold_resolved = n` for every split — **0 gold references are missing from the corpus**; the shortfall is
entirely ambiguity *expansion*, and both candidate nodes are kept rather than one being silently dropped. The flag's
definition was left unchanged, which is why `READY_FOR_INDEX_BUILD` is false for 2wiki.

---

## 2. 2WIKI — the contract that changed

### What was rejected

The previously accepted canonical_v1 2wiki table: **398,354 nodes**, corpus_hash `ec7fbbe8cd783040…`, identity =
exact paragraph *title*, node_id `2wiki:<sha256(title)[:24]>`; corpus = the deduplicated union of the `context`
field of all 192,606 official questions across all three splits.

It is invariant to the *evaluated subset*, but its membership is still a function of the *question set* — which
is precisely the defect for which HotpotQA's 507,494-node distractor-context union was rejected. The asymmetry
recorded as an open flag in the previous version of this document is now resolved **in favour of the full
universe**.

### What replaced it

`para_with_hyperlink.jsonl` (one member of `para_with_hyperlink.zip`, 7,023,046,781 bytes uncompressed), which
the official 2Wiki release describes as containing all articles/paragraphs with hyperlink information except
error paragraphs — i.e. the complete retrieval corpus.

Evidence that it is a Wikipedia-snapshot derivative and not a question by-product:

1. Record schema is `{id, title, sentences, mentions}` keyed by the Wikipedia curid. No question id, question
   string, answer, supporting_fact or split label appears anywhere in a record.
2. **5,989,847** article records vs **398,354** distinct context titles across every official question of every
   split ⇒ **93.3% of the corpus is never referenced by any 2Wiki question.** A question-derived collection
   cannot contain 5.59M unreferenced records.
3. The official release itself designates this file as the corpus.

Derived N **5,989,847** — streamed from the archive and counted; it happens to equal the expected value exactly,
but it was derived, not forced. `raw = 5,989,847 = nodes 5,989,847 + duplicates 0 + dropped 0`.

### Supersession, not deletion

The old artifacts were **moved**, not removed, to
`data/final_canonical/2wiki/_superseded_context_union_398354/` (nodes.jsonl, dataset_manifest.json,
integrity_report.json, legacy_comparison.json, node_id_map_legacy.json, query_independence_test.json,
build_info.json, reuse_map/, eval_2000.jsonl, SOURCE_CONTRACT.json, queries_lanes_superseded/, plus a new
`SUPERSESSION.json` recording the old hash and the four reasons for rejection). **The old node table is required
to remap the 398,354 already-encoded embedding rows, so it must never be deleted.**

### Node id

`2wiki:c<curid>`, readable rather than hashed. The curid is already source-stable and globally unique, so
hashing buys no collision safety and costs debuggability and remapping. Fixed before any node was written,
since node_id determines file order.

### Downstream

Every 2wiki downstream artifact (KNN, structural/NER graphs, H4, partitions, router caches, halo/SP1) is
**INVALID**. **None was rebuilt** — see `DOWNSTREAM_REBUILD_POLICY.md`.

---

## 3. HOTPOTQA — built

Corpus = the official FullWiki abstracts archive. Derived N **5,233,329**
(`raw = 5,233,329 = nodes + 0 duplicates + 0 dropped`), 15,517 tar members read.

**Why the abstracts archive and not the 7.4 GB `withlinks-processed` archive.** "Full corpus" means the
benchmark's full **official retrieval universe** — the collection FullWiki indexes and retrieves over — and that
is the introductory-paragraph corpus, one document per article. The larger archive holds the full body text of
the same articles: it is a strictly broader collection with a **different retrieval unit** (article body /
arbitrary passage instead of article abstract), so indexing it would be a different task, not a more complete
version of this one. (The 7.4 GB file is also not on disk; `results/data_audit` lists it under
`pending_downloads`.)

Node id `hotpotqa:c<curid>` — same scheme and rationale as 2wiki, fixed before any node was written. The
earlier draft's `hotpotqa:<sha256(curid)[:24]>` was never used by any artifact.

Canonical directory is `data/final_canonical/hotpotqa/`. `hotpotqa_clean` is a **legacy substrate alias only**:
its `status.json` was migrated (moved, not deleted) to
`data/final_canonical/hotpotqa/_legacy_alias_hotpotqa_clean/status.json` alongside an `ALIAS.json`, and the
duplicate directory no longer exists — there is exactly one live directory per dataset.

Text rules: `' '.join(non-empty sentences of the official 'text' list)`; **no `.strip()`**; empty abstracts
**kept** as nodes (94 of them — membership is never decided by a text-quality filter); **never** the
`text_with_links` fallback.

**TEXTUALIZATION_REV 2 (2026-09-05).** Those 94 nodes originally carried the empty string as their canonical
text. `gte-Qwen2` tokenizes `""` to the **zero-length sequence `[]`**, so all 94 collapsed onto a single
degenerate forward pass while the same source record carried a usable `title`. A second construction step was
added: *if the join is empty, use that record's own source-native `title`.* Fallback only — nothing is
prepended to a record that already has text, and an empty title stays empty. `text_with_links` was rejected by
measurement, not preference (readable residue **0/94** after tag-stripping; `charoffset == [[]]` for all 94).
Forensics: `results/data_audit/final_canonical_v1/hotpotqa/EMPTY_TEXT_94_AUDIT.md`.

The revision moved `CORPUS_HASH fe139de9… → 1b7eeac2…` and `nodes.jsonl 109377a6… → c777ab4b…`. It changed
**no** membership, node id, or node ordering: `NODE_ORDER_HASH 8f043c35…` is unchanged, which is exactly the
signature a text-only change is structurally required to produce (`NODE_ORDER_HASH` hashes node_ids alone;
`CORPUS_HASH` hashes `node_id \t content_hash`). A lockstep row-by-row diff confirmed only `text` and
`content_hash` moved, on exactly 94 rows, every one of which was previously empty and now equals its title.
The rev-1 artifacts are preserved under `data/final_canonical/_superseded_textualization_rev1/`.

**Phase-C divergence — MEASURED, not assumed:**

| divergence | records |
|---|---:|
| `.strip()` changes the joined text | **1,594** |
| Phase-C fell back to `text_with_links` because `text` was empty | **94** |
| **differing encoder inputs** | **1,688** of 5,233,329 (0.032%) |

This replaces the earlier "0–94 rows" projection in `ARTIFACT_REUSE_REPORT.md`, which had been downgraded to
*unmeasured* once the second divergence was found. It is now measured.

**Re-encode requirement, re-measured against the rev-2 text** (`reuse_map_kb.py`, token-ID equality under the
frozen tokenizers, then verified per node against the 94 audited ids — not inferred by adding 94 to the old
figure):

| | strip-delta | empty-abstract 94 | total | was (rev 1) |
|---|---:|---:|---:|---:|
| dense rows needing a re-encode | 1,594 | 94 | **1,688** (0.0323%) | 1,594 |
| SPLADE rows needing a re-encode | 0 | 94 | **94** (0.0018%) | 0 |

Under rev 1 the 94 empty canonical inputs were *token-identical to a stored Phase-C empty input* and so were
credited as reusable; that is why the dense figure then read 1,594. Now they carry real content and no longer
match. **SPLADE reuse is consequently no longer 100%** — the 1,594 strip-deltas remain SPLADE-free because its
uncased WordPiece is insensitive to leading/trailing whitespace, but the 94 are a genuine content change that
no tokenizer normalization folds away. No fallback title collided with any already-encoded input (checked: 0
of 94 reusable from any family). No encoder was run; the requirement is reported only.

---

## 4. WEBQSP — BLOCKED

No corpus was built and none can be: `build_kb.py` hard-codes `NEVER_BUILD = {"webqsp"}` and refuses
unconditionally — no flag, environment variable or argument lifts it.

- The RoG all-question subgraph union is **REJECTED**: membership is query-derived.
- The **1,316,466** figure is **not a canonical target**: `scratchpad/c1c2_webqsp.py:30-33` injected every
  question's answer entities and topic entities after collecting graph endpoints.
- **No Freebase dump exists anywhere on this machine** (exhaustive search recorded in `webqsp/status.json`).
- Measured, since the contract previously quoted an estimate: **59.01%** (776,788 / 1,316,466) of Phase-C webqsp
  rows have a bare MID as their entire text — exact, from a full streaming pass. The earlier **55.3%** came from
  the first 200,000 rows only. The bare-MID text is **not** preserved for embedding-reuse reasons.

Full proposal (source, version, URL, hash policy, entity/triple/relation counts, scope options, deterministic
textualization, storage, embedding cost, reuse analysis):
**`data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md`**.

`WEBQSP_ROG_UNION` may be retained as a **separate, clearly-labelled benchmark-specific experiment**; it must
never be presented as the webqsp member of the six-dataset full-corpus result.

---

## 5. Remaining judgment calls the reviewer should see

1. **2wiki asymmetry — RESOLVED** in favour of the full universe. The cost is that all 2wiki downstream
   artifacts are invalidated; the embeddings are **not** (see `ARTIFACT_REUSE_REPORT.md`).
2. **`_APPROVALS.json` subtotal is 1,000 low** (§1). Not edited. Per-dataset targets all met exactly.
3. **webqsp bare-MID fraction is 59.01%, not 55.3%** (§4). Conclusion unchanged.
4. **MetaQA QUALITY_LOCKED uses official TEST (39,093)** because MetaQA test labels are public, while every
   other dataset falls back to dev. Consistent rule, different resulting split.
5. **Readable node ids for the two curid-keyed corpora** but hashed ids for musique/squad/metaqa's index id.
   This is a deliberate per-identity choice (hash only when the identity key is a long text string), not an
   inconsistency — but it is visible in the manifests and worth a sentence in any write-up.
6. **`data/raw` vs `data/original`.** All corpus reads are from the immutable `data/original/<ds>/<version>/`
   copies. The legacy FlashRAG dumps appear in provenance text only, never as a canonical source.
7. **2wiki `all_eval_gold_nodes_present = false` is an ambiguity artefact, not a coverage gap** (§1c), and it is
   why `READY_FOR_INDEX_BUILD` is false for 2wiki alone. The strict definition was deliberately **not** relaxed;
   the explanation is stored next to the flag instead. A reviewer may reasonably decide the flag should count
   *unresolved* refs rather than exact cardinality — that is a definition change and was not made unilaterally.
8. **The 398,354 pre-existing 2wiki Phase-C rows are 100% token-compatible but contribute 0 marginal reuse**,
   because the full-universe store already covers the same nodes. Nothing was discarded and both stores remain
   on disk (`ARTIFACT_REUSE_REPORT.md` §2wiki).
9. **hotpotqa lanes overlap by 158 questions** (LEGACY_CONTINUITY ∩ QUALITY_LOCKED): the legacy 2,000-question
   eval subset draws 1,842 from train and 158 from the official fullwiki dev split, so those 158 appear in both
   lanes. Disclosed in `dataset_manifest.json → query_lane_overlaps`; it is a property of the frozen legacy
   subset, not a new sampling decision.

---

## 6. Builder integrity

**UPDATED 2026-09-05 — both builders were deliberately unfrozen and patched.** The previous text here said
`build.py` (sha256 `0879e428a41e636bb4089ef10c8ca217030331e22d54ca1d9dc87ba38988dcd9`) "was not modified";
that is no longer true. The authorised change replaced the cardinality-based gold-presence test
`len(gold_node_ids) == len(gold_refs)` with per-reference resolution semantics (see
`integrity_report.json → GOLD_SEMANTICS`, and `_WEBQSP_ACCEPTANCE_GATE.json → 2WIKI_FIX_AT_SOURCE`).
Current builder identities:

| builder | datasets | sha256 |
|---|---|---|
| `build.py` | metaqa, musique, squad (+ superseded 2wiki) | `26455c9f72191dd46281d0d107d2236d8beec3168552ec67ddecd13832311e93` |
| `build_kb.py` **as it stands now** | 2wiki (TEXTUALIZATION_REV 2), webqsp | `e723192e739de38880f04d92fdfc788604d4cfd95c17529c5bd4a52633e326f6` |
| `build_kb.py` **as hotpotqa's table was built** | hotpotqa (TEXTUALIZATION_REV 2) | `a46d745af5e77331d424ae06cd73fc721b046d59c66332d331c4b1794237ec72` |
| `build_kb.py` **as 2wiki's rev1 table was built** | — superseded — | `862e0b3c284fd12a08a97e7b1cefd1de1467effffdd62025fe09fa81eef1c3b1` |

**Updated 2026-09-06.** These three rows are one rule applied twice, a day apart, not two policies.
`862e0b3c… → a46d745a…` added the empty-text title fallback and its accounting to `build_corpus_hotpot`;
`a46d745a… → e723192e…` added the *same* fallback to `build_corpus_2wiki` and generalised the module docstring
to state the rule once. Each dataset's stored `builder_sha256` is the sha of the builder that actually produced
its table, and neither is retro-fitted: hotpotqa's table was built by `a46d745a…` and was **not** rebuilt when the
builder moved to `e723192e…`, so its `CORPUS_HASH 1b7eeac2bfbd7100…` and `NODE_ORDER_HASH 8f043c35cbf629a2…`
stand exactly as built. The `a46d745a… → e723192e…` edit touched only the module docstring and
`build_corpus_2wiki`; that is a statement about the edit, and the evidence offered for it is the test suite
below, not a diff.
2wiki's table was rebuilt by `e723192e…` and its `CORPUS_HASH` moved `81fa7d1a5d4bbb24… → 94fe68b8d5f291f1…`
accordingly, with `NODE_ORDER_HASH`, node count, node ids and node order unchanged — the rev1 artifact set is
preserved whole at `data/final_canonical/_superseded_textualization_rev1/2wiki/`.

`build_kb.py` is not git-tracked, so no diff is quotable here. The regression evidence is the 89-test suite in
`scratchpad/final_canonical_build/test_build_kb.py`, which passes 89/89 on `e723192e…` and covers **both**
datasets' corpus contracts including each one's title fallback, plus a query-independence case that rebuilds each
corpus under a completely different question set and asserts identical `(n, NODE_ORDER_HASH, CORPUS_HASH)`.

Prior identities, retained so the audit trail is readable rather than rewritten: `build.py`
`0879e428a41e…8dcd9`, `build_kb.py` `00dfeb460392…4cbeb` and `862e0b3c284f…c3b1`. **No corpus changed by the
gold-semantics patch**: all five datasets were
re-run with `--queries-only`, which reloads the frozen `nodes.jsonl` and asserts its CORPUS_HASH,
NODE_ORDER_HASH, node count and file sha256 against the existing `integrity_report.json` before writing
anything, and the query-independence test was re-run end-to-end under the patched builders.

The two large corpora are built by the
separate streaming module `build_kb.py`, whose `NodeWriter` is verified byte-identical **and** hash-identical to
`build.py::write_nodes`, so all datasets share one CORPUS_HASH / NODE_ORDER_HASH definition.
`scratchpad/final_canonical_build/test_build_kb.py` passes 78/78 and pins this behaviour, including the gate
tests (webqsp refused unconditionally and writes no nodes; the obsolete `_APPROVED_SOURCE_CONTRACTS` path is
never even tested for existence).

**No encoder was run in this task.** Token-ID hashing is metadata and is the only tokenizer work performed.
