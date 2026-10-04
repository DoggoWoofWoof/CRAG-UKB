# Canonical freeze — 2026-09-12

What was done on 2026-09-11/12 to turn the versioned, pointer-indexed package into six
normalised frozen datasets, what was deleted, and what a reader must know that the records
do not say by themselves. The served state is described by
`data/final_canonical/HANDOFF.md`; this note is the diary. The freeze was made twice on
2026-09-12: the first record (`RECORD_SHA256 abd72f9bc14a707ba302d62225169c3cb91879c0ffb370a731781bccc312f513`,
23:57Z, quick-verified PASS) was superseded sixteen minutes later by the direction-flag
correction (§5.3). That first record, and the six `CANONICAL_GRAPH` manifests it pinned, were
**overwritten in place** by the second run of the tool, not moved to `_history/` — the one
place this campaign departs from the records convention. Its hash survives only in the
build log quoted here; nothing had been handed off or read against it.

## 1. The ruling

> "at the end of all this i dont want versions, i just want canonical frozen datasets which has
> all cleaned edges, embeddings, queries, everything that we might need in a normalized form
> for all, we will try to get freebase also to same level but till then atleast these 6 need
> to be ready and the rest other versions i give you permission to delete and freeup space"

together with the earlier *"then heal it, i want everything fixed"* and *"only keep the proper
datasets with their fixed stuff rather than bloating up all"*. This overrides the package's
own §11 convention ("a freeze describes; it does not repair") for the **data**: the damaged
rows were healed in place, the superseded copies were deleted. It does **not** override it for
the **records**: every superseded record was moved unchanged to `_history/` and its self-hash
still verifies; nothing frozen was edited.

## 2. What the tree looked like before, and why it had to change

* Vectors lived in `data/canonical/<build-tree>/encodings/` (the Phase-C trees, named after
  the build that made them, not the dataset) plus two patch stores, and were reached through
  a per-dataset **pointer index** `(src, row)`; `data/final_canonical/` held no vectors.
* 142,633 dense rows in the Phase-C stores were zero (the Modal volume divergence); the good
  copies lived only in `_dense_repair_patch/`.
* The REV2 re-textualisation of 2wiki/hotpotqa had its own encoder patch store.
* WebQSP's stores were sharded at 12,000 rows, the five at 40,000.
* The families added after `LOCKED_5_OF_5` (metaqa NER, hotpotqa/2wiki/webqsp kNN, webqsp
  NER, the musique kNN rebuild) lived in `<ds>/graph2/` under a precedence rule, beside the
  stale frozen musique kNN.
* Query stores for metaqa were three per-hop trees; 2wiki's queries lived in a docs-superseded
  tree.
* Four freeze records in a chain, ~45 root-level records, two pointer resolvers, and a HANDOFF
  whose stale sentences were struck through and indexed.

## 3. What was done, in order

| step | tool | result |
|---|---|---|
| heal (2026-09-11) | `_pre_v3/consolidate_v3.py` | 129,819 zeroed dense rows in the nine served stores overwritten in place from the repair vectors (the other 12,814 patch rows belonged to a webqsp docs store no pointer served and were skipped as orphans); REV2 rows folded into the nodes' own rows; every dense channel scans clean |
| fold (2026-09-11) | `_pre_v3/consolidate_v3_layout.py` | graph2 families folded into `graph/`; stale musique kNN retired; hop-tree query stores merged |
| materialise docs | `materialize_canonical.py docs --delete-old` | every docs store rewritten in **position order** from the pointer index, banded (1 GB dense / 0.5 GB splade) for 5.8 GB of RAM; proof = per-row sha1 of old and new store, `new[i] == old[row[i]]` for all i, chain digest recomputable from the new store; the V2 `resolved_vector_fingerprint` reproduced for the two channels the V2 record carries one for (hotpotqa, 2wiki dense docs); old store deleted after proof |
| materialise queries | `materialize_canonical.py queries` | query stores already in row order (identity pointer): moved, digested, manifests rewritten |
| layout | `materialize_canonical.py layout` | pointer indexes → `_history/materialize/`, reuse maps deleted (sha-logged), `query_ids.json` moved up, webqsp `_enc/`, `status.json`, `encoder_row_of_node.npz` archived or deleted (text equality checked before deleting `encoder_inputs.jsonl`) |
| graphs | `materialize_canonical.py graphs` | `GRAPH_MANIFEST.json` rewritten as `RECORD: CANONICAL_GRAPH` (per family: digest, n_edges, direction, provenance, origin, history) after re-reading every family npz |
| direction | `materialize_canonical.py direction` | row census of every family (reverse rows, self loops, distinct unordered pairs); every NER/kNN family is one row per unordered pair, so the blanket `directed: true` of the Phase-C ner/knn manifests was corrected to `directed: false` with the census recorded beside it; npz files byte-identical |
| purge | `materialize_canonical.py purge` | `data/canonical/` and `data/_retired_v3/` deleted (Phase-C manifests copied to `_history/phase_c/` first); the sha-checked `CLEANUP_PASS_3` list applied (only files whose digest still matched; missing skipped, mismatches kept) |
| freeze | `freeze_canonical.py` | root tidied into `_history/{records,docs,logs}`, shim resolver written, six `DATASET.json`, `CANONICAL_FREEZE.json`, `_history/INDEX.json` |
| verify | `verify_canonical.py --full` | PASS, 501 checks, 0 failures, mode `full`, 14 min; `VERIFICATION.json` pins freeze `86bc74e3d879252d` |
| handoff | `HANDOFF.md` + `check_handoff.py` | `HANDOFF.md` (326 lines, sha256 `bf0e981b9abb3866`) checked by `check_handoff.py`: ALL_PASS, 0 failures; 38 paths named / 35 resolved, 1 digests quoted / 0 unmatched; quickstart executed on all six; record `_history/logs/HANDOFF_CLAIMS_CHECK.json` |

## 4. Numbers

| dataset | nodes | queries | structural | ner | knn | dense | splade |
|---|---|---|---|---|---|---|---|
| metaqa | 43,234 | 407,513 | 133,582 | 5,949 | 97,920 | 1.38 GB | 0.15 GB |
| squad | 20,233 | 142,192 | 874,190 | 156,130 | 42,969 | 0.50 GB | 0.07 GB |
| musique | 117,534 | 24,814 | 2,744,076 | 800,278 | 265,366 | 0.44 GB | 0.15 GB |
| hotpotqa | 5,233,329 | 105,257 | 15,367,541 | 20,369,074 | 11,946,289 | 16.40 GB | 5.79 GB |
| 2wiki | 5,989,847 | 192,606 | 28,963,600 | 34,067,241 | 13,643,063 | 18.99 GB | 7.09 GB |
| webqsp | 2,592,894 | 4,737 | 8,309,195 | 2,994,802 | 6,470,520 | 7.98 GB | 1.07 GB |
| **total** | 13,997,071 | 877,119 | 56,392,184 | 58,393,474 | 32,466,127 | 45.69 GB | 14.33 GB |

* served edges 147,251,785 in 18 families; 24 embedding channels, 60.03 GB; tree 88.85 GB
* freeze `CANONICAL_FREEZE.json` RECORD_SHA256 `86bc74e3d879252d7d7fab6660370f7388b28a5879c66ceda6952b434a0f312f`, frozen 2026-09-12T00:13:22Z
* history `_history/`: 196 files, 0.30 GB
* deleted 111.72 GB in total: old vector stores replaced 1:1 by their materialised copies 53.14 GB; per-dataset layout leftovers 9.10 GB; `data/canonical/` 22.33 GB; `data/_retired_v3/` 2.85 GB; cleanup pass 3 24.30 GB (1,890 files, 57 already gone, 0 kept on digest mismatch); net freed ≈ 58.58 GB


## 5. Things a reader should know that no record states on its own

1. **The V1 resolver pin is not reproducible.** `LOCKED_5_OF_5` pins `pointer_resolver.py`
   at 10,298 B; that file was overwritten before it was archived. `_history/pre_v3/
   pointer_resolver_v1.py` is a reconstruction that matches the surviving `.pyc` (bytecode,
   constants, line tables); the byte pin itself cannot be re-verified. Stated in the freeze
   (`LINEAGE.v1_resolver_note`).
2. **Duplicate positions are real, not a bug.** WebQSP has 801,361 positions whose rendered
   text equals another position's, so their rows are byte-identical vectors; musique has one
   such pair (74545/8381). The materialisation copied a row once per position; the manifests
   carry `duplicate_positions`.
3. **Direction differs by family, and the frozen flag was wrong for six families.** The
   structural families are directed as found in the source (real reverse rows: squad 258,912,
   musique 185,592, hotpotqa 544,508, 2wiki 1,509,486, webqsp 4,351,530 incl. 93,692 self
   loops, metaqa 11 self loops). Every NER and kNN family is stored as one row per unordered
   pair (census: 0 reverse rows, 0 self loops, distinct pairs == rows). The Phase-C manifests
   nevertheless declared `directed: true` for metaqa knn, squad ner+knn, musique ner, hotpotqa
   ner and 2wiki ner as a blanket convention, and the first freeze of 2026-09-12 carried that
   flag forward; a reader honouring it saw one endpoint's adjacency only. The flag was
   corrected to `directed: false` on the census evidence (`MATERIALIZE_LOG` phase
   `direction`, `storage_census` + `directed_as_frozen` in each manifest entry), the package
   was re-frozen, and no edge file changed. `canonical.Graph` builds both directions at load
   for undirected families; anything that reads the npz directly must read `directed` from
   the manifest.
4. **The 2wiki SPLADE cache is a mixed-backend artifact** (13 shards rented CPU, 11 local);
   the meta file says which.
5. **SPLADE pooling** is batch-dependent for the five and masked for webqsp; published, not
   repaired.
6. **The caches were not rebuilt** — they index positions and positions did not change; the
   proof is `max_id_seen == n_nodes − 1` and the unchanged digests pinned since their build.
7. **`freebase_v3/` is untouched** and will be the next thing brought to this form; the
   Freebase intermediates that pass 3 deleted (17.5 GB) were build by-products the closure
   record does not cite by any name. *(Later the same day: done — the seventh tree
   `data/final_canonical/freebase/` is built, verified and in the re-freeze; §8 and
   `docs/FREEBASE_CANONICAL_2026-09-12.md`. `freebase_v3/` stays as the source layer.)*

## 6. What only the user can decide

* `.git` holds 24 reflog-only blobs (16.13 GB) with no on-disk copies; `git gc` destroys the
  only copies (`docs/reflog_only_big_blobs_2026-09-11.tsv`).
* Bringing `freebase_v3/` to the same normalised form (a separate campaign) — *done later the
  same day, see §8; what remains for the user there is listed in the Freebase note*.
* Nothing was committed; the working tree carries the moves and the new tools.

## 7. Files

* served: `data/final_canonical/{CANONICAL_FREEZE.json, VERIFICATION.json, HANDOFF.md, canonical.py}`
  and the six trees; since the re-freeze (§8) also the seventh, `data/final_canonical/freebase/`
* audit trail: `data/final_canonical/_history/` (`INDEX.json`)
* tools: `src/dataset_canonical/{materialize_canonical, freeze_canonical, verify_canonical, check_handoff}.py`;
  earlier tools under `src/dataset_canonical/_pre_v3/` (see its README)


## 8. Addendum (2026-09-12, later): the re-freeze and the peer audit

The freeze above was superseded the same day by `CANONICAL_FREEZE.json`
RECORD_SHA256 `eaab0b8f534f5b2f297dccb515d7d761d6629c2cee1cbdcb7652f1a654a4bb5d` (frozen 2026-09-12T14:37:32Z; VERIFICATION `full`, PASS,
679 checks, 0 failures). The record quoted in §4 was **moved unchanged** to
`data/final_canonical/_history/records/CANONICAL_FREEZE_20260912T001322Z_86bc74e3.json` (its RECORD_SHA256 still verifies;
`LINEAGE.superseded_records`) — this time by the records convention, not overwritten. Between the two stands an intermediate record, `f2b8eee1…` (frozen 2026-09-12T13:37:14Z), which already carried
everything below and was verified (`full`, PASS, 678 checks, 3,261 s) before `check_handoff.py` ran; the gate then
rewrote `_history/logs/HANDOFF_CLAIMS_CHECK.json`, whose digest that record's `_history/INDEX.json` had pinned, so
every later `--full` verification of it would have failed by construction. The fix is in the tools, not the data:
`freeze_canonical.py` leaves that one file out of the index (`HISTORY.not_indexed`, with the reason) and
`verify_canonical.py` checks it by content instead (present, `ALL_PASS`, made on the served `HANDOFF.md`). The
intermediate record too was moved unchanged, to `_history/records/CANONICAL_FREEZE_20260912T133714Z_f2b8eee1.json`, and
the pipeline order is now fixed so the gate converges: freeze → verify → HANDOFF → `check_handoff.py` → verify again. Nothing
under the six trees changed; the six `DATASET.json` records were reused byte for byte
(`--reuse-dataset-records`). The re-freeze added the seventh tree (`freebase/`, see
`docs/FREEBASE_CANONICAL_2026-09-12.md`) and folded in three findings of a read-only peer
audit of the first freeze ("Report #6 checked against disk"), which had reproduced every
number in this note and raised three things that were the package's to fix:

1. **The query lanes decide the evaluation population, and the package did not say so.**
   `<ds>/queries/lanes/` holds three Track-B subsets. `LEGACY_CONTINUITY` (the legacy
   2,000-query eval subsets mapped onto canonical ids) is drawn from the **train** split on
   squad, musique and 2wiki, is 1,842 train + 158 validation on hotpotqa, and on metaqa is
   1,998 dev questions whose legacy gold sets differ from the canonical ones on 46 rows
   (`CANONICALIZATION_AUDIT`: the legacy substrate lower-cased entity names, 3,081 collided
   groups). `QUALITY_LOCKED` is the official held-out split with public labels — which for
   metaqa is the **test** split (39,093), contradicting nothing in the data but everything in
   the standing rule: anyone evaluating on that lane produces a test result. The freeze now
   carries `QUERY_LANES` (semantics, a census read from the rows, and the ruling: metaqa
   evaluates on dev per `EVAL_SPLITS`; `LEGACY_CONTINUITY` is for paired legacy comparison
   only), `KNOWN_LIMITATIONS` states it, HANDOFF §6 tabulates it, `verify_canonical.py`
   re-counts every lane file against the census, and `check_handoff.py` checks the table.
   Which lane is an experiment's population is the experimenter's decision; the package
   states the facts and the rule.
2. **One frozen statement was wrong.** `KNOWN_LIMITATIONS` said musique position 74545
   ("Adei Ad", §5.2) "has no edges in any family". True for structural and NER; false for
   kNN — the audited 2026-09-11 kNN rebuild searched the resolved vectors of all 117,534
   positions and gives it 3 edges (`GRAPH_MANIFEST` had it right). Corrected in the new
   record and marked as a correction; the wrong sentence stays in the archived record.
3. **The builders were not durable.** Every `build_info.json` named an untracked
   `scratchpad/final_canonical_build/build*.py`. `archive_builders.py` now snapshots every
   builder and tool source (`scratchpad/final_canonical_build/**.py|.sh`,
   `src/dataset_canonical/**.py`) into `_history/builders/` with
   `_history/builders/BUILDERS.json` (RECORD `BUILDER_SNAPSHOT`, 275 files),
   pinned by the freeze as `BUILDERS` and digested by `_history/INDEX.json`. The metaqa,
   squad and musique revision (`build.py` `26455c9f…`) and the 2wiki revision (`build_kb.py`
   `e723192e…`) match their archived copies. **The hotpotqa revision (`build_kb.py`
   `a46d745a…`) exists nowhere** — the 2wiki build of 2026-09-05 overwrote the file in place
   before any copy was kept; it is recorded as `LOST` (not silently dropped), the hotpotqa
   tree stays verified by its own records but is not reproducible from an archived builder.
   webqsp has no `build_info.json` (its build record is `CANONICAL_V1_BUILD.json`, naming no
   builder digest); its builders are archived at their current revision.

The peer's own stale rows (their `M3A` docs and `graph_substrate_stats.json` still quoting
the deleted musique kNN, 266,488) are theirs to withdraw; their two filings await the
owner's authorization and are not part of this package.

## 9. Addendum (2026-09-13): the terminal Freebase state

Re-frozen a third time as `58958f33a3af74c21fa625ffd79da16c573cb5d2ba7e9745da0056b568591ab0` (frozen 2026-09-12T22:27:25Z; VERIFICATION `full`, PASS, 686 checks, 0 failures)
to record the terminal state of the Freebase layer: the superseded `freebase_v3` sub-layer data (810
files, 20.73 GB) deleted under the owner's authorisation, sha-matched, records kept; the layer's closure
record V2 and the deletion log pinned as `FREEBASE.source_layer`. The record of §8 (`eaab0b8f…`) was moved
unchanged to `_history/records/CANONICAL_FREEZE_20260912T143732Z_eaab0b8f.json`. Nothing under the seven trees changed. The diary is
`docs/FREEBASE_CANONICAL_2026-09-12.md` §7.
