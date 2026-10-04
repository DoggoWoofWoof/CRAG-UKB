# Cleanup, fixes, verification and Freebase closure — 2026-09-11

The record of what was asked, what was done, what was deliberately **not** done, and the three
steps that are left for the owner to run by hand. Every figure below is copied from a record
named next to it; nothing here is the authority for anything.

Requests answered (verbatim):

1. *"can we fix all these issues, delete and merge the files and keep a note and cross check if
   those datasets or files are being references by anything and clean up all and keep only the
   results for audit"*
2. *"verify the 6 datasets, and can we complete and finish freebase after all that too"*

---

## 1. What was wrong, and how each was resolved

| issue | resolution | record |
|---|---|---|
| **frozen musique kNN is stale** — 4,402 of 266,488 weights do not reproduce from the dataset's own vectors (max abs err 0.50; squad/metaqa sit at ~1e-6) | **Superseded, not repaired** (HANDOFF §11: a freeze describes; it does not repair). `musique/graph2/knn.npz` rebuilt with the parity-validated pipeline over the pointer-resolved vectors (265,366 edges), **every** weight audited against those vectors (max abs err 5.96e-07, 0 above 1e-2), declared in `musique/graph2/GRAPH_MANIFEST.json#supersedes.knn`, frozen by `LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json` (`324de57a…`, BUILDS_ON V1 `98bc1cf1…`). The frozen file is byte-identical and still pinned by `LOCKED_5_OF_5`. | `data/final_canonical/LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json`, `data/_family_v1/MUSIQUE_GRAPH2_KNN_WEIGHT_AUDIT.json` |
| **two trees can declare the same family** with nothing saying which one is served | **Precedence rule**: frozen `graph/` wins UNLESS the graph2 manifest carries an explicit `supersedes.<family>` block. Honoured by `pointer_resolver_v2` (`family_source`, `supersedes`, `CanonicalGraph(..., source="auto|frozen|graph2")`), `verify_manifest` (`<fam>@frozen_superseded` / `<fam>@graph2_shadowed` — an undeclared duplicate is *flagged*), `ukb_manifest`, `substrate_map`, `handoff_readiness`, `graph_degree_hub`. `pointer_resolver` (v1) still reads `graph/` only; it is superseded by `_v2`, not rewritten. | `data/final_canonical/pointer_resolver_v2.py`; §12(b) of `HANDOFF.md` |
| **142,633 damaged Phase-C dense rows** | **Not healed in place, on purpose.** A Phase-C "heal" and an `ENCODING_SHARD_HASHES_V2` were considered and rejected under HANDOFF §11; the rows stay routed around through `_dense_repair_patch` (128,740 live pointers; 0 PHASE_C pointers into a damaged row). | `HANDOFF_READINESS.json` §5 |
| **retrieval-cache slots had non-uniform authority** (metaqa/musique/squad freeze-pinned; hotpotqa/2wiki/webqsp measured from disk) | Regenerating `UKB_COMMON_MANIFEST.json` would have dropped the `provenance` / `superseded_slot_VERBATIM` blocks the earlier patch added. Folded into `ukb_manifest.s_retrieval_cache()` so a regeneration cannot lose the distinction; `SLOT_UNIFORMITY` now states `retrieval_cache_provenance_is_not_uniform`. | `data/final_canonical/UKB_COMMON_MANIFEST.json` (`3b390086…`) |
| **pin attribution**: after V2 every graph2 family was labelled "pinned by V2" | `ukb_manifest.declared_index` walks `[V1, V2]`; V1-added families stay attributed to V1, only the musique kNN supersession is V2 (mirrors `verify_manifest.declared_index`). | same |
| **stale status sentences still readable in place** ("webqsp BLOCKED", "kNN not built", "still open on the Freebase side", authority queue "IN_PROGRESS", V1 `family_source.knn = frozen`) | `SUPERSEDED_STATEMENTS_INDEX.json`: 25 statements over 9 files, each quoted verbatim by JSON pointer / line anchor (an entry whose statement is not there refuses to be written), the superseding record pinned by sha256, the current fact, and the pin or self-hash an edit would have broken. HANDOFF.md's three sentences were struck through in place with dated notes (§2c, pitfall 3, §10) and a §12 "Amendments of 2026-09-11" added. | `data/final_canonical/SUPERSEDED_STATEMENTS_INDEX.json` (`5560da4b…`) |

## 2. Cross-check before any deletion

`src/dataset_canonical/cleanup_reference_crosscheck.py` → `data/final_canonical/CLEANUP_REFERENCE_CROSSCHECK.json`
(224 candidates). A candidate is DELETE **only if all** of the following hold:

- no record cites it — by full path, by `freebase_v3/`-relative path, by distinctive basename,
  by member-file name, or by `dir-name/` (the by-name pass is what made this honest: several
  V3 intermediates are cited by name only);
- no live code (`src/`) references it; scratchpad-only references are listed, not counted;
- no `results/`, `archive/`, `docs/` reference (recovered_chats excluded);
- nothing resolves into it (`POINTER_INDEX` store paths, `RETRIEVAL_CACHE` files);
- for `ukb_storage` / `l2_corpus` trees the substrate token is absent from live code.

Labelling records (the cleanup plans themselves) are listed but do not count as citations.

| bucket | DELETE | KEEP |
|---|---:|---:|
| B staging & duplicates | 70 candidates, 4.31 GB | 4 |
| D freebase_v3 intermediates | 58, 17.52 GB | 36 |
| E per-doc NER postings (`_ner_work/`) | — | 10, 28.87 GB (REUSE class per `DOWNSTREAM_REBUILD_POLICY.md`; the webqsp copy kept by the same rule) |
| F legacy stack | 8, 2.91 GB | 38 |
| **total** | **136 candidates, 24.74 GB** | |

Two candidates that *looked* deletable were moved to KEEP after the by-name pass; the
`webqsp/_acquisition/nsm/extracted/webqsp/webqsp/*_simple.json` files are a dependency of
`k_semantics_and_splits.py` and were never candidates.

## 3. Cleanup pass 3 — planned, gated, **not yet applied**

`src/dataset_canonical/cleanup_pass_3.py` (dry run by default) re-derives the gate at run time:

- refuses anything PROTECTED, declared by a locked record, resolved by `POINTER_INDEX` /
  `RETRIEVAL_CACHE`, or git-tracked;
- `_dense_repair_patch/*__pNNN` staging parts must **byte-reproduce** their flat store
  (10/10 groups do: block-wise uint8 equality of `dense.npy` rows, `dense_ids`/`dense_rows`
  concatenation equal);
- `metaqa_temp/` must be identical to `data/raw/metaqa` at the same relative path
  (31/31 files identical);
- `crag_data_backup/` results must be preserved first (see §4);
- scratchpad candidates: bulk extensions and files > 5 MiB are deleted, small text notes kept
  (24 notes, 0.1 MB);
- every planned file is sha256-hashed **before** deletion and the hash goes into the record.

Dry run (`CLEANUP_PASS_3.json`, STATUS `DRY_RUN`, `126d48c4…`): **1,947 files, 24,737,155,536
bytes, 0 refusals** — B 155 files / 4.31 GB, D 1,561 / 17.52 GB, F 231 / 2.91 GB. 130 s.

**`--apply` was blocked by the auto-mode classifier** (bulk deletion) and was not retried or
worked around. Nothing has been deleted. The command is in §8.

## 4. Results kept for audit

- `crag_data_backup/results/level_2/{metaqa,squad}_level_2_reranking.{csv,json}` (4 files, none
  with a live counterpart) are copied, sha-verified, to
  `archive/baseline_results_2026-04/level_2/from_crag_data_backup/` with a README before the
  backup tree is removed. Of the backup's other 193 files, 73 are byte-identical to the live
  copy, 116 (1.53 GB) have no live counterpart and are pre-`_clean` 2wiki/musique/squad
  substrates superseded by the canonical package, and **4 differ from live**:
  `data/ukb_storage/metaqa/{centroid_pids.json, centroids.index, graph.pt, partition_map.json}` —
  older versions of files the live tree still has. Their sha256s are recorded in the pass-3
  record (before deletion) so the difference stays auditable.
- The **23 tracked `results/L2` files shown as deleted in `git status` are moves, not losses**:
  each HEAD blob is byte-identical to a file now under `results/L2/_archive/` or
  `results/L2/_backups/` (checked blob-by-blob, 23/23).
- No DELETE candidate is git-tracked; `data/` is gitignored.

## 5. Verification of the six (request 2, first half)

`verify_manifest.py --records --sizes --graphs --builders --full --out=data/final_canonical/UKB_COMMON_MANIFEST_VERIFICATION.json`

- record chain 5/5 → 6/6 → V1 → V2 all reproduce (self-hash and BUILDS_ON);
- graphs: 18 served families resolve; musique `knn: graph2 (265,366, OK)`,
  `knn@frozen_superseded: graph (266,488, OK, sha matches declaration)`; no undeclared duplicate;
- **full byte pass: 203 ok, 0 bad, 4 expected drift** (HANDOFF.md's 5-of-5-era hash and the
  three additive POINTER_INDEX/ID_BRIDGE drifts, all pre-explained), **0 unpinned**, 28.9 GB in 78 s;
- `UKB_COMMON_MANIFEST.json`: 6 datasets, 13,997,071 nodes, 877,119 queries, 147,251,785
  served edges, 207 declared artifacts, uniform slots, 0 absent.

Regenerated with the precedence-aware readers, all PASS: `CANONICAL_SUBSTRATE_MAP.json`
(6×8 roles, 0 broken, code scan 0 violations in `src/`, musique frozen kNN listed under
`graph_superseded_frozen` and byte-identical), `HANDOFF_READINESS.json` (ALL SIX READY: True),
`GRAPH_DEGREE_HUB.json` (served families counted; the superseded frozen copy measured under
`knn@frozen_superseded`, marked `served: false`).

The HANDOFF claims gate (`scratchpad/verify_handoff_claims.py`, ~105 s) was re-run **after**
the HANDOFF.md annotations: first result **FAIL (4)** — two path spellings in the new §12
(`_family_v1/…` is really `data/_family_v1/…`; a bare `cleanup_pass_3.py` is not on the
checker's search roots) and two quoted digests (V2 `324de57a…`, closure `bff1453f…`) that the
checker's digest universe, fixed at LOCKED_6_OF_6 + LOCKED_5_OF_5 when those were the only
frozen records, did not contain. The spellings were corrected in the annotation (the corpus
table was not touched) and the universe widened to every frozen record on disk (`LOCKED_*.json`
at the root and under `freebase_v3/`, plus `FREEBASE_V3_CLOSURE_STATUS.json`; the record now
lists it under `checks.digests.digest_universe`). Second run: **ALL_PASS** — 87 paths named,
81 resolved, 6 templated, 0 missing; 4 digests quoted, 0 unmatched; edge columns still sum to
the frozen 112,192,284. `HANDOFF_CLAIMS_CHECK.json` rewritten. All 25 entries of
`SUPERSEDED_STATEMENTS_INDEX.json` re-verified in place afterwards (3 HANDOFF anchors at lines
761/92/214, all struck); the index itself was not re-minted, its self-hash `5560da4b…` reproduces.

The two older cleanup gates were dry-run again with the supersession keys in place:
`canonical_dedup.py` — authority-level uniqueness holds for all six (one `nodes.jsonl`, split
files only, `graph`+`graph2` as the documented family-completion split), 0 deletable, 0 blocked,
the Sep-10 applied history carried forward; `cleanup_plan.py` — RECLAIMABLE **0.00 GB** (all 808
Sep-10 paths are gone), safety gate 0 violations across 4 checks, KEEP_DEPENDENCY 2,535 files /
90.06 GB (report written to the session scratchpad, the Sep-10 `CANONICAL_CLEANUP_PLAN.json` kept
as the audit copy). So pass 3 (§3) is the only deletion still outstanding.

Closing run after every edit of the day: `verify_manifest.py --records --sizes --graphs
--builders --full` → **OVERALL PASS**, 203 ok / 0 bad / 4 expected drift / 0 unpinned, 28.9 GB in 81 s.

Re-run the same chain after the deletion (§8) — the gate is cheap and the deletion is not.

## 6. Freebase (request 2, second half)

`data/final_canonical/freebase_v3/FREEBASE_V3_CLOSURE_STATUS.json` (`bff1453f…`): the layer is
**closed for computation**. Nine frozen layers pinned by sha256 of their records: raw-mirror
provenance lock, `CRAG_FREEBASE_CANONICAL` (301,977,131 nodes / 2,062,430,072 edges,
ALL_CHECKS_PASS, `FROZEN_WITH_ORACLE_GATE_OVERRIDDEN`), `CRAG_FREEBASE_RESOLUTION_OVERLAY_V1`,
provenance order V2 (LOCKED by user ruling 2026-09-08), exact-source sweep (534,532 named,
3.0013% of the ENTITY_MID residue; MusicBrainz and Discogs passes **completed** — the
authority queue's `IN_PROGRESS` is superseded), actual-name layer, inference layer
(62,022,081 admissible), name hierarchy (72.6685% named by tier 1 or 2), semantic text.

What is open, **by class** — none of it is a pass this repository can run on its own:

| class | item |
|---|---|
| BLOCKED (never CLOSED) | stanford 106 (bot challenge), thetvdb 153 and giantbomb 22 (API key), netflix 8 (retired), tvrage 3 (defunct) |
| NOT RUN — external acquisition, needs your approval | archived topic-page expansion: 30,570 mids, naive projection 6,413 names / 17.2 h — an extrapolation across populations; the record recommends sampling a few hundred first |
| PROPOSED, awaiting acceptance | URL site-chrome strip (5,302 rows); untyped-terminal classification (original form refuted, re-derived, not applied) |
| OPEN DECISION | type-mirror edges: keep both layers vs pre-register `ALL_NODE − TYPE_MIRROR_EDGES` as an ablation (reversible, one relation UID) |
| REFUSED permanently | 58,144 freeq pseudo-titles |
| INFEASIBLE here | weak components of the 302M-node graph (RAM); dense/SPLADE over the universe (CPU) — and not part of the six-dataset package, whose KB corpus is `WEBQSP_ROG_STANDARD` (2,592,894 nodes), never "full Freebase" |

No external download was performed.

## 7. Git

- Nothing was committed or pushed (not asked).
- `.git` is bloated by **reflog-only** loose objects: `count-objects` 5,789 loose objects,
  14.69 GiB, plus 19 `tmp_obj_*` garbage files (51 MB) from interrupted adds. The 24 reflog-only
  blobs ≥ 50 MB total **16.13 GB** — old raw downloads and checkpoints
  (`data/triviaqa_rc.tar.gz` 2.67 GB, `data/audio_mp3-003.tar.gz` 2.64 GB, five MetaQA zips,
  `metaqa_movie` checkpoints, MRQA JSONs, `data/real/graph/*.pt`, webqsp silvergraphs).
- **None of those 24 has a copy on disk at its recorded path** (`docs/reflog_only_big_blobs_2026-09-11.tsv`).
  A `git gc --prune=now` after `reflog expire` would therefore destroy the *only* copies. That is
  your decision, not a cleanup step: §8 gives the recovery command to run first for anything you
  want back (e.g. the MetaQA zips, if `data/raw/metaqa` is not a sufficient extraction).

## 8. Left for you to run

Bulk deletion and git pruning are blocked for an agent in this session by design. Each block
is one command.

Apply cleanup pass 3 (24.74 GB, 1,947 files; the gate is re-derived at run time and the
record is rewritten as `APPLIED` with every deleted file's sha256):

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0 python src/dataset_canonical/cleanup_pass_3.py --apply
```

Then re-verify and regenerate:

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0 python src/dataset_canonical/verify_manifest.py --records --sizes --graphs --builders --full --out=data/final_canonical/UKB_COMMON_MANIFEST_VERIFICATION.json
```

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0 python src/dataset_canonical/substrate_map.py --scan-code
```

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0 python src/dataset_canonical/handoff_readiness.py
```

Git — **only if you want the reflog-only blobs gone**. First recover anything you want to keep
(example for one blob; the full list with sizes and ids is `docs/reflog_only_big_blobs_2026-09-11.tsv`):

```bash
mkdir -p archive/git_reflog_recovered && git cat-file blob c6d0d539c2aabd413ea9a9dcaebae357dc3647bc > archive/git_reflog_recovered/MetaQA-20251223T084132Z-1-001.zip
```

then:

```bash
git count-objects -vH && git reflog expire --expire=now --all && git gc --prune=now && git count-objects -vH
```

## 9. Lessons written to memory

- Bash heredocs on this Windows/MSYS shell mangle backslash-bearing Python (`"\\n"` became a
  literal newline and broke a file once this session; an earlier heredoc failed to parse at all).
  Patches with backslashes go through the Write tool as `.py` files and are executed with python.
- A reference cross-check must search **by name** as well as by path; the first pass would have
  deleted V3 intermediates cited only by basename.
- "A freeze describes; it does not repair" rules out in-place heals even when the fix is
  mechanically trivial; the honest move is a superseding record with a declared precedence rule.
- Precedence: **frozen wins unless graph2 declares `supersedes.<family>`**; an undeclared
  duplicate is a flagged defect, not a silent shadow.
- The musique kNN memory (`frozen-musique-knn-stale`) is now: stale, superseded by V2, frozen bytes untouched.
- A claims checker's universe is a dated artefact too: it must grow with the frozen records, and
  the paths a living document cites must be spelled the way the checker resolves them (the
  document's own convention: bare names for `scratchpad/`/`src/dataset_canonical/` scripts do
  **not** both resolve — cite `src/dataset_canonical/<x>.py` in full).
