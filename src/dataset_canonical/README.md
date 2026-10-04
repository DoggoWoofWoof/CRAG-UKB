# `src/dataset_canonical` — tooling for the canonical datasets

The six benchmark datasets live under `data/final_canonical/<ds>/`, one normalised tree each,
frozen by `data/final_canonical/CANONICAL_FREEZE.json` (2026-09-12). Read
`data/final_canonical/HANDOFF.md` first; this directory holds only the tools that built,
froze, verify and document that state. Nothing here builds a dataset from its official
source — the builders of record are still `scratchpad/final_canonical_build/*` (pinned by
sha256 in every `<ds>/build_info.json`; do not move or reformat them).

Run everything from the repo root with `PYTHONHASHSEED=0`; that value is part of dataset
identity and the tools refuse to run without it where it can affect output.

## The tools

| script | run when | writes |
|---|---|---|
| `verify_canonical.py [--full] [--dataset ds]` | any time you want to trust the tree: re-checks every pinned file against the freeze (`--full` re-digests every shard and recomputes every row-digest chain, ~20 min; without it ~3 min) | `data/final_canonical/VERIFICATION.json` |
| `check_handoff.py` | after editing `HANDOFF.md`: every number, path, digest and the quickstart are checked against the records | `data/final_canonical/_history/logs/HANDOFF_CLAIMS_CHECK.json` |
| `freeze_canonical.py [--dataset ds]` | only to re-freeze after a deliberate change to the served tree (a new record with a new hash; never to make a mismatch go away) | `<ds>/DATASET.json`, `CANONICAL_FREEZE.json`, `_history/INDEX.json` |
| `materialize_canonical.py {docs,queries,layout,graphs,direction,purge,all}` | the 2026-09-12 materialisation; nothing is left for it to do, kept so `_history/logs/MATERIALIZE_LOG.json` can be read against the code that wrote it | `_history/logs/MATERIALIZE_LOG.json` |
| `freebase_v3_closure_status.py` | the Freebase layer's closure record (separate layer, not part of the freeze) | `data/final_canonical/freebase_v3/FREEBASE_V3_CLOSURE_STATUS.json` |
| `_pre_v3/` | historical: the tools of the pointer-indexed layouts, kept so the records under `_history/records/` can be read against their code; not runnable against the tree as it is now (see `_pre_v3/README.md`) | — |

```bash
PYTHONHASHSEED=0 python src/dataset_canonical/verify_canonical.py --full
PYTHONHASHSEED=0 python src/dataset_canonical/check_handoff.py
```

## What the freeze pins

`CANONICAL_FREEZE.json` (self-hashed, `RECORD_SHA256` over the CRLF serialisation and
`RECORD_SHA256_LF` over LF) pins, per dataset: `DATASET.json` (itself pinning `nodes.jsonl`,
every split file, `query_ids.json`, every embedding manifest and shard, every graph npz and
the graph manifest, both retrieval caches, every build/audit record in the dataset directory
and the kept source directories) plus the governance files, the loader, and
`_history/INDEX.json` (a digest inventory of the audit trail). `verify_canonical.py` walks
exactly that.

## Conventions

* **Position i** is line i of `nodes.jsonl`, row i of the document embeddings, an endpoint in
  `graph/*.npz` and an id in the retrieval caches; **query j** is entry j of
  `queries/query_ids.json`, row j of the query embeddings and row j of the caches. There is
  no pointer index and no second root.
* **Records are superseded, never edited.** A frozen record that is wrong gets a successor
  with its own hash; the old one moves to `_history/records/` unchanged. The 2026-09-12 ruling
  ("i dont want versions … the rest other versions i give you permission to delete") applied
  to the *data*: superseded data copies were deleted, superseded records were kept.
* **Hashing convention** (`record_hash`): `json.dumps(record, indent=1)` with the two
  `RECORD_SHA256*` keys removed; the CRLF form is the primary digest because the first freeze
  hashed a Windows text-mode handle. `sha256sum` of the file never equals `RECORD_SHA256`.
* **Corpus subsetting by query is forbidden; query subsetting is allowed.** `eval_*.jsonl`
  and `queries/lanes/` are query subsets; every corpus is the whole corpus.
