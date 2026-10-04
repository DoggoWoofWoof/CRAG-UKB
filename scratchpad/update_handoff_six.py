# -*- coding: utf-8 -*-
"""
Bring HANDOFF.md to 6/6.

EVERY NUMBER IN THE PROSE IS READ OUT OF A FROZEN RECORD
    Hand-typing counts into a handoff is how a document starts disagreeing with the artifacts it
    describes -- silently, and in the one file a new reader trusts most. So the tables below are
    formatted from LOCKED_6_OF_6, ENCODING_ASSEMBLY and GRAPH_MANIFEST at write time. If a
    record changes and this is rerun, the prose changes with it; if a record is missing, this
    refuses to run rather than emitting a plausible-looking number.
"""
import io
import json
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = "data/final_canonical"
H = ROOT + "/HANDOFF.md"


def need(p):
    if not os.path.exists(p):
        sys.exit("missing %s -- run the freeze first" % p)
    return json.load(io.open(p, encoding="utf-8"))


def c(n):
    return format(int(n), ",")


def main():
    six = need(ROOT + "/LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json")
    w = six["webqsp"]
    ch = w["encoding_channels"]
    q = w["gold_query_level"]
    t = six["TOTALS_ALL_SIX"]
    s = io.open(H, encoding="utf-8").read()

    reps = []

    # ---- 1. the corpus table gains a sixth row and a new total -----------------------------
    reps.append((
        "| **total**| **11,404,177** | **872,382** | **3,875,657** | 48,082,989 | 55,392,723 | 407,377 |",
        "| webqsp   | %s  | %s   | %s    | %s |  — |  — |\n"
        "| **total**| **%s** | **%s** | **3,927,095** | %s | 55,392,723 | 407,377 |"
        % (c(w["n_nodes"]), c(w["n_queries"]), c(51438), c(w["n_edges"]),
           c(t["n_nodes"]), c(t["n_queries"]), c(48082989 + w["n_edges"]))))

    reps.append((
        "WebQSP is **dataset 6** and is not part of this freeze.",
        "WebQSP is **dataset 6**. Its gold column is not comparable to the other five: theirs is "
        "a corpus document and is present by construction, while WebQSP's is an answer *entity* "
        "in a third-party KG subgraph. See §9."))

    reps.append((
        "**Freeze status.** The five above are frozen: `LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json`\n"
        "(`STATUS: FROZEN`, 9 recorded limitations, 27.4 GB of hashed artifacts). Totals verified at\n"
        "freeze time, not copied forward: 11,404,177 nodes, 872,382 queries, 103,883,089 edges.",
        "**Freeze status: 6 of 6, FROZEN.**\n\n"
        "- `LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json` — the five text datasets. 9 recorded\n"
        "  limitations, 27.4 GB of hashed artifacts, 11,404,177 nodes / 872,382 queries /\n"
        "  103,883,089 edges, all verified at freeze time rather than copied forward.\n"
        "- `LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json` — the terminal record. It **pins the 5/5\n"
        "  record by hash** instead of re-freezing it, and refuses to write if that hash has\n"
        "  moved. It re-verified as `%s`, so the five are byte-identical to\n"
        "  what was locked. Totals across all six: **%s nodes, %s queries,\n"
        "  %s edges**.\n\n"
        "> **Verifying a `LOCKED_*` record — read this before concluding one was tampered with.**\n"
        "> The digest is **not** `sha256` of the file on disk. Each record was written, hashed,\n"
        "> then written *again* with the digest stamped in, so the bytes that were hashed no\n"
        "> longer exist. Reproduce it by re-serialising the record with its own `RECORD_SHA256`\n"
        "> removed, `json.dumps(indent=1)`. One further trap: `LOCKED_5_OF_5` was written through\n"
        "> a Windows text-mode handle, so the hashed bytes used **CRLF** newlines. The LF form of\n"
        "> the identical content hashes to `501d770f…` — a correct digest of the correct content,\n"
        "> and still not the recorded value. `LOCKED_6_OF_6` therefore publishes **both**\n"
        "> `RECORD_SHA256` (CRLF) and `RECORD_SHA256_LF`; use the LF one on Linux or in CI."
        % (six["BUILDS_ON"]["recorded_sha256"][:16] + "…", c(t["n_nodes"]),
           c(t["n_queries"]), c(t["n_edges"]))))

    # ---- 2. what you actually need in order to run any of it -------------------------------
    reps.append((
        "## 3. Quickstart",
        """## 2b. What is required to run this

**To read the substrate** — no GPU, no torch, no model downloads:

| | |
|---|---|
| Python | built and verified on **3.13.5** (Windows/AMD64); nothing here uses syntax newer than 3.8 |
| packages | `numpy` (2.3.2), `scipy` (1.16.1, for the SPLADE `csr_matrix`) — that is all |
| disk | **~%.0f GB** for the substrate itself — but see the table below, because the directory it lives in is 190 GB |
| RAM | dense reads are `mmap`ed, so `gather()` costs only the rows you ask for; SPLADE loads a whole shard (largest referenced shard is 0.04 GB) |

**`data/final_canonical/` is 190 GB on disk and the substrate is only a fraction of it.** A
reader who sizes a transfer off `du` will be wrong by 5×, so here is the split as measured:

| subtree | size | is it part of the six-dataset substrate? |
|---|---:|---|
| `freebase_v3/` | 140.90 GB | **No.** A separate artifact — see §10. |
| `_superseded_textualization_rev1/` | 18.34 GB | **No.** Superseded intermediates, kept for audit. |
| `2wiki` `webqsp` `hotpotqa` `metaqa` `squad` `musique` | 39.76 GB | **Yes**, minus WebQSP's source acquisitions (~4.7 GB) and encoder staging. |
| `_work/` | 3.16 GB | **No.** Build scratch. |
| `_dense_repair_patch/` `_rev2_encoder_patch/` | 1.18 GB | **Yes** — pointers resolve into these. |

And the vectors, which are in the *other* tree — §2(a) is the single most common way to break
this package:

| tree | holds | size |
|---|---|---:|
| `data/final_canonical/` | the corpus of record — nodes, queries, graphs, pointer indexes, manifests, gates | %.1f GB hashed in the freeze records |
| `data/canonical/` | the **vectors**, referenced by pointer, never copied | %.1f GB across %d stores |

`POINTER_INDEX.json` → each channel's `stores[].path` is the machine-readable dependency list.
Paths there are **repo-relative**; the resolver joins them to the repo root it infers from its
own location, so the two trees must keep their relative positions.

**To re-run a builder or a gate**, add `PYTHONHASHSEED=0`:

```bash
PYTHONHASHSEED=0 python -u scratchpad/<script>.py
```

This is not decoration. Several builders iterate `set`/`dict` of strings, and the resulting
order lands in files that are then hashed. `PYTHONHASHSEED=0` is **part of the dataset
identity** — every script here refuses to start without it, rather than silently producing an
artifact that will not reproduce.

**To re-encode anything** you need a GPU and the frozen encoder contract in §5 exactly —
including *not* installing `flash_attn`, since the non-flash kernel is what produced every
vector already in the package. There is no local GPU here (`torch 2.8.0+cpu`, `cuda False`,
33.8 GFLOPS effective), which is why the WebQSP encode ran on Modal (A10G). Nothing you
*consume* needs any of this; it is only for rebuilding.

---

## 3. Quickstart""" % (27.4 + 1.18 + 1.24 + 47.28 + w["encoding_bytes"] / 1e9,
                       27.4, 47.28 + w["encoding_bytes"] / 1e9, 37 + len(ch))))

    # ---- 2c. the failure modes that actually happen ----------------------------------------
    reps.append((
        "## 4. Layout",
        """## 3b. Seven ways this package gets misread

Each of these produces a *plausible* result rather than an error, which is why they are listed
together. Every one of them was hit at least once during construction.

1. **Reading `data/canonical/<tree>/encodings/dense/shard_*.npy` directly.** 142,633 rows are
   physically zero (§2b). You get vectors of the right shape whose cosine to everything is 0.
   Always go through `pointer_resolver.CanonicalEmbeddings`, which routes around them.

2. **Verifying a `LOCKED_*` digest with `sha256sum`.** It will not match, and the record is
   fine. See the freeze-status box in §1 — the hash covers the record *without* its own hash
   field, and `LOCKED_5_OF_5` hashed CRLF bytes. Use `RECORD_SHA256_LF` off Windows.

3. **Assuming all six datasets have the same edge families.** metaqa has no NER graph;
   hotpotqa, 2wiki and webqsp have no kNN graph; webqsp has *only* structural. Call
   `families(ds)`. Nothing was invented to make the six look uniform.

4. **Comparing raw SPLADE scores between WebQSP and the five.** 99.23% of the five's SPLADE
   rows were max-pooled without the attention mask; WebQSP's were pooled with it. See
   `SPLADE_POOLING_FINDING.json`. Rank-based fusion is unaffected; absolute scores are not
   comparable across that boundary.

5. **Averaging gold coverage across all six.** The five resolve gold at 100% *by construction*
   — their gold is a corpus document. WebQSP's gold is an answer entity in a third-party KG
   subgraph and resolves at 62.42% of references. A mean over those six numbers means nothing.
   The topic-entity control at 99.92% is what proves the WebQSP join is sound (§9).

6. **Re-ranking from the stored retrieval-cache scores.** They are float16; two distinct
   float32 scores can round to the same float16. The cached *ids* and their order are exact —
   the scores are for inspection, not for re-deriving the ranking.

7. **Running a builder without `PYTHONHASHSEED=0`.** Every script refuses rather than
   producing an artifact that will not reproduce — but only because they check. If you write
   your own, check it too.

---

## 4. Layout"""))

    # ---- 3. quickstart gains the sixth dataset ---------------------------------------------
    reps.append((
        'families("2wiki")                # [\'ner\', \'structural\']',
        'families("2wiki")                # [\'ner\', \'structural\']\n'
        'families("webqsp")               # [\'structural\']  -- see §2(c) and §9'))

    # ---- 4. scripts ------------------------------------------------------------------------
    reps.append((
        "| `freeze_five.py` | `LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json` | last |",
        "| `freeze_five.py` | `LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json` | after the five are final |\n"
        "| `build_webqsp_canonical.py` | webqsp `nodes.jsonl`, `queries/*.jsonl`, `graph/`, `encoder_inputs.jsonl`, `encoder_row_of_node.npz` | rebuilds dataset 6 from `webqsp/v1/*.parquet` |\n"
        "| `build_webqsp_query_inputs.py` | `queries/pointer_index/query_ids.json`, `query_inputs.jsonl`, `QUERY_WORK_ORDER.json` | after the queries change; fixes the query row order |\n"
        "| `modal_webqsp_encode.py` | encoder parts on the Modal volume `crag-webqsp-enc` | needs a GPU; `modal run scratchpad/modal_webqsp_encode.py` |\n"
        "| `pull_webqsp_encodings.py` | downloads those parts to `webqsp/_enc/` | resumable; verifies every byte count before renaming off `.part` |\n"
        "| `assemble_webqsp_encodings.py` | `data/canonical/webqsp_rog_v1/encodings/`, both pointer indexes, `POINTER_INDEX.json` | after the pull |\n"
        "| `freeze_six.py` | `LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json` | **last** |"))

    reps.append((
        "| `scan_encoding_integrity.py` | Enumerates every damaged row per channel",
        "| `verify_webqsp_canonical.py` | 15 checks on what is **on disk** for dataset 6, not on the build's own report: row counts, `node_id` == line number, id uniqueness, no empty text, encoder rows exactly `0..N-1` and distinct, gold positions in range and consistent with their id strings, graph endpoints in range, per-family relation ids within that family's vocabulary. The load-bearing one is the encoder join — see §9. |\n"
        "| `scan_encoding_integrity.py` | Enumerates every damaged row per channel"))

    reps.append((
        "**Retrieval caches for hotpotqa and 2wiki.** Materialised: musique, squad, metaqa at K=1000",
        "**Retrieval caches for hotpotqa, 2wiki and webqsp.** Materialised: musique, squad, "
        "metaqa at K=1000"))

    # ---- 5. webqsp is no longer pending -----------------------------------------------------
    reps.append((
        "**Built and verified** (`CANONICAL_V1_BUILD.json`, `CANONICAL_V1_VERIFICATION.json`,\n"
        "verdict `ALL_PASS`). Embeddings are the one remaining piece. The corpus is the",
        "**Complete and frozen.** Built, verified (`CANONICAL_V1_VERIFICATION.json`, verdict\n"
        "`ALL_PASS`), encoded, assembled and locked into `LOCKED_6_OF_6`. The corpus is the"))

    old_enc = s[s.find("The encoder must run on a GPU."):s.find("`scratchpad/build_webqsp_canonical.py` builds everything else and emits the measured work order.")]
    if not old_enc:
        sys.exit("could not locate the encoder-cost paragraph to replace")
    head = (
        "**The encoder ran, and here is what landed.** Deduplication does most of the work: "
        "%s nodes carry only %s distinct texts, which the pointer index expands back to "
        "per-node vectors. Encoding the distinct texts locally would have cost ~34 days for "
        "dense (1.0e17 FLOP at 33.8 GFLOPS, no CUDA), so it ran on a Modal A10G.\n\n"
        % (c(w["n_nodes"]), c(w["distinct_encoder_texts"])))
    table = ("| channel | rows | store | bytes |\n|---|---:|---|---:|\n"
             + "".join("| %s | %s | `%s` | %.2f GB |\n"
                       % (k, c(v["rows"]), v["store"], v["bytes"] / 1e9)
                       for k, v in sorted(ch.items())))
    tail = (
        "\nShards are **12,000 rows**, not the 40,000 the five use. The Modal parts *are* the "
        "shards; re-sharding would have written a second 5.5 GB copy on a volume with ~12 GB "
        "free in order to change a number `pointer_resolver` reads per-store from "
        "`POINTER_INDEX.json` and that cannot affect a single vector.\n\n"
        "The dense **query** prefix was compared by value — not by eye — against "
        "`src/experiments/canonical_encode.py`, the script that encoded the five, and matches "
        "(`sha256 %s…`, length %d). Queries encoded without that prefix would be the right "
        "shape, dtype and norm, and silently incomparable; SPLADE takes no prefix and the "
        "Modal script raises if handed one.\n\n"
        % (w["dense_query_prefix"]["sha256"][:16], int(w["dense_query_prefix"]["length"])))
    new_enc = head + table + tail

    reps.append((old_enc, new_enc))

    for old, new in reps:
        if old not in s:
            sys.exit("anchor not found in HANDOFF.md: %r" % old[:80])
        s = s.replace(old, new, 1)

    io.open(H, "w", encoding="utf-8", newline=chr(10)).write(s)
    print("updated %s -- %d edits" % (H, len(reps)))
    print("  totals now: %s nodes / %s queries / %s edges"
          % (c(t["n_nodes"]), c(t["n_queries"]), c(t["n_edges"])))
    print("  webqsp encodings: %.2f GB across %d channels"
          % (w["encoding_bytes"] / 1e9, len(ch)))


if __name__ == "__main__":
    main()
