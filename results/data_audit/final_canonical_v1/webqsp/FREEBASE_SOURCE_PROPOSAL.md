# WebQSP — Freebase source proposal (PROPOSAL ONLY, NOTHING BUILT)

**Status:** `WEBQSP_BLOCKED_ON_FREEBASE_SOURCE`.
`data/final_canonical/_APPROVALS.json` → `approvals.webqsp.status = BLOCKED_PENDING_FREEBASE_SOURCE`.
`scratchpad/final_canonical_build/build_kb.py` refuses `webqsp` **unconditionally** (`NEVER_BUILD = {"webqsp"}`);
no flag, environment variable or argument can lift that refusal. **No webqsp node was built in this task.**

This document is the deliverable the block asks for: what source a question-independent WebQSP corpus would
have to come from, what it would cost, and what of the existing artifacts survives. It approves nothing.

---

## 0. What is actually blocked, and why the block is not about text quality

Two different defects are easy to conflate. Only the first one blocks.

| # | Defect | Severity | Fixable from data on disk? |
|---|---|---|---|
| 1 | **Corpus membership is query-derived.** Every entity in the candidate corpus exists *because some question's RoG subgraph contains it*. | **BLOCKING** — violates strong corpus-independence | **No.** No amount of post-processing can add the Freebase entities that no question reached. |
| 2 | **Document text is a bare identifier for most rows.** 59.01% of the Phase-C rows have a bare Freebase MID as their entire text. | Serious quality defect | Partly (see §5), but fixing it does not lift the block. |

The `WEBQSP_ROG_UNION` corpus (1,316,466 rows) fails **(1)**. Fixing **(2)** inside it produces a
better-worded but still query-derived corpus. So the deduction trick described in §5.3 is **not** a route to
approval; it is only relevant to a clearly-labelled, benchmark-specific side experiment.

### The forbidden shortcut, stated so it cannot be reintroduced by accident

```
corpus  <-  source graph only
gold coverage  <-  MEASURED afterwards, and REPORTED when it is incomplete
```

Answer entities (`a_entity`), topic entities (`q_entity`) and gold nodes are **never** injected to make gold
coverage come out at 100%. `scratchpad/c1c2_webqsp.py:30-33` does exactly that injection, which is why its
**1,316,466 is not a canonical target** and must not be quoted as one.

---

## 1. Does a Freebase source already exist on this machine?

**No.** A full search was run before proposing any download.

| Search | Scope | Result |
|---|---|---|
| `*freebase*`, `*fb15k*`, `*FB2M*`, `*FB5M*`, `*mid2name*`, `*mid2title*`, `*entity_names*`, `*fb_entity*`, `*simplequestions*`, `*graftnet*`, `*embedkgqa*`, `*.nt`, `*.nt.gz`, `*.ttl`, `*virtuoso*`, `*.mid` | full recursive scan of `C:\Users\Swastik\Desktop\CRAG` | **0 hits** |
| same patterns | `C:\Users\Swastik` recursive, depth 4 | **0 hits** (ran to completion; not truncated) |
| files > 800 MB | `CRAG\data` recursive | 17 hits — all Wikipedia corpora, FAISS indexes and `.npy` embeddings. No KG dump. |
| HuggingFace cache | `~/.cache/huggingface` datasets + hub | `hotpotqa___hotpot_qa` and 9 encoder/LLM models only. No freebase / webqsp / cwq / grailqa / rog entry. |
| drives | `Get-PSDrive` | `C:` is the only drive. |

**Freebase-*derived* data that does exist** (all WebQSP-scoped, none a general dump):

| Path | Size | What it is |
|---|---|---|
| `data/original/webqsp/rog_webqsp/*.parquet` (5 files) | 494 MB | RoG per-question Freebase subgraphs (`graph` column). The only KG triples on disk. |
| `data/original/webqsp/WebQSP/data/WebQSP.{train,test}.json` | 14.8 MB | Official Microsoft WebQSP. Yields 39,882 unique MID→name pairs (topic/answer entities only). |
| `data/canonical/webqsp/documents.jsonl` | 558.6 MB | The Phase-C RoG-union corpus, 1,316,466 rows. |

**Therefore a download IS required** for a question-independent corpus. (A download is *not* required merely
to improve the text of the existing query-derived corpus — see §5.3 — but that is the non-blocking defect.)

---

## 2. Proposed source (source of record)

### Candidate A — Google Freebase final RDF data dump  *(PROPOSED)*

| Field | Value | Verified? |
|---|---|---|
| Source name | Freebase Data Dumps — full RDF export | — |
| Version / edition | the **final** dump; Freebase data was frozen 2015-03-31 and the API retired 2016-05-02. The last published RDF file is dated **2015-08-09**. | publisher-stated |
| File | `freebase-rdf-latest.gz` | publisher-stated |
| Canonical URL | `https://developers.google.com/freebase/` (landing page) → object at `https://commondatastorage.googleapis.com/freebase-public/rdf/freebase-rdf-latest.gz` | publisher-stated |
| Mirrors | Internet Archive item `freebase-rdf-latest` (and academic mirrors). **Any mirror used must be recorded by URL and hash.** | — |
| Format | N-Triples (`.nt`), gzip-compressed, one `subject predicate object .` per line, `ns:` = `http://rdf.freebase.com/ns/` | publisher-stated |
| Compressed size | ~22 GB | publisher-stated |
| Uncompressed size | ~250–425 GB (mirror-dependent claims) | **UNVERIFIED — must be measured** |
| Triple count | ~3.1 × 10⁹ triples (~1.9 × 10⁹ "facts") | **UNVERIFIED — must be derived** |
| Entity/topic count | ~5.8 × 10⁷ topics | **UNVERIFIED — must be derived** |
| Distinct relation (property) count | ~ tens of thousands | **UNVERIFIED — must be derived** |
| **sha256** | **UNKNOWN — must be computed on download and recorded** | — |

> **Honesty note.** Every number in that table marked *publisher-stated* or *UNVERIFIED* is a claim about a file
> that is **not on this machine**. None of them was measured here, and none may be written into a manifest as a
> derived count. The same discipline applied to 2wiki (expected 5,989,847 → **derived 5,989,847**) and hotpotqa
> (expected 5,233,329 → derived from the archive) must apply here: *derive the exact counts from the archive,
> do not force the expected number.* The sha256 in particular cannot be guessed and must be recorded from the
> actual download.

### Candidate B — `dki-lab/Freebase-Setup` Virtuoso image  *(alternative, for baseline parity)*

`https://github.com/dki-lab/Freebase-Setup` distributes the same 2015 dump preloaded into a Virtuoso database
(`virtuoso_db.zip`, tens of GB). It is the de-facto standard triple store for WebQSP / CWQ / GrailQA baselines,
so using it maximises comparability with published WebQSP numbers. Drawback: it is a **database image, not a
text corpus**, so a deterministic dump-out step would be needed, and the filtering it applies must be read out
of the repo and recorded rather than assumed. Its filtering is *not* question-derived, so it remains admissible.

### Rejected sources (recorded so they are not proposed again)

| Source | Why rejected |
|---|---|
| FB15k / FB15k-237 | a link-prediction benchmark subset, chosen by frequency of entities in a task set. Not a corpus. |
| FB2M / FB5M (SimpleQuestions) | subsets built around the SimpleQuestions question set → **question-derived**. |
| GraftNet / NSM / EmbedKGQA "Freebase subsets" | built by k-hop expansion **seeded from each question's topic entity** → exactly the defect that blocks the RoG union. |
| RoG `graph` column (already on disk) | per-question subgraphs → **query-derived by construction**. This is the thing being rejected. |

---

## 3. Corpus scope — the one remaining design decision (USER REVIEW POINT)

Given the dump, "which entities are nodes?" must still be answered **without reference to any question**.
Admissible options:

| Option | Rule | Expected N | Query-independent? |
|---|---|---|---|
| **S1** | every MID that carries an English `type.object.name` | to be derived (order 4–5 × 10⁷) | **Yes** |
| **S2** | every MID appearing as the subject of ≥1 `ns:` triple (i.e. all topics) | to be derived (order 5.8 × 10⁷) | **Yes** |
| **S3** | the `dki-lab/Freebase-Setup` filtered graph as published | to be derived | **Yes** (filter is question-free) |
| ~~S4~~ | k-hop expansion from WebQSP topic entities | ~10⁶ | **NO — FORBIDDEN.** Seeded by questions. |
| ~~S5~~ | RoG per-question subgraph union | 1,316,466 | **NO — FORBIDDEN.** This is the rejected corpus. |

**Recommendation: S1.** It is the smallest admissible scope that is defined purely by the source (an entity is
a node iff Freebase gives it an English name), it is the scope for which a meaningful text exists at all, and
the count of MIDs it *excludes* is itself a reportable measurement. S2 is defensible but forces a bare-MID
document for every unnamed CVT/mediator node — reintroducing defect (2) by construction.

Whichever is chosen, **gold coverage is measured afterwards and reported**, never engineered:

```
for each official WebQSP question: resolve its answer MIDs against the corpus;
  if an answer MID is absent -> REPORT IT.  Never add it.
```

---

## 4. Proposed textualization (deterministic, query-independent)

Built **only** from source metadata. No question, no answer, no subgraph, no ordering that depends on a query.

```
node_id      = "webqsp:m<MID with '/'->'.' normalised>"     # readable, source-stable, like 2wiki:c<curid>
identity     = the MID, exact string

name     = object of (m, ns:type.object.name, ?o) with xml:lang "en"
             several -> take the lexicographically first; COUNT AND REPORT the ties
aliases  = sorted(set of objects of (m, ns:common.topic.alias, ?o) with xml:lang "en"))
desc     = object of (m, ns:common.topic.description, ?o) with xml:lang "en"   [optional lane, see below]

text = name                                            if aliases is empty
     = name + " (also known as: " + "; ".join(aliases) + ")"    otherwise
```

Properties of this rule:

* **Deterministic** — a pure function of the dump; `sorted()` fixes alias order; the tie-break is explicit.
* **Query-independent** — no question file is opened by the corpus stage. Enforced structurally, as in
  `build_kb.py`, by never passing `--eval-subset` into `build_corpus_*()`.
* **Literal** — the official strings are stored verbatim: no NFC/NFKC, no case-folding, no whitespace
  collapsing, **no `.strip()`**. Lossy alternatives are *quantified* in
  `integrity_report.corpus_accounting_detail.identity_normalization_diagnostics`, never applied.
* **Fallback is reported, not hidden** — for S2, a MID with no English name yields `text = ""` (kept as a node,
  as 2wiki keeps its 87,765 empty-text articles) and the count is reported. It is **not** back-filled with the
  MID string: a bare MID is not a retrieval representation, it is an identifier leaking into the text field.

**REVIEW POINT — the description lane.** Adding `common.topic.description` (a one-paragraph gloss) would make
these documents far closer in nature to the five text corpora and probably retrieve much better. It also
multiplies the corpus text size by roughly an order of magnitude and changes every embedding. It is proposed
as an **explicit A/B decision to be made before encoding**, not silently.

---

## 5. Can any of the existing 1,316,466 rows be reused?

### 5.1 The measurement

| Quantity | Value |
|---|---|
| Phase-C webqsp rows (`data/canonical/webqsp/documents.jsonl`) | 1,316,466 |
| Rows whose entire text matches `^[mg]\.[0-9a-z_]+$` (a bare MID) | **776,788** |
| Fraction | **59.01 %** (exact, full streaming pass) |
| Text rule that produced them | `scratchpad/c1c2_webqsp.py:25` — `{"title": name, "text": name, ...}`, i.e. the raw graph-node string verbatim |
| Dense store | `data/canonical/webqsp/encodings/dense/docs`, complete, 4.09 GB, 3,103 B/row |
| SPLADE store | `data/canonical/webqsp/encodings/splade/docs`, complete, 0.35 GB, 266 B/row |

> **Discrepancy against the brief, disclosed.** The task brief states 55.3 % of Phase-C webqsp docs are a bare
> MID. The exact figure measured here by a full streaming pass over all 1,316,466 rows is **59.01 %**
> (776,788 rows). A 20,000-row byte-offset sample gave 57.56 %, but byte-offset sampling is length-biased
> against short lines and MID-only rows are the shortest lines in the file, so that estimate is biased low; the
> streaming figure is the one to use. The conclusion is unchanged and only strengthened.

### 5.2 The answer: **no meaningful reuse, and reuse is not a reason to keep the text**

1. **Scope changes.** Under S1/S2 the node set is ~40–58 million Freebase topics; the 1,316,466 RoG-union
   entities are a ~2–3 % slice of it. 97 %+ of the corpus is new whatever happens.
2. **Encoder input changes for essentially every surviving row.** Reuse is decided by **frozen-tokenizer
   token-ID equality**, never raw-text equality. A row survives only if
   `name` (+ aliases) is byte-identical to the RoG surface string it was encoded from. For the 776,788 bare-MID
   rows that is false by construction (their old input was `m.0abc123`, their new input is a human name).
   For the remaining 539,678 named rows it is true only where Freebase gives that entity **no** English alias
   and the RoG surface string is exactly the Freebase `type.object.name` — plausible for many, but
   **unmeasurable until the dump exists**, and it is at most 539,678 / ~4×10⁷ ≈ **1.3 %** of the new corpus.
3. **Correctness outranks salvage.** Even for rows that would match, preserving a bare-MID document *in order
   to keep an embedding* would be optimising the wrong thing. The instruction stands: do **not** keep the
   bare-MID text for embedding-reuse reasons.

**Planned honest accounting when/if the corpus is built:** run `reuse_map_kb.py` with `webqsp` added to
`FAMILIES` (family `PHASE_C_ROG_UNION`, id parser stripping the `webqsp_` prefix), joined on the MID, and report
`DENSE_ROWS_REUSED` / `SPLADE_ROWS_REUSED` / `NEW_*_ROWS_REQUIRED` exactly as for 2wiki. The expectation to be
tested, not assumed, is that reuse ≈ 0–1.3 %.

### 5.3 The MID-resolution finding (relevant to the side experiment only)

Measured on the on-disk RoG graph:

| Quantity | Value |
|---|---|
| Distinct MID-only nodes in the RoG graph | 776,660 |
| …with ≥1 **named** (non-MID) 1-hop neighbour → resolvable locally | **775,346 (99.83 %)** |
| …with only MID neighbours → not 1-hop resolvable | 1,314 (0.17 %) |
| MID→name pairs extractable from official `WebQSP.{train,test}.json` | 39,882 |
| …of the 776,788 bare-MID docs they cover | **31 (0.00 %)** — they describe topic/answer entities, already stored by name |

`src/pipeline/loader_webqsp.py:52-60,71` already implements this neighbour deduction (commit `ba6bd71`) and
writes `data/processed/master_nodes_webqsp.json`; it never touched the canonical corpus.

**This does not unblock webqsp.** A deduced name is (a) derived from the query-conditioned subgraph union, so
it cannot make that corpus question-independent, and (b) a *guess* from graph context rather than the entity's
own Freebase label. It is recorded here because it is the correct thing to do **inside** the clearly-labelled
`WEBQSP_ROG_UNION` side experiment, if that experiment is ever run.

---

## 6. Expected cost

Storage per row is measured from the existing stores in this repo (fp16 1536-dim dense ≈ 3,090 B/row; SPLADE
varies with text length: 266 B/row on the current MID-heavy webqsp text, 749–804 B/row on real prose).

| Scope | N (to be derived) | Dense (3.09 KB/row) | SPLADE (≈0.8 KB/row, prose-like text) | Corpus text on disk |
|---|---|---|---|---|
| RoG union (rejected) | 1,316,466 | 4.1 GB *(already spent, unusable)* | 0.35 GB *(already spent, unusable)* | 0.56 GB |
| **S1** English-named topics | ~4 × 10⁷ *(derive)* | **~124 GB** | **~32 GB** | ~5–8 GB (name+alias) |
| **S2** all topics | ~5.8 × 10⁷ *(derive)* | **~179 GB** | **~46 GB** | ~7–11 GB |
| S1 + description lane | ~4 × 10⁷ | ~124 GB | ~40+ GB | ~40–60 GB |

Plus the raw dump itself: ~22 GB compressed, ~250–425 GB if fully decompressed (a streaming parse avoids the
decompressed copy — the same external-sort/streaming discipline that built 2wiki at 1,021 MB peak RSS).

**Disk on this machine: ~124 GB free. S1 dense alone does not fit.** Any webqsp build at full Freebase scale
needs either external storage or the Modal pool. This is a hard blocker independent of the source question and
must be resolved before approval.

**Encode time.** No wall-clock throughput is recorded anywhere in this repo's encoding manifests (the status
shards carry `rows`/`ok` but no timing), so an honest estimate can only be given as a multiple of jobs that
were already paid for: S1 is **≈ 6.7 ×** the 2wiki_universe job (5,989,847 rows) and **≈ 7.6 ×** the hotpotqa
job (5,233,329 rows), for dense **and** SPLADE. Nothing here was encoded: **no encoder was run in this task.**

---

## 7. What would have to be true to lift the block

1. A Freebase dump is on disk, with its **sha256 and byte count recorded** (measured, not quoted).
2. A scope rule from §3 (S1 / S2 / S3) is chosen by the user and written into `SOURCE_CONTRACT.json`.
3. A textualization from §4 is chosen (with or without the description lane) and written into the contract.
4. Storage for ~124–179 GB of dense vectors is arranged.
5. `_APPROVALS.json` → `approvals.webqsp.status` is changed from `BLOCKED_PENDING_FREEBASE_SOURCE` to an
   `APPROVED*` value **by the user**, and `NEVER_BUILD` is removed from `build_kb.py` **in a separate task**.
6. Only then: build, and **measure** gold coverage; report every missing answer entity rather than injecting it.

Until all six hold, `FULL_CANONICAL_V1_STATUS` stays `READY_5_OF_6_SOURCE_CONTRACTS` and
`WEBQSP_STATUS` stays `WEBQSP_BLOCKED_ON_FREEBASE_SOURCE`. **`LOCKED_6_OF_6` must not be written.**

---

## 8. Disposition of `WEBQSP_ROG_UNION`

It may be retained as a **separate, benchmark-specific experiment** — the corpus that WebQSP KGQA baselines
actually use — provided it is always labelled as a per-question subgraph union and **never** presented as the
webqsp member of the six-dataset full-corpus result. Its artifacts (`data/canonical/webqsp/*`, 1,316,466 rows
with complete dense + SPLADE stores) are untouched by this task.

*Written by Track B canonical_v1 finalisation. Nothing in this document has been built, downloaded or encoded.*
