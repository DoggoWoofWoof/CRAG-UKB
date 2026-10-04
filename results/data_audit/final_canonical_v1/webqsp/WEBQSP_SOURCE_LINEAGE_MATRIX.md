# WebQSP — source lineage matrix and acquisition plan

**Status: REPORT ONLY. Nothing was built, encoded, downloaded (beyond READMEs / file indexes /
dataset cards / checksum files, ~200 KB total), or promoted.** No encoder, no kNN, no H4, no SAFE,
no halo, no SP1, no graph build. `build_kb.py` and `test_build_kb.py` were not opened or run.
Writes confined to `data/final_canonical/webqsp/` and `results/data_audit/final_canonical_v1/webqsp/`.
Nothing deleted.

Generated 2026-09-05. Builds on `WEBQSP_FREEBASE_PIPELINE_AUDIT.md` (RoG/NSM code audit, local
filesystem scan) and `FREEBASE_SOURCE_PROPOSAL.md`. Established facts from those documents are
**not re-derived** here.

**Evidence discipline.** Every row is tagged:

| tag | meaning |
|---|---|
| **[V]** | **VERIFIED** in this session — an HTTP response I received, a file listing I retrieved, a page of source code or PDF I read. |
| **[P]** | **PUBLISHER-STATED** — a claim made by the artifact's own README / dataset card / paper. Quoted, not measured. |
| **[I]** | **INFERRED** — my reasoning from [V]/[P] evidence. Not a measurement. |
| **[U]** | **UNKNOWN / UNVERIFIABLE without acquiring the artifact.** |

---

## 0. Headline

| target | graph source status | one-line reason |
|---|---|---|
| **`WEBQSP_ROG_MID`** | **STILL UNRESOLVED — but a ~1.0 GB falsifiable candidate was newly identified** | NSM's released `webqsp.tgz` + `CWQ.tgz` are MID-keyed per-question subgraphs produced by the exact recipe RoG says it followed. They are a *candidate preimage* of the RoG graph, not a proven one. |
| **`WEBQSP_KBQA_24M`** | **STILL UNRESOLVED — and, on present evidence, unresolvable by download** | The 24.9M/164.6M artifact is **PullNet's** intermediate KB. PullNet released no code and no data. The nearest relative, GraftNet's `freebase_prepro.tgz`, sits on a host that no longer answers TCP and was never archived. No clean file of it exists anywhere I could find. |

**Two framing corrections this audit forces:**

1. **The 24.9M/164.6M figures are PullNet's, not GraftNet's.** Verified verbatim from the PullNet
   PDF (§4.1). The GraftNet paper does **not** contain them — I read the relevant pages.
2. **`WEBQSP_KBQA_24M` is question-seeded at the corpus level.** PullNet defines it as *"all facts
   that are within 2-hops of any entity mentioned in the questions of WebQuestionsSP."* It is a
   filtered *and benchmark-conditioned* Freebase substrate. It is emphatically not "full Freebase",
   and it is also **not query-independent** in the sense `FREEBASE_SOURCE_PROPOSAL.md` §3 requires
   (it fails that document's S4 test). Acquiring it would close a *paper-comparability* target, not
   the corpus-independence block.

---

## 1. The lineage, as actually distributed

Priority order requested: GraftNet → NSM → ReaRev → GNN-RAG → RoG. PullNet is inserted because the
numeric target originates there.

| # | work | repo / release | what it ACTUALLY distributes | is it a whole-KB snapshot? |
|---|---|---|---|---|
| 1 | **GraftNet** (Sun et al., EMNLP 2018) | `haitian-sun/GraftNet` **[V]** | `freebase_prepro.tgz` (2.8 G) = `relations`, `all_entities`, `stagg.neighborhoods/<questionId>.nxhd` **[V, from its own README]** | **NO** — per-question 2-hop neighborhoods, further pruned to relations appearing in ≥1 question's semantic parse |
| 2 | **PullNet** (Sun et al., EMNLP-IJCNLP 2019) | **none exists** **[V]** | nothing | **N/A** — the 24.9M/164.6M KB is described in prose only |
| 3 | **NSM** (He et al., WSDM 2021) | `RichardHGL/WSDM2021_NSM` **[V]** | Google Drive: `webqsp.tgz` 136 M, `CWQ.tgz` 872 M, `metaqa-{1,2,3}hop.tgz` **[V]** | **NO** — per-question subgraphs (`*_simple.json` + `entities.txt` + `relations.txt`) |
| 4 | **ReaRev** (Mavromatis & Karypis, EMNLP-F 2022) | `cmavro/ReaRev_KGQA` **[V]** | **nothing of its own** — README points at NSM's Drive folder, same folder id `1qRXeuoL-…` **[V]** | **NO** — pure consumer |
| 5 | **GNN-RAG** (Mavromatis & Karypis) | `cmavro/GNN-RAG` **[V]** | Google Drive: `data.zip` 1.5 G, **`entities_names.json` 23,377,816 B**, `pretrained_lms.zip` **[V]** | **NO** — but `entities_names.json` is a real MID→name metadata file |
| 6 | **RoG** (Luo et al., ICLR 2024) | `RManLuo/reasoning-on-graphs` @ `ccf8ec84` | HF `rmanluo/RoG-webqsp` (5 shards, 518,205,557 B) + `rmanluo/RoG-cwq` (24 shards, 3,504,045,643 B) **[V]** | **NO** — per-question subgraphs, **surfaces substituted for MIDs** |

### 1.1 Verbatim evidence for the chain

**RoG → NSM** (`README.md` on branch `master`, retrieved this session) **[V]**:

> We extract the subgraphs from the Freebase following previous studies. The code can be found
> [here](https://github.com/RichardHGL/WSDM2021_NSM/tree/main/preprocessing/Freebase).

Note the verb: *"We extract"*. RoG **re-ran** the recipe; it did not state that it consumed NSM's
released tarballs. This matters in §3.

**NSM → GraftNet** (`README.md`, retrieved this session) **[V]**:

> We follow [GraftNet](https://github.com/OceanskySun/GraftNet) to preprocess the datasets and
> construct question-specific graph.

(`OceanskySun/GraftNet` resolves to `haitian-sun/GraftNet` — GitHub API returns
`full_name: haitian-sun/GraftNet`, `fork: false`, `parent: null`. Same repository, renamed account.
It is **not** a second source. **[V]**)

**ReaRev → NSM** (`README.md`) **[V]**:

> We use the pre-processed data from: https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv

Identical folder id to NSM's. ReaRev contributes **no** new graph artifact.

**PullNet — the origin of the 24.9M/164.6M target** (arXiv 1904.09537, §4.1 *Datasets*, read from
the PDF this session) **[V]**:

> We use Freebase as our knowledge base but for ease of experimentation restrict it to a subset of
> Freebase which contains all facts that are within 2-hops of any entity mentioned in the questions
> of WebQuestionsSP. This smaller KB contains 164.6 million facts and 24.9 million entities.

PullNet also states it uses *"the same KB and corpus as used for WebQuestionsSP"* for Complex
WebQuestions 1.1 **[V]**.

**The GraftNet paper does NOT state 24.9M/164.6M.** I read pages 5–7 of `aclanthology.org/D18-1455.pdf`.
What it does say about WebQSP **[V]**: 4,737 questions; entity links from S-MART; *"retrieve 500
entities from the neighbourhood around the question seeds in Freebase to populate the question
subgraphs"*; answer recall in the constructed subgraphs 94.0 %; aggregate retrieved-subgraph
statistics 528,617 entity nodes / 513 edge types / 235,567 document nodes. **A previously plausible
attribution of 24.9M/164.6M to GraftNet is therefore refuted.**

---

## 2. TARGET 1 — `WEBQSP_ROG_MID`

*Benchmark-conditioned RoG-scale graph, MIDs preserved. Target ~2.5M nodes / 8,309,195 triples /
7,058 relations.*

### 2.1 The three candidate routes, and what each actually is

| route | artifact | closes the target? |
|---|---|---|
| **A** | NSM released per-question subgraphs (`webqsp.tgz` + `CWQ.tgz`), unioned | **CANDIDATE — unproven.** Cheapest falsifiable test. |
| **B** | Re-run NSM `preprocessing/Freebase` Steps 0–3 against a Freebase backend | **Would close it structurally**, but not byte-identically (unreleased RoG seed list + post-step). Costly. |
| **C** | `camazlucas/Freebase-WebQSP-CWQ-Subgraph` | **NO — refuted.** Downstream *of* RoG. |

**Route C is closed, restated with new evidence.** Its own dataset card **[V]**:

> The subgraphs were reconstructed from the serialized graph structures provided in the preprocessing
> pipeline of the RoG (Reasoning on Graphs) framework

It is built **from** the released RoG surfaces, so the MID collapse has already happened upstream of
it. It cannot be a MID preimage of anything. (Its published stats — Unified 8,258,891 tri /
2,574,900 ent / 7,042 rel **[V]** — also remain 50,304 triples and 16 relations short, matching the
prior audit.) Diagnostic only, as the acceptance gate already says.

### 2.2 Route A — the new lead, stated honestly

**What it is.** NSM distributes, on the Drive folder ReaRev and GNN-RAG both reuse, the
**question-specific subgraphs that NSM's own `preprocessing/Freebase` recipe produced** — the same
recipe RoG says it followed. Prior audit already established that
`preprocess_step1.py:get_subgraph()` serialises `{"kb_id": head, "text": head}`, i.e. **MIDs in both
fields**. If the union of those released subgraphs reproduces RoG's published counts, then the
MID-level preimage of the RoG graph is obtainable for ~1.0 GB instead of ~52–64 GB.

**Verified facts supporting the lead:**

* Drive folder `1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv` contains exactly `webqsp.tgz`, `CWQ.tgz`,
  `metaqa-1hop.tgz`, `metaqa-2hop.tgz`, `metaqa-3hop.tgz` **[V]**.
* `webqsp.tgz` = **136 M**, `CWQ.tgz` = **872 M** — read from Google's own download interstitial;
  **no payload was fetched** **[V]**.
* NSM per-dataset layout is `*_simple.json` + `entities.txt` + `relations.txt`, where entities and
  relations are mapped to global ids in those two files **[V, NSM README]**.
* GNN-RAG ships a separate `entities_names.json` "as GNNs use the dense graphs" **[V]** — a name
  lookup would be pointless if the graph already carried names. **[I]** the NSM/GNN-RAG graph is
  MID-keyed.

**Verified facts that must stop this from being called resolved:**

* RoG's own wording is *"We extract the subgraphs"* — a re-run, not a re-use **[V]**.
* **Split counts differ.** RoG-webqsp: train 2,826 / validation 246 / test 1,628 = **4,700**
  examples **[V, HF dataset card]**. NSM WebQSP: 2,848 / 250 / 1,639 = **4,737** **[V, NSM README]**.
  **RoG has 37 fewer questions.** A union over a different question set is a different union.
* NSM's `entities.txt` content is **[U]** — I have not opened `webqsp.tgz` and will not, per the
  no-download constraint. The MID claim rests on the code audit plus the `entities_names.json`
  inference, not on the file.
* Prior audit's §1.4 caveats stand: RoG's WebQSP seed population is unpublished, PPR is per-question
  with a 2,000-entity cap, and the RoG name-projection post-step is unrecoverable.

**The falsifiable test, in full (this is the deliverable, not an instruction to run it now):**

```
acquire  webqsp.tgz (136 M) + CWQ.tgz (872 M)          total ~1.008 GB
union    all triples across all questions, both datasets, MIDs as-is
measure  |E| (triples), |R| (distinct relations), |V| (distinct endpoints)
compare  against  8,309,195 triples / 7,058 relations / ~2,566,291 entities   (strong: E and R)
         and against the measured released union 2,592,894 endpoints / 7,058 / 8,309,195
```

Outcome semantics, fixed in advance so the result cannot be tuned:

* **E and R both match exactly** → `WEBQSP_ROG_MID` is **RESOLVED**; the artifact is the MID
  preimage; the RoG surfaces become a *derived display field*, and the 37-question delta must still
  be reported.
* **E/R close but not exact** → **NOT resolved**. It is a *sibling* extraction, and must be named
  `WEBQSP_NSM_MID` — never presented as the RoG graph. Report the delta.
* **E/R far off** → route A refuted; only route B remains.

### 2.3 Matrix — `WEBQSP_ROG_MID`

| field | Route A (NSM released subgraphs) | Route B (re-extract from Freebase) |
|---|---|---|
| **RAW SOURCE** | Freebase, via NSM's `fb_en.txt` run — the dump behind it is the 2015-08-09 final RDF export **[P]** | Freebase 2015-08-09 final RDF export, as `fb_en.txt` **[P]** |
| **DOWNLOADABLE ARTIFACT** | Google Drive folder `1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv`; files `webqsp.tgz` (id `1KcIVAi4nf2uyflMOz5OSr54FOL2s2tAi`), `CWQ.tgz` (id `1ua7h88kJ6dECih6uumLeOIV9a3QNdP-g`) **[V]** | `https://download.microsoft.com/download/A/E/4/AE428B7A-9EF9-446C-85CF-D8ED0C9B1F26/FastRDFStore-data.zip` **[V]** |
| **FILE SIZE / HASH** | 136 M and 872 M, from Drive's interstitial; **no hash published** **[V size, U hash]** | 68,549,649,154 B (63.84 GiB), `Last-Modified: Tue, 16 Oct 2018 08:46:14 GMT`, HTTP 200; **no published checksum** **[V]** |
| **MID PRESERVATION** | **[I] YES** — code audit (`kb_id == text == MID`) + the existence of GNN-RAG's separate `entities_names.json`. **Not opened; [U] at file level.** | **YES by construction** — `fb_en.txt` is MID-keyed **[P]** |
| **CVT SOURCE** | inherited: NSM ran `util/deal_cvt.py` against `cvtnodes.bin` **[V, code]**; the bin itself is not in the tarball **[I]** | **`cvtnodes.bin`, authoritative**, inside `FastRDFStore-data.zip`; **not separately distributed** **[V, prior audit]**. Alternative: `freebase.type_hints.mediator` via Freebase-Setup Virtuoso **[V, its README]** |
| **RELATION FILTERING** | NSM Step 0 `manual_filter_rel.py` — 14 **substring** drops (`music.release`, `authority.musicbrainz`, `22-rdf-syntax-ns#type`, `book.isbn`, `common.licensed_object`, `tv.tv_series_episode`, `type.namespace`, `type.content`, `type.permission`, `type.object.key`, `type.object.permission`, `type.type.instance`, `topic_equivalent_webpage`, `dataworld.freeq`) **[V, prior audit]** | identical |
| **REVERSE-EDGE POLICY** | **Extraction is directed** — `get_2hop_subgraph.py` matches subject side only (`if spline[0] in seed_set`). PPR symmetrises into a row-normalised matrix for *scoring only*; the retained triple set stays directed **[V, prior audit]** | identical |
| **EXTRACTION / FILTERING CODE** | `RichardHGL/WSDM2021_NSM/preprocessing/Freebase` @ `2e20915` — **released** **[V]**. RoG's own extraction driver and its MID→name projection: **NOT RELEASED** **[V, prior audit]** | same |
| **TOPIC SEED SOURCE** | NSM `preprocess_step0.py` → `find_entity()` scans `ns:`-prefixed tokens in each question's **SPARQL parse**, keeps `m.`/`g.` **[V, prior audit]**. The WebQSP analogue of that script is **not in the repo** (it is written for CWQ) **[V, prior audit]** | same, plus the unpublished RoG seed list |
| **HOP / EXTRACTION POLICY** | 2 directed hops + an explicit extra hop through CVT objects; then per-question PPR `restart_prob=0.8`, `max_iter=20`, `max_ent=2000`, re-induced with `cvt_add_flag=False` **[V, prior audit]** | identical |
| **EXPECTED N / E / R** | **[U] — this is exactly what the test measures.** Target: 8,309,195 tri / 7,058 rel / ~2,566,291 ent | not byte-reproducible; report against **both** the paper (2,566,291 / 7,058 / 8,309,195) and the measured released union (2,592,894 / 7,058 / 8,309,195) |
| **HUMAN-READABLE METADATA** | **GNN-RAG `entities_names.json`, 23,377,816 B** (Drive id `1H7BqbVsQXr0bGuBsLUphb_Cb0Jf0rSoN`), verified by `Content-Range: bytes 0-0/23377816` **[V]**. Scope/coverage **[U]**. Plus `idirlab/freebases` metadata (§4) | same |

---

## 3. TARGET 2 — `WEBQSP_KBQA_24M`

*GraftNet/NSM/ReaRev filtered Freebase backend. Target ~24.9M entities / ~164.6M facts.*

### 3.1 The central finding: nobody distributes it

The 24.9M/164.6M object is **PullNet's intermediate whole-KB snapshot**. Tracing what each repo in
the lineage distributes:

| repo | distributes the 24.9M/164.6M snapshot? | what it distributes instead |
|---|---|---|
| PullNet | **no repo exists** **[V]** | — |
| GraftNet | **NO** **[V]** | `freebase_prepro.tgz` — per-question 2-hop `.nxhd` neighborhoods, relation-pruned to semantic-parse relations |
| NSM | **NO** **[V]** | per-question subgraphs; requires you to fetch `FastRDFStore-data.zip` yourself |
| ReaRev | **NO** **[V]** | nothing (consumes NSM) |
| GNN-RAG | **NO** **[V]** | per-question dense subgraphs + `entities_names.json` |
| RoG | **NO** **[V]** | per-question subgraphs with surfaces |

**PullNet has no code release.** GitHub repository search: `pullnet kbqa` → `total_count: 0`;
`PullNet` → 5 results, all unrelated (`joshuardavis/pullnet`, `JedSonpack/pullNet`,
`sanjay156/pullnet.java`, `Brian-Egan/pullNetworkSchedules`,
`Nandinishyam/Picknet-Pullnet-for-Entangled-Objects_CPU-`), all 0 stars **[V]**. This is a
search-based negative, so **[I]** rather than proof of non-existence — but combined with the
paper's own footnote pointing readers at *GraftNet's* repo for the baseline implementation, the
conclusion is firm.

**This is the reproducibility gap the brief asked me to characterise, and it is total for this
target: the artifact is described in one sentence of prose and was never shipped in any form.**

### 3.2 The nearest relative, and its host is gone

GraftNet's `freebase_prepro.tgz` is the closest released thing to a KBQA whole-KB snapshot — and it
is still not the 24.9M object (it is relation-pruned by semantic parses and organised per question).
Its status:

* **Published URL** (from `preprocessing/run_pipeline`, read verbatim **[V]**):
  `http://curtis.ml.cmu.edu/datasets/graftnet/freebase_prepro.tgz`
* **Host reachability [V]:** DNS resolves (`curtis.ml.cmu.edu → 128.2.204.193`), but
  `Connection timed out after 20010 milliseconds` on TCP :80. **Unreachable from this machine.**
  The Wayback CDX shows successful captures of *sibling* files as recently as **2025-09-22**, so
  **[I]** the host has gone down (or begun refusing) within roughly the last year — I cannot
  distinguish a dead host from a network block from here.
* **Archived directory index, 2019-04-05 [V]** — sizes are the server's own listing:

  | file | last modified | size |
  |---|---|---|
  | `data_webqsp.zip` | 2018-08-22 14:43 | 2.4G |
  | `data_wikimovie.zip` | 2018-08-22 14:43 | 370M |
  | **`freebase_prepro.tgz`** | **2018-11-29 10:28** | **2.8G** |
  | `model_webqsp.zip` | 2018-08-22 14:44 | 1.2G |
  | `model_wikimovie.zip` | 2018-08-22 14:44 | 54M |

* **Payload archived? NO.** Wayback CDX for `…/graftnet/freebase_prepro.tgz` and
  `…/graftnet/data_webqsp.zip`: **zero rows** **[V]**. The availability API returns
  `{"archived_snapshots": {}}` **[V]**. Only `data_wikimovie.zip` (388,180,105 B) and
  `model_wikimovie.zip` (56,966,180 B) were ever captured — the two files that are *not* about
  Freebase. **The Freebase side of GraftNet's release is not recoverable from the Internet Archive.**
* **A second, previously unnoticed CMU artifact [V]:** the archived index of
  `http://curtis.ml.cmu.edu/datasets/freebase/` lists exactly one file —
  **`upto3.nt.tgz`, 2018-04-30 19:08, 1.0G**. Name implies an N-Triples Freebase subset "up to 3
  hops". **[I]** at ~10:1 gzip on short MID-only N-Triples lines (~60 B/line), 1.0 G compressed is
  the right order of magnitude for ~10^8 triples — i.e. plausibly the whole-KB snapshot behind the
  GraftNet/PullNet line. **This is an inference from a filename and a size, nothing more.** Its
  payload was also **never archived** **[V]**, and the host is unreachable, so it cannot currently
  be tested.
* **No mirror found.** HuggingFace dataset search for `graftnet`, `freebase webqsp`,
  `webqsp-subgraph`, `cwq-subgraph`, `NSM webqsp`, `ReaRev`, `GNN-RAG`, `fb_en` returned no mirror
  of `freebase_prepro.tgz` or `upto3.nt.tgz` **[V]**.

### 3.3 Matrix — `WEBQSP_KBQA_24M`

| field | value |
|---|---|
| **RAW SOURCE** | Freebase final RDF export, 2015-08-09 **[P]**, reached through `fb_en.txt` (NSM) or an unspecified Google-internal load (PullNet) |
| **DOWNLOADABLE ARTIFACT** | **NONE EXISTS.** Not distributed by PullNet (no repo **[V]**), NSM, ReaRev, GNN-RAG or RoG **[V]**. Nearest relative `freebase_prepro.tgz` is host-dead and unarchived **[V]** |
| **FILE SIZE / HASH** | **[U]** — no file, therefore no size and no hash. `freebase_prepro.tgz` was 2.8G per the archived index **[V]**, hash never published **[U]** |
| **MID PRESERVATION** | **YES if rebuilt** — GraftNet `.nxhd` triples and NSM's `fb_en.txt` path are both MID-keyed **[I from code]** |
| **CVT SOURCE** | **GraftNet: none.** Its filter drops the whole `type` and `common` domains, so no CVT flag is consulted **[V, `step4_extract_subgraphs.py`]**. **NSM: `cvtnodes.bin`**, authoritative, only inside `FastRDFStore-data.zip` **[V]**. **Third option: `freebase.type_hints.mediator`** — documented as *"true if a type is a CVT"* in `dki-lab/Freebase-Setup`'s README **[V]**, queryable over its Virtuoso image, and **not gated behind the 63.84 GiB zip**. **Fourth: `idirlab/freebases` ships CVT-included and CVT-excluded variants**, so its CVT partition is recoverable by set difference **[V]** |
| **RELATION FILTERING** | **Two different rules, do not conflate.** **GraftNet** `_filter_relation()`: drop `<fb:common.topic.notable_types>`, and drop any relation whose leading domain segment is `type` or `common` **[V, source]**; plus a corpus-level prune to relations appearing in ≥1 question's semantic parse **[V, its README]**. **NSM** `manual_filter_rel.py`: the 14-substring blacklist in §2.3 **[V, prior audit]**. PullNet's own rule is **[U]** — the paper states only the 2-hop condition |
| **REVERSE-EDGE POLICY** | **GraftNet**: extraction directed; PPR adjacency explicitly symmetrised — `all_row_ones.append(entity_map[e2]); all_col_ones.append(entity_map[e1])` **[V, source]** — i.e. reverse edges exist for *scoring*, not in the stored triple set. **NSM**: same shape **[V, prior audit]**. **PullNet**: **[U]** |
| **EXTRACTION / FILTERING CODE** | **GraftNet**: `haitian-sun/GraftNet/preprocessing/` — `run_pipeline`, `step0_preprocess_webqsp.py`, `step1_process_entity_links.py`, `step2_relation_embeddings.py`, `step3_question_embeddings.py`, `step4_extract_subgraphs.py` (11,437 B) — **released** **[V]**, but the step that *built* `freebase_prepro.tgz` from Freebase is **NOT among them** — the pipeline downloads it pre-made **[V]**. **PullNet**: **NOT RELEASED** **[V]** |
| **TOPIC SEED SOURCE** | **GraftNet/PullNet**: S-MART / STAGG entity links — `scottyih/STAGG/webquestions.examples.{train,test}.e2e.top10.filter.tsv` **[V, `run_pipeline`]**. **NSM**: MIDs scanned out of each question's SPARQL parse **[V, prior audit]**. *These are different seed sets and will not produce the same graph.* |
| **HOP / EXTRACTION POLICY** | **The 24.9M/164.6M snapshot**: 2 hops from any WebQSP question entity, union over all questions **[V, PullNet §4.1]**. **GraftNet per-question**: 2-hop `.nxhd` → PPR `RESTART=0.8`, `MAX_ITER=20`, convergence `δ < 1e-5`, `MAX_ENT=500`, `MAX_FACTS=5000000`, `MAX_SEEDS=1`, zero-score prune at `1e-6` **[V, source]**. **PullNet's own retrieval baseline**: PageRank-Nibble, `ε = 1e-6`, top-*m* **[V, paper]** |
| **EXPECTED N / E / R** | **N ≈ 24.9 M entities, E ≈ 164.6 M facts [V, PullNet §4.1]. R: never published — [U].** Downstream checkpoints for a rebuild: GraftNet's WebQSP subgraphs aggregate to 528,617 entity nodes / 513 edge types **[V, GraftNet paper]**; PullNet WebQSP retrieval recall 0.927 at 1,876.9 avg entities **[V, PullNet Table 4]**; NSM WebQSP avg 1,429.8 entities, 94.9 % coverage **[V, NSM README]** |
| **HUMAN-READABLE METADATA** | Not in any KBQA release. Available only from a Freebase backend (`type.object.name`) or from `idirlab/freebases`' `object_names` / `entities_id_label` **[V]**, or GNN-RAG's `entities_names.json` (scoped to the KBQA graph, coverage **[U]**) |

---

## 4. Candidate backends — what exists, what it costs, what it would prove

**Never an automatic equivalent.** Each row states what would have to be *proven* for it to qualify.

| candidate | artifact | size / hash | what it IS | qualifies as… |
|---|---|---|---|---|
| **`microsoft/FastRDFStore` data** | `FastRDFStore-data.zip` | **68,549,649,154 B (63.84 GiB)**, HTTP 200, `Last-Modified 2018-10-16`; no published checksum **[V]** | the NSM-cited source; contains `fb_en.txt` **and** `cvtnodes.bin` **[P]** | **the only backend that closes the CVT question authoritatively.** Infeasible here: 63.84 GiB vs 112.9 GiB free, before two full-size derived files **[V, prior audit]** |
| **`dki-lab/Freebase-Setup`** | `virtuoso_db.zip` (Dropbox) | **56,324,228,100 B (52.46 GiB)**, HTTP 200 **[V]** | full 2015 dump, N-Triples literal-format fixed, preloaded into Virtuoso **[P]**. Its README documents `freebase.type_hints.mediator`, `type.object.name`, `common.topic.alias`, `type.object.type`, `type.property.reverse_property` **[V]** | **best backend for a rebuild if disk is arranged.** Gives CVT flags *and* names *and* the reverse-property twin map from one artifact. Needs a deterministic dump-out step and ≥100 GB RAM recommended **[P]**. **Not** the KBQA-24M graph — it is unfiltered |
| **`alex999819/freebase` (HF)** | `freebase-rdf-latest.gz` | **32,191,685,487 B**, **SHA-256 `b3d21c80dc512c8ed3c37279516340ef7a77027d593932c7c3fb22d40dfd47f4`**, published in a `SHA256SUMS` file **[V]** | community preservation mirror of the 2015-08-09 final dump. Card claims byte-identity with `freebase-rdf-2015-08-09-00-01.gz` *extracted from `FastRDFStore-data.zip`* **[P]** | **the cheapest hashed raw dump — half the size of the alternatives.** But it is the **raw RDF only**: no `cvtnodes.bin`, no `fb_en.txt`. The byte-identity claim is the publisher's, unverified **[P]** |
| **`CleverThis/freebase` (HF)** | 964 parquet shards | **32,476,443,230 B** total **[V]** | same dump, parquet-ised | equivalent to the above; no hash advantage; prior audit already flagged the ~300 GB extracted footprint |
| **`Sheppp/freebase` (HF)** | 75 files: `store/{mid,name,rel}.{idx,str}`, `edges.u32`, … | **59,674,173,045 B** total **[V]** | a columnar MID / name / relation edge store | **provenance UNVERIFIED** — its README and `store/generic.txt` were unreadable (HF rate limit, `maximum queue size reached`). Interesting shape, but nothing about it is established. Do not use until its build is documented |
| **`idirlab/freebases`** | Zenodo `10.5281/zenodo.7909511` → `idirlab-freebases.zip` | **14,148,416,296 B (13.18 GiB)**, `md5:170689b7aad9f029566a4deb36605b01` **[V]** | independently reprocessed Freebase for **link prediction**, four variants, plus rich metadata files | **METADATA source — see §4.1.** *Not* a graph substitute |
| **`camazlucas/…-Subgraph`** | `webqsp_subgraph.tsv` 260,208,520 B; `cwq_subgraph.tsv` 471,465,645 B; `unified_graph.tsv` 542,254,271 B **[V]** | reconstruction **from** RoG's released graph **[V, card]** | **REFUTED for both targets.** Downstream of the MID collapse. Diagnostic only |
| GraftNet CMU host | `freebase_prepro.tgz` 2.8G, `upto3.nt.tgz` 1.0G | sizes from archived index **[V]**; no hash **[U]** | the actual KBQA intermediates | **unreachable and unarchived [V]** — see §3.2 |

### 4.1 `idirlab/freebases` — metadata candidate, and the naming trap

**Verified variant statistics** (from the repository README's own table):

| variant | CVT nodes | reverse triples | #entities | #properties | #triples |
|---|---|---|---|---:|---:|
| FB-CVT-REV | removed | removed | 46,069,321 | 3,055 | 125,124,274 |
| FB-CVT+REV | removed | retained | 46,077,533 | 5,028 | 238,981,274 |
| FB+CVT-REV | retained | removed | 59,894,890 | 2,641 | 134,213,735 |
| FB+CVT+REV | retained | retained | 59,896,902 | 4,425 | 244,112,599 |

**This is the trap the brief names, now with numbers attached.** No variant is close to
24.9 M / 164.6 M. The nearest on triples (FB+CVT-REV, 134.2 M) is **2.4× off on entities**
(59.9 M vs 24.9 M). **A graph built from any of these is a different experiment and must be named
`WEBQSP_FREEBASE_60M_SCALE` (or `_46M_SCALE`), never `KBQA_24M`, and never compared head-to-head
against PullNet/GraftNet/NSM/ReaRev numbers.**

**What would have to be proven for `idirlab` to qualify as the KBQA-24M graph** — all four,
none currently established:

1. **Same underlying snapshot.** Both derive from Freebase 2015-08-09 — plausible **[I]**, but
   idirlab applied its own type-system repair and reverse-pair handling. Provable only by triple-set
   comparison against a `fb_en.txt`-derived graph.
2. **Same relation filter.** idirlab restricts to "subject matter domains"; GraftNet drops
   `type`/`common` domains and semantic-parse-absent relations; NSM applies a 14-substring
   blacklist. **These are three different rules and demonstrably yield different property counts**
   (2,641–5,028 for idirlab vs 7,058 in the RoG union). Provable only by exhibiting the relation
   sets and their difference.
3. **Same CVT treatment.** idirlab's `+CVT`/`-CVT` split is a whole-graph policy; NSM's is a
   traversal rule with an authoritative per-node lookup and a `cvt_add_flag=False` re-induction.
   Provable only against `cvtnodes.bin` or `freebase.type_hints.mediator`.
4. **Same reverse-edge policy.** idirlab's `±REV` is explicit and whole-graph; the KBQA line stores
   directed triples and symmetrises only inside PPR. Provable by checking twin-pair presence.

**As metadata it needs none of that**, because a name lookup is keyed on MIDs and is
topology-neutral. It ships `object_names` (`/type/object/name` + `@en`), `object_types`
(`/type/object/type`), `object_ids`, `entities_id_label`, `properties_id_label`, `types_id_label`,
`domains_id_label`, `freebase_endtypes`, and `uri_original2simplified` / `uri_simplified2original`
**[V, Zenodo description]**. That is precisely the Phase-4 shopping list in
`WEBQSP_FREEBASE_PIPELINE_AUDIT.md` §4.4.

---

## 5. ACQUISITION PLAN

Ranked by (a) closes a **graph** target vs only **metadata**, (b) download size, (c) certainty of
identity. **Nothing below has been executed.** Items 1–2 are cheap enough to be reversible; items
4–6 are capacity decisions, not preferences.

| # | action | size | closes | certainty | verdict |
|---|---|---:|---|---|---|
| **1** | **NSM `webqsp.tgz` + `CWQ.tgz`, union, count E/R/V** | **~1.008 GB** | **GRAPH — `WEBQSP_ROG_MID`, conditionally** | **UNPROVEN, but cheaply falsifiable** | **DO THIS FIRST.** Highest value-per-byte in the whole matrix by three orders of magnitude. Pre-commit to the §2.2 outcome semantics before measuring |
| **2** | **GNN-RAG `entities_names.json`** | **23,377,816 B** | metadata (MID→name, KBQA-scoped) | scope/coverage **[U]** | **cheap, do alongside 1.** Measure its coverage of the 62,375-node name-resolvable band and the 1,652,618 bare MIDs. Never let it name a CVT |
| **3** | **`idirlab-freebases.zip` (Zenodo), metadata files only** | 13.18 GiB (hashed, md5 verified-as-published) | **metadata only** — names, aliases-adjacent, types, relation labels | high for metadata, **zero for graph identity** | **the standing recommendation from the prior audit, unchanged.** If exactly one download is ever authorised beyond item 1, this is it. **Must not** be used as a graph |
| **4** | `dki-lab/Freebase-Setup` `virtuoso_db.zip` | **52.46 GiB** | **GRAPH (backend for a rebuild)** + CVT flags + names + reverse-twin map, from one artifact | high — but a *rebuild*, never byte-identical to RoG | **the best rebuild backend**, and 11.4 GiB smaller than FastRDFStore. Needs ≥100 GB RAM (publisher-stated) and external storage. Blocked on capacity, not on evidence |
| **5** | `alex999819/freebase` `freebase-rdf-latest.gz` | **29.98 GiB**, SHA-256 published | GRAPH (raw backend only) | dump identity **high** (hashed); **no CVT, no `fb_en.txt`** | **the cheapest hashed raw dump.** Choose over #4 only if you accept deriving CVT membership structurally (the prior audit's classifier: 96.10 % of bare MIDs, 0.045 % control FP, 95.75 % cross-signal agreement) |
| **6** | `FastRDFStore-data.zip` | **63.84 GiB** | GRAPH + **authoritative `cvtnodes.bin`** | highest fidelity to NSM/RoG | **infeasible on this machine as configured** (prior audit §2.2), and now strictly dominated by #4 for everything except `cvtnodes.bin` itself |
| **—** | `camazlucas/…-Subgraph` | 1.27 GB | nothing | refuted | **DO NOT ADOPT.** Downstream of RoG's MID collapse |
| **—** | `Sheppp/freebase` | 59.67 GB | unknown | **provenance unverified** | **DO NOT ADOPT** until its README is readable and its build documented |
| **—** | GraftNet `freebase_prepro.tgz` / `upto3.nt.tgz` | 2.8G / 1.0G | would close KBQA-24M-adjacent | — | **cannot be acquired.** Host times out; never archived. The only remaining route is to email the authors or CMU LTI for a copy — **out of scope, and a human action, not mine** |

### 5.1 What each target needs to reach RESOLVED

**`WEBQSP_ROG_MID`** — needs **either**:
* item 1 to return E = 8,309,195 **and** R = 7,058 exactly (then RESOLVED, with the 37-question
  delta reported); **or**
* item 4/5/6 plus a re-run of NSM Steps 0–3 (then RESOLVED-BY-RECONSTRUCTION, never
  byte-identical — RoG's seed list and name-projection step remain unreleased **[V, prior audit]**).

**`WEBQSP_KBQA_24M`** — needs a rebuild, full stop. There is no file to acquire. Concretely:
a Freebase backend (item 4/5/6) + the S-MART/STAGG seed files (tiny, on GitHub, verified live in
`run_pipeline` **[V]**) + a 2-hop union over all WebQSP question entities + PullNet's unpublished
relation rule, which would have to be **chosen and documented as ours**, not recovered. Even done
perfectly, the result should be reported as `WEBQSP_KBQA_24M_RECONSTRUCTED` with its measured
N/E/R printed next to 24.9 M / 164.6 M and every difference explained rather than tuned.

---

## 6. What I could not establish

Stated plainly, so none of it is mistaken for a gap in the record:

* **Whether NSM's released `webqsp.tgz` actually contains MIDs.** Inferred from the code audit and
  from GNN-RAG needing a separate name file. **Not opened.** This is the single assumption item 1
  is designed to test.
* **Whether the union of NSM's subgraphs equals RoG's graph.** Unknown, and the 37-question split
  delta is direct evidence against exact equality.
* **What `upto3.nt.tgz` contains.** A filename and a 1.0 G size from a 2019 archived directory
  listing. The gzip-ratio argument in §3.2 is arithmetic, not evidence.
* **Whether `curtis.ml.cmu.edu` is dead or blocking.** TCP timeout from this machine is all I have.
* **PullNet's relation filter and its R count.** Never published.
* **Any hash for `FastRDFStore-data.zip`, `virtuoso_db.zip`, or the CMU files.** None published.
* **`Sheppp/freebase` provenance.** HF rate-limited both its README and `store/generic.txt`.
* **`entities_names.json` coverage.** Size verified; contents not.
* **The byte-identity of `alex999819/freebase` with the FastRDFStore-embedded dump.** The
  publisher's claim, reproduced as such.

---

## 7. Constraints honoured

No encoder, kNN, H4, SAFE, halo, SP1 or graph build. `scratchpad/final_canonical_build/build_kb.py`
and `test_build_kb.py` were neither run nor opened nor edited; no corpus builder was invoked; the
shared `ExtSort` work dir was not touched. Writes confined to `data/final_canonical/webqsp/` and
`results/data_audit/final_canonical_v1/webqsp/`. Nothing written to `data/canonical/`,
`data/processed/`, `data/ukb_storage/`, `data/original/`, `results/GENERALIZATION/` or
`scratchpad/ablation_qwen/`. Nothing deleted; no artifact superseded. `data/` is gitignored;
nothing committed.

**Downloads:** READMEs, GitHub tree listings, HuggingFace file listings, dataset cards, one
`SHA256SUMS` (88 B), Zenodo record metadata, Wayback CDX/index responses, and two conference PDFs
(D18-1455, 718.6 KB; arXiv 1904.09537, 475.3 KB) fetched by the WebFetch tool for the sole purpose
of reading the KB-size statements quoted above. Total well under 5 MB against the 100 MB cap.
`idirlab/freebases` (13.18 GiB), `FastRDFStore-data.zip` (63.84 GiB), `virtuoso_db.zip`
(52.46 GiB), `freebase-rdf-latest.gz` (29.98 GiB) and every NSM/GNN-RAG Drive payload were probed
by HTTP header, interstitial page or API listing **only** — **no payload byte was fetched.**
