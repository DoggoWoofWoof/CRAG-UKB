# WebQSP — Freebase pipeline audit (MID-level reconstruction)

**Status: AUDIT ONLY. Nothing was built, encoded or promoted. No Freebase source was downloaded.**
No `data/final_canonical/webqsp/nodes.jsonl`. No encoder, no kNN, no H4.
Generated 2026-09-05. Every number below is derived on this machine from the artifacts named beside
it, or read from a live HTTP `HEAD` / API response quoted verbatim. Reproduction scripts, all under
`scratchpad/final_canonical_build/_analysis/`: `phase2_source_scan.py`,
`webqsp_mid_cvt_classification.py`, `webqsp_mid_pass2_inrelation.py`,
`webqsp_phase78_compare_reuse.py`, `webqsp_phase7_followup.py`.
Raw outputs: `scratchpad/final_canonical_build/_audit/phase*.json`.
(These are ad-hoc analysis scripts and were deliberately moved out of the builder directory so
`manifest.py`'s `builder_files_sha256` sweep keeps pinning exactly the 18 pipeline modules and is not
polluted by one-off audit code.)

**Headline: Phase 3 could not be executed. There is no benchmark Freebase source on this machine
(Phase 2: 1,996,229 files scanned, 0 hits), and the canonical one is a live 63.8 GB download that
was explicitly out of scope. Every other phase was completed, including the quantitative MID/CVT
classification the acceptance gate marked `MID_CLASSIFICATION_REQUIRED`.**

---

## PHASE 1 — the actual benchmark pipeline

### 1.1 Trace, with revisions

| step | repo | revision pinned during this audit | what it contributes |
|---|---|---|---|
| 1 | `RManLuo/reasoning-on-graphs` (RoG, ICLR 2024) | HEAD `ccf8ec847bf61005a1b27cc9e5aff5c8ead7a24b`, 2025-03-05 | consumes the published graph; **contains no extraction code** |
| 2 | `RichardHGL/WSDM2021_NSM` → `preprocessing/Freebase` | HEAD `2e20915956a69b6f832143ded9d54442ccbe698e`, 2022-03-04 | the actual extraction recipe |
| 3 | `microsoft/FastRDFStore` (data release) | repo archived, last push 2023-06-12 | `fb_en.txt` + `cvtnodes.bin` |

RoG's README defers the whole graph question in one sentence — its subgraph-extraction section says
only that subgraphs are extracted from Freebase following previous studies, and links to
`https://github.com/RichardHGL/WSDM2021_NSM/tree/main/preprocessing/Freebase`.

**Verified against the repository tree, not taken on trust.** The RoG tree at
`ccf8ec84` contains 50 blobs: `src/qa_prediction/`, `src/joint_training/`, `src/llms/`,
`src/align_kg/`, `scripts/`, `prompts/`, `config/`, `datasets/RoG_train_data.tar.tz`, and the paper
assets. There is **no** subgraph-extraction module, **no** Freebase loader, and **no** MID→name
resolver anywhere in it.

> **This is the single most important Phase-1 finding.** The released RoG graph is *not* byte-
> reproducible from released RoG code. The reproducible half of the pipeline is NSM's, and NSM's
> output is **MID-keyed**. The MID→surface projection that produced the mixed-endpoint artifact we
> hold happened in an unreleased RoG build step. Reconstructing at MID level is therefore not a
> re-derivation of the released strings — it is a *reconstruction of the object RoG started from*.

### 1.2 The NSM recipe, verified file by file

Tree of `preprocessing/Freebase` at `2e20915` (blob sha prefixes as returned by the GitHub API):

| file | bytes | blob | role |
|---|---:|---|---|
| `README.md` | 2,516 | `81f6d46d9a4c` | the 7-step recipe |
| `manual_filter_rel.py` | 996 | `2e41becae50f` | Step 0 relation filter |
| `preprocess_step0.py` | 2,872 | `e2316cb825b3` | question / answer / **topic-entity** extraction |
| `get_seed_set.py` | 435 | `9c09b0e5cf3d` | topic entities → seed set |
| `get_2hop_subgraph.py` | 3,125 | `044a6570325f` | 2-hop expansion **with an explicit CVT hop** |
| `preprocess_step1.py` | 8,491 | `3d2a33ef0f4c` | PPR reservation + CVT-aware subgraph assembly |
| `util/deal_cvt.py` | 3,808 | — | `cvtnodes.bin` binary reader, `is_cvt()` |
| `util/ppr_util.py` | 1,300 | — | personalised PageRank |
| `map_kb_id.py`, `build_vocab_from_dep.py`, `update_vocab_with_rel.py`, `load_emb_glove.py` | — | — | NSM-model-only vocab steps, irrelevant to the graph |

**Source acquisition (README, verbatim command):**

```
wget https://download.microsoft.com/download/A/E/4/AE428B7A-9EF9-446C-85CF-D8ED0C9B1F26/FastRDFStore-data.zip --no-check-certificate
```
then "unzip `FastRDFStore-data.zip` and keep `fb_en.txt` and `cvtnodes.bin` in `data/` folder."

**Step 0 — relation filtering** (`manual_filter_rel.py`). Reads `data/fb_en.txt`, drops any triple
whose relation string *contains* one of 14 substrings, writes `manual_fb_filter.txt`:

```
music.release, authority.musicbrainz, 22-rdf-syntax-ns#type, book.isbn,
common.licensed_object, tv.tv_series_episode, type.namespace, type.content,
type.permission, type.object.key, type.object.permission, type.type.instance,
topic_equivalent_webpage, dataworld.freeq
```

Note this is a **substring** test, not a prefix test, and `type.object.key` is filtered — so
`key.wikipedia.en` survives only if it is not routed through `type.object.key` in `fb_en.txt`.
That matters for Phase 4 and is flagged, not assumed.

**Step 1 — seeds.** `preprocess_step0.py` extracts topic entities from the **SPARQL query** of each
question (`find_entity()` scans `ns:`-prefixed tokens and keeps those matching `m.`/`g.`), then
`get_seed_set.py` unions them into a flat seed file. Seeds are MIDs, never names.

**Step 2 — 2-hop expansion with an explicit CVT hop** (`get_2hop_subgraph.py`). Two identical
passes of `fetch_triple_1hop`, hop 1 seeded by the topic entities and hop 2 seeded by every entity
appearing in hop 1. Inside each pass, when the object of a kept triple is a CVT
(`is_cvt(spline[2], cvt_nodes)`) and is not itself a seed, it is collected into `cvt_set` and the
file is streamed a second time to pull that CVT's own outgoing triples. **Only the subject side is
matched** (`if spline[0] in seed_set`), so the extraction is directed.

**Step 3 — PPR reservation** (`preprocess_step1.py`). Per question: build the 2-hop triple set,
symmetrise it into a row-normalised sparse matrix, run PPR (`restart_prob=0.8`, `max_iter=20`) from
the topic entities, keep the top `max_ent=2000` entities, then re-induce the subgraph over exactly
that node set with `cvt_add_flag=False` — a triple whose object is a CVT that did not itself survive
is **dropped**, because "single hop to cvt entity is of no sense".

**CVT identity is authoritative, not heuristic.** `util/deal_cvt.py` is a byte-level reader for
`cvtnodes.bin`, itself a port of `microsoft/FastRDFStore`'s `FastRDFStore.cs`. It yields
`{2-to-4-char MID prefix → {MID → isCVT bool}}`, and `is_cvt(subject, cvt_nodes)` is a direct lookup.
**NSM never needs a name to identify a node**, and it never assigns one.

### 1.3 What NSM actually emits — and where the names came from

`preprocess_step1.py:get_subgraph()` serialises every node and relation as

```python
readable_tuples.append([
    {"kb_id": head, "text": head},
    {"rel_id": rel, "text": rel},
    {"kb_id": tail, "text": tail},
])
```

`text` **is** `kb_id`. NSM's output carries bare MIDs in both fields.

The released `rmanluo/RoG-{webqsp,cwq}` parquet, by contrast, is 35.70 % human-readable surfaces.
Therefore the MID→name projection is a **RoG-side step that was never published**, which is exactly
consistent with the prior audit's §B.2 finding that no mapping file exists in either HF repo and
that `q_entity`/`a_entity` are themselves surfaces. Two independent lines of evidence now agree.

### 1.4 Consequence for reconstruction

Running the NSM pipeline against `fb_en.txt` + `cvtnodes.bin` would produce a MID-level graph with
the same relation filter, the same seeds, the same hop semantics and authoritative CVT flags. It
would **not** reproduce the released 8,309,195-triple union exactly, because:

* RoG's seed population for WebQSP is not published (NSM's `preprocess_step0.py` is written for CWQ
  and reads `KBQA-GST`'s CWQ files; the WebQSP analogue is not in the repo);
* PPR is per-question with a 2000-entity cap, so the union depends on the exact question list and
  split assignment RoG used;
* the unreleased RoG post-step (name projection, and whatever dedup it applied) is unrecoverable.

That is a statement about the benchmark, not about our method, and it must be reported rather than
tuned away. See "recommendation" below for how to handle it.

---

## PHASE 2 — source availability

**Scan:** `scratchpad/final_canonical_build/_analysis/phase2_source_scan.py`, single filesystem traversal,
read-only, downloads nothing. Roots `C:\Users\Swastik` (contains the repo), `C:\ProgramData`,
`C:\Program Files`, `C:\Program Files (x86)`. **195,273 directories / 1,996,229 files visited in
343.7 s.** `D:` and `E:` do not exist; the machine has a single volume with 112.9 GiB free.

| target | found | path / evidence |
|---|---|---|
| `FastRDFStore-data.zip` | **NO** | — |
| `fb_en.txt` | **NO** | — |
| `cvtnodes.bin` | **NO** | — |
| `Freebase-Setup` | **NO** | — |
| Virtuoso DB / `virtuoso.ini` | **NO** | — |
| `object_names`, `object_types` | **NO** | — |
| `id2name_parts`, `entities_id_label`, `properties_id_label` | **NO** | — |
| FACC1 mention resources | **NO** | 2 hits are false positives: npm content-addressed cache blobs whose hex digest happens to contain `facc1` (`…npm-cache\_cacache\content-v2\sha512\65\87\9268…facc11959…`, 10,124 B and 752 B) |
| `surface_map_file_freebase_complete_all_mention` | **NO** | — |
| `mid2name` / `entity2name` / `name2mid` | **NO** | — |
| NSM intermediates (`manual_fb_filter.txt`, `subgraph_hop*.txt`, `*_step0.json`) | **NO** | — |
| any `freebase*` payload | **NO** | 6 hits, all non-payload: `torch_geometric/datasets/freebase.py` (4,075 B) under Python 3.12 and 3.13 plus their `.pyc`, and this repo's own two copies of `FREEBASE_SOURCE_PROPOSAL.md` (18,130 B) |

**Confirmed: no Freebase graph source and no Freebase metadata source exists on this machine.**
This reproduces the prior scan independently.

### 2.1 Remote availability (probed by HTTP `HEAD` / API only — nothing fetched)

| resource | status | size | identity |
|---|---|---:|---|
| `FastRDFStore-data.zip` (the NSM-cited source) | **LIVE**, `HTTP/1.1 200 OK`, `Content-Type: application/octet-stream` | **68,549,649,154 B = 63.84 GiB** | Microsoft download CDN; no published checksum |
| `microsoft/FastRDFStore` (reader code for `cvtnodes.bin`) | archived, last push 2023-06-12 | small | GitHub |
| `idirlab/freebases` → Zenodo `10.5281/zenodo.7909511` | LIVE | **14,148,416,296 B = 13.18 GiB**, single `idirlab-freebases.zip` | `md5:170689b7aad9f029566a4deb36605b01` (concept DOI `10.5281/zenodo.7909510`) |
| `CleverThis/freebase` (HF mirror of `freebase-rdf-latest.gz`) | LIVE | 964 parquet shards, card states **300 GB extracted**, ~3 B triples | HF dataset card |
| `camazlucas/Freebase-WebQSP-CWQ-Subgraph` | LIVE | `unified_graph.tsv`, `webqsp_subgraph.tsv`, `cwq_subgraph.tsv` | third-party reconstruction; the gate already records it as **diagnostic only** (2,574,900 ent / 7,042 rel / 8,258,891 tri — 50,304 triples and 16 relations *short* of the published graph) |

**Nothing was downloaded.** Per the directive, Phase 2 stops at the availability report.

### 2.2 Feasibility, stated plainly

`FastRDFStore-data.zip` at 63.84 GiB against 112.9 GiB free is not merely large, it is
**infeasible as specified**: the zip alone is 57 % of free space, and `fb_en.txt` must be extracted
(Freebase English triples, order 10^8 lines) and then rewritten by `manual_filter_rel.py` into a
second full-size file before hop extraction even starts. Realistic peak footprint is well past the
remaining headroom on a single 118 GB volume, before any of the 5.99 M-node 2wiki and 5.23 M-node
hotpot corpora are touched. This is a capacity finding, not a preference.

`cvtnodes.bin` — the one artifact that would settle CVT identity authoritatively — is **not
separately distributed**. It exists only inside that 63.84 GiB zip. That is why Phase 5 below
derives CVT membership from the graph instead, and validates it two independent ways.

---

## PHASE 3 — MID-level reconstruction: **NOT EXECUTED**

**Reason: the graph source is unavailable locally (§2) and acquiring it was out of scope (§2.2).**

Nothing was fabricated and there was **no silent fallback to the released endpoint union**. The
released union remains what it has always been — a *measured description of the published artifact*
(2,592,894 endpoints / 7,058 relations / 8,309,195 triples, triples and relations matching the paper
exactly, entities `+26,603` / `+1.04 %`) — and it is **not** promoted to canonical identity here.

What Phase 3 would have produced, and the comparisons it would have been graded against, are
specified in the recommendation below so the work can be picked up unchanged.

---

## PHASES 4 & 5 — MID and CVT resolution, measured

`MID_CLASSIFICATION_REQUIRED` in the acceptance gate says the prior audit's blanket claim — that the
1,652,618 bare MIDs are mediator nodes with no Freebase name — must be **verified quantitatively,
not accepted as a statement**. It now is, without a Freebase dump.

### 4.1 The method, and why it is sound

The released RoG graph is itself a **labelled observation of Freebase name coverage**: RoG rendered
an endpoint as a name where Freebase had `type.object.name`, and left the bare MID where it did not.
So for each Freebase **type** — the relation id minus its last component, e.g. `people.marriage` for
`people.marriage.spouse` — we can *measure* what fraction of the endpoints asserting that type were
named. Two independent signals per node:

* **pass 1, outgoing:** the types the node asserts as a head (`phase45_mid_cvt_classification.json`);
* **pass 2, incoming:** the *range* of the relations pointing at it, `obj_named_rate(R)` over R's
  distinct tails (`phase45_mid_pass2_inrelation.json`).

Query-independent throughout. No question text, no answers, no gold paths, no topic annotations, no
neighbour names.

### 4.2 The split is real, not a threshold artefact

Per-type named rate over 3,119 distinct Freebase types (pass 1):

| named-rate bin | types | node-type assertions |
|---|---:|---:|
| `[0, 0.001)` | 531 | 859,604 |
| `[0.001, 0.01)` | 25 | 150,094 |
| `[0.01, 0.05)` | 9 | 2,767 |
| `[0.05, 0.5)` | 11 | 5,586 |
| `[0.5, 0.9)` | 12 | 146,912 |
| `[0.9, 1.0]` | **2,531** | **1,458,739** |

**Extreme bimodality: 556 types below 0.01 and 2,531 above 0.9 — 98.98 % of all types sit at one
extreme or the other (3,087 / 3,119 = 98.97 %), leaving just 32 types in the whole middle range.** The relation-range
histogram (pass 2) is the same shape: 1,460 relations below 0.01, 5,514 above 0.9, 84 in between.

Threshold sensitivity of `NAMELESS_MEDIATOR_MIDS` under pass 1:
`τ=0.005 → 964,780`; `τ=0.01 → 966,364`; `τ=0.02 → 967,122`; `τ=0.05 → 969,037`; `τ=0.10 → 969,160`.
**A 20× change in τ moves the answer by 0.45 %.** τ = 0.02 is used throughout.

**Control.** Of 533,758 endpoints RoG *did* name (so they demonstrably have `type.object.name`) and
which have outgoing edges, only **238 (0.045 %)** are false-positively called mediator-like.

**Domain validation.** The types the method calls mediators are the textbook Freebase CVTs:
`film.performance` (198,122 nodes, rate 0.0002), `common.webpage` (121,320, 0.0023),
`award.award_nomination` (63,096, 0.00005), `film.film_crew_gig` (56,716, 0.00002),
`film.film_regional_release_date` (50,559, 0.00002), `education.education`
(34,469), `award.award_honor` (34,368), `tv.tv_guest_role` (28,798),
`measurement_unit.dated_integer` (26,953, exactly 0), `sports.sports_team_roster` (19,581),
`music.track_contribution` (18,573, exactly 0), `film.film_cut` (17,777). The types it calls ordinary
are `common.topic` (0.9622), `people.person` (0.99997),
`film.film` (0.9982), `location.location` (0.9602), `book.written_work` (0.9990).

**Cross-validation.** Pass 2 re-labels, from incoming relations *only*, the 1,014,462 bare MIDs pass 1
could label from outgoing types only. 1,011,820 carry an in-relation signal; **agreement 95.75 %**.

### 4.3 MID/name/CVT resolution table

| population | N | % of union | % of bare MIDs |
|---|---:|---:|---:|
| **ordinary topic — named by RoG** | **925,761** | 35.70 % | — |
| **ordinary topic — bare MID, name-resolvable** | **62,375** | 2.41 % | 3.77 % |
| **CVT / mediator — nameless by construction** | **1,588,085** | 61.25 % | 96.10 % |
| mixed / undecided | 2,158 | 0.08 % | 0.13 % |
| undecidable from the graph alone | **0** | 0 % | 0 % |
| literal (numeric, date, URL, filename) | 14,515 | 0.56 % | — |
| **union total** | **2,592,894** | 100 % | — |

Gate fields, as demanded:

```
TOTAL_BARE_MIDS         1,652,618
NAMELESS_MEDIATOR_MIDS  1,588,085   (96.10 %)
NAME_RESOLVABLE_MIDS       62,375   ( 3.77 %)   -- uncertainty band [19,566 .. 62,375], see below
UNKNOWN_MIDS                    0
```

**Honest uncertainty band on `NAME_RESOLVABLE_MIDS`.** The merged figure takes pass 1 where pass 1
could decide and pass 2 elsewhere. The two signals disagree on 42,743 nodes that pass 1 calls
name-resolvable and pass 2 calls mediator. Both-signals-agree gives **19,566**; either-signal gives
**62,375**. So the population a Freebase metadata pass would actually rescue is
**0.75 %–2.41 % of union entities**. It is reported as a band, not collapsed to a point.

The leaves in this class are reached by relations that genuinely range over named topics —
`location.location.contains` (2,738 leaves, range named-rate 0.932),
`location.location.people_born_here` (2,047, 0.968), `visual_art.visual_artist.artworks` (1,268),
`music.artist.album` (874), `book.author.works_written` (810) — and they are overwhelmingly `g.`
machine ids (`g.122tpp_z`, `g.120l2rdv`, `g.11b62w3281`, …), i.e. late Freebase objects that
plausibly never had `type.object.name` at all.

**The prior audit's blanket claim is therefore CONFIRMED at 96.10 %, and its residual quantified for
the first time.**

### 4.4 Phase 4 metadata resolution: **NOT EXECUTED**

Resolving `type.object.name`, `common.topic.alias`, `type.object.type` and `key.wikipedia.en`
requires a source that is not present (§2). Nothing was invented in its place; in particular no
structural description was substituted for a real name on an ordinary topic. What *can* be stated:

| metadata target | resolvable today | why |
|---|---|---|
| display name for the 925,761 named endpoints | already present, verbatim from the released graph | RoG resolved these once |
| display name for the 62,375 name-resolvable bare MIDs | **NO** | needs a Freebase name source |
| aliases (`common.topic.alias`) for any node | **NO** | absent from every artifact on disk |
| types (`type.object.type`) for any node | **partially, structurally** | the asserted-type set is derivable from relation prefixes (§4.1); this is the *schema* type, not `type.object.type` |
| `key.wikipedia.en` | **NO** | and note NSM's Step-0 filter drops `type.object.key` |

### 4.5 PHASE 5 — CVT policy

**Recommendation: CVTs are structure-only nodes. Give them a graph identity and NO Dense/SPLADE
text. If retrieval text is required, use the deterministic schema string — never a neighbour name.**

Evidence, in order of weight:

1. **Prior KGQA code treats CVTs as structure, never as text.** NSM's `deal_cvt.py`/`is_cvt()` is a
   pure boolean membership test used only to steer traversal; `get_2hop_subgraph.py` uses it to
   decide whether to take an extra hop; `preprocess_step1.py` uses it to decide whether a dangling
   CVT triple is meaningless and should be dropped. At no point does NSM read or assign a CVT label.
   The Freebase design intent matches: a mediator exists to carry a compound value, and Google
   documents it separately from ordinary topics.
2. **They carry almost no distinguishing text.** Under the deterministic recipe
   `CVT | types: <sorted types> | relations: <sorted relations>`, the 967,122 pass-1 CVTs collapse to
   **2,155 distinct strings — a 449× compression.** The single most common string covers 129,982
   nodes (`CVT | types: film.performance | relations: film.performance.film`); the top eight cover
   ~343 k. A Dense/SPLADE vector shared by 130,000 nodes cannot discriminate between them; it is
   pure index weight. This is a measurement, not an intuition.
3. **The alternative is the documented catastrophe.** `src/pipeline/loader_webqsp.py:52-60,68`
   assigns a mediator the name of an arbitrary named neighbour. Verified in Phase 7 below:
   458,872 of 459,417 MID nodes carry another entity's name, and *every single one* of those names
   is genuinely one of that MID's named neighbours — with only 373,462 matching the neighbour a
   *re-run* picks, because the loader iterates a Python `set`. **The fabrication is not even
   reproducible across processes.** Never reproduce this.

Sample of the deterministic recipe (real nodes, real relations):

```
m.0k8nh0b   CVT | types: olympics.olympic_athlete_affiliation
                 | relations: olympics.olympic_athlete_affiliation.athlete,
                              olympics.olympic_athlete_affiliation.country, ...
m.05nr4wx   CVT | types: military.military_command
                 | relations: military.military_command.military_combatant,
                              military.military_command.military_commander, ...
```

Both variants are cheap to build; the recipe is retained as a documented option so the
structure-only choice can be A/B'd at L1 without any corpus change. **Under either variant the CVT
keeps its own MID as identity and never borrows a name.**

---

## PHASE 6 — canonical representation (schema, not a build)

```
NODE
  node_id       canonical Freebase MID / g-id          "m.02mjmr"        REQUIRED, identity
  display_name  type.object.name (English) or null     "Barack Obama"
  aliases       [common.topic.alias]                   ["Barack Hussein Obama II"]
  types         [type.object.type]                     ["people.person", "government.politician"]
  is_cvt        bool, from cvtnodes.bin when available  false
  text          encoder input, DERIVED, never identity
                  ordinary topic : "<display_name>; aliases: <...>; types: <...>"
                  CVT            : null  (structure-only, recommended)
                                   or "CVT | types: <...> | relations: <...>"  (deterministic option)
EDGE
  source_mid    "m.02mjmr"
  relation_id   canonical Freebase property id         "people.person.place_of_birth"   NEVER replaced
  target_mid    "m.0d6lp"
```

Invariants:

* `node_id` is the MID. A surface string is never an identity key — the released artifact's own
  1,782-of-34,972 name→multi-MID collision rate (max 46) proves why.
* `relation_id` stays the canonical property id. A `relation_display_text` may be stored **beside**
  it and must never replace it.
* `text` is a derived field. Changing it must never change topology or identity.
* Forbidden text sources, enforced: WebQSP question text, answers, gold paths, topic annotations,
  arbitrary graph neighbours.
* Serialization: JSON/JSONL only. The gate's `SERIALIZATION_BLOCKER` is already cleared for the
  endpoint set (`ROUNDTRIP_FAILURES = 0` over 2,592,894 strings, `union_entities.jsonl`); the same
  lossless format applies here.

---

## PHASE 7 — four-way comparison

| | 1. legacy 781k | 2. Phase-C 1,316,466 | 3. released RoG union | 4. reconstructed MID graph |
|---|---|---|---|---|
| **N** | 781,485 graph nodes (+1,628 question nodes = 783,113 stored) | 1,316,466 | 2,592,894 | **NOT BUILT** |
| **identity** | `webqsp_doc_<i>`, positional | `webqsp_ent_<sha256(text)[:16]>` | released endpoint string (mixed) | would be the MID |
| **MID endpoints** | 459,417 | 776,788 | 1,652,618 | — |
| **named endpoints** | 322,068 | 539,678 | 925,761 | — |
| **fabricated names** | **458,872** | 0 | 0 | 0 by construction |
| **surface collisions manufactured** | **458,872** | 0 | 0 (upstream, ~5.10 % est.) | 0 |
| **MID recoverable** | only by re-deriving `sorted(entities)`; not stored | for the 59.0 % stored *as* MIDs | **NO** for the 925,761 named | yes, by definition |
| **graph source** | RoG-webqsp **TEST shards only**, 2,277,228 triples | all 5 WebQSP shards | WebQSP ∪ CWQ, 29 shards | Freebase |
| **gold-conditioned** | no (doc set) | **yes, 16,937 rows** | no | no |

**Overlap, measured:**

```
legacy781k ⊂ union          781,485 / 781,485      (0 outside)
legacy781k ⊂ PhaseC         781,485                (0 outside)
PhaseC ∩ union            1,299,529
PhaseC \ union               16,937                (the gold/topic injection residue)
union \ PhaseC            1,293,365                (the CWQ side)
```

`PhaseC ∩ union = 1,299,529` corrects the prior audit's 1,299,527 by **+2**: that run joined through
a non-invertible backslash escape which corrupted 5 of 2,592,894 strings; this one reads the
lossless `union_entities.jsonl`.

### 7.1 The legacy neighbour-name bug, verified directly

Re-implemented `deduce()` from `loader_webqsp.py:52-60` against the same TEST parquet shards the
loader read, and compared with the stored titles in `data/processed/master_nodes_webqsp.json`:

| metric | value |
|---|---:|
| MID nodes | 459,417 |
| MID with no named neighbour (title left blank) | 545 |
| **fabricated names** | **458,872** |
| fabricated title is **some** named neighbour of that MID | **458,872 (100.000 %)** |
| fabricated title is **not** any named neighbour | **0** |
| fabricated title equals the neighbour **this run** picks | 373,462 (81.4 %) |

**`BUG_VERIFIED = true`, and additionally `NON_DETERMINISTIC = true`.** Every stored title is
provably a stolen neighbour name — but 85,410 of them differ from what a re-run produces, because
the loader builds its adjacency by iterating a Python `set` of triples, whose order is
hash-seed dependent. So the legacy substrate is not only wrong, it is **unreproducible**. Examples:
`g.112yf9tt7 → "Wassily Kandinsky"` (text `"Wassily Kandinsky. artworks Wassily Kandinsky"`),
`g.112yfc45p → "Colombia"`, `g.112yfd_tp → "Dunkirk"`.

Commit `ba6bd71` ("resolve all raw Freebase MIDs to names") did not resolve anything; it made the
fabrication total. Its claim "ZERO raw IDs in any content/title" is true and is precisely the defect.

### 7.2 Phase-C identity rule confirmed

`canonical_doc_id == "webqsp_ent_" + sha256(text)[:16]` — verified on the stored `text_sha256`
field (`sha256("Jamaica")[:16] = 42c298ba17fed5da`, matching `webqsp_ent_42c298ba17fed5da`), and
`text == original_source_id` in **1,316,466 of 1,316,466** rows. The store is clean but bare.

---

## PHASE 8 — embedding-reuse projection (nothing encoded)

**Conditional, and labelled as such: canonical human-readable text is NOT frozen, because Phase 4
could not run.** What is projected is the *best text derivable today* under MID-level identity:
named endpoints and literals keep their released string; CVTs take schema text; name-resolvable
bare MIDs are blocked on metadata. Reuse is token-ID equality under the frozen tokenizers; a text
that changes cannot reuse and a byte-identical text always does, so the set-level answer is exact
and the two measured tokenizer corrections from the prior audit (dense `+8`, SPLADE `+3,231`, both
on surface rows — exactly the rows kept here) are carried forward.

| | baseline (released endpoint text) | under MID-level canonical text | Δ |
|---|---:|---:|---:|
| `DENSE_REUSE_N` | 1,299,535 (50.12 %) | **522,873 (20.17 %)** | **−776,662** |
| `DENSE_REENCODE_N` | 1,293,359 | **2,070,021** | +776,662 |
| `SPLADE_REUSE_N` | 1,302,766 (50.24 %) | **526,096 (20.29 %)** | **−776,670** |
| `SPLADE_REENCODE_N` | 1,290,128 | **2,066,798** | +776,670 |

Decomposition of the loss: of the 1,316,466 Phase-C rows, **742,946** become CVT schema text and
**33,718** are bare MIDs blocked on Freebase metadata; 522,865 keep byte-identical text. The
940,276 union endpoints whose text is unchanged are the 925,761 surfaces plus 14,515 literals.

**Reported plainly rather than optimised around**, per the directive. Two things make the drop
smaller than it looks:

* If CVTs are **structure-only** (the Phase-5 recommendation), 1,588,085 nodes need **no vector at
  all** and `DENSE_REENCODE_N` falls from 2,070,021 to **481,936** — the encode bill shrinks by 77 %
  relative to the projection above, and the *effective* corpus is ~1.00 M nodes.
* If CVT text is kept, the encoder cost is bounded by **distinct texts, not nodes**: 2,155 distinct
  strings for the pass-1 CVT population (449× compression). Either way the real work is the
  ~1.0 M ordinary topics.

**Also note the drop is not yet final and will get worse, correctly so.** Once real Freebase
metadata lands, the 925,761 named endpoints gain `; aliases: …; types: …` and *their* text changes
too, driving dense reuse toward ~0. That is the right trade: correct representation over preserved
embeddings.

---

## Recommendation

**Source: `FastRDFStore-data.zip` is the only correct graph source, and it is not viable on this
machine as configured. Do not substitute anything for it.**

1. **Do not adopt the third-party reconstruction.** `camazlucas/Freebase-WebQSP-CWQ-Subgraph` is
   50,304 triples and 16 relations short of the published graph — measurably worse than our own
   union on the two figures that can be checked. Diagnostic only, as the gate already says.
2. **Do not download the 300 GB `freebase-rdf-latest` mirror.** It is the wrong tool for the graph
   (it is all of Freebase, not the benchmark subgraph) and it does not fit.
3. **The metadata problem is much cheaper than the graph problem, and worth separating.** The
   `idirlab/freebases` Zenodo bundle is **13.18 GiB** (`md5:170689b7aad9f029566a4deb36605b01`) and
   carries `object_names`, `object_types`, `entities_id_label`, `properties_id_label` — enough for
   Phase 4 on a *fixed* node set. Fetching it is a 13 GiB, one-volume operation, versus 63.84 GiB
   plus two full-size derived files for the graph. **If only one download is ever authorised, this
   is the one**: it upgrades all 925,761 named endpoints with aliases and types and closes the
   62,375-node band, without touching topology.
4. **Sequencing, if the graph reconstruction is later authorised:** free ≥ 200 GB or use an external
   volume; run NSM Steps 0–3 against `fb_en.txt` + `cvtnodes.bin` with the WebQSP+CWQ topic-entity
   population; report entity/relation/triple counts against **both** the paper (2,566,291 /
   7,058 / 8,309,195) **and** the released union (2,592,894 / 7,058 / 8,309,195) with differences
   explained, never tuned; then attach metadata from step 3.
5. **Meanwhile, `is_cvt` does not block anything.** The structural classifier above reproduces the
   CVT partition at 96.10 % of bare MIDs with 0.045 % control false-positives, 95.75 % cross-signal
   agreement and 0.45 % threshold sensitivity. When `cvtnodes.bin` is eventually obtained it becomes
   a validation set for this classifier rather than a prerequisite.
6. **What must not happen:** promoting the released mixed-endpoint union to canonical identity;
   re-running the legacy `deduce()` heuristic in any form; or naming a CVT.

---

## Constraints honoured

Writes confined to `data/final_canonical/`, `scratchpad/final_canonical_build/` and
`results/data_audit/final_canonical_v1/`. Nothing written to `data/canonical/`, `data/processed/`,
`data/ukb_storage/`, `results/GENERALIZATION/`, `scratchpad/ablation_qwen/` or `data/original/` —
all four legacy artifacts were opened **read-only**. Nothing deleted. The five built node tables are
untouched. No canonical WebQSP node table was written, no encoder was run, no kNN, no H4.
**Nothing was downloaded.** All parquet and JSONL reads were streamed; peak RSS stayed well inside
16 GB. `data/` is gitignored; nothing committed.
