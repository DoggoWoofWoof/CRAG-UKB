# WebQSP representation audit

**Status: AUDIT ONLY. Nothing was built, encoded, or promoted.**
No `data/final_canonical/webqsp/nodes.jsonl` was written. No encoder was run. H4 was not touched.
The only new bytes outside the scratchpad are the 10 remaining official RoG-CWQ shards, this
report, `data/original/{webqsp,cwq}/SOURCE_PROVENANCE.json`, and `data/final_canonical/GOLD_SEMANTICS.json`
(that last file has since been folded into each dataset's `integrity_report.json` and moved to
`data/final_canonical/_superseded_sidecars/GOLD_SEMANTICS_preSourceFix.json` — see the correction in
"Gold semantics" below).

Generated 2026-09-05. Every number below is derived from the artifacts on this machine; none is
copied from a paper or a third party. Reproduction scripts:
`scratchpad/final_canonical_build/audit_webqsp_{A_legacy,BCD_union,E_reuse,E2_persource}.py`,
`audit_2wiki_gold.py`, `fetch_rog_cwq_resume.py`. Raw outputs: `scratchpad/final_canonical_build/_audit/`.

---

## 0. Deliverable table

| SOURCE | ENTITY_N | MID_ONLY_N | READABLE_N | IDENTITY_FORMAT | TEXT_SOURCE | GOLD_CONDITIONED | DENSE_REUSE_N | SPLADE_REUSE_N | SAFE_FOR_CANONICAL |
|---|---:|---:|---:|---|---|---|---:|---:|---|
| **legacy WebQSP-only — Phase-C `data/canonical/webqsp` (1,316,466)** | 1,316,466 | 776,788 | 539,675 | `webqsp_ent_<sha256(surface)[:16]>` — content hash of the surface string; **the MID is not the id** | RoG endpoint surface string, **verbatim** (`text == original_source_id` in 1,316,466/1,316,466 rows) | **YES** — 18,160 rows injected from `a_entity` (`c1c2_webqsp.py:32`) | 1,316,466 (it *is* the store) | 1,316,466 | **NO** — gold-conditioned; WebQSP-only |
| *legacy 781k substrate — `data/processed/master_nodes_webqsp.json`* | 781,485 | **0** | 781,483 | `webqsp_doc_<i>`, positional index into `sorted(entities)` | **fabricated**: verbalized name + ≤15 relations, MID names *deduced from a graph neighbour* (`loader_webqsp.py:52-60,71`) | no (doc set), but graph is **RoG TEST-split only** | n/a (no `data/canonical` store) | n/a | **NO** — test-split-only, and 459,417 names are fabricated |
| **official RoG-WebQSP** | 1,298,306 | 776,660 | 513,090 (+8,556 OTHER) | mixed: bare Freebase MID **or** human-readable surface; MID discarded for named entities | official RoG graph endpoint, query-independent | no | 1,298,304 | 1,298,304 | partial — WebQSP-only union is the **rejected** object |
| **official RoG-CWQ** | 2,259,510 | 1,442,908 | 803,858 (+12,744 OTHER) | same as above | official RoG graph endpoint, query-independent | no | 966,151 | 969,382 | partial — half of the required union |
| **unified deduplicated union (WebQSP ∪ CWQ)** | **2,592,894** | **1,652,618** | **925,761** (+14,515 OTHER) | mixed MID / surface; **surface-keyed, MID not recoverable** | official RoG graph endpoint, query-independent | **no** | **1,299,535** | **1,302,766** | **YES for membership and text; identity carries a measured collapse risk (§D)** |

`MID_ONLY_N` = endpoints matching `^[mg]\.[0-9A-Za-z_]+$`. `READABLE_N` = human-readable surfaces;
`OTHER` (numeric/date literals, filenames, URLs) is reported separately and is 0.56 % of the union.
Reuse columns are **token-ID measurements, not estimates** (§E).

---

## A. Legacy WebQSP forensics

There are **two** legacy WebQSP representations, built by different code, from different splits,
with different identity and text rules. Neither is the object the acceptance gate asks for.

### A.1 Provenance chain

| | legacy 781k (`master_nodes_webqsp.json`) | Phase-C 1,316,466 (`data/canonical/webqsp/`) |
|---|---|---|
| graph source | `src/pipeline/loader_webqsp.py:118-123` — `--paths` defaults to `data/raw/full/kb/webqsp_test0.parquet` + `webqsp_test1.parquet`, byte-identical to RoG **test** shards. **TEST SPLIT ONLY**, 2,277,228 unique triples | `scratchpad/c1c2_webqsp.py:13` — `glob("data/original/webqsp/rog_webqsp/*.parquet")`, **all splits** |
| endpoint identity format | `loader_webqsp.py:37-40` `str(t[0]), str(t[1]), str(t[2])` — RoG strings verbatim, no normalization | `c1c2_webqsp.py:30-31` `add(tr[0]); add(tr[2])` — RoG strings verbatim |
| node ID rule | `loader_webqsp.py:46-47` `entities = sorted(ents)`; `ent2nid[e] = f"{source}_doc_{i}"` — **positional**, so the id changes if the corpus changes | `c1c2_webqsp.py:23` `did = f"webqsp_ent_{sha(name)[:16]}"` — **content hash of the surface string** |
| node text rule | `loader_webqsp.py:63-75` — `f"{name}. {'; '.join(rel + ' ' + obj)}"`, ≤ `max_rels=15` incident relations | `c1c2_webqsp.py:25` — `"text": name`. The surface string and nothing else |
| MID resolution source | **the graph itself** — `loader_webqsp.py:52-60` `deduce(x)` returns *the first non-MID neighbour in adjacency order*; `:71` uses it as the node's own name | **none.** Bare MIDs are left bare |
| name/alias resolution source | **none** | **none** |
| gold/topic injection point | none in the doc set (`:92` uses `a_entity` only to attach gold *edges*) | **`c1c2_webqsp.py:32-33`** — `for a in aslist(r["a_entity"]): add(a)` and `for q in aslist(r["q_entity"]): add(q)` |
| encoder input recipe | n/a — no `data/canonical` store | `encodings/_src/docs/shard_*.jsonl` = `{"id": canonical_doc_id, "text": <surface verbatim>}`. Measured: encoder input differs from `documents.text` in **0 of 1,316,466** rows. No prefix, no instruction |

### A.2 The critical question — where did the non-MID legacy texts come from?

**Answer: from RoG endpoint surface strings, and nothing else. There is no MID→name dictionary in this project.**

* Phase-C (1,316,466): `text == original_source_id` for **all 1,316,466 rows**. The text *is* the RoG
  endpoint string. Nothing was resolved, enriched, or looked up.
* Commit `ba6bd71` ("resolve all raw Freebase MIDs to names in doc text") did **not** use a name source.
  It used `deduce()` — a graph-local heuristic that gives a MID the name of an adjacent named node.
  That is not resolution; it is **fabrication**, and it manufactures identity collisions:

  | endpoint | legacy title | legacy text |
  |---|---|---|
  | `g.112yf9tt7` | `Wassily Kandinsky` | `Wassily Kandinsky. artworks Wassily Kandinsky` |
  | `g.112yfc45p` | `Colombia` | `Colombia. agencies Colombia` |
  | `g.112yfd_tp` | `Dunkirk` | `Dunkirk. contains Dunkirk` |
  | `g.112yfddg1` | `Nicolas Poussin` | `Nicolas Poussin. artworks Nicolas Poussin` |

  459,417 of the 781,485 nodes (58.8 %) are CVT/mediator MIDs that were given a neighbour's name this
  way. The commit's claim "ZERO raw IDs in any content/title" is true and is exactly the problem: a
  mediator node is now textually indistinguishable from the real entity it hangs off.
* Repo-wide search for `mid2name`, `entity2name`, `id2entity`, `entities_id_label`, `freebase_names`,
  `surface_map`, `entity_vocab`, `name2mid`, `type.object.name`: **no such file exists.** The only
  `EntityName` reads are `c1c2_webqsp.py:51` and `build_kb.py:551`, which read
  `WebQSP.{train,test}.json → Parses[].Answers[].EntityName` — the **answer annotation**, i.e.
  query-conditioned, and forbidden for node text.

**So no, we do not already possess a query-independent metadata resolver.** What we possess is the RoG
surface strings themselves — which, as §B shows, is most of what a resolver would have given us anyway.

### A.3 Requested counts

| | LEGACY_N | MID_ONLY_N | READABLE_TEXT_N | MIXED_OR_OTHER_N |
|---|---:|---:|---:|---:|
| legacy 781k substrate (text as encoded) | 781,485 | **0** | 781,483 | 2 |
| legacy 781k substrate (*true* endpoint identity) | 781,485 | *459,417* | *322,068* | 0 |
| Phase-C 1,316,466 (text as encoded) | 1,316,466 | **776,788** | 539,675 | 3 |
| Phase-C 1,316,466 (endpoint identity) | 1,316,466 | 776,788 | 539,678 | 0 |

The two rows per artifact are the point: for the 781k substrate `MID_ONLY_N = 0` **at the text level and
459,417 at the identity level** — the MIDs did not go away, their names were invented. For Phase-C the
two rows agree, because the text is the identity.

Gold conditioning, measured: **18,160** Phase-C rows are `a_entity` strings that are not RoG-WebQSP graph
endpoints (`1,316,466 − 1,298,306 = 18,160`). Against the full WebQSP ∪ CWQ union, 16,939 of them remain
non-endpoints (1,221 happen to appear in CWQ subgraphs). `q_entity` injection added 0 new rows.

---

## B. Official RoG sources

| repo | revision | files | all verify |
|---|---|---|---|
| `rmanluo/RoG-webqsp` | `c0632533135a06f8c5d536b420deec5fcb5c58f3` | 5 parquet (train 2 / validation 1 / test 2) | **YES** — size + LFS sha256, `fileset_sha256 = 5ddc95d0315cc3f37defd5d3743e53857cea5d96784e804f60e20aeef3d8cb64` |
| `rmanluo/RoG-cwq` | `b0f6275586286312c4e99ef4b0adc65463c91f8e` | **24** parquet (train 18 / validation 3 / test 3) + README, 3,504,046,556 B | **YES** — `fileset_sha256 = 193c950555d050b07753da24a5b830bcb285d0fba5e06cdc17d12b1fa0a763b0` |

Per-file sizes and hashes: `results/data_audit/final_canonical_v1/webqsp/ROG_{WEBQSP,CWQ}_SOURCE_PROVENANCE.json`
(the CWQ record is also written by the fetcher to `data/original/cwq/SOURCE_PROVENANCE.json`).

**CWQ acquisition.** The previous run left 14 files and was described as "14 of 21". The true file set is
**24 parquet shards, not 21** — the earlier count omitted the three `validation-*-of-00003` shards. All 14
pre-existing files re-verified against the HF LFS sha256 (including `train-00010`, the last one logged);
**nothing was truncated**, and the shard feared lost (`train-00011`) had never landed at all. The remaining
10 were fetched and verified. RoG-CWQ is **fully acquired**; no substitution was needed.

### B.1 Endpoint classification

| | ENTITY_N | FREEBASE_MID | HUMAN_READABLE_SURFACE | OTHER |
|---|---:|---:|---:|---:|
| RoG-WebQSP | 1,298,306 | 776,660 (59.82 %) | 513,090 (39.52 %) | 8,556 (0.66 %) |
| RoG-CWQ | 2,259,510 | 1,442,908 (63.86 %) | 803,858 (35.58 %) | 12,744 (0.56 %) |
| **union** | **2,592,894** | **1,652,618 (63.74 %)** | **925,761 (35.70 %)** | **14,515 (0.56 %)** |

`OTHER` breakdown (union): numeric literals 12,513 (`"37317"`), filename/URL-shaped strings 1,628
(`"male.jpg"`, `"www.oica.net"`), bare years/dates 374 (`"1776"`). MIDs split `m.` 1,247,261 / `g.` 405,357.
Relations are Freebase schema paths (`people.person.nationality`) and are 100 % well-formed.

### B.2 Does an MID↔surface mapping survive in the official artifacts?

**No.**

* Both HF repos contain only `.gitattributes`, `README.md`, and parquet shards. There is no mapping file.
* The parquet schema is `id, question, answer, q_entity, a_entity, graph, choices`. There is **no MID column**
  anywhere, and `graph` is `list<list<string>>` of three bare strings.
* `q_entity` / `a_entity` are themselves surfaces, not MIDs: MIDs are 1/2,249 and 153/33,651 (WebQSP),
  218/11,748 and 6/6,145 (CWQ).
* `rmanluo` publishes exactly two datasets (`RoG-webqsp`, `RoG-cwq`) — no mapping dataset.
* The RoG repository README documents no entity-label resource and defers subgraph extraction to
  external code ("We extract the subgraphs from the Freebase following previous studies").

**RoG resolved MIDs to names once, at build time, and threw the MIDs away.** What survives is a
*one-way* projection: name where Freebase had one, bare MID where it did not.

---

## C. The official union (temp only)

Built by deterministic streaming union + dedup of every `graph` triple in all 29 official parquet shards,
one shared entity/relation vocabulary, triples packed to `int64` and `np.unique`'d. Verified against a
naive Python `set` on a held-out shard: **exact match** on triples, entities and relations.
Nothing was written to `data/final_canonical/webqsp/` except this report.

| | ENTITY_N | RELATION_N | TRIPLE_N |
|---|---:|---:|---:|
| **our union (derived)** | **2,592,894** | **7,058** | **8,309,195** |
| paper target | 2,566,291 | 7,058 | 8,309,195 |
| Δ vs paper | **+26,603 (+1.04 %)** | **0 — EXACT** | **0 — EXACT** |
| third-party reconstruction (diagnostic only) | 2,574,900 | 7,042 | 8,258,891 |
| Δ vs third party | +17,994 | +16 | **+50,304** |

Supporting counts: 158,771,837 raw triple instances (19,986,134 WebQSP + 138,785,703 CWQ) collapsing to
8,309,195 unique — a 19.1× redundancy factor across per-question subgraphs. 2,751,557 triples occur in both
datasets; 1,039,746 are WebQSP-only and 4,517,892 CWQ-only. 964,922 entities are shared, 5,685 relations
shared. 93,692 self-loops. **0 malformed-arity rows, 0 null endpoints, 0 dangling endpoints** (every endpoint
defines an entity by construction).

**Explaining the differences.**

* **Triples and relations match the published figures exactly.** Matching 8,309,195 and 7,058 simultaneously
  by coincidence is not credible, so our triple set *is* their triple set. This is the strongest available
  evidence that the union is correctly reproduced.
* Given an identical triple set, the endpoint set is forced, so the **+26,603 entity difference must be a
  counting convention on their side, not a construction error on ours.** Tested and **rejected** as
  explanations: `.strip()` (−2), whitespace collapse (−29), NFC (−30), casefold (−9,881), NFC+ws+casefold
  (−9,951), dropping all `OTHER` literals (−14,515), and every combination of those; also head-only
  (1,555,843) and non-tail-only positional filters. None lands on 2,566,291, and no single-property subset
  of the union (non-ASCII 44,560; backslash-bearing 5,032; URL/file-shaped 7,260; `g.` MIDs 405,358) has
  cardinality 26,603. The residual is 1.04 % and is **reported, not forced**.
* The third-party reconstruction is **worse than ours on the two figures that can be checked**: it is
  50,304 triples and 16 relations short of the published graph. That is why it is not adopted.

**Guard against the rejected object.** The WebQSP-only union is **1,298,306 entities / 6,094 relations /
3,791,303 triples** — precisely the "~1.30 M entities / ~3.79 M triples" the directive warns about. It was
computed here only as a component. **The reported union is WebQSP ∪ CWQ. We did not fall back.**

---

## D. Identity safety

Graph endpoints are **mixed**: 63.74 % bare Freebase MIDs, 35.70 % resolved display names, 0.56 % literals.

| metric | value |
|---|---|
| `UNIQUE_SURFACE_N` | 925,761 |
| `UNIQUE_MID_N` | 1,652,618 |
| `DUPLICATE_SURFACE_GROUPS` (exact string) | **0 — by construction** |
| `MAX_COLLISION_SIZE` (observable in the union) | **0 — by construction** |
| case/whitespace-variant groups | 9,334 |
| surfaces involved in those | 19,228 |
| max case/whitespace group size | 5 |

**The zero is the finding, not the reassurance.** The union is keyed on the endpoint string, so any two
Freebase objects that shared a display name were **already merged upstream by RoG, before the data reached
us**. The collapse is invisible in this artifact and cannot be measured from it. It has to be estimated
from a MID-bearing source.

### D.1 Measured collision rate from a MID-bearing source

`WebQSP.{train,test}.json` carries `(TopicEntityMid, TopicEntityName)` and `(AnswerArgument, EntityName)`
pairs — 57,131 pair instances over 34,972 distinct names. Used here **as a diagnostic only**; no node text
is derived from it.

* **1,782 of 34,972 names (5.10 %) map to more than one distinct Freebase MID.**
* **Maximum collision size: 46.**
* Examples: `Matilda` → ≥6 MIDs; `Night` → ≥6; `The Prince and the Pauper` → ≥6; `Being With You` → ≥6.

Extrapolated to the union's 925,761 surfaces, order 10⁴–10⁵ entity identities are silently merged. The
9,334 case/whitespace-variant groups are a second, independent symptom: RoG's surface identity is not even
case-consistent (`the house of mirth` occurs 5 times with different casing).

### D.2 Can original MID identity be recovered?

**No — not from the official artifacts.** For the 1,652,618 endpoints that *are* MIDs, identity is exact and
already present. For the 925,761 named endpoints the MID was discarded (§B.2), and nothing in either repo,
its schema, or the RoG code restores it. Recovering it requires an external MID-bearing Freebase source.

**Verdict on the preferred outcome.** The directive's preferred pair —
`canonical ID = MID`, `canonical text = official query-independent surface` — is **not achievable from RoG
alone**, and this is proven rather than assumed. The achievable pair is:

```
canonical ID   = the RoG endpoint string (MID where RoG kept one, surface otherwise)
canonical text = the same string, verbatim
```

which is exactly what `_WEBQSP_ACCEPTANCE_GATE.json → text_representation_decision` already specifies,
and it carries the ~5 % surface-collapse risk quantified above. The gate's `NODE_ID: "EXPECTED: Freebase MID"`
is **only satisfiable for 63.74 % of nodes**; that expectation needs amending or an external source.

---

## E. Legacy ↔ RoG reuse

Legacy side: `data/canonical/webqsp/documents.jsonl` (1,316,466 rows) and the exact encoder inputs in
`encodings/_src/docs/shard_*.jsonl`. Both stores are complete (`rows_covered == n_items == 1,316,466`).
Official side: the union entity list, canonical text = the surface string verbatim.
Join key: the endpoint string (which is also the legacy text and the legacy encoder input).

| class | N |
|---|---:|
| `EXACT_ID_AND_TEXT` | **1,299,527** |
| `SAME_ENTITY_TEXT_CHANGED` | 0 |
| `SAME_TEXT_ID_CHANGED` | 0 |
| `ROG_ONLY` | 1,293,367 |
| `LEGACY_ONLY` | 16,939 |
| `AMBIGUOUS_MAPPING` | 0 |

`EXACT_ID_AND_TEXT` means: the endpoint is in both, the legacy text *and* the legacy encoder input are the
RoG surface verbatim, **and** `webqsp_ent_<sha256(surface)[:16]>` reproduces the stored id exactly.
`SAME_TEXT_ID_CHANGED` is 0 only because the Phase-C id rule is kept; adopting `build_kb.py:545`'s
`webqsp:<sha256(surface)[:24]>` would move all 1,299,527 rows into that class — a pure remap, not a re-encode.
`LEGACY_ONLY` = 16,939 is exactly the gold-injected residue (§A.3).

### E.1 Token-ID reuse — measured, not estimated

Decision rule identical to `reuse_map_kb.py:38-41,100-112`: sha256 over the comma-joined token-ID sequence.
Dense `Alibaba-NLP/gte-Qwen2-1.5B-instruct`, verbatim text, no prefix, `add_special_tokens=True`,
`truncation=True`, `max_length=32768`. SPLADE `naver/splade-cocondenser-ensembledistil`, same flags,
`max_length=256`. 2,609,833 distinct encoder inputs tokenized; memoization re-verified in a fresh process on
500 random inputs — **PASS**.

| target | ENTITY_N | DENSE_REUSABLE | DENSE_REENCODE | SPLADE_REUSABLE | SPLADE_REENCODE |
|---|---:|---:|---:|---:|---:|
| RoG-WebQSP | 1,298,306 | 1,298,304 | 2 | 1,298,304 | 2 |
| RoG-CWQ | 2,259,510 | 966,151 | 1,293,359 | 969,382 | 1,290,128 |
| **union** | **2,592,894** | **1,299,535 (50.12 %)** | **1,293,359** | **1,302,766 (50.24 %)** | **1,290,128** |

Two independent details confirm the rule is doing real work rather than proxying text equality:

* dense reuse (1,299,535) **exceeds** raw-text equality (1,299,527) by 8 — eight official entities tokenize
  identically to a stored row despite differing bytes;
* SPLADE reuse **exceeds** dense reuse by 3,231 because SPLADE's WordPiece tokenizer is **uncased**, so
  case-variant surfaces share a token sequence — the same phenomenon as the 9,334 case-variant groups in §D.

16,939 stored dense rows (16,767 SPLADE) match no official entity and become dead weight.

**Reuse is ~50 %, not the ~0–1.3 % predicted in `FREEBASE_SOURCE_PROPOSAL.md`.** That prediction assumed
the corpus would move to full-Freebase scale with name-enriched text. Under the approved RoG-standard
decision — surfaces kept verbatim — half the union is already encoded and paid for. Only the CWQ-side
1,293,359 dense / 1,290,128 SPLADE rows are new.

*Measurement caveat, stated in full:* the temp entity list uses a non-invertible escape for backslashes.
Exactly **5 lines of 2,592,894** fail to round-trip; those five are the only rows whose measured text could
be wrong. Everything else is exact.

---

## F. Resolution decision

**Recommendation: use the official RoG surfaces. Do NOT download a Freebase dump.**

Evidence:

1. **RoG already did the resolution.** 925,761 union endpoints (35.70 %) are human-readable, query-independent
   surface strings supplied by the official artifact. A dump would re-derive names we already have.
2. **The 1,652,618 bare MIDs are not an unresolved-name problem.** They are CVT/mediator and unlabelled
   Freebase nodes — objects that *have no* `type.object.name`. A dump would not name them either; the legacy
   `deduce()` heuristic only made that visible by inventing names for 459,417 of them (§A.2). `UNRESOLVED_MID_SET`
   is therefore **not** a small lookup that a metadata source would fix — it is 63.74 % of the graph, and the
   correct policy is the one already frozen: **leave bare MIDs bare and report the fraction.** That fraction is
   **63.74 % of union entities**.
3. **Half the encoding cost is already paid.** 1,299,535 dense and 1,302,766 SPLADE rows are reusable by
   token-ID equality (§E.1). Switching to dump-enriched text would invalidate **all** of them.
4. **A dump does not fix the real identity risk.** The risk is the ~5 % surface collapse (§D.1), which happened
   upstream when RoG discarded the MID. A dump gives names for MIDs we keep as MIDs; it cannot restore the MID
   for a name RoG already merged. Only a *re-extraction* of the subgraphs from a MID-preserving pipeline would,
   and that is a different project.

**One flag for the acceptance gate, not a blocker.** `_WEBQSP_ACCEPTANCE_GATE.json` asserts
`NODE_ID: "EXPECTED: Freebase MID"` and `EDGE: (source MID, relation, target MID)`. That is **false for
35.70 % of endpoints** and is not achievable from RoG (§D.2). The gate should be amended to
`NODE_ID = RoG endpoint string (MID for 63.74 %, surface for 35.70 %)`, with the measured 5.10 %
surface-collision rate recorded as a known limitation of the standard benchmark graph. Everything else in the
gate is satisfiable from what is now on disk.

**Not recommended, and not done:** a 22 GB Freebase dump; any enrichment from WebQSP questions, answers,
gold paths or topic annotations. That route is forbidden *and* worthless: the official annotations yield
39,882 distinct MIDs with a name, which cover **33 of the union's 1,652,618 bare MIDs (0.002 %)** — and 31 of
the 776,788 bare-MID Phase-C rows — because they describe topic/answer entities, which RoG already stores by
name. (Measured here directly, independently of the earlier `FREEBASE_SOURCE_PROPOSAL.md` estimate, which it
confirms.)

---

## Secondary task — 2Wiki gold semantics (and the other four datasets)

`build.py:344` (and `:564,570,580`) computes
`n_all_gold_resolved = sum(1 for r in recs if len(r["gold_node_ids"]) == len(set(r["gold_refs"])))` —
**exactly the `len(gold_node_ids) == len(gold_refs)` test the directive forbids.** It fails a question merely
because one of its references is ambiguous. That, and nothing else, is why `ALL_EVAL_GOLDS_PRESENT = false`:
2wiki train reports `gold_refs_total = 404,170` against `gold_nodes_total = 404,178`, and the 8-node excess is
the 8 questions whose supporting-fact title `"Unconquered"` maps to two curids.

**Re-derived independently** (not trusted from the cancelled run): a full scan of the frozen
`nodes.jsonl` (5,989,847 lines) rebuilding `title → {node_id}` for all 188,350 distinct gold titles.

| dataset | splits with gold | GOLD_REFS_TOTAL | ALL_GOLD_REFS_RESOLVE | MISSING_GOLD_REFS | AMBIGUOUS_GOLD_REFS |
|---|---|---:|---|---:|---:|
| 2wiki | train, dev | 434,824 | **true** | **0** | 8 |
| musique | train, dev | 53,017 | **true** | **0** | 0 |
| metaqa | train, dev, test | 3,049,920 | **true** | **0** | 0 |
| squad | train, dev | 142,192 | **true** | **0** | 0 |
| hotpotqa | train, validation | 195,704 | **true** | **0** | 0 |

Independent re-derivation for 2wiki: `titles_needed = 188,350`, `titles_matched = 188,350`,
`titles_unmatched = 0`, `ambiguous_titles = 1` (`Unconquered → {2wiki:c62717110, 2wiki:c8073502}`),
and **0 records disagreed** with the frozen `gold_node_ids`.

**`AMBIGUITY_RESOLVED_BY_SOURCE = 8 of 8.`** Every one of the 8 questions carries its own official context
paragraph for the title, and in all 8 it is *"Unconquered is a 1947 adventure film produced and directed by
Cecil B. DeMille…"* → **`2wiki:c8073502`** uniquely. The other candidate, `2wiki:c62717110`, is the
disambiguation page (*"Unconquered is the title of several American films:"*) and is never the referent.
All 8 questions ask about the film's **director**, consistent with the DeMille article. So the disjunction
fallback is **not needed**; it is recorded as the correct semantics should any future ref stay ambiguous
(satisfied iff `retrieved ∩ {B, C} ≠ ∅` — **never** requiring both).

`ALL_GOLD_REFS_RESOLVE = true` for all five datasets. `MISSING_GOLD_REFS = 0` everywhere.

**Invariant held:** 2wiki `CORPUS_HASH` asserted `81fa7d1a5d4bbb24…` **before and after** — unchanged. This is
evaluation semantics, not corpus membership. Corrected fields are emitted to
`data/final_canonical/GOLD_SEMANTICS.json`; `integrity_report.json` was **not** rewritten (that would be a
build action, and this is an audit), so it still carries the stale `ALL_EVAL_GOLDS_PRESENT: false`.

**[CORRECTION 2026-09-05 — the build action described above has since been carried out and this paragraph
is superseded.** Both builders were patched to compute per-reference gold semantics natively, and all five
datasets were re-run with `--queries-only`. `integrity_report.json` now carries a `GOLD_SEMANTICS` block and
`ALL_EVAL_GOLDS_PRESENT: true`; the sidecar was moved to
`data/final_canonical/_superseded_sidecars/GOLD_SEMANTICS_preSourceFix.json`. One number changed in the
process: 2wiki `AMBIGUOUS_GOLD_REFS` went **8 → 0**, because the builder now *resolves* the eight
`"Unconquered"` references from each question's own official context paragraph (all eight → `2wiki:c8073502`,
the 1947 DeMille film) instead of merely observing that two candidates exist. `CORPUS_HASH` is still
`81fa7d1a5d4bbb24…`.**]**

**[SECOND CORRECTION 2026-09-06 — the 2wiki corpus hash quoted twice above has since moved, deliberately.**
Both statements were true when written; neither is a claim about this audit's own findings, which are
unaffected. On 2026-09-06 `TEXTUALIZATION_REV 2` was applied to 2wiki (87,765 empty-text records
retextualized to their source-native title, the same general rule hotpotqa received a day earlier), moving
`CORPUS_HASH 81fa7d1a5d4bbb24… → 94fe68b8d5f291f1…` while leaving `NODE_ORDER_HASH`, membership, node ids and
node order unchanged. The gold semantics recorded above survived that change unaltered and were re-verified
against the new table: `MISSING_GOLD_REFS = 0`, `AMBIGUOUS_GOLD_REFS = 0`, `ALL_GOLD_REFS_RESOLVE = true`, and
the eight `"Unconquered"` references still resolve uniquely to `2wiki:c8073502`. Where this audit says the
2wiki corpus hash is `81fa7d1a…`, read `94fe68b8…`; the rev1 table is preserved at
`data/final_canonical/_superseded_textualization_rev1/2wiki/`.**]**

---

## Constraints honoured

No writes to `data/canonical/`, `data/processed/`, `data/ukb_storage/`, `results/GENERALIZATION/`,
`scratchpad/ablation_qwen/`. **One deviation, self-reported:** a WebQSP source-provenance record was first
written to `data/original/webqsp/SOURCE_PROVENANCE.json`, which is outside the permitted write list; it was
moved (not deleted) to `results/data_audit/final_canonical_v1/webqsp/ROG_WEBQSP_SOURCE_PROVENANCE.json`, and
`data/original/webqsp/` is back to its prior contents. Nothing else was created outside
`data/final_canonical/`, `scratchpad/final_canonical_build/`, `results/data_audit/final_canonical_v1/`, and
`data/original/cwq/`. Nothing deleted. The five built node tables are untouched and
`scratchpad/final_canonical_build/build.py` still hashes
`0879e428a41e636bb4089ef10c8ca217030331e22d54ca1d9dc87ba38988dcd9`.
**[CORRECTION 2026-09-05 — the sentence above was true when this audit was written and is now stale.**
Both builders were subsequently unfrozen and patched under the authorised 2Wiki gold fix:
`build.py` → `26455c9f72191dd46281d0d107d2236d8beec3168552ec67ddecd13832311e93`,
`build_kb.py` → `862e0b3c284fd12a08a97e7b1cefd1de1467effffdd62025fe09fa81eef1c3b1` *— itself superseded 2026-09-06, see next block*.
The claim that the node tables are untouched still holds: the re-runs used `--queries-only`, which
verifies the frozen corpus hashes and never opens `nodes.jsonl` for writing. See
`_WEBQSP_ACCEPTANCE_GATE.json → 2WIKI_FIX_AT_SOURCE._RESOLVED_2026_09_05`.**]**
**[CORRECTION 2026-09-06 (builder identity) — the correction above has itself gone one revision stale, and
the "node tables are untouched" claim no longer holds for 2Wiki.** `build_kb.py` was patched again under
TEXTUALIZATION_REV 2 and now hashes
`e723192e739de38880f04d92fdfc788604d4cfd95c17529c5bd4a52633e326f6`; `862e0b3c284fd12a…` above is the
sha as of the 2026-09-05 gold fix and is kept as the historical record of that patch, not as a live value.
The rev-2 change did rewrite `2wiki/nodes.jsonl` — deliberately and with approval — retextualizing the
87,765 records whose canonical body was empty to their source-native title, which moved 2Wiki's
`CORPUS_HASH` `81fa7d1a5d4bbb24…` → `94fe68b8d5f291f1…` while leaving `NODE_ORDER_HASH`, N, the node-id
set and the node order identical. `build.py` is unaffected and still hashes `26455c9f72191dd4…`; webqsp
holds no node table of its own and is untouched by all of this. See
`_superseded_textualization_rev1/SUPERSEDED.md` and `2wiki/REV2_DIFF_AUDIT.json`.**]** All parquet reads
were streamed (peak well inside 16 GB). Nothing committed — `data/` is gitignored.
