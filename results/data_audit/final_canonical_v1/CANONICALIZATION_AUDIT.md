# FINAL DATASET CANONICALIZATION AUDIT (Track B, canonical_v1)

Generated 2026-09-06T03:15:49 at git ba6bd71bcc9f367dc28d3e4cde777a3931501406. Invariant: **QUERY SUBSETTING IS ALLOWED. CORPUS SUBSETTING BY QUERY IS FORBIDDEN. Canonical corpus is fixed once per dataset, independent of the evaluated queries.**

## Summary table

| dataset | legacy substrate | legacy N | canonical_v1 N | legacy classification | corpus depends on eval subset | query-independent | all golds resolve | CORPUS_HASH |
|---|---|---:|---:|---|---|---|---|---|
| metaqa | metaqa | 40,151 | 43,234 | OTHER: LOSSY_IDENTITY_COLLAPSE (full-KB coverage, entity identity merged case variants) | False | True | True | 5f55719a08c1 |
| 2wiki | 2wiki_clean | 65,865 | 5,989,847 | SAMPLE_CONDITIONED | True | True | True | 94fe68b8d5f2 |
| musique | musique_clean | 13,672 | 117,534 | SAMPLE_CONDITIONED + GOLD_CONDITIONED (all-gold pool, no distractors, no dev/test paragraphs) | True | True | True | dc00da21ea8a |
| squad | squad_clean | 19,029 | 20,233 | OTHER: SPLIT_CONDITIONED_TRAIN_ONLY | False | True | True | f592d4ab6195 |
| hotpotqa | hotpotqa_clean | 507,494 | 5,233,329 | SETTING_CONDITIONED (distractor contexts, not fullwiki) + SAMPLE_CONDITIONED | True | True | True | 1b7eeac2bfbd |
| webqsp | webqsp | 781,485 | **BLOCKED — none built** (rejected candidate 1,316,466) | SAMPLE_CONDITIONED (corpus = union of the evaluated questions' own subgraphs; also STALE vs the RoG train+val+test entity union) | True | BLOCKED | BLOCKED | - |

## How each legacy substrate was generated (traced in code)

### metaqa (legacy `metaqa`, 40,151 nodes, eval 1998 = `val, hop-balanced (identical to _ta_resid.py / _ta_comb.py)`)

- **Generation:** FULL SOURCE KB with LOSSY IDENTITY: src/pipeline/loaders.py:481-590 load_metaqa() builds one node per kb.txt triple endpoint after strip().lower().replace('_',' ') (lines 529-531); node id = 'metaqa_ent_'+lowercased name (554); title = 'prettiest casing' seen (533-538); content = display name + up to 10 verbalized triples (562-571). Questions: metaqa_q_{hop}hop_{split}_{global counter} over {hop}-hop/vanilla/qa_{split}.txt (594-609); golds = answer strings lower()-matched to entity ids (624-628). data/raw/metaqa == data/original/metaqa (all files sha-identical).
- **Legacy normalization:** lower(), '_'->' ', strip() on entity names -> 43,234 official entities collapse to 40,151 (3,081 groups, 3,083 names absorbed); answer matching case-insensitive
- **Legacy dedup:** by lowercased name (implicit)
- **Legacy ids:** metaqa_ent_<lowercased name>
- **Depends on eval subset:** False; **stale:** False; **classification:** OTHER: LOSSY_IDENTITY_COLLAPSE (full-KB coverage, entity identity merged case variants)
- **Phase-C (2026-08-23) rule:** scratchpad/c1_metaqa.py:20-31 -- one doc per kb_entity_dict.txt row, id metaqa_ent_{idx}, text = name, no normalization; kb.txt entity set cross-checked (lines 15-18, 35) -> 43,234 (cross-check only)
- **canonical_v1 identity:** official kb_entity_dict index; names exact; node_id metaqa:e<idx:05d>; raw records 43,234 = nodes 43,234 + duplicates 0 + dropped 0
- **Count reconciliation vs clue 43,234:** delta 0 -- identical

- **Legacy -> canonical mapping:** 40,151/40,151 mapped ({'exact_name': 40151}), missing 0, ambiguous 0; canonical nodes matched 40,151; extra canonical nodes 3,083 by split provenance {'kb': 3083}; changed text {}

- **Frozen eval subset gold consistency (legacy golds mapped vs canonical golds):** equal 1952/1998, differ 46

- **Eval subset recovery:** legacy id metaqa_q_{hop}hop_{split}_{global_counter}; counter reproduced over vanilla qa files in (hop 1..3) x (train,dev,test) order, skipping blank/no-tab lines; question text verified equal -> 1998/1998 recovered, 0 missed; official split distribution {'dev': 1998}

- **Lowercase collapse:** 40,151 lowercase keys from 43,234 names; 3081 groups; 3083 names absorbed (examples: [('william dieterle', ['William Dieterle', 'william dieterle']), ('marlene dietrich', ['Marlene Dietrich', 'marlene dietrich']), ('english', ['English', 'english'])]).

### 2wiki (legacy `2wiki_clean`, 65,865 nodes, eval 2000 = `val`)

- **Generation:** EVAL-POOL CONTEXT UNION (SAMPLE-CONDITIONED): data/raw/2wiki.jsonl (FlashRAG 2wikimultihopqa/train.jsonl, a 15,000-question sample of official train; file no longer on disk, URL at src/pipeline/loaders.py:669) -> loaders.load_2wiki() (347-407): one doc per context paragraph per question (2wiki_doc_{qid}_{i}, content=' '.join(sentences), 386-387) -> build_clean('2wiki') dedup (title, md5) -> 65,865 nodes from 150,000 records. The 15,000 legacy questions were matched to official train by exact text: 15,000/15,000 unique matches, all in train.
- **Legacy normalization:** none on text (FlashRAG's text is a DE-TOKENIZED variant of the official text: 3,575 legacy nodes differ from official by punctuation spacing only, 153 by whitespace only, 0 by content)
- **Legacy dedup:** (title, md5(content)) -> 1,744 titles kept 2+ text variants (1,750 extra nodes) because FlashRAG shipped variant texts for the same title
- **Legacy ids:** 2wiki_clean_doc_<first-seen counter>
- **Depends on eval subset:** True; **stale:** False; **classification:** SAMPLE_CONDITIONED
- **Phase-C (2026-08-23) rule:** scratchpad/build_2wiki_universe.py:42 -- one doc per para_with_hyperlink article curid, id 2wu:<curid>, text=' '.join(sentences).strip()  (the .strip() is the ONLY divergence from canonical_v1's rule; measured impact = 0 records) -> 5,989,847 (cross-check only)
- **canonical_v1 identity:** official Wikipedia curid (record 'id'), exact string; node_id 2wiki:c<curid>; raw records 5,989,847 = nodes 5,989,847 + duplicates 0 + dropped 0
- **Count reconciliation vs clue 5,989,847:** delta 0 -- identical -- _APPROVALS.json target_n 5,989,847 was an EXPECTED value and the archive was streamed to derive it independently; the two agree exactly

- **Legacy -> canonical mapping:** 65,865/65,865 mapped ({'title': 65863, 'title_ambiguous_first_taken': 2}), missing 0, ambiguous 2; canonical nodes matched 64,115; extra canonical nodes 5,925,732 by split provenance {'full_hyperlink_universe': 5925731, 'title_present_but_not_the_mapped_curid': 1}; changed text {'real_text_diff': 3576, 'exact_equal': 62136, 'whitespace_only_diff': 153}

- **Frozen eval subset gold consistency (legacy golds mapped vs canonical golds):** equal 2000/2000, differ 0

- **Eval subset recovery:** legacy 2wiki_clean questions came from the FlashRAG 15k train sample (raw file no longer present); matched to official train/dev/test by EXACT question text; ambiguities resolved by supporting-fact title set == legacy gold title set (identical to build.py::legacy_subset's 2wiki branch) -> 2000/2000 recovered, 0 missed; official split distribution {'train': 2000}

- **Legacy (FlashRAG) text vs official text:** {'punct_or_case_only': 3575, 'exact': 62137, 'whitespace_only': 153}; 1744 legacy titles carried 2+ text variants. All non-exact differences are punctuation spacing / whitespace (official text keeps tokenizer artefacts, e.g. `Rosenberg( August 11`).

- **Phase-C divergence (MEASURED, not assumed):** `.strip()` changes the text for **0** records; total differing encoder inputs **87,765**. Reference store: `data/canonical/2wiki_universe/documents.jsonl (5,989,847 rows, dense+SPLADE COMPLETE)`. This is the exact upper bound on dense/SPLADE rows needing a re-encode.

- **SUPERSESSION:** replaces the 398,354-node pooled question-context union of train/dev/test 'context' fields (corpus_hash `ec7fbbe8cd783040`); superseded artifacts PRESERVED at `data/final_canonical/2wiki/_superseded_context_union_398354/`.

### musique (legacy `musique_clean`, 13,672 nodes, eval 2000 = `val`)

- **Generation:** GOLD-ONLY UNION OF THE TRAIN POOL (SAMPLE- and GOLD-CONDITIONED): data/raw/musique.jsonl (FlashRAG musique/train.jsonl, 19,938 rows = official train, line i == official train line i, verified by question text 19,938/19,938) -> loaders.load_musique() (176-251): the FlashRAG record has no 'paragraphs', so only metadata.question_decomposition[*].support_paragraph (the GOLD paragraphs) are loaded (198-212, is_supporting forced True at 211) -> build_clean('musique') (title, md5) dedup -> 13,672 nodes. Titles were initially dropped and later recovered from the same FlashRAG dump (src/pipeline/recover_musique_titles.py:59-111, baked 125-129; data/processed/musique_title_map.json).
- **Legacy normalization:** none on text
- **Legacy dedup:** (title, md5(content))
- **Legacy ids:** musique_clean_doc_<first-seen counter>
- **Depends on eval subset:** True; **stale:** False; **classification:** SAMPLE_CONDITIONED + GOLD_CONDITIONED (all-gold pool, no distractors, no dev/test paragraphs)
- **Phase-C (2026-08-23) rule:** scratchpad/c1c2_musique.py:11-21 -- union of all paragraphs of train/dev/test, dedup by sha256(paragraph_text), id musique_{sha[:16]} -> 117,533 (cross-check only)
- **canonical_v1 identity:** (title, paragraph_text) exact; node_id musique:<sha256(title+US+text)[:24]>; raw records 496,194 = nodes 117,534 + duplicates 378,660 + dropped 0
- **Count reconciliation vs clue 117,533:** delta 1 -- +1: (title,text) identity keeps the one paragraph text shared by two titles as two nodes; text-hash identity would give 117,533

- **Legacy -> canonical mapping:** 13,672/13,672 mapped ({'title+text': 13672}), missing 0, ambiguous 0; canonical nodes matched 13,672; extra canonical nodes 103,862 by split provenance {'train': 65694, 'test': 15572, 'train+test': 2461, 'dev': 11757, 'dev+test': 5646, 'train+dev': 1828, 'train+dev+test': 904}; changed text {'exact_equal': 13672}

- **Frozen eval subset gold consistency (legacy golds mapped vs canonical golds):** equal 2000/2000, differ 0

- **Eval subset recovery:** legacy id musique_clean_q_train_{i} (FlashRAG train_i) = official musique_ans_v1.0_train.jsonl line i; question text verified equal -> 2000/2000 recovered, 0 missed; official split distribution {'train': 2000}

### squad (legacy `squad_clean`, 19,029 nodes, eval 2000 = `val`)

- **Generation:** OFFICIAL TRAIN CONTEXTS ONLY (SPLIT-CONDITIONED, not eval-sample-conditioned): src/pipeline/rebuild_dataset.py:47-53 rebuild_squad() -> loaders.load_squad(data/raw/squad_v2.json) (13-64; file is sha-identical to official train-v2.0.json): one doc per paragraph (squad_{chunk}, 26), one question node per qa (squad_q_{counter}, 43) -> build_clean('squad', master=raw) (title, md5) dedup -> 19,029 nodes from 19,035 paragraphs. Official dev contexts (1,204) were never loaded.
- **Legacy normalization:** none on text
- **Legacy dedup:** (title, md5(content)); 6 exact duplicate paragraphs collapsed
- **Legacy ids:** squad_clean_doc_<first-seen counter>
- **Depends on eval subset:** False; **stale:** False; **classification:** OTHER: SPLIT_CONDITIONED_TRAIN_ONLY
- **Phase-C (2026-08-23) rule:** scratchpad/c1c2_squad.py:11-21 -- union of train+dev contexts, dedup by sha256(context), id squad_{sha[:16]} -> 20,233 (cross-check only)
- **canonical_v1 identity:** (article title, context) exact; node_id squad:<sha256(title+US+context)[:24]>; raw records 20,239 = nodes 20,233 + duplicates 6 + dropped 0
- **Count reconciliation vs clue 20,233:** delta 0 -- identical

- **Legacy -> canonical mapping:** 19,029/19,029 mapped ({'title+text': 19029}), missing 0, ambiguous 0; canonical nodes matched 19,029; extra canonical nodes 1,204 by split provenance {'dev': 1204}; changed text {'exact_equal': 19029}

- **Frozen eval subset gold consistency (legacy golds mapped vs canonical golds):** equal 2000/2000, differ 0

- **Eval subset recovery:** legacy id squad_clean_q_{counter}; counter reproduced over official train-v2.0.json qas in file order; question text verified equal -> 2000/2000 recovered, 0 missed; official split distribution {'train': 2000}

### hotpotqa (legacy `hotpotqa_clean`, 507,494 nodes, eval 2000 = `val`)

- **Generation:** DISTRACTOR-SETTING CONTEXT UNION (SETTING- and SAMPLE-CONDITIONED): src/pipeline/rebuild_dataset.py:70-84 rebuild_hotpotqa merges data/raw/review_public/hotpot_{dev,train}_distractor.jsonl -> loaders.load_hotpotqa (title-keyed article_cache, lines 91-134) -> build_clean('hotpotqa') (title, md5) dedup -> 507,494 nodes, i.e. exactly the 10 paragraphs shown to the model per question, deduplicated over the 97,852-question pool. The evaluated 2,000 are a seeded cut of that same pool (internal 70/20/10 re-split: 68,496/19,570/9,786).
- **Legacy normalization:** none on text (4,410 of the 507,494 legacy nodes differ from the official abstract by whitespace only; 503,084 are byte-exact)
- **Legacy dedup:** (title, md5(content)); titles are the legacy identity, which is why the legacy table cannot represent two articles sharing a title
- **Legacy ids:** hotpotqa_clean_doc_<first-seen counter>
- **Depends on eval subset:** True; **stale:** False; **classification:** SETTING_CONDITIONED (distractor contexts, not fullwiki) + SAMPLE_CONDITIONED
- **Phase-C (2026-08-23) rule:** scratchpad/c1_hotpot_fullwiki.py:38-39 -- one doc per abstract curid, id hotpot_<curid>, body=flat_text(text or text_with_links).strip()  (TWO divergences from canonical_v1: the .strip() and the text_with_links fallback; both measured, 1,594 and 94 records) -> 5,233,329 (cross-check only)
- **canonical_v1 identity:** official Wikipedia curid (record 'id'), exact string; node_id hotpotqa:c<curid>; raw records 5,233,329 = nodes 5,233,329 + duplicates 0 + dropped 0
- **Count reconciliation vs clue 5,233,329:** delta 0 -- identical -- _APPROVALS.json target_n 5,233,329 was an EXPECTED value and the tarball was streamed to derive it independently; the two agree exactly

- **Legacy -> canonical mapping:** 507,494/507,494 mapped ({'title': 507494}), missing 0, ambiguous 0; canonical nodes matched 507,494; extra canonical nodes 4,725,835 by split provenance {'fullwiki_corpus': 4725835}; changed text {'exact_equal': 503084, 'whitespace_only_diff': 4410}

- **Frozen eval subset gold consistency (legacy golds mapped vs canonical golds):** equal 2000/2000, differ 0

- **Eval subset recovery:** legacy id 'hotpot_q_<official_question_id>' -> official query id by exact prefix strip; question text verified against data/final_canonical/_work/legacy/hotpotqa_clean/questions.jsonl for 2000/2000 ids (0 mismatches) -> 2000/2000 recovered, 0 missed; official split distribution {'train': 1842, 'validation': 158}

- **Phase-C divergence (MEASURED, not assumed):** `.strip()` changes the text for **1,594** records; Phase-C fell back to `text_with_links` for **94**; total differing encoder inputs **1,688**. Reference store: `data/canonical/hotpotqa/documents.jsonl`. This is the exact upper bound on dense/SPLADE rows needing a re-encode.


## Common legacy plumbing (citations)

- **legacy_cache_node_space:** scratchpad/_ta_prepartition.py:49-113 load_topology(): docs = non-question nodes of data/processed/master_nodes_{ds}.json (lines 64-73; falls back to master_nodes.json only if the per-dataset file is absent -- it is present for all four), row order = file order; partition_map = scratchpad/ablation_qwen/{ds}/variant_C/partition_map.json (line 65)
- **legacy_eval_rows:** cache 'rows' index data/ukb_storage/{ds}/gte_qwen/query_ids_all.json['ids'] (scratchpad/_ta_comb.py:75-92, _ta_resid.py:63-76): rows = split_indices['val'] (metaqa: hop-balanced 666/666/666 with np.random.seed(0); others: shuffled, first 2000, sorted); query_ids_all.json written by scratchpad/phase1_l1_allq.py:13-69 get_all_queries() (question ids sorted as strings, line 36)
- **legacy_split_rule:** src/experiments/overlap_retrain.py:171-206 _splits(): official split metadata is used only when every question carries one (metaqa: 329,282/39,138/39,093 = official train/dev/test); otherwise seeded shuffle random.Random(42) with 70/20/10 (line 39) -- so the legacy 'val' of 2wiki_clean / musique_clean / squad_clean is a random 20% of OFFICIAL TRAIN questions, not the official dev split
- **clean_semantics:** src/pipeline/build_clean.py:121-181 build_clean(): dedup docs by (metadata.title, md5(content)) (131-134), ids {ds}_clean_doc_{n} in first-seen order (135), doc->question back-edges removed (docs start with neighbors=[] then label-free title-mention edges, 147), question ids {ds}_q_* -> {ds}_clean_q_* (164), questions with no mapped gold dropped (155-157)

## `_clean` provenance

- **_clean_suffix:** UKB-era (src/experiments) source name produced by src/pipeline/build_clean.py from a raw per-source master; label-free doc graph + deduped docs
- **squad_clean:** rebuilt label-free after the 2026-08 audit found loaders.load_squad back-linked the doc chain to nodes[-1] (a QUESTION node), giving ~14% of questions a spurious 2nd gold and putting 130,319 doc->question back-edges in the doc graph (src/pipeline/rebuild_dataset.py:5-9, loaders.py:23,34-36). The clean master contains 19,029 document nodes + 130,319 question nodes; question nodes are eval labels, not corpus. canonical_v1 tables documents only.
- **musique_clean:** build_clean over the FlashRAG gold-only dump; titles recovered afterwards (recover_musique_titles.py) -> 13,672/13,672 legacy docs carry the official title (verified: every legacy doc matched canonical_v1 by (title, text)).
- **2wiki_clean:** build_clean over the FlashRAG 15k-train dump; audited clean (no bridge edges), never rebuilt.
- **metaqa:** no _clean variant; master_nodes_metaqa.json (2026-08-30) is the loaders.load_metaqa output restricted to source==metaqa (40,151 docs + 407,513 questions).

## BLOCKED (status record only; nothing built, and the builder refuses unconditionally)

- **webqsp: BLOCKED_PENDING_FREEBASE_SOURCE** — legacy 781,485 = Freebase entity nodes of the RoG-webqsp per-question subgraphs of the RoG TEST split ONLY. Verified: src/pipeline/loader_webqsp.py builds master_nodes_webqsp.js... | REJECTED candidate 1,316,466 (data/canonical/webqsp/documents.jsonl) | SAMPLE_CONDITIONED (corpus = union of the evaluated questions' own subgraphs; also STALE vs the RoG train+val+test entity union) | refusal: `scratchpad/final_canonical_build/build_kb.py NEVER_BUILD = {'webqsp'} -- unconditional` | proposal: `data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md` | file `data/final_canonical/webqsp/status.json`

## Judgment calls

- musique identity = (title, paragraph_text) exact rather than text-hash: +1 node (117,534 vs 117,533) -- one paragraph text appears under two article titles ('Adei Ad', 'Etz Efraim'); benchmark records are identified by title+text, and merging would make one gold refer to two articles.
- 2wiki CORPUS REPLACED (2026-09-05): the first canonical_v1 2wiki table -- the 398,354-node pooled question-context union over all three official splits, corpus_hash ec7fbbe8cd783040... -- was REJECTED and replaced by the full official article universe (para_with_hyperlink.jsonl, 5,989,847 articles, node_id 2wiki:c<curid>). Reason: although invariant to the EVALUATED subset, its membership was still a function of the question set, which is the same defect for which HotpotQA's 507,494-node distractor-context union was rejected; the previously recorded 2wiki-vs-hotpot asymmetry is resolved in favour of the full universe. The superseded artifacts are PRESERVED, not deleted, at data/final_canonical/2wiki/_superseded_context_union_398354/ -- the old node table is required to remap the 398,354 already-encoded embedding rows. Consequence: every 2wiki downstream artifact (KNN, graphs, H4, partitions, router/halo/SP1 caches) is INVALID; none was rebuilt (see DOWNSTREAM_REBUILD_POLICY.md).
- 2wiki + hotpotqa node ids are READABLE (2wiki:c<curid>, hotpotqa:c<curid>) rather than hashed like musique/squad. The curid is already a source-stable unique identity, so hashing buys no collision safety while costing debuggability and remapping. Both were fixed BEFORE any node was written, since node_id determines file order.
- 2wiki stored text = OFFICIAL text verbatim (including its tokenizer artefacts such as 'Rosenberg( August 11'); the legacy FlashRAG text was a de-tokenized variant. No detokenization is applied (lossy, heuristic); downstream encoders may normalize at encoding time.
- hotpotqa corpus = the enwiki-20171001 ABSTRACTS archive (1.55 GB), not the 7.4 GB withlinks-processed archive. 'Full corpus' means the benchmark's full official RETRIEVAL universe: FullWiki indexes one document per article abstract, so the larger archive is a broader collection with a different retrieval unit (article body instead of abstract), i.e. a different task, not a more complete version of this one.
- Neither 2wiki nor hotpotqa applies .strip() to the joined sentence list, although both pre-existing Phase-C extractions did. The divergence was MEASURED rather than assumed: 2wiki 0 records, hotpotqa 1,594 records (+94 where Phase-C fell back to text_with_links, which canonical_v1 never does; under TEXTUALIZATION_REV 2 those 94 carry the source-native title). Those counts bound the rows a .strip() DIVERGENCE makes stale; they are NOT the whole re-encode requirement, because TEXTUALIZATION_REV 2 is a second and independent source of divergence -- it is why 2wiki's requirement is 87,764 dense rows despite its strip divergence being 0. The realised requirement is measured in ARTIFACT_REUSE_REPORT.md.
- metaqa identity = official entity index with exact names; the legacy lowercase collapse (3,083 merged entities) is NOT reproduced. Consequence: 46/1,998 frozen eval queries have gold sets that differ from legacy (exact answer strings resolve to the exact-case entity).
- squad identity = (title, context) exact; identical to text-hash identity here (0 contexts shared across titles).
- No stored-text normalization anywhere (no NFC/NFKC/casefold/whitespace collapse); the lossy alternatives are quantified per dataset in integrity_report.corpus_accounting_detail.identity_normalization_diagnostics (extra identity collapses each would cause): metaqa: NFC 0, whitespace 0, NFC+ws+casefold 3083; 2wiki: NFC 0, whitespace 0, NFC+ws+casefold 0; musique: NFC 0, whitespace 6, NFC+ws+casefold 14; squad: NFC 0, whitespace 0, NFC+ws+casefold 0; hotpotqa: NFC 0, whitespace 0, NFC+ws+casefold 0.
- Eval-subset recovery: metaqa/squad by reproducing the legacy global counters over the official files (question text verified equal), musique by line index (FlashRAG train_i == official line i, verified), 2wiki by exact question text (15,000/15,000 unique). 0 misses on all four.
- webqsp is BLOCKED, not deferred: no corpus was built and the builder refuses unconditionally (NEVER_BUILD). The RoG all-question subgraph union is REJECTED as query-derived, and its 1,316,466 figure additionally came from a builder that injected answer and topic entities (scratchpad/c1c2_webqsp.py:30-33). No Freebase dump exists anywhere on this machine (exhaustive search recorded in webqsp/status.json). See data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md. Its legacy substrate = RoG TEST-split subgraph union (not train+val as an earlier brief suggested).
- hotpotqa_clean is a LEGACY SUBSTRATE ALIAS, not a dataset. Its deferral record was migrated (moved, not deleted) to data/final_canonical/hotpotqa/_legacy_alias_hotpotqa_clean/status.json so exactly one live directory exists per dataset.
- _APPROVALS.json arithmetic: its expected_final_corpus_table lists metaqa 43,234 + 2wiki 5,989,847 + musique 117,534 + squad 20,233 + hotpotqa 5,233,329, which sums to 11,404,177, but records _subtotal_excluding_webqsp = 11,403,177 (1,000 low). The per-dataset targets are the authority and every one of them was met exactly; the subtotal field is simply wrong and was NOT edited, because _APPROVALS.json is the user's authority file.
- 2wiki all_eval_gold_nodes_present = FALSE is an AMBIGUITY-EXPANSION artefact, not a coverage gap, and the flag was deliberately NOT redefined to hide it. Measured in dataset_manifest.json -> gold_shortfall_diagnostic: all 8 affected questions are the ones whose supporting-fact title 'Unconquered' matches TWO curids (2wiki:c62717110, 2wiki:c8073502), so len(gold_node_ids) = len(gold_refs)+1 and the strict equality test fails; both candidates are kept rather than one silently dropped. unresolved_gold_refs.counts is empty and n_any_gold_resolved == n on every split, i.e. 0 gold references are missing from the corpus. This single flag was also why READY_FOR_INDEX_BUILD was false for 2wiki and true for the other four. A reviewer may decide the flag should count unresolved refs instead of exact cardinality; that is a definition change and was not made unilaterally. [RESOLVED AT SOURCE 2026-09-05, restated 2026-09-06: the fix was not a redefinition of the flag but a change to the builder, which now RESOLVES each ambiguous gold reference from the question's own official context paragraph. All 8 'Unconquered' question-title pairs resolve uniquely to 2wiki:c8073502 (the 1947 DeMille film) by source-context prefix agreement, 578 chars against 15, so AMBIGUOUS_GOLD_REFS went 8 -> 0 and ALL_EVAL_GOLDS_PRESENT is now true. The disjunctive semantics are retained for any future genuinely ambiguous reference. Re-verified against the TEXTUALIZATION_REV 2 node table on 2026-09-06: GOLD_REFS_TOTAL 434,824, MISSING_GOLD_REFS 0, AMBIGUOUS_GOLD_REFS 0, ALL_GOLD_REFS_RESOLVE true. READY_FOR_INDEX_BUILD is recomputed by manifest.py from that flag together with the executed query-independence verdict.]
- 2wiki TEXTUALIZATION_REV 2 (2026-09-06): 87,765 of 5,989,847 nodes -- 1.47% of the corpus -- carried EMPTY canonical text, because their source record's sentence list is empty. They now carry their source-native title, under the SAME general source-level rule hotpotqa received a day earlier: if the joined body is empty and the record's own title is non-empty, the title IS the canonical text; no title is prepended anywhere else, and a record with an empty title stays empty and is still KEPT. The decision rested on measurement, not preference: TITLE_NONEMPTY 87,765/87,765, ALT_TEXT_NONEMPTY 0 (the source schema has exactly four keys, so no alternative text field exists), TRULY_TEXTLESS 0, and 20,524 of them (23.4%) are the target of at least one incoming hyperlink -- i.e. a quarter would have entered any KNN/H4 structure over this corpus as degenerate points. Applied BEFORE any such structure exists, which is why it triggered no downstream rebuild. Consequences, all measured: CORPUS_HASH moved 81fa7d1a5d4bbb24... -> 94fe68b8d5f291f1..., while NODE_ORDER_HASH, node count, node ids and node order are unchanged (proved by streaming both tables in lockstep: CHANGED_NODE_COUNT 87,765, fields_that_changed {text, content_hash} only, NODE_ID_SET_IDENTICAL true) and every one of the 5,989,847 content_hash values was independently recomputed from the installed table with 0 mismatches. Queries and gold are untouched: all four query files are byte-identical across revisions and the legacy comparison re-derived to the same numbers. The rev1 artifact set is preserved whole at data/final_canonical/_superseded_textualization_rev1/2wiki/; see its SUPERSEDED.md for the full hash history.
- 2wiki's re-encode requirement under TEXTUALIZATION_REV 2 is 87,764 dense rows and 87,763 SPLADE rows (1.465% of the corpus on either side), MEASURED by re-running reuse_map_kb.py end-to-end against the rev-2 table after the patched purge removed the rev-1 canonical-side digest caches -- never obtained by adding 87,765 to the previous figure of 0/0, which would have been wrong on both sides. Attribution was measured separately by partitioning the per-node map on the changed-id list: of the 5,902,082 nodes the rule did not touch, 0 need a dense or SPLADE row, so ALL_ENCODE_DEMAND_COMES_FROM_THE_CHANGED_ROWS is true. Two rows escape the bill because a token-identical input already exists elsewhere in the store (1 dense, 2 SPLADE); the SPLADE side saves one more than dense for the same reason hotpotqa's two sides diverged -- gte's BPE preserves case, accents and whitespace while SPLADE's WordPiece does not, so SPLADE's equivalence classes are strictly coarser. In DISTINCT FORWARD PASSES, which is what actually bounds cost, the bill is 87,764 dense and 87,655 SPLADE. No encoder was run.
- Both TEXTUALIZATION_REV 2 builds wrote into a staging directory and were installed by rename, so each integrity_report's eval_subset.file records the BUILD-TIME path (hotpotqa _work/hotpot_rev2/, 2wiki _stage_2wiki_rev2/) rather than the live one. That is a faithful record of the build and matches build_info.command; the files now live at data/final_canonical/<ds>/eval_2000.jsonl and the moves are recorded in the supersession INSTALL_RECORD.json. The build outputs were NOT hand-edited to hide the difference.
- hotpotqa's re-encode requirement under TEXTUALIZATION_REV 2 is 1,688 dense rows and 94 SPLADE rows, both MEASURED against the rev-2 node table (CORPUS_HASH 1b7eeac2...) and then verified per node, never obtained by adding 94 to the previous figure. Under rev 1 it read 1,594 dense / 0 SPLADE: the 94 empty-abstract nodes then stored '' and the matching Phase-C rows also stored '' (text_with_links was empty markup for all 94), so they were token-identical and counted as reusable. Under rev 2 those nodes carry their source-native title, which matches nothing already encoded -- all 94 audited ids come back dense_reusable=false, splade_reusable=false, reuse_source=null. A collision was possible in principle (114,187 nodes corpus-wide have text exactly equal to their own title) and was tested for: 0 of 94 matched. SPLADE is consequently no longer 100% reusable -- the 1,594 .strip() deltas still cost nothing there (uncased WordPiece discards leading/trailing whitespace) but the 94 are a genuine content change that no tokenizer normalization folds away. No encoder was run; the requirement is reported only.
- HARNESS BUG FOUND AND FIXED during finalization: query_lanes.py's BUILDER map still routed 2wiki to build.py after 2wiki's corpus moved to the streaming builder, so the constructive lane rebuild-check rebuilt the SUPERSEDED 398,354-node context-union corpus and reported a spurious FAIL (ec7fbbe8cd783040). The map was corrected to build_kb.py and the check re-run: PASS at 81fa7d1a5d4bbb24 (the rev1 hash -- TEXTUALIZATION_REV 2 deliberately moved it on 2026-09-06; the live value is in build_info.json), 5,989,847 nodes. build.py itself was NOT modified (its pinned sha256 is unchanged) and no canonical artifact was built by the wrong builder -- only the throwaway _work/lanecheck/ directory, which is deleted by the check itself. Recorded because a stale copy of that FAIL may exist in any artifact snapshot taken between 15:44 and 16:30 on 2026-09-05.

## Official source hashes (all verified against results/data_audit/dataset_manifest.json)

- metaqa: 20 files hashed; mismatches: none
- 2wiki: 4 files hashed; mismatches: none
- musique: 4 files hashed; mismatches: none
- squad: 2 files hashed; mismatches: none
- hotpotqa: 5 files hashed; mismatches: none

## Artifacts

- `data/final_canonical/ARTIFACT_REUSE_REPORT.md`
- `data/final_canonical/CANONICALIZATION_AUDIT.json`
- `data/final_canonical/CANONICALIZATION_AUDIT.md`
- `data/final_canonical/DOWNSTREAM_REBUILD_POLICY.md`
- `data/final_canonical/MANIFEST.json`
- `data/final_canonical/SOURCE_CONTRACTS_FOR_REVIEW.md`
- `data/final_canonical/STALE_HASH_SWEEP.json`
- `data/final_canonical/_APPROVALS.json`
- `data/final_canonical/_APPROVAL_CRITERIA.md`
- `data/final_canonical/_GATE_HOTPOT_HEAVY_OK`
- `data/final_canonical/_WEBQSP_ACCEPTANCE_GATE.json`
- `data/final_canonical/_build.log`
- `data/final_canonical/_work_phase0_2wiki.log`
- `data/final_canonical/2wiki/EMPTY_TEXT_AUDIT.jsonl`
- `data/final_canonical/2wiki/EMPTY_TEXT_AUDIT.md`
- `data/final_canonical/2wiki/EMPTY_TEXT_SUMMARY.json`
- `data/final_canonical/2wiki/REV2_CHANGED_NODE_IDS.txt`
- `data/final_canonical/2wiki/REV2_DIFF_AUDIT.json`
- `data/final_canonical/2wiki/REV2_REUSE_ATTRIBUTION.json`
- `data/final_canonical/2wiki/REV2_VERIFY.json`
- `data/final_canonical/2wiki/SOURCE_CONTRACT.json`
- `data/final_canonical/2wiki/build_info.json`
- `data/final_canonical/2wiki/dataset_manifest.json`
- `data/final_canonical/2wiki/eval_2000.jsonl`
- `data/final_canonical/2wiki/integrity_report.json`
- `data/final_canonical/2wiki/legacy_comparison.json`
- `data/final_canonical/2wiki/node_id_map_legacy.json`
- `data/final_canonical/2wiki/nodes.jsonl`
- `data/final_canonical/2wiki/query_independence_test.json`
- `data/final_canonical/2wiki/queries/dev.jsonl`
- `data/final_canonical/2wiki/queries/test.jsonl`
- `data/final_canonical/2wiki/queries/train.jsonl`
- `data/final_canonical/2wiki/queries/lanes/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/2wiki/queries/lanes/QUALITY_LOCKED.jsonl`
- `data/final_canonical/2wiki/queries/lanes/SCALE_ALL.ids.jsonl`
- `data/final_canonical/2wiki/reuse_map/index.json`
- `data/final_canonical/2wiki/reuse_map/shard_000.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_001.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_002.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_003.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_004.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_005.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_006.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_007.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_008.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_009.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_010.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_011.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_012.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_013.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_014.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_015.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_016.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_017.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_018.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_019.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_020.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_021.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_022.jsonl`
- `data/final_canonical/2wiki/reuse_map/shard_023.jsonl`
- `data/final_canonical/2wiki/_superseded_context_union_398354/SOURCE_CONTRACT.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/SUPERSESSION.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/build_info.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/dataset_manifest.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/eval_2000.jsonl`
- `data/final_canonical/2wiki/_superseded_context_union_398354/integrity_report.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/legacy_comparison.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/node_id_map_legacy.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/nodes.jsonl`
- `data/final_canonical/2wiki/_superseded_context_union_398354/query_independence_test.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/queries_lanes_superseded/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/2wiki/_superseded_context_union_398354/queries_lanes_superseded/QUALITY_LOCKED.jsonl`
- `data/final_canonical/2wiki/_superseded_context_union_398354/queries_lanes_superseded/SCALE_ALL.ids.jsonl`
- `data/final_canonical/2wiki/_superseded_context_union_398354/reuse_map/index.json`
- `data/final_canonical/2wiki/_superseded_context_union_398354/reuse_map/shard_000.jsonl`
- `data/final_canonical/2wiki/_superseded_context_union_398354/reuse_map/shard_001.jsonl`
- `data/final_canonical/hotpotqa/EMPTY_TEXT_94_AUDIT.jsonl`
- `data/final_canonical/hotpotqa/EMPTY_TEXT_94_AUDIT.md`
- `data/final_canonical/hotpotqa/EMPTY_TEXT_94_SUMMARY.json`
- `data/final_canonical/hotpotqa/REV2_DIFF_AUDIT.json`
- `data/final_canonical/hotpotqa/REV2_VERIFY.json`
- `data/final_canonical/hotpotqa/SOURCE_CONTRACT.json`
- `data/final_canonical/hotpotqa/TEXTUALIZATION_REV2_REPORT.md`
- `data/final_canonical/hotpotqa/build_info.json`
- `data/final_canonical/hotpotqa/dataset_manifest.json`
- `data/final_canonical/hotpotqa/eval_2000.jsonl`
- `data/final_canonical/hotpotqa/integrity_report.json`
- `data/final_canonical/hotpotqa/legacy_comparison.json`
- `data/final_canonical/hotpotqa/node_id_map_legacy.json`
- `data/final_canonical/hotpotqa/nodes.jsonl`
- `data/final_canonical/hotpotqa/query_independence_test.json`
- `data/final_canonical/hotpotqa/queries/test.jsonl`
- `data/final_canonical/hotpotqa/queries/train.jsonl`
- `data/final_canonical/hotpotqa/queries/validation.jsonl`
- `data/final_canonical/hotpotqa/queries/lanes/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/hotpotqa/queries/lanes/QUALITY_LOCKED.jsonl`
- `data/final_canonical/hotpotqa/queries/lanes/SCALE_ALL.ids.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/index.json`
- `data/final_canonical/hotpotqa/reuse_map/shard_000.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_001.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_002.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_003.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_004.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_005.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_006.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_007.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_008.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_009.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_010.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_011.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_012.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_013.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_014.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_015.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_016.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_017.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_018.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_019.jsonl`
- `data/final_canonical/hotpotqa/reuse_map/shard_020.jsonl`
- `data/final_canonical/hotpotqa/_legacy_alias_hotpotqa_clean/ALIAS.json`
- `data/final_canonical/hotpotqa/_legacy_alias_hotpotqa_clean/status.json`
- `data/final_canonical/metaqa/SOURCE_CONTRACT.json`
- `data/final_canonical/metaqa/build_info.json`
- `data/final_canonical/metaqa/dataset_manifest.json`
- `data/final_canonical/metaqa/eval_1998.jsonl`
- `data/final_canonical/metaqa/integrity_report.json`
- `data/final_canonical/metaqa/legacy_comparison.json`
- `data/final_canonical/metaqa/node_id_map_legacy.json`
- `data/final_canonical/metaqa/nodes.jsonl`
- `data/final_canonical/metaqa/query_independence_test.json`
- `data/final_canonical/metaqa/queries/dev.jsonl`
- `data/final_canonical/metaqa/queries/test.jsonl`
- `data/final_canonical/metaqa/queries/train.jsonl`
- `data/final_canonical/metaqa/queries/lanes/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/metaqa/queries/lanes/QUALITY_LOCKED.jsonl`
- `data/final_canonical/metaqa/queries/lanes/SCALE_ALL.ids.jsonl`
- `data/final_canonical/metaqa/reuse_map/index.json`
- `data/final_canonical/metaqa/reuse_map/shard_000.jsonl`
- `data/final_canonical/musique/SOURCE_CONTRACT.json`
- `data/final_canonical/musique/build_info.json`
- `data/final_canonical/musique/dataset_manifest.json`
- `data/final_canonical/musique/eval_2000.jsonl`
- `data/final_canonical/musique/integrity_report.json`
- `data/final_canonical/musique/legacy_comparison.json`
- `data/final_canonical/musique/node_id_map_legacy.json`
- `data/final_canonical/musique/nodes.jsonl`
- `data/final_canonical/musique/query_independence_test.json`
- `data/final_canonical/musique/queries/dev.jsonl`
- `data/final_canonical/musique/queries/test.jsonl`
- `data/final_canonical/musique/queries/train.jsonl`
- `data/final_canonical/musique/queries/lanes/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/musique/queries/lanes/QUALITY_LOCKED.jsonl`
- `data/final_canonical/musique/queries/lanes/SCALE_ALL.ids.jsonl`
- `data/final_canonical/musique/reuse_map/index.json`
- `data/final_canonical/musique/reuse_map/shard_000.jsonl`
- `data/final_canonical/squad/SOURCE_CONTRACT.json`
- `data/final_canonical/squad/build_info.json`
- `data/final_canonical/squad/dataset_manifest.json`
- `data/final_canonical/squad/eval_2000.jsonl`
- `data/final_canonical/squad/integrity_report.json`
- `data/final_canonical/squad/legacy_comparison.json`
- `data/final_canonical/squad/node_id_map_legacy.json`
- `data/final_canonical/squad/nodes.jsonl`
- `data/final_canonical/squad/query_independence_test.json`
- `data/final_canonical/squad/queries/dev.jsonl`
- `data/final_canonical/squad/queries/train.jsonl`
- `data/final_canonical/squad/queries/lanes/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/squad/queries/lanes/QUALITY_LOCKED.jsonl`
- `data/final_canonical/squad/queries/lanes/SCALE_ALL.ids.jsonl`
- `data/final_canonical/squad/reuse_map/index.json`
- `data/final_canonical/squad/reuse_map/shard_000.jsonl`
- `data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md`
- `data/final_canonical/webqsp/ROG_CWQ_SOURCE_PROVENANCE.json`
- `data/final_canonical/webqsp/ROG_WEBQSP_SOURCE_PROVENANCE.json`
- `data/final_canonical/webqsp/SOURCE_CONTRACT.json`
- `data/final_canonical/webqsp/WEBQSP_FREEBASE_PIPELINE_AUDIT.md`
- `data/final_canonical/webqsp/WEBQSP_REPRESENTATION_AUDIT.md`
- `data/final_canonical/webqsp/WEBQSP_SOURCE_LINEAGE_MATRIX.md`
- `data/final_canonical/webqsp/WEBQSP_SOURCE_MATRIX.json`
- `data/final_canonical/webqsp/status.json`
- `data/final_canonical/_superseded_sidecars/GOLD_SEMANTICS_preSourceFix.json`
- `data/final_canonical/_superseded_sidecars/SUPERSESSION.json`
- `data/final_canonical/_superseded_textualization_rev1/SUPERSEDED.md`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/INSTALL_RECORD.json`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/build_info.json`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/dataset_manifest.json`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/eval_2000.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/integrity_report.json`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/nodes.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/query_independence_test.json`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/queries/dev.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/queries/test.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/queries/train.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/queries/lanes/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/queries/lanes/QUALITY_LOCKED.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/queries/lanes/SCALE_ALL.ids.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/index.json`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_000.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_001.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_002.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_003.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_004.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_005.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_006.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_007.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_008.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_009.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_010.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_011.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_012.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_013.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_014.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_015.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_016.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_017.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_018.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_019.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_020.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_021.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_022.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/2wiki/reuse_map/shard_023.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/SOURCE_CONTRACT.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/build_info.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/dataset_manifest.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/eval_2000.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/integrity_report.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/legacy_comparison.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/node_id_map_legacy.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/nodes.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/query_independence_test.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/queries/test.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/queries/train.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/queries/validation.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/queries/lanes/LEGACY_CONTINUITY.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/queries/lanes/QUALITY_LOCKED.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/queries/lanes/SCALE_ALL.ids.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/index.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_000.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_001.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_002.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_003.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_004.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_005.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_006.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_007.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_008.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_009.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_010.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_011.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_012.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_013.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_014.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_015.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_016.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_017.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_018.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_019.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/reuse_map/shard_020.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/_legacy_alias_hotpotqa_clean/ALIAS.json`
- `data/final_canonical/_superseded_textualization_rev1/hotpotqa/_legacy_alias_hotpotqa_clean/status.json`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/query_independence_test.json`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/A/build_info.json`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/A/eval_2000.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/A/integrity_report.json`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/B/build_info.json`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/B/eval_2000_random_validation_s7.jsonl`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/B/integrity_report.json`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/none/build_info.json`
- `data/final_canonical/_superseded_textualization_rev1/_qi_rev1/hotpotqa/none/integrity_report.json`
- `data/final_canonical/_superseded_textualization_rev1/_reuse_kb_rev1_canonical_side_caches/spill_todo_0.npz`
- `data/final_canonical/_superseded_textualization_rev1/_reuse_kb_rev1_canonical_side_caches/verify_0.npz`
- `data/final_canonical/_superseded_textualization_rev1/_reuse_kb_rev1_canonical_side_caches/verify_10000.npz`
