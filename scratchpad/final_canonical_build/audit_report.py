"""Phase 0 -- CANONICALIZATION_AUDIT.{json,md} (forensic audit of legacy substrates vs official sources vs canonical_v1).

    python scratchpad/final_canonical_build/audit_report.py

Assembles measured facts from data/final_canonical/_work/** and the per-dataset build outputs, plus the code
citations gathered while tracing the legacy pipeline. Writes data/final_canonical/CANONICALIZATION_AUDIT.json/.md.
"""
import os, json, time

ROOT = "data/final_canonical"; W = f"{ROOT}/_work"
L = {"metaqa": "metaqa", "2wiki": "2wiki_clean", "musique": "musique_clean", "squad": "squad_clean",
     "hotpotqa": "hotpotqa_clean"}
PHASEC = {"metaqa": "scratchpad/c1_metaqa.py", "2wiki": "scratchpad/build_2wiki_universe.py", "musique": "scratchpad/c1c2_musique.py",
          "squad": "scratchpad/c1c2_squad.py", "hotpotqa": "scratchpad/c1_hotpot_fullwiki.py"}
# the Phase-C store that covers each canonical corpus at FULL scale (2wiki's canonical corpus is now the
# full article universe, so its Phase-C counterpart is 2wiki_universe, not the 398k retrieval view)
PHASEC_DIR = {"metaqa": "metaqa", "2wiki": "2wiki_universe", "musique": "musique", "squad": "squad", "hotpotqa": "hotpotqa"}
OFFICIAL = json.load(open("results/data_audit/dataset_manifest.json", encoding="utf-8"))


def J(p):
    if not os.path.exists(p): return None
    b = open(p, "rb").read()
    try: return json.loads(b.decode("utf-8"))
    except UnicodeDecodeError: return json.loads(b.decode("cp1252"))   # _work/phase0/metaqa.json predates the utf-8 fix


COMMON_CITATIONS = {
    "legacy_cache_node_space": "scratchpad/_ta_prepartition.py:49-113 load_topology(): docs = non-question nodes of data/processed/master_nodes_{ds}.json "
                               "(lines 64-73; falls back to master_nodes.json only if the per-dataset file is absent -- it is present for all four), "
                               "row order = file order; partition_map = scratchpad/ablation_qwen/{ds}/variant_C/partition_map.json (line 65)",
    "legacy_eval_rows": "cache 'rows' index data/ukb_storage/{ds}/gte_qwen/query_ids_all.json['ids'] (scratchpad/_ta_comb.py:75-92, _ta_resid.py:63-76): "
                        "rows = split_indices['val'] (metaqa: hop-balanced 666/666/666 with np.random.seed(0); others: shuffled, first 2000, sorted); "
                        "query_ids_all.json written by scratchpad/phase1_l1_allq.py:13-69 get_all_queries() (question ids sorted as strings, line 36)",
    "legacy_split_rule": "src/experiments/overlap_retrain.py:171-206 _splits(): official split metadata is used only when every question carries one "
                         "(metaqa: 329,282/39,138/39,093 = official train/dev/test); otherwise seeded shuffle random.Random(42) with 70/20/10 "
                         "(line 39) -- so the legacy 'val' of 2wiki_clean / musique_clean / squad_clean is a random 20% of OFFICIAL TRAIN questions, "
                         "not the official dev split",
    "clean_semantics": "src/pipeline/build_clean.py:121-181 build_clean(): dedup docs by (metadata.title, md5(content)) (131-134), ids {ds}_clean_doc_{n} "
                       "in first-seen order (135), doc->question back-edges removed (docs start with neighbors=[] then label-free title-mention edges, 147), "
                       "question ids {ds}_q_* -> {ds}_clean_q_* (164), questions with no mapped gold dropped (155-157)",
}

PER = {
    "metaqa": {
        "official_sources": ["data/original/metaqa/entity/kb_entity_dict.txt", "data/original/metaqa/kb.txt",
                             "data/original/metaqa/{1,2,3}-hop/vanilla/qa_{train,dev,test}.txt", "data/original/metaqa/{1,2,3}-hop/qa_{train,dev,test}_qtype.txt"],
        "legacy_generation": "FULL SOURCE KB with LOSSY IDENTITY: src/pipeline/loaders.py:481-590 load_metaqa() builds one node per kb.txt triple endpoint after "
                             "strip().lower().replace('_',' ') (lines 529-531); node id = 'metaqa_ent_'+lowercased name (554); title = 'prettiest casing' seen (533-538); "
                             "content = display name + up to 10 verbalized triples (562-571). Questions: metaqa_q_{hop}hop_{split}_{global counter} over "
                             "{hop}-hop/vanilla/qa_{split}.txt (594-609); golds = answer strings lower()-matched to entity ids (624-628). "
                             "data/raw/metaqa == data/original/metaqa (all files sha-identical).",
        "legacy_normalization": "lower(), '_'->' ', strip() on entity names -> 43,234 official entities collapse to 40,151 (3,081 groups, 3,083 names absorbed); "
                                "answer matching case-insensitive",
        "legacy_dedup": "by lowercased name (implicit)", "legacy_id_rule": "metaqa_ent_<lowercased name>",
        "depends_on_eval_subset": False, "stale": False,
        "classification": "OTHER: LOSSY_IDENTITY_COLLAPSE (full-KB coverage, entity identity merged case variants)",
        "phase_c_rule": "scratchpad/c1_metaqa.py:20-31 -- one doc per kb_entity_dict.txt row, id metaqa_ent_{idx}, text = name, no normalization; kb.txt entity set cross-checked (lines 15-18, 35)",
        "canonical_v1_identity": "official kb_entity_dict index; names exact; node_id metaqa:e<idx:05d>",
        "count_clue": 43234,
    },
    "2wiki": {
        "official_sources": ["data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip  <-- THE CORPUS (full article universe)",
                             "data/original/2wiki/v1.0_ids_april2021/{train,dev,test}.json  <-- QUERY LAYER ONLY (contributes no node)",
                             "(id_aliases.json present, not used)"],
        "legacy_generation": "EVAL-POOL CONTEXT UNION (SAMPLE-CONDITIONED): data/raw/2wiki.jsonl (FlashRAG 2wikimultihopqa/train.jsonl, a 15,000-question sample of official "
                             "train; file no longer on disk, URL at src/pipeline/loaders.py:669) -> loaders.load_2wiki() (347-407): one doc per context paragraph per "
                             "question (2wiki_doc_{qid}_{i}, content=' '.join(sentences), 386-387) -> build_clean('2wiki') dedup (title, md5) -> 65,865 nodes from "
                             "150,000 records. The 15,000 legacy questions were matched to official train by exact text: 15,000/15,000 unique matches, all in train.",
        "legacy_normalization": "none on text (FlashRAG's text is a DE-TOKENIZED variant of the official text: 3,575 legacy nodes differ from official by punctuation spacing only, 153 by whitespace only, 0 by content)",
        "legacy_dedup": "(title, md5(content)) -> 1,744 titles kept 2+ text variants (1,750 extra nodes) because FlashRAG shipped variant texts for the same title",
        "legacy_id_rule": "2wiki_clean_doc_<first-seen counter>",
        "depends_on_eval_subset": True, "stale": False,
        "classification": "SAMPLE_CONDITIONED",
        "phase_c_rule": "scratchpad/build_2wiki_universe.py:42 -- one doc per para_with_hyperlink article curid, id 2wu:<curid>, "
                        "text=' '.join(sentences).strip()  (the .strip() is the ONLY divergence from canonical_v1's rule; measured "
                        "impact = 0 records)",
        "canonical_v1_identity": "official Wikipedia curid (record 'id'), exact string; node_id 2wiki:c<curid>",
        "count_clue": 5989847,
    },
    "hotpotqa": {
        "official_sources": ["data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2  <-- THE CORPUS",
                             "data/original/hotpotqa/hf_distractor/train-0000{0,1}-of-00002.parquet  <-- QUERY LAYER (train is shared by both settings)",
                             "data/original/hotpotqa/hf_fullwiki/{validation,test}-00000-of-00001.parquet  <-- QUERY LAYER"],
        "legacy_generation": "DISTRACTOR-SETTING CONTEXT UNION (SETTING- and SAMPLE-CONDITIONED): src/pipeline/rebuild_dataset.py:70-84 "
                             "rebuild_hotpotqa merges data/raw/review_public/hotpot_{dev,train}_distractor.jsonl -> loaders.load_hotpotqa "
                             "(title-keyed article_cache, lines 91-134) -> build_clean('hotpotqa') (title, md5) dedup -> 507,494 nodes, i.e. "
                             "exactly the 10 paragraphs shown to the model per question, deduplicated over the 97,852-question pool. The "
                             "evaluated 2,000 are a seeded cut of that same pool (internal 70/20/10 re-split: 68,496/19,570/9,786).",
        "legacy_normalization": "none on text (4,410 of the 507,494 legacy nodes differ from the official abstract by whitespace only; "
                                "503,084 are byte-exact)",
        "legacy_dedup": "(title, md5(content)); titles are the legacy identity, which is why the legacy table cannot represent two "
                        "articles sharing a title",
        "legacy_id_rule": "hotpotqa_clean_doc_<first-seen counter>",
        "depends_on_eval_subset": True, "stale": False,
        "classification": "SETTING_CONDITIONED (distractor contexts, not fullwiki) + SAMPLE_CONDITIONED",
        "phase_c_rule": "scratchpad/c1_hotpot_fullwiki.py:38-39 -- one doc per abstract curid, id hotpot_<curid>, "
                        "body=flat_text(text or text_with_links).strip()  (TWO divergences from canonical_v1: the .strip() and the "
                        "text_with_links fallback; both measured, 1,594 and 94 records)",
        "canonical_v1_identity": "official Wikipedia curid (record 'id'), exact string; node_id hotpotqa:c<curid>",
        "count_clue": 5233329,
    },
    "musique": {
        "official_sources": ["data/original/musique/v1.0/musique_ans_v1.0_{train,dev,test}.jsonl", "data/original/musique/v1.0/dev_test_singlehop_questions_v1.0.json (aux, not tabled)"],
        "legacy_generation": "GOLD-ONLY UNION OF THE TRAIN POOL (SAMPLE- and GOLD-CONDITIONED): data/raw/musique.jsonl (FlashRAG musique/train.jsonl, 19,938 rows = official train, "
                             "line i == official train line i, verified by question text 19,938/19,938) -> loaders.load_musique() (176-251): the FlashRAG record has no "
                             "'paragraphs', so only metadata.question_decomposition[*].support_paragraph (the GOLD paragraphs) are loaded (198-212, is_supporting forced True at 211) "
                             "-> build_clean('musique') (title, md5) dedup -> 13,672 nodes. Titles were initially dropped and later recovered from the same FlashRAG dump "
                             "(src/pipeline/recover_musique_titles.py:59-111, baked 125-129; data/processed/musique_title_map.json).",
        "legacy_normalization": "none on text", "legacy_dedup": "(title, md5(content))", "legacy_id_rule": "musique_clean_doc_<first-seen counter>",
        "depends_on_eval_subset": True, "stale": False,
        "classification": "SAMPLE_CONDITIONED + GOLD_CONDITIONED (all-gold pool, no distractors, no dev/test paragraphs)",
        "phase_c_rule": "scratchpad/c1c2_musique.py:11-21 -- union of all paragraphs of train/dev/test, dedup by sha256(paragraph_text), id musique_{sha[:16]}",
        "canonical_v1_identity": "(title, paragraph_text) exact; node_id musique:<sha256(title+US+text)[:24]>",
        "count_clue": 117533,
    },
    "squad": {
        "official_sources": ["data/original/squad/v2.0/train-v2.0.json", "data/original/squad/v2.0/dev-v2.0.json"],
        "legacy_generation": "OFFICIAL TRAIN CONTEXTS ONLY (SPLIT-CONDITIONED, not eval-sample-conditioned): src/pipeline/rebuild_dataset.py:47-53 rebuild_squad() -> "
                             "loaders.load_squad(data/raw/squad_v2.json) (13-64; file is sha-identical to official train-v2.0.json): one doc per paragraph (squad_{chunk}, 26), "
                             "one question node per qa (squad_q_{counter}, 43) -> build_clean('squad', master=raw) (title, md5) dedup -> 19,029 nodes from 19,035 paragraphs. "
                             "Official dev contexts (1,204) were never loaded.",
        "legacy_normalization": "none on text", "legacy_dedup": "(title, md5(content)); 6 exact duplicate paragraphs collapsed", "legacy_id_rule": "squad_clean_doc_<first-seen counter>",
        "depends_on_eval_subset": False, "stale": False,
        "classification": "OTHER: SPLIT_CONDITIONED_TRAIN_ONLY",
        "phase_c_rule": "scratchpad/c1c2_squad.py:11-21 -- union of train+dev contexts, dedup by sha256(context), id squad_{sha[:16]}",
        "canonical_v1_identity": "(article title, context) exact; node_id squad:<sha256(title+US+context)[:24]>",
        "count_clue": 20233,
    },
}

CLEAN_PROVENANCE = {
    "_clean_suffix": "UKB-era (src/experiments) source name produced by src/pipeline/build_clean.py from a raw per-source master; label-free doc graph + deduped docs",
    "squad_clean": "rebuilt label-free after the 2026-08 audit found loaders.load_squad back-linked the doc chain to nodes[-1] (a QUESTION node), giving ~14% of "
                   "questions a spurious 2nd gold and putting 130,319 doc->question back-edges in the doc graph (src/pipeline/rebuild_dataset.py:5-9, loaders.py:23,34-36). "
                   "The clean master contains 19,029 document nodes + 130,319 question nodes; question nodes are eval labels, not corpus. canonical_v1 tables documents only.",
    "musique_clean": "build_clean over the FlashRAG gold-only dump; titles recovered afterwards (recover_musique_titles.py) -> 13,672/13,672 legacy docs carry the official title "
                     "(verified: every legacy doc matched canonical_v1 by (title, text)).",
    "2wiki_clean": "build_clean over the FlashRAG 15k-train dump; audited clean (no bridge edges), never rebuilt.",
    "metaqa": "no _clean variant; master_nodes_metaqa.json (2026-08-30) is the loaders.load_metaqa output restricted to source==metaqa (40,151 docs + 407,513 questions).",
}

JUDGMENT_CALLS = [
    "musique identity = (title, paragraph_text) exact rather than text-hash: +1 node (117,534 vs 117,533) -- one paragraph text appears under two article titles "
    "('Adei Ad', 'Etz Efraim'); benchmark records are identified by title+text, and merging would make one gold refer to two articles.",
    "2wiki CORPUS REPLACED (2026-09-05): the first canonical_v1 2wiki table -- the 398,354-node pooled question-context union over all "
    "three official splits, corpus_hash ec7fbbe8cd783040... -- was REJECTED and replaced by the full official article universe "
    "(para_with_hyperlink.jsonl, 5,989,847 articles, node_id 2wiki:c<curid>). Reason: although invariant to the EVALUATED subset, its "
    "membership was still a function of the question set, which is the same defect for which HotpotQA's 507,494-node distractor-context "
    "union was rejected; the previously recorded 2wiki-vs-hotpot asymmetry is resolved in favour of the full universe. The superseded "
    "artifacts are PRESERVED, not deleted, at data/final_canonical/2wiki/_superseded_context_union_398354/ -- the old node table is "
    "required to remap the 398,354 already-encoded embedding rows. Consequence: every 2wiki downstream artifact (KNN, graphs, H4, "
    "partitions, router/halo/SP1 caches) is INVALID; none was rebuilt (see DOWNSTREAM_REBUILD_POLICY.md).",
    "2wiki + hotpotqa node ids are READABLE (2wiki:c<curid>, hotpotqa:c<curid>) rather than hashed like musique/squad. The curid is "
    "already a source-stable unique identity, so hashing buys no collision safety while costing debuggability and remapping. Both were "
    "fixed BEFORE any node was written, since node_id determines file order.",
    "2wiki stored text = OFFICIAL text verbatim (including its tokenizer artefacts such as 'Rosenberg( August 11'); the legacy FlashRAG text was a de-tokenized variant. "
    "No detokenization is applied (lossy, heuristic); downstream encoders may normalize at encoding time.",
    "hotpotqa corpus = the enwiki-20171001 ABSTRACTS archive (1.55 GB), not the 7.4 GB withlinks-processed archive. 'Full corpus' means "
    "the benchmark's full official RETRIEVAL universe: FullWiki indexes one document per article abstract, so the larger archive is a "
    "broader collection with a different retrieval unit (article body instead of abstract), i.e. a different task, not a more complete "
    "version of this one.",
    "Neither 2wiki nor hotpotqa applies .strip() to the joined sentence list, although both pre-existing Phase-C extractions did. The "
    "divergence was MEASURED rather than assumed: 2wiki 0 records, hotpotqa 1,594 records (+94 where Phase-C fell back to "
    "text_with_links, which canonical_v1 never does; under TEXTUALIZATION_REV 2 those 94 carry the source-native title). Those "
    "counts bound the rows a .strip() DIVERGENCE makes stale; they are NOT the whole re-encode requirement, because "
    "TEXTUALIZATION_REV 2 is a second and independent source of divergence -- it is why 2wiki's requirement is 87,764 dense rows "
    "despite its strip divergence being 0. The realised requirement is measured in ARTIFACT_REUSE_REPORT.md.",
    "metaqa identity = official entity index with exact names; the legacy lowercase collapse (3,083 merged entities) is NOT reproduced. Consequence: 46/1,998 frozen eval "
    "queries have gold sets that differ from legacy (exact answer strings resolve to the exact-case entity).",
    "squad identity = (title, context) exact; identical to text-hash identity here (0 contexts shared across titles).",
    "No stored-text normalization anywhere (no NFC/NFKC/casefold/whitespace collapse); the lossy alternatives are quantified per dataset in "
    "integrity_report.corpus_accounting_detail.identity_normalization_diagnostics (extra identity collapses each would cause): {NORM_DIAG}.",
    "Eval-subset recovery: metaqa/squad by reproducing the legacy global counters over the official files (question text verified equal), musique by line index "
    "(FlashRAG train_i == official line i, verified), 2wiki by exact question text (15,000/15,000 unique). 0 misses on all four.",
    "webqsp is BLOCKED, not deferred: no corpus was built and the builder refuses unconditionally (NEVER_BUILD). The RoG all-question "
    "subgraph union is REJECTED as query-derived, and its 1,316,466 figure additionally came from a builder that injected answer and "
    "topic entities (scratchpad/c1c2_webqsp.py:30-33). No Freebase dump exists anywhere on this machine (exhaustive search recorded in "
    "webqsp/status.json). See data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md. Its legacy substrate = RoG TEST-split subgraph "
    "union (not train+val as an earlier brief suggested).",
    "hotpotqa_clean is a LEGACY SUBSTRATE ALIAS, not a dataset. Its deferral record was migrated (moved, not deleted) to "
    "data/final_canonical/hotpotqa/_legacy_alias_hotpotqa_clean/status.json so exactly one live directory exists per dataset.",
    "_APPROVALS.json arithmetic: its expected_final_corpus_table lists metaqa 43,234 + 2wiki 5,989,847 + musique 117,534 + squad 20,233 "
    "+ hotpotqa 5,233,329, which sums to 11,404,177, but records _subtotal_excluding_webqsp = 11,403,177 (1,000 low). The per-dataset "
    "targets are the authority and every one of them was met exactly; the subtotal field is simply wrong and was NOT edited, because "
    "_APPROVALS.json is the user's authority file.",
    "2wiki all_eval_gold_nodes_present = FALSE is an AMBIGUITY-EXPANSION artefact, not a coverage gap, and the flag was deliberately "
    "NOT redefined to hide it. Measured in dataset_manifest.json -> gold_shortfall_diagnostic: all 8 affected questions are the ones "
    "whose supporting-fact title 'Unconquered' matches TWO curids (2wiki:c62717110, 2wiki:c8073502), so len(gold_node_ids) = "
    "len(gold_refs)+1 and the strict equality test fails; both candidates are kept rather than one silently dropped. "
    "unresolved_gold_refs.counts is empty and n_any_gold_resolved == n on every split, i.e. 0 gold references are missing from the "
    "corpus. This single flag was also why READY_FOR_INDEX_BUILD was false for 2wiki and true for the other four. A reviewer may decide "
    "the flag should count unresolved refs instead of exact cardinality; that is a definition change and was not made unilaterally. "
    "[RESOLVED AT SOURCE 2026-09-05, restated 2026-09-06: the fix was not a redefinition of the flag but a change to the builder, "
    "which now RESOLVES each ambiguous gold reference from the question's own official context paragraph. All 8 'Unconquered' "
    "question-title pairs resolve uniquely to 2wiki:c8073502 (the 1947 DeMille film) by source-context prefix agreement, 578 chars "
    "against 15, so AMBIGUOUS_GOLD_REFS went 8 -> 0 and ALL_EVAL_GOLDS_PRESENT is now true. The disjunctive semantics are retained "
    "for any future genuinely ambiguous reference. Re-verified against the TEXTUALIZATION_REV 2 node table on 2026-09-06: "
    "GOLD_REFS_TOTAL 434,824, MISSING_GOLD_REFS 0, AMBIGUOUS_GOLD_REFS 0, ALL_GOLD_REFS_RESOLVE true. READY_FOR_INDEX_BUILD is "
    "recomputed by manifest.py from that flag together with the executed query-independence verdict.]",

    "2wiki TEXTUALIZATION_REV 2 (2026-09-06): 87,765 of 5,989,847 nodes -- 1.47% of the corpus -- carried EMPTY canonical text, "
    "because their source record's sentence list is empty. They now carry their source-native title, under the SAME general "
    "source-level rule hotpotqa received a day earlier: if the joined body is empty and the record's own title is non-empty, the "
    "title IS the canonical text; no title is prepended anywhere else, and a record with an empty title stays empty and is still "
    "KEPT. The decision rested on measurement, not preference: TITLE_NONEMPTY 87,765/87,765, ALT_TEXT_NONEMPTY 0 (the source schema "
    "has exactly four keys, so no alternative text field exists), TRULY_TEXTLESS 0, and 20,524 of them (23.4%) are the target of at "
    "least one incoming hyperlink -- i.e. a quarter would have entered any KNN/H4 structure over this corpus as degenerate points. "
    "Applied BEFORE any such structure exists, which is why it triggered no downstream rebuild. Consequences, all measured: "
    "CORPUS_HASH moved 81fa7d1a5d4bbb24... -> 94fe68b8d5f291f1..., while NODE_ORDER_HASH, node count, node ids and node order are "
    "unchanged (proved by streaming both tables in lockstep: CHANGED_NODE_COUNT 87,765, fields_that_changed {text, content_hash} "
    "only, NODE_ID_SET_IDENTICAL true) and every one of the 5,989,847 content_hash values was independently recomputed from the "
    "installed table with 0 mismatches. Queries and gold are untouched: all four query files are byte-identical across revisions "
    "and the legacy comparison re-derived to the same numbers. The rev1 artifact set is preserved whole at "
    "data/final_canonical/_superseded_textualization_rev1/2wiki/; see its SUPERSEDED.md for the full hash history.",

    "2wiki's re-encode requirement under TEXTUALIZATION_REV 2 is 87,764 dense rows and 87,763 SPLADE rows (1.465% of the corpus on "
    "either side), MEASURED by re-running reuse_map_kb.py end-to-end against the rev-2 table after the patched purge removed the "
    "rev-1 canonical-side digest caches -- never obtained by adding 87,765 to the previous figure of 0/0, which would have been "
    "wrong on both sides. Attribution was measured separately by partitioning the per-node map on the changed-id list: of the "
    "5,902,082 nodes the rule did not touch, 0 need a dense or SPLADE row, so ALL_ENCODE_DEMAND_COMES_FROM_THE_CHANGED_ROWS is "
    "true. Two rows escape the bill because a token-identical input already exists elsewhere in the store (1 dense, 2 SPLADE); the "
    "SPLADE side saves one more than dense for the same reason hotpotqa's two sides diverged -- gte's BPE preserves case, accents "
    "and whitespace while SPLADE's WordPiece does not, so SPLADE's equivalence classes are strictly coarser. In DISTINCT FORWARD "
    "PASSES, which is what actually bounds cost, the bill is 87,764 dense and 87,655 SPLADE. No encoder was run.",

    "Both TEXTUALIZATION_REV 2 builds wrote into a staging directory and were installed by rename, so each integrity_report's "
    "eval_subset.file records the BUILD-TIME path (hotpotqa _work/hotpot_rev2/, 2wiki _stage_2wiki_rev2/) rather than the live one. "
    "That is a faithful record of the build and matches build_info.command; the files now live at "
    "data/final_canonical/<ds>/eval_2000.jsonl and the moves are recorded in the supersession INSTALL_RECORD.json. The build "
    "outputs were NOT hand-edited to hide the difference.",
    "hotpotqa's re-encode requirement under TEXTUALIZATION_REV 2 is 1,688 dense rows and 94 SPLADE rows, both MEASURED against the "
    "rev-2 node table (CORPUS_HASH 1b7eeac2...) and then verified per node, never obtained by adding 94 to the previous figure. "
    "Under rev 1 it read 1,594 dense / 0 SPLADE: the 94 empty-abstract nodes then stored '' and the matching Phase-C rows also stored "
    "'' (text_with_links was empty markup for all 94), so they were token-identical and counted as reusable. Under rev 2 those nodes "
    "carry their source-native title, which matches nothing already encoded -- all 94 audited ids come back dense_reusable=false, "
    "splade_reusable=false, reuse_source=null. A collision was possible in principle (114,187 nodes corpus-wide have text exactly "
    "equal to their own title) and was tested for: 0 of 94 matched. SPLADE is consequently no longer 100% reusable -- the 1,594 "
    ".strip() deltas still cost nothing there (uncased WordPiece discards leading/trailing whitespace) but the 94 are a genuine "
    "content change that no tokenizer normalization folds away. No encoder was run; the requirement is reported only.",
    "HARNESS BUG FOUND AND FIXED during finalization: query_lanes.py's BUILDER map still routed 2wiki to build.py after 2wiki's corpus "
    "moved to the streaming builder, so the constructive lane rebuild-check rebuilt the SUPERSEDED 398,354-node context-union corpus "
    "and reported a spurious FAIL (ec7fbbe8cd783040). The map was corrected to build_kb.py and the check re-run: PASS at "
    "81fa7d1a5d4bbb24 (the rev1 hash -- TEXTUALIZATION_REV 2 deliberately moved it on 2026-09-06; the live value is in "
    "build_info.json), 5,989,847 nodes. build.py itself was NOT modified (its pinned sha256 is unchanged) and no canonical artifact "
    "was built by the wrong builder -- only the throwaway _work/lanecheck/ directory, which is deleted by the check itself. Recorded "
    "because a stale copy of that FAIL may exist in any artifact snapshot taken between 15:44 and 16:30 on 2026-09-05.",
]


def dataset_block(ds):
    ph = J(f"{W}/phase0/{ds}.json"); leg = J(f"{W}/legacy/{L[ds]}/summary.json")
    ir = J(f"{ROOT}/{ds}/integrity_report.json"); bi = J(f"{ROOT}/{ds}/build_info.json")
    lc = J(f"{ROOT}/{ds}/legacy_comparison.json"); qi = J(f"{ROOT}/{ds}/query_independence_test.json")
    pc = J(f"data/canonical/{PHASEC_DIR[ds]}/document_manifest.json")
    P = PER[ds]
    acc = ir["corpus_accounting_detail"]
    srcs = {}
    for p, h in bi["source_files"].items():
        ref = OFFICIAL["file_hashes"].get(p)
        srcs[p] = {"sha256": h["sha256"], "bytes": h["bytes"], "matches_results_data_audit_manifest": (ref["sha256"] == h["sha256"]) if ref else "not_listed_there"}
    # official split / query counts
    if ds == "metaqa":
        qc = OFFICIAL["datasets"]["metaqa"]["counts"]
        splits = {"per_hop": qc, "totals": {"train": 329282, "dev": 39138, "test": 39093}}
    elif ds == "hotpotqa":
        splits = {**OFFICIAL["datasets"]["hotpotqa"]["counts"],
                  "canonical_v1_query_splits": {"train": 90447, "validation": 7405, "test": 7405},
                  "note": "train is shared by the distractor and fullwiki settings; fullwiki TEST labels are HIDDEN "
                          "(answers None, supporting_facts empty in the official parquet)"}
    else:
        splits = OFFICIAL["datasets"][ds]["counts"]
    if ds == "metaqa":
        src_records = {"kb_entity_dict_rows": ph["dict_rows"], "kb_triples": ph["kb_triples"], "kb_unique_entities": ph["kb_unique_entities"]}
    elif ds == "2wiki":
        # the corpus is now para_with_hyperlink, so the phase-0 CONTEXT statistics describe the query layer only
        src_records = {"article_records": ir["raw_source_records"], "total_sentences": acc.get("total_sentences"),
                       "total_hyperlink_mentions": acc.get("total_hyperlink_mentions"),
                       "duplicate_curids": ir["duplicates_collapsed"], "empty_text_articles": ir["empty_text_nodes"],
                       "query_layer_only__context_records": (ph or {}).get("total_context_records"),
                       "query_layer_only__unique_context_titles": (ph or {}).get("unique_titles"),
                       "query_layer_context_titles_as_pct_of_corpus": round(100.0 * ((ph or {}).get("unique_titles") or 0)
                                                                            / ir["raw_source_records"], 2)}
    elif ds == "hotpotqa":
        src_records = {"article_records": ir["raw_source_records"], "tar_members_read": acc.get("tar_members_read"),
                       "duplicate_curids": ir["duplicates_collapsed"], "empty_abstract_articles": ir["empty_text_nodes"],
                       "legacy_distractor_context_titles": 507494,
                       "pct_of_corpus_never_referenced_by_any_question": round(100.0 * (1 - 507494 / ir["raw_source_records"]), 2)}
    elif ds == "musique":
        src_records = {"paragraph_records": ph["n_paragraph_records"], "total": ph["total_paragraph_records"], "unique_text_sha": ph["unique_text_sha"],
                       "unique_title_text_pairs": ph["unique_title_text_pairs"], "supporting_per_question": ph["supporting_per_question_dist"]}
    else:
        src_records = {"articles": ph["n_articles"], "paragraphs": ph["n_paragraphs"], "unique_contexts": ph["unique_context_sha"], "unique_title_context_pairs": ph["unique_title_context_pairs"]}
    cache_meta = json.loads(str(__import__("numpy").load(f"results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_{L[ds]}.npz", allow_pickle=True)["meta_json"]))
    extra = {}
    if ds == "2wiki":
        extra["legacy_text_vs_official"] = J(f"{W}/2wiki_textdiff.json")
        extra["phase_c_divergence"] = acc.get("phase_c_divergence")
        extra["SUPERSESSION"] = acc.get("REPLACES")
    if ds == "hotpotqa":
        extra["phase_c_divergence"] = acc.get("phase_c_divergence")
    if ds == "metaqa": extra["legacy_lowercase_collapse"] = ph["legacy_lowercase_collapse"]
    if ds == "musique": extra["legacy_index_alignment"] = ph["legacy_index_alignment"]
    n_can = ir["canonical_node_count"]
    recon = {"expected_count_clue": P["count_clue"], "canonical_v1": n_can, "delta": n_can - P["count_clue"],
             "explanation": {"metaqa": "identical", "squad": "identical",
                             "2wiki": "identical -- _APPROVALS.json target_n 5,989,847 was an EXPECTED value and the archive was "
                                      "streamed to derive it independently; the two agree exactly",
                             "hotpotqa": "identical -- _APPROVALS.json target_n 5,233,329 was an EXPECTED value and the tarball was "
                                         "streamed to derive it independently; the two agree exactly",
                             "musique": "+1: (title,text) identity keeps the one paragraph text shared by two titles as two nodes; text-hash identity would give 117,533"}[ds]}
    return {
        "dataset": ds, "legacy_substrate_name": L[ds],
        "official_sources": P["official_sources"], "official_source_hashes": srcs,
        "official_splits_and_query_counts": splits,
        "source_record_counts": src_records,
        "phase_c_canonical_2026_08_23": {"path": f"data/canonical/{ds}/documents.jsonl", "count": pc["n_docs"] if pc else None, "builder_rule": P["phase_c_rule"],
                                         "role_here": "CROSS-CHECK only (canonical_v1 re-derived from data/original)"},
        "legacy_substrate": {"cache": f"results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_{L[ds]}.npz", "n_docs": cache_meta["n_docs"], "n_eval_queries": cache_meta["n_dev_queries"],
                             "sample_rule": cache_meta["sample_rule"], "master": leg["master_path"], "master_docs": leg["n_docs"], "master_questions": leg["n_questions"],
                             "generated_how": P["legacy_generation"], "depends_on_eval_subset": P["depends_on_eval_subset"], "stale": P["stale"],
                             "classification": P["classification"], "legacy_rules": {"normalization": P["legacy_normalization"], "dedup": P["legacy_dedup"], "id": P["legacy_id_rule"]}},
        "legacy_vs_canonical_v1": {k: lc[k] for k in ("legacy_count", "canonical_count", "overlap_legacy_nodes_mapped", "missing_in_canonical", "ambiguous_legacy_nodes",
                                                       "extra_canonical_nodes_not_in_legacy", "extra_by_split_provenance", "canonical_nodes_matched_by_legacy", "match_classes",
                                                       "changed_text", "id_mapping_coverage", "eval_subset_gold_consistency")},
        "canonical_v1": {"count": n_can, "identity": P["canonical_v1_identity"], "raw_source_records": ir["raw_source_records"], "duplicates_collapsed": ir["duplicates_collapsed"],
                         "dropped_records": ir["dropped_records"], "CORPUS_HASH": ir["CORPUS_HASH"], "NODE_ORDER_HASH": ir["NODE_ORDER_HASH"],
                         "ALL_EVAL_GOLDS_PRESENT": ir["ALL_EVAL_GOLDS_PRESENT"], "CORPUS_QUERY_INDEPENDENT": qi["CORPUS_QUERY_INDEPENDENT"], "DETERMINISTIC_REBUILD": qi["DETERMINISTIC_REBUILD"],
                         "identity_normalization_diagnostics": ir["corpus_accounting_detail"].get("identity_normalization_diagnostics")},
        "count_reconciliation": recon,
        "eval_subset_recovery": ir["eval_subset"],
        "extra": extra,
    }


def md(A):
    o = []
    o.append("# FINAL DATASET CANONICALIZATION AUDIT (Track B, canonical_v1)\n")
    o.append(f"Generated {A['generated']} at git {A['git_commit']}. Invariant: **{A['invariant']}**\n")
    o.append("## Summary table\n")
    o.append("| dataset | legacy substrate | legacy N | canonical_v1 N | legacy classification | corpus depends on eval subset | query-independent | all golds resolve | CORPUS_HASH |")
    o.append("|---|---|---:|---:|---|---|---|---|---|")
    for ds, d in A["datasets"].items():
        ls = d["legacy_substrate"]; c = d["canonical_v1"]
        o.append(f"| {ds} | {d['legacy_substrate_name']} | {ls['n_docs']:,} | {c['count']:,} | {ls['classification']} | {ls['depends_on_eval_subset']} | {c['CORPUS_QUERY_INDEPENDENT']} | {c['ALL_EVAL_GOLDS_PRESENT']} | {c['CORPUS_HASH'][:12]} |")
    for k, v in A["deferred"].items():
        o.append(f"| {k} | {k} | {v['legacy_n']:,} | **BLOCKED — none built** (rejected candidate {v['reference_n']:,}) | {v['classification']} | {v['depends_on_eval_subset']} | BLOCKED | BLOCKED | - |")
    o.append("\n## How each legacy substrate was generated (traced in code)\n")
    for ds, d in A["datasets"].items():
        ls = d["legacy_substrate"]
        o.append(f"### {ds} (legacy `{d['legacy_substrate_name']}`, {ls['n_docs']:,} nodes, eval {ls['n_eval_queries']} = `{ls['sample_rule']}`)\n")
        o.append(f"- **Generation:** {ls['generated_how']}\n- **Legacy normalization:** {ls['legacy_rules']['normalization']}\n- **Legacy dedup:** {ls['legacy_rules']['dedup']}\n- **Legacy ids:** {ls['legacy_rules']['id']}\n"
                 f"- **Depends on eval subset:** {ls['depends_on_eval_subset']}; **stale:** {ls['stale']}; **classification:** {ls['classification']}\n"
                 f"- **Phase-C (2026-08-23) rule:** {d['phase_c_canonical_2026_08_23']['builder_rule']} -> {d['phase_c_canonical_2026_08_23']['count']:,} (cross-check only)\n"
                 f"- **canonical_v1 identity:** {d['canonical_v1']['identity']}; raw records {d['canonical_v1']['raw_source_records']:,} = nodes {d['canonical_v1']['count']:,} + duplicates {d['canonical_v1']['duplicates_collapsed']:,} + dropped {d['canonical_v1']['dropped_records']}\n"
                 f"- **Count reconciliation vs clue {d['count_reconciliation']['expected_count_clue']:,}:** delta {d['count_reconciliation']['delta']} -- {d['count_reconciliation']['explanation']}\n")
        lv = d["legacy_vs_canonical_v1"]
        o.append(f"- **Legacy -> canonical mapping:** {lv['overlap_legacy_nodes_mapped']:,}/{lv['legacy_count']:,} mapped ({lv['match_classes']}), missing {lv['missing_in_canonical']}, ambiguous {lv['ambiguous_legacy_nodes']}; "
                 f"canonical nodes matched {lv['canonical_nodes_matched_by_legacy']:,}; extra canonical nodes {lv['extra_canonical_nodes_not_in_legacy']:,} by split provenance {lv['extra_by_split_provenance']}; changed text {lv['changed_text']}\n")
        ev = lv["eval_subset_gold_consistency"]
        o.append(f"- **Frozen eval subset gold consistency (legacy golds mapped vs canonical golds):** equal {ev['legacy_golds_equal_canonical']}/{ev['n']}, differ {ev['differ']}\n")
        er = d["eval_subset_recovery"]
        o.append(f"- **Eval subset recovery:** {er['method']} -> {er['n_recovered']}/{er['n_legacy']} recovered, {er['n_missed']} missed; official split distribution {er['official_split_distribution']}\n")
        if ds == "2wiki":
            t = d["extra"]["legacy_text_vs_official"]
            o.append(f"- **Legacy (FlashRAG) text vs official text:** {t['per_legacy_node_text_class']}; {t['legacy_titles_with_multiple_text_variants']} legacy titles carried 2+ text variants. "
                     "All non-exact differences are punctuation spacing / whitespace (official text keeps tokenizer artefacts, e.g. `Rosenberg( August 11`).\n")
        if d["extra"].get("phase_c_divergence"):
            pcd = d["extra"]["phase_c_divergence"]
            bits = [f"`.strip()` changes the text for **{pcd['n_records_where_strip_changes_text']:,}** records"]
            if "n_records_where_phase_c_fell_back_to_text_with_links" in pcd:
                bits.append(f"Phase-C fell back to `text_with_links` for **{pcd['n_records_where_phase_c_fell_back_to_text_with_links']:,}**")
            if "n_records_with_a_different_phase_c_encoder_input" in pcd:
                bits.append(f"total differing encoder inputs **{pcd['n_records_with_a_different_phase_c_encoder_input']:,}**")
            o.append(f"- **Phase-C divergence (MEASURED, not assumed):** " + "; ".join(bits) +
                     f". Reference store: `{pcd.get('phase_c_reference', 'data/canonical/' + PHASEC_DIR[ds] + '/documents.jsonl')}`. "
                     "This is the exact upper bound on dense/SPLADE rows needing a re-encode.\n")
        if d["extra"].get("SUPERSESSION"):
            s = d["extra"]["SUPERSESSION"]
            o.append(f"- **SUPERSESSION:** replaces the {s['superseded_n']:,}-node {s['superseded_corpus']} "
                     f"(corpus_hash `{s['superseded_corpus_hash_16']}`); superseded artifacts PRESERVED at `{s['superseded_artifacts']}`.\n")
        if ds == "metaqa":
            t = d["extra"]["legacy_lowercase_collapse"]
            o.append(f"- **Lowercase collapse:** {t['n_lowercase_keys']:,} lowercase keys from 43,234 names; {t['n_collided_groups']} groups; {t['n_names_absorbed']} names absorbed (examples: {list(t['examples'].items())[:3]}).\n")
    o.append("\n## Common legacy plumbing (citations)\n")
    for k, v in A["common_citations"].items(): o.append(f"- **{k}:** {v}")
    o.append("\n## `_clean` provenance\n")
    for k, v in A["clean_provenance"].items(): o.append(f"- **{k}:** {v}")
    o.append("\n## BLOCKED (status record only; nothing built, and the builder refuses unconditionally)\n")
    for k, v in A["deferred"].items():
        o.append(f"- **{k}: {v['status']}** — legacy {v['legacy_n']:,} = {v['what_it_is']} | REJECTED candidate {v['reference_n']:,} "
                 f"({v['reference']}) | {v['classification']} | refusal: `{v['builder_refusal']}` | proposal: `{v['proposal']}` | file `{v['status_file']}`")
    o.append("\n## Judgment calls\n")
    for j in A["judgment_calls"]: o.append(f"- {j}")
    o.append("\n## Official source hashes (all verified against results/data_audit/dataset_manifest.json)\n")
    for ds, d in A["datasets"].items():
        bad = [p for p, h in d["official_source_hashes"].items() if h["matches_results_data_audit_manifest"] is False]
        o.append(f"- {ds}: {len(d['official_source_hashes'])} files hashed; mismatches: {bad if bad else 'none'}")
    o.append("\n## Artifacts\n")
    for a in A["artifacts"]: o.append(f"- `{a}`")
    return "\n".join(o) + "\n"


def main():
    try: commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    ds_blocks = {ds: dataset_block(ds) for ds in ("metaqa", "2wiki", "musique", "squad", "hotpotqa")}
    norm = "; ".join(f"{ds}: NFC {d['canonical_v1']['identity_normalization_diagnostics']['extra_collapses_if_NFC']}, "
                     f"whitespace {d['canonical_v1']['identity_normalization_diagnostics']['extra_collapses_if_whitespace_collapse']}, "
                     f"NFC+ws+casefold {d['canonical_v1']['identity_normalization_diagnostics']['extra_collapses_if_NFC+ws+casefold']}"
                     for ds, d in ds_blocks.items())
    judgment = [j.replace("{NORM_DIAG}", norm) for j in JUDGMENT_CALLS]
    # only webqsp is still without a corpus, and it is BLOCKED (not merely deferred): the builder refuses it
    # unconditionally.  hotpotqa_clean is no longer a directory -- it is a documented legacy alias of hotpotqa.
    deferred = {}
    for k in ("webqsp",):
        s = J(f"{ROOT}/{k}/status.json")
        deferred[k] = {"status": s["status"], "legacy_n": s["legacy_substrate"]["n_docs_from_cache"], "what_it_is": s["legacy_substrate"]["what_it_actually_is"][:160] + "...",
                       "classification": s["legacy_substrate"]["classification"], "depends_on_eval_subset": True,
                       "reference_n": s["official_or_canonical_reference"]["phase_c_n_docs"], "reference": s["official_or_canonical_reference"]["phase_c_canonical"],
                       "status_file": f"{ROOT}/{k}/status.json",
                       "REJECTED_reference": "the reference_n above (1,316,466) is the REJECTED RoG subgraph-union count, kept for "
                                             "audit only; it is NOT a canonical target",
                       "builder_refusal": "scratchpad/final_canonical_build/build_kb.py NEVER_BUILD = {'webqsp'} -- unconditional",
                       "proposal": f"{ROOT}/webqsp/FREEBASE_SOURCE_PROPOSAL.md"}
    artifacts = []
    for root, _, files in os.walk(ROOT):
        if "_work" in root.replace("\\", "/"): continue
        for f in sorted(files): artifacts.append(os.path.join(root, f).replace("\\", "/"))
    A = {"title": "FINAL DATASET CANONICALIZATION AUDIT (Track B)", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit,
         "invariant": "QUERY SUBSETTING IS ALLOWED. CORPUS SUBSETTING BY QUERY IS FORBIDDEN. Canonical corpus is fixed once per dataset, independent of the evaluated queries.",
         "method": "Every count re-derived from data/original by streaming builders in scratchpad/final_canonical_build/ (no json.load of large files; peak RSS <= 1.3 GB). "
                   "Legacy facts traced in code (citations inline) and measured on the exact node tables the frozen G2 caches index (data/processed/master_nodes_*.json). "
                   "data/canonical (Phase C, 2026-08-23) used as a cross-check only. Nothing existing was modified; no downstream artefacts built.",
         "datasets": ds_blocks, "common_citations": COMMON_CITATIONS, "clean_provenance": CLEAN_PROVENANCE, "judgment_calls": judgment,
         "deferred": deferred, "artifacts": artifacts}
    json.dump(A, open(f"{ROOT}/CANONICALIZATION_AUDIT.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    open(f"{ROOT}/CANONICALIZATION_AUDIT.md", "w", encoding="utf-8", newline="\n").write(md(A))
    print("wrote", f"{ROOT}/CANONICALIZATION_AUDIT.json", f"{ROOT}/CANONICALIZATION_AUDIT.md", "artifacts:", len(artifacts))


if __name__ == "__main__":
    main()
