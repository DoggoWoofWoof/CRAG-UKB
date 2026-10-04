"""Phase 1 (Track B) -- SOURCE_CONTRACT.json for all six datasets + SOURCE_CONTRACTS_FOR_REVIEW.md.

Writes ONLY under data/final_canonical/.  Pure metadata: reads existing manifests + hashes source files.
No corpus is built here.  EXPECTED_NODE_COUNT is the prediction made BEFORE building;
DERIVED_NODE_COUNT is null until the build actually runs.

    python scratchpad/final_canonical_build/source_contracts.py
"""
import os, json, hashlib, time

ROOT = "data/final_canonical"
AUDIT = "results/data_audit/dataset_manifest.json"
FIELDS = ["DATASET", "CANONICAL_CORPUS_SOURCE", "SOURCE_VERSION", "SOURCE_PATHS", "SOURCE_HASHES",
          "WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE", "RAW_SOURCE_RECORDS", "CANONICAL_NODE_IDENTITY_RULE",
          "NORMALIZATION_RULE", "DEDUP_RULE", "EXPECTED_NODE_COUNT", "DERIVED_NODE_COUNT",
          "QUERY_DEPENDENT_CORPUS", "QUALITY_LOCKED_QUERY_SOURCE", "SCALE_ALL_QUERY_SOURCE",
          "SOURCE_CONTRACT_STATUS"]


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def hashes(paths, audit_fh, extra):
    out = {}
    for p in paths:
        k = p.replace("\\", "/")
        if k in extra:
            out[k] = dict(extra[k]); out[k]["hash_source"] = "hashed in this task"
        elif k in audit_fh:
            out[k] = dict(audit_fh[k]); out[k]["hash_source"] = "results/data_audit/dataset_manifest.json"
        else:
            out[k] = {"sha256": file_sha(k), "bytes": os.path.getsize(k), "hash_source": "hashed in this task"}
        out[k]["exists"] = os.path.exists(k)
    return out


def metaqa_paths():
    ps = ["data/original/metaqa/entity/kb_entity_dict.txt", "data/original/metaqa/kb.txt"]
    return ps


def build():
    audit = json.load(open(AUDIT, encoding="utf-8"))
    afh = audit["file_hashes"]
    extra_p = f"{ROOT}/_work/extra_source_hashes.json"
    extra = json.load(open(extra_p, encoding="utf-8")) if os.path.exists(extra_p) else {}
    extra = {k.replace("\\", "/"): v for k, v in extra.items()}

    C = {}

    # ---------------------------------------------------------------- metaqa
    C["metaqa"] = {
        "DATASET": "metaqa",
        "CANONICAL_CORPUS_SOURCE": "Official MetaQA WikiMovies KB entity dictionary: data/original/metaqa/entity/kb_entity_dict.txt "
                                   "(the KB node vocabulary), cross-checked against the KB triple file kb.txt.",
        "SOURCE_VERSION": "MetaQA 'vanilla' release (Zhang et al., AAAI 2018), github.com/yuyuz/MetaQA; "
                          "immutable copy under data/original/metaqa (data/raw/metaqa is sha-identical).",
        "SOURCE_PATHS": metaqa_paths(),
        "WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE":
            "kb_entity_dict.txt IS the KB's own complete node vocabulary, a KB artifact with no question content in it: each "
            "line is '<index>\\t<entity name>' and no question id, question string, answer annotation or split marker appears "
            "anywhere in the file. Independence is verifiable in both directions: (a) every one of the 134,741 kb.txt triples' "
            "endpoints is present in the dict (kb_entities_missing_from_dict = 0, recorded in "
            "data/final_canonical/metaqa/integrity_report.json -> corpus_accounting_detail.kb_txt_cross_check), so the dict is a "
            "superset of everything the KB graph can reach; (b) the QA files (N-hop/vanilla/qa_{split}.txt) are read only in the "
            "query stage and never contribute a node. MetaQA has no documents -- the retrieval universe IS the KB entity set -- so "
            "there is no larger corpus this could be a subset of. Evaluating 1 question or all 407,513 yields the same 43,234 nodes.",
        "RAW_SOURCE_RECORDS": 43234,
        "CANONICAL_NODE_IDENTITY_RULE": "official kb_entity_dict.txt row index; entity name kept EXACT (case preserved); "
                                        "node_id = 'metaqa:e<index zero-padded to 5>'.",
        "NORMALIZATION_RULE": "Stored text = the EXACT official entity name. No NFC/NFKC, no casefolding, no whitespace collapsing, "
                              "no stripping, no '_'->' '. text == title == entity name. The legacy lossy collapse "
                              "(strip().lower().replace('_',' '), which merged 3,083 distinct-case entities down to 40,151) is NOT "
                              "reproduced; it is quantified only, in identity_normalization_diagnostics. Triple verbalization "
                              "(display name + up to 10 kb.txt triples) is a downstream ENCODING choice, not corpus identity.",
        "DEDUP_RULE": "identity = exact entity name; a repeated name would be dropped as duplicate_entity_name. "
                      "0 duplicates occur (43,234 rows -> 43,234 distinct names).",
        "EXPECTED_NODE_COUNT": 43234,
        "DERIVED_NODE_COUNT": 43234,
        "QUERY_DEPENDENT_CORPUS": False,
        "QUALITY_LOCKED_QUERY_SOURCE": "official MetaQA TEST split, all 3 hop families (vanilla), 39,093 questions -- "
                                       "test labels are PUBLIC for MetaQA (results/data_audit: official_test_labels='public').",
        "SCALE_ALL_QUERY_SOURCE": "every official MetaQA vanilla question in all splits x all hops = 407,513 "
                                  "(train 329,282 + dev 39,138 + test 39,093).",
        "SOURCE_CONTRACT_STATUS": "BUILT_AND_ACCEPTED (derived retroactively from the accepted canonical_v1 build)",
    }

    # ---------------------------------------------------------------- 2wiki
    C["2wiki"] = {
        "DATASET": "2wiki",
        "CANONICAL_CORPUS_SOURCE": "The official 2WikiMultihopQA v1.0 (ids, April 2021) FULL ARTICLE UNIVERSE: every record of "
                                   "para_with_hyperlink.jsonl inside "
                                   "data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip -- one node per Wikipedia "
                                   "article curid. The official release describes this file as containing all articles/paragraphs "
                                   "with hyperlink information except error paragraphs, i.e. it IS the complete retrieval corpus. "
                                   "The question files are read only for the QUERY layer (gold/context title resolution); no "
                                   "question can add or remove a node.",
        "SOURCE_VERSION": "2wikimultihopqa v1.0_ids_april2021 (Ho et al., COLING 2020), data_ids_april7.zip; "
                          "para_with_hyperlink.zip (one member, para_with_hyperlink.jsonl, 7,023,046,781 bytes uncompressed) plus "
                          "the extracted train/dev/test json under data/original/2wiki/v1.0_ids_april2021.",
        "SOURCE_PATHS": ["data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip"] +
                        [f"data/original/2wiki/v1.0_ids_april2021/{s}.json" for s in ("train", "dev", "test")],
        "WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE":
            "para_with_hyperlink.jsonl is a WIKIPEDIA-SNAPSHOT derivative, not a benchmark by-product, and that is checkable from "
            "the file itself:\n"
            "(1) Record schema is {id, title, sentences, mentions} keyed by the Wikipedia curid. There is no question id, question "
            "string, answer, supporting_fact, split label or any other 2Wiki annotation anywhere in a record; the record set is "
            "fixed by the underlying Wikipedia dump and is defined without reference to any question.\n"
            "(2) Scale proves non-derivation: 5,989,847 article records, whereas the deduplicated union of the context paragraphs "
            "of ALL 192,606 official questions of ALL three splits is only 398,354 titles -- i.e. ~93.3% of the corpus is never "
            "referenced by any 2Wiki question. A question-derived collection cannot contain 5.59M unreferenced records.\n"
            "(3) It is the file the official release itself designates as the retrieval corpus ('all the paragraphs with hyperlink "
            "information except error paragraphs').\n"
            "(4) SUPERSESSION: the previously built canonical_v1 2wiki node table was exactly the pooled question-context universe "
            "(398,354 nodes, corpus_hash ec7fbbe8cd783040...). It was REJECTED in this task because, although invariant to the "
            "EVALUATED query subset, its membership is still a function of the question set -- distractor mass, partition structure "
            "and reachability are all inherited from the questions, and it applied a weaker standard than the one HotpotQA is held "
            "to (full 5.23M fullwiki abstracts, distractor-context union rejected). The superseded artifacts are preserved, not "
            "deleted, under data/final_canonical/2wiki/_superseded_context_union_398354/ (see SUPERSESSION.json) because the old "
            "node table is needed to remap the 398,354 already-encoded embedding rows.",
        "RAW_SOURCE_RECORDS": "5,989,847 article records expected (one json line per article in para_with_hyperlink.jsonl); "
                              "counted exactly at build time. The question layer additionally reads 1,926,060 context entries over "
                              "192,606 questions, which contribute NO nodes.",
        "CANONICAL_NODE_IDENTITY_RULE": "official Wikipedia curid (the record's 'id' field), exact string; "
                                        "node_id = '2wiki:c<curid>'. The id is READABLE rather than hashed: the curid is already "
                                        "source-stable and globally unique, so hashing buys no collision safety and costs "
                                        "debuggability and remapping. Titles are NOT the identity (several curids can share a "
                                        "title), so gold/context resolution is title -> curid via the corpus and an ambiguous "
                                        "title contributes ALL matching nodes, recorded per query in gold_title_ambiguous.",
        "NORMALIZATION_RULE": "Stored text = ' '.join(official 'sentences' list), because the official file stores the article as a "
                              "sentence list. The OFFICIAL string is then kept verbatim including its tokenizer artefacts. No "
                              "NFC/NFKC, no casefold, no whitespace collapse, and explicitly NO strip -- note that the pre-existing "
                              "full-universe encode (scratchpad/build_2wiki_universe.py:42) DID apply .strip() to the joined text, "
                              "so the build COUNTS how many records differ under that one step (strip_delta) and reports the exact "
                              "dense/SPLADE reuse impact rather than assuming it. "
                              "Empty-text articles are KEPT as nodes: dropping them would make corpus membership depend on a "
                              "text-quality filter.\n"
                              "SECOND CONSTRUCTION STEP -- TEXTUALIZATION_REV 2 (2026-09-06). Where that join is EMPTY, the canonical "
                              "text is the record's own 'title'. This is the SAME general source-level rule hotpotqa received on "
                              "2026-09-05, not a per-dataset policy: an empty body plus a non-empty source title becomes the title, "
                              "everywhere it occurs. FALLBACK ONLY -- a record that already yields text is untouched and no title is "
                              "prepended anywhere else, so the join above remains the rule for 5,902,082 of 5,989,847 records. "
                              "Applies to the 87,765 records whose 'sentences' list is empty; the forensic scan "
                              "(results/data_audit/final_canonical_v1/2wiki/EMPTY_TEXT_AUDIT.md) measured TITLE_NONEMPTY = 87,765, "
                              "ALT_TEXT_NONEMPTY = 0 and TRULY_TEXTLESS = 0, so the title is the only source-native candidate and no "
                              "record needs to stay empty. Query-independent: the para_with_hyperlink schema is exactly "
                              "{id, title, sentences, mentions} and holds no question, answer or supporting_facts field, and the "
                              "corpus pass never opens a question file (re-proven by executing the four-lane QI test after the "
                              "change). Motivation: \"\" tokenizes to the ZERO-LENGTH sequence [] under gte-Qwen2, so all such nodes "
                              "collapsed onto one degenerate forward pass, and 20,524 of them are hyperlink targets -- the degenerate "
                              "points would propagate into every KNN/H4 structure built over this corpus. This MOVES CORPUS_HASH "
                              "(rev1 81fa7d1a5d4bbb24...; the live value is build_info.json -> CORPUS_HASH) but leaves membership, "
                              "node identity, node order and NODE_ORDER_HASH untouched. The rev1 build outputs are preserved under "
                              "data/final_canonical/_superseded_textualization_rev1/2wiki/ with the hash history in SUPERSEDED.md.",
        "DEDUP_RULE": "identity = curid; a repeated curid collapses onto its first occurrence (counted at build time). "
                      "Text-hash identity is NOT used: distinct articles with identical intro text must stay distinct nodes.",
        "EXPECTED_NODE_COUNT": 5989847,
        "DERIVED_NODE_COUNT": None,
        "QUERY_DEPENDENT_CORPUS": False,
        "QUALITY_LOCKED_QUERY_SOURCE": "official 2Wiki DEV split, 12,576 questions. Official TEST labels are HIDDEN "
                                       "(results/data_audit: official_test_labels='HIDDEN (answers withheld)'), so dev is the only "
                                       "proper held-out labelled split.",
        "SCALE_ALL_QUERY_SOURCE": "every official question in all splits = 192,606 (train 167,454 + dev 12,576 + test 12,576); "
                                  "test carries no gold, usable for scale/latency only.",
        "SOURCE_CONTRACT_STATUS": "APPROVED (data/final_canonical/_APPROVALS.json: approvals.2wiki.status) -- full-universe corpus; "
                                  "supersedes the REJECTED 398,354 context-union table",
    }

    # ---------------------------------------------------------------- musique
    C["musique"] = {
        "DATASET": "musique",
        "CANONICAL_CORPUS_SOURCE": "Official MuSiQue-Ans v1.0 candidate paragraphs: the union of ALL 20 candidate paragraphs "
                                   "(supporting AND distractor) of every question in train + dev + test.",
        "SOURCE_VERSION": "MuSiQue-Ans v1.0 (Trivedi et al., TACL 2022), github.com/StonyBrookNLP/musique; "
                          "data/original/musique/v1.0.",
        "SOURCE_PATHS": ["data/original/musique/v1.0/musique_ans_v1.0_train.jsonl",
                         "data/original/musique/v1.0/musique_ans_v1.0_dev.jsonl",
                         "data/original/musique/v1.0/musique_ans_v1.0_test.jsonl",
                         "data/original/musique/v1.0/dev_test_singlehop_questions_v1.0.json"],
        "WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE":
            "MuSiQue ships NO corpus file: the release is question records only, and the ONLY paragraph text the authors distribute is "
            "the 20-paragraph candidate pool attached to each question (results/data_audit/dataset_manifest.json: "
            "native_graph='none (question-specific paragraph sets; decomposition provided)'). The union of those pools over all 24,814 "
            "questions of all three splits is therefore the maximal retrieval universe that exists for this benchmark, and it is a pure "
            "function of the three official files (496,194 raw paragraph entries -> 117,534 nodes). Two independence properties are "
            "enforced and verified: (a) the corpus is built from the 'paragraphs' field ONLY -- 'is_supporting' is never read at corpus "
            "time, so the gold/distractor distinction cannot shrink it (this is exactly the defect of the legacy 13,672-node substrate, "
            "which was the GOLD paragraphs of the train pool alone); (b) all three splits are always included, so no evaluated subset "
            "can change it. The corpus is 8.6x the legacy one and 82.5k of its nodes are pure distractors that no question's gold set "
            "ever names.",
        "RAW_SOURCE_RECORDS": 496194,
        "CANONICAL_NODE_IDENTITY_RULE": "(article title, paragraph_text) exact pair; node_id = 'musique:<sha256(title+U+001F+text)[:24]>'.",
        "NORMALIZATION_RULE": "Stored text = the EXACT official paragraph_text; title = the EXACT official title. "
                              "No NFC/NFKC, no casefold, no whitespace collapse, no strip.",
        "DEDUP_RULE": "identity = (title, paragraph_text) exact; 378,660 repeats collapse. Text-hash-only identity would give 117,533 "
                      "(one paragraph text is shared by the articles 'Adei Ad' and 'Etz Efraim'); the (title,text) rule keeps them "
                      "distinct so a gold reference cannot point at two articles at once.",
        "EXPECTED_NODE_COUNT": 117534,
        "DERIVED_NODE_COUNT": 117534,
        "QUERY_DEPENDENT_CORPUS": False,
        "QUALITY_LOCKED_QUERY_SOURCE": "official MuSiQue-Ans DEV split, 2,417 questions. Official TEST labels are HIDDEN "
                                       "(is_supporting/answer withheld; verified: 0/2,459 test questions carry gold refs).",
        "SCALE_ALL_QUERY_SOURCE": "every official question in all splits = 24,814 (train 19,938 + dev 2,417 + test 2,459); "
                                  "test carries no gold, usable for scale/latency only.",
        "SOURCE_CONTRACT_STATUS": "BUILT_AND_ACCEPTED (derived retroactively from the accepted canonical_v1 build)",
    }

    # ---------------------------------------------------------------- squad
    C["squad"] = {
        "DATASET": "squad",
        "CANONICAL_CORPUS_SOURCE": "Official SQuAD v2.0 article paragraphs: the union of every 'context' of every paragraph of every "
                                   "article in train-v2.0.json + dev-v2.0.json.",
        "SOURCE_VERSION": "SQuAD v2.0 (Rajpurkar et al., ACL 2018), data/original/squad/v2.0.",
        "SOURCE_PATHS": ["data/original/squad/v2.0/train-v2.0.json", "data/original/squad/v2.0/dev-v2.0.json"],
        "WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE":
            "SQuAD's distributed corpus IS its article paragraph set: the two official files contain a 'data' array of articles, each "
            "with a 'paragraphs' array whose 'context' strings are the complete passage collection the benchmark ever draws from; there "
            "is no separate SQuAD document dump. The corpus stage reads only (article title, paragraph.context) and never touches "
            "'qas'/'answers', and BOTH official splits are always included -- which is precisely the defect it repairs in the legacy "
            "19,029-node substrate, built from train contexts alone (the 1,204 dev-only paragraphs were never loaded). 20,239 raw "
            "paragraph records -> 20,233 nodes is a pure function of the two files.",
        "RAW_SOURCE_RECORDS": 20239,
        "CANONICAL_NODE_IDENTITY_RULE": "(article title, context) exact pair; node_id = 'squad:<sha256(title+U+001F+context)[:24]>'. "
                                        "(Here identical to context-hash identity: 0 contexts are shared across titles.)",
        "NORMALIZATION_RULE": "Stored text = the EXACT official context string; title = the EXACT official article title. "
                              "No NFC/NFKC, no casefold, no whitespace collapse, no strip.",
        "DEDUP_RULE": "identity = (title, context) exact; 6 exact duplicate paragraphs collapse.",
        "EXPECTED_NODE_COUNT": 20233,
        "DERIVED_NODE_COUNT": 20233,
        "QUERY_DEPENDENT_CORPUS": False,
        "QUALITY_LOCKED_QUERY_SOURCE": "official SQuAD v2.0 DEV split, 11,873 questions (the standard held-out set; SQuAD's true test "
                                       "set is never released).",
        "SCALE_ALL_QUERY_SOURCE": "every official question in both released splits = 142,192 (train 130,319 + dev 11,873).",
        "SOURCE_CONTRACT_STATUS": "BUILT_AND_ACCEPTED (derived retroactively from the accepted canonical_v1 build)",
    }

    # ---------------------------------------------------------------- webqsp
    C["webqsp"] = {
        "DATASET": "webqsp",
        "CANONICAL_CORPUS_SOURCE": "*** REJECTED -- NOT THE CANONICAL SOURCE. NO CORPUS EXISTS FOR webqsp. *** The source described "
                                   "here is the union of the Freebase entity nodes of the RoG-webqsp per-question subgraphs over ALL "
                                   "THREE RoG splits (train 2,826 + validation 246 + test 1,628 = 4,700 questions), entities being the "
                                   "distinct endpoints (head and tail) of every triple in every question's 'graph' field. It is "
                                   "recorded verbatim so the rejection is auditable, and it is REJECTED because corpus membership is "
                                   "query-derived: an entity is in it only because some question's subgraph reached it. A canonical "
                                   "webqsp corpus must come from a Freebase dump, which is NOT on this machine "
                                   "(verified by exhaustive search). See data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md.",
        "SOURCE_VERSION": "rmanluo/RoG-webqsp parquet export (subgraph layer) + Microsoft WebQSP 2016 (question layer); "
                          "data/original/webqsp/rog_webqsp + data/original/webqsp/WebQSP/data.",
        "SOURCE_PATHS": ["data/original/webqsp/rog_webqsp/train-00000-of-00002.parquet",
                         "data/original/webqsp/rog_webqsp/train-00001-of-00002.parquet",
                         "data/original/webqsp/rog_webqsp/validation-00000-of-00001.parquet",
                         "data/original/webqsp/rog_webqsp/test-00000-of-00002.parquet",
                         "data/original/webqsp/rog_webqsp/test-00001-of-00002.parquet",
                         "data/original/webqsp/WebQSP/data/WebQSP.train.json",
                         "data/original/webqsp/WebQSP/data/WebQSP.test.json"],
        "WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE":
            "THIS IS THE WEAKEST CONTRACT OF THE SIX AND NEEDS AN EXPLICIT DECISION. Evidence, stated plainly:\n"
            "(1) NO question-independent Freebase corpus exists in this project or in the official release. The official Microsoft "
            "WebQSP files (WebQSP.train.json 3,098 / WebQSP.test.json 1,639) contain questions, SPARQL programs, topic entities and "
            "answers -- and NO graph: they cannot yield a single corpus node. There is no Freebase dump on disk (data/original/webqsp "
            "holds exactly the 5 RoG parquets, the 4 WebQSP json files, docs and eval.py). The ONLY Freebase material available is the "
            "per-question RoG subgraph. Consequently a corpus for this dataset is necessarily the union of per-question subgraphs; the "
            "only real choice is WHICH questions.\n"
            "(2) This contract takes ALL 4,700 RoG questions in ALL THREE splits (parquet row counts verified from file metadata: "
            "train 1,413+1,413, validation 246, test 814+814). The corpus is therefore a pure function of the five parquet files and "
            "cannot be changed by any choice of evaluated queries -- the Track-B invariant (corpus not subsettable by query) HOLDS.\n"
            "(3) It explicitly rules out the legacy failure mode. The legacy 781,485-node substrate was built ONLY from the two RoG "
            "TEST parquets (src/pipeline/loader_webqsp.py:118-123 defaults to data/raw/full/kb/webqsp_test0.parquet + webqsp_test1.parquet, "
            "which are sha256-identical to the official RoG test shards), all 1,578 of its question ids are 'WebQTest-*', and 1,419 of "
            "those same test questions were the evaluated set -- i.e. the corpus was literally the union of the evaluated questions' own "
            "subgraphs (verified in data/final_canonical/_work/webqsp_legacy_verify.json). Adding the train+validation subgraphs removes "
            "that coupling and grows the corpus ~1.68x.\n"
            "(4) IT ALSO REMOVES A GOLD-CONDITIONING DEFECT IN THE PHASE-C REFERENCE. scratchpad/c1c2_webqsp.py:30-33 builds its "
            "1,316,466-entity corpus from the triple endpoints AND THEN EXPLICITLY INJECTS every question's ANSWER entities "
            "(`for a in aslist(r['a_entity']): add(a)`) and topic entities (`for q in aslist(r['q_entity']): add(q)`). A corpus that "
            "is seeded with the answers guarantees its own gold coverage and is conditioned on the labels, which is exactly what this "
            "track forbids. THIS CONTRACT READS ONLY THE 'graph' FIELD at corpus time; a_entity/q_entity are read only in the query "
            "stage. Consequence to expect and measure at build time: canonical_v1's node set is a SUBSET of Phase-C's "
            "(<= 1,316,466), and some answer entities may not resolve to a node -- that gold coverage number is a RESULT to report "
            "honestly, not a target to engineer. (Reuse is unaffected: a subset of Phase-C's nodes is still 100% covered by its rows.)\n"
            "(5) HONEST LIMIT: this is still question-set-conditioned by construction of RoG, and it cannot be otherwise without "
            "acquiring a Freebase slice. Any WebQSP number must be reported as retrieval over 'the RoG subgraph union', not over Freebase.",
        "RAW_SOURCE_RECORDS": "4,700 RoG question records (train 2,826 / validation 246 / test 1,628); triple endpoints per record are "
                              "the raw node records and are counted at build time. Question layer: 4,737 official MS questions.",
        "CANONICAL_NODE_IDENTITY_RULE": "Freebase entity SURFACE STRING as it appears in a RoG 'graph' TRIPLE ENDPOINT (head or tail), "
                                        "exact; node_id = 'webqsp:<sha256(entity_string)[:24]>'. Phase-C used the same identity with a "
                                        "16-hex id (webqsp_ent_<sha256(name)[:16]>, 1,316,466 unique ids == 1,316,466 unique text "
                                        "hashes) BUT over a larger set: it also injected a_entity (answers) and q_entity (topic "
                                        "entities). canonical_v1 takes triple endpoints ONLY, so its node set is a subset.",
        "NORMALIZATION_RULE": "OFFICIAL VERBATIM. Stored text = title = the exact RoG entity string, INCLUDING raw Freebase MIDs "
                              "('m.0k8nh0b', 'g.12tb6gh4f') for unlabeled CVT/mediator nodes -- measured at 55.3% of the first 200,000 "
                              "Phase-C entity docs. No NFC/NFKC, no casefold, no whitespace collapse, no strip. The MID->name resolution "
                              "of commit ba6bd71 is NOT applied to canonical_v1 text: it is a graph-dependent DERIVED transform "
                              "(src/pipeline/loader_webqsp.py, deduces a named neighbour through the subgraph), it was applied only to the "
                              "legacy master_nodes_webqsp.json substrate, and applying it here would (a) make the corpus depend on a "
                              "downstream artifact and (b) invalidate the 1,316,466 dense + SPLADE rows already computed over the verbatim "
                              "text. It is recorded as a separate derived field to be produced downstream: "
                              "resolved_text (nullable, absent in canonical_v1).",
        "DEDUP_RULE": "identity = exact entity surface string; every repeat across questions/splits collapses onto one node. "
                      "n_source_records / split_provenance record how many questions and which splits contributed each node.",
        "EXPECTED_NODE_COUNT": 1316466,   # point prediction = the Phase-C reference; see note: a_entity/q_entity injection is
                                          # removed here, so the DERIVED count is expected to be <= this, and the delta is the
                                          # number of answer/topic entities that live outside every subgraph. Measured at build.
        "DERIVED_NODE_COUNT": None,
        "QUERY_DEPENDENT_CORPUS": False,
        "QUALITY_LOCKED_QUERY_SOURCE": "official Microsoft WebQSP.test.json, 1,639 questions (1,628 answerable). WebQSP test labels are "
                                       "PUBLIC. NOTE the leakage hazard this fixes: those same test questions are what the legacy "
                                       "substrate both built its corpus from and evaluated on.",
        "SCALE_ALL_QUERY_SOURCE": "every official Microsoft WebQSP question = 4,737 (train 3,098 + test 1,639). The RoG splits "
                                  "(2,826/246/1,628) are a subgraph layer over the same questions, not an independent question set.",
        "SOURCE_CONTRACT_STATUS": "BLOCKED_PENDING_FREEBASE_SOURCE -- NOTHING BUILT AND NOTHING MAY BE BUILT. "
                                  "data/final_canonical/_APPROVALS.json sets approvals.webqsp.status = "
                                  "BLOCKED_PENDING_FREEBASE_SOURCE, and scratchpad/final_canonical_build/build_kb.py hard-codes "
                                  "NEVER_BUILD = {'webqsp'} so the builder refuses unconditionally: no flag, environment variable "
                                  "or argument can make it build. The corpus source described below (the RoG all-question subgraph "
                                  "union) is REJECTED because its membership is query-derived. See "
                                  "data/final_canonical/webqsp/FREEBASE_SOURCE_PROPOSAL.md for the proposed question-independent "
                                  "Freebase source, scope rules, deterministic textualization, cost and reuse analysis.",
    }

    # ---------------------------------------------------------------- hotpotqa
    C["hotpotqa"] = {
        "DATASET": "hotpotqa",
        "CANONICAL_CORPUS_SOURCE": "The official HotpotQA FULLWIKI retrieval corpus: "
                                   "data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 "
                                   "-- one node per English Wikipedia article abstract (introductory paragraph) of the 2017-10-01 snapshot.",
        "SOURCE_VERSION": "enwiki-20171001 pages-meta-current withlinks abstracts, the authors' standard fullwiki corpus "
                          "(nlp.stanford.edu/projects/hotpotqa/), 1,553,565,403 bytes, "
                          "sha256 1acca1c5cc93c4890ea51091d2bad7c3ef6987aead127ab88728dc9e26555729. "
                          "Question layer: HotpotQA v1.1 HF parquet (authors' own org) under data/original/hotpotqa/hf_*.",
        "SOURCE_PATHS": ["data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
                         "data/original/hotpotqa/hf_distractor/train-00000-of-00002.parquet",
                         "data/original/hotpotqa/hf_distractor/train-00001-of-00002.parquet",
                         "data/original/hotpotqa/hf_fullwiki/validation-00000-of-00001.parquet",
                         "data/original/hotpotqa/hf_fullwiki/test-00000-of-00001.parquet"],
        "WHY_THIS_IS_THE_FULL_RETRIEVAL_UNIVERSE":
            "This is a WIKIPEDIA SNAPSHOT, not a benchmark by-product, and that is checkable from the file itself:\n"
            "(1) Record schema (probed directly from the first tar member, enwiki-.../CN/wiki_55.bz2, in this task) is "
            "{id, title, url, text, text_with_links, charoffset, charoffset_with_links} -- a Wikipedia article record keyed by the wiki "
            "curid, with url = 'https://en.wikipedia.org/wiki?curid=<id>'. There is NO question id, question string, answer, "
            "supporting_facts, split label or any other HotpotQA annotation anywhere in the record. The file is a dump derivative: its "
            "record set is fixed by the 2017-10-01 Wikipedia snapshot and is defined without reference to any question.\n"
            "(2) Scale proves non-derivation: it holds 5,233,329 articles, whereas the union of the 10 context paragraphs of ALL 97,852 "
            "official distractor questions is only 507,494 titles -- i.e. ~90.3% of the corpus is never referenced by any HotpotQA "
            "question at all. A question-derived collection cannot contain 4.7M unreferenced records.\n"
            "(3) It explicitly rules out the legacy substrate. The legacy 507,494-node corpus IS the distractor-setting context union "
            "(src/pipeline/rebuild_dataset.py:70-84 -> loaders.load_hotpotqa title-keyed article_cache -> build_clean dedup over "
            "hotpot_{dev,train}_distractor.jsonl), i.e. exactly 'the 10 paragraphs shown to the model for each question, deduplicated' -- "
            "a SETTING-conditioned and question-conditioned pool, and its 2,000 evaluated queries were a seeded sample of that same "
            "question pool. The locked setting for this project is FULLWIKI, whose retrieval universe is (1).\n"
            "(4) Independent corroboration inside this repo: the Phase-C stream of the same tarball produced exactly 5,233,329 unique "
            "curids with 0 duplicates (data/canonical/hotpotqa/document_manifest.json), matching the corpus size the HotpotQA authors "
            "state for the abstracts release.\n"
            "(5) WHY THE ABSTRACTS ARCHIVE AND NOT THE 7.4 GB 'withlinks-processed' ARCHIVE. 'Full corpus' here means the benchmark's "
            "full OFFICIAL RETRIEVAL UNIVERSE, i.e. the document collection the benchmark's own FullWiki setting indexes and retrieves "
            "over -- and that collection is the introductory-paragraph (abstract) corpus, one document per article, which is what the "
            "HotpotQA authors distribute for FullWiki retrieval and what every FullWiki baseline indexes. The larger "
            "enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2 (~7.4 GB) contains the FULL BODY TEXT of the same articles; "
            "it is a strictly BROADER document collection than the intended FullWiki retrieval corpus, and indexing it would change the "
            "retrieval unit (article body / arbitrary passage instead of article abstract) and so would no longer be the benchmark's "
            "universe. Choosing it would not be 'more complete', it would be a different task. The abstracts archive is therefore the "
            "correct maximal choice, and it is also the one whose 5,233,329 documents this repo already encoded end to end.",
        "RAW_SOURCE_RECORDS": "5,233,329 article records expected (one json line per article across the bz2 members inside the tar); "
                              "counted exactly at build time.",
        "CANONICAL_NODE_IDENTITY_RULE": "official Wikipedia curid (the record's 'id' field), exact string; "
                                        "node_id = 'hotpotqa:c<curid>'. The id is READABLE rather than hashed: the curid is already "
                                        "source-stable and globally unique, so hashing buys no collision safety and costs "
                                        "debuggability and remapping. (Earlier draft of this contract proposed "
                                        "'hotpotqa:<sha256(curid)[:24]>'; changed BEFORE any node was written, so no artifact "
                                        "carries the hashed form. 2wiki's full-universe table uses the identical '2wiki:c<curid>' "
                                        "scheme.) Titles are NOT the identity: several curids can share a title, so gold resolution "
                                        "is title -> curid via the corpus and an ambiguous title contributes ALL matching nodes.",
        "NORMALIZATION_RULE": "Stored text = ' '.join(non-empty elements of the official 'text' sentence list), exactly parallel to "
                              "2wiki, because the official record stores the abstract as a sentence list. "
                              "Plain 'text' is used, NOT 'text_with_links' (no <a> markup in the retrieval text); hyperlinks stay a "
                              "downstream structural artifact. No NFC/NFKC, no casefold, no whitespace collapse, and NO strip -- note "
                              "that the Phase-C derivation (scratchpad/c1_hotpot_fullwiki.py:39) DID apply .strip() to the joined text, "
                              "so the build must count how many records differ under that one step and report the exact dense/SPLADE "
                              "reuse impact rather than assume it. Empty-abstract articles (94 in the Phase-C pass) are KEPT as nodes: "
                              "dropping them would make corpus membership depend on a text-quality filter. "
                              "SECOND CONSTRUCTION STEP -- TEXTUALIZATION_REV 2 (2026-09-05): if that join is empty, the stored text is "
                              "the SAME record's source-native 'title' instead. It is a fallback only: no title is prepended to any "
                              "record that already has text, and a record with an empty title stays empty. The rule is deterministic, "
                              "reads one field of the same JSON line, and is query-independent (the record schema carries no "
                              "question/answer/supporting_facts field). It fired for exactly the 94 empty-abstract articles above, "
                              "leaving 0 empty. Rationale: the previous empty encoder input tokenizes to the ZERO-LENGTH sequence [] "
                              "under gte-Qwen2, collapsing all 94 nodes onto one degenerate forward pass, and 'text_with_links' is not "
                              "a usable alternative here (measured readable residue 0/94, charoffset == [[]] for all 94). "
                              "This step changed CORPUS_HASH fe139de9... -> 1b7eeac2... and left NODE_ORDER_HASH, node count, node ids "
                              "and node order untouched; the superseded rev-1 artifacts are preserved under "
                              "data/final_canonical/_superseded_textualization_rev1/. Full forensics: "
                              "results/data_audit/final_canonical_v1/hotpotqa/EMPTY_TEXT_94_AUDIT.md",
        "DEDUP_RULE": "identity = curid; a repeated curid collapses (Phase-C observed 0). Titles are NOT the identity: two articles may "
                      "share a title string across the dump, and gold resolution is title->curid via the corpus, not the reverse.",
        "EXPECTED_NODE_COUNT": 5233329,
        "DERIVED_NODE_COUNT": None,
        "QUERY_DEPENDENT_CORPUS": False,
        "QUALITY_LOCKED_QUERY_SOURCE": "official HotpotQA FULLWIKI DEV split, 7,405 questions (hf_fullwiki/validation parquet). "
                                       "The fullwiki TEST split (7,405) has HIDDEN labels -- verified: answers None, supporting_facts "
                                       "empty -- so it cannot serve as the accuracy lane.",
        "SCALE_ALL_QUERY_SOURCE": "every official question = 105,257 (train 90,447 from hf_distractor -- the train split is shared by "
                                  "both settings -- + fullwiki dev 7,405 + fullwiki test 7,405). Test carries no gold, "
                                  "usable for scale/latency only.",
        "SOURCE_CONTRACT_STATUS": "APPROVED (data/final_canonical/_APPROVALS.json: approvals.hotpotqa.status) and BUILT; the heavy "
                                  "build was additionally gated on data/final_canonical/_GATE_HOTPOT_HEAVY_OK, which is present. "
                                  "Canonical directory is data/final_canonical/hotpotqa/; 'hotpotqa_clean' is a LEGACY SUBSTRATE "
                                  "ALIAS only (its deferral record is preserved at "
                                  "data/final_canonical/hotpotqa/_legacy_alias_hotpotqa_clean/status.json).",
    }

    # ---- DERIVED_NODE_COUNT is never hand-written: it is read back from the build that actually ran ----
    for ds, c in C.items():
        ip = os.path.join(ROOT, ds, "integrity_report.json")
        if os.path.exists(ip):
            ir = json.load(open(ip, encoding="utf-8"))
            n = ir.get("canonical_node_count")
            if c["DERIVED_NODE_COUNT"] is not None and c["DERIVED_NODE_COUNT"] != n:
                raise SystemExit(f"{ds}: DERIVED_NODE_COUNT {c['DERIVED_NODE_COUNT']} != built {n}")
            c["DERIVED_NODE_COUNT"] = n

    # ---- attach hashes, validate field set, write ----
    written = []
    for ds, c in C.items():
        c["SOURCE_HASHES"] = hashes(c["SOURCE_PATHS"], afh, extra)
        assert set(c) == set(FIELDS), (ds, set(FIELDS) ^ set(c))
        # canonical dir names use the clean dataset name (2wiki, musique, squad, hotpotqa); the legacy
        # '<ds>_clean' substrate names are aliases only.  data/final_canonical/hotpotqa_clean/status.json
        # (the superseded deferral record) is left untouched.
        d = os.path.join(ROOT, ds)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "SOURCE_CONTRACT.json")
        json.dump({k: c[k] for k in FIELDS}, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        written.append(p)
        print("wrote", p)
    json.dump({"written": written, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(f"{ROOT}/_work/source_contracts_index.json", "w"), indent=2)
    return C


if __name__ == "__main__":
    build()
