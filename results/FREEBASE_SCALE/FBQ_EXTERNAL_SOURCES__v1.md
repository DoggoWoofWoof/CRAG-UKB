# FBQ_EXTERNAL_SOURCES v1 -- Freebase-MID-keyed question sources that could extend the FBQ_SET

READ-ONLY web survey (2026-10-04). **Nothing was downloaded.** Purpose: let the user APPROVE specific downloads (file, source, size). Verification flags: **V** = read from the host page/API this session; **S** = secondary summary, confirm; **U** = unverified.

## Local evidence (what resolves in our tree)

- Tree: freebase_v3 (301,977,131 nodes) built from the IDIR-lab Zenodo Freebase release (record 7909511, data/final_canonical/freebase_v3/V3_ACQUISITION_RECORD.json); WebQSP's own FreebaseVersion field = 2015-08-09
- NSM WebQSP (3,098 q): topic MID occurrences 100.0% (2,915/2,915 train + 254/254 dev), answer MID occurrences 99.10% (34,409/34,721 = train 31,450/31,731 + dev 2,959/2,990; plus 115 literal answers), questions fully mappable 95.03% (2,944/3,098)
- NSM CWQ (31,158 q): topic MID occurrences 98.97% (47,079/47,568), answer MID occurrences 99.85% (69,636/69,741), questions fully mappable 98.13% (30,575/31,158)
- Unresolved ids: overwhelmingly g.* guids (WebQSP answers 207/296 unique unresolved are g.*, CWQ topics 196/212, CWQ answers 18/26): the cleaned tree lacks most g.* guids, so a source with many g.* ids resolves worse
- CWQ ids are `<WebQSP parent id>_<hash>`: 100% of CWQ train/dev rows whose parent is WebQTrn-* share a topic/answer MID with the parent (random-parent baseline 0.8%); 9,469 CWQ train+dev rows have a WebQTest-* parent (FBQ_SET__v1__PARENT_CHECK.json).

## Recommended downloads (for approval; none started)

| # | file | source | size | note |
|---|---|---|---|---|
| 1 | FreebaseQA-train.json + FreebaseQA-dev.json | https://raw.githubusercontent.com/kelvin-jiang/FreebaseQA/master/ (GitHub) | 28,544,438 B (27.2 MiB) | 23,888,089 + 4,656,349 bytes; do NOT take FreebaseQA-eval.json (4,660,561) or FreebaseQA-partial.json (15,950,670); CC-BY-4.0; +24,352 q |
| 2 | GrailQA dataset archive | https://dl.orangedox.com/WyaCpL/ (GrailQA homepage) | ~120 MB (stated, exact bytes unverified) | '120 MB' as stated on the homepage (exact bytes unverified); use train+dev only (+51,100 q); the test questions in the archive are label-masked and stay sealed; CC BY-SA 4.0 |
| 3 | ComplexWebQuestions_train.json + ComplexWebQuestions_dev.json (original release) | Dropbox links in the HF loader https://huggingface.co/datasets/drt/complex_web_questions/blob/main/complex_web_questions.py | unverified | sizes unverified; metadata enrichment only (compositionality_type, SPARQL, webqsp_ID, aliases); NOT the test file |
| 4 | graphquestions_v1_fb15_training_091420.json (optional) | https://github.com/dki-lab/GrailQA/tree/main/data | 17,995,362 B (17.2 MiB) | only if its answers turn out to be MIDs; skip the test file (10,390,906 bytes) |
| 5 | SimpleQuestions_v2.tgz (optional probe) | Meta bAbI Dropbox link / figshare 5812818 (539,411,811 bytes, V) | unverified | ~423 MB (S) for the tgz; only for a single-hop volume probe; resolve 1,000 rows first |

Firm total for priorities 1+2: ~148.5 MB (FreebaseQA train+dev 28.5 MB + GrailQA 120 MB stated).

## Usable Freebase-MID-keyed candidates

### WebQSP (Microsoft original, v1.0)

- MID-keyed: YES (TopicEntityMid, Answers.AnswerArgument, SPARQL, InferentialChain)
- Official: https://www.microsoft.com/en-us/download/details.aspx?id=52763
- License: not stated on the download page (V)
- Counts: 4,737 fully parsed questions (train 3,098 [local WebQSP.train.json] + test 1,639 [NSM test_simple newline count]) + 1,073 'partial' (page V); local WebQSP.train.partial.json = 680 questions, 660 Partial (topic MID, NO answers), only 20 Complete with answers, none overlapping train.json
- Answer format: Freebase MIDs + names + SPARQL + chain
- Files (host-reported size):
  - `WebQSP.zip` -- 4.2 MB (download page V); local copy data/original/webqsp/WebQSP.zip = 4,383,010 bytes (V, ls) -- [ALREADY LOCAL]
- Freebase snapshot: 2015-08-09 (FreebaseVersion field of the local WebQSP.train.json, V)
- Expected resolution in our tree: measured 100% topic / 99.10% answer (via NSM, same MIDs)
- Overlap/leakage: this IS WebQSP (NSM/RoG are re-packagings)
- **Recommendation:** ALREADY USED. Nothing to download. Optional (not done in v1): the 20 Complete rows in the local train.partial file are the only extra answered questions; test.partial is sealed.

### ComplexWebQuestions (Talmor & Berant 2018), original release

- MID-keyed: YES (answers[].answer_id = MID; sparql)
- Official: https://www.tau-nlp.org/compwebq (HTTP 404 at fetch time, V); project page tau-nlp.sites.tau.ac.il/compwebq (404 at fetch); code https://github.com/alontalmor/WebAsKB; HF loader https://huggingface.co/datasets/drt/complex_web_questions (data files are Dropbox shared links)
- License: Apache-2.0 on the HF card (V); the HF card 'Licensing Information: Not specified' for the data itself
- Counts: 34,689 total. HF card: train 27,734 / dev 3,480 / test 3,475. Our local NSM/RoG CWQ: 27,639 / 3,519 / 3,531 (same total 34,689; the splits differ -> NSM/RoG follow a different (repartitioned) split; we observe 0 WebQSP parents shared by NSM-train and NSM-dev)
- Answer format: MIDs + names + aliases; fields ID, webqsp_ID (explicit parent id), webqsp_question, machine_question, question, sparql, compositionality_type (composition|conjunction|comparative|superlative), composition_answer, created
- Files (host-reported size):
  - `ComplexWebQuestions_train.json` -- unverified -- [V (URL in the HF loader script), U (size)]
  - `ComplexWebQuestions_dev.json` -- unverified -- [U (size)]
  - `ComplexWebQuestions_test.json` -- unverified -- [DO NOT DOWNLOAD (sealed test; URL exists in the HF loader)]
- Freebase snapshot: not stated by the paper; inherits WebQSP (2015-08-09)
- Expected resolution in our tree: measured on the same questions via NSM: topics 98.97%, answers 99.85%, fully mappable 98.13%
- Overlap/leakage: the SAME 34,689 questions as the local NSM CWQ (no new questions); CWQ is built from WebQSP seeds: 100% of the local CWQ train/dev rows with a WebQTrn parent share a MID with it vs 0.8% for a random parent (FBQ_SET__v1__PARENT_CHECK.json)
- **Recommendation:** WORTH ADDING AS METADATA ONLY (no new questions): compositionality_type, SPARQL, explicit webqsp_ID, answer aliases for stratifying/auditing CWQ. Fetch train + dev only; size unverified (JSON of ~31k questions).

### GrailQA (Gu et al., WWW 2021)

- MID-keyed: YES (answer.answer_argument = MID for Entity answers, a literal value for Value answers; SPARQL; s-expression)
- Official: https://dki-lab.github.io/GrailQA/ (download: https://dl.orangedox.com/WyaCpL/); code https://github.com/dki-lab/GrailQA; HF loader https://huggingface.co/datasets/dki-lab/grail_qa
- License: CC BY-SA 4.0 (homepage, V); Apache-2.0 for the code repo; 'unknown' on the HF card
- Counts: 64,331 total: train 44,337 / dev 6,763 / test 13,231 (HF card + API, V). The public test split has all fields except the question text MASKED (labels not released) -> not usable. Usable with answers: 51,100 (train+dev). Dev/test are stratified i.i.d./compositional/zero-shot (the 'level' field).
- Answer format: MID (Entity) or value (Value); SPARQL; S-expression; graph_query
- Files (host-reported size):
  - `GrailQA dataset archive (inner file names unverified)` -- 120 MB (homepage text, V) -- [V (size), U (inner names)]
- Freebase snapshot: S: built on 'Freebase Commons' of the latest Freebase (86 domains, ~45M entities per the paper summary); exact dump date not stated on the pages read (2015-08-09 is the date of the last official Freebase dump but is NOT confirmed for GrailQA)
- Expected resolution in our tree: U: probably high (same final-dump lineage, our tree is the full 302M-node universe) but g.* guids would be missing; measure on a 1,000-row sample before committing
- Overlap/leakage: crowdsourced from graph queries; documented as independent of WebQSP/CWQ but no overlap audit is published -> dedupe by normalised question after download
- **Recommendation:** WORTH ADDING: largest MID-keyed set with answers (+51,100 train+dev questions, multi-relation, with generalisation levels). Use dev (6,763) as the eval-eligible part; treat the test questions as sealed.

### GraphQuestions (Su et al., EMNLP 2016) + the fb15 update

- MID-keyed: PARTIAL: original answers are canonical NAMES (spec.), grounded nodes carry MIDs, SPARQL returns ids; the fb15 update is GrailQA-style (answers likely MIDs, U)
- Official: https://github.com/ysu1989/GraphQuestions ; fb15 update inside https://github.com/dki-lab/GrailQA/tree/main/data
- License: LICENSE.txt present in the GitHub repo (1,081 bytes, V via GitHub listing); terms not read -> unverified
- Counts: 5,166 questions derived from 500 unique graph queries (heavy paraphrasing) (S); per-split counts unverified
- Answer format: original: names (+MIDs of grounded nodes, SPARQL); fb15: unverified
- Files (host-reported size):
  - `freebase13/graphquestions.training.json` -- 5,474,828 B (5.2 MiB) -- [V (GitHub API)]
  - `freebase13/graphquestions.testing.json` -- 5,430,961 B (5.2 MiB) -- [V (GitHub API)]
  - `data/graphquestions_v1_fb15_training_091420.json` -- 17,995,362 B (17.2 MiB) -- [V (GitHub API, dki-lab/GrailQA)]
  - `data/graphquestions_v1_fb15_test_091420.json` -- 10,390,906 B (9.9 MiB) -- [V (GitHub API; test = sealed, do not download)]
- Freebase snapshot: original = 'freebase13' directory (a 2013 snapshot, S); fb15 = the 2015 dump (name, S)
- Expected resolution in our tree: U (original: names only -> not directly mappable; fb15: likely high)
- Overlap/leakage: partly built on the same Freebase subset family as WebQSP; paraphrases of one graph query sit in the same file -> near-duplicate answer sets (leakage inside train/test)
- **Recommendation:** LOW PRIORITY. Only the fb15 training file (17,995,362 bytes), after GrailQA, and only if its answers are MIDs; skip the original (names only).

### FreebaseQA (Jiang, Wu, Jiang; NAACL 2019)

- MID-keyed: YES (Parses[].TopicEntityMid, Answers[].AnswersMid, InferentialChain)
- Official: https://github.com/kelvin-jiang/FreebaseQA ; HF mirror https://huggingface.co/datasets/KelvinJiang/freebase_qa
- License: CC-BY-4.0 (GitHub README, V); 'unknown' on the HF card
- Counts: 28,348 trivia-style questions: train 20,358 / dev 3,994 / eval 3,996 (V, README + HF API)
- Answer format: MIDs + answer strings + topic MID + predicate chain
- Files (host-reported size):
  - `FreebaseQA-train.json` -- 23,888,089 B (22.8 MiB) -- [V (GitHub API)]
  - `FreebaseQA-dev.json` -- 4,656,349 B (4.4 MiB) -- [V (GitHub API)]
  - `FreebaseQA-eval.json` -- 4,660,561 B (4.4 MiB) -- [V; DO NOT DOWNLOAD (the benchmark's test split = sealed)]
  - `FreebaseQA-partial.json` -- 15,950,670 B (15.2 MiB) -- [V; not needed]
- Freebase snapshot: U: not stated in the README / HF card (matched against Freebase; the authors also point to a ~2.2 GB compressed Freebase extract, not needed here)
- Expected resolution in our tree: U: unknown until sampled (trivia answers are popular entities, so m.* MIDs probably resolve)
- Overlap/leakage: sources are TriviaQA + quiz sites; no documented overlap with WebQSP/CWQ (not audited)
- **Recommendation:** WORTH ADDING (cheap): train + dev = 28,545,438 bytes (~27.2 MiB), +24,352 questions with MID answers; skip eval and partial.

### SimpleQuestions v2 (Bordes et al. 2015)

- MID-keyed: YES (subject MID, relation, object MID; ids appear as www.freebase.com/m/<id>)
- Official: Facebook/Meta bAbI downloads (SimpleQuestions_v2.tgz on Dropbox, e.g. https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz) [the research.facebook.com page redirects to ai.meta.com/resources and lists nothing readable]; alt: https://figshare.com/articles/dataset/SimpleQuestions/5812818
- License: figshare record: CC BY 4.0 (V); original tgz license unverified (U)
- Counts: 108,442 single-hop questions: train 75,910 / valid 10,845 / test 21,687 (S)
- Answer format: object MID (a single (subject, relation) can have several objects -> documented answer ambiguity, Petrochuk & Zettlemoyer 2018)
- Files (host-reported size):
  - `SimpleQuestions_v2.tgz` -- ~423 MB (S: from a downloader progress bar; host size unverified) -- [S]
  - `figshare 'SimpleQuestions' linked-data version, SimpleQuestions_v2.zip` -- 539,411,811 B (514.4 MiB) -- [V (size from figshare listing); article https://figshare.com/articles/dataset/SimpleQuestions/5812818]
- Freebase snapshot: U: FB2M / FB5M subsets of an early Freebase dump (older than the 2015 one); not confirmed
- Expected resolution in our tree: U: unmeasured (older MIDs; deleted/merged MIDs possible)
- Overlap/leakage: independent of WebQSP/CWQ; single-hop only
- **Recommendation:** OPTIONAL scale/probe set: 1-hop only (no multi-hop signal), ambiguous gold. Skip for now; if the user wants a single-hop volume probe, approve the tgz and resolve 1,000 rows first.

## Not usable / skip

| dataset | verdict and reason |
|---|---|
| CFQ (Keysers et al. 2020) | 239,357 synthetic questions with SPARQL only; the HF version uses entity placeholders (M0, M1, ...) instead of MIDs and ships SPARQL only, no gold answers (S). No gold answers -> skip. Source: https://github.com/google-research/google-research/tree/master/cfq ; https://huggingface.co/datasets/google-research-datasets/cfq (CC-BY-4.0; the HF parquet files total ~2.14 GB) |
| WebQuestions (Berant et al. 2013) | 5,810 questions; answers are names (not MIDs) and the entity is a Freebase url slug (en.<key>) -> not MID-keyed; 4,737 of the 5,810 are exactly WebQSP (the page says 4,737 full + 1,073 partial = 5,810). The 1,073 'partial' (no answers) cannot be scored. Host/size: SEMPRE page (U). Skip. |
| Free917 (Cai & Yates 2013) | 917 questions with lambda-calculus over fb:en.<key> (not MIDs, S); a key->MID step would be needed; host via the SEMPRE page, file/size U. Too small. Skip. |
| ComplexQuestions (Bao et al. 2016) | 2,100 questions built by augmenting a subset of WebQSP (S) -> overlap/leak with WebQSP; no official download host found (U). Skip. |
| TempQuestions (Jia et al. 2018) | 1,271 temporal questions drawn from Free917 / WebQuestions / ComplexQuestions (S; ACM page returned 403) -> overlap; release format/host unverified. Skip. |
| 30M Factoid QA corpus (Serban et al. 2016) | 30M machine-generated questions over Freebase (subject, relation, object MIDs) (S); Academic Torrents 529.34 MB (30MQA_1.tar.gz 315.96 MB + 30MQA_2.tar.gz 213.39 MB), BitTorrent only (S); synthetic single-hop, older Freebase. Skip unless a synthetic volume probe is wanted. Source: Academic Torrents (search '30M Factoid Question-Answer Corpus'; info-hash not recorded) |
| PathQuestion PQ/PQL (Zhou et al. 2018) | template-generated 2/3-hop questions over two Freebase subsets (S); id format (names vs MIDs) not verified. Skip. |
| RoG-webqsp / RoG-cwq, GNN-RAG, SR (SubgraphRetrievalKBQA), ReKnoS, KG-TRACES, UniK-QA, YF0808/*, camazlucas/Freebase-WebQSP-CWQ-Subgraph | all re-package the SAME WebQSP / CWQ questions: RoG-webqsp and RoG-cwq are already local (names only, no MIDs; RoG-cwq HF API: train ~2.76 GB, validation ~360.9 MB, test ~372.8 MB); SR/NSM ship the MID-keyed WebQSP/CWQ that we already use; HF 'camazlucas/Freebase-WebQSP-CWQ-Subgraph' (webqsp_subgraph.tsv 260,208,520 B, cwq_subgraph.tsv 471,465,645 B, unified_graph.tsv 542,254,271 B) is a RoG-derived subgraph dump. None adds a new question or a new MID (HF file listings V; the per-method statements for GNN-RAG/ReKnoS/UniK-QA/KG-TRACES are S/U). Skip. |
| KQA Pro, Mintaka, LC-QuAD 1.0/2.0, QALD, MetaQA | NOT usable as shipped: they are keyed to Wikidata QIDs (KQA Pro 117,970 QA on a Wikidata subset seeded from FB15k-237 entities; Mintaka 20,000 EN questions 14,000/2,000/4,000 with Wikidata entities; LC-QuAD 2.0 30,000 Wikidata+DBpedia; LC-QuAD 1.0 5,000 DBpedia; QALD-10 Wikidata/DBpedia) or to DBpedia / the WikiMovies KB (MetaQA, already a separate local substrate); none ships Freebase MIDs (S). A bridge exists only via a Freebase<->Wikidata id mapping: HF kdm-daiict/freebase-wikidata-mapping, fb_wiki_mapping.tsv 76,176,160 bytes, 2,076,868 rows, Apache-2.0 (V) -> would cover only entities with a Freebase link and needs a second download plus a resolution probe; not recommended now. |

## Caveats

- WebFetch summaries are produced by a small model and were wrong at least once (GrailQA split counts in one arXiv summary); counts quoted here come from the HF API/card or GitHub API wherever possible, otherwise flagged S/U
- No resolution rate for any external source has been measured: nothing was downloaded; the only measured rates are the NSM WebQSP/CWQ ones
- WebFetch of a PDF (N19-1028) was stored by the tool harness at C:/Users/Swastik/.claude/projects/C--Users-Swastik-Desktop-CRAG/348e519a-c28c-4463-a7fc-3fb109605d7e/tool-results/webfetch-1791063452124-1439z1.pdf (479.6 KB): a harness-side cache of a fetched page, outside the repo, not requested by this task
