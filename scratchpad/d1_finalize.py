"""Fold tightened-rule details into dataset_manifest.json and write the final availability table."""
import json
OUT="results/data_audit"
m=json.load(open(f"{OUT}/dataset_manifest.json"))
hyp=json.load(open(f"{OUT}/2wiki_hyperlink_stats.json"))

# WebQSP tightened
m["datasets"]["webqsp"]["zip_sha256"]="95cb9cd2f6b4e1116ba4bec348f08a7f5ca2a5df4f223ea5af3f9b25a971b0ee"
m["datasets"]["webqsp"]["zip_bytes"]=4383010
m["datasets"]["webqsp"]["fully_parsed"]={"train":3072,"test":1628}
m["datasets"]["webqsp"]["partial_annotation_file"]={"train":680,"test":393,"total":1073}
m["datasets"]["webqsp"]["canonical_question_source"]="Microsoft WebQSP.zip ONLY; RoG parquet = subgraph layer, NOT canonical questions"

# 2wiki tightened: native hyperlink graph
m["datasets"]["2wiki"]["zip_sha256"]={"data_ids_april7.zip":"95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef",
                                       "para_with_hyperlink.zip":"a585bdc3c39425446e4b2701a5f7f30051cb6c100d179322055d53dc0a71a723"}
m["datasets"]["2wiki"]["native_hyperlink_graph"]={
    "file":"para_with_hyperlink.jsonl (7,023,046,781 B uncompressed)",
    "schema":"{id,title,sentences[],mentions[{start,end,ref_url,ref_ids,sent_idx}]}",
    "article_nodes":hyp["paragraphs"],"unique_titles_per_record":True,
    "directed_hyperlink_edges":hyp["total_mentions"],"approx_unique_pairs":"~98% (measured on 4.5M-para exact pass)",
    "self_loops":hyp["self_loops"],
    "policy":"REPLACES heuristic title-mention as structural_native for 2Wiki"}

# hotpot tightened: official file inventory + corpus (NOT downloaded)
m["datasets"]["hotpotqa"]["official_question_files"]={
    "hotpot_train_v1.1.json":{"purpose":"train QA","setting":"distractor+fullwiki(shared)","compressed":"535MB","have_via":"HF parquet 90447"},
    "hotpot_dev_distractor_v1.json":{"purpose":"dev QA","setting":"distractor","compressed":"44MB","have_via":"HF parquet 7405"},
    "hotpot_dev_fullwiki_v1.json":{"purpose":"dev QA","setting":"fullwiki","compressed":"45MB","have_via":"HF parquet 7405"},
    "hotpot_test_fullwiki_v1.json":{"purpose":"test QA (hidden)","setting":"fullwiki","compressed":"46MB","have_via":"HF parquet 7405 (answers None)"}}
m["datasets"]["hotpotqa"]["fullwiki_corpus_options"]={
    "abstracts":{"file":"enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
                 "url":"https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
                 "compressed":"1.55 GB","est_extracted":"~6-8 GB","fields":"text_with_links + charoffset_with_links (native hyperlinks)",
                 "note":"STANDARD fullwiki retrieval corpus (intro paragraphs, ~5.2M articles)"},
    "full_withlinks":{"file":"enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2",
                 "url":"https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2",
                 "compressed":"7.4 GB","est_extracted":"~30-40 GB","note":"all paragraphs + hyperlinks (superset)"},
    "host_note":"CMU curtis host DOWN; Stanford nlp.stanford.edu host UP",
    "local_corpus_provenance":"current local hotpot corpus = 507,494 passages = DISTRACTOR-context-derived, NOT fullwiki abstracts (~5.2M). Reuse only valid for distractor setting.",
    "status":"HELD — report-before-download per user rule"}

m["_tightened_rules_applied"]="2026-08-23: WebQSP canonical=MS-only; 2Wiki native hyperlink graph measured; Hotpot corpus reported not downloaded; graph policy G_main=structural+NER, kNN=ablation-only"
json.dump(m,open(f"{OUT}/dataset_manifest.json","w"),indent=2)
print("manifest updated")
print("2wiki native graph:", hyp["paragraphs"],"nodes,",hyp["total_mentions"],"edges,",hyp["self_loops"],"self-loops")
