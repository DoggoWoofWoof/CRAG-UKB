"""Assemble Phase-D0 current_inventory.{json,md} from the read-only facts gathered.
No mutation of any data artifact."""
import json, os
OUT = "results/data_audit"; os.makedirs(OUT, exist_ok=True)

# Official anchors (user-supplied; each corroborated EXACTLY by a local official raw file where present).
OFFICIAL = {
  "webqsp": {"paper":"Yih et al. 2016, The Value of Semantic Parse Labeling for KBQA (ACL)",
             "official_repo":"https://aka.ms/WebQSP (Microsoft WebQuestionsSP)",
             "official_splits":{"train":3098,"dev":None,"test":1639},
             "derived_variants":{"GraftNet/NSM":{"train":2848,"dev":250,"test":1639},
                                 "RoG-webqsp (rmanluo/RoG-webqsp)":{"train":2826,"dev":246,"test":1628}},
             "test_labels":"public"},
  "metaqa": {"paper":"Zhang et al. 2018, Variational Reasoning over Knowledge Graphs (AAAI)",
             "official_repo":"https://github.com/yuyuz/MetaQA",
             "official_splits":{"1hop":{"train":96106,"dev":9992,"test":9947},
                                "2hop":{"train":118980,"dev":14872,"test":14872},
                                "3hop":{"train":114196,"dev":14274,"test":14274}},
             "kb":"kb.txt subject|relation|object", "test_labels":"public"},
  "2wiki":  {"paper":"Ho et al. 2020, Constructing A Multi-hop QA Dataset (COLING)",
             "official_repo":"https://github.com/Alab-NII/2wikimultihop",
             "official_splits":{"train":167454,"dev":12576,"test":12576},
             "test_labels":"HIDDEN (test answers withheld)"},
  "musique":{"paper":"Trivedi et al. 2022, MuSiQue: Multihop Questions via Single-hop QG (TACL)",
             "official_repo":"https://github.com/StonyBrookNLP/musique",
             "official_splits_ans":{"train":19938,"dev":2417,"test":2459},
             "test_labels":"HIDDEN"},
  "hotpot": {"paper":"Yang et al. 2018, HotpotQA (EMNLP)",
             "official_repo":"https://hotpotqa.github.io / https://github.com/hotpotqa/hotpot",
             "official_splits":{"train":90447,"dev":7405,"test":7405},
             "settings":["distractor","fullwiki"],
             "count_note":"local train_v1.1 = 90447 (canonical). Some community loaders report 90564; verify in D2.",
             "test_labels":"HIDDEN"},
  "squad":  {"paper":"Rajpurkar et al. 2018, Know What You Don't Know: SQuAD 2.0 (ACL)",
             "official_repo":"https://rajpurkar.github.io/SQuAD-explorer/",
             "official_splits":{"train":130319,"dev":11873},
             "note":"NO public labeled test; dev is the public eval set. SQuAD2.0 includes UNANSWERABLE.",
             "test_labels":"HIDDEN"},
}

# Local RAW files present (measured this session).
RAW_LOCAL = {
  "webqsp": {"files":{"data/raw/full/kb/webqsp_test0.parquet":814,"data/raw/full/kb/webqsp_test1.parquet":814},
             "interpretation":"RoG-webqsp TEST ONLY (814+814=1628). NO official/RoG train or dev parquet present.",
             "columns":["id","question","answer","q_entity","a_entity","graph","choices"]},
  "metaqa": {"files":{"1hop":{"train":96106,"dev":9992,"test":9947},
                      "2hop":{"train":118980,"dev":14872,"test":14872},
                      "3hop":{"train":114196,"dev":14274,"test":14274}},
             "kb_txt_triples":134741,
             "interpretation":"COMPLETE official MetaQA (all hops, all splits) — EXACT match to README."},
  "2wiki":  {"files":{"data/raw/full/2wiki_dev.jsonl":12576,"data/raw/review_public/2wiki_dev.parquet":"~12576",
                      "data/raw/full/hipporag/2wikimultihopqa.json":1000,
                      "data/raw/full/hipporag/2wikimultihopqa_corpus.json":6119},
             "interpretation":"Only official DEV (12576) present. Official TRAIN (167454) ABSENT. hipporag json = 1000-q HippoRAG eval subset (NOT official split)."},
  "musique":{"files":{"data/raw/musique.jsonl":19938,"data/raw/full/musique_ans_dev.jsonl":2417,
                      "data/raw/review_public/musique_v1.0/musique_ans_v1.0_train.jsonl":"19938 (full)",
                      "data/raw/review_public/musique_v1.0/musique_ans_v1.0_dev.jsonl":2417,
                      "data/raw/review_public/musique_v1.0/musique_ans_v1.0_test.jsonl":"2459 (labels hidden)",
                      "data/raw/review_public/musique_v1.0/musique_full_v1.0_train.jsonl":"present (Full variant)",
                      "data/raw/full/hipporag/musique.json":1000},
             "interpretation":"COMPLETE official MuSiQue-Ans (train 19938, dev 2417, test 2459-hidden) AND MuSiQue-Full present. hipporag json = 1000-q subset."},
  "hotpot": {"files":{"data/raw/review_public/hotpot_train_distractor.jsonl":90447,
                      "data/raw/review_public/hotpot_dev_distractor.jsonl":7405,
                      "data/raw/hotpotqa_dev.jsonl":7405,
                      "data/raw/full/hipporag/hotpotqa.json":1000,
                      "data/raw/full/hipporag/hotpotqa_corpus.json":9811},
             "interpretation":"COMPLETE official HotpotQA-DISTRACTOR (train 90447, dev 7405). FULLWIKI Wikipedia corpus NOT present. hipporag json = 1000-q subset (its own 9811-doc mini-corpus)."},
  "squad":  {"files":{"data/raw/squad_v2.json":130319,"data/raw/full/squad_dev.json":11873},
             "versions":{"data/raw/squad_v2.json":"v2.0 (442 articles)","data/raw/full/squad_dev.json":"v2.0 (35 articles)"},
             "interpretation":"COMPLETE official SQuAD 2.0 train (130319) + dev (11873) — EXACT match. Includes unanswerable."},
}

# Current CANONICAL feature dumps (results/L2/relsig_feats_{ds}.pt) — from d0_feat_inventory.json.
FEAT = json.load(open(f"{OUT}/feat_inventory.json"))
FEAT_SUMMARY = {ds: {"train": v.get("splits",{}).get("train",{}).get("n_queries"),
                     "dev": v.get("splits",{}).get("dev",{}).get("n_queries"),
                     "test": v.get("splits",{}).get("test",{}).get("n_queries"),
                     "feat_names": v.get("feat_names")}
                for ds, v in FEAT.items()}

CORPUS = {"webqsp":781485,"metaqa":40151,"2wiki_clean":65865,"musique_clean":13672,
          "hotpotqa_clean":507494,"squad_clean":19029}

CAP_SOURCE = {"file":"src/experiments/kg_relsig.py",
              "signature":"_run(... tr_cap=1104, te_cap=2000 ...); d=_load(dataset,'gte_qwen',8000,tr_cap,te_cap)",
              "effect":"EVERY relsig_feats_{ds}.pt was built with a silent train cap 1104 and test cap 2000 (webqsp smaller because its question pool itself is small). None is uncapped; none has a dev split."}

GRAPH_PROV = {
  "storage":"data/ukb_storage/{ds}/graph.pt as a torch_geometric Data with a SINGLE node.neighbors adjacency (no edge_type field).",
  "build_clean_note":"build_clean.py: 'doc.neighbors = LABEL-FREE title-mention links only; the indexer adds its own kNN edges on top.'",
  "families_present_but_COLLAPSED":{
     "KB triples (official)":"webqsp (Freebase triples from RoG graphs), metaqa (kb.txt triples)",
     "derived title/entity-mention":"2wiki/musique/hotpot/squad via graph_builder.enrich_with_title_links",
     "synthetic gte-kNN":"added on top for ALL datasets by the indexer (build_all)",
     "NER shares-entity (weighted)":"ner_edges_w_df25.pkl present for metaqa/2wiki/musique/squad (separate file, df-weighted)"},
  "issue":"Official + derived + synthetic edges are merged into one UNLABELED adjacency. No per-edge provenance. Violates Phase-D5 requirement."}

inv = {"_generated":"Phase D0 inventory (read-only)","date":"2026-08-23",
       "official_anchors":OFFICIAL,"raw_local":RAW_LOCAL,
       "canonical_feature_dumps":FEAT_SUMMARY,"corpus_doc_counts":CORPUS,
       "silent_cap_source":CAP_SOURCE,"graph_provenance":GRAPH_PROV}
json.dump(inv, open(f"{OUT}/current_inventory.json","w"), indent=2)
print("wrote current_inventory.json")
print(json.dumps(FEAT_SUMMARY, indent=1))
