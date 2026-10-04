"""Phase 8 -- DEFERRED status records for webqsp and hotpotqa_clean (NO building).

    python scratchpad/final_canonical_build/deferred.py

Writes data/final_canonical/webqsp/status.json and data/final_canonical/hotpotqa_clean/status.json.
Every number below is read from existing artefacts (cache meta_json, query_ids_all.json, canonical manifests,
_work/webqsp_legacy_verify.json) -- nothing is rebuilt or downloaded.
"""
import os, json, time, collections, re
import numpy as np

ROOT = "data/final_canonical"


def cache_meta(ds):
    z = np.load(f"results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_{ds}.npz", allow_pickle=True)
    return json.loads(str(z["meta_json"])), int(len(z["hard"])), int(len(z["rows"]))


def qids_summary(ds):
    j = json.load(open(f"data/ukb_storage/{ds}/gte_qwen/query_ids_all.json", encoding="utf-8"))
    ids = j["ids"]; si = j["split_indices"]
    pref = collections.Counter(re.sub(r"[-_]?[0-9a-f]+$", "", i) for i in ids)
    return {"n_question_ids": len(ids), "split_sizes": {k: len(v) for k, v in si.items()}, "id_prefixes": dict(pref.most_common(5))}


def webqsp():
    meta, n_hard, n_rows = cache_meta("webqsp")
    ver = json.load(open(f"{ROOT}/_work/webqsp_legacy_verify.json", encoding="utf-8"))
    canon = json.load(open("data/canonical/webqsp/document_manifest.json", encoding="utf-8"))
    q = qids_summary("webqsp")
    same = all(v.get("identical_to") for k, v in ver.items() if k.startswith("data/raw"))
    st = {
        "dataset": "webqsp", "status": "DEFERRED_FULLWIKI_CANONICALIZATION", "dataset_version": None,
        "current_substrate_is_final_canonical": False,
        "statement": "The current webqsp experimental substrate (781,485 nodes) is NOT a final canonical corpus. No canonical_v1 build was attempted in this track.",
        "legacy_substrate": {
            "cache": "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_webqsp.npz",
            "n_docs_from_cache": n_hard, "meta_n_docs": meta["n_docs"], "n_eval_queries": n_rows, "sample_rule": meta["sample_rule"],
            "master": "data/processed/master_nodes_webqsp.json",
            "what_it_actually_is": (
                "Freebase entity nodes of the RoG-webqsp per-question subgraphs of the RoG TEST split ONLY. Verified: "
                "src/pipeline/loader_webqsp.py builds master_nodes_webqsp.json from --paths defaulting to data/raw/full/kb/webqsp_test0.parquet "
                "+ webqsp_test1.parquet (lines 118-123); those two files are sha256-identical to data/original/webqsp/rog_webqsp/test-0000{0,1}-of-00002.parquet"
                + (" (VERIFIED IDENTICAL)" if same else " (NOT identical -- see verify file)") +
                f"; recounting the unique triple endpoints of the RoG test parquets gives {ver['rog_test_unique_entities']} entities over {ver['rog_test_questions']} questions "
                f"(legacy n_docs = {meta['n_docs']}); and every legacy question id is WebQTest-* ({q['id_prefixes']}). "
                "The 'val+train' in sample_rule refers to the substrate's INTERNAL seeded 70/20/10 re-split of those RoG test questions "
                "(src/experiments/overlap_retrain._splits fallback), not to official WebQSP train. 'TEST never touched' therefore means the internal 10% cut."),
            "legacy_question_ids": q,
            "classification": "SAMPLE_CONDITIONED (corpus = union of the evaluated questions' own subgraphs; also STALE vs the RoG train+val+test entity union)",
            "brief_lead_correction": "The brief's lead ('RoG train+val subgraph nodes only, built before the RoG test parquet') is NOT what the code shows: "
                                     "the legacy master is built from the RoG TEST parquets; it predates the acquisition of the RoG train/validation parquets (2026-08-23).",
            "verify_file": f"{ROOT}/_work/webqsp_legacy_verify.json",
        },
        "official_or_canonical_reference": {
            "phase_c_canonical": "data/canonical/webqsp/documents.jsonl", "phase_c_n_docs": canon["n_docs"],
            "phase_c_definition": canon["doc_unit"] + " -- " + canon["provenance"],
            "official_questions": "Microsoft WebQSP.train.json (3098) / WebQSP.test.json (1639); RoG parquet = subgraph layer (train 2826 / dev 246 / test 1628)",
            "open_decision": "canonical corpus definition for a KB dataset (RoG subgraph-union entities vs. a fixed Freebase slice) is a user decision; "
                             "the RoG union (1,316,466) is itself a union of per-question subgraphs (question-conditioned by construction of RoG)."},
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(f"{ROOT}/webqsp", exist_ok=True)
    json.dump(st, open(f"{ROOT}/webqsp/status.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return st


def hotpot():
    meta, n_hard, n_rows = cache_meta("hotpotqa_clean")
    canon = json.load(open("data/canonical/hotpotqa/document_manifest.json", encoding="utf-8"))
    q = qids_summary("hotpotqa_clean")
    st = {
        "dataset": "hotpotqa_clean", "status": "DEFERRED_FULLWIKI_CANONICALIZATION", "dataset_version": None,
        "current_substrate_is_final_canonical": False,
        "statement": "The current hotpotqa_clean experimental substrate (507,494 nodes) is NOT a final canonical corpus for the locked FULLWIKI setting. "
                     "No canonical_v1 build was attempted in this track (fullwiki abstracts file is 3.9 GB and out of scope).",
        "legacy_substrate": {
            "cache": "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/runs/cache_hotpotqa_clean.npz",
            "n_docs_from_cache": n_hard, "meta_n_docs": meta["n_docs"], "n_eval_queries": n_rows, "sample_rule": meta["sample_rule"],
            "master": "data/processed/master_nodes_hotpotqa_clean.json",
            "what_it_actually_is": (
                "The HotpotQA DISTRACTOR-setting corpus: union of the 10 context paragraphs of every official distractor train (90,447) + dev (7,405) "
                "question, deduplicated by article title. Verified from code: src/pipeline/rebuild_dataset.py:70-84 rebuild_hotpotqa merges "
                "data/raw/review_public/hotpot_{dev,train}_distractor.jsonl -> loaders.load_hotpotqa (title-keyed article_cache, lines 91-134) -> "
                "build_clean('hotpotqa') (title, md5) dedup -> master_nodes_hotpotqa_clean.json. Legacy question pool = "
                f"{q['n_question_ids']} ids (= 90,447 + 7,405), internally re-split 70/20/10 ({q['split_sizes']}); the eval 2,000 are a seeded "
                "sample of that internal 'val' cut, i.e. official train+dev distractor questions."),
            "legacy_question_ids": q,
            "classification": "SETTING_CONDITIONED (distractor contexts, not fullwiki) + SAMPLE_CONDITIONED (union of the question pool's own contexts); "
                              "not a random sample of the fullwiki corpus",
        },
        "official_or_canonical_reference": {
            "locked_setting": "FULLWIKI", "phase_c_canonical": "data/canonical/hotpotqa/documents.jsonl", "phase_c_n_docs": canon["n_docs"],
            "phase_c_definition": canon["doc_unit"] + " -- " + canon["provenance"],
            "official_fullwiki_questions": "dev 7,405 (labels public) / test 7,405 (labels hidden); train 90,447 shared with distractor",
            "note": "a fullwiki canonical_v1 would be a streaming re-derivation of the 5,233,329-abstract corpus (id = wiki curid) plus gold title->curid resolution; "
                    "deferred by the parent's instruction (3.9 GB file, RAM budget)."},
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(f"{ROOT}/hotpotqa_clean", exist_ok=True)
    json.dump(st, open(f"{ROOT}/hotpotqa_clean/status.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return st


if __name__ == "__main__":
    a = webqsp(); b = hotpot()
    print(json.dumps({"webqsp": a["legacy_substrate"]["classification"], "hotpotqa_clean": b["legacy_substrate"]["classification"]}, indent=1))
