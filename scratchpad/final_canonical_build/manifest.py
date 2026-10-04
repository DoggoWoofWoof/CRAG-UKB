"""Phase 7 -- per-dataset dataset_manifest.json + root MANIFEST.json (canonical_v1).

    python scratchpad/final_canonical_build/manifest.py            # all four + root

Assembles from: build_info.json, integrity_report.json, query_independence_test.json, legacy_comparison.json
(all produced by build.py / qi_test.py / legacy_compare.py) plus the per-dataset corpus definitions below.
"""
import os, json, hashlib, time, subprocess, collections

ROOT = "data/final_canonical"
VERSION = "canonical_v1"

CORPUS_DEF = {
    "metaqa": "CANONICAL_CORPUS_DEFINITION: the complete official MetaQA knowledge-base entity set = every row of "
              "data/original/metaqa/entity/kb_entity_dict.txt (43,234 entities; verified identical to the entity set "
              "appearing in kb.txt triples). One node per entity; identity = official entity index; text = official entity "
              "name (exact, no case-folding). Independent of every question file.",
    "2wiki": "CANONICAL_CORPUS_DEFINITION: the complete official 2WikiMultiHopQA v1.0 (ids_april2021) ARTICLE UNIVERSE -- every "
             "record of para_with_hyperlink.jsonl inside para_with_hyperlink.zip, one node per Wikipedia article curid; "
             "text = ' '.join(sentences). The official release describes this file as all articles/paragraphs with hyperlink "
             "information except error paragraphs, i.e. the complete retrieval corpus. A Wikipedia-snapshot derivative, not a "
             "benchmark by-product: ~93.3% of its articles are never referenced by any 2Wiki question. Independent of which "
             "queries are evaluated. SUPERSEDES the previously accepted 398,354-node pooled question-context union (preserved, "
             "not deleted, at data/final_canonical/2wiki/_superseded_context_union_398354/).",
    "musique": "CANONICAL_CORPUS_DEFINITION: the complete union of official MuSiQue-Ans v1.0 candidate paragraphs (all 20 per "
               "question, supporting AND distractor) over train + dev + test, one node per distinct (article title, "
               "paragraph_text) exact pair. Independent of selected queries and of is_supporting labels.",
    "squad": "CANONICAL_CORPUS_DEFINITION: the complete official SQuAD v2.0 context-paragraph corpus over train + dev (no "
             "public test), one node per distinct (article title, context) exact pair. Independent of which questions are evaluated.",
    "webqsp": "*** BLOCKED_PENDING_FREEBASE_SOURCE -- NO CANONICAL CORPUS EXISTS AND NONE WAS BUILT. *** The definition below is the "
              "REJECTED candidate, kept verbatim so the rejection is auditable; see data/final_canonical/webqsp/"
              "FREEBASE_SOURCE_PROPOSAL.md. REJECTED CANDIDATE: the union of the Freebase entity nodes of the RoG-webqsp per-question subgraphs over ALL "
              "THREE RoG splits (train 2,826 + validation 246 + test 1,628 = 4,700 questions), one node per distinct entity SURFACE "
              "STRING appearing as a triple endpoint. Only the 'graph' column is read at corpus time -- answer (a_entity) and topic "
              "(q_entity) entities are NOT injected, unlike the Phase-C reference. HONEST LIMIT: no question-independent Freebase "
              "corpus exists on disk or in the official release, so a WebQSP corpus is necessarily a subgraph union; taking all 4,700 "
              "questions of all splits makes it invariant to the evaluated query subset, which is what this track requires.",
    "hotpotqa": "CANONICAL_CORPUS_DEFINITION: the official HotpotQA FULLWIKI retrieval corpus -- every English Wikipedia article "
                "abstract of the 2017-10-01 snapshot (enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2), one node per "
                "article curid. A Wikipedia snapshot, not a benchmark by-product: ~90.3% of its articles are never referenced by any "
                "HotpotQA question. Independent of which questions are evaluated and of the distractor/fullwiki setting.",
}
NODE_ID_SCHEME = {
    "metaqa": "metaqa:e<official kb_entity_dict index, zero-padded to 5 digits>; file order = numeric index order",
    "2wiki": "2wiki:c<official Wikipedia curid> (readable, not hashed: the curid is already source-stable and unique, so "
             "hashing buys no collision safety and costs debuggability/remapping); file order = lexicographic node_id",
    "musique": "musique:<sha256(title + U+001F + paragraph_text)[:24]>; file order = lexicographic node_id",
    "squad": "squad:<sha256(title + U+001F + context)[:24]>; file order = lexicographic node_id",
    "webqsp": "webqsp:<sha256(entity surface string)[:24]>; file order = lexicographic node_id",
    "hotpotqa": "hotpotqa:c<official Wikipedia curid> (readable, not hashed -- identical rationale and scheme to 2wiki); "
                "file order = lexicographic node_id",
}
NORMALIZATION = {
    "stored_text": "EXACT official string; no NFC/NFKC, no case-folding, no whitespace collapsing, no stripping",
    "identity_keys": "EXACT strings (see dedup_policy); lossy alternatives are only quantified in "
                     "integrity_report.corpus_accounting_detail.identity_normalization_diagnostics, never applied",
    "construction_steps": {"2wiki": "text = ' '.join(official 'sentences' list) -- the official file stores a sentence list; "
                                    "no .strip() (the pre-existing full-universe encode applied one; the divergence is COUNTED, "
                                    "not assumed -- see corpus_accounting_detail.phase_c_divergence)",
                           "hotpotqa": "text = ' '.join(non-empty elements of the official 'text' sentence list); plain 'text', "
                                       "never the 'text_with_links' fallback; no .strip(); empty abstracts kept as nodes",
                           "metaqa": "text = entity name (no verbalization at corpus level)"},
    "encoding": "UTF-8, JSON with ensure_ascii=False, sort_keys=True, LF line endings",
}
DEDUP = {
    "metaqa": "identity = official entity index (names are unique in kb_entity_dict.txt; 0 duplicates). Legacy lowercase+underscore "
              "collapse NOT applied (it merged 3,083 distinct official entities).",
    "2wiki": "identity = official Wikipedia curid (exact string); a repeated curid would collapse onto its first occurrence "
             "(observed 0 duplicates in 5,989,847 records). Titles are NOT the identity: several curids can share a title, so "
             "gold/context resolution is title -> curid via the corpus and an ambiguous title contributes ALL matching nodes "
             "(recorded per query in gold_title_ambiguous) rather than being resolved last-wins. Text-hash identity is not used: "
             "distinct articles with identical intro text must stay distinct nodes.",
    "musique": "identity = (title, paragraph_text) exact. Alternative text-hash-only identity gives 117,533 (one paragraph text "
               "appears under two article titles: 'Adei Ad' and 'Etz Efraim'); chosen identity keeps them distinct (117,534).",
    "squad": "identity = (title, context) exact; 6 raw paragraph records are exact duplicates of another paragraph under the same "
             "title and are collapsed; text-hash identity gives the same 20,233.",
    "webqsp": "identity = exact entity surface string; every repeat of the entity across triples, questions and splits collapses onto "
              "one node (n_source_records = mention count, split_provenance = contributing RoG splits). Raw Freebase MIDs are kept "
              "verbatim as separate entities -- the ba6bd71 MID->name resolution is graph-derived and is NOT applied here.",
    "hotpotqa": "identity = official wiki curid (exact string). Titles are NOT the identity: several curids can share a title, so gold "
                "resolution is title -> curid via the corpus and an ambiguous title contributes ALL matching nodes (recorded per query "
                "in gold_title_ambiguous) rather than being silently resolved last-wins as in the Phase-C reference.",
}
LEGACY_NAME = {"metaqa": "metaqa", "2wiki": "2wiki_clean", "musique": "musique_clean", "squad": "squad_clean",
               "webqsp": "webqsp", "hotpotqa": "hotpotqa_clean"}
LEGACY_CLASS = {
    "metaqa": {"classification": "OTHER: LOSSY_IDENTITY_COLLAPSE (not sample-conditioned)",
               "evidence": "legacy corpus = all kb.txt entities after lower()+'_'->' ' normalization (src/pipeline/loaders.py:529-554): "
                           "43,234 official entities -> 40,151 legacy nodes (3,081 groups, 3,083 entities absorbed into a case-variant "
                           "sibling). Coverage of the KB is complete but entity identity is lossy; every legacy node maps to a canonical "
                           "node, 3,083 canonical entities have no legacy node. CONSEQUENCE visible on the frozen eval subset: for 46/1,998 "
                           "queries the legacy gold node (a merged case-group) maps to a different canonical entity than the exact official "
                           "answer string (e.g. KB holds both 'Good' and 'good'); canonical_v1 resolves answers by exact name, so those "
                           "queries' gold sets legitimately differ from the legacy ones (see legacy_comparison.eval_subset_gold_consistency)."},
    "2wiki": {"classification": "SAMPLE_CONDITIONED",
              "evidence": "legacy corpus = union of the 10 context paragraphs of the 15,000 FlashRAG-sampled official TRAIN questions "
                          "(data/raw/2wiki.jsonl -> loaders.load_2wiki -> build_clean), 150,000 records -> 65,865 (title, md5) nodes. "
                          "The eval subset (2,000) is a random 20% 'val' cut of those same 15,000 questions, so the corpus is a function "
                          "of the query sample. Against the canonical full article universe, 5,989,847 - 65,865 = 5,923,982 official "
                          "articles are absent from the legacy substrate (98.9%). All sampled-query golds being present does NOT make "
                          "this benign: distractor mass, partition structure and reachability all depend on the sample. "
                          "SECOND SUPERSESSION, 2026-09-05: the FIRST canonical_v1 2wiki table (398,354 nodes = the pooled "
                          "question-context union over all three official splits, corpus_hash ec7fbbe8cd783040...) was itself REJECTED "
                          "and replaced by the full para_with_hyperlink article universe, because although invariant to the EVALUATED "
                          "subset its membership was still a function of the question set -- the same defect for which HotpotQA's "
                          "507,494-node distractor-context union was rejected. The superseded artifacts are preserved (not deleted) at "
                          "data/final_canonical/2wiki/_superseded_context_union_398354/."},
    "musique": {"classification": "SAMPLE_CONDITIONED + GOLD_CONDITIONED (all-gold pool)",
                "evidence": "legacy corpus = union of ONLY the gold supporting paragraphs (question_decomposition.support_paragraph) of the "
                            "19,938 official TRAIN questions (FlashRAG train.jsonl -> loaders.load_musique lines 198-212 -> build_clean): "
                            "13,672 nodes; no distractor paragraphs, no dev/test paragraphs (117,534 - 13,672 = 103,862 absent). The corpus "
                            "is literally the gold set of the query pool."},
    "squad": {"classification": "OTHER: SPLIT_CONDITIONED_TRAIN_ONLY (not eval-sample conditioned)",
              "evidence": "legacy corpus = all 19,035 official TRAIN contexts (data/raw/squad_v2.json == train-v2.0.json, sha-identical) "
                          "-> build_clean (title, md5) dedup -> 19,029 nodes; the 1,204 official dev contexts are absent. The eval "
                          "subset is a random 20% of train questions; the corpus does not depend on that sample but excludes the "
                          "official dev split."},
    "webqsp": {"classification": "SAMPLE_CONDITIONED (corpus = union of the EVALUATED questions' own subgraphs) + STALE",
               "evidence": "legacy corpus = the Freebase entity nodes of the RoG TEST parquets ONLY (src/pipeline/loader_webqsp.py:118-123 "
                           "defaults to data/raw/full/kb/webqsp_test0.parquet + webqsp_test1.parquet, sha256-identical to the official RoG "
                           "test shards): 781,485 nodes over 1,628 test questions, and 1,419 of those same test questions were the evaluated "
                           "set -- the corpus was literally the union of the evaluated questions' own subgraphs "
                           "(data/final_canonical/_work/webqsp_legacy_verify.json; all 1,578 legacy question ids are WebQTest-*). "
                           "canonical_v1 spans all three RoG splits, which breaks that coupling. Separately, the legacy substrate's TEXT "
                           "carries the commit-ba6bd71 MID->name resolution, which canonical_v1 does not apply."},
    "hotpotqa": {"classification": "SETTING_CONDITIONED (distractor contexts, not fullwiki) + SAMPLE_CONDITIONED",
                 "evidence": "legacy corpus = the union of the 10 context paragraphs of every official DISTRACTOR train (90,447) + dev "
                             "(7,405) question, deduplicated by (title, md5) (src/pipeline/rebuild_dataset.py:70-84 -> loaders.load_hotpotqa "
                             "title-keyed article_cache -> build_clean): 507,494 nodes, i.e. exactly the paragraphs shown to the model. The "
                             "locked setting for this project is FULLWIKI, whose universe is the 5,233,329-article snapshot; 90.3% of it is "
                             "absent from the legacy substrate. The evaluated 2,000 are a seeded cut of that same question pool."},
}


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""): h.update(c)
    return h.hexdigest()


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# written by LATER phases into dataset_manifest.json; one() must carry them forward instead of clobbering them
PRESERVE = ("query_lanes", "query_lane_overlaps", "corpus_hash_equality_assertion", "legacy_substrate_aliases",
            "source_contract", "reuse_map", "dense_rows_needing_encode", "splade_rows_needing_encode")


def gold_shortfall_diagnostic(d, ir):
    """MEASURE why all_eval_gold_nodes_present is false.  Nothing is changed by this -- the flag keeps its
    strict definition (len(gold_node_ids) == len(set(gold_refs)) for every labelled question).  This only
    separates the two ways that equality can fail:
      * FEWER node ids than refs  -> a gold reference genuinely does not exist in the corpus (real coverage gap)
      * MORE  node ids than refs  -> one reference title matched several corpus nodes and ALL of them were kept
                                     (ambiguity expansion; every reference did resolve)
    Computed only when the flag is false, by streaming the query files."""
    out = {"_what": "measured breakdown of the questions where len(gold_node_ids) != len(set(gold_refs))",
           "_flag_definition_unchanged": True, "per_split": {}}
    tot = collections.Counter()
    for s in ir.get("splits_with_gold") or []:
        p = f"{d}/queries/{s}.jsonl"
        if not os.path.exists(p):
            continue
        c = collections.Counter()
        amb_titles = collections.Counter()
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                refs = set(map(str, r.get("gold_refs") or []))
                if not refs:
                    continue
                g = r.get("gold_node_ids") or []
                if len(g) == len(refs):
                    continue
                if len(g) > len(refs):
                    c["more_nodes_than_refs_AMBIGUOUS_TITLE_EXPANSION"] += 1
                    for t in (r.get("gold_title_ambiguous") or []):
                        amb_titles[t] += 1
                else:
                    c["fewer_nodes_than_refs_REAL_MISSING_GOLD"] += 1
        out["per_split"][s] = {**c, "ambiguous_titles_involved": dict(amb_titles)}
        tot.update(c)
    out["totals"] = dict(tot)
    out["ALL_SHORTFALL_IS_AMBIGUITY_EXPANSION"] = tot["fewer_nodes_than_refs_REAL_MISSING_GOLD"] == 0
    out["cross_check_unresolved_gold_refs"] = ir.get("unresolved_gold_refs", {}).get("counts")
    return out


def one(ds):
    d = f"{ROOT}/{ds}"
    bi = load(f"{d}/build_info.json"); ir = load(f"{d}/integrity_report.json")
    qi = load(f"{d}/query_independence_test.json"); lc = load(f"{d}/legacy_comparison.json")
    qs = ir["query_stats"]; ev = ir.get("eval_subset") or {}
    acc = ir["corpus_accounting_detail"]
    man = {
        "dataset_name": ds, "dataset_version": VERSION, "legacy_substrate_name": LEGACY_NAME[ds],
        "canonical_corpus_definition": CORPUS_DEF[ds],
        "source_files": sorted(bi["source_files"]), "source_hashes": bi["source_files"],
        "raw_node_count": ir["raw_source_records"],
        "deduplicated_node_count": ir["raw_source_records"] - ir["duplicates_collapsed"],
        "canonical_node_count": ir["canonical_node_count"],
        "duplicates_removed": ir["duplicates_collapsed"], "dropped_records": ir["dropped_records"], "drop_reasons": ir["drop_reasons"],
        # a real split iff queries/<split>.jsonl exists (covers hotpotqa's 'validation'; excludes aux entries
        # such as musique's singlehop_dev_test_aux, which is an official file reference, not a query split)
        "query_counts": {**{s: qs[s]["n"] for s in qs if isinstance(qs[s], dict) and "n" in qs[s]
                            and os.path.exists(f"{d}/queries/{s}.jsonl")},
                         "eval_subset": ev.get("n")},
        "eval_subset": {"file": os.path.basename(ev["file"]) if ev.get("file") else None, "n": ev.get("n"),
                        "recovery_method": ev.get("method"), "n_legacy": ev.get("n_legacy"), "n_recovered": ev.get("n_recovered"),
                        "n_missed": ev.get("n_missed"), "official_split_distribution": ev.get("official_split_distribution")},
        "corpus_depends_on_eval_subset": False,
        "node_id_scheme": NODE_ID_SCHEME[ds], "node_id_stable": True,
        "normalization_rules": NORMALIZATION, "dedup_policy": DEDUP[ds],
        "identity_normalization_diagnostics": acc.get("identity_normalization_diagnostics"),
        "all_eval_gold_nodes_present": ir["ALL_EVAL_GOLDS_PRESENT"],
        "gold_shortfall_diagnostic": (None if ir["ALL_EVAL_GOLDS_PRESENT"] else gold_shortfall_diagnostic(d, ir)),
        "missing_eval_gold_nodes": ir["unresolved_gold_refs"],
        "gold_resolution_per_split": {s: {k: qs[s][k] for k in ("n", "n_with_gold_refs", "n_all_gold_resolved", "gold_refs_total", "gold_nodes_total")}
                                      for s in qs if isinstance(qs[s], dict) and "n_with_gold_refs" in qs[s]},
        "corpus_hash": ir["CORPUS_HASH"], "ordered_node_id_hash": ir["NODE_ORDER_HASH"], "nodes_jsonl_sha256": ir["nodes_jsonl_sha256"],
        "CORPUS_QUERY_INDEPENDENT": qi["CORPUS_QUERY_INDEPENDENT"] if qi else None,
        "DETERMINISTIC_REBUILD": qi["DETERMINISTIC_REBUILD"] if qi else None,
        "query_independence_test": {"runs": {k: {"eval_subset": v["eval_subset"], "CORPUS_HASH": v["CORPUS_HASH"][:16], "n": v["canonical_node_count"]}
                                             for k, v in qi["runs"].items()}, "eval_subsets_A_vs_B": qi["eval_subsets_A_vs_B"]} if qi else None,
        "legacy_comparison": ({**LEGACY_CLASS[ds], "legacy_count": lc["legacy_count"], "canonical_count": lc["canonical_count"],
                               "legacy_nodes_mapped": lc["overlap_legacy_nodes_mapped"], "missing_in_canonical": lc["missing_in_canonical"],
                               "ambiguous_legacy_nodes": lc["ambiguous_legacy_nodes"],
                               "extra_canonical_nodes": lc["extra_canonical_nodes_not_in_legacy"], "extra_by_split_provenance": lc["extra_by_split_provenance"],
                               "changed_text": lc["changed_text"], "id_mapping_coverage": lc["id_mapping_coverage"],
                               "eval_subset_gold_consistency": lc["eval_subset_gold_consistency"],
                               "files": ["legacy_comparison.json", "node_id_map_legacy.json"]} if lc else LEGACY_CLASS[ds]),
        "builder": {"path": bi["builder"], "builder_sha256": bi["builder_sha256"], "jsonstream_sha256": bi["jsonstream_sha256"],
                    "command": bi["command"], "git_commit": bi["git_commit"], "build_started": bi["started"], "build_finished": bi["finished"],
                    "seconds": bi["seconds"], "peak_rss_mb": bi["peak_rss_mb"]},
        "files": {"nodes": "nodes.jsonl", "queries": sorted(os.listdir(f"{d}/queries")), "eval_subset": os.path.basename(ev["file"]) if ev.get("file") else None,
                  "reports": [f for f in sorted(os.listdir(d)) if f.endswith(".json")]},
        "downstream_artifacts_built": "NONE (no embeddings / SPLADE / graphs / partitions / indexes) -- by design of this track",
        "READY_FOR_INDEX_BUILD": bool(ir["ALL_EVAL_GOLDS_PRESENT"] and qi and qi["CORPUS_QUERY_INDEPENDENT"] and qi["DETERMINISTIC_REBUILD"] and ir["unique_node_ids"]),
        "manifest_written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    prev = load(f"{d}/dataset_manifest.json") or {}
    for k in PRESERVE:
        if k in prev and k not in man:
            man[k] = prev[k]
    json.dump(man, open(f"{d}/dataset_manifest.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    return man


ALL_DATASETS = ("metaqa", "2wiki", "musique", "squad", "webqsp", "hotpotqa")


def main():
    # only datasets whose corpus actually exists on disk are tabled; the other two stay in "deferred"
    built = [ds for ds in ALL_DATASETS if os.path.exists(f"{ROOT}/{ds}/build_info.json")]
    mans = {ds: one(ds) for ds in built}
    prev = load(f"{ROOT}/MANIFEST.json") or {}
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: commit = None
    root = {"track": "B -- FINAL dataset canonicalization", "dataset_version": VERSION, "git_commit": commit,
            "invariant": "QUERY SUBSETTING IS ALLOWED. CORPUS SUBSETTING BY QUERY IS FORBIDDEN. Canonical corpus = fixed once per dataset, "
                         "independent of which queries are evaluated. Changing the query subset must not change nodes, ids, graph, embeddings, partitions, index.",
            "datasets": {ds: {"dir": f"{ROOT}/{ds}", "legacy_substrate_name": m["legacy_substrate_name"], "canonical_node_count": m["canonical_node_count"],
                              "corpus_hash": m["corpus_hash"], "ordered_node_id_hash": m["ordered_node_id_hash"], "nodes_jsonl_sha256": m["nodes_jsonl_sha256"],
                              "CORPUS_QUERY_INDEPENDENT": m["CORPUS_QUERY_INDEPENDENT"], "all_eval_gold_nodes_present": m["all_eval_gold_nodes_present"],
                              "legacy_classification": m["legacy_comparison"]["classification"], "READY_FOR_INDEX_BUILD": m["READY_FOR_INDEX_BUILD"],
                              "manifest": f"{ROOT}/{ds}/dataset_manifest.json"} for ds, m in mans.items()},
            "FULL_CANONICAL_V1_STATUS": "READY_5_OF_6_SOURCE_CONTRACTS",
            "WEBQSP_STATUS": "WEBQSP_BLOCKED_ON_FREEBASE_SOURCE",
            "approval_source": f"{ROOT}/_APPROVALS.json",
            "approval_gate_note": "Per-dataset approval, read from _APPROVALS.json at build time and recorded in each "
                                  "build_info.json 'gates' block. The all-or-nothing file _APPROVED_SOURCE_CONTRACTS is "
                                  "OBSOLETE, is never consulted, and was deliberately NOT created. LOCKED_6_OF_6 must not be "
                                  "written while webqsp is BLOCKED_PENDING_FREEBASE_SOURCE.",
            "blocked": {k: json.load(open(f"{ROOT}/{k}/status.json", encoding="utf-8"))
                        for k in ("webqsp",)
                        if k not in built and os.path.exists(f"{ROOT}/{k}/status.json")},
            "reports": {"audit": [f"{ROOT}/CANONICALIZATION_AUDIT.json", f"{ROOT}/CANONICALIZATION_AUDIT.md"],
                        "source_contracts": f"{ROOT}/SOURCE_CONTRACTS_FOR_REVIEW.md",
                        "artifact_reuse": f"{ROOT}/ARTIFACT_REUSE_REPORT.md",
                        "downstream_policy": f"{ROOT}/DOWNSTREAM_REBUILD_POLICY.md", "build_log": f"{ROOT}/_build.log"},
            "builder_dir": "scratchpad/final_canonical_build/",
            "builder_files_sha256": {f: file_sha(f"scratchpad/final_canonical_build/{f}") for f in sorted(os.listdir("scratchpad/final_canonical_build")) if f.endswith(".py")},
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    # never silently drop the status block a previous phase wrote; only the six-dataset table is regenerated.
    # CANONICAL_V1_STATUS is intentionally NOT carried forward: it is recomputed below from what actually exists.
    for k in ("READY", "PENDING_FULL_CANONICAL", "status_note", "status_updated", "phase_status"):
        if k in prev and k not in root:
            root[k] = prev[k]
    root["CANONICAL_V1_STATUS"] = ("BUILT_5_OF_6 (webqsp BLOCKED_PENDING_FREEBASE_SOURCE)"
                                   if sorted(built) == ["2wiki", "hotpotqa", "metaqa", "musique", "squad"]
                                   else f"PARTIAL_{len(built)}_OF_6")
    root["READY"] = sorted(built)
    root["PENDING_FULL_CANONICAL"] = ["webqsp"]
    root["status_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    root["corpus_totals"] = {**{ds: m["canonical_node_count"] for ds, m in mans.items()},
                             "_subtotal_excluding_webqsp": sum(m["canonical_node_count"] for m in mans.values()),
                             "webqsp": "BLOCKED -- no corpus"}
    json.dump(root, open(f"{ROOT}/MANIFEST.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(json.dumps({ds: {k: v for k, v in d.items() if k in ("canonical_node_count", "CORPUS_QUERY_INDEPENDENT", "READY_FOR_INDEX_BUILD", "legacy_classification")}
                      for ds, d in root["datasets"].items()}, indent=1))


if __name__ == "__main__":
    main()
