"""Phase 2 (Track B) -- three independent query lanes per dataset, materialized under
data/final_canonical/<ds>/queries/lanes/ and recorded in dataset_manifest.json.

    LEGACY_CONTINUITY  paired legacy->canonical analysis ONLY (may be train-derived; then NOT a valid eval set)
    QUALITY_LOCKED     proper official held-out labelled queries; final accuracy / significance
    SCALE_ALL          every eligible official query; scale / latency / robustness

The corpus is never read by lane selection.  The invariant
    corpus_hash(LEGACY_CONTINUITY) == corpus_hash(QUALITY_LOCKED) == corpus_hash(SCALE_ALL)
is asserted two ways and both results are recorded:
  (a) BOOKKEEPING  -- one nodes.jsonl serves all lanes; its CORPUS_HASH is recomputed here and compared
                      to build_info/integrity/manifest.
  (b) CONSTRUCTIVE -- the corpus was rebuilt from scratch under different --eval-subset arguments
                      (query_independence_test.json), and this script additionally rebuilds it with
                      --eval-subset <QUALITY_LOCKED id file> when --rebuild-check is passed.

    python scratchpad/final_canonical_build/query_lanes.py [ds ...] [--rebuild-check]
"""
import os, sys, json, hashlib, collections, subprocess, time, shutil

ROOT = "data/final_canonical"
BUILT = ["metaqa", "2wiki", "musique", "squad", "hotpotqa"]   # the five non-blocked datasets
PENDING = ["webqsp"]                                          # BLOCKED_PENDING_FREEBASE_SOURCE

# lane rule, per dataset: QUALITY_LOCKED = official held-out split with PUBLIC labels
#   (test when public, else dev).  LEGACY_CONTINUITY = the frozen legacy eval subset.
LANE_SPEC = {
    "metaqa": {"quality_split": "test",
               "quality_why": "official MetaQA test labels are PUBLIC (results/data_audit: official_test_labels='public'); "
                              "dev is reserved for model selection and is what LEGACY_CONTINUITY already occupies",
               "legacy_file": "eval_1998.jsonl"},
    "2wiki": {"quality_split": "dev",
              "quality_why": "official 2wiki TEST answers are HIDDEN, so dev (12,576) is the only proper held-out labelled split",
              "legacy_file": "eval_2000.jsonl"},   # resolved dynamically by _legacy_file(); kept as the expected name
    "musique": {"quality_split": "dev",
                "quality_why": "official MuSiQue-Ans TEST labels are HIDDEN (0/2,459 test records carry gold refs), "
                               "so dev (2,417) is the only proper held-out labelled split",
                "legacy_file": "eval_2000.jsonl"},
    "squad": {"quality_split": "dev",
              "quality_why": "SQuAD's true test set is never released; dev-v2.0 (11,873) is the standard held-out set",
              "legacy_file": "eval_2000.jsonl"},
    # --- the two corpora built by build_kb.py (Phase 4); lanes materialize only after their nodes.jsonl exists ---
    "webqsp": {"quality_split": "test",
               "quality_why": "official Microsoft WebQSP.test.json (1,639 questions) labels are PUBLIC, and this is the split the "
                              "legacy substrate both built its corpus from and evaluated on -- canonical_v1 keeps the split as the "
                              "accuracy lane while removing the corpus coupling (the corpus now spans all three RoG splits)",
               "legacy_file": "eval_2000.jsonl"},
    "hotpotqa": {"quality_split": "validation",
                 "quality_why": "official HotpotQA FULLWIKI DEV (7,405) is the only held-out split with public labels -- the fullwiki "
                                "TEST split's answers and supporting_facts are empty in the official parquet",
                 "legacy_file": "eval_2000.jsonl"},
}
# 2wiki moved to build_kb.py on 2026-09-05 together with its corpus (the full para_with_hyperlink article
# universe, ~6M records, which only the streaming/external-sort builder can produce).  Leaving it on build.py
# made rebuild_check() rebuild the SUPERSEDED 398,354-node context-union corpus and report a spurious FAIL.
BUILDER = {"metaqa": "build.py", "2wiki": "build_kb.py", "musique": "build.py", "squad": "build.py",
           "webqsp": "build_kb.py", "hotpotqa": "build_kb.py"}

COMPACT = ("query_id", "split", "question", "hop", "type", "qtype", "answers", "answer",
           "gold_node_ids", "gold_refs", "is_impossible", "answerable", "topic_entity_node_id",
           "legacy_query_id", "legacy_cache_row")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def iter_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compact(r):
    return {k: r[k] for k in COMPACT if k in r}


def corpus_hash_of(ds):
    """Recompute CORPUS_HASH + ordered_node_id_hash straight from nodes.jsonl (streaming)."""
    ids, pairs = [], []
    prev = None
    ordered = True
    for r in iter_jsonl(f"{ROOT}/{ds}/nodes.jsonl"):
        nid = r["node_id"]
        if prev is not None and nid < prev:
            ordered = False
        prev = nid
        ids.append(nid)
        pairs.append(f"{nid}\t{r['content_hash']}")
    return {"n": len(ids), "CORPUS_HASH": sha("\n".join(pairs)),
            "ordered_node_id_hash": sha("\n".join(ids)),
            "file_order_is_lexicographic_node_id": ordered}


def _legacy_file(d, spec):
    """The frozen legacy eval subset file. Its name embeds the RECOVERED count, which can change when the corpus
    changes, so prefer the expected name and otherwise take the unique non-random eval_*.jsonl actually present."""
    if os.path.exists(f"{d}/{spec['legacy_file']}"):
        return spec["legacy_file"]
    cands = sorted(f for f in os.listdir(d) if f.startswith("eval_") and f.endswith(".jsonl") and "random" not in f)
    assert len(cands) == 1, f"{d}: expected exactly one legacy eval file, found {cands}"
    return cands[0]


def build_lanes(ds):
    d = f"{ROOT}/{ds}"
    spec = dict(LANE_SPEC[ds])
    spec["legacy_file"] = _legacy_file(d, spec)
    ld = f"{d}/queries/lanes"
    os.makedirs(ld, exist_ok=True)
    man = json.load(open(f"{d}/dataset_manifest.json", encoding="utf-8"))
    lanes = {}

    # ---- LEGACY_CONTINUITY: the frozen legacy eval subset already recovered by the builder ----
    ev = man["eval_subset"]
    legacy_recs = list(iter_jsonl(f"{d}/{spec['legacy_file']}"))
    dist = dict(collections.Counter(r["split"] for r in legacy_recs))
    train_derived = set(dist) == {"train"}
    with open(f"{ld}/LEGACY_CONTINUITY.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in legacy_recs:
            f.write(json.dumps(compact(r), ensure_ascii=False, sort_keys=True) + "\n")
    lanes["LEGACY_CONTINUITY"] = {
        "n": len(legacy_recs),
        "file": f"queries/lanes/LEGACY_CONTINUITY.jsonl",
        "official_split_distribution": dist,
        "provenance": (
            f"the frozen legacy evaluation subset of substrate '{man['legacy_substrate_name']}', recovered onto canonical query ids. "
            f"Rows = data/ukb_storage/{man['legacy_substrate_name']}/gte_qwen/query_ids_all.json indexed by split_indices['val'] "
            f"(scratchpad/_ta_comb.py:75-92, _ta_resid.py:63-76); the 'val' cut itself comes from "
            f"src/experiments/overlap_retrain.py:171-206 _splits(). Recovery method: {ev['recovery_method']}. "
            f"Recovered {ev['n_recovered']}/{ev['n_legacy']}, missed {ev['n_missed']}."),
        "VALID_HELD_OUT_EVAL_SET": not train_derived,
        "labelling": (
            "TRAIN-DERIVED -- NOT A VALID HELD-OUT EVAL SET. These are a seeded 20% cut of OFFICIAL TRAIN "
            "(random.Random(42), 70/20/10 split in src/experiments/overlap_retrain.py::_splits, used because these questions carry "
            "no official split metadata). Use ONLY for paired legacy->canonical continuity analysis; never quote as accuracy."
            if train_derived else
            "REAL HELD-OUT DEV SPLIT -- these questions carry official split metadata, so _splits() used the official "
            "train/dev/test partition and this lane is a genuine official dev sample (hop-balanced 666/666/666). "
            "Still intended for continuity analysis; QUALITY_LOCKED is the accuracy lane."),
    }

    # ---- QUALITY_LOCKED: official held-out labelled split ----
    qs = spec["quality_split"]
    n_q = 0
    n_gold = 0
    with open(f"{ld}/QUALITY_LOCKED.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in iter_jsonl(f"{d}/queries/{qs}.jsonl"):
            f.write(json.dumps(compact(r), ensure_ascii=False, sort_keys=True) + "\n")
            n_q += 1
            if r.get("gold_node_ids"):
                n_gold += 1
    gs = man["gold_resolution_per_split"][qs]
    lanes["QUALITY_LOCKED"] = {
        "n": n_q,
        "file": "queries/lanes/QUALITY_LOCKED.jsonl",
        "official_split_distribution": {qs: n_q},
        "provenance": f"the complete official {ds} {qs.upper()} split, taken verbatim from the official source files; "
                      f"{spec['quality_why']}.",
        "VALID_HELD_OUT_EVAL_SET": True,
        "n_with_gold_node_ids": n_gold,
        "n_with_gold_refs": gs["n_with_gold_refs"],
        "n_all_gold_resolved": gs["n_all_gold_resolved"],
        "labelling": "OFFICIAL HELD-OUT LABELLED -- the lane for final accuracy and significance testing. "
                     "Disjoint from LEGACY_CONTINUITY unless noted." ,
    }

    # ---- SCALE_ALL: every official query, all splits ----
    ids_p = f"{ld}/SCALE_ALL.ids.jsonl"
    per_split = collections.Counter()
    gold_per_split = collections.Counter()
    with open(ids_p, "w", encoding="utf-8", newline="\n") as f:
        for split in sorted(os.listdir(f"{d}/queries")):
            if not split.endswith(".jsonl"):
                continue
            sname = split[:-6]
            for r in iter_jsonl(f"{d}/queries/{split}"):
                f.write(json.dumps({"query_id": r["query_id"], "split": r["split"]}, sort_keys=True) + "\n")
                per_split[sname] += 1
                if r.get("gold_node_ids"):
                    gold_per_split[sname] += 1
    lanes["SCALE_ALL"] = {
        "n": sum(per_split.values()),
        "file": "queries/lanes/SCALE_ALL.ids.jsonl",
        "record_source": "queries/<split>.jsonl (already materialized in full; the lane file is the ordered id+split index, "
                         "so no query record is duplicated on disk)",
        "official_split_distribution": dict(per_split),
        "n_with_gold_node_ids_per_split": dict(gold_per_split),
        "provenance": "every official query of every released split of the dataset, as parsed by "
                      f"scratchpad/final_canonical_build/{BUILDER[ds]}'s query stage from the official source files.",
        "VALID_HELD_OUT_EVAL_SET": False,
        "labelling": "MIXED SPLITS -- scale / latency / robustness only. Contains train questions and (where the official test "
                     "labels are hidden) unlabelled questions; never quote as accuracy.",
    }

    # ---- overlaps between lanes (disclosure, not a failure) ----
    lc = {r["query_id"] for r in iter_jsonl(f"{ld}/LEGACY_CONTINUITY.jsonl")}
    ql = {r["query_id"] for r in iter_jsonl(f"{ld}/QUALITY_LOCKED.jsonl")}
    overlaps = {"LEGACY_CONTINUITY_n_in_QUALITY_LOCKED": len(lc & ql)}

    # ---- the corpus-hash-equality assertion ----
    ch = corpus_hash_of(ds)
    stored = {"manifest": man["corpus_hash"], "integrity": json.load(open(f"{d}/integrity_report.json", encoding="utf-8"))["CORPUS_HASH"],
              "build_info": json.load(open(f"{d}/build_info.json", encoding="utf-8"))["CORPUS_HASH"]}
    per_lane = {k: ch["CORPUS_HASH"] for k in lanes}
    equal = len(set(per_lane.values())) == 1 and all(v == ch["CORPUS_HASH"] for v in stored.values())
    qi = man["query_independence_test"]["runs"]
    assertion = {
        "statement": "corpus_hash(LEGACY_CONTINUITY) == corpus_hash(QUALITY_LOCKED) == corpus_hash(SCALE_ALL)",
        "per_lane_corpus_hash": per_lane,
        "RESULT": "PASS" if equal else "FAIL",
        "how_bookkeeping": "one nodes.jsonl serves every lane; CORPUS_HASH recomputed from nodes.jsonl in this script and "
                           "compared against dataset_manifest / integrity_report / build_info",
        "recomputed_from_nodes_jsonl": ch,
        "stored_corpus_hashes": stored,
        "how_constructive": "the corpus was rebuilt from the official sources under three DIFFERENT --eval-subset arguments "
                            "(legacy / random_dev:7 / none) and produced an identical CORPUS_HASH each time",
        "constructive_evidence": {k: {"eval_subset": v["eval_subset"], "CORPUS_HASH_prefix": v["CORPUS_HASH"], "n": v["n"]}
                                  for k, v in qi.items()},
    }

    man["query_lanes"] = lanes
    man["query_lane_overlaps"] = overlaps
    man["corpus_hash_equality_assertion"] = assertion
    man["legacy_substrate_aliases"] = sorted({man["legacy_substrate_name"], ds})
    man["manifest_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(man, open(f"{d}/dataset_manifest.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[{ds}] lanes " + " ".join(f"{k}={v['n']}" for k, v in lanes.items()) +
          f" | corpus_hash_assertion={assertion['RESULT']} | overlap(LC in QL)={overlaps['LEGACY_CONTINUITY_n_in_QUALITY_LOCKED']}")
    return lanes, assertion


def rebuild_check(ds):
    """Constructive per-lane test: rebuild the corpus with --eval-subset = the QUALITY_LOCKED id file."""
    d = f"{ROOT}/{ds}"
    tmp = f"{ROOT}/_work/lanecheck/{ds}"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([sys.executable, f"scratchpad/final_canonical_build/{BUILDER[ds]}", ds,
                        "--eval-subset", f"{d}/queries/lanes/QUALITY_LOCKED.jsonl", "--out", tmp, "--tag", "lane_QL"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"RESULT": "ERROR", "stderr": r.stderr[-2000:]}
    bi = json.load(open(f"{tmp}/build_info.json", encoding="utf-8"))
    man = json.load(open(f"{d}/dataset_manifest.json", encoding="utf-8"))
    ok = (bi["CORPUS_HASH"] == man["corpus_hash"] and bi["NODE_ORDER_HASH"] == man["ordered_node_id_hash"]
          and bi["canonical_node_count"] == man["canonical_node_count"])
    out = {"RESULT": "PASS" if ok else "FAIL", "eval_subset": "QUALITY_LOCKED lane file",
           "CORPUS_HASH": bi["CORPUS_HASH"], "ordered_node_id_hash": bi["NODE_ORDER_HASH"],
           "n": bi["canonical_node_count"], "out_dir": tmp}
    shutil.rmtree(tmp, ignore_errors=True)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_rebuild = "--rebuild-check" in sys.argv
    targets = args or BUILT
    summary = {}
    for ds in targets:
        lanes, assertion = build_lanes(ds)
        summary[ds] = {"lanes": {k: v["n"] for k, v in lanes.items()}, "assertion": assertion["RESULT"]}
        if do_rebuild:
            rc = rebuild_check(ds)
            summary[ds]["rebuild_check_QUALITY_LOCKED"] = rc
            man = json.load(open(f"{ROOT}/{ds}/dataset_manifest.json", encoding="utf-8"))
            man["corpus_hash_equality_assertion"]["constructive_evidence"]["LANE_QUALITY_LOCKED_REBUILD"] = rc
            json.dump(man, open(f"{ROOT}/{ds}/dataset_manifest.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
            print(f"[{ds}] rebuild-check with QUALITY_LOCKED lane -> {rc['RESULT']} {rc.get('CORPUS_HASH','')[:16]}")
    json.dump(summary, open(f"{ROOT}/_work/query_lanes_summary.json", "w"), indent=2)
