"""Step 9: re-freeze the L1 manifest on the canonical substrates.

Collects, per dataset, every L1-owned artefact built on canonical positions and its digest --
adapter validation, hypergraph (+ parity vs legacy), partition (local WSL or external lane),
replay cache (+ parity vs legacy), L1 BASE/SAFE replay -- and writes one self-hashed record
(repo convention: RECORD_SHA256 / RECORD_SHA256_LF) to results/L1_CANONICAL/L1_CANONICAL_MANIFEST.json.
A dataset whose chain is incomplete is recorded with its blocking stage, never omitted; the legacy
results/L1/L1_LOCKED_MANIFEST.json is pinned as the comparison baseline and never edited.

    python src/l1_canonical/freeze_l1.py
"""
import io
import json
import os
import sys
import tarfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.l1_canonical.adapter import CanonicalDataset, DATASETS, contract_hash, sha_file  # noqa: E402
from src.dataset_canonical.freeze_canonical import record_hash, write_record  # noqa: E402

OUT = os.path.join(REPO, "results", "L1_CANONICAL")
STAGES = ["adapter_validation", "hypergraph", "partition", "replay_cache", "l1_replay"]


def _rj(p):
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def _pin(p):
    return {"path": os.path.relpath(p, REPO).replace("\\", "/"), "bytes": os.path.getsize(p), "sha256": sha_file(p)} if os.path.exists(p) else None


def external_lane():
    """the packed kits for corpora the local host cannot partition (data/l1_canonical/_external/*.tar.gz): pin + KIT.json summary."""
    kdir = os.path.join(REPO, "data", "l1_canonical", "_external")
    out = {}
    for fn in sorted(os.listdir(kdir)) if os.path.isdir(kdir) else []:
        if not (fn.startswith("l1_partition_kit_") and fn.endswith(".tar.gz")):
            continue
        ds = fn[len("l1_partition_kit_"):-len(".tar.gz")]
        kit = None
        with tarfile.open(os.path.join(kdir, fn), "r:gz") as tf:
            for m in tf:
                if m.name.endswith("/KIT.json"):
                    kit = json.loads(tf.extractfile(m).read().decode("utf-8"))
                    break
        out[ds] = {"kit": _pin(os.path.join(kdir, fn)),
                   "summary": {k: kit.get(k) for k in ("dataset", "N", "k", "STRUCT", "KNN", "expected_pins_upper_bound", "keys_npz_sha256",
                                                        "memory_expectation", "L1_CONTRACT_SHA256", "DATASET_json_RECORD_SHA256", "packed_utc")} if kit else None,
                   "result_imported": os.path.exists(os.path.join(REPO, "data", "l1_canonical", ds, "parts", "H4_SK.json"))}
    return out


def collect(ds, adapter_val, hyper_parity, cache_parity):
    d = CanonicalDataset(ds)
    e = {"DATASET_json_RECORD_SHA256": d.record_sha, "n_nodes": d.n_nodes, "n_queries": d.n_queries, "eval_split": d.eval_split,
         "k_frozen_rule": max(1, d.n_nodes // 100), "stages": {}}
    av = ((adapter_val or {}).get("datasets") or {}).get(ds)
    e["stages"]["adapter_validation"] = {"PASS": bool(av and av.get("PASS")), "n_checks": av.get("n_checks") if av else None,
                                         "keysets": av.get("keysets") if av else None} if av else None
    hj = _rj(os.path.join(d.derived_dir, "hypergraph", "H4_SK.json"))
    e["stages"]["hypergraph"] = ({"file": _pin(os.path.join(d.derived_dir, "hypergraph", "H4_SK.npz")), "content_digest": hj["content_digest"],
                                  "hyperedges": hj["hyperedges"], "pins": hj["pins"], "k": hj["k"], "cap": hj["cap"],
                                  "record_matches": hj["inputs"]["DATASET_json_RECORD_SHA256"] == d.record_sha} if hj else None)
    pj = _rj(os.path.join(d.derived_dir, "parts", "H4_SK.json"))
    pf = _rj(os.path.join(d.derived_dir, "parts", "H4_SK.FAILED.json"))
    if pj:
        e["stages"]["partition"] = {"STATUS": pj["STATUS"], "file": _pin(os.path.join(d.derived_dir, "parts", "H4_SK.npy")),
                                    "ran_on": pj["contract"].get("ran_on", "local_wsl"), "threads": pj["contract"].get("threads"),
                                    "worker_sha256": pj["contract"]["worker_sha256"], "km1": pj["worker_stats"].get("objective_km1"),
                                    "imbalance": pj["worker_stats"].get("imbalance"), "peak_rss_mb": pj["worker_stats"].get("peak_rss_mb"),
                                    "wall_seconds": pj["worker_stats"].get("wall_seconds"), "post_checks": pj["post_checks"],
                                    "hypergraph_content_digest": pj["inputs"]["hypergraph_content_digest"],
                                    "record_matches": pj["inputs"]["DATASET_json_RECORD_SHA256"] == d.record_sha}
    elif pf:
        e["stages"]["partition"] = {"STATUS": pf["STATUS"], "guard": pf.get("guard"), "note": pf.get("note")}
    else:
        e["stages"]["partition"] = None
    cj = _rj(d.cache_path()[:-4] + ".json")
    e["stages"]["replay_cache"] = ({"file": _pin(d.cache_path()), "n_dev_queries": cj["meta"]["n_dev_queries"], "sample_rule": cj["meta"]["sample_rule"],
                                    "FROZEN_AGAINST": cj["meta"]["FROZEN_AGAINST"], "row_query_ids_sha256": cj["meta"]["row_query_ids_sha256"],
                                    "BASE_ALL_P50": cj["meta"]["BASE_ALL_P50"], "BASE_ANY_P50": cj["meta"]["BASE_ANY_P50"],
                                    "node_embedding_path": cj["meta"].get("node_embedding_path"),
                                    "digest_ok": os.path.exists(d.cache_path()) and sha_file(d.cache_path()) == cj["sha256"]} if cj else None)
    rj = _rj(os.path.join(OUT, "L1_REPLAY_%s.json" % ds))
    e["stages"]["l1_replay"] = ({k: rj[k] for k in ("BASE_ALL_P50", "BASE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_ALL_P50", "SAFE_ANY_P50",
                                                     "SAFE_SCOPE_NODES", "BND_ALL_P50", "CORR", "BALANCE", "CUT", "npart", "utc")}
                                | {"nq": rj["population"]["nq"], "by_hop": rj.get("by_hop"), "file": _pin(os.path.join(OUT, "L1_REPLAY_%s.json" % ds)),
                                   "contract_matches": rj.get("L1_CONTRACT_SHA256")} if rj else None)
    lv = _rj(os.path.join(OUT, "LEGACY_VS_CANONICAL_%s.json" % ds))
    e["legacy_vs_canonical"] = ({"status": lv["status"], "overlap": lv.get("overlap"), "BASE": lv.get("BASE"), "SAFE": lv.get("SAFE"),
                                 "legacy_cell": lv.get("legacy_cell"), "file": _pin(os.path.join(OUT, "LEGACY_VS_CANONICAL_%s.json" % ds))} if lv else None)
    lc = _rj(os.path.join(OUT, "LEGACY_VS_CANONICAL_%s_legacyrows.json" % ds))          # comparison.py: canonical cache on the bridged legacy rows
    e["legacy_vs_canonical_on_legacy_rows"] = ({"status": lc["status"], "overlap": lc.get("overlap"), "BASE": lc.get("BASE"), "SAFE": lc.get("SAFE"),
                                                "agreement": lc.get("agreement"), "legacy_cell": lc.get("legacy_cell"),
                                                "comparison_cache": _pin(d.cache_path()[:-4] + "_legacyrows.npz"),
                                                "file": _pin(os.path.join(OUT, "LEGACY_VS_CANONICAL_%s_legacyrows.json" % ds)),
                                                "note": "COMPARISON_ONLY population (query subsetting to the legacy cache rows, non-test); never a production number"}
                                               if lc else None)
    blocked = next((s for s in STAGES if not e["stages"].get(s) or (s == "partition" and e["stages"][s].get("STATUS") not in ("OK",))
                    or (s == "adapter_validation" and not e["stages"][s].get("PASS"))), None)
    e["CHAIN_STATUS"] = "COMPLETE" if blocked is None else "BLOCKED_AT_" + blocked.upper()
    e["parity"] = {"hypergraph_builder_vs_legacy": (hyper_parity or {}).get("datasets", {}).get({"metaqa": "metaqa", "squad": "squad_clean", "musique": "musique_clean",
                                                                                                  "2wiki": "2wiki_clean", "hotpotqa": "hotpotqa_clean", "webqsp": "webqsp"}[ds], {}).get("status"),
                   "replay_cache_builder_vs_legacy": (cache_parity or {}).get("datasets", {}).get({"metaqa": "metaqa", "squad": "squad_clean", "musique": "musique_clean",
                                                                                                    "2wiki": "2wiki_clean", "hotpotqa": "hotpotqa_clean", "webqsp": "webqsp"}[ds], {}).get("status")}
    return e


def main():
    adapter_val = _rj(os.path.join(OUT, "ADAPTER_VALIDATION.json"))
    hyper_parity = _rj(os.path.join(OUT, "HYPERGRAPH_PARITY.json"))
    cache_parity = _rj(os.path.join(OUT, "REPLAY_CACHE_PARITY.json"))
    base = _rj(os.path.join(OUT, "RETURN_TO_L1_BASELINE.json"))
    ch = contract_hash()
    datasets = {ds: collect(ds, adapter_val, hyper_parity, cache_parity) for ds in DATASETS}
    rec = {"RECORD": "L1_CANONICAL_MANIFEST", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "PURPOSE": "L1 re-frozen on the canonical substrates (data/final_canonical/), canonical position space, zero-copy adapter; "
                      "every artefact pinned to the DATASET.json record it was built from and to the L1 contract hash",
           "L1_CONTRACT": ch,
           "PINS": {"CANONICAL_FREEZE.json": _pin(os.path.join(REPO, "data", "final_canonical", "CANONICAL_FREEZE.json")),
                    "RETURN_TO_L1_BASELINE.json": {"RECORD_SHA256": base["RECORD_SHA256"] if base else None, **(_pin(os.path.join(OUT, "RETURN_TO_L1_BASELINE.json")) or {})},
                    "ADAPTER_VALIDATION.json": {"ALL_PASS": adapter_val.get("ALL_PASS") if adapter_val else None, **(_pin(os.path.join(OUT, "ADAPTER_VALIDATION.json")) or {})},
                    "HYPERGRAPH_PARITY.json": {"ALL_IDENTICAL": hyper_parity.get("ALL_IDENTICAL") if hyper_parity else None, **(_pin(os.path.join(OUT, "HYPERGRAPH_PARITY.json")) or {})},
                    "REPLAY_CACHE_PARITY.json": {"ALL_IDENTICAL": cache_parity.get("ALL_IDENTICAL") if cache_parity else None, "datasets": {k: v.get("status") for k, v in (cache_parity or {}).get("datasets", {}).items()},
                                                 **(_pin(os.path.join(OUT, "REPLAY_CACHE_PARITY.json")) or {})},
                    "legacy_L1_LOCKED_MANIFEST": {**(_pin(os.path.join(REPO, "results", "L1", "L1_LOCKED_MANIFEST.json")) or {}),
                                                  "status": "comparison baseline (legacy universes); never edited"},
                    "EXTERNAL_LANE_MEMORY_EXPECTATION.json": _pin(os.path.join(OUT, "EXTERNAL_LANE_MEMORY_EXPECTATION.json")),
                    "METAQA_ENCODING_DIAGNOSIS.json": _pin(os.path.join(OUT, "METAQA_ENCODING_DIAGNOSIS.json"))},
           "RULINGS_PENDING": [
               "EXTERNAL_LANE_MEMORY: the frozen recipe extrapolates to 382/799/1184 GB (linear in pins) or 1.3/5.4/9.1 TB (pins*k) for webqsp/hotpotqa/2wiki; "
               "the >=250 GB lane cannot run it -- procure more, pin the model first, or rule a new partitioning; never a silent algorithm swap",
               "KB_CORPUS_ENCODING: canonical metaqa/webqsp encodings are NAME_ONLY (legacy: name + verbalised triples); legacy KB-corpus L1 cells are not "
               "reproducible on the canonical layout (metaqa paired BASE 0.6898 -> 0.5168); KEEP NAME_ONLY or re-encode NAME_PLUS_FACTS (supersedes the freeze channels)"],
           "CONTRACT_SUMMARY": {"router": "Dense[:100] + SPLADE[:100] -> partition RRF (K0 60) -> P50", "partition": "H4_SPLIT_PRESERVE over STRUCT+KNN closed neighbourhoods, "
                                "k = N // 100, Mt-KaHyPar DETERMINISTIC_QUALITY / KM1 / eps 0.03 / seed 0", "selector": "SAFE = B6_S4_F6_Ms64_Mr32 (frozen)",
                                "population": "EVAL_SPLITS of CANONICAL_FREEZE (webqsp: train_holdout carve); EVAL_CAP 2000 seed-0 sample (metaqa hop-balanced); TEST never read",
                                "webqsp_lane": "WEBQSP_ROG_RESOLVED / BENCHMARK_CONDITIONED / query_independent NOT CLAIMED / MID completeness NOT REQUIRED FOR V1"},
           "DATASETS": datasets,
           "EXTERNAL_LANE": {"kits": external_lane(),
                             "note": "corpora the local host cannot partition under the frozen recipe (RSS guard, HARD_CAP 6 GB): the kit ships the "
                                     "contract code + byte-identical keys.npz; external_kit.py import verifies digests, counts and k before the "
                                     "partition enters the chain. musique failed locally at the 3.59 GB cap (FAILED_MEMORY_CAP); a retry at the "
                                     "6 GB cap is queued -- if that fails too, musique goes through the same lane"},
           "CHAIN_STATUS": {ds: datasets[ds]["CHAIN_STATUS"] for ds in DATASETS},
           "COMPLETE_ON": [ds for ds in DATASETS if datasets[ds]["CHAIN_STATUS"] == "COMPLETE"]}
    p = os.path.join(OUT, "L1_CANONICAL_MANIFEST.json")
    if os.path.exists(p):                                   # supersession, never an edit: the old record moves unchanged to _history/
        old = _rj(p)
        hdir = os.path.join(OUT, "_history")
        os.makedirs(hdir, exist_ok=True)
        hp = os.path.join(hdir, "L1_CANONICAL_MANIFEST_%s_%s.json" % (old["utc"].replace("-", "").replace(":", ""), old["RECORD_SHA256"][:8]))
        os.replace(p, hp)
        rec["SUPERSEDES"] = {"file": os.path.relpath(hp, REPO).replace("\\", "/"), "RECORD_SHA256": old["RECORD_SHA256"], "CHAIN_STATUS": old.get("CHAIN_STATUS")}
    write_record(p, rec)
    chk = _rj(p)
    assert record_hash(chk) == chk["RECORD_SHA256"]
    print("wrote", os.path.relpath(p, REPO), chk["RECORD_SHA256"][:16], json.dumps(rec["CHAIN_STATUS"]))
    return rec


if __name__ == "__main__":
    main()
