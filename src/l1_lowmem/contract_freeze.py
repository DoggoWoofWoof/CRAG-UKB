"""Step 1 -- freeze the existing H4_SK partitioning contract before testing any alternative.

Every value is quoted from the defining source line (pattern -> (lineno, line)) and from the
partition manifests that actually ran; nothing is inferred.  Writes
results/L1_LOWMEM/H4_SK_CONTRACT_FREEZE.json (self-hashed record; supersedes into _history/ if re-run).

    python src/l1_lowmem/contract_freeze.py
"""
import json
import os
import sys
import time

from src.l1_lowmem.common import REPO, OUT, log, pin, rj, source_lines, wsl  # noqa: E402

sys.path.insert(0, REPO)
from src.l1_canonical.adapter import CanonicalDataset, contract_hash  # noqa: E402
from src.dataset_canonical.freeze_canonical import record_hash, write_record  # noqa: E402

SRC = {
    "src/l1_canonical/hypergraph.py": ["TARGET = 100", "RULE = \"H4_SPLIT_PRESERVE\"", "FAMSET = \"SK\"", "return max(1, int(N) // TARGET)",
                                       "eptr int64 [E+1], eidx int32 [pins]", "hyperedge weight = max(1, rint(1000 / (|e| - 1)))"],
    "scratchpad/_l1hu_build.py": ["WSCALE = 1000", "\"H4_SPLIT_PRESERVE\": {\"mode\": \"split\"", "w = np.maximum(1, np.rint(WSCALE / (sz - 1))",
                                  "H4_SPLIT_PRESERVE   oversized e decomposed"],
    "scratchpad/_l1hu_local_worker.py": ["N, k, EPS = int(z['N'][0]), int(z['k'][0]), 0.03", "PresetType.DETERMINISTIC_QUALITY", "set_partitioning_parameters(k, EPS, mtkahypar.Objective.KM1)",
                                         "mtkahypar.set_seed(0)", "[1] * N, ew.tolist()", "part = hg.partition(ctx)", "part.block_id(i)", "ru_maxrss"],
    "src/l1_canonical/partition.py": ["\"partitioner\": \"mtkahypar\"", "\"vertex_weights\": \"unit\"", "DEFAULT_THREADS = 8", "HARD_CAP_GB = 6.0",
                                      "MARGIN_GB = 1.0", "deterministic_preset_note"],
    "src/l1_canonical/adapter.py": ["indexed in canonical POSITION space", "position -> block id, when that L1 artefact exists"],
    "scratchpad/_ta_prepartition.py": ["K0 = 60", "K_LOCK = 100", "TOP200 = 200", "def rrf_partitions(rankings, npart, masks=None):",
                                       "BASE  dense[:K] -> PR_d ;  splade[:K] -> PR_s ;  RRF(PR_d, PR_s) -> top-P"],
    "scratchpad/_l1ps_router.py": ["P = 50", "output             EXACTLY 50 canonical C partitions, no stray nodes"],
    "scratchpad/_l1kb_core.py": ["BASE_CFG = dict(B=6, M_struct=64, M_ret=32, agg=\"S4\", fusion=\"F6\", T=1, tag=\"F6\")"],
    "src/l1_canonical/replay_cache.py": ["TOPP = 200", "SMAX = 256", "EVAL_CAP = 2000"],
}


def main():
    quotes = {}
    for rel, pats in SRC.items():
        q = source_lines(os.path.join(REPO, rel), pats)
        missing = [p for p, v in q.items() if not v]
        if missing:
            raise RuntimeError("%s: defining line not found for %s -- the contract moved; do not infer" % (rel, missing))
        quotes[rel] = {"sha256": pin(os.path.join(REPO, rel))["sha256"], "lines": {p: v for p, v in q.items()}}
    r = wsl("python3 -c \"import mtkahypar,sys,platform;print(getattr(mtkahypar,'__version__','?'));print(sys.version.split()[0]);"
            "print(platform.platform())\"; pip show mtkahypar 2>/dev/null | grep -E '^(Name|Version):'; nproc; grep MemTotal /proc/meminfo", timeout=120)
    env = [ln for ln in (r.stdout + r.stderr).split("\n") if ln.strip() and "screen size" not in ln]
    ran = {}
    for ds in ("metaqa", "squad", "musique"):
        d = CanonicalDataset(ds)
        pj = rj(os.path.join(d.derived_dir, "parts", "H4_SK.json"))
        pf = rj(os.path.join(d.derived_dir, "parts", "H4_SK.FAILED.json"))
        hj = rj(os.path.join(d.derived_dir, "hypergraph", "H4_SK.json"))
        src = pj or pf
        ran[ds] = {"N": d.n_nodes, "k_frozen_rule": max(1, d.n_nodes // 100), "DATASET_json_RECORD_SHA256": d.record_sha,
                   "hypergraph": {k: hj.get(k) for k in ("rule", "famset", "families", "k", "k_rule", "cap", "cap_rule", "mode", "weighted", "weight_scale",
                                                         "hyperedges", "pins", "weight_min", "weight_max", "content_digest", "file_sha256")} if hj else None,
                   "partition_run": ({"STATUS": src["STATUS"], "contract": src["contract"], "worker_stats": src.get("worker_stats"), "guard": src.get("guard"),
                                      "wsl": src.get("wsl"), "output": src.get("output"), "post_checks": src.get("post_checks")} if src else None)}
    ch = contract_hash()
    rec = {"RECORD": "H4_SK_CONTRACT_FREEZE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "PURPOSE": "the exact H4_SK partitioning contract every low-memory candidate is compared against; quoted from the defining source lines "
                      "and from the partition manifests that ran -- no inferred values",
           "CONTRACT": {
               "hypergraph_construction": {"rule": "H4_SPLIT_PRESERVE", "families": ["STRUCT", "KNN"], "famset": "SK",
                                           "hyperedge": "closed neighbourhood of each anchor node over each family; oversized ones split into anchor-preserving "
                                                        "chunks (mode split), no pin discarded; cap = target block size (('blk', 1.0) -> 100)",
                                           "vertex_weights": "unit ([1] * N)", "hyperedge_weights": "max(1, rint(1000 / (|e| - 1))) (WSCALE 1000)",
                                           "pins": "canonical positions (nodes.jsonl line order, pinned by DATASET.json nodes.sha256)",
                                           "file_format": "H4_SK.npz: eptr int64 [E+1], eidx int32 [pins], N int64 [1], k int64 [1], ew int32 [E]"},
               "partitioner": {"name": "mtkahypar", "version_in_use": "see environment", "preset": "DETERMINISTIC_QUALITY", "objective": "KM1",
                               "epsilon": 0.03, "seed": 0, "threads_default": 8, "thread_independence": "DETERMINISTIC_QUALITY is thread-count independent "
                               "(LOCAL_COMPUTE.json verified local == Modal bit-identical)", "k_rule": "max(1, N // 100)", "output": "position -> block id (int64 npy)"},
               "node_order": "canonical position order = nodes.jsonl line i is position i; every L1 artefact indexes that space",
               "P50_post_processing": {"router": "dense[:K] -> partition ranking PR_d; splade[:K] -> PR_s; partition-level RRF(PR_d, PR_s) -> top-P",
                                       "K": 100, "K0": 60, "P": 50, "rrf_tie_break": "rankings[0] (dense) supplies the deterministic tie-break order",
                                       "cache_depth": {"TOPP": 200, "SMAX": 256, "TOP200": 200, "EVAL_CAP": 2000},
                                       "selector": "SAFE = BASE_CFG B=6, M_struct=64, M_ret=32, agg S4, fusion F6 (_l1kb_core.sel_f6); BASE = sel_base"},
               "balance_envelope_downstream": "l1_eval ELIGIBLE = max_block/mean_block <= 1.05 and every block used",
           },
           "SOURCE_QUOTES": quotes,
           "L1_CONTRACT": ch,
           "environment_wsl": env,
           "RAN": ran,
           "guard_note": "peak_rss_mb in worker_stats is the WSL worker's ru_maxrss: it includes the Python-side pin lists handed to create_hypergraph, "
                         "not only Mt-KaHyPar's internal structures; the guard's peak_rss_kb is the 1 s RSS poll of the same process"}
    p = os.path.join(OUT, "H4_SK_CONTRACT_FREEZE.json")
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(p):
        old = rj(p)
        hdir = os.path.join(OUT, "_history")
        os.makedirs(hdir, exist_ok=True)
        hp = os.path.join(hdir, "H4_SK_CONTRACT_FREEZE_%s_%s.json" % (old["utc"].replace("-", "").replace(":", ""), old["RECORD_SHA256"][:8]))
        os.replace(p, hp)
        rec["SUPERSEDES"] = {"file": os.path.relpath(hp, REPO).replace("\\", "/"), "RECORD_SHA256": old["RECORD_SHA256"]}
    write_record(p, rec)
    chk = rj(p)
    assert record_hash(chk) == chk["RECORD_SHA256"]
    log("wrote", os.path.relpath(p, REPO), chk["RECORD_SHA256"][:16])
    log("environment:", env)
    for ds, v in ran.items():
        pr = v["partition_run"]
        log("%-8s N %s k %s | hypergraph %s | partition %s" % (ds, v["N"], v["k_frozen_rule"], "built" if v["hypergraph"] else "MISSING",
                                                            pr["STATUS"] if pr else "NOT RUN"))
    return rec


if __name__ == "__main__":
    main()
