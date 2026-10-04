"""PHG_SCALE_BRIDGE -- MuSiQue under the experimental PHG contract frozen by PHG_SMALL_DATASET_VALIDATION_RULING.json.

MuSiQue (117,534 nodes, 6,014,938 H4_SK pins, k 1,175) is the first corpus where the frozen Mt-KaHyPar recipe is RESOURCE_INFEASIBLE_LOCAL
(MUSIQUE_MTKAHYPAR_RETRY.json: killed at the 6.0 GB RSS cap), so no paired Mt-KaHyPar reference exists.  The question is feasibility,
not equivalence:
    does exact H4_SK -> PHG (NP=4, frozen parameters) -> PHG_NONEMPTY_REPAIR_V1 if needed -> P50 -> canonical L1
    complete locally with sane partition / L1 behaviour?
Absolute L1 numbers are recorded; NO comparative quality claim against Mt-KaHyPar is made; nothing is tuned.

    python -u src/l1_lowmem/phg_bridge.py prereg        -> results/L1_LOWMEM/PHG_BRIDGE_PREREG.json (written once, before the run)
    python -u src/l1_lowmem/phg_bridge.py run musique   -> results/L1_LOWMEM/PHG_BRIDGE_RUNS_musique.json (stage artifacts under data/l1_lowmem/musique/phg/,
                                                           experimental import parts/LOWMEM__PHG_C1_con.*, replay cache, L1_DOWNSTREAM_musique__phg_c1.json)
    python -u src/l1_lowmem/phg_bridge.py report        -> results/L1_LOWMEM/PHG_BRIDGE_REPORT.{json,md}  (STOP_FOR_REVIEW)
Stage boundaries: 1 H4_SK input verified -> 2 PHG input/callback manifest (exact structure gate) -> 3 completed PHG partition (+ repeat run)
-> 3b PHG_NONEMPTY_REPAIR_V1 (no-op when no block is empty) -> 4 experimental import + P50 replay cache -> 5 canonical L1 numerics (absolute).
Nothing frozen (FREIGHT records / musique FREIGHT arm, canonical H4_SK, H4_SK.FAILED.json, PHG squad/metaqa records) is written; phg.py and
phg_repair.py are imported unchanged (both sha-pinned by their preregistrations).
"""
import io
import json
import math
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, log, sha_file, rj, wj, pin, ds_dir  # noqa: E402
from src.l1_lowmem import freight as FR  # noqa: E402
from src.l1_lowmem import hgr as HG  # noqa: E402
from src.l1_lowmem import phg as P  # noqa: E402
from src.l1_lowmem import phg_repair as PR  # noqa: E402
from src.l1_lowmem import phg_ruling as RL  # noqa: E402

CLASS = "PHG_SCALE_BRIDGE"
TAG_B = "LOWMEM__PHG_C1_con"          # C1 = the experimental PHG contract of the 2026-09-14 ruling: PHG NP4 (frozen parameters) + PHG_NONEMPTY_REPAIR_V1
SUFFIX_B = "__phg_c1"
DATASETS = ["musique"]
THIS = os.path.abspath(__file__)
RULING = os.path.join(OUT, RL.RULING_RECORD)
LABELS = ["PHG_SCALE_BRIDGE_COMPLETE", "PHG_STRUCTURE_MISMATCH", "PHG_BRIDGE_RESOURCE_FAIL", "PHG_REPAIR_POSTCONDITION_FAIL", "PHG_PARTITION_INVALID",
          "PHG_BRIDGE_L1_EXECUTION_FAIL"]
SANITY = {"S1_downstream_executed": "the unchanged canonical L1 path ran to its record (l1_downstream.run: replay cache -> l1_eval numerics); status "
                                    "BASELINE_ABSENT is the expected outcome (no production Mt-KaHyPar replay exists for musique)",
          "S2_population": "the evaluated population is the frozen canonical dev sample (EVAL_SPLITS dev, nq 2000) with row_query_ids_sha256 identical to the "
                           "frozen musique FREIGHT replay's -- the sample rule is partition-independent, so any difference is a pipeline fault",
          "S3_canonical_eligibility": "l1_eval's existing ELIGIBLE flag (max block / mean <= 1.05 and every block used) is True -- an existing canonical rule, "
                                      "not a new threshold",
          "S4_definitional_consistency": "n_docs == N, npart == k, 0 <= BASE_ALL <= BASE_ANY <= 1, 0 <= SAFE_ALL <= SAFE_ANY <= 1, scope nodes > 0, every number "
                                         "finite, SAFE selector executed (CORR present) -- ALL is a subset of ANY per query by definition",
          "S5_no_numeric_retrieval_threshold": "no acceptance threshold on BASE / SAFE / coverage exists or will be added after seeing the numbers; the "
                                               "absolute numbers are recorded as they come"}
DIAGNOSTICS_NO_RULE = ["KM1 / connectivity (absolute; no Mt-KaHyPar value exists for musique)", "STRUCT cut", "KNN cut", "block distribution", "P50 scope",
                       "candidate coverage (ANY)", "SAFE additions", "hop-wise results where the canonical replay carries hop labels", "memory per stage",
                       "runtime per stage", "repeat-run identity", "context only: the frozen musique FREIGHT arm's absolute numbers (FREIGHT_CLOSED; not a "
                       "reference partitioner) and a paired exact McNemar of PHG vs that arm on the identical 2,000 ids -- diagnostic, no decision depends on it",
                       "blocker comparison (resources, not quality): the Mt-KaHyPar retry's RSS at kill / wall vs PHG's per-rank peaks / wall"]


def exact_p(g, l):
    from math import comb
    n = g + l
    return 1.0 if n == 0 else min(1.0, 2.0 * sum(comb(n, i) for i in range(min(g, l) + 1)) / (2.0 ** n))


def host_snapshot():
    import psutil
    vm = psutil.virtual_memory()
    me = os.getpid()
    foreign = []
    for pr in psutil.process_iter(["pid", "name", "create_time", "memory_info"]):
        try:
            if pr.info["pid"] != me and (pr.info["name"] or "").lower().startswith("python") and pr.info["memory_info"] and pr.info["memory_info"].rss > (1 << 30):
                foreign.append({"pid": pr.info["pid"], "name": pr.info["name"], "rss_mb": round(pr.info["memory_info"].rss / 2.0 ** 20),
                                "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(pr.info["create_time"]))})
        except Exception:
            pass
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host_total_mb": round(vm.total / 2.0 ** 20), "host_available_mb": round(vm.available / 2.0 ** 20),
            "foreign_python_processes_over_1GB_rss_untouched": foreign}


def py_peak_mb():
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2.0 ** 20, 1)


def data_pins(ds):
    d = ds_dir(ds)
    cd = os.path.join(REPO, "data", "l1_canonical", ds)
    files = {"H4_SK_npz": os.path.join(cd, "hypergraph", "H4_SK.npz"), "H4_SK_manifest": os.path.join(cd, "hypergraph", "H4_SK.json"),
             "mtkahypar_FAILED_record": os.path.join(cd, "parts", "H4_SK.FAILED.json"),
             "freight_import_npy": os.path.join(cd, "parts", "LOWMEM__FREIGHT_con.npy"), "freight_import_json": os.path.join(cd, "parts", "LOWMEM__FREIGHT_con.json"),
             "freight_replay_cache_npz": os.path.join(d, "replay_cache__LOWMEM__FREIGHT_con.npz"), "freight_replay_cache_json": os.path.join(d, "replay_cache__LOWMEM__FREIGHT_con.json"),
             "stream_manifest_H4_SK_STREAM_V1": os.path.join(d, "stream", "H4_SK_STREAM_V1.json"), "reconstruction_gate": os.path.join(d, "stream", "RECONSTRUCTION_GATE.json"),
             "freight_stream_manifest_txt": os.path.join(d, "freight", "stream_manifest.txt"), "official_netl_json": os.path.join(d, "H4_SK.official.netl.json"),
             "semantic_gate": os.path.join(d, "H4_SK.semantic_gate.json"), "hgr_record": os.path.join(d, "H4_SK.hgr.json")}
    return {k: pin(p) for k, p in files.items()}


# ----------------------------------------------------------------------------- preregistration
def prereg():
    pp = os.path.join(OUT, "PHG_BRIDGE_PREREG.json")
    if os.path.exists(pp):
        raise RuntimeError("bridge preregistration already exists -- written once, before the run")
    rul = rj(RULING)
    if rul is None or rul["PHG_SMALL_DATASET_VALIDATION"] != "PASS":
        raise RuntimeError("the ruling record is missing or does not read PASS")
    pre, prr, b = rj(os.path.join(OUT, "PHG_PREREG.json")), rj(os.path.join(OUT, "PHG_REPAIR_PREREG.json")), rj(os.path.join(OUT, "PHG_BUILD.json"))
    ds = DATASETS[0]
    hm = rj(os.path.join(REPO, "data", "l1_canonical", ds, "hypergraph", "H4_SK.json"))
    hg = rj(os.path.join(ds_dir(ds), "H4_SK.hgr.json"))
    fl = rj(os.path.join(REPO, "data", "l1_canonical", ds, "parts", "H4_SK.FAILED.json"))
    fr = rj(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds))
    frl = rj(os.path.join(OUT, "L1_REPLAY_%s__LOWMEM__FREIGHT_con.json" % ds))
    N, M, Pp, k = hm["N"], hm["hyperedges"], hm["pins"], hm["k"]
    frozen = [fn for fn in P.FROZEN_READ_ONLY] + ["PHG_PREREG.json", "PHG_BUILD.json", "PHG_RUNS_squad.json", "PHG_RUNS_metaqa.json", "PHG_REPAIR_PREREG.json",
                                                   "PHG_REPAIR_RUNS_squad.json", "PHG_REPAIR_RUNS_metaqa.json", "PHG_REPORT.json", "PHG_REPORT.md",
                                                   "L1_DOWNSTREAM_squad__phg.json", "L1_DOWNSTREAM_metaqa__phg_repair1.json", "L1_REPLAY_squad__LOWMEM__PHG_con.json",
                                                   "L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json", "L1_REPLAY_musique__LOWMEM__FREIGHT_con.json",
                                                   "H4_SK_CONTRACT_FREEZE.json", RL.RULING_RECORD]
    rec = {"RECORD": "PHG_BRIDGE_PREREG", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "classification": CLASS, "dataset": ds, "tag": TAG_B, "suffix": SUFFIX_B,
           "ruling": pin(RULING), "contract": {"name": RL.EXPERIMENTAL_PHG_CONTRACT, "lane_preregistration": pin(os.path.join(OUT, "PHG_PREREG.json")),
                                               "repair_preregistration": pin(os.path.join(OUT, "PHG_REPAIR_PREREG.json")), "build": pin(os.path.join(OUT, "PHG_BUILD.json")),
                                               "driver_binary_sha256": pre["build"]["driver_binary_sha256"], "driver_source_sha256": pre["build"]["driver_source_sha256"],
                                               "phg_parameters_in_order": P.PARAMS, "NUM_GLOBAL_PARTS": k, "mpi_ranks": P.NP,
                                               "rank_count_rule": rul["experimental_PHG_contract"]["rank_count_rule"],
                                               "repair": PR.REPAIR, "repair_algorithm": PR.ALGORITHM, "no_tuning": "no parameter change of any kind between datasets; "
                                               "no second configuration; no other rank count; PHG is run once (R1 = the arm's partition) plus one identical repeat run "
                                               "(diagnostic only)",
                                               "modules": {"phg.py": pin(P.__file__), "phg_repair.py": pin(PR.__file__), "phg_ruling.py": pin(RL.__file__)}},
           "program": pin(THIS),
           "question": rul["next"]["authorized_now"]["question"],
           "why_musique": rul["next"]["authorized_now"]["why"],
           "no_reference": {"mtkahypar": "RESOURCE_INFEASIBLE_LOCAL -- %s (threads %d, cap %.1f GB as recorded by partition.py, RSS at kill %.0f MB, wall %.0f s)" % (
                                fl["STATUS"], fl["contract"]["threads"], fl["guard"]["rss_cap_gb"], fl["guard"]["peak_rss_kb"] / 1024.0, fl["guard"]["wall_seconds"]),
                            "consequence": "no paired comparison, no equivalence claim, no comparative quality claim against Mt-KaHyPar; absolute L1 numbers only"},
           "inputs": {"N": N, "M": M, "P": Pp, "k": k, "k_rule": "max(1, N // 100)", "contract_bound_ceil_1.03_N_over_k": int(math.ceil(1.03 * N / float(k))),
                      "DATASET_json_RECORD_SHA256": hm["inputs"]["DATASET_json_RECORD_SHA256"], "H4_SK_npz_sha256": hm["file_sha256"],
                      "hypergraph_content_digest": hm.get("content_digest"), "ORIGINAL_STRUCTURE_SHA256": hg["ORIGINAL_STRUCTURE_SHA256"],
                      "stream_manifest": fr["inputs"]["stream_manifest"], "data_pins": data_pins(ds)},
           "expected_population": {"eval_split": "dev", "sample_rule": frl["population"]["sample_rule"], "nq": frl["population"]["nq"],
                                   "row_query_ids_sha256": frl["population"]["row_query_ids_sha256"],
                                   "note": "identical to the frozen musique FREIGHT replay's population (partition-independent sample rule); the sealed test split is never read"},
           "stages": ["1 H4_SK input verified (shards, official netl, semantic gate, stream manifest pins; structure digest recomputed == ORIGINAL)",
                      "2 PHG input/callback manifest = exact structure gate (gate-mode run; STRUCTURE_MISMATCH = STOP)",
                      "3 completed PHG partition R1 (+ dumps byte-identical to the gate run; + R1_repeat identity diagnostic)",
                      "3b %s on R1 (one minimum-delta-KM1 move per empty block; no-op when no block is empty); postconditions as preregistered in PHG_REPAIR_PREREG.json; "
                      "then the canonical validity gate (no empty block, max <= ceil(1.03 N/k))" % PR.REPAIR,
                      "4 experimental import parts/%s.* -> P50 replay cache (unchanged canonical builder)" % TAG_B,
                      "5 canonical L1 numerics (l1_eval, unchanged) -> L1_DOWNSTREAM_%s%s.json (BASELINE_ABSENT expected)" % (ds, SUFFIX_B)],
           "labels": LABELS, "sanity_checks": SANITY,
           "decision_rule": "PHG_STRUCTURE_MISMATCH (gate 1) | PHG_BRIDGE_RESOURCE_FAIL (a stage did not complete locally: mpirun failure, out-of-memory, cache build "
                            "failure -- the failing stage and error are recorded; NP is never changed) | PHG_REPAIR_POSTCONDITION_FAIL | PHG_PARTITION_INVALID (after "
                            "the repair) | PHG_BRIDGE_L1_EXECUTION_FAIL (stage 5 did not complete or a sanity check S1-S4 failed) | PHG_SCALE_BRIDGE_COMPLETE "
                            "(every stage completed and S1-S4 pass).  No label depends on the value of any retrieval number.",
           "diagnostics_no_rule": DIAGNOSTICS_NO_RULE,
           "measurements": ["per stage: wall seconds, peak RSS per MPI rank (/usr/bin/time -v + getrusage), host Python peak working set after each host stage",
                            "host snapshot before / after (total / available memory; foreign python processes > 1 GB RSS, untouched)",
                            "replay-cache builder's predicted peak and BASE / STRUCT cost from its manifest"],
           "stop_rule": "STOP_FOR_REVIEW after the musique record; WebQSP / HotpotQA / 2Wiki require a separate authorization (the ruling names WebQSP as next only if "
                        "MuSiQue works); no promotion; no tuning; FREIGHT stays FREIGHT_CLOSED; the squad / metaqa PHG records are not touched",
           "frozen_read_only": {fn: pin(os.path.join(OUT, fn)) for fn in frozen if os.path.exists(os.path.join(OUT, fn))}}
    wj(pp, rec)
    log("preregistered", CLASS, ds, "->", os.path.relpath(pp, REPO))
    return rec


# ----------------------------------------------------------------------------- run
def stop(R, fp, decision, stage, err=None):
    R["DECISION"] = decision
    R["stopped_at_stage"] = stage
    if err is not None:
        R["error"] = err
    R["host_after"] = host_snapshot()
    wj(fp, R)
    log("%s DECISION %s (stage %s)%s -> %s" % (R["dataset"], decision, stage, (": " + str(err)[:300]) if err else "", os.path.relpath(fp, REPO)))
    return R


def run(ds):
    pre = rj(os.path.join(OUT, "PHG_BRIDGE_PREREG.json"))
    if pre is None:
        raise RuntimeError("run prereg first")
    if ds not in DATASETS or ds != pre["dataset"]:
        raise RuntimeError("%s is not the preregistered bridge dataset" % ds)
    if pin(THIS)["sha256"] != pre["program"]["sha256"]:
        raise RuntimeError("phg_bridge.py changed since preregistration")
    for name, mod in (("phg.py", P), ("phg_repair.py", PR), ("phg_ruling.py", RL)):
        if sha_file(mod.__file__) != pre["contract"]["modules"][name]["sha256"]:
            raise RuntimeError("%s changed since preregistration" % name)
    if sha_file(RULING) != pre["ruling"]["sha256"]:
        raise RuntimeError("ruling record changed since preregistration")
    fp = os.path.join(OUT, "PHG_BRIDGE_RUNS_%s.json" % ds)
    if os.path.exists(fp):
        raise RuntimeError("%s exists -- the bridge runs once; supersede, never overwrite" % os.path.relpath(fp, REPO))
    lane_pre = rj(os.path.join(OUT, "PHG_PREREG.json"))
    b = P.build()
    if b["binary"]["sha256"] != lane_pre["build"]["driver_binary_sha256"] or b["source"]["sha256"] != lane_pre["build"]["driver_source_sha256"]:
        raise RuntimeError("driver changed since the lane preregistration (contract element)")
    b["packages_versions"] = rj(os.path.join(OUT, "PHG_PACKAGES.json"))["after"]["dpkg_versions"]
    T = {}
    R = {"RECORD": "PHG_BRIDGE_RUNS", "classification": CLASS, "arm": RL.EXPERIMENTAL_PHG_CONTRACT, "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "preregistration": pin(os.path.join(OUT, "PHG_BRIDGE_PREREG.json")), "ruling": pin(RULING), "program": pin(THIS), "host_before": host_snapshot(),
         "contract": {"parameters": P.PARAMS, "mpi_ranks": P.NP, "driver": b["binary"], "driver_source": b["source"], "packages": b["packages_versions"], "repair": PR.REPAIR},
         "stages": {}, "runs": {}, "timing_seconds": T}
    # ---- stage 1: H4_SK input verified
    t = time.time()
    try:
        H, man, gate_rec, off, netl = FR.load_inputs(ds)
        N, k, M, Pp = H["N"], H["k"], H["M"], H["P"]
        ins = pre["inputs"]
        if (N, M, Pp, k) != (ins["N"], ins["M"], ins["P"], ins["k"]) or H["npz_sha256"] != ins["H4_SK_npz_sha256"] or man["ORIGINAL_STRUCTURE_SHA256"] != ins["ORIGINAL_STRUCTURE_SHA256"]:
            raise RuntimeError("H4_SK differs from the preregistered input")
        smf = os.path.join(ds_dir(ds), "freight", "stream_manifest.txt")
        if sha_file(smf) != ins["stream_manifest"]["sha256"]:
            raise RuntimeError("stream manifest changed since its frozen record")
        dig, dstats = HG.structure_digest(H["eptr"], H["eidx"], H["ew"], N)
        if dig != man["ORIGINAL_STRUCTURE_SHA256"]:
            raise RuntimeError("H4_SK structure digest %s != ORIGINAL" % dig)
    except Exception as e:
        return stop(R, fp, "PHG_BRIDGE_RESOURCE_FAIL", "1_input_verified", "%s: %s" % (type(e).__name__, e))
    T["1_input_verified"] = round(time.time() - t, 1)
    stage1 = {"stage": "1_H4_SK_input_verified", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "N": N, "M": M, "P": Pp, "k": k,
              "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "structure_digest_recomputed": dig, "structure_stats": dstats,
              "hypergraph_content_digest": H["meta"].get("content_digest"), "H4_SK_npz_sha256": H["npz_sha256"],
              "reconstruction_gate": {kk: gate_rec[kk] for kk in ("gate_digest", "gate_bytes") if kk in gate_rec},
              "shards": [{"index": c["index"], "sha256": c["sha256"], "nodes": c["nodes"], "pins": c["pins"]} for c in man["shards"]],
              "stream_manifest": ins["stream_manifest"], "official_netl_sha256": off["output"]["sha256"], "seconds": T["1_input_verified"], "py_peak_mb": py_peak_mb()}
    R["inputs"] = {"N": N, "M": M, "P": Pp, "k": k, "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "official_netl": off["output"], "stream_manifest": ins["stream_manifest"]}
    R["stages"]["1_input_verified"] = stage1
    wj(os.path.join(ds_dir(ds), "phg", "STAGE1_INPUT_VERIFIED.json"), stage1)
    log("=== %s %s: N %d M %d P %d k %d shards %d ranks %d  (H4_SK digest %s == ORIGINAL) ===" % (CLASS, ds, N, M, Pp, k, len(man["shards"]), P.NP, dig[:16]))
    # ---- stage 2: structure gate (gate-mode run)
    t = time.time()
    try:
        g_res, g_dir = P.mpirun(ds, "G", "gate", k, smf, b)
    except Exception as e:
        return stop(R, fp, "PHG_BRIDGE_RESOURCE_FAIL", "2_structure_gate_run", "%s: %s" % (type(e).__name__, e))
    R["runs"]["G"] = g_res
    gate = P.structure_gate(ds, g_dir, man, H=H, zoltan_eval=g_res["zoltan_eval"])
    gate.update({"stage": "2_PHG_input_callback_manifest", "run": "G", "dump_sha256": g_res["dump_sha256"], "global_from_queries": g_res["global_from_queries"],
                 "zoltan_removed_or_warning_lines": g_res["zoltan_removed_or_warning_lines"]})
    T["2_structure_gate"] = round(time.time() - t, 1)
    R["stages"]["2_structure_gate"] = gate
    wj(os.path.join(ds_dir(ds), "phg", "STAGE2_PHG_INPUT_CALLBACK_MANIFEST.json"), gate)
    log("  structure gate: %s  (N %s M %s P %s weights %d..%d sum %d; digest %s; weight reception rel diff %s)" % (
        gate["gate"], gate["N_returned"], gate["M_with_pins"], gate["P_returned"], gate["weight_min"] or -1, gate["weight_max"], gate["weight_sum"],
        (gate["digest_from_queries"] or "-")[:16], (gate.get("weight_reception") or {}).get("relative_diff")))
    if gate["gate"] != "PASS":
        R["problems"] = gate["problems"]
        return stop(R, fp, "PHG_STRUCTURE_MISMATCH", "2_structure_gate")
    # ---- stage 3: completed PHG partition R1 (+ repeat)
    t = time.time()
    try:
        r1, d1 = P.mpirun(ds, "R1", "partition", k, smf, b)
    except Exception as e:
        return stop(R, fp, "PHG_BRIDGE_RESOURCE_FAIL", "3_partition_R1", "%s: %s" % (type(e).__name__, e))
    R["runs"]["R1"] = r1
    r1["dumps_identical_to_gate_run"] = r1["dump_sha256"] == g_res["dump_sha256"]
    if not r1["dumps_identical_to_gate_run"]:
        R["problems"] = ["partition-run query dumps differ from the gate run's"]
        return stop(R, fp, "PHG_STRUCTURE_MISMATCH", "3_partition_R1")
    hard_r1 = P.assemble_partition(d1, N, k)
    v_r1 = P.validity(hard_r1, N, k)
    m_r1 = FR.km1_metrics(H, hard_r1); m_r1["family_cuts"] = FR.family_cuts(ds, hard_r1)
    R["stages"]["2_structure_gate"]["partition_run_dumps_identical"] = True
    try:
        r2, d2 = P.mpirun(ds, "R1_repeat", "partition", k, smf, b)
        hard2 = P.assemble_partition(d2, N, k)
        R["runs"]["R1_repeat"] = r2
        R["repeat"] = {"identical_partition": bool(np.array_equal(hard_r1, hard2)), "n_diff": int((hard_r1 != hard2).sum()), "km1_repeat": FR.km1_metrics(H, hard2)["km1_weighted"],
                       "note": "same NP, same input, same parameters; diagnostic only -- R1 is the arm's partition"}
    except Exception as e:
        R["repeat"] = {"identical_partition": None, "error": "%s: %s" % (type(e).__name__, e), "note": "repeat run failed; diagnostic only, R1 stands"}
    T["3_partition_R1_and_repeat"] = round(time.time() - t, 1)
    zc = r1["zoltan_eval"]["cutl_global"]
    stage3 = {"stage": "3_completed_PHG_partition", "run": "R1", "validity_raw": v_r1, "metrics_raw": m_r1, "partition_sha256_txt": {i: sha_file(os.path.join(d1, "part_rank%d.txt" % i)) for i in range(P.NP)},
              "vector_sha256_raw": PR.vec_sha(hard_r1), "zoltan_eval": r1["zoltan_eval"],
              "python_km1_vs_zoltan_cutl": {"python_km1": m_r1["km1_weighted"], "zoltan_cutl_float": zc, "relative_diff": round(abs(m_r1["km1_weighted"] - zc) / max(1.0, m_r1["km1_weighted"]), 8),
                                            "consistent_within_float32": bool(abs(m_r1["km1_weighted"] - zc) <= max(1.0, 1e-6 * m_r1["km1_weighted"]))},
              "memory": r1["memory"], "timing": r1["timing"], "repeat": R["repeat"], "mpi_ranks": P.NP, "py_peak_mb": py_peak_mb()}
    R["stages"]["3_partition"] = stage3
    wj(os.path.join(ds_dir(ds), "phg", "STAGE3_PHG_PARTITION.json"), stage3)
    log("%-7s R1: raw validity %s (max %d, bound %d, empty %d)  km1 %d  STRUCT %.4f KNN %.4f  repeat identical %s  zoltan cutl %.0f  peak RSS/rank max %.0f MB  partition %.1f s" % (
        ds, v_r1["gate"], v_r1["max_block"], v_r1["contract_bound_ceil_1.03_N_over_k"], v_r1["empty_blocks"], m_r1["km1_weighted"], m_r1["family_cuts"]["STRUCT"]["edge_cut_fraction"],
        m_r1["family_cuts"]["KNN"]["edge_cut_fraction"], R["repeat"].get("identical_partition"), zc, r1["memory"]["max_rank_peak_kb"] / 1024.0, r1["timing"]["partition_wall_seconds"]))
    # ---- stage 3b: PHG_NONEMPTY_REPAIR_V1 (part of the contract; no-op when no block is empty)
    t = time.time()
    hard, empties, moves = PR.repair(H, hard_r1)
    v = P.validity(hard, N, k)
    m = FR.km1_metrics(H, hard); m["family_cuts"] = FR.family_cuts(ds, hard)
    changed = np.where(hard != hard_r1)[0]
    post = {"same_N": len(hard) == N, "assigned_once": bool(len(hard) == N and hard.min() >= 0 and hard.max() < k), "same_k": True,
            "empty_blocks_after": v["empty_blocks"], "max_block_after": v["max_block"], "contract_bound": v["contract_bound_ceil_1.03_N_over_k"], "H4_SK_unchanged": True,
            "moves": len(moves), "empty_blocks_before": len(empties), "moves_equal_empties": len(moves) == len(empties), "nodes_changed": int(len(changed)),
            "nodes_changed_equal_moves": int(len(changed)) == len(moves), "km1_consistent": m["km1_weighted"] == m_r1["km1_weighted"] + sum(mv["delta_km1_weighted"] for mv in moves)}
    post["PASS"] = bool(post["same_N"] and post["assigned_once"] and post["empty_blocks_after"] == 0 and post["max_block_after"] <= post["contract_bound"] and post["moves_equal_empties"]
                        and post["nodes_changed_equal_moves"] and post["km1_consistent"])
    T["3b_repair"] = round(time.time() - t, 1)
    rdir = os.path.join(ds_dir(ds), "phg_repair1"); os.makedirs(rdir, exist_ok=True)
    npy_rep = os.path.join(rdir, "repaired.npy"); np.save(npy_rep, hard.astype(np.int64))
    stage3b = {"stage": "3b_%s" % PR.REPAIR, "noop": len(empties) == 0, "empty_blocks": empties, "moves": moves, "postconditions": post,
               "before": {"validity": v_r1, "km1_weighted": m_r1["km1_weighted"], "cut_weighted": m_r1["cut_weighted"], "blocks": m_r1["blocks"], "family_cuts": m_r1["family_cuts"], "vector_sha256": PR.vec_sha(hard_r1)},
               "after": {"validity": v, "km1_weighted": m["km1_weighted"], "cut_weighted": m["cut_weighted"], "blocks": m["blocks"], "family_cuts": m["family_cuts"], "vector_sha256": PR.vec_sha(hard),
                         "delta_km1_total": m["km1_weighted"] - m_r1["km1_weighted"],
                         "struct_cut_delta": round(m["family_cuts"]["STRUCT"]["edge_cut_fraction"] - m_r1["family_cuts"]["STRUCT"]["edge_cut_fraction"], 6),
                         "knn_cut_delta": round(m["family_cuts"]["KNN"]["edge_cut_fraction"] - m_r1["family_cuts"]["KNN"]["edge_cut_fraction"], 6), "repaired_npy": pin(npy_rep)},
               "seconds": T["3b_repair"], "py_peak_mb": py_peak_mb()}
    R["stages"]["3b_repair"] = stage3b
    R["metrics"] = m
    wj(os.path.join(ds_dir(ds), "phg", "STAGE3b_REPAIR.json"), stage3b)
    log("%-7s repair: empties %s  moves %d  KM1 %d -> %d (%+d)  max %d -> %d  empty %d -> %d  post %s  validity %s" % (
        ds, empties, len(moves), m_r1["km1_weighted"], m["km1_weighted"], stage3b["after"]["delta_km1_total"], v_r1["max_block"], v["max_block"], v_r1["empty_blocks"], v["empty_blocks"],
        "PASS" if post["PASS"] else "FAIL", v["gate"]))
    if not post["PASS"]:
        return stop(R, fp, "PHG_REPAIR_POSTCONDITION_FAIL", "3b_repair")
    if v["gate"] != "PASS":
        return stop(R, fp, "PHG_PARTITION_INVALID", "3b_validity")
    # context (diagnostic, no rule): the frozen musique FREIGHT arm's objective, and the Mt-KaHyPar blocker's resources
    fr = rj(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds)) or {}
    frm = fr.get("metrics_B") or {}
    fl = rj(os.path.join(REPO, "data", "l1_canonical", ds, "parts", "H4_SK.FAILED.json")) or {}
    R["mtkahypar_baseline"] = {"status": "RESOURCE_INFEASIBLE_LOCAL", "record": pin(os.path.join(REPO, "data", "l1_canonical", ds, "parts", "H4_SK.FAILED.json")),
                               "peak_rss_kb_at_kill": (fl.get("guard") or {}).get("peak_rss_kb"), "rss_cap_gb": (fl.get("guard") or {}).get("rss_cap_gb"),
                               "wall_seconds_at_kill": (fl.get("guard") or {}).get("wall_seconds"), "threads": (fl.get("contract") or {}).get("threads"),
                               "km1_weighted": None, "note": "no partition exists; nothing to compare against; resources only"}
    R["freight_one_pass_frozen"] = {"km1_weighted": frm.get("km1_weighted"), "cut_weighted": frm.get("cut_weighted"), "blocks": frm.get("blocks"), "family_cuts": frm.get("family_cuts"),
                                    "status": "FREIGHT_CLOSED; context only, not a reference partitioner"}
    R["diagnostic_context"] = {"km1_ratio_vs_freight_one_pass": round(m["km1_weighted"] / float(frm["km1_weighted"]), 4) if frm.get("km1_weighted") else None,
                               "blocker_resources": {"mtkahypar_retry_peak_rss_mb_at_kill": round((fl.get("guard") or {}).get("peak_rss_kb", 0) / 1024.0, 1),
                                                     "mtkahypar_retry_wall_seconds_at_kill": (fl.get("guard") or {}).get("wall_seconds"),
                                                     "phg_peak_rss_mb_max_rank": round(r1["memory"]["max_rank_peak_kb"] / 1024.0, 1),
                                                     "phg_peak_rss_mb_sum_ranks_upper_bound": round(r1["memory"]["sum_of_rank_peaks_kb"] / 1024.0, 1),
                                                     "phg_partition_wall_seconds": r1["timing"]["partition_wall_seconds"], "phg_job_wall_seconds": r1["timing"]["job_wall_seconds"]}}
    # ---- stage 4/5: experimental import -> replay cache (P50) -> canonical L1 numerics (absolute; BASELINE_ABSENT expected)
    t = time.time()
    try:
        Rimp = dict(R); Rimp["metrics"] = m; Rimp["repeat"] = R["repeat"]
        extra = {"repair": {"adapter": PR.REPAIR, "noop": len(empties) == 0, "empty_blocks": empties, "moves": moves, "km1_before": m_r1["km1_weighted"], "km1_after": m["km1_weighted"],
                            "postconditions": post, "vector_sha256_in": stage3b["before"]["vector_sha256"], "vector_sha256_out": stage3b["after"]["vector_sha256"]},
                 "bridge": {"classification": CLASS, "preregistration": R["preregistration"], "ruling": R["ruling"], "program": R["program"],
                            "note": "PHG_SCALE_BRIDGE: no Mt-KaHyPar reference exists for this corpus (RESOURCE_INFEASIBLE_LOCAL); absolute L1 numbers only, no equivalence claim"}}
        rec = P.import_partition(ds, H, hard, Rimp, b, off, man, gate, tag=TAG_B, extra=extra)
        mp = os.path.join(H["d"].derived_dir, "parts", "%s.json" % TAG_B)
        rec = rj(mp)
        rec["arm"] = RL.EXPERIMENTAL_PHG_CONTRACT
        rec["contract"]["repair"] = "%s applied to the PHG R1 vector (%d empty block(s), %d move(s)); no-op when no block is empty" % (PR.REPAIR, len(empties), len(moves))
        wj(mp, rec)
        R["import"] = rec
        from src.l1_lowmem import l1_downstream as LD
        D = LD.run(ds, tag=TAG_B, suffix=SUFFIX_B)
    except Exception as e:
        return stop(R, fp, "PHG_BRIDGE_RESOURCE_FAIL", "4_5_import_replay_cache_L1", "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2000:]))
    T["4_5_import_replay_cache_L1"] = round(time.time() - t, 1)
    cache_man = rj(os.path.join(ds_dir(ds), "replay_cache__%s.json" % TAG_B)) or {}
    replay = rj(os.path.join(OUT, "L1_REPLAY_%s__%s.json" % (ds, TAG_B))) or {}
    B = D.get("B") or {}
    R["downstream"] = {"status": D["status"], "record": pin(os.path.join(OUT, "L1_DOWNSTREAM_%s%s.json" % (ds, SUFFIX_B))), "B_phg": B, "population": D.get("population_B"),
                       "replay_record": pin(os.path.join(OUT, "L1_REPLAY_%s__%s.json" % (ds, TAG_B))),
                       "cache": {"manifest": pin(os.path.join(ds_dir(ds), "replay_cache__%s.json" % TAG_B)), "bytes": cache_man.get("bytes"), "sha256": cache_man.get("sha256"),
                                 "predicted_peak_gb": (cache_man.get("meta") or {}).get("predicted_peak_gb"), "COST": (cache_man.get("meta") or {}).get("COST"),
                                 "fp16_path": (cache_man.get("meta") or {}).get("fp16_path")},
                       "eval_seconds": replay.get("seconds"), "py_peak_mb": py_peak_mb(), "by_hop": replay.get("by_hop")}
    wj(os.path.join(ds_dir(ds), "phg", "STAGE4_5_P50_REPLAY_ABSOLUTE.json"), R["downstream"])
    # ---- sanity checks S1-S4 (structural; S5 = no numeric threshold)
    ep = pre["expected_population"]
    popB = D.get("population_B") or {}
    nums = [B.get(kk) for kk in ("BASE_ALL_P50", "BASE_ANY_P50", "SAFE_ALL_P50", "SAFE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_SCOPE_NODES", "BND_ALL_P50")]
    finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in nums)
    S = {"S1_downstream_executed": {"status": D["status"], "pass": D["status"] in ("BASELINE_ABSENT", "PAIRED"), "expected": "BASELINE_ABSENT"},
         "S2_population": {"nq": popB.get("nq"), "row_query_ids_sha256": popB.get("row_query_ids_sha256"), "expected_nq": ep["nq"], "expected_sha256": ep["row_query_ids_sha256"],
                           "pass": popB.get("nq") == ep["nq"] and popB.get("row_query_ids_sha256") == ep["row_query_ids_sha256"]},
         "S3_canonical_eligibility": {"ELIGIBLE": (B.get("BALANCE") or {}).get("ELIGIBLE"), "max_over_mean": (B.get("BALANCE") or {}).get("max_over_mean"),
                                      "blocks_used": (B.get("BALANCE") or {}).get("blocks_used"), "npart": (B.get("BALANCE") or {}).get("npart"), "pass": bool((B.get("BALANCE") or {}).get("ELIGIBLE"))},
         "S4_definitional_consistency": {"n_docs": replay.get("n_docs"), "N": N, "npart": replay.get("npart"), "k": k, "finite": finite,
                                         "BASE_ALL_le_ANY": finite and 0 <= B["BASE_ALL_P50"] <= B["BASE_ANY_P50"] <= 1, "SAFE_ALL_le_ANY": finite and 0 <= B["SAFE_ALL_P50"] <= B["SAFE_ANY_P50"] <= 1,
                                         "scope_positive": finite and B["BASE_SCOPE_NODES"] > 0 and B["SAFE_SCOPE_NODES"] > 0, "CORR_present": bool(B.get("CORR"))}}
    S["S4_definitional_consistency"]["pass"] = bool(replay.get("n_docs") == N and replay.get("npart") == k and finite and S["S4_definitional_consistency"]["BASE_ALL_le_ANY"]
                                                    and S["S4_definitional_consistency"]["SAFE_ALL_le_ANY"] and S["S4_definitional_consistency"]["scope_positive"] and B.get("CORR"))
    S["S5_no_numeric_retrieval_threshold"] = {"pass": True, "note": "no threshold exists; numbers recorded as they come"}
    S["ALL_PASS"] = all(S[kk]["pass"] for kk in ("S1_downstream_executed", "S2_population", "S3_canonical_eligibility", "S4_definitional_consistency"))
    R["sanity_checks"] = S
    # ---- diagnostic (no rule): paired vs the frozen musique FREIGHT arm on the identical ids
    frl = rj(os.path.join(OUT, "L1_REPLAY_%s__LOWMEM__FREIGHT_con.json" % ds))
    if frl and replay.get("_ind_SAFE") and frl["population"]["row_query_ids_sha256"] == popB.get("row_query_ids_sha256"):
        from src.l1_canonical import bridge as BR
        def mc(a, b_):
            a, b_ = np.asarray(a, np.int8), np.asarray(b_, np.int8)
            x = BR.mcnemar(a, b_)
            return {"FREIGHT": round(float(a.mean()), 4), "PHG": round(float(b_.mean()), 4), "delta_PHG_minus_FREIGHT": round(float(b_.mean() - a.mean()), 4),
                    "PHG_only_covered": x["canonical_only_covered"], "FREIGHT_only_covered": x["legacy_only_covered"], "mcnemar_p_reporter": x["mcnemar_p"],
                    "exact_p": exact_p(x["canonical_only_covered"], x["legacy_only_covered"])}
        R["diagnostic_vs_freight_closed_arm"] = {"note": "diagnostic only: FREIGHT is a closed arm (FREIGHT_CLOSED), not a reference partitioner; no decision depends on this; "
                                                         "no claim is made", "nq": len(replay["_ind_SAFE"]), "SAFE_ALL_P50": mc(frl["_ind_SAFE"], replay["_ind_SAFE"]),
                                                 "BASE_ALL_P50": mc(frl["_ind_BASE"], replay["_ind_BASE"]),
                                                 "freight_absolute": {kk: frl.get(kk) for kk in ("BASE_ALL_P50", "BASE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_ALL_P50", "SAFE_ANY_P50", "SAFE_SCOPE_NODES", "BND_ALL_P50")},
                                                 "freight_CORR": frl.get("CORR"), "freight_BALANCE": frl.get("BALANCE")}
    R["host_after"] = host_snapshot()
    R["DECISION"] = "PHG_SCALE_BRIDGE_COMPLETE" if S["ALL_PASS"] else "PHG_BRIDGE_L1_EXECUTION_FAIL"
    R["claim"] = ("feasibility only: the chain completed locally under the frozen experimental PHG contract; absolute L1 numbers recorded; no comparative quality claim "
                  "against Mt-KaHyPar (no reference exists for this corpus)")
    wj(fp, R)
    log("%-7s DECISION %s  BASE_ALL %.4f  SAFE_ALL %.4f  BASE_ANY %.4f  SAFE_ANY %.4f  scope %.0f/%.0f  ELIGIBLE %s  sanity %s -> %s" % (
        ds, R["DECISION"], B["BASE_ALL_P50"], B["SAFE_ALL_P50"], B["BASE_ANY_P50"], B["SAFE_ANY_P50"], B["BASE_SCOPE_NODES"], B["SAFE_SCOPE_NODES"],
        (B.get("BALANCE") or {}).get("ELIGIBLE"), S["ALL_PASS"], os.path.relpath(fp, REPO)))
    return R


# ----------------------------------------------------------------------------- report
def report():
    pre = rj(os.path.join(OUT, "PHG_BRIDGE_PREREG.json"))
    ds = pre["dataset"]
    R = rj(os.path.join(OUT, "PHG_BRIDGE_RUNS_%s.json" % ds))
    if R is None:
        raise RuntimeError("no bridge run record")
    rp = os.path.join(OUT, "PHG_BRIDGE_REPORT.json")
    moved = {fn: PR.supersede(fn) for fn in ("PHG_BRIDGE_REPORT.json", "PHG_BRIDGE_REPORT.md")} if os.path.exists(rp) else {}
    # integrity: every pinned frozen record and data file unchanged; no canonical musique partition appeared
    integ = {"frozen_records_changed": [fn for fn, pn in pre["frozen_read_only"].items() if not os.path.exists(os.path.join(OUT, fn)) or sha_file(os.path.join(OUT, fn)) != pn["sha256"]],
             "data_pins_changed": [kk for kk, pn in pre["inputs"]["data_pins"].items() if pn and (not os.path.exists(os.path.join(REPO, pn["path"])) or sha_file(os.path.join(REPO, pn["path"])) != pn["sha256"])],
             "canonical_musique_partition_still_absent": not os.path.exists(os.path.join(REPO, "data", "l1_canonical", ds, "parts", "H4_SK.npy")),
             "experimental_files_written": sorted(os.path.relpath(p, REPO).replace("\\", "/") for p in [
                 os.path.join(REPO, "data", "l1_canonical", ds, "parts", "%s.npy" % TAG_B), os.path.join(REPO, "data", "l1_canonical", ds, "parts", "%s.json" % TAG_B),
                 os.path.join(ds_dir(ds), "replay_cache__%s.npz" % TAG_B), os.path.join(ds_dir(ds), "replay_cache__%s.json" % TAG_B),
                 os.path.join(ds_dir(ds), "phg_repair1", "repaired.npy")] if os.path.exists(p)) + ["data/l1_lowmem/%s/phg/{G,R1,R1_repeat,STAGE*.json}" % ds],
             "superseded_to_history": moved}
    s3, s3b, dsn = R["stages"].get("3_partition") or {}, R["stages"].get("3b_repair") or {}, R.get("downstream") or {}
    B = dsn.get("B_phg") or {}
    fr, mt = R.get("freight_one_pass_frozen") or {}, R.get("mtkahypar_baseline") or {}
    dc = R.get("diagnostic_context") or {}
    T = R.get("timing_seconds") or {}
    stage_rows = ["| Stage | Completed | Wall s | Memory | Note |", "|---|---|---|---|---|"]
    st1 = R["stages"].get("1_input_verified") or {}
    stage_rows.append("| 1 H4_SK input verified | %s | %s | py peak %s MB | N %s M %s P %s k %s; digest recomputed == ORIGINAL |" % (bool(st1), T.get("1_input_verified"), st1.get("py_peak_mb"), st1.get("N"), st1.get("M"), st1.get("P"), st1.get("k")))
    g = R["stages"].get("2_structure_gate") or {}
    gm = (R.get("runs") or {}).get("G") or {}
    stage_rows.append("| 2 structure gate (gate-mode PHG run) | %s | %s | peak RSS/rank max %s MB | %s; weight reception rel diff %s; removal/warning lines %s |" % (
        g.get("gate"), T.get("2_structure_gate"), round((gm.get("memory") or {}).get("max_rank_peak_kb", 0) / 1024.0) if gm else "-", g.get("gate"),
        (g.get("weight_reception") or {}).get("relative_diff"), g.get("zoltan_removed_or_warning_lines") or "none"))
    if s3:
        stage_rows.append("| 3 PHG partition R1 (+ repeat) | True | %s (partition %.2f, job %s) | peak RSS/rank %s MB (max %.0f, sum %.0f) | raw validity %s: max %d (bound %d), empty %d; repeat identical %s; Python KM1 vs Zoltan cutl rel diff %s |" % (
            T.get("3_partition_R1_and_repeat"), s3["timing"]["partition_wall_seconds"], s3["timing"]["job_wall_seconds"], [round(x / 1024.0, 1) for x in s3["memory"]["peak_rss_kb_per_rank_time_v"]],
            s3["memory"]["max_rank_peak_kb"] / 1024.0, s3["memory"]["sum_of_rank_peaks_kb"] / 1024.0, s3["validity_raw"]["gate"], s3["validity_raw"]["max_block"],
            s3["validity_raw"]["contract_bound_ceil_1.03_N_over_k"], s3["validity_raw"]["empty_blocks"], (s3.get("repeat") or {}).get("identical_partition"), s3["python_km1_vs_zoltan_cutl"]["relative_diff"]))
    if s3b:
        stage_rows.append("| 3b %s | %s | %s | py peak %s MB | empty blocks %s -> %d move(s) %s; KM1 %d -> %d (%+d); STRUCT cut delta %s; KNN cut delta %s; postconditions %s; validity after %s |" % (
            PR.REPAIR, "PASS" if s3b["postconditions"]["PASS"] else "FAIL", T.get("3b_repair"), s3b.get("py_peak_mb"), s3b["empty_blocks"], len(s3b["moves"]),
            "(no-op)" if s3b["noop"] else "; ".join("node %d: %d -> %d (delta %+d, incident nets %d, ties %d)" % (mv["node_position"], mv["donor_block"], mv["empty_block"], mv["delta_km1_weighted"], mv["incident_nets"], mv["min_delta_candidates_tied"]) for mv in s3b["moves"]),
            s3b["before"]["km1_weighted"], s3b["after"]["km1_weighted"], s3b["after"]["delta_km1_total"], s3b["after"]["struct_cut_delta"], s3b["after"]["knn_cut_delta"],
            "PASS" if s3b["postconditions"]["PASS"] else "FAIL", s3b["after"]["validity"]["gate"]))
    if dsn:
        c = dsn.get("cache") or {}
        stage_rows.append("| 4/5 import + P50 replay cache + canonical L1 | %s | %s (eval %s) | builder predicted peak %s GB; py peak %s MB | cache %s bytes; BASE %s s / STRUCT %s s; status %s |" % (
            dsn.get("status"), T.get("4_5_import_replay_cache_L1"), dsn.get("eval_seconds"), c.get("predicted_peak_gb"), dsn.get("py_peak_mb"), c.get("bytes"),
            (c.get("COST") or {}).get("BASE_sec"), (c.get("COST") or {}).get("STRUCT_sec"), dsn.get("status")))
    m = R.get("metrics")
    part_rows = ["| Arm | KM1 | STRUCT cut | KNN cut | blocks min/p50/p95/max (bound) | empty | peak RSS | wall | status |", "|---|---|---|---|---|---|---|---|---|"]
    if m:
        part_rows.append("| PHG C1 (R1 + %s, NP=%d) | %d | %.4f | %.4f | %d/%.0f/%.0f/%d (%d) | %d | %.0f MB max rank / %.0f MB sum | %.2f s partition / %s s job | %s |" % (
            PR.REPAIR, P.NP, m["km1_weighted"], m["family_cuts"]["STRUCT"]["edge_cut_fraction"], m["family_cuts"]["KNN"]["edge_cut_fraction"], m["blocks"]["min"], m["blocks"]["p50"],
            m["blocks"]["p95"], m["blocks"]["max"], (s3b.get("after") or {}).get("validity", {}).get("contract_bound_ceil_1.03_N_over_k", 0), m["blocks"]["empty"],
            s3["memory"]["max_rank_peak_kb"] / 1024.0, s3["memory"]["sum_of_rank_peaks_kb"] / 1024.0, s3["timing"]["partition_wall_seconds"], s3["timing"]["job_wall_seconds"], R["DECISION"]))
    if fr.get("km1_weighted"):
        part_rows.append("| FREIGHT 1-pass (frozen, CLOSED; context) | %d | %.4f | %.4f | %d/%.0f/%.0f/%d | %d | (frozen record) | (frozen record) | FREIGHT_CLOSED |" % (
            fr["km1_weighted"], fr["family_cuts"]["STRUCT"]["edge_cut_fraction"], fr["family_cuts"]["KNN"]["edge_cut_fraction"], fr["blocks"]["min"], fr["blocks"]["p50"], fr["blocks"]["p95"], fr["blocks"]["max"], fr["blocks"]["empty"]))
    if mt:
        part_rows.append("| Mt-KaHyPar (frozen recipe, local) | - | - | - | - | - | %.0f MB at kill (cap %s GB) | %s s at kill | RESOURCE_INFEASIBLE_LOCAL |" % (
            (mt.get("peak_rss_kb_at_kill") or 0) / 1024.0, mt.get("rss_cap_gb"), mt.get("wall_seconds_at_kill")))
    l1_rows = ["| Arm | nq | BASE_ALL | SAFE_ALL | BASE_ANY | SAFE_ANY | scope BASE / SAFE | SAFE additions (+/-) | BND_ALL | balance max/mean | ELIGIBLE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    if B:
        l1_rows.append("| PHG C1 (absolute) | %s | %.4f | %.4f | %.4f | %.4f | %.1f / %.1f | +%d/-%d | %.4f | %s | %s |" % (
            (dsn.get("population") or {}).get("nq"), B["BASE_ALL_P50"], B["SAFE_ALL_P50"], B["BASE_ANY_P50"], B["SAFE_ANY_P50"], B["BASE_SCOPE_NODES"], B["SAFE_SCOPE_NODES"],
            B["CORR"]["gained"], B["CORR"]["lost"], B["BND_ALL_P50"], B["BALANCE"]["max_over_mean"], B["BALANCE"]["ELIGIBLE"]))
    dv = R.get("diagnostic_vs_freight_closed_arm") or {}
    if dv:
        fa, fc, fb = dv["freight_absolute"], dv.get("freight_CORR") or {}, dv.get("freight_BALANCE") or {}
        l1_rows.append("| FREIGHT 1-pass (frozen, CLOSED; context) | %d | %.4f | %.4f | %.4f | %.4f | %.1f / %.1f | +%s/-%s | %.4f | %s | %s |" % (
            dv["nq"], fa["BASE_ALL_P50"], fa["SAFE_ALL_P50"], fa["BASE_ANY_P50"], fa["SAFE_ANY_P50"], fa["BASE_SCOPE_NODES"], fa["SAFE_SCOPE_NODES"], fc.get("gained"), fc.get("lost"),
            fa["BND_ALL_P50"], fb.get("max_over_mean"), fb.get("ELIGIBLE")))
    hop_rows = ["| hop | n | BASE_ALL | SAFE_ALL |", "|---|---|---|---|"] + ["| %s | %d | %.4f | %.4f |" % (h, x["n"], x["BASE_ALL"], x["SAFE_ALL"]) for h, x in (dsn.get("by_hop") or {}).items()]
    S = R.get("sanity_checks") or {}
    rep = {"RECORD": "PHG_BRIDGE_REPORT", "classification": CLASS, "dataset": ds, "arm": RL.EXPERIMENTAL_PHG_CONTRACT, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "supersedes": moved or None, "preregistration": pin(os.path.join(OUT, "PHG_BRIDGE_PREREG.json")), "ruling": pin(RULING), "run_record": pin(os.path.join(OUT, "PHG_BRIDGE_RUNS_%s.json" % ds)),
           "STATUS": "STOP_FOR_REVIEW", "DECISION": R["DECISION"], "claim": R.get("claim"), "question": pre["question"], "sanity_checks": S, "timing_seconds": T,
           "structure_gate": {kk: g.get(kk) for kk in ("gate", "N_returned", "M_with_pins", "P_returned", "weight_sum", "digest_from_queries", "ORIGINAL_STRUCTURE_SHA256",
                                                       "partition_run_dumps_identical", "zoltan_removed_or_warning_lines", "weight_reception", "problems")},
           "partition": {"metrics_final": m, "raw_validity": (s3 or {}).get("validity_raw"), "repair": {kk: s3b.get(kk) for kk in ("noop", "empty_blocks", "moves", "postconditions", "after")},
                         "memory": (s3 or {}).get("memory"), "timing": (s3 or {}).get("timing"), "repeat": R.get("repeat"), "python_km1_vs_zoltan_cutl": (s3 or {}).get("python_km1_vs_zoltan_cutl")},
           "l1_absolute": B, "population": dsn.get("population"), "by_hop": dsn.get("by_hop"), "downstream_status": dsn.get("status"),
           "context_not_claims": {"freight_one_pass_frozen": fr, "mtkahypar_blocker": mt, "diagnostic_context": dc, "diagnostic_vs_freight_closed_arm": dv},
           "host": {"before": R.get("host_before"), "after": R.get("host_after")}, "integrity": integ, "error": R.get("error"), "stopped_at_stage": R.get("stopped_at_stage"),
           "stage_table_markdown": "\n".join(stage_rows), "partition_table_markdown": "\n".join(part_rows), "l1_table_markdown": "\n".join(l1_rows), "hop_table_markdown": "\n".join(hop_rows),
           "not_done": "WebQSP / HotpotQA / 2Wiki not run (separate authorization required); no promotion; no tuning; PHG not rerun; FREIGHT stays FREIGHT_CLOSED; squad / metaqa PHG records untouched"}
    wj(rp, rep)
    md = ["# %s -- %s under the experimental PHG contract (%s)" % (CLASS, ds, RL.EXPERIMENTAL_PHG_CONTRACT), "",
          "Generated %s -- STATUS: **STOP_FOR_REVIEW** -- DECISION: **%s**" % (rep["utc"], R["DECISION"]), "",
          "Question (preregistered): %s" % pre["question"], "",
          "No Mt-KaHyPar reference exists for this corpus (%s), so this is a feasibility record: absolute L1 numbers, no equivalence claim, no comparative quality claim against Mt-KaHyPar." % (
              pre["no_reference"]["mtkahypar"]), "",
          "Stages:", "", rep["stage_table_markdown"], "", "Partition (diagnostics; the KM1 has no Mt-KaHyPar value to be compared with):", "", rep["partition_table_markdown"], "",
          "Canonical L1 on the frozen dev sample (absolute; the FREIGHT row is context from a closed arm, not a reference):", "", rep["l1_table_markdown"], ""]
    if dv:
        md += ["Diagnostic only (no rule, no claim): PHG vs the closed FREIGHT arm on the identical %d ids -- SAFE %.4f -> %.4f (%+.4f, exact p %.2e, +%d/-%d); BASE %.4f -> %.4f (%+.4f, exact p %.2e, +%d/-%d)." % (
            dv["nq"], dv["SAFE_ALL_P50"]["FREIGHT"], dv["SAFE_ALL_P50"]["PHG"], dv["SAFE_ALL_P50"]["delta_PHG_minus_FREIGHT"], dv["SAFE_ALL_P50"]["exact_p"], dv["SAFE_ALL_P50"]["PHG_only_covered"], dv["SAFE_ALL_P50"]["FREIGHT_only_covered"],
            dv["BASE_ALL_P50"]["FREIGHT"], dv["BASE_ALL_P50"]["PHG"], dv["BASE_ALL_P50"]["delta_PHG_minus_FREIGHT"], dv["BASE_ALL_P50"]["exact_p"], dv["BASE_ALL_P50"]["PHG_only_covered"], dv["BASE_ALL_P50"]["FREIGHT_only_covered"]), ""]
    if dsn.get("by_hop"):
        md += ["Hop-wise (absolute, diagnostic):", "", rep["hop_table_markdown"], ""]
    else:
        md += ["Hop-wise: the canonical musique replay carries no hop labels (by_hop absent), as in the frozen FREIGHT record.", ""]
    md += ["Sanity checks (structural, preregistered; no numeric retrieval threshold): " + "; ".join("%s = %s" % (kk, S[kk].get("pass")) for kk in ("S1_downstream_executed", "S2_population", "S3_canonical_eligibility", "S4_definitional_consistency", "S5_no_numeric_retrieval_threshold") if kk in S) + "; ALL_PASS = %s" % S.get("ALL_PASS"), "",
           "Blocker (resources, not quality): the frozen Mt-KaHyPar recipe was killed at %.0f MB RSS after %s s (cap %s GB as recorded, %s threads); PHG's peak was %s MB on the largest rank (%s MB summed over %d ranks, an upper bound) and the partition took %s s." % (
               (mt.get("peak_rss_kb_at_kill") or 0) / 1024.0, mt.get("wall_seconds_at_kill"), mt.get("rss_cap_gb"), mt.get("threads"), (dc.get("blocker_resources") or {}).get("phg_peak_rss_mb_max_rank"),
               (dc.get("blocker_resources") or {}).get("phg_peak_rss_mb_sum_ranks_upper_bound"), P.NP, (dc.get("blocker_resources") or {}).get("phg_partition_wall_seconds")), "",
           "Host: before %s; after %s." % (json.dumps(R.get("host_before")), json.dumps(R.get("host_after"))), "",
           "Integrity: frozen records changed = %s; input/data pins changed = %s; canonical musique partition still absent = %s; experimental files written = %s." % (
               integ["frozen_records_changed"] or "none", integ["data_pins_changed"] or "none", integ["canonical_musique_partition_still_absent"], integ["experimental_files_written"]), "",
           "Caveats: 4 local MPI ranks validate feasibility on this host only (no aggregate-RAM claim at larger scale); the rank count is a contract element and was not changed; "
           "PHG_NONEMPTY_REPAIR_V1 is part of the contract (no-op when no block is empty).", "", rep["not_done"], ""]
    with io.open(os.path.join(OUT, "PHG_BRIDGE_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    log("STOP_FOR_REVIEW", ds, R["DECISION"])
    print(rep["stage_table_markdown"]); print(); print(rep["partition_table_markdown"]); print(); print(rep["l1_table_markdown"])
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "prereg":
        prereg()
    elif a[0] == "run":
        for ds in a[1:]:
            run(ds)
    elif a[0] == "report":
        report()
    else:
        print(__doc__)
