"""Zoltan-PHG lane -- algorithm substitution for the canonical Mt-KaHyPar H4_SK partitioner, validated on SQuAD + MetaQA.

Question (preregistered): does canonical L1 using the PHG partition achieve at least Mt-KaHyPar-level final retrieval recall on the
identical frozen H4_SK input and identical (dev) query population?  Partition equality with Mt-KaHyPar is NOT a requirement; KM1 and
cut fractions are diagnostics only.

    python -u src/l1_lowmem/phg.py build                -> results/L1_LOWMEM/PHG_BUILD.json     (driver compiled in WSL, sha-pinned)
    python -u src/l1_lowmem/phg.py prereg               -> results/L1_LOWMEM/PHG_PREREG.json    (written once, before any run)
    python -u src/l1_lowmem/phg.py run squad metaqa     -> results/L1_LOWMEM/PHG_RUNS_<ds>.json  (stage-boundary artifacts under
                                                           data/l1_lowmem/<ds>/phg/, experimental partition parts/LOWMEM__PHG_con.*,
                                                           replay cache + L1_DOWNSTREAM_<ds>__phg.json)
    python -u src/l1_lowmem/phg.py report               -> results/L1_LOWMEM/PHG_REPORT.{json,md}  (STOP_FOR_REVIEW)
Stage boundaries (the only checkpoints this lane claims):
    1 H4_SK input verified -> 2 PHG input/callback manifest (structure gate) -> 3 completed PHG partition -> 4 P50 / replay cache -> 5 paired L1
Nothing frozen (FREIGHT records, canonical H4_SK, production partitions / caches / manifests / L1 results) is written.
"""
import io
import json
import math
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, DISTRO, log, sha_file, rj, wj, pin, ds_dir, wsl_path, wsl  # noqa: E402
from src.l1_lowmem import freight as FR  # noqa: E402
from src.l1_lowmem import hgr as HG  # noqa: E402
from src.l1_lowmem import stream as ST  # noqa: E402

ARM = "H4_SK_ZOLTAN_PHG_CONNECTIVITY_NP4"
TAG = "LOWMEM__PHG_con"
SUFFIX = "__phg"
NP = 4                                   # MPI ranks: part of the experimental contract (results may differ across rank counts)
DATASETS = ["squad", "metaqa"]
BUILD_WSL = "~/phg_build"
SRC_DIR = os.path.join(REPO, "src", "l1_lowmem", "phg_driver")
# explicit Zoltan / PHG configuration, applied in this order (NUM_GLOBAL_PARTS = k is appended by the driver)
PARAMS = [("DEBUG_LEVEL", "0"), ("NUM_GID_ENTRIES", "1"), ("NUM_LID_ENTRIES", "1"), ("OBJ_WEIGHT_DIM", "1"), ("EDGE_WEIGHT_DIM", "1"),
          ("DETERMINISTIC", "1"), ("RETURN_LISTS", "PARTS"), ("REMAP", "0"),
          ("LB_METHOD", "HYPERGRAPH"), ("HYPERGRAPH_PACKAGE", "PHG"), ("LB_APPROACH", "PARTITION"), ("IMBALANCE_TOL", "1.03"),
          ("PHG_CUT_OBJECTIVE", "CONNECTIVITY"), ("PHG_EDGE_SIZE_THRESHOLD", "1.0"), ("PHG_RANDOMIZE_INPUT", "0"), ("PHG_MULTILEVEL", "1"),
          ("PHG_COARSENING_METHOD", "AGG"), ("PHG_COARSEPARTITION_METHOD", "AUTO"), ("PHG_REFINEMENT_METHOD", "FM"), ("PHG_REFINEMENT_QUALITY", "1"),
          ("PHG_EDGE_WEIGHT_OPERATION", "ERROR"), ("PHG_PROCESSOR_REDUCTION_LIMIT", "0.5"), ("PHG_REPART_MULTIPLIER", "100"),
          ("CHECK_HYPERGRAPH", "1"), ("PHG_OUTPUT_LEVEL", "1")]
PARAM_NOTES = {"IMBALANCE_TOL": "1.03 = the canonical eps 0.03; Zoltan bounds max part weight / average, the canonical contract bounds max block by "
                                "ceil(1.03 N/k) -- the contract bound is the validity gate, Zoltan's own (never looser) tolerance is recorded",
               "PHG_EDGE_SIZE_THRESHOLD": "1.0 mandatory: the documented default 0.25 omits hyperedges larger than 25 % of the vertices; no canonical "
                                          "hyperedge may be discarded (structure gate checks what the query functions returned; Zoltan output is "
                                          "grepped for removal messages)",
               "PHG_EDGE_WEIGHT_OPERATION": "ERROR: every rank supplies the H4 weight of every net it holds a pin of; any cross-rank disagreement is a Zoltan error",
               "REMAP": "0: raw PHG part labels (the documented default 1 only renumbers parts; labels are never compared)",
               "DETERMINISTIC": "1: library parameter (present in the installed parameter table; the user guide pages fetched do not document it); "
                                "same-rank-count repeatability is MEASURED by a repeat run, not assumed",
               "PHG_RANDOMIZE_INPUT": "0 as specified", "CHECK_HYPERGRAPH": "1: Zoltan validates the hypergraph it received",
               "RETURN_LISTS": "PARTS: the new part of every local object is returned", "PHG_REPART_MULTIPLIER": "documented default; inert under LB_APPROACH=PARTITION",
               "coarsening/refinement": "AGG / FM / quality 1 / multilevel 1 / coarse-partition AUTO / processor reduction 0.5 = the quality-oriented documented "
                                        "defaults identified in AUDIT_ZOLTAN_PHG.json, set explicitly"}
UNDOCUMENTED_LEFT_AT_LIBRARY_DEFAULT = ["PHG_COARSENING_LIMIT", "PHG_COARSENING_METHOD_FAST", "PHG_COARSENING_NCANDIDATE", "PHG_REFINEMENT_LOOP_LIMIT",
                                        "PHG_REFINEMENT_MAX_NEG_MOVE", "PHG_BAL_TOL_ADJUSTMENT", "PHG_VERTEX_VISIT_ORDER", "PHG_DIRECT_KWAY", "PHG_EDGE_SCALING",
                                        "PHG_VERTEX_SCALING", "PHG_USE_TIMERS", "PHG_NPROC_VERTEX", "PHG_NPROC_EDGE", "PHG_KEEP_TREE", "PHG_MATCH_EDGE_SIZE_THRESHOLD"]
FROZEN_READ_ONLY = ["H4_SK_LOW_MEMORY_PARTITIONING_REPORT.json", "H4_SK_LOW_MEMORY_PARTITIONING_REPORT.md", "FREIGHT_RUNS_squad.json", "FREIGHT_RUNS_metaqa.json",
                    "FREIGHT_RUNS_musique.json", "FREIGHT_BUILD.json", "L1_DOWNSTREAM_squad.json", "L1_DOWNSTREAM_metaqa.json", "L1_DOWNSTREAM_musique.json",
                    "MUSIQUE_MTKAHYPAR_RETRY.json", "FREIGHT_RESTREAM_PREREG.json", "FREIGHT_RESTREAM_RUNS_squad.json", "FREIGHT_RESTREAM_RUNS_metaqa.json",
                    "FREIGHT_RESTREAM_REPORT.json", "FREIGHT_RESTREAM_REPORT.md", "L1_DOWNSTREAM_metaqa__restream2.json", "AUDIT_ZOLTAN_PHG.json",
                    "PHG_PACKAGES.json"]
GATES = {
    "1_structure": "EXACT: the union of what the Zoltan query functions returned on all ranks (dumped by the query functions themselves at their first "
                   "invocation) must reconstruct N, M, pin count, unit vertex weights, hyperedge weights, the global node positions 0..N-1 (each exactly "
                   "once), the global hyperedge ids 0..M-1 (each with >= 1 pin), the incidence multiset and the canonical H4_SK_STRUCTURE_V1 digest == "
                   "ORIGINAL_STRUCTURE_SHA256 of the frozen H4_SK_STREAM_V1 manifest; weight reception: Zoltan_LB_Eval_HG's connectivity of the "
                   "rank-ownership assignment (gate mode) == Python weighted KM1 of the same assignment within 1e-4 relative (float accumulation); "
                   "the dumps of the partition run must be byte-identical to the gate run's.  Any mismatch = STRUCTURE_MISMATCH and STOP (no "
                   "partition is used).",
    "2_partition_validity": "length N, block ids in [0, k), no empty block, max block <= ceil(1.03 N/k) (the canonical contract bound).  Failure = "
                            "PARTITION_INVALID (no downstream evaluation).  Partition ids are NOT compared with Mt-KaHyPar's.",
    "3_L1_primary": "the PHG partition goes through the unchanged canonical L1 (replay cache -> l1_eval numerics) on the identical dev query ids as the "
                    "frozen Mt-KaHyPar replay; PRIMARY = final SAFE_ALL_P50 paired exact two-sided McNemar: SIG_LOSS iff delta(PHG - MtK) < 0 and "
                    "p < 0.05, otherwise NO_SIGNIFICANT_PAIRED_LOSS (label renamed 2026-09-13 from NONINFERIOR, rule identical; no non-inferiority margin was preregistered).  SECONDARY (reported, same rule, not the decision): BASE_ALL_P50.  DIAGNOSTICS (no rule): "
                    "hop-wise paired results, candidate coverage (ANY), SAFE additions, scope nodes, balance.",
    "4_partition_metrics": "KM1 / connectivity, imbalance, STRUCT cut, KNN cut, block-size min/p50/p95/max, empty blocks, memory, runtime = diagnostics; "
                           "no numeric acceptance threshold exists or will be added after seeing the numbers."}
DECISION_RULE = ("per dataset: PHG_STRUCTURE_MISMATCH | PHG_PARTITION_INVALID | PHG_L1_NO_SIGNIFICANT_PAIRED_LOSS (formerly PHG_L1_NONINFERIOR) | PHG_L1_SIG_LOSS (from gates 1-3 in that order).  "
                 "SQuAD runs first; MetaQA runs only if SQuAD is structurally valid (gate 1 PASS) -- with the identical contract.  After both: "
                 "STOP_FOR_REVIEW.  No tuning, no second configuration, no other rank count, no other dataset.")


# ----------------------------------------------------------------------------- build
def build():
    rec_path = os.path.join(OUT, "PHG_BUILD.json")
    src = os.path.join(SRC_DIR, "phg_driver.c")
    wrap = os.path.join(SRC_DIR, "rank_wrap.sh")
    old = rj(rec_path)
    h = FR.home()
    bdir = BUILD_WSL.replace("~", h)
    if old and old["source"]["sha256"] == sha_file(src) and old["wrapper_source"]["sha256"] == sha_file(wrap):
        r = FR.sh("sha256sum %s/phg_driver %s/rank_wrap.sh" % (bdir, bdir), quiet=True).stdout.split()
        if r and r[0] == old["binary"]["sha256"] and r[2] == old["wrapper"]["sha256"]:
            log("driver build unchanged (binary %s)" % old["binary"]["sha256"][:16])
            return old
    FR.sh("mkdir -p %s && cp %s %s/phg_driver.c && cp %s %s/rank_wrap.sh && sed -i 's/\\r$//' %s/phg_driver.c %s/rank_wrap.sh && chmod +x %s/rank_wrap.sh" % (
        bdir, wsl_path(src), bdir, wsl_path(wrap), bdir, bdir, bdir, bdir), quiet=True)
    cmd = "mpicc -O2 -Wall -Wextra -I/usr/include/trilinos phg_driver.c -o phg_driver -ltrilinos_zoltan -lm"
    r = FR.sh("cd %s && %s 2>&1; echo rc=$?" % (bdir, cmd), quiet=True)
    if not r.stdout.strip().endswith("rc=0"):
        raise RuntimeError("driver build failed:\n%s" % r.stdout[-3000:])
    shas = FR.sh("cd %s && sha256sum phg_driver rank_wrap.sh phg_driver.c" % bdir, quiet=True).stdout.split()
    info = FR.sh("mpicc --version | head -1; mpirun --version | head -1; ldd %s/phg_driver | grep -E 'zoltan|mpi'; dpkg-query -W libtrilinos-zoltan-13.2 libopenmpi-dev openmpi-bin" % bdir,
                 quiet=True).stdout.strip().split("\n")
    rec = {"RECORD": "PHG_BUILD", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "distro": DISTRO, "build_dir": bdir, "compile": cmd,
           "compiler_output": r.stdout.strip().split("\n")[:-1], "source": pin(src), "wrapper_source": pin(wrap),
           "binary": {"path": "%s/phg_driver" % bdir, "sha256": shas[0]}, "wrapper": {"path": "%s/rank_wrap.sh" % bdir, "sha256": shas[2]},
           "source_in_build_dir_sha256": shas[4], "toolchain_and_libs": info, "zoltan_id_type": "unsigned int (Zoltan_config.h: no *_GLOBAL_IDS override)",
           "packages": pin(os.path.join(OUT, "PHG_PACKAGES.json"))}
    wj(rec_path, rec)
    log("driver built: binary %s  (%s)" % (rec["binary"]["sha256"][:16], os.path.relpath(rec_path, REPO)))
    return rec


# ----------------------------------------------------------------------------- preregistration
def prereg():
    pre_path = os.path.join(OUT, "PHG_PREREG.json")
    if os.path.exists(pre_path):
        raise RuntimeError("PHG preregistration already exists -- written once, before any run")
    b = build()
    pk = rj(os.path.join(OUT, "PHG_PACKAGES.json"))
    rec = {"RECORD": "PHG_PREREG", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arm": ARM, "tag": TAG,
           "question": "does canonical L1 using the PHG partition achieve at least Mt-KaHyPar-level final retrieval recall (SAFE) on the identical frozen "
                       "H4_SK input and identical dev query ids?  NOT: does PHG reproduce Mt-KaHyPar's partition.",
           "canonical_input": {"hypergraph_rule": "H4_SPLIT_PRESERVE", "families": "STRUCT + KNN", "tag": "H4_SK", "objective": "connectivity / KM1",
                               "k": "existing canonical rule max(1, N // 100)", "imbalance": "1.03 (eps 0.03)", "vertex_weights": "unit",
                               "hyperedge_weights": "H4 build weights as frozen", "node_universe": "all canonical positions, canonical order",
                               "representation": "frozen H4_SK_STREAM_V1 shards (read-only), rank r owns a contiguous shard range; the mathematical "
                                                 "hypergraph is never sharded, only its representation"},
           "phg_parameters_in_order": PARAMS, "parameter_notes": PARAM_NOTES, "NUM_GLOBAL_PARTS": "k (appended by the driver)",
           "undocumented_parameters_left_at_library_default": UNDOCUMENTED_LEFT_AT_LIBRARY_DEFAULT,
           "mpi_ranks": NP, "rank_count_caveat": "PHG's 2D layout and coarsening depend on the rank count; partitions are not assumed identical across "
                                                  "rank counts; NP is part of the contract.  Multiple LOCAL ranks validate implementation and quality "
                                                  "only -- no claim about lower aggregate host RAM at large scale is made from this run.",
           "reproducibility": "no checkpoint/restart guarantee is claimed for PHG (FREIGHT's proven exact restart does not transfer); stage-boundary "
                              "artifacts only: H4_SK input verified -> PHG input/callback manifest -> completed PHG partition -> P50 -> replay cache.  "
                              "Same-rank-count repeatability is measured by one identical repeat run (diagnostic; the FIRST run is the arm's partition).",
           "datasets_in_order": DATASETS, "query_population": "dev (EVAL_SPLITS dev of the canonical replay, the frozen hop-balanced sample rule); the "
                                                                "sealed test split is never read; identical query ids to the frozen Mt-KaHyPar replay",
           "gates": GATES, "decision_rule": DECISION_RULE,
           "measurements": ["MPI rank count", "wall time (partition, total, per rank)", "peak RSS per rank (/usr/bin/time -v + getrusage)",
                            "aggregate peak RSS = sum of per-rank peaks (upper bound, peaks need not coincide) and max per-rank peak",
                            "KM1 / connectivity (Python recomputation = authority; Zoltan_LB_Eval_HG cutl as cross-check)", "imbalance", "STRUCT cut", "KNN cut",
                            "block sizes min/p50/p95/max, empty blocks", "P50 statistics (BASE/SAFE ANY, scope nodes, BND, balance, by hop)",
                            "BASE, SAFE, candidate coverage, SAFE additions, hop-wise, exact paired McNemar vs frozen Mt-KaHyPar"],
           "packages": {"record": pin(os.path.join(OUT, "PHG_PACKAGES.json")), "dpkg_versions_after_install": pk["after"]["dpkg_versions"],
                        "mpirun": pk["after"]["commands"][1]["stdout"].split("\n")[0], "mpicxx": pk["after"]["commands"][2]["stdout"].split("\n")[0],
                        "zoltan_version_header": "ZOLTAN_VERSION_NUMBER 3.90", "authorized_downloads": pk["authorized_by_user"]},
           "build": {"record": pin(os.path.join(OUT, "PHG_BUILD.json")), "driver_source_sha256": b["source"]["sha256"], "driver_binary_sha256": b["binary"]["sha256"],
                     "wrapper_sha256": b["wrapper"]["sha256"]},
           "frozen_read_only": {fn: pin(os.path.join(OUT, fn)) for fn in FROZEN_READ_ONLY if os.path.exists(os.path.join(OUT, fn))},
           "scope": "SQuAD + MetaQA only; MuSiQue / WebQSP / HotpotQA / 2Wiki are not run; FREIGHT stays FREIGHT_CLOSED (not reopened, tuned, swept or overwritten)"}
    wj(pre_path, rec)
    log("preregistered", ARM, "->", os.path.relpath(pre_path, REPO))
    return rec


# ----------------------------------------------------------------------------- run helpers
def run_dir(ds, name):
    import shutil
    d = os.path.join(ds_dir(ds), "phg", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def write_params(d):
    p = os.path.join(d, "params.txt")
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join("%s %s\n" % kv for kv in PARAMS))
    return p


def mpirun(ds, name, mode, k, smf, b, timeout=7200):
    d = run_dir(ds, name)
    pf = write_params(d)
    wd = wsl_path(d)
    cmd = ("cd %s && PHG_TIME_DIR=%s /usr/bin/time -v -o %s/time_job.txt mpirun -np %d -x PHG_TIME_DIR %s %s %s %d %s %s %s > %s/mpirun_stdout.txt 2> %s/mpirun_stderr.txt; echo rc=$?"
           % (wd, wd, wd, NP, b["wrapper"]["path"], b["binary"]["path"], wsl_path(smf), k, mode, wd, wsl_path(pf), wd, wd))
    with io.open(os.path.join(d, "command.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(cmd + "\n")
    t = time.time()
    r = wsl(cmd, timeout=timeout)
    wall = round(time.time() - t, 2)
    m = re.search(r"rc=(\d+)\s*$", r.stdout)
    rc = int(m.group(1)) if m else -1
    ranks = []
    for i in range(NP):
        rj_ = rj(os.path.join(d, "rank%d.json" % i))
        tv = FR.parse_time_v(os.path.join(d, "time_rank%d.txt" % i))
        ranks.append({"rank": i, "json": rj_, "time_v": tv})
    so = io.open(os.path.join(d, "mpirun_stdout.txt"), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(d, "mpirun_stdout.txt")) else ""
    se = io.open(os.path.join(d, "mpirun_stderr.txt"), encoding="utf-8", errors="replace").read() if os.path.exists(os.path.join(d, "mpirun_stderr.txt")) else ""
    res = {"name": name, "mode": mode, "np": NP, "rc": rc, "wall_seconds_outer": wall, "job_time_v": FR.parse_time_v(os.path.join(d, "time_job.txt")),
           "dir": os.path.relpath(d, REPO).replace("\\", "/"), "ranks": ranks,
           "zoltan_stdout_lines": [ln for ln in so.split("\n") if ln.strip()][:80], "stderr_tail": [ln for ln in se.split("\n") if ln.strip()][-20:],
           "zoltan_removed_or_warning_lines": [ln for ln in (so + "\n" + se).split("\n") if re.search(r"remov|omit|warn|error|fatal", ln, re.I)]}
    if rc != 0 or any(x["json"] is None for x in ranks):
        raise RuntimeError("%s %s: mpirun rc %d / missing rank json; stderr tail: %s" % (ds, name, rc, res["stderr_tail"]))
    peaks = [x["time_v"].get("peak_rss_kb", 0) for x in ranks]
    rus = [x["json"]["ru_maxrss_kb"] for x in ranks]
    res["memory"] = {"peak_rss_kb_per_rank_time_v": peaks, "ru_maxrss_kb_per_rank": rus, "max_rank_peak_kb": max(peaks), "sum_of_rank_peaks_kb": int(sum(peaks)),
                     "sum_note": "sum of per-rank peaks = upper bound on the simultaneous aggregate (peaks need not coincide)"}
    r0 = ranks[0]["json"]
    res["timing"] = {"partition_wall_seconds": r0["seconds"]["partition_wall"], "driver_total_seconds_rank0": r0["seconds"]["total"],
                     "load_seconds_per_rank": [x["json"]["seconds"]["load"] for x in ranks], "job_wall_seconds": res["job_time_v"].get("wall_seconds")}
    res["global_from_queries"] = r0["global_from_queries"]
    res["params_applied_rank0"] = r0["params_applied"]
    res["zoltan_eval"] = r0["eval"]
    res["dump_sha256"] = {fn: sha_file(os.path.join(d, fn)) for fn in sorted(os.listdir(d)) if fn.endswith(".bin")}
    log("  %-10s %-9s rc=%d  partition %.2fs  job %.1fs  peak RSS/rank %s MB  queries: objects %d pins %d  eval cutl %.0f imb %.4f" % (
        name, mode, rc, res["timing"]["partition_wall_seconds"], wall, [round(p / 1024.0, 1) for p in peaks], r0["global_from_queries"]["objects"],
        r0["global_from_queries"]["pins"], r0["eval"]["cutl_global"], r0["eval"]["imbalance"]))
    return res, d


def read_dumps(d):
    objs, cs, ew = [], [], []
    for i in range(NP):
        with open(os.path.join(d, "objs_rank%d.bin" % i), "rb") as f:
            n, wdim = np.frombuffer(f.read(8), dtype=np.int32)
            g = np.frombuffer(f.read(4 * int(n)), dtype=np.uint32)
            w = np.frombuffer(f.read(4 * int(n) * int(wdim)), dtype=np.float32).reshape(int(n), int(wdim)) if wdim > 0 else None
            objs.append((g, w))
        with open(os.path.join(d, "cs_rank%d.bin" % i), "rb") as f:
            nv, npn, fmt = np.frombuffer(f.read(12), dtype=np.int32)
            g = np.frombuffer(f.read(4 * int(nv)), dtype=np.uint32)
            ptr = np.frombuffer(f.read(4 * int(nv)), dtype=np.int32)
            pins = np.frombuffer(f.read(4 * int(npn)), dtype=np.uint32)
            cs.append((int(fmt), g, ptr, pins))
        with open(os.path.join(d, "ewts_rank%d.bin" % i), "rb") as f:
            ne, dim = np.frombuffer(f.read(8), dtype=np.int32)
            g = np.frombuffer(f.read(4 * int(ne)), dtype=np.uint32)
            w = np.frombuffer(f.read(4 * int(ne) * int(dim)), dtype=np.float32).reshape(int(ne), int(dim)) if dim > 0 else None
            ew.append((g, w))
    return objs, cs, ew


def structure_gate(ds, d, man, H=None, zoltan_eval=None):
    """exact reconstruction of the global H4_SK from what the query functions returned on all ranks."""
    N, M, P = man["N"], man["M"], man["P"]
    objs, cs, ew = read_dumps(d)
    G = {"gate": "PASS", "problems": [], "per_rank": []}
    all_obj = np.concatenate([g for g, _ in objs]).astype(np.int64)
    if len(all_obj) != N or not np.array_equal(np.sort(all_obj), np.arange(N)):
        G["problems"].append("objects: %d returned, expected the positions 0..%d exactly once" % (len(all_obj), N - 1))
    elif H is not None and zoltan_eval is not None:
        # weight reception: Zoltan's own connectivity metric of the rank-ownership assignment (gate mode, parts = ranks) must equal
        # the weighted KM1 recomputed in Python from the frozen CSR (preregistered float tolerance 1e-4 relative)
        own = np.empty(N, np.int64)
        for i, (g, _) in enumerate(objs):
            own[g.astype(np.int64)] = i
        py = FR.km1_metrics(H, own)["km1_weighted"]
        zc = float(zoltan_eval["cutl_global"])
        G["weight_reception"] = {"python_weighted_km1_of_rank_ownership": int(py), "zoltan_cutl_gate_mode": zc,
                                 "relative_diff": round(abs(py - zc) / max(1.0, py), 9), "tolerance_relative": 1e-4}
        if abs(py - zc) > 1e-4 * max(1.0, py):
            G["problems"].append("Zoltan connectivity of the rank ownership (%.0f) != Python weighted KM1 (%d): hyperedge weights not received as given" % (zc, py))
    for i, (g, w) in enumerate(objs):
        if w is None or w.shape[1] < 1 or not np.all(w[:, 0] == 1.0) or (w.shape[1] > 1 and not np.all(w[:, 1:] == 0.0)):
            G["problems"].append("rank %d: vertex weights are not unit" % i)
    vtx, net = [], []
    for i, (fmt, g, ptr, pins) in enumerate(cs):
        if fmt != 2:
            G["problems"].append("rank %d: format %d is not ZOLTAN_COMPRESSED_VERTEX" % (i, fmt))
        if not np.array_equal(np.sort(g), np.sort(objs[i][0])):
            G["problems"].append("rank %d: compressed-vertex ids differ from the rank's object ids" % i)
        if len(ptr) and (ptr[0] != 0 or np.any(np.diff(ptr) < 0) or ptr[-1] > len(pins)):
            G["problems"].append("rank %d: vtxedge_ptr not a valid CSR pointer" % i)
        if len(pins) and pins.max() >= M:
            G["problems"].append("rank %d: hyperedge id >= M" % i)
        deg = np.diff(np.append(ptr, len(pins)))
        vtx.append(np.repeat(g.astype(np.int64), deg)); net.append(pins.astype(np.int64))
        G["per_rank"].append({"rank": i, "objects": int(len(g)), "pins": int(len(pins)), "edge_weight_entries": int(len(ew[i][0]))})
    vtx, net = np.concatenate(vtx), np.concatenate(net)
    if len(net) != P:
        G["problems"].append("pins: %d returned, expected P=%d" % (len(net), P))
    wnet = np.full(M, -1, np.int64)
    for i, (g, w) in enumerate(ew):
        if w is None or w.shape[1] < 1:
            G["problems"].append("rank %d: no edge weights" % i); continue
        wi = w[:, 0].astype(np.float64)
        if not np.all(wi == np.rint(wi)) or np.any(wi < 1):
            G["problems"].append("rank %d: non-integral or non-positive hyperedge weight" % i)
        gi = g.astype(np.int64)
        if len(gi) != len(np.unique(gi)):
            G["problems"].append("rank %d: duplicate hyperedge in the weight list" % i)
        seen = wnet[gi] >= 0
        if np.any(wnet[gi][seen] != np.rint(wi[seen]).astype(np.int64)):
            G["problems"].append("rank %d: hyperedge weight disagrees with another rank" % i)
        wnet[gi] = np.rint(wi).astype(np.int64)
        # every net the rank holds a pin of must carry a weight from that rank
        if not np.array_equal(np.unique(cs[i][3].astype(np.int64)), np.sort(gi)):
            G["problems"].append("rank %d: weighted hyperedge set != hyperedge set of the rank's pins" % i)
    if np.any(wnet < 0):
        G["problems"].append("hyperedges without weight: %d" % int((wnet < 0).sum()))
    # incidence multiset -> canonical digest
    order = np.argsort(net * np.int64(N) + vtx, kind="stable")
    net_s, vtx_s = net[order], vtx[order]
    if len(net_s) and np.any(np.diff(net_s * np.int64(N) + vtx_s) == 0):
        G["problems"].append("duplicate pin (same vertex twice in one hyperedge)")
    cnt = np.bincount(net_s, minlength=M) if len(net_s) else np.zeros(M, np.int64)
    if len(cnt) != M or np.any(cnt == 0):
        G["problems"].append("hyperedge ids: %d of %d have no pin (or an id >= M was returned)" % (int((cnt == 0).sum()) if len(cnt) == M else -1, M))
    ptr = np.zeros(M + 1, np.int64); ptr[1:] = np.cumsum(cnt[:M])
    if not G["problems"]:
        def it():
            for e in range(M):
                yield e, int(wnet[e]), vtx_s[ptr[e]:ptr[e + 1]]
        dig = HG.digest_stream(N, M, len(net_s), it())
    else:
        dig = None
    G.update({"N_returned": int(len(all_obj)), "M_with_pins": int((cnt > 0).sum()) if len(cnt) == M else None, "P_returned": int(len(net)),
              "weight_sum": int(wnet[wnet >= 0].sum()), "weight_min": int(wnet[wnet >= 0].min()) if np.any(wnet >= 0) else None,
              "weight_max": int(wnet.max()), "vertex_weights": "unit" if not any("unit" in p for p in G["problems"]) else "NOT unit",
              "digest_from_queries": dig, "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "expected": {"N": N, "M": M, "P": P}})
    if dig != man["ORIGINAL_STRUCTURE_SHA256"]:
        G["problems"].append("digest mismatch: %s != ORIGINAL %s" % (dig, man["ORIGINAL_STRUCTURE_SHA256"]))
    G["gate"] = "PASS" if not G["problems"] else "STRUCTURE_MISMATCH"
    return G


def assemble_partition(d, N, k):
    gid, part = [], []
    for i in range(NP):
        a = np.loadtxt(os.path.join(d, "part_rank%d.txt" % i), dtype=np.int64, ndmin=2)
        if a.size:
            gid.append(a[:, 0]); part.append(a[:, 1])
    gid, part = np.concatenate(gid), np.concatenate(part)
    if len(gid) != N or not np.array_equal(np.sort(gid), np.arange(N)):
        raise RuntimeError("part files do not cover every position exactly once (%d rows)" % len(gid))
    hard = np.empty(N, np.int64); hard[gid] = part
    return hard


def validity(hard, N, k):
    sizes = np.bincount(hard, minlength=k) if hard.min() >= 0 and hard.max() < k else None
    bound = int(math.ceil(1.03 * N / float(k)))
    v = {"length_ok": len(hard) == N, "ids_in_range": bool(hard.min() >= 0 and hard.max() < k),
         "empty_blocks": int((sizes == 0).sum()) if sizes is not None else None, "max_block": int(sizes.max()) if sizes is not None else None,
         "contract_bound_ceil_1.03_N_over_k": bound, "zoltan_tolerance_bound_1.03_x_mean": round(1.03 * N / float(k), 3)}
    v["gate"] = "PASS" if v["length_ok"] and v["ids_in_range"] and v["empty_blocks"] == 0 and v["max_block"] <= bound else "PARTITION_INVALID"
    return v


def import_partition(ds, H, hard, R, b, off, man, gate, tag=TAG, extra=None):
    d = H["d"]
    pdir = os.path.join(d.derived_dir, "parts")
    os.makedirs(pdir, exist_ok=True)
    npy = os.path.join(pdir, "%s.npy" % tag)
    np.save(npy, hard.astype(np.int64))
    m = R["metrics"]
    rec = {"dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "EXPERIMENTAL": True, "arm": ARM,
           "not_canonical": "L1_LOWMEM validation arm (Zoltan-PHG); the canonical partition is parts/H4_SK.*; nothing canonical was modified",
           "contract": {"partitioner": "Zoltan-PHG (Trilinos %s, ZOLTAN_VERSION_NUMBER 3.90, Ubuntu 22.04 apt libtrilinos-zoltan-dev)" % b["packages_versions"].get("libtrilinos-zoltan-dev:amd64"),
                        "mpi": "Open MPI %s, %d ranks (contract element)" % (b["packages_versions"].get("openmpi-bin"), NP),
                        "worker": b["binary"]["path"], "worker_sha256": b["binary"]["sha256"],
                        "worker_note": "src/l1_lowmem/phg_driver/phg_driver.c (sha %s) driven by src/l1_lowmem/phg.py; rank r owns a contiguous H4_SK_STREAM_V1 shard range" % b["source"]["sha256"][:16],
                        "algorithm": "multilevel PHG: agglomerative coarsening (AGG), AUTO coarse partition, FM refinement quality 1, recursive scheme as shipped",
                        "objective": "connectivity (lambda-1) = weighted KM1 (PHG_CUT_OBJECTIVE=CONNECTIVITY)", "epsilon": 0.03, "imbalance_percent": 3,
                        "IMBALANCE_TOL": "1.03", "seed": "none exposed by Zoltan; DETERMINISTIC=1; repeat-run diagnostic recorded", "num_streams_passes": None,
                        "restream_vcycle": None, "parameters": dict(PARAMS), "NUM_GLOBAL_PARTS": H["k"], "mpi_ranks": NP,
                        "vertex_weights": "unit", "hyperedge_weights": "H4 build (max(1, rint(1000/(|e|-1))))",
                        "k": H["k"], "k_rule": "max(1, N // 100)", "hypergraph_rule": "H4_SPLIT_PRESERVE", "families": H["meta"]["families"],
                        "node_distribution": "contiguous canonical position ranges (H4_SK_STREAM_V1 shards) per rank; PHG_RANDOMIZE_INPUT=0"},
           "inputs": {"DATASET_json_RECORD_SHA256": d.record_sha, "hypergraph_file": os.path.relpath(H["npz"], REPO).replace("\\", "/"),
                      "hypergraph_sha256": H["npz_sha256"], "hypergraph_content_digest": H["meta"].get("content_digest"),
                      "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "official_netl_sha256": off["output"]["sha256"],
                      "stream_manifest_sha256": R["inputs"]["stream_manifest"]["sha256"], "structure_gate": gate["gate"],
                      "digest_from_queries": gate["digest_from_queries"], "hyperedges": H["M"], "pins": H["P"], "N": H["N"]},
           "STATUS": "OK",
           "worker_stats": {"N": H["N"], "k": H["k"], "hyperedges": H["M"], "pins": H["P"], "objective_km1": m["km1_weighted"], "objective_cut": m["cut_weighted"],
                            "zoltan_eval_cutl": R["runs"]["R1"]["zoltan_eval"]["cutl_global"], "mpi_ranks": NP,
                            "peak_rss_mb_max_rank": round(R["runs"]["R1"]["memory"]["max_rank_peak_kb"] / 1024.0, 1),
                            "peak_rss_mb_sum_ranks": round(R["runs"]["R1"]["memory"]["sum_of_rank_peaks_kb"] / 1024.0, 1),
                            "wall_seconds_partition": R["runs"]["R1"]["timing"]["partition_wall_seconds"], "wall_seconds_job": R["runs"]["R1"]["timing"]["job_wall_seconds"],
                            "ran_on": "local_wsl"},
           "post_checks": {"length_ok": True, "min_block": int(hard.min()), "max_block": int(hard.max()), "blocks_used": m["blocks"]["used"], "k": H["k"],
                           "every_block_used": m["blocks"]["empty"] == 0, "size_min": m["blocks"]["min"], "size_max": m["blocks"]["max"],
                           "size_mean": m["blocks"]["mean"], "ceil_N_over_k": m["blocks"]["ceil_N_over_k"], "balance_max_over_mean": m["blocks"]["max_over_mean"],
                           "balance_within_eps": m["blocks"]["within_eps_0.03"]},
           "repeatability": R["repeat"],
           "output": {"file": os.path.relpath(npy, REPO).replace("\\", "/"), "bytes": os.path.getsize(npy), "sha256": sha_file(npy), "dtype": "int64", "n": int(len(hard))}}
    from src.l1_canonical.adapter import contract_hash
    rec["L1_CONTRACT_SHA256"] = contract_hash()["L1_CONTRACT_SHA256"]
    if extra:
        rec.update(extra)
    wj(os.path.join(pdir, "%s.json" % tag), rec)
    log("  imported parts/%s.npy (%s)" % (tag, rec["output"]["sha256"][:16]))
    return rec


# ----------------------------------------------------------------------------- run
def run(ds):
    pre = rj(os.path.join(OUT, "PHG_PREREG.json"))
    if pre is None:
        raise RuntimeError("run `prereg` first")
    if ds not in DATASETS:
        raise RuntimeError("%s is not in the preregistered dataset list %s" % (ds, DATASETS))
    if ds != DATASETS[0]:
        prev = rj(os.path.join(OUT, "PHG_RUNS_%s.json" % DATASETS[0]))
        if prev is None or prev["stages"]["2_structure_gate"]["gate"] != "PASS":
            raise RuntimeError("%s: SQuAD must be structurally valid before MetaQA runs" % ds)
    b = build()
    if b["binary"]["sha256"] != pre["build"]["driver_binary_sha256"] or b["source"]["sha256"] != pre["build"]["driver_source_sha256"]:
        raise RuntimeError("driver changed since preregistration")
    pk = rj(os.path.join(OUT, "PHG_PACKAGES.json"))
    b["packages_versions"] = pk["after"]["dpkg_versions"]
    # ---- stage 1: H4_SK input verified (frozen shards / official netl / semantic gate; stream manifest as frozen)
    H, man, gate_rec, off, netl = FR.load_inputs(ds)
    N, k, M, P = H["N"], H["k"], H["M"], H["P"]
    R1 = rj(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds))
    smf = os.path.join(ds_dir(ds), "freight", "stream_manifest.txt")
    if R1 is None or sha_file(smf) != R1["inputs"]["stream_manifest"]["sha256"]:
        raise RuntimeError("%s: stream manifest missing or changed since the frozen record" % ds)
    stage1 = {"stage": "1_H4_SK_input_verified", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "N": N, "M": M, "P": P, "k": k,
              "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "hypergraph_content_digest": H["meta"].get("content_digest"),
              "H4_SK_npz_sha256": H["npz_sha256"], "reconstruction_gate": {kk: gate_rec[kk] for kk in ("gate_digest", "gate_bytes") if kk in gate_rec},
              "shards": [{"index": c["index"], "sha256": c["sha256"], "nodes": c["nodes"], "pins": c["pins"]} for c in man["shards"]],
              "stream_manifest": R1["inputs"]["stream_manifest"], "official_netl_sha256": off["output"]["sha256"]}
    log("=== %s %s: N %d M %d P %d k %d shards %d ranks %d ===" % (ARM, ds, N, M, P, k, len(man["shards"]), NP))
    R = {"RECORD": "PHG_RUNS", "arm": ARM, "dataset": ds, "utc": stage1["utc"], "preregistration": pin(os.path.join(OUT, "PHG_PREREG.json")),
         "inputs": {"N": N, "M": M, "P": P, "k": k, "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "official_netl": off["output"],
                    "stream_manifest": R1["inputs"]["stream_manifest"]},
         "contract": {"parameters": PARAMS, "NUM_GLOBAL_PARTS": k, "mpi_ranks": NP, "driver": b["binary"], "driver_source": b["source"],
                      "packages": b["packages_versions"]},
         "stages": {"1_input_verified": stage1}, "runs": {}}
    wj(os.path.join(ds_dir(ds), "phg", "STAGE1_INPUT_VERIFIED.json"), stage1)
    # ---- stage 2: PHG input / callback manifest = structure gate (Zoltan pulls the hypergraph through the query functions; no partition)
    g_res, g_dir = mpirun(ds, "G", "gate", k, smf, b)
    R["runs"]["G"] = g_res
    gate = structure_gate(ds, g_dir, man, H=H, zoltan_eval=g_res["zoltan_eval"])
    gate.update({"stage": "2_PHG_input_callback_manifest", "run": "G", "dump_sha256": g_res["dump_sha256"], "global_from_queries": g_res["global_from_queries"],
                 "zoltan_removed_or_warning_lines": g_res["zoltan_removed_or_warning_lines"]})
    R["stages"]["2_structure_gate"] = gate
    wj(os.path.join(ds_dir(ds), "phg", "STAGE2_PHG_INPUT_CALLBACK_MANIFEST.json"), gate)
    log("  structure gate: %s  (N %s M %s P %s weights %d..%d sum %d; digest %s)" % (
        gate["gate"], gate["N_returned"], gate["M_with_pins"], gate["P_returned"], gate["weight_min"] or -1, gate["weight_max"], gate["weight_sum"],
        (gate["digest_from_queries"] or "-")[:16]))
    if gate["gate"] != "PASS":
        R["DECISION"] = "PHG_STRUCTURE_MISMATCH"; R["problems"] = gate["problems"]
        wj(os.path.join(OUT, "PHG_RUNS_%s.json" % ds), R)
        log("  STOP: STRUCTURE_MISMATCH", gate["problems"][:5])
        return R
    # ---- stage 3: completed PHG partition (R1 = the arm's partition; R1_repeat = same-rank-count repeatability diagnostic)
    r1, d1 = mpirun(ds, "R1", "partition", k, smf, b)
    R["runs"]["R1"] = r1
    r1["dumps_identical_to_gate_run"] = r1["dump_sha256"] == g_res["dump_sha256"]
    if not r1["dumps_identical_to_gate_run"]:
        R["DECISION"] = "PHG_STRUCTURE_MISMATCH"; R["problems"] = ["partition-run query dumps differ from the gate run's"]
        wj(os.path.join(OUT, "PHG_RUNS_%s.json" % ds), R)
        log("  STOP: partition-run dumps differ from the gate run")
        return R
    hard = assemble_partition(d1, N, k)
    v = validity(hard, N, k)
    m = FR.km1_metrics(H, hard)
    m["family_cuts"] = FR.family_cuts(ds, hard)
    R["metrics"] = m
    R["stages"]["2_structure_gate"]["partition_run_dumps_identical"] = True
    r2, d2 = mpirun(ds, "R1_repeat", "partition", k, smf, b)
    R["runs"]["R1_repeat"] = r2
    hard2 = assemble_partition(d2, N, k)
    R["repeat"] = {"identical_partition": bool(np.array_equal(hard, hard2)), "n_diff": int((hard != hard2).sum()),
                   "km1_repeat": FR.km1_metrics(H, hard2)["km1_weighted"], "note": "same NP, same input, same parameters; diagnostic only -- R1 is the arm's partition"}
    zc = r1["zoltan_eval"]["cutl_global"]
    stage3 = {"stage": "3_completed_PHG_partition", "run": "R1", "validity": v, "metrics": m, "partition_sha256_txt": {i: sha_file(os.path.join(d1, "part_rank%d.txt" % i)) for i in range(NP)},
              "zoltan_eval": r1["zoltan_eval"], "python_km1_vs_zoltan_cutl": {"python_km1": m["km1_weighted"], "zoltan_cutl_float": zc,
                                                                               "relative_diff": round(abs(m["km1_weighted"] - zc) / max(1.0, m["km1_weighted"]), 8),
                                                                               "consistent_within_float32": bool(abs(m["km1_weighted"] - zc) <= max(1.0, 1e-6 * m["km1_weighted"]))},
              "memory": r1["memory"], "timing": r1["timing"], "repeat": R["repeat"], "mpi_ranks": NP}
    R["stages"]["3_partition"] = stage3
    wj(os.path.join(ds_dir(ds), "phg", "STAGE3_PHG_PARTITION.json"), stage3)
    bm = FR.baseline_metrics(ds, H)
    R["mtkahypar_baseline"] = {kk: bm.get(kk) for kk in ("status", "km1_weighted", "cut_weighted", "blocks", "family_cuts", "manifest")}
    fr = R1.get("metrics_B") or {}
    R["freight_one_pass_frozen"] = {"km1_weighted": fr.get("km1_weighted"), "cut_weighted": fr.get("cut_weighted"), "blocks": fr.get("blocks"), "family_cuts": fr.get("family_cuts")}
    if bm.get("status") == "OK":
        R["diagnostic_vs_mtkahypar"] = {"km1_ratio": round(m["km1_weighted"] / float(bm["km1_weighted"]), 4),
                                        "struct_cut_delta": round(m["family_cuts"]["STRUCT"]["edge_cut_fraction"] - bm["family_cuts"]["STRUCT"]["edge_cut_fraction"], 4),
                                        "knn_cut_delta": round(m["family_cuts"]["KNN"]["edge_cut_fraction"] - bm["family_cuts"]["KNN"]["edge_cut_fraction"], 4),
                                        "mtkahypar_peak_rss_mb": (bm.get("manifest") or {}).get("worker_stats", {}).get("peak_rss_mb"),
                                        "mtkahypar_wall_seconds": (bm.get("manifest") or {}).get("worker_stats", {}).get("wall_seconds")}
    log("%-7s partition: validity %s (max %d, bound %d, empty %d)  km1 %d (Mt-KaHyPar %s, x%s; FREIGHT 1-pass %s)  STRUCT %.4f KNN %.4f  repeat identical %s  zoltan cutl %.0f" % (
        ds, v["gate"], v["max_block"], v["contract_bound_ceil_1.03_N_over_k"], v["empty_blocks"], m["km1_weighted"], bm.get("km1_weighted"),
        (R.get("diagnostic_vs_mtkahypar") or {}).get("km1_ratio"), fr.get("km1_weighted"), m["family_cuts"]["STRUCT"]["edge_cut_fraction"],
        m["family_cuts"]["KNN"]["edge_cut_fraction"], R["repeat"]["identical_partition"], zc))
    if v["gate"] != "PASS":
        R["DECISION"] = "PHG_PARTITION_INVALID"
        wj(os.path.join(OUT, "PHG_RUNS_%s.json" % ds), R)
        return R
    # ---- stage 4/5: experimental import -> canonical replay cache (P50) -> paired L1 on identical dev query ids
    R["import"] = import_partition(ds, H, hard, R, b, off, man, gate)
    from src.l1_lowmem import l1_downstream as LD
    D = LD.run(ds, tag=TAG, suffix=SUFFIX)
    R["downstream"] = {"status": D["status"], "record": pin(os.path.join(OUT, "L1_DOWNSTREAM_%s%s.json" % (ds, SUFFIX))), "paired": D.get("paired"),
                       "B_phg": D.get("B"), "A_mtkahypar": D.get("A"), "by_hop_paired": D.get("by_hop_paired"), "coverage": D.get("coverage"),
                       "safe_additions": D.get("safe_additions"), "population": D.get("population_B")}
    wj(os.path.join(ds_dir(ds), "phg", "STAGE4_5_P50_REPLAY_PAIRED.json"), R["downstream"])
    p = D.get("paired") or {}
    if D["status"] != "PAIRED":
        R["DECISION"] = "PHG_L1_UNPAIRED (%s)" % D["status"]
    else:
        s, bsec = p["SAFE_ALL_P50"], p["BASE_ALL_P50"]
        R["l1_primary_SAFE"] = {"delta_phg_minus_mtk": s["delta_B_minus_A"], "p": s["mcnemar_p"], "sig": s["sig"], "phg_only": s["B_only_covered"], "mtk_only": s["A_only_covered"],
                                "verdict": "SIG_LOSS" if (s["sig"] and s["delta_B_minus_A"] < 0) else "NO_SIGNIFICANT_PAIRED_LOSS"}
        R["l1_secondary_BASE"] = {"delta_phg_minus_mtk": bsec["delta_B_minus_A"], "p": bsec["mcnemar_p"], "sig": bsec["sig"], "phg_only": bsec["B_only_covered"],
                                  "mtk_only": bsec["A_only_covered"], "verdict": "SIG_LOSS" if (bsec["sig"] and bsec["delta_B_minus_A"] < 0) else "NO_SIGNIFICANT_PAIRED_LOSS"}
        R["DECISION"] = "PHG_L1_" + R["l1_primary_SAFE"]["verdict"]
    fp = os.path.join(OUT, "PHG_RUNS_%s.json" % ds)
    wj(fp, R)
    log("%-7s DECISION %s  (SAFE %s p=%s; BASE %s p=%s)  -> %s" % (ds, R["DECISION"], (R.get("l1_primary_SAFE") or {}).get("delta_phg_minus_mtk"),
                                                                   (R.get("l1_primary_SAFE") or {}).get("p"), (R.get("l1_secondary_BASE") or {}).get("delta_phg_minus_mtk"),
                                                                   (R.get("l1_secondary_BASE") or {}).get("p"), os.path.relpath(fp, REPO)))
    return R


# ----------------------------------------------------------------------------- report
def report():
    if os.path.exists(os.path.join(OUT, "PHG_REPAIR_PREREG.json")):
        raise RuntimeError("PHG_REPORT is superseded by phg_repair.py report (terminology amendment + repair follow-up)")
    pre = rj(os.path.join(OUT, "PHG_PREREG.json"))
    recs = {ds: rj(os.path.join(OUT, "PHG_RUNS_%s.json" % ds)) for ds in DATASETS}
    recs = {ds: R for ds, R in recs.items() if R}
    rows = ["| Dataset | Arm | Ranks | Partition wall | Peak RSS max rank / sum | KM1 | KM1 ratio vs MtK | STRUCT cut | KNN cut | blocks min/p50/p95/max (bound) | empty | repeat identical | SAFE (paired) | BASE (paired) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    integ = {"frozen_records_changed_since_prereg": [fn for fn, pn in pre["frozen_read_only"].items()
                                                      if not os.path.exists(os.path.join(OUT, fn)) or sha_file(os.path.join(OUT, fn)) != pn["sha256"]],
             "canonical_H4_SK_partition": {}}
    from src.l1_canonical.adapter import CanonicalDataset
    frozen_report = rj(os.path.join(OUT, "H4_SK_LOW_MEMORY_PARTITIONING_REPORT.json")) or {}
    cu = frozen_report.get("canonical_untouched") or {}
    for ds, R in recs.items():
        d = CanonicalDataset(ds)
        now = sha_file(os.path.join(d.derived_dir, "parts", "H4_SK.npy"))
        integ["canonical_H4_SK_partition"][ds] = {"sha256_now": now, "pinned_by_frozen_report": (cu.get(ds) or {}).get("H4_SK_partition_sha256"),
                                                  "unchanged": now == (cu.get(ds) or {}).get("H4_SK_partition_sha256")}
        m, v = R.get("metrics"), (R["stages"].get("3_partition") or {}).get("validity")
        bm = R.get("mtkahypar_baseline") or {}
        pr = (R.get("downstream") or {}).get("paired") or {}
        def pf(key):
            x = pr.get(key)
            return "%.4f -> %.4f (%+.4f, p=%s, +%d/-%d)" % (x["A"], x["B"], x["delta_B_minus_A"], x["mcnemar_p"], x["B_only_covered"], x["A_only_covered"]) if x else "-"
        if m:
            r1 = R["runs"]["R1"]
            rows.append("| %s | PHG (%s) | %d | %.2f s | %.0f / %.0f MB | %d | %s | %.4f | %.4f | %d/%.0f/%.0f/%d (%d) | %d | %s | %s | %s |" % (
                ds, R["DECISION"], NP, r1["timing"]["partition_wall_seconds"], r1["memory"]["max_rank_peak_kb"] / 1024.0, r1["memory"]["sum_of_rank_peaks_kb"] / 1024.0,
                m["km1_weighted"], (R.get("diagnostic_vs_mtkahypar") or {}).get("km1_ratio", "-"), m["family_cuts"]["STRUCT"]["edge_cut_fraction"],
                m["family_cuts"]["KNN"]["edge_cut_fraction"], m["blocks"]["min"], m["blocks"]["p50"], m["blocks"]["p95"], m["blocks"]["max"],
                v["contract_bound_ceil_1.03_N_over_k"], m["blocks"]["empty"], R["repeat"]["identical_partition"], pf("SAFE_ALL_P50"), pf("BASE_ALL_P50")))
        else:
            rows.append("| %s | PHG (%s) | %d | - | - | - | - | - | - | - | - | - | - | - |" % (ds, R["DECISION"], NP))
        if bm.get("status") == "OK":
            ws = (bm.get("manifest") or {}).get("worker_stats") or {}
            rows.append("| %s | Mt-KaHyPar (canonical, frozen) | - | %s s | %s MB | %d | 1.0 | %.4f | %.4f | %d/%.0f/%.0f/%d | %d | - | (reference) | (reference) |" % (
                ds, ws.get("wall_seconds", "-"), ws.get("peak_rss_mb", "-"), bm["km1_weighted"], bm["family_cuts"]["STRUCT"]["edge_cut_fraction"],
                bm["family_cuts"]["KNN"]["edge_cut_fraction"], bm["blocks"]["min"], bm["blocks"]["p50"], bm["blocks"]["p95"], bm["blocks"]["max"], bm["blocks"]["empty"]))
        fr = R.get("freight_one_pass_frozen") or {}
        if fr.get("km1_weighted"):
            rows.append("| %s | FREIGHT 1-pass (frozen, CLOSED) | 1 | - | - | %d | %s | %.4f | %.4f | %d/%.0f/%.0f/%d | %d | - | (frozen record) | (frozen record) |" % (
                ds, fr["km1_weighted"], round(fr["km1_weighted"] / float(bm["km1_weighted"]), 4) if bm.get("status") == "OK" else "-",
                fr["family_cuts"]["STRUCT"]["edge_cut_fraction"], fr["family_cuts"]["KNN"]["edge_cut_fraction"], fr["blocks"]["min"], fr["blocks"]["p50"],
                fr["blocks"]["p95"], fr["blocks"]["max"], fr["blocks"]["empty"]))
    hop_rows = []
    for ds, R in recs.items():
        bh = (R.get("downstream") or {}).get("by_hop_paired") or {}
        for h, x in bh.items():
            hop_rows.append("| %s | %s | %d | %.4f -> %.4f (%+.4f, p=%s) | %.4f -> %.4f (%+.4f, p=%s) |" % (
                ds, h, x["n"], x["SAFE"]["A"], x["SAFE"]["B"], x["SAFE"]["delta_B_minus_A"], x["SAFE"]["mcnemar_p"], x["BASE"]["A"], x["BASE"]["B"],
                x["BASE"]["delta_B_minus_A"], x["BASE"]["mcnemar_p"]))
    cov_rows = []
    for ds, R in recs.items():
        dsn = R.get("downstream") or {}
        c, sa, B, A = dsn.get("coverage") or {}, dsn.get("safe_additions") or {}, dsn.get("B_phg") or {}, dsn.get("A_mtkahypar") or {}
        if c:
            cov_rows.append("| %s | %.4f / %.4f | %.4f / %.4f | %.1f / %.1f | %s / %s | %s / %s | %s / %s |" % (
                ds, c["SAFE_ANY"]["A"], c["SAFE_ANY"]["B"], c["BASE_ANY"]["A"], c["BASE_ANY"]["B"], c["SAFE_SCOPE_NODES"]["A"], c["SAFE_SCOPE_NODES"]["B"],
                sa.get("A"), sa.get("B"), A.get("BND_ALL_P50"), B.get("BND_ALL_P50"), (A.get("BALANCE") or {}).get("max_over_mean"), (B.get("BALANCE") or {}).get("max_over_mean")))
    rep = {"RECORD": "PHG_REPORT", "arm": ARM, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "preregistration": pin(os.path.join(OUT, "PHG_PREREG.json")),
           "STATUS": "STOP_FOR_REVIEW", "decisions": {ds: R["DECISION"] for ds, R in recs.items()},
           "l1_primary_SAFE": {ds: R.get("l1_primary_SAFE") for ds, R in recs.items()}, "l1_secondary_BASE": {ds: R.get("l1_secondary_BASE") for ds, R in recs.items()},
           "gates": GATES, "decision_rule": DECISION_RULE, "integrity": integ,
           "datasets": {ds: {kk: R.get(kk) for kk in ("DECISION", "metrics", "mtkahypar_baseline", "freight_one_pass_frozen", "diagnostic_vs_mtkahypar", "repeat", "downstream",
                                                       "l1_primary_SAFE", "l1_secondary_BASE", "problems")} for ds, R in recs.items()},
           "stage3_summaries": {ds: {kk: (R["stages"].get("3_partition") or {}).get(kk) for kk in ("validity", "memory", "timing", "python_km1_vs_zoltan_cutl")} for ds, R in recs.items()},
           "structure_gates": {ds: {kk: R["stages"]["2_structure_gate"].get(kk) for kk in ("gate", "N_returned", "M_with_pins", "P_returned", "weight_sum", "digest_from_queries",
                                                                                          "ORIGINAL_STRUCTURE_SHA256", "partition_run_dumps_identical", "zoltan_removed_or_warning_lines", "problems")}
                               for ds, R in recs.items()},
           "table_markdown": "\n".join(rows), "hop_table_markdown": "\n".join(["| Dataset | hop | n | SAFE MtK -> PHG | BASE MtK -> PHG |", "|---|---|---|---|---|"] + hop_rows),
           "coverage_table_markdown": "\n".join(["| Dataset | SAFE_ANY MtK / PHG | BASE_ANY MtK / PHG | SAFE scope nodes | SAFE additions MtK / PHG | BND_ALL MtK / PHG | balance max/mean MtK / PHG |",
                                                 "|---|---|---|---|---|---|---|"] + cov_rows),
           "not_done": "MuSiQue / WebQSP / HotpotQA / 2Wiki not run; no promotion; no tuning; FREIGHT stays FREIGHT_CLOSED"}
    obs = {}
    for ds, R in recs.items():
        if R["DECISION"] == "PHG_PARTITION_INVALID":
            from src.l1_canonical.adapter import CanonicalDataset as CD
            d = CD(ds)
            hard = np.load(os.path.join(d.derived_dir, "parts", "%s.npy" % TAG)) if os.path.exists(os.path.join(d.derived_dir, "parts", "%s.npy" % TAG)) else None
            if hard is None:
                rd = os.path.join(ds_dir(ds), "phg", "R1")
                hard = assemble_partition(rd, R["inputs"]["N"], R["inputs"]["k"])
            sz = np.bincount(hard, minlength=R["inputs"]["k"])
            obs[ds] = {"empty_block_ids": [int(x) for x in np.where(sz == 0)[0]], "blocks_below_half_mean": {int(i): int(sz[i]) for i in np.where(sz < 0.5 * sz.mean())[0]},
                       "mtkahypar_min_block": (R.get("mtkahypar_baseline") or {}).get("blocks", {}).get("min"),
                       "mechanism_note": "Zoltan's balance constraint is one-sided (max part weight <= IMBALANCE_TOL x average); an empty or tiny part "
                                         "violates nothing PHG enforces, and PHG_OUTPUT_LEVEL=1 shows recursive bisection (p=2) of k=%d with that "
                                         "upper-only bound at every split.  The canonical partition.py marks a vector with unused blocks BUILT_WITH_WARNINGS, "
                                         "not OK, so the preregistered 'no empty block' gate is the canonical contract, not a new threshold." % R["inputs"]["k"],
                       "not_acted_on": "no re-run, no parameter change, no downstream evaluation of this partition (preregistered gate 2)"}
    rep["post_hoc_observation_not_a_gate"] = obs
    wj(os.path.join(OUT, "PHG_REPORT.json"), rep)
    md = ["# %s -- Zoltan-PHG substitution validation (SQuAD + MetaQA)" % ARM, "", "Generated %s -- STATUS: **STOP_FOR_REVIEW** -- decisions: %s" % (
        rep["utc"], ", ".join("%s = **%s**" % kv for kv in rep["decisions"].items())), "",
          "Primary question (preregistered): does canonical L1 with the PHG partition reach at least Mt-KaHyPar-level final SAFE recall on the identical "
          "frozen H4_SK input and identical dev query ids?  Partition equality is not required; KM1 / cuts are diagnostics.", "",
          rep["table_markdown"], "", "Hop-wise (diagnostic, no rule):", "", rep["hop_table_markdown"], "", "Coverage / SAFE additions / P50 (diagnostic):", "",
          rep["coverage_table_markdown"], "", "Structure gates:"]
    for ds, g in rep["structure_gates"].items():
        md.append("- %s: %s -- N %s M %s P %s weight sum %s; digest from the query functions %s == ORIGINAL %s; partition-run dumps identical: %s; Zoltan removal/warning lines: %s" % (
            ds, g["gate"], g["N_returned"], g["M_with_pins"], g["P_returned"], g["weight_sum"], (g["digest_from_queries"] or "-")[:16], (g["ORIGINAL_STRUCTURE_SHA256"] or "-")[:16],
            g["partition_run_dumps_identical"], g["zoltan_removed_or_warning_lines"] or "none"))
    md += ["", "Stage 3 (memory / timing / cross-check):"]
    for ds, s in rep["stage3_summaries"].items():
        if s.get("memory"):
            md.append("- %s: %d ranks; peak RSS per rank %s MB (max %.0f, sum %.0f -- sum is an upper bound); partition wall %.2f s, job wall %s s; Python KM1 vs Zoltan cutl: %s" % (
                ds, NP, [round(x / 1024.0, 1) for x in s["memory"]["peak_rss_kb_per_rank_time_v"]], s["memory"]["max_rank_peak_kb"] / 1024.0, s["memory"]["sum_of_rank_peaks_kb"] / 1024.0,
                s["timing"]["partition_wall_seconds"], s["timing"]["job_wall_seconds"], s["python_km1_vs_zoltan_cutl"]))
    for ds, o in obs.items():
        md += ["", "Post-hoc observation (%s, NOT a gate, not acted on): empty blocks %s; blocks below half the mean size %s (Mt-KaHyPar min block %s).  %s" % (
            ds, o["empty_block_ids"], o["blocks_below_half_mean"], o["mtkahypar_min_block"], o["mechanism_note"])]
    md += ["", "Gates and decision rule (preregistered): " + DECISION_RULE, "",
           "Integrity: frozen records changed since preregistration = %s; canonical H4_SK partitions unchanged = %s" % (
               integ["frozen_records_changed_since_prereg"] or "none", all(v["unchanged"] for v in integ["canonical_H4_SK_partition"].values())),
           "", "Caveats: %s.  %s" % (pre["rank_count_caveat"], pre["reproducibility"]), "", rep["not_done"], ""]
    with io.open(os.path.join(OUT, "PHG_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    log("STOP_FOR_REVIEW", rep["decisions"])
    print(rep["table_markdown"])
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "build":
        build()
    elif a[0] == "prereg":
        prereg()
    elif a[0] == "run":
        for ds in a[1:]:
            run(ds)
    elif a[0] == "report":
        report()
    else:
        print(__doc__)
