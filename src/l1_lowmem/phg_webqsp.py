"""PHG_SCALE_WEBQSP -- WebQSP (canonical WEBQSP_ROG_RESOLVED lane, NAME_ONLY encoding) under the experimental PHG contract
H4_SK_ZOLTAN_PHG_CONNECTIVITY_NP4 + PHG_NONEMPTY_REPAIR_V1, authorized by the user's message of 2026-09-14 after the MuSiQue bridge
(PHG_SCALE_BRIDGE_COMPLETE, frozen unchanged).  The first genuinely large lane: ~2.59M nodes, ~29.6M canonical SK pins, k 25,928.

The question (preregistered, no tuning, no equivalence claim):
    can the frozen PHG contract produce a valid partition of the exact canonical WebQSP H4_SK and carry it through unchanged canonical L1
    with sane retrieval behaviour at a memory footprint that makes the scale lane practical?

    python -u src/l1_lowmem/phg_webqsp.py ruling      -> results/L1_LOWMEM/PHG_WEBQSP_AUTHORIZATION_RULING.json (write-once: the authorization
                                                         verbatim; MuSiQue frozen as PHG_SCALE_BRIDGE_COMPLETE with pins; exclusions)
    python -u src/l1_lowmem/phg_webqsp.py preflight   -> one host / WSL / swap / paging / disk / foreign-process sample (rc 0 = CLEAN streak reached,
                                                         rc 2 = wait); appended to results/L1_LOWMEM/logs/phg_webqsp_preflight.jsonl
    python -u src/l1_lowmem/phg_webqsp.py arm         -> starts the wait budget (logs/phg_webqsp_wait_state.json)
    python -u src/l1_lowmem/phg_webqsp.py inputs      -> builds the exact H4_SK input chain for webqsp if absent (canonical H4_SK.npz -> H4_SK.hgr ->
                                                         H4_SK_STREAM_V1 shards -> official net-list + reconstruction + semantic gates ->
                                                         freight/stream_manifest.txt), verifies it, records PHG_WEBQSP_INPUTS.json; rc 3 = host contended
    python -u src/l1_lowmem/phg_webqsp.py prereg      -> results/L1_LOWMEM/PHG_WEBQSP_PREREG.json (write-once, after the inputs, before any PHG run)
    python -u src/l1_lowmem/phg_webqsp.py run         -> results/L1_LOWMEM/PHG_WEBQSP_RUNS.json (stages 1-4, every launch preflight-gated; rc 3 = contended)
    python -u src/l1_lowmem/phg_webqsp.py not_run     -> PHG_WEBQSP_RUNS.json = WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED (only after the wait budget)
    python -u src/l1_lowmem/phg_webqsp.py report      -> results/L1_LOWMEM/PHG_WEBQSP_REPORT.{json,md}  (STOP_FOR_REVIEW)

Nothing frozen is written or edited: phg.py / phg_repair.py / phg_ruling.py / phg_bridge.py are imported unchanged (sha-pinned by their
preregistrations); the canonical webqsp partition slot (parts/H4_SK.*) stays empty; the musique files are untouched; the foreign process is
never touched; NP is never changed; no minimum-part-size rule exists; the sealed test split is never read.
"""
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import hashlib

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, log, sha_file, rj, wj, pin, ds_dir, wsl  # noqa: E402
from src.l1_lowmem import freight as FR  # noqa: E402
from src.l1_lowmem import hgr as HG  # noqa: E402
from src.l1_lowmem import stream as ST  # noqa: E402
from src.l1_lowmem import phg as P  # noqa: E402
from src.l1_lowmem import phg_repair as PR  # noqa: E402
from src.l1_lowmem import phg_ruling as RL  # noqa: E402
from src.l1_lowmem import phg_bridge as PB  # noqa: E402

CLASS = "PHG_SCALE_WEBQSP"
DS = "webqsp"
TAG_W = PB.TAG_B                 # LOWMEM__PHG_C1_con: per-dataset files under data/l1_canonical/webqsp/parts/ and data/l1_lowmem/webqsp/; musique's are untouched
SUFFIX_W = PB.SUFFIX_B           # __phg_c1
SHARD_NODES = ST.SHARD_NODES_LOCAL   # 5000 = the shard size every PHG run used; rank ownership derives from shard boundaries, so it is recorded as an input element
MPIRUN_TIMEOUT = 14400           # harness parameter (musique needed 88 s; the frozen default is 7200 s); not a PHG parameter
FOREIGN_PID = 8628               # the unrelated host process (python -u scripts/m3b_run.py --stage screen --threads 8, started 2026-09-13 20:29 local) -- never touched
THIS = os.path.abspath(__file__)
RULING_S = os.path.join(OUT, RL.RULING_RECORD)
RULING_W = "PHG_WEBQSP_AUTHORIZATION_RULING.json"
INPUTS_W = "PHG_WEBQSP_INPUTS.json"
PREREG_W = "PHG_WEBQSP_PREREG.json"
RUNS_W = "PHG_WEBQSP_RUNS.json"
REPORT_W = "PHG_WEBQSP_REPORT"
LOGDIR = os.path.join(OUT, "logs")
PREFLIGHT_LOG = os.path.join(LOGDIR, "phg_webqsp_preflight.jsonl")
PREFLIGHT_STATE = os.path.join(LOGDIR, "phg_webqsp_preflight_state.json")
WAIT_STATE = os.path.join(LOGDIR, "phg_webqsp_wait_state.json")
LANE = "WEBQSP_ROG_RESOLVED / BENCHMARK_CONDITIONED / query_independent NOT CLAIMED / MID completeness NOT REQUIRED FOR V1"
ENCODING = "NAME_ONLY (results/L1_CANONICAL/KB_CORPUS_ENCODING_RULING.json: KEEP_NAME_ONLY; the legacy webqsp corpus was NAME + verbalised facts)"
CANON = os.path.join(REPO, "data", "l1_canonical", DS)
CANON_L1 = os.path.join(REPO, "results", "L1_CANONICAL")
BRIDGE_RECORDS = ["PHG_BRIDGE_PREREG.json", "PHG_BRIDGE_RUNS_musique.json", "PHG_BRIDGE_REPORT.json", "PHG_BRIDGE_REPORT.md",
                  "L1_REPLAY_musique__LOWMEM__PHG_C1_con.json", "L1_DOWNSTREAM_musique__phg_c1.json"]
LABELS = ["PHG_SCALE_WEBQSP_COMPLETE", "WEBQSP_INPUT_VERIFICATION_FAIL", "STRUCTURE_MISMATCH", "WEBQSP_NP4_RESOURCE_INFEASIBLE",
          "WEBQSP_NP4_RUN_FAIL_UNCLASSIFIED", "PHG_REPAIR_POSTCONDITION_FAIL", "PHG_PARTITION_INVALID", "WEBQSP_L1_EXECUTION_FAIL",
          "WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED", "WEBQSP_SUSPENDED_HOST_RESOURCE_CONTENDED"]
DECISION_RULE = ("WEBQSP_INPUT_VERIFICATION_FAIL (stage 1: an input file or pin differs from the preregistration, or the recomputed structure digest != ORIGINAL) | "
                 "STRUCTURE_MISMATCH (the gate-mode PHG run's query-function dumps do not reconstruct the exact canonical H4_SK -- node count, hyperedge count, pin "
                 "count, unit vertex weights, hyperedge weights, global node positions, global hyperedge ids, incidence multiset, structure digest -- or the "
                 "partition run's dumps differ from the gate run's) | WEBQSP_NP4_RESOURCE_INFEASIBLE (the NP=4 partition run did not complete AND memory evidence "
                 "exists -- OOM / bad_alloc / signal 9 / exit 137 / WSL dmesg OOM lines -- AND the pre-launch preflight was CLEAN; NP is never changed; a "
                 "different rank count needs a separate ruling) | WEBQSP_NP4_RUN_FAIL_UNCLASSIFIED (the partition run failed without memory evidence, or the "
                 "launch preflight was not clean; the raw evidence is recorded for review) | PHG_REPAIR_POSTCONDITION_FAIL | PHG_PARTITION_INVALID (after the "
                 "repair) | WEBQSP_L1_EXECUTION_FAIL (stage 4 did not complete or a structural sanity check S1-S4 failed) | "
                 "WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED (the wait budget elapsed before the host was clean; nothing launched) | "
                 "WEBQSP_SUSPENDED_HOST_RESOURCE_CONTENDED (the host stopped being clean between stages and the budget elapsed; completed stages recorded) | "
                 "PHG_SCALE_WEBQSP_COMPLETE (every stage completed and S1-S4 pass).  No label depends on the value of any retrieval number.")
SANITY = {"S1_downstream_executed": "the unchanged canonical L1 path ran to its record (l1_downstream.run: replay cache -> l1_eval numerics); status BASELINE_ABSENT "
                                    "is the expected outcome (no canonical Mt-KaHyPar replay exists for webqsp); PAIRED would also pass",
          "S2_population": "the evaluated population is the frozen webqsp train_holdout carve (EVAL_SPLITS train_holdout, stride 2 by sorted query_id, carve digest "
                           "verified by the adapter) passed through replay_cache.build's unchanged sample rule; nq and row_query_ids_sha256 must equal the values "
                           "precomputed in the preregistration -- the rule is partition-independent, so any difference is a pipeline fault; the test split is never read",
          "S3_canonical_eligibility": "l1_eval's existing ELIGIBLE flag (max block / mean <= 1.05 and every block used) is True -- an existing canonical rule, not a new threshold",
          "S4_definitional_consistency": "n_docs == N, npart == k, 0 <= BASE_ALL <= BASE_ANY <= 1, 0 <= SAFE_ALL <= SAFE_ANY <= 1, scope nodes > 0, every number finite, "
                                         "SAFE selector executed (CORR present) -- ALL is a subset of ANY per query by definition",
          "S5_no_numeric_retrieval_threshold": "no acceptance threshold on BASE / SAFE / coverage / KM1 exists or will be added after seeing the numbers"}
DIAGNOSTICS_NO_RULE = ["Zoltan connectivity (cutl) and the independently recomputed weighted KM1 (absolute; no canonical Mt-KaHyPar value exists for webqsp)",
                       "imbalance (Zoltan's and max block / mean)", "block min / p50 / p95 / max, empty block count before the repair", "STRUCT cut", "KNN cut",
                       "P50 scope", "candidate coverage (ANY)", "SAFE additions (+/-)", "hop-wise results only where the canonical replay carries hop labels",
                       "peak RSS per rank, aggregate rank RSS (sum of peaks = upper bound), host Python peak per stage", "partition wall, job wall, per-stage wall",
                       "repeat-run identity", "repair moves and their exact delta-KM1",
                       "context only, never a baseline: the legacy webqsp L1 cell (NAME + verbalised facts corpus, 781,485 docs, k 7,814, all 1,578 evaluable "
                       "queries) and the legacy webqsp Mt-KaHyPar SKN resource point (10.9M pins, 140.6 GB) -- a different encoding, corpus, k, family set and "
                       "population; shown as historical context, never compared numerically",
                       "context only: the frozen Mt-KaHyPar recipe's memory expectation for canonical webqsp (EXTERNAL_LANE_MEMORY_EXPECTATION.json: 382 GB "
                       "linear-in-pins, 1,266 GB pins x k -> NOT_FEASIBLE_AT_250GB) versus PHG's measured per-rank peaks -- resources, not quality"]

# ----------------------------------------------------------------------------- resource preflight (execution hygiene, not a PHG parameter)
PREFLIGHT = {
    "host_available_gb_min": 6.0, "pages_input_per_sec_max": 300.0, "pages_samples": 5, "wsl_available_mb_min": 5000, "disk_free_gb_min": 8.0,
    "swap_growth_gb_max_between_samples": 0.1, "consecutive_clean_samples": 3, "sample_spacing_seconds_wait_loop": 60, "sample_spacing_seconds_in_run": 20,
    "wait_budget_hours": 24.0,
    "what_is_recorded_immediately_before_every_launch": ["host total / available memory", "WSL total / available memory", "swap usage (host and WSL)",
                                                          "foreign-process RSS (pid %d and every python process over 1 GB RSS)" % FOREIGN_PID,
                                                          "number of MPI ranks", "disk free space", "Pages Input/sec (typeperf, %d x 1 s)" % 5],
    "derivation": {
        "host_available_gb_min": "6.0 >= 1.7 x the predicted summed PHG rank footprint (~ 3.5 GB, see wsl_available_mb_min) and ~ the predicted peak of the "
                                 "largest host-Python stage: the semantic gate (freight.semantic_gate, pinned) parses the official net-list and the shards "
                                 "independently -- per-node-line numpy objects plus three O(P) int64 arrays, concatenated, sorted, then CSR -- at P ~ 29.6M "
                                 "that is ~ 4-6 GB with the canonical CSR and the digest temporaries; the repair / KM1 recomputation ~ 2 GB; the L1 "
                                 "replay-cache builder streams the 8.0 GB fp16 shards memory-mapped (page cache, not RSS) plus ~ 1.5 GB of arrays.  A "
                                 "MemoryError in the host-Python input chain is a harness retry after a new clean preflight, never evidence about PHG",
        "wsl_available_mb_min": "5000 >= 1.4 x the predicted summed rank peak: musique R1 peaked at 177.9 MB on the largest rank at 6,014,938 pins; webqsp has "
                                "~ 29.6M pins (x 4.92) -> ~ 0.9 GB per rank, <= 3.5 GB summed over 4 ranks inside the 7.8 GB WSL VM, plus the page cache of "
                                "the ~ 0.3 GB shard files",
        "pages_input_per_sec_max": "300: the contended host measured 26,000-64,000 hard-fault pages/s (2026-09-14 01:39-01:47 local, ~ 0.5-0.7 GB available, "
                                   "swap 6.9-7.8 GB); an idle host reads under ~ 100; 300 sits two orders of magnitude under the contended readings and above "
                                   "idle noise -- it operationalises 'actively paging'",
        "disk_free_gb_min": "8.0 = the canonical replay-cache builder's 5 GB free-space floor + ~ 1.8 GB of new webqsp artefacts (hgr, shards, net-list, PHG dumps, "
                            "replay cache) + margin",
        "swap_growth": "a host whose available memory reads high only while it is paging out is not clean: swap use may not grow by more than 0.1 GB between "
                       "consecutive samples",
        "consecutive_samples": "3 consecutive clean samples (60 s apart in the wait loop; 20 s apart immediately before each launch inside the run) so a transient "
                              "dip of the foreign process's working set does not trigger a launch",
        "wait_budget": "24 h from arming; if the host is never clean for that long, WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED is recorded and nothing is launched; "
                       "the foreign process is never signalled",
        "status": "set before any WebQSP artefact exists; gates execution hygiene only -- no PHG parameter, rank count, objective or representation depends on it"},
    "user_requirement_verbatim": "Do NOT start the PHG partition while the host is actively paging or while only ~1.45 GB host memory is available. The unrelated "
                                 "foreign process must remain untouched. Wait until the unrelated process naturally releases enough memory for a clean attempt, "
                                 "then record immediately before launch: host total / available memory; WSL total / available memory; swap usage; foreign-process "
                                 "RSS; number of MPI ranks; disk free space. This is not a PHG parameter change; it is an execution hygiene requirement so a "
                                 "host-starvation failure is not misclassified as an algorithmic resource failure. If adequate clean memory never becomes "
                                 "available, record: WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED rather than killing unrelated work."}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def typeperf_pages_input(n):
    cmd = ["typeperf", r"\Memory\Pages Input/sec", r"\Memory\Available MBytes", "-sc", str(n), "-si", "1"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90 + 3 * n)
    except Exception as e:
        return {"pages_input_per_sec": [], "available_mbytes": [], "error": "%s: %s" % (type(e).__name__, e)}
    pages, avail = [], []
    for ln in r.stdout.splitlines():
        parts = [x.strip().strip('"') for x in ln.strip().split('","')]
        if len(parts) == 3 and re.match(r"\d{2}/\d{2}/\d{4}", parts[0]):
            try:
                pages.append(float(parts[1])); avail.append(float(parts[2]))
            except ValueError:
                pass
    return {"pages_input_per_sec": pages, "available_mbytes": avail, "rc": r.returncode, "error": None if pages else (r.stdout[-300:] + r.stderr[-300:])}


def wsl_mem():
    try:
        r = wsl("free -m; cat /proc/loadavg", timeout=120)
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}
    out = {}
    for ln in r.stdout.splitlines():
        f = ln.split()
        if ln.startswith("Mem:") and len(f) >= 7:
            out.update({"wsl_total_mb": int(f[1]), "wsl_used_mb": int(f[2]), "wsl_free_mb": int(f[3]), "wsl_buff_cache_mb": int(f[5]), "wsl_available_mb": int(f[6])})
        elif ln.startswith("Swap:") and len(f) >= 4:
            out.update({"wsl_swap_total_mb": int(f[1]), "wsl_swap_used_mb": int(f[2])})
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if lines:
        la = lines[-1].split()
        if len(la) >= 3:
            try:
                out["wsl_loadavg_1_5_15"] = [float(la[0]), float(la[1]), float(la[2])]
            except ValueError:
                pass
    if "wsl_available_mb" not in out:
        out["error"] = "free -m not parsed: %s %s" % (r.stdout[-200:], r.stderr[-200:])
    return out


def foreign_processes():
    import psutil
    me = os.getpid()
    out = []
    for pr in psutil.process_iter(["pid", "name", "create_time", "memory_info"]):
        try:
            mi = pr.info["memory_info"]
            if pr.info["pid"] == me or mi is None:
                continue
            if pr.info["pid"] == FOREIGN_PID or ((pr.info["name"] or "").lower().startswith("python") and mi.rss > (1 << 30)):
                out.append({"pid": pr.info["pid"], "name": pr.info["name"], "rss_mb": round(mi.rss / 2.0 ** 20), "private_mb": round(getattr(mi, "private", 0) / 2.0 ** 20),
                            "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(pr.info["create_time"]))})
        except Exception:
            pass
    return out


def preflight_sample(prev=None, purpose="wait"):
    import psutil
    vm = psutil.virtual_memory(); sw = psutil.swap_memory()
    tp = typeperf_pages_input(PREFLIGHT["pages_samples"])
    w = wsl_mem()
    du = shutil.disk_usage(REPO)
    fps = foreign_processes()
    pages = tp["pages_input_per_sec"]
    pm = round(float(np.mean(pages)), 1) if pages else None
    fpid = [p for p in fps if p["pid"] == FOREIGN_PID]
    s = {"utc": utc(), "purpose": purpose, "host_total_gb": round(vm.total / 2.0 ** 30, 2), "host_available_gb": round(vm.available / 2.0 ** 30, 2),
         "host_percent_used": vm.percent, "swap_used_gb": round(sw.used / 2.0 ** 30, 2), "swap_total_gb": round(sw.total / 2.0 ** 30, 2),
         "pages_input_per_sec_samples": pages, "pages_input_per_sec_mean": pm, "typeperf_available_mbytes": tp["available_mbytes"], "typeperf_error": tp.get("error"),
         "wsl": w, "disk_free_gb": round(du.free / 2.0 ** 30, 2), "foreign_processes": fps, "foreign_pid": FOREIGN_PID, "foreign_pid_present": bool(fpid),
         "foreign_pid_rss_mb": fpid[0]["rss_mb"] if fpid else None, "mpi_ranks": P.NP}
    growth = None if prev is None else round(s["swap_used_gb"] - prev["swap_used_gb"], 3)
    c = {"host_available_ge_min": s["host_available_gb"] >= PREFLIGHT["host_available_gb_min"],
         "not_paging": pm is not None and pm <= PREFLIGHT["pages_input_per_sec_max"],
         "wsl_available_ge_min": w.get("wsl_available_mb") is not None and w["wsl_available_mb"] >= PREFLIGHT["wsl_available_mb_min"],
         "disk_free_ge_min": s["disk_free_gb"] >= PREFLIGHT["disk_free_gb_min"],
         "swap_not_growing": growth is None or growth <= PREFLIGHT["swap_growth_gb_max_between_samples"]}
    s["swap_growth_gb_since_previous"] = growth
    s["checks"] = c
    s["CLEAN"] = all(c.values())
    return s


def one_line(s, streak=None):
    w = s.get("wsl") or {}
    return "%s %s  host avail %.2f/%.2f GB  pages_in/s %s  swap %.2f GB (%s)  wsl avail %s/%s MB  disk %.1f GB  foreign pid %s rss %s MB  ranks %d%s" % (
        s["utc"], "CLEAN" if s["CLEAN"] else "CONTENDED", s["host_available_gb"], s["host_total_gb"], s["pages_input_per_sec_mean"], s["swap_used_gb"],
        "%+.2f" % s["swap_growth_gb_since_previous"] if s["swap_growth_gb_since_previous"] is not None else "first", w.get("wsl_available_mb"), w.get("wsl_total_mb"),
        s["disk_free_gb"], FOREIGN_PID if s["foreign_pid_present"] else "gone", s["foreign_pid_rss_mb"], s["mpi_ranks"],
        "" if streak is None else "  streak %d/%d" % (streak, PREFLIGHT["consecutive_clean_samples"]))


def append_log(s):
    os.makedirs(LOGDIR, exist_ok=True)
    with io.open(PREFLIGHT_LOG, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(s) + "\n")


def deadline():
    st = rj(WAIT_STATE)
    if st and st.get("deadline_epoch"):
        return float(st["deadline_epoch"])
    return time.time() + PREFLIGHT["wait_budget_hours"] * 3600.0


def arm():
    st = rj(WAIT_STATE)
    if st:
        log("wait budget already armed at %s (deadline %s)" % (st["armed_utc"], st["deadline_utc"]))
        return st
    t = time.time()
    st = {"armed_utc": utc(), "armed_epoch": t, "wait_budget_hours": PREFLIGHT["wait_budget_hours"], "deadline_epoch": t + PREFLIGHT["wait_budget_hours"] * 3600.0,
          "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t + PREFLIGHT["wait_budget_hours"] * 3600.0)), "criterion": PREFLIGHT}
    wj(WAIT_STATE, st)
    log("wait budget armed: %.0f h -> deadline %s" % (st["wait_budget_hours"], st["deadline_utc"]))
    return st


def preflight_cmd():
    """one wait-loop sample; rc 0 when the clean streak is reached, rc 2 otherwise."""
    st = rj(PREFLIGHT_STATE) or {"streak": 0, "last": None, "n": 0}
    s = preflight_sample(st["last"], purpose="wait_loop")
    st["streak"] = st["streak"] + 1 if s["CLEAN"] else 0
    st["last"] = s
    st["n"] += 1
    append_log(dict(s, streak=st["streak"]))
    wj(PREFLIGHT_STATE, st)
    ready = st["streak"] >= PREFLIGHT["consecutive_clean_samples"]
    print("PREFLIGHT #%d %s%s" % (st["n"], one_line(s, st["streak"]), "  -> READY" if ready else ""), flush=True)
    return 0 if ready else 2


def formal_preflight(purpose, wait_until=None):
    """3 consecutive clean samples 20 s apart, recorded.  wait_until None: refuse (return None) on the first non-clean sample;
    otherwise keep sampling every 60 s until the streak is reached or the deadline passes (return None)."""
    n, spacing = PREFLIGHT["consecutive_clean_samples"], PREFLIGHT["sample_spacing_seconds_in_run"]
    samples, prev = [], None
    while True:
        s = preflight_sample(prev, purpose=purpose)
        append_log(dict(s, formal=True))
        log("  preflight [%s] %s" % (purpose, one_line(s, len([x for x in samples if x["CLEAN"]]) + (1 if s["CLEAN"] else 0))))
        samples.append(s); prev = s
        clean_tail = 0
        for x in reversed(samples):
            if not x["CLEAN"]:
                break
            clean_tail += 1
        if clean_tail >= n:
            return samples
        if not s["CLEAN"]:
            if wait_until is None:
                return None
            if time.time() > wait_until:
                return None
            time.sleep(PREFLIGHT["sample_spacing_seconds_wait_loop"])
        else:
            time.sleep(spacing)


def preflight_summary(samples):
    ok = [s for s in samples if s.get("CLEAN")]
    return {"n": len(samples), "clean": len(ok), "all_clean": bool(samples) and len(ok) == len(samples), "first_utc": samples[0]["utc"] if samples else None,
            "last_utc": samples[-1]["utc"] if samples else None, "last": samples[-1] if samples else None,
            "host_available_gb_min_max": [min(s["host_available_gb"] for s in samples), max(s["host_available_gb"] for s in samples)] if samples else None,
            "pages_input_per_sec_mean_min_max": [min(s["pages_input_per_sec_mean"] or 0 for s in samples), max(s["pages_input_per_sec_mean"] or 0 for s in samples)] if samples else None}


# ----------------------------------------------------------------------------- the authorization (user, 2026-09-14), frozen verbatim
AUTHORIZATION_TEXT_VERBATIM = """The MuSiQue bridge is a clear pass for feasibility. I’d freeze it exactly as `PHG_SCALE_BRIDGE_COMPLETE` and authorize WebQSP next, but not while the host has only ~1.45 GB available.
The important result from MuSiQue is that the whole real chain worked:

```text
exact H4_SK
→ PHG NP=4
→ deterministic non-empty repair
→ P50
→ canonical L1
```

with no tuning and without touching the canonical substrate. That is enough evidence to move from a bridge dataset to the first genuinely large lane.
The 9 empty blocks also don’t concern me now. `PHG_NONEMPTY_REPAIR_V1` behaved exactly as intended: minimum number of moves, every move ΔKM1=0, and no change to STRUCT/KNN cuts. It is now a legitimate part of the experimental PHG contract.
For WebQSP, though, I would add a resource preflight. MuSiQue was ~6.0M pins and reached 178 MB on the largest rank. WebQSP is ~29.6M pins, so it is a much larger job. I would not intentionally launch it into an already paging host with only ~1.45 GB available and then interpret the resulting failure as evidence against PHG.
Send Claude this:
Authorize the WebQSP PHG scale run as the next experiment.
Freeze the completed MuSiQue record unchanged:
`PHG_SCALE_BRIDGE_COMPLETE`
Do not rerun or tune MuSiQue.
The WebQSP run must use the exact existing experimental PHG contract:

* dataset lane: canonical `WEBQSP_ROG_RESOLVED`
* H4 construction: `H4_SPLIT_PRESERVE`
* families: `STRUCT + KNN` only
* tag: `H4_SK`
* PHG connectivity/KM1 objective
* NP = 4
* identical frozen PHG parameter set
* `PHG_EDGE_SIZE_THRESHOLD=1.0`
* deterministic mode
* same imbalance contract
* `PHG_NONEMPTY_REPAIR_V1`
* same P50 and canonical L1 downstream machinery
* no parameter or rank-count tuning

Do not touch the full 302M Freebase scale lane. This authorization is WebQSP V1 / RoG-resolved only.
Resource preflight
Do NOT start the PHG partition while the host is actively paging or while only ~1.45 GB host memory is available.
The unrelated foreign process must remain untouched.
Wait until the unrelated process naturally releases enough memory for a clean attempt, then record immediately before launch:

* host total / available memory
* WSL total / available memory
* swap usage
* foreign-process RSS
* number of MPI ranks
* disk free space

This is not a PHG parameter change; it is an execution hygiene requirement so a host-starvation failure is not misclassified as an algorithmic resource failure.
If adequate clean memory never becomes available, record:
`WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED`
rather than killing unrelated work.
Preregistration
Before executing WebQSP, write a new WebQSP-specific preregistration that pins:

* frozen PHG small-dataset ruling
* MuSiQue bridge result
* WebQSP canonical dataset / node-order / H4_SK / stream hashes
* PHG binary and driver hashes
* repair implementation hash
* NP=4
* all effective PHG parameters
* evaluation population
* decision rules

No results from WebQSP may be inspected before this record is written.
Stage 1 — exact H4_SK structure gate
Require exact reconstruction of the canonical WebQSP H4_SK input:

* node count
* hyperedge count
* pin count
* vertex weights
* hyperedge weights
* global node positions
* global hyperedge IDs
* incidence multiset
* canonical structure digest

Any mismatch:
`STRUCTURE_MISMATCH`
and STOP.
Do not silently remove large hyperedges or change the family set.
Stage 2 — PHG partition
Run the frozen NP=4 contract.
Record:

* partition wall time
* total job wall time
* peak RSS for every rank
* aggregate rank RSS diagnostic
* Zoltan connectivity
* independently recomputed weighted KM1
* imbalance
* block min/p50/p95/max
* empty block count
* STRUCT cut
* KNN cut

Run the frozen deterministic repeat check if it remains feasible under the existing protocol.
Do not change NP if NP=4 fails.
If NP=4 cannot complete because of genuine rank/process memory requirements on an otherwise clean host, record:
`WEBQSP_NP4_RESOURCE_INFEASIBLE`
and STOP_FOR_REVIEW.
A different rank count requires a separate ruling because MPI rank count is part of the PHG experimental contract.
Stage 3 — non-empty repair
If empty parts exist, apply the already-frozen:
`PHG_NONEMPTY_REPAIR_V1`
without modification.
Record every move and exact ΔKM1.
Require after repair:

* all nodes assigned once
* same k
* zero empty blocks
* maximum-size bound satisfied
* exact H4_SK unchanged

Do not create any new minimum-part-size condition.
Stage 4 — canonical L1
If the repaired partition is valid:
PHG partition
→ canonical P50
→ canonical replay cache
→ unchanged canonical L1 evaluation
Use only the already-authorized non-test WebQSP evaluation population (`train_holdout`; test remains barred).
Report:

* BASE_ALL
* SAFE_ALL
* BASE_ANY
* SAFE_ANY
* candidate coverage
* SAFE additions
* scope
* hop-wise results if already defined for this population
* all existing L1 sanity checks

If a valid frozen Mt-KaHyPar WebQSP comparison exists on exactly the same representation/evaluation contract, report it separately.
Do NOT compare current NAME_ONLY canonical WebQSP numerically against a legacy NAME+FACTS result as if it were a reproduction target.
If there is no truly comparable canonical Mt-KaHyPar reference, classify this as a scale/absolute-L1 result rather than claiming equivalence.
Interpretation
Partition equality with Mt-KaHyPar is NOT required.
KM1 and edge-cut metrics are diagnostics.
The primary scientific question is:
Can the frozen PHG contract produce a valid partition of the exact canonical WebQSP H4_SK and carry it through unchanged canonical L1 with sane retrieval behavior at a memory footprint that makes the scale lane practical?
Do not tune PHG based on WebQSP L1 outcomes.
STOP_FOR_REVIEW after WebQSP.
Do not proceed automatically to HotpotQA or 2Wiki.
One thing I especially want preserved is the WebQSP representation distinction. This run is canonical `NAME_ONLY`. Any old WebQSP Mt-KaHyPar/L1 number obtained from the legacy `NAME + verbalized facts` corpus is not a fair paired baseline. It can be shown as historical context only.
So the progression now is:

```text
SQuAD          PHG quality evidence       ✓
MetaQA         PHG quality evidence       ✓
MuSiQue        PHG feasibility bridge     ✓
WebQSP         first large-scale test      NEXT
HotpotQA       unauthorized
2Wiki          unauthorized
```

If WebQSP completes on NP=4 with the exact H4_SK structure and canonical L1 runs successfully, then we have much stronger evidence that the original single-machine Mt-KaHyPar memory wall is no longer blocking canonical L1."""

QUESTION = ("Can the frozen PHG contract produce a valid partition of the exact canonical WebQSP H4_SK and carry it through unchanged canonical L1 with sane "
            "retrieval behaviour at a memory footprint that makes the scale lane practical?")


def musique_files():
    md = ds_dir("musique")
    cm = os.path.join(REPO, "data", "l1_canonical", "musique")
    files = {"parts_npy": os.path.join(cm, "parts", "%s.npy" % TAG_W), "parts_json": os.path.join(cm, "parts", "%s.json" % TAG_W),
             "replay_cache_npz": os.path.join(md, "replay_cache__%s.npz" % TAG_W), "replay_cache_json": os.path.join(md, "replay_cache__%s.json" % TAG_W),
             "repaired_npy": os.path.join(md, "phg_repair1", "repaired.npy")}
    for fn in ("STAGE1_INPUT_VERIFIED.json", "STAGE2_PHG_INPUT_CALLBACK_MANIFEST.json", "STAGE3_PHG_PARTITION.json", "STAGE3b_REPAIR.json", "STAGE4_5_P50_REPLAY_ABSOLUTE.json"):
        files["phg_" + fn] = os.path.join(md, "phg", fn)
    return {k: pin(p) for k, p in files.items()}


def ruling():
    rp = os.path.join(OUT, RULING_W)
    if os.path.exists(rp):
        raise RuntimeError("the WebQSP authorization record already exists -- written once; supersede, never edit")
    small = rj(RULING_S)
    if small is None or small["PHG_SMALL_DATASET_VALIDATION"] != "PASS":
        raise RuntimeError("PHG_SMALL_DATASET_VALIDATION ruling missing or not PASS")
    bridge = rj(os.path.join(OUT, "PHG_BRIDGE_RUNS_musique.json"))
    brep = rj(os.path.join(OUT, "PHG_BRIDGE_REPORT.json"))
    if bridge is None or bridge["DECISION"] != "PHG_SCALE_BRIDGE_COMPLETE" or brep is None or brep["DECISION"] != "PHG_SCALE_BRIDGE_COMPLETE" or brep["STATUS"] != "STOP_FOR_REVIEW":
        raise RuntimeError("the musique bridge records do not read PHG_SCALE_BRIDGE_COMPLETE / STOP_FOR_REVIEW")
    if bridge["sanity_checks"]["ALL_PASS"] is not True:
        raise RuntimeError("the musique bridge sanity checks are not ALL_PASS")
    B = bridge["downstream"]["B_phg"]; s3 = bridge["stages"]["3_partition"]; s3b = bridge["stages"]["3b_repair"]
    kit = ((rj(os.path.join(CANON_L1, "L1_CANONICAL_MANIFEST.json")) or {}).get("EXTERNAL_LANE") or {}).get("kits", {}).get(DS) or {}
    ks = kit.get("summary") or {}
    host = preflight_sample(None, purpose="at_ruling")
    append_log(host)
    rec = {
        "RECORD": "PHG_WEBQSP_AUTHORIZATION_RULING", "utc": utc(), "classification": CLASS, "dataset": DS,
        "ruled_by": "user, chat message of 2026-09-14 (transcript timestamp 2026-09-13T19:59:21Z) after the STOP_FOR_REVIEW of PHG_BRIDGE_REPORT.json",
        "authorization_text_verbatim": AUTHORIZATION_TEXT_VERBATIM,
        "authorization_text_sha256": hashlib.sha256(AUTHORIZATION_TEXT_VERBATIM.encode("utf-8")).hexdigest(),
        "musique_frozen_unchanged": {
            "DECISION": "PHG_SCALE_BRIDGE_COMPLETE", "no_rerun_no_tuning": True,
            "records": {fn: pin(os.path.join(OUT, fn)) for fn in BRIDGE_RECORDS},
            "data_files": musique_files(),
            "numbers_as_frozen": {"N": bridge["inputs"]["N"], "M": bridge["inputs"]["M"], "P": bridge["inputs"]["P"], "k": bridge["inputs"]["k"],
                                  "km1_weighted": bridge["metrics"]["km1_weighted"], "peak_rss_mb_max_rank": round(s3["memory"]["max_rank_peak_kb"] / 1024.0, 1),
                                  "partition_wall_seconds": s3["timing"]["partition_wall_seconds"], "repair_moves": len(s3b["moves"]),
                                  "repair_delta_km1_total": s3b["after"]["delta_km1_total"], "BASE_ALL_P50": B["BASE_ALL_P50"], "SAFE_ALL_P50": B["SAFE_ALL_P50"],
                                  "BASE_ANY_P50": B["BASE_ANY_P50"], "SAFE_ANY_P50": B["SAFE_ANY_P50"], "downstream_status": bridge["downstream"]["status"]}},
        "small_dataset_ruling": pin(RULING_S),
        "authorized": {
            "dataset": DS, "classification": CLASS, "lane": LANE, "encoding": ENCODING, "scope": "WebQSP V1 / RoG-resolved only",
            "contract": RL.EXPERIMENTAL_PHG_CONTRACT,
            "contract_elements_verbatim": ["dataset lane: canonical WEBQSP_ROG_RESOLVED", "H4 construction: H4_SPLIT_PRESERVE", "families: STRUCT + KNN only", "tag: H4_SK",
                                           "PHG connectivity/KM1 objective", "NP = 4", "identical frozen PHG parameter set", "PHG_EDGE_SIZE_THRESHOLD=1.0",
                                           "deterministic mode", "same imbalance contract", "PHG_NONEMPTY_REPAIR_V1", "same P50 and canonical L1 downstream machinery",
                                           "no parameter or rank-count tuning"],
            "evaluation_population": "the already-authorized non-test webqsp population: EVAL_SPLITS train_holdout (stride-2 carve of train by sorted query_id, "
                                     "carve digest recorded in CANONICAL_FREEZE.json); the test split remains barred",
            "expected_scale": {"N": ks.get("N"), "k": ks.get("k"), "STRUCT_keys": ks.get("STRUCT"), "KNN_keys": ks.get("KNN"), "expected_pins_upper_bound": ks.get("expected_pins_upper_bound"),
                               "keys_npz_sha256": ks.get("keys_npz_sha256"), "expected_stream_shards_at_%d_nodes" % SHARD_NODES: int(math.ceil((ks.get("N") or 0) / float(SHARD_NODES)))}},
        "excluded": ["the full 302M Freebase scale lane (CRAG_FREEBASE_CANONICAL) -- not touched", "hotpotqa", "2wiki", "any NP change (a different rank count needs a separate ruling)",
                     "any PHG parameter change", "any new minimum-part-size condition", "any MuSiQue rerun or tuning", "any tuning on WebQSP L1 outcomes",
                     "any numeric comparison of the canonical NAME_ONLY result against the legacy NAME + verbalised-facts webqsp cells",
                     "any equivalence claim without a truly comparable canonical Mt-KaHyPar reference"],
        "resource_preflight_requirement": {"classification": "execution hygiene requirement, not a PHG parameter change", "criterion": PREFLIGHT,
                                           "if_never_clean": "WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED", "foreign_process": "pid %d remains untouched" % FOREIGN_PID,
                                           "host_at_ruling": host,
                                           "host_at_ruling_verdict": "CONTENDED" if not host["CLEAN"] else "CLEAN"},
        "stages_required": ["preregistration written before any PHG run (no WebQSP result inspected before it)",
                            "stage 1 exact H4_SK structure gate (node / hyperedge / pin counts, unit vertex weights, hyperedge weights, global node positions, global "
                            "hyperedge ids, incidence multiset, canonical structure digest) -> STRUCTURE_MISMATCH and STOP on any mismatch; no hyperedge removed, family set unchanged",
                            "stage 2 PHG partition under the frozen NP=4 contract with the full measurement list; deterministic repeat check if feasible; "
                            "WEBQSP_NP4_RESOURCE_INFEASIBLE + STOP_FOR_REVIEW on a genuine memory failure on an otherwise clean host; NP never changed",
                            "stage 3 PHG_NONEMPTY_REPAIR_V1 unchanged if empty parts exist; every move and exact delta-KM1 recorded; postconditions: all nodes assigned "
                            "once, same k, zero empty blocks, maximum-size bound satisfied, exact H4_SK unchanged; no new minimum-part-size condition",
                            "stage 4 PHG partition -> canonical P50 -> canonical replay cache -> unchanged canonical L1 evaluation on train_holdout only; report BASE_ALL, "
                            "SAFE_ALL, BASE_ANY, SAFE_ANY, candidate coverage, SAFE additions, scope, hop-wise if defined, all existing L1 sanity checks"],
        "reference_policy": {"canonical_mtkahypar_webqsp_reference": "ABSENT (results/L1_CANONICAL/L1_REPLAY_webqsp.json and data/l1_canonical/webqsp/parts/H4_SK.npy do not exist; "
                                                                    "EXTERNAL_LANE_MEMORY_EXPECTATION.json: NOT_FEASIBLE_AT_250GB)",
                             "classification_if_absent": "scale / absolute-L1 result; no equivalence claim",
                             "legacy_name_plus_facts_cells": "historical context only; never a paired baseline; never compared numerically"},
        "interpretation": {"partition_equality_with_mtkahypar_required": False, "km1_and_edge_cut": "diagnostics", "primary_question": QUESTION,
                           "no_tuning_on_webqsp_l1_outcomes": True, "after": "STOP_FOR_REVIEW; HotpotQA / 2Wiki not proceeded to automatically"},
        "progression": {"squad": "PHG quality evidence -- done", "metaqa": "PHG quality evidence -- done", "musique": "PHG feasibility bridge -- done",
                        "webqsp": "first large-scale test -- NEXT (this authorization)", "hotpotqa": "unauthorized", "2wiki": "unauthorized"},
        "evidence_pins": {fn: pin(os.path.join(CANON_L1, fn)) for fn in ("KB_CORPUS_ENCODING_RULING.json", "RETURN_TO_L1_BASELINE.json", "L1_CANONICAL_MANIFEST.json",
                                                                           "EXTERNAL_LANE_MEMORY_EXPECTATION.json")},
        "legacy_context_pin": pin(os.path.join(REPO, "results", "L1", "L1_LOCKED_MANIFEST.json")),
        "not_done_by_this_record": "nothing executed; no WebQSP artefact exists yet; the inputs are built only after the host is clean; the preregistration is written "
                                   "after the inputs and before any PHG run"}
    wj(rp, rec)
    log("authorization frozen ->", os.path.relpath(rp, REPO), "| host at ruling:", rec["resource_preflight_requirement"]["host_at_ruling_verdict"])
    return rec


# ----------------------------------------------------------------------------- inputs (the exact H4_SK chain for webqsp)
def inputs():
    rul = rj(os.path.join(OUT, RULING_W))
    if rul is None:
        raise RuntimeError("write the authorization ruling first")
    fp = os.path.join(OUT, INPUTS_W)
    prev = rj(fp)
    if prev is not None:
        changed = [k for k, pn in prev["pins"].items() if pn and (not os.path.exists(os.path.join(REPO, pn["path"])) or sha_file(os.path.join(REPO, pn["path"])) != pn["sha256"])]
        if changed:
            raise RuntimeError("inputs record exists but these pinned files changed: %s -- a human decision is required" % changed)
        log("inputs record exists and every pin verifies -> nothing to do")
        return prev
    pf = formal_preflight("inputs_start", wait_until=None)
    if pf is None:
        log("inputs: host CONTENDED -- refusing to start input preparation (nothing written)")
        sys.exit(3)
    from src.l1_canonical.adapter import CanonicalDataset
    T, peaks = {}, {}
    hm_p = os.path.join(CANON, "hypergraph", "H4_SK.json")
    npz_p = os.path.join(CANON, "hypergraph", "H4_SK.npz")
    steps = {}
    # A. canonical H4_SK (frozen rule H4_SPLIT_PRESERVE over STRUCT + KNN, k = max(1, N // 100); the canonical builder, unchanged)
    t = time.time()
    if rj(hm_p) is None or not os.path.exists(npz_p):
        from src.l1_canonical import hypergraph as HY
        log("building the canonical H4_SK for %s (src/l1_canonical/hypergraph.build_canonical, unchanged) ..." % DS)
        HY.build_canonical(DS)
        steps["A_hypergraph"] = "built"
    else:
        steps["A_hypergraph"] = "existed"
    T["A_hypergraph"] = round(time.time() - t, 1); peaks["A_hypergraph"] = PB.py_peak_mb()
    hm = rj(hm_p)
    # B. hMETIS hgr + ORIGINAL_STRUCTURE_SHA256
    t = time.time()
    hrec = rj(HG.hgr_path(DS) + ".json")
    if hrec is None or not os.path.exists(HG.hgr_path(DS)) or hrec["hgr"]["sha256"] != sha_file(HG.hgr_path(DS)) or hrec["source"]["npz"]["sha256"] != hm["file_sha256"]:
        hrec = HG.write_hgr(DS)
        steps["B_hgr"] = "written"
    else:
        steps["B_hgr"] = "existed"
    T["B_hgr"] = round(time.time() - t, 1); peaks["B_hgr"] = PB.py_peak_mb()
    # C. H4_SK_STREAM_V1 shards (idempotent; completed shards are never rewritten)
    t = time.time()
    man = ST.convert(DS, SHARD_NODES)
    steps["C_stream"] = man["this_run"]
    T["C_stream"] = round(time.time() - t, 1); peaks["C_stream"] = PB.py_peak_mb()
    if man["shard_nodes"] != SHARD_NODES or man["hgr_sha256"] != hrec["hgr"]["sha256"]:
        raise RuntimeError("stream manifest does not match the hgr / shard size")
    # D. official monolithic net-list + reconstruction gate (digest, bytes) + semantic gate
    t = time.time()
    off = rj(os.path.join(ds_dir(DS), "H4_SK.official.netl.json"))
    sg = rj(os.path.join(ds_dir(DS), "H4_SK.semantic_gate.json"))
    netl = os.path.join(ds_dir(DS), "H4_SK.official.netl")
    fresh = (off is not None and os.path.exists(netl) and off["output"]["sha256"] == sha_file(netl) and off["input_hgr"]["sha256"] == hrec["hgr"]["sha256"]
             and (off.get("reconstruction_gate") or {}).get("gate_bytes") == "PASS" and (off.get("reconstruction_gate") or {}).get("gate_digest") == "PASS"
             and sg is not None and sg["gate_semantic"] == "PASS" and sg["official_netl"]["sha256"] == off["output"]["sha256"] and sg["STREAM_MANIFEST_SHA256"] == man["STREAM_MANIFEST_SHA256"])
    if not fresh:
        off = FR.convert(DS)
        steps["D_official_netl_and_gates"] = "converted + gated"
    else:
        steps["D_official_netl_and_gates"] = "existed"
    T["D_official_netl_and_gates"] = round(time.time() - t, 1); peaks["D_official_netl_and_gates"] = PB.py_peak_mb()
    # E. the driver's stream manifest (deterministic bytes)
    smf = FR.write_stream_manifest(DS, man)
    with io.open(smf, encoding="utf-8") as f:
        smf_lines = f.read().split("\n")
    if smf_lines[0] != man["header"] or len([ln for ln in smf_lines if ln.strip()]) != man["shard_count"] + 1:
        raise RuntimeError("stream_manifest.txt does not list every shard")
    # F. verification: every gate PASS, digest recomputed from the canonical CSR == ORIGINAL == semantic gate's
    t = time.time()
    H, man2, gate_rec, off2, netl2 = FR.load_inputs(DS)
    dig, dstats = HG.structure_digest(H["eptr"], H["eidx"], H["ew"], H["N"])
    sg = rj(os.path.join(ds_dir(DS), "H4_SK.semantic_gate.json"))
    checks = {"digest_recomputed_eq_ORIGINAL": dig == hrec["ORIGINAL_STRUCTURE_SHA256"], "digest_eq_stream_manifest": dig == man2["ORIGINAL_STRUCTURE_SHA256"],
              "digest_eq_semantic_gate_official": dig == sg["STRUCTURE_SHA256_official"], "digest_eq_semantic_gate_shards": dig == sg["STRUCTURE_SHA256_shards"],
              "reconstruction_gate_PASS_PASS": gate_rec["gate_digest"] == "PASS" and gate_rec["gate_bytes"] == "PASS", "semantic_gate_PASS": sg["gate_semantic"] == "PASS",
              "official_vs_canonical_npz_all_true": all(v is True for k_, v in sg["official_vs_canonical_npz"].items() if k_ != "node_weights"),
              "official_vs_shards_all_true": all(v is True for v in sg["official_vs_shards"].values()),
              "counts": {"N": H["N"], "M": H["M"], "P": H["P"], "k": H["k"], "k_rule_max_1_N_over_100": max(1, H["N"] // 100) == H["k"],
                         "shards": man2["shard_count"], "shard_nodes": man2["shard_nodes"], "families": H["meta"]["families"], "famset": H["meta"]["famset"]},
              "families_STRUCT_KNN_only": sorted(H["meta"]["families"]) == ["KNN", "STRUCT"],
              "DATASET_json_RECORD_SHA256_eq_adapter": hm["inputs"]["DATASET_json_RECORD_SHA256"] == H["d"].record_sha}
    checks["ALL"] = all(v is True for k_, v in checks.items() if k_ not in ("counts",)) and checks["counts"]["k_rule_max_1_N_over_100"]
    T["F_verify"] = round(time.time() - t, 1); peaks["F_verify"] = PB.py_peak_mb()
    d = H["d"]
    files = {"H4_SK_npz": npz_p, "H4_SK_manifest": hm_p, "hgr": HG.hgr_path(DS), "hgr_record": HG.hgr_path(DS) + ".json", "stream_manifest_json": ST.manifest_path(DS),
             "stream_pass1_json": os.path.join(ST.stream_dir(DS), "pass1.json"), "reconstruction_gate": os.path.join(ST.stream_dir(DS), "RECONSTRUCTION_GATE.json"),
             "official_netl": netl, "official_netl_json": os.path.join(ds_dir(DS), "H4_SK.official.netl.json"), "semantic_gate": os.path.join(ds_dir(DS), "H4_SK.semantic_gate.json"),
             "freight_stream_manifest_txt": smf, "keys_npz": d._keys_path(), "query_index_npz": d._query_index_path()}
    del H
    rec = {"RECORD": "PHG_WEBQSP_INPUTS", "utc": utc(), "classification": CLASS, "dataset": DS, "ruling": pin(os.path.join(OUT, RULING_W)),
           "preflight_before_inputs": pf, "steps": steps, "timing_seconds": T, "py_peak_mb_after_step(monotone)": peaks,
           "hypergraph_manifest": {k_: hm.get(k_) for k_ in ("N", "hyperedges", "pins", "k", "rule", "families", "famset", "cap", "pin_retention_total", "weight_min", "weight_max",
                                                              "content_digest", "file_sha256", "bytes", "seconds", "built_utc", "inputs")},
           "hgr_record": {"ORIGINAL_STRUCTURE_SHA256": hrec["ORIGINAL_STRUCTURE_SHA256"], "counts": hrec["counts"], "hgr": hrec["hgr"], "seconds": hrec.get("seconds")},
           "stream": {"STREAM_MANIFEST_SHA256": man2["STREAM_MANIFEST_SHA256"], "shard_count": man2["shard_count"], "shard_nodes": man2["shard_nodes"], "header": man2["header"],
                      "ordered_shard_sha256": man2["ordered_shard_sha256"], "pass1": man2["pass1"], "this_run": man["this_run"]},
           "reconstruction_gate": {k_: gate_rec.get(k_) for k_ in ("gate_digest", "gate_bytes", "STREAM_STRUCTURE_SHA256", "seconds")},
           "official_netl": {"output": off2["output"], "time": off2.get("time"), "seconds": off2.get("seconds"), "converter": off2.get("converter")},
           "semantic_gate": {k_: sg.get(k_) for k_ in ("gate_semantic", "STRUCTURE_SHA256_official", "STRUCTURE_SHA256_shards", "counts_official", "counts_shards", "seconds")},
           "structure_digest_recomputed": dig, "structure_stats": dstats, "checks": checks, "pins": {k_: pin(p) for k_, p in files.items()},
           "dataset_pins": d.pins(), "expected_pins_upper_bound_from_kit": (rul.get("authorized") or {}).get("expected_scale", {}).get("expected_pins_upper_bound"),
           "host_after": preflight_sample(None, purpose="inputs_done")}
    if not checks["ALL"]:
        rec["STATUS"] = "INPUT_CHAIN_NOT_EXACT"
        wj(fp, rec)
        raise RuntimeError("input chain verification failed: %s" % {k_: v for k_, v in checks.items() if v is not True})
    rec["STATUS"] = "EXACT"
    wj(fp, rec)
    log("%s inputs EXACT: N %d M %d P %d k %d shards %d  digest %s  py peak %.0f MB  -> %s" % (
        DS, checks["counts"]["N"], checks["counts"]["M"], checks["counts"]["P"], checks["counts"]["k"], checks["counts"]["shards"], dig[:16], max(peaks.values()), os.path.relpath(fp, REPO)))
    return rec


# ----------------------------------------------------------------------------- preregistration
def expected_population():
    """replay_cache.build's sample rule, unchanged, applied to the webqsp train_holdout carve (partition-independent); test never read."""
    from src.l1_canonical import replay_cache as RC
    from src.l1_canonical.adapter import CanonicalDataset
    d = CanonicalDataset(DS)
    rows = [int(x) for x in d.eval_rows()]
    n_carve = len(rows)
    np.random.seed(0)
    np.random.shuffle(rows)
    rows = sorted(rows[:RC.EVAL_CAP])
    keep = [r for r in rows if len(d.gold([r])[0]) > 0]
    ids = d.query_ids
    qids = [ids[r] for r in keep]
    hops = d.hops(keep)
    return {"eval_split": d.eval_split, "carve_ids_sha256": d._carve_sha(), "carve_rows": n_carve, "EVAL_CAP": RC.EVAL_CAP, "sampled_rows": len(rows),
            "dropped_without_gold": len(rows) - len(keep), "nq": len(keep), "row_query_ids_sha256": hashlib.sha256(",".join(qids).encode()).hexdigest(),
            "sample_rule": "EVAL_SPLITS %s -> np.random.seed(0); shuffle; sorted(rows[:EVAL_CAP]); rows without gold dropped (replay_cache.build, unchanged)" % d.eval_split,
            "hop_labels_present": bool((np.asarray(hops) >= 0).any()), "test_rows_read": False,
            "split_ranges": {sp: [int(a), int(b)] for sp, (a, b) in d.split_ranges.items()}}


def prereg():
    pp = os.path.join(OUT, PREREG_W)
    if os.path.exists(pp):
        raise RuntimeError("the WebQSP preregistration already exists -- written once, before the run")
    rul = rj(os.path.join(OUT, RULING_W))
    inp = rj(os.path.join(OUT, INPUTS_W))
    if rul is None or inp is None or inp.get("STATUS") != "EXACT":
        raise RuntimeError("ruling + EXACT inputs record required first")
    if os.path.exists(os.path.join(OUT, RUNS_W)) or os.path.exists(os.path.join(ds_dir(DS), "phg")):
        raise RuntimeError("a WebQSP PHG run directory / record already exists -- the preregistration must precede any run")
    small = rj(RULING_S)
    lane_pre, rep_pre, bld = rj(os.path.join(OUT, "PHG_PREREG.json")), rj(os.path.join(OUT, "PHG_REPAIR_PREREG.json")), rj(os.path.join(OUT, "PHG_BUILD.json"))
    pk = rj(os.path.join(OUT, "PHG_PACKAGES.json"))
    from src.l1_canonical.adapter import contract_hash
    ch = contract_hash()
    pop = expected_population()
    hm = rj(os.path.join(CANON, "hypergraph", "H4_SK.json"))
    N, M, Pp, k = hm["N"], hm["hyperedges"], hm["pins"], hm["k"]
    ref_replay = os.path.join(CANON_L1, "L1_REPLAY_%s.json" % DS)
    ref_part = os.path.join(CANON, "parts", "H4_SK.npy")
    frozen = [fn for fn in P.FROZEN_READ_ONLY] + ["PHG_PREREG.json", "PHG_BUILD.json", "PHG_PACKAGES.json", "PHG_RUNS_squad.json", "PHG_RUNS_metaqa.json", "PHG_REPAIR_PREREG.json",
                                                   "PHG_REPAIR_RUNS_squad.json", "PHG_REPAIR_RUNS_metaqa.json", "PHG_REPORT.json", "PHG_REPORT.md", "L1_DOWNSTREAM_squad__phg.json",
                                                   "L1_DOWNSTREAM_metaqa__phg_repair1.json", "L1_REPLAY_squad__LOWMEM__PHG_con.json", "L1_REPLAY_metaqa__LOWMEM__PHG_REPAIR1_con.json",
                                                   "L1_REPLAY_musique__LOWMEM__FREIGHT_con.json", "H4_SK_CONTRACT_FREEZE.json", RL.RULING_RECORD, RULING_W, INPUTS_W] + BRIDGE_RECORDS
    mods = {"phg.py": P.__file__, "phg_repair.py": PR.__file__, "phg_ruling.py": RL.__file__, "phg_bridge.py": PB.__file__, "hgr.py": HG.__file__, "stream.py": ST.__file__,
            "freight.py": FR.__file__, "l1_downstream.py": os.path.join(os.path.dirname(THIS), "l1_downstream.py"), "common.py": os.path.join(os.path.dirname(THIS), "common.py")}
    rec = {"RECORD": "PHG_WEBQSP_PREREG", "utc": utc(), "classification": CLASS, "dataset": DS, "lane": LANE, "encoding": ENCODING, "tag": TAG_W, "suffix": SUFFIX_W,
           "question": QUESTION,
           "authorization": pin(os.path.join(OUT, RULING_W)), "small_dataset_ruling": pin(RULING_S),
           "musique_bridge_result": {"DECISION": "PHG_SCALE_BRIDGE_COMPLETE", "records": {fn: pin(os.path.join(OUT, fn)) for fn in BRIDGE_RECORDS}, "frozen_unchanged": True},
           "contract": {"name": RL.EXPERIMENTAL_PHG_CONTRACT, "lane_preregistration": pin(os.path.join(OUT, "PHG_PREREG.json")), "repair_preregistration": pin(os.path.join(OUT, "PHG_REPAIR_PREREG.json")),
                        "build": pin(os.path.join(OUT, "PHG_BUILD.json")), "driver_binary_sha256": lane_pre["build"]["driver_binary_sha256"], "driver_source_sha256": lane_pre["build"]["driver_source_sha256"],
                        "wrapper_sha256": lane_pre["build"].get("wrapper_sha256"), "packages": {"record": pin(os.path.join(OUT, "PHG_PACKAGES.json")), "dpkg_versions": (pk or {}).get("after", {}).get("dpkg_versions")},
                        "phg_parameters_in_order": P.PARAMS, "parameter_notes": P.PARAM_NOTES, "undocumented_left_at_library_default": P.UNDOCUMENTED_LEFT_AT_LIBRARY_DEFAULT,
                        "NUM_GLOBAL_PARTS": k, "k_rule": "max(1, N // 100)", "mpi_ranks": P.NP, "rank_count_rule": small["experimental_PHG_contract"]["rank_count_rule"],
                        "imbalance": {"IMBALANCE_TOL": "1.03", "validity_bound": "ceil(1.03 N/k) = %d, no empty block" % int(math.ceil(1.03 * N / float(k)))},
                        "objective": "PHG_CUT_OBJECTIVE=CONNECTIVITY (weighted KM1)", "PHG_EDGE_SIZE_THRESHOLD": "1.0", "deterministic": "DETERMINISTIC=1, PHG_RANDOMIZE_INPUT=0",
                        "repair": PR.REPAIR, "repair_algorithm": PR.ALGORITHM, "repair_implementation_sha256": sha_file(PR.__file__),
                        "no_minimum_part_size_condition": True, "no_tuning": "no parameter change, no second configuration, no other rank count; PHG is run once (R1 = the arm's partition) "
                                                                            "plus one identical repeat run (diagnostic only)",
                        "mpirun_timeout_seconds_harness": MPIRUN_TIMEOUT, "modules": {name: pin(p) for name, p in mods.items()}, "canonical_L1_CONTRACT_SHA256": ch["L1_CONTRACT_SHA256"]},
           "program": pin(THIS),
           "inputs": {"record": pin(os.path.join(OUT, INPUTS_W)), "N": N, "M": M, "P": Pp, "k": k, "contract_bound_ceil_1.03_N_over_k": int(math.ceil(1.03 * N / float(k))),
                      "dataset_pins": inp["dataset_pins"], "H4_SK": {"npz_sha256": hm["file_sha256"], "content_digest": hm["content_digest"], "manifest": pin(os.path.join(CANON, "hypergraph", "H4_SK.json")),
                                                                    "rule": hm["rule"], "families": hm["families"], "famset": hm["famset"], "DATASET_json_RECORD_SHA256": hm["inputs"]["DATASET_json_RECORD_SHA256"],
                                                                    "keys_npz_sha256": hm["inputs"]["keys_npz_sha256"]},
                      "ORIGINAL_STRUCTURE_SHA256": inp["hgr_record"]["ORIGINAL_STRUCTURE_SHA256"], "structure_stats": inp["structure_stats"],
                      "stream": {"STREAM_MANIFEST_SHA256": inp["stream"]["STREAM_MANIFEST_SHA256"], "shard_count": inp["stream"]["shard_count"], "shard_nodes": inp["stream"]["shard_nodes"],
                                 "ordered_shard_sha256": inp["stream"]["ordered_shard_sha256"]},
                      "stream_manifest": pin(os.path.join(ds_dir(DS), "freight", "stream_manifest.txt")), "official_netl_sha256": inp["official_netl"]["output"]["sha256"],
                      "semantic_gate": inp["semantic_gate"], "reconstruction_gate": inp["reconstruction_gate"], "data_pins": inp["pins"]},
           "expected_population": pop,
           "preflight": PREFLIGHT,
           "reference_policy": {"canonical_mtkahypar_reference_exists": os.path.exists(ref_replay) and os.path.exists(ref_part),
                                "canonical_replay_record": pin(ref_replay), "canonical_partition": pin(ref_part),
                                "expected_downstream_status": "PAIRED" if (os.path.exists(ref_replay) and os.path.exists(ref_part)) else "BASELINE_ABSENT",
                                "classification_when_absent": "scale / absolute-L1 result; no equivalence claim; no comparative quality claim",
                                "mtkahypar_memory_expectation": pin(os.path.join(CANON_L1, "EXTERNAL_LANE_MEMORY_EXPECTATION.json")),
                                "legacy_name_plus_facts_webqsp": {"policy": "historical context only -- different corpus encoding (NAME + verbalised facts), N 781,485, k 7,814, "
                                                                            "all 1,578 evaluable queries, legacy partition rule; never compared numerically; never a reproduction target",
                                                                  "record": pin(os.path.join(REPO, "results", "L1", "L1_LOCKED_MANIFEST.json"))}},
           "stages": ["preflight before stage 1 (refuse without a record when contended); preflight (waiting up to the budget) before every mpirun launch and before stage 4",
                      "1 H4_SK input verified against this record (files, pins, recomputed structure digest == ORIGINAL)",
                      "2 gate-mode PHG run G = exact structure gate from the query-function dumps (STRUCTURE_MISMATCH = STOP); partition run R1 (NP=4, timeout %d s); "
                      "R1_repeat identity diagnostic if feasible; dumps of R1 byte-identical to G's" % MPIRUN_TIMEOUT,
                      "3 %s on R1 (no-op when no block is empty); postconditions; exact H4_SK unchanged (structure digest recomputed from the in-memory CSR after the repair); "
                      "canonical validity gate" % PR.REPAIR,
                      "4 experimental import parts/%s.* -> P50 replay cache (unchanged canonical builder, fp16 memory-mapped path) -> canonical L1 numerics (l1_eval, unchanged) "
                      "-> L1_DOWNSTREAM_%s%s.json" % (TAG_W, DS, SUFFIX_W)],
           "labels": LABELS, "decision_rule": DECISION_RULE, "sanity_checks": SANITY, "diagnostics_no_rule": DIAGNOSTICS_NO_RULE,
           "measurements": ["per stage: wall seconds; peak RSS per MPI rank (/usr/bin/time -v + getrusage); sum of rank peaks (upper bound on the simultaneous aggregate); "
                            "host Python peak working set after each host stage", "preflight samples immediately before every launch (the six quantities the authorization lists + paging rate)",
                            "Zoltan cutl / imbalance; Python KM1 / cut / lambda; blocks min p50 p95 max empty; STRUCT / KNN edge-cut fractions",
                            "replay-cache builder's predicted peak and BASE / STRUCT cost from its manifest; l1_eval numerics as recorded"],
           "stop_rule": "STOP_FOR_REVIEW after the WebQSP record; HotpotQA / 2Wiki / the 302M Freebase lane are not started; no promotion; no tuning; no NP change; "
                        "MuSiQue is not rerun; the canonical webqsp partition slot stays empty",
           "frozen_read_only": {fn: pin(os.path.join(OUT, fn)) for fn in frozen if os.path.exists(os.path.join(OUT, fn))},
           "musique_files_untouched": musique_files(),
           "no_results_inspected": "no WebQSP PHG run exists at this time (data/l1_lowmem/webqsp/phg absent); the inputs record holds structure facts only"}
    wj(pp, rec)
    log("preregistered", CLASS, "-> %s  (N %d M %d P %d k %d; population nq %d sha %s; reference %s)" % (
        os.path.relpath(pp, REPO), N, M, Pp, k, pop["nq"], pop["row_query_ids_sha256"][:16], rec["reference_policy"]["expected_downstream_status"]))
    return rec


# ----------------------------------------------------------------------------- run
MEM_PAT = re.compile(r"cannot allocate|out of memory|bad_alloc|oom|killed|signal 9|ENOMEM|exit code 137|status 137|MPI_ERR_NO_MEM|malloc", re.I)


def classify_mpirun_failure(name, err, pf_ok):
    d = os.path.join(ds_dir(DS), "phg", name)
    txt = str(err)
    ev = {"error": txt[:3000], "stderr_memory_pattern": bool(MEM_PAT.search(txt))}
    ranks = [FR.parse_time_v(os.path.join(d, "time_rank%d.txt" % i)) for i in range(P.NP)] if os.path.isdir(d) else []
    ev["rank_time_v"] = ranks
    ev["rank_signal_9_or_exit_137"] = any(tv.get("signal") == 9 or tv.get("exit_status") == 137 for tv in ranks)
    ev["rank_peak_rss_mb"] = [round(tv.get("peak_rss_kb", 0) / 1024.0, 1) for tv in ranks]
    ev["job_time_v"] = FR.parse_time_v(os.path.join(d, "time_job.txt")) if os.path.isdir(d) else {}
    for fn in ("mpirun_stderr.txt", "mpirun_stdout.txt"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            tail = io.open(p, encoding="utf-8", errors="replace").read()
            ev[fn + "_tail"] = tail[-2000:]
            ev["stderr_memory_pattern"] = ev["stderr_memory_pattern"] or bool(MEM_PAT.search(tail))
    try:
        r = wsl("dmesg 2>/dev/null | grep -i -E 'out of memory|killed process|oom-kill|oom_reaper' | tail -5", timeout=60)
        ev["wsl_dmesg_oom_lines"] = [ln for ln in r.stdout.splitlines() if ln.strip()]
        ev["wsl_dmesg_readable"] = True
    except Exception as e:
        ev["wsl_dmesg_oom_lines"] = []
        ev["wsl_dmesg_readable"] = "unavailable: %s" % e
    ev["host_after_failure"] = preflight_sample(None, purpose="after_%s_failure" % name)
    ev["memory_evidence"] = bool(ev["stderr_memory_pattern"] or ev["rank_signal_9_or_exit_137"] or ev["wsl_dmesg_oom_lines"])
    ev["preflight_clean_at_launch"] = pf_ok
    ev["wsl_vm_note"] = "the WSL VM holds the 4 ranks; its size is the distro default (host RAM / 2, ~7.8 GB); .wslconfig was not modified"
    label = "WEBQSP_NP4_RESOURCE_INFEASIBLE" if (ev["memory_evidence"] and pf_ok) else "WEBQSP_NP4_RUN_FAIL_UNCLASSIFIED"
    return label, ev


def stop(R, fp, decision, stage, err=None):
    R["DECISION"] = decision
    R["stopped_at_stage"] = stage
    if err is not None:
        R["error"] = err
    R["host_after"] = preflight_sample(None, purpose="run_end")
    wj(fp, R)
    log("%s DECISION %s (stage %s)%s -> %s" % (DS, decision, stage, (": " + str(err)[:300]) if err else "", os.path.relpath(fp, REPO)))
    return R


def gate_launch(R, fp, purpose, launched_any):
    """preflight immediately before a launch, waiting (60 s polls) up to the deadline; None = budget exhausted."""
    pf = formal_preflight(purpose, wait_until=deadline())
    R["preflight"][purpose] = pf if pf is not None else {"budget_exhausted": True, "utc": utc()}
    wj(fp, R)
    if pf is None:
        stop(R, fp, "WEBQSP_SUSPENDED_HOST_RESOURCE_CONTENDED" if launched_any else "WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED", purpose,
             "the host was not clean for the preflight before %s within the wait budget (deadline %s); nothing was killed" % (purpose, rj(WAIT_STATE) and rj(WAIT_STATE).get("deadline_utc")))
    return pf


def run():
    pre = rj(os.path.join(OUT, PREREG_W))
    if pre is None:
        raise RuntimeError("run prereg first")
    if pin(THIS)["sha256"] != pre["program"]["sha256"]:
        raise RuntimeError("phg_webqsp.py changed since preregistration")
    for name, pn in pre["contract"]["modules"].items():
        if sha_file(os.path.join(REPO, pn["path"])) != pn["sha256"]:
            raise RuntimeError("%s changed since preregistration" % name)
    for key, pn in (("authorization", pre["authorization"]), ("small_dataset_ruling", pre["small_dataset_ruling"]), ("inputs_record", pre["inputs"]["record"])):
        if sha_file(os.path.join(REPO, pn["path"])) != pn["sha256"]:
            raise RuntimeError("%s changed since preregistration" % key)
    fp = os.path.join(OUT, RUNS_W)
    if os.path.exists(fp):
        raise RuntimeError("%s exists -- the run happens once; supersede, never overwrite" % os.path.relpath(fp, REPO))
    pf0 = formal_preflight("before_stage1", wait_until=None)
    if pf0 is None:
        log("run: host CONTENDED -- refusing to start (nothing written)")
        sys.exit(3)
    lane_pre = rj(os.path.join(OUT, "PHG_PREREG.json"))
    b = P.build()
    if b["binary"]["sha256"] != lane_pre["build"]["driver_binary_sha256"] or b["source"]["sha256"] != lane_pre["build"]["driver_source_sha256"]:
        raise RuntimeError("driver changed since the lane preregistration (contract element)")
    b["packages_versions"] = rj(os.path.join(OUT, "PHG_PACKAGES.json"))["after"]["dpkg_versions"]
    T = {}
    R = {"RECORD": "PHG_WEBQSP_RUNS", "classification": CLASS, "arm": RL.EXPERIMENTAL_PHG_CONTRACT, "dataset": DS, "lane": LANE, "encoding": ENCODING, "utc": utc(),
         "preregistration": pin(os.path.join(OUT, PREREG_W)), "authorization": pin(os.path.join(OUT, RULING_W)), "small_dataset_ruling": pin(RULING_S), "program": pin(THIS),
         "wait_state": rj(WAIT_STATE), "preflight": {"criterion": PREFLIGHT, "before_stage1": pf0},
         "contract": {"parameters": P.PARAMS, "mpi_ranks": P.NP, "driver": b["binary"], "driver_source": b["source"], "packages": b["packages_versions"], "repair": PR.REPAIR,
                      "mpirun_timeout_seconds_harness": MPIRUN_TIMEOUT},
         "stages": {}, "runs": {}, "timing_seconds": T}
    wj(fp, R)
    # ---- stage 1: H4_SK input verified against the preregistration
    t = time.time()
    try:
        H, man, gate_rec, off, netl = FR.load_inputs(DS)
        N, k, M, Pp = H["N"], H["k"], H["M"], H["P"]
        ins = pre["inputs"]
        if (N, M, Pp, k) != (ins["N"], ins["M"], ins["P"], ins["k"]) or H["npz_sha256"] != ins["H4_SK"]["npz_sha256"] or man["ORIGINAL_STRUCTURE_SHA256"] != ins["ORIGINAL_STRUCTURE_SHA256"]:
            raise RuntimeError("H4_SK differs from the preregistered input")
        if H["meta"].get("content_digest") != ins["H4_SK"]["content_digest"] or sorted(H["meta"]["families"]) != ["KNN", "STRUCT"]:
            raise RuntimeError("H4_SK content digest / family set differs from the preregistered input")
        smf = os.path.join(ds_dir(DS), "freight", "stream_manifest.txt")
        if sha_file(smf) != ins["stream_manifest"]["sha256"] or man["STREAM_MANIFEST_SHA256"] != ins["stream"]["STREAM_MANIFEST_SHA256"] or man["shard_nodes"] != ins["stream"]["shard_nodes"]:
            raise RuntimeError("stream manifest changed since the preregistration")
        if man["ordered_shard_sha256"] != ins["stream"]["ordered_shard_sha256"]:
            raise RuntimeError("shard digests changed since the preregistration")
        for key, pn in ins["data_pins"].items():
            if pn and sha_file(os.path.join(REPO, pn["path"])) != pn["sha256"]:
                raise RuntimeError("input pin %s changed since the preregistration" % key)
        dig, dstats = HG.structure_digest(H["eptr"], H["eidx"], H["ew"], N)
        if dig != man["ORIGINAL_STRUCTURE_SHA256"]:
            raise RuntimeError("H4_SK structure digest %s != ORIGINAL" % dig)
    except Exception as e:
        return stop(R, fp, "WEBQSP_INPUT_VERIFICATION_FAIL", "1_input_verified", "%s: %s" % (type(e).__name__, e))
    T["1_input_verified"] = round(time.time() - t, 1)
    stage1 = {"stage": "1_H4_SK_input_verified", "utc": utc(), "N": N, "M": M, "P": Pp, "k": k, "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"],
              "structure_digest_recomputed": dig, "structure_stats": dstats, "hypergraph_content_digest": H["meta"].get("content_digest"), "H4_SK_npz_sha256": H["npz_sha256"],
              "reconstruction_gate": {kk: gate_rec[kk] for kk in ("gate_digest", "gate_bytes") if kk in gate_rec}, "shard_count": man["shard_count"], "shard_nodes": man["shard_nodes"],
              "STREAM_MANIFEST_SHA256": man["STREAM_MANIFEST_SHA256"], "stream_manifest": ins["stream_manifest"], "official_netl_sha256": off["output"]["sha256"],
              "seconds": T["1_input_verified"], "py_peak_mb": PB.py_peak_mb()}
    R["inputs"] = {"N": N, "M": M, "P": Pp, "k": k, "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "official_netl": off["output"], "stream_manifest": ins["stream_manifest"]}
    R["stages"]["1_input_verified"] = stage1
    wj(os.path.join(ds_dir(DS), "phg", "STAGE1_INPUT_VERIFIED.json"), stage1)
    wj(fp, R)
    log("=== %s: N %d M %d P %d k %d shards %d ranks %d  (H4_SK digest %s == ORIGINAL) ===" % (CLASS, N, M, Pp, k, man["shard_count"], P.NP, dig[:16]))
    # ---- stage 2a: gate-mode run G = exact structure gate
    if gate_launch(R, fp, "before_G", launched_any=False) is None:
        return R
    t = time.time()
    try:
        g_res, g_dir = P.mpirun(DS, "G", "gate", k, smf, b, timeout=MPIRUN_TIMEOUT)
    except Exception as e:
        label, ev = classify_mpirun_failure("G", e, True)
        R["failure_evidence"] = ev
        return stop(R, fp, label, "2_structure_gate_run_G", "%s: %s" % (type(e).__name__, e))
    R["runs"]["G"] = g_res
    wj(fp, R)
    gate = P.structure_gate(DS, g_dir, man, H=H, zoltan_eval=g_res["zoltan_eval"])
    gate.update({"stage": "2_PHG_input_callback_manifest", "run": "G", "dump_sha256": g_res["dump_sha256"], "global_from_queries": g_res["global_from_queries"],
                 "zoltan_removed_or_warning_lines": g_res["zoltan_removed_or_warning_lines"],
                 "required": ["node count", "hyperedge count", "pin count", "unit vertex weights", "hyperedge weights", "global node positions", "global hyperedge ids",
                              "incidence multiset", "canonical structure digest"]})
    T["2_structure_gate"] = round(time.time() - t, 1)
    R["stages"]["2_structure_gate"] = gate
    wj(os.path.join(ds_dir(DS), "phg", "STAGE2_PHG_INPUT_CALLBACK_MANIFEST.json"), gate)
    wj(fp, R)
    log("  structure gate: %s  (N %s M %s P %s weights %d..%d sum %d; digest %s; weight reception rel diff %s; removal/warning lines %d)" % (
        gate["gate"], gate["N_returned"], gate["M_with_pins"], gate["P_returned"], gate["weight_min"] or -1, gate["weight_max"], gate["weight_sum"],
        (gate["digest_from_queries"] or "-")[:16], (gate.get("weight_reception") or {}).get("relative_diff"), len(gate["zoltan_removed_or_warning_lines"] or [])))
    if gate["gate"] != "PASS":
        R["problems"] = gate["problems"]
        return stop(R, fp, "STRUCTURE_MISMATCH", "2_structure_gate")
    # ---- stage 2b: the partition run R1 (+ repeat)
    if gate_launch(R, fp, "before_R1", launched_any=True) is None:
        return R
    t = time.time()
    try:
        r1, d1 = P.mpirun(DS, "R1", "partition", k, smf, b, timeout=MPIRUN_TIMEOUT)
    except Exception as e:
        label, ev = classify_mpirun_failure("R1", e, True)
        R["failure_evidence"] = ev
        return stop(R, fp, label, "2_partition_R1", "%s: %s" % (type(e).__name__, e))
    R["runs"]["R1"] = r1
    r1["dumps_identical_to_gate_run"] = r1["dump_sha256"] == g_res["dump_sha256"]
    wj(fp, R)
    if not r1["dumps_identical_to_gate_run"]:
        R["problems"] = ["partition-run query dumps differ from the gate run's"]
        return stop(R, fp, "STRUCTURE_MISMATCH", "2_partition_R1")
    hard_r1 = P.assemble_partition(d1, N, k)
    v_r1 = P.validity(hard_r1, N, k)
    m_r1 = FR.km1_metrics(H, hard_r1); m_r1["family_cuts"] = FR.family_cuts(DS, hard_r1)
    R["stages"]["2_structure_gate"]["partition_run_dumps_identical"] = True
    T["2_partition_R1"] = round(time.time() - t, 1)
    zc = r1["zoltan_eval"]["cutl_global"]
    stage3 = {"stage": "2_completed_PHG_partition", "run": "R1", "validity_raw": v_r1, "metrics_raw": m_r1, "partition_sha256_txt": {i: sha_file(os.path.join(d1, "part_rank%d.txt" % i)) for i in range(P.NP)},
              "vector_sha256_raw": PR.vec_sha(hard_r1), "zoltan_eval": r1["zoltan_eval"],
              "python_km1_vs_zoltan_cutl": {"python_km1": m_r1["km1_weighted"], "zoltan_cutl_float": zc, "relative_diff": round(abs(m_r1["km1_weighted"] - zc) / max(1.0, m_r1["km1_weighted"]), 8),
                                            "consistent_within_float32": bool(abs(m_r1["km1_weighted"] - zc) <= max(1.0, 1e-6 * m_r1["km1_weighted"]))},
              "memory": r1["memory"], "timing": r1["timing"], "mpi_ranks": P.NP, "py_peak_mb": PB.py_peak_mb()}
    R["stages"]["2_partition"] = stage3
    R["metrics_raw"] = m_r1
    wj(os.path.join(ds_dir(DS), "phg", "STAGE2_PHG_PARTITION.json"), stage3)
    wj(fp, R)
    log("%s R1: raw validity %s (max %d, bound %d, empty %d)  km1 %d  cut %d  STRUCT %.4f KNN %.4f  zoltan cutl %.0f imb %.4f  peak RSS/rank max %.0f MB sum %.0f MB  partition %.1f s job %s s" % (
        DS, v_r1["gate"], v_r1["max_block"], v_r1["contract_bound_ceil_1.03_N_over_k"], v_r1["empty_blocks"], m_r1["km1_weighted"], m_r1["cut_weighted"],
        m_r1["family_cuts"]["STRUCT"]["edge_cut_fraction"], m_r1["family_cuts"]["KNN"]["edge_cut_fraction"], zc, r1["zoltan_eval"]["imbalance"],
        r1["memory"]["max_rank_peak_kb"] / 1024.0, r1["memory"]["sum_of_rank_peaks_kb"] / 1024.0, r1["timing"]["partition_wall_seconds"], r1["timing"]["job_wall_seconds"]))
    # repeat run (diagnostic only; R1 stands); preflight-gated like every launch
    t = time.time()
    pfr = formal_preflight("before_R1_repeat", wait_until=deadline())
    R["preflight"]["before_R1_repeat"] = pfr if pfr is not None else {"budget_exhausted": True, "utc": utc()}
    if pfr is None:
        R["repeat"] = {"identical_partition": None, "not_run": "host not clean within the wait budget; the repeat is a diagnostic, R1 stands"}
    else:
        try:
            r2, d2 = P.mpirun(DS, "R1_repeat", "partition", k, smf, b, timeout=MPIRUN_TIMEOUT)
            hard2 = P.assemble_partition(d2, N, k)
            R["runs"]["R1_repeat"] = r2
            R["repeat"] = {"identical_partition": bool(np.array_equal(hard_r1, hard2)), "n_diff": int((hard_r1 != hard2).sum()), "km1_repeat": FR.km1_metrics(H, hard2)["km1_weighted"],
                           "note": "same NP, same input, same parameters; diagnostic only -- R1 is the arm's partition"}
            del hard2
        except Exception as e:
            label, ev = classify_mpirun_failure("R1_repeat", e, True)
            R["repeat"] = {"identical_partition": None, "error": "%s: %s" % (type(e).__name__, e), "failure_evidence": ev, "would_be_label": label,
                           "note": "repeat run failed; diagnostic only, R1 stands"}
    T["2_repeat"] = round(time.time() - t, 1)
    stage3["repeat"] = R["repeat"]
    wj(os.path.join(ds_dir(DS), "phg", "STAGE2_PHG_PARTITION.json"), stage3)
    wj(fp, R)
    log("%s repeat: identical %s" % (DS, R["repeat"].get("identical_partition")))
    # ---- stage 3: PHG_NONEMPTY_REPAIR_V1 (unchanged; no-op when no block is empty)
    t = time.time()
    hard, empties, moves = PR.repair(H, hard_r1)
    v = P.validity(hard, N, k)
    m = FR.km1_metrics(H, hard); m["family_cuts"] = FR.family_cuts(DS, hard)
    changed = np.where(hard != hard_r1)[0]
    dig_after, _ = HG.structure_digest(H["eptr"], H["eidx"], H["ew"], N)
    post = {"same_N": len(hard) == N, "assigned_once": bool(len(hard) == N and hard.min() >= 0 and hard.max() < k), "same_k": True,
            "empty_blocks_after": v["empty_blocks"], "max_block_after": v["max_block"], "contract_bound": v["contract_bound_ceil_1.03_N_over_k"],
            "H4_SK_unchanged": dig_after == man["ORIGINAL_STRUCTURE_SHA256"] and sha_file(H["npz"]) == H["npz_sha256"],
            "moves": len(moves), "empty_blocks_before": len(empties), "moves_equal_empties": len(moves) == len(empties), "nodes_changed": int(len(changed)),
            "nodes_changed_equal_moves": int(len(changed)) == len(moves), "km1_consistent": m["km1_weighted"] == m_r1["km1_weighted"] + sum(mv["delta_km1_weighted"] for mv in moves),
            "no_minimum_part_size_condition": True}
    post["PASS"] = bool(post["same_N"] and post["assigned_once"] and post["empty_blocks_after"] == 0 and post["max_block_after"] <= post["contract_bound"] and post["H4_SK_unchanged"]
                        and post["moves_equal_empties"] and post["nodes_changed_equal_moves"] and post["km1_consistent"])
    T["3_repair"] = round(time.time() - t, 1)
    rdir = os.path.join(ds_dir(DS), "phg_repair1"); os.makedirs(rdir, exist_ok=True)
    npy_rep = os.path.join(rdir, "repaired.npy"); np.save(npy_rep, hard.astype(np.int64))
    stage3b = {"stage": "3_%s" % PR.REPAIR, "noop": len(empties) == 0, "empty_blocks": empties, "moves": moves, "postconditions": post,
               "before": {"validity": v_r1, "km1_weighted": m_r1["km1_weighted"], "cut_weighted": m_r1["cut_weighted"], "blocks": m_r1["blocks"], "family_cuts": m_r1["family_cuts"], "vector_sha256": PR.vec_sha(hard_r1)},
               "after": {"validity": v, "km1_weighted": m["km1_weighted"], "cut_weighted": m["cut_weighted"], "blocks": m["blocks"], "family_cuts": m["family_cuts"], "vector_sha256": PR.vec_sha(hard),
                         "delta_km1_total": m["km1_weighted"] - m_r1["km1_weighted"],
                         "struct_cut_delta": round(m["family_cuts"]["STRUCT"]["edge_cut_fraction"] - m_r1["family_cuts"]["STRUCT"]["edge_cut_fraction"], 6),
                         "knn_cut_delta": round(m["family_cuts"]["KNN"]["edge_cut_fraction"] - m_r1["family_cuts"]["KNN"]["edge_cut_fraction"], 6), "repaired_npy": pin(npy_rep),
                         "structure_digest_after_repair": dig_after},
               "seconds": T["3_repair"], "py_peak_mb": PB.py_peak_mb()}
    R["stages"]["3_repair"] = stage3b
    R["metrics"] = m
    wj(os.path.join(ds_dir(DS), "phg", "STAGE3_REPAIR.json"), stage3b)
    wj(fp, R)
    log("%s repair: empty blocks %d  moves %d  delta-KM1 total %+d (zero-delta moves %d)  KM1 %d -> %d  max %d -> %d  empty %d -> %d  H4_SK unchanged %s  post %s  validity %s  (%.0f s)" % (
        DS, len(empties), len(moves), stage3b["after"]["delta_km1_total"], sum(1 for mv in moves if mv["delta_km1_weighted"] == 0), m_r1["km1_weighted"], m["km1_weighted"],
        v_r1["max_block"], v["max_block"], v_r1["empty_blocks"], v["empty_blocks"], post["H4_SK_unchanged"], "PASS" if post["PASS"] else "FAIL", v["gate"], T["3_repair"]))
    if not post["PASS"]:
        return stop(R, fp, "PHG_REPAIR_POSTCONDITION_FAIL", "3_repair")
    if v["gate"] != "PASS":
        return stop(R, fp, "PHG_PARTITION_INVALID", "3_validity")
    # context (resources, not quality): the frozen Mt-KaHyPar recipe's expectation for canonical webqsp; no partition exists
    exp = (rj(os.path.join(CANON_L1, "EXTERNAL_LANE_MEMORY_EXPECTATION.json")) or {})
    ex = (exp.get("expectation_GB") or {}).get(DS) or {}
    R["mtkahypar_reference"] = {"status": "ABSENT", "canonical_replay_record_exists": os.path.exists(os.path.join(CANON_L1, "L1_REPLAY_%s.json" % DS)),
                               "canonical_partition_exists": os.path.exists(os.path.join(CANON, "parts", "H4_SK.npy")),
                               "memory_expectation_frozen_recipe": {"linear_in_pins_GB": ex.get("linear_in_pins_GB"), "pins_times_k_GB": ex.get("pins_times_k_GB"), "verdict": str(exp.get("verdict") or "").split(":")[0]},
                               "note": "no partition, no L1 replay; nothing to compare against; the memory expectation is a resource context, not a quality reference"}
    R["diagnostic_context"] = {"blocker_resources": {"mtkahypar_expectation_GB_linear_in_pins": ex.get("linear_in_pins_GB"), "mtkahypar_expectation_GB_pins_times_k": ex.get("pins_times_k_GB"),
                                                     "phg_peak_rss_mb_max_rank": round(r1["memory"]["max_rank_peak_kb"] / 1024.0, 1),
                                                     "phg_peak_rss_mb_sum_ranks_upper_bound": round(r1["memory"]["sum_of_rank_peaks_kb"] / 1024.0, 1),
                                                     "phg_partition_wall_seconds": r1["timing"]["partition_wall_seconds"], "phg_job_wall_seconds": r1["timing"]["job_wall_seconds"]}}
    # ---- stage 4: experimental import -> P50 replay cache (fp16 memory-mapped path) -> canonical L1 numerics (absolute; BASELINE_ABSENT expected)
    if gate_launch(R, fp, "before_stage4", launched_any=True) is None:
        return R
    t = time.time()
    try:
        Rimp = dict(R); Rimp["metrics"] = m; Rimp["repeat"] = R["repeat"]
        extra = {"repair": {"adapter": PR.REPAIR, "noop": len(empties) == 0, "empty_blocks": empties, "moves": moves, "km1_before": m_r1["km1_weighted"], "km1_after": m["km1_weighted"],
                            "postconditions": post, "vector_sha256_in": stage3b["before"]["vector_sha256"], "vector_sha256_out": stage3b["after"]["vector_sha256"]},
                 "scale_run": {"classification": CLASS, "lane": LANE, "encoding": ENCODING, "preregistration": R["preregistration"], "authorization": R["authorization"], "program": R["program"],
                               "note": "PHG_SCALE_WEBQSP: no canonical Mt-KaHyPar reference exists for this corpus; absolute L1 numbers only, no equivalence claim; legacy NAME+FACTS cells are "
                                       "historical context only"}}
        rec = P.import_partition(DS, H, hard, Rimp, b, off, man, gate, tag=TAG_W, extra=extra)
        mp = os.path.join(H["d"].derived_dir, "parts", "%s.json" % TAG_W)
        rec = rj(mp)
        rec["arm"] = RL.EXPERIMENTAL_PHG_CONTRACT
        rec["contract"]["repair"] = "%s applied to the PHG R1 vector (%d empty block(s), %d move(s)); no-op when no block is empty" % (PR.REPAIR, len(empties), len(moves))
        wj(mp, rec)
        R["import"] = rec
        wj(fp, R)
        del H, hard_r1, changed
        from src.l1_lowmem import l1_downstream as LD
        D = LD.run(DS, tag=TAG_W, suffix=SUFFIX_W)
    except Exception as e:
        R["failure_evidence"] = {"memory_error": isinstance(e, MemoryError) or bool(MEM_PAT.search(str(e))), "host_after_failure": preflight_sample(None, purpose="after_stage4_failure")}
        return stop(R, fp, "WEBQSP_L1_EXECUTION_FAIL", "4_import_replay_cache_L1", "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2000:]))
    T["4_import_replay_cache_L1"] = round(time.time() - t, 1)
    cache_man = rj(os.path.join(ds_dir(DS), "replay_cache__%s.json" % TAG_W)) or {}
    replay = rj(os.path.join(OUT, "L1_REPLAY_%s__%s.json" % (DS, TAG_W))) or {}
    B = D.get("B") or {}
    R["downstream"] = {"status": D["status"], "record": pin(os.path.join(OUT, "L1_DOWNSTREAM_%s%s.json" % (DS, SUFFIX_W))), "B_phg": B, "population": D.get("population_B"),
                       "paired": D.get("paired"), "replay_record": pin(os.path.join(OUT, "L1_REPLAY_%s__%s.json" % (DS, TAG_W))),
                       "cache": {"manifest": pin(os.path.join(ds_dir(DS), "replay_cache__%s.json" % TAG_W)), "bytes": cache_man.get("bytes"), "sha256": cache_man.get("sha256"),
                                 "predicted_peak_gb": (cache_man.get("meta") or {}).get("predicted_peak_gb"), "COST": (cache_man.get("meta") or {}).get("COST"),
                                 "fp16_path": (cache_man.get("meta") or {}).get("fp16_path"), "node_embedding_path": (cache_man.get("meta") or {}).get("node_embedding_path")},
                       "eval_seconds": replay.get("seconds"), "py_peak_mb": PB.py_peak_mb(), "by_hop": replay.get("by_hop")}
    wj(os.path.join(ds_dir(DS), "phg", "STAGE4_P50_REPLAY_ABSOLUTE.json"), R["downstream"])
    # ---- sanity checks S1-S4 (structural; S5 = no numeric threshold)
    ep = pre["expected_population"]
    popB = D.get("population_B") or {}
    nums = [B.get(kk) for kk in ("BASE_ALL_P50", "BASE_ANY_P50", "SAFE_ALL_P50", "SAFE_ANY_P50", "BASE_SCOPE_NODES", "SAFE_SCOPE_NODES", "BND_ALL_P50")]
    finite = all(isinstance(x, (int, float)) and math.isfinite(x) for x in nums)
    S = {"S1_downstream_executed": {"status": D["status"], "pass": D["status"] in ("BASELINE_ABSENT", "PAIRED"), "expected": pre["reference_policy"]["expected_downstream_status"]},
         "S2_population": {"eval_split": popB.get("eval_split"), "nq": popB.get("nq"), "row_query_ids_sha256": popB.get("row_query_ids_sha256"), "expected_nq": ep["nq"],
                           "expected_sha256": ep["row_query_ids_sha256"], "expected_split": ep["eval_split"],
                           "pass": popB.get("nq") == ep["nq"] and popB.get("row_query_ids_sha256") == ep["row_query_ids_sha256"] and popB.get("eval_split") == ep["eval_split"]},
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
    R["host_after"] = preflight_sample(None, purpose="run_end")
    R["DECISION"] = "PHG_SCALE_WEBQSP_COMPLETE" if S["ALL_PASS"] else "WEBQSP_L1_EXECUTION_FAIL"
    R["claim"] = ("scale / absolute-L1 result: the chain completed locally under the frozen experimental PHG contract on the exact canonical WebQSP H4_SK; absolute L1 numbers "
                  "on the train_holdout population recorded; no canonical Mt-KaHyPar reference exists (no equivalence claim, no comparative quality claim); the legacy "
                  "NAME + verbalised-facts webqsp cells are historical context only and are not compared numerically")
    wj(fp, R)
    log("%s DECISION %s  BASE_ALL %.4f  SAFE_ALL %.4f  BASE_ANY %.4f  SAFE_ANY %.4f  scope %.0f/%.0f  ELIGIBLE %s  nq %s  status %s  sanity %s -> %s" % (
        DS, R["DECISION"], B["BASE_ALL_P50"], B["SAFE_ALL_P50"], B["BASE_ANY_P50"], B["SAFE_ANY_P50"], B["BASE_SCOPE_NODES"], B["SAFE_SCOPE_NODES"],
        (B.get("BALANCE") or {}).get("ELIGIBLE"), popB.get("nq"), D["status"], S["ALL_PASS"], os.path.relpath(fp, REPO)))
    return R


def not_run():
    fp = os.path.join(OUT, RUNS_W)
    if os.path.exists(fp):
        raise RuntimeError("%s exists" % os.path.relpath(fp, REPO))
    st = rj(WAIT_STATE)
    if st is None or time.time() < float(st["deadline_epoch"]):
        raise RuntimeError("the wait budget has not elapsed (deadline %s); not_run is only for an exhausted budget" % (st or {}).get("deadline_utc"))
    samples = []
    if os.path.exists(PREFLIGHT_LOG):
        for ln in io.open(PREFLIGHT_LOG, encoding="utf-8"):
            if ln.strip():
                samples.append(json.loads(ln))
    R = {"RECORD": "PHG_WEBQSP_RUNS", "classification": CLASS, "dataset": DS, "utc": utc(), "DECISION": "WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED",
         "authorization": pin(os.path.join(OUT, RULING_W)), "preregistration": pin(os.path.join(OUT, PREREG_W)), "program": pin(THIS), "wait_state": st,
         "preflight": {"criterion": PREFLIGHT, "history_summary": preflight_summary(samples), "log": pin(PREFLIGHT_LOG)},
         "foreign_process": "pid %d never signalled; nothing killed" % FOREIGN_PID, "launched": "nothing (no mpirun, no PHG dump, no partition, no replay cache)",
         "claim": "no WebQSP result exists; this is a host-resource outcome, not evidence about PHG", "host_after": preflight_sample(None, purpose="not_run")}
    wj(fp, R)
    log("%s DECISION %s (%d preflight samples, %d clean) -> %s" % (DS, R["DECISION"], R["preflight"]["history_summary"]["n"], R["preflight"]["history_summary"]["clean"], os.path.relpath(fp, REPO)))
    return R


# ----------------------------------------------------------------------------- report
def report():
    pre = rj(os.path.join(OUT, PREREG_W))
    R = rj(os.path.join(OUT, RUNS_W))
    if R is None:
        raise RuntimeError("no WebQSP run record")
    rp = os.path.join(OUT, REPORT_W + ".json")
    moved = {fn: PR.supersede(fn) for fn in (REPORT_W + ".json", REPORT_W + ".md")} if os.path.exists(rp) else {}
    rul = rj(os.path.join(OUT, RULING_W)) or {}
    integ = {"frozen_records_changed": [fn for fn, pn in (pre or {}).get("frozen_read_only", {}).items() if not os.path.exists(os.path.join(OUT, fn)) or sha_file(os.path.join(OUT, fn)) != pn["sha256"]],
             "data_pins_changed": [kk for kk, pn in ((pre or {}).get("inputs") or {}).get("data_pins", {}).items() if pn and (not os.path.exists(os.path.join(REPO, pn["path"])) or sha_file(os.path.join(REPO, pn["path"])) != pn["sha256"])],
             "musique_files_changed": [kk for kk, pn in (rul.get("musique_frozen_unchanged") or {}).get("data_files", {}).items() if pn and (not os.path.exists(os.path.join(REPO, pn["path"])) or sha_file(os.path.join(REPO, pn["path"])) != pn["sha256"])],
             "canonical_webqsp_partition_still_absent": not os.path.exists(os.path.join(CANON, "parts", "H4_SK.npy")),
             "canonical_webqsp_replay_still_absent": not os.path.exists(os.path.join(CANON_L1, "L1_REPLAY_%s.json" % DS)),
             "experimental_files_written": sorted(os.path.relpath(p, REPO).replace("\\", "/") for p in [
                 os.path.join(CANON, "parts", "%s.npy" % TAG_W), os.path.join(CANON, "parts", "%s.json" % TAG_W), os.path.join(ds_dir(DS), "replay_cache__%s.npz" % TAG_W),
                 os.path.join(ds_dir(DS), "replay_cache__%s.json" % TAG_W), os.path.join(ds_dir(DS), "phg_repair1", "repaired.npy")] if os.path.exists(p)) + (
                 ["data/l1_lowmem/%s/phg/{G,R1,R1_repeat,STAGE*.json}" % DS] if os.path.isdir(os.path.join(ds_dir(DS), "phg")) else []),
             "input_chain_files_built": ["data/l1_canonical/webqsp/hypergraph/H4_SK.{npz,json}", "data/l1_lowmem/webqsp/H4_SK.hgr(+.json)", "data/l1_lowmem/webqsp/stream/*",
                                         "data/l1_lowmem/webqsp/H4_SK.official.netl(+.json,.time.txt)", "data/l1_lowmem/webqsp/H4_SK.semantic_gate.json", "data/l1_lowmem/webqsp/freight/stream_manifest.txt"],
             "superseded_to_history": moved, "foreign_process_untouched": True}
    s1, g, s3, s3b, dsn = (R.get("stages") or {}).get("1_input_verified") or {}, (R.get("stages") or {}).get("2_structure_gate") or {}, (R.get("stages") or {}).get("2_partition") or {}, (R.get("stages") or {}).get("3_repair") or {}, R.get("downstream") or {}
    B = dsn.get("B_phg") or {}
    T = R.get("timing_seconds") or {}
    gm = (R.get("runs") or {}).get("G") or {}
    dc = R.get("diagnostic_context") or {}
    stage_rows = ["| Stage | Completed | Wall s | Memory | Note |", "|---|---|---|---|---|"]
    if s1:
        stage_rows.append("| 1 H4_SK input verified | True | %s | py peak %s MB | N %s M %s P %s k %s; %d shards x %s nodes; digest recomputed == ORIGINAL |" % (
            T.get("1_input_verified"), s1.get("py_peak_mb"), s1.get("N"), s1.get("M"), s1.get("P"), s1.get("k"), s1.get("shard_count") or 0, s1.get("shard_nodes")))
    if g:
        stage_rows.append("| 2a exact structure gate (gate-mode PHG run G) | %s | %s (job %s) | peak RSS/rank %s MB | weight reception rel diff %s; removal/warning lines %s |" % (
            g.get("gate"), T.get("2_structure_gate"), (gm.get("timing") or {}).get("job_wall_seconds"), [round(x / 1024.0, 1) for x in (gm.get("memory") or {}).get("peak_rss_kb_per_rank_time_v", [])],
            (g.get("weight_reception") or {}).get("relative_diff"), len(g.get("zoltan_removed_or_warning_lines") or []) or "none"))
    if s3:
        stage_rows.append("| 2b PHG partition R1 (NP=%d) | True | %s (partition %.2f, job %s) | peak RSS/rank %s MB (max %.0f, sum %.0f) | raw validity %s: max %d (bound %d), empty %d; Zoltan cutl %.0f imb %.4f; Python KM1 vs Zoltan cutl rel diff %s; repeat identical %s |" % (
            P.NP, T.get("2_partition_R1"), s3["timing"]["partition_wall_seconds"], s3["timing"]["job_wall_seconds"], [round(x / 1024.0, 1) for x in s3["memory"]["peak_rss_kb_per_rank_time_v"]],
            s3["memory"]["max_rank_peak_kb"] / 1024.0, s3["memory"]["sum_of_rank_peaks_kb"] / 1024.0, s3["validity_raw"]["gate"], s3["validity_raw"]["max_block"],
            s3["validity_raw"]["contract_bound_ceil_1.03_N_over_k"], s3["validity_raw"]["empty_blocks"], s3["zoltan_eval"]["cutl_global"], s3["zoltan_eval"]["imbalance"],
            s3["python_km1_vs_zoltan_cutl"]["relative_diff"], (R.get("repeat") or {}).get("identical_partition")))
    if s3b:
        zero = sum(1 for mv in s3b["moves"] if mv["delta_km1_weighted"] == 0)
        stage_rows.append("| 3 %s | %s | %s | py peak %s MB | empty blocks %d -> %d move(s), %d with delta 0, delta-KM1 total %+d; KM1 %d -> %d; STRUCT cut delta %s; KNN cut delta %s; H4_SK unchanged %s; validity after %s |" % (
            PR.REPAIR, "PASS" if s3b["postconditions"]["PASS"] else "FAIL", T.get("3_repair"), s3b.get("py_peak_mb"), len(s3b["empty_blocks"]), len(s3b["moves"]), zero, s3b["after"]["delta_km1_total"],
            s3b["before"]["km1_weighted"], s3b["after"]["km1_weighted"], s3b["after"]["struct_cut_delta"], s3b["after"]["knn_cut_delta"], s3b["postconditions"]["H4_SK_unchanged"], s3b["after"]["validity"]["gate"]))
    if dsn:
        c = dsn.get("cache") or {}
        stage_rows.append("| 4 import + P50 replay cache + canonical L1 | %s | %s (eval %s) | builder predicted peak %s GB (fp16 memory-mapped %s); py peak %s MB | cache %s bytes; BASE %s s / STRUCT %s s |" % (
            dsn.get("status"), T.get("4_import_replay_cache_L1"), dsn.get("eval_seconds"), c.get("predicted_peak_gb"), c.get("fp16_path"), dsn.get("py_peak_mb"), c.get("bytes"),
            (c.get("COST") or {}).get("BASE_sec"), (c.get("COST") or {}).get("STRUCT_sec")))
    m = R.get("metrics")
    part_rows = ["| Arm | KM1 (weighted) | cut nets | STRUCT cut | KNN cut | blocks min/p50/p95/max (bound) | empty | peak RSS | wall | status |", "|---|---|---|---|---|---|---|---|---|---|"]
    if m and s3:
        part_rows.append("| PHG C1 (R1 + %s, NP=%d) | %d | %d / %d | %.4f | %.4f | %d/%.0f/%.0f/%d (%d) | %d | %.0f MB max rank / %.0f MB sum | %.2f s partition / %s s job | %s |" % (
            PR.REPAIR, P.NP, m["km1_weighted"], m["cut_nets"], m["nets"], m["family_cuts"]["STRUCT"]["edge_cut_fraction"], m["family_cuts"]["KNN"]["edge_cut_fraction"], m["blocks"]["min"], m["blocks"]["p50"],
            m["blocks"]["p95"], m["blocks"]["max"], (s3b.get("after") or {}).get("validity", {}).get("contract_bound_ceil_1.03_N_over_k", 0), m["blocks"]["empty"],
            s3["memory"]["max_rank_peak_kb"] / 1024.0, s3["memory"]["sum_of_rank_peaks_kb"] / 1024.0, s3["timing"]["partition_wall_seconds"], s3["timing"]["job_wall_seconds"], R["DECISION"]))
    mt = R.get("mtkahypar_reference") or {}
    if mt:
        me = mt.get("memory_expectation_frozen_recipe") or {}
        part_rows.append("| Mt-KaHyPar (frozen recipe, canonical webqsp) | - | - | - | - | - | - | expected %s GB (linear in pins) / %s GB (pins x k) | not run | ABSENT (%s) |" % (
            me.get("linear_in_pins_GB"), me.get("pins_times_k_GB"), "NOT_FEASIBLE_AT_250GB"))
    l1_rows = ["| Arm | nq | BASE_ALL | SAFE_ALL | BASE_ANY | SAFE_ANY | scope BASE / SAFE | SAFE additions (+/-) | BND_ALL | balance max/mean | ELIGIBLE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    if B:
        l1_rows.append("| PHG C1 (absolute, canonical NAME_ONLY, train_holdout) | %s | %.4f | %.4f | %.4f | %.4f | %.1f / %.1f | +%d/-%d | %.4f | %s | %s |" % (
            (dsn.get("population") or {}).get("nq"), B["BASE_ALL_P50"], B["SAFE_ALL_P50"], B["BASE_ANY_P50"], B["SAFE_ANY_P50"], B["BASE_SCOPE_NODES"], B["SAFE_SCOPE_NODES"],
            B["CORR"]["gained"], B["CORR"]["lost"], B["BND_ALL_P50"], B["BALANCE"]["max_over_mean"], B["BALANCE"]["ELIGIBLE"]))
    hop_rows = ["| hop | n | BASE_ALL | SAFE_ALL |", "|---|---|---|---|"] + ["| %s | %d | %.4f | %.4f |" % (h, x["n"], x["BASE_ALL"], x["SAFE_ALL"]) for h, x in (dsn.get("by_hop") or {}).items()]
    legacy = ((rj(os.path.join(REPO, "results", "L1", "L1_LOCKED_MANIFEST.json")) or {}).get("datasets") or {}).get(DS) or {}
    hist = {"policy": "historical context only -- NOT a baseline, NOT a reproduction target, never compared numerically with the canonical NAME_ONLY result",
            "legacy_corpus": "NAME + verbalised facts encoding, %s docs (canonical NAME_ONLY: %s nodes), legacy partition rule (variant C, npart %s; canonical k %s), population %s x %s queries (canonical: train_holdout, test barred)" % (
                legacy.get("N_docs"), (R.get("inputs") or {}).get("N"), legacy.get("npart_C"), (R.get("inputs") or {}).get("k"), legacy.get("eval_population"), legacy.get("N_eval")),
            "legacy_operating_point_C_dense+splade_K100_P50": ((legacy.get("L1_operating_point") or {}).get("C_dense+splade_K100_P50")),
            "legacy_status": legacy.get("C_STATUS"), "legacy_mtkahypar_SKN_resource_point": "10,900,573 pins, k 7,814, peak RSS 140.6 GB (EXTERNAL_LANE_MEMORY_EXPECTATION.json legacy_points)",
            "record": pin(os.path.join(REPO, "results", "L1", "L1_LOCKED_MANIFEST.json"))}
    S = R.get("sanity_checks") or {}
    pfh = R.get("preflight") or {}
    launches = {kk: (preflight_summary(v) if isinstance(v, list) else v) for kk, v in pfh.items() if kk != "criterion"}
    rep = {"RECORD": "PHG_WEBQSP_REPORT", "classification": CLASS, "dataset": DS, "lane": LANE, "encoding": ENCODING, "arm": RL.EXPERIMENTAL_PHG_CONTRACT, "utc": utc(),
           "supersedes": moved or None, "authorization": pin(os.path.join(OUT, RULING_W)), "preregistration": pin(os.path.join(OUT, PREREG_W)), "run_record": pin(os.path.join(OUT, RUNS_W)),
           "STATUS": "STOP_FOR_REVIEW", "DECISION": R["DECISION"], "claim": R.get("claim"), "question": QUESTION, "sanity_checks": S, "timing_seconds": T,
           "preflight_at_launches": launches,
           "structure_gate": {kk: g.get(kk) for kk in ("gate", "N_returned", "M_with_pins", "P_returned", "weight_sum", "weight_min", "weight_max", "digest_from_queries", "ORIGINAL_STRUCTURE_SHA256",
                                                       "partition_run_dumps_identical", "zoltan_removed_or_warning_lines", "weight_reception", "problems", "required")},
           "partition": {"metrics_final": m, "metrics_raw": R.get("metrics_raw"), "raw_validity": s3.get("validity_raw"), "zoltan_eval": s3.get("zoltan_eval"),
                         "repair": {kk: s3b.get(kk) for kk in ("noop", "empty_blocks", "moves", "postconditions", "after")}, "memory": s3.get("memory"), "timing": s3.get("timing"),
                         "repeat": R.get("repeat"), "python_km1_vs_zoltan_cutl": s3.get("python_km1_vs_zoltan_cutl")},
           "l1_absolute": B, "population": dsn.get("population"), "by_hop": dsn.get("by_hop"), "downstream_status": dsn.get("status"), "paired": dsn.get("paired"),
           "reference": {"canonical_mtkahypar": mt or {"status": "ABSENT"}, "classification": "scale / absolute-L1 result; no equivalence claim", "historical_context_not_a_baseline": hist,
                         "diagnostic_context": dc},
           "failure_evidence": R.get("failure_evidence"), "error": R.get("error"), "stopped_at_stage": R.get("stopped_at_stage"),
           "host": {"before": (pfh.get("before_stage1") or [None])[-1] if isinstance(pfh.get("before_stage1"), list) else pfh.get("before_stage1"), "after": R.get("host_after")},
           "integrity": integ, "stage_table_markdown": "\n".join(stage_rows), "partition_table_markdown": "\n".join(part_rows), "l1_table_markdown": "\n".join(l1_rows),
           "hop_table_markdown": "\n".join(hop_rows),
           "not_done": "HotpotQA / 2Wiki / the 302M Freebase lane not run (unauthorized); no promotion; no tuning; NP unchanged; MuSiQue not rerun; the canonical webqsp partition slot stays empty; "
                       "FREIGHT stays FREIGHT_CLOSED; the squad / metaqa / musique PHG records untouched; the foreign process untouched"}
    wj(rp, rep)
    md = ["# %s -- webqsp (%s; %s) under the experimental PHG contract (%s)" % (CLASS, LANE, ENCODING.split(" (")[0], RL.EXPERIMENTAL_PHG_CONTRACT), "",
          "Generated %s -- STATUS: **STOP_FOR_REVIEW** -- DECISION: **%s**" % (rep["utc"], R["DECISION"]), "",
          "Question (preregistered): %s" % QUESTION, "",
          "Reference policy: no canonical Mt-KaHyPar webqsp reference exists (the frozen recipe is NOT_FEASIBLE_AT_250GB: %s GB linear in pins / %s GB pins x k), so this is a "
          "scale / absolute-L1 result -- no equivalence claim, no comparative quality claim.  The legacy webqsp cells (NAME + verbalised facts corpus) are historical context only "
          "and are not compared numerically." % ((mt.get("memory_expectation_frozen_recipe") or {}).get("linear_in_pins_GB"), (mt.get("memory_expectation_frozen_recipe") or {}).get("pins_times_k_GB")), ""]
    if R["DECISION"] == "WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED":
        hs = (R.get("preflight") or {}).get("history_summary") or {}
        md += ["The host never met the preregistered clean-host criterion within the wait budget (%s samples, %s clean, host available %s GB, pages in/s %s); nothing was launched and "
               "nothing was killed.  This is a host-resource outcome, not evidence about PHG." % (hs.get("n"), hs.get("clean"), hs.get("host_available_gb_min_max"), hs.get("pages_input_per_sec_mean_min_max")), ""]
    md += ["Preflight immediately before each launch (criterion: host available >= %.1f GB, Pages Input/sec <= %.0f, WSL available >= %d MB, disk >= %.1f GB, swap not growing; 3 consecutive samples):" % (
               PREFLIGHT["host_available_gb_min"], PREFLIGHT["pages_input_per_sec_max"], PREFLIGHT["wsl_available_mb_min"], PREFLIGHT["disk_free_gb_min"]), ""]
    for kk, v in launches.items():
        last = (v or {}).get("last") if isinstance(v, dict) else None
        md.append("- %s: %s" % (kk, one_line(last) if last else json.dumps(v)))
    md += ["", "Stages:", "", rep["stage_table_markdown"], "", "Partition (diagnostics; no canonical Mt-KaHyPar value exists to compare the KM1 with):", "", rep["partition_table_markdown"], "",
           "Canonical L1 on the frozen train_holdout population (absolute; canonical NAME_ONLY encoding; test never read):", "", rep["l1_table_markdown"], ""]
    if dsn.get("by_hop"):
        md += ["Hop-wise (absolute, diagnostic):", "", rep["hop_table_markdown"], ""]
    elif dsn:
        md += ["Hop-wise: the canonical webqsp replay carries no hop labels (by_hop absent) -- not defined for this population.", ""]
    if S:
        md += ["Sanity checks (structural, preregistered; no numeric retrieval threshold): " + "; ".join("%s = %s" % (kk, S[kk].get("pass")) for kk in (
            "S1_downstream_executed", "S2_population", "S3_canonical_eligibility", "S4_definitional_consistency", "S5_no_numeric_retrieval_threshold") if kk in S) + "; ALL_PASS = %s" % S.get("ALL_PASS"), ""]
    if dc:
        br = dc.get("blocker_resources") or {}
        md += ["Resources (the scale question): PHG's largest rank peaked at %s MB (%s MB summed over %d ranks, an upper bound) and the partition took %s s (job %s s); the frozen Mt-KaHyPar recipe "
               "was expected to need %s GB (linear in pins) to %s GB (pins x k) for this corpus and was never run." % (
                   br.get("phg_peak_rss_mb_max_rank"), br.get("phg_peak_rss_mb_sum_ranks_upper_bound"), P.NP, br.get("phg_partition_wall_seconds"), br.get("phg_job_wall_seconds"),
                   br.get("mtkahypar_expectation_GB_linear_in_pins"), br.get("mtkahypar_expectation_GB_pins_times_k")), ""]
    if R.get("error"):
        md += ["Stopped at stage %s: %s" % (R.get("stopped_at_stage"), str(R.get("error"))[:600]), ""]
    md += ["Historical context (NOT a baseline; different corpus encoding, N, k, family set and population; never compared numerically): legacy webqsp %s, %s docs, npart %s, %s x %s queries, "
           "operating point C_dense+splade_K100_P50 = %s; legacy Mt-KaHyPar SKN resource point %s." % (
               hist.get("legacy_status"), legacy.get("N_docs"), legacy.get("npart_C"), legacy.get("eval_population"), legacy.get("N_eval"), json.dumps(hist.get("legacy_operating_point_C_dense+splade_K100_P50")),
               hist.get("legacy_mtkahypar_SKN_resource_point")), "",
           "Host: before %s; after %s." % (json.dumps(rep["host"]["before"]), json.dumps(rep["host"]["after"])), "",
           "Integrity: frozen records changed = %s; input/data pins changed = %s; musique files changed = %s; canonical webqsp partition still absent = %s; canonical webqsp replay still absent = %s; "
           "experimental files written = %s." % (integ["frozen_records_changed"] or "none", integ["data_pins_changed"] or "none", integ["musique_files_changed"] or "none",
                                                integ["canonical_webqsp_partition_still_absent"], integ["canonical_webqsp_replay_still_absent"], integ["experimental_files_written"]), "",
           "Caveats: 4 local MPI ranks inside one WSL VM validate feasibility on this host only (no aggregate-RAM claim at larger scale); the rank count is a contract element and was not changed; "
           "%s is part of the contract (no-op when no block is empty; no minimum-part-size condition exists); the preflight is execution hygiene, not a PHG parameter." % PR.REPAIR, "",
           rep["not_done"], ""]
    with io.open(os.path.join(OUT, REPORT_W + ".md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    log("STOP_FOR_REVIEW", DS, R["DECISION"])
    print(rep["stage_table_markdown"]); print(); print(rep["partition_table_markdown"]); print(); print(rep["l1_table_markdown"])
    return rep


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "ruling":
        ruling()
    elif a[0] == "preflight":
        sys.exit(preflight_cmd())
    elif a[0] == "arm":
        arm()
    elif a[0] == "deadline_passed":
        sys.exit(0 if time.time() > deadline() else 1)
    elif a[0] == "inputs":
        inputs()
    elif a[0] == "prereg":
        prereg()
    elif a[0] == "run":
        run()
    elif a[0] == "not_run":
        not_run()
    elif a[0] == "report":
        report()
    else:
        print(__doc__)
