"""P50 hard partitions over canonical positions -- the frozen Mt-KaHyPar contract, run under WSL.

The partitioner is the UNCHANGED worker scratchpad/_l1hu_local_worker.py (DETERMINISTIC_QUALITY,
KM1, eps 0.03, seed 0, unit vertex weights, hyperedge weights from the H4 build), the one
LOCAL_COMPUTE.json verified bit-identical to the Modal runs.  This driver only (1) hands it the
canonical H4_SK hypergraph, (2) guards host memory -- the worker is killed if its RSS crosses a
cap derived from what the host has free, so the foreign processes are never squeezed -- and (3)
checks and pins the result.

    python src/l1_canonical/partition.py <ds> [ds ...] [--threads 8] [--cap-gb X]

Outputs (data/l1_canonical/<ds>/parts/):  H4_SK.npy  position -> block id (int64, len N)
                                          H4_SK.stats.json   the worker's own stats
                                          H4_SK.json         pins + post-checks (the manifest)
A run that hits the memory cap writes H4_SK.FAILED.json instead and leaves no partition behind:
that corpus then belongs to the external >=250 GB lane (or to a new explicit ruling), never to a
cheaper algorithm.
"""
import io
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.l1_canonical.adapter import CanonicalDataset, contract_hash, sha_file  # noqa: E402

WORKER = "scratchpad/_l1hu_local_worker.py"
DISTRO = "Ubuntu-22.04"
DEFAULT_THREADS = 8
MARGIN_GB = 1.0            # host memory kept free for everything else, always
HARD_CAP_GB = 6.0          # never above the WSL VM default ceiling (7.6 GB) minus headroom
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def wsl_path(p):
    p = os.path.abspath(p).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def host_available_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def wsl_env():
    cmd = ("python3 -c \"import mtkahypar, numpy, sys, platform; print(mtkahypar.__version__); print(numpy.__version__); "
           "print(sys.version.split()[0]); print(platform.platform())\"; pip show mtkahypar 2>/dev/null | grep -i ^version; nproc; free -m | awk 'NR==2{print $2}'")
    r = subprocess.run(["wsl", "-d", DISTRO, "-e", "bash", "-lc", cmd], capture_output=True, text=True, timeout=120)
    lines = [ln.strip() for ln in r.stdout.replace("\0", "").splitlines() if ln.strip()]
    return {"distro": DISTRO, "raw": lines, "returncode": r.returncode}


GUARD_SH = r'''#!/bin/bash
# run the frozen worker under an RSS cap; poll every second; kill -9 on breach
SRC="$1"; OUT="$2"; STATS="$3"; THREADS="$4"; CAP_KB="$5"; LOG="$6"
cd "$7"
python3 scratchpad/_l1hu_local_worker.py "$SRC" "$OUT" "$STATS" "$THREADS" > "$LOG" 2>&1 &
PID=$!
PEAK=0
KILLED=0
while kill -0 $PID 2>/dev/null; do
  RSS=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
  if [ -n "$RSS" ]; then
    if [ "$RSS" -gt "$PEAK" ]; then PEAK=$RSS; fi
    if [ "$RSS" -gt "$CAP_KB" ]; then kill -9 $PID; KILLED=1; fi
  fi
  sleep 1
done
wait $PID
RC=$?
echo "GUARD peak_rss_kb=$PEAK killed=$KILLED rc=$RC"
'''


def run_worker(src_npz, out_npy, stats_json, threads, cap_gb, log_path):
    guard = os.path.join(REPO, "data", "l1_canonical", "_guard.sh")
    with io.open(guard, "w", encoding="utf-8", newline="\n") as f:
        f.write(GUARD_SH)
    args = ["wsl", "-d", DISTRO, "-e", "bash", wsl_path(guard), wsl_path(src_npz), wsl_path(out_npy), wsl_path(stats_json),
            str(threads), str(int(cap_gb * 1e6)), wsl_path(log_path), wsl_path(REPO)]
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    out = r.stdout.replace("\0", "")
    g = {}
    for ln in out.splitlines():
        if ln.startswith("GUARD "):
            for kv in ln.split()[1:]:
                k, v = kv.split("=")
                g[k] = int(v)
    g["wall_seconds"] = round(time.time() - t, 1)
    g["returncode"] = r.returncode
    return g


def post_checks(d, hard, k):
    N = d.n_nodes
    ok = len(hard) == N
    sizes = np.bincount(hard, minlength=k) if ok else np.array([])
    used = int((sizes > 0).sum()) if ok else 0
    target = int(np.ceil(N / k))
    return {"length_ok": bool(ok), "min_block": int(hard.min()) if ok else None, "max_block": int(hard.max()) if ok else None,
            "blocks_used": used, "k": int(k), "every_block_used": bool(used == k and ok and hard.max() == k - 1),
            "size_min": int(sizes.min()) if ok else None, "size_max": int(sizes.max()) if ok else None,
            "size_mean": round(float(sizes.mean()), 3) if ok else None, "ceil_N_over_k": target,
            "balance_max_over_mean": round(float(sizes.max() / sizes.mean()), 4) if ok else None,
            "max_over_allowed_eps003": round(float(sizes.max() / (1.03 * target)), 4) if ok else None,
            "balance_within_eps": bool(ok and sizes.max() <= np.floor(1.03 * target) + 1)}


def partition(name, threads=DEFAULT_THREADS, cap_gb=None):
    d = CanonicalDataset(name)
    hg_npz = os.path.join(d.derived_dir, "hypergraph", "H4_SK.npz")
    hg_json = hg_npz[:-4] + ".json"
    if not (os.path.exists(hg_npz) and os.path.exists(hg_json)):
        raise RuntimeError("%s: build the hypergraph first (src/l1_canonical/hypergraph.py build %s)" % (name, name))
    hg = json.load(io.open(hg_json, encoding="utf-8"))
    if hg["inputs"]["DATASET_json_RECORD_SHA256"] != d.record_sha:
        raise RuntimeError("%s: hypergraph was built from another DATASET.json record" % name)
    if sha_file(hg_npz) != hg["file_sha256"]:
        raise RuntimeError("%s: hypergraph file digest changed since its manifest" % name)
    avail = host_available_gb()
    if cap_gb is None:
        cap_gb = min(HARD_CAP_GB, (avail - MARGIN_GB) if avail is not None else HARD_CAP_GB)
    if cap_gb < 0.5:
        raise RuntimeError("%s: only %.1f GB free on the host; refusing to start the partitioner" % (name, avail))
    pdir = os.path.join(d.derived_dir, "parts")
    os.makedirs(pdir, exist_ok=True)
    out_npy = os.path.join(pdir, "H4_SK.npy")
    stats_json = os.path.join(pdir, "H4_SK.stats.json")
    log_path = os.path.join(pdir, "H4_SK.worker.log")
    for p in (out_npy, stats_json):
        if os.path.exists(p):
            os.remove(p)
    log("%s: N=%s k=%s hyperedges=%s pins=%s | host free %.1f GB -> RSS cap %.2f GB, %d threads"
        % (name, "{:,}".format(hg["N"]), hg["k"], "{:,}".format(hg["hyperedges"]), "{:,}".format(hg["pins"]), avail or -1, cap_gb, threads))
    env = wsl_env()
    g = run_worker(hg_npz, out_npy, stats_json, threads, cap_gb, log_path)
    worker_log = io.open(log_path, encoding="utf-8", errors="replace").read() if os.path.exists(log_path) else ""
    rec = {"dataset": name, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "contract": {"partitioner": "mtkahypar", "preset": "DETERMINISTIC_QUALITY", "objective": "KM1", "epsilon": 0.03, "seed": 0,
                        "vertex_weights": "unit", "hyperedge_weights": "H4 build (max(1, rint(1000/(|e|-1))))",
                        "k": hg["k"], "k_rule": hg["k_rule"], "hypergraph_rule": hg["rule"], "families": hg["families"],
                        "worker": WORKER, "worker_sha256": sha_file(os.path.join(REPO, WORKER)), "threads": threads,
                        "deterministic_preset_note": "DETERMINISTIC_QUALITY is thread-count independent; LOCAL_COMPUTE.json verified local == Modal bit-identical"},
           "wsl": env, "guard": {"rss_cap_gb": cap_gb, "host_available_gb_at_start": avail, "margin_gb": MARGIN_GB, **g},
           "inputs": {"DATASET_json_RECORD_SHA256": d.record_sha, "hypergraph_file": hg["file"], "hypergraph_sha256": hg["file_sha256"],
                      "hypergraph_content_digest": hg["content_digest"], "hyperedges": hg["hyperedges"], "pins": hg["pins"], "N": hg["N"]}}
    if g.get("killed") or not os.path.exists(out_npy) or not os.path.exists(stats_json):
        rec["STATUS"] = "FAILED_MEMORY_CAP" if g.get("killed") else "FAILED"
        rec["worker_log_tail"] = worker_log[-3000:]
        rec["note"] = ("the frozen partitioner needs more than %.2f GB RSS on this host for this corpus; this corpus moves to the external "
                       ">=250 GB lane or to a new explicit partitioning ruling -- the algorithm is not swapped" % cap_gb)
        fp = os.path.join(pdir, "H4_SK.FAILED.json")
        with io.open(fp, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, indent=1))
        for p in (out_npy, stats_json):
            if os.path.exists(p):
                os.remove(p)
        log("%s: %s (guard %s) -> %s" % (name, rec["STATUS"], g, fp))
        return rec
    st = json.load(io.open(stats_json, encoding="utf-8"))
    hard = np.load(out_npy)
    pc = post_checks(d, hard, hg["k"])
    rec.update({"STATUS": "OK" if pc["every_block_used"] and pc["length_ok"] else "BUILT_WITH_WARNINGS",
                "worker_stats": st, "post_checks": pc,
                "output": {"file": os.path.relpath(out_npy, REPO).replace("\\", "/"), "bytes": os.path.getsize(out_npy),
                           "sha256": sha_file(out_npy), "dtype": str(hard.dtype), "n": int(len(hard))},
                "L1_CONTRACT_SHA256": contract_hash()["L1_CONTRACT_SHA256"]})
    fp = os.path.join(pdir, "H4_SK.json")
    with io.open(fp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    failed = os.path.join(pdir, "H4_SK.FAILED.json")
    if os.path.exists(failed):
        os.remove(failed)
    log("%s: %s  km1=%s imbalance=%s  peak_rss=%.0f MB (guard peak %.0f MB)  wall %.0fs  blocks %d/%d  size[%d..%d] max/mean %.4f  sha %s"
        % (name, rec["STATUS"], st.get("objective_km1"), st.get("imbalance"), st.get("peak_rss_mb", -1), g.get("peak_rss_kb", 0) / 1e3,
           st.get("wall_seconds", -1), pc["blocks_used"], pc["k"], pc["size_min"], pc["size_max"], pc["balance_max_over_mean"], rec["output"]["sha256"][:16]))
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    threads, cap = DEFAULT_THREADS, None
    names = []
    i = 0
    while i < len(a):
        if a[i] == "--threads":
            threads = int(a[i + 1]); i += 2
        elif a[i] == "--cap-gb":
            cap = float(a[i + 1]); i += 2
        else:
            names.append(a[i]); i += 1
    if not names:
        print(__doc__)
    for n in names:
        partition(n, threads=threads, cap_gb=cap)
