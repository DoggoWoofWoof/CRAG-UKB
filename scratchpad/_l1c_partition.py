"""L1_COVPART step 2: partition a scratch hypergraph with the FROZEN Mt-KaHyPar contract (scratchpad/_l1hu_local_worker.py,
DETERMINISTIC_QUALITY, KM1, eps 0.03, seed 0, unit vertex weights, hyperedge weights from the build), under the same RSS guard
as src/l1_canonical/partition.py, gated on host memory so the foreign processes are never squeezed:

    python -u _l1c_partition.py <ds> <tag> [--gate-gb 2.0] [--cap-gb 2.0] [--threads 4] [--budget-h 3]

Waits (60 s samples) until host available memory >= gate for 3 consecutive samples, then runs the worker with the RSS cap; a
worker killed by the guard or a budget expiry is recorded (parts/<ds>__<tag>.RUN.json) and leaves no partition behind.
DETERMINISTIC_QUALITY is thread-count independent (LOCAL_COMPUTE.json), so --threads only changes wall time.
Outputs: results/L1_COVPART/parts/<ds>__<tag>.npy / .stats.json / .worker.log / .RUN.json.  Nothing under data/ is written.
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1s_core as S
from src.l1_canonical import partition as PT  # noqa: E402  (imported, never edited)
from src.l1_canonical.adapter import sha_file  # noqa: E402

PDIR = os.path.join(S.X.REPO, "results", "L1_COVPART", "parts")
a = sys.argv[1:]
ds, tag = a[0], a[1]
opts = {"--gate-gb": 2.0, "--cap-gb": 2.0, "--threads": 4, "--budget-h": 3.0}
for i in range(2, len(a), 2):
    opts[a[i]] = float(a[i + 1])
gate, cap, threads, budget = opts["--gate-gb"], opts["--cap-gb"], int(opts["--threads"]), opts["--budget-h"]
src = os.path.join(PDIR, "%s__%s.npz" % (ds, tag))
out_npy = os.path.join(PDIR, "%s__%s.npy" % (ds, tag))
stats = os.path.join(PDIR, "%s__%s.stats.json" % (ds, tag))
logp = os.path.join(PDIR, "%s__%s.worker.log" % (ds, tag))
runp = os.path.join(PDIR, "%s__%s.RUN.json" % (ds, tag))
meta = json.load(io.open(src[:-4] + ".json", encoding="utf-8"))
assert sha_file(src) == meta["file_sha256"], "hypergraph file changed since its manifest"
if os.path.exists(out_npy):
    S.log("%s exists; nothing to do" % out_npy)
    sys.exit(0)
rec = {"dataset": ds, "tag": tag, "hypergraph": {"file": meta["file"], "sha256": meta["file_sha256"], "content_digest": meta["content_digest"],
                                                 "hyperedges": meta["hypergraph"]["hyperedges"], "pins": meta["hypergraph"]["pins"], "N": meta["N"], "k": meta["k"]},
       "contract": {"partitioner": "mtkahypar", "preset": "DETERMINISTIC_QUALITY", "objective": "KM1", "epsilon": 0.03, "seed": 0, "vertex_weights": "unit",
                    "hyperedge_weights": "H4 build (max(1, rint(1000/(|e|-1))))", "worker": PT.WORKER, "worker_sha256": sha_file(os.path.join(S.X.REPO, PT.WORKER)),
                    "threads": threads, "k": meta["k"]},
       "gate": {"host_available_gb_min": gate, "consecutive_clean_samples": 3, "sample_seconds": 60, "rss_cap_gb": cap, "budget_hours": budget},
       "samples": [], "armed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
t0 = time.time()
streak = 0
while True:
    avail = PT.host_available_gb()
    rec["samples"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host_available_gb": round(avail, 2)})
    streak = streak + 1 if avail >= gate else 0
    S.log("gate: host available %.2f GB (need >= %.2f) streak %d/3" % (avail, gate, streak))
    if streak >= 3:
        break
    if time.time() - t0 > budget * 3600:
        rec["STATUS"] = "NOT_RUN_HOST_MEMORY_CONTENDED"
        S.wj(runp, rec)
        S.log("budget expired without a clean window -> %s" % runp)
        sys.exit(2)
    time.sleep(60)
rec["launched_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
S.log("%s %s: N=%d k=%d hyperedges=%d pins=%d -> worker (%d threads, RSS cap %.1f GB)" % (
    ds, tag, meta["N"], meta["k"], meta["hypergraph"]["hyperedges"], meta["hypergraph"]["pins"], threads, cap))
for p in (out_npy, stats):
    if os.path.exists(p):
        os.remove(p)
g = PT.run_worker(src, out_npy, stats, threads, cap, logp)
rec["guard"] = g
if g.get("killed") or g.get("rc", 1) != 0 or not os.path.exists(out_npy) or not os.path.exists(stats):
    rec["STATUS"] = "FAILED_MEMORY_CAP" if g.get("killed") else "FAILED"
    rec["worker_log_tail"] = (io.open(logp, encoding="utf-8", errors="replace").read()[-3000:] if os.path.exists(logp) else "")
    for p in (out_npy, stats):
        if os.path.exists(p):
            os.remove(p)
    S.wj(runp, rec)
    S.log("%s: %s guard %s -> %s" % (tag, rec["STATUS"], g, runp))
    sys.exit(1)
hard = np.load(out_npy)
st = json.load(io.open(stats, encoding="utf-8"))
k = meta["k"]
sizes = np.bincount(hard, minlength=k)
rec.update({"STATUS": "OK", "worker_stats": st,
            "post_checks": {"n": int(len(hard)), "blocks_used": int((sizes > 0).sum()), "k": k, "size_min": int(sizes.min()), "size_max": int(sizes.max()),
                            "balance_max_over_mean": round(float(sizes.max() / sizes.mean()), 4), "balance_within_eps": bool(sizes.max() <= np.floor(1.03 * np.ceil(len(hard) / k)) + 1)},
            "output": {"file": os.path.relpath(out_npy, S.X.REPO).replace("\\", "/"), "sha256": sha_file(out_npy), "n": int(len(hard))}})
S.wj(runp, rec)
S.log("%s: OK km1=%s cut=%s peak_rss=%.0f MB wall %.0fs blocks %d/%d sizes [%d, %d] -> %s" % (
    tag, st.get("objective_km1"), st.get("objective_cut"), st.get("peak_rss_mb", -1), st.get("wall_seconds", -1), rec["post_checks"]["blocks_used"], k, sizes.min(), sizes.max(), out_npy))
