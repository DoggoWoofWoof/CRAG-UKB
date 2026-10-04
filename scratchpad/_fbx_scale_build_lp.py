"""FBX_SCALE stage 3b -- build the R3 (LDG init + balanced label propagation) maps at scale and measure the build (systems validation; nothing is tuned here).

A sibling of _fbx_scale_build.py (kept unchanged so its records keep the code hash they were written with); it reuses that module's stream readers.

  python scratchpad/_fbx_scale_build_lp.py webqsp   <tag> [--K=100,250]  [--sweeps=10]    # regression on WebQSP: must reproduce CALIB_MAPS_WEBQSP__lp_v1's LP_S map hashes
  python scratchpad/_fbx_scale_build_lp.py freebase <tag> [--K=100,1000,...] [--sweeps=10] # data/final_canonical/freebase, read-only, memory-mapped

Per K it records, for the LDG init and for the LP refinement separately and summed: wall seconds, peak RSS of the C program (memory-mapped adjacency pages included), anonymous bytes,
sweeps run and vertices moved per sweep, block sizes and skew, cut fraction, closed-neighbourhood connectivity, output bytes and sha256.  Maps go to work/FBX_SCALE/<tag>/ (outside
data/final_canonical, which is never written); the record is results/FREEBASE_SCALE/FBX_BUILD__<dataset>__LP_S__<tag>.json (write-once; a per-K partial is kept beside the maps so a
failed run resumes).
"""
import io
import json
import os
import sys
import time

import numpy as np

import _fbx_part as P
import _fbx_lp as LP
import _fbx_scale_build as B

log = B.log


def run(dataset, tag, Ks, sweeps):
    method = "LP_S"
    out_rec = os.path.join(B.OUTR, "FBX_BUILD__%s__%s__%s.json" % (dataset, method, tag))
    assert not os.path.exists(out_rec), "write-once: %s exists" % out_rec
    work = os.path.join(B.ROOT, "work", "FBX_SCALE", "%s_%s_%s" % (dataset, method, tag))
    os.makedirs(work, exist_ok=True)
    N, streams, ginfo = B.freebase_streams() if dataset == "freebase" else B.webqsp_streams(work)
    _, ldg_sha = P.c_binary()
    _, lp_sha = LP.c_binary_lp()
    rec = {"stage": "FBX_SCALE 3b: %s maps on %s" % (method, dataset), "tag": tag, "method": method, "dataset": dataset, "N": N, "graph": ginfo, "K_grid": Ks, "sweeps_max": sweeps,
           "ldg_c_source_sha12": ldg_sha, "lp_c_source_sha12": lp_sha, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "maps": {}}
    for K in Ks:
        part = os.path.join(work, "K%d.json" % K)
        if os.path.exists(part):
            rec["maps"][str(K)] = json.load(io.open(part, encoding="utf-8"))
            log("K %d: resumed from its partial record" % K)
            continue
        npy = os.path.join(work, "%s_k%d.npy" % (method, K))
        init_raw, out_raw = npy[:-4] + ".init.raw", npy[:-4] + ".raw"
        t0 = time.time()
        lab, li = P.ldg_c(N, K, streams, init_raw)
        del lab
        log("K %d: LDG %.1fs (%d least-loaded fallbacks)" % (K, li["seconds"], li["least_loaded_fallbacks"]))
        lab, pi = LP.lp_c(N, K, sweeps, init_raw, out_raw, streams)
        np.save(npy, lab)
        del lab
        os.remove(init_raw)
        os.remove(out_raw)
        wall = round(time.time() - t0, 3)
        log("K %d: LP %.1fs, %d sweeps, moved %s" % (K, pi["seconds"], pi["sweeps_run"], pi["moved_per_sweep"]))
        t1 = time.time()
        st = P.stats_c(N, K, npy, streams)
        log("K %d: stats (%.1fs): cut %.4f, closed-nbhd blocks mean %.3f, skew %.4f" % (K, time.time() - t1, st["cut_fraction"], st["closed_nbhd_distinct_blocks_mean"], st["load_skew_max_over_mean"]))
        build = {"seconds_ldg_pass": li["seconds"], "seconds_lp": pi["seconds"], "seconds_build_total_pass": round(li["seconds"] + pi["seconds"], 3), "seconds_wall_incl_io": wall,
                 "ldg": {"cap": li["cap"], "least_loaded_fallbacks": li["least_loaded_fallbacks"], "c_peak_rss_kb": li["peak_rss_kb"], "c_anon_bytes": li["anon_bytes"]},
                 "lp": {"cap": pi["cap"], "lo": pi["lo"], "sweeps_run": pi["sweeps_run"], "moved_per_sweep": pi["moved_per_sweep"], "seconds_per_sweep": pi["seconds_per_sweep"],
                        "c_peak_rss_kb": pi["peak_rss_kb"], "c_anon_bytes": pi["anon_bytes"]}}
        ent = {"K": K, "build": build, "stats": st, "map_file": os.path.relpath(npy, B.ROOT).replace("\\", "/"), "map_bytes": os.path.getsize(npy), "map_sha256": B.sha_file(npy)}
        with io.open(part, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(ent, indent=1, ensure_ascii=False))
        rec["maps"][str(K)] = ent
    rec["seconds_total"] = round(time.time() - B.T0, 1)
    rec["python_peak_rss_mb"] = B.peak_rss_mb()
    here = os.path.dirname(os.path.abspath(__file__))
    rec["code"] = {n: B.sha_file(os.path.join(here, n)) for n in ("_fbx_scale_build_lp.py", "_fbx_scale_build.py", "_fbx_part.py", "_fbx_lp.py", "_fbx_ldg.c", "_fbx_lp.c")}
    with io.open(out_rec, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    log("record -> %s %s" % (out_rec, B.sha_file(out_rec)[:12]))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    ks = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--K=")), None)
    sw = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--sweeps=")), 10))
    if ks:
        Ks = [int(x) for x in ks.split(",")]
    else:
        Ks = [100, 1000, 2000, 5000, 10000, 20000] if sys.argv[1] == "freebase" else [100, 250, 500, 1000, 2000, 5000, 25928]
    run(sys.argv[1], sys.argv[2], Ks, sw)
