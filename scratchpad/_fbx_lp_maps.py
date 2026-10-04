"""FBX_SCALE stage 3a, rung 3 (addendum 2): build the R3 balanced-LP maps on WebQSP from the R2 LDG maps the MAPS run already wrote.

  python scratchpad/_fbx_lp_maps.py <tag> [--sweeps=10]

Reads work/FBX_CALIB/maps/webqsp__LDG_{S,SK}__k<K>.npy and the raw CSR streams work/FBX_CALIB/csr/webqsp_{S,SK}.* the MAPS run wrote (both are pinned by
CALIB_MAPS_WEBQSP__v1.json), runs _fbx_lp.c on each (LP_S from LDG_S on graph S; LP_SK from LDG_SK on graph SK) at every K of the grid, and records per K:
seconds, sweeps run, vertices moved per sweep, block sizes, peak RSS and the map statistics on both graphs.  Writes work/FBX_CALIB/maps/webqsp__LP_*__k<K>.npy
and the write-once record results/FREEBASE_SCALE/CALIB_MAPS_WEBQSP__<tag>.json.  No recall is computed here.
"""
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _fbx_calib as C
import _fbx_part as P
import _fbx_lp as LP

log = D.log


def main():
    tag = sys.argv[1]
    sweeps = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--sweeps=")), 10))
    rec_path = os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % tag)
    assert not os.path.exists(rec_path), "write-once: %s exists" % rec_path
    prev = C.jl(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__v1.json"))
    N = int(prev["N"])
    streams = {g: [P.wsl_path(os.path.join(C.WORK, "csr", "webqsp_%s.xadj.i64" % g)), P.wsl_path(os.path.join(C.WORK, "csr", "webqsp_%s.adj.i32" % g))] for g in ("S", "SK")}
    _, c_sha = LP.c_binary_lp()
    rec = {"stage": "FBX_SCALE 3a rung 3: R3 balanced LP on WebQSP (LDG init)", "N": N, "K_grid": list(C.KGRID), "sweeps_max": sweeps, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "lp_c_source_sha12": c_sha, "init_maps_record": {"CALIB_MAPS_WEBQSP__v1.json": D.sha_file(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__v1.json"))}, "maps": {}}
    tmp = os.path.join(C.WORK, "maps_lp_tmp")
    os.makedirs(tmp, exist_ok=True)
    for K in C.KGRID:
        for g in ("S", "SK"):
            src, dst = "LDG_%s" % g, "LP_%s" % g
            init = np.load(C.map_path(src, K)).astype(np.int32)
            assert D.sha_file(C.map_path(src, K)) == prev["maps"]["%s__k%d" % (src, K)]["sha256"], "the LDG map differs from its MAPS record"
            init_raw, out_raw = os.path.join(tmp, "init.raw"), os.path.join(tmp, "out.raw")
            init.tofile(init_raw)
            lab, info = LP.lp_c(N, K, sweeps, init_raw, out_raw, streams[g])
            np.save(C.map_path(dst, K), lab)
            st = {gg: P.stats_c(N, K, C.map_path(dst, K), streams[gg]) for gg in ("S", "SK")}
            st0 = prev["maps"]["%s__k%d" % (src, K)]["stats"]
            rec["maps"]["%s__k%d" % (dst, K)] = {"method": "R3 balanced LP (LDG init) on graph %s (C)" % g, "K": K, "seconds": info["seconds"], "sweeps_run": info["sweeps_run"],
                                                 "moved_per_sweep": info["moved_per_sweep"], "seconds_per_sweep": info["seconds_per_sweep"], "cap": info["cap"], "lo": info["lo"],
                                                 "c_peak_rss_kb": info["peak_rss_kb"], "c_anon_bytes": info["anon_bytes"], "sha256": D.sha_file(C.map_path(dst, K)), "stats": st,
                                                 "init_stats": st0}
            log("K %5d %-5s: %d sweeps (%.1fs), moved %s, cut %.4f -> %.4f (on S: %.4f -> %.4f), closed-nbhd blocks %.3f -> %.3f, sizes [%d, %d]" % (
                K, dst, info["sweeps_run"], info["seconds"], info["moved_per_sweep"], st0[g]["cut_fraction"], st[g]["cut_fraction"], st0["S"]["cut_fraction"], st["S"]["cut_fraction"],
                st0[g]["closed_nbhd_distinct_blocks_mean"], st[g]["closed_nbhd_distinct_blocks_mean"], st[g]["size_min"], st[g]["size_max"]))
    rec["code"] = {"scratchpad/_fbx_lp_maps.py": D.sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_lp.py": D.sha_file(os.path.join(D.HERE, "_fbx_lp.py")),
                   "scratchpad/_fbx_lp.c": D.sha_file(os.path.join(D.HERE, "_fbx_lp.c")), "scratchpad/_fbx_part.py": D.sha_file(os.path.join(D.HERE, "_fbx_part.py"))}
    C.wj(rec_path, rec)
    log("LP MAPS -> %s %s" % (rec_path, D.sha_file(rec_path)[:12]))


if __name__ == "__main__":
    main()
