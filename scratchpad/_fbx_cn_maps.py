"""FBX_SCALE stage 3a, rung 3b (addendum 3): build the R3b closed-neighbourhood-connectivity LP maps on WebQSP from the R3 LP maps.

  python scratchpad/_fbx_cn_maps.py <tag> [--sweeps=10] [--T=4] [--dmax=5000] [--K=100,250,500,1000]

Reads work/FBX_CALIB/maps/webqsp__LP_{S,SK}__k<K>.npy (pinned by CALIB_MAPS_WEBQSP__lp_v1.json) and the raw CSR streams work/FBX_CALIB/csr/webqsp_{S,SK}.*, runs _fbx_cn.c on each
(CN_S from LP_S on graph S; CN_SK from LP_SK on graph SK) and records per K: seconds, sweeps, vertices moved and the KM1 change per sweep, block sizes, pool bytes, peak RSS and the map
statistics on both graphs.  Writes work/FBX_CALIB/maps/webqsp__CN_*__k<K>.npy and the write-once record results/FREEBASE_SCALE/CALIB_MAPS_WEBQSP__<tag>.json.  No recall is computed here.
"""
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _fbx_calib as C
import _fbx_part as P
import _fbx_cn as CN

log = D.log
KS_CN = (100, 250, 500, 1000)          # the gate cells and one diagnostic K; the sparse-histogram scan is O(K) per lookup, so K = 25,928 is not built here


def opt(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--%s=" % name)), default)


def main():
    tag = sys.argv[1]
    sweeps, T, dmax = int(opt("sweeps", 10)), int(opt("T", 4)), int(opt("dmax", 5000))
    Ks = tuple(int(x) for x in opt("K", ",".join(str(k) for k in KS_CN)).split(","))
    rec_path = os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % tag)
    assert not os.path.exists(rec_path), "write-once: %s exists" % rec_path
    lrec_path = os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__%s.json" % C.LP_MAPS_TAG)
    prev = C.jl(lrec_path)
    N = int(C.jl(os.path.join(C.OUTR, "CALIB_MAPS_WEBQSP__v1.json"))["N"])
    streams = {g: [P.wsl_path(os.path.join(C.WORK, "csr", "webqsp_%s.xadj.i64" % g)), P.wsl_path(os.path.join(C.WORK, "csr", "webqsp_%s.adj.i32" % g))] for g in ("S", "SK")}
    _, c_sha = CN.c_binary_cn()
    rec = {"stage": "FBX_SCALE 3a rung 3b: R3b closed-neighbourhood LP on WebQSP (LP init)", "N": N, "K_grid": list(Ks), "sweeps_max": sweeps, "T": T, "dmax": dmax,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cn_c_source_sha12": c_sha, "init_maps_record": {os.path.basename(lrec_path): D.sha_file(lrec_path)}, "maps": {}}
    tmp = os.path.join(C.WORK, "maps_cn_tmp")
    os.makedirs(tmp, exist_ok=True)
    for K in Ks:
        for g in ("S", "SK"):
            src, dst = "LP_%s" % g, "CN_%s" % g
            init = np.load(C.map_path(src, K)).astype(np.int32)
            assert D.sha_file(C.map_path(src, K)) == prev["maps"]["%s__k%d" % (src, K)]["sha256"], "the LP map differs from its record"
            init_raw, out_raw = os.path.join(tmp, "init.raw"), os.path.join(tmp, "out.raw")
            init.tofile(init_raw)
            lab, info = CN.cn_c(N, K, sweeps, T, dmax, init_raw, out_raw, streams[g])
            np.save(C.map_path(dst, K), lab)
            st = {gg: P.stats_c(N, K, C.map_path(dst, K), streams[gg]) for gg in ("S", "SK")}
            st0 = prev["maps"]["%s__k%d" % (src, K)]["stats"]
            rec["maps"]["%s__k%d" % (dst, K)] = {"method": "R3b closed-neighbourhood LP (LP init) on graph %s (C)" % g, "K": K, "seconds": info["seconds"], "sweeps_run": info["sweeps_run"],
                                                 "moved_per_sweep": info["moved_per_sweep"], "km1_delta_per_sweep": info["km1_delta_per_sweep"], "seconds_per_sweep": info["seconds_per_sweep"],
                                                 "km1_before": info["km1_before"], "km1_after": info["km1_after"], "km1_bookkeeping_consistent": info["km1_bookkeeping_consistent"],
                                                 "hubs_skipped": info["hubs_skipped"], "dense_anchors": info["dense_anchors"], "pool_bytes": info["pool_bytes"], "cap": info["cap"], "lo": info["lo"],
                                                 "c_peak_rss_kb": info["peak_rss_kb"], "c_anon_bytes": info["anon_bytes"], "sha256": D.sha_file(C.map_path(dst, K)), "stats": st, "init_stats": st0}
            log("K %5d %-5s: %d sweeps (%.1fs), moved %s, KM1 %d -> %d (%s), cut on %s %.4f -> %.4f, closed-nbhd blocks %.3f -> %.3f, sizes [%d, %d]" % (
                K, dst, info["sweeps_run"], info["seconds"], info["moved_per_sweep"], info["km1_before"], info["km1_after"], "consistent" if info["km1_bookkeeping_consistent"] else "multi-edge heuristic",
                g, st0[g]["cut_fraction"], st[g]["cut_fraction"], st0[g]["closed_nbhd_distinct_blocks_mean"], st[g]["closed_nbhd_distinct_blocks_mean"], st[g]["size_min"], st[g]["size_max"]))
    rec["code"] = {"scratchpad/_fbx_cn_maps.py": D.sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_cn.py": D.sha_file(os.path.join(D.HERE, "_fbx_cn.py")),
                   "scratchpad/_fbx_cn.c": D.sha_file(os.path.join(D.HERE, "_fbx_cn.c")), "scratchpad/_fbx_part.py": D.sha_file(os.path.join(D.HERE, "_fbx_part.py"))}
    C.wj(rec_path, rec)
    log("CN MAPS -> %s %s" % (rec_path, D.sha_file(rec_path)[:12]))


if __name__ == "__main__":
    main()
