"""FBX_SCALE stage 3b -- build one partitioner's maps at scale and measure the build (systems validation; no retrieval mechanism is tuned here).

The method is an argument, chosen by the stage-3a verdict of addendum 1 (LDG on the served STRUCT graph, or the balanced-hash control):

  python scratchpad/_fbx_scale_build.py webqsp   <LDG_S|HASH> <tag> [--K=100,250]      # the generic code path on WebQSP (a regression of MAPS' maps)
  python scratchpad/_fbx_scale_build.py freebase <LDG_S|HASH> <tag> [--K=100,1000,...]  # data/final_canonical/freebase, read-only, memory-mapped

For every K it records: wall seconds of the map build, peak RSS of the builder process (C program: getrusage of the mapped run, which counts the
memory-mapped adjacency pages it touched; the anonymous allocation is reported separately), output bytes, sha256 of the map, block sizes and
load skew, cut fraction, the mean and max number of distinct blocks in a closed neighbourhood (KM1 of the anchor hyperedges) and the number of
least-loaded fallbacks.  The Freebase graph is read as the served OUT and IN CSRs (two adjacency streams, multiplicities kept, self loops never
counted because a vertex is unlabelled while it is placed); a map is written to work/FBX_SCALE/<tag>/ (outside data/final_canonical, which is never
written) and the record to results/FREEBASE_SCALE/FBX_BUILD__<method>__<tag>.json (write-once; a per-K partial is kept beside the maps so a failed
run resumes).  Balanced hash uses seed 0 (the control, not a candidate).
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

import _fbx_part as P

ROOT = os.environ.get("FBX_ROOT") or os.getcwd()
OUTR = os.path.join(ROOT, "results", "FREEBASE_SCALE")
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def peak_rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 1048576.0, 1)
    except Exception:
        return None


def freebase_streams():
    g = os.path.join(os.path.realpath(os.path.join(ROOT, "data", "final_canonical", "freebase")), "graph")
    spec = lambda n: P.npy_spec(os.path.join(g, n))
    ind = np.load(os.path.join(g, "out_indptr.npy"), mmap_mode="r")
    N = int(len(ind) - 1)
    assert int(ind[N]) == 2062430072 and N == 301977131
    return N, [spec("out_indptr.npy"), spec("out_dst.npy"), spec("in_indptr.npy"), spec("in_src.npy")], {"graph": "served OUT + IN CSR (multiplicities kept)", "directed_edges": int(ind[N])}


def webqsp_streams(work):
    z = np.load(os.path.join(ROOT, "data", "l1_canonical", "webqsp", "keys.npz"))
    N = int(z["N"][0])
    k = z["STRUCT"]
    a, b = k // N, k % N
    m = a != b
    k = np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))
    xa, aa = P.csr_from_pairs(k // N, k % N, N)
    return N, P.write_csr_raw(xa, aa, os.path.join(work, "webqsp_S")), {"graph": "webqsp STRUCT undirected", "undirected_pairs": int(len(k))}


def run(dataset, method, tag, Ks):
    assert method in ("LDG_S", "HASH"), method
    out_rec = os.path.join(OUTR, "FBX_BUILD__%s__%s__%s.json" % (dataset, method, tag))
    assert not os.path.exists(out_rec), "write-once: %s exists" % out_rec
    work = os.path.join(ROOT, "work", "FBX_SCALE", "%s_%s_%s" % (dataset, method, tag))
    os.makedirs(work, exist_ok=True)
    N, streams, ginfo = freebase_streams() if dataset == "freebase" else webqsp_streams(work)
    _, c_sha = P.c_binary()
    rec = {"stage": "FBX_SCALE 3b: %s maps on %s" % (method, dataset), "tag": tag, "method": method, "dataset": dataset, "N": N, "graph": ginfo, "K_grid": Ks,
           "c_source_sha12": c_sha, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "maps": {}}
    for K in Ks:
        part = os.path.join(work, "K%d.json" % K)
        if os.path.exists(part):
            rec["maps"][str(K)] = json.load(io.open(part, encoding="utf-8"))
            log("K %d: resumed from its partial record" % K)
            continue
        npy = os.path.join(work, "%s_k%d.npy" % (method, K))
        t0 = time.time()
        if method == "LDG_S":
            raw = npy[:-4] + ".raw"
            lab, info = P.ldg_c(N, K, streams, raw)
            build = {"seconds_pass": info["seconds"], "cap": info["cap"], "least_loaded_fallbacks": info["least_loaded_fallbacks"], "c_peak_rss_kb": info["peak_rss_kb"],
                     "c_anon_bytes": info["anon_bytes"]}
            np.save(npy, lab)
            os.remove(raw)
            del lab
        else:
            h = P.hash_balanced(N, K, 0)
            build = {"seconds_pass": round(time.time() - t0, 3), "python_peak_rss_mb": peak_rss_mb(), "seed": 0}
            np.save(npy, h)
            del h
        build["seconds_wall_incl_write"] = round(time.time() - t0, 3)
        log("K %d: built (%.1fs pass, %.1fs incl. write)" % (K, build["seconds_pass"], build["seconds_wall_incl_write"]))
        t1 = time.time()
        st = P.stats_c(N, K, npy, streams)
        log("K %d: stats (%.1fs): cut %.4f, closed-nbhd blocks mean %.3f, skew %.4f" % (K, time.time() - t1, st["cut_fraction"], st["closed_nbhd_distinct_blocks_mean"], st["load_skew_max_over_mean"]))
        ent = {"K": K, "build": build, "stats": st, "map_file": os.path.relpath(npy, ROOT).replace("\\", "/"), "map_bytes": os.path.getsize(npy), "map_sha256": sha_file(npy)}
        with io.open(part, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(ent, indent=1, ensure_ascii=False))
        rec["maps"][str(K)] = ent
    rec["seconds_total"] = round(time.time() - T0, 1)
    rec["python_peak_rss_mb"] = peak_rss_mb()
    rec["code"] = {"scratchpad/_fbx_scale_build.py": sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_part.py": sha_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fbx_part.py")),
                   "scratchpad/_fbx_ldg.c": sha_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fbx_ldg.c"))}
    with io.open(out_rec, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    log("record -> %s %s" % (out_rec, sha_file(out_rec)[:12]))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    ks = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--K=")), None)
    if ks:
        Ks = [int(x) for x in ks.split(",")]
    else:
        Ks = [100, 1000, 2000, 5000, 10000, 20000] if sys.argv[1] == "freebase" else [100, 250, 500, 1000, 2000, 5000, 25928]
    run(sys.argv[1], sys.argv[2], sys.argv[3], Ks)
