"""FBX_SCALE stage 4C -- the SK-arm multilevel lab on WebQSP: clusterings of the frozen H4_SK hypergraph, the coarse PHG cells, the maps.

  python -u scratchpad/_ml2_run.py COARSEN <M1|M2|M3|M4> <Wmax>[,<Wmax>...]   # structure only: K-independent cluster maps after every level + one write-once record per (method, Wmax)

Every record is under results/FREEBASE_SCALE/ml2/.  The clustering reads NO gold label and no recall; the level statistics are the pin reductions of the UNSPLIT SK nets.
"""
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

import _l1h_host as H
import _ml2_coarsen as C2
import _ml_coarsen as C

REPO = H.REPO
log, wj, sha_file, rel = H.log, H.wj, H.sha_file, H.rel
DS = "webqsp"
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "ml2")
KEYS = os.path.join(REPO, "data", "l1_canonical", DS, "keys.npz")
PARAMS = {"Lmax": 200, "R": 1.0e9, "MAXLEVELS": 12, "MIN_GAIN": 0.05, "twins": "open, both families"}


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
    except Exception:
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)
        except Exception:
            return None


def graph_SK():
    z = np.load(KEYS)
    N = int(z["N"][0])
    fams, cnt = [], {}
    for f in ("STRUCT", "KNN"):
        k = z[f]
        a, b = k // N, k % N
        m = a != b
        k = np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))
        cnt[f] = int(len(k))
        fams.append(C.sorted_csr(k // N, k % N, N))
    return N, fams, cnt


def code_pins():
    fs = ["scratchpad/_ml2_run.py", "scratchpad/_ml2_coarsen.py", "scratchpad/_ml2_agg.c", "scratchpad/_ml_coarsen.py", "scratchpad/_fbx_part.py", "scratchpad/_l1h_host.py",
          "src/l1_canonical/hypergraph.py"]
    return {f: sha_file(os.path.join(REPO, f)) for f in fs}


def map_path(method, W, level):
    return os.path.join(OUT, "clusters__%s_W%d_L%d.npy" % (method, W, level))


def rec_path(method, W):
    return os.path.join(OUT, "COARSEN__%s_W%d.json" % (method, W))


def cmd_coarsen(method, Ws):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    N, fams, cnt = graph_SK()
    log("graph SK: N %d, STRUCT pairs %d, KNN pairs %d (%.0fs)" % (N, cnt["STRUCT"], cnt["KNN"], time.time() - t0))
    for W in Ws:
        runp = rec_path(method, W)
        assert not os.path.exists(runp), "write-once: %s exists" % runp
        t = time.time()
        tmp = tempfile.mkdtemp(prefix="ml2co_")
        total, levels, (V, ep, ei, ew, vw), maps = C2.coarsen2(N, fams, method, PARAMS["Lmax"], W, PARAMS["R"], PARAMS["MAXLEVELS"], tmp, min_gain=PARAMS["MIN_GAIN"])
        P0 = levels[0]["P"]
        outs = []
        for lv, cl in sorted(maps.items()):
            p = map_path(method, W, lv)
            np.save(p[:-4] + ".tmp.npy", cl)
            os.replace(p[:-4] + ".tmp.npy", p)
            sizes = np.bincount(cl)
            info = next(x for x in levels if x["level"] == lv and "skipped" not in x)
            outs.append({"level": lv, "kind": info["kind"], "clusters": int(cl.max() + 1), "cluster_size": {"max": int(sizes.max()), "mean": round(float(sizes.mean()), 3),
                                                                                                        "p50": float(np.median(sizes)), "singletons": int((sizes == 1).sum())},
                         "unsplit_pins": int(info["P"]), "unsplit_nets": int(info["M"]), "pin_reduction": round(P0 / max(int(info["P"]), 1), 3), "file": rel(p), "sha256": sha_file(p)})
        rec = {"RECORD": "ML2_COARSEN", "dataset": DS, "method": method, "rating": C2.METHODS[method][0], "schedule": C2.METHODS[method][1], "Wmax": W, "params": PARAMS,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "N": N, "pairs": cnt, "fine_unsplit": {"M": levels[0]["M"], "P": P0}, "levels": levels, "maps": outs,
               "final": {"V": int(V), "M": int(len(ew)), "P": int(len(ei)), "pin_reduction": round(P0 / max(len(ei), 1), 3), "max_vertex_weight": int(vw.max())},
               "seconds": round(time.time() - t, 1), "peak_rss_mb": peak_rss_mb(), "placement": H.placement(), "code": code_pins()}
        wj(runp, rec)
        log("%s W%d: V %d, unsplit pins %d -> %d (%.2fx), %d levels kept, %.0fs" % (method, W, V, P0, len(ei), rec["final"]["pin_reduction"], len(outs), rec["seconds"]))
        del ep, ei, ew, vw, maps, total


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "COARSEN":
        cmd_coarsen(a[1], [int(x) for x in a[2].split(",")])
    else:
        raise SystemExit(__doc__)
