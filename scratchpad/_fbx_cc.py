"""FBX_SCALE Track A -- exact connected components of the served Freebase STRUCT graph (driver + test for _fbx_cc.c).

  python scratchpad/_fbx_cc.py TEST     # the C program equals scipy connected_components on random graphs (with self loops, isolated nodes, multi-edges)
  python scratchpad/_fbx_cc.py RUN      # write-once results/FREEBASE_SCALE/FBX_STRUCT_COMPONENTS__v1.json  (reads data/final_canonical/freebase/graph/out_{indptr,dst}.npy, read-only)
"""
import hashlib
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

import _fbx_part as P

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GRAPH = os.path.join(REPO, "data", "final_canonical", "freebase", "graph")
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "FBX_STRUCT_COMPONENTS__v1.json")


def c_binary():
    src = os.path.join(HERE, "_fbx_cc.c")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    binp = "/tmp/fbx_cc_%s" % sha
    try:
        P._run(["test", "-x", binp])
    except RuntimeError:
        P._run(["gcc", "-O2", "-Wall", "-o", binp, P.wsl_path(src)])
    return binp, sha


def run_c(N, indptr_npy, dst_npy):
    binp, sha = c_binary()
    out = P._run([binp, str(int(N)), P.wsl_path(indptr_npy), P.wsl_path(dst_npy)])
    r = json.loads(out.strip().splitlines()[-1])
    r["c_source_sha12"] = sha
    return r


def _test():
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    rng = np.random.RandomState(11)
    tmp = tempfile.mkdtemp(prefix="fbxcc_")
    for ci, (N, m) in enumerate(((50, 40), (1000, 700), (5000, 4000), (20000, 30000), (7, 0))):
        u = rng.randint(0, N, m)
        v = rng.randint(0, N, m)
        v[: m // 10] = u[: m // 10]                                # self loops
        u = np.concatenate([u, u[: m // 7]])                        # multi-edges
        v = np.concatenate([v, v[: m // 7]])
        o = np.lexsort((v, u))
        u, v = u[o], v[o]
        indptr = np.zeros(N + 1, np.int64)
        np.add.at(indptr, u + 1, 1)
        indptr = np.cumsum(indptr)
        pi, pd = os.path.join(tmp, "ip%d.npy" % ci), os.path.join(tmp, "d%d.npy" % ci)
        np.save(pi, indptr)
        np.save(pd, v.astype(np.int32))
        assert np.load(pi, mmap_mode="r").offset == 128 and (m == 0 or np.load(pd, mmap_mode="r").offset == 128), "the C reader assumes a 128-byte npy header"
        got = run_c(N, pi, pd)
        keep = u != v
        A = sp.coo_matrix((np.ones(int(keep.sum())), (u[keep], v[keep])), shape=(N, N))
        nc, lab = connected_components(A, directed=False)
        sizes = np.sort(np.bincount(lab))[::-1]
        assert got["components"] == nc, (got["components"], nc)
        assert got["singleton_components"] == int((sizes == 1).sum())
        assert got["largest_ten"][: min(10, len(sizes))] == [int(x) for x in sizes[:10]] and all(x == 0 for x in got["largest_ten"][len(sizes):])
        assert got["self_loops_skipped"] == int((u == v).sum()) and got["edges_read"] == len(u)
        hist = np.bincount(np.floor(np.log2(sizes)).astype(int))
        assert got["component_size_hist_log2"] == [int(x) for x in hist], (got["component_size_hist_log2"], hist)
        print("ok case %d: N %d, %d edge rows -> %d components (largest %d)" % (ci, N, len(u), nc, sizes[0]))
    print("TEST PASS (C source sha12 %s)" % c_binary()[1])


def _run():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    ip = os.path.join(GRAPH, "out_indptr.npy")
    dst = os.path.join(GRAPH, "out_dst.npy")
    N = int(np.load(ip, mmap_mode="r").shape[0]) - 1
    t = time.time()
    r = run_c(N, ip, dst)
    r.update({"RECORD": "FBX_STRUCT_COMPONENTS", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tree": "data/final_canonical/freebase/graph (read-only)",
              "view": "undirected view of the served STRUCT edges (out-CSR; the in-CSR is its transpose); self loops skipped; multi-edges collapse",
              "isolated_nodes_from_stats_record": "results/FREEBASE_SCALE/FBX_GRAPH_STATS__v1.json stats.isolated_nodes",
              "wall_seconds": round(time.time() - t, 1), "code_sha256": hashlib.sha256(open(os.path.join(HERE, "_fbx_cc.c"), "rb").read()).hexdigest(),
              "driver_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()})
    r["largest_component_share"] = round(r["largest_ten"][0] / float(N), 6)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["TEST"]:
        _test()
    elif a == ["RUN"]:
        _run()
    else:
        raise SystemExit(__doc__)
