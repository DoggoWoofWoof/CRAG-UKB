"""FBX_SCALE addendum 17 -- a mid-scale SYNTHETIC stress run of the whole out-of-core chain (csr -> twins -> ladder -> level-4 labels -> V-cycle) on a Freebase-shaped graph (59 % degree-1 leaves, power-law hubs of
~15 % of N), to look for scaling hazards (time, RSS, heap, int widths, split chunks) before the Freebase run.  Not a result: no regression target exists at this size; every stage has its own bit-identity test.

  python -u scratchpad/_ooc_stress.py [N] [threads]      # default N = 3,000,000
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _ooc_lib as L  # noqa: E402
import _ooc_fb as FBD  # noqa: E402
import _ooc_vc as VC  # noqa: E402
from _ooc_e1 import tmpdir  # noqa: E402

LOG = L.log


def make_graph(N, d):
    rng = np.random.RandomState(3)
    leaf_n = int(0.55 * N)
    core = N - leaf_n
    src_l = np.arange(core, N, dtype=np.int64)                                   # one edge per leaf, to a hub-heavy target in the core
    dst_l = (core * rng.random(leaf_n) ** 3).astype(np.int64)
    E_core = int(5.5 * core)
    src_c = rng.randint(0, core, E_core).astype(np.int64)
    dst_c = (core * rng.random(E_core) ** 4).astype(np.int64)
    src = np.concatenate([src_l, src_c])
    dst = np.concatenate([dst_l, dst_c])
    hub_deg = int(np.bincount(dst, minlength=N).max())
    LOG("synthetic graph: N %d, directed edges %d, max in-degree %d (%.1f %% of N)" % (N, len(src), hub_deg, 100.0 * hub_deg / N))
    o = np.argsort(src, kind="stable")
    oi = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=oi[1:])
    od = dst[o].astype(np.int32)
    o2 = np.argsort(dst, kind="stable")
    ii = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(dst, minlength=N), out=ii[1:])
    isr = src[o2].astype(np.int32)
    g = os.path.join(d, "graph")
    os.makedirs(g)
    for nm, a in (("out_indptr", oi), ("out_dst", od), ("in_indptr", ii), ("in_src", isr)):
        np.save(os.path.join(g, nm + ".npy"), a)
    return g


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
    thr = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    d = tmpdir("stress")
    t0 = time.time()
    g = make_graph(N, d)
    csr = os.path.join(d, "csr")
    os.makedirs(csr)
    r = L.run("csr", {"graph": L.wp(g), "out": L.wp(csr), "threads": thr, "chunk": 1000000}, stream=False)
    LOG("csr: %s" % json.dumps({k: r[k] for k in r if k in ("und_sum", "anchors", "seconds", "peak_rss_gb")}))
    fam = [L.wp(csr) + "/xa.i64," + L.wp(csr) + "/aa.i32"]
    l1 = os.path.join(d, "l1")
    os.makedirs(l1)
    tw = L.run("twins", {"N": N, "fam": fam, "wmax": 512, "out": L.wp(l1) + "/cl1.i32", "threads": thr}, stream=False)
    LOG("twins: %d clusters (%.2fx), %.0fs, peak RSS %s GB" % (tw["clusters"], N / tw["clusters"], tw["seconds"], tw.get("peak_rss_gb")))
    lad = FBD.ladder(L.wp(csr), L.wp(l1), thr, N=N, V1=tw["clusters"])
    Vs = {l: lad["levels"][l]["V"] for l in (1, 2, 3, 4)}
    for l in (1, 2, 3, 4):
        LOG("ladder level %d: V %d M %d P %d" % (l, Vs[l], lad["levels"][l]["M"], lad["levels"][l]["P"]))
    work = os.path.join(d, "work")
    os.makedirs(work)
    rec, sdir = VC.surrogate(fam, N, L.wp(l1), L.wp(work), Vs[4], thr, "stress", with_zoltan=False)
    LOG("surrogate: V %d, M %d, P %d (%.1f %% of the level-4 unsplit pins)" % (rec["surrogate"]["V"], rec["surrogate"]["M"], rec["surrogate"]["P"], 100.0 * rec["surrogate"]["P"] / lad["levels"][4]["P"]))
    cw = np.bincount(np.fromfile(os.path.join(l1, "cl4.i32"), np.int32), minlength=Vs[4])
    lab_top = np.zeros(Vs[4], np.int32)
    load = np.zeros(VC.S, np.int64)
    for v in np.argsort(-cw, kind="stable"):
        b = int(np.argmin(load))
        lab_top[v] = b
        load[b] += cw[v]
    L.wr(lab_top, os.path.join(work, "zlab4.i32"), np.int32)
    vr = VC.vcycle(fam, N, L.wp(l1), L.wp(work), Vs, thr)
    for t in vr["trajectory"]:
        LOG("level %d: V %d, km1 %d -> %d, %d moves, passes %d, heap peak %d, pushes %d, cache %d B/entry, FM %.1fs, RSS %.2f GB" % (
            t["level"], t["V"], t["km1_in"], t["km1_out"], t["moves"], t["passes_run"], t["heap_peak_entries"], t["pushes"], t["cache_bytes_per_entry"], t["fm_seconds"], t["fm_peak_rss_gb"]))
    if os.environ.get("STRESS_SPLITCHECK"):                                       # the level-range jobs compose: levels 4-2 then 1-0 in a second work dir == the one-shot V-cycle, byte for byte
        work2 = os.path.join(d, "work2")
        os.makedirs(work2)
        L.wr(lab_top, os.path.join(work2, "zlab4.i32"), np.int32)
        VC.vcycle(fam, N, L.wp(l1), L.wp(work2), Vs, thr, hi=4, lo=2)
        VC.vcycle(fam, N, L.wp(l1), L.wp(work2), Vs, thr, hi=1, lo=0)
        assert open(os.path.join(work, "lab0.i32"), "rb").read() == open(os.path.join(work2, "lab0.i32"), "rb").read(), "level-range V-cycle differs from the one-shot"
        LOG("SPLITCHECK PASS: levels 4-2 + 1-0 in two runs == the one-shot V-cycle")
    v = VC.validity(L.wp(os.path.join(work, "lab0.i32")), N)
    LOG("final blocks %d..%d (bound %d): %s; total %.0fs" % (v["min_block"], v["max_block"], v["contract_bound_ceil_1.03_N_over_k"], v["gate"], time.time() - t0))


if __name__ == "__main__":
    main()
