"""STRUCT_VERTEXCUT_V1 step 2: the pre-registered matched-k fixed point (PREREGISTRATION_STRUCT_VERTEXCUT_V1.json, k_grid.matched_k_star).

    target: mean unique nodes per block = C = round(N / k_f)          (the hard block capacity; 100 on all three graphs)
    k_1 = round(RF(2 k_f) * N / C);  k_{i+1} = round(RF(k_i) * N / C)  until |mean |B_p| - C| <= 0.05 C or three k* runs
Every k is built by _l1c_vcut_build.py and partitioned by the validated PHG driver (_l1c_phg.py); existing artifacts are reused.
    python -u _l1c_vcut_kstar.py <ds>    -> results/L1_COVPART/vcut_kstar_<ds>.json
"""
import os
import subprocess
import sys
import time

import numpy as np

import _l1c_vcut_lib as V

HERE = os.path.dirname(os.path.abspath(__file__))
ENV = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
ds = sys.argv[1]
t0 = time.time()
G = V.Graph(ds)
C, N = G.C, G.N


def ensure(k):
    npy = os.path.join(V.PDIR, "%s__%s_k%d__PHG_con.npy" % (ds, V.TAG, k))
    if not os.path.exists(npy):
        subprocess.run([sys.executable, "-u", os.path.join(HERE, "_l1c_vcut_build.py"), ds, str(k)], check=True, env=ENV, cwd=HERE)
        subprocess.run([sys.executable, "-u", os.path.join(HERE, "_l1c_phg.py"), ds, "%s_k%d" % (V.TAG, k)], check=True, env=ENV, cwd=HERE)
    z, meta, run = V.load_vcut(ds, k)
    st = V.vcut_size_stats(G, z, k)
    V.log("  k %d: RF %.4f mean |B_p| %.2f max %d blocks > C %d" % (k, st["RF"], st["mean_block_size"], st["max_block_size"], st["blocks_gt_C"]))
    return st


seq = []
for k in (G.k_f, 2 * G.k_f, 3 * G.k_f):
    seq.append(dict(k=k, role="curve", **ensure(k)))
rf = seq[1]["RF"]
k = int(round(rf * N / float(C)))
runs = 0
while True:
    st = ensure(k)
    runs += 1
    seq.append(dict(k=k, role="kstar_%d" % runs, **st))
    if abs(st["mean_block_size"] - C) <= 0.05 * C or runs >= 3:
        break
    k = int(round(st["RF"] * N / float(C)))
    if any(s["k"] == k for s in seq):
        break
rec = {"dataset": ds, "N": N, "k_frozen": G.k_f, "C": C, "target_mean_block_size": C, "tolerance": 0.05 * C, "sequence": seq,
       "k_star": seq[-1]["k"], "k_star_mean_block_size": seq[-1]["mean_block_size"], "k_star_RF": seq[-1]["RF"], "k_star_runs": runs,
       "converged": bool(abs(seq[-1]["mean_block_size"] - C) <= 0.05 * C), "seconds": round(time.time() - t0, 1)}
V.S.wj(os.path.join(V.OUT, "vcut_kstar_%s.json" % ds), rec)
V.log("k* = %d (mean |B_p| %.2f, RF %.4f, converged %s) after %d k* runs (%.0fs)" % (rec["k_star"], rec["k_star_mean_block_size"], rec["k_star_RF"], rec["converged"], runs, time.time() - t0))
