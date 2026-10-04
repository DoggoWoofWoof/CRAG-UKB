"""stage 4A development probe (structure only: no gold labels, no recall): how far does the coarsener go on WebQSP graph S?  python scratchpad/_ml_probe.py <Wmax> <Lmax> <R> [levels]"""
import json
import os
import sys
import tempfile

import numpy as np

import _ml_coarsen as C

Wmax, Lmax, R = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
levels = int(sys.argv[4]) if len(sys.argv) > 4 else 10
z = np.load(os.path.join(C.HERE, "..", "data", "l1_canonical", "webqsp", "keys.npz"))
N = int(z["N"][0])
k = z["STRUCT"]
a, b = k // N, k % N
m = a != b
k = np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))
xa, aa = C.sorted_csr(k // N, k % N, N)
C.log("N %d pairs %d" % (N, len(k)))
tmp = tempfile.mkdtemp(prefix="mlprobe_")
total, lv, (V, ep, ei, ew, vw) = C.coarsen(N, xa, aa, Lmax, Wmax, R, levels, tmp)
C.log("FINAL V %d M %d P %d (%.2fx of %d), max vertex weight %d" % (V, len(ew), len(ei), lv[0]["P"] / len(ei), lv[0]["P"], int(vw.max())))
np.save(os.path.join(tmp, "cl.npy"), total)
C.log("tmp", tmp)
