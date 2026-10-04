"""Which faiss OPQ / PQ training calls crash on this host (each case in its own process, exit code reported)?  No side effects.  python scratchpad/_pq_probe.py"""
import subprocess
import sys

CASE = r"""
import sys, time, numpy as np
n, m, mode, torch_first = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
if torch_first:
    import torch
    a = torch.randn(512, 512); (a @ a.T).sum().item()
import faiss
faiss.omp_set_num_threads(4)
rng = np.random.RandomState(0)
x = rng.randn(n, 1536).astype("float32"); x /= np.linalg.norm(x, axis=1, keepdims=True)
t = time.time()
if mode == "opq":
    o = faiss.OPQMatrix(1536, m); o.train(x); print("opq ok %.1fs" % (time.time() - t), flush=True)
    y = o.apply(x)
else:
    y = x
p = faiss.ProductQuantizer(1536, m, 8); p.train(y); print("pq ok %.1fs" % (time.time() - t), flush=True)
c = p.compute_codes(y); d = p.decode(c); print("codec ok", c.shape, d.shape, flush=True)
"""

for n, m, mode, tf in ((20000, 16, "pq", 0), (20000, 16, "pq", 1), (1400, 16, "opq", 0), (20000, 16, "opq", 0), (20000, 16, "opq", 1), (30000, 64, "opq", 0)):
    r = subprocess.run([sys.executable, "-c", CASE, str(n), str(m), mode, str(tf)], capture_output=True, text=True, timeout=1500)
    out = [l for l in (r.stdout + r.stderr).splitlines() if "WARNING clustering" not in l]
    print("CASE n=%d m=%d %s torch_first=%d -> rc %d | %s" % (n, m, mode, tf, r.returncode, " / ".join(out[-3:])), flush=True)
