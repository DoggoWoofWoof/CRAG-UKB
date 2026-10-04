import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, _l1tp_core as TP
ds = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
n  = int(sys.argv[2]) if len(sys.argv) > 2 else 200
t0 = time.time(); C = TP.ctx(ds)
print(f"[{ds}] ctx {time.time()-t0:.0f}s  nq={C['nq']} ndocs={C['ndocs']} npart={C['npart']}", flush=True)
ok = 0; ne = []; nu = []; tid_ok = 0
for qi in range(min(n, C["nq"])):
    E, rq, par, st = TP.replay(C, qi)
    ok += par; ne.append(len(E["v"])); nu.append(len(np.unique(E["v"])))
    p = E["parent_tid"]; t = E["tid"]
    tid_ok += int(((p < 0) | (p < t)).all() and (t == np.arange(len(t))).all())
ne = np.array(ne); nu = np.array(nu)
print(f"PARITY {ok}/{min(n,C['nq'])}   TID_CONSISTENT {tid_ok}/{min(n,C['nq'])}")
print(f"edges/q  mean {ne.mean():.0f}  median {np.median(ne):.0f}  p95 {np.percentile(ne,95):.0f}  max {ne.max()}")
print(f"uniq targets/q mean {nu.mean():.0f}   collapse ratio {ne.sum()/max(nu.sum(),1):.2f}x")
print(f"total edges {ne.sum():,}   est full-corpus {int(ne.mean()*C['nq']):,}   {time.time()-t0:.0f}s")
