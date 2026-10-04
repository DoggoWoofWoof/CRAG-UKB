import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, _l1tp_core as TP
ds = sys.argv[1]; n = int(sys.argv[2])
C = TP.ctx(ds); hard = C["hard"]
E_, U_, HU_, TR_, PJ_, SRC_ = [], [], [], [], [], []
for qi in range(min(n, C["nq"])):
    E, rq, par, st = TP.replay(C, qi)
    assert par
    v, h, u = E["v"], E["T3_HOP"].astype(np.int64), E["u"]
    E_.append(len(v)); U_.append(len(np.unique(v)))
    HU_.append(len(np.unique(h * (C["ndocs"] + 1) + v)))          # per-hop unique targets
    pi, pj = hard[u], hard[v]
    TR_.append(len(np.unique(pi * (C["npart"] + 1) + pj)))
    PJ_.append(len(np.unique(pj))); SRC_.append(len(np.unique(u)))
f = lambda a: f"{np.mean(a):8.1f}"
print(f"[{ds}] edges{f(E_)} perhop-uniq-v{f(HU_)} (collapse {np.sum(E_)/np.sum(HU_):.2f}x)"
      f"  uniq-v{f(U_)}  uniq-u{f(SRC_)}  transitions{f(TR_)}  uniq-Pj{f(PJ_)}"
      f"  => transition/partition ratio {np.sum(TR_)/np.sum(PJ_):.2f}x", flush=True)
