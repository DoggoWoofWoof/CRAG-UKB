"""Step 17 evaluation of results/L1_X/colloc17_<ds>__<tag>: router comparison (contact = the plateau) and the PRE-DECLARED confidence gates.  Nothing is tuned: the four gates and their constants are fixed here, before any result was read.

  G1 diversity            : the top BM block is at least as diverse as the BM-weighted mean of the contacted blocks      dpr1 >= dprb
  G2 relation concentration: one relation carries a majority of the seed-edge mass (query-weighted)                      relconc >= 1/2
  G3 rank agreement       : BM and BM*Dpr agree on at least half of the contacted (top-B_P) blocks                      agreeb >= 1/2
  G4 consensus            : at least two of G1, G2, G3
  gated arm: S(q) = T+CRD if the gate is on else Sro+Sri+Tin+CRD, per (K, row); the signals use the ES_m block order.
Diagnostics (no selection): per signal, the Mann-Whitney AUC separating half-A rows where the aggressive form is the (only) winner from rows where the conservative form is, B_N <= 1000.
  python work/_colloc17_gate.py <ds> <tag>"""
import json
import sys
from math import comb

import numpy as np

ds, tag = sys.argv[1], sys.argv[2]
OUT = "results/L1_X/"
rec = json.load(open(OUT + "colloc17_%s__%s.json" % (ds, tag)))
z = np.load(OUT + "colloc17_%s__%s.npz" % (ds, tag))
names, NM = rec["rules"], len(rec["budgets"])
cells = sorted({int(c.split("|")[0]) for c in rec["cells"]})
ROUT = rec["routers"]
rows = z["rows"]
A, B = (rows % 2) == 0, (rows % 2) == 1
CAND, CON, AGG = "Sro+Sri+Tin+BM", "Sro+Sri+Tin+CRD", "T+CRD"
NB = 4
SIGN = ("dpr1", "dprb", "marg", "ent", "agree1", "agreeb", "agreec", "relconc")


def arr(K, rt, nm):
    return np.unpackbits(z["ALL__%d|%s" % (K, rt)], axis=2)[:, :, :NM][names.index(nm)].astype(bool)


def sig(K, n):
    return z["SIG__%d__%s" % (K, n)]


def gates(K):
    g1 = sig(K, "dpr1") >= sig(K, "dprb")
    g2 = sig(K, "relconc") >= 0.5
    g3 = sig(K, "agreeb") >= 0.5
    return {"G1 diversity": g1, "G2 relation-conc": g2, "G3 rank-agreement": g3, "G4 consensus(>=2)": (g1.astype(int) + g2 + g3) >= 2}


def p2(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def objv(get, half):
    return float(np.mean([get(K)[half][:, :NB].mean() for K in cells]))


def sigcells(get, ref, half):
    gp = gn = 0
    for K in cells:
        a, b = ref(K)[half], get(K)[half]
        for m in range(NB):
            g, l = int((b[:, m] & ~a[:, m]).sum()), int((a[:, m] & ~b[:, m]).sum())
            if p2(g, l) < 0.05:
                gp += g > l
                gn += l > g
    return gp, gn


ref = lambda K: arr(K, "ES_m", CAND)
print("%s %s  rows A/B %d/%d  cells %s  budgets %s" % (ds, tag, A.sum(), B.sum(), cells, rec["budgets"][:NB]))
print("\n== ROUTERS: contact (all gold blocks contacted = the large-budget plateau) and the objective of three arms; delta vs the ES_m candidate")
print("%-8s %-30s %s" % ("router", "gold blocks all contacted", " | ".join("%-16s A    B     dB" % a[:16] for a in (CAND, CON, AGG))))
for rt in ROUT:
    ct = {K: rec["gold_blocks_all_contacted"]["%d|%s" % (K, rt)] for K in cells}
    out = []
    for nm in (CAND, CON, AGG):
        get = lambda K, nm=nm, rt=rt: arr(K, rt, nm)
        out.append("%.4f %.4f %+.4f" % (objv(get, A), objv(get, B), objv(get, B) - objv(ref, B)))
    print("%-8s %-30s %s" % (rt, " ".join("%.3f" % ct[K] for K in cells), " | ".join(out)))

print("\n== six-B_N ALL-gold (all rows) of the conservative +CRD per router, K %s" % cells[1])
for rt in ROUT:
    print("  %-8s %s" % (rt, [round(float(arr(cells[1], rt, CON)[:, m].mean()), 3) for m in range(NM)]))

print("\n== PRE-DECLARED GATES (router ES_m; objective = mean over K cells and B_N 100..1000)")
print("%-22s %7s %7s %9s %-9s %s" % ("arm", "A", "B", "dB vs cand", "sig +/-", "AGG-on rate per K"))
for label, get in [("candidate (BM)", ref), ("conservative +CRD", lambda K: arr(K, "ES_m", CON)), ("aggressive T+CRD", lambda K: arr(K, "ES_m", AGG))]:
    sp = sigcells(get, ref, B)
    print("%-22s %7.4f %7.4f %+9.4f +%d/-%d" % (label, objv(get, A), objv(get, B), objv(get, B) - objv(ref, B), sp[0], sp[1]))
for gname in gates(cells[0]):
    def get(K, gname=gname):
        g = gates(K)[gname]
        return np.where(g[:, None], arr(K, "ES_m", AGG), arr(K, "ES_m", CON))
    sp = sigcells(get, ref, B)
    on = [round(float(gates(K)[gname].mean()), 3) for K in cells]
    print("%-22s %7.4f %7.4f %+9.4f +%d/-%d   %s" % (gname, objv(get, A), objv(get, B), objv(get, B) - objv(ref, B), sp[0], sp[1], on))
def orc(K):
    return arr(K, "ES_m", AGG) | arr(K, "ES_m", CON)
print("%-22s %7.4f %7.4f %+9.4f  (upper bound of any gate between the two forms)" % ("ORACLE{CON,AGG}", objv(orc, A), objv(orc, B), objv(orc, B) - objv(ref, B)))

print("\n== DIAGNOSTIC: AUC of each signal for 'aggressive wins' vs 'conservative wins' (half A, B_N<=1000, per K pooled)")
try:
    from scipy.stats import rankdata
except Exception:  # noqa: BLE001
    rankdata = None
if rankdata is not None:
    for n in SIGN:
        xs, ys = [], []
        for K in cells:
            a, c = arr(K, "ES_m", AGG)[A][:, :NB], arr(K, "ES_m", CON)[A][:, :NB]
            d = (a & ~c).sum(1) - (c & ~a).sum(1)
            s_ = sig(K, n)[A]
            xs.append(s_[d != 0])
            ys.append(d[d != 0] > 0)
        x, y = np.concatenate(xs), np.concatenate(ys)
        if y.all() or (~y).all():
            print("  %-8s degenerate" % n)
            continue
        r = rankdata(x)
        auc = (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
        print("  %-8s AUC %.3f   (n_aggwin %d, n_conwin %d; median signal: aggwin %.3f, conwin %.3f)" % (n, auc, y.sum(), (~y).sum(), np.median(x[y]), np.median(x[~y])))
