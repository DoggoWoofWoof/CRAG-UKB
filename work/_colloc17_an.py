"""Analysis of results/L1_X/colloc17_<ds>__<tag>.{json,npz}: half-A objective (selection), half-B objective (report), paired McNemar vs the candidate arm (ES_m, Sro+Sri+Tin+BM) per (K, B_N)."""
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
CAND = ("ES_m", "Sro+Sri+Tin+BM")
NB = 4                                                                  # objective = mean over the first four B_N (100..1000)


def arr(K, rt, nm):
    return np.unpackbits(z["ALL__%d|%s" % (K, rt)], axis=2)[:, :, :NM][names.index(nm)].astype(bool)


def obj(rt, nm, half):
    return float(np.mean([arr(K, rt, nm)[half][:, :NB].mean() for K in cells]))


def p2(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


print("dataset %s  rows A/B %d/%d  cells %s  (objective = mean over cells and B_N %s)" % (ds, A.sum(), B.sum(), cells, rec["budgets"][:NB]))
print("%-24s %8s %8s %8s %8s   %9s %5s %5s" % ("arm", "ES A", "ES B", "ES_m A", "ES_m B", "dB vs cand", "cell+", "cell-"))
order = sorted(names, key=lambda n: -obj("ES_m", n, A))
cand_B = obj(CAND[0], CAND[1], B)
for nm in order:
    gp = gn = 0
    for K in cells:
        a, b = arr(K, CAND[0], CAND[1])[B], arr(K, "ES_m", nm)[B]
        for m in range(NB):
            g, l = int((b[:, m] & ~a[:, m]).sum()), int((a[:, m] & ~b[:, m]).sum())
            p = p2(g, l)
            if p < 0.05:
                gp += g > l
                gn += l > g
    print("%-24s %8.4f %8.4f %8.4f %8.4f   %+9.4f %5d %5d" % (nm, obj("ES", nm, A), obj("ES", nm, B), obj("ES_m", nm, A), obj("ES_m", nm, B), obj("ES_m", nm, B) - cand_B, gp, gn))
best = [n for n in order if n not in ("SHIPPED", "Sro+Sri+Tin", "Sro+Sri+Tin+BM", "Sro+Sri+BRin+BM")][:3]
print("\nhalf-A best new arms under ES_m:", best)
for nm in best + [CAND[1]]:
    print("\n%s under ES_m, half B, ALL-gold per K at B_N %s" % (nm, rec["budgets"][:NB]))
    for K in cells:
        a, b = arr(K, CAND[0], CAND[1])[B], arr(K, "ES_m", nm)[B]
        print("   K%-5d %s   dCand %s" % (K, [round(float(b[:, m].mean()), 3) for m in range(NB)], [("%+d/%d" % (int((b[:, m] & ~a[:, m]).sum()), int((a[:, m] & ~b[:, m]).sum()))) for m in range(NB)]))
if "hopB" in next(iter(rec["table"].values()))[CAND[1]]:
    print("\nper-hop half B (ES_m), B_N %s: candidate vs best new arm" % rec["budgets"][:NB])
    for K in cells:
        for nm in [CAND[1]] + best[:1]:
            e = rec["table"]["%d|ES_m" % K][nm]["hopB"]
            print("   K%-5d %-22s %s" % (K, nm, {h: v[:NB] for h, v in e.items()}))
