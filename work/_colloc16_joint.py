"""Joint (MetaQA + WebQSP) view of results/L1_X/colloc16_<ds>__v1: a universal rule is SELECTED on the mean of the two datasets' half-A objectives (router ES_m), REPORTED on half B per dataset, with paired McNemar cell counts vs the named candidate
(ES_m, Sro+Sri+Tin+BM).  Text datasets are untouched by construction (typed switch off).  python work/_colloc16_joint.py"""
import json
from math import comb

import numpy as np

OUT = "results/L1_X/"
DS = ["metaqa", "webqsp"]
CAND = "Sro+Sri+Tin+BM"
NB = 4
D = {}
for ds in DS:
    rec = json.load(open(OUT + "colloc16_%s__v1.json" % ds))
    z = np.load(OUT + "colloc16_%s__v1.npz" % ds)
    D[ds] = (rec, z, rec["rules"], len(rec["budgets"]), sorted({int(c.split("|")[0]) for c in rec["cells"]}), (z["rows"] % 2) == 0)
names = D["metaqa"][2]
assert names == D["webqsp"][2]


def arr(ds, K, rt, nm):
    rec, z, nms, NM, _, _ = D[ds]
    return np.unpackbits(z["ALL__%d|%s" % (K, rt)], axis=2)[:, :, :NM][nms.index(nm)].astype(bool)


def obj(ds, rt, nm, half):
    rec, z, nms, NM, cells, A = D[ds]
    m = A if half == "A" else ~A
    return float(np.mean([arr(ds, K, rt, nm)[m][:, :NB].mean() for K in cells]))


def p2(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def sig(ds, rt, nm):
    rec, z, nms, NM, cells, A = D[ds]
    gp = gn = 0
    for K in cells:
        a, b = arr(ds, K, rt, CAND)[~A], arr(ds, K, rt, nm)[~A]
        for m in range(NB):
            g, l = int((b[:, m] & ~a[:, m]).sum()), int((a[:, m] & ~b[:, m]).sum())
            if p2(g, l) < 0.05:
                gp += g > l
                gn += l > g
    return gp, gn


rows = []
for nm in names:
    aM, aW = obj("metaqa", "ES_m", nm, "A"), obj("webqsp", "ES_m", nm, "A")
    bM, bW = obj("metaqa", "ES_m", nm, "B"), obj("webqsp", "ES_m", nm, "B")
    rows.append((nm, (aM + aW) / 2, aM, aW, bM, bW, sig("metaqa", "ES_m", nm), sig("webqsp", "ES_m", nm)))
rows.sort(key=lambda r: -r[1])
cM, cW = obj("metaqa", "ES_m", CAND, "B"), obj("webqsp", "ES_m", CAND, "B")
print("router ES_m; objective = mean over K cells and B_N 100..1000; selection = mean of half-A objectives of the two KB datasets")
print("%-24s %8s | %7s %7s | %7s %7s | %8s %8s | %-9s %-9s" % ("arm", "A joint", "A meta", "A wqsp", "B meta", "B wqsp", "dB meta", "dB wqsp", "sig meta", "sig wqsp"))
for nm, aj, aM, aW, bM, bW, sM, sW in rows:
    print("%-24s %8.4f | %7.4f %7.4f | %7.4f %7.4f | %+8.4f %+8.4f | +%d/-%d     +%d/-%d" % (nm, aj, aM, aW, bM, bW, bM - cM, bW - cW, sM[0], sM[1], sW[0], sW[1]))
print("\ncandidate B: metaqa %.4f webqsp %.4f   joint-A winner: %s" % (cM, cW, rows[0][0]))
