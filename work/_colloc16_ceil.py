"""Scoreboard + gate headroom from results/L1_X/colloc16_<ds>__v1 (no new run): ALL-gold at ALL six B_N for the named candidate, the conservative +CRD and the aggressive T+CRD, per K; per-hop on MetaQA (stored hopB);
and the per-row ORACLE over {T+CRD, Sro+Sri+Tin+CRD} (an upper bound for ANY per-query gate between the two forms) and over all typed arms.  python work/_colloc16_ceil.py <metaqa|webqsp>"""
import json
import sys

import numpy as np

ds = sys.argv[1]
OUT = "results/L1_X/"
rec = json.load(open(OUT + "colloc16_%s__v1.json" % ds))
z = np.load(OUT + "colloc16_%s__v1.npz" % ds)
names, NM = rec["rules"], len(rec["budgets"])
cells = sorted({int(c.split("|")[0]) for c in rec["cells"]})
rows = z["rows"]
A, B = (rows % 2) == 0, (rows % 2) == 1
CAND, CON, AGG = "Sro+Sri+Tin+BM", "Sro+Sri+Tin+CRD", "T+CRD"
TYPED = [n for n in names if n != "SHIPPED" and not n.startswith("PROT")]


def arr(K, rt, nm):
    return np.unpackbits(z["ALL__%d|%s" % (K, rt)], axis=2)[:, :, :NM][names.index(nm)].astype(bool)


print("%s  budgets %s  rows A/B %d/%d" % (ds, rec["budgets"], A.sum(), B.sum()))
print("\nALL-gold, ES_m router, ALL rows (A+B), per K and B_N")
for nm in (CAND, CON, AGG):
    for K in cells:
        print("  %-16s K%-5d %s" % (nm, K, [round(float(arr(K, "ES_m", nm)[:, m].mean()), 3) for m in range(NM)]))
print("\nper-row ORACLE (upper bound of any per-query selection), half B, mean over cells, B_N 100..1000 (the objective) | and at all six B_N")
def objm(get, half):
    return float(np.mean([get(K)[half][:, :4].mean() for K in cells]))
base = objm(lambda K: arr(K, "ES_m", CAND), B)
for label, get in [("candidate (BM)", lambda K: arr(K, "ES_m", CAND)), ("conservative +CRD", lambda K: arr(K, "ES_m", CON)), ("aggressive T+CRD", lambda K: arr(K, "ES_m", AGG)),
                   ("ORACLE{CON,AGG}", lambda K: arr(K, "ES_m", CON) | arr(K, "ES_m", AGG)),
                   ("ORACLE{CAND,CON,AGG}", lambda K: arr(K, "ES_m", CAND) | arr(K, "ES_m", CON) | arr(K, "ES_m", AGG)),
                   ("ORACLE all typed arms", lambda K: np.any([arr(K, "ES_m", n) for n in TYPED], axis=0))]:
    o = objm(get, B)
    six = [round(float(np.mean([get(K)[B][:, m].mean() for K in cells])), 3) for m in range(NM)]
    print("  %-22s objB %.4f (%+.4f vs cand)   six B_N %s" % (label, o, o - base, six))
nrow_aggwin = {K: (int((arr(K, "ES_m", AGG) & ~arr(K, "ES_m", CON))[B][:, :4].sum()), int((arr(K, "ES_m", CON) & ~arr(K, "ES_m", AGG))[B][:, :4].sum())) for K in cells}
print("\n(row,B_N<=1000) cells where AGG alone is right / CON alone is right (half B):", nrow_aggwin)
if "hopB" in next(iter(rec["table"].values()))[CAND]:
    print("\nper-hop half B at the six B_N, ES_m")
    for K in cells:
        for nm in (CAND, CON, AGG):
            e = rec["table"]["%d|ES_m" % K][nm]["hopB"]
            print("  K%-5d %-16s %s" % (K, nm, {h: [round(x, 3) for x in v] for h, v in e.items()}))
