"""Read-only report on results/L1_X/rrt_<ds>__<tag>.{json,npz}: shipped vs one rule, half B (odd row id), per routed K cell and B_N, with paired gain/loss counts and an exact two-sided McNemar p.
usage: python _l1x_rrt_report.py <tag> <rule> ds [ds ...]"""
import json
import os
import sys
import numpy as np
from math import comb

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "L1_X")
BN = (100, 250, 500, 1000, 2000, 5000)


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2 * p)


def main():
    tag, rule = sys.argv[1], sys.argv[2]
    for ds in sys.argv[3:]:
        rec = json.load(open(os.path.join(OUT, "rrt_%s__%s.json" % (ds, tag)), encoding="utf-8"))
        z = np.load(os.path.join(OUT, "rrt_%s__%s.npz" % (ds, tag)))
        rules = rec["rules"]
        ri, si = rules.index(rule), rules.index("SHIPPED")
        rows = z["rows"]
        B = (rows % 2) == 1
        print("== %s  rule %s  (half B n=%d)" % (ds, rule, int(B.sum())))
        print("  cell    B_N   SHIPPED   RULE    delta   +gain/-loss    McNemar p")
        for c in rec["cells"] if "cells" in rec else []:
            key = "ALL__%s" % ("UNR" if str(c) == "UNR" else c)
            if key not in z.files:
                continue
            nq = len(rows)
            A = np.unpackbits(z[key], axis=2)[:, :nq, :len(BN)] if z[key].shape[2] * 8 >= len(BN) else None
            for j, bn in enumerate(BN):
                s = A[si, B, j].astype(bool)
                r = A[ri, B, j].astype(bool)
                g, l = int((r & ~s).sum()), int((s & ~r).sum())
                print("  %-6s %5d  %.4f   %.4f  %+.4f   +%3d/-%3d     %.2g" % (c, bn, s.mean(), r.mean(), r.mean() - s.mean(), g, l, mcnemar(g, l)))
        print()


if __name__ == "__main__":
    main()
