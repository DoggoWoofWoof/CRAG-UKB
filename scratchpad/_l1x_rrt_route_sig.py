"""Read-only: half-B paired comparison (router, rule) vs (ES, SHIPPED), MetaQA (routed2_metaqa__v2) and WebQSP (rrtroute_webqsp__<tag>), with exact McNemar.
  python _l1x_rrt_route_sig.py <wq tag> <router> <rule>"""
import json
import os
import sys
from math import comb
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "L1_X")
BN = (100, 250, 500, 1000, 2000, 5000)


def mcn(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2.0 ** n)


def cmp(tag_ds, a, b, B, K):
    print("  K %-5s  B_N  base   new    delta   +gain/-loss  p" % K)
    for j, bn in enumerate(BN):
        s, r = a[B, j].astype(bool), b[B, j].astype(bool)
        g, l = int((r & ~s).sum()), int((s & ~r).sum())
        print("          %5d %.3f  %.3f  %+.3f   +%3d/-%3d   %.2g" % (bn, s.mean(), r.mean(), r.mean() - s.mean(), g, l, mcn(g, l)))


def main():
    tag, rv, rule = sys.argv[1:4]
    mj = json.load(open(os.path.join(OUT, "routed2_metaqa__v2.json"), encoding="utf-8"))
    mz = np.load(os.path.join(OUT, "routed2_metaqa__v2.npz"))
    B = (mz["rows"] % 2) == 1
    print("== metaqa half B n=%d: (%s, %s) vs (ES, SHIPPED)" % (int(B.sum()), rv, rule))
    for K in (100, 250, 432, 500, 1000):
        cb = "PHG_k%d" % K
        ca = cb if rv == "ES" else "%s@%s" % (cb, rv)
        cmp("metaqa", mz["ALL__" + cb][mj["rules"].index("SHIPPED")], mz["ALL__" + ca][mj["rules"].index(rule)], B, K)
    wj = json.load(open(os.path.join(OUT, "rrtroute_webqsp__%s.json" % tag), encoding="utf-8"))
    wz = np.load(os.path.join(OUT, "rrtroute_webqsp__%s.npz" % tag))
    B = (wz["rows"] % 2) == 1
    nq = len(wz["rows"])
    print("== webqsp half B n=%d: (%s, %s) vs (ES, SHIPPED)" % (int(B.sum()), rv, rule))
    for K in (100, 250, 500):
        up = lambda key: np.unpackbits(wz[key], axis=2)[:, :nq, :len(BN)]
        cmp("webqsp", up("ALL__%d__ES" % K)[wj["rules"].index("SHIPPED")], up("ALL__%d__%s" % (K, rv))[wj["rules"].index(rule)], B, K)


if __name__ == "__main__":
    main()
