"""FBX_SCALE stage 4C -- structure-only diagnostic: where do the pins of the coarse SK hypergraph sit?  (No gold label, no recall.)

  python -u scratchpad/_ml2_diag.py <cluster_map.npy> [<cluster_map.npy> ...]

For each clustering: the pins of the exact quotient of the UNSPLIT H4_SK nets, by family (STRUCT anchors / KNN anchors) and by coarse net size, next to the fine pin mass
by fine net size, so that the reason the pin reduction saturates is a measurement.
"""
import sys

import numpy as np

import _ml2_coarsen as C2
import _ml2_run as R
import _ml_coarsen as C

BUCKETS = [2, 3, 4, 5, 8, 16, 64, 256, 1024, 10 ** 9]


def bucket_table(sz, label):
    sz = np.asarray(sz)
    tot = int(sz.sum())
    lo = 2
    rows = []
    for hi in BUCKETS:
        m = (sz >= lo) & (sz < hi) if hi < 10 ** 9 else (sz >= lo)
        rows.append((lo, hi, int(m.sum()), int(sz[m].sum())))
        lo = hi
    print("  %s: nets %d, pins %d" % (label, len(sz), tot))
    for lo, hi, n, p in rows:
        print("    size [%d, %s): nets %9d (%5.1f %%)  pins %11d (%5.1f %%)" % (lo, "inf" if hi >= 10 ** 9 else hi, n, 100 * n / max(len(sz), 1), p, 100 * p / max(tot, 1)))


def main(paths):
    N, fams, cnt = R.graph_SK()
    eptr, eidx, ew = C2.sk_nets(N, fams)
    M = len(eptr) - 1
    nS = int((np.diff(fams[0][0]) >= 1).sum())            # family-0 anchors come first in sk_nets
    print("fine: N %d, nets %d (STRUCT-first %d anchors), pins %d" % (N, M, len(np.diff(fams[0][0])), len(eidx)))
    bucket_table(np.diff(eptr), "FINE unsplit")
    fam_of = np.zeros(M, np.int8)
    fam_of[len(np.diff(fams[0][0])):] = 1
    for p in paths:
        cl = np.load(p).astype(np.int64)
        Vc, ep, ei, ew2, vw, st = C.quotient(N, eptr, eidx, ew, np.ones(N, np.int64), cl)
        print("\n%s: clusters %d, coarse nets %d, coarse pins %d (%.3fx), dropped singleton nets %d, merged identical %d" % (
            p, Vc, st["M"], st["P"], len(eidx) / max(st["P"], 1), st["nets_dropped_singleton"], st["nets_merged_identical"]))
        bucket_table(np.diff(ep), "COARSE")
        pins_per_vertex = np.bincount(ei, minlength=Vc)
        print("  coarse pins per cluster: mean %.1f, max %d, top-10 clusters hold %.1f %% of the pins" % (pins_per_vertex.mean(), pins_per_vertex.max(),
                                                                                                     100 * np.sort(pins_per_vertex)[-10:].sum() / max(pins_per_vertex.sum(), 1)))
        top = np.argsort(np.diff(ep))[-5:][::-1]
        print("  five largest coarse nets: %s" % [int(x) for x in np.diff(ep)[top]])
        big = np.diff(ep) >= 64
        print("  nets >= 64 pins: %d holding %.1f %% of the pins; nets <= 8 pins: %d holding %.1f %%" % (
            int(big.sum()), 100 * np.diff(ep)[big].sum() / max(st["P"], 1), int((np.diff(ep) <= 8).sum()), 100 * np.diff(ep)[np.diff(ep) <= 8].sum() / max(st["P"], 1)))


if __name__ == "__main__":
    main(sys.argv[1:])
