"""L1 DEVELOPMENT diagnostic (gold-aware, never a candidate): the block ceiling of static partition localization.

Every step-1..4 variant scores a node through its block (L(u) = score of P(u)), so within the LOC arm a query's gold set
is complete at budget M only if every gold block has been served; with the blocks served in the best possible order (the
gold blocks first, smallest total first) the query is complete at M iff  sum over the distinct gold blocks B of |B| <= M.
    BLOCK_CEILING@M = share of queries whose distinct gold blocks hold <= M nodes in total
The same at the step-3 region levels (regions of level l instead of blocks), and the number of distinct gold blocks per
query by stratum.  Reads only the population's gold nodes and the frozen partitions; no query embedding, no product.

Usage: python scratchpad/_l1d_ceiling.py <tag>  ->  results/L1_DEV/ceiling_blocks__<tag>.json (write-once) + a markdown table."""
import json
import os
import sys

import numpy as np

import _l1d_lib as D
import _l1d_hier as H

log = D.log


def main():
    tag = sys.argv[1]
    fo = os.path.join(D.OUT, "ceiling_blocks__%s.json" % tag)
    assert not os.path.exists(fo), "write-once: %s exists" % fo
    res = {"mode": "L1_DEVELOPMENT_BLOCK_CEILING (gold-aware diagnostic; never a candidate)", "definitions": __doc__,
           "code": {"path": D.rel(os.path.abspath(__file__)), "sha256": D.sha_file(os.path.abspath(__file__)),
                    "hier": {"path": "scratchpad/_l1d_hier.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_hier.py"))},
                    "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))}},
           "datasets": {}}
    md = []
    for ds in ("metaqa", "musique", "squad"):
        cd = D.AD.CanonicalDataset(ds)
        pop = D.Population(cd)
        parts = {c: D.Part(cd, D.TAG_OF[c]) for c in D.CELLS[ds]}
        S = H.HierSpec(cd, None, parts, log_only=True)
        E = {"n_rows": pop.nq, "population": pop.record, "cells": {}}
        for c, P in parts.items():
            h = S.H[c]
            e = {"partition": P.tag, "levels": {}}
            for l in range(len(h["maps"])):
                rsz = h["rsz"][l]
                nm = h["nodemaps"][l]
                tot = np.zeros(pop.nq, np.int64)
                nb = np.zeros(pop.nq, np.int64)
                for j, g in enumerate(pop.golds):
                    ub = np.unique(nm[g])
                    tot[j] = int(rsz[ub].sum())
                    nb[j] = len(ub)
                ce = {str(M): D.q4((tot <= M).mean()) for M in D.M_CURVE}
                st = {sn: {k: {"n": int(m.sum()), "ceiling@1000": D.q4((tot[m] <= 1000).mean()),
                               "ceiling@%d" % D.MMAX: D.q4((tot[m] <= D.MMAX).mean()),
                               "distinct_gold_regions": D.stats(nb[m])} for k, m in masks.items() if m.any()}
                      for sn, masks in pop.ST.items()}
                e["levels"][str(l)] = {"regions": h["nreg"][l], "region_nodes_median": float(np.median(rsz)), "ceiling": ce,
                                       "distinct_gold_regions": D.stats(nb), "gold_region_nodes_total": D.stats(tot), "strata": st}
            E["cells"][c] = e
            l0 = e["levels"]["0"]
            md.append("| %s | %s | %s | %s |" % (c, " | ".join("%.4f" % l0["ceiling"][str(M)] for M in D.M_CURVE),
                                                 l0["distinct_gold_regions"]["mean"], l0["gold_region_nodes_total"]["median"]))
            log("%s: block ceiling %s" % (c, l0["ceiling"]))
        res["datasets"][ds] = E
    D.G.S.wj(fo, res)
    print("| cell | " + " | ".join("M=%d" % M for M in D.M_CURVE) + " | distinct gold blocks (mean) | gold-block nodes (median) |")
    print("|---|" + "---|" * (len(D.M_CURVE) + 2))
    print("\n".join(md))
    print("-> %s sha256 %s" % (fo, D.sha_file(fo)))


if __name__ == "__main__":
    main()
