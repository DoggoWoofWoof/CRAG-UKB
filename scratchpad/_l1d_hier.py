"""L1 DEVELOPMENT step 3 -- hierarchical regions (user ruling 2026-09-26, iteration order: 1 hard node-level LOC, 2 soft
memberships, 3 hierarchical regions, 4 alternative static incidence / activation).  Development numbers; any p-value is
descriptive.

Hierarchy (static, query-independent, no tuned constant).  The quotient graph of each frozen partition over the SK graph
(STRUCT u KNN, the edge families of the H4_SK hypergraph the partitions were cut from): W(a, b) = number of SK edges between
blocks a != b.  It is coarsened by deterministic heavy-edge matching, level after level: regions are visited in ascending
(node count, id) order; an unmatched region is matched to its unmatched neighbour of largest W (ties -> smaller id), or stays
single when it has none; merged regions sum W and node counts.  Level 0 = the frozen blocks; region sizes about double per
level; coarsening stops when no merge is possible.  The levels scored are 0 .. K, K = the first level whose median region
node count is >= MMAX (a region as large as the whole budget curve).
    w(v)     fv(v) for v in FLAT_RRF[:200], else 0            (the step-1 activation weights)
    A_l(R)   sum of w over the region R of level l;   L_l(u) = A_l(R_l(u))
Variants (per cell = one frozen partition)
    HARD     L = L_0                          the step-1 order (asserted == the v1 record)
    LEV<l>   L = L_l                          one granularity, l = 1 .. K
    FINE     (L_0, L_1, ..., L_K) descending, then FLAT rank: the step-1 order, then the unactivated blocks of activated
             level-1 regions, then of activated level-2 regions, ... (hierarchical back-off; == HARD wherever the HARD order
             is longer than M)
    TREE     (L_K, L_(K-1), ..., L_0) descending, then FLAT rank: best-first descent of the region tree (the heaviest level-K
             region, inside it the heaviest level-(K-1) region, ..., down to blocks)
    HDENS    L(u) = max_l A_l(R_l(u)) / |R_l(u)|, l = 0 .. K: the densest enclosing region (activation mass per node served)
    HRRF     f(u) = sum_l [L_l(u) > 0] / (K0 + rank of u in the level-l order), l = 0 .. K  (RRF over granularities)
Diagnostics per gold node (all levels, not only 0 .. K): the finest level whose region of the gold is activated, and the
quotient-graph distance from the gold's block to the nearest activated block; per level, the activated node mass (the LOC
order length of LEV<l>) and the share of gold nodes inside it -- the reach / cost curve of coarsening.

Modes: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]  ->  results/L1_DEV/hier_<dataset>__<tag>.{json,npz}
       LEVELS <dataset>   print the hierarchy only (no query read)
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1d_lib as D
import _l1d_arms as R

log = D.log
K0, ACT, MMAX = D.K0, D.ACT, D.MMAX
LV_HIST = 12                                   # histogram cap for the finest-activated-level diagnostic


def quotient(P, xadj, adj):
    """npart x npart symmetric CSR: W(a, b) = number of SK edges between blocks a != b (the CSR lists each edge from both ends)."""
    N = len(P.hard)
    src = np.repeat(np.arange(N, dtype=np.int64), np.diff(xadj))
    a, b = P.hard[src], P.hard[np.asarray(adj, np.int64)]
    m = a != b
    W = sp.csr_matrix((np.ones(int(m.sum()), np.float64), (a[m], b[m])), shape=(P.npart, P.npart))
    W.sum_duplicates()
    W.sort_indices()
    assert abs(W - W.T).max() == 0 if W.nnz else True
    return W


def hem_levels(W0, sizes0):
    """deterministic heavy-edge matching coarsening; returns [block -> region id at level l] for l = 0, 1, ..."""
    maps = [np.arange(W0.shape[0], dtype=np.int64)]
    W, sizes = W0.tocsr(), np.asarray(sizes0, np.int64).copy()
    matched = []
    while W.shape[0] > 1:
        n = W.shape[0]
        match = np.full(n, -1, np.int64)
        for r in np.lexsort((np.arange(n), sizes)):
            if match[r] >= 0:
                continue
            nb, wt = W.indices[W.indptr[r]:W.indptr[r + 1]], W.data[W.indptr[r]:W.indptr[r + 1]]
            ok = (match[nb] < 0) & (nb != r)
            if ok.any():
                nb, wt = nb[ok], wt[ok]
                c = int(nb[np.lexsort((nb, -wt))[0]])
                match[r], match[c] = c, r
            else:
                match[r] = r
        rep = np.minimum(np.arange(n), match)
        uniq, new = np.unique(rep, return_inverse=True)
        if len(uniq) == n:
            break
        matched.append(round(float((match != np.arange(n)).mean()), 4))
        C = sp.csr_matrix((np.ones(n), (np.arange(n), new)), shape=(n, len(uniq)))
        W = (C.T @ W @ C).tocsr()
        W.setdiag(0)
        W.eliminate_zeros()
        W.sort_indices()
        sizes = np.bincount(new, weights=sizes, minlength=len(uniq)).astype(np.int64)
        maps.append(new[maps[-1]].astype(np.int64))
    return maps, matched


class HierSpec(object):
    def __init__(self, cd, pop, parts, log_only=False):
        xadj, adj, grec = D.sk_csr(cd)
        self.parts, self.cells = parts, list(parts)
        self.H = {}
        self.record = {"graph": grec, "cells": {}}
        for c, P in parts.items():
            W0 = quotient(P, xadj, adj)
            maps, matched = hem_levels(W0, P.sizes)
            nodemaps = [m[P.hard].astype(np.int32) for m in maps]
            nreg = [int(m.max()) + 1 for m in maps]
            rsz = [np.bincount(nm, minlength=nr).astype(np.int64) for nm, nr in zip(nodemaps, nreg)]
            med = [float(np.median(s)) for s in rsz]
            K = next((l for l, x in enumerate(med) if x >= MMAX), len(maps) - 1)
            assert K >= 1, (c, med)
            self.H[c] = {"W0": W0, "maps": maps, "nodemaps": nodemaps, "nreg": nreg, "K": K, "rsz": rsz,
                         "inv_rsz": [1.0 / rsz[l][nodemaps[l]].astype(np.float64) for l in range(K + 1)]}
            self.record["cells"][c] = {"partition": P.tag, "npart": P.npart, "quotient_edges": int(W0.nnz // 2),
                                       "quotient_degree": D.stats(np.diff(W0.indptr)), "levels": len(maps), "K": K,
                                       "matched_share_per_coarsening": matched,
                                       "regions_per_level": nreg, "region_nodes_median_per_level": med,
                                       "region_nodes_max_per_level": [int(s.max()) for s in rsz]}
        del xadj, adj
        if log_only:
            return
        self.names, self.hard_ref, self.v1_hard = [], {}, {}
        for c in self.cells:
            vs = ["HARD"] + ["LEV%d" % l for l in range(1, self.H[c]["K"] + 1)] + ["FINE", "TREE", "HDENS", "HRRF"]
            for v in vs:
                n_ = "%s__%s" % (c, v)
                self.names.append(n_)
                self.hard_ref[n_] = None if v == "HARD" else "%s__HARD" % c
            self.v1_hard[c] = "%s__HARD" % c
        self.lat = {n_: [] for n_ in self.names}
        ng = pop.ng_tot
        self.nq = pop.nq
        self.LV = {c: np.full(ng, -1, np.int16) for c in self.cells}        # finest activated level of the gold's region
        self.QD = {c: np.full(ng, -1, np.int16) for c in self.cells}        # quotient distance to the nearest activated block
        self.MASS = {c: np.zeros((pop.nq, len(self.H[c]["maps"])), np.int64) for c in self.cells}   # activated node mass per level

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        top = of[:ACT]
        out = {}
        for c in self.cells:
            P, h = self.parts[c], self.H[c]
            K, maps, nodemaps = h["K"], h["maps"], h["nodemaps"]
            ub, A = D.activation(top, fv, P.hard)
            t0 = time.perf_counter()
            ords = [D.loc_order(ub, A, frank, P)]
            self.lat["%s__HARD" % c].append(time.perf_counter() - t0)
            # per-level region activation (sum of the block activations of the region's blocks; nested, so monotone in l)
            Areg = [np.bincount(maps[l][ub], weights=A, minlength=h["nreg"][l]) for l in range(len(maps))]
            gb = P.hard[g]
            lv = np.full(len(g), -1, np.int64)
            for l in range(len(maps) - 1, -1, -1):
                act = Areg[l] > 0
                lv[act[maps[l][gb]]] = l
                self.MASS[c][j, l] = int(h["rsz"][l][act].sum())
            self.LV[c][sl] = lv
            # quotient-graph BFS from the activated blocks
            dist = np.full(P.npart, -1, np.int64)
            dist[ub] = 0
            fr = np.zeros(P.npart, np.float64)
            fr[ub] = 1.0
            d_ = 0
            while fr.any():
                d_ += 1
                nb = (h["W0"] @ fr) > 0
                new = nb & (dist < 0)
                dist[new] = d_
                fr = new.astype(np.float64)
            self.QD[c][sl] = dist[gb]
            Ls = []
            for l in range(K + 1):
                Ls.append(Areg[l][nodemaps[l]])
            for l in range(1, K + 1):
                t0 = time.perf_counter()
                ords.append(D.order_from_score(Ls[l], frank))
                self.lat["%s__LEV%d" % (c, l)].append(time.perf_counter() - t0)
            for l in range(K + 1):
                out["%s__%s" % (c, "HARD" if l == 0 else "LEV%d" % l)] = ords[l]
            t0 = time.perf_counter()
            cand = np.flatnonzero(Ls[K] > 0)
            keys = [frank[cand]] + [-Ls[l][cand] for l in range(K, -1, -1)]
            out["%s__FINE" % c] = cand[np.lexsort(keys)]
            self.lat["%s__FINE" % c].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            keys = [frank[cand]] + [-Ls[l][cand] for l in range(K + 1)]
            out["%s__TREE" % c] = cand[np.lexsort(keys)]
            self.lat["%s__TREE" % c].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            dens = Ls[0] * h["inv_rsz"][0]
            for l in range(1, K + 1):
                np.maximum(dens, Ls[l] * h["inv_rsz"][l], out=dens)
            out["%s__HDENS" % c] = D.order_from_score(dens, frank)
            self.lat["%s__HDENS" % c].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            f = np.zeros(len(frank), np.float64)
            for l in range(K + 1):
                f[ords[l]] += 1.0 / (K0 + np.arange(len(ords[l]), dtype=np.float64))
            out["%s__HRRF" % c] = D.order_from_score(f, frank)
            self.lat["%s__HRRF" % c].append(time.perf_counter() - t0)
        return out

    def finish(self, ctx):
        POS_FLAT, POS, pop = ctx["POS_FLAT"], ctx["POS"], ctx["pop"]
        miss = POS_FLAT >= MMAX
        hops = np.repeat(pop.hops, pop.ngold)
        diag = {}
        arrays = {}
        for c in self.cells:
            h = self.H[c]
            silent = POS[self.v1_hard[c]]["lpos"] < 0                                  # HARD leaves the gold's block silent
            typeA = miss & silent
            LV, QD = self.LV[c], self.QD[c]

            def hist(x, m, cap):
                x = x[m]
                o = {"n": int(m.sum())}
                for k in range(0, cap):
                    o[str(k)] = int((x == k).sum())
                o["%d+" % cap] = int((x >= cap).sum())
                o["none"] = int((x < 0).sum())
                return o
            e = {"finest_activated_level_of_the_gold_region": {
                     "all_gold": hist(LV, np.ones(len(LV), bool), LV_HIST),
                     "FLAT@%d_missed" % MMAX: hist(LV, miss, LV_HIST),
                     "FLAT@%d_missed_and_HARD_silent (Type A)" % MMAX: hist(LV, typeA, LV_HIST)},
                 "quotient_distance_to_nearest_activated_block": {
                     "all_gold": hist(QD, np.ones(len(QD), bool), 5),
                     "FLAT@%d_missed" % MMAX: hist(QD, miss, 5),
                     "FLAT@%d_missed_and_HARD_silent (Type A)" % MMAX: hist(QD, typeA, 5)},
                 "Type_A_by_hop": {str(int(hp)): {"n": int((typeA & (hops == hp)).sum()),
                                                  "quotient_distance_1": int((typeA & (hops == hp) & (QD == 1)).sum()),
                                                  "finest_level_le_K": int((typeA & (hops == hp) & (LV >= 0) & (LV <= h["K"])).sum())}
                                   for hp in np.unique(hops)},
                 "per_level (activated node mass per query; share of gold nodes / of FLAT@MMAX-missed gold nodes inside it)": [
                     {"level": l, "regions": h["nreg"][l], "activated_node_mass": D.stats(self.MASS[c][:, l]),
                      "gold_inside": D.q4(((LV >= 0) & (LV <= l)).mean()),
                      "FLAT@%d_missed_gold_inside" % MMAX: D.q4(((LV >= 0) & (LV <= l))[miss].mean()) if miss.any() else None}
                     for l in range(len(h["maps"]))]}
            diag[c] = e
            arrays["gold_finest_level__" + c] = LV
            arrays["gold_qdist__" + c] = QD
            arrays["act_mass__" + c] = self.MASS[c]
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "LEVELS":
        ds = sys.argv[2]
        cd = D.AD.CanonicalDataset(ds)
        parts = {c: D.Part(cd, D.TAG_OF[c]) for c in D.CELLS[ds]}
        import json
        S = HierSpec(cd, None, parts, log_only=True)
        print(json.dumps(S.record, indent=1))
        return
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>] | LEVELS <dataset>"
    R.run(sys.argv[2], sys.argv[3], "hier", "L1_DEVELOPMENT_HIERARCHICAL_REGIONS", __file__, HierSpec, __doc__)


if __name__ == "__main__":
    main()
