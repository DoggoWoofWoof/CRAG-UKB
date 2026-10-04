"""L1 DEVELOPMENT -- ADAPTBP: the query-adaptive partition fan-out B_P(q) of the L1b routing (user decisions 2026-09-27,
answering REPORT section 32.7).  Development numbers; any p-value is descriptive.

The decisions this harness implements:
  (1) the L1a node score is IR_L1 on every dataset -- no dataset branch, no target-degree term:
          L(u | q) = sum_{s in H_q} (1 / rank(s)) sum_f A_f(s, u) / log2(1 + deg_f^+(s)),      H_q = FLAT_RRF[:200],
      over the frozen universal edge union E = STRUCT_out u STRUCT_in u KNN u NER (_l1d_node1h.build_families, read-only;
      the arithmetic of _l1d_node1h's IR_L1 rung, asserted identical to its v1 records);
  (2) B_P is neither one global constant nor c * B_N: it follows the concentration of the partition scores of the query,
          a_j = Agg_{u in P_j} L(u | q),   p_j = a_j / sum_k a_k,   N_eff(q) = 1 / sum_j p_j^2,   B_P(q) = ceil(N_eff(q))
      (NEFF, the participation ratio); the alternative B_P(q) = ceil(exp H(P | q)), H = -sum_j p_j log p_j, is measured
      alongside (EXPH).  Both are <= the number of partitions with a_j > 0.  An external cap B_P^max is a deployment
      resource constraint, not a retrieval hyperparameter: min(B_P^max, B_P(q)) is tabulated for CAPS, never selected.
      A query with sum_j a_j = 0 (no localization evidence; none in the development population) contacts every partition.
  (3) one aggregation rule for every dataset, to be chosen on development evidence before the final evaluation:
          SUM   a_j = sum of L over P_j                (== _l1d_node1h LSUM)
          MAX   a_j = max of L over P_j                (== _l1d_node1h LMAX)
          TOPk  a_j = sum of the k largest L in P_j    (the user's starting candidate, "a tiny fixed k": k = 3, fixed before
                                                        any run; k = 2, 5, 10 are a declared sensitivity band, not a search)
      The rule ranks the partitions (descending a_j; ties and a_j = 0 fall back to the first appearance in O, as in
      _l1d_node1h) and its own p_j give N_eff.  The served order O is the IR_L1 FLAT+LOC order of section 32 (RRF, K0 = 60):
      the first B_N nodes of O inside the contacted partitions are returned (fewer when they hold fewer than B_N nodes).
  (4) surfaces R(B_P, B_N), not only R(M): for every ranking, EVERY integer B_P = 1 .. |partitions| and every B_N in M_CURVE,
      stored per query as the interval [LO, HI(B_N)] of fan-outs at which ALL its gold nodes are returned (the routed rank of a
      contacted gold only grows with B_P, so the ALL-served fan-outs of a query form one interval).

Arms per cell, "<ranking>|<aggregation>|<concentration>":
  OWN|<agg>|NEFF, OWN|<agg>|EXPH   the candidates: the aggregation ranks the partitions and sets the count;
  S|<agg>|NEFF,   S|<agg>|EXPH     reference: partitions by first appearance in O (section 32's best text ranking) with the
                                   aggregation's adaptive count ("which" from the served order, "how many" from L);
  every fixed B_P of every ranking (the surfaces), and unrouted (B_P = every partition).
Gold-dependent diagnostic (never a rule): LO, the fan-out that contacts every gold partition of the query under a ranking, and
its rank correlation with N_eff.

Identities asserted: FLAT == loc_<ds>__v1 (runner); IR_L1 LOC / FLAT+LOC positions == node1h_<ds>__v1.npz; the routed rank of
every gold under S, MAX, SUM at B_P in (1, 2, 5, 10, 20, 50, 100) and the contacted masses == node1h's route_rr / route_mass
(IR_L1|S, IR_L1|LMAX, IR_L1|LSUM); contacting every partition == the unrouted FLAT+LOC arm (per row).
No learned weight, no tuned constant, no traversal: one sparse product per query, then per-partition sums.

Usage: python scratchpad/_l1d_adaptbp.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/adaptbp_<dataset>__<tag>.{json,npz} (write-once)
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_arms as R
import _l1d_edgediag as E
import _l1d_node1h as NH

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = NH.FAMS
NODE = "IR_L1"
TOPK_PRIMARY = 3
TOPKS = (2, 3, 5, 10)
AGGS = ("SUM", "MAX") + tuple("TOP%d" % k for k in TOPKS)
PRIMARY_AGGS = ("SUM", "MAX", "TOP%d" % TOPK_PRIMARY)
RANKS = ("S",) + AGGS
CONCS = ("NEFF", "EXPH")
ARMS = [(rr, a, cc) for rr in ("OWN", "S") for a in AGGS for cc in CONCS]
ARM_NAMES = ["%s|%s|%s" % x for x in ARMS]
CAPS = (5, 10, 20, 50, 100)
BP_GRID = NH.BP_GRID
SENT = int(NH.SENT)
ID_CFG = (("S", "S"), ("MAX", "LMAX"), ("SUM", "LSUM"))       # this harness's ranking -> _l1d_node1h's IR_L1 route config
BP_SHOW = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500)
USER_POINTS = ((5, 500), (20, 500), (20, 2000), (100, 1000), (None, 1000))      # the user's example rows; None = unrestricted
TOL = 1e-9
NH_PATH = os.path.join(D.HERE, "_l1d_node1h.py")
E_PATH = os.path.join(D.HERE, "_l1d_edgediag.py")
MSA = np.asarray(D.M_CURVE, np.int64)
assert set(PRIMARY_AGGS) <= set(AGGS) and NH.FAMS == ("STRUCT_out", "STRUCT_in", "KNN", "NER")


def cumcounts(prO, pg, npart):
    """C[k, b - 1] = the nodes of O[:pg[k] + 1] whose partition has routing position < b, for b = 1 .. npart (the incremental
    bincount of _l1d_node1h.restricted_ranks, uncapped)."""
    out = np.empty((len(pg), npart), np.int64)
    cum = np.zeros(npart, np.int64)
    prev = 0
    for k in np.argsort(pg, kind="stable"):
        p = int(pg[k])
        if p >= prev:
            cum += np.bincount(prO[prev:p + 1], minlength=npart)
            prev = p + 1
        out[k] = np.cumsum(cum)
    return out


def bp_of(x, npos, npart):
    """ceil of a concentration (NEFF or EXPH), within [1, the partitions with a_j > 0]; no evidence -> every partition."""
    if npos == 0:
        return npart
    assert 1.0 - TOL <= x <= npos * (1.0 + TOL) + TOL, (x, npos)
    return int(min(max(int(np.ceil(x - TOL)), 1), npos))


def surface(LO, HI, npart, mask=None):
    """ALL(b, B_N) for b = 1 .. npart and B_N in M_CURVE from the per-query ALL-served intervals [LO, HI(B_N)]."""
    m = np.ones(len(LO), bool) if mask is None else mask
    out = np.zeros((len(D.M_CURVE), npart))
    n = int(m.sum())
    for mi in range(len(D.M_CURVE)):
        lo, hi = LO[m], HI[m, mi]
        ok = lo <= hi
        d = np.zeros(npart + 2)
        np.add.at(d, lo[ok], 1.0)
        np.add.at(d, hi[ok] + 1, -1.0)
        out[mi] = np.cumsum(d)[1:npart + 1] / max(n, 1)
    return out


def served_at(LO, HI_m, b):
    return (LO <= b) & (b <= HI_m)


def rankavg(x):
    x = np.asarray(x, np.float64)
    ux, inv = np.unique(x, return_inverse=True)
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=np.float64)
    return (np.bincount(inv, weights=r) / np.bincount(inv))[inv]


def spearman(x, y):
    a, b = rankavg(x), rankavg(y)
    if a.std() == 0 or b.std() == 0:
        return None
    return D.q4(np.corrcoef(a, b)[0, 1])


def pct(x, q):
    return float(np.percentile(np.asarray(x, np.float64), q))


def dist(x):
    x = np.asarray(x, np.float64)
    return {"mean": round(float(x.mean()), 2), "median": float(np.median(x)), "p10": pct(x, 10), "p90": pct(x, 90),
            "min": float(x.min()), "max": float(x.max())}


class AdaptSpec(object):
    def __init__(self, cd, pop, parts):
        self.ds, self.parts, self.cells = cd.name, parts, list(parts)
        N = self.N = int(cd.n_nodes)
        nq, ng = pop.nq, pop.ng_tot
        # ---- the section-32 records this harness must reproduce (same code, same population, same partitions)
        fr = os.path.join(D.OUT, "node1h_%s__v1.json" % self.ds)
        R1 = json.load(open(fr, encoding="utf-8"))
        self.node1h_json = fr
        self.node1h_npz = os.path.join(D.OUT, R1["npz"]["path"])
        assert D.sha_file(self.node1h_npz) == R1["npz"]["sha256"], "node1h npz changed"
        self.sha_nh, self.sha_e = D.sha_file(NH_PATH), D.sha_file(E_PATH)
        assert self.sha_nh == R1["code"]["harness"]["sha256"], "_l1d_node1h.py differs from the code of its v1 record"
        assert self.sha_e == R1["structures"]["imports"]["scratchpad/_l1d_edgediag.py"], "_l1d_edgediag.py differs"
        assert R1["code"]["arms"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_arms.py"))
        assert R1["code"]["lib"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))
        rc = R1["structures"]["routing"]["cells"]
        assert sorted(rc) == sorted(self.cells)
        for c, P in parts.items():
            assert rc[c]["partition"] == P.tag and rc[c]["npart"] == P.npart and P.npart >= max(BP_GRID)
        self.F, famrec = NH.build_families(cd, N)
        self.names = [NODE]
        self.hard_ref = {NODE: None}
        self.v1_hard = {}
        self.lat = {NODE: []}
        self.FUSE_LAT, self.EVAL_LAT = [], {c: [] for c in self.cells}
        self.SEL_LAT = {c: [] for c in self.cells}
        self.FRK = np.zeros(ng, np.int64)
        nr, na, nm = len(RANKS), len(ARMS), len(D.M_CURVE)
        self.LO = {c: np.zeros((nq, nr), np.int32) for c in self.cells}
        self.HI = {c: np.zeros((nq, nr, nm), np.int32) for c in self.cells}
        self.MSUM = {c: np.zeros((nr, P.npart)) for c, P in parts.items()}
        self.RET = {c: np.zeros((nr, nm, P.npart)) for c, P in parts.items()}
        self.BPQ = {c: np.zeros((nq, na), np.int32) for c in self.cells}
        self.MASSA = {c: np.zeros((nq, na), np.int64) for c in self.cells}
        self.RRA = {c: np.full((ng, na), SENT, np.int32) for c in self.cells}
        self.CONC = {c: np.zeros((nq, len(AGGS), 3)) for c in self.cells}           # N_eff, exp H, partitions with a_j > 0
        self.RRID = {c: np.full((ng, len(ID_CFG), len(BP_GRID)), SENT, np.int64) for c in self.cells}
        self.MSID = {c: np.zeros((nq, len(ID_CFG), len(BP_GRID)), np.int64) for c in self.cells}
        self.NOEV = {c: 0 for c in self.cells}
        self.record = {
            "node_score": "IR_L1: L(u) = sum_{s in FLAT[:200]} (1 / rank(s)) sum_f A_f(s, u) / log2(1 + deg_f^+(s)); E = STRUCT_out u "
                          "STRUCT_in u KNN u NER (the section-32 rung, recomputed and asserted identical)",
            "served_order": "IR_L1 FLAT+LOC (RRF K0 = 60); the first B_N nodes of it inside the contacted partitions",
            "aggregations": {"SUM": "sum of L over the partition", "MAX": "max of L over the partition",
                             "TOPk": "sum of the k largest L in the partition; k = %d primary (fixed a priori), %s sensitivity" % (
                                 TOPK_PRIMARY, [k for k in TOPKS if k != TOPK_PRIMARY])},
            "primary_aggregations": list(PRIMARY_AGGS),
            "concentrations": {"NEFF": "B_P(q) = ceil(1 / sum_j p_j^2)", "EXPH": "B_P(q) = ceil(exp(-sum_j p_j ln p_j))",
                               "p_j": "a_j / sum_k a_k over the partitions of the cell", "ceil tolerance": TOL,
                               "no evidence (sum a_j = 0)": "contact every partition"},
            "rankings": {"S": "first appearance in the served order O", "<agg>": "descending a_j, ties and a_j = 0 by first appearance in O"},
            "arms": ARM_NAMES, "caps (deployment table only)": list(CAPS),
            "families": famrec, "cells": {c: {"partition": P.tag, "npart": P.npart, "partition_size": D.stats(P.sizes)} for c, P in parts.items()},
            "reproduces": {"record": {"path": D.rel(fr), "sha256": D.sha_file(fr)}, "npz": {"path": D.rel(self.node1h_npz), "sha256": R1["npz"]["sha256"]}},
            "imports": {"scratchpad/_l1d_node1h.py": self.sha_nh, "scratchpad/_l1d_edgediag.py": self.sha_e}}

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        N = self.N
        top = of[:ACT]
        nh = len(top)
        # ---- L1a: the IR_L1 node score (the arithmetic of _l1d_node1h.row for this rung)
        t0 = time.perf_counter()
        U, H, GW = [], [], []
        for f in FAMS:
            Fm = self.F[f]
            st = Fm["xadj"][top]
            hit, pos = E.entries(st, Fm["xadj"][top + 1] - st)
            U.append(Fm["adj"][pos].astype(np.int64))
            H.append(hit)
            GW.append(Fm["g"][top][hit])
        u, hit, gw = np.concatenate(U), np.concatenate(H), np.concatenate(GW)
        x = (1.0 / np.arange(1.0, nh + 1.0))[hit]
        x = x * gw
        L = np.bincount(u, weights=x, minlength=N)
        Lord = D.order_from_score(L, frank)
        self.lat[NODE].append(time.perf_counter() - t0)
        # ---- the served order O = IR_L1 FLAT+LOC
        t0 = time.perf_counter()
        f = 1.0 / (K0 + frank.astype(np.float64))
        f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        FO = np.lexsort((frank, -f))
        frk = np.empty(N, np.int64)
        frk[FO] = np.arange(N)
        self.FUSE_LAT.append(time.perf_counter() - t0)
        pg = frk[g]
        self.FRK[sl] = pg
        lim = int(pg.max()) + 1
        nz = np.flatnonzero(L)
        Lv = L[nz]
        for c in self.cells:
            P = self.parts[c]
            npart = P.npart
            # ---- L1b selection (what a deployment computes): aggregates, rankings, concentrations
            t1 = time.perf_counter()
            fa = np.minimum.reduceat(frk[P.order_nodes], P.ptr[:-1])
            ph = P.hard[nz]
            A = {"SUM": np.bincount(ph, weights=Lv, minlength=npart)}
            mx = np.zeros(npart)
            np.maximum.at(mx, ph, Lv)
            A["MAX"] = mx
            o = np.lexsort((-Lv, ph))
            phs, lvs = ph[o], Lv[o]
            wr = np.arange(len(phs)) - np.searchsorted(phs, phs, side="left")
            for k in TOPKS:
                m = wr < k
                A["TOP%d" % k] = np.bincount(phs[m], weights=lvs[m], minlength=npart)
            RO = {"S": np.argsort(fa, kind="stable")}
            BPC = {}
            for ai, a in enumerate(AGGS):
                RO[a] = np.lexsort((fa, -A[a]))
                v = A[a][A[a] > 0]
                if len(v) == 0:
                    ne = eh = float(npart)
                else:
                    p = v / v.sum()
                    ne = 1.0 / float(np.dot(p, p))
                    eh = float(np.exp(-np.dot(p, np.log(p))))
                self.CONC[c][j, ai] = (ne, eh, len(v))
                BPC[(a, "NEFF")] = bp_of(ne, len(v), npart)
                BPC[(a, "EXPH")] = bp_of(eh, len(v), npart)
            if len(nz) == 0:
                self.NOEV[c] += 1
            self.SEL_LAT[c].append(time.perf_counter() - t1)
            # ---- evaluation (gold-dependent; not part of the rule): every fan-out of every ranking
            t1 = time.perf_counter()
            hO, hg = P.hard[FO[:lim]], P.hard[g]
            for ri, rk in enumerate(RANKS):
                ro = RO[rk]
                pr = np.empty(npart, np.int64)
                pr[ro] = np.arange(npart)
                prg = pr[hg]
                C = cumcounts(pr[hO], pg, npart)
                assert (C[:, npart - 1] - 1 == pg).all(), "every partition contacted != unrouted (%s %s)" % (c, rk)
                ms = np.cumsum(P.sizes[ro])
                lo = int(prg.max()) + 1
                qmax = (C[:, lo - 1:] - 1).max(axis=0)
                assert (np.diff(qmax) >= 0).all()
                self.LO[c][j, ri] = lo
                self.HI[c][j, ri] = lo - 1 + np.searchsorted(qmax, MSA, side="left")
                self.MSUM[c][ri] += ms
                self.RET[c][ri] += np.minimum(MSA[:, None], ms[None, :])
                for ii, (rk_id, _) in enumerate(ID_CFG):
                    if rk_id == rk:
                        for bi, BP in enumerate(BP_GRID):
                            self.RRID[c][sl, ii, bi] = np.where(prg < BP, C[:, BP - 1] - 1, SENT)
                            self.MSID[c][j, ii, bi] = ms[BP - 1]
                for ai, (rr, a, cc) in enumerate(ARMS):
                    if (rr == "OWN" and a == rk) or (rr == "S" and rk == "S"):
                        b = BPC[(a, cc)]
                        self.BPQ[c][j, ai] = b
                        self.MASSA[c][j, ai] = ms[b - 1]
                        self.RRA[c][sl, ai] = np.where(prg < b, C[:, b - 1] - 1, SENT)
            self.EVAL_LAT[c].append(time.perf_counter() - t1)
        return {NODE: Lord}

    def finish(self, ctx):
        POS, gptr, ngold, pop, ST = ctx["POS"], ctx["gptr"], ctx["ngold"], ctx["pop"], ctx["ST"]
        nq, ng = pop.nq, pop.ng_tot
        assert D.sha_file(NH_PATH) == self.sha_nh and D.sha_file(E_PATH) == self.sha_e, "imported code changed during the run"
        allv = lambda pos, M: D.per_query(pos, M, gptr, ngold)[0]
        diag, arrays = {}, {}
        # (0) identities against the section-32 records
        z = np.load(self.node1h_npz)
        assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == gptr).all()
        assert (z["pos_LOC__" + NODE][:ng] == POS[NODE]["LOC"]).all(), "IR_L1 LOC positions differ from node1h v1"
        assert (z["pos_FLATLOC__" + NODE][:ng] == POS[NODE]["FLAT+LOC"]).all(), "IR_L1 FLAT+LOC positions differ from node1h v1"
        assert (self.FRK == POS[NODE]["FLAT+LOC"]).all()
        cfg = [str(s) for s in z["route_cfg"]]
        assert [int(b) for b in z["route_bp"]] == list(BP_GRID)
        for c in self.cells:
            for ii, (rk, nm_) in enumerate(ID_CFG):
                ci = cfg.index("%s|%s" % (NODE, nm_))
                assert (z["route_rr__" + c][:ng, ci, :].astype(np.int64) == self.RRID[c][:, ii, :]).all(), "routed ranks differ (%s %s)" % (c, rk)
                assert (z["route_mass__" + c][:nq, ci, :] == self.MSID[c][:, ii, :]).all(), "contacted mass differs (%s %s)" % (c, rk)
        diag["identity"] = ("IR_L1 LOC and FLAT+LOC positions == %s on all %d gold nodes; routed ranks of every gold and contacted "
                            "masses under S / MAX / SUM at B_P %s == its route_rr / route_mass for IR_L1|S / LMAX / LSUM in every cell; "
                            "every partition contacted == unrouted (per row, per ranking)" % (D.rel(self.node1h_npz), ng, list(BP_GRID)))
        hop_m = ST.get("per_hop", {})
        unr = {M: allv(POS[NODE]["FLAT+LOC"], M) for M in D.M_CURVE}
        surf, adap, orc, conc = {}, {}, {}, {}
        for c in self.cells:
            P = self.parts[c]
            npart = P.npart
            LO, HI = self.LO[c].astype(np.int64), self.HI[c].astype(np.int64)
            # (1) surfaces R(B_P, B_N) per ranking
            SF = {}
            surf[c] = {}
            for ri, rk in enumerate(RANKS):
                S_ = surface(LO[:, ri], HI[:, ri], npart)
                SF[rk] = S_
                mass_mean = self.MSUM[c][ri] / nq
                e = {"contacted_mass_mean": {str(b): round(float(mass_mean[b - 1]), 1) for b in BP_SHOW if b <= npart}}
                e["contacted_mass_mean"]["all (%d)" % npart] = round(float(mass_mean[-1]), 1)
                for mi, M in enumerate(D.M_CURVE):
                    row_ = S_[mi]
                    assert abs(row_[-1] - unr[M].mean()) < 1e-12, "surface at every partition != unrouted (%s %s %d)" % (c, rk, M)
                    best = float(row_.max())
                    bstar = int(np.argmax(row_)) + 1
                    b01 = int(np.flatnonzero(row_ >= best - 0.01)[0]) + 1
                    em = {"ALL_at_B_P": {str(b): D.q4(row_[b - 1]) for b in BP_SHOW if b <= npart},
                          "unrouted_ALL": D.q4(row_[-1]), "best_fixed": {"B_P": bstar, "ALL": D.q4(best)},
                          "smallest_B_P_within_0.01_of_best": b01,
                          "smallest_B_P_within_0.01_of_unrouted": int(np.flatnonzero(row_ >= row_[-1] - 0.01)[0]) + 1,
                          "returned_nodes_mean_at_B_P": {str(b): round(float(self.RET[c][ri, mi, b - 1] / nq), 1) for b in BP_SHOW if b <= npart}}
                    if hop_m:
                        em["per_hop"] = {}
                        for hk, qm in hop_m.items():
                            Sh = surface(LO[:, ri], HI[:, ri], npart, qm)
                            em["per_hop"][hk] = {"n": int(qm.sum()), "ALL_at_B_P": {str(b): D.q4(Sh[mi, b - 1]) for b in BP_SHOW if b <= npart},
                                                 "unrouted_ALL": D.q4(Sh[mi, -1]), "best_fixed": {"B_P": int(np.argmax(Sh[mi])) + 1, "ALL": D.q4(Sh[mi].max())}}
                    e[str(M)] = em
                e["user_points"] = {}
                for bp, M in USER_POINTS:
                    if M in D.M_CURVE:
                        b = npart if bp is None else min(bp, npart)
                        e["user_points"]["(%s, %d)" % ("unrestricted" if bp is None else bp, M)] = D.q4(S_[D.M_CURVE.index(M), b - 1])
                surf[c][rk] = e
            # (2) adaptive arms
            adap[c] = {}
            for ai, (rr, a, cc) in enumerate(ARMS):
                rk = a if rr == "OWN" else "S"
                ri = RANKS.index(rk)
                b = self.BPQ[c][:, ai].astype(np.int64)
                ranks = self.RRA[c][:, ai].astype(np.int64)
                e = {"ranking": rk, "B_P(q)": dist(b), "contacted_mass": dist(self.MASSA[c][:, ai]),
                     "at_cap": {str(cp): D.q4((b > cp).mean()) for cp in CAPS}}
                bm = float(b.mean())
                bl, bh = int(np.floor(bm)), int(np.ceil(bm))
                br = int(np.floor(bm + 0.5))
                for mi, M in enumerate(D.M_CURVE):
                    a_, y_, f_, _ = D.per_query(ranks, M, gptr, ngold)
                    assert (a_ == served_at(LO[:, ri], HI[:, ri, mi], b)).all(), "adaptive ALL != its interval (%s %s %d)" % (c, ARM_NAMES[ai], M)
                    row_ = SF[rk][mi]
                    interp = row_[bl - 1] + (bm - bl) * (row_[bh - 1] - row_[bl - 1])
                    fixed_r = served_at(LO[:, ri], HI[:, ri, mi], br)
                    bstar = int(np.argmax(row_)) + 1
                    em = {"ALL": D.q4(a_.mean()), "ANY": D.q4(y_.mean()), "FRAC": D.q4(f_.mean()),
                          "returned_nodes_mean": round(float(np.minimum(M, self.MASSA[c][:, ai]).mean()), 1),
                          "unrouted_ALL": D.q4(unr[M].mean()), "paired_vs_unrouted": D.paired(unr[M], a_),
                          "fixed_at_mean_fan_out": {"mean_B_P": round(bm, 2), "ALL_interpolated": D.q4(interp), "B_P_rounded": br,
                                                    "ALL_at_rounded": D.q4(fixed_r.mean()), "paired (fixed -> adaptive)": D.paired(fixed_r, a_)},
                          "best_fixed": {"B_P": bstar, "ALL": D.q4(row_.max()),
                                         "paired (best fixed -> adaptive)": D.paired(served_at(LO[:, ri], HI[:, ri, mi], bstar), a_)},
                          "capped": {str(cp): D.q4(served_at(LO[:, ri], HI[:, ri, mi], np.minimum(b, cp)).mean()) for cp in CAPS}}
                    if hop_m:
                        em["per_hop"] = {hk: {"n": int(qm.sum()), "ALL": D.q4(a_[qm].mean()), "unrouted_ALL": D.q4(unr[M][qm].mean()),
                                              "fixed_at_rounded_mean_fan_out_ALL": D.q4(fixed_r[qm].mean()),
                                              "mean_B_P": round(float(b[qm].mean()), 2)} for hk, qm in hop_m.items()}
                    e[str(M)] = em
                e["capped_mean_B_P"] = {str(cp): round(float(np.minimum(b, cp).mean()), 2) for cp in CAPS}
                if hop_m:
                    e["B_P(q)_per_hop"] = {hk: dist(b[qm]) for hk, qm in hop_m.items()}
                adap[c][ARM_NAMES[ai]] = e
            # (3) gold-dependent diagnostic: the fan-out that contacts every gold partition (LO), servability, and N_eff vs LO
            orc[c] = {}
            for ri, rk in enumerate(RANKS):
                e = {"LO (fan-out contacting every gold partition)": dist(LO[:, ri]),
                     "servable_at_some_B_P": {str(M): D.q4((LO[:, ri] <= HI[:, ri, mi]).mean()) for mi, M in enumerate(D.M_CURVE)}}
                if rk in AGGS:
                    ai_n = AGGS.index(rk)
                    e["spearman(N_eff, LO)"] = spearman(self.CONC[c][:, ai_n, 0], LO[:, ri])
                    e["spearman(exp H, LO)"] = spearman(self.CONC[c][:, ai_n, 1], LO[:, ri])
                    for cc in CONCS:
                        aix = ARM_NAMES.index("OWN|%s|%s" % (rk, cc))
                        b = self.BPQ[c][:, aix].astype(np.int64)
                        e["%s: B_P(q) >= LO (all gold partitions contacted)" % cc] = D.q4((b >= LO[:, ri]).mean())
                        e["%s: over-contacted (served at some smaller B_P, lost at B_P(q))" % cc] = {
                            str(M): D.q4(((LO[:, ri] <= HI[:, ri, mi]) & (b > HI[:, ri, mi]) & (LO[:, ri] <= b)).mean()) for mi, M in enumerate(D.M_CURVE)}
                orc[c][rk] = e
            conc[c] = {a: {"N_eff": dist(self.CONC[c][:, ai, 0]), "exp_H": dist(self.CONC[c][:, ai, 1]),
                           "partitions_with_a_j>0": dist(self.CONC[c][:, ai, 2])} for ai, a in enumerate(AGGS)}
            conc[c]["queries_without_evidence"] = self.NOEV[c]
            arrays["LO__" + c] = self.LO[c]
            arrays["HI__" + c] = self.HI[c]
            arrays["BPQ__" + c] = self.BPQ[c]
            arrays["MASSA__" + c] = self.MASSA[c]
            arrays["RRA__" + c] = self.RRA[c]
            arrays["CONC__" + c] = self.CONC[c]
            arrays["MSUM__" + c] = self.MSUM[c]
            arrays["RET__" + c] = self.RET[c]
        diag["surfaces"] = surf
        diag["adaptive"] = adap
        diag["gold_fan_out_diagnostic"] = orc
        diag["concentration"] = conc
        diag["latency_ms"] = {"node_score_and_LOC_order": D.ms_stats(self.lat[NODE]), "fused_order": D.ms_stats(self.FUSE_LAT),
                              "selection (6 aggregations + 7 rankings + 12 concentrations, per cell)": {c: D.ms_stats(v) for c, v in self.SEL_LAT.items()},
                              "evaluation_only (every fan-out of 7 rankings, per cell)": {c: D.ms_stats(v) for c, v in self.EVAL_LAT.items()}}
        arrays["ranks"] = np.array(RANKS)
        arrays["arms"] = np.array(ARM_NAMES)
        arrays["aggs"] = np.array(AGGS)
        arrays["m_curve"] = MSA
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    R.run(sys.argv[2], sys.argv[3], "adaptbp", "L1_DEVELOPMENT_ADAPTIVE_BP", __file__, AdaptSpec, __doc__)


if __name__ == "__main__":
    main()
