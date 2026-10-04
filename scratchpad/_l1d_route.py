"""L1 DEVELOPMENT -- ROUTE: non-parametric L1b partition routers on the section-33 surfaces (user 2026-09-27, after REPORT
section 33: "exhaustively try out all these different ways of non parametric L1 where we are only dealing with partitions ...
identify only the right ones to bring ... avoid a model or learned method as much as possible", with a forwarded proposal of
evidence-coverage routing).  Development numbers; any p-value is descriptive.

Fixed (sections 32-33, asserted identical): the L1a node score IR_L1, the served order O (IR_L1 FLAT+LOC, RRF K0 = 60), the
cells and their frozen hard single-owner partitions (the shards), the population, and the serving rule -- the first B_N nodes
of O inside the B_P contacted partitions.  Only the partition ORDER (a ranking) and the partition COUNT (a stopping rule)
vary.  Every ranking and count is a fixed function of the query's own FLAT / IR_L1 evidence and the static shard map: no
learned weight, no tuned constant (K0 = 60, H_q = FLAT[:200] and k = 3 are the frozen values of sections 32-33), no gold.

Rankings (descending score; ties and zero scores fall back to the first appearance in O, as in sections 32-33):
  semantic   S       first appearance in O (max fused score; section-33 identity)
             F       first appearance in FLAT (block-max of the semantic score fv)
             D, SP   first appearance in the dense / SPLADE order alone (channel diagnostics)
             FSUM    sum of 1/rank(s) over the seeds s in H_q inside the partition     (seed mass, inverse-rank weights)
             FSUMFV  sum of fv(s) over the seeds s in H_q inside the partition         (seed mass, FLAT scores)
             FT3     sum of the 3 largest fv in the partition                           (block top-3 of the semantic score)
  localised  SUM, MAX, TOP3  of L over the partition (section-33 identities).  SUM == the greedy marginal NODE coverage of the
                     proposal, because the shards are disjoint (the literal greedy is asserted equal on the first rows)
             SDIV    sum of 1/rank(s) over the seeds with localised evidence in the partition   (distinct supporting seeds)
  fused      OT3     sum of the 3 largest fused scores in the partition
  coverage   the greedy maximum of F(S) = sum_{s in H_q} w(s) max_{P in S} phi(s, P), w(s) = 1/rank(s) (the IR_L1 hit weight):
             SC_L    phi = 1[P holds localised evidence of s]                    (binary seed cover)
             SC_LS   phi = 1[P holds localised evidence of s, or s itself]
             FL_L    phi = the share of s's localised mass that P holds         (facility location)
             FL_LS   phi(s, the partition of s) = 1, otherwise as FL_L
             The greedy takes the largest marginal gain (ties within 1e-9 relative -> first appearance in O) while it is
             positive, then the rest by first appearance in O.  Node-level facility location with phi = 1[u in P] is modular
             (== SUM) on disjoint shards, so it is not run separately.
  structure  PD      r0 + A_P^T D^-1 r0, r0 = SUM; A_P[P', P] = the entries of E from P' to P != P' (static), D = their row
                     sums.  One application of a static partition-level operator; FLAGGED: a second hop at shard granularity
                     (the 2026-09-27 ruling placed the node-level A^T(A^T x) in L3), run because the user asked for every
                     suggestion.
  fusion     PF_S_T3, PF_F_T3   partition-level RRF (K0 = 60, 0-based ranks) of S (or F) with TOP3
             IL_S_T3            round-robin interleave of S and TOP3, S first
             U_NEFF, U_KNEE     the union S[:c_F] u TOP3[:c_L] first (key min(rank_S / c_F, rank_TOP3 / c_L), ties -> O), with
                                (c_F, c_L) = (NEFF_FSUMFV, NEFF_TOP3) or (KNEE_FSUMFV, KNEE_TOP3); the union size is its count
Counts B_P(q), each in [1, |partitions|]; a query without evidence contacts every partition (as in section 33):
  NEFF_x = ceil(1 / sum p^2) over the positive scores of ranking x (section 33's rule, with its 1e-9 tolerance);
  KNEE_x = #{positive scores >= their mean}: the knee (Kneedle) of the query's own cumulative evidence curve, both axes
           normalised -- for a non-increasing sequence the chord-distance maximum C(k) - k/K is the last k with a gain >= mean;
  EXPH_TOP3 (section 33); NSEEDP / NOACT / NLOCP = the distinct partitions of H_q = FLAT[:200] / of O[:200] / of the first 200
  nodes of the IR_L1 LOC order (the evidence spread at the frozen activation depth 200);
  for a coverage ranking: NEFFG / KNEEG over its greedy marginal gains, COVER = the number of positive gains (saturation);
  U_NEFF / U_KNEE = the union sizes.
  The rule G_{t+1} < mean(G_1..G_t) is not run: on a non-increasing gain sequence it stops at the first strict decrease.
Reference, B_N-coupled (not a rule): PSTAR_M = the distinct partitions of O[:M]; S at PSTAR_M == unrouted (asserted).
Stored per query: the ALL-served interval [LO, HI(B_N)] of every ranking (so every count of every ranking is evaluated post
hoc from the npz), every count and its raw value, the routing position of every gold's partition, the contacted mass at every
(ranking, count).
Identities asserted: FLAT == loc v1 (runner); IR_L1 LOC / FLAT+LOC == node1h v1; LO / HI of S, SUM, MAX, TOP3 and the counts
NEFF_SUM / NEFF_MAX / NEFF_TOP3 / EXPH_TOP3 == adaptbp v1 (section 33); every partition contacted == unrouted (per row, per
ranking); S at PSTAR_M == unrouted; the per-seed masses sum to SUM; the greedy gains never increase; on the first rows, the
literal greedy node cover == SUM, the incremental greedy == a from-scratch greedy, and every ranking is a permutation.

Usage: python scratchpad/_l1d_route.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/route_<dataset>__<tag>.{json,npz} (write-once)
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
import _l1d_adaptbp as AB

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = NH.FAMS
NODE = "IR_L1"
TOPK = 3
TOL = AB.TOL
TOLG = 1e-9                    # relative tie band of the greedy argmax, and the non-increase check of its gains
EPS_G = 1e-12                  # a greedy gain <= EPS_G * the first gain counts as zero (saturation)
CHECK_ROWS = 3
SEM = ("S", "F", "D", "SP", "FSUM", "FSUMFV", "FT3")
LOCR = ("SUM", "MAX", "TOP3", "SDIV", "OT3")
COVS = ("SC_L", "SC_LS", "FL_L", "FL_LS")
FUS = ("PF_S_T3", "PF_F_T3", "IL_S_T3", "U_NEFF", "U_KNEE")
RANKS = SEM + LOCR + COVS + ("PD",) + FUS
SCORED = ("FSUM", "FSUMFV", "SUM", "MAX", "TOP3", "SDIV", "PD")
COUNTS = (["NEFF_" + x for x in SCORED] + ["KNEE_" + x for x in SCORED] + ["EXPH_TOP3", "NSEEDP", "NOACT", "NLOCP"]
          + ["%s_%s" % (c, k) for c in COVS for k in ("NEFFG", "KNEEG", "COVER")] + ["U_NEFF", "U_KNEE"])
AB_RANKS = ("S", "SUM", "MAX", "TOP3")
AB_COUNTS = {"NEFF_SUM": "OWN|SUM|NEFF", "NEFF_MAX": "OWN|MAX|NEFF", "NEFF_TOP3": "OWN|TOP3|NEFF", "EXPH_TOP3": "OWN|TOP3|EXPH"}
NATIVE = ([("SUM", "NEFF_SUM"), ("MAX", "NEFF_MAX"), ("TOP3", "NEFF_TOP3"), ("TOP3", "EXPH_TOP3")]
          + [(x, "KNEE_" + x) for x in ("SUM", "MAX", "TOP3")]
          + [(x, "%s_%s" % (p, x)) for x in ("FSUM", "FSUMFV", "SDIV", "PD") for p in ("NEFF", "KNEE")]
          + [(x, "%s_%s" % (x, k)) for x in COVS for k in ("NEFFG", "KNEEG", "COVER")]
          + [("U_NEFF", "U_NEFF"), ("U_KNEE", "U_KNEE"), ("S", "NSEEDP"), ("F", "NSEEDP"), ("S", "NEFF_FSUMFV"), ("S", "KNEE_FSUMFV"),
             ("S", "NOACT"), ("S", "NLOCP"), ("TOP3", "NLOCP")])
BP_SHOW = AB.BP_SHOW
MSA = np.asarray(D.M_CURVE, np.int64)
NH_PATH = os.path.join(D.HERE, "_l1d_node1h.py")
E_PATH = os.path.join(D.HERE, "_l1d_edgediag.py")
AB_PATH = os.path.join(D.HERE, "_l1d_adaptbp.py")
AP_CHUNK = 1 << 22
assert len(set(RANKS)) == len(RANKS) and len(set(COUNTS)) == len(COUNTS)
assert all(r in RANKS and c in COUNTS for r, c in NATIVE) and set(AB_RANKS) <= set(RANKS) and FAMS == AB.FAMS


def inv(order):
    r = np.empty(len(order), np.int64)
    r[order] = np.arange(len(order))
    return r


def neff(v):
    v = v[v > 0]
    if not len(v):
        return None, 0
    p = v / v.sum()
    return 1.0 / float(np.dot(p, p)), len(v)


def knee(v):
    """#{v >= mean(v)} over the positive values (>= 1 whenever one value is positive)."""
    v = v[v > 0]
    if not len(v):
        return 0, 0
    return int((v >= (v.sum() / len(v)) * (1.0 - TOL)).sum()), len(v)


def topk_by_order(order, val, hard, npart, k):
    """per partition, the sum of val over its first k nodes in `order` (val non-increasing along `order`)."""
    hp = hard[order]
    o2 = np.argsort(hp, kind="stable")
    hps = hp[o2]
    wr = np.arange(len(hps)) - np.searchsorted(hps, hps, side="left")
    m = wr < k
    return np.bincount(hps[m], weights=val[order][o2][m], minlength=npart)


def pick(gains, alive):
    gm = np.where(alive, gains, -np.inf)
    mx = gm.max()
    return int(np.flatnonzero(gm >= mx - TOLG * abs(mx))[0]), mx


def greedy(Phi, w):
    """greedy maximum of sum_s w(s) max_{P in S} Phi[s, P] over the columns of Phi (column order = the tie priority), with
    incremental gain updates.  Returns (selected columns in order, their marginal gains)."""
    nh, T = Phi.shape
    best = np.zeros(nh)
    gains = w @ Phi
    alive = np.ones(T, bool)
    sel, G = [], []
    g0 = None
    while alive.any():
        k, gk = pick(gains, alive)
        if g0 is None:
            g0 = gk
        if not (gk > 0 and gk > EPS_G * g0):
            break
        sel.append(k)
        G.append(gk)
        alive[k] = False
        col = Phi[:, k]
        dl = np.flatnonzero(col > best)
        if len(dl):
            M = Phi[dl]
            old, new = best[dl][:, None], col[dl][:, None]
            gains -= (w[dl][:, None] * (np.maximum(M - old, 0.0) - np.maximum(M - new, 0.0))).sum(0)
            best[dl] = col[dl]
    return np.asarray(sel, np.int64), np.asarray(G, np.float64)


def greedy_ref(Phi, w):
    """the same greedy with the gains recomputed from scratch at every step (a check on the first rows)."""
    nh, T = Phi.shape
    best = np.zeros(nh)
    alive = np.ones(T, bool)
    sel, G = [], []
    g0 = None
    while alive.any():
        gains = (w[:, None] * np.maximum(Phi - best[:, None], 0.0)).sum(0)
        k, gk = pick(gains, alive)
        if g0 is None:
            g0 = gk
        if not (gk > 0 and gk > EPS_G * g0):
            break
        sel.append(k)
        G.append(gk)
        alive[k] = False
        best = np.maximum(best, Phi[:, k])
    return np.asarray(sel, np.int64), np.asarray(G, np.float64)


def node_cover_literal(hnz, Lv, fa, npart):
    """the proposal's greedy marginal node coverage, literally: P_t = argmax sum of L over the uncovered nodes of P."""
    covered = np.zeros(len(Lv), bool)
    alive = np.ones(npart, bool)
    order = []
    while True:
        gains = np.bincount(hnz[~covered], weights=Lv[~covered], minlength=npart)
        gains[~alive] = -1.0
        mx = gains.max()
        if mx <= 0:
            break
        cand = np.flatnonzero(gains == mx)
        k = int(cand[np.argmin(fa[cand])])
        order.append(k)
        alive[k] = False
        covered |= hnz == k
    rest = np.flatnonzero(alive)
    return np.concatenate([np.asarray(order, np.int64), rest[np.argsort(fa[rest], kind="stable")]])


def partition_adjacency(F, hard, npart, N):
    """A_P[P', P] = the entries s -> u of the four families of E with P(s) = P' != P(u) = P; plus the internal entries."""
    AP = np.zeros(npart * npart, np.float64)
    tot = internal = 0
    for f in FAMS:
        xadj, adj = F[f]["xadj"], F[f]["adj"]
        for c0 in range(0, len(adj), AP_CHUNK):
            c1 = min(len(adj), c0 + AP_CHUNK)
            rows = np.searchsorted(xadj, np.arange(c0, c1, dtype=np.int64), side="right") - 1
            ps, pt = hard[rows], hard[adj[c0:c1].astype(np.int64)]
            x = ps != pt
            AP += np.bincount(ps[x] * npart + pt[x], minlength=npart * npart)
            tot += c1 - c0
            internal += int((~x).sum())
    AP = AP.reshape(npart, npart)
    return AP, {"entries": int(tot), "internal_share": D.q4(internal / float(max(tot, 1))),
                "cross_entries": int(AP.sum()), "partition_pairs_with_cross_entries": int((AP > 0).sum()),
                "cross_entry_degree_per_partition": D.stats(AP.sum(1))}


class RouteSpec(object):
    def __init__(self, cd, pop, parts):
        self.ds, self.parts, self.cells = cd.name, parts, list(parts)
        N = self.N = int(cd.n_nodes)
        nq, ng = pop.nq, pop.ng_tot
        # ---- the section-32 and section-33 records this harness must reproduce (same code, population, partitions)
        fr = os.path.join(D.OUT, "node1h_%s__v1.json" % self.ds)
        R1 = json.load(open(fr, encoding="utf-8"))
        self.node1h_npz = os.path.join(D.OUT, R1["npz"]["path"])
        assert D.sha_file(self.node1h_npz) == R1["npz"]["sha256"], "node1h npz changed"
        fa_ = os.path.join(D.OUT, "adaptbp_%s__v1.json" % self.ds)
        RA = json.load(open(fa_, encoding="utf-8"))
        self.adapt_npz = os.path.join(D.OUT, RA["npz"]["path"])
        assert D.sha_file(self.adapt_npz) == RA["npz"]["sha256"], "adaptbp npz changed"
        self.sha_nh, self.sha_e, self.sha_ab = D.sha_file(NH_PATH), D.sha_file(E_PATH), D.sha_file(AB_PATH)
        assert self.sha_nh == R1["code"]["harness"]["sha256"], "_l1d_node1h.py differs from the code of its v1 record"
        assert self.sha_e == R1["structures"]["imports"]["scratchpad/_l1d_edgediag.py"], "_l1d_edgediag.py differs"
        assert self.sha_ab == RA["code"]["harness"]["sha256"], "_l1d_adaptbp.py differs from the code of its v1 record"
        assert RA["structures"]["imports"]["scratchpad/_l1d_node1h.py"] == self.sha_nh
        for rec in (R1, RA):
            assert rec["code"]["arms"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_arms.py"))
            assert rec["code"]["lib"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))
        rc = RA["structures"]["cells"]
        assert sorted(rc) == sorted(self.cells)
        for c, P in parts.items():
            assert rc[c]["partition"] == P.tag
        t0 = time.time()
        self.F, famrec = NH.build_families(cd, N)
        t_fam = time.time() - t0
        self.APT, self.DEGP, aprec = {}, {}, {}
        for c, P in parts.items():
            AP, aprec[c] = partition_adjacency(self.F, P.hard, P.npart, N)
            self.DEGP[c] = AP.sum(1)
            self.APT[c] = np.ascontiguousarray(AP.T)
            del AP
        self.names = [NODE]
        self.hard_ref = {NODE: None}
        self.v1_hard = {}
        self.lat = {NODE: []}
        self.FUSE_LAT = []
        self.TL = {c: {k: [] for k in ("semantic", "localised", "coverage", "structure", "fusion_and_counts", "evaluation")}
                   for c in self.cells}
        self.FRK = np.zeros(ng, np.int64)
        nr, ncn, nm = len(RANKS), len(COUNTS), len(D.M_CURVE)
        self.LO = {c: np.zeros((nq, nr), np.int32) for c in self.cells}
        self.HI = {c: np.zeros((nq, nr, nm), np.int32) for c in self.cells}
        self.PRG = {c: np.zeros((ng, nr), np.int32) for c in self.cells}
        self.CNT = {c: np.zeros((nq, ncn), np.int32) for c in self.cells}
        self.CRAW = {c: np.full((nq, ncn), np.nan) for c in self.cells}
        self.CMASS = {c: np.zeros((nq, nr, ncn), np.int32) for c in self.cells}
        self.PSTAR = {c: np.zeros((nq, nm), np.int32) for c in self.cells}
        self.MSUM = {c: np.zeros((nr, P.npart)) for c, P in parts.items()}
        self.RET = {c: np.zeros((nr, nm, P.npart)) for c, P in parts.items()}
        self.NOEV = {c: 0 for c in self.cells}
        self.GSTEPS = {c: {k: [] for k in COVS} for c in self.cells}
        self.checks = {c: 0 for c in self.cells}
        self.record = {
            "node_score": "IR_L1 (sections 32-33, recomputed and asserted identical)",
            "served_order": "IR_L1 FLAT+LOC (RRF K0 = 60); the first B_N nodes of it inside the contacted partitions",
            "rankings": list(RANKS), "counts": list(COUNTS), "native_arms": ["%s|%s" % x for x in NATIVE],
            "topk": TOPK, "greedy": {"tie_band_relative": TOLG, "zero_gain_relative": EPS_G, "tie_priority": "first appearance in O"},
            "families": famrec, "seconds_families": round(t_fam, 1),
            "cells": {c: {"partition": P.tag, "npart": P.npart, "partition_size": D.stats(P.sizes), "partition_graph": aprec[c]}
                      for c, P in parts.items()},
            "reproduces": {"node1h": {"path": D.rel(fr), "sha256": D.sha_file(fr), "npz_sha256": R1["npz"]["sha256"]},
                           "adaptbp": {"path": D.rel(fa_), "sha256": D.sha_file(fa_), "npz_sha256": RA["npz"]["sha256"]}},
            "imports": {"scratchpad/_l1d_node1h.py": self.sha_nh, "scratchpad/_l1d_edgediag.py": self.sha_e,
                        "scratchpad/_l1d_adaptbp.py": self.sha_ab}}

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        N = self.N
        top = of[:ACT]
        nh = len(top)
        # ---- L1a: the IR_L1 node score (the arithmetic of _l1d_node1h / _l1d_adaptbp for this rung)
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
        wseed = 1.0 / np.arange(1.0, nh + 1.0)
        pdn = np.empty(N, np.int64)
        pdn[od] = np.arange(N)
        psn = np.full(N, npos, np.int64)
        psn[os_] = np.arange(npos)
        chk = j < CHECK_ROWS
        for c in self.cells:
            P = self.parts[c]
            npart, hard, TL = P.npart, P.hard, self.TL[c]
            SC, RO, CN, CR = {}, {}, {}, {}
            # ---- semantic partition scores
            t1 = time.perf_counter()
            fa = np.minimum.reduceat(frk[P.order_nodes], P.ptr[:-1])
            ff = np.minimum.reduceat(frank[P.order_nodes], P.ptr[:-1])
            fdn = np.minimum.reduceat(pdn[P.order_nodes], P.ptr[:-1])
            fsp = np.minimum.reduceat(psn[P.order_nodes], P.ptr[:-1])
            hs = hard[top]
            SC["FSUM"] = np.bincount(hs, weights=wseed, minlength=npart)
            SC["FSUMFV"] = np.bincount(hs, weights=fv[top], minlength=npart)
            SC["FT3"] = topk_by_order(of, fv, hard, npart, TOPK)
            RO["S"] = np.argsort(fa, kind="stable")
            RO["F"] = np.lexsort((fa, ff))
            RO["D"] = np.lexsort((fa, fdn))
            RO["SP"] = np.lexsort((fa, fsp))
            for r_ in ("FSUM", "FSUMFV", "FT3"):
                RO[r_] = np.lexsort((fa, -SC[r_]))
            TL["semantic"].append(time.perf_counter() - t1)
            # ---- localised partition scores (the section-33 arithmetic for SUM / MAX / TOP3)
            t1 = time.perf_counter()
            ph = hard[nz]
            SC["SUM"] = np.bincount(ph, weights=Lv, minlength=npart)
            mx = np.zeros(npart)
            np.maximum.at(mx, ph, Lv)
            SC["MAX"] = mx
            o = np.lexsort((-Lv, ph))
            phs, lvs = ph[o], Lv[o]
            wr = np.arange(len(phs)) - np.searchsorted(phs, phs, side="left")
            m = wr < TOPK
            SC["TOP3"] = np.bincount(phs[m], weights=lvs[m], minlength=npart)
            Cm = np.bincount(hit * npart + hard[u], weights=x, minlength=nh * npart).reshape(nh, npart)
            assert np.allclose(Cm.sum(0), SC["SUM"], rtol=1e-9, atol=1e-15), "per-seed masses do not sum to SUM"
            SC["SDIV"] = wseed @ (Cm > 0)
            SC["OT3"] = topk_by_order(FO, f, hard, npart, TOPK)
            for r_ in ("SUM", "MAX", "TOP3", "SDIV", "OT3"):
                RO[r_] = np.lexsort((fa, -SC[r_]))
            TL["localised"].append(time.perf_counter() - t1)
            if chk:
                assert (node_cover_literal(ph, Lv, fa, npart) == RO["SUM"]).all(), "literal greedy node cover != SUM (%s)" % c
            # ---- seed coverage / facility location
            t1 = time.perf_counter()
            so = RO["S"]
            tot = Cm.sum(1)
            PhiL = np.zeros((nh, npart))
            okm = tot > 0
            PhiL[okm] = Cm[okm] / tot[okm][:, None]
            PhiLS = PhiL.copy()
            PhiLS[np.arange(nh), hs] = 1.0
            BL = (Cm > 0).astype(np.float64)
            BLS = BL.copy()
            BLS[np.arange(nh), hs] = 1.0
            GG = {}
            for name, Phi in (("SC_L", BL), ("SC_LS", BLS), ("FL_L", PhiL), ("FL_LS", PhiLS)):
                Ps = Phi[:, so]
                sel, G = greedy(Ps, wseed)
                if len(G) > 1:
                    assert (np.diff(G) <= TOLG * G[0]).all(), "greedy gains increased (%s %s)" % (c, name)
                if chk:
                    s2, G2 = greedy_ref(Ps, wseed)
                    assert len(s2) == len(sel) and (s2 == sel).all() and np.allclose(G2, G, rtol=1e-9, atol=0), \
                        "incremental greedy != from-scratch greedy (%s %s)" % (c, name)
                chosen = so[sel]
                rest = so[np.isin(so, chosen, invert=True)]
                RO[name] = np.concatenate([chosen, rest])
                GG[name] = G
                self.GSTEPS[c][name].append(len(G))
            TL["coverage"].append(time.perf_counter() - t1)
            # ---- partition graph: one static partition-level step
            t1 = time.perf_counter()
            dg = self.DEGP[c]
            y = np.zeros(npart)
            pos_ = dg > 0
            y[pos_] = SC["SUM"][pos_] / dg[pos_]
            SC["PD"] = SC["SUM"] + self.APT[c] @ y
            RO["PD"] = np.lexsort((fa, -SC["PD"]))
            TL["structure"].append(time.perf_counter() - t1)
            # ---- counts, then fusion / union rankings
            t1 = time.perf_counter()
            for x_ in SCORED:
                ne, npz_ = neff(SC[x_])
                CR["NEFF_" + x_] = ne if npz_ else np.nan
                CN["NEFF_" + x_] = AB.bp_of(ne, npz_, npart) if npz_ else npart
                kn, npz_ = knee(SC[x_])
                CR["KNEE_" + x_] = kn if npz_ else np.nan
                CN["KNEE_" + x_] = kn if npz_ else npart
            v = SC["TOP3"][SC["TOP3"] > 0]
            if len(v):
                p = v / v.sum()
                eh = float(np.exp(-np.dot(p, np.log(p))))
                CR["EXPH_TOP3"], CN["EXPH_TOP3"] = eh, AB.bp_of(eh, len(v), npart)
            else:
                CN["EXPH_TOP3"] = npart
                self.NOEV[c] += 1
            CN["NSEEDP"] = CR["NSEEDP"] = int(len(np.unique(hs)))
            CN["NOACT"] = CR["NOACT"] = int(len(np.unique(hard[FO[:ACT]])))
            if len(Lord):
                CN["NLOCP"] = CR["NLOCP"] = int(len(np.unique(hard[Lord[:ACT]])))
            else:
                CN["NLOCP"] = npart
            for name in COVS:
                G = GG[name]
                if len(G):
                    ne, _ = neff(G)
                    CR[name + "_NEFFG"], CN[name + "_NEFFG"] = ne, AB.bp_of(ne, len(G), npart)
                    kn, _ = knee(G)
                    CR[name + "_KNEEG"], CN[name + "_KNEEG"] = kn, kn
                    CR[name + "_COVER"], CN[name + "_COVER"] = len(G), len(G)
                else:
                    for k_ in ("NEFFG", "KNEEG", "COVER"):
                        CN["%s_%s" % (name, k_)] = npart
            rS, rT, rF = inv(RO["S"]), inv(RO["TOP3"]), inv(RO["F"])
            RO["PF_S_T3"] = np.lexsort((fa, -(1.0 / (K0 + rS) + 1.0 / (K0 + rT))))
            RO["PF_F_T3"] = np.lexsort((fa, -(1.0 / (K0 + rF) + 1.0 / (K0 + rT))))
            RO["IL_S_T3"] = np.argsort(np.minimum(2 * rS, 2 * rT + 1), kind="stable")
            for uname, (cf, cl) in (("U_NEFF", (CN["NEFF_FSUMFV"], CN["NEFF_TOP3"])), ("U_KNEE", (CN["KNEE_FSUMFV"], CN["KNEE_TOP3"]))):
                RO[uname] = np.lexsort((fa, np.minimum(rS / float(cf), rT / float(cl))))
                inU = (rS < cf) | (rT < cl)
                CN[uname] = CR[uname] = int(inU.sum())
                assert inU[RO[uname][:CN[uname]]].all(), "union prefix != union (%s %s)" % (c, uname)
            TL["fusion_and_counts"].append(time.perf_counter() - t1)
            cnt = np.array([CN[k_] for k_ in COUNTS], np.int64)
            assert (cnt >= 1).all() and (cnt <= npart).all(), (c, dict(zip(COUNTS, cnt.tolist())))
            self.CNT[c][j] = cnt
            self.CRAW[c][j] = [CR.get(k_, np.nan) for k_ in COUNTS]
            for mi, M in enumerate(D.M_CURVE):
                self.PSTAR[c][j, mi] = len(np.unique(hard[FO[:M]]))
            # ---- evaluation (gold-dependent; not part of any rule): every fan-out of every ranking
            t1 = time.perf_counter()
            hO, hg = hard[FO[:lim]], hard[g]
            for ri, rk in enumerate(RANKS):
                ro = RO[rk]
                if chk:
                    assert len(ro) == npart and (np.sort(ro) == np.arange(npart)).all(), "not a permutation (%s %s)" % (c, rk)
                pr = inv(ro)
                prg = pr[hg]
                C = AB.cumcounts(pr[hO], pg, npart)
                assert (C[:, npart - 1] - 1 == pg).all(), "every partition contacted != unrouted (%s %s)" % (c, rk)
                ms = np.cumsum(P.sizes[ro])
                lo = int(prg.max()) + 1
                qmax = (C[:, lo - 1:] - 1).max(axis=0)
                assert (np.diff(qmax) >= 0).all()
                self.LO[c][j, ri] = lo
                self.HI[c][j, ri] = lo - 1 + np.searchsorted(qmax, MSA, side="left")
                self.PRG[c][sl, ri] = prg
                self.MSUM[c][ri] += ms
                self.RET[c][ri] += np.minimum(MSA[:, None], ms[None, :])
                self.CMASS[c][j, ri] = ms[cnt - 1]
            TL["evaluation"].append(time.perf_counter() - t1)
            if chk:
                self.checks[c] += 1
        return {NODE: Lord}

    def finish(self, ctx):
        POS, gptr, ngold, pop, ST = ctx["POS"], ctx["gptr"], ctx["ngold"], ctx["pop"], ctx["ST"]
        nq, ng = pop.nq, pop.ng_tot
        assert D.sha_file(NH_PATH) == self.sha_nh and D.sha_file(E_PATH) == self.sha_e and D.sha_file(AB_PATH) == self.sha_ab, \
            "imported code changed during the run"
        allv = lambda pos, M: D.per_query(pos, M, gptr, ngold)[0]
        diag, arrays = {}, {}
        # (0) identities against the section-32 and section-33 records
        z = np.load(self.node1h_npz)
        assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == gptr).all()
        assert (z["pos_LOC__" + NODE][:ng] == POS[NODE]["LOC"]).all(), "IR_L1 LOC positions differ from node1h v1"
        assert (z["pos_FLATLOC__" + NODE][:ng] == POS[NODE]["FLAT+LOC"]).all(), "IR_L1 FLAT+LOC positions differ from node1h v1"
        assert (self.FRK == POS[NODE]["FLAT+LOC"]).all()
        za = np.load(self.adapt_npz)
        assert (za["rows"][:nq] == pop.rows).all() and (za["gptr"][:nq + 1] == gptr).all()
        aranks, aarms = [str(s) for s in za["ranks"]], [str(s) for s in za["arms"]]
        unr = {M: allv(POS[NODE]["FLAT+LOC"], M) for M in D.M_CURVE}
        for c in self.cells:
            for rk in AB_RANKS:
                ai, ri = aranks.index(rk), RANKS.index(rk)
                assert (za["LO__" + c][:nq, ai] == self.LO[c][:, ri]).all(), "LO differs from adaptbp v1 (%s %s)" % (c, rk)
                assert (za["HI__" + c][:nq, ai] == self.HI[c][:, ri]).all(), "HI differs from adaptbp v1 (%s %s)" % (c, rk)
            for cn, an in AB_COUNTS.items():
                assert (za["BPQ__" + c][:nq, aarms.index(an)] == self.CNT[c][:, COUNTS.index(cn)]).all(), "count differs (%s %s)" % (c, cn)
            si = RANKS.index("S")
            for mi, M in enumerate(D.M_CURVE):
                assert (AB.served_at(self.LO[c][:, si], self.HI[c][:, si, mi], self.PSTAR[c][:, mi]) == unr[M]).all(), \
                    "S at PSTAR != unrouted (%s %d)" % (c, M)
        diag["identity"] = ("IR_L1 LOC and FLAT+LOC positions == node1h v1 on all %d gold nodes; LO / HI of %s and the counts %s == "
                            "adaptbp v1 in every cell; every partition contacted == unrouted (per row, per ranking); S at PSTAR_M == "
                            "unrouted at every M; per-seed masses sum to SUM and greedy gains never increase (every row); on the first "
                            "%d rows of every cell: the literal greedy node cover == SUM, the incremental greedy == a from-scratch "
                            "greedy for %s, every ranking a permutation" % (ng, list(AB_RANKS), sorted(AB_COUNTS), CHECK_ROWS, list(COVS)))
        diag["checked_rows"] = self.checks
        hop_m = ST.get("per_hop", {})
        cells_out = {}
        for c in self.cells:
            P = self.parts[c]
            npart = P.npart
            LO, HI, CNT = self.LO[c].astype(np.int64), self.HI[c].astype(np.int64), self.CNT[c].astype(np.int64)
            e_c = {"unrouted_ALL": {str(M): D.q4(unr[M].mean()) for M in D.M_CURVE}, "rankings": {}, "counts": {}, "native_arms": {}}
            SF = {}
            for ri, rk in enumerate(RANKS):
                S_ = AB.surface(LO[:, ri], HI[:, ri], npart)
                SF[rk] = S_
                e = {"contacted_mass_mean": {str(b): round(float(self.MSUM[c][ri, b - 1] / nq), 1) for b in BP_SHOW if b <= npart},
                     "gold_partition_contacted_share": {str(b): D.q4((self.PRG[c][:, ri] < b).mean()) for b in BP_SHOW if b <= npart},
                     "LO": AB.dist(LO[:, ri])}
                for mi, M in enumerate(D.M_CURVE):
                    row_ = S_[mi]
                    assert abs(row_[-1] - unr[M].mean()) < 1e-12, "surface at every partition != unrouted (%s %s %d)" % (c, rk, M)
                    e[str(M)] = {"ALL_at_B_P": {str(b): D.q4(row_[b - 1]) for b in BP_SHOW if b <= npart},
                                 "best_fixed": {"B_P": int(np.argmax(row_)) + 1, "ALL": D.q4(row_.max())},
                                 "smallest_B_P_within_0.01_of_unrouted": int(np.flatnonzero(row_ >= row_[-1] - 0.01)[0]) + 1,
                                 "smallest_B_P_at_or_above_unrouted": int(np.flatnonzero(row_ >= row_[-1] - 1e-12)[0]) + 1}
                e_c["rankings"][rk] = e
            for ci, cn in enumerate(COUNTS):
                e_c["counts"][cn] = {"B_P(q)": AB.dist(CNT[:, ci]), "raw": AB.dist(self.CRAW[c][:, ci][np.isfinite(self.CRAW[c][:, ci])])
                                     if np.isfinite(self.CRAW[c][:, ci]).any() else None}
            for rk, cn in NATIVE:
                ri, ci = RANKS.index(rk), COUNTS.index(cn)
                b = CNT[:, ci]
                e = {"B_P(q)": AB.dist(b), "contacted_mass": AB.dist(self.CMASS[c][:, ri, ci])}
                for mi, M in enumerate(D.M_CURVE):
                    a_ = AB.served_at(LO[:, ri], HI[:, ri, mi], b)
                    row_ = SF[rk][mi]
                    bm = float(b.mean())
                    bl, bh = int(np.floor(bm)), int(np.ceil(bm))
                    e[str(M)] = {"ALL": D.q4(a_.mean()), "delta_vs_unrouted": D.q4(a_.mean() - unr[M].mean()),
                                 "paired_vs_unrouted": D.paired(unr[M], a_),
                                 "same_ranking_fixed_at_mean_fan_out_interpolated": D.q4(row_[bl - 1] + (bm - bl) * (row_[bh - 1] - row_[bl - 1])),
                                 "same_ranking_best_fixed": {"B_P": int(np.argmax(row_)) + 1, "ALL": D.q4(row_.max())},
                                 "lost_reach (B_P(q) < LO)": D.q4((b < LO[:, ri]).mean()),
                                 "lost_crowding (LO <= HI < B_P(q))": D.q4(((LO[:, ri] <= HI[:, ri, mi]) & (HI[:, ri, mi] < b)).mean()),
                                 "unservable_at_any_B_P (LO > HI)": D.q4((LO[:, ri] > HI[:, ri, mi]).mean())}
                    if hop_m:
                        e[str(M)]["per_hop"] = {hk: {"n": int(qm.sum()), "ALL": D.q4(a_[qm].mean()), "unrouted_ALL": D.q4(unr[M][qm].mean())}
                                                for hk, qm in hop_m.items()}
                e_c["native_arms"]["%s|%s" % (rk, cn)] = e
            e_c["PSTAR (distinct partitions of O[:M]; S at PSTAR == unrouted)"] = {str(M): AB.dist(self.PSTAR[c][:, mi]) for mi, M in enumerate(D.M_CURVE)}
            e_c["greedy_steps"] = {k: AB.dist(v) for k, v in self.GSTEPS[c].items()}
            e_c["queries_without_evidence"] = self.NOEV[c]
            cells_out[c] = e_c
            for nm_, A_ in (("LO", self.LO), ("HI", self.HI), ("PRG", self.PRG), ("CNT", self.CNT), ("CRAW", self.CRAW),
                            ("CMASS", self.CMASS), ("PSTAR", self.PSTAR), ("MSUM", self.MSUM), ("RET", self.RET)):
                arrays["%s__%s" % (nm_, c)] = A_[c]
        diag["cells"] = cells_out
        diag["latency_ms"] = {"node_score_and_LOC_order": D.ms_stats(self.lat[NODE]), "fused_order": D.ms_stats(self.FUSE_LAT),
                              "per_cell": {c: {k: D.ms_stats(v) for k, v in self.TL[c].items()} for c in self.cells}}
        arrays["ranks"] = np.array(RANKS)
        arrays["counts"] = np.array(COUNTS)
        arrays["native"] = np.array(["%s|%s" % x for x in NATIVE])
        arrays["m_curve"] = MSA
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    R.run(sys.argv[2], sys.argv[3], "route", "L1_DEVELOPMENT_ROUTE", __file__, RouteSpec, __doc__)


if __name__ == "__main__":
    main()
