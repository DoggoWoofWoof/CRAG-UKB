"""L3 DEVELOPMENT -- HOP: a bounded second and third STRUCT hop on top of the carried-forward L1, at equal exposure.
(User 2026-09-29: "we need to finish the full KB path: L1 localization -> L2 reranking -> L3 multi-hop completion"; "First,
improve MetaQA on development data"; REPORT section 38.10: "MetaQA development moves to L3, not L2.  On top of the carried-forward
L1, on the same development rows: a bounded second STRUCT hop, in its partition-local form and in a form that fetches its
targets; then a third hop for hop 3; all measured at equal exposure, with fan-out and fetches counted.")
L1 is not changed: the traversal below runs AFTER the carried-forward L1 and reads only its outputs (the served order, the
contacted shards, the evidence set V_q and the STRUCT part of the IR_L1 score).  Development rows only: the L1_DEV population
(c7f70806...), legacy-exposed; no reserved held-out row, no split-B row and no TEST row is read.  No encoder training, no LLM, no
learned weight, no tuned constant (the RRF constant is the frozen K0 60).  STATUS: DEVELOPMENT -- descriptive numbers, every
p-value descriptive, no rule selected here.

Fixed input, the carried-forward L1 (results/L1_DEV/L1_ROUTING_CARRY_FORWARD__v1.json, pinned through _l1d_kbres.KBResSpec):
  IR_L1   L(u) = sum_{s in H_q} (1/(FLAT rank(s)+1)) g_f(s) A_f(s,u) over E = STRUCT_out u STRUCT_in u KNN u NER; LOC = {L > 0} by
          (-L, FLAT rank); the served order O: f(u) = 1/(K0 + FLAT rank(u)) + [u in LOC] / (K0 + LOC rank(u))
  routing ES shard order on the native K 432 maps (PHG, MTK); B_P(q) = min(K, ceil(B100(q) (K/100)^0.75)) (kscale v1 CNTA);
          contacted = the nodes of the first B_P shards; L1 serves the first B_N nodes of O inside the contacted shards
  V_q     the evidence set: the 200 hits and every node with L > 0
The L3 operator (the next applications of IR_L1's arithmetic restricted to the two STRUCT families -- the same operator that the
section 38 probe measured):
  LS      IR_L1's score restricted to STRUCT_out u STRUCT_in (the first hop's STRUCT mass; support inside V_q)
  prop    y(u) = sum_{f in STRUCT_out, STRUCT_in} sum_{m in src} A_f(m, u) g_f(m) w(m)  (undirected; g = 1/log2(1 + row length))
  hop 2   y2 = prop(src2, LS), src2 = the LS support inside the contacted shards (routed) or all of it (UNROUTED)
  hop 3   y3 = prop(src3, y2), src3 = the y2 support: inside the contacted shards, or all of it (reading the adjacency rows of
          nodes in uncontacted shards: "rows fetched", counted)
  C2      the second-hop candidates: y2 > 0, not in V_q; ordered by (-y2, FLAT rank)
  C3      the third-hop candidates: y3 > 0, not in V_q, not in C2; ordered by (-y3, FLAT rank)
  merge   RRF with the frozen K0: f2(u) = f(u) + [u in C2] / (K0 + C2 rank(u)); f3(u) = f2(u) + [u in C3] / (K0 + C3 rank(u));
          ties -> FLAT rank.  No weight, no threshold, no query-type or hop information (the hop label is not system-visible and
          is used only to stratify the report)
Arms per routed cell (PHG_k432, MTK_k432); every arm of a cell serves exactly the same number of nodes per query as L1 does,
n(q, B_N) = min(B_N, contacted mass(q)) (equal exposure):
  L1            the carried-forward served list (== REPORT section 37.3 / section 38)
  H2_LOCAL      sources and targets inside the contacted shards (C2 restricted to contacted; partition-local completion)
  H2_PUSH       sources inside the contacted shards, targets anywhere: a served C2 node outside the contacted shards is FETCHED
                (counted, with the number of distinct extra shards it comes from); V_q nodes outside the contacted shards stay
                unservable (that is L1's fan-out, not L3's)
  H3_LOCAL      H2_LOCAL + a third hop with sources and targets inside the contacted shards
  H3_PUSH       H2_PUSH + a third hop from the y2 support inside the contacted shards, targets anywhere (fetched when served)
  H3_PUSH_ROWS  H2_PUSH + a third hop from the whole y2 support: the adjacency rows of y2-support nodes in uncontacted shards are
                read (counted: rows fetched and their distinct shards); targets anywhere
UNROUTED (every shard contacted; B_N nodes served): L1 (the IR_L1 order), H2, H3.
Exposure B_N in M_CURVE (100 .. 5000).  Reported per cell, arm and B_N: ALL / ANY / FRAC and the gold nodes served, per hop, per
answer-count bucket and hop x bucket, per qtype at B_N 1000 and 5000; McNemar gained / lost vs the cell's L1 (descriptive); the
fetched nodes and extra shards per query; the traversal cost (source rows expanded, adjacency entries scanned, candidate list
sizes); latency per stage.

Identities asserted: FLAT == loc v1; the dense / SPLADE top-100 agreement gate; per row and native cell the recomputed ES LO ==
kscale v1 LOES; on every gold node: the UNROUTED L1 position == kbres v1 G_FRK, the routed L1 contacted flag and restricted rank ==
kbres v1 cont__ / rr__, the UNROUTED C2 rank == kbres v1 G_R2 (the section 38 probe); per row the contacted mass == kbres v1
Q_CMASS__ and |C2 UNROUTED| == kbres v1 Q_C2; on the full population the L1 ALL counts at every B_N == the kbres v1 tables (1,015 /
1,112 / 1,082 at B_N 1000); LOCAL arms never serve outside the contacted shards; every arm serves exactly n(q, B_N) nodes.

Usage: python scratchpad/_l3d_hop.py RUN metaqa <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L3_DEV/l3hop_<dataset>__<tag>.{json,npz} (write-once)
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_arms as R
import _l1d_edgediag as E
import _l1d_route as RT
import _l1d_kbres as KB

log = D.log
K0, ACT = D.K0, D.ACT
FAMS, SFAMS = KB.FAMS, KB.SFAMS
M_CURVE = D.M_CURVE
UNR = KB.UNR
OUT3 = os.path.join(D.REPO, "results", "L3_DEV")
KBRES_MOD_SHA = "3c002aa3cbaa24f0ab61da9acac9709b0c1a427bbae294a6fca219e4cf82d785"
KBRES_JSON = os.path.join(D.OUT, "kbres_metaqa__v1.json")
KBRES_JSON_SHA = "282791b4ffc02c6fe306ab67f7db905702f49f2e61a499dfd88c60be0a1d741a"
KBRES_NPZ = os.path.join(D.OUT, "kbres_metaqa__v1.npz")
KBRES_NPZ_SHA = "6c34e20cf338bff01b40109e4b11599704a20b94498e23637c52138be0cefb7f"
ROUTED_ARMS = ("L1", "H2_LOCAL", "H2_PUSH", "H3_LOCAL", "H3_PUSH", "H3_PUSH_ROWS")
PUSH_ARMS = ("H2_PUSH", "H3_PUSH", "H3_PUSH_ROWS")
UNR_ARMS = ("L1", "H2", "H3")
QT_SHOW = (1000, 5000)
MODULES = ("_l1d_lib.py", "_l1d_arms.py", "_l1d_edgediag.py", "_l1d_route.py", "_l1d_kbres.py", "_l1d_node1h.py", "_l1d_adaptbp.py")


def prop(F, src, w, N):
    """one STRUCT application from the sources src (ascending) with weights w: (y, adjacency entries scanned).  The arithmetic of
    the section 38 probe (_l1d_kbres.KBResSpec.row), so the UNROUTED second hop reproduces it bit for bit."""
    U2, W2 = [], []
    ent = 0
    for fn in SFAMS:
        Fm = F[fn]
        st = Fm["xadj"][src]
        h2, p2 = E.entries(st, Fm["xadj"][src + 1] - st)
        U2.append(Fm["adj"][p2].astype(np.int64))
        W2.append((w[src] * Fm["g"][src])[h2])
        ent += len(p2)
    return np.bincount(np.concatenate(U2), weights=np.concatenate(W2), minlength=N), ent


def clist(mask, y, frank):
    """the candidate list of a mask: its nodes by (-y, FLAT rank)."""
    c = np.flatnonzero(mask)
    return c[np.lexsort((frank[c], -y[c]))]


def ranked(s, elig, frank):
    """the served order of an arm: the eligible nodes by (-s, FLAT rank)."""
    c = np.flatnonzero(elig)
    return c[np.lexsort((frank[c], -s[c]))]


def rrf_add(f, o):
    """f + [u in o] / (K0 + rank of u in o)."""
    x = f.copy()
    x[o] += 1.0 / (K0 + np.arange(len(o), dtype=np.float64))
    return x


def positions(o, g, N):
    """0-based position of each gold in the order o; N = not eligible (never served, N > max B_N)."""
    rk = np.full(N, N, np.int64)
    rk[o] = np.arange(len(o))
    return rk[g]


def c2rank(o, g, N):
    rk = np.full(N, -1, np.int64)
    rk[o] = np.arange(len(o))
    return rk[g]


class HopSpec(object):
    def __init__(self, cd, pop, parts):
        # the carried-forward L1: every pin of the KBRES diagnostic (carry-forward record, kscale v1 json/npz, served partitions,
        # families == kscale v1, query records) is asserted by its constructor
        self.K = KB.KBResSpec(cd, pop, parts)
        K = self.K
        self.N, self.F, self.pop = K.N, K.F, pop
        assert self.N > max(M_CURVE)
        nq, ng = pop.nq, pop.ng_tot
        self.cells = list(K.cells)
        self.cellsU = [UNR] + self.cells
        self.arms = {UNR: UNR_ARMS}
        for c in self.cells:
            self.arms[c] = ROUTED_ARMS
        self.POS = {(c, a): np.full(ng, -1, np.int64) for c in self.cellsU for a in self.arms[c]}
        self.CONT = {c: np.zeros(ng, bool) for c in self.cells}
        self.CMASS = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.CMASS[UNR] = np.full(nq, self.N, np.int64)
        # candidate ranks of every gold (-1: not in the list)
        self.GR = {k: np.full(ng, -1, np.int64) for k in (["UNR_C2", "UNR_C3"] + ["%s__%s" % (c, x) for c in self.cells
                                                                                   for x in ("C2L", "C2P", "C3L", "C3P", "C3R")])}
        # per query: traversal cost and list sizes
        ck = ["src2", "ent2", "src3", "ent3", "nC2", "nC3"]
        self.QC = {UNR: {k: np.zeros(nq, np.int64) for k in ck}}
        rk = ["src2", "ent2", "nC2L", "nC2P", "src3c", "ent3c", "nC3L", "nC3P", "src3r", "ent3r", "rows3_out", "shards3_out",
              "nC3R", "elig_H2_PUSH", "elig_H3_PUSH", "elig_H3_PUSH_ROWS"]
        for c in self.cells:
            self.QC[c] = {k: np.zeros(nq, np.int64) for k in rk}
        # per query and B_N: fetched nodes and distinct extra shards of the served list (PUSH arms)
        self.FETCH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in PUSH_ARMS}
        self.XSH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in PUSH_ARMS}
        self.LAT = {"base (IR_L1 score, served order, router order, V_q, LS)": [], "UNROUTED H2": [], "UNROUTED H3": []}
        for c in self.cells:
            for s in ("ES contacted set + L1 order", "H2 (y2, C2 lists, two orders)", "H3 (two third-hop propagations, three orders)"):
                self.LAT["%s %s" % (c, s)] = []
        self.record = {"l1_spec (the KBRES constructor's pins)": K.record, "cells": self.cells, "arms": {c: list(a) for c, a in self.arms.items()},
                       "M_curve": list(M_CURVE), "K0": K0, "ACT": ACT, "struct_families": list(SFAMS)}

    def row(self, j, of, frank, g, sl):
        K, N, F = self.K, self.N, self.F
        top = of[:ACT]
        nh = len(top)
        t0 = time.perf_counter()
        # ---- the carried-forward L1a (== _l1d_kbres.KBResSpec.row == _l1d_scale.ScaleSpec.row)
        U, H, GW = [], [], []
        for f_ in FAMS:
            Fm = F[f_]
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
        f = 1.0 / (K0 + frank.astype(np.float64))
        f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        FO = np.lexsort((frank, -f))
        vq = np.union1d(top, Lord)
        fq = np.zeros(len(vq))
        tq = np.zeros(len(vq))
        fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
        il = np.searchsorted(vq, Lord)
        fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        tq[il] = ACT + np.arange(len(Lord))
        tq[np.searchsorted(vq, top)] = np.arange(nh)
        oq = vq[np.lexsort((tq, -fq))]
        ev = np.zeros(N, bool)
        ev[vq] = True
        nS = len(U[0]) + len(U[1])
        LS = np.bincount(u[:nS], weights=x[:nS], minlength=N)
        ms = np.flatnonzero(LS)
        self.LAT["base (IR_L1 score, served order, router order, V_q, LS)"].append(time.perf_counter() - t0)
        # ---- UNROUTED: L1, H2, H3
        self.POS[(UNR, "L1")][sl] = positions(FO, g, N)
        t0 = time.perf_counter()
        y2, e2 = prop(F, ms, LS, N)
        m2 = (y2 > 0) & ~ev
        o2 = clist(m2, y2, frank)
        f2 = rrf_add(f, o2)
        self.POS[(UNR, "H2")][sl] = positions(ranked(f2, np.ones(N, bool), frank), g, N)
        self.LAT["UNROUTED H2"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        s3 = np.flatnonzero(y2)
        y3, e3 = prop(F, s3, y2, N)
        m3 = (y3 > 0) & ~ev & ~m2
        o3 = clist(m3, y3, frank)
        f3 = rrf_add(f2, o3)
        self.POS[(UNR, "H3")][sl] = positions(ranked(f3, np.ones(N, bool), frank), g, N)
        self.LAT["UNROUTED H3"].append(time.perf_counter() - t0)
        self.GR["UNR_C2"][sl] = c2rank(o2, g, N)
        self.GR["UNR_C3"][sl] = c2rank(o3, g, N)
        q = self.QC[UNR]
        q["src2"][j], q["ent2"][j], q["src3"][j], q["ent3"][j], q["nC2"][j], q["nC3"][j] = len(ms), e2, len(s3), e3, len(o2), len(o3)
        # ---- routed cells
        for c in self.cells:
            t0 = time.perf_counter()
            P = K.parts[c]
            npart, hard = P.npart, P.hard
            hq = hard[oq]
            up, first = np.unique(hq, return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            ro = np.lexsort((np.arange(npart), fe))
            pr = RT.inv(ro)
            b = int(K.B[c][j])
            assert 1 <= b <= npart
            assert int(pr[hard[g]].max()) + 1 == K.LOK[c][j], "recomputed ES LO differs (%s row %d)" % (c, j)
            contn = pr[hard] < b
            cm = int(contn.sum())
            self.CMASS[c][j] = cm
            self.CONT[c][sl] = contn[g]
            o1 = FO[contn[FO]]
            self.POS[(c, "L1")][sl] = positions(o1, g, N)
            nmax = min(max(M_CURVE), cm)
            self.LAT["%s ES contacted set + L1 order" % c].append(time.perf_counter() - t0)
            # hop 2: sources = the LS support inside the contacted shards
            t0 = time.perf_counter()
            src2 = ms[contn[ms]]
            y2c, e2c = prop(F, src2, LS, N)
            m2P = (y2c > 0) & ~ev
            m2L = m2P & contn
            o2P = clist(m2P, y2c, frank)
            o2L = clist(m2L, y2c, frank)
            f2L = rrf_add(f, o2L)
            f2P = rrf_add(f, o2P)
            oH2L = ranked(f2L, contn, frank)
            elP2 = contn | m2P
            oH2P = ranked(f2P, elP2, frank)
            self.LAT["%s H2 (y2, C2 lists, two orders)" % c].append(time.perf_counter() - t0)
            # hop 3
            t0 = time.perf_counter()
            s3c = np.flatnonzero((y2c > 0) & contn)
            y3c, e3c = prop(F, s3c, y2c, N)
            s3r = np.flatnonzero(y2c)
            y3r, e3r = prop(F, s3r, y2c, N)
            m3L = (y3c > 0) & ~ev & ~m2L & contn
            m3P = (y3c > 0) & ~ev & ~m2P
            m3R = (y3r > 0) & ~ev & ~m2P
            o3L = clist(m3L, y3c, frank)
            o3P = clist(m3P, y3c, frank)
            o3R = clist(m3R, y3r, frank)
            oH3L = ranked(rrf_add(f2L, o3L), contn, frank)
            elP3 = elP2 | m3P
            oH3P = ranked(rrf_add(f2P, o3P), elP3, frank)
            elR3 = elP2 | m3R
            oH3R = ranked(rrf_add(f2P, o3R), elR3, frank)
            self.LAT["%s H3 (two third-hop propagations, three orders)" % c].append(time.perf_counter() - t0)
            orders = {"H2_LOCAL": oH2L, "H2_PUSH": oH2P, "H3_LOCAL": oH3L, "H3_PUSH": oH3P, "H3_PUSH_ROWS": oH3R}
            for a, o in orders.items():
                self.POS[(c, a)][sl] = positions(o, g, N)
                assert len(o) >= cm
            for a in ("H2_LOCAL", "H3_LOCAL"):
                assert contn[orders[a]].all() and len(orders[a]) == cm
            for a in PUSH_ARMS:
                o = orders[a][:nmax]
                out = ~contn[o]
                for mi, M in enumerate(M_CURVE):
                    n_ = min(M, cm)
                    ob = out[:n_]
                    self.FETCH[(c, a)][j, mi] = int(ob.sum())
                    self.XSH[(c, a)][j, mi] = int(len(np.unique(hard[o[:n_][ob]])))
            for k_, o in (("C2L", o2L), ("C2P", o2P), ("C3L", o3L), ("C3P", o3P), ("C3R", o3R)):
                self.GR["%s__%s" % (c, k_)][sl] = c2rank(o, g, N)
            q = self.QC[c]
            rout = s3r[~contn[s3r]]
            vals = {"src2": len(src2), "ent2": e2c, "nC2L": len(o2L), "nC2P": len(o2P), "src3c": len(s3c), "ent3c": e3c,
                    "nC3L": len(o3L), "nC3P": len(o3P), "src3r": len(s3r), "ent3r": e3r, "rows3_out": len(rout),
                    "shards3_out": len(np.unique(hard[rout])), "nC3R": len(o3R), "elig_H2_PUSH": int(elP2.sum()),
                    "elig_H3_PUSH": int(elP3.sum()), "elig_H3_PUSH_ROWS": int(elR3.sum())}
            for k_, v in vals.items():
                q[k_][j] = v

    # ---- tables
    def served(self, c, a, M):
        pop = self.pop
        cap = np.minimum(M, self.CMASS[c])[pop.row_of_gold]
        sv = self.POS[(c, a)] < cap
        cnt = np.add.reduceat(sv.astype(np.int64), pop.gptr[:-1])
        return cnt == pop.ngold, cnt > 0, cnt / pop.ngold.astype(np.float64), sv

    def finish(self):
        pop = self.pop
        nq, ng, hops, ngold = pop.nq, pop.ng_tot, pop.hops, pop.ngold
        rog = pop.row_of_gold
        QS = {"all": np.ones(nq, bool)}
        HS = sorted(set(int(h) for h in hops))
        for h in HS:
            QS["hop%d" % h] = hops == h
        for nm, lo, hi in D.NG_BUCKETS:
            QS["ng_" + nm] = (ngold >= lo) & (ngold <= hi)
        for h in HS:
            for nm, lo, hi in D.NG_BUCKETS:
                m = (hops == h) & (ngold >= lo) & (ngold <= hi)
                if m.any():
                    QS["hop%d|ng_%s" % (h, nm)] = m
        qts = sorted(set(self.K.QTYPE))
        QT = {qt: np.array([x == qt for x in self.K.QTYPE]) for qt in qts}
        res = {}
        for c in self.cellsU:
            res[c] = {}
            ref = {M: self.served(c, "L1", M) for M in M_CURVE}
            for a in self.arms[c]:
                res[c][a] = {}
                for mi, M in enumerate(M_CURVE):
                    all_, any_, frac, sv = self.served(c, a, M)
                    nserv = np.minimum(M, self.CMASS[c])
                    e = {"ALL": int(all_.sum()), "ANY": int(any_.sum()), "FRAC_mean": D.q4(frac.mean()), "gold_nodes_served": int(sv.sum()),
                         "served_nodes_mean": round(float(nserv.mean()), 2),
                         "strata (ALL rows, rows)": {k: [int(all_[m].sum()), int(m.sum())] for k, m in QS.items()},
                         "strata (gold nodes served, gold nodes)": {k: [int(sv[m[rog]].sum()), int(m[rog].sum())] for k, m in QS.items()
                                                                    if k in ["all"] + ["hop%d" % h for h in HS]}}
                    if a != "L1":
                        r_all = ref[M][0]
                        e["paired_vs_L1 (gained = arm serves ALL gold, L1 does not)"] = dict(
                            D.paired(r_all, all_), **{"strata": {k: D.paired(r_all, all_, m) for k, m in QS.items() if k != "all"}})
                        e["gold_nodes gained / lost vs L1"] = [int((sv & ~ref[M][3]).sum()), int((~sv & ref[M][3]).sum())]
                    if (c, a) in self.FETCH:
                        fe, xs = self.FETCH[(c, a)][:, mi], self.XSH[(c, a)][:, mi]
                        e["fetched_nodes_per_query (served outside the contacted shards)"] = {
                            "mean": round(float(fe.mean()), 2), "median": float(np.median(fe)), "p95": float(np.percentile(fe, 95)),
                            "max": int(fe.max()), "queries_with_any": int((fe > 0).sum()), "share_of_served": D.q4(fe.sum() / float(nserv.sum()))}
                        e["extra_shards_per_query (distinct uncontacted shards of the fetched nodes)"] = {
                            "mean": round(float(xs.mean()), 2), "median": float(np.median(xs)), "p95": float(np.percentile(xs, 95)),
                            "max": int(xs.max())}
                    if M in QT_SHOW:
                        e["per_qtype (ALL rows, rows)"] = {qt: [int(all_[m].sum()), int(m.sum())] for qt, m in QT.items()}
                    res[c][a][str(M)] = e
        cost = {}
        for c in self.cellsU:
            cost[c] = {k: KB.AB.dist(v) for k, v in self.QC[c].items()}
            if c != UNR:
                cost[c]["contacted_mass"] = KB.AB.dist(self.CMASS[c])
                cost[c]["B_P"] = KB.AB.dist(self.K.B[c])
        cand = {}
        for k, r in self.GR.items():
            cand[k] = {"golds_in_list": int((r >= 0).sum()), **{"rank < %d" % t: int(((r >= 0) & (r < t)).sum()) for t in (100, 500, 1000, 5000)}}
        lat = {k: D.ms_stats(v) for k, v in self.LAT.items()}
        arrays = {"HOPS": hops, "QTYPE": np.array(self.K.QTYPE), "m_curve": np.asarray(M_CURVE, np.int64), "cells": np.array(self.cellsU)}
        for (c, a), v in self.POS.items():
            arrays["POS__%s__%s" % (c, a)] = v
        for c in self.cells:
            arrays["CONT__" + c] = self.CONT[c]
            arrays["CMASS__" + c] = self.CMASS[c]
            arrays["B__" + c] = self.K.B[c]
            for k, v in self.QC[c].items():
                arrays["QC__%s__%s" % (c, k)] = v
            for a in PUSH_ARMS:
                arrays["FETCH__%s__%s" % (c, a)] = self.FETCH[(c, a)]
                arrays["XSH__%s__%s" % (c, a)] = self.XSH[(c, a)]
        for k, v in self.QC[UNR].items():
            arrays["QC__%s__%s" % (UNR, k)] = v
        for k, v in self.GR.items():
            arrays["GR__" + k] = v
        return {"results (per cell, arm, B_N)": res, "traversal_cost_per_query": cost, "candidate_ranks_of_gold_nodes": cand,
                "latency_ms_per_row": lat}, arrays


def kbres_identity(S, POS_FLAT, full):
    """the L1 arms and the UNROUTED second hop against the section 38 records (write-once, sha pinned)."""
    assert D.sha_file(KBRES_NPZ) == KBRES_NPZ_SHA and D.sha_file(KBRES_JSON) == KBRES_JSON_SHA
    z = np.load(KBRES_NPZ)
    pop = S.pop
    nq, ng = pop.nq, pop.ng_tot
    assert (z["gptr"][:nq + 1] == pop.gptr).all() and (z["pos_FLAT"][:ng] == POS_FLAT).all()
    assert (z["G_FRK"][:ng] == S.POS[(UNR, "L1")]).all(), "UNROUTED L1 differs from kbres v1"
    assert (z["G_R2"][:ng] == S.GR["UNR_C2"]).all(), "UNROUTED C2 ranks differ from the kbres v1 probe"
    assert (z["Q_C2"][:nq] == S.QC[UNR]["nC2"]).all()
    for c in S.cells:
        cont, rr = z["cont__" + c][:ng], z["rr__" + c][:ng]
        assert (cont == S.CONT[c]).all(), "contacted flags differ from kbres v1 (%s)" % c
        assert (np.where(cont, rr, S.N) == S.POS[(c, "L1")]).all(), "routed L1 positions differ from kbres v1 (%s)" % c
        assert (z["Q_CMASS__" + c][:nq] == S.CMASS[c]).all()
    out = "on all %d gold nodes / %d rows: UNROUTED L1 position == G_FRK, UNROUTED C2 rank == G_R2, routed contacted flags and L1 " \
          "restricted ranks == cont__ / rr__ (%s), contacted mass == Q_CMASS__, |C2| == Q_C2 of %s (sha %s)" % (
              ng, nq, ", ".join(S.cells), D.rel(KBRES_NPZ), KBRES_NPZ_SHA[:16])
    if full:
        T = json.load(open(KBRES_JSON, encoding="utf-8"))["diagnostics"]["tables (per cell, B_N, stratum: query level and gold level)"]
        for c in S.cellsU:
            for M in M_CURVE:
                all_ = S.served(c, "L1", M)[0]
                assert int(all_.sum()) == T[c][str(M)]["all"]["queries"]["ALL"], "L1 ALL differs from kbres v1 (%s, %d)" % (c, M)
        out += "; the L1 ALL counts of every cell and B_N == the kbres v1 tables (section 37.3 / 38)"
    return out


def run(ds, tag):
    ROWS, SMOKE_OUT = R.argv_opts()
    here = os.path.abspath(__file__)
    shas = {"harness": D.sha_file(here)}
    for m in MODULES:
        shas[m] = D.sha_file(os.path.join(D.HERE, m))
    assert shas["_l1d_kbres.py"] == KBRES_MOD_SHA, "the KBRES module changed"
    host0 = D.host_state()
    t_all = time.time()
    outdir = SMOKE_OUT or OUT3
    os.makedirs(outdir, exist_ok=True)
    fp_out = os.path.join(outdir, "l3hop_%s__%s.json" % (ds, tag))
    fz = fp_out.replace(".json", ".npz")
    assert not os.path.exists(fp_out) and not os.path.exists(fz), "write-once: %s exists" % fp_out
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    cells = D.CELLS[ds]
    pop = D.Population(cd, ROWS)
    nq, gptr, ngold = pop.nq, pop.gptr, pop.ngold
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in cells}
    t_ = time.time()
    S = HopSpec(cd, pop, parts)
    S.record["seconds_build"] = round(time.time() - t_, 1)
    log("RUN l3hop %s %s: N %d, %d rows, %d gold nodes, cells %s" % (ds, tag, N, nq, pop.ng_tot, S.cellsU))
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    LAT = {"products_amortized": [], "flat_rrf": []}
    agd, ags = np.zeros(nq), np.zeros(nq)
    d200 = np.asarray(cd.dense_topk(D.ACT, pop.rows), np.int64)
    s200 = np.asarray(cd.splade_topk(D.ACT, pop.rows), np.int64)
    Qu = D.unit_queries(cd, pop.rows)
    t_ = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        LAT["products_amortized"] += [sec / float(j1 - j0)] * (j1 - j0)
        for i in range(j1 - j0):
            j = j0 + i
            t0 = time.perf_counter()
            of, od, os_, npos_j, fv, frank = D.flat_row(SD[i], SS[i])
            LAT["flat_rrf"].append(time.perf_counter() - t0)
            k_ = min(D.N_AGREE, npos_j)
            agd[j] = len(set(od[:D.N_AGREE].tolist()) & set(d200[j, :D.N_AGREE].tolist())) / float(D.N_AGREE)
            ags[j] = (len(set(os_[:k_].tolist()) & set(s200[j, :k_].tolist())) / float(k_)) if k_ else 1.0
            g = pop.golds[j]
            sl = slice(gptr[j], gptr[j + 1])
            POS_FLAT[sl] = frank[g]
            S.row(j, of, frank, g, sl)
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nq, time.time() - t_, D._rss_mb()))
    t_loop = round(time.time() - t_, 1)
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    f1 = os.path.join(D.OUT, "loc_%s__%s.npz" % (ds, R.V1_TAG))
    z1 = np.load(f1)
    assert (z1["rows"][:nq] == pop.rows).all()
    ng = int(z1["gptr"][nq])
    assert ng == pop.ng_tot and (z1["pos_FLAT"][:ng] == POS_FLAT).all(), "FLAT positions differ from the v1 record"
    v1check = "FLAT positions == %s (sha %s) on all %d gold nodes" % (D.rel(f1), D.sha_file(f1)[:16], ng)
    log("v1 check: " + v1check)
    idcheck = kbres_identity(S, POS_FLAT, ROWS is None)
    log("kbres check: " + idcheck)
    diag, arrays = S.finish()
    for c in S.cellsU:
        for a in S.arms[c]:
            log("%-9s %-13s ALL %s" % (c, a, " ".join("%5d" % diag["results (per cell, arm, B_N)"][c][a][str(M)]["ALL"] for M in M_CURVE)))
    res = {"dataset": ds, "tag": tag, "stage": "L3 development (HOP)", "status": "DEVELOPMENT (descriptive; p-values descriptive; no rule selected)",
           "definitions": __doc__, "N": N, "n_rows": nq, "n_gold_nodes": pop.ng_tot, "population": pop.record, "structures": S.record,
           "v1_check": v1check, "kbres_identity": idcheck, "served_list_agreement": agree, "diagnostics": diag,
           "latency_ms_flat": {"products_amortized": D.ms_stats(LAT["products_amortized"]), "flat_rrf": D.ms_stats(LAT["flat_rrf"])},
           "code": {k: {"path": "scratchpad/" + (os.path.basename(here) if k == "harness" else k), "sha256": v} for k, v in shas.items()},
           "inputs": {"kbres_v1_json": {"path": D.rel(KBRES_JSON), "sha256": KBRES_JSON_SHA},
                      "kbres_v1_npz": {"path": D.rel(KBRES_NPZ), "sha256": KBRES_NPZ_SHA}},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "constants": D.CONSTANTS, "platform": D.platform_record(),
           "host_at_start": host0, "seconds_loop": t_loop, "_row_query_ids": pop.qids}
    arrays.update({"rows": pop.rows, "gptr": gptr, "pos_FLAT": POS_FLAT})
    res["seconds"] = round(time.time() - t_all, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    for m in MODULES:
        assert D.sha_file(os.path.join(D.HERE, m)) == shas[m], "code changed during the run"
    assert D.sha_file(here) == shas["harness"], "code changed during the run"
    np.savez_compressed(fz, **arrays)
    res["npz"] = {"path": os.path.basename(fz), "sha256": D.sha_file(fz)}
    D.G.S.wj(fp_out, res)
    log("done (%.0fs, peak RSS %s MB) -> %s sha256 %s" % (res["seconds"], res["process_peak_rss_mb"], fp_out, D.sha_file(fp_out)[:12]))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN metaqa <tag> [--rows=K --out=<dir outside the repository>]"
    ds, tag = sys.argv[2], sys.argv[3]
    assert ds in ("metaqa",), "HOP is defined on the KB development dataset with a served structural pair and a KBRES record"
    run(ds, tag)


if __name__ == "__main__":
    main()
